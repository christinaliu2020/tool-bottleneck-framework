# tbm/train.py
import os, json, argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import FeatureTensorDataset
from .model import ToolDropoutSharedEffB0
from .specs import FeatureSpec  # must provide .tool_names and .tools

# === Regimes imported from dropout_regimes.py ===
from .dropout_regimes import (
    load_vlm_selection_csv,
    mask_all,
    mask_random,
    mask_random_k,
    mask_exact,
    mask_informed,
    global_prior_from_selected,
    mask_global,
)

# ----------------------------
# Helpers
# ----------------------------

def _load_tool_slices(tool_slices_json: Optional[str]) -> Dict[str, Tuple[int, int]]:
    """
    Return dict tool_name -> (start, end), end-exclusive, matching channel ranges.
    """
    if tool_slices_json:
        with open(tool_slices_json, "r") as f:
            m = json.load(f)
        # normalize to tuples
        return {k: (int(v[0]), int(v[1])) for k, v in m.items()}
    # Default matches HoverNetRasterizer(order: nuc_type [6ch], box [1], centroid [1], contour [1])
    return {"nuc_type": (0, 6), "box": (6, 7), "centroid": (7, 8), "contour": (8, 9)}

def _build_featurespec(tool_slices: Dict[str, Tuple[int,int]]) -> FeatureSpec:
    """
    Construct a FeatureSpec from a name->(start,end) dict.
    Supports FeatureSpec.from_slices(...) if present, otherwise FeatureSpec(tool_slices).
    """
    if hasattr(FeatureSpec, "from_slices"):
        return FeatureSpec.from_slices(tool_slices)  # type: ignore[attr-defined]
    return FeatureSpec(tool_slices)  # fall back: your FeatureSpec __init__ should accept dict

def _make_mask(
    regime: str,
    B: int,
    stems: List[str],
    spec: FeatureSpec,
    device,
    *,
    p_keep: float,
    top_k: int,
    selected_map: Dict[str, set],
    global_pi: Optional[np.ndarray],
    alpha: float,
    base_p: float,
    deterministic: bool,
):
    if regime == "all":
        return mask_all(B, spec, device)
    if regime == "random":
        return mask_random(B, spec, device, p_keep=p_keep)
    if regime == "random_k":
        return mask_random_k(B, spec, device, k=top_k)
    if regime == "global":
        if global_pi is None:
            raise ValueError("global regime requires global_pi")
        return mask_global(B, global_pi, device)
    if regime == "informed":
        return mask_informed(stems, selected_map, spec, device, alpha=alpha, base_p=base_p, deterministic=deterministic)
    if regime == "exact":
        return mask_exact(stems, selected_map, spec, device)
    return None

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    *,
    eval_regime: str,
    spec: FeatureSpec,
    selected_map: Dict[str, set],
    global_pi: Optional[np.ndarray],
    alpha: float,
    base_p: float,
):
    model.eval()
    correct = total = 0
    for x, y, stems in loader:
        x, y = x.to(device), y.to(device)
        keep = _make_mask(
            eval_regime, B=x.size(0), stems=list(stems), spec=spec, device=device,
            p_keep=1.0, top_k=len(spec.tools),  # unused in eval unless regime uses them
            selected_map=selected_map, global_pi=global_pi,
            alpha=alpha, base_p=base_p, deterministic=True,
        )
        logits = model(x, keep_mask=keep)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=str, default="outputs/features")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="id_val")
    ap.add_argument("--ood_split", type=str, default="val")
    ap.add_argument("--test_split", type=str, default="test")
    ap.add_argument("--label_csv", type=str, default=None,
                    help="Optional CSV with columns: stem,label[,split] (enables flat split dirs)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")

    # tool slice / spec config
    ap.add_argument("--tool_slices_json", type=str, default=None,
                    help='JSON mapping tool_name -> [start,end] (end-exclusive).')

    # regime config (now routed through dropout_regimes.py)
    ap.add_argument("--regime", choices=["all","random","random_k","global","informed","exact"], default="informed")
    ap.add_argument("--p_keep", type=float, default=0.5, help="for random")
    ap.add_argument("--top_k", type=int, default=3, help="for random_k")
    ap.add_argument("--alpha", type=float, default=1.0, help="for informed p-blend")
    ap.add_argument("--base_p", type=float, default=0.5, help="baseline keep prob in informed/random")
    ap.add_argument("--vlm_csv", type=str, default=None,
                    help="CSV with columns: image_path,tool_ids (used by informed/exact/global)")

    ap.add_argument("--ckpt", type=str, default="checkpoints/tbm_best.pth")
    args = ap.parse_args()

    # device
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)

    # datasets
    ds_train = FeatureTensorDataset(args.features_root, split=args.train_split, label_csv=args.label_csv)
    ds_val   = FeatureTensorDataset(args.features_root, split=args.val_split,   label_csv=args.label_csv)
    ds_ood   = FeatureTensorDataset(args.features_root, split=args.ood_split,   label_csv=args.label_csv)
    ds_test  = FeatureTensorDataset(args.features_root, split=args.test_split,  label_csv=args.label_csv)

    ld_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,  num_workers=8, pin_memory=True)
    ld_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)
    ld_ood   = DataLoader(ds_ood,   batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)
    ld_test  = DataLoader(ds_test,  batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)

    # tool spec
    tool_slices = _load_tool_slices(args.tool_slices_json)
    spec = _build_featurespec(tool_slices)

    # model
    num_classes = len({y for _, y, _ in ds_train})
    model = ToolDropoutSharedEffB0(in_channels=ds_train.C, num_classes=num_classes, pretrained=args.pretrained).to(device)
    # let the backbone know channel layout for block-wise masking
    model.set_tool_slices(tool_slices)

    # informed/exact/global selection map (optional)
    selected_map = load_vlm_selection_csv(args.vlm_csv) if args.vlm_csv else {}
    global_pi = global_prior_from_selected(selected_map, spec.tool_names) if args.regime == "global" else None

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, stems in ld_train:
            x, y = x.to(device), y.to(device)
            keep = _make_mask(
                args.regime,
                B=x.size(0),
                stems=list(stems),
                spec=spec,
                device=device,
                p_keep=args.p_keep,
                top_k=args.top_k,
                selected_map=selected_map,
                global_pi=global_pi,
                alpha=args.alpha,
                base_p=args.base_p,
                deterministic=False,
            )
            logits = model(x, keep_mask=keep)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(ld_train))

        # Validation: if we have per-image selections, use exact (deterministic) at eval;
        # else use "all" (no dropout) for a stable checkpointing signal.
        eval_regime = "exact" if selected_map else "all"
        val_acc = evaluate(
            model, ld_val, device,
            eval_regime=eval_regime,
            spec=spec,
            selected_map=selected_map,
            global_pi=global_pi,
            alpha=args.alpha,
            base_p=args.base_p,
        )
        print(f"[ep {ep}/{args.epochs}] loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.ckpt)
            print(f"  ✓ saved best to {args.ckpt}")

    # Final eval (OOD + Test)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device).eval()

    eval_regime = "exact" if selected_map else "all"
    ood_acc = evaluate(
        model, ld_ood, device,
        eval_regime=eval_regime,
        spec=spec,
        selected_map=selected_map,
        global_pi=global_pi,
        alpha=args.alpha,
        base_p=args.base_p,
    )
    test_acc = evaluate(
        model, ld_test, device,
        eval_regime=eval_regime,
        spec=spec,
        selected_map=selected_map,
        global_pi=global_pi,
        alpha=args.alpha,
        base_p=args.base_p,
    )
    print(f"[FINAL] best_val={best_val:.4f} | ood={ood_acc:.4f} | test={test_acc:.4f}")

if __name__ == "__main__":
    main()
