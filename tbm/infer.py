# infer.py
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from tbm.data import FeatureTensorDataset
from tbm.model import ToolDropoutSharedEffB0
from tbm.specs import FeatureSpec
from tbm.dropout_regimes import (
    load_vlm_selection_csv,
    mask_all, mask_random, mask_random_k, mask_exact, mask_informed,
    global_prior_from_selected, mask_global,
)

def _make_mask(regime, B, stems, spec, device, *, p_keep, top_k, selected_map, global_pi, alpha, base_p, deterministic):
    if regime == "all":      return mask_all(B, spec, device)
    if regime == "random":   return mask_random(B, spec, device, p_keep=p_keep)
    if regime == "random_k": return mask_random_k(B, spec, device, k=top_k)
    if regime == "global":   return mask_global(B, global_pi, device)
    if regime == "informed": return mask_informed(stems, selected_map, spec, device, alpha=alpha, base_p=base_p, deterministic=deterministic)
    if regime == "exact":    return mask_exact(stems, selected_map, spec, device)
    raise ValueError(f"Unknown regime: {regime}")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", default="outputs/features")
    ap.add_argument("--split", default="test")
    ap.add_argument("--label_csv", default=None)
    ap.add_argument("--tool_slices_json", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--regime", choices=["all","random","random_k","global","informed","exact"], default="exact")
    ap.add_argument("--p_keep", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--base_p", type=float, default=0.5)
    ap.add_argument("--vlm_csv", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # dataset + loader
    ds = FeatureTensorDataset(args.features_root, split=args.split, label_csv=args.label_csv)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)

    # spec
    if args.tool_slices_json:
        spec = FeatureSpec.from_json(args.tool_slices_json)
    else:
        # default HoverNet-like layout
        spec = FeatureSpec.from_slices({"nuc_type": (0,6), "box": (6,7), "centroid": (7,8), "contour": (8,9)})

    # model
    num_classes = len({y for _, y, _ in ds})
    model = ToolDropoutSharedEffB0(in_channels=ds.C, num_classes=num_classes, pretrained=False)
    model.set_tool_slices(spec.tool_slices)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device).eval()

    # selection maps
    selected_map = load_vlm_selection_csv(args.vlm_csv) if args.vlm_csv else {}
    global_pi = global_prior_from_selected(selected_map, spec.tool_names) if args.regime == "global" else None

    # run
    correct = total = 0
    for x, y, stems in ld:
        x, y = x.to(device), y.to(device)
        keep = _make_mask(
            args.regime, B=x.size(0), stems=list(stems), spec=spec, device=device,
            p_keep=args.p_keep, top_k=args.top_k,
            selected_map=selected_map, global_pi=global_pi,
            alpha=args.alpha, base_p=args.base_p, deterministic=True
        )
        logits = model(x, keep_mask=keep)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = correct / max(1, total)
    print(f"[infer] split={args.split} regime={args.regime} acc={acc:.4f}")

if __name__ == "__main__":
    main()
