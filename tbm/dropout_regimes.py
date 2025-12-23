# tbm/dropout_regimes.py
from __future__ import annotations
import csv
from typing import Dict, Set, List, Tuple, Optional
import torch
import numpy as np
from .specs import FeatureSpec

def load_vlm_selection_csv(csv_path: str,
                           image_col: str = "image_path",
                           tool_ids_col: str = "tool_ids") -> Dict[str, Set[str]]:
    """
    CSV with at least columns:
      - image_path (or image_id): identifies the sample
      - tool_ids: comma-separated tool names (e.g., 'nuc_type,box,contour')
    Returns: stem -> set(tool_name)
    """
    def _stem(p: str) -> str:
        base = p.split("/")[-1]
        if "__" in base: base = base.split("__", 1)[-1]
        return base.rsplit(".", 1)[0]

    mp: Dict[str, Set[str]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = _stem(row.get(image_col, row.get("image_id", "")))
            ids = [s.strip() for s in str(row.get(tool_ids_col, "")).split(",") if s.strip()]
            mp[stem] = set(ids)
    return mp

# ---------------------- regimes (build masks) ----------------------

def mask_all(B: int, spec: FeatureSpec, device) -> torch.Tensor:
    return torch.ones((B, len(spec.tools)), device=device)

def mask_random(B: int, spec: FeatureSpec, device, p_keep: float = 0.5) -> torch.Tensor:
    probs = torch.full((B, len(spec.tools)), float(p_keep), device=device)
    return torch.bernoulli(probs)

def mask_random_k(B: int, spec: FeatureSpec, device, k: int) -> torch.Tensor:
    T = len(spec.tools); k = max(0, min(k, T))
    keep = torch.zeros((B, T), device=device)
    if k == 0: return keep
    for b in range(B):
        idx = torch.randperm(T, device=device)[:k]
        keep[b, idx] = 1.0
    return keep

def _index_map(tool_names: List[str]) -> Dict[str, int]:
    return {t: i for i, t in enumerate(tool_names)}

def mask_exact(stems: List[str], selected: Dict[str, Set[str]], spec: FeatureSpec, device) -> torch.Tensor:
    T = len(spec.tools); name2i = _index_map(spec.tool_names)
    keep = torch.zeros((len(stems), T), device=device)
    for b, s in enumerate(stems):
        for t in selected.get(s, set()):
            if t in name2i:
                keep[b, name2i[t]] = 1.0
    return keep

def mask_informed(stems: List[str], selected: Dict[str, Set[str]],
                  spec: FeatureSpec, device,
                  alpha: float = 1.0, base_p: float = 0.5,
                  deterministic: bool = False) -> torch.Tensor:
    """
    p_i = (1-alpha)*base_p + alpha*s_i, where s_i ∈ {0,1} indicates VLM selection.
    """
    T = len(spec.tools); name2i = _index_map(spec.tool_names)
    probs = torch.full((len(stems), T), float(base_p), device=device)
    for b, s in enumerate(stems):
        sel = selected.get(s, set())
        for t in sel:
            if t in name2i:
                probs[b, name2i[t]] = (1.0 - alpha) * base_p + alpha * 1.0
    if deterministic:
        return (probs >= 0.5).float()
    return torch.bernoulli(probs)

def global_prior_from_selected(selected: Dict[str, Set[str]],
                               tool_names: List[str],
                               eps: float = 0.05) -> np.ndarray:
    name2i = _index_map(tool_names)
    T = len(tool_names)
    counts = np.zeros(T, dtype=np.float32)
    N = max(1, len(selected))
    for sel in selected.values():
        for t in set(sel):
            if t in name2i:
                counts[name2i[t]] += 1.0
    pi = counts / float(N)
    return np.clip(pi, eps, 1.0 - eps)

def mask_global(B: int, pi: np.ndarray, device) -> torch.Tensor:
    probs = torch.tensor(pi, device=device, dtype=torch.float32).expand(B, -1)
    return torch.bernoulli(probs)
