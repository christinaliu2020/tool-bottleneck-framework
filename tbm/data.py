# tbm/data.py
import os, re, glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset

STEM_RE = re.compile(r"^(.*?)(?:_features)?\.pt$")

def _stem_from_path(p: Path) -> str:
    s = p.stem
    m = STEM_RE.match(p.name)
    return (m.group(1) if m else s)

class FeatureTensorDataset(Dataset):
    """
    Loads [C,H,W] float32 tensors saved as *.pt along with labels.

    Two supported labeling modes (pick one):
    (A) Directory-mode (recommended):
        features_root/<split>/<class_name>/*_features.pt
        label = class_name
    (B) CSV-mode:
        Pass label_csv with columns: stem,label[,split]
        - stem must match the PT filename stem (without suffix)
        - if split column exists, you can point the Dataset at a split

    Args:
        features_root: path to 'outputs/features'
        split: one of {"train","id_val","val","test"} or any folder name
        label_csv: optional CSV with columns stem,label[,split]
        class_map: optional dict class_name -> int label id
    """
    def __init__(
        self,
        features_root: str,
        split: str,
        label_csv: Optional[str] = None,
        class_map: Optional[Dict[str,int]] = None,
    ):
        import pandas as pd
        self.features_root = Path(features_root)
        self.split = split
        self.samples: List[Tuple[Path, int, str]] = []  # (pt_path, y, stem)

        if label_csv is None:
            # (A) directory mode
            split_dir = self.features_root / split
            classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
            if class_map is None:
                class_map = {c:i for i,c in enumerate(classes)}
            self.class_map = class_map
            for c in classes:
                for p in (split_dir / c).glob("*.pt"):
                    stem = _stem_from_path(p)
                    self.samples.append((p, self.class_map[c], stem))
        else:
            # (B) CSV mode
            df = pd.read_csv(label_csv)
            if "stem" not in df.columns or "label" not in df.columns:
                raise ValueError("label_csv must contain columns: stem,label[,split]")
            if "split" in df.columns:
                df = df[df["split"] == split]
            if class_map is None:
                classes = sorted(df["label"].unique().tolist())
                class_map = {c:i for i,c in enumerate(classes)}
            self.class_map = class_map

            split_dir = self.features_root / split
            for p in split_dir.glob("*.pt"):
                stem = _stem_from_path(p)
                row = df[df["stem"] == stem]
                if len(row) == 0: 
                    continue
                cls = str(row.iloc[0]["label"])
                y = self.class_map[cls]
                self.samples.append((p, y, stem))

        if len(self.samples) == 0:
            raise RuntimeError(f"No feature tensors found for split='{split}' in {self.features_root}")

        # infer C from one example
        ex = torch.load(self.samples[0][0], map_location="cpu")
        if ex.ndim != 3:
            raise ValueError(f"Expected [C,H,W] tensor, got shape {tuple(ex.shape)}")
        self.C = ex.shape[0]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, y, stem = self.samples[idx]
        x = torch.load(p, map_location="cpu")  # [C,H,W]
        return x, y, stem
