# tbm/rasterizers.py

from abc import ABC, abstractmethod
import torch
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.draw import polygon
from skimage.morphology import binary_erosion

def rasterize_boxes(instances: List[Dict], H: int, W: int, fill=True) -> torch.Tensor:
    """Convert bounding boxes to masks."""
    mask = torch.zeros((H, W), dtype=torch.float32)
    for inst in instances:
        box = inst.get("box")
        if box is None: continue
        x1, y1, x2, y2 = map(float, box)
        x1, y1 = int(np.clip(np.floor(x1), 0, W-1)), int(np.clip(np.floor(y1), 0, H-1))
        x2, y2 = int(np.clip(np.ceil(x2), 0, W)), int(np.clip(np.ceil(y2), 0, H))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    return mask


def rasterize_centroids(instances: List[Dict], H: int, W: int) -> torch.Tensor:
    """Convert centroids to point mask."""
    mask = torch.zeros((H, W), dtype=torch.float32)
    for inst in instances:
        c = inst.get("centroid")
        if c is None: continue
        x, y = map(int, map(round, c))
        if 0 <= y < H and 0 <= x < W:
            mask[y, x] = 1.0
    return mask


def rasterize_contours(instances: List[Dict], H: int, W: int, edge_only=False) -> torch.Tensor:
    """Convert contours to filled masks or edges."""
    def _as_xy_array(contour):
        if contour is None: return np.empty((0, 2), dtype=np.float32)
        arr = np.asarray(contour, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 2: arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] != 2: return np.empty((0, 2), dtype=np.float32)
        return arr[np.isfinite(arr).all(axis=1)]
    
    mask = torch.zeros((H, W), dtype=torch.float32)
    for inst in instances:
        coords = _as_xy_array(inst.get("contour", []))
        if coords.shape[0] < 3: continue
        xs, ys = np.clip(coords[:, 0], 0, W-1), np.clip(coords[:, 1], 0, H-1)
        if np.max(xs) == np.min(xs) or np.max(ys) == np.min(ys): continue
        try:
            rr, cc = polygon(ys, xs, shape=(H, W))
            mask[rr, cc] = 1.0
        except: continue
    
    if edge_only:
        filled = mask.numpy().astype(bool)
        if filled.any():
            edge = filled & (~binary_erosion(filled))
            mask = torch.from_numpy(edge.astype(np.float32))
    return mask


def rasterize_types(instances: List[Dict], H: int, W: int, num_types: int, 
                    mode='points') -> torch.Tensor:
    """Convert hovernet instance types to multi-channel maps."""
    maps = torch.zeros((num_types, H, W), dtype=torch.float32)
    
    if mode == 'points':
        for inst in instances:
            t, c = int(inst.get("type", 0)), inst.get("centroid")
            if c is None or not (0 <= t < num_types): continue
            x, y = map(int, map(round, c))
            if 0 <= y < H and 0 <= x < W:
                maps[t, y, x] = 1.0
    
    elif mode == 'gaussian':
        yy = torch.arange(H, dtype=torch.float32).view(H, 1)
        xx = torch.arange(W, dtype=torch.float32).view(1, W)
        for inst in instances:
            t, c = int(inst.get("type", 0)), inst.get("centroid")
            if c is None or not (0 <= t < num_types): continue
            x_c, y_c = map(float, c)
            
            box = inst.get("box")
            sigma = 2.0
            if box:
                w, h = max(1.0, box[2]-box[0]), max(1.0, box[3]-box[1])
                sigma = max(2.0, 0.25 * max(w, h))
            
            r = int(max(1, 3*sigma))
            x, y = int(np.clip(np.round(x_c), 0, W-1)), int(np.clip(np.round(y_c), 0, H-1))
            y0, y1 = max(0, y-r), min(H, y+r+1)
            x0, x1 = max(0, x-r), min(W, x+r+1)
            
            g = torch.exp(-((yy[y0:y1]-y_c)**2 + (xx[x0:x1]-x_c)**2) / (2*sigma*sigma))
            maps[t, y0:y1, x0:x1] = torch.maximum(maps[t, y0:y1, x0:x1], g)
    
    return maps.clamp_(0, 1)

# ============================================================================
# RASTERIZER CLASSES
# ============================================================================

class Rasterizer(ABC):
    """
    Contract:
      - load_tool_output(path) -> raw tool output (dict/list/etc.)
      - rasterize(tool_output, H, W, **kwargs) -> Dict[str, torch.Tensor]
        Keys should be stable feature names (e.g., "type", "box", "centroid", "contour")
      - stack_features(features) -> torch.Tensor [C,H,W] (float32, [0,1])
      - get_num_channels() -> int
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
    
    @abstractmethod
    def load_tool_output(self, path: Path) -> Any:
        """Load tool output from file."""
        pass
    
    @abstractmethod
    def rasterize(self, tool_output: Any, H: int, W: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert to feature maps."""
        pass
    
    @abstractmethod
    def stack_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack into [C, H, W]."""
        pass
    
    @abstractmethod
    def get_num_channels(self) -> int:
        """Return total channels."""
        pass
    
    def infer_size(self, tool_output: Any, image_path: Optional[Path] = None) -> Tuple[int, int]:
        """Infer (H, W) from image or tool output."""
        if image_path and Path(image_path).exists():
            with Image.open(image_path) as im:
                W, H = im.size
            return H, W
        return 256, 256
    
    def process_and_save(self, tool_output_path: Path, save_dir: Path, 
                        image_path: Optional[Path] = None, 
                        save_individual=False, **kwargs) -> Dict[str, Path]:
        """Complete pipeline: load → rasterize → save."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        tool_output = self.load_tool_output(tool_output_path)
        H, W = self.infer_size(tool_output, image_path)
        features = self.rasterize(tool_output, H, W, **kwargs)
        
        stem = Path(tool_output_path).stem
        for suffix in ['_hovernet', '_custom']:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        
        saved = {}
        if save_individual:
            for key, tensor in features.items():
                path = save_dir / f"{stem}_{key}.pt"
                torch.save(tensor, path)
                saved[key] = path
        
        stacked = self.stack_features(features)
        stacked_path = save_dir / f"{stem}_features.pt"
        torch.save(stacked, stacked_path)
        saved['stacked'] = stacked_path
        
        return saved


class HoverNetRasterizer(Rasterizer):
    """Rasterizer for HoVer-Net nucleus outputs."""
    
    def __init__(self, name="hovernet_features", num_types=6,
                 include_box=True, include_centroid=True, include_contour=True,
                 include_types=True, type_mode='points', contour_mode='filled', **kwargs):
        super().__init__(name, **kwargs)
        self.num_types = num_types
        self.include_box = include_box
        self.include_centroid = include_centroid
        self.include_contour = include_contour
        self.include_types = include_types
        self.type_mode = type_mode
        self.contour_mode = contour_mode
    
    def load_tool_output(self, path: Path) -> Dict:
        return joblib.load(path)
    
    def rasterize(self, tool_output: Dict, H: int, W: int, **kwargs) -> Dict[str, torch.Tensor]:
        instances = list(tool_output.values()) if isinstance(tool_output, dict) else []
        features = {}
        
        if self.include_box:
            features['box'] = rasterize_boxes(instances, H, W)
        if self.include_centroid:
            features['centroid'] = rasterize_centroids(instances, H, W)
        if self.include_contour:
            features['contour'] = rasterize_contours(instances, H, W, 
                                                     edge_only=(self.contour_mode=='edge'))
        if self.include_types:
            features['type'] = rasterize_types(instances, H, W, self.num_types, 
                                              mode=self.type_mode)
        
        return features
    
    def stack_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        feature_list = []
        if 'type' in features:
            feature_list.append(features['type'])
        for key in ['box', 'centroid', 'contour']:
            if key in features:
                f = features[key]
                if f.ndim == 2: f = f.unsqueeze(0)
                feature_list.append(f)
        return torch.cat(feature_list, dim=0)
    
    def get_num_channels(self) -> int:
        count = 0
        if self.include_types: count += self.num_types
        if self.include_box: count += 1
        if self.include_centroid: count += 1
        if self.include_contour: count += 1
        return count


# Users add custom rasterizers here:
# class MyCustomRasterizer(Rasterizer):
#     ...