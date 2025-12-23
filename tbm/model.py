# tbm/model.py
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ToolDropoutSharedEffB0(nn.Module):
    """
    EfficientNet-B0 with a custom input stem to accept [C,H,W] feature tensors.
    Tool masks are applied by channel blocks defined in TOOL_SLICES.
    """
    def __init__(self, in_channels: int, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # swap input conv
        old_stem = backbone.features[0][0]
        new_stem = nn.Conv2d(
            in_channels, old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=False
        )
        backbone.features[0][0] = new_stem
        self.blocks = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.classifier[1].in_features, num_classes)

        # default: treat whole C as one block if no TOOL_SLICES passed
        self.tool_slices: List[Tuple[int,int]] = [(0, in_channels)]

    def set_tool_slices(self, slices: Dict[str, Tuple[int,int]]):
        # expects dict: name -> (start, end)
        self.tool_slices = [slices[k] for k in slices]

    def _apply_mask(self, x: torch.Tensor, keep_mask: Optional[torch.Tensor]):
        """
        x: [B,C,H,W]
        keep_mask: [B, T] or None  (T = number of tool blocks)
        Each block is filled with -1 when dropped (missing-value token).
        """
        if keep_mask is None:
            return x
        B = x.size(0)
        assert keep_mask.shape[0] == B, "keep_mask batch mismatch"
        assert keep_mask.shape[1] == len(self.tool_slices), "mask/tool_slices mismatch"

        for t_idx, (s, e) in enumerate(self.tool_slices):
            m = keep_mask[:, t_idx].view(B,1,1,1)
            x[:, s:e] = x[:, s:e] * m + (-1.0) * (1.0 - m)
        return x

    def forward(self, x, keep_mask: Optional[torch.Tensor] = None):
        x = self._apply_mask(x, keep_mask)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
