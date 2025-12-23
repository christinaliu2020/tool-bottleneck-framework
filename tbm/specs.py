# tbm/specs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

@dataclass(frozen=True)
class ToolSpec:
    name: str                 # canonical id, e.g. "nuc_type", "box"
    channels: int             # how many channels this tool contributes
    aliases: Tuple[str, ...] = ()  # optional alternative ids

class FeatureSpec:
    """
    Minimal registry of tool -> channel layout for stacked [C,H,W] tensors.
    One source of truth used by datasets, models, and dropout regimes.
    """
    def __init__(self, tools: List[ToolSpec]):
        if not tools:
            raise ValueError("FeatureSpec needs at least one ToolSpec.")
        self._tools = tools
        self._slices: Dict[str, Tuple[int, int]] = {}
        self._alias2name: Dict[str, str] = {}
        c = 0
        for t in tools:
            if t.channels <= 0:
                raise ValueError(f"Tool {t.name} has non-positive channels.")
            self._slices[t.name] = (c, c + t.channels)
            c += t.channels
            for a in t.aliases:
                key = self._norm(a)
                if key in self._alias2name and self._alias2name[key] != t.name:
                    raise ValueError(f"Alias clash: {a}")
                self._alias2name[key] = t.name
        self._C = c

    @staticmethod
    def _norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    @property
    def C(self) -> int:
        return self._C

    @property
    def names(self) -> List[str]:
        return [t.name for t in self._tools]

    @property
    def slices(self) -> Dict[str, Tuple[int, int]]:
        return dict(self._slices)

    def resolve(self, name_or_alias: str) -> str:
        k = self._norm(name_or_alias)
        for n in self._slices:
            if self._norm(n) == k:
                return n
        return self._alias2name.get(k, name_or_alias)

    def slice(self, name_or_alias: str) -> Tuple[int, int]:
        n = self.resolve(name_or_alias)
        if n not in self._slices:
            raise KeyError(f"Unknown tool '{name_or_alias}'. Known: {self.names}")
        return self._slices[n]

    # tiny helpers to customize
    def subset(self, keep: Iterable[str]) -> "FeatureSpec":
        keep_resolved = set(self.resolve(k) for k in keep)
        return FeatureSpec([t for t in self._tools if t.name in keep_resolved])

    def extend(self, extra: Iterable[ToolSpec]) -> "FeatureSpec":
        return FeatureSpec(self._tools + list(extra))


# ---- Tiny example builders (optional) ----
def hovernet_spec() -> FeatureSpec:
    # 6 type maps + 4 single-channel maps
    tools = [
        ToolSpec("nuc_type", 6, aliases=("histo_nuc_type",)),
        ToolSpec("box", 1,   aliases=("histo_nuc_bbox","bbox")),
        ToolSpec("centroid", 1, aliases=("histo_nuc_centroid",)),
        ToolSpec("contour", 1,  aliases=("histo_nuc_contour",)),
        ToolSpec("nuc_type_prob", 1,  aliases=("histo_nuc_type_prob",)),
    ]
    return FeatureSpec(tools)
