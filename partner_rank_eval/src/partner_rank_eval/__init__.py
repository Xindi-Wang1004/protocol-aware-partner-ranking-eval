"""Protocol-aware evaluation for constrained biological partner ranking."""

from .metrics import (
    QueryRecord,
    decompose,
    identity_residual,
    metrics_from_ranks,
)
from .report import CHECKLIST_ITEMS, empty_checklist, fill_checklist
from .simulate import (
    simulate_gallery_nesting,
    simulate_grid,
    simulate_retrieve_rerank,
)

__version__ = "0.1.0"

__all__ = [
    "QueryRecord",
    "CHECKLIST_ITEMS",
    "decompose",
    "empty_checklist",
    "fill_checklist",
    "identity_residual",
    "metrics_from_ranks",
    "simulate_gallery_nesting",
    "simulate_grid",
    "simulate_retrieve_rerank",
]
