# mypy: ignore-errors
"""Shared utilities for GPU kernel launch functions (cuTile and Triton)."""


def _select_block_b(n_oi: int, batch: int, base: int = 32) -> int:
    """Choose BLOCK_B adaptively based on grid size.

    For large grids, bigger blocks reduce total program count, state memory
    allocation, and per-block indexing overhead.
    """
    if base != 32:
        return base  # non-standard base (e.g. BLOCK_B=1 for f32 real)
    # Only scale up when n_oi is large enough that reducing batch blocks helps
    if n_oi >= 256 and batch >= 128:
        return 128
    if n_oi >= 256 and batch >= 64:
        return 64
    return 32
