"""Plotting helpers shared by UI entry points."""

from __future__ import annotations


def padded_axis_limits(
    values: list[float],
    *,
    min_span: float,
    padding_ratio: float = 0.18,
    nonnegative: bool = False,
) -> tuple[float, float]:
    """Return display limits that avoid over-tight matplotlib autoscaling."""
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return 0.0, min_span

    low = min(clean_values)
    high = max(clean_values)
    center = (low + high) / 2.0
    span = max(high - low, min_span)
    padded_span = span * (1.0 + padding_ratio)
    lower = center - padded_span / 2.0
    upper = center + padded_span / 2.0

    if nonnegative and low >= 0:
        lower = 0.0
        if upper - lower < min_span:
            upper = lower + min_span

    return lower, upper
