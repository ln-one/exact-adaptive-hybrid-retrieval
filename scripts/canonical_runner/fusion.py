"""The frozen Stratumind WRRF order used by the exhaustive oracle."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import numpy as np

PointId = int | str


def identity_key(value: PointId) -> tuple[int, int]:
    if isinstance(value, bool):
        raise TypeError("boolean is not a point identity")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("numeric point identity must be non-negative")
        return (0, value)
    return (1, uuid.UUID(value).int)


def position_score(position: int, k: int, weight: float) -> np.float32:
    if position < 0 or k <= 0:
        raise ValueError("position must be non-negative and k must be positive")
    weight32 = np.float32(weight)
    if not np.isfinite(weight32) or weight32 <= np.float32(0):
        return np.float32(0)
    denominator = np.float32(position + 1) / weight32 + np.float32(k) - np.float32(1)
    return np.float32(np.float32(1) / denominator)


def exact_wrrf(
    channel_orders: Sequence[Sequence[PointId]],
    *,
    k: int,
    weights: Sequence[float],
    limit: int,
) -> list[PointId]:
    return [
        point_id
        for point_id, _ in exact_wrrf_with_scores(channel_orders, k=k, weights=weights, limit=limit)
    ]


def exact_wrrf_with_scores(
    channel_orders: Sequence[Sequence[PointId]],
    *,
    k: int,
    weights: Sequence[float],
    limit: int,
) -> list[tuple[PointId, np.float32]]:
    if len(channel_orders) != len(weights):
        raise ValueError("one weight is required for each channel")
    if limit <= 0:
        raise ValueError("limit must be positive")

    scores: dict[PointId, np.float32] = {}
    for order, weight in zip(channel_orders, weights, strict=True):
        seen: set[PointId] = set()
        for position, point_id in enumerate(order):
            if point_id in seen:
                raise ValueError(f"duplicate identity in one channel: {point_id}")
            seen.add(point_id)
            contribution = position_score(position, k, weight)
            scores[point_id] = np.float32(scores.get(point_id, np.float32(0)) + contribution)

    positive = ((point_id, score) for point_id, score in scores.items() if score > 0)
    ordered = sorted(positive, key=lambda item: (-float(item[1]), identity_key(item[0])))
    return ordered[:limit]
