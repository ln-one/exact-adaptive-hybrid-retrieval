"""Deterministic synthetic rank regimes and an exact balanced ED-WRRF certificate."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

REGIMES = (
    "correlated",
    "partial-overlap",
    "independent",
    "anti-correlated",
    "all-tied",
)


@dataclass(frozen=True)
class SyntheticRankings:
    first: np.ndarray
    second: np.ndarray
    top_window_overlap: float


@dataclass(frozen=True)
class CertificateResult:
    depth: int
    output: tuple[int, ...]
    checks: int
    kth_lower: float
    anonymous_upper: float


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))


def generate_rankings(
    *,
    size: int,
    seed: int,
    regime: str,
    overlap_window: int = 1_000,
) -> SyntheticRankings:
    if size <= 1:
        raise ValueError("synthetic universe must contain at least two identities")
    if regime not in REGIMES:
        raise ValueError(f"unsupported synthetic regime: {regime}")
    window = min(size, overlap_window)

    if regime == "all-tied":
        first = np.arange(size, dtype=np.int32)
        second = first.copy()
        return SyntheticRankings(first, second, 1.0)

    first = _rng(seed, 0).permutation(size).astype(np.int32, copy=False)
    if regime == "correlated":
        block = 32
        full = size // block * block
        second = first.copy()
        if full:
            local_order = _rng(seed, 1).permutation(block)
            second[:full] = first[:full].reshape(-1, block)[:, local_order].reshape(-1)
    elif regime == "partial-overlap":
        shared_count = window // 2
        shared = first[:shared_count]
        replacement_end = min(size, window + (window - shared_count))
        replacements = first[window:replacement_end]
        top = np.concatenate((shared, replacements))
        _rng(seed, 2).shuffle(top)
        selected = np.zeros(size, dtype=np.bool_)
        selected[top] = True
        remainder = first[~selected[first]]
        _rng(seed, 3).shuffle(remainder)
        second = np.concatenate((top, remainder))
    elif regime == "independent":
        second = _rng(seed, 4).permutation(size).astype(np.int32, copy=False)
    else:
        second = first[::-1].copy()

    top_overlap = len(set(first[:window].tolist()) & set(second[:window].tolist())) / window
    return SyntheticRankings(first, second, top_overlap)


def _rank_positions(order: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(len(order), dtype=np.int32)
    return ranks


def _contribution(ranks_one_based: np.ndarray | float, *, k: int, weight: float):
    if weight <= 0:
        return np.zeros_like(ranks_one_based, dtype=np.float64)
    return 1.0 / (np.asarray(ranks_one_based, dtype=np.float64) / weight + k - 1.0)


def _top_identities(values: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(values))
    if count <= 0:
        return np.empty(0, dtype=np.int32)
    selected = np.argpartition(-values, count - 1)[:count]
    order = np.lexsort((selected, -values[selected]))
    return selected[order].astype(np.int32, copy=False)


class BalancedCertificate:
    def __init__(
        self,
        rankings: SyntheticRankings,
        *,
        k: int,
        weights: tuple[float, float],
        limit: int,
        batch_size: int,
    ) -> None:
        if k <= 0 or limit <= 0 or batch_size <= 0:
            raise ValueError("k, limit, and batch size must be positive")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not any(
            weight > 0 for weight in weights
        ):
            raise ValueError("weights must be finite and non-negative with one positive value")
        if len(rankings.first) != len(rankings.second):
            raise ValueError("synthetic channels must share one identity universe")
        self.size = len(rankings.first)
        self.k = k
        self.weights = weights
        self.limit = limit
        self.batch_size = batch_size
        self.first_ranks = _rank_positions(rankings.first)
        self.second_ranks = _rank_positions(rankings.second)
        first_scores = _contribution(
            self.first_ranks.astype(np.int64) + 1,
            k=k,
            weight=weights[0],
        )
        second_scores = _contribution(
            self.second_ranks.astype(np.int64) + 1,
            k=k,
            weight=weights[1],
        )
        self.exhaustive_scores = first_scores + second_scores
        self.oracle = tuple(
            int(identity) for identity in _top_identities(self.exhaustive_scores, limit)
        )

    def _at_depth(self, depth: int) -> tuple[bool, tuple[int, ...], float, float]:
        if depth <= 0 or depth > self.size:
            raise ValueError("certificate depth is outside the synthetic universe")
        seen_first = self.first_ranks < depth
        seen_second = self.second_ranks < depth
        seen = seen_first | seen_second
        seen_count = int(np.count_nonzero(seen))
        if seen_count < self.limit:
            return False, (), 0.0, math.inf

        lower = np.zeros(self.size, dtype=np.float64)
        lower[seen_first] += _contribution(
            self.first_ranks[seen_first].astype(np.int64) + 1,
            k=self.k,
            weight=self.weights[0],
        )
        lower[seen_second] += _contribution(
            self.second_ranks[seen_second].astype(np.int64) + 1,
            k=self.k,
            weight=self.weights[1],
        )
        if depth == self.size:
            next_first = next_second = 0.0
        else:
            next_first = float(_contribution(depth + 1, k=self.k, weight=self.weights[0]))
            next_second = float(_contribution(depth + 1, k=self.k, weight=self.weights[1]))
        upper = lower.copy()
        upper[seen & ~seen_first] += next_first
        upper[seen & ~seen_second] += next_second
        anonymous_upper = next_first + next_second

        seen_ids = np.flatnonzero(seen).astype(np.int32, copy=False)
        lower_seen = lower[seen_ids]
        upper_seen = upper[seen_ids]
        lower_local = _top_identities(lower_seen, self.limit)
        lower_order = seen_ids[lower_local]
        upper_local = _top_identities(upper_seen, self.limit + 1)
        upper_order = seen_ids[upper_local]

        fixed: set[int] = set()
        for candidate_value in lower_order:
            candidate = int(candidate_value)
            candidate_lower = float(lower[candidate])
            if candidate_lower <= anonymous_upper:
                return (
                    False,
                    tuple(int(value) for value in lower_order),
                    float(lower[lower_order[-1]]),
                    anonymous_upper,
                )
            competitor = next(
                (
                    int(value)
                    for value in upper_order
                    if int(value) not in fixed and int(value) != candidate
                ),
                None,
            )
            if competitor is not None:
                competitor_upper = float(upper[competitor])
                if candidate_lower < competitor_upper or (
                    candidate_lower == competitor_upper and candidate > competitor
                ):
                    return (
                        False,
                        tuple(int(value) for value in lower_order),
                        float(lower[lower_order[-1]]),
                        anonymous_upper,
                    )
            fixed.add(candidate)
        return (
            True,
            tuple(int(value) for value in lower_order),
            float(lower[lower_order[-1]]),
            anonymous_upper,
        )

    def find_minimum_batch_depth(self) -> CertificateResult:
        batches = math.ceil(self.size / self.batch_size)
        low = 0
        high = batches
        checks = 0
        while low + 1 < high:
            middle = (low + high) // 2
            depth = min(self.size, middle * self.batch_size)
            certified, _, _, _ = self._at_depth(depth)
            checks += 1
            if certified:
                high = middle
            else:
                low = middle
        depth = min(self.size, high * self.batch_size)
        certified, output, kth_lower, anonymous_upper = self._at_depth(depth)
        checks += 1
        if not certified:
            raise RuntimeError("balanced certificate failed at complete exhaustion")
        return CertificateResult(
            depth=depth,
            output=output,
            checks=checks,
            kth_lower=kth_lower,
            anonymous_upper=anonymous_upper,
        )
