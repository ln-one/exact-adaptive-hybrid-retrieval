"""Strict HTTP client for the frozen Stratumind and Qdrant query contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import httpx

from .artifacts import QueryInput
from .fusion import PointId, identity_key

E5_PRODUCER_PLANS = {
    "pvs-pbm": "canonical-e5-pvs-pbm",
    "scalar-pbm": "canonical-e5-scalar-pbm",
    "scan-pbm": "canonical-e5-scan-pbm",
    "pvs-sparse-materialized": "canonical-e5-pvs-sparse-materialized",
}

E2_EXHAUSTIVE_PRODUCER_PLANS = {
    "pvs-pbm": "canonical-e2-pvs-pbm-exhaustive",
    "scalar-pbm": "canonical-e2-scalar-pbm-exhaustive",
    "scan-pbm": "canonical-e2-scan-pbm-exhaustive",
    "pvs-sparse-materialized": "canonical-e2-pvs-sparse-materialized-exhaustive",
}


@dataclass(frozen=True)
class ExactRrfResult:
    point_ids: tuple[PointId, ...]
    execution: dict[str, Any]


@dataclass(frozen=True)
class ExactChannelPrefix:
    point_ids: tuple[PointId, ...]
    fetched_points: int
    request_count: int
    exhausted: bool


class QueryClient:
    def __init__(
        self,
        base_url: str,
        collection: str,
        *,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.collection = collection
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> QueryClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def server_info(self) -> dict[str, Any]:
        response = self._client.get("/")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Qdrant root response is not an object")
        return {
            "title": payload.get("title"),
            "version": payload.get("version"),
            "commit": payload.get("commit"),
        }

    def collection_info(self) -> dict[str, Any]:
        response = self._client.get(f"/collections/{self.collection}")
        response.raise_for_status()
        payload = response.json().get("result")
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            raise RuntimeError("Qdrant collection response is missing result.config")
        return {
            "pointsCount": payload.get("points_count"),
            "indexedVectorsCount": payload.get("indexed_vectors_count"),
            "segmentsCount": payload.get("segments_count"),
            "config": payload["config"],
        }

    def exact_channel_order(
        self,
        query: QueryInput,
        *,
        channel: str,
        limit: int,
        dense_name: str,
        sparse_name: str,
    ) -> list[PointId]:
        return list(
            self.exact_channel_prefix(
                query,
                channel=channel,
                limit=limit,
                corpus_points=limit,
                dense_name=dense_name,
                sparse_name=sparse_name,
            ).point_ids
        )

    def exact_channel_prefix(
        self,
        query: QueryInput,
        *,
        channel: str,
        limit: int,
        corpus_points: int,
        dense_name: str,
        sparse_name: str,
    ) -> ExactChannelPrefix:
        if limit <= 0 or corpus_points <= 0:
            raise ValueError("channel prefix limit and corpus size must be positive")
        if channel == "dense":
            vector: object = query.dense
            using = dense_name
        elif channel == "sparse":
            vector = {"indices": query.sparse_indices, "values": query.sparse_values}
            using = sparse_name
        else:
            raise ValueError(f"unsupported channel: {channel}")
        target = min(limit, corpus_points)
        fetch_limit = min(corpus_points, target + 1)
        request_count = 0
        while True:
            request_count += 1
            scored = self._exact_channel_scores(vector=vector, using=using, limit=fetch_limit)
            exhausted = len(scored) < fetch_limit or fetch_limit == corpus_points
            boundary_closed = len(scored) <= target or scored[target - 1][1] > scored[target][1]
            if exhausted or boundary_closed:
                return ExactChannelPrefix(
                    point_ids=tuple(identity for identity, _ in scored[:target]),
                    fetched_points=len(scored),
                    request_count=request_count,
                    exhausted=exhausted,
                )
            fetch_limit = min(corpus_points, max(fetch_limit + 1, fetch_limit * 2))

    def _exact_channel_scores(
        self,
        *,
        vector: object,
        using: str,
        limit: int,
    ) -> list[tuple[PointId, float]]:
        response = self._client.post(
            f"/collections/{self.collection}/points/query",
            json={
                "query": vector,
                "using": using,
                "params": {"exact": True},
                "limit": limit,
                "with_payload": False,
                "with_vector": False,
            },
        )
        response.raise_for_status()
        points = response.json().get("result", {}).get("points")
        if not isinstance(points, list):
            raise RuntimeError("Qdrant exact channel response is missing result.points")
        scored: list[tuple[PointId, float]] = []
        for point in points:
            identity = point.get("id")
            score = point.get("score")
            if (
                not isinstance(identity, int | str)
                or isinstance(identity, bool)
                or not isinstance(score, int | float)
                or isinstance(score, bool)
                or not math.isfinite(score)
            ):
                raise RuntimeError("Qdrant exact channel returned an invalid scored identity")
            scored.append((identity, float(score)))
        identities = [identity for identity, _ in scored]
        if len(identities) != len(set(identities)):
            raise RuntimeError("Qdrant exact channel returned duplicate identities")
        scored.sort(key=lambda item: (-item[1], identity_key(item[0])))
        return scored

    def exact_rrf(
        self,
        query: QueryInput,
        *,
        dense_name: str,
        sparse_name: str,
        k: int,
        weights: tuple[float, float],
        limit: int,
    ) -> ExactRrfResult:
        return self._exact_rrf(
            query,
            endpoint=f"/collections/{self.collection}/points/query/exact-rrf",
            expected_plan="exact-rank-session-v1",
            dense_name=dense_name,
            sparse_name=sparse_name,
            k=k,
            weights=weights,
            limit=limit,
        )

    def exhaustive_rrf(
        self,
        query: QueryInput,
        *,
        dense_name: str,
        sparse_name: str,
        k: int,
        weights: tuple[float, float],
        limit: int,
    ) -> ExactRrfResult:
        result = self._exact_rrf(
            query,
            endpoint=(f"/internal/collections/{self.collection}/points/query/exact-rrf-exhaustive"),
            expected_plan="exact-rank-session-exhaustive-benchmark-v1",
            dense_name=dense_name,
            sparse_name=sparse_name,
            k=k,
            weights=weights,
            limit=limit,
        )
        if result.execution.get("stopReason") != "all-sources-exhausted" or result.execution.get(
            "sourceExhausted"
        ) != [True, True]:
            raise RuntimeError("same-producer exhaustive baseline did not drain both channels")
        return result

    def producer_rrf(
        self,
        query: QueryInput,
        *,
        producer: str,
        exhaustive: bool = False,
        dense_name: str,
        sparse_name: str,
        k: int,
        weights: tuple[float, float],
        limit: int,
    ) -> ExactRrfResult:
        try:
            expected_plan = (
                E2_EXHAUSTIVE_PRODUCER_PLANS if exhaustive else E5_PRODUCER_PLANS
            )[producer]
        except KeyError as error:
            raise ValueError(f"unsupported exact producer: {producer}") from error
        exhaustive_parameter = "&exhaustive=true" if exhaustive else ""
        result = self._exact_rrf(
            query,
            endpoint=(
                f"/internal/collections/{self.collection}/points/query/"
                f"exact-rrf-producer?producer={producer}{exhaustive_parameter}"
            ),
            expected_plan=expected_plan,
            dense_name=dense_name,
            sparse_name=sparse_name,
            k=k,
            weights=weights,
            limit=limit,
        )
        self._validate_e5_producer(result.execution, producer)
        if exhaustive and (
            result.execution.get("stopReason") != "all-sources-exhausted"
            or result.execution.get("sourceExhausted") != [True, True]
        ):
            raise RuntimeError("forced exact producer baseline did not drain both channels")
        return result

    @staticmethod
    def _validate_e5_producer(execution: dict[str, Any], producer: str) -> None:
        telemetry = execution.get("producer")
        if not isinstance(telemetry, dict):
            raise RuntimeError("E5 execution is missing producer telemetry")
        dense_counts = {
            "pvs": telemetry.get("densePvsSegments"),
            "scalar": telemetry.get("denseScalarSegments"),
            "scan": telemetry.get("denseScanSegments"),
        }
        sparse_counts = {
            "pbm": telemetry.get("sparsePbmSegments"),
            "materialized": telemetry.get("sparseMaterializedSegments"),
        }
        if any(not isinstance(value, int) or value < 0 for value in dense_counts.values()):
            raise RuntimeError(f"E5 Dense producer counters are invalid: {dense_counts!r}")
        if any(not isinstance(value, int) or value < 0 for value in sparse_counts.values()):
            raise RuntimeError(f"E5 Sparse producer counters are invalid: {sparse_counts!r}")
        expected_dense = (
            "scalar" if producer == "scalar-pbm" else "scan" if producer == "scan-pbm" else "pvs"
        )
        expected_sparse = "materialized" if producer == "pvs-sparse-materialized" else "pbm"
        if dense_counts[expected_dense] <= 0 or any(
            value != 0 for name, value in dense_counts.items() if name != expected_dense
        ):
            raise RuntimeError(
                f"E5 forced Dense producer was not honored for {producer}: {dense_counts!r}"
            )
        if sparse_counts[expected_sparse] <= 0 or any(
            value != 0 for name, value in sparse_counts.items() if name != expected_sparse
        ):
            raise RuntimeError(
                f"E5 forced Sparse producer was not honored for {producer}: {sparse_counts!r}"
            )
        expected_fallback = producer == "pvs-sparse-materialized"
        if execution.get("exhaustiveFallback") is not expected_fallback:
            raise RuntimeError(f"E5 fallback state does not match the selected producer {producer}")

    def _exact_rrf(
        self,
        query: QueryInput,
        *,
        endpoint: str,
        expected_plan: str,
        dense_name: str,
        sparse_name: str,
        k: int,
        weights: tuple[float, float],
        limit: int,
    ) -> ExactRrfResult:
        response = self._client.post(
            endpoint,
            json={
                "exact_rrf": {
                    "dense": {"query": query.dense, "using": dense_name},
                    "sparse": {
                        "query": {
                            "indices": query.sparse_indices,
                            "values": query.sparse_values,
                        },
                        "using": sparse_name,
                    },
                    "k": k,
                    "weights": list(weights),
                },
                "limit": limit,
            },
        )
        if not response.is_success:
            raise RuntimeError(
                f"Stratumind exact request failed ({response.status_code}): {response.text}"
            )
        payload = response.json().get("result")
        if not isinstance(payload, dict):
            raise RuntimeError("Stratumind response is missing result")
        guarantee = payload.get("guarantee")
        expected_guarantee = {
            "scope": "selected-local-shards-frozen-segment-view",
            "orderedTopKExact": True,
            "tieBreak": "point-identity-ascending",
            "channelInput": "exact-channel-rank-streams",
        }
        if guarantee != expected_guarantee:
            raise RuntimeError(f"Stratumind exact guarantee mismatch: {guarantee!r}")
        points = payload.get("points")
        if not isinstance(points, list):
            raise RuntimeError("Stratumind response is missing result.points")
        if [point.get("rank") for point in points] != list(range(1, len(points) + 1)):
            raise RuntimeError("Stratumind response ranks are not contiguous and one-based")
        identities = tuple(point.get("id") for point in points)
        if any(not isinstance(value, int | str) or isinstance(value, bool) for value in identities):
            raise RuntimeError("Stratumind returned an invalid point identity")
        if len(identities) != len(set(identities)):
            raise RuntimeError("Stratumind returned duplicate identities")
        execution = payload.get("execution")
        if not isinstance(execution, dict):
            raise RuntimeError("Stratumind response is missing execution telemetry")
        if execution.get("plan") != expected_plan:
            raise RuntimeError(f"Stratumind execution plan mismatch: {execution.get('plan')!r}")
        return ExactRrfResult(point_ids=identities, execution=execution)
