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
    "pvs-pbm": "canonical-e2-bulk-native-exhaustive",
    "scalar-pbm": "canonical-e2-bulk-native-exhaustive",
    "scan-pbm": "canonical-e2-bulk-native-exhaustive",
    "pvs-sparse-materialized": "canonical-e2-bulk-native-exhaustive",
}

E2_SAME_PRODUCER_EXHAUSTIVE_PLANS = {
    "pvs-pbm": "canonical-e2-pvs-pbm-exhaustive",
    "scalar-pbm": "canonical-e2-scalar-pbm-exhaustive",
    "scan-pbm": "canonical-e2-scan-pbm-exhaustive",
    "pvs-sparse-materialized": "canonical-e2-pvs-sparse-materialized-exhaustive",
}


@dataclass(frozen=True)
class ExactRrfResult:
    point_ids: tuple[PointId, ...]
    execution: dict[str, Any]
    point_scores: tuple[float, ...] = ()


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
        slot_retry_max: int | None = None,
    ) -> None:
        if slot_retry_max is not None and slot_retry_max <= 0:
            raise ValueError("slot retry maximum must be positive")
        self.collection = collection
        self._slot_retry_max = slot_retry_max or self._SLOT_RETRY_MAX
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            # Canonical managed Qdrant runs must use the reserved local port
            # directly; an ambient HTTP proxy would invalidate timings.
            trust_env=False,
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

    def external_ids(
        self,
        point_ids: list[PointId] | tuple[PointId, ...],
        *,
        batch_size: int = 256,
    ) -> dict[PointId, str]:
        """Resolve frozen evaluation identities from canonical point payloads."""
        if batch_size <= 0:
            raise ValueError("payload batch size must be positive")
        unique = list(dict.fromkeys(point_ids))
        resolved: dict[PointId, str] = {}
        for offset in range(0, len(unique), batch_size):
            batch = unique[offset : offset + batch_size]
            response = self._client.post(
                f"/collections/{self.collection}/points",
                json={"ids": batch, "with_payload": True, "with_vector": False},
            )
            response.raise_for_status()
            points = response.json().get("result")
            if not isinstance(points, list):
                raise RuntimeError("Qdrant point lookup response is missing result")
            for point in points:
                identity = point.get("id")
                payload = point.get("payload")
                external_id = payload.get("external_id") if isinstance(payload, dict) else None
                if (
                    not isinstance(identity, int | str)
                    or isinstance(identity, bool)
                    or not isinstance(external_id, str)
                    or not external_id
                ):
                    raise RuntimeError("Qdrant point payload lacks a valid external_id")
                if identity in resolved:
                    raise RuntimeError("Qdrant point lookup returned a duplicate identity")
                resolved[identity] = external_id
        missing = set(unique) - resolved.keys()
        if missing:
            raise RuntimeError(f"Qdrant point lookup omitted {len(missing)} identities")
        if len(resolved.values()) != len(set(resolved.values())):
            raise RuntimeError("canonical payload maps multiple points to one external_id")
        return resolved

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
            positive_support_closed = False
            if channel == "sparse":
                if any(score < 0 for _, score in scored):
                    raise RuntimeError("exact Sparse channel returned a negative score")
                positive_support_closed = any(score == 0 for _, score in scored)
                scored = [(identity, score) for identity, score in scored if score > 0]
            exhausted = (
                positive_support_closed or len(scored) < fetch_limit or fetch_limit == corpus_points
            )
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
        mode: str | None = None,
        dense_name: str,
        sparse_name: str,
        k: int,
        weights: tuple[float, float],
        limit: int,
    ) -> ExactRrfResult:
        # Resolve execution mode: explicit mode param takes precedence over legacy bool.
        if mode is None:
            mode = "native-bulk-exhaustive" if exhaustive else "proof-driven"
        plan_maps = {
            "proof-driven": E5_PRODUCER_PLANS,
            "same-producer-exhaustive": E2_SAME_PRODUCER_EXHAUSTIVE_PLANS,
            "native-bulk-exhaustive": E2_EXHAUSTIVE_PRODUCER_PLANS,
        }
        try:
            expected_plan = plan_maps[mode][producer]
        except KeyError as error:
            raise ValueError(f"unsupported mode/producer: {mode}/{producer}") from error
        mode_parameter = f"&mode={mode}" if mode != "proof-driven" else ""
        result = self._exact_rrf(
            query,
            endpoint=(
                f"/internal/collections/{self.collection}/points/query/"
                f"exact-rrf-producer?producer={producer}{mode_parameter}"
            ),
            expected_plan=expected_plan,
            dense_name=dense_name,
            sparse_name=sparse_name,
            k=k,
            weights=weights,
            limit=limit,
        )
        if mode == "native-bulk-exhaustive":
            self._validate_bulk_exhaustive(result.execution)
        elif mode == "same-producer-exhaustive":
            self._validate_same_producer_exhaustive(result.execution, producer)
        else:
            self._validate_e5_producer(result.execution, producer)
        if mode != "proof-driven" and (
            result.execution.get("stopReason") != "all-sources-exhausted"
            or result.execution.get("sourceExhausted") != [True, True]
        ):
            raise RuntimeError("forced exact producer baseline did not drain both channels")
        return result

    @staticmethod
    def _validate_bulk_exhaustive(execution: dict[str, Any]) -> None:
        telemetry = execution.get("producer")
        if not isinstance(telemetry, dict):
            raise RuntimeError("bulk exhaustive execution is missing producer telemetry")
        dense_scan = telemetry.get("denseScanSegments")
        sparse_materialized = telemetry.get("sparseMaterializedSegments")
        if not isinstance(dense_scan, int) or dense_scan <= 0:
            raise RuntimeError(f"bulk exhaustive did not use native Dense scans: {telemetry!r}")
        if not isinstance(sparse_materialized, int) or sparse_materialized <= 0:
            raise RuntimeError(
                f"bulk exhaustive did not use native Sparse materialization: {telemetry!r}"
            )

    @staticmethod
    def _validate_same_producer_exhaustive(execution: dict[str, Any], producer: str) -> None:
        telemetry = execution.get("producer")
        if not isinstance(telemetry, dict):
            raise RuntimeError("same-producer exhaustive execution is missing producer telemetry")
        # The same-producer exhaustive arm must use the declared PVS/PBM producer,
        # not native Dense scan or Sparse materialization.
        expected_dense = (
            "denseScalarSegments"
            if producer == "scalar-pbm"
            else "denseScanSegments"
            if producer == "scan-pbm"
            else "densePvsSegments"
        )
        expected_sparse = (
            "sparseMaterializedSegments"
            if producer == "pvs-sparse-materialized"
            else "sparsePbmSegments"
        )
        dense_count = telemetry.get(expected_dense)
        if not isinstance(dense_count, int) or dense_count <= 0:
            raise RuntimeError(
                f"same-producer exhaustive did not use expected Dense producer "
                f"{expected_dense}: {telemetry!r}"
            )
        sparse_count = telemetry.get(expected_sparse)
        if not isinstance(sparse_count, int) or sparse_count <= 0:
            raise RuntimeError(
                f"same-producer exhaustive did not use expected Sparse producer "
                f"{expected_sparse}: {telemetry!r}"
            )

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

    _SLOT_RETRY_MAX = 15
    _SLOT_RETRY_BASE_SECONDS = 1.0
    _SLOT_RETRY_CAP_SECONDS = 10.0

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
        import time as _time

        payload_body = {
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
        }
        last_response = None
        for attempt in range(self._slot_retry_max):
            response = self._client.post(endpoint, json=payload_body)
            if response.is_success:
                last_response = response
                break
            # Transient slot-reservation failure: retry with backoff.
            if response.status_code == 500 and "cannot reserve" in response.text:
                last_response = response
                _time.sleep(
                    min(
                        self._SLOT_RETRY_BASE_SECONDS * (2**attempt),
                        self._SLOT_RETRY_CAP_SECONDS,
                    )
                )
                continue
            # Non-retryable error.
            raise RuntimeError(
                f"Stratumind exact request failed ({response.status_code}): {response.text}"
            )
        if last_response is None or not last_response.is_success:
            raise RuntimeError(
                f"Stratumind exact request failed after {self._slot_retry_max} retries "
                f"({last_response.status_code if last_response else 'no response'}): "
                f"{last_response.text if last_response else ''}"
            )
        response = last_response
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
        scores = tuple(point.get("score") for point in points)
        if any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in scores
        ):
            raise RuntimeError("Stratumind returned an invalid fused score")
        execution = payload.get("execution")
        if not isinstance(execution, dict):
            raise RuntimeError("Stratumind response is missing execution telemetry")
        if execution.get("plan") != expected_plan:
            raise RuntimeError(f"Stratumind execution plan mismatch: {execution.get('plan')!r}")
        return ExactRrfResult(
            point_ids=identities,
            execution=execution,
            point_scores=tuple(float(value) for value in scores),
        )
