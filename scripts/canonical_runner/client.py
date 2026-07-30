"""Strict HTTP client for the frozen Stratumind and Qdrant query contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import httpx

from .artifacts import QueryInput
from .fusion import PointId, identity_key


@dataclass(frozen=True)
class ExactRrfResult:
    point_ids: tuple[PointId, ...]
    execution: dict[str, Any]


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
        if channel == "dense":
            vector: object = query.dense
            using = dense_name
        elif channel == "sparse":
            vector = {"indices": query.sparse_indices, "values": query.sparse_values}
            using = sparse_name
        else:
            raise ValueError(f"unsupported channel: {channel}")
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
        payload = response.json()
        points = payload.get("result", {}).get("points")
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
        return [identity for identity, _ in scored]

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
        response.raise_for_status()
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
