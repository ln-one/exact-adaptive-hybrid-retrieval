from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_e5_v2 import aggregate
from canonical_runner.artifacts import QueryInput
from canonical_runner.client import ExactRrfResult, QueryClient
from canonical_runner.e5_v2 import (
    PRODUCERS,
    WILLIAMS_ORDERS,
    _protocol_labels,
    _query_record,
    load_campaign,
)
from canonical_runner.logs import AtomicJsonlWriter
from canonical_runner.provenance import canonical_hash
from canonical_runner.server import ManagedQdrant
from canonical_runner.validation import validate_log


def _write_checksummed_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def _campaign(*, shards: int = 1) -> dict[str, object]:
    query_specs = [
        {
            "queryId": f"q{index + 1}",
            "canonicalIndex": index,
            "blockOrder": list(WILLIAMS_ORDERS[index % 4]),
        }
        for index in range(shards)
    ]
    return {
        "schema": "ed-wrrf-e5-v2-campaign-v1",
        "campaignId": "campaign",
        "experiment": "E5-v2",
        "dataset": "test",
        "queryIds": [spec["queryId"] for spec in query_specs],
        "parameters": {
            "rounds": 1,
            "warmups": 0,
            "repetitions": 1,
            "producers": list(PRODUCERS),
        },
        "rounds": [
            {
                "round": 1,
                "rotation": 0,
                "shards": [
                    {"shard": index + 1, "queries": [spec]}
                    for index, spec in enumerate(query_specs)
                ],
            }
        ],
    }


def _observation(producer: str, latency: int = 10) -> dict[str, object]:
    return {
        "warmup": False,
        "repetition": 1,
        "orderedIds": [1],
        "orderedResultSha256": canonical_hash([1]),
        "latencyNs": latency,
        "plan": producer,
        "stopReason": "certified",
        "sourcePulls": [16, 16],
        "sourceExhausted": [False, False],
        "certificationChecks": 1,
        "sourcePointsMaterialized": [16, 16],
        "producer": {"denseExactScores": 1},
    }


def _query_record_for_log(query_id: str, run_id: str) -> dict[str, object]:
    logical = {
        "sourcePulls": [16, 16],
        "sourceExhausted": [False, False],
        "certificationChecks": 1,
        "sourcePointsMaterialized": [16, 16],
        "stopReason": "certified",
    }
    return {
        "recordType": "query",
        "runId": run_id,
        "queryId": query_id,
        "sequence": 0,
        "round": 1,
        "shard": int(query_id[1:]),
        "blockOrder": list(WILLIAMS_ORDERS[(int(query_id[1:]) - 1) % 4]),
        "status": "ok",
        "orderedIds": [1],
        "oracleOrderedIds": [1],
        "orderedResultSha256": canonical_hash([1]),
        "oracleOrderedResultSha256": canonical_hash([1]),
        "membershipMismatch": False,
        "orderMismatch": False,
        "tieMismatch": False,
        "logicalSignature": logical,
        "blocks": {
            producer: [_observation(producer, latency=10 + index)]
            for index, producer in enumerate(PRODUCERS)
        },
    }


def _write_log(
    path: Path,
    campaign_path: Path,
    query_id: str,
    *,
    superseded_failures: list[dict[str, str]] | None = None,
) -> None:
    run_id = f"run-{query_id}"
    query = _query_record_for_log(query_id, run_id)
    run = {
        "recordType": "run",
        "schema": "ed-wrrf-results-v1",
        "runId": run_id,
        "experiment": "E5-v2",
        "campaignId": "campaign",
        "campaignManifestSha256": hashlib.sha256(campaign_path.read_bytes()).hexdigest(),
        "round": 1,
        "shard": int(query_id[1:]),
        "queryIds": [query_id],
        "dirty": True,
        "supersedesFailedAttempts": superseded_failures or [],
        "parameters": {
            "producers": list(PRODUCERS),
            "warmups": 0,
            "repetitions": 1,
            "queryLimit": 1,
        },
    }
    summary = {
        "recordType": "summary",
        "runId": run_id,
        "attemptedQueries": 1,
        "uniqueQueries": 1,
        "warmupObservations": 0,
        "measuredObservations": 4,
        "okQueries": 1,
        "mismatchQueries": 0,
        "timeoutQueries": 0,
        "errorQueries": 0,
        "queryRecordSha256": canonical_hash([canonical_hash(query)]),
    }
    path.write_text(
        "\n".join(json.dumps(value, sort_keys=True) for value in (run, query, summary)) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


class E5V2ScheduleTests(unittest.TestCase):
    def test_protocol_labels_distinguish_counterbalanced_and_self_warmed_runs(self) -> None:
        self.assertEqual(
            _protocol_labels(0),
            (
                "counterbalanced-producer-ablation",
                "counterbalanced-no-per-plan-warmup",
            ),
        )
        self.assertEqual(
            _protocol_labels(2),
            (
                "method-self-warmed-producer-ablation",
                "method-self-warmed-blocked",
            ),
        )

    def test_williams_orders_balance_positions_and_predecessors(self) -> None:
        for position in range(4):
            self.assertEqual({order[position] for order in WILLIAMS_ORDERS}, set(PRODUCERS))
        predecessors = {
            (left, right)
            for order in WILLIAMS_ORDERS
            for left, right in zip(order, order[1:])
        }
        expected = {(left, right) for left in PRODUCERS for right in PRODUCERS if left != right}
        self.assertEqual(predecessors, expected)

    def test_campaign_requires_every_query_in_every_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign.json"
            value = _campaign()
            value["parameters"]["rounds"] = 2  # type: ignore[index]
            _write_checksummed_json(path, value)
            with self.assertRaisesRegex(ValueError, "every Query"):
                load_campaign(path)


class E5V2BlockTests(unittest.TestCase):
    def test_query_executes_complete_self_warmed_blocks(self) -> None:
        calls: list[str] = []

        class FakeClient:
            def producer_rrf(
                self, query: QueryInput, *, producer: str, **_: object
            ) -> ExactRrfResult:
                calls.append(producer)
                return ExactRrfResult(
                    point_ids=(1,),
                    execution={
                        "plan": producer,
                        "stopReason": "certified",
                        "sourcePulls": [16, 16],
                        "sourceExhausted": [False, False],
                        "certificationChecks": 1,
                        "sourcePointsMaterialized": [16, 16],
                        "producer": {"denseExactScores": 1},
                    },
                )

        order = list(WILLIAMS_ORDERS[0])
        campaign = {
            "parameters": {
                "warmups": 2,
                "repetitions": 4,
                "shardWallTimeoutSeconds": 60,
                "denseName": "dense",
                "sparseName": "sparse",
                "rrfK": 60,
                "weights": [1.0, 1.0],
                "limit": 20,
            }
        }
        record = _query_record(
            FakeClient(),  # type: ignore[arg-type]
            QueryInput("q", [0.0], [], []),
            {"queryId": "q", "blockOrder": order},
            campaign,
            0,
            1,
            1,
            time.monotonic(),
        )
        self.assertEqual(calls, [producer for producer in order for _ in range(6)])
        for producer in order:
            self.assertEqual(
                [observation["warmup"] for observation in record["blocks"][producer]],
                [True, True, False, False, False, False],
            )


class E5V2ArtifactTests(unittest.TestCase):
    def test_atomic_writer_can_preserve_a_failed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "success.jsonl"
            failed = root / "failed" / "attempt.failed.jsonl"
            with AtomicJsonlWriter(destination) as writer:
                writer.write({"recordType": "run"})
                writer.commit_as(failed)
            self.assertFalse(destination.exists())
            self.assertTrue(failed.exists())

    def test_validation_and_aggregation_require_complete_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign_path = root / "campaign.json"
            _write_checksummed_json(campaign_path, _campaign(shards=2))
            first = root / "r1-s1.jsonl"
            second = root / "r1-s2.jsonl"
            _write_log(first, campaign_path, "q1")
            _write_log(second, campaign_path, "q2")
            self.assertEqual(validate_log(first, require_clean=False)["okQueries"], 1)
            with self.assertRaisesRegex(ValueError, "missing E5-v2 shards"):
                aggregate(campaign_path, [first], seed=7, samples=10, require_clean=False)
            result = aggregate(
                campaign_path, [first, second], seed=7, samples=10, require_clean=False
            )
            self.assertEqual(result["queryCount"], 2)

    def test_aggregation_requires_every_failed_attempt_to_be_superseded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign_path = root / "campaign.json"
            _write_checksummed_json(campaign_path, _campaign())
            failed_dir = root / "failed"
            failed_dir.mkdir()
            failed = failed_dir / "r1-s1-attempt-1.failed.jsonl"
            failed.write_text(
                json.dumps({"campaignId": "campaign", "status": "failed"}) + "\n",
                encoding="utf-8",
            )
            failed_sha256 = hashlib.sha256(failed.read_bytes()).hexdigest()
            failed.with_suffix(failed.suffix + ".sha256").write_text(
                f"{failed_sha256}  {failed.name}\n", encoding="utf-8"
            )
            success = root / "r1-s1.jsonl"
            _write_log(success, campaign_path, "q1")
            with self.assertRaisesRegex(ValueError, "unresolved or incorrectly superseded"):
                aggregate(
                    campaign_path,
                    [success],
                    seed=7,
                    samples=10,
                    require_clean=False,
                    failed_dir=failed_dir,
                )

            _write_log(
                success,
                campaign_path,
                "q1",
                superseded_failures=[
                    {"file": failed.name, "sha256": failed_sha256}
                ],
            )
            result = aggregate(
                campaign_path,
                [success],
                seed=7,
                samples=10,
                require_clean=False,
                failed_dir=failed_dir,
            )
            self.assertEqual(result["queryCount"], 1)

    def test_managed_server_preserves_startup_failure_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "failing-server"
            binary.write_text("#!/bin/sh\necho startup-failure\nexit 1\n", encoding="utf-8")
            binary.chmod(0o755)
            failure_log = root / "evidence" / "qdrant.log"
            with self.assertRaisesRegex(RuntimeError, "exited during startup"):
                with ManagedQdrant(
                    binary=binary,
                    system_repo=root,
                    collection="c",
                    snapshot=None,
                    startup_timeout_seconds=2,
                    failure_log_path=failure_log,
                ):
                    self.fail("failing server unexpectedly became ready")
            self.assertIn("startup-failure", failure_log.read_text(encoding="utf-8"))


class E5V2NoRetryTests(unittest.TestCase):
    def test_formal_client_does_not_retry_slot_reservation_failure(self) -> None:
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(500, text="cannot reserve exact query slot")

        with QueryClient(
            "http://test",
            "c",
            transport=httpx.MockTransport(handler),
            slot_retry_max=1,
        ) as client:
            with self.assertRaisesRegex(RuntimeError, "after 1 retries"):
                client.producer_rrf(
                    QueryInput("q", [0.0], [], []),
                    producer="pvs-pbm",
                    dense_name="dense",
                    sparse_name="sparse",
                    k=60,
                    weights=(1.0, 1.0),
                    limit=20,
                )
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
