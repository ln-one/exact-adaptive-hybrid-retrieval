from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from canonical_runner.artifacts import QueryInput
from canonical_runner.client import QueryClient
from canonical_runner.e2 import E2Config, run_e2
from canonical_runner.fusion import exact_wrrf, position_score
from canonical_runner.logs import AtomicJsonlWriter
from canonical_runner.provenance import canonical_hash
from canonical_runner.runner import E1Config, run_e1
from canonical_runner.validation import validate_log
from create_qdrant_collection import collection_schema


class FusionTests(unittest.TestCase):
    def test_position_score_matches_frozen_weighting(self) -> None:
        self.assertEqual(position_score(0, 60, 1.0), np.float32(1.0 / 60.0))
        self.assertEqual(position_score(0, 60, 2.0), np.float32(1.0 / 59.5))
        self.assertEqual(position_score(0, 60, 0.0), np.float32(0.0))

    def test_exact_wrrf_uses_deterministic_identity_ties(self) -> None:
        left = "00000000-0000-0000-0000-000000000001"
        right = "00000000-0000-0000-0000-000000000002"
        result = exact_wrrf([[right, left], [left, right]], k=60, weights=[1.0, 1.0], limit=2)
        self.assertEqual(result, [left, right])

    def test_duplicate_identity_in_one_channel_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate identity"):
            exact_wrrf([[1, 1], []], k=60, weights=[1.0, 1.0], limit=1)


class CollectionSchemaTests(unittest.TestCase):
    def test_schema_enables_the_explicit_production_profile(self) -> None:
        schema = collection_schema(
            dense_vector_name="dense",
            sparse_vector_name="sparse",
            shards=2,
            exact_rank_profile="dense_sparse_v1",
        )
        self.assertEqual(schema["exact_rank_config"], {"profile": "dense_sparse_v1"})
        self.assertEqual(schema["shard_number"], 2)


class RunnerConfigTests(unittest.TestCase):
    def test_invalid_weights_fail_before_repository_or_http_access(self) -> None:
        config = E1Config(
            artifact_root=Path("/unused"),
            dataset="unused",
            collection="unused",
            url="http://unused",
            output=Path("/unused"),
            bench_repo=Path("/unused"),
            system_repo=Path("/unused"),
            system_commit="unused",
            system_artifact="sha256:test",
            hardware_profile="test",
            weights=(0.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "weights"):
            run_e1(config)

    def test_e2_rejects_invalid_repetition_count_before_repository_access(self) -> None:
        config = E2Config(
            artifact_root=Path("/unused"),
            dataset="unused",
            collection="unused",
            url="http://unused",
            output=Path("/unused"),
            bench_repo=Path("/unused"),
            system_repo=Path("/unused"),
            system_commit="unused",
            system_artifact="sha256:test",
            hardware_profile="test",
            repetitions=0,
        )
        with self.assertRaisesRegex(ValueError, "repetitions"):
            run_e2(config)


class AtomicLogTests(unittest.TestCase):
    def test_writer_commits_once_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "run.jsonl"
            with AtomicJsonlWriter(destination) as writer:
                writer.write({"recordType": "run"})
                writer.commit()
            self.assertEqual(json.loads(destination.read_text()), {"recordType": "run"})
            with self.assertRaises(FileExistsError):
                AtomicJsonlWriter(destination)

    def test_failed_writer_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "run.jsonl"
            with (
                self.assertRaisesRegex(RuntimeError, "stop"),
                AtomicJsonlWriter(destination) as writer,
            ):
                writer.write({"recordType": "run"})
                raise RuntimeError("stop")
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])


class ValidationTests(unittest.TestCase):
    def test_validator_checks_summary_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            query_record = {
                "recordType": "query",
                "runId": "r1",
                "queryId": "q1",
                "sequence": 0,
                "status": "ok",
                "orderedIds": [1],
                "oracleOrderedIds": [1],
                "orderedResultSha256": canonical_hash([1]),
                "oracleOrderedResultSha256": canonical_hash([1]),
                "membershipMismatch": False,
                "orderMismatch": False,
            }
            records = [
                {
                    "recordType": "run",
                    "schema": "ed-wrrf-results-v1",
                    "runId": "r1",
                    "dirty": False,
                },
                query_record,
                {
                    "recordType": "summary",
                    "runId": "r1",
                    "attemptedQueries": 1,
                    "okQueries": 1,
                    "mismatchQueries": 0,
                    "timeoutQueries": 0,
                    "errorQueries": 0,
                    "queryRecordSha256": canonical_hash([canonical_hash(query_record)]),
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".jsonl.sha256").write_text(
                f"{digest}  {path.name}\n", encoding="utf-8"
            )
            self.assertEqual(
                validate_log(path),
                {
                    "attemptedQueries": 1,
                    "okQueries": 1,
                    "mismatchQueries": 0,
                    "timeoutQueries": 0,
                    "errorQueries": 0,
                },
            )

    def test_validator_rejects_dirty_publication_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            records = [
                {
                    "recordType": "run",
                    "schema": "ed-wrrf-results-v1",
                    "runId": "r1",
                    "dirty": True,
                },
                {
                    "recordType": "summary",
                    "runId": "r1",
                    "attemptedQueries": 0,
                    "okQueries": 0,
                    "mismatchQueries": 0,
                    "timeoutQueries": 0,
                    "errorQueries": 0,
                    "queryRecordSha256": canonical_hash([]),
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".jsonl.sha256").write_text(
                f"{digest}  {path.name}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "dirty repository"):
                validate_log(path)


class ClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query = QueryInput(
            query_id="q1",
            dense=[1.0, 0.0],
            sparse_indices=[3],
            sparse_values=[1.0],
        )

    def test_exact_channel_completes_score_ties_by_identity(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/collections/c/points/query")
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [
                            {"id": "00000000-0000-0000-0000-000000000002", "score": 0.5},
                            {"id": "00000000-0000-0000-0000-000000000001", "score": 0.5},
                        ]
                    }
                },
            )

        with QueryClient("http://test", "c", transport=httpx.MockTransport(handler)) as client:
            result = client.exact_channel_order(
                self.query,
                channel="dense",
                limit=2,
                dense_name="dense",
                sparse_name="sparse",
            )
        self.assertEqual(
            result,
            [
                "00000000-0000-0000-0000-000000000001",
                "00000000-0000-0000-0000-000000000002",
            ],
        )

    def test_exact_rrf_rejects_weakened_guarantee(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [],
                        "guarantee": {"orderedTopKExact": False},
                        "execution": {},
                    }
                },
            )

        with (
            QueryClient("http://test", "c", transport=httpx.MockTransport(handler)) as client,
            self.assertRaisesRegex(RuntimeError, "guarantee mismatch"),
        ):
            client.exact_rrf(
                self.query,
                dense_name="dense",
                sparse_name="sparse",
                k=60,
                weights=(1.0, 1.0),
                limit=20,
            )

    def test_exhaustive_rrf_requires_the_internal_plan_and_drained_sources(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                request.url.path,
                "/internal/collections/c/points/query/exact-rrf-exhaustive",
            )
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [{"id": 1, "rank": 1, "version": 1}],
                        "guarantee": {
                            "scope": "selected-local-shards-frozen-segment-view",
                            "orderedTopKExact": True,
                            "tieBreak": "point-identity-ascending",
                            "channelInput": "exact-channel-rank-streams",
                        },
                        "execution": {
                            "plan": "exact-rank-session-exhaustive-benchmark-v1",
                            "stopReason": "all-sources-exhausted",
                            "sourceExhausted": [True, True],
                        },
                    }
                },
            )

        with QueryClient("http://test", "c", transport=httpx.MockTransport(handler)) as client:
            result = client.exhaustive_rrf(
                self.query,
                dense_name="dense",
                sparse_name="sparse",
                k=60,
                weights=(1.0, 1.0),
                limit=20,
            )
        self.assertEqual(result.point_ids, (1,))


if __name__ == "__main__":
    unittest.main()
