from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
import unittest
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_e3 import aggregate as aggregate_e3
from build_qdrant_snapshot import _collection_is_ready, _expected_indexed_vectors
from canonical_runner.artifacts import (
    CANONICAL_COLLECTION_FORMAT,
    DatasetSnapshot,
    QueryInput,
    load_collection_snapshot,
)
from canonical_runner.client import QueryClient
from canonical_runner.e2 import E2Config, run_e2
from canonical_runner.e3 import E3Config, run_e3
from canonical_runner.e4 import E4Config, run_e4
from canonical_runner.e5 import E5Config, run_e5
from canonical_runner.fusion import exact_wrrf, position_score
from canonical_runner.logs import AtomicJsonlWriter
from canonical_runner.provenance import canonical_hash, verify_hardware_manifest
from canonical_runner.runner import E1Config, run_e1
from canonical_runner.server import sha256_file
from canonical_runner.synthetic import REGIMES, BalancedCertificate, generate_rankings
from canonical_runner.validation import validate_log
from create_qdrant_collection import collection_schema
from load_qdrant_sparse import valid_sparse_vector


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
        self.assertEqual(
            schema["quantization_config"],
            {
                "scalar": {
                    "type": "int8",
                    "quantile": 0.99,
                    "always_ram": True,
                }
            },
        )
        self.assertEqual(schema["optimizers_config"], {"indexing_threshold": 1})
        self.assertEqual(schema["shard_number"], 2)

    def test_empty_sparse_document_is_a_valid_zero_support_vector(self) -> None:
        self.assertTrue(valid_sparse_vector([], []))

    def test_indexed_vector_count_includes_each_present_named_vector(self) -> None:
        self.assertEqual(
            _expected_indexed_vectors(
                {"points": 3},
                {"points": 2, "empty": 1},
            ),
            5,
        )

    def test_collection_readiness_uses_exact_points_and_a_physical_lower_bound(self) -> None:
        info = {
            "status": "green",
            # This field is a Segment-level progress estimate and deliberately
            # disagrees with the exact count during a merge.
            "points_count": 97,
            "indexed_vectors_count": 201,
        }
        self.assertTrue(
            _collection_is_ready(
                info,
                exact_points=100,
                expected_points=100,
                minimum_indexed_vectors=200,
            )
        )
        self.assertFalse(
            _collection_is_ready(
                info,
                exact_points=99,
                expected_points=100,
                minimum_indexed_vectors=200,
            )
        )
        self.assertFalse(
            _collection_is_ready(
                {**info, "indexed_vectors_count": 199},
                exact_points=100,
                expected_points=100,
                minimum_indexed_vectors=200,
            )
        )

    def test_collection_snapshot_can_reuse_a_declared_document_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = "trec-dl-2020"
            shared = "trec-dl-2019"
            source = root / "datasets" / dataset / "source"
            source.mkdir(parents=True)
            (source / "manifest.json").write_text(
                json.dumps({"documents_shared_from": shared}),
                encoding="utf-8",
            )
            collection_root = root / "collections" / shared / CANONICAL_COLLECTION_FORMAT
            collection_root.mkdir(parents=True)
            snapshot_path = collection_root / "snapshot.snapshot"
            snapshot_path.write_bytes(b"snapshot")
            snapshot_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
            (collection_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "canonical-qdrant-collection-snapshot-v1",
                        "dataset": shared,
                        "collection": "canonical",
                        "points": 10,
                        "collectionConfigSha256": "config-sha256",
                        "snapshot": {
                            "path": snapshot_path.name,
                            "sha256": snapshot_sha256,
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = load_collection_snapshot(
                root,
                DatasetSnapshot(
                    dataset=dataset,
                    split="judged",
                    document_count=10,
                    queries=(),
                    source_manifest_sha256="source",
                    dense_manifest_sha256="dense",
                    sparse_manifest_sha256="sparse",
                ),
                "canonical",
            )
            self.assertEqual(result.path, snapshot_path)


class ProvenanceTests(unittest.TestCase):
    def test_hardware_manifest_is_bound_without_machine_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hardware.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "canonical-hardware-v1",
                        "hardwareProfile": "test-machine",
                        "architecture": platform.machine(),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                verify_hardware_manifest(path, hardware_profile="test-machine"),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_hardware_manifest_rejects_machine_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hardware.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "canonical-hardware-v1",
                        "hardwareProfile": "test-machine",
                        "architecture": platform.machine(),
                        "serialNumber": "not-publication-safe",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "machine identifier"):
                verify_hardware_manifest(path, hardware_profile="test-machine")


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

    def test_e2_rejects_unknown_baseline_before_repository_access(self) -> None:
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
            baseline="unknown",
        )
        with self.assertRaisesRegex(ValueError, "baseline"):
            run_e2(config)

    def test_ms_marco_launcher_hard_codes_native_bulk(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1] / "scripts" / "run_ms_marco_scale_final.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--baseline native-bulk-exhaustive", launcher)
        self.assertNotIn("--baseline same-producer-exhaustive", launcher)

    def test_e2_rejects_unbound_external_server_for_publication(self) -> None:
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
        )
        with self.assertRaisesRegex(RuntimeError, "managed --system-binary"):
            run_e2(config)

    def test_e2_requires_a_hardware_manifest_for_publication(self) -> None:
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
            system_binary=Path("/unused/qdrant"),
            system_build_manifest=Path("/unused/build.json"),
        )
        with self.assertRaisesRegex(RuntimeError, "hardware-manifest"):
            run_e2(config)

    def test_e3_rejects_unsorted_or_duplicate_depths(self) -> None:
        config = E3Config(
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
            depths=(20, 20),
        )
        with self.assertRaisesRegex(ValueError, "unique and strictly increasing"):
            run_e3(config)

    def test_e4_rejects_duplicate_seeds(self) -> None:
        config = E4Config(
            output=Path("/unused"),
            bench_repo=Path("/unused"),
            hardware_profile="test",
            sizes=(100,),
            seeds=(7, 7),
        )
        with self.assertRaisesRegex(ValueError, "seeds"):
            run_e4(config)

    def test_e5_rejects_invalid_process_start_count(self) -> None:
        config = E5Config(
            artifact_root=Path("/unused"),
            dataset="unused",
            collection="unused",
            output=Path("/unused"),
            bench_repo=Path("/unused"),
            system_repo=Path("/unused"),
            system_commit="unused",
            system_artifact="sha256:test",
            hardware_profile="test",
            system_binary=Path("/unused"),
            system_build_manifest=Path("/unused"),
            process_starts=0,
        )
        with self.assertRaisesRegex(ValueError, "process starts"):
            run_e5(config)


class ManagedServerTests(unittest.TestCase):
    def test_sha256_file_streams_the_exact_binary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binary"
            path.write_bytes(b"canonical-qdrant")
            self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())


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

    def test_e3_aggregation_treats_fixed_depth_mismatch_as_a_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e3.jsonl"
            run = {
                "recordType": "run",
                "schema": "ed-wrrf-results-v1",
                "runId": "r1",
                "experiment": "E3",
                "dataset": "test",
                "dirty": False,
                "systemArtifact": f"sha256:{'a' * 64}",
                "serverProvenance": {
                    "mode": "managed-isolated-snapshot",
                    "binarySha256": "a" * 64,
                    "snapshotSha256": "b" * 64,
                    "collectionSnapshotManifestSha256": "c" * 64,
                    "systemBuildManifestSha256": "d" * 64,
                },
                "parameters": {
                    "queryLimit": None,
                    "depths": [20, 50],
                    "limit": 1,
                    "rrfK": 60,
                    "weights": [1.0, 1.0],
                },
            }
            observations = [
                {
                    "recordType": "query",
                    "runId": "r1",
                    "queryId": "q1",
                    "sequence": 0,
                    "depth": 20,
                    "status": "mismatch",
                    "orderedIds": [2],
                    "oracleOrderedIds": [1],
                    "orderedResultSha256": canonical_hash([2]),
                    "oracleOrderedResultSha256": canonical_hash([1]),
                    "membershipMismatch": True,
                    "orderMismatch": False,
                    "candidateUnionContainsOracle": False,
                    "oracleRecall": 0.0,
                    "exactPrefixLength": 0,
                    "candidateUnionSize": 2,
                    "exposedRanks": 40,
                },
                {
                    "recordType": "query",
                    "runId": "r1",
                    "queryId": "q1",
                    "sequence": 1,
                    "depth": 50,
                    "status": "ok",
                    "orderedIds": [1],
                    "oracleOrderedIds": [1],
                    "orderedResultSha256": canonical_hash([1]),
                    "oracleOrderedResultSha256": canonical_hash([1]),
                    "membershipMismatch": False,
                    "orderMismatch": False,
                    "candidateUnionContainsOracle": True,
                    "oracleRecall": 1.0,
                    "exactPrefixLength": 1,
                    "candidateUnionSize": 3,
                    "exposedRanks": 100,
                },
            ]
            summary = {
                "recordType": "summary",
                "runId": "r1",
                "attemptedQueries": 2,
                "uniqueQueries": 1,
                "okQueries": 1,
                "mismatchQueries": 1,
                "timeoutQueries": 0,
                "errorQueries": 0,
                "queryRecordSha256": canonical_hash(
                    [canonical_hash(record) for record in observations]
                ),
            }
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in [run, *observations, summary]),
                encoding="utf-8",
            )
            path.with_suffix(".jsonl.sha256").write_text(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n",
                encoding="utf-8",
            )
            result = aggregate_e3(path)
            self.assertEqual(result["frontier"][0]["orderedExact"]["estimate"], 0.0)
            self.assertEqual(result["frontier"][1]["orderedExact"]["estimate"], 1.0)


class SyntheticTests(unittest.TestCase):
    def test_regimes_are_permutations_and_have_declared_overlap(self) -> None:
        for regime in REGIMES:
            with self.subTest(regime=regime):
                rankings = generate_rankings(size=2_000, seed=1_729, regime=regime)
                self.assertEqual(len(set(rankings.first.tolist())), 2_000)
                self.assertEqual(len(set(rankings.second.tolist())), 2_000)
                if regime == "partial-overlap":
                    self.assertEqual(rankings.top_window_overlap, 0.5)
                if regime == "all-tied":
                    self.assertEqual(rankings.top_window_overlap, 1.0)

    def test_balanced_certificate_is_monotone_and_matches_exhaustive_order(self) -> None:
        for regime in REGIMES:
            with self.subTest(regime=regime):
                rankings = generate_rankings(size=1_024, seed=2_027, regime=regime)
                certificate = BalancedCertificate(
                    rankings,
                    k=60,
                    weights=(1.0, 1.0),
                    limit=20,
                    batch_size=16,
                )
                first_certified: int | None = None
                for depth in range(16, 1_025, 16):
                    certified, output, _, _ = certificate._at_depth(depth)
                    if certified:
                        first_certified = first_certified or depth
                        self.assertEqual(output, certificate.oracle)
                    elif first_certified is not None:
                        self.fail("certificate ceased to hold after a certified depth")
                result = certificate.find_minimum_batch_depth()
                self.assertEqual(result.depth, first_certified)
                self.assertEqual(result.output, certificate.oracle)


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

    def test_exact_channel_prefix_expands_until_boundary_tie_is_complete(self) -> None:
        scored = [
            {"id": 5, "score": 0.9},
            {"id": 4, "score": 0.5},
            {"id": 3, "score": 0.5},
            {"id": 2, "score": 0.5},
            {"id": 1, "score": 0.4},
        ]
        requested_limits: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            requested_limits.append(body["limit"])
            return httpx.Response(
                200,
                json={"result": {"points": scored[: body["limit"]]}},
            )

        with QueryClient("http://test", "c", transport=httpx.MockTransport(handler)) as client:
            result = client.exact_channel_prefix(
                self.query,
                channel="dense",
                limit=2,
                corpus_points=5,
                dense_name="dense",
                sparse_name="sparse",
            )
        self.assertEqual(requested_limits, [3, 5])
        self.assertEqual(result.point_ids, (5, 2))
        self.assertEqual(result.request_count, 2)
        self.assertTrue(result.exhausted)

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
                        "points": [{"id": 1, "rank": 1, "score": 0.1, "version": 1}],
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

    def test_e5_client_rejects_a_silent_dense_fallback(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                request.url.path,
                "/internal/collections/c/points/query/exact-rrf-producer",
            )
            self.assertEqual(request.url.params["producer"], "pvs-pbm")
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [{"id": 1, "rank": 1, "score": 0.1, "version": 1}],
                        "guarantee": {
                            "scope": "selected-local-shards-frozen-segment-view",
                            "orderedTopKExact": True,
                            "tieBreak": "point-identity-ascending",
                            "channelInput": "exact-channel-rank-streams",
                        },
                        "execution": {
                            "plan": "canonical-e5-pvs-pbm",
                            "exhaustiveFallback": False,
                            "producer": {
                                "densePvsSegments": 0,
                                "denseScalarSegments": 1,
                                "denseScanSegments": 0,
                                "sparsePbmSegments": 1,
                                "sparseMaterializedSegments": 0,
                            },
                        },
                    }
                },
            )

        with (
            QueryClient("http://test", "c", transport=httpx.MockTransport(handler)) as client,
            self.assertRaisesRegex(RuntimeError, "forced Dense producer"),
        ):
            client.producer_rrf(
                self.query,
                producer="pvs-pbm",
                dense_name="dense",
                sparse_name="sparse",
                k=60,
                weights=(1.0, 1.0),
                limit=20,
            )

    def test_bulk_exhaustion_requires_native_plan_and_drained_sources(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                request.url.path,
                "/internal/collections/c/points/query/exact-rrf-producer",
            )
            self.assertEqual(request.url.params["producer"], "pvs-pbm")
            self.assertEqual(request.url.params["mode"], "native-bulk-exhaustive")
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [{"id": 1, "rank": 1, "score": 0.1, "version": 1}],
                        "guarantee": {
                            "scope": "selected-local-shards-frozen-segment-view",
                            "orderedTopKExact": True,
                            "tieBreak": "point-identity-ascending",
                            "channelInput": "exact-channel-rank-streams",
                        },
                        "execution": {
                            "plan": "canonical-e2-bulk-native-exhaustive",
                            "stopReason": "all-sources-exhausted",
                            "sourceExhausted": [True, True],
                            "exhaustiveFallback": True,
                            "producer": {
                                "densePvsSegments": 0,
                                "denseScalarSegments": 0,
                                "denseScanSegments": 1,
                                "sparsePbmSegments": 0,
                                "sparseMaterializedSegments": 1,
                            },
                        },
                    }
                },
            )

        with QueryClient("http://test", "c", transport=httpx.MockTransport(handler)) as client:
            result = client.producer_rrf(
                self.query,
                producer="pvs-pbm",
                exhaustive=True,
                dense_name="dense",
                sparse_name="sparse",
                k=60,
                weights=(1.0, 1.0),
                limit=20,
            )
        self.assertEqual(result.point_ids, (1,))

    def test_same_producer_exhaustion_uses_the_explicit_mode(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.params["mode"], "same-producer-exhaustive")
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [{"id": 1, "rank": 1, "score": 0.1, "version": 1}],
                        "guarantee": {
                            "scope": "selected-local-shards-frozen-segment-view",
                            "orderedTopKExact": True,
                            "tieBreak": "point-identity-ascending",
                            "channelInput": "exact-channel-rank-streams",
                        },
                        "execution": {
                            "plan": "canonical-e2-pvs-pbm-exhaustive",
                            "stopReason": "all-sources-exhausted",
                            "sourceExhausted": [True, True],
                            "exhaustiveFallback": False,
                            "producer": {
                                "densePvsSegments": 1,
                                "denseScalarSegments": 0,
                                "denseScanSegments": 0,
                                "sparsePbmSegments": 1,
                                "sparseMaterializedSegments": 0,
                            },
                        },
                    }
                },
            )

        with QueryClient("http://test", "c", transport=httpx.MockTransport(handler)) as client:
            result = client.producer_rrf(
                self.query,
                producer="pvs-pbm",
                mode="same-producer-exhaustive",
                dense_name="dense",
                sparse_name="sparse",
                k=60,
                weights=(1.0, 1.0),
                limit=20,
            )
        self.assertEqual(result.point_ids, (1,))


if __name__ == "__main__":
    unittest.main()
