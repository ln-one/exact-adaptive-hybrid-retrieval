from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import httpx
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_target_validity import (  # noqa: E402
    MEASURE_LABEL,
    RunData,
    fold_id,
    nested_chronological_bootstrap,
    select_l,
)
from canonical_runner.artifacts import QueryInput  # noqa: E402
from canonical_runner.client import QueryClient  # noqa: E402
from canonical_runner.fusion import exact_wrrf  # noqa: E402
from canonical_runner.provenance import canonical_hash  # noqa: E402
from canonical_runner.target_validity import _verify_query_checkpoint  # noqa: E402
from prepare_trec_covid_chronological import read_valid_ids, write_documents  # noqa: E402


class TargetValidityClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query = QueryInput(
            query_id="q1",
            dense=[1.0, 0.0],
            sparse_indices=[3],
            sparse_values=[1.0],
        )

    def test_sparse_prefix_ends_at_strictly_positive_support(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [
                            {"id": 1, "score": 0.5},
                            {"id": 2, "score": 0.0},
                            {"id": 3, "score": 0.0},
                        ]
                    }
                },
            )

        with QueryClient("http://test", "c", transport=httpx.MockTransport(handler)) as client:
            result = client.exact_channel_prefix(
                self.query,
                channel="sparse",
                limit=2,
                corpus_points=3,
                dense_name="dense",
                sparse_name="sparse",
            )
        self.assertEqual(result.point_ids, (1,))
        self.assertTrue(result.exhausted)

    def test_empty_sparse_support_is_normal_exhaustion(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"result": {"points": []}})

        with QueryClient("http://test", "c", transport=httpx.MockTransport(handler)) as client:
            result = client.exact_channel_prefix(
                self.query,
                channel="sparse",
                limit=100,
                corpus_points=1_000,
                dense_name="dense",
                sparse_name="sparse",
            )
        self.assertEqual(result.point_ids, ())
        self.assertTrue(result.exhausted)

    def test_external_id_lookup_is_batched_and_complete(self) -> None:
        requested: list[list[int]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            requested.append(body["ids"])
            return httpx.Response(
                200,
                json={
                    "result": [
                        {"id": point_id, "payload": {"external_id": f"d{point_id}"}}
                        for point_id in body["ids"]
                    ]
                },
            )

        with QueryClient("http://test", "c", transport=httpx.MockTransport(handler)) as client:
            result = client.external_ids([1, 2, 1, 3], batch_size=2)
        self.assertEqual(requested, [[1, 2], [3]])
        self.assertEqual(result, {1: "d1", 2: "d2", 3: "d3"})

    def test_one_hot_wrrf_exposes_a_certified_sparse_prefix(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.assertEqual(body["exact_rrf"]["weights"], [0.0, 1.0])
            self.assertEqual(body["limit"], 3)
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [
                            {"id": point_id, "rank": rank, "version": 1}
                            for rank, point_id in enumerate((7, 2, 9), start=1)
                        ],
                        "guarantee": {
                            "scope": "selected-local-shards-frozen-segment-view",
                            "orderedTopKExact": True,
                            "tieBreak": "point-identity-ascending",
                            "channelInput": "exact-channel-rank-streams",
                        },
                        "execution": {
                            "plan": "exact-rank-session-v1",
                            "sourceExhausted": [False, True],
                            "sourcePulls": [0, 3],
                        },
                    }
                },
            )

        with QueryClient("http://test", "c", transport=httpx.MockTransport(handler)) as client:
            result = client.certified_channel_prefix(
                self.query,
                channel="sparse",
                limit=3,
                dense_name="dense",
                sparse_name="sparse",
                k=60,
            )
        self.assertEqual(result.point_ids, (7, 2, 9))
        self.assertEqual(result.fetched_points, 3)
        self.assertEqual(result.request_count, 1)
        self.assertTrue(result.exhausted)


class TargetValidityProtocolTests(unittest.TestCase):
    def test_chronological_docid_rows_are_audited_before_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "docids.txt"
            path.write_text("d1\nd2\nd1\n", encoding="utf-8")
            values, evidence = read_valid_ids(
                path,
                expected_rows=3,
                expected_unique=2,
            )
            self.assertEqual(values, {"d1", "d2"})
            self.assertEqual(evidence, {"rows": 3, "unique": 2, "duplicate_rows": 1})
            with self.assertRaisesRegex(RuntimeError, "rows=3 unique=2"):
                read_valid_ids(path, expected_rows=3, expected_unique=3)

    def test_metadata_duplicates_use_audited_last_occurrence_upsert(self) -> None:
        header = "cord_uid,title,abstract\n"
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            identical = base / "identical.csv"
            identical.write_text(header + "d1,Title,Text\nd1,Title,Text\n", encoding="utf-8")
            identities, evidence = write_documents(
                identical,
                base / "identical.parquet",
                batch_rows=10,
            )
            self.assertEqual(identities, {"d1"})
            self.assertEqual(evidence["duplicate_rows"], 1)
            self.assertEqual(evidence["conflicting_duplicate_identities"], 0)

            conflict = base / "conflict.csv"
            conflict.write_text(header + "d1,Title,Text\nd1,Other,Text\n", encoding="utf-8")
            identities, evidence = write_documents(
                conflict,
                base / "conflict.parquet",
                batch_rows=10,
            )
            self.assertEqual(identities, {"d1"})
            self.assertEqual(evidence["conflicting_duplicate_identities"], 1)
            values = pq.read_table(base / "conflict.parquet").to_pylist()
            self.assertEqual(values, [{"id": "d1", "text": "Other\nText"}])

    def test_empty_sparse_support_reduces_to_dense_order(self) -> None:
        self.assertEqual(
            exact_wrrf([[3, 1, 2], []], k=60, weights=[1.0, 1.0], limit=3),
            [3, 1, 2],
        )

    def test_chronological_folds_are_disjoint_and_balanced(self) -> None:
        folds = {
            fold: {str(query_id) for query_id in range(1, 31) if fold_id(str(query_id)) == fold}
            for fold in range(5)
        }
        self.assertTrue(all(len(values) == 6 for values in folds.values()))
        self.assertEqual(set().union(*folds.values()), {str(value) for value in range(1, 31)})
        self.assertEqual(sum(len(values) for values in folds.values()), 30)

    def test_fixed_depth_selection_uses_smaller_depth_for_an_exact_tie(self) -> None:
        metrics = {
            "fixed-L10": {"1": {"nDCG@10": 0.5}, "2": {"nDCG@10": 0.7}},
            "fixed-L100": {"1": {"nDCG@10": 0.5}, "2": {"nDCG@10": 0.7}},
        }
        self.assertEqual(select_l(metrics, ["1", "2"]), "fixed-L10")

    def test_nested_bootstrap_reselects_depth_and_is_deterministic(self) -> None:
        query_ids = tuple(str(value) for value in range(1, 31))
        rounds = {
            round_id: RunData(Path(str(round_id)), f"r{round_id}", query_ids, {}, ())
            for round_id in range(1, 6)
        }
        metrics = {}
        for round_id in rounds:
            metrics[round_id] = {}
            for method, offset in (
                ("fixed-L10", -0.02),
                ("fixed-L100", -0.01),
                ("full-wrrf", 0.0),
                ("dense", -0.03),
                ("sparse", -0.04),
            ):
                metrics[round_id][method] = {
                    query_id: {
                        metric_name: 0.7 + offset + int(query_id) / 10_000
                        for metric_name in MEASURE_LABEL.values()
                    }
                    for query_id in query_ids
                }
        first = nested_chronological_bootstrap(rounds, metrics, replicates=50, seed=17)
        second = nested_chronological_bootstrap(rounds, metrics, replicates=50, seed=17)
        self.assertEqual(first, second)
        _, frequencies = first
        self.assertTrue(all(sum(counts.values()) == 50 for counts in frequencies.values()))
        self.assertTrue(all(counts == {"fixed-L100": 50} for counts in frequencies.values()))

    def test_query_checkpoint_hash_is_verified_before_resume(self) -> None:
        record = {
            "queryId": "q1",
            "runConfigSha256": "run",
            "status": "ok",
            "methods": {},
        }
        record["recordSha256"] = canonical_hash(record)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "query.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            loaded = _verify_query_checkpoint(
                path,
                query_id="q1",
                run_config_sha256="run",
            )
            self.assertEqual(loaded["recordSha256"], record["recordSha256"])
            mutated = json.loads(path.read_text(encoding="utf-8"))
            mutated["methods"] = {"full-wrrf": ["d1"]}
            path.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                _verify_query_checkpoint(
                    path,
                    query_id="q1",
                    run_config_sha256="run",
                )


if __name__ == "__main__":
    unittest.main()
