from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_public_paper_results import verify_package  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicPaperResultsTests(unittest.TestCase):
    def make_package(self, root: Path) -> None:
        derived = root / "derived"
        derived.mkdir(parents=True)
        csv_path = derived / "sample.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["value", "source_sha256"])
            writer.writeheader()
            writer.writerow({"value": "1", "source_sha256": "a" * 64})

        evidence = {
            "schema": "eahr-public-source-evidence-v1",
            "files": [
                {
                    "family": "test",
                    "relative_path": "aggregates/test.json",
                    "sha256": "a" * 64,
                }
            ],
        }
        evidence_path = root / "source-evidence.json"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        manifest = {
            "schema": "eahr-public-paper-results-v1",
            "source_evidence": {
                "path": "source-evidence.json",
                "sha256": digest(evidence_path),
                "files": 1,
            },
            "derived_files": [
                {
                    "path": "derived/sample.csv",
                    "sha256": digest(csv_path),
                    "bytes": csv_path.stat().st_size,
                    "rows": 1,
                    "columns": ["value", "source_sha256"],
                    "source_sha256": ["a" * 64],
                }
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_valid_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_package(root)
            self.assertEqual(verify_package(root), [])

    def test_absolute_path_leak_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_package(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["local_path"] = "/private/example"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("unsafe paths" in error for error in verify_package(root)))


if __name__ == "__main__":
    unittest.main()
