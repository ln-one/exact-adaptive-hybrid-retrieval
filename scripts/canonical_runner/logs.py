"""Atomic immutable JSONL output for canonical experiment records."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class AtomicJsonlWriter:
    def __init__(self, destination: Path) -> None:
        if destination.exists():
            raise FileExistsError(f"canonical raw log already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination = destination
        self.temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        self._handle = self.temporary.open("x", encoding="utf-8")
        self._committed = False

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
        self._handle.write("\n")
        self._handle.flush()

    def commit(self) -> None:
        self.commit_as(self.destination)

    def commit_as(self, destination: Path) -> None:
        if destination.exists():
            raise FileExistsError(f"canonical raw log already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        os.link(self.temporary, destination)
        self.temporary.unlink()
        self._committed = True

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        if not self._committed:
            self.temporary.unlink(missing_ok=True)

    def __enter__(self) -> AtomicJsonlWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
