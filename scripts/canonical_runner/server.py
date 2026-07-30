"""Managed Qdrant process for publication-grade canonical experiments."""

from __future__ import annotations

import hashlib
import os
import signal
import socket
import subprocess
import tempfile
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import httpx


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reserve_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(("127.0.0.1", 0))
            sockets.append(server)
        return [int(server.getsockname()[1]) for server in sockets]
    finally:
        for server in sockets:
            server.close()


@dataclass(frozen=True)
class ManagedServerEvidence:
    url: str
    binary_sha256: str
    snapshot_sha256: str | None
    process_id: int


class ManagedQdrant(AbstractContextManager[ManagedServerEvidence]):
    """Start one exact binary in isolated storage and stop it deterministically."""

    def __init__(
        self,
        *,
        binary: Path,
        system_repo: Path,
        collection: str,
        snapshot: Path | None,
        startup_timeout_seconds: float = 900.0,
    ) -> None:
        self.binary = binary.resolve(strict=True)
        self.system_repo = system_repo.resolve(strict=True)
        self.collection = collection
        self.snapshot = snapshot.resolve(strict=True) if snapshot is not None else None
        self.startup_timeout_seconds = startup_timeout_seconds
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._log = None

    def __enter__(self) -> ManagedServerEvidence:
        binary_sha256 = sha256_file(self.binary)
        snapshot_sha256 = sha256_file(self.snapshot) if self.snapshot is not None else None
        http_port, grpc_port = _reserve_ports(2)
        self._temporary = tempfile.TemporaryDirectory(prefix="stratumind-canonical-qdrant-")
        root = Path(self._temporary.name)
        log_path = root / "qdrant.log"
        self._log = log_path.open("xb")

        environment = os.environ.copy()
        environment.update(
            {
                "QDRANT__STORAGE__STORAGE_PATH": str(root / "storage"),
                "QDRANT__STORAGE__SNAPSHOTS_PATH": str(root / "snapshots"),
                "QDRANT__SERVICE__HOST": "127.0.0.1",
                "QDRANT__SERVICE__HTTP_PORT": str(http_port),
                "QDRANT__SERVICE__GRPC_PORT": str(grpc_port),
                "QDRANT__TELEMETRY_DISABLED": "true",
            }
        )
        command = [str(self.binary), "--disable-telemetry"]
        if self.snapshot is not None:
            command.extend(["--snapshot", f"{self.snapshot}:{self.collection}"])
        self._process = subprocess.Popen(
            command,
            cwd=self.system_repo,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        url = f"http://127.0.0.1:{http_port}"
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                self._log.flush()
                message = log_path.read_text(encoding="utf-8", errors="replace")[-8_000:]
                self.__exit__(None, None, None)
                raise RuntimeError(
                    f"managed Qdrant exited during startup ({return_code}):\n{message}"
                )
            try:
                response = httpx.get(f"{url}/", timeout=1.0)
                if response.is_success:
                    return ManagedServerEvidence(
                        url=url,
                        binary_sha256=binary_sha256,
                        snapshot_sha256=snapshot_sha256,
                        process_id=self._process.pid,
                    )
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        self.__exit__(None, None, None)
        raise TimeoutError("managed Qdrant did not become ready before the startup deadline")

    def __exit__(self, *_: object) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if self._log is not None:
            self._log.close()
        if self._temporary is not None:
            self._temporary.cleanup()
        self._process = None
        self._log = None
        self._temporary = None
