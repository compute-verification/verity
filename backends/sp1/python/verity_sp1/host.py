"""Drive the ``veritor-zk-host`` binary: one subprocess per command, one JSON object on its stdout.

The Rust workspace beside this package holds the guest (the zkVM program), the shared kernel (``common``) and the
host CLI.  This module speaks the host's protocol (``host/src/main.rs``) and knows nothing about what a statement
means; the statement and witness are opaque bytes here.  Three commands matter:

* ``execute``: run the guest in the SP1 executor without proving; the verdict, the public values and the exact
  cycle count per phase (``io``, ``parse``, ``digest``, ``merkle``, ``gates``).  Sub-second; what tests and the cost
  table use.
* ``prove``: one core or compressed proof of one statement, written to a file, verified once by the host.
* ``verify``: check a saved proof under the guest's verifying key, and its committed statement digest against
  bytes the caller supplies.

The guest commits ``h_O || h_L || verdict`` (65 bytes) when given ``--typed-object-digest h_O``, else
``h_L || verdict`` (33 bytes), with ``h_L = sha256(statement bytes)``.  Which object those bytes bind, and whether
a proof is *accepted*, is :mod:`verity_sp1.typed`'s business; this module only reports what the host said.

``APPROVED`` is the identity of this repository's guest (ELF digest and verifying-key hash as ``info`` prints them);
a rebuilt guest that prints another identity is a different prover and the tests say so.  A guest built with an
``unsound-*`` diagnostic feature commits a verdict byte no verifier accepts; the host reports it as ``unsound``.

Promoted from veritor ``protocol/proofs/sp1.py`` (eb865ce2), reduced to the bytes-level protocol: the veritor
class encoded statements and witnesses itself and did not carry the typed-object digest.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from verity.errors import ProtocolError

__all__ = [
    "APPROVED",
    "BINARY_ENV",
    "HOST_BINARY",
    "WORKSPACE",
    "ExecutionReport",
    "GuestIdentity",
    "Host",
    "ProofReport",
    "SP1Unavailable",
    "VerifyReport",
    "toolchain_env",
]

WORKSPACE = Path(__file__).resolve().parents[2]
HOST_BINARY = "veritor-zk-host"

type ProveMode = Literal["core", "compressed", "groth16", "plonk"]


class SP1Unavailable(ProtocolError):
    """The host binary is not built or the SP1 toolchain is not installed."""


@dataclass(frozen=True, slots=True)
class GuestIdentity:
    """What ``info`` prints: the embedded guest ELF's SHA-256 and its verifying-key hash."""

    elf_sha256: str
    vk_hash: str
    sp1_version: str
    guest_features: tuple[str, ...] = ()

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> GuestIdentity:
        return cls(
            elf_sha256=str(document["elf_sha256"]),
            vk_hash=str(document["vk_hash"]),
            sp1_version=str(document["sp1_version"]),
            guest_features=tuple(str(f) for f in document.get("guest_features", ())),
        )


#: This repository's sound guest: ``cargo build --release -p veritor-zk-host`` at the pinned ``sp1-sdk`` 6.4.0 with
#: the reproducible ``host/build.rs`` (paths remapped, ``--locked``), built on 2026-09-21 on a Linux pod from two
#: unrelated paths with identical results.  The veritor guest at eb865ce2 had another ELF (the trimmed
#: ``guest/src/main.rs`` moves four panic locations), so this identity is verity's own; see ``backends/sp1/README.md``.
APPROVED = GuestIdentity(
    elf_sha256="cef2b78aef7ae16a3703d972c4cea7ec1cb5fed31797736a77886dc01fa5c567",
    vk_hash="0x007d9347080c44a23742498ce3b1fb7ca0df111119f60ac2e094c5cd9c2438ab",
    sp1_version="6.4.0",
)


def toolchain_env() -> dict[str, str]:
    """The environment the host runs in: ``~/.sp1/bin`` and ``~/.cargo/bin`` ahead of ``PATH``."""

    home = Path.home()
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(home / ".sp1" / "bin"), str(home / ".cargo" / "bin"), env.get("PATH", "")])
    return env


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """What ``execute`` (or the execution inside ``prove``) measured for one statement."""

    verdict: bool
    public_values: bytes
    statement_digest: bytes
    total_cycles: int
    cycle_tracker: Mapping[str, int]
    syscalls: Mapping[str, int]
    gas: int | None
    unsound: bool
    document: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ProofReport:
    """What ``prove`` produced besides the proof file: the execution, the mode, sizes and timings."""

    execution: ExecutionReport
    mode: str
    proof_path: Path
    proof_bytes: int
    shards: int
    vk_hash: str
    setup_seconds: float
    prove_seconds: float
    verify_seconds: float
    document: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifyReport:
    """What ``verify`` said about a saved proof: each fact apart, none of them an acceptance."""

    ok: bool
    public_values: bytes
    statement_digest: bytes
    verdict: bool
    unsound: bool
    vk_hash: str
    statement_match: bool | None
    document: Mapping[str, Any]


#: Names an installed host binary outside the workspace (a pod's ``/workspace/bin/veritor-zk-host-cuda``, say).
BINARY_ENV = "VERITY_SP1_HOST"


@dataclass
class Host:
    """The host binary.  ``binary`` defaults to ``$VERITY_SP1_HOST`` if set, else
    ``<workspace>/target/release/veritor-zk-host``; nothing is built implicitly (building needs the SP1 toolchain
    and ~10 minutes).  ``timeout`` bounds every command."""

    binary: Path | None = None
    timeout: float | None = None

    def path(self) -> Path:
        if self.binary is not None:
            candidate = Path(self.binary)
        elif os.environ.get(BINARY_ENV):
            candidate = Path(os.environ[BINARY_ENV])
        else:
            candidate = WORKSPACE / "target" / "release" / HOST_BINARY
        if not candidate.exists():
            raise SP1Unavailable(
                f"{candidate} is not built; run `cargo build --release -p {HOST_BINARY}` in {WORKSPACE} with the SP1 toolchain"
                f" (or point {BINARY_ENV} at an installed host)"
            )
        return candidate

    @staticmethod
    def toolchain_available() -> bool:
        """Whether ``cargo prove --version`` runs (needed to *build* the guest, not to run the host)."""

        env = toolchain_env()
        cargo = shutil.which("cargo", path=env["PATH"])
        if cargo is None:
            return False
        try:
            return subprocess.run([cargo, "prove", "--version"], capture_output=True, env=env, timeout=60, check=False).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _run(self, *args: str) -> dict[str, Any]:
        command = [str(self.path()), *args]
        try:
            result = subprocess.run(command, capture_output=True, text=True, env=toolchain_env(), timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise SP1Unavailable(f"{HOST_BINARY} {args[0]} timed out after {self.timeout}s") from error
        if result.returncode != 0:
            raise ProtocolError(f"{HOST_BINARY} {args[0]} failed ({result.returncode}):\n{result.stderr[-4000:]}")
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise ProtocolError(f"{HOST_BINARY} {args[0]} printed no result")
        try:
            document = json.loads(lines[-1])
        except json.JSONDecodeError as error:
            raise ProtocolError(f"{HOST_BINARY} {args[0]} printed no JSON result") from error
        if type(document) is not dict:
            raise ProtocolError(f"{HOST_BINARY} result is not an object")
        return document

    def info(self) -> GuestIdentity:
        """The embedded guest's identity (derives the verifying key: tens of CPU-seconds)."""

        return GuestIdentity.from_document(self._run("info"))

    @staticmethod
    def _counters(document: Mapping[str, Any], key: str) -> dict[str, int]:
        table = document.get(key, {})
        if not isinstance(table, Mapping):
            raise ProtocolError(f"{HOST_BINARY} {key} is not an object")
        return {str(name): int(value) for name, value in table.items()}

    @classmethod
    def _execution(cls, document: Mapping[str, Any]) -> ExecutionReport:
        gas = document.get("gas")
        return ExecutionReport(
            verdict=bool(document["verdict"]),
            public_values=bytes.fromhex(str(document["public_values"])),
            statement_digest=bytes.fromhex(str(document["statement_digest"])),
            total_cycles=int(document["total_cycles"]),
            cycle_tracker=cls._counters(document, "cycle_tracker"),
            syscalls=cls._counters(document, "syscalls"),
            gas=None if gas is None else int(gas),
            unsound=bool(document.get("unsound", False)),
            document=document,
        )

    @staticmethod
    def _typed(typed_object_digest: str | None) -> list[str]:
        if typed_object_digest is None:
            return []
        if len(typed_object_digest) != 64 or set(typed_object_digest) - set("0123456789abcdef"):
            raise ProtocolError("typed_object_digest must be 32 bytes of lowercase hex")
        return ["--typed-object-digest", typed_object_digest]

    def execute(self, statement: bytes, witness: bytes, *, typed_object_digest: str | None = None) -> ExecutionReport:
        """Run the guest without proving: verdict, public values and exact cycles."""

        with TemporaryDirectory(prefix="verity-sp1-") as directory:
            root = Path(directory)
            (root / "statement.bin").write_bytes(statement)
            (root / "witness.bin").write_bytes(witness)
            document = self._run(
                "execute", "--statement", str(root / "statement.bin"), "--witness", str(root / "witness.bin"),
                *self._typed(typed_object_digest),
            )
        return self._execution(document)

    def prove(
        self, statement: bytes, witness: bytes, out: Path, *, typed_object_digest: str | None = None, mode: ProveMode = "core"
    ) -> ProofReport:
        """One proof of one statement, written to ``out``.  The guest's verdict is whatever it is: a proof of a
        rejection is still a proof (of ``verdict = 0``); callers decide what they wanted proved."""

        out = Path(out)
        with TemporaryDirectory(prefix="verity-sp1-") as directory:
            root = Path(directory)
            (root / "statement.bin").write_bytes(statement)
            (root / "witness.bin").write_bytes(witness)
            document = self._run(
                "prove", "--statement", str(root / "statement.bin"), "--witness", str(root / "witness.bin"),
                *self._typed(typed_object_digest), "--out", str(out), "--mode", mode,
            )
        if not out.exists():
            raise ProtocolError(f"{HOST_BINARY} prove reported success but wrote no proof at {out}")
        return ProofReport(
            execution=self._execution(document),
            mode=str(document["mode"]),
            proof_path=out,
            proof_bytes=int(document["proof_bytes"]),
            shards=int(document["shards"]),
            vk_hash=str(document["vk_hash"]),
            setup_seconds=float(document["setup_seconds"]),
            prove_seconds=float(document["prove_seconds"]),
            verify_seconds=float(document["verify_seconds"]),
            document=document,
        )

    def verify(self, proof: Path, statement: bytes | None = None) -> VerifyReport:
        """Check a saved proof under the guest's verifying key; with ``statement``, also whether the committed
        statement digest is ``sha256(statement)``.  An unloadable proof file is a rejected proof, not an error."""

        with TemporaryDirectory(prefix="verity-sp1-") as directory:
            root = Path(directory)
            args = ["verify", "--proof", str(Path(proof))]
            if statement is not None:
                (root / "statement.bin").write_bytes(statement)
                args += ["--statement", str(root / "statement.bin")]
            try:
                document = self._run(*args)
            except ProtocolError as error:
                if isinstance(error, SP1Unavailable):
                    raise
                return VerifyReport(False, b"", b"", False, False, "", None, {"error": str(error)})
        match = document.get("statement_match")
        return VerifyReport(
            ok=bool(document.get("ok")),
            public_values=bytes.fromhex(str(document.get("public_values", ""))),
            statement_digest=bytes.fromhex(str(document.get("statement_digest", ""))),
            verdict=bool(document.get("verdict")),
            unsound=bool(document.get("unsound", False)),
            vk_hash=str(document.get("vk_hash", "")),
            statement_match=None if match is None else bool(match),
            document=document,
        )
