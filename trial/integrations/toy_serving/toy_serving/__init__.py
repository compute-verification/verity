"""A toy serving engine that plays the developer in `verity.protocol.epoch.run`, the way vLLM would.

What it owns is what only an engine can: where the hooks go, running the work, keeping each execution's trace, and its
account. What it does not own: the phase order, the challenges, the checks, the configuration's refusals.
"""

from __future__ import annotations

import secrets

from verity.circuits import Circuit, trace
from verity.protocol import registry
from verity.protocol.accounting.account import Account, CircuitId, Device, Execution, Inventory, IsolationUnit, SpatialUnit
from verity.protocol.accounting.interfaces import EpochStart
from verity.protocol.commitments import Commitment

CHAT = Circuit(inputs=2, gates=(("mul", 0, 1), ("add", 2, 0), ("xor", 3, 1)), outputs=(4,))
GPU = Device("gpu0", "h100-sxm", work=1000, memory_bits=80_000)
INVENTORY = Inventory((GPU,), (SpatialUnit("rack0", ("gpu0",), ("eth0",)),))
NOVEL_BITS = {"honest": 800, "rollout": 1_000_000}  # what each request reads beyond the baseline


class Engine:
    def __init__(self, scheme: str = "sha256-leaves", mode: str = "honest", space: bool = True):
        self.scheme = registry.load(f"commitments/{scheme}")
        self.mode, self.space = mode, space
        self.runs: list[tuple[Circuit, tuple[int, ...], tuple[int, ...]]] = []

    def commit(self, epoch: int) -> tuple[Commitment, Commitment]:
        return self.scheme.commit([b"weights of record"]), self.scheme.commit([b"baseline"])

    def serve(self, start: EpochStart) -> None:
        for request in range(4):
            inputs = (request + 3, 7)
            wires = trace(CHAT, inputs)
            if self.mode == "lazy":
                wires = wires[:-1] + (0,)
            self.runs.append((CHAT, inputs, wires))

    def close(self, epoch: int) -> tuple[Commitment, Account, bytes]:
        executions = tuple(Execution(CircuitId(c.id()), "u0", "gpu0", work=100, state_bits=500) for c, _, _ in self.runs)
        self.account = Account(epoch, (IsolationUnit("u0", "rack0"),), executions)
        self.nonce = secrets.token_bytes(16)
        return self.scheme.commit([self.account.encode(), self.nonce]), self.account, self.nonce

    def side(self, interface: str) -> object:
        return self

    # integrity (sampled proofs)
    def executions(self) -> list[tuple[Circuit, tuple[int, ...], object]]:
        return self.runs

    # space (PoUS): everything not declared holds the salted encoding, unless the engine skipped it
    def occupied(self, salt: bytes) -> dict[str, int]:
        return {GPU.id: GPU.memory_bits - self.account.declared(GPU.id)[1]} if self.space else {}

    # work (PoUW)
    def certified(self, salt: bytes) -> dict[str, int]:
        return {GPU.id: self.account.declared(GPU.id)[0]}

    # compliance (NCI, disclosed)
    def disclose(self):
        novel = NOVEL_BITS["rollout" if self.mode == "rollout" else "honest"]
        return self.account, self.nonce, {CircuitId(CHAT.id()): (len(CHAT.gates), novel)}
