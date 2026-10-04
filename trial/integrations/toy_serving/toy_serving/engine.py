"""A toy serving engine that plays the developer in `verity.protocol.run`, the way vLLM would.

What it owns is what only an engine can: running the work, committing each execution's boundary values as it serves,
choosing the partition, rematerializing a drawn unit by replay, and its account. What it does not own: the phase order,
the coins, the checks, the configuration's refusals.
"""

from __future__ import annotations

import random
import secrets
from collections.abc import Callable, Sequence

from verity.circuits import partition as P
from verity.circuits.circuit import Circuit, trace
from verity.primitives.commitments import SCHEMES, Commitment
from verity.protocol.accounting.account import Account, CircuitId, Device, Execution, Inventory, IsolationUnit, SpatialUnit
from verity.protocol.compliance import Disclosure
from verity.protocol.epoch import EpochStart
from verity.subprotocols.sampled_proofs import registration as R
from verity.subprotocols.sampled_proofs.audit import Opening
from verity.subprotocols.sampled_proofs.law import Subset

# gate g reads the two wires before the one it writes: regions of two consecutive gates have 4 boundary wires
CHAT = Circuit(inputs=2, gates=tuple((op, g, g + 1) for g, op in enumerate(("mul", "add", "xor", "add", "mul", "xor", "add", "mul"))), outputs=(9,))
PAIRS: P.Partition = tuple((g, g + 1) for g in range(0, len(CHAT.gates), 2))
GPU = Device("gpu0", "h100-sxm", work=1000, memory_bits=80_000)
INVENTORY = Inventory((GPU,), (SpatialUnit("rack0", ("gpu0",), ("eth0",)),))


def chat_requests(epoch: int) -> list[tuple[int, ...]]:
    return [(request + 3, 7) for request in range(4)]


class Engine:
    def __init__(
        self,
        scheme: str,
        integrity: str,
        law: Subset,
        circuit: Circuit = CHAT,
        partition: P.Partition = PAIRS,
        requests: Callable[[int], Sequence[tuple[int, ...]]] = chat_requests,
        device: Device = GPU,
        work: int = 100,  # work units of record per request
        novel_bits: int = 800,  # what each request reads beyond the baseline
        mode: str = "honest",  # honest | lazy (commits made-up values) | rollout (reads far more than it declares)
        lazy_epochs: frozenset[int] = frozenset(),
        corrupt: int = 0,  # units per epoch whose committed outputs it flips
        space: bool = True,
    ):
        self.scheme, self.integrity, self.law = SCHEMES[scheme](), integrity, law
        self.circuit, self.partition, self.requests, self.device = circuit, partition, requests, device
        self.work, self.novel_bits = work, novel_bits
        self.mode, self.lazy_epochs, self.corrupt, self.space = mode, lazy_epochs, corrupt, space
        self.wires = R.boundary_wires(circuit, partition)
        self.registered: str | None = None

    def commit(self, epoch: int) -> tuple[Commitment, Commitment]:
        return self.scheme.commit([b"weights of record"]), self.scheme.commit([b"baseline"])

    def serve(self, start: EpochStart) -> None:
        self.inputs: list[tuple[int, ...]] = []  # all it keeps of each execution: traces are ephemeral
        self.committed: list[dict[int, int]] = []  # the boundary values it committed, per execution
        for request, inputs in enumerate(self.requests(start.epoch)):
            values = trace(self.circuit, inputs)
            committed = {w: values[w] for w in self.wires}
            if self.mode == "lazy" and request > 0 and start.epoch in self.lazy_epochs:
                committed = {w: (v if w < self.circuit.inputs else secrets.randbits(1)) for w, v in committed.items()}
            self.inputs.append(tuple(inputs))
            self.committed.append(committed)
        for i, r in random.sample([(i, r) for i in range(len(self.inputs)) for r in range(len(self.partition))], self.corrupt):
            w = P.boundary(self.circuit, self.partition[r])[1][0]
            self.committed[i][w] ^= 1
        self.roots = [self.scheme.commit([R.leaf(w, c[w]) for w in self.wires]) for c in self.committed]

    def close(self, epoch: int) -> tuple[Commitment, Account, bytes]:
        executions = tuple(Execution(CircuitId(self.circuit.id()), "u0", self.device.id, self.work, state_bits=500) for _ in self.inputs)
        self.account = Account(epoch, (IsolationUnit("u0", "rack0"),), executions)
        self.nonce = secrets.token_bytes(16)
        return self.scheme.commit([self.account.encode(), self.nonce]), self.account, self.nonce

    def prover(self, subprotocol: str) -> object:
        return self

    # sampled proofs
    def register(self) -> tuple[R.Registration, dict[str, Circuit]]:
        executions = tuple(R.Execution(self.circuit.id(), self.partition, root) for root in self.roots)
        registration = R.Registration(executions, self.law, self.registered)
        self.registered = registration.digest()
        return registration, {self.circuit.id(): self.circuit}

    def open_unit(self, u: int) -> Opening:
        i, r = divmod(u, len(self.partition))
        region = self.partition[r]
        ins, outs = P.boundary(self.circuit, region)
        leaves = [R.leaf(w, self.committed[i][w]) for w in self.wires]
        values = {w: self.committed[i][w] for w in (*ins, *outs)}
        proofs = {w: self.scheme.open(leaves, self.wires.index(w)) for w in values}
        trace(self.circuit, self.inputs[i])  # rematerialize: an untrusted replay of the execution, for the proof's intermediates
        witness = None
        if self.integrity == "sp1":
            from verity.subprotocols.sampled_proofs.sp1.construction import prove

            witness = prove(self.circuit, region, {w: values[w] for w in ins})
        return Opening(values, proofs, witness)

    # space (PoUS): everything not declared holds the salted encoding, unless the engine skipped it
    def occupied(self, salt: bytes) -> dict[str, int]:
        return {self.device.id: self.device.memory_bits - self.account.declared(self.device.id)[1]} if self.space else {}

    # work (PoUW)
    def certified(self, salt: bytes) -> dict[str, int]:
        return {self.device.id: self.account.declared(self.device.id)[0]}

    # compliance (NCI, disclosed)
    def disclose(self) -> Disclosure:
        novel = 1_000_000 if self.mode == "rollout" else self.novel_bits
        return Disclosure(self.account, self.nonce, {CircuitId(self.circuit.id()): (len(self.circuit.gates), novel)})
