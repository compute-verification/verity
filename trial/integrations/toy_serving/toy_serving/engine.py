"""A toy serving engine that plays the developer in `verity.protocol.epoch.run`, the way vLLM would.

What it owns is what only an engine can: running the work, committing each execution's boundary values as it serves,
choosing the partition, rematerializing a drawn unit by replay, and its account. What it does not own: the phase order,
the coins, the checks, the configuration's refusals.
"""

from __future__ import annotations

import secrets

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
NOVEL_BITS = {"honest": 800, "rollout": 1_000_000}  # what each request reads beyond the baseline


class Engine:
    def __init__(self, scheme: str, integrity: str, law: Subset, mode: str = "honest", space: bool = True):
        self.scheme, self.integrity, self.law = SCHEMES[scheme](), integrity, law
        self.mode, self.space = mode, space
        self.inputs: list[tuple[int, ...]] = []  # all it keeps of each execution: traces are ephemeral
        self.committed: list[dict[int, int]] = []  # the boundary values it committed, per execution
        self.roots: list[Commitment] = []

    def commit(self, epoch: int) -> tuple[Commitment, Commitment]:
        return self.scheme.commit([b"weights of record"]), self.scheme.commit([b"baseline"])

    def serve(self, start: EpochStart) -> None:
        wires = R.boundary_wires(CHAT, PAIRS)
        for request in range(4):
            inputs = (request + 3, 7)
            values = trace(CHAT, inputs)
            committed = {w: values[w] for w in wires}
            if self.mode == "lazy" and request > 0:  # skipped the work: commits made-up values
                committed = {w: (v if w < CHAT.inputs else secrets.randbits(32)) for w, v in committed.items()}
            self.inputs.append(inputs)
            self.committed.append(committed)
            self.roots.append(self.scheme.commit([R.leaf(w, committed[w]) for w in wires]))

    def close(self, epoch: int) -> tuple[Commitment, Account, bytes]:
        executions = tuple(Execution(CircuitId(CHAT.id()), "u0", "gpu0", work=100, state_bits=500) for _ in self.inputs)
        self.account = Account(epoch, (IsolationUnit("u0", "rack0"),), executions)
        self.nonce = secrets.token_bytes(16)
        return self.scheme.commit([self.account.encode(), self.nonce]), self.account, self.nonce

    def prover(self, subprotocol: str) -> object:
        return self

    # sampled proofs
    def register(self) -> tuple[R.Registration, dict[str, Circuit]]:
        executions = tuple(R.Execution(CHAT.id(), PAIRS, root) for root in self.roots)
        return R.Registration(executions, self.law), {CHAT.id(): CHAT}

    def open_unit(self, u: int) -> Opening:
        i, r = divmod(u, len(PAIRS))
        region = PAIRS[r]
        ins, outs = P.boundary(CHAT, region)
        wires = R.boundary_wires(CHAT, PAIRS)
        leaves = [R.leaf(w, self.committed[i][w]) for w in wires]
        values = {w: self.committed[i][w] for w in (*ins, *outs)}
        proofs = {w: self.scheme.open(leaves, wires.index(w)) for w in values}
        trace(CHAT, self.inputs[i])  # rematerialize: an untrusted replay of the execution, for the proof's intermediates
        witness = None
        if self.integrity == "sp1":
            from verity.subprotocols.sampled_proofs.sp1.construction import prove

            witness = prove(CHAT, region, {w: values[w] for w in ins})
        return Opening(values, proofs, witness)

    # space (PoUS): everything not declared holds the salted encoding, unless the engine skipped it
    def occupied(self, salt: bytes) -> dict[str, int]:
        return {GPU.id: GPU.memory_bits - self.account.declared(GPU.id)[1]} if self.space else {}

    # work (PoUW)
    def certified(self, salt: bytes) -> dict[str, int]:
        return {GPU.id: self.account.declared(GPU.id)[0]}

    # compliance (NCI, disclosed)
    def disclose(self) -> Disclosure:
        novel = NOVEL_BITS["rollout" if self.mode == "rollout" else "honest"]
        return Disclosure(self.account, self.nonce, {CircuitId(CHAT.id()): (len(CHAT.gates), novel)})
