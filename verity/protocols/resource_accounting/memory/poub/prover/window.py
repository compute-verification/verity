"""PoUB's read session's statements, handed to the proof system as any integration hands a window over
(`verity.commitments.periods.interface`): one opening unit (`block_check.BlockOpeningUnitV2`, `BlockOpeningCheck_v2` in
its six rows) a sample of the session's window, its values recorded into a `Window`. The verifier decides the window from
the proof system's accepted runs of those units (`Poub.Span.checkSpan`), opening each unit's ``pub``, ``signs``
and ``o0`` rows (the verifier's `span`).

`SessionIntegration` is a window's statements, an `Integration`: its `circuit()` and its `run(window)`, and the rows the
verifier makes (`supplied`). Until the proof system commits rows the verifier makes (`Registered.supplied`,
`Window.give`), `circuit()` raises through its interface.
"""
from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.commitments.periods.interface import Circuit, Read, Registered, Window
from verity.protocols.resource_accounting.memory.poub.prover import audit_prover
from verity.protocols.resource_accounting.memory.poub.verifier import audit, codec, receipts, span
from verity.protocols.resource_accounting.memory.poub.verifier.commitments.gates import block_check as BC
from verity.protocols.resource_accounting.memory.poub.verifier.opening import depth

GOAL = "memory-accounting/read-session-in-zero-knowledge"
#: the registered parameter the units' ``block`` rows read: the store's encoded blocks, a block's BYTES/2 leaves each
STORE = "store"
#: the registered parameters the verifier makes, a row a unit, each read by the unit's row of its name
SUPPLIED = span.SUPPLIED


def statics(reg: audit.Registration, samples: Sequence[audit.PositionSample]) -> SimpleNamespace:
    """The opening unit's statics for a window's samples of the store ``reg`` registers: its codec, 4,096-byte blocks, the
    registration's sketch rows, the samples' receipt leaf size, and the trees' depths (the deepest pass's receipt tree,
    the registered payload tree's, a value tree of one level, which a registered block never reads)."""
    sizes = {d.leaf_size for d in samples}
    if len(sizes) != 1:
        raise NotImplementedError(f"{GOAL}: samples whose receipt leaves are {sorted(sizes)} receipts, one opening unit's "
                                  "binding each, in one window (the span reads one template)")
    c = reg.codec
    dr = max(depth(receipts.leaves(d.plan.length)) for d in samples)
    return SimpleNamespace(N=c.field.n, C=c.field.c, K=c.rounds, MASKED=int(c.masked), BYTES=codec.BLOCK_BYTES,
                           ROWS=reg.rows, LEAF=sizes.pop(), DR=dr, DP=depth(reg.blocks), DV=1)


class SessionIntegration:
    """PoUB's read session as an integration: one opening unit per sample (i, p) of the window, in sample order, over
    block j = p of a pass whose plan is the identity, a registered block (no append's coin). Its ``block`` row is a
    registered read of the store's block j; its ``pub`` and ``signs`` rows are registered reads of rows the verifier makes
    itself from the registration, the pass and the sample (`span.public_rows`), its `Registered.supplied` (`supplied`,
    which the caller gives the window, `Window.give`); its ``payload``, ``leaf`` and ``aux`` rows are the prover's, held
    inside the proof to the payload root and the pass's receipt root; its output ``o0`` is the verdict word.

    `registration` is the store's, `samples` the window's position samples (`audit.PositionSample`, in sample order);
    `store` (the prover's blocks, payloads and their salts and paths) and `passes` (by (window, index), each pass's
    receipts, salts and tree) only `run` reads."""

    def __init__(self, registration: audit.Registration, samples: Sequence[audit.PositionSample],
                 store: audit_prover.Store | None = None,
                 passes: dict[tuple[int, int], audit_prover.Pass] | None = None) -> None:
        self.registration, self.samples = registration, tuple(samples)
        self.store, self.passes = store, passes
        if not self.samples:
            raise ValueError("a read session's window has a unit a sample: at least one")
        if any(d.coin is not None or d.block != d.position for d in self.samples):
            raise ValueError("a sample's unit opens a registered block at its own position: every pass's plan the identity")
        self.S = statics(registration, self.samples)
        self.unit = bind(BC.BlockOpeningUnitV2, **vars(self.S))

    def circuit(self) -> Circuit:
        """The samples' units, unit i reading block j of the registered ``store`` at leaf ``j·BYTES/2`` and row i of
        each supplied parameter."""
        tid, n, bw = descriptor_id(self.unit), len(self.samples), self.S.BYTES // 2
        width = {r: bits // 16 for r, bits, _ in BC.opening_rows_v2(self.S)}
        reads = tuple(x for i, d in enumerate(self.samples) for x in (
            Read(tid, "block", STORE, i, (1,), d.block * bw, (0,)),
            *(Read(tid, r, r, i, (1,), i * width[r], (0,)) for r in SUPPLIED)))
        return Circuit("poub-read-session", ((self.unit, n),),
                       registered=Registered({STORE: self.registration.blocks * bw, **{r: n * width[r] for r in SUPPLIED}},
                                             reads=reads, supplied=SUPPLIED))

    def supplied(self) -> dict[str, list[bytes]]:
        """Each supplied parameter's rows, a unit's each in read order, as the verifier makes them and `Window.give` takes
        them (`span.supplied`)."""
        return span.supplied(self.samples, self.registration.codec_seed, self.S)

    def run(self, window: Window) -> None:
        """Each unit's prover rows (``block`` and the supplied rows by reference to their registered reads) and its
        verdict word (`block_check.opening_check_v2`)."""
        if self.store is None or self.passes is None:
            raise ValueError("SessionIntegration.run reads the store's blocks and payloads and the passes' receipts")
        laid = BC.opening_rows_v2(self.S)
        per: dict[str, list[list[int]]] = {r: [] for r, _bits, _ in laid}
        verdicts = []
        for d in self.samples:
            a = self._args(d)
            for r, _bits, params in laid:
                per[r].append([w for p in params for w in a[p]])
            verdicts.append(BC.opening_check_v2(self.S, a))
        inputs = [None if r in ("block", *SUPPLIED) else np.array(per[r], dtype=np.uint16) for r, _bits, _ in laid]
        window.record(self.unit, inputs, [np.array(verdicts, dtype=np.uint16)])

    def _args(self, d: audit.PositionSample) -> dict[str, list[int]]:
        """Sample ``d``'s arguments, the honest prover's (`opening_prover.witness`'s, under v2): its block and payload from
        the store, the receipt leaf under its salt and path in its pass's tree, the payload under its salt and path in the
        registered payload tree."""
        j, c = d.block, d.leaf
        ps, store = self.passes[d.window, d.index], self.store
        pay_salt, pay_path, _ = store.committed(j)
        leaf = b"".join(ps.receipts[receipts.PER_LEAF * c:receipts.PER_LEAF * (c + 1)])
        public = span.public_v2(d, self.registration.codec_seed, self.S)
        return BC.opening_args_v2(self.S, public, store.blocks[j], store.payload(j), leaf, ps.salts[c], pay_salt,
                                  tuple(ps.tree.open(c).path), tuple(pay_path), ())

