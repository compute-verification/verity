[2026-11-11T11:20Z] d8c63aadaec4
FORM to training, TRN-103, Wed 11 Nov 11:20Z. Short answers; the statements note follows after one node try of the statements.

1. cpu-8, for infra: on vy-nebius-cpu-8 the siblings step of `verity lean try` fails after `lean exit 0` with "object file .../Proofs/Accounting/ReadUnit/Witnesses.olean ... does not exist"; runs r20261111-100121-b240 and r20261111-100123-790a [ran].
2. h0: DROPPED, with the linter option. My call site needs nothing of the writer's position [inferred: the property's WriterLeaf carries 0 <= wu itself].
3. COUNTS' branch is merged; head af84cff48b on cursor/record-property-laidout-3b1c [ran: pushed]. One root for both families (RecordCounts imports RecordLaidOut and the four), so the lakefile's count stays 27. The package's Lean check is NOT run yet: two runs of mine are in flight.
4. M4's resolution half compiled at its first try: ResolveRows, 15 theorems, three axioms, siblings clean, r20261111-105219-e96e (no wait; 261 s) [ran].
5. ownRows reading dims: one commit of its own, eef1ce17ff on cursor/record-ownrows-dims-3b1c [ran: pushed]. On the 670 pinned difftest cases a probe finds 0 verdicts moved, 0 arguments with an axis and no dims or dims and no axis, over 8,370 batch nodes [ran, on the VM]. The node run with the benchmark's check and the probe on the Programs main builds is r20261111-110708-253e, WAITING for a check slot on vy-nebius-cpu-1 since 11:07Z. Until it is in, M2 is not proved.
6. A FINDING for Counts' staffing: the landing bound needs "a gate is as wide as the leaf that names it", which nothing proves. The sizing called it short (it cited a Width.lean that is about another check) [read]. It rests on the decoder's width check (Flock/Program.lean, namespace Widths, about 290 lines of closed-form comparisons) being sound: a large family of its own. LaidOut does not read it [inferred]. Statement in the note.

