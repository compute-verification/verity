[2026-11-11T11:47Z] 1a60101ac9fc
FORM to training, TRN-103, Wed 11 Nov 11:47Z. THE STATEMENTS FOR COUNTS AND FOR A SECOND WORKER ARE IN ONE NOTE.

Note: 20261111T1146Z-handoff-trn103-brief-counts-c5-c6-c1-and-wired-circuit (synced). It is written as a brief: where to start, the files, the header, each statement as a Lean block, why it holds, what not to change.
Start: cursor/record-property-laidout-3b1c at 9d60cc0fb6 [ran: pushed]. It has COUNTS' four files merged and my RecordTakes.lean, the hypotheses as definitions.

a. C5 and C6, for COUNTS: eight short lemmas about a row's bits (RecordRowBits.lean) and four about a block's gates (RecordBlockSize.lean). No hypothesis of mine is a proof; each is arithmetic or a statement.
b. C1, for COUNTS: YES, it can go now. One theorem, landing_le (RecordLanding.lean): the gates a row's reads land on are together no wider than the row. It reads neither M6 nor M7 and no statement is moving: it takes Served's clauses and eight statements of mine as hypotheses (LeafWide, ReturnsOwn, UnitsComputed, PortsNamed, OutFits, Ranked, and OwnRows, Aligned, LaysOut from the property). Large: an induction over the units by the wires' rank.
c. M10, for a second worker: YES. One theorem, laysOut_exists (RecordWired.lean), with the plan in the note: order the unwired gates by input gates first, then by the least rank of a unit holding the gate, then by gate index. It needs no theorem about extending an order and reads no width, and shares nothing with COUNTS but RecordTakes.lean.

What I ran: all 14 statements and the definitions elaborated on a node with no proofs (r20261111-113239-b98b, lean exit 0; waited 5 min for a slot, 367 s in all). After it I moved the definitions into RecordTakes.lean and gave OutFits one more hypothesis; that module alone is not compiled yet. I try it as soon as a run slot is free and tell you if anything moves.
What I ran for Ranked: SPOT's check_acyclic held its rank inside the proof. Two new lemmas state it, check_ranked and writerOf_reads (Spot/WiresSound.lean, compiled on my VM, commit e5b0ab4c17).

ONE THING YOU NEED TO DECIDE, from my 11:21Z message: LeafWide (a gate is as wide as the leaf that names it) is a hypothesis of landing_le that nothing in the tree proves. It is the soundness of the decoder's width check, in four parts in the note's last section; the large one (W1b) is self-contained and could go to a worker, but I have not written its statement. Counts cannot close without it. LaidOut does not read it [inferred].
What stays mine: ReturnsOwn's two bounds, UnitsComputed, OperandsOwn, GatesOwn, PortsNamed, OutFits, the grids' period for C6, and the assembly of Counts and LaidOut.

