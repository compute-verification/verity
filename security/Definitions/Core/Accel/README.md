# The ACCEL model

How much computing can a card do in `T` seconds? The accounting answers with ACCEL: a machine 𝓜 (`Accel.Machine`) made
of a card's table of numbers (`Accel.Card`, from its card file under `hardware/cards/`) and a small set of ops, where a
run fits `T` when its code fits the card's code space and each resource's units are at most its rate times `T`. The card
itself is untrusted; one named assumption, below, ties it to the model. [`DESIGN.md`](DESIGN.md) has the headline with
its conditions (P1) and (P2) on the semantics, the table, the ops, what is proved and what is open.

## Try it

~~~sh
bash tools/verity/lean/setup.sh security
(cd security && lake build Definitions.Core.Accel)
~~~

The first line installs the toolchain and restores `main`'s build from the Lean cache; the second builds the model's
definitions (`Card.lean`, `Isa.lean`, `Machine.lean`).

## The card assumption (`Assumptions.CardWithinAccel`)

The card does no more than 𝓜 at its numbers: for some semantics meeting (P1) and (P2), whatever the card computes in
`T` seconds, some program of 𝓜 computes on a schedule fixed before the content, within `T` (`Machine.Fits`). It rests on
one claim: every instruction the card executes is dominated by an ACCEL op (or a branch) that reads and writes at least
its words, at no more units of each resource. The per-instruction evidence is the opcode table,
`benchmarks/machine_model/data/cover/rows.toml`. At a table, the premise also says the cards are that table's SKU
and firmware.

