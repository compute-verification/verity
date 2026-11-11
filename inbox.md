[2026-11-11T12:44Z] bea24253be96
From the training lane's coordinator, Wed 11 Nov 4:48 AM PST. Your four answers are received, and they are the
best reading I have had of this property. FORM answered (a) and (c) on its own the same hour and agrees with you
on both; your answer 2's last paragraph is what neither FORM nor I had: nothing says the committed circuit file's
circuit is one the record lays out. It is on LEAN-238 from me, with your lines, asking lean whether it owns that
step.

## Your next job: size that step. Read and state; prove nothing yet.

The step: `Served F c (V.declaration θ F)`, so that `Complies` gives the composite's
`Nci.CompliesRate (V.gates θ) (V.fresh θ) (V.declaration θ F)` (Game.lean 805).

1. STATE IT. One Lean statement, with its hypotheses, that would close `InferenceOnlySound` (or the part of it
   the lock leaves open) from `Complies` and what the check accepts. Say what F, c and θ must be tied by: which
   field says the record's Program is the one the committed circuit file was compiled from (`d.B.run d.log.theta
   d.a`, `circuitOfFile`, the Program's digest in the record). If two readings are possible, give both.
2. SPLIT IT by `Served`'s clauses (`LaysOut`'s four: the gate count through `at_`, widths, input flags, reads;
   `Names`; `Divides`; `Filled`; the `Spot.Wires.check` clause). For each: which existing lemma reaches it, by
   name and file (the agreement of the check's descriptor with the circuit, `DescAgree.*`, is the first place to
   look; `CircuitOf.lean`; whatever relates `Flock.Program.resolve` to a gate of the file's circuit), or that none
   does, and then its size as you judge it: small (a day's lemma), medium, large, with the reason.
3. THE WIRED LEAVES. `LaysOut` is of the Program's gates "with the wired leaves spliced out". Does the committed
   circuit file's circuit have those leaves spliced out too, or does it keep them as gates? If it keeps them, the
   step is false as stated and what is true is a containment or a quotient: say which, with the smallest example.
   This is the question I most want answered, and an answer of "false as stated" is a result, not a failure.
4. WHO HAS PART OF IT. `rg` the lean lane's notes and the lock for anything that already states or plans this
   (the lane's report you cited, `InferenceOnlySound`'s record in the lock, `Properties/Remote.lean` 52-54's
   sentence). One line each: what it says and where.
5. If the statement of 1 elaborates on a node in one try, try it (a file of your own under
   `security/Proofs/Properties/Nci/`, with `sorry` nowhere: state it as a `def ... : Prop`, not a theorem). If
   it would take more than two tries, leave it as text.

Rules as before: read first; heavy work on a CPU node through `research run`; your own branch
(`cursor/training-record-wired-05f2`) only; edit no file of FORM's, COUNTS' or lean's; no pull request; a note at
least every 60 minutes; comments on TRN-116 with `--worker WIRED`, at most 100 words (TRN-120 is cancelled: no
comment there). Your inbox send to me takes no `--local`.

DATE: Thu 12 Nov noon PST, in one message: the statement, the split with sizes, the answer to 3, what you read
and did not. If 3's answer is "false as stated", send that the hour you know it, before the rest.

[2026-11-11T12:53Z] 69abfea12fa3
From @training (coordinator), Wed 11 Nov 4:54 AM PST. One addition to the sizing job (brief-wired-bridge), from
FORM at 4:51 AM PST. FORM READ it, ran nothing; verify each line before you build on it.

THE QUESTION IS ONE STEP EARLIER THAN "spliced or not". What the composite calls the circuit is not the Program's
circuit at all:
- `V.circuitOf θ F = (V.read θ F).circuit` (Properties/Verity/Game.lean 745, 560-562)
  `= Flock.Bridge.circuitOfFile tags cf tables pf` (Properties/Flock/Circuit/CircuitOf.lean 165)
  `= Circuit.ofAudit (keyCircuit D hU n)` (54-64, 100-113): the Flock circuit file's key program, the descriptor
  `Flock.Desc.parse` reads, with ONE unit's rows placed n times (`keyProg D.unroll hU n`, n the public file's
  population), every gate ONE BIT (`w _ := 1`), gates Boolean (input/and/xor/not/row).
- `LaysOut P W S rs D at_` is about the circuit whose gates are the Program's (`P.gate g`, the served graph's JSON,
  `Flock.Program.Prog`), each gate as wide as its primitive's `ret`, wired leaves spliced out.
- No wire appears in `circuitOf`'s definition: it reads no `Spot.Wires` wiring and no Program.
- NOT READ by FORM: `keyProg`, `Placed`, `Layout.outsOf`, `HmOut.loadPublic`; so not known how the key program
  treats a unit's input that another unit writes.
- At lean's head 72a184b3e792 the guarantee is `ServedComplies` (Properties/Nci/Record.lean 495-496), proved by
  `complies_of` (Proofs/Properties/Nci/Record.lean 314-316): your reading of `Complies` is a reading of it.

SO SIZE THIS, in place of question 2 of the brief (the rest of the brief stands, and so does Thu 12 Nov noon PST):
what ties the circuit file's key program (bit gates, one unit placed n times) to the record's Program (value
gates, the whole graph). Say:
1. Whether `Served F c D` can hold at all with D = `.ofCircuit (V.circuitOf θ F)` as the two are defined today, or
   whether its clauses (gate count, widths, reads) are false of a one-bit-gate circuit for any real Program. If
   false, the statement to prove is another one: write the one you think is meant (a relation between the two
   circuits under which `CompliesRate` moves across), and say which of `CompliesRate`'s terms that relation must
   carry (the block sizes in bits, fresh, the declaration's reads).
2. Whether anything already relates a Program to its Flock circuit file in Lean (grep `Program` under
   Properties/Flock and Proofs/Flock, `Desc.parse`'s users, the partition check, `NciCheck.Desc`), or only Python
   does (the builder that writes both files): name the function if so.
3. Who would own it: is it a statement about the builder (an integration's completeness, which the design rules
   leave unproved) or about the verifier (which must then CHECK the tie between the two files and be proved to)?
   One paragraph, your reading, marked as a reading. This is the part lean and Daniel will want first.
Send what you have when (1) is answered, without waiting for the rest.

