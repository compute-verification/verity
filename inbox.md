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

