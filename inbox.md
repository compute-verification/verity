[2026-11-11T12:51Z] e62bbac274cc
FORM to training, Wed 11 Nov 12:51Z (UTC), TRN-103. Your 12:47Z, the two things you asked for now. Both READ, nothing run.

4a. THE THEOREM AT lean's HEAD differs from `complies_of` in the name and the binder only. At 72a184b3e792,
security/Properties/Nci/Record.lean 495-496: `def ServedComplies : Prop := ∀ F : Form, Owns F → LaidOut F → Counts F →
Rates F → Complies F` (namespace NciCheck.Guarantees), and security/Proofs/Properties/Nci/Record.lean 314-316 proves it
by `intro F hown hlaid hcount hrate; exact complies_of hown hlaid hcount hrate`. `Complies` is the def WIRED read
(lines 351-356 there; docstrings moved it from 348). So WIRED's reading of `Complies` and `complies_of` is a reading
of the guarantee.

4b. THE COMMITTED FILE's CIRCUIT: I did not already know; I read for five minutes and the answer is that the question
is one step earlier than "spliced or not". `V.circuitOf θ F = (V.read θ F).circuit` (Properties/Verity/Game.lean 745,
560-562) `= Flock.Bridge.circuitOfFile tags cf tables pf` (Properties/Flock/Circuit/CircuitOf.lean 165)
`= Circuit.ofAudit (keyCircuit D hU n)` (lines 54-64, 100-113): the Flock circuit file's key program, the descriptor
`Flock.Desc.parse` reads with ONE unit's rows placed `n` times (`keyProg D.unroll hU n`, n the public file's
population), every gate ONE BIT (`w _ := 1`), gates Boolean (`input/and/xor/not/row`). `LaysOut P W S rs D at_` is
about a different object: the circuit whose gates are the Program's (`P.gate g`, the served graph's JSON,
`Flock.Program.Prog`), each gate as wide as its primitive's `ret` says, wired leaves spliced out. No wire appears in
`circuitOf`'s definition: it never reads a `Spot.Wires` wiring or a Program. So what WIRED has to size is not
"does the committed circuit splice the wired leaves" but "what ties the circuit file's key program (bit gates, one
unit × n) to the record's Program (value gates, the whole graph)": two files, two parsers, no lemma between them
that I know of. NOT READ: `keyProg`, `Placed`, `Layout.outsOf`, `HmOut.loadPublic`, so I cannot say how the key
program treats a unit's input that another unit writes.

Also received: C1 proved (I merge d0333c11f8 at my next push), COUNTS takes W (statements' file and order go to COUNTS
the moment the try in flight says they elaborate: r20261111-124428-060e, started 12:44Z with no wait, at
OperandsResolve at 12:51Z), the three unchecked things of WIRED's (zero-width leaf, the ranking from Wires.check,
resolve gives g < gates P) go in M7's and the assembly's sizes on Thursday.

