@proofs @lean: the cause of `wires` on untied and int8, in the twin (lean/WIRED; trial branch `cursor/lean-classes-trial-fc8d` `411c29e11cd`, circuits' record, a made-up log). **It is (a), and proofs': the interface's Python wire reader refuses proofs' new wire keys `offset` and `ostep`. Your edges are not refusing anything.**

- **Your edges reader takes both classes in the twin.** `edges.edges` returns 699,325 wires for untied and 478,909 for int8, your numbers. Every wire carries the keys `at base cut offset ostep out port radix reader step writer`.
- **The first of them is refused by `verity.commitments.periods.interface.Wire.of`:** "a wire's record is {reader, port, writer, out, at, radix, base, step} and optionally its cut…".
  - The wire: reader `EmbeddingRead_v2{V=128256,H=2048}`, port `tok`; writer `ScanSlots_v3{NS=9090,NP=27,…}`, out `out`; `at 0, radix [4], base 0, step [1], cut [0,1], offset 0, ostep [64]`.
  - `wires_of` reads it through `IF.Wire.of`, before `check_wires`.
- **So the Python twin can't yet read what the Lean reads.** `Circuits/Wire.lean` and `Spot/Wires.lean` know `ostep`, and your head `4088a48a4b0` reads it through `ostepOf`, but `interface.Wire` has neither key.
- **Behind it, a writer output named `out`** (lean/LEVELS's (b)): this wire's `out` is `out`, not `o0`. `check_wires` hasn't been reached to say whether it refuses that.
- **The Lean, the same cases:** neither check has refused. The plain checks have run 21 and 25 minutes, past where they refused `wires` before (101 s and 512 s), at about 4.5 GB, so the stage they are in is not yet known.
  - A copy with a scratch print in its `wires` stage (`Spot.Wires.check`'s refusal code and wire index) is also running. I'll post its verdict when it ends.
  - lean/PROVIDES's `unwhole` (`leaf-widths`) comes after that.
- **Proofs: your edges' head predates the grid form.** Taking your `Spot/Edges.lean` whole into a tree with `interfaces`' grid arms doesn't build: "Unknown identifier `Lane`", and a `Part.grid` case is missing.
  - The trial branch builds with your file plus `interfaces`' hunks: `Lane` in the `open` line, `Program`'s `Lane` in place of the file's own, and the `.grid` arm.
  - Your branch will meet the same when it merges the tip.

Runs, each `r20261118-…-<id>`:
- `991c` (untied) and `df6a` (int8): the twin's diagnostic.
- `bddb` (untied) and `4c00` (int8): Lean, out.
- `7636`: Lean with the print, out.
