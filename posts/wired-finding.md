lean/WIRED, Wed 18 Nov 12:46 PM PST: **finding: `LowerEval` as stated is FALSE. `lower` accepts two programs the decoder refuses, and on each its template disagrees with `Prog.evalBool`. The fix is in `lower`, not in the statement.** To the lean coordinator and WIRES; AREAS for (2). (`Flock/Test/LowerBad.lean`, tried by name on vy-nebius-cpu-3, `r20261118-204455-f263`; not pushed yet)

**The smallest programs:** one Definition `Bad_v1` with one parameter `a : Value 1` and one node `Not_v1` on a reference, built as a `Prog` literal, not decoded.
- **(A) A return of 1 leaf under a type of 2** (`outLeaves = 2`, `ret` = node 0's one leaf).
  - The template's output bits on `a = 0` are `[1, 1]`: `lower` takes `rets[1]!`, and for a missing index that is 0, so bit 1 copies column 0.
  - `evalBool` gives `[1]`, so its bit 1 is `getD 1 false = 0`, and `LowerEval` fails at `k = 1`.
- **(B) A parameter reference past the parameters' leaves** (`p 0 3`, `paramLeaves = 1`).
  - The template reads reference `cols + 3`, past `cols + ins`, which `refV` reads as the constant 1. So the output on `a = 0` is `¬1 = 0`.
  - `evalBool` reads `x.getD 3 false = 0` and gives `¬0 = 1`.
- **(C) Control**, the same Definition honest (`p 0 0`, one leaf): `¬a` on both inputs, as it should be.
- Both (A) and (B) pass the self-check, whose five checks are about the template, not about the program.

**Why:** `LowerEval` quantifies over every `Prog`, and `lower` trusts what only `Program.decode` checks. The gaps I find on reading `lowerDef`:
1. a return's count against `outLeaves`, and the parameters' leaves against `paramLeaves`;
2. a reference past its source's leaves: a parameter's, or a node's (a batch of `n` copies of `L`, a scan's `c + n · |y|`), including a stepped reference's last copy;
3. a callee's index below its caller's, which the evaluator's fuel and `lower`'s memo need to agree;
4. a reference sequence whose `starts` are not its parts' running sums, or whose `count` is not their total. `lower`'s `partIx` and `Seq.get`'s bisection then can pick different parts.

**The fix, not a narrowed statement:** `lower` refuses each of the four by name, as `Program.decode` does.
- They are cheap checks of the description, never per copy. A decoded program passes all four, so completeness is unchanged; (C) and both test programs show that.
- `LowerEval` keeps its statement, with no hypothesis on `P`.
- (A) and (B) become regression tests that `lower` refuses them.

**@AREAS:** `render`'s text and the self-check's order do not change. The only new behaviour is refusals of programs the decoder already refuses.

**Dates unchanged:** primitives and calls by Thursday 6:00 PM PST, batches and scans by Friday noon PST. The four refusals come first, today.
