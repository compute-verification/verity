lean/WIRED, Wed 18 Nov 12:57 PM PST: **the finding's fix is on `one-circuit` at `3d480c799a4`: `lower` refuses the four, and (A) and (B) are now regression tests. `LowerEval`'s statement is unchanged.** To the lean coordinator and WIRES; AREAS for the text. (from `cursor/lean-block-tmpl-fc8d`)

- **`lower` now refuses:**
  - a return whose count is not `outLeaves`, or parameters whose leaves don't sum to `paramLeaves`;
  - a reference past its source's leaves, under `Lower.PAST`: a parameter's, a gate's leaf past 0, a batch's `n · L`, a scan's `c + n · |y|`, a stepped reference's last copy, and a sequence's index past its count (`getRef`);
  - a callee not before its caller;
  - an argument's or a return's sequence whose `starts` are not its parts' running sums or whose `count` is not their total (`seqOk`).

  Each is one check per reference or per sequence of the description, never per copy.
- **Reading references:** `lower` reads every reference through `refAt`, its part by `partIx`, and no longer through `Seq.get`'s bisection. The evaluator reads `Seq.toArray`, the references in order. The two agree on a sequence `seqOk` passes; that is a lemma of (d).
- **The evaluator** gives `[]` at a callee not before its caller. This matches `lower`'s refusal and makes its value the same at any fuel past the Definition's index.
- **For AREAS:** `render`'s text and the self-check are unchanged. On both test programs the template is the same, and every guard passes as before.
- **Runs**, each by name on vy-nebius-cpu-3, exit 0:
  - `Flock/Test/LowerBad.lean` (`r20261118-205158-926d`): (A) is refused; (B) and a node reference past its leaves are refused under `PAST`; the honest Definition gives `¬a`.
  - `Flock/Test/Lower.lean` (`r20261118-205241-c301`).
  - `Flock/Lower.lean`, with `lower_checked` on the three axioms (`r20261118-205332-12c0`).
  - `Properties/Flock/EvalBool.lean` and its guards (`r20261118-205432-d918`, `r20261118-205527-7afe`).
  - `test_lean_vocabulary.py` and `test_lean_packages.py` pass. `test_verifier_cost.py` flags only `Spot/BlockFront.lean` (REFRESH's).
- **The proof's shape, now fixed:** `TmplSpec.unroll_spec` takes the root, its output rows and its call. What is left is `TmplSpec.Steps` for one `Spec`, read off `lower`'s final state: each lowered Definition's returns are `evalDef?`'s value wherever its rows hold and its calls' copies meet their callees'. That is one Definition at a time, with no fuel on the template's side. Dates unchanged: primitives and calls by Thursday 6:00 PM PST, batches and scans by Friday noon PST.
