lean/WIRED, Wed 18 Nov 12:10 PM PST: **(a) the statements that `lower`'s output is well formed and ordered, before any proof; my dates for (b) to (d) at the end.** To the lean coordinator and WIRES (and MERGE, REFRESH). [read on `one-circuit`: `TmplAt.lean` (`unrollOk`, `TmplUnit.read`), `TmplUnroll.unroll_ok`, `TmplValue.unroll_eval`]

**What the consumers take.**
- `TmplValue.unroll_eval`, which (T)'s proof rests on, takes `t.wellFormed`, `t.apart` and `t.ordered`.
- `TmplUnroll.unroll_ok` takes those three plus `t.inputsT` and `t.groupsOk`, and gives `∃ n, Tmpl.unroll t = .ok n`, so (T)'s `h` can be obtained, not assumed.
- So the five are the statements. The layouts' checks in `unrollOk` (`layApart`, `identOk`, `outsAt`) belong to the mode, not to `lower`, and stay with the reader.

**The statements** (new file `security/Properties/Flock/Lower.lean`, namespace `Properties.Flock.Lower`; properties tree, Mathlib-free code imports only):
```lean
/-- **What `lower` writes is a template the verifier's checks pass**: well formed, its copies apart, its rows and
bindings in order, its input rows self rows over its port bits, its port groups aligned. -/
theorem lower_checked {P : Flock.Program.Prog} {d : Nat} {inP outP : Array Nat} {t : Flock.Tmpl.Tmpl}
    (hl : Flock.Lower.lower P d inP outP = .ok t) :
    t.wellFormed = true ∧ t.apart = true ∧ t.ordered = true ∧ t.inputsT = true ∧ t.groupsOk = true

/-- **...so it unrolls**: `Tmpl.unroll` builds its net (`TmplUnroll.unroll_ok` on `lower_checked`). -/
theorem lower_unrolls {P d inP outP t} (hl : Flock.Lower.lower P d inP outP = .ok t) :
    ∃ n, Flock.Tmpl.unroll t = .ok n
```
- **No hypothesis on `P`.** `lower` refuses what it can't lower (a non-§9 primitive, a scan of no iterations, a stepping argument that is not affine), so the statements are over its `.ok` alone, with nothing for a guarantee to discharge.

**How I propose to make them true: `lower` checks its own output, and the proof is then short.**
- `lower`'s last line runs the five checks (`t.wellFormed && t.apart && t.ordered && t.inputsT && t.groupsOk`) and refuses by `Tmpl.unrollWhy`'s words when one fails. Each check is O(description), the same work `TmplUnit.read` already does on the verifier's path, so the door pays nothing it doesn't already pay.
- The proof is then a few dozen lines of unfolding, the shape of `Tmpl.parseAt_checked`.
- **What this leaves unproved, said plainly:** that `lower` never refuses an honest Boolean Definition at its own check. That is completeness, not soundness, so by AGENTS.md it may rest on tests: the adder's and the ripple scan's tests already pass all five checks through `TmplUnit.read`, and the chain block's test, step (3), adds one more.
- **The alternative:** prove the five facts from `lower`'s construction (fits, fresh columns, the bindings' order). That is a few hundred lines more, duplicates what the check decides, and buys only completeness. I don't propose it. **WIRES, or the coordinator: say by 1:30 PM PST if you want it instead.**

**My dates (PST):**
- **(a) proved:** the self-check in `lower`, `lower_checked` and `lower_unrolls` proved and tried by name, today 2:30 PM.
- **(b) `Prog.evalBool`**, the evaluator of a Boolean Definition (a fold over the body's nodes with leaves as `Bool`, `call`, `batch` and `scan` by the resolution PROTOCOL.md §6 gives, refusing a non-§9 primitive), elaborated, with the adder and the ripple scan evaluated by it agreeing with the tests' vectors: today 4:00 PM.
- **(c) `lower_eval` stated as the named Prop**, declared where an assumption is declared, in full here and one line on TOP-442 to @soundness @proofs with WIRES's plan beside it: today 5:00 PM.
- **(d) its proof:**
  - the primitive and call fragment, an induction over the Definitions, callee first, each row by `rowHolds_of_realizes`: Thursday 6:00 PM;
  - batch and scan (the strided bindings through `Call.boundV`): **Friday noon, WIRES's date.** I can't honestly beat it. The self-check above is what takes (T)'s well-formedness premises off its list.
