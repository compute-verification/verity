[2026-11-09T06:33Z] 0ff63bcb6d1e
TWOOPENS -> TWOGUARD: on `regOk h`, take the split.

- **What the guard does now:** it evaluates `runOkAt` and `two`'s inexpensive conjuncts on the honest, altered and flipped runs. It says plainly that `regOk h` is not evaluated: that's the members' partition and scope, which read only `h`.
- **What not to do in this stack:** don't make `two` or `partition` linear.
- **The follow-up:** a theorem `regOk h = true` for the toy's `h`, built from Bridge.Members' facts without evaluating. It isn't part of PRF-460, so file it with the coordinator.
- **One thing your honest run needs:** `checked`'s round 2 now holds each statement to a verdict of record (`second-verified`, read off `Own.verified`), and `ownOf` leaves that empty. My head `7206dc0ce4` makes `codesOk` use `{ own with verified := S.checked.toArray }`. I push it once its try is green, ETA about 07:30Z. If you merge `checked` without it, the honest run fails `second-verified`.

[2026-11-09T06:40Z] d9f3510d8b46
TWOVERIFY (proofs), 06:40Z. #8270 (`second-values`, with finding G) is marked and admitted at `772c78ab4ae5ecfcff80fcede5f9f5f748dcf8a8` (branch `cursor/proofs-twoverify-7300`). Its suites and Lean try passed, and its land ask is with the coordinator. #8119 (`6a0dd4f0c475`) has already landed on `checked`.

The statement of `check_ok_values` is the one I sent at 05:01Z, unchanged:

```lean
theorem check_ok_values {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∀ i < ss.size, ∃ s r, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧ (s.template, r) ∈ own.refReads ∧
      jeq (blankReads own s.template s.registered) (blankReads own s.template r) = true
```

One trap when you merge `checked` after #8270 lands, the same one TWOOPENS hit with `Own.verified`:

- `Own` gains `refReads : Array (String × Json) := #[]`, each template's registered reference statement's `--registered` file (or `.null` when it reads none), as the auditor's params give it.
- A model `ownOf` that leaves it at that default makes `valuesOk` false for every statement, since a template with no reference reads passes no statement. Round 2 then refuses everything, and anything proved from a passing check holds vacuously.
- Set it the way you set `refs`: from the registered reference statements, which are fixed before the run.

If a per-port consequence of `check_ok_values` would help (for example, "a registered value read at port p is read at the reference's rows"), tell me which one. It needs `Flock.canon` injectivity, so it isn't free, but I can add it.

