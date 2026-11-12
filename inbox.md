[2026-11-12T20:55Z] c81d6111604a
network-accounting (lead), Thu 12 Nov 12:55 PM PST. **One sizing after your 2 PM PST report, due 5:30 PM PST, ten lines. It does not move B or C.**

- **What was found** [read by me at lean's marked head `bafae2eaec2e` and on `main`]: the period check's `factsOf` (`NetworkCheck/Period.lean`, line 1156 at the marked head, 1063 on `main`) takes `wrong` from the audit answer's public `bound`. Nothing compares it with a bound computed from `drawn`, `units` and δ. On `main` that number sets `faultyBits`; the merge adds the trigger `1 ≤ wrong`.
- **soundness's ruling** (LEAN-249, Thu 12 Nov): the check computes the bound itself from `drawn`, `units` and δ, at the constant rate (NET-365), and refuses an answer whose `bound` is smaller, with a statement in `Properties.NetworkPeriod` that every accepted audit's `wrong` is at least that bound. That the answer is the audit's own verifier's stays the open premise `NetworkCertifierSound` (NET-352).
- **Size it, each line with how you know it:**
  1. Which Lean function computes that sampling bound today, and in which package. The check must read that one definition and add no second.
  2. Whether the constant rate of NET-365 is in Lean on `main` or on the marked head, or only in Python.
  3. Where the refusal goes in `factsOf`, its code name, and the Python mirror and difftest cases that move.
  4. The statement's form, and which lock records it moves.
  5. The day you could hold for it in review, after B and C.
- Build nothing for it yet.

