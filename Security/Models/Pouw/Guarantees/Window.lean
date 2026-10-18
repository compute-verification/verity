import Definitions.Pouw.Window

/-!
# A window's ε from its calls' (M2), with a draw that depends on the registration (C1) (trusted)

The window guarantee (`audit_window_split_of_record`, `keyedWindowReg_audit_of_record`) takes one `Analysis` of the whole
window, with the window's `ε_ks` and `δ_link` (its M2). This states what they are in terms of the calls': a window of `C`
calls, registered at once, drawn at once by a law that depends on the registration (C1), and proved call by call
(`Pouw.Window`). Stated, not proved: Daniel's order is statement, red team, card, then proof (17 Oct), so
`lean-audit.json` lists `WindowEps` under `shape.frozen` as `unstated`. The statement, its premises, the proof plan and
what could make it false are in note:proofs/20261018T0400Z-draft-pouw-gap6-statement.

* `WindowEps`: `Pr[accept ∧ wrong(X) ∈ 𝓑] ≤ sup_{B ∈ 𝓑} escape_{∏ L_c(R)}(B) + η + max_c ε_c + ∑_c δ_c`, where `X` is
  the committed transcript (fixed at registration), `∏ L_c(R)` the calls' laws at the prover's registration drawn
  independently, `η` the draw's distance from that product, `ε_c` call `c`'s knowledge error from every state of its
  session, and `δ_c` call `c`'s link term in the window.

Knowledge soundness enters at its largest call's: whatever the draw, the rule picks one unit, of one call, and that
call's extraction fails from the state the window brings with probability at most its `ε_c`. The link terms add: each
bounds the picked unit's disagreement with the window's committed transcript on the draws where the unit is its call's.

`WindowEps` holds for every family `𝓑`. PoUW reads it at `harmSets Wref ε`, the wrong-unit sets holding the share `ε`
of the window's work by the scheme's work table, where a unit is a self-contained `r × c` tile with harm `W_ref(tile)`,
fan 1 and no steering term (decision 106). At the sizing of record (`K = K_y = 27,713`, a 0.1% share of the work or of
y's cells, `hone` and y's floors) the first term is at most `2^-40`, as `audit_window_split_of_record` bounds it for the
product law.

Each unit's statement carries its position: its seed inputs (tag, salt, position, root(A_j), root(B_j)) are
`UnitStmts.seedIn X ⟨c, u⟩`, the tag and salt public, the roots committed in `X` and bound through the opening path,
and the position its place `⟨c, u⟩` in the registered partition. So a prover can't pick a unit's position, or with it
its seed, inside the unit.

Two things are outside this statement by design:
* **Luck.** Wrong means committed outputs that aren't the unit's function of its inputs (`UnitStmts.wrong`), so a
  deficient unit whose receipt came out right by luck isn't wrong. PoUW adds luck through gap (4), TT or the TS model's
  `ε_luck`, composed with the window audit, which reads `WindowEps`'s bound.
* **Fiat–Shamir grinding** of the per-matmul seeds (decision 103) enters PoUW's `δ` through gap (7), not here.
-/

namespace Pouw.Guarantees.Window

open Pouw.Window
open scoped ENNReal

/-- **M2 with C1: the window's ε from its calls'.** For a window of `C` calls, call `c` with `n c` units, a registration
`R` (the receipt), the calls' laws `L R c` at each registration and the law `Lw R` the verifier runs there, each call's
session `sess c` on its share of the draw, the window's transcripts and its units' statements `U` (each unit's seed
inputs, with its place as its position, and what it computes), a committed transcript that is a function of the
prover's state at registration, and an analysis of each call: if the draw's escapes are within `η` of the calls'
product law at every registration (`EscapesWithin`), every call's extraction fails from any state of its session with
probability at most `ε c` (`CallAnalysis.KnowledgeSound`), and every call's link term in the window is at most `δ c`
(`LinkSoundIn`), then for every family `𝓑` of wrong-unit sets and every prover, the window accepts with its committed
transcript's wrong units in `𝓑` with probability at most the largest escape over `𝓑` of the calls' product law at the
prover's registration, plus `η`, plus the largest `ε c`, plus the sum of the `δ c`.

A unit is wrong when its committed outputs aren't its function of its inputs, so luck is not in the event: a deficient
unit whose receipt came out right isn't wrong, and gap (4) adds it. Fiat–Shamir grinding of the per-matmul seeds enters
PoUW's `δ` through gap (7). -/
def WindowEps : Prop :=
  ∀ {Reg Tr : Type} {C : ℕ} {n : Fin C → ℕ} (L : Reg → (c : Fin C) → Law (Fin (n c))) (Lw : Reg → Law (WUnit n))
    (η : ℝ≥0∞) (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool) (U : UnitStmts Tr n)
    (committed : (R : Reg) → Cont Lw (windowSession sess) R → Tr) (A : (c : Fin C) → CallAnalysis U c (sess c))
    (ε δ : Fin C → ℝ≥0∞),
    EscapesWithin Lw L η →
    (∀ c, (A c).KnowledgeSound (ε c)) →
    (∀ c, LinkSoundIn Lw sess committed A c (δ c)) →
    ∀ (𝓑 : Finset (WUnit n) → Prop) (σ : Strategy (auditReg Lw (windowSession sess))),
      prob (fun o => o.1 = true ∧ 𝓑 (U.wrong (committed (reg σ) (cont σ)))) (auditReg Lw (windowSession sess)) σ ≤
        (⨆ (B : Finset (WUnit n)) (_ : 𝓑 B), (Law.prod (L (reg σ))).escape B) + η + (⨆ c, ε c) + ∑ c, δ c

end Pouw.Guarantees.Window
