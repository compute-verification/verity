module

public import Proofs.Flock.SamplingGeneric.TwoRegFirstHitClass
public import Proofs.Flock.SamplingGeneric.TwoFirstHitAt
import Proofs.Flock.Claims.Prob

/-!
# The two-stage first-hit bound by class of gates, at a budget a run, as one closed right-hand side

`eventTwoX_le_firstHit_cls` bounds the event of `KSRegTwoX` by `κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X]` and
`4·E[c + Σ_{q ∈ classes of the sample} c_q]/2^256`, the sum over the representatives of the classes of the gates a
run's sample needs. Where each run calls SHA-512 at most `T` times, both parties', that expectation is at most
`Compiled.firstHitCost` at the classes' counts (`Compiled.ex_calls_le_at_cls`) and `Pr[B_conf,X]` at most
`2·Compiled.runCost/2^256` (`Compiled.bconfX_le_at`), so the event is at most

`κ_s + √(k·γ_s·Pr[B]) + (2·(T·(1 + e_s·N_s) + n_PU·openCalls) + 4·(T·(1 + e_s·N_s + K_s + e_s·K_m)`
`+ (1 + K_s)·n_PU·openCalls))/2^256`

(`CompiledTwo.eventTwoX_le_cls_at_budget`; `CompiledTwo.εclsAt` is that right-hand side).

* `N_s` bounds the expected number of sessions in a run's sample, `K_s` the expected number of classes of the gates
  the sample needs, and `K_m` the expected number of the classes' representatives its sessions need, session by
  session (`Compiled.sessCountIn`); the three are hypotheses on the coins alone.
* The premises are `eventTwoX_le_firstHit_cls`'s (a run's extractions open a class whole or not at all, `hallX` and
  `hallI`; the linear expected-time bound at the two by-class finders of every cap) and `eventTwoX_le_at_budget`'s
  (the budget, `OpenCallsLe`, the linear expected-time bound at `B_conf,X`). No statement of the definitions tree
  changes.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The by-class first-hit bound's right-hand side at a budget `T` a run**:
`κ_s + √(k·γ_s·Pr[B]) + (2·runCost + 4·firstHitCost)/2^256`, with `runCost = T·(1 + e_s·N_s) + n_PU·openCalls` and
`firstHitCost = T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·n_PU·openCalls`. -/
noncomputable def εclsAt (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (T : ℕ) (Ns Ks Km : ℝ) : ℝ :=
  L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
    (2 * L.toCompiled.runCost x T Ns + 4 * L.toCompiled.firstHitCost x T Ns Ks Km) / 2 ^ (256 : ℝ)

/-- **The event of `KSRegTwoX` by class of gates, at a budget `T` a run, with one closed right-hand side.** On a
statement in scope, under `eventTwoX_le_firstHit_cls`'s premises, where every run calls `H` at most `T` times, both
parties' calls counted (`hT`), the opening checks call `H` at most `openCalls` times a slot (`hopen`), `N_s` is at
least the expected number of sessions in a run's sample, `K_s` at least the expected number of classes of the gates
the sample needs and `K_m` at least the expected number of the classes' representatives its sessions need, session by
session, and `B_conf,X` meets the linear expected-time bound, an accepted run samples a proof block of `FregTwoX`
with probability at most `εclsAt`:

`κ_s + √(k·γ_s·Pr[B]) + (2·(T·(1 + e_s·N_s) + n_PU·openCalls) + 4·(T·(1 + e_s·N_s + K_s + e_s·K_m)`
`+ (1 + K_s)·n_PU·openCalls))/2^256`. -/
theorem eventTwoX_le_cls_at_budget {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u)
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.iopTable H P x t.1 d.part t.2 ec g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤ L.εclsAt H x P T Ns Ks Km := by
  have h := eventTwoX_le_firstHit_cls hA hbind hS hs hPt hd hin P needs hpos cls hrep hallX hallI hX hI
  have hc := Compiled.ex_calls_le_at_cls hS.toCostContract hopen hs P needs cls T hT hNs hKs hKm
  have hdiv := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 4))
    (by positivity : (0 : ℝ) ≤ 2 ^ (256 : ℝ))
  have hb := Compiled.bconfX_le_at hS.toCostContract hopen x P T hT hNs hB
  unfold εclsAt
  rw [add_div]
  exact (h.trans (add_le_add (le_refl _) hdiv)).trans
    ((add_le_add (add_le_add (le_refl _) hb) (le_refl _)).trans (le_of_eq (add_assoc _ _ _)))

end CompiledTwo

end Claims.SamplingGeneric
