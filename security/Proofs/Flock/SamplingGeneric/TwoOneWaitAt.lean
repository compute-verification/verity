module

public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassAt
public import Proofs.Flock.SamplingGeneric.TwoOneWait
import Proofs.Flock.Claims.Prob

/-!
# The two-stage first-hit bound by class of gates, with the finders run on the one-wait extraction

`eventTwoX_le_cls_at_budget` bounds the event of `KSRegTwoX` by `κ_s + √(k·γ_s·Pr[B])` and the binding finders'
cost over `2^256`, the cost that of the proof system's extraction, `e_s = 1 + m′·k` runs' worth of reruns a
session: an extractor of `m′` waits, each for `k` accepted reruns.

Here the finders run the extraction of the one-wait proof system (`CompiledTwo.oneWait`), which pays one wait a
session, and the event is the one at that extraction's committed traces:

* `eventTwoX_oneWait_le_of_off`: `eventTwo_oneWait_le_of_off` at the boundary trace read at the statement, with one
  run's own disagreement at `B_conf,X`'s first run.
* `εclsOneWait`, `eventTwoX_oneWait_le_cls_at_budget`: **the bound at a budget `T` a run**,
  `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·(2·runCost + 4·firstHitCost)/2^256`, the two costs the one-wait compiled
  verifier's. Against `εclsAt` it pays `2·κ_s + √(k·γ_s·Pr[B])` more and twice the cost term, at a cost that is
  one wait's, `e_s ≤ 1 + k`, where `εclsAt`'s is `m′` waits'.

The premises are `eventTwoX_le_cls_at_budget`'s, read at the one-wait compiled verifier (its extractions open a class
whole or not at all, the linear expected-time bound at its `B_conf,X` and its two by-class finders at every `K`), and
`Arg.OneWaitFinds` beside the proof system's knowledge soundness. No statement of the definitions tree changes.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The event at the one-wait extraction's traces from bounds on its two off-trace terms**:
`eventTwo_oneWait_le_of_off` at the boundary trace read at the statement, with one run's own disagreement at
`B_conf,X`'s first run. An accepted run samples a proof block of the one-wait compiled verifier's `FregTwoX` with
probability at most `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·(Pr[B_conf,X] + b_B + b_N)`, the three binding terms the
one-wait compiled verifier's. -/
theorem eventTwoX_oneWait_le_of_off {H : Msg → Msg} {L : CompiledTwo Coin A} (hF : A.OneWaitFinds H)
    (hA : A.KS H) (hbind : L.commit.Binds H) (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) {bB bN : ℝ}
    (hB : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.oneWait.OffRegX H P x q.1 q.2.1 q.2.2) ≤ bB)
    (hN : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.oneWait.OffIreg H P x q.1 d.part q.2.1 q.2.2) ≤ bN) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤
      3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        2 * ((Compiled.BconfX H L.oneWait.toCompiled Ω P x).Pr H + bB + bN) := by
  simp only [FregTwoX_eq H L.oneWait P hs hPt hd]
  exact eventTwo_oneWait_le_of_off hF hA hS hs hPt hd hin P (fun _ => L.oneWait.regValX H P x)
    (Compiled.pr_inConfX_le (L := L.oneWait.toCompiled) hbind hs P) hB hN

/-- **The one-wait by-class first-hit bound's right-hand side at a budget `T` a run**:
`3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·(2·runCost + 4·firstHitCost)/2^256`, with `runCost` and `firstHitCost` the one-wait
compiled verifier's: `εclsAt`'s two costs at an extraction of one wait a session. -/
noncomputable def εclsOneWait (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (T : ℕ) (Ns Ks Km : ℝ) : ℝ :=
  3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
    2 * (2 * L.oneWait.toCompiled.runCost x T Ns + 4 * L.oneWait.toCompiled.firstHitCost x T Ns Ks Km) /
      2 ^ (256 : ℝ)

/-- **The event at the one-wait extraction's traces, by class of gates, at a budget `T` a run.** On a statement in
scope, where a reveal for proof block `u` opens only gates of `needs u`, the one-wait compiled verifier's extractions
open a class of gates whole or not at all under the statement's roots (`hallX`) and under a replay block's interior
roots (`hallI`), every run calls `H` at most `T` times, both parties' calls counted (`hT`), the opening checks call
`H` at most `openCalls` times a slot (`hopen`), `N_s`, `K_s` and `K_m` are `eventTwoX_le_cls_at_budget`'s, and the
one-wait compiled verifier's `B_conf,X` and its two by-class finders at every `K` meet the linear expected-time
bound, an accepted run samples a proof block of the one-wait compiled verifier's `FregTwoX` with probability at most
`εclsOneWait`:

`3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·(2·runCost + 4·firstHitCost)/2^256`,

the costs those of an extraction that pays one wait a session. The proof system's part is its knowledge soundness
(`hA`) and that what its extractor finds its one wait finds at half the rate (`hF`, `Arg.OneWaitFinds`). -/
theorem eventTwoX_oneWait_le_cls_at_budget {H : Msg → Msg} {L : CompiledTwo Coin A} (hF : A.OneWaitFinds H)
    (hA : A.KS H) (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)
      (g g' : Fin (L.nG x)), cls g = cls g' → (L.oneWait.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.oneWait.iopTable H P x t.1 d.part t.2 ec g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.oneWait.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.oneWait.finderICls H P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤ L.εclsOneWait H x P T Ns Ks Km := by
  have hS₁ : L.oneWait.Sessions H := hS.oneWait
  have hopen₁ : L.oneWait.toCompiled.OpenCallsLe H := hopen.oneWait
  have key : ∀ a b c e n k : ℝ,
      a ≤ b + 2 * (c + (2 * e / 2 ^ (256 : ℝ) + n / k) + (2 * e / 2 ^ (256 : ℝ) + n / k)) →
        a ≤ b + 2 * c + 8 * e / 2 ^ (256 : ℝ) + (4 * n) / k :=
    fun a b c e n k h => h.trans (le_of_eq (by ring))
  have hcap : ∀ K : ℕ,
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤
        3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
            2 * (Compiled.BconfX H L.oneWait.toCompiled Ω P x).Pr H +
          8 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
            ((L.oneWait.callsAt H P x s q.1 q.2.1 q.2.2 +
              ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                  decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
                L.oneWait.toCompiled.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
            2 ^ (256 : ℝ) +
          (4 * (L.nG x : ℝ)) / ((K : ℝ) + 1) := by
    intro K
    have hB' := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
        (run H P L.V x q.1 q.2.1).out = .acc ∧ L.oneWait.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
      (Compiled.pr_offRegX_le_cls_cap (L := L.oneWait.toCompiled) hbind hs P needs hpos cls hallX K (hX K))
    have hN' := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
        (run H P L.V x q.1 q.2.1).out = .acc ∧ L.oneWait.OffIreg H P x q.1 d.part q.2.1 q.2.2) fun _ h => h.2).trans
      (pr_offIreg_le_cls_cap (L := L.oneWait) hbind hS₁ hs d.part P needs hpos cls hallI K (hI K))
    exact key _ _ _ _ _ _ (eventTwoX_oneWait_le_of_off hF hA hbind hS hs hPt hd hin P hB' hN')
  have h := Verity.FirstHit.le_of_forall_cap hcap
  have hc := Compiled.ex_calls_le_at_cls (L := L.oneWait.toCompiled) hS₁.toCostContract hopen₁ hs P needs cls T hT
    hNs hKs hKm
  have hb := Compiled.bconfX_le_at (L := L.oneWait.toCompiled) hS₁.toCostContract hopen₁ x P T hT hNs hB
  have key2 : ∀ a b c e r f p : ℝ, a ≤ b + 2 * c + 8 * e / p → c ≤ 2 * r / p → e ≤ f → 0 < p →
      a ≤ b + 2 * (2 * r + 4 * f) / p := by
    intro a b c e r f p h1 h2 h3 hp
    have h4 : 8 * e / p ≤ 8 * f / p := div_le_div_of_nonneg_right (by linarith) hp.le
    have h5 : b + 2 * (2 * r + 4 * f) / p = b + 2 * (2 * r / p) + 8 * f / p := by ring
    rw [h5]
    linarith
  exact key2 _ _ _ _ _ _ _ h hb hc (by positivity)

end CompiledTwo

end Claims.SamplingGeneric
