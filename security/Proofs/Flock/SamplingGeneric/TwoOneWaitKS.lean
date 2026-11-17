module

public import Proofs.Flock.SamplingGeneric.TwoOneWaitAt
public import Proofs.Flock.SamplingGeneric.TwoRegMiss
public import Proofs.Flock.SamplingGeneric.TwoRegX
import Proofs.Flock.Claims.Prob

/-! The two-stage committed form at the one-wait extraction's boundary trace, its threshold form and headline from its event without the knowledge soundness the one-wait proof system does not have. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition InScopeTwo footprintAt)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **What the committed form at the one-wait extraction's boundary trace concludes**, at a statement `x` that
parses to `s` with the partition `Pt` and the two-stage design `d`, one prover and an error `ε`:

* `Pr[acc ∧ a sampled proof block is in the one-wait compiled verifier's FregTwoX] ≤ ε`;
* at every `δ ≥ 0`, `Pr[acc ∧ no interiors put the footprint at the one-wait boundary trace read at the statement
  in M̄_δ]` is at most `δ + ε`;
* `Pr[acc] ≤ p̄ + ε` at every `p̄` that bounds `P̄_a` at that footprint with any interiors. -/
def OneWaitTwoXAt (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) (s : AuditStatement) (Pt : Partition)
    (d : Sampling.Design.Two) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (ε : ℝ) : Prop :=
  Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤ ε ∧
    (∀ δ : ℝ, 0 ≤ δ →
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∀ ι : ℕ → ℕ → Bool, footprintAt s Pt d.part (L.oneWait.regValX H P x) ι ∉ d.M Pt δ) ≤ δ + ε) ∧
    ∀ pb : ℝ, (∀ ι, d.Pa Pt (footprintAt s Pt d.part (L.oneWait.regValX H P x) ι) ≤ pb) →
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc) ≤ pb + ε

theorem oneWaitTwoXAt_of_event {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {ε : ℝ}
    (hev : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤ ε) :
    L.OneWaitTwoXAt H x s Pt d P ε := by
  have hS₁ : L.oneWait.Sessions H := hS.oneWait
  have hev' := hev
  simp only [FregTwoX_eq H L.oneWait P hs hPt hd] at hev'
  unfold OneWaitTwoXAt
  refine ⟨hev, fun δ hδ => ?_, fun pb hpb => ?_⟩
  · exact threshold_of_event H L.oneWait hS₁ hs hPt hd hin P (fun _ => L.oneWait.regValX H P x) hev' hδ
  · exact (headline_of_event H L.oneWait hS₁ hs hPt hd hin P (fun _ => L.oneWait.regValX H P x) hev'
      (fun _ => pb) fun _ ι => hpb ι).trans_eq (by rw [Ex_const])

end CompiledTwo

end Claims.SamplingGeneric
