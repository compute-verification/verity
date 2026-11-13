module

public import Proofs.Properties.Verity.SampledTwo.WiredParse
public import Proofs.Properties.Verity.SampledTwo.JoinedCoupling
public import Proofs.Properties.Verity.Soundness

/-!
# The wired statement is in the session's scope, or the sampled verdict never accepts

`CouplesAt.scope` at the statement `JoinsAt` fixes (`stmtX`: the record the commitment `c` carries), from `JoinsAt`
and that the held copies' design evaluates (`TwoHolds`, `scope_wired`):

* where no play after `θ`, `F` and `c` meets the sampled verdict, the verdict's probability is zero
  (`prob_play_eq_zero`);
* where one does, the form accepts it (`wiring`). Its case reads the commitment `c` (`caseCommits`), so its run's
  boundary record is the one `c` carries as the verifier's parser reads its canonical text (`input_of_accepts`), and
  the proved verifier's reader reads the statement back as that record (`jsonMsg_of_readBack`): the statement decodes.
  A statement that parses where the held copies' design evaluates is in the two-stage scope at the connected program
  (`SpotFlock.Honest.scope_of_two`).
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The wired statement is in the session's scope, or the sampled verdict never accepts** (`CouplesAt.scope`), from
the join's premise and that the held copies' design evaluates. -/
theorem scope_wired {V : Vocab d α} {k : Verity.Registration.Composite} {A : Live₂} {x : Msg} {θ : ByteArray}
    {F : Array ByteArray} {c : ByteArray} {σ : Verity.Game.Strategy (V.game θ F c)} {C : CaseOf}
    {Ms : SessionReading} {tab : ℕ → String} {ℓ₁ ℓ₂ : String} {Dg : Verity.Exchange → ℕ → Option Msg}
    {P : Sessions.Deployment} {fs₁ fs₂ : ByteArray → List (String × ℕ)} {cut : List Verity.Release → ℕ}
    {P' : Verity.Exchange → ℕ → Claims.Prover Unit}
    (h : JoinsAt V k A x θ F c σ C Ms tab ℓ₁ ℓ₂ Dg P fs₁ fs₂ cut P') (hT : SpotFlock.RunsAs.Round2.TwoHolds A.h) :
    A.L.Scope x ∨ Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ = 0 := by
  by_cases hE : ∃ o : Verity.Exchange, (o.params = θ ∧ o.files = F ∧ o.commitment = c) ∧ V.Verdict k .sampled o
  · left
    obtain ⟨o, ⟨hθ, -, hc⟩, hv⟩ := hE
    have hA : FormAccepts C o := h.wiring o hθ hv
    obtain ⟨kk, hk, hAcc⟩ := hA
    obtain ⟨cm, i, hcm, -, hrb, -⟩ := input_of_accepts hAcc
    have hkc : kk.commitment = c := (h.caseCommits o kk hk).trans hc
    have hrec : recordMsg i.record = recordMsg cm.record := by
      obtain ⟨s, hs, -, hs'⟩ := VStar.Recursive.SampledTwo.readBack_ok hrb
      unfold recordMsg
      rw [hs, hs']
    have hsx : stmtOf c = recordMsg cm.record := by
      rw [← hkc]
      unfold stmtOf
      simp only [hcm]
    have hxr : recordMsg i.record = some x := by
      rw [hrec, ← hsx]
      exact h.stmtX
    have hdec : SpotFlock.decode x = some ⟨i.record, SpotFlock.portRoots i.record⟩ := by
      unfold SpotFlock.decode
      rw [jsonMsg_of_readBack hrb hxr]
      rfl
    obtain ⟨s, hs⟩ : ∃ s, SpotFlock.parse A.h x = some s := by
      unfold SpotFlock.parse
      rw [hdec]
      exact ⟨_, rfl⟩
    obtain ⟨dd, hdd⟩ := Option.isSome_iff_exists.1 (hT s)
    exact SpotFlock.Honest.scope_of_two hs hdd
  · right
    exact prob_play_eq_zero (V.families θ) θ F c _ (fun o ho hv => hE ⟨o, ho, hv⟩) σ

end Verity.SampledTwo
