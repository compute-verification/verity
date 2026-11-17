module

public import Proofs.Properties.Verity.SampledTwo.Reading

/-! The cases at one statement (`atStmt`), so that a premise naming one session of record speaks only of
the accepted plays at one boundary record. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims

def stmtOfCase (k : Case) : Option Msg := (inputOfCase k).bind fun i => recordMsg i.record

open Classical in
noncomputable def atStmt (C : CaseOf) (x : Msg) : CaseOf :=
  fun o => (C o).bind fun k => if stmtOfCase k = some x then some k else none

theorem atStmt_some {C : CaseOf} {x : Msg} {o : Verity.Exchange} {k : Case} (h : atStmt C x o = some k) :
    C o = some k ∧ stmtOfCase k = some x := by
  unfold atStmt at h
  cases hC : C o with
  | none => rw [hC] at h; cases h
  | some k' =>
    rw [hC, Option.bind_some] at h
    split at h
    · cases h; exact ⟨rfl, by assumption⟩
    · cases h

theorem atStmt_of {C : CaseOf} {x : Msg} {o : Verity.Exchange} {k : Case} (hk : C o = some k)
    (hs : stmtOfCase k = some x) : atStmt C x o = some k := by
  unfold atStmt
  rw [hk, Option.bind_some]
  split
  · rfl
  · exact absurd hs ‹_›

theorem formAccepts_of_at {C : CaseOf} {x : Msg} {o : Verity.Exchange} (h : FormAccepts (atStmt C x) o) :
    FormAccepts C o := by
  obtain ⟨k, hk, ha⟩ := h
  exact ⟨k, (atStmt_some hk).1, ha⟩

theorem inputOn_at {C : CaseOf} {x : Msg} {o : Verity.Exchange} {i : Spot.Run.Input}
    (h : inputOn (atStmt C x) o = some i) : inputOn C o = some i ∧ recordMsg i.record = some x := by
  unfold inputOn at h ⊢
  cases hk : atStmt C x o with
  | none => rw [hk] at h; cases h
  | some k =>
    rw [hk, Option.bind_some] at h
    obtain ⟨hC, hs⟩ := atStmt_some hk
    rw [hC, Option.bind_some]
    refine ⟨h, ?_⟩
    unfold stmtOfCase at hs
    rw [h, Option.bind_some] at hs
    exact hs

theorem formReading_at {A : Live₂} {C C' : CaseOf} {S : SessionReading}
    {dev : List Verity.Release → ℕ → Option FlockDeploy.Coin} {o : Verity.Exchange}
    (h : inputOn C' o = inputOn C o) :
    (formReading A C' S dev).stmt o = (formReading A C S dev).stmt o ∧
      (formReading A C' S dev).coin o = (formReading A C S dev).coin o ∧
      (formReading A C' S dev).msg o = (formReading A C S dev).msg o ∧
      (formReading A C' S dev).agree o = (formReading A C S dev).agree o := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp only [formReading, h]

theorem at_of_stmt {A : Live₂} {x : Msg} {C : CaseOf} {S : SessionReading}
    {dev : List Verity.Release → ℕ → Option FlockDeploy.Coin} {o : Verity.Exchange} (hA : FormAccepts C o)
    (hx : (formReading A C S dev).stmt o = some x) :
    FormAccepts (atStmt C x) o ∧ inputOn (atStmt C x) o = inputOn C o := by
  obtain ⟨k, hk, hacc⟩ := hA
  have hs : stmtOfCase k = some x := by
    have := hx
    simp only [formReading, inputOn, hk, Option.bind_some] at this
    exact this
  have hat : atStmt C x o = some k := atStmt_of hk hs
  exact ⟨⟨k, hat, hacc⟩, by unfold inputOn; rw [hat, hk]⟩

theorem acceptsRun_at {A : Live₂} {x : Msg} {C : CaseOf} {S : SessionReading}
    {dev : List Verity.Release → ℕ → Option FlockDeploy.Coin}
    (h : AcceptsRun A x (formReading A (atStmt C x) S dev) (FormAccepts (atStmt C x))) :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) := by
  intro o hA hx Ω P ω r hc hm
  obtain ⟨hA', hin⟩ := at_of_stmt hA hx
  obtain ⟨e₁, e₂, e₃, e₄⟩ := formReading_at (A := A) (S := S) (dev := dev) hin
  refine h o hA' (e₁.trans hx) Ω P ω r ?_ ?_
  · rw [e₂, e₄]; exact hc
  · rw [e₃]; exact hm

end Verity.SampledTwo
