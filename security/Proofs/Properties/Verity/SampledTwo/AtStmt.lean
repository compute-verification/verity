module

public import Proofs.Properties.Verity.SampledTwo.Reading

/-!
# The cases at one statement

The form's premises each speak of every play the form's check accepts on a case map `C`, and most of them name one
session of record `A`, whose held copies are built from one boundary record and one set of parameters. Accepted plays
that carry two boundary records share no `A`, so such a premise is false at a `C` that decodes both.

`atStmt C x` keeps the cases whose check reads the statement `x`, the boundary record's bytes (`stmtOfCase`). A
premise at `atStmt C x` speaks only of the accepted plays at `x` (`inputOn_at`), which are plays `C` accepts
(`formAccepts_of_at`). `AcceptsRun` at `x` looks at no other play, and at such a play the reading is `C`'s
(`formReading_at`), so `AcceptsRun` at `C` follows from `AcceptsRun` at `atStmt C x` (`acceptsRun_at`).

The parameters need no such map: a `C` that decodes only the plays at one parameters' bytes is the assembly's to
choose.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- **The statement the check reads of a case**: its run of record's boundary record, as bytes. -/
def stmtOfCase (k : Case) : Option Msg := (inputOfCase k).bind fun i => recordMsg i.record

open Classical in
/-- **The cases at the statement `x`**: `C`'s case of a play where the check reads `x` of it, none elsewhere. -/
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

/-- **A play the cases at `x` accept is one `C` accepts.** -/
theorem formAccepts_of_at {C : CaseOf} {x : Msg} {o : Verity.Exchange} (h : FormAccepts (atStmt C x) o) :
    FormAccepts C o := by
  obtain ⟨k, hk, ha⟩ := h
  exact ⟨k, (atStmt_some hk).1, ha⟩

/-- **The run of record's input on a case at `x`** is `C`'s, and its boundary record's bytes are `x`. -/
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

/-- **The reading at a play reads the case map only through the play's input**: two case maps with one input on `o`
read the same statement, coins, messages and agreement of `o`. -/
theorem formReading_at {A : Live₂} {C C' : CaseOf} {S : SessionReading}
    {dev : List Verity.Release → ℕ → Option FlockDeploy.Coin} {o : Verity.Exchange}
    (h : inputOn C' o = inputOn C o) :
    (formReading A C' S dev).stmt o = (formReading A C S dev).stmt o ∧
      (formReading A C' S dev).coin o = (formReading A C S dev).coin o ∧
      (formReading A C' S dev).msg o = (formReading A C S dev).msg o ∧
      (formReading A C' S dev).agree o = (formReading A C S dev).agree o := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp only [formReading, h]

/-- **A play `C` accepts at the statement `x` is one the cases at `x` accept**, with the same input. -/
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

/-- **`AcceptsRun` at `C` from `AcceptsRun` at the cases at `x`**: `AcceptsRun` at `x` looks only at the accepted plays
whose statement is `x`, each a play the cases at `x` accept with the same reading. -/
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
