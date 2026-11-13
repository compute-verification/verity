module

public import Proofs.Properties.Verity.SampledTwo.GoodCase
public import Proofs.Properties.Verity.SampledTwo.AcceptsTop
public import Proofs.Properties.Verity.SampledTwo.Reached
public import Proofs.Properties.Verity.SampledTwo.RefsOfSent

/-!
# The run's accept at the case kept on good plays

`playAccepts_form` and `partsHeldAt_of_accepts` read `HeldOfAt A C` at every accepted play. `HeldOfAtOr A C F₀` gives
the held copies only at the accepted plays whose blobs don't collide with the files `F₀`, so the accept is taken at
`goodCase C (NoCollide F₀)`, the case kept on those plays, where `HeldOfAt` holds outright (`heldOfAt_goodCase`).

Each premise the two read reads the case through `FormAccepts` and `inputOn` alone, so it passes from `C` to any kept
case (`portsDistinctAt_goodCase` to `sessionMsgs_goodCase`). The form's reading reads the case through `inputOn` alone,
so on a kept play the two readings are one, and `ReachedAccepts` at the kept case is `ReachedAccepts` at `C` on the
plays it keeps (`reachedAccepts_goodCase`).
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- **The play's blobs don't collide with the files `F₀`.** -/
def NoCollide (F₀ : Array ByteArray) (o : Verity.Exchange) : Prop := ¬ BlobsCollide o.sent.blobs F₀

theorem heldOfAt_goodCase {A : Live₂} {C : CaseOf} {F₀ : Array ByteArray} (h : HeldOfAtOr A C F₀) :
    HeldOfAt A (goodCase C (NoCollide F₀)) := by
  intro o i it hA hi hit
  obtain ⟨hA, hG⟩ := formAccepts_goodCase.1 hA
  rw [inputOn_goodCase hG] at hi
  exact (h o i it hA hi hit).resolve_right hG

theorem portsDistinctAt_goodCase {C : CaseOf} {G : Verity.Exchange → Prop} (h : PortsDistinctAt C) :
    PortsDistinctAt (goodCase C G) := by
  intro o i it hA hi hit
  obtain ⟨hA, hG⟩ := formAccepts_goodCase.1 hA
  rw [inputOn_goodCase hG] at hi
  exact h o i it hA hi hit

theorem stmtsRead_goodCase {A : Live₂} {C : CaseOf} {G : Verity.Exchange → Prop} {S : SessionReading}
    (h : StmtsRead A C S) : StmtsRead A (goodCase C G) S := by
  intro o i it hA hi hit
  obtain ⟨hA, hG⟩ := formAccepts_goodCase.1 hA
  rw [inputOn_goodCase hG] at hi
  exact h o i it hA hi hit

theorem roundsHeldAt_goodCase {A : Live₂} {C : CaseOf} {G : Verity.Exchange → Prop} {S : SessionReading}
    (h : RoundsHeldAt A C S) : RoundsHeldAt A (goodCase C G) S :=
  fun o m₂ hA hs => h o m₂ (formAccepts_goodCase.1 hA).1 hs

theorem scopeHeldAt_goodCase {A : Live₂} {C : CaseOf} {G : Verity.Exchange → Prop} {S : SessionReading}
    (h : ScopeHeldAt A C S) : ScopeHeldAt A (goodCase C G) S :=
  fun o m₂ hA hs => h o m₂ (formAccepts_goodCase.1 hA).1 hs

theorem pinHeldAt_goodCase {A : Live₂} {C : CaseOf} {G : Verity.Exchange → Prop} {S : SessionReading}
    (h : PinHeldAt A C S) : PinHeldAt A (goodCase C G) S :=
  fun o m₂ hA hs => h o m₂ (formAccepts_goodCase.1 hA).1 hs

theorem sessionsRun_goodCase {A : Live₂} {C : CaseOf} {G : Verity.Exchange → Prop} {S : SessionReading}
    {P' : Verity.Exchange → ℕ → Claims.Prover Unit} (h : SessionsRun A C S P') : SessionsRun A (goodCase C G) S P' :=
  fun o m₂ hA hs => h o m₂ (formAccepts_goodCase.1 hA).1 hs

theorem sessionMsgs_goodCase {A : Live₂} {C : CaseOf} {G : Verity.Exchange → Prop} {S : SessionReading}
    {P' : Verity.Exchange → ℕ → Claims.Prover Unit} (h : SessionMsgs A C S P') : SessionMsgs A (goodCase C G) S P' := by
  intro o i it x m₁ m₂ hA hi hit hx hm₁ hs
  obtain ⟨hA, hG⟩ := formAccepts_goodCase.1 hA
  rw [inputOn_goodCase hG] at hi
  exact h o i it x m₁ m₂ hA hi hit hx hm₁ hs

/-- The form's reading reads a play's statement from its run of record's input alone. -/
theorem formReading_stmt (A : Live₂) {C C' : CaseOf} (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) {o : Verity.Exchange}
    (h : inputOn C' o = inputOn C o) :
    (formReading A C' S dev).stmt o = (formReading A C S dev).stmt o := by
  simp only [formReading, h]

/-- The form's reading reads a play's messages from its run of record's input alone. -/
theorem formReading_msg (A : Live₂) {C C' : CaseOf} (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) {o : Verity.Exchange}
    (h : inputOn C' o = inputOn C o) (n : ℕ) :
    (formReading A C' S dev).msg o n = (formReading A C S dev).msg o n := by
  simp only [formReading, h]

/-- **`ReachedAccepts` at the case, on the plays the kept case keeps.** -/
theorem reachedAccepts_goodCase {A : Live₂} {x : Msg} {C : CaseOf} {G : Verity.Exchange → Prop}
    {S : SessionReading} {dev : List Verity.Release → ℕ → Option FlockDeploy.Coin}
    (h : ReachedAccepts A x (formReading A (goodCase C G) S dev) (FormAccepts (goodCase C G))) :
    ReachedAccepts A x (formReading A C S dev) (fun o => FormAccepts C o ∧ G o) := by
  intro o hAG hs Ω P ω r hc hm
  have hin : inputOn (goodCase C G) o = inputOn C o := inputOn_goodCase hAG.2
  refine h o (formAccepts_goodCase.2 hAG) ((formReading_stmt A S dev hin).trans hs) Ω P ω r
    (fun i hr c hc' => (formReading_agree A S dev hin i c (r i)).2
      (hc i hr c ((formReading_coin A S dev hin i).symm.trans hc'))) (fun i hi => ?_)
  rw [formReading_msg A S dev hin]
  exact hm i hi

end Verity.SampledTwo
