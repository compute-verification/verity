import Proofs.Flock.Bridge.HiddenLayer
import Proofs.Flock.EndToEndHidden.Statement

/-!
# `EndToEndHiddenModel` at the verifier that accepts everything, and its toy

`EndToEndHiddenModelAll` is `EndToEndHiddenModel` with its event's acceptance replaced by `True`: the statement at the
verifier that accepts everything, which `EndToEndHiddenModel`'s nontrivial witness refutes
(`Flock.Bridge.endToEndHiddenModel_nontrivial`).

The toy is `HiddenLayer`'s at the draw file that reads no draw (`toyDj`, `null` at every draw): `Draw.UnitDraw.ofJson`
refuses `null`, so no table is set up at any draw and each is the reference setup's refused table, which no witness
satisfies (`toy_tab_refused`). So the toy's tables lower soundly to any decoder, and the law that draws the one unit
(`toyLaw`) has no sampling term.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

open Classical in
/-- **`EndToEndHiddenModel` at the verifier that accepts everything**: its statement with the event's acceptance
`o.1 = true` replaced by `True`. -/
def EndToEndHiddenModelAll : Prop :=
  ∀ (n : ℕ) (Reg : Type) (I : Inputs) (mPts : ℕ)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (C : FlockSoundness.Audit.Circuit) (P : FlockSoundness.Audit.Partition C n)
    (decode : Discharge.ZkSession.DecoderZK (C := C) (planZHJ (Reg := Reg) dj y₀) (tabZHJ dj y₀))
    (Dh : Type) (Hc : List UInt8 → Dh) (vb : ValueBindingZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) decode Hc)
    (_hlow : Discharge.ZkSession.LoweringSoundZK execArith P (planZHJ dj y₀) (tabZHJ dj y₀) decode)
    (_hown : ∀ u, P.ReadsOwn u)
    (canon : Fin C.N → vb.Opening → Prop) (_hcan : OpeningsCanon execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb canon)
    (isRead : vb.Pos → Prop) (wo : vb.Pos → vb.Opening)
    (L : Law n)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (K₀ : ℕ)
    (σ : Strategy (audit L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)))),
    prob (fun _ => True ∧
        K₀ ≤ (wrongRegZc execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb canon (reg σ)
            (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw) ∪
          rowsWrongZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb isRead wo
            (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀))) σ ≤
      L.miss K₀ + ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        (linkPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb k Rw M ρ (reg σ) (cont σ) +
          rowsPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb isRead wo k ρ (reg σ) (cont σ))

end Flock.Guarantees

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.ZkSession FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden

/-- **The law that draws the one unit**: one coin, which draws it. -/
def toyLaw : Law 1 := { Ω := Unit, draw := fun _ => Finset.univ }

/-- **The draw file that reads no draw**: `null` at every draw. -/
def toyDj : Finset (Fin 1) → Lean.Json := fun _ => Lean.Json.null

/-- `Draw.UnitDraw.ofJson` refuses `null`: a draw is an object. -/
theorem unitDraw_null :
    Flock.Draw.UnitDraw.ofJson Lean.Json.null = Except.error "U2: unit_draw is not an object" := rfl

/-- Every table's part of the draw `null` is `null` (`partJson` leaves a draw it can't read as it is). -/
theorem drawJ_null (I : Inputs) (j : ℕ) : drawJ I Lean.Json.null j = Lean.Json.null := by
  simp [drawJ, FlockSoundness.Discharge.DecodeZLJ.partJson, unitDraw_null]

/-- **The toy's every table is the reference setup's refused one**: at the draw file `toyDj`, no draw has a table
`verify --zk` sets up (`unitDraw_null`), so `tableZHJ` puts `y₀.refused` in each table's place. -/
theorem toy_tab_refused {I : Inputs} {mPts : ℕ} {Reg : Type} {S₀ : Finset (Fin 1)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (S : Finset (Fin 1)) (R : Reg) (u : Fin 1) :
    tabOfZ (planZHJ (Reg := Reg) toyDj y₀) (tabZHJ toyDj y₀) S R u = y₀.refused := by
  rw [tabOfZHJ_eq]
  unfold tableZHJ
  split
  · rename_i h
    obtain ⟨y⟩ := h
    obtain ⟨D, hD⟩ : ∃ D, Flock.Draw.UnitDraw.ofJson (drawJ I Lean.Json.null _) = Except.ok D := ⟨_, y.x.hD⟩
    rw [drawJ_null, unitDraw_null] at hD
    cases hD
  · rfl

/-- **The toy's tables lower soundly to the decoder that reads the registered bits**: no message satisfies a refused
table (`toy_tab_refused`, `DrawSetupH.refused_unsat`). -/
theorem toy_lowering_refused {I : Inputs} {mPts : ℕ} {Reg : Type} {S₀ : Finset (Fin 1)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) :
    LoweringSoundZK execArith toyPartition (planZHJ (Reg := Reg) toyDj y₀) (tabZHJ toyDj y₀)
      (toyDecode (planZHJ (Reg := Reg) toyDj y₀) (tabZHJ toyDj y₀)) :=
  toy_lowering execArith _ _ fun S R u => by
    rw [toy_tab_refused y₀ S R u]
    exact fun M => y₀.x.refused_unsat ∅ _

end Flock.Bridge
