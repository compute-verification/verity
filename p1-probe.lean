import Proofs.Flock.C3.Compose.SrcOKFamily
import Proofs.Flock.C3.Compose.AtHoloHeld
import Proofs.Flock.C3.Compose.DecodeFix

/-! P1 probe (dropped once it passes): `PointsFromHistory` at the consumers' free source. -/

namespace FlockC3.P1Probe

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockC3.Reps (repsOfVals UnitVals)
open FlockC3.HoloValues (repAt)
open Flock.Guarantees (holoCTs holoKinds)

/-- The run `u` with every link point moved by one. -/
def shiftPts {e : ExecHolo} (u : C3Run e) : C3Run e := { u with pts := fun i j => u.pts i j + 1 }

theorem length_flatMap_pair {α β : Type} (l : List α) (f g : α → β) :
    (l.flatMap fun a => [f a, g a]).length = 2 * l.length := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    show ([f a, g a] ++ l.flatMap fun a => [f a, g a]).length = 2 * (l.length + 1)
    rw [List.length_append, ih]
    simp only [List.length_cons, List.length_nil]
    omega

theorem extraClaims_length (S : FlockSoundness.Model.Statement) (ptLocal mPts : ℕ) (p : Fin 2 → Fin mPts → GF128) :
    (extraClaims S ptLocal mPts p).length = 2 * S.regions.length := by
  unfold extraClaims
  exact length_flatMap_pair _ _ _

/-- A tie reads the points only through the reps' extra claims, which are `[]` at a statement with no regions. -/
theorem tie_shiftPts {e : ExecHolo} (u : C3Run e) (h : (e.cls.pubStatement e.regs).regions = []) :
    tie (shiftPts u) = shiftPts (tie u) := by
  have hx : extraClaims (e.cls.pubStatement e.regs) e.cls.ptLocal e.cls.mPts (shiftPts u).pts =
      extraClaims (e.cls.pubStatement e.regs) e.cls.ptLocal e.cls.mPts u.pts := by
    simp [extraClaims, h]
  have hr : repAt e (shiftPts u) = repAt e u := funext fun i => by simp only [repAt, hx]
  unfold tie tieLig tieOpening tieSparse tieReps
  rw [hr]
  rfl

/-- What `OpeningReadAt` reads of a run's steps, past its cap and points, is the same at the shifted run. -/
theorem afterSparse_runSteps_shift {e : ExecHolo} (u : C3Run e) :
    FlockC3.Opening.afterSparse (shiftPts u) (runSteps (shiftPts u)) =
      FlockC3.Opening.afterSparse u (runSteps u) := by
  unfold FlockC3.Opening.afterSparse
  rw [tape_runSteps, tape_runSteps] <;> rfl

theorem openingReadAt_shift {st : Flock.Stmt} {hp : HoloParams} (u : C3Run (holoAt st hp)) :
    OpeningReadAt (runSteps (shiftPts u)) (shiftPts u) ↔ OpeningReadAt (runSteps u) u := by
  unfold OpeningReadAt
  rw [afterSparse_runSteps_shift] <;> rfl

section src

variable {nT : ℕ} {T : Fin nT → Type} {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n}
  {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T (holoCTs st hp) st hp

/-- The consumers' source has the free `stp`'s points: no wrapper sets them. -/
theorem pts_tiedSrc_repsOfVals
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) :
    (tiedSrc (repsOfVals stp sh rep vals) R ω t Lb).pts = (stp R ω t Lb).pts := rfl

/-- `repsOfVals` reads the points only through the extra claims' length, which they don't change. -/
theorem repsOfVals_shift
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) :
    repsOfVals (fun R ω t Lb => shiftPts (stp R ω t Lb)) sh rep vals R ω t Lb =
      shiftPts (repsOfVals stp sh rep vals R ω t Lb) := by
  first
  | (simp only [repsOfVals, shiftPts, extraClaims_length]; done)
  | exact c3Run_ext rfl rfl (funext fun i => by simp only [repsOfVals, shiftPts, extraClaims_length]) rfl rfl rfl
      rfl rfl rfl rfl rfl rfl

/-- **The shifted source is the source with its points shifted**, at every input, at a statement with no regions. -/
theorem tiedSrc_repsOfVals_shift
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals)
    (h : ((holoAt st hp).cls.pubStatement (holoAt st hp).regs).regions = [])
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) :
    tiedSrc (repsOfVals (fun R ω t Lb => shiftPts (stp R ω t Lb)) sh rep vals) R ω t Lb =
      shiftPts (tiedSrc (repsOfVals stp sh rep vals) R ω t Lb) := by
  show tie _ = shiftPts (tie _)
  rw [repsOfVals_shift stp sh rep vals, tie_shiftPts _ h]

/-- **The consumers' conclusion gives the premise**: `StageCoins`' head conjunct is `PointsFromHistory` at `holoMap`
(`pointsEncoded_holoMap`). -/
theorem pointsFromHistory_of_stageCoins {ish : RecVStar.Inner} {tbl : String} {bc : Cm → String → Nat → List UInt8}
    {t : List (Cm × (ℐ).Coin)} {u : C3Run (holoAt st hp)}
    (h : StageCoins (CT := holoCTs st hp) (t.map Prod.snd) u) : FlockC3.HPort.PointsFromHistory ish tbl bc t u := by
  intro e he
  obtain ⟨_, -, hh, -⟩ := h
  cases t with
  | nil => simp at he
  | cons e' t' =>
    simp only [List.getElem?_cons_zero, Option.some.injEq] at he
    subst he
    have hr : 0 < 1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length := by omega
    simp only [List.map_cons, FlockVBridge.Family.coinSteps, hr, ↓reduceDIte, List.singleton_append,
      List.head?_cons, Option.some.injEq] at hh
    have hc := (pointsEncoded_holoMap st hp Cm ish tbl bc hr e'.2).trans hh
    exact (eq_of_heq (Sigma.mk.inj hc).2).symm

theorem not_pointsFromHistory_shift {ish : RecVStar.Inner} {tbl : String} {bc : Cm → String → Nat → List UInt8}
    {t : List (Cm × (ℐ).Coin)} {u : C3Run (holoAt st hp)} (ht : t ≠ []) (hm : 0 < (holoAt st hp).cls.mPts)
    (h : FlockC3.HPort.PointsFromHistory ish tbl bc t u) :
    ¬ FlockC3.HPort.PointsFromHistory ish tbl bc t (shiftPts u) := by
  intro h'
  obtain ⟨e, he⟩ : ∃ e, t[0]? = some e := by
    cases t with
    | nil => exact absurd rfl ht
    | cons e _ => exact ⟨e, rfl⟩
  have h1 := congrFun (congrFun (h e he) 0) ⟨0, hm⟩
  have h2 := congrFun (congrFun (h' e he) 0) ⟨0, hm⟩
  have h3 : u.pts 0 ⟨0, hm⟩ + 1 = u.pts 0 ⟨0, hm⟩ := h2.trans h1.symm
  first
  | simpa using h3
  | exact absurd (add_eq_left.mp h3) one_ne_zero
  | exact absurd (add_right_eq_self.mp h3) one_ne_zero

/-- **The counterexample**: at a statement with no regions and a link point, wherever a consumer's conclusion holds at
the source `stp` at an accepted outer proof, it fails there at the source with every point shifted, which differs from
the first only in its points (`tiedSrc_repsOfVals_shift`). -/
theorem stageCoins_fails_shift (ish : RecVStar.Inner) (tbl : String) (bc : Cm → String → Nat → List UInt8)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    (hreg : ((holoAt st hp).cls.pubStatement (holoAt st hp).regs).regions = [])
    (hm : 0 < (holoAt st hp).cls.mPts)
    (hc : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      StageCoins (CT := holoCTs st hp) (t.map Prod.snd) u)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hA : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) (ht : t ≠ []) :
    ¬ StageCoins (CT := holoCTs st hp) (t.map Prod.snd)
      (tiedSrc (repsOfVals (fun R ω t Lb => shiftPts (stp R ω t Lb)) sh rep vals) R ω t
        (layers zs dg N k Rw R ω t R₂ τ)) := by
  rw [tiedSrc_repsOfVals_shift stp sh rep vals hreg]
  intro h'
  exact not_pointsFromHistory_shift (ish := ish) (tbl := tbl) (bc := bc) ht hm
    (pointsFromHistory_of_stageCoins (hc R ω t R₂ τ o hA)) (pointsFromHistory_of_stageCoins h')

/-- **The fix**: the source whose points are its definition, the history's first coin under `holoMap`. -/
def histPts (ish : RecVStar.Inner) (tbl : String) (bc : Cm → String → Nat → List UInt8)
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) :
    Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp) := fun R ω t Lb =>
  { stp R ω t Lb with
    pts := (t[0]?.map fun e => holoPts (holoAt st hp).cls.mPts
      ((Flock.Guarantees.holoMap st hp Cm ish tbl bc).points e.2)).getD (stp R ω t Lb).pts }

theorem pts_histPts (ish : RecVStar.Inner) (tbl : String) (bc : Cm → String → Nat → List UInt8)
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) :
    (tiedSrc (repsOfVals (histPts ish tbl bc stp) sh rep vals) R ω t Lb).pts =
      (t[0]?.map fun e => holoPts (holoAt st hp).cls.mPts
        ((Flock.Guarantees.holoMap st hp Cm ish tbl bc).points e.2)).getD (stp R ω t Lb).pts := rfl

/-- `hpts` at `histPts`' source, 0 premises. -/
theorem pointsFromHistory_histPts {ish : RecVStar.Inner} {tbl : String} {bc : Cm → String → Nat → List UInt8}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals} :
    AtSrc zs dg N k Rw (tiedSrc (repsOfVals (histPts ish tbl bc stp) sh rep vals)) fun t u =>
      FlockC3.HPort.PointsFromHistory ish tbl bc t u := by
  intro R ω t R₂ τ o _ e he
  rw [pts_histPts, he] <;> rfl

end src

end

end FlockC3.P1Probe

#print axioms FlockC3.P1Probe.tie_shiftPts
#print axioms FlockC3.P1Probe.openingReadAt_shift
#print axioms FlockC3.P1Probe.pts_tiedSrc_repsOfVals
#print axioms FlockC3.P1Probe.tiedSrc_repsOfVals_shift
#print axioms FlockC3.P1Probe.pointsFromHistory_of_stageCoins
#print axioms FlockC3.P1Probe.not_pointsFromHistory_shift
#print axioms FlockC3.P1Probe.stageCoins_fails_shift
#print axioms FlockC3.P1Probe.pointsFromHistory_histPts
