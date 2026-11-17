module

public import Proofs.Flock.Recursive.Flock
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.Registered.Binding

/-! A session's registered read off the table's leaf, priced once per audit by a finder of the recursive audit (`prob_leafColl_le`). -/

@[expose] public section

namespace FlockSoundness.Registered

open Game Refine Binding Audit

variable {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : HmRowsX lay}

noncomputable def ReadsX.leafCands (rd : ReadsX lay rs) (R : Reg) (path : lay.Pos → ByteArray) (ps : List lay.Pos) :
    List (Prop × (List UInt8 × List UInt8)) :=
  ps.map fun p => (rd.isRead p ∧ (rd.openings R path p).Conflict, (rd.openings R path p).pair)

theorem ReadsX.leafCands_spec {rd : ReadsX lay rs} {R : Reg} {path : lay.Pos → ByteArray} {ps : List lay.Pos}
    {c : Prop × (List UInt8 × List UInt8)} (hc : c ∈ rd.leafCands R path ps) (h : c.1) :
    Hash.Collision H512 c.2.1 c.2.2 := by
  obtain ⟨p, -, rfl⟩ := List.mem_map.1 hc
  have hw : rd.isRead p ∧ (rd.openings R path p).Conflict := h
  exact TwoOpenings.collides _ hw.2

theorem ReadsX.leafCands_mem {rd : ReadsX lay rs} {R : Reg} {path : lay.Pos → ByteArray} {ps : List lay.Pos}
    {p : lay.Pos} (hps : p ∈ ps) (hp : rd.isRead p) (h : (rd.openings R path p).Conflict) :
    ∃ c ∈ rd.leafCands R path ps, c.1 :=
  ⟨_, List.mem_map.2 ⟨p, hps, rfl⟩, hp, h⟩

end FlockSoundness.Registered

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open scoped ENNReal

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
variable {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type}

def LeafColl (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (R : Reg) (ω : L.Ω)
    (t : List (Cm × I.Coin)) (R₂ : Reg₂) (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript) :
    Prop :=
  ∃ j p, (zs R ω t R₂ j).rd.isRead p ∧ ((zs R ω t R₂ j).rd.openings (o j).1
    (regPathsHJ (zs R ω t R₂ j).I (Vs j).c (zs R ω t R₂ j).pub (o j).2.2 (zs R ω t R₂ j).rd.port) p).Conflict

noncomputable def leafCands (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (R : Reg)
    (ω : L.Ω) (t : List (Cm × I.Coin)) (R₂ : Reg₂)
    (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript) :
    List (Prop × (List UInt8 × List UInt8)) :=
  (List.finRange S).flatMap fun j => (zs R ω t R₂ j).rd.leafCands (o j).1
    (regPathsHJ (zs R ω t R₂ j).I (Vs j).c (zs R ω t R₂ j).pub (o j).2.2 (zs R ω t R₂ j).rd.port)
    (zs R ω t R₂ j).posList

theorem leafCands_spec {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {R : Reg}
    {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
    {o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript} {c : Prop × (List UInt8 × List UInt8)}
    (hc : c ∈ leafCands zs R ω t R₂ o) (h : c.1) : Hash.Collision H512 c.2.1 c.2.2 := by
  obtain ⟨j, -, hc⟩ := List.mem_flatMap.1 hc
  exact Registered.ReadsX.leafCands_spec hc h

theorem LeafColl.cand {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {R : Reg}
    {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
    {o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript} (h : LeafColl zs R ω t R₂ o) :
    ∃ c ∈ leafCands zs R ω t R₂ o, c.1 := by
  obtain ⟨j, p, hp, h⟩ := h
  obtain ⟨c, hc, hc1⟩ := Registered.ReadsX.leafCands_mem ((zs R ω t R₂ j).mem_posList p) hp h
  exact ⟨c, List.mem_flatMap.2 ⟨j, List.mem_finRange j, hc⟩, hc1⟩

/-- Every candidate is computed from what the run holds: the session's record, and the registration and transcript its
outcome records. -/
noncomputable def leafFinder (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) :
    Finder Pair :=
  oneRun _ σ (onePick fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
    leafCands zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 fun j => (o.2.2.2.2 j).2)

theorem prob_leafColl_le (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    {qL : ℝ} (hL : (leafFinder zs dg N σ).CR H512 qL) :
    prob (fun o => LeafColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 fun j => (o.2.2.2.2 j).2)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤ ENNReal.ofReal (qL ^ 2 / 2 ^ (513 : ℝ)) := by
  have hev : (prob (fun o => LeafColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 fun j => (o.2.2.2.2 j).2)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ).toReal ≤ qL ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    exact (expect_one_le H512 _ σ (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
      LeafColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 fun j => (o.2.2.2.2 j).2)
      (fun o => leafCands zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 fun j => (o.2.2.2.2 j).2)
      (fun _ h => h.cand) fun _ _ hc hp => leafCands_spec hc hp).trans hL.le
  exact (ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (by positivity)).2 hev

end FlockSoundness.Discharge.Recursion
