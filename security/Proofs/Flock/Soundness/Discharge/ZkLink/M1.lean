module

public import Proofs.Flock.Soundness.Discharge.ZkLink.TablesAt
public import Proofs.Flock.Soundness.Discharge.ZkSession.Numbers
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
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
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.Soundness.Discharge.ZkLink.Sched

/-! The live `--zk` tables are at Flock ZK's parameters (`TabZK.M1`). -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink

open Flock FlockLevel3 Model Accounting Refine Exec Integrate ZkSession ZkExec
open scoped NNReal

local notation "𝔽" => FlockLevel3.GF128

variable {wideLeaf : Bool}

/-- `true` also where `Zk.schedule` refuses `m`. -/
def zkM1Facts (wideLeaf : Bool) (m : ℕ) : Bool :=
  match Zk.schedule wideLeaf m with
  | .ok (s, t) => decide (schedOf s = m1Schedule wideLeaf m) && decide (t = m1Pad wideLeaf m)
  | .error _ => true

theorem zkM1Facts_all : ∀ m, 25 ≤ m → m ≤ 35 → zkM1Facts wideLeaf m = true := by
  intro m h1 h2; cases wideLeaf <;> interval_cases m <;> decide +kernel

theorem schedule_m1 {m : ℕ} {sch : Flock.Schedule} {tPad : ℕ} (h : Zk.schedule wideLeaf m = .ok (sch, tPad)) :
    schedOf sch = m1Schedule wideLeaf m ∧ tPad = m1Pad wideLeaf m := by
  obtain ⟨h1, h2, -⟩ := schedule_spec h
  have hf := zkM1Facts_all (wideLeaf := wideLeaf) m h1 h2
  unfold zkM1Facts at hf
  rw [h] at hf
  simpa only [Bool.and_eq_true, decide_eq_true_eq] using hf

variable {n : ℕ} {I : Inputs} {mPts : ℕ}

namespace SampleSetupZK

variable {S : Finset (Fin n)} {d : Lean.Json} (y : SampleSetupZK I mPts S d)

theorem inRange (ht : I.tags.typed = false) :
    y.x.st.c.kLog ≤ 27 ∧ y.x.st.regions.size ≤ 1024 ∧ mPts ≤ 64 := by
  obtain ⟨hk, hr, hp⟩ :=
    (checkInRange_ok _ _ _).1 (setupH_spec ((zkTags_typed I).trans ht) y.x.h).2.2.2.2.2.2
  exact ⟨hk, hr, y.x.hmPts ▸ hp⟩

theorem tabAt_M1 (ht : I.tags.typed = false) (regs : List (Model.Region y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) (hR : regs.length ≤ 1024) :
    (y.tabAt regs hregs).M1 := by
  obtain ⟨hk, -, hp⟩ := y.inRange ht
  obtain ⟨h1, h2, -⟩ := schedule_spec y.hz
  obtain ⟨hsch, hpad⟩ := schedule_m1 y.hz
  have h0 : y.sch.k0 ≤ 35 := by
    have := y.hshape.2.2.1
    have hm := y.hmsch
    simp only [schedOf] at this hm
    omega
  obtain ⟨b, hb, hb13⟩ := innerN_pow (layOf y.x.st y.sch) (layLen_le h2 hk h0)
  have h8 : 8 * ((layOf y.x.st y.sch).len + Zk.K_PAD) ≤ innerN (layOf y.x.st y.sch) := by
    have := Nat.le_nextPowerOfTwo ((layOf y.x.st y.sch).len + Zk.K_PAD)
    rw [ZkExec.innerN, Zk.INNER_LOG_INV_RATE, show (2 : ℕ) ^ 3 = 8 by norm_num]
    omega
  have : NeZero ((layOf y.x.st y.sch).len + Zk.K_PAD) := ⟨by simp [Zk.K_PAD]⟩
  refine ⟨y.x.st.wideLeaf, y.x.st.m, h1, h2, hsch, hpad, rfl, ⟨hk, hR, hp⟩, ?_, rfl, relUDR_ge_m1 y.innerDom h8⟩
  show innerN (layOf y.x.st y.sch) ≤ 2 ^ 13
  rw [hb]
  exact Nat.pow_le_pow_right (by norm_num) hb13

theorem table_M1 (ht : I.tags.typed = false) : y.table.M1 :=
  y.tabAt_M1 ht _ _ (by
    have := (y.inRange ht).2.1
    simpa [regionsOf] using this)

theorem refused_M1 (ht : I.tags.typed = false) : y.refused.M1 :=
  y.tabAt_M1 ht _ _ (by simp [SampleSetup.contra])

end SampleSetupZK

theorem tableAtZK_M1 (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : SampleSetupZK I mPts S₀ (dj S₀)) (S : Finset (Fin n)) : (tableAtZK dj y₀ S).M1 := by
  unfold tableAtZK
  split
  · exact SampleSetupZK.table_M1 _ ht
  · exact y₀.refused_M1 ht

end FlockSoundness.Discharge.ZkLink
