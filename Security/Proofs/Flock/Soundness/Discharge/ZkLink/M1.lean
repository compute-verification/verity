import Proofs.Flock.Soundness.Discharge.ZkLink.TablesAt
import Proofs.Flock.Soundness.Discharge.ZkSession.Numbers

/-!
# The live `--zk` tables are at C-Flock ZK's parameters

Every `--zk` table `tableAtZK` builds meets `TabZK.M1`, from what `verify --zk` checks at setup:

* `Zk.schedule m`'s schedule is `m1Schedule m` and its `t_pad` is `m1Pad m`, at every `m` it accepts (`schedule_m1`,
  from `zkM1Facts_all`, checked by `decide +kernel` at each `m`);
* setupH's `checkInRange` puts `k_log`, the statement's region count and `m_pts` in `Stmt.InRange` (`setupH_spec`);
* the inner code's length is `8·next_pow2(len + K_PAD)`, at most `2^13` (`innerN_pow`), so its radius is at least `7/16`
  (`relUDR_ge_m1`); 168 queries (`Zk.Q`).

A draw's table carries setupH's regions (`table_M1`), a refused draw's `contra`'s two (`refused_M1`), so both are in
range; `tableAtZK_M1` is either.
-/

namespace FlockSoundness.Discharge.ZkLink

open Flock FlockLevel3 Model Accounting Refine Exec Integrate ZkSession ZkExec
open scoped NNReal

local notation "𝔽" => FlockLevel3.GF128

/-- What `TabZK.M1` needs of `Zk.schedule m`'s schedule and `t_pad`, as a `Bool` (`true` where it refuses `m`). -/
def zkM1Facts (m : ℕ) : Bool :=
  match Zk.schedule m with
  | .ok (s, t) => decide (schedOf s = m1Schedule m) && decide (t = m1Pad m)
  | .error _ => true

/-- **The m check**: at every `m` `Zk.schedule` accepts (25 to 35), its schedule and `t_pad` are C-Flock ZK's, by the
kernel's `decide`. -/
theorem zkM1Facts_all : ∀ m, 25 ≤ m → m ≤ 35 → zkM1Facts m = true := by
  intro m h1 h2; interval_cases m <;> decide +kernel

/-- **`Zk.schedule`'s schedule is `m1Schedule`, its `t_pad` `m1Pad`.** -/
theorem schedule_m1 {m : ℕ} {sch : Flock.Schedule} {tPad : ℕ} (h : Zk.schedule m = .ok (sch, tPad)) :
    schedOf sch = m1Schedule m ∧ tPad = m1Pad m := by
  obtain ⟨h1, h2, -⟩ := schedule_spec h
  have hf := zkM1Facts_all m h1 h2
  unfold zkM1Facts at hf
  rw [h] at hf
  simpa only [Bool.and_eq_true, decide_eq_true_eq] using hf

variable {n : ℕ} {I : Inputs} {mPts : ℕ}

namespace DrawSetupZK

variable {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZK I mPts S d)

/-- **The range setupH checked** (`checkInRange`): `k_log ≤ 27`, at most 1024 regions and `m_pts ≤ 64`. -/
theorem inRange (ht : I.tags.typed = false) :
    y.x.st.c.kLog ≤ 27 ∧ y.x.st.regions.size ≤ 1024 ∧ mPts ≤ 64 := by
  obtain ⟨hk, hr, hp⟩ :=
    (checkInRange_ok _ _ _).1 (setupH_spec ((zkTags_typed I).trans ht) y.x.h).2.2.2.2.2.2
  exact ⟨hk, hr, y.x.hmPts ▸ hp⟩

/-- **A `--zk` table of a draw `verify --zk` sets up is at C-Flock ZK's parameters**, at any regions of at most 1024. -/
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
  refine ⟨y.x.st.m, h1, h2, hsch, hpad, rfl, ⟨hk, hR, hp⟩, ?_, rfl, relUDR_ge_m1 y.innerDom h8⟩
  show innerN (layOf y.x.st y.sch) ≤ 2 ^ 13
  rw [hb]
  exact Nat.pow_le_pow_right (by norm_num) hb13

/-- **The draw's `--zk` table is at C-Flock ZK's parameters**: setupH's regions, at most 1024 (`checkInRange`). -/
theorem table_M1 (ht : I.tags.typed = false) : y.table.M1 :=
  y.tabAt_M1 ht _ _ (by
    have := (y.inRange ht).2.1
    simpa [regionsOf] using this)

/-- **A refused draw's `--zk` table is at C-Flock ZK's parameters**: `contra`'s two regions. -/
theorem refused_M1 (ht : I.tags.typed = false) : y.refused.M1 :=
  y.tabAt_M1 ht _ _ (by simp [DrawSetup.contra])

end DrawSetupZK

/-- **Every `--zk` table `tableAtZK` builds is at C-Flock ZK's parameters**: the draw's table, or the refused one. -/
theorem tableAtZK_M1 (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (S : Finset (Fin n)) : (tableAtZK dj y₀ S).M1 := by
  unfold tableAtZK
  split
  · exact DrawSetupZK.table_M1 _ ht
  · exact y₀.refused_M1 ht

end FlockSoundness.Discharge.ZkLink
