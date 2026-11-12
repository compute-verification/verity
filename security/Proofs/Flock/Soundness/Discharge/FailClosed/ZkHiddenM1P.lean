module

public import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHiddenM1
public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJP
public import Proofs.Flock.Soundness.Discharge.Hidden.SpecP

/-!
# The hidden-output plan at the verifier's public inputs is at Flock ZK's parameters (`planZHJP_M1W`)

`ZkHiddenM1`'s `planZHJ_M1W` at the plan `verify --zk` sets up at the verifier's own public input words `own`
(`planZHJP`): each table a sample sets up, or the reference setup's refused one, is at `M1W` (`tabAt_M1W`, read off
`setupHidden`'s range check at `own`, `Hidden.setupHidden_specP`), so every table's level-0 rate is at most `rateM1 k`
(`rateZ_planZHJP_le_m1`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Flock FlockLevel3 Model Accounting Refine ZkSession ZkLink ZkExec Composed

variable {n mPts : ℕ}

namespace DrawSetupZHJP

variable {I' : Inputs} {own : Option ByteArray} {S : Finset (Fin n)} {d : Lean.Json}
  (y : DrawSetupZHJP I' own mPts S d)

/-- **The range setupHidden checked at `own`** (`checkInRange`): `k_log ≤ 27`, at most 1024 regions and
`m_pts ≤ 64`. -/
theorem inRange : y.x.st.c.kLog ≤ 27 ∧ y.x.st.regions.size ≤ 1024 ∧ mPts ≤ 64 := by
  obtain ⟨pi, s⟩ := Hidden.setupHidden_specP y.x.h
  obtain ⟨hk, hr, hp⟩ := (checkInRange_ok _ _ _).1 s.inRange
  have h1 := y.x.wf.2.mPts
  have h2 := y.x.hmPts
  exact ⟨hk, hr, by omega⟩

/-- **A hidden-output `--zk` table `verify --zk` sets up at `own` is at Flock ZK's parameters**, at any regions of
at most 1026 (`DrawSetupZHJ.tabAt_M1W`'s twin). -/
theorem tabAt_M1W (regs : List (Model.Region y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) (hR : regs.length ≤ 1026) :
    (y.tabAt regs hregs).M1W := by
  obtain ⟨hk, -, hp⟩ := y.inRange
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

/-- **The table's hidden-output `--zk` table at `own` is at Flock ZK's parameters**: setupHidden's regions, at most
1024. -/
theorem table_M1W : y.table.M1W :=
  y.tabAt_M1W _ _ (by
    have := y.inRange.2.1
    rw [FlockVBridge.Family.length_regionsOf]
    omega)

/-- **A refused hidden-output `--zk` table at `own` is at Flock ZK's parameters**: setupHidden's regions and
`contra`'s two, at most 1026. -/
theorem refused_M1W : y.refused.M1W :=
  y.tabAt_M1W _ _ (by
    have := y.inRange.2.1
    have hc : (Integrate.DrawSetup.contra y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)).length = 2 := by
      simp [Integrate.DrawSetup.contra]
    rw [List.length_append, FlockVBridge.Family.length_regionsOf, hc]
    omega)

end DrawSetupZHJP

variable {I : Inputs} {own : Option ByteArray} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀) {Reg : Type}

/-- **Every table `tableZHJP` builds is at Flock ZK's parameters**: table `j`'s own, or the reference setup's refused
one. -/
theorem tableZHJP_M1W (S : Finset (Fin n)) (j : Fin (nTab I)) : (tableZHJP dj y₀ S j).M1W := by
  unfold tableZHJP
  split
  · exact DrawSetupZHJP.table_M1W _
  · exact y₀.refused_M1W

theorem planZHJP_get (S : Finset (Fin n)) (R : Reg) (i : Fin (planZHJP (Reg := Reg) dj y₀ S R).length) :
    (planZHJP (Reg := Reg) dj y₀ S R).get i = tableZHJP dj y₀ S (Fin.cast (planZHJP_length dj y₀ S R) i) :=
  List.get_ofFn _ _

/-- **Every table of the hidden-output plan at `own` is at Flock ZK's parameters** (`tableZHJP_M1W`). -/
theorem planZHJP_M1W :
    ∀ S (R : Reg) (j : Fin (planZHJP (Reg := Reg) dj y₀ S R).length), ((planZHJP dj y₀ S R).get j).M1W := by
  intro S R j
  rw [planZHJP_get dj y₀ S R j]
  exact tableZHJP_M1W dj y₀ S _

/-- **Every table's level-0 rate at the hidden-output plan at `own` is at most `rateM1 k`** (`rateZ_le_m1w` at
`planZHJP_M1W`). -/
theorem rateZ_planZHJP_le_m1 (S : Finset (Fin n)) (R : Reg) (j : Fin (planZHJP (Reg := Reg) dj y₀ S R).length)
    (k : ℕ) : rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ rateM1 k :=
  rateZ_le_m1w _ j (planZHJP_M1W dj y₀ S R j) k

end FlockSoundness.Discharge.ZkHidden
