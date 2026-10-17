import Proofs.Flock.Soundness.Discharge.FailClosed.One
import Proofs.Flock.Soundness.Discharge.Hidden.Theorems
import Proofs.Flock.Soundness.Accounting.Numbers
import Proofs.Flock.VBridge.Family.Lig

/-!
# Soundness at `fast100`'s numbers, over everything flock-verify accepts without `--zk`

`flock_verify_sound`'s claims without `--zk`, `SoundExec` (public outputs) and `SoundExecH` (hidden), bound the knowledge
term by `StrictCR.ksAvgStrict`, whose per-table term is the table's `tableError`. `flock_verify_sound_205` states them
with that term replaced by `2^-205` (`ksStrict205`, `ksAvgStrict205`), and with no hypothesis on the statement's range:
acceptance gives it.

* `tableError_le_205_wide`: `tableError` of a `fast100` schedule is at most `2^-205` at every shape in
  `Stmt.InRangeWide`, `Stmt.InRange` with two regions more (`tableError_le_of_le_33`'s kernel check at 1026 regions).
* `ksAvgStrict_le_205`: with every plan table's shape in that range, `ksAvgStrict` is at most `ksAvgStrict205`.
* `planAt_inRangeWide`, `planAtH_inRangeWide`: every table of the one-table plans is in that range. A draw flock-verify
  sets up has its statement's range, which `Stmt.setupH` and `Stmt.setupHidden` check (`checkInRange`:
  `setupH_inRange`, `Hidden.Spec.inRange`). A refused draw's table is a reference setup's statement with `contra`'s two
  regions added (`DrawSetup.refused`), which is why the range is two regions wider.
* `SoundExec205`, `SoundExecH205`: `SoundExec` and `SoundExecH` at `ksAvgStrict205`, which they give by monotonicity
  (`SoundExec.to205`, `SoundExecH.to205`).

Under `--zk` the per-table term is each table's `bound`, at most `2^-203` at C-Flock ZK's parameters (`TabZK.M1`,
`ksAvgStrictZ_le_m1`). Acceptance does not give `M1` here, so the `--zk` claims stay as `flock_verify_sound` states them.
-/

namespace FlockSoundness.Accounting

/-- **The shapes of every plan table**: `Stmt.InRange` with two regions more, the two `contra` regions a refused draw's
table adds to a reference setup's statement. -/
def Stmt.InRangeWide (st : Stmt) : Prop := st.kLog ≤ 27 ∧ st.nRegions ≤ 1026 ∧ st.mPts ≤ 64

/-- `piopNumMax` at 1026 regions (2054 claims). -/
def piopNumMaxWide (m L₀ : ℕ) : ℕ :=
  (L₀ + 1) * (m - 13) + L₀ * 127 + L₀ * (2 * (m - 6)) + L₀ * m
    + L₀ * (2 + 2 * 21 + 63) + L₀ * (7 * 2054 + 1) + L₀

theorem piopError_le_wide (m L₀ : ℕ) (st : Stmt) (hst : st.InRangeWide) :
    piopError m st.kLog st.nClaims L₀ ≤ ((piopNumMaxWide m L₀ : ℚ) / 2 ^ 128 : ℚ) := by
  obtain ⟨hk, hR, _⟩ := hst
  unfold piopError qF
  push_cast
  gcongr
  have h1 : m - st.kLog ≤ m := Nat.sub_le _ _
  have h2 : st.kLog - 6 ≤ 21 := by omega
  have h3 : st.nClaims ≤ 2054 := by unfold Stmt.nClaims; omega
  unfold piopNumMaxWide
  exact_mod_cast (by gcongr : (L₀ + 1) * (m - 13) + L₀ * 127 + L₀ * (2 * (m - 6))
      + L₀ * (m - st.kLog) + L₀ * (2 + 2 * (st.kLog - 6) + 63) + L₀ * (7 * st.nClaims + 1) + L₀
      ≤ (L₀ + 1) * (m - 13) + L₀ * 127 + L₀ * (2 * (m - 6)) + L₀ * m
      + L₀ * (2 + 2 * 21 + 63) + L₀ * (7 * 2054 + 1) + L₀)

/-- `repUp` at 1026 regions. -/
def repUpWide (s : Schedule) : ℚ := (piopNumMaxWide s.m s.listBound₀ : ℚ) / 2 ^ 128 + ligUp s.levels

/-- `linkUp` at 1026 regions. -/
def linkUpWide (s : Schedule) : ℚ := s.listBound₀ * 1026 * (64 / 2 ^ 128) ^ 2

theorem tableError_le_up_wide (s : Schedule) (hs : ∀ l ∈ s.levels, l.Ok) (st : Stmt) (hst : st.InRangeWide) :
    tableError s st ≤ ((linkUpWide s + repUpWide s ^ 2 : ℚ) : ℝ) := by
  have h1 : linkError s st ≤ (linkUpWide s : ℝ) := by
    obtain ⟨_, hR, hp⟩ := hst
    unfold linkError linkUpWide qF
    push_cast
    have hR' : (st.nRegions : ℝ) ≤ 1026 := by exact_mod_cast hR
    have hp' : (st.mPts : ℝ) ≤ 64 := by exact_mod_cast hp
    gcongr
  have h2 : repError s st ≤ (repUpWide s : ℝ) := by
    unfold repError repUpWide
    push_cast
    have h1 := piopError_le_wide s.m s.listBound₀ st hst
    have h2 := ligeritoError_le s.levels hs
    push_cast at h1
    linarith
  have h0 := repError_nonneg s st hs
  unfold tableError
  push_cast
  nlinarith

/-- The rational check at 1026 regions for `22 ≤ m ≤ 33`. -/
theorem check_wide_le_33 : ∀ m, 22 ≤ m → m ≤ 33 → ∀ s, fast100 m = some s →
    linkUpWide s + repUpWide s ^ 2 ≤ (1 / 2) ^ 205 := by
  intro m h1 h2 s hs
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at hs; subst hs; decide +kernel)

/-- The rational check at 1026 regions for `m = 34, 35`. -/
theorem check_wide_34_35 : ∀ m, 34 ≤ m → m ≤ 35 → ∀ s, fast100 m = some s →
    linkUpWide s + repUpWide s ^ 2 ≤ (1 / 2) ^ 205 := by
  intro m h1 h2 s hs
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at hs; subst hs; decide +kernel)

/-- The rational check at 1026 regions, for every `m` that `fast100` supports. -/
theorem check_wide (m : ℕ) (h1 : 22 ≤ m) (h2 : m ≤ 35) (s : Schedule) (hs : fast100 m = some s) :
    linkUpWide s + repUpWide s ^ 2 ≤ (1 / 2) ^ 205 := by
  rcases le_or_gt m 33 with h | h
  · exact check_wide_le_33 m h1 h s hs
  · exact check_wide_34_35 m (by omega) h2 s hs

theorem fast100_range : ∀ m s, fast100 m = some s → 22 ≤ m ∧ m ≤ 35 := by
  intro m s h
  have hm : ∀ m', fast100 m' ≠ none → 22 ≤ m' ∧ m' ≤ 35 := by
    intro m' hm'
    by_contra hc
    apply hm'
    unfold fast100
    split <;> simp_all
  exact hm m (by simp [h])

/-- **The soundness number for `fast100` at every shape in `Stmt.InRangeWide`**: at most `2^-205` per table. -/
theorem tableError_le_205_wide {m : ℕ} {s : Schedule} (hs : fast100 m = some s) (st : Stmt) (hst : st.InRangeWide) :
    tableError s st ≤ (2 : ℝ) ^ (-(205 : ℝ)) := by
  obtain ⟨h1, h2⟩ := fast100_range m s hs
  refine (tableError_le_up_wide s (fast100_levels_ok m s hs) st hst).trans ?_
  refine le_trans ?_ half_pow_205_le
  have h := Rat.cast_le (K := ℝ) |>.2 (check_wide m h1 h2 s hs)
  refine h.trans (le_of_eq ?_)
  push_cast; ring

end FlockSoundness.Accounting

namespace FlockSoundness.StrictCR

open Game Model Accounting
open scoped ENNReal

variable {mPts : ℕ}

/-- **A table's knowledge bound at `fast100`'s numbers**: `ksStrict` with the table's `tableError` replaced by
`2^-205`. -/
noncomputable def ksStrict205 (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ) (qF qS : ℝ) : ℝ :=
  2 ^ (-(205 : ℝ)) +
    ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 ((ts.get j).l₀ :: (ts.get j).l₁ :: (ts.get j).rest).length,
      Real.sqrt (2 ^ (lvOf (ts.get j).sch ℓ).logLen * (lvOf (ts.get j).sch ℓ).queries * (qF ^ 2 / 2 ^ (513 : ℝ))) +
    qS ^ 2 / 2 ^ (513 : ℝ) + Real.sqrt (k * (qF ^ 2 / 2 ^ (513 : ℝ)))

/-- **The knowledge term at `fast100`'s numbers**: `ksAvgStrict` at `ksStrict205`. -/
noncomputable def ksAvgStrict205 {n : ℕ} {Reg : Type} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (L : Audit.Law n) (k : ℕ) (qF qS : ℝ) (R : Reg) : ℝ≥0∞ :=
  avg fun ω : L.Ω => ⨆ j : Fin (plan (L.draw ω) R).length,
    ENNReal.ofReal (ksStrict205 (plan (L.draw ω) R) j k qF qS / (1 - rateB (plan (L.draw ω) R) j k))

theorem ksStrict_nonneg (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ) (qF qS : ℝ) :
    0 ≤ ksStrict ts j k qF qS := by
  unfold ksStrict
  refine add_nonneg (add_nonneg (add_nonneg (tableError_nonneg _ _) ?_) ?_) ?_ <;> positivity

theorem ksStrict_le_205 (ts : List (TabSpec mPts)) (j : Fin ts.length) (hT : ((ts.get j).S.shape mPts).InRangeWide)
    (k : ℕ) (qF qS : ℝ) : ksStrict ts j k qF qS ≤ ksStrict205 ts j k qF qS := by
  unfold ksStrict ksStrict205
  gcongr
  exact tableError_le_205_wide (ts.get j).hsch _ hT

/-- **The knowledge term's closed form at `fast100`'s numbers**: with every table of the plan in `Stmt.InRangeWide`,
`ksAvgStrict` is at most `ksAvgStrict205`. -/
theorem ksAvgStrict_le_205 {n : ℕ} {Reg : Type} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (hR : ∀ S R (j : Fin (plan S R).length), (((plan S R).get j).S.shape mPts).InRangeWide) (L : Audit.Law n)
    (k : ℕ) (qF qS : ℝ) (R : Reg) : ksAvgStrict plan L k qF qS R ≤ ksAvgStrict205 plan L k qF qS R := by
  refine avg_mono fun ω => iSup_mono fun j => ?_
  rcases le_or_gt (1 - rateB (plan (L.draw ω) R) j k) 0 with h0 | h0
  · exact (le_of_eq (ENNReal.ofReal_of_nonpos
      (div_nonpos_of_nonneg_of_nonpos (ksStrict_nonneg _ j k qF qS) h0))).trans zero_le
  · exact ENNReal.ofReal_le_ofReal
      (div_le_div_of_nonneg_right (ksStrict_le_205 _ j (hR _ R j) k qF qS) h0.le)

end FlockSoundness.StrictCR

namespace FlockSoundness.Discharge.Integrate

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs)

variable {n mPts : ℕ} {I : Inputs} (ht : I.tags.typed = false)

/-- **A table at a draw setup's statement is in the wide range** when its regions number at most two more than the
statement's: `Stmt.setupH` checks the statement's range (`setupH_inRange`). -/
theorem DrawSetup.specAt_inRangeWide {S : Finset (Fin n)} {d : Lean.Json} (x : DrawSetup I mPts S d)
    (regs : List (Model.Region x.st.c.kLog (x.st.m - x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) (hlen : regs.length ≤ x.st.regions.size + 2) :
    ((x.specAt ht regs hregs).S.shape mPts).InRangeWide := by
  obtain ⟨hk, hr, hp⟩ := (checkInRange_ok _ _ _).1 (FlockVBridge.Family.setupH_inRange x.h)
  have h1 := x.wf.2.mPts
  have h2 := x.hmPts
  show x.st.c.kLog ≤ 27 ∧ regs.length ≤ 1026 ∧ mPts ≤ 64
  exact ⟨hk, by omega, by omega⟩

/-- **Every table of the one-table plan is in the wide range**: the draw's table has its statement's regions, and a
refused draw's table the reference setup's and `contra`'s two. -/
theorem planAt_inRangeWide (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀))
    {Reg : Type} (S : Finset (Fin n)) (R : Reg) (j : Fin (planAt (Reg := Reg) ht dj x₀ S R).length) :
    (((planAt (Reg := Reg) ht dj x₀ S R).get j).S.shape mPts).InRangeWide := by
  rcases j with ⟨_ | i, hi⟩
  · show ((tableAt ht dj x₀ S).T.S.shape mPts).InRangeWide
    rcases tableAt_cases ht dj x₀ S with ⟨x, e⟩ | e <;> rw [e]
    · exact x.specAt_inRangeWide ht (regionsOf x.st) (stmtOf_linkLayout x.st x.wf.1 x.wf.2).1
        (by rw [FlockVBridge.Family.length_regionsOf]; omega)
    · exact x₀.specAt_inRangeWide ht (regionsOf x₀.st ++ DrawSetup.contra x₀.st.c.kLog (x₀.st.m - x₀.st.c.kLog))
        x₀.refused_wf (by rw [List.length_append, FlockVBridge.Family.length_regionsOf]; simp [DrawSetup.contra])
  · have h1 : (planAt (Reg := Reg) ht dj x₀ S R).length = 1 := rfl
    omega

end FlockSoundness.Discharge.Integrate

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs)
open FlockSoundness.Discharge.Integrate (DrawSetup)

variable {n mPts : ℕ} {I : Inputs}

/-- **A table at a hidden-output draw setup's statement is in the wide range** when its regions number at most two more
than the statement's: `Stmt.setupHidden` checks the statement's range (`Hidden.Spec.inRange`). -/
theorem DrawSetupH.specAt_inRangeWide {S : Finset (Fin n)} {d : Lean.Json} (x : DrawSetupH I mPts S d)
    (regs : List (Model.Region x.st.c.kLog (x.st.m - x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) (hlen : regs.length ≤ x.st.regions.size + 2) :
    ((x.specAt regs hregs).S.shape mPts).InRangeWide := by
  obtain ⟨hk, hr, hp⟩ := (checkInRange_ok _ _ _).1 (Hidden.setupHidden_spec x.h).inRange
  have h1 := x.wf.2.mPts
  have h2 := x.hmPts
  show x.st.c.kLog ≤ 27 ∧ regs.length ≤ 1026 ∧ mPts ≤ 64
  exact ⟨hk, by omega, by omega⟩

/-- **Every table of the hidden-output one-table plan is in the wide range** (`planAt_inRangeWide`'s twin). -/
theorem planAtH_inRangeWide (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type} (S : Finset (Fin n)) (R : Reg)
    (j : Fin (planAtH (Reg := Reg) dj x₀ S R).length) :
    (((planAtH (Reg := Reg) dj x₀ S R).get j).S.shape mPts).InRangeWide := by
  rcases j with ⟨_ | i, hi⟩
  · show ((tableAtH dj x₀ S).T.S.shape mPts).InRangeWide
    rcases tableAtH_cases dj x₀ S with ⟨x, e⟩ | e <;> rw [e]
    · exact x.specAt_inRangeWide (regionsOf x.st) (stmtOf_linkLayout x.st x.wf.1 x.wf.2).1
        (by rw [FlockVBridge.Family.length_regionsOf]; omega)
    · exact x₀.specAt_inRangeWide (regionsOf x₀.st ++ DrawSetup.contra x₀.st.c.kLog (x₀.st.m - x₀.st.c.kLog))
        x₀.refused_wf (by rw [List.length_append, FlockVBridge.Family.length_regionsOf]; simp [DrawSetup.contra])
  · have h1 : (planAtH (Reg := Reg) dj x₀ S R).length = 1 := rfl
    omega

end FlockSoundness.Discharge.HiddenExec

namespace FlockSoundness.Discharge.FailClosed

section

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.DecodeZL

set_option linter.unusedVariables false in
/-- `SoundExec` at `fast100`'s numbers: its knowledge term `ksAvgStrict205`, each table's term `2^-205`. -/
def SoundExec205 (I : Inputs) (hh : I.tags.hiddenOutputs = false) : Prop :=
  ∀ {n : ℕ}
    {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀))
    {Reg : Type}
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ hs : Scope I, ∀ S R j, rateB (Integrate.planAt (Reg := Reg) hs.typed dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLive I dj wr)
    (hdraw : ∀ ω, DrawOk I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : ∀ hs : Scope I, LinkCRL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 leaf512
      (Integrate.planAt hs.typed dj x₀) (Integrate.tabAt hs.typed dj x₀) ((rsAt hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh) (Reg := Reg)).binding
        (rsAt_computes hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh)))
      (reg (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
        (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
        (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ hs : Scope I, ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (Integrate.planAt hs.typed dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
          (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ))) j
      (batchStratL H512 enc512 leaf512 (Integrate.planAt hs.typed dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
          (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ) ω)) qF qS),
    prob (fun o => FlockVerifyAccepts false I o.2.2 ∧ ∀ hs : Scope I,
        K ≤ (((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 leaf512 (Integrate.planAt hs.typed dj x₀)
              (Integrate.tabAt hs.typed dj x₀) ((rsAt hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh) (Reg := Reg)).binding
                (rsAt_computes hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh)))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
                (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N
                (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
                (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N
                (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : Scope I, ((Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict205 (Integrate.planAt hs.typed dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
            (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ)) +
        ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (Integrate.planAt hs.typed dj x₀) (Integrate.tabAt hs.typed dj x₀)
          ((rsAt hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh) (Reg := Reg)).binding
            (rsAt_computes hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh))) k Rw M ρ t')

/-- **`SoundExec` gives `SoundExec205`**: every table of the plan is in the wide range (`planAt_inRangeWide`). -/
theorem SoundExec.to205 {I : Inputs} {hh : I.tags.hiddenOutputs = false} (h : SoundExec I hh) : SoundExec205 I hh :=
  @fun _ _ dj _ x₀ Reg _ σs ks hks St hS Kd pop _ _ _ Lb src hA3 _ hk _ _ hRw hM _ hρ hr _ ht' K wr dg N hwr hdraw τ
      hCR _ _ hKS =>
    (h dj x₀ σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K wr dg N hwr hdraw τ hCR hKS).trans
      (iSup_mono fun hs => add_le_add (add_le_add le_rfl
        (StrictCR.ksAvgStrict_le_205 (planAt (Reg := Reg) hs.typed dj x₀) (planAt_inRangeWide hs.typed dj x₀) _ _ _ _ _))
        le_rfl)

end

section

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.HiddenDec FlockSoundness.Discharge.DecodeZL

set_option linter.unusedVariables false in
/-- `SoundExecH` at `fast100`'s numbers: its knowledge term `ksAvgStrict205`, each table's term `2^-205`. -/
def SoundExecH205 (I : Inputs) (hh : I.tags.hiddenOutputs = true) : Prop :=
  ∀ {n : ℕ}
    {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))
    {Reg : Type}
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLive I dj wr)
    (hdraw : ∀ ω, DrawOkH I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : ∀ hs : Scope I, LinkCRL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 leaf512
      (planAtH dj x₀) (tabAtH dj x₀) ((rsAtH hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
        (rsAtH_computes hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables))
      (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ hs : Scope I, ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))) j
      (batchStratL H512 enc512 leaf512 (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ) ω)) qF qS),
    prob (fun o => FlockVerifyAccepts false I o.2.2 ∧ ∀ hs : Scope I,
        K ≤ (((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 leaf512 (planAtH dj x₀)
              (tabAtH dj x₀) ((rsAtH hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
                (rsAtH_computes hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : Scope I, ((Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict205 (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
            (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) +
        ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtH hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
            (rsAtH_computes hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables)) k Rw M ρ t')

/-- **`SoundExecH` gives `SoundExecH205`**: every table of the plan is in the wide range (`planAtH_inRangeWide`). -/
theorem SoundExecH.to205 {I : Inputs} {hh : I.tags.hiddenOutputs = true} (h : SoundExecH I hh) : SoundExecH205 I hh :=
  @fun _ _ dj _ x₀ Reg _ σs ks hks St hS Kd pop _ _ _ Lb src hA3 _ hk _ _ hRw hM _ hρ hr _ ht' K wr dg N hwr hdraw τ
      hCR _ _ hKS =>
    (h dj x₀ σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K wr dg N hwr hdraw τ hCR hKS).trans
      (iSup_mono fun _ => add_le_add (add_le_add le_rfl
        (StrictCR.ksAvgStrict_le_205 (planAtH (Reg := Reg) dj x₀) (planAtH_inRangeWide dj x₀) _ _ _ _ _)) le_rfl)

end

section

open Flock Exec ZkExec

/-- **Soundness at `fast100`'s numbers, over everything flock-verify accepts without `--zk`.** At every statement `I`:
the live session's probability that `flock-verify verify` accepts without `--zk` (`FlockVerifyAccepts false I`) while
at least `K` of the statement's proof units are wrong is at most `flock_verify_sound`'s bound with each table's term
`2^-205`: `SoundExec205` at public outputs, `SoundExecH205` at hidden. No hypothesis on the statement's range: the
verifier checks it (`checkInRange`, in `Stmt.setupH` and `Stmt.setupHidden`), so every plan table is in range
(`planAt_inRangeWide`, `planAtH_inRangeWide`), and `fast100`'s kernel check bounds each table there
(`tableError_le_205_wide`). With `--registered F` and no `--zk`, what the verifier accepts it accepts without
`--registered` (the second conjunct), so the bound covers it too. -/
theorem flock_verify_sound_205 (I : Inputs) :
    (if hh : I.tags.hiddenOutputs = true then SoundExecH205 I hh else SoundExec205 I (Bool.eq_false_of_ne_true hh)) ∧
    ∀ (own : Array Flock.Registered.Port) (t : Exec.Transcript), FlockVerifyAcceptsR false I own t →
      FlockVerifyAccepts false I t := by
  refine ⟨?_, fun own t h => ((flock_verify_sound false I).2 own t h).resolve_left fun h' => Bool.false_ne_true h'.1⟩
  have h := (flock_verify_sound false I).1
  by_cases hh : I.tags.hiddenOutputs = true
  · rw [dite_eq_left hh] at h ⊢
    rw [ite_eq_right Bool.false_ne_true] at h
    exact @SoundExecH.to205 I hh h
  · rw [dite_eq_right hh] at h ⊢
    rw [ite_eq_right Bool.false_ne_true] at h
    exact @SoundExec.to205 I (Bool.eq_false_of_ne_true hh) h

end

end FlockSoundness.Discharge.FailClosed
