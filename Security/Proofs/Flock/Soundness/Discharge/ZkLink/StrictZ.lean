import Proofs.Flock.Soundness.Discharge.ZkLink.SessionZ
import Proofs.Flock.Soundness.Discharge.ZkLink.Count
import Proofs.Flock.Soundness.StrictCR

/-!
# The compiled `--zk` knowledge term under strict-time collision resistance

`StrictCRL.lean` for the compiled `--zk` session. Every term of `ksBoundAccZ` is a finder's collision probability: the
rewinding finders (`forkFinder` at each committed object the lock fixes: each rep's level caps, pads root and `τ`
commitment; `twoRuns` at level 0) and the self-clash finder (`oneRun`) run the compiled prover, and every pair they
count is a collision of `H` (`opRZ_conflict_collision`, `opPadsAtZ_conflict_collision`, `opTauZ_conflict_collision`,
`op₀Z_conflict_collision`, `SelfClashZ.collision`). A cap's finder counts every pair, accepting or not, so it bounds
`advRSZ`, which counts only runs accepting every table up to the table's (`forkE_capAcc_le`). So:

* `ksBoundAccZ_le_strict`: a `--zk` table's knowledge bound in a session is at most `ksStrictZ`, at finders that
  satisfy `SHA512CRStrict` at budgets `qF` (rewinding) and `qS` (self-clash);
* `ksAvgZC_le_strict`: the compiled audit's knowledge term is at most `ksAvgStrictZ`, which reads only the plan, the
  law and the budgets;
* `zk_flock_countC_execOS_ks` and `zk_flock_countC_execOS_strict`: the compiled count curve at the live draw with
  `JointZC` discharged (`jointZC_ksBound`), and with `ε_ks` in closed form under A1;
* `zk_flock_countC_execOS_cr`: the same with each unit read at its place, so `ε_ks`, `δ_link` and `δ_tree` rest on A1,
  A2 and A3 alone;
* `zk_flock_countC_execOS_m1_cr`: the same with every table at C-Flock ZK's parameters, each table's bound at most
  `2^-203` (`ksAvgStrictZ_le_m1`).

The budgets are the hypotheses' to state: the prover is a strategy, and Lean counts no evaluations.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

/-! ## The finders, in the session -/

section Session

variable (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
  (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (cap₀ : Cap D)

/-- **`Adv_{r,ℓ}`'s finder in the session**: along the primed prover, at the table's rep `r`'s level-`ℓ` cap, two
continuations, and a level-`ℓ` position they open to different rows. -/
noncomputable def advRSFinderZ (r ℓ : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Finder Pair :=
  forkFinder (plurSZ Hs T pre.length) (slotOf r ℓ + 4)
    (fun d => conflictPick Hs.H
      (fun outs : List (OutZC F D Sl) => opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d (outs.getD pre.length dfltOutZC))
      fun x _ h => opRZ_conflict_collision (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h)
    none (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- **`Adv^pads_r`'s finder in the session**: at the table's rep `r`'s pads root, two continuations, and a pads position
they open to different pairs. -/
noncomputable def advPadsSFinderZ (r : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Finder Pair :=
  forkFinder (plurSZ Hs T pre.length) r
    (fun d => conflictPick Hs.H
      (fun outs : List (OutZC F D Sl) => opPadsAtZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC))
      fun x _ h => opPadsAtZ_conflict_collision (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h)
    none (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- **`Adv^τ_r`'s finder in the session**: at the table's rep `r`'s `τ` commitment, two continuations that open it to
different values. -/
noncomputable def advTauSFinderZ (r : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Finder Pair :=
  forkFinder (plurSZ Hs T pre.length) (2 + r)
    (fun d => conflictPick Hs.H
      (fun outs : List (OutZC F D Sl) => opTauZ Hs r d (outs.getD pre.length dfltOutZC))
      fun x _ h => opTauZ_conflict_collision (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h)
    none (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- **`Adv₀`'s finder in the session**: two runs after `Commit`, and a level-0 position of the table they open to
different rows. -/
noncomputable def adv₀SFinderZ (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Finder Pair :=
  twoRuns (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ
    (conflictPick Hs.H
      (fun outs : List (OutZC F D Sl) => op₀Z Hs.H Hs.E Hs.Lf T cap₀ (outs.getD pre.length dfltOutZC))
      fun x _ h => op₀Z_conflict_collision (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h)

/-- **The self-clash finder in the session**: one run after `Commit`, and what the table opens two ways in it. -/
noncomputable def selfSFinderZ (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Finder Pair :=
  oneRun (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ
    (onePick Hs.H (fun outs : List (OutZC F D Sl) => SelfClashZ Hs T (outs.getD pre.length dfltOutZC))
      fun _ h => h.collision)

/-- **`SHA512CRStrict` for a `--zk` table's finders in a session**: the rewinding finders at budget `qF`, the self-clash
finder at budget `qS`.

Assumption record. Claim: `cr/sha-512` (`Claims.StrictBirthday`). It assumes that each knowledge finder of one table
(the rewinding finders at each level, the pads', `τ`'s and level 0's, and the self-clash finder) finds a SHA-512
collision with probability at most `q²/2^513` at its budget: the generic birthday bound for 512-bit outputs. A hash
premise: each field is `StrictCR.Finder.CR`, which is `Assumptions.SHA512CRStrict`, so it reduces to `StrictBirthday`
and asks nothing of the verifier. As `Finder.CR` says, Lean doesn't tie `qF` and `qS` to the finders' real evaluations:
they must bound them (`qF ≥ 2t′ + 2v`, `qS ≥ t′ + v`, `ASSUMPTIONS.md`). -/
structure SessCRZ (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (qF qS : ℝ) : Prop where
  fork : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 T.sch.levels.length,
    (advRSFinderZ A Hs T pre post cap₀ r ℓ σ).CR Hs.H qF
  pads : ∀ r ∈ Finset.range 2, (advPadsSFinderZ A Hs T pre post cap₀ r σ).CR Hs.H qF
  tau : ∀ r ∈ Finset.range 2, (advTauSFinderZ A Hs T pre post cap₀ r σ).CR Hs.H qF
  level0 : (adv₀SFinderZ A Hs T pre post cap₀ σ).CR Hs.H qF
  self : (selfSFinderZ A Hs T pre post cap₀ σ).CR Hs.H qS

theorem advRSZ_le_strict (r ℓ : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) {q : ℝ}
    (h : (advRSFinderZ A Hs T pre post cap₀ r ℓ σ).CR Hs.H q) :
    advRSZ A Hs T pre post cap₀ r ℓ σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  ((forkE_capAcc_le Hs T (plurSZ Hs T pre.length) (slotOf r ℓ + 4) pre.length r ℓ
    (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)).trans
  (forkE_conflict_le Hs.H (plurSZ Hs T pre.length) (slotOf r ℓ + 4)
    (fun d (outs : List (OutZC F D Sl)) => opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d (outs.getD pre.length dfltOutZC))
    (fun _ x _ h => opRZ_conflict_collision (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h)
    (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ))).trans h.le

theorem advPadsSZ_le_strict (r : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) {q : ℝ}
    (h : (advPadsSFinderZ A Hs T pre post cap₀ r σ).CR Hs.H q) :
    advPadsSZ A Hs T pre post cap₀ r σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (forkE_conflict_le Hs.H (plurSZ Hs T pre.length) r
    (fun d (outs : List (OutZC F D Sl)) => opPadsAtZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC))
    (fun _ x _ h => opPadsAtZ_conflict_collision (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h)
    (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)).trans h.le

theorem advTauSZ_le_strict (r : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) {q : ℝ}
    (h : (advTauSFinderZ A Hs T pre post cap₀ r σ).CR Hs.H q) :
    advTauSZ A Hs T pre post cap₀ r σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (forkE_conflict_le Hs.H (plurSZ Hs T pre.length) (2 + r)
    (fun d (outs : List (OutZC F D Sl)) => opTauZ Hs r d (outs.getD pre.length dfltOutZC))
    (fun _ x _ h => opTauZ_conflict_collision (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h)
    (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)).trans h.le

omit [Fintype K] in
theorem adv₀SZ_le_strict (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) {q : ℝ}
    (h : (adv₀SFinderZ A Hs T pre post cap₀ σ).CR Hs.H q) :
    adv₀SZ A Hs T pre post cap₀ σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (expect₂_conflict_le Hs.H (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ
    (fun outs : List (OutZC F D Sl) => op₀Z Hs.H Hs.E Hs.Lf T cap₀ (outs.getD pre.length dfltOutZC))
    fun x _ h => op₀Z_conflict_collision (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h).trans
    h.le

omit [Fintype K] in
theorem selfSZ_le_strict (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) {q : ℝ}
    (h : (selfSFinderZ A Hs T pre post cap₀ σ).CR Hs.H q) :
    expect (ind fun outs : List (OutZC F D Sl) => SelfClashZ Hs T (outs.getD pre.length dfltOutZC))
      (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (expect_one_le Hs.H _ σ (fun outs : List (OutZC F D Sl) => SelfClashZ Hs T (outs.getD pre.length dfltOutZC))
    fun _ h => h.collision).trans h.le

/-- **`ε_c⁻` in the session under `SHA512CRStrict`.** -/
theorem epsSZ_le_strict (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) {qF qS : ℝ}
    (h : SessCRZ A Hs T pre post cap₀ σ qF qS) :
    epsSZ A Hs T pre post cap₀ σ ≤
      T.bound.toReal +
        ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 T.sch.levels.length,
          Real.sqrt (2 ^ (lvOf T.sch ℓ).logLen * (lvOf T.sch ℓ).queries * (qF ^ 2 / 2 ^ (513 : ℝ))) +
        ∑ _r ∈ Finset.range 2, Real.sqrt (T.N * T.N * (qF ^ 2 / 2 ^ (513 : ℝ))) +
        ∑ _r ∈ Finset.range 2, Real.sqrt (qF ^ 2 / 2 ^ (513 : ℝ)) +
        qS ^ 2 / 2 ^ (513 : ℝ) :=
  add_le_add (add_le_add (add_le_add (add_le_add le_rfl
      (Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => Real.sqrt_le_sqrt
        (mul_le_mul_of_nonneg_left (advRSZ_le_strict A Hs T pre post cap₀ r ℓ σ (h.fork r hr ℓ hℓ))
          (by positivity))))
      (Finset.sum_le_sum fun r hr => Real.sqrt_le_sqrt
        (mul_le_mul_of_nonneg_left (advPadsSZ_le_strict A Hs T pre post cap₀ r σ (h.pads r hr)) (by positivity))))
      (Finset.sum_le_sum fun r hr => Real.sqrt_le_sqrt (advTauSZ_le_strict A Hs T pre post cap₀ r σ (h.tau r hr))))
    (selfSZ_le_strict A Hs T pre post cap₀ σ h.self)

end Session

/-! ## The knowledge term -/

/-- **A `--zk` table's knowledge bound in closed form**, at finder budgets `qF` (rewinding) and `qS` (self-clash):
$$\text{bound}+\sum_{r<2}\sum_{\ell\ge1}\sqrt{2^{\log N_\ell}Q_\ell\frac{q_F^2}{2^{513}}}
+2\sqrt{N^2\frac{q_F^2}{2^{513}}}+2\sqrt{\frac{q_F^2}{2^{513}}}+\frac{q_S^2}{2^{513}}+\sqrt{k\frac{q_F^2}{2^{513}}}.$$ -/
noncomputable def ksStrictZ (ts : List (TabZK F mPts)) (j : Fin ts.length) (k : ℕ) (qF qS : ℝ) : ℝ :=
  (ts.get j).bound.toReal +
    ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (ts.get j).sch.levels.length,
      Real.sqrt (2 ^ (lvOf (ts.get j).sch ℓ).logLen * (lvOf (ts.get j).sch ℓ).queries * (qF ^ 2 / 2 ^ (513 : ℝ))) +
    ∑ _r ∈ Finset.range 2, Real.sqrt ((ts.get j).N * (ts.get j).N * (qF ^ 2 / 2 ^ (513 : ℝ))) +
    ∑ _r ∈ Finset.range 2, Real.sqrt (qF ^ 2 / 2 ^ (513 : ℝ)) +
    qS ^ 2 / 2 ^ (513 : ℝ) + Real.sqrt (k * (qF ^ 2 / 2 ^ (513 : ℝ)))

/-- **The compiled `--zk` knowledge term in closed form**: after each draw, the largest over the plan's tables of
`ksStrictZ/(1 − N₀/(ek))`, averaged over the draw. -/
noncomputable def ksAvgStrictZ {n : ℕ} {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
    (L : Law n) (k : ℕ) (qF qS : ℝ) (R : Reg) : ℝ≥0∞ :=
  avg fun ω : L.Ω => ⨆ j : Fin (planZ (L.draw ω) R).length,
    ENNReal.ofReal (ksStrictZ (planZ (L.draw ω) R) j k qF qS / (1 - rateZ (planZ (L.draw ω) R) j k))

section Batch

variable (A : Arith F K) (Hs : Hashes F D Sl)

/-- **`SHA512CRStrict` for table `j`'s finders in a compiled `--zk` session**, with the prover's session strategy
`τ`. -/
abbrev TableCRZ (ts : List (TabZK F mPts)) (j : Fin ts.length) (τ : Strategy (sessionZC A Hs ts)) (qF qS : ℝ) : Prop :=
  SessCRZ A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j)
    (stratZ A Hs ts j τ) qF qS

theorem ksBoundAccZ_le_strict (ts : List (TabZK F mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionZC A Hs ts)) {qF qS : ℝ} (h : TableCRZ A Hs ts j τ qF qS) :
    ksBoundAccZ A Hs ts j k τ ≤ ksStrictZ ts j k qF qS :=
  add_le_add
    (epsSZ_le_strict A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j)
      (stratZ A Hs ts j τ) h)
    (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
      (adv₀SZ_le_strict A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j)
        (stratZ A Hs ts j τ) h.level0)
      (Nat.cast_nonneg k)))

/-- **The compiled knowledge term under `SHA512CRStrict`**: `ksAvgZC` at the knowledge bound `ksBoundAccZ` is at most
`ksAvgStrictZ`, at a prover each of whose tables' finders, after each draw, satisfies `SHA512CRStrict` at budgets `qF`,
`qS` (`TableCRZ`). -/
theorem ksAvgZC_le_strict {n : ℕ} {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts)) {L : Law n}
    (k : ℕ) {qF qS : ℝ} (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)
    (hCR : ∀ ω j, TableCRZ A Hs (planZ (L.draw ω) R) j (zcStrat A Hs planZ (τ ω)) qF qS) :
    ksAvgZC A Hs planZ k (fun S R τ j => ksBoundAccZ A Hs (planZ S R) j k τ) R τ ≤
      ksAvgStrictZ planZ L k qF qS R := by
  refine avg_mono fun ω => iSup_mono fun j => ?_
  have ha := ksBoundAccZ_nonneg A Hs (planZ (L.draw ω) R) j k (zcStrat A Hs planZ (τ ω))
  rcases le_or_gt (1 - rateZ (planZ (L.draw ω) R) j k) 0 with h0 | h0
  · exact (le_of_eq (ENNReal.ofReal_of_nonpos (div_nonpos_of_nonneg_of_nonpos ha h0))).trans zero_le
  · exact ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right
      (ksBoundAccZ_le_strict A Hs (planZ (L.draw ω) R) j k _ (hCR ω j)) h0.le)

end Batch

/-! ## At C-Flock ZK's parameters -/

/-- **A `--zk` table's knowledge bound at C-Flock ZK's parameters**: `ksStrictZ` with the table's `bound` replaced by
`2^-203` (`TabZK.bound_le_m1`). -/
noncomputable def ksStrictZm1 (ts : List (TabZK F mPts)) (j : Fin ts.length) (k : ℕ) (qF qS : ℝ) : ℝ :=
  2 ^ (-(203 : ℝ)) +
    ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (ts.get j).sch.levels.length,
      Real.sqrt (2 ^ (lvOf (ts.get j).sch ℓ).logLen * (lvOf (ts.get j).sch ℓ).queries * (qF ^ 2 / 2 ^ (513 : ℝ))) +
    ∑ _r ∈ Finset.range 2, Real.sqrt ((ts.get j).N * (ts.get j).N * (qF ^ 2 / 2 ^ (513 : ℝ))) +
    ∑ _r ∈ Finset.range 2, Real.sqrt (qF ^ 2 / 2 ^ (513 : ℝ)) +
    qS ^ 2 / 2 ^ (513 : ℝ) + Real.sqrt (k * (qF ^ 2 / 2 ^ (513 : ℝ)))

/-- **The compiled `--zk` knowledge term at C-Flock ZK's parameters in closed form**: `ksAvgStrictZ` at `ksStrictZm1`.
-/
noncomputable def ksAvgStrictZm1 {n : ℕ} {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
    (L : Law n) (k : ℕ) (qF qS : ℝ) (R : Reg) : ℝ≥0∞ :=
  avg fun ω : L.Ω => ⨆ j : Fin (planZ (L.draw ω) R).length,
    ENNReal.ofReal (ksStrictZm1 (planZ (L.draw ω) R) j k qF qS / (1 - rateZ (planZ (L.draw ω) R) j k))

theorem ksStrictZ_nonneg (ts : List (TabZK F mPts)) (j : Fin ts.length) (k : ℕ) (qF qS : ℝ) :
    0 ≤ ksStrictZ ts j k qF qS := by
  unfold ksStrictZ
  positivity

theorem ksStrictZ_le_m1 (hF : Fintype.card F = 2 ^ 128) (ts : List (TabZK F mPts)) (j : Fin ts.length)
    (hT : (ts.get j).M1) (k : ℕ) (qF qS : ℝ) : ksStrictZ ts j k qF qS ≤ ksStrictZm1 ts j k qF qS := by
  unfold ksStrictZ ksStrictZm1
  gcongr
  exact ENNReal.toReal_le_of_le_ofReal (by positivity) ((ts.get j).bound_le_m1 hF hT)

/-- **The compiled knowledge term's closed form at C-Flock ZK's parameters**: with every table of the plan at C-Flock
ZK's parameters, `ksAvgStrictZ` is at most `ksAvgStrictZm1`. -/
theorem ksAvgStrictZ_le_m1 (hF : Fintype.card F = 2 ^ 128) {n : ℕ} {Reg : Type}
    (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
    (hM1 : ∀ S R (j : Fin (planZ S R).length), ((planZ S R).get j).M1) (L : Law n) (k : ℕ) (qF qS : ℝ) (R : Reg) :
    ksAvgStrictZ planZ L k qF qS R ≤ ksAvgStrictZm1 planZ L k qF qS R := by
  refine avg_mono fun ω => iSup_mono fun j => ?_
  rcases le_or_gt (1 - rateZ (planZ (L.draw ω) R) j k) 0 with h0 | h0
  · exact (le_of_eq (ENNReal.ofReal_of_nonpos
      (div_nonpos_of_nonneg_of_nonpos (ksStrictZ_nonneg _ j k qF qS) h0))).trans zero_le
  · exact ENNReal.ofReal_le_ofReal
      (div_le_div_of_nonneg_right (ksStrictZ_le_m1 hF _ j (hM1 _ R j) k qF qS) h0.le)

/-! ## The count curve -/

section Count

variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc)

/-- **The compiled `--zk` count curve at the live draw, with `ε_ks` discharged** (`zk_flock_countC_execOS` at
`jointZC_ksBound`): for `k ≥ 1` accepted reruns,
`Pr[execAccept ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link + δ_tree`, with `ε_ks` the compiled analysis's
knowledge term at `ksBoundAccZ`, every table's fork terms at its committed objects and its level-0 term in the session. -/
theorem zk_flock_countC_execOS_ks (hA : A.Correct) (hlow : LoweringSoundZK A P planZ tab decode)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)))
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (P.wrong (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZC A Hs planZ)) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgZC A Hs planZ k (fun S R τ j => ksBoundAccZ A Hs (planZ S R) j k τ) (reg σ) (cont σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) + ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS A Hs P planZ tab vb hlow σs ks hks St hS Kd pop Lb src hA3 hRw hM hρ hr
    (jointZC_ksBound A hA Hs planZ hk) ht K σ tr hT

/-- **The compiled `--zk` count curve at the live draw, every term in closed form.** With the units drawn by
`Flock.Draw.drawOS`'s stratified law on a byte source that A3 makes uniform (`hA3`), every table's finders after each draw
under A1 at budgets `qF`, `qS` (`hCR`), the link finders under A2 (in `linkBoundZC`) and the serving roots' rewinding
finder under A1 at budget `qT` (`hT`):
`Pr[execAccept ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link + δ_tree`, with `miss(K)` `Law.stratified`'s count
curve and `ε_ks = ksAvgStrictZ`. -/
theorem zk_flock_countC_execOS_strict (hA : A.Correct) (hlow : LoweringSoundZK A P planZ tab decode)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (P.wrong (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZC A Hs planZ)) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ planZ (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) + ENNReal.ofReal (tr.bound qT) :=
  (zk_flock_countC_execOS_ks A Hs P planZ tab vb hA hlow σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht K σ
    tr hT).trans
    (add_le_add (add_le_add (add_le_add le_rfl (ksAvgZC_le_strict A Hs planZ k (reg σ) (cont σ) hCR)) le_rfl)
      le_rfl)

end Count

section CountPlace

variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- **The compiled `--zk` count curve at the live draw, every term on the accepted assumptions**
(`zk_flock_count_execOS_m1_cr`'s compiled counterpart at any table parameters; `zk_flock_countC_execOS_m1_cr` is the one
at C-Flock ZK's). Each drawn unit is read from its table at its place (`place`),
and the committed transcript is the plurality value layer of a binding of wires to registered commit strings (`vb`).
Then `Pr[execAccept ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link + δ_tree`, with `miss(K)` `Law.stratified`'s
count curve on bytes A3 makes uniform (`hA3`), `ε_ks = ksAvgStrictZ` under A1 for every table's finders after each draw
(`hCR`), `δ_link = linkBoundZC` under A2 for the link finders (`⊤` at a prover whose finders fail it), and
`δ_tree = tr.bound qT` under A1 for the serving roots' rewinding finder (`hT`). -/
theorem zk_flock_countC_execOS_cr (hA : A.Correct)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (P.wrong (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZC A Hs planZ)) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ planZ (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) + ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_strict A Hs P planZ tab vb hA (loweringSoundZK_of_place A P planZ tab place) σs ks hks St hS
    Kd pop Lb src hA3 hk hRw hM hρ hr ht K σ hCR tr hT

/-- **The compiled `--zk` count curve at the live draw, every table at C-Flock ZK's parameters, every term on the
accepted assumptions**: `zk_flock_countC_execOS_cr` with each table's bound at most `2^-203` (`hM1`), so
`ε_ks = ksAvgStrictZm1`, as `zk_flock_count_execOS_m1_cr` has `2^-203` at the interactive-oracle layer. -/
theorem zk_flock_countC_execOS_m1_cr (hA : A.Correct)
    (hM1 : ∀ S R (j : Fin (planZ S R).length), ((planZ S R).get j).M1)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (P.wrong (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZC A Hs planZ)) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZm1 planZ (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) + ENNReal.ofReal (tr.bound qT) :=
  (zk_flock_countC_execOS_cr A Hs P planZ tab hA σs ks hks St hS Kd pop Lb src hA3 place vb hk hRw hM hρ hr ht K σ hCR
    tr hT).trans
    (add_le_add (add_le_add (add_le_add le_rfl (ksAvgStrictZ_le_m1 hA.card_F planZ hM1 _ k qF qS (reg σ))) le_rfl)
      le_rfl)

end CountPlace

end FlockSoundness.Discharge.ZkLink
