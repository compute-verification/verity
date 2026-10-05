import FlockSoundness.Protocol.Discharge.ZkLink.SessionZ
import FlockSoundness.Protocol.Discharge.ZkLink.CompiledLink
import FlockSoundness.Protocol.Binding.Rows
import FlockSoundness.Protocol.StrictCR

/-!
# The compiled `--zk` knowledge term under strict-time collision resistance

`StrictCRL.lean` for the compiled `--zk` session. Every term of `ksBoundAccZ` is a finder's collision probability: the
rewinding finders (`forkFinder` at each committed object the lock fixes: each rep's level caps, pads root and `τ`
commitment; `twoRuns` at level 0) and the self-clash finder (`oneRun`) run the compiled prover, and every pair they
count is a collision of `H` (`opRZ_conflict_collision`, `opPadsZ_conflict_collision`, `opTauZ_conflict_collision`,
`op₀Z_conflict_collision`, `SelfClashZ.collision`). So:

* `ksBoundAccZ_le_strict`: a `--zk` table's knowledge bound in a session is at most `ksStrictZ`, at finders that
  satisfy `SHA512CRStrict` at budgets `qF` (rewinding) and `qS` (self-clash);
* `ksAvgZC_le_strict`: the compiled audit's knowledge term is at most `ksAvgStrictZ`, which reads only the plan, the
  law and the budgets;
* `zk_flock_countC_execOS_ks` and `zk_flock_countC_execOS_strict`: the compiled count curve at the live draw with
  `JointZC` discharged (`jointZC_ksBound`), and with `ε_ks` in closed form under A1;
* `zk_flock_countC_execOS_cr`: the same with each unit read at its place, so `ε_ks`, `δ_link` and `δ_tree` rest on A1,
  A2 and A3 alone;
* `zk_flock_countC_execOS_m1_cr`: the same with every table at M1's parameters, each table's bound at most `2^-203`
  (`ksAvgStrictZ_le_m1`).

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
      (fun outs : List (OutZC F D Sl) => opPadsZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC))
      fun x _ h => opPadsZ_conflict_collision (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h)
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
finder at budget `qS`. -/
structure SessCRZ (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (qF qS : ℝ) : Prop where
  fork : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 T.sch.levels.length,
    (advRSFinderZ A Hs T pre post cap₀ r ℓ σ).CR Hs.H qF
  pads : ∀ r ∈ Finset.range 2, (advPadsSFinderZ A Hs T pre post cap₀ r σ).CR Hs.H qF
  tau : ∀ r ∈ Finset.range 2, (advTauSFinderZ A Hs T pre post cap₀ r σ).CR Hs.H qF
  level0 : (adv₀SFinderZ A Hs T pre post cap₀ σ).CR Hs.H qF
  self : (selfSFinderZ A Hs T pre post cap₀ σ).CR Hs.H qS

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

end Batch

/-! ## At M1's parameters -/

/-! ## The count curve -/

section Count

variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc)

end Count

section CountPlace

variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

end CountPlace

end FlockSoundness.Discharge.ZkLink
