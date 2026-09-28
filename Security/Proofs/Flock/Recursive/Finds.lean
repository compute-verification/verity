import Proofs.Flock.Recursive.Transfer
import Proofs.Flock.Recursive.Multi

/-!
# Recursion: forks, the runs of the prover's own strategy that find a SHA-512 collision

`Finder.CR H F q` (`cr/sha-512` at one finder) bounds a finder's success by `q²/2^513`. The recursive theorem's fork
form reads what the finder's runs did instead: some outcome its strategy reaches outputs two different strings `H` maps
to one digest, or none does and its success is `0`. Every `∃` ranges over what the runs output, never over arbitrary
strings.

* `Game.Outcome g s a` (`Defs`): `s` plays `g` to `a` at some coins; an event no reached outcome has has probability `0`
  (`Game.prob_eq_zero_of_outcome`);
* `FindsColl H g s out` (`Defs`): some outcome `s` reaches outputs, by `out`, a collision of `H`; `Finder.Finds` at a
  finder;
* a finder that finds nothing satisfies `SHA512CRStrict` at every budget (`Finder.cr_of_not_finds`), and
  `SHA512CRExpected` at any nonnegative cost (`crExpected_of_not_finds`);
* `SessForkZ`, `TableForkZ`, `RegForkZC`, `LinkForkZC`: a fork of a `--zk` session's knowledge, registered-value and link
  finders; with none, their `cr/sha-512` at budget `0` (`sessCRZ_of_not_fork`, `regCRZC_of_not_fork`) and their
  `ecr/sha-512` at cost `0` (`linkCRZC_of_not_fork`);
* `tableErrZ`, `linkErrZC`, `waitErrZC`: the statistical terms the session's bounds keep at budget `0`
  (`ksAvgStrictZ_zero`, `linkBoundZC_zero`, `regTermZC_zero`);
* `ExAt`: a property at some leaf the recursive prover reaches; with none, its negation at every leaf (`allAt_of_not_exAt`).
-/

namespace FlockSoundness.Game

variable {α : Type}

/-- **An event that no outcome the strategy reaches has is never seen.** -/
theorem prob_eq_zero_of_outcome {p : α → Prop} :
    ∀ (g : Game α) (s : Strategy g), (∀ a, Outcome g s a → ¬ p a) → prob p g s = 0
  | .ret b, _, h => by simp [prob, h b rfl]
  | .send _ k, s, h => prob_eq_zero_of_outcome (k s.1) s.2 h
  | @Game.coin _ _ _ _ k, s, h => by
      have hc : ∀ c, prob p (k c) (s c) = 0 := fun c =>
        prob_eq_zero_of_outcome (k c) (s c) fun a ha => h a ⟨c, ha⟩
      simp only [prob, hc, mul_zero, Finset.sum_const_zero]

end FlockSoundness.Game

namespace FlockSoundness.StrictCR

open Game

variable {D β : Type} {H : List UInt8 → D}

theorem prob_coll_eq_zero {g : Game β} {s : Strategy g} {out : β → Pair} (h : ¬ FindsColl H g s out) :
    prob (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) g s = 0 :=
  Game.prob_eq_zero_of_outcome g s fun b hb hc => h ⟨b, hb, hc⟩

/-- **A finder that finds nothing satisfies `SHA512CRStrict` at every budget.** -/
theorem Finder.cr_of_not_finds {Fd : Finder Pair} (h : ¬ Fd.Finds H) (q : ℝ) : Fd.CR H q := by
  unfold Finder.CR Assumptions.SHA512CRStrict
  intro _
  rw [prob_coll_eq_zero h, ENNReal.toReal_zero]
  positivity

/-- **A run that finds nothing satisfies `SHA512CRExpected`** at any nonnegative cost. -/
theorem crExpected_of_not_finds {g : Game β} {s : Strategy g} {out : β → Pair} {cost : β → ℝ}
    (h : ¬ FindsColl H g s out) (hc : ∀ b, 0 ≤ cost b) : Assumptions.SHA512CRExpected H g s out cost := by
  unfold Assumptions.SHA512CRExpected
  rw [prob_coll_eq_zero h, ENNReal.toReal_zero]
  exact div_nonneg (expect_nonneg hc g s) (by positivity)

end FlockSoundness.StrictCR

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession ZkLink StrictCR ZkReg
open scoped ENNReal

section Cost

variable {τ Val : Type} [Fintype Val] [DecidableEq Val] (rd : τ → Finset Val)

omit [Fintype Val] [DecidableEq Val] in
theorem loopCost_nonneg {c : τ → ℝ} (hc : ∀ e, 0 ≤ c e) : ∀ es, 0 ≤ loopCost rd c es
  | [] => le_rfl
  | e :: es => by
      simp only [loopCost]
      split_ifs
      · simpa using hc e
      · exact add_nonneg (hc e) (loopCost_nonneg hc es)

theorem finderCost_nonneg {c : τ → ℝ} (hc : ∀ e, 0 ≤ c e) (x : τ × List τ) : 0 ≤ finderCost rd c x := by
  unfold finderCost
  split_ifs
  · exact add_nonneg (hc _) (loopCost_nonneg rd hc _)
  · simpa using hc x.1

end Cost

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

section Session

variable (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
  (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (cap₀ : Cap D)

/-- **A fork of a `--zk` table's finders in a session** (`SessCRZ`'s): at some rep `r`, two continuations of the prover
from one of the table's committed objects (a level cap, the pads root, the `τ` commitment) that open it to different
rows, pairs or values; two runs after `Commit` that open a level-0 position to different rows; or one run that opens the
table two ways. Each outputs a collision of `H` at an outcome it reaches. -/
def SessForkZ (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Prop :=
  (∃ r ∈ Finset.range 2, ∃ ℓ ∈ Finset.Icc 1 T.sch.levels.length,
      (advRSFinderZ A Hs T pre post cap₀ r ℓ σ).Finds Hs.H) ∨
    (∃ r ∈ Finset.range 2, (advPadsSFinderZ A Hs T pre post cap₀ r σ).Finds Hs.H) ∨
    (∃ r ∈ Finset.range 2, (advTauSFinderZ A Hs T pre post cap₀ r σ).Finds Hs.H) ∨
    (adv₀SFinderZ A Hs T pre post cap₀ σ).Finds Hs.H ∨
    (selfSFinderZ A Hs T pre post cap₀ σ).Finds Hs.H

theorem sessCRZ_of_not_fork {σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))}
    (h : ¬ SessForkZ A Hs T pre post cap₀ σ) : SessCRZ A Hs T pre post cap₀ σ 0 0 where
  fork r hr ℓ hℓ := Finder.cr_of_not_finds (fun hf => h (Or.inl ⟨r, hr, ℓ, hℓ, hf⟩)) 0
  pads r hr := Finder.cr_of_not_finds (fun hf => h (Or.inr (Or.inl ⟨r, hr, hf⟩))) 0
  tau r hr := Finder.cr_of_not_finds (fun hf => h (Or.inr (Or.inr (Or.inl ⟨r, hr, hf⟩)))) 0
  level0 := Finder.cr_of_not_finds (fun hf => h (Or.inr (Or.inr (Or.inr (Or.inl hf))))) 0
  self := Finder.cr_of_not_finds (fun hf => h (Or.inr (Or.inr (Or.inr (Or.inr hf))))) 0

end Session

section Batch

variable (A : Arith F K) (Hs : Hashes F D Sl)

/-- **A fork of table `j`'s finders in a compiled `--zk` session** with the prover's session strategy `τ`
(`TableCRZ`'s finders). -/
abbrev TableForkZ (ts : List (TabZK F mPts)) (j : Fin ts.length) (τ : Strategy (sessionZC A Hs ts)) : Prop :=
  SessForkZ A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j)
    (stratZ A Hs ts j τ)

theorem tableCRZ_of_not_fork {ts : List (TabZK F mPts)} {j : Fin ts.length} {τ : Strategy (sessionZC A Hs ts)}
    (h : ¬ TableForkZ A Hs ts j τ) : TableCRZ A Hs ts j τ 0 0 :=
  sessCRZ_of_not_fork A Hs _ _ _ _ h

/-- **The drawn tables' error**: after each draw, the largest over the plan's tables of its `bound` over
`1 − N₀/(ek)`, averaged over the draw (`ksAvgStrictZ` with no finder budget). -/
noncomputable def tableErrZ {n : ℕ} {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts)) (L : Law n)
    (k : ℕ) (R : Reg) : ℝ≥0∞ :=
  avg fun ω : L.Ω => ⨆ j : Fin (planZ (L.draw ω) R).length,
    ENNReal.ofReal (((planZ (L.draw ω) R).get j).bound.toReal / (1 - rateZ (planZ (L.draw ω) R) j k))

theorem ksAvgStrictZ_zero {n : ℕ} {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts)) (L : Law n)
    (k : ℕ) (R : Reg) : ksAvgStrictZ planZ L k 0 0 R = tableErrZ planZ L k R := by
  simp [ksAvgStrictZ, tableErrZ, ksStrictZ]

end Batch

section Link

variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc) (wo : vb.Pos → vb.Opening)
  (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

/-- **A fork of the registered-value finders**: for some commit string `c` a draw can read, a trial from the prover's
continuation reads at `c` a value other than the registrant's opening of `c`'s registered commitment, and the two
openings are a collision of `Hc` (`regFinderZC`). -/
def RegForkZC (k Rw : ℕ) : Prop :=
  ∃ (c : vb.Pos) (_ : Nonempty (ΩcZ A P planZ tab vb L c)), (regFinderZC A Hs P planZ tab vb wo R τ c k Rw).Finds Hc

omit [Fintype K] in
theorem regCRZC_of_not_fork {k Rw : ℕ} (h : ¬ RegForkZC A Hs P planZ tab vb wo R τ k Rw) :
    RegCRZC A Hs P planZ tab vb wo R τ k Rw 0 :=
  fun c _ => Finder.cr_of_not_finds (fun hf => h ⟨c, inferInstance, hf⟩) 0

/-- **A fork of the link finders**: for some commit string `c` a draw can read, the link finder's trials from the
prover's continuation (`finderG`, `M` further trials) open `c` to two different values, a collision of `Hc`
(`outCZC`). -/
def LinkForkZC (k Rw M : ℕ) : Prop :=
  ∃ (c : vb.Pos) (_ : Nonempty (ΩcZ A P planZ tab vb L c)),
    FindsColl Hc (finderG (trialGZC A Hs P planZ tab vb R τ c k Rw) M)
      (finderStrat _ (trialStratZC A Hs P planZ tab vb R τ c k Rw) M) (outCZC A Hs P planZ tab vb R τ c)

omit [Fintype K] in
theorem linkCRZC_of_not_fork {k Rw M : ℕ} (h : ¬ LinkForkZC A Hs P planZ tab vb R τ k Rw M) :
    LinkCRZC A Hs P planZ tab vb R τ k Rw M 0 :=
  fun c _ => crExpected_of_not_finds (fun hf => h ⟨c, inferInstance, hf⟩)
    (finderCost_nonneg _ fun e => by simp [cTZC])

/-- **`Q_s`**: the expected number of commit strings the drawn units read. -/
noncomputable def qsZC (L : Law n) : ℝ :=
  ∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω

/-- **The link error**: `Q_s/(1 − ρ)·(1/(eM) + k/(eR_w))`, the link finder's miss after `M` further trials and the
waits' miss after `R_w` reruns (`linkBoundZC` with no SHA-512 evaluations to count). -/
noncomputable def linkErrZC (L : Law n) (k Rw M : ℕ) (ρ : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (qsZC A P planZ tab vb L * (1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

/-- **The waits' error**: `Q_s/(1 − ρ)·k/(eR_w)` (`regTermZC` at no finder budget, and `candSlackZC`). -/
noncomputable def waitErrZC (L : Law n) (k Rw : ℕ) (ρ : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (qsZC A P planZ tab vb L * (k / (Real.exp 1 * Rw)) / (1 - ρ))

omit [Fintype K] in
theorem linkBoundZC_zero {k Rw M : ℕ} {ρ : ℝ} (h : LinkCRZC A Hs P planZ tab vb R τ k Rw M 0) :
    linkBoundZC A Hs P planZ tab vb k Rw M ρ 0 R τ = linkErrZC A P planZ tab vb L k Rw M ρ := by
  unfold linkBoundZC linkErrZC qsZC
  rw [ite_eq_left h]
  congr 1
  ring

omit [Fintype K] [DecidableEq K] in
theorem regTermZC_zero (L : Law n) (k Rw : ℕ) (ρ : ℝ) :
    regTermZC A P planZ tab vb L k Rw ρ 0 = waitErrZC A P planZ tab vb L k Rw ρ := by
  unfold regTermZC waitErrZC qsZC
  congr 1
  ring

omit [Fintype K] [DecidableEq K] in
theorem candSlackZC_eq (L : Law n) (k Rw : ℕ) (ρ : ℝ) :
    candSlackZC (L := L) A P planZ tab vb k Rw ρ = waitErrZC A P planZ tab vb L k Rw ρ := rfl

end Link

section Leaves

variable {M C β : Type} [Fintype C] [Nonempty C]

/-- **A property at some leaf** the strategy reaches. -/
def ExLeaves : (r : ℕ) → (f : List (M × C) → Game β) → ((t : List (M × C)) → Strategy (f t) → Prop) →
    Strategy (rounds M C r f) → Prop
  | 0, _, P, s => P [] s
  | r + 1, f, P, s => ∃ ρ, ExLeaves r (fun t => f ((s.1, ρ) :: t)) (fun t => P ((s.1, ρ) :: t)) (s.2 ρ)

theorem allLeaves_of_not_exLeaves : ∀ {r : ℕ} {f : List (M × C) → Game β}
    {P : (t : List (M × C)) → Strategy (f t) → Prop} {s : Strategy (rounds M C r f)},
    ¬ ExLeaves r f P s → AllLeaves r f (fun t s' => ¬ P t s') s
  | 0, _, _, _, h => h
  | _ + 1, _, _, _, h => fun ρ => allLeaves_of_not_exLeaves fun h' => h ⟨ρ, h'⟩

end Leaves

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm β : Type}
  {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}

/-- **A property of the prover's strategy at the start of some outer session it reaches**, after some draw `ω` and
transcript `t` (`AllAt`'s dual). -/
def ExAt (σ : Strategy (recGame L Reg I Cm outer))
    (P : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → Prop) : Prop :=
  ∃ ω, ExLeaves I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
    (fun t s => P ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω)

theorem allAt_of_not_exAt {σ : Strategy (recGame L Reg I Cm outer)}
    {P : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → Prop} (h : ¬ ExAt σ P) :
    AllAt σ fun ω t s => ¬ P ω t s :=
  fun ω => allLeaves_of_not_exLeaves fun h' => h ⟨ω, h'⟩

theorem AllAt.mono {σ : Strategy (recGame L Reg I Cm outer)}
    {P Q : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → Prop} (h : AllAt σ P)
    (hPQ : ∀ ω t s, P ω t s → Q ω t s) : AllAt σ Q :=
  fun ω => (h ω).mono fun t _ _ hp => hPQ ω t _ hp

theorem expAt_mono_all {σ : Strategy (recGame L Reg I Cm outer)}
    {F G : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → ℝ≥0∞}
    (h : AllAt σ fun ω t s => F ω t s ≤ G ω t s) : expAt σ F ≤ expAt σ G :=
  avg_mono fun ω => leafAvg_mono_all (h ω)

/-- **An expectation is at most another when it is at every outcome the strategy reaches.** -/
theorem exG_mono_outcome {α : Type} {F G : α → ℝ≥0∞} :
    ∀ (g : Game α) (s : Strategy g), (∀ a, Game.Outcome g s a → F a ≤ G a) → exG F g s ≤ exG G g s
  | .ret a, _, h => h a rfl
  | .send _ k, s, h => exG_mono_outcome (k s.1) s.2 h
  | @Game.coin _ _ _ _ k, s, h => avg_mono fun c => exG_mono_outcome (k c) (s c) fun a ha => h a ⟨c, ha⟩

end FlockSoundness.Discharge.Recursion
