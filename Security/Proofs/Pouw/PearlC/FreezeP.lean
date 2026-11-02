import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Proofs.Pouw.PearlC.Freeze
import Definitions.Pouw.PearlC.SkipP
import Definitions.Pouw.PearlC.Sm120

/-!
# P1 over any accumulator model, and on sm_120 (staging)

`Freeze`'s statements with the atom a parameter `p : Pipeline` (`Accum`'s chain and `SkipP`'s twins), instantiated at
`hopperE4m3K32`, where every twin is today's definition by `rfl`, and at the sm_120 atom `sm120E4m3K32 floor`
(`⟨[32], 26, floor⟩`, `Sm120`).

* **The floor** (`step_floor`): every nonzero term the step aligns has an exponent of at least −126 (the products at
  least −12, FP32 values and group outputs at least −126), so two floors at most −126 give the same step, for any
  width and groups. sm_120's floor, somewhere in `[−145, −126]`, is unobservable (`sm120_floor`); so is Hopper's −139.
* **(a) An identity needs domination** at width `w ≥ 24` (`identity_needs_dominationW`): on a normal accumulator `c`
  of exponent `e_c ≥ w − 127`, `D = acc` only if `c` is on the step's grid `g = 2^(E − (w − 1))` and the products
  truncated to `g` (`P`) leave `c` in place when truncated to `c`'s own ulp `u = 2^(e_c − 23)`: `trunc_u(c + P) = c`.
  So `|Σ a·b| < 32·g + u` (`identity_cellW`); on sm_120 with no product above `c`, `|Σ a·b| < 9·u`
  (`sm120_identity_cell`). At Hopper's width `u ≤ g` and the same statement holds with `P = 0`
  (`identity_needs_domination_hopper`, from `Freeze.identity_needs_domination`).
* **(b1) From `+0`** (`step_zero_iffW`, `24 ≤ w ≤ 115`): the atom is an identity exactly when the products truncated
  to the grid of their largest exponent sum to zero (`preSumW`; `preSumW (−139) 14 = preSum`).
* **(b) Confinement** (`identity_confinedW`): in a group from `+0` of `n` atoms with `(w − 1)·n ≤ 114` the running
  words stay normal (`tame_inWordW`), so an identity is (b1)'s on `+0` or (a)'s on a normal word; on sm_120 `n ≤ 4`
  (`sm120_identity_confined`), against Hopper's `n ≤ 7` (`Freeze.identity_confined`).
* **(c) Promotion** does not read the atom: `Freeze.promote_identity` and `Freeze.rnAdd_identity` apply as they are.
* **(d) Skipping** (`totalSkip_eqP`, `chainSkip_frozenP`, `skipOne_iffP`, `joint_completeP`, `jointFlags_completeP`,
  `joint_flagsCompleteP`): `Freeze`'s proofs, which read the atom only through the fold, over any `p` and promotion
  depth `G > 0`. At `hopperE4m3K32` and `G = 4` they are `Freeze`'s theorems (`joint_flagsComplete_hopper`).
-/

namespace Pouw.PearlC.FreezeP

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T Pouw.Fp8Atom.H1T.Proofs Pouw.PearlC Pouw.PearlC.Skip
  Pouw.PearlC.Freeze Pouw.PearlC.Accum Pouw.PearlC.SkipP

/-! ## The twins at `hopperE4m3K32` are today's definitions (the chain's are `Accum`'s `groupWordP_hopper`, …) -/

theorem stepP_hopper : stepP hopperE4m3K32 = Skip.step := rfl
theorem promoteGroupP_hopper : promoteGroupP hopperE4m3K32 = promoteGroup := rfl
theorem wordInP_hopper : wordInP hopperE4m3K32 = wordIn := rfl
theorem inTotalP_hopper : inTotalP hopperE4m3K32 = inTotal := rfl
theorem laterWordsP_hopper : laterWordsP hopperE4m3K32 = laterWords := rfl
theorem keepsP_hopper : keepsP hopperE4m3K32 = keeps := rfl
theorem jointAtP_hopper : jointAtP hopperE4m3K32 = jointAt := rfl
theorem jointFlagsP_hopper : jointFlagsP hopperE4m3K32 = jointFlags := rfl
theorem wordFlagsP_hopper : wordFlagsP hopperE4m3K32 4 = wordFlags := rfl
theorem unitFlagsP_hopper {m n k : ℕ} : @unitFlagsP hopperE4m3K32 m n k 4 = unitFlags := rfl

/-- `sm120Step` is `stepP` of the sm_120 atom at floor −133. -/
theorem sm120Step_eq (acc : Nat) (a b : Fin 32 → Code) : sm120Step acc a b = stepP (sm120E4m3K32 (-133)) acc (a, b) :=
  rfl

/-! ## The floor is unobservable at or below −126 -/

/-- A value whose finite term, if nonzero, has an exponent of at least −126. -/
def Floored (v : Value) : Prop := ∀ t, v = .fin t → t.mag ≠ 0 → -126 ≤ t.exp

theorem maxExp_floor (B z z' : ℤ) (hz : z ≤ B) (hz' : z' ≤ B) (ts : List Term)
    (h : ∀ t ∈ ts, t.mag ≠ 0 → B ≤ t.exp) : maxExp z ts = maxExp z' ts ∨ ∀ t ∈ ts, t.mag = 0 := by
  induction ts generalizing z z' with
  | nil => exact Or.inr fun t ht => by cases ht
  | cons t ts ih =>
    have h' : ∀ t ∈ ts, t.mag ≠ 0 → B ≤ t.exp := fun s hs => h s (List.mem_cons_of_mem _ hs)
    unfold maxExp at ih ⊢
    rw [List.foldl_cons, List.foldl_cons]
    by_cases hm : t.mag = 0
    · simp only [hm, ne_eq, not_true_eq_false, false_and, ite_false]
      rcases ih z z' hz hz' h' with e | e
      · exact Or.inl e
      · exact Or.inr fun s hs => by
          rcases List.mem_cons.1 hs with rfl | hs
          · exact hm
          · exact e s hs
    · left
      have hB := h t List.mem_cons_self hm
      have e1 : (if t.mag ≠ 0 ∧ z < t.exp then t.exp else z) = t.exp := by
        by_cases hlt : z < t.exp
        · rw [ite_eq_left ⟨hm, hlt⟩]
        · rw [ite_eq_right (fun h => hlt h.2)]; omega
      have e2 : (if t.mag ≠ 0 ∧ z' < t.exp then t.exp else z') = t.exp := by
        by_cases hlt : z' < t.exp
        · rw [ite_eq_left ⟨hm, hlt⟩]
        · rw [ite_eq_right (fun h => hlt h.2)]; omega
      rw [e1, e2]

theorem alignedSum_zero (w : ℕ) (E : ℤ) (ts : List Term) (h : ∀ t ∈ ts, t.mag = 0) : alignedSum w E ts = 0 := by
  rw [alignedSum]
  apply List.sum_eq_zero
  intro x hx
  obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hx
  simp [aligned, h t ht]

theorem groupSum_floor (w : ℕ) {z z' : ℤ} (hz : z ≤ -126) (hz' : z' ≤ -126) (ts : List Term)
    (h : ∀ t ∈ ts, t.mag ≠ 0 → -126 ≤ t.exp) : groupSum w z ts = groupSum w z' ts := by
  unfold groupSum
  rcases maxExp_floor (-126) z z' hz hz' ts h with he | h0
  · rw [he]
  · rw [alignedSum_zero _ _ _ h0, alignedSum_zero _ _ _ h0]; simp [normalize]

theorem floored_shape (M : ℕ) (neg : Bool) (e : ℤ) :
    (if M = 0 then Term.zero else (⟨neg, if e < -126 then -126 else e, M⟩ : Term)).mag ≠ 0 →
      -126 ≤ (if M = 0 then Term.zero else (⟨neg, if e < -126 then -126 else e, M⟩ : Term)).exp := by
  intro h
  by_cases hM : M = 0
  · rw [ite_eq_left hM] at h; exact absurd rfl h
  · rw [ite_eq_right hM]; dsimp only; split_ifs <;> omega

theorem normalize_floored (w : ℕ) (E total : ℤ) :
    (normalize w E total).mag ≠ 0 → -126 ≤ (normalize w E total).exp := by
  unfold normalize
  by_cases h0 : total = 0
  · rw [ite_eq_left h0]; intro h; exact absurd rfl h
  · rw [ite_eq_right h0]; exact floored_shape _ _ _

theorem fp32Value_floored (w : Nat) : Floored (fp32Value w) := by
  intro t ht hm
  unfold fp32Value at ht
  dsimp only at ht
  split_ifs at ht with h1 h2 h3 h4 <;> cases ht
  · simp [Term.zero] at hm
  · exact le_refl _
  · dsimp only; omega

theorem productTotal_floored (a b : Code) : Floored (productTotal a b) := by
  intro t ht hm
  unfold productTotal at ht
  by_cases hn : (a.isNaN || b.isNaN) = true
  · rw [ite_eq_left hn] at ht; exact Value.noConfusion ht
  · rw [ite_eq_right hn] at ht
    have e := Value.fin.inj ht
    have := (product_exp a b (e ▸ hm)).1
    rw [← e]; omega

theorem groupStepTotal_floored (w : ℕ) (z : ℤ) (acc : Value) (ps : List Value) :
    Floored (groupStepTotal w z acc ps) := by
  intro t ht hm
  unfold groupStepTotal at ht
  dsimp only at ht
  split_ifs at ht
  cases ht
  exact normalize_floored _ _ _ hm

theorem groupStepTotal_floor (w : ℕ) {z z' : ℤ} (hz : z ≤ -126) (hz' : z' ≤ -126) (acc : Value) (ps : List Value)
    (ha : Floored acc) (hp : ∀ v ∈ ps, Floored v) : groupStepTotal w z acc ps = groupStepTotal w z' acc ps := by
  have h : ∀ t ∈ (acc :: ps).filterMap Value.finite?, t.mag ≠ 0 → -126 ≤ t.exp := by
    intro t ht hm
    obtain ⟨v, hv, hvt⟩ := List.mem_filterMap.1 ht
    have hvf : v = .fin t := by cases v <;> simp_all [Value.finite?]
    rcases List.mem_cons.1 hv with rfl | hv
    · exact ha t hvf hm
    · exact hp v hv t hvf hm
  unfold groupStepTotal
  dsimp only
  rw [groupSum_floor w hz hz' _ h]

theorem fold_floor (p p' : Pipeline) (hw : p.width = p'.width) (hz : p.zeroExp ≤ -126) (hz' : p'.zeroExp ≤ -126)
    (gs : List Nat) (v : Value) (ab : List (Code × Code)) (hv : Floored v) : p.fold gs v ab = p'.fold gs v ab := by
  induction gs generalizing v ab with
  | nil => rfl
  | cons s gs ih =>
    simp only [Pipeline.fold]
    have hps : ∀ x ∈ (ab.take s).map (fun x => productTotal x.1 x.2), Floored x := by
      intro x hx
      obtain ⟨y, -, rfl⟩ := List.mem_map.1 hx
      exact productTotal_floored _ _
    rw [groupStepTotal_floor p.width hz hz' v _ hv hps, hw]
    exact ih _ _ (groupStepTotal_floored _ _ _ _)

/-- **The floor is unobservable**: two pipelines with the same groups and width and floors at most −126 have the same
step. -/
theorem step_floor (p p' : Pipeline) (hg : p.groups = p'.groups) (hw : p.width = p'.width) (hz : p.zeroExp ≤ -126)
    (hz' : p'.zeroExp ≤ -126) : p.step = p'.step := by
  funext acc a b
  unfold Pipeline.step
  rw [hg]
  exact congrArg packValue (fold_floor p p' hw hz hz' _ _ _ (fp32Value_floored acc))

/-- **sm_120's floor is unobservable** over `[−145, −126]` (and below). -/
theorem sm120_floor {z z' : ℤ} (hz : z ≤ -126) (hz' : z' ≤ -126) : (sm120E4m3K32 z).step = (sm120E4m3K32 z').step :=
  step_floor _ _ rfl rfl hz hz'

/-- Every floor in `[−145, −126]` gives `sm120Step`. -/
theorem sm120Step_floor {z : ℤ} (hz : z ≤ -126) : sm120StepAt z = sm120Step := by
  funext acc a b
  unfold sm120StepAt sm120Step sm120E4m3
  rw [sm120_floor hz (by norm_num : (-133 : ℤ) ≤ -126)]

/-- Hopper's floor −139 is unobservable too. -/
theorem hopper_floor {z : ℤ} (hz : z ≤ -126) : (⟨[32], 14, z⟩ : Pipeline).step = hopperE4m3K32.step :=
  step_floor _ _ rfl rfl hz (by norm_num [hopperE4m3K32])

/-! ## (d) Skipping in place, over any atom -/

section Skipping

variable (p : Pipeline)

/-- A group's identity atoms under `p`. -/
def idFlagsP (g : List AtomIn) : List Bool := fixFlags (stepP p) 0 g

/-- The promotion identities of a chain under `p`. -/
def promFlagsP (gs : List (List AtomIn)) : List Bool := fixFlags (promoteGroupP p) (some 0) gs

/-- The total under `p` with atoms and groups skipped in place: `am` flags each group's atoms, `gm` whole groups. -/
def totalSkipP (gs : List (List AtomIn)) (am : List (List Bool)) (gm : List Bool) : Option ℚ :=
  foldMask (fun t q => promote t (foldMask (stepP p) 0 q.1 q.2)) (some 0) (gs.zip am) gm

/-- The chain under `p` with atoms and groups skipped in place, its groups those of `chainP p G`. -/
def chainSkipP (G : Nat) (atoms : List AtomIn) (am : List (List Bool)) (gm : List Bool) : Option ℚ :=
  totalSkipP p (atoms.toChunks G) am gm

/-- The total under `p` with the atoms `m` flags in group `i` skipped in place. -/
def skipInP (gs : List (List AtomIn)) (i : Nat) (m : List Bool) : Option ℚ := totalSkipP p gs (groupMask gs i m) []

/-- The total under `p` with atom `j` of group `i` skipped alone. -/
def skipOneP (gs : List (List AtomIn)) (i j : Nat) : Option ℚ := skipInP p gs i (oneMask (gs.getD i []).length j)

/-- **Absorbed** under `p`: skipping atom `j` alone leaves the group's word, its promotion or the later promotions
unchanged. -/
def absorbedAtP (g : List AtomIn) (t : Option ℚ) (ws : List Nat) (j : Nat) : Bool :=
  decide (wordInP p g (oneMask g.length j) = groupWordP p g) ||
    decide (promote t (wordInP p g (oneMask g.length j)) = promote t (groupWordP p g)) ||
    decide (ws.foldl promote (promote t (wordInP p g (oneMask g.length j))) =
      ws.foldl promote (promote t (groupWordP p g)))

/-- Atom `j` of group `i` is absorbed under `p`. -/
def absorbedP (gs : List (List AtomIn)) (i j : Nat) : Bool :=
  absorbedAtP p (gs.getD i []) (inTotalP p gs i) (laterWordsP p gs i) j

/-- Atom `t` of the word over `atoms` is **skippable** under `p` and promotion every `G` atoms, in place, together
with some set of atoms of its group `t / G`. -/
def WordSkippableP (G : Nat) (atoms : List AtomIn) (t : Nat) : Prop :=
  ∃ M : List Bool, M.length = atoms.length ∧ M.getD t false = true ∧ (∀ s, M.getD s false = true → s / G = t / G) ∧
    chainSkipP p G atoms (M.toChunks G) [] = chainP p G atoms

theorem idFlagsP_hopper : idFlagsP hopperE4m3K32 = idFlags := rfl
theorem promFlagsP_hopper : promFlagsP hopperE4m3K32 = promFlags := rfl
theorem totalSkipP_hopper : totalSkipP hopperE4m3K32 = totalSkip := rfl
theorem chainSkipP_hopper : chainSkipP hopperE4m3K32 = chainSkip := rfl
theorem skipInP_hopper : skipInP hopperE4m3K32 = skipIn := rfl
theorem skipOneP_hopper : skipOneP hopperE4m3K32 = skipOne := rfl
theorem absorbedP_hopper : absorbedP hopperE4m3K32 = absorbed := rfl
theorem wordSkippableP_hopper : WordSkippableP hopperE4m3K32 4 = WordSkippable := rfl

theorem total_eq_foldlP (gs : List (List AtomIn)) : totalP p gs = gs.foldl (promoteGroupP p) (some 0) := by
  rw [totalP, List.foldl_map]; rfl

theorem totalSkip_auxP (t : Option ℚ) (gs : List (List AtomIn)) (am : List (List Bool)) (gm : List Bool)
    (ha : List.Forall₂ (fun m g => SubMask m (idFlagsP p g)) am gs)
    (hg : SubMask gm (fixFlags (promoteGroupP p) t gs)) :
    foldMask (fun t q => promote t (foldMask (stepP p) 0 q.1 q.2)) t (gs.zip am) gm =
      gs.foldl (promoteGroupP p) t := by
  induction gs generalizing t am gm with
  | nil => cases ha; cases hg; rfl
  | cons g gs ih =>
    cases ha with
    | cons hm ha =>
      rename_i m am
      cases hg with
      | cons hs hg =>
        rename_i s gm
        rw [List.zip_cons_cons, foldMask, List.foldl_cons]
        have hw : foldMask (stepP p) 0 g m = groupWordP p g := foldMask_fix (stepP p) 0 g m hm
        have : (if s then t else promote t (foldMask (stepP p) 0 (g, m).1 (g, m).2)) = promoteGroupP p t g := by
          rw [hw]
          cases s
          · rfl
          · rw [ite_eq_left rfl]; exact (of_decide_eq_true (hs rfl)).symm
        rw [this]
        exact ih _ _ _ ha hg

/-- **(d) The frozen atoms are jointly skippable**, under any atom. -/
theorem totalSkip_eqP (gs : List (List AtomIn)) (am : List (List Bool)) (gm : List Bool)
    (ha : List.Forall₂ (fun m g => SubMask m (idFlagsP p g)) am gs) (hg : SubMask gm (promFlagsP p gs)) :
    totalSkipP p gs am gm = totalP p gs := by
  rw [total_eq_foldlP]; exact totalSkip_auxP p _ gs am gm ha hg

/-- **(d) for the chain**, under any atom and promotion depth: skipping every identity atom and every
promotion-identity group at once leaves the chain's output unchanged. -/
theorem chainSkip_frozenP (G : Nat) (atoms : List AtomIn) :
    chainSkipP p G atoms ((atoms.toChunks G).map (idFlagsP p)) (promFlagsP p (atoms.toChunks G)) =
      chainP p G atoms := by
  apply totalSkip_eqP
  · generalize atoms.toChunks G = gs
    induction gs with
    | nil => exact List.Forall₂.nil
    | cons g gs ih => exact List.Forall₂.cons (subMask_refl _) ih
  · exact subMask_refl _

theorem foldl_promote_wordsP (gs : List (List AtomIn)) (t : Option ℚ) :
    (gs.map (groupWordP p)).foldl promote t = gs.foldl (promoteGroupP p) t := by
  rw [List.foldl_map]; rfl

theorem zip_replicate_foldlP (gs : List (List AtomIn)) (t : Option ℚ) :
    (gs.zip (gs.map fun g => List.replicate g.length false)).foldl
      (fun t q => promote t (foldMask (stepP p) 0 q.1 q.2)) t = gs.foldl (promoteGroupP p) t := by
  induction gs generalizing t with
  | nil => rfl
  | cons g gs ih =>
    simp only [List.map_cons, List.zip_cons_cons, List.foldl_cons]
    rw [foldMask_replicate]
    exact ih _

theorem skipIn_auxP (gs : List (List AtomIn)) (i : Nat) (m : List Bool) (t : Option ℚ) (hi : i < gs.length) :
    (gs.zip (groupMask gs i m)).foldl (fun t q => promote t (foldMask (stepP p) 0 q.1 q.2)) t =
      (gs.drop (i + 1)).foldl (promoteGroupP p)
        (promote ((gs.take i).foldl (promoteGroupP p) t) (wordInP p gs[i] m)) := by
  induction gs generalizing i t with
  | nil => simp at hi
  | cons g gs ih =>
    cases i with
    | zero =>
      simp only [groupMask, List.zip_cons_cons, List.foldl_cons]
      exact zip_replicate_foldlP p gs _
    | succ i =>
      simp only [groupMask, List.zip_cons_cons, List.foldl_cons]
      rw [foldMask_replicate]
      exact ih i _ (by simpa using hi)

/-- A skip inside one group is confined to it, under any atom. -/
theorem skipIn_eqP (gs : List (List AtomIn)) (i : Nat) (m : List Bool) (hi : i < gs.length) :
    skipInP p gs i m = (laterWordsP p gs i).foldl promote (promote (inTotalP p gs i) (wordInP p gs[i] m)) := by
  rw [laterWordsP, foldl_promote_wordsP, skipInP, totalSkipP, foldMask_nil]
  exact skipIn_auxP p gs i m _ hi

theorem total_splitP (gs : List (List AtomIn)) (i : Nat) (hi : i < gs.length) :
    totalP p gs = (laterWordsP p gs i).foldl promote (promote (inTotalP p gs i) (groupWordP p gs[i])) := by
  rw [laterWordsP, foldl_promote_wordsP, total_eq_foldlP]
  exact foldl_split _ _ i hi _

/-- **Completeness and soundness of `absorbedP`**: skipping atom `j` of group `i` alone leaves the output unchanged
exactly when the atom is absorbed. -/
theorem skipOne_iffP (gs : List (List AtomIn)) (i j : Nat) (hi : i < gs.length) :
    skipOneP p gs i j = totalP p gs ↔ absorbedP p gs i j = true := by
  rw [skipOneP, absorbedP, absorbedAtP, List.getD_eq_getElem _ _ hi, skipIn_eqP p gs i _ hi, total_splitP p gs i hi]
  simp only [Bool.or_eq_true, decide_eq_true_eq]
  constructor
  · exact fun h => Or.inr h
  · rintro ((h | h) | h)
    · rw [h]
    · rw [h]
    · exact h

theorem length_jointFlagsP (gs : List (List AtomIn)) (i : Nat) (hi : i < gs.length) :
    (jointFlagsP p gs i).length = gs[i].length := by
  rw [jointFlagsP, jointAtP, List.getD_eq_getElem _ _ hi]; simp

theorem jointFlags_getElemP (gs : List (List AtomIn)) (i : Nat) (hi : i < gs.length) (m : List Bool)
    (hm : m.length = gs[i].length) (h : skipInP p gs i m = totalP p gs) (j : Nat)
    (hj : j < (jointFlagsP p gs i).length) (hmj : m.getD j false = true) : (jointFlagsP p gs i)[j] = true := by
  rw [skipIn_eqP p gs i m hi, total_splitP p gs i hi] at h
  have hk : keepsP p gs[i] (inTotalP p gs i) (laterWordsP p gs i) m = true := by simp [keepsP, h]
  simp only [jointFlagsP, jointAtP, List.getD_eq_getElem _ _ hi, List.getElem_map, List.getElem_range]
  rw [List.any_eq_true]
  exact ⟨m, hm ▸ mem_allMasks m, by rw [Bool.and_eq_true]; exact ⟨hmj, hk⟩⟩

/-- **Completeness for one group**, under any atom: every set of one group's atoms whose joint skip leaves the output
unchanged is flagged. -/
theorem joint_completeP (gs : List (List AtomIn)) (i : Nat) (hi : i < gs.length) (m : List Bool)
    (hm : m.length = gs[i].length) (h : skipInP p gs i m = totalP p gs) : SubMask m (jointFlagsP p gs i) := by
  apply subMask_of_getElem _ _ (by rw [length_jointFlagsP p gs i hi, hm])
  intro j h1 h2 hmj
  exact jointFlags_getElemP p gs i hi m hm h j h2 (by rw [List.getD_eq_getElem _ _ h1]; exact hmj)

/-- Absorbed atoms are jointly flagged, under any atom. -/
theorem joint_of_absorbedP (gs : List (List AtomIn)) (i j : Nat) (hi : i < gs.length)
    (hj : j < (jointFlagsP p gs i).length) (h : absorbedP p gs i j = true) : (jointFlagsP p gs i)[j] = true := by
  have hj' : j < gs[i].length := by rwa [length_jointFlagsP p gs i hi] at hj
  have hs := (skipOne_iffP p gs i j hi).2 h
  rw [skipOneP, List.getD_eq_getElem _ _ hi] at hs
  exact jointFlags_getElemP p gs i hi _ (length_oneMask _ _) hs j hj (oneMask_getD _ _ hj')

/-- **Completeness of the joint flags on a word**, under any atom and promotion depth `G > 0`. -/
theorem jointFlags_completeP {G : Nat} (hG : 0 < G) (atoms : List AtomIn) (t : Nat)
    (h : WordSkippableP p G atoms t) : (jointFlagsP p (atoms.toChunks G) (t / G)).getD (t % G) false = true := by
  obtain ⟨M, hM, hMt, hMs, hs⟩ := h
  have ht : t < atoms.length := by
    by_contra hc
    rw [List.getD_eq_default _ _ (by omega)] at hMt
    exact Bool.noConfusion hMt
  have hi := lt_length_toChunks hG ht
  have hL : (M.toChunks G).length = (atoms.toChunks G).length := by rw [length_toChunks hG, length_toChunks hG, hM]
  have hc : ∀ i' (h1 : i' < (M.toChunks G).length),
      (M.toChunks G)[i'].length = ((atoms.toChunks G)[i']'(hL ▸ h1)).length := by
    intro i' h1; rw [length_getElem_toChunks hG, length_getElem_toChunks hG, hM]
  have hr : ∀ i' (h1 : i' < (M.toChunks G).length), i' ≠ t / G →
      (M.toChunks G)[i'] = List.replicate ((atoms.toChunks G)[i']'(hL ▸ h1)).length false := by
    intro i' h1 hne
    refine List.ext_getElem (by rw [List.length_replicate, hc]) fun q hq _ => ?_
    rw [List.getElem_replicate]
    have hqG : q < G := by rw [length_getElem_toChunks hG] at hq; omega
    simp only [getElem_toChunks hG, List.getElem_take, List.getElem_drop]
    by_contra hf
    have := hMs (G * i' + q) (by rw [List.getD_eq_getElem]; simpa using hf)
    have hdiv : (G * i' + q) / G = i' := by
      rw [Nat.add_comm, Nat.add_mul_div_left _ _ hG, Nat.div_eq_of_lt hqG, zero_add]
    exact hne (hdiv ▸ this)
  have hLg := eq_groupMask _ _ hi _ hL hr
  have hskip : skipInP p (atoms.toChunks G) (t / G) ((M.toChunks G)[t / G]'(hL ▸ hi)) =
      totalP p (atoms.toChunks G) := by
    rw [skipInP, ← hLg]; exact hs
  have hj : t % G < (jointFlagsP p (atoms.toChunks G) (t / G)).length := by
    rw [length_jointFlagsP p _ _ hi]; exact mod_lt_length_getElem_toChunks hG ht
  have hmj : ((M.toChunks G)[t / G]'(hL ▸ hi)).getD (t % G) false = true := by
    rw [← List.getD_eq_getElem _ [], getD_getD_toChunks hG]; exact hMt
  rw [List.getD_eq_getElem _ _ hj]
  exact jointFlags_getElemP p _ _ hi _ (hc _ _) hskip _ hj hmj

theorem wordFlags_completeP {G : Nat} (hG : 0 < G) (k : ℕ) (a b : Fin k → Code) (t : ℕ)
    (h : WordSkippableP p G (rowAtoms a b) t) : wordFlagsP p G k a b t = true :=
  jointFlags_completeP p hG _ t h

/-- **The joint flags are complete** for in-group skips, under any atom and promotion depth `G > 0`. -/
theorem joint_flagsCompleteP {G : Nat} (hG : 0 < G) {m n k : ℕ} (Aq : Matrix (Fin m) (Fin k) Code)
    (Bq : Matrix (Fin n) (Fin k) Code) :
    ∀ i j t, WordSkippableP p G (rowAtoms (Aq i) (Bq j)) t → unitFlagsP p G Aq Bq i j t = true :=
  fun i j t h => wordFlags_completeP p hG k (Aq i) (Bq j) t h

end Skipping

/-- At `hopperE4m3K32` and `G = 4`, `joint_flagsCompleteP` is `Freeze.joint_flagsComplete`'s statement. -/
theorem joint_flagsComplete_hopper {m n k : ℕ} (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) :
    ∀ i j t, WordSkippable (rowAtoms (Aq i) (Bq j)) t → unitFlags Aq Bq i j t = true :=
  joint_flagsCompleteP hopperE4m3K32 (by decide) Aq Bq

/-- On sm_120, at any floor and promotion depth `G > 0`, the joint flags are complete for in-group skips. -/
theorem sm120_joint_flagsComplete (z : ℤ) {G : Nat} (hG : 0 < G) {m n k : ℕ} (Aq : Matrix (Fin m) (Fin k) Code)
    (Bq : Matrix (Fin n) (Fin k) Code) :
    ∀ i j t, WordSkippableP (sm120E4m3K32 z) G (rowAtoms (Aq i) (Bq j)) t →
      unitFlagsP (sm120E4m3K32 z) G Aq Bq i j t = true :=
  joint_flagsCompleteP _ hG Aq Bq

/-! ## (a) An identity needs domination, at width `w ≥ 24` -/

/-- The step's alignment exponent on the accumulator term `tc` under the floor `z`. -/
def stepExpW (z : ℤ) (tc : Term) (a b : Fin 32 → Code) : ℤ := maxExp z (tc :: List.ofFn fun l => product (a l) (b l))

theorem stepExpW_hopper : stepExpW (-139) = stepExp := rfl

/-- The step of one group of 32 on a finite accumulator term, on finite codes. -/
theorem step_fin {w : ℕ} {z : ℤ} {acc : Nat} {tc : Term} (hw : fp32Value acc = .fin tc) (a b : Fin 32 → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) :
    (Pipeline.mk [32] w z).step acc (List.ofFn a) (List.ofFn b) = packValue
      (let t := groupSum w z (tc :: List.ofFn fun l => product (a l) (b l))
       if t.mag ≠ 0 ∧ 127 < t.exp then .inf t.neg else .fin t) := by
  unfold Pipeline.step
  simp only [Pipeline.fold, hw, zip_ofFn]
  congr 1
  rw [List.take_of_length_le (by simp), List.map_ofFn]
  have hp : ((fun x : Code × Code => productTotal x.1 x.2) ∘ fun l => (a l, b l)) =
      fun l => Value.fin (product (a l) (b l)) := by
    funext l; simp [productTotal, ha l, hb l]
  rw [hp]
  unfold groupStepTotal
  have h1 : ((Value.fin tc :: List.ofFn fun l => Value.fin (product (a l) (b l))).any Value.isNaN) = false := by
    rw [List.any_eq_false]; intro v hv
    simp only [List.mem_cons, List.mem_ofFn] at hv
    rcases hv with rfl | ⟨l, rfl⟩ <;> simp [Value.isNaN]
  have h2 : ∀ s, ((Value.fin tc :: List.ofFn fun l => Value.fin (product (a l) (b l))).any (Value.isInf s)) =
      false := by
    intro s; rw [List.any_eq_false]; intro v hv
    simp only [List.mem_cons, List.mem_ofFn] at hv
    rcases hv with rfl | ⟨l, rfl⟩ <;> simp [Value.isInf]
  have h3 : (Value.fin tc :: List.ofFn fun l => Value.fin (product (a l) (b l))).filterMap Value.finite? =
      tc :: List.ofFn fun l => product (a l) (b l) := by
    simp [Value.finite?]
  simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.and_self]

/-- At width `w ≥ 24` an aligned term, on the grid `2^(E − w + 1)`, is its value truncated toward zero to the grid. -/
theorem aligned_valueW {w : ℕ} (hw : 24 ≤ w) {E : ℤ} {t : Term} (hE : t.mag ≠ 0 → t.exp ≤ E) :
    (aligned w E t : ℚ) * 2 ^ (E - w + 1) = truncTo (2 ^ (E - w + 1)) t.value := by
  by_cases hm : t.mag = 0
  · have : t.value = 0 := by simp [Term.value, hm]
    rw [this, truncTo_zero]; simp [aligned, hm]
  · have hle := hE hm
    have hs : (((E - t.exp).toNat : ℕ) : ℤ) = E - t.exp := Int.toNat_of_nonneg (by omega)
    have hal : aligned w E t =
        (if t.neg then -1 else 1) * (((t.mag * 2 ^ (w - 24)) / 2 ^ (E - t.exp).toNat : ℕ) : ℤ) := by
      unfold aligned toAdder
      rw [ite_eq_right hm, ite_eq_left hw, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]
      dsimp only
      split <;> simp
    have e : (2 : ℚ) ^ (t.exp - 23) =
        (2 : ℚ) ^ ((w - 24 : ℕ) : ℤ) / 2 ^ (((E - t.exp).toNat : ℕ) : ℤ) * 2 ^ (E - w + 1) := by
      rw [← zpow_sub₀ two_ne_zero, ← zpow_add₀ two_ne_zero, hs, Nat.cast_sub hw]; congr 1; push_cast; ring
    rw [zpow_natCast, zpow_natCast] at e
    have hval : t.value = ((if t.neg then -1 else 1) *
        (((t.mag * 2 ^ (w - 24) : ℕ) : ℚ) / 2 ^ (E - t.exp).toNat)) * 2 ^ (E - w + 1) := by
      unfold Term.value
      rw [e]; push_cast; ring
    have hz : aligned w E t = truncZ ((if t.neg then -1 else 1) *
        (((t.mag * 2 ^ (w - 24) : ℕ) : ℚ) / 2 ^ (E - t.exp).toNat)) := by
      rw [hal]
      generalize t.mag * 2 ^ (w - 24) = M
      generalize (E - t.exp).toNat = s
      have hq : ((M / 2 ^ s : ℕ) : ℤ) = (⌊(M : ℚ) / 2 ^ s⌋₊ : ℤ) := by
        congr 1; rw [← Nat.floor_div_eq_div (K := ℚ)]; simp
      have hnn : 0 ≤ (M : ℚ) / 2 ^ s := by positivity
      unfold truncZ
      cases t.neg
      · simp [hq, hnn]
      · by_cases h0 : (M : ℚ) / 2 ^ s = 0
        · simp [h0, hq]
        · have : ¬ (0 ≤ -((M : ℚ) / 2 ^ s)) := by intro h; exact h0 (by linarith)
          simp [hq, this]
    rw [hval, truncTo_mul (two_zpow_pos _), hz]

theorem floor_shiftQ (q n L : ℕ) : (⌊(n : ℚ) * 2 ^ ((q : ℤ) - 1 - L)⌋ : ℚ) =
    ((if q < L + 1 then n >>> (L + 1 - q) else n <<< (q - (L + 1)) : ℕ) : ℚ) := by
  split_ifs with h
  · rw [Nat.shiftRight_eq_div_pow, ← Nat.floor_div_eq_div (K := ℚ)]
    have : (q : ℤ) - 1 - L = -((L + 1 - q : ℕ) : ℤ) := by omega
    rw [this, zpow_neg, zpow_natCast, ← div_eq_mul_inv]
    have hnn : (0 : ℚ) ≤ (n : ℚ) / 2 ^ (L + 1 - q) := by positivity
    rw [← Int.natCast_floor_eq_floor hnn]
    push_cast; rfl
  · rw [Nat.shiftLeft_eq]
    have : (q : ℤ) - 1 - L = ((q - (L + 1) : ℕ) : ℤ) := by omega
    rw [this, zpow_natCast]
    have e : (n : ℚ) * 2 ^ (q - (L + 1)) = ((n * 2 ^ (q - (L + 1)) : ℕ) : ℤ) := by push_cast; ring
    rw [e, Int.floor_intCast]; push_cast; ring

/-- `normalize` at width `w ≥ 24` on a nonzero sum whose exponent is at least −126: a normal term at that exponent,
the sum on the grid `2^(E − w + 1)` truncated toward zero to 24 significant bits. -/
theorem normalize_specW {w : ℕ} (hw : 24 ≤ w) {E total : ℤ} (h0 : total ≠ 0)
    (he : -126 ≤ E + (bitLength total.natAbs : ℤ) - w) :
    2 ^ 23 ≤ (normalize w E total).mag ∧ (normalize w E total).mag < 2 ^ 24 ∧
      (normalize w E total).exp = E + (bitLength total.natAbs : ℤ) - w ∧
      (normalize w E total).value =
        truncTo (2 ^ (E + (bitLength total.natAbs : ℤ) - w - 23)) ((total : ℚ) * 2 ^ (E - w + 1)) := by
  have hn : total.natAbs ≠ 0 := by omega
  have hL1 := Nat.log2_self_le hn
  have hL2 := @Nat.lt_log2_self total.natAbs
  have hq := floor_shiftQ 24 total.natAbs total.natAbs.log2
  set L := total.natAbs.log2 with hLdef
  have hbl : bitLength total.natAbs = L + 1 := by unfold bitLength; rw [ite_eq_right hn]
  set m1 := (if w < L + 1 then total.natAbs >>> (L + 1 - w) else total.natAbs <<< (w - (L + 1))) with hm1
  set m := (if 24 < L + 1 then total.natAbs >>> (L + 1 - 24) else total.natAbs <<< (24 - (L + 1))) with hm
  have hfa : fromAdder w m1 = m := by
    rw [fromAdder, ite_eq_left hw, hm1, hm]
    simp only [Nat.shiftRight_eq_div_pow, Nat.shiftLeft_eq]
    split_ifs with h1 h2 h2
    · rw [Nat.div_div_eq_div_mul, ← pow_add]; congr 2; omega
    · omega
    · rw [show w - 24 = (w - (L + 1)) + (L + 1 - 24) by omega, pow_add, ← Nat.div_div_eq_div_mul,
        Nat.mul_div_cancel _ (by positivity)]
    · rw [show w - (L + 1) = (24 - (L + 1)) + (w - 24) by omega, pow_add, ← mul_assoc,
        Nat.mul_div_cancel _ (by positivity)]
  have hb1 : 2 ^ 23 ≤ m ∧ m < 2 ^ 24 := by
    rw [hm]; split_ifs with h
    · rw [Nat.shiftRight_eq_div_pow]
      constructor
      · rw [Nat.le_div_iff_mul_le (by positivity), ← pow_add]; rwa [show 23 + (L + 1 - 24) = L by omega]
      · rw [Nat.div_lt_iff_lt_mul (by positivity), ← pow_add]; rwa [show 24 + (L + 1 - 24) = L + 1 by omega]
    · rw [Nat.shiftLeft_eq]
      constructor
      · calc 2 ^ 23 = 2 ^ L * 2 ^ (24 - (L + 1)) := by rw [← pow_add]; congr 1; omega
          _ ≤ _ := Nat.mul_le_mul_right _ hL1
      · calc total.natAbs * 2 ^ (24 - (L + 1)) < 2 ^ (L + 1) * 2 ^ (24 - (L + 1)) :=
              Nat.mul_lt_mul_of_pos_right hL2 (by positivity)
          _ = 2 ^ 24 := by rw [← pow_add]; congr 1; omega
  rw [hbl] at he ⊢
  have hne : ¬ (E + ((L + 1 : ℕ) : ℤ) - (w : ℤ) < -126) := by omega
  unfold normalize
  simp only [h0, ite_false, hbl, hne, ← hm1]
  rw [hfa, ite_eq_right (by omega)]
  dsimp only
  refine ⟨hb1.1, hb1.2, rfl, ?_⟩
  have hy0 : (total : ℚ) * 2 ^ (E - w + 1) ≠ 0 := mul_ne_zero (by exact_mod_cast h0) (two_zpow_ne _)
  have habs : |(total : ℚ) * 2 ^ (E - w + 1)| = (total.natAbs : ℚ) * 2 ^ (E - w + 1) := by
    rw [abs_mul, abs_of_pos (two_zpow_pos _), Nat.cast_natAbs, Int.cast_abs]
  unfold truncTo Term.value
  rw [habs]
  have hdiv : (total.natAbs : ℚ) * 2 ^ (E - w + 1) / 2 ^ (E + ((L + 1 : ℕ) : ℤ) - w - 23) =
      (total.natAbs : ℚ) * 2 ^ (((24 : ℕ) : ℤ) - 1 - L) := by
    rw [mul_div_assoc, ← zpow_sub₀ two_ne_zero]; congr 2; push_cast; ring
  rw [hdiv, hq]
  have hs : (total : ℚ) * 2 ^ (E - w + 1) < 0 ↔ total < 0 := by
    rw [mul_neg_iff]; constructor
    · rintro (⟨-, h⟩ | ⟨h, -⟩)
      · exact absurd h (not_lt.2 (two_zpow_pos _).le)
      · exact_mod_cast h
    · intro h; right; exact ⟨by exact_mod_cast h, two_zpow_pos _⟩
  by_cases hneg : total < 0
  · simp only [hneg, decide_true, ite_true, hs.2 hneg]
  · simp only [hneg, decide_false, ite_false, Bool.false_eq_true, (not_congr hs).2 hneg]

theorem term_value_int (t : Term) :
    t.value = (((if t.neg then -1 else 1) * t.mag : ℤ) : ℚ) * 2 ^ (t.exp - 23) := by
  unfold Term.value; split <;> push_cast <;> ring

theorem truncTo_pow_mul {G H : ℤ} (h : G ≤ H) (n : ℤ) : truncTo (2 ^ G) ((n : ℚ) * 2 ^ H) = n * 2 ^ H := by
  have e : (n : ℚ) * 2 ^ H = ((n * 2 ^ (H - G).toNat : ℤ) : ℚ) * 2 ^ G := by
    push_cast
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega), mul_assoc, ← zpow_add₀ two_ne_zero]
    congr 2; ring
  rw [e, truncTo_intMul (two_zpow_pos _)]

/-- **(a) An identity needs domination**, at any width `w ≥ 24` and floor. On finite codes and a normal accumulator
`acc` of value `c` and exponent `e_c ≥ w − 127`, `D = acc` only if `c` lies on the step's grid `g = 2^(E − w + 1)`
(`E = stepExpW`) and the products truncated toward zero to `g`, added to `c` and truncated to `c`'s ulp
`2^(e_c − 23)`, give `c`. -/
theorem identity_needs_dominationW {w : ℕ} (hw24 : 24 ≤ w) {z : ℤ} {acc : Nat} {tc : Term}
    (hw : fp32Value acc = .fin tc) (hm : 2 ^ 23 ≤ tc.mag) (he : (w : ℤ) - 127 ≤ tc.exp)
    (a b : Fin 32 → Code) (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false)
    (hid : (Pipeline.mk [32] w z).step acc (List.ofFn a) (List.ofFn b) = acc) :
    truncTo (2 ^ (stepExpW z tc a b - w + 1)) tc.value = tc.value ∧
      truncTo (2 ^ (tc.exp - 23))
        (tc.value + ∑ l, truncTo (2 ^ (stepExpW z tc a b - w + 1)) (val (a l) * val (b l))) = tc.value := by
  obtain ⟨hm2, he2⟩ := fp32Value_fin hw
  set E := stepExpW z tc a b with hEdef
  have hcE : tc.exp ≤ E := le_maxExp _ _ tc List.mem_cons_self (by omega)
  have hle : ∀ l, (product (a l) (b l)).mag ≠ 0 → (product (a l) (b l)).exp ≤ E := fun l hp =>
    le_maxExp _ _ _ (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨l, rfl⟩)) hp
  set total := alignedSum w E (tc :: List.ofFn fun l => product (a l) (b l)) with htdef
  rw [step_fin hw a b ha hb] at hid
  unfold groupSum at hid
  change packValue (let t := normalize w E total
    if t.mag ≠ 0 ∧ 127 < t.exp then Value.inf t.neg else Value.fin t) = acc at hid
  clear_value total E
  have hsum : total = aligned w E tc + ∑ l, aligned w E (product (a l) (b l)) := by
    rw [htdef, alignedSum, List.map_cons, List.sum_cons, List.map_ofFn, List.sum_ofFn]
    simp only [Function.comp_apply]
  have htot : (total : ℚ) * 2 ^ (E - w + 1) =
      truncTo (2 ^ (E - w + 1)) tc.value + ∑ l, truncTo (2 ^ (E - w + 1)) (val (a l) * val (b l)) := by
    rw [hsum]; push_cast
    rw [add_mul, Finset.sum_mul, aligned_valueW hw24 (fun _ => hcE), add_right_inj]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [aligned_valueW hw24 (hle l), productExact _ _ _ _ (val_eq (ha l)) (val_eq (hb l))]
  have hval : wordQ acc = some tc.value := by simp [wordQ, hw, Value.finite?]
  by_cases h0 : total = 0
  · rw [h0] at hid
    simp [normalize, Term.zero, packValue, packTerm] at hid
    rw [← hid, fp32Value_zero] at hw
    cases hw; simp [Term.zero] at hm
  have hbl1 : 1 ≤ bitLength total.natAbs := by unfold bitLength; rw [ite_eq_right (by omega)]; omega
  obtain ⟨m1, m2, e1, hv⟩ := normalize_specW hw24 h0 (E := E) (by omega)
  dsimp only at hid
  split_ifs at hid with hinf
  · have : fp32Value acc = .inf (normalize w E total).neg := by
      rw [← hid]; cases (normalize w E total).neg <;> rfl
    rw [hw] at this; cases this
  · have e2 : (normalize w E total).exp ≤ 127 := by
      by_contra hc; exact hinf ⟨by omega, by omega⟩
    have hq := wordQ_packTerm m1 m2 (by omega) e2
    simp only [packValue] at hid
    rw [hid, hval, Option.some.injEq] at hq
    have hexp : (normalize w E total).exp = tc.exp := by
      have h1 := binade_term m1 m2
      rw [← hq, binade_term hm hm2] at h1
      exact h1.symm
    have hE' : E + (bitLength total.natAbs : ℤ) - w = tc.exp := by rw [← e1]; exact hexp
    rw [hv, hE'] at hq
    have hc : truncTo (2 ^ (E - w + 1)) tc.value = tc.value := by
      by_cases hge : E - w + 1 ≤ tc.exp - 23
      · rw [term_value_int tc]; exact truncTo_pow_mul hge _
      · rw [hq, truncTo_pow_mul (by omega) total]; exact truncTo_pow_mul le_rfl total
    refine ⟨hc, ?_⟩
    rw [hc] at htot
    rw [← htot]; exact hq.symm

theorem abs_sum_sub_truncTo_lt {g : ℚ} (hg : 0 < g) (p : Fin 32 → ℚ) :
    |∑ l, p l - ∑ l, truncTo g (p l)| < 32 * g := by
  rw [← Finset.sum_sub_distrib]
  calc |∑ l, (p l - truncTo g (p l))| ≤ ∑ l, |p l - truncTo g (p l)| := Finset.abs_sum_le_sum_abs _ _
    _ < ∑ _l : Fin 32, g :=
        Finset.sum_lt_sum_of_nonempty Finset.univ_nonempty fun l _ => abs_sub_truncTo_lt hg _
    _ = 32 * g := by simp

/-- With no product above the accumulator's exponent, the step's alignment exponent is the accumulator's. -/
theorem stepExpW_eq {z : ℤ} {tc : Term} (a b : Fin 32 → Code) (hm : tc.mag ≠ 0) (he : z ≤ tc.exp)
    (h : ∀ l, (product (a l) (b l)).mag ≠ 0 → (product (a l) (b l)).exp ≤ tc.exp) :
    stepExpW z tc a b = tc.exp := by
  refine le_antisymm (maxExp_le _ _ _ he fun t ht htm => ?_) (le_maxExp _ _ tc List.mem_cons_self hm)
  rcases List.mem_cons.1 ht with rfl | ht
  · exact le_rfl
  · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
    exact h l htm

/-- **(a), the cell**, at width `w ≥ 24`: under `identity_needs_dominationW`'s hypotheses the exact product sum is
below `32·g + u`, `u = 2^(e_c − 23)` the accumulator's ulp, and `g`'s exponent is at least `e_c`'s, with equality when
no product exponent exceeds `e_c`. -/
theorem identity_cellW {w : ℕ} (hw24 : 24 ≤ w) {z : ℤ} {acc : Nat} {tc : Term}
    (hw : fp32Value acc = .fin tc) (hm : 2 ^ 23 ≤ tc.mag) (he : (w : ℤ) - 127 ≤ tc.exp)
    (a b : Fin 32 → Code) (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false)
    (hid : (Pipeline.mk [32] w z).step acc (List.ofFn a) (List.ofFn b) = acc) :
    |∑ l, val (a l) * val (b l)| < 32 * 2 ^ (stepExpW z tc a b - w + 1) + 2 ^ (tc.exp - 23) ∧
      tc.exp ≤ stepExpW z tc a b ∧
      (z ≤ tc.exp → (∀ l, (product (a l) (b l)).mag ≠ 0 → (product (a l) (b l)).exp ≤ tc.exp) →
        stepExpW z tc a b = tc.exp) := by
  obtain ⟨-, h2⟩ := identity_needs_dominationW hw24 hw hm he a b ha hb hid
  have hu : (0 : ℚ) < 2 ^ (tc.exp - 23) := two_zpow_pos _
  have hP := abs_sub_truncTo_lt hu
    (tc.value + ∑ l, truncTo (2 ^ (stepExpW z tc a b - w + 1)) (val (a l) * val (b l)))
  rw [h2, add_sub_cancel_left] at hP
  have hS := abs_sum_sub_truncTo_lt (two_zpow_pos (stepExpW z tc a b - w + 1)) fun l => val (a l) * val (b l)
  refine ⟨?_, le_maxExp _ _ tc List.mem_cons_self (by omega), fun hz h => stepExpW_eq a b (by omega) hz h⟩
  have := abs_add_le (∑ l, val (a l) * val (b l) -
      ∑ l, truncTo (2 ^ (stepExpW z tc a b - w + 1)) (val (a l) * val (b l)))
    (∑ l, truncTo (2 ^ (stepExpW z tc a b - w + 1)) (val (a l) * val (b l)))
  rw [sub_add_cancel] at this
  linarith

/-- **(a) on sm_120**: at any floor at most −126, an identity on a normal accumulator of exponent `e_c ≥ −101` whose
products do not exceed `e_c` needs `|Σ a·b| < 9·2^(e_c − 23)`, nine ulps of the accumulator. -/
theorem sm120_identity_cell {z : ℤ} (hz : z ≤ -126) {acc : Nat} {tc : Term} (hw : fp32Value acc = .fin tc)
    (hm : 2 ^ 23 ≤ tc.mag) (he : -101 ≤ tc.exp) (a b : Fin 32 → Code) (ha : ∀ l, (a l).isNaN = false)
    (hb : ∀ l, (b l).isNaN = false) (hid : (sm120E4m3K32 z).step acc (List.ofFn a) (List.ofFn b) = acc)
    (hdom : ∀ l, (product (a l) (b l)).mag ≠ 0 → (product (a l) (b l)).exp ≤ tc.exp) :
    |∑ l, val (a l) * val (b l)| < 9 * 2 ^ (tc.exp - 23) := by
  obtain ⟨h1, -, h3⟩ := identity_cellW (w := 26) (by norm_num) hw hm (by push_cast; omega) a b ha hb hid
  rw [h3 (by omega) hdom] at h1
  have h4 : (2 : ℚ) ^ (tc.exp - 23) = 2 ^ (tc.exp - ((26 : ℕ) : ℤ) + 1) * 2 ^ (2 : ℤ) := by
    rw [← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
  rw [h4] at h1 ⊢
  norm_num at h1 ⊢
  linarith

/-- **(a) at `hopperE4m3K32`**, in `identity_needs_dominationW`'s form: at width 14 the accumulator's ulp is below the
grid, and the truncated products sum to zero (`Freeze.identity_needs_domination`). -/
theorem identity_needs_domination_hopper {acc : Nat} {tc : Term} (hw : fp32Value acc = .fin tc)
    (hm : 2 ^ 23 ≤ tc.mag) (he : ((14 : ℕ) : ℤ) - 127 ≤ tc.exp) (a b : Fin 32 → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false)
    (hid : hopperE4m3K32.step acc (List.ofFn a) (List.ofFn b) = acc) :
    truncTo (2 ^ (stepExpW (-139) tc a b - (14 : ℕ) + 1)) tc.value = tc.value ∧
      truncTo (2 ^ (tc.exp - 23))
        (tc.value + ∑ l, truncTo (2 ^ (stepExpW (-139) tc a b - (14 : ℕ) + 1)) (val (a l) * val (b l))) =
        tc.value := by
  have hg : stepExpW (-139) tc a b - ((14 : ℕ) : ℤ) + 1 = stepExp tc a b - 13 := by
    rw [stepExpW_hopper]; push_cast; ring
  rw [hg]
  obtain ⟨h1, h2⟩ := identity_needs_domination hw hm (by push_cast at he; omega) a b ha hb hid
  refine ⟨h1, ?_⟩
  rw [h2, add_zero, term_value_int tc]
  exact truncTo_pow_mul le_rfl _

/-! ## (b1) From `+0` -/

/-- The products truncated toward zero to the grid `2^(E − w + 1)` of their largest exponent `E` (at least `z`), summed. -/
def preSumW (z : ℤ) (w : ℕ) (a b : Fin 32 → Code) : ℚ :=
  ∑ l, truncTo (2 ^ (maxExp z (List.ofFn fun l => product (a l) (b l)) - w + 1)) (val (a l) * val (b l))

theorem preSumW_hopper : preSumW (-139) 14 = preSum := by
  have h : ∀ E : ℤ, E - ((14 : ℕ) : ℤ) + 1 = E - 13 := fun E => by push_cast; ring
  funext a b
  simp only [preSumW, preSum, gridExp, h]

theorem packValue_inf_ne (s : Bool) : packValue (.inf s) ≠ 0 := by cases s <;> decide

/-- **(b1)** at width `24 ≤ w ≤ 115` and any floor: from `+0`, on finite codes, the atom is an identity exactly when
`preSumW = 0`. -/
theorem step_zero_iffW {w : ℕ} (hw24 : 24 ≤ w) (hw : w ≤ 115) (z : ℤ) (a b : Fin 32 → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) :
    (Pipeline.mk [32] w z).step 0 (List.ofFn a) (List.ofFn b) = 0 ↔ preSumW z w a b = 0 := by
  rw [step_fin fp32Value_zero a b ha hb]
  unfold groupSum
  rw [maxExp_zero_cons]
  set E := maxExp z (List.ofFn fun l => product (a l) (b l)) with hEdef
  have hle : ∀ l, (product (a l) (b l)).mag ≠ 0 → (product (a l) (b l)).exp ≤ E := fun l hm =>
    le_maxExp _ _ _ (List.mem_ofFn.2 ⟨l, rfl⟩) hm
  have hsum : alignedSum w E (Term.zero :: List.ofFn fun l => product (a l) (b l)) =
      ∑ l, aligned w E (product (a l) (b l)) := by
    have hz : aligned w E Term.zero = 0 := by simp [aligned, Term.zero]
    rw [alignedSum, List.map_cons, List.sum_cons, hz, zero_add, List.map_ofFn, List.sum_ofFn]
    simp only [Function.comp_apply]
  set total := alignedSum w E (Term.zero :: List.ofFn fun l => product (a l) (b l)) with htdef
  have htot : (total : ℚ) * 2 ^ (E - w + 1) = preSumW z w a b := by
    rw [hsum, preSumW]; push_cast; rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [aligned_valueW hw24 (hle l), productExact _ _ _ _ (val_eq (ha l)) (val_eq (hb l))]
  constructor
  · intro h
    by_contra hP
    have h0 : total ≠ 0 := by
      intro h0; rw [h0] at htot; simp at htot; exact hP htot.symm
    have hE : -12 ≤ E := by
      by_contra hlt
      apply h0; rw [hsum]
      refine Finset.sum_eq_zero fun l _ => ?_
      by_cases hp : (product (a l) (b l)).mag = 0
      · simp [aligned, hp]
      · have := (product_exp _ _ hp).1; have := hle l hp; omega
    have hbl1 : 1 ≤ bitLength total.natAbs := by unfold bitLength; rw [ite_eq_right (by omega)]; omega
    obtain ⟨m1, m2, e1, -⟩ := normalize_specW hw24 h0 (E := E) (by omega)
    dsimp only at h
    split_ifs at h with hinf
    · exact packValue_inf_ne _ h
    · simp only [packValue] at h
      have e2 : (normalize w E total).exp ≤ 127 := by by_contra hc; exact hinf ⟨by omega, by omega⟩
      have hf := fp32Value_packTerm m1 m2 (by omega) e2
      rw [h, fp32Value_zero] at hf
      have := Value.fin.inj hf
      rw [← this] at m1; simp [Term.zero] at m1
  · intro h
    have h0 : total = 0 := by
      rw [h] at htot; exact_mod_cast (mul_eq_zero.1 htot).resolve_right (two_zpow_ne _)
    rw [h0]; simp [normalize, Term.zero, packValue, packTerm]

/-- (b1) at `hopperE4m3K32`, in `step_zero_iffW`'s form (`Freeze.hopperStep_zero_iff`). -/
theorem step_zero_iff_hopper (a b : Fin 32 → Code) (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) :
    hopperE4m3K32.step 0 (List.ofFn a) (List.ofFn b) = 0 ↔ preSumW (-139) 14 a b = 0 := by
  rw [preSumW_hopper]; exact hopperStep_zero_iff a b ha hb

/-- (b1) on sm_120, at any floor. -/
theorem sm120_step_zero_iff (z : ℤ) (a b : Fin 32 → Code) (ha : ∀ l, (a l).isNaN = false)
    (hb : ∀ l, (b l).isNaN = false) :
    (sm120E4m3K32 z).step 0 (List.ofFn a) (List.ofFn b) = 0 ↔ preSumW z 26 a b = 0 :=
  step_zero_iffW (by norm_num) (by norm_num) z a b ha hb

/-! ## (b) Confinement: the running words of a group -/

theorem aligned_natAbs_ltW {w : ℕ} (hw : 24 ≤ w) (E : ℤ) (t : Term) (hm : t.mag < 2 ^ 25) :
    (aligned w E t).natAbs < 2 ^ (w + 1) := by
  have hM : t.mag * 2 ^ (w - 24) < 2 ^ (w + 1) := by
    calc t.mag * 2 ^ (w - 24) < 2 ^ 25 * 2 ^ (w - 24) := Nat.mul_lt_mul_of_pos_right hm (by positivity)
      _ = 2 ^ (w + 1) := by rw [← pow_add]; congr 1; omega
  unfold aligned toAdder
  rw [ite_eq_left hw, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]
  split
  · simp
  · dsimp only
    have := Nat.div_le_self (t.mag * 2 ^ (w - 24)) (2 ^ (E - t.exp).toNat)
    split <;> simp only [Int.natAbs_neg, Int.natAbs_natCast] <;> omega

/-- The incoming word of atom `j` of a group from `+0`, under `p`. -/
def inWordP (p : Pipeline) (g : List AtomIn) (j : Nat) : Nat := (g.take j).foldl (stepP p) 0

theorem inWordP_hopper : inWordP hopperE4m3K32 = inWord := rfl

/-- A running word of a group from `+0` after `j` atoms at width `w`: `+0`, or normal with exponent in
`[−12 − (w − 1)·j, 16 + 7j]`. -/
def TameW (w j : Nat) (acc : Nat) : Prop :=
  acc = 0 ∨ ∃ tc, fp32Value acc = .fin tc ∧ 2 ^ 23 ≤ tc.mag ∧ -12 - ((w : ℤ) - 1) * j ≤ tc.exp ∧
    tc.exp ≤ 16 + 7 * (j : ℤ)

theorem tame_stepW {w : ℕ} (hw24 : 24 ≤ w) {z : ℤ} (hz : z ≤ 16) {j : Nat} (hj : ((w : ℤ) - 1) * (j + 1) ≤ 114)
    {acc : Nat} (hacc : TameW w j acc) (x : AtomIn) (hx : FiniteAtom x) :
    TameW w (j + 1) (stepP ⟨[32], w, z⟩ acc x) := by
  obtain ⟨ha, hb⟩ := hx
  have hw1 : (23 : ℤ) ≤ (w : ℤ) - 1 := by omega
  have hj0 : (0 : ℤ) ≤ ((w : ℤ) - 1) * j := by positivity
  have hj3 : (j : ℤ) ≤ 3 := by nlinarith
  have hsplit : ((w : ℤ) - 1) * ((j + 1 : ℕ) : ℤ) = ((w : ℤ) - 1) * j + ((w : ℤ) - 1) := by push_cast; ring
  obtain ⟨tc, hfw, htc⟩ : ∃ tc, fp32Value acc = .fin tc ∧
      (tc.mag = 0 ∨ (2 ^ 23 ≤ tc.mag ∧ -12 - ((w : ℤ) - 1) * j ≤ tc.exp ∧ tc.exp ≤ 16 + 7 * (j : ℤ))) := by
    rcases hacc with rfl | ⟨tc, h1, h2, h3, h4⟩
    · exact ⟨Term.zero, fp32Value_zero, Or.inl rfl⟩
    · exact ⟨tc, h1, Or.inr ⟨h2, h3, h4⟩⟩
  unfold stepP
  rw [step_fin hfw x.1 x.2 ha hb]
  unfold groupSum
  set E := maxExp z (tc :: List.ofFn fun l => product (x.1 l) (x.2 l)) with hEdef
  set total := alignedSum w E (tc :: List.ofFn fun l => product (x.1 l) (x.2 l)) with htdef
  change TameW w (j + 1) (packValue (let t := normalize w E total
    if t.mag ≠ 0 ∧ 127 < t.exp then Value.inf t.neg else Value.fin t))
  clear_value total E
  have hsum : total = aligned w E tc + ∑ l, aligned w E (product (x.1 l) (x.2 l)) := by
    rw [htdef, alignedSum, List.map_cons, List.sum_cons, List.map_ofFn, List.sum_ofFn]
    simp only [Function.comp_apply]
  have hEle : E ≤ 16 + 7 * (j : ℤ) := by
    rw [hEdef]
    refine maxExp_le _ _ _ (by omega) fun t ht htm => ?_
    rcases List.mem_cons.1 ht with rfl | ht
    · rcases htc with h | h
      · exact absurd h htm
      · exact h.2.2
    · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
      have := (product_exp _ _ htm).2
      omega
  have hmc : tc.mag < 2 ^ 24 := (fp32Value_fin hfw).1
  have hT64 : total.natAbs < 64 * 2 ^ (w + 1) := by
    have hc : (aligned w E tc).natAbs < 2 ^ (w + 1) := aligned_natAbs_ltW hw24 E tc (by omega)
    have h1 : |∑ l : Fin 32, aligned w E (product (x.1 l) (x.2 l))| ≤
        ∑ l : Fin 32, |aligned w E (product (x.1 l) (x.2 l))| := Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ l : Fin 32, |aligned w E (product (x.1 l) (x.2 l))| ≤ ∑ _l : Fin 32, ((2 ^ (w + 1) : ℕ) : ℤ) := by
      refine Finset.sum_le_sum fun l _ => ?_
      have := aligned_natAbs_ltW hw24 E (product (x.1 l) (x.2 l)) (product_mag_lt _ _)
      rw [Int.abs_eq_natAbs]; omega
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h2
    rw [Int.abs_eq_natAbs] at h1
    rw [hsum]
    generalize 2 ^ (w + 1) = B at hc h2 ⊢
    omega
  have hT : total.natAbs < 2 ^ (w + 7) :=
    calc total.natAbs < 64 * 2 ^ (w + 1) := hT64
      _ = 2 ^ (w + 7) := by rw [show w + 7 = 6 + (w + 1) by omega, pow_add 2 6 (w + 1)]; norm_num
  by_cases h0 : total = 0
  · left; rw [h0]; simp [normalize, Term.zero, packValue, packTerm]
  have hEge : -12 - ((w : ℤ) - 1) * j ≤ E := by
    by_contra hlt
    apply h0
    rw [hsum]
    have hc : aligned w E tc = 0 := by
      rcases htc with h | h
      · simp [aligned, h]
      · exact absurd (le_trans h.2.1 (le_maxExp _ _ tc List.mem_cons_self (by omega) |>.trans_eq hEdef.symm)) hlt
    rw [hc, zero_add]
    refine Finset.sum_eq_zero fun l _ => ?_
    by_cases hp : (product (x.1 l) (x.2 l)).mag = 0
    · simp [aligned, hp]
    · have h1 := (product_exp _ _ hp).1
      have h2 : (product (x.1 l) (x.2 l)).exp ≤ E := by
        rw [hEdef]; exact le_maxExp _ _ _ (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨l, rfl⟩)) hp
      omega
  have hn : total.natAbs ≠ 0 := by omega
  have hbl1 : 1 ≤ bitLength total.natAbs := by unfold bitLength; rw [ite_eq_right hn]; omega
  have hbl2 : bitLength total.natAbs ≤ w + 7 := by
    unfold bitLength; rw [ite_eq_right hn]
    have := (Nat.log2_lt hn).2 hT
    omega
  obtain ⟨m1, m2, e1, -⟩ := normalize_specW hw24 h0 (E := E) (by push_cast at hsplit ⊢; nlinarith)
  have e2 : (normalize w E total).exp ≤ 127 := by rw [e1]; omega
  dsimp only
  rw [ite_eq_right (fun h => absurd h.2 (not_lt.2 e2))]
  right
  refine ⟨_, fp32Value_packTerm m1 m2 (by rw [e1]; nlinarith) e2, m1, ?_, ?_⟩
  · rw [e1, hsplit]; omega
  · rw [e1]; push_cast; omega

theorem tame_inWordW {w : ℕ} (hw24 : 24 ≤ w) {z : ℤ} (hz : z ≤ 16) (g : List AtomIn)
    (hg : ((w : ℤ) - 1) * g.length ≤ 114) (hfin : ∀ x ∈ g, FiniteAtom x) :
    ∀ j ≤ g.length, TameW w j (inWordP ⟨[32], w, z⟩ g j) := by
  intro j hj
  induction j with
  | zero => left; rfl
  | succ j ih =>
    have hjl : j < g.length := by omega
    have : inWordP ⟨[32], w, z⟩ g (j + 1) = stepP ⟨[32], w, z⟩ (inWordP ⟨[32], w, z⟩ g j) g[j] := by
      unfold inWordP
      rw [List.take_add_one, List.getElem?_eq_getElem hjl, Option.toList_some, List.foldl_append]
      rfl
    rw [this]
    have hj' : ((w : ℤ) - 1) * ((j : ℤ) + 1) ≤ 114 := by
      have : ((j : ℤ) + 1) ≤ g.length := by exact_mod_cast hjl
      nlinarith
    exact tame_stepW hw24 hz hj' (ih (by omega)) _ (hfin _ (List.getElem_mem hjl))

theorem getElem_idFlagsP (p : Pipeline) (g : List AtomIn) (j : Nat) (h : j < (idFlagsP p g).length)
    (hj : j < g.length) : (idFlagsP p g)[j] = decide (stepP p (inWordP p g j) g[j] = inWordP p g j) :=
  getElem_fixFlags (stepP p) 0 g j hj h

/-- **(b) Confinement** at width `w ≥ 24`, in a group from `+0` of `n` atoms with `(w − 1)·n ≤ 114` (on sm_120,
`n ≤ 4`), on finite codes: an identity at atom `j` is either on the word `+0`, where it is exactly `preSumW = 0`, or on
a normal word `c` that the atom's products do not move, (a)'s domination. -/
theorem identity_confinedW {w : ℕ} (hw24 : 24 ≤ w) {z : ℤ} (hz : z ≤ 16) (g : List AtomIn)
    (hg : ((w : ℤ) - 1) * g.length ≤ 114) (hfin : ∀ x ∈ g, FiniteAtom x) (j : Nat) (hj : j < g.length)
    (hid : (idFlagsP ⟨[32], w, z⟩ g)[j]'(by rw [idFlagsP, length_fixFlags]; exact hj) = true) :
    (inWordP ⟨[32], w, z⟩ g j = 0 ∧ preSumW z w g[j].1 g[j].2 = 0) ∨
      ∃ tc : Term, fp32Value (inWordP ⟨[32], w, z⟩ g j) = .fin tc ∧ 2 ^ 23 ≤ tc.mag ∧
        truncTo (2 ^ (stepExpW z tc g[j].1 g[j].2 - w + 1)) tc.value = tc.value ∧
        truncTo (2 ^ (tc.exp - 23)) (tc.value +
          ∑ l, truncTo (2 ^ (stepExpW z tc g[j].1 g[j].2 - w + 1)) (val (g[j].1 l) * val (g[j].2 l))) =
          tc.value := by
  rw [getElem_idFlagsP _ g j _ hj, decide_eq_true_iff] at hid
  obtain ⟨ha, hb⟩ := hfin _ (List.getElem_mem hj)
  have hj' : ((w : ℤ) - 1) * ((j : ℤ) + 1) ≤ 114 := by
    have : ((j : ℤ) + 1) ≤ g.length := by exact_mod_cast hj
    have : (0 : ℤ) ≤ (w : ℤ) - 1 := by omega
    nlinarith
  rcases tame_inWordW hw24 hz g hg hfin j hj.le with h0 | ⟨tc, h1, h2, h3, -⟩
  · left
    refine ⟨h0, (step_zero_iffW hw24 (by nlinarith) z _ _ ha hb).1 ?_⟩
    rw [h0] at hid; exact hid
  · right
    exact ⟨tc, h1, h2, identity_needs_dominationW hw24 h1 h2 (by nlinarith) _ _ ha hb hid⟩

/-- (b) on sm_120: groups of at most four atoms, at any floor at most 16. -/
theorem sm120_identity_confined {z : ℤ} (hz : z ≤ 16) (g : List AtomIn) (hg : g.length ≤ 4)
    (hfin : ∀ x ∈ g, FiniteAtom x) (j : Nat) (hj : j < g.length)
    (hid : (idFlagsP (sm120E4m3K32 z) g)[j]'(by rw [idFlagsP, length_fixFlags]; exact hj) = true) :
    (inWordP (sm120E4m3K32 z) g j = 0 ∧ preSumW z 26 g[j].1 g[j].2 = 0) ∨
      ∃ tc : Term, fp32Value (inWordP (sm120E4m3K32 z) g j) = .fin tc ∧ 2 ^ 23 ≤ tc.mag ∧
        truncTo (2 ^ (stepExpW z tc g[j].1 g[j].2 - (26 : ℕ) + 1)) tc.value = tc.value ∧
        truncTo (2 ^ (tc.exp - 23)) (tc.value +
          ∑ l, truncTo (2 ^ (stepExpW z tc g[j].1 g[j].2 - (26 : ℕ) + 1)) (val (g[j].1 l) * val (g[j].2 l))) =
          tc.value :=
  identity_confinedW (w := 26) (by norm_num) hz g (by push_cast; omega) hfin j hj hid

end Pouw.PearlC.FreezeP
