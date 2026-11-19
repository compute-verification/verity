module

public import Proofs.Flock.SamplingGeneric.LevelsTwoKS
public import Proofs.Core.LevelsCredited
public import Proofs.Core.LevelsLate

/-!
# The compiler at levels is knowledge sound: three levels or more

`CompilerKSLev` on the statements with a middle level (`0 < CompiledLev.mid`), at `Elev`. The outer wait unions over
the sets of proof blocks the levels miss with probability at most `δ` (`cands`); the first middle wait over the sets
the levels below the first miss no more often than the candidate in any first-level block (`aboveDeep`); every wait
below it over one set, the kept outer run's incorrect set, which every run below it shows alike (`Sessions.interior`).

* `BadBelow`, `pr_badBelow_le`: **a run bad below**, wait by wait, and its probability (`waitErr`): each wait's
  rounds hold the next one's (`Chain`, from `Sessions.nested`), so the acceptance below a kept run averages through
  the wait (`Pr_acc_forkOn`).
* `escB_le_escLev`: **the escape, level by level**: fresh coins on a middle level's rounds average the escape of the
  level below at most to the level's (`Samples`' middle clause, `missProb_sampleBelow_cons`).
-/

@[expose] public section

open scoped ENNReal

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeLev RLev slot topOf incorrectLev levM forkOn forkOn_forkOn'
  Ex_forkOn accList AllBad wait_bad_le slack slack_pos Pr_acc_forkOn OwnColl agree_of_passes val_assemble ιOf ιOf_eq
  acc_of_accList accList_length_eq assemble KSLinOn Bext replayOf TopIs)

theorem forkOn_forkOn_of_subset {k : ℕ} {Coin : Type} {F f : Finset (Fin k)} (h : F ⊆ f) (r d e : Fin k → Coin) :
    forkOn f r (forkOn F d e) = forkOn F (forkOn f r d) e := by
  funext t
  by_cases hF : t ∈ F
  · simp [forkOn, hF, h hF]
  · by_cases hf : t ∈ f <;> simp [forkOn, hF, hf]

namespace CompiledLev

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The waits below a kept run -/

/-- The rounds the wait below a run reruns on, with the middle levels' rounds `fs` left: the next middle level's, or
the last level's. -/
def nextF (L : CompiledLev Coin A) (x : Msg) : List (Finset (Fin (L.V.k x))) → Finset (Fin (L.V.k x))
  | [] => L.secondStage x
  | f :: _ => f

/-- Each wait's rounds hold the next one's. -/
def Chain (L : CompiledLev Coin A) (x : Msg) : List (Finset (Fin (L.V.k x))) → Prop
  | [] => True
  | f :: fs => L.nextF x fs ⊆ f ∧ Chain L x fs

/-- **A run bad below for the sets `Xs`**, with the middle levels' rounds `fs` left: past the middle levels,
`CompiledTwo.BadOuterTwo`; at a middle level, its wait comes up short, or for some `X ∈ Xs` every kept run is bad below
for `X` alone. -/
def BadBelow (L : CompiledLev Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) :
    (fs : List (Finset (Fin (L.V.k x)))) → Finset (Finset ℕ) → (Fin (L.V.k x) → Coin) → L.WaitC x fs.length → Prop
  | [], Xs, r, c => L.twoView.BadOuterTwo x run' found Xs (forkOn (L.secondStage x) r) c
  | f :: fs, Xs, r, c =>
    (accList (fun d => (run' (forkOn f r d)).out = .acc) (L.kOut x) ((List.finRange (L.capOut x)).map c.1)).length <
        L.kOut x ∨
      ∃ X ∈ Xs, AllBad (fun d => (run' (forkOn f r d)).out = .acc) (L.kOut x)
        (fun d c' => BadBelow L x run' found fs {X} (forkOn f r d) c') ((List.finRange (L.capOut x)).map c.1) c.2

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- A run is bad below as every run fresh on the next wait's rounds is. -/
theorem badBelow_fork (L : CompiledLev Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop)
    (fs : List (Finset (Fin (L.V.k x)))) (Xs : Finset (Finset ℕ)) (r e : Fin (L.V.k x) → Coin)
    (c : L.WaitC x fs.length) :
    L.BadBelow x run' found fs Xs (forkOn (L.nextF x fs) r e) c ↔ L.BadBelow x run' found fs Xs r c := by
  cases fs with
  | nil => exact Iff.of_eq (congrArg (fun g => L.twoView.BadOuterTwo x run' found Xs g c)
      (forkOn_forkOn' (L.secondStage x) r e))
  | cons f fs => simp only [BadBelow, nextF, forkOn_forkOn']

/-- **The escape below a run** of a set `X`: past the middle levels, the probability that the last level's sample
misses `X`; at a middle level, its average over the level's fresh coins. -/
noncomputable def escB (L : CompiledLev Coin A) (x : Msg) :
    List (Finset (Fin (L.V.k x))) → Finset ℕ → (Fin (L.V.k x) → Coin) → ℝ
  | [], X, r => Pr (fun e : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.secondStage x) r e), u ∉ X)
  | f :: fs, X, r => Ex (fun d => escB L x fs X (forkOn f r d))

/-- **The sessions' error below a run**: past the middle levels, `κr`; at a middle level, its average over the
level's fresh coins. -/
noncomputable def κB (L : CompiledLev Coin A) (x : Msg) (κr : (Fin (L.V.k x) → Coin) → ℝ) :
    List (Finset (Fin (L.V.k x))) → (Fin (L.V.k x) → Coin) → ℝ
  | [], r => κr r
  | f :: fs, r => Ex (fun d => κB L x κr fs (forkOn f r d))

omit [Nonempty Coin] [Inhabited Coin] in
theorem escB_nonneg (L : CompiledLev Coin A) (x : Msg) :
    ∀ (fs : List (Finset (Fin (L.V.k x)))) (X : Finset ℕ) (r : Fin (L.V.k x) → Coin), 0 ≤ L.escB x fs X r
  | [], _, _ => Pr_nonneg _
  | f :: fs, X, r => Ex_nonneg fun d => escB_nonneg L x fs X (forkOn f r d)

omit [Nonempty Coin] [Inhabited Coin] in
theorem κB_nonneg (L : CompiledLev Coin A) (x : Msg) {κr : (Fin (L.V.k x) → Coin) → ℝ} (hκr0 : ∀ r, 0 ≤ κr r) :
    ∀ (fs : List (Finset (Fin (L.V.k x)))) (r : Fin (L.V.k x) → Coin), 0 ≤ L.κB x κr fs r
  | [], r => hκr0 r
  | f :: fs, r => Ex_nonneg fun d => κB_nonneg L x hκr0 fs (forkOn f r d)

omit [Inhabited Coin] in
/-- The sessions' error below a run averages, over the run's coins, to `κr`'s: fresh coins are uniform. -/
theorem Ex_κB (L : CompiledLev Coin A) (x : Msg) (κr : (Fin (L.V.k x) → Coin) → ℝ) :
    ∀ fs : List (Finset (Fin (L.V.k x))), Ex (L.κB x κr fs) = Ex κr
  | [] => rfl
  | f :: fs => by
    calc Ex (fun r => Ex fun d => L.κB x κr fs (forkOn f r d))
        = Ex (fun p : (Fin (L.V.k x) → Coin) × (Fin (L.V.k x) → Coin) => L.κB x κr fs (forkOn f p.1 p.2)) :=
          (Ex_prod (fun p : (Fin (L.V.k x) → Coin) × (Fin (L.V.k x) → Coin) =>
            L.κB x κr fs (forkOn f p.1 p.2))).symm
      _ = Ex (L.κB x κr fs) := Ex_forkOn f _
      _ = Ex κr := Ex_κB L x κr fs

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem waitErr_eq (L : CompiledLev Coin A) (x : Msg) :
    ∀ (m : ℕ) (a : ℝ), L.waitErr x m a = L.waitErr x m 0 + slack ^ (m + 1) * a
  | 0, a => by simp only [waitErr]; ring
  | m + 1, a => by
    simp only [waitErr]
    rw [waitErr_eq L x m a]
    ring

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem waitErr_mono (L : CompiledLev Coin A) (x : Msg) (m : ℕ) {a b : ℝ} (h : a ≤ b) :
    L.waitErr x m a ≤ L.waitErr x m b := by
  rw [L.waitErr_eq x m a, L.waitErr_eq x m b]
  have := slack_pos
  gcongr

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem waitErr_nonneg (L : CompiledLev Coin A) (x : Msg) : ∀ (m : ℕ) {a : ℝ}, 0 ≤ a → 0 ≤ L.waitErr x m a
  | 0, a, ha => by
    have := slack_pos
    have := Real.exp_pos (-(L.kIn x : ℝ) / 8)
    unfold waitErr
    positivity
  | m + 1, a, ha => by
    have := slack_pos
    have := Real.exp_pos (-(L.kOut x : ℝ) / 8)
    have := waitErr_nonneg L x m ha
    unfold waitErr
    positivity

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem Ex_waitErr {α : Type} [Fintype α] [Nonempty α] (L : CompiledLev Coin A) (x : Msg) (m : ℕ) (g : α → ℝ) :
    Ex (fun a => L.waitErr x m (g a)) = L.waitErr x m (Ex g) := by
  rw [show (fun a => L.waitErr x m (g a)) = fun a => L.waitErr x m 0 + slack ^ (m + 1) * g a from
    funext fun a => L.waitErr_eq x m (g a), Ex_add, Ex_const, Ex_mul_left, ← L.waitErr_eq x m (Ex g)]

set_option synthInstance.maxSize 1024 in
omit [Inhabited Coin] in
/-- **A run bad below**, at sets `Xs` of proof blocks below `n_PU`, at most `2^m` of them (`m ≤ 2n_out + 1`, and
`m = 0` past the middle levels), each escaping below `r` with at most `y`: the run's acceptance below times the
probability that it is bad below is at most `waitErr` at `y` plus the sessions' error below. Past the middle levels,
`CompiledTwo.inner_badTwo_le`; at a middle level, each kept run's bound averages through the wait (`Pr_acc_forkOn`,
the next wait's rounds inside this one's), and the wait unions over `Xs` (`wait_bad_le`). -/
theorem pr_badBelow_le (L : CompiledLev Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop)
    (hslots : ∀ r', (L.checked x r').length ≤ L.nPU x) (κr : (Fin (L.V.k x) → Coin) → ℝ) (hκr0 : ∀ r, 0 ≤ κr r)
    (hsess : ∀ r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run' (forkOn (L.secondStage x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.secondStage x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.secondStage x) r p.1))[i.val]? = some u.val ∧
              ¬ found u (forkOn (L.secondStage x) r p.1) (p.2 i u)) ≤ κr r) :
    ∀ (fs : List (Finset (Fin (L.V.k x)))), L.Chain x fs → ∀ (Xs : Finset (Finset ℕ)) (m : ℕ), Xs.card ≤ 2 ^ m →
      m ≤ 2 * L.nOut x + 1 → (fs = [] → m = 0) → (∀ X ∈ Xs, ∀ v ∈ X, v < L.nPU x) →
      ∀ (r : Fin (L.V.k x) → Coin) (y : ℝ), 0 ≤ y → (∀ X ∈ Xs, L.escB x fs X r ≤ y) →
      Pr (fun d : Fin (L.V.k x) → Coin => (run' (forkOn (L.nextF x fs) r d)).out = .acc) *
          Pr (fun c : L.WaitC x fs.length => L.BadBelow x run' found fs Xs r c) ≤
        L.waitErr x fs.length (y + L.κB x κr fs r)
  | [], _, Xs, m, hcard, _, hm0, hXs, r, y, hy0, hy => by
    have hm : m = 0 := hm0 rfl
    subst hm
    exact L.twoView.inner_badTwo_le x run' found hXs hcard (Nat.zero_le _) hslots r hy0 (hκr0 r) hy (hsess r)
  | f :: fs, hch, Xs, m, hcard, hm, _, hXs, r, y, hy0, hy => by
    have hc := slack_pos
    have hκ0 : 0 ≤ L.κB x κr (f :: fs) r := L.κB_nonneg x hκr0 _ r
    have hy' : ∀ X ∈ Xs, Pr (fun p : (Fin (L.V.k x) → Coin) × L.WaitC x fs.length =>
        (run' (forkOn f r p.1)).out = .acc ∧ L.BadBelow x run' found fs {X} (forkOn f r p.1) p.2) ≤
        L.waitErr x fs.length (y + L.κB x κr (f :: fs) r) := by
      intro X hX
      rw [Pr_acc_forkOn (L.nextF x fs) (fun d => (run' (forkOn f r d)).out = .acc)
        (fun d c => L.BadBelow x run' found fs {X} (forkOn f r d) c)
        (fun d e c => by
          show L.BadBelow x run' found fs {X} (forkOn f r (forkOn (L.nextF x fs) d e)) c ↔ _
          rw [forkOn_forkOn_of_subset hch.1, badBelow_fork])]
      calc Ex (fun d => Pr (fun e => (run' (forkOn f r (forkOn (L.nextF x fs) d e))).out = .acc) *
            Pr (fun c => L.BadBelow x run' found fs {X} (forkOn f r d) c))
          = Ex (fun d => Pr (fun e => (run' (forkOn (L.nextF x fs) (forkOn f r d) e)).out = .acc) *
            Pr (fun c => L.BadBelow x run' found fs {X} (forkOn f r d) c)) := by
            simp only [forkOn_forkOn_of_subset hch.1]
        _ ≤ Ex (fun d => L.waitErr x fs.length (L.escB x fs X (forkOn f r d) + L.κB x κr fs (forkOn f r d))) :=
            Ex_mono fun d => pr_badBelow_le L x run' found hslots κr hκr0 hsess fs hch.2 {X} 0 (by simp)
              (Nat.zero_le _) (fun _ => rfl)
              (fun X' hX' => by rw [mem_singleton.1 hX']; exact hXs X hX) (forkOn f r d) _
              (L.escB_nonneg x fs X _) (fun X' hX' => by rw [mem_singleton.1 hX'])
        _ = L.waitErr x fs.length (L.escB x (f :: fs) X r + L.κB x κr (f :: fs) r) := by
            rw [L.Ex_waitErr x fs.length, Ex_add]
            rfl
        _ ≤ L.waitErr x fs.length (y + L.κB x κr (f :: fs) r) :=
            L.waitErr_mono x _ (by linarith [hy X hX])
    have hw0 := L.waitErr_nonneg x fs.length (add_nonneg hy0 hκ0)
    have hw := wait_bad_le (fun d => (run' (forkOn f r d)).out = .acc) (L.kOut x) (L.capOut x) (L.capOut_pos x)
      (L.kOut_pos x) Xs (fun X d c => L.BadBelow x run' found fs {X} (forkOn f r d) c) hw0 hy'
    have hcs : (Xs.card : ℝ) ^ (1 / (L.kOut x : ℝ)) ≤ slack := L.rpow_kOut_le_slack x hcard hm
    refine hw.trans ?_
    show _ ≤ (2 * L.kOut x / L.capOut x + Real.exp (-(L.kOut x : ℝ) / 8)) +
      slack * L.waitErr x fs.length (y + L.κB x κr (f :: fs) r)
    exact add_le_add le_rfl (mul_le_mul_of_nonneg_right hcs hw0)

end CompiledLev

/-! ## The escape, level by level -/

/-- **The levels after level `j` miss `X`** inside its block `Z`: the sample of the levels after `j`, drawn inside
`Z`, misses the slots of `X`. -/
noncomputable def escAt (Pt : Partition) (Lv : Verity.Levels (Fin Pt.blocks.length)) (j : ℕ)
    (Z : Finset (Fin Pt.blocks.length)) (X : Finset ℕ) : ℝ≥0∞ :=
  (Verity.Level.sampleBelow (Lv.levels.drop (j + 1)) Z).missProb (X.image (slot Pt))

theorem escAt_le_one (Pt : Partition) (Lv : Verity.Levels (Fin Pt.blocks.length)) (j : ℕ)
    (Z : Finset (Fin Pt.blocks.length)) (X : Finset ℕ) : escAt Pt Lv j Z X ≤ 1 :=
  Verity.SamplingDesign.missProb_le_one _ _

theorem escAt_cons (Pt : Partition) (Lv : Verity.Levels (Fin Pt.blocks.length)) {j : ℕ}
    {l' : Verity.Level (Fin Pt.blocks.length)} (hl : Lv.levels[j + 1]? = some l') (Z : Finset (Fin Pt.blocks.length))
    (X : Finset ℕ) :
    escAt Pt Lv j Z X = Verity.Game.avg fun ω => ∏ Y ∈ (l'.design Z).draw ω, escAt Pt Lv (j + 1) Y X := by
  obtain ⟨hlt, hget⟩ := List.getElem?_eq_some_iff.1 hl
  unfold escAt
  rw [List.drop_eq_getElem_cons hlt, hget, Verity.Level.missProb_sampleBelow_cons]

theorem escAt_last (Pt : Partition) (Lv : Verity.Levels (Fin Pt.blocks.length)) {j : ℕ}
    (hj : j + 2 = Lv.levels.length) (Z : Finset (Fin Pt.blocks.length)) (X : Finset ℕ) :
    escAt Pt Lv j Z X = (Lv.levels[j + 1].design Z).missProb (X.image (slot Pt)) := by
  unfold escAt
  rw [List.drop_eq_getElem_cons (by omega : j + 1 < Lv.levels.length),
    List.drop_eq_nil_of_le (by omega : Lv.levels.length ≤ j + 1 + 1)]
  rfl

theorem getLastD_eq_getD {α : Type} {l : List α} {n : ℕ} (h : l.length = n + 1) (d : α) :
    l.getLastD d = l.getD n d := by
  rw [List.getLastD_eq_getLast?, List.getLast?_eq_getElem?, List.getD_eq_getElem?_getD, h]
  rfl

theorem getD_zero_eq_headD {α : Type} (l : List α) (d : α) : l.getD 0 d = l.headD d := by
  cases l <;> rfl

open Classical in
/-- The sets of proof blocks the levels after the first miss no more often than `Y`, in every first-level block. -/
noncomputable def aboveDeep (Pt : Partition) (Lv : Verity.Levels (Fin Pt.blocks.length))
    (l₀ : Verity.Level (Fin Pt.blocks.length)) (Y : Finset ℕ) : Finset (Finset ℕ) :=
  (range Pt.blocks.length).powerset.filter fun X => ∀ Z ∈ l₀.blocks.parts, escAt Pt Lv 0 Z X ≤ escAt Pt Lv 0 Z Y

/-- A set below `Y` in every first-level block escapes the levels after the first, over the sampled first-level
blocks, no more often than `Y` over those that are first-level blocks. -/
theorem prod_escAt_le {Pt : Partition} {Lv : Verity.Levels (Fin Pt.blocks.length)}
    {l₀ : Verity.Level (Fin Pt.blocks.length)} (U : Finset (Finset (Fin Pt.blocks.length))) {X Y : Finset ℕ}
    (hX : X ∈ aboveDeep Pt Lv l₀ Y) :
    ∏ Z ∈ U, (escAt Pt Lv 0 Z X).toReal ≤ ∏ Z ∈ U.filter (· ∈ l₀.blocks.parts), (escAt Pt Lv 0 Z Y).toReal := by
  classical
  rw [← prod_filter_mul_prod_filter_not U (· ∈ l₀.blocks.parts)]
  have h1 : ∏ Z ∈ U.filter (fun Z => ¬ Z ∈ l₀.blocks.parts), (escAt Pt Lv 0 Z X).toReal ≤ 1 :=
    prod_le_one₀ (fun _ _ => ENNReal.toReal_nonneg) fun Z _ =>
      ENNReal.toReal_le_of_le_ofReal zero_le_one (by rw [ENNReal.ofReal_one]; exact escAt_le_one _ _ _ _ _)
  refine (mul_le_mul_of_nonneg_left h1 (prod_nonneg fun _ _ => ENNReal.toReal_nonneg)).trans ?_
  rw [mul_one]
  refine Finset.prod_le_prod₀ (fun _ _ => ENNReal.toReal_nonneg) fun Z hZ => ?_
  exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (escAt_le_one _ _ _ _ _))
    ((mem_filter.1 hX).2 Z (mem_filter.1 hZ).2)

/-- **The first level's average at three levels or more**: over coins whose first-level sample misses each set of
first-level blocks no more often than the first level's design, the escape of `Y` over the sampled first-level blocks
averages at most the levels' miss of `Y` (`Verity.Levels.missProb_cons`). -/
theorem Ex_escAt_le {R : Type} [Fintype R] [Nonempty R] {Pt : Partition} (Lv : Verity.Levels (Fin Pt.blocks.length))
    {l₀ : Verity.Level (Fin Pt.blocks.length)} {rest : List (Verity.Level (Fin Pt.blocks.length))}
    (hL : Lv.levels = l₀ :: rest) (hrest : rest ≠ []) (S : R → Finset (Finset (Fin Pt.blocks.length)))
    (hS : ∀ T, Pr (fun r => Disjoint (S r) T) ≤ ((l₀.design Finset.univ).missProb T).toReal) (Y : Finset ℕ) :
    Ex (fun r => ∏ Z ∈ (S r).filter (· ∈ l₀.blocks.parts), (escAt Pt Lv 0 Z Y).toReal) ≤
      (Lv.design.missProb (Y.image (slot Pt))).toReal := by
  have hi := Lv.inside
  rw [hL] at hi
  simp only [Verity.Level.InsideBelow] at hi
  refine (CompiledLev.Ex_prod_parts_le_avg l₀ (fun ω b hb => (hi.1 Finset.univ (Finset.mem_singleton_self _) ω b hb).1) S hS
    (fun Z => escAt Pt Lv 0 Z Y) fun Z => escAt_le_one _ _ _ _ _).trans_eq ?_
  rw [Lv.missProb_cons hL hrest]
  unfold escAt
  rw [hL]
  rfl

namespace CompiledLev

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem forks_length (L : CompiledLev Coin A) (x : Msg) : (L.forks x).length = L.mid x := by
  simp [forks]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem forks_drop (L : CompiledLev Coin A) (x : Msg) {j : ℕ} (hj : j < L.mid x) :
    (L.forks x).drop j = L.stages x j :: (L.forks x).drop (j + 1) := by
  rw [List.drop_eq_getElem_cons (by rw [forks_length]; exact hj)]
  simp [forks]

omit [Inhabited Coin] in
/-- **The escape below a run, level by level**: below the middle levels from `j` on, the escape of `X` at `r` is at
most the product, over the blocks level `j` sampled at `r`, of the levels after `j`'s miss of `X` inside each
(`Samples`' last clause, then its middle clause at each level up, `escAt_cons`). -/
theorem escB_drop_le (L : CompiledLev Coin A) (x : Msg) {Pt : Partition} {Lv : Verity.Levels (Fin Pt.blocks.length)}
    (hsam : L.Samples x Pt Lv) (hmid : L.mid x + 2 = Lv.levels.length) (X : Finset ℕ) (k : ℕ) :
    ∀ j, j + k = L.mid x → ∀ r, L.escB x ((L.forks x).drop j) X r ≤
      (∏ Z ∈ (L.upper x Pt r).getD j ∅, escAt Pt Lv j Z X).toReal := by
  induction k with
  | zero =>
    intro j hjk r
    have hj : j = L.mid x := by omega
    subst hj
    rw [List.drop_eq_nil_of_le (by rw [forks_length])]
    have hlast : Lv.levels.getLast? = some Lv.levels[L.mid x + 1] := by
      rw [List.getLast?_eq_getElem?, ← List.getElem?_eq_getElem]
      congr 1
      omega
    have hlen : (L.upper x Pt r).length = L.mid x + 1 := by have := hsam.2.2.2.2 r; omega
    have h := hsam.2.2.2.1 _ hlast (by omega) r X
    rw [getLastD_eq_getD hlen] at h
    calc L.escB x [] X r
        = Pr (fun e : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.secondStage x) r e), u ∉ X) := rfl
      _ ≤ _ := h
      _ = (∏ Z ∈ (L.upper x Pt r).getD (L.mid x) ∅, escAt Pt Lv (L.mid x) Z X).toReal := by
          rw [ENNReal.toReal_prod]
          exact Finset.prod_congr rfl fun Z _ => by rw [escAt_last Pt Lv hmid]
  | succ k ih =>
    intro j hjk r
    have hj : j < L.mid x := by omega
    rw [L.forks_drop x hj]
    obtain ⟨l', hl⟩ : ∃ l', Lv.levels[j + 1]? = some l' :=
      ⟨_, List.getElem?_eq_getElem (by omega : j + 1 < Lv.levels.length)⟩
    have h3 := (hsam.2.2.1 j l' hl (by omega) r (fun Y => escAt Pt Lv (j + 1) Y X)
      fun Y => escAt_le_one Pt Lv (j + 1) Y X).1
    calc L.escB x (L.stages x j :: (L.forks x).drop (j + 1)) X r
        = Ex (fun d => L.escB x ((L.forks x).drop (j + 1)) X (forkOn (L.stages x j) r d)) := rfl
      _ ≤ Ex (fun d => (∏ Y ∈ (L.upper x Pt (forkOn (L.stages x j) r d)).getD (j + 1) ∅,
            escAt Pt Lv (j + 1) Y X).toReal) :=
          Ex_mono fun d => ih (j + 1) (by omega) _
      _ ≤ _ := h3
      _ = (∏ Z ∈ (L.upper x Pt r).getD j ∅, escAt Pt Lv j Z X).toReal := by
          congr 1
          exact Finset.prod_congr rfl fun Z _ => (escAt_cons Pt Lv hl Z X).symm

omit [Inhabited Coin] in
/-- **The escape below a run at a set below `Y`** is at most `Y`'s escape over the first-level blocks the run sampled. -/
theorem escB_le_of_above (L : CompiledLev Coin A) (x : Msg) {Pt : Partition}
    {Lv : Verity.Levels (Fin Pt.blocks.length)} (hsam : L.Samples x Pt Lv) (hmid : L.mid x + 2 = Lv.levels.length)
    {l₀ : Verity.Level (Fin Pt.blocks.length)} {X Y : Finset ℕ} (hX : X ∈ aboveDeep Pt Lv l₀ Y)
    (r : Fin (L.V.k x) → Coin) :
    L.escB x (L.forks x) X r ≤
      ∏ Z ∈ ((L.upper x Pt r).headD ∅).filter (· ∈ l₀.blocks.parts), (escAt Pt Lv 0 Z Y).toReal := by
  have h := L.escB_drop_le x hsam hmid X (L.mid x) 0 (by omega) r
  rw [List.drop_zero, getD_zero_eq_headD, ENNReal.toReal_prod] at h
  exact h.trans (prod_escAt_le _ hX)

/-! ## The failure event -/

/-- **The extraction's failure event at three levels or more**: the statement is in scope with a middle level, and the
outer wait comes up short or, for some candidate `Y`, every kept outer run is bad below for the sets below `Y`
(`BadBelow` along `forks`). -/
def FailsDeep (L : CompiledLev Coin A) (δ : ℝ) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) (c : L.Elev.C x) : Prop :=
  ∃ s Pt Lv l₀ rest, L.parse x = some s ∧ L.partition s = some Pt ∧ L.levels s Pt = some Lv ∧
    InScopeLev Pt Lv (L.top s) s ∧ Lv.levels = l₀ :: rest ∧ 0 < L.mid x ∧
    ((accList (fun r => (run' r).out = .acc) (L.kOut x) ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x ∨
      ∃ Y ∈ cands Pt Lv δ, AllBad (fun r => (run' r).out = .acc) (L.kOut x)
        (fun r c' => L.BadBelow x run' found (L.forks x) (aboveDeep Pt Lv l₀ Y) r c')
        ((List.finRange (L.capOut x)).map c.1) c.2)

/-- The middle levels' rounds hold each other's and the last level's (`Sessions.nested`). -/
theorem chain_forks {H : Msg → Msg} {L : CompiledLev Coin A} (hS : L.Sessions H) (x : Msg) (k : ℕ) :
    ∀ j, L.mid x = j + k → L.Chain x ((L.forks x).drop j) := by
  induction k with
  | zero =>
    intro j hj
    rw [List.drop_eq_nil_of_le (by rw [forks_length]; omega)]
    trivial
  | succ k ih =>
    intro j hj
    have hjm : j < L.mid x := by omega
    rw [L.forks_drop x hjm]
    refine ⟨?_, ih (j + 1) (by omega)⟩
    by_cases hj1 : j + 1 < L.mid x
    · rw [L.forks_drop x hj1]
      exact (hS.nested x j hjm).2 hj1
    · rw [List.drop_eq_nil_of_le (by rw [forks_length]; omega)]
      exact (hS.nested x j hjm).1

set_option synthInstance.maxSize 1024 in
omit [Inhabited Coin] in
/-- **One tape's outer wait at three levels or more**: for each candidate `Y`, the kept run's acceptance below averages
through the first middle wait (`Pr_acc_forkOn`), `pr_badBelow_le` bounds each run's term at its escape of `Y`
(`escB_le_of_above`), and the escape averages at most `Y`'s miss (`Ex_escAt_le`); the outer wait unions over at most
`2^{n_PU}` sets. -/
theorem outer_badDeep_le (L : CompiledLev Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (x : Msg)
    (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop)
    (Pt : Partition) (Lv : Verity.Levels (Fin Pt.blocks.length)) {l₀ : Verity.Level (Fin Pt.blocks.length)}
    {rest : List (Verity.Level (Fin Pt.blocks.length))} (hL : Lv.levels = l₀ :: rest)
    (hmid : L.mid x + 2 = Lv.levels.length) (hm : 0 < L.mid x) (hPU : L.nPU x = Pt.blocks.length)
    (hnOut : L.nPU x ≤ L.nOut x) (hch : L.Chain x (L.forks x)) (hsam : L.Samples x Pt Lv)
    (κr : (Fin (L.V.k x) → Coin) → ℝ) (hκr0 : ∀ r, 0 ≤ κr r)
    (hslots : ∀ r', (L.checked x r').length ≤ L.nPU x)
    (hsess : ∀ r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run' (forkOn (L.secondStage x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.secondStage x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.secondStage x) r p.1))[i.val]? = some u.val ∧
              ¬ found u (forkOn (L.secondStage x) r p.1) (p.2 i u)) ≤ κr r) :
    Pr (fun r : Fin (L.V.k x) → Coin => (run' r).out = .acc) *
        Pr (fun c : (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → L.WaitC x (L.forks x).length) =>
          (accList (fun r => (run' r).out = .acc) (L.kOut x) ((List.finRange (L.capOut x)).map c.1)).length <
              L.kOut x ∨
            ∃ Y ∈ cands Pt Lv δ, AllBad (fun r => (run' r).out = .acc) (L.kOut x)
              (fun r c' => L.BadBelow x run' found (L.forks x) (aboveDeep Pt Lv l₀ Y) r c')
              ((List.finRange (L.capOut x)).map c.1) c.2) ≤
      2 * (L.kOut x : ℝ) / L.capOut x + Real.exp (-(L.kOut x : ℝ) / 8) +
        slack * L.waitErr x (L.mid x) (δ + Ex κr) := by
  classical
  have hc := slack_pos
  have hEκ : 0 ≤ Ex κr := Ex_nonneg hκr0
  have hrest : rest ≠ [] := by
    rintro rfl
    rw [hL] at hmid
    simp at hmid
  obtain ⟨l₁, rest', rfl⟩ := List.exists_cons_of_ne_nil hrest
  have hfirst := hsam.2.1 l₀ l₁ rest' hL
  have hlen : (L.forks x).length = L.mid x := L.forks_length x
  have hnil : L.forks x ≠ [] := fun h => by rw [h] at hlen; simp at hlen; omega
  have hcard : ∀ Y, (aboveDeep Pt Lv l₀ Y).card ≤ 2 ^ Pt.blocks.length := fun Y =>
    (card_le_card (filter_subset _ _)).trans (by rw [card_powerset, card_range])
  have hy : ∀ Y ∈ cands Pt Lv δ,
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.WaitC x (L.forks x).length =>
        (run' p.1).out = .acc ∧ L.BadBelow x run' found (L.forks x) (aboveDeep Pt Lv l₀ Y) p.1 p.2) ≤
      L.waitErr x (L.mid x) (δ + Ex κr) := by
    intro Y hY
    have hYδ : missLev Pt Lv Y ≤ δ := (mem_filter.1 hY).2
    let esc : (Fin (L.V.k x) → Coin) → ℝ := fun r =>
      ∏ Z ∈ ((L.upper x Pt r).headD ∅).filter (· ∈ l₀.blocks.parts), (escAt Pt Lv 0 Z Y).toReal
    have hesc0 : ∀ r, 0 ≤ esc r := fun r => prod_nonneg fun _ _ => ENNReal.toReal_nonneg
    have hEesc : Ex esc ≤ δ :=
      (Ex_escAt_le Lv hL (List.cons_ne_nil _ _) (fun r => (L.upper x Pt r).headD ∅) hfirst Y).trans hYδ
    rw [Pr_acc_forkOn (L.nextF x (L.forks x)) (fun r => (run' r).out = .acc)
      (fun r c => L.BadBelow x run' found (L.forks x) (aboveDeep Pt Lv l₀ Y) r c)
      (fun r d c => L.badBelow_fork x run' found (L.forks x) _ r d c)]
    calc Ex (fun r => Pr (fun d => (run' (forkOn (L.nextF x (L.forks x)) r d)).out = .acc) *
          Pr (fun c => L.BadBelow x run' found (L.forks x) (aboveDeep Pt Lv l₀ Y) r c))
        ≤ Ex (fun r => L.waitErr x (L.forks x).length (esc r + L.κB x κr (L.forks x) r)) :=
          Ex_mono fun r => L.pr_badBelow_le x run' found hslots κr hκr0 hsess (L.forks x) hch _ Pt.blocks.length
            (hcard Y) (by omega) (fun h => absurd h hnil)
            (fun X hX v hv => by
              rw [hPU]
              exact mem_range.1 (mem_powerset.1 (mem_filter.1 hX).1 hv))
            r (esc r) (hesc0 r) fun X hX => L.escB_le_of_above x hsam hmid hX r
      _ = L.waitErr x (L.forks x).length (Ex esc + Ex κr) := by
          rw [L.Ex_waitErr x, Ex_add, L.Ex_κB x κr]
      _ ≤ L.waitErr x (L.mid x) (δ + Ex κr) := by
          rw [hlen]
          exact L.waitErr_mono x _ (by linarith)
  have hy0 : 0 ≤ L.waitErr x (L.mid x) (δ + Ex κr) := L.waitErr_nonneg x _ (add_nonneg hδ hEκ)
  have hy' : ∀ Xs ∈ (cands Pt Lv δ).image (aboveDeep Pt Lv l₀),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.WaitC x (L.forks x).length =>
        (run' p.1).out = .acc ∧ L.BadBelow x run' found (L.forks x) Xs p.1 p.2) ≤
      L.waitErr x (L.mid x) (δ + Ex κr) := by
    intro Xs hXs
    obtain ⟨Y, hY, rfl⟩ := mem_image.1 hXs
    exact hy Y hY
  have hw := wait_bad_le (fun r => (run' r).out = .acc) (L.kOut x) (L.capOut x) (L.capOut_pos x) (L.kOut_pos x)
    ((cands Pt Lv δ).image (aboveDeep Pt Lv l₀))
    (fun Xs r c => L.BadBelow x run' found (L.forks x) Xs r c) hy0 hy'
  have hcard0 : ((cands Pt Lv δ).image (aboveDeep Pt Lv l₀)).card ≤ 2 ^ Pt.blocks.length :=
    card_image_le.trans ((card_le_card (filter_subset _ _)).trans (by rw [card_powerset, card_range]))
  have hcs : ((((cands Pt Lv δ).image (aboveDeep Pt Lv l₀)).card : ℕ) : ℝ) ^ (1 / (L.kOut x : ℝ)) ≤ slack :=
    L.rpow_kOut_le_slack x hcard0 (by omega)
  refine (mul_le_mul_of_nonneg_left (Pr_mono fun c hev => ?_) (Pr_nonneg _)).trans
    (hw.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hcs hy0)))
  rcases hev with h | ⟨Y, hY, h⟩
  · exact Or.inl h
  · exact Or.inr ⟨aboveDeep Pt Lv l₀ Y, mem_image_of_mem _ hY, h⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem κLev_nonneg (L : CompiledLev Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (x : Msg) : 0 ≤ L.κLev δ x := by
  have := slack_pos
  have := Real.exp_pos (-(L.kOut x : ℝ) / 8)
  have := L.waitErr_nonneg x (L.mid x) (add_nonneg hδ (L.toCompiled.κs_nonneg x))
  unfold κLev
  positivity

set_option synthInstance.maxSize 1024 in
/-- **The composition at each tape at levels**: `Compiled.pr_le_of_tape` at `Elev`'s coins, `κLev` and `γLev`: when at
each tape `Pr[acc]·Pr[F] ≤ κLev(δ) + c^{m+2}·E_r[√(k(x)·γ_s·β(ω, r))]`, over tapes, judged coins and the extractor's
coins the judged run accepts and `F` holds with probability at most `κLev(δ) + √(γLev·Pr[B])`. -/
theorem pr_le_of_tapeLev (L : CompiledLev Coin A) (H : Msg → Msg) {δ : ℝ} (x : Msg)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (F : Ω → L.Elev.C x → Prop)
    (β : Ω → (Fin (L.V.k x) → Coin) → ℝ) (hβ0 : ∀ ω r, 0 ≤ β ω r)
    (hβ : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x)
    (hper : ∀ ω, Pr (fun r => (run H P L.V x ω r).out = .acc) * Pr (F ω) ≤
      L.κLev δ x + slack ^ (L.mid x + 2) * Ex (fun r => Real.sqrt (L.V.k x * L.γs x * β ω r))) :
    Pr (fun s : Ω × (Fin (L.V.k x) → Coin) × L.Elev.C x => (run H P L.V x s.1 s.2.1).out = .acc ∧ F s.1 s.2.2) ≤
      L.κLev δ x + Real.sqrt (L.γLev x * PrB H L.V P x) := by
  refine (Claims.Pr_indep (fun ω r => (run H P L.V x ω r).out = .acc) F).trans_le ?_
  have hc := slack_pos
  have hγ0 : 0 ≤ (L.V.k x : ℝ) * L.γs x := mul_nonneg (Nat.cast_nonneg _) (L.toCompiled.γs_nonneg x)
  have hp0 : 0 ≤ slack ^ (L.mid x + 2) := pow_nonneg hc.le _
  calc Ex (fun ω => Pr (fun r => (run H P L.V x ω r).out = .acc) * Pr (F ω))
      ≤ Ex (fun ω => L.κLev δ x + slack ^ (L.mid x + 2) *
          Ex (fun r => Real.sqrt (L.V.k x * L.γs x * β ω r))) := Ex_mono hper
    _ = L.κLev δ x + slack ^ (L.mid x + 2) *
          Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2)) := by
        rw [Ex_add, Ex_const, Ex_mul_left,
          Ex_prod (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2))]
    _ ≤ L.κLev δ x + slack ^ (L.mid x + 2) * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) := by
        refine add_le_add le_rfl (mul_le_mul_of_nonneg_left ?_ hp0)
        refine (Claims.Ex_sqrt_le _ fun p => mul_nonneg hγ0 (hβ0 _ _)).trans (Real.sqrt_le_sqrt ?_)
        rw [Ex_mul_left]
        exact mul_le_mul_of_nonneg_left hβ hγ0
    _ = L.κLev δ x + Real.sqrt (L.γLev x * PrB H L.V P x) := by
        rw [show L.γLev x * PrB H L.V P x = (slack ^ (L.mid x + 2)) ^ 2 * (L.V.k x * L.γs x * PrB H L.V P x) by
            unfold γLev; rw [← pow_mul, mul_comm (L.mid x + 2) 2]; ring,
          Real.sqrt_mul (pow_nonneg hp0 2) (L.V.k x * L.γs x * PrB H L.V P x), Real.sqrt_sq hp0]

set_option synthInstance.maxSize 1024 in
/-- **The failure event's probability at three levels or more**: `pr_le_of_tapeLev` with, at each tape, the outer
wait's bound (`outer_badDeep_le`), its samples from `Samples`, its unions from `Unions` and its chain from `nested`. -/
theorem pr_failsDeep_le (H : Msg → Msg) (L : CompiledLev Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (hS : L.Sessions H)
    (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (found : Ω → (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop)
    (β : Ω → (Fin (L.V.k x) → Coin) → ℝ) (hβ0 : ∀ ω r, 0 ≤ β ω r)
    (hβ : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x)
    (hsess : ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run H P L.V x ω (forkOn (L.secondStage x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.secondStage x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.secondStage x) r p.1))[i.val]? = some u.val ∧
              ¬ found ω u (forkOn (L.secondStage x) r p.1) (p.2 i u)) ≤
        L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) :
    Pr (fun s : Ω × (Fin (L.V.k x) → Coin) × L.Elev.C x =>
      (run H P L.V x s.1 s.2.1).out = .acc ∧ L.FailsDeep δ x (run H P L.V x s.1) (found s.1) s.2.2) ≤
      L.κLev δ x + Real.sqrt (L.γLev x * PrB H L.V P x) := by
  refine L.pr_le_of_tapeLev H x Ω P (fun ω c => L.FailsDeep δ x (run H P L.V x ω) (found ω) c) β hβ0 hβ
    fun ω => ?_
  have hκ0 := L.κLev_nonneg hδ x
  by_cases hsc : ∃ s Pt Lv l₀ rest, L.parse x = some s ∧ L.partition s = some Pt ∧ L.levels s Pt = some Lv ∧
      InScopeLev Pt Lv (L.top s) s ∧ Lv.levels = l₀ :: rest ∧ 0 < L.mid x
  swap
  · have h0 : Pr (fun c : L.Elev.C x => L.FailsDeep δ x (run H P L.V x ω) (found ω) c) = 0 :=
      (Pr_congr fun c => ⟨fun ⟨s, Pt, Lv, l₀, rest, h1, h2, h3, h4, h5, h6, _⟩ =>
        hsc ⟨s, Pt, Lv, l₀, rest, h1, h2, h3, h4, h5, h6⟩, False.elim⟩).trans Pr_false
    rw [h0, mul_zero]
    exact add_nonneg hκ0 (mul_nonneg (pow_nonneg slack_pos.le _) (Ex_nonneg fun _ => Real.sqrt_nonneg _))
  obtain ⟨s, Pt, Lv, l₀, rest, hs, hPt, hLv, hin, hL, hm⟩ := hsc
  have hF : Pr (fun c : L.Elev.C x => L.FailsDeep δ x (run H P L.V x ω) (found ω) c) =
      Pr (fun c : (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → L.WaitC x (L.forks x).length) =>
        (accList (fun r => (run H P L.V x ω r).out = .acc) (L.kOut x)
          ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x ∨
          ∃ Y ∈ cands Pt Lv δ, AllBad (fun r => (run H P L.V x ω r).out = .acc) (L.kOut x)
            (fun r c' => L.BadBelow x (run H P L.V x ω) (found ω) (L.forks x) (aboveDeep Pt Lv l₀ Y) r c')
            ((List.finRange (L.capOut x)).map c.1) c.2) := by
    refine Pr_congr fun c => ⟨fun ⟨s', Pt', Lv', l₀', rest', hs', hPt', hLv', _, hL', _, h⟩ => ?_,
      fun h => ⟨s, Pt, Lv, l₀, rest, hs, hPt, hLv, hin, hL, hm, h⟩⟩
    rw [hs, Option.some.injEq] at hs'
    subst hs'
    rw [hPt, Option.some.injEq] at hPt'
    subst hPt'
    rw [hLv, Option.some.injEq] at hLv'
    subst hLv'
    rw [hL, List.cons.injEq] at hL'
    obtain ⟨rfl, -⟩ := hL'
    exact h
  rw [hF]
  have hsam := hS.samples x s Pt Lv hs hPt hLv hin
  have hun := hS.unions x s Pt Lv hs hPt hLv hin
  have hnOut : L.nPU x ≤ L.nOut x := by
    cases htop : L.top s with
    | none => rw [htop] at hun; exact hun
    | some Q => rw [htop] at hun; exact hun.1
  have hmid : L.mid x + 2 = Lv.levels.length := by
    rw [mid_eq hs hPt hLv]
    have h := hin.2.2.2
    have : 2 ≤ Lv.levels.length := by
      rw [mid_eq hs hPt hLv] at hm
      omega
    omega
  have hch : L.Chain x (L.forks x) := by
    have h := chain_forks hS x (L.mid x) 0 (by omega)
    rwa [List.drop_zero] at h
  refine (L.outer_badDeep_le hδ x (run H P L.V x ω) (found ω) Pt Lv hL hmid hm (Compiled.nPU_eq hs hPt) hnOut
    hch hsam (fun r => L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r))
    (fun r => add_nonneg (L.toCompiled.κs_nonneg x) (Real.sqrt_nonneg _)) (hS.slots x) (hsess ω)).trans
    (le_of_eq ?_)
  rw [Ex_add, Ex_const, L.waitErr_eq x (L.mid x) (δ + _)]
  unfold κLev
  rw [L.waitErr_eq x (L.mid x) (δ + L.κs x)]
  ring

end CompiledLev

end Claims.SamplingGeneric
