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

end Claims.SamplingGeneric
