import Proofs.Flock.Recursive.Defs
import Proofs.Flock.Soundness.StrictCR

/-!
# Recursion: binding a stage to a reference opening fixed before its coin

The stage-`k` commitment `c_k` is a vector of hiding commitments, one digest per position `pos`. The outer session's
extraction reads openings of them (a **trial** at each leaf, `T t s pos`, which outputs the openings it read at `pos`).
The binding hypothesis Flock proves is a **transfer**: for every candidate value vector `v`, the decoded round-`k` message
differs from `msg v` with probability at most a term `εtr` plus, at each position, `κ pos` times the probability that the
trial reads an opening of `c_k`'s digest at `pos` to a value other than `v pos`.

The reference opening has to be a function of the prover's state when it sends `c_k`, so that the simulated inner prover
fixes its round-`k` message before `ρ_k`. A plurality over the extraction's openings at the leaf is not one: it depends on
`ρ_k`. This file fixes it at the node instead:

* `refv p c pos` is the value the rest of the run from the node (`contF`: the remaining coins, then the trial) reads
  **alone** at `pos` most often, the readings `A ⊆ V` counted when `A = {v}` (`alone_plurality_sq`);
* off that value, two independent continuations from the node conflict: if a run reads some `u ∈ A`, every run whose
  readings are nonempty and not `{u}` reads a value other than `u`, and by plurality those runs weigh at least the
  off-reference mass `b`; so `b² ≤ Pr₂[conflict]`;
* a conflict is two openings of one digest to different values, a collision of `H` that the **fork finder** (`forkF`:
  the coins to the node, then two continuations) outputs (`Openings.collide`);
* `(Σ b_node)² ≤ Σ b_node²` over the coins to the node (Cauchy–Schwarz), so with `cr/sha-512` for the fork finder at
  budget `q` the off-reference mass is at most `√(q²/2^513) = q/2^256.5`: the rewinding costs a square root.

`stage_bind` puts the two together: the stage's binding error is `leafAvg εtr + (Σ_pos κ pos)·√(q²/2^513)`.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

/-- **Openings of a binding commitment**, as the stage binding reads them: an opening `o` opens the digest `com o` to the
value `val o`, and two openings of one digest to different values are a collision of `H` (`collide`), as
`ValueBindingZ`'s are. -/
structure Openings {D : Type} (H : List UInt8 → D) (V : Type) : Type 1 where
  Op : Type
  Dg : Type
  com : Op → Dg
  val : Op → V
  collide : Op → Op → List UInt8 × List UInt8
  collide_spec : ∀ o o', com o = com o' → val o ≠ val o' →
    (collide o o').1 ≠ (collide o o').2 ∧ H (collide o o').1 = H (collide o o').2

variable {D : Type} {H : List UInt8 → D} {V : Type} (ob : Openings H V)

/-- The openings `os` open the digest `d` to a value other than `v`. -/
def OffAt (d : ob.Dg) (v : V) (os : Finset ob.Op) : Prop := ∃ o ∈ os, ob.com o = d ∧ ob.val o ≠ v

variable {Cm C Pos β γ : Type} [Fintype C] [Nonempty C]

open Classical in
/-- **A fork's output**: an opening from each run, of the digest `d` to different values, as a collision. -/
noncomputable def forkPick (d : ob.Dg) (os os' : Finset ob.Op) : Pair :=
  if h : ∃ o ∈ os, ∃ o' ∈ os', ob.com o = d ∧ ob.com o' = d ∧ ob.val o ≠ ob.val o' then
    some (ob.collide h.choose h.choose_spec.2.choose)
  else none

/-- **The fork finder at stage `k` and position `pos`**: the round coins to the node where the prover sends its stage-`k`
commitment `c`, then the rest of the run twice, independently (`contF`, the trial `T … pos` at each leaf), and the two
runs' openings of `c`'s digest at `pos` to different values (`forkPick`). With no stage `k` it finds nothing. -/
noncomputable def forkF (cmt : Cm → Pos → ob.Dg) (pos : Pos) : (k r : ℕ) → (f : List (Cm × C) → Game β) →
    ((t : List (Cm × C)) → Strategy (f t) → Pos → Finder (Finset ob.Op)) → Strategy (rounds Cm C r f) → Finder Pair
  | _, 0, _, _, _ => ⟨.ret none, PUnit.unit⟩
  | 0, r + 1, f, T, s =>
      twoRuns (contF (r + 1) f (fun t s' => T t s' pos) s).g (contF (r + 1) f (fun t s' => T t s' pos) s).s
        (forkPick ob (cmt s.1 pos))
  | k + 1, r + 1, f, T, s =>
      ⟨.coin C fun ρ => (forkF cmt pos k r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).g,
        fun ρ => (forkF cmt pos k r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).s⟩

/-- **The probability, at a leaf, that the trial at `pos` reads `c_k`'s digest off the reference value.** -/
noncomputable def offProb (cmt : Cm → Pos → ob.Dg) (k : ℕ) (refv : List (Cm × C) → Cm → Pos → V)
    {f : List (Cm × C) → Game β} (T : (t : List (Cm × C)) → Strategy (f t) → Pos → Finder (Finset ob.Op))
    (t : List (Cm × C)) (s : Strategy (f t)) (pos : Pos) : ℝ≥0∞ :=
  match t[k]? with
  | some e => prob (OffAt ob (cmt e.1 pos) (refv (t.take k) e.1 pos)) (T t s pos).g (T t s pos).s
  | none => 0

theorem prob_contF (p : γ → Prop) : ∀ (r : ℕ) (f : List (Cm × C) → Game β)
    (T : (t : List (Cm × C)) → Strategy (f t) → Finder γ) (s : Strategy (rounds Cm C r f)),
    prob p (contF r f T s).g (contF r f T s).s = leafAvg r f (fun t s' => prob p (T t s').g (T t s').s) s
  | 0, _, _, _ => rfl
  | r + 1, f, T, s => congrArg avg (funext fun ρ =>
      prob_contF p r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ))

open Classical in
/-- The values the openings `os` open the digest `d` to. -/
noncomputable def Aof (d : ob.Dg) (os : Finset ob.Op) : Finset V := (os.filter fun o => ob.com o = d).image ob.val

theorem offAt_iff (d : ob.Dg) (v : V) (os : Finset ob.Op) : OffAt ob d v os ↔ ∃ x ∈ Aof ob d os, x ≠ v := by
  classical
  simp only [OffAt, Aof, Finset.mem_image, Finset.mem_filter]
  constructor
  · rintro ⟨o, ho, hd, hv⟩
    exact ⟨_, ⟨o, ⟨ho, hd⟩, rfl⟩, hv⟩
  · rintro ⟨_, ⟨o, ⟨ho, hd⟩, rfl⟩, hv⟩
    exact ⟨o, ho, hd, hv⟩

theorem ind_offAt (d : ob.Dg) (v : V) :
    ind (OffAt ob d v) = ind fun os => ∃ x ∈ Aof ob d os, x ≠ v :=
  funext fun os => by
    unfold ind
    by_cases h : OffAt ob d v os
    · rw [ite_eq_left h, ite_eq_left ((offAt_iff ob d v os).1 h)]
    · rw [ite_eq_right h, ite_eq_right fun h' => h ((offAt_iff ob d v os).2 h')]

/-- **A conflict of two runs is a collision** the fork outputs. -/
theorem ind_le_forkPick (d : ob.Dg) (os os' : Finset ob.Op) :
    ind (fun os' => ∃ x ∈ Aof ob d os, ∃ y ∈ Aof ob d os', x ≠ y) os' ≤ ind (IsColl H) (forkPick ob d os os') := by
  classical
  by_cases h : ∃ x ∈ Aof ob d os, ∃ y ∈ Aof ob d os', x ≠ y
  · obtain ⟨x, hx, y, hy, hxy⟩ := h
    simp only [Aof, Finset.mem_image, Finset.mem_filter] at hx hy
    obtain ⟨o, ⟨ho, hod⟩, rfl⟩ := hx
    obtain ⟨o', ⟨ho', hod'⟩, rfl⟩ := hy
    have hex : ∃ o ∈ os, ∃ o' ∈ os', ob.com o = d ∧ ob.com o' = d ∧ ob.val o ≠ ob.val o' :=
      ⟨o, ho, o', ho', hod, hod', hxy⟩
    have hc : IsColl H (forkPick ob d os os') := by
      unfold forkPick
      rw [dite_eq_left hex]
      obtain ⟨-, h1, h2, h3⟩ := hex.choose_spec.2.choose_spec
      exact ⟨_, _, rfl, ob.collide_spec _ _ (h1.trans h2.symm) h3⟩
    simp only [ind, ite_eq_left hc]
    split_ifs <;> norm_num
  · simp only [ind, ite_eq_right h]
    split_ifs <;> norm_num

open Classical in
/-- **The alone plurality**: for readings `A` of a run, some value `vs` is read off (some reading other than `vs`) with a
mass `b` whose square is at most the probability that two independent runs read different values. `vs` is the value
read alone most often: a run reading `u` conflicts with every run whose readings are nonempty and not `{u}`, and those
weigh at least `b`. -/
theorem alone_plurality_sq {α W : Type} [Fintype W] [Nonempty W] (g : Game α) (s : Strategy g)
    (A : α → Finset W) :
    ∃ vs : W, expect (ind fun a => ∃ x ∈ A a, x ≠ vs) g s ^ 2 ≤
      expect₂ (fun a a' => ind (fun a' => ∃ x ∈ A a, ∃ y ∈ A a', x ≠ y) a') g s := by
  set al : W → ℝ := fun v => expect (ind fun a => A a = {v}) g s
  obtain ⟨vs, -, hvs⟩ := Finset.exists_max_image (Finset.univ : Finset W) al Finset.univ_nonempty
  refine ⟨vs, ?_⟩
  set o : ℝ := expect (ind fun a => (A a).Nonempty) g s
  set b : ℝ := expect (ind fun a => ∃ x ∈ A a, x ≠ vs) g s
  have hb0 : 0 ≤ b := expect_nonneg (ind_nonneg _) g s
  have single : ∀ (B : Finset W) (u : W), B.Nonempty → B ≠ {u} → ∃ y ∈ B, y ≠ u := by
    intro B u hne h1
    by_contra hno
    push Not at hno
    obtain ⟨y, hy⟩ := hne
    exact h1 (Finset.eq_singleton_iff_unique_mem.2 ⟨hno y hy ▸ hy, hno⟩)
  have hbo : b + al vs = o := by
    simp only [b, al, o]
    rw [← expect_add]
    refine congrArg (fun F => expect F g s) (funext fun a => ?_)
    by_cases h1 : A a = {vs}
    · have h2 : ¬ ∃ x ∈ A a, x ≠ vs := by rw [h1]; simp
      simp [ind, h1]
    · by_cases h3 : (A a).Nonempty
      · simp [ind, h1, h3, single (A a) vs h3 h1]
      · have h2 : ¬ ∃ x ∈ A a, x ≠ vs := fun ⟨x, hx, _⟩ => h3 ⟨x, hx⟩
        simp [ind, h1, h2, h3]
  have hin : ∀ a, b * ind (fun a => ∃ x ∈ A a, x ≠ vs) a ≤
      expect (fun a' => ind (fun a' => ∃ x ∈ A a, ∃ y ∈ A a', x ≠ y) a') g s := by
    intro a
    by_cases ha : ∃ x ∈ A a, x ≠ vs
    · obtain ⟨u, hu, -⟩ := ha
      have hpt : ∀ a', ind (fun a' => (A a').Nonempty) a' - ind (fun a' => A a' = {u}) a' ≤
          ind (fun a' => ∃ x ∈ A a, ∃ y ∈ A a', x ≠ y) a' := by
        intro a'
        by_cases h1 : A a' = {u}
        · have : ind (fun a' => (A a').Nonempty) a' = 1 := by simp [ind, h1]
          rw [this, show ind (fun a' => A a' = {u}) a' = 1 by simp [ind, h1], sub_self]
          exact ind_nonneg _ _
        · by_cases h2 : (A a').Nonempty
          · obtain ⟨y, hy, hyu⟩ := single (A a') u h2 h1
            have hc : ∃ x ∈ A a, ∃ y ∈ A a', x ≠ y := ⟨u, hu, y, hy, Ne.symm hyu⟩
            simp [ind, h1, h2, hc]
          · have : ind (fun a' => (A a').Nonempty) a' = 0 := by simp [ind, h2]
            rw [this, show ind (fun a' => A a' = {u}) a' = 0 by simp [ind, h1], sub_self]
            exact ind_nonneg _ _
      have hE : o - al u ≤ expect (fun a' => ind (fun a' => ∃ x ∈ A a, ∃ y ∈ A a', x ≠ y) a') g s := by
        simp only [o, al]
        rw [← expect_sub]
        exact expect_mono hpt g s
      have hα : al u ≤ al vs := hvs u (Finset.mem_univ u)
      have h1 : ind (fun a => ∃ x ∈ A a, x ≠ vs) a ≤ 1 := ind_le_one _ _
      nlinarith
    · rw [show ind (fun a => ∃ x ∈ A a, x ≠ vs) a = 0 by simp [ind, ha], mul_zero]
      exact expect_nonneg (fun _ => ind_nonneg _ _) g s
  calc b ^ 2 = expect (fun a => b * ind (fun a => ∃ x ∈ A a, x ≠ vs) a) g s := by
        rw [expect_const_mul, sq]
    _ ≤ _ := expect_mono hin g s

/-- **Cauchy–Schwarz over a coin**: the square of an average is at most the average of the squares. -/
theorem avg_toReal_sq_le {D : Type} [Fintype D] [Nonempty D] (f : D → ℝ≥0∞) (hf : ∀ c, f c ≠ ⊤) :
    (avg f).toReal ^ 2 ≤ (∑ c, (f c).toReal ^ 2) / Fintype.card D := by
  have hN : (0 : ℝ) < Fintype.card D := by exact_mod_cast Fintype.card_pos
  have havg : (avg f).toReal = (∑ c, (f c).toReal) / Fintype.card D := by
    unfold avg
    rw [ENNReal.toReal_sum fun c _ => ENNReal.mul_ne_top
      (ENNReal.inv_ne_top.2 (by exact_mod_cast Fintype.card_ne_zero)) (hf c)]
    simp only [ENNReal.toReal_mul, ENNReal.toReal_inv, ENNReal.toReal_natCast]
    rw [← Finset.mul_sum, inv_mul_eq_div]
  have hcs : (∑ c, (f c).toReal) ^ 2 ≤ Fintype.card D * ∑ c, (f c).toReal ^ 2 := by
    simpa using sq_sum_le_card_mul_sum_sq (s := (Finset.univ : Finset D)) (f := fun c => (f c).toReal)
  rw [havg, div_pow, div_le_div_iff₀ (by positivity) hN]
  calc (∑ c, (f c).toReal) ^ 2 * Fintype.card D ≤ (Fintype.card D * ∑ c, (f c).toReal ^ 2) * Fintype.card D := by
        gcongr
    _ = (∑ c, (f c).toReal ^ 2) * (Fintype.card D : ℝ) ^ 2 := by ring

omit [Fintype C] [Nonempty C] in
theorem offProb_le_one (cmt : Cm → Pos → ob.Dg) (k : ℕ) (refv : List (Cm × C) → Cm → Pos → V)
    {f : List (Cm × C) → Game β} (T : (t : List (Cm × C)) → Strategy (f t) → Pos → Finder (Finset ob.Op))
    (t : List (Cm × C)) (s : Strategy (f t)) (pos : Pos) : offProb ob cmt k refv T t s pos ≤ 1 := by
  unfold offProb
  split
  · exact prob_le_one _ _ _
  · exact zero_le_one

theorem leafAvg_zero_offProb {cmt : Cm → Pos → ob.Dg} {k : ℕ} {refv : List (Cm × C) → Cm → Pos → V}
    {f : List (Cm × C) → Game β} {T : (t : List (Cm × C)) → Strategy (f t) → Pos → Finder (Finset ob.Op)}
    {s : Strategy (rounds Cm C 0 f)} {pos : Pos} :
    leafAvg 0 f (fun t s' => offProb ob cmt k refv T t s' pos) s = 0 := by
  show offProb ob cmt k refv T [] s pos = 0
  simp [offProb]

/-- The reference values one level down, after the coin `ρ`. -/
noncomputable def consRef [Nonempty V] (rv : C → List (Cm × C) → Cm → Pos → V) : List (Cm × C) → Cm → Pos → V
  | [], _, _ => Classical.arbitrary V
  | (_, ρ) :: p, c, pos => rv ρ p c pos

omit [Fintype C] [Nonempty C] in
theorem offProb_cons [Nonempty V] (cmt : Cm → Pos → ob.Dg) (k : ℕ) (rv : C → List (Cm × C) → Cm → Pos → V)
    {f : List (Cm × C) → Game β} (T : (t : List (Cm × C)) → Strategy (f t) → Pos → Finder (Finset ob.Op))
    (m : Cm) (ρ : C) (t : List (Cm × C)) (s : Strategy (f ((m, ρ) :: t))) (pos : Pos) :
    offProb ob cmt (k + 1) (consRef rv) T ((m, ρ) :: t) s pos =
      offProb ob cmt k (rv ρ) (fun t => T ((m, ρ) :: t)) t s pos := by
  unfold offProb
  simp only [List.getElem?_cons_succ, List.take_succ_cons, consRef]

/-- **The plurality reference and its fork**: some reference values, fixed at the node where each stage-`k` commitment is
sent, are read off by the trials with a mass whose square is at most the fork finder's success. -/
theorem fork_ref [Fintype V] [DecidableEq V] [Nonempty V] (cmt : Cm → Pos → ob.Dg) :
    ∀ (k r : ℕ) (f : List (Cm × C) → Game β)
      (T : (t : List (Cm × C)) → Strategy (f t) → Pos → Finder (Finset ob.Op)) (s : Strategy (rounds Cm C r f)),
    ∃ refv : List (Cm × C) → Cm → Pos → V, ∀ pos,
      (leafAvg r f (fun t s' => offProb ob cmt k refv T t s' pos) s).toReal ^ 2 ≤
        (forkF ob cmt pos k r f T s).ev (ind (IsColl H)) := by
  intro k
  induction k with
  | zero =>
      intro r f T s
      cases r with
      | zero =>
          refine ⟨fun _ _ _ => Classical.arbitrary V, fun pos => ?_⟩
          rw [leafAvg_zero_offProb, ENNReal.toReal_zero, sq, mul_zero]
          exact expect_nonneg (ind_nonneg _) _ _
      | succ r =>
          choose vs hvs using fun pos => alone_plurality_sq (contF (r + 1) f (fun t s' => T t s' pos) s).g
            (contF (r + 1) f (fun t s' => T t s' pos) s).s (Aof ob (cmt s.1 pos))
          refine ⟨fun _ _ pos => vs pos, fun pos => ?_⟩
          have hL : leafAvg (r + 1) f (fun t s' => offProb ob cmt 0 (fun _ _ pos => vs pos) T t s' pos) s =
              prob (OffAt ob (cmt s.1 pos) (vs pos)) (contF (r + 1) f (fun t s' => T t s' pos) s).g
                (contF (r + 1) f (fun t s' => T t s' pos) s).s := by
            rw [prob_contF]
            exact congrArg avg (funext fun ρ => congrArg (fun F => leafAvg r _ F (s.2 ρ))
              (funext fun t => funext fun s' => by simp [offProb]))
          rw [hL, prob_toReal, ind_offAt]
          refine (hvs pos).trans ?_
          show _ ≤ (twoRuns _ _ _).ev _
          rw [twoRuns_ev]
          exact expect_mono (fun os => expect_mono (fun os' => ind_le_forkPick ob _ os os') _ _) _ _
  | succ k ih =>
      intro r f T s
      cases r with
      | zero =>
          refine ⟨fun _ _ _ => Classical.arbitrary V, fun pos => ?_⟩
          rw [leafAvg_zero_offProb, ENNReal.toReal_zero, sq, mul_zero]
          exact expect_nonneg (ind_nonneg _) _ _
      | succ r =>
          choose rv hrv using fun ρ => ih r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)
          refine ⟨consRef rv, fun pos => ?_⟩
          have hL : leafAvg (r + 1) f (fun t s' => offProb ob cmt (k + 1) (consRef rv) T t s' pos) s =
              avg fun ρ => leafAvg r (fun t => f ((s.1, ρ) :: t))
                (fun t s' => offProb ob cmt k (rv ρ) (fun t => T ((s.1, ρ) :: t)) t s' pos) (s.2 ρ) :=
            congrArg avg (funext fun ρ => congrArg (fun F => leafAvg r _ F (s.2 ρ))
              (funext fun t => funext fun s' => offProb_cons ob cmt k rv T s.1 ρ t s' pos))
          rw [hL]
          refine (avg_toReal_sq_le _ fun ρ => ne_top_of_le_ne_top ENNReal.one_ne_top
            (leafAvg_le_one (fun t s' => offProb_le_one ob cmt k _ _ t s' pos) _)).trans ?_
          show _ ≤ (∑ ρ, (forkF ob cmt pos k r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t))
            (s.2 ρ)).ev (ind (IsColl H))) / Fintype.card C
          gcongr with ρ
          exact hrv ρ pos

/-- **Stage binding.** Given the transfer `htr` at stage `k` and `cr/sha-512` for the fork finders at budget `q`, some
reference values fixed at the stage's nodes bind the decoded round-`k` message: `E t s m` is the leaf's probability that
the decoded message is not `m`. -/
theorem stage_bind [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos] {Msg : Type}
    (cmt : Cm → Pos → ob.Dg) (msg : (Pos → V) → Msg) {k r : ℕ} (hk : k < r) (f : List (Cm × C) → Game β)
    (T : (t : List (Cm × C)) → Strategy (f t) → Pos → Finder (Finset ob.Op)) (s : Strategy (rounds Cm C r f))
    (E : (t : List (Cm × C)) → Strategy (f t) → Option Msg → ℝ≥0∞)
    (εtr : (t : List (Cm × C)) → Strategy (f t) → ℝ≥0∞) (κ : Pos → ℝ≥0∞)
    (htr : AllLeaves r f (fun t s' => ∀ (v : Pos → V) (e : Cm × C), t[k]? = some e →
      E t s' (some (msg v)) ≤ εtr t s' + ∑ pos, κ pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T t s' pos).g (T t s' pos).s) s)
    {q : ℝ} (hCR : ∀ pos, (forkF ob cmt pos k r f T s).CR H q) :
    ∃ refv : List (Cm × C) → Cm → Pos → V,
      leafAvg r f (fun t s' => E t s' ((t[k]?).map fun e => msg (refv (t.take k) e.1))) s ≤
        leafAvg r f εtr s + (∑ pos, κ pos) * ENNReal.ofReal (Real.sqrt (q ^ 2 / 2 ^ (513 : ℝ))) := by
  obtain ⟨refv, hinv⟩ := fork_ref ob cmt k r f T s
  refine ⟨refv, ?_⟩
  have hoff : ∀ pos, leafAvg r f (fun t s' => offProb ob cmt k refv T t s' pos) s ≤
      ENNReal.ofReal (Real.sqrt (q ^ 2 / 2 ^ (513 : ℝ))) := by
    intro pos
    have h1 : leafAvg r f (fun t s' => offProb ob cmt k refv T t s' pos) s ≠ ⊤ :=
      ne_top_of_le_ne_top ENNReal.one_ne_top (leafAvg_le_one (fun t s' => offProb_le_one ob cmt k _ _ t s' pos) s)
    rw [← ENNReal.ofReal_toReal h1]
    exact ENNReal.ofReal_le_ofReal (Real.le_sqrt_of_sq_le ((hinv pos).trans (hCR pos).le))
  calc leafAvg r f (fun t s' => E t s' ((t[k]?).map fun e => msg (refv (t.take k) e.1))) s
      ≤ leafAvg r f (fun t s' => εtr t s' + ∑ pos, κ pos * offProb ob cmt k refv T t s' pos) s := by
        refine leafAvg_mono_all (htr.mono fun t s' ht h => ?_)
        obtain ⟨e, he⟩ : ∃ e, t[k]? = some e := ⟨t[k], List.getElem?_eq_getElem (by omega)⟩
        simp only [he, Option.map_some, offProb]
        exact h _ e he
    _ = leafAvg r f εtr s + ∑ pos, κ pos * leafAvg r f (fun t s' => offProb ob cmt k refv T t s' pos) s := by
        rw [leafAvg_add, leafAvg_sum]
        simp only [leafAvg_const_mul]
    _ ≤ leafAvg r f εtr s + ∑ pos, κ pos * ENNReal.ofReal (Real.sqrt (q ^ 2 / 2 ^ (513 : ℝ))) := by
        gcongr with pos
        exact hoff pos
    _ = _ := by rw [Finset.sum_mul]

end FlockSoundness.Discharge.Recursion
