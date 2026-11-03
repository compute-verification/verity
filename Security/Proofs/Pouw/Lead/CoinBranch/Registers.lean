import Proofs.Pouw.Lead.CoinBranch.PointwiseProofs

/-!
# Coin branch: the chunk model against a program that branches on the coins (proofs; draft model, pending review)

At each coin outcome `ω` such a program is a run `P ω` from the memory `m₀ ω` that holds the targets `Y ω` at some
point, as in `adaptive_pairs_le`. Each bound reads only the targets an outcome realizes, so it sums over any outcomes.

* **Chunk provenance** (`run_chunks_le`, `adaptive_chunks_le`). By `good_prefix`, every slot after every prefix holds
  an initial chunk or one of `writes P` new ones, so the targets no initial chunk holds number at most `writes P`.
* **Balls against any initial memory** (`run_ball_table_le`, `adaptive_ball_table_le`). An initial chunk within `r y`
  of a target is within it of no other, so `run_ball_le` holds with no hypothesis on `m₀`, less `R + 1` a slot. Its
  count is at most `⌊(w − 1)/2⌋ + 1` a target, about half of the target's `w` words.
* **Per-entry credit at word writes** (`run_words_le`, `adaptive_entries_le`). Each entry sits at one position of a
  chunk; a run writes at least the new (position, value) pairs it holds, so when no two entries at one position share
  a value and the initial memory holds none, the entries' distinct values are paid in full.
* **Per-entry credit under whole-register writes** (`card_lanes_le`, `adaptive_lanes_le`, `adaptive_registers_le`).
  With a register as one word, a write sets all its lanes; the lanes' distinct values over the steps are at most the
  lanes times the register's, so the per-entry distinct count is paid, less the lanes for each register value
  repeated across groups or held initially, and in full when none is.
* **The attained side** (`pathRun_le_card`, `adaptive_attained`, `adaptive_levels_attained`,
  `adaptive_ones_attained`). The path run (`pathRun_attained`) holds any targets at `w` writes a distinct value.
  Row-step chunks that are a function of the row and its level, `G i (c i t)`, number at most `Σ_i |{c i t}|`, and a
  row whose chunk repeats at every step whose coin bit is `false` takes at most `1 + #true bits` values.
* **Priced selection** (`Priced`: `credit_le_cost`, `credit_le_cost_floor`, `credit_le_cost_separated`,
  `prog_credit_le`). Uniform branches split the outcomes into classes, each running one static stream whose
  instructions are priced whether or not their predicates hold. Check (b)'s credit, in writes, is at most the cost,
  plus a write for each instruction priced below a write and for each read the class shares across entries.
  Select-floor and copy-floor (a selection on the lane's own bit and a copy each cost at least a write) leave only the
  shared reads, and a class over which no two entries' values agree everywhere shares none.
* **Premises per path** (`Tree.mem_cls_iff`, `small_cls_le`). A class is the outcomes that take its path, and the
  outcomes in classes of fewer than `c` number at most `c` a path, so a premise stated on every path's share of the
  coins of at least `c` outcomes holds at all others.
-/

namespace Pouw.SecurityProofs.Lead.CoinBranch

open Finset Pouw.Dimension.Pointwise

variable {A V : Type} {w : ℕ}

/-- Row-step values number at most the sum over rows of each row's distinct values. -/
theorem card_rows_le {X ι T : Type} [DecidableEq X] (rows : Finset ι) (steps : Finset T) (F : ι → T → X) :
    ((rows ×ˢ steps).image fun it => F it.1 it.2).card ≤ ∑ i ∈ rows, (steps.image (F i)).card := by
  calc _ ≤ (rows.biUnion fun i => steps.image (F i)).card := by
        apply card_le_card
        intro y hy
        obtain ⟨⟨i, t⟩, hit, rfl⟩ := mem_image.1 hy
        rw [mem_product] at hit
        exact mem_biUnion.2 ⟨i, hit.1, mem_image.2 ⟨t, hit.2, rfl⟩⟩
    _ ≤ _ := card_biUnion_le

/-- **Levels collapse**: row-step values that are a function of the row and its level, `G i (c i t)`, number at most
`Σ_i |{c i t}|`. The walk's chunks `base i + W_t · s` (s fixed a row: `1` on the tag lead, the column signs on hw) are
of this form. -/
theorem card_levels_le {X L ι T : Type} [DecidableEq X] [DecidableEq L] (rows : Finset ι) (steps : Finset T)
    (G : ι → L → X) (c : ι → T → L) :
    ((rows ×ˢ steps).image fun it => G it.1 (c it.1 it.2)).card ≤ ∑ i ∈ rows, (steps.image (c i)).card := by
  refine (card_rows_le rows steps fun i t => G i (c i t)).trans (sum_le_sum fun i _ => ?_)
  calc (steps.image fun t => G i (c i t)).card = ((steps.image (c i)).image (G i)).card := by
        rw [image_image]; rfl
    _ ≤ _ := card_image_le

/-- **A register's lanes hold at most its values**: over any steps, the lanes' distinct values sum to at most the
number of lanes times the register's distinct values. -/
theorem card_lanes_le {Λ T : Type} [Fintype Λ] [DecidableEq V] (steps : Finset T) (R : T → Λ → V) :
    ∑ ℓ : Λ, (steps.image fun t => R t ℓ).card ≤ Fintype.card Λ * (steps.image R).card := by
  calc ∑ ℓ : Λ, (steps.image fun t => R t ℓ).card = ∑ ℓ : Λ, ((steps.image R).image fun r => r ℓ).card :=
        sum_congr rfl fun ℓ _ => by rw [image_image]; rfl
    _ ≤ ∑ _ℓ : Λ, (steps.image R).card := sum_le_sum fun _ _ => card_image_le
    _ = _ := by rw [sum_const, card_univ, smul_eq_mul]

/-- **A sequence that repeats at every step whose bit is `false` takes at most `1 + #true bits` values.** -/
theorem card_image_le_ones {X : Type} [DecidableEq X] (f : ℕ → X) (b : ℕ → Bool) (k : ℕ)
    (hf : ∀ t < k, b t = false → f (t + 1) = f t) :
    ((range (k + 1)).image f).card ≤ 1 + ((range k).filter fun t => b t).card := by
  induction k with
  | zero => simp
  | succ k ih =>
    have ih' := ih fun t ht => hf t (by omega)
    rw [range_add_one (n := k + 1), image_insert]
    cases hb : b k with
    | true =>
      have hf' : ((range (k + 1)).filter fun t => b t) = insert k ((range k).filter fun t => b t) := by
        rw [range_add_one, filter_insert]; simp [hb]
      have hk : k ∉ (range k).filter fun t => b t := by simp
      rw [hf', card_insert_of_notMem hk]
      calc _ ≤ ((range (k + 1)).image f).card + 1 := card_insert_le _ _
        _ ≤ _ := by omega
    | false =>
      have hm : f k ∈ (range (k + 1)).image f := mem_image_of_mem f (mem_range.2 (Nat.lt_succ_self k))
      rw [hf k (Nat.lt_succ_self k) hb, insert_eq_of_mem hm]
      exact ih'.trans (by gcongr; omega)

theorem pathLen_le [DecidableEq V] (x : Fin w → V) (ys : List (Fin w → V)) : pathLen x ys ≤ w * ys.length := by
  induction ys generalizing x with
  | nil => simp [pathLen]
  | cons y ys ih =>
    have h : hammingDist x y ≤ w := (hammingDist_le_card_fintype).trans_eq (Fintype.card_fin w)
    have := ih y
    simp only [pathLen, List.length_cons, Nat.mul_succ]
    omega

variable [DecidableEq A]

open Classical in
/-- **A run writes at least its targets' new chunk values**, whatever the program. -/
theorem run_chunks_le [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w))
    (Y : Finset (Fin w → V)) (hY : ∀ y ∈ Y, ∃ i a, run m₀ (P.take i) a = y) :
    (Y.filter fun y => ∀ a, m₀ a ≠ y).card ≤ writes P := by
  obtain ⟨L, -, hcov⟩ := good_prefix m₀ P
  calc (Y.filter fun y => ∀ a, m₀ a ≠ y).card ≤ ((range (writes P)).image L).card := by
        apply card_le_card
        intro y hy
        rw [mem_filter] at hy
        obtain ⟨i, a, hia⟩ := hY y hy.1
        rcases hcov i a with ⟨b, hb⟩ | ⟨k, hk, hkL⟩
        · exact absurd (hb.trans hia) (hy.2 b)
        · exact mem_image.2 ⟨k, mem_range.2 hk, hkL.trans hia⟩
    _ ≤ (range (writes P)).card := card_image_le
    _ = writes P := card_range _

open Classical in
/-- **Branching on the coins, chunk by chunk**: summed over any outcomes, a program writes at least each outcome's new
target chunk values. -/
theorem adaptive_chunks_le [DecidableEq V] [Inhabited V] {Ω : Type} (O : Finset Ω) (m₀ : Ω → A → Fin w → V)
    (P : Ω → List (Step A V w)) (Y : Ω → Finset (Fin w → V))
    (hY : ∀ ω ∈ O, ∀ y ∈ Y ω, ∃ i a, run (m₀ ω) ((P ω).take i) a = y) :
    ∑ ω ∈ O, ((Y ω).filter fun y => ∀ a, m₀ ω a ≠ y).card ≤ ∑ ω ∈ O, writes (P ω) :=
  sum_le_sum fun ω hω => run_chunks_le (m₀ ω) (P ω) (Y ω) (hY ω hω)

/-- **Balls against any initial memory**: targets pairwise further apart than the sum of their radii, each radius at
most `R`, need `Σ_y (r y + 1)` writes less `R + 1` a slot. -/
theorem run_ball_table_le [Fintype A] [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w))
    (Y : Finset (Fin w → V)) (hY : ∀ y ∈ Y, ∃ i a, run m₀ (P.take i) a = y) (r : (Fin w → V) → ℕ) (R : ℕ)
    (hR : ∀ y ∈ Y, r y ≤ R) (hYY : ∀ y ∈ Y, ∀ y' ∈ Y, y ≠ y' → r y + r y' < hammingDist y y') :
    ∑ y ∈ Y, (r y + 1) ≤ writes P + Fintype.card A * (R + 1) := by
  classical
  let far : (Fin w → V) → Prop := fun y => ∀ a, r y < hammingDist (m₀ a) y
  have h1 : ∑ y ∈ Y.filter far, (r y + 1) ≤ writes P :=
    run_ball_le m₀ P _ (fun y hy => hY y (mem_filter.1 hy).1) r (fun y hy => (mem_filter.1 hy).2)
      fun y hy y' hy' => hYY y (mem_filter.1 hy).1 y' (mem_filter.1 hy').1
  have hnear : ∀ y ∈ Y.filter (¬ far ·), ∃ a, hammingDist (m₀ a) y ≤ r y := fun y hy => by
    have := (mem_filter.1 hy).2
    simp only [far, not_forall, not_lt] at this
    exact this
  have hcard : (Y.filter (¬ far ·)).card ≤ Fintype.card A :=
    calc _ ≤ (univ.biUnion fun a => Y.filter fun y => hammingDist (m₀ a) y ≤ r y).card := by
          apply card_le_card
          intro y hy
          obtain ⟨a, ha⟩ := hnear y hy
          exact mem_biUnion.2 ⟨a, mem_univ _, mem_filter.2 ⟨(mem_filter.1 hy).1, ha⟩⟩
      _ ≤ ∑ a, (Y.filter fun y => hammingDist (m₀ a) y ≤ r y).card := card_biUnion_le
      _ ≤ ∑ _a : A, 1 := sum_le_sum fun a _ => card_le_one.2 fun y hy y' hy' => by
          by_contra hne
          rw [mem_filter] at hy hy'
          have := hammingDist_triangle y (m₀ a) y'
          rw [hammingDist_comm y (m₀ a)] at this
          have := hYY y hy.1 y' hy'.1 hne
          omega
      _ = Fintype.card A := by simp
  have h2 : ∑ y ∈ Y.filter (¬ far ·), (r y + 1) ≤ Fintype.card A * (R + 1) :=
    calc _ ≤ ∑ _y ∈ Y.filter (¬ far ·), (R + 1) := sum_le_sum fun y hy => by
          have := hR y (mem_filter.1 hy).1
          omega
      _ = (Y.filter (¬ far ·)).card * (R + 1) := by rw [sum_const, smul_eq_mul]
      _ ≤ _ := Nat.mul_le_mul_right _ hcard
  calc ∑ y ∈ Y, (r y + 1) = ∑ y ∈ Y.filter far, (r y + 1) + ∑ y ∈ Y.filter (¬ far ·), (r y + 1) :=
        (sum_filter_add_sum_filter_not Y far _).symm
    _ ≤ _ := Nat.add_le_add h1 h2

/-- **Branching on the coins, ball by ball**: summed over any outcomes, `run_ball_table_le`'s count, less `R + 1` a
slot an outcome, whatever each outcome's initial memory holds. -/
theorem adaptive_ball_table_le [Fintype A] [DecidableEq V] [Inhabited V] {Ω : Type} (O : Finset Ω)
    (m₀ : Ω → A → Fin w → V) (P : Ω → List (Step A V w)) (Y : Ω → Finset (Fin w → V)) (r : Ω → (Fin w → V) → ℕ)
    (R : ℕ) (hY : ∀ ω ∈ O, ∀ y ∈ Y ω, ∃ i a, run (m₀ ω) ((P ω).take i) a = y)
    (hR : ∀ ω ∈ O, ∀ y ∈ Y ω, r ω y ≤ R)
    (hYY : ∀ ω ∈ O, ∀ y ∈ Y ω, ∀ y' ∈ Y ω, y ≠ y' → r ω y + r ω y' < hammingDist y y') :
    ∑ ω ∈ O, ∑ y ∈ Y ω, (r ω y + 1) ≤ ∑ ω ∈ O, writes (P ω) + O.card * (Fintype.card A * (R + 1)) := by
  have h := sum_le_sum fun ω hω =>
    run_ball_table_le (m₀ ω) (P ω) (Y ω) (hY ω hω) (r ω) R (hR ω hω) (hYY ω hω)
  rwa [sum_add_distrib, sum_const, smul_eq_mul] at h

open Classical in
/-- **Per-entry credit under whole-register writes** (model W). Read the chunk model at one word a chunk, the word a
whole register of lanes `Λ`, so that a write sets a register and costs `|Λ|` lane-writes. Then a program holding every
outcome's registers `R ω e t` pays, in lane-writes, at least the entries' distinct values (each lane of each group,
over the steps), less `|Λ|` for each register value a group shares with another group or the initial memory. -/
theorem adaptive_lanes_le [DecidableEq V] [Inhabited V] {Λ Ω E T : Type} [Fintype Λ] [DecidableEq Λ] (O : Finset Ω)
    (m₀ : Ω → A → Fin 1 → Λ → V) (P : Ω → List (Step A (Λ → V) 1)) (groups : Finset E) (steps : Finset T)
    (R : Ω → E → T → Λ → V)
    (hY : ∀ ω ∈ O, ∀ e ∈ groups, ∀ t ∈ steps, ∃ i a, run (m₀ ω) ((P ω).take i) a 0 = R ω e t) :
    ∑ ω ∈ O, ∑ e ∈ groups, ∑ ℓ : Λ, (steps.image fun t => R ω e t ℓ).card +
        Fintype.card Λ * ∑ ω ∈ O,
          (((groups ×ˢ steps).image fun et => R ω et.1 et.2).filter fun r => ∀ a, m₀ ω a 0 ≠ r).card ≤
      Fintype.card Λ * ∑ ω ∈ O, writes (P ω) +
        Fintype.card Λ * ∑ ω ∈ O, ∑ e ∈ groups, (steps.image (R ω e)).card := by
  have h1 : ∑ ω ∈ O, (((groups ×ˢ steps).image fun et => R ω et.1 et.2).filter fun r => ∀ a, m₀ ω a 0 ≠ r).card ≤
      ∑ ω ∈ O, writes (P ω) := sum_le_sum fun ω hω => run_values_le (m₀ ω) (P ω) _ (fun r hr => by
        obtain ⟨⟨e, t⟩, het, rfl⟩ := mem_image.1 (mem_filter.1 hr).1
        rw [mem_product] at het
        exact hY ω hω e het.1 t het.2) fun r hr => (mem_filter.1 hr).2
  have h2 : ∑ ω ∈ O, ∑ e ∈ groups, ∑ ℓ : Λ, (steps.image fun t => R ω e t ℓ).card ≤
      Fintype.card Λ * ∑ ω ∈ O, ∑ e ∈ groups, (steps.image (R ω e)).card := by
    rw [mul_sum]
    exact sum_le_sum fun ω _ => by rw [mul_sum]; exact sum_le_sum fun e _ => card_lanes_le steps (R ω e)
  calc _ ≤ Fintype.card Λ * ∑ ω ∈ O, ∑ e ∈ groups, (steps.image (R ω e)).card +
        Fintype.card Λ * ∑ ω ∈ O, writes (P ω) := add_le_add h2 (Nat.mul_le_mul_left _ h1)
    _ = _ := add_comm _ _

/-- **A run writes at least the new (position, value) pairs it holds**, each at some slot after some prefix. -/
theorem run_words_le [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w))
    (Z : Finset (Fin w × V)) (hZ : ∀ pv ∈ Z, ∃ i a, run m₀ (P.take i) a pv.1 = pv.2)
    (hnew : ∀ pv ∈ Z, ∀ a, m₀ a pv.1 ≠ pv.2) : Z.card ≤ writes P := by
  classical
  choose i a hia using hZ
  refine le_trans (card_le_card fun pv hpv => ?_) (run_pairs_le m₀ P
    (Z.attach.image fun pv => run m₀ (P.take (i pv.1 pv.2)) (a pv.1 pv.2)) fun y hy => ?_)
  · exact mem_filter.2 ⟨mem_biUnion.2 ⟨_, mem_image.2 ⟨⟨pv, hpv⟩, mem_attach _ _, rfl⟩,
      mem_pairs.2 (hia pv hpv)⟩, hnew pv hpv⟩
  · obtain ⟨pv, -, rfl⟩ := mem_image.1 hy
    exact ⟨_, _, rfl⟩

open Classical in
/-- **Per-entry credit at word writes** (models P and C), check (b)'s obligation. Entry `e` sits at position `pos e`
of a chunk (`w = 1`: copies move one word, free; `w = 4`: copies move four), and every outcome's program holds each of
its values `R ω e t` there at some point. When no two entries at one position share a value and no initial slot
holds an entry's value at its position, the entries' distinct values over the steps, summed, are at most the
writes. Both hypotheses are what a program's chance or chosen coincidences break: a value another entry or the
initial memory holds is a copy, not a write. -/
theorem adaptive_entries_le [DecidableEq V] [Inhabited V] {Ω ι T : Type} (O : Finset Ω) (m₀ : Ω → A → Fin w → V)
    (P : Ω → List (Step A V w)) (entries : Finset ι) (steps : Finset T) (pos : ι → Fin w) (R : Ω → ι → T → V)
    (hY : ∀ ω ∈ O, ∀ e ∈ entries, ∀ t ∈ steps, ∃ i a, run (m₀ ω) ((P ω).take i) a (pos e) = R ω e t)
    (hdist : ∀ ω ∈ O, ∀ e ∈ entries, ∀ e' ∈ entries, ∀ t ∈ steps, ∀ t' ∈ steps,
      pos e = pos e' → R ω e t = R ω e' t' → e = e')
    (hnew : ∀ ω ∈ O, ∀ e ∈ entries, ∀ t ∈ steps, ∀ a, m₀ ω a (pos e) ≠ R ω e t) :
    ∑ ω ∈ O, ∑ e ∈ entries, (steps.image (R ω e)).card ≤ ∑ ω ∈ O, writes (P ω) := by
  refine sum_le_sum fun ω hω => ?_
  let f : ι → Finset (Fin w × V) := fun e => steps.image fun t => (pos e, R ω e t)
  have hf : ∀ e, (f e).card = (steps.image (R ω e)).card := fun e =>
    calc (f e).card = ((steps.image (R ω e)).image fun v => (pos e, v)).card := by rw [image_image]; rfl
      _ = _ := card_image_of_injective _ fun v v' h => (Prod.ext_iff.1 h).2
  have hdisj : (entries : Set ι).PairwiseDisjoint f := fun e he e' he' hne =>
    disjoint_left.2 fun pv h1 h2 => by
      obtain ⟨t, ht, rfl⟩ := mem_image.1 h1
      obtain ⟨t', ht', h⟩ := mem_image.1 h2
      exact hne (hdist ω hω e he e' he' t ht t' ht' (Prod.ext_iff.1 h).1.symm (Prod.ext_iff.1 h).2.symm)
  calc ∑ e ∈ entries, (steps.image (R ω e)).card = ∑ e ∈ entries, (f e).card := sum_congr rfl fun e _ => (hf e).symm
    _ = (entries.biUnion f).card := (card_biUnion hdisj).symm
    _ ≤ writes (P ω) := run_words_le (m₀ ω) (P ω) _ (fun pv hpv => by
          obtain ⟨e, he, hpv⟩ := mem_biUnion.1 hpv
          obtain ⟨t, ht, rfl⟩ := mem_image.1 hpv
          exact hY ω hω e he t ht) fun pv hpv => by
          obtain ⟨e, he, hpv⟩ := mem_biUnion.1 hpv
          obtain ⟨t, ht, rfl⟩ := mem_image.1 hpv
          exact hnew ω hω e he t ht

/-- **Per-entry credit under whole-register writes, given distinct registers** (model W): when no two groups share a
register value and no initial slot holds one, the lanes' distinct values over the steps are at most `|Λ|` a write.
The hypotheses are `adaptive_entries_le`'s at the grain of a whole register, which chance rarely breaks. -/
theorem adaptive_registers_le [DecidableEq V] [Inhabited V] {Λ Ω E T : Type} [Fintype Λ] [DecidableEq Λ]
    (O : Finset Ω) (m₀ : Ω → A → Fin 1 → Λ → V) (P : Ω → List (Step A (Λ → V) 1)) (groups : Finset E)
    (steps : Finset T) (R : Ω → E → T → Λ → V)
    (hY : ∀ ω ∈ O, ∀ e ∈ groups, ∀ t ∈ steps, ∃ i a, run (m₀ ω) ((P ω).take i) a 0 = R ω e t)
    (hdist : ∀ ω ∈ O, ∀ e ∈ groups, ∀ e' ∈ groups, ∀ t ∈ steps, ∀ t' ∈ steps, R ω e t = R ω e' t' → e = e')
    (hnew : ∀ ω ∈ O, ∀ e ∈ groups, ∀ t ∈ steps, ∀ a, m₀ ω a 0 ≠ R ω e t) :
    ∑ ω ∈ O, ∑ e ∈ groups, ∑ ℓ : Λ, (steps.image fun t => R ω e t ℓ).card ≤
      Fintype.card Λ * ∑ ω ∈ O, writes (P ω) := by
  calc _ ≤ Fintype.card Λ * ∑ ω ∈ O, ∑ e ∈ groups, (steps.image (R ω e)).card := by
        rw [mul_sum]
        exact sum_le_sum fun ω _ => by rw [mul_sum]; exact sum_le_sum fun e _ => card_lanes_le steps (R ω e)
    _ ≤ _ := Nat.mul_le_mul_left _ (adaptive_entries_le O m₀ P groups steps (fun _ => 0) R hY
          (fun ω hω e he e' he' t ht t' ht' _ h => hdist ω hω e he e' he' t ht t' ht' h) hnew)

/-- **Any targets are held at `w` writes a distinct value**: the path run through them in one slot. -/
theorem pathRun_le_card [DecidableEq V] (m₀ : A → Fin w → V) (a : A) (Y : Finset (Fin w → V)) :
    (∀ y ∈ Y, ∃ i, run m₀ ((pathRun a (m₀ a) Y.toList).take i) a = y) ∧
      writes (pathRun a (m₀ a) Y.toList) ≤ w * Y.card := by
  obtain ⟨h1, h2⟩ := pathRun_attained m₀ a Y.toList
  exact ⟨fun y hy => h1 y (mem_toList.2 hy), h2 ▸ (pathLen_le _ _).trans_eq (by rw [length_toList])⟩

/-- **The attained side over the outcomes**: path runs hold every outcome's targets at `w` writes a distinct value. -/
theorem adaptive_attained [DecidableEq V] {Ω : Type} (O : Finset Ω) (m₀ : Ω → A → Fin w → V) (a : A)
    (Y : Ω → Finset (Fin w → V)) :
    ∃ P : Ω → List (Step A V w), (∀ ω ∈ O, ∀ y ∈ Y ω, ∃ i a, run (m₀ ω) ((P ω).take i) a = y) ∧
      ∑ ω ∈ O, writes (P ω) ≤ w * ∑ ω ∈ O, (Y ω).card := by
  refine ⟨fun ω => pathRun a (m₀ ω a) (Y ω).toList, fun ω _ y hy => ?_, ?_⟩
  · obtain ⟨i, hi⟩ := (pathRun_le_card (m₀ ω) a (Y ω)).1 y hy
    exact ⟨i, a, hi⟩
  · rw [mul_sum]
    exact sum_le_sum fun ω _ => (pathRun_le_card (m₀ ω) a (Y ω)).2

/-- **The walk's obstruction**: when every outcome's targets are row-step chunks `G ω i (c ω i t)`, a function of the
row and its level, some program holds them all at `w·Σ_ω Σ_i |{c ω i t}|` writes. -/
theorem adaptive_levels_attained [DecidableEq V] {Ω ι T L : Type} [DecidableEq L] (O : Finset Ω)
    (m₀ : Ω → A → Fin w → V) (a : A) (rows : Finset ι) (steps : Finset T) (G : Ω → ι → L → Fin w → V)
    (c : Ω → ι → T → L) :
    ∃ P : Ω → List (Step A V w),
      (∀ ω ∈ O, ∀ y ∈ (rows ×ˢ steps).image (fun it => G ω it.1 (c ω it.1 it.2)),
        ∃ i a, run (m₀ ω) ((P ω).take i) a = y) ∧
      ∑ ω ∈ O, writes (P ω) ≤ w * ∑ ω ∈ O, ∑ i ∈ rows, (steps.image (c ω i)).card := by
  obtain ⟨P, hP, hw⟩ := adaptive_attained O m₀ a fun ω => (rows ×ˢ steps).image fun it => G ω it.1 (c ω it.1 it.2)
  exact ⟨P, hP, hw.trans (Nat.mul_le_mul_left w (sum_le_sum fun ω _ => card_levels_le rows steps (G ω) (c ω)))⟩

/-- **The cancel's obstruction**: when every row's chunk at step `t + 1` repeats its chunk at step `t` whenever the
coin bit `b ω i t` is `false`, some program holds every outcome's row-step chunks at
`w·Σ_ω Σ_i (1 + #{t < k : b ω i t})` writes. -/
theorem adaptive_ones_attained [DecidableEq V] {Ω ι : Type} (O : Finset Ω) (m₀ : Ω → A → Fin w → V) (a : A)
    (rows : Finset ι) (k : ℕ) (F : Ω → ι → ℕ → Fin w → V) (b : Ω → ι → ℕ → Bool)
    (hF : ∀ ω i t, t < k → b ω i t = false → F ω i (t + 1) = F ω i t) :
    ∃ P : Ω → List (Step A V w),
      (∀ ω ∈ O, ∀ y ∈ (rows ×ˢ range (k + 1)).image (fun it => F ω it.1 it.2),
        ∃ i a, run (m₀ ω) ((P ω).take i) a = y) ∧
      ∑ ω ∈ O, writes (P ω) ≤ w * ∑ ω ∈ O, ∑ i ∈ rows, (1 + ((range k).filter fun t => b ω i t).card) := by
  obtain ⟨P, hP, hw⟩ := adaptive_attained O m₀ a fun ω => (rows ×ˢ range (k + 1)).image fun it => F ω it.1 it.2
  exact ⟨P, hP, hw.trans (Nat.mul_le_mul_left w (sum_le_sum fun ω _ => (card_rows_le rows _ (F ω)).trans
    (sum_le_sum fun i _ => card_image_le_ones (F ω i) (b ω i) k (hF ω i))))⟩

/-! ## Priced selection

A program is a tree of uniform branches (`Tree`) down to static streams. Every outcome of a class, the outcomes that
take one path, runs the same stream: instructions that each set one slot (a word) to a computed value or another slot's
value, optionally under the lane's own predicate, and hashes that read a slot as an entry's value. An instruction is
priced at every outcome, taken or not, since a predicated instruction issues either way: its selection `cs` if
predicated and, if taken, a write `cw` or a copy `cc`. -/

namespace Priced

variable {Ω E : Type}

/-- Slot `dst` gets `val ω`, or with `src = some b` slot `b`'s value; with `guarded`, only where `guard ω` holds. -/
structure Instr (Ω A V : Type) where
  dst : A
  src : Option A
  val : Ω → V
  guarded : Bool
  guard : Ω → Bool

/-- A stream's events: an instruction, or a hash of slot `a` as entry `e`, which must read `v ω`. -/
inductive Ev (Ω A V E : Type)
  | ins (i : Instr Ω A V)
  | hash (a : A) (v : Ω → V) (e : E)

def Instr.taken (i : Instr Ω A V) (ω : Ω) : Bool := !i.guarded || i.guard ω

def Instr.apply (i : Instr Ω A V) (ω : Ω) (m : A → V) : A → V :=
  if i.taken ω then Function.update m i.dst (i.src.elim (i.val ω) m) else m

/-- From memory `m` at outcome `ω`, every hash reads its entry's value. -/
def Correct (ω : Ω) : (A → V) → List (Ev Ω A V E) → Prop
  | _, [] => True
  | m, .ins i :: L => Correct ω (i.apply ω m) L
  | m, .hash a v _ :: L => m a = v ω ∧ Correct ω m L

def Instr.cost (cw cc cs : ℕ) (ω : Ω) (i : Instr Ω A V) : ℕ :=
  (if i.guarded then cs else 0) + if i.taken ω then (if i.src.isSome then cc else cw) else 0

def cost (cw cc cs : ℕ) (ω : Ω) : List (Ev Ω A V E) → ℕ
  | [] => 0
  | .ins i :: L => i.cost cw cc cs ω + cost cw cc cs ω L
  | .hash .. :: L => cost cw cc cs ω L

def insCount : List (Ev Ω A V E) → ℕ
  | [] => 0
  | .ins _ :: L => insCount L + 1
  | .hash .. :: L => insCount L

/-- The instructions priced below a write at `ω`. W1 as stated prices copies and untaken predicated writes at 0. -/
def freeCount (cw cc cs : ℕ) (ω : Ω) : List (Ev Ω A V E) → ℕ
  | [] => 0
  | .ins i :: L => (if i.cost cw cc cs ω < cw then 1 else 0) + freeCount cw cc cs ω L
  | .hash .. :: L => freeCount cw cc cs ω L

open Classical in
/-- The hashes that read, throughout class `C`, what their slot last held for another entry (`src a`, its last value
`last a`) or initially: the reads a class gets without an instruction. -/
noncomputable def shared (C : Set Ω) : (A → Ω → V) → (A → Option E) → List (Ev Ω A V E) → ℕ
  | _, _, [] => 0
  | last, src, .ins _ :: L => shared C last src L
  | last, src, .hash a v e :: L => (if (∀ ω ∈ C, v ω = last a ω) ∧ src a ≠ some e then 1 else 0) +
      shared C (Function.update last a v) (Function.update src a (some e)) L

/-- Check (b)'s credit at `ω`: the (entry, value) pairs the stream hashes. -/
def pairsAt [DecidableEq E] [DecidableEq V] (ω : Ω) : List (Ev Ω A V E) → Finset (E × V)
  | [] => ∅
  | .ins _ :: L => pairsAt ω L
  | .hash _ v e :: L => insert (e, v ω) (pairsAt ω L)

open Classical in
/-- The invariant behind `credit_le_ins`: a slot outside `D` still holds, throughout `C`, its last demanded value. -/
theorem pairs_le [DecidableEq E] [DecidableEq V] (C : Set Ω) (ω : Ω) (hω : ω ∈ C) :
    ∀ (L : List (Ev Ω A V E)) (m : Ω → A → V) (last : A → Ω → V) (src : A → Option E) (seen : Finset (E × V))
      (D : Finset A), (∀ a ∉ D, ∀ ω' ∈ C, m ω' a = last a ω') → (∀ ω' ∈ C, Correct ω' (m ω') L) →
      (∀ a e, src a = some e → (e, last a ω) ∈ seen) →
      (pairsAt ω L \ seen).card ≤ D.card + insCount L + shared C last src L
  | [], _, _, _, _, _, _, _, _ => by simp [pairsAt]
  | .ins i :: L, m, last, src, seen, D, hm, hc, hs => by
    have ih := pairs_le C ω hω L (fun ω' => i.apply ω' (m ω')) last src seen (insert i.dst D)
      (fun a ha ω' hω' => by
        rw [mem_insert, not_or] at ha
        have h : i.apply ω' (m ω') a = m ω' a := by
          unfold Instr.apply
          split_ifs
          · exact Function.update_of_ne ha.1 _ _
          · rfl
        rw [h]
        exact hm a ha.2 ω' hω') (fun ω' hω' => hc ω' hω') hs
    simp only [pairsAt, insCount, shared]
    have := card_insert_le i.dst D
    omega
  | .hash a v e :: L, m, last, src, seen, D, hm, hc, hs => by
    have hva : ∀ ω' ∈ C, m ω' a = v ω' := fun ω' hω' => (hc ω' hω').1
    have ih := pairs_le C ω hω L m (Function.update last a v) (Function.update src a (some e))
      (insert (e, v ω) seen) (D.erase a)
      (fun a' ha' ω' hω' => by
        by_cases h : a' = a
        · subst h
          rw [Function.update_self]
          exact hva ω' hω'
        · rw [Function.update_of_ne h]
          exact hm a' (fun hD => ha' (mem_erase.2 ⟨h, hD⟩)) ω' hω')
      (fun ω' hω' => (hc ω' hω').2)
      (fun a' e' h => by
        by_cases ha : a' = a
        · subst ha
          rw [Function.update_self, Option.some_inj] at h
          subst h
          rw [Function.update_self]
          exact mem_insert_self _ _
        · rw [Function.update_of_ne ha] at h ⊢
          exact mem_insert_of_mem (hs a' e' h))
    simp only [pairsAt, insCount, shared]
    have hsub : insert (e, v ω) (pairsAt ω L) \ seen ⊆ insert (e, v ω) (pairsAt ω L \ insert (e, v ω) seen) := by
      intro x hx
      rw [mem_sdiff, mem_insert] at hx
      rw [mem_insert, mem_sdiff, mem_insert]
      tauto
    have h1 := (card_le_card hsub).trans (card_insert_le _ _)
    by_cases hD : a ∈ D
    · have := card_erase_add_one hD
      omega
    · rw [erase_eq_of_notMem hD] at ih
      have hall : ∀ ω' ∈ C, v ω' = last a ω' := fun ω' hω' => (hva ω' hω').symm.trans (hm a hD ω' hω')
      by_cases hsrc : src a = some e
      · have hp : (e, v ω) ∈ seen := by
          have := hs a e hsrc
          rwa [← hall ω hω] at this
        have hsub' : insert (e, v ω) (pairsAt ω L) \ seen ⊆ pairsAt ω L \ insert (e, v ω) seen := by
          intro x hx
          rw [mem_sdiff, mem_insert] at hx
          rw [mem_sdiff, mem_insert]
          rcases hx with ⟨rfl | hx, hns⟩
          · exact absurd hp hns
          · exact ⟨hx, fun h => h.elim (fun h => hns (h ▸ hp)) hns⟩
        have := card_le_card hsub'
        omega
      · rw [ite_eq_left ⟨hall, hsrc⟩]
        omega

open Classical in
/-- **Credit within a class, unpriced**: at any outcome of a class that runs one stream, check (b)'s credit is at most
the stream's instructions plus its shared reads. -/
theorem credit_le_ins [DecidableEq E] [DecidableEq V] (C : Set Ω) (m₀ : Ω → A → V) (L : List (Ev Ω A V E))
    (hc : ∀ ω ∈ C, Correct ω (m₀ ω) L) (ω : Ω) (hω : ω ∈ C) :
    (pairsAt ω L).card ≤ insCount L + shared C (fun a ω => m₀ ω a) (fun _ => none) L := by
  simpa using pairs_le C ω hω L m₀ (fun a ω => m₀ ω a) (fun _ => none) ∅ ∅ (fun _ _ _ _ => rfl) hc
    (fun _ _ h => by cases h)

omit [DecidableEq A] in
theorem ins_le_cost (cw cc cs : ℕ) (ω : Ω) :
    ∀ L : List (Ev Ω A V E), cw * insCount L ≤ cost cw cc cs ω L + cw * freeCount cw cc cs ω L
  | [] => by simp [insCount, cost, freeCount]
  | .ins i :: L => by
    have ih := ins_le_cost cw cc cs ω L
    simp only [insCount, cost, freeCount, mul_add, mul_one]
    split_ifs with h
    · rw [mul_one]
      omega
    · rw [mul_zero]
      omega
  | .hash .. :: L => by simpa [insCount, cost, freeCount] using ins_le_cost cw cc cs ω L

omit [DecidableEq A] in
/-- **Select-floor and copy-floor leave nothing free**: when a selection on the lane's own bit and a copy each cost
at least a write, every instruction costs at least a write at every outcome. -/
theorem Instr.cw_le_cost {cw cc cs : ℕ} (hcc : cw ≤ cc) (hcs : cw ≤ cs) (ω : Ω) (i : Instr Ω A V) :
    cw ≤ i.cost cw cc cs ω := by
  rcases i with ⟨dst, src, val, g, gd⟩
  cases g <;> cases h : gd ω <;> cases src <;> simp [Instr.cost, Instr.taken, h] <;> omega

omit [DecidableEq A] in
theorem freeCount_eq_zero {cw cc cs : ℕ} (hcc : cw ≤ cc) (hcs : cw ≤ cs) (ω : Ω) :
    ∀ L : List (Ev Ω A V E), freeCount cw cc cs ω L = 0
  | [] => rfl
  | .ins i :: L => by
    simp only [freeCount, freeCount_eq_zero hcc hcs ω L, ite_eq_right (Nat.not_lt.2 (i.cw_le_cost hcc hcs ω))]
  | .hash .. :: L => by simp only [freeCount, freeCount_eq_zero hcc hcs ω L]

open Classical in
/-- **Credit within a class, priced, with its exact residual**: check (b)'s credit at any outcome of a class, in
writes, is at most the stream's cost there, plus a write for each instruction priced below a write there and for each
read the class shares across entries. -/
theorem credit_le_cost [DecidableEq E] [DecidableEq V] (cw cc cs : ℕ) (C : Set Ω) (m₀ : Ω → A → V)
    (L : List (Ev Ω A V E)) (hc : ∀ ω ∈ C, Correct ω (m₀ ω) L) (ω : Ω) (hω : ω ∈ C) :
    cw * (pairsAt ω L).card ≤
      cost cw cc cs ω L + cw * (freeCount cw cc cs ω L + shared C (fun a ω => m₀ ω a) (fun _ => none) L) := by
  have h1 := Nat.mul_le_mul_left cw (credit_le_ins C m₀ L hc ω hω)
  have h2 := ins_le_cost cw cc cs ω L
  rw [mul_add] at h1 ⊢
  omega

open Classical in
/-- **Priced selection** (P with select-floor and copy-floor): the residual is only the class's shared reads. -/
theorem credit_le_cost_floor [DecidableEq E] [DecidableEq V] {cw cc cs : ℕ} (hcc : cw ≤ cc) (hcs : cw ≤ cs)
    (C : Set Ω) (m₀ : Ω → A → V) (L : List (Ev Ω A V E)) (hc : ∀ ω ∈ C, Correct ω (m₀ ω) L) (ω : Ω) (hω : ω ∈ C) :
    cw * (pairsAt ω L).card ≤ cost cw cc cs ω L + cw * shared C (fun a ω => m₀ ω a) (fun _ => none) L := by
  simpa [freeCount_eq_zero hcc hcs ω L] using credit_le_cost cw cc cs C m₀ L hc ω hω

open Classical in
/-- **No shared reads when the class separates the entries**: if, throughout `C`, any two entries' demands at one slot
differ somewhere and no demand is its slot's prior content everywhere, nothing is shared. -/
theorem shared_eq_zero (C : Set Ω) : ∀ (L : List (Ev Ω A V E)) (last : A → Ω → V) (src : A → Option E),
    (∀ a v e, Ev.hash a v e ∈ L → ∀ v' e', Ev.hash a v' e' ∈ L → e ≠ e' → ∃ ω ∈ C, v ω ≠ v' ω) →
    (∀ a v e, Ev.hash a v e ∈ L → src a ≠ some e → ∃ ω ∈ C, v ω ≠ last a ω) →
    shared C last src L = 0
  | [], _, _, _, _ => rfl
  | .ins _ :: L, last, src, hsep, hinit => by
    simp only [shared]
    exact shared_eq_zero C L last src
      (fun a v e h v' e' h' => hsep a v e (List.mem_cons_of_mem _ h) v' e' (List.mem_cons_of_mem _ h'))
      fun a v e h => hinit a v e (List.mem_cons_of_mem _ h)
  | .hash a v e :: L, last, src, hsep, hinit => by
    have hind : ¬ ((∀ ω ∈ C, v ω = last a ω) ∧ src a ≠ some e) := fun ⟨hall, hs⟩ => by
      obtain ⟨ω, hω, hne⟩ := hinit a v e List.mem_cons_self hs
      exact hne (hall ω hω)
    simp only [shared, ite_eq_right hind, zero_add]
    refine shared_eq_zero C L _ _
      (fun a v e h v' e' h' => hsep a v e (List.mem_cons_of_mem _ h) v' e' (List.mem_cons_of_mem _ h'))
      fun a' v' e' h hs' => ?_
    by_cases ha : a' = a
    · subst ha
      rw [Function.update_self] at hs' ⊢
      have hee : e ≠ e' := fun hee => hs' (hee ▸ rfl)
      obtain ⟨ω, hω, hne⟩ := hsep a' v e List.mem_cons_self v' e' (List.mem_cons_of_mem _ h) hee
      exact ⟨ω, hω, Ne.symm hne⟩
    · rw [Function.update_of_ne ha] at hs' ⊢
      exact hinit a' v' e' (List.mem_cons_of_mem _ h) hs'

open Classical in
/-- **The masked cross-row channel closes on a separating class**: under select-floor and copy-floor, when the
class separates the entries (`shared_eq_zero`'s hypotheses from the initial memory), check (b)'s credit at any of its
outcomes, in writes, is at most the stream's cost there. -/
theorem credit_le_cost_separated [DecidableEq E] [DecidableEq V] {cw cc cs : ℕ} (hcc : cw ≤ cc) (hcs : cw ≤ cs)
    (C : Set Ω) (m₀ : Ω → A → V) (L : List (Ev Ω A V E)) (hc : ∀ ω ∈ C, Correct ω (m₀ ω) L)
    (hsep : ∀ a v e, Ev.hash a v e ∈ L → ∀ v' e', Ev.hash a v' e' ∈ L → e ≠ e' → ∃ ω ∈ C, v ω ≠ v' ω)
    (hinit : ∀ a v e, Ev.hash a v e ∈ L → ∃ ω ∈ C, v ω ≠ m₀ ω a) (ω : Ω) (hω : ω ∈ C) :
    cw * (pairsAt ω L).card ≤ cost cw cc cs ω L := by
  have h := credit_le_cost_floor hcc hcs C m₀ L hc ω hω
  rwa [shared_eq_zero C L _ _ hsep (fun a v e h _ => hinit a v e h), mul_zero, add_zero] at h

omit [DecidableEq A] in
/-- **A register's lanes hold at most its values, per entry**: with a slot a register of lanes `Λ`, the lanes'
(entry, value) pairs number at most `|Λ|` times the register pairs. -/
theorem lanes_le_pairs {Λ W : Type} [Fintype Λ] [DecidableEq E] [DecidableEq W] (ω : Ω)
    (L : List (Ev Ω A (Λ → W) E)) :
    ∑ ℓ : Λ, ((pairsAt ω L).image fun p => (p.1, p.2 ℓ)).card ≤ Fintype.card Λ * (pairsAt ω L).card :=
  (sum_le_sum fun _ _ => card_image_le).trans_eq (by rw [sum_const, card_univ, smul_eq_mul])

open Classical in
/-- **At register grain the channel closes for any class**: price whole-register instructions (every instruction
issues for all its lanes, predicated or not, so a predicated write or a move costs at least a register write) and
read check (b)'s per-lane credit. When the class separates the groups' register values (two groups' registers at one
slot differ somewhere in it, and none is its slot's prior content everywhere), the lanes' credit in register writes is
at most `|Λ|` times the cost. Separation at register grain holds for every class that contains an outcome at which
no two groups share a whole register value, the outcome itself included. -/
theorem lanes_le_cost_separated {Λ W : Type} [Fintype Λ] [DecidableEq E] [DecidableEq W] {cw cc cs : ℕ}
    (hcc : cw ≤ cc) (hcs : cw ≤ cs) (C : Set Ω) (m₀ : Ω → A → Λ → W) (L : List (Ev Ω A (Λ → W) E))
    (hc : ∀ ω ∈ C, Correct ω (m₀ ω) L)
    (hsep : ∀ a v e, Ev.hash a v e ∈ L → ∀ v' e', Ev.hash a v' e' ∈ L → e ≠ e' → ∃ ω ∈ C, v ω ≠ v' ω)
    (hinit : ∀ a v e, Ev.hash a v e ∈ L → ∃ ω ∈ C, v ω ≠ m₀ ω a) (ω : Ω) (hω : ω ∈ C) :
    cw * ∑ ℓ : Λ, ((pairsAt ω L).image fun p => (p.1, p.2 ℓ)).card ≤ Fintype.card Λ * cost cw cc cs ω L :=
  calc cw * ∑ ℓ : Λ, ((pairsAt ω L).image fun p => (p.1, p.2 ℓ)).card
      ≤ cw * (Fintype.card Λ * (pairsAt ω L).card) := Nat.mul_le_mul_left cw (lanes_le_pairs ω L)
    _ = Fintype.card Λ * (cw * (pairsAt ω L).card) := by ring
    _ ≤ _ := Nat.mul_le_mul_left _ (credit_le_cost_separated hcc hcs C m₀ L hc hsep hinit ω hω)

/-- A branching tree: uniform branches, each priced `cb`, down to static streams. -/
inductive Tree (Ω A V E : Type)
  | leaf (L : List (Ev Ω A V E))
  | branch (d : Ω → Bool) (p q : Tree Ω A V E)

def Tree.stream : Tree Ω A V E → Ω → List (Ev Ω A V E)
  | .leaf L, _ => L
  | .branch d p q, ω => if d ω then p.stream ω else q.stream ω

def Tree.depth : Tree Ω A V E → Ω → ℕ
  | .leaf _, _ => 0
  | .branch d p q, ω => 1 + if d ω then p.depth ω else q.depth ω

/-- `ω`'s class: the outcomes that take its path. -/
def Tree.cls : Tree Ω A V E → Ω → Set Ω
  | .leaf _, _ => Set.univ
  | .branch d p q, ω => {ω' | d ω' = d ω} ∩ if d ω then p.cls ω else q.cls ω

omit [DecidableEq A] in
theorem Tree.mem_cls : ∀ (P : Tree Ω A V E) (ω : Ω), ω ∈ P.cls ω
  | .leaf _, _ => Set.mem_univ _
  | .branch d p q, ω => by
    refine ⟨rfl, ?_⟩
    cases d ω
    · exact q.mem_cls ω
    · exact p.mem_cls ω

omit [DecidableEq A] in
theorem Tree.stream_eq : ∀ (P : Tree Ω A V E) (ω ω' : Ω), ω' ∈ P.cls ω → P.stream ω' = P.stream ω
  | .leaf _, _, _, _ => rfl
  | .branch d p q, ω, ω', ⟨hd, h⟩ => by
    have hd : d ω' = d ω := hd
    simp only [Tree.stream, hd]
    cases hdω : d ω <;> simp only [hdω] at h ⊢
    · exact q.stream_eq ω ω' h
    · exact p.stream_eq ω ω' h

open Classical in
/-- **A program that branches uniformly on the coins**: at every outcome, check (b)'s credit in writes is at most what
the program costs there (its branches and its stream), plus a write for each instruction priced below a write and for
each read its class shares across entries. A class can share a read only if its branches pin it. -/
theorem prog_credit_le [DecidableEq E] [DecidableEq V] (cb cw cc cs : ℕ) (P : Tree Ω A V E) (m₀ : Ω → A → V)
    (hc : ∀ ω, Correct ω (m₀ ω) (P.stream ω)) (ω : Ω) :
    cw * (pairsAt ω (P.stream ω)).card ≤ cb * P.depth ω + cost cw cc cs ω (P.stream ω) +
      cw * (freeCount cw cc cs ω (P.stream ω) + shared (P.cls ω) (fun a ω => m₀ ω a) (fun _ => none) (P.stream ω)) := by
  have h := credit_le_cost cw cc cs (P.cls ω) m₀ (P.stream ω)
    (fun ω' hω' => P.stream_eq ω ω' hω' ▸ hc ω') ω (P.mem_cls ω)
  omega

/-- The decisions `ω`'s path takes. -/
def Tree.path : Tree Ω A V E → Ω → List Bool
  | .leaf _, _ => []
  | .branch d p q, ω => d ω :: if d ω then p.path ω else q.path ω

omit [DecidableEq A] in
/-- A class is the outcomes that take its path. -/
theorem Tree.mem_cls_iff : ∀ (P : Tree Ω A V E) (ω ω' : Ω), ω' ∈ P.cls ω ↔ P.path ω' = P.path ω
  | .leaf _, _, _ => by simp [Tree.cls, Tree.path]
  | .branch d p q, ω, ω' => by
    have hp := p.mem_cls_iff ω ω'
    have hq := q.mem_cls_iff ω ω'
    simp only [Tree.cls, Tree.path, Set.mem_inter_iff, Set.mem_ofPred_eq]
    cases hω : d ω <;> cases hω' : d ω' <;> simp [hp, hq]

omit [DecidableEq A] in
/-- **Few outcomes sit in small classes**: the outcomes of `O` whose class (its path's outcomes, `Tree.mem_cls_iff`)
holds fewer than `c` of `O`'s number at most `c` a path taken. So a premise stated on every path's share of the coins,
for each share of at least `c` outcomes, holds on all of `O` but at most `c` times the paths, `2^p` of them for a
program `p` branches deep. -/
theorem small_cls_le (P : Tree Ω A V E) (O : Finset Ω) (c : ℕ) :
    (O.filter fun ω => (O.filter fun ω' => P.path ω' = P.path ω).card < c).card ≤ c * (O.image P.path).card := by
  set s := O.filter fun ω => (O.filter fun ω' => P.path ω' = P.path ω).card < c
  calc s.card ≤ c * (s.image P.path).card := card_le_mul_card_image s c fun κ hκ => by
        obtain ⟨ω, hω, rfl⟩ := mem_image.1 hκ
        exact le_trans (card_le_card fun x hx => by
          rw [mem_filter] at hx ⊢
          exact ⟨(mem_filter.1 hx.1).1, hx.2⟩) (mem_filter.1 hω).2.le
    _ ≤ c * (O.image P.path).card := Nat.mul_le_mul_left _ (card_le_card (image_subset_image (filter_subset _ _)))

end Priced

end Pouw.SecurityProofs.Lead.CoinBranch
