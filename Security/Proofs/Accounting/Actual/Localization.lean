import Models.Accounting.Guarantees.Localization
import Definitions.Accounting.Localization
import Definitions.Accounting.Physical
import Proofs.Accounting.Actual.Split
import Proofs.Accounting.Actual.Confidence
import Mathlib.Tactic.FinCases

/-!
# Localization, proved

`Localization`, over strategies, is function composition. What the rest reads from outside before `u ≤ t₁`
(`crossing_split`) is what it read before `t₀` (part of what it can tell at `t₀`), what `X` sent it during `[t₀, u)`,
and what it heard from the leaves in neither during `W`, cut at `u`. With its inputs fixed, causality's global form
(`global_of_causal`) makes what the rest does before `u` a function of what `X` sent. The table picks a run for
each key, and the count is the image's. `memory_of_mem`: memory per run on the state bounds the circuit's `state`.
`markov_mem`: memory accounting's `not_markov` circuit (a gate of `X` reading nothing, with the coin's value) has
`mem X 2 ≥ 1` at some coin in every strategy causal from every tick, whatever it holds (`markov_holding`: it can).
`zero_link_alphabet`: memory accounting's `zero_link` (a width-0 wire sent at the coin's tick) meets `Links` at
capacity 0 (`zero_links`) and fails `Alphabet` there. `Recall` is `CausalFrom` read through a table, as `Localization`
is, with the content's length bound (`content_length_le`).

`localization_wires` is the per-run lemma (not the model's `e = 0`, which reads the strategy): under causality a gate
downstream of `C` runs at or after `C`'s ticks (`le_t_of_downstream`), so one that `A[Y, W]` reads runs during `W`; and
one not in `C` reads a gate downstream of `C`, so it is computed (`downstream_pred`). Placed in `Y` it would be in
`A[Y, W]`, and placed at the verifier's `Reaches` refuses it, so it is in `A[X, W]`. When no wire from `X` to `Y`
carries a gate downstream of `C`, a gate of `A[Y, W]` downstream of `C` reads one, which is then in `A[Y, W]` and
earlier (the gates are in topological order), so none is.
-/

namespace Accounting.SecurityProofs

open Finset

namespace Actual

variable {L : Type*} (A : Placed L)

theorem mem_downstream {C : Finset (Fin A.N)} {g : Fin A.N} :
    g ∈ A.downstream C ↔ ∃ c ∈ C, Relation.ReflTransGen (fun a g => a ∈ A.reads g) c g := by
  simp [Placed.downstream]

/-- A gate downstream of `C` that is not in `C` is computed and reads a gate downstream of `C`. -/
theorem downstream_pred {C : Finset (Fin A.N)} {g : Fin A.N} (hg : g ∈ A.downstream C) (hC : g ∉ C) :
    A.isInput g = false ∧ ∃ a ∈ A.reads g, a ∈ A.downstream C := by
  obtain ⟨c, hc, h⟩ := (mem_downstream A).1 hg
  rcases h.cases_tail with rfl | ⟨a, ha, hag⟩
  · exact absurd hc hC
  refine ⟨?_, a, hag, (mem_downstream A).2 ⟨c, hc, ha⟩⟩
  cases hi : A.isInput g
  · rfl
  · simp [A.input_reads g hi] at hag

/-- Under causality a gate downstream of `C` runs no earlier than a tick `t₀` at most each of `C`'s. -/
theorem le_t_of_downstream {ℓ : L → L → ℕ} (hc : A.Causal ℓ) {C : Finset (Fin A.N)} {t₀ : ℕ}
    (h₀ : ∀ c ∈ C, t₀ ≤ A.t c) {g : Fin A.N} (hg : g ∈ A.downstream C) : t₀ ≤ A.t g := by
  obtain ⟨c, hcC, h⟩ := (mem_downstream A).1 hg
  clear hg
  induction h with
  | refl => exact h₀ c hcC
  | tail _ hab ih => exact ih.trans ((Nat.le_add_right _ _).trans (hc _ _ hab))

/-- **Localization on one circuit's wires.** Under causality, for a set `C` of gates, a location `X` it reaches the
prover through during `W = [t₀, t₁)` (`t₀` at most each tick of `C`) and the rest `Y` of the prover: every gate
downstream of `C` that `A[Y, W]` reads from outside itself is in `A[X, W]`; and when no gate of `C` is placed in `Y` and
no wire from `X` to `Y` during `W` carries a gate downstream of `C`, no gate of `A[Y, W]` is downstream of `C`, nor is
any gate it reads from outside itself, nor any gate placed in `Y` that `A[X, W]` reads. -/
theorem localization_wires {ℓ : L → L → ℕ} (hc : A.Causal ℓ) {C : Finset (Fin A.N)} {X Y : Finset L} {t₀ t₁ : ℕ}
    (h₀ : ∀ c ∈ C, t₀ ≤ A.t c) (hR : A.Reaches C X Y (Set.Ico t₀ t₁)) :
    (∀ a ∈ A.into (A.region Y (Set.Ico t₀ t₁)), a ∈ A.downstream C → a ∈ A.region X (Set.Ico t₀ t₁)) ∧
    ((∀ c ∈ C, A.x c ∉ Y) → A.SendsIndependent C X Y (Set.Ico t₀ t₁) →
      Disjoint (A.region Y (Set.Ico t₀ t₁)) (A.downstream C) ∧
      Disjoint (A.into (A.region Y (Set.Ico t₀ t₁))) (A.downstream C) ∧
      ∀ g ∈ A.region X (Set.Ico t₀ t₁), ∀ a ∈ A.reads g, A.x a ∈ Y → a ∉ A.downstream C) := by
  have egress : ∀ a ∈ A.into (A.region Y (Set.Ico t₀ t₁)), a ∈ A.downstream C → a ∈ A.region X (Set.Ico t₀ t₁) := by
    intro a ha had
    obtain ⟨haY, g, hg, hag⟩ := (mem_into A).1 ha
    obtain ⟨haC, hver⟩ := hR g hg a hag
    have hai := (downstream_pred A had haC).1
    have hat : A.t a ∈ Set.Ico t₀ t₁ :=
      ⟨le_t_of_downstream A hc h₀ had,
        ((Nat.le_add_right _ _).trans (hc g a hag)).trans_lt ((mem_region A).1 hg).2.2.2⟩
    by_cases hx : A.x a ∈ X
    · exact (mem_region A).2 ⟨hai, hx, hat⟩
    by_cases hy : A.x a ∈ Y
    · exact absurd ((mem_region A).2 ⟨hai, hy, hat⟩) haY
    · exact absurd had (hver hx hy)
  refine ⟨egress, fun hCY hS => ?_⟩
  have region : ∀ g ∈ A.region Y (Set.Ico t₀ t₁), g ∉ A.downstream C := by
    intro g
    induction g using WellFoundedLT.induction with
    | _ g ih =>
      intro hg hgd
      obtain ⟨a, hag, had⟩ := (downstream_pred A hgd fun hgC => hCY g hgC ((mem_region A).1 hg).2.1).2
      by_cases ha : a ∈ A.region Y (Set.Ico t₀ t₁)
      · exact ih a (A.reads_lt g a hag) ha had
      · exact hS g hg a hag ((mem_region A).1 (egress a ((mem_into A).2 ⟨ha, g, hg, hag⟩) had)).2.1 had
  refine ⟨disjoint_left.2 region, disjoint_left.2 fun a ha had => ?_, fun g hg a hag hy had => ?_⟩
  · obtain ⟨-, g, hg, hag⟩ := (mem_into A).1 ha
    exact hS g hg a hag ((mem_region A).1 (egress a ha had)).2.1 had
  · have hai := (downstream_pred A had fun haC => hCY a haC hy).1
    refine region a ((mem_region A).2 ⟨hai, hy, le_t_of_downstream A hc h₀ had, ?_⟩) had
    exact ((Nat.le_add_right _ _).trans (hc g a hag)).trans_lt ((mem_region A).1 hg).2.2.2

/-- What `Y` reads from outside before `u ≤ t₁`, for `X` apart from `Y`: what it read before `t₀` (cut at `u`), what it
read from `X` during `[t₀, u)`, and what it read from the leaves in neither during `[t₀, t₁)` (cut at `u`). -/
theorem crossing_split {L : Type*} (R : Actual.Run L) {X Y : Finset L} (h : Disjoint X Y) {t₀ u t₁ : ℕ}
    (hu : u ≤ t₁) : R.crossing (↑Y)ᶜ ↑Y (Set.Iio u) = R.crossing (↑Y)ᶜ ↑Y (Set.Iio t₀) ∩ {p | p.1 < u} ∪
      (R.crossing ↑X ↑Y (Set.Ico t₀ u) ∪ R.crossing (↑X ∪ ↑Y)ᶜ ↑Y (Set.Ico t₀ t₁) ∩ {p | p.1 < u}) := by
  ext p
  simp only [Actual.Run.crossing, Set.mem_union, Set.mem_inter_iff, Set.mem_ofPred_eq, Set.mem_Iio, Set.mem_Ico]
  constructor
  · rintro ⟨g, a, hr, ha, hg, ht, rfl⟩
    by_cases h0 : R.A.t g < t₀
    · exact Or.inl ⟨⟨g, a, hr, ha, hg, h0, rfl⟩, ht⟩
    by_cases hx : R.A.x a ∈ (↑X : Set L)
    · exact Or.inr (Or.inl ⟨g, a, hr, hx, hg, ⟨not_lt.1 h0, ht⟩, rfl⟩)
    · exact Or.inr (Or.inr ⟨⟨g, a, hr, fun hm => hm.elim hx ha, hg, ⟨not_lt.1 h0, ht.trans_le hu⟩, rfl⟩, ht⟩)
  · rintro (⟨⟨g, a, hr, ha, hg, h0, rfl⟩, ht⟩ | ⟨g, a, hr, ha, hg, ht, rfl⟩ | ⟨⟨g, a, hr, ha, hg, ht, rfl⟩, hlt⟩)
    · exact ⟨g, a, hr, ha, hg, ht, rfl⟩
    · exact ⟨g, a, hr, fun hy => Finset.disjoint_left.1 h ha hy, hg, ht.2, rfl⟩
    · exact ⟨g, a, hr, fun hy => ha (Or.inr hy), hg, hlt, rfl⟩

/-- Memory per run on the state bounds the circuit's `state`: its gates are in the run's state (in flight between
`X`'s leaves, or held). -/
theorem memory_of_mem {L : Type*} {R : Actual.Run L} {𝒳 : Set (Finset L)} {M : Finset L → ℕ} (h : R.Memory 𝒳 M) :
    R.A.Memory 𝒳 M := by
  classical
  intro X hX t
  refine le_trans ?_ (h X hX t)
  unfold Placed.state Actual.Run.mem Verity.Circuit.width
  refine Finset.sum_le_sum_of_subset fun a ha => ?_
  obtain ⟨hai, hx⟩ := Finset.mem_filter.1 ha
  simp only [Actual.Run.state, Finset.mem_filter, Finset.mem_univ, true_and]
  exact Or.inr ⟨hai, hx⟩

/-- The actual circuit of memory accounting's `not_markov` at coin `b`, over leaves `true` (the verifier) and `false`
(`X`): the verifier's input `g0` and its copy `g1` at tick 0, `g2` at `X` at tick 1 reading `g1`, and `g3` at `X` at
tick 3, reading nothing, with value `b`; gate `gi` in cell `i`. -/
abbrev markovPlaced (b : Bool) : Placed Bool where
  N := 4
  w _ := 1
  isInput g := decide (g = 0)
  reads g := if g = 1 then {0} else if g = 2 then {1} else ∅
  reads_lt g a ha := by fin_cases g <;> simp_all
  input_reads g hg := by fin_cases g <;> simp_all
  op g v := if g = 1 then v 0 else if g = 2 then v 1 else BitVec.ofBool b
  op_congr g v v' h := by fin_cases g <;> simp_all
  outputs := ∅
  x g := decide (g.val < 2)
  t g := if g.val = 2 then 1 else if g.val = 3 then 3 else 0
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun p => p.2.1) h)

/-- Its run at coin `b`, every gate's value `b`, with what each leaf holds given by `held`. -/
abbrev markovRun (b : Bool) (held : Bool → ℕ → Finset (Fin 4)) (h : (markovPlaced b).Holding held) :
    Actual.Run Bool where
  A := markovPlaced b
  val _ := BitVec.ofBool b
  faithful g hg := by fin_cases g <;> simp_all
  held := held
  holding := h

/-- **`not_markov` is closed** (memory accounting's counterexample to v2.1's per-run memory): a strategy running that
circuit at each coin, holding anything `Holding` allows, and causal from every tick has `mem X 2 ≥ 1` at some coin, so
`M_X = 0` fails there. `g3`'s value is what `X` does during `[2, 4)` and nothing reaches `X` then, so `X`'s state at
tick 2 tells the coins apart: `b` is held at `X` across tick 2. -/
theorem markov_mem (held : Bool → Bool → ℕ → Finset (Fin 4)) (h : ∀ b, (markovPlaced b).Holding (held b))
    (hc : CausalFrom fun b => markovRun b (held b) (h b)) : ∃ b, 1 ≤ (markovRun b (held b) (h b)).mem {false} 2 := by
  by_contra hne
  push Not at hne
  have hs : ∀ b, (markovRun b (held b) (h b)).state {false} 2 = ∅ := fun b => by
    have := hne b
    simp only [Actual.Run.mem, Verity.Circuit.width, sum_const, smul_eq_mul, mul_one, Nat.lt_one_iff,
      card_eq_zero] at this
    exact this
  have hx : ∀ b, (markovRun b (held b) (h b)).crossing (↑({false} : Finset Bool))ᶜ ↑({false} : Finset Bool)
      (Set.Ico 2 4) = ∅ := fun b => by
    ext p
    simp only [Actual.Run.crossing, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
    intro g a ha hxa hxg hg
    fin_cases g <;> fin_cases a <;> simp_all
  have key := congrArg Prod.fst
    (hc {false} 2 4 false true (by simp only [Actual.Run.content, hs, image_empty, sort_empty]) (by simp only [hx]))
  have : ((3 : ℕ), false, 3, 1, 1) ∈ ((markovRun true (held true) (h true)).acts ↑({false} : Finset Bool)
      (Set.Ico 2 4)).1 := ⟨3, by simp, by simp, rfl⟩
  rw [← key] at this
  obtain ⟨g, hg, ht, he⟩ := this
  fin_cases g <;> simp_all [Actual.Run.seen]

/-- The circuit admits a holding (the verifier's input at tick 0, nothing else), so `markov_mem` turns on
`CausalFrom`, not on `Holding`. -/
theorem markov_holding (b : Bool) :
    (markovPlaced b).Holding fun y t => if y = true ∧ t = 0 then {0} else ∅ := by
  refine ⟨fun y t a ha => ?_, fun a ha => ?_, fun g a ha hx s hs hst => ?_⟩
  · simp only at ha
    split_ifs at ha with hy <;> simp_all
  · fin_cases a <;> simp_all
  · fin_cases g <;> fin_cases a <;> simp_all

/-- The honest location of the 09:35Z `MODEL:` line, at one leaf: the inputs `g0` and `g1` at tick 0, in cells 0 and
1, and `g2` at tick 1, in cell 2, computing `g0 ∧ ¬g1`. -/
abbrev andPlaced : Placed Unit where
  N := 3
  w _ := 1
  isInput g := decide (g.val < 2)
  reads g := if g = 2 then {0, 1} else ∅
  reads_lt g a ha := by fin_cases g <;> fin_cases a <;> simp_all
  input_reads g hg := by fin_cases g <;> simp_all
  op g v := if g = 2 then BitVec.ofBool ((v 0).getLsbD 0 && !(v 1).getLsbD 0) else 0
  op_congr g v v' h := by fin_cases g <;> simp_all
  outputs := ∅
  x _ := ()
  t g := if g.val = 2 then 1 else 0
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun p => p.2.1) h)

/-- Its run at coins `c = (a, b)`, the inputs held through tick 1. -/
abbrev andRun (c : Bool × Bool) : Actual.Run Unit where
  A := andPlaced
  val g := BitVec.ofBool (if g.val = 0 then c.1 else if g.val = 1 then c.2 else c.1 && !c.2)
  faithful g hg := by fin_cases g <;> simp_all
  held _ t := if t ≤ 1 then {0, 1} else ∅
  holding := by
    refine ⟨fun y t a ha => ?_, fun a ha => ?_, fun g a ha _ s _ hs => ?_⟩
    · simp only at ha
      split_ifs at ha <;> fin_cases a <;> simp_all
    · fin_cases a <;> simp_all
    · fin_cases g <;> fin_cases a <;> simp_all

/-- Through tick 1 its content is `[a, b]`: the inputs, in the order of their cells. -/
theorem and_content (c : Bool × Bool) {t : ℕ} (ht : t ≤ 1) : (andRun c).content univ t = [c.1, c.2] := by
  obtain ⟨a, b⟩ := c
  have hs : (andRun (a, b)).state univ t = {0, 1} := by
    ext g
    simp only [Actual.Run.state, Placed.region, Verity.Circuit.into, Verity.Circuit.computed, mem_filter, mem_univ,
      true_and, Set.mem_Ici]
    fin_cases g <;> simp_all
  rw [Actual.Run.content, hs, image_insert, image_singleton, image_insert, image_singleton,
    sort_insert (h₁ := fun p hp => by
      rw [mem_singleton.1 hp]
      exact Prod.Lex.toLex_le_toLex.2 (Or.inr ⟨rfl, Prod.Lex.toLex_le_toLex.2 (Or.inl Nat.zero_lt_one)⟩))
      (h₂ := by simp [Actual.position, Actual.Run.seen]), sort_singleton]
  cases a <;> cases b <;> rfl

/-- **The honest location is causal** (v2.5): `g0` and `g1` are in different cells, so the coins `(0, 1)` and
`(1, 0)` have contents `[false, true]` and `[true, false]`, and the strategy running it at each coin is causal from
every tick. Without cells, the content order put them by value, alike at those coins, while `g2` tells them apart. -/
theorem and_causal : CausalFrom andRun := by
  intro X t u r r' hc _
  rcases X.eq_empty_or_nonempty with rfl | ⟨⟨⟩, hx⟩
  · simp [Actual.Run.acts, Actual.Run.sent]
  · obtain rfl : X = univ := eq_univ_iff_forall.2 fun _ => hx
    by_cases ht : t ≤ 1
    · rw [and_content r ht, and_content r' ht] at hc
      simp only [List.cons.injEq, and_true] at hc
      rw [Prod.ext hc.1 hc.2]
    · have h0 : ∀ c, (andRun c).acts ↑(univ : Finset Unit) (Set.Ico t u) = (∅, ∅, ∅) := fun c => by
        simp only [Actual.Run.acts, Actual.Run.sent, Prod.mk.injEq]
        refine ⟨?_, ?_, ?_⟩ <;> ext p <;> simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
        · rintro ⟨g, -, hg, -⟩
          fin_cases g <;> simp_all [Set.mem_Ico]; omega
        · rintro ⟨g, a, -, -, hg, -⟩
          exact hg (mem_univ _)
        · rintro ⟨y, v, a, -, hv, ha, -⟩
          simp only [Set.mem_Ico] at hv
          simp [show ¬v + 1 ≤ 1 by omega] at ha
      rw [h0 r, h0 r']

/-- The actual circuit of memory accounting's `zero_link` at coin `b`, over leaves `true` (the verifier) and `false`
(`X`): the verifier's input `g0` at tick 0, `g1` at the verifier of width 0 at tick `b`, reading `g0`, and `g2` at `X`
at tick 3, reading `g1`. Every gate is `b` at its width; gate `gi` in cell `i`. -/
abbrev zeroPlaced (b : Bool) : Placed Bool where
  N := 3
  w g := if g.val = 1 then 0 else 1
  isInput g := decide (g = 0)
  reads g := if g = 1 then {0} else if g = 2 then {1} else ∅
  reads_lt g a ha := by fin_cases g <;> simp_all
  input_reads g hg := by fin_cases g <;> simp_all
  op g _ := BitVec.ofNat _ b.toNat
  op_congr _ _ _ _ := rfl
  outputs := ∅
  x g := decide (g.val < 2)
  t g := if g.val = 1 then b.toNat else if g.val = 2 then 3 else 0
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun p => p.2.1) h)

/-- Its run at coin `b`, with what each leaf holds given by `held`. -/
abbrev zeroRun (b : Bool) (held : Bool → ℕ → Finset (Fin 3)) (h : (zeroPlaced b).Holding held) :
    Actual.Run Bool where
  A := zeroPlaced b
  val _ := BitVec.ofNat _ b.toNat
  faithful _ _ := rfl
  held := held
  holding := h

/-- Its one link has capacity 0 and carries every wire between the leaves: `g1`'s width is 0. -/
theorem zero_links (b : Bool) : (zeroPlaced b).Links (fun _ : Unit => (true, false)) (fun _ => 0) fun _ _ => () := by
  have w0 : ∀ T : Finset (Fin 3), (∀ a ∈ T, a = 1) → ∀ n, (zeroPlaced b).width T ≤ 0 * n := fun T hT n => by
    simp only [zero_mul, nonpos_iff_eq_zero, Verity.Circuit.width]
    exact sum_eq_zero fun a ha => by rw [hT a ha]; rfl
  refine ⟨fun g a ha hne => ?_, fun e y a c => ⟨w0 _ (fun z hz => ?_) _, w0 _ (fun z hz => ?_) _⟩⟩
  · fin_cases g <;> fin_cases a <;> simp_all
  · simp only [Placed.readOn, mem_filter, mem_univ, true_and] at hz
    obtain ⟨g, hzg, hne, -⟩ := hz
    fin_cases g <;> fin_cases z <;> simp_all
  · simp only [Placed.sentOn, mem_filter, mem_univ, true_and] at hz
    obtain ⟨-, g, hzg, hne, -⟩ := hz
    fin_cases g <;> fin_cases z <;> simp_all

/-- **`zero_link` fails `Alphabet` at `β = 0`** (memory accounting's counterexample to v2.1's links, "a link of
capacity 0 carries a bit"): whatever its runs hold, the signal its link delivers to `X` at tick 3 is `g1` as seen, whose
sender's tick is the coin, so the link takes two signals across the strategy where an alphabet of capacity 0 allows
one. It meets `Links` at capacity 0 (`zero_links`); a bound that counts what crosses goes through `Alphabet`. -/
theorem zero_link_alphabet (held : Bool → Bool → ℕ → Finset (Fin 3)) (h : ∀ b, (zeroPlaced b).Holding (held b)) :
    ¬Alphabet (fun b => zeroRun b (held b) (h b)) (fun _ : Unit => 0) fun _ _ _ => () := by
  intro hA
  have hle : (Set.range fun b => (zeroRun b (held b) (h b)).signal (fun _ _ => ()) () false 3).encard ≤ 1 := by
    simpa using hA () false 3
  have h1 := Set.encard_le_one_iff.1 hle _ _ (Set.mem_range_self false) (Set.mem_range_self true)
  have : ((0 : ℕ), true, 1, 0, 0) ∈ (zeroRun false (held false) (h false)).signal (fun _ _ => ()) () false 3 :=
    ⟨2, 1, by simp, by simp, rfl, by simp, by simp, rfl⟩
  rw [h1] at this
  obtain ⟨g, a, ha, -, -, hg, ht, he⟩ := this
  fin_cases g <;> fin_cases a <;> simp_all [Actual.Run.seen]

end Actual

theorem Localization : Guarantees.Localization := by
  intro L Ω _ S X Y t₀ t₁ hXY hc hh hv
  have hin : ∀ r r', (S r).inputs ↑Y = (S r').inputs ↑Y := fun r r' => congrArg Prod.fst (hh r r')
  have hpre : ∀ r r', (S r).crossing (↑Y)ᶜ ↑Y (Set.Iio t₀) = (S r').crossing (↑Y)ᶜ ↑Y (Set.Iio t₀) :=
    fun r r' => congrArg (fun p => p.2.2) (hh r r')
  have key : ∀ u ≤ t₁, ∀ r r', (S r).crossing ↑X ↑Y (Set.Ico t₀ u) = (S r').crossing ↑X ↑Y (Set.Ico t₀ u) →
      (S r).acts ↑Y (Set.Iio u) = (S r').acts ↑Y (Set.Iio u) := by
    intro u hu r r' hx
    refine Actual.global_of_causal hc Y hin u r r' ?_
    rw [Actual.crossing_split (S r) hXY hu, Actual.crossing_split (S r') hXY hu, hpre r r', hx, hv r r']
  have table : ∀ u ≤ t₁, ∃ T : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) →
      Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ × ℕ),
      ∀ r, (S r).acts ↑Y (Set.Iio u) = T ((S r).crossing ↑X ↑Y (Set.Ico t₀ u)) := by
    intro u hu
    have hf : Function.FactorsThrough (fun r => (S r).acts ↑Y (Set.Iio u))
        (fun r => (S r).crossing ↑X ↑Y (Set.Ico t₀ u)) := fun r r' h => key u hu r r' h
    exact ⟨Function.extend _ _ fun _ => (∅, ∅, ∅), fun r => (hf.extend_apply _ r).symm⟩
  refine ⟨table, fun e he => ?_⟩
  obtain ⟨T, hT⟩ := table t₁ le_rfl
  refine le_trans (Set.encard_le_encard ?_) ((Set.encard_image_le T _).trans he)
  rintro _ ⟨r, rfl⟩
  exact ⟨_, ⟨r, rfl⟩, (hT r).symm⟩

theorem Recall : Guarantees.Recall := by
  intro L Ω _ S X t u hc hx
  have hf : Function.FactorsThrough (fun r => (S r).acts ↑X (Set.Ico t u)) (fun r => (S r).content X t) :=
    fun r r' h => hc X t u r r' h (hx r r')
  exact ⟨Function.extend _ _ fun _ => (∅, ∅, ∅), fun r =>
    ⟨(hf.extend_apply _ r).symm, Actual.content_length_le _ _ _⟩⟩

end Accounting.SecurityProofs
