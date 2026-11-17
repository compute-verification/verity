module

public import Definitions.ResourceAccounting.Guarantees.Localization
public import Proofs.ResourceAccounting.Defs.Localization
public import Definitions.Registration.Physical
public import Mathlib.Tactic.FinCases
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific
import Mathlib.Tactic.NormNum.Pow
import Proofs.ResourceAccounting.Actual.Confidence
import Proofs.ResourceAccounting.Actual.Split

/-! `Localization`, proved, with the counterexamples that fix its premises (`not_markov`, `and_causal`, `zero_link`). -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs

open Finset

namespace Actual

variable {L : Type*} (A : Placed L)

theorem mem_downstream {C : Finset (Fin A.N)} {g : Fin A.N} :
    g ∈ A.downstream C ↔ ∃ c ∈ C, Relation.ReflTransGen (fun a g => a ∈ A.reads g) c g := by
  simp [Placed.downstream]

theorem downstream_pred {C : Finset (Fin A.N)} {g : Fin A.N} (hg : g ∈ A.downstream C) (hC : g ∉ C) :
    A.isInput g = false ∧ ∃ a ∈ A.reads g, a ∈ A.downstream C := by
  obtain ⟨c, hc, h⟩ := (mem_downstream A).1 hg
  rcases h.cases_tail with rfl | ⟨a, ha, hag⟩
  · exact absurd hc hC
  refine ⟨?_, a, hag, (mem_downstream A).2 ⟨c, hc, ha⟩⟩
  cases hi : A.isInput g
  · rfl
  · simp [A.input_reads g hi] at hag

theorem le_t_of_downstream {ℓ : L → L → ℕ} (hc : A.Causal ℓ) {C : Finset (Fin A.N)} {t₀ : ℕ}
    (h₀ : ∀ c ∈ C, t₀ ≤ A.t c) {g : Fin A.N} (hg : g ∈ A.downstream C) : t₀ ≤ A.t g := by
  obtain ⟨c, hcC, h⟩ := (mem_downstream A).1 hg
  clear hg
  induction h with
  | refl => exact h₀ c hcC
  | tail _ hab ih => exact ih.trans ((Nat.le_add_right _ _).trans (hc _ _ hab))

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

abbrev markovRun (b : Bool) (held : Bool → ℕ → Finset (Fin 4)) (h : (markovPlaced b).Holding held) :
    Actual.Run Bool where
  A := markovPlaced b
  val _ := BitVec.ofBool b
  faithful g hg := by fin_cases g <;> simp_all
  held := held
  holding := h

/-- The counterexample to per-run memory: at some coin `mem X 2 ≥ 1`, so `M_X = 0` fails there. -/
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
  have : ((3 : ℕ), false, 3, 1, 1) ∈ ((markovRun true (held true) (h true)).acts ({false} : Finset Bool)
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

/-- Without cells, the content order puts `g0` and `g1` by value, alike at the coins `(0, 1)` and `(1, 0)`, while
`g2` tells them apart. -/
theorem and_causal : CausalFrom andRun := by
  intro X t u r r' hc _
  rcases X.eq_empty_or_nonempty with rfl | ⟨⟨⟩, hx⟩
  · simp [Actual.Run.acts, Actual.Run.sent, Actual.Run.state]
  · obtain rfl : X = univ := eq_univ_iff_forall.2 fun _ => hx
    by_cases ht : t ≤ 1
    · rw [and_content r ht, and_content r' ht] at hc
      simp only [List.cons.injEq, and_true] at hc
      rw [Prod.ext hc.1 hc.2]
    · have h0 : ∀ c, (andRun c).acts (univ : Finset Unit) (Set.Ico t u) = (∅, ∅, ∅) := fun c => by
        simp only [Actual.Run.acts, Actual.Run.sent, Prod.mk.injEq]
        refine ⟨?_, ?_, ?_⟩ <;> ext p <;> simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
        · rintro ⟨g, -, hg, -⟩
          fin_cases g <;> simp_all [Set.mem_Ico]; omega
        · rintro ⟨g, a, -, -, hg, -⟩
          exact hg (mem_univ _)
        · rintro ⟨v, a, hv, ha, -⟩
          simp only [Set.mem_Ico] at hv
          simp only [Actual.Run.state, Placed.region, Verity.Circuit.into, Verity.Circuit.computed, mem_filter,
            mem_univ, true_and, Set.mem_Ici, show ¬v + 1 ≤ 1 by omega, ite_false, notMem_empty, exists_false,
            false_or] at ha
          obtain ⟨-, g, ⟨-, hg⟩, -⟩ := ha.1
          fin_cases g <;> simp at hg
          omega
      rw [h0 r, h0 r']

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

abbrev zeroRun (b : Bool) (held : Bool → ℕ → Finset (Fin 3)) (h : (zeroPlaced b).Holding held) :
    Actual.Run Bool where
  A := zeroPlaced b
  val _ := BitVec.ofNat _ b.toNat
  faithful _ _ := rfl
  held := held
  holding := h

theorem zero_links (b : Bool) : (zeroPlaced b).Links (fun _ : Unit => (true, false)) (fun _ => 0) fun _ _ => () := by
  have w0 : ∀ T : Finset (Fin 3), (∀ a ∈ T, a = 1) → ∀ n, (zeroPlaced b).width T ≤ 0 * n := fun T hT n => by
    simp only [zero_mul, nonpos_iff_eq_zero, Verity.Circuit.width]
    exact sum_eq_zero fun a ha => by rw [hT a ha]; rfl
  refine ⟨fun g a ha hne => ?_, fun e y a c => w0 _ (fun z hz => ?_) _⟩
  · fin_cases g <;> fin_cases a <;> simp_all
  · simp only [Placed.readOn, mem_filter, mem_univ, true_and] at hz
    obtain ⟨g, hzg, hne, -⟩ := hz
    fin_cases g <;> fin_cases z <;> simp_all

/-- The counterexample to a bound by `Links` alone: `zero_link` meets `Links` at capacity 0 (`zero_links`), so a
bound that counts what crosses goes through `Alphabet`. -/
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
      (S r).acts Y (Set.Iio u) = (S r').acts Y (Set.Iio u) := by
    intro u hu r r' hx
    refine Actual.global_of_causal hc Y hin u r r' ?_
    rw [Actual.crossing_split (S r) hXY hu, Actual.crossing_split (S r') hXY hu, hpre r r', hx, hv r r']
  have table : ∀ u ≤ t₁, ∃ T : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) →
      Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ × ℕ),
      ∀ r, (S r).acts Y (Set.Iio u) = T ((S r).crossing ↑X ↑Y (Set.Ico t₀ u)) := by
    intro u hu
    have hf : Function.FactorsThrough (fun r => (S r).acts Y (Set.Iio u))
        (fun r => (S r).crossing ↑X ↑Y (Set.Ico t₀ u)) := fun r r' h => key u hu r r' h
    exact ⟨Function.extend _ _ fun _ => (∅, ∅, ∅), fun r => (hf.extend_apply _ r).symm⟩
  refine ⟨table, fun e he => ?_⟩
  obtain ⟨T, hT⟩ := table t₁ le_rfl
  refine le_trans (Set.encard_le_encard ?_) ((Set.encard_image_le T _).trans he)
  rintro _ ⟨r, rfl⟩
  exact ⟨_, ⟨r, rfl⟩, (hT r).symm⟩

theorem Recall : Guarantees.Recall := by
  intro L Ω _ S X t u hc hx
  have hf : Function.FactorsThrough (fun r => (S r).acts X (Set.Ico t u)) (fun r => (S r).content X t) :=
    fun r r' h => hc X t u r r' h (hx r r')
  exact ⟨Function.extend _ _ fun _ => (∅, ∅, ∅), fun r =>
    ⟨(hf.extend_apply _ r).symm, Actual.content_length_le _ _ _⟩⟩

end ResourceAccounting.SecurityProofs
