module

public import Definitions.Pouw.Confidence
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.Order.CompletePartialOrder
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.ResourceAccounting.Actual.Confidence
import Std.Tactic.BVDecide.Normalize.Prop

/-! A location as a transducer over what reaches it, in a strategy causal from every tick. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Actual

variable {Ω L : Type*}

theorem mem_crossing_iff (R : Actual.Run L) (X Y : Set L) (I : Set ℕ) (p : ℕ × L × ℕ × L × ℕ × ℕ × ℕ) :
    p ∈ R.crossing X Y I ↔ p.1 ∈ I ∧ p ∈ R.crossing X Y {p.1} := by
  constructor
  · rintro ⟨g, a, hr, ha, hg, ht, rfl⟩
    exact ⟨ht, g, a, hr, ha, hg, rfl, rfl⟩
  · rintro ⟨hI, g, a, hr, ha, hg, -, rfl⟩
    exact ⟨g, a, hr, ha, hg, hI, rfl⟩

theorem crossing_Ico_congr {R R' : Actual.Run L} {X Y : Set L} {t n : ℕ}
    (h : ∀ i < n, R.crossing X Y {t + i} = R'.crossing X Y {t + i}) :
    R.crossing X Y (Set.Ico t (t + n)) = R'.crossing X Y (Set.Ico t (t + n)) := by
  have key : ∀ u ∈ Set.Ico t (t + n), R.crossing X Y {u} = R'.crossing X Y {u} := by
    intro u hu
    obtain ⟨i, rfl⟩ := Nat.exists_eq_add_of_le hu.1
    exact h i (by have := hu.2; omega)
  ext p
  rw [mem_crossing_iff, mem_crossing_iff R']
  constructor
  · rintro ⟨hI, hp⟩
    exact ⟨hI, by rwa [← key _ hI]⟩
  · rintro ⟨hI, hp⟩
    exact ⟨hI, by rwa [key _ hI]⟩

theorem mem_crossing_signal {E : Type*} (R : Actual.Run L) (link : Fin R.A.N → Fin R.A.N → E) (X : Finset L) (u : ℕ)
    (p : ℕ × L × ℕ × L × ℕ × ℕ × ℕ) :
    p ∈ R.crossing (↑X)ᶜ ↑X {u} ↔ p.1 = u ∧ p.2.1 ∈ X ∧ p.2.2.2.1 ∉ X ∧ ∃ e, p.2.2 ∈ R.signal link e p.2.1 u := by
  obtain ⟨u', y, s⟩ := p
  constructor
  · rintro ⟨g, a, hr, ha, hg, ht, he⟩
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl, rfl⟩ := he
    refine ⟨ht, hg, ha, link g a, g, a, hr, fun hx => ha (hx ▸ hg), rfl, rfl, ht, rfl⟩
  · rintro ⟨rfl, hy, hx, e, g, a, hr, -, -, hxg, htg, rfl⟩
    exact ⟨g, a, hr, hx, hxg ▸ hy, htg, by rw [hxg, htg]⟩

theorem crossing_of_signals {E : Type*} {R R' : Actual.Run L} (link : Fin R.A.N → Fin R.A.N → E)
    (link' : Fin R'.A.N → Fin R'.A.N → E) (X : Finset L) (u : ℕ)
    (h : ∀ e, ∀ y ∈ X, R.signal link e y u = R'.signal link' e y u) :
    R.crossing (↑X)ᶜ ↑X {u} = R'.crossing (↑X)ᶜ ↑X {u} := by
  ext p
  rw [mem_crossing_signal R link, mem_crossing_signal R' link']
  constructor
  · rintro ⟨h1, h2, h3, e, he⟩
    exact ⟨h1, h2, h3, e, by rwa [← h e _ h2]⟩
  · rintro ⟨h1, h2, h3, e, he⟩
    exact ⟨h1, h2, h3, e, by rwa [h e _ h2]⟩

theorem crossing_of_ports {E : Type*} (S : Ω → Actual.Run L) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (X : Finset L) (ports : List (E × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X → (link r g a, (S r).A.x g) ∈ ports)
    (u : ℕ) (r r' : Ω) (h : portSignals S link ports r u = portSignals S link ports r' u) :
    (S r).crossing (↑X)ᶜ ↑X {u} = (S r').crossing (↑X)ᶜ ↑X {u} := by
  have hp : ∀ p ∈ ports, (S r).signal (link r) p.1 p.2 u = (S r').signal (link r') p.1 p.2 u :=
    List.map_inj_left.1 h
  have key : ∀ {r r' : Ω}, (∀ p ∈ ports, (S r).signal (link r) p.1 p.2 u = (S r').signal (link r') p.1 p.2 u) →
      (S r).crossing (↑X)ᶜ ↑X {u} ⊆ (S r').crossing (↑X)ᶜ ↑X {u} := by
    intro r r' hp q hq
    have hq' := hq
    obtain ⟨g, a, hr, ha, hg, -, rfl⟩ := hq'
    rw [mem_crossing_signal (S r) (link r)] at hq
    rw [mem_crossing_signal (S r') (link r')]
    refine ⟨hq.1, hq.2.1, hq.2.2.1, link r g a, ?_⟩
    rw [← hp _ (hports r g a hr ha hg)]
    exact ⟨g, a, hr, fun hx => ha (hx ▸ hg), rfl, rfl, hq.1, rfl⟩
  exact (key hp).antisymm (key fun p hm => (hp p hm).symm)

theorem portSignals_encard_le {E : Type*} (S : Ω → Actual.Run L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (hα : Alphabet S β link) (u : ℕ) :
    ∀ ports : List (E × L), (Set.range fun r => portSignals S link ports r u).encard ≤
      ((2 ^ (ports.map fun p => β p.1).sum : ℕ) : ℕ∞)
  | [] => by
    have : (Set.range fun r => portSignals S link [] r u) ⊆ {[]} := by
      rintro _ ⟨r, rfl⟩
      rfl
    exact (Set.encard_le_encard this).trans (by simp)
  | p :: ps => by
    have hsub : (Set.range fun r => portSignals S link (p :: ps) r u) ⊆
        (fun q : Set (ℕ × L × ℕ × ℕ × ℕ) × List (Set (ℕ × L × ℕ × ℕ × ℕ)) => q.1 :: q.2) ''
          ((Set.range fun r => (S r).signal (link r) p.1 p.2 u) ×ˢ Set.range fun r => portSignals S link ps r u) := by
      rintro _ ⟨r, rfl⟩
      exact ⟨(_, _), ⟨⟨r, rfl⟩, ⟨r, rfl⟩⟩, rfl⟩
    refine (Set.encard_le_encard hsub).trans ((Set.encard_image_le _ _).trans ?_)
    rw [Set.encard_prod]
    calc _ ≤ ((2 ^ β p.1 : ℕ) : ℕ∞) * ((2 ^ (ps.map fun p => β p.1).sum : ℕ) : ℕ∞) :=
          mul_le_mul' (hα p.1 p.2 u) (portSignals_encard_le S β link hα u ps)
      _ = _ := by simp [List.map_cons, List.sum_cons, pow_add]

def pad (M : ℕ) (c : List Bool) : Fin (M + 1) → Bool := fun i =>
  (List.replicate (M - c.length) false ++ true :: c).getD i false

theorem replicate_cancel : ∀ (a a' : ℕ) (c c' : List Bool),
    List.replicate a false ++ true :: c = List.replicate a' false ++ true :: c' → c = c'
  | 0, 0, c, c', h => by simpa using h
  | 0, a' + 1, c, c', h => by simp [List.replicate_succ] at h
  | a + 1, 0, c, c', h => by simp [List.replicate_succ] at h
  | a + 1, a' + 1, c, c', h => replicate_cancel a a' c c' (by simpa [List.replicate_succ] using h)

theorem pad_inj {M : ℕ} {c c' : List Bool} (hc : c.length ≤ M) (hc' : c'.length ≤ M) (h : pad M c = pad M c') :
    c = c' := by
  refine replicate_cancel (M - c.length) (M - c'.length) c c'
    (List.ext_getElem (by simp only [List.length_append, List.length_replicate, List.length_cons]; omega)
      fun i h1 h2 => ?_)
  have h1' := h1
  simp only [List.length_append, List.length_replicate, List.length_cons] at h1'
  have := congrFun h ⟨i, by omega⟩
  simp only [pad] at this
  rwa [List.getD_eq_getElem _ _ h1, List.getD_eq_getElem _ _ h2] at this

variable [LinearOrder L]

theorem transducer {Ob : Type*} {S : Ω → Actual.Run L} (hc : CausalFrom S) (X : Finset L) (t : ℕ) (obs : Ω → ℕ → Ob)
    (hobs : ∀ u r r', obs r u = obs r' u → (S r).crossing (↑X)ᶜ ↑X {u} = (S r').crossing (↑X)ᶜ ↑X {u}) :
    ∃ F : ℕ → List Bool → List Ob →
        Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ × ℕ),
      ∀ r n, (S r).acts ↑X (Set.Ico t (t + n)) =
        F n ((S r).content X t) ((List.range n).map fun i => obs r (t + i)) := by
  have hf : ∀ n, Function.FactorsThrough (fun r => (S r).acts ↑X (Set.Ico t (t + n)))
      (fun r => ((S r).content X t, (List.range n).map fun i => obs r (t + i))) := by
    intro n r r' h
    simp only [Prod.mk.injEq] at h
    exact hc X t (t + n) r r' h.1
      (crossing_Ico_congr fun i hi => hobs _ r r' (List.map_inj_left.1 h.2 i (List.mem_range.2 hi)))
  exact ⟨fun n c os => Function.extend (fun r => ((S r).content X t, (List.range n).map fun i => obs r (t + i)))
    (fun r => (S r).acts ↑X (Set.Ico t (t + n))) (fun _ => (∅, ∅, ∅)) (c, os),
    fun r n => ((hf n).extend_apply _ r).symm⟩

theorem transducer_bits {Ob : Type*} {S : Ω → Actual.Run L} (hc : CausalFrom S) (X : Finset L) (t : ℕ)
    (obs : Ω → ℕ → Ob)
    (hobs : ∀ u r r', obs r u = obs r' u → (S r).crossing (↑X)ᶜ ↑X {u} = (S r').crossing (↑X)ᶜ ↑X {u})
    (M : ℕ) (hM : ∀ r, (S r).mem X t ≤ M) :
    ∃ F : ℕ → (Fin (M + 1) → Bool) → List Ob →
        Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ × ℕ),
      ∀ r n, (S r).acts ↑X (Set.Ico t (t + n)) =
        F n (pad M ((S r).content X t)) ((List.range n).map fun i => obs r (t + i)) := by
  obtain ⟨F, hF⟩ := transducer hc X t obs hobs
  have hf : ∀ n, Function.FactorsThrough (fun r => (S r).acts ↑X (Set.Ico t (t + n)))
      (fun r => (pad M ((S r).content X t), (List.range n).map fun i => obs r (t + i))) := by
    intro n r r' h
    simp only [Prod.mk.injEq] at h
    simp only
    rw [hF r n, hF r' n, pad_inj ((content_length_le _ _ _).trans (hM r)) ((content_length_le _ _ _).trans (hM r'))
      h.1, h.2]
  exact ⟨fun n b os => Function.extend (fun r => (pad M ((S r).content X t), (List.range n).map fun i => obs r (t + i)))
    (fun r => (S r).acts ↑X (Set.Ico t (t + n))) (fun _ => (∅, ∅, ∅)) (b, os),
    fun r n => ((hf n).extend_apply _ r).symm⟩

theorem lists_encard_le (M : ℕ) : {l : List Bool | l.length ≤ M}.encard + 1 ≤ 2 ^ (M + 1) := by
  induction M with
  | zero =>
    have : {l : List Bool | l.length ≤ 0} = {[]} := by
      ext l
      simp
    rw [this, Set.encard_singleton]
    norm_num
  | succ M ih =>
    have hsub : {l : List Bool | l.length ≤ M + 1} ⊆
        {[]} ∪ (fun p : Bool × List Bool => p.1 :: p.2) '' (Set.univ ×ˢ {l | l.length ≤ M}) := by
      intro l hl
      cases l with
      | nil => exact Or.inl rfl
      | cons x l => exact Or.inr ⟨(x, l), ⟨trivial, by simpa using hl⟩, rfl⟩
    have h1 := (Set.encard_le_encard hsub).trans ((Set.encard_union_le _ _).trans
      (add_le_add le_rfl (Set.encard_image_le (fun p : Bool × List Bool => p.1 :: p.2)
        (Set.univ ×ˢ {l : List Bool | l.length ≤ M}))))
    rw [Set.encard_prod, Set.encard_univ, ENat.card_eq_coe_fintype_card, Fintype.card_bool, Set.encard_singleton]
      at h1
    calc {l : List Bool | l.length ≤ M + 1}.encard + 1
        ≤ (1 + ((2 : ℕ) : ℕ∞) * {l : List Bool | l.length ≤ M}.encard) + 1 := by gcongr
      _ = 2 * ({l : List Bool | l.length ≤ M}.encard + 1) := by push_cast; ring
      _ ≤ 2 * 2 ^ (M + 1) := by gcongr
      _ = 2 ^ (M + 1 + 1) := by ring

theorem obs_encard_le {Ob : Type*} (obs : Ω → ℕ → Ob) (t β : ℕ)
    (hα : ∀ u, (Set.range fun r => obs r u).encard ≤ ((2 ^ β : ℕ) : ℕ∞)) (n : ℕ) :
    (Set.range fun r => (List.range n).map fun i => obs r (t + i)).encard ≤ ((2 ^ (β * n) : ℕ) : ℕ∞) := by
  induction n with
  | zero =>
    have : (Set.range fun r : Ω => (List.range 0).map fun i => obs r (t + i)) ⊆ {[]} := by
      rintro _ ⟨r, rfl⟩
      simp
    exact (Set.encard_le_encard this).trans (by simp)
  | succ n ih =>
    have hsub : (Set.range fun r => (List.range (n + 1)).map fun i => obs r (t + i)) ⊆
        (fun p : List Ob × Ob => p.1 ++ [p.2]) ''
          ((Set.range fun r => (List.range n).map fun i => obs r (t + i)) ×ˢ Set.range fun r => obs r (t + n)) := by
      rintro _ ⟨r, rfl⟩
      exact ⟨(_, _), ⟨⟨r, rfl⟩, ⟨r, rfl⟩⟩, by simp [List.range_succ]⟩
    refine (Set.encard_le_encard hsub).trans ((Set.encard_image_le _ _).trans ?_)
    rw [Set.encard_prod]
    calc _ ≤ ((2 ^ (β * n) : ℕ) : ℕ∞) * ((2 ^ β : ℕ) : ℕ∞) := mul_le_mul' ih (hα _)
      _ = _ := by push_cast; ring

theorem acts_encard_le {Ob : Type*} {S : Ω → Actual.Run L} (hc : CausalFrom S) (X : Finset L) (t : ℕ)
    (obs : Ω → ℕ → Ob)
    (hobs : ∀ u r r', obs r u = obs r' u → (S r).crossing (↑X)ᶜ ↑X {u} = (S r').crossing (↑X)ᶜ ↑X {u})
    (M β : ℕ) (hM : ∀ r, (S r).mem X t ≤ M) (hα : ∀ u, (Set.range fun r => obs r u).encard ≤ ((2 ^ β : ℕ) : ℕ∞))
    (n : ℕ) :
    (Set.range fun r => (S r).acts ↑X (Set.Ico t (t + n))).encard ≤ ((2 ^ (M + 1 + β * n) : ℕ) : ℕ∞) := by
  obtain ⟨F, hF⟩ := transducer hc X t obs hobs
  have hsub : (Set.range fun r => (S r).acts ↑X (Set.Ico t (t + n))) ⊆
      (fun p : List Bool × List Ob => F n p.1 p.2) ''
        ({l | l.length ≤ M} ×ˢ Set.range fun r => (List.range n).map fun i => obs r (t + i)) := by
    rintro _ ⟨r, rfl⟩
    exact ⟨(_, _), ⟨(content_length_le _ _ _).trans (hM r), ⟨r, rfl⟩⟩, (hF r n).symm⟩
  refine (Set.encard_le_encard hsub).trans ((Set.encard_image_le _ _).trans ?_)
  rw [Set.encard_prod]
  have h1 : {l : List Bool | l.length ≤ M}.encard ≤ 2 ^ (M + 1) := le_self_add.trans (lists_encard_le M)
  calc _ ≤ (2 : ℕ∞) ^ (M + 1) * ((2 ^ (β * n) : ℕ) : ℕ∞) := mul_le_mul' h1 (obs_encard_le obs t β hα n)
    _ = _ := by push_cast; ring

theorem acts_encard_le_links {E : Type*} {S : Ω → Actual.Run L} (hc : CausalFrom S) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (hα : Alphabet S β link) (X : Finset L) (ports : List (E × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X → (link r g a, (S r).A.x g) ∈ ports)
    (t M : ℕ) (hM : ∀ r, (S r).mem X t ≤ M) (n : ℕ) :
    (Set.range fun r => (S r).acts ↑X (Set.Ico t (t + n))).encard ≤
      ((2 ^ (M + 1 + (ports.map fun p => β p.1).sum * n) : ℕ) : ℕ∞) :=
  acts_encard_le hc X t (portSignals S link ports) (fun u r r' => crossing_of_ports S link X ports hports u r r') M _
    hM (fun u => portSignals_encard_le S β link hα u ports) n

end ResourceAccounting.SecurityProofs.Actual
