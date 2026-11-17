module

public import Proofs.Flock.Claims.Sampling.Select.Layout
public import Properties.Spot.Draw
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
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.ENatToNat
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
import Proofs.Flock.Claims.Sampling.Select.Take

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

open Lean (Json)

/-- The sampling design object `Select.select` reads: the expansion's, else the named sampling design's. -/
def samplingDesignObj (named : Json) (derived : Option Flock.Draw.SamplingDesign) : Json := (derived.map (·.toJson)).getD named

theorem select_ok (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json)
    (cm : Option (Array (Int × Array Int))) (src : ByteArray) (draw : Json) (proved : Array ℕ) (read : ℕ)
    (h : Select.select rule d query table cm src = .ok (draw, proved, read)) :
    ∃ n named derived, Partition.units d query = .ok n ∧ rule.samplingDesignOver n = .ok named ∧
      Draw.deriveSamplingDesign d query named table = .ok derived ∧ ∃ units : Array ℕ, (∀ x ∈ units, x ∈ proved) ∧
        ((rule = .all ∧ units = Array.range n) ∨
          (rule ≠ .all ∧ ∃ parsed read', Draw.flockSamplingDesign (samplingDesignObj named derived) n = .ok parsed ∧
            Draw.sampleUnits (derived.getD parsed) n src = .ok (units, read'))) := by
  unfold Select.select at h
  obtain ⟨n, hn, h⟩ := bind_eq_ok.1 h
  obtain ⟨named, hnamed, h⟩ := bind_eq_ok.1 h
  obtain ⟨derived, hderived, h⟩ := bind_eq_ok.1 h
  refine ⟨n, named, derived, hn, hnamed, hderived, ?_⟩
  dsimp only at h
  have tail : ∀ (units : Array ℕ) (c : Bool) (e : String) (X : Json) (C : Array ℕ) (r : ℕ),
      (if c = true then ((throw e : Flock.V Unit) >>= fun _ => (pure (X, units ++ C, r) : Flock.V (Json × Array ℕ × ℕ)))
        else pure (X, units ++ C, r)) = .ok (draw, proved, read) → ∀ x ∈ units, x ∈ proved := by
    intro units c e X C r ht x hx
    split at ht
    · exact absurd ht (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
    · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at ht
      obtain ⟨-, rfl, -⟩ := ht
      exact Array.mem_append_left _ hx
  split at h
  · split at h
    · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
    · split at h
      · rw [pure_bind] at h
        exact ⟨_, tail _ _ _ _ _ _ h, Or.inl ⟨rfl, rfl⟩⟩
      · rename_i hne
        obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
        obtain ⟨⟨units, read'⟩, hu, h⟩ := bind_eq_ok.1 h
        exact ⟨units, tail _ _ _ _ _ _ h, Or.inr ⟨fun e => hne e, parsed, read', hp, hu⟩⟩
  · split at h
    · rw [pure_bind] at h
      exact ⟨_, tail _ _ _ _ _ _ h, Or.inl ⟨rfl, rfl⟩⟩
    · rename_i hne
      obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
      obtain ⟨⟨units, read'⟩, hu, h⟩ := bind_eq_ok.1 h
      exact ⟨units, tail _ _ _ _ _ _ h, Or.inr ⟨fun e => hne e, parsed, read', hp, hu⟩⟩

def kindOf (samplingDesign : Json) : String := match getD samplingDesign "law" with | .str s => s | _ => ""

theorem probability_ok (p : String) (a b : ℕ) (h : Draw.probability? p = some (a, b)) : 0 < b ∧ a ≤ b := by
  unfold Draw.probability? at h
  split at h
  · split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      rename_i hc
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      exact ⟨hc.1.2, hc.2⟩
    · exact absurd h (by simp)
  · exact absurd h (by simp)

theorem throw_bind_ne {ε α β : Type} (e : ε) (f : α → Except ε β) (b : β) :
    ((throw e : Except ε α) >>= f) ≠ .ok b := by
  simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind]

theorem throw_ne {ε α : Type} (e : ε) (b : α) : (throw e : Except ε α) ≠ .ok b := by
  simp [throw, throwThe, MonadExceptOf.throw]

theorem flockSamplingDesign_cases (samplingDesign : Json) (n : ℕ) (parsed : Flock.Draw.SamplingDesign) (h : Draw.flockSamplingDesign samplingDesign n = .ok parsed) :
    (∃ k, parsed = .subset k ∧ k ≤ n ∧ nat? (getD samplingDesign "k") = some k ∧ 1 ≤ k) ∨
      (∃ a b, parsed = .bernoulli a b ∧ 0 < b ∧ a ≤ b) ∨ kindOf samplingDesign = "stratified" ∨ kindOf samplingDesign = "work" := by
  unfold Draw.flockSamplingDesign at h
  split at h
  next s hs =>
    have hk : kindOf samplingDesign = s := by simp [kindOf, hs]
    dsimp only at h
    split at h
    · split at h
      · exact absurd h (throw_bind_ne _ _ _)
      split at h
      · rename_i k hnat
        split at h
        · exact absurd h (throw_bind_ne _ _ _)
        split at h
        · exact absurd h (throw_bind_ne _ _ _)
        · obtain rfl := (Except.ok.inj h).symm
          exact Or.inl ⟨_, rfl, by omega, hnat, by omega⟩
      · exact absurd h (throw_ne _ _)
    · split at h
      · exact absurd h (throw_bind_ne _ _ _)
      split at h
      · split at h
        · obtain rfl := (Except.ok.inj h).symm
          exact Or.inr (Or.inl ⟨_, _, rfl, probability_ok _ _ _ (by assumption)⟩)
        · exact absurd h (throw_ne _ _)
      · exact absurd h (throw_ne _ _)
    · exact Or.inr (Or.inr (Or.inl (by rwa [hk])))
    · exact Or.inr (Or.inr (Or.inr (by rwa [hk])))
    · exact absurd h (throw_ne _ _)
  next => exact absurd h (throw_ne _ _)

theorem deriveSamplingDesign_cases (d : Program) (query samplingDesign : Json) (table : Option Json)
    (derived : Option Flock.Draw.SamplingDesign) (h : Draw.deriveSamplingDesign d query samplingDesign table = .ok derived) :
    (derived = none ∧ kindOf samplingDesign ≠ "stratified" ∧ kindOf samplingDesign ≠ "work") ∨
      ∃ S k samplingDesign', Partition.strata d query = .ok S ∧ derived = some samplingDesign' ∧
        (Flock.Draw.derive k S = .ok samplingDesign' ∨
          ∃ t tab, Flock.Draw.workTableOfJson t = .ok tab ∧ Flock.Draw.deriveWork k S tab = .ok samplingDesign') := by
  unfold Draw.deriveSamplingDesign at h
  split at h
  next s hs =>
    have hk : kindOf samplingDesign = s := by simp [kindOf, hs]
    dsimp only at h
    split at h
    · rename_i hc
      obtain rfl := (Except.ok.inj h).symm
      simp only [Bool.not_eq_true', Bool.or_eq_false_iff, beq_eq_false_iff_ne, ne_eq] at hc
      exact Or.inl ⟨rfl, hk ▸ hc.1, hk ▸ hc.2⟩
    · obtain ⟨S, hS, h⟩ := bind_eq_ok.1 h
      split at h
      · split at h
        · obtain ⟨samplingDesign', hd, h⟩ := bind_eq_ok.1 h
          obtain rfl := (Except.ok.inj h).symm
          exact Or.inr ⟨S, _, samplingDesign', hS, rfl, Or.inl hd⟩
        · split at h
          · obtain ⟨tab, ht, h⟩ := bind_eq_ok.1 h
            obtain ⟨samplingDesign', hd, h⟩ := bind_eq_ok.1 h
            obtain rfl := (Except.ok.inj h).symm
            exact Or.inr ⟨S, _, samplingDesign', hS, rfl, Or.inr ⟨_, tab, ht, hd⟩⟩
          · exact absurd h (throw_ne _ _)
      · exact absurd h (throw_ne _ _)
  next hne =>
    have hk : kindOf samplingDesign = "" := by
      unfold kindOf; split
      · rename_i s hs; exact absurd hs (hne s)
      · rfl
    dsimp only at h
    split at h
    swap
    · rename_i hc; exact absurd (by decide) hc
    obtain rfl := (Except.ok.inj h).symm
    exact Or.inl ⟨rfl, by rw [hk]; decide, by rw [hk]; decide⟩

theorem deriveSamplingDesign_none (d : Program) (query samplingDesign : Json) (table : Option Json) (h1 : kindOf samplingDesign ≠ "stratified")
    (h2 : kindOf samplingDesign ≠ "work") : Draw.deriveSamplingDesign d query samplingDesign table = .ok none := by
  unfold Draw.deriveSamplingDesign
  split
  next s hs =>
    have hk : kindOf samplingDesign = s := by simp [kindOf, hs]
    rw [hk] at h1 h2
    dsimp only
    rw [ite_eq_left (by simp [h1, h2])]
    rfl
  next => rfl

def rsSize (rs : Array (ℕ × ℕ)) : ℕ := (expand rs.toList).length

def strataSize (S : Array (String × Array (ℕ × ℕ))) : ℕ := (S.toList.map fun p => rsSize p.2).sum

theorem foldl_rs (l : List (ℕ × ℕ)) (a : ℕ) :
    l.foldl (fun a (x : ℕ × ℕ) => a + (x.2 - x.1)) a = a + (expand l).length := by
  induction l generalizing a with
  | nil => simp [expand]
  | cons p l ih => simp only [List.foldl_cons, ih, expand, List.flatMap_cons, List.length_append, List.length_range']; omega

theorem size_rs (rs : Array (ℕ × ℕ)) : rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0 = rsSize rs := by
  rw [← Array.foldl_toList]
  exact (foldl_rs rs.toList 0).trans (Nat.zero_add _)

theorem stratum_size (s : Flock.Draw.Stratum) : s.size = rsSize s.units := size_rs s.units

theorem foldl_strata (l : List (String × Array (ℕ × ℕ))) (a : ℕ) :
    l.foldl (fun acc (x : String × Array (ℕ × ℕ)) => acc + rsSize x.2) a = a + (l.map fun p => rsSize p.2).sum := by
  induction l generalizing a with
  | nil => simp
  | cons p l ih => simp only [List.foldl_cons, ih, List.map_cons, List.sum_cons]; omega

theorem size_strata (S : Array (String × Array (ℕ × ℕ))) :
    S.foldl (fun acc (x : String × Array (ℕ × ℕ)) => acc + rsSize x.2) 0 = strataSize S := by
  rw [← Array.foldl_toList]
  exact (foldl_strata S.toList 0).trans (Nat.zero_add _)

theorem derive_ok (k : ℕ) (S : Array (String × Array (ℕ × ℕ))) (samplingDesign' : Flock.Draw.SamplingDesign)
    (h : Flock.Draw.derive k S = .ok samplingDesign') :
    samplingDesign' = .stratified k (S.map fun p => { template := p.1, units := p.2, k := Flock.Draw.stratumK k (rsSize p.2) (strataSize S) }) := by
  unfold Flock.Draw.derive at h
  dsimp only at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  · obtain rfl := (Except.ok.inj h).symm
    simp only [size_rs, size_strata]

theorem deriveWork_ok (k : ℕ) (S : Array (String × Array (ℕ × ℕ))) (tab : Array (String × ℕ × ℕ))
    (samplingDesign' : Flock.Draw.SamplingDesign) (h : Flock.Draw.deriveWork k S tab = .ok samplingDesign') :
    ∃ ws : Array Flock.Draw.WorkStratum, List.Forall₂ (fun p (w : Flock.Draw.WorkStratum) => w.units = p.2) S.toList ws.toList ∧
      samplingDesign' = .work k (ws.map fun s => { s with k := Flock.Draw.workK k s.work s.size (Flock.Draw.totalWork ws) s.floor }) := by
  unfold Flock.Draw.deriveWork at h
  dsimp only at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  · obtain ⟨ws, hws, h⟩ := bind_eq_ok.1 h
    refine ⟨ws, (mapM_ok _ _ _ hws).imp fun {p w} hp => ?_, ?_⟩
    · obtain ⟨t, rs⟩ := p
      split at hp
      · obtain rfl := (Except.ok.inj hp).symm; rfl
      · exact absurd hp (throw_ne _ _)
    · split at h
      · exact absurd h (throw_bind_ne _ _ _)
      · exact (Except.ok.inj h).symm

theorem slots_select_ok (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json)
    (cm : Option (Array (Int × Array Int))) (src : ByteArray) (l : Slots.SlotOrder) (c : Array ℕ)
    (h : Slots.select rule d query table cm src = .ok (l, c)) :
    Slots.layout d query = .ok l ∧ ∃ draw proved read, Select.select rule d query table cm src = .ok (draw, proved, read) ∧
      c = Slots.checked l proved := by
  unfold Slots.select at h
  obtain ⟨l', hl, h⟩ := bind_eq_ok.1 h
  obtain ⟨⟨draw, proved, read⟩, hs, h⟩ := bind_eq_ok.1 h
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj h)
  exact ⟨hl, draw, proved, read, hs, rfl⟩

theorem slotDesign_all (d : Program) (query : Json) (table : Option Json) (n : ℕ) (named : Json)
    (derived : Option Flock.Draw.SamplingDesign) (hn : Partition.units d query = .ok n) (hnamed : Select.Rule.all.samplingDesignOver n = .ok named)
    (hd : Draw.deriveSamplingDesign d query named table = .ok derived) :
    Properties.Spot.slotDesign .all d query table = .ok ⟨[.all, .all]⟩ := by
  unfold Properties.Spot.slotDesign
  rw [hn, bind_ok, hnamed, bind_ok, hd, bind_ok]
  rfl

theorem slotDesign_samplingDesign (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json) (n : ℕ)
    (named : Json) (derived : Option Flock.Draw.SamplingDesign) (parsed : Flock.Draw.SamplingDesign) (hne : rule ≠ .all)
    (hn : Partition.units d query = .ok n) (hnamed : rule.samplingDesignOver n = .ok named)
    (hd : Draw.deriveSamplingDesign d query named table = .ok derived) (hp : Draw.flockSamplingDesign (samplingDesignObj named derived) n = .ok parsed) :
    Properties.Spot.slotDesign rule d query table =
      .ok ⟨[Properties.Spot.oneStageOf (derived.getD parsed), .all]⟩ := by
  unfold Properties.Spot.slotDesign
  rw [hn, bind_ok, hnamed, bind_ok, hd, bind_ok]
  split
  · exact absurd rfl hne
  · unfold samplingDesignObj at hp
    rw [hp, bind_ok]
    rfl

end Properties.Spot.SecurityProofs
