module

public import ArkLib.Data.CodingTheory.GuruswamiSudan.GuruswamiSudan
public import Mathlib.NumberTheory.LucasLehmer
public import Proofs.Flock.Claims.Counted
import Definitions.Claims
import Mathlib
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Star.Real
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
import Mathlib.NumberTheory.SelbergSieve
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
import Std.Tactic.BVDecide.Normalize.BitVec
import Std.Tactic.BVDecide.Normalize.Prop

/-! A list decoder for interleaved Reed–Solomon codes (`IRS.decode`), from Guruswami–Sudan. -/

@[expose] public section

namespace Claims

open Polynomial

def Flock.ListComplete {Col T : Type} [DecidableEq Col] (enc : T → ℕ → Col) (agree : ℝ) (nMin : T → ℕ)
    (decode : List (ℕ × Col) → List T) : Prop :=
  ∀ (os : List (ℕ × Col)) (M : T), (os.map Prod.fst).Nodup → nMin M ≤ os.length →
    agree * os.length ≤ (os.filter fun p => decide (enc M p.1 = p.2)).length → M ∈ decode os

noncomputable def johnsonCount (k : ℕ) (agree : ℝ) : ℕ := ⌊((k : ℝ) + 1) / agree ^ 2⌋₊ + 1

theorem one_le_johnsonCount (k : ℕ) (agree : ℝ) : 1 ≤ johnsonCount k agree := Nat.le_add_left 1 _

theorem johnsonCount_le_iff {k N : ℕ} {agree : ℝ} (ha : 0 < agree) :
    johnsonCount k agree ≤ N ↔ Real.sqrt (((k : ℝ) + 1) * N) < agree * N := by
  have ha2 : 0 < agree ^ 2 := by positivity
  have hk : (0 : ℝ) ≤ (k : ℝ) + 1 := by positivity
  constructor
  · intro h
    have hN : ((k : ℝ) + 1) / agree ^ 2 < N := by
      have hl := Nat.lt_floor_add_one (((k : ℝ) + 1) / agree ^ 2)
      have h' : (johnsonCount k agree : ℝ) ≤ N := by exact_mod_cast h
      unfold johnsonCount at h'
      push_cast at h'
      linarith
    have hN0 : (0 : ℝ) < N := lt_of_le_of_lt (by positivity) hN
    rw [div_lt_iff₀ ha2] at hN
    rw [Real.sqrt_lt' (by positivity)]
    have := mul_lt_mul_of_pos_right hN hN0
    nlinarith
  · intro h
    have hN0 : (0 : ℝ) < N := by
      rcases (Nat.cast_nonneg (α := ℝ) N).lt_or_eq with h0 | h0
      · exact h0
      · rw [← h0] at h; simp at h
    have hsq := (Real.sqrt_lt' (by positivity)).1 h
    have hlt : ((k : ℝ) + 1) / agree ^ 2 < N := by
      rw [div_lt_iff₀ ha2]
      have : ((k : ℝ) + 1) * N < (N * agree ^ 2) * N := by nlinarith
      exact lt_of_mul_lt_mul_right this hN0.le
    have := (Nat.floor_lt (div_nonneg hk ha2.le)).2 hlt
    unfold johnsonCount
    omega

theorem johnsonCount_div (k a b : ℕ) : johnsonCount k ((a : ℝ) / b) = (k + 1) * b ^ 2 / a ^ 2 + 1 := by
  unfold johnsonCount
  congr 1
  rw [div_pow, div_div_eq_mul_div]
  have : ((k : ℝ) + 1) * (b : ℝ) ^ 2 / (a : ℝ) ^ 2 = (((k + 1) * b ^ 2 : ℕ) : ℝ) / ((a ^ 2 : ℕ) : ℝ) := by
    push_cast; ring
  rw [this, Nat.floor_div_eq_div]

theorem lt_johnsonCount {k : ℕ} {agree : ℝ} (ha0 : 0 < agree) (ha1 : agree ≤ 1) : k + 1 < johnsonCount k agree := by
  have ha2 : agree ^ 2 ≤ 1 := pow_le_one₀ ha0.le ha1
  have hle : ((k + 1 : ℕ) : ℝ) ≤ ((k : ℝ) + 1) / agree ^ 2 := by
    push_cast
    exact le_div_self (by positivity) (by positivity) ha2
  have := Nat.le_floor hle
  unfold johnsonCount
  omega

theorem card_filter_get {α : Type} (p : α → Prop) [DecidablePred p] :
    ∀ l : List α, (Finset.univ.filter fun j : Fin l.length => p (l.get j)).card =
      (l.filter fun a => decide (p a)).length
  | [] => rfl
  | a :: l => by
    change (Finset.univ.filter fun j : Fin (l.length + 1) => p ((a :: l).get j)).card = _
    rw [Fin.card_filter_univ_succ']
    simp only [List.get_eq_getElem, Fin.val_zero, List.getElem_cons_zero, Fin.val_succ,
      List.getElem_cons_succ, List.filter_cons]
    have ih := card_filter_get p l
    simp only [List.get_eq_getElem] at ih
    rw [ih]
    split_ifs with h <;> simp [h, add_comm]

/-- Join the candidate lists `L 0, L 1, …` row by row, keeping the prefixes `keep` accepts: the surviving prefixes of
`m` rows. -/
def extendRows {α : Type} (L : ℕ → List α) (keep : List α → Bool) : ℕ → List (List α)
  | 0 => [[]]
  | m + 1 => ((extendRows L keep m).flatMap fun ps => (L m).map fun p => ps ++ [p]).filter keep

theorem mem_extendRows {α : Type} (L : ℕ → List α) (keep : List α → Bool) (P : ℕ → α) (n : ℕ)
    (hL : ∀ m < n, P m ∈ L m) (hkeep : ∀ m ≤ n, keep ((List.range m).map P) = true) :
    ∀ m ≤ n, (List.range m).map P ∈ extendRows L keep m
  | 0, _ => by simp [extendRows]
  | m + 1, hm => by
    rw [extendRows, List.mem_filter, List.mem_flatMap]
    refine ⟨⟨(List.range m).map P, mem_extendRows L keep P n hL hkeep m (by omega), ?_⟩, hkeep (m + 1) hm⟩
    rw [List.range_succ, List.map_append, List.map_singleton]
    exact List.mem_map.2 ⟨P m, hL m (by omega), rfl⟩

/-- An interleaved Reed–Solomon code with `s` rows over `F`: a message `M : T` is `s` row polynomials of degree below
`k` (`rows`), which `ofRows` reads back, and its column at position `i` is their values at the point `dom i`. -/
structure IRS (F T : Type) [Field F] (s : ℕ) where
  k : ℕ
  dom : ℕ → F
  rows : T → Fin s → F[X]
  natDegree_rows : ∀ M r, (rows M r).natDegree < k
  ofRows : (Fin s → F[X]) → T
  ofRows_rows : ∀ M, ofRows (rows M) = M

namespace IRS

/-- `M`'s column at position `i`. -/
def enc {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (M : T) (i : ℕ) : Fin s → F :=
  fun r => (C.rows M r).eval (C.dom i)

def agreeCount {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (M : T)
    (os : List (ℕ × (Fin s → F))) : ℕ :=
  (os.filter fun p => decide (C.enc M p.1 = p.2)).length

noncomputable def fits {F : Type} [Field F] [DecidableEq F] {s : ℕ} (ps : List F[X]) (x : F) (c : Fin s → F) : Bool :=
  decide (∀ r : Fin s, r.val < ps.length → (ps.getD r 0).eval x = c r)

noncomputable def jointCount {F : Type} [Field F] [DecidableEq F] {s : ℕ} (dom : ℕ → F) (ps : List F[X])
    (os : List (ℕ × (Fin s → F))) : ℕ :=
  (os.filter fun p => fits ps (dom p.1) p.2).length

def points {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (os : List (ℕ × (Fin s → F)))
    (hnd : (os.map fun p => C.dom p.1).Nodup) : Fin os.length ↪ F :=
  ⟨fun j => C.dom (os.get j).1, fun j j' h => by
    have := (List.Nodup.getElem_inj_iff hnd (i := j.val) (j := j'.val) (hi := by simp) (hj := by simp)).1
      (by simpa using h)
    exact Fin.ext this⟩

def rowWord {F : Type} {s : ℕ} (os : List (ℕ × (Fin s → F))) (r : Fin s) : Fin os.length → F :=
  fun j => (os.get j).2 r

noncomputable def rowList {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (e r : ℕ) : List F[X] :=
  if h : r < s then GuruswamiSudan.decoder C.k 0 0 e (C.points os hnd) (rowWord os ⟨r, h⟩) else []

/-- The decoding threshold `⌈agree·N⌉` at `N` opened columns. -/
noncomputable def threshold {F : Type} {s : ℕ} (agree : ℝ) (os : List (ℕ × (Fin s → F))) : ℕ :=
  ⌈agree * os.length⌉₊

noncomputable def gsPath {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (agree : ℝ)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) : List T :=
  (extendRows (C.rowList os hnd (os.length - threshold agree os))
    (fun ps => decide (threshold agree os ≤ jointCount C.dom ps os)) s).map
    fun ps => C.ofRows fun r => ps.getD r 0

def Regime {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (agree : ℝ) (os : List (ℕ × (Fin s → F))) : Prop :=
  (os.map fun p => C.dom p.1).Nodup ∧
    (((os.length - threshold agree os : ℕ) : ℝ) < os.length - Real.sqrt ((C.k + 1) * os.length))

noncomputable def nMin {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (agree : ℝ) : ℕ := johnsonCount C.k agree

open Classical in
noncomputable def exhaustive {F T : Type} [Field F] [DecidableEq F] [Fintype T] {s : ℕ} (C : IRS F T s)
    (agree : ℝ) (os : List (ℕ × (Fin s → F))) : List T :=
  (Finset.univ.filter fun M => agree * os.length ≤ C.agreeCount M os).toList

open Classical in
noncomputable def decode {F T : Type} [Field F] [DecidableEq F] [Fintype T] {s : ℕ} (C : IRS F T s)
    (agree : ℝ) (os : List (ℕ × (Fin s → F))) : List T :=
  if h : C.Regime agree os then C.gsPath agree os h.1
  else if (os.map fun p => C.dom p.1).Nodup ∨ os.length < C.nMin agree then [] else C.exhaustive agree os

theorem hammingDist_rowWord_le {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (M : T) (r : Fin s) :
    hammingDist (rowWord os r) ((C.rows M r).eval ∘ C.points os hnd) + C.agreeCount M os ≤ os.length := by
  have hsub : hammingDist (rowWord os r) ((C.rows M r).eval ∘ C.points os hnd) ≤
      (Finset.univ.filter fun j : Fin os.length => ¬ (C.enc M (os.get j).1 = (os.get j).2)).card := by
    unfold hammingDist
    refine Finset.card_le_card fun j hj => ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj ⊢
    intro he
    apply hj
    have := congrFun he r
    simp only [enc] at this
    simp only [rowWord, Function.comp_apply, ← this]
    rfl
  rw [card_filter_get (fun a : ℕ × (Fin s → F) => ¬ (C.enc M a.1 = a.2)) os] at hsub
  have hsplit := List.length_eq_length_filter_add (l := os) (fun p => decide (C.enc M p.1 = p.2))
  unfold agreeCount
  simp only [decide_not] at hsub
  omega

theorem mem_rowList {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (e : ℕ)
    (he : (e : ℝ) < os.length - Real.sqrt ((C.k + 1) * os.length)) (M : T)
    (hM : os.length ≤ e + C.agreeCount M os) (r : Fin s) :
    C.rows M r ∈ C.rowList os hnd e r := by
  rw [rowList, dite_eq_left r.isLt, Fin.eta]
  refine GuruswamiSudan.mem_decoder_of_dist he (C.natDegree_rows M r) ?_
  have := C.hammingDist_rowWord_le os hnd M r
  omega

theorem agreeCount_le_jointCount {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s)
    (os : List (ℕ × (Fin s → F))) (M : T) (m : ℕ) :
    C.agreeCount M os ≤
      jointCount C.dom ((List.range m).map fun i => if h : i < s then C.rows M ⟨i, h⟩ else 0) os := by
  refine (List.monotone_filter_right os fun p hp => ?_).length_le
  simp only [decide_eq_true_eq] at hp
  simp only [fits, decide_eq_true_eq, List.length_map, List.length_range]
  intro r hr
  rw [List.getD_eq_getElem _ _ (by simpa using hr), List.getElem_map, List.getElem_range, dite_eq_left r.isLt,
    ← hp]
  rfl

theorem mem_gsPath {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (agree : ℝ)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup)
    (hJ : (((os.length - threshold agree os : ℕ) : ℝ) < os.length - Real.sqrt ((C.k + 1) * os.length)))
    (M : T) (hM : agree * os.length ≤ C.agreeCount M os) : M ∈ C.gsPath agree os hnd := by
  have ht : threshold agree os ≤ C.agreeCount M os := Nat.ceil_le.2 hM
  set P : ℕ → F[X] := fun i => if h : i < s then C.rows M ⟨i, h⟩ else 0 with hP
  have hpre := mem_extendRows (C.rowList os hnd (os.length - threshold agree os))
    (fun ps => decide (threshold agree os ≤ jointCount C.dom ps os)) P s
    (fun m hm => by
      have := C.mem_rowList os hnd _ hJ M (by have := C.agreeCount M os; omega) ⟨m, hm⟩
      simpa [hP, hm] using this)
    (fun m _ => decide_eq_true (ht.trans (C.agreeCount_le_jointCount os M m))) s le_rfl
  refine List.mem_map.2 ⟨_, hpre, ?_⟩
  conv_rhs => rw [← C.ofRows_rows M]
  congr 1
  funext r
  rw [List.getD_eq_getElem _ _ (by simp), List.getElem_map, List.getElem_range, hP]
  exact dite_eq_left r.isLt

theorem decode_eq_gsPath {F T : Type} [Field F] [DecidableEq F] [Fintype T] {s : ℕ} (C : IRS F T s)
    (agree : ℝ) (os : List (ℕ × (Fin s → F))) (h : C.Regime agree os) :
    C.decode agree os = C.gsPath agree os h.1 := by
  rw [decode, dite_eq_left h]

theorem regime_of_johnson {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (agree : ℝ)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (ha : agree ≤ 1)
    (hJ : Real.sqrt ((C.k + 1) * os.length) < agree * os.length) : C.Regime agree os := by
  refine ⟨hnd, ?_⟩
  have hN : (0 : ℝ) ≤ os.length := Nat.cast_nonneg _
  have htN : threshold agree os ≤ os.length :=
    Nat.ceil_le.2 (by nlinarith [mul_le_mul_of_nonneg_right ha hN])
  rw [Nat.cast_sub htN]
  have := Nat.le_ceil (agree * os.length)
  unfold threshold
  linarith

theorem regime_of_nMin_le {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (agree : ℝ)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (ha0 : 0 < agree) (ha1 : agree ≤ 1)
    (hN : C.nMin agree ≤ os.length) : C.Regime agree os :=
  C.regime_of_johnson agree os hnd ha1 ((johnsonCount_le_iff ha0).1 hN)

theorem decode_eq_nil {F T : Type} [Field F] [DecidableEq F] [Fintype T] {s : ℕ} (C : IRS F T s)
    (agree : ℝ) (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (h : ¬ C.Regime agree os) :
    C.decode agree os = [] := by
  rw [decode, dite_eq_right h, ite_eq_left (Or.inl hnd)]

theorem listComplete_decode {F T : Type} [Field F] [DecidableEq F] [Fintype T] {s : ℕ} (C : IRS F T s)
    (agree : ℝ) (ha : 0 < agree) : Flock.ListComplete C.enc agree (fun _ => C.nMin agree) (C.decode agree) := by
  intro os M _ hN hM
  unfold decode
  split_ifs with h h'
  · exact C.mem_gsPath agree os h.1 h.2 M hM
  · exfalso
    rcases h' with hnd | hlt
    · rcases le_or_gt agree 1 with ha1 | ha1
      · exact h (C.regime_of_nMin_le agree os hnd ha ha1 hN)
      · have hN1 : (1 : ℝ) ≤ os.length := by exact_mod_cast (one_le_johnsonCount C.k agree).trans hN
        have hc : ((os.filter fun p => decide (C.enc M p.1 = p.2)).length : ℝ) ≤ os.length := by
          exact_mod_cast List.length_filter_le _ _
        nlinarith
    · exact absurd hN (not_le.2 hlt)
  · classical
    unfold exhaustive
    rw [Finset.mem_toList, Finset.mem_filter]
    exact ⟨Finset.mem_univ M, hM⟩

end IRS

namespace Flock

theorem ListComplete.mem_of_agree {Col T : Type} [DecidableEq Col] {enc : T → ℕ → Col} {agree : ℝ} {nMin : T → ℕ}
    {decode : List (ℕ × Col) → List T} (h : ListComplete enc agree nMin decode) (ha : agree ≤ 1)
    (os : List (ℕ × Col)) (hnd : (os.map Prod.fst).Nodup) (M : T) (hN : nMin M ≤ os.length)
    (hM : ∀ q ∈ os, enc M q.1 = q.2) : M ∈ decode os := by
  refine h os M hnd hN ?_
  rw [List.filter_eq_self.2 fun q hq => decide_eq_true (hM q hq)]
  exact mul_le_of_le_one_left (Nat.cast_nonneg _) ha

theorem ListComplete.card_le_eval_of_agree {Col T : Type} [DecidableEq Col] {enc : T → ℕ → Col} {agree : ℝ}
    {nMin : T → ℕ} {decode : List (ℕ × Col) → List T} (h : ListComplete enc agree nMin decode) (ha : agree ≤ 1)
    {counted : List (ℕ × Col) → List T × ℕ} {size : List (ℕ × Col) → ℕ} {p : Polynomial ℕ}
    (hd : DecoderComplexity decode counted size p) (os : List (ℕ × Col)) (hnd : (os.map Prod.fst).Nodup)
    (hc : (decode os).length ≤ (counted os).2) (S : Finset T)
    (hS : ∀ M ∈ S, nMin M ≤ os.length ∧ ∀ q ∈ os, enc M q.1 = q.2) : S.card ≤ p.eval (size os) := by
  classical
  calc S.card ≤ (decode os).toFinset.card := Finset.card_le_card fun M hM =>
        List.mem_toFinset.2 (h.mem_of_agree ha os hnd M (hS M hM).1 (hS M hM).2)
    _ ≤ (decode os).length := List.toFinset_card_le _
    _ ≤ (counted os).2 := hc
    _ ≤ p.eval (size os) := (hd os).2

theorem ListComplete.mem_nil {Col T : Type} [DecidableEq Col] {enc : T → ℕ → Col} {agree : ℝ}
    {decode : List (ℕ × Col) → List T} (h : ListComplete enc agree (fun _ => 0) decode) (M : T) : M ∈ decode [] :=
  h [] M List.nodup_nil le_rfl (by simp)

theorem ListComplete.card_le_eval {Col T : Type} [DecidableEq Col] [Fintype T] {enc : T → ℕ → Col} {agree : ℝ}
    {decode : List (ℕ × Col) → List T} (h : ListComplete enc agree (fun _ => 0) decode)
    {counted : List (ℕ × Col) → List T × ℕ} {size : List (ℕ × Col) → ℕ} {p : Polynomial ℕ}
    (hd : DecoderComplexity decode counted size p) (hc : (decode []).length ≤ (counted []).2) :
    Fintype.card T ≤ p.eval (size []) := by
  classical
  calc Fintype.card T = (Finset.univ : Finset T).card := Finset.card_univ.symm
    _ ≤ (decode []).toFinset.card := Finset.card_le_card fun M _ => List.mem_toFinset.2 (h.mem_nil M)
    _ ≤ (decode []).length := List.toFinset_card_le _
    _ ≤ (counted []).2 := hc
    _ ≤ p.eval (size []) := (hd []).2

end Flock

end Claims
