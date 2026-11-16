/-
Part B's select: the comparator `Flock.ReadNet.below` is `tok < bound` on the token's unsigned value (`sound_below`), and
the select `Flock.ReadNet.select` (`EmbeddingReadSelect_v1{V, H}`) returns the row inside the vocabulary and the fill row
outside, its verdict `opened` inside and 1 outside (`select_spec`, `Gates.SelectSpec`).
-/
module

public import Flock.ReadNet.Select
public import Proofs.Accounting.ReadUnit.Gates
public import Proofs.Flock.Soundness.Discharge.Hm.Word
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
import Mathlib.Topology.Sheaves.Init

@[expose] public section

namespace Accounting.ReadUnit.Gates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

/-! ## The comparator -/

/-- One bit of the comparator: below `bound` on the bits up to `i` from below it on those under `i` and bit `i`. -/
theorem below_step (bound i N : ℕ) (hN : N < 2 ^ i) (b : Bool) :
    (if bound.testBit i then (!b || decide (N < bound % 2 ^ i)) else (!b && decide (N < bound % 2 ^ i))) =
      decide (N + 2 ^ i * b.toNat < bound % 2 ^ (i + 1)) := by
  have hmod : bound % 2 ^ (i + 1) = bound % 2 ^ i + 2 ^ i * (bound / 2 ^ i % 2) := by
    rw [Nat.pow_succ, Nat.mod_mul]
  have hR : bound % 2 ^ i < 2 ^ i := Nat.mod_lt _ (Nat.two_pow_pos i)
  rw [hmod, Nat.testBit_eq_decide_div_mod_eq]
  generalize bound % 2 ^ i = R at hR ⊢
  generalize 2 ^ i = P at hN hR ⊢
  rcases Nat.mod_two_eq_zero_or_one (bound / P) with h | h <;> rw [h] <;> cases b <;> simp <;> omega

theorem sound_belowStep (bound i : ℕ) (b lt : Lin) :
    Sound (if bound.testBit i then do
        let r ← Flock.HmNets.and b (lt ^^^ Lin.one)
        pure (r ^^^ Lin.one)
      else Flock.HmNets.and (b ^^^ Lin.one) lt)
      fun z r => ev z r = (if bound.testBit i then (!ev z b || ev z lt) else (!ev z b && ev z lt)) := by
  cases bound.testBit i
  · show Sound (Flock.HmNets.and (b ^^^ Lin.one) lt) fun z r => ev z r = (!ev z b && ev z lt)
    refine Sound.mono (sound_and _ _) fun z r hr => ?_
    rw [hr, ev_xor, ev_one]
    cases ev z b <;> cases ev z lt <;> rfl
  · show Sound (Flock.HmNets.and b (lt ^^^ Lin.one) >>= fun r => pure (r ^^^ Lin.one)) fun z r =>
      ev z r = (!ev z b || ev z lt)
    refine Sound.bind' (sound_and _ _) fun r => Sound.pure' fun z hr => ?_
    rw [ev_xor, hr, ev_xor, ev_one]
    cases ev z b <;> cases ev z lt <;> rfl

/-- `belowGo`'s step as one bind: `do` compiles the step's `if` with a join point, which `Sound.bind'` can't match. -/
theorem belowGo_cons (bound : ℕ) (b : Lin) (bs : List Lin) (i : ℕ) (lt : Lin) :
    Flock.ReadNet.belowGo bound (b :: bs) i lt =
      (if bound.testBit i then do
          let r ← Flock.HmNets.and b (lt ^^^ Lin.one)
          pure (r ^^^ Lin.one)
        else Flock.HmNets.and (b ^^^ Lin.one) lt) >>= fun lt' => Flock.ReadNet.belowGo bound bs (i + 1) lt' := by
  rw [Flock.ReadNet.belowGo]
  cases bound.testBit i <;> simp

/-- The comparator from bit `i` on: where the bits below `i` read `N` and `lt` says whether `N` is below `bound`'s low
bits, the result says whether the whole value is below `bound`'s low bits. -/
theorem sound_belowGo (bound : ℕ) : ∀ (bs : List Lin) (i : ℕ) (lt : Lin),
    Sound (Flock.ReadNet.belowGo bound bs i lt) fun z r => ∀ N : ℕ, N < 2 ^ i →
      ev z lt = decide (N < bound % 2 ^ i) →
        ev z r = decide (N + 2 ^ i * num (evs z bs) < bound % 2 ^ (i + bs.length))
  | [], i, lt => Sound.pure' fun z N _ h => by simp [num, evs, h]
  | b :: bs, i, lt => by
    rw [belowGo_cons]
    refine Sound.bind' (sound_belowStep bound i b lt) fun lt' =>
      Sound.mono (sound_belowGo bound bs (i + 1) lt') fun z r hih hstep => ?_
    intro N hN hlt
    have key : ev z lt' = decide (N + 2 ^ i * (ev z b).toNat < bound % 2 ^ (i + 1)) := by
      rw [hstep, hlt]
      exact below_step bound i N hN (ev z b)
    have hN' : N + 2 ^ i * (ev z b).toNat < 2 ^ (i + 1) := by
      rw [Nat.pow_succ]
      cases ev z b <;> simp <;> omega
    rw [hih _ hN' key]
    have e1 : N + 2 ^ i * (ev z b).toNat + 2 ^ (i + 1) * num (evs z bs) = N + 2 ^ i * num (evs z (b :: bs)) := by
      simp only [evs, List.map_cons, num, Nat.pow_succ]
      ring
    have e2 : i + 1 + bs.length = i + (b :: bs).length := by simp only [List.length_cons]; omega
    rw [e1, e2]

/-- **The comparator**: whenever its rows hold, `below`'s bit is whether the bits' unsigned value (least significant
first) is below `bound` modulo `2 ^ n`, `n` the bits' count. -/
theorem sound_below (bits : List Lin) (bound : ℕ) :
    Sound (Flock.ReadNet.below bits bound) fun z r => ev z r = decide (num (evs z bits) < bound % 2 ^ bits.length) :=
  Sound.mono (sound_belowGo bound bits 0 Lin.zero) fun z r h => by
    have := h 0 (by norm_num) (by simp; exact Nat.mod_one bound)
    simpa using this

/-! ## The select -/

theorem ev_fill (z : Var → Bool) (k : ℕ) :
    ((List.replicate k Flock.ReadNet.fillBits).flatten).map (ev z) = Hash.EmbeddingRead.fill k := by
  simp [List.map_flatten, List.map_replicate, Hash.EmbeddingRead.fill, Hash.EmbeddingRead.FILL_WORD,
    Flock.ReadNet.fillBits, Flock.ReadNet.FILL, List.map_map, Function.comp_def]

theorem length_fillBits : Flock.ReadNet.fillBits.length = 16 := by simp [Flock.ReadNet.fillBits]

theorem length_fill_flat : ∀ k : ℕ, ((List.replicate k Flock.ReadNet.fillBits).flatten).length = 16 * k
  | 0 => rfl
  | k + 1 => by
    rw [List.replicate_succ, List.flatten_cons, List.length_append, length_fill_flat k, length_fillBits]
    ring

theorem xor_and_sel (a b c : Bool) : (b ^^ (c && (a ^^ b))) = (if c then a else b) := by
  cases a <;> cases b <;> cases c <;> rfl

/-- **Part B, the select**: the select generator meets `Gates.SelectSpec` for a table of `V < 2 ^ 32` rows: on a 32-bit
token it returns the row inside the vocabulary and `V`'s fill row outside, and its verdict is `opened` inside and 1
outside. -/
theorem select_spec {V : ℕ} (hV : V < 2 ^ 32) : SelectSpec V (Flock.ReadNet.select V) := by
  intro tok opened row
  unfold Flock.ReadNet.select
  refine Sound.bind' (sound_below tok V) fun inside => ?_
  refine Sound.bind' (sound_mapM (fun rf : Lin × Lin => do
        let r ← Flock.HmNets.and inside (rf.1 ^^^ rf.2)
        pure (rf.2 ^^^ r))
      (fun z rf => (ev z rf.2 ^^ (ev z inside && (ev z rf.1 ^^ ev z rf.2))))
      (fun rf => Sound.bind' (sound_and _ _) fun r => Sound.pure' fun z hr => by rw [ev_xor, hr, ev_xor])
      (row.zip (List.replicate (row.length / 16) Flock.ReadNet.fillBits).flatten))
    fun h => ?_
  refine Sound.bind' (sound_and _ _) fun r => Sound.pure' fun z hr => ?_
  intro hh hin htok hrow
  obtain ⟨hlen, hbits⟩ := hh
  set ff := (List.replicate (row.length / 16) Flock.ReadNet.fillBits).flatten with hff
  have hffl : ff.length = row.length := by
    rw [hff, length_fill_flat]
    omega
  have hzl : (row.zip ff).length = row.length := by rw [List.length_zip, hffl, Nat.min_self]
  have hfill : ff.map (ev z) = Hash.EmbeddingRead.fill (row.length / 16) := by rw [hff]; exact ev_fill z _
  have hmod : V % 2 ^ tok.length = V := by rw [htok]; exact Nat.mod_eq_of_lt hV
  rw [hmod] at hin
  have hval : ∀ i (hi : i < row.length), ev z (h[i]'(by rw [hlen, hzl]; exact hi)) =
      if num (evs z tok) < V then ev z row[i] else ev z (ff[i]'(by rw [hffl]; exact hi)) := by
    intro i hi
    have hb := hbits i (by rw [hzl]; exact hi)
    have hi' : i < h.length := by rw [hlen, hzl]; exact hi
    simp only [FlockSoundness.Discharge.Hm.bit, List.getElem?_eq_getElem hi', Option.getD_some, List.getElem_zip] at hb
    rw [hb, xor_and_sel, hin]
    by_cases hlt : num (evs z tok) < V <;> simp [hlt]
  refine ⟨?_, ?_⟩
  · by_cases hlt : num (evs z tok) < V
    · rw [ite_eq_left hlt]
      apply List.ext_getElem
      · simp only [evs, List.length_map, hlen, hzl]
      · intro i h1 h2
        have hi : i < row.length := by simpa [evs, hlen, hzl] using h1
        simp only [evs, List.getElem_map]
        rw [hval i hi, ite_eq_left hlt]
    · rw [ite_eq_right hlt, ← hfill]
      apply List.ext_getElem
      · simp only [evs, List.length_map, hlen, hzl, hffl]
      · intro i h1 h2
        have hi : i < row.length := by simpa [evs, hlen, hzl] using h1
        simp only [evs, List.getElem_map]
        rw [hval i hi, ite_eq_right hlt]
  · show ev z (Lin.one ^^^ r) = _
    rw [ev_xor, ev_one, hr, ev_xor, ev_one, hin]
    by_cases hlt : num (evs z tok) < V
    · simp [hlt, Nat.not_le.2 hlt]
    · simp [hlt, Nat.le_of_not_lt hlt]

end Accounting.ReadUnit.Gates
