module

public import Proofs.Accounting.ReadUnit.Gates
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

/-!
# The read's net computes the read's reference, given its parts

`readNet_sound`: where the four parts' nets meet their statements (`Gates.RowSegSpec`, `Gates.Hm96LeafSpec`,
`Gates.FrameReadSpec`, `Gates.SelectSpec`), the read's net (`Gates.readNet`) returns `Hash.EmbeddingRead.h` of what its
inputs hold, and its verdict is set only where `Hash.EmbeddingRead.ok` holds of them at SHA-512, in the table's dense
domain. The frame read climbs at the token's low `D` bits, which are the token itself below the table's `V ≤ 2 ^ D` rows,
and the climb reads only a domain's id and positions, not its count.
-/

@[expose] public section

namespace Accounting.ReadUnit.Gates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

theorem num_append : ∀ l₁ l₂ : List Bool, num (l₁ ++ l₂) = num l₁ + 2 ^ l₁.length * num l₂
  | [], l₂ => by simp [num]
  | b :: bs, l₂ => by
    simp only [List.cons_append, num, num_append bs l₂, List.length_cons, pow_succ]
    ring

theorem num_lt : ∀ l : List Bool, num l < 2 ^ l.length
  | [] => by simp [num]
  | b :: bs => by
    have := num_lt bs
    simp only [num, List.length_cons, pow_succ]
    cases b <;> simp only [Bool.toNat_false, Bool.toNat_true] <;> omega

/-- A number below `2 ^ k` is its low `k` bits'. -/
theorem num_take_of_lt (l : List Bool) (k : ℕ) (h : num l < 2 ^ k) : num (l.take k) = num l := by
  by_cases hk : k ≤ l.length
  · have hsplit := num_append (l.take k) (l.drop k)
    rw [List.take_append_drop, List.length_take_of_le hk] at hsplit
    have hdrop : num (l.drop k) = 0 := by
      by_contra hne
      have hle : 2 ^ k ≤ 2 ^ k * num (l.drop k) := Nat.le_mul_of_pos_right _ (Nat.pos_of_ne_zero hne)
      generalize 2 ^ k * num (l.drop k) = t at hsplit hle
      omega
    rw [hdrop, mul_zero, add_zero] at hsplit
    exact hsplit.symm
  · rw [List.take_of_length_le (by omega)]

theorem evs_take (z : Var → Bool) (l : List Lin) (k : ℕ) : evs z (l.take k) = (evs z l).take k := by
  simp [evs, List.map_take]

theorem length_evs (z : Var → Bool) (l : List Lin) : (evs z l).length = l.length := by
  simp [evs]

theorem length_bytesOf (bs : List Bool) : (bytesOf bs).length = bs.length / 64 * 8 := by
  simp [bytesOf, List.length_flatMap, List.map_const', List.sum_replicate, smul_eq_mul]

theorem length_flatten_of_shaped {k : ℕ} {ws : List Word} (h : Shaped k ws) : ws.flatten.length = k * 64 := by
  obtain ⟨hk, hw⟩ := h
  subst hk
  induction ws with
  | nil => simp
  | cons w ws ih =>
    simp only [List.flatten_cons, List.length_append, List.length_cons, hw w List.mem_cons_self,
      ih fun v hv => hw v (List.mem_cons_of_mem _ hv)]
    ring

theorem shaped_drop {k : ℕ} {ws : List Word} (h : Shaped k ws) (j : ℕ) : Shaped (k - j) (ws.drop j) :=
  ⟨by rw [List.length_drop, h.1], fun w hw => h.2 w (List.mem_of_mem_drop hw)⟩

/-- The climb reads a domain's id, not its count. -/
theorem climb_count (H : Hash.Msg → Verity.Digest) (id : Verity.Digest) (c₁ c₂ : ℕ) (p : ℕ → ℕ)
    (sibs : List Verity.Digest) : ∀ (lv cur : ℕ) (x : Verity.Digest),
      Hash.Tree.climb H ⟨id, c₁, p⟩ lv cur x sibs = Hash.Tree.climb H ⟨id, c₂, p⟩ lv cur x sibs := by
  induction sibs with
  | nil => intro lv cur x; rfl
  | cons s ss ih => intro lv cur x; simp only [Hash.Tree.climb, Hash.Tree.nodeMsg, ih]

/-- The opening the read's witness forms hold under `z`: the row, the salt's bytes and the siblings' digests. -/
def openingOf (z : Var → Bool) (row : List Lin) (salt : List Word) (path : List (List Word)) : Hash.RowLeaf.Opening :=
  ⟨evs z row, wordsBytes z salt, path.map (wordsDigest z)⟩

/-- **The read's net computes the read's reference**, given its four parts: for a table of `V ≤ 2 ^ D` rows of `H`
16-bit words with `D` its tree's depth, whenever the net's rows hold, the row it returns is `Hash.EmbeddingRead.h` of
the token and the witness row its inputs hold, and its verdict is set only where `Hash.EmbeddingRead.ok` holds of them
at SHA-512, under the hm96 key `key`, in the dense domain of `V` rows under the domain id its inputs hold. -/
theorem readNet_sound (key : Hash.Msg) {V H D : ℕ} (hD : D = Hash.Tree.depth V) (hVD : V ≤ 2 ^ D) (hD32 : D ≤ 32)
    (hH : 16 * H < 2 ^ 64) {rowSeg : List Lin → M (List Word)} {leaf : List Word → List Word → M (List Word)}
    {frame : List Word → List Word → List Word → List Lin → List (List Word) → M Lin}
    {select : List Lin → Lin → List Lin → M (List Lin × Lin)} (hA : RowSegSpec (16 * H) rowSeg)
    (hB : Hm96LeafSpec key leaf) (hC : FrameReadSpec D Hash.RowLeaf.Format.seg.schema frame)
    (hS : SelectSpec V select) (tok : List Lin) (root domain : List Word) (row : List Lin) (salt : List Word)
    (path : List (List Word)) :
    Sound (readNet D rowSeg leaf frame select tok root domain row salt path) fun z out =>
      tok.length = 32 → Shaped 8 root → Shaped 8 domain → row.length = 16 * H → Shaped 24 salt → path.length = D →
      (∀ s ∈ path, Shaped 8 s) →
      evs z out.1 = Hash.EmbeddingRead.h V H (num (evs z tok)) (openingOf z row salt path) ∧
        (ev z out.2 = true → Hash.EmbeddingRead.ok Verity.sha512 Verity.sha512 ⟨wordsDigest z domain, V, fun r => r⟩
          key (wordsDigest z root) V (num (evs z tok)) (openingOf z row salt path)) := by
  unfold readNet
  refine Sound.bind' (hA row) fun x => Sound.bind' (hB x salt) fun bcl =>
    Sound.bind' (hC root domain (bcl.drop 16) (tok.take D) path) fun opened =>
      Sound.mono (hS tok opened row) fun z out hSz hCz hBz hAz htok hroot hdom hrow hsalt hpath hpath' => ?_
  obtain ⟨hx, hxd⟩ := hAz hrow
  obtain ⟨hbcl, hbd⟩ := hBz hx hsalt
  have hrow16 : row.length % 16 = 0 := by rw [hrow]; exact Nat.mul_mod_right 16 H
  obtain ⟨hh, hok⟩ := hSz htok hrow16
  refine ⟨?_, fun hout => ?_⟩
  · rw [hh, hrow, Nat.mul_div_cancel_left H (by norm_num : 0 < 16)]
    rfl
  · by_cases hlt : num (evs z tok) < V
    · right
      have hop : ev z opened = true := by
        rw [hok] at hout
        simpa [Nat.not_le.mpr hlt] using hout
      have hpos : (tok.take D).length = D := by rw [List.length_take, htok]; exact min_eq_left hD32
      have hcl := hCz hroot hdom (shaped_drop hbcl 16) hpos hpath hpath' hop
      have hrank : num (evs z (tok.take D)) = num (evs z tok) := by
        rw [evs_take]
        exact num_take_of_lt _ _ (lt_of_lt_of_le hlt hVD)
      rw [hrank, hbd, hxd] at hcl
      refine ⟨?_, ?_, hlt, ?_, ?_⟩
      · show (evs z row).length < 2 ^ 64
        rw [length_evs, hrow]
        exact hH
      · show (wordsBytes z salt).length = Hash.Hm96.SALT_BYTES
        rw [wordsBytes, length_bytesOf, length_evs, length_flatten_of_shaped hsalt]
        rfl
      · show (path.map (wordsDigest z)).length = Hash.Tree.depth V
        rw [List.length_map, hpath, hD]
      · show (Hash.Tree.climb Verity.sha512 ⟨wordsDigest z domain, V, fun r => r⟩ 0 (num (evs z tok))
            (Verity.sha512 (Hash.Tree.leafMsg (denseDomain D (wordsDigest z domain)) (num (evs z tok))
              Hash.RowLeaf.Format.seg.schema (Hash.Hm96.leaf Verity.sha512 key
                (Hash.RowSeg.digest Verity.sha512 (evs z row)) (wordsBytes z salt)).bytes))
            (path.map (wordsDigest z))).2 = wordsDigest z root
        rw [climb_count Verity.sha512 (wordsDigest z domain) V (2 ^ D) (fun r => r)]
        exact hcl
    · left
      omega

end Accounting.ReadUnit.Gates
