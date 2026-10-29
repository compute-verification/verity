import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Finset.Sort
import Definitions.Pous.Model.Scheme
import Definitions.Pous.Model.M1
import Definitions.Pous.Model.Labelling
import Definitions.Pous.Model.Invertible

/-!
# P3 with a concrete `H`: the overwrite chain on `Π₂`

The definitions the concrete-`H` P3 pins read (`Pous.Guarantees.P3ConcreteMeets64`, `P3ConcreteMeets14GB`,
`P3ConcreteMeets64D12`), copied verbatim from the proofs' sources. Each source namespace `PousX` becomes `Pous.X` (the
`Pous.Dense` precedent), so these copies never clash with the sources' own files, and a graded submission, with the
same renaming, states and proves the pins about these constants:
- `Pous.Sponge` (`lean/submissions/sponge-dense/SpongeDense.lean`, §§2, 7 and 9): the overwrite chain `H` on `Π₂`, the
  two-permutation model `p3ChainModel` (a tweakable `P` of labels, one independent uniform permutation per tweak, and
  an untweaked `Π₂` of `2m + 512` bits, `m = n·wc`), and the labels `c¹`, `c²`;
- `Pous.ColumnAware.basePath` and `Pous.P3Column.bandGraph` (`pebbling/ColumnAwareDraft.lean`,
  `p3-column/BandComb.lean`): the band DRG `B(n, k)`;
- `Pous.P3Meets` (`p3-meets/P3Meets64.lean`): the reference tag `salt ‖ u64(s)` and the packing of `W`;
- `Pous.P3Concrete` (`p3-concrete/P3ConcreteModel.lean`): the public decoder and the schemes `p3SchemeC` (one segment)
  and `p3SchemeNC` (`N` segments).

**The scheme.** Labels are `n = 220` components of `wc` bits. With `ω = (P, Π₂)` and `tag = salt ‖ u64(s)`:
- `H(tag, L, v; ps)` absorbs the parents' labels newest first, one `m`-bit block per call, then `padBlk`, from
  `h₀ = IV(domP3 tag L v)`. Each call is `(r, h) = Π₂(B ‖ h)`: `r` is dropped and `h` chained, and the key is the
  pad call's `r` (`ovKey`).
- The lower labels are `c¹_v = P_{(tag,0,v)}(W_v ⊕ H(tag, 0, v; c¹_{par v}))`, and the top labels are
  `c²_v = P_{(tag,1,v)}(transpose(c¹)_v ⊕ H(tag, 1, v; c²_{par v}))`. The code is the top labels.
- Decoding inverts `P` and recomputes `H` with forward `Π₂` queries.

The definitions are total; the side conditions the proofs need are hypotheses of their theorems.
-/

namespace Pous.Sponge

open Pous Pous.Labelling Pous.Invertible
open scoped ENNReal

section Sponge

variable {r c : ℕ}

/-- The initial state `IV(dom) = 0^r ‖ dom`: the domain (salt and index) is the whole capacity. -/
def iv (dom : Bits c) : Bits (r + c) := Fin.append (fun _ => false) dom

/-- The pad10*1 block for an input that is a whole number of blocks: first and last bit set. -/
def padBlk : Bits r := fun j => decide (j.val = 0 ∨ j.val + 1 = r)

end Sponge

/-- `P`'s tweak: the (salt, segment) tag, the layer and the node. -/
abbrev Tweak (n : ℕ) : Type := Bits 256 × Fin 2 × Fin n

/-- The key domain of node `v` in layer `L`: the tag, then `2v + L` in binary. -/
def domP3 (tag : Bits 256) (L : Fin 2) (v : ℕ) : Bits 512 :=
  Fin.append tag fun j : Fin 256 => (2 * v + L.val).testBit j

/-- A node's parents, newest first (decreasing index). -/
def parentsDesc {n : ℕ} (G : TopoDAG n) (v : Fin n) : List (Fin n) := (G.par v).sort (fun a b => b ≤ a)

section Overwrite

variable {r c : ℕ}

/-- The dropped part of a `Π₂` output: its first `r` bits. -/
def ovR (o : Bits (r + (r + c))) : Bits r := fun j => o (Fin.castAdd (r + c) j)

/-- The chained part of a `Π₂` output: its last `r + c` bits. -/
def ovH (o : Bits (r + (r + c))) : Bits (r + c) := fun j => o (Fin.natAdd r j)

/-- One overwrite call on the chaining value `p.2`: `Π₂(blk ‖ h)`, split as `(r, h')`; the previous `r` is unused. -/
def ovStep (f : Bits (r + (r + c)) → Bits (r + (r + c))) (p : Bits r × Bits (r + c)) (blk : Bits r) :
    Bits r × Bits (r + c) :=
  (ovR (f (Fin.append blk p.2)), ovH (f (Fin.append blk p.2)))

/-- Absorb `blks` from the chaining value `h₀`: the last call's `(r, h)`. -/
def ovAbsorb (f : Bits (r + (r + c)) → Bits (r + (r + c))) (h₀ : Bits (r + c)) (blks : List (Bits r)) :
    Bits r × Bits (r + c) :=
  blks.foldl (ovStep f) (fun _ => false, h₀)

/-- **The overwrite key**: `r` of the last call (the pad call), from `h_0 = IV(dom)`. -/
def ovKey (f : Bits (r + (r + c)) → Bits (r + (r + c))) (dom : Bits c) (blks : List (Bits r)) : Bits r :=
  (ovAbsorb f (iv dom) blks).1

end Overwrite

/-- P3's primitives over the chain: a tweakable `P` of labels, and `Π₂` of `2m + 512` bits (`m = n·wc`). -/
abbrev P3COmega (n wc : ℕ) : Type :=
  (Tweak n → Equiv.Perm (Lab n wc)) × Equiv.Perm (Bits (n * wc + (n * wc + 512)))

/-- `P3COmega` is finite. The source declares this, and `instNonemptyP3COmega`, as global instances at this point of
`SpongeDense`. Here they are a definition and a theorem under the same names, local instances for `p3ChainModel` only:
as imported global instances they would be tried, before the sources' own instances, on every product type of this
shape, and on `SpongeDense`'s `P3SOmega` that unification exceeds the recursion limit. -/
@[instance_reducible]
noncomputable def instFintypeP3COmega (n wc : ℕ) : Fintype (P3COmega n wc) := Fintype.ofFinite _
theorem instNonemptyP3COmega (n wc : ℕ) : Nonempty (P3COmega n wc) := ⟨⟨fun _ => 1, 1⟩⟩

attribute [local instance] instFintypeP3COmega instNonemptyP3COmega

/-- **The P3 model over the chain**: queries `Π₂`, `Π₂⁻¹`, `P_T`, `P_T⁻¹`. -/
noncomputable def p3ChainModel (n wc : ℕ) : IdealModel where
  Ω := P3COmega n wc
  Qry := (Bits (n * wc + (n * wc + 512)) ⊕ Bits (n * wc + (n * wc + 512))) ⊕ (Tweak n × (Lab n wc ⊕ Lab n wc))
  Ans := Bits (n * wc + (n * wc + 512)) ⊕ Lab n wc
  answer ω q := match q with
    | .inl (.inl x) => .inl (ω.2 x)
    | .inl (.inr y) => .inl (ω.2.symm y)
    | .inr (T, .inl a) => .inr (ω.1 T a)
    | .inr (T, .inr z) => .inr ((ω.1 T).symm z)

/-- `H` over the chain: `key_Π₂(IV(tag, L, v); ps newest first, pad)`, each label one `m`-bit block. -/
def keyP3c {n wc : ℕ} (f : Bits (n * wc + (n * wc + 512)) → Bits (n * wc + (n * wc + 512))) (tag : Bits 256)
    (L : Fin 2) (v : Fin n) (ps : List (Lab n wc)) : Lab n wc :=
  (codeEquiv n wc).symm (ovKey f (domP3 tag L v) (ps.map (codeEquiv n wc) ++ [padBlk]))

/-- Layer-1 labels over the chain. -/
def c1c {n wc : ℕ} (tag : Bits 256) (G : TopoDAG n) (f : Bits (n * wc + (n * wc + 512)) → Bits (n * wc + (n * wc + 512)))
    (P : Tweak n → Equiv.Perm (Lab n wc)) (W : Fin n → Lab n wc) : Fin n → Lab n wc
  | v => P (tag, 0, v) (labXor (W v)
      (keyP3c f tag 0 v ((parentsDesc G v).attach.map fun u => c1c tag G f P W u.1)))
  termination_by v => v.val
  decreasing_by exact G.par_lt v _ ((Finset.mem_sort _).1 u.2)

/-- Layer-2 (top) labels over the chain. -/
def c2c {n wc : ℕ} (tag : Bits 256) (G : TopoDAG n) (f : Bits (n * wc + (n * wc + 512)) → Bits (n * wc + (n * wc + 512)))
    (P : Tweak n → Equiv.Perm (Lab n wc)) (W : Fin n → Lab n wc) : Fin n → Lab n wc
  | v => P (tag, 1, v) (labXor (transpose (c1c tag G f P W) v)
      (keyP3c f tag 1 v ((parentsDesc G v).attach.map fun u => c2c tag G f P W u.1)))
  termination_by v => v.val
  decreasing_by exact G.par_lt v _ ((Finset.mem_sort _).1 u.2)

/-- The top label of node `v` over the chain, as a function of `ω = (P, Π₂)`. -/
def topLabelC {n wc : ℕ} (tag : Bits 256) (G : TopoDAG n) (W : Fin n → Lab n wc) (ω : P3COmega n wc)
    (v : Fin n) : Lab n wc :=
  c2c tag G ω.2 ω.1 W v

end Pous.Sponge

namespace Pous.ColumnAware

open Pous Pous.Invertible
open scoped ENNReal

/-- A base-path DRG: node `v` has its spine parent `v − 1` plus given extra parents `extra v ⊆ {0,…,v−1}`. -/
def basePath (n : ℕ) (extra : Fin n → Finset (Fin n)) (hextra : ∀ v, ∀ u ∈ extra v, u < v) : Labelling.TopoDAG n where
  par v := Finset.univ.filter (fun u => (u : ℕ) + 1 = (v : ℕ)) ∪ extra v
  par_lt v u hu := by
    simp only [Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and] at hu
    rcases hu with hu | hu
    · show (u : ℕ) < (v : ℕ); omega
    · exact hextra v u hu

end Pous.ColumnAware

namespace Pous.P3Column

open Pous Pous.ColumnAware

/-- The extra parents of a band node: the `k` previous nodes other than the spine parent. -/
def bandExtra (n k : ℕ) (v : Fin n) : Finset (Fin n) :=
  Finset.univ.filter fun u => u.val + 1 < v.val ∧ v.val ≤ u.val + k

theorem bandExtra_lt (n k : ℕ) : ∀ v, ∀ u ∈ bandExtra n k v, u < v := by
  intro v u hu
  simp only [bandExtra, Finset.mem_filter, Finset.mem_univ, true_and] at hu
  exact Fin.lt_def.2 (by omega)

/-- **The band graph `B(n, k)`**: node `v`'s parents are the `k` nodes before it. -/
abbrev bandGraph (n k : ℕ) : Labelling.TopoDAG n := basePath n (bandExtra n k) (bandExtra_lt n k)

end Pous.P3Column

namespace Pous.P3Meets

open Pous Pous.ColumnAware Pous.P3Column Pous.Invertible
open scoped ENNReal

/-- `tag_s = salt (24 B) ‖ u64(s)`, little-endian bits. -/
def tagR (salt : Bits 192) (s : ℕ) : Bits 256 :=
  fun i => if h : i.val < 192 then salt ⟨i.val, h⟩ else s.testBit (i.val - 192)

variable (wc : ℕ) (Gr : Labelling.TopoDAG 220)

/-- `W` as 220 lower inputs of `220·wc` bits. -/
def WlabW (W : Bits (220 * (220 * wc))) : Fin 220 → Lab 220 wc :=
  fun u => (codeEquiv 220 wc).symm ((codeEquiv 220 (220 * wc)).symm W u)

/-- The 220 lower inputs of `220·wc` bits as `W`. -/
def WbitsW (L : Fin 220 → Lab 220 wc) : Bits (220 * (220 * wc)) :=
  codeEquiv 220 (220 * wc) fun u => codeEquiv 220 wc (L u)

variable (N : ℕ) [NeZero N]

/-- Segment `s` of `W`, at width `wc`. -/
def WsegW (W : Bits (N * (220 * (220 * wc)))) (s : Fin N) : Bits (220 * (220 * wc)) :=
  (codeEquiv N (220 * (220 * wc))).symm W s

end Pous.P3Meets

namespace Pous.P3Concrete

open Pous Pous.Labelling Pous.Invertible Pous.Sponge Pous.P3Meets
open scoped ENNReal

section Access

variable {n wc : ℕ}

/-- `Π₂` forward through the oracle `f` (a label answer, which the real oracle never gives, is read as the input). -/
def pi2Of (f : (p3ChainModel n wc).Qry → (p3ChainModel n wc).Ans) (x : Bits (n * wc + (n * wc + 512))) :
    Bits (n * wc + (n * wc + 512)) :=
  match f (.inl (.inl x)) with
  | .inl y => y
  | .inr _ => x

/-- `P_T⁻¹` through the oracle `f`. -/
def pInvOf (f : (p3ChainModel n wc).Qry → (p3ChainModel n wc).Ans) (T : Tweak n) (z : Lab n wc) : Lab n wc :=
  match f (.inr (T, .inr z)) with
  | .inr a => a
  | .inl _ => z

end Access

/-- **Public decode over the chain**: `x²_v = P⁻¹_{(tag,1,v)}(c²_v) ⊕ H(tag,1,v; c²_par)`, transpose, then
`W_u = P⁻¹_{(tag,0,u)}(c¹_u) ⊕ H(tag,0,u; c¹_par)`, with `H` the overwrite chain through `f`. -/
def decodeC {n wc : ℕ} (tag : Bits 256) (G : TopoDAG n) (c2 : Fin n → Lab n wc)
    (f : (p3ChainModel n wc).Qry → (p3ChainModel n wc).Ans) : Fin n → Lab n wc :=
  let x2 : Fin n → Lab n wc := fun v =>
    labXor (pInvOf f (tag, 1, v) (c2 v)) (keyP3c (pi2Of f) tag 1 v ((parentsDesc G v).map c2))
  let c1 := transpose x2
  fun u => labXor (pInvOf f (tag, 0, u) (c1 u)) (keyP3c (pi2Of f) tag 0 u ((parentsDesc G u).map c1))

variable (wc : ℕ) (Gr : TopoDAG 220)

/-- **P3 over the chain, one segment on `Gr`, labels of `220·wc` bits**, 24-byte salt, segment-0 tag
`salt ‖ u64(0)`. -/
noncomputable def p3SchemeC : Scheme (p3ChainModel 220 wc) where
  n := 220 * (220 * wc)
  t := 192
  B := 220
  ℓ := 220 * wc
  B_pos := by norm_num
  Coins := Bits 192
  enc W ω salt := (salt, fun i => codeEquiv 220 wc (topLabelC (tagR salt 0) Gr (WlabW wc W) ω i))
  dec pp C f := WbitsW wc (decodeC (tagR pp 0) Gr (fun v => (codeEquiv 220 wc).symm (C v)) f)

variable (N : ℕ) [NeZero N]

/-- **P3 over the chain, `N` segments on `Gr`**, segment `s` keyed by `salt ‖ u64(s)`, segment-major in `W` and
`C`. -/
noncomputable def p3SchemeNC : Scheme (p3ChainModel 220 wc) where
  n := N * (220 * (220 * wc))
  t := 192
  B := N * 220
  ℓ := 220 * wc
  B_pos := Nat.mul_pos (Nat.pos_of_ne_zero (NeZero.ne N)) (by norm_num)
  Coins := Bits 192
  enc W ω salt := (salt, fun i => codeEquiv 220 wc (topLabelC (tagR salt (finProdFinEquiv.symm i).1.val) Gr
    (WlabW wc (WsegW wc N W (finProdFinEquiv.symm i).1)) ω (finProdFinEquiv.symm i).2))
  dec pp C f := codeEquiv N (220 * (220 * wc)) fun s =>
    WbitsW wc (decodeC (tagR pp s.val) Gr (fun v => (codeEquiv 220 wc).symm (C (finProdFinEquiv (s, v)))) f)

end Pous.P3Concrete
