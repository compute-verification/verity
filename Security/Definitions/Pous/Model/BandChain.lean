import Definitions.Pous.Model.P3Chain
import Definitions.Pous.Model.Dense

/-!
# The band scheme and the dense chain scheme over the overwrite chain on `Π₂`

The definitions the band pins read (`BandChainMeets64`, `BandChainMeets64D5`, `BandChainMeets64D12`,
`BandMultiMeets14`, `BandMultiMeets14K106`, `ChainDenseMeets64`), copied verbatim from the proofs' source modules
(the like-named sections of `Security/Proofs/Pous/Chain/Flat.lean` hold the rest of each), with
each source namespace `PousX` renamed `Pous.X` (the `Pous.Dense` and `Definitions.Pous.Model.P3Chain` precedent). The
overwrite chain itself (`Pous.Sponge.{iv, padBlk, parentsDesc, ovR, ovH, ovStep, ovAbsorb, ovKey}`) is reused from
`Definitions.Pous.Model.P3Chain`, whose copies are of the same source.
- `Pous.Sponge` (`SpongeDense`): the ideal-permutation model
  `idealPerm` (one uniform permutation `Π₂`, queried both ways), the domain `dom salt i = salt ‖ bin(i)`, and the dense
  scheme over the chain, `chainDense`, on the complete DAG;
- `Pous.DenseReCert` (`DenseReCert`): the segment tag `salt ‖ u64(s)`;
- `Pous.BandChain` (`BandChainModel`): the band DAG `bandDAG B d` (parents `v − d, …, v − 1`), the
  labels over the chain on any DAG, `labelG`, and the band scheme `chainBand`;
- `Pous.BandMulti` (`BandMultiModel`): the `N`-segment band DAG, the per-segment domains `segDom`, and
  the multi-segment scheme `chainBandSeg`.

The public-encoder definitions (`PubMeets`, `exactCheck`) are not here: they are in `Definitions.Pous.Model.PubEncoder`
and `Definitions.Pous.Accounting.PubMeets`.

**The schemes.** All run in `idealPerm (ℓ + (ℓ + c))`, with `ℓ = 524288` (64 KiB labels) and capacity `c = 512`.
Label `v` is `C_v = key_Π₂(IV(dom v); C_par(v) newest first, pad) ⊕ W_v`, where each call is `(r, h) = Π₂(B ‖ h)`: `r` is
dropped and `h` chained, and the key is the pad call's `r` (`ovKey`). The code is the labels; there is no `pp` and
there are no coins. Decoding recomputes each block's key from its parents in the code, with forward `Π₂` queries.

The definitions are total; the side conditions the proofs need are hypotheses of their theorems.
-/

namespace Pous.Sponge

open Pous Pous.Labelling Pous.Invertible
open scoped ENNReal

/-! ## `SpongeDense.lean`: the ideal-permutation model, the domains and the dense chain scheme -/

/-- **The ideal-permutation model** on `b`-bit states. Setup draws `Π` uniformly from every permutation of `Bits b`;
the encoder and the preprocessor see it whole. A query is `.inl x`, answered `Π x`, or `.inr y`, answered `Π⁻¹ y`. -/
def idealPerm (b : ℕ) : IdealModel where
  Ω := Equiv.Perm (Bits b)
  Qry := Bits b ⊕ Bits b
  Ans := Bits b
  answer π q := match q with
    | .inl x => π x
    | .inr y => π.symm y

/-- Domain separation: the 256-bit salt, then the block index in binary. -/
def dom (salt : Bits 256) (i : ℕ) : Bits 512 := Fin.append salt fun j : Fin 256 => i.testBit j

/-- Block `i`'s parents on the complete DAG, newest first: `C_{i−1}, …, C_0`. -/
def parentsNF {B ℓ : ℕ} (C : Fin B → Bits ℓ) (i : Fin B) : List (Bits ℓ) :=
  List.ofFn fun j : Fin i.val => C ⟨i.val - 1 - j.val, by omega⟩

/-- What block `i` absorbs: its parents newest-first, then the pad. That is `i + 1` calls. -/
def absorbed {B ℓ : ℕ} (C : Fin B → Bits ℓ) (i : Fin B) : List (Bits ℓ) := parentsNF C i ++ [padBlk]

/-- **The dense labels over the chain**: `C_i = key_Π₂(IV(dom i); C_{i−1}, …, C_0, pad) ⊕ W_i`. -/
def labelOv {B ℓ c : ℕ} (f : Bits (ℓ + (ℓ + c)) → Bits (ℓ + (ℓ + c))) (dom : ℕ → Bits c) (Wb : Fin B → Bits ℓ) :
    Fin B → Bits ℓ
  | i => bxor (ovKey f (dom i) ((List.ofFn fun j : Fin i.val =>
      labelOv f dom Wb ⟨i.val - 1 - j.val, by omega⟩) ++ [padBlk])) (Wb i)
  termination_by i => i.val
  decreasing_by omega

/-- **S1 over the overwrite chain**, in `idealPerm (ℓ + (ℓ + c))` (`Π₂` of width `2ℓ + c`). No `pp`, no coins, rate 1.
Decoding recomputes `W_i = C_i ⊕ key(IV(dom i); C_{i−1}, …, C_0, pad)` with `i + 1` forward queries. -/
def chainDense (B ℓ c : ℕ) [NeZero B] (dom : ℕ → Bits c) : Scheme (idealPerm (ℓ + (ℓ + c))) where
  n := B * ℓ
  t := 0
  B := B
  ℓ := ℓ
  B_pos := Nat.pos_of_ne_zero (NeZero.ne B)
  Coins := Unit
  enc W π _ := (Fin.elim0,
    labelOv (fun x => (idealPerm (ℓ + (ℓ + c))).answer π (.inl x)) dom (Dense.blocks B ℓ W))
  dec _ C f := codeEquiv B ℓ fun i => bxor (C i) (ovKey (fun x => f (.inl x)) (dom i) (absorbed C i))

end Pous.Sponge

namespace Pous.DenseReCert

open Pous Pous.Labelling
open scoped ENNReal

/-! ## `DenseReCert.lean`: the segment tag -/

/-- The low 64 bits of `s` (`u64 s`), as in `tag_s = salt ‖ u64 s`. -/
def u64 (s : ℕ) : Bits 64 := fun j => s.testBit j.val

/-- The implementation's segment tag `tag_s = salt ‖ u64 s`: a 192-bit salt then `u64 s`, filling `chainDense`'s 256-bit
salt slot exactly. -/
def tag (salt : Bits 192) (s : ℕ) : Bits (192 + 64) := Fin.append salt (u64 s)

end Pous.DenseReCert

namespace Pous.BandChain

open Pous Pous.Labelling Pous.Sponge
open scoped ENNReal

/-! ## `BandChainModel.lean`: the band DAG and the band scheme -/

/-- **The band DAG** on `Fin B` with in-degree `d`: node `v`'s parents are `v − d, …, v − 1`, or every earlier node
when `v < d`. With `d ≥ B − 1` it is the complete DAG. -/
def bandDAG (B d : ℕ) : TopoDAG B where
  par v := Finset.univ.filter fun u => u < v ∧ v.val ≤ u.val + d
  par_lt _ _ hu := (Finset.mem_filter.1 hu).2.1

/-- **Labels over the overwrite chain on any DAG**: `C_v = key_Π₂(IV(dom v); C_par(v) newest first, pad) ⊕ W_v`,
the analogue of `labelOv` (complete DAG) and of `c1c` (P3's layer 1 without `P`). -/
def labelG {N ℓ c : ℕ} (G : TopoDAG N) (f : Bits (ℓ + (ℓ + c)) → Bits (ℓ + (ℓ + c))) (dom : ℕ → Bits c)
    (Wb : Fin N → Bits ℓ) : Fin N → Bits ℓ
  | v => bxor (ovKey f (dom v) (((parentsDesc G v).attach.map fun u => labelG G f dom Wb u.1) ++ [padBlk]))
      (Wb v)
  termination_by v => v.val
  decreasing_by exact G.par_lt v _ ((Finset.mem_sort _).1 u.2)

/-- What row `v` absorbs, given the code `C`: its parents' blocks newest first, then the pad (`ovCalls G v` calls). -/
def absorbedG {N ℓ : ℕ} (G : TopoDAG N) (C : Fin N → Bits ℓ) (v : Fin N) : List (Bits ℓ) :=
  (parentsDesc G v).map C ++ [padBlk]

/-- **The single-layer scheme over the overwrite chain on a DAG `G`**, in `idealPerm (ℓ + (ℓ + c))`: no `pp`, no
coins, rate 1. Decoding recomputes `W_v = C_v ⊕ key(IV(dom v); C_par(v), pad)` with `ovCalls G v` forward queries. -/
def chainScheme {B : ℕ} [NeZero B] (G : TopoDAG B) (ℓ c : ℕ) (dom : ℕ → Bits c) :
    Scheme (idealPerm (ℓ + (ℓ + c))) where
  n := B * ℓ
  t := 0
  B := B
  ℓ := ℓ
  B_pos := Nat.pos_of_ne_zero (NeZero.ne B)
  Coins := Unit
  enc W π _ := (Fin.elim0,
    labelG G (fun x => (idealPerm (ℓ + (ℓ + c))).answer π (.inl x)) dom (Dense.blocks B ℓ W))
  dec _ C f := codeEquiv B ℓ fun i => bxor (C i) (ovKey (fun x => f (.inl x)) (dom i) (absorbedG G C i))

/-- **The band scheme**: `chainScheme` on `bandDAG B d`. Block `v` decodes with `min(v, d) + 1` calls. -/
def chainBand (B d ℓ c : ℕ) [NeZero B] (dom : ℕ → Bits c) : Scheme (idealPerm (ℓ + (ℓ + c))) :=
  chainScheme (bandDAG B d) ℓ c dom

end Pous.BandChain

namespace Pous.BandMulti

open Pous Pous.Labelling Pous.Sponge Pous.BandChain
open scoped ENNReal

/-! ## `BandMultiModel.lean`: the multi-segment band DAG, domains and scheme -/

/-- **The 14 GB DAG**: `N` band segments of `B` blocks, on global indices. Node `v`'s parents are the nodes `u` of its
own segment (`u / B = v / B`) with `v − d ≤ u < v`. -/
def bandSegDAG (N B d : ℕ) : TopoDAG (N * B) where
  par v := Finset.univ.filter fun u => u.val / B = v.val / B ∧ u < v ∧ v.val ≤ u.val + d
  par_lt _ _ hu := (Finset.mem_filter.1 hu).2.2.1

/-- **The implementation's per-segment domains**: global block `v` is block `v mod B` of segment `v / B`, keyed by
`dom (tag_{v / B}) (v mod B)` with `tag_s = salt ‖ u64 s`. -/
def segDom (salt : Bits 192) (B : ℕ) (v : ℕ) : Bits 512 :=
  Pous.Sponge.dom (Pous.DenseReCert.tag salt (v / B)) (v % B)

/-- **The 14 GB band scheme**: `chainScheme` on `bandSegDAG N B d`. Block `v` decodes with `min(v mod B, d) + 1`
sequential `Π₂` calls, within its own segment. -/
def chainBandSeg (N B d ℓ c : ℕ) [NeZero (N * B)] (dom : ℕ → Bits c) : Scheme (idealPerm (ℓ + (ℓ + c))) :=
  chainScheme (bandSegDAG N B d) ℓ c dom

instance instNeZeroHMulOfNatHPow_pous : NeZero (448 * 2 ^ 9) := ⟨by norm_num⟩

end Pous.BandMulti
