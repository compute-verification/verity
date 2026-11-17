module

public import Definitions.Pous.Model.StreamedWipe
public import Definitions.ResourceAccounting.Actual.Sampler
public import Definitions.Pouw.TS.Sampled
public import Definitions.Hash.RowLeaf
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
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

/-! PoUB's opening sample as the verifier runs it, with the coin commitment and the receipts' binding term; that
`audit.pooled`'s greedy order reaches `pooled`'s supremum is not stated. -/

@[expose] public section

namespace ResourceAccounting.Remote.Sample

open Finset
open scoped ENNReal

/-- A window's first blocks: sample `i`'s first SHA-512 block of its stream, read as its big-endian value. -/
abbrev Blocks (m : ℕ) : Type := Fin m → Pous.Bits 512

/-- The sampler's word width at `N`: `8 ⌈(bitlen N + 64) / 8⌉` bits (`Key._uniform`). -/
def width (N : ℕ) : ℕ := 8 * ((Nat.size N + 71) / 8)

/-- Word `j` of an `L`-bit block at `N`: the `width N` bits below its top `j·width N`, the stream's `width N / 8` bytes
from byte `j·width N / 8` on read big-endian, when `(j + 1)·width N ≤ L`. -/
def wordAt (N j : ℕ) {L : ℕ} (b : Pous.Bits L) : Pous.Bits (width N) :=
  (Pous.bitsEquiv (width N)).symm
    ⟨Pous.bitsVal b / 2 ^ (L - (j + 1) * width N) % 2 ^ width N, Nat.mod_lt _ (Nat.two_pow_pos _)⟩

/-- **The sampler on one block**, as deployed: the first of the block's `512 / width N` words it accepts, none when it
rejects them all. -/
def pick (N : ℕ) (b : Pous.Bits 512) : Option (Fin N) :=
  (List.range (512 / width N)).findSome? fun j => Actual.wordIndex N (wordAt N j b)

/-- A sample's offset at `N`: the sampler's on the block's words when it accepts one, else `r`. -/
def offset (N r : ℕ) (b : Pous.Bits 512) : ℕ := ((pick N b).map Fin.val).getD r

/-- The arrangement after `i` swaps, positions to receipts: step `l` swaps positions `l` and `l + j l`
(`swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)`). -/
def arrange (j : ℕ → ℕ) : ℕ → Equiv.Perm ℕ
  | 0 => 1
  | i + 1 => arrange j i * Equiv.swap i (i + j i)

/-- The offsets of a window of `U` receipts with first blocks `b` and fallbacks `r`: step `i` samples below `U − i`. -/
def offsets {m : ℕ} (U : ℕ) (b : Blocks m) (r : ℕ → ℕ) (i : ℕ) : ℕ :=
  if h : i < m then offset (U - i) (r i) (b ⟨i, h⟩) else r i

/-- The receipts a window of `U` opens with `k` samples: those at the arrangement's first `k` positions. -/
def openings {m : ℕ} (U k : ℕ) (b : Blocks m) (r : ℕ → ℕ) : Finset ℕ :=
  (range k).image (arrange (offsets U b r) k)

/-- Every one of the `k` samples of a window of `U` takes its offset from a word of its first block. -/
def accepted {m : ℕ} (U k : ℕ) (b : Blocks m) : Prop :=
  ∀ i : Fin m, (i : ℕ) < k → (pick (U - i) (b i)).isSome = true

/-- **The window test**: when the sampler accepts a word of each of the `k` samples' first blocks at `U`, whether the
openings lie in `C`; true when it rejects every word of one. -/
def test {m : ℕ} (U k : ℕ) (C : Finset ℕ) (b : Blocks m) : Prop :=
  accepted U k b → openings U k b (fun _ => 0) ⊆ C

/-- **The threshold coin**: `u` read as a number below `2^512` is below `⌊δ·2^512/P⌋`, or `P ≤ δ`. -/
def below (δ P : ℝ) (u : Pous.Bits 512) : Prop := P ≤ δ ∨ Pous.bitsVal u < ⌊δ * 2 ^ 512 / P⌋₊

/-- A session's coins: the other coins and window `t`'s openings coin. Not the live coins (`Verity.Coins`): each
openings coin seeds a sample without replacement over its window's own receipts, and the other coins hold the prover's.
-/
abbrev SampleCoins (n κ : ℕ) (X : Type) : Type := X × (Fin n → Pous.Bits κ)

/-- A session of `n` windows of at most `m` receipts, window `t` sampling `min(s_t, U_t)` of its `U_t`. -/
structure Run (n m : ℕ) (s : Fin n → ℕ) (κ : ℕ) (X : Type) where
  size : Fin n → SampleCoins n κ X → Fin (m + 1)
  correct : Fin n → SampleCoins n κ X → Finset ℕ

/-- `audit.pooled`'s budget: `S·(1 − δ^(1/S))` (`-expm1(log δ / S)·S`). -/
noncomputable def budget (S : ℕ) (δ : ℝ) : ℝ := S * (1 - δ ^ ((S : ℝ)⁻¹))

/-- The pooled bound of windows of sizes `U` sampling `d`: the most `Σ_t f_t·U_t` over the shares `f` with
`0 ≤ f_t ≤ 1 − d_t/U_t` and `Σ_t d_t·f_t ≤ budget (Σ_t d_t) δ`. -/
noncomputable def pooled (n : ℕ) (U d : Fin n → ℕ) (δ : ℝ) : ℝ :=
  sSup ((fun f : Fin n → ℝ => ∑ t, f t * U t) ''
    {f | (∀ t, 0 ≤ f t ∧ f t ≤ 1 - (d t : ℝ) / U t) ∧ ∑ t, (d t : ℝ) * f t ≤ budget (∑ t, d t) δ})

namespace Run

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} (D : Run n m s κ X)

/-- Window `t`'s sample count, `min(s_t, U_t)`. -/
def samples (t : Fin n) (ω : SampleCoins n κ X) : ℕ := min (s t) (D.size t ω)

/-- The receipts window `t` opens at coins `ω`: its sample on the first blocks `prf t` gives its coin, with the
fallbacks `rest ω t` where the sampler rejects every word of a first block. -/
def opened (prf : Fin n → Pous.Bits κ → Blocks m) (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n)
    (ω : SampleCoins n κ X) : Finset ℕ :=
  openings (D.size t ω) (D.samples t ω) (prf t (ω.2 t)) (rest ω t)

/-- Window `t`'s correct receipts among its `U_t`. -/
def good (t : Fin n) (ω : SampleCoins n κ X) : ℕ :=
  ((range (D.size t ω)).filter (· ∈ D.correct t ω)).card

/-- Window `t`'s bound on uniform first blocks: `(g_t/U_t)^(s'_t)`, `0` when `g_t < s'_t`. -/
noncomputable def bound (t : Fin n) (ω : SampleCoins n κ X) : ℝ :=
  if D.samples t ω ≤ D.good t ω then ((D.good t ω : ℝ) / (D.size t ω : ℕ)) ^ D.samples t ω else 0

/-- Window `t`'s test at outcome `ω` on first blocks `b`: its sample of `s'_t` of its `U_t` lands in its correct
receipts. -/
def lands (t : Fin n) (ω : SampleCoins n κ X) (b : Blocks m) : Prop :=
  test (D.size t ω) (D.samples t ω) (D.correct t ω) b

/-- **Window `t`'s distinguisher** on coins `y.1 = (ω, u)` and first blocks `y.2`: windows `j < t` land on `prf j` at
their coins in `ω`, window `t` lands on `y.2`, and the threshold coin `u` falls below `∏_{j ≤ t} bound j`. -/
def distinguisher (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ) (t : Fin n)
    (y : (SampleCoins n κ X × Pous.Bits 512) × Blocks m) : Prop :=
  (∀ j, j < t → D.lands j y.1.1 (prf j (y.1.1.2 j))) ∧ D.lands t y.1.1 y.2 ∧
    below δ (∏ j ∈ univ.filter (· ≤ t), D.bound j y.1.1) y.1.2

/-- **The hybrid's potential after `t` windows** at coins `z = (ω, u)`: windows `j < t` land on `prf j` at their coins,
and the threshold coin `u` falls below the product of their bounds. -/
def potential (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ) (t : ℕ) (z : SampleCoins n κ X × Pous.Bits 512) : Prop :=
  (∀ j : Fin n, (j : ℕ) < t → D.lands j z.1 (prf j (z.1.2 j))) ∧
    below δ (∏ j ∈ univ.filter (fun j : Fin n => (j : ℕ) < t), D.bound j z.1) z.2

/-- The session's wrong receipts, `Σ_t (U_t − good_t)`. -/
def incorrect (ω : SampleCoins n κ X) : ℕ := ∑ t, ((D.size t ω : ℕ) - D.good t ω)

/-- The session's pooled bound at `δ`, at the sizes and sample counts of outcome `ω`. -/
noncomputable def pooled (δ : ℝ) (ω : SampleCoins n κ X) : ℝ :=
  Sample.pooled n (fun t => D.size t ω) (fun t => D.samples t ω) δ

end Run

/-- **A session that reads the coin commitment**: `view c` is the session the prover plays holding commitment `c`. -/
structure Committed (n m : ℕ) (s : Fin n → ℕ) (κ : ℕ) (X Cm : Type) where
  view : Cm → Run n m s κ X

namespace Committed

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X Cm : Type} (F : Committed n m s κ X Cm)

/-- The session the prover plays holding commitment `com ω` at coins `ω`. -/
def play (com : SampleCoins n κ X → Cm) : Run n m s κ X where
  size t ω := (F.view (com ω)).size t ω
  correct t ω := (F.view (com ω)).correct t ω

/-- The commitment with openings coin `t`'s leaf committing zeros: `com` at the coins with window `t`'s openings coin
replaced by zeros, the other coins and every salt as they are. -/
def junk (com : SampleCoins n κ X → Cm) (t : Fin n) (ω : SampleCoins n κ X) : Cm :=
  com (ω.1, Function.update ω.2 t fun _ => false)

/-- **Window `t`'s hiding term**: how much likelier the potential after `t + 1` windows is at the deployed commitment
than with openings coin `t`'s leaf committing zeros, plus how much likelier the potential after `t` windows is with
that leaf committing zeros than at the deployed commitment (each truncated at 0). -/
noncomputable def hide [Fintype X] (com : SampleCoins n κ X → Cm) (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ)
    (t : Fin n) : ℝ≥0∞ :=
  (Pous.pr ((F.play com).potential prf δ ((t : ℕ) + 1)) -
      Pous.pr ((F.play (junk com t)).potential prf δ ((t : ℕ) + 1))) +
    (Pous.pr ((F.play (junk com t)).potential prf δ t) - Pous.pr ((F.play com).potential prf δ t))

end Committed

/-- **Window `t`'s continuation** of coins `ω` on `ζ`: `ω`'s other coins and its openings coins before window `t`'s,
then `ζ`'s openings coins from window `t`'s on. -/
def cont {n κ : ℕ} {X : Type} (t : Fin n) (ω ζ : SampleCoins n κ X) : SampleCoins n κ X :=
  (ω.1, fun j => if j < t then ω.2 j else ζ.2 j)

/-- **A session's receipts as the windows' roots hold them**: receipt `p` of window `t` at coins `ω` is a row of an
hm96 bit-row leaf at rank `rank t ω p` of the tree whose root `root t ω p` the window registered, checked by
`verifier t ω p` under `hash` (SHA-512). `opened t ω p` is the opening the prover presents when the window opens it;
the row the root holds for it is `Receipts.held`, read off the prover's runs. -/
structure Receipts (n κ : ℕ) (X : Type) where
  /-- The hash, SHA-512. -/
  hash : Hash.Msg → Hash.Digest
  /-- What the verifier holds for receipt `p`'s leaf: its domain, the hm96 key and the format (`row/v2`). -/
  verifier : Fin n → SampleCoins n κ X → ℕ → Hash.RowLeaf.Verifier
  root : Fin n → SampleCoins n κ X → ℕ → Hash.Digest
  rank : Fin n → SampleCoins n κ X → ℕ → ℕ
  opened : Fin n → SampleCoins n κ X → ℕ → Hash.RowLeaf.Opening

namespace Receipts

variable {n κ : ℕ} {X : Type} (R : Receipts n κ X)

/-- **Receipt `p` of window `t` shows row `v` at coins `ω`**: the window opens it, the verifier accepts the opening the
prover presents at the receipt's root and rank, and its row is `v`. -/
def Shows {m : ℕ} {s : Fin n → ℕ} (D : Run n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (p : ℕ) (ω : SampleCoins n κ X) (v : Hash.Row) : Prop :=
  p ∈ D.opened prf rest t ω ∧
    Hash.RowLeaf.accepts R.hash R.hash (R.verifier t ω p) (R.root t ω p) (R.rank t ω p) (R.opened t ω p) ∧
    (R.opened t ω p).row = v

/-- **The row receipt `p`'s root holds** at window `t` and coins `ω`: the plurality of the rows it shows over window
`t`'s continuations of `ω`, a row that the most of them show (`ResourceAccounting.TS.plural` at `cont t`, ties broken by
`Classical.epsilon`). -/
noncomputable def held [Fintype X] {m : ℕ} {s : Fin n → ℕ} (D : Run n m s κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n)
    (ω : SampleCoins n κ X)
    (p : ℕ) : Hash.Row :=
  ResourceAccounting.TS.plural (cont t) (R.Shows D prf rest t p) ω

/-- **Receipt `p` of window `t` moved**: the verifier accepts the opening presented, at the receipt's root and rank,
with a row other than the one its root holds. -/
def Moved [Fintype X] {m : ℕ} {s : Fin n → ℕ} (D : Run n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (ω : SampleCoins n κ X) (p : ℕ) : Prop :=
  Hash.RowLeaf.accepts R.hash R.hash (R.verifier t ω p) (R.root t ω p) (R.rank t ω p) (R.opened t ω p) ∧
    (R.opened t ω p).row ≠ R.held D prf rest t ω p

/-- **The receipts' binding term**: the probability that some window opens a receipt that moved. -/
noncomputable def binding [Fintype X] {m : ℕ} {s : Fin n → ℕ} (D : Run n m s κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) : ℝ≥0∞ :=
  Pous.pr fun ω => ∃ t, ∃ p ∈ D.opened prf rest t ω, R.Moved D prf rest t ω p

end Receipts

end ResourceAccounting.Remote.Sample
