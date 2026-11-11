module

public import Definitions.Accounting.ReadUnit
public import Proofs.Flock.Soundness.Discharge.Hm.Gadget
public import Properties.Composition
public import Properties.Flock.Circuit.CircuitOf
public import Properties.Verity.Sha512
public import Definitions.Accounting.JointLive
public import Definitions.Core.Blocks
public import Definitions.Core.Game
public import Definitions.Core.Site
public import Definitions.Hash.Assumptions
public import Definitions.Hash.RowLeaf
public import Definitions.Nci
public import Flock.CoinTree
public import Flock.Hash
public import Flock.Hm96
public import Lms.Verify
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
# The table read's net, part by part: what each part's net must compute

A read unit's gates (`Accounting.ReadUnit.GatesRefine`) are `EmbeddingRead_v1{V, H}`'s rows, which
`verity/ml/commitments/gates/embedding.py` composes from four parts' nets. Each part is built here over the verifier's
builder (`Flock.HmNets`: forms over committed bits, a row a committed product), and each part's statement says what its
net computes whenever its rows hold (`FlockSoundness.Discharge.Hm.Sound`), against `Hash`'s definitions at SHA-512
(`Verity.sha512`):

* `RowSegSpec`: `Sha512RowSeg_v1{n}`, the row's `sha512/row-seg/v1` digest (`Hash.RowSeg.digest`).
* `Hm96LeafSpec`: `Hm96Sha512Leaf_v1`, the hm96 leaf of that digest under the salt (`Hash.Hm96.leaf`).
* `FrameReadSpec`: `FrameV3Read_v1{D, schema}`, the frame-v3 leaf at the rank climbed to the root (`Hash.Tree.climb`).
* `SelectSpec`: `EmbeddingReadSelect_v1{V, H}`, the row or the fill row by the token, and the verdict.

Each statement takes its part's generator as a parameter, so a generator that mirrors the part's Python lowering is
proved against it; the read's template composes the four as `EmbeddingRead_v1`'s body does. Each holds of every
input, with the inputs' shapes premises of what it concludes, as `FlockSoundness.Discharge.Hm`'s gadgets' statements
do, so that `Sound.bind'` composes them. A digest or a byte string is held in 64-bit words of forms in leaf order, each word's bits
least significant first and its bytes big-endian (`Accounting.ReadUnit.bytesOf`), as a read unit's ports hold them.
-/

@[expose] public section

namespace Accounting.ReadUnit.Gates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

/-- `k` words of 64 forms each. -/
def Shaped (k : ℕ) (ws : List Word) : Prop := ws.length = k ∧ ∀ w ∈ ws, w.length = 64

/-- The bits the forms `fs` hold under the committed bits `z`. -/
def evs (z : Var → Bool) (fs : List Lin) : List Bool := fs.map (ev z)

/-- The byte string the words `ws` hold under `z`, in leaf order. -/
def wordsBytes (z : Var → Bool) (ws : List Word) : Hash.Msg := ReadUnit.bytesOf (evs z ws.flatten)

/-- The digest the 8 words `ws` hold under `z`, in leaf order. -/
def wordsDigest (z : Var → Bool) (ws : List Word) : Verity.Digest := ReadUnit.digestOf (evs z ws.flatten)

/-- **Part A: the row-seg digest.** On a row of `n` forms, the net of `gen` returns 8 words that hold the row's
`sha512/row-seg/v1` digest whenever its rows hold. -/
def RowSegSpec (n : ℕ) (gen : List Lin → M (List Word)) : Prop :=
  ∀ row : List Lin, Sound (gen row) fun z x => row.length = n →
    Shaped 8 x ∧ wordsDigest z x = Hash.RowSeg.digest Verity.sha512 (evs z row)

/-- **Part B: the hm96 leaf**, under the hm96 key `key`. On a digest's 8 words `x` and a salt's 24 words `y`, the net of
`gen` returns 24 words `b ‖ c ‖ leaf` whose last 8 hold the hm96 leaf of the digest and the salt they hold, whenever its
rows hold. -/
def Hm96LeafSpec (key : Hash.Msg) (gen : List Word → List Word → M (List Word)) : Prop :=
  ∀ x y : List Word, Sound (gen x y) fun z out => Shaped 8 x → Shaped 24 y →
    Shaped 24 out ∧ wordsDigest z (out.drop 16) = Hash.Hm96.leaf Verity.sha512 key (wordsDigest z x) (wordsBytes z y)

/-- **Part B: the select**, for a table of `V` rows. On the token's 32 forms (least significant first), the frame read's
verdict `opened` and the row's forms, the net of `gen` returns the row where the token is below `V` and the fill row
elsewhere, and a verdict set exactly where the token is at least `V` or `opened` is, whenever its rows hold. -/
def SelectSpec (V : ℕ) (gen : List Lin → Lin → List Lin → M (List Lin × Lin)) : Prop :=
  ∀ (tok : List Lin) (opened : Lin) (row : List Lin), Sound (gen tok opened row) fun z out =>
    tok.length = 32 → row.length % 16 = 0 →
      evs z out.1 = (if num (evs z tok) < V then evs z row else Hash.EmbeddingRead.fill (row.length / 16)) ∧
        ev z out.2 = (decide (V ≤ num (evs z tok)) || ev z opened)

/-- The dense frame-v3 domain of `2 ^ D` ranks under the domain id `id`: rank `r` at position `r`. The climb reads only
its id and positions. -/
def denseDomain (D : ℕ) (id : Verity.Digest) : Hash.Tree.Domain := ⟨id, 2 ^ D, fun r => r⟩

/-- **Part C: the frame-v3 read** of depth `D` under the leaf schema `schema`. On the root's, the domain id's and the
leaf's 8 words, the rank's `D` forms (least significant first) and `D` sibling digests' words, bottom first, the net of
`gen` returns a verdict that is set only where the frame leaf over the leaf digest at that rank, climbed through the
siblings in the dense domain under that domain id, reaches the root, whenever its rows hold. -/
def FrameReadSpec (D : ℕ) (schema : String)
    (gen : List Word → List Word → List Word → List Lin → List (List Word) → M Lin) : Prop :=
  ∀ (root domain leaf : List Word) (pos : List Lin) (path : List (List Word)),
    Sound (gen root domain leaf pos path) fun z opened => Shaped 8 root → Shaped 8 domain → Shaped 8 leaf →
      pos.length = D → path.length = D → (∀ s ∈ path, Shaped 8 s) → ev z opened = true →
      (Hash.Tree.climb Verity.sha512 (denseDomain D (wordsDigest z domain)) 0 (num (evs z pos))
        (Verity.sha512 (Hash.Tree.leafMsg (denseDomain D (wordsDigest z domain)) (num (evs z pos)) schema
          (wordsDigest z leaf).bytes))
        (path.map (wordsDigest z))).2 = wordsDigest z root

end Accounting.ReadUnit.Gates
