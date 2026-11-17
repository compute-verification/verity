module

public import Definitions.ResourceAccounting.ReadUnit
public import Proofs.Flock.Soundness.Discharge.Hm.Gadget
public import Properties.Composition
public import Properties.Flock.Circuit.CircuitOf
public import Properties.Verity.Sha512
public import Definitions.Compose.JointLive
public import Definitions.Core.Blocks
public import Definitions.Core.Game
public import Definitions.Core.Site
public import Definitions.Hash.Assumptions
public import Definitions.Hash.RowLeaf
public import Definitions.InferenceOnly
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

/-! The table read's net, part by part: what each part's net computes whenever its rows hold. -/

@[expose] public section

namespace ResourceAccounting.ReadUnit.Gates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

def Shaped (k : ℕ) (ws : List Word) : Prop := ws.length = k ∧ ∀ w ∈ ws, w.length = 64

def evs (z : Var → Bool) (fs : List Lin) : List Bool := fs.map (ev z)

def wordsBytes (z : Var → Bool) (ws : List Word) : Hash.Msg := ReadUnit.bytesOf (evs z ws.flatten)

def wordsDigest (z : Var → Bool) (ws : List Word) : Verity.Digest := ReadUnit.digestOf (evs z ws.flatten)

def RowSegSpec (n : ℕ) (gen : List Lin → M (List Word)) : Prop :=
  ∀ row : List Lin, Sound (gen row) fun z x => row.length = n →
    Shaped 8 x ∧ wordsDigest z x = Hash.RowSeg.digest Verity.sha512 (evs z row)

def Hm96LeafSpec (key : Hash.Msg) (gen : List Word → List Word → M (List Word)) : Prop :=
  ∀ x y : List Word, Sound (gen x y) fun z out => Shaped 8 x → Shaped 24 y →
    Shaped 24 out ∧ wordsDigest z (out.drop 16) = Hash.Hm96.leaf Verity.sha512 key (wordsDigest z x) (wordsBytes z y)

def SelectSpec (V : ℕ) (gen : List Lin → Lin → List Lin → M (List Lin × Lin)) : Prop :=
  ∀ (tok : List Lin) (opened : Lin) (row : List Lin), Sound (gen tok opened row) fun z out =>
    tok.length = 32 → row.length % 16 = 0 →
      evs z out.1 = (if num (evs z tok) < V then evs z row else Hash.EmbeddingRead.fill (row.length / 16)) ∧
        ev z out.2 = (decide (V ≤ num (evs z tok)) || ev z opened)

/-- The dense frame-v3 domain of `2 ^ D` ranks, rank `r` at position `r`; the climb reads only its id and positions. -/
def denseDomain (D : ℕ) (id : Verity.Digest) : Hash.Tree.Domain := ⟨id, 2 ^ D, fun r => r⟩

def FrameReadSpec (D : ℕ) (schema : String)
    (gen : List Word → List Word → List Word → List Lin → List (List Word) → M Lin) : Prop :=
  ∀ (root domain leaf : List Word) (pos : List Lin) (path : List (List Word)),
    Sound (gen root domain leaf pos path) fun z opened => Shaped 8 root → Shaped 8 domain → Shaped 8 leaf →
      pos.length = D → path.length = D → (∀ s ∈ path, Shaped 8 s) → ev z opened = true →
      (Hash.Tree.climb Verity.sha512 (denseDomain D (wordsDigest z domain)) 0 (num (evs z pos))
        (Verity.sha512 (Hash.Tree.leafMsg (denseDomain D (wordsDigest z domain)) (num (evs z pos)) schema
          (wordsDigest z leaf).bytes))
        (path.map (wordsDigest z))).2 = wordsDigest z root

end ResourceAccounting.ReadUnit.Gates
