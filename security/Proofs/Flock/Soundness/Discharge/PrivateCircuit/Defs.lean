module

public import Proofs.Flock.Soundness.Binding.Leaf
public import Definitions.Flock.ZK.SHVZK
public import Proofs.Flock.Soundness.Discharge.ZkView.Defs
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

/-! The objects of the private circuit, whose program the public universal unit `U cls` interprets; that `U` reads
only the class, the owed outputs and the registration's view is what "the universal unit is public" means. -/

@[expose] public section

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock ZK FlockSoundness.Discharge.ZkView
open FlockSoundness.Game (prCoin avg)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

structure Scheme (Cls Circ Out : Type) (Prog Rr Rv : Cls → Type) (K W Hi VL : Type) [Field K] where
  fits : Circ → Cls → Prop
  pad : Circ → (c : Cls) → Prog c
  reg : (c : Cls) → Prog c → Rr c → Rv c
  J : Cls → ℕ
  zs : (c : Cls) → Fin (J c) → Sizes
  U : (c : Cls) → Out → Rv c → ∀ j, Coined K W Hi VL (zs c j)

def PadCorrect {Cls Circ In Out : Type} {Prog : Cls → Type} (fits : Circ → Cls → Prop) (pad : Circ → (c : Cls) → Prog c)
    (run : (c : Cls) → Prog c → In → Out) (eval : Circ → In → Out) : Prop :=
  ∀ C c, fits C c → ∀ x, run c (pad C c) x = eval C x

section Tables

variable {K W Hi VL : Type} [Field K] {J : ℕ} {zs : Fin J → Sizes}

def Sat (T : ∀ j, Coined K W Hi VL (zs j)) (w : Fin J → W) : Prop :=
  ∀ j, (T j).tab.sReg (w j) = (T j).tab.pubReg ∧ (T j).tab.InnerHolds (w j)

def ZkOk (T : ∀ j, Coined K W Hi VL (zs j)) : Prop :=
  ∀ j, (T j).tab.Rank ∧ (T j).tab.PadOnto ∧ (∀ r, Function.Bijective fun p : 𝔽 × 𝔽 => (T j).tab.Wr r p.1 p.2) ∧
    (T j).tab.PadsOnto ∧ ∀ r, (T j).tab.β r ≠ 0

end Tables

noncomputable def zkGap {J : ℕ} (zs : Fin J → Sizes) : ℝ≥0∞ :=
  ∑ j, 2 * (((zs j).nU + 2 * (zs j).nPU : ℕ) : ℝ≥0∞) * ENNReal.ofReal (CROnly.hidingGap My saltHash)

/-- **The registration hides the program** up to `ε`: for any two programs `p`, `p'` and any event `E` of the
registration's view, over the registrant's uniform salts, `E` at `p` is at most `E` at `p'` plus `ε`. -/
def Hides {P R V : Type} [Fintype R] (reg : P → R → V) (ε : ℝ≥0∞) : Prop :=
  ∀ (p p' : P) (E : V → Prop), prCoin (fun r => E (reg p r)) ≤ prCoin (fun r => E (reg p' r)) + ε

section Views

variable {Cls Circ Out : Type} {Prog Rr Rv : Cls → Type} [∀ c, Fintype (Rr c)] {K W Hi VL : Type} [Field K]
  [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] {Obs : Type}

/-- Over the registrant's uniform salts `r`, then the session's randomness and hidden leaves' salts, with the honest
witness `w r` (every hidden leaf the executable's, `Coined.viewExe`). -/
noncomputable def realPr (S : Scheme Cls Circ Out Prog Rr Rv K W Hi VL) (c : Cls) (o : Out) (p : Prog c)
    (w : Rr c → Fin (S.J c) → W) (enc : ∀ v, (∀ j, (S.U c o v j).tab.View) → Obs) (E : Rv c → Obs → Prop) : ℝ≥0∞ :=
  avg fun r : Rr c =>
    prCoin fun q : ∀ j, (S.U c o (S.reg c p r) j).tab.Rand × ((S.U c o (S.reg c p r) j).tab.Hid → Salt) =>
      E (S.reg c p r) (enc (S.reg c p r) fun j => (S.U c o (S.reg c p r) j).viewExe (w r j) (q j).1 (q j).2)

/-- It reads the class `c` and the owed outputs `o`, never the circuit. -/
noncomputable def simPr (S : Scheme Cls Circ Out Prog Rr Rv K W Hi VL) (c : Cls) (o : Out) (p₀ : Prog c)
    (w₀ : Fin (S.J c) → W) (enc : ∀ v, (∀ j, (S.U c o v j).tab.View) → Obs) (E : Rv c → Obs → Prop) : ℝ≥0∞ :=
  avg fun r : Rr c =>
    prCoin fun q : ∀ j, (S.U c o (S.reg c p₀ r) j).tab.SimRand × ((S.U c o (S.reg c p₀ r) j).tab.Hid → Salt) =>
      E (S.reg c p₀ r) (enc (S.reg c p₀ r) fun j => (S.U c o (S.reg c p₀ r) j).simExe (w₀ j) (q j).1 (q j).2)

end Views

/-- **A program commitment**: `Opens rt p op`, the opening `op` shows that the root `rt` commits to the program `p`; and
`ext op op'`, the extractor's two SHA-512 inputs from two openings. -/
structure Commit (Root P Op : Type) where
  Opens : Root → P → Op → Prop
  ext : Op → Op → List UInt8 × List UInt8

/-- **The commitment binds under SHA-512**: two openings of one root to two different programs give an explicit
collision of SHA-512 (`Binding.Collides`), the extractor's pair. -/
def Commit.Binding {Root P Op : Type} (com : Commit Root P Op) : Prop :=
  ∀ rt p p' op op', com.Opens rt p op → com.Opens rt p' op' → p ≠ p' → Binding.Collides (com.ext op op')

end FlockSoundness.Discharge.PrivateCircuit
