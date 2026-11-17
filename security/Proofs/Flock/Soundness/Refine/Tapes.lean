module

public import Flock.Opening
public import Flock.Verify
public import Proofs.Flock.Soundness.Instance
public import Proofs.Flock.Soundness.Refine.Run
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.CategoryTheory.Category.Init
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import VCVio.Prelude.Core

/-! The model's messages read off a decoded proof, and the executable's phase outputs as the model's. -/

@[expose] public section

namespace FlockSoundness.Refine

open Flock Game

local notation "𝔽" => FlockLevel3.GF128

/-- A word of the proof as a model field element. -/
abbrev fOf (a : F128) : 𝔽 := FlockLevel3.GF128.of a

/-- An array of words as a model vector of length `n`. -/
def vecOf (n : ℕ) (a : Array F128) : Fin n → 𝔽 := fun i => fOf a[i.val]!

/-- A pair of words as a pair of model field elements. -/
def pairOf (p : F128 × F128) : 𝔽 × 𝔽 := (fOf p.1, fOf p.2)

def zcMsgs (m : ℕ) (p : ZerocheckProof) : List Msg :=
  ⟨(Fin 64 → 𝔽) × (Fin 64 → 𝔽), (vecOf 64 p.round1ab, vecOf 64 p.round1c)⟩ ::
    ((List.range (m - 6)).map fun i => ⟨𝔽 × 𝔽, pairOf p.rounds[i]!⟩) ++
    [⟨𝔽 × 𝔽, (fOf p.finalA, fOf p.finalB)⟩]

def lcMsgs (kLog : ℕ) (p : LincheckProof) : List Msg :=
  ((List.range (kLog - 6)).map fun j => ⟨𝔽 × 𝔽, pairOf p.rounds[j]!⟩) ++ [⟨Fin 64 → 𝔽, vecOf 64 p.zPartial⟩]

structure ClaimRel (m : ℕ) (c : Flock.Claim) (c' : Model.Claim 𝔽 m) : Prop where
  value : c'.value = fOf c.value
  skip_size : c.skip.size = 64
  skip : ∀ s, c'.skip s = fOf c.skip[s.val]!
  x_size : c.x.size = m - 6
  x : ∀ i, c'.x i = fOf c.x[i.val]!

def opMsgs (n : ℕ) (rs : Array (Array F128)) : List Msg :=
  (List.range n).map fun k => ⟨Fin 128 → 𝔽, vecOf 128 rs[k]!⟩

structure OpenRel (m n : ℕ) (T : F128) (sws : Array Switched) (op : Model.OpenOut 𝔽) : Prop where
  T : op.T = fOf T
  size : sws.size = n
  rpp_size : ∀ k < n, (sws[k]!).rpp.size = 7
  z_size : ∀ k < n, (sws[k]!).z.size = m - 7
  b : ∀ y, op.b y = ∑ k : Fin n, fOf (sws[k.val]!).gamma *
    Model.phi execArith (vecOf 7 (sws[k.val]!).rpp) (Model.eqAt (vecOf (m - 7) (sws[k.val]!).z) y)

structure ZcRel (m : ℕ) (zc : Flock.ZcOut) (zc' : Model.ZcOut 𝔽 m) : Prop where
  z : zc'.z = fOf zc.z
  rho_size : zc.rho.size = m - 6
  rho : ∀ i, zc'.ρ i = fOf zc.rho[i.val]!
  r_size : zc.rRest.size = m - 6
  r : ∀ i, zc'.r i = fOf zc.rRest[i.val]!
  va : zc'.va = fOf zc.vA
  vb : zc'.vb = fOf zc.vB
  vc : zc'.vc = fOf zc.vC

end FlockSoundness.Refine
