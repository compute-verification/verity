import Proofs.Flock.Soundness.Refine.Run
import Proofs.Flock.Soundness.Instance
import Flock.Verify

/-!
# The model's messages, read off a decoded proof

A run of the model (`Game.Plays`) takes the prover's messages as a typed tape. For the executable's session these are
the decoded proof's fields, each read as the model's type through the executable's field embedding (`GF128.of`), in the
order the model receives them. The coins need no such function: they are the record's rounds' coins, in order.

This file holds those tapes and the correspondence between the executable's and the model's phase outputs.
-/

namespace FlockSoundness.Refine

open Flock Game

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

/-- A word of the proof as a model field element. -/
abbrev fOf (a : F128) : 𝔽 := FlockLevel3.GF128.of a

/-- An array of words as a model vector of length `n`. -/
def vecOf (n : ℕ) (a : Array F128) : Fin n → 𝔽 := fun i => fOf a[i.val]!

/-- A pair of words as a pair of model field elements. -/
def pairOf (p : F128 × F128) : 𝔽 × 𝔽 := (fOf p.1, fOf p.2)

/-- **The zerocheck's messages** (PROTOCOL.md §9.2), in the model's order: `(round1_ab, round1_c)` (Z4), the `m − 6`
residual rounds' `(G(1), G(∞))` (Z6), then `(final_a_eval, final_b_eval)` (Z7). -/
def zcMsgs (m : ℕ) (p : ZerocheckProof) : List Msg :=
  ⟨(Fin 64 → 𝔽) × (Fin 64 → 𝔽), (vecOf 64 p.round1ab, vecOf 64 p.round1c)⟩ ::
    ((List.range (m - 6)).map fun i => ⟨𝔽 × 𝔽, pairOf p.rounds[i]!⟩) ++
    [⟨𝔽 × 𝔽, (fOf p.finalA, fOf p.finalB)⟩]

/-- **The lincheck's messages** (PROTOCOL.md §10), in the model's order: the `k_log − 6` rounds' `(q(1), q(∞))` (L3),
then `z_partial` (L4). -/
def lcMsgs (kLog : ℕ) (p : LincheckProof) : List Msg :=
  ((List.range (kLog - 6)).map fun j => ⟨𝔽 × 𝔽, pairOf p.rounds[j]!⟩) ++ [⟨Fin 64 → 𝔽, vecOf 64 p.zPartial⟩]

/-- An executable claim is the model's (§11): the value, the 64 skip weights and the `m − 6` coordinates. -/
structure ClaimRel (m : ℕ) (c : Flock.Claim) (c' : Model.Claim 𝔽 m) : Prop where
  value : c'.value = fOf c.value
  skip_size : c.skip.size = 64
  skip : ∀ s, c'.skip s = fOf c.skip[s.val]!
  x_size : c.x.size = m - 6
  x : ∀ i, c'.x i = fOf c.x[i.val]!

/-- **The ring switches** (PROTOCOL.md §12.2): one `ŝ` of 128 words per claim, in claim order (O3). -/
def opMsgs (n : ℕ) (rs : Array (Array F128)) : List Msg :=
  (List.range n).map fun k => ⟨Fin 128 → 𝔽, vecOf 128 rs[k]!⟩

/-- The executable's opening output is the model's (§12.5): the batched value `T`, and the batched basis `b`, which
the executable keeps as each switched claim's batching coin, ring-switch coins `r''` and suffix point `x[1 …]`. -/
structure OpenRel (m n : ℕ) (T : F128) (sws : Array Switched) (op : Model.OpenOut 𝔽) : Prop where
  T : op.T = fOf T
  size : sws.size = n
  rpp_size : ∀ k < n, (sws[k]!).rpp.size = 7
  z_size : ∀ k < n, (sws[k]!).z.size = m - 7
  b : ∀ y, op.b y = ∑ k : Fin n, fOf (sws[k.val]!).gamma *
    Model.phi execArith (vecOf 7 (sws[k.val]!).rpp) (Model.eqAt (vecOf (m - 7) (sws[k.val]!).z) y)

/-- The executable's zerocheck output is the model's (§9.2): the skip point, the residual challenges, the zerocheck
vector and the three claimed values. -/
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
