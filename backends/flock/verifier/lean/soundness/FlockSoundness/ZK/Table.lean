import FlockSoundness.ZK.Blocks
import FlockSoundness.ZK.Dist
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.CharP.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Logic.Function.Basic
import FlockSoundness.Protocol.ZK.Table

/-!
# Zero knowledge: one masked table, its reference prover and `S_shvzk`, at fixed verifier coins

The public-circuit proof (`docs/zk-proof-public.md` in the Project store) proves special honest-verifier zero knowledge
of one masked table (Lemma B, §4.3) and witness indistinguishability of its pre-final messages (Lemma A, §4.2), each at a
fixed coin vector `e`. This file is the spec-level model those lemmas are about, for one table: the paper's lemmas are per
session of `J` tables, and the product over tables is `Session`'s.

**The reference prover** is `Table.view`: a deterministic function of the statement and coins (`Table`), the witness `w`
and the prover's randomness (`Table.Rand`), which returns every message the verifier sees, the final message included
(§2.3–§2.5 for one table, `J = 1`). What masking adds is written out:

* **the masked ring switch** (line 13): `ŝ_k = ν_k(w) + Φ_k(u)`, `Φ_k(u)[v] = Σ_x eq(p_k, x)·u[x][v]` over the mask slot
  `u`, for the `ab` and `c` claims of both reps; region claims send their public value (H_reg, §2.1);
* **level 0** (lines 5–7, 15): every lane's codeword entry at an opened position is `Enc(P_l)(q) + Pad(μ_l)(q)`, the
  message part plus its padding; the four extra lanes `R^r_i`, padded the same way; the restarted table
  `y₁'^r = y₁^r + W^r(R^r)` and `ȳ^r = Σ_l eq(r̄^r, l)·μ_l + W^r(μ_{R^r})`, with `W^r(a, b) = ρ^r₀·a + ρ^r₁·b`;
* **the pads** (line 0 and every "`v + h`"): each exposed value is the clear value plus its own pad; the triple
  `h_ab := h_fa·h_fb`, `h_tw := h_tx·h_ty`, and `ρ_in := ζ·h_fa + h_tx`, `σ_in := h_fb + h_ty` (line 16);
* **the inner proof** (line 16): `τ := ⟨λ, μ⟩`, `Y₁ := h + β·μ`, `Y₂ := pad_h + β·pad_μ`, and the pads code's columns
  `C_h = Code(h ‖ pad_h)`, `C_μ = Code(μ ‖ pad_μ)` at the opened positions;
* **hm96 leaves as ideal hiding commitments** (world `W₁` of Lemma B): a leaf the final message opens is `leaf(content;
  salt)`, with its salt revealed; every other level-0 and pads-tree leaf is ideal, `ideal(seed)` of a fresh seed,
  independent of what it would commit. `τ`'s commitment is opened (`τ` and `y_τ` are in the final message).

**What stays abstract** is what masking doesn't touch, so every theorem holds for any of it: the clear prover's values
under the pads (`vals`: the zerocheck and lincheck rounds, level 0's out-of-domain value, lane-round messages and `e^r`,
the final evaluations), the packing of the witness into lanes (`lanes`), the non-mask part of each masked claim (`nu`),
and levels ≥ 1 from `T'` on (`lig`), a function of `y₁'^r` and fresh salts (§4.3 row (v)). The verifier's replay gives
the inner proof's batched constraint `(λ, θ)` from the messages before it (`lam`, `theta`). An honest prover's pads
satisfy `⟨λ, h⟩ = θ` (§2.4, "An honest prover's `h` satisfies all of them"): that is `InnerHolds`, the one place the
witness's validity enters.

**The view** (`Table.View`) holds every message, with each Merkle tree given by its leaves (the caps and paths are a
public function of them), and the restarted tables `y₁'^r`. The real verifier sees less than `y₁'^r` (`T'` and levels ≥ 1
are functions of it), so a statement about this view implies the one about the real view by post-processing.

**The simulator** `Table.sim` is §3.2's `S_shvzk(x, e)`: run the prover on the dummy witness `w₀` with fresh randomness
and keep its level-0 openings and its Ligerito part from `T'` on (step 1); fresh uniform masked values, masked `ŝ`, `ρ_in`,
`σ_in`, `Y₁`, `Y₂` and `C_h` (steps 2, 3, 4, 6); the public region values (step 3); `τ := (⟨λ, Y₁⟩ − θ)/β` and
`C_μ := (Code(Y) − C_h)/β` from the replayed constraint (steps 5, 6); a pads tree with the simulated columns at the opened
positions and ideal leaves elsewhere, and `τ`'s commitment under a fresh salt.
-/

namespace FlockSoundness.ZK

open Finset

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)

/-! ## The prover's randomness -/

/-! ## The prover's computations -/

variable [CharP F 2]

/-- **The prover's rank check** (§2.4 line 11), passing at `e`: the claims' weights on the mask slot span, so a GF(2)
vector over the mask-slot words reaches every vector of claim values. -/
def Rank : Prop :=
  Function.Surjective fun c : Fin T.nMask → ZMod 2 => fun k => ∑ x, T.eqw k x * castF (c x)

/-! ## The view -/

/-- The messages the verifier's replay reads before the inner proof's constraint (`lam`, `theta`). -/
def View.pre (v : T.View) :
    (Fin T.nC → Fin 128 → F) × (Fin 2 → Fin T.nE → F) × (Fin 2 → Fin T.L → K) × (Fin 2 → Fin T.tpad → K)
      × (Fin 2 → F × F) × (Fin 2 → VL) :=
  (v.sHat, v.masked, v.y1, v.ybar, v.inner, v.lig)

/-! ## The simulator -/

/-! ## What the theorems assume of the coins, the codes and the witness -/

/-- The inverse of `W^r`, where it is invertible. -/
noncomputable def Winv (r : Fin 2) : K → F × F := Function.invFun fun p : F × F => T.Wr r p.1 p.2

/-- (★) solved for rep `r`'s extra lanes at opened position `q`: `Enc(y₁'^r)(q) + Pad(ȳ^r)(q) − Σ_l eq(r̄^r, l)·row_q[l]`. -/
def xTarget (rows : Fin T.nS → Fin T.nL → F) (y1 : Fin 2 → Fin T.L → K) (yb : Fin 2 → Fin T.tpad → K) (q : Fin T.nS)
    (r : Fin 2) : K :=
  ∑ c, y1 r c * T.embed (T.E q c) + ∑ j, yb r j * T.embed (T.Gp q j) - ∑ l, T.rbar r l * T.embed (rows q l)

/-- The level-0 padding reaches every vector of values at the opened positions (T4: at most `t_pad` distinct positions,
where `X_L + κ` vanishes at none; `pad_onto_M1` for M1's code). -/
def PadOnto : Prop := Function.Surjective fun μ : Fin T.tpad → F => fun q => ∑ j, μ j * T.Gp q j

/-- Rep `r`'s pads-code padding reaches every vector of values at its opened positions (T4 with the monomials: at most
192 distinct nonzero points). -/
def PadsOnto : Prop := ∀ r, Function.Surjective fun p : Fin T.nPad → F => fun k => ∑ j, T.Cp r k j * p j

/-- **The honest prover's pads satisfy the inner proof's constraint** (§2.4 line 16): `⟨λ, h⟩ = θ` in every run, `λ` and
`θ` the verifier's replay on the prover's messages. Each of the ten constraints is affine in `h` and holds for an honest
prover's `h` because its clear values pass the clear checks. This is where a valid witness is needed; Lemma A needs
none. -/
def InnerHolds (w : W) : Prop :=
  ∀ (ω : T.Rand) (r : Fin 2), ∑ i, T.lam r (T.view w ω).pre i * T.hFull (ω.2.2.1 r) i = T.theta r (T.view w ω).pre

/-! ## The pre-final view -/

/-- **The pre-final view in world `W₀`** (§4.2): every message up to the last coin opening, with every level-0 leaf,
every pads-tree leaf and every `τ`-commitment ideal. It holds `final_c + h_fc` too, which Lemma A's translation matches
as well. -/
@[ext] structure PreView where
  sHat : Fin T.nC → Fin 128 → F
  sReg : Fin T.nR → Fin 128 → F
  masked : Fin 2 → Fin T.nE → F
  y1 : Fin 2 → Fin T.L → K
  ybar : Fin 2 → Fin T.tpad → K
  lig : Fin 2 → VL
  inner : Fin 2 → F × F
  Y1 : Fin 2 → PadIdx T.nE → F
  Y2 : Fin 2 → Fin T.nPad → F
  leaves0 : (Fin T.nS → D) × (Fin T.nU → D)
  tauCom : Fin 2 → D
  leavesP : Fin 2 → (Fin T.nQ → D) × (Fin T.nPU → D)

/-- The prover's pre-final messages in world `W₀`, on witness `w` and randomness `ω`. -/
noncomputable def preView (w : W) (ω : T.Rand) : T.PreView :=
  let v := T.view w ω
  let s := ω.2.2.2.2.2.2
  { sHat := v.sHat, sReg := v.sReg, masked := v.masked, y1 := v.y1, ybar := v.ybar, lig := v.lig, inner := v.inner
    Y1 := v.Y1, Y2 := v.Y2
    leaves0 := (fun q => T.ideal (s.2.1 q), fun q => T.ideal (s.2.2.1 q))
    tauCom := fun r => T.ideal (s.2.2.2.2.2.2.2.1 r)
    leavesP := fun r => (fun k => T.ideal (s.2.2.2.2.1 r k), fun k => T.ideal (s.2.2.2.2.2.1 r k)) }

end Table

end FlockSoundness.ZK
