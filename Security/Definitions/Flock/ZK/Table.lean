import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.CharP.Defs
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.Finite.Prod
import Mathlib.Data.Finset.Card
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.ZMod.Basic
import Mathlib.Logic.Function.Basic
import Mathlib.Logic.Function.Iterate
import Mathlib.Tactic.Abel
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring

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

/-- Rep `r`'s pads (§2.2): one per exposed value (`inl`), then `h_tx`, `h_ty`, `h_ab`, `h_tw` (`inr 0 … 3`). -/
abbrev PadIdx (nE : ℕ) := Fin nE ⊕ Fin 4

/-- **One masked table at a fixed coin vector** (`docs/zk-proof-public.md` §2.3–§2.5, `J = 1`): sizes; what the coins
fix (claim points, `r̄`, `ρ`, `ζ`, `β`, the opened positions and their code rows); the clear prover's values, which the
masked prover wraps; the verifier's replay of the inner proof's constraint; and the leaf functions. `W` is the witness
type; `Salt`, `Seed`, `Hi` are an hm96 salt, an ideal leaf's seed and the salts of levels ≥ 1; `D` a digest; `VL` the
messages of levels ≥ 1. -/
structure Table (F K W Salt Seed D Hi VL : Type) [Field F] [Field K] where
  /-- Mask-slot words, masked claims (`ab`, `c` of both reps), region claims. -/
  nMask : ℕ
  nC : ℕ
  nR : ℕ
  /-- Witness lanes (`2^k₀`), message words per lane, level-0 padding per lane (`t_pad`). -/
  nL : ℕ
  L : ℕ
  tpad : ℕ
  /-- The distinct level-0 positions the two reps open (`S = Q⁰ ∪ Q¹`), and the others. -/
  nS : ℕ
  nU : ℕ
  /-- Exposed values per rep: every value sent as `v + h_i` (lines 4–10, level 0's `y`, `msg`, `e`, and the final
  evaluations, `final_c` included). -/
  nE : ℕ
  /-- The distinct pads-code positions a rep opens, the others, and the pads-code padding (192). -/
  nQ : ℕ
  nPU : ℕ
  nPad : ℕ
  /-- The embedding `F ⊂ K`. -/
  embed : F →+* K
  /-- `eq(p_k[7..], x)` for masked claim `k` at mask-slot word `x`. -/
  eqw : Fin nC → Fin nMask → F
  /-- The non-mask words' part of masked claim `k`'s ring-switch message, `ν_k(w)` (§4.2 step 2). -/
  nu : W → Fin nC → Fin 128 → F
  /-- A region claim's ring-switch message, as the prover computes it from the witness, and the public value. -/
  sReg : W → Fin nR → Fin 128 → F
  pubReg : Fin nR → Fin 128 → F
  /-- The witness lanes' messages `P_l(z)`, `z = (w, u)`. -/
  lanes : W → (Fin nMask → Fin 128 → ZMod 2) → Fin nL → Fin L → F
  /-- The level-0 code at the opened positions: `Enc(P)(q) = Σ_c P[c]·E q c`, `Pad(μ)(q) = Σ_j μ[j]·Gp q j`. -/
  E : Fin nS → Fin L → F
  Gp : Fin nS → Fin tpad → F
  /-- `eq(r̄^r, l)`, and rep `r`'s extra-lane coins `ρ^r₀`, `ρ^r₁`. -/
  rbar : Fin 2 → Fin nL → K
  ρ : Fin 2 → Fin 2 → K
  /-- The clear values under rep `r`'s exposed pads, from `z = (w, u)` and the extra lanes (`e^r` reads them). -/
  vals : Fin 2 → W → (Fin nMask → Fin 128 → ZMod 2) → (Fin 2 → Fin 2 → Fin L → F) → Fin nE → F
  /-- The exposed indices of `final_a`, `final_b`, which the triple multiplies. -/
  ia : Fin nE
  ib : Fin nE
  /-- The sacrifice coin `ζ` and the inner proof's `β`. -/
  ζ : Fin 2 → F
  β : Fin 2 → F
  /-- Levels ≥ 1 from `T'` on (§2.4 line 15): a function of `y₁'^r` and fresh salts. -/
  lig : Fin 2 → (Fin L → K) → Hi → VL
  /-- The verifier's replay (line 16): the batched constraint `(λ, θ)` of rep `r`, from the messages before it. -/
  lam : Fin 2 → (Fin nC → Fin 128 → F) × (Fin 2 → Fin nE → F) × (Fin 2 → Fin L → K) × (Fin 2 → Fin tpad → K)
    × (Fin 2 → F × F) × (Fin 2 → VL) → PadIdx nE → F
  theta : Fin 2 → (Fin nC → Fin 128 → F) × (Fin 2 → Fin nE → F) × (Fin 2 → Fin L → K) × (Fin 2 → Fin tpad → K)
    × (Fin 2 → F × F) × (Fin 2 → VL) → F
  /-- The pads code at rep `r`'s opened positions: `Code(h ‖ pad)_k = Σ_i Ch k i·h_i + Σ_j Cp k j·pad_j`. -/
  Ch : Fin 2 → Fin nQ → PadIdx nE → F
  Cp : Fin 2 → Fin nQ → Fin nPad → F
  /-- hm96 leaves: a level-0 column, a pads column, `τ`'s commitment; and an ideal leaf from its seed. -/
  leaf0 : (Fin nL → F) × (Fin 2 → Fin 2 → F) → Salt → D
  leafP : F × F → Salt → D
  leafT : F → Salt → D
  ideal : Seed → D

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)

/-! ## The prover's randomness -/

/-- The mask slot `u` (§2.1): its words' bits. -/
abbrev Mask := Fin T.nMask → Fin 128 → ZMod 2

/-- The extra lanes `R^r_i` (rep `r` uses two), and their paddings `μ_{64+2r+i}`. -/
abbrev XLanes := Fin 2 → Fin 2 → Fin T.L → F
abbrev XPad := Fin 2 → Fin 2 → Fin T.tpad → F

/-- Every hm96 salt and ideal-leaf seed, and the salts of levels ≥ 1: level 0's salts at the opened positions, its ideal
seeds there (world `W₀`) and elsewhere; the pads trees' likewise; `y_τ` and an ideal seed for `τ`'s commitment (`W₀`);
levels ≥ 1. -/
abbrev Seeds := (Fin T.nS → Salt) × (Fin T.nS → Seed) × (Fin T.nU → Seed) × (Fin 2 → Fin T.nQ → Salt)
  × (Fin 2 → Fin T.nQ → Seed) × (Fin 2 → Fin T.nPU → Seed) × (Fin 2 → Salt) × (Fin 2 → Seed) × (Fin 2 → Hi)

/-- The prover's randomness (§2.2), in four blocks: the mask slot; the witness lanes' paddings with the extra lanes and
theirs; each rep's free pads (exposed ones, `h_tx`, `h_ty`); each rep's `pad_h`, `pad_μ`, `μ`, with every salt. -/
abbrev Rand := T.Mask × ((Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad) × (Fin 2 → (Fin T.nE → F) × F × F)
  × ((Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds)

/-! ## The prover's computations -/

variable [CharP F 2]

/-- `GF(2) ⊂ F`. -/
def castF : ZMod 2 →+* F := ZMod.castHom (dvd_refl 2) F

/-- **The masked claims' mask part** `Φ(u)`: `Φ_k(u)[v] = Σ_x eq(p_k, x)·u[x][v]`, GF(2)-linear in `u`. -/
def maskHom : T.Mask →+ (Fin T.nC → Fin 128 → F) where
  toFun u k v := ∑ x, T.eqw k x * castF (u x v)
  map_zero' := by
    funext k v
    simp
  map_add' u u' := by
    funext k v
    simp [Finset.sum_add_distrib, mul_add]

/-- **The prover's rank check** (§2.4 line 11), passing at `e`: the claims' weights on the mask slot span, so a GF(2)
vector over the mask-slot words reaches every vector of claim values. -/
def Rank : Prop :=
  Function.Surjective fun c : Fin T.nMask → ZMod 2 => fun k => ∑ x, T.eqw k x * castF (c x)

/-- The witness lanes' entries at the opened positions: `Enc(P_l)(q) + Pad(μ_l)(q)` (line 7). -/
def rows (w : W) (u : T.Mask) (μ0 : Fin T.nL → Fin T.tpad → F) : Fin T.nS → Fin T.nL → F :=
  fun q l => ∑ c, T.lanes w u l c * T.E q c + ∑ j, μ0 l j * T.Gp q j

/-- The extra lanes' entries at the opened positions. -/
def xrows (R : T.XLanes) (μR : T.XPad) : Fin T.nS → Fin 2 → Fin 2 → F :=
  fun q r i => ∑ c, R r i c * T.E q c + ∑ j, μR r i j * T.Gp q j

/-- `W^r(a, b) = ρ^r₀·a + ρ^r₁·b`: rep `r`'s combination of its two extra lanes (line 15). -/
def Wr (r : Fin 2) (a b : F) : K := T.ρ r 0 * T.embed a + T.ρ r 1 * T.embed b

/-- The folded witness lanes `y₁^r = Σ_l eq(r̄^r, l)·P_l`. -/
def y1 (w : W) (u : T.Mask) (r : Fin 2) : Fin T.L → K := fun c => ∑ l, T.rbar r l * T.embed (T.lanes w u l c)

/-- **The restarted table** `y₁'^r = y₁^r + ρ^r₀·R_{2r} + ρ^r₁·R_{2r+1}` (line 15). -/
def y1' (w : W) (u : T.Mask) (R : T.XLanes) (r : Fin 2) : Fin T.L → K :=
  fun c => T.y1 w u r c + T.Wr r (R r 0 c) (R r 1 c)

/-- `ȳ^r = Σ_l eq(r̄^r, l)·μ_l + ρ^r₀·μ_{64+2r} + ρ^r₁·μ_{64+2r+1}` (line 15). -/
def ybar (μ0 : Fin T.nL → Fin T.tpad → F) (μR : T.XPad) (r : Fin 2) : Fin T.tpad → K :=
  fun j => ∑ l, T.rbar r l * T.embed (μ0 l j) + T.Wr r (μR r 0 j) (μR r 1 j)

/-- A rep's full pad vector from its free pads, with the triple's products `h_ab := h_fa·h_fb`, `h_tw := h_tx·h_ty`
(line 0). -/
def hFull (h : (Fin T.nE → F) × F × F) : PadIdx T.nE → F
  | .inl i => h.1 i
  | .inr j => ![h.2.1, h.2.2, h.1 T.ia * h.1 T.ib, h.2.1 * h.2.2] j

/-- The triple's two messages `ρ_in = ζ·h_fa + h_tx`, `σ_in = h_fb + h_ty` (line 16). -/
def innerMsg (r : Fin 2) (h : (Fin T.nE → F) × F × F) : F × F :=
  (T.ζ r * h.1 T.ia + h.2.1, h.1 T.ib + h.2.2)

/-! ## The view -/

/-- **Everything the verifier sees from one masked table at fixed coins**: the pre-final messages and the final message
(§2.3–§2.5). Trees are given by their leaves: opened leaves `leaf(content; salt)` with their contents and salts, the
others ideal. -/
@[ext] structure View where
  /-- The masked ring-switch messages `ŝ_ab`, `ŝ_c` of both reps, and the region claims' `ŝ`. -/
  sHat : Fin T.nC → Fin 128 → F
  sReg : Fin T.nR → Fin 128 → F
  /-- Every masked value `v + h_i` of each rep, `final_c + h_fc` included. -/
  masked : Fin 2 → Fin T.nE → F
  /-- The restarted tables `y₁'^r`, `ȳ^r`, and levels ≥ 1. -/
  y1 : Fin 2 → Fin T.L → K
  ybar : Fin 2 → Fin T.tpad → K
  lig : Fin 2 → VL
  /-- The inner proof: `(ρ_in, σ_in)`, `Y₁`, `Y₂`, `τ`, `y_τ` and `τ`'s commitment. -/
  inner : Fin 2 → F × F
  Y1 : Fin 2 → PadIdx T.nE → F
  Y2 : Fin 2 → Fin T.nPad → F
  tau : Fin 2 → F
  ytau : Fin 2 → Salt
  tauCom : Fin 2 → D
  /-- Level 0: the opened entries of the witness lanes and the extra lanes, their salts, and the tree's leaves. -/
  rows : Fin T.nS → Fin T.nL → F
  xrows : Fin T.nS → Fin 2 → Fin 2 → F
  salts0 : Fin T.nS → Salt
  leaves0 : (Fin T.nS → D) × (Fin T.nU → D)
  /-- Each rep's pads tree: the opened columns `(C_h[k], C_μ[k])`, their salts, and the tree's leaves. -/
  ch : Fin 2 → Fin T.nQ → F
  cm : Fin 2 → Fin T.nQ → F
  saltsP : Fin 2 → Fin T.nQ → Salt
  leavesP : Fin 2 → (Fin T.nQ → D) × (Fin T.nPU → D)

/-- The messages the verifier's replay reads before the inner proof's constraint (`lam`, `theta`). -/
def View.pre (v : T.View) :
    (Fin T.nC → Fin 128 → F) × (Fin 2 → Fin T.nE → F) × (Fin 2 → Fin T.L → K) × (Fin 2 → Fin T.tpad → K)
      × (Fin 2 → F × F) × (Fin 2 → VL) :=
  (v.sHat, v.masked, v.y1, v.ybar, v.inner, v.lig)

/-- **The reference prover** for one masked table at fixed coins: every message it sends, on witness `w` and randomness
`ω` (§2.3–§2.5), with the final message's leaves opened and the others ideal (world `W₁`). -/
noncomputable def view (w : W) (ω : T.Rand) : T.View :=
  let u := ω.1
  let μ0 := ω.2.1.1
  let R := ω.2.1.2.1
  let μR := ω.2.1.2.2
  let h := ω.2.2.1
  let padh := ω.2.2.2.1
  let padμ := ω.2.2.2.2.1
  let μ := ω.2.2.2.2.2.1
  let s := ω.2.2.2.2.2.2
  let sHat : Fin T.nC → Fin 128 → F := T.nu w + T.maskHom u
  let masked : Fin 2 → Fin T.nE → F := fun r i => T.vals r w u R i + (h r).1 i
  let y1 := T.y1' w u R
  let yb := T.ybar μ0 μR
  let lig := fun r => T.lig r (y1 r) (s.2.2.2.2.2.2.2.2 r)
  let inner := fun r => T.innerMsg r (h r)
  let pre := (sHat, masked, y1, yb, inner, lig)
  let tau := fun r => ∑ i, T.lam r pre i * μ r i
  let ch := fun r k => ∑ i, T.Ch r k i * T.hFull (h r) i + ∑ j, T.Cp r k j * padh r j
  let cm := fun r k => ∑ i, T.Ch r k i * μ r i + ∑ j, T.Cp r k j * padμ r j
  { sHat, sReg := T.sReg w, masked, y1, ybar := yb, lig, inner
    Y1 := fun r i => T.hFull (h r) i + T.β r * μ r i
    Y2 := fun r j => padh r j + T.β r * padμ r j
    tau, ytau := s.2.2.2.2.2.2.1, tauCom := fun r => T.leafT (tau r) (s.2.2.2.2.2.2.1 r)
    rows := T.rows w u μ0, xrows := T.xrows R μR, salts0 := s.1
    leaves0 := (fun q => T.leaf0 (T.rows w u μ0 q, T.xrows R μR q) (s.1 q), fun q => T.ideal (s.2.2.1 q))
    ch, cm, saltsP := s.2.2.2.1
    leavesP := fun r => (fun k => T.leafP (ch r k, cm r k) (s.2.2.2.1 r k), fun k => T.ideal (s.2.2.2.2.2.1 r k)) }

/-! ## The simulator -/

/-- `S_shvzk`'s randomness: the dummy run's randomness (§3.2 step 1), each block beside the fresh values that replace
its outputs: fresh masked `ŝ` (step 3); fresh masked values with fresh `ρ_in`, `σ_in` (steps 2, 4, 6); fresh `C_h`, `Y₂`,
`Y₁` (step 6). -/
abbrev SimRand := (T.Mask × (Fin T.nC → Fin 128 → F)) × ((Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad)
  × ((Fin 2 → (Fin T.nE → F) × F × F) × (Fin 2 → (Fin T.nE → F) × F × F))
  × (((Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds)
      × (Fin 2 → Fin T.nQ → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F))

/-- The dummy run's randomness inside `S_shvzk`'s. -/
def dummyOf (σ : T.SimRand) : T.Rand := (σ.1.1, σ.2.1, σ.2.2.1.1, σ.2.2.2.1)

/-- **`S_shvzk(x, e)`** (§3.2), on the dummy witness `w₀`: the prover's run on `w₀` with fresh randomness, keeping its
level-0 openings and leaves, its restarted tables and levels ≥ 1 (step 1); fresh masked `ŝ` and the public region values
(step 3); fresh masked values, `ρ_in`, `σ_in`, `Y₁`, `Y₂`, `C_h` (steps 2, 4, 6); `τ := (⟨λ, Y₁⟩ − θ)/β` and
`C_μ := (Code(Y) − C_h)/β` from the verifier's replay on the simulated messages (steps 5, 6); the pads tree with the
simulated columns at the opened positions, ideal leaves elsewhere, and `τ`'s commitment under the dummy run's fresh
salt. -/
noncomputable def sim (w0 : W) (σ : T.SimRand) : T.View :=
  let d := T.view w0 (T.dummyOf σ)
  let sHat := σ.1.2
  let fr := σ.2.2.1.2
  let masked : Fin 2 → Fin T.nE → F := fun r => (fr r).1
  let inner : Fin 2 → F × F := fun r => (fr r).2
  let ch := σ.2.2.2.2.1
  let Y2 := σ.2.2.2.2.2.1
  let Y1 := σ.2.2.2.2.2.2
  let pre := (sHat, masked, d.y1, d.ybar, inner, d.lig)
  let tau := fun r => (∑ i, T.lam r pre i * Y1 r i - T.theta r pre) / T.β r
  let cm := fun r k => (∑ i, T.Ch r k i * Y1 r i + ∑ j, T.Cp r k j * Y2 r j - ch r k) / T.β r
  { d with
    sHat, sReg := T.pubReg, masked, inner, Y1, Y2, tau
    tauCom := fun r => T.leafT (tau r) (d.ytau r)
    ch, cm
    leavesP := fun r => (fun k => T.leafP (ch r k, cm r k) (d.saltsP r k), (d.leavesP r).2) }

/-! ## What the theorems assume of the coins, the codes and the witness -/

/-- The inverse of `W^r`, where it is invertible. -/
noncomputable def Winv (r : Fin 2) : K → F × F := Function.invFun fun p : F × F => T.Wr r p.1 p.2

/-- (★) solved for rep `r`'s extra lanes at opened position `q`: `Enc(y₁'^r)(q) + Pad(ȳ^r)(q) − Σ_l eq(r̄^r, l)·row_q[l]`. -/
def xTarget (rows : Fin T.nS → Fin T.nL → F) (y1 : Fin 2 → Fin T.L → K) (yb : Fin 2 → Fin T.tpad → K) (q : Fin T.nS)
    (r : Fin 2) : K :=
  ∑ c, y1 r c * T.embed (T.E q c) + ∑ j, yb r j * T.embed (T.Gp q j) - ∑ l, T.rbar r l * T.embed (rows q l)

/-- The level-0 padding reaches every vector of values at the opened positions (T4: at most `t_pad` distinct positions,
where `X_L + κ` vanishes at none; `pad_onto_M1` for C-Flock ZK's code). -/
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
