import Definitions.Pouw.TileBound.Defs
import Definitions.Pouw.TileBound.Fp8Defs
/-!
# The tile-cost bound's guarantees (trusted)

Each guarantee is a `def G : Prop`, with the docstring, binders and `open`s of the statement in its source
module. Its proof is the theorem `Pouw.SecurityProofs.TileBound.G : Pouw.Guarantees.TileBound.G`.
-/

namespace Pouw.TileBound
open Finset

/-- BF16 or FP16 at 3 entries: the count admits a cheaper program. -/
def _root_.Pouw.Guarantees.TileBound.Bf16OpenAt8 : Prop :=
  price .bf16 < (3 : ℕ) * price .fp8

end Pouw.TileBound

namespace Pouw.TileBound

/-- **BF16 or FP16 sub-products undercut the honest E4M3 chain exactly from depth 6.** -/
def _root_.Pouw.Guarantees.TileBound.Bf16PaysIff : Prop :=
  ∀ (L : ℕ),
    price .bf16 * (7 / 8 : ℚ) ^ L < price .fp8 ↔ 6 ≤ L

end Pouw.TileBound

namespace Pouw.TileBound

/-- A paying BF16 or FP16 recursion needs an all-exact region of at least 1,024 × 512 words over at least 1,024 of k. -/
def _root_.Pouw.Guarantees.TileBound.Bf16RegionOfPays : Prop :=
  ∀ {L : ℕ} (h : price .bf16 * (7 / 8 : ℚ) ^ L < price .fp8),
    1024 ≤ 16 * 2 ^ L ∧ 512 ≤ 8 * 2 ^ L

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **Theorem 1, bilinear form.** A bilinear program correct on the live entries reads at least `nLive` X-blocks over its
products. -/
def _root_.Pouw.Guarantees.TileBound.BilinSupport : Prop :=
  ∀ {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] (P : BilinProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : P.toQuad.CrossOn liveX liveY) (BX : XBlocks m k β),
    nLive liveX liveY ≤ ∑ t, BX.touches (P.u t)

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

open Classical in
/-- **Per block.** The live products in X-block `b` are at most the (product, side) pairs whose X-part reads `b`. -/
def _root_.Pouw.Guarantees.TileBound.BlockCount : Prop :=
  ∀ {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : P.CrossOn liveX liveY) (B : XBlocks m k β) (b : β),
    Fintype.card {x // LiveIn B liveX liveY b x} ≤ Fintype.card {q // SideIn P B b q}

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **The cost bound.** A slot program correct on the live entries, under the representability rule, costs at least
FP4's price for every live product, provided every format it uses has `capacity f · price .fp4 ≤ price f`. -/
def _root_.Pouw.Guarantees.TileBound.CostGe : Prop :=
  ∀ {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] {γ : Type*} [Fintype γ] (P : SlotProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : P.toQuadProg.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (hr : P.Representable BX BY) (hf : ∀ t, (capacity (P.fmt t) : ℚ) * price .fp4 ≤ price (P.fmt t)),
    price .fp4 * nLive liveX liveY ≤ P.cost

end Pouw.TileBound

namespace Pouw.TileBound
open Finset

/-- **The E4M3 cost bound.** Under the representability rule at E4M3, a slot program correct on the live entries costs
at least E4M3's price per live product, provided every format it uses has `capacity8 f · price .fp8 ≤ price f`. -/
def _root_.Pouw.Guarantees.TileBound.CostGe8 : Prop :=
  ∀ {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] {γ : Type*} [Fintype γ] (P : SlotProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : P.toQuadProg.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (hr : P.Representable8 BX BY) (hf : ∀ t, (capacity8 (P.fmt t) : ℚ) * price .fp8 ≤ price (P.fmt t)),
    price .fp8 * nLive liveX liveY ≤ P.cost

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **The cost bound, per-slot supports.** A slot program correct on the live entries whose slot `t` has both inputs
reading at most `s t` blocks costs at least FP4's price for every live product, provided each slot's price covers
`s t` FP4 slots. The shared lemma at `p₀ = price .fp4`. -/
def _root_.Pouw.Guarantees.TileBound.CostGeSupport : Prop :=
  ∀ {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] {γ : Type*} [Fintype γ] (P : SlotProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : P.toQuadProg.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (s : τ → ℕ) (hs : ∀ t, BX.touches (P.a t) + BY.touches (P.b t) ≤ s t ∧ BX.touches (P.a' t) + BY.touches (P.b' t) ≤ s t) (hf : ∀ t, (s t : ℚ) * price .fp4 ≤ price (P.fmt t)),
    price .fp4 * nLive liveX liveY ≤ P.cost

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **The shared cost lemma** (bc-b58c6093's, moved here from its FP8-tile files). A quadratic program correct on the live
entries, whose product `t` has both inputs reading at most `s t` blocks, costs at least `p₀` per live product under
any per-product prices `c` with `s t · p₀ ≤ c t`. -/
def _root_.Pouw.Guarantees.TileBound.CostGeSupportAt : Prop :=
  ∀ {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] {γ : Type*} [Fintype γ] (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : P.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (s : τ → ℕ) (hs : ∀ t, BX.touches (P.a t) + BY.touches (P.b t) ≤ s t ∧ BX.touches (P.a' t) + BY.touches (P.b' t) ≤ s t) (c : τ → ℚ) (p₀ : ℚ) (hp₀ : 0 ≤ p₀) (hf : ∀ t, (s t : ℚ) * p₀ ≤ c t),
    p₀ * nLive liveX liveY ≤ ∑ t, c t

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **Correct outputs at three inputs give `CrossOn`.** A quadratic program whose output is `X·Y` at the unit and zero
inputs `(e_x, e_y)`, `(e_x, 0)` and `(0, e_y)`, for every live X-entry `x` and live Y-entry `y`, satisfies `CrossOn` on the
live entries. Those inputs are FP4-representable (E2M1 has 0 and 1, and a UE4M3 scale 1), so a tile program's firings
can carry them. The fourth input, `(0, 0)`, isn't needed. -/
def _root_.Pouw.Guarantees.TileBound.CrossOnOfEval : Prop :=
  ∀ {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : ∀ x, liveX x → ∀ y, liveY y → ∀ o, P.eval (unitForm x) (unitForm y) o = matMul (unitForm x) (unitForm y) o ∧ P.eval (unitForm x) 0 o = matMul (unitForm x) 0 o ∧ P.eval 0 (unitForm y) o = matMul 0 (unitForm y) o),
    P.CrossOn liveX liveY

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **The X-side count.** A quadratic program correct on the live entries has at least `nLive` (product, side, X-block)
incidences. -/
def _root_.Pouw.Guarantees.TileBound.CrossXCount : Prop :=
  ∀ {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : P.CrossOn liveX liveY) (B : XBlocks m k β),
    nLive liveX liveY ≤ ∑ t, (B.touches (P.a t) + B.touches (P.a' t))

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **The Y-side count.** -/
def _root_.Pouw.Guarantees.TileBound.CrossYCount.{u_1, u_2, u_4} : Prop :=
  ∀ {K : Type u_1} [Field K] {m k n : ℕ} {τ : Type u_2} [Fintype τ] {γ : Type u_4} [Fintype γ] (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : P.CrossOn liveX liveY) (B : YBlocks k n γ),
    nLive liveX liveY ≤ ∑ t, (B.touches (P.b t) + B.touches (P.b' t))

end Pouw.TileBound

namespace Pouw.TileBound
open Finset

/-- **Which formats the E4M3 bound closes:** exactly NVFP4 (capacity 0 on in-domain operands) and E4M3. -/
def _root_.Pouw.Guarantees.TileBound.Fmt8ClosedIff : Prop :=
  ∀ (f : Fmt),
    (capacity8 f : ℚ) * price .fp8 ≤ price f ↔ f = .fp4 ∨ f = .fp8

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **Which formats the representability rule closes:** exactly FP4, FP8 and BF16. FP16 and int8 at their prices, and
`dp4a`, hold more blocks than they pay for (`open_fp16`, `open_int8`, `open_dp4a`). -/
def _root_.Pouw.Guarantees.TileBound.FmtClosedIff : Prop :=
  ∀ (f : Fmt),
    (capacity f : ℚ) * price .fp4 ≤ price f ↔ f = .fp4 ∨ f = .fp8 ∨ f = .bf16

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **FP32 (FFMA) pre-adds reading at most 16 blocks are closed, as a tie:** 16 FP4 slots cost 8 units, FFMA 8.46 in the
model and 8.0–8.15 measured (16.0–16.3 FP4 slots, GPU 4 F1). At the bottom of that range the two are equal; the
statement is `≤`, so it holds across it. A Strassen level-4 pre-add reads 16 blocks. -/
def _root_.Pouw.Guarantees.TileBound.Fp32ClosedUpto : Prop :=
  (16 : ℚ) * price .fp4 ≤ price .fp32

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- From 17 blocks per input, FFMA's price is below the count. That holds for any price below 8.5 units (17 FP4 slots),
so for the model's 8.46 and the measured 8.0–8.15 alike. At 8.5 or more the closure would reach 17 blocks; below 8.0
(16 slots) `fp32_closed_upto` would fail and the closure would stop at 15. -/
def _root_.Pouw.Guarantees.TileBound.Fp32OpenAt : Prop :=
  price .fp32 < (17 : ℕ) * price .fp4

end Pouw.TileBound

namespace Pouw.TileBound
open Finset

/-- FFMA at 9 entries. -/
def _root_.Pouw.Guarantees.TileBound.Fp32OpenAt8 : Prop :=
  price .fp32 < (9 : ℕ) * price .fp8

end Pouw.TileBound

namespace Pouw.TileBound

/-- A paying FFMA recursion needs more than 8,192 rows of output. -/
def _root_.Pouw.Guarantees.TileBound.Fp32RegionOfPays : Prop :=
  ∀ {L : ℕ} (h : price .fp32 * (7 / 8 : ℚ) ^ L < price .fp8),
    8192 < 16 * 2 ^ L

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

def _root_.Pouw.Guarantees.TileBound.NLiveAll : Prop :=
  ∀ {m k n : ℕ},
    nLive (m := m) (k := k) (n := n) (fun _ => True) (fun _ => True) = m * k * n

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- FP32 (FFMA) is not closed: 2¹⁸ blocks per value at about 16–17 FP4 slots' price (16.92 at the model's 8.46; 16.0–16.3
measured). -/
def _root_.Pouw.Guarantees.TileBound.OpenFp32 : Prop :=
  price .fp32 < capacity .fp32 * price .fp4

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- TF32 is not closed: 32 blocks per value at 8 FP4 slots' price. -/
def _root_.Pouw.Guarantees.TileBound.OpenTf32 : Prop :=
  price .tf32 < capacity .tf32 * price .fp4

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **Theorem 1, quadratic form.** A quadratic program correct on the live entries reads, over its products' two
inputs, at least `2·nLive` blocks. No factor of 2 is lost against bilinear programs. -/
def _root_.Pouw.Guarantees.TileBound.QuadSupport : Prop :=
  ∀ {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] {γ : Type*} [Fintype γ] (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : P.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ),
    2 * nLive liveX liveY ≤ ∑ t, ((BX.touches (P.a t) + BY.touches (P.b t)) +
    (BX.touches (P.a' t) + BY.touches (P.b' t)))

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **TF32 pre-adds reading at most 8 blocks are closed** (at a tie): 8 FP4 slots' price is TF32's. A Strassen level-3
pre-add reads 8 blocks. -/
def _root_.Pouw.Guarantees.TileBound.Tf32ClosedUpto : Prop :=
  (8 : ℚ) * price .fp4 ≤ price .tf32

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- From 9 blocks per input, TF32's price is below the count (`support_admits_open`). -/
def _root_.Pouw.Guarantees.TileBound.Tf32OpenAt : Prop :=
  price .tf32 < (9 : ℕ) * price .fp4

end Pouw.TileBound

namespace Pouw.TileBound
open Finset

/-- TF32 at 5 entries. -/
def _root_.Pouw.Guarantees.TileBound.Tf32OpenAt8 : Prop :=
  price .tf32 < (5 : ℕ) * price .fp8

end Pouw.TileBound

namespace Pouw.TileBound

/-- A paying TF32 recursion needs more than 8,192 rows of output. -/
def _root_.Pouw.Guarantees.TileBound.Tf32RegionOfPays : Prop :=
  ∀ {L : ℕ} (h : price .tf32 * (7 / 8 : ℚ) ^ L < price .fp8),
    8192 < 16 * 2 ^ L

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **The tile cost bound.** Such a tile program costs at least FP4's per-slot price for every live product. -/
def _root_.Pouw.Guarantees.TileBound.TileCostGe.{u_1, u_3} : Prop :=
  ∀ {K : Type u_1} [Field K] {m k n : ℕ} {β : Type u_3} [Fintype β] {G : ℕ} (P : TileProg K m k n G) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : P.toBilin.toQuad.CrossOn liveX liveY) (BX : XBlocks m k β) (hr : P.Representable BX),
    price .fp4 * nLive liveX liveY ≤ P.cost

end Pouw.TileBound

namespace Pouw.TileBound
open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

/-- **The tile count.** A tile program correct on the live entries, whose `P`-inputs each read at most one X-block,
fires at least `nLive / 8192` gates. -/
def _root_.Pouw.Guarantees.TileBound.TileCount.{u_1, u_3} : Prop :=
  ∀ {K : Type u_1} [Field K] {m k n : ℕ} {β : Type u_3} [Fintype β] {G : ℕ} (P : TileProg K m k n G) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop} (h : P.toBilin.toQuad.CrossOn liveX liveY) (BX : XBlocks m k β) (hr : P.Representable BX),
    nLive liveX liveY ≤ G * gateSlots

end Pouw.TileBound
