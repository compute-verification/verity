import Definitions.Pouw.RelativeSecurity
import Definitions.Pouw.RelativeSecurityFork
import Definitions.Pouw.Assumptions.RelativeSecurity

/-!
# Relative security's guarantees for the staged window (trusted)

Each guarantee is a `def G : Prop` over `Definitions.Pouw.RelativeSecurity` with the named premises it rests on as
hypotheses (`Definitions.Pouw.Assumptions.RelativeSecurity`). Its proof is the theorem
`Pouw.SecurityProofs.RelativeSecurity.G : Pouw.Guarantees.RelativeSecurity.G`.

The four floors are `c = 1` on K's two proved scopes:
- the floors of record, one-sided operands of any form, charge-pure, at the parity luck 0.854
  (`StagedFloorKOneSided`, `OverlapFloorKOneSided`);
- their siblings, affine operands, row-pure, at ¾ (`StagedFloorK`, `OverlapFloorK`), proved on their own scope. The
  affine scope is not inside the one-sided one (`(x + y)²`, for `x` one of U's bits and `y` a coin, is `AffProd` and
  not `OneSided`), so these are not derived from the floors of record: the same chain proves each on its scope.

K1 is not claimed outside the two proved scopes, and no guarantee here covers a run outside them. Theorem O's priced
floor holds there on paper only. The verifier can't observe which case a run is in.

The four floors rest on `ServiceSoundStaged` and `ServiceSoundOverlap`, which are false below 1 at SHA-512 (an
adversary that hard-codes a collision breaks binding in every round). Each has a restatement per prover beside it,
proved (`StagedFloorKPerProver`, `StagedFloorKOneSidedPerProver`, `OverlapFloorKPerProver`,
`OverlapFloorKOneSidedPerProver`): the same floor and the same luck, on the premises per prover, with each adversary's
error its own (`breakPr`), read at the fork finders built from that adversary (`forkFinders`), and each value of its
coins capped at 1. `ecr/sha-512` applies to
those finders outside the floors. Each floor per prover has an honest lemma (its premises hold together, with an honest
answer accepted and a wrong one refused: `StagedFloorPerProverHonest`, `OverlapFloorPerProverHonest`,
`StagedFloorJ10RPerProverHonest`, `OverlapFloorJ10RPerProverHonest`) and a nontrivial one (its conclusion fails at a
verifier that accepts everything: `FloorPerProverNontrivial`).
-/

namespace Pouw.Guarantees.RelativeSecurity

open Pouw.RelativeSecurity Pouw.Assumptions

/-- **Theorem S for the staged spot check**, with no premise. In every staged play (`R` rounds of `M` stages, each
stage's error table fixed before its draw, one check a round of a drawn stage and cell), if every stage's error is of
degree at most 2 on the cube, the play is accepted with deficiency at least `D` with probability at most
`e^{−D/4}`: the luck ¾ (a nonzero degree-2 error is nonzero on a quarter of the cube), through the stages'
supermartingale. -/
def TheoremSStaged : Prop :=
  ∀ (C V A : Type) [Fintype C] [Fintype V] [DecidableEq V] [CommRing A] (M R : ℕ) (P : StagedPlay C V A M R),
    (∀ x w s c, CubeQuad (P.errAt x w s c)) → ∀ D : ℝ,
      prob (fun x => P.game.accepts x ∧ D ≤ P.game.deficiency x) ≤ Real.exp (-(D / 4))

/-- **The staged window's service floor for affine operands**: Theorem K at chain factor 1 (Lemma G, s32 wrap read
mod 2), the luck ¾ and the stage floors summed. In every machine `Mc` and for every service `S` whose seed is `R`
rounds of `St` stage draws on the re-encoded design (7 planes at `wRe`, integers mod `2^32`), under the three
premises, every adversary that commits first has its accepted runs charged below
`(R − 160 ln 2) · St · κ L n n' B − Vmax` with probability at most `2^{−40} + R (δb + δz)`: `c = 1` on K's proved
affine scope (`KScopeAffine`: affine operands, mixed included, row-pure). It is a sibling of the floor of record
`StagedFloorKOneSided`, proved on its own scope, which isn't that floor's, at the sharper luck. -/
def StagedFloorK : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St n L B n' : ℕ) (e : S.Seed ≃ Paths (DStage St n L 7 B n') R)
    (run : RunMap Mc S R St n L 7 B n' (ZMod (2 ^ 32))) (δb δz κ Vmax : ℝ),
    0 < St → 0 < κ → 0 < L * (n * n' * B) → 0 ≤ δb → 0 ≤ δz →
    ServiceSoundStaged Mc S run e δb δz → KScopeAffine Mc S run wRe → ChargeVF3j16 Mc S run e κ Vmax →
    ServiceFloor Mc S ((R - 160 * Real.log 2) * (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) - Vmax)
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz))

/-- **The staged window's service floor for one-sided operands of any form**, a floor of record: the parity luck
0.854 in the product deficit (RS-E-REF r6 VI.2), its count by charging purity (a cell-block not short of products is
charged `L` of them, none twice) and the stage floors summed. In every machine `Mc` and for every service `S` whose
seed is `R` rounds of `St` stage draws on the re-encoded design (7 planes at `wRe`, integers mod `2^32` read mod 2 by
`toF2`, plane 0 at an odd weight), under the three premises, every adversary that commits first has its accepted runs
charged below `(R − 274 ln 2) · St · κ L n n' B − Vmax` with probability at most `2^{−40} + R (δb + δz)`
(`0.146 · 274 ≥ 40`): `c = 1` on K's proved one-sided scope (`KScopeOneSided`, a sibling of the ledger's K1). -/
def StagedFloorKOneSided : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St n L B n' : ℕ) (e : S.Seed ≃ Paths (DStage St n L 7 B n') R)
    (run : RunMap Mc S R St n L 7 B n' (ZMod (2 ^ 32))) (δb δz κ Vmax : ℝ),
    0 < St → 0 < κ → 0 < L * (n * n' * B) → 0 ≤ δb → 0 ≤ δz →
    ServiceSoundStaged Mc S run e δb δz → KScopeOneSided Mc S run wRe toF2 0 → ChargeVF3j16 Mc S run e κ Vmax →
    ServiceFloor Mc S ((R - 274 * Real.log 2) * (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) - Vmax)
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz))

/-- **Theorem S with overlapping draws**, with no premise. In every play over `St` draws a round (`R` rounds, each
round's error tables fixed before its first draw, one check a round of a cell-block drawn uniformly among every
stage's, at every draw's drawn bits), if every cell-block's error is of degree at most 2 on the cube of its own draw's
bits at every bit point of the other draws', the play is accepted with deficiency `ovDef` at least `D` with
probability at most `e^{−D/4}`: the luck ¾ per draw, with the other draws' bits given. This is the luck
`OverlapFloorK` reads. -/
def TheoremSOverlap : Prop :=
  ∀ (St : ℕ) (κ ι : Fin St → Type) [∀ t, Fintype (κ t)] [∀ t, Fintype (ι t)] [∀ t, DecidableEq (ι t)] (A : Type)
    [CommRing A] (R : ℕ) (P : OvPlay κ ι A R), (∀ x w c U, CubeQuad fun z => P.errAt x w c (mergeAt U c.1 z)) →
    ∀ D : ℝ, prob (fun x => P.game.accepts x ∧ D ≤ P.game.deficiency x) ≤ Real.exp (-(D / 4))

/-- **The overlapped window's service floor for operands affine per draw**: Theorem K∥ (K at chain factor 1 per row
of one draw at every bit point of the other draws', Lemma G over F₂), the luck ¾ per draw and every stage's
cell-blocks counted once. In every machine `Mc` and for every service `S` whose seed is `R` rounds of `St` stage
draws of any shapes (stage `t`: `n t` rows, `B t` blocks of `L`, `n' t` columns, 7 planes at `wRe`, integers mod
`2^32`) overlapping at any depth, its check's cell-block drawn uniformly among every stage's (`ServiceSoundOverlap`'s
deployment requirement), under the three premises, every adversary that commits first has its accepted runs charged
below `(R − 160 ln 2) · κ L Σ_t n_t n'_t B_t − Vmax` with probability at most `2^{−40} + R (δb + δz)`: `c = 1` on
K∥'s proved affine scope (`KScopeOverlapAffine`). It is a sibling of the floor of record `OverlapFloorKOneSided`,
proved on its own scope, which isn't that floor's, at the sharper luck. -/
def OverlapFloorK : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St L : ℕ) (n B n' : Fin St → ℕ) (e : S.Seed ≃ Paths (OSpot St n L 7 B n') R)
    (run : ORunMap Mc S R St n L 7 B n' (ZMod (2 ^ 32))) (δb δz κ Vmax : ℝ),
    0 < κ → 0 < L * ∑ t, n t * n' t * B t → 0 ≤ δb → 0 ≤ δz →
    ServiceSoundOverlap Mc S run e δb δz → KScopeOverlapAffine Mc S run wRe →
      ChargeVF3j16Overlap Mc S run e κ Vmax →
    ServiceFloor Mc S ((R - 160 * Real.log 2) * (κ * L * ((∑ t, n t * n' t * B t : ℕ) : ℝ)) - Vmax)
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz))

/-- **The overlapped window's service floor for one-sided operands of any form per draw**, a floor of record: the
parity luck 0.854 per draw in the product deficit (RS-E-REF r6 VI.2), its count by charging purity across draws at
every bit point of the draws, and every stage's cell-blocks counted once. In every machine `Mc` and for every
service `S` whose seed is `R` rounds of `St` stage draws of any shapes (as `OverlapFloorK`; integers mod `2^32` read
mod 2 by `toF2`, plane 0 at an odd weight) overlapping at any depth, its check's cell-block drawn uniformly among every
stage's (`ServiceSoundOverlap`'s deployment requirement), under the three premises, every adversary that commits
first has its accepted runs charged below `(R − 274 ln 2) · κ L Σ_t n_t n'_t B_t − Vmax` with probability at most
`2^{−40} + R (δb + δz)`: `c = 1` on K∥'s proved one-sided scope (`KScopeOverlapOneSided`). -/
def OverlapFloorKOneSided : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St L : ℕ) (n B n' : Fin St → ℕ) (e : S.Seed ≃ Paths (OSpot St n L 7 B n') R)
    (run : ORunMap Mc S R St n L 7 B n' (ZMod (2 ^ 32))) (δb δz κ Vmax : ℝ),
    0 < κ → 0 < L * ∑ t, n t * n' t * B t → 0 ≤ δb → 0 ≤ δz →
    ServiceSoundOverlap Mc S run e δb δz → KScopeOverlapOneSided Mc S run wRe toF2 0 →
      ChargeVF3j16Overlap Mc S run e κ Vmax →
    ServiceFloor Mc S ((R - 274 * Real.log 2) * (κ * L * ((∑ t, n t * n' t * B t : ℕ) : ℝ)) - Vmax)
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz))

/-! ### The floors per prover

Each restates one floor above on the service premise per prover (`ServiceSoundStagedPerProver`,
`ServiceSoundOverlapPerProver`) and concludes it adversary by adversary (`ServiceFloorPerProver`): the same floor `F`
and luck `2^{−40}`, every adversary in one class, with no budget on its time. The premise is read at one finder built
from the adversary, the fork (`forkFinders`, one adversary class): at `k = 1` for binding breaks, two runs
of the adversary from its commitment forked at the round, and at `kz` continuations for zero-knowledge breaks, the
form of C-Flock's link finder. The service is a `HashedService`, so the finder reads the verifier's SHA-512 inputs in
those runs and outputs their first collision. Each adversary's error is
`breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) 2^{−40} δb βb δz βz Ω M`: at each value `r` of its
coins, `2^{−40} + R (δb + δz)` plus `βb` and `βz` times its fork finders' collision probabilities summed over the
rounds, capped at 1 and averaged over `r`. Nothing here assumes anything of SHA-512: `ecr/sha-512`
(`Claims.ExpectedCR`) applies to the fork finders outside the floors, at their cost (the verifier's calls and the
adversary's runs at its time), and turns each probability into a number, or `⊤` at an adversary whose finder fails
it (`breakBound`, `Pouw.RelativeSecurity.breakPr_le_breakBound`). A machine that hard-codes a collision gets about 1:
its forks open a committed leaf two ways, and the verifier hashes both openings. At `βb = βz = 0` the error is the old
floor's, capped at 1, for every adversary. -/

/-- `StagedFloorK` per prover: the staged window's service floor for affine operands, at each adversary's own error.
In every machine `Mc` and for every service `S` whose seed is `R` rounds of `St` stage draws on the re-encoded design
(7 planes at `wRe`, integers mod `2^32`), under `ServiceSoundStagedPerProver` at the fork finders (`forkFinders`, `1`
and `kz` continuations) and `StagedFloorK`'s other two premises, every adversary `M` that commits first has its accepted
runs charged below `(R − 160 ln 2) · St · κ L n n' B − Vmax` with probability at most
`breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) 2^{−40} δb βb δz βz Ω M`. -/
def StagedFloorKPerProver : Prop :=
  ∀ (Mc : Machine) (S : HashedService) (R St n L B n' : ℕ) (e : S.Seed ≃ Paths (DStage St n L 7 B n') R)
    (run : RunMap Mc S.toService R St n L 7 B n' (ZMod (2 ^ 32))) (sha : Claims.Msg → Claims.Msg) (kz : ℕ)
    (δb βb δz βz κ Vmax : ℝ),
    0 < St → 0 < κ → 0 < L * (n * n' * B) → 0 ≤ δb → 0 ≤ βb → 0 ≤ δz → 0 ≤ βz →
    ServiceSoundStagedPerProver Mc S.toService run e sha (forkFinders S e sha 1) (forkFinders S e sha kz)
      δb βb δz βz → KScopeAffine Mc S.toService run wRe →
      ChargeVF3j16 Mc S.toService run e κ Vmax →
    ServiceFloorPerProver Mc S.toService ((R - 160 * Real.log 2) * (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) - Vmax)
      (breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) ((2 : ℝ) ^ 40)⁻¹ δb βb δz βz)

/-- `StagedFloorKOneSided` per prover, the staged floor of record at each adversary's own error. In every machine
`Mc` and for every service `S` whose seed is `R` rounds of `St` stage draws on the re-encoded design (7 planes at
`wRe`, integers mod `2^32` read mod 2 by `toF2`, plane 0 at an odd weight), under `ServiceSoundStagedPerProver` at the
fork finders (`1` and `kz` continuations) and `StagedFloorKOneSided`'s other two premises, every adversary `M` that
commits first has its accepted runs charged below `(R − 274 ln 2) · St · κ L n n' B − Vmax` with probability at most
`breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) 2^{−40} δb βb δz βz Ω M`. -/
def StagedFloorKOneSidedPerProver : Prop :=
  ∀ (Mc : Machine) (S : HashedService) (R St n L B n' : ℕ) (e : S.Seed ≃ Paths (DStage St n L 7 B n') R)
    (run : RunMap Mc S.toService R St n L 7 B n' (ZMod (2 ^ 32))) (sha : Claims.Msg → Claims.Msg) (kz : ℕ)
    (δb βb δz βz κ Vmax : ℝ),
    0 < St → 0 < κ → 0 < L * (n * n' * B) → 0 ≤ δb → 0 ≤ βb → 0 ≤ δz → 0 ≤ βz →
    ServiceSoundStagedPerProver Mc S.toService run e sha (forkFinders S e sha 1) (forkFinders S e sha kz)
      δb βb δz βz → KScopeOneSided Mc S.toService run wRe toF2 0 →
      ChargeVF3j16 Mc S.toService run e κ Vmax →
    ServiceFloorPerProver Mc S.toService ((R - 274 * Real.log 2) * (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) - Vmax)
      (breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) ((2 : ℝ) ^ 40)⁻¹ δb βb δz βz)

/-- `OverlapFloorK` per prover: the overlapped window's service floor for operands affine per draw, at each
adversary's own error. In every machine `Mc` and for every service `S` whose seed is `R` rounds of `St` stage draws
of any shapes overlapping at any depth (as `OverlapFloorK`), under `ServiceSoundOverlapPerProver` at the fork
finders (`1` and `kz` continuations) and `OverlapFloorK`'s other two premises, every adversary `M` that commits first
has its accepted runs charged below `(R − 160 ln 2) · κ L Σ_t n_t n'_t B_t − Vmax` with probability at most
`breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) 2^{−40} δb βb δz βz Ω M`. -/
def OverlapFloorKPerProver : Prop :=
  ∀ (Mc : Machine) (S : HashedService) (R St L : ℕ) (n B n' : Fin St → ℕ)
    (e : S.Seed ≃ Paths (OSpot St n L 7 B n') R)
    (run : ORunMap Mc S.toService R St n L 7 B n' (ZMod (2 ^ 32))) (sha : Claims.Msg → Claims.Msg) (kz : ℕ)
    (δb βb δz βz κ Vmax : ℝ),
    0 < κ → 0 < L * ∑ t, n t * n' t * B t → 0 ≤ δb → 0 ≤ βb → 0 ≤ δz → 0 ≤ βz →
    ServiceSoundOverlapPerProver Mc S.toService run e sha (forkFinders S e sha 1) (forkFinders S e sha kz)
      δb βb δz βz → KScopeOverlapAffine Mc S.toService run wRe →
      ChargeVF3j16Overlap Mc S.toService run e κ Vmax →
    ServiceFloorPerProver Mc S.toService
      ((R - 160 * Real.log 2) * (κ * L * ((∑ t, n t * n' t * B t : ℕ) : ℝ)) - Vmax)
      (breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) ((2 : ℝ) ^ 40)⁻¹ δb βb δz βz)

/-- `OverlapFloorKOneSided` per prover, the overlapped floor of record at each adversary's own error. In every
machine `Mc` and for every service `S` whose seed is `R` rounds of `St` stage draws of any shapes overlapping at any
depth (as `OverlapFloorKOneSided`), under `ServiceSoundOverlapPerProver` at the fork finders (`1` and `kz`
continuations) and `OverlapFloorKOneSided`'s other two premises, every adversary `M` that commits first has its accepted runs charged
below `(R − 274 ln 2) · κ L Σ_t n_t n'_t B_t − Vmax` with probability at most
`breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) 2^{−40} δb βb δz βz Ω M`. -/
def OverlapFloorKOneSidedPerProver : Prop :=
  ∀ (Mc : Machine) (S : HashedService) (R St L : ℕ) (n B n' : Fin St → ℕ)
    (e : S.Seed ≃ Paths (OSpot St n L 7 B n') R)
    (run : ORunMap Mc S.toService R St n L 7 B n' (ZMod (2 ^ 32))) (sha : Claims.Msg → Claims.Msg) (kz : ℕ)
    (δb βb δz βz κ Vmax : ℝ),
    0 < κ → 0 < L * ∑ t, n t * n' t * B t → 0 ≤ δb → 0 ≤ βb → 0 ≤ δz → 0 ≤ βz →
    ServiceSoundOverlapPerProver Mc S.toService run e sha (forkFinders S e sha 1) (forkFinders S e sha kz)
      δb βb δz βz → KScopeOverlapOneSided Mc S.toService run wRe toF2 0 →
      ChargeVF3j16Overlap Mc S.toService run e κ Vmax →
    ServiceFloorPerProver Mc S.toService
      ((R - 274 * Real.log 2) * (κ * L * ((∑ t, n t * n' t * B t : ℕ) : ℝ)) - Vmax)
      (breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) ((2 : ℝ) ^ 40)⁻¹ δb βb δz βz)

/-- **The honest lemma of `StagedFloorKPerProver` and `StagedFloorKOneSidedPerProver`: their premises hold together.**
At every `sha` and `kz`, on one round of one stage over one cell-block (`n = L = B = n' = 1`, 7 planes at `wRe`), in
the machine of functions at no cost (`funMachine`), some service with its verifier's SHA-512 inputs, coin map and run
map meet `ServiceSoundStagedPerProver` at the fork finders with `δb = βb = δz = βz = 0`, both scopes
(`KScopeAffine`, `KScopeOneSided`) and `ChargeVF3j16` at `κ = 1`, while its verifier accepts one answer to a commitment at every seed
(the honest prover's) and refuses another at some seed. That machine charges nothing, so the noise reads carry the
products and the floor there is negative: this shows the premises consistent at zero break terms, not the floor met. -/
def StagedFloorPerProverHonest : Prop :=
  ∀ (sha : Claims.Msg → Claims.Msg) (kz : ℕ), ∃ (S : HashedService)
    (e : S.Seed ≃ Paths (DStage 1 1 1 7 1 1) 1)
    (run : RunMap funMachine S.toService 1 1 1 1 7 1 1 (ZMod (2 ^ 32))) (c : S.Com) (a a' : S.Ans),
    ServiceSoundStagedPerProver funMachine S.toService run e sha (forkFinders S e sha 1) (forkFinders S e sha kz)
      0 0 0 0 ∧
    KScopeAffine funMachine S.toService run wRe ∧ KScopeOneSided funMachine S.toService run wRe toF2 0 ∧
    ChargeVF3j16 funMachine S.toService run e 1 1 ∧ (∀ σ, S.verify c σ a = true) ∧
    ∃ σ, S.verify c σ a' = false

/-- **The honest lemma of `OverlapFloorKPerProver` and `OverlapFloorKOneSidedPerProver`: their premises hold
together.** As `StagedFloorPerProverHonest`, on one round of one draw over one cell-block: some service, coin map and
run map meet `ServiceSoundOverlapPerProver` at the fork finders with zero break terms, both scopes across draws
(`KScopeOverlapAffine`, `KScopeOverlapOneSided`) and `ChargeVF3j16Overlap` at `κ = 1` in `funMachine`, while the
verifier accepts the honest answer at every seed and refuses another at some seed. -/
def OverlapFloorPerProverHonest : Prop :=
  ∀ (sha : Claims.Msg → Claims.Msg) (kz : ℕ), ∃ (S : HashedService)
    (e : S.Seed ≃ Paths (OSpot 1 (fun _ => 1) 1 7 (fun _ => 1) fun _ => 1) 1)
    (run : ORunMap funMachine S.toService 1 1 (fun _ => 1) 1 7 (fun _ => 1) (fun _ => 1) (ZMod (2 ^ 32)))
    (c : S.Com) (a a' : S.Ans),
    ServiceSoundOverlapPerProver funMachine S.toService run e sha (forkFinders S e sha 1) (forkFinders S e sha kz)
      0 0 0 0 ∧
    KScopeOverlapAffine funMachine S.toService run wRe ∧
    KScopeOverlapOneSided funMachine S.toService run wRe toF2 0 ∧
    ChargeVF3j16Overlap funMachine S.toService run e 1 1 ∧ (∀ σ, S.verify c σ a = true) ∧
    ∃ σ, S.verify c σ a' = false

/-! ### The floors at call scope under J10-R

Each floor of record and its form per prover, restated on J10-R's call design (`Definitions.Pouw.RelativeSecurityJ10R`):
a stage is a call of `n` rows, `n'` columns and `B` segments of `L`, its run one straight-line program over the call's
bits, both sides shared across the call, each segment's coin × coin part one `⟨n, L, n'⟩` product. The premises are the
J10-R mirrors of the floors' own (`ServiceSoundStagedJ10R`, `KScopeOneSidedJ10R`, `ChargeVF3j16J10R`; overlapped,
`ServiceSoundOverlapJ10R`, `KScopeOverlapOneSidedJ10R`, `ChargeVF3j16OverlapJ10R`; per prover, the service premises'
per-prover forms). The floor and the luck are the per-unit floors': `L n n' B` products a call, `M N K`, at the parity
luck 0.854 and `2^{−40}`. The frame is a row's dither bits and a column's coin bits of one segment, so no plane weight
enters. The floors per unit above stay as they are. -/

/-- `StagedFloorKOneSided` at call scope under J10-R: the staged window's service floor for one-sided operands of any
form, each stage a call. In every machine `Mc` and for every service `S` whose seed is `R` rounds of `St` call draws on
J10-R's call design (`JStage`: `n` rows, `n'` columns, `B` segments of `L`, integers mod `2^32` read mod 2 by `toF2`),
under the three premises, every adversary that commits first has its accepted runs charged below
`(R − 274 ln 2) · St · κ L n n' B − Vmax` with probability at most `2^{−40} + R (δb + δz)`: `c = 1` on the call's
one-sided scope (`KScopeOneSidedJ10R`). -/
def StagedFloorKOneSidedJ10R : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St n L B n' : ℕ) (e : S.Seed ≃ Paths (JStage St n L B n') R)
    (run : JRunMap Mc S R St n L B n' (ZMod (2 ^ 32))) (δb δz κ Vmax : ℝ),
    0 < St → 0 < κ → 0 < L * (n * n' * B) → 0 ≤ δb → 0 ≤ δz →
    ServiceSoundStagedJ10R Mc S run e δb δz → KScopeOneSidedJ10R Mc S run → ChargeVF3j16J10R Mc S run e κ Vmax →
    ServiceFloor Mc S ((R - 274 * Real.log 2) * (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) - Vmax)
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz))

/-- `OverlapFloorKOneSided` at call scope under J10-R: the overlapped window's service floor for one-sided operands of
any form per call. In every machine `Mc` and for every service `S` whose seed is `R` rounds of `St` call draws of any
shapes (call `t`: `n t` rows, `n' t` columns, `B t` segments of `L`, integers mod `2^32` read mod 2 by `toF2`)
overlapping at any depth, its check's cell-block drawn uniformly among every call's, under the three premises, every
adversary that commits first has its accepted runs charged below `(R − 274 ln 2) · κ L Σ_t n_t n'_t B_t − Vmax` with
probability at most `2^{−40} + R (δb + δz)`: `c = 1` on the calls' one-sided scope (`KScopeOverlapOneSidedJ10R`). -/
def OverlapFloorKOneSidedJ10R : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St L : ℕ) (n B n' : Fin St → ℕ) (e : S.Seed ≃ Paths (OJSpot St n L B n') R)
    (run : JORunMap Mc S R St n L B n' (ZMod (2 ^ 32))) (δb δz κ Vmax : ℝ),
    0 < κ → 0 < L * ∑ t, n t * n' t * B t → 0 ≤ δb → 0 ≤ δz →
    ServiceSoundOverlapJ10R Mc S run e δb δz → KScopeOverlapOneSidedJ10R Mc S run →
      ChargeVF3j16OverlapJ10R Mc S run e κ Vmax →
    ServiceFloor Mc S ((R - 274 * Real.log 2) * (κ * L * ((∑ t, n t * n' t * B t : ℕ) : ℝ)) - Vmax)
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz))

/-- `StagedFloorKOneSidedPerProver` at call scope under J10-R: `StagedFloorKOneSidedJ10R` on
`ServiceSoundStagedJ10RPerProver` at the fork finders (`1` and `kz` continuations), every adversary `M` that commits
first having its accepted runs charged below `(R − 274 ln 2) · St · κ L n n' B − Vmax` with probability at most
`breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) 2^{−40} δb βb δz βz Ω M`. -/
def StagedFloorKOneSidedJ10RPerProver : Prop :=
  ∀ (Mc : Machine) (S : HashedService) (R St n L B n' : ℕ) (e : S.Seed ≃ Paths (JStage St n L B n') R)
    (run : JRunMap Mc S.toService R St n L B n' (ZMod (2 ^ 32))) (sha : Claims.Msg → Claims.Msg) (kz : ℕ)
    (δb βb δz βz κ Vmax : ℝ),
    0 < St → 0 < κ → 0 < L * (n * n' * B) → 0 ≤ δb → 0 ≤ βb → 0 ≤ δz → 0 ≤ βz →
    ServiceSoundStagedJ10RPerProver Mc S.toService run e sha (forkFinders S e sha 1) (forkFinders S e sha kz)
      δb βb δz βz → KScopeOneSidedJ10R Mc S.toService run →
      ChargeVF3j16J10R Mc S.toService run e κ Vmax →
    ServiceFloorPerProver Mc S.toService ((R - 274 * Real.log 2) * (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) - Vmax)
      (breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) ((2 : ℝ) ^ 40)⁻¹ δb βb δz βz)

/-- `OverlapFloorKOneSidedPerProver` at call scope under J10-R: `OverlapFloorKOneSidedJ10R` on
`ServiceSoundOverlapJ10RPerProver` at the fork finders (`1` and `kz` continuations), every adversary `M` that commits
first having its accepted runs charged below `(R − 274 ln 2) · κ L Σ_t n_t n'_t B_t − Vmax` with probability at most
`breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) 2^{−40} δb βb δz βz Ω M`. -/
def OverlapFloorKOneSidedJ10RPerProver : Prop :=
  ∀ (Mc : Machine) (S : HashedService) (R St L : ℕ) (n B n' : Fin St → ℕ)
    (e : S.Seed ≃ Paths (OJSpot St n L B n') R)
    (run : JORunMap Mc S.toService R St n L B n' (ZMod (2 ^ 32))) (sha : Claims.Msg → Claims.Msg) (kz : ℕ)
    (δb βb δz βz κ Vmax : ℝ),
    0 < κ → 0 < L * ∑ t, n t * n' t * B t → 0 ≤ δb → 0 ≤ βb → 0 ≤ δz → 0 ≤ βz →
    ServiceSoundOverlapJ10RPerProver Mc S.toService run e sha (forkFinders S e sha 1) (forkFinders S e sha kz)
      δb βb δz βz → KScopeOverlapOneSidedJ10R Mc S.toService run →
      ChargeVF3j16OverlapJ10R Mc S.toService run e κ Vmax →
    ServiceFloorPerProver Mc S.toService
      ((R - 274 * Real.log 2) * (κ * L * ((∑ t, n t * n' t * B t : ℕ) : ℝ)) - Vmax)
      (breakPr sha (forkFinders S e sha 1) (forkFinders S e sha kz) ((2 : ℝ) ^ 40)⁻¹ δb βb δz βz)

/-- **The honest lemma of `StagedFloorKOneSidedJ10RPerProver`: its premises hold together.** As
`StagedFloorPerProverHonest`, on one round of one call over one cell-block of J10-R's call design
(`n = L = B = n' = 1`): some service, coin map and run map meet `ServiceSoundStagedJ10RPerProver` at the fork finders
with zero break terms, `KScopeOneSidedJ10R` and `ChargeVF3j16J10R` at `κ = 1` in `funMachine`, while the verifier
accepts the honest answer at every seed and refuses another at some seed. -/
def StagedFloorJ10RPerProverHonest : Prop :=
  ∀ (sha : Claims.Msg → Claims.Msg) (kz : ℕ), ∃ (S : HashedService) (e : S.Seed ≃ Paths (JStage 1 1 1 1 1) 1)
    (run : JRunMap funMachine S.toService 1 1 1 1 1 1 (ZMod (2 ^ 32))) (c : S.Com) (a a' : S.Ans),
    ServiceSoundStagedJ10RPerProver funMachine S.toService run e sha (forkFinders S e sha 1)
      (forkFinders S e sha kz) 0 0 0 0 ∧
    KScopeOneSidedJ10R funMachine S.toService run ∧ ChargeVF3j16J10R funMachine S.toService run e 1 1 ∧
    (∀ σ, S.verify c σ a = true) ∧ ∃ σ, S.verify c σ a' = false

/-- **The honest lemma of `OverlapFloorKOneSidedJ10RPerProver`: its premises hold together.** As
`StagedFloorJ10RPerProverHonest`, on one round of one call draw over one cell-block: some service, coin map and run map
meet `ServiceSoundOverlapJ10RPerProver` at the fork finders with zero break terms, `KScopeOverlapOneSidedJ10R` and
`ChargeVF3j16OverlapJ10R` at `κ = 1` in `funMachine`, while the verifier accepts the honest answer at every seed and
refuses another at some seed. -/
def OverlapFloorJ10RPerProverHonest : Prop :=
  ∀ (sha : Claims.Msg → Claims.Msg) (kz : ℕ), ∃ (S : HashedService)
    (e : S.Seed ≃ Paths (OJSpot 1 (fun _ => 1) 1 (fun _ => 1) fun _ => 1) 1)
    (run : JORunMap funMachine S.toService 1 1 (fun _ => 1) 1 (fun _ => 1) (fun _ => 1) (ZMod (2 ^ 32)))
    (c : S.Com) (a a' : S.Ans),
    ServiceSoundOverlapJ10RPerProver funMachine S.toService run e sha (forkFinders S e sha 1)
      (forkFinders S e sha kz) 0 0 0 0 ∧
    KScopeOverlapOneSidedJ10R funMachine S.toService run ∧
    ChargeVF3j16OverlapJ10R funMachine S.toService run e 1 1 ∧
    (∀ σ, S.verify c σ a = true) ∧ ∃ σ, S.verify c σ a' = false

/-- **The nontrivial lemma of the six floors per prover: their conclusion fails at a verifier that accepts
everything.** For every service with its verifier's SHA-512 inputs whose verifier accepts every commitment, seed and
answer, every coin map onto `R` rounds, every `sha` and `kz`, and every positive floor `F`, the conclusion
`ServiceFloorPerProver` at `breakPr` with the floors' luck `2^{−40}` and zero break terms fails in the machine of
functions at no cost: a constant adversary is accepted at every seed and charged nothing, below `F`, while its error is
`2^{−40}`. Each floor per prover concludes `ServiceFloorPerProver` at that `breakPr`, at its own `F`. -/
def FloorPerProverNontrivial : Prop :=
  ∀ (S : HashedService) [Nonempty S.Com] [Nonempty S.Ans] (Ω : Type) (R : ℕ) (e : S.Seed ≃ Paths Ω R)
    (sha : Claims.Msg → Claims.Msg) (kz : ℕ) (F : ℝ), (∀ c σ a, S.verify c σ a = true) → 0 < F →
    ¬ ServiceFloorPerProver funMachine S.toService F
      (breakPr sha (forkFinders (Mc := funMachine) S e sha 1) (forkFinders S e sha kz) ((2 : ℝ) ^ 40)⁻¹ 0 0 0 0)

end Pouw.Guarantees.RelativeSecurity
