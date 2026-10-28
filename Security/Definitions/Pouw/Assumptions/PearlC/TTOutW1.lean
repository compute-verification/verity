import Definitions.Pouw.PearlC.Instance
import Definitions.Pouw.PearlC.Tile
import Definitions.Pouw.PearlC.Device
import Definitions.Pouw.PearlC.DeviceRev1
import Definitions.Pouw.PearlC.W1

/-!
# TT_OUT at W1 (trusted; named assumptions, `Prop`s taken as hypotheses, never `axiom`s)

`TTOut` (`Definitions.Pouw.Assumptions.PearlC.TTOut`) is stated over any `CostModel`, and which one it is believed at (W1 at
the device's measured prices) is prose in its docstring. Here that cost model is a term: TT_OUT's game with
`CM := W1.w1Priced dv salt`, the W1 machine on the device `dv` with the salt's bytes `salt`, priced with no refund
(by ruling). A strategy's cost is W1's cost, so an attack on these assumptions is a W1 program.

**What W1 charges** (`W1.w1Priced`): every instruction a program executes at its class price (a tensor-core instruction
by its whole fragment, a SHA-512 compression at `sha`), the taint rule's INT32 per word, and `fetch` per distinct
pre-salt or supplied word it reads. A move is free. It makes no oracle call, so `q` bounds the pre-salt phase's calls,
and `CM.Queries` holds (`Proofs.Pouw.PearlC.TTOutW1.w1Priced_queries`); `TTOutW1 … ↔ TTOut (W1.w1Priced dv salt) …`
(`ttOutW1_iff`).

**What is credited** is the protocol's credit, unchanged: `creditDev` (rev1: without the first promotion of each word)
minus the replayed debit. The weight side (B̃'s forming, B̃·F_A, P_B) depends on the salt, so a program runs it after
the salt; W1 charges it and the credit counts none of it, nor `qa·m·k` (in `W_ref` only), the SHA-512 work or any fetch.
The bound on guessing whole output words rests on exactly that.

**Why `w1Priced`.** It charges at least what `w1` (with the operand refund) charges, so TT_OUT at `w1` implies it here:
it is the weaker assumption. It is the cost model the compute audit reads (`GγPerCoin (w1Priced dv salt)`), and it does
not depend on the protocol, so a reader whose conclusion's protocol differs from its hypothesis' (the kernel `W_ref`
variants) uses one cost model on both sides.

**The device.** `dv` is the W1 setting of the record `d`: `W1Device.sm120` for `devSm120v1/v2 Prices.sm120`,
`W1Device.sm120Loop` for `… Prices.sm120Loop`, and `W1Device.h100` for `devH100` (`W1Checks.toPrices_*`: the coarse
prices agree). `sha` and `fetch` are parameters; each assumption is believed at the measured ones, and it only gets
weaker as they grow.

**The noise.** W1 runs no oracle, so at a `sem` whose noise reads `H` a W1 program cannot evaluate the noise and these
assumptions say nothing about the deployment. They speak at `NoiseSha512.sha512Sem base salt rootB` with the same `salt`,
the deployment's noise, which a W1 program computes with `sha` instructions.

**One salt.** Each form here, and each guarantee stated at `W1.w1Priced dv salt` and a semantics `sem`, takes the two
separately. It speaks at the deployment when `sem := NoiseSha512.sha512Sem base salt rootB` with the `salt` its cost
model reads: one salt, which the program reads as its salt bytes (`W1.init`'s region 1) and the protocol reads as the
noise's. At any other pair it is still proved, but about a game the deployment doesn't play.

**The program's text is bounded** (`W1.Listing.bounded`: at every layout `2^64` code nodes and `2^64` constants).
W1 charges only the path a run takes, so a strategy with unbounded text could hold the reference at
`sha512Sem` as a table of one leaf per salt in its code. A W1 program can't
(`Proofs.Pouw.PearlC.W1Tabulate.tree_not_code`).

**The honest program** pays the weight side and the weight read, neither in `W_ref`, so `AdmitsRef (W1.w1Priced …)`
fails at decode. No reader of these assumptions takes `AdmitsRef`: it is completeness, and γ bounds credited work.

The rest (the credit, debit and cap, the second hole carried, the falsifiers) is `TTOut`'s and `TTOutPearlCDevRev1`'s.
-/

namespace Pouw.PearlC.Assumptions

open Finset

variable {Q R S : Type}

/-- **TT_OUT(γ₀) at W1**: `TTOut`'s game at `CM := W1.w1Priced dv salt`. With `K` the units whose ticket and useful
words are correct and whose debit is within the cap, no W1 program bounded by `T` makes
`∑_{u ∈ K} credit(u) > T / (1 − γ₀)` with probability over `(H, s)` more than `ε(q, N)`. -/
def TTOutW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (P : Protocol Q R S) (D : UnitLayout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary (W1.w1Priced dv salt : CostModel Q R S), A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - γ₀) < ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.transcript U x.1 x.2) u))
      ≤ ε q U.layout.N

/-- **TT_OUT(γ₀) per tile at W1**: `TTOutTile`'s game at `CM := W1.w1Priced dv salt`, the good tiles of `TR` in place of
the correct units. -/
def TTOutTileW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : UnitLayout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary (W1.w1Priced dv salt : CostModel Q R S), A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - γ₀) < ∑ g ∈ goodTiles P TR U x.1 x.2 (A.transcript U x.1 x.2),
        TR.credit x.1 x.2 U g (actOf (A.transcript U x.1 x.2) ((TR.tiling U.layout).unit g)))
      ≤ ε q U.layout.N

/-- **TT_OUT(1/400) for Pearl-C at a device record `d`, at W1 on `dv`**: `TTOutPearlCDev` with its cost model the term
`W1.w1Priced dv salt`. -/
def TTOutPearlCDevW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutW1 dv salt (pearlCProtocolDev d sem ρ) (pearlCDomainDev d) (1 / 400) εPearlC

/-- **TT_OUT(1/400) per audit tile at a device record `d`, at W1 on `dv`**: `TTOutTilePearlCDev` at
`W1.w1Priced dv salt`. -/
def TTOutTilePearlCDevW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileW1 dv salt (pearlCProtocolDev d sem ρ) (pearlCTilesDev d sem ρ) (pearlCDomainDev d) (1 / 400) εPearlC

/-- **TT_OUT(1/400) rev1 at a device record `d`, at W1 on `dv`**: `TTOutPearlCDevRev1` (every promotion into a +0 total
out of the credit) at `W1.w1Priced dv salt`. -/
def TTOutPearlCDevRev1W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutW1 dv salt (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400) εPearlC

/-- **TT_OUT(1/400) rev1 per audit tile at a device record `d`, at W1 on `dv`**: `TTOutTilePearlCDevRev1` at
`W1.w1Priced dv salt`. -/
def TTOutTilePearlCDevRev1W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileW1 dv salt (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400)
    εPearlC

/-- **`tt-out/pearl-c-h100-rev1` at W1**: rev1 at the H100 record on `W1Device.h100 sha fetch`. -/
def TTOutPearlCH100Rev1W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutPearlCDevRev1W1 (W1.W1Device.h100 sha fetch) salt devH100 sem ρ

/-- **`tt-out-tile/pearl-c-h100-rev1` at W1**: rev1 per audit tile at the H100 record on `W1Device.h100 sha fetch`. -/
def TTOutTilePearlCH100Rev1W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ)
    (salt : S → List ℕ) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTilePearlCDevRev1W1 (W1.W1Device.h100 sha fetch) salt devH100 sem ρ

end Pouw.PearlC.Assumptions
