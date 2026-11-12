module

public import Definitions.Pouw.Assumptions.PearlC.TTOutW1

/-!
# TT_OUT-FP4 at W1 (trusted; named assumptions, `Prop`s taken as hypotheses, never `axiom`s)

`TTOutFp4` and `TTOutTileFp4` (`Definitions.Pouw.Assumptions.PearlC.TTOutFp4`) are stated over any `CostModel`, and which one
they are believed at (W1 at sm_120's prices, in FP4 units) is prose in `TTOutFp4Sm120`'s docstring. Here that cost model
is a term, as `TTOutW1` makes it for the FP8 records: TT_OUT's game at `CM := W1.w1Priced dv salt`. Each form is the old
one's body at that cost model (same protocol, domain, `g₀ = 1/400` and `δPearlC`); `TTOutFp4W1 … ↔ TTOutFp4
(W1.w1Priced dv salt) …` (`Proofs.Pouw.PearlC.TTOutFp4W1`). What W1 charges, why `w1Priced`, and the honest program's
`AdmitsRef` are `TTOutW1`'s.

**The device.** `devFp4Sm120` is set on `W1Device.fp4Sm120 sha fetch`: sm_120 at the loop's prices (`sm120Loop`) in FP4
units, an FP8 unit worth 2, with `sha` and `fetch` given in FP4 units. Its table gives the record's prices:
`W1Prices.toFp4Prices` at the `lut256` scale path is `Fp4Prices.sm120` (`W1Checks.toFp4Prices_fp4Sm120`), as
`W1Checks.toPrices_*` pairs the FP8 records. `sha` and `fetch` are unmeasured parameters, and the assumption only gets
weaker as they grow.

**What is credited** is `creditFp4` less the replayed debit, unchanged. W1 charges, and nothing credits: the weight
side (B̃'s forming, B̃·F_A, P_B), which D-SK keys by the salt; `qa·m·k` (in `wrefFp4` only); SHA-512 work; and every
fetch.

**The noise.** W1 makes no oracle call. D-SK (`Fp4Sem.SaltKeyedB`, in `pearlCDomainFp4`) states `B̃`'s lines as the
ones SHA-512 gives from `seed_B` of the salt's bytes (`NoiseSha512Fp4`), which a W1 program computes with `sha`
instructions. These assumptions speak at `sha512Fp4Sem base salt rootA rootB`, the deployment's noise
(`pearl_c4.py`'s lines op for op), which meets D-SK.

**One salt.** As for the FP8 forms (`TTOutW1`), each form here takes the cost model's `salt` and the semantics
separately. It speaks at the deployment when `sem := sha512Fp4Sem base salt rootA rootB` with the `salt` its cost model
reads.

**The program's text is bounded** (`W1.Prog.bounded`), so a W1 program can't hold the reference at `sha512Fp4Sem` as a
table of one leaf per salt in its code (`Proofs.Pouw.PearlC.W1Tabulate.tree_not_code`, which is about W1 programs alone,
not a protocol's noise).
-/

@[expose] public section

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **TT_OUT(1/400) at an FP4 record `d`, at W1 on `dv`**: `TTOutFp4` with its cost model the term
`W1.w1Priced dv salt`. -/
def TTOutFp4W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutW1 dv salt (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4 sem) (1 / 400) δPearlC

/-- **TT_OUT(1/400) per audit tile at an FP4 record `d`, at W1 on `dv`**: `TTOutTileFp4` at `W1.w1Priced dv salt`. -/
def TTOutTileFp4W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileW1 dv salt (pearlCProtocolFp4 d sem ρ) (pearlCTilesFp4 d sem ρ) (pearlCDomainFp4 sem) (1 / 400) δPearlC

/-- **`tt-out/fp4-sm120` at W1**: TT_OUT(1/400) for Pearl-C4 at `devFp4Sm120` on `W1Device.fp4Sm120 sha fetch`. -/
def TTOutFp4Sm120W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutFp4W1 (W1.W1Device.fp4Sm120 sha fetch) salt devFp4Sm120 sem ρ

/-- **`tt-out/fp4-sm120` per audit tile at W1**: `TTOutTileFp4W1` at `devFp4Sm120` on `W1Device.fp4Sm120 sha fetch`. -/
def TTOutTileFp4Sm120W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileFp4W1 (W1.W1Device.fp4Sm120 sha fetch) salt devFp4Sm120 sem ρ

end Pouw.PearlC.Assumptions
