module

public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1
public import Definitions.Pouw.PearlC.W1Fp4Sm120

/-!
# TT_OUT-FP4 at W1 on sm_120 (trusted; named assumptions, `Prop`s taken as hypotheses, never `axiom`s)

`tt-out/fp4-sm120` at W1, per unit (`TTOutFp4Sm120W1`) and per audit tile (`TTOutTileFp4Sm120W1`): `TTOutFp4W1` and
`TTOutTileFp4W1` at `devFp4Sm120` on `W1Device.fp4Sm120 sha fetch`, the premises of Pearl-C4's listed statements
(`Definitions.Pouw.Guarantees.PearlC4`).
-/

@[expose] public section

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **`tt-out/fp4-sm120` at W1**: TT_OUT(1/400) for Pearl-C4 at `devFp4Sm120` on `W1Device.fp4Sm120 sha fetch`. -/
def TTOutFp4Sm120W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutFp4W1 (W1.W1Device.fp4Sm120 sha fetch) salt devFp4Sm120 sem ρ

/-- **`tt-out/fp4-sm120` per audit tile at W1**: `TTOutTileFp4W1` at `devFp4Sm120` on `W1Device.fp4Sm120 sha fetch`. -/
def TTOutTileFp4Sm120W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileFp4W1 (W1.W1Device.fp4Sm120 sha fetch) salt devFp4Sm120 sem ρ

end Pouw.PearlC.Assumptions
