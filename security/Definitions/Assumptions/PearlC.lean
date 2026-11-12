module

public import Definitions.Pouw.Assumptions.PearlC.TTOutW1

/-!
# Pearl-C's TT_OUT on the H100, split into a device model and a conjecture

`TTOutTilePearlCH100Rev1 CM` (`Definitions.Pouw.Assumptions.PearlC.TTOutRev1`) is TT_OUT per audit tile over an
abstract cost model: which `CM` it holds at is prose, so it mixes a claim about the card with a claim about the
programs that run on it. By ruling (8 Oct) a mixed statement is split, and each part is named:

* **The device model, `w1-complete-h100`** (`W1CompleteH100`): the card's cost model `CM` reaches the oracle only
  through counted queries, and every program it runs is simulated by a W1 program on `W1Device.s8H100Best` (the H100's
  and the H200's tensor-core and FP32 rates, INT8 included, with a compression and a first fetch free) with the same
  transcript at no more cost (`CostModel.SimulatedBy`). It is the data sheets' reading: NVIDIA's H100 Tensor Core GPU
  Architecture whitepaper V1.04, Table 1, and the H200 Tensor Core GPU data sheet (`W1.W1Prices.h100`). It is
  falsified by a measurement: a program on the card that produces a transcript for less than every W1 program that
  produces it.
* **The conjecture, `tt-out-tile-pearl-c-h100-rev1`** (`TTOutTilePearlCH100Rev1S8`): TT_OUT per audit tile, rev1, at the
  H100 record, over W1 on that setting. It is about W1 programs only, and it is falsified by one: a W1 program that
  outputs a set of good tiles' words for less than `(1 − g₀)` of their credit. The standing falsification test is the
  census's known skip classes replayed against rev1's credit
  (`benchmarks/pouw/tests/test_tt_out_tile_h100_rev1.py`).

Together they give the abstract form at `CM` (`Pouw.PearlC.ttOutTilePearlCH100Rev1_of_s8`, in
`Proofs.Pouw.PearlC.S8H100Rev1Gap`).
W1 makes no oracle call, so a `CM` whose runs read `H` is not simulated: the device model speaks at the deployment's
noise, SHA-512 under the live salt (`NoiseSha512.sha512Sem`), which a W1 program computes with `sha` instructions.

It is a module of its own under `Definitions.Assumptions` that reads only PoUW's definitions, so the PoUW statements
that take these import it without the rest of the named assumptions. It also holds the two definitions they read
beside W1's and the game's: the simulation `CostModel.SimulatedBy` and the setting `W1.W1Device.s8H100Best`.
-/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **`CM'` simulates `CM`**: every program of `CM` has a program of `CM'`, fixed before the oracle and the salt, with the
same run at no more cost and no more oracle calls on every oracle, workload, activation input, state and salt. An
adversary of `CM` is then one of `CM'` within the same bounds, so a bound over every adversary of `CM'` holds over
`CM`'s (`Pouw.PearlC.ttOutTile_of_simulatedBy`). -/
def CostModel.SimulatedBy (CM CM' : CostModel Q R S) : Prop :=
  ∀ p : CM.Prog, ∃ p' : CM'.Prog, ∀ (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S),
    CM'.run p' H U act σ s = CM.run p H U act σ s ∧ CM'.cost p' H U act σ s ≤ CM.cost p H U act σ s ∧
      CM'.calls p' H U act σ s ≤ CM.calls p H U act σ s

/-- **The Hopper INT8 setting at the adversary's best case**: `W1Device.s8H100` with a SHA-512 compression and a first
fetch free. Neither price is on a data sheet (the H100 whitepaper's Table 1 and the H200 data sheet give tensor-core and
FP32 rates, not a compression's or a word's first read), and the audited hardware and its vendor are untrusted, so
neither is taken at a measured rate. W1's cost only grows with each: at any `sha, fetch ≥ 0`, `s8H100 sha fetch` is
simulated by this setting (`Pouw.PearlC.W1.w1Priced_s8H100_simulatedBy_best`), so a bound here holds at every price. -/
def W1.W1Device.s8H100Best : W1.W1Device := W1.W1Device.s8H100 0 0

end Pouw.PearlC

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- `w1-complete-h100` (measured; falsified by a program on an H100 or an H200 that produces a transcript for less than
every W1 program on `W1Device.s8H100Best` that produces it). **W1 is complete for the H100 and the H200**: the card's
cost model `CM` reaches `H` only through counted queries, and W1 on `W1Device.s8H100Best` with the salt's bytes
`salt` simulates it (`CostModel.SimulatedBy`). Its prices are the data sheets': NVIDIA's H100 Tensor Core GPU
Architecture whitepaper V1.04, Table 1, and the H200 Tensor Core GPU data sheet (`W1.W1Prices.h100`), with a
compression and a first fetch at 0. -/
def W1CompleteH100 (salt : S → List ℕ) (CM : CostModel Q R S) : Prop :=
  CM.Queries ∧ CM.SimulatedBy (W1.w1Priced W1.W1Device.s8H100Best salt)

/-- `tt-out-tile-pearl-c-h100-rev1` (conjectured; refuted by a W1 program on `W1Device.s8H100Best` that outputs a set of
good tiles' words for less than `(1 − g₀)` of their credit). **TT_OUT(1/400) rev1 per audit tile at the H100 record,
on the H100 with INT8**: `TTOutTilePearlCDevRev1W1` on `W1Device.s8H100Best`, the adversary's best case for the
prices no data sheet gives. -/
def TTOutTilePearlCH100Rev1S8 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTilePearlCDevRev1W1 W1.W1Device.s8H100Best salt devH100 sem ρ

end Pouw.PearlC.Assumptions
