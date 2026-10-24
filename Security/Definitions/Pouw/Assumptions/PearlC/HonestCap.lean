import Definitions.Pouw.PearlC.Tile

/-!
# The honest tiles' measured cap rate, completeness's one assumption (trusted; a `Prop` taken as a hypothesis)

The sampled audit's completeness (`CompleteSampled`, `Definitions.Pouw.PearlC.Completeness`) takes the workload's weights
passing the rules as a hypothesis (`Workload.WeightsOK`), and needs two facts about the honest prover's drawn tile: its
words are the reference's, and its replayed debit is within the cap. The first is
proved (`HonestWords`: every reference value is an FP32 word's). The second depends on the activations and the noise,
and nothing proves it; `HonestTileCapRate` assumes it at the rate a measurement gives.

It is **empirical**, not a hardness assumption: `ε` is an upper confidence limit on a measured rate of honest tiles that
fail the per-tile cap. It is a parameter, so the theorem that reads it (`completeSampledDevRev1K`) holds unchanged
whatever the table's number. How the rate is measured and cited (lean's rulings on #923):

- **Per input.** An input is one `(U, act)`: one linear's GEMM in one pass. Its rate is `p = E[X]`, `X` its
  work-weighted failing fraction under the tiles' noise, which is the probability the assumption bounds. Each measured
  input is bounded on its own, and inputs are never pooled.
- **An upper confidence limit, simultaneous over the `m` measured inputs.** Each input is bounded at `δ = 0.05/m`
  (Bonferroni, 95% overall), and the cited `ε` is the maximum of the `m` limits, rounded up. `m` counts every input
  chosen in every pass the table measures, decode's and prefill's alike, including any not yet measured when a bound is
  first computed (Qwen3-8B's untimed pass: 8 decode and 8 prefill inputs, `m = 16`). With `a = w_max/W`, its
  heaviest tile's share of its work:
  - every tile measured (a census), no failure: `ln(1/δ)·a = ln(20m)·a`, the rule of three for a work-weighted draw;
  - `n` work-weighted draws, no failure: `ln(1/δ)·a/(1 − e^(−n·a))`. Clopper–Pearson on the draws is not used: it bounds
    the realized pass's fraction `X`, not the rate `p`;
  - a census with failures: Bernstein's bound for a weighted sum of independent Bernoullis;
  - draws with failures: no bound. Such an input is measured in full;
  - N fresh trials, each with its own salt and work-weighted draw and with `act` fixed, and k failures: the
    Clopper–Pearson upper limit at δ = 0.05/m; at k = 0, 1 − δ^(1/N).

  The census and draw bounds hold only if **the input's tiles fail independently**, which they do when each tile's
  noise is the oracle at points of its own. The fresh trials are independent because their salts and draws are:
  each is derived from recorded OS randomness drawn after the pass's activations are fixed, under a domain naming the
  table and the input, with `N`, `m`, the inputs and the source fixed before the run, and the cited limit uses every
  trial run for the input. With the deployed hash fixed, the trials bound the rate at that hash, while the assumption
  averages over oracles: the same random-oracle step as the rest of PoUW. Unlike draws with failures, they bound an
  input with failures too (Clopper–Pearson at k).
- **`Served` carries the meaning.** The assumption is that *every* member of `Served` is within `ε`, not only the
  measured sample, and that each member's activations are fixed independently of the oracle and the salt. `Served` is
  the population the measured requests were drawn from: their prompt source and how it was sampled, on the model's
  pinned weights, both named with the table. Applying the bound to requests nobody measured is a further empirical
  step, not a confidence statement. A population can meet it and say nothing: on activations whose every row fails Π's
  rules (all NaN, say) no row is credited, every tile's debit and cap are both 0, and the completeness that follows is
  of an audit that credits nothing.
- **The counted event** is `pearlCTilesDevRev1K`'s `capOK`, through the Python check that #865's L5 test ties to it.
- **The measurement** is `benchmarks/pouw/pearl_c_vllm/served_debit.py --per-input` (at `f96f5b780`; the fresh trials
  are its `--fresh-draws N --fresh-source HEX`). Its `inputs.json`
  holds each input's failures, `W`, `w_max` and upper limit, and their maximum `cited_upper`; `ε` is `cited_upper`
  rounded up (Llama-3.1-8B and Qwen3-8B, `art:<pending: FP8 served table>`).

Until that art id lands with `Served`, `m` and the per-input bounds, no table or ledger cites a completeness number.
-/

namespace Pouw.PearlC.Assumptions

open Pouw.PearlC

variable {Q R S : Type}

/-- **An honest drawn tile exceeds the cap at most `ε` of the time** (empirical: an upper confidence limit on a measured
rate of honest tiles that fail the check, not a hardness assumption). For every in-domain workload `U` whose weights
pass the rules, every activation input `act` in the population `Served`, and every ticket count `w` proportional to the
tiles' `W_ref`: over the oracle, the salt and one work-weighted draw, the probability that the drawn tile is of a unit
of the layout and its replayed debit on the honest activations exceeds the cap (`TR.capOK` fails) is at most `ε`.

The bound is per input, an input being one `(U, act)`: one linear's GEMM in one pass. `ε` is the cited maximum over the
`m` measured inputs of each input's upper confidence limit at `δ = 0.05/m` (95% simultaneous), rounded up: at no failure
in a census of the input's tiles, `ln(20m)·w_max/W`; at no failure in `n` work-weighted draws,
`ln(20m)·a/(1 − e^(−n·a))` with `a = w_max/W`. Every such limit assumes the input's tiles fail independently (each
tile's noise is the oracle at points of its own). `Served` carries the meaning: every member of it is assumed within
`ε`, not only the measured ones (a further empirical step, not a confidence statement), with activations fixed
independently of the oracle and the salt. It is the population the measured requests were drawn from (their prompt
source and how it was sampled) on the model's pinned weights. The counted event is `pearlCTilesDevRev1K`'s `capOK`,
through the Python check that #865's L5 test ties to it. The measurement is `served_debit.py --per-input`
(`inputs.json`'s `cited_upper`), `art:<pending: FP8 served table>`; the module docstring gives the bounds for an input
with failures and for fresh trials. -/
def HonestTileCapRate [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) (Served : Workload → (ℕ → Codes) → Prop) (ε : ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → U.WeightsOK P → ∀ act : ℕ → Codes, Served U act →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
      pr (fun x : (Q → R) × S × Ticket w (TR.tiling U.layout).NT =>
          (TR.tiling U.layout).unit x.2.2.1 < U.layout.N ∧
            ¬ TR.capOK x.1 x.2.1 U x.2.2.1 (act ((TR.tiling U.layout).unit x.2.2.1)))
        ≤ ε

end Pouw.PearlC.Assumptions
