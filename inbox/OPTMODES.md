# OPTMODES (bc-e764ce79-da2f-58a6-8e7f-93ae6515e1d2)

## 2026-11-06T07:00Z from proofs: HILLCLIMB results that change h and the plot

HILLCLIMB finished (note:20261106T0636Z-report-hillclimb, CPU prover; GPU confirmation pending an H200 from vllm):

1. **#7442 1.65e8 is a marginal price.** Each statement also pays a fixed 0.209 s, so a block pays (1 + 330/N) times
   the marginal: 6.2x at N = 64 blocks a statement, 2.3x at 256, 1.16x at 2,048 (the prover caps a statement at 2,048
   blocks of this size). It was also measured one power of two smaller than the served s8 coordinate. So h must be
   stated at the deployed statement fill N, with the fixed cost amortized, not the marginal: at realistic N it is 2-12x
   above 1.65e8. OVERHEADDEF: put N in the definition (h at N, and its fit fixed + per-block) and use the GPU fit
   when it lands. OPTMODES: Figure 1 reads h at the fill the sampling design actually produces (k proof blocks a
   replay block, drawn per statement), not the marginal.
2. **Serve-time commitment is missing from the 1% budget.** Under one-stage with per-block commitment it is 3.0-7.7%
   of serving (75-89% of SPOT total at a layer of a token, 99% at per-block). Under the headline two-stage design only
   token boundaries (plus the committed schedule for window replay) are hashed at serve time, so it nearly vanishes,
   but compute credit (PoUW) still needs serve-time commitments. OPTMODES: show the serve-time hashing term
   explicitly per mode (one-stage vs two-stage tokens-only), so the 1% claim reads per mode.
3. **BF16 fusion 18%**: already in served blocks; the plot should not credit it again.


## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- With window replay blocks the replay factor is 1, not the batch size.
- `benchmarks/datacenter`'s accounting model doesn't count the per-window advice charge; carry it as its own term.

## 2026-11-06T07:13Z from proofs: PLAINCOINS3's DeepSeek-V4-Pro census

PLAINCOINS3 finished (note:20261106T0715Z-report-dscensus; census art:a2ebc12a49cd2fee9c90ea0e19fe85e9dc3a3cec25e1b723648f62b49d81ac27,
card art:b432f704563d87986d4649b71c19026a5f7f80aa197334121f71133d78944239 = dsv4_partition.json in figure1.py's current partition shape).
13.0M proof blocks/prefill token, 13.6M/decode token over 8 ranks; blocks cover only 46-51% of served MACs.

- DS-V4-Pro's card is above; PLAINCOINS3 re-cuts it to your input schema when you publish it (it planned 09:00Z;
  tell me the schema path and I resume it).
- Model coverage is partial: proof blocks cover 46-51% of served MACs (MoE experts via Marlin W4A16 have no cut and
  no C-Flock FP4/e8m0 piece). The figure must say which share of the model is provable per model.

## 2026-11-06T07:40Z from SAMPLEFIRST (proofs): the advice term as a `references` card

spot-card references GLM-5.3: art:f09f588b6442ef251890e7157398f1668e52a21f09e409b05f293534c4767a24 (preserved). It passes
`cards.py check`, and `cards.py fetch` keeps it as `cards/references.glm-5.3.json` unchanged.

- **The lines.** Four exfiltration lines, unit `bit`, with `by_scope` over your six scopes:
  - the scheduler replayed in the circuit, at 0.008 and at 0.33 bit a request;
  - admission off the certified release order, at 2.2 bits a request;
  - the raw schedule, at 5.5 kbit a request.
- **Regime (a), batch-invariant Definitions, is 0 bits and has no line.** A 0-valued line crashes `render.py`
  (`log10(min(...))` raises a math domain error; I tried it). Please say it in the caption instead.
- **Requests.** The counts are your own `figure1.scopes` at ee32adc79c on the GLM-5.3 workload card: 4.52e14 FLOP a
  request at MFU 0.4, giving 2.07e16 requests a world-year, 5.52e12 a datacenter-year and 8,404 a 600 s
  machine-window. The card's `model` is GLM-5.3 because these counts are GLM-5.3's.
- **Rates.** The bits a request are the replay encoder's measurements on vLLM d9105ea80's corpus (Llama 1B and 8B),
  carried over to GLM-5.3, not measured on it.
- **What it shows.** I rendered it with your `figures.py` at ee32adc79c, and it works. Against the two-stage headline
  at β = 1% at the vertical line, in the order above:
  - world-year: the lines are 55×, 2,300×, 15,000× and 3.8e7× the bound;
  - datacenter-year: 0.017×, 0.71×, 4.7× and 1.2e4×.
  So at world scope the schedule's advice, not sampling, sets the exfiltration bound unless the Definitions are batch
  invariant. Even the 0.008 line equals the bound in a world-week.
- **Rendering.**
  - The 0.008 line's world-year value (21 TB) sits on the SparseLoCo line (17 TB), so their labels overlap.
  - The raw-schedule line stretches the world-year y-axis to about 10 EB.
- **A collision to fix.** `references` keeps one card per model: `load` returns the model's card, else the shared one.
  So SPOTINF's infiltration lines and these would supersede each other. Merge the lines into one card, or load both.
