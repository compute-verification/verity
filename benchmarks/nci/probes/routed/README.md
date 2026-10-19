# Routed-mixture probes (experiments A and B1)

These probes measure g, the routing gain in F3's units: the dense size that reaches a configuration's bits per byte with
unlimited data, divided by the weights one sequence's computation uses (P). The fit bound then applies with
N*/g, and `analysis.py` turns g into D through `benchmarks/nci/fit.py`. The plan, the costs and the decision rule are on the
Notion page [Settling F3: routed-mixture experiments](https://app.notion.com/p/3fe399515d9e81aa98eefdfd7774e90d).

**Not launched.** The probes need a budget line: about 20 GPU-hours central and 73 worst on one 8-GPU RTX PRO 6000 node,
USD 36–132, plus about 1 TB of downloads.

| Path | Holds |
| --- | --- |
| `probe.py` | the GPU half: `windows`, `score` (bits per window; Qwen3-MoE routing records and confinement), `codebook`, `analyze`, `selftest` |
| `analysis.py` | the standard-library half: windows, oracle and prefix composition, the dense-ladder fit, g with bootstrap intervals, D at g, the verdict |
| `manifest.json` | every model `run.sh` scores (repo, revision, role) and the studies `analyze` fits |
| `run.sh` | the whole job on one node: one model per GPU lane, then `analyze` |

## The configurations

| Study | Configuration | P | What it is |
| --- | --- | --- | --- |
| A | `flex-oracle` | 11.63B | each window goes to whichever of FlexOlmo's seven released models (`Flex-public-7B-1T` and the six `Flex-*-2x7B-1T`) scores its span best, plus log₂7 bits. **Decides A.** |
| A | `flex-prefix` | 11.63B | the same models, chosen on the window's first 1 KiB. Robustness only: the choice runs every member |
| A | `control-oracle` | 7.30B | the oracle over seven nearly identical late OLMo 2 7B checkpoints: what choosing alone buys |
| A | `flexolmo`, `flex-public` | 33.3B, 7.30B | the token-level merge and the trunk alone, for scale |
| A-annealed | `flex-oracle` | 11.63B | A refitted on the four final (annealed) OLMo 2 checkpoints alone, with no data term |
| B1 | `moe-codebook-K` | 1.541B + 0.2265B·K | Qwen3-30B-A3B-Base with every token of a window confined to K experts per layer, the subset of the codebook path nearest the window's unconfined routing profile, plus log₂16 = 4 bits. **Decides B1.** |
| B1 | `moe-prefix-K` | the same | confined to the K experts with the most routing mass over the first 1 KiB. Robustness only: the ranking runs the whole MoE |
| B1 | `moe-full` | 30.53B | the unconfined MoE |

A's dense ladder is OLMo 2 1B, 7B, 13B and 32B, final checkpoints plus stage-1 checkpoints from 0.5T to 4T tokens. B1's
is the Qwen3 0.6B–14B bases, all trained on the MoE's 36T tokens, so it has no data term. Every configuration is scored
on the same windows: bytes 1,024–8,191 of each 8 KiB window, with the first 1 KiB as context.

The verdict, on 95% intervals of g taken across the Besiroglu, Hoffmann and Kaplan-like exponents:

- If A's lower end is at or above ×2.15 (the gain where D at a 6 GB wipe and 2.5 bits falls below 100), **drop** the
 claim.
- Otherwise, if B1's best codebook K's lower end is at or above ×2.15, **hedge**.
- If both upper ends are at most ×1.2, **keep ×2 as the margin**.
- Otherwise **keep ×2 and run the trend run**.
- With no valid interval for A or B1, the verdict is **incomplete**.

## Running it

1. Build the evaluation corpus on CPU (the page's section 4), publish it with `research data put --kind corpus --tree DIR
 --preserve`, and record its `art:` id on the page before scoring anything.
2. Cut windows from the corpus (a JSONL of `id`, `stratum` and `text`), and cut calibration windows from a disjoint split
 of about 500 windows:

 ~~~bash
 python benchmarks/nci/probes/routed/probe.py windows corpus.jsonl --out windows.jsonl
 python benchmarks/nci/probes/routed/probe.py windows calibration.jsonl --out calibration-windows.jsonl
 ~~~

3. Run the job on a node under a budget line:

 ~~~bash
 research run --on NODE --cwd source -- env WINDOWS=/abs/windows.jsonl CALIBRATION=/abs/calibration-windows.jsonl \
 HF_HOME=/abs/hf-cache bash benchmarks/nci/probes/routed/run.sh
 ~~~

`run.sh` runs in three phases, then analyzes:

1. Every plain model scores `$WINDOWS`, and the MoE scores `$CALIBRATION` with its routing profiles recorded.
2. `codebook` clusters those profiles into 16 paths. The unconfined MoE then scores `$WINDOWS`, recording each window's
 prefix ranking and codebook path.
3. The seven confined configurations score `$WINDOWS`.

A score file is written under a `.part` name and renamed when complete, and a rerun skips every complete file. A failed
job is listed in `$OUT/failed.txt` with its log in `$OUT/logs/`, and stops the run at the end of its phase.

What the run leaves in `$OUT`:

- `scores/NAME.jsonl`, one record per window (`prefix_bits`, `span_bits`, `span_bytes`, and for the MoE `experts_used` and
 `mass_in_top`), with `NAME.jsonl.meta.json` recording the versions and any window longer than the model's context.
- `routing/` holds `calibration.pt`, `codebook.pt` and `windows.pt`, a few tens of MB.
- `analysis.json` holds every configuration's bits per byte overall and by stratum, the ladder fits, g and G_same per law
 form with intervals, D at g on four rungs, and the verdict. `verdict.txt` is the printed summary.

Environment variables:

- `PY` is the Python command. The default is `uv run` with torch 2.14.1, transformers 5.20.1 and accelerate 1.15.0
 pinned. torch 2.14.1 from PyPI is a CUDA 13.0 build, so the node's driver must support CUDA 13; otherwise point `PY`
 at another build.
- `GPUS` (default 8), `DEVICE` (default `cuda`) and `DTYPE` (default `bfloat16`) set the lanes, the device and the
 precision.
- `MANIFEST` (default `manifest.json`) names the models and studies.

Before downloading anything, run `probe.py selftest`. On CPU, in about 10 s, it checks the confinement on a tiny random
Qwen3-MoE: a free confinement reproduces the stock forward, and confined tokens stay in their subsets. The analysis half is
tested in `tests/test_routed_probe.py`.

## Limits

- Confinement and routing records support Qwen3-MoE routers under transformers 5.x only (`probe.routers`). B2
 (DeepSeek-V3-Base against Llama-3.1-405B-Base) isn't covered.
- OLMo 2's stage-1 checkpoints are taken before the learning-rate anneal, so they sit above the annealed curve. The A
 ladder fits them a constant offset (`fit_ladder`), and A-annealed reports the fit without them.
- Qwen3's dense ladder stops at 14.8B, so any dense equivalent above that, `moe-codebook-64` among them, is extrapolated.
- Free exponents need at least three sizes and three token counts; through two, every exponent fits exactly. A study
 that lacks them records `free_error`. A ladder whose bits per byte don't fall with size records `ladder_error` and
 gives no interval.
- Every released model is one seed. The intervals are a bootstrap over windows, paired across models, together with the
 spread over law forms.
