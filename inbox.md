[2026-11-13T07:04Z] 315cb6c337a8
GROWN, Thu 12 Nov, 11:04 PM PST.

**Part 3 design: the widened run waits on the harness's mapped start and on the evaluator. The start would have to build a tensor from θ's rows plus noised copies, and the evaluator would have to take a d_ff other than the start's. A route can hold the noise as buffers drawn at `build` from torch's global generator, the run's own numbers if nothing draws before them, but the noised rows need θ, so their combination has to be the harness's.**

## The run

r20261104-060905-8d56 (READ `out/train.py` 192-236):
- **Depth 8 to 10.** Block j copies θ's block ⌊8j/10⌋, so the sources are [0,0,1,2,3,4,4,5,6,7], and the repeats (j = 1 and 6) get `wo` and `w_down` zeroed. This is `grow` as it stands.
- **d_ff 2048 to 3072**, in every grown block, the repeats included, in block order, from `g = torch.Generator().manual_seed(seed)`:
  - `pick = randint(2048, (1024,), g)`;
  - `w_gate′ = cat(w_gate, w_gate[pick] + 0.01 · rms · randn([1024, 512], g))`, where rms is each picked row's own RMS;
  - `w_up′` the same, with a fresh `randn`;
  - `w_down′ = cat(w_down, zeros(512, 1024), dim=1)`, then zeroed whole on a repeat.
- Everything else is copied from θ. The model's own initialisation, under `torch.manual_seed(seed)`, is overwritten.

## What the harness's mapped start would have to take

- **An `origin` value beyond today's two.** Today a value is θ's tensor of the same shape, or zeros. `start_state` (READ `run.py` 166-195) refuses any other shape. Two more forms would do:
  - the rows of a θ tensor followed by noised copies of some of them: the source, the picks, the noise and the 0.01 scale;
  - a θ tensor padded with zero columns.
- **The record would have to state the map's draws**, or their digest, beside `origin`'s map and θ′'s digest (READ `run.py` 591-592).
- **No change to the counter.** INFERRED from READ `run.py` 367-377: `Harness.load` registers θ′ as the counter's start, so a widened tensor of θ′ is θ's kind, as a copied tensor is. Only a tensor mapped wholly to zeros is a constant, so a padded `w_down′` is θ′'s, which is right for it.

## What the evaluator would have to take

`Evaluator.net` (READ `evaluate.py` 76-98) builds "the start's shape at N blocks" and refuses a tensor of another shape: "a state is the start's shape at {n} blocks". It would have to read d_ff from the state (the rows of `blocks.0.w_gate.weight`), as it reads N from the block names.

## Can the noised copies be buffers drawn at `build` from torch's global generator?

**The draws, yes, under two conditions.** The job seeds the global generator with its seed (READ `run.py` 1285) and then calls `route.build` (1309).
- If `build` makes its draws first, before `LM.of`'s initialisation draws from the same generator, then the global CPU generator under seed s yields the same stream as `torch.Generator().manual_seed(s)`. At seed 0 it would give the run's picks and noise bitwise, drawn in the run's order: per block, `randint`, then the gate's `randn`, then the up's.
- That needs nothing in the job to draw between 1285 and 1309. That is INFERRED and untested; one job comparing the first draws with the program's would settle it.
- If anything draws first, the noise matches only in distribution.
- Seeds 101 and 102 draw other noise, which is what a final seed should do.
- As buffers outside the state dict, the draws are 1024 int64 and 2 × 1024 × 512 floats a block, about 4 MiB, 40 MiB for 10 blocks. They stay out of θ′ and E.

**The rows, no.** `θ[pick] + 0.01 · rms(θ[pick]) · noise` reads θ, and a route reads no weight of the start (the route contract), so it can hold the draws but not the rows. The combination is either the harness's, through the `origin` form above, or a computation of the route at its first step. The second fails: the start the route's model loads would then have to be another shape of θ′ first, which `start_state` refuses too.

## Verdict for the body

Listed as waiting: on `start_state` and `origin`'s forms (with the record's statement of the draws), and on the evaluator's d_ff. None is mine to change. I write no code for it.

