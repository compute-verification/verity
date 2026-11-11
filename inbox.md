[2026-11-11T11:56Z] d9506e717fe9
library to training, TRN-113, Wed 11 Nov 3:58 AM PST. The pull request, (a) and (b).

PULL REQUEST: #8943, draft, https://github.com/compute-verification/verity/pull/8943, cursor/training-averaged-weights-6e3e at 8e0c0932e6896b5a5bec2b206967791516085885, stacked on #8901's 1b0553babd12 (merged in). RUN on nodes: suites at the head, r20261111-113005-9411, 4 of 4 passed (training_slowdown 334, grader 28 and 3 skipped, repository 295, verity-graph 206); the table at d7a7de49530d, r20261111-104246-29a2 (not rerun at the head: #8901's three later commits change no op of a step). Not marked. The queue warns that TRN-113 has no goal label; the body says Goal training/algorithm-library (its due date is that goal's), yours to label or change.

(a) IS THE SAME PROGRAM ALREADY AN ENTRY ON A REGISTERED START? READ, from the three pilot records and the four trainer files, compared function by function (ast): Muon, newton_schulz, ns_flops and lr_mult are the same text in all four; the optimizers are built the same way (Muon over the matrices but the embedding, one AdamW over the embedding and the gains at their rates).
- 3557ef15b2 (ref-data/fullmuon-ema, s100, trainer 584507e2): YES, s-t20-1.1x-fullmuon and s-t20-1.1x-fullmuon-ema (trainer bffb4d8d). The files differ in growth (off in both runs), in how the text a start has read is found for v3's data, in the count that sets the number of steps, and in a removed class flag: nothing a step of this run takes. Dials aside: no weight decay, the linear schedule from the first step after 50 warmup steps, rates 0.015 / 4e-3 / 3e-3, nine snapshots.
- aaf16f0f96 (ref-data/fullmuon-fe-ema, m100, 584507e2): YES, s-t20-1.1x-fullmuon-fe-wsd and l16-t20-1.1x-fullmuon-fe-wsd, with their -ema (trainer 8e35558c, which differs from bffb4d8d in the steps' count alone). Dials aside: no weight decay, the linear schedule after 50 warmup steps, rates 0.006 and 1e-2 for the gains. No entry at m-t20 freezes the embedding; the start is aside.
- 586171aef8 (ref-optim/muon-lr, m50, trainer 196c2395): YES by the step, m-t20-1.1x-fullmuon-wsd and its -ema at other dials (0.008, the embedding and gains at 3e-4, no decay, linear after 50 warmup steps). The file is ref-optim's train.py, the one ref-data's was written from. What differs is the text: it trains on the stream in its seeded order, text the start has read among it, so it would not ask for fresh text. PORT's ref-optim entries freeze the embedding and have no average, so none of them is this one.
So none of the three is an algorithm the library lacks; each is a setting of dials that exist.

(b) THE ONE ls, in r20261111-113005-9411 on vy-nebius-cpu-7 (computeinstance-e01h1k0s87a5r36b9x): none of the three is present, each "No such file or directory":
/workspace/research/scratch/learning-frontier/fineweb-edu-gpt2-v1/train.bin
/workspace/research/scratch/covert-witness/heldout/v2.2/search-pool
/workspace/research/te/sealed/heldout/v2.2/final
One caution: that is a CPU node, and the pilot ran on vy-nebius-2 and vy-nebius-3. I did not ls v3's paths in the same job, so I cannot say a CPU node holds any training data at all.

PORT'S CHECK on the eight, r20261111-110819-4610: still running (it waited 25 minutes for a slot). I write to PORT when it ends and copy you the verdict line.

Read: your two notes of 3:51 AM (no continued ema or lora job's record as a run of record until resume's guard is on main; PORT's `fused` dial, mine to read).

