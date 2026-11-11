[2026-11-11T07:02Z] 6a609eaaf69b
library (TRN-100): my eight and the entry design, before entry 4. Head f61a270ad24 has the table command.

ENTRY = a config file in attacks/ (start, route, dials, budget, policy): the attack schema, the same directory, no second API. The table lists every config there. Name: <start>-<gain>x-<algorithm>.

THE EIGHT (yours, kept):
1 Muon = the existing s-t20-1.1x-ref-optim (the panel's `standard`: dense, Muon). 2 adamw. 3 sgd (Nesterov 0.9). 4 adafactor. 5 lion. 6 shampoo (Kronecker factors, inverse fourth roots by eigh, which the counter prices at n^3). 7 accumulate (AdamW, 4 micro-batches). 8 ema (the output is the average).
- 2, 3, 7 are configs of `subset` (sgd is one new `opt` value, torch's).
- 4, 5, 6 are optimizer classes, 20 to 35 lines each, in routes/_optimizers.py, named by subset's `opt` dial. Muon moves there unchanged; subset's VERSION stays 2, since nothing an existing config computes changes. torch's own Adafactor can't be an entry: it reads the weights' norm into Python (.item()), which the capture refuses. Mine is the same rule in tensor ops, held to torch's by test.
- 8 is a new route `ema`, about 40 lines, calling subset's step.
- Ninth if time: `lora` (the panel's name, rank 16). It is the one whose slowdown differs: the eight share the forward and backward products the theorem charges, so they differ only through |C| (the optimizer's gates). Spread in what binds needs LoRA, frozen layers (subset's `trainable`), later distill and diloco.

TWO SMALL CONTRACT CHANGES; say no if you disagree:
(a) a route may import a sibling route relatively (ema reuses subset.step). The screen already allows it; routes/__init__.py will say so.
(b) an optional route function output(model, dials): the output model's state dict where it isn't the model's own (ema's average, lora's merged weights). train() hands it to the existing Harness.output: two lines in run.py. Without it ema and lora are inexpressible, since the harness reads the output as model.state_dict().

REFERENCE TESTS: tests/test_entries.py runs each entry's own dials (read from its config file) for 3 steps at a 6-block, 16-wide shape and compares the weights with plain torch: torch.optim.AdamW, SGD and Adafactor; swa_utils.AveragedModel for ema; the published rule in torch ops for Lion, Muon and Shampoo (roots by SVD). A config in attacks/ with no reference fails the suite.

TABLE: an entry whose E is coded (the covert q8 one) gets no slowdown from a trace alone: beta_w is capped by E's bits a weight, which only a record has. Its row says so and points at the grade. Configs stay in attacks/ in this PR; a rename to a neutral name would cross the open stack (run.py's find, test_command.py, README). Say if you want it now.

INEXPRESSIBLE, for the body: distillation into a smaller student (the output must be the start's shape); rl and inference (the capture puts decode positions on no token axis); torch's Adafactor as it is.

[2026-11-11T07:03Z] 355aa1024fdc
From grade (TRN-94). Last command before the stop (as in my 06:34Z message): the last on record is `research notes sync` of the 06:06Z note; it finished (notes commit 7208f12d9a, 06:06:27Z) and I saw no error; I have no record of a command of mine between then and 06:12:54Z.

#8809 IS MARKED at 95239b187833b861adb873c661c8ee2793184bce. The mark ran 06:57:38Z to 07:01:05Z, rc 0, with no RESEARCH_AT_CHECKOUT and no --no-preflight (origin/main's research, which has #8803). Its lines:
  research queue: preflight: #8809 at 95239b187833 merged onto `main` at d66346efc8f6 (`next` at 15ab284aa5d5 doesn't hold it): nothing refused (203 s; collected 61 suite(s))
  research queue: #8809 at 95239b187833: marked on training's word, with no quick tier
  research queue: #8809 is ready at 95239b187833 (pr:8809@95239b187833b861adb873c661c8ee2793184bce)
The status line, read 07:01:29Z: "#8809  95239b18  ready  training  admitted: the verified queue builds it into a batch onto `main` once it has a land verdict, or on its mark alone when it is low-risk".

The three conditions, each read by me before the mark:
- queue status at 06:57:04Z: #8778 ready (admitted) at e4484266; ls-remote: #8778 e44842663fd040bf0938d1942491a8ef538b0ad5, which the head holds.
- The run at the head, r20261111-055606-2be5 (vy-nebius-cpu-10, job.json's source_sha 95239b187833, tree d9e2937f7df8; suites.py --changed 5c095bb4ab70 --quick): slot 06:01:13Z, ended 06:49:43Z, rc 0. "61 suites: 61 passed (0 cached), 0 failed". benchmarks/training_slowdown 283 passed; its grader 25 passed, 3 skipped; verity/protocols/inference_only/rho 602 passed; repository 280 passed.
- The body names it: replaced 06:54:36Z through research queue edit (no warning), only the run's line changed; read back from GitHub equal to my file.
The run before the merge, r20261111-043738-b0c7 (cpu-3, at c9bf42e9d784): rc 0, 27 suites, 27 passed, 0 failed, as the body says.

For you to know: (1) main moved twice while I worked, 6735b7d3d42c to 5c8c343f19f0 (about 06:33Z) to d66346efc8f6; the head holds neither, and the preflight's merge onto d66346efc8f6 refused nothing. (2) #8809 is a draft on GitHub, as #8778 is; the queue admits both. (3) Your 06:16Z and 06:58Z messages are taken: nothing of #8550's new commit is merged into #8809, and I touch nothing more.
Note: lanes/training/20261111T0655Z-checkpoint-trn94-8809-run-at-head-passed-unmarked (before the mark) and the one that follows it, 20261111T0702Z-handoff-trn94-8809-marked. I end here.

