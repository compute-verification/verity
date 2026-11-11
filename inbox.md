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

