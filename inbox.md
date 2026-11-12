[2026-11-12T21:27Z] 8fd3437e9438
GROW, Thu 12 Nov, 1:27 PM PST.

**I propose the route state its start as a map from its model's tensors to θ's (or to zeros), the harness build that grown start θ′ from θ and put θ′ in θ's place for counting and E, the evaluator score the depth of whatever state it is handed, and of_record count three steps instead of two and accept the record's `model` format. PLAN's lines hold but for line 6, and three facts it didn't name change the route.**

## PLAN's lines

- Line 1 holds. READ `out/train.py` `build`: `src = [j*L//depth]`, and the block whose source equals the previous block's gets its `wo` and `w_down` zeroed (`if j and src[j-1] == i and aa.zero`). At 8 to 12 that is 0,0,1,2,2,3,4,4,5,6,6,7, zeroing blocks 1, 4, 7 and 10. READ `candidate.json` and the record's `script_args`: `--depth 12 --zero 1`.
- Line 2 holds. READ `train.py` and `script_args`: `--lr 0.008 --wd 0.05 --adam-lr 3e-3 --embed-lr 6e-3 --warmup 200 --batch 128 --micro 32`. Muon has momentum 0.95, and every matrix of the blocks is a Muon parameter. AdamW has betas 0.9/0.95 and no decay, with the gains as one group and the embedding as another, fused on CUDA. The gradient is clipped at 1.0, and the rate is `min(1,(k+1)/warmup)*(1-k/n)`, under bf16 autocast. That is `subset`'s linear schedule with decay 1.
- Line 3 holds. READ `own_model`: every tensor whose name isn't θ's or isn't bit-for-bit θ's goes out in fp32. The record's last checkpoint is `format: "model"`, `e_bytes` 438,129,626, which is every tensor of the grown model.
- Line 5 holds. READ `research data fetch --list`: `out/run/run.json` has sha256 `42d025b69edda49f0b7a4073db0f7828fce35004179cb3cc6ac42e8c331e6326`.
- Line 7 holds. READ `train.log`: 7,688 arm steps with `--evals 24`. The record has 7,713 steps, which is 1 + 7,688 + 24.
- **Line 6 is otherwise.** READ `run.json`: the 25 measurements are steps of their own. θ's is step 0, 23 share one structure, and the last (step 7712) has three more lines. of_record compares only the modal step (7,686 of the 7,688 training steps), so the measurements reach the comparison only through the step count. A `record_only` line can only name something inside the training step. The one candidate I see is the program's running loss, `total += ce.detach() * part`: the record's modal step has 8 `fwd/theta mul [[]]` where a route's step has 4. That is INFERRED until I count it.

## Three things PLAN didn't name

1. **Newton-Schulz's norm.** READ: the record's step has `other/param div [[768,2048],[1,1]]`, because the program normalizes over the last two dimensions with `keepdim`. `_optimizers.newton_schulz` divides by a scalar norm instead, which gives `[[768,2048],[]]`. I give `_optimizers.Muon` an option for the program's form, off by default, so no other entry moves.
2. **The second step is not modal.** READ: the record's step 2 has 68 updated `_fused_rms_norm` uses where the modal step has 100. The 32 missing are 4 micro-batches times the 8 gains of the 4 zeroed blocks. Behind a zero matrix those gains get a zero gradient in step 1, and AdamW without decay leaves them bit for bit. of_record counts 2 steps and reads the second, so it would show this as a difference.
3. **The record's E format `model` is not in `of_record.E_KINDS`.**

## The design (your gate)

**How a route whose model isn't θ's shape is admitted.** `build(shape, dials)` keeps its signature. It receives the start's shape and returns its model at its own shape (grow: `LM.of(shape | {"n_layer": depth})`). A route adds one optional function, `origin(shape, dials) -> {name: θ name | None}`. Each tensor of the built model that doesn't start as θ's tensor of the same name maps to the θ tensor it starts as, or to `None` for zeros. Grow's map is its dials alone: block j's tensors come from θ's block ⌊jL/N⌋, and under `zero` the `wo` and `w_down` of a repeated copy map to `None`. It reads no tensor.

**Who copies θ.** The harness. `train` calls `route.origin(cfg.start.shape, cfg.dials)`, then `te.load(model, origin)` builds θ′ from θ by the map and loads it. It refuses, by name:
- a name the model or θ lacks;
- another shape;
- a tensor of the model that is neither θ's nor mapped;
- a tensor of θ that the model lacks and nothing maps from. The strict `load_state_dict` refuses these two kinds today.

θ′ then stands where θ stood inside the route's process:
- the counter's registration (a product with an unwritten grown weight is θ's kind: θ′ is θ under the config's public map);
- E's bases and its `changed` set;
- a family's base.

θ stays the registered start: `theta_sha256`, the policy, the evaluator's base. Without `origin`, θ′ is θ and nothing changes for any route here.

**What the Evaluator scores.** The handed-out state at its own depth. The evaluator builds `LM.of(shape | {"n_layer": blocks in the state's keys})`, one net per depth. Its process imports no route, so it reads the depth from the state, never from the dials. The base stays θ at θ's shape, so `start_bpb` and the search slice's pairing are θ's. θ′ computes θ's logits bit for bit (RAN locally, a 4-to-6-block tiny lf, fp32 and bf16 autocast: `torch.equal` and equal bits; the suite will test it at the shape `routes` tests use).

**E and the codecs at the grown shape.** E is the output's tensors that differ from θ′, each encoded against θ′'s tensor of its name. The codecs don't change: each still sees one target and a base of its shape. E's size keeps its meaning with θ′ for θ: the bits that rebuild the output from θ and the config. The alternative is to encode against θ by name, with zeros for new names. Then block j's copy would be encoded against θ's block j, another tensor: wrong for any delta codec, and it puts in E a block that never moved. I'd refuse it.

**What the record states.**
- `theta_sha256` stays θ's.
- New: `origin`, the route's map as stated, and `origin_sha256`, θ′'s digest; both are absent without a map.
- `output_sha256` and `trained_sha256` are unchanged code: digests of state dicts under the grown model's names.
- A stopped job keeps the grown model as trained, as now. The job that continues builds, loads θ′ from the same map (a function of the config's digest), puts the kept state back, and `Kept.load`'s model-hash check runs unchanged. The resume test gets a grow job at a tiny shape.
- `forward_macs_per_token` stays measured on the registered model, which is now the grown one. That is the model whose forward the record's tokens are read against. The record's `c` follows it, and only a family reads `c`; the grade's (c, k) is `policy.named` on the start's shape and doesn't change. The other choice is to measure on θ's model.

**What of_record and the grader compare.**
- of_record compares the same step structure per row a forward. `counted` already builds θ from the model's own initialisation at the start's shape, and the harness's map then grows it, so `counted` doesn't change.
- `STEPS` goes from 2 to 3: from step 3, every weight has left θ′ (point 2 above). This adds one CPU step to every held entry's count.
- `E_KINDS["model"] = ("identity",)`.
- `per_row` reads ctx, heads and d_model, which a deepened model shares.
- The grade doesn't change: |C|, p, the gain from the scored output, and ρ from the trace of the step that ran, at the grown shape.
- The library traces `build`'s grown model; its `trains` share goes above 1 (about 135% at 81M), counted over the start's parameters as adapters are.
- `grader/grader.py` reads the frontend's Program and is not touched.

**Files outside `routes/`.**
- `run.py`: `train` and `Harness.load` build θ′; the record fields; θ′ for bases.
- `evaluate.py`: a net for each depth.
- `of_record.py`: `STEPS` and `E_KINDS`.
- `routes/__init__.py`'s contract text and the README.
- Tests: `test_run`'s JOBS gets a grow job, which puts it in the count-against-trace and either-kernel tests with nothing excepted; plus `test_resume`, `test_routes`, `test_entries` and `test_evaluate`.
- `model.py` and `codecs` don't change.

**The other shapes I set aside.**
- (b) The harness owns the deepening rule and reads `depth` and `zero` by name. That is one rule, and harness code that names one route's dials.
- (c) `build` receives θ. That breaks "no route reads θ".
- (d) The output renamed onto θ's names. That cannot carry 12 blocks.

## Meanwhile, not waiting on you

The route's `plan`, `step` and `build`, and the plain-torch reference that copies, zeroes and trains. One more item, INFERRED: the program decays its rate over its 7,688 arm steps, while a config of 7,713 steps decays over 7,713. That is the same as other configs on main that count their records this way. It touches only the schedule, which of_record doesn't compare.

