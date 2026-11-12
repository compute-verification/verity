[2026-11-12T21:22Z] a13acf765fba
BACKWARD2, Thu 12 Nov 1:22 PM PST

**Musts 1 to 3 and shoulds 7 to 10 are made and tested on my VM, not yet pushed; the GPU probe I would run on vy-nebius-2, one GPU leased under 5 minutes inside a job of about 20, and I launch nothing on a GPU before your word. One change goes past the review's text, in `Dq`, and I flag it below for your yes.**

Done on the branch, uncommitted (READ: `git diff --stat`, 3 files):
- Must 1: `Fa2TrainBwdPre` takes each thread's first product by `F32MulV2` and each after it by `F32FmaV2` into the sum; module docstring and Pre's say "FMUL, then FFMA". Test `test_the_dp_sum_is_fmul_then_ffma`: 64 rows at D 64 with dO and O at 2^-75 to 2^-60, Pre against the chain written out with the primitives' `evaluate`, and the FADD chain differs on more than a tenth of them.
- Must 2: a part's add into its buffer is `F32AddFtzV2`, the buffers' adds `F32AddV2`; the docstring names the atomics as the one flush-to-zero. Test `test_a_dq_part_s_add_into_its_buffer_flushes_to_zero` at NP 5 and 8, NS 4, parts at 2^-127 to 2^-120, against the chain written out; the FTZ-free chain differs on more than half the words.
- **Past the review, in `Dq`, for your yes:** with FTZ a buffer can be -0 (a negative subnormal sum flushes to -0; RAN: `F32AddFtzV2(2^-126, -(2^-126 + 2^-149))` is -0). The convert kernel clears its sum to +0 and adds every buffer by FADD in a runtime loop (READ: `flash_bwd_preprocess_kernel.h` at `14c37795`, lines 241 to 247), so its first add maps a -0 buffer to +0. The Definition used to start at the first buffer and leave that add out as "the identity on every word"; it no longer is, so the sum now starts at +0 and adds every buffer (D more gates a row). It shows only when every buffer of a row flushed to -0, possible at NP of at least 2 NS (an H 24 head's last key blocks). The test's NP 8 row pins it: +0 with the add, 0x8000 without. The docstring's "leave out" list loses the first add and dV's dropout FMUL (should 9).
- Must 3: `CONFORMANCE` now says what the fixture holds: dQ of five rows of the H 24 head, one telling the buffers' order, and dK and dV of the H 16 head's last key block; the FFMA and the FTZ are the capture's SASS, which no captured word tells apart. It gets widened only by the every-word run at the fixed head.
- Should 7: `test_the_head_at_a_stand_in_shape_keeps_its_signature_and_builds` (`monkeypatch.setitem` of `(2, 2, 16, 32): (16, 16)`, signature by type, builds on words and bits, equals the loop).
- Should 8: `NS=min(-(-SMS // (B * H)), T // blocks[1])`: H 8 now binds `NS=8`, H 12's id; the stand-in binds `NS=1`, the frontend's structure stand-in's id.
- Should 10: the generator takes BN and NS from `train_sdpa_backward(B, H, T, D).bindings` rather than repeating `SMS` and `BN`; checking it still writes the fixture byte for byte once the capture's fetch ends.
- RAN on the VM: quick tier 7 passed in 7.6 s; the slow key-block test passed (17 s), so the FFMA Pre still gives the captured dK and dV; ruff clean on the three files. Mutations, each alone and restored: the FADD dP sum, the buffers without FTZ, the sum without its first add, and the NS without its clamp each fail exactly the test meant for it.
- Shoulds 4 and 5 (the order pins) are next, for Fri 10 AM.

Probe (should 6), the plan:
- Node: vy-nebius-2, the capture's own node and card (READ: the TRN-140 finding note's capture section), research pool, the capture's job shape: a fresh torch 2.14.0 venv outside the lease, then one `gpu-lease 1 --max-min 5` command for the backward calls (seconds of GPU), then a CPU fit of the probe's rows through Pre, Row and Dq on a CPU node.
- Cases: dO near 2^-120 with q, k, v N(0, 1), so dQ parts are subnormal and FTZ shows in most dQ words; and dO, O and v near 2^-70, so the dP sum's products are subnormal. INFERRED: the second may tell FFMA from FADD on few words only, since bf16(dS) has 2^-133 resolution and the two sums differ by about 2^-149 a product; before asking you to launch I count on my VM, with the Definitions on crafted rows, how many words each case's readings differ on.

Runs: `r20261112-192536-b84b` (suites at the old head) took check slot r2 at 20:27Z and is running: verity-vllm passed (14,314 passed, 400 skipped), no failure in its log so far (RAN: fetched `out/suites.txt`). `r20261112-194241-5573` (every word at the old head) took slot r0 at 21:05Z and is running, 16 processes, ETA about 22:00Z. I let both finish: the every-word run still checks the head's wiring (Row, Key, the loop), though it counts for nothing at the fixed head.

[2026-11-12T21:23Z] ba3e9177ff6c
THREADS, Thu 12 Nov, 1:23 PM PST.

**PLAN's three lines hold, and the third is worse than PLAN read: a stop during the state's fetch can fail the job outright, not only send it back to θ. My bound counts kept states, not steps. Item 5 is about eight lines in the same place, so I build it. A keep between checkpoints touches four ties in `run.py`, and the trace can already run in a lease of its own.** About my context: my part of this conversation is near half. I take this job and keep the reads short; if I run short, I send you the hand-off instead. Branch: `cursor/training-stop-before-up-45e7`, cut from `main` at `5f0c008aa0`.

PLAN's three lines (READ: `benchmarks/training_slowdown/run.py`; `training_slowdown/run.py` `outside`):

1. Confirmed. In `store_state`, a run is done when it has rc 0 and a record. Otherwise it is stopped only when `told(said)`, where `said` is the record's `out/run.json`. Every other run is failed.
2. Confirmed. The only writer of the stop record is `outside`'s `stop` handler, installed by `signal.signal(SIGTERM, stop)` once `run.main` has imported torch and parsed its arguments. `ENV`'s `trap : TERM` writes nothing. A stop during `uv sync`, during the fetch or during Python's start therefore leaves no `run.json`.
3. Confirmed (`run = runs[-1]`, then `store.keeps(record)` of that run alone). One correction:
   - The fetch writes straight into `$OUT/state`. A run stopped partway leaves a partial `out/state` in its record.
   - If its `state.json` arrived before the tensors, `keeps` lists that file's sha256, and the next run fetches the partial state under `--continues`. `Kept.load` then refuses it ("its tensors ... are missing" or "don't load"). That is a `SystemExit` with no stop record, so the job fails for good. (INFERRED: I haven't checked the order in which `research data fetch` writes files.)
   - If `state.json` did not arrive, the job starts from θ, as PLAN read.

The bound I chose:

- A label is failed when its last three runs (`STALLS = 3`) were each stopped and none of them kept a state newer than the one before it. "Newer" means the sha256 of the `state.json` in its record is present and differs from the latest state an earlier run kept.
- I can't observe "no step" from the store without fetching every run's `state.json`. The kept state is the observable that matters: steps after the last kept state are lost anyway, so such a run advanced nothing for the label.
- Against your wording, this also fails a job whose runs keep being stopped before their next checkpoint (a lease shorter than a checkpoint interval, or three evictions in a row inside one interval), rather than letting it loop.
- The message names the three runs and says why.
- If you want stops before the job is up counted apart from stops after it, the shell's stop record would need a mark of its own, which makes two kinds of stop record. I'd rather not.

Item 5, about eight lines, so I build it:

- `Store.says` takes a path.
- For the run whose state is continued, `store_state` reads that run's `out/state/state.json` and takes `record.threads`.
- `research_argv` adds `--threads N` when the record states one.
- Cost: one more fetch per stopped job each time the command runs. `state.json` holds the record so far, so it can be several MB.

Sizing a keep between checkpoints (READ: `run.py`; nothing built):

- What is tied to a checkpoint today:
  - `Kept.write` names the tensors by the last checkpoint's step, `tensors-{checkpoints[-1].step}`.
  - `Kept.load` refuses a model that doesn't hash to the record's last checkpoint (`trained_sha256` or `output_sha256`).
  - `Harness._after_step` calls `keep` only at a checkpoint step that isn't the last.
  - `joined` and `assert_continues` assume the stopped segment's last kept step is a checkpoint. They count a route `output` computed at that checkpoint into `setup`, and the next segment's first step is short of it.
- What a keep between checkpoints needs:
  - A trigger in `_after_step` every few minutes, read from the harness's injected clock.
  - Tensors named by step.
  - The model check moved from "hashes to the last checkpoint" to the segment's own SHA-512 of the model part, which `kept` already states.
  - A `joined` and `assert_continues` that expect no short step when the stop was between checkpoints.
  - The keep's own time paused out of `elapsed()`. It currently runs in `uncounted()` with the clock going, so frequent keeps eat `budget.minutes`.
- What already works between checkpoints: the scores are only the checkpoints' (`with_scores` on the record so far), and the clock is `te.elapsed()`. The record, the stream's position, the generators and the fresh-windows record are all whole at any step boundary.
- Resume tests to add:
  - A stop at a non-checkpoint step whose joined record equals the unbroken job's, through the quick harness and through the slow two-process test.
  - A kept state whose model differs from its own `kept` digest, refused.
  - A keep whose time the budget doesn't count, through the injected clock.
- The trace in its own lease: possible with what exists. `--retrace RUN_JSON` takes a record's trace again from the record's config text, on a freshly built route (`route.build`). It needs no trained weights, because the capture reads shapes. A job could write its record with `trace.failed`, and a separate queued `--retrace` run would write the traced record. The command would need a second label or a step to queue it. That is not built.

Next: build items 1 to 5 with a test for each that fails without its change, then run the suite on a node, slow tests included.

