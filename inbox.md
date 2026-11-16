[2026-11-16T23:15Z] e22f4c18cf62
model: claude-opus-5-5-high (Claude Opus 5.5); subscriptions removed: none held

ROUTES to the training lead, 3:15 PM PST. STEP 1: the Lean verifier still refuses with θ set. I stopped there; there is no PR. Each fact is marked: read, ran or inferred.

What I ran (run r20261116-230053-f70c, done, on CPU from a fresh clone of origin/main 6469852e760; an earlier run, r20261116-224758-2211, gave the same table but my print cut off Lean's numbers):
- I built the class the benchmark builds: verity-int8/Llama-3.2-1B-INT8, class 4,32,4, 4 slots. Same c 30,087,822,573,678; Python's k = 145,568.
- θ came from the same pins as Python's count (`built.record`'s pins = every registered weight; `IF.held_params`).
- Of the session's 171 parameters, the 10 whose every read is a held weight (the ScaledMmS8 and Gemm `w` and `sw` ports, RoPE's `cs`) went to `Program(population(fn, m), registered=..., theta=...)`, which makes them `RegisteredInput<w>_v1` roots (10 θ root nodes, from `annotations["theta"]`).
- This needs no change to verity/circuits/population.py: `Program` already takes `registered=` and `theta=`.
- Then `inference-only-check --count --detail` on each instance:
  - θ empty, 38 s: row [gates 30087822573678, fresh 446008121680, blocks 4], total 4, "complies": false, "failure": [0, "fresh"].
  - θ = held weights, 42 s: row [gates 30087822573678, fresh 25967909200, blocks 4], total 4, "complies": false, "failure": [0, "fresh"].
- So with θ: not complies, 25,967,909,200 fresh bits a block against k = 145,568, about 178,000 times over.
- θ took off exactly 420,040,212,480 bits, which is exactly Python's held figure for those 10 parameters. On the weights, Lean and Python agree to the bit.
- Lean prints no message of its own beyond that JSON (with --detail it prints the verdict object). The "no θ and no dataflow" sentence is `built.judged`'s Python text, not Lean's.

What Lean still counts as fresh, per parameter (ran; per-parameter bits from the session signature, sorted by the record's declarations):
- 26,094,850,352 bits: wired rows. These are ports the record's wires fill from other units' outputs in the same slot: ScaledMmS8.x 6.2e9 and 9.4e9, Gemm.x 9.2e9, SiluMul.gu, Int8RowQuant.x, the attention k, v and o, the RMSNorm res and so on.
  - Python's `IF.input_count` subtracts them as `wired`.
  - The session Program Lean gets (`population_program`: independent units, no wires) has them as root inputs, so they are fresh to Lean by its own definition.
- 1,541,120 bits: EmbeddingRead's `row`, `path`, `salt`, `root` and `domain`, the openings of the pinned embedding table.
  - Python counts them as `registered` (openings of a table over a held parameter).
  - They are per-run values, so they can't be θ (`theta-not-registered`).
- 145,568 bits: the scan state (`verity/certifier/scan-state/v1`, read by ScanSlots_v2.state). It is fresh in both, and it is Python's whole k.
- (Lean counts only the leaves a block's gates read, so its 25,967,909,200 is about 128.6M under the 26,096,537,040 sum of the first three.)

Is it a difftest finding? Yes, inferred from the files read:
- Python's count and Lean's count are not run on the same object. Python counts the circuit record (its wire declarations and table openings). Lean counts the session Program, which carries neither.
  - block_counts.py's docstring says so: "The wires are the record's declarations, which the session Definition doesn't carry, so this is the record's count and not InferenceOnlyCheck.Count's fresh on the session Definition, which would count every wired read".
- The difftest holds them together only on gates and the count, never on fresh bits. `tests/adapters/vllm/tests/program/predict/test_block_counts.py::test_the_count_is_ncicheck_counts_on_the_session_program` compares gates and total with the Lean reference, runs it at k = 10^12, and holds `cls.fresh` only to Python's own `input_count`.
- So k = 145,568 has always been Python's figure, with nothing in Lean behind it. The accept in the workload's record is that figure.
- I didn't paper over it and didn't change what counts. I built no made instance yet. The smallest one would be the `small` fixture in that test (GEMM and RoPE over two slots) plus one declared wire from a GEMM output to a RoPE input: Python's fresh drops by the wire's rows, Lean's doesn't. On your word I'll build it as the difftest's failing vector.

What closing it takes (inferred): two things, neither in benchmarks/one_stage/inference_only.py, so not today's wiring.
- First: Lean must see the wires. Either the Program handed to it carries the record's wires as dataflow (a reader port computed from its writer's output), which changes the session Definition the vLLM adapter builds (`block_counts.session`, `population.py`'s `mixed_population_program`), or Count's instance takes the record's wire declarations.
  - The second path is a Lean change to Count.lean, and to what `InferenceOnly.CompliesAtMost` reads. A wired row is then not fresh only on the premise that SPOT's link verification binds it to its writer, which is a statement in security/Definitions/InferenceOnly.lean (Daniel's).
- Second: the table openings, 1,541,120 bits. Even with every wire internal, the Lean verifier's fresh count would be at least 1,686,688 > 145,568 unless the openings of a pinned table count as registered. Same choice: model them in Count's instance, or charge them, which raises k.
- The θ half, which I built, is right and small. It goes in both `count_instance` (inference_only.py) and `built.instance` (benchmarks/inference_only), and it isn't enough. The branch `cursor/training-inference-only-lean-decides-4207` exists at origin/main with no commit; the θ builder is the experiment's ten lines, not yet in the tree.

#10619 and the composed record (read): it copies the inference-only policy and limits into the compose hand-off and runs assemble.py beside the record under PEARLPLUS_TIE. It adds no row.
- The composed verdicts.json's rows come from served_verdicts.py's record step: `rows = {"integrity": ..., **PIECES (work, space, bandwidth, communication)}` plus wipe. There is no inference-only key.
- So after my change or #10619, the composed verdicts.json still would not carry the row. It needs a row built from inference-only.json in `rows`, plus its key in `record()`'s subprotocols and in the composite's product list. That is a few lines in served_verdicts.py, not one.

Nothing posted to Linear or Slack. I hold the branch for your word: (a) stop here, (b) build the made difftest vector, or (c) land θ alone, which changes no verdict.

