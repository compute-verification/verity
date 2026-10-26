# e2e_h200: the H200 walking skeleton's step driver

The GPU serving loop of the one-H200 demo (CMP-96): steps of the frozen MQA shape (`shape.FROZEN`, ALLP-IFACE's `H200`) from the
weights' store through decode-to-staging, every op's output committed through the per-step commit call, drawn units
replayed bit for bit against the Definitions, each leg run against the step or named, and a replay bundle and block record
written. `verity bench e2e_h200 --out OUT [...]` runs it (`run.py --help`).

It lives here, not in `integrations/`, because it is a run harness that composes pieces from several owners (memory's
store and decode call, kernels' GEMMs and commit call, circuits' Definitions, the legs) for one recorded run, as
`benchmarks/` holds; `integrations/torch_loop` stays the CPU loop of `verity.ml.windows.interface`. When the real pieces land
they replace the fixtures here, one module each, and the record's `stands_in_for` fields go to `null`.

## What runs, and what stands in

The units are circuits' plan's at size classes, one unit a row (CMP-92 af6de1f1, revised 23:30Z, section 1b): a row op's
output is one row unit (`RmsNormRow`, `QuantRowS8`, `SoftmaxRow`, `Bf16AddRow`, the embedding and argmax rows), and each
linear, QK^T and PV coordinate is one unit (`ScaledMmS8Coordinate_v1{K}`, no K-split; `GemmCoordinate_v2{K=d}`;
`GemmCoordinatePad_v1{T, K=32}`). `driver.py` names the step's tables.

| Piece | Here now | Replaced by |
|---|---|---|
| Weights | ALLP-IFACE's generator and weights stream (`verity_torch_loop.weights`, CMP-95 item 1); the record carries the registration digest under the test salt key for small shapes | the prover's salt key's digest |
| Store and decode call | `store.py`: P2 v4 (`p2-16448/v4`) rows over the weights stream, the P2 reference decode on the CPU, or kernels' `Stage.decode` with `--stage-lib`; `--store generator` regenerates instead | memory-accounting's layout and decode call (CMP-95 item 2), MEM-69's store on the card |
| W8A8 linears | `ops.s8_linear`: `torch._int_mm` on CUDA (int64 matmul on the CPU), then `bf16((f32(acc) · sx) · sw)` | kernels' KP2-S8 |
| Row ops | `ops.py`: the plan's row functions; sums sequential (on the host), division and sqrt correctly rounded through f64 | circuits' row-op kernels |
| Attention | `attention.py`: `GemmCoordinate_v2{GEMM_HOPPER}` and the MUFU ex2 table, numpy on the CPU | KP2-ATTN |
| Commit | `commit.py`: `verity_vllm.commit.row_tables` (`--commit reference`), its C++ twin (`host`) or its CUDA kernel (`device`); one table an op, a row a slot | kernels' commit tree with a subtree a slot (CMP-95 item 3) |
| Replay | `e2e_h200/units.py`: coordinate units through `replay.py`'s templates (main's Definitions); row units through compositions of `replay.py`'s word functions until circuits' row Definitions land; a fixture coin draws K a family | circuits' row Definitions; PRF-170's plan through `replay.draw` |
| Bundle and record | `record.py`: provisional v0 schemas | IFACE-B's (CMP-95 items 5 and 6, `benchmarks/e2e_demo`) |
| Legs | `legs.py`: PoUS spot reads, wipe and reload, certifier digests as fixtures; bandwidth and PoUW not run; integrity is the replay | MEM-67/69, CMP-97, NET-72, ALLP-H1LOOP/SALT, C-Flock `--zk` and Lean |

`replay.py` and `cpu_step.py` are the replay lane's (#4460): the CPU replay of drawn units in the 16-bit forms, and an
independent CPU decode step; the suite holds the driver's rows equal to `cpu_step`'s.

## Commands

~~~text
uv run python benchmarks/e2e_h200/run.py --out /tmp/e2e --commit host           # the small shape, CPU, generator store
uv run python benchmarks/e2e_h200/run.py --out /tmp/e2e --shape tiny --store p2-encode --workers 8   # the tiny shape's P2 store (~1 s a block)
uv run python benchmarks/e2e_h200/run.py --out /tmp/e2e --tamper l1.wup   # exits 3: replay finds it
uv run python benchmarks/e2e_h200/run.py --out OUT --shape h200 --seed demo --device cuda --commit device --store p2:C.bin --stage-lib DIR
~~~

Exit 0: every drawn unit replays to its committed word; 3: one doesn't; 4: a leg's fixture check fails.
