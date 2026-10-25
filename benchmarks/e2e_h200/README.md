# e2e_h200: the H200 walking skeleton's step driver

The GPU serving loop of the one-H200 demo (CMP-96): steps of the frozen MQA shape (`shape.FROZEN`, `mqa16k-l52`) from the
weights' store through decode-to-staging, every op's output committed through the per-step commit call, drawn units
replayed bit for bit against the Definitions, each leg run against the step or named, and a replay bundle and block record
written. `verity bench e2e_h200 --out OUT [...]` runs it (`run.py --help`).

It lives here, not in `integrations/`, because it is a run harness that composes pieces from several owners (memory's
store and decode call, kernels' GEMMs and commit call, circuits' Definitions, the legs) for one recorded run, as
`benchmarks/` holds; `integrations/torch_loop` stays the CPU loop of `verity.ml.windows.interface`. When the real pieces land
they replace the fixtures here, one module each, and the record's `stands_in_for` fields go to `null`.

## What runs, and what stands in

| Piece | Here now | Replaced by |
|---|---|---|
| Weights and their root | `weights.py`: seeded s8 codes and f32 scales; a FIXTURE sha512 | ALLP-IFACE's generator and registration root (CMP-95 item 1) |
| Store and decode call | `store.py`: P2 v4 (`p2-16448/v4`) rows, the P2 reference decode on the CPU, or kernels' `Stage.decode` with `--stage-lib`; `--store generator` regenerates instead | memory-accounting's layout and decode call (CMP-95 item 2), MEM-69's store on the card |
| W8A8 linears | `ops.s8_linear`: `torch._int_mm` on CUDA (int64 matmul on the CPU), then `bf16((f32(acc) · sx) · sw)` | kernels' KP2-S8 |
| Row ops | `ops.py`: the plan's functions (CMP-92 af6de1f1 §1/1b), sequential reductions, division and sqrt correctly rounded through f64 | circuits' row-op kernels |
| Slots and KV cache | `slots.py`: `SlotKV` admits requests FIFO into the lowest free sequence slots, runs each at its own position, frees and reuses a slot when its request finishes, and holds each slot's KV rows (one head of 64, ctx 32) as the step's bf16 bits; `check_slots.py` checks it at the frozen KV shape on a device | stays |
| Attention | `attention.py`: `GemmCoordinate_v2{GEMM_HOPPER}` and the MUFU ex2 table, numpy on the CPU, per group of equal positions | KP2-ATTN |
| Commit | `commit.py`: `verity_vllm.commit.row_tables` (`--commit reference`), its C++ twin (`host`) or its CUDA kernel (`device`); one table an op, a row a slot keyed (slot, position), the KV rows as QKV's words; `slots.commit_step` is the per-slot-subtree stand-in | kernels' commit tree with a subtree a slot (CMP-95 item 3) |
| Replay | `replay.py`: linears and QKᵀ/PV through `evaluate_batch` on their Definitions; row ops through numpy twins (`twin:<Name>`) until their Definitions are on `main`; a fixture coin draws K a family | PRF-170's sampling plan |
| Bundle and record | `record.py`: provisional v0 schemas | ALLP-IFACE's (CMP-95 items 5 and 6) |
| Legs | `legs.py`: PoUS spot reads, wipe and reload, certifier digests as fixtures; bandwidth and PoUW not run; integrity is the replay | MEM-67/69, CMP-97, NET-72, ALLP-H1LOOP/SALT, C-Flock `--zk` and Lean |

## Commands

~~~text
uv run python benchmarks/e2e_h200/run.py --out /tmp/e2e --commit host           # tiny shape, CPU, generator store
uv run python benchmarks/e2e_h200/run.py --out /tmp/e2e --batch 8 --requests 24 --prompt-bytes 8 --generate 24   # continuous batching
uv run python benchmarks/e2e_h200/run.py --out /tmp/e2e --store p2-encode --workers 8   # the tiny shape's P2 store (~1 s a block)
uv run python benchmarks/e2e_h200/run.py --out /tmp/e2e --tamper model.layers.1.mlp.up_proj   # exits 3: replay finds it
uv run python benchmarks/e2e_h200/run.py --out OUT --shape mqa16k-l52 --device cuda --commit device --store p2:C.bin --stage-lib DIR
~~~

Exit 0: every drawn unit replays to its committed word; 3: one doesn't; 4: a leg's fixture check fails.
