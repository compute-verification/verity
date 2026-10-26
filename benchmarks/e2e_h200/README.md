# e2e_h200: the H200 walking skeleton's step driver

The GPU serving loop of the one-H200 demo (CMP-96): steps of the frozen MQA shape (`shape.FROZEN`, ALLP-IFACE's `H200`) on
the plain INT8 weights, loaded once into the card's memory and resident across steps (Demo A, M1-M3; `--weights staged` keeps
M4's decode-to-staging path from the store), every op's output committed through the per-step commit call, drawn slots
opened by replaying their requests and drawn units replayed bit for bit against the Definitions, each leg run against the
step or named, and a replay bundle and block record written. `verity bench e2e_h200 --out OUT [...]` runs it
(`run.py --help`).

It lives here, not in `integrations/`, because it is a run harness that composes pieces from several owners (memory's
store and decode call, kernels' GEMMs, attention and commit call, circuits' Definitions, the legs) for one recorded run, as
`benchmarks/` holds; `integrations/torch_loop` stays the CPU loop of `verity.ml.windows.interface`. When the real pieces land
they replace the fixtures here, one module each, and the record's `stands_in_for` fields go to `null`.

## What runs, and what stands in

The units are circuits' FAMILIES map's (PRF-166 db928eb0; `e2e_h200/plan.py`, `shape.Forms`). The residual stream and every
row op's rows are chunk rows of 128 words. RMSNorm is `RmsSumSqChunks_v1{4096,128}` (32 chunk sums a row) and
`RmsNormOutSums_v1` a chunk row. The quantizer is `RowAbsMax_v1` over parts of 16,384 and `QuantChunkS8_v1{8192}` a chunk. Each
linear coordinate is one `ScaledMmS8CoordinateRows_v1{K, R = K/8192}`, gathered into its readers' rows by `AllGather_v1`.
Attention is padded to 32 keys: `QkSoftmaxRowPad_v1{T=32,D=64}` and `PvRow_v1{T=32}` a head, the heads paired by
`AllGather_v1{2,64}`. Then `Bf16AddRow_v2{128}` a chunk row, `EmbeddingCoordinate_v1` a coordinate and `Argmax_v2`. At the
frozen shape a slot is 6,102,102 units in 6,102,832 rows. `driver.py` names the step's tables, and `commit.py` commits each
unit's output rows as one commit unit. The `MINI` shape's forms are small enough for the CPU suite to reach every level of
the map.

A step (`Driver.step_slots`) runs a batch of rows, each the KV pool's slot at its own position; `Driver.step` is the
lockstep step `run.py` drives, and `Driver.serve` (`run.py --requests`) serves requests through `slots.SlotKV`'s slots by
continuous batching (#4470, lane vllm), each step's K and V written to and read from the SlotKV. Each layer runs in micro-batches of rows (`--micro-batch`) written into the layer's
full-batch tables, which go to the commit call as the layer ends and are dropped.

| Piece | Here now | Replaced by |
|---|---|---|
| Weights | ALLP-IFACE's generator and weights stream (`verity_torch_loop.weights`, CMP-95 item 1), every linear generated once into the device's memory (`driver.Params`, over `--workers` processes); the record carries the registration digest under the test salt key for small shapes | the prover's salt key's digest |
| Store and decode call (M4) | `--weights staged`: `store.py`'s P2 v4 (`p2-16448/v4`) rows over the weights stream, the P2 reference decode on the CPU, or kernels' `Stage.decode` with `--stage-lib`, a layer a step | memory-accounting's layout and decode call (CMP-95 item 2), MEM-69's store on the card |
| W8A8 linears | `ops.s8_linear`: `torch._int_mm` on CUDA (an exact f64 matmul on the CPU), then `bf16((f32(acc) · sx) · sw)` | kernels' KP2-S8 |
| Row ops | `ops.py`: the plan's row functions; sums sequential (on the host), division and sqrt correctly rounded through f64 | circuits' row-op kernels |
| Slots and KV cache | `slots.py`: `SlotKV` admits requests FIFO into the lowest free sequence slots, runs each at its own position, frees and reuses a slot when its request finishes, and holds each slot's K and V rows (the one KV head, the context's positions) as the step's bf16 bits, written and read a micro-batch of rows at a time; `check_slots.py` checks it at the frozen KV shape on a device | stays |
| Attention | `attention.py`: on CUDA kernels' `attn_ref` (#4466, `benchmarks/torch_gpu/attn_ref`; its cubin from `build.sh`, `$ATTN_REF_CUBIN`) for QK^T and PV and `softmax_t` (the MUFU ex2 table in torch integer ops); elsewhere the Definitions' numpy kernels | kernels' fast attention (`attn_bf16`) |
| Commit | `commit.py`: kernels' per-slot tree (`--commit slots`, #4469, `benchmarks/torch_gpu/commit_slots`: row/v2 leaves, step → slot → unit → rows), a group at a time on a side stream, salt keys fresh a step; or the by-table `verity_vllm.commit.row_tables` (`reference`, `host`, `device`), which keeps every step's tables | — |
| Opening | `opening.py`: `--open K` slots drawn by a fixture coin, each replayed alone on the resident weights under its steps' salt keys and its own nonce, its slot roots compared with the committed ones; each drawn unit's one commit unit opened against the step's root (`slots.verify_unit`), its rows the unit's outputs, and the reference's slot root compared with the engine's | PRF-170's plan's live coins |
| Replay | `e2e_h200/units.py`: one family a unit kind of the map, each through compositions of `replay.py`'s templates (main's Definitions where they exist) until the replay lane adds the map's; a row's attention reads its slot's K and V rows at the steps that wrote them; a fixture coin draws K a family | the replay lane's templates for the map; PRF-170's plan through `replay.draw` |
| PoUW | `--subcircuit e2e_h200.percall:kraqwxsc` (`percall.py`, a `units.Subcircuit` whose ``call`` the driver runs on each served linear call): f = 1, every unit of every call through J10-R's `PouwH1Tile_v6` on kernels' sm_90a kraqwxsc build (#4538; `percall:kraq` is #4540's), z the call's served s32 sums, the receipts and the sums committed under the step's salt key, sampled units held to the exact fallback (`j10r_ref.so`); `percall:python` is its host twin. Credit is not computed: its two blockers are named in the record (`percall.CREDIT`) | h1zy's J10-R lane Definitions (#4654) and Credit.window over the tile stratum (CMP-127); window_salt's live salt and the real roots (`percall.STANDS_IN`) |
| Network | `e2e_h200/engine.py`: the loop as network's demo endpoint's engine (`demo_endpoint.py --engine e2e_h200.engine:frozen`, NET-77): each sequence its own slot and positions, no EOS (8 prompt ids, 24 generated) | — |
| Bundle and record | `record.py`: provisional v0 schemas | IFACE-B's `verity/drawn-unit-bundle/v1` and `verity/block-record/v1` (CMP-95 items 5 and 6, PR #4454) |
| Legs | `legs.py`: certifier digests as a fixture; PoUS spot reads and wipe and reload on the M4 path's P2 store; bandwidth not run (M4); integrity is the replay and the opening | MEM-67/69, CMP-97, NET-72, C-Flock `--zk` and Lean |

`replay.py` and `cpu_step.py` are the replay lane's (#4460): the CPU replay of drawn units in the 16-bit forms, and an
independent CPU decode step; the suite holds the driver's rows equal to `cpu_step`'s.

## Commands

~~~text
uv run python benchmarks/e2e_h200/run.py --out /tmp/e2e                         # the small shape, CPU, resident, slots commit
uv run python benchmarks/e2e_h200/run.py --out /tmp/e2e --batch 8 --requests 24 --prompt-bytes 8 --generate 24   # continuous batching
uv run python benchmarks/e2e_h200/run.py --out /tmp/e2e --tamper l1.wup         # exits 3: the opened slots' roots don't replay
uv run python benchmarks/e2e_h200/run.py --out /tmp/e2e --shape tiny --weights staged --store p2-encode --workers 8 --commit host
uv run python benchmarks/e2e_h200/run.py --out OUT --shape h200 --seed demo --device cuda --batch 4096 --steps 32 \
    --prompt-bytes 8 --micro-batch 1024 --workers "$(nproc)" --open 2 --draw 1     # the skeleton at batch 4,096 (ATTN_REF_CUBIN set)
bash benchmarks/e2e_h200/percall_job.sh                                         # the skeleton at f = 1 on the H200 (its header: the research run)
PYTHONPATH=benchmarks/e2e_h200 python benchmarks/network_traces/demo_endpoint.py --out DIR --engine e2e_h200.engine:tiny
uv run python benchmarks/e2e_h200/run.py --out OUT --shape h200 --seed demo --device cuda \
    --weights-file W --weights-manifest M --workers "$(nproc)"                   # serve the loaded W (NET-76's sink), checked first
E2E_H200_SEED=demo E2E_H200_WEIGHTS_FILE=W E2E_H200_WEIGHTS_MANIFEST=M PYTHONPATH=benchmarks/e2e_h200 \
    python benchmarks/network_traces/demo_endpoint.py --out DIR --engine e2e_h200.engine:frozen   # the endpoint on it
~~~

Exit 0: every drawn unit replays to its committed word, every opened slot to its committed root and every subcircuit's own
check holds; 3: one doesn't; 4: a leg's fixture check fails; 5: ``--weights-file`` isn't its manifest's W (a tensor's
sha256, its size or its model), refused before the first step.
