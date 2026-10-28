# The H200 example

It shows the remote protocol's sampled-proofs half (`examples/README.md`) at a frozen shape on one H200: a step's every
value committed, proof blocks drawn by one live coin, each proved by C-Flock `--zk` and checked by the Lean verifier
(`phases.sh`, the run of record). It claims nothing else.

The GPU serving loop of the one-H200 demo: steps of the frozen MQA shape (`shape.FROZEN`, `served_weights`'s `H200`) on the
plain INT8 weights, loaded once into the card's memory and resident across steps (`--weights staged` decodes them from the
store into staging instead), every op's output committed through the per-step commit call, drawn slots
opened by replaying their requests and drawn proof blocks replayed bit for bit against the Definitions, each subprotocol run against the
step or named, and a replay bundle and block record written. `verity example h200 --out OUT [...]` runs it
(`run.py --help`): on the CPU at the small shapes, or on an H200 at the frozen one (`--shape h200 --device cuda`).

Around sampled proofs, `e2e_h200/legs.py` runs the other subprotocols as a skeleton on fixture coins, not as runs, and
nothing claims them: PoUW is per-call noise's tiles as a `--subcircuit`, PoUS is spot reads and the memory wipe and reload
on the staged path's store, network certification's digests are a fixture, and the bandwidth audit isn't run. `interfaces/` holds the
interfaces between its subprotocols, the drawn-unit replay bundle and the block record, a suite of its own (its README).

It is an example, not an integration: it composes pieces from several parts of the tree (the weights' store and decode
call, the GEMMs, attention and commit call, circuits' Definitions, the subprotocols) for one recorded run. When the real pieces land
they replace the fixtures here, one module each, and the record's `stands_in_for` fields go to `null`.

## Workload and placement

The workload is the frozen MQA shape on seeded INT8 weights (`e2e_h200/served_weights.py`), with `--shape
tiny|small|slice` for CPU runs; it isn't an integration yet (`example.toml`: `workload = "none"`). Every party runs in the
one run on one GPU node (an H200 node for `percall_job.sh`): the driver and its commit call on the prover's side, and the
replay and the opening on the verifier's, on fixture coins. Where the run of record's parties ran, how far the example runs and
the runs behind it are `example.toml`'s (`examples/README.md`).

## What runs, and what stands in

The units are circuits' FAMILIES map's (`e2e_h200/plan.py`, `shape.Forms`). The residual stream and every
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
continuous batching, each step's K and V written to and read from the SlotKV. Each layer runs in micro-batches of rows (`--micro-batch`) written into the layer's
full-batch tables, which go to the commit call as the layer ends and are dropped.

| Piece | Here now | Replaced by |
|---|---|---|
| Weights | The seeded generator and weights stream (`e2e_h200/served_weights.py`), every linear generated once into the device's memory (`driver.Params`, over `--workers` processes); the record carries the registration digest under the test salt key for small shapes | the prover's salt key's digest |
| Weights registration | `weights_commit.py`: every tensor committed as the registered value the served Program reads (`registered.domain` under its digest, its widths from the Program's parameter types, never its gate lists), streamed a chunk a worker under the operating system's salts into a 0600 witness; `open` opens a ScaledMm proof block's `wrow`/`sw` reads against it (`verify_read`) | — |
| Store and decode call | `--weights staged`: `store.py`'s P2 v4 (`p2-16448/v4`) rows over the weights stream, the P2 reference decode on the CPU, or the staging decode on the GPU (`benchmarks/torch_gpu/p2_stage`'s `Store.decode_into` over `libp2v1.so`) with `--stage-lib`, a layer a step | memory-accounting's layout and decode call, and its store on the card |
| W8A8 linears | `ops.s8_linear`: `torch._int_mm` on CUDA (an exact f64 matmul on the CPU), then `bf16((f32(acc) · sx) · sw)` | kernels' s8 GEMM |
| Row ops | `ops.py`: the plan's row functions; sums sequential (on the host), division and sqrt correctly rounded through f64 | circuits' row-op kernels |
| Slots and KV cache | `slots.py`: `SlotKV` admits requests FIFO into the lowest free sequence slots, runs each at its own position, frees and reuses a slot when its request finishes, and holds each slot's K and V rows (the one KV head, the context's positions) as the step's bf16 bits, written and read a micro-batch of rows at a time; `check_slots.py` checks it at the frozen KV shape on a device | stays |
| Attention | `attention.py`: on CUDA `attn_bf16`'s reference kernels (`benchmarks/torch_gpu/attn_bf16`'s `attn.py` with `path="ref"`; its cubin from `build.sh`, `$ATTN_BF16_CUBIN`) for QK^T and PV and `softmax_t` (the MUFU ex2 table in torch integer ops); elsewhere the Definitions' numpy kernels | kernels' fast attention (`attn_bf16`) |
| Commit | `commit.py`: kernels' per-slot tree (`--commit slots`, `benchmarks/torch_gpu/commit_slots`: row/v2 leaves, step → slot → unit → rows), a group at a time on a side stream, salt keys fresh a step; or the by-table `verity_vllm.commit.row_tables` (`reference`, `host`, `device`), which keeps every step's tables | — |
| Opening | `opening.py`: `--open K` slots drawn by a fixture coin, each replayed alone on the resident weights under its steps' salt keys and its own nonce, its slot roots compared with the committed ones; each drawn unit's one commit unit opened against the step's root (`slots.verify_unit`), its rows the unit's outputs, and the reference's slot root compared with the engine's | the sampling plan's live coins |
| Window writer | `e2e_h200/window.py`: FAMILIES' shared-row layout (`shared_layout`: 1 KiB rows, 21,857 a slot at the frozen shape, circuits' writers). Each proof block's input ports as pieces (`Window.locate`): committed rows' bytes named by step, slot, row and bytes (a linear's x over its quantizer's R rows; attention's K and V at the steps and rows that wrote them), registered rows and run inputs. `Window.inputs` assembles them from a slot's rows; the same calls work on `families/v1` | the run's commit under the shared layout, and the drawn-block bundle over it |
| Replay | `e2e_h200/units.py`: one family a unit kind of the map, each through compositions of `replay.py`'s templates (main's Definitions where they exist) until the map's own exist; a row's attention reads its slot's K and V rows at the steps that wrote them; a fixture coin draws K a family | templates for the whole map; the sampling plan through `replay.draw` |
| PoUW | `--subcircuit e2e_h200.percall:kraqwxsc` (`percall.py`, a `units.Subcircuit` whose ``call`` the driver runs on each served linear call): f = 1, every unit of every call through J10-R's `PouwH1Tile_v6` on kernels' sm_90a kraqwxsc build (`percall:kraq` is the earlier build), z the call's served s32 sums, the receipts and the sums committed under the step's salt key, sampled units held to the exact fallback (`j10r_ref.so`); `percall:python` is its host twin. Credit is not computed: its two blockers are named in the record (`percall.CREDIT`) | J10-R's lane Definitions and Credit.window over the tile stratum; window_salt's live salt and the real roots (`percall.STANDS_IN`) |
| Network | `e2e_h200/engine.py`: the loop as network's demo endpoint's engine (`demo_endpoint.py --engine e2e_h200.engine:frozen`): each sequence its own slot and positions, no EOS (8 prompt ids, 24 generated) | — |
| Bundle and record | `record.py`: provisional v0 schemas | `verity/drawn-unit-bundle/v1` and `verity/block-record/v1` (`examples/h200/interfaces`) |
| Subprotocols | `legs.py`: certifier digests as a fixture; PoUS spot reads and wipe and reload on the staged path's P2 store; bandwidth not run; sampled proofs are the replay and the opening | PoUS's and the memory wipe's runs, the bandwidth audit, network certification's signed records, C-Flock `--zk` and the Lean verifier |

`e2e_h200/served_weights.py` is the demo's seeded weights and their registration root, for any decoder
shape (`H200`, Phase 1's CPU model, `TINY`): a SHA-512 counter generator from (seed, tensor, row) to every tensor, the weights
stream's layout (`manifest`), and each tensor as a registered value of core's for the program that reads it, its root computed
in parallel chunks (`root`). No torch. `python -m e2e_h200.served_weights manifest|write|root|fixture`; the fixture is
`tests/fixtures/weights-v1.json`. The H200 manifest under `DEMO_SEED`, with each tensor's SHA-256, and its registration digest
under the test salt key for the placeholder program `TEST_PROGRAM` are in the evidence store
(the `manifest` command below writes them again).

`replay.py` and `cpu_step.py` are the CPU replay of drawn proof blocks in the 16-bit forms, and an
independent CPU decode step; the suite holds the driver's rows equal to `cpu_step`'s.

## Commands

~~~text
uv run python examples/h200/run.py --out /tmp/e2e                         # the small shape, CPU, resident, slots commit
uv run python examples/h200/run.py --out /tmp/e2e --batch 8 --requests 24 --prompt-bytes 8 --generate 24   # continuous batching
uv run python examples/h200/run.py --out /tmp/e2e --tamper l1.wup         # exits 3: the opened slots' roots don't replay
uv run python examples/h200/run.py --out /tmp/e2e --shape tiny --weights staged --store p2-encode --workers 8 --commit host
uv run python examples/h200/run.py --out OUT --shape h200 --seed demo --device cuda --batch 4096 --steps 32 \
    --prompt-bytes 8 --micro-batch 1024 --workers "$(nproc)" --open 2 --draw 1     # the skeleton at batch 4,096 (ATTN_BF16_CUBIN set)
bash examples/h200/percall_job.sh                                         # the skeleton at f = 1 on the H200 (its header: how it's launched)
PYTHONPATH=examples/h200 python benchmarks/network_traces/demo_endpoint.py --out DIR --engine e2e_h200.engine:tiny
uv run python examples/h200/run.py --out OUT --shape h200 --seed demo --device cuda \
    --weights-file W --weights-manifest M --workers "$(nproc)"                   # serve the loaded W (the file network's sink), checked first
E2E_H200_SEED=demo E2E_H200_WEIGHTS_FILE=W E2E_H200_WEIGHTS_MANIFEST=M PYTHONPATH=examples/h200 \
    python benchmarks/network_traces/demo_endpoint.py --out DIR --engine e2e_h200.engine:frozen   # the endpoint on it
PYTHONPATH=examples/h200 python -m e2e_h200.weights_commit commit --shape h200 --seed demo \
    --program DIGEST --state STATE --out OUT/registration.json        # the H200 weights registration (STATE private)
PYTHONPATH=examples/h200 python -m e2e_h200.weights_commit open --state STATE --seed demo \
    --op l0.wup --coordinate N                                        # a ScaledMm proof block's reads opened: OPEN or REFUSED
PYTHONPATH=examples/h200 python -m e2e_h200.served_weights manifest --shape h200 --seed text:verity/cmp92/h200/weights/v1 \
    --digests --out M          # the served weights' manifest under DEMO_SEED, a frozen string (also write|root|fixture)
~~~

Exit 0: every drawn proof block replays to its committed word, every opened slot to its committed root and every subcircuit's own
check holds; 3: one doesn't; 4: a subprotocol's fixture check fails; 5: ``--weights-file`` isn't its manifest's W (a tensor's
sha256, its size or its model), refused before the first step.
