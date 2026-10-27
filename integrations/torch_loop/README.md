# torch_loop

A minimal serving loop that is not vLLM: a two-layer W8A8 MLP in plain PyTorch on CPU (`verity_torch_loop/loop.py`),
serving a few requests for a few decode steps and recording every linear's output coordinates into a window of Verity's
interface (`verity.ml.windows.interface`). Its one template is `ScaledMmS8Coordinate_v1{K=128, DOT=BlackwellS8ImmaDot32_v1}`
from `verity.ml.circuits.ops.gemm`: the loop builds nothing of its own into the circuit, and imports nothing from
`integrations/vllm`. Its circuit is fixed by its shape, so it can register no compiler or advice; or it registers its
schedule as advice and its circuit as its compiler's output on it. The width is 128 because C-Flock's unit statement needs a
committed row longer than one SHA-512 block.

- `verity_torch_loop/serve.py`: the window written for the audit (`--out DIR`, `--tamper` for a loop whose outputs are not
  its model's, `--opening FILE` to commit its schedule as advice under its compiler), or the circuit's SHA-512 from the
  loop's shape and run alone (`--circuit-only`), the one the verifier registers when there is no compiler. `--run-id ID`
  names the run, which its root binds: two runs of one shape are two roots, since infra's gate opens a root's coins once.
- `verity_torch_loop/compile.py`: its registered compiler, the command the audit runs on the opened advice (the contract in
  `benchmarks/prove_path/records_audit.py`): the schedule to the circuit record, exit 2 on advice that is not a schedule.
  It registers under its name, the digest over its code (this package and `verity/`, `interface.compiler_digest`) and the
  commit that code is at; the audit runs it from its own checkout of that commit (`--table ARGV...`: the verifier's entry).
- Registered weights (`--registered`, opt-in): `serve --commit-weights OPENING --pin PIN` commits the weights once, before
  any run (`loop.commit_weights`), the pin naming checkpoint role `TORCH_LOOP_MLP_H128_W8A8`. The circuit then reads each
  unit's `wrow` and `sw` from them at public positions (`interface.Read`), so they are no free input: 1,081,344 of the
  2,162,688 input bits of the default shape stay free. With it on, the members' private bodies are shared-row.
- Committed activations: each window commits its `x` and `sx` rows, one a (request, step, layer), under domains
  binding the run root (`TorchLoop.commit_activations`). Their entries are in the registration's window, fixed before the
  draw, and their salts in the window's `activations-private.json`. No read of them is declared, so they leave the free
  count as it is.
- The wired circuit (`--wired`, opt-in, CIR-100): each linear at each token row is one `ScaledMmS8_v1{K=128, N=128}` unit,
  its outputs one 256-byte row, and each ReLU and quantize one `QuantRowS8_v1{N=128, RELU}` unit (`verity.ml.circuits.ops.quant`),
  which reads that row and writes the codes and the scale the next linear reads. Each of those ties is a per-window value
  (`TorchLoop.ties_of`) that the writer's output port and the reader's input port both read row for row (registered reads
  of output ports, #4575), committed after the run under a domain bound to the run root and named in the registration, so
  the Lean verifier opens a reader's row against the root its writer's row opens. Only each request's own row is free, so
  an altered activation is the refusal of the drawn quantize unit (`--tamper-activations`), and a row read other than its
  writer's opens no root. With `--registered` the weights are one row a linear (`commit_weights(wired=True)`), and the
  default shape's free inputs drop to 2,112 bits. A member holds one unit (`--cap 1`). The audit pins the ties from the
  registration (`records_audit.window_pins`) and keeps them out of the output class, so the draw samples their units.
- `verity_torch_loop/checkpoint.py`, `checkpoint/` and `manifests/checkpoints-int8.json`: the loop's model as the int8
  checkpoint role `TORCH_LOOP_MLP_H128_W8A8`. The auditor's gate (`benchmarks/live_challenger/models.py`) registers that role
  to sign the loop's weights root.
  - The files are the weights `Model()` serves (H = 128, seed 0) in the gate's layout: each linear's s8 codes and s_w under
    its op path, plus the embedding, in `model.safetensors`, beside `config.json`.
  - The manifest entry pins each file's size and sha256 and lists `linears`, the loop's op paths in order.
  - `python -m verity_torch_loop.checkpoint` writes the files, and `--check` checks them.
  - The gate reads this manifest only once its `MANIFEST` names it; today that is vLLM's alone.
- `verity_torch_loop/pouw.py` (`serve.py --pouw h1`): the loop's PoUW window, every linear served as split-K tiles of the
  H1 tile PoUW's verifier pins (`core_check.TILES`, `PouwH1Tile_v1` at K = 64, 4 × 8), whose exact z gives the loop's own
  output; written to `DIR/pouw` under a one-stage registration in PoUW's window form, at all-protocols' PoUW parameters.
  The tiles' A and B rows are committed before the salt as per-window values (PoUW's `operands`), which the
  registration's ``activations`` names and the tiles' ports read; `serve.py` prints their root X, which the salt is
  released against. `verity_torch_loop/pouw_audit.py` audits it from its records (`records_audit.py`'s audit) and answers it with
  `core_check` and Lean's `pouw-check`. Its gaps are in `pouw.py`'s docstring.
- `tests/`: the loop's outputs are the Definition's, unit by unit; its H1 tiles carry its linears and core's PoUW check
  registers its PoUW window; core's registration check passes the written window and
  refuses a port under another domain; the verifier's window from the records (`interface.of_records`) is the loop's; its
  compiler compiles its committed schedule to the circuit it registers and refuses advice that is not a schedule; its unit
  composes into C-Flock's unit statement (slow); a tampering loop's outputs are not its Definition's; the loop loads no
  vLLM. The checkpoint role's files are the served weights bit for bit, its entry pins them and names the linears in order,
  and the gate's own code reads the loop's rows from them and digests them to the role's checkpoint digest.
- `verity_torch_loop/frozen.py` and `serve_frozen.py`: one decode step of the frozen H200 shape (width
  16,384, 256 query heads of 64 with one KV head, MLP 65,536), the first `--layers` of its 52, every matmul coordinate a unit
  on the H200's bindings (`ScaledMmS8Coordinate_v1{K=16384}` and `{K=65536}` on `GEMM_SM90_S8`, `GemmCoordinate{K=64}` on
  `GEMM_HOPPER` for QKᵀ and PV), one window per template. The row ops run in torch and are committed as inputs until
  their Definitions land. `frozen_w.py`: W per block (core's `Stratified`) and pooled per week, month and year
  (`density_bound`), at δ = 1% and 1e-6. `run.sh` with `PP_TORCH_FROZEN=1` audits each window, honest and tampered.
- End to end: `benchmarks/prove_path/run.sh` with `PP_TORCH_LOOP=1` audits the honest and the tampered window from their
  records alone, against the registered circuit (`records_audit.py audit --circuit SHA512`) and against the compiler's
  output (`--opening`, `--compilers`): live coins, C-Flock `--zk` behind the Lean firewall, the Lean verifier of record,
  each tampered window refused. `PP_TORCH_LOOP_REGISTERED=1` serves the registered weights, audits against their pin
  (`records_audit.py --registered PIN`) and adds an arm serving other weights, which Lean's registered reads refuse
  (r20261025-220011-057f). `PP_TORCH_LOOP_WIRED=1` audits the wired circuit under `stratified:3` (a unit of every template),
  with an arm whose quantize units' codes are tampered.
- The demo's seeded weights (CMP-95 interface 1) are `benchmarks/e2e_h200/e2e_h200/served_weights.py`.

## The decoder

`verity_torch_loop/transformer.py` is the demo's standard workload: a 2-layer decoder-only transformer (width 128, 2 heads
of 64, MLP 512, byte vocabulary, context 32, greedy decoding). Its static matmuls are `ScaledMmS8Coordinate_v1{K=128|512}`
units. Attention's QKᵀ and PV are `GemmCoordinate_v2{K=64, DOT=HopperBF16WgmmaDot16_v1}` units over the KV cache
(`--bf16-dot ampere`: `AmpereBF16TcDot16_v2`), computed through the Definition's kernel. PV's 32 positions are padded with
+0 to 64, since C-Flock pins a unit through a row of more than one SHA-512 block. RMSNorm, softmax, ReLU, the quantizer,
the adds and argmax run in torch, and are not units until their Definitions land. `serve_transformer.py` writes
its window as `serve.py` writes the MLP's, `tests/test_transformer.py` covers both, and `benchmarks/prove_path/run.sh`
with `PP_TORCH_DECODER=1` audits it.
