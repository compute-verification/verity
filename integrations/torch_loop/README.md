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
- `verity_torch_loop/checkpoint.py`, `checkpoint/` and `manifests/checkpoints-int8.json`: the loop's model as the int8
  checkpoint role `TORCH_LOOP_MLP_H128_W8A8`. The auditor's gate (`benchmarks/live_challenger/models.py`) registers that role
  to sign the loop's weights root.
  - The files are the weights `Model()` serves (H = 128, seed 0) in the gate's layout: each linear's s8 codes and s_w under
    its op path, plus the embedding, in `model.safetensors`, beside `config.json`.
  - The manifest entry pins each file's size and sha256 and lists `linears`, the loop's op paths in order.
  - `python -m verity_torch_loop.checkpoint` writes the files, and `--check` checks them.
  - The gate reads this manifest only once its `MANIFEST` names it; today that is vLLM's alone.
- `tests/`: the loop's outputs are the Definition's, unit by unit; core's registration check passes the written window and
  refuses a port under another domain; the verifier's window from the records (`interface.of_records`) is the loop's; its
  compiler compiles its committed schedule to the circuit it registers and refuses advice that is not a schedule; its unit
  composes into C-Flock's unit statement (slow); a tampering loop's outputs are not its Definition's; the loop loads no
  vLLM. The checkpoint role's files are the served weights bit for bit, its entry pins them and names the linears in order,
  and the gate's own code reads the loop's rows from them and digests them to the role's checkpoint digest.
- End to end: `benchmarks/prove_path/run.sh` with `PP_TORCH_LOOP=1` audits the honest and the tampered window from their
  records alone, against the registered circuit (`records_audit.py audit --circuit SHA512`) and against the compiler's
  output (`--opening`, `--compilers`): live coins, C-Flock `--zk` behind the Lean firewall, the Lean verifier of record,
  each tampered window refused.
