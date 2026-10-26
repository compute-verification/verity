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
- Committed activations (CMP-91): each window commits its `x` and `sx` rows, one a (request, step, layer), under domains
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
  `verity_torch_loop/pouw_audit.py` audits it from its records (`records_audit.py`'s audit) and answers it with
  `core_check` and Lean's `pouw-check`. Its gaps are in `pouw.py`'s docstring.
- `tests/`: the loop's outputs are the Definition's, unit by unit; its H1 tiles carry its linears and core's PoUW check
  registers its PoUW window; core's registration check passes the written window and
  refuses a port under another domain; the verifier's window from the records (`interface.of_records`) is the loop's; its
  compiler compiles its committed schedule to the circuit it registers and refuses advice that is not a schedule; its unit
  composes into C-Flock's unit statement (slow); a tampering loop's outputs are not its Definition's; the loop loads no
  vLLM. The checkpoint role's files are the served weights bit for bit, its entry pins them and names the linears in order,
  and the gate's own code reads the loop's rows from them and digests them to the role's checkpoint digest.
- `verity_torch_loop/frozen.py` and `serve_frozen.py`: one decode step of the frozen H200 shape (CMP-92 Phase 2: width
  16,384, 256 query heads of 64 with one KV head, MLP 65,536), the first `--layers` of its 52, every matmul coordinate a unit
  on the H200's bindings (`ScaledMmS8Coordinate_v1{K=16384}` and `{K=65536}` on `GEMM_SM90_S8`, `GemmCoordinate{K=64}` on
  `GEMM_HOPPER` for QKᵀ and PV), one window per template. The row ops run in torch and are committed as inputs until
  CIR-100's Definitions land. `frozen_w.py`: W per block (core's `Stratified`) and pooled per week, month and year
  (`density_bound`), at δ = 1% and 1e-6. `run.sh` with `PP_TORCH_FROZEN=1` audits each window, honest and tampered (PRF-173).
- End to end: `benchmarks/prove_path/run.sh` with `PP_TORCH_LOOP=1` audits the honest and the tampered window from their
  records alone, against the registered circuit (`records_audit.py audit --circuit SHA512`) and against the compiler's
  output (`--opening`, `--compilers`): live coins, C-Flock `--zk` behind the Lean firewall, the Lean verifier of record,
  each tampered window refused. `PP_TORCH_LOOP_REGISTERED=1` serves the registered weights, audits against their pin
  (`records_audit.py --registered PIN`) and adds an arm serving other weights, which Lean's registered reads refuse
  (r20261025-220011-057f). `PP_TORCH_LOOP_WIRED=1` audits the wired circuit under `stratified:3` (a unit of every template),
  with an arm whose quantize units' codes are tampered.
- `verity_torch_loop/weights.py`: the demo's seeded weights and their registration root (CMP-95 interface 1), for any
  decoder shape (the H200 demo's frozen shape, Phase 1's CPU model): a SHA-512 counter generator from (seed, tensor, row)
  to every tensor, the weights stream's layout (`manifest`), and each tensor as a registered value of core's for the
  program that reads it (its digest and parameter widths, as `registered.registered_entries` derives the domains; `tok`
  one coordinate a row, (H, V), in `hm96-sha512/row/v2`), its root computed in parallel chunks (`root`). No torch.
  `python -m verity_torch_loop.weights manifest|write|root|fixture`; the fixture is `tests/fixtures/weights-v1.json`.
  The H200 demo's manifest under its seed (`DEMO_SEED`), with each tensor's SHA-256, and its registration digest under
  the test salt key for the placeholder program `TEST_PROGRAM` are
  art:a6775aeef1f6d11badf5a7a38460cff7d4d59f27e70dbc3bf8dd995fbb4c82da (run r20261026-015727-d475, 48 workers on
  vy-nebius-cpu-4: 32 s for the digests, 2 min 41 s for the root).

## The demo of record

Run r20261025-172312-cc4f (vy-nebius-1, 25 Oct 17:23Z, rc 0), at a9ab779742: c269f7c5 with a5_coins' tunnel wait
(`connect_when_listening`). Its arguments to `run.sh` were `PP_TORCH_LOOP=1 PP_ONLY=forged PP_TORCH_LOOP_CAP=256
PP_LAW=subset:64 PP_TORCH_LOOP_SHAPE="--requests 3 --steps 1"`. The coins came through infra's gate from vy-challenger-1
(`PP_CHALLENGER=gate:research@204.12.178.84`, `PP_PEER=vy-nebius-1`): its pin was 169118abc1, on main's first-parent line,
and its key record is art:d12f85becc, main's `CHALLENGER_KEY_PIN`.

The window is one request for one decode step. That is N = 256 units: two linears of 128 output coordinates each. The
256 units form one member, so one C-Flock session (J = 1 table, hidden outputs) is the whole audit, and it draws 64 of
the 256 units (q = 1/4).

**Over a monitoring period.** Serve windows like this one through any monitoring period, a week, a month or a year,
each window on its own live coins. If every window is accepted, the wrong units over the whole period total at most 18
at δ = 1%, and at most 55 at δ = 10⁻⁶. The bound adds Σ (ε_ks + δ_link) over the period's windows to δ.
- The theorem is `Flock.Guarantees.IntegrityHorizon` (`Security/Proofs/Flock/EndToEnd/Period/Statement.lean`): the total
  stays below ln(1/δ)/q.
- The total is the same for a week, a month and a year, whatever the number of windows. Only the ε_ks + δ_link sum grows
  with the number of windows, and no number for it is computed here.
- As a share of compute, the total is divided by all the units of the period's compute over a datacenter. The
  datacenter's size and the window rate are parameters (`benchmarks/census`). This demo checks one machine, and its one
  window is 256 units.

The per-window figures follow. Each arm is a run of its own, with its own root and its own gate session:

| arm | session | verdict |
|---|---|---|
| honest | s20261025T172710Z-dd8f | accepted by Lean's draw, Lean's `verify --zk` and M0's verifier; 19/19 negatives refused |
| tampered | s20261025T173015Z-41b0 | refused: the Lean firewall stopped the prover, so no proof left it; 18/18 negatives refused |
| compiled | s20261025T173320Z-e96c | accepted, its circuit compiled at the registered commit; 26/26 negatives refused |
| compiled, tampered | s20261025T173628Z-7c0e | refused, as tampered; 25/25 negatives refused |

The tampered windows flip bit 0 of every output. The firewall's refusal of each tampered arm reads "FIREWALL-REFUSED:
circuit: the constraints lincheck final fail on a rep's pads". Per window, beside the period's total, the integrity
profile's bound for each accepted arm: an accepted transcript has at most 2 wrong units of 256, except with probability
δ = 0.5. r20261025-170501-8a09 ran the
same four arms at subset:2, and they behaved the same way, but its bound is only 74 of 256.

The theorem is `Flock.Guarantees.EndToEndHidden` (`Security/Proofs/Flock/EndToEndHidden/Statement.lean`, proved as
`Flock.SecurityProofs.EndToEndHidden`, in the lock): `EndToEnd`'s statement under hidden outputs. At n = 256 and kd = 64,
for every live prover,

Pr[`flock-verify` accepts and at least K₀ of the 256 units are wrong] ≤ C(256 − K₀, 64) / C(256, 64) + ε_ks + δ_link.

The first term is 0.42 at K₀ = 3 (the profile's 2 at δ = 0.5), 0.096 at K₀ = 8, 0.0085 at K₀ = 16 and 4.9 × 10⁻⁵ at
K₀ = 32. The statement's named assumptions are hypotheses: A3 (the verifier's bytes are uniform), `live-verifier`
(with `uniform/flock-coin-server` for the coin server's coins), `cr/sha-512` and `ecr/sha-512`.

The draw on the gate's coins is `Select.select`'s. A6 (`Security/Proofs/Flock/Claims/Sampling/Select/OS.lean`,
`Properties.SampledProofs.SecurityProofs`) proves it is the theorem's subset draw on the same bytes:
- `drawUnits_subset_runOS`, with no hypotheses: the code's subset draw equals `Law.runOS f (.subset k) n`.
- `select_subset_runOS`: an accepted `Select.select` at a subset:k rule checks every unit of that draw.
- `select_subset_escape_le`: under A3 alone, Pr[accepted and a set B escapes] ≤ `(Law.subset n k).escape B`.

What it doesn't cover:

- The record's custody, which `live-verifier` still takes as a hypothesis: that the record `flock-verify` reads is the
  coin server's. A3 is read here of the gate challenger's bytes, an auditor device under `trusted-host`.
- The checks a5 and `records_audit.py` make outside the session (R1 to R7, the receipt, the registration fixed before the
  draw), which the bound does not need.
- The chain from this bound into `Claims.Sampling.KS` (the "session verdict ⇒ Flock KS" row that
  `benchmarks/prove_path/README.md` lists as broken). The demo needs no claims-form statement.
- Zero knowledge. The demo claims soundness only. The session's `proof_class` reads `NON_ZK_PROOF`, the upstream label
  that `--zk` keeps (`verity/ml/flock/verifier/PROTOCOL.md` §16.14), with zk hooks on and nothing left unmasked. Its zero
  knowledge is `Flock.Guarantees.ZeroKnowledgeHidden`, honest-verifier at each fixed coin vector. What keeps it from a
  claim: the hiding gap at hm96-sha512's pinned key is not proved small, and `avoidsMask` is a hypothesis that no check
  implies yet.

## The decoder (CMP-92)

`verity_torch_loop/transformer.py` is the demo's standard workload: a 2-layer decoder-only transformer (width 128, 2 heads
of 64, MLP 512, byte vocabulary, context 32, greedy decoding). Its static matmuls are `ScaledMmS8Coordinate_v1{K=128|512}`
units. Attention's QKᵀ and PV are `GemmCoordinate_v2{K=64, DOT=HopperBF16WgmmaDot16_v1}` units over the KV cache
(`--bf16-dot ampere`: `AmpereBF16TcDot16_v2`), computed through the Definition's kernel. PV's 32 positions are padded with
+0 to 64, since C-Flock pins a unit through a row of more than one SHA-512 block. RMSNorm, softmax, ReLU, the quantizer,
the adds and argmax run in torch, and are not units until their Definitions land (CIR-100). `serve_transformer.py` writes
its window as `serve.py` writes the MLP's, `tests/test_transformer.py` covers both, and `benchmarks/prove_path/run.sh`
with `PP_TORCH_DECODER=1` audits it. The plan, counts and wiring: `note:compute-accounting/20261025T2154Z-report-allp-xfmr`.
