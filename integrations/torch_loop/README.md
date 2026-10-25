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
- `tests/`: the loop's outputs are the Definition's, unit by unit; core's registration check passes the written window and
  refuses a port under another domain; the verifier's window from the records (`interface.of_records`) is the loop's; its
  compiler compiles its committed schedule to the circuit it registers and refuses advice that is not a schedule; its unit
  composes into C-Flock's unit statement (slow); a tampering loop's outputs are not its Definition's; the loop loads no
  vLLM.
- End to end: `benchmarks/prove_path/run.sh` with `PP_TORCH_LOOP=1` audits the honest and the tampered window from their
  records alone, against the registered circuit (`records_audit.py audit --circuit SHA512`) and against the compiler's
  output (`--opening`, `--compilers`): live coins, C-Flock `--zk` behind the Lean firewall, the Lean verifier of record,
  each tampered window refused.

## The demo of record

Run r20261025-170501-8a09 (vy-nebius-1, 25 Oct 17:05Z, rc 0), at a9ab779742: c269f7c5 with a5_coins' tunnel wait
(`connect_when_listening`). Its arguments to `run.sh` were `PP_TORCH_LOOP=1 PP_ONLY=forged PP_TORCH_LOOP_CAP=256
PP_LAW=subset:2 PP_TORCH_LOOP_SHAPE="--requests 3 --steps 1"`. The coins came through infra's gate from vy-challenger-1
(`PP_CHALLENGER=gate:research@204.12.178.84`, `PP_PEER=vy-nebius-1`): its pin was 169118abc1, on main's first-parent line,
and its key record is art:d12f85becc, main's `CHALLENGER_KEY_PIN`.

The window is one request for one decode step. That is N = 256 units: two linears of 128 output coordinates each. The
256 units form one member, so one C-Flock session (J = 1 table, hidden outputs) is the whole audit. Each arm is a run of
its own, with its own root and its own gate session:

| arm | session | units drawn of 256 | verdict |
|---|---|---|---|
| honest | s20261025T170914Z-8c76 | 65, 92 | accepted: Lean's draw, Lean's `verify --zk` and M0's verifier; 19/19 negatives refused |
| tampered | s20261025T171127Z-c854 | 101, 228 | refused: the Lean firewall stopped the prover, so no proof left it; 18/18 negatives refused |
| compiled | s20261025T171342Z-9625 | 144, 246 | accepted, its circuit compiled at the registered commit; 26/26 negatives refused |
| compiled, tampered | s20261025T171551Z-dbd6 | 170, 193 | refused, as tampered; 25/25 negatives refused |

The firewall's refusal of each tampered arm reads "FIREWALL-REFUSED: circuit: the constraints lincheck final fail on a
rep's pads". The integrity profile's bound for each accepted arm: an accepted transcript has at most 74 wrong units of 256,
except with probability δ = 0.5.

The theorem is `Flock.Guarantees.EndToEndHidden` (`Security/Proofs/Flock/EndToEndHidden/Statement.lean`, proved as
`Flock.SecurityProofs.EndToEndHidden`, in the lock): `EndToEnd`'s statement under hidden outputs. At n = 256 and kd = 2,
for every live prover,

Pr[`flock-verify` accepts and at least K₀ of the 256 units are wrong] ≤ C(256 − K₀, 2) / C(256, 2) + ε_ks + δ_link.

At K₀ = 75 the first term is C(181, 2) / C(256, 2) = 0.4991, which is the profile's 74 at δ = 0.5. The statement's named
assumptions are hypotheses: A3 (the verifier's bytes are uniform), `live-verifier` (with A6 for the coin server's coins),
`cr/sha-512` and `ecr/sha-512`. What it doesn't cover:

- `live-verifier`, a hypothesis. It says the draw a5 serves the session (its share of the gate challenger's draw) is the
  one `flock-verify draw` printed. That premise is not proved for the served path. A3 is read here of the gate
  challenger's bytes, an auditor device under `trusted-nebius-host`.
- The checks a5 and `records_audit.py` make outside the session (R1 to R7, the receipt, the registration fixed before the
  draw), which the bound does not need.
- The chain from this bound into `Claims.Sampling.KS` (the "session verdict ⇒ Flock KS" row that
  `benchmarks/prove_path/README.md` lists as broken). The demo needs no claims-form statement.
- Zero knowledge: the session's identity labels its proof class `NON_ZK_PROOF` (zk hooks on, nothing left unmasked, stage
  m2-prototype). The demo claims soundness only.
