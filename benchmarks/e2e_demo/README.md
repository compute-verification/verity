# E2E demo interfaces

The interfaces between the legs of the one-H200 E2E demo (P-CMP-1, CMP-95 items 5 and 6). Not a package: run the
modules with this directory on `PYTHONPATH` (the suite's `pyproject.toml` sets it, with `integrations/torch_loop`).

## The drawn-unit replay bundle (`verity/drawn-unit-bundle/v2`)

`verity_e2e_demo/replay_bundle.py`, schema `verity_e2e_demo/schemas/drawn-unit-bundle.v2.schema.json`. What the prover
behind the firewall reads to prove a window's drawn units: per unit its coordinates (block, step, slot, layer, op, unit),
its template id and binding, each row's opening (input and output rows against the step commitment, weight rows against
their registered value's root, every row against its window port's registered root) and the C-Flock `--zk` session's
member, subset draw and `frame_v3` header entry. A JSON document holds the public part; the private rows and salts are a
binary payload beside it (`<name>.bin`, mode 0600), each named by `[offset, length]`. The payload is each unit's rows in
order, each row's u16 words little-endian then its 192-byte salt.

The step commitment is kernels' commit call (CMP-95 item 3, `verity/commit-slots/v1`): a step tree over slot trees over
unit trees over rows, only each unit's outputs committed. An output row opens in its unit's own commitment unit. An input
row opens where its producer committed it, and names the producer's coordinates (the torch loop's `x` and `sx` open at
their layer's input quantizer). The bundle adds a block tree over each server-time block's step roots.
`verity_e2e_demo/commit_slots.py` is kernels' CPU reference vendored at the commit its header names (#4469). Once their
commit_slots package is on `main`, import it and delete the copy. The suite pins the SHA-512 of their fixture,
`commit-slots-v1.json` at the commit `REFERENCE` pins, which the copy must reproduce.

`target` names the card's binding of the templates' dot: `GEMM_SM90_S8` for the H200's INT8, `GEMM_HOPPER` for its BF16,
`GEMM_SM120_S8` for the RTX PRO 6000. `check` refuses a template that binds another dot. On `main`, `GEMM_SM90_S8`
resolves to the same s8 dot as `GEMM_SM120_S8`, so only the field tells the two cards apart. A `live` draw carries
`draw.receipt`, the challenger's live-coin receipt; `check` and `build` refuse one without it. Against the window's
registration, `check` refuses a draw whose law isn't the registered one (fixed before any coin, so the prover can't
pick it), a subset draw of other than its k units, and a `draw.record` that isn't a draw of the bundle's units under
that law over the registered population (Lean's `flock-verify draw --registration F` refuses to draw one).

    python -m verity_e2e_demo.replay_bundle fixture --out DIR --scratch "$(mktemp -d)"
    python -m verity_e2e_demo.replay_bundle check DIR/drawn-unit-bundle.json [--registration SCRATCH/registration.json]

The fixture (`fixture` writes it as `drawn-unit-bundle.{json,bin}`) is the torch loop's window (two requests, one step,
both linears, width 128: 512 units, registered `subset:4`) with four drawn units, its weights registered as core's registered values and its step
committed under test salt keys, its target the RTX PRO 6000's. It is built in seconds, so it isn't tracked: the suite
builds it and pins its SHA-512s. `check` recomputes every string, path and domain and replays each unit's Definition on its input rows.

## The block record (`verity/block-record/v1`)

`verity_e2e_demo/block_record.py`, schema `verity_e2e_demo/schemas/block-record.v1.schema.json`. What each leg writes
for one server-time block and what the composed verdict reads. It extends the all-protocols runner's report
(`verity/all-protocols-period/v1` and `-window/v1`) with:

- the block's server and steps, `server_time: [{server, steps: [lo, hi)}]`;
- the legs integrity, pouw, pous, bandwidth, certifier and wipe. Each is `null` when the block ran no such audit, else
  its protocol, the core answer, its certificate (integrity bound, W_cert, S_cert, Q_cert, K, f·M), its error and the
  certificate at δ = 1/100 and 1/10^6, its run ids, and `source: real|fixture`;
- the composed verdict and the error of `Accounting.Guarantees.JointHorizon` over the block: the accepted legs' errors
  summed. The statement is not yet proved and not on `main` (lane fc8d);
- the monitoring period: week, month and year over a census deployment's machines (a parameter), with JointHorizon's
  union bound over that many blocks and the per-block error each δ needs.

`check` recomputes every derived field from the legs.

    python -m verity_e2e_demo.block_record check tests/fixtures/block-record-r2-period0.json
    python -m verity_e2e_demo.block_record tonight --r2 R2/reports.jsonl --r1 R1/reports.jsonl --wipe WIPE/results.json --out DIR

The fixtures are real. `block-record-r2-period0` is period 0 of all-protocols run r20261025-210358-993e: five legs
accept, and the wipe of run r20261025-224222-91e2 is carried but not composed, since it ran on another server. So the
verdict is pending, and the error sum is at least integrity's 1/2 from its 2-unit draw. `block-record-r1-window10` is
window 10 of r20261025-203534-186b: integrity at δ = 1/100 and the wipe on the same server, with the other legs absent.
