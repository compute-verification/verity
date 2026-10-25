# E2E demo interfaces

The interfaces between the legs of the one-H200 E2E demo (P-CMP-1, CMP-95 items 5 and 6). Not a package: run the
modules with this directory on `PYTHONPATH` (the suite's `pyproject.toml` sets it, with `integrations/torch_loop`).

## The drawn-unit replay bundle (`verity/drawn-unit-bundle/v1`)

`verity_e2e_demo/replay_bundle.py`, schema `verity_e2e_demo/schemas/drawn-unit-bundle.v1.schema.json`. What the prover
behind the firewall reads to prove a window's drawn units: per unit its coordinates (block, step, slot, layer, op, unit),
its template id and binding, each row's opening (input and output rows against the per-slot commitment subtree, weight
rows against their registered value's root, every row against its window port's registered root) and the C-Flock `--zk`
session's member, subset draw and `frame_v3` header entry. A JSON document holds the public part; the private rows and
salts are a binary payload beside it (`<name>.bin`, mode 0600), each named by `[offset, length]`. The payload is each unit's
rows in order, each row's u16 words little-endian then its 192-byte salt.

The slot tree is the fixture shape `verity/slot-tree/fixture-v0` until kernels posts the per-slot commitment (CMP-95
item 3): per server-time block, per step, per slot, a frame-v3-sha512 tree over the slot's committed rows; a step's tree
over its slots' roots; a block's tree over its steps' roots.

    python -m verity_e2e_demo.replay_bundle check tests/fixtures/drawn-unit-bundle.json [--registration REG.json]
    python -m verity_e2e_demo.replay_bundle fixture --out tests/fixtures --scratch "$(mktemp -d)"

The fixture `tests/fixtures/drawn-unit-bundle.{json,bin}` is the torch loop's window (two requests, one step, both
linears, width 128: 512 units) with four drawn units, its weights registered as core's registered values under a test
salt key. `check` recomputes every string, path and domain and replays each unit's Definition on its input rows.

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
