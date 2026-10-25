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
