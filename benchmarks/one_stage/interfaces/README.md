# E2E demo interfaces

The one-stage audit's parties hand each other two documents, and this directory defines both. The replay bundle
(`verity_e2e_demo/replay_bundle.py`) is everything the prover behind the ZK server reads to prove an execution
phase's sampled proof blocks: their coordinates, templates and every row's opening, with the private rows and salts in
a payload beside it. The block record (`verity_e2e_demo/block_record.py`) is what each subprotocol writes for one
server block, with the Lean check's verdict on it and the error over the monitoring period. Each module's docstring is
its format, each has a JSON schema under `verity_e2e_demo/schemas/`, and each one's `check` recomputes every derived
field. Not a package: run the modules with this directory on `PYTHONPATH` (the suite's `pyproject.toml` sets it).
[`DESIGN.md`](DESIGN.md) says how the bundle's step commitments are built.

## Try it

~~~sh
PYTHONPATH=benchmarks/one_stage/interfaces uv run python -m verity_e2e_demo.replay_bundle check \
  tests/benchmarks/one_stage/interfaces/fixtures/drawn-unit-bundle.json \
  --registration tests/benchmarks/one_stage/interfaces/fixtures/drawn-unit-registration.json
~~~

It checks the fixture bundle from its bytes alone, replaying each proof block's Definition on its input rows, and
prints `{"bad": {}, "ok": true, "units": 4}`.

## The fixtures

`build` writes a bundle from a window and the prover's step records, given where each port's row is in the step layout
(`locate`), a proof block's coordinates (`coords`) and its registered reads (`reads`). The fixture,
`tests/benchmarks/one_stage/interfaces/fixtures/drawn-unit-bundle.{json,bin}` with its window's docket, is four sampled proof blocks of a W8A8 MLP's window
(two requests, one step, both linears, width 128: 512 proof blocks, registered `subset:4`), its weights registered as core's
registered values and its step committed under test salt keys, its target the RTX PRO 6000's. The suite pins its
SHA-512s. `check` recomputes every string, path and domain and replays each proof block's Definition on its input rows. The
window itself, which `benchmarks/one_stage/prove_bundle.sh` proves the bundle against, is the registered fixture
`tests/benchmarks/one_stage/fixtures/bundle-window` (its `BUNDLE_WINDOW`).

The block record's fixtures check the same way:

    python -m verity_e2e_demo.block_record check tests/benchmarks/one_stage/interfaces/fixtures/block-record-r2-period0.json
    python -m verity_e2e_demo.block_record from-runs --r2 R2/reports.jsonl --r1 R1/reports.jsonl --wipe WIPE/results.json --out DIR

The fixtures are records of real runs, in the record's form of today: a bandwidth entry names the term its error leaves
out, which those runs' answers did not, so `from-runs` refuses a report of theirs that has a bandwidth row.
`block-record-r2-period0` is one period of an all-protocols run: five
subprotocols accept, and a wipe is carried but not composed, since it ran on another server. It has no Lean answer, so the
verdict is undecided, and the error sum is at least SPOT's 1/2 from its 2-unit sample. `block-record-b71a-w30`
carries its window's Lean `onsite check` answer, which accepts. `block-record-r1-window10` is one window with
SPOT at δ = 1/100 and the wipe on the same server, the other subprotocols absent.
`integrity-profile-stratified` is the profile of a served run's SPOT answer, which samples each template's proof blocks as a
stratum of their own (all 160 of one, 2 of another's 4): its bound is 2 of 164 at δ = 1/100, where a uniform sample of
162 of 164 would say 1.
