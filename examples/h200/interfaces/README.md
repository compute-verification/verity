# E2E demo interfaces

The interfaces between the six subprotocols of the one-H200 E2E demo: the drawn-unit replay bundle and the block record.
Not a package: run the modules with this directory on `PYTHONPATH` (the suite's `pyproject.toml` sets it).

## The drawn-unit replay bundle (`verity/drawn-unit-bundle/v2`)

`verity_e2e_demo/replay_bundle.py`, schema `verity_e2e_demo/schemas/drawn-unit-bundle.v2.schema.json`. What the prover
behind the firewall reads to prove a window's drawn units: per unit its coordinates (block, step, slot, layer, op, unit),
its template id and binding, each row's opening (input and output rows against the step commitment, weight rows against
their registered value's root, every row against its window port's registered root) and the C-Flock `--zk` session's
member, subset draw and `frame_v3` header entry. A JSON document holds the public part; the private rows and salts are a
binary payload beside it (`<name>.bin`, mode 0600), each named by `[offset, length]`. The payload is each unit's rows in
order, each row's u16 words little-endian then its 192-byte salt.

The step commitment is kernels' commit call (`verity/commit-slots/v1`): a step tree over slot trees over
unit trees over rows, only each unit's outputs committed. An output row opens in its unit's own commitment unit. An input
row opens where its producer committed it, and names the producer's coordinates (the fixture's `x` and `sx` open at
their layer's input quantizer). The bundle adds a block tree over each server-time block's step roots.
The bundle builds its step commitments with kernels' CPU reference, `benchmarks/torch_gpu/commit_slots/slots.py`,
imported from the tree. The suite pins the SHA-512 of their fixture, `commit-slots-v1.json`, which the reference must
reproduce; its roots and digests are those of the commit `REFERENCE` pins.

`target` names the card's binding of the templates' dot: `GEMM_SM90_S8` for the H200's INT8, `GEMM_HOPPER` for its BF16,
`GEMM_SM120_S8` for the RTX PRO 6000. `check` refuses a template that binds another dot. On `main`, `GEMM_SM90_S8`
resolves to the same s8 dot as `GEMM_SM120_S8`, so only the field tells the two cards apart. A `live` draw carries
`draw.receipt`, the challenger's live-coin receipt; `check` and `build` refuse one without it. Against the window's
registration, `check` refuses a draw whose law isn't the registered one (fixed before any coin, so the prover can't
pick it), a subset draw of other than its k units, and a `draw.record` that isn't a draw of the bundle's units under
that law over the registered population (Lean's `flock-verify draw --registration F` refuses to draw one).

    python -m verity_e2e_demo.replay_bundle check tests/fixtures/drawn-unit-bundle.json [--registration tests/fixtures/drawn-unit-registration.json]

`build` writes a bundle from a window and the prover's step records, given where each port's row is in the step layout
(`locate`), a unit's coordinates (`coords`) and its registered reads (`reads`). The fixture,
`tests/fixtures/drawn-unit-bundle.{json,bin}` with its window's registration, is four drawn units of a W8A8 MLP's window
(two requests, one step, both linears, width 128: 512 units, registered `subset:4`), its weights registered as core's
registered values and its step committed under test salt keys, its target the RTX PRO 6000's. The suite pins its
SHA-512s. `check` recomputes every string, path and domain and replays each unit's Definition on its input rows. The
window itself, which `benchmarks/one_stage/prove_bundle.sh` proves the bundle against, is in the evidence store
(its `BUNDLE_WINDOW`).

## The block record (`verity/block-record/v1`)

`verity_e2e_demo/block_record.py`, schema `verity_e2e_demo/schemas/block-record.v1.schema.json`. What each subprotocol writes
for one server-time block and what the composed verdict reads. It extends the all-protocols runner's report
(`verity/all-protocols-period/v1` and `-window/v1`) with:

- the block's server and steps, `server_time: [{server, steps: [lo, hi)}]`;
- the subprotocols, keyed `integrity` (sampled proofs), `pouw`, `pous`, `bandwidth`, `certifier` and `wipe`. Each is
  `null` when the block ran no such audit, else its protocol, the core answer, its certificate (integrity bound, W_cert,
  S_cert, Q_cert, K, the wipe's foreign_bits at its reads `k` and ratio `r`), its error and the certificate at δ = 1/100
  and 1/10^6, its run ids, and `source: real|fixture`;
- the composed verdict and the error of `Accounting.Guarantees.JointHorizonLive` over the block: the accepted
  subprotocols' errors summed;
- the monitoring period: week, month and year over a census deployment's machines (a parameter), with JointHorizonLive's
  union bound over that many blocks and the per-block error each δ needs.

`check` recomputes every derived field from the subprotocols' entries.

    python -m verity_e2e_demo.block_record check tests/fixtures/block-record-r2-period0.json
    python -m verity_e2e_demo.block_record from-runs --r2 R2/reports.jsonl --r1 R1/reports.jsonl --wipe WIPE/results.json --out DIR

The fixtures are records of real runs. `block-record-r2-period0` is one period of an all-protocols run: five
subprotocols accept, and a wipe is carried but not composed, since it ran on another server. So the verdict is pending,
and the error sum is at least sampled proofs' 1/2 from its 2-unit draw. `block-record-r1-window10` is one window with
sampled proofs at δ = 1/100 and the wipe on the same server, the other subprotocols absent.

## Demo A's report

`verity_e2e_demo/report.py` renders block records and a measurements file (`verity/demo-a-measurements/v1`) through
`demo_a.template`; its module docstring is the contract. A report with an unfilled number is refused; only a draft
(`render(..., draft=True)`, never the command) shows `⟨NAME⟩`. Part 3 is circuit size in AND gates, with no FLOP figure or
price; its H and B are one recorded run of proofs' planner (`benchmarks/sampling_plan/demo_a.py --spans week`), filled
together with `PLANNER`. A measurements file with `sections` or `proved` renders the H200 report (`m3.template`): every row is one number with
its run, its own verdict line quoted and its target, whose misses are listed as visible failures. `proved` is the
drawn-block proving run's VERDICT and PROVEBUNDLE lines verbatim: the first line is the fallback's until every drawn
H200 proof block is proved in C-Flock and accepted by Lean. Per-call noise's serving overhead is one line,
`serve_noise_line`: the seconds it adds a step with both ratios, never γ_step alone, ending the section marked
`"serve_noise": true`; a row stating it on its own is refused. The rendered report goes to the
evidence store (kind `demo-a-report/v1`), never Git.

    python -m verity_e2e_demo.report [RECORD.json ...] --measurements M.json --out FILE
