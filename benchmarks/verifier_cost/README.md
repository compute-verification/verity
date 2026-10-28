# verifier_cost

A verifier of record's time against the size of the description it reads and against the size of the circuit that
description unrolls to. The rule it measures (AGENTS.md "Design rules"): a verifier's cost is
O(circuit description), never O(unrolled circuit).

The workload is Serve_v1 (`benchmarks/network_sim/workload/served.py`) at a ladder of configs up to B0 (`Config.b0()`,
30 layers), word-level and on bits. On bits it is `lift.on_bits`: every Definition Serve_v1 reaches, lifted to bits all
the way down (`ir="boolean"` can't derive Serve_v1; `lift.py`'s docstring says why). Two verifiers are measured:

* `measure`, the NCI check: `nci-check --policy` (`NciCheck.Desc`, one summary per Definition) on an instance whose θ is
  the registered inputs and whose isolation blocks are the root's Calls, one each. `--policy` leaves out θ's provenance
  (`NciCheck.Baseline`), which reads the registration log and adds work in θ's length and the log's. Where the unrolled
  circuit is small enough (`--unrolled-max`), the instance expanded gate by gate also goes through
  `nci-check --policy --unrolled` (`NciCheck.check`, the gate-by-gate reference), and the two verdicts must agree.
* `partition`, the one-stage verifier's partition: `sampled-proofs-difftest`'s `partition` (`Partition.units`, `strata`,
  the expanded law and the digest, as the verifier of record computes them) on Serve_v1 under the template query of its
  Calls (`Q_template_instances`), under `Q_word` v1 and `Q_call` v1 at `X = 16` where its Calls' distinct Definitions are
  small enough (`--unrolled-max`), and on the served shape: the population program of Serve_v1's largest Definition at
  1 to 10⁶ instances under `Q_template_instance`, as a served window registers one. A template query must apply.

~~~sh
verity bench verifier_cost measure --out DIR [--config NAME ...] [--ir word|boolean ...] [--unrolled-max GATES]
verity bench verifier_cost partition --out DIR [--config NAME ...] [--ir word|boolean ...] [--unrolled-max GATES]
research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone -- \
  bash benchmarks/verifier_cost/job.sh measure|partition --unrolled-max 40000000   # every config, on a node
verity bench verifier_cost store RUN                     # its results.json or partition.json to the store; prints the art: id
verity bench verifier_cost plot ART --to PNG             # from the store
~~~

`nci_glm.sh` runs `nci-check --policy` on GLM-5.3's registered Programs: the eight per-rank workload Programs of the B1
Build (vLLM, TP8 on 8×H200, a 256-token prompt and 32 sampled tokens), each checked against the Build's pinned
digest. A registered Program's root is its inputs and one Call, the request, so the only division `NciCheck.Desc` can
check on it is that one isolation block; and its weights are bound per run (`Input<w>_v1`), so θ can't hold them
(`theta-not-registered`). `nci_glm.py` (its docstring has the cases) derives Programs with the same computed gates and
wires: rank 0's with its weights bound at registration (θ), one isolation block per rank; and the eight ranks' requests joined into
one Definition the node's root calls once, each collective reading the other ranks' partials where it read its
`peers_*` inputs, one isolation block per node session. The node's session is checked at the demo's policy, Γ = 2^44 gates and
α = 2^-25 bits per gate (`GAMMA`, `ALPHA`), and at its four neighbours. Finer divisions of a whole session (per step,
per layer, per Call) are out of `NciCheck.Desc`'s reach at this size: it holds each root node's reads as pieces of
gates, about one per leaf on GLM-5.3, so only a Call's internal reads stay small. Three of rank 0's Definitions (the
router, the fused all-reduce and norm, an FP8 linear layer) go per Call and in all through `--policy`,
`--policy --unrolled` and the Python reference, which must agree. `$OUT/results.json` (`verity/nci-glm/v1`) has a row
per check: the Program's SHA-512, bytes, root nodes and gates, the division, the verdict summarised, wall seconds and
peak RSS; `$OUT/ius/` has each verdict's isolation blocks. A rank's check takes about 21 minutes and 7 GB, the node's (a 2.2 GB
descriptor) about 2.6 hours and 50 GB, and the whole run about 3 hours (r20261023-064803-3874).

~~~sh
research data fetch art:32c488d8ca7bb12f80b38e45ec5a97d95312bcb62974a4d776da0b58d1adec35 --to B1 \
  --path 'build/rank*/build_workload/descriptor.json.gz'    # the Build; its meta's rank_workload_digests are the pins
research run --project verity --on vy-nebius-cpu-2 --source . --cwd clone --mem-gb 420 --mem-wait 3600 --timeout 6h \
  --send glm53-b1-pins.json $(for r in 0 1 2 3 4 5 6 7; do echo --send glm53-b1-rank$r.descriptor.json.gz; done) \
  -- bash benchmarks/verifier_cost/nci_glm.sh --jobs 10 [--case registered|rank|slice|node ...]
~~~

`--case registered-node` checks the node Program that `verity-vllm predict node` registered, as registered.
- **Inputs.** Its descriptor, sent as `glm53-b1-node.descriptor.json.gz`. Its registration, from `predict
  node-registration`'s `--out`: `trusted.json`, `provenance.json` and `registration.json`, sent as
  `glm53-b1-node.<file>`. Without the registration, only the `--policy` checks run.
- **Before any check**, it holds the descriptor to what it should be:
  - its SHA-512 is `--node-digest`;
  - θ is its root's `RegisteredInput` nodes, and its `annotations.theta` where it has them;
  - given the registration, it is the program of the record that the trusted input names as the run's own, and
    there is one provenance entry per θ node, each citing the log's first record.
- **The isolation block.** One isolation block holds every root node that isn't an input, which is the whole node's session (the demo's
  division).
- **The checks**, in the order they start (`NODE_POINTS`):
  - given the registration, the check of record (`nci-check INSTANCE TRUSTED`, `NciCheck.Baseline`), at A: the
    policy check, then θ's provenance against the registration log;
  - `--policy` at A (Γ = 2^44, α = 2^-25);
  - at C (α = 2^-28) with Γ = 2^44 and with Γ = 2^52;
  - at A's four neighbours.
- **Each `nci-check`** decodes the whole descriptor (about 50 GB resident for a whole-node session).
  `--check-mem-gb` caps each one's address space, which runs about 15 GB above its resident peak, so a check that
  outgrows its share of the run's `--mem-gb` fails rather than the node. A dry run on a root `Desc` can't read ends
  there.
- **The root must be inputs and one Call.** `NciCheck.Desc` can't hold a session laid out as the root's own nodes
  (`root_calls` in `results.json` counts them), for the reason above.

~~~sh
research run --project verity --on vy-nebius-cpu-4 --source . --cwd clone --mem-gb 120 --timeout 16h \
  $(for f in descriptor.json.gz trusted.json provenance.json registration.json; do echo --send glm53-b1-node.$f; done) \
  -- bash benchmarks/verifier_cost/nci_glm.sh --case registered-node --node-digest HEX --jobs 2
~~~

`flock_read.sh K...` measures Flock's Lean verifier (`flock-verify verify --zk`) on the H1 tile unit, step by step:
`flock_read.py` builds the 4×8 tile's statement circuit at each contraction K with one instance (`h1unit.Shape` in `benchmarks/pouw/h1_prove/h1unit.py`), and `flock_read/Main.lean`, built in a scratch Lake package against the checkout's verifier, times the
file's read, its SHA-512, `Zk.maskLogOf`, the decode and line split, each section's `Net.parse` (its tokenizer,
`Sparse.ofRows` and row checks), the class digest, `HmOut.parse`, `Stmt.setupHiddenOf` and one lincheck fold. Per K,
`$OUT` gets build-K.json (the unit's ANDs, matrix entries, text bytes and its Program descriptor's size) and
phases-K.jsonl (one row per step: milliseconds, RSS and peak RSS in MB).

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-1 --source . --pool proofs --cwd clone -- \
  bash benchmarks/verifier_cost/flock_read.sh 256 1024
~~~

`registration.sh N...` times the one-stage verifier's R3 against the window's ports: `registration.py` writes, per N, a
valid v1 audit with N more ports (`test_difftest._many_ports`, a MoE window's shape; TPEVAL's w13 window at chunk cap 64
is 1,536 members, about 12k ports), and `registration/Main.lean`, built in a scratch Lake package against the checkout's
verifier, times `Registration.check` (one canonical key per root, `Registration.portIndex`) and `Registration.checkRef`
(R3 as first written, quadratic in the ports; `check_eq_checkRef` proves them equal) on the parsed input, and fails where
their codes differ. `checkRef` runs up to `REF_MAX` ports (default 12288). `$OUT` gets registration.jsonl, one row per N
and function (ports, roots, milliseconds, peak RSS, codes).

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone -- \
  bash benchmarks/verifier_cost/registration.sh 1024 4096 12288 49152
~~~

`flock_tmpl.sh K...` is the same unit in `flock-ir-tmpl/v2` (`Flock/Tmpl.lean`): `flock_read.py --circuit-only`
builds the circuit, `python -m verity_flock.tmpl` writes its unit's templated form (Definitions read off the flat unit's
repeated blocks), and `flock_tmpl/Main.lean` times the templated read (`Tmpl.parse`) and diffs `Tmpl.unroll`'s lines
against the circuit's flat unit byte for byte. Per K, `$OUT` gets emit-K.json (the levels, Definitions, rows and bytes),
unit-K.tmpl, read-K.jsonl and diff-K.jsonl (the PHASE lines) and each step's `/usr/bin/time -v`.

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-1 --source . --pool proofs --cwd clone -- \
  bash benchmarks/verifier_cost/flock_tmpl.sh 256 4096
~~~

`flock_unroll.sh K...` times what the templated statement's read costs against the flat one's on the same unit: `Tmpl.unrollAt` at
`MODE` (default `iv/hybrid/4`) whole and a stage at a time (`rows`, `posMap`, `Net.ofRows`, `invOf`, `posOk`, `placed`,
`Sparse.ofRows`), the logical `Tmpl.unroll`, and `Net.parse` on the circuit's flat unit. Per K, `$OUT` gets unit-K.tmpl,
unroll-K.jsonl (every PHASE line, with its step) and each step's `/usr/bin/time -v`.

`qcall.sh CONFIG X...` measures `Q_call` v1 (`Flock.Qcall`, an open walk in `tests/test_verifier_cost.py`) on one Serve_v1
config's word-level descriptor: `flock-verify qcall-program` at each X, with the output's SHA-256 so two commits' outputs
compare, a `perf` profile at the first X, and the one-stage `partition` piece under `Q_call` v1 at X = 16. Its header has
the command.

`results.json` (`verity/verifier-cost/v1`) has one row per config and IR: the description's bytes, Definitions and body
nodes, the unrolled circuit's gates, and per check its wall seconds (the best of `--repeat`), peak RSS and the verdict.
`partition.json` (`verity/verifier-cost-partition/v1`) has one row per config, IR, shape and query: the description's
bytes, the unrolled gates, the gates of the Calls' distinct Definitions (what `Q_word` and `Q_call` lay out), the wall
seconds, peak RSS and the population (null where the verifier refuses the query). `store` puts either in the store as
an `evidence/v1` record naming the run and its commit; it runs where the store's keys are (a VM), since a node's shell
has none.

Apart from `nci_glm`'s inputs, nothing here is a served Program: `on_bits`'s output and `nci_glm`'s derived Programs are
this benchmark's constructions, and nothing outside this directory reads them.
