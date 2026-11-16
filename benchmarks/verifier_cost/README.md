# verifier_cost

A verifier must work in proportion to the description it reads, never to the circuit that description unrolls to
(AGENTS.md "Design rules": O(circuit description), never O(unrolled circuit)). This benchmark holds the verifiers to
that rule by timing them against both sizes: the inference-only check (`measure`) and the one-stage verifier's
partition (`partition`) on the Serve_v1 workload at a ladder of configs, and a set of scripts that time the Lean
verifiers' other steps one by one.

## Try it

The inference-only check on the smallest config, word-level, once its Lean executable is built:

~~~sh
P=verity/protocols/workload_classification/training_detection/inference_only/verifier/lean
bash tools/verity/lean/setup.sh $P && (cd $P && lake build inference-only-check)
uv run --extra torch-cpu verity bench verifier_cost measure --out /tmp/verifier_cost --config tiny --ir word
~~~

It prints one JSON line per config: the description's bytes, the gates it unrolls to, the seconds the check took on the
description (`desc_s`) and on the unrolled circuit (`unrolled_s`), and whether the two verdicts agree (`agrees`).

## Commands

~~~sh
verity bench verifier_cost measure --out DIR [--config NAME ...] [--ir word|boolean ...] [--unrolled-max GATES]
verity bench verifier_cost partition --out DIR [--config NAME ...] [--ir word|boolean ...] [--unrolled-max GATES]
research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone -- \
  bash benchmarks/verifier_cost/job.sh measure|partition --unrolled-max 40000000   # every config, on a node
verity bench verifier_cost store RUN                     # its results.json or partition.json to the store; prints the art: id
verity bench verifier_cost plot ART --to PNG             # from the store
~~~

## Four of the measurements

`inference_only_glm.sh` runs `inference-only-check --policy` on GLM-5.3's registered Programs: the eight per-rank workload Programs of the B1
Build (vLLM, TP8 on 8×H200, a 256-token prompt and 32 sampled tokens), each checked against the Build's pinned
digest. A registered Program's root is its inputs and one Call, the request, so the only division `InferenceOnlyCheck.Desc` can
check on it is that one inference block; and its weights are bound per run (`Input<w>_v1`), so θ can't hold them
(`theta-not-registered`). `inference_only_glm.py` (its docstring has the cases) derives Programs with the same computed gates and
wires: rank 0's with its weights bound at registration (θ), one inference block per rank; and the eight ranks' requests joined into
one Definition the node's root calls once, each collective reading the other ranks' partials where it read its
`peers_*` inputs, one inference block per node session. The node's session is checked at the demo's policy, Γ = 2^44 gates and
α = 2^-25 bits per gate (`GAMMA`, `ALPHA`), and at its four neighbours. Finer divisions of a whole session (per step,
per layer, per Call) aren't checked here: their root would be the request's body, whose reads `InferenceOnlyCheck.Desc` holds as
segments of gates, each Definition's return coalesced into runs (`Desc.coalesce`); `inference_only_cost.py glm` counts and times
them. Three of rank 0's Definitions (the router, the fused all-reduce and norm, an FP8 linear layer) go per Call and in
all through `--policy`, `--policy --unrolled` and the Python reference, which must agree. `$OUT/results.json`
(`verity/nci-glm/v1`) has a row per check: the Program's SHA-512, bytes, root nodes and gates, the division, the verdict
summarised, wall seconds and peak RSS; `$OUT/ius/` has each verdict's inference blocks.

~~~sh
B1_BUILD=$(python3 -c 'import json; print(json.load(open("benchmarks/sampling_plan/glm53_blocks.json"))["build"])')
research data fetch "$B1_BUILD" --to B1 \
  --path 'build/rank*/build_workload/descriptor.json.gz'    # the Build; its meta's rank_workload_digests are the pins
research run --project verity --on vy-nebius-cpu-2 --source . --cwd clone --mem-gb 420 --mem-wait 3600 --timeout 6h \
  --send glm53-b1-pins.json $(for r in 0 1 2 3 4 5 6 7; do echo --send glm53-b1-rank$r.descriptor.json.gz; done) \
  -- bash benchmarks/verifier_cost/inference_only_glm.sh --jobs 10 [--case registered|rank|slice|node ...]
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
- **The inference block.** One inference block holds every root node that isn't an input, which is the whole node's session (the demo's
  division).
- **The checks**, in the order they start (`NODE_POINTS`):
  - given the registration, the check of record (`inference-only-check INSTANCE TRUSTED`, `InferenceOnlyCheck.Theta`), at A: the
    policy check, then θ's provenance against the registration log;
  - `--policy` at A (Γ = 2^44, α = 2^-25);
  - at C (α = 2^-28) with Γ = 2^44 and with Γ = 2^52;
  - at A's four neighbours.
- **Each `inference-only-check`** decodes the whole descriptor (about 50 GB resident for a whole-node session).
  `--check-mem-gb` caps each one's address space, which runs about 15 GB above its resident peak, so a check that
  outgrows its share of the run's `--mem-gb` fails rather than the node. A dry run on a root `Desc` can't read ends
  there.
- **The root must be inputs and one Call.** `InferenceOnlyCheck.Desc` can't hold a session laid out as the root's own nodes
  (`root_calls` in `results.json` counts them), for the reason above.

~~~sh
research run --project verity --on vy-nebius-cpu-4 --source . --cwd clone --mem-gb 120 --timeout 16h \
  $(for f in descriptor.json.gz trusted.json provenance.json registration.json; do echo --send glm53-b1-node.$f; done) \
  -- bash benchmarks/verifier_cost/inference_only_glm.sh --case registered-node --node-digest HEX --jobs 2
~~~

`registration.sh N...` times the one-stage verifier's R3 against the window's ports: `registration.py` writes, per N, a
valid v1 audit with N more ports (`test_difftest._many_ports`, a MoE window's shape; TPEVAL's w13 window at chunk cap 64
is 1,536 members, about 12k ports), and `registration/Main.lean`, built in a scratch Lake package against the checkout's
verifier, times `Docket.check` (one canonical key per root, `Docket.portIndex`) and `Docket.checkRef`
(R3 as first written, quadratic in the ports; `check_eq_checkRef` proves them equal) on the parsed input, and fails where
their codes differ. `checkRef` runs up to `REF_MAX` ports (default 12288). `$OUT` gets registration.jsonl, one row per N
and function (ports, roots, milliseconds, peak RSS, codes).

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone -- \
  bash benchmarks/verifier_cost/registration.sh 1024 4096 12288 49152
~~~

`flock_served.sh KIND [N]` measures the Lean verifier on a served statement: one kind of a saved served step
(`one_stage/a11.py --save-values`, sent as a `.tar.gz`) staged as a11 stages it, flat, or with `--layout MODE` as a
templated statement (with `--structural`, `struct_lower`'s template, the flat proof block never built). With a recorded a11
session of that kind sent (a `.tar` of its session directory), it times `archive-put` and `verify --zk --partition
--program` as `a0.lean_verify` runs them, under `perf record` where the node has it, so the verify's own phase lines
(`inputs:`, `setup (drawn):`, the session) split the profile, a templated statement's as well as a flat one's. With
`--phases`, and at every templated statement, `flock_served/Main.lean` times verify's work before the session a step
at a time: the read, SHA-512, `Zk.maskLogOf`, `ProvedScope.circuitOf`, `ProvedScope.checkOf`, `PortRoots.checkOf`,
`KsScope.checkOf` and `Stmt.setupTablesOf` with the sample, each with its RSS. `build.json` has the proof block's ANDs and matrix
entries, k_log, the circuit file's bytes and lines, and its Program descriptor's bytes, the size a description-cost
verifier would read.

~~~sh
research run --as proofs --project verity --on auto --mem-gb 128 --timeout 7200 --source . --cwd clone \
  --send values.tar.gz [--send session.tar] -- bash benchmarks/verifier_cost/flock_served.sh KIND [N] [--phases] \
  [--layout iv/hybrid/4 [--structural] [--stmt [--base REV]]]
~~~

`flock_served/Main.lean` parses the circuit as `verify` does (`circuitOfKnowing`: the file's SHA-512 already known). On a
templated statement, `--stmt` times that parse instead with `flock-tmpl stmt` (`flock_tmpl/Main.lean`): whole, then a
stage at a time (`tmplOf`, `Tmpl.parse`, each of `TmplUnit.read`'s checks, `HmRow.parseWith`'s decode, split, class
digest and sections), with the template's size against the unrolled proof block's (rows and entries, the copies multiplied
through the calls) into stmt.jsonl; `--base REV` first times the same `flock-tmpl` built against REV's verifier on the
same staged statement (stmt-base.jsonl).

`ledger_replay/ledger_replay.sh S...` times one replay of the auditor's ledger against its sessions:
`ledger_replay/ledger_replay.py` writes, per S, a ledger of S sessions as the challenger's writer signs one (two-step:
a reserve, an open and an end a session, a declaration a day, the sessions in one period, or in periods of `PERIOD`)
under a key at the deployed LMS parameters, with what the `challenger_ledger` piece's reference
(`challenger.audited_session`) answers on it, and times the reference's replay. `ledger_replay/Main.lean`, built in a
scratch Lake package against the checkout's verifier, times the compiled replay (`Spot.Challenger.checked`), the
difftest's piece around it, and the replay's parts made apart from it: each entry's parse, its line's SHA-512, its
message, its LMS verification, and the loop's searches over what it has read so far. It refuses a ledger on which the
piece's answer is not the reference's. A period's close lists its sessions, so one period's sessions stop where the
close passes a ledger line's cap (`MAX_LEDGER_LINE`): both refuse that size, and its row says so. `$OUT` gets
inputs.jsonl (one row per S: entries, bytes, the longest line, the reference's seconds and signature checks),
ledger_replay.jsonl (one row per S and phase: microseconds, the least of `REPEAT`, and peak RSS), table.jsonl
(`ledger_replay.py --table`: one row per S, the split, each part's share, and a least-squares fit of what the parts
leave to `c + a S + b S²`), cpu.txt and the logs.

~~~sh
research run --as proofs --project verity --on vy-nebius-cpu-11 --source . --pool research --cwd source \
  --mem-wait 1800 -- bash benchmarks/verifier_cost/ledger_replay/ledger_replay.sh 10 30 100 300 1000
~~~

