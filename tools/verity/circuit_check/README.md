# circuit-check

`circuit-check` runs the property checks every circuit passes whenever it is built or changed: a Definition, a
subcircuit template, a served workload's request Program, and their Boolean lowerings. It writes one machine-readable
report. Any change that adds or modifies a circuit passes it, and the report goes in the PR.

## Running it

```text
uv run circuit-check F32Add_v1 'RMSNormTriton_v1{N=64,EPS=1e-05}'   # Definitions, by id
uv run circuit-check rope-head gemm-coordinate/k64/sm80-mma-bf16     # a template, a subcircuit
uv run circuit-check path/to/build_request_LP3_T2                    # a served workload's request Program
uv run circuit-check verity/ml/circuits/ops/boolean/gf2k.circuit-check.py   # what a sidecar's bindings reach
uv run circuit-check --all --out report.json                         # every registered Definition and template
```

A target is one of:

- **definition**: a registered Definition bound to the statics its id writes, such as `F32Add_v1`,
  `RMSNormTriton_v1{N=64,EPS=1e-05}` or `GemmCoordinate_v2{K=64,DOT=AmpereBF16TcDot16_v2}`;
- **template**: a subcircuit template at its example parameters (`rope-head`), at given ones (`rope-head:D=128`), or one
  of its subcircuits (`gemm-coordinate/k64/sm80-mma-bf16`);
- **row**: a served workload's request Program, a Build's directory or its `instances.json.gz`;
- **sidecar**: a Definition module's `<module>.circuit-check.py`, which checks the module's own Definitions its bindings
  reach, and its templates.

Three modes check one thing each: `--structural UNIT.tmpl TARGET` holds a `flock-ir-tmpl` template to the Definition's
flat C-Flock lowering; `--ambient TARGET...` lowers each target in a plain and a perturbed interpreter and requires one
text, with no read of the environment, a clock or an unseeded draw; `--batch` checks the batch invariance of the
registration's templates.

Useful options: `--out FILE` (the report), `-j N` (workers; one per core by default), `--strict` (fail on known failures
too), `--fail-on-warnings`, `--update-pins` (pin the lowering counts measured now; review the diff), and `--no-cache`
(check every target now). `circuit-check --help` lists the rest.

A Definition with a few thousand gates checks in seconds. `--all`, and any large target, needs tens of GB:
run it on a node through `research run` (the writing-runs skill), with `--out $OUT/report.json` so the report is kept.

## The checks

`checks.py` is the full statement. In short:

- **correspondence**: the reference evaluator (`verity.ml.circuits.evaluation.evaluate`) runs on edge and random input
  vectors (`vectors.py`: NaN, infinities, zeros, subnormals, max-finite and integer edges at each width, then seeded
  random words). It must be total, and every realization must equal it bit for bit: each registered kernel, a template's
  evaluated target, the C-Flock Boolean lowering, and, for a Boolean Definition, the word Definition it names through the
  word view.
- **ports**: a Definition's declared return intervals (`verity/ml/circuits/PROTOCOL.md` §12) hold on those vectors, and a
  primitive's transfer stays within the interval it gives.
- **redundant gates**: a value computed twice inside one proof block, and ANDs on identical operands. A warning, counted.
- **dead gates**: gates no output depends on. A warning, counted.
- **partition**: the word partition `Q_word_v1{X,W,R=no-recompute}`: every gate certified by exactly one proof block,
  reads across proof blocks only through committed values, no value computed in two proof blocks, and the width rule.
  `Q_call` v1 is recorded beside it.
- **lowering pins**: a lowering's AND/XOR/NOT counts and its circuit digest against the pin recorded for it.

A failure is a correctness or invariant violation, named `"<check>/<code> <definition> ..."`. Redundant and dead gates
are warnings with counts.

## The report

The report's schema is `verity-circuit-check/v1`. It has `options` (the vector count, seed, partition widths and
budgets), `summary`, and one entry per target in `targets`. The summary says whether the run passed (`ok`), names every
failure (`failure_names`), separates new failures from known ones, and counts each warning by Definition. A target's
entry has its `definition`, `form` (primitive or composite), `gates`, the `correspondence` record (vectors compared and,
per realization, its AND count, mismatches and pin status), `partition`, `redundant_gates`, `dead_gates`, its `failures`
and `warnings`, and where its time went (`seconds_by_check`). One line per target goes to stdout, for example:

```text
ok   definition F32Add_v1  failures=0 warnings=1 q_call=1u/32b (0.1s)
warn redundant-gates/boolean: 42 over 1 target(s)
```

The exit status is 1 when a check fails that `known/` doesn't list, or, with `--all`, when a known entry no longer fails;
0 otherwise.

## Where things are

- `src/circuit_check/`: `cli.py` (the command), `checks.py` (the checks and the report), `targets.py` (what a target
  string names, and the catalog `--all` runs over), `vectors.py`, `bodies.py` (a composite's evaluation, body by body),
  `by_calls.py` (a root too large to build whole, checked by its calls), `structural.py`, `ambient.py`, `cache.py` (a
  target's result reused while everything its check read is unchanged).
- Sidecars (`sidecars.py`): each Definition module has `<module>.circuit-check.py` beside it, its bindings (the small
  statics the catalog checks it at), and `<module>.circuit-check.json`, its pins. A new Definition needs a binding: the
  suite fails by name for a registered family no binding reaches.
- `src/circuit_check/known/`: the failures `main` already has, one file each with its cause (`known.py`).
- `src/circuit_check/pins/`: pins of Definitions no single module owns, one file each (`entries.py`).
- `tests/`: the suite. It runs circuit-check over the whole catalog, one target per test, and tests the tool itself.
