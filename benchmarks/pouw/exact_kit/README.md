# exact_kit

Primitives for the lane's theory and referee scripts, and their one check registry. The distribution is
`verity-exact-kit` and its module `exact_kit`, a workspace member, so a script anywhere in the workspace environment
imports it with no `sys.path` entry.

The kit holds no theory and no fold semantics. A referee and the author it checks may both use it, and neither re-runs
the other's argument. Fold semantics belong to the receipt-fold module. MMA semantics come from
`verity.definitions.silicon` and are never re-derived here.

**What a caller may import** is each module's `__all__` (`exact_kit.checks`, `exact_kit.f2`, `exact_kit.wrap`,
`exact_kit.search`, `exact_kit.sat`); the package itself re-exports nothing. Its readers are `../theory/`'s scripts and
`../edited_gemm/egemm_depthr_check.py`.

**What it never does.** It imports nothing of Verity and nothing of `research`: the standard library, numpy and pysat
alone. It reads the environment in one place, `exact_kit.settings` (`OUT` and the three variables `research run` sets),
which only `Checks.finish` asks, for where the verdict and the run's result go, and writes no other file.

## API

- `exact_kit.checks.Checks`: the registry.
  - `check(name, ok, detail=None)` and `mutation(name, caught, detail=None)` print `[OK  ] name: detail` or
    `[FAIL] …`, and `mutation caught: name` or `mutation NOT caught: name`. A detail of None prints no colon.
  - `with checks.mutant(name):` runs checks under a mutation, which is caught iff one of them fails.
  - `checks.mutations` is the `--mutations` flag, and `checks.args` is argv without the kit's flags.
  - `sys.exit(checks.finish())` prints `N/N checks OK, M/M mutations caught`. It writes the verdict JSON (schema
    `exact-kit/verdict/v1`) to `--verdict PATH`, else to `$OUT/<script>.verdict.json`, and under `research run` the run's
    `result.json`.
  - The verdict holds no times or paths, so two runs of one script give the same bytes.
- `exact_kit.f2`: works on int-bitmask rows, where bit j is column j: `rank`, and `from_matrix` for a 0/1 matrix's rows.
- `exact_kit.wrap`: `rotl` and `rotr` of k-bit words, on ints and numpy arrays.
- `exact_kit.search`: exact kernel searches over a packed domain. Probes only prune; every result is checked on the whole
  domain.
  - `meet(left, right)`: sort-merge matching of signatures.
  - `meet_kernel(g, h, xs, A, B)`: every (a, b) with g(x, a) = h(x, b) on all of xs, found by meeting in the middle.
- `exact_kit.sat.Cnf`: a CNF over named variables, solved by pysat (CaDiCaL; `SOLVER` names it).
  - Variables and clauses: `var`, `word`, `add` and `true`.
  - Gates: `and_`, `or_`, `xor`, `xor_all` and `parity`.
  - Solving: `solve`, `models(over, limit)` and `value`.

Every primitive is cross-checked in `tests/`, against brute force on small cases and against sympy's linear algebra over
GF(2). Run them from the repository root, with `uv run tools/verity/check/suites.py exact_kit`. Inside this directory uv
stops at `benchmarks/pouw`'s project, takes the kit for a workspace of its own, and builds it a separate `.venv` and
`uv.lock`.

## A script as a recorded run

A script on the kit runs as a `research run`, and its verdict lands in `$OUT`. Without `pipefail` the run takes
`tee`'s exit code, and a failed verdict records a passing run:

~~~sh
research run --cwd source --run-record -- bash -c \
  'set -o pipefail; d=$(mktemp -d); trap "rm -rf $d" EXIT; research data fetch art:<script tree> --to "$d" >/dev/null &&
   cd "$d" && python -B ref_k.py | tee "$OUT/ref_k.out"'
~~~

A re-run is `research compare` against that Attempt. Same bytes in, same `ref_k.verdict.json` out.
