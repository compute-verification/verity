# exact_kit

Primitives for the lane's theory and referee scripts, and their one check registry. The distribution is
`verity-exact-kit` and its module `exact_kit`, a workspace member, so a script anywhere in the workspace environment
imports it with no `sys.path` entry.

The kit holds no theory and no fold semantics. A referee and the author it checks may both use it, and neither re-runs
the other's argument. Fold semantics belong to the receipt-fold module. MMA semantics come from
`verity.ml.circuits.silicon` and are never re-derived here.

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
- `exact_kit.f2`: works on int-bitmask rows, where bit j is column j.
  - `rank`, `rref` (canonical: sorted by pivot, the lowest set bit), `reduce` and `in_span`.
  - `nullspace(rows, n)`, `span`, `kernel(rows, n)` and `dot`.
  - `from_matrix`, `to_matrix` and `transpose`.
  - `alternating(n, pairs)`, `wedge(n, a, b)` and `xor_rows` for bilinear forms.
- `exact_kit.fp`: the same over F_p on int lists: `rref`, `rank`, `nullspace`, `span`, `kernel`, `normal` (projective)
  and `lines`.
- `exact_kit.cube`: functions on {0,1}^n, indexed by sum x_i 2^i.
  - `mobius(values, n, bits=None)` gives the coefficients exactly over Z, or in Z/2^bits. `evaluate` is its inverse.
  - `quadratic_rows(coeff, n, plane)` gives the degree-2 part as an F_2 matrix.
  - Also `points`, `bit`, `degree` and `popcount`.
- `exact_kit.wrap`: `u` and `s` (k-bit unsigned and two's complement), `u32` and `s32`, `rotl`, `rotr` and `v2`, on ints
  and numpy arrays.
- `exact_kit.search`: exact kernel searches over a packed domain. Probes only prune; every result is checked on the whole
  domain.
  - `kernel(f, xs, deltas, act=xor)`: every delta with f(act(x, delta)) = f(x) on all of xs.
  - `meet(left, right)`: sort-merge matching of signatures.
  - `meet_kernel(g, h, xs, A, B)`: every (a, b) with g(x, a) = h(x, b) on all of xs, found by meeting in the middle.
  - `lift(ok, bits)`: 2-adic lifting, one bit at a time.
- `exact_kit.sat.Cnf`: a CNF builder over named variables, solved by pysat (CaDiCaL).
  - Gates: `and_`, `or_`, `xor`, `xor_all`, `parity` and `maj`.
  - Words: `word`, `const`, `add_words`, `xor_words`, `mul_const`, `equal`, `differ` and `value`.
  - Cardinality: `at_most`, `at_least` and `exactly`.
  - Solving: `solve`, `models(over, limit)` and `count`.

Every primitive is cross-checked in `tests/`, against brute force on small cases and against sympy's linear algebra over
GF(p). Run them from the repository root, with `uv run tools/verity/check/suites.py exact_kit`. Inside this directory uv
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
