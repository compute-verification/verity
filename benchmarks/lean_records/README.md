# lean_records

A guard on the Lean tools' own cost. A change to `tools/verity/lean/` can make a records run of `security/Proofs`
slower or hungrier for memory, so this benchmark runs that records run on one node under the base tools and under the
change's, and judges the change's against the base's: within 1.1 of its median on the build step's seconds and on the
tools' own CPU, with each kind's peak within its reservation. `guard.py`'s docstring says what each arm measures and how
the verdict reads it.

## Try it

~~~sh
uv run pytest tests/benchmarks/lean_records/test_guard.py
~~~

It needs nothing but the checkout, and holds the guard's reading of a log and its verdict to fixtures (4 tests, under a
second).

## Run

On a node with a `build` Lean pool, three rounds (the warm-up and nine arms), from the base `REF`, sending each of the
guard's `TOOLS` that `REF` has; arm A runs without the others. Each arm writes its tools and its lemma into the checkout,
which the guard restores at its end, so it runs in a clone (`--cwd clone`):

~~~text
mkdir -p /tmp/base && for f in check.py lean_changed.py lean_audit.py reserve.py reserve.json; do git cat-file -e REF:tools/verity/lean/$f 2>/dev/null && git show REF:tools/verity/lean/$f > /tmp/base/base-$f; done
research run --on NODE --pool research --as lean --project verity --mem-gb "$(python3 tools/verity/lean/lean_changed.py --ask security/Proofs)" --timeout 8h --source . --cwd clone $(for f in /tmp/base/base-*; do printf -- '--send %s ' "$f"; done) -- bash -lc 'python3 benchmarks/lean_records/guard.py --rounds 3 --first-lemma 0 --base "$RESEARCH_RUN_DIR/inputs" --out "$OUT"'
~~~

It writes `$OUT/guard.json` (each arm, the summary and the verdict) and per arm its log and memory series.
