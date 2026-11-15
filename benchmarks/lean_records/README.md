# lean_records

The Lean tools' guard (`guard.py`): what a records run of `security/Proofs` (`lean_changed.py --records`, with its
requirers) costs under a base version of `tools/verity/lean/` and under this tree's, at the job's reservation and at a
lower one. A warm-up, then rounds of three arms on one kept records tree of a node, each arm on a one-line lemma of its own:

- **A**: the base tools, at the job's reservation;
- **B**: this tree's tools, at the job's reservation;
- **C**: this tree's tools, at `--low-gb` (by default a heavy build's figure, `BUILD_HEAVY_GIB`, read as GB, so its build
  runs fewer jobs).

Each arm reads, every second, what its process tree holds, its `lean` processes at once, and the user and system CPU of
the tools' own Python processes; at its end, the tree's CPU and each `security/Proofs` audit's build step and spot-check
from its log.

## The verdict

- **Over clean arms alone.** An arm is clean where its first `security/Proofs` build compiled modules. One whose lemma an
  earlier run on the node built restores every module from the node's artifact cache, which its `sampled:` line shows.
  Start a later job on a node past the lemmas the earlier ones built (`--first-lemma`).
- **B against A** on the build step's seconds and on the tools' own CPU, each within 1.1 of A's median, and each kind's
  peak within its reservation.
- **Whole seconds are reported and judge nothing.** The audit's spot-check recompiles a random sample of the modules its
  build restored (`check.py`'s `SAMPLE_RNG`), whose seconds differ several-fold from one sample to the next, and an arm's
  seconds move with the node's load. At B's reservation B runs A's plan, the same jobs on the same modules, so all B can
  add is its own code, which its CPU reads.

## Run

On a node with a `build` Lean pool, three rounds (the warm-up and nine arms), from the base `REF`, sending each of the
guard's `TOOLS` that `REF` has; arm A runs without the others. Each arm writes its tools and its lemma into the checkout,
which the guard restores at its end, so it runs in a clone (`--cwd clone`):

~~~text
mkdir -p /tmp/base && for f in check.py lean_changed.py lean_audit.py reserve.py reserve.json; do git cat-file -e REF:tools/verity/lean/$f 2>/dev/null && git show REF:tools/verity/lean/$f > /tmp/base/base-$f; done
research run --on NODE --pool research --as lean --project verity --mem-gb "$(python3 tools/verity/lean/lean_changed.py --ask security/Proofs)" --timeout 8h --source . --cwd clone $(for f in /tmp/base/base-*; do printf -- '--send %s ' "$f"; done) -- bash -lc 'python3 benchmarks/lean_records/guard.py --rounds 3 --first-lemma 0 --base "$RESEARCH_RUN_DIR/inputs" --out "$OUT"'
~~~

It writes `$OUT/guard.json` (each arm, the summary and the verdict) and per arm its log and memory series. The tests
(`tests/`) hold the log reading and the verdict to fixtures.
