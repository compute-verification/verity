"""The texts mkbody-9636.py puts into #9636's body, for the final carried head (after #9458's new head is merged in).

    python3 facts-9636-final.py FACTS.json

The constants below and FILL are what the final head and its runs decide; the script refuses to write while one is None.
Everything else is what holds at `b8ca65943400` and still holds after the last merge, unless that merge says otherwise.
"""
import json
import sys

HEAD = None            # the final carried head, full sha
MAIN = None            # the main it holds, 12 hex
H9458 = None           # #9458's head it holds (CLOSER's carry), 12 hex
LEAN_RUN = None        # records run and the nine tries at HEAD
QUICK_RUN = None       # caps, ratchet gate, retired-terms lint, suites --changed MAIN --quick at HEAD
SLOW_RUN = None        # the SPOT verifier's two suites whole at HEAD

FILL = {
    # one clause: what the records run did with the two committed locks, e.g. "passes on the committed files without
    # regenerating either"; if a lock was regenerated and committed, which commit holds it
    "records": None,
    # "exit 0" for each of the nine, or what failed
    "tries": None,
    # "N suites, N passed, 0 failed", then the passes worth naming
    "quick": None,
    # the slow tier's two suites with their counts
    "slow": None,
    # the gate against MAIN: its exit and the sites, each with whose line it is
    "gate": None,
    # the caps gate's line, with the verifier README's bytes against 49,152
    "caps": None,
    # files against MAIN: all, only this change's, only #9458's, both (git diff --name-only against MAIN and against
    # MAIN merged with H9458)
    "files": None,
    # how the last merge (H9458, then MAIN if it moved) was settled, one sentence a hunk, or "no conflict"
    "last_merge": None,
}
missing = [k for k, v in {**FILL, "HEAD": HEAD, "MAIN": MAIN, "H9458": H9458, "LEAN_RUN": LEAN_RUN,
                          "QUICK_RUN": QUICK_RUN, "SLOW_RUN": SLOW_RUN}.items() if v is None]
if missing:
    sys.exit(f"facts-9636-final.py: fill in {missing} first")

holds = (
    f"**What the branch holds beside its own change.** The branch holds `main` (`{MAIN}`), the complete answer's head "
    f"(#9458, `{H9458}`, open) and #9023's three commits (`6b782d972ec4`, open). #9810 and #9717, which #9458's head "
    f"held, are on `main`. Against `main` it is {FILL['files']}. When #9458 lands, `main` is merged in here again and "
    "the difference is this change's files alone. This change's own files are:"
)

lock = f"""## The lock

Both locks are as a records run writes them at the head: run `{LEAN_RUN}` (`python3 tools/verity/lean/lean_changed.py --records verity/protocols/proofs/spot/verifier/lean --update`, which audits the SPOT verifier's Lean and each package that requires it: the certifier's verifier, `security` and `security/Properties`) {FILL['records']}.

What moved against `main`, in plain words:

- **No guarantee's own record moves**, in either lock. `Properties.Spot.acceptMeans` keeps its signature and both of its hashes: its own text is what it was. The lists of witnesses, of guarantees without witnesses, of deployed guarantees and their trust gaps, of assumptions and of domain premises are `main`'s.
- **What `Properties.Spot.acceptMeans` reads moves**, which is the statement above. From this change: `Properties.Spot.ExpectsV2`, `R1`, `R7` and `Registered` changed, and `Properties.Spot.CertifiedHeld` is new. From #9023's commits: `Properties.Spot.UnitsProved` changed, with `OpenedRegistered` in place of `SameRoots`.
- **One open premise's digest moves**: `Properties.Spot.ExpectsV2`, the definition widened above. It is the open premise it was, with its owner and its due date.
- **The compiled check's definitions move** in what the guarantees that read the compiled check read, none of whose statements moves. From this change: `Spot.Registration.Verifier` (with its new field `certified`), `check`, `checkOutputs` and the new `certifiedHeld`; `Spot.OneStage.check`, `paramsOfJson`, `paramsParsed`, `recordChecks` and the new `CERTIFIED_KEYS` and `CERTIFIED_REFUSED`; the new `Spot.Record.certifiedCheck`. From #9023's commits: `Spot.Record.checks` and the new `subDict`.
- **The lines of #9458's head** (held here until it lands): `Properties.Spot.publicOf`, `Spot.OneStage.Public` with its new field `complete`, `publicJson`, and the new `Spot.Record.allRan`. #9458's own committed lock (`1b4b9acc84f3`) does not yet read `Spot.OneStage.Public.complete` or `Spot.Record.allRan`, and reads `Spot.OneStage.Public` at its old digest; the records run writes all three here, as this branch's records commit `a7b27f9f8a2` has them.

Nothing else differs from `main`'s locks."""

runs = f"""## Runs at the head (`{HEAD}`)

- **Records** (`python3 tools/verity/lean/lean_changed.py --records verity/protocols/proofs/spot/verifier/lean --update`), run `{LEAN_RUN}`: {FILL['records']}. It is the fast path, with no kernel replay.
- **Tries** (`python3 tools/verity/lean/lean_changed.py --try FILE --siblings`, the nine files), the same run: {FILL['tries']}.
- **Quick tier** (`uv run tools/verity/check/suites.py --changed {MAIN} --quick`), run `{QUICK_RUN}`: {FILL['quick']}.
- **The SPOT verifier's two suites whole, slow tests included** (`uv run tools/verity/check/suites.py verity/protocols/proofs/spot/verifier verity/protocols/proofs/spot/verifier/lean`), run `{SLOW_RUN}`: {FILL['slow']}.
- **Retired terms** (`tests/lint/test_repo_retired_terms.py`), run `{QUICK_RUN}`: passed.
- **Ratchet gate** (`python3 adapters/vllm/tools/ratchet_gate.py {MAIN} HEAD`), run `{QUICK_RUN}`: {FILL['gate']}.
- **Markdown caps** (`python3 tools/verity/check/gates/markdown_caps.py {MAIN} HEAD`), run `{QUICK_RUN}`: {FILL['caps']}."""

merged = f"""## Merged tree

The head descends from the head this pull request had before the carry (`a287d4bf1247`), so no commit of it was rewritten. Over that head:

- **The layout and word moves `main` took since are crossed by `tools/move/restack.py`, none by hand**: the 30 up to Dither's and the clear audit's, each pass ending `clean`, and then the registration map (`8469d5dcc164`), carried against the move's own commit because the branch was based after its base (`d9c4f62b2ac4`, `clean`).
- **#9458's line is merged at the commits where it had settled its own files**: `a275f10724a9`, at the terms move's base, its head `1b4b9acc84f3`, and its new head `{H9458}`.
- **The hunks settled by hand each keep both sides.** This change gives `verifier.run` and its two calls in `service.py` the keyword `certified`, beside #9810's `family` and #9555's `key` and `main`. One sentence of `Spot/Pieces.lean`'s docstring holds both sides' clauses. `Spot/Run.lean`'s `run` passes this change's `v` to `audit` and `main`'s `form` to `interiorAudit`. `service.py`'s Stream docstring is `main`'s sentence, with this change's clause on certified entries. {FILL['last_merge']}
- **`main` is merged in**, `tests/lint/allowlists/repo_retired_terms.json` is `main`'s file whole, and `tests/import_order.toml`, `verity/hardware/README.md` and `verity/hardware/tests/test_package.py`, which this change does not touch, are `main`'s.

With #9549's head (`51c523f27a87`) it meets one conflict, in one record literal of `Spot/OneStage.lean` (`certified` beside `tables`); the settlement is both lines, made by whichever of the two lands second."""

tries_ref = f"run `{LEAN_RUN}`, at the head."
json.dump({"holds": holds, "tries_ref": tries_ref, "lock": lock, "runs": runs, "merged": merged}, open(sys.argv[1], "w"), indent=1)
print(f"wrote {sys.argv[1]}")
