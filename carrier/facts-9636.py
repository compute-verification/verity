"""The texts mkbody-9636.py puts into #9636's body, for the carried head.

    python3 facts-9636.py FACTS.json

What is known at the carried head is written out below. What the two runs still out decide is in FILL, each key with what
it must say; the script refuses to write while a value there is still None.
"""
import json
import sys

HEAD = "4f51d8ff9b161a922570980d44c5c73d965b9ec6"
LEAN_RUN, QUICK_RUN, SLOW_RUN = "r20261114-070915-cc7b", "r20261114-071129-b00e", "r20261114-071306-9328"

FILL = {
    # one sentence: what the records run did with the two committed locks. If it passed on them: "passes on the
    # committed files without regenerating either". If a lock was regenerated and committed, say which commit holds it.
    "records": None,
    # the tries: "exit 0" for each of the nine, or what failed
    "tries": None,
    # the quick tier: "N suites, N passed, 0 failed", then the passes worth naming (the SPOT verifier's own suite,
    # benchmarks/one_stage 1,134, verity-network-traces 728, verity-core-sampled-proofs 327 with 2 skipped,
    # the certifier's verifier's network suite 91, repository)
    "quick": None,
    # how many guarantees read the compiled check's changed definitions, from the records run's log, as a word
    # ("thirty"); or "" to leave the count out
    "readers": None,
}
missing = [k for k, v in FILL.items() if v is None]
if missing:
    sys.exit(f"facts-9636.py: fill in {missing} from the runs' logs first")

holds = (
    "**What the branch holds beside its own change.** The branch holds `main` (`512644d14fbc`), the complete answer's head "
    "(#9458, `1b4b9acc84f3`, open), which itself holds the heads of #9810 (`12ee02ed7e12`, open) and #9717 "
    "(`3e1103ded562`, open), and #9023's three commits (`6b782d972ec4`). Against `main` it is 48 files: ten that only this "
    "change touches, 28 that only #9458's head touches, and ten that both do. When #9458 lands, `main` is merged in here "
    "again and the difference is this change's twenty files alone. This change's own files are twenty:"
)

readers = f"in what {FILL['readers']} guarantees read" if FILL["readers"] else "in what the guarantees that read the compiled check read"
lock = f"""## The lock

Both locks are as a records run writes them at the head: run `{LEAN_RUN}` (`python3 tools/verity/lean/lean_changed.py --records verity/protocols/proofs/spot/verifier/lean --update`, which audits the SPOT verifier's Lean and each package that requires it: the certifier's verifier, `security` and `security/Properties`) {FILL['records']}.

What moved against `main`, in plain words:

- **No guarantee's own record moves**, in either lock. `Properties.Spot.acceptMeans` keeps its signature and both of its hashes: its own text is what it was. The lists of witnesses, of guarantees without witnesses, of deployed guarantees and their trust gaps, of assumptions and of domain premises are `main`'s.
- **What `Properties.Spot.acceptMeans` reads moves**, which is the statement above. From this change: `Properties.Spot.ExpectsV2`, `R1`, `R7` and `Registered` changed, and `Properties.Spot.CertifiedHeld` is new. From #9023's commits: `Properties.Spot.UnitsProved` changed, with `OpenedRegistered` in place of `SameRoots`.
- **One open premise's digest moves**: `Properties.Spot.ExpectsV2`, the definition widened above. It is the open premise it was, with its owner and its due date.
- **The compiled check's definitions move** {readers}, none of whose statements moves. From this change: `Spot.Registration.Verifier` (with its new field `certified`), `check`, `checkOutputs` and the new `certifiedHeld`; `Spot.OneStage.check`, `paramsOfJson`, `paramsParsed`, `recordChecks` and the new `CERTIFIED_KEYS` and `CERTIFIED_REFUSED`; the new `Spot.Record.certifiedCheck`. From #9023's commits: `Spot.Record.checks` and the new `subDict`.
- **The lines of #9458's head** (held here until it lands): `Properties.Spot.publicOf`, `Spot.OneStage.Public` with its new field `complete`, `publicJson`, and the new `Spot.Record.allRan`; and what its own locks record of `Spot.ChallengerCoins`, `Spot.ChallengerRecord`, `Spot.ChallengerSession` and `Spot.Json`.

Nothing else differs from `main`'s locks. Against this pull request's head before the carry, 211 guarantees' own records differ (106 and 39 in `security/lean-audit.json`'s `guarantees` and `unwitnessed`, 38 and 28 in `security/Properties/lean-audit.json`'s), and each is `main`'s record as it stands now; none is of `Properties.Spot`."""

runs = f"""## Runs at the head (`{HEAD}`)

- **Records** (`python3 tools/verity/lean/lean_changed.py --records verity/protocols/proofs/spot/verifier/lean --update`), run `{LEAN_RUN}`, vy-nebius-cpu-10: {FILL['records']}. It is the fast path, with no kernel replay.
- **Tries** (`python3 tools/verity/lean/lean_changed.py --try FILE --siblings`, the nine files), the same run: {FILL['tries']}.
- **Quick tier** (`uv run tools/verity/check/suites.py --changed 512644d14fbc --quick`), run `{QUICK_RUN}`, vy-nebius-cpu-5: {FILL['quick']}.
- **The SPOT verifier's two suites whole, slow tests included** (`uv run tools/verity/check/suites.py verity/protocols/proofs/spot/verifier verity/protocols/proofs/spot/verifier/lean`), run `{SLOW_RUN}`, vy-nebius-cpu-4: `verity/protocols/proofs/spot/verifier` 951 passed; `verity-core-sampled-proofs` 329 passed, 2 skipped; none cached.
- **Retired terms** (`tests/lint/test_repo_retired_terms.py`), run `{QUICK_RUN}`: 20 passed.
- **Ratchet gate** (`python3 adapters/vllm/tools/ratchet_gate.py BASE HEAD`). Against `main` (`512644d14fbc`), run `{QUICK_RUN}`: exit 1, with 17 retired-term sites in six files (`benchmarks/one_stage/tests/test_stop_audit.py` 1 to 2, `benchmarks/one_stage/two_stage.py` 98 to 102, the verifier's README 71 to 72, `tests/test_two_stage.py` 102 to 107, `tests/test_window_place.py` 0 to 3, `window_place.py` 0 to 3). Every one of them is a line of #9458's head: `main` merged with #9458's head alone gives the same 17 in the same files, and against that merge as the base the gate passes at this head ("no retired-term site is new in the 18 file(s) it changes"; nothing new from ruff, environment reads, silent handlers or lane text). The last two were run on the author's VM, not on a node.
- **Markdown caps** (`python3 tools/verity/check/gates/markdown_caps.py 512644d14fbc HEAD`), run `{QUICK_RUN}`: two Markdown files changed, none over its cap. The verifier's README is 47,658 bytes against 49,152: `main` cut it under the cap itself (#9672), #9458's paragraph brings it to 47,544 and this change adds one line of 114."""

merged = f"""## Merged tree

The head descends from the head this pull request had before the carry (`a287d4bf1247`), so no commit of it was rewritten. Over that head:

- **The 30 layout and word moves `main` took since are crossed by `tools/move/restack.py`, none by hand**, in the order `main` took them: four (run `r20261114-055226-0840`), then 25 up to Dither's and the clear audit's last (run `r20261114-061855-bc20`), each pass ending `clean`. No pin is stale (`python3 tools/verity/check/pins.py --stale`).
- **#9458's line is merged at the two commits where it had settled its own files**: `a275f10724a9`, at the terms move's base (`b443ddb01f1f`), and its head `1b4b9acc84f3` (`cee0a04f27c7`). Its settlement of `benchmarks/one_stage/tests/test_stop_audit.py` is taken as it is.
- **Seven hunks are settled by hand, each with both sides.** This change gives `verifier.run` and its two calls in `service.py` the keyword `certified`; #9810 gave the same lines `family`, and #9555 `key` and `main`. The lines now pass all four, the others' first. One sentence of `Spot/Pieces.lean`'s docstring holds both sides' clauses.
- **`main` (`512644d14fbc`) is merged in** with no conflict (`e3eb62b196db`), and `tests/lint/allowlists/repo_retired_terms.json` is `main`'s file whole (`4f51d8ff9b16`): the reruns of two moves had left three of its entries in another order.

Against `main` merged with #9458's head, this change's twenty files differ by the same added and removed lines as before the carry, but one line more in `service.py`, a call wrapped onto a second line.

`git merge-tree --write-tree` of the head merges clean with `main` as it has moved since (`f28bdad24636`). With #9549's head (`51c523f27a87`) it meets one conflict, in one record literal of `Spot/OneStage.lean` (`certified` beside `tables`); the settlement is both lines, made by whichever of the two lands second."""

tries_ref = f"run `{LEAN_RUN}`, at the head."
json.dump({"holds": holds, "tries_ref": tries_ref, "lock": lock, "runs": runs, "merged": merged}, open(sys.argv[1], "w"), indent=1)
print(f"wrote {sys.argv[1]}")
