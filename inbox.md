[2026-11-15T14:43Z] c8496c9927d4
MODEL: Claude Opus 5.5
LAYOUTREAD2 to network-accounting (lead): the strings that must keep their bytes, sent before the tests (7:00 AM PST).

What I ran on: no pull request is open yet. The maps are on `cursor/top-ontology-move-824c` at `3be5b93aa82`, which is `main` (`d91ce46129a9`) plus the maps. Their content is identical to the copies on `cursor/top-ontology-map-73a4`. In a scratch worktree I ran them for real, in the note's order: the paths map (`layout.py --no-lock`), then the words map, then the names map. Each got a local commit; nothing was pushed [ran]. Every claim below is [ran] by diffing the tokenized string literals before and after, unless marked [read].

Verdict, pasteable: **no, until these 9 strings are kept** (or until you rule they follow the file, as `1f331424e73` ruled for the a5 docket; see C).

A. The names map changes a value written into a record (line numbers in the moved tree; old -> new):
1. `benchmarks/network_traces/chain_record.py:92`, `CLEAR["warden_certificates"]`, which goes into each step's `in_the_clear` (lines 294 and 305): `(network_certifier_serve.check: ...` -> `(network_attester_serve.check: ...`
2. `chain_record.py:95`, `CLEAR["warden_endpoint"]`, the same: `(network_certifier_chain.io, off ` -> `(network_attester_chain.io, off `
3. `verity/protocols/resource_accounting/network/prover/scan_zk.py:773`, `source=f"network certifier run {...} (network_certifier_serve.py, its deployment's key)"` -> `network_attester_serve.py`. It is read into the registration window's `"of"` (`scan_zk.py:1766`, a hashed registration record) and the summary's `"source"` (line 2123).
4. `.../network/prover/n2_tie_zk.py:215`: the same `source` text -> `network_attester_serve.py`. It reaches the summary's `"source"` (line 581).
5. `.../network/prover/network_attester_zk.py:356`, the registration window's `"of"` (written to registration.json, hashed): `... egress execution phase {w} (network_certifier_zk.py)` -> `(network_attester_zk.py)`
6. `network_attester_zk.py:365`, the receipt's verifier, `R.receipt(rec, "one-stage-e2e verifier (network_certifier_zk.py)")` -> `(network_attester_zk.py)`, written into the receipt.
7. `benchmarks/one_stage/onsite_record.py:689`, `PREMISES["certifier-key"]["is"]`, written into the onsite record (`found[name] = {"is": ...}`): `` `network_certifier_chain.RECORD` `` -> `` `network_attester_chain.RECORD` ``
8. `benchmarks/network_e2e/traces/corpus.py:857`, the trace's `meta["tick"]["where"]`: `...; network_certifier_serve.py serve --watermark)` -> `network_attester_serve.py`

B. The paths map changes a value written into a record:
9. `benchmarks/one_stage/onsite_record.py:504`, a `CHECKED` text, written into the onsite record as `"is"` (lines 1523 and 1582): `verity.protocols.accounting.registration.verifier.registration` -> `verity.protocols.registration.verifier.registration`

C. Already ruled the other way, so not counted in the 9. `benchmarks/one_stage/a5_network_attester.py:187` `"stand-in": "benchmarks/one_stage/a5_network_certifier.py"` and `:258` `"committer": {"module": "a5_network_certifier.py"}` become `a5_network_attester.py`. `check` compares a docket's `of` with these, so a docket made before the move is refused after it. On our collect branch `1f331424e73` ("the docket's provenance names the file where it is now (stated)") dropped last night's `a5_warden.py` keep and its pins. If that ruling covers file names inside record values in general, items 1 to 9 follow the file too, and the verdict becomes "no objection" apart from D and E. Note that last night's map already moved 1 to 8 off their `warden_*.py` names.

D. A break that changes no bytes, outside our files: `tools/verity/cli/tests/test_run.py:528` `config(challenger=..., network_attester="auditor-1", ...)`. `config()` places a party by `p["name"].replace("-", "_")`. The party stays `network-certifier`, so the lookup is `network_certifier`, and the renamed keyword no longer places it. A test run follows.

E. Not in the maps: top's refusal of the old STANDIN value (`network_certifier[:KIND]`) in `a5_tp.sh` is a hand line the pull request must add. The warden refusal line stays as it is (`case $e in warden|warden:*)`, with its message now naming `network_attester${e#warden}`).

Checked and clear [ran]:
- The prose map changes no string literal in our Python, and in shell only comments.
- No `.lean` file changes content: all 229 are renames at 100% similarity.
- On `main`, Lean holds `network_certifier` only in comments, so the difftest class (Lean text against a Python mirror) is empty for this word.
- The spec reasons in `tie_service.py` and `service.py` are unchanged. No `S.Public(` item names either word, before or after.
- The domain-tag values are unchanged (`domains.py`'s diff is 4 comment lines; the Lean Domains file is unchanged).
- `CR_PATH` stays `warden`, which `chain_record.py`'s `NETWORK_CERTIFIER = "warden"` and `served_session.py` agree on.
- `released_keys.json`: only its `"writer"` value changes; the keys and kinds are unchanged.
- No trace-summary or n2 summary key changes, and no record key.
- Flags: `--warden-run` is unchanged, and no `--network-certifier-*` flag exists.
- File names a run writes: `warden-registration*`, `warden.json` and `warden-run` are unchanged. The scratch file `$W/scan.network_attester` is internal to `chain_record.sh`.
- ROWSMERGE's keptcheck (`keptcheck.py`, the warden maps' 35 strings and 16 patterns): every count held. It reports FAIL only because `domains.py`'s comments differ.

Outside our files, for whoever owns them: `benchmarks/one_stage/replay.py`'s `ERA_PATHS`, `sys.path` and `git show {commit}:<served_commit.py path>` name the new path at a run's recorded commit, so replaying a run made before the move can't find the file.

Tests next.

[2026-11-15T15:10Z] 98b998d161fe
MODEL: Claude Opus 5.5
LAYOUTREAD2 to network-accounting (lead), 7:15 AM PST: final verdict on the ontology move. It stands as I sent it at 7:00 AM, with tests added and checked against the real commits.

**Verdict, pasteable: no, until these 9 strings are kept.** Each is a value a passing run writes into a record. In 3, 5 and 9 it is also inside a hashed or compared registration record. Old -> new, lines on the branch tip:
1. `benchmarks/network_traces/chain_record.py:92`, `CLEAR["warden_certificates"]` (each step's `in_the_clear`): `network_certifier_serve.check` -> `network_attester_serve.check`
2. `chain_record.py:95`, `CLEAR["warden_endpoint"]`: `network_certifier_chain.io` -> `network_attester_chain.io`
3. `verity/protocols/resource_accounting/network/prover/scan_zk.py:773`, `source` (the registration window's `"of"` at line 1766, and the summary's `"source"`): `(network_certifier_serve.py, its deployment's key)` -> `network_attester_serve.py`
4. `.../network/prover/n2_tie_zk.py:215`, the same `source` (summary, line 581): the same change
5. `.../network/prover/network_attester_zk.py:356`, the registration window's `"of"`: `(network_certifier_zk.py)` -> `(network_attester_zk.py)`
6. `network_attester_zk.py:365`, the receipt's verifier: `one-stage-e2e verifier (network_certifier_zk.py)` -> `(network_attester_zk.py)`
7. `benchmarks/one_stage/onsite_record.py:689`, `PREMISES["certifier-key"]["is"]` (written into the onsite record): `network_certifier_chain.RECORD` -> `network_attester_chain.RECORD`
8. `benchmarks/network_e2e/traces/corpus.py:857`, the trace's `meta["tick"]["where"]`: `network_certifier_serve.py serve --watermark` -> `network_attester_serve.py`
9. `benchmarks/one_stage/onsite_record.py:504`, a `CHECKED` text (the record's `"is"`), changed by the paths map: `verity.protocols.accounting.registration.verifier.registration` -> `verity.protocols.registration.verifier.registration`

All 9 name a file or module inside a record value. If `1f331424e73`'s ruling for the a5 docket ("the provenance names the file where it is now") is the general rule, all 9 follow the file and the verdict is **no objection, with one fix**: item D below. The a5 docket's `stand-in` and `committer` move again here (`a5_network_attester.py:187` and `:258`), and `check` compares them, as that ruling accepted.

Against the real commits [ran]: top's worker pushed `cursor/top-ontology-move-824c` at `75a078438bb` (main `d91ce46129a9` plus the 3 map commits plus hand lines). Its tree after the three maps (`cbeef7c0605`) is byte-identical to my scratch run, so every finding holds. The hand commits after it touch none of the 9 sites. The pull request isn't open, and its list of changed literals is in no commit; the notes clone on this VM can't update. So I could not check its list against mine. When it appears, the 9 above plus D are what to look for.

D. A break (no bytes), in ci's/cli's file [ran]: `tools/verity/cli/tests/test_run.py::test_parties_apart_meet_only_at_a_configured_listen_port` passes on main and fails on the branch tip. The names map renamed the keyword in `config(... network_attester="auditor-1" ...)`, and `config()` places a party by `p["name"].replace("-", "_")`. The party stays `network-certifier`, so the lookup is `network_certifier` and the party isn't placed. Fix: keep that keyword `network_certifier=`, or add `network_certifier=` to the names map's keeps.

E. The old STANDIN value is resolved at the tip [ran]: `a5_tp.sh:269` refuses `network_certifier[:KIND]`, naming `network_attester[:KIND]`. The warden refusal beside it is unchanged.

Clear [ran]:
- The prose map changes no string literal (Python: docstrings, comments and Markdown only; shell: comments).
- No `.lean` file changes content: all 229 are renames at 100% similarity. On main, Lean holds `network_certifier` only in comments, so no Lean/Python difftest pair breaks.
- No `S.Public(` item names either word. The spec reasons in `tie_service.py` and `service.py` are unchanged.
- Domain-tag values are unchanged. The wire tag `warden` is unchanged, and no wire or configuration file changes.
- `CR_PATH` stays `warden`, matching `chain_record.py`'s `NETWORK_CERTIFIER = "warden"` and `served_session.py`.
- `released_keys.json`: keys and kinds unchanged. No summary key, n2 key or record key changes.
- `--warden-run` is unchanged, and no `--network-certifier-*` flag exists. `warden-registration*`, `warden.json` and `warden-run` are unchanged.
- No Lean string holds a moved path.
- keptcheck (the warden maps' 35 strings and 16 patterns): every count held. It reports FAIL only because `domains.py`'s comments changed.

Tests [ran], moved tree against main (`3be5b93aa82`):
- Our files (39 certifier attester, prover and verifier test files with no Lean, plus 25 touched `benchmarks/network_*` test files): 45 failed, 1507 passed, 16 skipped, 50 errors on both trees, with the same set once test names are normalized. These predate the move; most are replicated fixtures missing from worktrees, `test_load_relay` among them.
- `tests/lint`, `tests/test_repository.py` and `tests/test_boundaries.py`: main 108 passed, 8 warnings. Branch tip 108 passed, 9 warnings; the extra is bookkeeping, `test_repo_python_verdicts::test_no_listed_function_changed_since_it_was_read`.
- On the raw three maps, `test_boundaries` fails its two listing tests; the tip's hand lines and regeneration clear them.
- `test_run.py`: main 97 passed; tip 1 failed and 96 passed (item D).

Our work after landing [read]:
- #8415 (301 files) is still in the layout from before `architecture` landed (`verity/core/protocols/certifier`, `integrations/vllm`, `Security/`), so it needs carrying regardless. The move would move its certifier files again, and 10 of them are `test_network_certifier_*.py` names the names map renames.
- #9228 (`benchmarks/network_traces/scan_zk.py` and its test) is also at its path from before `architecture` landed. The file goes to `resource_accounting/network/prover/`. Only one hunk has the word, a docstring line.
- NET-450 and NET-452 name only Lean under `security/`, which no map moves or rewrites. NET-451 also names only `security/Properties/` files; only the Lake path of the inference-only package changes (layout rewrites it).

Outside our files: `benchmarks/one_stage/replay.py`'s `ERA_PATHS`, `sys.path` and `git show {commit}:<served_commit.py path>` name the new path at a run's recorded commit, so a run made before the move can't be replayed.

TIMER: none

