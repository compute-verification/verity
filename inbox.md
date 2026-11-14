[2026-11-14T04:05Z] c8af1eb164af
NAMES, Fri 13 Nov, 8:05 PM PST. Deliverable 1: Monday's sequence, read from today's tools (main 6953e38e55a4).
Context is under three quarters; I started. Handoff note so far: training-20261114T0404Z-handoff-names-monday-rename.

## 1. The changes, their order and kinds

Shape: one pull request, a chain as #9697 and #9734 landed: kept commits, then one generated commit a map, then kept commits, then the lock records.
- READ (`tools/research/src/research/queue.py`, `_regenerated`; `vqueue_loop.py`, `World._move`, `_uncertified`): the queue certifies by itself only a PR whose body has one `Move: tools/move/NAME.toml` line, by running `rename.py` on that one map. It never runs `layout.py`, and it takes one map a PR.
- INFERRED: so Monday's rename cannot go as "a move the queue certifies": it has two path maps and twelve moves. It lands as an ordinary PR with a land verdict (it changes Lean, locks, Lake files and a `TRUSTED.sha256`, all in drive.toml's `lowrisk_excludes`). Its body must hold no line that starts `Move: tools/move/`, or the queue reads it as a one-map move.
- READ (`tools/move/README.md`, `stage.py`): each generated commit is made and certified by `stage.py` (`move_check.py PARENT COMMIT`, then the map run again). The reviewer repeats `verity check move PARENT COMMIT` on each.

Order, as rehearsed tonight (RAN on my VM, main 6953e38e55a4 merged into #8449's branch, 17 maps, about 20 minutes for the whole staging):

| # | change | kind | tonight |
|---|---|---|---|
| 0 | the 17 maps in `tools/move/` | ordinary (on `main` first by #8449, or the chain's first commit) | `tools/move/tests/test_maps.py` 2 passed |
| 1 | `inference_only_check_tests_map.toml` run | ordinary, `rename.py --map` | 1 file, 3 rows; run again, nothing |
| 2 | `bracket_map.toml` | move, word map | certified: 1 rename, 14 path-only edits |
| 3 | `fresh_input_map.toml` | move, word map, Lean names | certified: 57 path-only edits |
| 4 | `theta_map.toml` | move, word map, Lean names | certified: 53 path-only edits |
| 5 | `inference_only_map.toml` | move, word map, Lean names | certified: 1 rename, 395 path-only edits |
| 6 | `slowdown_map.toml` | move, word map, Lean names | certified: 40 path-only edits |
| 7 | `inference_only_prose_map.toml` | move, prose | certified: 41 edits |
| 8 | `dampener_prose_map.toml` | move, prose | certified: 15 edits |
| 9 | `inference_block_prose_map.toml` | move, prose | certified: 41 edits |
| 10 | `fresh_input_prose_map.toml` | move, prose | certified: 41 edits |
| 11 | `slowdown_calculator_map.toml` | move, path map (`layout.py`) | certified: 8 renames, 96 path-only edits |
| 12 | `slowdown_prose_map.toml` (main's) | move, prose | certified: 9 edits |
| 13 | `inference_only_check_map.toml` | move, path map: the flatten | certified: 37 renames, 61 path-only edits |
| 14 | `inference_only_ids_map.toml` run | ordinary | 38 files; run again, nothing |
| 15 | `slowdown_calculator_names_map.toml` run | ordinary | 39 files; run again, nothing |
| 16 | `slowdown_python_map.toml` run | ordinary | 33 files; run again, nothing |
| 17 | `slowdown_harness_map.toml` run | ordinary | 8 files; run again, nothing |
| 18 | hand commits (below) | ordinary | not yet rehearsed |
| 19 | pins: `tools/pous_grader/TRUSTED.sha256` by its `sha256sum` line; the three Lean-built fixtures by their `generate.sh` | ordinary; the fixtures build Lean | RAN `pins.py --stale` at the rehearsed head: exactly these four are stale |
| 20 | lock records: `check.py --update --moved lean-moves.json` on `security` and on `security/Properties` | the records commit, ci's slot | not run, by your order |

Hand commits of row 18 (READ on the branches; none applied yet): RENAME's part E from e034ac9f528b (the formulas 24ea56efad2, the Glossary's older names 2325de7c485, the retired-terms rule 46c5e1fd8f5 and its toy e034ac9f528); ν in `circuit_properties/work.py`; the harness's formulas 11f028250ff, ν 88f19f0a479 and the rewrap; your d7c997d0e27 (`benchmarks/README.md`'s row) and 19bf7263f75 (the exfiltration print); `NO_VERDICTS` by running its test. RENAME's e0aa56f02cb edits `move_check.py`'s docstring example: ci's file, I leave it and list it.
- READ (README, "Staging a move"): after each word map's commit, whoever stages runs `python3 tools/move/resolve_check.py PARENT COMMIT` and reads each issue before the Lean build. My script does; I have not read the outputs yet.

## 2. What must be on main first

- READ (`stage.py`'s docstring; `move_check.py` reads the word maps at the move's parent; `restack.py` reruns a move from its commit's parent): a map must be in `tools/move/` at the commit its move runs on. It need not be on `main`. Chains 1 and 2 carried their maps as kept commits.
- READ (`World._uncertified`): only a move that lands on its mark alone needs its map on `main`. Monday's does not land that way.
- INFERRED: so nothing must land before Monday. #8449 on `main` first is a convenience: the chain is then `stage.py … --base origin/main` with one kept commit before the moves. #8449 holds only `tools/move/*.toml`, which `lowrisk_excludes` does not name, so it lands on its mark alone.
- Not known: the queue's mark-to-main time today. `research queue status` did not answer in two minutes on my VM. With your own figure for Monday (mark at noon, main at 3 PM), a mark by Sun 15 Nov 9 PM PST leaves room for one ejection. I would mark it only after Sunday's rehearsal on that day's `main`, so `main` holds the maps as rehearsed. If it is not on `main` at 6 AM, the chain carries the maps and nothing else changes.

## 3. Where Tuesday's plan no longer holds

- RAN: fifteen maps became seventeen. Gone: `slowdown_calculator_terms_map.toml` (its move ran on `main`). New from `main`: `slowdown_python_map.toml`, `slowdown_harness_map.toml`. Twelve moves and five ordinary runs, where Tuesday had eleven and four.
- RAN: `main`'s `slowdown_prose_map.toml` and mine of that file name are different maps. Main's rewrites prose that called the calculator rho ("the rho model", "rho has"). Mine renames "compute dampener" to "slowdown" in prose. Main's keeps the name; mine is now `dampener_prose_map.toml`, same rows. The other three of the four that conflicted (`bracket`, `slowdown_calculator`, `slowdown_calculator_names`) say the same as mine did, in `main`'s reviewed form: main's names map has the rows mine added on Tuesday (`rho.md`, `rho.json`, `rho/params.py`, "rho's" with `benchmarks/property_redteam/**` kept) and more (`declared_rho`, the quoted key `"rho"` in three files, the six module names, `benchmarks/training_slowdown/**` kept).
- RAN: the paths. Every map on `main` and on my carrier named a tree that is gone. I restated each path across the 31 moves `main` landed after the layout (`regen.py` with each move's map), took your two maps for B and C from `cursor/slowdown-path-rehearsal-4207`, and restated the paths in `main`'s other four by their directories (`verity/protocols/inference_only/{nci,rho,slowdown}` are `benchmarks/…`; the remote verifier's README is `verity/protocols/compose/verifier/remote/README.md`).
- RAN: the flatten. The check is `verity/protocols/policy/inference_only/verifier/nci/` (its README and `lean/`, 37 files). `inference_only_map.toml` names it `verifier/inference_only/`; `inference_only_check_map.toml` then moves it into `verifier/`, beside `__init__.py`, `parties.py`, `pyproject.toml` and `tests/`. No name collides. The policy's Lean is then `verity/protocols/policy/inference_only/verifier/lean/`, as a protocol's `verifier/` holds its `lean/`.
- RAN: the flatten needed two more rows before the moves. The check's README names `../parties.py` and `../tests`, which the move writes as `parties.py` and `tests`, and `move_check.py` refused that ("edit beyond moved paths and module names"). The README's ordinary map now names both from the root. Wording for you to judge.
- RAN: a new `[keep]` pattern in `inference_only_map.toml`. `infra/ops/tests/test_control_door.py` writes a line break and then the role ci inside a string, twice; the map read that as the lowercase name and broke both strings. The pattern keeps them.
- RAN: `tests/import_order.toml`. After all seventeen maps, `tools/verity/check/import_order.py` prints "tests/import_order.toml: the same". Your calculator map has no `[tokens]` row and needs none.
- READ: the lock update. `rename.py --moves` on one map prints prefixes, and the maps run one after another, so the file for `--moved` is written once, from the four name maps as one map, on the commit before the moves: 719 entries (717 on Tuesday). It is `lean-moves.json`; I keep it with the rehearsal.

## 4. Stored keys, digests, columns and other lanes' readers that hold a renamed name

From the rehearsal's diff (RAN: 522 files; 285 renamed, 201 changed, 18 added and 18 deleted as pairs). None is renamed outside the rehearsal. Each needs its reader's word:

| what | where | renamed how | who reads it |
|---|---|---|---|
| lock records | `security/lean-audit.json`: 106 of 1,136 guarantees and 650 read definitions move; `security/Properties/lean-audit.json`: 23 of 540 and 494. 984 distinct names | `--moved`, rows 20 | lean; ci's slot; Daniel's DM on a changed record |
| the check's own lock | `…/policy/inference_only/verifier/lean/lean-audit.json` | by text, four rows of the ids map (`NciCheck`, `Nci.Complies'`, `nci_reference`, `Baseline`) | lean, `tools/verity/lean/check.py` |
| Lake names | package `nci_check` is `inference_only_check`, library `NciCheck` is `InferenceOnlyCheck`, eight executables `nci-…` are `inference-only-…` (`nci-check`, the seven difftests) | maps | the five packages that require it (their lakefiles and manifests change); ci's Lean build cache, keyed by package; the network certifier's difftests |
| four pins | `tools/pous_grader/TRUSTED.sha256`; `randomness_lean.json` (Flock's prover tests); `exfiltration_lean.json`, `exfiltration_two_lean.json` (SPOT's verifier tests) | stale after the moves; regenerated, row 19 | proofs (the grader), lean (the three fixtures' `generate.sh` build Lean) |
| `NO_VERDICTS` | `benchmarks/harness/tests/test_exfiltration.py` | recomputed by running the test | the harness's owner |
| the registry's paths | `.agents/skills/using-slack/registry.json`: three paths under `security/…/Nci` | maps | infra and ci (it is a `SIGNED_GUARDS` file; `signed_verdicts` is false today) |
| the export's boundary | `tools/verity/check/public_boundary.toml`, five lines | maps only, no hand edit | top; deleted by #9876, which is open, not on `main` |
| recorded moves | `tools/research/src/research/store/moves.json` | each move appends its paths | infra (the evidence store's path mapping) |
| caps and allowlists keyed by path | `tools/verity/lean/unread-caps.json` (`Proofs/Nci`), `tests/lint/allowlists/{lean_bar,repo_env_reads,repo_retired_terms,repo_ruff,repo_silent_excepts}.json` | maps | lean; ci (the ratchet gate) |
| the calculator's outputs | files `rho.md`, `rho.json` become `slowdown.md`, `slowdown.json`; the JSON keys `"rho"` and `declared_rho` become `"slowdown"` and `declared_slowdown` in `run.py`, `figures.py`, `test_analysis.py` | `slowdown_calculator_names_map.toml` | whoever publishes from the calculator's JSON: not known to me. In the tree the board keeps its key (`"rho": slowdown` in `bracket.py`) and the harness keeps its keys (`[keep_in]`) |
| the grade's rows | `benchmarks/training_slowdown/…/grade.py`: `"rho"` is `"slowdown"`, the row head `rho` is `record's slowdown` | `slowdown_python_map.toml` | a reader of the printed grade outside the tree: RENAME did not know of one, nor do I |
| the board's data | `benchmarks/nci/arguments.json` (8 lines: Lean names and a Lean path in strings), `rows.json` (1 line) | maps | training; whatever publishes the board |
| suite names | the calculator's suite is named by its path, `benchmarks/rho` | the path map | ci (cache keys); the evidence store's verdicts by suite. `policy.toml`, `ci.toml` and `drive.toml` name none of the renamed names (RAN: the rehearsal does not touch them) |

Kept by the maps, so not renamed: the board's directory and package `nci` and its ids `nci-…` (`[keep_paths]`, `[keep_in]`); the calculator's family kind `nci` and `families.Nci`; `"rho"` as the board's and the harness's stored key; `tests/tree.toml` and `moves.json`'s older rows.

## What I do next

By Sat 6 PM PST: #8449's branch pushed with the seventeen maps (merge of `main`, no rebase), the glob counts (tonight: 31 entries match nothing, all `[rewrite]` excludes or includes such as `archive/**` and `**/CHANGELOG.md`, and one kept string of `main`'s Python map; no scoped glob is stale), the hand commits on the rehearsal, the named tests on my VM, and the scratch branch.

