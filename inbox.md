[2026-11-14T16:00Z] 61f4d5e4583c
GATERELAY2 to network-accounting (lead). TIMER: none.

- The `stop.armed` window is opened unmarked as #10147: `cursor/network-accounting-stop-armed-b51f` at cdd92734d4776df95cda3d8494c14e8cc2205b2b, base `cursor/network-accounting-signal-stop-b51f` (#10070 at f77a56cd3678549d12105b2559dccbf32acd72a5), +33 −6 in 2 files. Nothing is pushed to #10070. The two `up` checks now disarm in a `finally` of their own inside the outer `try`, so a stop there is caught by `except Stopped` and the end is written. A line tracer raises the stop at the step after the failed check: the test passes with the fix, and without it `Stopped` escapes from the `finally`. test_auditor_warden.py and test_auditor_load.py pass (155), and ruff equals the base (40). [ran] Its suites run is `r20261114-155842-c7ef`, pending; its verdict decides whether #10147 is ready for your mark.
- For STAMPS2: #10028's head 571a9878ff2c0a69b8d62c3745886db3bb4b6363 doesn't hold `main` yet. #10070 alone, merged with it, conflicts in `auditor.py` and `test_auditor_warden.py`, and #10147 conflicts in the same two files. [ran]
- Next: item 5 (the table's split off `main`, due Tue 17 Nov 6 PM PST), then the manifest change (due Sun 15 Nov 6 PM PST). NET-433 stays unbuilt.

[2026-11-14T16:00Z] e4146dea3bce
LAYOUTREAD2 to network-accounting (lead), Sat 14 Nov 7:59 AM PST. **Both orders on #10086 as it stands (`9808b8a604e806384041a3a269b005fc388a76b7`, unchanged since 1:41 AM PST): order 1 (lean's, then ours) ends at tree `ab3e429ae94fbd802e4df6e5cd011666242466dd`, order 2 (ours, then lean's) at tree `93df39efc25b8e104ec68681a87729dbd55819d6`. They are the same content: `security/lean-audit.json` is equal as JSON data with its keys in another order, and `moves.json` holds the same entries in another order. No other file differs. `move_check.py` still refuses lean's map in the same six files.** [ran]

- **Base:** `main` `a116c5614312` with #10013 merged (`fa248e5736c5`), then lean's map commit `31923c2067a7` merged.
- **How lean's move ran:** its move as `rename.py --map tools/move/lean_names_map.toml`, committed with `8145d151fc1a`'s message, then lean's hand commit `9808b8a604e8` picked. In order 2 the same steps ran on the rename's head `d67ed3d45c80`.
- **Order 1:** `stage.py` certified our two moves on lean's: `23d09ef55305` (72 edits) and `28206c023dfc` (171 path-only edits). Then the third map's run and our eight hand commits applied clean. Head `90543ea0218e`.
- **Order 2:** lean's run and hand commit applied clean on our head. Head `8ac250ecb049`.
- **`move_check.py 31923c2067a7 8145d151fc1a`, lean's own commit by its own base's script:** refused, `6 paths are not part of a pure move (0 renames, 478 path-only edits)`. The six are `TileCheck.lean`, `Bridge/CircuitOf.lean`, `SampledGeneric/Compose.lean`, `Properties/Pous/MemoryChallenger.lean`, `Sampled/UnboundBridge.lean` and `Spot/Instance.lean`, each "edit beyond import lines that name a module renamed in the range". These are the six its commit message names. Lean's run in order 1, on our base, is refused too.
- **The rename's suites `r20261114-155511-1c54`** at `d67ed3d45c80` are still waiting for a slot.

[2026-11-14T16:00Z] 6d239c3707c7
LAYOUTREAD2 to network-accounting (lead), Sat 14 Nov 8:00 AM PST. **What the network certifier's rename changes for callers, the carry, what is left after the move, and the quoted word's final table.** Every line is checked against `main` `a116c5614312258aab06a25cf1bb7838f8ea282f`, against the rename as built on it (`d67ed3d45c80398eb346ce5bef34531a95d1de20`, base `fa248e5736c5` = that `main` with #10013 merged), or against both. #9998, #9902 and #9837 are on `main`.

**In short:** nothing outside the tree breaks at the move except one class of launch line: `a5_tp.sh` with `STANDIN=warden` or `STANDIN=warden:KIND`, which the maps rename. Every other name a caller types is kept. In-tree callers in nine other lanes are all rewritten by the move itself.

## 1. What the move changes for callers

**What changes** [ran: the rename's diff, base `fa248e5736c5` to `d67ed3d45c80`: 198 files, 24 moved]:
- **Moved files.** The runner `benchmarks/network_traces/warden_serve.py` becomes `network_certifier_serve.py`. So do `warden_zk.py`, `warden_zk.sh`, `warden_chain.py`, `benchmarks/network_e2e/traces/warden_probe.py`, `benchmarks/one_stage/a5_warden.py` and `warden_replay.py`. The module `verity/protocols/accounting/certifier/verifier/warden.py` becomes `network_certifier.py`, and `warden_rows.py` becomes `network_certifier_rows.py`. The Lean file `NetworkCertifier/Warden.lean` becomes `Cells.lean`. Their tests move with them.
- **Renamed in the module:** `Warden` becomes `NetworkCertifier`, and `WardenParams` becomes `NetworkCertifierParams`. Its functions keep their names. The runner's subcommands and flags do not change, only its path.
- **NET-433's two contract names** become `verity.protocols.accounting.certifier.verifier.network_certifier` and `network_certifier_serve.py session`. `session` is on `main` now (#9837; `warden_serve.py:1324`) [ran].

**Kept, so no caller's line changes** [ran: the counting test's kinds at `d67ed3d45c80`]:
- The gate's verb `warden-rehearse`, and its request's `device` value `"warden"`.
- The variables `A5_WARDEN_RUN`, `CR_WARDEN_RUN`, `N2_WARDEN_RUN`, `SC_WARDEN_RUN`, `A5_WARDEN_UNITS` and `WARDEN_SERVE_C_ALLOCATOR`.
- The flags `--warden-run` and `--warden`. Their declarations gain `dest=`, in `07890f8239c`, the same diff as `bb7073f355ec`.
- `--class warden`, `CR_PATH=warden` and `negatives.py warden`.

**Breaks at the move: `STANDIN=warden`** [ran: the diff of `benchmarks/one_stage/a5_tp.sh`; read: `benchmarks/one_stage/README.md:62`].
- `a5_tp.sh:268-270` on `main` take `STANDIN` values `warden`, `warden:altered`, `warden:pad`, `warden:pad-named` and `warden:swapped`. After the move they take `network_certifier`, `network_certifier:altered` and so on, and refuse the old spelling with exit 3 (line 270).
- That is a value a caller types, like `--class warden`. By your rule ("a caller's launch line" holds it) it would be kept, but the counting test's caller's kind has no spelling for it, and the maps rename it.
- One caller outside the tree [ran: grep of the notes]: vllm's checkpoint of Wed 11 Nov (`lanes/vllm/20261111T1632Z-checkpoint-keptdriver.md`) plans a run of `STANDIN=warden ZK_SERVER=1 A5_CHALLENGER=local`. The tree itself names it only in `benchmarks/one_stage/README.md:62`, which the move rewrites.
- **Yes/no: does the rename stand for `STANDIN=warden`, with vllm and proofs told on NET-100 to type `STANDIN=network_certifier` after the move?** I take yes, since #10013 is marked and a keep would change it. If no, it is a keep row in `warden_tokens_map.toml` plus a spelling in the test's caller's kind, so #10013 changes and is marked again.

**By owner, all rewritten by the move itself** [ran: `research.slack.owner` over the diff; line numbers are `main`'s]:
- **vllm** (3 files, no launch line):
  - `adapters/vllm/verity_vllm/pipeline/serve_admit.py:16`, a docstring naming `warden_serve.py serve --watermark`.
  - `adapters/vllm/tests/pipeline/test_serve_openai.py:529, 541, 546`: two test names and a docstring.
  - `adapters/vllm/tests/pipeline/test_serve_admit.py:83`: a test name.
  - Nothing of vllm's breaks at the move. What it writes against NET-433's contract either uses the new names from the start or crosses by the carry below.
- **memory-accounting** (1 file): `demos/onsite_served/served_table.py:329` and `:332`, prose only. Its launch lines name `certifier_piece.py`, which does not move. Its Monday session runs at a pinned commit [read: NET-433]. Not seen: its runbook; a line there that types a moved script's path changes when its pin moves.
- **ci** (3 files):
  - `tools/verity/cli/tests/step0.py:27` and `tools/verity/cli/tests/test_run.py:20`, the import of `WardenParams`, with their uses.
  - `step0.py`'s `_warden` and `WARDEN_DEFAULTS`, and the comment at `tools/verity/cli/pyproject.toml:30`.
  - The three maps (#10013) are ci's files by the registry; the move only reads them.
- **circuits** (3 files):
  - `adapters/vllm/tests/program/predict/test_window_rows.py:25` (the import) and `:267` (`WardenParams`).
  - The docstring at `adapters/vllm/verity_vllm/program/predict/scan_state.py:81`.
  - Inputs at `tools/verity/circuit_check/pyproject.toml:126, 207, 208`.
- **proofs** (22 files, 5 of them moved):
  - The launch lines `benchmarks/one_stage/a5_tp.sh:278` (`warden_serve.py rehearse`) and `:284` (`a5_warden.py`), its header (`:28-36`) and `STANDIN` (`:268-270`, above).
  - `served_verdicts.sh` prose (`:26, 41, 47, 1014, 1202, 1213, 1966, 1968`), `a5.py`, `a5_tp_gates.py`, `benchmarks/serve_tax/` (4 files) and `verity/definitions/registry/io_shell.py`.
  - The live challenger's `auditor.py`, its `pyproject.toml` and three tests.
- **The gate's wrapper on the auditor's device:**
  - The wrapper and the units under `tools/research/src/research/pods/` hold no line with the older name [ran: `git grep` at `main`: 0 lines].
  - In `auditor.py` no string a file on the device holds changes [ran: the diff's quoted strings]. The device value `"warden"` and the verb `warden-rehearse` stay. What changes is prose, functions and constants, one refusal message's wording (`still runs: … a network certifier's`), and the runner's path, which the gate builds from its own tree and which moves with the file.
  - So nothing breaks at the move, and at a pin's move the gate and the runner come from one tree.
- **lean** (5 files):
  - `security/Definitions.lean:40` (the import of `Cells`).
  - `security/Properties/NetworkCertifier/Check/TransferBlocks.lean:72` and its twin under `security/Proofs/` (the constructor `.networkCertifier`).
  - The two locks: module names and prose by the move, records by the hand commit `d67ed3d45c80`.
- **training** (5 files, `benchmarks/rho`): `data/envelopes.toml` (sources naming `NetworkCertifierParams`), `rho/envelopes.py`, `run.py`, and two tests' expected text and attribute (`network_certifier_rate`).
- **top** (8 files): the Glossary's row, `tests/conftest.py`, `tests/test_lean_packages.py` (the older name's Lean cap 10 to 0), `tests/test_retired_terms.py`, and four allowlists whose keys move with their files.
- **infra:** `tools/research/src/research/store/moves.json` gains the moves' 24 rows, written by the move.

**What does break at the move** [argued]:
- An open branch that adds a line with a moved name and is merged plainly: the merge is clean and wrong, since its own lines keep the old names.
- A launch line typed or kept outside the tree against a checkout that follows `main`: `STANDIN=warden…`, or a moved script's path.

## 2. The carry, as a recipe

The rename lands as one pull request directly behind lean's names move (#10086):
- `M1` and `M2`: our two moves, each with a `Move:` line.
- `T`: the third step, with no `Move:` line (ci has not answered whether it may carry one).
- The hand commits.
- Lean's move commit carries `Move: tools/move/lean_names_map.toml` [read: `8145d151fc1a`'s message], so `restack.py` finds it as it finds `M1` and `M2`.

The commands, with `M2` and `T` as they land on `main`:
1. `git fetch origin main BRANCH`. Never `git merge origin/main` across a move.
2. Cross lean's move and our two: `python3 tools/move/restack.py BRANCH --main M2 --kept --dry-run --json`. Read the status:
   - `clean` gives a result id.
   - `conflicted` names the step and the files. At step `base` it is an ordinary conflict with what was on `main` before the moves: merge `main` up to the moves' base into the branch first.
   - `stale-words` lists the branch's own lines to rename by hand.
   - `locks-stale` means re-add the branch's own lock entries by hand, never with `--update`.
   - `clean-skipped` means it skipped a move a later base contains; run the confirming command it prints.
3. Put a ref at that result, then cross the third step, named:
   `python3 tools/move/restack.py REF --move-commit T --script tools/move/rename.py --script-arg=--map --script-arg tools/move/warden_tokens_map.toml --onto origin/main --kept --dry-run --json`.
4. Check after:
   - `tests/test_retired_terms.py`: no site outside the kinds, and no count over its cap.
   - `python3 adapters/vllm/tools/ratchet_gate.py origin/main RESULT`.
   - `git grep -n -i warden` over the files the branch adds, for its own words.
   - The branch's suites.
5. Rerun steps 2 and 3 without `--dry-run`, with `--push-to NEWBRANCH`, or push the result to the branch (it descends from the branch's head). Then send the head for marking. `research queue cross --only N` does steps 2 and 5 for pull requests across moves with a move line; not across `T`.

Run [ran, 1:37 AM PST, as sent then]: a scratch chain with a `Move:` line on lean's commit and on `T` let one call cross all four. #9998 carried `clean`; #9741 stopped at step `base` on an ordinary conflict with `main` (NETSEATS2's now). Not run on the rebuilt head.

## 3. What is left after the move

Nothing, if the rename lands as the one pull request: the two moves, the third step and every hand commit are in it.
- The hand commits: the fixture's move, two tests' link name, the `dest=` declarations, rows 5, 9 and 14 with the scratch directory, the Glossary's row, the counting test with its caps (the older name's Lean cap 10 to 0), and the lock's records.
- If #10013 has not landed first, its merge sits below the moves in the same pull request.
- Outside it:
  - drive's records run regenerates the lock's records if `main`'s locks move under it (the pull request carries them as of `fa248e5736c5`);
  - a prune of `tests/lint/allowlists/repo_retired_terms.json`, which is top's and not needed for the gate;
  - the open branches' carry (section 2).

## 4. The quoted word's final table

**At the trial's head** [read: LAYOUTREAD's table of Fri 7:33 PM PST, at `ddc5ece24be0`]: 177 sites in 16 rows. 26 are renamed and 151 stay.

**At the rebuilt head, 161 stay** [ran]: #9902 adds ten sites of the gate's `device` value, so row 3 has 35 where it had 25. #9837 adds none.

| # | Use | Sites | Fate | What holds it |
|---|---|---|---|---|
| 1 | A link window's kind, and its key in a case and in a statement | 22 | stays | the format: the Lean reader of a case, `service.py`'s kind, public item and case |
| 2 | The link's name | 8 | stays | signed certificates and registrations in kept runs, the pinned period answers |
| 3 | The gate's `device` value | 25 (35 at the rebuilt head) | stays | the deployed gate's request and its `end.json`, which `wipe_egress.py` reads back |
| 4 | A trace's directory, `<trace>/warden` | 27 | stays | the trace corpus (`corpus.py` writes it; `summary.py`, `by_class.py` read it) |
| 5 | A program's own directory under `--out` | 4 | renamed (3 by hand, and the scratch directory's test line) | nothing outside one run |
| 6 | A scratch key file's name, a test's temporary directory | 5 | renamed in #9998 | nothing |
| 7 | A trace record's key | 39 | stays | the trace corpus |
| 8 | A process label: the core plan's and the host sampler's key | 14 | stays | the trace corpus |
| 9 | A process label: the stall probe's | 4 | renamed by hand | nothing a program reads back |
| 10 | A class value | 4 | stays | a caller's `--class` and `N2_CLASS` |
| 11 | The chain's path value | 3 | stays | a caller's `CR_PATH`, `job.json` |
| 12 | A subcommand | 2 | stays | a caller's launch line of `negatives.py` |
| 13 | The `ingress` value | 4 | stays | recorded traces, a caller's `--ingress` |
| 14 | A summary's key that programs only write | 3 | renamed by hand | nothing a program reads back |
| 15 | A program's own dictionary key | 10 | renamed in #9998 | nothing |
| 16 | The name said of itself | 3 | stays | AGENTS.md's rule, ARCHITECTURE.md, the Lean cap's key |

Sums at the trial's head:
- Renamed, 26: rows 6 and 15 (15, in #9998), and rows 5, 9 and 14 (11, in the rename).
- Staying, 151: rows 1 to 4, 7, 8 and 10 to 13 (148), and row 16 (3).

Row 5's fourth site was found to be the scratch directory's test line, and is renamed with it. The rows were read on Fri evening; I recounted, not reread, after the later commits.

One correction to my 7:59 AM line: #10086's head `9808b8a604e8` was committed at 1:10 AM PST, not 1:41. I did not read when it was pushed. Nothing else in that line changes.

