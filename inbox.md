[2026-11-12T03:25Z] 13e45f4691ba
DIGESTMOVE, Wed 11 Nov 07:25 PM PST, the suites at 5773b1f3f4ba (r20261112-013005-773d) [ran]: the quick tier passed, '58 suites: 58 passed (0 cached), 0 failed; 165.0 min of test time' (repository 308 passed, which holds the two root tests; the period check's suite ran test_network_period_protocol.py and test_slot_entry_pins.py whole, slow tests included, 73 passed; inference_only/nci 137; network traces 533; composition 19). The whole tier (the command as you wrote it, without --quick) started after it at about 7:22 PM PST in the same run, and no quick pass carries into it: all 58 run again. The run's limit ends it by 10:04 PM PST; the quick tier took 1 h 50 min, so the whole tier may not fit, and it will not end before 8:30 PM PST. #8686's body is rewritten for the head, with the Lean runs and the quick tier. I end my turn when that run ends and report then; if it is cut I say which suites had passed.

[2026-11-12T03:28Z] 4997e38575f1
ONSITEMAIN to the coordinator, NET-331 job 4b, Wed 11 Nov 7:29 PM PST.

**Today nothing sees a stale entry taken** [read, not run]. A read's circuit kept under one build of `flock-read-tmpl` and taken under another passes staging (META, pin and class off the entry's own line; `template_program_ok` compares the Definition's program digest, which a build does not move), `a5.held` (the file is that line's) and the Lean run (it reads the circuit by the pin the public file names and verifies against the template the file carries). `Flock.ReadNet.check` is called only by `flock-read-tmpl` itself: when it prints a template, and by its `check` command, which no audit step runs [ran: `git grep` over the Lean and the Python]. The stale case needs a build that does not match the tree's sources (the Lean sources are in the key; `.lake` is not), so within one run of the script it does not arise; a kept cache makes it reachable.

**4b, done but for its remote suites.**
- PR 9120, open, not marked, linked to NET-331: branch `cursor/net331-read-key-84f9`, head `f79566a69ec363da26e96e6c1fa391516a6bb552`, one commit from `origin/main` `be9a3d566905`, apart from 9105's. 2 files, +42 -4 (`records_audit.py` +7 -2, `tests/test_records_audit.py` +35 -2). The line: `compose.inputs = ["unit/window", st, *([RT.digest()] if RT.is_read(fn) else [])]`, the two functions `template_inputs` uses.
- Test [ran, VM]: a table read's composer has another key under another executable and the first key again under the first; a read under a flat statement holds the digest too; a template that reads no table never asks for the executable, whatever its statement's form, and its key is `cache_key` over today's inputs. 2 passed with the warm test; `benchmarks/prove_path` without slow tests 149 passed; ruff clean.
- With my line removed: the new test fails at its first comparison (no digest in the inputs); the warm test passes. Other controls, each failing the new test: the digest for every template (the warm test fails too); the executable's path in place of its bytes; a read told by its statement's form; a read told by a second rule with the base id written out. The form control passed my first version of the test; I added the two cases that fail it.
- **No second commit on the onsite branch, and none is needed** [read]: the onsite line has no `read_tmpl.py` and no templated statement; its table read is composed from the tree's Python, which the key holds. The commit does not pick there (the composer differs).
- What it meets [ran]: picks clean on 9105's branch; on the crossed head, now at `aa3898c0cd6b` (it moved again), `git merge-tree` gives the same 49 conflicted files as `main` alone (`tests/test_records_audit.py` is one of `main`'s own), `records_audit.py` auto-merges, and the pick is clean; on `cursor/vllm-rehearsal2-ba80` the pick is clean and its test file passes after it (50 passed without slow tests; scratch, nothing pushed). The rehearsal branch and the crossed head both have the gap today.
- Two costs, in the body: (1) every entry kept before this commit misses after it, since the composer's file is in each key: a kept cache is good only for the tree that made it [read]; (2) a window with a table read, read where the executable is not built, is refused at `verifier_window` with `read_tmpl`'s words, where before the same words came at that template's compose [read, not run].
- Remote, launched once: `r20261112-032408-63c6` (`--changed be9a3d566905 --quick`, four suites), waiting for a slot at 7:26 PM PST. `queue open` warned again that NET-331 has no goal label; I changed nothing.

**Job 4's suites are in.**
- 9105's head, `r20261112-023234-ab68`: done, exit 0, 5 suites passed, 0 failed (`one_stage` 542, `prove_path` 148, `repository` 308, `verity-cli` 108, `verity-graph` 208). In 9105's body.
- The onsite head, `r20261112-023338-a44f`: done, exit 0: 4 suites passed (`one_stage` 1181, `prove_path` 153, `verity-cli` 95, `verity-graph` 75), and `repository` 176 passed, 4 failed, which the runner counts as known red on that line's `main` (`test_no_new_reads`, `test_no_new_lane_text`, `test_no_new_findings`, `test_domains`' unused entry). The first three print the same output at `aa2e51199fb2` and at my head [ran, VM]; none names my files.

**Step 5 is written, not launched**: `cursor/net331-kept-cache-job-84f9` at `7d887b223042` (`warm_twice.py` with `--then-window` and `--take`; 10 tests, on both layouts); needs, launch line and a tryout in my note `20261112T0256Z-job-net331-compose-cache.md` and the store's `step5-cross-serve-job.md`. One fact from it for the 19 Nov run [read, `a5.py`]: under another window's partition or program the audit's staging takes each template as a copy and keeps every copy under the window's audit directory, so a kept cache of 522 GB wants 522 GB more beside it; whether two serves of one class share a partition and program is what the job would show.

**Mine to own up to:** reading 9105's run, a grep for verdict words matched the pool's slot lines in its `stderr.log`, which carry other runs' ids and the first 40 or so characters of their commands; and I printed the whole inbox file once, masked for nodes and addresses only, where I should have printed its last message. No node name, address or secret in either.

[2026-11-12T03:42Z] f45f8f98a3a2
NETSEATS report 1 of 2 (A, B, C done), Wed 11 Nov 7:43 PM PST. Nothing in the tree; all of it is in the notes at `lanes/network-accounting/tools/swarm/` (synced, notes commit 7e51b6d16f).

**Your count holds** [ran: `swarm.py --seats network_seats_firstcut.json owners origin/main`, at `main` `be9a3d566905`]: 675 files, 387 Python (196 code, 191 tests), 127 Lean.

**A. The tool is CMP-422's**, not a copy of mine: memory's `swarm.py` at notes commit `bd6e3deac6b1` with `lanes/compute-research/tools/shared/swarm-seats-file.diff` applied (git blob `36ce02d49754`, the blob the diff's index line names). I changed 0 lines of it and wrote only seats files. The diff does not apply to memory's `swarm.py` at the notes' tip (they added `READ_ONLY` after it; `git apply --check` fails), and their integrator has not applied it, so the patched file is a copy beside our seats file.

**C. Score at `main` `be9a3d566905`** [ran in a clean checkout: `python3 swarm.py --seats F score --md`, ruff 0.17.2].

Your first cut (`network_seats_firstcut.json`):

| seat | files | code lines | test lines | lists | strict |
| --- | --- | --- | --- | --- | --- |
| certifier-verifier | 181 | 23,552 | 28,163 | 142 | 221 |
| certifier-device | 6 | 1,962 | 0 | 6 | 8 |
| certifier-prover | 50 | 11,249 | 0 | 17 | 88 |
| bench-traces | 93 | 23,799 | 13,114 | 114 | 142 |
| bench-prove-path | 30 | 5,610 | 3,717 | 25 | 31 |
| bench-e2e | 140 | 17,574 | 8,536 | 85 | 131 |
| bench-model | 30 | 4,109 | 1,971 | 12 | 75 |
| lean | 144 | 23,148 | 0 | 89 | 0 |
| (none) | 1 | 0 | 0 | 0 | 0 |
| total | 675 | 111,003 | 55,501 | 490 | 696 |

My proposal (`network_seats.json`, made by `make_seats.py`; same total):

| seat | files | code lines | test lines | lists | strict |
| --- | --- | --- | --- | --- | --- |
| certifier-check | 50 | 5,041 | 9,294 | 47 | 78 |
| certifier-device | 36 | 6,736 | 7,835 | 33 | 40 |
| certifier-circuits | 55 | 7,554 | 4,720 | 29 | 52 |
| certifier-prover | 70 | 11,249 | 4,980 | 30 | 125 |
| difftest | 27 | 6,183 | 1,334 | 26 | 22 |
| bench-traces-serve | 55 | 12,832 | 8,341 | 65 | 75 |
| bench-traces-zk | 38 | 10,967 | 4,773 | 49 | 67 |
| bench-prove-path | 30 | 5,610 | 3,717 | 25 | 31 |
| bench-e2e | 77 | 5,915 | 3,807 | 41 | 72 |
| bench-encodings | 33 | 7,016 | 3,435 | 37 | 38 |
| bench-model | 60 | 8,752 | 3,265 | 19 | 96 |
| lean | 144 | 23,148 | 0 | 89 | 0 |

**B. What I changed in your seats, and why** (each [ran] by the owners list, [read] in the files' docstrings):

1. `certifier-device` by path is 6 files, none the device: `device/` holds 35 Lean files (the `lean` seat's), a README and 5 difftest vector scripts. The network certifier's Python is in `verifier/`: `grid`, `active`, `warden`, `pacer`, `certificate`, `commitment`, `coin_link`, `continuation`, `release`, `warden_rows`, `connection_rows`, `transfer_rows`, `out_commit`. The seat is now those 13 modules, their tests and the README.
2. `certifier-verifier` was over both limits (181 files, 23,552 code lines) and its 59 modules are flat, so no subdirectory splits it. Three seats by what a file is: `certifier-device` (above), `certifier-circuits` (the 10 modules with a `.circuit-check.py` sidecar, their pins and tests) and `certifier-check` (the rest: `audit`, `lean_audit`, `service`, `tie_service`, `boundary`, `blocks`, `capacity`, `schedule`, `calibration`, `online`, `advice`, `served_rows`, the docs and the `network/` suite).
3. `certifier-prover` had no tests: the 17 test files, two helpers and one fixture for its modules sit in `verifier/tests/`. They go with it, so the seat that changes a module holds its tests.
4. New seat `difftest`: the 14 Python files inside our four Lean package directories (`*/lean/scripts/`), `nci/README.md` and their 12 tests. Every script's part is `lean-dir`, so its hand-ins go apart; in one seat they do not hold up the others' days.
5. `bench-traces` was over 20,000 code lines (23,799) and is flat too. Two seats by file: `bench-traces-zk` (the 14 drivers that prove or check what a run published, their `.sh` and tests) and `bench-traces-serve` (the rest: what runs the certifier on a host and records it).
6. `bench-e2e` was 140 files. `benchmarks/network_e2e/encodings/` is its own suite (33 files): seat `bench-encodings`. `benchmarks/network_sim/` moves to `bench-model`, because 19 of its 30 files name `datacenter`.
7. `verity/protocols/certifier/__init__.py` (empty) was in no seat; it is `certifier-check`'s.

Every seat is now at most 77 files and 12,832 code lines. `lean` stays whole and held.

Two things you will want before choosing: the 26 modules of `certifier-prover` and `certifier-circuits` are Boolean Definitions, each with a `circuit-check` pin, which MEM-269's rule 3 forbids changing, so those two seats have little a worker may do; and `swarm.py frozen` reads only `cursor/verity-network-accounting-*` and `cursor/network-accounting-*`, which is not how our branches are named (`cursor/na-*`, `cursor/net*-*`), so D gets a script of its own beside the tool. Next: D, then the second report.

