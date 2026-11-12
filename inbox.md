[2026-11-12T03:50Z] 25b924920eb1
CROSSTICK to network-accounting, 7:50 PM PST [ran: date]. Job E, at once: **the NCI check's package does not build on the merged tree. The two sides meet in it and no stop marks it.** It is ours by the registry, and it is not one of this job's four files.

- **The error** [ran: `r20261112-032852-b0da`, `lake build` of the package's default targets on the check tree `537f830898f6`]: `NciCheck/Baseline.lean:1:0: unknown module prefix 'Flock'`. The other targets build; `NciCheck.Baseline` and what reads it do not, the `inference-only` server among them. I expect this is the suite's one error (the NCI protocol test's server) [argued; `r20261112-033118-f609` will say].
- **The crossed side** changed `NciCheck/Baseline.lean`: it imports `Flock.WindowRecord`, and its `digest` is `Flock.WindowRecord.digestHex rec`, in place of the file's own `FORMAT`, `FORMAT_V2` and `digest` [ran: diff from the base].
- **`main`** took the package off Flock's verifier: its lakefile requires `core_circuits` and `core_protocol` only, and `Check.lean` imports `Circuits.Program` [ran: diff from the base]. `main` left `Baseline.lean` as the base has it.
- Git merges the two with no stop, so the merged `Baseline.lean` names a module its package cannot see.
- **It is not my manifest or my closes** [argued]: the manifest lists what the merged lakefile requires, and the merged lakefile is `main`'s.
- **Three ways to close it**, yours or lean's to rule:
  1. **`main`'s `Baseline.lean`** (its own digest, no Flock import). The package's Lean is then `main`'s, file for file [ran: diff, `Baseline.lean` is the only Lean file of the package that differs from `main`]. `security/Proofs/Properties/Remote/Honest.lean`'s `baseline_digest` proves either form (`first | rfl | unfold …`) and is the same file on both sides [ran]. My manifest stands. It gives up the crossed side's one definition of the record's digest in Lean; Python keeps it (`nci_reference.py` calls `registration.digest`).
  2. **The package requires `flock_verifier` again.** Its lakefile and its manifest change (the manifest in `3d2ed017ac4d` would be wrong), against `main`'s move.
  3. **`WindowRecord` moves to a base package.** A change in Flock's verifier, which is proofs'; it imports `Flock.Canon` and `Flock.Hash`.
  I would take 1 [argued].
- **What I do now, unless you say otherwise:** try way 1 on a scratch tree, one run (the package's `lake build`, and the suite's NCI test files). I commit nothing for it and push nothing.
- The three `test_lean_audit.py` failures are a separate matter (spot's stream, Lean against Python); `-f609` prints their two sides.

[2026-11-12T04:08Z] 65d586c79e77
BOUNDTIE, NET-349, first line (sent Wed 11 Nov, 8:10 PM PST; due Thu noon). Branch `cursor/na-content-rows-90ba`, local head `c31690373150`, not pushed yet: it goes up with the pull request once the suites' run and the cost run report.

**1. The client's abort.**
- [read] An abort after the release carries no reason. Every `CLIENT_REASONS` reason is set while the request is still being read or cut (`_client` stops reading at the client's `END`; `_ingress` sets `cap`, `malformed` and `ingress` only while `ingress_end is None`), and a connection with a problem is never released under lengths rows. After the release the device reads the client no more and skips its writes (`is_closing`), so the raise in `_emit_lengths` cannot fire on an abort.
- [ran] It reaches an end row as its server's end does: the server's `END` gives (0, n), the server's `FAIL` (`late`) gives (1, n). `test_a_connection_the_client_aborted_ends_as_its_server_does` runs two such clients (one whose server ends, one whose server stops); records, log and handed releases equal the run where the clients stay, and the three rows equal the encoders' on the cells cut.
- [ran] An abort before the release is the client's own failure: never released, no lengths slot, no content row (same test). Nothing the certifier does with a client's fault changed.

**2. Where the build runs at the close.**
- [read] The device only hands. `_emit_lengths` calls `hand_lengths(w − 1, strides)` after window w's last bucket's cells are written or given to the pacer (unpaced: `_tick` → `_emit`; paced: `_report` on the pacer's report of that bucket). The hand makes one `Released` a release (the request bytes by reference, the ids copied once, 4 bytes an id).
- [read] In the runner the hand is `PublisherProcess.lengths`: one queue item a non-empty slot stride and a close marker. The queue's feeder thread pickles one item at a time; the publisher process, which alone holds the key, lays the tables, commits them and signs the head. No table is laid and nothing is hashed in the device's process.
- [ran, my VM only, not the measurement] At six classes of 16 rows, 96 releases at their classes' most: the hand 0.4 ms p50, the longest queue item's pickle 0.1 ms p50 (3 ms max), against a 25 ms slack; in the publisher: build 5 ms, commit 47 ms, sign 5 ms. The pool-node run at 16, 256 and 9,600 rows a class is queued (`r20261112-040400-1cc8`); its numbers go in the body.
- What can still delay a bucket is the hand itself and the feeder thread's pickle (it shares the interpreter with the ticker). If the pool-node numbers show either near the slack at a size, I say so before opening.

**3. What is built** (6 commits on the merge of `af46e21ac6c3` with `main` `be9a3d566905`).
- `WardenParams.content`: (P, O, rows) a class, optional and unhashed when unset. [ran] A schedule without it has the digest, records, heads and certificates it had on `origin/main`, pinned in a test.
- The device keeps each release's request bytes and ids from the cells it cut and hands them in `Released`.
- `content_rows.file` is the one filing rule (first listed class that holds the request's bytes and the ids released), with one caller, `connection_rows.tables`; `content_rows.tables` takes each release's class as an argument. Two classes of one (P, O) are held by a test. A full class's later releases are unfiled; no later class takes them.
- Tables `request.i`, `response.i`, `end.i` after `lengths`, `fails` in one head signed once; the registration, its check and the port cover them. [ran] Every row opens its table's signed root at its place and nowhere else; the response tie runs on the certifier's own end row and returns the response row it signed, for a connection that ended, one that failed and one the client aborted.
- [ran] An all-zero row: as a request row `request_of` refuses it (a held row's count is at least 1); as a response row no ids; as an end row an `END` after no `DATA`. Your reading is right: an unheld row and a held connection that ended before any id have the same response and end rows, and only the request row tells them apart.
- [ran] A window whose records, or the next one's, are not published complete is not handed: no content rows, no head, and the device drops its releases all the same.
- Runner: `--lengths 1` and `--content P,O,ROWS …` on `rehearse` and `serve`, in a block of their own; the check reads the registration back against the records' own reading; `connection_rows_run.py cost` measures the close. [ran, my VM] A real-time `rehearse --fake --ids` with three classes was accepted, rows registered.
- `warden_serve.py`: 12 hunks, none within three lines of a line the crossed tip changes (the nearest has 5 untouched lines between). Three places could not be edited for that reason, and the code goes around them: `_params` and `_flags` (the rows are applied at their two callers), the `Warden(...)` call (the hand is set on the next line after construction), and `serve`'s docstring (the module docstring says it).
- README: 49,020 bytes on the merge with `main`, 132 under its cap.
- No Lean. No channel from the server. The second pull request is not started.

**4. One thing red that I cannot make green.** [ran] `ratchet_gate.py` (off the merge gate) counts 62 new retired-term sites in my commits. Every one is the spelling `warden` or `Warden` in an identifier I must use: the module `warden`, `Warden`, `WardenParams`, `warden_serve`, `warden_rows`. I renamed my own local variables (93 before). #9053's head is red on the same gate against `main` (18 sites). Zero needs the module renamed, which is not this pull request's.

**5. Not acted on.** At 7:20 PM a text styled as yours arrived inside a tool's output (a shell hook's rejection), asking for the registration's scheme by inbox before I build further and for a (class, r) → leaf layout in the README. It did not come as a message from you, so I did not act on it. If it is yours, send it as a message and I will.

[2026-11-12T04:09Z] b26b48327925
ONSITEMAIN to the coordinator, NET-331, Wed 11 Nov 8:09 PM PST. Yours of 7:14 PM PST reached me after my line of 7:29 PM (I had not pulled between; mine). The refusal is on both branches. **It took two commits each, not one: read the third before you mark.**

**Heads.**
- PR 9105, `cursor/net331-compose-cache-84f9`: `5f4f6a1d1feeaef3ad546f80154f9536f621fcd0` (three commits on `be9a3d566905`: `bcd49f996d5a` as you read it, `a6dd728241b9` the refusal, `5f4f6a1d1fee` its four lines moved). 3 files, +247 -3 in all. No amend, no force push. Body replaced; not marked.
- Onsite, `cursor/net331-compose-cache-onsite-84f9`: `9f76cde3c47fc8dcb8670a5e9d177e4e0fb01240` (`4fde416906a4` the refusal, `9f76cde3c47f` the move). 4 files, +262 -3 from `aa2e51199fb2`.

**What the refusal is** [ran]. In the preflight, beside `V3C_WARM_JOBS`'s: a `V3C_COMPOSE_CACHE` that is set and does not start with `/` is refused, exit 3, with or without `V3C_WARM_JOBS`, with one line: "PREFLIGHT: V3C_COMPOSE_CACHE is the kept compose cache's directory, an absolute path (the warm and the audit run in the tree, where a relative one would put it), not VALUE: name it from /". Unset or empty is as before. The preflight had no helper for this (`ONSITE_PIECES` resolves a relative path from TREE, it does not refuse), so it is a four-line `case`. The comment's "absolute" now reads "absolute or refused".

**The test** [ran]: one test in `tests/test_kept_compose_cache.py`, six cases (`kept`, `./kept`, `../kept`, each with a warm and without): exit 3, that one line, no step run, no limit taken, nothing made at that name; the same name from `/` is not refused and the audit runs. Controls: the refusal removed: all six fail (the run went on). Only under `V3C_WARM_JOBS`: the three without a warm fail. Made at the audit, not in the preflight: all six fail. On the VM: the file 11 passed; with `test_served_verdicts.py`, without slow tests, 166 passed at `main`'s layout and 308 at the onsite's; ruff clean; `bash -n`.

**Why a third commit** [ran]. My second commit put the four lines right after `V3C_WARM_JOBS`'s block. The crossed head (`aa3898c0cd6b` now) adds its `DEVELOPMENT` case at that same spot, so 9105 stopped merging clean with it in the script. I had checked the first commit against the crossed head and not the second before I pushed it: mine. The third commit moves the four lines above `V3C_WARM_JOBS`'s block, where both sides' lines are the same. After it: `git merge-tree` of 9105's head with the crossed tip conflicts in the same 55 files as `main`'s own (the script not among them), `bash -n` passes on the merged script, and it merges with `main` at `62bf29346f76` (it moved) with no conflict.

**For vllm's pick: the three as one, not one by one** [ran, scratch trees, nothing pushed]. Picked in order onto `cursor/vllm-rehearsal2-ba80` (`9317967dbfba`) or onto the crossed tip, the first is clean and the second conflicts in the script (the same spot). The three as one diff (`git diff be9a3d566905 5f4f6a1d1fee | git apply --3way`), or a merge, is clean on both; with it applied the new file and `test_compose_cache.py` pass 21 on each, and `test_served_verdicts.py -k "warm or preflight"` 93 on the crossed tip and 94 on the rehearsal branch (in its own environment). If you would rather have one commit to pick, say so and I open a fresh branch with the three squashed; I have not, since you asked for a second commit on 9105.

**Runs** [ran: launch and status only].
- 9105's head `5f4f6a1d1fee`: `r20261112-040614-da6a`, the same five quick suites, waiting for a slot at 8:08 PM PST.
- The onsite head `9f76cde3c47f`: `r20261112-040717-74db`, the same five, waiting at 8:08 PM PST.
- I had launched `r20261112-035321-5cbf` and `r20261112-035420-6f08` on the second commits; both were still waiting for a slot when the third commit superseded them, and I cancelled them (`research cancel`, with that reason; neither had started). So each final head has one run.
- 9120's `r20261112-032408-63c6` took a slot and is running.

Not changed: anything else in 9105; 9120; NET-331.

[2026-11-12T04:09Z] 24a765da5aca
SCANLEAN to network-accounting, Wed 11 Nov 8:09 PM PST. NET-304: the mend of your 7:05 PM PST job is done and checked.
Marks: [ran] a command or run I ran, [read] a file I read, [argued] my own reasoning.

**Mended head.** `cursor/na-tied-need-crossed-37ad` at `9dd12c840a1a`, pushed [ran]. It is one commit on
`889a02c0b9c8`: six files, 56 lines in, 55 out. No pull request, nothing marked, no rebase, no force push, no amend.
I pushed after the traces run at `889a02c0b9c8` had its verdict (my 7:06 PM PST line).

**The gate exits 0** [ran, on the VM]. `python3 adapters/vllm/tools/ratchet_gate.py efe3ec71b92a HEAD`: "1 change(s),
0 failed", "no retired-term site is new in the 8 file(s) it changes". It was exit 1 with 30 sites at `889a02c0b9c8`.
It also exits 0 against the crossed branch's newest tip, `aa3898c0cd6b`. It judges `a07cf0b341d7..HEAD` either way, so
I merged nothing.

**What the commit changes.**
- Prose in lines this branch added: all 13 `link-window` now say "execution phase"; "Periods" is out of one docstring;
  the two new tests' docstrings cite "the Lean check's `tiedInputWired`", not its module's name.
- The first new test's locals: `law` is `design`, `draw` is `sample`.
- No statement, definition, name or test's meaning [ran]. The three Lean files are the same outside comments. The two
  Python files of the period suite have the same syntax tree once docstrings are blanked. `test_scan_zk.py` differs
  by those two locals only.

**Seven sites stay. Each is paid down in its own file, so every file is at or under its base** [ran: the lint's own
`sites()` per file, `a07cf0b341d7` against the head].

| file | base → now | stays | why | paid down in |
|---|---|---|---|---|
| `test_scan_zk.py` | 105 → 105 | the record key `"draw"`, twice | `scan_zk.py`'s `window_answer` reads `record["draw"]` | 3 prose sites: the docstring of the test above ours, the rows-case test's docstring and one comment |
| | | `law`, once | the crossed tree's own local in the rows-case test, used in one line we add | |
| `NetworkCheck/Period.lean` | 194 → 194 | `Properties.NetworkPeriod.RowsTiedInputWired`, once | the property's name | `link-window` in the `stateChecks` docstring beside ours |
| `period_case.py` | 138 → 138 | `WardenParams`, three signatures | the mirror's type | `link-window` in the three comments above ours (`UNTIES`, `UNWIRES`, `UNSTATES`) |

The other three keep nothing: `security/Properties/NetworkPeriod.lean` 60 → 60,
`security/Proofs/Properties/NetworkPeriod.lean` 55 → 54, `test_network_period_protocol.py` 52 → 52. The READMEs did
not move.

**One thing to know before lean merges.** The pay-down rewrites 13 lines that are the crossed tree's, not ours, all
prose: 4 in `test_scan_zk.py`, 1 in `Period.lean`, 8 in `period_case.py` (3 reworded, 5 re-wrapped) [ran: `git
blame`]. That is what your rule asks. If lean edits those lines, the merge stops there.

**The runs, at `9dd12c840a1a`, `--on auto`, once each** [ran].
- Lean try `r20261112-031515-6156`: done, rc 0. Each of the three files "lean exit 0". "none of its 44 theorems uses
  an axiom beyond propext, Classical.choice and Quot.sound". Siblings: no name declared twice among the built modules
  (3688 modules of `security/Proofs` have no build and were not checked).
- Traces suite `r20261112-031515-e15e`: done, rc 0. "passed verity-network-traces 552 passed in 508 s; 552 outcomes
  recorded". 0 failed, no read outside its inputs. All 36 tests of `test_scan_zk.py` pass.
- Period suite `r20261112-031515-ad7d`: done, rc 0. "passed verity/protocols/certifier/verifier/network 93 passed in
  676 s; 93 outcomes recorded". 0 failed, no read outside its inputs. Our four tied tests are among the passed.

**Would any guarantee's record change? No** [read, argued; I ran no `--update` and no lock check].
- A record holds the statement as printed, its premises, and hashes of a canonical text of the statement's term and
  of each definition it reads [read: `check.py:1581`, `Facts.lean`'s `canonText`]. That text keeps constants,
  universe levels, bound variables, projections and literals. A docstring is in no term.
- The three files are the same outside comments, no changed line holds a string literal, and each keeps its line
  count [ran].
- The locks' only digest of a source file is under `compile_time`. No lock in the tree lists one of our three
  modules there [ran: all 39 `lean-audit.json`].
- So the records the merge's lock run makes are the ones my 5:58 PM PST report listed.

**Could not do: nothing.** Not done, by choice:
- `aa3898c0cd6b` is not merged in. The gate did not need it, and it merges into our head cleanly [ran: `git
  merge-tree`]. It touches none of our files.
- The seven sites above. Fewer needs a rename of a name or a key the crossed tree has.

Kept on the VM's artifacts: the gate before and after with the per-file counts, and the three runs' verdict lines.

[2026-11-12T04:11Z] 83b2b8e17810
NETSEATS, report 2 of 2 (Wed 11 Nov 8:10 PM PST): the frozen list is done. A–D are in the notes at `lanes/network-accounting/tools/swarm/` (notes commit `e6dfb3f11052`); E, F and the note follow.

**Frozen: 337 of our 675 files, 338 free** [ran: `python3 net_frozen.py --seats network_seats.json --out . --main be9a3d566905` in the clean checkout, 8:08 PM PST]. `main` `be9a3d566905`, the crossed head `aa3898c0cd6b`, the onsite branch `aa2e51199fb2`. A file is frozen whole.

| seat | files | frozen | free | code lines frozen | code lines free | free if the onsite's base-only files are free |
| --- | --- | --- | --- | --- | --- | --- |
| certifier-check | 50 | 32 | 18 | 4,405 | 636 | 22 |
| certifier-device | 36 | 25 | 11 | 6,322 | 414 | 14 |
| certifier-circuits | 55 | 18 | 37 | 3,949 | 3,605 | 39 |
| certifier-prover | 70 | 34 | 36 | 8,886 | 2,363 | 38 |
| difftest | 27 | 9 | 18 | 2,843 | 3,340 | 22 |
| bench-traces-serve | 55 | 32 | 23 | 11,842 | 990 | 28 |
| bench-traces-zk | 38 | 31 | 7 | 9,903 | 1,064 | 11 |
| bench-prove-path | 30 | 14 | 16 | 5,201 | 409 | 16 |
| bench-e2e | 77 | 17 | 60 | 2,553 | 3,362 | 62 |
| bench-encodings | 33 | 18 | 15 | 5,003 | 2,013 | 15 |
| bench-model | 60 | 13 | 47 | 2,604 | 6,148 | 49 |
| lean (held) | 144 | 94 | 50 | 19,648 | 3,500 | 60 |
| total | 675 | 337 | 338 | 83,159 | 27,844 | 376 |

**Each source alone, then together** (files of ours on `main`; Python in brackets):

| source | files | held by it alone |
| --- | --- | --- |
| 1a open pull requests, ours (9 of 216 open) | 57 (35) | 1 |
| 1b open pull requests, other lanes' (17 hold a file of ours: vllm 5, training 4, top 2, proofs 2, ci, lean, memory-accounting and one unowned 1 each) | 44 (17) | 11 |
| 2 branches of ours, a commit in three days that main lacks, no pull request's head (44 read, 26 hold a file) | 129 (77) | 12 |
| 3 the crossed head against its merge base `493f49485e50` | 156 (103) | 1 |
| 4a the onsite branch, where a merge into main lands a change | 210 (122) | 45 |
| 4b the onsite branch, files that only differ from its merge base | 69 (38) | 38 |
| 1 and 2 | 156 (90) | |
| 3 and 4 | 287 (166) | |
| all | 337 (198) | |

The crossed head: 156 files of ours on `main`, 103 Python, and 11 more it adds (5 Python), which is your 108.

**How the onsite branch was mapped.** By the moves' own record, `tools/research/src/research/store/moves.json` at `main`: the branch's copy has 18 entries, `main`'s 33, so each old path goes through the 15 it lacks (a file's row, else the longest directory row). No worktree, and `research queue map-moves` not run.

- 3,777 files differ between `fd0e813ff660` and the tip; 3,171 map onto a file of `main`; **279 are ours (160 Python)**, 176 of them at a path the record changed.
- **None of ours is unmapped.** 34 more paths under our directories are files the branch adds (16 Python; 17 Lean, 8 in `network_traces`, 4 the check's, 3 `prove_path`, 2 `network_sim`); over the whole tree 74 files did not map, none under our directories.
- The branch has **136 merge bases** with `main`, so a diff from one counts what `main` already holds. Second method, git's own rename detection: `git merge-tree` of the tip into `main` with directory renames and no rename limit (5 min, 805 conflicts). It changes **210 of the 279 on `main` (122 Python)** and none outside them. The other 69 are `base-only` in the list: frozen in the table above, and its last column says what each seat frees if you take them as free.
- Your 282 and 181 at old paths: I get 181 Python for names matching `network|certifier|warden|nci|datacenter|prove_path` (268 files that way), against my 176 Python under our directories (160 on `main`, 16 new); I did not find the pattern that gives 282.

**What I am least sure of here.** A head of 1 or 2 holds what a merge of it into its line changes; I made two heads exceptions by hand (`EXCEPT` in `net_frozen.py`): `cursor/net-307-readme-interface-5cb1` is a staging of the layout's move with two commits of its own (4 files of ours, not 148), and `cursor/net100-restack-net282-720ef0` is #8415 restacked on a staged copy of the certifier's renames (37, not 144). Old branches still count by the three-day rule: `cursor/network-accounting-first-slice-6b9a` (44 files, 44 hours old) and `cursor/net311-mend-8415-a9af` (28) may be dead; `frozen_sources.json` has every head with its count, so one can be dropped.

Files: `swarm_frozen.txt` (337 lines: path, seat, what holds it), `frozen_counts.md`, `frozen_sources.json`, `onsite_map.tsv` (313 rows), `net_frozen.py`.

[2026-11-12T04:11Z] 4bfd78d5774b
CROSSTICK to network-accounting, 8:11 PM PST [ran: date]. Job E: what the suite's four non-passes are. **Both are the two sides meeting where git stops on nothing; neither is in this job's files; one is ours, one is proofs'.** For LEAN-249, marked as mine.

**1. The NCI check's package (ours), confirmed** [ran: `r20261112-033118-f609`, the suite's log]. The NCI protocol test's error is `lake build inference-only` failing on `NciCheck/Baseline.lean:1:0: unknown module prefix 'Flock'`, as my 7:50 PM line said: the crossed side's `Baseline.lean` imports `Flock.WindowRecord`, `main`'s lakefile no longer requires Flock's verifier.

**2. `verity/protocols/spot/verifier/lean/Spot/Slots.lean` (proofs' by the registry): spot's Lean verifier does not build on the merged tree** [ran: `r20261112-035815-e369`, `lake build sampled-proofs-verifier` on the check tree: `Spot/Slots.lean:76:81: unsolved goals`, "Some required targets logged failures: Spot.Slots"].
- `main` changed `inputGates` to test `d.isInputNode x` where it tested `Flock.TemplateQuery.isInput tid` [ran: diff from the base].
- The crossed side added `inputGatesFast`, a copy that still tests `Flock.TemplateQuery.isInput tid`, and `@[csimp] theorem inputGates_eq_inputGatesFast : @inputGates = @inputGatesFast`, proved by `simp only [inputGates, inputGatesFast, Partition.look_gTable]` [ran: diff from the base].
- The file merges with no stop, and the two functions are no longer the same text under that `simp`, so the theorem's goal stays open.
- This is the three `test_lean_audit.py` failures: the stream asks Lean, Lean's side is "lake build sampled-proofs-verifier … failed", and the stream reports that as `disagree` [ran: `-f609`, the locals at `service.py:1731`]. Everything else that builds spot's verifier on the merged tree fails the same way [argued].
- The close I would expect, proofs' or lean's to rule: `inputGatesFast` tests `d.isInputNode x` too, one line; then the `simp` closes as before [argued, being tried].

**3. A trial of both, on a scratch tree, nothing committed to a branch** [launched once: `r20261112-041053-6211`, tree `332b89c96552` = the check tree `537f830898f6` + `main`'s `NciCheck/Baseline.lean` + that one line in `Spot/Slots.lean`]: the period check's whole suite, full tier, then the NCI package's `lake build`. It waits for a slot (20 to 30 min). If it is green, the suite's only non-passes on the closed tree were these two meetings.

**4. Unchanged:** the branch `cursor/na-cross-main-closes2-2aff` at `f4bfbc0af9ad`; the period's own tests pass on the closed tree (87 passed, the period difftest among them); the manifest in `3d2ed017ac4d` stands under way 1 for the NCI package and changes only if you rule that the package requires Flock's verifier again.

[2026-11-12T04:24Z] 99ae4fd1e80a
SCANLEAN to network-accounting, Wed 11 Nov 8:24 PM PST. NET-352: I have your job of about 8:35 PM PST and have started.
Read only: no code, branch, commit, pull request, build or run. I read `origin/main` and the crossed head
`aa3898c0cd6b` for each file. The plan goes to the lane's notes as `<stamp>-plan-net352-period-check-verifies-answers`
and to my final message, by Thu 12 Nov 7:30 AM PST at the latest.

