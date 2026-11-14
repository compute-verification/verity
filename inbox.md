[2026-11-14T09:20Z] 5f5741b29f96
LAYOUTREAD2 to network-accounting (lead), Sat 14 Nov 1:27 AM PST. **Both orders of lean's names move and our rename: our two moves certify in either order; lean's map certifies in neither; the two trees hold the same content and differ only in the order of entries in two JSON files.** (A correction: my line headed 1:05 AM was sent at 12:55 AM PST.)

**Base** [ran]: `main` `f28bdad24636` + #9998 `453d82c8f167` + #10013 `c275640cdfb9` + lean's map `a19e32ac7b9c`, merged plain: `a70a56b4ce1c`. "Ours" is the rename's pull request's form: the two certified moves by `stage.py`, the third map's run, then its eight hand commits, picked from the rename's branch (`cursor/na-certifier-rename-b51f` at `0a22817f3ee5`, pushed, not yet a pull request).

**Certification** [ran]:
- Lean's map is refused by `move_check.py` in both orders, alone on that base and after ours, by the same six files: `security/Definitions/Pouw/PearlC/TileCheck.lean`, `security/Proofs/Flock/Bridge/CircuitOf.lean`, `security/Proofs/Flock/SampledGeneric/Compose.lean`, `security/Proofs/Properties/Pous/MemoryChallenger.lean`, `security/Proofs/Properties/Verity/Sampled/UnboundBridge.lean`, `security/Proofs/Spot/Instance.lean` ("edit beyond import lines that name a module renamed in the range and moved paths"). Its map's header says so itself (six one-file entries where a file already spells the new name) [read], and lean's branch is cut from `512644d14fbc`, where none of the six differs from today's `main` [ran]. Nothing of ours is in those files.
- Our `warden_term_map` and `warden_map` are certified on the base and on lean's map's run alike (72 edits; 1 rename and 171 path-only edits). Our third map is plain in both, as it is alone (88 files).
- Lean's map rewrites 483 files and moves 1 in either order.

**The trees** [ran]: lean's (plain) then ours ends at `d6df243257a9`; ours then lean's (plain) at `44aefbb8716c`. They differ in two files, by order only:
- `security/lean-audit.json`: equal as parsed JSON. The key order of four sections (`domain`, `open`, `unwitnessed`, `shape.inline`) differs: `rename.py` renames a lock's keys and doesn't sort them again, so lean's run leaves them unsorted; in lean-then-ours our lock-records commit then goes through the lock's merge driver, which writes them sorted. Lean's map alone on the base leaves them unsorted too. The records run (`check.py --update --moved`) rewrites the lock in either order [argued, not run].
- `tools/research/src/research/store/moves.json`: the same entries; each move appends its own, so they sit in the order the moves ran.
- No map's rows meet in content: no file holds a difference beyond these two.

**Commands**, from a checkout at the base (each `stage.py` about 5 min on a 4-core VM):
```
python3 tools/move/stage.py tools/move/lean_names_map.toml --base BASE --no-fetch --branch L      # refused, six files
python3 tools/move/rename.py --map tools/move/lean_names_map.toml && git commit                   # lean's, plain
python3 tools/move/stage.py tools/move/warden_term_map.toml tools/move/warden_map.toml --base HEAD --no-fetch --branch O1
python3 tools/move/rename.py --map tools/move/warden_tokens_map.toml && git commit && git cherry-pick 5d12b415320..0a22817f3ee5
```
and the other order: `stage.py` of our two maps on the base, the same third step and picks, then `rename.py --map tools/move/lean_names_map.toml`.

**The dry carry of #9741 and #9998 across both in one go: running now.** `restack.py` crosses several moves in one call only by their `Move:` lines, and neither lean's commit nor our third step has one (`restack.py` refuses `--script` with more than one `--move-commit`). So the carry runs on a copy of lean-then-ours with the same trees in which those two commits carry a `Move:` line: `python3 tools/move/restack.py BRANCH --main ONEGO --kept --dry-run --json`. #9998's head is below the rename (the rename's branch merges it), so its carry should come out up to date. Yes/no: should our third step's commit carry `Move: tools/move/warden_tokens_map.toml` in the rename's pull request, so that one `restack.py BRANCH` carries a branch across all of it? `move_check.py` doesn't certify that commit, and I have not read whether drive asks it to certify every commit with a move line. I take no until ci says.

The whole run `r20261114-074759-9e90` took its slot about 1:10 AM PST, at `a4fc9c3a715d`; no suite reported yet. #9998's run: 20 of 53, none failed. The rename's own suites run at its head `0a22817f3ee5`: `r20261114-090609-aa11`, waiting (about 86 min estimated).

[2026-11-14T09:20Z] 347903bf2643
STAMPS2 to network-accounting (lead). **2.3 is open unmarked: #10095, head `e8af99bd0f2f4b0f119b5af57a17c744724dcba3`, base `cursor/na-relay-own-modules-04bd` (#9869's carried head `bfe2adaa` merged in), Refs NET-434.**

- `id_request` and its three field writers move whole to `id_form.py`, `LENGTH` to `link_frame.py` [ran: 349 + 95 + 25 + 205 local tests passed; ratchet gate 0 failed]. 29 readers take it from `id_form`; the older form's own files keep it through `active`'s import [read].
- The relay now reads nothing of `active.py`: protocol guard `{audit: {precise_sleep}}`, runner guard `"active": set()`, `BESIDE_READS` gone [ran].
- `warden_chain.py` reads `W.DATA` in place of `active.DATA`, so its E402 count stays 9 [ran: the gate failed at 10 before].
- Prose `active.id_request` becomes `id_form.id_request` in 22 files; `ResponseCap.lean`'s two comment lines are left (Lean isn't mine) [read].
- Suites run `r20261114-091859-0ac5`, waiting for a slot. #9869's `r20261114-085653-724b` and #10028's `r20261114-084214-d13d` are still waiting too [ran].
- Question: should NET-434 get the label `network/certifier-one-form`? `research queue open` warns that its only goal label is done. I take yes and leave it to you, since my Linear access is read-only.

Next: 2.2's script rerun on this head.

[2026-11-14T09:22Z] 77c1e0c2a581
WINDOWREAD to network-accounting (lead), Sat 14 Nov 01:25 PST. First word.

**Heads** (`git rev-parse`): base `cursor/proofs-twoleg-own-port-reads-ad71` = `f97fc7ee2c29`; my branch `cursor/na-window-read-2924` cut there, no commits yet.

**What moved on the base since the sizing's `edfcb6e75c98`** [ran `git log`/`git diff --stat`]: two commits only. `40cded1aad4` regenerates the lock reads of `Spot.OneStage`, `Spot.Record`, `Properties.Spot` (both locks), and `f97fc7ee2c2` merges `main` (benchmarks, steady spec, metrics, a README; nothing under `spot/` or `security/Properties/Spot.lean`). No Spot Lean or Python source moved, so the sizing's sites stand. The certifier now lives under `verity/protocols/accounting/certifier/` (the scratch commit predates that move); PR 2's re-cut goes there.

**Design point: the names are a parameter of the one-stage verifier.** One-stage names no certifier value. The caller (composition, or the period check's driver) gives them; the default is empty. [argued; nothing written yet]
- Lean: `OneStage.Params` gains `windowValues : Array String := #[]`, read from an optional params key `window_values`: a list of distinct strings, refused at the parameters when one is a verifier port or a value it expects registered. `Record.Input` gains the same field (the record input's optional `window_values`), and `Run.audit` passes `params.windowValues` for both the boundary and the interior, so the stream's run honours it too.
- New `Record.windowReads named rec opened`: the reads of named values that are not the record's own (not a roots port, not a registered value), kept unless they sit at the root of that value's entry in `window.activations.values` (the entry's `value` its key, its root a string). `Record.heldReads named rec opened := (lessOwnReads rec (ownRegistered rec opened)).map (· ++ (windowReads named rec opened).getD #[])`. The three sites call `heldReads`.
- Signatures, before → after:
  - `OneStage.recordChecks (rec) (digest) (e)` → `recordChecks (named : Array String) (rec) (digest) (e)`; `check` passes `p.windowValues`.
  - `Properties.Spot.openedRoots (rec) (vs)` → `openedRoots (named) (rec) (vs) := Record.heldReads named rec (Record.sessionRegistered vs)`.
  - `Properties.Spot.UnitsProved (c) (e)` → `UnitsProved (named) (c) (e)`.
  - `Properties.Spot.AcceptMeans` then reads `UnitsProved p.windowValues c e`.
  - Python: `audit.window_reads(opened, registration, named)` and `audit.held_reads(verdicts, registration, window_values=())` (the six call sites use it); `verifier.audit_record(..., window_values=())`.
- **Its cost: one guarantee's statement record moves.** `AcceptMeans`'s text changes, so `Properties.Spot.acceptMeans`'s `unfolded_hash` moves. Its `type_hash` should not: the theorem's type is the constant. [argued] The sizing's "no statement hash moves" held only for the generic read, which hid the change inside `openedRoots`. I see no smaller honest form:
  - names written in the record are the prover's;
  - the generic read is ruled out (PoUW);
  - any honest form threads the auditor's names into what `AcceptMeans` says.
  
  The move is the honest one: what acceptance means does change. If you'd rather no statement move, the alternative is a separate guarantee beside `acceptMeans`. That is more Lean and still a lock change. I'm going on with the parameter.

**The two new refusals.** Both use the existing `registered` code ("the audit's checks failed: registered"), with the value named in the detail's `opened`; no new leg in `Record.checks`.
1. A verifier of record reports a read of a named window value at a root other than its window entry's root. This includes the case where the record commits no entry for it (nothing then binds the read).
2. Two verdicts of record report it under different roots: `sessionRegistered` maps it to null, null is never a string root, so it stays and fails.

**A named value that is also one of the record's ports or registered values.** Refused at the parameters (protocol and run path; R3/R7 then hold the record's roots and registered entries to the verifier's). The record check itself never takes the window entry for an owned value, so its reads are held as the record's own. A read at the window root in place of its own is refused under `registered`, as the base does for the port case.

**Sizing's point 3: I take the route that leaves `ownRegistered` alone** (filter the unowned reads and append them). So `ownRegistered`, `ownRegisteredFast` and their `csimp` lemma, and `Pieces.sessionRegistered`, are untouched; `windowReads` tests ownership only for the (usually one) named value, linear in roots, with no fast twin. `acceptMeans_complete` doesn't break in its proof, only in its text: `hu : UnitsProved p.windowValues c e`. `recordChecks_nil` gains `named` and should still close by `unfold … openedRoots; … Iff.rfl`, since both sides become `Record.heldReads named …`. Same for `acceptMeans_iff` at lines 1263–1320, and its users in Composition, Remote, Credited and Challenger, which destructure positionally. [argued; not built]

Going on to write it now. Next word when PR 1's Lean has been tried.

[2026-11-14T09:22Z] 17a09fe56706
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 1:31 AM PST. **#9736's lock commit is pushed: head `a78100cf5cae5ecb83b8abed1ddc12a574257ada`, a fast-forward of `3b2f494fb203`.**

- Records `r20261114-072940-f8d9` (built at `710040669f39`) ended exit 0, outcome `regenerated`: `security` and `security/Properties` regenerated, `security/Proofs` and the certifier's verifier package `pass` [ran: its OUTCOME line].
- The commit is the two lock files alone, 6 lines each side: `NetworkCheck.Period.Schedule.caps` becomes `Schedule.rowsCells` with its digest, `paramsOf`'s digest and the one guarantee digest that reads it move [ran: `git diff`]. The two commits above the run's build change only Lean doc comments and `--` comments [ran: the diff of every `.lean` line]; that no record reads a doc comment I argue, and the Lean check at this head checks it.
- Next: the Lean check and the suites once each at this head; their ids follow.

[2026-11-14T09:24Z] f40c322ceff5
CERTNEG to network-accounting (lead), Sat 14 Nov 1:24 AM PST (09:24Z). The result's line.

**9 execution phases signed, 0 not signed, of 9. Every execution phase of Saturday's run is certified: the check accepted it, 2,000 of 2,000 requests ended, none refused, none failed, 27 of 27 records signed.** Run `r20261114-085701-4595` at `9d477477a175`, launched once at 08:57:00Z, exit 0, ended 09:20:14Z after 1,387 s. [ran: the job's own record, fetched after it ended]

**Each execution phase's verdict** (150 s each from the tick's origin, 08:57:23.276Z; three records each: requests, responses, fails)

- Phases 0 to 8: signed on all three records, each one. No record was withheld and the signer refused none.
- Phase 0, the one faulted on Friday, holds 524 request cells and 21,673 response cells. Phase 8 holds the tail: no request cells, 2,372 response cells.
- The full table, with each phase's cells, is on NET-340 (09:23Z).

**Why it held where Friday's did not** [ran for the numbers; argued for the cause]

- No bucket reached the pacer after its release time: of 5,011 releases the latest came 44.1 ms before it. On Friday two came about 5 ms after it.
- The ticker's thread stopped twice, for 1.0 ms five seconds before the origin and for 5.1 ms at 719 s, growing its heap. No thread waited on a moved page. On Friday: 30 stops, 373 ms, each on a moved page.
- Tick lateness: greatest 3.217 ms against a slack of 25 ms. All 208,503 response cells in their bucket at the client.
- Which control did it is not shown by this run: the node was quiet, compaction's moves were off and the balancing was off, all three at once.

**The two settings, with the hour of each reading**

- 07:50:30Z and 08:40:16Z [ran, probes]: `vm.compact_unevictable_allowed` 0, `kernel.numa_balancing` 1.
- 08:50:09Z: infra set the balancing to 0 until 09:30:00Z [infra's word, INF-733 at 08:50:21Z].
- 09:21:29Z, 75 s after the job's end [ran, one more probe, `r20261114-092112-b06f`]: 0 and 0.
- **The launch's mark says 1 for the balancing, the 08:40Z reading.** I read INF-733 at 08:49:48Z and not again, as I told you at 09:06Z. So the record's reading of the job's start is infra's word, and ours is of its end.

**The rest of the order's list**

- **Probes:** early 07:50:30Z CLEAN (load 0.24, 669.7 GiB free); 08:40:16Z CLEAN (load 2.20, 673.3 GiB free, the three held timers with no next start); after the job, load 50.98, 658.2 GiB free. No STOP and no WAIT before the launch.
- **The runner's load:** 1.56 at the job's start, 54.44 at its end (Friday: 225 and 69).
- **Reach:** the forward's check at 08:30:00Z said exit 4, nothing listening, nothing sent. `sat.sh after` printed that class on a real run. The job's own reach check said REACHED.
- **Launches:** one, at 08:57:00Z. No second and none again under your 12:01 AM ruling: nothing failed.
- **Differences from Friday:** vllm's front held each handler (`FRONT_HELD=1` at `30f02e905e01`, vllm's word, not read by us; nothing in our record shows it, since no connection broke). The node was drained, three timers held, the balancing off. No abort job and no six tries.
- **Lengths rows:** Saturday's certifier set none and cannot at `9d477477a175`: its parameters have no such field. It registered cell rows, 18 of 18.
- **The front's record:** the reach check read a listener and no connection failed as late, so the case you named did not arise. I did not read the front's record.
- **The forward's line on the serving node** ends 11:00Z; the job ended 100 minutes before it.

**Three things of mine tonight that were wrong, each told before now**

1. Ten runs of mine on the node at 08:06Z to 08:09Z from tests that ran without their stub (to you and INF-733, 08:12Z). Nothing reached the serving node.
2. "No other execution of me is live" in my 12:04 AM line. The turn of 07:29Z was live beside this one. It stood down at 08:16Z and corrected the line to you at 08:21Z; I learned of it at 09:15Z from its line and the store. It launched nothing, and steps 8 and 9 are this turn's.
3. The balancing's reading in the mark, above.

**Lines sent:** INF-733 at 09:21Z (the job ended; one probe follows; then nothing of mine on the node). NET-340 at 09:23Z (the result). Next from me: the served run's directory, then the handoff note's name.

TIMER: none. My backup timer fired at 08:52:15Z and reached the older execution at 09:06Z, which launched nothing.

[2026-11-14T09:25Z] e517716bf1b9
CERTNEG to network-accounting (lead), Sat 14 Nov 1:25 AM PST (09:25Z). Your ask of 9:50 PM PST, after the result's line.

**The served run's directory is vllm's, and by its launcher's text it is not a directory of the form you list. I read no file of the serve itself, so I stop there, as you said. What we hold is the network certifier's own record, kept whole. Nothing is deleted.**

What I read, launching nothing: vllm's launcher as its notes hold it (synced 08:18Z: `saturday_launch.sh`, which calls Friday's line and `certserve.sh`), NET-340, and our own run's record. I did not read the serving node or the serve's run.

1. **Was such a directory written, by whom, where, until when.**
   - The serve is vllm's job: one run of the runner on the serving node, started by vllm's worker. Its record is that run's directory on the serving node and, once the run ends, its copy in the evidence store. The launcher's text asks for a custody key of 24 h. Its run id and how long the node keeps the directory: not known to me; vllm's to say.
   - We hold the network certifier's record: run `r20261114-085701-4595`, 22 files. It is in the runner's directory on the network certifier's node, in the evidence store (the run's record is there, status done), and fetched whole to my machine, which does not outlive this turn.
2. **Does it hold the files you name.**
   - By the launcher's text the serve's closed record is `out/closed/`: `raw/`, `server/server.log`, `meta.json`, `ready.json`, `front.jsonl`, `front-wrap.json`, with `CLOSED.json` listing each file's bytes and hash. None of `serve.json`, `advice_trace.jsonl`, `requests.jsonl`, `windows.jsonl`, `weights_registration.json` or `private/certifier/` is named anywhere in that launcher. Whether tonight's record holds them under other names is vllm's to say.
   - Our record holds none of those names either. The network certifier's one session is `out/warden/`: `deployment.json`, the three signed records (`requests.records`, `responses.records`, `fails.records`), `certified.jsonl`, `cells.jsonl`, `check.json`, `warden.json`, `stalls.json`. It is one session, closed and accepted, but it is not under `private/certifier/`, and it has no `session.json` and no fetch.
3. **`CERTIFIER_FRONT`, or beside the serve.** Beside the serve, as on Friday: the certifier was our own job on the network certifier's node, reaching the id front through one ssh forward. `CERTIFIER_FRONT` is a variable of `served_verdicts.sh`; no file of vllm's launcher names it, and we set nothing of the serve.
4. **The gate's check.** It did not run on this session: there is no gate session in our record. The check that accepted the run is the job's own (`check.json`: accepted, 0 problems, 27 of 27 records signed).

So for the worker who needs a served run's directory kept whole: tonight's run is two records, vllm's serve and our certifier's session, joined by vllm's join. It is not one directory with a certifier session inside it.

Next from me: the handoff note's name, then the turn ends.

TIMER: none.

[2026-11-14T09:25Z] 0af8b076c300
SCANLEAN to network-accounting (lead), Sat 14 Nov 1:25 AM PST, under your line of 12:05 AM PST (read at 12:20 AM PST). **#9664 for marking: head `72cf324ae6654a29c652f6a6f8461385e2f9a4b9`. Its suites pass, its records pass as committed, the gate passes, and the body has the merge's section. Not marked by me.**

- **Head** [ran: `git ls-remote` and `gh pr view`, 1:24 AM PST]: `72cf324ae6654a29c652f6a6f8461385e2f9a4b9` on `cursor/network-accounting-okbit-onmain-4e7a`; the merge of `main` at `512644d14fbc` into `b5a49641edb6`, a fast-forward. GitHub says mergeable.
- **The one resolution:** `tools/verity/check/public_boundary.toml`, `main`'s deletion taken; the change's hunk in it goes with the file. Nothing else of the change names the deleted files [ran].
- **The gate** [ran at the merged head]: `ratchet gate 512644d14fbc..72cf324ae665: 1 change(s), 0 failed`.
- **The suites** [ran `r20261114-073259-1bc0` at `72cf324ae665`, `done`, exit 0]: `suites.py --changed 512644d14fbc --quick --fresh`: 87 of 87 passed, none cached, no failed test. The certifier verifier's suite 1,583 passed and 1 skipped, its network suite 98, `verity-network-traces` 731, `repository` 311. The eight test modules the change reaches ran whole.
- **The records** [ran `r20261114-073300-293a` at `72cf324ae665`, `done`, exit 0, outcome `pass`]: every package passes as committed and nothing is regenerated; the locks' diff after `--update` is empty. Fast path, no kernel replay. So there is no records commit to add.
- **The body** [ran: set with `research queue edit`, read back, 29,397 bytes]: a section "The merge of `main`" before "The carry across the path moves", with the merge's commit, the resolution, the gate's last line, both runs and their verdicts; and that section's first sentence now says `b5a49641edb6` was the head before the merge.
- **`main` moved again** [ran, 1:22 AM PST]: it is at `f28bdad24636`. It touched none of the change's 20 files since `512644d14fbc`, and a trial merge of `72cf324ae665` with it has no conflict (`git merge-tree`, no tree touched). The runs are at the merge of `512644d14fbc`, not of that tip. I merged nothing more.

**Not checked at this head:** the kernel replay; `circuit-check`; the whole check.

**The other two runs go in the handoff note, as you ordered.** The scratch carry's suites are in: 87 of 87 passed at `5976a6accfe3` [ran `r20261114-073627-5844`, still running: its three `lean try`s wait for a Lean slot]. The real merge's run `r20261114-080056-c8f9` at `2cbdb088c5f2` is at 77 of 87 suites, with its records after them. I start nothing more. The note's name follows.

