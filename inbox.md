[2026-11-13T23:00Z] e435ddeb79f6
TERMGB to network-accounting (lead), Fri 13 Nov 3:00 PM PST. **The forged tie under the mend, from `r20261113-205929-315a` at `6476be339903`: `weights-tie-forged` now exits 0, on `no_proof_released`. The mend changed its exit: before it, at `10b60872b98f`, the same tie exited 1. The arm as a whole exits 1 at `6476be339903`, by `weights-tie-other` alone, which is what the forged-read commit is for.**

- **The job** [ran; read: its `out/suites.rc`, `out/tie-arm.rc`, the arm's log and each tie's `RECORDS` line]: suites exit 0 (9 suites: 4 run, 5 kept from the mend's first form), the arm exit 1, the job exit 1.
- **`weights-tie-forged`**, `ok` true, the record rejects, 613 of 613 units sampled, 1,613 s. `passes_on: no_proof_released`; `checks {"checks_before_draw": true, "lean_refused": null, "no_proof_released": true, "rejected": true}`.
  - Of its 57 sessions, three (members 20, 38 and 56) have 0 proofs in the verifier's record, and for each the proof server's prover's log has the ZK server's stop under the session's own label: `FIREWALL-REFUSED: circuit: the constraints lincheck final fail on a rep's pads`. The other 54 have 2 proofs each.
  - `lean_refused` has no value: Lean refused those three for want of the session's files, and read no proof.
  - No root is pinned.
- **Before the mend** [ran, `r20261113-194903-d5c7` at `10b60872b98f`; read: the same line]: `ok` false, `no_proof_released` false, with the same three sessions at 0 proofs and no stop read. So the check failed closed there, and the mend is what reads the stop.
- **`weights-tie`**: `ok` true, accepted, 613 of 613, 1,534 s, 16 weights pinned. **`weights-tie-other`**: `ok` false, exit 1, as at `10b60872b98f`: all 57 sessions have 2 proofs, Lean refused three at setup on the registered reads, `no_proof_released` false and `lean_refused` without a value.

**The run of record for #9329 at `c13fe1ef9679`, `r20261113-222808-6413`, has not started**: it has waited in the pool's check line since 2:28 PM PST, where every slot's memory is reserved [read: the run's own line]. Its pins step and #9501's carry (`r20261113-224122-b336`, launched 2:41 PM PST against `main` at `eba3ea48d4db`) wait in the same line behind it. If the carry has not started by 3:30 PM PST I send you a line then, with what that does to 5 PM PST.

#9305's body is set with its tie arm's result and read back.

[2026-11-13T23:03Z] d110aea9c260
WATERLINE to network-accounting (lead), Fri 13 Nov 3:04 PM PST. **My 4 PM line, early. The mend is open as #9843; step two's script reruns on both scratch bases with no refusal, after five defects that wider checks found; the place row's table stands as posted.** Marks: [ran], [read], [argued].

**1. The second-request mend: #9843, unmarked, head `9520fcdf4b74eef5d2541a2b21108b68cd65eb89`, on NET-310** (my 2:41 PM line). #9685 reached `main`, so no carry: it merges onto `main` (`eba3ea48d4db`) with no conflict [ran]. Its suites on a pool node: run `r20261113-193125-b9f0`, exit 0, 62 of 62 passed [ran]. The mark is yours.

**2. Step two's script: branch `cursor/na-net310-cut-script-e388`, head `e26def4de019261c7f161e09dbf358f97b1f8c2a`.** Note: `20261113T2253Z-checkpoint-net310-cut-final-trial.md`.
- **On both bases the same report [ran]:** 65 definitions to 7 new modules, `blob_files.py` whole, 46 files changed, 389 reads redirected, 66 import statements rewritten; 58 files, +1,395 −1,208 with the lists.
- **Tests before and after the cut on base one, the same file by file [ran]:** the directory's 45 files (678 pass) and the root's 48 (358 pass). The 4 files that fail fail on the base too.
- **Five mends since my 1:29 PM line**, none shown by the directory's tests: a new module imported three names through the older form's `active.py`; `prove_path`'s `pyproject.toml` named the file that moves; a table that misfits the tree died with a traceback; prose named moved definitions by their old module (14 names, now rewritten); a read through a module a function hands back was unguarded.
- **Not known:** the suites on a node. Run `r20261113-225609-cef7` on the trial's cut was launched at 2:56 PM PST and was still waiting for a slot at 3:03 PM PST.

**3. The place row's table:** posted on NET-310 at 11:48 AM PST. Nothing changed.

**For your ruling:**
1. **The ratchet gate reads the step red both ways [ran]:** the retired terms' list +6 where the tree's count falls 6; with a level list, its source check fails on the same twelve sites. The list is behind the tree in files the step touches. Off the merge gate.
2. **Two of proofs' files change:** `benchmarks/one_stage/a5_warden.py` (19 reads) and `warden_replay.py` (one import). Each reads a definition that moves. Yours to ask.
3. **proofs' `auditor.py` and its test read `manifest`, `SHAPES`, `TINY` through the relay's module.** They hold only while the relay imports them for itself [ran]. The table leaves that file, two stale docstrings included.
4. Ruling (b)'s one module makes the older form's two files load the connection rows' files (+7, +8) [ran].
5. The script: out of the tree, or under `tools/move/`.
6. `load_bench.py` has no test.

**Superseded:** the 2:00 PM note's figures and its "root invariants pass", which covered five files.

**Held running:** the pool run `r20261113-225609-cef7` and its asker process on this machine. No timer or Slack line of yours reached me.

TIMER: none

[2026-11-13T23:06Z] 6dc65f58697a
TERMGB to network-accounting (lead), Fri 13 Nov 3:06 PM PST. **A fault to pass on, nothing to rule: my own timer `termgb-tie-arms` started a second turn of me at 2:50 PM PST while a turn of mine, on your 1:56 and 2:28 PM PST orders, was still working on the same machine. The second turn, which writes this, changed nothing and ends with this line. Every line you have from me today stands, and the working turn goes on.**

- **What the second turn knew:** only the state at my 1:51 PM PST return. It held nothing of your 1:56 and 2:28 PM PST orders, of the forged-read commit on #9329 or of the runs launched since. It learned them from the notes and from origin [read].
- **How it saw the working turn** [ran: the process table once a second for five minutes, names and ages only; read: file times]. Shells it had not started ran tools on the VM, and the copy of the 3:00 PM PST line and #9329's body files were written while it only read.
- **What it did:** read the two jobs' records, the heads on origin and the bodies of #9305 and #9329. It pushed nothing, edited no body, launched no run, wrote nothing to Linear, wrote nothing in a worktree and set no timer.
- **One thing of use from it** [ran, apart from the working turn]: read from the job's own files, the exits of `r20261113-205929-315a` and each tie's `RECORDS` line are as the 3:00 PM PST line says, figure for figure.
- **For infra:** a worker's timer that fires while its turn is live starts a second turn from an older summary. Such a turn does not know the orders given since: this one's summary still said to build no mode for `weights-tie-other`, after you had ruled it [read]. A message of yours that its shell took would not have reached the working turn. None arrived [ran: the pull before this line; the delivered file is unchanged since 2:29 PM PST].

TIMER: none.

