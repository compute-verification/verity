[2026-11-12T15:29Z] 02f75e06ef1b
CERTNEG, Thu 12 Nov 7:29 AM PST. **#9290 is ready for your mark: head `f919f768d435`, suites run `r20261112-150642-ed97`.** It is open and unmarked.

**The four things you asked for**
- **`baseline.json` locked to the run:** `r20261112-135528-7404`, at `4d784714a84d`, written by `result.py baseline --lock` from the evidence store (commit `9621a9d7b8`). A test holds it to the identity `run.sh` runs at with no setting. [ran]
- **`main` merged in** at `06728b3b7c66` (commit `d2ba5106ae`), no conflict; the layout move of 5:04 AM PST is in it. `main` is now `5c0ae25d6c28`, 60 commits on: `git merge-tree` against it exits 0. Of my 13 files it changed one since, `benchmarks/README.md`, in other rows. [ran]
- **The changed suites at the final head:** `r20261112-150642-ed97`, done, exit 0, 11 of 11 passed, none cached. Its record names the commit `f919f768d435`, tree not dirty. `benchmarks/scan_cost` 25; `verity-network-traces` 607; `benchmarks/one_stage` 571; `verity-network-e2e` 70; `verity-network-sim` 21; `verity-check` 947; `verity-ops` 575; `verity-slack-events` 78; `verity-graph` 215; `research` 4648 and 3 skipped; `repository` 350. [read, the run's own log and record]
- **The body** is replaced (11,820 bytes) and has: the baseline's run, the one line on the gap to the proposal's figures, the three gaps as not measured with what would measure each, the merges.

**`program.json` (your condition)**
- **The file that keeps each digest is the session's `lean-verify.log`:** the verifier's command names its circuit and its program by SHA-512. `archive-put.log` lists the same digests. The body names both. [read]
- **`run.sh` checks it for each file before it deletes it** (commit `f919f768d4`): a copy goes only where a kept `lean-verify.log` of its round holds its SHA-512; one no log names stays, with a line. So the result did not need the five digests written into it. `scan_zk.sh` and `scan_zk.py` are unchanged by it.
- **On a hard-linked copy of the baseline's own `out/`** it removed all 11 circuit texts and all 5 programs, left the other 294 files and `joint/program.json`: 2,091 MB became 183 MB. [ran]
- **Not run on a node in this form.** The second rehearsal (`r20261112-145219-d446`, test class, at `d2ba5106ae`, done, exit 0, guard holds, 17 measurements against `main`'s kit) ran the earlier form, the circuit texts alone, and left `out/` at 118 MB. The form with the digest check is held by a test and by the copy above.

**The rest of your 6:55 AM PST order**
- The README's target now opens with what a session pays once, with no figure. The proposal's section E is marked stale in my note (notes commit `4f02d07bcb`).
- Against lean's landing tip (`f3837f57d113`): 31 conflicted files, the same 31 as `main` alone, none of my 13. [ran, at this head]

**Mine to own**
- **I launched the suites twice.** `r20261112-145030-a002` went at `d2ba5106ae` before your 6:55 AM PST order moved the head. It ended done, exit 0, 11 of 11, and is superseded.
- **Two header times were wrong:** my numbers message said 6:52 AM PST and went at 6:48; my #9296 head message said 6:04 and went at 6:02.
- **#9296's title:** my message said it was replaced before the command had succeeded; it failed for want of `--body-file`, I ran it again, and the title stood changed before your mark.

**Tonight**
- Taken: from `main` at `d7a5af15c9dc`, named in the launch line, no later tip without your word; vllm's tree `30f02e905e01` in the table's heading as vllm's word.
- I replaced my own timer, whose text had the old launch rule, with a backup at 10:25 PM PST that names `d7a5af15c9dc`. Your resume at 10:15 PM PST is what I act on.
- Nothing of this benchmark runs from 9 PM PST. The G = 8 run waits for your word. NET-380 starts after tonight's tries; its plan line by Fri 2 PM PST.

