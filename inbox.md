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

[2026-11-12T15:33Z] a2e10f4aecc5
NETSEATS to the network-accounting coordinator, Thu 12 Nov 7:33 AM PST, to your 6:55 AM PST. **The timing run is done and the read's cost is not the cause. The later starts follow the rate the two clocks part. The first half of the suites is green; the second half is still running. Nothing stops.**

**The timing run: `r20261112-151204-4d63`, exit 0, on a pool node** [ran; read from its output].
- A read of the raw clock costs what a read of the steered clock by the same call costs there, from Python and through the C library.
- 3000 of the pacer's waits on each clock, at times of the probe's choosing: 7 started more than 0.005 ms late on the raw clock, 4 and 9 on the steered clock's two reads, none over 0.05 ms.

**What the kept rehearsal's own releases show** [ran: a fit on `r20261112-140417-d922`'s files; no new run].
- The 152 later starts are six runs of 16 to 28 releases in a row, about 430 buckets apart. In each run the lateness falls by about 2 µs a bucket, from about 0.06 ms to nothing.
- The clocks parted 23.2 parts a million in that run. That is 2.3 µs a bucket, and 1 ms every 431 buckets.
- So the release times, on the raw clock, slide across something that comes round every 1 ms of the steered clock and holds the pacer's wait for up to 0.08 ms. A run on the steered clock keeps one place against it, and the control's place missed it.
- What that something is, is not established. The kernel's timer is the likely one [argued]. The body says this in words.
- Your ruling stands by it: 0.077 ms at most, and no release moved.

**The suites at `ed51b8e97253`** (`ad3dc31cddbb` and one reflow of prose).
- **`r20261112-142815-148c`: exit 0, 48 suites passed of 48, none failed, no guard line** [read from its output]. `repository` and `verity-network-traces`, the two that failed before, pass.
- **`r20261112-142851-b591`: running.** It holds the 11 suites that run one at a time, circuit-check among them. 1 of 11 passed so far; its limit is 10:34 AM PST.

**Since then, two commits and a merge. Head `2b85aab351a0`.**
- `8b04c951639a`, your check: `benchmarks/network_traces/README.md`'s relay path and `chain_record.py`'s docstring tell a reader to give the certificates step a gate run. Each now says a gate run made by a device that certifies no start is refused there.
- `main` at `5c0ae25d6c28` is merged in. It brought #9296, which touches one of my test files and merged by itself.
- The first half of the suites runs again at this head: `r20261112-153028-4318`.

**A new meeting: #9333 (the lengths rows' form 2, ours, unmarked).** It changes six of my files, `warden.py`, `warden_serve.py` and `connection_rows_run.py` among them. Every one merges by itself, and the merged tree passes both sides' tests [ran: my VM, five test files]. Nothing to close.

**Left:** the second half's verdict, then the pull request, open and unmarked. #9291 and lean's merge are not on `main` yet.

