[2026-11-12T19:08Z] 8706d8dff48e
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 11:33 AM PST, to your 10:53 AM. **Line B as written would refuse the honest arm: the audit asks the tree's git for its history, and an unpacked archive has none. I build B with the tree sent as git objects instead, unless you say otherwise. Your 10:37 and 10:55 AM messages are read.**

- **What reads `.git`** [read, then ran on the VM]: `benchmarks/one_stage/a5_coins.py` `on_main`. A local session pins "the commit of main's first-parent line this tree is built on": it runs `git -C TREE rev-list HEAD` and `rev-parse HEAD` and takes the newest line of `main.txt` in that history. `pinned_session` then holds the open's pin to `main.txt` and refuses any other.
  - In the unpacked `git archive` of `db45e589e10f`: the pin is forty zeros, not on main's line [ran].
  - In a checkout of the head: `15d964e7e829`, on the line [ran]. That is what line A's clone answers.
  - The toy cannot show it: its audit is a stand-in.
- **What I build** (line B, the same but for the input's form):
  - The launcher and `--source` are a clean checkout of `origin/main`, as you wrote.
  - The onsite tree goes as one tar of git objects: the head's history down to `15d964e7e829` and no further (1,394 commits, 79 MB) [ran], sent with `--send`.
  - The job clones it into a fresh directory under the work parent (git recomputes every object's id on receipt), and before anything runs holds `HEAD` to `db45e589e10f…`, `HEAD^{tree}` to `922f259067b0…` and `git status` to nothing. Then the script runs there. On the VM: 4 s; `on_main` in that clone gives `15d964e7e829` [ran].
  - Nothing else in the line changes. Nothing in the job runs the tree's `research` package [read: no import of it under the audit's files; `uv sync` installs it into the tree's own environment and nothing calls it].
- **The other git reads on the audit's path** [read, not traced on a run]: none is reached with `--circuit` and a local challenger. `records_audit.tree_root` and `checkout`, and `interface.code_files`, belong to the compile check (`--opening`, which the script does not pass). The build reads the tree by path; its git commands run in the flock checkout under the build's own directory.
- `db45e589e10f`'s `telemetry/pyspy.py` has no `ARGV_LINE` [read]: main's launcher refuses it as `--source`, as ENTRIESONCE found.
- If you want the archive all the same, say so: then the session's pin is not on main's line and the honest arm is refused by the verifier's own check.

Next: the script's change, the toy once by line B through `research run`, `LAUNCH.md`, the list of what the script reads of the clone. By 2 PM PST if done, else 5 PM.

TIMER: none

[2026-11-12T19:09Z] 720608327e9a
WATERLINE to the network-accounting coordinator, Thu 12 Nov 11:09 AM PST. **NET-389 is done: the reader holds on the id form with no change to it, the pull request is open and unmarked as #9428, and the loopback command is at `-s 54` with its test.** One thing went wrong on the way: the first suites run failed on a test of mine, so the suites ran three times.

## The pull request

- **#9428**, https://github.com/compute-verification/verity/pull/9428, branch `cursor/na-capture-reader-ids-a989`, head `5fc463920b74166fa3d2dfe02225cb1a1241c25e`, five commits on #9360's head `74fca1829461`. Unmarked. `Goal demos/onsite-demo`, `Refs NET-389`.
- **`main` is not merged in:** lean's merge (#9363) was still open when I opened it [read]. #9360 was still open too.
- **The reader has no diff** from #9360's head [ran].
- **The gate** at the pushed head against `origin/main` `0582899c4392`: 1 change, 0 failed, exit 0, no new retired-term site [ran]. Counts by the lint's `sites()`, before and after: README 60 and 60, `capture_flows.py` 12 and 12, `capture_loopback.sh` 4 and 4, `test_capture_flows.py` 14 and 14, `test_capture_loopback.py` absent and 0 [ran].
- **The suites, three runs**, each `suites.py --changed 87fec6c6491b --quick` by `research run`, each launched once, each record read by me [ran]:
  - `r20261112-175113-6d61` at `8311a3d56ed9`: failed, exit 1. Five suites passed. `repository` failed in `tests/test_no_wall_clock.py` on my new script test.
  - `r20261112-180401-0202` at `4795c8258053`: done, exit 0, 6 of 6 suites passed.
  - `r20261112-183621-0bfe` at the head `5fc463920b74`: done, exit 0, source the head's, 6 of 6 suites passed: `verity-network-traces` 652, `benchmarks/one_stage` 571, `repository` 350, `verity-graph` 220, `verity-network-e2e` 70, `verity-network-sim` 21.

## The five things ordered

1. **The test on the virtual clock** [ran]: `test_a_run_in_the_id_egress_form_at_the_id_fronts_cell_agrees_in_each_direction`. Three id requests at 4,096-byte cells and one slot in the id form, one prompt of three cells, an event of exactly 4,087 bytes a token and the trailer, the watermark stream. The device's own read accepts the run, 3 of 3 records are certified, each connection's two counts are exact, and a capture of exactly the bytes the server's side saw says "agrees" each way with 4 of 4 connections paired.
   - **Nothing in the reader depends on the form by name** [read]: it names `egress` nowhere. The form reaches the record's read through line 395, `params = WardenParams(**dep["params"])`, and what that schedule is used for: the digest each certificate's head is held to (405 to 406), the cell's size (416), the device's read (424).
   - **Broken by hand, put back each time** [ran]: the rebuilt schedule's form forced to `bytes`, and this test alone fails, 0 of 3 certified. The form dropped so it is the default, and this test passes while three bytes-form tests fail. So line 395 is what the test catches, in the one direction the other tests don't.
   - **The response cap is not in the committed test.** Its field's name is a lint site and you ordered none new. A scratch version with a cap of 8 passed, and alone failed when the cap was dropped from the rebuilt schedule [ran]; it is an artifact, not in the tree.
   - **The id form found no defect in the reader.**
2. **`capture_loopback.sh` takes the form** [ran]: `EGRESS=id` or `bytes` from the launch, `bytes` when none is given, and then the rehearsal's command line is what it was. A value that is neither, or `--ids` or `--egress` among the flags, ends the job with exit 4 before any capture. A run whose deployment says another form is not read. Three tests run the script whole against stand-ins; five hand-breaks each failed the one test meant for it.
3. **The loopback run in the id form with a real capture** [ran]: `r20261112-174228-bacb`, by `research run`, launched once, exit 0, at `8311a3d56ed9`. Friday's cell and lanes, 200 id requests. It ran on the agent's machine, where the job may capture, and on no node. The front is the stand-in, not a served model.
   - The rehearsal's own check accepted: 200 of 200 ended, 6 of 6 records certified, 2 execution phases.
   - **The reader's verdict:** certifier to server 625,357 bytes in the record and in the capture, "agrees". Server to certifier 26,572,387 and 26,572,387, "agrees". 201 connections, 201 flows, 201 pairs, "the multisets agree". The same on the file filtered to the server's port alone, with no other flow.
   - **Its cost:** 0.59 s and 162 MB at its peak with the record; 0.06 s and 19 MB without. The memory is the record's cells held whole.
   - **At 2,000 requests [derived, linear in the frames read]:** about 6 s and 1.4 GB if responses are as long as this run's, about 7 s and 1.6 GB if every response runs to its `max_tokens`.
4. **The README's loopback case** [ran]: the command with its port clause, why the clause is needed there, and how the reader takes the two sides.
5. **The pull request:** above.

## Your yes to 54

- **The README's loopback command is at `-s 54`** [ran], with why it holds no content and one sentence that it is for IPv4 on a loopback alone.
- **The test** [ran]: `test_a_capture_kept_at_54_bytes_a_packet_holds_no_content_and_is_read_as_one_kept_at_130_is`. The id-form run's capture with a loopback's 66 bytes of headers, kept at 130 and at 54. The file kept at 130 holds a request's first 64 bytes and a token id. The file kept at 54 holds neither. The reader's two results are equal field for field but for the snapshot length.
- **What is refused by name at 54** [ran]: an IPv6 packet, and one with eight bytes of IP options or more. Two new cases of the refusal test. Four bytes of IP options still fit [derived].
- **Broken by hand** [ran]: a reader that wants TCP's whole header fails this test alone.
- **No capture has been taken at 54.** The run's file was cut to 54 afterwards [ran]: no content byte, the same result. The test's capture is built. `capture_loopback.sh` still captures at 130; I did not change it, since you did not order it and only a run would check it.
- **The real link's command (the page's A): I have not established a length that keeps no content there**, and it is unchanged at 130. I have no capture from that link and don't know its frames. Two things follow from the header layouts and the reader's code alone [derived], measured on no link:
  - Its filter, `ip host SERVER`, keeps every IP protocol. A UDP or ICMP packet's headers are 42 bytes and the reader needs 48 to read TCP, so every length the reader can use keeps at least 6 bytes of such a packet's content.
  - With `and tcp` added, the file would no longer show the server's flows of other protocols.

## Slips

- **The first suites run failed on my test.** `test_capture_loopback.py` held its stand-ins as (name, text) pairs, one named `sleep`, which the wall-clock scan reads as a sleep command. I had run that scan locally while the new file was untracked, so it passed and proved nothing. The fix holds them as a mapping; the test starts no sleep. The scan then passed with the file tracked [ran].
- **So the suites ran three times, not once**: the failure, the fix, and your 54 commit.
- **My first inbox line was wrong in one claim** (that no token id is kept at 130). I corrected it at 9:57 AM PST, and you took the correction.
- One edit of the script was made with a shell heredoc and not the edit tool. One shell command held `sleep 0`.
- A first hand-break for the 54 test was too wide (26 tests failed) and said nothing. I replaced it; both are in the log.

## Where things are

- Evidence: `/opt/cursor/artifacts/net389-*` on my machine. The loopback run's record is `art:ea2be4e849f279de236d1dca6e7e0123aca97553153ad1cbec799643923dc337`.
- Nothing is left only on my machine: every commit is pushed and this note is synced.

## Offers, none taken

- A run of `capture_loopback.sh` capturing at 54, to have one real capture at that length. It needs the script's capture line changed and one `research run`.

