[2026-11-13T13:01Z] 2950e86ee274
NETSEATS to network-accounting (lead), Fri 13 Nov 5:01 AM PST. **Both rulings are built and pushed at `e13350660c0a5403a965b9b5f6872cd31c76adf5`. The chain record run `r20261113-125709-a3c7` is launched once at that head and was placed at once (no wait on the window). The changed suites run as `r20261113-125953-929b`. No pull request yet: it waits for the run, as ordered.**

- **Pins, checked before the writer changed** [ran + read]: nothing pins the bytes of the four writers' registered-reads files. No such file is committed; `fixtures/artifacts.json` names none of theirs; the one pins file that names a registered-reads file (`verity/protocols/flock/agreement/vectors.json`) names another writer's case; no test and no document holds a digest of one. The verdict's `registered_sha512` is of the file's canonical JSON, in Lean (`FlockVerify.lean`, `sha512Of`) and in Python (`stop_audit.py`), so the compact text changes no digest a verifier reads.
- **Ruling 1** [ran, local tests]: the certificates step writes `accepted`, how many problems, and the gate's list of counts; none of the summary's other keys and none of the problems' text. A test holds the step's list to the gate's by name and fails on a key more. The guard's fourth entry is out.
- **Ruling 2** [ran, local tests]: one function, `warden_rows.reads_text`, writes the four writers' files with sorted keys and no space. A guard test shows a port's reads written so are no piece of a record head's counts, and that the default form and a copy of the counts still are.
- **The job now runs the control itself** [ran on a gate run the tests build; the node's run is the one that counts]: a registration's `wiring` written back into a copy of the released `cells-registrations.jsonl`, read by the guard with `--dry` beside the released files as they are, in a directory outside the run's. On the test's gate run: exit 3, the written file alone found.
- **For your issue on values under 32 bytes, not ruled and not changed** [read]: the scan step's `summary.json` writes from the witness `admitted`, `anchors`, `responses` (each connection's count of response ids) and `cells` (data and end cells a side), and on a void run `void.why`, the text of the check's problems that name a connection failed closed (`scan_zk.py`, `prove`).
- **The earlier suites run** `r20261113-103408-7e46`, at the head before these two commits: 72 passed, 1 cached, 0 failed of 85 when I read it at 4:59 AM PST; still running [ran].
- Nothing carried, no merge of `main`, no rebase. I return when the chain record run ends.

TIMER: none

[2026-11-13T13:01Z] 04d86e53533e
ONSITEMAIN, Fri 13 Nov 5:01 AM PST.

**The rehearsal is launched, once: `r20261113-130017-cae8`, typed at 5:00:16 AM PST, the launcher's exit 0. H's pull request is #9736, open, unmarked, at `42a962a3b7d0`.**

- The script's digest read in the minute before: `257179d66b68877eb7cbaece4d7d204745374e51127fa0a1124da04f1879850d`; no `launched` file before, one now [ran].
- The launcher's lines [ran, read through the mask]: research from the pinned worktree at `5f0c008aa040`, its tree clean; `main.txt` held, 6,933 lines, its digest as held; the module's identity `d489c1ebe020fca4`; `--pool network`, `--timeout` 18,000 s from taking a slot, a wait of up to 30 min for one; `--mem-gb 645` goes to the slot; the run launched on the kept-runs node.
- I have not read whether the slot is taken yet: the launcher returned, and the run's `stdout.log` is yours at 5:25 AM PST. I look at its state at about 5:10 AM PST and say at once if it gave up or exited early. Nothing is rerun.
- #9736 holds #9664 and says so on its first screen, with the one merge commit of job 6's second head. Its run `r20261113-085713-e22d`: exit 0, the Lean check of the three packages exit 0, 59 of 59 suites [ran].

TIMER: none

