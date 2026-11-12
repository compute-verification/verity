[2026-11-12T17:47Z] 5ea202e55185
TERMGB to the network-accounting coordinator, Thu 12 Nov, 9:50 AM PT. **One stop needs your word before I choose: the rule's default under lean's `PEARLPLUS_TIE`. Everything else in the six files I am closing now. If I hear nothing by 10:30 AM PT, I build the scratch with what #9305's body says for it, so the runs can go; it is one line to change.**

**Lean's branch moved before I took it.** It is at `238ad6621c27` (9:36 AM PST): your `ca2e80f1edf7`, a merge of `main`'s `6d95fcc68dd3`, and one commit. I took it there, once. The six stops are the same six [ran, `git merge-tree`].

**No branch of the other agent is on origin** for #9305 or #9329: only my own five `-b4e6` branches match [ran, `git ls-remote`].

**What lean's side does not change** [read, its diff from the merge base]: what a rate means. No line on rates changes in `service.py` or `draw.py`, and its `records_audit.py` reads `--law` where `main`'s does.

**Where lean's side adds a reader of the rule: the Pearl+ tie in `served_verdicts.sh`.**

- Under `PEARLPLUS_TIE` its default is `V3C_LAW=stratified:1` (its line `[ -z "${H1_TIE:-}$PEARLPLUS_TIE" ] || V3C_LAW=${V3C_LAW:-stratified:1}`), its preflight asks for `stratified:*`, and its `pearl-tie` hands the rule to `window-commit --pearl-passes`.
- That window is not the one I counted. It is the BF16 checkpoint, cut at Pearl+'s linears [read, the script's header]. Nobody has counted it.
- So the rate that stands for its `stratified:1` is not known. `stratified-rate:1/3481648` registers `stratified:1` there only if no template of that window has more than 3,481,648 proof blocks; with a larger one it samples more of it [argued, `draw.strat_k`].

**The choices**

- **A (what #9305's body says, and my proposal):** no default under `PEARLPLUS_TIE`. The preflight asks for `V3C_LAW`, a stratified rate, and says so. A Pearl+ launch that states none exits 3 before it serves.
- **B:** the same default as under `H1_TIE`, `stratified-rate:1/3481648`, with the comment saying it samples one of each template while none has more than that many blocks. Nothing is refused; the sample is lean's unless a template is larger, which is not established.
- **C:** count that window first (a CPU run of the class circuit at the BF16 registration with Pearl+'s tables), then state its own rate. More work than this job's; say if you want it.

**Also in the merge, as the body gives it:** the weights-tie lines (`run.sh`'s `PP_LAW`, `weights_tie.py`, its test). They still apply to the merged tree [ran, `git apply --check`]. I put them in a commit of their own after the merge commit, so you can read them apart.

TIMER: none

[2026-11-12T17:51Z] 820a601986b3
NETSEATS to the network-accounting coordinator, Thu 12 Nov 9:51 AM PST, to your 9:15 and 9:39 AM PST. **The meeting with #9291 is closed: head `08740c9dfa46`, and `test_warden_probes.py` passes 12 of 12 [ran: my VM]. The pull request is #9369, open and unmarked at that head.** `main` is merged in at `da1bdbda0648`, which holds #9291. Both tests are kept, and mine takes its `probe_mod` and `runner` fixtures.

**Your 9:39 AM PST order is read and I start it now.** A scratch branch of mine from `08740c9dfa46`, lean's `cursor/lean-remote-merge-onmain-fc8d` merged into it there, nothing of lean's pushed to the pull request's branch. I send its head, what stopped and each run when it is built.

[2026-11-12T17:54Z] 298f67f06a50
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 9:55 AM PST. Friday's three things. Nothing launched; no limit and no path changed.

## 1. The member's line [ran on the toy, seven runs]

- Script: `0d9116903be538d410751ffb7de1b257c361aa08e1d9b51b02f75071db8371e4`. Reader unchanged: `67d04ddeb9040ea292b663f6cdba6832bd45d36d814d2b80626fb9a576ed3a48`. Both in `/tmp/rehearsal/` and in my store beside `LAUNCH.md`; the 9:33 files are kept as `prev-0933/`.
- The line, printed once, within 5 s of the session's directory appearing (the sample is known a moment before):
  `REHEARSAL +Ns sampled: the member's circuit file is B bytes; by the table the audit ends about R s from here, at about +Es, inside the job's 12600 s`
  It says `PAST the job's … s` where E is over the job's seconds. It prints no index and no count of the window.
- R = 1,460 s × B / 3,764,130,558 + W [derived from Monday's kept run]: 1,460 s is what the member took there in its session, its Lean check and the registered reads' four checks; W is the outcome and the negatives' workers, 2,020 s at seven workers or more, 2,600 s at five or six. Under five the line gives the bytes and says the table has no seconds.
- The toy, with the stand-in audit [ran]: 3.76 GB → `about 2020 s`; 20 GB → `about 9777 s … inside the job's 12600 s`; 20 GB with `JOB_SECONDS=16200` → `inside the job's 16200 s`; 20 GB with a job of 5,000 s → `PAST the job's 5000 s`; 8.2 GB at six workers → `about 5780 s`; three workers → no seconds; the memory stop still ends with exit 72 after the line. Exits 0, 0, 0, 0, 0, 0, 72; the work parent empty after each.
- What you read at 6 AM: E against 12,600 (or 16,200). By the table E passes 12,600 from a member of about 18.4 GB at seven workers, about 17 GB at five or six; with the five-hour line from about 27.7 GB and 26.2 GB.

## 2. `LAUNCH.md` again [written; the custody check ran]

- In my store, `rehearsal/LAUNCH.md`, final as of 9:53 AM PST: the new digest, both lines whole, the `sampled:` line under "what the coordinator reads".
- The five-hour line differs in two places only: `--timeout 5h` and `--env JOB_SECONDS=16200`. 1,800 + 18,000 = 19,800 s passes the custody key's check without `--custody-ttl` [ran].
- It says: the coordinator names the line at 4:48 AM PST; until infra has said yes, the four-hour line is the one; if you rule another count, only `V3C_NEGATIVE_WORKERS=N` changes.

## 3. Five, six or seven workers [derived from Monday's sampler; no run of seven at this size exists]

The member's size enters neither the workers' seconds nor their memory: no worker's task reads the member's circuit [read], and the audit's Python stays flat at 48.4 GB through the session and the member's check [ran]. So the three columns differ only in the total.

| workers | workers' seconds | memory, my reading | memory, every worker un-sharing all | total 3.76 GB | total 8.2 GB | total 15 GB |
|---|---|---|---|---|---|---|
| 5 | 2,600 | 196 GB | 461 to 483 GB | 7,500 s | 9,200 s | 11,900 s |
| 6 | 2,600 | 224 GB | 542 to 569 GB | 7,500 s | 9,200 s | 11,900 s |
| 7 | 2,020 | 253 GB | 624 to 655 GB | 6,900 s | 8,600 s | 11,300 s |

- **The largest count whose worst reading stays under the stop at 8.2 GB: six** (542 to 569 GB, 81 to 108 GB under 650; the same at every size). Seven's worst reading is 624 to 655 GB: on the stop's line.
- Seconds: Monday's stage is eight blocks of about 866 s (the outcome, then seven coin negatives), then R2, R3, R4 in 788 s, the transfer gates and the audit record in 184 s [ran]. Seven workers: 866 + 866 + about 300. Five or six: 866 + two waves of 866. Six costs 580 s against seven; five costs the same as six. At 15 GB with five or six the total leaves about 700 s inside 12,600 s.
- Memory: the audit's Python at the fork about 53 GB (48.4 to 57.2 over the stage); a worker's own about 28.5 GB (its Lean server's peak 24.2 GB plus its Python's growth 4.3 GB). My reading: 53 + 28.5 n. Un-sharing all: 53 + 81.5 n; the upper end takes 54.6 and 85.7. The gate as coded reads about 54 + 77.2 n against 698 GB and admits seven at once (594 GB).
- I correct my 9:33 figure: 665 GB took the stage's peaks for both terms; the sampler gives 624 to 655 GB.
- The stop reads MemAvailable against its value at the script's start, so whatever else is held then counts with the workers; by Monday's sampler the session's memory is released before the workers start [ran], and the warm's page cache is reclaimable.

Least sure: the seconds for workers at once. Monday ran them one after another; I assume a block takes the same 866 s with six others beside it, and Lean's threads may slow each other. Next: whether a forked worker un-shares anything near all of its parent; nothing measured it at this size. I did not look at the node.

Going on to the port's recount.

