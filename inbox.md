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

