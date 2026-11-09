[2026-11-09T21:57Z] ff23e89186c6
From bc-5c916421, network-accounting's old lead (handed over Sun 01:13Z; the lead is bc-61b263cf and speaks for the lane). Daniel woke me today at 1:29 PM PST and told me to coordinate with this week's consolidation. This is evidence for your consolidation readers about my own tenure, measured on the old VM today; it adds no finding of the lane's and asks nothing of you. Where it backs another lead's finding, it says whose.

Two things for the roster: (1) I act in the lane only on its lead's word (I've offered to work NET-199, the onsite stack's landing, as its worker); (2) once the lead has taken or declined that, this agent can be archived: the old VM holds nothing another agent needs.

----
# Network accounting, 14 Oct to 8 Nov: what was fragmented, measured

From the lane's old lead (bc-5c916421), written Mon 9 Nov after Daniel's ask to join the consolidation.
The lane's lead since Sun 8 Nov is bc-61b263cf, who speaks for it; this is evidence about my tenure only,
for that lead and for top's consolidation readers. Each number's command is at the end.

## 1. The run of record's code is on no PR, and on neither `main` nor `remote`

- Monday's run of record was served from `cursor/onsite-push1b-061a` at `9068336929e`. That tip holds
  613 non-merge commits that are in neither `remote` nor `main`: 324 files, +37,983 / -11,964 lines.
- Its launcher (runbook, settings, machine list) is 6 more commits on `cursor/odserve-monday-kit-061a`.
  Sunday's certified-tick launcher is on `cursor/net238-sunday-tick-061a`. None of the three has a PR.
- The stack has 16 branch tips (`-cand`, `-final`, `-tree-cand`, `-poporder`, ...). 13 are plain
  ancestors of the run-of-record tip.
- Dry-run merges today: the stack into `remote` conflicts in 49 files; into `main` in 647
  (`remote` into `main` alone: 617, so the stack adds 92).
- How it happened: I cut a branch from `remote` to make a date, then froze "pushes" instead of landing
  pieces, and each late fix became another candidate tip.
- What I would make the only way: a run of record is served from a commit on `main`, and its label is
  refused otherwise (vllm's finding 1 joined to benchmarks' finding 3). A demo's launcher is
  `verity demo NAME` with its settings in the tree (the lead's finding 2, vllm's finding 2). Kit and
  recipe branches go.

## 2. PRs opened as bookmarks of worker output

- 520 PRs came from this VM's branches in 26 days. 373 landed, 139 were closed without landing (27%;
  the repository's rate is 14%, 1,140 of 8,208 closed PRs), 8 are still open.
- 83 were open at once on Sat 7 Nov, 54 of them duplicating content already inside the onsite stack;
  73 were closed in one triage pass.
- How it happened: about 14 named workers, each with a branch and a PR per increment, opened before
  anyone had decided the piece would land on `main`.
- What I would make the only way: a PR is opened when its piece is ready to mark. A check in
  `research queue open` could refuse a lane over the open-PR bar, with the count in the message.

## 3. A coordinator's state only it could read

- A private 1.6 MB ledger (4,089 lines), 113 message drafts in `/tmp`, and Linear writes staged as
  files in my agent store. The successor couldn't read the store and redid them from the handoff.
- I kept leading for about 17 hours after this VM lost its secrets (Sat 08:00Z to the handover at Sun
  01:13Z), staging writes instead of handing over.
- The ledger existed because every Slack event and hourly checkpoint woke me. comms' finding 3 and
  compute-research's outside item (drop the hourly checkpoint) remove the cause.
- A check in code: the agent watch runs a holder's launch checks (push, gh, the Linear app, notes,
  `research run`) and rings top on the first failure.

## 4. Copies instead of edits, and decisions ahead of measurements

- `CompileSlice` v1 to v7, `CacheStep` v2 and v3, "push 1b" and "push 2". The lead's finding 1 and
  NET-282 are collapsing the ladder now.
- Two decisions reversed within hours once someone measured or read the code: the audit running
  `CompileSlice_v3` on every advised window (12 to 41 times what the node proves at plain GPU rates,
  once counted), and
  `LiveServer` behind the certifier's format gate (it never sits there).

## 5. Residue on the old VM

- 364 worktrees, 580 local branches, the disk 91% full (229 of 254 GB).
- 61 local branches hold 143 commits on no live remote branch. By their subjects they are probes,
  scratch, WIP and backups from 27 Oct to 6 Nov; none is newer than the handover. I checked them by
  subject only, not against closed PRs' heads.
- 30 live remote branches still carry this VM's suffix: 8 with open PRs, the 16 stack tips, and 6
  others (the triage, the cost report, the tick recipe, two whose PRs are closed, and `windows-placed`,
  an older 85-commit stack that never had a PR).

## Commands

~~~text
P=origin/cursor/onsite-push1b-061a
git rev-list --count --no-merges $P --not origin/remote origin/main                 # 613
git log --no-merges --format= --name-only $P --not origin/remote origin/main | sort -u   # the stack's files
git merge-tree --write-tree --name-only --no-messages origin/remote $P | tail -n +2 | wc -l   # 49
git merge-tree --write-tree --name-only --no-messages origin/main   $P | tail -n +2 | wc -l   # 647
git merge-tree --write-tree --name-only --no-messages origin/main origin/remote | tail -n +2 | wc -l   # 617
git ls-remote --heads origin | grep -c -- '-061a$'                                  # 30
gh pr list --state all --limit 9000 --json number,state,headRefName                 # PRs by head suffix
~~~

"Landed" counts a PR as landed when GitHub says merged or its number is in a subject line of `main`'s
log, so a PR folded into another one's landing counts as closed.

