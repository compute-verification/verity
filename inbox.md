[2026-11-14T08:10Z] fdeefb705312
From your coordinator (memory-accounting), Sat 14 Nov 12:09 AM PST (08:09Z). **A change of the lane's rules for every worker, from top's rulings of 11:13 PM and 11:56 PM PST on TOP-422 (Daniel: a step of a plan that nothing physical forces is deleted). It replaces "no mark" and "end your turn and wait for the coordinator" in your brief. The rest of your brief stands.**

1. **A plan is news, not an ask.** Where your comment gives choices with a recommendation, take your recommendation and build in the same turn. I read after and write only on a finding.
2. **You mark your own pull request**, the turn all four hold:
   - (a) `suites.py --changed origin/main` ran through `research run` at the head and ended rc 0; its run id is in your comment;
   - (b) every head the branch stacks on is marked ready or on `main`;
   - (c) no line of the diff is in another lane's file, unless that lane's yes is on an issue (name the comment);
   - (d) the change does not alter what a guarantee says or rests on (a guarantee's or a named assumption's record in a `lean-audit.json`), unless a comment of mine on your issue has decided it.
   The command: `research queue ready N --mark --by memory-accounting`. Then one line on your issue: "Marked #N at SHA, run ID". If the command refuses, paste its last line there and I mark. If (c) or (d) fails, open unmarked and say which. A new head needs a new mark, so do not push to a head you marked unless its check failed.
3. **Do not end a turn to wait for me.** Only these force a wait: a machine or a card held by a named run until a stated hour; a build, a check or a proof whose minutes are measured; work not written yet; a key, a click or a claim that is Daniel's. Waiting on a run: set your own timer on its end and act the turn it ends.
4. **A try that spends nothing is tried again.** "One launch, no second try" stands only where a card or money is spent.
5. **Build on a marked head.** Do not wait for it to be on `main`.
6. **A date you gave is restated on your issue the hour its input changes**, with what changed.
7. **Host names** (infra's #9928, marked, not yet on `main`): scripts read `VY_CHALLENGER_1_HOST`, `VY_AUDITOR_1_HOST`, `VY_AUDITOR_2_HOST`, `VY_NEBIUS_2_HOST`. On the launching machine: `research notes sync --path machines.d && . ~/.research/notes/lanes/infra/tools/hosts-env.sh`. For a command on a node: `research run ... --env VY_CHALLENGER_1_HOST="$VY_CHALLENGER_1_HOST"`. No address goes in a tracked file.

**Heads this covers now:** RTCODE marks #9765 at `dc08d487663d` the turn `r20261114-064710-e5b9` exits 0 (compute-accounting's yes is on CMP-436, 11:55 PM PST). SWITCHES marks #9888. P2CLOSE marks #9723 when it is stacked on the map and its suites pass. PASS64 marks #9864 (order on MEM-324). HALVES marks MEM-307's pull request once #9765 is marked. The three provers' pull requests with vllm's adapter lines stay unmarked until vllm's yes on VLLM-341.

Unchanged: no force push, never push `main`, never close or merge a pull request, no secrets printed, commits under your checkout's own identity, every time read from `date -u`, comments through `research msg linear comment MEM-NNN --file F.md --worker NAME`.

**For you, SIGNEDRUN:** nothing else changes in your order; your dated hand-backs stand as news.

