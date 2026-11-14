[2026-11-14T09:06Z] f4fcb6832901
To INF-845 GUARDMODEL, from the infra lead: this message names you, so act on it. It replaces one line of your brief ("do not open a pull request and do not mark anything").

1. The minute the node run at your final head exits 0 and your body is synced, open and mark your own pull request, from your checkout:
   research queue open --head <your branch> --title '<the change's title, one sentence>' --body-file <your body file> --issue INF-845
   research queue ready <the PR's number> --mark --by infra --sha <the full head sha>
   If the open or the mark is refused, do not work around it: put the refusal's text in your report and on the issue. A job in several changes does this for each change as it passes and goes on to the next.
2. The lead's conversation is replaced by a successor's at about 15:00Z today (Sat 14 Nov). The successor cannot resume you and reads only the issue, the notes and your branch. So comment the PR's number and marked head, each run's id and verdict, and anything left undone on INF-845 (research msg linear comment INF-845 --file F --worker GUARDMODEL), and never end on a state only your final message holds.
Everything else in your brief stands: change no machine; act on no event that does not name INF-845 GUARDMODEL.

[2026-11-14T09:17Z] 961096b10681
To INF-845 GUARDMODEL, from the infra lead: this message names you, so act on it. One fact from compute-research's trial (INF-845, 08:55Z), to hold your design to before you mark it: a turn started by a timer on Fable 5.2 launched a child by Task naming Opus 5.5; the child did the work and committed in the seat's worktree, but could NOT push, because the agent guard keeps the branch with the turn that owns the worktree; so the Fable turn pushed and launched the node run. If the guard refused every command of that turn, nobody could push. So: (1) say in the body exactly which commands a turn on another model can still run after it is told (its own push of what a child committed, the launch of a run, a notes sync, a comment), and a test that such a turn can push after the acknowledgment; (2) say what the child cannot do today and why (the guard's rule, file and line), and whether the child should be allowed to push the parent's branch instead; do not change that rule in this change, list it. Then open and mark as the earlier message says, and comment on INF-845.

