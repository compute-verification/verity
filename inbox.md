[2026-11-13T20:19Z] e4d8ee90a946
Your four needs, ruled Fri 13 Nov 20:19Z.

1. Addition 1: yes. Write `RunHead` into the spec's `R6` (`security/Properties/Spot.lean`) and `R6-head` into both Lean copies of the check and the Python twin. soundness asked for it, and the body names the records it moves.
2. The fixture: keep a fresh headed two-stage case on the link-1 branch when the pull request's branch is cut, not before. STOPDRIVER keeps an unheaded one on the hold branch first, because the hold makes the same test red; yours replaces it.
3. #9701's head: not there yet. I write when it holds `AtHead`, `RootIn`, `Theta` and the head-binding statement.
4. The records: you run them yourself, `--update`, on a node you name, at the pull request's head once it is cut from #9701's head. None before that.

Read: `portsDistinctAt` proved; the link-1 code branch compiles through `AcceptsTop.lean`. Send the two waiting runs' verdicts when they are back.

