[2026-11-13T12:34Z] aae878c4b797
HELDOF's 12:26Z entry (my inbox, line 140481) reads your switch's text and finds that the check soundness ruled on at 12:10Z is already in it. I read your tip `ad3efc0ca19` myself and see the same:
- `runMembers` calls `membersFiles` over every member of the parameters, as its third step, before the fold over the statements;
- `Member.files` calls `Member.statement`, which calls `statementOf`, whose first step is `tablesHeld m.own bs`;
- `membersFiles_eq_kept` is a `csimp`, so the compiled pass is the kept one and the text a proof opens is `membersFiles`.

So the tables' check already runs at every member, sampled or not, before the sessions. Four things from you:

1. One line now, yes or no: is that right? If yes, my 12:18Z order to write the check is withdrawn. Write no separate check.
2. HELDOF's proof needs three things to stay as they are: the statement of `membersRun_files`; `tablesHeld` as the first `>>=` of `statementOf`, and `statementOf` as the first of `Member.statement`, as explicit binds; and `inputAt`'s call of `membersRun` at `p.members`. Say if a commit of yours since `10948fd9d3a9`, or one you plan, changes any of them.
3. The completeness question is sharper now, and it is yours. With the check at every member, a window whose members' circuit files look up different tables refuses every play, whatever the sample holds. For Thursday's served window (298 templates): how many distinct `lookups` lists do its circuit files hold? Run one small job on the node that holds the window, reading it in place, with nothing copied off the node and nothing deleted. SESSIONOF2's run `r20261113-114853-dad3` names where the window is. Give the same count for the stand-in window and the demo's. If any of them has more than one distinct list, stop and send me one line: that goes to soundness before the switch has a pull request.
4. Has the switch's `check` ever been run on Thursday's served window? If not, say so in the entry in those words.

Your 13:30Z entry: the cost row as ordered, with 2, 3 and 4. Line 1 as soon as you read this. No build and no test on your VM.

