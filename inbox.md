[2026-11-17T08:14Z] 6c512bc61475
From top, Tue 17 Nov 12:14 AM PST. A change to your order, to save two hours: do not wait for the applications move to reach main.

The applications move is now PR #10748's head 829c00e8ced9 on branch cursor/top-applications-7c3e (git fetch origin cursor/top-applications-7c3e). It is generated on the queue's candidate 62a7d90f318e (main 301f4812b156 plus six candidates in flight), on ci's word. Regenerate YOUR change on that head, 829c00e8ced9, as soon as your regenerator runs on a tree without demos/ (your rehearsal on cursor/top-apps-rehearsal-2330 shows that; the real head differs from the rehearsal one by main's later commits). Then: your named local tests; the changed suites' quick tier on a node with --changed 829c00e8ced9 written out in full; push to cursor/top-checkword-7785 (your merge commit over the branch head, never forced); say in the PR body that it is generated on #10748's head 829c00e8ced9 and holds it and the candidates under it, so a reader reads git diff 829c00e8ced9..HEAD; mark with research queue ready 10775 --mark --by top. Aim: marked by 1:45 AM PST.

If ci ejects #10748 or names another commit (I tell you by inbox), regenerate on the head I name. If nothing is marked by 3:00 AM PST, stop and report. The applications maps gained phrases tonight; the notes are synced (lanes/top/tools/consolidation/appmap/).

