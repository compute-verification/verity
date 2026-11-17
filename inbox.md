[2026-11-17T15:37Z] 9fd61c58157f
From the lead, 7:40 AM PST: main moved to e195d4a89c42 (the calibration move #10835: hardware/ is calibration/hardware/, 19 directories left benchmarks/ for calibration/). Your pushed head 68279ae4d398 of #10678 conflicts with it by my git merge-tree. Cross it too before you return: if the move carries a Move:/Layout-move: line, by tools/move/restack.py against the move own commit, not a plain merge. Say in your return which road and what conflicted.

