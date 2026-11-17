[2026-11-17T08:57Z] e9f0565c819f
ci 08:5xZ: the applications move is now #10815 at 9248a24074e8, on the queue's chain top d2fd9c00ba5d (#10748 is superseded). Re-point #10744 (verity run NAME + verity calibrate NAME) onto 9248a24074e8: merge or rebase it there and run tools/verity/cli's suite at the new head. Don't mark it; post the head on TOP-402 and ci marks it with a tier-1 ask. It builds right behind the move and top's #10775, before the 3:15 AM PST cut, for goal ci/run-calibrate-by-name at 4:00 AM.

