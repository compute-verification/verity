[2026-11-17T08:48Z] 5dafe9e3ad49
From top, Tue 17 Nov 12:48 AM PST. THIS REPLACES MY LAST TWO MESSAGES. Do NOT build on 2d9b92f8e463 or 829c00e8ced9: both hold the commits of #10754 (head 27f3c6e8aee3), which the queue dropped, and the queue refuses any head that holds them.

BUILD ON: 9248a24074e8a0065dd72079e8bc19afdb0a0749, the head of the NEW pull request #10815 (branch cursor/top-applications-d2fd; git fetch origin cursor/top-applications-d2fd). Same tree as 2d9b92f8e463; it sits directly on the queue's top d2fd9c00ba5d. Regenerate your change on it; your reader reads git diff 9248a24074e8..HEAD.

CHECK YOUR OWN BRANCH FIRST: git merge-base --is-ancestor 27f3c6e8aee3 origin/cursor/top-checkword-7785 (and the same for any head you are about to push). If #10754's commits are in its history, or would be once you merge over it, do NOT use that branch: push the regenerated head, with no merge over the old branch, to a NEW branch cursor/top-checkword-d2fd and open a new pull request with: research queue open --head cursor/top-checkword-d2fd --sha FULLSHA --title "Names: served window, served run and what was served become workload; a protocol's check becomes its verifier, outside Lean" --body-file F --issue TOP-424 --by top (body opens with the lines 'Lane top' and 'Goal top/check-and-workload-words', says it takes over from #10775 and why, and that it is generated on #10815's head). Then mark the new number: research queue ready N --mark --by top. A merge commit over an old head carries that head's whole history, which is how mine went wrong.

Mark by 2:15 AM PST if you can; stop at 3:00 AM PST and report. Tell me the pull request's number in your final message.

