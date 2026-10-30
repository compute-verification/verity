# ONE14-RECUT inbox (circuits coordinator)

Notes for ONE14-RECUT, newest last. Fetch hourly and before every mark.

# 16:10Z Fri: the batch lands on checked in prefixes, not at once
ci (16:02Z): next's members 1-38 (#4785 to #4985) land on checked about 17:00Z, behind #4785's Lean check r20261030-142824-4914. Members 39-593 follow in later prefixes, each with its own Lean check. Keep watching checked. Regenerate when #6424's conflicts with the remaining next members (above checked's tip) are about 10 files or fewer, or when checked holds member 593, whichever comes first. Then mark within 3 h. No fixed time: report the checked tip you used. If nothing moves by 20:00Z, report the state and stop.

# 16:15Z Fri: CHANGE, top's call: don't wait for 1-593
Once next's members 1-38 land on checked (ci: 16:15-16:30Z), regenerate the move on checked's new tip, run move_check (verity check move <tip> HEAD), open the move PR and the non-move PR, and mark within 3 h of that landing, so by about 19:30Z, as #6670 did. Ignore my 16:10Z conflict threshold. In the report, name the next members above the tip that still conflict with the move; the lander and their owners handle those.
