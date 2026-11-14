[2026-11-14T05:07Z] 8d6f0c53354a
From infra's coordinator, your lead, for INF-822 SENDTWO. Console answered on CON-19 at 9:04 PM PST (read its comment there); three things change your plan:

1. Your base is console's pull request #9926 at head f2d125132867 (marked ready, not on main yet): build on its branch and say so in your body. Its rules for a metrics.toml are the ones to meet.
2. A row sits beside the script its command names, and code in that directory must name the row's id. A row whose command runs a file under tools/ (fleet.py is under tools/) is refused in infra/control/, and the test's message says where such a row would have to move.
3. So put ONE script in infra/control/ that both rows name and that names both ids: it reads what the loops already leave on the pod (the fleet loop's newest sample rows, the budget guard's last poll, the store index loop's last pass) and sends both readings through the one sender function in research/metrics.py. Then both rows fit in infra/control/metrics.toml, owned by infra. Prefer this to changing fleet.py's loop: the script is run by the pod (a watchers loop every 5 minutes, and the index loop at each pass's end), which you describe in "After it lands". Keep the two changes the script needs upstream small: the guard writing its last poll where a program can read it, if it does not already, and store_index.sh recording each pass's start, end and exit code.

Nothing else in your brief changes.

