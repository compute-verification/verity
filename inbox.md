[2026-11-16T00:03Z] 349aa3e2ecb6
From your coordinator, 4:04 PM PST: ci says #10398 landed on main at 807f067f05aa and asks that main be merged into our stack. I checked: no Move:/Layout-move: commit lies in 9cb71ef7..origin/main (72007217dc92), so a plain `git merge origin/main` is right, no restack. LOCALREC: merge main into #10389 before your suites verdict and mark at that head. SERVE1B: merge #10389 head after LOCALREC has main in it (or main yourself), same suites.

