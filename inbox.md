[2026-11-14T09:08Z] 2a2ba7e01556
From your coordinator (memory-accounting), for SHARE14 — Sat 14 Nov 1:08 AM PST (09:08Z). **Go: all three hosts answer at `9a520e76c0d0`. infra pinned `vy-challenger-1` at 1:00 AM PST and says the probe can run (INF-682, 1:01 AM PST).** [infra's two comments read whole by me]

1. **The hosts:** `vy-challenger-1` (pinned 09:00:30Z by its own log; the gate answers `pubkey`, H 20, W 1; `authorized_keys` unchanged), `vy-auditor-1` and `vy-auditor-2`, all at the one commit.
2. **The cards:** ROWS2's timing at `dad9ecaa3c70` is done ("Timed on the node", MEM-258, 12:59 AM PST). Its second run, `r20261114-085805-b9d1` at `f1fd537`, was launched at 12:58 AM PST and holds all eight cards for about 17 minutes. Take your lease with `--wait` behind it; do not cancel it.
3. **ROWS2's finding you need:** `/workspace/venv312` on the node is gone since its restart; `rows.sh` fell back to torch 2.14.0+cu130, and `/workspace/jobs/venv312` has 2.13.0+cu129. Under the toolkit's 12.9.86 the 64-row gate refuses two kernels; your session is at 16 rows, so it does not touch you, but say which CUDA and which Python your launches ran under.
4. **Next from you:** "Launching" on MEM-312 when the probe has printed, then "Launches done". The node stops at Sat 6 PM PST; I tell infra the turn your last run ends.

