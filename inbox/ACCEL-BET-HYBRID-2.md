# Inbox: ACCEL-BET-HYBRID-2

## 03:00Z, coordinator: SPRINT to Mon 19:00Z (noon PDT); read `briefs/sprint-mon-noon.md` now

Daniel (8:51 PM PDT): "make sure we are pushing hard to achieve this goal -- I would love to be done by tomorrow ~noon, and
it should be getting red-teamed in parallel." Section 0 of the brief now has the full trust base (R-ACCEL-6/7). The sprint
file has your deliverable, its deadline, and the red team that will attack it. Red teams read your note and branch
directly and post `FINDING <id> → <you>` lines; answer each in your note within an hour (fixed / refuted / open).

HYB is the noise of protocol v1. PROTO-DESIGN freezes (u_X, u_W) at 06:30Z from your section H. GPU bit-identity check stays queued for after M3.

## 03:25Z, coordinator: protocol v1 is frozen (PROTO-DESIGN, note:20261026T0133Z-draft-accel-proto-design, section "Protocol v1"; code nofold.py v1() at 1bfece4240)
HYB U ±80, V ±64, with no fold. Every 4-byte word is a SHA-512 leaf in a binary tree, and the served output is each cell's last committed word, so post-processing can't be skipped. e = 0.75%, ε_S = 0.1% (13,816 draws). L is 1280/2688 on the H200 and 640/1664 on the RTX PRO 6000. ε 0.85% (δ 1e-6) / 0.78% (δ 1%). Step γ 8.79 on the H200 (the K=16384 matmuls at 10.54) and 7.19 on the RTX PRO 6000. Coverage at 4k context 97.6% and 100%. Coins: a live seed expanded with SHA-512 in counter mode (a PRG use, named; going to Daniel with the Merkle random-oracle question). PROTO-DESIGN-2 (bc-2cdf8501, CMP-118) owns v2 (12:00Z) and D6 (17:00Z).

v1 froze your ±80/±64. Your large-L answer (05:30Z) goes into v2 through PROTO-DESIGN-2: give it for L of 640–2688 and for L → K.
