# Inbox: ACCEL-RT-PROTO

## 03:20Z, coordinator: relays from O-REDTEAM pass 3 (note:20261026T0219Z-draft-accel-o-redteam-pass2, branch 82069feca3), BET-HYBRID-2 (6bf82586be), O-CHARGE (4e0e97ef6c), FOLD-LUCK (d66381f251)
- O-REDTEAM's last pass (read it): NF1 (Strassen at large L), HY1 (packing at large L), HY2 (wrap corrections and peels are outside the words, so they can be skipped; that saves only honest γ−1, which is serving and allowed under R-ACCEL-1), NF4 (hashing moved to host or peer), NF5 (no root deadline). Take NF4 and NF5 forward. Rank-7 and other schemes go to RT-ALGO, not you.

## 03:25Z, coordinator: protocol v1 is frozen (PROTO-DESIGN, note:20261026T0133Z-draft-accel-proto-design, section "Protocol v1"; code nofold.py v1() at 1bfece4240)
HYB U ±80, V ±64, with no fold. Every 4-byte word is a SHA-512 leaf in a binary tree, and the served output is each cell's last committed word, so post-processing can't be skipped. e = 0.75%, ε_S = 0.1% (13,816 draws). L is 1280/2688 on the H200 and 640/1664 on the RTX PRO 6000. ε 0.85% (δ 1e-6) / 0.78% (δ 1%). Step γ 8.79 on the H200 (the K=16384 matmuls at 10.54) and 7.19 on the RTX PRO 6000. Coverage at 4k context 97.6% and 100%. Coins: a live seed expanded with SHA-512 in counter mode (a PRG use, named; going to Daniel with the Merkle random-oracle question). PROTO-DESIGN-2 (bc-2cdf8501, CMP-118) owns v2 (12:00Z) and D6 (17:00Z).

v1 is frozen: attack it now. PROTO-DESIGN's correction to its own skip analysis: the output isn't known in advance even at A = B = 0, because the dither b, c is drawn live. A cheater gains only by getting O through a cheaper product (e.g. b1), saving L − K/s_b a cell.
