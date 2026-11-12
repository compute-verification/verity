[2026-11-12T21:11Z] 00b517b89c7c
Coordinator to ONSITE, Thu 12 Nov 21:12Z. lean ruled on LEAN-80 (12:46 PM PST): the renames do not wait for our move of the configurations, and our move does not wait for them.
1. The move has lean's say from the hour #9375 is on `main`. Land it when we choose, except between Sunday 15 Nov's 8 AM PST cut and the renames' landing that evening. On `main` by Sat 14 Nov 10 PM PST, the renames are staged on it; after, the move carries five files as renamed (stage 1 one under `remote/`, stage 6 three under `remote/` and one under `onsite/`).
2. So the target is: the move's pull request open with its pins commit within about 2 h 30 min of #9375 landing, as you sized, and on `main` by Sat 14 Nov 10 PM PST. Say in your next entry whether that holds.
3. kernels opens a move of 45 leftover names on Sat noon PST (HASH-125; 37 declarations in our files, under `security/Proofs/Flock/` and `verity/protocols/spot/verifier/lean/Spot/`). Say whether any path of your three maps is one of its files; if so the two moves need an order and I set it.
My 21:03Z decisions stand.

