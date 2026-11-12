[2026-11-12T04:46Z] 149476b27f70
FORM to training, TRN-103, Wed 11 Nov 8:46 PM PST (04:46Z). MAIN MOVED AFTER THE MERGE; #8686's HEAD DID NOT. I go on at 9cf54a37d7 as ordered; this is for you to take to network-accounting if you want #8686 to hold the new main.

- Read by ls-remote at 04:43Z [RAN]: `main` = 62bf29346f76 (it was be9a3d566905); #8686's head still 5773b1f3f4ba; #8676's branch still c1ac6b770a.
- main gained eight queue merges (#9042, #8773, #8914, #9040, #8934, #8949, #9019, #8977). Of the 60 files they touch, two are in this diff: `security/lean-audit.json` and `security/Properties/lean-audit.json` [RAN: git diff --name-only].
- `git merge-tree --write-tree` of 9cf54a37d7 with 62bf29346f76 gives a tree and names no conflict [RAN]. That is text only: no build and no records ran on that merge.
- What our proofs read of it [READ]: `Circuits/Program.lean` gains `readJson`, `decodeOf` and `decode_eq_of`, 57 added lines and none changed, so `decode` is as it was. `Flock/TemplateQuery.lean`, `Flock/Sigma.lean`, `Spot/Wires*.lean` and the check's package are untouched.
- So the body will say "holds `main` at be9a3d566905 through #8686's head", which is behind main by those eight. I do not merge main alone: your order. If #8686's head moves before my push I merge it and run the records again, as ordered.

SUITES RUNNING at 9cf54a37d7, both launched 04:44Z, both waiting for a slot ("about 17 min"): r20261112-044349-43ec (the eleven large, vy-nebius-cpu-9) and r20261112-044433-e691 (verity-vllm, vy-nebius-cpu-11). The other 57 go when one of them ends.

