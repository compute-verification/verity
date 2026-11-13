[2026-11-13T00:06Z] 3cb90d6cb26f
BACKWARD2, Fri 13 Nov 4:06 PM PST.

**The second probe is launched: r20261113-000631-529d on vy-nebius-2 (research pool, as training, --mem-gb 16, cwd source), with your eight cases fma0 to fma7 and the other seven products at -2^125.**

- RAN on the VM (numpy twin, one head per case): under FFMA only k = 0 has a non-finite dP sum. k = 1 to 7 give exactly 2^125 on all 1,024 rows, and every dQ, dK and dV word is finite. Under FMUL then FADD all eight have a non-finite dP sum on every row.
- RAN r20261113-000037-ce60 (CPU node, smoke mode): venv with torch 2.14.0, the eight cases drawn and archived, rc 0. RAN locally on the smoke archive: each case's products in thread 0 are +2^128 at dim k and -2^125 at the other seven, on every row.
- The 'ftz' case is not in it: the first probe already holds its card words, and the fits at both heads (r20261112-234814-8ecd at a809ac0598, r20261112-234851-c9e1 at 84e7fc9161, still running) compare them under FTZ and without. If they leave the flush open, I say what case would settle it.
- I send the result lines when it ends, and the table when the fits end.

