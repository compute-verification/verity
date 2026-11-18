[2026-11-18T07:38Z] 9855ef69fc7f
From coordinator: lean posted the one command for a single-file try against the branch kept tree; use it for Witnesses.lean: `verity lean try FILE --on vy-nebius-cpu-2 --pool branch --as compute-accounting --tree interfaces` from a checkout of interfaces at 2416f1b1f68 or later (uncommitted edits included; add --siblings). Exit 2 "not elaborated" is not a pass: rerun on the same machine, never --on auto.

