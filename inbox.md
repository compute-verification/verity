[2026-11-12T23:30Z] 0d3bffb9bf3c
BACKWARD2, Thu 12 Nov 3:30 PM PST.

**My one launch of the probe was refused for a wrong flag of mine, before it touched the node; nothing runs, and I'm stopping until your word on a second launch with `--project verity`.**

- RAN, 23:30Z: `research run --on vy-nebius-2 --project training ... --mem-gb 16 -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/probe_job.sh"'`, which refused with: "research run --on vy-nebius-2: refused: machine 'vy-nebius-2' belongs to project(s) ['verity'], not 'training'; refusing to touch it".
- Your 3:01 PM and 3:19 PM messages reached me only at 3:30 PM: my VM slept through the wait, so I launched at 3:30 PM and not earlier.
- The fix is `--project verity` and nothing else: the same node, one card, a lease of at most 5 minutes, 16 GB, and the same job and script as the CPU check (r20261112-213411-af3a). On your word I launch it once and send you the run id at once.
- Meanwhile, READ: the every-word run at `1178eb5439` (r20261112-212456-3c41, still running) has every dQ, dK and dV word of all 32 deterministic heads equal to the card's. The old-head runs ended with rc 0. r20261112-194241-5573 has every word of all 32 deterministic heads equal (2,097,152 of each gradient), and r20261112-192536-b84b ran "59 suites: 59 passed (1 cached), 0 failed".

