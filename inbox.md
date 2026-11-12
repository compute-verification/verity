[2026-11-12T19:45Z] 84321dbeb3b3
BACKWARD to training, Thu 12 Nov 11:45 AM PST (19:45Z). **The every-word run is launched again, before it started: as I had written its job it would have passed its memory limit. Both remaining runs are now far back in their lines (13 and 18 checks). The handoff note is synced in the next minutes so nothing waits on me.**

- **What was wrong.** The job started one process a head by `os.cpu_count()`, which ignores the slot's cores: 64 heads at once. RAN on the VM (a 256-row head, 55 s): a head takes about 1.6 GB and 11 minutes, so about 105 GB against the 64 GB I had asked. I cancelled `r20261112-182341-2949` while it was still first in line and launched `r20261112-194241-5573` (16 processes, 40 GB, on vy-nebius-cpu-7, `--on auto`). My slip; it cost the place in line.
- **One more check from that measurement:** the first 256 rows' dQ of an H 24 head through the head Definition, 16,384 of 16,384 words.
- **The lines.** A check took about 11 minutes of line today. So the suites (`r20261112-192536-b84b`, 13 back) may land around 2 PM PST and the every-word run (18 back, then about 50 minutes) around 4 PM PST. These are guesses from one hour's line.
- **What I do unless you say otherwise:** sync the handoff note now; stay for the suites' result and put it in #9417's body; leave the every-word run's result to the next worker, with the exact steps in the note. If you would rather I end now, say so.

