[2026-11-10T16:17Z] 578a01fd126b
TIMEPAIR, Tue 10 Nov 8:20 AM PST. **Window holds.** The rehearsal across the two hosts worked whole [run: serve r20261110-160259-c494 on cpu-10, observe r20261110-160306-012b on cpu-8, no pool, START 8:05:00, 300 buckets a case]:
- All 9 cases left their record on both sides, none with a problem; the guard passes (planted 1.000, ignored 0.000); `pair_score.py` scored them from the two fetched runs.
- Route and address: the observer on cpu-8 reached cpu-10 on 47811 in every case. Both clocks follow PHC0, RMS offset under 2 µs [run: `chronyc tracking` in each host.txt].
- Its rows are a rehearsal's (150 scored a row), no result: write as is 0.037 [0.003, 0.104].
- One fault, fixed: `window.sh` ended rc 2 on its last `ls` though every case ran. It now exits 0 only when every case left its record; a 8-minute second rehearsal checks that now (START 8:22).
- How cpu-10 is A: its address goes to both sides as a sent file; no address is in `window.sh`, host.txt, Linear or the notes.
- Next: launch both about 9:45 with START 10:00:00 AM PST, then 10:40 with START 10:55:00 (`--timeout 50m`, ends 11:26).

