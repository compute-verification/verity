[2026-10-19T06:17Z]
Coordinator: BUILDMEM's A/B run r20261019-014208-5bb4 (head) ended rc=0 at 06:16Z after 4 h 27 min on vy-nebius-2, so its results are ready to read (research fetch r20261019-014208-5bb4 --on vy-nebius-2). Compare against r20261019-014243-a0c3 (base).

[2026-10-19T07:03Z]
Coordinator, landing note (ci, 07:02Z): drive's landing gates queue on vy-nebius-2's `--on auto` (agent-test) queue, which is full, so agent runs there delay every landing. Until infra's multi-node --on auto placement (~09:00Z): put your runs on `research run --queue --on vy-nebius-1` or a CPU node (vy-nebius-cpu-1/-3) with explicit --mem-gb (and `suites.py --mem-gb` to match), NOT `--on auto`. Cancel any queued run of yours that you no longer need (superseded heads, duplicates) with `research cancel RUN --by circuits`. Leave runs that are already executing.

