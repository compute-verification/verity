vllm coordinator, 22:33Z Sun 1 Nov:
- Plan read, approved. I told compute-accounting (owner of benchmarks/pouw/served_h1/) and kernels (dither/j10r_units.py) about moving the producer, and passed receipt_value's 0.7 s/proof-block cost to compute-accounting. If either objects, I'll tell you here.
- vy-dc-h200 runs rehearsal 2 (r20261101-192017-2226) until about 23:30Z. Don't launch anything there before it ends. After that it's free until it stops itself at Mon 18:00Z. Put your served proof run there only if it ends well before 17:30Z Mon; otherwise use h200-2, -3 or -4.
- Your cpu-1 runs fd3a, 7165 (rc 143) and 9217 failed. If cpu-1 is the problem, use `--on auto`.

vllm coordinator, 12:31Z Mon 2 Nov: infra started vy-dc-h200-2 as the fallback (up, H200 idle, one gpu-lease slot, deadline 18:00Z). Use it only for a rerun after 30bd ends, since a lane has one open gate period.
