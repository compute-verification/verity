[2026-11-17T20:48Z] c9cdb5c69efd
KYARM, from the kernels lead, Tue 17 Nov 20:48Z. SETTLED, by benchmarks' lead at 12:46 PM PST on TOP-435, agreeing with top; this is the last word on the metric's shape:
- Two NEW ids WITHOUT a card: `benchmarks/pearl-plus-gamma-c` and `benchmarks/pearl-plus-gamma-l`. Declare those two rows in `benchmarks/overhead/metrics.toml` beside the two H200 ones (owner benchmarks; same unit, better, floor and command pattern), the axis saying each point's member names its model and card and whether it times the output's commitment.
- `run.py`'s map sends `pearl-plus/h100/C` and `/L` to them. EVERY reading of them carries a member, the large model's included: `llama-3.2-1b-h100` and `llama-3.1-70b-h100`. Nothing on them is a line's own value without a card.
- The H200's two ids (`...-c-h200`, `...-l-h200`) and their rows stay exactly as today, byte for byte. Do not add an alias key: the site refuses a row's unknown keys.
- W = 0.48, noted in the point.
No ids named `-h100`. The machine `vy-h100-1` STAYS UP (infra agreed; benchmarks' harness point runs on it next; deadline 03:00Z): so once the PR's suites pass you may run the not-of-record trial from its head yourself when `gpu-lease` shows the card free, with a row on `calendars/h100-1.md` after benchmarks' row. Still unmarked; return the number to me.

