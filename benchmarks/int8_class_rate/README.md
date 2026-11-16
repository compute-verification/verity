# int8_class_rate: the H200's sustained int8 rate per operand class

How fast does an H200 run int8 tensor work on each class of operand values? A card registers its capacity for each
operand class (`Verity.OperandClass`: s8 uniform on [-128, 127], s2 on [-2, 1], u2 on [0, 3], u4 on [0, 15], zero), and
this directory measures what one H200 sustains on int8 tensor-core work in each class, at the declared clock: alone, and
beside SHA-512 paced at a ratio of compressions a T MAC. It is the measured reading of the capacity a card registers per
class; the bound reads the data sheet until a reading is accepted.

It only times: every rate and clock comes from the kernels' own `clock64` over `%globaltimer`. Nothing here is a proof:
a run's rates are its record.

## Run it

`run.sh` measures one card under one GPU lease, with the clock locked to `CLOCK`, and `summarize.py` reads one or more
runs: the most each class sustained, alone and at each ratio.

~~~text
research run --on NODE --project verity --source . --cwd source --env CARD=NODE \
  --env CLOCK=MHZ --env GPU_LEASE_WHO=LANE -- bash benchmarks/int8_class_rate/run.sh
PYTHONPATH=. python3 benchmarks/int8_class_rate/summarize.py ~/.research/runs/RUN1/out ~/.research/runs/RUN2/out \
  [--json FILE]
~~~

`run.sh`'s header lists its knobs. Its tests hold `summarize.py`'s reading to made-up arms, on a CPU:

~~~sh
uv run tools/check/suites.py benchmarks/int8_class_rate
~~~

[`DESIGN.md`](DESIGN.md) says what each kernel runs and how the hash is paced beside it.
