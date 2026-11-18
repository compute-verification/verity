# tensor_kinds: what every native MMA form costs in issue

How much of a card's issue does each kind of tensor instruction take? A matrix multiply instruction (an MMA) and the
scalar instructions beside it share the SM's issue. This directory times each native `mma.sync` form of a card alone and
beside a lane of scalar instructions, and from those timings gives the form's joint issue coefficient: the issue cycles
one MMA of that form takes from the lanes beside it. A form's issue price for each output slot is then read against IMMA
s8 m16n8k32's. The instruction counts are read from each kernel's SASS, so a program counts what ptxas emitted. It only
times: clocks are read, never set.

The ACCEL model reads it: `../../machine_model/data/forms.toml` takes its forms from `tensor_kinds.py` and names these
runs as its source.

## Run it

Each job is the command of a `research run` on a node with the card, from a pushed commit. `run.sh` builds for the
card it holds: the arch of the card record (`../../hardware/cards/`) whose `driver` holds the name nvidia-smi gives it.
It times every native form of an RTX PRO 6000, the int8 and f16/bf16 `mma.sync` forms of an H200 or H100, and the legacy
forms of a B200; a card no record holds refuses:

~~~text
research run --on NODE --project verity --source . --cwd source -- env GPU_LEASE_WHO=LANE \
  gpu-lease 1 --wait --on CARD --cores CORES --max-min 25 -- bash calibration/pouw/tensor_kinds/run.sh
~~~

## Read a run

The readers are plain Python and need no GPU:

~~~text
python3 tensor_kinds.py OUT                             # the fit of a run.sh run; exits 3 when a gate fails
python3 tensor_kinds.py search OUT                      # a search.sh run: every schedule's program, the control, the gates
python3 tensor_kinds.py refit OUT... [--search OUT...]  # the ACCEL model's joint issue rows from these runs, as JSON
~~~
