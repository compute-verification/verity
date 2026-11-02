# Direction `lowrank` (islands `e1mb` and `e100kb`)

The most loss per training FLOP from an update whose encoding E fits 1 MB, then 100 kB, and how bits per byte at fixed
FLOPs moves with |E|: LoRA rank and placement (attention or MLP, which layers), norm gains, initialization, the factors'
learning rates and optimizer, and their precision.

`islands/e1mb/train.py` and `islands/e100kb/train.py` are the same file (the class is its directory's name); its docstring
lists every knob. `--e-bytes N` fits the rank to N bytes instead of the cap, for the curve's points.

- `drive.py`: the queue (`~/.research/lowrank/queue.txt`, one `CLASS START ; NOTE ; TRAIN_ARGS` per line), at most two
  candidates per island in flight, launched with `launch.py --no-wait` from this checkout when it is clean and collected
  with `launch.py --collect` when they end; its state file lets a restarted driver resume.
- `report.py`: every ended candidate by start, class and bpb.

Results and negatives go on the board (cursor/te-board-4207, `benchmarks/covert_witness/BOARD.md`) with their run and art
ids; the records are in the evidence store.
