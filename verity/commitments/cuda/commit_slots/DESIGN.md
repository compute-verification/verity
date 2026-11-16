# commit_slots: design

What measures the call on a GPU and what its tests cover. `README.md` is the call's specification.

## Measuring

What measures it is in `benchmarks/torch_gpu/commit_slots/`: `check.py`, the GPU's roots against the reference's;
`bench.py`, the call's throughput at the H200 example's layout, alone and beside the layer's int8 GEMMs; `step.py`, one
streamed step at a layout file, timed by phase (`time`), and a layout's compressions a row-slot on the host twin
(`count`); and `gpu.sh`, these with the tests under one GPU lease (`OUT=DIR bash benchmarks/torch_gpu/commit_slots/gpu.sh`
from the tree's root; its header has the options).

## Tests

`tests/verity/commitments/cuda/commit_slots/` is the directory's suite: the reference against the fixture, Flock's salts and an independent tree build;
openings and their refusals; layouts written and loaded back; the host twin against the reference; the streamed slot
trees; a step's memory and a row-slot's compressions, counted; planning at scale; and, with a GPU, the device against the
host twin. Two say why they skip where they can't run: the FAMILIES layout's round trip where its fetched fixture is
absent (`research data fetch-fixtures tests/verity/commitments/fixtures/layouts`; the suites' runner and `gpu.sh` fetch
it), and the plan on some SMs where this Python's torch has no driver bindings (`slots_gpu.driver_bindings`).
