# `pouw_window`: the difftest

What the difftest holds, beside [the README](README.md).

`PouwWindow.Difftest` is the difftest's table: one `Difftest.Fn` per ported function of `window.py`, and `PouwWindow.Lanes`' and
`PouwWindow.KnownRows`'.

`pouw-window-difftest` (`PouwWindow/DifftestMain.lean`) serves that table on the shared harness
(`tools/difftest/`), and the suite `verity-pouw-window-difftest`, which `check` runs after its Lean steps, holds
each function to Python on Hypothesis-generated inputs, refusals included: `tests/verity/protocols/resource_accounting/compute/pouw/verifier/lean/window/test_difftest.py` to `window.py`, and
`tests/verity/protocols/resource_accounting/compute/pouw/verifier/lean/window/test_lanes.py` `PouwWindow.Lanes` to `h1zy.statement_ports` and `h1tile.Layout`, also at the served 32 × 32 leaf at
K = 448, 4096 and 14336 and at the sidecar's 8 × 16 at K = 64, every lane and the combine, and its input wires to the
circuits Flock lowers the statements to (the combine's at the sidecar's leaf, at 8 × 16 and at the served leaf at K = 448,
with its circuit file's META; the sidecar's lane, which Flock refuses; and, with `VERITY_POUW_WINDOW_LOWERED_LANES=1`,
two lanes' circuit files), and `tests/verity/protocols/resource_accounting/compute/pouw/verifier/lean/window/test_known_rows.py` `PouwWindow.KnownRows` (of record) to the served path's stand-in
known rows, `benchmarks/pouw/h1_lanes/leaf.py`'s `known_rows`.

A problem a function returns (`layout_problem`, `record_problem`, `window_problem`) is a value, and its text is
Python's; a Python refusal (`ValueError`) is an `Except` error. The suite samples each function's inputs on and off its
domain and reaches every reason each of those three returns.

The commands, from this directory:

~~~text
lake build                                                         the library and pouw-window-difftest
uv run tools/check/suites.py verity-pouw-window-difftest    the difftest
python tools/verity/lean/check.py verity/protocols/resource_accounting/compute/pouw/verifier/lean/window   the Lean check
~~~
