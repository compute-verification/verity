# `pouw_window`: PoUW's audit window as executable Lean

The port of `verity/experimental/protocols/pouw/window.py`, Lean core only. Once its properties are proven it moves, as a
pure move, to the same path under `verity/core/`; the Python keeps running until then.

| module | `window.py` |
| --- | --- |
| `PouwWindow.Grid` | `GRID_BITS`, `on_grid`, `floor_point`, `pad`, `level`, `point`, `levels`, `Work`, `bucket`, with their refusals |
| `PouwWindow.Draw` | `k_prime` (K′, exact: `missAtMost`, then bisection), `sampling` |
| `PouwWindow.Advice` | `KINDS`, `tiles`, `tile_in`, `tile_at`, `window_problem` (L4) and its reasons (`Problem.text`) |
| `PouwWindow.Public` | `PUBLIC`, `REGISTERED` |
| `PouwWindow.Json` | the entries read in `window.py`'s dict shape; the items written as Python writes them |
| `PouwWindow.Difftest` | the difftest's table: one `Difftest.Fn` per function of `window.py` |

`pouw-window-difftest` (`PouwWindow/DifftestMain.lean`) serves that table on the shared harness
(`verity/experimental/core/difftest/`), and `tests/test_difftest.py`, the suite `verity-pouw-window-difftest`, which `check`
runs after its Lean steps, holds each function to `window.py` on Hypothesis-drawn inputs, refusals included.

Its properties are stated in `Security/Properties/Pouw.lean` (the package `security_properties`), and the proved ones
are in `Security/Properties/Proofs/Pouw.lean`.

~~~text
lake build                                                         the library and pouw-window-difftest
uv run tools/verity/check/suites.py verity-pouw-window-difftest    the difftest
python tools/verity/lean/check.py verity/experimental/core/protocols/pouw/window   the Lean check
~~~
