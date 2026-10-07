# `pouw_window`: PoUW's audit window as executable Lean

The port of `verity/experimental/protocols/pouw/window.py`, Lean core only, until its properties are proven and it moves to
`verity/core/protocols/pouw/window/`. The Python keeps running until then.

| module | `window.py` |
| --- | --- |
| `PouwWindow.Grid` | `GRID_BITS`, `on_grid`, `floor_point`, `pad`, `level`, `point`, `levels`, `Work`, `bucket`, with their refusals |
| `PouwWindow.Draw` | `k_prime` (K′, exact: `missAtMost`, then bisection), `sampling` |
| `PouwWindow.Advice` | `KINDS`, `tiles`, `tile_in`, `tile_at`, `window_problem` (L4) and its reasons (`Problem.text`) |
| `PouwWindow.Public` | `PUBLIC`, `REGISTERED` |
| `PouwWindow.Json` | the entries read in `window.py`'s dict shape; the items written as Python writes them |
| `PouwWindow.Difftest` | the difftest's evaluator and runner |

The properties are stated in `Security/Properties/Pouw.lean`.

~~~text
lake build                                                        the library and pouw-window-difftest
python3 scripts/difftest_vectors.py REPO --out OUT                 OUT/difftest/{instances,python}.jsonl
lake exe pouw-window-difftest OUT/difftest/instances.jsonl OUT/difftest/python.jsonl 12000
python tools/verity/lean/check.py verity/experimental/core/protocols/pouw/window   the Lean check, the difftest among its runs
~~~

`verity/experimental/protocols/pouw/tests/test_pouw_window_lean_difftest.py` holds the generated data to the SHA-256 the
record pins.
