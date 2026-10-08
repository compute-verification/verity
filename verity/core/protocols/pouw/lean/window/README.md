# `pouw_window`: PoUW's audit window as executable Lean

The port of `verity/core/protocols/pouw/window.py`, Lean core only. Once its properties are proven it moves, as a
pure move, to the same path under `verity/core/`; the Python keeps running until then.

| module | `window.py` |
| --- | --- |
| `PouwWindow.Grid` | `GRID_BITS`, `on_grid`, `floor_point`, `pad`, `level`, `point`, `levels`, `Work`, `bucket`, with their refusals |
| `PouwWindow.Draw` | `k_prime` (K′, exact: `missAtMost`, then bisection), `sampling` |
| `PouwWindow.Public` | `PUBLIC`, `REGISTERED`, `KINDS` |
| `PouwWindow.Json` | integers read as Python's `json` reads them; the items written as Python writes them |
| `PouwWindow.Difftest` | the difftest's table: one `Difftest.Fn` per ported function of `window.py` |

`pouw-window-difftest` (`PouwWindow/DifftestMain.lean`) serves that table on the shared harness
(`verity/core/difftest/`), and `tests/test_difftest.py`, the suite `verity-pouw-window-difftest`, which `check`
runs after its Lean steps, holds each function to `window.py` on Hypothesis-drawn inputs, refusals included.

Not ported yet, and the follow-up: the window statement (L4) on #1426's fixed-width entry rows, that is `Entry` and
`Entry.from_row`, `layout_problem`, `record_problem`, `tile_read`, `tiles`, `tile_in`, `tile_at`, `window_problem`, the
side trees (`side_row`, `side_index`, `side_domain`, `side_levels`, `side_path`) and `public_inputs`. An earlier port of
the JSON-entry advice those replaced is gone with it.

Its properties are stated in `Security/Properties/Pouw.lean` (the package `security_properties`), and the proved ones
are in `Security/Properties/Proofs/Pouw.lean`.

~~~text
lake build                                                         the library and pouw-window-difftest
uv run tools/verity/check/suites.py verity-pouw-window-difftest    the difftest
python tools/verity/lean/check.py verity/core/protocols/pouw/lean/window   the Lean check
~~~
