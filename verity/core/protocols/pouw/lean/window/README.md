# `pouw_window`: PoUW's audit window as executable Lean

The port of `verity/core/protocols/pouw/window.py`, on Lean core and PoUW's commitments (`pouw_commit`: its frame-v3-sha512
domains, C-Flock's SHA-512 and hm96, core's tagged identity). Once its properties are proven it moves, as a
pure move, to the same path under `verity/core/`; the Python keeps running until then.

| module | `window.py` |
| --- | --- |
| `PouwWindow.Grid` | `GRID_BITS`, `on_grid`, `floor_point`, `pad`, `level`, `point`, `levels`, `Work`, `bucket`, with their refusals |
| `PouwWindow.Draw` | `k_prime` (K′, exact: `missAtMost`, then bisection), `sampling` |
| `PouwWindow.Public` | `PUBLIC`, `REGISTERED`, `KINDS` |
| `PouwWindow.Json` | integers read as Python's `json` reads them; the items written as Python writes them |
| `PouwWindow.Entry` | `T_BITS`, `DEPTHS`, `PORT`, `ENTRY_SCHEMA`, `TILE`, `ROW_STEP`, `Entry` (`row`, `words`, `from_row`, `from_words`), `pass_tiles`, `tiles`, `layout_problem`, `tile_in`, `tile_at`, `tile_read`, `entry_digest`, `entry_bc`, `entry_leaf` |
| `PouwWindow.Record` | `record_digest` (`canonical_json` as Python writes it), `record_problem`, and how Python reads a record's JSON |
| `PouwWindow.Side` | `SIDE_DOMAIN`, `OWNER`, `SIDE_INDEX_WORDS`, `side_row`, `side_index`, `side_domain`, `pad_node`, `side_levels`, `side_path` |
| `PouwWindow.Statement` | `window_problem`, the window statement L4 on the entry rows, with the records and side indices |
| `PouwWindow.Inputs` | `PUBLIC_INPUTS_FORMAT`, `PUBLIC_PORTS`, `M_MAX`, `session_cap`, `sessions`, `instance_ts`, `public_inputs` |
| `PouwWindow.Difftest` | the difftest's table: one `Difftest.Fn` per ported function of `window.py` |

`pouw-window-difftest` (`PouwWindow/DifftestMain.lean`) serves that table on the shared harness
(`verity/core/difftest/`), and `tests/test_difftest.py`, the suite `verity-pouw-window-difftest`, which `check`
runs after its Lean steps, holds each function to `window.py` on Hypothesis-drawn inputs, refusals included.

A problem a function returns (`layout_problem`, `record_problem`, `window_problem`) is a value, and its text is
Python's; a Python refusal (`ValueError`) is an `Except` error. The suite draws each function's inputs on and off its
domain and reaches every reason each of those three returns.

Its properties are stated in `Security/Properties/Pouw.lean` (the package `security_properties`), and the proved ones
are in `Security/Properties/Proofs/Pouw.lean`.

~~~text
lake build                                                         the library and pouw-window-difftest
uv run tools/verity/check/suites.py verity-pouw-window-difftest    the difftest
python tools/verity/lean/check.py verity/core/protocols/pouw/lean/window   the Lean check
~~~
