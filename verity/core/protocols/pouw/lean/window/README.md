# `pouw_window`: PoUW's audit window as executable Lean

The port of `verity/core/protocols/pouw/window.py`, on Lean core and PoUW's commitments (`pouw_commit`: its frame-v3-sha512
domains, C-Flock's SHA-512 and hm96, core's tagged identity), and the proof service's challenger for a live window's
coins (`core_sampled_proofs`). Once its properties are proven it moves, as a
pure move, to the same path under `verity/core/`; the Python keeps running until then.

| module | `window.py` |
| --- | --- |
| `PouwWindow.Grid` | `GRID_BITS`, `on_grid`, `floor_point`, `pad`, `level`, `point`, `levels`, `Work` (`Work.at`), `credit`, with their refusals |
| `PouwWindow.Draw` | `k_prime` (K′, exact: `missAtMost`, then bisection), `sampling` |
| `PouwWindow.Public` | `PUBLIC`, `REGISTERED`, `KINDS` |
| `PouwWindow.Json` | integers read as Python's `json` reads them; the items written as Python writes them |
| `PouwWindow.Entry` | `T_BITS`, `DEPTHS`, `PORT`, `ENTRY_SCHEMA`, `TILE`, `ROW_STEP`, `Entry` (`row`, `words`, `from_row`, `from_words`), `pass_tiles`, `layout_problem`, `tile_read`, `entry_digest`, `entry_bc`, `entry_leaf` |
| `PouwWindow.Shape` | the window's public shape: `Call`, `shape_of`, `shape_json`, `shape_tiles` (its N), `shape_problem`, `shape_entries`, `public_fields`, `tile_at` (a unit's call, A row and B row, fixed by the shape alone); the window's fixed length: `Length`, `length_of`, `fixed_length`, `length_problem` (each call's max_tokens, its rows ignoring end-of-sequence) |
| `PouwWindow.Record` | `record_digest` (`canonical_json` as Python writes it), `record_problem`, and how Python reads a record's JSON |
| `PouwWindow.Side` | `SIDE_DOMAIN`, `OWNER`, `SIDE_INDEX_WORDS`, `side_row` (a slot's row with its index, `ok` and `admit`), `side_field`, `side_domain`, `pad_node`, `side_levels`, `side_path` |
| `PouwWindow.Statement` | `window_problem`, the window statement L4 on the entry rows: the entries the shape's, every slot's side field (an unadmitted slot the canonical empty row), and the credit the masked sum of admitted, accepted rows rounded down to the grid |
| `PouwWindow.Inputs` | `PUBLIC_INPUTS_FORMAT`, `PUBLIC_PORTS`, `M_MAX`, `session_cap`, `sessions`, `instance_ts`, `public_inputs` |
| `PouwWindow.Rounds` | `COINS`, `MODES` (`CoinMode`), `mode`, `LIVE_SCHEME`, `SALT`, `DRAW`, `ROUND`, `live_families`, `live_epoch_reason`, `rounds_reason`, `round_key`: a live window's round coins over the proof service's challenger (`core_sampled_proofs`'s `SampledProofs.Challenger`) |
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
