# `pouw_window`: PoUW's audit window as executable Lean

The port of `verity/core/protocols/pouw/window.py`, on Lean core and PoUW's commitments (`pouw_commit`: its frame-v3-sha512
domains, Flock's SHA-512 and hm96, core's tagged identity), and the proof system's challenger for a live window's
coins (`core_sampled_proofs`). Once its properties are proven it moves, as a
pure move, to the same path under `verity/core/`; the Python keeps running until then.

Tokens served past the honest run's end-of-sequence are not useful work, so a workload of record serves at a public
fixed length, ignore_eos at each slot's public max_tokens, which the window registers and the verifier checks
(`PouwWindow.Length`).

| module | `window.py` |
| --- | --- |
| `PouwWindow.Grid` | `GRID_BITS`, `on_grid`, `floor_point`, `pad`, `level`, `point`, `levels`, `Work`, `bucket`, with their refusals |
| `PouwWindow.Draw` | `k_prime` (K′, exact: `missAtMost`, then bisection), `eps_at` (`k_prime` inverted: the least grid ε whose K′ is at most K), `sampling` |
| `PouwWindow.Public` | `PUBLIC`, `REGISTERED`, `LEGACY_REGISTERED`, `KINDS` |
| `PouwWindow.Length` | `STOP_FIXED`, `LENGTH_KEYS`, `length_item`, `length_problem`: the workloads of record's fixed length |
| `PouwWindow.Json` | integers read as Python's `json` reads them; the items written as Python writes them |
| `PouwWindow.Entry` | `T_BITS`, `DEPTHS`, `PORT`, `ENTRY_SCHEMA`, `TILE`, `ROW_STEP`, `Entry` (`row`, `words`, `from_row`, `from_words`), `pass_tiles`, `tiles`, `layout_problem`, `tile_in`, `tile_at`, `tile_read`, `entry_digest`, `entry_bc`, `entry_leaf` |
| `PouwWindow.Record` | `record_digest` (`canonical_json` as Python writes it), `record_problem`, and how Python reads a record's JSON |
| `PouwWindow.Side` | `SIDE_DOMAIN`, `OWNER`, `SIDE_INDEX_WORDS`, `side_row`, `side_index`, `side_domain`, `pad_node`, `side_levels`, `side_path` |
| `PouwWindow.Statement` | `window_problem`, the window statement L4 on the entry rows, with the records and side indices |
| `PouwWindow.Inputs` | `PUBLIC_INPUTS_FORMAT`, `PUBLIC_PORTS`, `M_MAX`, `session_cap`, `sessions`, `instance_ts`, `public_inputs` |
| `PouwWindow.Rounds` | `COINS`, `MODES` (`CoinMode`), `mode`, `LIVE_SCHEME`, `SALT`, `DRAW`, `ROUND`, `live_families`, `live_epoch_reason`, `rounds_reason`, `round_key`: a live window's round coins over the proof system's challenger (`core_sampled_proofs`'s `SampledProofs.Challenger`) |
| `PouwWindow.Lanes` | not `window.py`'s: a drawn H1 leaf's lane and combine statements under option A (`circuit/h1zy.py`'s `PouwH1LaneZy_v1`, `PouwH1Combine_v1`), port by port: `Tile`, `nDigits`, `lanePorts`, `combinePorts`, `fieldOf?`, each input's and output's port, bits and 16-bit words, and the row the verifier opens it at (`h1zy.statement_ports`, `pouw.leaf_lanes`' reads), and `wires?`, `wireOf?`, `Port.fieldAt?`, the port, row bit and field each input wire of the lowered unit holds (`class_statement.unit_lowering`), which proofs' decoders (`decL`, `decC`, `leafOf`) read |
| `PouwWindow.KnownRows` | not `window.py`'s: the rows the verifier registers at their known content, lane `l`'s `pub` (salt, position, roots and `l`) and `admit` and every statement's `o0` (`knownRows`, `rowWords`, `halvesNat`, `positionNat`, `rootsNat`), the known content `LeafLanesSound` hands each session (`Accounting.TS.Lanes.knownOf`) |
| `PouwWindow.Difftest` | the difftest's table: one `Difftest.Fn` per ported function of `window.py`, and `PouwWindow.Lanes`' and `PouwWindow.KnownRows`' |

`pouw-window-difftest` (`PouwWindow/DifftestMain.lean`) serves that table on the shared harness
(`verity/core/difftest/`), and the suite `verity-pouw-window-difftest`, which `check` runs after its Lean steps, holds
each function to Python on Hypothesis-drawn inputs, refusals included: `tests/test_difftest.py` to `window.py`, and
`tests/test_lanes.py` `PouwWindow.Lanes` to `h1zy.statement_ports` and `h1tile.Layout`, also at the served 32 × 32 leaf at
K = 448, 4096 and 14336 and at the sidecar's 8 × 16 at K = 64, every lane and the combine, and its input wires to the
circuits Flock lowers the statements to (the combine's at the sidecar's leaf, at 8 × 16 and at the served leaf at K = 448,
with its circuit file's META; the sidecar's lane, which Flock refuses; and, with `VERITY_POUW_WINDOW_LOWERED_LANES=1`,
two lanes' circuit files), and `tests/test_known_rows.py` `PouwWindow.KnownRows` (of record) to the served path's stand-in
known rows, `benchmarks/pouw/h1_lanes/run.py`'s `known_rows`.

A problem a function returns (`layout_problem`, `record_problem`, `window_problem`) is a value, and its text is
Python's; a Python refusal (`ValueError`) is an `Except` error. The suite draws each function's inputs on and off its
domain and reaches every reason each of those three returns.

Its properties are stated in `Security/Properties/Pouw.lean` (the package `security_properties`), and the proved ones
are in `Security/Proofs/Properties/Pouw.lean`.

~~~text
lake build                                                         the library and pouw-window-difftest
uv run tools/verity/check/suites.py verity-pouw-window-difftest    the difftest
python tools/verity/lean/check.py verity/core/protocols/pouw/lean/window   the Lean check
~~~
