# `pouw_check`: PoUW's window check as a core protocol

One `Core.Protocol` (`verity/experimental/core/protocols/interface`), `PouwCheck.protocol`, id `verity/pouw/v1`: the port of
`verity/experimental/protocols/pouw/core_check.py`, Lean core only, until its properties are proven and it moves to
`verity/core/`. The Python keeps running until then.

| field | here |
| --- | --- |
| `Params` | the verifier's own: the check item (`entries`, `scheme`, `template`, `tile_work`), `eps` and `delta` as ratio strings, `grid_bits`, the declared `range` of tile counts, the window's `context`, the check query's SHA-512 |
| `Commitment` | the window's registration, one-stage's record (`registration.record`) as canonical JSON, its window `{kind, context, public}` holding `window.REGISTERED` |
| `digest` | the record's own: SHA-512(format ‖ 0 ‖ the canonical record), `registration.digest` |
| `Evidence` | the proof service's answer for a registration, read by `Core.Answer.ofJson` (protocol one-stage's): on acceptance the profile, read by `Core.Integrity.ofJson`, and the public outputs `{"draw": d}`, `d` the verifier's draw as one-stage's `draw.py` writes it (`subset`, `bernoulli`, `stratified` or `work`, with exactly `check_draw`'s keys) |
| `check` | `PouwCheck.check`: PoUW's rule over the registration (`registered`), then the service's answer to it (`outcome`); every refusal is the first rule it breaks, with `core_check.py`'s reason |
| `Public` | `window.PUBLIC`: `verdict`, `work`, `check`, `sampling`, `drawn`, `root_W` |
| `Profile` | the service's `Core.Integrity`, passed through once it is the registered partition's N units drawn subset:K′ at the window's δ, and the draw is subset:K′ over N |

| module | `core_check.py` |
| --- | --- |
| `PouwCheck.Json` | `sorted(dict)`, `_nat`, `_canon` |
| `PouwCheck.Inputs` | `params`, `commitment`, `digest`, `unit_draw`, `evidence`, with their refusals (`answer_of` and `integrity` are the interface's decoders) |
| `PouwCheck.Check` | `expected_items` and `check` |
| `PouwCheck.Protocol` | the `Core.Protocol` value |
| `PouwCheck.Difftest` | the difftest's runner |

The properties are stated in `Security/Properties/Pouw/Check.lean` and proved in
`Security/Properties/Proofs/Pouw/Check.lean`.

~~~text
lake build                                                        the library, pouw-check and pouw-check-difftest
lake exe pouw-check id | check PARAMS COMMITMENT EVIDENCE | serve   the interface's command line
python3 scripts/difftest_vectors.py REPO --out OUT                 OUT/difftest/{cases,python}.jsonl
lake exe pouw-check-difftest OUT/difftest/cases.jsonl OUT/difftest/python.jsonl 4000
python tools/verity/lean/check.py verity/experimental/core/protocols/pouw/check   the Lean check, the difftest among its runs
~~~

`verity/experimental/protocols/pouw/tests/test_pouw_core_check.py` holds the generated data to the SHA-256 the record pins,
and checks that every mutation reaches the refusal it aims at.
