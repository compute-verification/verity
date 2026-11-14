[2026-11-14T23:05Z] a40d0ca7237a
STAMPS2, the lengths row's replay-chain test, pushed.
Commit f59553a27eb5f57742437d3a9eac98fcf37501d1. Collect head after the push: e0d8bb23ceba74d928cf4e6b86cb20186d5c0f8b.

Where the function went: I split it rather than move it whole, so that no assertion is lost or doubled.
- Its check against vLLM (lines 108-111: `RC.Shape`, `RC.widths(chain).LW == sh.LW`, `RC.arrival_fields(chain) == LR.request_fields(sh)`) is in a new file, benchmarks/network_traces/tests/test_lengths_row_arrivals.py, as `test_the_request_row_is_the_arrival_slots_layout`, with its assertions exactly. It imports `verity_vllm.compiler.replay_chain` (the module's name on the tip) and `verity.protocols.accounting.certifier.prover.lengths_row`. No file under network_traces/tests already tested the lengths row; the suite declares `adapters/vllm/verity_vllm` whole as an input.
- The rest of the function, the request row's bits checked against its own widths, slot refusals and the fails row, stays in the verifier's file, unchanged, as `test_the_request_row_is_k_slots_as_a_struct_of_arrays`. It gains one assertion with no adapter: `LR.request_fields(sh) == [("valid", 3, 1), ("id", 3, 7), ("P", 3, 10), ("O", 3, 10)]`. The file's import of replay_chain is gone, and its docstring's first line no longer says "the arrival slots' layout". Its other tests are untouched.

Item 3: had the function moved whole, nothing under verity/ would have held the lengths row's request layout beyond its total bit count (`test_the_block_reads_two_rows_and_returns_the_lengths_row`). So the self-contained test belongs in the verifier's suite, and it is there now: the same lines, with the stated fields asserted directly.

Item 2: the map had not rewritten it. lengths_row.py's docstring now says "the layout of `verity_vllm.compiler.replay_chain.arrival_fields`" (the paragraph rewrapped, nothing else).

Tests by file [ran]:
- test_network_certifier_lengths_row.py: 35 passed. Before, the whole file failed at collection (an import error).
- benchmarks/network_traces/tests/test_lengths_row_arrivals.py: 1 passed.
- tests/test_boundaries.py with tests/test_integration_boundaries.py: 9 failed, 23 passed both before and after, the same nine tests, and no failure line names lengths_row, replay_chain or the new file in either run.
- Ruff is clean on the three files.

Prose vllm's comment names, not changed by me (docs and docstrings that still say `replay_chain`): prover/__init__.py (1), verifier/__init__.py (1), charge_chain.py (2, trusted), verifier/PROTOCOL.md (4), verifier/README.md (1), protocol/charge-chain.md (2), protocol/served-shape.md (7). Question (yes/no, my reading stands at 15:25 PST): do I point these at the new path before 4:50 PM? I take no: the 5:00 PM rename rewrites the certifier's prose anyway, and charge_chain.py is trusted.

Crossing: `restack.py na-lengths --main origin/cursor/na-architecture-b51f` refused 85f22cb11aa0, a docket_names_map rerun whose result the script does not reproduce on its base. The original docket_names move was already in my branch, and none of the map's 8 old names is in my added lines [ran], so I merged plainly, with no conflict.

