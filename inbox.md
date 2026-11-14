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

[2026-11-14T23:06Z] 0f3e2ca0c057
network-accounting worker to lead, Sat 14 Nov 3:09 PM PST. The three CoreRemoteNetwork guards are case (a), as you ruled. They are mended and pushed, and the onsite package builds on a node.

PER GUARD (verity/protocols/compose/verifier/onsite/CoreRemoteNetwork.lean). Each one tests egress.strays [monitoringPeriodOn AUDITS, policyOn (record 0)].
- Expected (all three): "record: verity/inference-only/v0 committed {digest of record 0}, verity/certifier/v0 committed {D}".
  - Line 130 (AUDITS [record 1]) and line 133 (AUDITS [record 0, record 1]): D is the digest of record 1.
  - Line 136 (AUDITS []): D is the empty string. auditRecords names ByteArray.empty for a period with no compute audit.
- Got (all three): "record: verity/inference-only/v0 and verity/certifier/v0 named different values".
- Why: e9ca86fb740 ("A tie's rejection names the tie and its two subprotocols, and neither value") changed Tie.strays to "{what}: {s} and {x.1} named different values", since either value can be witness. The ids and the order are unchanged: the tie's first name is the policy, then the period.
  - e9ca86fb740 moved the guards of CoreRemote.lean and CoreOnsite.lean, but this file was not at e9ca86fb740, so nothing moved its guards. It reached architecture later, from the collect branch, with the old text.
  - The β candidate was not involved: proofs' CoreOnsite β break is already mended on the collect branch (06c17686182, β := some n).

SORT: (a). The guards were stale against a deliberate change, and I updated them to the value Tie.strays gives, the form CoreRemote's own guards use since e9ca86fb740. The comment above them now says the tie "names the tie and the two subprotocols and neither value".
- Owners: the handle registry gives verity/protocols/compose/verifier/onsite/ to @soundness, not proofs and not us. I edited it on your ruling that the change is ours; the commit says why.
- Not changed, on purpose: line 109 and tools/verity/cli/tests/lifecycle/test_lifecycle.py:210 still expect "record: … committed …". Both read Same.strays (the record subprotocols' check), which e9ca86fb740 left naming both digests, and they still pass.
- The Python reference already gives the new text (compose/verifier/lean/tests/reference.py:640). No kept record or other lane's test compares this tie's old text.

COMMIT: 484cdd0920b24ce7855725077cbe3fa27f49adc0, merged into the collect branch at 7a15e7f53828010a9c555a757ed1e132780b525e (pushed, no force).
- Moves crossed by plain merge: d05c1cbd290 (replay_chain_map), 07a82e30b34 (inference_only_map), 5838ae69d1a (draw_compound_map) and ea478e5cebb (terms_map). None of their old names is in my added lines: terms_map's "period" is scoped to PoUW files, and replay_chain_map's "verity" root maps to itself. None was crossed by hand.

BUILD: node run r20261114-225913-d05c at 7a15e7f53828.
- `lake build CoreRemoteNetwork CoreRemoteNetworkNoPouw`: exit 0, both built.
- `lake build` of the whole core_onsite package: exit 0, 783 jobs, no errors. CoreOnsite, CoreComposites and the onsite, onsite-counted and product executables are among them.
- SCANLEAN2 has been told the module, the commit and the build. No marks and no check runs.

