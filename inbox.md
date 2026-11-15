[2026-11-15T15:19Z] a4d2870e4a0e
MODEL: Claude Opus 5.5

LAYOUTREAD2 to network-accounting (lead), Sun 15 Nov 7:18 AM PST. The compare reading on the nine.

**Verdict: no objection, the nine stated as changed recorded values.** No reader compares any of the nine to a constant, a pin or another record, and none looks anything up by one. All nine are only carried.

Where I read: the branch head, which is still `75a078438bb5` (no new commit since its import_order regeneration), and `main`'s base `3be5b93aa82`. None of the reader files below changed between `3be5b93aa82` and today's `main`, `adb33036ac0` [ran: git diff --stat]. Paths below are on the moved tree; NET means `verity/protocols/resource_accounting/network`.

1. `CLEAR["warden_certificates"]` (`chain_record.py:92`): **only carried** [read].
   - It is written as a step's `in_the_clear`. It is rendered into report.md's table (`chain_record.py:968`).
   - `certifier_piece.py:201` lists `steps.*.in_the_clear` as a public key name.
   - `kind_scan.py:135` checks only that the value is a string.
   - `released_keys.json:86` gives the key the kind `structure`. The released-keys guard rules by key and kind, never by bytes (`released_keys.py`, docstring lines 1-2).
   - The leak guard searches for private values, never for an expected text.
   - Tests (`test_certifier_piece.py:839`, `test_demo_stub.py:304`, `test_network_attester_chain.py:420`) compare against the live `CR.CLEAR[...]`, so code is compared with code.
   - No Lean, `block_certifier` or composite code reads `in_the_clear` [ran: git grep].
2. `CLEAR["warden_endpoint"]` (`chain_record.py:95`): **only carried**, with the same readers as item 1 [read].
3. scan_zk's `source` (`NET/prover/scan_zk.py:773`): **only carried** [read].
   - It goes into `summary["source"]` (2123) and the log (2154).
   - It also goes into the stand-in round window's `of` (1766), which is carried like item 5's.
   - No reader reads a scan summary's `source` [ran: git grep for `["source"]` and `get("source")` over network_traces, network_e2e, one_stage, NET, compose and tools/verity]. The hits are other records' `source` fields.
   - The chain compares only `warden_run` (`chain_record.py:337`).
4. n2_tie's `source` (`NET/prover/n2_tie_zk.py:215`, written at 581): **only carried**, same reading as item 3 [read].
5. The registration window's `of` (`NET/prover/network_attester_zk.py:356`): **only carried** [read].
   - The docket's `window` is provenance except its binding and seed salt (`verity/protocols/proofs/spot/verifier/one_stage/docket.py:42`, `:65`). `docket.check` reads neither `of` nor the window's other provenance.
   - `of` reaches only `digest(rec)`. That digest is compared only with the same run's own receipt (`one_stage/audit.py:237`, Lean `Spot/Record.lean:127` and `:224`). A record made before the move still carries its old `of` in both, so it stays self-consistent.
   - None of these reads a stand-in docket's `of`: the network verifier Python, Lean under NET, compose's Lean, `service.py`, `registered.py` or the period check [ran: git grep `"of"`].
   - The readers that do compare `of` are each a5 stand-in's own check (`a5_network_attester.py:276`, `a5_sparse_mla.py:112`, `a5_row_tables.py:82`, `a5_tp_gates.py`). None of them reads network_attester_zk's or scan_zk's docket.
   - No test pins a 128-hex digest in the network, network_e2e or onsite tests [ran].
   - No committed .json, .jsonl, .toml, .txt or .csv file holds any of the nine texts. `artifacts.json` has no entry for a registration, receipt, scan or onsite record [ran].
6. The receipt's verifier text (`network_attester_zk.py:365`): **only carried** [read].
   - It is written by `docket.receipt` (`docket.py:229`) and Lean `Spot/Run.lean:230`.
   - `audit_record` (`one_stage/audit.py:237`) and `Spot/Record.lean` read only the receipt's `registration`. `service.py` reads only `received`.
   - `exfiltration.py:798` compares a verdict record's `verifier` (its module and commit), not a docket receipt.
7. The onsite `PREMISES["certifier-key"]["is"]` (`onsite_record.py:689`): **only carried** [read].
   - It is copied into `found[name]["is"]` (around 1484). Lookups go by the premise's name, `discharged_by` and `checked` keys, never by `is`. `RESTS_ON` and the lock are compared through statement digests.
   - Test `test_onsite_record.py:654` checks five substrings, none of them the changed part. Its other uses compare `OR.PREMISES` with itself.
   - The committed onsite fixtures (`tests/fixtures/onsite_periods/*.json`) hold no `is` at all, and `onsite_periods.py` reads none [ran].
8. The trace's `meta["tick"]["where"]` (`corpus.py:857`): **only carried** [read].
   - It is read only by `encodings/by_class.py:261` into a row's `certifier`, which the table displays (469).
   - `test_by_class.py:397` writes its own `where` into meta.
   - `chain.py:682` reads the chain report's top-level `where` by prefix (`off the developer's node`, `gate: `), not the tick's.
   - No pin covers a trace's digest: there is no corpus pin in `network_e2e`, and the `art:` encodings fixtures hold no text of the nine [ran: git grep].
   - So no pin fails on the moved tree.
9. The onsite `CHECKED[AUDITOR_RECORD]` (`onsite_record.py:504`, the paths map's module path): **only carried** [read].
   - It is written as `"is"` at 1523, 1582, 1617 and 1960.
   - Tests compare it with the live `OR.CHECKED[...]` (1122, 1789, 1809, 1886). Test 656 checks two substrings that are not the module path. Test 931 reads `CHECKED`'s keys only.
   - The canonical-JSON digest at 1240 covers the census line, not the record.
   - No fixture holds the text.

Recorded-text pins: `benchmarks/one_stage/tests/test_recorded_texts.py` pins only the a5 root tag, `a8.window.of`, `tampered.arm.*` and `negatives.*`. None of them covers the nine, and no other pin of that kind exists in our paths or proofs' six files [read].

Hand commits: none of the worker's hand commits on the branch (`8727a5d853f` through `75a078438bb`) touches the nine sites or states them [ran: git log and diff]. The tip already carries the moved texts at all nine sites [ran]:
- `network_attester_serve.check` and `network_attester_chain.io` (items 1 and 2);
- `network_attester_serve.py` (items 3, 4 and 8);
- `network_attester_zk.py` twice (items 5 and 6);
- `network_attester_chain.RECORD` (item 7);
- `verity.protocols.registration.verifier.registration` (item 9).

Pull request: none open for `cursor/top-ontology-move-824c` at 7:13 AM PST [ran: gh pr list --head].

D confirmed [ran], with each tree importing its own `verity`:
- `tools/verity/cli/tests/test_run.py` on `main`'s base `3be5b93aa82`: 97 passed.
- On the tip `75a078438bb`: 1 failed, 96 passed. The failure is `test_parties_apart_meet_only_at_a_configured_listen_port`.

Whether the worker's 102 named tests include `test_run.py`: not known. Its list is in no commit message, and my notes clone is gone from this machine. It will be known when the pull request opens with its list.

The pull request should state by name that items 1 to 9 changed as recorded values, and that records made before the move carry the old texts and are still read, since nothing compares them.

TIMER: none

