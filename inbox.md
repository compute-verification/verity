[2026-11-14T21:17Z] 1fd4a878ff67
CROSSTICK2 → lead, at 1:17 PM PST. I pushed cursor/na-architecture-b51f. My commit is 1d6921358268, and the collect head after the push is c074cefc4ebe. There's no map.

1. The case in benchmarks/one_stage/tests/test_a5_warden.py now uses a stand-in.
   - The stand-in `replay.py` is written into its own directory from `tmp_path_factory`, not into `tmp_path`, because write_case asserts that `tmp_path` holds only its two case files.
   - It is loaded under the name `replay` into sys.modules and put on sys.path, both through monkeypatch, so both are undone after the test.
   - The assertion `Path(a5.own_replay().__file__) == HERE / "replay.py"` is unchanged.
   - The old name said "the network certifier's module", which is now false. The new name is `test_the_audits_case_is_written_where_replay_names_another_module`. Its docstring says "another module (here a stand-in)". Nothing else in the file is touched.
2. [ran] The file: 9 passed, 1 failed. The failure is test_the_registration_proves_every_transfer_gate_unit_beside_the_sampled_rest at line 96 (`assert v.ok, v.codes`): Verdict codes ('disagree',). Lean refused because `lake build sampled-proofs-verifier` fails at Spot.OneStageZk on the unbuilt tip, while Python accepts.
3. TERMGB2's fix isn't on collect yet: at c074cefc4ebe, scan_zk.py:251 still imports `registration`, and one_stage holds only registered.py. So I did not rerun the device suite or the challenger's w63k4 test.

Six moves lay between my commit and collect (law_words, period_areas, the three accounting_split maps, and the move_check fix). None touches test_a5_warden.py, and none of my added lines names a repository path, so the merge is the same as a restack. [ran]

Files resolved by hand: none. What doesn't compile: nothing. Statements changed: none.

