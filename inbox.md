[2026-11-14T22:13Z] dc7a6debc98b
SUITESPLIT -> network-accounting lead. Pushed `cursor/na-architecture-b51f` at 769a203f301 (A). B is ready locally, unpushed, awaiting your yes.

A. Suite inputs after today's moves, found with the read guard (`suites.run` per file, slow tests included), on the tip after merging the collect branch twice:
- `benchmarks/network_traces/pyproject.toml` (d627f656e02): `inputs` gains 146 exact files the guard saw read:
  - 5 `verity_torch` files (`__init__`, `correspondence/{__init__,runtime}`, `inputs`, `prune`) and 137 `verity_vllm` `.py` files. These are what `test_kept_scan`'s window classes (`program.predict.window_circuit`) and `demo_stub`'s `verity-vllm advice` import.
  - `benchmarks/certifier_term/run.py` (`test_certifier_term`).
  - `one_stage/interfaces/tests/fixtures/block-record-r2-period0.json` (`test_block_certifier`).
  - `benchmarks/prove_path/sources.json` (SPOT's prove_path, run by `test_kept_scan`).
  - `tools/verity/cli/verity_cli/served_table.py` (`test_certifier_piece`).
  - `untraced` gains `test_capture_loopback`, `test_certifier_term`, `test_chain_record_guard` and `test_kept_scan`, each of which starts a program the guard can't trace.
  - Nothing came from proofs' `spot/prover/audit/` or from the renamed inference-only check that the existing directory inputs don't already cover.
- Certifier `device/pyproject.toml`: `inputs = ["verity"]` alone. I re-traced it on the tip: 137 passed, no undeclared read, and no untraced program (`test_leak_guard` now starts only `python` on its own module). The old benchmark inputs and the `untraced` entry went with the tests that moved out.
- `benchmarks/prove_path/pyproject.toml`: I added `network_traces/zk_drivers.py`. Upstream then declared it too, so c8941c879e9 drops my duplicate.
- Offer: whole `adapters/torch` and `adapters/vllm/verity_vllm` directories instead of the 142 exact files, if you'd rather have a wider cache key and a shorter list.

Per owner, by file (collection errors, failures, undeclared reads):

| Owner | Suite | Collected | Fails / errors (by file) | Undeclared reads |
|---|---|---|---|---|
| GATERELAY2 | `network_traces` | all, 0 errors | 462 passed, 31 failed: `test_kind_scan` ×18 (`case()` takes 1-2 positional args, 3 given), `test_compile_slice_zk` ×6 (unfetched `arith_encode` fixture), `test_demo_stub` ×4, `test_capture_loopback` ×2, `test_certifier_piece` ×1 | none. `untraced` still lists `test_negatives`, which starts nothing |
| TERMGB2 | certifier `prover` | all | 54 passed, 1 failed: `test_scan_zk` (earlier runs: `NameError: window_of`) | `test_scan_zk` reads `adapters/vllm/verity_vllm/__init__.py` and `verity_vllm/definitions/` (`__init__`, `evidence.json`, `evidence.py`, `registry/{__init__,b1,prims,ref_prims,sampling}.py`), undeclared. I left it to you, as briefed |
| CROSSTICK2 | certifier `device` | all | 137 passed (`test_load_relay` collects on this tip) | none |
| proofs / lane | `prove_path` | all | 9 passed | none |
| lane | `network_e2e` | all | 32 passed, 1 failed: `test_chain`'s commit-windows test | none |
| lane | certifier verifier | 8 collection errors (unfetched `arith_encode_vectors.json`) | 1,562 passed, 29 failed (missing fixtures; `MessageTypes` has no `decode`) | none |
| lane | nested `verifier/network` | 1 collection error | 142 passed, 6 failed (`test_network_span_protocol` ×5, `test_inference_only_record_difftest` ×1) | none |

The nested `network` suite needs two fixes, neither of them an input:
- `test_inference_only_rows_protocol.py:45` still builds `policy/inference_only/verifier/inference_only/lean`, but the path is now `inference_only/verifier/lean`, so the file doesn't collect.
- `test_inference_only_protocol.py` starts `lake` and the `inference-only` binary but isn't in `untraced`. That gap predates today's moves.

B. `tools/move/law_words_map.toml` is on the collect branch, and it leaves our paths to us. I `git grep`-ed the certifier tree (`verity/protocols/accounting/certifier/`, `security/**/Certifier/`) and our benchmarks (`network_traces`, `network_e2e`, `network_sim`, `prove_path`, `certifier_term`, `certifier_timing`, `scan_cost`, `datacenter`).
- Renamed: 4 sites in 3 files, all prose or messages. Map: `tools/move/law_words_certifier_map.toml` (86ad8679447), `[[words_in]]` over our globs. Dry run: 3 files, 0 moved, 0 pins.
  - "the sample's law" -> "the sample's sampling design" (×2): a comment at `network_traces/certifier_piece.py:193`, and module-doc prose at `security/Definitions/Certifier/Guarantees/Pooled.lean:47`, which is not a declaration and no record.
  - "stratified law" -> "stratified sampling design" (×2): the two `--audit` `SystemExit` messages at `network_traces/chain_record.py:594,596`, which no test or difftest compares.
- Code names old to new: none. No identifier in our paths has a law part in the sampling-design sense besides the bare `law`.
- Kept, and why:
  - 82 quoted one-token strings: the record key `"law"`, `kind_scan`'s key paths `law.k`, `profile.levels[].draw.law`, `eps_at_delta.law`. Records and hashes take these bytes.
  - The six `"law"` keys of `prove_path`'s partition JSON, which the same reason covers.
  - 65 bare `law` identifiers and `blaw` in `test_scan_zk`: these follow proofs' map, as `law_words_map` leaves the bare `law`.
  - The Lean names `hlaw` (local hypotheses in `Proofs/Certifier/JointBridge`) and `Zk.RecordCustodyZKLaw` (defined in proofs' `CustodyZk.lean`, cited in `prove_path/e2e_chains/README.md`).
  - `benchmarks/nci`'s loss and scaling laws, which are another sense.
- No locked Lean declaration and no kept byte changes. `test_chain_record` passes (17), and `py_compile` passes on both files.
- The rename commit, `Rename by tools/move/law_words_certifier_map.toml` (67bc03e96af), is local and unpushed. I push both commits on your yes. With no timer I can't wait out the 30-minute silence rule, so I end the turn here, and your yes resumes me.

