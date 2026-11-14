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

[2026-11-14T22:15Z] 3d82d8f5bac9
STAMPS2, the charge's width cut (TOP-424's hunk), trusted file: applied and pushed.
Commit 9a2faee1a5ccf4fa536512bb2d31a3165ece9a41. Collect head after the push: 71d156cc0842deb240d7c3619c2e60a7af8130eb.

1. Equality, before applying [ran]. `replay_chain.widths(S).LW` is `_bitlen(S.L)`, which is `max(int(v).bit_length(), 1)`; the hunk computes `max(int(st["L"]).bit_length(), 1)` [read]. The run called the real `RC.widths(SimpleNamespace(**st)).LW` against the hunk's expression for every L from 0 through 2**32 inclusive: 4,294,967,297 values, 0 differing. L is the engine's max_model_len, and nothing below the charge caps it: the check itself refuses from 2**32 (33 bits), and the width only grows past that, so 0..2**32 decides every verdict. It also compared the old and new verdicts (taken or refused, with the width) at 0, 1 and 2**k - 1, 2**k, 2**k + 1 for k = 0..80: 240 values, none differing. At the cap, 2**32 - 1 is taken at 32 bits; 2**32 and 2**32 + 1 are refused at 33.
One difference outside L, which I state because the file is trusted: the old call also read NB, A, Q, D and P (`widths` evaluates every field), so a statics dict missing one of them raised AttributeError, and one with a non-integer value raised there. The hunk reads L alone. Both callers, compile_slice_v7 (`vars(S)`) and serve_commit (`CS.statics(...)`), hand full statics [read].

2. The hunk as applied, verbatim:
```
@@ -49,7 +49,6 @@ from verity.circuits.types import Array
 from verity.commitments import hm96
 from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
 from verity.protocols.accounting.certifier.verifier import arith_encode as AE
-from verity.protocols.accounting.certifier.verifier import replay_chain as RC
 from verity.protocols.accounting.certifier.verifier.arith_encode import Carry, ChargeBook
 from verity.protocols.accounting.certifier.verifier.head_check import BIT, _part
 from verity.protocols.proofs.spot.verifier.one_stage import produced
@@ -273,8 +272,8 @@ LENGTH_BITS = 32
 
 def check_length_slots(st: dict) -> dict:
     """`st`, a compile slice's statics, as a slice charging the length terms takes them (`compile_slice_v7`): refused
-    where an output length, LW = bitlen(L) bits (`replay_chain.widths`), is wider than a length slot's `LENGTH_BITS`."""
-    if (lw := RC.widths(SimpleNamespace(**st)).LW) > LENGTH_BITS:
+    where an output length, LW = bitlen(L) bits, is wider than a length slot's `LENGTH_BITS`."""
+    if (lw := max(int(st["L"]).bit_length(), 1)) > LENGTH_BITS:
         raise ValueError(f"statics {dict(st)}: an output length of {lw} bits, wider than a length slot's {LENGTH_BITS}")
     return st
```
No import added; `SimpleNamespace` stays imported for its two other uses. The module docstring still names `replay_chain` twice, in prose.
What the charge is already held to: the window difftest, `tests/test_network_certifier_window_lean_difftest.py`. Its vectors carry max_tokens at and past `MT_CAP`, the audit pins them against the device program's `WindowCharge.charges`, and `window_charge.MT_CAP = (1 << charge_chain.LENGTH_BITS) - 1`. No test held `check_length_slots`'s cut to that cap, so I added `test_a_slices_lengths_fit_a_length_slot_up_to_the_leans_mt_cap` to that file. It reads `def MT_CAP : Nat := 2 ^ 32 - 1` from `device/program/lean/NetworkCertifier/WindowCharge.lean` and asserts it equals `WC.MT_CAP` and `(1 << LENGTH_BITS) - 1`. Then, at 0, 1 and 2**k - 1, 2**k, 2**k + 1 for k up to 40, it asserts `check_length_slots({"L": L})` is taken exactly when L <= MT_CAP. It does not import replay_chain.
Tests, by file [ran]: test_network_certifier_window_lean_difftest.py (3 passed, the new test included), test_network_certifier_charge_chain.py (9 passed, after `research data fetch-fixtures` brought arith_encode_vectors.json, which this restarted VM lacked; without it the same 6 fail before and after my change), test_network_certifier_charge_lean_difftest.py (2 passed); 14 in all on the merged tree. Ruff is clean. Not run: the adapter's compile_slice_v7 tests, whose conftest cannot import verity_torch on this VM.

3. test_network_certifier_lengths_row.py: only `test_the_request_row_is_the_arrival_slots_layout` reads replay_chain. It builds `RC.Shape(...)` and asserts `RC.widths(chain).LW == sh.LW` and `RC.arrival_fields(chain) == LR.request_fields(sh)`, holding the certifier's lengths row to the replay chain's arrival slots. That is not the same helper: it tests where the lengths row meets vLLM's replay chain, so it belongs with vllm's move, as that one test function. I did not cut it.

4. After the change, these under verity/ still import the two (git grep): replay_chain's and replay_step's own sidecars (`replay_chain.circuit-check.py`, `replay_step.circuit-check.py`) and their own tests (`test_network_certifier_replay_chain.py`, which `importorskip`s replay_step, and `test_network_certifier_replay_step.py`), all of which move with them; plus the lengths-row test function in 3. Nothing else, and no non-test module.

5. No Lean touched: the commit changes the two .py files alone (WindowCharge.lean and AdviceV1.lean, both trees, are unchanged). The files that hold the charge to them, run by file: test_network_certifier_window_lean_difftest.py and test_network_certifier_charge_lean_difftest.py, both passing. Their Lean halves run in the Lean audit, which I did not launch.

6. `verity/commitments/pool_row.py` line 1-2 is prose: its module docstring says the pool row is "the row `CompileSlice_v6` carries from window to window (`verity_vllm.compiler.compile_slice_v6`)". It is not an import (the module imports only `verity.commitments.rowleaf` and `.gates.rowleaf`) and not an id a record holds.

Crossing: `restack.py na-charge --main origin/cursor/na-architecture-b51f` failed on 9f71545ac43, an `audit_code_map` rerun over #9896's lines whose base's rename.py can't read the map (KeyError 'words'). So I crossed by hand. Four maps were new to my branch: memory_gpu_weights (layout), pouw_sample, sampled_proofs_words and warden_tokens. The other seven had only reruns in the window. Across all eleven maps, none of the old names or paths is in my 18 added lines, and no commit in the window touches my two files [ran]. Then I made a plain merge, with no conflict. 4974327abaf, f7b30e1f7ce and b2112f543e46 were already in my branch.

