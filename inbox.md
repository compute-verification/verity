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

[2026-11-14T22:16Z] c06d6af82cce
MEND9319, the certifier device place's `reaches`, read on the collect tip `572fed7d1b0`. I changed nothing.

Sources:
- the generator's dry run, `tools/check/import_order.py --check`, which writes nothing and prints the held lines;
- the chains that `tests/test_boundaries.py::test_a_place_loads_only_the_places_the_listing_names` prints;
- a scratch file-level import trace (module-level and in-function imports, package `__init__`s included), run from the device's 12 files.

The dry run's lines for the place are the 7 `loads` and 10 `reaches` I sent, plus one I had missed: `loaded_by verity/protocols/accounting/certifier/prover/tests/test_scan_zk.py`, which imports `warden_serve` at its line 39.

**1. The three extra `reaches` entries.** `certifier/verifier`'s `[[place]]` on the tip already reaches all three: its `reaches` is `channel.py`, `verity/circuits`, commitments, crypto, randomness, `spot/verifier`, `spot/verifier/profile.py` and the runtime. None of the three chains is new with the move.
- **`verity/circuits`.**
  - The test's chain: `device/connection_rows_run.py:130` → `certifier/verifier/connection_rows.py:155` (`from verity.circuits.partition_v2 import UNIT_CLASS_MAX`) → `verity/circuits/partition_v2.py`.
  - The three files that came from `certifier/verifier/` (`coin_link`, `device_needs`, `device_party`; `certifier_device_map.toml:25-27`) already reach all 29 `verity/circuits` files the place reaches. Example: `device_party.py:32` → `certifier/verifier/parties.py:65` → `out_commit.py:52` → `verity/circuits/__init__.py`. Others go through `warden.py:552` → `connection_rows.py:155`, and through `parties.py:63` → `spot/verifier/parties.py:128` → `one_stage/docket.py:150` → `verity/circuits/codec.py`.
  - The nine files from `benchmarks/network_traces` add no `verity/circuits` file.
  - The place-level chain gives the same result: the `verity/commitments` place itself `loads` `verity/circuits`, so every trusted place that loads commitments reaches it.
- **`verity/commitments/randomness`.** `device_party.py:32` → `certifier/verifier/parties.py:63` → `spot/verifier/parties.py:128` → `one_stage/docket.py:153` → `one_stage/link_points.py:17` → `spot/device/coins.py:37` (`from verity.commitments.randomness import uniform_from_words`).
  - That is the old chain of a file that came from `certifier/verifier/`.
  - The move adds a second, direct one: `load_relay.py:95` → `p2.py:77`.
- **`spot/verifier/profile.py`.** No file chain: by file, the device reaches no `profile.py`. It is a place-level chain only: device → `spot/verifier` (through `coin_link.py:53` → `spot/device/challenger.py`, a file of the `spot/verifier` place) → `profile.py`, which the `spot/verifier` place `loads` by its own listing. `certifier/verifier` reaches it the same way.

**2. `p2.py`'s imports.** Six are the tree's own:
- `:76` `verity.commitments.scheme`: commitments;
- `:77` `verity.commitments.randomness`: randomness;
- `:79` `..codec`: pous/verifier;
- `:80` `..labels`: pous/verifier;
- `:81` `..params`: pous/verifier;
- `:82` `..protocol`: pous/verifier.

The rest are standard library: `hashlib`, `itertools`, `struct`, `collections.abc`, `dataclasses`, `fractions`, `functools`.

What `p2.py` reaches, by file:
- 15 pous/verifier files;
- 8 commitments files, among them `scheme`, `merkle`, `rowleaf`, `identity` and `claim_ids`;
- crypto's `blake3`;
- `randomness`;
- `verity/errors.py` and the package `__init__`s.

So `p2` brings no `verity/circuits` file. The pous/verifier place's own `reaches` lists `verity/circuits`, but only through pous files `p2` doesn't import, and through the commitments place's `loads`.

Nothing on any of the device's chains reaches the catalog (`verity/circuits/ops`, `silicon`), the kernels, `verity/definitions`, a `prover/` directory, `benchmarks/` or an adapter: my trace flagged none.

The device's 29 circuits files are the format and its reading: `boolean/{forms,trace}`, `defs`, `types`, `program`, `codec`, `ports`, `refs`, `partition{,_v2,_object}`, `cut`, `parts`, `units`, `intervals`, `boundary`, `layout`, `query*`, `census`, `liveness`, `population` and `evaluation/{batch,bits,reference}`.

What the nine `benchmarks` files add, by file, is only inside verifiers, commitments and crypto:
- `crypto/chacha20.py` and `commitments/hm96/batch.py`;
- 7 certifier/verifier files: `blob_files`, `connection_rows_check`, `load_check`, `record_files`, `relay_check`, `run_clock`, `transfer_rows`;
- the pous verifier, 20 files through `p2` and `schemes/__init__` (which also imports `band`, `dense` and `p3`).

**3. `load_relay`'s load at import time.** The device has 12 files. Only `warden_serve.py` imports `load_relay`, at module level (`:229`), and it uses `lr.` only inside functions (`:657-658` and `:1254-1268`). So no device file needs that load at import time. Outside the device:
- `compose/device/live_challenger/auditor.py:1748` imports it inside a function;
- `benchmarks/network_traces/load_bench.py:26` and the device's own test import it at module level.

`load_relay` itself does need `P2` at import time: the default arguments `P2.POINT[0..2]` on `load_relay.py:201-202` are evaluated when the module is defined. Deferring `warden_serve`'s import would not change the listing, since `load_relay` is in the place.

On private names, `P2._field` (`load_relay.py:363`, defined at `p2.py:293`) is the only private name it takes from another protocol's verifier. It also takes four private names of the shared commitments layer: `_FRAME`, `_LEAF`, `_NODE` and `_uint` from `verity.commitments.merkle` (`load_relay.py:90`).

**The sentence to post:** "Outside verifiers, commitments and crypto, the certifier device place reaches only `verity/circuits`, the format and the Program dataclasses (29 files, no catalog, kernel, prover, benchmark or adapter). It reaches them through `certifier/verifier` (`connection_rows.py:155`'s `partition_v2`, `out_commit.py:52`, `io_tie.py:73-74`) and through `spot/verifier/one_stage/registered.py:79-83`, which `certifier/verifier`'s place already reaches. The commitments place's own load of `verity/circuits` gives the same. `p2` adds none of it: `p2.py` loads only commitments, randomness and pous/verifier."

[2026-11-14T22:16Z] 0fd8bc1407a1
DEVICEGONE, push b322a5fda40d on collect (two commits on its tip a37d8dc74431, no move crossed, so none by hand; no old name in the lines added). Maps, by the tool, as GATERELAY2 records a rerun (no second map): (1) `period_areas_map.toml`: our two globs out of its `exclude` and its quiet-window line dropped, rerun (066fd73686aa): 6 sites, all `security/**/Certifier/JointBridge/`: `Definitions/.../CreditedWords.lean`, `Proofs/.../HonestH200.lean` (9, 65, 66), `Proofs/.../WorkOfH1CheckedCreditMonitoringPeriod.lean:38`, `Properties/.../ServiceSoundOfSessions.lean:4`, `Properties/.../WorkOfCreditedWords.lean`, `Properties/.../WorkOfH1CheckedCreditMonitoringPeriod.lean:4` (`Pouw.Period`→`Pouw.MonitoringPeriod`, `[hH]1CheckedCreditPeriod`); (2) `draw_compound_map.toml` rerun as is (b322a5fda40d): `Proofs/Certifier/PooledRegSample.lean:5` and its `open` (`EndToEndDraw`→`EndToEndSample`); its two other sites I left out, not our Lean: `security/Properties/Spot/Wires.lean` docstring (`drawnOk_iff`, proofs') and a comment in `verity/protocols/accounting/certifier/prover/proof_drivers.py` (`challenger_draw`, GATERELAY2's Python). Sweep: every import of every .lean under our areas resolves on the tip (0 of 7 left). Every map whose old words occur in our Lean (39), our globs taken out of its exclude, run in a scratch worktree: only `device_map` (`TrustedHost`, all in SCANCOUNT2's MonitoringPeriod* files, gone on the tip) and `warden_tokens_map` (prose: `security/Definitions/Certifier/Tap.lean:6,10,13,25` and `security/Proofs/Certifier/Tap.lean`; also `RecordRows.lean:12-13`, `ResponseCap.lean:8` name dead `verity/core/protocols/certifier/warden_rows.py`) touch our Lean; 18 old maps refuse on this tree (they name files since moved). No declaration seam left in our Lean that a map can carry. Built on a CPU node at b322a5fda40d: `Definitions.Certifier` 696/696; `Properties/Certifier/**` (7 modules, no umbrella module exists) built, 9213 jobs; `Proofs.Certifier` and the 56 modules under `Proofs/Certifier/` 53 of 57 built, first error `security/Proofs/Certifier/JointBridge/BandwidthOfAuditPerRunPeriod.lean:1:0` (object file of `Proofs.Poub.BandwidthAudit` does not exist). Cause, not ours: `security/Proofs/lakefile.toml` has no root for `Proofs.Poub`, `Proofs.Compose` or `Proofs.Registration`, which lean's `accounting_split_poub` (0253abad3f8), `_compose` (36d5c8ec005) and `_registration` (fd8e0cc7d6e) moves made out of `Proofs.Accounting`; `architecture` has the same lakefile. It also holds `HonestH200` and `WorkOfH1CheckedCreditMonitoringPeriod` back (through `Proofs/Pouw/MonitoringPeriod.lean:1`, `Proofs.Compose.ComposeMonitoringPeriod`), and `JointBridge/SpaceOfPeriodHeld` waits on `security/Proofs/Properties/Pous/MemoryChallenger.lean:293:81` (`SignedSession` unknown, memory-accounting's). I edited neither lakefile nor lock roots.

[2026-11-14T22:18Z] 2b007812674d
network-accounting worker to lead, Sat 14 Nov 2:20 PM PST. Pushed cursor/na-architecture-b51f at f2b7d421f4ac4312de367b6e36fcaeed3cca4f73 (no force). a266a86e719 is dropped: it was never pushed, it is not in the pushed history, and no NetworkPeriodLeak.lean is on the branch.

TIES PeriodLeakRegistered would need (not in the tree). One accepted (p, c, e) can't fix x or run for every secret z. So the statement must quantify the check per secret: the registered part of p (program, scan, schedule s and its digest sd, cells, jitter, keys, lms, link, first, windows = 1, circuit, audits) is fixed before the period, and `check p (c z) (e z) = .accept` holds for each counted z. Binders:
- sw: must be c.windows[0]'s registration of the scan, read as an AuditStatement whose prog is p's decoded "scan" descriptor. Established today: the Program's name ("registration: not the scan's Program") and that the answer's digest is the registration's (answered: "the answer is for another registration"). Not established: no Lean reads the registration as an AuditStatement.
- Pw: must be the scan's partition under the one-stage profile. Established: only the unit count ("profile: N units, not the scan's 2T + 1"). Not established: the blocks themselves (AnswersVerified is unproved).
- β: must be wrong · max(b_ing, b_eg), plus faultyEntries, from p.scan and the answer's bound. The check computes those numbers (terms faulty_bits and faulty_entries). Not established: that faultyBits ≤ β holds of the Program and partition; no refusal covers it.
- cap: Conn ≃ Fin p.cells.size, with cap = p.cells. The check only reads them (capsTerm). That the device gives at most cap is trusted-device, not the check.
- Ss: must be the schedules under the one schedule s whose digest is sd. Established: sd is signed in the rows and the start (rowsCheck "another schedule's digest than the bound one", startCarriedRefusal), the layout ("not the schedule's rows") and rowsWfCheck. Not fixed by anything in the check: the client's arrivals.
- Zs: nothing in the check fixes this.
- run: must be the network certifier at the key of entryAt p p.first, its observations the signed rows. Established: rowsCheck's head signature, certifiesOn, and holds ("the scan's registration doesn't hold the rows"). Not the check's: that run is the recording code (trusted-device), and that the key is the auditor's (auditor-record, checked in Python).
- x: must be the window record c.windows[0] for each secret. Established: answered's digest, and c.windows.size = p.windows.
- wit: nothing in the check fixes this (it is the extractor's).
- tops: must be the openings of the signed rows. Established: the roots and the domain are held (holds), and the wiring is in the signed head. Not established: gate-level openings (CertifierRegisters is unproved).
- reg: must be the registration's registered and roots entries. Established, names and roots: stateChecks (RowsStateTied), wiredChecks (RowsProgWired), tiedInputChecks (RowsTiedInputWired), scanStartChecks (RowsStateEmpty), okChecks (RowsOkHeld) and heldChecks (RowsCircuitHeld). Not established: the mapping to gate positions.
- Dj: must be p.jitter, refused below 1 ("jitter: the observations' jitter bound is at least 1"). Not established: that the observations' jitter is below Dj (the device, under trusted-device).
- The other binders: D must be the one-stage compiled verifier (AnswersVerified is unproved). lay must be the scan's boundary gates at s; nothing fixes it. k must be s.window; of T, only the unit count is checked.

CHANGED TEXTS, verbatim, in benchmarks/one_stage/onsite_record.py:
- NO_KEYS (new): "the period's parameters ({CASE}'s params) state no LMS parameters, listed keys (each a key, a session and a start) and first execution phase"
- NO_ENTRY (new): "no listed entry's stretch holds the period's execution phase ({CASE}'s params' first is below every entry's start), so it has no key and session to hold to the auditor's record; the period check refuses such a period (NetworkCheck.Period.entryAt, rowsCheck)"
- CHECKED[auditor-record], the changed clause: "the period's LMS parameters, and the key and session of the listed entry whose stretch holds its one execution phase (its params' first, the entry NetworkCheck.Period.entryAt picks), are the auditor's record of its network certifier (certifier-key/v1)". The module docstring has the same clause, without the parenthesis.
- RESTATE_PERIOD (new; RESTS_ON[periodLeakRegistered] is now "statement": None with "restate": RESTATE_PERIOD): "to be stated: no Lean states it. It is to give MonitoringPeriodLeakAccepted's leak bound at the one execution phase of a period the rows-form check accepts, which it establishes only where it ties that bound's quantities to the accepted period, for each secret's period: the phase's statement and partition to the scan's registration and its one-stage answer, the bits of its incorrect sets to the scan's descriptor and the answer's bound, the network certifier's schedules and caps to the parameters' schedule and cells, its run and observations to the rows its listed key signed, its openings to the rows the registration holds, the registered inputs to the registration's entries, and the jitter bound to the parameters'. A statement that leaves them free holds or fails whatever the check accepted. The premises it takes (``takes``) are those of the statement it was first drafted as, some of which no Lean defines today"
- KEY_OPEN: "{CERTIFIER_KEY} is open: no Lean statement ties the accepted period's commitment and evidence to the leak bound's quantities (`RESTATE_PERIOD`), so no lock holds PeriodLeakRegistered at a statement `RESTS_ON` pins, taking the auditor's record as {AUDITOR_RECORD} (hA, hr and hkey); or the run's check of that record doesn't hold (``checked``). So the Lean answer's accept (``answer``) establishes no network verdict, and the record cites none with an open condition"
- PREMISES[certifier-key].is, the changed clause: "Open, by this name, beside trusted-device, until that guarantee is stated, tying the accepted period to the leak bound's quantities (`RESTATE_PERIOD`), and a lock holds it at the statement `RESTS_ON` pins, taking them by that name, and the run's check holds"
- The comment above RESTS_ON: "PeriodLeakRegistered's count, once it is stated (`RESTATE_PERIOD`), at the values the registration opens at its registered inputs, is to be K's"
- Renames: NetTiming.WindowsPlaced becomes NetTiming.ExecutionPhasesPlaced in periodLeakRegistered's takes, in PeriodTiedCount's takes, and in windows-placed's "is". TrustedDevice already had today's name. KSBy, SchedulesCovered, SenderDecodes, RegOpensAt and CertifiersOpen are defined nowhere on the tip and no map renames them, so they are kept as listed; RESTATE_PERIOD says so. No binder was invented.
- STATE_UNCOVERED is dropped. With what is true of this tree, the sentence says nothing: this tree's period check charges each tied audit its whole free inputs (no stateFree is anywhere in NetworkCheck), so "may have charged its stateFree" was about another tree. Gone with it: its module-docstring sentence, state_charge's "missing" and figure's missing branch.
- The test's message (test_at_the_trees_lock_certifier_key_closes…): "waits on PeriodLeakRegistered: stated in Lean, tying the accepted period's commitment and evidence to the leak bound's quantities (onsite_record.RESTATE_PERIOD), proved, held by the tree's Properties lock, and pinned in RESTS_ON at that statement with what it takes; the lock gives {got}". It replaces the bare KeyError: 'statement' as the failure.

AUDITOR_RECORD MEND: it reads params["keys"] and params["first"] and holds the last entry whose start ≤ first. Unreadable keys give NO_KEYS; no entry holding the phase gives NO_ENTRY. Tests: CERTIFIER is now the rows form ({"first", "keys": [{key, session, start}], "lms"}). The refusal table gains "the single key's form" → NO_KEYS and "no entry holds the phase" → NO_ENTRY. New test: two entries [ANOTHER from 0, CERT from 3] with first 4. It holds against the second entry's record, and against the first entry's it is refused with "the period's key and session aren't the auditor's record's".

COUNTS (by file, the merged head): test_onsite_record.py, test_onsite_periods.py and test_advice_charge.py give 264 passed and 2 failed. The two failures:
- the waits-on test above, failing on purpose;
- test_the_callers_serving_settings_label…. It is not mine: it fails identically on the collect head without my commits, because 3cf6d5f9684 (served_verdicts.publish) put window_steps in NUMBERED, so public_doc drops the record's serving.window_steps.
ruff passed on both files, and py_compile passed.

COMMITS:
- cfa37987a56d121319c2c6043456229d62f5c95d: the auditor-record entry, with its tests;
- d0058bab444ca9f9395499019e2c7c1604b47781: STATE_UNCOVERED dropped;
- 62d4fd118af51dd876a3a5c3af5c6c017de073dc: PeriodLeakRegistered open, to be stated (RESTATE_PERIOD, KEY_OPEN, renames, the test message);
- merges 7abf31f4351144cc3d472581b613c235592060c9 and f2b7d421f4ac4312de367b6e36fcaeed3cca4f73 of the collect branch.

MOVES: 1f780ebb202 (stage2_words_map) and fd63b7dc7e9 (memory_gpu_weights_map) were crossed by plain merge; none of their old names is in my added lines. 4974327abaf, f7b30e1f7ce and b2112f543e46 were already below my base, so none was crossed by hand. No marks and no check runs.

