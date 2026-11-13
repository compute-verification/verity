# SWEEP-PROOFS: what the proofs' group knows from outside itself

**The proofs' group reaches outside itself in Python only through SPOT's verifier and challenger. Its Lean reaches the
interface package's eight modules. SPOT's record, challenger and parties carry the notions of the layers above it:
registration, a day's declared circuit, periods, windows, served and stand-in, a stream, a prover's identity, a seed salt
and a pool. They also hold compose's table of protocol ids.** From the other side, three protocols and the vLLM adapter
import SPOT's challenger and its produced-rows format, which are not on SPOT's surface (section 4). The registration
record's digest format lives in Flock's verifier, where PoUW's and the network check read it. The challenger's own code
decides periods, a lane's days, credit toward a period and whether a host ran `main`'s code (section 5). Each row says
what the code does, with a file and a line, and what it does not judge.

- **Main read:** `9182900fb94eefbc4a6cc5d6e14885dc6b18f5ff` (Queue: #9668, 2026-11-13 05:59 UTC). Every path and line is at that
  commit.
- **Author:** top's worker SWEEP-PROOFS. Brief: `inbox/LAYERS.md` on `cursor/verity-top-4db4`, section 4.
- **How it was read:** `rg`, `git` and the files, plus three read-only scanners of my own in `/tmp` (Python's `ast`
  for imports; a tokenizer that splits each line into identifiers, string literals and prose). There was no build, no
  test and no node run.
- **Marks:** each row is [read by me] (I opened the lines) or [counted by me] (a count from my scan, whose command is
  given).

## 0. What I took as the group

| Part | Paths at `9182900fb94e` | Files | Lines |
| --- | --- | --- | --- |
| Flock | `verity/protocols/flock/` (`verifier/` Lean, `prover/` = the `verity_flock` distribution with Rust, CUDA and patches, `sigma/`, `agreement/`, `pod/`, `tool.py`) | 1,066 for the three directories together (`git ls-files verity/protocols/{flock,spot,recursion} \| wc -l`) | 285,804 (`xargs wc -l`) |
| SPOT | `verity/protocols/spot/` (`verifier/` Python and `verifier/lean/` = Lake package `core_spot`, `device/` = the challenger, and five loose files) | (in the 1,066) | |
| Recursion | `verity/protocols/recursion/verifier/` (`firewall/` = `core_firewall`, the ZK server's release side; `recursive/` = `core_vstar_recursive`; `vstar_cli/` = `core_vstar`). It has no Python outside tests. | (in the 1,066) | |
| Security: definitions | `security/Definitions/{Spot,Flock,Claims}` (+ `.lean`). `Claims` is "Flock's claim statements" (`security/Definitions/Claims.lean:10`), SPOT's sampled soundness among them. | 10 + 36 + 23 | 923 + 4,598 + 6,690 (`Claims/` with `Claims.lean`, `xargs wc -l`) |
| Security: properties | `security/Properties/{Spot,Flock}`, `Challenger.lean`, `Firewall.lean` | 13 + 8 + 2 | 1,533 + 797 |
| Security: proofs | `security/Proofs/{Spot,Flock}` (+ `.lean`) | 21 + 2,817 | 5,447 + 856,206 |

`Proofs/Flock` is 856k lines. Its rows below are counts from my scan and the lines I opened. I did not read it through.

The scanners covered 3,997 files: every tracked file under the paths above, but `.gz` files, `.jsonl` files and JSON files
over 200 kB.

---

## 1. Everything the group imports, requires or loads from outside itself

### 1a. Python imports leaving the group: code, not tests [counted by me]

Scan: an `ast` walk of every `.py` under the three protocol directories (`import`, `from … import`, `import_module("…")`),
leaving out the standard library, the group's own modules (`verity.protocols.{flock,spot,recursion}`, `verity_flock`) and
modules a script puts on its own path from the group. That gives 1,445 import statements leaving the group, tests
included. A test is a file under a `tests/` directory. Recursion has no Python outside tests, so it has no rows here.

| Target (the listing's place) | From | Statements | Files | Where (first three files) |
| --- | --- | --- | --- | --- |
| `verity/circuits` | flock: `prover` 27 files, `agreement` 6, `pod` 2 | 168 | 35 | `flock/agreement/cut_check_agree.py:16`; `flock/agreement/layout_agree.py:212`; `flock/prover/python/verity_flock/circuit.py:…` |
| `verity/circuits` | spot: `verifier` 12, `candidates.py`, `optimize.py` | 40 + 1 | 14 + 1 | `spot/candidates.py:28`; `spot/optimize.py:68`; `spot/verifier/one_stage/audit.py:29`; `spot/verifier/one_stage/proofs.py:66` (`verity.definitions.circuits.library`, which the listing puts in `verity/circuits`) |
| `verity/commitments` (the package, `frame_v3`, `gates`, `merkle`, `rowleaf`, `window`, `history`, `coin_tree`, `hm96`, `scheme`, `indexed`) | flock: `prover` 17, `pod` 1 | 67 | 18 | `flock/pod/read_gadget.py:27`; `flock/prover/python/verity_flock/circuit.py:257`; `…/class_statement.py:764` |
| `verity/commitments` | spot: `verifier` 4, `device` 2, `optimize.py` | 14 | 7 | `spot/device/challenger.py:221`; `spot/device/remote.py:103`; `spot/verifier/one_stage/registration.py:138` (`verity.commitments.window`: `WINDOW_DOMAIN`, `window_domain`) |
| `verity/commitments/identity` | flock `prover` 3; spot `device` 2, `verifier` 1 | 6 | 6 | `flock/prover/python/verity_flock/circuit.py:1061`; `spot/device/challenger.py:225`; `spot/verifier/one_stage/registered.py:73` |
| `verity/commitments/randomness` | flock `prover` 3; spot `device` 1 | 4 | 4 | `flock/prover/python/verity_flock/partition_units.py:39`; `spot/device/coins.py:31` |
| `verity/commitments/windows` (what an adapter hands the prover) | flock `prover` 2 | 2 | 2 | `flock/prover/pod/rec_inner.py:132`; `flock/prover/python/verity_flock/circuit.py:69` |
| `verity/definitions` (`registry`, `numerics`, `word`, `encoding`, `tiny`) | flock `prover` | 50 | 10 | `flock/prover/python/verity_flock/boolean_export.py:398`; `…/class_statement.py:461`; `…/ir_lower.py:58` |
| `verity/definitions/ops` (the catalog) | flock: `prover` 14, `agreement` 1, `pod` 1 | 24 | 16 | `flock/agreement/qcall_agree.py:120`; `flock/pod/gemm_fp.py:21`; `flock/prover/pod/rec_reprice.py:39` |
| `verity/definitions/silicon` (the catalog) | flock `prover` | 14 | 8 | `flock/prover/python/verity_flock/backend.py:22`; `…/fp.py:5`; `…/instance_file.py:120` |
| `verity/definitions/kernels` | flock `prover` | 2 | 1 | `flock/prover/python/verity_flock/subcircuit_templates.py:389` |
| `verity/definitions/circuits` (outside the circuits place) | flock: `prover` 8, `agreement` 2 | 13 | 10 | `flock/agreement/circuit_type_agree.py:21`; `flock/agreement/layout_agree.py:26`; `flock/prover/python/verity_flock/backend.py:24` |
| `verity/configuration/deploy` (`config.COMPOSITES`, `manifest`, `party`, `runtime.ProtocolDef`) | spot `verifier/parties.py` | 4 | 1 | `spot/verifier/parties.py:123`, `:124`, `:125`, `:126` |
| `verity/configuration/wire` | spot `verifier/parties.py` | 1 | 1 | `spot/verifier/parties.py:127` |
| `verity/claims` (`from verity import claims`) | spot `device/challenger.py` | 1 | 1 | `spot/device/challenger.py:219`. It reads `claims.INSTANCES` at `:378` (`coin_claim`). |
| `verity/hardware/record.py` (`from verity.hardware import record as core_record`) | spot `device/challenger.py` | 1 | 1 | `spot/device/challenger.py:220`. It reads `core_record.ZERO` at `:325`, and its docstring says each ledger entry is "a record entry (`verity.hardware.record`, `Core.Record`)" at `:113`. |
| external `numpy` | flock `prover` | 36 | 30 | `flock/prover/pod/rec_reprice.py:112`; `…/boolean_export.py:174`; `…/circuit.py:67` |
| external `blake3` | flock `prover` | 4 | 4 | `…/instance_file.py:103`; `…/instances.py:213`; `…/ir_frame.py:238` |
| external `verity_numerical` (the `benchmarks/numerical` distribution) | flock: `prover` 5, `pod` 2 | 10 | 7 | `flock/pod/gemm_hill.py:569`; `flock/pod/read_gadget.py:127`; `flock/prover/python/verity_flock/bench.py:312`. The listing's `[[against]]` names these files (`tests/import_order.toml:108-119`). |
| external `research` (team tooling, `tools/research`) | flock `tool.py`, `agreement/tool.py` | 3 | 2 | `flock/tool.py:9`, `:10`; `flock/agreement/tool.py:9`. These are `[[against]]` too (`tests/import_order.toml:99-106`). |
| external `verity_config` (`verity/configuration/config`) | flock `prover/settings.py` | 1 | 1 | `flock/prover/settings.py:9` |
| external `circuit_check` (`tools/verity/circuit_check`) | flock `pod/catalog.py` | 1 | 1 | `flock/pod/catalog.py:9` |
| external `torch` | flock `pod/gemm_slowdown.py` | 1 | 1 | `flock/pod/gemm_slowdown.py:54` |
| external `matplotlib` | flock `pod/gemm_hill.py` | 2 | 1 | `flock/pod/gemm_hill.py:851`, `:854` |
| `test_format_spec`, `test_qcall_vectors`, `test_qword_vectors` (test modules of `verity/circuits`, loaded from flock's agreement) | flock `agreement` | 3 | 2 | `flock/agreement/qword_program_agree.py:73`, `:255`; `flock/agreement/qcall_agree.py:114` |

What the listing says SPOT's place may load (`tests/import_order.toml:672-691`, [read by me]): `verity/circuits`,
`verity/claims`, `verity/commitments`, `verity/commitments/randomness`, `verity/configuration/deploy`,
`verity/configuration/wire`, `verity/hardware/record.py` and `profile.py`. Its `reaches` adds `verity/commitments/crypto`
(`:693-703`). Recursion's place is `trusted = true` with no `loads` (`:649-653`). Flock is a distribution
(`:51-54`), not a place, so it has no `loads` row.

### 1b. Python imports leaving the group: tests [counted by me]

| Target | From | Statements | Files | Where |
| --- | --- | --- | --- | --- |
| `verity/protocols/pouw/prover` (`circuit`, `circuit.anchors`) | flock `prover/tests` | 5 | 2 | `flock/prover/tests/test_class_statement_unit_stager.py:31`, `:92`; `flock/prover/tests/test_pouw_rows.py:29`, `:433` |
| `verity/protocols/pouw/verifier` (and `.protocol`) | flock `prover/tests` 1; spot tests 2 | 5 | 3 | `flock/prover/tests/test_lean_registered_bit_rows.py:26`; `spot/verifier/lean/tests/test_difftest.py:62`, `:63`; `spot/verifier/tests/test_registered.py:22`, `:23` |
| `verity/hardware/site` | spot `verifier/lean/tests` | 1 | 1 | `spot/verifier/lean/tests/test_site_difftest.py:25` |
| `verity/hardware/record.py` | spot `verifier/tests` | 1 | 1 | `spot/verifier/tests/test_one_stage.py:260` |
| `verity/configuration/deploy`, `verity/configuration/wire` | spot `verifier/tests` | 7 | 2 | `spot/verifier/tests/test_parties.py:18`, `:19`, `:22`; `spot/verifier/tests/test_spot_parties.py:24-29` |
| `verity/claims` | spot `verifier/tests` | 1 | 1 | `spot/verifier/tests/test_coin_claims.py:13` |
| `verity.settings` (`from verity import settings`) | flock `prover/tests` | 4 | 4 | `flock/prover/tests/test_ring_switch.py:17`; `test_ligerito_folds.py:15`; `test_zc_round1.py:18`; `test_zerocheck_fused.py:17` |
| `verity/circuits`, `verity/commitments` (and `crypto`, `identity`, `randomness`, `windows`), `verity/definitions` (and `ops`, `silicon`, `kernels`, `circuits`) | flock, spot and recursion tests | 177 + 68 + 1 (circuits); 60 + 24 + 1 (commitments); more by subpackage | 57 + 20 + 1 (circuits) | e.g. `flock/prover/tests/plain_hidden.py:18`; `spot/verifier/lean/tests/test_difftest.py:76`; `recursion/verifier/vstar_cli/tests/test_vstar_difftest.py:27` |
| `verity_difftest` | flock 5, spot 11, recursion 14 | 30 | 25 | `flock/prover/tests/test_challenger_audit.py:15`; `spot/verifier/lean/tests/test_challenger_difftest.py:68`; `recursion/verifier/firewall/tests/_release.py:23` |
| `verity_numerical`, `circuit_check`, `research`, `verity_config`, `scipy`, `hypothesis`, `pytest`, `numpy`, `blake3` | tests | 6, 6, 1, 1, 2, 14, 186, 51, 1 | | `flock/prover/tests/test_backend_bench.py:24`; `flock/prover/tests/test_class_statement.py:21`; `flock/prover/tests/test_ir_lowering_bench.py:218` |

The listing's `loaded_by` rows name the group's tests that import trusted places from outside them: `verity/difftest`
(`:340-344`), `verity/protocols/pouw/verifier` (`:622`), `spot/verifier` (`:704-754`, 50 files of `verity_flock` and its
tests, PoUW's `h1window.py` and the certifier's `window_charge.py`) and `profile.py` (`:767-773`). [read by me]

### 1c. Lake `require` by path, leaving the group [read by me]

From each package's `lakefile.toml`:

| Package (dir) | Requires outside the group | Line |
| --- | --- | --- |
| `flock_verifier` (`flock/verifier`) | `core_base` (`verity/circuits/base`), `core_circuits` (`verity/circuits/lean`), `core_commitments` (`verity/commitments/lean`) | `lakefile.toml:10-20` |
| `core_sigma` (`flock/sigma`) | `verity_difftest` (`verity/difftest/lean`) | `:15-17` |
| `core_spot` (`spot/verifier/lean`) | **`core_protocol` (`verity/protocols/interface`)**, `lms` (`verity/commitments/lms/lean`), `verity_difftest` | `:15-17`, `:24-26`, `:28-30` |
| `core_firewall` (`recursion/verifier/firewall`) | **`core_protocol`**, `verity_difftest` | `:19-21`, `:23-25` |
| `core_vstar_recursive` (`recursion/verifier/recursive`) | **`core_protocol`** (and `core_spot`, `flock_verifier` in the group) | `:15-17` |
| `core_vstar` (`recursion/verifier/vstar_cli`) | `verity_difftest` | `:16-18` |

Which security packages require group code (for the security rows of 1e): `security_proofs` requires
`flock_verifier`, `core_vstar` and `core_vstar_recursive` (`security/Proofs/lakefile.toml:21-43`), and also
`certifier_program` and `core_randomness`. `security_properties` requires `core_spot`, `core_firewall` and
`flock_verifier` (`security/Properties/lakefile.toml:53-87`), beside fourteen packages of other protocols.

### 1d. Lean `import`s from the group's code packages to modules outside the group [read by me]

Command: `rg -n '^(public |private )?(meta )?import\s+(all\s+)?(CoreProtocol|Proofs|Mathlib|Lms|Difftest|Commitments|Circuits|Base)\b' -g '*.lean' verity/protocols/{flock,spot,recursion}`.

| Imported module (package) | Importing file:line |
| --- | --- |
| `CoreProtocol.Profile` (interface) | `spot/verifier/lean/Spot/Record.lean:24`, `:32` (meta); `Spot/OneStage.lean:78`; `Spot/Bound.lean:14`, `:15` (meta) |
| `CoreProtocol.Cli` (interface) | `Spot/OneStage.lean:77`; `Spot/Main.lean:3`; `recursion/verifier/firewall/Firewall/Protocol.lean:25` |
| `CoreProtocol.Live` (interface) | `Spot/OneStage.lean:79` |
| `CoreProtocol.Record` (interface: the record entry) | `Spot/ChallengerSession.lean:65` |
| `CoreProtocol.Ratio` (interface) | `Spot/PyJson.lean:12` |
| **`CoreProtocol.Site` (interface: the device tree)** | `Spot/SiteDifftest.lean:3`. Its docstring at `:8-11` says: "The Site is core's (`CoreProtocol.Site`, in `core_protocol`); its port is here because this package already requires the interface and the harness, and the registration log is the proof system's record." The Lake exe is `site-difftest` (`spot/verifier/lean/lakefile.toml:60`). |
| `CoreProtocol.Blobs` (interface) | `Spot/OneStageZk.lean:97`, `:105` (meta); `recursion/verifier/recursive/VStarRecursive/Protocol.lean:48`, `:55` (meta) |
| `CoreProtocol.Protocol` (interface) | `recursion/verifier/firewall/Firewall/Protocol.lean:26` |
| `Lms.Verify` | `Spot/ChallengerRecord.lean:24` |
| `Difftest` | `Spot/DifftestMain.lean:3`, `Spot/SiteDifftestMain.lean:3`, `Spot/VerifierMain.lean:3`, `flock/sigma/SigmaDifftest.lean:3`, `recursion/…/FirewallDifftest.lean:3`, `FirewallHonestDifftest.lean:3`, `VStarDifftest.lean:3` |
| `Base.*`, `Circuits.*`, `Commitments.*` (the three base packages) | 16 modules of `flock/verifier/Flock/` re-export them, e.g. `Flock/Bytes.lean:3` (`Base.Bytes`), `Flock/Program.lean:3` (`Circuits.Program`), `Flock/FV3.lean:3` (`Commitments.FV3`), `Flock/Draw.lean:3` (`Commitments.Draw`) |
| `Proofs.Flock.*`, `Mathlib` (security) | tests only: `flock/prover/tests/lean/RandomnessVectors.lean:1`; `spot/verifier/tests/lean/ExfiltrationVectors.lean:1-3`; `ExfiltrationTwoVectors.lean:1-2` |

All told, the group's Lean imports 8 of the interface's modules: `Blobs`, `Cli`, `Live`, `Profile`, `Protocol`, `Ratio`,
`Record` and `Site`, from 10 files. [counted by me]

### 1e. Lean `import`s from the group's security areas to other areas [counted by me]

Over `security/{Definitions,Properties,Proofs}/{Spot,Flock}`, `Definitions/Claims`, `Properties/Challenger.lean` and
`Properties/Firewall.lean`, these are the import lines by target. `Mathlib` (40,784), `VCVio` (158), `ArkLib` (33),
`Batteries` (64) and `Aesop` (27) are external libraries.

| Target | Import lines | What it is, and where |
| --- | --- | --- |
| `Definitions.Core.*` (Game 28, Interaction 9, Partition 2, Draw 2, Coins 2, Boolean 2, ValueSalts, Rewinding, Hm96, Game.Seq) | 55 | The shared vocabulary. `security/lean-audit.json` `layers` lets `Definitions.Flock` import `Definitions.Core.{Boolean,Game,Draw,Rewinding}` and `Definitions.Hash.Basic`, and lets `Definitions.Spot` import `Definitions.Core`. |
| `Proofs.Core.*` (Game.Poly 17, QSort 4, Game.Basic 4, Circuit 4, ValueSalts 2, Coins 2, Game.*, SamplingGeneric.FirstHit*, Rec, ProofSystem, Boolean) | 51 | Shared proofs |
| `Definitions.Hash.*` and `Proofs.Hash.Bytes` | 7 | Shared vocabulary |
| `Definitions.Claims.*` | 119 | In the group (see section 0) |
| **`Definitions.Accounting.TS.ProofLayer`** | 1 | `security/Proofs/Flock/SoundAtBridge/Union.lean:3`. The trees' `except` keeps it: "Proofs.Flock.SoundAtBridge: the bridge from C-Flock's rows link to the proof layer". |
| `Definitions.Assumptions.Flock` | 1 | `Proofs/Flock/Soundness/Assumptions.lean:8` |
| `Properties.Verity.Sha512` | 1 | `Properties/Spot/HistoryTable.lean:5` |
| `Proofs.Properties.{Flock.Nets, Spot.Bits, Firewall}` | 5 | `Proofs/Flock/SessionLoop/Concrete/Serves.lean:4`; `Proofs/Flock/Bridge/Members.lean:5`; `Proofs/Flock/Recursive/StopCountProof.lean:3` |
| Group code packages: `Flock` 555, `VStar` 20, `Spot` 29, `Firewall` 2, `Randomness` 3 (`core_randomness`, commitments), `Base` 3 | 612 | e.g. `Properties/Challenger.lean:3` (`Spot.OneStageZk`); `Properties/Firewall.lean:3` (`Firewall.Release`) |

There is also a name with no import behind it. `security/Proofs/Flock/Bytes.lean:26` declares Flock's byte lemmas inside
`namespace Properties.NetworkCertifier.Bytes` and `:72` inside `namespace Properties.NetworkRecords.Bytes`. Its docstring at
`:16-17` says the certifier's byte lemmas "build on these, under the same names". 21 files of `Proofs/Flock` open that
namespace, e.g. `Proofs/Flock/Bridge/Deployed.lean:85` and `Bridge/LeafRoots.lean:35`.
(`rg -l 'Properties\.NetworkCertifier' security/Proofs/Flock | wc -l` gives 21.) [read by me]

`security/lean-audit.json`'s `trees.except` has 28 entries. 21 of them are the group's (`Proofs.Flock.*`), e.g.
`Proofs.Flock.SpotFlock` and `Proofs.Flock.Claims`, most "C-Flock is excepted until its spec is extracted". [counted by me]

### 1f. Loads by path: scripts, path strings, test inputs [read by me]

Command: `rg -o '(benchmarks|adapters|demos|tools|infra)/…|verity/(protocols/(pouw|…)|composite|hardware|configuration|periods|claims)…' verity/protocols/{flock,spot,recursion} -g '!**/tests/**' -g '!*.md'`
gives 165 hits: `tools/research` 39, `benchmarks/numerical` 30, `adapters/vllm` 22, `verity/protocols/pouw` 12,
`tools/verity` 11, `benchmarks/pouw` 10, `verity/composite/device` 9, `benchmarks/verifier_cost` 6, `benchmarks/census` 5,
`verity/protocols/interface` 4, `verity/hardware/*` 6, and one each for pous, poub, inference_only and certifier.

| What is loaded | How | Where |
| --- | --- | --- |
| `benchmarks/pouw/hidden_zk/job.sh` | `source` in a shell script | `flock/agreement/firewall_controls.sh:23`; `flock/agreement/firewall_finish_agree.sh:23`; `flock/pod/92-zk-sessions.sh:18`; `flock/pod/93-zk-tree-from.sh:27` |
| `verity/composite/device/live_challenger/serve.py` (the auditor host's challenger service) | run by path | `flock/prover/pod/challenger.sh:53`; `flock/pod/93-zk-tree-from.sh:6`; named in `spot/device/remote.py:64` |
| `benchmarks/numerical/python` | put on `PYTHONPATH` | 18 pod scripts, e.g. `flock/pod/20-pure.sh:23`, `flock/pod/rec-env.sh:50` |
| `adapters/vllm` | put on `PYTHONPATH` or in a tool's closure | `flock/pod/92-zk-sessions.sh:19`; `flock/pod/93-zk-tree-from.sh:28`; `flock/tool.py:60`; `flock/agreement/ci.py:52` |
| `benchmarks/census/hardware.json`, `benchmarks/census/subcircuits.json` | read as data | `flock/pod/gemm_hill.py:46`, `:971`; `flock/pod/74-gemm-hill.sh:186`; `flock/prover/python/verity_flock/boolean_export.py:1075` |
| `benchmarks/numerical/tests/bench/data/f6-menu.json` | generated into Flock's Lean verifier | `flock/verifier/VStar/Menu/Pinned.lean:5-6` ("Generated by `benchmarks/numerical/…/unit_costs_menu.py --lean` … do not edit") |
| Suite inputs (`[tool.verity.tests] inputs`) outside the group | `pyproject.toml` | `flock/prover/pyproject.toml`: `verity/hardware/record.py` (`:84`), `benchmarks/census` (`:92`), `verity/protocols/inference_only/verifier/nci/lean/scripts` (`:97`), `verity/hardware/__init__.py` (`:109`), `verity/protocols/interface` (`:144`, `:159`), `verity/composite/device/live_challenger` (`:158`), `adapters/vllm/verity_vllm` (`:180`, `:187`, `:211`), `verity/protocols/pouw/verifier/tests` (`:196`), `verity/protocols/pouw/verifier/lean/check/models.json` (`:255`), `verity/protocols/{certifier,pous,pouw/prover}` (`:261-263`); `flock/sigma/pyproject.toml:12-19` (`adapters/vllm/verity_vllm`, five PoUW files); `recursion/verifier/{firewall,vstar_cli}/pyproject.toml` (`verity/protocols/interface`, `verity/hardware/__init__.py`) |
| A path named in an error message | string | `flock/prover/python/verity_flock/circuit.py:604`, `:689` ("split the unit (e.g. PoUW's lanes form, verity/protocols/pouw/verifier/leaf_lanes.py)", and `adapters/vllm/verity_vllm/pipeline/serving_templates.py` at `:688`) |
| `adapters/vllm/verity_vllm/commitments/vllm_v1/PROTOCOL.md` | cited as the spec | `flock/prover/live/src/vllm_block.rs:2` |

---

## 2. The notions of the layers above, by name [counted by me unless marked]

**How counted:** `/tmp/sweep/notions.py` reads the 3,997 files and splits each line into Python and Lean identifiers,
string literals (Python strings up to 100 characters, Lean `"…"`, and JSON and TOML values) and prose (comments,
docstrings, Markdown). Rust, CUDA and shell are split at `//`, `#` or `/*`. A notion's regex matches either snake case or
camel case (e.g. `day` matches `day`, `_day`, `declaredDay` and `DAY`, not `today`). A line counts once per notion and
kind. "Code" means identifiers plus string literals in files outside `tests/`. Each count is lines (files). The prose
column is every area together.

Several words also have an unrelated sense in the proof system. The **Sense** column says what I read the hits to be.
A large count with no upper-layer sense is marked "noise".

| Notion | All lines (files) | Code: flock | Code: spot | Code: recursion | Code: security | Prose | Tests | Kind | Three places that matter, and the sense |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **registration** | 3,637 (454) | 193 | 393 | 22 | 103 | 2,152 | 821 | **Record format, check, parameter** | SPOT's "registration record", format `verity/registration/v1` and `v2`, checked by R1-R8 before any sample: `spot/verifier/one_stage/registration.py:145` (`FORMAT`), `:229` (`check`), and its Lean twin `Spot/Registration.lean:108-113`, `:163` (`structure Verifier`). Its digest is defined in Flock's verifier: `flock/verifier/Flock/WindowRecord.lean:21-23`, whose docstring at `:9-13` says PoUW's check, the network check, SPOT and `RecVStar` read it there "without a new requirement". The statement is in `security/Properties/Spot.lean:14`, `:62-111` (R1, R2, …, 40 lines). The ZK server also numbers "the value service's registrations" (`recursion/verifier/firewall/Firewall/Record.lean:21-30`, event `register`). `spot/verifier/service.py` has 84 lines (`_registration_token`, `:1021`). [read by me] |
| **registered / register** | 9,883 (837) | 981 | 742 | 44 | 1,756 | 3,958 | 2,474 | **Record field, check, parameter** | Registered values: weights, constants and tables committed once. They are a v2 record's `registered` field (`registration.py:50-55`; `Spot/Registration.lean:113` `KEYS_V2`), held by R7 to the verifier's `registered`, `pinned`, `supplied`, `held` and `outputs` parameters (`Spot/Registration.lean:179-197`). Flock's verifier reads a `--registered` file (`Flock/Registered.lean:23`, format `verity/registered-reads/v1`; `FlockVerify.lean` 91 lines). Proofs: `Proofs/Flock/SpotFlock/EntriesAt.lean` (45 lines), `Proofs/Flock/Soundness/Discharge/ZkReg/Checked.lean` (50). [read by me] |
| **declared / declaration** | 606 (126) | 75 | 120 | 0 | 9 | 227 | 180 | **Parameter, check, ledger entry** | A day's declared circuit. The caller passes `day` and `declarations` (`registration.py:237-238`; `Spot/Registration.lean:200` `declared : Option (Nat × Array (Nat × String))`). SPOT decides `R2-undeclared`, `R2-declared` and `R2-declared-twice` (`Spot/Registration.lean:222-227` `declaredCode`). The challenger's ledger has an entry kind `declaration` (`spot/device/challenger.py:306`, checked by `_days` at `:1468`, read by `declared_day` at `:1659`), with format `verity/registration/declaration/v1` (`registration.py:71`, prose). The statement is `security/Properties/Spot.lean:71` (`DeclaredDay`). In Flock, `Flock/RegBound.lean` (25 lines) uses "declared" for the ports a window's registered interface declares (`declaredOf`): a separate sense. [read by me] |
| **day** | 220 (11) | 0 | 98 | 0 | 3 | 66 | 67 | **Parameter, check, ledger field** | Only SPOT. A lane's UTC day numbered from 1970 (`challenger.py:1259` `day_of`). The window's day is its open's `started_utc` (`challenger.py:1672`), passed to `registration.check` (`:238`). `Spot/ChallengerSession.lean` has 31 lines (`:28`, `:39-41` `verifyDays`, `declaredDay`), `Spot/OneStage.lean` 8, `security/Properties/Spot.lean` 3. [read by me] |
| **period** | 863 (57) | 20 | 134 | 0 | 257 | 335 | 129 | **Ledger entry, check, game** | The challenger's ledger has `period-open` and `period-close` entries (`challenger.py:304` `PERIOD_KINDS`; a period's open commits "its cap and circuit", `Spot/ChallengerSession.lean:27`). SPOT decides "a window counts toward its period only when it proves the circuit the period's open committed" (`challenger.py:1752` `circuit_reason`, `:1772` `credited_session`; `verify_periods` at `:1296`). The pool's `week` is "its index among the monitoring period's weeks on the verifier's clock" (`registration.py:116`; `Spot/Registration.lean:138` `week : Nat`). Security: `Proofs/Flock/EndToEnd/Period.lean`, `Period/Union.lean` (74 lines), `Period/Statement.lean`, `Carry.lean` (`Flock.SecurityProofs.IntegrityPeriod`, `Period.game`, `PeriodGood`, `Proofs/Flock/EndToEnd/Blocks.lean:58`). [read by me] |
| **window** | 3,210 (235) | 134 | 492 | 7 | 346 | 989 | 1,291 | **Record field, parameter, ledger key** | The record's `window` field is `{kind, stream, prover, seed_salt, pool_reset, pool_week, pool_spent, …provenance}` (`registration.py:9`, `:149-157`). R6 refuses a window seen before (`R6-window`, `registration.py:385`). The caller passes `window_root`, "from the challenger's signed open of the window" (`Spot/Registration.lean:195`). The challenger's window is `{root}`, "the served Commit's root" (`challenger.py:333` `WINDOW_FORM`), and the ledger gives each window one session. The ZK server's release schedule has "one window a statement" (`windows_ms`, `Firewall/Record.lean:14`, `:59`): a time slot, a separate sense. Security: `Proofs/Flock/Soundness/KeyedDraw.lean` (88 lines), `Soundness/Audit/Window.lean` (39), `Proofs/Flock/EndToEnd/Blocks.lean:60` (`WindowRec`). Imported from commitments as `verity.commitments.window` (`registration.py:138`). [read by me] |
| **served / serving** | 2,966 (401) | 420 | 166 | 11 | 700 | 961 | 727 | **Record field value, check** | `KINDS = ("served", "stand-in")` (`registration.py:149`; `Spot/Registration.lean:109`). R1 requires `window.kind` to be one of them (`security/Properties/Spot.lean:67`). `spot/verifier/one_stage/served.py` (239 lines) is the module for served windows. Statements: `security/Definitions/Claims/SamplingServed.lean` ("what a set of incorrect proof blocks serves, request by request", `:4-7`), `Proofs/Flock/SamplingGeneric/TwoServed.lean` (67 lines), `Proofs/Flock/Claims/Recursion/Serve.lean` (32). In Flock's prover, "a served row" is a row of a vLLM block (`verity_flock/class_statement.py`, 32 lines). [read by me] |
| **stand-in** | 152 (67) | 17 | 2 | 0 | 13 | 69 | 52 | **Record field value** | `KINDS` (above). Its docstring calls a stand-in "a fresh commitment to values captured from its run (… whose link to the run's root is provenance, not proof)" (`registration.py:45-47`). In Flock and security, "stand-in" mostly means a placeholder frame (`Flock/Firewall/Frame.lean`, `Proofs/Flock/Claims/ZK/Release.lean`): a separate sense. [read by me] |
| **request** | 2,061 (342) | 168 | 70 | 3 | 125 | 1,391 | 304 | **Statement parameter; otherwise a separate sense** | `Definitions/Claims/SamplingServed.lean:5-6`: "Under a split `req` of the served outputs into requests, request `r` serves `|r|` bits". The challenger's ledger `reserve` carries "its request" (`challenger.py:98`). The other code hits are requests to the ZK server or the challenger (`flock/agreement/firewall_agree.py:95`; `flock/prover/live/src/bin/flock-circuit.rs:236-247`; `Proofs/Flock/ZkAny/Relay.lean`, 41 lines). In prose, `parties.py:20` says the certifier "stamps each request's release order". [read by me] |
| **model** | 8,829 (1,205) | 107 | 33 | 0 | 6,232 | 2,353 | 112 | **Generated data in the verifier; otherwise noise** | `flock/verifier/VStar/Menu/Pinned.lean:9-…` ("F6's pinned menu of each model, by workload") names 9 vLLM workloads by model (`vllm/gemma2-2b`, `llama32-1b` ×2, `mistral-7b`, `olmoe-1b-7b` ×2, `qwen25-15b`, `qwen3-4b`, `qwen3-4b-fp8`; `rg -o '"vllm/[^"]+'`). In security, "model" is a probability model (`RunModel`, `innerModel`, `Definitions/Claims/RecursionGeneric/Comp.lean:185`): noise. In `spot/optimize.py` it is the cost model (`CostModel`). [read by me] |
| **weights** | 2,211 (482) | 144 | 35 | 0 | 748 | 1,203 | 82 | **Mostly noise; prose for registered weights** | Code hits are Lagrange and optimizer weights (`flock/prover/cuda/prove_chunk.cuh:69`; `spot/optimize.py:35`). The model's weights appear as prose: "the program's weights, constants and tables" (`registration.py:50-53`; `Flock/Registered.lean:3`) and `pinned`, "weights recorded before the salt" (`registration.py:54`). [read by me] |
| **checkpoint** | 13 (1) | 0 | 0 | 0 | 0 | 13 | 0 | Prose only | One security file. |
| **stream** | 13,277 (956) | 1,759 | 304 | 44 | 5,554 | 4,018 | 1,812 | **Record field, check; also the group's own senses** | The record's `binding.stream`, "the verifier's stream digest", held by R6 (`R6-stream`, `registration.py:79-81`, `:387-390`; `Spot/Registration.lean:190`). The ZK server opens each stream with a failure budget (`Firewall/Record.lean:16-20`; event `open` with `stream`, `verity_flock/firewall.py:220`). Coin and transcript streams are the proof system's own sense (`challenger.py:267` `_STREAM`, `TREE_SPEC_FORM` `streams`; `flock/prover/live/src/lib.rs`, 132 lines). `Spot/Stream.lean` (521 lines) is "a `Stream` of registrations, samples and outcomes" (`service.py:2`). [read by me] |
| **prover identity / binding** | binding 3,797 (732); `"prover"` and `lane` in `challenger.py` 50 lines | 301 | 107 | 13 | 1,633 | 1,413 | 350 | **Record field, check, ledger field** | The record's `binding: {"stream", "prover"}`, "the identity it registered the prover under", held by R6 (`R6-prover`, `registration.py:79-81`; `Spot/Registration.lean:190` `binding : Option (String × String)`). The ledger names the prover in a period's `lane` (`challenger.py:99-100`, `:1226` `lane_of`). Most "binding" hits are a commitment's binding, the proof system's own (`Proofs/Flock/Soundness/Discharge/FailClosed/One.lean`, 80 lines). [read by me] |
| **salt** | 13,873 (1,176) | 1,395 | 256 | 18 | 7,450 | 3,835 | 1,032 | **Parameter, record field, check; mostly hiding salts** | The window's `seed_salt` (`registration.py:151` `SEED_SALT`; `R6-seed-salt` at `:391-392`; `Spot/Registration.lean:203`), drawn by `coins.draw_seed_salt`, "a certified input the window's circuits read" (`registration.py:83-87`). Formats `verity/one-stage/window-salt/v1` and `supplied-salt/v1` (`Spot/Supplied.lean:22`). Most hits are hm96 hiding salts and the ZK server's value salts (`Flock/Firewall/Salts.lean:10`): the commitment's own sense. [read by me] |
| **pool, week** | `rg -c 'pool' spot/verifier/one_stage/registration.py` = 96 lines; `Spot/Registration.lean` `Pool` | | | | | | | **Parameter, record field, check** | The `pool` parameter `{pin, pout, genesis, lag, reset, verdicts, week}` (`registration.py:655` `POOL_KEYS`; `Spot/Registration.lean:131-139` `structure Pool`). Serving's reset is "at most one a week" (`R6-pool-reset-week`, `registration.py:403-404`), and "an engine crash needs" it (`:114`). `verdicts` is the verifier's own verdict log over earlier windows (`Spot/Registration.lean:144` `Pool.refused`). [read by me] |
| **pinned / held / supplied / outputs** | 5,990 (865) | 718 | 569 | 19 | 1,619 | 2,161 | 955 | **Parameters, check** | `Spot/Registration.lean:183-197` (`supplied` `:183`, `pinned` `:185`, `outputs` `:188`, `held` `:197`). `outputs` reuses "an output of an earlier run" by the log's record digest (`registration.py:88-94`). `pinned` is also the challenger host's pinned git commit (`challenger.py:151-170`, `pinned_session` `:1869`). In `Flock/HmRow.lean` (62 lines), "pinned" is the pinned circuit: the proof system's own sense. [read by me] |
| **PoUW** | 342 (75) | 20 | 4 | 0 | 14 | 92 | 212 | **Protocol ids; import (tests); prose** | `spot/verifier/parties.py:146-157` (`verity/pouw/v1`, `verity/pouw/period/v1`, `verity/pouw/pearl-plus/v1`). `challenger.py:200` (prose: "The first client is PoUW's live window"). `security/Proofs/Flock/SoundAtBridge/Union.lean` (8 lines) imports `Definitions.Accounting.TS.ProofLayer` (`:3`). Tests import PoUW (section 1b). `flock/sigma/pyproject.toml:14-19`. [read by me] |
| **PoUS, PoUB, wipe, bandwidth** | PoUS 12 (6), PoUB 2 (2), wipe 20 (6) | | | | | | | **Protocol ids** | `spot/verifier/parties.py:147-157`: `"pous": "verity/pous/p2/v4/run"`, `"wipe": "verity/wipe/v0"`, `"bandwidth": "verity/remote-bandwidth/p2/v0"` and `"verity/remote-bandwidth/v0"`. "wipe" in `flock/prover/cuda/prove_circuit.cuh` (7) is GPU memory: a separate sense. [read by me] |
| **certifier, network** | 71 (38); 67 (43) | 1; 5 | 2; 0 | 0 | 24; 26 | 25; 30 | 19; 6 | **Protocol id; Lean namespace; prose** | `parties.py:147`, `:155` (`verity/certifier/v0`). `Proofs/Flock/Bytes.lean:26`, `:72`: the namespaces `Properties.NetworkCertifier.Bytes` and `Properties.NetworkRecords.Bytes`, opened by 21 Flock proof files (section 1e). `registration.py:31-39`, `:126-127` (prose: R8's `tied` values "the network period check ties", and roots "the network certifier signed for the window"). `parties.py:19-21` (prose). In Flock's prover, "network" is the prover's network wait (`verity_flock/register.py:56`): a separate sense. [read by me] |
| **inference** | 65 (12) | 1 | 15 | 0 | 1 | 15 | 34 | **Party role, protocol id** | `parties.py:142` (`INFERENCE_SERVER = "inference-server"`, a role of the developer with `needs=("serve", "replay")` at `:194`), `:148-156` (`verity/inference-only/v0`). [read by me] |
| **remote, onsite** | 170 (36); 32 (4) | 14; 0 | 14; 1 | 0 | 0 | 42; 10 | 100; 21 | **Composite ids** | `parties.py:146-154` (`SUBPROTOCOLS`, keyed by `verity/onsite/v0` and `verity/remote/v1`). `spot/device/remote.py` (802 lines) is the remote challenger client (`:5`, `:64`: `verity/composite/device/live_challenger/serve.py`). [read by me] |
| **compose, composite** | 3,823 (760); 599 (128) | 103; 135 | 10; 4 | 0 | 2,540; 2 | 886; 194 | 285; 265 | **Import, constant; mostly the group's own senses** | `parties.py:123` (`from verity.configuration.deploy.config import COMPOSITES`), `:132-135` (`COMPOSED = "verity/sampled-proofs/v1"` at `:135`, "SPOT as both composites compose it" at `:132`). Most hits are proof composition (`VStar/Compose.lean`, `Proofs/Flock/Soundness/Compose.lean`, `Proofs/Flock/Claims/ZK/CausalExe.lean` 139 lines) and composite circuit nodes in Flock's prover (`flock/prover/live/src/circuit.rs`, 45): the group's own senses. [read by me] |
| **hardware** | 42 (25) | 12 | 3 | 1 | 0 | 17 | 9 | **Import; data read** | `challenger.py:220` (`verity.hardware.record`); `flock/pod/gemm_hill.py:46` (`benchmarks/census/hardware.json`); tests import `verity.hardware.site` (section 1b). [read by me] |
| **site** | 1,590 (133) | 33 | 23 | 0 | 912 | 567 | 56 | **Lean import, Lake exe; otherwise noise** | `Spot/SiteDifftest.lean:3` (`CoreProtocol.Site`) and its exe `site-difftest` (`spot/verifier/lean/lakefile.toml:60`); `spot/verifier/lean/tests/test_site_difftest.py:25`. In security, "site" is a gate or lookup site (`Proofs/Flock/Soundness/Discharge/HiddenExec/SiteH.lean`), and in Flock's prover a lookup site (`verity_flock/struct_lower.py:94` `class Site`): noise. [read by me] |
| **device** | 1,466 (198) | 354 | 10 | 1 | 50 | 912 | 141 | **The group's own senses** | GPU devices of Flock's prover (`flock/prover/live/src/bin/flock-circuit.rs`, 72 lines; `cuda/prove_chunk.cuh`, 34). `spot/device/` is SPOT's challenger directory. No code hit names an auditor's device other than the challenger. [read by me] |
| **gpu** | 1,111 (100) | 692 | 0 | 0 | 0 | 290 | 140 | The prover's own GPUs | `flock-circuit.rs` (195), `flock/pod/gemm_hill.py` (54), `flock-vllm-v1.rs` (34). |
| **memory** | 184 (52) | 37 | 0 | 0 | 0 | 121 | 27 | The prover's own GPU memory | `flock/prover/cuda_chunk_patch.py` (7). There are no hits for the memory protocols. |
| **account** | 1,538 (557) | 2 | 0 | 0 | 1,492 | 41 | 3 | **Noise (a name collision)** | `security/Definitions/Flock/Accounting/` is the soundness error's accounting (`namespace FlockSoundness.Accounting`, `Bound.lean:34`; "`repError` … `tableError` …", `:8-12`). The KS proofs' "account" hits are the same sense (`Proofs/Flock/Claims/KS/PlainLeafFork.lean`, 48). [read by me] |
| **vLLM** | 268 (89) | 110 | 2 | 0 | 0 | 65 | 91 | **Prover binaries, formats, generated verifier data** | `flock/prover/live/src/bin/flock-vllm-v1.rs` (35 lines; format `verity/flock-vllm-block/sigma/v1` at `:40`), `flock/prover/live/src/vllm_block.rs` (18; `verity/vllm-v1/domain/v1` at `:569`, `verity/fa2c/root/v0` at `:571`, spec `adapters/vllm/…/PROTOCOL.md` at `:2`). `flock/verifier/VStar/Menu/Pinned.lean` (9, see "model"). `flock/agreement/vectors.json:3323-3362` (`flock-vllm-block`, "out of scope"). [read by me] |
| **torch** | 56 (23) | 28 | 0 | 0 | 0 | 4 | 26 | **Import (pod script); a constant's name** | `flock/pod/gemm_slowdown.py:54` (`import torch`, 15 lines). `verity_flock/ir_lower.py:39` (`NAN_TORCH = 0x7FC0`, the bf16 NaN PyTorch writes). [read by me] |
| **adapter** | 25 (23) | 20 | 0 | 0 | 0 | 3 | 2 | Paths, pyproject inputs | `flock/prover/pyproject.toml` (3); pod scripts (`flock/pod/60-circuit.sh`, …). |
| **advice, kv cache** | 24 (9); 2 (2) | 1; 0 | 0 | 0 | 0 | 20; 2 | 3; 0 | Prose only | The one code hit is `madvise` (`flock/prover/live/src/huge_pages.rs`). |
| **epoch** | 332 (28) | 13 | 107 | 1 | 10 | 112 | 103 | **The challenger's own field** | A live coin commitment's `epoch` (`challenger.py:235` `COMMITMENT_FORM`; `Spot/Challenger.lean`, 31). It is the challenger's own sense. |
| **host** | 1,622 (165) | 479 | 105 | 1 | 15 | 551 | 502 | **Record field (challenger); otherwise GPU host** | The challenger's record `{commitment, host, signature}` names the auditor machine it runs on (`challenger.py:48`, `:244` `RECORD_FORM`; `Spot/ChallengerSession.lean`, 30). In Flock's prover, "host" is host memory (`flock/prover/live/src/gpu_circuit.rs`, 127). |
| **token** | 400 (77) | 116 | 44 | 0 | 48 | 92 | 103 | **The challenger's tokens; not LLM tokens** | `service.py:1011-1021` (`_token`, `_registration_token`); `Spot/Stream.lean:44` ("the challenger's tokens for them (the SHA-512 of a `Registration`'s fields"). In Flock's prover, `subcircuit_templates.py` (46) has its own sense. [read by me] |
| **batch, scheduler** | 8,630 (1,272) | 749 | 63 | 15 | 3,754 | 3,483 | 595 | **The group's own senses** | Proof batching and the ZK server's schedule (`flock/verifier/Flock/Firewall/Contract.lean`, 49; `Firewall/Record.lean:59` "A release schedule"). |
| **data sheet, card** | 5,715 (644) for `card`; 0 for "data sheet" | | | | | | | **Noise** | `card` is `Finset.card` (security 5,436 lines). `rg -ci 'data[ _-]?sheet\|datasheet'` over the group gives no hits. [counted by me] |

**SPOT's parties name the other protocols and both composites** [read by me]. `spot/verifier/parties.py:146-157` holds
`SUBPROTOCOLS`: for `verity/onsite/v0`, `bandwidth`, `certifier`, `inference-only`, `pous`, `pouw`, `sampled-proofs` and
`wipe`, each with its protocol id; for `verity/remote/v1`, `inference-only`, `pearl-plus` and `sampled-proofs`. It also
holds `IDS`, each subprotocol's id outside a composite. The comment at `:143-146` names `Core.Onsite.components` and
`Core.Remote.V1.components` as its Lean of record. `ids_of` (`:212`) returns the ids of a composite's names. The module
docstring (`:9-12`) says the core deployment "wraps SPOT[Flock] for both composites (remote and onsite)" and that its
verifier "composes each block's audit with the composite's other subprotocols".

---

## 3. Format strings and protocol ids (`verity/…/vN`) the group defines or reads [counted by me]

Command: `rg -o --no-filename '\bverity/[a-z0-9_./-]+/v[0-9]+[a-z0-9]*\b'` over the group's paths finds 143 distinct ids.
For each id, `git grep` over the whole tree counts the group's own files (code and tests apart) and every file outside the
group, by directory. The first table has the ids that name an upper layer or are read on both sides of the group's
boundary. The full list is in the appendix.

| Id | Group code files | Group test files | Files outside | Defined at | Read outside by |
| --- | --- | --- | --- | --- | --- |
| `verity/registration/v1` | 5 | 3 | 14 | `spot/verifier/one_stage/registration.py:145`; `flock/verifier/Flock/WindowRecord.lean:21` | `benchmarks/one_stage` 2, `benchmarks/prove_path` 2, `verity/protocols/pouw` 2, `adapters/vllm` 1, … |
| `verity/registration/v2` | 13 | 4 | 30 | `Flock/WindowRecord.lean:22`; `registration.py` (`FORMAT_V2`) | `verity/protocols/pouw` 8, `adapters/vllm` 5, `verity/protocols/inference_only` 4, `verity/commitments/windows`, … |
| `verity/registration/v3` | 0 | 1 | 1 | `spot/verifier/lean/tests/test_difftest.py:2112` | `verity/protocols/certifier` 1 |
| `verity/registration/declaration/v1` | 1 | 0 | 4 | `registration.py:71` (prose) | `verity/protocols/interface` 2, `adapters/vllm` 1, `verity/hardware/registration.py` 1 |
| `verity/registration-receipt/v1` | 3 | 1 | 4 | `registration.py:148`; `Spot/Run.lean:71` | `verity/protocols/pouw` 2, `verity/commitments/domains.py`, `verity/commitments/identity` |
| `verity/registration/produced/v0`, `/v1` | 2 each | | | `spot/verifier/one_stage/produced.py` | |
| `verity/sampled-proofs/one-stage/v1` | 4 | 1 | 21 | `registration.py:147`; `Spot/Registration.lean:108`; `Spot/Record.lean:448` | `benchmarks/one_stage` 8, `benchmarks/network_traces` 2, `verity/protocols/inference_only` 2, … |
| `verity/sampled-proofs/one-stage/v2` | 3 | 0 | 10 | `Spot/OneStageZk.lean:2` | `benchmarks/one_stage` 3, `verity/protocols/remote` 3, `verity/protocols/onsite` 2, … |
| `verity/sampled-proofs/v1` (`COMPOSED`) | 4 | 0 | 25 | `parties.py:135`; `recursion/verifier/recursive/VStarRecursive/Sampled.lean:2` | `benchmarks/one_stage` 7, `verity/protocols/remote` 5, `security/Properties` 3, `verity/protocols/onsite` 3, … |
| `verity/sampled-proofs/recursive/v1` | 5 | 0 | 4 | `security/Proofs/Flock/Recursive/AuditC3.lean:766` | `security/Proofs` 2, `security/lean-audit.json` 1, `verity/protocols/remote` 1 |
| `verity/sampled-proofs/spot/v1`, `…/spot/two-stage/v1` | 1 each | 0 | 0 | `parties.py:138`, `:140` | none |
| `verity/onsite/v0` | 1 | 1 | 25 | `parties.py:147` | `benchmarks/one_stage` 7, `demos/lifecycle` 3, `demos/onsite_served` 3, `verity/protocols/onsite` 3, … |
| `verity/remote/v1` | 2 | 1 | 17 | `parties.py:151` | `verity/protocols/remote` 5, `benchmarks/one_stage` 2, `demos/vllm` 2, `security/Properties` 2, … |
| `verity/pouw/v1`, `verity/pouw/period/v1`, `verity/pouw/pearl-plus/v1` | 1, 3, 3 | | 21, … | `parties.py:148-157` | `benchmarks/one_stage` 7, `verity/protocols/pouw` 3, `security/Properties` 2, `verity/protocols/interface` 2, … |
| `verity/pous/p2/v4` (`/run`) | 2 | | | `parties.py:148`, `:157` | |
| `verity/wipe/v0` | 1 | 0 | 46 | `parties.py:150` | `verity/protocols/pous` 12, `benchmarks/pous` 8, `benchmarks/one_stage` 6, `security/Properties` 5, … |
| `verity/certifier/v0` | 2 | 0 | 29 | `parties.py:147`, `:155` | `benchmarks/network_traces` 8, `verity/protocols/certifier` 7, `benchmarks/one_stage` 4, … |
| `verity/remote-bandwidth/v0`, `…/p2/v0` | 1, 1 | 0 | 24, 14 | `parties.py:155`, `:147` | `verity/protocols/poub` 10 and 4, `benchmarks/remote_session` 4, … |
| `verity/inference-only/v0` | 3 | | | `parties.py:148` | |
| `verity/challenger/coins/v1` | 4 | | | `challenger.py:227` (`BINDING`) | |
| `verity/challenger-record/v1`, `verity/challenger-close/v1`, `verity/challenger/commitment/v1`, `verity/challenger/release-log/v1` | 2 each | | | `challenger.py:239-253` | |
| `verity/challenger-auditor/v1`, `/v2` (`LEDGER_SCHEMA`), `…/entry/v1`, `/v2`, `verity/challenger-auditor-record/v1` | 3, 3, 2, 2, 2 | | | `challenger.py:301-303` | |
| `verity/window-circuit/v1` | 3 | 1 | 9 | `flock/verifier/Flock/RegBound.lean:62` | `benchmarks/one_stage` 2, `verity/protocols/certifier` 2, `benchmarks/prove_path` 1, … |
| `verity/one-stage/registered-domain/v1`, `served-domain/v1`, `window-salt/v1`, `supplied-salt/v1`, `history-table/v1`, `audit/v0` | 4, 2, 3, 2, 2, 3 | | 2, …, 2 | `flock/verifier/Flock/Registered.lean:29`; `Spot/Supplied.lean:22`; … | `verity/commitments/domains.py`, `verity/commitments/identity` |
| `verity/registered-reads/v0`, `/v1`, `/v2` | 5, 5, 2 | 6, 4 | 2, 10 | `flock/verifier/Flock/Registered.lean:23`, `:25` | `verity/protocols/certifier` 5, `benchmarks/one_stage` 1, `verity/commitments/*` |
| `verity/flock-vllm-block/v1`, `…/sigma/v1`, `verity/vllm-v1/domain/v1`, `verity/fa2c/root/v0` | 6, 1, 1, 1 | 1 | 0, 0, 4, 5 | `flock/prover/live/src/bin/flock-vllm-v1.rs:40`; `vllm_block.rs:569`, `:571` | `adapters/vllm` 2 and 3, `verity/commitments/domains.py`, `verity/commitments/identity` |
| `verity/vllm-vu-export/select/v1` | 0 | 15 | 2 | `flock/prover/tests/data/captured-101/manifests/…json:71` | `adapters/vllm` 2 |
| `verity/vllm/commit-coins/v1`, `verity/pouw/epoch-coins/live/v1`, `verity/memory/audit-coins/v1` | 0 | 1 each | 6, 3, 0 | `spot/verifier/lean/tests/test_challenger_difftest.py:79` (a list of coin labels) | `verity/protocols/pouw` 4, `adapters/vllm` 2 |
| `verity/pouw/closure-draw/v1`, `verity/pouw/noise-seed/v1` | 0 | 2, 1 | 2, 3 | `flock/prover/tests/fixtures/randomness_lean.json:9`, `:1090` | `verity/commitments/randomness`, `verity/commitments/domains.py` |
| `verity/integrity-profile/v1` | 3 | 1 | 2 | `spot/verifier/lean/Spot/Record.lean:49` | `verity/commitments/domains.py`, `verity/commitments/identity` |
| `verity/zk-server/registration/v1`, `value-salt/v1`, `value-commitment/v1`, `interior-key-tag/v1` | 2, 2, 3, 2 | | 0 | `flock/verifier/Flock/Firewall/Salts.lean:10`, `:104` | none |
| `verity/firewall-transcript/v0`, `firewall-public/v0`, `firewall-release/v0`, `firewall-coins/v0` | 3, 3, 4, 2 | | 2 | `verity_flock/firewall_transcript.py:1`, `:4` | `verity/commitments/domains.py`, `verity/commitments/identity` |
| `verity/proof-service/stream/v0`, `…/outcome/v0` | 3, 2 | | | `spot/verifier/service.py` | |
| `verity/partition/v1` | 24 | 11 | 36 | `security/Proofs/Flock/Claims/Recursion/SetupCount.lean:451` | `adapters/vllm` 11, `verity/circuits/tests` 6, `benchmarks/proof_throughput` 3, `verity/protocols/pouw` 2, … |
| `verity/circuit-type/v1` | 5 | 3 | 3 | `flock/prover/live/src/circuit.rs:3971` | `tools/verity`, `verity/commitments/domains.py`, `verity/commitments/identity` |
| `verity/f6-menu/v1` | 1 | 0 | 3 | `flock/verifier/VStar/Menu/Pinned.lean:7` | `benchmarks/numerical` 3 |

The file `firewall/v0` is the ZK server's record kind (`Firewall/Record.lean:54` `KIND`). It has no `verity/` prefix. [read by me]

---

## 4. Imports from outside into the group's modules that are not its public surface

**What I took as the public surface, and why.** The group declares no export list: `verity_flock/__init__.py` and
`spot/__init__.py` are empty, and the Lake packages set no `roots` or `globs`, so any module is importable. I therefore
took the surface each part names for its readers:

- **SPOT, Python.** The modules listed under "Core API" in `spot/verifier/README.md:245-318`: `service`; `one_stage`'s
  `registration`, `draw`, `partition`, `registered`, `served`, `audit`, `verdict`, `hello`, `verifier` and `proofs`;
  `parties`, `consumers`, `device/coins`, `optimize`, `candidates` and `costfit`. I added `verifier/profile.py`, which
  AGENTS.md names ("with the integrity profile (`profile`)") and the listing gives a place of its own. Not on the surface:
  `device/challenger.py`, `device/remote.py`, `plancheck.py`, `second_stage.py`, and `one_stage`'s `produced`,
  `closing`, `members` and `link_points`.
- **Flock, Python (`verity_flock`).** The README has a Layout and no API section. I took the modules it names as
  what other parts read: `circuit` ("Flock's circuit prover … module `verity_flock.circuit`"), `proof_server` ("SPOT's
  proof server"), `class_statement`, `typed_statement`, `circuit_types` ("the object the Boolean export, the docs site
  and the prover's layouts share"), `boolean_export`, `partition_units` and `instance_file`
  (`flock/prover/README.md:8-55`). Everything else, `ir_lower`, `gf2`, `tmpl` and `subcircuit_templates` among them, I
  counted as internal. This choice is mine; another reading of the README would move `ir_lower` onto the surface.
- **Lean.** Each `lean_lib`'s root module: `Flock`, `FlockZkServer`, `FlockRec`, `VStar` (`flock/verifier/lakefile.toml:22-35`),
  `Spot`, `Firewall`, `FirewallHonestDifftest`, `VStarRecursive` and `VStarTraceCli`. Separately, 16 modules under
  `Flock/` are one-line re-exports of the base packages (`Flock.Bytes`, `Canon`, `Draw`, `FV3`, `Hash`, `JsonParse`,
  `Program`, `Sizes`, `Annot`, `AnnotExpand`, `Evaluated`, `Extract`, `Field`, `Gate`, `Partition`, `Qword`), which
  the Flock README says stay "until every importer names" the base package (`flock/verifier/README.md:10-12`). An
  import of one is really a load of `verity/circuits/base`, `circuits` or `commitments`, not of Flock; I list them apart.
- **The group's own security areas** (section 0, plus `security/Proofs/Properties/{Flock,Challenger,Firewall,Spot}`)
  are not "outside" and are left out below.

### 4a. Python, from outside the group, by target module [counted by me]

Command: `rg -n -t py -e '^\s*(from|import)\s+(verity\.protocols\.(spot|flock|recursion)|verity_flock)\b'` over the
tree without the group (945 statements). A short script then resolves each `from X import name` to the module file it
names, and counts files by the importer's directory, tests apart (a file under `tests/` or named `test_*`).

| Target (not on the surface) | Code files | Test files | Code importers, by directory | What they take |
| --- | --- | --- | --- | --- |
| `spot/device/challenger.py` | 45 | 52 | `adapters/vllm` 13, `benchmarks/one_stage` 7, `benchmarks/pouw` 7, `verity/protocols/poub` 4, `verity/protocols/pouw` 4, `verity/composite` 3, `benchmarks/hash` 2, five more `benchmarks/` 1 each | The coin protocol's types (`LiveCommitment`, `LiveOpening`, `LiveRecord`, `ChallengerKey`, `CoinError`, `live_reason`, `Verify`), `coin_claim` (`pouw/verifier/coins.py:34`), and the auditor's ledger format (`DECLARATION`, `LEDGER_ROLE`, `verity/composite/device/live_challenger/auditor.py:319`) [read by me] |
| `spot/device/remote.py` | 20 | 18 | `adapters/vllm` 5, `benchmarks/one_stage` 3, `verity/composite` 3, `benchmarks/pouw` 2, `benchmarks/remote_session` 2, `verity/protocols/poub` 2, `verity/protocols/certifier` 1, two more `benchmarks/` | The TCP framing (`HEADER`, `MAX_HEAD`, `verity/protocols/certifier/verifier/coin_link.py:53`) and the client [read by me] |
| `spot/verifier/one_stage/produced.py` | 19 | 6 | `verity/protocols/certifier` 8, `benchmarks/pouw` 3, `verity/protocols/pouw` 3, `adapters/vllm` 2, `benchmarks/network_traces` 2, `benchmarks/one_stage` 1 | The produced-rows format (`verity/registration/produced/v0`, `/v1`). Importers: `certifier/verifier/{transfer_rows,served_rows,tie_service,charge_chain,io_tie,warden_rows,connection_rows,replay_chain}.py`, `pouw/verifier/{credit,leaf_lanes}.py`, `pouw/prover/circuit/h1window.py:97` [read by me] |
| `spot/verifier/second_stage.py` | 4 | 9 | `adapters/vllm` 2 (`pipeline/interior_links.py:38`, `pipeline/boundary_rows.py:91`), `benchmarks/one_stage` 1, `benchmarks/two_stage` 1 | Two-stage sampling's interior rows |
| `spot/plancheck.py` | 2 | 0 | `benchmarks/spot_figures` 2 | |
| `spot/verifier/one_stage/{closing,members,link_points}.py` | 1 each | 1 each | `benchmarks/one_stage` | |
| `verity_flock.ir_lower` | 18 | 16 | `benchmarks/one_stage` 6, `benchmarks/pouw` 5, `tools/verity` 3 (circuit-check: `structural.py`, `ambient.py`, `checks.py`), `adapters/vllm` 1 (`pipeline/chunk_rows.py:86`), four more `benchmarks/` | The generic Program lowering |
| `verity_flock.input_sets` | 11 | 3 | `benchmarks/one_stage` 8, `benchmarks/numerical` 2, `benchmarks/commitments` 1 | |
| `verity_flock.subcircuit_templates` | 9 | 6 | `benchmarks/numerical` 7, `tools/verity` 2 (`circuit_check/targets.py`, `sidecars.py`) | |
| `verity_flock.tmpl` | 8 | 1 | `benchmarks/one_stage` 2, `benchmarks/pouw` 2, `benchmarks/verifier_cost` 2, `tools/verity` 2 | |
| `verity_flock.gf2` | 2 | 8 | `benchmarks/pouw` 1, `tools/verity` 1; tests in `adapters/vllm` 3, `tools/verity` 3, `verity/definitions` 1 | |
| `verity_flock.rec_*` (`algebra`, `bindings`, `outer`, `residuals`, `vstage`, `vstar`) | 6 | 2 | `benchmarks/recursion_e2e` 4, `benchmarks/numerical` 2 | |
| 17 other `verity_flock` modules (`challengers`, `derive`, `fp`, `ir_sampling`, `layouts`, `lowering`, `lowerings`, `multi_unit`, `read_slots`, `read_tmpl`, `register`, `struct_lower`, `struct_slots`, `synthetic`, `tail`, `tail_pieces`, `topp_word`, `type_trace`) | 1-3 each | 0-2 each | `tools/verity` (circuit-check) 7 modules, `benchmarks/*`, `adapters/vllm` 2 (`read_slots`, `tail`, both in `pipeline/chunk_rows.py:86`) | |

Imports onto the surface, for scale: `one_stage.registered` 109 files (52 tests), `one_stage.registration` 108 (54),
`one_stage.partition` 95 (40), `verity_flock.class_statement` 72 (34), `one_stage.draw` 52 (27), `verity_flock.circuit`
48 (15), `service` 46 (27), `one_stage.proofs` 37 (20), `profile` 27 (11), `one_stage.verifier` 27 (9), `optimize` 22
(8), `device/coins` 15 (8) [counted by me].

What the listing says about these [read by me]. In `tests/import_order.toml`, the SPOT place's paths include
`device/challenger.py`, `device/remote.py` and `plancheck.py`, so the listing treats them as one place with the
verifier. Six places load it: `certifier/prover`, `certifier/verifier`, `poub/verifier`,
`poub/verifier/remote_session/verifier_process.py`, `pouw/prover` and `pouw/verifier`. Its `loaded_by` names, outside
the group, only `certifier/prover/window_charge.py` and `pouw/prover/circuit/h1window.py`. `verity/composite/device/live_challenger/`
is not in the listing; it is a distribution brought from `benchmarks/` (AGENTS.md). `adapters/` and `benchmarks/` are
outside `verity`'s places.

Three things the code does not judge [read by me]:

- **The challenger is a library for three other protocols and an adapter.** PoUW's verifier
  (`pouw/verifier/{window,credit,core_check,coins}.py`), PoUB's verifier (`poub/verifier/coins.py`,
  `session/challengers.py`, `session/verifier.py:142`), the certifier's coin link (`certifier/verifier/coin_link.py:52-53`),
  the composite live challenger and the vLLM adapter's PoUW path (`adapters/vllm/verity_vllm/protocols/pouw/h1/*.py`,
  `pearl_c/*.py`) each import its internal types directly. No module of SPOT's surface exposes them.
- **`produced.py` is a row format the certifier reads in eight files.** The certifier builds network rows
  (`warden_rows.py`, `connection_rows.py`, `transfer_rows.py`) on SPOT's produced-rows format.
- **Circuit-check imports Flock's lowering internals.** `tools/verity/circuit_check/src/circuit_check/` imports
  `ir_lower`, `subcircuit_templates`, `tmpl`, `struct_lower`, `tail_pieces`, `type_trace`, `topp_word`, `lowerings`,
  `ir_sampling` and `gf2` (`checks.py:435-1611`, `targets.py:284-1027`). The circuit language's own checker depends on the
  prover's code for what a lowering is.

### 4b. Lean, from outside the group, by target module [counted by me]

Command: `rg -n -g '*.lean' -e '^\s*(public\s+)?(meta\s+)?import\s+(all\s+)?(Flock|FlockFirewall|…|VStar\w*|Firewall\w*|Spot)(\.|\s|$)'`
over the tree without the group's directories and `.lake/` (754 lines, 439 files). Then I dropped the group's own
security areas: 137 imports in 81 files remain. Of these, 7 are of a root module, 42 (33 files) of a re-export shim and
88 (53 files) of an internal module.

| Importer | Files | Root | Shim | Internal | Internal modules imported |
| --- | --- | --- | --- | --- | --- |
| `verity/protocols/pouw` (Lean: `PouwCheck`, `PouwWindow`, `PouwCommit`, `PouwPearlPlus`) | 18 | 0 | 16 | 22 | `Flock.WindowRecord` ×3, `Flock.Hm96` ×3, `Flock.HmRow` ×3, `Spot.ChallengerSession` ×2, `Spot.Produced` ×2, `Flock.RegKnown`, `Flock.Registered`, `Flock.KnownRead`, `Flock.Norm`, `Spot.ChallengerCoins`, `Spot.Challenger`, `Spot.PyJson`, `Spot.Registered`, `Spot.Record` |
| `security/Proofs/Accounting` | 11 | 0 | 1 | 12 | `Flock.ReadNet.{Template ×2, Sha, Leaf, RowSeg, Select, Frame}`, `Flock.CoinTree`, `Flock.Hm96`, `Flock.RegKnown`, `Flock.KnownRead`, `Spot.Record` |
| `verity/protocols/certifier` (`NetworkCheck`, `CertifierDevice`) | 5 | 0 | 6 | 5 | `Flock.Public` ×2, `Flock.WindowRecord` ×2 (`NetworkCheck/Period.lean:8`, `:16`), `Flock.Hm96` |
| `verity/protocols/pous` (`PousCheck`, `MemoryChallenger`) | 7 | 0 | 8 | 2 | `Flock.HmRow` ×2 |
| `security/Proofs/Properties/Nci` | 10 | 0 | 4 | 6 | `Spot.WiresSound` ×3, `Spot.Bits`, `Spot.WiresSpan`, `Flock.Sigma` |
| `security/Proofs/Properties/Verity` | 7 | 1 | 0 | 6 | `VStarRecursive.Protocol` ×3, `Spot.OneStageZk` ×2, `Spot.ChallengerSession` |
| `verity/protocols/onsite` (`CoreOnsite.lean:39-41`, `:52-54`) | 1 | 0 | 0 | 6 | `Spot.OneStage`, `Spot.OneStageZk`, `VStarRecursive.Sampled` (each ×2) |
| `verity/protocols/remote` (`CoreRemote.lean:60-62`, `:69-71`) | 1 | 0 | 0 | 6 | the same three |
| `verity/protocols/inference_only` (`NciCheck/Record.lean:6-7`, `RecordText.lean:3-5`) | 2 | 0 | 0 | 5 | `Spot.Wires` ×2, `Flock.TemplateQuery` ×2, `Flock.Sigma` |
| `verity/protocols/poub` (`RemoteAudit`) | 4 | 0 | 3 | 2 | `Flock.Hm96` ×2 |
| `security/Properties/Verity` | 2 | 0 | 2 | 2 | `Flock.CoinTree`, `Flock.Hm96` |
| `security/Properties/Nci` | 1 | 0 | 0 | 3 | `Flock.TemplateQuery`, `Spot.Wires`, `Spot.Bits` |
| `benchmarks/verifier_cost/*` | 7 | 6 | 0 | 8 | `flock_tmpl`: `Flock.TmplSplit`, `NetAt`, `TmplAt`, `ProvedScope`, `KnownRead`, `With.Setup`, `Zk`; `table_root`: `Spot.HistoryTable` |
| `security/Proofs/Pouw`, `Proofs/Properties/{Remote,Pouw}` | 3 | 0 | 0 | 3 | `Spot.HiddenDraw`, `VStarRecursive.Sampled`, `Spot.Challenger` |
| `security/Properties/Annot.lean`, `security/Proofs/Core` | 2 | 0 | 2 | 0 | |

Lake packages outside the group that require the group's packages by path [read by me]: `security/Proofs` (`flock_verifier`,
`core_vstar`, `core_vstar_recursive`), `security/Properties` (`core_firewall`, `core_spot`, `flock_verifier`),
`remote/verifier` and `onsite/verifier` (`flock_verifier`, `core_spot`, `core_vstar_recursive`), `pouw/verifier/lean/window`
and `inference_only/verifier/nci/lean` (`core_spot`), and `pous/device/lean`, `pous/verifier/check`,
`pouw/verifier/lean/commit`, `certifier/verifier/lean` and `poub/verifier/lean` (`flock_verifier`). Command:
`rg -l -g lakefile.toml -e 'name = "(flock_verifier|core_spot|core_vstar_recursive|core_firewall|core_vstar)"'`.

What this shows [read by me]:

- **`Flock.WindowRecord` is the registration record's digest format.** It is read by PoUW's check (`PouwCheck/Inputs.lean:6`,
  `PouwPearlPlus/Tie.lean:5`) and the network check (`NetworkCheck/Period.lean:8`, `:16`), besides SPOT and RecVStar
  (`Flock/WindowRecord.lean:9-13`). A format of the registration layer lives in the proof system and is read across it.
- **PoUW's Lean reads SPOT's challenger internals.** `Spot.ChallengerSession` holds the challenger's signed close and the
  auditor's ledger, with its reserve, open, end, period and declaration entries (`Spot/ChallengerSession.lean:1-30`).
  PoUW's check imports it with `Spot.ChallengerCoins` and `Spot.Challenger` (`PouwCheck/Credit.lean:17-18`,
  `PouwCheck/Pass.lean:6`, `PouwWindow/Rounds.lean:4`). It opens the ledger's types (`Pass.lean:51`: `Audited`,
  `LedgerEntry`, `ledgerEntry?`) and reads a reserve's, an open's and an end's body fields itself (`Pass.lean:385-390`).
- **42 of the 137 imports are shims.** They load a base package under a Flock name, so they are not true dependencies
  on Flock. Renaming them would remove the edge from 33 files.

---

## 5. What the group's devices sign or store that is not a challenge, a commitment or a proof [read by me]

The group has two devices: SPOT's challenger (`spot/device/challenger.py`, 2,171 lines, with its service and client
`spot/device/remote.py` and Lean `Spot/ChallengerSession.lean`, 996 lines) and the ZK server (Lean
`recursion/verifier/firewall/Firewall/Record.lean`, 500 lines, and `flock/verifier/Flock/Firewall/`; Python
`verity_flock/firewall.py`). `spot/device/coins.py` stores nothing: it keeps the verifier's own randomness and its
segments in memory (`coins.py:201`, `:233`). The proof server (`verity_flock/proof_server.py`) is the prover's, not a
device.

### 5a. SPOT's challenger: what it signs (under its LMS key) and stores

| Item | Beyond coins and commitments, it holds | Who writes it | Where |
| --- | --- | --- | --- |
| Record (`record.json`) | `host`: "the auditor machine the challenger runs on", ASCII letters | the challenger, signed, before any coin | `challenger.py:44-52`, `:244`, `:618-652`; `remote.py:9-11` |
| Key (`key.json`) | The challenger's LMS key, with its leaf state, on its host | the challenger | `remote.py:9-11` |
| Release log (`releases.jsonl`) | Each received root and each release, with times in integer nanoseconds on its monotonic clock (`received_at`, `released_at`); a line per coin tree and per round opened | the challenger, fsynced before the opening leaves its host | `remote.py:32`, `:47-49`; `challenger.py:742-743`; `Spot/ChallengerSession.lean:5-7` |
| Close (`challenger.json`) | `{chain, ended, lines, session, signature}`: why the session ended (`ended`), how many lines | the challenger, signed | `challenger.py:64-71`, `:290`; `Spot/ChallengerSession.lean:9-11` |
| Session secret | A `secret` the auditor host's gate gives a window's opener; a hello without it is refused and counted (`refused_hellos`) | given by the auditor host's gate | `remote.py:17`, `:50`, `:347-353` |
| Withheld session | The commitment is kept on the host until close; each open answers the coin's raw bytes only | the challenger | `remote.py:35-43` |
| The auditor's ledger (`windows.jsonl`): `reserve` | `{by, [for], peer, scheme, epoch, families, minutes, started_utc, expires_utc}`: who reserved and for whom, its peer, start and expiry | the auditor (`verity/composite/device/live_challenger/auditor.py`), signed under the challenger's key | `challenger.py:95-104` |
| Ledger: `open` | `{window: {root}, host, pinned, key_sha512, by, [for], reserve, opened_by, started_utc, …}` and `pins: {lines, sha512}` | the auditor, signed | `challenger.py:105-106`, `:157` |
| Ledger: `end` | `{closed, files, record_leaf, commitment, refused_hellos, complete, …}`, with each file's SHA-512 | the auditor, signed | `challenger.py:107` |
| Ledger: `period-open`, `period-close` | `{period, lane, …}`: the prover by its `lane`; the open carries "its cap and circuit", the close lists its windows `{root, session, complete}` | the auditor, signed | `challenger.py:99-100`, `:108-109`; `Spot/ChallengerSession.lean:27-28` |
| Ledger: `declaration` | `{by, [for], blocks, day, declared_utc, pinned, program}`: the program a lane serves on a UTC day | the auditor, signed | `challenger.py:110`, `:136-143` |
| Pins log (on its host) | `{commit, main, pinned_utc}` per line: the git commit the host runs and the `origin/main` commit whose history held it | the auditor, sealed by each open | `challenger.py:151-158` |
| Run record (`RunRecord`) | A recursive run's `events` (each `Hello`, retries included) | the challenger | `challenger.py:75-93`, `:2085` |

What the challenger's code itself decides from these, beyond the coins [read by me]:

- **Who the prover is.** `lane_of` (`challenger.py:1226`): the open's `for`, else its `by` (`:115`). A ledger's prover is a
  string the auditor writes; the code does not judge who that is.
- **Periods.** `verify_periods` (`:1296`) checks that every period's close lists exactly its prover's windows,
  counted at their reserves (`:124-125`).
- **Days and declarations.** `_days` (`:1468`) checks that a lane declares each UTC day once, and that each later open
  of its windows is on a day it declared earlier, by the challenger's own clock (`:136-143`). `declared_day` (`:1659`)
  is what a per-day check reads.
- **Credit toward a period.** `credited_session` (`:1772`) counts a window toward its period only when the SHA-512 of
  the circuit descriptor equals the period-open's `circuit` (`circuit_reason`, `:1752`; prose `:145-149`).
- **That the host runs main's code.** `pinned_session` (`:1869`), `pins_reason` (`:1817`) and `first_parent_reason`
  (`:1852`) check that the open's pinned commit is in the pins log and on `origin/main`'s first-parent line in the
  verifier's own copy (`git rev-list --first-parent origin/main`, passed in as an argument). The code does not judge that
  the host ran that commit: "That the host ran the commit its open names is the auditor-trusted host's word
  (`trusted-host`)" (`:164-166`).
- **What the entropy is.** `coin_claim` (`:370-382`) names the coin source by a `uniform/<instance>` id from
  `verity.claims` (`claims.INSTANCES`), and refuses a misnamed one.

### 5b. The ZK server: what it stores

It signs nothing. `rg -i sign` over `Firewall/*.lean` and `Flock/Firewall/*.lean` finds one hit, the word "sign" in
prose at `Flock/Firewall/Hello.lean:87` [counted by me]. It stores `record.jsonl` (`firewall/v0`,
`Firewall/Record.lean:1-36`, `:54`) [read by me]:

| Event | Beyond the statements and proofs, it holds | Rule |
| --- | --- | --- |
| `open` | `{session, stream, parts}`: the stream a session belongs to; a failure `budget` b ("a public parameter of the deployment"); a release schedule `{start_ms, windows_ms}`, one window a statement | rules 1, 3, 4 (`:7-20`) |
| `release`, `admit`, `wait`, `stop` `{why}`, `finish`, `refused` | When each statement is released (`releaseMs` = `start_ms` plus the windows so far); a missed time is a stop | rules 1, 3 |
| `register` `{n, record}` | The value service's registrations, numbered 0, 1, 2 with no gap. An interior session's carries its own `key` and `key_tag`, "which are witness: they stay on the record, the ZK server's own" | rule 5 (`:21-28`) |
| `skip` | A registration no statement read; it counts as a failure on the stream | rules 4, 5 |
| `slot` | Who holds the one outer slot | rule 2 (`:12-13`) |

What the ZK server decides [read by me]: per stream, a failure budget and a release clock in milliseconds
(`start_ms`, `windows_ms`). The word "window" here is a release window, not the registration's window. The "stream" is
a ZK-server stream, not the registration's `binding.stream`. It stores interior sessions' keys (`key`, `key_tag`) as
witness on its own record.

---

## What I did not read

- `security/Proofs/Flock` (2,817 files, 856,206 lines): counted by `rg` and my scanner. I opened only the lines cited.
- `flock/prover/live/src/*.rs` and `flock/prover/cuda`: counted. I opened only the cited lines.
- The `.jsonl` and `.gz` fixtures, and JSON files over 200 kB, are not in the notion counts.
- The bodies of the 945 outside Python imports and 137 outside Lean imports in section 4: I resolved and counted them, and
  opened only the importers I cite.
- `verity/composite/device/live_challenger/auditor.py`, which writes the challenger's ledger: only its imports
  (`:319`, `:2252-2281`). It is outside the group.
- `Spot/ChallengerSession.lean` past its header (`:1-30`), and the Lean of the ZK server past `Firewall/Record.lean:1-60`.
- How each of the 143 ids in the appendix is used outside the group: counted by directory, not read, except the rows of
  section 3.
- Rust: whether any crate outside the group depends on `flock-live` (`rg -l -g Cargo.toml flock` outside the group finds
  none). The executables' names (`flock-verify`, `flock-circuit`, …) appear in 703 files outside the group, 2,164 lines;
  I did not sort those by sense.

## Appendix: all 143 ids [counted by me]

Generated by `/tmp/sweep/fmt.py` (in the VM, not committed): `git grep -I -n -o -E 'verity/[a-z0-9_./-]+/v[0-9]+[a-z0-9]*'`
over the whole tree, kept for the 143 ids the group's files name. "Group code" and "group tests" count the group's files
(section 0, a path with `/tests/` as a test). "First hit" is the group's first non-test, non-Markdown file in path
order, not necessarily where the id is defined. "Outside" names at most five directories, with file counts.

| Id | Group code | Group tests | Files outside | First hit in the group | Outside, by directory |
| --- | --- | --- | --- | --- | --- |
| `verity/blake3-leaf/v1` | 6 | 0 | 4 | `flock/prover/live/src/bin/flock-link.rs:70` | verity/commitments/domains.py(1); verity/commitments/frame_v3(1); verity/commitments/identity(1); verity/commitments/rowleaf.py(1) |
| `verity/certifier/v0` | 1 | 0 | 29 | `spot/verifier/parties.py:147` | benchmarks/network_traces(8); verity/protocols/certifier(7); benchmarks/one_stage(4); verity/protocols/onsite(3); tools/verity(2) |
| `verity/challenger-auditor-record/v1` | 2 | 0 | 2 | `spot/device/challenger.py:302` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/challenger-auditor/entry/v1` | 2 | 0 | 2 | `spot/device/challenger.py:303` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/challenger-auditor/entry/v2` | 2 | 0 | 2 | `spot/device/challenger.py:303` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/challenger-auditor/v1` | 2 | 0 | 5 | `spot/device/challenger.py:300` | verity/commitments/domains.py(1); verity/commitments/identity(1); verity/protocols/interface(1); verity/protocols/pous(1); verity/protocols/pouw(1) |
| `verity/challenger-auditor/v2` | 2 | 0 | 6 | `spot/device/challenger.py:301` | adapters/vllm(1); benchmarks/one_stage(1); verity/commitments/domains.py(1); verity/commitments/identity(1); verity/protocols/pous(1) |
| `verity/challenger-auditor/v3` | 0 | 1 | 0 | `spot/verifier/lean/tests/test_challenger_difftest.py:1154` | none |
| `verity/challenger-close/v1` | 2 | 0 | 2 | `spot/device/challenger.py:252` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/challenger-record/v1` | 2 | 0 | 2 | `spot/device/challenger.py:239` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/challenger/coins/v1` | 2 | 0 | 2 | `spot/device/challenger.py:12` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/challenger/commitment/v1` | 2 | 0 | 2 | `spot/device/challenger.py:240` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/challenger/release-log/v1` | 2 | 0 | 2 | `spot/device/challenger.py:253` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/circuit-type/v1` | 5 | 3 | 3 | `flock/prover/live/src/circuit.rs:3971` | tools/verity(1); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/coin-tree/v2` | 1 | 0 | 1 | `flock/prover/live/src/coin_tree.rs:548` | verity/commitments/tests(1) |
| `verity/f6-menu/v1` | 1 | 0 | 3 | `flock/verifier/VStar/Menu/Pinned.lean:7` | benchmarks/numerical(3) |
| `verity/fa2c/root/v0` | 1 | 0 | 5 | `flock/prover/live/src/vllm_block.rs:571` | adapters/vllm(3); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/firewall-coins/v0` | 2 | 0 | 2 | `flock/prover/python/verity_flock/rec_live.py:150` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/firewall-public/v0` | 3 | 0 | 2 | `flock/prover/python/verity_flock/firewall_transcript.py:4` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/firewall-public/v1` | 0 | 1 | 0 | `recursion/verifier/firewall/tests/test_firewall_release_protocol.py:85` | none |
| `verity/firewall-release/v0` | 1 | 1 | 2 | `recursion/verifier/firewall/Firewall/Protocol.lean:2` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/firewall-transcript/v0` | 3 | 1 | 2 | `flock/prover/python/verity_flock/firewall_transcript.py:1` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/flock-chunk-block/v1` | 1 | 0 | 0 | `flock/prover/live/src/chunk.rs:52` | none |
| `verity/flock-circuit/binding/v2` | 1 | 0 | 3 | `flock/prover/python/verity_flock/circuit.py:85` | benchmarks/one_stage(2); benchmarks/network_traces(1) |
| `verity/flock-circuit/coins/v1` | 1 | 0 | 0 | `flock/prover/live/src/coin_seed.rs:12` | none |
| `verity/flock-circuit/stage-salt/v1` | 1 | 0 | 0 | `flock/prover/python/verity_flock/circuit.py:1093` | none |
| `verity/flock-glue/v1` | 1 | 0 | 0 | `flock/prover/live/src/glue.rs:18` | none |
| `verity/flock-ir-block/sigma/v1` | 1 | 0 | 0 | `flock/prover/live/src/bin/flock-ir-block.rs:37` | none |
| `verity/flock-ir-block/v1` | 5 | 1 | 0 | `flock/prover/live/src/bin/flock-ir-block.rs:1` | none |
| `verity/flock-ir-frame/binding/v1` | 1 | 0 | 2 | `flock/prover/python/verity_flock/ir_frame.py:34` | benchmarks/one_stage(2) |
| `verity/flock-ir-frame/sigma/v2` | 1 | 0 | 0 | `flock/prover/live/src/bin/flock-ir-sampling.rs:79` | none |
| `verity/flock-ir-frame/sigma/v3` | 1 | 0 | 0 | `flock/prover/live/src/bin/flock-ir-frame.rs:43` | none |
| `verity/flock-ir-frame/v1` | 1 | 0 | 1 | `flock/prover/python/verity_flock/ir_bench.py:1` | benchmarks/numerical(1) |
| `verity/flock-ir-frame/v2` | 4 | 1 | 0 | `flock/agreement/vectors.json:1600` | none |
| `verity/flock-ir-frame/v3` | 17 | 2 | 1 | `flock/agreement/vectors.json:2025` | benchmarks/numerical(1) |
| `verity/flock-ir-sampling/sigma/v1` | 1 | 0 | 0 | `flock/prover/live/src/bin/flock-ir-sampling.rs:79` | none |
| `verity/flock-ir-sampling/v1` | 9 | 1 | 1 | `flock/agreement/vectors.json:3225` | benchmarks/numerical(1) |
| `verity/flock-links/v1` | 2 | 0 | 0 | `flock/prover/live/src/circuit.rs:2180` | none |
| `verity/flock-netlist/sigma/v1` | 1 | 0 | 2 | `flock/verifier/Flock/Tags.lean:113` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/flock-netlist/v1` | 6 | 1 | 4 | `flock/agreement/agree.py:151` | tests/lint(1); tests/test_retired_terms.py(1); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/flock-once/v1` | 0 | 1 | 0 | `flock/prover/tests/keyed.py:31` | none |
| `verity/flock-public-inputs/v1` | 2 | 1 | 6 | `flock/prover/python/verity_flock/circuit.py:1159` | verity/protocols/pouw(4); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/flock-pure-block/sigma/v2` | 1 | 0 | 0 | `flock/prover/live/src/bin/flock-pure-gpu.rs:36` | none |
| `verity/flock-pure-block/v1` | 4 | 0 | 0 | `flock/prover/cuda/prove_chunk.cuh:547` | none |
| `verity/flock-pure-block/v2` | 6 | 1 | 7 | `flock/agreement/vectors.json:314` | benchmarks/numerical(7) |
| `verity/flock-pure/v1` | 2 | 0 | 0 | `flock/prover/live/src/bin/flock-pure.rs:3` | none |
| `verity/flock-vllm-block/sigma/v1` | 1 | 0 | 0 | `flock/prover/live/src/bin/flock-vllm-v1.rs:40` | none |
| `verity/flock-vllm-block/v1` | 6 | 1 | 0 | `flock/agreement/vectors.json:3362` | none |
| `verity/flock-zk/tau/v1` | 3 | 0 | 2 | `flock/prover/live/src/zk_veil.rs:976` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/flock/partition-units/v0` | 1 | 0 | 0 | `flock/prover/python/verity_flock/partition_units.py:45` | none |
| `verity/flock/tree-from/v1` | 1 | 0 | 0 | `flock/prover/pod/challenger.sh:54` | none |
| `verity/hillclimb-point/v0` | 2 | 0 | 0 | `flock/pod/74-gemm-hill.sh:10` | none |
| `verity/hillclimb-rollup/v0` | 1 | 0 | 0 | `flock/pod/gemm_hill.py:93` | none |
| `verity/hm96-sha256/key/v1` | 1 | 0 | 4 | `flock/verifier/Flock/Hm96.lean:82` | verity/commitments/hm96(2); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/hm96-sha256/leaf/v1` | 1 | 0 | 5 | `flock/verifier/Flock/Hm96.lean:81` | verity/commitments/hm96(2); adapters/vllm(1); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/hm96-sha256/salt/v1` | 1 | 0 | 4 | `flock/verifier/Flock/Hm96.lean:81` | verity/commitments/hm96(2); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/hm96-sha512/key/v1` | 3 | 0 | 5 | `flock/prover/flock-sha512-b684b12.patch:1321` | verity/commitments/hm96(2); verity/commitments/domains.py(1); verity/commitments/identity(1); verity/commitments/tests(1) |
| `verity/hm96-sha512/leaf/v1` | 4 | 0 | 9 | `security/Proofs/Flock/Bridge/Spec/Row.lean:57` | security/Proofs(2); verity/commitments/hm96(2); security/Definitions(1); verity/commitments/coin_tree.py(1); verity/commitments/domains.py(1) |
| `verity/hm96-sha512/salt/v1` | 5 | 0 | 9 | `security/Proofs/Flock/Bridge/Spec/Row.lean:55` | security/Proofs(2); verity/commitments/hm96(2); security/Definitions(1); verity/commitments/coin_tree.py(1); verity/commitments/domains.py(1) |
| `verity/inference-only/v0` | 1 | 0 | 33 | `spot/verifier/parties.py:148` | benchmarks/one_stage(7); verity/protocols/remote(6); verity/protocols/inference_only(5); security/Properties(3); demos/lifecycle(2) |
| `verity/input-sets/synthetic/v1` | 1 | 17 | 0 | `flock/prover/python/verity_flock/synthetic.py:28` | none |
| `verity/integrity-profile/v1` | 3 | 1 | 2 | `spot/verifier/lean/Spot/Record.lean:49` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/layout/v1` | 3 | 1 | 2 | `flock/prover/live/src/typed.rs:34` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/ligero-b/auth-binding/v2h` | 1 | 0 | 0 | `flock/prover/python/verity_flock/instance_file.py:25` | none |
| `verity/lms/v0` | 0 | 1 | 7 | `spot/verifier/lean/tests/test_challenger_difftest.py:408` | verity/commitments/lms(2); security/Proofs(1); verity/commitments/crypto(1); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/memory/audit-coins/v1` | 0 | 1 | 0 | `spot/verifier/lean/tests/test_challenger_difftest.py:79` | none |
| `verity/one-stage/audit/v0` | 3 | 0 | 6 | `spot/verifier/lean/Spot/Record.lean:48` | benchmarks/one_stage(3); benchmarks/pouw(1); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/one-stage/history-table/v1` | 1 | 0 | 1 | `spot/verifier/lean/Spot/HistoryTable.lean:10` | verity/commitments/history.py(1) |
| `verity/one-stage/registered-domain/v1` | 4 | 0 | 2 | `flock/verifier/Flock/Registered.lean:29` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/one-stage/served-domain/v1` | 1 | 1 | 6 | `spot/verifier/lean/Spot/SecondStage.lean:747` | adapters/vllm(1); verity/commitments/domains.py(1); verity/commitments/identity(1); verity/commitments/tests(1); verity/commitments/window.py(1) |
| `verity/one-stage/supplied-salt/v1` | 2 | 1 | 2 | `spot/verifier/lean/Spot/Supplied.lean:22` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/one-stage/window-salt/v1` | 2 | 1 | 2 | `spot/verifier/lean/Spot/Supplied.lean:40` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/onsite/v0` | 1 | 1 | 25 | `spot/verifier/parties.py:147` | benchmarks/one_stage(7); demos/lifecycle(3); demos/onsite_served(3); verity/protocols/onsite(3); verity/protocols/remote(2) |
| `verity/partition/v1` | 24 | 11 | 36 | `security/Proofs/Flock/Claims/Recursion/SetupCount.lean:451` | adapters/vllm(11); verity/circuits/tests(6); benchmarks/proof_throughput(3); verity/protocols/pouw(2); ARCHITECTURE.md(1) |
| `verity/pos-leaf/v0` | 1 | 0 | 8 | `flock/prover/live/src/vllm_block.rs:7` | adapters/vllm(3); verity/commitments/hm96(2); benchmarks/commitments(1); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/pous/p2/v4` | 1 | 0 | 25 | `spot/verifier/parties.py:148` | verity/protocols/pous(6); benchmarks/one_stage(5); benchmarks/pous(3); demos/onsite_served(2); security/Properties(2) |
| `verity/pouw/closure-draw/v1` | 0 | 2 | 2 | `flock/prover/tests/fixtures/randomness_lean.json:9` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/pouw/epoch-coins/live/v1` | 0 | 1 | 3 | `spot/verifier/lean/tests/test_challenger_difftest.py:79` | adapters/vllm(1); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/pouw/noise-seed/v1` | 0 | 1 | 3 | `flock/prover/tests/fixtures/randomness_lean.json:1090` | verity/commitments/randomness(3) |
| `verity/pouw/pearl-plus/v1` | 2 | 0 | 26 | `spot/verifier/parties.py:151` | benchmarks/one_stage(7); benchmarks/pouw(5); verity/protocols/pouw(4); benchmarks/compute(3); verity/protocols/remote(3) |
| `verity/pouw/pearl-plus/y-rows/v1` | 0 | 2 | 2 | `spot/verifier/lean/tests/test_difftest.py:6164` | adapters/vllm(1); verity/protocols/pouw(1) |
| `verity/pouw/period/v1` | 2 | 1 | 31 | `spot/verifier/parties.py:149` | benchmarks/one_stage(9); verity/protocols/pouw(7); verity/protocols/remote(4); verity/protocols/onsite(3); demos/lifecycle(2) |
| `verity/pouw/v1` | 1 | 0 | 21 | `spot/verifier/parties.py:157` | benchmarks/one_stage(7); verity/protocols/pouw(3); security/Properties(2); verity/protocols/interface(2); verity/protocols/remote(2) |
| `verity/proof-service/outcome/v0` | 1 | 0 | 3 | `spot/verifier/service.py:148` | benchmarks/one_stage(1); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/proof-service/stream/v0` | 2 | 0 | 3 | `spot/verifier/service.py:144` | tests/test_retired_terms.py(1); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/randomness/derive/v2` | 1 | 0 | 13 | `flock/prover/live/src/coin_seed.rs:13` | benchmarks/pous(4); verity/commitments/randomness(3); verity/protocols/pous(2); benchmarks/codec_tensor(1); benchmarks/remote_session(1) |
| `verity/randomness/stream/v2` | 1 | 0 | 7 | `flock/prover/live/src/coin_seed.rs:14` | verity/commitments/randomness(3); benchmarks/codec_tensor(1); benchmarks/remote_session(1); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/rec-coins/v1` | 1 | 0 | 2 | `flock/verifier/Flock/RecCoins.lean:550` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/rec-inner/v1` | 3 | 2 | 2 | `security/Proofs/Flock/VBridge/Family/Join.lean:334` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/rec-inner/v2` | 3 | 6 | 0 | `flock/verifier/Flock/RecVStar.lean:121` | none |
| `verity/rec-link/v1` | 1 | 0 | 0 | `flock/verifier/Flock/RecLinkRef.lean:67` | none |
| `verity/rec-links-vectors/v1` | 2 | 0 | 0 | `flock/prover/python/verity_flock/rec_links.py:35` | none |
| `verity/rec-links/v1` | 3 | 0 | 0 | `security/Definitions/Claims/Recursion.lean:342` | none |
| `verity/rec-run/v0` | 1 | 1 | 0 | `flock/verifier/Flock/Test/RecCompose.lean:192` | none |
| `verity/rec-run/v1` | 9 | 1 | 0 | `flock/pod/85-rec-reprice.sh:954` | none |
| `verity/rec-sparse/v1` | 1 | 0 | 0 | `flock/prover/python/verity_flock/rec_session.py:61` | none |
| `verity/rec-v0/v-star/v0` | 1 | 0 | 0 | `flock/prover/python/verity_flock/rec_outer.py:74` | none |
| `verity/rec-v1/verifier-salt/v1` | 2 | 0 | 2 | `flock/prover/python/verity_flock/rec_residuals.py:80` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/rec-v1/zero-residual-salt/v1` | 2 | 0 | 2 | `flock/prover/python/verity_flock/rec_residuals.py:79` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/registered-reads/v0` | 5 | 6 | 2 | `flock/agreement/agree.py:241` | verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/registered-reads/v1` | 5 | 4 | 10 | `flock/verifier/Flock/Registered.lean:23` | verity/protocols/certifier(5); benchmarks/one_stage(1); verity/commitments/domains.py(1); verity/commitments/identity(1); verity/commitments/windows(1) |
| `verity/registered-reads/v2` | 1 | 1 | 0 | `flock/verifier/Flock/Test/RegFormats.lean:153` | none |
| `verity/registration-receipt/v1` | 3 | 1 | 4 | `spot/verifier/lean/Spot/Run.lean:71` | verity/protocols/pouw(2); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/registration/declaration/v1` | 1 | 0 | 4 | `spot/verifier/one_stage/registration.py:71` | verity/protocols/interface(2); adapters/vllm(1); verity/hardware/registration.py(1) |
| `verity/registration/produced/v0` | 0 | 2 | 0 | `spot/verifier/lean/tests/test_difftest.py:3200` | none |
| `verity/registration/produced/v1` | 2 | 0 | 6 | `spot/verifier/lean/Spot/Produced.lean:37` | verity/protocols/pouw(4); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/registration/v0` | 0 | 2 | 2 | `spot/verifier/lean/tests/test_difftest.py:1970` | benchmarks/one_stage(1); verity/protocols/pouw(1) |
| `verity/registration/v1` | 5 | 3 | 14 | `flock/verifier/Flock/WindowRecord.lean:10` | benchmarks/one_stage(2); benchmarks/prove_path(2); verity/protocols/pouw(2); adapters/vllm(1); benchmarks/pouw(1) |
| `verity/registration/v2` | 13 | 4 | 30 | `flock/pod/88-rec-c3-inner.sh:37` | verity/protocols/pouw(8); adapters/vllm(5); verity/protocols/inference_only(4); verity/commitments/windows(3); benchmarks/one_stage(2) |
| `verity/registration/v3` | 0 | 1 | 1 | `spot/verifier/lean/tests/test_difftest.py:2112` | verity/protocols/certifier(1) |
| `verity/remote-bandwidth/p2/v0` | 1 | 0 | 14 | `spot/verifier/parties.py:147` | verity/protocols/poub(4); verity/protocols/remote(3); benchmarks/one_stage(1); benchmarks/pous(1); demos/lifecycle(1) |
| `verity/remote-bandwidth/v0` | 1 | 0 | 24 | `spot/verifier/parties.py:155` | verity/protocols/poub(10); benchmarks/remote_session(4); benchmarks/one_stage(3); verity/protocols/remote(3); demos/vllm(1) |
| `verity/remote/v0` | 1 | 1 | 28 | `spot/verifier/README.md:13` | benchmarks/one_stage(7); verity/protocols/remote(6); security/Properties(4); demos/lifecycle(2); security/Proofs(2) |
| `verity/remote/v1` | 2 | 1 | 17 | `spot/verifier/parties.py:151` | verity/protocols/remote(5); benchmarks/one_stage(2); demos/vllm(2); security/Properties(2); demos/README.md(1) |
| `verity/rope/neox-bf16/cos-sin/v0` | 0 | 1 | 0 | `flock/prover/tests/test_lean_public_inputs.py:170` | none |
| `verity/rope/neox-bf16/cos-sin/v1` | 0 | 2 | 1 | `flock/prover/tests/test_circuit.py:239` | verity/circuits/tests(1) |
| `verity/sampled-proofs/one-stage/v1` | 4 | 1 | 21 | `spot/verifier/lean/Spot/Record.lean:448` | benchmarks/one_stage(8); benchmarks/network_traces(2); verity/protocols/inference_only(2); verity/protocols/onsite(2); verity/protocols/pouw(2) |
| `verity/sampled-proofs/one-stage/v2` | 3 | 0 | 10 | `spot/verifier/lean/Spot/OneStageZk.lean:2` | benchmarks/one_stage(3); verity/protocols/remote(3); verity/protocols/onsite(2); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/sampled-proofs/recursive/v1` | 5 | 0 | 4 | `security/Proofs/Flock/Recursive/AuditC3.lean:766` | security/Proofs(2); security/lean-audit.json(1); verity/protocols/remote(1) |
| `verity/sampled-proofs/ru-sample/v1` | 0 | 2 | 3 | `flock/prover/tests/fixtures/randomness_lean.json:536` | verity/commitments/randomness(3) |
| `verity/sampled-proofs/spot/two-stage/v1` | 1 | 0 | 0 | `spot/verifier/parties.py:140` | none |
| `verity/sampled-proofs/spot/v1` | 1 | 0 | 0 | `spot/verifier/parties.py:138` | none |
| `verity/sampled-proofs/two-stage/v1` | 0 | 1 | 1 | `spot/verifier/lean/tests/test_difftest.py:1971` | verity/protocols/certifier(1) |
| `verity/sampled-proofs/v1` | 4 | 0 | 25 | `recursion/verifier/recursive/VStarRecursive/Sampled.lean:2` | benchmarks/one_stage(7); verity/protocols/remote(5); security/Properties(3); verity/protocols/onsite(3); demos/lifecycle(2) |
| `verity/sampled-proofs/vu-sample/v1` | 1 | 0 | 4 | `spot/device/coins.py:16` | adapters/vllm(3); GLOSSARY.md(1) |
| `verity/sha256-row-nvfp4/v1` | 1 | 0 | 4 | `flock/prover/live/src/pure_block.rs:195` | verity/commitments/domains.py(1); verity/commitments/frame_v3(1); verity/commitments/identity(1); verity/commitments/rowleaf.py(1) |
| `verity/sha256-row/v1` | 1 | 0 | 4 | `flock/prover/live/src/pure_block.rs:180` | verity/commitments/domains.py(1); verity/commitments/frame_v3(1); verity/commitments/identity(1); verity/commitments/rowleaf.py(1) |
| `verity/sha512-row-seg/v1` | 5 | 0 | 14 | `security/Proofs/Flock/Bridge/Spec/RowSeg.lean:32` | security/Proofs(3); security/Definitions(2); verity/protocols/pouw(2); adapters/vllm(1); benchmarks/pouw(1) |
| `verity/sha512-row/v1` | 9 | 0 | 5 | `security/Proofs/Flock/Claims/Recursion/LayoutBTops.lean:169` | security/Definitions(1); verity/commitments/domains.py(1); verity/commitments/frame_v3(1); verity/commitments/identity(1); verity/commitments/rowleaf.py(1) |
| `verity/sha512-row/v2` | 6 | 0 | 12 | `security/Proofs/Flock/Bridge/Spec/Row.lean:39` | adapters/vllm(1); benchmarks/torch_gpu(1); benchmarks/word_codec(1); security/Definitions(1); verity/commitments/domains.py(1) |
| `verity/tagged-sha512/v1` | 3 | 0 | 14 | `security/Properties/Challenger.lean:190` | verity/commitments/identity(2); verity/protocols/pous(2); adapters/vllm(1); benchmarks/pous(1); security/Definitions(1) |
| `verity/test/coins/v1` | 0 | 7 | 7 | `spot/verifier/tests/test_challenger.py:33` | verity/composite/device(5); benchmarks/prove_path(1); verity/protocols/certifier(1) |
| `verity/test/kat/v1` | 0 | 1 | 2 | `flock/prover/tests/fixtures/randomness_lean.json:826` | verity/commitments/randomness(2) |
| `verity/vllm-v1/domain/v1` | 1 | 0 | 4 | `flock/prover/live/src/vllm_block.rs:569` | adapters/vllm(2); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/vllm-vu-export/select/v1` | 0 | 15 | 2 | `flock/prover/tests/data/captured-101/manifests/attention-head-fa2-d64-bn128.json:71` | adapters/vllm(2) |
| `verity/vllm/commit-coins/v1` | 0 | 1 | 6 | `spot/verifier/lean/tests/test_challenger_difftest.py:79` | verity/protocols/pouw(4); adapters/vllm(2) |
| `verity/window-circuit/v0` | 1 | 0 | 1 | `flock/verifier/Flock/Test/RegBound.lean:166` | verity/protocols/certifier(1) |
| `verity/window-circuit/v1` | 3 | 1 | 9 | `flock/verifier/Flock/RegBound.lean:62` | benchmarks/one_stage(2); verity/protocols/certifier(2); benchmarks/prove_path(1); verity/commitments/domains.py(1); verity/commitments/identity(1) |
| `verity/wipe/v0` | 1 | 0 | 46 | `spot/verifier/parties.py:150` | verity/protocols/pous(12); benchmarks/pous(8); benchmarks/one_stage(6); security/Properties(5); verity/protocols/onsite(3) |
| `verity/zk-server/interior-key-tag/v1` | 2 | 1 | 0 | `flock/verifier/Flock/Firewall/Salts.lean:104` | none |
| `verity/zk-server/registration/v1` | 1 | 1 | 0 | `flock/verifier/Flock/Firewall/Salts.lean:263` | none |
| `verity/zk-server/value-commitment/v1` | 1 | 2 | 0 | `flock/verifier/Flock/Firewall/Salts.lean:103` | none |
| `verity/zk-server/value-salt/v1` | 2 | 1 | 0 | `flock/verifier/Flock/Firewall/Salts.lean:10` | none |
