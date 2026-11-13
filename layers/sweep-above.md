# SWEEP-ABOVE: the layers above the proofs, as `main` holds them

**Status: all six items. Final.**

- Commit of `main` read: `9182900fb94eefbc4a6cc5d6e14885dc6b18f5ff` ("Queue: #9668 Retired terms: the PoUW verifier's
  six stale entries…", committed 2026-11-13 05:59 UTC). Every path below is as that commit holds it.
- Brief: `inbox/LAYERS.md` on `cursor/verity-top-4db4`, section 4, part SWEEP-ABOVE. Layers named as in
  `inbox/ARCHFILE-page.md` (cited "page §N").
- Groups used in every row: **proofs** = `verity/protocols/{flock,spot,recursion}`; **accounting** =
  `verity/protocols/{pouw,pous,poub,certifier}`; **policy** = `verity/protocols/inference_only`; **compose** =
  `verity/protocols/{remote,onsite}` (with `remote/verifier/composition/`) and `verity/composite/`; **interface** =
  `verity/protocols/interface`; **security** = `security/`.
- Tags: [read by me] I opened the file at that line; [counted by me] a count, with the command beside it.
- A row says what the code does. It does not judge.

## Ten rows to look at first

Each is a row below, by section.

1. 2a, 2b: the certifier's Lean requires the policy's `nci_check` (`certifier/verifier/lean/lakefile.toml:16`) and imports
   `NciCheck.Desc`, `NciCheck.DefinitionDigest` (`NetworkCheck/Period.lean:9,10`); the page draws no accounting → policy arrow.
2. 5b: `security/Definitions/Core/{Site,Blocks,Accel/Card}.lean` import `Definitions.Accounting.Device` (the layers allow it).
3. 5b: accounting's guarantees read the policy's definitions: `Definitions/Accounting/Guarantees/Actual.lean:5,6`,
   `Actual/Dampener.lean:4`, `Actual/Threaded.lean:4,5`.
4. 3b: PoUW's Lean verifier embeds 7 model checkpoints' SHA-512 pins (`PouwCheck/Credit.lean:1634`, `models.json`).
5. 3b: vLLM's scheduler step, pinned to a vLLM build, is a Definition in the certifier's verifier (`certifier/verifier/replay_step.py:1-15`).
6. 3b, 3c: `verity/commitments/domains.py` holds 14 `VLLM_*` constants (183-196, 210, 397-400) and sections per protocol.
7. 3a: PoUS, PoUB and SPOT import the runtime (`runtime.ProtocolDef`, `stream.Side`), e.g. `poub/verifier/session_parties.py:47`;
   the shared `config.py:59-61` names each composite's protocols.
8. 4a: a device's data sheet is defined four ways: `security/…/Accel/Card.lean:31`, `…/Accounting/TS/Sheet.lean:68`
   (hand-written instances), `CoreProtocol/Site.lean:150` with its Python mirror, and `pouw/prover/circuit/h1device.py` literals.
9. 4d: the week/month/year list twice, `verity/periods/__init__.py:29` (read by its own test only) and `verity/protocols/reporting.py:11`.
10. 6: PoUS's verifier keeps a second claims table (`pous/verifier/protocol.py:45`, 9 entries), and PoUW's provers write claim
    ids as literals without importing `verity.claims` (`pouw/prover/schemes/pearl_c.py:526` and 3 more).

## 1. The interface package `verity/protocols/interface/` (Lake package `core_protocol`, lib `CoreProtocol`)

The package requires `core_base` (`verity/circuits/base`), `core_commitments` (`verity/commitments/lean`) and
`verity_difftest` (`verity/difftest/lean`) (`verity/protocols/interface/lakefile.toml:15-25`) [read by me]. 3,735 lines of
Lean in 19 files (`find verity/protocols/interface -name '*.lean' -not -path '*/.lake/*' | xargs wc -l`) [counted by
me]. 13 Lake packages require it by path (`rg -l -g 'lakefile.*' core_protocol`, minus itself): proofs 3 (`spot/verifier/lean`,
`recursion/verifier/firewall`, `recursion/verifier/recursive`; `recursion/verifier/vstar_cli` lists it only among its
test inputs, `vstar_cli/pyproject.toml:22`), accounting 6 (`pouw/verifier/lean/check`, `pouw/verifier/lean/pearlplus`, `pous/verifier/check`, `pous/device/lean`,
`poub/verifier/lean`, `certifier/verifier/lean`), policy 1 (`inference_only/verifier/nci/lean`), compose 3
(`remote/verifier`, `remote/verifier/composition`, `onsite/verifier`) [counted by me]. `security/` requires it only
through those packages (`security/Properties/lakefile.toml` requires 17 code packages; it does not name
`core_protocol`) [read by me].

Importers below are direct Lean `import CoreProtocol.X` lines outside the package, deduplicated by file
(`rg -n -g '*.lean' -g '!**/.lake/**' '^\s*(public |meta |private )*import CoreProtocol' . | grep -v protocols/interface/`:
49 file-module pairs) [counted by me]. "Inside" lists the package's own importers.

| Module (lines) | What it defines (one line) [read by me] | Direct importers outside, by group [counted by me] | Inside the package | Python mirror [read by me] | Layer by the page |
| --- | --- | --- | --- | --- | --- |
| `CoreProtocol/Protocol.lean` (157) | `Core.Protocol`: params, commitment, evidence as canonical bytes, one total `check`, `Outcome` (verdict, public outputs, profile); `Protocol.answer` (`Protocol.lean:1-14`) | 9 files: accounting 5 (`poub/verifier/lean/RemoteAudit/P2Protocol.lean`, `…/Window.lean`, `pous/device/lean/MemoryChallenger/Verdict.lean`, `pous/verifier/check/PousCheck/SetupProtocol.lean`, `pouw/verifier/lean/check/PouwCheck/Inputs.lean`); proofs 1 (`recursion/verifier/firewall/Firewall/Protocol.lean`); policy 1 (`inference_only/verifier/nci/lean/NciCheck/Protocol.lean`); compose 1 (`remote/verifier/composition/CoreComposition/Product.lean`); security 1 (`security/Properties/Pouw/Lead/V2.lean`) | Account, Cli, Example, Live, Registration | `verity/difftest/python/verity_difftest/protocol.py:2` (the Python reference of `Core.Protocol.answer`) | page §6 `protocol.py`, "what a protocol declares" (page line 203) |
| `CoreProtocol/Protocol/Canonical.lean` (387) | canonical JSON and hex read at any size (`parseUpTo`, `parse`, `unhex`, `writes`) (`Canonical.lean:1-14`) | 1: compose (`remote/verifier/composition/CoreComposition/Product.lean`) | Cli | none found | page does not say; it is a reader of canonical JSON, which the page places nowhere by name |
| `CoreProtocol/Cli.lean` (118) | the one command line every protocol executable has: `id`, `check P C E`, `serve` (`Cli.lean:1-11`) | 18 files: accounting 10 (certifier `NetworkCheck/Check.lean`; poub `RemoteAudit/P2Protocol.lean`, `RemoteAudit/Protocol.lean`; pous `MemoryChallenger/Main.lean`, `TestMain.lean`, `WipeProtocol.lean`, `PousCheck/RunProtocol.lean`, `SetupProtocol.lean`; pouw `PouwCheck/Main.lean`, `PouwCheck/Period.lean`); proofs 3 (`Firewall/Protocol.lean`, `Spot/Main.lean`, `Spot/OneStage.lean`); policy 1 (`NciCheck/ProtocolMain.lean`); compose 4 (`onsite/verifier/CoreComposites.lean`, `OnsiteMain.lean`, `remote/verifier/RemoteMain.lean`, `RemoteV1Main.lean`) | Blobs, Example, ExampleMain, RegistrationMain | none | page §6 `protocol.py` (page line 203); the page does not name a command line |
| `CoreProtocol/Blobs.lean` (112) | content-addressed bytes beside the three strings; `BlobProtocol` with `checkBlobs` (`Blobs.lean:1-11`) | 4: proofs 2 (`recursion/verifier/recursive/VStarRecursive/Protocol.lean`, `spot/verifier/lean/Spot/OneStageZk.lean`); accounting 1 (`pouw/verifier/lean/pearlplus/PouwPearlPlus/Protocol.lean`); compose 1 (`CoreComposition/Blobs.lean`) | BlobsExample | none | page §6 `protocol.py` (page line 203) |
| `CoreProtocol/Live.lean` (49) | `Live P`: a check that factors through the challenger's signed coins (`checkAt`, `answerAt`) (`Live.lean:1-12`) | 1: proofs (`spot/verifier/lean/Spot/OneStage.lean`) | none | none | page §6 `protocol.py` (page line 203); coins are drawn by the challenger, page §4 Challenger row |
| `CoreProtocol/Profile.lean` (192) | the integrity profile: strata, sampling design over levels, members sampled, δ (`Profile.lean:1-9`); imports `Commitments.Draw` | 5: proofs 3 (`Spot/Bound.lean`, `Spot/OneStage.lean`, `Spot/Record.lean`); accounting 2 (`certifier/verifier/lean/NetworkCheck/Period.lean`, `pouw/verifier/lean/check/PouwCheck/Inputs.lean`) | Example | `verity/protocols/spot/verifier/profile.py:97` `IntegrityProfile` (the Lean docstring names it under the old path `verity.core.protocols.profile`, `Profile.lean:2`) | page §4 SPOT row: "At most this share is wrong, except with this probability" (page line 78); AGENTS.md: "the proof system owns … the integrity profile" |
| `CoreProtocol/Ratio.lean` (55) | exact non-negative rationals written as `"n/d"` strings (`Ratio.lean:1-5`) | 5: accounting 4 (`RemoteAudit/P2Protocol.lean`, `RemoteAudit/Protocol.lean`, `MemoryChallenger/WipeProtocol.lean`, `PousCheck/RunProtocol.lean`); proofs 1 (`Spot/PyJson.lean`) | Account, Profile, Site | none | page does not say (a number type) |
| `CoreProtocol/Record.lean` (89) | a record entry: one signed, numbered, chained transcript message `{body, kind, prev, schema, seq, session, sig}` (`Record.lean:1-12`) | 1: proofs (`spot/verifier/lean/Spot/ChallengerSession.lean`) | none | `verity/hardware/record.py:2` | page §6 `channel.py`, "a message and its types" (page line 201), and `runtime/`'s "transcripts" (page line 206) |
| `CoreProtocol/Period.lean` (147) | ticks, spans, the monitoring period cut into execution and sampling blocks: the code's copy of `security/Definitions/Core/Period.lean` (`Period.lean:1-12`) | 1: security (`security/Properties/Core/Period.lean`) | PeriodDifftest | `verity/periods/__init__.py:2-3`, held by `core-periods-difftest` (`verity/periods/tests/test_periods_difftest.py:23`) | page does not say where; AGENTS.md "Results are per monitoring period" |
| `CoreProtocol/Site.lean` (702) | the Site in code: registered hardware folded from the registration log (`Site.ofLog`), each entry's check, data sheets with `DeviceFigures`, locations, links (`Site.lean:8-21`); 117 declarations, types include `Sheet`, `DeviceFigures`, `Loc`, `Link`, `Site`, `Site.Registration`, `Auditor`, `DeviceModel` (`grep -E '^\s*(structure\|inductive\|abbrev)'`) | 3: accounting 1 (`certifier/verifier/lean/NetworkCheck/Registered.lean`, which reads `Core.Site` at lines 57-97); proofs 1, a difftest (`spot/verifier/lean/Spot/SiteDifftest.lean`, exe `site-difftest`, `spot/verifier/lean/lakefile.toml:60`; its docstring line 9-10: "its port is here because this package already requires the interface and the harness, and the registration log is the proof system's record"); security 1 (`security/Properties/Core/Site.lean`) | Account, Registration | `verity/hardware/site/__init__.py` (`Sheet` :280, `Entry` :400, `Registration` :432, `Site` :614), held by `spot/verifier/lean/tests/test_site_difftest.py` | page §6 `device.py`, "the registered hardware: hosts, devices, memory, links" (page line 202); the check is page §4 Registration row |
| `CoreProtocol/Registration.lean` (535) | the registration log: entries of three kinds, `verity/registration/site/v1`, `…/information/v1` (θ, registered values by root, compiler `B` with cap ℓ), `…/declaration/v1` (a day's circuit: day, program digest, block Definitions); its check; `View` (`Registration.lean:10-40`) | 0 Lean files outside import it (`rg` above). Driven from Python by `remote/verifier/composition/tests/test_registration_difftest.py:28,271` (exe `core-registration`, `lakefile.toml:44-46`). Named in prose by `pouw/verifier/lean/check/PouwCheck/Inputs.lean:20` and `certifier/verifier/lean/NetworkCheck/Registered.lean:110` | Account, RegistrationMain | `verity/hardware/registration.py:1` | page §4 Registration row (page line 82): accounting, "runs once, before any challenge"; the day's declaration is not in the page's Registration row |
| `CoreProtocol/Account.lean` (861) | what a protocol accounts for: `Part` (devices' compute, memory of tiers, read rate, a link's egress) with `Part.capacity` from the Site, `Side`, `Time`, `Pool`, `Amount`, `Account`, `Row` (`Account.lean:14-27`); imports Example, Protocol, Ratio, Registration, Site (`Account.lean:3-8`) | 0 Lean files outside import it; 0 Python files name it (`rg -c -g '*.py' 'Core\.Account\|CoreProtocol/Account'`) | none (only `CoreProtocol.lean`) | none | page §6 `protocol.py`: "which part of the hardware it accounts for" (page line 203-204); the rows are compose's "the period's account" (page line 83) |
| `CoreProtocol/Example.lean` (163), `ExampleMain.lean` (8) | a toy protocol that exercises every path; "It isn't a protocol of Verity's" (`Example.lean:1-6`); exe `core-protocol-example` | 1: compose (`remote/verifier/composition/CoreComposition/Stubs.lean`); Python `verity/difftest/tests/test_protocol_port.py:93` | Account, BlobsExample, ExampleMain | `test_protocol_port.py`'s reference | a test fixture of the interface (page does not say) |
| `CoreProtocol/BlobsExample.lean` (75) | a toy `BlobProtocol` with `#guard`s (`BlobsExample.lean:1-4`) | 0 | `CoreProtocol.lean` | none | a test fixture (page does not say) |
| `CoreProtocol/PeriodDifftest.lean` (52), `PeriodDifftestMain.lean` (7) | the periods difftest server `core-periods-difftest` | Python `verity/periods/tests/test_periods_difftest.py` | none | n/a | a test of `Period` |
| `CoreProtocol/RegistrationMain.lean` (8) | the `core-registration` executable's entry | Python `test_registration_difftest.py` (compose's tests) | none | n/a | registration's executable |
| `CoreProtocol.lean` (18) | imports all 14 library modules (`CoreProtocol.lean`) | (no importer by this name found) | n/a | n/a | n/a |

What the proofs' group imports of the package [counted by me, same `rg`]: `Blobs`, `Cli`, `Live`, `Profile`, `Protocol`, `Ratio`,
`Record`, and `Site` (only in `Spot/SiteDifftest.lean`, a difftest port). It imports none of `Registration`, `Account`,
`Period`. Since `Account` and `Registration` import `Site` and `Protocol` (`Account.lean:3-8`, `Registration.lean` imports
`CoreProtocol.Protocol`, `CoreProtocol.Site`, `Base.Hash`), the package's library target `CoreProtocol` builds all of
them for every requirer (`lakefile.toml:8`: `defaultTargets = ["CoreProtocol", "core-protocol-example",
"core-registration"]`) [read by me].

In `tests/import_order.toml` the interface is row 6 (line 20), below `verity/configuration` (row 8), `verity/hardware`
(row 9) and `verity/protocols/spot` (row 11) [read by me]. `tests/test_lean_vocabulary.py:76-78` pins
`PROTOCOL = …/CoreProtocol/Protocol.lean` and `SITE_LEAN, SITE_PY = …/CoreProtocol/Site.lean, verity/hardware/site/__init__.py`,
and line 139 makes `Core.Record` the canonical `Record` [read by me].

## 2. Imports among accounting, policy and compose, every direction

Python: `ast` over every `.py` under the seven protocol directories and `verity/composite`, absolute imports whose module
is `verity.protocols.<other>` or `verity.composite` (script `/tmp/sw/imp.py`; 48 import lines) [counted by me]. Strings
and loads by path: `rg -n -g '*.py' 'verity[./]protocols[./]<h>\b'` minus import lines [counted by me]. Lean: `import`
lines whose root module belongs to another package's group (script `/tmp/sw/leanimp.py`), and Lake `require` by path
(`/tmp/sw/lake.py` over every `lakefile.toml`) [counted by me]. A Lean line appears twice where the file has both a
`public import` and a `meta import` of one module; the counts below are per file.

### 2a. Lake `require` by path (Lean) [read by me]

| From (package) | To | Line |
| --- | --- | --- |
| certifier `network_check` | policy `nci_check` (`inference_only/verifier/nci/lean`) | `verity/protocols/certifier/verifier/lean/lakefile.toml:16` |
| poub `remote_audit` | pous `pous_check` (`pous/verifier/check`) | `verity/protocols/poub/verifier/lean/lakefile.toml:35` (comment line 31: "A P2 store's openings are checked on PoUS's vk (`RemoteAudit.P2`)") |
| pous `pous_check` | pous `core_memory_challenger` (`pous/device/lean`) | `verity/protocols/pous/verifier/check/lakefile.toml:37` (inside one protocol: verifier requires device) |
| certifier `certifier_device` | certifier `network_check` (`certifier/verifier/lean`) | `verity/protocols/certifier/device/lean/lakefile.toml:19` (inside one protocol: device requires verifier) |
| certifier `certifier_device` | `core_world` (`verity/hardware/world`) | `verity/protocols/certifier/device/lean/lakefile.toml:23` |
| compose `core_remote` | `core_composition`, pouw `pouw_check`, policy `nci_check`, pouw `pouw_pearlplus` (and proofs `core_spot`, `core_vstar_recursive`) | `verity/protocols/remote/verifier/lakefile.toml:23,32,37,48` (proofs at 28, 43) |
| compose `core_onsite` | `core_composition` :24, pouw `pouw_check` :34, policy `nci_check` :39, certifier `network_check` :45, pous `pous_check` :50, poub `remote_audit` :55, pous `core_memory_challenger` :66, compose `core_remote` :72 (and proofs `core_spot` :30, `core_vstar_recursive` :61) | `verity/protocols/onsite/verifier/lakefile.toml` |
| accounting or policy → compose | none | (`/tmp/sw/lake.tsv`, 132 requires in all) |

### 2b. Lean imports across groups [counted by me]

| From | To | File:line | Module |
| --- | --- | --- | --- |
| certifier | policy | `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean:9,10` (meta at 17,18) | `NciCheck.Desc`, `NciCheck.DefinitionDigest` |
| poub | pous | `verity/protocols/poub/verifier/lean/RemoteAudit/P2.lean:3` | `PousCheck.Merkle` |
| certifier device | hardware | `verity/protocols/certifier/device/lean/CertifierDevice/Driver.lean:4` | `World` |
| pous | security | `verity/protocols/pous/lean/submissions/reference.lean:1` | `Proofs.Pous.Statements` (an `against` tie, `tests/import_order.toml` "loader = verity/protocols/pous, loaded = security") |
| compose remote | pouw | `verity/protocols/remote/verifier/CoreRemote.lean:58,59` | `PouwCheck.Period`, `PouwPearlPlus.Protocol` |
| compose remote | policy | `verity/protocols/remote/verifier/CoreRemote.lean:57` | `NciCheck.Protocol` |
| compose onsite | pous | `verity/protocols/onsite/verifier/CoreOnsite.lean:33,36` | `MemoryChallenger.WipeProtocol`, `PousCheck.RunProtocol` |
| compose onsite | pouw | `CoreOnsite.lean:37` | `PouwCheck.Period` |
| compose onsite | poub | `CoreOnsite.lean:38` | `RemoteAudit.P2Protocol` |
| compose onsite | certifier | `CoreOnsite.lean:35` | `NetworkCheck.Period` |
| compose onsite | policy | `CoreOnsite.lean:34` | `NciCheck.Protocol` |
| compose onsite | compose remote | `CoreOnsite.lean:29-32`; `CoreComposites.lean:17` | `CoreComposition.Blobs`, `.Cases`, `.Tie`, `CoreRemote` |
| pouw, pous, policy → another accounting protocol, the policy or compose | none other than the rows above | | |

### 2c. Python imports across groups (source, not tests) [counted by me]

| From | To | File:line | What |
| --- | --- | --- | --- |
| poub verifier | pous verifier | `verity/protocols/poub/verifier/monitoring.py:37` | `verity.protocols.pous.verifier.wipe_monitoring` |
| poub verifier | pous verifier | `verity/protocols/poub/verifier/p2.py:54-57` | `pous.verifier`, `.params`, `.schemes.p2`, `.verifier` |
| poub verifier | pous verifier | `verity/protocols/poub/verifier/p2_case.py:40` | `pous.verifier.run` |
| poub verifier | pous device | `verity/protocols/poub/verifier/p2_party.py:75` | `verity.protocols.pous.device` |
| poub verifier | pous verifier | `verity/protocols/poub/verifier/p2_party.py:76` | `pous.verifier.schemes.p2` |
| poub verifier | pous verifier | `verity/protocols/poub/verifier/period.py:48,49` | `pous.verifier.period` |
| compose `composite` device | pouw verifier | `verity/composite/device/live_challenger/models.py:45,46` | `verity.protocols.pouw.verifier…` |
| compose `composite` device | certifier verifier | `verity/composite/device/live_challenger/coin_link.py:40` | `verity.protocols.certifier.verifier.coin_link` |

That is 10 source import lines poub → pous (5 files) and 3 compose → accounting (2 files); no source import from pouw,
pous, the certifier or the policy into another of the five, and none from the five into compose. The listing allows
poub's verifier to load pous's (`tests/import_order.toml`, place `verity/protocols/poub/verifier`, `loads` holds
`verity/protocols/pous/verifier`) [read by me].

### 2d. Python, tests only [counted by me]

| From | To | File:line |
| --- | --- | --- |
| poub tests | pous | 19 import lines in 7 files: `poub/verifier/lean/tests/test_p2.py:26`, `test_p2_case.py:24-26`, `test_p2_party.py:25,26,317-319`; `poub/verifier/tests/test_p2.py:18-20`, `test_p2_party.py:17,18`, `test_same_store.py:19-23` |
| poub tests | pouw | `verity/protocols/poub/verifier/tests/test_p2_party.py:19` |
| pouw tests | certifier | `verity/protocols/pouw/verifier/tests/test_pouw_salt.py:27-31` (`certifier.verifier`, `.active`, `.certificate`, `.grid`, `.schedule`) |
| certifier tests | pouw | `verity/protocols/certifier/verifier/network/tests/period_case.py:82` |
| compose (composition) tests | pous, poub, pouw | `remote/verifier/composition/tests/periods.py:22-27`; `reference.py:54` |
| compose (composite) tests | pouw | `verity/composite/device/live_challenger/tests/test_auditor_models.py:17,18` |

### 2e. Names, strings and loads by path across groups (not import statements) [read by me]

| From | To | File:line | What it does |
| --- | --- | --- | --- |
| certifier tests | policy | 7 files `verity/protocols/certifier/verifier/network/tests/test_nci_*.py` (line 3 or 4 of each) | the certifier's test directory holds the difftests of the policy's Lean check (`inference_only/verifier/nci/lean/scripts/*_vectors.py`) |
| policy scripts | certifier tests | `inference_only/verifier/nci/lean/scripts/desc_fuzz_programs.py:10`, `difftest_vectors.py:10` | names the certifier's tests as the checks of these scripts |
| policy `rho` | certifier | `verity/protocols/inference_only/rho/rho/envelopes.py:280,309` | prose: the grammar of `certifier/verifier/io_request.py` |
| policy `nci` | pouw | `verity/protocols/inference_only/nci/tiers.py:77` (comment: the clock PoUW credits, from `pouw/prover/circuit/h1device.py`); `nci/tests/test_tiers.py:39` reads that file's text |
| poub device | pouw | `verity/protocols/poub/device/live_verifier/host.py:165` `MODEL_PINS = Path("verity/protocols/pouw/verifier/lean/check/models.json")`; also `poub/verifier/store.py:146` (prose) |
| poub device | pous | `verity/protocols/poub/device/live_verifier/host.py:167-168` `PARTIES` runs `verity.protocols.pous.verifier.wipe_party` and `verity.protocols.pous.device.challenger_party` as modules; `:764` `KEYGEN` imports `pous.verifier.period` in a string |
| poub tests | pous | `poub/verifier/tests/test_same_store.py:26`, `poub/verifier/lean/tests/test_p2_case.py:32` load `pous/verifier/tests/pous_test_signer.py` by path |
| pous device | poub | `verity/protocols/pous/device/challenger_session.py:25,202`; `challenger_party.py:46` (prose naming `poub.verifier.p2_party --pous-record`) |
| poub verifier | compose `composite` | `poub/verifier/session/challengers.py:6,12`; `session/verifier.py:88`; `device/live_verifier/host.py:13` (prose: the challenger is `verity/composite/device/live_challenger/serve.py`, `auditor.py`) |
| certifier verifier | compose `composite` | `verity/protocols/certifier/verifier/coin_link.py:5` (prose: the challenger served by `composite/device/live_challenger/serve.py`) |
| compose `composite` | pouw | `verity/composite/device/live_challenger/models.py:53` `PINS = Path("verity/protocols/pouw/verifier/lean/check/models.json")`; `auditor.py:157` |
| compose `composite` | certifier | `auditor.py:82,205,236` (prose: `certifier.verifier.coin_link`, `.warden`, `.release`) |
| compose (composition) tests | pouw, pous, certifier | `remote/verifier/composition/tests/reference.py:80,83,86,88` and `periods.py:104` load test modules of other protocols by path (`pouw/verifier/lean/check/tests/pouw_period_cases.py`, `certifier/verifier/network/tests/test_network_period_protocol.py`, `…/test_inference_only_protocol.py`, `pous/device/lean/tests/wipe_protocol_case.py`, `pous/device/lean/tests/mc_cases.py`) |
| pouw verifier | certifier | `verity/protocols/pouw/verifier/salt.py:1` (docstring: "PoUW's read of the network certifier's per-execution-phase salt (`verity/protocols/certifier/verifier/active.py`)") |

The listing's order (`tests/import_order.toml:14-43`) is one total order: pous (row 14), poub, pouw, inference_only,
certifier, remote, onsite, composite (row 21). So by the listing the certifier (row 18) may load the policy (row 17), which
2a and 2b show it does [read by me]. The page's diagram has policy → accounting and no arrow from accounting to policy
(page lines 171-179) [read by me].

## 3. The runtime; adapters, PyTorch, vLLM and models; `commitments/` and `circuits/` naming a protocol

### 3a. Who imports the runtime

What `main` has as the runtime is inside `verity/configuration/deploy/` (2,352 lines, 9 files, `wc -l`): its
`__init__.py:1-8` names `party` (a party's interface), `manifest` (what a protocol declares), `config` (the configuration),
`runtime` (what runs the parties: `run`, `runtime.py:553`), `stream` (one party over a byte stream), `sources` (clocks and
coins), and internal `transcript`, `channels`, `process`; "The layer … imports no protocol" [read by me]. By the page,
`party`/`manifest`/`config` are the shared files `party.py`, `protocol.py`, `config.py` (page §6 lines 200-205) and
`runtime`, `stream`, `sources`, `channels`, `process`, `transcript` are `runtime/` ("sockets, randomness and clock,
transcripts", page line 206). Inside the layer: `process.py:24` imports `runtime`; `runtime.py:51-55` imports `sources`,
`channels`, `manifest`, `party`, `transcript`; `party.py`, `manifest.py`, `config.py` import only `verity.configuration.wire`
and `manifest` [read by me].

Imports of the runtime's modules from outside `verity/configuration` (`ast` over `verity/`, script `/tmp/sw/dep.py`: 23
import lines name `runtime`, `stream` or `sources`; 16 name only `party`, `manifest`, `config`) [counted by me]:

| Importer | Line | What it takes | Source or test |
| --- | --- | --- | --- |
| poub verifier `session_parties.py` | 47 | `runtime.ProtocolDef` (`runtime.py:73`: "A protocol as the runtime runs it: its manifest, and a factory for each of its parties") | source |
| poub verifier `verifier_party.py` | 34 | `stream.Side` | source |
| pous device `challenger_party.py` | 64, 66 | `stream`, `runtime.ProtocolDef` | source |
| pous device `challenger_session.py` | 46, 49 | `stream`, `stream.Side` | source |
| pous verifier `period.py`, `wipe_party.py`, `wipe_period.py` | 60, 42, 59 | `stream.Side` | source |
| pous verifier `wipe_parties.py` | 33 | `runtime.ProtocolDef` | source |
| spot verifier `parties.py` (proofs) | 126 | `runtime.ProtocolDef`; also `config.COMPOSITES` at 123 | source |
| poub tests (`test_session_parties.py:13-14`, `test_verifier_party.py:17`), pous tests (`test_pous_challenger_party.py:23-26`, `test_pous_wipe_parties.py:19-21`, `test_pous_wipe_period.py:28`, `test_pous_wipe_store.py:28`), spot tests (`test_parties.py:19`, `test_spot_parties.py:26`) | | `runtime.run`, `sources`, `stream.Side`, `ProtocolDef` | tests |

No file of PoUW, the certifier, the policy, compose (`remote`, `onsite`, `composite`), `verity/hardware`,
`verity/circuits` or `verity/commitments` imports a runtime module (same scan) [counted by me]. The shared file
`verity/configuration/deploy/config.py:59-61` names the protocol ids of both composites in `COMPOSITES`
(`"verity/onsite/v0": ("bandwidth", "certifier", "inference-only", "pous", "pouw", "sampled-proofs", "wipe")`,
`"verity/remote/v1": ("inference-only", "pearl-plus", "sampled-proofs")`); its docstring line 25: "this module imports
no protocol and no workload" [read by me].

### 3b. Adapters, PyTorch, vLLM and models named in `verity/`

Import statements (and `import_module` strings) of `torch`, `vllm`, `verity_vllm`, `transformers`, `safetensors`,
`triton`, `huggingface_hub` anywhere under `verity/` (`/tmp/sw/adp.py`) [counted by me]:

| File | Line | Module |
| --- | --- | --- |
| `verity/protocols/flock/pod/gemm_slowdown.py` | 54 | `torch` |
| `verity/definitions/requirement/tests/test_census_and_memory_map.py` | 179, 180, 198, 242, 269, 270, 271, 404 | `verity_vllm.program.predict.build`, `.phi`, `verity_vllm.query.required`, `verity_vllm.protocols.remote_bandwidth.memory_map`, `verity_vllm.commit`, `…remote_bandwidth.interface`, `…remote_bandwidth.registry` |

`tests/test_integration_boundaries.py` `KNOWN` (lines 35-81) has 43 entries: the 7 `verity_vllm` modules of
`test_census_and_memory_map.py` above, and 36 entries naming `verity.definitions.*` modules (in `flock/prover`,
`flock/sigma/tests`, `definitions/tests`, `definitions/evaluate/tests`). `INTEGRATIONS` is read from each
`adapters/*/pyproject.toml`'s wheel packages, which is `['verity_vllm']` (`adapters/vllm/pyproject.toml`); the check for
gone entries (line 137) looks only at modules whose root is in `INTEGRATIONS`, so the 36 `verity.definitions.*`
entries are neither found nor checked as gone [read by me; counts `sed -n 36,80p … | grep -c`].

Names in text (code, comments, docstrings, data) under `verity/`, excluding `.lake` [counted by me]:

| Pattern | Files / lines | Where most (files/lines) | Places that are code, not prose [read by me] |
| --- | --- | --- | --- |
| `vllm` (case-insensitive; `rg -c -i vllm verity`) | 416 / 1,260 | `definitions/registry` 120/391, `protocols/flock` 79/238, `protocols/certifier` 47/155, `definitions/numerics` 31/97, `protocols/pouw` 29/54, `composite` 7/58, `commitments/*` 11 files | `verity/commitments/domains.py:183-196,210,397-400`: 14 constants named `VLLM_*` with bytes `"verity-vllm/…"` (e.g. `VLLM_PEARL_C_RUN = "verity-vllm/pouw-pearl-c-run/v2"`, `VLLM_SERVING_ROWS = "verity-vllm/serving-rows/v2"`), the same in Lean `verity/commitments/identity/Identity/Domains.lean:30-48,236-239` and `Identity/Domains/Shared.lean:61-64`; `verity/commitments/windows/rows.py:93,97,1269` and `windows/interface.py:132` write `"verity-vllm/serving-rows/…"` and `"verity-vllm/advice-windows/advice/v1"`; `verity/commitments/pyproject.toml:10-11` lists `adapters/vllm/verity_vllm/settings.py`, `__init__.py`, `check` as test inputs; `verity/errors.py:5` (docstring: verdict codes `verity_vllm.check.codes` build on `ProtocolError`); `verity/settings.py:5-9` (the vLLM adapter keeps its own copy) |
| vLLM's scheduler (`rg -c -i scheduler verity`) | 42 / 121 | `protocols/certifier/verifier/replay_step.py` 26 | `replay_step.py:1-15`: "One vLLM v1 scheduler step as a Boolean Definition … vLLM v1's `Scheduler.schedule` and `update_from_output` (the pinned 0.28.1rc1.dev472)"; it is in the certifier's verifier place |
| `\btorch\b\|PyTorch` | 137 / 483 | `definitions/registry` 53/203, `definitions/tests` 14/146, `protocols/flock` 22/44, `definitions/numerics` 20/46, `definitions/ops` 14/20, `protocols/pouw` 6/13 | the one import above; the rest not split into code and prose (not read) |
| model families (`llama\|qwen\|deepseek\|mistral\|mixtral\|gpt-oss\|GPT-n\|gemma\|kimi\|phi-n\|granite\|olmo`, case as written) | 209 / 638 | `protocols/inference_only` 31/282, `definitions/registry` 75/165, `protocols/flock` 29/62, `protocols/pouw` 19/34, `composite` 1/20, `protocols/certifier` 11/13, `protocols/poub` 6/13, `hardware/cards` 1/2 | see the next row |
| model checkpoints pinned in a verifier | 1 file, 7 models | `verity/protocols/pouw/verifier/lean/check/models.json` | keys `GLM53_INT8`, `LLAMA31_8B_INT8`, `LLAMA32_1B_BF16`, `LLAMA32_1B_INT8`, `LLAMA32_1B_INT8_K2048`, `QWEN25_32B_BF16`, `TORCH_LOOP_MLP_H128_W8A8`, each a `checkpoint_sha512` and the SHA-512 of each file (`model-00001-of-00141.safetensors`, …). Read by PoUW's Lean verifier `PouwCheck/Credit.lean:1634` (`include_str "../models.json"`, `PINNED_MODELS`, `MODEL_FIELD := "model"` at 1625), by `pouw/verifier/credit.py:284-285`, by PoUB's device `poub/device/live_verifier/host.py:165` (`MODEL_PINS`), by `poub/verifier/store.py:146` (prose) and by compose's device `verity/composite/device/live_challenger/models.py:53` (`PINS`) |
| other files named for models | | `verity/protocols/inference_only/nci/models.json`, `verity/protocols/inference_only/rho/data/models.toml` | not opened |

### 3c. `verity/commitments/` and `verity/circuits/` naming a protocol

Imports [counted by me]: no Lean `import` and no Lake `require` from `verity/circuits/*` or `verity/commitments/*` into a
protocol (`/tmp/sw/leanimp.py`, `/tmp/sw/lake.tsv`). Python: one file, a test,
`verity/commitments/tests/test_block_check.py:17-24` imports `verity.protocols.poub.verifier.audit`, `.codec`,
`.receipts`, `.commitments.gates.block_check` (the listing names it in poub verifier's `loaded_by`). The listing's
`verity/commitments` place holds two files of PoUB's verifier, `verity/protocols/poub/verifier/commitments/gates/block_check.py`
and its `.circuit-check.py` (`tests/import_order.toml`, the `verity/commitments` place's `paths`) [read by me].

Names in text, `rg -c` per pattern, `.lake` excluded [counted by me]:

| Pattern | `verity/circuits` files/lines | `verity/commitments` files/lines |
| --- | --- | --- |
| PoUW (`\b(pouw\|PoUW\|Pouw)\b`) | 10 / 19 | 21 / 240 |
| PoUS | 0 / 0 | 4 / 72 |
| PoUB (`poub\|bandwidth`) | 0 / 0 | 4 / 56 |
| certifier | 0 / 0 | 12 / 94 |
| SPOT (`spot\|sampled.proofs`) | 1 / 1 | 14 / 43 |
| Flock | 30 / 75 | 46 / 178 |
| registration | 12 / 29 | 17 / 109 |
| inference-only, NCI | 0 / 0 | 0 / 0 |

The largest is `verity/commitments/domains.py` (409 lines, 341 constants `^[A-Z_0-9]+ = "`), "One table of domain
separators: every versioned string a protocol hashes…" (`domains.py:1-17`), mirrored by
`verity/commitments/identity/Identity/Domains.lean` and held equal by `tests/test_domains.py`. Its sections, constants
per section [counted by me, `awk` over the `# ` headers]: PoUW 91 (lines 19-125) and Pearl 5 (126); shared 44 (133-186,
including "sampled proofs' one-stage statement, registrations and partitions" 9 at 171 and "the served path's PoUW
manifests (`verity_vllm.protocols.pouw`)" 3 at 182); "Not yet read from here": the commitments' own 24, signatures 1,
sampled proofs and the challenger 23 (216), Flock and the ZK server 23 (241), PoUS, the wipe and the memory challenger
44 (266), the remote bandwidth audit 27 (312), the network certifier 49 (341), composition 2 (392), served periods 5
(396), the wire and the deployment layer 3 (403).

## 4. Shared concepts defined more than once

Definitions found by `rg` for `structure|inductive|abbrev|def|class <Name>` in Lean (`.lake` and, unless named,
`security/Proofs` excluded) and `class <Name>` or a module constant in Python; importers counted with
`rg -l 'import <Module>$'` [counted by me]. "Who reads" is direct importers only.

### 4a. The data sheet (a device model's figures)

| Definition | File:line [read by me] | What it is | Who reads it [counted by me] |
| --- | --- | --- | --- |
| card files (TOML) | `verity/hardware/cards/*.toml` (7: `b200-sxm-180gb`, `dgx-h200`, `h200-sxm-141gb`, `nebius-vm-monotonic-raw`, `rtx-pro-6000-server`, `xeon-6776p-2s`, `xeon-platinum-8468-2s`) and `verity/hardware/nodes/*.toml` (2) | one data sheet per GPU, host or server model | `verity/hardware/cards/__init__.py` (`load`); Python importers of `verity.hardware.cards`: `pouw/verifier/tests/test_h1device_cards.py:7`; adapter `verity_vllm` 3 files (2 source, 1 test); 12 files in `benchmarks/` (`rg -l 'verity\.hardware\.cards\|from verity\.hardware import cards'`) |
| `Verity.Accel.Card` (Lean, spec) | `security/Definitions/Core/Accel/Card.lean:31` (store sizes, each resource's rate a second, MAC weights, P1's range) | "A card's numbers (trusted)" | 4 files of `Definitions/Core` (`Accel.lean`, `Accel/H200.lean`, `Accel/Sm120.lean`, `Accel/Machine.lean`); `Accel/H200.lean` and `Accel/Sm120.lean` are generated from the card files by `verity/hardware/lean.py:1-5` ("The Lean card tables, generated from the card files") |
| `Accounting.TS.Sheet K P` (Lean, spec) | `security/Definitions/Accounting/TS/Sheet.lean:68` (`fmax`, `macs`, `peak`, `pipe`, `width`, `smem`) | "A device's data sheet"; its `fmax` docstring holds the H200's 1,980 MHz and sm_120's 2,430 MHz | 5 files of `Definitions/Accounting` (`TS.lean`, `Assumptions/Sheet.lean`, `Sheets/H200.lean`, `Sheets/B200.lean`, `Headline/Encoding.lean`); `Sheets/H200.lean` and `Sheets/B200.lean` are hand-written tables (not named by `verity/hardware/lean.py`) |
| `DeviceFigures` (Lean, spec) | `security/Definitions/Core/DeviceFigures.lean:93` | the figures a registered device carries | 4 files (`Definitions/Core` 2, `Proofs/Core` 1, `Definitions.lean`) |
| `Core.Site.DeviceFigures`, `Core.Site.Sheet` (Lean, code) | `verity/protocols/interface/CoreProtocol/Site.lean:102,150` | the code's copy of the spec's (`Site.lean:16-17`) | `Properties/Core/Site.lean` states it is the spec's; importers of `CoreProtocol.Site` in item 1 |
| `DeviceFigures`, `Sheet` (Python) | `verity/hardware/site/__init__.py:214,280` | the Python mirror, held by `site-difftest` | `spot/verifier/lean/tests/test_site_difftest.py:25`; `remote/verifier/composition/tests/test_registration_difftest.py:24`; benchmarks `certifier_term`, `compute` |
| H1's device records (Python literals) | `verity/protocols/pouw/prover/circuit/h1device.py:1-3` | "H1's device records: the hardware facts an H1 execution phase … is built and served on, one a GPU architecture" | `test_h1device_cards.py:1-2`: the records "state each device's SMs, clocks and dense s8 MACs as literals, and the hardware cards … state the same numbers: each literal equals its card's entry"; the policy names it at `inference_only/nci/tiers.py:77` |
| `Sheet` (Python, benchmarks) | `benchmarks/machine_model/verity_machine_model/sheet.py:82`, `benchmarks/pouw/proved_ticks.py:90` | outside `verity/` | not traced |

### 4b. The resource, or part of the hardware, a protocol accounts for

| Definition | File:line [read by me] | Constructors | Who reads it [counted by me] |
| --- | --- | --- | --- |
| `Accounting.Resource` | `security/Definitions/Accounting/Joint.lean:38` | `compute`, `bandwidth`, `capacity` ("The resources a joint audit certifies, each by an audit of its own") | `Joint.lean` imported by 9 files (`Definitions/Accounting` 7 and `Definitions/Accounting.lean`, `Proofs/Accounting/Device.lean`) |
| `Core.Account.Part` | `verity/protocols/interface/CoreProtocol/Account.lean:103` | `compute`, `memory (tiers)`, `read (tiers)`, `egress` | no importer outside the package (item 1) |
| `Verity.Accel.Res` | `security/Definitions/Core/Accel/Isa.lean:55` | `tensor`, `issue`, `alu`, `imad`, `shfl`, `f64`, `sfu`, `smem`, `l2`, `dram` (a card's pooled resources) | the `Accel` files |
| memory tier, Lean | `security/Definitions/Core/Site.lean:116` (spec) and `CoreProtocol/Site.lean:66` (code): `hbm`, `dram`, `disk` | | |
| memory tier, Python | `verity/hardware/site/__init__.py:63` `TIERS = ("hbm", "dram", "disk")`; `verity/protocols/certifier/verifier/registered.py:32` `TIERS = ("hbm", "dram", "disk")`; `verity/protocols/poub/verifier/session/verifier.py:153` `TIERS = ("hbm", "host")`; `verity/protocols/poub/verifier/ledger.py:71` `class Tier`; `verity/protocols/inference_only/nci/tiers.py:169` `TIERS = {` | | not traced per reader |

### 4c. A protocol's id, and which protocols a composite holds

| Definition | File:line [read by me] | Content |
| --- | --- | --- |
| `Accounting.Audit` (Lean, spec) | `security/Definitions/Accounting/JointPeriod.lean:39` | `sampled`, `pouw`, `pous`, `poub`, `network` ("The audits that decide a period, one per protocol") |
| `Verity.Subprotocol` (Lean, properties) | `security/Properties/Verity/Game.lean:132`, with `Subprotocol.ofAudit` at 154 | `sampled`, `pouw`, `inferenceOnly`, `pous`, `poub`, `network`, `wipe` |
| `Composite` (Lean) | spec `security/Definitions/Core/Site.lean:351`; code `CoreProtocol/Site.lean:285` | `onsite`, `remote` |
| composite ids and their components (Lean, compose) | `verity/protocols/onsite/verifier/CoreOnsite.lean:84,96-97`; `CoreRemote.lean:102` (`"verity/remote/v0"`); `CoreComposites.lean:70` (`["verity/onsite/v0", "verity/remote/v1"]`) | the seven component ids of onsite |
| `COMPOSITES` (Python, shared config) | `verity/configuration/deploy/config.py:59-61` | composite id → component names |
| `COMPOSITES` (Python, hardware) | `verity/hardware/site/__init__.py:61` | `("onsite", "remote")` |
| `SUBPROTOCOLS`, `IDS` (Python, SPOT's parties) | `verity/protocols/spot/verifier/parties.py:146-157` | composite id → component name → protocol id (e.g. `"pouw": "verity/pouw/period/v1"`, `"bandwidth": "verity/remote-bandwidth/p2/v0"`), and each component's id outside a composite |
| domain strings | `verity/commitments/domains.py:19` ("PoUW: the protocol and its identifier", 3 constants) | |

### 4d. A period or window

| Definition | File:line [read by me] | Who reads it [counted by me] |
| --- | --- | --- |
| `Verity.Tick`, `TickSpan`, `MonitoringPeriod` (spec) | `security/Definitions/Core/Period.lean:31,35,69` | |
| `Core.Tick`, `TickSpan`, `MonitoringPeriod` (code) | `verity/protocols/interface/CoreProtocol/Period.lean:34,37,70` | `security/Properties/Core/Period.lean` only (item 1) |
| `TickSpan`, `MonitoringPeriod`, `block_count` (Python) | `verity/periods/__init__.py:68,120` | `verity/protocols/pous/verifier/wipe_monitoring.py:45` (`block_count`) |
| `MonitoringPeriod` (SPOT's challenger ledger) | `verity/protocols/spot/verifier/lean/Spot/ChallengerSession.lean:481` (`name`, `lane`, `windows`, `closed`, `cap`, `circuit`) | (proofs' group; not traced) |
| `MonitoringPeriod` (Python, the slowdown calculator) | `verity/protocols/inference_only/rho/rho/hardware.py:101` (`name`, `seconds`) | not traced |
| the week/month/year list | `verity/periods/__init__.py:29` `MONITORING_PERIODS` and `verity/protocols/reporting.py:11` `PERIODS`, the same three pairs | `MONITORING_PERIODS`: only `verity/periods/tests/test_periods.py`. `PERIODS`: `pous/verifier/wipe_monitoring.py:46`, `poub/verifier/monitoring.py:38`, `inference_only/rho/rho/params.py:39,49`, and 20 lines in 16 files of `benchmarks/` (`rg -n 'reporting\.PERIODS\|import.*\bPERIODS\b' benchmarks`) |
| `Window` (spec, accounting) | `security/Definitions/Accounting/Device.lean:81` (`start`, `len`) | |
| `Window` (certifier, Lean) | `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean:412` | |
| `Window` (commitments, Python) | `verity/commitments/windows/interface.py:2214` | |
| `Window` (requirement difftest) | `verity/definitions/requirement/CircuitRequirement/Difftest.lean:92` | |
| `Period` (an attested certification period) | `verity/commitments/attest/__init__.py:176`; Lean `Attest/Period.lean` | |
| `Epoch` | `verity/protocols/pouw/verifier/audit.py:169`; `verity/protocols/flock/verifier/Flock/Firewall/Salts.lean:287` | |

### 4e. A commitment's root

The shared scheme is frame-v3: Python `verity/commitments/frame_v3/__init__.py:132` (`root`), Lean
`verity/commitments/lean/Commitments/FV3.lean:79,166` [read by me]. Other tree roots computed outside it [read by me]:

| Definition | File:line | What it computes |
| --- | --- | --- |
| PoUS's verification tree | `verity/protocols/pous/verifier/check/PousCheck/Merkle.lean:76` (`root`): "The SHA-512 Merkle tree whose root binds the stored blocks", leaves `SHA512(0x00 ‖ salt ‖ …)`, nodes `SHA512(0x01 ‖ l ‖ r)` (`Merkle.lean:6-14`) | imported by PoUB's `RemoteAudit/P2.lean:3` (item 2b) |
| PoUW's tree from subtree roots | Lean `verity/protocols/pouw/verifier/lean/check/PouwCheck/Credit.lean:893` (`treeRoot`); Python `verity/protocols/pouw/verifier/credit.py:3195` (`tree_root`, "The root of the frame-v3 SHA-512 tree … from the subtree roots") | |
| the memory challenger's coin tree | `verity/protocols/pous/device/memory_challenger.py:139` (`_root`, over `tree_levels`) | |
| the attested period's root | `verity/commitments/attest/__init__.py:96`; Lean `verity/commitments/attest/lean/Attest/Period.lean:49` | |
| the firewall's frame tree (proofs) | `verity/protocols/flock/verifier/Flock/Firewall/Frame.lean:142` (`treeRoot`, "each node SHA-512(left ‖ right)") | |
| spec | `security/Definitions/Hash/Tree.lean:121,163` (`rootOf`, `root`) | |

`rg -n '^\s*def (root|tree_root|_root|rootOf|treeRoot)\b'` over `verity/` finds 56 Python definitions, and the Lean form
over `verity/` and `security/` 41 (not all of them commitment roots; not each read) [counted by me].

### 4f. A challenge or coin derivation

The one derivation is `verity/commitments/randomness/__init__.py:154` (`derive`) and Lean
`verity/commitments/randomness/Randomness.lean:61` [read by me]. PoUS's challenge indices go through it
(`pous/verifier/streamed_wipe.py:109-112` calls `derive`; Lean `pous/device/lean/MemoryChallenger/Records.lean:113`
calls `FlockSoundness.Randomness.derive`) [read by me]. Other derivations from SHA-512 [read by me]:

| Definition | File:line | What it is |
| --- | --- | --- |
| `xof` (SHA-512 counter mode) | Python `verity/protocols/poub/verifier/codec.py:251`; Lean `poub/verifier/lean/RemoteAudit/Codec.lean:73` and `RemoteAudit/Unboxed.lean:33` | PoUB's codec stream, two Lean definitions in one package |
| `derive` (codec key) | `verity/protocols/poub/verifier/codec.py:298` | a Sloth key from a seed (a codec key, not a challenge) |
| `xof` | `verity/protocols/pouw/prover/schemes/ncp.py:52` | NCP's stream |
| `stream` | `verity/protocols/pouw/prover/circuit/seeds.py:143`; `pous/verifier/schemes/p2.py:327`; `pouw/receipts/receipt_conform.py:60` | per-scheme streams |
| `Commitments.Draw.derive` | `verity/commitments/lean/Commitments/Draw.lean:134` | a sampling law from strata (imported by `CoreProtocol/Profile.lean`) |
| `coins` | `pouw/verifier/verifier.py:123`, `spot/verifier/one_stage/verifier.py:177` | not read |

### 4g. A device's key

| Definition | File:line [read by me] | What it is |
| --- | --- | --- |
| `Core.Site.Loc.key` | `CoreProtocol/Site.lean:169` (`key : String := ""`), `Auditor.key` at 301, checked by `Owners` at 365-368 ("each new certifier or challenger is one of the auditor's own devices, owned by it and with its key") | the key a location registers; read by the certifier's `NetworkCheck/Registered.lean:25,84` |
| `PartyKey`, `Identity`, `PrivateKey` | `verity/commitments/crypto/lms.py:658,598,116` | "One party's lower tree on its device, certified by the device's identity" |
| `Signer` (protocol) | `verity/commitments/attest/__init__.py:124`; Lean `Attest/Period.lean:63` `PartyKey` | |
| `Signer`, `LmsSigner` | `verity/protocols/pous/device/memory_challenger.py:297,326` | "A memory challenger's S1 key (`lms.PrivateKey`)" |
| `Signer`, `PublicKey`, `LmsKey`, `LmsSigner` | `verity/protocols/certifier/verifier/certificate.py:321,335,343,361` | "A certifier's LMS public key, `I ‖ T[1]`" |
| `ChallengerKey` | Python `verity/protocols/spot/device/challenger.py:576`; Lean `Spot/ChallengerRecord.lean:65` | "The verifier's key for its own challenger: LMS over SHA-512" |
| `Signer` (game) | `security/Properties/Verity/Game.lean:474` | the device's signer in the game |
| `Identity` (party) | `verity/configuration/deploy/party.py:40`; handshake `Key` `verity/configuration/wire/handshake.py:39` | |

### 4h. "Registration", which item 4's list does not name but the brief's leads do

| Definition | File:line [read by me] | Format strings |
| --- | --- | --- |
| spec `Verity.Registration` (entries, `Declared`, `Builder`, `Item`) | `security/Definitions/Core/Site.lean:189-222`; `Registration.Theta` `security/Definitions/Core/Compiler.lean:208` | |
| code `Core.Registration` (the log) | `CoreProtocol/Registration.lean:51`; Python `verity/hardware/registration.py` | `verity/registration/{information,declaration,log,genesis}/v1` |
| code `Core.Site.Registration` | `CoreProtocol/Site.lean:200-205`; Python `verity/hardware/site/__init__.py:432` | |
| SPOT's window registration | `Spot/Registration.lean:98`; Python `spot/verifier/one_stage/registration.py` | `verity/registration/v0` to `/v3`, `verity/registration-receipt/v1`, `verity/registration/produced/v0`, `/v1` (in `verity/protocols/spot/` and `verity/commitments/domains.py`) |
| PoUB's registrations | Lean `poub/verifier/lean/RemoteAudit/P2.lean:49`, `RemoteAudit/Window.lean:52`; Python `poub/verifier/audit.py:373`, `poub/verifier/p2.py:91` | |
| `ServedRegistration` | `verity/protocols/spot/verifier/one_stage/served.py:103` | |

SPOT's `verity/registration/v1|v2` is named outside the proofs' group as a literal string [read by me]: the policy
`inference_only/verifier/nci/lean/NciCheck/Baseline.lean:106-107` (`FORMAT`, `FORMAT_V2`), PoUW
`pouw/verifier/lean/check/PouwCheck/Inputs.lean:37` (`formats`) and `PouwCheck/Credit.lean:211` (`COMMIT_FORMAT`),
compose `remote/verifier/CoreRemote.lean:212,238`, and the table `verity/commitments/domains.py:175-176`
(`REGISTRATION_V1`, `REGISTRATION`). The policy's Python reference imports SPOT's
`verity.protocols.spot.verifier.one_stage.registration.digest` (`inference_only/verifier/nci/lean/scripts/nci_reference.py:497`).
`rg -n '^\s*(public )?(structure|inductive|abbrev|def) Registration\b' -g '*.lean'` counts 26 Lean definitions named
`Registration` (`security/` 7, `verity/` 19), and `rg -n '^class \w*Registration\b'` 4 Python classes (`site` 1, `poub` 2,
`spot` 1) [counted by me].

## 5. `security/`: statements that read another area or a code package

Areas are the directories under `security/Definitions/`, `Properties/`, `Proofs/` (a top-level file is its own area).
Import lines by `/tmp/sw/secimp.py` (21,608 import lines in all three trees) [counted by me]. Flock and Spot are the other
sweep's; they appear here only where another area reads them.

### 5a. The lock's own rules [read by me]

- `security/lean-audit.json` `layers`: 47 entries, all over the `Definitions` tree; each names the modules a layer may
  import, an `except` list of modules held apart, and `sealed`. Cross-area imports the layers allow (an entry's
  `imports` outside its own area): `Definitions.Core` → `Definitions.Accounting.Device`; `Definitions.Nci` → `Core`;
  `Definitions.Pouw` → `Core`, `Claims`, `Hash`; `Definitions.Pous` → `Core.Game`; `Definitions.NetworkCertifier` →
  `Claims`, `Hash`; `Definitions.Accounting` → `Core`, `Pouw`, `Pous.Game.Prob`, `Pous.Model.Audit`,
  `Pous.Model.SampledAudit`, `Pous`, `Hash`, `Claims`; `Definitions.Accounting.Guarantees` → `Core`, `Nci`,
  `Nci.Assumptions`, `Pouw`, `Pous.P2.Cost`, `Pouw.Assumptions`, `Pous.Assumptions.P2`, `Pous`, `Pous.Assumptions`,
  `Claims`, `Assumptions`; `Definitions.Nci.Guarantees.FitRate` → `Core`, `Accounting`; the three
  `Definitions.Accounting.JointBridge.CommOf*` → `NetworkCertifier.Period`, `.Tap`, `.PooledLeak`, `.Checked`,
  `.Witness`; `Definitions.Spot` → `Core`; `Definitions.WeightSecurity` → `Core`, `Spot.Dataflow`;
  `Definitions.Assumptions` → `Hash`, `Core`, `Accounting`, `Nci`, `Pouw`, `WeightSecurity`.
- Its `except` lists (non-empty): `Definitions.Nci.Guarantees` except `FitRate`; `Definitions.Pouw` except
  `PearlC.Defs`, `Lead.Model`, `Lead.Deployed`; `Definitions.Accounting` except `Bandwidth.Game` and the three
  `JointBridge.CommOf*`.
- `trees`: `definitions` = `Definitions`, `properties` = `Properties`, `proofs` = `Proofs`, `Proofs.Properties`; its
  `except` has 28 entries, 21 of them `Proofs.Flock.*`; the 7 others: `Proofs.Accounting.TS.Session`,
  `Proofs.Accounting.JointBridge.Guarantees.IntegrityOfIntegrityPeriod`, `…IntegrityOfIntegrityBlock`,
  `Proofs.Pouw.OnsiteFreed`, `Proofs.Accounting.ReadUnit.Guarantees`, `Proofs.Properties.Verity.Recursive`,
  `Proofs.Properties.Verity.Sampled`; each reason says a statement reads C-Flock's (or the toy instance's) definitions
  "still in Proofs".
- `reads_exempt`: `Flock`, `Randomness`, `PouwWindow.Rounds`, `PouwCheck.Rounds` (the last two "read through remote's
  PoUW period component").
- `security/Properties/lean-audit.json` `layers`: 61 entries over `Properties`. Those whose `imports` include a code
  package of another protocol than their own: `Properties.Accounting.Compose` and `Properties.Accounting.Served` →
  `Spot`; `Properties.Challenger` → `CoreProtocol`, `Spot`, `Flock`, `Lms`; `Properties.Nci.Record` →
  `Flock.TemplateQuery`, `Spot.Wires`, `Spot.Bits`; `Properties.Verity` → `CoreOnsite`, `CoreComposites`, `Flock.Hash`,
  `Flock.CoinTree`, `Flock.Hm96`, `Lms.Verify`; `Properties.Pouw.Lead.V2` → `CoreProtocol.Protocol`. Non-empty `except`:
  `Properties.Pouw` (11 modules), `Properties.Pous` (3), `Properties.Accounting` (7), `Properties.Accounting.Remote`
  (7), `Properties.Spot` (10). `security/Proofs/lean-audit.json` `layers`: 1 entry (`Proofs.Pouw.Dimension.Lifting`,
  Mathlib only).

### 5b. Definitions tree, from my own scan [counted by me]

No file of `security/Definitions/` imports a code package, a `Properties` module or a `Proofs` module. Area → other
area, by files: Accounting → Pouw 25, Core 22, Pous 12, Hash 3, Nci 3, NetworkCertifier 3, Claims 2, Assumptions 1;
Pouw → Core 13, Hash 5, Claims 2; Nci → Core 12, Assumptions 3, Accounting 1; Core → Accounting 3; NetworkCertifier →
Core 4, Claims 2, Hash 1; Pous → Core 1; Hash → Core 5; Claims → Core 3; Assumptions → Core 2, Accounting 1, Hash 1, Nci
1, Pouw 1, WeightSecurity 1; WeightSecurity → Core 2, Assumptions 1, Spot 1; Flock → Core 5, Claims 3, Hash 1; Spot →
Core 4.

The ones against the page's direction (accounting below policy; core below accounting) [read by me]:

| File:line | Imports |
| --- | --- |
| `security/Definitions/Core/Site.lean:3` | `Definitions.Accounting.Device` |
| `security/Definitions/Core/Blocks.lean:4` | `Definitions.Accounting.Device` |
| `security/Definitions/Core/Accel/Card.lean:4` | `Definitions.Accounting.Device` |
| `security/Definitions/Accounting/Guarantees/Actual.lean:5,6` | `Definitions.Nci.Reads`, `Definitions.Nci.Assumptions` |
| `security/Definitions/Accounting/Guarantees/Actual/Dampener.lean:4` | `Definitions.Nci.Dampener` |
| `security/Definitions/Accounting/Guarantees/Actual/Threaded.lean:4,5` | `Definitions.Nci.Assumptions.Threads`, `Definitions.Nci.Dark` |
| `security/Definitions/Nci/Guarantees/FitRate.lean:3,4` | `Definitions.Accounting.Actual`, `Definitions.Accounting.Physical` (the layers' `except`) |
| `security/Definitions/WeightSecurity/Tied.lean:3` | `Definitions.Spot.Dataflow` |

Guarantee modules (`…/Guarantees/…`) of Accounting reading a protocol's area [read by me]: `Accounting/Guarantees/BoundReceipt.lean:5-7`,
`Bridges.lean:5-6`, `FpG1.lean:5-6`, `PerRun.lean:5-7`, `Pouw.lean:4-5`, `TS/FoldFanIn.lean:5`, `TS/LowBit.lean:4` → Pouw;
`Memory.lean:4`, `RemoteDraw.lean:5`, `Responder.lean:5-7` → Pous; `Actual*.lean` → Nci (above).

### 5c. Properties tree, from my own scan [counted by me]

Each Properties area reads its own protocol's code package (Pouw → `pouw` 11 files and `CoreProtocol` 1, Pous → `pous` 7, Spot → `spot` 12,
Nci → `inference_only` 7, NetworkCertifier/NetworkPeriod/NetworkScan → `certifier` 7, Flock → `flock` 5). Reads of
another protocol's code, or of the interface, compose or shared code [read by me]:

| File:line | Imports | Package |
| --- | --- | --- |
| `security/Properties/Accounting/Remote.lean:3,4`; `Remote/Line/Record.lean:5`; `Remote/Live.lean:3`; `Remote/Pooled.lean:4`; `Remote/Schedule.lean:4`; `Remote/Shared.lean:4`; `Remote/Subprotocol.lean:3` | `RemoteAudit.*` | poub |
| `security/Properties/Accounting/Requirement.lean:4` | `CircuitRequirement` | definitions |
| `security/Properties/Pouw/Lead/V2.lean:3` | `CoreProtocol.Protocol` | interface |
| `security/Properties/Core/Period.lean:3`; `Core/Site.lean:4` | `CoreProtocol.Period`, `CoreProtocol.Site` | interface |
| `security/Properties/Nci/Record.lean:7-9` | `Flock.TemplateQuery`, `Spot.Wires`, `Spot.Bits` | flock, spot |
| `security/Properties/Challenger.lean:3` | `Spot.OneStageZk` | spot |
| `security/Properties/Composition.lean:3` | `CoreComposition.Tie` | compose (remote) |
| `security/Properties/Remote.lean:3` | `CoreRemote` | compose (remote) |
| `security/Properties/Verity/Game.lean:3,14-17` | `CoreComposites`, `Flock.CoinTree`, `Flock.Hash`, `Flock.Hm96`, `Lms.Verify` | compose (onsite), flock, commitments |
| `security/Properties/NetworkCertifier/Check/LinkWindow.lean:4` | `Lms.Sign` | commitments |
| `security/Properties/Attest.lean:3` | `Attest.Period` | commitments |
| `security/Properties/Hash/Identity.lean:3` | `Identity.Tagged` | commitments |
| `security/Properties/Core/Evaluate.lean:3` | `Evaluate.Prims` | definitions |
| `security/Properties/Firewall.lean:3` | `Firewall.Release` | recursion |

Properties area → another area's Properties [read by me]: `Accounting/Compose.lean:4` → `Properties.Spot.Draw`;
`Accounting/JointBridge/ServiceSoundOfSessions.lean:4` and `WorkOfH1CheckedCreditPeriod.lean:4` → `Properties.Pouw.Period`;
`Accounting/JointBridge/SpaceOfPeriodHeld.lean:3` → `Properties.Pous.Period`; `Pouw/Credit.lean:5` and
`Pouw/PearlPlusDraw.lean:4` → `Properties.Spot.Draw`; `Attest.lean:7` → `Properties.Pous.MemoryChallenger`;
`NetworkCertifier/Check.lean:4` and `Pous/EndToEnd.lean:7` → `Properties.Hash.Identity`; `Remote.lean:5-8` →
`Properties.Composition`, `Properties.Nci.Desc`, `Properties.Nci.Baseline`, `Properties.Spot`;
`Verity/Game.lean:4-6` → `Properties.Composition`, `Properties.Pouw.PearlPlus`, `Properties.Flock.Circuit.CircuitOf`;
`Spot/HistoryTable.lean:5` → `Properties.Verity.Sha512`. No Properties file imports a `Proofs` module.

The Properties package requires 17 code packages by path (`security/Properties/lakefile.toml`); the Proofs package
requires 6 more by path besides `security` and `security_properties`: `flock_verifier`, `certifier_program`,
`core_randomness`, `core_vstar`, `core_vstar_recursive` (`security/Proofs/lakefile.toml`) [read by me].

## 6. Readers of `verity/claims`

`verity/claims/__init__.py` (623 lines) defines `ENTRIES` (87 bare ids), `PROPERTIES` (8), `INSTANCES` (15), so 120
composed `<property>/<instance>` ids, and `get` (loaded with `importlib`, `len(...)`) [counted by me]. The listing marks it
`trusted`, with `loaded_by = ["verity/commitments/tests/test_hm96.py"]` (`tests/import_order.toml`), and
`tests/test_boundaries.py:79-80` makes it stdlib-only [read by me].

| Reader | File:line | What it takes [read by me] |
| --- | --- | --- |
| SPOT's challenger (proofs; device) | `verity/protocols/spot/device/challenger.py:219` (`from verity import claims`), used at 378 | `claims.INSTANCES` (all 15), to accept a coin source's `uniform/<instance>` claim (`coin_claim`, 371-381) |
| PoUS's verifier | `verity/protocols/pous/verifier/protocol.py:30-31,93` | `Entry`, `get`; and a second table `LOCAL_CLAIMS` (line 45, 9 entries, "Claims a certificate cites that `verity.claims` does not hold yet", 86-87) |
| PoUW's provers (no import) | `verity/protocols/pouw/prover/schemes/pearl_c.py:526`, `pearl_kw.py:629`, `pearl_c4.py:1145`, `ncp.py:242` | claim ids as literal strings in `claims=(…)` (e.g. `"tt-out"`, `"random-oracle"`, `"cr/blake3"`, `"live-verifier"`); `pouw/verifier/protocol.py:109` says they "are `verity.claims` ids, or ids in the same format that no render cites yet" |
| tests | `spot/verifier/tests/test_coin_claims.py:13,22`; `poub/verifier/tests/test_ledger.py:27,243`; `commitments/tests/test_hm96.py:153,158,254-257`; `pous/verifier/tests/test_pous_protocol.py:17` | `get(id)` and its `.name`/`.kind` |
| docstrings (Python) | `commitments/crypto/lms.py:3,19`, `crypto/aead.py:17`, `commitments/gates/lms.py:13`, `pous/device/memory_challenger.py:407` | prose naming `euf-cma/lms-sha512-n64`, `faithful`, `prf/sha-512` |
| the vLLM adapter's lints | `adapters/vllm/conftest.py:41` and 5 test files (`tests/lint/test_p01_core_abstractions.py:46`, `tests/program/*boundar*.py`, `test_moe_fp8_definition_imports.py:64`) | `"verity.claims"` as an allowed import prefix |
| `tests/test_provider_names.py:5,22` | scans `verity/claims/` text | |
| Lean files citing an id in backticks or quotes | 308 files, 915 citations (a script over every `.lean`, matching `` `id` `` or `"id"` against the 207 ids) | most: `security/Proofs/Flock` 135 files, `security/Definitions/Accounting` 61, `security/Proofs/Accounting` 19, `security/Definitions/Pouw` 16; in `verity/`: pouw 3, pous 3, spot 2, flock 2, hardware 1, certifier 1. Most cited: `cr/sha-512` 216, `ecr/sha-512` 167, `live-verifier` 103, `trusted-host` 78, `uniform/io-getrandombytes` 36 |
| locks | `security/lean-audit.json` (177 occurrences, 19 distinct ids), `security/Properties/lean-audit.json` (24, 9), `security/Proofs/lean-audit.json` (11, 8) | ids inside text fields (a regex over each lock for the 207 ids as whole words); the locks' own `claims` key is a different thing (Lean `Claims` modules and their witnesses) |
| TOML files naming the path | 29 (`rg -l verity/claims --glob '*.toml'`): 10 under `benchmarks/`, 12 pyprojects under `verity/` and `tools/` (e.g. `verity/commitments/pyproject.toml:10`), `tests/import_order.toml`, `tests/tree.toml`, `tools/verity/check/drive.toml:76`, `tools/move/` 2, `inference_only/rho/data/premises.toml` | a suite's declared inputs, the listing, the layout |

## Not read (whole sweep)

- `main` moved from `9182900fb9` to `b3902993ca` (07:46 UTC) while I read: 1,519 files changed, one rename
  (`adapters/vllm/…/descriptor_equivalence.py`) and three deletions (two adapter files,
  `certifier/verifier/network/README.md`), none of them a path cited here. I did not re-read at `b3902993ca`.
- `tests/test_verifier_cost.py`; `tests/test_boundaries.py` beyond lines 70-126; `tests/tree.toml`.
- Python relative imports (`from . import`) were not followed; Python loads by path were found only by text (`rg`).
- In 3b, the `torch`, `vllm` and model-family counts are not split into code and prose beyond the rows named.
- `verity/protocols/inference_only/nci/models.json`, `rho/data/models.toml`, `verity/definitions/silicon/models.py`
  (GPU semantics) were not opened.
- 4e and 4f list definitions by name; I did not read every `root`/`derive` to say which compute a commitment's root or a
  challenge.
- 5: the `Proofs` tree's cross-area imports were counted (in `/tmp/sw/sec.tsv`) but not tabulated; statements were
  taken as "the files of a `Definitions` or `Properties` area", not each guarantee in the lock's `guarantees`.
- The proofs group (`flock`, `spot`, `recursion`) beyond where another group reads it: the other sweep's.
- Notion's "SPOT security definitions" page, Linear, Slack: not read.
