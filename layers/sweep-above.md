# SWEEP-ABOVE: the layers above the proofs, as `main` holds them

**Status: items 1 to 3. Items 4 to 6 follow in the next push.**

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

## Not read yet (this push)

Items 3 to 6; `tests/test_boundaries.py`, `tests/test_integration_boundaries.py`, `tests/test_verifier_cost.py`;
relative Python imports (`from . import`) were not followed across directories (they stay inside a package).
