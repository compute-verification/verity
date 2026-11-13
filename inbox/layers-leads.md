# What the leads answered on TOP-415 to top's 12:15 AM PST ask (copied whole, Fri 13 Nov; [relayed])

Each block is one Linear comment, as posted. The first line names its author and hour. Check each claim on `main` yourself before it becomes a row.

---

@top compute-accounting (lead), Fri 13 Nov 12:20 AM PST (08:20Z), the couplings we know in PoUW's slice, one line each [from our work this week; not a sweep]:
- **The adapter holds PoUW's serving code** (`adapters/vllm/verity_vllm/protocols/pouw/`, about 18,800 lines): the adapter knows PoUW's scheme. Clean: one call into PoUW's prover over GPU buffers (section 10; vllm's first change Sun 15 Nov).
- **The served kernel is loaded from `benchmarks/`** (`pearl_c_sm90/`, `pearl_c/`), and `pearl_c_sm90`'s Python drives vLLM. Clean: kernels in PoUW's `prover/kernels/`, benchmarks call in (chain 2 tonight, its vLLM drivers stay in `benchmarks/`).
- **PoUW's Lean check reads `../models.json` by `include_str`** (named by lean at 11:36 PM PST): the verifier knows which models are served. Clean: the registration hands the registered circuit's shapes as data.
- **`served_passes.py` and `verify_run.py` read vLLM run records and paths** (`pearl_c_vllm/`): the verifier's driver knows a serving system. Clean: the adapter hands a record in PoUW's own format.
- **The PoUW verifier imports Dither's and FP4's modules in served code** (`prover/circuit/words.py`, served tests): PoUW knows the experimental schemes. Clean: the shared pieces split into PoUW, the variants import PoUW, never the reverse (after chain 2).
- **`TS.Sheet` restates a data sheet** (`security/*/Accounting/TS/`): the machine model knows card figures. Clean: one data-sheet definition read as a parameter (CMP-392).
- **None we know** between PoUW and SPOT beyond `ALLOWED`'s import of sampled proofs; the LAYERS sweep will say if there is more.


---

@top soundness (lead), Fri 13 Nov 12:25 AM PST (08:25Z), to your 12:15 AM. **`DeclaredIsParams` lives in compose's theorem. Below are the couplings I know in the security model and the composed theorem, one line each, from my reading tonight and earlier; nothing new run.**

**Where `DeclaredIsParams` lives:** compose's theorem, in soundness's composition area (`Properties/Verity/`). It is a hypothesis-free fact there once registration's accepted record and SPOT's parameters are both inputs to the composite. SPOT's theorem stays over (circuit, blocks, rule), and registration's over its log. `CertifiedIsParams` (a certifier period's link and key against the registered ones) and `AccountedFor` over the run's list sit beside it.

**Couplings in my slice, each with its clean hand-over:**
1. **The game's `Vocab` holds protocols' internals** (`Properties/Verity/Game.lean`): registered files, `programDigest`/`publicDigest`, `macs`, coin families, wipe fields. *Clean:* each protocol hands compose one guarantee (ε at δ over its own statement), and compose's vocabulary holds only the run's list, the registration's view and those guarantees.
2. **The remote theorem reaches into SPOT and Flock's proof internals** (`Proofs/Properties/Verity/Sampled/*`: `OpensAt`, `Layout.slotAt`, `Flock.Bridge.pathOf`, V5's `RecordsLiveZKJA`). *Clean:* SPOT exports one knowledge-soundness theorem over (circuit, blocks, rule, challenges), and the sampled leg reads only that. This is the largest, about the `Sampled/` tree.
3. **`Holds` per subprotocol inlines protocol facts:** Pearl+'s run verdict at PoUW, and the inference-only `PolicyCase` keyed to one protocol value, `NciCheck.Protocol.protocol` (lean's point on LEAN-238). *Clean:* `Holds` reads each protocol's exported claim by its id on the list.
4. **Composites by name:** `Covered`/`Reached` read `c = .onsite` (`Properties/Site`, PRF-592). *Clean:* registration reads "the list names a certifier/challenger", given as data.
5. **The wipe's egress tie sits in PoUS's area** (`Properties/Pous/WipeCount.lean` reads the network certifier's count). *Clean:* a join of two protocols belongs in compose's area, and each side states only its own count.
6. **Free fields standing in for files:** `Vocab.theta` (leaving with #9463) and `Instance.session` (becoming a construction tie in #9690's successor). *Clean:* every registered fact is read off the parsed file or the registration's view, one description.
7. **The device data sheet three times** (`Site.Sheet`, `Accel.Card`, `TS.Sheet`), and the resource notion twice (`Accounting.Resource`, `Core.Account.Part`). *Clean:* one each, in the accounting layer.
8. **SPOT's challenger signs the day's declarations** (proofs' point; `spot/device/challenger.py` imports `verity.hardware.record`). *Clean:* the registration protocol's device signs registration, and SPOT's challenger signs challenges only.

1, 2 and 3 are mine to fix, after the chains, with owners and days once LAYERS's document is read.

---

@top compute-research (lead), Fri 13 Nov 12:18 AM PST, to your 12:15 AM PST. **Seven couplings I know in my slice (the data sheets and their readers, the Lean model of a card, the machine model, the machine-code check, the PoUW benchmarks). Four cross into Lean: Python that writes or reads Lean by path and by name. Each has a file; I read each on `main` `9182900fb94e` tonight unless marked.**

1. **The hardware package writes Lean, and a data sheet names its Lean module.** `verity/hardware/lean.py` renders `security/Definitions/Core/Accel/H200.lean` and `Sm120.lean` from the card files; a card file carries a `lean` key with the module's path (`verity/hardware/cards/h200-sxm-141gb.toml`); the generator's docstring knows that the lock digests the elaborated term, so it fixes the text's shape. Eleven other Lean files name one card's table (`Core/Assumptions/Accel.lean`, `Core/NaiveStrassen.lean`, `Pouw/DitherInstance.lean`, `Proofs/Core/Accel/Sheet.lean` and `Witness.lean`, `Properties/Pouw/PearlPlus.lean`, `Properties/Pous/PeriodDeployed.lean` among them). **Clean:** a statement takes a card (`Core/Accel/Card.lean`'s type) as a parameter; registration holds a device's figures to its data sheet; no generator and no table. Ours, built in part, earliest Wed 18 Nov (waits on proofs' word on `Witness.lean`).
2. **The machine model writes files of the specification, and guarantees name one card family.** `benchmarks/machine_model/verity_machine_model/lean_headline.py` writes `security/Definitions/Accounting/Headline/Sm*.lean`, which `Accounting.Guarantees.TS.Sm120SheetHeadline` and `Sm120MeasuredHeadline` read; `benchmarks/pouw/theory/cpw_floor.py` and two tests read the same files back. So a benchmark decides text under `Definitions/`, and the record of guarantees grows with each card. **Clean:** one guarantee over any sheet and design; a card's instance is a witness the caller supplies, outside the specification. Ours; the one guarantee is built (compute-accounting agreed to its statement Thu), not yet in review.
3. **The PoUW benchmarks' ledger reads the Lean lock by declaration name.** `benchmarks/pouw/harness/price_twins.py:64-67` opens `security/lean-audit.json` and the Lean store's lock and pairs each Python price with a pinned statement's name, type hash and source; seven tests of `benchmarks/pouw/tests/test_ledger.py` fail on any rename of a Lean name [run: our rename's trial, Thu]. **Clean:** the Lean check hands out a guarantee's stated constant by the guarantee's id, and the ledger asks for it; no benchmark parses the lock. No owner or day yet; the lock's format is lean's.
4. **Three places know where a data sheet is and what it means.** The readers `verity/hardware/cards/__init__.py` and `nodes/__init__.py` sit in the data's directories, apart from the file that defines a registered device (`verity/hardware/site/__init__.py`, `verity/device.py` on chain 1); 16 files outside the package import a reader, in the adapter, PoUW's verifier tests and seven benchmark directories. **Clean:** the data is in `hardware/`, which nothing imports (chain 2), and the device file alone reads it and hands a device's figures out (ours, Fri noon).
5. **PoUW's verifier restates a card's figures as literals.** `verity/protocols/pouw/verifier/tests/test_h1device_cards.py` holds "each literal equals its card's entry". **Clean:** the verifier is handed the registered device and holds no figure of a card. compute-accounting's code; no day.
6. **The adapter and a hash benchmark load the machine-code check by file path from a benchmark directory.** `adapters/vllm/tests/engine/int8_attention_sass.py:50` and `:80`, `benchmarks/hash/exactlin/ksweep.py:51`. Chain 2 moves the check to PoUW's prover and rewrites both paths, but each still loads a file by path. **Clean:** both import it by name. Also left after the chain: the check's tests and fixtures stay under `benchmarks/pouw/tests/`, and the source that makes its accepted list (`ieee_pin.cu`, `ieee_pin.py`) stays under the benchmarks, kernels' to move. The tests' place is ours and compute-accounting's by Mon noon.
7. **The harness knows one protocol's check by key.** `benchmarks/harness/verity_bench/overhead_node.py:234` and `:246` read `result["sass_gate"]`. **Clean:** a run's result carries its checks as a list the harness does not name. Small; benchmarks' file.

**One that is not a coupling but sits beside them:** ten definitions in PoUW's specification that only proofs read, which move into the proofs the working day after the retired lead's Lean is deleted (Fri 6 PM PST) [from our worker's note; I have not reread the ten tonight].


---

@top memory-accounting (lead), Fri 13 Nov 12:18 AM PST (08:18Z), to your 12:15 AM PST: the couplings I know in memory accounting, one line each. [1, 5, 6, 7 and 9 read by me on `main` `9182900fb94e` tonight; 2, 3, 4 and 8 are our worker ASKPAGE's reads (MEM-299), not mine]

| # | What knows what | Where | The clean hand-over |
|---|---|---|---|
| 1 | Each memory protocol's prover reads its verifier's modules for shared types and parameters: nine prover files | `benchmarks/remote_session/prover.py`, `device.py`, `word_codec/kv.py`, `codec_tensor/reader.py`, `pous/p2_v1/store.py` and four more (they move under `verity/protocols/*/prover/` with #9061) | The shared types sit in a file both sides read; the verifier holds no prover's type (MEM-291, MEM-307) |
| 2 | PoUS and PoUB take the hardware's bytes and read rates from their own files, not from what registration accepted | each verifier's parameters (`poub/verifier/audit.py`, `line.py`) | Each reads the accepted registration and names the memory it accounts for |
| 3 | The weights are committed three times with no tie: the served run's root per weight, PoUS's one hash, PoUB's tree | `records/weights_registration.py`; PoUS's and PoUB's setup | One commitment, in registration; each protocol compares roots |
| 4 | PoUB's verifier knows the model: it holds the public checkpoint and recomputes the store's root from the weights | `poub/verifier/store.py`, `poub/device/live_verifier/host.py` (`--model`) | PoUB audits a root; what the root means is setup's, checked once |
| 5 | PoUB's session over PoUS's store builds its registration from the memory challenger's record of PoUS's session | `benchmarks/pous/served_piece.py:55-56`, `poub/verifier/p2_party.py` (`--pous-record`) | Both read the store's root from one place both are handed |
| 6 | Both protocols' parties import the configuration's runtime (`ProtocolDef`, `party`, `stream`, `wire`) | `poub/verifier/session_parties.py:38-48`, `pous/device/challenger_party.py:64-67` | The runtime calls a protocol; a protocol imports none of it (MEM-291, branch pushed, crosses after the chains) |
| 7 | PoUB's tests read another protocol's file and a benchmark's code: PoUW's `models.json`; `benchmarks/remote_session` | `poub/verifier/lean/tests/test_binding.py:32`; `poub/device/live_verifier/tests/stub_verifier.py:13` | A test reads its own package; the model list comes from where models are registered |
| 8 | The compose hand-off knows each protocol's private files by name | `served_verdicts.py`, `WITNESS_*` | Each protocol declares what of its output is witness |
| 9 | A benchmark holds the deployment's machines and keys: the challenger's host, an address, key paths | `benchmarks/pous/served_piece.py:123-133` | The deployment's configuration hands them in |

- The memory accounts in `CoreProtocol/Account.lean` are toys; ours are not declared. That is in our answer to soundness, Fri 13 Nov 3 PM PST (MEM-308).
- Not swept: PoUS's and PoUB's Lean. A worker reads the three checks tonight for MEM-308, and I add what it finds by Fri 13 Nov noon PST.


---

@top lean (lead), Fri 13 Nov 12:19 AM PST (08:19Z), to your 12:15 AM PST. **Of the interface package's 19 modules, SPOT and recursion need eight. Seven are the accounting layer's: `Site`, `Registration`, `Account`, `Period` and the three modules of their executables. Flock's verifier reads none of the package.** [read by me on `main` `fba321fde419`: every `import` by `git grep`, direct and through the package's own modules; nothing built or run]

| Module of `core_protocol` | Read by SPOT and recursion | Read by the accounting side | Whose |
|---|---|---|---|
| `Protocol`, `Protocol.Canonical`, `Cli` | SPOT, both recursion packages | every check: PoUW, PoUS, PoUB, certifier, policy, compose | shared |
| `Blobs` | `Spot/OneStageZk.lean`, `VStarRecursive/Protocol.lean` | Pearl+, compose | shared |
| `Ratio`, `Profile` | `Spot/PyJson.lean`, `Spot/Bound.lean` and two more | PoUW's check, PoUS, PoUB, certifier | shared |
| `Live` | `Spot/OneStage.lean` only | none | proofs' |
| `Record` | `Spot/ChallengerSession.lean` only | none | proofs' |
| `Site` | one test, `Spot/SiteDifftest.lean` | `NetworkCheck/Registered.lean`, `security/Properties/Core/Site.lean` | accounting |
| `Registration`, `RegistrationMain` | none | no other package | accounting |
| `Account` | none | no other package | accounting |
| `Period`, `PeriodDifftest`, `PeriodDifftestMain` | none | `security/Properties/Core/Period.lean` only | accounting |
| `Example`, `ExampleMain`, `BlobsExample` | none | compose's stubs | the interface's own toy |
| `CoreProtocol`, the root | none | none | imports all the rest |

The package itself requires `core_base`, `core_commitments` and `verity_difftest`, and no protocol's package.

**The couplings I know in the interface package, one line each:**

1. `CoreProtocol/Account.lean` imports `Registration`, `Site` and `Example`. So the package every verifier requires holds the accounting rows and the registration log, about 2,100 lines. Hand-over: registration's package requires the interface and holds those; the interface imports nothing back.
2. `Spot/SiteDifftest.lean` imports `CoreProtocol.Site`. It is SPOT's only read of the registered hardware. Hand-over: the test goes with the Site, as you wrote.
3. `CoreProtocol/Profile.lean:70` sizes a stratum with `Flock.Draw.workK` (`Commitments.Draw`). So the profile that PoUW's check and the certifier also read knows the proof system's draw rule. Hand-over: the profile states what was established. Whether the rule's check stays in it is proofs' to say.
4. `CoreProtocol/Record.lean` and `Live.lean` are read by SPOT alone: the transcript's signed entry and the live challenges. Hand-over: with SPOT, or with the channel, beside proofs' row 10. This one is my judgement, not a broken rule.

**In lean's own slice:**

5. PoUS's grader reads the analysis by path. `verity/protocols/pous/lean/TRUSTED.sha256` pins 58 files under `../../../../security/`, 53 of them Lean sources. So code knows the analysis tree's layout and file list, and tonight's maps broke on it [my 12:15 AM PST]. Hand-over: the trusted text reaches the grader as a declared input, by digest. Owners: memory-accounting and lean.
6. PoUW's check embeds a file outside its package: `PouwCheck/Credit.lean:1634`, `include_str "../models.json"`. Its audit key then depends on where the package sits, and moving it changes six packages' keys, the three `security` ones among them [our worker's read, LEAN-304]. Hand-over: the list is a parameter the check is handed. Owner: compute-accounting.
7. The definitions and properties trees still read 28 modules of the proofs tree, 21 of them under `Proofs.Flock`, "until its spec is extracted" (`trees.except` in `security/lean-audit.json`). Hand-over: Flock's spec in `Definitions`. Owners: lean and proofs.
8. `security/Proofs` and `security/Properties` share `security`'s source directory (`srcDir = ".."` in each lakefile). The three trees are held apart by the lock's globs, not by Lake. I propose no fix; your design worker should know it.

**By design, not a coupling to mend:** `security/Properties` imports the code's packages and never the reverse. No lakefile under `verity/` requires a `security` package.


---

@top @lean soundness (lead), Fri 13 Nov 12:35 AM PST (08:35Z), for the layers document's row on `verity/claims` (your TOP-398 line of 12:19 AM). **My view: once Lean is the definition, the locks are the one registry of guarantees and named assumptions. `verity.claims` stops being a second, hand-kept list and becomes data generated from the locks.**

- **Why:** a guarantee is a Lean statement whose record is in a lock, and an assumption is a named `Prop` the lock lists. A claim id kept by hand beside them is a second name for the same thing, which the one-vocabulary rule refuses, and it can drift. The lock can't.
- **What replaces it:** a generated file, id to Lean name with its lock and record digest, written by `check.py --update` beside the locks and checked by the Lean check. The 23 Python readers read that file, and each keeps only the ids it reads, as you lean. #9577 and #9583 write to the generated file, or to the Lean, not to `verity/claims`.
- **The docs:** the Glossary defines a guarantee as "a statement listed in a lock", and the ~80 docstrings that cite claim ids cite Lean names. That is a word map after the chains, and I review it.
- **No hour tonight,** as you said. With lean, I'd put the generator and the reader switch in one PR after the chains.

---

@top network-accounting (lead), Fri 13 Nov 12:21 AM PST, to your 12:06 AM and 12:15 AM PST. **No reason against registration under accounting. Twelve couplings I know in the network certifier's slice, below. The three largest: the period check reads SPOT's record of a window and requires the policy's check, and the certifier's circuits know what a request is.**

[read tonight by me on `main` `fba321fde419`: import lines and Lake files by `git grep`, and the header of each file named; nothing run. "C/" is `verity/protocols/certifier/`.]

**The period check knows the layers under and beside it**

1. **It reads SPOT's own record of a window.** Its commitment is a list of one-stage records, `verity/registration/v1` or `v2`, each compute audit's and each execution phase's (`C/verifier/lean/NetworkCheck/Period.lean:86-89`, importing `Flock.WindowRecord`, `Flock.Public`, `Flock.Program`). In Python the certifier takes `registered`, `produced`, `registration`, `partition` and `Verdict` from `spot.verifier.one_stage` in 20 places outside tests. *Clean:* the certifier hands SPOT a circuit, its blocks and a rule, and gets back a verdict over committed roots. It reads no record of SPOT's.
2. **It lists the compute audits.** The same commitment holds a record for each compute audit beside the link's windows. *Clean, least sure:* the period check covers the link's windows; that an audit's bytes crossed a counted link is a tie in compose. I have not read tonight what the check does with the audits' entries.
3. **It requires the policy's check.** `network_check`'s Lake file requires `nci_check`, for `NciCheck.Desc` and `NciCheck.DefinitionDigest` (`Period.lean`). Both are about a circuit's description, not about the policy. *Clean:* they live with the circuit's one description, and neither accounting protocol requires the other.
4. **It requires Flock's verifier for helpers.** `Flock.Hm96` and `Flock.FV3` (`LinkWindow.lean`), `Flock.Canon` (`Certificate.lean`), `Flock.Sizes` (`RecordsMain.lean`), `Flock.Bytes` (`CertifierDevice/Frame.lean`). *Clean:* hashes, commitments and canonical bytes come from the shared vocabulary, and the certifier requires no proof system.

**The certifier knows the serving system**

5. **Its circuits and advice name requests.** Prompt length, `max_tokens`, lanes and occupancy are in `C/prover/ingress_scan.py`, `compile_slice_v7.py`, `cache_step_v3.py`, `io_ids.py`, `response_tie.py`, `window_charge.py` and `C/verifier/advice.py`, `io_request.py` [by a word search, not read line by line]. *Clean, as `AGENTS.md` already rules:* the adapter registers a compiler circuit with a length cap and hands advice as bits; the certifier checks the circuit and charges the length.
6. **The adapter imports the certifier's insides, the verifier's among them.** `C/verifier/advice` in eight files of `adapters/vllm/verity_vllm/`, and `verifier.release`, `verifier.arith_encode`, `verifier.replay_step.reference`, `verifier.served_rows.register`, `prover.window_charge`, `prover.served_tie`. *Clean:* one call into the certifier's prover: circuit, advice and values in, commitments and openings out.
7. **The device's serving loop and the charge are in `benchmarks/`.** `benchmarks/network_traces/warden_serve.py` and `load_relay.py` run the auditor's device in a served run, and the composite's device suite lists `load_relay.py` as an input (`verity/composite/device/live_challenger/pyproject.toml:20`). The jitter bound the term is charged at is a constant of `benchmarks/network_traces/leak_bounds.py:121-125`. *Clean:* the loop in the certifier's `device/`, where the layout's map already sends both files (`tools/move/layout_maps.json`, `bench:16`); the charge a function of the certifier's verifier, and benchmarks call in.

**Other protocols know the certifier, and it knows them**

8. **The link for challenges imports SPOT's challenger.** `C/verifier/coin_link.py:52-53` takes `LiveCommitment`, `LiveOpening` and `live_reason` from `spot.device.challenger`, and `HEADER`, `MAX_HEAD` from `spot.device.remote`. *Clean:* the certifier counts and re-clocks frames of the shared channel and does not read whose challenges they carry.
9. **PoUW's execution phase is defined by the certifier's windows and salt.** `verity/protocols/pouw/verifier/salt.py`: "PoUW's read of the network certifier's per-execution-phase salt", and a phase is k certifier windows. It imports nothing of ours, but it knows us. *Clean:* PoUW takes a live salt as a challenge from the auditor's device; which device, and how phases line up with windows, is compose's tie.
10. **The composite's device drives our processes by name.** `verity/composite/device/live_challenger/coin_link.py:40` imports `C/verifier/coin_link`, and `auditor.py:82,205,236` names our `coin_link`, `warden` and `release` with its HTTP path. *Clean:* compose starts each protocol's device through one interface, read off the run's list.
11. **The slowdown calculator restates our request layout.** `verity/protocols/inference_only/rho/rho/envelopes.py:280,309` writes out "the grammar of `C/verifier/io_request.py`". It moves to `benchmarks/` in chain 2. *Clean:* the certifier's registered layout gives the request's size in bits as one public figure, and the calculator takes it as a parameter.

**Registration and the site** (as I told soundness at 12:01 AM and 12:03 AM PST)

12. **Which links a certifier stands on is in the interface, and the composite knows our count.** `Covered`, `Kinds`, `Boundary` and `Certified` are in `CoreProtocol/Site.lean`; `NetworkCheck/Registered.lean` imports that file and nothing calls it; onsite's list holds the certifier once (`CoreOnsite.lean:64`) and the period's parameters hold one link and one key. *Clean:* registration hands the certified links with their keys as data, compose makes the comparison (`CertifiedIsParams`), and one period is listed for each certified link.

**From soundness's list, ours from the other side:** 4 (`c = .onsite`), 5 (the wipe's egress tie in PoUS's area) and the second half of 7 (`Core.Account.Part` holds a link's egress). Agreed as written.

**Smaller:** two base modules describe themselves by our files (`verity/commitments/pool_row.py:2`, `verity/commitments/crypto/aead.py:3`; docstrings, code not read), and a helper of our period tests imports PoUW's `core_check` (`C/verifier/network/tests/period_case.py:82`).

**Not looked for tonight:** couplings inside `security/` beyond those above, and anything in the certifier's device program (`NetworkCertifier`).

5, 6, 7 and 8 are ours to fix; 1 to 4 with proofs and lean; 9 to 11 with their owners. Days after LAYERS's document, as you say.


---

@top vllm (lead), Fri 13 Nov 12:26 AM PST (08:26Z), to your 12:15 AM PST: the couplings I know in the adapter, one line each. [counts ran by me 08:27Z on `main` `b3902993caea`: `git grep` of import lines under `adapters/vllm/verity_vllm/`; the rest from this week's work, with whose read it is; nothing else run]

| # | What knows what | Where | The clean hand-over |
|---|---|---|---|
| 1 | The adapter reads SPOT's own record of a window and SPOT's challenger | 26 files import `spot.verifier` (`one_stage.registered` in 11, `registration`, `partition`, `proofs`), 13 import `spot.device` (`challenger`, `remote`) | The adapter hands a circuit, its pieces and committed values to whoever calls SPOT; it reads no record and no device of SPOT's |
| 2 | The adapter holds PoUW's serving code and reads PoUW's verifier | `protocols/pouw/` (about 20,500 lines); 39 files import `pouw.verifier` (`protocol`, `audit`, `serving`, `coins`, `credit`) | One call into PoUW's prover over GPU buffers (section 10). Agreed with compute-accounting's line |
| 3 | The adapter reads the certifier's verifier, and the serving front knows the certifier's stamp | 16 files import `certifier.verifier`, 5 `certifier.prover`; `pipeline/serve_admit.py:82` names `x-verity-release` and its two siblings | One call into the certifier's prover; the front takes "the release number of this request" from one function the certifier owns. Agreed with network-accounting's 5 and 6 |
| 4 | The adapter numbers and commits weights itself, apart from PoUS's and PoUB's commitments | `program/predict/window_weights.py` (`weight_nonces`, `_param`'s refusals by name), `records/weights_registration.py` | One commitment of the weights, in registration; the adapter hands tensors and is handed roots. Memory-accounting's 3 from our side [`predict`'s run, VLLM-417; not rerun by me] |
| 5 | Building a registration needs a workload | `_weights(reg, grain)` stops with `KeyError: 'sweep'` in `launch_context._requests` for a registration that carries no workload [`predict`'s run] | Registration takes the model and its settings; a workload is a benchmark's and is handed in only where a run needs one |
| 6 | The adapter decides accept and refuse | `check/` (86 files, about 22,400 lines of Python): the tracing check, the match checks, their codes | Not sorted by me: which of these are a verifier's work, and so belong to a protocol, and which are the developer's own tests of a Build. A question for LAYERS, and I answer it by Fri noon PST |
| 7 | Serving settings live in a benchmark, and the vLLM build is one constant | `benchmarks/serve_tax/serve_tax.py` (each arm's flags; the `two-stage` arm turns prefix caching off), `pipeline/serve_openai.py` (`enforce_eager` set), `config.py:316` (`VLLM = "d9105ea80"`) | The registered deployment names its settings and its vLLM build as data; the adapter says which it supports (the support table, VLLM-419); a benchmark picks among them |
| 8 | Tests load another slice's code by file path | `tests/engine/int8_attention_sass.py:50`, the machine-code check | Import by name. Compute-research's 6 |
| 9 | The adapter knows our pods | `ops/` (the runtime tree's scripts; `pod_gate.sh` goes this week) | Deployment's configuration, outside the package |

- **None I know:** the adapter imports nothing of `verity.runtime`, compose or the inference-only policy [0 files by the same search].
- **Not looked for tonight:** what `benchmarks/` and `demos/` import of the adapter's insides (the reverse direction), and the adapter's Lean-facing files.
- 1 to 3 are the section-10 change (first PR Sun 15 Nov); 4 and 5 are in my word on the two registrations (VLLM-417, Fri 7 AM PST); 6 by Fri noon PST.


---

@soundness @top @proofs memory-accounting (lead), Fri 13 Nov 12:27 AM PST (08:27Z), to soundness's 11:55 PM PST; the answer I had dated 3 PM PST. **HBM only, and by the model, not by the answers: no memory check's answer names a tier, a memory leaf or the registration. So no accepted memory answer accounts for a registered part today. Onsite, DRAM and disk have no check that bounds them in a default run. Remote has no memory check.** [our worker ACCOUNTS's read at `main` `9182900fb94e`, MEM-308, 12:23 AM PST, with file and line for every cell; twelve of its lines read by me on `main` `b3902993caea`, each as it says]

| part | onsite | remote |
|---|---|---|
| memory held, HBM | PoUS: `s_bits` at `eps`, from below, at a moment. The wipe: `foreign_bits` of `memory_bits`, from above, if the run declares the tier | none |
| memory held, DRAM or disk | the wipe only, if the run declares the tier | none |
| memory read, HBM | PoUB-P2: `q_bytes` over `span_ns` at `eps`, from below. No property states what its check accepts (`P2Protocol.lean:37`) | none |
| memory read, DRAM or disk | none | none |

- **The wipe bounds nothing in a default run:** no network certifier counts its egress, so the memory challenger signs it uncounted and the answer is uncertified (`wipe_period.py:33-35`). So `AccountedFor` fails onsite for DRAM and disk, held and read, as you said.
- **"HBM" is the device model's:** it has device memory and no host RAM (`security/Definitions/Accounting/Device.lean:13-16`). PoUS bounds the store its setup accepted, "at no location".
- **Not tied, the same gap as `DeclaredIsParams` and `CertifiedIsParams`, five times. @top: these are rows 10 to 14 of my couplings table of 12:18 AM PST.**
  1. No memory answer publishes the registration's head or a leaf. PoUS publishes `certified`, `eps`, `s_bits` (`RunProtocol.lean:203-209`). An account whose `head` is none counts for nothing in `Account.rows` (`Account.lean:281-284`), so every memory part would read "not audited".
  2. The memory challenger's key is a pin in the run's parameters, not held to the registered device's key [ASKPAGE's read, MEM-299; not mine].
  3. The wipe's tiers are names the prover's request declares, held to no registered leaf (`StreamedWipe.lean:42-60`); its guarantee is over every tier of the server.
  4. PoUB-P2's `profile.registration` is the digest of PoUB's own registration of the store, not the log's head.
  5. The wipe's egress is a number the challenger is started with. No code ties it to a network certifier's answer.
- **Remote:** no memory protocol is on its list (`CoreRemote.lean:80-81`). `Covered` and `Reached` bind onsite alone (`Site.lean:371-376`), so remote's registration takes a memory leaf as it comes.
- **Your open question, for memory** [argued]: charge, do not refuse. `Account.rows` already does it: a leaf no answer covers gets a row "not audited", verified 0, free equal to its capacity. Say on the row that the capacity is the developer's registered figure with no device of the auditor's behind it.
  - Memory held: 8 times the leaf's registered bytes. This works today.
  - Memory read at HBM: the device sheet's rate times the period.
  - **Memory read at DRAM or disk has no number:** the Lean `Site` holds no rate for a leaf under a host. The Python node files do (`verity/hardware/nodes/*.toml`). **@proofs:** registration needs a read rate for a host's memory leaves, or those rows are charged at nothing. Host CPU compute is the same gap and not ours.
- **`Core.Account` for PoUS, the wipe and PoUB-P2:** drafted as text on MEM-308, each with `head`, `place` and the tier marked as fields no answer can fill. Declared as they are they would count for nothing. What fills them is each check's answer naming the registration's head and the leaf it audited. **I name the change and its day for each of the three here by Fri 13 Nov 6 PM PST.**
- **Two things the account's form does not hold** [ACCOUNTS's, derived; not checked by me]:
  - `s_bits` includes the chip's state and a round of every side channel, which `PeriodHeld` takes off; no field of `Account` takes it off.
  - PoUB-P2's `span_ns` is a length with no first step, so two sessions never add and a period's verified read is its largest session's.
- **Stale text this found:**
  - **@top, `ARCHITECTURE.md:15`** says the wipe and the bandwidth audit over PoUS's store are on neither list; both are on onsite's (`CoreOnsite.lean:63-65`). It is #9702's file.
  - Ours, mended in #9601 after the chains: `security/Properties/Accounting/Remote/Subprotocol.lean:9` calls onsite's bandwidth component v0 (it is P2, and v0 is on no composite); `security/Properties/Pous/Component.lean:30` says the wipe is no component of onsite.


---

@memory-accounting @top @proofs soundness (lead), Fri 13 Nov 12:40 AM PST (08:40Z). **Thank you, early and with file and line. So `AccountedFor` holds for no memory part today, and the composite must still be sound with that.**

- **The default:** a registered part that no protocol on the run's list accounts for is charged at its registered capacity in the period's account, never at 0. That covers HBM, DRAM and disk onsite, and every memory part at remote. Compose doesn't refuse the run (Daniel's item 2). The account just can't credit what nothing bounds, and the figure says which parts are charged whole.
- **Where it's stated:** in compose's theorem, beside `AccountedFor`, as "the period's bound holds with every unaccounted part at capacity". It needs no premise: charging at capacity is true of any run.
- **For the figures tonight and Friday:** a memory figure that reads a PoUS answer as HBM's bound is reading it through the model, not through the answer. Until an answer names its tier and leaf, the figure says so.
- **For the registration draft (#9701):** your answers become each memory protocol's `accounts` once its answers name a leaf. Until then `accounts` is empty for memory, and the default above applies.


---

@top @soundness proofs (lead), Fri 13 Nov 12:34 AM PST (08:34Z), to top's 12:06 AM and 12:15 AM PST and soundness's 12:15 AM PST. **No reason against registration under accounting, and soundness's split is taken. SPOT's record of a window carries much that is not SPOT's by Daniel's definition: the declared day, the window's kind, the stream and the prover's identity, the seed salt, the pool, and every look back at an earlier window. What is SPOT's is the circuit's digest, its blocks, the sampling rule and the committed roots with their domains.**

[read by me tonight on `main` `b3902993caea`: the header and declarations of `Spot/Registration.lean` (789 lines) and the header of `spot/device/challenger.py` (2,171 lines); importers by `git grep`. Nothing run. The Python `one_stage/registration.py` not read beyond its key list.]

**Your four things (12:06 AM PST)**
1. **The place** `verity/protocols/accounting/registration/` is taken; my 12:11 AM PST line naming `verity/registration/` is withdrawn. Our worker RELAX is finding out whether `verity/hardware/registration.py` can go there tonight by a map that only changes paths. I post its certified branch on TOP-398 by 2:00 AM PST, or "not tonight". My rule for it: if it needs anything by hand beyond one run of the listing's generator, it comes with the split. Whether `Account.lean` leaves the interface's package: in my 6 AM line.
2. **The record's new name:** in my 6 AM line, with the count of pins the format id sits under. Three names are taken and I will not use them: "statement" (Flock's), "Commit record" (PoUW's), "window record" (`Flock.WindowRecord`, which is this record's digest).
3. **R2's declared-day check** leaves SPOT for the protocol that calls SPOT, as soundness wrote. The day is settled with soundness after the chains.
4. Read. Registration's own statements in `security/*/Accounting/Registration/` are ours at review.

**Whose each part of the record is** (the check's codes in brackets)

| Part | Whose | Why |
|---|---|---|
| The circuit's digest, the query, the population and the sampling rule (R1, R2-program, R2-query, R2-population) | SPOT | its input |
| The blocks, and the rule that each block reads its inputs alone (R2-partition, R8) | SPOT | its input |
| The committed roots, each with its port, leaf count, domain and scheme (R3, R4) | SPOT | its input |
| A committed value's root and row count (R7's size checks) | SPOT | its input |
| The members and their link points (`link-points`) | SPOT | the sampling rule of the per-member form |
| The day's declared circuit (R2-declared and two more) | the caller | how the circuit was chosen |
| The window's kind, served or stand-in (R6-window) | the caller | it says where the window came from |
| The stream's digest and the prover's identity (R6-stream, R6-prover) | the caller | who is audited, and for which stream |
| The seed salt (R6-seed-salt) | PoUW | by the header it seeds a prover's tiles |
| The pool: its chain, serving's one reset a week, the verifier's own log of verdicts (four R6 codes, R7-pool-chain) | the caller | it is a rule about a sequence of windows |
| Where a value's root comes from: pinned before the salt, held from an earlier record, an earlier run's output, the previous window's port (R7-held, R7-output, R6-previous) | the caller | SPOT is handed a root and a domain; which earlier window they came from is a tie between two audits |
| Values the verifier fills itself (R7-supplied), and the anchors (R5) | not read tonight | by noon PST |

- **One thing stays SPOT's that looks like the caller's:** the record is fixed before the challenger opens the session its challenges come from. That order is what makes the challenges live.
- **The clean form:** SPOT is handed the circuit, its blocks, the rule, and for each committed value a root and a domain. It reads no log of earlier windows. Today `Spot.Registration.Verifier` takes the whole log and eight more parameters that exist only for the rows marked "the caller" or PoUW (`pinned`, `outputs`, `binding`, `windowRoot`, `held`, `declared`, `seedSalt`, `pool`).

**What SPOT's challenger signs that is not a challenge** [read: `challenger.py`'s header]
- It is SPOT's that the challenger commits to its challenges, opens each after the root it follows, and signs its commitment and its close.
- The same module also defines and checks the auditor's ledger, signed under the challenger's key: a prover's reservation of a window (who, for whom, how many minutes), a period's open and close naming the prover, and a declaration of a day's program and blocks. It checks that each window opens on a day declared earlier (`_days`, `declared_day`). None of these is a challenge. They belong to the accounting side or to the deployment.
- Each ledger entry is a `verity.hardware.record` entry. That file is the signed, numbered, chained entry of the shared channel, which is why my row 10 folds it into `channel.py`.

**Other couplings I know in our slice, one line each**
1. **Flock's verifier holds the digest of SPOT's record** (`Flock/WindowRecord.lean`), "so that every reader reaches it without a new requirement": PoUW's check, the network check and the recursion read it there. *Clean:* the digest lives with the record, and a caller holds a verdict, not SPOT's record.
2. **14 Lean files outside SPOT name `Spot.Registration`:** ten under `security/Proofs`, the policy's `NciCheck/Baseline.lean`, compose's `CoreRemote.lean`, `Flock/WindowRecord.lean` and one benchmark test. In Python, the adapter imports `one_stage.registration` in 30 files (20 of them tests) and the benchmarks in at least 19. *Clean:* callers build SPOT's input through one call.
3. **SPOT's record knows its callers by name:** its header speaks of NCI's baseline and of PoUW's tiles.
4. **#9701's tie compares digests of two different objects** [our worker RELAX's reading of #9701, not mine]: `DeclaredIsParams` sets the circuit file's digest against the program descriptor's digest. They are tied only through the circuit's header. For soundness: the tie needs to say which object each layer names.
5. **Flock's Lean verifier accepts a templated statement with "the Program not held to the circuit"** [its own label, on our runs tonight]. Which layer holds the Program to the circuit is not settled in code.
6. **The interface package** (lean's rows 3 and 4): the profile sizes a stratum with Flock's draw rule (`Profile.lean:70`). My view: the profile states what was established and the rule's arithmetic is SPOT's; I confirm by noon PST after reading its readers. `Record.lean` goes with the channel and `Live.lean` with SPOT.
7. **The specification still reads 21 modules of `Proofs.Flock`** (lean's row 7). Ours with lean; no day yet.
8. **From the others' lists, ours from the other side, agreed as written:** network-accounting's 1, 4 and 8, soundness's 2 and 8, memory-accounting's 3. On the last, our worker GATHER reports by 1:30 AM PST which root the record names for the weights and which the served proofs open.

Not looked for tonight: the recursion's packages, the two-stage form beyond what reads the record back, and `one_stage/` in Python.


---

@top circuits (lead), Fri 13 Nov 12:40 AM PST (08:40Z), to your 12:15 AM PST: the couplings I know in circuits' slice (the Definitions, circuit-check, the adapter's circuit builder), one line each. **The largest: how a circuit is cut into blocks is stated in three places, and each of them knows the Flock block.** [read by me on `main` `b3902993caea` tonight: import lines by `git grep` and the lines named; nothing run; each hand-over is my view]

| # | What knows what | Where | The clean hand-over |
|---|---|---|---|
| 1 | The cut of a circuit into blocks is in the Definitions, in the adapter and in circuit-check, each sized to the Flock block | Nine modules of `verity/definitions/registry/` name their own cut (`CUTS`: `greedy_stages.py:253`, `sparse_mla.py:383`, `sampling.py:574` and six more). The adapter holds a second table by Definition name (`adapters/vllm/verity_vllm/pipeline/serving_templates.py`: `CUT` at 16, `COORDS` 22, `COLS` 35, `CHAINS` 158, `EXPERTS` 182). circuit-check holds each unit to "Flock's proved range" (`circuit_check/checks.py:1445`, `block_held.py:1`: 2^27 bits) | A Definition is a circuit and knows no block. One function takes a circuit and a cap and returns its blocks, and the cap is a parameter from whoever calls SPOT. The adapter names no Definition's axes |
| 2 | circuit-check knows the proof system's lowering, and Flock's prover code knows single Definitions | `tools/verity/circuit_check/src/circuit_check/`: 30 import lines of `verity_flock` in five files (`checks.py` has 18: `ir_lower`, `boolean_export`, `tmpl`, `struct_lower`, `ir_sampling`, `topp_word` among them). 31 files of `verity/protocols/flock/prover/python/verity_flock/` outside tests import `verity.definitions`; `ir_sampling.py` lowers `GumbelTopPTokenSelect_v1`, and `lowerings.py` finds one module per template | The Boolean circuit is the one thing handed over. circuit-check holds a Definition to its Boolean circuit; the proof system takes any Boolean circuit and names no Definition. The second half is proofs' to say |
| 3 | circuit-check's suite reads protocols' verifiers and devices | `tools/verity/circuit_check/pyproject.toml`: 148 lines of its declared inputs name paths under `verity/protocols/`, `adapters/` or `benchmarks/`, among them PoUW's `verifier/salt.py` and `coins.py`, SPOT's `device/challenger.py`, the certifier's `verifier/warden.py` and `release.py`. 54 of the 230 sidecars sit under `verity/protocols/` (certifier 27, PoUW 15, Flock 7, PoUS 4, PoUB 1) | A protocol's circuits are in a module that imports circuits and Definitions only, so loading one loads no verifier and no device [that the inputs are this wide because a sidecar's import loads its package is derived, not traced] |
| 4 | The builder of a served window's circuit reads SPOT's own record, the certifier's insides and PoUW's schemes | `adapters/vllm/verity_vllm/program/predict/`: `window_weights.py:29`, `window_rows.py:63`, `weights_tie.py:59` (`spot.verifier.one_stage`: `registered`, `registration`, `draw`); `hits.py:9`, `steps.py:122` (`certifier.verifier.advice`); `scan_state.py:22-23`, `window_rows.py:62` (`certifier.prover.ingress_scan`, `certifier.verifier.blocks`); `window_class.py:39` (`pouw.prover.schemes`). Also `commit/registered_rows.py:21` and `program/frontend/rules/vllm_bindings/triton_launches.py:365` | The builder hands a circuit, its blocks and named values. Registration numbers and commits them; the certifier's and PoUW's circuits are Calls it is handed. This is vllm's 1 to 4 and network-accounting's 5 and 6, seen from the builder |
| 5 | The model's circuit holds the commitment scheme: the weights' tie and the embedding's table reads open registered roots inside the circuit | `program/predict/weights_tie.py` (SHA-512 chain units over registered reads); `serving_templates.py:464,477,513` (`commitments.gates.embedding.read`) and `:297,478` (`commitments.history.CAPACITY`) | I am not sure this is one to mend: by rule a value is touched only inside a proof. The question for LAYERS is whether a tie to a registered root is a circuit the builder writes, or a rule SPOT is handed with the roots |
| 6 | The Definitions' tests import PoUW's and PoUB's code | `verity/definitions/tests/test_pearl_c_call.py:29-31`, `test_pearl_c_forming.py:23-26`, `test_pearl_c_noise.py:29`, `evaluate/tests/test_difftest.py:57`, `requirement/tests/test_census_and_memory_map.py:47-48` | Each test sits with the protocol whose circuit it tests |
| 7 | One model is named where any circuit should do | `circuit_check/block_held.py:11` (`_GLM53_TREES`); `verity/definitions/ops/prims.py:140-145` (a Definition's text names GLM-5.3's tiles and an adapter test by path); `benchmarks/sampling_plan/glm53_blocks.json`, a pinned card of the model's blocks that #9075 had to change when the select was cut | The model's shapes and blocks come from its compile as one object, which the checker and the planner take as input |

- **None I know:** no file of `verity/definitions/` or `verity/circuits/` outside tests imports a protocol, and the adapter's `query/`, `correspondence/`, `acquire/` and `properties/` and our four `pipeline/` files (`build.py`, `match.py`, `commit.py`, `commit_plan.py`) import none [the same search].
- **Not looked for tonight:** the other direction (what protocols and benchmarks import of the builder's insides), the Lean under `verity/circuits/lean/`, and `benchmarks/served_units/`, `served_commit/` and `compile_cost/`.
- 1 and 3 are ours to fix, 1 with proofs for the cap; 2 with proofs; 4 with vllm, inside its word on the two registrations (VLLM-417, Fri 7 AM PST). Days after LAYERS's document, as you say.

## Answers to the first cut's questions (part 5.4), as they arrive

### compute-accounting, to W1, G6, W3 (TOP-415, about 1:07 AM PST Fri 13 Nov; copied whole)

@top compute-accounting (lead), Fri 13 Nov 1:15 AM PST (09:15Z), the layers document's three questions to us. [read by me on `main` `e4da505fe488`: `PouwCheck/Credit.lean` 100-115 and 1625-1650; the rest from tonight's reports]

1. **W1, the salt's source: yes, and two more things.** PoUW needs a salt that (a) one of the auditor's devices drew after the window's commitments it seeds (root(A), root(B), the round's request table) are recorded, (b) under a key the auditor pinned (PRF-579, #9649), and (c) bound to that window and epoch by the device's signature. Which device draws it, and how phases line up with a certifier's windows, is not PoUW's to know: compose's tie, as network-accounting's 9 says.
2. **G6, `models.json`:** the check reads one thing of it: `PINNED_MODELS`, each role's `checkpoint_sha512` (`Credit.lean:1633-1636`), to require that the period's open names a role and checkpoint pinned there before its weights' root counts (`committedWeights`, refusals `noModel`, `unpinnedModel`, `otherCheckpoint`). **Yes, registration can hand it:** the registered model's role and checkpoint digest as a parameter of the check, and `include_str` goes, which also frees the check's audit key from its directory (lean's 6).
3. **W3, served imports of experimental schemes:** correcting my 12:20 AM PST line: **Dither has none** (no served file or test imports its moved Python; MAPROWS, 12:50 AM PST). **The FP4 variant is the coupling:** served `prover/circuit/cap.py`, `pc8.py`, `rowk.py` and `plan.py` build on `prover/circuit/pc4.py`, and served tests import `pearl_c4`, `pc4` and `pearl_c4_vectors` (`test_pouw_pearl_c.py:624`, `test_pouw_pearl_c_sm120.py:155`, `test_pouw_pearl_c_work.py:348`, `test_pouw_registration.py:14`, `pouw_regions_vectors.py:19,22`, `test_circuit_rowk.py`). Fix: what the served scheme shares moves out of `pc4.py` into PoUW, and the FP4 variant imports PoUW; ours, after the chains.

### compute-accounting, a correction of the page's section 7 (TOP-415, about 1:03 AM PST; copied whole)

@top compute-accounting (lead), Fri 13 Nov 1:08 AM PST (09:08Z). **Our chain-2 rows are ready as MAPROWS posted at 1:03 AM PST: maps `cursor/pouw-move-rows-chain1-1de3` at `5c0f590bd8f3`, staged `cursor/pouw-chain2-staged-1de3` at `17c48f082aa9`, six separable commits, suites green on the pushed staged head.** One correction to the page's section 7 row for PoUW's loose files: `pouw/clear/` goes to `pouw/prover/clear/`, not `pouw/verifier/` (it is a reference tool that runs the prover's schemes, which a verifier may not import), in its own PR after the chain; `receipts/` goes to `verifier/` tonight as written. Please carry that into `ARCHITECTURE.md` (#9702) and `tests/tree.toml`.

### soundness, its four questions (TOP-415, about 1:07 AM PST Fri 13 Nov; copied whole)

@top soundness (lead), Fri 13 Nov 1:25 AM PST (09:25Z), the layers document's four questions to me. [read: parts 1, 3 and 5.1 and the rows naming soundness at `0f875a6e82bb`]

1. **The form of an accounting answer: agreed, with two fields more.** The answer carries:
   - the registration's head;
   - the part (device, memory leaf or link);
   - the amount;
   - ε at δ;
   - the roots read;
   - **added:** the period's span, so compose can sum answers over one period;
   - **added:** the key of the auditor's device whose challenges or signatures it rests on, so compose can hold it to the registration's pinned key. Tonight's finding is that no check does this today (PRF-579).
2. **One shape, one lemma, but over a public map, not equality.**
   - **The lemma:** "the answer names `f(x)` for some `x` the registration's view holds", one statement, `NamesRegistered f`, with compose's comparison proved once.
   - **Where `f` is the identity:** keys, links, memory leaves and the weights' root. The answer names the registered thing itself.
   - **Where it isn't, the circuit:** proofs is right that the two digests are of different objects. Registration names the Program by its descriptor's digest, and the compiler by its own. SPOT is handed a circuit file. `DeclaredIsParams` is then "SPOT's circuit file is the layout of the registered Program", which a caller checks once, deterministically, with `f` the layout. So one lemma, and each use names its `f`.
3. **What the composition area may import of a protocol:**
   - **In Definitions and Properties:** the protocol's `Guarantees` and the vocabulary its statements read.
   - **In Proofs:** only its `<Area>.SecurityProofs.G` theorems, never an inner lemma.
   - **Code facts:** a check's agreement lemma, such as `accepts_iff`, is a protocol's exported property and may be read.

   So T8, with the Proofs tree spelled out. Today's `Sampled/` tree breaks it (X2), and that is mine to fix.
4. **"To Daniel" rows that only move a statement, struck:**
   - **U5:** the wipe's joined statement moves to composition, same statement.
   - **Y11:** the joined guarantee changes area and owner, same statement.
   - **X2:** struck if SPOT's exported theorem states what the leg reads today. I keep it if it differs.
   - **G3:** restated over the list, it is the same for the two lists that exist, and changes only future lists.

   **The rest stay to Daniel:** P1 where a guarantee cites R2, R6 or R7; P2 (which device signs a declaration is a premise); W1 (the device the noise seeds rest on); X1 and X3 (compose's verdict and a new tie); and N7 with G2.

### training, on I2 (TOP-415, 1:11 AM PST; copied whole)

@top training, Fri 13 Nov 1:11 AM PST, your question on I2. **Five figures, and one test that reads code.** [read by me on `main` `e4da505fe488`: every `verity` import of the calculator]

1. The monitoring periods' lengths and the reporting δs (`reporting`, `params.py:39`).
2. The per-window δ (SPOT's profile, line 38).
3. A proof block's smallest and widest output class (`partition_v2`, line 40).
4. The policy's `K` and `Γ`: already a data file.
5. A request's declared bits, a bucket's, and requests a link-window.

`test_held_to_core.py` imports SPOT's `Bernoulli`: a root test, above both.

### lean and ci (TOP-398, 1:09 to 1:10 AM PST; not answers yet)

- lean answers its questions on TOP-415 by 6:30 AM PST, after chain 2's asks.
- ci (TOP-398, 1:09 AM PST), a coupling of the queue's own, for the record: chain 1 moved the tests of
  `verity/channel.py` and `verity/device.py` to `verity/tests/tests/`, which no low-risk exclude in `drive.toml` names,
  so a change to those tests alone would land on its ready mark without a reader. ci mends it in one `drive.toml` PR
  after chain 2.

