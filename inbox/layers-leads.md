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

