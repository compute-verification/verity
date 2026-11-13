# What each layer of Verity may know, and every coupling on `main` that breaks it

**The layers are held apart by import rules only, and three things leak through them: SPOT's record of a window, which
every protocol and the adapter read and which carries its callers' facts; the registration, which no accounting
protocol reads, so each restates what it needs; and the adapter, which holds protocol code and reads verifiers and
devices. Part 2 lists 73 couplings. Part 3 names the five that matter most.**

First whole cut, Fri 13 Nov 2026, for 5:00 AM PST (13:00 UTC). `main` was read at `9182900fb94e`. Written by LAYERS,
a worker of top. Nothing was built or run: this is `rg`, `git` and the files.

How I know each claim:

- **[read by me]**: I opened the file at that line on that commit.
- **[counted by me]**: a search I ran. Unless it says otherwise it counts files that are not tests, by whole word,
  ignoring case.
- **[relayed]**: a lead's word on TOP-415, copied in `inbox/layers-leads.md` on `cursor/verity-top-4db4`. Not re-read by
  me. "(soundness 2)" is that lead's own row number.
- **[page]**: the architecture page's words (sections 1 to 7 and 10). **[SPOT page]**: Daniel's page "SPOT security
  definitions". **proposed**: mine, where the pages do not say.
- **[not checked]**: nobody read it for this document.

Words. I use the architecture page's. A **piece** is one part of a circuit's cut (the SPOT page calls it a block). The
**sampling rule** is the SPOT page's sampling design D. A **challenge** is a random value the auditor's side draws
after a commitment. **SPOT's window record** is the record SPOT's code calls a "registration record": the page says it
takes another name, and this one is proposed. **The registration** is only what the developer registers once, before
any period. The **head** of a registration is the 64-byte digest that names an accepted registration (the word is
`CoreProtocol/Account.lean`'s).

## Part 1. Each layer in four lines

Each layer has four lines: what it is **handed** (as data), what it **hands back**, what it **must not know**, and
who **calls** it. Bottom up.

### 1.1 The code

**Circuits** (`verity/circuits/`)

- Handed: nothing [page]. Its functions take only a circuit's bytes, and for the evaluator a circuit and input bits.
- Hands back: a circuit as canonical bytes, named by the SHA-512 of those bytes (64 bytes); the layout that gives every
  gate its index; a cut of the circuit into pieces, with the check that the pieces cover it; for each piece, the gate
  indexes it reads and writes; the evaluator's output bits [page, section 10]. And the circuits a protocol itself names:
  each GPU operation bit for bit, SHA-512, the universal circuit [page].
- Must not know: models, hardware schedules, proof systems [page]. Proposed: nor a commitment scheme, a protocol, a
  registration, a period or a request.
- Called by: commitments and everything above it. Its Lean (`core_base`, `core_circuits`) requires nothing above it.

**Commitments** (`verity/commitments/`)

- Handed: values, as bit strings [page]. Proposed: with a key and a salt for a hiding leaf, a position to open, and a
  tag the caller chooses, as bytes.
- Hands back: roots (64 bytes each), and openings on request [page]; the check of an opening against a root; a
  signature and its check; the one derivation of challenges from a seed.
- Must not know: what a value means [page]. Proposed: so it names no protocol, window, tile, request or adapter, and it
  defines no caller's format string.
- Called by: every protocol, and the PyTorch adapter, which commits values where they sit on the GPU [page, section 5].

**The shared files** (`party.py`, `channel.py`, `device.py`, `protocol.py`, `config.py`)

- Handed: nothing [page]. `config.py` is handed the one public file a run starts from.
- Hands back: types every protocol uses [page]: a party; a message and its types; the registered hardware (machines,
  devices, memory, links); what a protocol declares (its parties and messages, what it needs of the workload, which part
  of the hardware it accounts for); and a parsed configuration (the run's list of protocols, which party runs on which
  machine).
- Must not know: any one protocol [page]. Proposed: nor compose's lists by name, nor the runtime.
- Called by: every protocol (`protocol.py`), registration (`device.py`), the runtime (`config.py`). `party.py`,
  `channel.py` and `device.py` use nothing; `protocol.py` uses those three; `config.py` uses `protocol.py` [page, rule 4].

**Flock** (the proof system)

- Handed: a circuit, commitments, challenges [page]. Its prover also holds the values and their openings.
- Hands back: a proof. Its verifier says accept or reject [page].
- Must not know: where the circuit came from [page]. Proposed: nor that a sample, a window, a registration, an outer
  proof or a ZK server exists above it.
- Called by: SPOT and recursion [page, section 5].

**SPOT** (sampling)

- Handed: "The public input is C and a sampling design D. The prover holds a trace." [SPOT page, Protocol 1]. As data:
  the circuit with its pieces, their order and their reads; the sampling rule; then the commitments; then the
  challenges; then the proofs of the picked pieces [page]. Proposed: where a caller has already fixed part of the
  trace (a registered value, a certified input), it hands SPOT that part's root, and SPOT holds the commitment to it.
- Hands back: "At most this share is wrong, except with this probability." [page]. Proposed: with the roots it held the
  proofs to, so a caller can compare them with its own.
- Must not know: what the pieces compute [page]; the registration (Daniel, Thu 12 Nov, 11:55 PM PST) [page, section 10].
  The limits "hold for one fixed C, so any influence the prover has over C must be accounted for separately", and
  tying the trace's input and output nodes to a real system "takes a mechanism outside SPOT" [SPOT page, section 4].
  Proposed: so SPOT knows no period, day, kind of window, stream, pool, week or served run, and not who calls it.
- Called by: a protocol that proves a statement about committed values: PoUW and the certifier today [read by me]. The
  caller passes in the circuit, its pieces and the sampling rule [page, rule 7].

**Recursion** (one outer proof, and the ZK server)

- Handed: Flock's proofs [page]. Proposed: with the statement they are proofs of.
- Hands back: one outer proof per statement [page]. Through the ZK server, the only bytes that leave the datacenter for
  the auditor [page].
- Must not know: which protocol asked [page]. Proposed: nor SPOT's window record or a registration.
- Called by: a protocol whose proofs leave through it. A demo's configuration starts the ZK server, through the runtime.

**Registration** (an accounting protocol; runs once, before any challenge)

- Handed: the registration (the hardware, the circuit compiler, commitments to other information such as the model's
  weights), the auditor's data sheets, and the run's list of protocols [page].
- Hands back: accepted or refused [page]. Proposed: an accepted registration is named by its head, and it gives, as
  data, each registered part (a device, a memory leaf, a link) with its capacity, each auditor's device's key, the
  compiler's digest and the weights' roots.
- Must not know: how any other protocol works inside [page]. Proposed: nor compose's lists by name. It asks only "does
  the run's list hold a protocol that accounts for this part".
- Called by: compose, once. It uses the shared files and commitments and no other protocol. The other accounting
  protocols, the policy and compose read what it accepted [page, rule 7].

**The other accounting protocols.** Proposed, for all four, from soundness's and memory-accounting's words tonight
[relayed]: each is handed the registration's head and the part it accounts for, and **its answer names both**. That
is what compose needs to join answers, and no answer does it today (rows G2, U6, N7).

**PoUW** (compute)

- Handed: the registered devices and their data sheets, the prover's commitments to each multiply, the challenger's
  challenges, proofs of the sampled tiles [page]. Proposed: the execution phase's live salt is one of those challenges,
  from whichever of the auditor's devices compose names.
- Hands back: credited work for the period, with its error, to compose's account [page].
- Must not know: memory, network; the served values in the clear; the GPU's own word on what it is [page]. Proposed: so
  not the network certifier or its windows, no list of models, and no card's figure as a literal.
- Called by: compose. A workload reaches its prover by one call, "run this matrix multiply under PoUW", over GPU
  buffers [page].

**PoUS and PoUB** (memory held, memory read)

- Handed: the registered memory and its data sheets, the prover's commitments, the memory challenger's signed records,
  the challenger's challenges; through compose, the network certifier's counts of what entered and left the node
  [page].
- Hands back: bits held and bits read for the period, each with its error, to compose's account [page].
- Must not know: compute; the weights; another protocol's code [page] (PoUB uses PoUS [page, section 5]). Proposed: so
  not a model, a checkpoint or a serving system's layer order. The store is a registered root.
- Called by: compose. A workload reaches each prover by one call (store these weights; say where the store sits; give
  up a server's memory for a wipe) [page, section 10].

**The certifier** (network)

- Handed: the registered links, the network certifier's signed records, the prover's commitments to requests and
  responses, the challenger's challenges [page]. Proposed: with the registered compiler (a circuit with a public length
  cap) and the period's advice as bits it never interprets (`AGENTS.md`).
- Hands back: to compose's account, the period's exfiltration capacity with its error; through compose, to memory, what
  entered and left each node [page].
- Must not know: compute, memory, the weights [page]. Proposed: so not PoUW's salt or weights, not the policy's check,
  not SPOT's window record, and not a request's fields, a scheduler or a serving system.
- Called by: compose.

**The policy** (inference-only)

- Handed: the circuit's description. From training, as data: how much such a circuit slows training [page].
- Hands back: accept or reject, to compose [page].
- Must not know: values, traffic [page]. Proposed: so no record of a served run, and no accounting protocol's code.
- Called by: compose.

**Compose** (remote and onsite)

- Handed: registration's outcome and each protocol's [page]. Proposed, as data: for each protocol on the run's list,
  accept or reject, the head and the part its answer names, the amount with its error, and the roots it committed to.
- Hands back: one verdict, and the period's account [page].
- Must not know: how any one protocol works inside [page]. Proposed: so no protocol's record format, file name or
  proof internals. It compares heads and roots, and adds errors.
- Called by: the auditor. A demo's configuration names the list.

**The runtime** (`verity/runtime/`)

- Handed: a configuration [page].
- Hands back: running parties and their transcripts [page].
- Must not know: any protocol's rules [page]. It never says accept or reject, and no verifier loads it [page, section
  10].
- Called by: only what starts a run [page, rule 3]: a demo, the `verity` command. No protocol and no adapter uses it.

**An adapter** (`adapters/vllm/` on `adapters/torch/`)

- Handed: the serving system and its settings. From the prover: what a protocol asks of the workload [page].
- Hands back: to the prover only. Up front: the compiler and the weights. Each period: the values and the advice
  [page].
- Must not know: the verifier, the auditor's devices, the runtime; how a protocol samples or decides; any protocol's
  code beyond the one call it makes [page].
- Called by: a demo. A benchmark, through its public interface only [page, rule 11].

**A demo and a benchmark**

- A demo is handed all of the above and hands back a run anyone can repeat with one command. Nothing uses a demo
  [page].
- A benchmark is handed the system, as a user, and hands back numbers in the evidence store. It never decides a
  verdict, and nothing uses a benchmark [page].
- Must not know (proposed): a benchmark writes no file of the system and none under `security/`. A measurement reaches
  a guarantee one way: a person writes it into a data sheet, and the guarantee takes the sheet as a parameter [page,
  rule 9].
- Called by: the `verity` command.

### 1.2 The Lean

**Each protocol's own Lean package** (its check: `flock_verifier`; `core_spot`; recursion's `core_vstar_recursive` and
`core_firewall`; PoUW's `pouw_commit`, `pouw_window`, `pouw_check`; PoUS's `pous_check` and `core_memory_challenger`;
PoUB's `remote_audit`; the certifier's `network_check`; the policy's `nci_check`; compose's `core_composition`,
`core_remote`, `core_onsite`) [read by me: each Lake file]

- Handed: the protocol's public input and the transcript, as bytes in the interface's one canonical format.
- Hands back: accept or reject and the published answer, as bytes. To `security/`: its Lean functions, which the
  properties are stated about.
- Must not know: `security/` (code never requires it; `AGENTS.md`). Proposed: another protocol's package, except the
  proofs it calls and, for compose, the protocols it lists; and of a package it calls, only the modules that package
  names as its entry. A check that verifies no Flock proof does not require Flock's verifier.
- Called by: compose's package, the protocol's own executable, and `security/Properties`.

**The interface package** (`verity/protocols/interface/`, `core_protocol`)

- Handed: nothing.
- Hands back: what each protocol's check implements [`AGENTS.md`]: the type of a protocol over bytes (commitment,
  challenge, response, answer), its canonical encoding, the command line, exact ratios.
- Must not know (proposed): any one protocol; the registered hardware, the registration, compose's account, the
  period; the proof system's draw rule.
- Called by: every protocol's Lean package. It requires only the three base packages.

**The areas of `security/`** (three trees: `Definitions`, `Properties`, `Proofs`)

- Handed: the verifiers' Lean [page]. `Definitions` is handed nothing from code (Mathlib only; `AGENTS.md`).
- Hands back: guarantees, each with its named assumptions and its open premises; one composition theorem for any list
  of protocols; and the error budget [page].
- Must not know: a statement never reads a proof [page]. Nothing in the system uses `security/` [page]. Proposed, by
  area: `Core` and `Hash` read no other area; `Spot` reads `Core` only; `Flock` reads `Core` only and holds the proof
  system alone; each accounting protocol's area (`Pouw`, `Pous`, `NetworkCertifier`) and the policy's (`Nci`) reads
  `Core`, SPOT's guarantee and the registration's view, and no other protocol's area; the composition area
  (`Properties/Verity`, soundness's) reads each protocol's guarantee and the registration's view, and nothing inside a
  protocol.
- Called by: the Lean check, and a reader. One definition of each shared concept, in `Core` or `Hash` [`AGENTS.md`].

## Part 2. Every coupling on `main` that breaks part 1

One row each. The columns: **id**; **who** knows too much; **what** it knows, with the **evidence** (file and line,
up to three places, and a count); its **kind**; the **clean** interface in its place (who hands what to whom); the
**fix**; the **owner**.

Kinds: **import**; **name** (a name or a format string); **shape** (a parameter or a record field shaped for one
caller); **job** (a part or a device doing another layer's job); **reads** (a statement or theorem that reads another
layer's definition); **dup** (a second definition of one shared concept); **test** (a test only).

Fixes: **move** (a path move); **rename**; **code**; **Lean** (a change to a Lean statement). **To Daniel** marks a fix
that changes what a guarantee says or rests on. Paths are under `verity/protocols/` unless they start with another
directory. Rows from a lead that I did not re-read say [relayed].

### 2.1 The proofs: SPOT

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | SPOT's window record | Its callers' facts. Of its 24 fields, 10 are SPOT's own (the circuit, the sampling rule, the commitment's form) and 14 were added for callers, each optional: the declared day, the stream and the prover, the seed salt, the pool with its week, lag and reset, and pinned, held, supplied and output values. `spot/verifier/lean/Spot/Registration.lean:163-213` (the record), `:545-614` (its check: `R2-declared` 573-574, `R6-stream` and `R6-prover` 599-601, `R6-seed-salt` 603, `R7-pool-chain` 611), `:131-139` (`Pool`). Python twin: `spot/verifier/one_stage/registration.py`, 874 lines. [read by me; the 10 and 14 split is my reading of the field comments] | shape | SPOT is handed the circuit with its pieces, the sampling rule and the commitment: a list of roots, and for a part of the trace a caller already fixed, that part's root. The caller checks its own facts (the day, the kind of window, the pool) and hands SPOT roots only. | code and Lean: split the record in two, SPOT's public input and the caller's record over it. R2, R6's stream, prover and salt, and R7 go to the protocol that calls SPOT (soundness says the same of R2 [relayed]). **To Daniel** where a guarantee cites one of those checks. | proofs, soundness |
| P2 | SPOT's challenger, `spot/device/challenger.py` (2,171 lines) | The auditor's ledger and two other parts' records: a period's open and close, a day's declarations, which window counts toward a period, the pins of `main`'s first-parent line, "a lane's day", PoUW's live window as "the first client", and recursion's run record. Lines 95-112, 137-166, 199-201, 304 (`PERIOD_KINDS`), 1296 (`verify_periods`). It imports the claims' ids (219) and the hardware's record entry (220). [read by me] | job | The challenger is handed a commitment's digest, draws challenges after it and signs both. The period's ledger is compose's. A day's declarations are registration's, signed by registration's device. A run record is recursion's. | code: three files from one. **To Daniel** if which device signs the declarations is part of a premise (soundness 8 [relayed]). | proofs, soundness |
| P3 | SPOT's Python | Three things below or beside it that it should not load: the claims' ids, the deployment layer, the hardware's record entry. `tests/import_order.toml`, SPOT's `loads`; `spot/verifier/parties.py:123,126`; `spot/device/challenger.py:219-220`. [read by me] | import | SPOT loads circuits, commitments and the shared files. | code: the record entry goes into `channel.py` (proofs [relayed]); the party types come from `protocol.py`; `claims` goes (S5). | proofs |
| P4 | SPOT's code and docs | Names from above it. Lines in SPOT's files: pool 316 (7 files), lane 144 (9), PoUW 142 (23), week 96 (8), certifier 39 (12), vLLM 25 (8) [counted by me]. `spot/verifier/second_stage.py:838` (`INTERIOR_ROOT = "verity-vllm/interior-run/v1"`); `spot/verifier/one_stage/proofs.py:66` (imports the served circuits' tables); the verdict's field `rests_on_period_check`, the certifier's check by name (`Spot/Pieces.lean:247`). [read by me] | name | SPOT's words are circuit, piece, sampling rule, root, challenge, proof and bound. A caller's tag reaches it as bytes. Its verdict lists the roots it did not check itself, and the caller says what they rest on. | rename and code, mostly carried out by P1 and P2. | proofs |
| P5 | Every caller of SPOT | SPOT's insides, because SPOT states no entry. Lean: 9 inner modules are imported from outside (`pouw/verifier/lean/check/PouwCheck/Credit.lean:17-20`, `PouwWindow/Rounds.lean:4`, `inference_only/verifier/nci/lean/NciCheck/Record.lean:6`). Python: `spot.verifier.one_stage` is imported in 153 lines outside SPOT; the certifier imports SPOT in 29 lines of 13 files, PoUW in 23 of 12, PoUB in 10 of 8. [counted by me] | import | One entry in Lean and one in Python: circuit, pieces, sampling rule, commitment, challenges, proofs in; accept or reject, the bound and the roots out. Callers import that and nothing else. | code: name the entry; a test holds the rest private (T5). | proofs |
| P6 | Recursion's package | It defines SPOT's protocol: the protocol compose lists as `verity/sampled-proofs/v1` is `VStar.Recursive.Sampled.protocol`, so recursion requires SPOT. `recursion/verifier/recursive/VStarRecursive/Sampled.lean:33,81`; its Lake file; `remote/verifier/CoreRemote.lean:37,116`. [read by me] | import | SPOT's protocol is defined in SPOT's package. Recursion is handed Flock's proofs and one statement, and does not know which protocol asked. | move `Sampled.lean`. Whether recursion may still require SPOT is a question for proofs (part 5). | proofs |
| P7 | SPOT's Lean package | It holds the registered hardware's difftest. `spot/verifier/lean/Spot/SiteDifftest.lean:3`. [read by me] | test | The test goes with registration. | move (on the page, with registration). | proofs |

### 2.2 The proofs: Flock and recursion

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | Flock's verifier | The window record's format, defined there "so that every reader reaches it without a new requirement", and the registered reads ("a program's weights"). `flock/verifier/Flock/WindowRecord.lean:7-16,26-28`; `Flock/Registered.lean`. [read by me] | name, job | Flock is handed a circuit, commitments and challenges. A record of what is fixed before the challenges is SPOT's input. | move to SPOT's package, after P1. | proofs |
| F2 | Flock's Lean package, `flock_verifier` | It is three parts in one: of 57,859 lines in 269 files, recursion is 10,800 (`Rec*` 24 files, `VStar` 21) and the ZK server 8,409 (21 files). 14 packages require it (`tests/test_boundaries.py:115`), several for helpers only: the certifier's check imports `Flock.Hm96`, `FV3`, `Canon`, `Sizes`, `Bytes`, `WindowRecord`, `Public`, `Program`. [counted by me] | import | Hashes, canonical bytes and commitments come from the three base packages. Flock's package is the proof system. Recursion and the ZK server are recursion's. A check that verifies no Flock proof requires none of it. | move (Lean modules between packages; the lock's records move, no statement changes). | proofs, lean |
| F3 | Flock's verifier | A table of served models that a benchmark writes. `flock/verifier/Flock/VStar/Menu/Pinned.lean:5-7` ("Generated by `benchmarks/numerical/...`"), `:15-103` (`"vllm/gemma2-2b__bf16__l40s__..."` and five more model families); imported at `VStar/Cli.lean:5`. [read by me] | name | The cost table is a file the caller hands in: measured proving costs "are handed to it as data" [page, section 10]. | code: read it as a parameter; delete the generated module. | proofs, benchmarks |
| F4 | Flock's prover (`flock/prover/`, the distribution `verity_flock`) | Recursion's prover and part of SPOT's: 17 `rec_*.py` modules, `pod/rec_inner.py`, `inner_registered.py`. It imports SPOT's verifier in 42 lines of 14 files and the served circuits in 103 lines of 36 files. The import listing therefore puts Flock's prover above SPOT. `tests/import_order.toml` (the order; SPOT's `loaded_by`). [read and counted by me] | import | Flock's prover: a circuit, commitments, challenges in; a proof out. Recursion's prover sits in `recursion/prover/`. Then the listing's order is flock, spot, recursion, as the page draws it. | move | proofs |
| F5 | Flock's verifier and its command line | Names of PoUW and the adapter. `flock/verifier/FlockVerify.lean:344` (`--work-table`, "the PoUW work"); `Flock/Firewall/Salts.lean:51` (`PouwCheck.Credit.registeredFile`); `Flock/Sigma.lean:4` (`verity_vllm`). Files under `flock/` naming: registration 59, window 38, vLLM 29, PoUW 19. [read and counted by me] | name | As F1. | rename, after F1 to F4. | proofs |
| F6 | Flock | Tools and benchmarks: the listing's ties flock to tools (3 files) and flock to benchmarks (7 files). `tests/import_order.toml`, `[[against]]`. [read by me] | import | Nothing uses a benchmark (rule 9). | code | proofs |
| F7 | The base Lean packages | They carry one proof system's name: `namespace Flock` or `open Flock` in 7 of 8 files of `core_base`, 9 of 10 of `core_circuits`, 2 of 3 of `core_commitments` and 14 of 19 of the interface. [counted by me] | name | The circuit language and the hash name no proof system. | rename. Many records would change for no change in meaning, so I list it and do not propose it now. | lean |

### 2.3 Circuits and commitments

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | The circuits written in the language, `verity/definitions/` | PyTorch, vLLM and models. Files naming vLLM 195, PyTorch 89, a model 111 [counted by me, by substring]. | name | Rule 1. The served kernels' circuits go with the adapter that serves them. | move (on the page: Sun 15 Nov). | top, vllm, circuits |
| M1 | `verity/commitments/domains.py` (409 lines, 341 constants) | Every protocol's and the adapter's format strings, in one table: PoUW's (lines 19-126), SPOT's, Flock's, the ZK server's, PoUS's, PoUB's, the certifier's, compose's, the wire's, and vLLM's (`VLLM_*` at 182-196 and 397-400, such as `"verity-vllm/pouw-pearl-c-run/v1"`). Lean twin: `verity/commitments/identity/Identity/Domains/`. [read by me] | name | Commitments is handed a tag as bytes. Each protocol defines its own tags, and a test of the whole tree holds them distinct. | code: split the table by owner (T3). | proofs (proposed), each lead for its lines |
| M2 | Commitments | Modules shaped for one caller: `window.py` (`WINDOW_DOMAIN = "verity/one-stage/served-domain/v1"`, which binds program, partition, port, schema, leaves and run), `advice.py`, `history.py`, `pool_row.py`, and `windows/` (7 modules: "what an adapter hands the prover for each execution phase"), which loads the served circuits' catalog (`tests/import_order.toml`). [read by me] | shape | Values in, roots and openings out. What an adapter hands a prover is that prover's interface. | move: `windows/` to SPOT's `prover/`; `advice.py` and `pool_row.py` to their callers. | proofs, vllm |
| M3 | Commitments' tests | The adapter's files, listed as test inputs. `verity/commitments/pyproject.toml:10-11`. [read by me] | test | A package's tests read its own package. | code | proofs |

### 2.4 The shared files, the runtime and registration

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Nine protocol files | The runtime's package, for a type. `ProtocolDef`: `pous/device/challenger_party.py:66`, `poub/verifier/session_parties.py:47`, `spot/verifier/parties.py:126`, and one more. `stream.Side`: 5 files, such as `poub/verifier/verifier_party.py:34`. [read by me] | import | The types sit in `protocol.py` and `channel.py`. The runtime calls a protocol, and no protocol imports the runtime (rule 3). | move the types (memory's part is MEM-291 [relayed]). | proofs, memory-accounting |
| S2 | The configuration's reader | Compose's two lists, with every protocol's name, and it refuses any other list in Python. `verity/configuration/deploy/config.py:59-61` (`COMPOSITES`: onsite's 7 names, remote's 3). [read by me] | name, job | `config.py` reads a list of protocol ids. Whether a list is acceptable is compose's to say, in Lean [page]. | code | proofs |
| S3 | The hardware package | The transcript's signed entry, `verity/hardware/record.py`, which SPOT's challenger reads. [read by me] | job | A signed entry of a transcript is the channel's. | move into `channel.py` (proofs [relayed]). | proofs |
| S4 | Two modules | The monitoring periods, twice: `verity/periods/__init__.py:29` (`MONITORING_PERIODS`) and `reporting.py` (`PERIODS`), the same three lengths. The period's type three times: `security/Definitions/Core/Period.lean`, its stated copy `interface/CoreProtocol/Period.lean`, and the Python mirror. [read by me] | dup | One period type, beside the account it is summed in. One list. | code | proofs (proposed) |
| S5 | `verity/claims` (623 lines) | A list of claim ids kept by hand beside the locks. 10 Python files read it [counted by me]; SPOT's challenger uses it for one membership test (`spot/device/challenger.py:378`). About 80 Lean docstrings and three locks name its ids [relayed, top]. | dup | The locks are the one registry of guarantees and assumptions. A generated file maps an id to its Lean name (soundness [relayed]). | code: a generator in the Lean check; the Glossary's line on a guarantee changes. | lean, soundness |
| S6 | The import listing, on the first chain of moves | One row holds `channel.py`, `config.py`, `party.py`, `protocol.py` and `runtime`, so a load of a shared type reads as a load of the runtime. [relayed, top; not on `main` yet] | test | The shared files are a row below every protocol. The runtime is a row above them all. | code (TOP-398 [relayed]). | top |
| G1 | The interface package | Registration, the registered hardware, compose's account and the period: `Site.lean` 702 lines, `Registration.lean` 535, `Account.lean` 861, `Period.lean` 147, of the package's 3,717. Outside the package `Site` is read by 3 files, `Period` by 1, `Registration` and `Account` by none. `interface/CoreProtocol/`. [counted by me; lean's table says the same, relayed] | job | The interface hands back only what a check implements. Registration is a package of its own on it. The account is compose's. | move: cut the Lean out (about 45 files by proofs' sizing [relayed]). | proofs, soundness |
| G2 | PoUW, PoUS, PoUB, the certifier | None reads the accepted registration. No file of a protocol imports the registered hardware or the registration log in Python. In Lean one module does (`certifier/verifier/lean/NetworkCheck/Registered.lean:3`) and only a test's vector file calls it. [counted by me] An answer that names no head counts for nothing in the account (`interface/CoreProtocol/Account.lean:280-283`) [read by me]. The page says "None does". | shape | Each accounting protocol is handed the registration's head and its part, and its answer names both. Compose joins answers by comparing them. | code and Lean: each published answer gains two fields. **To Daniel**: it changes what each answer says. | compute-accounting, memory-accounting, network-accounting; proofs for the type |
| G3 | The registered hardware | Compose's two lists, by name. `interface/CoreProtocol/Site.lean:285-286` (`Composite`: onsite, remote), `:372,376` (`Covered` and `Reached` hold "if the composite is onsite"); `verity/hardware/site/__init__.py:61`. [read by me] | name | Registration is handed the run's list and asks whether it holds a protocol that accounts for this link or this memory. | Lean: restate both over the list (soundness 4 [relayed]). **To Daniel**. | proofs, soundness |
| G4 | The adapter, PoUS, PoUB | The weights are committed three times with nothing tying the roots: the served run's root for each weight, PoUS's one hash, PoUB's tree [relayed, memory-accounting 3, vllm 4]. Compose's own text says of two of its forms: "nothing here holds PoUW's committed weights to the weights the record registers" (`remote/verifier/CoreRemote.lean`, header) [read by me]. | dup | One commitment to the weights, in the registration. Each protocol is handed that root and compares. | code and Lean. **To Daniel**: it adds a tie a guarantee lacks. | proofs, memory-accounting, compute-accounting, vllm |
| G5 | The data sheet | Four Lean definitions, one set of Python literals, and the TOML: `interface/CoreProtocol/Site.lean:150`, `security/Definitions/Core/Site.lean:137`, `security/Definitions/Accounting/TS/Sheet.lean:68`, `security/Definitions/Core/Accel/Card.lean:31`; `pouw/prover/circuit/h1device.py:19-23`; `verity/hardware/cards/`. The hardware package also writes Lean under `security/`, and a card names its Lean module (`verity/hardware/lean.py:21-22`, `cards/h200-sxm-141gb.toml:14`). [read by me] | dup | One type. A statement takes a sheet as a parameter. The registration check holds a device's figures to the auditor's sheet [page, section 10]. | Lean and code (on the page: Tue 24 Nov). | compute-research, proofs |
| G6 | PoUW's check and PoUB's device | A list of served models, by file path. `pouw/verifier/lean/check/PouwCheck/Credit.lean:1634` (`include_str "../models.json"`); `poub/device/live_verifier/host.py:165` (`MODEL_PINS`, PoUW's file). [read by me] | name, import | The registration hands the registered circuit's shapes and the weights' roots, as data. | code | compute-accounting, memory-accounting |

### 2.5 The accounting protocols and the policy

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| W1 | The network certifier's device, and PoUW | The device draws PoUW's challenge and keeps PoUW's weights' root: `certifier/verifier/active.py:72-95` (`Proxy(salt=...)`, one sealed salt for each PoUW window, and "the 64-byte root of the weights PoUW's noise seed reads"). PoUW's execution phase is counted in the certifier's windows: `pouw/verifier/salt.py:1-13`, `pouw/verifier/core_check.py:1129-1166`. 16 of PoUW's files name the certifier and 12 of the certifier's name PoUW. [read and counted by me] | job | PoUW is handed a live salt as a challenge from one of the auditor's devices. Which device, and how phases line up with the certifier's windows, is compose's tie. The certifier's device stamps and counts traffic and does not know whose salt it carries. | code and Lean. **To Daniel**: PoUW's noise seeds would rest on a named device's challenge, not on the certifier's. | compute-accounting, network-accounting, soundness |
| W2 | PoUW's driver | A serving system's run records and paths (`served_passes.py`, `verify_run.py`). [relayed, compute-accounting 4] | name | The adapter hands a record in PoUW's own format. | code | compute-accounting |
| W3 | PoUW's verifier | Two experimental schemes' modules, in served code [relayed, compute-accounting 5]. My search of the import lines under `pouw/verifier/` found none [counted by me], so this needs its file and line. | import | The variants import PoUW, never the reverse. | code | compute-accounting |
| U1 | PoUB's verifier and device | The model: a pinned checkpoint, and a serving system's parameter order and layer names. `poub/verifier/store.py:1-30` ("every parameter of vLLM's model with two or more dimensions, in its module order"); `poub/device/live_verifier/host.py:41-59` (`--model`, `--checkpoint`). 23 files under `poub/` name a model or a checkpoint. [read and counted by me] | name, job | PoUB audits reads of a store named by a root. What the root stands for is fixed once, at registration. | code and Lean (`RemoteAudit/Store.lean`). Daniel refused the memory challenger reading the store at setup [relayed]. | memory-accounting |
| U2 | The memory challenger's device | Another protocol's store, in the clear: 24,576 blocks a session (6 windows, 1,024 sampled units each, 4 blocks a unit), kept in `private/cases.jsonl`. `poub/verifier/p2_party.py:87`, `p2.py:65`, `p2_party.py:774`. [read by me; the product is mine] | job | A device signs what it sees and hands it to the prover. The auditor sees only proofs about it [page, section 1]. | Lean and code: the session proved in zero knowledge (on the page, not dated). **To Daniel** (the page's section 9). | memory-accounting |
| U3 | PoUB's session over PoUS's store | PoUS's device's record: PoUB builds its own registration of the store from the memory challenger's record of PoUS's session. `benchmarks/pous/served_piece.py:53-56` [read by me]; `p2_party.py --pous-record` [relayed]. | shape | Both are handed the store's root from the registration. | code | memory-accounting |
| U4 | PoUS's grader | The layout of `security/`: `pous/lean/TRUSTED.sha256` pins 58 files under `security/` by path [counted by me]; the listing ties PoUS to `security/` (`pous/lean/submissions/reference.lean`) [read by me]. | import | No code uses `security/` (rule 2). The trusted text reaches the grader as a declared input, by digest (lean 5 [relayed]). | code | memory-accounting, lean |
| U5 | PoUS's area of `security/` | The network certifiers' count: the wipe's bound is stated over "every count of the egress by the network certifiers". `security/Properties/Pous/Wipe.lean:14-26,49-52`. [read by me] | reads | Each protocol states its own count. The two meet in compose [page, section 10]. | Lean: the joined statement moves to the composition area (soundness 5 [relayed]). **To Daniel**. | soundness, memory-accounting, network-accounting |
| U6 | Memory's answers | Five ties are missing: no memory answer publishes the head or a memory leaf (`pous/verifier/check/PousCheck/RunProtocol.lean:203-209` publishes `certified`, `eps`, `s_bits` only [read by me]); the memory challenger's key is a pin in the run's parameters; the wipe's tiers are names the prover's request declares; PoUB's `profile.registration` is its own; the wipe's egress is a number the challenger is started with. [relayed, memory-accounting] | shape | As G2. | Lean and code. **To Daniel**, with G2. | memory-accounting |
| N1 | The certifier's period check | SPOT's window record: its commitment is `{"audits": [record, ...], "windows": [record, ...]}`, each record of format `verity/registration/v1` or `v2`. `certifier/verifier/lean/NetworkCheck/Period.lean:86-89`, with `Flock.WindowRecord`, `Flock.Public` and `Flock.Program` imported. [read by me] | shape, import | The certifier hands SPOT a circuit, its pieces and a rule, and gets back a verdict over roots. It commits to roots, not to SPOT's records. That a compute audit's bytes crossed a counted link is compose's tie (network-accounting 1, 2 [relayed]). | code and Lean: the commitment's form changes, so its digest does. **To Daniel**. | network-accounting, proofs |
| N2 | The certifier's Lean package | The policy's: `network_check` requires `nci_check` for `NciCheck.Desc` and `NciCheck.DefinitionDigest`. Its Lake file; `certifier/verifier/lean/NetworkCheck/Period.lean:9-10`, used at 266, 541, 562, 772. [read by me] | import | The summary of a circuit's description lives with the circuit language. Neither protocol requires the other for it. | move to `core_circuits`. | network-accounting, lean, proofs |
| N3 | The certifier's verifier and prover | The serving system. The advice format is "vLLM v1, the pinned 0.28.1rc1.dev472's `Scheduler.schedule` and `SchedulerOutput`" (`certifier/verifier/advice.py:1-20`) [read by me]. Files under `certifier/` naming: request 136, vLLM 39, scheduler 21 [counted by me]. | name, job | `AGENTS.md`'s rule: the adapter registers a compiler, a circuit with a public length cap, and hands advice as bits Verity never interprets. The certifier checks the circuit and charges the length. | move the compiler and the advice format to the adapter; code. | network-accounting, vllm, circuits |
| N4 | The certifier's `verifier/` | The device's own code: `Proxy` sits in `certifier/verifier/active.py` [read by me]. The device's serving loop is under `benchmarks/network_traces/`, about 25,000 lines [relayed: the page, section 7]. | job | `device/` holds the device. `verifier/` holds the check. | move (the plan with days is due Mon 16 Nov [page]). | network-accounting |
| N5 | The certifier's link for challenges | SPOT's challenger: `certifier/verifier/coin_link.py:52-53` imports `LiveCommitment`, `LiveOpening`, `live_reason`, `HEADER` and `MAX_HEAD` from SPOT's device. [read by me] | import | The certifier counts and re-clocks frames of the shared channel and does not read whose challenges they carry. | code: the frame's header goes into `channel.py`. | network-accounting, proofs |
| N6 | The certifier's tests | The policy's difftests: 8 test files under `certifier/verifier/network/tests/`. [counted by me] | test | They go with the policy. | move | network-accounting |
| N7 | The period's parameters | One link and one key, held to no registered link. [relayed, network-accounting 12] | shape | As G2: the registration hands the certified links with their keys, and compose compares. | Lean and code. **To Daniel**, with G2. | network-accounting, soundness |
| I1 | The policy's Lean package | SPOT's insides: `nci_check` requires `core_spot`, and `inference_only/verifier/nci/lean/NciCheck/Record.lean:6` and `RecordText.lean:5` import `Spot.Wires`. [read by me] | import | The policy is handed the circuit's description [page]. It reads no record of a run. | code | network-accounting |
| I2 | The policy's directory | Two calculators (`nci/`, `rho/`). One imports SPOT's profile (`inference_only/rho/rho/params.py:38-39`) and restates the certifier's request layout (`rho/rho/envelopes.py:280`). [read by me] | job | A calculator is a benchmark. A layout's size reaches it as one public figure. | move (on the page: Fri 13 Nov). | network-accounting, training |

### 2.6 Compose and `security/`

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| X1 | Compose's remote check | Each protocol's record format: it ties Pearl+'s run to "the digest of the registration record", holds the policy's record to be "chained right after W (its `previous` is W's digest)", and reads W's `pearl-b/` values. `remote/verifier/CoreRemote.lean`, header. [read by me] | shape | Each protocol hands compose the head and the roots it committed to. Compose compares them and reads no record's fields. | Lean. **To Daniel** through soundness. | proofs, soundness |
| X2 | Compose's theorem | SPOT's and Flock's proof internals (`OpensAt`, `Layout.slotAt`, `Flock.Bridge.pathOf` under `security/Proofs/Properties/Verity/Sampled/`) [relayed, soundness 2]. That tree is 95 files and 41,677 lines [counted by me]. | reads | SPOT exports one knowledge-soundness theorem over the circuit, its pieces, the sampling rule and the challenges. The sampled leg reads only that. | Lean. **To Daniel** only if the exported statement differs from today's. | soundness, proofs |
| X3 | The security game's vocabulary | Protocols' internals: it reads the circuit's digest out of the verifier's parameters (`security/Properties/Verity/Game.lean:676`, `:781`) [read by me]. Nothing ties that digest to the circuit the registration declared (`interface/CoreProtocol/Registration.lean:94`) [relayed, soundness]. | reads | The registration's accepted record and SPOT's parameters are both inputs of the composite, and the tie (`DeclaredIsParams`) is a fact of compose's theorem. | Lean. **To Daniel**. | soundness |
| X4 | Compose's lists | Four homes: `onsite/`, `remote/` (onsite's package requires remote's), `remote/verifier/composition/`, and `verity/composite/`. [read by me] | import | One `compose/` package. Remote and onsite are two lists of ids. | move (on the page: Sat 14 Nov). | proofs |
| X5 | The composite's device | Another protocol's processes, and a benchmark: `verity/composite/device/live_challenger/coin_link.py:40` imports the certifier's link [read by me]; its `auditor.py` names the certifier's processes [relayed] and is tied to `benchmarks/` in the listing [read by me]. | import, name | Compose starts each protocol's device through one interface, read off the run's list, through the runtime. | code | proofs |
| X6 | Compose's hand-off | Each protocol's private files by name (`served_verdicts.py`, `WITNESS_*`). [relayed, memory-accounting 8] | name | Each protocol declares which of its outputs are witness. | code | proofs |
| X7 | The game's `Holds` | Protocol facts written inline for each protocol. [relayed, soundness 3] | reads | `Holds` reads each protocol's exported claim by its id on the list. | Lean | soundness |
| Y1 | `security/Definitions/Core` | An accounting module: `Core/Accel/Card.lean:4`, `Core/Blocks.lean:4` and `Core/Site.lean:3` import `Definitions.Accounting.Device`, and `security/lean-audit.json`'s `layers` allows it. [read by me] | reads | The shared vocabulary reads no area. | Lean: definitions move, statements do not change. | soundness, lean |
| Y2 | The definitions and properties trees | 28 modules of the proofs tree (`trees.except` in `security/lean-audit.json`): 21 under `Proofs.Flock`, 27 "until its spec is extracted". [counted by me] | reads | A statement never reads a proof [page]. | Lean: Flock's spec goes into `Definitions`. | lean, proofs |
| Y3 | `security/Definitions/Flock` (35 files) | SPOT's and recursion's vocabulary (`Audit/Law.lean`, `Partitioning.lean`, `Guarantees/SamplingClaims.lean`, `Recursive/Holo.lean`), and a directory named `Accounting/` that is Flock's own error budget. `Definitions/Spot` (9 files, 888 lines) imports three `Core` modules only and matches Protocol 1. [read and counted by me] | dup, name | One area for each layer: sampling in `Spot`, the outer proof in an area of recursion's, `Flock` the proof system alone. | move and rename. | proofs, lean |
| Y4 | `security/Definitions/Core` (84 files, 9,823 lines) | Things one layer owns: `Site`, `Period`, `Accel`, `DeviceFigures`, `Compose`, `RegisteredCircuit`, `Challenger`, `Compiler`, `Advice`, `Rec`, `Blocks`, `Transport`, `NaiveStrassen`. [counted by me; the files listed, not all read] | job | `Core` holds what more than one guarantee reads. Registration's, compose's, recursion's and PoUW's files go to their areas. | move | soundness, lean |
| Y5 | Shared concepts with two definitions | A resource: `security/Definitions/Accounting/Joint.lean:38` and `interface/CoreProtocol/Account.lean:103`. A protocol's id: `Definitions/Accounting/JointPeriod.lean:39` and `Properties/Verity/Game.lean:132`. A window: `Definitions/Accounting/Device.lean:81`, `NetworkCheck/Period.lean:412`. A tier: `interface/CoreProtocol/Site.lean:66` and `Definitions/Core/Site.lean:116`. [read by me] | dup | One definition each, in the layer that owns it. | Lean | soundness |
| Y6 | Guarantees | One card family, by name. A benchmark writes `security/Definitions/Accounting/Headline/Sm*.lean` (`benchmarks/machine_model/verity_machine_model/lean_headline.py:2`) [read by me]; eleven Lean files name one card's table [relayed, compute-research 1]. | name, job | One guarantee over any sheet. A card is a witness the caller supplies. No Python writes under `security/`. | Lean. **To Daniel**: a guarantee about one card becomes one about any sheet. | compute-research, compute-accounting |
| Y7 | The three trees of `security/` | They share one source directory (`srcDir = ".."` in `security/Properties/lakefile.toml:9` and `security/Proofs/lakefile.toml:4`), so the lock's globs hold them apart, not Lake. [read by me] | (a weak wall) | No fix proposed (lean 8 [relayed]). | none | lean |
| Y8 | A benchmark | The lock, by Lean declaration name: `benchmarks/pouw/harness/price_twins.py:62-67`. [read by me] | name | The Lean check hands out a guarantee's stated constant by the guarantee's id (compute-research 3 [relayed]). | code | lean, compute-research |
| L1 | The interface's profile | The proof system's draw rule: `interface/CoreProtocol/Profile.lean:70` sizes a stratum with `Flock.Draw.workK`. [read by me] | reads | The profile states what was established. The rule's check is SPOT's (lean 3 [relayed]). | Lean | proofs, lean |
| L2 | The interface | Two modules only SPOT reads: `Record.lean` and `Live.lean`. [read by me] | job | They go with SPOT, or with the channel (lean 4 [relayed]). | move | proofs |

### 2.7 The adapter, the demos and the benchmarks

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | The adapter | The verifiers and devices of six protocols. Import lines under `adapters/vllm/`: PoUW's verifier 93 (37 files), SPOT's verifier 33 (24 files), SPOT's device 22 (13), the certifier's verifier 18 (14), PoUS's verifier 17, PoUB's verifier 9, PoUS's device 3 [counted by me; vllm's own count agrees within two files]. Such as `adapters/vllm/verity_vllm/pipeline/serve_pearl.py:130-131`. | import | Rule 6: one call into a protocol's prover over GPU buffers. The adapter hands a circuit, its pieces and committed values, and is handed roots. | code (vllm's first change is Sun 15 Nov [relayed]). | vllm, each accounting lead |
| A2 | The adapter | Protocol code: `adapters/vllm/verity_vllm/protocols/pouw/` is 54 files and 16,908 lines [counted by me]; `protocols/pous.py` and `protocols/remote_bandwidth/` sit beside it. | job | PoUW's prover, its kernels and the one call are one package under PoUW [page, section 10]. | move and code | compute-accounting, vllm, kernels |
| A3 | The adapter | It decides accept and refuse: `adapters/vllm/verity_vllm/check/`, 78 files, 21,275 lines [counted by me]. | job | My answer to vllm's question is in part 5. | not sorted yet | vllm, circuits |
| A4 | The adapter | The data sheets, the tools and a benchmark: `adapters/vllm/verity_vllm/acquire/served_device.py:38`; two files in `BENCHMARKS_IMPORTERS`; the listing's tie adapters to tools. [read by me] | import | A run reads the data sheets and passes them in (rule 2). Nothing uses a benchmark (rule 9). | code | vllm |
| A5 | The adapter's serving front | The certifier's stamp, by header name: `adapters/vllm/verity_vllm/pipeline/serve_admit.py:11-13` (`x-verity-release`). [read by me] | name | One function of the certifier's prover gives "the release number of this request" (vllm 3 [relayed]). | code | vllm, network-accounting |
| A6 | The adapter's registration | It needs a workload to build, and serving settings live in a benchmark. [relayed, vllm 5, 7] | shape | The registered deployment names its settings and its build as data. | code | vllm |
| B1 | 35 files outside `benchmarks/` | A benchmark's code (`BENCHMARKS_IMPORTERS`, `tests/test_boundaries.py:126-173`) [counted by me]. A served run loads PoUW's kernel from `benchmarks/`, and the certifier's device runs from there [page, section 7]. | import | Rule 9. | move (dated on the page). | kernels, compute-accounting, network-accounting, proofs |
| B2 | A test under `verity/` | The adapter: one test imports 7 adapter modules. The same allowlist (`KNOWN`, `tests/test_integration_boundaries.py`) holds 36 more lines that match no import today. [counted by me] | test | Rule 1. | move the test; empty the list. | circuits, top |

That is 73 rows: 7 for SPOT, 7 for Flock and recursion, 4 for circuits and commitments, 12 for the shared files and
registration, 18 for the accounting protocols and the policy, 17 for compose, the interface and `security/`, and 8 for
the adapter and the benchmarks [counted by me]. 15 of them carry a fix that goes to Daniel: P1, P2, G2, G3, G4, W1, U2,
U5, U6, N1, N7, X1, X2, X3 and Y6. 7 rest on a lead's word alone (S6, W2, W3, N7, X6, X7, A6). The other 66 have at
least one place I read or counted on `main`, and each says where a part of it is relayed.

End of the first cut.
