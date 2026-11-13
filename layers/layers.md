# What each layer of Verity may know, and every coupling on `main` that breaks it

**The layers are held apart by import rules only, and three things leak through them: SPOT's record of a window, which
every protocol and the adapter read and which carries its callers' facts; the registration, which no accounting
protocol reads, so each restates what it needs; and the adapter, which holds protocol code and reads verifiers and
devices. Part 2 lists 86 couplings. Part 3 names the five that matter most.**

First whole cut, Fri 13 Nov 2026, for 5:00 AM PST (13:00 UTC). `main` was read at `9182900fb94e`. The first chain of
path moves landed after that (`e4da505fe488`), so I checked each file citation's path and line against that commit and
wrote those that moved or shifted as `main` has them now. Counts are as at the first commit; where I recounted, the
row says so. Written by LAYERS, a worker of top. Nothing was built
or run: this is `rg`, `git` and the files.

How I know each claim:

- **[read by me]**: I opened the file at that line on that commit.
- **[counted by me]**: a search I ran. Unless it says otherwise it counts files that are not tests, by whole word,
  ignoring case.
- **[relayed]**: a lead's word on TOP-415, copied in `inbox/layers-leads.md` on `cursor/verity-top-4db4`, or a sweep's
  (`layers/sweep-proofs.md` on `cursor/top-layers-sweep-proofs-5d1a` at `c6f75df4d548`, `layers/sweep-above.md` on
  `cursor/top-layers-sweep-above-8c2f` at `96c3b77635a8`). Not re-read by me. "(soundness 2)" is that lead's own row
  number, and "SWEEP-PROOFS 4a" that sweep's section.
- **[page]**: the architecture page's words (sections 1 to 7 and 10). **[SPOT page]**: Daniel's page "SPOT security
  definitions". **proposed**: mine, where the pages do not say.
- **[not checked]**: nobody read it for this document.

Words. I use the architecture page's. A **piece** is one part of a circuit's cut (the SPOT page calls it a block). The
**sampling rule** is the SPOT page's sampling design D. A **challenge** is a random value the auditor's side draws
after a commitment. **SPOT's record of a window** (after its first use, SPOT's record) is the record SPOT's code calls
a "registration record". The page says it takes another name, and proofs names it in its 6 AM PST line [relayed].
**The registration** is only what the developer registers once, before any period. The **head** of a registration is
the 64-byte digest that names an accepted registration (the word is `CoreProtocol/Account.lean`'s).

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
- Must not know: which protocol asked [page]. Proposed: nor SPOT's record of a window or a registration.
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
  not SPOT's record of a window, and not a request's fields, a scheduler or a serving system.
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
| P1 | SPOT's record of a window | Its callers' facts. Of its 24 fields, 10 are SPOT's own (the circuit, the sampling rule, the commitment's form) and 14 were added for callers, each optional: the declared day, the stream and the prover, the seed salt, the pool with its week, lag and reset, and pinned, held, supplied and output values. `spot/verifier/lean/Spot/Registration.lean:163-213` (the record), `:545-614` (its check: `R2-declared` 573-574, `R6-stream` and `R6-prover` 599-601, `R6-seed-salt` 603, `R7-pool-chain` 611), `:131-139` (`Pool`). Python twin: `spot/verifier/one_stage/registration.py`, 874 lines. [read by me; the 10 and 14 split is my reading of the field comments] proofs' own table, by check code, agrees: the caller's are the declared day, the window's kind, the stream and the prover, the pool, and every look back at an earlier window; the seed salt is PoUW's; eight parameters exist only for those rows [relayed, proofs]. | shape | SPOT is handed the circuit with its pieces, the sampling rule and the commitment: a list of roots, and for a part of the trace a caller already fixed, that part's root. The caller checks its own facts (the day, the kind of window, the pool) and hands SPOT roots only. SPOT reads no log of earlier windows. One thing stays SPOT's: the record is fixed before the challenger opens the session, which is what makes the challenges live (proofs [relayed]). | code and Lean: split the record in two, SPOT's public input and the caller's record over it. R2, R6's stream, prover and salt, and R7 go to the protocol that calls SPOT (soundness says the same of R2 [relayed]). **To Daniel** where a guarantee cites one of those checks. | proofs, soundness |
| P2 | SPOT's challenger, `spot/device/challenger.py` (2,171 lines) | The auditor's ledger and two other parts' records: a period's open and close, a day's declarations, which window counts toward a period, the pins of `main`'s first-parent line, "a lane's day", PoUW's live window as "the first client", and recursion's run record. Lines 95-112, 137-166, 199-201, 304 (`PERIOD_KINDS`), 1296 (`verify_periods`). It imports the claims' ids (219) and the hardware's record entry (220). [read by me] The ledger's entries are a reserve, an open, an end, a period's open and close, a declaration and the pins log; compose's device writes them and signs them under the challenger's key (`verity/composite/device/live_challenger/auditor.py`) [relayed, SWEEP-PROOFS 5a]. | job | The challenger is handed a commitment's digest, draws challenges after it and signs both. The period's ledger is compose's. A day's declarations are registration's, signed by registration's device. A run record is recursion's. | code: three files from one. **To Daniel** if which device signs the declarations is part of a premise (soundness 8 [relayed]). | proofs, soundness |
| P3 | SPOT's Python | Three things below or beside it that it should not load: the claims' ids, the runtime, the hardware's record entry. `tests/import_order.toml`, SPOT's `loads` (`verity/claims`, `verity/runtime`, `verity/hardware/record.py`); `spot/verifier/parties.py:126`; `spot/device/challenger.py:219-220`. [read by me] | import | SPOT loads circuits, commitments and the shared files. | code: the record entry goes into `channel.py` (proofs [relayed]); the party types come from `protocol.py`; `claims` goes (S5). | proofs |
| P4 | SPOT's code and docs | Names from above it. Lines in SPOT's files: pool 316 (7 files), lane 144 (9), PoUW 142 (23), week 96 (8), certifier 39 (12), vLLM 25 (8) [counted by me]. `spot/verifier/second_stage.py:838` (`INTERIOR_ROOT = "verity-vllm/interior-run/v1"`); `spot/verifier/one_stage/proofs.py:66` (imports the served circuits' tables); the verdict's field `rests_on_period_check`, the certifier's check by name (`Spot/Pieces.lean:247`). [read by me] | name | SPOT's words are circuit, piece, sampling rule, root, challenge, proof and bound. A caller's tag reaches it as bytes. Its verdict lists the roots it did not check itself, and the caller says what they rest on. | rename and code, mostly carried out by P1 and P2. | proofs |
| P5 | Every caller of SPOT | SPOT's insides, because SPOT states no entry. Lean: 9 inner modules are imported from outside (`pouw/verifier/lean/check/PouwCheck/Credit.lean:17-20`, `PouwWindow/Rounds.lean:4`, `inference_only/verifier/nci/lean/NciCheck/Record.lean:6`). Python: `spot.verifier.one_stage` is imported in 153 lines outside SPOT; the certifier imports SPOT in 29 lines of 13 files, PoUW in 23 of 12, PoUB in 10 of 8. [counted by me] 14 Lean files outside SPOT name `Spot.Registration`, ten of them under `security/Proofs` [relayed, proofs 2]. SPOT's README lists a core API, and the three files with the most outside importers are not on it: the challenger (45 code files), `device/remote.py` (20) and the produced rows (19) [relayed, SWEEP-PROOFS 4a, 3b]. | import | One entry in Lean and one in Python: circuit, pieces, sampling rule, commitment, challenges, proofs in; accept or reject, the bound and the roots out. Callers import that and nothing else. | code: name the entry; a test holds the rest private (T5). | proofs |
| P6 | Recursion's package | It defines SPOT's protocol: the protocol compose lists as `verity/sampled-proofs/v1` is `VStar.Recursive.Sampled.protocol`, so recursion requires SPOT. `recursion/verifier/recursive/VStarRecursive/Sampled.lean:33,81`; its Lake file; `remote/verifier/CoreRemote.lean:37,116`. [read by me] | import | SPOT's protocol is defined in SPOT's package. Recursion is handed Flock's proofs and one statement, and does not know which protocol asked. | move `Sampled.lean`. Whether recursion may still require SPOT is a question for proofs (part 5). | proofs |
| P7 | SPOT's Lean package | It holds the registered hardware's difftest. `spot/verifier/lean/Spot/SiteDifftest.lean:3`. [read by me] | test | The test goes with registration. | move (on the page, with registration). | proofs |
| P8 | SPOT's parties | Compose's two lists, with every protocol's id: `spot/verifier/parties.py:146-152` (`SUBPROTOCOLS`: onsite's seven and remote's three, each with its id), `:123` (imports the configuration's `COMPOSITES`), `:10-14` ("The core deployment ... wraps SPOT[Flock] for both composites ... Its verifier composes each block's audit with the composite's other subprotocols"). [read by me; found by SWEEP-PROOFS] | name, job | SPOT's parties name SPOT's own roles. The lists and the composed verifier are compose's. | move the table and the core deployment to compose. | proofs |
| P9 | PoUW, PoUB, the certifier, compose's device and the adapter | SPOT's challenger as a library: 45 code files outside the proofs import `spot/device/challenger.py` (the adapter 13, PoUW 4, PoUB 4, compose's device 3, benchmarks the rest), for the live challenge's types and for the ledger's format. 19 import SPOT's produced-rows format, 8 of them the certifier's, which builds its network rows on it. [relayed, SWEEP-PROOFS 4a] PoUW's Lean check opens the ledger's entries itself (`pouw/verifier/lean/check/PouwCheck/Pass.lean:51`) [read by me]. | import, shape | The types of a live challenge (its commitment, its opening, its signed record) are the shared channel's. A row format several protocols build on is commitments'. No protocol reads the ledger: compose hands it "this window counts". | move the types to `channel.py` and the row format to commitments; code. | proofs |

### 2.2 The proofs: Flock and recursion

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | Flock's verifier | The format and digest of SPOT's record, defined there "so that every reader reaches it without a new requirement", and the registered reads ("a program's weights"). `flock/verifier/Flock/WindowRecord.lean:7-16,26-28`; `Flock/Registered.lean`. [read by me] | name, job | Flock is handed a circuit, commitments and challenges. A record of what is fixed before the challenges is SPOT's input. | move to SPOT's package, after P1. | proofs |
| F2 | Flock's Lean package, `flock_verifier` | It is three parts in one: of 57,859 lines in 269 files, recursion is 10,800 (`Rec*` 24 files, `VStar` 21) and the ZK server 8,409 (21 files). 14 packages require it (`tests/test_boundaries.py:116`), several for helpers only: the certifier's check imports `Flock.Hm96`, `FV3`, `Canon`, `Sizes`, `Bytes`, `WindowRecord`, `Public`, `Program`. [counted by me] Of the 137 Lean imports of the proofs' packages from outside, 42 (in 33 files) are of 16 one-line modules that only re-export a base package under Flock's name, 88 are of inner modules and 7 of a root [relayed, SWEEP-PROOFS 4b]. | import | Hashes, canonical bytes and commitments come from the three base packages. Flock's package is the proof system. Recursion and the ZK server are recursion's. A check that verifies no Flock proof requires none of it. | rename the 42 imports to the base packages first (33 files, no record changes). Then move the Lean modules between packages (the lock's records move, no statement changes). | proofs, lean |
| F3 | Flock's verifier | A table of served models that a benchmark writes. `flock/verifier/VStar/Menu/Pinned.lean:5-7` ("Generated by `benchmarks/numerical/...`"), `:15-103` (`"vllm/gemma2-2b__bf16__l40s__..."` and five more model families); imported at `VStar/Cli.lean:5`. [read by me] | name | The cost table is a file the caller hands in: measured proving costs "are handed to it as data" [page, section 10]. | code: read it as a parameter; delete the generated module. | proofs, benchmarks |
| F4 | Flock's prover (`flock/prover/`, the distribution `verity_flock`) | Recursion's prover and part of SPOT's: 17 `rec_*.py` modules, `pod/rec_inner.py`, `inner_registered.py`. It imports SPOT's verifier in 42 lines of 14 files and the served circuits in 103 lines of 36 files. The import listing therefore puts Flock's prover above SPOT. `tests/import_order.toml` (the order; SPOT's `loaded_by`). [read and counted by me] | import | Flock's prover: a circuit, commitments, challenges in; a proof out. Recursion's prover sits in `recursion/prover/`. Then the listing's order is flock, spot, recursion, as the page draws it. | move | proofs |
| F5 | Flock's verifier and its command line | Names of PoUW and the adapter. `flock/verifier/FlockVerify.lean:344` (`--work-table`, "the PoUW work"); `Flock/Firewall/Salts.lean:51` (`PouwCheck.Credit.registeredFile`); `Flock/Sigma.lean:4` (`verity_vllm`). Files under `flock/` naming: registration 59, window 38, vLLM 29, PoUW 19. [read and counted by me] The Rust prover has a binary named for vLLM, whose block format cites the adapter's `PROTOCOL.md` (`flock/prover/live/src/bin/flock-vllm-v1.rs`, `vllm_block.rs:2`) [relayed, SWEEP-PROOFS 2]. | name | As F1. | rename, after F1 to F4. | proofs |
| F6 | Flock | Tools and benchmarks: the listing's ties flock to tools (3 files) and flock to benchmarks (7 files). `tests/import_order.toml`, `[[against]]`. [read by me] And 165 path strings in Flock's and SPOT's code name what is above them: the team's tooling 39, `benchmarks/numerical` 30, the adapter 22, PoUW 12; scripts put the adapter and a benchmark on the Python path (`flock/tool.py:60`, `flock/pod/92-zk-sessions.sh:18-19`) [relayed, SWEEP-PROOFS 1f]. | import | Nothing uses a benchmark (rule 9). The pod scripts are the team's tooling, outside `verity/`. | code and move | proofs |
| F7 | Flock's verifier | It accepts a statement whose Program nobody holds to the circuit, and says so: `flock/verifier/Flock/ProvedScope.lean:655` ("the Program was handed and not held to the circuit"). [read by me] Which layer holds the Program to the circuit is not settled in code (proofs 5 [relayed]). | job | One layer owns it. Proposed: Flock, since "one statement about one circuit" is its input [page]; a caller hands the circuit, and never a Program beside it. | Lean. **To Daniel** if a deployed guarantee covers such a statement. | proofs |
| F8 | The base Lean packages | They carry one proof system's name: `namespace Flock` or `open Flock` in 7 of 8 files of `core_base`, 9 of 10 of `core_circuits`, 2 of 3 of `core_commitments` and 14 of 19 of the interface. [counted by me] | name | The circuit language and the hash name no proof system. | rename. Many records would change for no change in meaning, so I list it and do not propose it now. | lean |

### 2.3 Circuits and commitments

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | The circuits written in the language, `verity/definitions/` | PyTorch, vLLM and models. Files naming vLLM 195, PyTorch 89, a model 111 [counted by me, by substring]. | name | Rule 1. The served kernels' circuits go with the adapter that serves them. | move (on the page: Sun 15 Nov). | top, vllm, circuits |
| C2 | The served circuits, the adapter and the circuit checker | The cut of a circuit into pieces, stated three times, each sized to Flock's piece: `CUTS` in nine modules of `verity/definitions/registry/` (`greedy_stages.py:253`); a second table by circuit name in the adapter (`adapters/vllm/verity_vllm/pipeline/serving_templates.py:16`); the checker's "C-Flock's proved range of 2^27 bits" (`tools/verity/circuit_check/src/circuit_check/block_held.py:1-3`, `checks.py:1445`). [read by me; found by circuits 1] | dup | A circuit knows no piece. One function takes a circuit and a cap and returns its pieces. The cap is a parameter from whoever calls SPOT. | code | circuits, proofs for the cap |
| C3 | Flock's prover, and the circuit checker | The lowering of a circuit to Boolean gates lives in the proof system's prover, so the checker imports the prover: 30 import lines of `verity_flock` in 5 files of `tools/verity/circuit_check/src/circuit_check/` (`checks.py` has 18) [counted by me]. And Flock's prover names single served circuits (F4; circuits 2 [relayed]). | job | The Boolean circuit is the one thing handed over. The lowering is the circuit language's. The proof system takes any Boolean circuit and names none. | move the lowering to `verity/circuits/`. | circuits, proofs |
| C4 | The circuit checker's suite | Protocols' verifiers and devices: 148 lines of its declared inputs name paths under `verity/protocols/`, `adapters/` or `benchmarks/`. [relayed, circuits 3] | test | A protocol's circuits sit in a module that imports circuits only, so loading one loads no verifier and no device. | code | circuits |
| C5 | The circuit checker and one named circuit | One model, where any circuit should do: `tools/verity/circuit_check/src/circuit_check/block_held.py:11` (`_GLM53_TREES`); `verity/definitions/ops/prims.py:140-142` (a circuit's text names GLM-5.3's tiles). [read by me; found by circuits 7] | name | The model's shapes and pieces come from its compile as one object, which the checker takes as input. | code | circuits |
| C6 | The served circuits' tests | PoUW's and PoUB's code (`verity/definitions/tests/test_pearl_c_call.py:29-31` and four more). [relayed, circuits 6] | test | Each test sits with the protocol whose circuit it tests. | move | circuits |
| M1 | `verity/commitments/domains.py` (409 lines, 341 constants) | Every protocol's and the adapter's format strings, in one table: PoUW's (lines 19-126), SPOT's, Flock's, the ZK server's, PoUS's, PoUB's, the certifier's, compose's, the wire's, and vLLM's (`VLLM_*` at 182-196 and 397-400, such as `"verity-vllm/pouw-pearl-c-run/v1"`). Lean twin: `verity/commitments/identity/Identity/Domains/`. [read by me] By section: PoUW 91 constants, the certifier 49, PoUS 44, PoUB 27, SPOT 23, Flock and the ZK server 23, commitments' own 24 [relayed, SWEEP-ABOVE 3c]. | name | Commitments is handed a tag as bytes. Each protocol defines its own tags, and a test of the whole tree holds them distinct. | code: split the table by owner (T3). | proofs (proposed), each lead for its lines |
| M2 | Commitments | Modules shaped for one caller: `window.py` (`WINDOW_DOMAIN = "verity/one-stage/served-domain/v1"`, which binds program, partition, port, schema, leaves and run), `advice.py`, `history.py`, `pool_row.py`, and `windows/` (7 modules: "what an adapter hands the prover for each execution phase"), which loads the served circuits' catalog (`tests/import_order.toml`). [read by me] | shape | Values in, roots and openings out. What an adapter hands a prover is that prover's interface. | move: `windows/` to SPOT's `prover/`; `advice.py` and `pool_row.py` to their callers. | proofs, vllm |
| M3 | Commitments' tests | The adapter's files, listed as test inputs (`verity/commitments/pyproject.toml:10-11`) [read by me]. One test imports PoUB's verifier, and the listing counts two files of PoUB's verifier as commitments' (`verity/commitments/tests/test_block_check.py:17-24`; `poub/verifier/commitments/gates/block_check.py`) [relayed, SWEEP-ABOVE 3c]. | test | A package's tests read its own package. | code and move | proofs, memory-accounting |
| M4 | PoUS, PoUB, the ZK server | A commitment's root, computed outside commitments: PoUS's own Merkle tree (`pous/verifier/check/PousCheck/Merkle.lean:6-9,76`), which PoUB imports (`poub/verifier/lean/RemoteAudit/P2.lean:3`) [read by me]; the ZK server's frame tree, the memory challenger's challenge tree, and PoUB's stream function twice in one package [relayed, SWEEP-ABOVE 4e, 4f]. | dup | Commitments hands back roots. A protocol hands values. One tree, the shared one (`AGENTS.md`: one hash, one kind of tree). | Lean and code. **To Daniel** if a guarantee's hash premise changes with the tree. | memory-accounting, proofs |

### 2.4 The shared files, the runtime and registration

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Nine protocol files | The runtime's package, for a type. `ProtocolDef` (`from verity.runtime.runner import ProtocolDef`): `pous/device/challenger_party.py:66`, `poub/verifier/session_parties.py:47`, `spot/verifier/parties.py:126`, and one more. `stream.Side`: 5 files, such as `poub/verifier/verifier_party.py:34`. [read by me] `ProtocolDef` is 5 lines: a manifest and one factory per party (`verity/runtime/runner.py:73-77`). | import | The types sit in `protocol.py` and `channel.py`. The runtime calls a protocol, and no protocol imports the runtime (rule 3). | move the types (memory's part is MEM-291 [relayed]). | proofs, memory-accounting |
| S2 | The configuration's reader | Compose's two lists, with every protocol's name, and it refuses any other list in Python. `verity/config.py:59-61` (`COMPOSITES`: onsite's 7 names, remote's 3). [read by me] With P8 and G3 the lists are stated four times: compose's Lean (`onsite/verifier/CoreOnsite.lean:63-65`), the configuration, SPOT's parties and the registered hardware. | name, job, dup | `config.py` reads a list of protocol ids. Whether a list is acceptable is compose's to say, in Lean [page]. | code | proofs |
| S3 | The hardware package | The transcript's signed entry, `verity/hardware/record.py`, which SPOT's challenger reads. [read by me] | job | A signed entry of a transcript is the channel's. | move into `channel.py` (proofs [relayed]). | proofs |
| S4 | The period | The period's type three times, and a fourth and fifth by name. The lengths (a week, a month, a year) are stated once now, in `verity/protocols/reporting.py:11`; the second list went with the first chain of moves. The type: `security/Definitions/Core/Period.lean`, its stated copy `interface/CoreProtocol/Period.lean`, and the Python mirror. [read by me] Two more types named `MonitoringPeriod` sit in SPOT's challenger and in the slowdown calculator (`spot/verifier/lean/Spot/ChallengerSession.lean:481`, `inference_only/rho/rho/hardware.py:101`) [relayed, SWEEP-ABOVE 4d]. | dup | One period type, beside the account it is summed in. | code | proofs (proposed) |
| S5 | `verity/claims` (623 lines) | A list of claim ids kept by hand beside the locks. 10 Python files read it [counted by me]; SPOT's challenger uses it for one membership test (`spot/device/challenger.py:378`). About 80 Lean docstrings and three locks name its ids [relayed, top]; SWEEP-ABOVE counts 308 Lean files that cite one of its 207 ids. PoUS's verifier keeps a second table of 9 claims (`pous/verifier/protocol.py:45`), and PoUW's provers write claim ids as literals [relayed, SWEEP-ABOVE 6]. | dup | The locks are the one registry of guarantees and assumptions. A generated file maps an id to its Lean name (soundness [relayed]). | code: a generator in the Lean check; the Glossary's line on a guarantee changes. | lean, soundness |
| S6 | The import listing | One row holds `channel.py`, `config.py`, `party.py`, `protocol.py` and `runtime`, and that row sits below every protocol. One place holds the runtime with `config.py`, `party.py` and `protocol.py`, so a load of a shared type reads as a load of the runtime, and SPOT's `loads` names `verity/runtime`. `tests/import_order.toml`, `order` and the places. [read by me, on `e4da505fe488`] | test | The shared files are a row below every protocol. The runtime is a row above them all. | code (TOP-398 [relayed]). | top |
| S7 | The hardware directory | The devices' outside world: a Lean package of clock, random bytes and sleep, "the outside world, as a core device or challenger reaches it" (`verity/hardware/world/World.lean:1-12`), which the certifier's device requires (`certifier/device/lean/lakefile.toml:23` [relayed, SWEEP-ABOVE 2a]). [read by me] | job | Sockets, randomness and the clock are the runtime's [page]. The hardware directory is data only (rule 2). | move to the runtime. | proofs |
| G1 | The interface package | Registration, the registered hardware, compose's account and the period: `Site.lean` 702 lines, `Registration.lean` 535, `Account.lean` 861, `Period.lean` 147, of the package's 3,717. Outside the package `Site` is read by 3 files, `Period` by 1, `Registration` and `Account` by none. `interface/CoreProtocol/`. [counted by me; lean's table says the same, relayed] 13 Lake packages require the package, and its one library target builds all of it for each (SWEEP-ABOVE 1 [relayed]). | job | The interface hands back only what a check implements. Registration is a package of its own on it. The account is compose's. | move: cut the Lean out (about 45 files by proofs' sizing [relayed]). | proofs, soundness |
| G2 | PoUW, PoUS, PoUB, the certifier | None reads the accepted registration. No file of a protocol imports the registered hardware or the registration log in Python. In Lean one module does (`certifier/verifier/lean/NetworkCheck/Registered.lean:3`) and only a test's vector file calls it. [counted by me] An answer that names no head counts for nothing in the account (`interface/CoreProtocol/Account.lean:280-283`) [read by me]. The page says "None does". | shape | Each accounting protocol is handed the registration's head and its part, and its answer names both. Compose joins answers by comparing them. | code and Lean: each published answer gains two fields. **To Daniel**: it changes what each answer says. | compute-accounting, memory-accounting, network-accounting; proofs for the type |
| G3 | The registered hardware | Compose's two lists, by name. `interface/CoreProtocol/Site.lean:285-286` (`Composite`: onsite, remote), `:372,376` (`Covered` and `Reached` hold "if the composite is onsite"); `verity/device.py:61`. [read by me] | name | Registration is handed the run's list and asks whether it holds a protocol that accounts for this link or this memory. | Lean: restate both over the list (soundness 4 [relayed]). **To Daniel**. | proofs, soundness |
| G4 | The adapter, PoUS, PoUB | The weights are committed three times with nothing tying the roots: the served run's root for each weight, PoUS's one hash, PoUB's tree [relayed, memory-accounting 3, vllm 4]. Compose's own text says of two of its forms: "nothing here holds PoUW's committed weights to the weights the record registers" (`remote/verifier/CoreRemote.lean`, header) [read by me]. | dup | One commitment to the weights, in the registration. Each protocol is handed that root and compares. | code and Lean. **To Daniel**: it adds a tie a guarantee lacks. | proofs, memory-accounting, compute-accounting, vllm |
| G5 | The data sheet | Four Lean definitions, one set of Python literals, and the TOML: `interface/CoreProtocol/Site.lean:150`, `security/Definitions/Core/Site.lean:137`, `security/Definitions/Accounting/TS/Sheet.lean:68`, `security/Definitions/Core/Accel/Card.lean:31`; `pouw/prover/circuit/h1device.py:19-23`; `verity/hardware/cards/`. The hardware package also writes Lean under `security/`, and a card names its Lean module (`verity/hardware/lean.py:21-22`, `cards/h200-sxm-141gb.toml:14`). [read by me] | dup | One type. A statement takes a sheet as a parameter. The registration check holds a device's figures to the auditor's sheet [page, section 10]. | Lean and code (on the page: Tue 24 Nov). | compute-research, proofs |
| G6 | PoUW's check and PoUB's device | A list of served models, by file path. `pouw/verifier/lean/check/PouwCheck/Credit.lean:1634` (`include_str "../models.json"`); `poub/device/live_verifier/host.py:165` (`MODEL_PINS`, PoUW's file). [read by me] The file pins 7 models' checkpoints and has five readers, compose's device among them (`verity/composite/device/live_challenger/models.py:53`) [relayed, SWEEP-ABOVE 3b]. | name, import | The registration hands the registered circuit's shapes and the weights' roots, as data. | code | compute-accounting, memory-accounting |

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
| N1 | The certifier's period check | SPOT's record of a window: its commitment is `{"audits": [record, ...], "windows": [record, ...]}`, each record of format `verity/registration/v1` or `v2`. `certifier/verifier/lean/NetworkCheck/Period.lean:86-89`, with `Flock.WindowRecord`, `Flock.Public` and `Flock.Program` imported. [read by me] | shape, import | The certifier hands SPOT a circuit, its pieces and a rule, and gets back a verdict over roots. It commits to roots, not to SPOT's records. That a compute audit's bytes crossed a counted link is compose's tie (network-accounting 1, 2 [relayed]). | code and Lean: the commitment's form changes, so its digest does. **To Daniel**. | network-accounting, proofs |
| N2 | The certifier's Lean package | The policy's: `network_check` requires `nci_check` for `NciCheck.Desc` and `NciCheck.DefinitionDigest`. Its Lake file; `certifier/verifier/lean/NetworkCheck/Period.lean:9-10`, used at 266, 541, 562, 772. The listing's order allows it: the policy's row is below the certifier's (`tests/import_order.toml`, `order`), where the page puts the policy on top of accounting. [read by me] | import | The summary of a circuit's description lives with the circuit language. Neither protocol requires the other for it. | move to `core_circuits`. | network-accounting, lean, proofs |
| N3 | The certifier's verifier and prover | The serving system. The advice format is "vLLM v1, the pinned 0.28.1rc1.dev472's `Scheduler.schedule` and `SchedulerOutput`" (`certifier/verifier/advice.py:1-20`) [read by me]. Files under `certifier/` naming: request 136, vLLM 39, scheduler 21 [counted by me]. One step of that build's scheduler is a circuit in the certifier's verifier (`certifier/verifier/replay_step.py:1-15`) [relayed, SWEEP-ABOVE 3b]. | name, job | `AGENTS.md`'s rule: the adapter registers a compiler, a circuit with a public length cap, and hands advice as bits Verity never interprets. The certifier checks the circuit and charges the length. | move the compiler and the advice format to the adapter; code. | network-accounting, vllm, circuits |
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
| X5 | The composite's device | Another protocol's processes, and a benchmark: `verity/composite/device/live_challenger/coin_link.py:40` imports the certifier's link [read by me]; its `auditor.py` names the certifier's processes [relayed] and is tied to `benchmarks/` in the listing [read by me]; its `models.py:45-46` imports PoUW's verifier [relayed, SWEEP-ABOVE 2c]. | import, name | Compose starts each protocol's device through one interface, read off the run's list, through the runtime. | code | proofs |
| X6 | Compose's hand-off | Each protocol's private files by name (`served_verdicts.py`, `WITNESS_*`). [relayed, memory-accounting 8] | name | Each protocol declares which of its outputs are witness. | code | proofs |
| X7 | The game's `Holds` | Protocol facts written inline for each protocol. [relayed, soundness 3] | reads | `Holds` reads each protocol's exported claim by its id on the list. | Lean | soundness |
| Y1 | `security/Definitions/Core` | An accounting module: `Core/Accel/Card.lean:4`, `Core/Blocks.lean:4` and `Core/Site.lean:3` import `Definitions.Accounting.Device`, and `security/lean-audit.json`'s `layers` allows it. [read by me] | reads | The shared vocabulary reads no area. | Lean: definitions move, statements do not change. | soundness, lean |
| Y2 | The definitions and properties trees | 28 modules of the proofs tree (`trees.except` in `security/lean-audit.json`): 21 under `Proofs.Flock`, 27 "until its spec is extracted". [counted by me] | reads | A statement never reads a proof [page]. | Lean: Flock's spec goes into `Definitions`. | lean, proofs |
| Y3 | `security/Definitions/Flock` (35 files) | SPOT's and recursion's vocabulary (`Audit/Law.lean`, `Partitioning.lean`, `Guarantees/SamplingClaims.lean`, `Recursive/Holo.lean`), and a directory named `Accounting/` that is Flock's own error budget. `Definitions/Spot` (9 files, 888 lines) imports three `Core` modules only and matches Protocol 1. [read and counted by me] A third home: `security/Definitions/Claims` (23 files) holds Flock's claim statements, SPOT's sampled soundness among them [relayed, SWEEP-PROOFS 1e]. | dup, name | One area for each layer: sampling in `Spot`, the outer proof in an area of recursion's, `Flock` the proof system alone. | move and rename. | proofs, lean |
| Y4 | `security/Definitions/Core` (84 files, 9,823 lines) | Things one layer owns: `Site`, `Period`, `Accel`, `DeviceFigures`, `Compose`, `RegisteredCircuit`, `Challenger`, `Compiler`, `Advice`, `Rec`, `Blocks`, `Transport`, `NaiveStrassen`. [counted by me; the files listed, not all read] | job | `Core` holds what more than one guarantee reads. Registration's, compose's, recursion's and PoUW's files go to their areas. | move | soundness, lean |
| Y5 | Shared concepts with two definitions | A resource: `security/Definitions/Accounting/Joint.lean:38` and `interface/CoreProtocol/Account.lean:103`. A protocol's id: `Definitions/Accounting/JointPeriod.lean:39` and `Properties/Verity/Game.lean:132`. A window: `Definitions/Accounting/Device.lean:81`, `NetworkCheck/Period.lean:412`. A tier: `interface/CoreProtocol/Site.lean:66` and `Definitions/Core/Site.lean:116`. [read by me] In Python the memory tiers are listed five times, and not alike: the registered hardware's, the certifier's and the policy's say hbm, dram, disk, and PoUB's says hbm, host (`poub/verifier/session/verifier.py:153`). 26 Lean definitions are named `Registration`. [relayed, SWEEP-ABOVE 4b, 4h] | dup | One definition each, in the layer that owns it. | Lean | soundness |
| Y6 | Guarantees | One card family, by name. A benchmark writes `security/Definitions/Accounting/Headline/Sm*.lean` (`benchmarks/machine_model/verity_machine_model/lean_headline.py:2`) [read by me]; eleven Lean files name one card's table [relayed, compute-research 1]. | name, job | One guarantee over any sheet. A card is a witness the caller supplies. No Python writes under `security/`. | Lean. **To Daniel**: a guarantee about one card becomes one about any sheet. | compute-research, compute-accounting |
| Y7 | The three trees of `security/` | They share one source directory (`srcDir = ".."` in `security/Properties/lakefile.toml:9` and `security/Proofs/lakefile.toml:4`), so the lock's globs hold them apart, not Lake. [read by me] | (a weak wall) | No fix proposed (lean 8 [relayed]). | none | lean |
| Y8 | A benchmark | The lock, by Lean declaration name: `benchmarks/pouw/harness/price_twins.py:62-67`. [read by me] | name | The Lean check hands out a guarantee's stated constant by the guarantee's id (compute-research 3 [relayed]). | code | lean, compute-research |
| Y9 | Flock's proofs | The certifier's name: Flock's byte lemmas are declared inside `namespace Properties.NetworkCertifier.Bytes`, so that the certifier's lemmas "build on these, under the same names"; 21 files of `security/Proofs/Flock` open it. `security/Proofs/Flock/Bytes.lean:16-17,26`. [read by me; found by SWEEP-PROOFS 1e] | name | Byte lemmas are the hash area's, under its own name. | rename | proofs, lean |
| Y10 | The accounting areas' and the policy's proofs | The proof system's inner modules: `security/Proofs/Accounting` imports `Flock.ReadNet.*` and four more in 11 files, and `security/Proofs/Properties/Nci` imports `Spot.WiresSound` and three more in 10. In the other direction one Flock proof imports an accounting definition (`security/Proofs/Flock/SoundAtBridge/Union.lean:3`). [relayed, SWEEP-PROOFS 4b, 1e] | reads | A protocol's proof reads the proofs' exported theorems, not their inner modules (T5, T8). | Lean | soundness, proofs |
| Y11 | Accounting's guarantees and the policy's | Each other's definitions, both ways: `security/Definitions/Accounting/Guarantees/Actual.lean:5-6` imports `Definitions.Nci.Reads` and `Nci.Assumptions`, and `security/Definitions/Nci/Guarantees/FitRate.lean:3-4` imports `Definitions.Accounting.Actual` and `Physical`. [read by me; found by SWEEP-ABOVE 5b] | reads | The policy sits on accounting [page]. An accounting guarantee reads no policy definition; what joins the two is compose's or the slowdown bound's. | Lean. **To Daniel** if the joined guarantee changes area and so owner. | soundness, training, network-accounting |
| L1 | The interface's profile | The proof system's draw rule: `interface/CoreProtocol/Profile.lean:70` sizes a stratum with `Flock.Draw.workK`. [read by me] | reads | The profile states what was established. The rule's check is SPOT's (lean 3 [relayed]). | Lean | proofs, lean |
| L2 | The interface | Two modules only SPOT reads: `Record.lean` and `Live.lean`. [read by me] | job | They go with SPOT, or with the channel (lean 4 [relayed]). | move | proofs |

### 2.7 The adapter, the demos and the benchmarks

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | The adapter | The verifiers and devices of six protocols. Import lines under `adapters/vllm/`: PoUW's verifier 93 (37 files), SPOT's verifier 33 (24 files), SPOT's device 22 (13), the certifier's verifier 18 (14), PoUS's verifier 17, PoUB's verifier 9, PoUS's device 3 [counted by me; vllm's own count agrees within two files]. Such as `adapters/vllm/verity_vllm/pipeline/serve_pearl.py:130-131`. The builder of a served window's circuit reads SPOT's record, the certifier's advice and PoUW's schemes (`program/predict/window_weights.py:29`, `steps.py:122`, `window_class.py:39`; circuits 4 [relayed]). | import | Rule 6: one call into a protocol's prover over GPU buffers. The adapter hands a circuit, its pieces and committed values, and is handed roots. | code (vllm's first change is Sun 15 Nov [relayed]). | vllm, each accounting lead |
| A2 | The adapter | Protocol code: `adapters/vllm/verity_vllm/protocols/pouw/` is 54 files and 16,908 lines [counted by me]; `protocols/pous.py` and `protocols/remote_bandwidth/` sit beside it. | job | PoUW's prover, its kernels and the one call are one package under PoUW [page, section 10]. | move and code | compute-accounting, vllm, kernels |
| A3 | The adapter | It decides accept and refuse: `adapters/vllm/verity_vllm/check/`, 78 files, 21,275 lines [counted by me]. | job | My answer to vllm's question is in part 5. | not sorted yet | vllm, circuits |
| A4 | The adapter | The data sheets, the tools and a benchmark: `adapters/vllm/verity_vllm/acquire/served_device.py:41`; two files in `BENCHMARKS_IMPORTERS`; the listing's tie adapters to tools. [read by me] | import | A run reads the data sheets and passes them in (rule 2). Nothing uses a benchmark (rule 9). | code | vllm |
| A5 | The adapter's serving front | The certifier's stamp, by header name: `adapters/vllm/verity_vllm/pipeline/serve_admit.py:11-13` (`x-verity-release`). [read by me] | name | One function of the certifier's prover gives "the release number of this request" (vllm 3 [relayed]). | code | vllm, network-accounting |
| A6 | The adapter's registration | It needs a workload to build, and serving settings live in a benchmark. [relayed, vllm 5, 7] | shape | The registered deployment names its settings and its build as data. | code | vllm |
| B1 | 35 files outside `benchmarks/` | A benchmark's code (`BENCHMARKS_IMPORTERS`, `tests/test_boundaries.py:127-174`) [counted by me]. A served run loads PoUW's kernel from `benchmarks/`, and the certifier's device runs from there [page, section 7]. | import | Rule 9. | move (dated on the page). | kernels, compute-accounting, network-accounting, proofs |
| B2 | A test under `verity/` | The adapter: one test imports 7 adapter modules. The same allowlist (`KNOWN`, `tests/test_integration_boundaries.py`) holds 36 more lines that match no import today. [counted by me] | test | Rule 1. | move the test; empty the list. | circuits, top |

That is 86 rows: 9 for SPOT, 8 for Flock and recursion, 10 for circuits and commitments, 13 for the shared files and
registration, 18 for the accounting protocols and the policy, 20 for compose, the interface and `security/`, and 8 for
the adapter and the benchmarks [counted by me]. 18 of them carry a fix that goes to Daniel: P1, P2, F7, G2, G3, G4, W1,
U2, U5, U6, N1, N7, X1, X2, X3 and Y6, and two only if the fix turns out to change a premise or an owner (M4, Y11).
9 rest on another's word alone (W2, W3, N7, X6, X7, A6, C4, C6 on a lead's; Y10 on a sweep's). The other 77 have at
least one place I read or counted on `main`, and each says where a part of it is relayed.

Seven rows came from the two sweeps (P8, P9, M4, S7, Y9, Y10, Y11), and fifteen more carry a sweep's count or place beside mine.
I re-read the line each new row's judgement rests on, except Y10's.

Two patterns run through the rows, and each has one answer:

- **A layer holds another layer's thing so that readers reach it without a new requirement** (F1, F2, G1, L1, N2,
  M1). The answer is always the same: the thing lives with its owner, and a reader is handed the result (a verdict, a
  root, a digest), not the thing.
- **One fact is stated in several layers, each sized to one user** (C2, G4, G5, S4, Y5). The answer: one owner states
  it, and every other layer takes it as a parameter.

## Part 3. The five that matter most

Each is a group of rows with one cause. For each: what is wrong, in words a reader outside the field can follow; the
smallest change that removes it; and what that buys.

### 3.1 SPOT's record of a window is the hub of the system, and it carries its callers' facts (P1, F1, N1, P5, P8, P9, X1, A1)

SPOT is the part that checks a random sample of a computation and bounds how much of the rest can be wrong. By
Daniel's definition its public input is a circuit and a sampling rule, and nothing else. On `main` its input is a
record with 24 fields. Ten are that input. Fourteen say things only a caller can mean: which day's circuit was
declared, who the prover is, a salt for PoUW's noise, a pool of earlier windows with its weekly reset. Because the
record holds everyone's facts, everyone reads it: PoUW's check, the certifier's period check (whose commitment is a
list of these records), the policy's check, compose's remote check, the adapter in 24 files, and Flock's verifier,
which holds the record's format so that the others can reach it. The knowledge runs the other way too: SPOT's own
parties file lists every protocol compose runs, by id. So a change to one caller's fact changes SPOT's verifier, and
SPOT's theorem is about a record and not about a circuit and a rule.

- **The smallest change:** split the record in two. SPOT's part holds the circuit's digest, its pieces, the sampling
  rule and the committed roots with their domains. The caller's part is a second record in the caller's package: it
  names SPOT's part by its digest and adds the caller's facts with the caller's checks. No check is dropped. About nine
  check codes change package (R2's declared day, R6's kind, stream, prover, salt and pool, R7's look back).
- **What it buys:** SPOT's code and its theorem match Protocol 1. A new protocol that calls SPOT changes nothing in
  SPOT. The certifier, PoUW and compose compare roots and digests, and read no field of a record. The format string
  leaves Flock.

### 3.2 No accounting protocol's answer says which registered thing it is about (G2, U6, N7, X3, G4, G5, G6)

Registration is where the developer says, once, what hardware it has, which circuit it runs and which weights. Each
accounting protocol then measures one part of that hardware. On `main` no protocol reads what registration accepted.
PoUW keeps its own table of a card's figures. PoUS and PoUB take the memory's size and read rate from their own
parameters, and PoUB recomputes the store's root from the model's checkpoint. The certifier's period names one link
and one key in its own parameters. The weights are committed three times with nothing tying the three roots. And no
answer names the registration it was measured against. So compose cannot check that the compute, the memory and the
network it adds up belong to the same registered devices, or that every registered device was measured at all. The
three open ties in the security model (the circuit, the link, the memory leaf) are this one gap, seen three times.

- **The smallest change:** every accounting protocol's answer gains two fields: the registration's head (64 bytes) and
  the part it accounts for. The protocol is handed that part's figures from the registration, not from its own files.
  Compose compares heads. A registered part with no answer is charged at its registered capacity, never at 0
  (soundness's default [relayed]).
- **What it buys:** the account means "of this registered hardware". A device's figures have one source. The three
  ties become equalities of digests inside compose's theorem and need no premise.

### 3.3 The auditor's devices do each other's jobs (P2, P9, W1, U2, N5, N4, X5)

The auditor has three kinds of device. The challenger draws challenges. The network certifier sits on a link, and
counts and releases traffic. The memory challenger times answers from memory. Each is trusted for one thing. On
`main` SPOT's challenger also keeps the auditor's ledger: it signs a period's open and close, a day's declared
circuit, and which window counts toward which period; 45 code files outside the proofs import it, and PoUW's Lean
check opens the ledger's entries itself. The network certifier's device also draws the salt that PoUW's
noise is seeded from, and keeps the root of PoUW's weights. The memory challenger receives 24,576 blocks of the store
a session, in the clear, and keeps them. So what one must trust a device for is wider than its name, and one
protocol's guarantee rests on another protocol's device.

- **The smallest change:** one device, one job, and three kinds of signed message in the shared channel: a challenge,
  a count, a declaration. The period's ledger moves to compose and a day's declarations to registration. PoUW is handed
  "a live salt from one of the auditor's devices", and compose says which device. The memory challenger's blocks need
  the zero-knowledge session that is already on the page, so that part stays open.
- **What it buys:** each device's trust is one named assumption with a stated scope. The security model can take the
  devices as parties with one kind of message each. PoUW runs where no certifier stands, with no code that names one.

### 3.4 The common ground holds single layers' things (G1, F2, Y1, Y4, Y5, Y9, L1, N2, M1, M4, S7)

Two things are meant to be common ground: the interface package, which says what a protocol's check is, and `Core`,
the vocabulary every guarantee reads. On `main` the interface package also holds the registered hardware, the
registration, compose's account and the period: 2,245 of its 3,717 lines. Flock's Lean package also holds recursion
and the ZK server, and 14 packages require it, several only to reach a hash or a byte format. `Core` reads an
accounting module and holds 84 files, among them the site, the compiler, the wipe's blocks and one matrix algorithm.
Because the common ground holds particular things, a second definition appears whenever a layer needs its own view:
four data sheets, two resources, compose's lists four times, the period's type three times, a Merkle tree of PoUS's
own beside the shared one. One table in commitments holds every protocol's format strings, the adapter's among them.
And names leak: Flock's byte lemmas are declared under the certifier's name.

- **The smallest change:** three moves, and no new statement. Registration, the account and the period leave the
  interface package. Recursion and the ZK server leave Flock's package, and the helpers go to the base packages (33 files already import
  a base package through a one-line module of Flock's, so renaming those imports is the first step and changes no record).
  `Core`'s files that one layer owns go to that layer's area. Then one rule on each: the interface requires only the
  base packages, and `Core` imports no area.
- **What it buys:** a change to registration rebuilds no proof system's package. Each duplicate has one home to merge
  into. A lock's records follow the layer that owns them.

### 3.5 The adapter holds protocol code and reads verifiers and devices, and protocols name the serving system (A1, A2, A3, N3, U1, F3, C1, C2)

An adapter is meant to make one serving system speak Verity, by one call for each protocol, and to know nothing of
how a protocol samples or decides. On `main` the vLLM adapter imports PoUW's verifier in 37 files and SPOT's
device in 13, holds 16,908 lines of PoUW's serving code, and has 21,275 lines that decide accept and refuse. In
the other direction, the certifier's advice format is defined as one pinned vLLM build's scheduler output, PoUB's
verifier knows vLLM's layer order, Flock's verifier holds a table of model names, and the cut of a circuit into pieces
is stated in the circuits, in the adapter and in the checker.

- **The smallest change:** for each protocol, the one call its prover offers (the page states PoUW's and memory's).
  The compiler and the advice format move to the adapter, and the certifier is handed a registered compiler circuit and
  advice as bits. The cut is one function with the cap as a parameter.
- **What it buys:** a second serving system is a new adapter and changes nothing in `verity/` [page]. A new vLLM build
  changes no verifier. An auditor trusts no line of the adapter.

## Part 4. The tests that would hold each rule of part 1

Proposals, not code. Each test starts from today's findings as a list that only shrinks, each line with an owner and
a day, in the form `tests/tree.toml` already uses for its 36 `not_yet` lines. A new finding fails. A listed line warns
until its day.

| id | The rule it holds | The test, and where | What it lists today |
| --- | --- | --- | --- |
| T1 | The layers' order | `tests/import_order.toml` takes the page's order: circuits, commitments, the shared files (a row of their own), flock, spot, recursion, registration, pouw, pous, poub, certifier, inference_only, compose, runtime, adapters, demos, benchmarks. | F4 (Flock's prover above SPOT), S1, S6 (the shared files and the runtime are one row today), N2 (the policy's row is below the certifier's) |
| T2 | The proofs use neither layer | Each of the proofs' places has `loads` of circuits, commitments and the shared files only. | P3 (three loads), P8 (the configuration's lists) |
| T3 | An accounting protocol uses no other, but PoUB on PoUS | `loads` of pouw, pous and certifier names no other accounting protocol and not the policy. The same for tests: a test under one protocol imports no other protocol. | Four test files: the certifier's on PoUW (`certifier/verifier/network/tests/period_case.py:82`), PoUW's on the certifier (`pouw/verifier/tests/test_pouw_salt.py:27-31`), SPOT's on PoUW, PoUB's on PoUW [read by me]. No other file today. |
| T4 | A layer names nothing above it | One table, `MAY_NOT_NAME`, beside `SPOT_NOT_CERTIFIER` (`tests/test_boundaries.py:101-111`): for each layer, the whole words its files may not hold, in Python and Lean, comments and docstrings included. Circuits and commitments: every protocol's name, "window", "request". The proofs: pouw, pous, poub, certifier, nci, pool, lane, week, served, vllm. Each accounting protocol: the others' names. All of `verity/`: vllm, torch and model names (today's test of rule 1 reads imports only). Each layer and word starts at today's count. | P4, F5, C1, C5, N3, U1, W1 |
| T5 | A package is used through its entry | Each protocol declares its entry modules, in its `pyproject.toml` and its `lean-audit.json`. A file outside the package that imports any other module of it fails. | P5 (9 Lean modules, 153 Python lines for SPOT), P9 (45 code files on the challenger), Y10 |
| T6 | A format string has one owner | Every literal of the form `verity/.../vN` or `verity-.../vN` is defined once, in the directory of the layer its first part names. `verity/commitments/` defines none of a protocol's. All are distinct across the tree. | M1 (341 constants in one table) |
| T7 | Lean packages follow the layers | A table `LEAN_REQUIRES` beside `REQUIRES_FLOCK_VERIFIER` and `BASE_LEAN`. The interface requires the base packages only. Flock's verifier requires the base packages only. SPOT requires Flock and the interface. An accounting check requires the interface, SPOT, registration and its own packages. None requires another protocol's. Compose requires all. `REQUIRES_FLOCK_VERIFIER` shrinks to the packages that verify a Flock proof. | N2, I1, P6, F2 (14 packages), S7 (a device requires a package under the hardware directory) |
| T8 | The areas of `security/` | In `security/lean-audit.json`'s `layers`: `Definitions.Core` imports no area; `Definitions.Spot` imports `Core` only (it holds today); a protocol's area imports no other protocol's; the composition area imports a protocol's `Guarantees` modules only. `trees.except` only shrinks, as today. | Y1, Y2 (28), Y3, Y9, Y11, U5, X2 |
| T9 | One definition of a shared concept | `tests/test_lean_vocabulary.py` gains the names `Sheet`, `Tier`, `Resource` and `Part`, `Audit` and `Subprotocol`, `Window`, `Period`: each defined once outside `Proofs`. | G5, Y5, S4 |
| T10 | An accounting answer names the registration | For each accounting protocol on one of compose's lists, its declared account has a key for the head and a part. One case file: an answer that names another head is refused by compose. | G2, U6, N7 (no protocol passes today) |
| T11 | A device does one job | A file under a `device/` imports no other protocol's `device/` or `verifier/`. A `verifier/` holds no socket or process code. The kinds of message a device signs are declared in `channel.py`, and a device signs only its own. | P2, P9, W1, N4, N5, X5 |
| T12 | The adapter makes one call | Under `adapters/`, an import of a protocol is allowed only for that protocol's declared workload module, and never of a `verifier/`, a `device/`, `verity.hardware`, `tools` or `benchmarks`. | A1 (195 import lines), A4 |
| T13 | No code reads `security/` or the data sheets, and no Python writes under `security/` | A lint on path strings (`security/`, `hardware/cards`) in code under `verity/`, `adapters/` and `benchmarks/`. | U4, G5, Y6, Y8 |
| T14 | The runtime | No file under `verity/protocols/` or `adapters/` imports the runtime, and no `verifier/` loads it. A row of T1 once the types move. | S1 (9 files) |
| T15 | The configuration names no protocol | `config.py` holds no protocol's id as a literal, and no file outside compose holds a table from a list's names to protocol ids. Whether a list is acceptable is asked of compose's Lean. | S2, P8, G3 |
| T16 | The cut is stated once | No table of cuts outside the one function's module, and the cap appears only as a parameter. | C2 (three places) |

## Part 5. What I am unsure of, and the questions

### 5.1 What I am unsure of

1. **Who calls SPOT on a remote run.** Compose's remote list holds SPOT itself, as `verity/sampled-proofs/v1` [read
   by me]. If SPOT reads no registration, something must hand it the registered circuit and check the caller's facts
   (the day, the kind of window, the pool). Either compose does, or a thin protocol on the list does: "the compute
   audit" of the page's section 10, whose check is "SPOT accepted this registered circuit on these roots". I lean to
   the second, so that compose never calls the proofs. Neither page settles it.
2. **Whether recursion may require SPOT** (P6). The page draws both on Flock and neither on the other. If the outer
   proof's statement is "SPOT's verifier accepted", recursion needs SPOT's verifier as a circuit. That is data, not an
   import. I did not read recursion's Lean beyond two files.
3. **The split of SPOT's 24 fields** (P1). I read the field comments, not the proofs that use the fields. `registered`,
   `params`, `schemas`, `tied` and `members` may be SPOT's own. proofs has not read the anchors (R5) and the values the
   verifier fills itself (R7-supplied) [relayed].
4. **Which rows marked "to Daniel" change a statement, and which only move one.** I marked every row where a published
   answer, a commitment's digest or a premise changes. soundness and lean can strike some.
5. **Whose the period is** (S4): compose's, beside the account, or a shared file's. `tests/tree.toml` sends
   `reporting.py` "into verity/protocol.py" [read by me].
6. **Names against meaning.** A word count finds names, and a name in a docstring costs less than a field or an
   import. I ranked by fields and imports first, and T4 would still hold the names.
7. **Two counts that differ from the leads'.** PoUW's code in the adapter: 16,908 lines in 54 Python files under
   `protocols/pouw/` by my count, 18,800 on the page, 20,500 by vllm. The adapter's `check/`: 78 files by my count, 86
   by vllm's. I counted `.py` files that git tracks. After the first chain of moves it is 83 [counted by me]. The
   interface package: 3,717 lines by my count, 3,735 in 19 files by SWEEP-ABOVE's.

### 5.2 Two questions the leads put to this document

**circuits: is a tie to a registered root a circuit the builder writes, or a rule SPOT is handed with the roots?**
Proposed: neither the builder nor SPOT owns it, and there are three cases.

- **A read at a public position** (a weight's row, every time the same): it is no circuit. The piece's read names a
  position under a root, the proof system opens it inside the proof, and SPOT is handed the root with the commitments.
  The caller of SPOT checks that the root is the registered one, by comparing 64 bytes. Flock's "registered reads" do
  the opening today (`flock/verifier/Flock/Registered.lean`) [read by me: the header]. To the proof system a root is a
  root, and "registered" is the caller's word.
- **A read at a private position** (the embedding's row for a token): it must be a circuit, because the position is a
  value. The opening's gates are commitments' (`verity/commitments/gates/embedding.py` exists [read by me: the import
  lines in the adapter, not the gate]). The builder places it as a named circuit and writes no hash gate itself.
- **What a registered root stands for** (the weights' rows are one checkpoint's bytes:
  `adapters/vllm/verity_vllm/program/predict/weights_tie.py:1-12` [read by me]): it is registration's proof, made
  once. The adapter knows the checkpoint's file layout and hands it as data. It should not build the proof's window.
- **What the choice costs.** SPOT's input gains, for each committed value, a root and a domain, and nothing in the
  proof. The builder cannot read at a private position without a gate of commitments', and pays its gates. The model's
  circuit holds no commitment scheme except at those reads, so a change of scheme changes the circuit's digest only
  there. Daniel's rule holds in all three: a value is touched only inside a proof.

**vllm: which of the adapter's `check/` is a verifier's work, and which is the developer's own test?** Proposed: one
question sorts each file. Does a verifier, compose or a published answer read its accept or refuse?

- **If yes**, it is a protocol's check. It goes to that protocol's `verifier/`, and it must be Lean, since Python never
  returns accept or reject [page].
- **If no**, it is the developer's own test that the circuit computes what the served system computes. That is
  completeness: it decides only whether honest serving passes, and nobody proves it (`AGENTS.md`). It stays in the
  adapter, under a name that does not read as a verdict.

I did not read the 78 files, so I do not know how many fall on each side.

### 5.3 What I did not read

- Flock's Rust prover, and the two-stage form's Python beyond the lines cited.
- Recursion's packages, beyond `Sampled.lean` and the Lake files. The ZK server's code.
- The certifier's device program (`NetworkCertifier`), and PoUS's and PoUB's Lean beyond the lines cited.
- `security/Proofs`, beyond counts. `security/Properties/Verity/Game.lean` beyond the lines cited.
- `demos/` and `benchmarks/`, beyond their import lists. The reverse direction: what they import of the adapter's
  insides.
- The draft of registration's statements (#9701), which is not on `main`.
- Of the two sweeps: the appendix of SWEEP-PROOFS (its list of 143 format ids). I read the rest of both and took what
  part 2 marks. Findings of theirs that I left out, for want of a rule on the page that they break: PoUS's check
  requires the memory challenger's Lean package (a verifier requires a device's); the devices' keys sit under several
  signer types; the interface's registration holds a third kind of entry, a day's declared circuit, that the page's
  registration row does not name.
- What changed on `main` after `9182900fb94e` beyond the paths: 1,584 files were modified, and I re-read only the
  lines this document cites.

### 5.4 Questions, by lead

**proofs**

1. Of the 24 fields, which does SPOT's soundness proof read? Which theorem cites R2, R6 or R7, so that moving them
   changes what it rests on?
2. Does the outer proof's statement need SPOT's verifier? Can `Sampled.lean` move to SPOT's package (P6)?
3. After the challenger is split (P2), which device signs a day's declaration?
4. What is SPOT's entry, by name, in Lean and in Python (P5)?
5. Who holds the Program to the circuit (F7)?
6. Who calls SPOT on a remote run (5.1, item 1)?

**soundness**

1. Is the form of an accounting answer agreed: the head, the part, the amount, ε at δ, and the roots read?
2. `DeclaredIsParams`, `CertifiedIsParams`, the memory leaf and the weights' root are one shape: an answer names a
   registered thing. Is it one lemma over the form above? proofs says the first compares digests of two different
   objects, the circuit file's and the program descriptor's [relayed]: which object does each layer name?
3. What may the composition area import of a protocol: its `Guarantees` only (T8)?
4. Which "to Daniel" rows only move a statement?

**lean**

1. Can a package's entry be held by Lake, or is it the lock's rule (T5)?
2. What does splitting `flock_verifier` cost in records (F2)? Does `--moved` carry a record across packages?
3. Which definitions of `Accounting.Device` does `Core` read (Y1)?
4. The `Flock` namespace in the base packages (F8): leave it, as I propose?

**compute-accounting**

1. What does PoUW need of its salt's source: only that one of the auditor's devices drew it after the commitment (W1)?
2. What does the check read of `models.json` (G6)? Can the registration hand it?
3. Where does PoUW's served code import the two experimental schemes (W3)? I found no such import line.

**memory-accounting**

1. Which memory leaf does each answer name, once it names one (U6)?
2. Can PoUB be handed the store's root from the registration, with the model's tie checked once there (U1)?
3. What is the least the memory challenger must see (U2)?

**network-accounting**

1. Which fields of SPOT's record does the period check read today (N1)? Are roots enough?
2. Is there a reason against `NciCheck.Desc` in `core_circuits` (N2)?
3. With the compiler and the advice format in the adapter, what stays in the certifier: the circuit's check and the
   charge (N3)?
4. Which files of `certifier/verifier/` are the device's (N4)?

**vllm**

1. Of the 78 files of `check/`, which does a verifier or a published answer read (5.2)?
2. Is "the adapter is handed roots" enough for the weights, or does the adapter need more back from registration (G4)?

**circuits**

1. Is the three-case answer of 5.2 right for the embedding's reads? Which reads are at a private position today?
2. Can the cut be one function of a circuit and a cap (C2)? What besides the cap does it need from the proof system?

**compute-research**

1. Of the four Lean definitions of a data sheet (G5), which one stays?

**training**

1. Which figures does the slowdown calculator need handed to it as data, once it reads no protocol's module (I2)?

**Daniel** (through top, after the leads' corrections): the 16 rows marked "to Daniel", and 5.1's item 1.
