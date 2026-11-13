# What each layer of Verity may know, and every coupling on `main` that breaks it

**The layers are held apart by import rules only, and three things leak through them: SPOT's record of a window, which
every protocol and the adapter read and which carries its callers' facts; the registration, which no accounting
protocol reads, so each restates what it needs; and the adapter, which holds protocol code and reads verifiers and
devices. Part 2 lists 91 couplings, one of them withdrawn. Part 3 names the five that matter most. Part 5.4 has what
is left for Daniel.**

Cut 2, Fri 13 Nov 2026, for 5:30 AM PST (13:30 UTC). `main` is read at `053b8e50fe8a`. The first cut
(`0f875a6e82bb` on this branch) read `main` at `9182900fb94e` and then `e4da505fe488`. Every path and line that cut
cites still stands at `053b8e50fe8a` [checked by me with a script: 182 citations, none moved or shifted]. The second
chain of path moves has not landed, and it moves most protocols' directories, so the final re-reads every path. A count
that says "on `053b8e50fe8a`" was recounted for this cut; the others are the first cut's. Written by LAYERS, a worker
of top. Nothing was built or run: this is `rg`, `git` and the files.

## What the leads' answers changed

**All five headline findings stand, each changed as below. 18 rows were wrong in part, 5 are new, and Daniel's list is
15 rows, with two yes-or-no points.** [read]: by me, on `main` at `053b8e50fe8a`. [relayed]: a lead's word, time in PST.
- **3.1 stands, restated.** SPOT's record has 13 keys; the 24 "fields" are its check's parameters, at least 7 of them
  a caller's fact [read; proofs, 2:39 AM]. Compose's two lists already call SPOT through 1 thin protocol [read].
- **3.2 stands, wider.** 4 checks take a device's key from the prover's run files [relayed, soundness; 2 read]. The
  registration has no read rate for a server's memory [read] or CPU [relayed]. 0 callers pass weights [read; vllm].
- **3.3 stands.** The challenger signs a day's declaration today [read]. Nothing reads the 24,576 blocks a session
  that the memory challenger keeps [relayed, memory-accounting, 1:57 AM].
- **3.4 stands, priced.** Splitting Flock's package moves 11 records by hand [read], and 21 of 36 packages are audited
  again [relayed, lean, 2:01 AM]. "Records follow the layer" holds for 15 of 1,692 guarantees [read; found by lean].
- **3.5 changed.** The adapter imports 5 protocols' verifiers and devices in 198 lines [read]. But 0 of the 83 files of
  its `check/` are read by a verifier, compose or Lean; they are the developer's own checks [relayed, vllm, 1:59 AM].
- **Wrong, 18 rows.** As fact: P1, P5 (SPOT states 4 entries, not 0), F7, Y7 (withdrawn), A3, G1, L1. In the fix: P6
  and F1 (import cycles), C3, Y2, U3. Too wide or too narrow: Y4, X2, U1, N4, W3, F2. Each row says who found it and
  what I re-read. 9 counts took in tests [recounted, read; found by vllm, 1:49 AM].
- **New, 5 rows:** G7 device keys, G8 missing capacities, U7 PoUS's check requires its device's package, N8 the
  certifier's cell reads a request's `max_tokens`, B3 the queue's low-risk list names 8 dead paths [each read].
- **Daniel's list, 15 rows (16 before).** soundness struck G3, U5 and Y11 and ruled Y2, L1 and Y4 not his [relayed,
  1:07 and 2:02 AM]. F7 and X2 are his on a condition only. Added: G5's second step, G8, N3's charge (part 5.4).
- **Point 1, yes or no: SPOT is called only through that thin protocol, which takes the caller's facts out of SPOT.**
  Recommended: yes [relayed, proofs]. Cost: 0 lock records to move its package [relayed, lean, 2:42 AM]; 2 listed
  statements change for the declared day, 1 for the stream, pool and salt [relayed, proofs; the names searched, read].
- **Point 2, yes or no: the challenger stays the 1 device that signs a day's declaration.** Recommended: yes
  [relayed, proofs]. Cost: the same 2 statements. A "no" adds 1 device and a second clock to tie to the first.

## How I know each claim, and the words

- **[read by me]**: I opened the file at that line on that commit.
- **[counted by me]**: a search I ran. Unless it says otherwise it counts files that are not tests, by whole word,
  ignoring case.
- **[relayed]**: a lead's word on TOP-415, copied in `inbox/layers-leads.md` on `cursor/verity-top-4db4` (read whole
  at `b49ee9943259`; a time beside a lead's name is that answer's, in PST on Fri 13 Nov), or a sweep's
  (`layers/sweep-proofs.md` on `cursor/top-layers-sweep-proofs-5d1a` at `c6f75df4d548`, `layers/sweep-above.md` on
  `cursor/top-layers-sweep-above-8c2f` at `96c3b77635a8`). Not re-read by me. "(soundness 2)" is that lead's own row
  number, and "SWEEP-PROOFS 4a" that sweep's section.
- **[page]**: the architecture page's words (sections 1 to 7 and 10). **[SPOT page]**: Daniel's page "SPOT security
  definitions". **proposed**: mine, where the pages do not say.
- **[not checked]**: nobody read it for this document.

Words. I use the architecture page's. A **piece** is one part of a circuit's cut (the SPOT page calls it a block). The
**sampling rule** is the SPOT page's sampling design D. A **challenge** is a random value the auditor's side draws
after a commitment. **SPOT's record of a window** (after its first use, SPOT's record) is the record SPOT's code calls
a "registration record". The page says it takes another name. top took proofs' name for it, the **sampling frame**
(never "frame" alone), at 1:28 AM PST [relayed]; the code is renamed after the second chain, so this cut keeps the
longer name. A **lock** is a package's `lean-audit.json`, a **record** is a guarantee's recorded statement in it, and
a package is **audited again** when the hash of its inputs changes (lean's words; the tree has 36 such packages
[counted by me: 36 locks outside test fixtures]).
Only a renamed declaration changes a record. A moved or renamed module re-keys its group in the lock, and a changed
`require` changes which packages are audited again; neither changes a record [relayed, lean, 2:42 AM PST].
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
- Called by: a protocol that proves a statement about committed values. The caller passes in the circuit, its pieces
  and the sampling rule [page, rule 7]. Today, in Lean, only compose's two lists call SPOT's check, through
  `VStar.Recursive.Sampled.protocol`; PoUW's check imports SPOT's inner modules and never its check, and the
  certifier's check imports none [read by me on `053b8e50fe8a`]. The first cut said "PoUW and the certifier today",
  which the Lean does not do, as proofs found. SPOT's callers in Python: not read.

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
  data, each registered part (a device, a memory leaf, a link) with every capacity its data sheet states, each
  auditor's device's key, the compiler's digest and the weights' roots. Today it holds a device's key, which no
  period check reads (G7), and no read rate for memory under a server and no figure for the server's CPU (G8). Its log has a third
  kind of entry that the page's row does not name, a day's declared circuit (P2; point 2 of 5.4).
- Must not know: how any other protocol works inside [page]. Proposed: nor compose's lists by name. It asks only "does
  the run's list hold a protocol that accounts for this part".
- Called by: compose, once. It uses the shared files and commitments and no other protocol. The other accounting
  protocols, the policy and compose read what it accepted [page, rule 7].

**The other accounting protocols.** Proposed, for all four, from soundness's and memory-accounting's words
[relayed]: each is handed the registration's head and the part it accounts for, and **its answer names both**. That
is what compose needs to join answers, and no answer does it today (rows G2, U6, N7). soundness agreed the form of an
answer, with two fields more [relayed, 1:07 AM PST]: the head; the part (a device, a memory leaf or a link); the
amount; ε at δ; the roots read; the period's span, so compose can sum over one period; and the key of the auditor's
device it rests on, so compose can hold it to the registration's.

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
- Hands back: bits held and bits read for the period, each with its error, to compose's account [page]. PoUS alone
  hands back bits held at a moment; the period's statement is the joined one [relayed, memory-accounting].
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
- Called by: 13 Lean packages [relayed, lean]. It requires two base packages, `core_base` and `core_commitments`, and
  the difftest's Lean [read by me on `053b8e50fe8a`: its Lake file; the first cut said "the three base packages"].

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
| P1 | SPOT's record of a window, and its check | Its callers' facts. The record is a JSON object of 13 keys, 14 under v2 (`spot/verifier/lean/Spot/Registration.lean:111-113`). Callers' facts sit in its `window` key: the kind (served or stand-in), the stream and the prover, the seed salt, the pool's week and its spent mark (`:557`, `:600-603`, `:531-532`). The check holds the record to 24 parameters it is handed, "the verifier's own copies that R1–R8 hold the record to" (`:162-213`). At least seven of them carry a caller's fact by their own comment: `log` (this prover's earlier records), `binding` (the stream and the prover), `windowRoot` (the challenger's signed open), `declared` (the day and the log's declarations), `seedSalt`, `pool` and `tied`. The checks: `R2-declared` 573-574, `R6-stream` and `R6-prover` 599-601, `R6-seed-salt` 603, `R7-pool-chain` 611; `Pool` at 131-139. Python twin: `spot/verifier/one_stage/registration.py`, 874 lines. [read by me. The first cut called the 24 the record's fields and split them 10 and 14. proofs corrected the first, and the split is withdrawn.] proofs: eight parameters exist only for callers' rows; `registered`, `params`, `schemas` and `members` are SPOT's own, and `tied` is a caller's whose tie is in no theorem; proofs outside SPOT's own property read five of the 24 by name (`program`, `query`, `law`, `workTable`, `windowRoot`) [relayed, proofs, 2:39 AM PST]. | shape | SPOT is handed the circuit with its pieces, the sampling rule and the commitment: a list of roots, and for a part of the trace a caller already fixed, that part's root. The caller checks its own facts (the day, the kind of window, the pool) and hands SPOT roots only. SPOT reads no log of earlier windows. One thing stays SPOT's: the record is fixed before the challenger opens the session, which is what makes the challenges live (proofs [relayed]). | code and Lean: split the record in two, SPOT's public input and the caller's record over it. R2's declared day, R6's stream, prover, salt and pool marks, and R7's pool chain go to the protocol that calls SPOT (5.4, point 1). The cost in lock records: moving the declared day changes two listed statements, `Properties.Spot.AcceptMeans` (`security/Properties/Spot.lean:435`) and `AcceptInferenceOnly`, through `Properties.Remote.StageHeld` (`security/Properties/Remote.lean:119`); moving R6's or R7's facts changes `AcceptMeans` only, since no theorem outside SPOT's two property files cites them [relayed, proofs]. A search of the names agrees: outside SPOT's property files `Properties.Spot.R2` is cited in `StageHeld` (`security/Properties/Remote.lean:122,125`) and its proof file only, and `R6` and `R7` in no Lean file; both statements are listed in `security/Properties/lean-audit.json` [counted by me on `053b8e50fe8a`]. **To Daniel** (soundness keeps it on his list [relayed]). | proofs, soundness |
| P2 | SPOT's challenger, `spot/device/challenger.py` (2,171 lines) | The auditor's ledger and two other parts' records: a period's open and close, a day's declarations, which window counts toward a period, the pins of `main`'s first-parent line, "a lane's day", PoUW's live window as "the first client", and recursion's run record. Lines 95-112, 137-166, 199-201, 304 (`PERIOD_KINDS`), 1296 (`verify_periods`). It imports the claims' ids (219) and the hardware's record entry (220). [read by me] The ledger's entries are a reserve, an open, an end, a period's open and close, a declaration and the pins log; compose's device writes them and signs them under the challenger's key (`verity/composite/device/live_challenger/auditor.py`) [relayed, SWEEP-PROOFS 5a]. A day's circuit is declared in two places today. The challenger's ledger has a `declaration` entry, stamped on the challenger's clock and signed under its key, which orders a day's declaration before every challenge of the day (`spot/device/challenger.py:110,137-143`; `auditor.py:132-141,285`). The registration log has its own `declaration` entry, with no signature and no clock, which refuses a second declaration of a day (`interface/CoreProtocol/Registration.lean:24-27,105`). R2 takes the day and the declarations as parameters (`Spot/Registration.lean:198-199`). [read by me on `053b8e50fe8a`] No Lean check hands the ledger's reader to R2; the join is in a benchmark's audit only [relayed, proofs, 2:39 AM PST]. | job | The challenger is handed a commitment's digest, draws challenges after it and signs both. The period's ledger is compose's. A day's declaration has one form and one check, registration's. Which device signs it is point 2 of 5.4: proofs recommends the challenger, handed the declaration as a digest [relayed]. A run record is recursion's. | code: three files from one. **To Daniel**: which device signs a declaration is a premise (soundness [relayed]). It is worded as a yes or no in 5.4. | proofs, soundness |
| P3 | SPOT's Python | Three things below or beside it that it should not load: the claims' ids, the runtime, the hardware's record entry. `tests/import_order.toml`, SPOT's `loads` (`verity/claims`, `verity/runtime`, `verity/hardware/record.py`); `spot/verifier/parties.py:126`; `spot/device/challenger.py:219-220`. [read by me] | import | SPOT loads circuits, commitments and the shared files. | code: the record entry goes into `channel.py` (proofs [relayed]); the party types come from `protocol.py`; `claims` goes (S5). | proofs |
| P4 | SPOT's code and docs | Names from above it. Lines in SPOT's files that are not tests: pool 180 (4 files), lane 92 (6), PoUW 43 (13), week 51 (4), certifier 20 (6), vLLM 10 (4) [counted by me on `053b8e50fe8a`; the first cut's figures counted tests, as vllm found]. `spot/verifier/second_stage.py:838` (`INTERIOR_ROOT = "verity-vllm/interior-run/v1"`); `spot/verifier/one_stage/proofs.py:66` (imports the served circuits' tables); the verdict's field `rests_on_period_check`, the certifier's check by name (`Spot/Pieces.lean:247`). [read by me] | name | SPOT's words are circuit, piece, sampling rule, root, challenge, proof and bound. A caller's tag reaches it as bytes. Its verdict lists the roots it did not check itself, and the caller says what they rest on. | rename and code, mostly carried out by P1 and P2. | proofs |
| P5 | Every caller of SPOT | SPOT's insides. SPOT states four entries in Lean and no single one: `Spot.OneStageZk.protocol` (`spot/verifier/lean/Spot/OneStageZk.lean:1310`), `Spot.OneStage.protocol` (`Spot/OneStage.lean:576`), `Spot.Run.run` (`Spot/Run.lean:265`) and `VStar.Recursive.Sampled.protocol` (`recursion/verifier/recursive/VStarRecursive/Sampled.lean:81`), the one compose lists, which sits in recursion's package. Only compose's two verifiers name an entry from outside. Everything else imports past them: 11 `Spot.*` modules in 10 files under `verity/` outside SPOT's package, and 25 modules in 44 files under `security/`. PoUW's check imports seven inner modules and never the check; the certifier's check imports none. [read and counted by me on `053b8e50fe8a`; the first cut said "SPOT states no entry", which proofs corrected] Python: `spot.verifier.one_stage` is imported in 241 lines of 143 files outside SPOT that are not tests, 33 lines of 24 files in the adapter [counted by me; vllm counts 238 in 142; the first cut's 153 was wrong]. SPOT's README lists a core API, and the three files with the most outside importers are not on it: the challenger (46 code files [counted by me]), `device/remote.py` (20) and the produced rows (19) [relayed, SWEEP-PROOFS 4a, 3b]. | import | One entry in Lean and one in Python: circuit, pieces, sampling rule, commitment, challenges, proofs in; accept or reject, the bound and the roots out. Callers import that and nothing else. | code: one of the four is named the entry, and a test holds the rest private (T5). Lake cannot hold it. A table in `tests/test_boundaries.py` can, and editing it changes no package's audit key (lean; proofs agrees [relayed]). | proofs |
| P6 | Recursion's package | It defines SPOT's protocol as compose lists it: `verity/sampled-proofs/v1` is `VStar.Recursive.Sampled.protocol`, so recursion's package requires SPOT's. `recursion/verifier/recursive/VStarRecursive/Sampled.lean:33,81`; its Lake file; `remote/verifier/CoreRemote.lean:37,116`. [read by me] The outer proofs' statements are Flock's, not SPOT's: SPOT's check runs in Lean around them, as an import [relayed, proofs, 2:39 AM PST]. | import | The protocol compose lists is a caller of SPOT and of recursion, and sits above both. Recursion is handed Flock's proofs and one statement, and does not know which protocol asked. | move the package `core_vstar_recursive` whole (four Lean files, 807 lines; it requires the interface, Flock's verifier and SPOT [read by me]) above SPOT and recursion, as the thin protocol that calls SPOT (5.4, point 1). `Sampled.lean` cannot move into SPOT's package alone: it names five things of recursion's package, which requires SPOT's, so that is a cycle [relayed, proofs]. With the module names kept no lock record changes: the package's own lock holds no guarantee, and the two locks that read its definitions key them by module name. Path lines change in the Lake files and manifests of four packages, its own and the three that require it [relayed, lean, checked at `053b8e50fe8a`]. | proofs, lean |
| P7 | SPOT's Lean package | It holds the registered hardware's difftest. `spot/verifier/lean/Spot/SiteDifftest.lean:3`. [read by me] | test | The test goes with registration. | move (on the page, with registration). | proofs |
| P8 | SPOT's parties | Compose's two lists, with every protocol's id: `spot/verifier/parties.py:146-152` (`SUBPROTOCOLS`: onsite's seven and remote's three, each with its id), `:123` (imports the configuration's `COMPOSITES`), `:10-14` ("The core deployment ... wraps SPOT[Flock] for both composites ... Its verifier composes each block's audit with the composite's other subprotocols"). [read by me; found by SWEEP-PROOFS] | name, job | SPOT's parties name SPOT's own roles. The lists and the composed verifier are compose's. | move the table and the core deployment to compose. | proofs |
| P9 | PoUW, PoUB, the certifier, compose's device and the adapter | SPOT's challenger as a library: 46 code files outside the proofs that are not tests import `spot/device/challenger.py` [counted by me on `053b8e50fe8a`; the sweep says 45] (the adapter 13, PoUW 4, PoUB 4, compose's device 3, benchmarks the rest), for the live challenge's types and for the ledger's format. 19 import SPOT's produced-rows format, 8 of them the certifier's, which builds its network rows on it. [relayed, SWEEP-PROOFS 4a] PoUW's Lean check opens the ledger's entries itself (`pouw/verifier/lean/check/PouwCheck/Pass.lean:51`) [read by me]. | import, shape | The types of a live challenge (its commitment, its opening, its signed record) are the shared channel's. A row format several protocols build on is commitments'. No protocol reads the ledger: compose hands it "this window counts". | move the types to `channel.py` and the row format to commitments; code. | proofs |

### 2.2 The proofs: Flock and recursion

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | Flock's verifier | The format and digest of SPOT's record, defined there "so that every reader reaches it without a new requirement", and the registered reads ("a program's weights"). `flock/verifier/Flock/WindowRecord.lean:7-16,26-28`; `Flock/Registered.lean`. [read by me] | name, job | Flock is handed a circuit, commitments and challenges. A record of what is fixed before the challenges is SPOT's input. | move, after P1. Not up to SPOT's package: Flock's own recursive setup reads the digest (`flock/verifier/Flock/RecVStar.lean` imports `Flock.WindowRecord` [read by me]) and SPOT's package requires Flock's, so that is a cycle. It goes down to a base package, or waits until recursion leaves Flock's package (F2) [relayed, proofs]. | proofs |
| F2 | Flock's Lean package, `flock_verifier` | It is three parts in one: of 57,859 lines in 269 files, recursion is `Rec*` (22 files and 7,698 lines by lean's count; 19 files and 7,737 lines by mine, without tests) and `VStar` (21 files), and the ZK server is 8,409 lines in 21 files. [counted by me; the first cut's "`Rec*` 24 files" took in the session record's two files, as lean found] 14 Lake files require it (`tests/test_boundaries.py:116`). Five more packages import its modules through a package they require, and two of the 14 import none of it [relayed, lean]. Several require it for helpers only: the certifier's check imports `Flock.Hm96`, `FV3`, `Canon`, `Sizes`, `WindowRecord`, `Public`, `Program` [counted by me; `Flock.Bytes` is imported by the certifier's device, not by its check, as lean found and I re-read]. Of the Lean imports of the proofs' packages from outside, 48 in 37 files (lean's count; 42 in 33 by SWEEP-PROOFS) are of 16 one-line modules that only re-export a base package under Flock's name [relayed]. | import | Hashes, canonical bytes and commitments come from the three base packages. Flock's package is the proof system. Recursion and the ZK server are recursion's. A check that verifies no Flock proof requires none of it. | Two steps, priced by lean [relayed, 2:01 AM PST]. (1) Rename the 48 imports to the base packages: no record changes, and it frees one package, `certifier_device`; every other still imports a module that is Flock's own. (2) Move recursion and the ZK server out. `--moved` carries no record between packages, so the lock's 11 records (8 the ZK server's, 3 recursion's) and its 2 open premises [counted by me on `053b8e50fe8a` in the lock] go to new locks by hand, and 21 of the 36 packages are audited again. What stands in the way is Lean, not records: the part that stays imports the parts that leave (`Flock/HmRow.lean`, `Flock/Witness.lean`, the root). No statement changes. | proofs, lean |
| F3 | Flock's verifier | A table of served models that a benchmark writes. `flock/verifier/VStar/Menu/Pinned.lean:5-7` ("Generated by `benchmarks/numerical/...`"), `:15-103` (`"vllm/gemma2-2b__bf16__l40s__..."` and five more model families); imported at `VStar/Cli.lean:5`. [read by me] | name | The cost table is a file the caller hands in: measured proving costs "are handed to it as data" [page, section 10]. | code: read it as a parameter; delete the generated module. | proofs, benchmarks |
| F4 | Flock's prover (`flock/prover/`, the distribution `verity_flock`) | Recursion's prover and part of SPOT's: 17 `rec_*.py` modules, `pod/rec_inner.py`, `inner_registered.py`. It imports SPOT's verifier in 42 lines of 14 files and the served circuits in 103 lines of 36 files. The import listing therefore puts Flock's prover above SPOT. `tests/import_order.toml` (the order; SPOT's `loaded_by`). [read and counted by me] | import | Flock's prover: a circuit, commitments, challenges in; a proof out. Recursion's prover sits in `recursion/prover/`. Then the listing's order is flock, spot, recursion, as the page draws it. | move | proofs |
| F5 | Flock's verifier and its command line | Names of PoUW and the adapter. `flock/verifier/FlockVerify.lean:344` (`--work-table`, "the PoUW work"); `Flock/Firewall/Salts.lean:51` (`PouwCheck.Credit.registeredFile`); `Flock/Sigma.lean:4` (`verity_vllm`). Files under `flock/` that are not tests, naming: registration 41, window 22, vLLM 41, PoUW 13. [read and counted by me on `053b8e50fe8a`; vllm counts 48, 25, 41, 13; the first cut's 59, 38, 29, 19 were wrong] The Rust prover has a binary named for vLLM, whose block format cites the adapter's `PROTOCOL.md` (`flock/prover/live/src/bin/flock-vllm-v1.rs`, `vllm_block.rs:2`) [relayed, SWEEP-PROOFS 2]. | name | As F1. | rename, after F1 to F4. | proofs |
| F6 | Flock | Tools and benchmarks: the listing's ties flock to tools (3 files) and flock to benchmarks (7 files). `tests/import_order.toml`, `[[against]]`. [read by me] And 165 path strings in Flock's and SPOT's code name what is above them: the team's tooling 39, `benchmarks/numerical` 30, the adapter 22, PoUW 12; scripts put the adapter and a benchmark on the Python path (`flock/tool.py:60`, `flock/pod/92-zk-sessions.sh:18-19`) [relayed, SWEEP-PROOFS 1f]. | import | Nothing uses a benchmark (rule 9). The pod scripts are the team's tooling, outside `verity/`. | code and move | proofs |
| F7 | Flock's command, run alone | It can accept a statement whose Program nothing holds to the circuit, and says so: `flock/verifier/Flock/ProvedScope.lean:655` ("the Program was handed and not held to the circuit"). [read by me] The first cut said which layer holds the Program to the circuit is not settled in code. That overstated it. By proofs' table [relayed, 2:39 AM PST]: on every path SPOT's check accepts, the Program is held to the circuit by a digest in the circuit file's META (Flock's setup; SPOT's R2 and v2's refusal). The open case is Flock's command alone under the provisional unit cover, where it refuses nothing. On every path nothing holds the gates to the Program, which the design rules call completeness: the developer's to validate, not a gap. | job | One layer owns the tie of the Program to the circuit: Flock, by the digest in the circuit file, on every path. | Lean: Flock's command refuses under the provisional unit cover too, or the cover goes. No longer on Daniel's list, unless a deployed guarantee's statement covers a session under the provisional unit cover, which proofs has not read [relayed]. | proofs |
| F8 | The base Lean packages | They carry one proof system's name: `namespace Flock` or `open Flock` in 7 of 8 files of `core_base`, 9 of 10 of `core_circuits`, 2 of 3 of `core_commitments` and 14 of 19 of the interface. [counted by me] | name | The circuit language and the hash name no proof system. | rename. Listed, not proposed, and lean agrees: a rename changes recorded statements in four locks (about 50 first names, 653 declarations), and any edit of `core_base` has 33 of the 36 packages audited again [relayed, lean]. | lean |

### 2.3 Circuits and commitments

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | The circuits written in the language, `verity/definitions/` | PyTorch, vLLM and models. Files that are not tests: 195 name vLLM, 88 name `torch` as a whole word, 109 name a model family (Qwen, Llama, Gemma, Mistral, DeepSeek, GPT-2 or Phi) [counted by me on `053b8e50fe8a`; the first cut's "a model 111" gave no list of names and vllm could not reproduce it]. | name | Rule 1. The served kernels' circuits go with the adapter that serves them. | move (on the page: Sun 15 Nov). | top, vllm, circuits |
| C2 | The served circuits, the adapter and the circuit checker | The cut of a circuit into pieces, stated three times, each sized to Flock's piece: `CUTS` in nine modules of `verity/definitions/registry/` (`greedy_stages.py:253`); a second table by circuit name in the adapter (`adapters/vllm/verity_vllm/pipeline/serving_templates.py:16`); the checker's "C-Flock's proved range of 2^27 bits" (`tools/verity/circuit_check/src/circuit_check/block_held.py:1-3`, `checks.py:1445`). [read by me; found by circuits 1] The cap is three numbers, not one: a piece of at most 2^27 bits, at most 1,024 regions, at most 64 points (`flock/verifier/Flock/Statement.lean:117-121`). Python states the 27 again (`K_MAX`, `verity/commitments/windows/interface.py:308`) [read by me on `053b8e50fe8a`], and no test holds it to Lean's [relayed, proofs, 2:42 AM PST]. | dup | A circuit knows no piece. The cut is two things: one cutter, whose output is private, and one predicate the verifier holds, whose parameters are the caps (`verity/circuits/partition_v2.py` has this form) [relayed, circuits]. The caps are the range Flock's soundness bound covers: whoever calls SPOT may lower them and never raise them [relayed, proofs]. The cutter is handed the circuit in the language, with its calls and scans, and from the adapter the values it can commit, as data. | code (T16). | circuits, proofs for the cap |
| C3 | Flock's prover, and the circuit checker | The lowering of a circuit to Boolean gates lives in the proof system's prover, so the checker imports the prover: 30 import lines of `verity_flock` in 5 files of `tools/verity/circuit_check/src/circuit_check/` (`checks.py` has 18) [counted by me]; 9 of the 30 are of `ir_lower` and 9 of Flock's own formats [relayed, circuits]. The lowering is 14 modules and 11,956 lines of the prover's 31,457: 5,877 the lowering, 4,662 Flock's statement and file format, 1,417 three named circuits (`flock/prover/python/verity_flock/`) [counted by me on `053b8e50fe8a`; proofs' figures, which agree]. And Flock's prover names single served circuits (F4; circuits 2 [relayed]). | job | The Boolean circuit is the one thing handed over. The lowering is a unit beside the catalog, which may import circuits, the catalog and the one challenge derivation. The proof system takes any Boolean circuit and names none. `struct_lower` stays with Flock: it writes a statement, not gates [relayed, circuits, 2:55 AM PST]. | move the lowering beside the catalog, not into `verity/circuits/` as the first cut said. Two things stand in that way: `ir_lower` reads the catalog, and `struct_lower` writes Flock's template file [relayed, proofs, 2:42 AM PST]. Six of the eight lowering modules read the catalog, so a registry in circuits would be the catalog under another name, and the rule "circuits imports nothing" stays as it is [relayed, circuits]. No rule and no format changes. | circuits, proofs |
| C4 | The circuit checker's suite | Protocols' verifiers and devices: 148 lines of its declared inputs name paths under `verity/protocols/`, `adapters/` or `benchmarks/`. [relayed, circuits 3] | test | A protocol's circuits sit in a module that imports circuits only, so loading one loads no verifier and no device. | code | circuits |
| C5 | The circuit checker and one named circuit | One model, where any circuit should do: `tools/verity/circuit_check/src/circuit_check/block_held.py:11` (`_GLM53_TREES`); `verity/definitions/ops/prims.py:140-142` (a circuit's text names GLM-5.3's tiles). [read by me; found by circuits 7] | name | The model's shapes and pieces come from its compile as one object, which the checker takes as input. | code | circuits |
| C6 | The served circuits' tests | PoUW's and PoUB's code (`verity/definitions/tests/test_pearl_c_call.py:29-31` and four more). [relayed, circuits 6] | test | Each test sits with the protocol whose circuit it tests. | move | circuits |
| M1 | `verity/commitments/domains.py` (409 lines, 341 constants) | Every protocol's and the adapter's format strings, in one table: PoUW's (lines 19-126), SPOT's, Flock's, the ZK server's, PoUS's, PoUB's, the certifier's, compose's, the wire's, and vLLM's (13 `VLLM_*` constants at 183-185, 192-196, 210 and 397-400, such as `"verity-vllm/pouw-pearl-c-run/v1"`). Lean twin: `verity/commitments/identity/Identity/Domains/`. [read by me; the first cut's line ranges missed line 210, as vllm found] By section: PoUW 91 constants, the certifier 49, PoUS 44, PoUB 27, SPOT 23, Flock and the ZK server 23, commitments' own 24 [relayed, SWEEP-ABOVE 3c]. | name | Commitments is handed a tag as bytes. Each protocol defines its own tags, and a test of the whole tree holds them distinct. | code: split the table by owner (T3). | proofs (proposed), each lead for its lines |
| M2 | Commitments | Modules shaped for one caller: `window.py` (`WINDOW_DOMAIN = "verity/one-stage/served-domain/v1"`, which binds program, partition, port, schema, leaves and run), `advice.py`, `history.py`, `pool_row.py`, and `windows/` (9 modules [counted by me on `053b8e50fe8a`; the first cut said 7]: "what an adapter hands the prover for each execution phase"), which loads the served circuits' catalog (`tests/import_order.toml`). [read by me] | shape | Values in, roots and openings out. What an adapter hands a prover is that prover's interface. | move: `windows/` to SPOT's `prover/`; `advice.py` and `pool_row.py` to their callers. | proofs, vllm |
| M3 | Commitments' tests | The adapter's files, listed as test inputs (`verity/commitments/pyproject.toml:10-11`) [read by me]. One test imports PoUB's verifier, and the listing counts two files of PoUB's verifier as commitments' (`verity/commitments/tests/test_block_check.py:17-24`; `poub/verifier/commitments/gates/block_check.py`) [relayed, SWEEP-ABOVE 3c]. | test | A package's tests read its own package. | code and move | proofs, memory-accounting |
| M4 | PoUS, PoUB, the ZK server | A commitment's root, computed outside commitments: PoUS's own Merkle tree (`pous/verifier/check/PousCheck/Merkle.lean:6-9,76`), which PoUB imports (`poub/verifier/lean/RemoteAudit/P2.lean:3`) [read by me]; the ZK server's frame tree, the memory challenger's challenge tree, and PoUB's stream function twice in one package [relayed, SWEEP-ABOVE 4e, 4f]. | dup | Commitments hands back roots. A protocol hands values. One tree, the shared one (`AGENTS.md`: one hash, one kind of tree). | Lean and code. **To Daniel** if a guarantee's hash premise changes with the tree. | memory-accounting, proofs |

### 2.4 The shared files, the runtime and registration

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Nine protocol files | The runtime's package, for a type. `ProtocolDef` (`from verity.runtime.runner import ProtocolDef`): `pous/device/challenger_party.py:66`, `poub/verifier/session_parties.py:47`, `spot/verifier/parties.py:126`, and one more. `stream.Side`: 5 files, such as `poub/verifier/verifier_party.py:34`. [read by me] `ProtocolDef` is 5 lines: a manifest and one factory per party (`verity/runtime/runner.py:73-77`). | import | The types sit in `protocol.py` and `channel.py`. The runtime calls a protocol, and no protocol imports the runtime (rule 3). | move the types (memory's part is MEM-291 [relayed]). | proofs, memory-accounting |
| S2 | The configuration's reader | Compose's two lists, with every protocol's name, and it refuses any other list in Python. `verity/config.py:59-61` (`COMPOSITES`: onsite's 7 names, remote's 3). [read by me] With P8 and G3 the lists are stated four times: compose's Lean (`onsite/verifier/CoreOnsite.lean:63-65`), the configuration, SPOT's parties and the registered hardware. | name, job, dup | `config.py` reads a list of protocol ids. Whether a list is acceptable is compose's to say, in Lean [page]. | code | proofs |
| S3 | The hardware package | The transcript's signed entry, `verity/hardware/record.py`, which SPOT's challenger reads. [read by me] | job | A signed entry of a transcript is the channel's. | move into `channel.py` (proofs [relayed]). | proofs |
| S4 | The period | The period's type three times, and a fourth and fifth by name. The lengths (a week, a month, a year) are stated once now, in `verity/protocols/reporting.py:11`; the second list went with the first chain of moves. The type: `security/Definitions/Core/Period.lean`, its stated copy `interface/CoreProtocol/Period.lean`, and the Python mirror. [read by me] Two more types named `MonitoringPeriod` sit in SPOT's challenger and in the slowdown calculator (`spot/verifier/lean/Spot/ChallengerSession.lean:481`, `inference_only/rho/rho/hardware.py:101`) [relayed, SWEEP-ABOVE 4d]. | dup | One period type, beside the account it is summed in. | code | proofs (proposed) |
| S5 | `verity/claims` (623 lines) | A list of claim ids kept by hand beside the locks. 10 Python files read it [counted by me]; SPOT's challenger uses it for one membership test (`spot/device/challenger.py:378`). About 80 Lean docstrings and three locks name its ids [relayed, top]; SWEEP-ABOVE counts 308 Lean files that cite one of its 207 ids. PoUS's verifier keeps a second table of 9 claims (`pous/verifier/protocol.py:45`), and PoUW's provers write claim ids as literals [relayed, SWEEP-ABOVE 6]. | dup | The locks are the one registry of guarantees and assumptions. A generated file maps an id to its Lean name (soundness [relayed]). | code: a generator in the Lean check; the Glossary's line on a guarantee changes. Each named assumption gains an id field in its lock, so all 36 packages are audited again once; no record and no statement changes [relayed, lean]. | lean, soundness |
| S6 | The import listing | One row holds `channel.py`, `config.py`, `party.py`, `protocol.py` and `runtime`, and that row sits below every protocol. One place holds the runtime with `config.py`, `party.py` and `protocol.py`, so a load of a shared type reads as a load of the runtime, and SPOT's `loads` names `verity/runtime`. `tests/import_order.toml`, `order` and the places. [read by me, on `e4da505fe488`] | test | The shared files are a row below every protocol. The runtime is a row above them all. | code (TOP-398 [relayed]). | top |
| S7 | The hardware directory | The devices' outside world: a Lean package of clock, random bytes and sleep, "the outside world, as a core device or challenger reaches it" (`verity/hardware/world/World.lean:1-12`), which the certifier's device requires (`certifier/device/lean/lakefile.toml:23` [relayed, SWEEP-ABOVE 2a]). [read by me] | job | Sockets, randomness and the clock are the runtime's [page]. The hardware directory is data only (rule 2). | move to the runtime. | proofs |
| G1 | The interface package | Registration, the registered hardware, compose's account and the period: `Site.lean` 702 lines, `Registration.lean` 535, `Account.lean` 861, `Period.lean` 147, of the package's 3,717. Outside the package `Site` is read by 3 files, `Period` by 1, `Registration` and `Account` by none. `interface/CoreProtocol/`. [counted by me; lean's table says the same, relayed] 13 Lake packages require the package [relayed, SWEEP-ABOVE 1]. The first cut added "its one library target builds all of it for each", which is wrong: a package that requires another builds the modules it imports, and the package is built once [relayed, lean]. | job | The interface hands back only what a check implements. Registration is a package of its own on it. The account is compose's. | move: three modules leave for a package `registration_check` (`Registration.lean`, `RegistrationMain.lean`, `Account.lean`, 1,404 lines); `Site.lean` and `Period.lean` stay at first. Four PRs, about 31 files and 1,495 lines of Lean [relayed, proofs, 1:56 AM PST; this replaces the first cut's "about 45 files"]. | proofs, soundness |
| G2 | PoUW, PoUS, PoUB, the certifier | None reads the accepted registration. No file of a protocol imports the registered hardware or the registration log in Python. In Lean one module does (`certifier/verifier/lean/NetworkCheck/Registered.lean:3`) and only a test's vector file calls it. [counted by me] An answer that names no head counts for nothing in the account (`interface/CoreProtocol/Account.lean:280-283`) [read by me]. The page says "None does". The registration does hold each auditor's device's key, and no period check reads it, which is G7. | shape | Each accounting protocol is handed the registration's head and its part, and its answer names both, with the period's span and the auditor's device key (soundness's form [relayed]). Compose joins answers by comparing them. | code and Lean: each published answer gains the fields; one lemma, `NamesRegistered f`, says an accepted answer names a registered part [relayed, soundness]. **To Daniel**: it changes what each answer says (soundness keeps it on his list with N7 [relayed]). | compute-accounting, memory-accounting, network-accounting; proofs for the type |
| G3 | The registered hardware | Compose's two lists, by name. `interface/CoreProtocol/Site.lean:285-286` (`Composite`: onsite, remote), `:372,376` (`Covered` and `Reached` hold "if the composite is onsite"); `verity/device.py:61`. [read by me] | name | Registration is handed the run's list and asks whether it holds a protocol that accounts for this link or this memory. | Lean: restate both over the list (soundness 4 [relayed]). Not Daniel's: soundness struck it, since the restated form says the same of the two lists there are [relayed, soundness, 2:15 AM PST]. | proofs, soundness |
| G4 | The adapter, PoUS, PoUB | The weights are committed several times with nothing tying the roots: the served run's root for each weight, PoUS's one hash, PoUB's tree [relayed, memory-accounting 3, vllm 4]. vllm says this undercounts: the adapter has three registration forms, and the offline Commit has its own weights tree; the pin a root is compared with is the committer's own file; and a vLLM serve passes no committed weights, so a served proof opens no registered root [relayed, vllm; proofs, 1:25 AM PST]. No caller under `adapters/`, `benchmarks/` or `demos/` passes `weight_values=` [counted by me on `053b8e50fe8a`]. Compose's own text says of two of its forms: "nothing here holds PoUW's committed weights to the weights the record registers" (`remote/verifier/CoreRemote.lean`, header) [read by me]. The policy's Lean check does read a log's first record by a parameter's name and root (`inference_only/verifier/nci/lean/NciCheck/Record.lean:559-566`), but no command hands it such a log [relayed, vllm, 2:43 AM PST]. | dup | One commitment to the weights, in the registration. Each protocol is handed that root and compares. The adapter, which commits, needs more than roots back: for each value its root, domain, size, row width, row format and salt nonce, under one numbering rule stated once [relayed, vllm, 1:36 AM PST]. | code and Lean (vllm's VLLM-424 [relayed]). **To Daniel**: it adds a tie a guarantee lacks. The tie of the registered rows to the checkpoint is completeness and is not his (soundness [relayed]). | proofs, memory-accounting, compute-accounting, vllm |
| G5 | The data sheet | Four Lean definitions, one set of Python literals, and the TOML: `interface/CoreProtocol/Site.lean:150`, `security/Definitions/Core/Site.lean:137`, `security/Definitions/Accounting/TS/Sheet.lean:68`, `security/Definitions/Core/Accel/Card.lean:31`; `pouw/prover/circuit/h1device.py:19-23`; `verity/hardware/cards/`. The hardware package also writes Lean under `security/`, and a card names its Lean module (`verity/hardware/lean.py:21-22`, `cards/h200-sxm-141gb.toml:14`). [read by me] | dup | One type. The registration's `Sheet` stays and grows; `Card` and `TS.Sheet` become functions of it [relayed, compute-research]. A statement takes a sheet as a parameter. The registration check holds a device's figures to the auditor's sheet [page, section 10]. | Lean and code (on the page: Tue 24 Nov). **To Daniel** for its second step only: PoUW's work statements would read a card off the registration, which changes what they rest on [relayed, compute-research]. One figure differs today: the registration's reader fills a card's tensor clock from `clock.sm` (`verity/device.py:316-317`), 1,980 MHz for the H200, where the card's `clock.tensor` is 1,830 MHz (`verity/hardware/cards/h200-sxm-141gb.toml:23-24`) [read by me on `053b8e50fe8a`]. A measurement due Tue 17 Nov settles which; 1,830 would lower the registered tensor rate by 7.6% [relayed, compute-research]. | compute-research, proofs |
| G6 | PoUW's check and PoUB's device | A list of served models, by file path. `pouw/verifier/lean/check/PouwCheck/Credit.lean:1634` (`include_str "../models.json"`; the file is inside the check's own package); `poub/device/live_verifier/host.py:165` (`MODEL_PINS`, PoUW's file). [read by me] The file pins 7 models' checkpoints and has five readers, compose's device among them (`verity/composite/device/live_challenger/models.py:53`) [relayed, SWEEP-ABOVE 3b]. | name, import | The registration hands the registered circuit's shapes and the weights' roots, as data. The check reads each role's `checkpoint_sha512` from the list, and the registration can hand that [relayed, compute-accounting]. | code | compute-accounting, memory-accounting |
| G7 | The certifier, PoUW's challenger, PoUS's memory challenger and the wipe | A device's key comes from the run's own files, which the prover writes, not from the auditor's configuration or the registration. The registration has the place: a location's `key`, which registration's check holds to the auditor's own (`interface/CoreProtocol/Site.lean:169`, `:365-368`). One function reads it, `keyRefusal` (`certifier/verifier/lean/NetworkCheck/Registered.lean:84`, called at `:105`), in a module only a test's vector file calls (G2). The certifier's period scan takes the key from the deployment file beside the run (`benchmarks/network_traces/scan_period.py:109`), and the period check verifies under the key it is handed (`NetworkCheck/Period.lean:835,908`) [read by me on `053b8e50fe8a`]. PoUS's served case takes the memory challenger's keys from the run's own `run.json`, as `trusted` (`benchmarks/pous/served_case.py:237-242`) [read by me on `053b8e50fe8a`]. The same holds for PoUW's challenger and the wipe, and SPOT's one-stage check compares the challenger's public key with nothing [relayed, soundness, item 5 of the inbox; not checked by me]. A device's key also has a type of its own in each protocol: `Signer` in commitments, in PoUS's device and in the certifier's verifier, `ChallengerKey` in SPOT [relayed, SWEEP-ABOVE 4g; the first cut left this out]. | reads | One type for a device's key, in commitments. A device's key is registered, or pinned in the auditor's configuration. Each check is handed the pinned key by its caller and refuses a signature under any other. | code: each driver reads the pins from the auditor's configuration (soundness is mending it in #9649, which closes it for compose's verifier and not for the one-stage check run alone [relayed]); then each period check reads the registered key (T17). | soundness, each accounting lead |
| G8 | The registration | It holds no read rate for a memory leaf under a server, and no figure for the server's CPU [the CPU: relayed, proofs]. A location has eight fields and none is a read rate (`interface/CoreProtocol/Site.lean:161-169`); the account says the registration holds no rate for a leaf under a host (`interface/CoreProtocol/Account.lean:171-176`); the reader takes a leaf's tier and bytes only (`verity/device.py:497-501`), though the node's file states the rates (`verity/hardware/nodes/rtx-pro-6000-server-x8.toml:29-39`) [read by me on `053b8e50fe8a`]. So an account of such a part reads "unbounded" [relayed, soundness]. | shape | The registration records every capacity a data sheet states, and each accounting protocol reads its part's capacity from the registration only. | Lean and code: a ninth field. It changes every site entry's digest under `verity/registration/site/v1` [relayed, soundness; proofs: a fifth PR]. **To Daniel**: it changes what a memory bound rests on. The read rate on a memory leaf is option A of memory-accounting's page [relayed]. | proofs, memory-accounting, compute-research |

### 2.5 The accounting protocols and the policy

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| W1 | The network certifier's device, and PoUW | The device draws PoUW's challenge and keeps PoUW's weights' root: `certifier/verifier/active.py:72-95` (`Proxy(salt=...)`, one sealed salt for each PoUW window, and "the 64-byte root of the weights PoUW's noise seed reads"). PoUW's execution phase is counted in the certifier's windows: `pouw/verifier/salt.py:1-13`, `pouw/verifier/core_check.py:1129-1166`. 16 of PoUW's files name the certifier and 12 of the certifier's name PoUW. [read and counted by me] | job | PoUW is handed a live salt as a challenge from one of the auditor's devices. The salt is drawn after the window's commitments it seeds are recorded, under a key the auditor pinned, and bound to that window and epoch by the device's signature [relayed, compute-accounting]. Which device, and how phases line up with the certifier's windows, is compose's tie. The certifier's device stamps and counts traffic and does not know whose salt it carries. | code and Lean. **To Daniel**: PoUW's noise seeds would rest on a named device's challenge, not on the certifier's (soundness keeps it on his list [relayed]). | compute-accounting, network-accounting, soundness |
| W2 | PoUW's driver, under `benchmarks/` | A serving system's run records and paths (`benchmarks/pouw/pearl_c_vllm/served_passes.py`, `verify_run.py`). [relayed, compute-accounting 4; both files are under `benchmarks/` and not under PoUW, as vllm found and I read on `053b8e50fe8a`] | name | The adapter hands a record in PoUW's own format. | code | compute-accounting |
| W3 | PoUW's served prover | An experimental variant's module, in served code. Dither has no such import. The FP4 variant is the coupling [relayed, compute-accounting, 1:07 AM PST]: the served `pouw/prover/circuit/plan.py:283`, `pc8.py:74`, `cap.py:137` and `rowk.py:57` import `pc4`, the FP4 variant's module [read by me on `053b8e50fe8a`; the first cut searched `pouw/verifier/` only and found none]. Served tests import it too [relayed]. | import | A variant imports PoUW, never the reverse. | code: what the served scheme shares moves out of `pc4.py` into PoUW, and the FP4 variant imports PoUW (after the chains [relayed]). | compute-accounting |
| U1 | PoUB's v0 verifier and device | The model: a pinned checkpoint, and a serving system's parameter order and layer names. `poub/verifier/store.py:1-30` ("every parameter of vLLM's model with two or more dimensions, in its module order"); `poub/device/live_verifier/host.py:41-59` (`--model`, `--checkpoint`). 17 files under `poub/` that are not tests name a model or a checkpoint [read and counted by me on `053b8e50fe8a`; the first cut's 23 counted tests]. All of this evidence is v0's (`verity/remote-bandwidth/v0`), which is on neither of compose's lists. Onsite's PoUB-P2 names no model: its registration is PoUS's setup's published outputs [relayed, memory-accounting]. | name, job | PoUB audits reads of a store named by a root, and is handed PoUS's accepted setup answer by name. The model's tie is PoUS's alone, once (G4). | code and Lean (`RemoteAudit/Store.lean`), needed only if v0 goes on remote's list [relayed, memory-accounting]. Daniel refused the memory challenger reading the store at setup [relayed]. | memory-accounting |
| U2 | The memory challenger's device | Another protocol's store, in the clear: 24,576 blocks a session (6 windows, 1,024 sampled units each, 4 blocks a unit), kept in `private/cases.jsonl`. `poub/verifier/p2_party.py:87`, `p2.py:65`, `p2_party.py:774`. [read by me; the product is mine] The blocks are for the opening's check (`poub/verifier/p2.py:32-35` [read by me]), which runs inside the party on the device. Nothing reads the device's kept copy; a session opens 2.47% of the store's units, and 93 sessions open 90% [relayed, memory-accounting, 1:57 AM PST]. | job | A device signs what it sees and hands it to the prover. The auditor sees only proofs about it [page, section 1]. The least the memory challenger must see is nothing of the store: what it draws, each answer's two clock readings and length, and a digest only an answer made after the challenge could give [relayed, memory-accounting; argued, not proved or sized]. | Lean and code: the opening's check becomes a proof the prover makes (on the page, not dated). The device stops keeping the cases by Wed 18 Nov [relayed]. **To Daniel** (the page's section 9; memory-accounting's ASK-300 is with him, with a default on Mon 16 Nov 6:00 PM PST [relayed]). | memory-accounting |
| U3 | PoUB's session over PoUS's store | PoUS's device's record: PoUB builds its own registration of the store from the memory challenger's record of PoUS's session. `benchmarks/pous/served_piece.py:53-56` [read by me]; `p2_party.py --pous-record` [relayed]. | shape | PoUB is handed PoUS's accepted setup answer by name (`Account.Rests`), and compose checks it. Not "from the registration", as the first cut said: the store's root is made at PoUS's setup, under a salt the memory challenger draws after the commitment [relayed, memory-accounting]. | code | memory-accounting |
| U4 | PoUS's grader | The layout of `security/`: `pous/lean/TRUSTED.sha256` pins 58 files under `security/` by path [counted by me], and the grader's audit key does not see them, so a change to one does not have the grader audited again [relayed, lean]; the listing ties PoUS to `security/` (`pous/lean/submissions/reference.lean`) [read by me]. | import | No code uses `security/` (rule 2). The trusted text reaches the grader as a declared input, by digest (lean 5 [relayed]). | code | memory-accounting, lean |
| U5 | PoUS's area of `security/` | The network certifiers' count: the wipe's bound is stated over "every count of the egress by the network certifiers". `security/Properties/Pous/Wipe.lean:14-26,49-52`. [read by me] | reads | Each protocol states its own count. The two meet in compose [page, section 10]. | Lean: the joined statement moves to the composition area, word for word (soundness 5 [relayed]). Not Daniel's: soundness struck it, since the statement does not change [relayed, 1:07 AM PST]. | soundness, memory-accounting, network-accounting |
| U6 | Memory's answers | Five ties are missing: no memory answer publishes the head or a memory leaf (`pous/verifier/check/PousCheck/RunProtocol.lean:203-209` publishes `certified`, `eps`, `s_bits` only [read by me]); the memory challenger's key is a pin in the run's parameters; the wipe's tiers are names the prover's request declares; PoUB's `profile.registration` is its own; the wipe's egress is a number the challenger is started with. [relayed, memory-accounting] | shape | As G2. Each memory answer names a leaf of the registration: PoUS `memory [hbm]` at the card its store sits on; PoUB-P2 `read [hbm]` at the same card; the wipe every memory leaf at or under one server. Each check is handed `registration_head` and `place` in parameters the verifier owns, and publishes both [relayed, memory-accounting]. | Lean and code. **To Daniel**, with G2. | memory-accounting |
| U7 | PoUS's check | A device's package: the wipe's check sits in the memory challenger's package (`pous/device/lean/MemoryChallenger/WipeProtocol.lean`, `StreamedWipe.lean`, 436 lines), and PoUS's check requires that package (`pous/verifier/check/lakefile.toml:35-37`) and imports its judge (`PousCheck/Run.lean:3`). [read by me on `053b8e50fe8a`; found by memory-accounting, and by the sweeps as "a verifier requires a device's package"; the first cut left it out] | import, job | `device/` holds the device and `verifier/` holds the check (as N4). A check requires no device's package. | move the wipe's check and the judge to the check's package (after the chains [relayed, memory-accounting]). No record moves with it: the device's lock holds no guarantee [read by me on `053b8e50fe8a`]. | memory-accounting, lean |
| N1 | The certifier's period check | SPOT's record of a window: its commitment is `{"audits": [record, ...], "windows": [record, ...]}`, each record of format `verity/registration/v1` or `v2`. `certifier/verifier/lean/NetworkCheck/Period.lean:86-89`, with `Flock.WindowRecord`, `Flock.Public` and `Flock.Program` imported. [read by me] The check reads five fields of a record (`format`, `program`, `window`, and the entries of `roots` and `registered`) and only hashes the rest [relayed, network-accounting]. | shape, import | The certifier hands SPOT a circuit, its pieces and a rule, and gets back a verdict over roots. It commits to roots, each with its domain and its port's name, with the Program's digest and one digest that names SPOT's part; not to SPOT's records [relayed, network-accounting]. That a compute audit's bytes crossed a counted link is compose's tie. | code and Lean: the commitment's form changes, so its digest does. **To Daniel**. | network-accounting, proofs |
| N2 | The certifier's Lean package | The policy's: `network_check` requires `nci_check` for `NciCheck.Desc` and `NciCheck.DefinitionDigest`. Its Lake file; `certifier/verifier/lean/NetworkCheck/Period.lean:9-10`, used at 266, 541, 562, 772. The listing's order allows it: the policy's row is below the certifier's (`tests/import_order.toml`, `order`), where the page puts the policy on top of accounting. [read by me] The period check needs about 70 lines of `Desc` (the layout predicates, `Desc.lean:989-1056`) and never calls the summary; five files of `security/` import `NciCheck.Desc` [relayed, network-accounting]. `DefinitionDigest` imports the base and circuits packages only, and `Desc` imports the policy's `NciCheck.Check` [read by me on `053b8e50fe8a`]. After the group map this is the one Lake require that crosses from accounting to the policy [relayed, GROUPS]. | import | The summary of a circuit's description lives with the circuit language. Neither protocol requires the other for it. | move the digest and the layout predicates to `core_circuits`; the policy's tail of `Desc` stays with the policy. | network-accounting, lean, proofs |
| N3 | The certifier's verifier and prover | The serving system. The advice format is "vLLM v1, the pinned 0.28.1rc1.dev472's `Scheduler.schedule` and `SchedulerOutput`" (`certifier/verifier/advice.py:1-20`) [read by me]. Files under `certifier/` that are not tests, naming: request 65, vLLM 23, scheduler 14 [counted by me on `053b8e50fe8a`; vllm counts 71, 27, 14; the first cut's 136, 39, 21 counted tests]. One step of that build's scheduler is a circuit in the certifier's verifier (`certifier/verifier/replay_step.py:1-15`) [relayed, SWEEP-ABOVE 3b]. The charge is not the advice's length today: the committed advice is the raw decisions, and the charge is their coded length under a registered book, counted inside the compile slice's proof block; the guarantee reads that code (`security/Definitions/NetworkCertifier/AdviceV1.lean`) [relayed, network-accounting; not read by me]. | name, job | `AGENTS.md`'s rule: the adapter registers a compiler, a circuit with a public length cap, and hands advice as bits Verity never interprets. The certifier checks the circuit and charges the length. | move the compiler and the advice format to the adapter: seven files of `verifier/`, 5,159 lines (`advice.py`, `arith_encode.py`, `cache_step.py`, `compile_slice.py`, `replay_chain.py`, `replay_step.py`, `served_rows.py`) [counted by me on `053b8e50fe8a`]. Three things stay: the check that a window's circuit is the registered compiler's on the committed advice, the charge, and the certified input. **To Daniel** for the charge only: it is a statement, not code. Either the committed advice becomes the coded string and the charge is its committed length, or the code stays in Verity as the one registered code. Each changes what the bound's advice term is defined by. network-accounting leans to the first [relayed]. | network-accounting, vllm, circuits |
| N4 | The certifier's `verifier/` | The device's own code: four files, `active.py` (1,898 lines, the proxy), `warden.py` (1,213, the relay), `pacer.py` (345) and `coin_link.py` (436) [counted by me on `053b8e50fe8a`]. `benchmarks/network_traces/` is 36 files and 25,767 lines outside its tests, and the device's serving loop is part of it [counted by me; the first cut called all of it the serving loop]. And the check's Lake file requires a path under `device/`: `device/program/lean` is the certifier program, not the device (`certifier/verifier/lean/lakefile.toml:26`) [read by me; found by network-accounting]. | job | `device/` holds the device. `verifier/` holds the check. | code first, then move (the plan with days is due Mon 16 Nov [page]). Seven files of the formats and the check import `active` or `warden` for the cell's form and the schedule, and `active.py` holds one check, `check_salts`. So the cell's form and the schedule's parameters go to a module both sides import, and `check_salts` to the check, before any path moves [relayed, network-accounting]. | network-accounting |
| N5 | The certifier's link for challenges | SPOT's challenger: `certifier/verifier/coin_link.py:52-53` imports `LiveCommitment`, `LiveOpening`, `live_reason`, `HEADER` and `MAX_HEAD` from SPOT's device. [read by me] | import | The certifier counts and re-clocks frames of the shared channel and does not read whose challenges they carry. | code: the frame's header goes into `channel.py`. | network-accounting, proofs |
| N6 | The certifier's tests | The policy's difftests: 8 test files under `certifier/verifier/network/tests/`. [counted by me] | test | They go with the policy. | move | network-accounting |
| N7 | The period's parameters | One link and one key, held to no registered link: the period's key and session are copied from the certifier run's own deployment file (G7) [relayed, network-accounting 12; soundness]. | shape | As G2: the registration hands the certified links with their keys, and compose compares. | Lean and code (network-accounting's #9687 makes them lists [relayed]). **To Daniel**, with G2 (soundness keeps it on his list). | network-accounting, soundness |
| N8 | The certifier's cell | The serving system's request: a cell's payload holds an id request's `max_tokens` field, which the device reads off the request's first cell (`certifier/verifier/lean/NetworkCheck/Period.lean:67-68`, `maxTokensEnd` at `:292`; `certifier/verifier/active.py:462-468`). [read by me on `053b8e50fe8a`; found by network-accounting; the first cut had no row] | name, shape | The device stamps and counts bytes. A cap on a response is a number the adapter registers, and the device reads no field of a request. | code and Lean, with N3's move of the compiler. | network-accounting, vllm |
| I1 | The policy's Lean package | SPOT's insides: `nci_check` requires `core_spot`, and `inference_only/verifier/nci/lean/NciCheck/Record.lean:6` and `RecordText.lean:5` import `Spot.Wires`. [read by me] | import | The policy is handed the circuit's description [page]. It reads no record of a run. | code | network-accounting |
| I2 | The policy's directory | Two calculators (`nci/`, `rho/`). One imports SPOT's profile (`inference_only/rho/rho/params.py:38-39`) and restates the certifier's request layout (`rho/rho/envelopes.py:280`). [read by me] | job | A calculator is a benchmark. What it needs reaches it as five public figures: the monitoring periods' lengths and the reporting δs, the per-window δ, a piece's smallest and widest output class, the policy's `K` and `Γ`, and a request's declared bits with a bucket's and the requests a link-window [relayed, training]. | move (chain 2's map [relayed, network-accounting]). | network-accounting, training |

### 2.6 Compose and `security/`

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| X1 | Compose's remote check | Each protocol's record format: it ties Pearl+'s run to "the digest of the registration record", holds the policy's record to be "chained right after W (its `previous` is W's digest)", and reads W's `pearl-b/` values. `remote/verifier/CoreRemote.lean`, header. [read by me] | shape | Each protocol hands compose the head and the roots it committed to. Compose compares them and reads no record's fields. | Lean. **To Daniel** through soundness (it stays on his list [relayed, soundness]). | proofs, soundness |
| X2 | Compose's theorem | SPOT's and Flock's proof internals (`OpensAt`, `Layout.slotAt`, `Flock.Bridge.pathOf` under `security/Proofs/Properties/Verity/Sampled/`) [relayed, soundness 2]. That part is 76 files and 33,601 lines; the 95 files and 41,677 lines the first cut gave are `security/Proofs/Properties/Verity/` whole [counted by me on `053b8e50fe8a`; corrected by proofs]. | reads | SPOT exports one knowledge-soundness theorem over the circuit, its pieces, the sampling rule and the challenges. The sampled leg reads only that. In the proofs tree the composition area imports a protocol's `<Area>.SecurityProofs.G` theorems and never an inner lemma [relayed, soundness]. | Lean, soundness's to do. Not Daniel's if SPOT's exported theorem states what the leg reads today; soundness keeps it for him if it differs [relayed]. | soundness, proofs |
| X3 | The security game's vocabulary | Protocols' internals: it reads the circuit's digest out of the verifier's parameters (`security/Properties/Verity/Game.lean:676`, `:781`) [read by me]. Nothing ties that digest to the circuit the registration declared (`interface/CoreProtocol/Registration.lean:94`) [relayed, soundness]. | reads | The registration's accepted record and SPOT's parameters are both inputs of the composite, and the tie (`DeclaredIsParams`) is a fact of compose's theorem. | Lean. **To Daniel** (it stays on his list: a new tie [relayed, soundness]). soundness's one lemma `NamesRegistered f`, with `f` the layout, is the form: SPOT's circuit file is the layout of the registered Program [relayed]. | soundness |
| X4 | Compose's lists | Four homes: `onsite/`, `remote/` (onsite's package requires remote's), `remote/verifier/composition/`, and `verity/composite/`. [read by me] | import | One `compose/` package. Remote and onsite are two lists of ids. | move (on the page: Sat 14 Nov). | proofs |
| X5 | The composite's device | Another protocol's processes, and a benchmark: `verity/composite/device/live_challenger/coin_link.py:40` imports the certifier's link [read by me]; its `auditor.py` names the certifier's processes [relayed] and is tied to `benchmarks/` in the listing [read by me]; its `models.py:45-46` imports PoUW's verifier [relayed, SWEEP-ABOVE 2c]. | import, name | Compose starts each protocol's device through one interface, read off the run's list, through the runtime. | code | proofs |
| X6 | Compose's hand-off | Each protocol's private files by name (`served_verdicts.py`, `WITNESS_*`). [relayed, memory-accounting 8] | name | Each protocol declares which of its outputs are witness. | code | proofs |
| X7 | The game's `Holds` | Protocol facts written inline for each protocol. [relayed, soundness 3] | reads | `Holds` reads each protocol's exported claim by its id on the list. | Lean | soundness |
| Y1 | `security/Definitions/Core` | An accounting module: `Core/Accel/Card.lean:4`, `Core/Blocks.lean:4` and `Core/Site.lean:3` import `Definitions.Accounting.Device`, and `security/lean-audit.json`'s `layers` allows it. [read by me] `Core` reads three structures of it, `Device`, `OpClass` and `Window`, and nothing else [relayed, lean]. One more of the same kind: `security/Definitions/Core/Requirement.lean:33` declares its names under `Accounting.Requirement` [read by me on `053b8e50fe8a`; found by lean]. | reads | The shared vocabulary reads no area. | Lean: the three structures move into `Core` under their present names. No record changes (a statement's hash is over names, not modules); 3 of the 36 packages are audited again [relayed, lean]. | soundness, lean |
| Y2 | Listed guarantees' statements | 28 entries of `trees.except` in `security/lean-audit.json`: 21 under `Proofs.Flock`, and 26 name the spec's extraction as what they wait on [counted by me on `053b8e50fe8a`; the first cut said 27, and said modules of the two trees read them, where it is guarantees' statements that do, as lean found]. Under those entries the lock records 6,035 read definitions in 679 modules of the proofs tree; 546 of the modules reach VCVio or ArkLib through their imports, which the definitions and properties trees may not require; 133 modules (964 definitions) reach neither and can move as they are [relayed, lean, 2:06 AM PST]. | reads | A statement never reads a proof [page]. | Lean: Flock's spec, the definitions those statements read, is extracted out of the proofs tree into `Definitions`, or `Properties` where it is code. An entry leaves when its definitions do, not when a statement moves. A definition that moves under its own name changes no record. Not Daniel's [relayed, soundness, 2:07 AM PST; lean]. | proofs; soundness checks each extraction; lean |
| Y3 | `security/Definitions/Flock` (35 files) | SPOT's and recursion's vocabulary (`Audit/Law.lean`, `Partitioning.lean`, `Guarantees/SamplingClaims.lean`, `Recursive/Holo.lean`), and a directory named `Accounting/` that is Flock's own error budget. `Definitions/Spot` (9 files, 888 lines) imports three `Core` modules only and matches Protocol 1. [read and counted by me] A third home: `security/Definitions/Claims` (23 files) holds Flock's claim statements, SPOT's sampled soundness among them [relayed, SWEEP-PROOFS 1e]. | dup, name | One area for each layer: sampling in `Spot`, the outer proof in an area of recursion's, `Flock` the proof system alone. | move and rename. | proofs, lean |
| Y4 | `security/Definitions/Core` (84 files, 9,823 lines) | Things one area alone reads: `Compose`, `Rec`, `NaiveStrassen` and `Transport` [relayed, soundness, checked against the vocabulary test]. The first cut listed thirteen files, among them `Site`, `Period`, `Accel`, `DeviceFigures`, `RegisteredCircuit`, `Challenger`, `Compiler`, `Advice` and `Blocks` [counted by me; the files listed, not all read]. That was too wide: eight shared concepts are defined in those files (`Site`, `Link`, `Registration`, `Session`, `Card`, `Machine`, `Prog`, `Block`), more than one area reads them, and `tests/test_lean_vocabulary.py` holds them to `Core` or `Hash` [relayed, lean, soundness]. | job | `Core` holds what more than one guarantee reads, as `AGENTS.md` says. Compose's, recursion's and PoUW's four files go to their areas. | move the four. `AGENTS.md` does not change. Not Daniel's [relayed, soundness]. | soundness, lean |
| Y5 | Shared concepts with two definitions | A resource: `security/Definitions/Accounting/Joint.lean:38` and `interface/CoreProtocol/Account.lean:103`. A protocol's id: `Definitions/Accounting/JointPeriod.lean:39` and `Properties/Verity/Game.lean:132`. A window: `Definitions/Accounting/Device.lean:81`, `NetworkCheck/Period.lean:412`. A tier: `interface/CoreProtocol/Site.lean:66` and `Definitions/Core/Site.lean:116`. [read by me] In Python the memory tiers are listed five times, and not alike: the registered hardware's, the certifier's and the policy's say hbm, dram, disk, and PoUB's says hbm, host (`poub/verifier/session/verifier.py:153`). 26 Lean definitions are named `Registration`. [relayed, SWEEP-ABOVE 4b, 4h] | dup | One definition each, in the layer that owns it. | Lean | soundness |
| Y6 | Guarantees | One card family, by name. A benchmark writes `security/Definitions/Accounting/Headline/Sm*.lean` (`benchmarks/machine_model/verity_machine_model/lean_headline.py:2`) [read by me]; eleven Lean files name one card's table [relayed, compute-research 1]. Two things in different states [relayed, compute-research]: for the card's tables, the statement at every card is on `main` and proved (`WorkSoundAtTables`); for the machine model's headline files, five per-card headlines stand and nothing general is built. | name, job | One guarantee over any sheet. A card is a witness the caller supplies. No Python writes under `security/`. | Lean. **To Daniel**: retiring the four per-card guarantees and their four witnesses, once the family's guarantee lands [relayed, compute-research]. One guarantee over any sheet for the headlines is not scoped or dated. | compute-research, compute-accounting |
| Y7 | The three trees of `security/` | Withdrawn. The first cut said the three trees share one source directory, "so the lock's globs hold them apart, not Lake". The two lines it cites stand, and the conclusion is wrong: each Lake file's `roots` say which package builds which file, `security` requires only Mathlib, and `security/Properties` does not require `security/Proofs`, so an import the wrong way does not build [read by me on `053b8e50fe8a`: the three Lake files; found by lean]. The lock's `trees` holds only what Lake cannot see: that a statement reads no definition of the proofs tree (Y2). | (none) | As it is. | none | lean |
| Y8 | A benchmark | The lock, by Lean declaration name: `benchmarks/pouw/harness/price_twins.py:62-67`. [read by me] A rename of Lean names fails seven tests of the benchmark's ledger [relayed, compute-research]. | name | The Lean check hands out a guarantee's stated constant by the guarantee's id (compute-research 3 [relayed]). | code | lean, compute-research |
| Y9 | Flock's proofs | The certifier's name: Flock's byte lemmas are declared inside `namespace Properties.NetworkCertifier.Bytes`, so that the certifier's lemmas "build on these, under the same names"; 21 files of `security/Proofs/Flock` open it. `security/Proofs/Flock/Bytes.lean:16-17,26`. [read by me; found by SWEEP-PROOFS 1e] | name | Byte lemmas are the hash area's, under its own name. | rename. One recorded name changes (`Properties.NetworkCertifier.Bytes.bs`), and one entry of the moves file carries it [relayed, lean]. | proofs, lean |
| Y10 | The accounting areas' and the policy's proofs | The proof system's inner modules: `security/Proofs/Accounting` imports `Flock.ReadNet.*` and four more in 11 files, and `security/Proofs/Properties/Nci` imports `Spot.WiresSound` and three more in 10. In the other direction one Flock proof imports an accounting definition (`security/Proofs/Flock/SoundAtBridge/Union.lean:3`). [relayed, SWEEP-PROOFS 4b, 1e] | reads | A protocol's proof reads the proofs' exported theorems, not their inner modules (T5, T8). | Lean | soundness, proofs |
| Y11 | Accounting's guarantees and the policy's | Each other's definitions, both ways: `security/Definitions/Accounting/Guarantees/Actual.lean:5-6` imports `Definitions.Nci.Reads` and `Nci.Assumptions`, and `security/Definitions/Nci/Guarantees/FitRate.lean:3-4` imports `Definitions.Accounting.Actual` and `Physical`. [read by me; found by SWEEP-ABOVE 5b] | reads | The policy sits on accounting [page]. An accounting guarantee reads no policy definition; what joins the two is compose's or the slowdown bound's. | Lean. Not Daniel's: soundness struck it, since the joined guarantee changes area and owner and keeps its statement [relayed, soundness, 1:07 AM PST]. | soundness, training, network-accounting |
| L1 | The interface's profile | The draw rule, a second time: `interface/CoreProtocol/Profile.lean:70-71` checks a stratum's `k` against `Flock.Draw.workK`. The stratum's size, `k`, work and floor are already data in the profile; the rule is defined in the base package `core_commitments` (`verity/commitments/lean/Commitments/Draw.lean:118`), not in the proof system's, and the same comparison is at `Commitments/Draw.lean:326`; the Python twin is `spot/verifier/profile.py:689`. [read by me on `053b8e50fe8a`; the first cut said the profile "sizes a stratum" with the proof system's rule, which proofs corrected] | reads | The profile states what was established. The rule's check is SPOT's (lean 3 [relayed]). | Lean: drop the comparison at `Profile.lean:70-71` and its Python twin, and move `SamplingDesign.ofFlock` to its one caller. The profile's reader then accepts a work design whose `k` is not the rule's, so a reader that bounds anything from a work profile uses each stratum's own `k` or checks the rule itself [relayed, proofs; whether the certifier's reader relies on the refusal: not read by proofs or by me]. Every guarantee's record stays equal and one read definition changes, `Core.WorkStratum.problem`, in two locks; Daniel sees that one line at landing. Not Daniel's to decide [relayed, lean, soundness]. No hour yet [relayed, proofs]. | proofs, soundness, lean |
| L2 | The interface | Two modules only SPOT reads: `Record.lean` and `Live.lean`. [read by me] | job | They go with SPOT, or with the channel (lean 4 [relayed]). | move | proofs |

### 2.7 The adapter, the demos and the benchmarks

| id | Who | What it knows, and the evidence | Kind | Clean | Fix | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | The adapter | The verifiers and devices of five protocols (PoUW, SPOT, the certifier, PoUS, PoUB). Import lines under `adapters/vllm/` in files that are not tests: PoUW's verifier 91 (37 files), SPOT's verifier 36 (24), SPOT's device 22 (13), the certifier's verifier 18 (14), PoUS's verifier 17, PoUB's verifier 11 (6), PoUS's device 3; 198 lines in all. And two provers, as rule 6 intends: PoUW's 45 lines (23 files) and the certifier's 3. [counted by me on `053b8e50fe8a`; the first cut said six protocols, as vllm found] Such as `adapters/vllm/verity_vllm/pipeline/serve_pearl.py:130-131`. The builder of a served window's circuit reads SPOT's record, the certifier's advice and PoUW's schemes (`program/predict/window_weights.py:29`, `steps.py:122`, `window_class.py:39`; circuits 4 [relayed]). | import | Rule 6: one call into a protocol's prover over GPU buffers. The adapter hands a circuit, its pieces and committed values, and is handed roots. | code (vllm's first change is Sun 15 Nov [relayed]). | vllm, each accounting lead |
| A2 | The adapter | Protocol code: `adapters/vllm/verity_vllm/protocols/pouw/` is 54 files and 17,241 lines [counted by me on `053b8e50fe8a`]; `protocols/pous.py` and `protocols/remote_bandwidth/` sit beside it. | job | PoUW's prover, its kernels and the one call are one package under PoUW [page, section 10]. | move and code | compute-accounting, vllm, kernels |
| A3 | The adapter | A directory named `check/` that no verifier reads: `adapters/vllm/verity_vllm/check/`, 83 files, 22,409 lines [counted by me on `053b8e50fe8a`]. No verifier, compose or Lean file reads any of it. 15 files decide something (6,974 lines); each compares committed bytes with a reference the committer holds, and none samples on a live challenge. 22 are the developer's tests and 46 are helpers [relayed, vllm, 1:26 and 1:59 AM PST; its worker read each file's docstring and outline, not the 6,974 lines whole]. The first cut said the adapter "decides accept and refuse" here. By vllm's read it decides nothing an auditor relies on: these are the developer's checks that honest serving passes. | job | The 15 sit with the adapter's completeness evidence, under a name that does not read as a verifier's. Nothing an auditor runs lives in an adapter (5.2). | rename, after the chains. Nothing is deleted on this read [relayed, vllm]. | vllm, circuits |
| A4 | The adapter | The data sheets, the tools and a benchmark: `adapters/vllm/verity_vllm/acquire/served_device.py:41`; two files in `BENCHMARKS_IMPORTERS`; the listing's tie adapters to tools. [read by me] | import | A run reads the data sheets and passes them in (rule 2). Nothing uses a benchmark (rule 9). | code | vllm |
| A5 | The adapter's serving front | The certifier's stamp, by header name: `adapters/vllm/verity_vllm/pipeline/serve_admit.py:11-13` (`x-verity-release`). The three header names are set twice (`serve_admit.py:82`, `certifier/verifier/release.py:35`), and the function is written twice: `release_of` at `release.py:85` and the adapter's own at `serve_admit.py:283`. [read by me on `053b8e50fe8a`; found by vllm] | name | One function gives "the release number of this request". It exists, in the certifier's `release.py`, which is the server's side and neither the device's nor the check's (N4 [relayed, network-accounting]). | code: the adapter's copy goes. | vllm, network-accounting |
| A6 | The adapter's registration | It needs a workload to build, and serving settings live in a benchmark. [relayed, vllm 5, 7] | shape | The registered deployment names its settings and its build as data. | code | vllm |
| B1 | 27 files outside `benchmarks/`, 12 of them not tests | A benchmark's code (`BENCHMARKS_IMPORTERS`, `tests/test_boundaries.py:127-174`) [counted by me on `053b8e50fe8a`; the first cut said 35, as vllm found]. A served run loads PoUW's kernel from `benchmarks/`, and the certifier's device runs from there [page, section 7]. | import | Rule 9. | move (dated on the page). | kernels, compute-accounting, network-accounting, proofs |
| B2 | A test under `verity/` | The adapter: one test imports 7 adapter modules. The same allowlist (`KNOWN`, `tests/test_integration_boundaries.py`) holds 36 more lines that match no import today. [counted by me] | test | Rule 1. | move the test; empty the list. | circuits, top |
| B3 | The verified queue's low-risk rule | Paths, kept by hand: `lowrisk_excludes` (`tools/verity/check/drive.toml:74-80`) has 72 entries. 8 name paths that no longer exist (five under `verity/ml/`, and `verity/core/`, `verity/configuration/wire/`, `verity/hardware/site/`), and none names `verity/tests/`, where chain 1 put the tests of `verity/channel.py` and `verity/device.py`. [read and counted by me on `053b8e50fe8a`; how an entry matches a path was not read] So a change to those tests alone could land on its ready mark without a reader [relayed, ci]. | dup | One listing says which places are trusted (`tests/import_order.toml`). The queue's rule reads it and keeps no list of its own. | code: ci mends the list in one PR after chain 2 [relayed]; then the rule reads the listing (T18). | ci |

That is 91 rows: 9 for SPOT, 8 for Flock and recursion, 10 for circuits and commitments, 15 for the shared files and
registration, 20 for the accounting protocols and the policy, 20 for compose, the interface and `security/`, and 9 for
the adapter and the benchmarks [counted by me]. Five are new in this cut (G7, G8, U7, N8, B3), and one is withdrawn
with its id kept (Y7), so 90 couplings stand.

15 rows carry a fix that goes to Daniel: P1, P2, G2, G4, G5 (its second step), G8, W1, U2, U6, N1, N3 (the charge),
N7, X1, X3 and Y6. Three more go to him only if something not yet read turns out so: F7 (if a deployed guarantee covers
a session under the provisional unit cover), X2 (if SPOT's exported theorem differs from what the leg reads today) and
M4 (if a hash premise changes with the tree). The first cut had 16 and two conditional, which its part 5.4 gave as
"the 16 rows" and its tally as 18. Since then soundness struck G3, U5 and Y11, proofs' table made F7 conditional, and
the leads' answers added G5, G8 and N3. Part 5.4 has each one, with who settled it or what is asked.

8 rows rest on another's word alone (W2, N7, X6, X7, A6, C4, C6 on a lead's; Y10 on a sweep's); for W2 I read only
where its two files are. The other 83 have at least one place I read or counted on `main`, and each says where a part
of it is relayed.

Seven rows came from the two sweeps (P8, P9, M4, S7, Y9, Y10, Y11), and fifteen more carry a sweep's count or place
beside mine. I re-read the line each of those seven rows' judgement rests on, except Y10's. Of the five new rows, four
came from a lead (G7 soundness's, G8 proofs' and soundness's, U7 memory-accounting's, N8 network-accounting's) and one
from ci (B3); I read each one's cited lines on `053b8e50fe8a`, except the parts of G7 that row marks.

Two patterns run through the rows, and each has one answer:

- **A layer holds another layer's thing so that readers reach it without a new requirement** (F1, F2, G1, L1, N2,
  M1, U7). The answer is always the same: the thing lives with its owner, and a reader is handed the result (a verdict, a
  root, a digest), not the thing.
- **One fact is stated in several layers, each sized to one user** (C2, G4, G5, S4, Y5, B3). The answer: one owner states
  it, and every other layer takes it as a parameter.

The leads' answers add a third:

- **A check takes from the run's own files a thing only the auditor may fix** (G7, N7, U6, G4): a device's key, a
  link, the pin a weight's root is compared with. The answer: the registration or the auditor's configuration holds
  it, and the check's caller hands it in.

## Part 3. The five that matter most

Each is a group of rows with one cause. For each: what is wrong, in words a reader outside the field can follow; the
smallest change that removes it; and what that buys.

### 3.1 SPOT's record of a window is the hub of the system, and it carries its callers' facts (P1, F1, N1, P5, P8, P9, X1, A1)

SPOT is the part that checks a random sample of a computation and bounds how much of the rest can be wrong. By
Daniel's definition its public input is a circuit and a sampling rule, and nothing else. On `main` its input is a
record of 13 keys (14 under v2), and its check holds that record to 24 parameters it is handed [read by me on
`053b8e50fe8a`; the first cut called the 24 the record's fields, which proofs corrected]. At least seven of the 24, and
six facts inside the record's `window` key, say things only a caller can mean: which day's circuit was declared, who
the prover is, a salt for PoUW's noise, a pool of earlier windows with its weekly reset. Because the
record holds everyone's facts, everyone reads it: PoUW's check, the certifier's period check (whose commitment is a
list of these records), the policy's check, compose's remote check, the adapter in 24 files, and Flock's verifier,
which holds the record's format so that the others can reach it. The knowledge runs the other way too: SPOT's own
parties file lists every protocol compose runs, by id. So a change to one caller's fact changes SPOT's verifier, and
SPOT's theorem is about a record and not about a circuit and a rule.

- **The smallest change:** split the record in two. SPOT's part holds the circuit's digest, its pieces, the sampling
  rule and the committed roots with their domains. The caller's part is a second record in the caller's package: it
  names SPOT's part by its digest and adds the caller's facts with the caller's checks. No check is dropped. About nine
  check codes change package (R2's declared day, R6's kind, stream, prover, salt and pool, R7's look back).
- **Where the caller's part goes** [relayed, proofs, 2:39 AM PST]: the caller exists already. SPOT's check is called
  on both of compose's lists through one thin protocol, `VStar.Recursive.Sampled.protocol`, which sits in recursion's
  package (P6). Its package moves whole above SPOT and recursion and takes the caller's checks.
- **What it costs in lock records:** the move of the package changes none [relayed, lean]. Moving the declared day
  out of R2 changes two listed statements (`AcceptMeans`, and `AcceptInferenceOnly` through `StageHeld`); moving R6's
  and R7's facts changes one (`AcceptMeans`) [relayed, proofs]. Splitting the certifier's commitment (N1) changes its
  digest [not priced].
- **What it buys:** SPOT's code and its theorem match Protocol 1. A new protocol that calls SPOT changes nothing in
  SPOT. The certifier, PoUW and compose compare roots and digests, and read no field of a record. The format string
  leaves Flock.

### 3.2 No accounting protocol's answer says which registered thing it is about (G2, U6, N7, X3, G4, G5, G6, G7, G8)

Registration is where the developer says, once, what hardware it has, which circuit it runs and which weights. Each
accounting protocol then measures one part of that hardware. On `main` no protocol reads what registration accepted.
PoUW keeps its own table of a card's figures. PoUS and PoUB take the memory's size and read rate from their own
parameters, and PoUB's v0 form recomputes the store's root from the model's checkpoint. The certifier's period names one link
and one key in its own parameters. The weights are committed at least three times with nothing tying the roots. And no
answer names the registration it was measured against. So compose cannot check that the compute, the memory and the
network it adds up belong to the same registered devices, or that every registered device was measured at all. The
three open ties in the security model (the circuit, the link, the memory leaf) are this one gap, seen three times.

The leads' answers widen it three ways [relayed; G7's and G8's cited lines read by me]. A device's key is
registered, and no period check reads it: four checks take it from files the prover's run writes (G7). The
registration has no read rate for memory under a server and no figure for the server's CPU, so a part no answer covers
has no number to be charged at (G8).
And a vLLM serve passes no committed weights, so a served proof opens no registered root (G4).

- **The smallest change:** every accounting protocol's answer gains the registration's head (64 bytes) and the part
  it accounts for, with the period's span and the key of the auditor's device it rests on (soundness's form
  [relayed]). The protocol is handed that part's figures from the registration, not from its own files. Compose
  compares heads and keys. A registered part with no answer is charged at its registered capacity, never at 0
  (soundness's default [relayed]); until the registration records that capacity (G8), a figure says "unbounded" for
  such a part.
- **What it costs in lock records:** the ninth field of a site entry changes the location type's hash in the locks and
  every site entry's digest, so every registration's head, in one format bump [relayed, soundness, proofs]. The two
  fields of an answer change each published answer's statement [not priced].
- **What it buys:** the account means "of this registered hardware". A device's figures have one source. The three
  ties become equalities of digests inside compose's theorem and need no premise.

### 3.3 The auditor's devices do each other's jobs (P2, P9, W1, U2, N5, N4, X5)

The auditor has three kinds of device. The challenger draws challenges. The network certifier sits on a link, and
counts and releases traffic. The memory challenger times answers from memory. Each is trusted for one thing. On
`main` SPOT's challenger also keeps the auditor's ledger: it signs a period's open and close, a day's declared
circuit, and which window counts toward which period; 46 code files outside the proofs import it, and PoUW's Lean
check opens the ledger's entries itself. The network certifier's device also draws the salt that PoUW's
noise is seeded from, and keeps the root of PoUW's weights. The memory challenger receives 24,576 blocks of the store
a session, in the clear, and keeps them, though nothing reads the kept copy [relayed, memory-accounting]. So what one
must trust a device for is wider than its name, and one protocol's guarantee rests on another protocol's device.

- **The smallest change:** one device, one job, and three kinds of signed message in the shared channel: a challenge,
  a count, a declaration. The period's ledger moves to compose. A day's declaration takes registration's form and
  registration's one-a-day check, and the challenger still signs it, handed as a digest, so there is one signer and one
  clock (proofs' recommendation [relayed]; point 2 of 5.4). PoUW is handed "a live salt from one of the auditor's
  devices", drawn after the commitments it seeds, and compose says which device. The memory challenger keeps roots,
  drawn indexes and timings once the opening's check is a proof the prover makes; that part is not dated.
- **What it buys:** each device's trust is one named assumption with a stated scope. The security model can take the
  devices as parties with one kind of message each. PoUW runs where no certifier stands, with no code that names one.

### 3.4 The common ground holds single layers' things (G1, F2, Y1, Y4, Y5, Y9, L1, N2, M1, M4, S7)

Two things are meant to be common ground: the interface package, which says what a protocol's check is, and `Core`,
the vocabulary every guarantee reads. On `main` the interface package also holds the registered hardware, the
registration, compose's account and the period: 2,245 of its 3,717 lines. Flock's Lean package also holds recursion
and the ZK server, and 14 packages require it, several only to reach a hash or a byte format. `Core` reads three
structures of an accounting module and holds 84 files, four of which one area alone reads (compose's, recursion's and
two of PoUW's).
Because the common ground holds particular things, a second definition appears whenever a layer needs its own view:
four data sheets, two resources, compose's lists four times, the period's type three times, a Merkle tree of PoUS's
own beside the shared one. One table in commitments holds every protocol's format strings, the adapter's among them.
And names leak: Flock's byte lemmas are declared under the certifier's name.

- **The smallest change:** three moves, and no new statement. Registration and the account leave the interface
  package: three modules, 1,404 lines, to a package `registration_check`, in four PRs; the registered hardware's and
  the period's files stay at first [relayed, proofs]. Recursion and the ZK server leave Flock's package. `Core`'s four
  files that one area owns go to that area, and the three structures it reads of accounting's move into it. Then one
  rule on each: the interface requires only the base packages and the difftest's Lean, and `Core` imports no area.
- **What it costs in lock records** [relayed, lean, 2:01 AM PST]: no statement changes in any of the three. Splitting
  Flock's package moves its 11 records to new locks by hand, since `--moved` carries none between packages, and 21 of
  the 36 packages are audited again. Renaming the 48 imports that reach a base package through Flock's name changes no
  record, and frees one package only. The three structures move with no record changed and 3 of 36 audited again.
- **What it buys:** a change to registration rebuilds no proof system's package. Each duplicate has one home to merge
  into. The first cut added "a lock's records follow the layer that owns them". That holds for 15 of the 1,692
  guarantees: the other 1,677 are in `security`'s two locks whatever their area [counted by me on `053b8e50fe8a` in
  the 36 locks outside test fixtures; found by lean, which counted 1,688 one commit earlier].

### 3.5 The adapter holds protocol code and reads verifiers and devices, and protocols name the serving system (A1, A2, A3, N3, U1, F3, C1, C2)

An adapter is meant to make one serving system speak Verity, by one call for each protocol, and to know nothing of
how a protocol samples or decides. On `main` the vLLM adapter imports the verifiers and devices of five protocols in
198 lines (PoUW's verifier in 37 files, SPOT's device in 13), holds 17,241 lines of PoUW's serving code, and has a
directory named `check/` of 22,409 lines [counted by me on `053b8e50fe8a`]. No verifier, compose or Lean file reads
that directory: its 15 deciding files are the developer's own checks that honest serving passes [relayed, vllm]. So
the first cut's "lines that decide accept and refuse" was wrong, and what is left of that point is a name. In
the other direction, the certifier's advice format is defined as one pinned vLLM build's scheduler output, PoUB's v0
verifier knows vLLM's layer order, Flock's verifier holds a table of model names, and the cut of a circuit into pieces
is stated in the circuits, in the adapter and in the checker.

- **The smallest change:** for each protocol, the one call its prover offers (the page states PoUW's and memory's).
  The compiler and the advice format move to the adapter (seven files, 5,159 lines), and the certifier is handed a
  registered compiler circuit and advice as bits. How the advice is charged is a statement, and goes to Daniel (N3).
  The cut is one cutter and one predicate, with the caps as the predicate's parameters (C2).
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
| T3 | An accounting protocol uses no other, but PoUB on PoUS | `loads` of pouw, pous and certifier names no other accounting protocol and not the policy. The same for tests: a test under one protocol imports no other protocol. | Four test files: the certifier's on PoUW (`certifier/verifier/network/tests/period_case.py:82`), PoUW's on the certifier (`pouw/verifier/tests/test_pouw_salt.py:27-31`), SPOT's on PoUW, PoUB's on PoUW [read by me]. No other file today. After the group map 21 loads cross the three groups: 11 lines from the proofs to accounting, all in tests; 1 from the proofs to the policy; 9 from accounting to the policy, eight a test's load by path and one a Lake require (N2) [relayed, GROUPS, on its rehearsed head]. |
| T4 | A layer names nothing above it | One table, `MAY_NOT_NAME`, beside `SPOT_NOT_CERTIFIER` (`tests/test_boundaries.py:101-111`): for each layer, the whole words its files may not hold, in Python and Lean, comments and docstrings included. Circuits and commitments: every protocol's name, "window", "request". The proofs: pouw, pous, poub, certifier, nci, pool, lane, week, served, vllm. Each accounting protocol: the others' names. All of `verity/`: vllm, torch and model names (today's test of rule 1 reads imports only). Each layer and word starts at today's count. | P4, F5, C1, C5, N3, U1, W1 |
| T5 | A package is used through its entry | Each protocol declares its entry modules in one table in `tests/test_boundaries.py`, which already reads every Lean import and every `require` without a build. A file outside the package that imports any other module of it fails. Not in a lock, as the first cut wrote it: Lake cannot hold an entry, an `entry` list in a lock is read by nothing today, and the code to read it would have all 36 packages audited again, where editing the table changes no audit key [relayed, lean, proofs]. | P5 (for SPOT: 11 Lean modules in 10 files under `verity/`, 25 in 44 under `security/`, 241 Python lines), P9 (46 code files on the challenger), Y10 |
| T6 | A format string has one owner | Every literal of the form `verity/.../vN` or `verity-.../vN` is defined once, in the directory of the layer its first part names. `verity/commitments/` defines none of a protocol's. All are distinct across the tree. | M1 (341 constants in one table) |
| T7 | Lean packages follow the layers | A table `LEAN_REQUIRES` beside `REQUIRES_FLOCK_VERIFIER` and `BASE_LEAN`, held over imports as well as `require` lines: five packages import Flock's modules through a package they require, which a table over `require` lines misses [relayed, lean]. The interface requires the base packages and the difftest's Lean only. Flock's verifier requires the base packages only. SPOT requires Flock and the interface. An accounting check requires the interface, SPOT, registration and its own packages. None requires another protocol's, and no check requires a package under a `device/`. Compose requires all. `REQUIRES_FLOCK_VERIFIER` shrinks to the packages that verify a Flock proof. | N2, I1, P6, F2 (14 packages by `require`, five more by import), S7 (a device requires a package under the hardware directory), U7, N4 (a check requires a path under `device/`). As the first cut worded it, today's tree failed it twice more: the interface and SPOT require the difftest's Lean, and SPOT requires `lms` [relayed, lean]; the first is now allowed, and the second is listed. |
| T8 | The areas of `security/` | In `security/lean-audit.json`'s `layers`: `Definitions.Core` imports no area; `Definitions.Spot` imports `Core` only (it holds today); a protocol's area imports no other protocol's; the composition area imports a protocol's `Guarantees` and the vocabulary its statements read in the definitions and properties trees, and in the proofs tree only its `<Area>.SecurityProofs.G` theorems, never an inner lemma; a check's agreement lemma is an exported property and may be read [relayed, soundness]. `trees.except` only shrinks: its cap is 41 against 28 entries today (`tests/test_lean_packages.py:388` [read by me on `053b8e50fe8a`]), so 13 could be added before it fails; lean lowers it to 28 in its first Lean change after the second chain [relayed]. | Y1, Y2 (28), Y3, Y9, X2 (the `Sampled` tree, 76 files). U5 and Y11 leave the list when their joined statements move to the composition area. |
| T9 | One definition of a shared concept | `tests/test_lean_vocabulary.py` gains the names `Sheet`, `Tier`, `Resource` and `Part`, `Audit` and `Subprotocol`: each defined once outside `Proofs`. `Window` and `Period` are in the test already (`:66-67` [read by me on `053b8e50fe8a`; found by lean]). | G5, Y5, S4 |
| T10 | An accounting answer names the registration and the device's key | For each accounting protocol on one of compose's lists, its declared account has a key for the head, the part, the period's span and the auditor's device's key (soundness's form [relayed]). Case files: compose refuses an answer that names another head, and one whose device's key is not the registration's. In Lean, one lemma, `NamesRegistered f`, for every such tie. | G2, U6, N7, G7, X3 (no protocol passes today) |
| T11 | A device does one job | A file under a `device/` imports no other protocol's `device/` or `verifier/`. A `verifier/` holds no socket or process code, and a `device/` holds no check a verifier requires. The kinds of message a device signs are declared in `channel.py`, and a device signs only its own. | P2, P9, W1, N4, N5, X5, U7 |
| T12 | The adapter makes one call | Under `adapters/`, an import of a protocol is allowed only for that protocol's declared workload module, and never of a `verifier/`, a `device/`, `verity.hardware`, `tools` or `benchmarks`. | A1 (198 import lines), A4 |
| T13 | No code reads `security/` or the data sheets, and no Python writes under `security/` | A lint on path strings (`security/`, `hardware/cards`) in code under `verity/`, `adapters/` and `benchmarks/`. | U4, G5, Y6, Y8 |
| T14 | The runtime | No file under `verity/protocols/` or `adapters/` imports the runtime, and no `verifier/` loads it. A row of T1 once the types move. | S1 (9 files) |
| T15 | The configuration names no protocol | `config.py` holds no protocol's id as a literal, and no file outside compose holds a table from a list's names to protocol ids. Whether a list is acceptable is asked of compose's Lean. | S2, P8, G3 |
| T16 | The cut is stated once | No table of cuts outside the cutter's module. The caps appear only as the parameters of the one predicate the verifier holds, and one test holds Python's three caps to Lean's (`flock/verifier/Flock/Statement.lean:117-121`), which nothing does today [relayed, proofs, circuits]. | C2 (three places; `K_MAX` stated a second time) |
| T17 | A device's key is pinned by the auditor | No driver or check under `verity/`, `benchmarks/` or `demos/` reads a device's public key from a run's directory. A check is handed the key by its caller, from the auditor's configuration or the registration, and each check has one case file: a signature under another key is refused. | G7 (the certifier's period scan, PoUW's challenger, PoUS's memory challenger, the wipe; SPOT's one-stage check run alone), N7 |
| T18 | The queue's low-risk rule follows the listing | Every path of a trusted place in `tests/import_order.toml` is matched by the queue's rule, and the rule names no path that does not exist; best, the rule reads the listing and `drive.toml` keeps no list. | B3 (8 dead entries; `verity/tests/` unnamed) |

## Part 5. What I am unsure of, what is answered, and what is left for Daniel

### 5.1 What I am unsure of

1. **Who calls SPOT on a remote run.** Answered in fact, open as a decision. Today SPOT's check is called on both of
   compose's lists through one protocol, `VStar.Recursive.Sampled.protocol` [read by me]; compose builds none of
   SPOT's input, and the caller's facts are checked inside SPOT [relayed, proofs]. Whether that protocol becomes the
   thin caller is point 1 of 5.4.
2. **Whether recursion may require SPOT** (P6). Answered: the outer proofs' statements are Flock's, so recursion needs
   no SPOT. Its package requires SPOT's only because the protocol compose lists sits in it [relayed, proofs]. I read
   that package's Lake file and two of its four files.
3. **The split of the check's 24 parameters** (P1). The first cut called them the record's fields and split them 10
   and 14. They are parameters, and I withdraw the split. By proofs' read, `registered`, `params`, `schemas` and
   `members` are SPOT's own, and `tied` is a caller's whose tie is in no theorem [relayed]. Which caller fills the
   anchors (R5): not read by proofs or by me.
4. **Which theorem cites R2, R6 or R7.** The cost of point 1 rests on proofs' worker's read: one statement outside
   SPOT's cites R2, none cites R6 or R7 [relayed]. A search of the three names on `053b8e50fe8a` agrees [counted by
   me]. A search finds a name, not a use through another definition, and I read no proof.
5. **Whose the period is** (S4): compose's, beside the account, or a shared file's. `tests/tree.toml` sends
   `reporting.py` "into verity/protocol.py" [read by me]. No lead answered this.
6. **Names against meaning.** A word count finds names, and a name in a docstring costs less than a field or an
   import. I ranked by fields and imports first, and T4 would still hold the names.
7. **My counts.** vllm found nine of the first cut's counts wrong, most because they counted tests where the rule
   said not to. I recounted each on `053b8e50fe8a` and the rows give both figures where they still differ (P5: 241
   lines by me, 238 by vllm; F5 and N3 by a few files). Counts I did not recount are the first cut's.
8. **Lean's prices.** Every figure of the form "N of the 36 packages are audited again" is lean's [relayed]. In the
   locks I counted four things myself: 36 locks outside test fixtures; the 11 guarantees and 2 open premises of
   Flock's verifier's; 15 guarantees of 1,692 outside `security`'s two locks; and `trees.except`'s 28 entries.
9. **The device keys** (G7). I read the certifier's path and PoUS's served case. PoUW's challenger, the wipe and
   SPOT's one-stage check are soundness's and proofs' read [relayed].

### 5.2 Two questions the leads put to this document, and their replies

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
  `adapters/vllm/verity_vllm/program/predict/weights_tie.py:1-12` [read by me]): the first cut called this
  registration's proof, made once. soundness words it otherwise: that the registered rows are a checkpoint's is
  completeness, the developer's to validate, never proved and never a gap; every guarantee is about the registered
  weights, by their commitment [relayed, soundness, 2:00 AM PST]. I take that wording.
- **What the choice costs.** SPOT's input gains, for each committed value, a root and a domain, and nothing in the
  proof. The builder cannot read at a private position without a gate of commitments', and pays its gates. The model's
  circuit holds no commitment scheme except at those reads, so a change of scheme changes the circuit's digest only
  there. Daniel's rule holds in all three: a value is touched only inside a proof.

circuits' reply [relayed, 1:17 AM PST]: the three cases are right for the embedding's reads, and three things are
missing.

- **The second case is a circuit and a rule.** A Boolean circuit has no assertion, so the gate returns a verdict bit.
  SPOT's input gains, for each such read, a reader, its key port, its opening's ports, its verdict port and the table,
  not only a root and a domain. Without the rule the circuit proves nothing about the registered table.
- **A fourth case.** Sparse attention's history reads open a table that no registration holds: a request's own rows
  from earlier windows. The verifier computes that root from the windows' commitments.
- **The third case holds only for a table in the checkpoint's own row order.** For the experts' tables, that the
  registered rows are the checkpoint's is the committer's own check, which leaves no field in the record [relayed,
  circuits, vllm]. By soundness's wording above that is completeness too.
- So registration needs, from the builder, the list of tables and their shapes as data. Today that list is a name
  format the adapter's builder and the adapter's registration share.

**vllm: which of the adapter's `check/` is a verifier's work, and which is the developer's own test?** Proposed: one
question sorts each file. Does a verifier, compose or a published answer read its accept or refuse?

- **If yes**, it is a protocol's check. It goes to that protocol's `verifier/`, and it must be Lean, since Python never
  returns accept or reject [page].
- **If no**, it is the developer's own test that the circuit computes what the served system computes. That is
  completeness: it decides only whether honest serving passes, and nobody proves it (`AGENTS.md`). It stays in the
  adapter, under a name that does not read as a verdict.

vllm's reply [relayed, 1:26 and 1:59 AM PST]: every file falls on the second side. Of 83 files and 22,409 lines, a
verifier, compose or Lean reads 0. 15 files decide something (6,974 lines): 14 for the offline Commit and one a
retired scheme's replay. Each compares committed bytes with a reference the committer holds. 22 are the developer's
tests and 46 are helpers. Nothing is deleted on this read. Row A3 is rewritten to say so.

### 5.3 What I did not read

- Flock's Rust prover, and the two-stage form's Python beyond the lines cited.
- Recursion's packages, beyond `Sampled.lean`, the Lake files and line counts. The ZK server's code.
- The certifier's device program (`NetworkCertifier`), and PoUS's and PoUB's Lean beyond the lines cited.
- `security/Proofs`, beyond counts. `security/Properties/Verity/Game.lean` beyond the lines cited. Any theorem's
  proof, so every "changes N statements" is a lead's.
- `demos/` and `benchmarks/`, beyond their import lists. The reverse direction: what they import of the adapter's
  insides.
- The draft of registration's statements (#9701) and every other open pull request the leads name (#9649, #9687,
  #9607, #9593), none of which is on `main`.
- Of the two sweeps: the appendix of SWEEP-PROOFS (its list of 143 format ids). The three findings of theirs that the
  first cut left out are rows now: a verifier that requires a device's package (U7), the devices' keys under several
  signer types (G7), and the registration log's third kind of entry, a day's declared circuit (P2 and part 1).
- Of the leads' answers: I read all of `inbox/layers-leads.md` at `b49ee9943259`, lines 1 to 286 before the first cut
  and the rest for this one. I did not open the Linear issues, the Notion pages or the worker reports they cite.
- How `drive.toml`'s `lowrisk_excludes` matches a path (B3).
- Two small couplings compute-research offered as rows and I did not add or read: a machine-code check loaded by file
  path, and a harness that reads the `sass_gate` key.
- What changed on `main` between `9182900fb94e` and `053b8e50fe8a` beyond the paths and the lines this document
  cites.

### 5.4 What is answered, and what is left for Daniel

**Every question the first cut put to a lead is answered** [all relayed; the row named carries the answer]:

- **proofs** (2:39 and 2:42 AM PST): which parameters the proofs read and who cites R2, R6, R7 (P1); the outer
  statements are Flock's and the package moves whole (P6); the challenger signs a day's declaration today (P2); SPOT
  states four entries (P5); Flock holds the Program to the circuit on every path SPOT accepts (F7); a protocol on the
  list already calls SPOT (point 1 below); the cap is three numbers (C2).
- **soundness** (1:07 to 2:25 AM PST): the form of an answer, with two more fields (part 1, G2); one lemma,
  `NamesRegistered f` (X3, T10); what the composition area may import (T8); which rows are not Daniel's (below).
- **lean** (2:01 to 2:42 AM PST): Lake cannot hold an entry and a table can (T5); what splitting Flock's package
  costs (F2); the three structures `Core` reads (Y1); leave the `Flock` namespace (F8).
- **compute-accounting** (1:07 AM PST): what PoUW needs of its salt (W1); what the check reads of the models' list
  (G6); the FP4 variant is the coupling (W3).
- **memory-accounting** (1:13 and 1:57 AM PST): the leaf each answer names (U6); PoUB is handed PoUS's setup answer,
  not a registered root (U1, U3); the memory challenger need see nothing of the store (U2).
- **network-accounting** (1:17 AM PST): five fields read, roots are enough with a domain and a port's name (N1);
  about 70 lines of `Desc` (N2); three things stay and the charge is a statement (N3); four device files (N4).
- **vllm** (1:26 to 2:43 AM PST): no file of `check/` is read by a verifier (A3, 5.2); roots are not enough for the
  committer (G4). **circuits** (1:17 and 2:55 AM PST): 5.2; the cutter and the predicate (C2); the lowering (C3).
  **compute-research** (1:15 AM PST): the registration's `Sheet` stays and grows (G5). **training** (1:11 AM PST):
  five figures (I2).

One question stays with no answer: whose the period is (5.1, item 5; S4).

**The count.** The first cut's tally said 18 rows go to Daniel (16, and two on a condition) and its part 5.4 said
"the 16 rows". Both meant the same rows. Now it is 15, and three on a condition.

**No longer Daniel's** [relayed]:

| Row | Who settled it, and when | Why |
| --- | --- | --- |
| G3 | soundness, 1:07 AM PST | Restated over the run's list, it says the same of the two lists that exist. |
| U5 | soundness, 1:07 AM PST | The wipe's joined statement moves to the composition area, word for word. |
| Y11 | soundness, 1:07 AM PST | The joined guarantee changes area and owner, and keeps its statement. |
| F7 | proofs, 2:39 AM PST | One open case, Flock's command alone. His only if a deployed guarantee covers a session under the provisional unit cover [not read by proofs or by me]. |
| X2 | soundness, 1:07 AM PST | His only if SPOT's exported theorem differs from what the leg reads today. |
| Y2, L1, Y4 | soundness, 2:02 and 2:25 AM PST; lean, 2:06 AM PST | The first cut did not mark them; lean asked. None changes what a guarantee says. L1's landing shows him one changed definition, `Core.WorkStratum.problem`, in two locks. |

**Two points Daniel can answer yes or no.** Both come from the same split (P1, P2, P6).

**Point 1. Is SPOT called, on every run, through one thin protocol that holds the caller's facts?**

- **Yes means:** `VStar.Recursive.Sampled.protocol`, which is on both of compose's lists today, keeps its id and its
  place. Its package moves above SPOT and recursion. The declared day, the stream, the prover, the seed salt and the
  pool leave SPOT's check for that protocol's. SPOT is handed a circuit, its pieces, a sampling rule and roots.
  Compose stays generic over its list.
- **No means:** compose builds SPOT's input and checks those facts itself, so compose knows one protocol's insides.
- **Recommended: yes** (proofs, 2:39 AM PST [relayed]; it is also what the first cut leaned to).
- **Cost in lock records:** moving the package whole changes 0 records [relayed, lean, 2:42 AM PST]. Moving the
  declared day changes 2 listed statements, `AcceptMeans` and `AcceptInferenceOnly` (through `StageHeld`). Moving the
  stream, the prover, the seed salt and the pool changes 1, `AcceptMeans` [relayed, proofs; a search of the names on
  `053b8e50fe8a` agrees, counted by me]. How many packages are audited again: not priced.

**Point 2. Does the challenger stay the one device that signs a day's declaration?**

- **Yes means:** the declaration's form and its one-a-day check become registration's. The challenger is handed the
  declaration as a digest and signs it with its time, as it signs a window's open. One signer, one clock.
- **No means:** registration gets a device of its own: one more of the auditor's devices, a second clock, and a
  premise that ties the two clocks.
- **Recommended: yes** (proofs, 2:39 AM PST [relayed]). The first cut's row P2 said "signed by registration's
  device", which was the "no". I take proofs' recommendation.
- **Cost in lock records:** none beyond point 1's two statements, since the declared-day check leaves R2 either way
  [my reading of proofs' answer; not priced by lean]. A "no" adds a device's key to every registration and a named
  assumption [not priced].

**The other rows still Daniel's**, each with its recommendation and its cost in lock records. "Not priced" means no
lead gave a count of records and I read no lock.

| Rows | What is asked | Recommended, and by whom [relayed] | Cost in lock records |
| --- | --- | --- | --- |
| G2, U6, N7 | Each accounting answer names the registration's head, its part, the period's span and the auditor's device's key. | Yes: soundness's form; compute-accounting and memory-accounting took the names `registration_head` and `place`. | Each published answer's statement changes. Not priced. |
| G8 | A site entry records every capacity its data sheet states: a read rate for each memory leaf, and the server's CPU's rate. | Yes: soundness; memory-accounting agrees. The memory half is option A of memory-accounting's page, open to Daniel until Mon 16 Nov 6:00 PM PST. | The location type's hash changes in the locks, and every site entry's digest, so every registration's head. One format bump, a fifth PR after registration's split [proofs]. |
| G4 | One commitment to the weights, in the registration; a served proof reads the registered rows; the pin comes from the record. | Yes: vllm's VLLM-424, undated until Sat 14 Nov noon PST. Until then a line may say "registered" and not "proved over the registered weights" [soundness]. | Not priced. |
| G5 | PoUW's work statements read a card off the registration, where they hold at any card today. | Yes: compute-research; compute-accounting agrees, with the sheet carrying both clocks. He is not asked to pick a clock: a measurement due Tue 17 Nov settles the H200's. | Not priced. |
| W1 | PoUW's noise seeds rest on a named device's challenge, drawn after the commitments, not on the certifier's device. | Yes: compute-accounting, network-accounting. | Not priced. |
| U2 | The memory challenger sees nothing of the store; the opening's check becomes a proof the prover makes. | Yes: memory-accounting [argued, not proved or sized]. Its ask on a related point, which part checks what was registered at setup (ASK-300), is with Daniel, with a default on Mon 16 Nov 6:00 PM PST. | Not priced. |
| N1 | The certifier commits to roots, each with its domain and port's name, not to SPOT's records. | Yes: network-accounting. It follows from point 1. | The commitment's form changes, so its digest. Not priced. |
| N3 | How advice is charged: the committed advice becomes the coded string and the charge is its committed length, or the code stays in Verity as the one registered code. | The first: network-accounting leans to it, and it is `AGENTS.md`'s rule as written. | The charge's definition (`AdviceV1.lean`) and the guarantee that counts by it change. Not priced. |
| X1 | Compose compares heads and roots and reads no field of a protocol's record. | Yes: this document's proposal. soundness keeps it on his list and gave no recommendation. | Not priced. |
| X3 | The tie of SPOT's circuit to the registered one becomes a fact of compose's theorem (`NamesRegistered f`, with `f` the layout). | Yes: soundness. | A new tie. Not priced. |
| Y6 | The four per-card guarantees and their four witnesses are retired once the guarantee at every card lands. | Yes: compute-research. | 4 guarantees' records and 4 witnesses leave [compute-research]. |

On a condition only: F7 and X2 (the table above), and M4 if a guarantee's hash premise changes when PoUS's own tree
gives way to the shared one [not read by any lead].
