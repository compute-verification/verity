# What each layer of Verity may know, and every coupling on `main` that breaks it

**The layers are held apart by import rules only, and three things leak through them: SPOT's record of a window, which
every protocol and the adapter read and which carries its callers' facts; the registration, which no accounting
protocol reads, so each restates what it needs; and the adapter, which holds protocol code and reads verifiers and
devices. Part 2 lists 62 couplings. Part 3 names the five that matter most.**

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

End of the first cut.
