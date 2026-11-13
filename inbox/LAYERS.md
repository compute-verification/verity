# LAYERS: what each layer of Verity may know, and every place the code breaks it

Read `inbox/worker-rules.md` on `cursor/verity-top-4db4` first. This brief is for three workers. Each reads all of it
and does its own part: **LAYERS** (the design, section 3), **SWEEP-PROOFS** and **SWEEP-ABOVE** (the evidence,
section 4). Nobody changes code, a test, a lock or a `.lean` file, and nobody opens a pull request.

## 1. What Daniel asked (Fri 13 Nov, 12:09 AM PST)

top had told him: "The tie between the declared circuit and the circuit SPOT is handed now belongs where a protocol
hands it over, not inside SPOT. soundness found tonight that nothing states that tie yet (draft #9701)." His answer:

> yeah this is super key -- these are just different layers of the system entirely -- Just to make sure there's no
> confusion, it's definitely worth having someone think very carefully about what the neat separation of concerns is
> between this current mess of things. Because, yeah, this currently smells to me. It seems like we haven't properly
> decoupled the different layers of the system here. So I wouldn't be surprised if there's other tight couplings and
> some components that. You know, currently know about each other that shouldn't know about each other or something
> like that.

Earlier the same night (11:55 PM PST): "SPOT doesn't require a notion of registration… Not sure why it thinks that…
e.g. take a look at SPOT security definitions." And (11:58 PM PST): registration is a protocol and lives inside
accounting.

So the question is: **for each layer, what is it handed, what does it hand back, and what must it not know; and
where does the code, the Lean or a statement know more than that today.**

## 2. What you read

- `inbox/ARCHFILE-page.md` on `cursor/verity-top-4db4`: sections 1 to 7 of the architecture page Daniel agreed to,
  as it reads now. Sections 2, 4, 5 and 6 state the layers, each part's interface, the dependency diagrams, and the
  tree with its eleven rules. `inbox/LAYERS-page-s10.md` is its section 10, the boundaries the leads settled.
- Daniel's Notion page "SPOT security definitions" (id `41339951-5d9e-8162-bbb7-f8de48ca23e4`; read it with the
  Notion connector if your machine has one; it is not copied into Git). What matters from it, quoted: "**Protocol 1.**
  The public input is C and a sampling design D. The prover holds a trace." And its section 4: "The limits hold for
  one fixed C, so any influence the prover has over C must be accounted for separately. They also concern the input
  and output nodes of the trace, and applying them to the inputs and outputs of a real system takes a mechanism
  outside SPOT that ties those values to those nodes." This is the model of how clean a layer's interface should be.
  The same statements in Lean are `security/Definitions/Spot/`.
- The tree at `origin/main` (name the commit you read). Tonight two chains of path moves land (`#9697`, then one
  more); cite paths as `main` holds them when you read, and say the commit.
- `tests/import_order.toml` (the import graph by place: `loads`, `reaches`, `loaded_by`), `tests/test_boundaries.py`,
  `tests/test_integration_boundaries.py` (its `KNOWN` list), `tests/test_verifier_cost.py`,
  `security/lean-audit.json` (`layers`, `trees` and their `except` lists, each a known break), every package's
  `lakefile.toml` (`require` by path), `AGENTS.md` (Design rules; "Integrations are frontends"; "One vocabulary").

## 3. LAYERS (the design)

You think; the two sweeps count. Read the code yourself wherever a judgement rests on it.

Write one document, `layers/layers.md` on your branch `cursor/top-layers-3e7b` (from `origin/main`; a scratch branch,
never a pull request), and the same text as your note. It has five parts.

1. **Each layer in four lines:** what it is handed (as data: name the values), what it hands back, what it must not
   know, and who calls it. Use Daniel's two pages' words. The layers, bottom up: circuits; commitments; the shared
   files (party, channel, device, protocol, config); the proofs (Flock, SPOT, recursion, each on its own line);
   registration; the other accounting protocols (PoUW, PoUS, PoUB, the certifier); the policy; compose; the runtime;
   an adapter; a demo and a benchmark. Then the same for the Lean: each protocol's own Lean package, the interface
   package (`verity/protocols/interface/`, `core_protocol`), and the areas of `security/`.
   Where the pages do not say, propose, and mark it "proposed".
2. **Every coupling that breaks part 1,** one row each: an id; the part that knows too much; what it knows; the
   evidence (file and line, up to three places, and a count); its kind (an import; a name or a format string; a
   parameter or a record field shaped for one caller; a device doing another layer's job; a statement or theorem
   that reads another layer's definition; a duplicate definition of one shared concept; a test only); the clean
   interface in its place (who hands what to whom); what the fix is (a path move, a rename, a code change, a change to
   a Lean statement); and the lead who owns it. A fix that changes what a guarantee says or rests on is marked
   "to Daniel".
3. **The five that matter most,** each in one paragraph a reader outside the field can follow, with the smallest
   change that removes it and what it would buy.
4. **The tests that would hold each rule of part 1:** a row of the import listing, a name lint, a Lean import rule.
   Proposals, not code.
5. **What you are unsure of,** and the questions for proofs, soundness, lean and each accounting lead.

Leads to start from (top's reads on `main` `9182900fb94e` tonight, by `rg`; check each yourself):

- **SPOT has its own "registration record"** (`verity/protocols/spot/verifier/lean/Spot/Registration.lean`,
  `spot/verifier/one_stage/registration.py`, format `verity/registration/v1`): the circuit's digest, the partition,
  the sampling rule, the committed roots, and checks R1 to R8. It also carries things that look like the caller's:
  the day's declared circuit (`R2-declared`, `day`, `declarations`), a window's kind (`served` or `stand-in`), a
  `binding` to a stream and a prover's identity, values registered across windows (`pinned`, `held`, `supplied`,
  `outputs`), a pool chained across windows, a `seed_salt`. Which of these are SPOT's by Daniel's definitions (C, its
  blocks, D, the commitment) and which belong to whoever calls SPOT?
- **SPOT's challenger** (`spot/device/challenger.py`) signs each day's declarations (`declared_day`), imports
  `verity.hardware.record`, and takes all fifteen entries of `verity/claims`' `INSTANCES`.
- **The interface package every protocol builds on** (`verity/protocols/interface/CoreProtocol/`) holds the device
  tree (`Site.lean`), the registration log (`Registration.lean`), what a protocol accounts for (`Account.lean`, which
  imports both), periods, and more. Which modules does the proofs' group truly need, and which are the accounting
  layer's?
- **The soundness game reads the circuit's digest from the verifier's parameters** (`Vocab.programDigest`), never
  from the registration's declaration (`Declared.program`), and nothing ties the two (soundness's draft #9701,
  `DeclaredIsParams`). [relayed from soundness]
- **Duplicates soundness's worker found** [relayed]: the data-sheet constants three times (`Site.Sheet`,
  `Accel.Card`, `TS.Sheet`); the resource notion twice (`Accounting.Resource`, `Core.Account.Part`); protocol ids
  twice (`Accounting.Audit`, `Subprotocol`).
- **The registry of assumption ids** (`verity/claims`) is read by five protocol files, by about eighty Lean
  docstrings and by three locks. [relayed from top's worker CLAIMS]
- **Accounting protocols and each other:** AGENTS.md says PoUW's noise salt comes "from the certifier's draw", and
  that PoUW and the certifier import the sampled proofs. PoUB on PoUS is agreed. What else?
- **Serving notions inside `verity/`:** AGENTS.md says the certifier's advice format and vLLM's scheduler still sit
  there. `tests/test_integration_boundaries.py`'s `KNOWN` list names the sites.
- **The training slowdown's calculator** imports the periods' arithmetic, which after tonight's fold is
  `verity/protocol.py`, in the same place of the import listing as the runtime.

Hours (PST): a checkpoint line every 60 minutes. **5:00 AM: the first whole cut** (all five parts, part 2 with
whatever the sweeps have given), pushed and synced, and end your turn; top reads it and puts it before Daniel at
7:00 AM. top resumes you with the leads' corrections for a final by Fri 13 Nov noon. Read the sweeps' tables from
their branches at 2:45 AM and 4:15 AM (`git fetch origin cursor/top-layers-sweep-proofs-5d1a
cursor/top-layers-sweep-above-8c2f`; files `layers/sweep-proofs.md`, `layers/sweep-above.md`). If a sweep is late, go
on from your own reads.

## 4. The two sweeps (the evidence)

Each sweep writes ONE table file on its own branch (from `origin/main`; scratch, never a pull request) and the same
text as its note. Read only: `rg`, `git`, the files. No node run, no build. Every row has a file and a line, a count
where there are many, and [read by me]. A row says what the code does; it does not judge. First table pushed by
**2:30 AM PST**, the full one by **4:00 AM PST**, then end your turn.

**SWEEP-PROOFS**, branch `cursor/top-layers-sweep-proofs-5d1a`, file `layers/sweep-proofs.md`. The proofs' group:
`verity/protocols/flock/`, `spot/`, `recursion/` (Python, Lean, Rust, tests apart), and the areas of `security/` that
state and prove their guarantees.
1. Every import, Lake `require` and load by path that leaves the group, by target. Mark tests apart.
2. Every place the group's code, Lean, record formats or parameters name a notion of a layer above it. Search for at
   least: registration, registered, declared, declaration, day, period, window, served, stand-in, request, model,
   weights, checkpoint, stream, prover identity, binding, salt, PoUW, PoUS, PoUB, certifier, network, memory, wipe,
   hardware, device, site, account, compose, composite, remote, onsite, vllm, torch. For each notion: the count of
   files and lines, the three places that matter most, and whether it is a parameter the caller passes, a field of a
   record, a check the group itself decides, or only prose.
3. Every format string and protocol id the group defines or reads (`verity/.../vN`), with who else reads it.
4. The reverse: every import from outside the group into a module that is not the group's public surface (name what
   you took as the public surface and why).
5. What the group's devices (SPOT's challenger, the ZK server) sign or store that is not a challenge, a commitment or
   a proof.

**SWEEP-ABOVE**, branch `cursor/top-layers-sweep-above-8c2f`, file `layers/sweep-above.md`. Everything else under
`verity/`, plus `security/`'s other areas.
1. The interface package (`verity/protocols/interface/`): each module, its lines, what it defines in one line, who
   imports it (by group: proofs, accounting, policy, compose, security, tests), and which layer its content belongs
   to by the page.
2. Imports among the accounting protocols and the policy (pouw, pous, poub, certifier, inference_only) and from them
   into compose (remote, onsite, composition), in every direction, with the line. Mark tests apart.
3. Any protocol or shared file that imports the runtime; any file of `verity/` that names an adapter, PyTorch, vLLM or
   a model; `commitments/` or `circuits/` naming a protocol.
4. Shared concepts defined more than once, in Python or Lean: the data sheet, the resource or part of the hardware, a
   protocol's id, a period or window, a commitment's root, a challenge or coin, a device's key. For each: every
   definition's place and who reads which.
5. `security/`: for each area, the guarantee statements that read a definition of another area or of a protocol's
   code package, from `security/lean-audit.json`'s `layers` and `trees` (their `except` lists) and your own `rg` over
   imports.
6. `verity/claims`: every reader, and what each takes.

## 5. Your final message to top

At most 25 lines: the branch and head, the file, the note's path; for LAYERS the five that matter most in one line
each; what you did not read; who must answer what.
