---
name: friction
description: >
  What to do when a tool, rule, check or step gets in your way in Verity: fix it yourself, ask Daniel only when it is money,
  keys or a claim, record his ruling so nobody asks the same question twice, or leave a short friction note
  when you can't fix it now. Holds the rulings so far on which changes an agent makes without asking (spend, gates in
  `check`, tests, shared tools, other lanes' work) and the `kind: friction` note format with its `severity: incident` flag.
  Use when you worked around something, a step bit you twice, you see a better abstraction, you're unsure whether a change
  is yours to make, you just got a ruling from Daniel, you're the coordinator reading friction notes, or a command fails for
  missing credentials.
---

# Friction

When something gets in your way (a wrong command, a clumsy step, a check that fails late, a missing abstraction), take the
first of these that applies.

If `research data`, `research queue` or node ssh fail for missing credentials, rerun from a fresh tmux login shell before
anything else: on a cloud VM a long-lived tool shell can lose the injected secrets (store, node SSH key, Slack token) that a
fresh login shell still has. `research` itself takes them from the tmux server first; a plain `ssh` or script doesn't.
If git push fails because the VM's GitHub token lapsed, hand the branch over through the evidence store (git bundle create B
origin/BASE..BRANCH, then research data put --kind evidence/v1 --file B --preserve), and send the art id, the branch and its
head sha: no agent can read another agent's Cursor store.

## 1. Fix it, when it's clearly yours

Most friction is. Fix it at the source, in the PR you're working on or in a small one of its own, instead of working around
it:

- anything in the files your task owns: your package, its tests, your lane's scripts;
- a bug in a shared tool that stops it doing what its own docs or help say;
- a wrong or missing command, flag or gotcha in a skill or a `kb/` doc;
- a failure message that didn't say how to fix the failure;
- a flaky test: delete it; a needlessly slow one: make it smaller or delete it.

Say what you fixed in the PR description. It needs no note.

## 2. Ask, when you can't tell

A change is near the line when it changes what other agents must do or what a gate accepts, touches another lane's
in-flight files, or deletes others' work. Read the rulings below first, and follow one if it covers the case. Otherwise
ask:

- one stand-alone question to the owner of what the change touches, or to your coordinator when owners disagree or you
  can't tell who owns it (a handle holder: on Slack, `research msg send --to @HANDLE`; a worker: its report or reply to
  the coordinator that launched it). The coordinator decides, and says what it decided. Only keys (access, a credential,
  a node's root) and claims (what a guarantee says or rests on) go on to Daniel, as written; spend is its lead's call
  (13 Oct), with pods still bounding themselves under a `budgets.toml` line;
- the case, the change, and your recommendation, in a few lines;
- meanwhile, prepare the change on your branch, and mark it ready only after the answer.

For a new rule, gate or required review, price it first with `.agents/skills/process-design/SKILL.md`, and put the
proposal in the question.

## 3. Record the ruling

A ruling is a labelled example near the line. Record it where the next agent at the same spot will read it, in the PR that
makes the change or in a one-line PR of its own:

- a norm every agent needs: a line in `AGENTS.md`;
- a norm for one kind of work: a line in that work's skill (testing, Lean, ...);
- otherwise: a line under "Rulings so far" below, with its date.

Write it for the class of case, not the one instance, in one line. Delete it once code enforces it.

Then roll it out in the same PR: search for the old way's terms and paths (`rg`) and rewrite every place that still states it,
in `AGENTS.md`, `README.md`, the skills, docstrings and docs. A ruling written only in a log is one the next agent misses.

## 4. Note it, when you can't fix it now

Write a friction note when the fix isn't yours or can't happen now, and the friction cost something: a failed run or check,
about half an hour, or the second time you hit it. Not for what you fixed, and not for one-off slips. Search the notes first:
if one covers it, write a two-line note naming it under `recurs:`, because repeats are what get fixed.

A note is a file, never a message, so it pings nobody. A blocker still goes to your coordinator as a message (Slack, or
your report to it); the note is its record.

~~~markdown
---
id: <lane>/<YYYYMMDDTHHMMZ>-friction-<slug>
lane: <lane>
kind: friction
status: open
severity: incident     # only when a result, money, `main`, a gate or a deadline was lost; otherwise leave the line out
recurs: note:<id>      # optional: the earlier note about the same thing
---

# <one line: what got in the way>

What happened, with ids (run, PR, commit), and what it cost. What you did instead. The better abstraction, if you see one.
~~~

Save it as `~/.research/notes/lanes/<lane>/<YYYYMMDDTHHMMZ>-friction-<slug>.md`. A worker that can't push the notes puts
the note in its report, and its coordinator saves it. The notes are public: nothing private (lane contract §5b).

**The admin's pass.** The admin (`@top`) reads new friction notes once a day in its existing loop
(`research notes mcp`, `search_notes` with `kind: friction` and `since`). It turns a repeat into a fix assigned to its owner.
Only a repeat whose fix needs money, keys or a claim becomes a question for Daniel.

## Rulings so far

Newest first. Each is Daniel's unless it names the coordinator.

- 2026-10-14 (10:14 AM PDT): when Cursor's PR tool turns a PR creation into a dashboard approval request, the agent
  opens the PR itself with the GitHub command line through the broker instead, then says so where it registered the
  request, so nobody creates a duplicate from the dashboard. It replaces waiting on dashboard clicks, which cost about
  six hours across fourteen finished PRs overnight on 14 Oct.
- 2026-10-13 (11:01 PM PDT, cards 2e0ff59b and 54bd993d): a guarantee's checked family may fold in a check the compiled
  verifier already runs, and a new fail-closed refusal modelled the same way: the statement moves toward what the
  verifier refuses, and the landing DM shows the moved records. The same night: a free statement parameter whose
  adversarial values break a proof is closed by a side condition decidable at any literal instantiation, discharged by
  computation at use (never a listed assumption), keeping the general form.
- 2026-10-13 (about 2:24 PM PDT, relayed by top): no Cursor Projects, and no agent state kept only in a Cursor store.
  Each coordinator is a plain Cursor cloud agent that owns a vertical slice and runs its own subagents; coordinators
  talk to each other through Slack and Linear; one admin agent holds `@top` in place of the Project's top-level agent;
  and a successor takes over each lead: the handle stays, and its holder changes.
- 2026-10-13 (2:22 PM PDT): leads decide open questions themselves, spend included; Daniel's explicit rulings stand until
  he reverses them. Of 8 Oct's "money, keys and claims", spend is now its lead's own call (pods still bound themselves
  under a `budgets.toml` line, and a budget line named in a standing ruling stands); keys and claims still go to Daniel.
- 2026-10-13 (1:33 PM PDT, relayed by top): the recursive audit (`Flock.Guarantees.RecursiveAuditC3`) may rest on the
  model claim `registrant-holds-openings` (`verity/core/claims`): the prover holds a row and a salt under every leaf it
  registered, R's level-0 staging salts included. Those salts are held by the prover's own workers behind the firewall.
  In Lean the guarantee takes `salt` as a binder and is applied at the salts those workers committed under, never at
  `∃ salt` or a `Classical.choose` of one, and every docstring that reads it cites the claim (lean's review).
- 2026-10-13 (10:42 AM PDT): an agent closes an old PR without asking when every commit on it is already in another open
  PR ("close old PRs idgaf do that").
- 2026-10-12 (4:08 PM PDT, to top): Daniel took every recommendation then open in top's walkthrough of his decisions,
  unread, for maximum research velocity, and may revisit any of them later. The ones that set rules:
  - The network certifier and the memory challenger are the auditor's hardware, so they survive the 10:03 AM ruling
    below, on two conditions: each runs on a machine the prover doesn't control (the auditor's own VM or box, under the
    auditor's account, never a process on the prover's node), and `trusted-nebius-host` trusts only such a device and
    its host provider, never a host the prover controls. The bandwidth audit's challenger runs on the GPU's own node,
    so it is prover hardware and not one of them.
  - PoUS keeps sending each challenge when the last answer arrives, and the actual-circuit model's bridge covers that
    schedule, rather than PoUS moving to a fixed one.
  - PoUS's conjecture is restated once, per round and for any store size in a stated range, with today's whole-audit
    form at 2^19 blocks derived from it.
  - PoUW's machine stop restates `TT_OUT` with a per-coin budget, and the static refund stays held.
  - P2 at 2^25 blocks may land first as a stopgap, with the sampled setup check at k = 140, while the word codec is
    the direction.
- 2026-10-12 (about 10:03 AM PDT, in memory accounting's thread): no protocol or guarantee assumes trusted prover
  hardware ("in our setting we will never assume trusted prover hardware so forget about that more broadly"). Nothing
  the prover runs or hosts is trusted: not its GPUs, its host, its links, a confidential-computing enclave or a process
  on its machines. So the bandwidth audit treats host memory over PCIe as a slower tier of the prover's memory and
  prices relayed tags with `RateAuditInteractive`, instead of a trusted host challenger.
- 2026-10-12 (06:00Z, relayed by top): writing for Daniel puts its math in LaTeX, `\( \)` inline and `\[ \]`
  displayed, quantities included, and states a protocol or guarantee as its security assumptions, then the protocol's
  shape, then the properties provable under them.
- 2026-10-11 (8:53 PM PDT, through his console): a part of `check` whose cold run takes more than about 10 minutes may
  run right after its train lands instead of before, on every landing that changed its inputs, and a failure there
  reverts the landing at once; a reused verdict stays in the gate, and the Lean check always runs before landing.
  `main` then passes every part except those in the post-landing tier, and citations and tables read only commits
  whose trailing parts passed. ci and infra build it; until they do, the whole `check` stays before landing. This
  replaces 7 Oct's "not into teaching `research merge` to land on part of a check" for those parts.
- 2026-10-11 (about 7 PM PDT, relayed by top): check time comes down by a lot, and everyone may do whatever it takes.
  A change to how `check` runs (a suite's declared inputs, Lean reuse per module or declaration, circuit-check's task
  list, scheduling, sharding, more slots or machines under a budget line, splitting or cutting a slow test) lands on its
  own passing check with no review by the file's owner, who is told, not asked. infra owns check time as an outcome,
  and each lever's owner builds it, or infra does in its place. Only spend past a budget line and moving a part out of
  the gate that `main`'s passing check covers go to Daniel. The same evening: hand landing is to be automated, and ci
  owns it. From an owner's ready mark to `main` moving (the prep, the merge check, the landing, the sweep, the spec DM),
  no agent acts unless the automation refuses; the lander handles what it refuses (a conflict, a failed check, a
  revert), and infra hosts it.
- 2026-10-11 (relayed by top): the store has no API server. Writers and guards use the bucket, and displays query an
  index derived from it. D2–D4 (the keys, private metadata in the index, what is public) are as in the design,
  note:verity-top/20261011T1758Z-draft-store-contract.
- 2026-10-09 (7:04 PM PDT, relayed by top): the end goal is sampled proofs through recursion with hidden unit circuits,
  on four assumptions: record custody, the OS's randomness, the coin server's randomness, and SHA-512 entering only
  through reductions. The partition is hidden from the start. The auditor checks only padded counts. The outer proof
  proves coverage against the committed partition: each sampled unit sits in exactly one slot, of its committed type,
  on its committed values. Revealing each sampled unit's type would leak the architecture, since a uniform sample's
  type mix estimates the model's. The public-unit case is this design with the partition opened, and
  `RecursiveAudit` is stated for it first. The same night:
  - The universal unit (option 3) is frozen. Its registration, hidden reads, statement hiding and `circuit_indist`
    are kept; its unit-specific work stops.
  - VBridge is to become a theorem about one Lean function: Lean's `VStar.statements` builder, proved correct one
    building block at a time.
  - Hiding the type mix costs padding, not an evaluation of every type. A session picks its type's matrix by a hidden
    index, small types are padded to the shape's nonzero bound, and session counts are padded per shape.
  - No benchmark harness is built until the theory lands.
  - (7:08 PM) One key per protecting party, drawn once at setup from the OS before anything else: the developer's for
    its registrations and the firewall, the challenger's for coin commitments. It is public and a constant in V\*, so
    one V\* serves every registration. A Lean bound over the key chooses the data and the auditor's choices (prompts,
    coins) after the key, never before it. A key on a wire stays the fallback: hm96's key is a GF(2) Hankel multiplier,
    XORs as a constant and an AND per product as a wire.
- 2026-10-09 (4:48–5:00 PM PDT, relayed by top): a key that hides a commitment is drawn by the party the hiding
  protects, and fixed in the commitment before use (which party's key, and when it is drawn: 7:08 PM, above). hm96's
  pinned key stays only where the key is wired into a circuit (the row gadget), as a named exception. The inner phase's
  coins are sent live, because the auditor sees only hiding commitments that are never opened; committed coins are for
  V\*'s outer sessions. "The verifier accepts the inner statement" in a soundness event means the statement-level checks
  the auditor runs, never acceptance of the inner transcript, which VBridge derives.
- 2026-10-09 (9:01 AM PDT, relayed by top): a part's verdict in `check` is keyed on the code that decides that verdict (a
  suite's runner and harness), not on every line of the check tool; that is what the 7 Oct ruling's "the check's code" means.
  The other half of reuse is declared inputs narrowed to what each suite reads, and since the guard fails a suite that
  reads outside its inputs, narrowing fails closed. The same day:
  - Raising `maxHeartbeats` is not banned. The kernel checks a proof the same way at any limit, so an override is a
    question of cost, not soundness. The Lean check records each override (file, value, reason, measured time) and the
    nightly report lists them; nothing blocks on one.
  - A premise is listed only if it is a fact about the world. Anything the code could check, or that could be proved,
    becomes a check or a proof instead.
  - A property with several conjuncts gets one shape-check entry per conjunct, so it stays one statement.
  - A private circuit's class is "any description of its width, read in gate order", provided the format makes a
    malformed description impossible: each gate reads only inputs and earlier gates, so no description has a cycle. The
    class then reveals exactly its gate count, inputs and outputs.
  - Salts are drawn from the OS, not expanded by a PRG. What remains is salted SHA-512's hiding, stated as a reduction,
    and the OS's randomness, a listed premise.
  - A public constant that sets how many statements a proof has (V\*'s `MAX_SLOTS`) appears in the soundness bound,
    whose sum depends on it.
- 2026-10-08 (11:02 PM PDT, relayed by top): parallel proof work runs against fixed statements. The final theorem's
  statement and each open piece's statement go on the branch before a worker starts, workers prove against them, and
  integration runs continuously; what the deadline leaves open is a gap list of exact lemma names
  (`.agents/skills/lean-proofs/SKILL.md`). The same night: say "private circuit" or "the circuit's description", never
  "netlist" (the README's Glossary).
- 2026-10-08 (7:13 PM PDT, relayed by top): a protocol's verifier of record switches from Python to Lean only on
  evidence. Its gap list shows nothing the Python checks that the Lean doesn't, for the record versions in use, and every
  recorded run in the store, replayed through the Lean, gets the verdict the Python gave, negative controls included. The
  Python keeps running beside it as the reference, the switch is a one-line change that can be reversed, and it moves
  that protocol's properties from `Models/` to `Properties/`. The same day: lanes don't pick up rule changes on their
  own, so each lane rereads AGENTS.md and the rulings here and in `.agents/skills/lean-proofs/SKILL.md` at the start of
  every night, and each assignment from the coordinator lists the rules changed since that lane last read them.
- 2026-10-08 (5:59 PM PDT, relayed by top): a named conjecture is a premise in a section of its own, with a falsification
  test that runs, and a claim resting on one is listed as conditional. A model of how a device computes, such as
  `TTOutPearlC`, is an assumption about the world, checked by measurement. Claims resting on a falsified premise come off
  the list, and until the shape check unfolds `def G : Prop` bodies, the audit's past "no assumptions" results are
  unverified (`.agents/skills/lean-proofs/SKILL.md`).
- 2026-10-08 (5:11 PM PDT, relayed by top): the premises rule is mechanical. A property's only arguments are the input,
  the function's acceptance and premises from the list, a premise being an assumption about the world that the function
  can't see. Completeness theorems move to `Proofs/`, each with one concrete instance of its honest case, and
  properties that don't fit yet sit on a frozen list that only shrinks (`.agents/skills/lean-proofs/SKILL.md`).
- 2026-10-08 (4:30 PM PDT, relayed by top): a property's hypotheses come only from a short list of named premises kept
  in `Security/Definitions/`: physical premises, such as a device being intact or every uplink being certified, and
  VBridge until it's proved. The Lean check refuses any other hypothesis in `Properties/`, and adding a premise is a
  `Definitions/` change, so it messages Daniel. The same day, `RecursiveSound` took the fork form of the hash break
  (`.agents/skills/lean-proofs/SKILL.md`).
- 2026-10-08 (22:55Z, the coordinator's, under the 7:16 AM PDT ruling): the two-PR cap counts the PRs a lane is working
  on. A ready PR whose head is in a queued or running check, or in a train the lander holds, doesn't count, and counts
  again if that check fails, so a slow check never holds a lane's next PR.
- 2026-10-08 (14:16Z, relayed by top): no change waits on a per-head grant, whether red-team's, the vLLM coordinator's
  or a statement reviewer's: `research merge` and `research queue` read none. It retired the grant-carry rulings of 6 and
  7 Oct.
- 2026-10-08 (8:25 AM PDT): a guarantee never assumes a hash or a signature. It ends in a break tied to the run, such as
  two distinct inputs the run itself hashed with one SHA-512 digest, and an input condition becomes the verifier's
  refusal rather than a hypothesis. The form is in `.agents/skills/lean-proofs/SKILL.md`.
- 2026-10-08 (7:16 AM PDT): research waits on Daniel only for money, keys and claims: spend past a budget line, access
  and keys (a node's root, a credential), and what a guarantee says or rests on. Everything else is its owner's call, or
  the coordinator's when owners disagree, and the owner says what it decided. The clauses below that sent anything else to
  him are rewritten. The same day:
  - a lane works toward one or two outcomes a night, each with a done-condition anyone can check; its lead proposes them
    and Daniel approves or edits them in one pass. The coordinator's reports lead with which outcomes moved and what
    blocked the rest, not with landing counts;
  - only the lander lands, restacks and reruns. A lane pushes its PR, marks it ready and goes back to research, and hears
    back only if the PR fails. A lane holds at most two open PRs. This ends 6 Oct's trains without ready marks: a PR
    joins a train on its owner's `ready` label, and only the quick-tier pass stays waived. Since 12 Oct, ci's
    `research queue drive` lands, restacks and reruns, and the lander handles what it refuses (11 Oct);
  - infra keeps the machines up and the checks fast, and builds nothing new unless it removes a measured research cost.
    Kueue as the only scheduler (#1460) is one, since it takes slot assignment off the coordinator;
  - the Python verifiers move into `verity/core/`, by one agent with `main` frozen, and `Security/Models/` tracks which of
    them aren't proven yet.
- 2026-10-07 (3:54 PM PDT; 3:25 PM PDT: "skip check more", revert what breaks): `main` always passes the whole `check`,
  kept fast by reusing each part's verdict while its declared inputs, the base image and the check's code are unchanged;
  the merge queue runs it on the queued group, and a nightly uncached check must agree (the CI page in Notion). Until
  reuse and the queue exist, a change lands by hand once every part whose declared inputs it touches passes; layout
  moves and check or merge tooling get a full `check` first; a full `check` of the newest `main` follows on vy-nebius-1,
  and the landing that fails it is reverted at once (`AGENTS.md`, "Checking and merging"). Effort goes into reuse
  (declared inputs, a verdict cache, Lean replay per module), not into teaching `research merge` to land on part of a
  check. Before an expensive step (a full check, a move window), rehearse the cheap one after it: `research merge
  --dry-run` against a scratch `main` would have shown on 7 Oct, ten hours earlier, that main's gate could accept no
  check of the moved tree.
- 2026-10-06 (8:00 PM PDT, relayed by top): to make up for lost time, a lane uses as many workers and pods as its
  `budgets.toml` line allows without asking, and tells its coordinator when the line binds. GPU work still names its
  question.
- 2026-10-06 (9:49 AM PDT): the repository becomes the minimal, DRY implementation of the system as now decided. Every
  owner removes the smells it finds in its own files without asking: code serving a retired question, two sources of
  truth for one thing, variants a ruling already chose between, old vocabulary, machinery for a retired process,
  structure that doesn't earn itself, and dead references. Directory moves wait for the consolidation move. What
  deletes what another owner may use, or crosses owners, goes to that owner, and the coordinator settles a disagreement;
  only a change to a claim goes to Daniel (8 Oct).
- 2026-10-06 (9:45 AM PDT): the benchmark comparison tables (`verity_numerical.bench`'s `views`, `tables`, `drilldown`
  and `store_tables`, spec `kb/TABLES.md`) and `verity.core.claims`, their label registry, go to `archive/`. A
  certificate cites the Lean assumptions it rests on by name. The circuit templates, input
  sets and generators stay live.
- 2026-10-06 (9:32 AM PDT): the repository is consolidated in one pure move after the current trains land:
  `backends/flock` becomes `flock/`, `integrations/vllm` becomes `vllm/`, the GPU probes go to `benchmarks/`, the
  frozen numerical work goes to `archive/`, and `verity/protocols/accounting/{work,space,communication}/<name>` flattens
  to `verity/protocols/<name>`. Whether the cluster and repository tools live in `tools/` or `infra/` is infra's
  call. "Network transparency" is "network accounting".
- 2026-10-06 (8:53 AM PDT), the coordinator: a PR in a tip that is being checked keeps its head until the tip lands or ci
  asks for a change. A conflict is fixed at the tip (ci restacks it), not by pushing to the PR: a new head refuses every
  tip that pins the old one at landing, and their checks are spent for nothing. Since 12 Oct, drive leaves a conflicting
  PR out of `next` for the lander (11 Oct), and a new head still cancels the checks of the trains above it.
- 2026-10-06 (8:07 AM PDT): one hash covers derivations and PRFs: `derive` is SHA-512, P2's key expansion counter-mode
  SHA-512, and SHA-512 truncated to 256 bits counts as SHA-512; Pearl's FP8 scheme keeps BLAKE3 as Pearl's own
  definition.
- 2026-10-06 (8:03 and 8:14 AM PDT): research velocity first, no red tape (`AGENTS.md`, "Working principles"). Its
  applications so far:
  - a train member needs no quick-tier pass (its own line below);
  - a flaky test is deleted, not rerun or fixed (testing skill, "Write tests that mean something");
  - an auditor-trusted device runs on Nebius, with trust in Nebius a named assumption, rather than on hardware that
    attests (since 12 Oct: on a machine the prover doesn't control, 4:08 PM PDT above).
- 2026-10-06 (8:14 AM PDT): a protocol carries no policy parameter. It surfaces what its draw established (ε at δ, in the
  integrity profile), and accounting decides later what it accepts. The model's size is public.
- 2026-10-06 (7:51 to 8:14 AM PDT): one hash, SHA-512 with its whole output, everywhere: Pearl-C's `-h3` (its cost over
  the BLAKE3 formats accepted), PoUS's P2 (v4) and its key expansion, core's `randomness.derive`, and LMS (n = 64, no
  truncation). No hash flags or modular hashing in research mode, and performance requirements relax as far as that
  needs. BLAKE3 code leaves the served path for `archive/` (Pearl-C's is `archive/pouw-blake3/`), kept but not
  maintained. Pearl's FP8 scheme stays as Pearl defines it, semi-maintained (BLAKE3 in `schemes/pearl_kw.py`), with
  `cr/blake3` its assumption alone.
- 2026-10-06 (8:03 AM PDT): a train member needs no quick-tier pass. The train's own full check is the gate, and the
  owner's `ready` label at the PR's head, which a train member still needs (8 Oct, 7:16 AM PDT), may be written by hand
  (`research data label pr:N@SHA ready true --by LANE`).
- 2026-10-05 (10:45 PM PDT): when Daniel hands over a design to implement, build it unless there is a strong reason not
  to: a measured cost that breaks a budget, a soundness or zero-knowledge hole, or a contradiction with a proved theorem.
  Taste isn't one. With a strong reason, build the nearest version you can defend and leave him a report, rather than
  waiting for him.
- 2026-10-05 (10:45 PM PDT): the proof service is built to the decisions table of
  `note:verity-root/20261006T0545Z-draft-daniel-one-recursive-architecture-v2`. The rules from it that generalize:
  - One hash, SHA-512. Every commitment is hm96 rows in frame-v3-sha512 trees, and signatures are hash-based over
    SHA-512.
  - What an auditor-trusted device signs is witness and never reaches the auditor in the clear. Such devices are the
    network certifiers and the memory challenger.
  - Partitions are private, and their properties are public.
  - No retries for now. A failed proof is a visible failure, and a retry never redraws.
  - Outer proofs go through the firewall one at a time.
  - Live coins only; Fiat–Shamir is out.
- 2026-10-05 (10:45 PM PDT): the names are network certifier (was the warden), network certificates (its records),
  memory challenger (PoUS's timed verifier), firewall (prover gateway), challenger (verifier gateway), verifier (batch
  verifier) and workers (GPU workers). "Gateway" is retired. The rename move rolls them out with the Glossary.
- 2026-10-05 (7:48 PM PDT): every protocol is zero-knowledge, PoUS included: the developer's data and circuit are hidden,
  and so is anything that reveals them (shapes, row indices, layouts, unsalted digests). Only the minimum is public, each
  item with its reason, and there is no clear mode. In top's reading (8:15 PM), public items are of four kinds: protocol
  structure, the protocol's outputs (a verdict, PoUW's credited work), the verifier's own coins and draws, and hiding
  commitments (a salted root).
- 2026-10-05 (7:17 PM PDT): the proof layer is one service every accounting protocol uses. A protocol states what is
  committed, which units need proofs (or the rule that selects them), the public Program each must pass, and what it
  does with the outcome; the service owns commitments, live coins and the record, the selection, proving, verifying and
  the integrity profile, and no protocol keeps its own copy of them.
- 2026-10-05 (7:11 PM PDT): a circuit restatement of a verifier (VBridge's V\*) is proved in one direction, that the
  circuit accepting implies the verifier of record accepts; completeness stays a test.
- 2026-10-05 (rewritten 8 Oct): a lead resumes or re-dispatches a turn cut off by a console drop or a provider outage on
  its own judgment, and tells its coordinator.
- 2026-10-05 (1:27 PM PDT): a coordinator never loses contact with a worker it launched. Nothing reaches a running agent:
  a `Task` resume is refused, another agent's `SendToAgent` refuses it ("not a worker of this coordinator") and its
  dashboard follow-up box is disabled. So the Lean move lead went 85 minutes unreachable. Every brief therefore names an
  inbox branch, `cursor/inbox-<first 8 of the worker's id>-<launcher's suffix>`, and the worker runs `git fetch` on it
  at every checkpoint (at least every 20 minutes, contract §3), acting on each new commit message. The launcher writes
  there with empty commits; the inbox is a channel, not a note. A worker silent for 30 minutes gets one inbox message,
  and only if that brings no answer does the launcher start a parallel lane. A worker that is IDLE in
  `list-cloud-agents` with no result delivered was stopped: report it to your coordinator, don't resume it.
- 2026-10-05 (8:53 AM PDT), superseded: `main` kept `tools/verity/lean/check.py` and each package's `lean-audit.json`
  under their names (`f507299a4`), so no rename is pending. What stands: the Lean audit is the Lean check.
- 2026-10-04 (2:21 PM PDT): the repository migration finishes in short serial time. A lead launches workers for every
  piece that doesn't wait on another (Lean ones in separate checkouts), and spends compute wherever it shortens the
  critical path: reruns, check capacity, extra CPU pods. Pods still bound themselves under a `budgets.toml` line, and GPU
  work still names its question.
- 2026-10-04 (2:21 PM PDT): during the migration, ci resolves a conflict that is only a rename (a moved file that another
  PR edited) in the train tip's merge commit, instead of sending the PR back for a restack, and names each resolution in
  the handoff. A conflict that needs judgment still goes to the PR's owner, and `check` still runs on the merged tree.
- 2026-10-04 (8:42 AM PDT): the proof system proves batches of arbitrary Boolean circuits, so soundness is **one
  theorem over everything the compiled Lean verifier accepts**, with no hypothesis on the circuit file or the statement's
  form (no `scopeOk`, no `pub.tables = none`). The verifier fails closed: it refuses every form the proof doesn't cover,
  and rejecting an unproved form is a completeness limit, never a soundness gap. A circuit's inputs and outputs use one
  canonical format, proved once in general (any row length, any input groups, any output placement), and frontends
  normalize to it. Adding a workload never needs a verifier or proof change. The verifier gains no new form ahead of its
  proof, and progress is never measured by which workloads' files fall inside a proof's scope. Pushback is welcome:
  send it to Daniel through @proofs.
- 2026-10-03: a test tier too slow for `check` (the vLLM reference regression's manifest rebuilds, about an hour) runs as a
  detached job after each train lands on `main` (`tools/verity/check/post_train.py`, started by `research merge --push`) and labels
  the landing's check attempt `note=regression pass` or `fail:<rows>`, which circuits sweeps daily; `check` blocks on the
  cheap tier only (its T0 checks, about four minutes), and under `VERITY_STORE_REQUIRED=1` a fixture or input that doesn't
  resolve fails rather than skips.
- 2026-10-03 (the coordinator's): data you don't own on a node is deleted only with `research retention gc` or `research
  retention rm --approved-by @OWNER`, which refuses unless every retention record on the path names that owner; never a bare
  `rm`, and never on another lane's word about who owns it.
- 2026-10-03 (card `d9eb2f28`): PoUW may import sampled proofs, in Python (`ALLOWED` in
  `tests/test_boundaries.py`) as in Lean. Any other protocol → protocol import is the importing protocol's owner's
  call, told to the imported one's owner, and the edges stay acyclic. The layout that follows (lean with compute-accounting): a protocol guarantee that needs a proof
  system (PoUW's `TileProofSoundAll`) is stated over sampled proofs' statements of what a backend proves (its soundness on
  registered inputs, and the honest prover's membership in the backend's class), which the backend's own package proves
  for itself. No package requires both a protocol and a backend, there is no `bridges/` layer, and a citation names both
  guarantees. Its package clause is superseded by the Lean layout move (2026-10-06): every proof is in
  `Security/Proofs`, whose package requires C-Flock's verifier.
- 2026-10-02 (10:27 PM PDT): `hash-derived-key` (`Hm96Hiding` at the pinned key) stays a named assumption. Hiding is
  proved at a fresh uniform key, and nobody works on proving that the pinned key behaves like one.
- 2026-10-02 (10:27 PM PDT): the lander starts every merge check (`check --record`, `research merge`). The PR captain
  (`ci`) trial-merges, preflights and hands over train tips, but never starts a merge check itself. Since 11 Oct, ci's
  landing automation starts them too.
- 2026-10-02 (7:23 AM PDT): an overhead reported to Daniel is the `--zk` prover's only (no M0 number and no `--zk`/M0
  ratio), normalized to the chip's peak: prover seconds over the time the computation's FLOPs take at the census's dense
  datasheet peak for its dtype, never over a measured runtime such as a vLLM run's, written to one significant digit
  (6e7, 1e8).
- 2026-10-01 (4:31 PM PDT): a table that tabulates a fixed function is a value even with no program parameter
  (`gelu_tanh_bf16`, `tanh_rn`). The only tables in an operation's own definition are the hardware's measured ones (the
  MUFU tables, `tanh_mufu` included), which stay gates with the floating-point formats' constants. The hidden read's cost
  is accepted for conceptual clarity unless it proves a bottleneck.
- 2026-10-01 (3:32 PM PDT): no protocol deals with public values, so none gets an exemption; PoUW's tile check moves into
  a proof. During development a proof may be replaced by replay, or verification skipped, for research velocity: that's
  a shortcut, never the protocol. Its clause letting shapes stay public is superseded (2026-10-05, 7:48 PM PDT).
- 2026-10-01 (2:43 PM PDT): a decision Daniel must make (money, keys or a claim, since 8 Oct) goes where Slack notifies
  him, never only in a note or a lead's thread. Use one question per post, with its options, your recommendation, and its deadline or default:
  `research msg ask-daniel` (the card mentions him), or, if the card fails, `research msg send --to @daniel`.
- 2026-10-01 (1:14 PM PDT; this wording 3:06 PM): the circuit holds only gates, and values are always hidden, the core
  guarantee of even the most basic protocol. An operation's own definition is gates: the floating-point formats'
  constants, SHA-512's, the measured GPU tables for exp2, rcp and tanh. Everything a program or a request brings is a value:
  the weights, a program's constants (eps, the attention scale, a softcap) and tables (`gelu_tanh_bf16`, `tanh_rn`), a
  request's parameters (temperature, top-p, Philox positions), every intermediate and every output. A value is committed
  under a hiding leaf and touched only inside a zero-knowledge proof, never framed as part of the program the verifier
  already has. The hill-climb optimizes the hiding protocol (`--zk`), not the non-ZK circuit prover.
- 2026-10-01 (12:37 PM PDT): lanes work around the clock, so timelines are planned in hours: a pilot, trial or rollout gets
  hours (about eight for a process pilot), not the days or weeks a human team would give it.
- 2026-10-01 (10:56 AM PDT): only leads get handles (a name in `research msg`), never individual workers; a worker is
  reached through its lead. A new lead is a line in `.agents/skills/using-slack/registry.json`, and `names open`
  refuses any other name.
- 2026-10-01 (9:50 AM PDT): lane notes are records (reports, checkpoints, findings, drafts, friction), never a channel.
  A handoff, an order or an ask goes on Slack (`research msg`), or to your own Cursor workers as a follow-up; a note
  written as a message reaches nobody.
- 2026-10-01 (9:12 AM PDT): agents message each other directly, without a coordinator relaying, and adding a handle
  needs no Slack admin. Comms owns the routing (`@<name>` at the start of a post wakes that lead's agent).

- 2026-10-01 (12:05 AM PDT): a protocol's README is an outline, signposts and its core API, and Lean is the source of
  truth for a mature protocol. A protocol keeps a `PROTOCOL.md` only while it is experimental (Python is the truth until
  the Lean catches up), and the PR whose Lean covers it deletes the file.
- 2026-10-01: whoever lands a train runs `research merge --sweep` after each push, which closes the landed PRs, retargets
  their children and deletes landed branches. `research merge --push` pushes the merge and then sweeps, so a lander passes it.
- 2026-09-30 (2:52 PM PDT; at all hours since the coordinator's 06:33Z Oct 4): GPU work runs only as items its lane owner
  said yes to, each naming its research question (`--question`, or a `question` label on its run). There is no filler:
  idle GPUs beat padded ones, a steward reports idle GPUs instead of offering them as backfill, and approved backfill runs
  on preemptible leases.
- 2026-09-30: the Lean infrastructure lane changes `tools/verity/lean`, the `lean-proofs` skill, `check`'s Lean steps, shared
  Lean foundations and Lean or Mathlib bumps (in a PR of their own, announced first) without asking. It speeds up another
  lane's proofs when no guarantee's record changes; a changed record in another lane's package goes through that lane. What a
  headline theorem may rest on, what "proved" means, retiring a theorem a table cites, spend past a budget line and any
  new human gate go to Daniel.
- 2026-09-30: a lane fixes a bug it hits in `tools/research` in a small PR of its own, and the research coordinator only
  reviews and merges it; when an open PR already changes the same function, the fix goes to that PR's owner.
- 2026-09-30: a run's author declares what custody persists with `--declared-output`; undeclared files are hash-listed in
  `trimmed.json`, not uploaded, except that a transitional default uploads up to 256 MiB of them until lanes declare their
  outputs. How to declare them is the `writing-runs` skill (`.agents/skills/writing-runs/`, in progress).
- 2026-09-29: merge velocity outranks CI spend: a larger CI pool (about $120 a day standing, more on backlog days) beats
  trains waiting on pods.
- 2026-09-29: RunPod tops itself up, so a low balance is not a blocker and needs no request to Daniel; spend caps and guards
  are the controls.
- 2026-09-29: cache test results aggressively, per test where the key allows: velocity outranks gate purity there, with the
  nightly uncached check of `main` as the backstop.
- 2026-09-29: after a train lands, the coordinator closes each stacked PR whose code is on `main` but that GitHub didn't
  mark merged, with a "Merged into main at {sha} (train {name})" comment, without asking.
- 2026-09-28: changes that speed up checks or merges land ahead of other work.

## Before you finish

- Each fix you made is in a PR description.
- Each question you asked has its ruling recorded, or is still open in your checkpoint or report.
- Each friction note is a few lines, names its ids, and carries `severity: incident` only when something was lost.
