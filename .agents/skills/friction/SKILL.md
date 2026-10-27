---
name: friction
description: >
  What to do when a tool, rule, check or step gets in your way in Verity: fix it yourself, ask the owner or your
  coordinator when you can't tell, take only keys, access and claims to Daniel (spend is its lead's call), record a ruling as a
  rule where the next agent will read it, or leave a short friction note when you can't fix it now. Holds the standing
  rulings that AGENTS.md and the other skills don't state (what an agent may change without asking, and protocol and
  claim rulings), and the `kind: friction` note format with its `severity: incident` flag. Use when you worked around
  something, a step bit you twice, a step you repeat keeps you waiting, you see a better abstraction, you're unsure
  whether a change is yours to make, you just got a ruling from Daniel, you're the coordinator reading friction notes, or
  a command fails for missing credentials.
---

# Friction

When something gets in your way (a wrong command, a clumsy step, a check that fails late, a missing abstraction), take the
first of these that applies.

Two gotchas first. If `research data`, `research queue` or node ssh fail for missing credentials, rerun from a fresh tmux
login shell: a long-lived tool shell on a cloud VM can lose the injected secrets. If `git push` fails because the VM's
GitHub token lapsed, hand the branch over as a bundle (`git bundle create B origin/BASE..BRANCH`, then `research data put
--kind evidence/v1 --file B --preserve`) and send the art id, the branch and its head sha.

Friction in shared infrastructure (Linear, Slack and `research msg`, `research run`, the queue, notes, the store) is usually
everyone's: ask in the workflow thread first, `research msg reply --thread 1792037552.043289`, one question per post,
starting with the tool's name. The tool's owner decides, makes the answer stick in code or its skill, and replies
`settled: <PR or section>`.

## 1. Fix it, when it's clearly yours

Most friction is. Fix it at the source, in the PR you're working on or a small one of its own:

- anything in the files your task owns: your package, its tests, your lane's scripts;
- a bug in a shared tool that stops it doing what its own docs or help say;
- a wrong or missing command, flag or gotcha in a skill or a `kb/` doc;
- a failure message that didn't say how to fix the failure;
- a flaky test: delete it; a needlessly slow one: shrink it or delete it;
- a loop of yours that keeps you waiting, a build, run or check you repeat that takes more than about 10 minutes: make it
  incremental, warm or cached.

Say what you fixed in the PR description. It needs no note.

## 2. Ask, when you can't tell

A change is near the line when it changes what other agents must do or what a gate accepts, touches another lane's
in-flight files, or deletes others' work. Check `AGENTS.md`, the owning skill and the standing rulings below first, and
follow one if it covers the case. Otherwise ask the owner of what the change touches, or your coordinator when owners
disagree or you can't tell who owns it (a handle holder: `research msg send --to @HANDLE`; a worker: its report to the
coordinator that launched it). Give the case, the change and your recommendation in a few lines, and prepare the change
meanwhile. Only keys and access, and claims (what a guarantee says or rests on) go on to Daniel, through
`.agents/skills/asking-daniel/SKILL.md`. Spend, a short budget line included, is the call of the lead whose line pays (13 Oct).

For a new rule, gate or required review, price it first with `.agents/skills/process-design/SKILL.md`.

## 3. Record the ruling

Record a ruling as the rule it sets, where the next agent at the same spot will read it, in the PR that makes the change:

- a norm every agent needs: a line in `AGENTS.md`;
- a norm for one kind of work: a line in that work's skill;
- a ruling that fits neither: a line under "Standing rulings" below.

Write it for its class of case, in one line, in the present tense. Its date, its source and how it came about go in
Daniel's decisions log in Notion, not here. When a later ruling replaces it, rewrite the line; when code enforces it,
delete the line. Then roll it out in the same PR: `rg` for the old way's terms and paths, and rewrite every place that
still states it (`AGENTS.md`, `README.md`, the skills, docstrings, docs).

## 4. Note it, when you can't fix it now

Write a friction note when the fix isn't yours or can't happen now, and the friction cost something: a failed run or check,
about half an hour, the second time you hit it, or a wait of more than about 10 minutes that you repeat, even when nothing
fails (give its minutes a wait and waits a day). Not for what you fixed, and not for one-off slips. Search the notes
first: if one covers it, write a two-line note naming it under `recurs:`, because repeats are what get fixed. A note is a
file, never a message, so it pings nobody; a blocker still goes to your coordinator as a message.

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

Save it as `~/.research/notes/lanes/<lane>/<YYYYMMDDTHHMMZ>-friction-<slug>.md`. A worker that can't push the notes puts the
note in its report, and its coordinator saves it. The notes are public: nothing private (lane contract §5b).

The admin (`@top`) reads new friction notes once a day (`research notes mcp`, `search_notes` with `kind: friction` and
`since`) and turns a repeat or a repeated wait into a fix assigned to its owner with an ETA, then checks that the fix
landed and held; only a repeat whose fix needs keys or a claim goes to Daniel.

## Standing rulings

Daniel's rulings in force that `AGENTS.md` and the other skills don't state, dated so citations find them. The history
behind these and every earlier ruling, to 19 Oct, is `note:top/20261020T0640Z-report-friction-rulings-archive`.

What an agent may change:

- (6 Oct) Each owner removes the smells it finds in its own files without asking: code serving a retired question, two
  sources of truth for one thing, variants a ruling already chose between, old vocabulary, machinery for a retired
  process, structure that doesn't earn itself, dead references. What crosses owners goes to that owner.
- (13 Oct) Close an old PR without asking when every commit on it is already in another open PR.
- (14 Oct) Hard switches: once a replacement lands, no second path stays alive beside it, and every bridge kept for the
  transition names its end state. The admin audits for splits and tells Daniel when an abstraction fails or turns into
  red tape.
- (3 Oct) A protocol importing another protocol is the importing protocol's owner's call, told to the imported one's
  owner, and the edges stay acyclic (`ALLOWED` in `tests/test_boundaries.py`).
- (30 Sep) The Lean lane changes `tools/verity/lean`, the lean-proofs skill, `check`'s Lean steps, shared Lean foundations
  and Lean or Mathlib bumps (in a PR of their own, announced first) without asking. A changed record in another lane's
  package goes through that lane.
- (28–29 Sep) Changes that speed up checks or merges land ahead of other work, and merge velocity outranks CI spend.
- (5 Oct) Build a design Daniel hands over unless a measured cost breaks a budget, it opens a soundness or zero-knowledge
  hole, or it contradicts a proved theorem. Then build the nearest version you can defend and report. Taste isn't a reason.
- (1 Oct) Plan in hours: a pilot, trial or rollout gets hours, not days.
- (30 Sep) GPU work runs only as items its lane owner approved, each naming its research question; idle GPUs beat padded
  ones. RunPod tops itself up, so a low balance is not a blocker (29 Sep).
- (2 Oct) An overhead reported to Daniel is the `--zk` prover's only, over the time the computation's FLOPs take at the
  census's dense datasheet peak for its dtype, to one significant digit.

Keys and access:

- (25 Oct) The control pod holds a Nebius key that can only stop and read instances, root only, for infra's backstops,
  so stopping a machine never waits on an agent VM. A key that could also start, create or delete is a new ask.
- (25 Oct) Workers open and edit their own lane's PRs with `research pr open|edit`, through a GitHub App on the control
  pod with pull-requests write only (no contents) whose token no worker holds; the tool refuses a PR whose head branch
  isn't the caller's lane's. Drive's App stays the only one that can move `main`.
- (25 Oct, TOP-84) A lane's development gate key on an auditor's gate (vy-challenger-1, vy-auditor-2) is no ask: the lane
  makes an ed25519 key, sends infra only the public half, and infra registers it under that lane. Only the per-lane Cursor
  secret that credit-bearing windows need is Daniel's.
- (26 Oct 6:42 PM PDT) An onsite auditor device's transport key is an imprint key, a key class of its own: a shared secret
  installed over the cable once per auditor device, then a SHA-512 ratchet (`verity.ml.crypto.transport`). It is a
  protocol key, never in a record, a log or the prover's configuration.

Landing:

- (24 Oct 06:05Z) PRs land on an eyeball review, and the full check runs after landing; `main` no longer passes the
  whole check before a landing. A PR that changes a `lean-audit.json` lock, or Lean that runs as code (outside
  `Security/`), keeps its Lean check first; Lean proofs alone land on the verdict and are checked after landing, a
  failure reverting them (25 Oct 11:19 PM PDT). Reviewers are agents ci staffs, never the PR's owner. Drive does it with
  `land = "review"` (AGENTS.md "Checking and merging").
- (25 Oct 09:40Z, delegated to the vLLM coordinator) The vLLM integration's lint ratchets are no-growth: no change adds an
  exception or raises a recorded count, and a change to a lint's own code or thresholds is judged like source. Dedicated
  cleanup lanes pay the exceptions down; a feature PR owes none (`integrations/vllm/tools/ratchet_gate.py`).
- (25 Oct 11:20Z, the vLLM coordinator under that delegation) A `# noqa` is an exception too: in `integrations/vllm`, no
  change adds more `noqa` lines than it removes. One that moves with its code counts as kept, and deleting a stale one is a
  retirement. The ratchet gate counts them, with no allowlist.

Protocols and claims:

- (26 Oct 6:42 PM PDT) `WitnessPathPrivate c` is the per-channel premise, beside the per-party `Faithful p`, `Secret p`
  and `Confined p` and the site's `Covered` (`Security/Definitions/Core/Transport.lean`); a keyed channel takes
  `ChannelHides` in its place.
- (26 Oct 6:42 PM PDT) ML-KEM-1024 is the one-hash rule's transport exception, under AGENTS.md's conditions: a remote
  party's chain key comes from provisioning, and no session runs on a KEM secret alone (`transport.session` refuses it).
- (25 Oct) The warden and the network certifier are one device: a tap on each AI server's uplink, whose one code run
  under one `trusted-host` gives both the cells and the records (`Security/Definitions/NetworkCertifier/Tap.lean`). No statement
  ties them by a premise or trusts them separately.
- (21 Oct) θ comes only from registered public data, registered at the start of a deployment; outputs of earlier runs
  are never θ (drops `NciCheck.Baseline`'s output provenance).
- (8 Oct) A protocol's verifier of record switches from Python to Lean only on evidence: its gap list shows nothing the
  Python checks that the Lean doesn't, and every recorded run replayed through the Lean gets the Python's verdict,
  negative controls included. The Python keeps running as the reference, the switch is one reversible line, and it moves
  the protocol's properties from `Models/` to `Properties/`. Where it leads (18 Oct): Lean is the definition and the
  verifier of record, `verity.ml`'s Python mirrors it under difftest, and a verifier imports no `verity.ml` code.
- (17 Oct) The prover controls the audited circuit's inputs and knows every coin once it is issued, so no defense assumes
  the circuit can't learn a coin, and verifier coins stay public. Only the weights may take a private encoding from a
  trusted third party, salted after the commitment with secrets the prover never sees, as one deployment variant.
- (17 Oct) No private or trapdoor encoding for KV. Credited dynamic memory rests on the public codec, and the stacking
  bound on the named conjecture `EncodeChainHardness`, with the claims resting on it listed as conditional.
  (`EncodeChainHardness` is falsified, `EncodeChainRefuted`; since 26 Oct the bound rests on `SquaringFloor` and
  `ChainSum` in the repriced gate model, a thin lane a compare-select unit, listed the same way.)
- (12 Oct) Memory and bandwidth accounting: no bound assumes the weights are incompressible, and a scheme may refuse
  all-zero weights; exhaustion is over the node's whole memory, KV included; no overhead target beyond 99%; guarantees
  are for the whole node. The remote bandwidth audit treats the prover's resources as one pool (conservation, no identity
  assumption), claims 0.15% over a 10-minute monitoring period with every window at most 1%, and never changes the
  model's logits.
- (18 Oct, ASK-16) PoUW's guarantees, and only PoUW's, may rest on named random-oracle premises for SHA-512
  (`StraightLineExtraction`, Compose's `SeedExtractedAt`), each a named `Prop` the lock lists. C-Flock's
  knowledge soundness keeps the collision form.
- (19 Oct, ASK-72 and ASK-73) The deployed recursion's bound record states its inner claim at IVStar at the weight
  `coverW`, the least at which the premise P3 (`Bound.WeightCovers`) holds. Its security level is below the 100-bit floor
  under every weight (`Claims.Recursion.lam`), and every table, README or claim that cites it gives its λ and says so.
- (13 Oct) `Flock.Guarantees.RecursiveAuditC3` may rest on `registrant-holds-openings`; in Lean the salt is a binder
  applied at the salts the prover's workers committed under, never `∃ salt` or a `Classical.choose` of one.
- (13 Oct) A guarantee's checked family may fold in a check the compiled verifier already runs, or a new fail-closed
  refusal modelled the same way. A free parameter whose adversarial values break a proof is closed by a side condition
  decidable at any literal instantiation, discharged by computation at use, never a listed assumption.
- (9 Oct) The end goal is sampled proofs through recursion with hidden unit circuits and a hidden partition; the auditor
  checks only padded counts. One key per protecting party, drawn once at setup from the OS before anything else, public
  and a constant in V\*; hm96's pinned key stays only where a key is wired into a circuit, as a named exception. Salts
  are drawn from the OS, not expanded by a PRG.
- (5 Oct) A circuit restatement of a verifier (VBridge's V\*) is proved in one direction, that the circuit accepting
  implies the verifier of record accepts; completeness stays a test.
- (2 Oct) `hash-derived-key` (`Hm96Hiding` at the pinned key) stays a named assumption, and nobody works on proving the
  pinned key behaves like a fresh one.
- (1 Oct) A table that tabulates a fixed function is a value, even with no program parameter; an operation's own
  definition holds only the hardware's measured tables. No protocol gets an exemption for public values: in development a
  proof may be replaced by replay, or verification skipped, as a labelled shortcut, never as the protocol.

## Before you finish

- Each fix you made is in a PR description.
- Each question you asked has its ruling recorded as a rule, or is still open in your checkpoint or report.
- Each friction note is a few lines, names its ids, and carries `severity: incident` only when something was lost.
