---
name: friction
description: >
  What to do when a tool, rule, check or step gets in your way in Verity: fix it yourself, ask its owner or your
  coordinator when you can't tell, take only claims and his own clicks to Daniel, record a ruling where the next agent reads
  it, or leave a friction note. Holds the standing rulings AGENTS.md and the other skills don't state (what an agent may
  change without asking, protocol and claim rulings) and the friction note format. Use when you worked around something,
  a step bit you twice or keeps you waiting, you see a better abstraction, you're unsure a change is yours, you just got a
  ruling from Daniel, you read friction notes, or a command fails for missing credentials.
---

# Friction

When something gets in your way (a wrong command, a clumsy step, a check that fails late, a missing abstraction), take the
first of these that applies.

Two gotchas. A `research` command or node ssh that fails for missing credentials: rerun it from a fresh tmux login shell,
since a long-lived tool shell can lose the injected secrets. A `git push` refused for a lapsed GitHub token: hand over a
bundle (`git bundle create B origin/BASE..BRANCH`, `research data put --kind evidence/v1 --file B --preserve`) with the
art id, the branch and its head sha.

Friction in shared infrastructure (Linear, Slack and `research msg`, `research run`, the queue, notes, the store) is usually
everyone's: ask in the workflow thread first, `research msg reply --thread TS` with its `RULES_THREAD` in
`infra/control/top_watch.py`, one question per post,
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
in-flight files, or deletes others' work. If `AGENTS.md`, the owning skill or a standing ruling below covers the case,
follow it. Otherwise ask the owner of what the change touches, or your coordinator when owners disagree or you can't tell
(a handle holder: `research msg send --to @HANDLE`; a worker: its report), with the case, the change and your
recommendation in a few lines, and prepare the change meanwhile. Only keys and access, and claims (what a guarantee says
or rests on), go on to Daniel (`.agents/skills/asking-daniel/SKILL.md`). Spend, a short budget line included, is the call
of the lead whose line pays.

For a new rule, gate or required review, price it first with `.agents/skills/process-design/SKILL.md`.

## 3. Record the ruling

Record a ruling as the rule it sets, where the next agent at the same spot will read it, in the PR that makes the change:

- a norm every agent needs: a line in `AGENTS.md`;
- a norm for one kind of work: a line in that work's skill;
- a ruling that fits neither: a line under "Standing rulings" below.

Write it for its class of case, in one line, in the present tense. Its date, its source and how it came about go in
Daniel's decisions log in Notion, not here. When a later ruling replaces it, rewrite the line; when code enforces it,
delete the line. Then roll it out in the same PR (AGENTS.md, "A decision rolls out in the change that records it").

## 4. Note it, when you can't fix it now

Write a friction note when the fix isn't yours or can't happen now and the friction cost something: a failed run or check,
about half an hour, a second hit, or a repeated wait over about 10 minutes even when nothing fails (give its minutes a
wait and waits a day). Not for what you fixed or one-off slips. Search the notes first: if one covers it, write a
two-line note naming it under `recurs:`, since repeats are what get fixed. A note pings nobody; a blocker still goes to
your coordinator as a message.

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

The admin (`@top`) reads new friction notes daily (`research notes mcp`'s `search_notes`, `kind: friction`, `since`),
files each repeat or repeated wait as a dated issue in its owner's team, and checks that the fix landed and held; only one
whose fix needs a claim or his own accounts goes to Daniel.

## Standing rulings

Daniel's rulings in force that `AGENTS.md` and the other skills don't state, each stated as itself. Who set each one,
when and why, and every retired ruling, are in his decisions log in Notion.

What an agent may change:

- Each owner removes the smells it finds in its own files without asking: code serving a retired question, two
  sources of truth for one thing, variants a ruling already chose between, old vocabulary, machinery for a retired
  process, structure that doesn't earn itself, dead references. What crosses owners goes to that owner.
- Close an old PR without asking when every commit on it is already in another open PR.
- Hard switches: once a replacement lands, no second path stays alive beside it, and every bridge kept for the
  transition names its end state. The admin audits for splits and tells Daniel when an abstraction fails or turns into
  red tape.
- A protocol importing another protocol is the importing protocol's owner's call, told to the imported one's
  owner, and the edges stay acyclic (`ALLOWED` in `tests/test_boundaries.py`).
- The Lean lane changes `tools/verity/lean`, the lean-proofs skill, `check`'s Lean steps, shared Lean foundations
  and Lean or Mathlib bumps (in a PR of their own, announced first) without asking. A changed record in another lane's
  package goes through that lane.
- Changes that speed up checks or merges land ahead of other work, and merge velocity outranks CI spend.
- Build a design Daniel hands over unless a measured cost breaks a budget, it opens a soundness or zero-knowledge
  hole, or it contradicts a proved theorem. Then build the nearest version you can defend and report. Taste isn't a reason.
- Plan in hours: a pilot, trial or rollout gets hours, not days.
- GPU work runs only as items its lane owner approved, each naming its research question; idle GPUs beat padded
  ones. RunPod tops itself up, so a low balance is not a blocker.
- An overhead reported to Daniel is the `--zk` prover's only, over the time the computation's FLOPs take at the
  census's dense datasheet peak for its dtype, to one significant digit.

Keys and access:

- Keys and access between our own machines and accounts are the lanes' call: the lane that needs a key or a grant
  (a lane key, a forced-command key, a challenger's or a wipe's key) sets it up with the machine's owner, infra for
  shared nodes, with no ask and no wait. This is a simulation: nothing mimics deployment settings, and it needn't be
  perfect. Daniel only clicks what needs an account he holds (the GitHub organization's admin, billing, the Nebius and
  Cloudflare consoles).
- The control pod holds a Nebius key that can only stop and read instances, root only, for infra's backstops,
  so stopping a machine never waits on an agent VM.
- Workers open and edit their own lane's PRs with `research queue open|edit`, through a GitHub App on the
  control pod with pull-requests write only (no contents) whose token no worker holds; the tool refuses a PR whose head
  branch isn't the caller's lane's. Drive's App stays the only one that can move `main`.
- A lane's gate key on an auditor's gate (vy-challenger-1, vy-auditor-2): the lane makes an ed25519 key, sends
  infra only the public half, and infra registers it under that lane.

Landing:

- The vLLM integration's lint ratchets are no-growth: no change adds an
  exception or raises a recorded count, and a change to a lint's own code or thresholds is judged like source. Dedicated
  cleanup lanes pay the exceptions down; a feature PR owes none (`integrations/vllm/tools/ratchet_gate.py`).
- A `# noqa` is an exception too: in `integrations/vllm`, no
  change adds more `noqa` lines than it removes. One that moves with its code counts as kept, and deleting a stale one is a
  retirement. The ratchet gate counts them, with no allowlist.
- Outside `integrations/vllm/`, the same gate holds `tests/lint/`'s no-growth lists and checks each PR's changed Python,
  as drive lands before the suites run: fix the code, never the list; a rename moves its entries (`--move OLD NEW`); a
  list may catch up to what its base's lint finds in its tree, no further. Ruff: the root `ruff.toml`, per (file, rule),
  `noqa` uncounted. No new direct environment read: use the area's settings registry (a `settings.py`, or a list mark at
  the PR's base). No new swallowing handler: re-raise, record, or test first.

Protocols and claims:

- Credit and every report of computation are in AND gates (tentative). 𝓜's resources appear only inside
  exclusion, which reports in AND gates. Credit is a statement about values; SAK is exclusion's premise only. Whoever
  finds a reason this is wrong floats it at once.
- A workload is what a datacenter runs over a monitoring period (its Programs, their inputs and placement), at
  every scale; workloads compose, and "row" never names one.
- The network certifier is one device on each AI server's uplink, whose one code run under one `trusted-host` gives
  both the cells and the records (`Security/Definitions/NetworkCertifier/Tap.lean`). No statement
  ties them by a premise or trusts them separately.
- θ comes only from registered public data, registered at the start of a deployment; outputs of earlier runs
  are never θ (drops `NciCheck.Baseline`'s output provenance).
- A protocol's verifier switches from Python to Lean only on evidence: its gap list shows nothing the
  Python checks that the Lean doesn't, and every recorded run replayed through the Lean gets the Python's verdict,
  negative controls included. The Python keeps running as the reference, the switch is one reversible line, and it moves
  the protocol's properties into `Properties/`. Where it leads: Lean is the definition and the
  verifier, `verity.ml`'s Python mirrors it under difftest, and a verifier imports no `verity.ml` code.
- An auditor device releases its commitment to its coins only after the prover's commitment is in, so no
  soundness statement takes a hiding premise on it. A guarantee whose break includes a forged device release names two
  premises outside the theorem: `ecr/sha-512` for its finders and `euf-cma/lms-sha512-n64` for the forgery finder.
- The prover controls the audited circuit's inputs and knows every coin once it is issued, so no defense assumes
  the circuit can't learn a coin, and verifier coins stay public. Only the weights may take a private encoding from a
  trusted third party, salted after the commitment with secrets the prover never sees, as one deployment variant.
- No private or trapdoor encoding for KV. Credited dynamic memory rests on the public codec, and the stacking
  bound on the named conjecture `EncodeChainHardness`, with the claims resting on it listed as conditional.
  (`EncodeChainHardness` is falsified, `EncodeChainRefuted`; the bound now rests on `SquaringFloor` and
  `ChainSum` in the repriced gate model, a thin lane a compare-select unit, listed the same way.)
- Memory and bandwidth accounting: no bound assumes the weights are incompressible, and a scheme may refuse
  all-zero weights; exhaustion is over the node's whole memory, KV included; no overhead target beyond 99%; guarantees
  are for the whole node. The remote bandwidth audit treats the prover's resources as one pool (conservation, no identity
  assumption), claims 0.15% over a 10-minute monitoring period with every window at most 1%, and never changes the
  model's logits.
- PoUW's guarantees, and only PoUW's, may rest on named random-oracle premises for SHA-512
  (`StraightLineExtraction`, Compose's `SeedExtractedAt`), each a named `Prop` the lock lists. C-Flock's
  knowledge soundness keeps the collision form.
- PoUS's held-space bound in the shared coin game rests on SHA-512 as a PRF for the memory challenger's derived
  streams: the named premise `Hash.Assumptions.Sha512Prf`, its advantage `ε_prf(t, Q)` explicit in the bound
  (`Pous.Period.PeriodHeldSha512`), and protocol soundness proves `PousSound` from it.
- The deployed recursion's bound record states its inner claim at IVStar at the weight
  `coverW`, the least at which the premise P3 (`Bound.WeightCovers`) holds. Its security level is below the 100-bit floor
  under every weight (`Claims.Recursion.lam`), and every table, README or claim that cites it gives its λ and says so.
- `Flock.Guarantees.RecursiveAuditC3` may rest on `registrant-holds-openings`; in Lean the salt is a binder
  applied at the salts the prover's workers committed under, never `∃ salt` or a `Classical.choose` of one.
- A guarantee's checked family may fold in a check the compiled verifier already runs, or a new fail-closed
  refusal modelled the same way. A free parameter whose adversarial values break a proof is closed by a side condition
  decidable at any literal instantiation, discharged by computation at use, never a listed assumption.
- The end goal is sampled proofs through recursion with hidden unit circuits and a hidden partition; the auditor
  checks only padded counts. One key per protecting party, drawn once at setup from the OS before anything else, public
  and a constant in V\*; hm96's pinned key stays only where a key is wired into a circuit, as a named exception. Salts
  are drawn from the OS, not expanded by a PRG.
- A circuit restatement of a verifier (VBridge's V\*) is proved in one direction, that the circuit accepting
  implies the verifier accepts; completeness stays a test.
- `hash-derived-key` (`Hm96Hiding` at the pinned key) stays a named assumption, and nobody works on proving the
  pinned key behaves like a fresh one.
- A table that tabulates a fixed function is a value, even with no program parameter; an operation's own
  definition holds only the hardware's measured tables. No protocol gets an exemption for public values: in development a
  proof may be replaced by replay, or verification skipped, as a labelled shortcut, never as the protocol.

## Before you finish

- Each fix you made is in a PR description.
- Each question you asked has its ruling recorded as a rule, or is still open in your checkpoint or report.
- Each friction note is a few lines, names its ids, and carries `severity: incident` only when something was lost.
