# Design rulings

The standing rulings on Verity's protocols, claims and lints that `AGENTS.md` doesn't state, each stated as itself.
A new one is one present-tense line for its class of case, rewritten when a later ruling replaces it and deleted once
code enforces it; who set each one and why, and every retired ruling, live outside the repository (the decisions log).

## Protocols and claims

- Credit and every report of computation are in AND gates (tentative). 𝓜's resources appear only inside
  exclusion, which reports in AND gates. Whoever finds a reason this is wrong floats it at once.
- Credit of declared size, values certified, is a statement about values (sampled proofs'). Compute credit (Dither's;
  PoUW's bar is now T1–T3) also needs values that read live noise and SAK's lower bound, so SAK is a premise of compute
  credit as well as of exclusion. SAK is over noised operands only, never prover-chosen ones (pinned weights, served
  activations, un-noised outputs), and an un-noised call earns no compute credit.
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
  are for the whole node. PoUB treats the prover's resources as one pool (conservation, no identity
  assumption), claims 0.15% over a 10-minute monitoring period with every window at most 1%, and never changes the
  model's logits.
- PoUW's guarantees, and only PoUW's, may rest on named random-oracle premises for SHA-512
  (`StraightLineExtraction`, Compose's `SeedExtractedAt`), each a named `Prop` the lock lists. Flock's
  knowledge soundness keeps the collision form.
- The deployed recursion's bound record states its inner claim at IVStar at the weight
  `coverW`, the least at which the premise P3 (`Bound.WeightCovers`) holds. Its security level is below the 100-bit floor
  under every weight (`Claims.Recursion.lam`), and every table, README or claim that cites it gives its λ and says so.
- `Flock.Guarantees.RecursiveAuditHolo` may rest on `registrant-holds-openings`; in Lean the salt is a binder
  applied at the salts the prover's workers committed under, never `∃ salt` or a `Classical.choose` of one.
- A guarantee's checked family may fold in a check the compiled verifier already runs, or a new fail-closed
  refusal modelled the same way. A free parameter whose adversarial values break a proof is closed by a side condition
  decidable at any literal instantiation, discharged by computation at use, never a listed assumption.
- The end goal is sampled proofs through recursion with hidden unit circuits and a hidden partition; the auditor
  checks only padded counts. One key per protecting party, sampled once at setup from the OS before anything else, public
  and a constant in V\*; hm96's pinned key stays only where a key is wired into a circuit, as a named exception. Salts
  are sampled from the OS, not expanded by a PRG, but the lead keys its instances under one window key (`Hash.Keyed`).
- A circuit restatement of a verifier (VBridge's V\*) is proved in one direction, that the circuit accepting
  implies the verifier accepts; completeness stays a test.
- `hash-derived-key` (`Hm96Hiding` at the pinned key) stays a named assumption, and nobody works on proving the
  pinned key behaves like a fresh one.
- A table that tabulates a fixed function is a value, even with no program parameter; an operation's own
  definition holds only the hardware's measured tables. No protocol gets an exemption for public values: in development a
  proof may be replaced by replay, or verification skipped, as a labelled shortcut, never as the protocol.

## Lint ratchets

- The vLLM integration's lint ratchets are no-growth: no change adds an
  exception or raises a recorded count, and a change to a lint's own code or thresholds is judged like source. Dedicated
  cleanup lanes pay the exceptions down; a feature PR owes none (`integrations/vllm/tools/ratchet_gate.py`).
- A `# noqa` is an exception too: in `integrations/vllm`, no
  change adds more `noqa` lines than it removes. One that moves with its code counts as kept, and deleting a stale one is a
  retirement. The ratchet gate counts them, with no allowlist.
- Outside `integrations/vllm/`, the same gate holds `tests/lint/`'s no-growth lists and checks each PR's changed Python,
  as drive lands on `checked` before the suites run: fix the code, never the list; a rename moves its entries (`--move OLD NEW`); a
  list may catch up to what its base's lint finds in its tree, no further. Ruff: the root `ruff.toml`, per (file, rule),
  `noqa` uncounted. No new direct environment read: use the area's settings registry (a `settings.py`, or a list mark at
  the PR's base). No new swallowing handler: re-raise, record, or test first.
