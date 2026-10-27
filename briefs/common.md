## How you work (every soundness worker)

- You are a worker of the **soundness** coordinator (bc-d2745108-9b89-5c53-8137-de8e75809c4d), lane `soundness`, in the
  Verity repo (compute-verification/verity). First run
  `git -C /workspace fetch -q origin main && git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace && export PATH=/workspace/.git/verity-auth/bin:/workspace/.venv/bin:$PATH`,
  then reread `git show origin/main:AGENTS.md` (your snapshot can be older than main) and the lean-proofs skill
  (`.agents/skills/lean-proofs/SKILL.md`).
- **Branch:** create a new branch `cursor/soundness-<topic>-<4 hex>` from the base your brief names and push only to it,
  never to the launcher's branch, a PR head you didn't create, or `main`. Push after every commit.
- **Inbox:** at every checkpoint run `git fetch -q origin cursor/inbox-soundness-9c4d && git show origin/cursor/inbox-soundness-9c4d:inbox/<your bc id>.md`
  and act on each new section (it may not exist yet).
- **Note:** write `~/.research/notes/lanes/soundness/<UTC YYYYMMDDTHHMMZ>-report-<topic>.md` (front matter: id, campaign
  one-theorem, lane soundness, kind report, status open, repo compute-verification/verity, origin with your bc id) before
  long work, and add a one-line checkpoint at least every 60 minutes. Publish with
  `cd ~/.research/notes && research notes sync --path lanes/soundness` (last line must say `pushed`).
- **Linear is the status.** Post your exact Lean statement as a comment on your issue as soon as it's fixed (namespace,
  name, full type, module, the named assumptions it takes), and one comment per state change with the run id and its
  verdict line, or the PR. If you can't comment on Linear, put it in your note and your final message.
- **Heavy work never on your VM** (16 GB, shared). Lean edits: `verity lean try FILE --as soundness` (run from your checkout; it ships it itself). Since Tue 01:25Z prefer `--on vy-nebius-1 --pool research` (an older client without `--pool research` was refused there, r20261027-012722-2c01; 6 warm build slots since 01:44Z, shared with compute-research; tries only there, records runs stay on cpu-1/cpu-3; lent by compute-research until Tue 20:35Z; idle); then `--on vy-nebius-cpu-3` (3 trees); default cpu-1 (21 slots, but stalls on its memory ledger); cpu-5 overloaded; not cpu-4)
  (warm build slots; or `--on vy-nebius-cpu-3`; never cpu-4's soundness pool, whose extra trees are cold). Records: `research run --on vy-nebius-cpu-1 --pool research --project verity --source . --cwd clone --declared-output 'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records PKG --update`.
  Suites: `research run --on auto --cwd source -- bash -lc "uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py --changed <main sha> --quick"`.
  Never cpu-5 for warm copies; never a bare `rm` of data you don't own on a node.
- **Never block more than 20 minutes in one tool call.** Launch runs detached; when you'd wait on one, end your turn
  naming the run id and its ETA, and the coordinator resumes you.
- **PR:** when the branch is done, `research queue open --head BRANCH --sha SHA --title T --body-file F --by soundness`
  with a line `Lane soundness` in the body, then (only after its Lean check and quick suites pass, and only if this brief
  or your inbox says you may) `research queue ready N --mark --by soundness`. Don't push to a PR whose head is in a train.
  Before marking, `git grep` across `next` any name you delete or change.
- **Rules that bind the statement:** one soundness theorem, with no hypothesis on the circuit file or the statement's form;
  the verifier refuses any form the proof doesn't cover and gains no form ahead of its proof; assumptions are named Props
  listed in the lock; no `sorry`, `axiom` or `native_decide`. Never weaken a statement to make it provable: if it's false,
  stop, say why in your note and final message, and propose the restatement.
- **Final message** (at most ~15 lines): branch and head SHA, PR, run ids with verdict lines, the statement as posted,
  and what's left.

- **Names (Daniel, 11:09 AM PDT):** 
The composed protocol (onsite) is made of **six subprotocols**: sampled proofs, PoUW, PoUS, the bandwidth audit, the
network certifier and the wipe. Use those names only, with no aliases ("integrity", "work", "space", "communication"),
and **"leg" is retired** (in Linear: project, milestone, issue). The theorem's subprotocol targets are named
`SampledProofsSound`, `PouwSound`, `PousSound`, `BandwidthAuditSound`, `NetworkCertifierSound` and `WipeSound`
(ONETHM, `cursor/proofs-one20-48ae`); `Verity.soundness_of_subprotocols` is the composition (#4613). Record keys,
wire strings (the verdict keys `integrity`, `communication`) and `Core.Protocol` keep their names.
- **One adversary form, and a guard against vacuity (Daniel, 2:00 PM PDT):** every bound is for every prover τ, in the
  reduction form `Pr[accept ∧ claim fails on E^τ's values] ≤ δ + f(Pr[B^τ outputs a SHA-512 collision])`, with E and B fixed
  constructions; `ecr/sha-512` is asked only of B. Every guarantee gets `<name>_honest` (hypotheses hold at the honest
  deployed instance) and `<name>_nontrivial` (the conclusion fails for an accept-everything verifier).
