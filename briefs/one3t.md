# Brief: ONE3T, one-stage v2's accept gives the sampling verifier's run `.acc` (ONE-3t), Linear PRF-209

**Due:** statement on PRF-209 within about 1 h of starting; proved and marked by **Tue 27 Oct 18:00Z** (dates are ceilings).

**Why:** the network certifier's bound (`window_le`, network-accounting's NET-85) is `PrBad … D.V` over `Claims.Sampling`.
For it to hold at the compiled verdict, an accepted one-stage window must mean the sampling verifier's run accepted. This is
the one-stage twin of ONE9P's `acc_of_accepts` for recursion (PRF-202's 19:04Z statement and split: read them, the shapes
match). Proving it keeps the network bridge's claim as stated.

**Statement to prove:** `verity/sampled-proofs/one-stage/v2`'s accept (#4637, `SampledProofs.OneStageZk.checkBlobs`), given
the blobs show the run (records, rep files, statement) and the live-coins premise (`live-verifier`, until ONE-4 PRF-200
lands; then the coins come from ONE-4's signed log), implies for each drawn session
`(Claims.run sha512 P (SamplingDeploy.deploy rel c).V x ω r).out = .acc` at the statement the session's files give.
Build on ONE-3p's `Properties.SampledProofs.OneStageZk.AcceptMeans` (PRF-206, worker ONE3P bc-fffe15fb, branch
`cursor/soundness-one3p-fdea`): it gives `AcceptsZK` (`Flock.Zk.verify`'s event) per session. The remaining step is
`AcceptsZK` at live coins ⇒ the model run's `.acc` (C-Flock's own `verify`-to-model bridge: look in `Security/Proofs/Flock/`
for the existing refinement that `SampledKSLive*`'s `LiveAcceptsZK` uses, and reuse it rather than redo it).

**Base:** create your branch from `origin/cursor/soundness-one3p-fdea` (`git fetch origin cursor/soundness-one3p-fdea &&
git switch -c cursor/soundness-one3t-<4 hex> FETCH_HEAD`). Add a new file under `Security/Proofs/Properties/SampledProofs/`.
Open your own PR (stacked on ONE3P's when it's open) with `Lane soundness`. Post on PRF-209 and tell network-accounting's
NET-85 its name (a comment there). Names follow Daniel's 11:09 ruling: "sampled proofs", never "integrity" or "leg".
Lean runs: `--on vy-nebius-cpu-4 --pool soundness` (until #4794 lands, add the pool to `slot_only.POOLS` locally,
uncommitted, or use the `CHECK_SLOTS=/workspace/research/locks/soundness/slots` wrapper).
