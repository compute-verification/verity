- 15:15Z Fri, proofs coordinator, SCOPE CHANGE. Read lanes/proofs/20261030T1506Z-report-ficsks-rec128.md and
  lanes/proofs/tools/rec128.py (FICSKS-REC) in the notes repo.
  (a) Your L can't exceed about 165.1: the deployed m = 35 cap stages 16 V* statements, not 10. The docstrings' 165.8
      is the 10-statement figure.
  (b) Tight λ₂ alone gives 2^-125.1 at t = 2^40, which misses 2^-128. κ binds: 2^31.27 expected reruns x 16 x 2^-203
      = 2^-167.7, and lin2_bound_le multiplies it by t.
  (c) NEW GOAL, route 2: restate the bound as κ + t·ε, with κ charged once per session and ε per call. It needs no new
      premise, and RecursiveKS, κ and γ stay unchanged. Three pieces:
      1. refactor lin2_bound_le, or add a sibling, so κ isn't multiplied by t;
      2. a corollary beside bitsLin2On_of_ksLin2On: Pr[acc ∧ ¬R] ≤ κ + t·ε;
      3. a deployed lemma at the deployment's own sizes, with ε ≤ 2^-172.5 or so (Fdev's term, which sets ε).
      FICSKS-REC computes 2^-139.3 at t = 2^40 and 2^-134.9 at t = 2^44.39. At today's caps in
      lamLin2_deployed_ge_dep (setup ≤ 2^56, v ≤ 2^55) route 2 gives 2^-114.2 and fails, so the lemma must take the
      deployment's actual sizes as hypotheses. State them, and say in the PR how they're met.
      Keep the tight λ₂ ≥ L theorem: its term evaluation is most of route 2's work.
  (d) Docstrings: also fix the claim that KSLin2 at the dense statements rests on open premises (PartChecksBV,
      PartOpensB, DecompB). RecursiveKS is proved with none (KSProved; the lock lists it with none).
  Deadline: PR by Sat 31 Oct 18:00Z still, or tell me the new date by your next checkpoint. Don't list anything in the
  lock (a claim, Daniel's); recommend in the PR.
- 17:17Z Fri, proofs coordinator: vy-nebius-2 STOPS at 18:00Z (infra). Launch nothing new there; use cpu-1/3/6/11, never cpu-5. If a run of yours on vy-nebius-2 won't finish by 17:55Z, cancel it and relaunch on a cpu node now (its run copies are kept).
- 17:55Z Fri, proofs coordinator: lean's check.py has a Nat.gcd prefix allow-list bug (LEAN-141, fixed Tuesday). Until then, for affected packages, commit the --update record (after the usual check) rather than rerunning plain records, which fail on it.
- 18:10Z Fri, proofs coordinator: two computations of route 2's recursion bound disagree. At t = 2^40, FICSKS-REC (lanes/proofs/20261030T1506Z-report-ficsks-rec128.md, tools/rec128.py) has 2^-139.3, and FICSKS (20261030T1801Z-report-ficsks-weaker.md) has 2^-134.95. That's ~4.4 bits; pooled world-year 2^-93.7 vs 2^-89.4. Reconcile explicitly in your PR body: which terms each counts (the ε per call, κ's reruns, the link term Q_s), and which your Lean proves. Quote only the proved figure. Also: κ_MT's quadratic coefficient is 3/2, not 32 (FICSKS correction; no figure moves).
