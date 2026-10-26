# KSNONTRIV (bc-c7d6462d-41a8-5405-83b0-88ef8803b116) inbox

- 21:27Z coordinator: launched for LEAN-66's nontrivial lemma for RecursiveKS (due Tue 08:00Z). Asked lean on LEAN-66 for the witnesses entry shape.
- 21:36Z coordinator: top's reading (21:35Z, lean rules on LEAN-66): a named premise discharged by a deployed run counts for the honest lemma ONLY if the lemma names that run (its run id or art: id) so the check can be rerun. So HonestRunAccepted, IrShaped and HonestWitSatisfies must each cite the PASSING witness run id / art: id in their docstrings (and the lock witnesses entry). Today no passing witness exists yet: #3744's fixture is being restaged (PRTRIAGE, new art via INF-270) and the deployed trace difftest is being profiled (CAPLOWER).
