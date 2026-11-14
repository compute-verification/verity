[2026-11-14T08:11Z] 8afd224dc634
TRN-166 ROUTES, Sat 14 Nov, 12:11 AM PST.

**#9964's fix head is pushed: bebb06fe85c3f8b9ae3cad88daea62f6aa1df956.**

- The must-fix: the three QAT kinds train blocks.1 over a frozen blocks.0, and all three pass (RAN locally).
- Should-fix 1: `test_every_config_of_the_library_takes_a_kind_of_step_a_job_takes`, with the rule in `synthetic.kind_of`. All 64 configs map to a job, and no two jobs share a kind (RAN). With two kinds' jobs removed in a scratch run, the test fails naming s-lion and s-qat-top3-grad4, each with the kind its job would need.
- Should-fix 2: slow marks on the six new cases over 5 s and on the three QAT cases, which are now over 5 s too. That makes 9 marked.
- Your ruling 4: the products table now reads the optimizer from what the route builds, so grow's Muon is checked too. Every job's approx_macs is checked against the trace's.
- The quick tier launches next, and its id follows in my mark line.

