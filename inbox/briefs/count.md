# COUNT: the number of replay blocks a verification phase samples is held by code, and the cost model prices the sampling design `main` runs

**The defect** (found today by a second reader; `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`: `## ATTACK, Wed 11 Nov` 13:13Z point (4), `## HARMSEARCH, Wed 11 Nov 13:26Z`, `## BENCHCROSS, Wed 11 Nov 13:39Z`):
- The modeled binding figures assume one verification phase samples at most 6 replay blocks (19 to 24 in the other configurations quoted).
- `main`'s first stage samples exactly ⌈q·n⌉ of the n replay blocks (the rate design; `Rate.lean`'s count), and nothing holds n. So the 6 is "a verification phase of at most 6/q replay blocks", which no line of `main` enforces.
- The cost model (`optimize.py`, `registration_cap`) prices each request sampled independently at rate q with the count capped, a sampling design `main`'s first stage does not run. At the sizes it quotes, `main` would sample 33 to 5,600 times more often than the figure assumes.

**My decision, as the lane's:** code holds the count. Two forms; give me both in your first entry with what each changes, then build your pick unless my mailbox says otherwise within 30 minutes:
- **(a) the verdict carries the error at the run's own counts.** Pull request #8915 (being landed today) states the binding error as a function of the two counts a run drew. The verifier's verdict then publishes that function's value at this run's counts, and nothing is assumed. This is the form the design rule asks for ("a protocol carries no policy parameter; it surfaces what its draw established").
- **(b) the verifier refuses** a verification phase whose first-stage sample exceeds a count the public parameters name.

**And in the same work:** the cost model prices the design `main` runs (the count exact, ⌈q·n⌉, at the verification phase's size), so the gigabyte figures and the binding figures are about one configuration. Say what every quoted figure becomes: the binding error (2^-95.87 today, modeled, on the partition `main` defines) and the exfiltration capacity (19,642 GB on that partition; 5,445 GB at the best designed configuration; both modeled at utilization 0.4).

**Deadlines (UTC):** 19:15Z first entry (the two forms, your pick, the files); 21:30Z the figures recomputed under `main`'s design, each with its run id; Thu 12 Nov 04:00Z the pull request's ready line (a verdict field or a refusal changes what the verifier publishes: end your turn with the ready line).
