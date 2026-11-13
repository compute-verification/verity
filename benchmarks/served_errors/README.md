# served_errors

`verity bench served_errors [--json]` evaluates, at the sizes we serve, error terms of the quoted guarantees that
have no value from a Lean instance or a run. Each row is a term, the guarantee it sits in, its formula, the sizes it
is evaluated at, its value (log2 of a probability unless the row gives a percentage), the premises the formula
rests on and a tag: `modeled`, or `assumed` where the value rests on a constant taken rather than bounded. Evaluating a
formula discharges none of its premises, and a row whose guarantee is open says so. A sum takes every term at one
budget, which the row states. `run.py`'s docstring lists the sizes and where each parameter is read.

The terms:

- one Flock `--zk` session's knowledge error (`NumbersDeploy`'s numerals), which the PoUW, Sm120 and integrity rows read;
- PoUW's knowledge term δ_ks at Pearl+'s 459 draws, the served K_d (and at a docstring's example K_d);
- PoUW's period binding δ_bind at κ = 0 over every operand element the served run's tiles read, and the PoUW leg's p of the
  remote period at the window's δ and at the served run's δ;
- the Sm120 headlines' n_π·δ_ZK (`ProofSoundAt`, proof soundness) against their 2^-41 window;
- Pearl+'s ε_a, δ_a and composed ε as served, credited at the H100 record (the gap held to
  `security/Proofs/Pouw/PearlC/ServedH100Rev1Gap.lean` by the suite);
- Dither's κ_S, the outer proof's κ + t·ε, a proof and a replica-week, against the design point's budget;
- the integrity leg's ks_t at the workload card's floor and at one request, and its sum over hourly windows;
- the remote product's joint error a block, statistical and cryptographic (every term at one budget), and the latter over
  a period.
