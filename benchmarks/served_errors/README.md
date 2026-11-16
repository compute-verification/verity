# served_errors

Some of the guarantees we quote have error terms that no Lean instance or run gives a value for. This benchmark
evaluates each of those terms at the sizes we serve, beside the guarantee it sits in and what its formula rests on;
evaluating a formula discharges nothing. `run.py`'s docstring says how to read what it prints, and lists the sizes and
where each parameter is read.

## Try it

~~~sh
uv run verity bench served_errors
~~~

It needs only the checkout and takes under a second. For each term it prints its value, its tag and the sizes it is
evaluated at, and under it the guarantee, the formula and, in brackets, what the formula rests on. `--json` prints the
same as JSON.

The terms:

- one Flock `--zk` session's knowledge error (`NumbersDeploy`'s numerals), which the PoUW, Sm120 and integrity rows read;
- PoUW's knowledge term δ_ks at Pearl+'s sample of 459 tiles, the served K_d (and at a docstring's example K_d);
- PoUW's period binding δ_bind at κ = 0 over every operand element the served run's tiles read, and the PoUW subprotocol's p of the
  remote monitoring period at the window's δ and at the served run's δ;
- the Sm120 headlines' n_π·δ_ZK (`ProofSoundAt`, proof soundness) against their 2^-41 window;
- Pearl+'s ε_a, δ_a and composed ε as served, credited at the H100 record (the gap held to
  `security/Proofs/Pouw/PearlC/ServedH100Rev1Gap.lean` by the suite);
- the recursive audit's error for one proof (`RecursiveAuditC3`): its sample's δ_s by design, its knowledge term `ksAt`
  on and off the deployment's tables, and its binding term, which has no value;
- Dither's κ_S, the outer proof's κ + t·ε, a proof and a replica-week, against the design point's budget;
- the integrity subprotocol's ks_t at the workload card's floor and at one request, and its sum over hourly windows;
- the remote product's joint error a block, statistical and cryptographic (every term at one budget), and the latter over
  a period.
