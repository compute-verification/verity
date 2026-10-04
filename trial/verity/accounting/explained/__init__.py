"""Explained: how much of each resource the stream claims, as far as verification backs the claim. `integrity` says
whether the streamed executions are what their circuits compute; `work` and `space` are lower bounds on the compute and
memory the prover must have used between a challenge and a clock's reading, whatever its inputs. Work keys the stream's
own computation, so integrity says whether the compute it bounds was the declared computation; space bounds the memory
holding an encoding of the declared data. Each rests on a hash or a hardness conjecture, and work and space on a clock:
whoever holds it decides how tight the bound is, so work's verifier names whom it trusts for the clock; space's is
always the pinger, since only a clock at the site makes the deadline tight."""
