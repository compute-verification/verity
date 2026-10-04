"""Utilization: lower bounds on what the prover must have used between a challenge and a clock's reading, whatever its
inputs. `work` bounds compute (work keyed by a salt, found before the clock is read) and `space` bounds memory (blocks
of an encoding, answered before a deadline). Each rests on a hash or a hardness conjecture, and on a clock: whoever
holds it decides how tight the bound is, so the verifier names whom it trusts for each reading."""
