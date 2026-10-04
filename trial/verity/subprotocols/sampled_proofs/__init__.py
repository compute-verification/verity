"""Sampled proofs: draw units from a committed list of executions and verify each drawn one, by replaying its trace or
checking a proof. `audit` (the audit and the `IntegrityProfile` it establishes), `replay`, and `sp1` (heavy, its own
distribution). A construction supplies only `verify_unit`."""
