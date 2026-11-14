[2026-11-14T22:57Z] e7a2e6e03c01
From the proofs lead, 23:00Z: one more one-sided keyword: in the one-stage suite, test_spot_lean_names_the_records_digest_and_refuses_v1_by_name fails at the tip with TypeError: v1_case() got an unexpected keyword argument 'design'. Find which map renamed it on one side (yours or another lane's: say which), mend it. (The Rust fix at flock-circuit.rs:1019 still comes FIRST.) Then re-run your command.

