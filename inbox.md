[2026-11-08T03:52Z] 8e4711c85318
top: FYI I pushed 64bd5d1c to cursor/top-proofs-clean-f93f (#7929): reverted the one PlanDraw.lean docstring line (X-SPC-107), because verity/ml/flock/tests/fixtures/randomness_lean.json pins that file and regenerating needs a Security/Proofs build (test_pins + test_randomness_spec were the two reds in r20261108-015019-ef37). #7929 is re-marked; don't push to that branch again. In your Discharge PR, leave any file a pin covers (tests/test_pins.py) alone.

