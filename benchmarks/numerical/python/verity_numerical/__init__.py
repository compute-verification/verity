"""verity_numerical: the numerical relation of ``verity.ml.circuits.targets.target.FIRST``, campaign-side.

Shared by the proof backends and owned by the campaign, not by ``verity_sp1.proofs`` (which owns *what* must
be established, not a proof decomposition).  Subpackages, each owned by one lane and landing on ``main`` through
the coordinator:

    reference   WP1: the source-close evaluator (Hawkeye's simulator at verity/ml/circuits/silicon/hawkeye/PIN.json, as defined
                integer operations), the independence audit of the existing columns, the exhaustive test plan
    checker     WP2: one reference checker for the transition unit and the VU, witness generators, range lemmas,
                negative tests and the adversarial (exhaustive / SMT) search for accepted-but-wrong outputs

Backends (``archive/gkr``, ``archive/direct``) consume these; they keep their own gadget layouts.
"""
