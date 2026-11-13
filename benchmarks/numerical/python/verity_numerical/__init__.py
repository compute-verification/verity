"""verity_numerical: the Python the proof system's benchmarks share.  ``benchmarks/numerical/README.md`` says what a caller may
import (each module's ``__all__``) and what the package never does.

    bench       what a measured result must say about itself (``contract``), its interaction time, the census as Python,
                input sets, one measured cell on its pods, the subcircuit catalog and the per-unit-type cost inventory
    reference   the source-close evaluator (Hawkeye's simulator at verity/definitions/silicon/hawkeye/PIN.json, as defined
                integer operations): an independent reference for ``verity.definitions.circuits.targets.target.FIRST``
    checker     one reference checker for the transition unit and the VU, witness generators, range lemmas,
                negative tests and the adversarial (exhaustive / SMT) search for accepted-but-wrong outputs
    settings    every environment variable the package reads

``python -m verity_numerical JOB [ARGS]`` starts one of its command-line modules (``__main__``).
"""
