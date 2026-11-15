"""The workload's Programs, which `benchmarks/verifier_cost/run.py` reads.

    model      the circuit: a tiny integer MLP over a batch of requests, a Boolean Program, and its integer reference;
               `build` picks the workload
    served     verity_vllm's serving Program (`Serve_v1`) for a truncated SmolLM2, derived and evaluated on CPU, with the
               MLP's interface
"""
