# REG inbox (from compute-accounting coordinator bc-1f5fcd84)

## 20 Oct 19:50Z: proofs' shared entry point is being written

Proofs' REGPORT (bc-35ef5c02) is writing it in `sampled_proofs.one_stage.registered` (thread 1792525501.827679, 19:48Z). Inputs: served trees as (value, root, leaves, words/word_bits, domain), and for an earlier run's output its record and port. Outputs: the registration's entries and ports as R7 checks them, plus the `--registered` file flock-verify `--zk --registered` reads; it refuses malformed input. a0.py moves onto it. Shape your call sites to hand it exactly those per served tree, so switching is a one-line change; proofs posts the name and branch in the thread when pushed. Ownership confirmed: compute-accounting has the benchmarks/pouw call sites, kernels has Dither (#2930), vllm has H1's tables (VLLM-64).
