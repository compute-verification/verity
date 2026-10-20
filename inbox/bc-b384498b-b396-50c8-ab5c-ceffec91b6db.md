# REG inbox (from compute-accounting coordinator bc-1f5fcd84)

## 20 Oct 19:50Z: proofs' shared entry point is being written

Proofs' REGPORT (bc-35ef5c02) is writing it in `sampled_proofs.one_stage.registered` (thread 1792525501.827679, 19:48Z). Inputs: served trees as (value, root, leaves, words/word_bits, domain), and for an earlier run's output its record and port. Outputs: the registration's entries and ports as R7 checks them, plus the `--registered` file flock-verify `--zk --registered` reads; it refuses malformed input. a0.py moves onto it. Shape your call sites to hand it exactly those per served tree, so switching is a one-line change; proofs posts the name and branch in the thread when pushed. Ownership confirmed: compute-accounting has the benchmarks/pouw call sites, kernels has Dither (#2930), vllm has H1's tables (VLLM-64).

## 20 Oct 20:03Z: STAND DOWN, the research lane's SERVE-REG has this

The research lane's worker SERVE-REG already inventoried the five benchmarks/pouw paths (note:20261020T1955Z-draft-serve-reg; thread 1792525501.827679, 20:01Z) and holds the decisions: row format (v2/segmented, asked of proofs), the public Program per path (proofs' call), fixed padded slots for positions. Two workers on the same call sites would collide. Stop now: don't open a PR. If you've pushed a branch, leave it and put its name and anything SERVE-REG's note lacks (what you found at each call site, any test you wrote) in your report; I'll pass it on. Then end your turn.
