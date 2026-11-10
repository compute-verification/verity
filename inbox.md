[2026-11-10T02:09Z] 806b55ac40b0
From the circuits lead, Tue 10 Nov 02:10Z. One check to add to #8426 and to CIR-202's cut, no change of plan otherwise.

A served unit whose rows are all 111 B or less cannot be composed: `compose` refuses it (`verity_flock/circuit.py`, "no row has a compression before its last block"), and so does `verity.ml.windows.interface.check_proof_block`. circuit-check's block check passes such a unit, so it does not tell you. CIR-204's served slot unit (rows of 64 B, 4 B and 4 B) was caught this way by vllm an hour ago.

For every served unit your PRs add or re-bind (the greedy TokenSelect cut's tree nodes especially, with their 16-bit outputs): run `interface.check_proof_block` at the served statics, hold it in a test in the PR, and put the result in the body. If one is refused, bind it larger or cut its ports fewer ways, and say so on its issue.

