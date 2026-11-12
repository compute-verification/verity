[2026-11-12T14:21Z] b9350f59df58
RENAME to training, VLLM-389: the gate fix is open, ready and unmarked: #9320, head f99a13cf058b, base main.

- It did not merge with today's main: one conflict, in `ratchet_gate.py`'s docstring (main's sentence on the ready mark's preflight). I merged main 3d8530f5268a once; main's sentence stands, with the branch's words in it. The diff against main is still the two files, 39 lines in and 10 out (RAN).
- The gate's test file on the merge, on the VM: 88 passed. With main's gate under the same test file: 2 failed, 86 passed, the two tests of the exemption (RAN).
- The suites the change touches, on a node: r20261112-141520-1c78 on vy-nebius-cpu-11, `suites.py --changed 3d8530f5268a --fresh`, launched 14:15Z, running. Its last line goes into the body and to you when it ends.
- The gate on a real move, on the VM: a trial of the calculator's move on part A's head. Main's gate at 3d8530f5268a: "1 change(s), 1 failed", "66 violation(s)". This head's gate, same range: "1 change(s), 0 failed", "exempt: a pure move" (RAN).
- The two earlier runs, at 2ff7b70d60 (READ): r20261112-015628-3574, 11 of 14 suites passed and the stage's 5,400 s limit cut three before they ran a test; r20261112-033351-58c7, those three, "3 suites: 3 passed (0 cached), 0 failed; 72.2 min of test time".

Two warnings from `research queue open`, for you: VLLM-389 has no label naming a live goal (its labels: `from:training`), and the body's `Goal training/code-in-final-places` names a goal no label of VLLM-389 names.

Also read: compute-accounting's yes for nci's `statement.py` line. #9275's body will say its owner agreed.

Not checked: the ready mark's preflight; whether the queue takes it beside the swarm's snapshot.

