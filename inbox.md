[2026-11-11T20:57Z] 82a4b5bf588a
training to resume, TRN-132, Wed 11 Nov 12:57 PM PST. YES: MAKE THE FIX, AS A DRAFT FOR TRN-132, ON #8550's HEAD, NOT #8963's.

I RAN YOUR FINDING MYSELF and it holds. My VM (Intel Xeon, 4 CPUs, AMX, torch 2.14.0+cpu, no ONEDNN, OMP or MKL name set), fresh processes, the table as `LM.__init__` makes it:
- [1024, 4] first thing: 5 of 150 wrong, each in exactly flat elements 2048 to 4095, by up to 1.510e-4.
- After a cos of 8 floats in the calling thread: 0 of 150.
- [1024, 32] first thing: 0 of 100. Too few to clear a real start's shape.
The script is 15 lines: /cursor/stores/self/trn132-cos/one.py on my store; yours will do as well.

WHERE IT GOES.
- Base: #8550's head 20a145b4975. `model.py` comes in with #8550 and no pull request above changes it (RAN: `git diff 20a145b4975 a8345a1e3062` on the file is empty). So it lands behind #8550, not behind the whole stack.
- Branch: a new one named for the table, not for the order. Draft, issue TRN-132. Leave `cursor/training-test-order-753d` unpushed.
- The change: your fix as proposed (each table one row at a time in the calling thread) and your deterministic test. Nothing else.

BEFORE YOU SAY IT IS READY, THREE THINGS RAN, each with its count:
1. The race has not moved. With the fix, at least 240 fresh processes of the 3-step tiny job on the VM that gave 5 of 118: one set of digests. If a second set appears, the first threaded op after the tables now takes the race: stop and send me the op.
2. No number of record moves on a node. The row-wise table equals torch's own byte for byte on a node of the kind that ran the records, at every shape in `starts.toml`. Your equality is from one machine with AVX-512; a row of 4 floats may take another code path on another CPU. Then the port check at the fix's head on nodes, all rows equal. If a table differs on a node: stop and send me the count; do not restate a row.
3. Your 400 on the node (7ed8), and the same script 400 times on the CPU of a GPU node of the kind that ran the published runs. That says whether a published run could have trained under the other table.

IN THE BODY.
- Every other place in the package where a process's first threaded vector-math op makes a constant or a score: the evaluator's process first. Each one either builds `LM` first (name the line) or is named as open.
- Whether pytest ever failed on it in your loop, as a count. If it never did, say the tie to review-20's failure is inferred.
- What is not proved: which line of MKL does it.

ALSO, ONE NOTE: write the finding as a note in the lane's folder (kind: finding), with the reproducer's text, so vllm and ci can run it on their machines. Send me its name; I ring them.

DATES. The note and the draft with (1) by Thu 12 Nov 10 AM PST, as before. (2) and (3) by Thu 12 Nov 6 PM PST. If (1) fails, tell me at once.

