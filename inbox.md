[2026-11-20T00:10Z] 72f2ee62e661
FROM top (id walls-1), Thu 19 Nov 4:10 PM PST. Daniel agreed three additions; they change WHAT YOU SHIP, not your order (the first census table is still first, in your note).
1. Your row check IS two named open premises of the chain, run: `LowerEval` and `BlockEval` in `security/Properties/Flock/Circuit/Chain.lean`. Read their statements and say in the tool's docstring, in one sentence each, which of your two checks stands for which, and anything they say that your check does not test.
2. DOOR-READY evaluates the rows of EVERY template it is given (honest output satisfied, each wrong output refused), once per template digest, not only in the fuzzer. Report it as a column of the census table (rows evaluated: yes / refused honest / accepted wrong / too large, with the size).
3. EVERY table line names the sha256 of the descriptor file it read and the `one-circuit` commit it was built at.
4. The fuzzer keeps each finding as a seed in its test file, and its test is cheap enough to run in the suite on every change (say its seconds). Say in your final message the command that reruns it at length on an idle node.
Also: lean's second fault (3:33 PM PST, TOP-444): a template parameter or return field that is not a multiple of 16 bits, followed by another field: honest rows fail, a wrong output passes (confirmed at 3 bits). Reproduce it in the fuzzer as you do the first; `lift_shape` and `odd_fields` in `verity/circuits/codec.py` (circuits, merged 3:39 and 3:56 PM PST) are scanners you may call. A refusal for it is being written tonight by lean's worker; if the tip has it when you run, say the commit.
Re-run your command.

