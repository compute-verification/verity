[2026-11-14T19:11Z] 8724e71586df
circuits lead, Sat 14 Nov 19:12Z. Three conditions from proofs join CIR-290's design (my comment on CIR-290 just now; PRF-618, 11:10 AM PST). Put all three in class_window.py's docstrings and in the plan:
1. **The first step's bit is a constant inside the first step's Definition,** not a row at the window's boundary: a boundary row would be one the prover fills.
2. **The select that writes the next bit reads the limits (`max_tokens`, the prompt's length) from the same committed values the length check reads,** so the chain and the check cannot be given two different limits.
3. **Once off, off:** the next bit is the AND of this step's bit with the rest. The live steps are then a prefix, and the length check's n is the count of live steps and nothing else.

The wire is an ordinary step-to-step wire, the writer one step before its reader, as the decode's state is wired today.
So the first step's select is its own template (or the bit a constant its Definition holds), and the limits reach every step's select and the length check from one committed source: say in the plan which rows carry them.

