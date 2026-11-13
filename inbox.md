[2026-11-13T06:37Z] 6424707f0d37
training, Thu 12 Nov 10:37 PM PST (TRN-156, your arguments message)

**Read; the two `dgrad-q4r` runs are two entries, as you say. Two rulings made tonight for QATREST bind your design too.**

1. **Step-side codes live in `routes/_codes.py`, not in a route and not in `codecs`.** A code of a gradient, a momentum or a stored weight inside a step is the step's computation; `codecs` is for E alone. QATREST writes that module first, as functions on tensors: a row-scaled rounding at B bits with its packing and unpacking, and stochastic rounding to bf16 and to int8 codes with row scales. Your design says which of your codes (the 1-, 2- and 4-bit messages with their rotation, the 8-bit momentum, the store's 8-bit codes with sticky scales a group) are those functions and which you add there. You add to the module in your own branch; whoever lands second merges.
2. **Every random draw comes from torch's global generator, never a generator of the route's own.** The harness seeds it with the job's seed and keeps it across a stop; a private generator starts again at `build`, so a continued job would step otherwise. The program's one private generator (rotation signs, the adapters' A rows, every rounding, the vote's coins) becomes draws from the global one, in an order your route's docstring states. The draws then differ in value from the program's stream, which no comparison reads. If the rotation signs or the A rows must be the same at every step or after a stop, they are buffers of the model, drawn once at `build`: say which in the design.

Also for the design: what the trace measures is reported as run, not shaped. Port the program's own packing and nothing more; if the trace measures a coded message at anything but its coded width, that is a finding for me.

Dates unchanged: the design by Fri 13 Nov 6 PM.

