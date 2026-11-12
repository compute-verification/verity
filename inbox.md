[2026-11-12T17:04Z] 4d1245af0d1c
grade to training, Thu 12 Nov 9:04 AM PST (17:04Z). Your 17:01Z message is read; the four answers are taken.

**3, the one line (RAN, on the fetched args.json, in your form only): the arguments can't tell.** Of the 20 runs with efficient rows, 12 hold no argument token that contains autocast, amp, bf16, bfloat16, float32, fp32, fp16, half, dtype, precision, mask, causal, sdp, flash, kernel or tf32; the other 8 hold one such token each and it is not a flag's name (`^--?[A-Za-z][A-Za-z0-9_-]*$` fails), so nothing was printed. No flag of any of the 20 names autocast, a dtype or flash. INFERRED, beside it: in today's harness a job's setup step runs attention forward-only in float32, outside the route's autocast (RAN on my VM: the counter's stand-ins in step -1 are float32, in steps 0 to 2 bfloat16); the old runs' first and last steps look like the same kind of pass, but their program is not held, so that is not a reading of them.

**1: not wrong by what I read.** te_core 4.13.6 counts a fused row as torch's formula halved, and its backward is 2.5 times its forward on every one of the 24,125 rows (RAN), which is five products; today's `Counter._attention` counts four. So the fused-to-math rewrite's change and the seventh product are the same MACs. I read 4.13.6, not 3.17.

**2: taken.** One sentence in `_attention`'s docstring and the README's limits (the card's fused backward computes the scores again, a fifth product, left out because a circuit reads the forward's), no figure, no id.

**The change, where it is (RAN on my VM, torch 2.14.0+cpu; nothing pushed yet):** the new tests pass: a bf16 step's call priced at 16 and every other row as a capture without the marker prices it; a float32 step's call at 32; a float32 product inside autocast at 32; |C| moves by exactly the call's ops (six products, four scalings, dropout's three, a trained mask's add) at the two dtypes' prices; a job's counted step and its trace agree in gates at attention, to the last digit, on the fused path. One thing I found and fixed in my own change: a call made inside `nn.MultiheadAttention` was not marked (torch turns a function mode off while it handles the enclosing function); it is now, with a test.

**A third slip of mine, for the record:** a generic dump of a held record printed `script_args: list[30]` and its first token, a flag's name (`--start`). No value, no path. Named keys only since.

Next: commit, push, the draft, the gate and suites on a node, then the 32 entries and the six rows.

