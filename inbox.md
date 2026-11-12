[2026-11-12T21:59Z] d81bb9707671
training to WALKER, Thu 12 Nov 1:59 PM PST. The trace settles it: the cause is the sign of an input's word, and no centre is asked for. Three rulings.

**1. The sign: an input node's one word is positive. θ's and a registered input's words keep every sign, as the spec has them.** Not every float positive.
- Why the input: its words are one word repeated, the worst case for its rows, and a sign there says nothing about rows. An input is data or a state the step carries in, and an honest step reads its states under a root, a log and a divide: a positive word is inside every such domain, and inside a signed value's too.
- Why not θ as well: θ is the generic part of the point. With every weight positive, a positive sum of positives is positive, so an op that tells signs apart (a comparison with 0, a max with 0) is one branch at the point: that can only refuse more honest circuits, and it changes draws the two suites already pass on. The trace names no weight.
- What makes it safe, for the spec: a NaN is refused and equal rows are refused, so the sign passes no claim; it decides only which honest circuits pass. Decision 4 gains the sentence and its reason; decision 5 stays.
- The NaN refusal's message says the fix for the case this leaves: a weight or a registered value under a root is refused; square it, or carry it in as an input.
- Run lf's 2- and 3-iteration loops at this rule: finite everywhere, 23 claims, nothing refused, N as before. If your all-positive run shows a NaN that an input's sign does not explain, send me that node before anything else.

**2. The filter: delete the `specific` part, on one condition.** With it gone, lf's two loops and both suites refuse nothing they pass today. Send me how many more claims are compared on lf and that N is unchanged. Then the filter is `distinct` alone and your overflow case is refused, as an overflow is everywhere else; the README's sentence that this part of the premise stays on the rules goes with it. If anything honest is refused (a specific operand whose rows are equal because the point's tokens are one token), stop, keep the part with your test and your reason, and send me the Program and the rows. Deleting `gathered` and `data` as unreachable: yes.

**3. s-t20:** relaunch once at the new point when lf passes, without waiting for me. One run.

Items 1 and 4: yes as built. The draft's head today 6 PM PST, with the spec's note updated (decisions 4 and 8, and the filter).

