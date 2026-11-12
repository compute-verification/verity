[2026-11-12T18:06Z] ad78b9e062b3
TRN-145 (PRICER), the walker's shape before the tests. Two places where the code disagrees with your decisions, and one correction to my round 4b sizing.

CORRECTION. My round 4b count of lf's flattened step, 1.51e10 words, was wrong: it multiplied a batch node's outputs by its members twice. RAN now: lf's tiny step computes 1.33e7 words flattened (1.13e7 of them F32Fma, the multiply-adds), against 2.25e6 counted once a body. The 83 s is evaluate_batch running F32 words one at a time on the reference; no numpy kernel is registered for F32Add, F32Mul or F32Fma.

LEVEL (decision 2: the code disagrees). A body walked once at random parameter words sees a cancellation written inside that body, but not one written through a call. Two examples: W + (-1) * LinearRef(W, I), and M + f(M) with f a composite that negates. The F32Add is a shift, so the reading keeps W's rows distinct whatever the call returns, and a call output drawn fresh hides the identity. That is one spelling's check, which you ruled out. Refusing every spelling needs each call's output at the caller's words, so the walker evaluates the Program top-down along the reader's own summary tree:
- a batch node's members are vectorized as lanes;
- a scan runs only the iterations the reader reads as distinct contexts (2 to 3), never its n;
- each F32 primitive is a numpy twin, which a test holds to the reference;
- any other primitive falls to evaluate_batch, and the walker counts its words.
That is about the step's multiply-adds: 1.33e7 words for lf (seconds, estimated). For s-t20 it is about 6 P T, roughly 2.2e11 a 1,024-token sequence (minutes on numpy, estimated; the node run will measure it). A scan's output past those iterations is not evaluated, so a claim downstream of a scan goes unchecked and is counted. Memo: one pass a Program, keyed by its root's digest. The pass is the same at every k, so the reads at each k and the 2- and 3-iteration loops share it.

ALIASING (your 2). It comes for free: words flow through the views, so f(b, b) reads one array twice, and M beside swap(M) holds the same words. There is no alias-pattern key and no pattern count, because nothing is drawn below the root.

WHERE THE CHECK IS, AND THE POINT (decision 3: one addition). The check sits where a claim is consumed: at each product operand that the reading leaves charged only because its rows are distinct. That is an operand that is not gathered, not specific and reads no current input, a carried θ′ weight. The operand's words are cut into slices of the product's K and hashed across points. The claim is that its rows differ for every input, but the falsifier needs inputs that repeat. prev = looked-up rows + M (*) swap(M) has equal rows only where the tokens repeat, so a generic point never refutes it. So the point is SHA-512 of the root's id:
- θ's and registered words are finite floats (or ints) at their own width;
- every input repeats its outermost rows: all tokens are one token, and a float input's rows are one random row;
- a scalar input is drawn generic and nonzero.
Then W + z + (-1) z, a one-word multiplier, fitted inputs and a test of the circuit's own weights are not refused, and both families are refused. What it would refuse beyond the families: W (*) (z_r - z_r') for two rows of one input, which is 0 where the inputs repeat, as the reading itself allows they may.

FALSE REFUSALS (your 4). Rows are equal only where they agree at P points, P = ceil(64 / (K * b)), where b is the least width the operand's words can hold: a rounding's output width, or its source's. Each point is a lane. lf's float32 rows with K of at least 16 need P = 1; a 4-bit row of one word needs 16.

THE REFUSAL. A ValueError subclass in work.py, as Ambiguous and NotSeparable are in measure.py and ops.py. It names the Definition, the body node, the operand and the two rows' indices.

I build it this way unless you say otherwise; the draft's head comes by 5 PM.

