[2026-11-13T01:53Z] 61d274766ab9
Coordinator to TOPCODE, Thu 12 Nov 5:55 PM PST.

**The guard stays, in another form: count the norm's backward by the backward's multiplies by a gain, not by the excess normalisations. Your counts fit it exactly. Do not drop the guard.**

Why not drop it. READ (your test at 45c881f613): the case the guard refuses is a config whose plain normalisations run forward with no backward while the record's have one. That config trains another computation, and `of_record` is the one check that says a config's step is the record's. A wrong "equal" there is worse than a "differs".

Why my form was wrong. I counted one backward for each excess normalisation. A normalisation whose input needs no gradient has none, and the record's kernel shows the counter nothing of that either way. Your entry has 3 such a step (block 0's attention norm on the frozen embedding), as you inferred.

The form that holds, from autograd's own products (INFERRED by me from your table; check it on the counted record):
- A normalisation `y = x * s` whose input needs a gradient has, in the backward, one each of `div [x]`, `mul [s, s]`, `mul [s]`, `mul [x, s]`, `mul [x]`, and two `mul [x, x]`.
- A norm's gain multiply `z = y * g` adds one `bwd/grad mul [x, g]` exactly when `y` needs a gradient, and one `mul [x, x]` (the gain's own gradient, `sum(gz * y)`) exactly when the gain is updated.
- So let `G` = the config's excess of `("bwd", "grad", "mul", gain)`: the normalisations of the norm that pass a gradient to their input. For each backward key `k` the record holds, `rest = excess[k][0] - (updated[0] if k is mul [x, x] else 0)` must be a whole multiple of `G`; where `G` is 0, `rest` must be 0.
- Your table under it: `G` = 108. The five single products: 108 % 108 = 0. `mul [x, x]`: 327 - 111 = 216 = 2 x 108. The rule fires.
- The detached case under it: with the record's P plain backwards missing on the config, each single product's excess is `G - P`, no multiple of `G`. Refused.

What I want:
1. Replace 45c881f613's line with this guard, in its own commit. The docstring states it in one sentence: each backward product the record holds, in whole numbers for each normalisation that passes a gradient to its input, after one product of the stream by itself for each updated gain.
2. `norms()` in the tests gets the real backward: the `[x, g]` multiply, the gain's `[x, x]`, and a count of normalisations whose input needs no gradient. Tests: your detached case still refused; a norm on an input with no gradient fires; `G` = 0 with a nonzero rest refused.
3. Run your entry again under it. The every-entry runs you have in flight (f3a4, 3b8e) stand for every other entry if the new commit moves none of their verdicts; say how you know (the rule's guard runs only where the record holds a backward product of the norm's shapes: name the entries that do).

If the counted record contradicts any line above (say the gain's gradient is not a `mul [x, x]`, or counted step 4 gives other numbers), send me the counts and leave the guard as it is.

Your threads reproduction (5:52 PM) is read: thank you, that is what I needed. I file it as its own issue; nothing more of it is yours. Keep your test's pin.

Tag RAN, READ, INFERRED.

