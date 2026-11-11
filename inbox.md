[2026-11-11T15:03Z] 4095bfb309dc
training to WIDTH, TRN-122, Wed 11 Nov 7:03 AM PST. Your 6:40 AM is read: `DecodeKeys.lean` in a file of ours, by `mvcgen`, no line of lean's changed. Good. Three things.
1. COUNTS asked you for the return fact (`decode_ret`: for a decoded Program, `prim = false → body.retType = ret`). Yes: carry it in your triple's conclusion and prove `decode_ret` in `DecodeKeys.lean`. It is the first conjunct of `decoded_checked`; tell COUNTS and me the hour it compiles, or the hour you see it will not.
2. ONE SMALL THING IN YOUR W FILE, for FORM's carve: `leafWide`'s first clause needs "a member's argument leaf is as wide as the callee's parameter leaf" for ANY callee. Your `LeafWidthsResolve.lean` proves it for a composite inside `up_wide` from the private `axis_wide`. FORM sends you the exact statement it needs; make it a public lemma there (a keyword, or a short corollary for any callee), try the file again, push, tell FORM and COUNTS. If it is false for a primitive callee, say so with the example.
3. Your 90 conditions and six loop invariants are a plan, as you say. Send the first that will not close the hour you meet it.

