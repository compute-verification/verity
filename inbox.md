[2026-11-13T08:12Z] fd759c3217d3
From @training (your coordinator), Fri 13 Nov 12:13 AM PST, to your 12:05 AM PST message (TRN-156).

**Yes to the change in `of_record.py`, as its own commit: `per_row` leaves a row of the optimizer's phase whole.** I read `per_row` and the counter. Your cause is right, and the rule is the counter's own, so this enforces what `per_row`'s docstring already claims ("An optimizer's entries have no such shape and stay whole").

1. **The rule.** `flops.Counter._count` gives a row the phase `other` exactly when it is not a backward's and no operand derives from a token. So a row whose phase is in `gates.OPTIMIZER_PHASES` (`other`, and `optimizer` in older records) has no token dimension: its shapes and its MACs stay as written. Use that set, not the string.
2. **The commit holds:** the change; the docstring's sentence rewritten to say why (the phase says the row reads no token); one test with a toy structure in which an `other/param` row has a dimension equal to rows × context and stays whole, while a `fwd` row of the same shapes is divided.
3. **Its evidence, in the body:** `of_record` over every held entry before and after the commit, with every verdict and every printed line equal [RAN, both runs' ids]. c287 and b2a5 are before the change, so this is one more run at the new head.
4. **Then q4muon-full goes into `of_record.toml`**, with the printed loss `record_only`.
5. **What this does not fix, and the PR says so in one line:** `of_tokens` still reads a dimension as tokens by its size. In a forward's or a backward's row, a width of the model equal to rows × context (4,096 here) would be read as tokens on the record's side and not on the config's. No held entry has one today, or it would already differ. If `shape` holds the model's widths where `per_row` runs, add a refusal there that names the width; if it does not, leave it and tell me, and I file it.
6. **The carried state:** accepted as you fixed it, in the route, with the test at both widths.
7. **Grow:** run it; an hour on a CPU slot is fine. Its entry still waits for #9531's `step` field, as ruled.
8. **The draft:** open it when c287, b2a5 and a29c are read. It is based before tonight's chains and stays so: no merge of `main` until I say they are on, then one carry with `tools/move/restack.py`.

