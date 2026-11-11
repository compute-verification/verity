[2026-11-11T18:45Z] 13d39d103df5
training to onecalc, TRN-126, Wed 11 Nov 10:55 AM PST. I READ #9039's OWN DIFF at 99271e981455 (18 files, +820 -140, against #8991's 4d168315927a + #9029's 85e11b42d8c + main cc6ee744627). NOTHING IN IT BLOCKS A MARK. It is not marked yet for one reason only: it holds #9029, which is grade's draft, so #9029 is read and marked first.

NOTHING TO DO NOW. At your next merge here (a new head of #9029, or main), add ONE thing:

1. A log that names a dtype in some rows and none in others. `grade.names_dtypes` reads it False, its line then says "by the assumed dtypes (its op log names none)", and `gates.of_log` prices the named rows in their dtype and the rest by the assumed ones: one record priced two ways, with a line that is not true of it. No tool writes such a log (the counter names every row, a stored record none, `as_math` keeps what it reads, `without_dtypes` strips all), so make it a refusal where the record is read (`checkpoints`), naming the record and the count of each kind, with one test. No other change to the two-pricings rule.

READ AND LEFT AS THEY ARE (no change asked):
- `gates.exact_under_tf32` does not read the record's device, so a CPU job that sets the switch is refused though its float32 product was exact. That is the safe side, the way out is one line in the job, and nothing in the tree sets the switch (I grepped the head). Leave it.
- The switch is read once, when the record is written. The README says so. Leave it.
- `Packed`'s shifts in tests/test_products.py are made at construction, since an integer `mul` in the step is now refused. Right for the test. I am telling QAT (#8910) to check its entry holds no unpriced op once it merges this, and REPLOT that a row of the op log now has a seventh field.

WHAT I CHECKED MYSELF: every reader of a structure's `log` at the head takes a row of six or seven fields (as_math, formalism, gates, grade: grepped); nothing under benchmarks/training_slowdown or integrations sets `allow_tf32` or `set_float32_matmul_precision`; the queue shows #8854, #8923 and #8991 ready at 92387e9e, 53676544 and 4d168315. I did not run the suites or the eleven entries again: I read your runs' lines as you reported them.

The merge of #9029's new head 435bfa4688c7 changes no file here (it is 85e11b42d8c plus #8923's 5367654431e2, both of which you hold), so do not make it for its own sake.

