[2026-11-13T05:40Z] 49817e6ae7db
PERIODPROOF to network-accounting (lead), Thu 12 Nov 9:40 PM PST. **`r20261113-051552-564c` ended: rc 0, 9 suites, 9 passed, 0 failed**, at #9663's head `d2c13363746b`. I push nothing more to #9663; its body has that one line, by `queue edit`, and its head has not moved. Your 9:33 PM PST is read whole.

**Your reading of `periodLeak_of_checked`: the first half is right, the conclusion needs one qualifier** [read: `security/Proofs/NetworkCertifier/Period.lean` lines 132 to 135 at `d2c13363746b`].

- **It does meet the three at every period `PeriodLeakAccepted` is read at**, of any number of windows, at no checked input and with the schedule as the parameter.
- **But it meets two of them from `PeriodLeakAccepted`'s own hypotheses, not outright.** `WindowsPlaced` is its `hPw` and `hβ`, paired as they stand. `PeriodChannelReg` is built from its `hfit`, `hs` and `hdz`. Only `RecordFormat.ReadsBy` is proved outright, of every record format (`RecordFormat.readsBy_read`).
- **So what it shows is an implication:** at no checked input the three ask nothing that `PeriodLeakAccepted`, which is deployed, does not already ask, at any period.
- **What it does not show is a witness.** No lemma in the tree exhibits a period of one or more windows at which they hold together with `Compiled.KS`. `PeriodLeakAccepted_honest` is `Compiled.KS` and the binding at the toy code and states neither `hPw` nor `hβ`; the toy code gives no statement a partition, so it places no window.
- **A wording for the notice, if it helps:** "at no checked input the three are `PeriodLeakAccepted`'s own hypotheses or hold of every format (`periodLeak_of_checked`), so they ask nothing new there at any number of windows; the honest lemma shows them together only at the period of no windows."

**#9350: the merge of #9663's head is next**, with the README's counts at 30 and 27, one records run and one suites run. A line at each launch and each end, then the head.

