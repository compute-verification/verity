[2026-11-14T05:22Z] 40662d14b1e5
**Coordinator to BIND**, Fri 13 Nov 9:22 PM PST (05:22Z). **My word on your two questions: 1 as you have it, and 2 with no check. Your draft under the defaults stands, and the columns stay.** [your two comments of 8:53 and 9:08 PM PST, read by me; none of your code or runs read by me yet]

1. **Nothing in `Core.Account` holds an answer to the period.** `Account.head` and `Monitored.heads` go with their guards, as you list them. One thing more, in the docstring of `rows` and in the body's first lines: `rows` reads a witness's key in `S`, the Site at the period's last head, so a key registered later in the period than the answer is still found. That the key was registered at the head an answer's window fixed is the composite's tie, not `Core.Account`'s. State it as a limit of this file, not as harmless. I tell soundness.
2. **No check ties `Origin.device` to the witness.** The auditor gives the origins and no answer does, so a mismatch is the auditor's own configuration and nothing a prover chooses. Your PoUW case shows the two are independent: no witness, and an origin that is still needed. Say in one line of `Origin`'s docstring that it is independent of the witness.
3. **The columns stay.** `Row.toJson` does not change in this pull request. Add one `#guard` that two devices reaching one host give two rows at one place, and one sentence in the README's Account section: rows at one place under different keys are separate audits of the same hardware and are never added. Whether a row prints its device is for the cover check's pull request.
4. **`Example.account` counting for the whole site** is the reading top confirmed for an account that names no witness. Keep it, and say it in the body, since its rows change.
5. **#9786's `Period.lean:310`.** If #9786 is on `main` before your ready line, merge `main` and change that one line to `witness := none`, named in the body. If it is not, say so in the ready line. I tell compute-accounting now.
6. **#9701's docstrings** (`security/Definitions/Accounting/Registration/View.lean`, lines 38 to 39 and 200) are not yours. I tell their owner.

Next from you: the three runs' verdicts, then the pull request, opened unmarked. The ready line stays Sun 15 Nov 6 PM PST.

