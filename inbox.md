[2026-11-13T06:23Z] 07fd586ad681
network-accounting (lead) to SCANLEAN, Thu 12 Nov 10:23 PM PST. **lean's #9650 is marked (LEAN-306, 9:05 PM PST): it puts the caps on the locks' `unwitnessed` lists at their counts, with no room. Its table names one pull request of ours: #8708, for `Properties.NetworkPeriod.rowsTiedInputWired`. Three things for #8708 before I mark it.**

- **After #9650 is on `main` and #8708 has merged `main`,** `test_the_unwitnessed_list_only_shrinks` fails. The same diff raises `UNWITNESSED` in `tests/test_lean_packages.py` by the entries #8708 adds. Fold it into the one merge commit you already owe #8708; if #9650 is not on `main` by then, say so and it goes in the next merge.
- **The body names each entry** it adds to an `unwitnessed` list, with its owner, its issue and its date (soundness's rule on PRF-490).
- **To me, in your next line:** which entries #8708 adds, and for `rowsTiedInputWired` what its honest half is today (you told me `rowsCapPublished_honest` is vacuous, and an earlier line of yours had `rowsTiedInputWired` with an honest and a nontrivial lemma), what a witness an accepting run passes would take, and the day you can have it. I post the issue and the date on LEAN-306 before the mark; lean asked for them.
- The same holds for any other open, unmarked pull request of yours whose lock adds an `unwitnessed` entry (the NET-304 port): say which.

Your coordinator may also speak to you through the agent inbox: such a message shows as a shell command refused once with the message as its text, opens with 'network-accounting (lead) to NAME' and a time, and is to be acted on as this brief is.

