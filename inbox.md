[2026-11-12T20:13Z] a74148fb2de5
TERMGB, from the network-accounting coordinator, Thu 12 Nov 12:14 PM PST. **A new piece, ahead of everything but your 1:30 PM PST report: the benchmark's row reads the stop class's four verdicts from records, and a Python function says how the class is charged. Daniel asked through proofs (NET-390, 11:47 AM PST) for Friday's GLM-5.3 row with every verdict from a run and nothing assumed, by Fri 13 Nov 6 PM PST. In your 1:30 PM report give your hour for it; I put it on proofs' board at 3 PM PST.**

**Read first, on NET-390:** proofs (11:34 AM, 11:51 AM, 11:57 AM PST), soundness (12:00 PM PST), benchmarks (12:00 PM PST). NET-387's description, for what the Lean will do later.

**What to build, on `cursor/na-certifier-term-simulated-b4e6`** (proofs' worker commits there too, Way 1; one pull request, ours)
1. **One function in the certifier's verifier package**, not under `benchmarks/`: given the signed head and a class's verdict records, it says how the class is charged. Below its schedule only where the class has a count above 0 and all four verdicts are there (the stop record, the response tie, the certified read of the request's three words, the compute audit's second stage), each accept and each `complete`, each from a run on the one serve. Otherwise the class's whole schedule, naming each verdict that is missing, refused or not complete. A missing `complete` field reads as not complete (PRF-608). A class with a count of 0 owes no verdict.
2. **Tests for it:** a class with a count and no tie; a tie accepted and not complete; a verdict from another run; all four there.
3. **The row** reads the records under the label `run_of_record=exfiltration-glm53-stop-verdicts` (benchmarks' format: `run`, `verifier`, `verdict`, `proved_by`), calls the function, and prints each verdict under benchmarks' new mark `run, not proved`. With no records it prints the whole schedule's charge and says why. No flag switches the small figure on.
4. **The count's line and the head's time.** soundness ruled on NET-379 (11:06 AM PST) that a form 2 head leaves at a set tick, and that the count's line prints the late slot's charge beside the count's. The tick is not built (NET-399). Until it is, the line says the head's time is not held to a tick. It prints no late charge of zero.

**Who writes which record:** proofs the stop record's, the certified read's and the compute audit's; our worker OPTIN the response tie's (NET-386, on `cursor/na-scratch-net386-<4 hex>`). Agree the record's fields with OPTIN by inbox before either of you writes a second form.

**What waits:** the merges of #9305 and #9329 after lean's landing, by a day. Say in your report if anything else of yours moves.

**Also:** proofs' Way 1 is noted; you merge what its worker commits. Runs: one node of pool `network` takes no new run today (INF-756), name another. The repository is public: no node, host or pool machine's name, no IP address, of a key only its variable's name.

