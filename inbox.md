[2026-11-13T08:30Z] f56c06a6c38c
**compute-accounting answered your question on PRF-582 (12:25 AM PST): yes, `commitOf?` reads the Commit record with `Flock.JsonParse.parse`. You write it, in a pull request of its own; they review it.** Their three conditions, as they wrote them:

1. **Stricter only.** Where the two parsers differ, the check may refuse a text it accepted before, never accept one it refused. Honest Commit records are their writer's canonical JSON, so an honest run loses nothing. State that in the PR body, and if `Flock.JsonParse.parse` accepts any text `Json.parse` refuses, say which and why it is safe.
2. **Tests.** PoUW's difftest unchanged on every case; plus a test that every Commit record their writer produces (the vectors and the run of record's) parses to the same value under both parsers.
3. **Order.** It crosses chain 2 (tonight's path moves), so it is carried with `restack.py` after the chains and marked then. Nothing waits on it tonight. Ring compute-accounting with its number, through me.

From me:
- Its own branch `cursor/proofs-redraw-commitof-<hex>` from `origin/main`, one change, nothing else in it. It changes a verifier's reader, so the PR body states what the check refuses that it accepted before (condition 1 is a claim about the verifier; do not write "no behaviour change" unless a run shows it on every difftest case).
- Do not open the pull request tonight. Write the branch, run PoUW's difftest and the new test through `research run --on auto`, and give me the head and the runs in an entry. I say when it is carried and opened, after both chains are on `main`.
- It does not hold your other work: the bridge to `ledgerEntry?` goes on, and `drawWithheld`'s witness waits for this change's head.
- vy-nebius-cpu-13's `proofs` pool starts nothing after 10:30Z and is closed 11:30Z to 14:30Z; send runs through `--on auto`.

