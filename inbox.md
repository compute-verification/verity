[2026-11-12T14:17Z] 7af9ed3dbea1
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 6:17 AM PST, to your 6:13 AM PST. **Ruled: as you built it. No flag and no exception. A kept run made before the start was certified is checked at its own commit.**

- A check that passes a run with no start lets a device skip the start, so the refusal stays.
- **Tonight's tries are not touched by it.** This is not marked before lean's merge is on `main` and vllm has answered, so tonight's job runs a tree without it, and the job checks its runs in the same tree it made them in.
- **In the body, under its own heading, say what the landing refuses:** every kept run of the relay made before it, by the chain's certificates step, with the words of the refusal. Then name each place in the tree that re-checks a kept run it did not make: a fixture, a table's run of record, an `art:` entry. For each say whether it is remade in this pull request, or is checked at its own commit and by what.
- If the chain's run of record is one of them, do not remake it here. Tell me, and I file its remaking as its own issue.
- The two failures and their mends, the short try and the kept rehearsal in line: read. Send the rehearsal's numbers when it ends.

