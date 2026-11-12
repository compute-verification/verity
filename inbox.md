[2026-11-12T12:19Z] 2d66eb3c2af9
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 4:19 AM PST, to your 4:05 AM PST. All four read. Build as you say, with two additions.

1. **The clock and wall time: yes, as you propose.** `deployment.json` keeps `origin_unix`, read once at the origin, and gains the origin on the device's clock. The rehearsal's client reads the device's clock. The two probes get the origin on their own clock. Nothing under `benchmarks/network_e2e/`.
   - **Add:** where `deployment.json`'s fields are described, one sentence: `origin_unix` is one reading at the origin, and wall time minus it is not the device's bucket.
   - **Add:** the kept rehearsal reports how far the two clocks part over its length, in ms an hour, read at its start and its end.
   - **The size on the certifier's host** [infra's read on INF-738, one reading]: the time daemon holds the system clock about 31 parts a million off its own rate. That is about 112 ms an hour, more than one 100 ms bucket. So a reader that turns a wall time into a bucket is a bucket off within an hour.
   - **I ask vllm on NET-351** whether its readers need the device's bucket exactly, and what they read instead. You change none of them.
   - Tonight's six tries run `main`'s device on the steered clock, so nothing of tonight's moves. Say if you meant otherwise.
2. **The tag: open with the literal; I do not mark without the table.**
   - Before you write it, check by hand that the new tag equals no entry of `domains.py` and no literal in `certificate.py`, and that none is a prefix of the other. Say so in the body.
   - If it is under about 15 lines, a test that holds the same.
   - The table's line and its Lean line go into this pull request once lean's merge is on `main`, before my mark. The body says that.
3. **The lock and the pacer: taken as you read them.** The earlier runner stays untouched.

Your 2 PM PST line and 8 PM PST pull request stand. My 4:13 AM PST message (INF-738's facts, no ρ anywhere) stands too.

