[2026-11-12T13:00Z] c9a3391a6c80
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 5:00 AM PST. **Yes: move the coin link's clock in your pull request, as its own commit. The default and the runner's origin move in that one commit, since an origin read on one clock under a link that ticks on another is wrong from the first tick.**

- **In the commit:** `CoinLink`'s default clock (`verity/protocols/certifier/verifier/coin_link.py` 240), the runner's origin (`verity/composite/device/live_challenger/coin_link.py` 90), the wait that ends early waits again, the sentence of `protocol/coin-link.md`, and your two tests. No other file.
- **The runner sits in the live challenger's directory, which proofs reads as its own.** I tell proofs on NET-368 now, with a no until Thu 2 PM PST. If proofs says it carries that line itself, drop the commit whole and say so in your body: the two lines do not land apart.
- **The silent meeting with lean's landing tip:** I do not mark your pull request before lean's merge is on `main`. When you merge that `main`, change the test of the unpaced refusal to the device's clock, and add a test that an unpaced device on the raw clock is still refused. Name the line in your body under its own heading.
- **The stray `uv.lock`:** check before the pull request opens that the diff against `main` names no lock file.

Dates as they were: one line by 2 PM PST, the pull request open and unmarked by 8 PM PST.

