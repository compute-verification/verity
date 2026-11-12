[2026-11-12T14:22Z] 0d7607b29053
NETSEATS to the network-accounting coordinator, Thu 12 Nov 6:22 AM PST. Job D (NET-368). **The kept rehearsal ran clean (`r20261112-140417-d922`, exit 0, at `950e6791e7cf`). One thing in it is yours to judge: on the raw clock the pacer starts some releases later than on the steered clock, by at most 0.077 ms. No release left its bucket or passed the slack.**

The rehearsal: 240 requests at 1 a second, 5 link-windows of 60 s, then the device at `3498d4ad65` as the control, on the same node right after. [ran; read from its log and its `warden.json`]

**1. The pacer's start (the thing you asked to hear at once).**
- Raw clock: 152 of 2385 releases started more than 0.005 ms after their time; 45 more than 0.05 ms; the latest 0.077 ms. Median 0.001 ms.
- Control: none over 0.004 ms. Median 0.001 ms.
- Both released the same 2385 buckets. On both, none reached the pacer after its time, none started past the 25 ms slack, all were written whole. The short try showed the same direction (0.015 ms against 0.002 ms).
- **The cause is not established.** A read of the raw clock from Python cost 641 ns on the node (506 ns in the short try). On my VM it costs 90 ns, the same as the steered clock by id, so there it is a vDSO read [ran]. The pacer spins on the clock up to its release time. If a raw read is a system call on the node, each spin step enters the kernel [argued]. I did not time the steered clock's read on the node.
- I change nothing for it. If you want it established: one `research run` that times both clocks' reads and a few thousand spin-waits on each, no rehearsal. Say so and I launch it.

**2. The ticks.** 3000 on each side. Latest 4.26 ms (control 4.17 ms) against the 25 ms slack; median 0.79 ms (control 0.80 ms). None before its deadline. No link-window withheld; 15 of 15 certified; the check accepted; 240 of 240 requests ended.

**3. The start and the clocks.** The start certified at base 0. Wall time parted from the raw clock by 7.07 ms in 304.9 s: **83.5 ms an hour** on the pool's node, not the certifier's host. The short try read 103.8 ms an hour over 65 s.
- Scaled, if the rate held [argued; it is the node's time service's rate and is not constant]: 14 to 17 s a week, 60 to 75 s a month, 12 to 15 minutes a year.

**4. What the landing refuses: taken, it goes in the body under its own heading.** The words [ran]: `the warden's own check refuses the run: the session's start isn't certified (no start's certificate)`.
- **In the tree, nothing re-checks a kept relay run it did not make.** No fixture, no `art:` entry of `fixtures/artifacts.json` (147 entries) and no table holds a relay's `deployment.json` [ran: the registry's paths and a search of the tree]. The demos' tables name no run of record of it.
- **Tools that take a kept run by path:** only `chain_record.py certs --warden-run` reruns the check (`chain_record.sh` with `CR_PATH=warden`, and `served_session.py chain` through it). `scan_zk.py`, `n2_zk.py`, `n2_tie_zk.py`, `a5_warden.py`, `prove_path.py`'s kept scan and `served_session.py io` read the key, session, schedule and records and never the start: they take an old run as before [read].
- **So the chain's run of record is one of them**, if it is re-run on its kept gate run after this lands: its certificates step refuses. It is not in the tree, and I remake nothing. It is checked at its own commit by `chain_record.sh` from that commit's tree.

**5. Runs.** The changed suites at `ad3dc31cddbb`: `r20261112-141100-3629`, two checks ahead of it at 6:17 AM PST. The run at `be9188e8de32` is still going.

