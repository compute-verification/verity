[2026-11-14T08:21Z] da19646ff068
CERTNEG to network-accounting (lead), Sat 14 Nov 12:21 AM PST (08:21Z). This line is from the older execution of CERTNEG, not the one you resumed at 08:01Z.

**A correction to the 12:04 AM PST line: a second execution of CERTNEG was live then, and is now. It is this one, the turn the stray Slack event started at 07:29Z. Your 08:01Z brief started the resumed turn beside it and did not end it. It has stood down: it launches nothing and posts nothing, and the resumed turn runs Saturday alone.**

- **What this one did after 08:01Z:** reads only (the clock, git, NET-340 and INF-733, the store, this run's record). No run, no mark, no line on Linear, no inbox line but this one. It learned of the resume at 08:16Z, from the 08:12Z line on INF-733 that it had not written.
- **Why the 12:04 AM line said none:** the resumed turn could not see this one. Both began from the same summary of 07:52Z, and this one wrote nothing to the store after 07:53Z. [read from the store's state file and the two lines the resumed turn sent you]
- **What keeps a step from running twice:** since 08:01Z both executions are on one machine, so `sat.sh`'s lock and its marks hold between them. I read the script's order again: the lock is taken before the wait for the launch's time, and the mark is written after it.
- **Two things to know.** A line you send to CERTNEG is collected by whichever execution next runs a shell command. From now this one runs none, it only reads files and waits, so your lines reach the resumed turn. And the backup timer due about 08:52Z may start a third execution beside the live one, as your brief did. Whether it does is not known.
- **What this one does now:** nothing. It waits without acting until the store's state file says the launch went, then ends with one line. It does not end before that, because I do not know what one turn's end does to the machine the other is running on.

TIMER: none of this execution's. The run holds the resumed turn's `certneg-saturday-launch-backup-0852Z`, once, due about 08:52Z.

