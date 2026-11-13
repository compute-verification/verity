[2026-11-13T22:04Z] 9e21face858e
From training's coordinator, Fri 13 Nov 2:04 PM PST. **#9531 at `4e16b1bc223e`: I have read the fix and every run. The body is the one thing between it and the mark; send it as soon as it is true, without waiting for 2:30 PM or for the single every-entry run.**

What I read myself, so your report need not repeat it:

- The commit, and its test on my VM: it passes at the head, and with the parent's `of_record.py` the command exits 1 where the test asks 0.
- The three parts, each at a tree equal to the head, from their `stdout.log`: 16, 16 and 15 entries, every one `equal`, none refused. Together they name each of the 47 held entries of `of_record.toml` once.
- The suites run: 4 suites, 4 passed, 0 failed.

What the body still says, and must not: its head is `dcf3224400ba`, and "at this head the command refuses" the grown entry. Rewrite those parts for `4e16b1bc223e`:

1. The head, and the one commit added since: what was wrong (the count ran without the kernel a job holds, so its own check refused every entry counted at more than one thread), and the fix.
2. The three parts with their last lines, and that they cover all 47 held entries. The single run's state as it is when you write; the mark does not wait for it.
3. The grown entry's first lines as the command prints them now (equal), in place of the refusal.
4. The 81M entry: one line on what it is held to, or that nothing holds it and why.

Then one message with the body's edit time. I mark on it; the goal's hour is 4:00 PM PST.

Also: #9764 (the count of repeats) set the same version, 4.27.0. Yours keeps it and goes first; #9764 takes the next number after yours is on main. Nothing for you to do.

