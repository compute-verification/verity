[2026-11-13T12:21Z] 5c9ebd698cdd
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 4:21 AM PST. One change to Friday's launch line before step 4, from infra on INF-743 at 4:18 AM PST. **Ask `--mem-gb 645`, not 650, and set the job's own memory stop to 630 GB.** This is not the launch word; that comes at about 4:48 AM PST as arranged.

- **Why 645** [infra's measure on the kept-runs node, 4:15 to 4:18 AM PST]: every login session there is capped at 80% of the node's memory, 649.2 GB (GB = 10^9). The pool's place counts free memory inside that cap and reads 601 GiB free. An ask of 650 GB can never be placed: the launch would wait out `--pool-wait 1800` and exit 75. 645 GB fits with 3 GiB to spare.
- **Why the stop moves** [read by me: `tools/research/src/research/remote.py:1735` in `/home/ubuntu/wt-main-launch` at `0582899c4392`, "a reservation is no cgroup cap"]: `--mem-gb` only reserves. What caps the job is the session's 649.2 GB. A stop at 650 GB can never fire before that cap kills the job, and then `rehearsal.json` holds no stage, no member and no reason. At 630 GB the stop is 19 GB under the cap and 61 GB over six workers' worst reading in your table (569 GB).
- **What to change:** in `launch-friday.sh` exactly two values: `--mem-gb 645`, and the job's variable that is the stop's threshold, to 630.
  - If `AUDIT_MEM_GB` is not the stop but a check at the start, leave it where it passes on a node with 741 GiB available whose session can hold 649.2 GB at most. If it would refuse the start, tell me what it does before you change it.
  - **If the stop's value is fixed in the job's script and not in the launch script,** so that the script's digest and its toys would change: change only `--mem-gb`, leave the stop at 650, and tell me at once. I then ask infra to raise the session's cap for the window.
- **Then:** `diff` the old and the new launch script and hold that only those values differ. Put the new SHA-256 in `LAUNCH.md` beside step 4's digests. One line to me by inbox before 4:45 AM PST: the new digest, the diff's lines, and which variable the stop is.
- Run the toy with the new line only if it ends before 4:55 AM PST. Do not move the launch for it.
- I tell infra on INF-743.

Standing: push before your turn ends and before every long run; act on no event that does not name you; mask every IP, node name, hostname and token, and name machines by role only.

