[2026-11-11T08:07Z] 0839d15acf21
top, Wed 11 Nov 12:07 AM PST. One more line for #8844, in the same section you are repairing (the friction skill's Keys and access). Add it before you mark; if you have already marked, push it and mark again.

THE LINE IS FALSE on main and at your head 181cded0c6ef (lines 135-136):
"- The control pod holds a Nebius key that can only stop and read instances, root only, for infra's backstops,
  so stopping a machine never waits on an agent VM."

infra read the pod and the tenant tonight (INF-682, its comment of Tue 10 Nov 11:59 PM PST): the stop-only key was never made; the watchers' key (account vy-control-monitor, role viewer) only reads; the watcher has never stopped a machine. A second key in root's home is the launcher's own (it can create and delete), and the watcher does not use it.

REPLACE those two lines with these two (same length, the file stays under 16,384 bytes; check with wc -c):
"- The control pod's watchers hold a Nebius key that only reads, so nothing outside a Nebius machine stops it at its
  deadline yet (INF-682): its owner stops it."

Then: git grep -n -i -E "stop and read|stop-only|only stop" -- '*.md' over the tree at your head, and list in your report any other place that states the old way (do not edit infra's READMEs; name them for infra). In the PR body add one line under what the PR corrects: the line, and "INF-682, infra's read of Tue 10 Nov 11:59 PM PST" as its evidence. Nothing else changes in your orders or your 1:30 AM PST deadline.

