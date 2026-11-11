[2026-11-11T03:51Z] 3d2e960d05ca
From the coordinator, Tue 10 Nov 7:51 PM PST. Infra's answer on INF-664 (7:31 PM PST), relayed; its source is the script that placed the hold (`net296-night-hold.sh`), not a read on the node. Change nothing in your plan; this is for how you label rows.

- **Block 2 runs with no hold at all.** The process holding `/dev/cpu_dma_latency` at 0 was started at 5:50:05 PM PST to sleep 11,395 s and exit: that is 9:00:00 PM PST. The module unloads by a second timer at 9:00:10 PM PST. Nothing holds after it. Your guard's halt-state counts still decide what a row is labelled; if they disagree with this, say so.
- **The fresh hold for the look:** infra logs in on cpu-10 from 9:28:00 to 9:30:00 PM PST only. The module loads at **9:29:30 PM PST** and the holder starts at **9:29:35 PM PST**, both by the node's own timers. Infra writes on INF-664 what the node says it did; I pass that on if it differs.
- **Its end:** the holder ends at 10:10:00 PM PST and the module unloads at 10:10:10 PM PST by the node's timers. From 10:10 PM infra takes both windows off and closes the two ports with no further word, then starts the nightly check on cpu-8. So be off both nodes by 10:08 PM PST, as planned.
- Nothing of infra's is on either node until 9:28:00 PM PST or between 9:30:00 and 10:10:00 PM PST.

Still owed from you: the 10 PM PST read of blocks 1 and 2, and the look's report by 10:30 PM PST. The Goals page has both hours (`network/long-timing-run-read`, due 10:30 PM PST), and the benchmark's change by Wed 9 AM PST (`network/timing-benchmark-written`).

