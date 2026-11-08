[2026-11-08T18:09Z] 43bd1c2af73b
Coordinator 18:10Z: network-accounting runs the certifier side themselves (reach at 18:15Z, tick-sunday/launch.sh at 19:00Z from cpu-11, tree 6e165a8e54, TAIL=300, PORT 18745). Do NOT launch the certifier job, even if nobody confirms by 18:50Z. Serve only, on 18745, and post your server run id on thread 1794083350.111549 once READY. After the run, fetch their outs for the re-encode as planned.

[2026-11-08T19:45Z] b43dd7010619
Coordinator 19:47Z, URGENT: network-accounting REFUSED b108: 1,332/3,000 streams fell behind egress (100 ms buckets = 10 tok/s, GLM served about 4.5). RERUN NOW on node A with 250 ms buckets: relaunch the server (ticksun) with the tick bucket 250 ms (VERITY_TICK bucket; tau_sat = tau_free stay 220), START now, READY by about 20:15Z on 127.0.0.1:18745, END 21:05Z (the stop must be by 21:05Z: the 131k serve follows), TAKEOVER=1. Post READY and the run id on thread 1794083350.111549; network-accounting relaunches the certifier with --bucket-ms 250 on READY. Keep b108 results for the report (it is the refused point).

