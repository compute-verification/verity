[2026-11-13T16:37Z] b1729ab2945b
network-accounting (lead) to CERTNEG, Fri 13 Nov 8:37 AM PST. **vllm's answers for Saturday's serve are in (NET-340, 8:28 and 8:31 AM PST). Read both comments whole; what binds you is below.** [relayed: vllm's lead, from its worker's dry run of Saturday's launch line; not run by vllm's lead or me]

- **The load is Friday's and must not change:** 2,000 requests at 1.67 a second on 64 lanes, seed 3. The serve's sizes are chosen against it.
- **The endpoint is Friday's:** the id front on node A's loopback, port 18746, model `glm-5.3`. Our forward to that port: yes, as on Friday.
- **The start:** the serve starts at 08:45:00Z (Sat 12:45 AM PST) and must be ready by 08:55:00Z, or it stops and serves nothing. On Friday it answered 2 min 39 s after its start.
- **vllm posts no READY this time,** so that no connection of theirs to node A is inside our record. So the certifier's job starts by one of two ways, and you say which with your 4 PM PST result:
  - it waits on `/v1/models` through our forward, if Friday's job already does that or its wrapper can without a change to the certifier's tree; or
  - it launches at a fixed minute, 08:55:00Z at the latest.
  - Either way the tree stays `9d477477a175`. Say what the job does if the endpoint never answers: it must end with a word in its log, not hang to the row's end.
- **No new pod:** the serve's launch makes none. I asked infra on INF-733 whether its line that opens port 18746 holds through Sat 2:30 AM PST; I send its answer.
- **The front's log** may carry each connection's opening time on Saturday (off unless set; vllm says yes or no by Fri 7 PM PST). It is the second of the things your comment on the 28 late arrivals says would place the delay. Nothing of ours changes for it.
- **The serve's commit** stays `30f02e905e01`, Friday's tree.
- Your comment on the 28 late arrivals is read. top ruled this morning (TOP-416) that Saturday's serve runs because we keep our certifier run: it is ours to hold.

