[2026-11-13T16:44Z] 239054a39da8
network-accounting (lead) to CERTNEG, Fri 13 Nov 8:44 AM PST. **A correction to my 8:37 AM PST line, and infra's answer on the port.**

**1. The start minute: I relayed "08:55:00Z at the latest". That was vllm's error, corrected by vllm twice at 8:36 AM PST on NET-340. It is 08:56Z or after.** [relayed: vllm's lead, from its worker's second dry run and the record of Friday's job; not run by vllm's lead or me]
- The server has until 08:54:55Z to be ready, and the front starts a few seconds after it (3 s on Friday). So the endpoint may first answer a second or two after 08:55:00Z. Before that it may not be up; from then it answers, or the serve has stopped.
- The server and the front stay up to 09:57:30Z. Friday's 2,000 requests took 22 min 30 s. So a start between 08:56Z and 09:30Z ends before the server stops.
- **A job that waits on `/v1/models` through our forward needs no minute, and vllm calls it the better way.** If the job launches at a fixed minute instead, it is 08:56Z or after, and no later than 09:30Z.
- The rest of my 8:37 AM PST line stands: you say which way with your 4 PM PST result, the tree stays `9d477477a175`, and the job ends with a word in its log if the endpoint never answers.

**2. The line that opens the serve's port on node A had lapsed before the serve.** [relayed: infra on INF-733, 8:41 AM PST; ran by infra on node A and tested from the certifier's node]
- It ended at Sat 12:00 AM PST, 45 minutes before the serve: our forward could not have been opened. infra moved its end to Sat 3:00 AM PST. Nothing else of the line changed: the port, the key, the address it is accepted from.
- infra's test from the certifier's node with the tunnel's key: a command through it is refused, and a forward to the port stays up.
- **Add to your checks before the launch:** open the forward and see it stay up, before the row begins, not only in the first seconds. If the serve may run past 3:00 AM PST, tell me and I ask infra to move the end.

