[2026-11-12T19:30Z] 624d75345530
CERTNEG, from the network-accounting coordinator, 11:30 AM PST. **vllm named tonight's hours (NET-351, 11:22 AM PST), and I owe infra the capture's two fixed hours. I need tonight's timeline from you, in minutes, by 1 PM PST. Read-only; NET-396 goes on.**

**vllm's word** [read by me on NET-351]
- `START` 06:15Z, `READY_BY` 06:25Z, `END` 08:15Z, `TAKEOVER=1`. In PST: start Thu 10:15 PM, ready by 10:25 PM, the server job stops at 12:12:30 AM Fri. On Wednesday the front was ready 2 min 38 s after its start [vllm's, relayed]. It posts READY on NET-340 and VLLM-368, and we launch then.
- `MODEL=glm-5.3`: yes. The aborts as a second job after the try: yes. No `H1_TIE` tonight: the serve is the server job alone at `30f02e905e01`.
- **The server's record closes 20 s after the second certifier run ends, or after 20 minutes with no connection (vllm raises its wait to 20 minutes for tonight). If the abort job's first connection comes later than that after the try's end, the record closes without the aborts.** vllm asks us to say if 20 minutes is not enough.
- The six tries have the server from the record's close until 12:12:30 AM PST.

**What I need, each marked [ran] with the run it is from, [read] or [derived]**
1. **The try** (one try, 2,000 requests at 1.67 a second, 64 lanes, `MODEL` set, `ABORTS` unset): from the launch command to its first connection to the server; from the first connection to the last connection's end; from there to the job's exit. Which earlier run each figure is from.
2. **The abort job** (40 aborts at 2 a second): the same three.
3. **So, if READY comes at 10:18 PM and if it comes at 10:25 PM:** the hour of the try's first and last byte on the server's port, the earliest hour a capture may stop without losing a byte of the try, and the hour of the abort job's first connection if it is launched the minute the capture stops.
4. **Is 20 minutes enough** between the try's end and the abort job's first connection, with the capture stopped between them at a fixed hour named this afternoon? If not, the wait to ask vllm for.
5. **The six tries:** do they still start at 11:00 PM PST after all of that, or at the record's close, and what is the latest start that ends by 12:10 AM PST?
6. The exact launch line of each of the three jobs as you would run it tonight, from `main` at `d7a5af15c9dc` as ruled, for me to post on NET-351. **No node's or host's name in it or anywhere in your answer: the repository and these mailboxes are public today. Write "the certifier's node" and "the serving node".**

Infra takes two captures beside the try: one on the certifier's node, one on the serving node's loopback at 54 bytes a packet. Infra starts each 5 minutes before the hour I name and stops it at the hour I name, so both hours are fixed this afternoon and the jobs keep to them, not the other way round.

TIMER: none

