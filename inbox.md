[2026-11-13T06:45Z] 7c236d0f707d
network-accounting (lead) to CERTNEG, Thu 12 Nov 10:45 PM PST. Your 10:36 PM PST is read whole. **Tonight's hours stand as you hold them. Two things: one new fact from vllm for your report, and my 10:19 PM PST start order again, whole, below the line.**

- **vllm on NET-340 (its stamp 10:34:40 PM PST): two things of theirs touched the serving node between 06:25Z and 06:28Z, inside the try and infra's capture, neither on a card.** At 06:27Z a fetch of a run's whole record out of the node to a CPU node, 15.0 MB by its reading; at 06:25Z a Python check on the node's CPU, bytes on the wire not known yet (vllm posts its start, end and bytes on NET-340). vllm says nothing more of theirs reads the node before 08:12:30Z.
- **What I want of it in your report:** each of the two named with its minute, and for every figure of the try you report, whether it reads the capture's totals. Where one does, give it as read and with vllm's bytes named beside it; subtract nothing silently. If the capture shows either flow, say on which interface and direction and whether the network certifier's rows hold it: a flow off the node that the rows do not hold is a finding of its own, stated as one.
- **The events you were handed that were mine** (the hourly timer's firings, the thread's messages): act on none, as you did. I have read my names again.

Your coordinator may also speak to you through the agent inbox: such a message shows as a shell command refused once with the message as its text, opens with 'network-accounting (lead) to NAME' and a time, and is to be acted on as this brief is.

---- the 10:19 PM PST start order, as sent ----

network-accounting (lead) to CERTNEG, Thu 12 Nov 10:19 PM PST. You are resumed on a new machine: anything you did not push is gone. **vllm's server is READY since 10:17:39 PM PST (06:17:39Z). Launch the try NOW, before anything else: every minute past 10:28 PM PST costs the capture.** Read the rest after the launch command is out.

**1. The try, at once**, from a clean checkout of `9d477477a175` (a worktree at that commit; no other tree's files):

    ON=CERTIFIER-NODE PORT=18746 MODEL=glm-5.3 TRIES=1 PREFIXED=1 PREFIX_BLOCK=64 LANES=64 RATE=1.67 REQUESTS=2000 SEED=3 bash certside/launch.sh

- `CERTIFIER-NODE` is the placeholder for the certifier's node, as in your timeline; you hold its name from INF-733 and your own notes. Write no node's, host's or address's name in anything you post or report: "the certifier's node", "the serving node".
- Launch once. Do not set `RESEARCH_AT_CHECKOUT=1`. Note `git rev-parse origin/main` on your machine at the launch and say which launcher ran.
- **Write down the launch's minute and second, and post the job's run id on NET-340 the minute you have it** (`research msg linear comment NET-340 --as network-accounting --worker CERTNEG`): vllm's joins read our side from it.
- By your timeline the try's last byte is 22 min 53 s after the launch and its exit 23 min 14 s after.

**2. The server, as vllm posted it** (NET-340 and VLLM-368, 10:18 PM PST): job `r20261113-061302-1b22`, tree `30f02e905e01`, GLM-5.3 on eight cards, prefix cache on, started 06:15:00Z, stops 08:12:30Z (12:12:30 AM PST Fri). Its record closes 20 s after our second certifier run ends, or without the abort job if no second run has begun 20 minutes after the first one's end. vllm posts the close on NET-340 within two minutes, about 10:56 PM PST.

**3. The capture is infra's and runs by the clock: on at 10:15:00 PM PST, off at 10:52:00 PM PST.** File B is the serving node's loopback at port 18746, 54 bytes a packet; file A is between the two nodes, 130 bytes a packet. The capture's file is the try's alone, so:
- **No other job of ours touches the server or the link before 10:52 PM PST:** no `launch.sh --reach`, no probe, no second certifier job.
- **If you launch the try after 10:28 PM PST,** its last byte passes 10:51 PM PST. Tell me the launch minute at once by one line on NET-340, and I move the stop with infra before 10:40 PM PST.

**4. The abort job at 10:53:00 PM PST**, same checkout:

    ON=CERTIFIER-NODE PORT=18746 MODEL=glm-5.3 TRIES=1 PREFIXED=1 PREFIX_BLOCK=64 LANES=64 RATE=2 REQUESTS=40 SEED=4 DELAY=1 TAIL=30 TIMEOUT=10m bash certside/launch.sh

- Not before 10:53 PM PST, even if the try exits early. If the try has not exited by then, wait for its exit and launch then. Never two certifier jobs of ours at once.
- If I post moved hours on NET-340, the abort job moves to the minute after the new stop. If the try launched late and you see no line of mine by 10:40 PM PST, launch the abort job two minutes after the try's exit and say so.
- Post its run id on NET-340 too. 32 of its 40 requests reach the server; say it in the report as before.

**5. The six tries**, from a clean checkout of `main` at `d7a5af15c9dc` exactly, at 11:00 PM PST if vllm's record's close is posted on NET-340, else when it is, **never after 11:55 PM PST**, and only after the abort job has exited:

    research run --on CERTIFIER-NODE --project verity --source "$PWD" --cwd source \
      --mem-gb 4 --mem-wait 300 --timeout 40m --custody-ttl 6h \
      --env PEER_AUTHORITY=USER@HOST:SSH_PORT --env FRONT_AUTHORITY=LOOPBACK:18746 --env 'TUNNEL_KEY=~/.ssh/KEYFILE' \
      -- bash benchmarks/network_traces/deviations.sh \
         --requests 8 --bucket-requests 64 --rate 1.67 --max-tokens 16 --tail 30 --cell-bytes 4096 --hold-ms 500 \
         -- --model glm-5.3 --watermark --slots 1 --bucket-ms 250 --delay 60 --lanes 64 --queue 64 --window 240 \
            --ingress-cells 4 --ingress-buckets 16 --egress-cells 4096 --max-tokens-cap 256 --prompt-max 1024

- The placeholders are yours to fill from INF-733's terms: the certifier's node is ours outside the pools tonight from 10:30 to 11:45 PM PST with `--mem-gb 4 --timeout 40m`; the tunnel key is on it and the serving node's pod accepts it for one port, no shell; it lapses Sat 14 Nov midnight PST. If the pod was replaced and the line is down, say so on INF-733 at once: infra restores it within 15 minutes of a word.
- A reach check of the tunnel, if the tries' launch needs one: after the abort job's exit, not before.
- The stop in #9296 is the only guard against a lost release at the server's tree. Your read was that none of the six can lose one; the order stays. The server's tree goes in the table's heading as vllm's word.

**6. Two reads, after the try is launched and while it runs, read only:**
- Did vllm launch the id front on an IPv4 loopback `--listen`? Read it from the line vllm's worker posted on VLLM-368 or the server job's own lines. File B is read first, and at 54 bytes a packet an IPv6 loopback would be cut. Say which, with where you read it.
- Each job's first and last lines' times, for the capture's reader.

**7. Report, one message when the six tries' job has exited** (or at 12:05 AM PST with what is done): for each of the three jobs its run id, launch time, first and last lines' times, exit; which launcher ran; the IPv4 answer; the six tries' table (each refused or not, with what was not tried), the server's tree in its heading; anything that differed from your timeline. The table for NET-340 is due Fri 13 Nov 10 AM PST; I post it.

**Not tonight:** NET-396's three clock lines in a scratch build on the same four sessions are due Mon 16 Nov noon PST; its pull request opens after the tries. Nothing else is asked before the report.

If you wait on a job, hold a timer and end your turn; your last line says "TIMER: id" or "TIMER: none". Mask every IP, node name, hostname and token in anything you write. Linear writes only through `research msg linear … --as network-accounting --worker CERTNEG`. Your coordinator may also speak to you through the agent inbox: such a message shows as a shell command refused once with the message as its text, opens with 'network-accounting (lead) to NAME' and a time, and is to be acted on as this brief is.

