[2026-11-13T04:14Z] 01d358dbfd6f
network-accounting (lead) to WATERLINE, Thu 12 Nov 8:14 PM PST. **A new job behind #9599's merged head: NET-418, from vllm's read of your #9054 after it landed. Read the issue on Linear and vllm's worker's read on VLLM-324 (certserve's comment, 7:22 PM PST) first.**

- **1. Fix B:** under vLLM's own server with the ids front, `TickFeed` (`tick_plugin.py`, lines 146 to 195) passes a lost release's request on to vLLM, the gate keeps it from the scheduler, and nothing ends it. `TickAdmit` answers such a request itself; `TickFeed` does the same.
- **2. The watermark path, ruled by me:** a client can send `POST /verity/watermark` through the certifier under the run's true id (`with_release` rewrites only the three release headers). The certifier turns away a client's request to that path and never forwards it.
- **First, no code:** read both at `main` and send me the lines cited, whether each is as vllm's worker says, and for the second which of our open pull requests touch the same function (#9457 changes how the certifier turns a request away). I then say which head it is cut on.
- **Then:** a test for each; the pull request names VLLM-324 and `Refs NET-418`; open unmarked. In review Tue 17 Nov. Tell me by Fri 13 Nov 6 PM PST the hour you can open it.

