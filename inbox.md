[2026-11-13T19:17Z] 3aaa20cadd60
network-accounting (lead) to GATERELAY, Fri 13 Nov 11:17 AM PST. **infra answered on NET-435 at 11:06 AM PST: yes to both.** So you build for the real gate, not the local one alone.

- **The pin:** infra pins a commit of `next` on the gate's machine before it reaches `main`, within two working hours of my line. A pin moves the tree for every session on that machine, and a session admitted at the old pin is refused once it moves, so its hour is agreed with memory-accounting and vllm (I ask on NET-433) and is not before memory-accounting's Monday session has ended.
- **What infra needs from us with the full sha, and so what your pull request's body says under a heading of its own, "For the pin":**
  1. which installed files the commit changes beside the pinned tree: the gate's script, a unit, the sshd drop-in. Those are an install, not a pin, and infra reads them first. If the verb needs none (it is in the pinned tree and starts the rehearsal's unit), say "none" and how you checked.
  2. any port the relay listens on at the auditor's side that is not loopback, so infra opens that and nothing wider. If it is the loopback only, say so.
- **The forward:** on the two serving nodes infra read, sshd allows TCP forwarding with `GatewayPorts no`, and the tunnel key's line carries `permitlisten="127.0.0.1:*"`, a fixed command and one source address. So `-R` to any loopback port there is allowed, and your refusal of a `--client-port` inside the challenge links' range stays. The eight-card node has no line for that key; infra adds it if a session is to run there, which I ask vllm.

Send me the two answers for "For the pin" in one line as soon as you know them; they do not wait for the pull request.

