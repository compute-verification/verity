[2026-11-10T04:52Z] c6a6090811ad
From the network-accounting coordinator, Mon 9 Nov 8:52 PM PST. **Infra took INF-664 and answers by Tue 10 Nov 6 PM PST. Its first read, which it calls unchecked, changes what your Wed noon numbers can be measured on.** Read its comment on INF-664 (Mon 8:26 PM PST); its worker TXHOST posts findings there by Tue noon PST.

1. **No host of ours has a NIC that sets the release time.** Every Nebius node is a VM with a virtual NIC. A NIC with launch time (Intel I210, I225, I226) is a bare-metal rental, and a new provider's account and billing are Daniel's. Do not plan Wed noon's numbers on it.
2. **Root for the idle states: yes.** Infra would hold `/dev/cpu_dma_latency` at 0 for the length of your run from a root-owned unit, on a node it names. `idle=poll` needs a reboot and infra would avoid it.
3. **Two hosts: yes,** a pair of CPU nodes on the private network; infra names the pair and tests the route.

**So for Wed 11 Nov noon PST:** measure on what infra can give: the pair of CPU nodes, with and without `cpu_dma_latency` held at 0, the release timed in software (`SO_TXTIME` with the ETF qdisc and no offload, if the virtual NIC's driver takes it; else the pacer as it is). Say on INF-664, as a worker's comment, how long a run is and what you need the unit to hold, so infra can name the node. In your numbers, keep apart what the VM's host adds and what our process adds, and say what a NIC-timed host would be expected to remove. That tells me whether to ask Daniel for the rental.

