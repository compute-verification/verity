
04:35Z Tue (coordinator): the CPU fleet may stop at Tue 06:00Z except vy-nebius-cpu-6 (held to Wed 12:30Z); I asked infra to keep one more node. If a run of yours is still queued near 06:00Z, move it to cpu-6 rather than lose it.

04:50Z Tue (coordinator), correction: nothing stops tonight. cpu-1, cpu-2, cpu-4 and cpu-9 run to **Tue 10 PM PST (Wed 06:00Z)**; cpu-5 and cpu-6 run to **Fri 6 Nov 10 AM PST**. Queued runs don't need moving.

06:55Z Tue (coordinator), training on inference-only over a circuit cut at the linears (TOP-159): **each cut linear needs its own count.** Its gates go into its session's block by shape, and c is calibrated the same way. Its y counts as internal input only where the record ties y to PoUW's credited product for that session. Where it's untied, the verdict says so plainly. So the cut must carry each linear site's shape (m, k, n and the scheme) in the registered circuit, for the counts to add. pearlplus keeps that in the cut; ibcounts' tool adds the cut linears' gates by shape and marks y tied or untied. Read training's details on TOP-159.
