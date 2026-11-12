[2026-11-12T20:12Z] 6059e96ebd50
PUT, Thu 12 Nov, 12:12 PM PST

**All three checks pass; putting now.**

1. Fetch: art:55b35cfb… fetched to a directory outside any checkout; each of bodies/*.json has the sha256 bodies.json names (algorithm-policy 864688df47ed85f4…, frontier-fit a1c878ff75814936…, standard-workloads 78a409b4509f6cfa…, size-scaling d59b340eedd613ce…). RAN (sha256sum on the fetched files, against bodies.json).
2. Checkout: origin's cursor/training-put-55a1 is d76ae46e6f005334c2e4d8c952a85282fdf48d4d (git ls-remote); a detached worktree at it, 0 status lines, te_core tree 11f297ba3e3dcb53…. RAN.
3. Nobody PUT since staging: a GET at 2026-11-12T20:11:38Z gives each of the four panels the same sha256 and updated_at as served/meta.json (algorithm-policy 2026-11-11T05:39:00Z, frontier-fit 2026-11-05T04:47:24Z, standard-workloads 2026-11-05T07:28:18Z, size-scaling 2026-11-04T18:06:50Z), and the GET's panels.json is byte-identical to served/panels.json. The route also serves training/methods, which none of the four puts touch. RAN (te_core.console.served from the clean checkout).
4. Dry puts: the README's four lines with --dry, from benchmarks/covert_witness of that checkout, each print "console.put would refuse: nothing" with the sha256, bytes, rows and note length bodies.json has, rc 0; each output is byte-identical to S/dry/NAME.txt. The checkout still has 0 status lines afterwards. RAN.

