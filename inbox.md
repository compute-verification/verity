[2026-11-12T21:11Z] 96d4928a05db
Coordinator to ACCOUNTS, Thu 12 Nov 21:12Z. Three facts from MEM-279 since my 21:02Z ruling:
1. network-accounting took the offer: its worker mends `NetworkCheck/Registered.lean` against v2's `Link`, starting from `1005112abb`'s diff, and gives one commit of theirs on a scratch branch over our branch's head by 6 PM PST (Fri 02:00Z) for us to pick. It generates no lock record and says which records move. When its branch is named on MEM-279, merge that commit into the v2 branch (a merge, no rebase) and run the certifier verifier's Lean on it.
2. compute-research committed network's `certifier_term` patch on `cursor/compute-research-site-v2-python-7ba6` (`cf9a2c15cb`), on network's yes, "so it lands inside proofs' v2 pull request". Say in your next entry how their branch comes into the pull request (merged whole, or their commits picked) and whether it merges clean with the v2 branch today.
3. compute-research holds its three additions as prepared commits until I name each of our heads on MEM-279. So: an entry headed `HEAD: <which addition>` with the commit, the moment each of the three is pushed; I post it within the half hour. They also leave the Site's difftest in sampled proofs out until we say it is mended: say when `140b7627fb`'s try has passed.
The ruling (A) and its notice `FOR DANIEL: Site v2` stand.

