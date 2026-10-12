"""The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`): how many reads of a node's registered blocks the node
made, window by window, certified by a verifier with no hardware at the site.  The store is encoded under kv/v0's codec,
the one its registration names, p = 2^1024 − 105 masked at k = 3 by default (`codec`); the challenger's live coins,
committed before the session, are each window's challenge and its openings coin (`coins`); within a window each pass
reads a public plan of distinct blocks under a seed chained from the challenge and the root before it, and its receipts,
of the kind the registration names (SHA-512 over the block by default, or over a pass-keyed sketch of it), are bound to
PoUW's tile outputs (`receipts`); the lifecycle, the record and its γ terms are `audit`; `timing` simulates the honest
schedule.  `PROTOCOL.md` is the specification.
"""
