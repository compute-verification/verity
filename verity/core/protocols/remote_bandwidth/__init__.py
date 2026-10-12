"""The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`): how many reads of a node's registered blocks the node
made, pass by pass, certified by a verifier with no hardware at the site.  The store is encoded under kv/v0's codec
(`codec`); the challenger's live coins, committed before the session, seed one pass each and draw each window's
openings (`coins`); each pass reads a public plan of distinct blocks, and its receipts are bound to PoUW's tile outputs
(`receipts`); the lifecycle, the record and its γ terms are `audit`; `timing` simulates the coin stream.  `PROTOCOL.md`
is the specification.
"""
