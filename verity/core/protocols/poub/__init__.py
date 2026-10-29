"""The remote bandwidth audit v0 (`verity/poub/v0`): how many reads of a node's registered blocks the node
made, window by window, certified by a verifier with no hardware at the site.  The store is encoded under kv/v0's codec,
the one its registration names, p = 2^1024 − 105 masked at k = 3 by default (`codec`); the challenger's live coins,
committed before the session, are each window's challenge and its openings coin (`coins`); within a window each pass
reads a public plan of distinct blocks under a seed chained from the challenge and the root before it, and its receipts
are of the kind the registration names (SHA-512 over a pass-keyed ±1 sketch of the block by default, or over the block
itself for comparison), of the block alone unless the registration binds PoUW's tile outputs (`receipts`); the
lifecycle, the record, its γ terms and pacing (a live coin every k passes) are `audit`; the declared outputs, γ at
both R and the memory line, are `ledger`'s, from the record; `memory_map` is serving's record of the KV blocks it wrote
and freed (memory-map/v1) and turns it into the audit's appends, releases and plan changes; `timing` simulates the
honest schedule and evaluates the sketch's rank bound; `p2` is the audit over PoUS's P2-encoded weights store, its
receipts of units of four 2,056-byte blocks, checked on PoUS's vk.  `PROTOCOL.md` is the specification.
"""
