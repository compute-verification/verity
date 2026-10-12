"""The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`): how many full reads of a node's registered store the
node made in each window, certified by a verifier with no hardware at the site.  The store is encoded under kv/v0's
codec (`codec`), passes are chained by their roots and their receipts bound to PoUW's tile outputs (`receipts`), and
the lifecycle, the record and its γ terms are `audit`; `timing` simulates the remote schedule.  `PROTOCOL.md` is the
specification.
"""
