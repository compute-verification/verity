"""v0's ids and domain tags as PoUB sessions recorded before MEM-152's move (28 Oct 2026) hold them: each of today's
`verity/poub/` tags to the one, of the same version, that such a session's registration digest, receipts, openings,
coins, registry and record were made under. Frozen: nothing draws, signs or hashes under them. The replay of recorded
sessions (`lean/tests/replay.py`) is their one reader: it tells such a record by the id it names, and checks what of it
today's verifiers can (its registration's digest under v0's registration tag, its registry under today's schema).
"""

from __future__ import annotations

#: The prefix of every tag below: v0's name before the move.
RETIRED = "verity/remote-bandwidth"

#: Today's tag to the one sessions recorded before the move were made under.
V0: dict[str, str] = {f"verity/poub{rest}": f"{RETIRED}{rest}" for rest in (
    "/v0", "/registry/v0", "/registration/v0", "/codec/v0", "/openings/v0", "/payload/v0", "/append/v0", "/appends/v0",
    "/chained/v0", "/stub-tile/v0", "/seed/v0", "/receipt/v0", "/sketch-receipt/v0", "/sketch-signs/v0",
    "/receipt-tree/v0", "/plan/v0", "/window/v1", "/coins/v1", "/p2/v0", "/p2-registration/v0", "/timing-jitter/v0",
    "-period/challenger/v1", "-monitoring-period/v1")}
#: v0's protocol id, which a recorded session's record names, and its registry's schema, before the move.
PROTOCOL_ID = V0["verity/poub/v0"]
REGISTRY_SCHEMA = V0["verity/poub/registry/v0"]
#: v0's registration tag before the move, NUL-terminated as `audit.REGISTRATION_DOMAIN` is.
REGISTRATION_DOMAIN = V0["verity/poub/registration/v0"].encode() + b"\0"
