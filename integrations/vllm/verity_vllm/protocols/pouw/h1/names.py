"""How H1's served window names and salts its registered tables (`pouw_h1_served`'s, which re-exports these): the window
Program's parameter prefix (`PREFIX`) and each table's nonce in the hm96 salt stream of the
period's key (`nonce`, `B_NONCE`, `IDS_NONCE`, `RECEIPT_NONCE`, `LIVE_NONCE`). It imports nothing that imports `pouw_h1`, so
the Commit's tie (`pouw_h1_commit.reuse`), which `pouw_h1` imports, reads them."""
from __future__ import annotations

from verity.protocols.pouw.verifier import weights as Wt
from verity.protocols.pouw.prover.circuit import h1window as HW
from verity_vllm.protocols.pouw.h1 import requests as RQ

#: the window Program's parameter prefix, the same for every window of a period
PREFIX = "h1/"
#: the hm96 salt stream's first nonce of the window's registered tables, past every round's and call's tables
NONCE = RQ.REQUESTS_NONCE + (1 << 42)
#: ``b/<K>``'s nonces, past every call's two tables: the auditor's registration's (`weights.B_NONCE`)
B_NONCE = Wt.B_NONCE
#: round s's input-ids table's nonce is IDS_NONCE + s, past every ``b/<K>``'s
IDS_NONCE = B_NONCE + (1 << 39)
#: call c's receipt table's nonce is RECEIPT_NONCE + c, past every round's input-ids table's
RECEIPT_NONCE = IDS_NONCE + (1 << 38)
#: the live tables' nonces (`pouw_h1_live.nonce`): requests@0's LIVE_NONCE, round s's slots@<s>'s LIVE_NONCE + 1 + s
LIVE_NONCE = RECEIPT_NONCE + (1 << 37)


def nonce(call: int, table: str) -> int:
    """The salt nonce of call ``call``'s table ``table`` (`h1window.TABLES`): a step's A table of K takes its first call of
    K's ``a`` nonce, which no other table takes."""
    return NONCE + 2 * int(call) + HW.TABLES.index(table)


__all__ = ["B_NONCE", "IDS_NONCE", "LIVE_NONCE", "NONCE", "PREFIX", "RECEIPT_NONCE", "nonce"]
