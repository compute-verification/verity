"""`p2_party`'s command line with the memory challenger's key `pous_test_signer.AtLeaf`'s, for a test's party in the
gate's shape (``python -S -E p2_party_at_leaf.py ARGS`` from the tree): ``--challenger DIR`` keeps its session log,
and the close is signed at S1's deployed parameters without a deployed key's keygen."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "verity" / "protocols" / "accounting" / "pous" / "verifier" / "tests")]

import pous_test_signer as ts  # noqa: E402
from verity.protocols.accounting.pous.verifier import period  # noqa: E402
from verity.protocols.accounting.poub.verifier import p2_party  # noqa: E402

period.challenger = lambda directory: ts.AtLeaf(b"p2-party-at-leaf")
raise SystemExit(p2_party.main())
