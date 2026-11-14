"""A play whose blobs hold a pinned lookup table that no member's circuit looks up, on Lean's check of the per-member
form (`members.check`) and on the Python run of it (`members.members_run`): neither refuses it for the table, and both
answer as they do without it. A member is held to its own circuit's tables among the blobs (`OneStageZk.ownTables`),
so a pinned table none of them looks up is read by no member.

The window is the verifier's own tests' (`test_members_twin.Audit`: two members of stand-in files, the sessions' trees
on a challenger session), audited with a statement of its second member alone. v2 finds a blob by the SHA-512 of its
bytes, so the table is ``rcp`` itself, which the prover's `ir_lower.write_tables` writes and the verifier's suite does
not load: that is why this row is here. The other way, a circuit that looks up a table the blobs lack, is that suite's
(`test_a_member_no_statement_names_is_held_to_its_own_circuits_lookup_tables`)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from verity.protocols.proofs.spot.verifier.one_stage import members as M, proofs as PF
from verity_flock import ir_lower as IL

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity/protocols/proofs/spot/verifier/lean/tests"))
import test_members_twin as TW

#: slow: it runs v2's executable, which the first test to need it builds
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


def test_blobs_holding_a_table_no_circuit_looks_up_are_not_refused_for_it(tmp_path):
    """The audit's statement is of the second member alone, and neither member's circuit looks up a table. With
    ``rcp``'s table among the blobs, Lean and the Python run get as far as the first member's circuit, a stand-in that
    does not parse, as the same play does without the table: the table is none of a member's own
    (`OneStageZk.ownTables`), so no member's statement refuses for it."""
    a = TW.Audit("members-tables-held", opened=(TW.B,))
    assert [st["member"] for st in a.evidence["proofs"]["statements"]] == [TW.B]
    IL.write_tables(tmp_path / "tables")
    rcp = a.put((tmp_path / "tables" / "rcp.u32").read_bytes())
    assert rcp == PF.MUFU_SHA512["rcp"]

    held, (run, refused) = a.ask(tmp_path / "holds")
    assert run is None and held == M.run_reason(run, refused), (held, refused)
    assert held.startswith(f"proofs: member {TW.A}: {M.UNREAD[0]}"), held

    why, (run, refused) = a.ask(tmp_path / "without", without=(rcp,))
    assert run is None and why == M.run_reason(run, refused), (why, refused)
    assert why == held, (why, held)
