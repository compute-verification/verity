"""The toy's NCI declaration (`nci.declaration`) on a window the torch loop served, held to NCI's check of record
(``verity/nci/v1``, its protocol difftest's `reference`) as onsite's registration check hands it the case
(`compose.registration_case`), on the registration record integrity commits to:

* its Program is the one the window's registration binds, and NCI accepts it, every unit within `nci.ALPHA`;
* the toy's policy is tight: one bit a gate less and the division doesn't comply;
* on a record naming another Program NCI still accepts it: with θ empty its check reads the record's ``program`` only per
  θ entry (``theta-unbound``), so the declaration's Program is bound to the audited one by `nci.declaration`, not by NCI.
  When NCI's check binds it, that case is refused and this test says so.
"""
from __future__ import annotations

import json
import sys

from verity_all_protocols import compose as C
from verity_all_protocols import nci as NCI
from verity_all_protocols import runner as RN

from verity.core.protocols.sampled_proofs.one_stage import registration as SP
from verity.ml.circuits import codec

COMPOSITION_TESTS = C.ROOT / "verity" / "core" / "protocols" / "composition" / "tests"
LAW = "subset:2"


def test_the_toys_declaration_is_accepted_on_its_windows_record(tmp_path):
    sys.path.insert(0, str(COMPOSITION_TESTS))
    import reference as R

    RN.serve(["--law", LAW, "--cap", "8", "--requests", "3", "--steps", "1", "--run-id", "t/w0", "--out", str(tmp_path)])
    record = json.loads((tmp_path / "registration.json").read_text())
    declaration = NCI.declaration(tmp_path, LAW)
    assert codec.program_digest(json.loads(declaration["program"])) == record["program"]

    def answer(declaration: dict, record: dict = record) -> dict:
        case = C.registration_case(declaration, {"commitment": C.canonical_json(record).hex(), "params": {"log": []}})
        assert C.registration_digest(case) == SP.digest(record)
        return R.NCI_CHECK.answer(case)

    honest = answer(declaration)
    assert honest["verdict"] == "accept", honest
    assert bytes.fromhex(honest["commitment"]).hex() == SP.digest(record)
    tight = answer({**declaration, "alpha": [NCI.ALPHA[0] - 1, 1]})
    assert tight["verdict"] == "reject" and tight["reason"] == "the division does not comply", tight
    moved = answer(declaration, {**record, "program": "00" * 64})
    assert moved["verdict"] == "accept", "NCI now binds the record's Program: assert it refuses this case"
