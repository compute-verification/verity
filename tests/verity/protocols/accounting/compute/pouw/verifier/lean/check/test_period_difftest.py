"""`verity.protocols.accounting.compute.pouw.verifier.period_check`'s form check on `pouw_period_cases`: a
window's descriptor that isn't UTF-8 is no form of its own and is left to `credit.period_credit`."""
from __future__ import annotations

import random

import pytest

import pouw_period_cases as cases
from verity.protocols.accounting.compute.pouw.verifier import period_check as PC


@pytest.mark.parametrize("raw", [b"\xff", b"\xed\xa0\x80", b'{"program_fn":"\xc0\xaf"}'])
def test_a_descriptor_not_utf8_is_left_to_the_credit(raw):
    """A window whose descriptor isn't UTF-8 (a stray byte, a surrogate, an overlong) is no form of its own, as Lean's
    `String.fromUTF8?` reads it, and the period is rejected by `credit.period_credit`, not credited."""
    assert PC._window_form("p", 0, {"descriptor": raw.hex()}) is None
    d = cases.credit_cases.period_instance(random.Random("utf8"), "honest", **cases.COVERED)
    d["windows"][0]["descriptor"] = raw.hex()
    got = PC.case_twin(cases.serve(d))
    assert got["verdict"] == "reject" and not got["reason"].startswith("form: "), got["reason"]
