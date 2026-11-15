"""`credit.period_credit` verifies each signature once: a period's check reads the auditor's ledger 5W + 1 times, at one
key, and `credit._once` answers every repeat from the first."""
from __future__ import annotations

import random

import pouw_credit_cases as credit_cases
import pouw_period_cases as cases
from verity.protocols.accounting.pouw.verifier import period_check as PC


def test_each_signature_is_verified_once(monkeypatch):
    d = credit_cases.period_instance(random.Random("once/2"), "honest", windows=2, **cases.COVERED)
    asked: list[tuple[bytes, bytes]] = []
    verify = PC.CC.lms_verify

    def counted(key, message, signature):
        asked.append((message, signature))
        return verify(key, message, signature)

    monkeypatch.setattr(PC.CC, "lms_verify", counted)
    assert PC.case(cases.serve(d))["verdict"] == "accept"
    assert len(asked) == len(set(asked)) >= len(d["ledger"])
