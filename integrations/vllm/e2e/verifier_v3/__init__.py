"""[A3 contract 2026-09-10] ``e2e.verifier_v3``: the contract-ordered verifier (docs/PROTOCOL.md steps 2 and 4 run LIVE, before ``r_q`` /
``r_s`` are derived; the post-hoc re-verification keeps every check of ``verifier_v2/reverify.py`` and adds the contract's).

    steps.py     -- step2_check / step4_check / ru_fields: pure functions the live driver calls before each challenge (CPU-testable)
    driver.py    -- the contract-ordered live driver (rec_p1_protocol.py's phases with the checks inserted, seed held in memory, epoch barrier)
    reverify.py  -- verifier_v2/reverify.py copied, then changed only as listed in its docstring (V8 contract checks; B_ex from the instance)

``verifier_v2/`` is not edited; legacy records (no epoch, no exemption request, "exempt (charged)") are REFUSED here, not re-labelled.
"""
