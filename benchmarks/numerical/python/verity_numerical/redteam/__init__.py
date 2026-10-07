"""Red team: satisfying-but-wrong witnesses against the campaign's checkers, and the harness that keeps producing them.

Independent of the WP1/WP2 lanes.  Nothing here fixes ``verity.**``; a finding is a reproducer.

    harness          checker-agnostic: completeness, single-column mutations, coordinated search, a z3 driver, JSON report
    cases            the case list: the Ada golden records and the structured cases (max shift, flush, cancellation, ...)
    relation_target  the adapter for ``verity.ml.circuits.silicon.relation`` (integer and prime-field semantics) and for ``silicon.tc_dot``
    forge            closed-form forged witnesses ``relation`` accepts: three over fields below 2^52..2^64, one over any field
    z3_relation      the predicates of ``relation`` transcribed for z3: gadget and whole-group queries, integer or field
    campaign         runs everything and writes the findings report (``python -m verity_numerical.redteam.campaign``)
"""
