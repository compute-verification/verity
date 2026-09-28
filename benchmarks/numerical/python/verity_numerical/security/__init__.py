"""Security accounting for the campaign's proof candidates (cryptographic-composition lane).

``accounting``: soundness-error calculator for A-GKR (``archive/gkr/PROTOCOL.md``) and B-Ligero
(``archive/direct/PROTOCOL.md``).  Exact rationals, decomposed by component; refuses to turn a field size into
"bits of security".

``census``: the checker census from the WP2 gadget contracts (``verity_numerical.checker``), the source of every
per-unit default in ``accounting`` (tables, layered shape, committed columns per unit) and of the diff against
``verity.ml.circuits.silicon.relation.vu_census``.
"""
