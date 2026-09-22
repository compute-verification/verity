"""Checker v2 (PROTOCOL.md §14) in the GPU prover: the BabyBear export of ``gkr_export.v2``, the ``chain.txt``
link list, and the v2 negatives (30 unit + 4 chain cases, plus ``vu-k1536-neg``).  The prover, verifier and kernels
are the shared ones in ``gpu/``; what v2 adds there is additive (``itable`` parsing + device materialisation,
``R<w>`` range tables from the queries, ``Chain.links``, the table union over segments)."""
