"""Production-profile cost measurement (CONTRACT.md §16.7, §12): same pod, same workload, three
configurations (ordinary optimised vLLM -> batch-invariant optimised -> batch-invariant eager).

``throughput.py`` runs ON THE POD (one process per model x configuration: env flags must be set
before vLLM is imported); ``summarize.py`` runs anywhere (CPU, JSON only) and turns the raw rows
into Table 4 rows plus a Markdown table with the C1/C0 and C2/C1 ratios. Row schema:
``throughput.ROW_SCHEMA`` (documented in ``throughput.py``).
"""
