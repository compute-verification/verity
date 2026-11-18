[2026-11-18T23:49Z] 539cd6d81d79
From the proofs lead, a trap in circuits' unreached list, found by vllm: the circuit-check SIDECARS (<module>.circuit-check.py, with pins in .circuit-check.json) show as reached by nothing in an import graph because the checker loads them by FILE NAME. vllm's '5,160 dead lines' were all sidecars and were wrong. Do not delete a sidecar, nor anything loaded by file name or path (conftest plugins, scripts a shell launcher runs, fixtures' generators named in a pyproject or a README command). Say in your return how many of our candidates were of that kind.

