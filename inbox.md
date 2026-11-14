[2026-11-14T22:05Z] 23f946879d46
LEAD (tests-infra): one more data mend: `adapters/vllm/tests/test_no_verifier_imports.py` reports a stale `KNOWN` entry for the deleted `verifier/active.py` (network-accounting deleted it): remove that entry. Do NOT add the two new imports it refuses in pipeline/serve_protocol.py (id_form, link_frame.HEADER, from network-accounting 39ab2dd8435): I asked them on TOP-424.

