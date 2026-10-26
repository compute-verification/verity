"""verity_vllm: a version-pinned eager-vLLM observer whose only product is a raw observation log, and a pure
(torch-free) resolver + versioned-memory fold that turns that log into an unexpanded Verity Program.

    eager vLLM -> raw observations -> resolver + memory fold -> specializations + instances -> Program

Runtime side (imports torch / vLLM / triton): `observer`, `triton_adapter`, `storage`, `vllm_adapter`, `census`,
`noninterference`, `bench`.  Pure side (no torch): `events`, `log`, `tree`, `views`, `memory`, `resolver`,
`profiles`, `fold`, `compare`.  The contract between the two halves is the event schema in `events.py`
(`verity-capture/log/v1`); see `DESIGN.md`.

Vocabulary (the Verity IR design): Definition -> Specialization -> Instance -> subcircuit.  The
runtime artifact is the *raw observation log* (never "trace"); "execution transcript" is reserved for the IR.
"""

__all__ = ["SCHEMA"]

SCHEMA = "verity-capture/log/v1"
