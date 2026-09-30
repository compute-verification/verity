# Sampled proofs (`verity-sampled-proofs`)

The two-stage protocol: the prover commits a transcript's boundary, replay units are drawn and replayed, their interiors
are committed, and a second draw picks the proof units checked inside each replayed unit (`VU` in its code and domain
names). The verifier draws both stages from its own randomness (`OwnRandomness`), each after the registration it follows;
there is no beacon. The protocol and its law are `PROTOCOL.md`.

The current audit protocol is the one-stage audit (`protocols/one_stage`), where the verifier draws once from its own
randomness. The vLLM integration's challenge draw (`verity_vllm.commit.challenge`) imports this package's law.

- `law`: the two-stage law (`TwoStageLaw`), the verifier's randomness (`OwnRandomness`) and samplers
  (`select_replay_units`, `select_verification_units`); `ru_key` and `vu_key` remain only for vLLM's legacy challenge.
- `plan`: verification plans and obligations over IR Programs.

~~~sh
uv run pytest protocols/sampled_proofs/tests   # 17 tests, in `check`
~~~
