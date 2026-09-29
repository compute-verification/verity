# Sampled proofs (`verity-sampled-proofs`)

The two-stage protocol: the prover commits a transcript's boundary, replay units are drawn and replayed, their interiors
are committed, and a second draw picks the proof units checked inside each replayed unit (`VU` in its code and domain
names). Its draws go through `verity.randomness` from a beacon round or a key holder's secret. The protocol and its law
are `PROTOCOL.md`.

The current audit protocol is the one-stage audit (`protocols/one_stage`), where the verifier draws once from its own
randomness. The vLLM integration's challenge draw (`verity_vllm.commit.challenge`) imports this package's law.

- `law`: the two-stage law (`TwoStageLaw`), its draws (`ru_key`, `vu_key`) and samplers (`select_replay_units`,
  `select_verification_units`).
- `plan`: verification plans and obligations over IR Programs.

~~~sh
uv run pytest protocols/sampled_proofs/tests   # 17 tests, in `check`
~~~
