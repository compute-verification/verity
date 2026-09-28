# One-stage audit (`verity-one-stage`)

The current audit protocol. Serving commits every proof unit's boundary and registers the roots; the verifier
draws units once, from its own randomness (no beacon, nothing derived from a commitment); the drawn units are proved and
verified; the audit record carries an `IntegrityProfile`. The protocol is `PROTOCOL.md`. The two-stage protocol is
`protocols/sampled_proofs`.

`verity_one_stage` holds the protocol's objects and checks, not the sampler: the verifier samples the draw (the Lean
verifier's `flock-verify draw`), and this package holds it to the registration and the law.

- `registration`: the registration record (`verity/registration/v1`) and its checks, fixed before the draw.
- `draw`: `check_draw` (a draw against the registration and the law) and `check_proved` (the proved units against the draw).
- `partition`: the partition object (`verity/partition/v1`) with the queries `Q_template_instance` and `Q_template_instances` v0.
- `audit`: the audit record and its integrity profile; `verdict`: a check's outcome.

`benchmarks/one_stage` runs the audit end to end with C-Flock's circuit prover (M0) and the Lean verifier.

~~~sh
uv run pytest protocols/one_stage/tests        # 25 tests, in `check`
~~~
