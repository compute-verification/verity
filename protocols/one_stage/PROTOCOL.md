# One-stage audit

Every proof unit's boundary is committed while serving; the verifier draws units once, from its own randomness, and the
drawn units are proved. The two-stage protocol (replay, then a second draw) is `protocols/sampled_proofs`.

## Lifecycle

1. **Serving commits** the values the partition's units read and write, and sends the roots.
2. **Registration** (`registration.record`, `verity/registration/v1` of `docs/audit-protocols.md` §0.1): program, query
   and partition digests, the population the query derives, the law, the scheme, every root with its leaf count and domain,
   the leaf layer's digest, anchors, the window (provenance, with `kind` `served` or `stand-in`) and the previous record.
   The verifier checks R1-R6 against its own copies (`registration.check`), holds its own public file to it
   (`registration.matches_public`) and writes a receipt (`registration.receipt`) before it draws. Served roots use the
   domain rule agreed with vllm-serving-commit (`registration.served_domain`).
3. **The draw**: the verifier samples units from the operating system's randomness under its law (`subset:K` or
   `bernoulli:NUM/DEN`) and sends the draw in the clear, before the session's coin commitment. Nothing is derived from a
   commitment. `draw.check_draw` holds it to the registration and the law.
4. **Proofs** of the drawn units, in one session whose statement is the drawn units' (`draw.check_proved`: its instances
   are the drawn units, in order).
5. **Verification** by each verifier that runs, the Lean verifier of record among them.
6. **The audit record** (`audit.audit_record`): the registration, its receipt, the draw, the law, the proved units, the
   statement's digests, every verdict, and, when every check and verdict that ran accepted, the profile in two forms: the
   core `IntegrityProfile` (one level `unit` with `sizes = (population,)`, drawn by the law, and the bound on wrong units at
   delta) and `verity/integrity-profile/v1` (§0.2), whose ε terms come from the soundness analysis and stay null here.

## Partition

`{"program": <SHA-512 of the program>, "query": {...}}`, digest `SHA-512("verity/partition/v1\0" ‖ canon(object))`. The
only query here is `Q_template_instance` v0: each instance of one template in the program's root body is one unit, in
canonical order, with no width rule. It is a demo query for single-template runs, not the serving partition.

## Not here yet

- The serving path's own commitment: run A0 uses a stand-in, a fresh commitment to values captured from serving, and the
  registration says so (`window.kind = "stand-in"`).
- The Lean verifier of record on M0's current statement format, and the partition and program bound inside the statement.
- The protocol's security statement and its proof in Lean.

`benchmarks/one_stage/` runs the lifecycle end to end with M0's prover and the Lean verifier (`pod.sh`, `a0.py`).
