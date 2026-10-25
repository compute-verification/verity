"""Stage records, typed: one frozen dataclass per record kind one stage writes and another reads.

A record's `load` reads the JSON as its writer writes it and refuses a missing required field, naming the file and the field
(`fields.MissingField`).  A field a writer may leave out is `X | None`, with the reason beside it; nothing is defaulted.  The
package imports nothing else from `verity_vllm`, so every stage may import it.

* `required_manifest` -- the required-value manifest (`verity-sweep/required-manifest/v1`), as the replay's coverage reads it.
* `query_checks` -- the sampled replay's reconciliation against Q(P) (`population.query_checks`), as the C2 rules read it.
* `served_tokens` -- the driver's served tokens of one run (`capture/tokens.json`), as GM-01's chronology check and the padded
  conversion (`commit.padding_steps`) read them.
* `declared_stop` -- the workload Program's declared stop rule and LAG (`stop_rule`, `lag`), as the same two read them.
* `workload` -- the workload (`verity-vllm/workload/v2`, and the v1 layout older runs hold), as the stochastic replay reads it;
  its `load` also refuses a field no writer writes (`fields.UnknownField`), and `dump` writes it back.
* `global_match` -- GM-01's Match result (`cb-a/global-match/v1`), as the executed prefix's link to the Match account reads it;
  `load` refuses an unknown field and `dump` writes it back, as `workload`'s do.
"""
