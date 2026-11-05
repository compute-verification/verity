"""Stage records, typed: one frozen dataclass per record kind one stage writes and another reads.

A record's `load` reads the JSON as its writer writes it and refuses a missing required field, naming the file and the field
(`fields.MissingField`).  A field a writer may leave out is `X | None`, with the reason beside it; nothing is defaulted.  The
package imports nothing else from `verity_vllm`, so every stage may import it.

* `required_manifest` -- the required-value manifest (`verity-sweep/required-manifest/v1`), as the replay's coverage reads it.
* `query_checks` -- the sampled replay's reconciliation against Q(P) (`population.query_checks`), as the C2 rules read it.
* `served_tokens` -- the driver's served tokens of one run (`capture/tokens.json`), as the global match's chronology check and the padded
  conversion (`commit.padding_steps`) read them.
* `declared_stop` -- the workload Program's declared stop rule and LAG (`stop_rule`, `lag`), as the same two read them.
* `workload` -- the workload (`verity-vllm/workload/v2`, and the v1 layout older runs hold), as the stochastic replay and
  the global match's declared side (`check.match.declaration`) read it;
  its `load` also refuses a field no writer writes (`fields.UnknownField`), and `dump` writes it back.
* `global_match` -- the global match's Match result (`cb-a/global-match/v1`), as the executed prefix's link to the Match account reads it;
  `load` refuses an unknown field and `dump` writes it back, as `workload`'s do.
* `build_summary` -- the Build's summary (`build_summary.json`, in its `tp1`, `tp` and `tp2-r19` layouts), as the verdict reads the
  Build's correspondence digest; `load` refuses an unknown field and `dump` writes it back, as `workload`'s do.
* `commit_run` -- the Commit record, one run of a pair (a line of `commit/runs.jsonl`: `verity/commit-integ/delta/v1`, the TP
  writer's `delta-tp/v1` and a sealed serving window's), as the verdict reads it; `load` refuses an unknown top-level field.
* `commit_binding` -- its `binding` block: the identity coverage, the executed prefix of record with its two account links, the rows
  past the stop and the population, as the Commit's rules and the not-executed class's facts read them.
* `weights_registration` -- the committer's weights registration (`verity-vllm/weights-registration/v1`): each weight
  parameter's hiding root, as the verifier's pins and a window's registration read it; its `of` refuses an unknown field
  and `dump` writes it back.
* `inference_blocks` -- the inference-block table's public statement (`verity-vllm/inference-blocks/v1`): its budgets, padded
  counts and hiding root, as B emits it beside the window circuit (`program.predict.inference_blocks.commit`); its `of` refuses an
  unknown field and `dump` writes it.
* `predicted_row` -- a predicted row's record (`verity-vllm/predicted-row/v1`, ``predicted.json``): the Build of record the
  ahead-of-time compiler wrote and its manifest's sha256, as ``predict compare --predicted`` reads it; its `of` refuses an
  unknown field and `dump` writes it back.
"""
