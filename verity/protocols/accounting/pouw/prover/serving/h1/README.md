# verity.protocols.accounting.pouw.prover.serving.h1

PoUW's H1 scheme on vLLM's served int8 linears. The package's docstring states the scheme, and each module's docstring
says what the module holds.

## Interface

- **Provides** these names to code outside the package, each in its module's `__all__`. The package gives
  `CHALLENGER`, `H1Run`, `PROGRAM`, `PROGRAM_META`, `SCHEME`, `TILE`, `program_salt` and `units`. `answers` gives
  `Answers`. `challenger` gives `REPLY_FIELDS`, `SCHEME`, `SESSION_FIELDS`, `carried`, `families`, `lms_verify`,
  `manifest_reason`, `pinned_key`, `pins_head`, `program_salt_of`, `session_of` and `window_evidence`, and three names
  for a window's sample: its session's refusal, its K, and the proof blocks core samples. `commit` gives
  `CommitRecord`, `CommitTap`, `SCHEME`, `Sites`, `registration`, `reuse`, `shape` and `site`. The module that stages
  a window's sampled proof blocks as Flock statements gives `Refused` and `run`. `handoff` gives `FILE`. `hooks` gives
  `committed`, `expert_names`, `layers`, `warm_zy`, `weight_roots` and `wire`. `kept` gives `kept_row`. `live` gives
  `dump`. `prover` gives `STATE`, `Writer`, `kept_bytes` and `private`. `ranks` gives `RankExtension`, `RankRun` and
  `RankTap`. `receipts` gives `C_`, `Call`, `Cubin`, `LANES`, `R`, `Receipts`, `call_receipts`, `call_streams`,
  `gated_receipts`, `gated_reference`, `j10r_counters`, `layout`, `noise_blocks`, `noise_blocks_cpu`, `reference`,
  `seed_record`, `seed_words`, `stream_blocks`, `tile_receipts` and `unit_seeds`. `registered` gives `commit` and
  `salts`. `requests` gives `ROUND0`. `served` gives `Served`, `declared`, `declared_layout`, `descriptor`,
  `register_weights`, `unit_of`, `weights_of` and `write_program`. `settings` gives `CHALLENGED`, `fills`,
  `from_engine`, `of` and `refusals`. `tables_device` gives `available`. `window` gives `RESERVE_FIELDS`, `Window`,
  `closed_on_error`, `connect`, `finish`, `gate_of`, `open_period`, `open_session`, `reserve_salt`, `reserved_of` and
  `salt_session`, and three names for a window's sample session: the session, its reserve at the gate, and the tests'
  sampling coin held in process. Those six names and that module are spelled with a word the Glossary retires, so this
  file describes them and their modules' `__all__` spells them.
- **Loaded by:** the adapter's `pipeline`, `acquire`, `program.predict` and `protocols.pouw` packages, and the
  benchmarks under `benchmarks/pouw/`, `benchmarks/one_stage/`, `benchmarks/prove_path/` and `benchmarks/hash/dither/`.
- **Loads:** `verity.circuits`, `verity.commitments`, `verity.protocols.accounting.pouw` and
  `verity.protocols.proofs.spot`; the adapter's `engine`, `commit`, `records`, `program.frontend`, `errors` and
  `protocols.pouw`; torch, at import in `receipts` and inside functions elsewhere; and vLLM and `verity_flock`, inside
  the functions that use them.
