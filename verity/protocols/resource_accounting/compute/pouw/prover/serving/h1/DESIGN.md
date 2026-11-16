# H1's serving side: design

What the package exports and loads, beside [the README](README.md).

## Interface

- **Provides** these names to code outside the package, each in its module's `__all__`. The package gives
  `CHALLENGER`, `H1Run`, `PROGRAM`, `PROGRAM_META`, `SCHEME`, `TILE`, `program_salt` and `units`. `answers` gives
  `Answers`. `challenger` gives `REPLY_FIELDS`, `SCHEME`, `SESSION_FIELDS`, `carried`, `families`, `lms_verify`,
  `manifest_reason`, `pinned_key`, `pins_head`, `program_salt_of`, `session_of` and `window_evidence`, and three names
  for a window's sample: its session's refusal, its K, and the proof blocks core samples. `commit` gives
  `CommitRecord`, `CommitTap`, `SCHEME`, `Sites`, `registration`, `reuse`, `shape` and `site`. The module that stages
  a window's sampled proof blocks as Flock statements gives `Refused` and `run`. `handoff` gives `FILE`. `kept` gives
  `kept_row`. `live` gives `dump`. `prover` gives `STATE`, `Writer`, `kept_bytes` and `private`. `registered` gives
  `commit` and `salts`. `requests` gives `ROUND0`. `served` gives `Served`, `declared`, `declared_layout`, `descriptor`,
  `register_weights`, `unit_of`, `weights_of` and `write_program`. `window` gives `RESERVE_FIELDS`, `Window`,
  `closed_on_error`, `connect`, `finish`, `gate_of`, `open_period`, `open_session`, `reserve_salt`, `reserved_of` and
  `salt_session`, and three names for a window's sample session: the session, its reserve at the gate, and the tests'
  sampling coin held in process. Those six names and that module are spelled with a word the Glossary retires, so this
  file describes them and their modules' `__all__` spells them.
- **Loaded by:** the serving package, whose exports are how an adapter reaches these names (the serving engine's adapter
  keeps its own H1 hooks, ranks and settings); PoUW's prover tests; and the benchmarks under
  `benchmarks/pouw/`, `benchmarks/one_stage/` and `benchmarks/hash/`.
- **Loads:** `verity.circuits`, `verity.commitments`, `verity.errors`, `verity.protocols.resource_accounting.compute.pouw` and
  `verity.protocols.proofs.spot`; numpy; and `verity_flock`, inside the functions that use it. It loads no tensor library
  and nothing under `adapters/`: a served call's int8 GEMM and dequantization are the arm the workload hands in
  (`serving.int8_arm`), and its tensors and device half are what it hands in through `serving.tensors`.
