# Changelog: `verity_numerical` (the bench renderers and their rules)

## 2026-09-26: Table 2's columns are headed A-GKR, B-Ligero, C-Flock and D-SP1 (labelled change to published renders)

- The views' markdown prints each family column under its Glossary name: `SP1` is now `D-SP1` and `Flock` is now `C-Flock`,
  in Table 2's header, Table 3's cell labels and Table 1's family rows. A-GKR and B-Ligero are unchanged.
- The entities render follows: `table2.column_labels` maps each family to the same name, and `backend_families[].column`
  is that header.
- Column ids stay `SP1` and `Flock`: they key the cells, `--format json`'s `table2.columns` and the parity gate against
  the frozen tables (`tables.py`, whose own headers are unchanged). The first published render after this change differs
  from the previous one only in those headers and labels; re-record the parity baseline with it.

## 2026-09-26: the README Glossary's terms in docstrings and help

- Docstrings, comments and `--help` say subcircuit, backend, statement and input set where they said target, candidate,
  configuration and instance set. Output is unchanged: printed tables, entity fields, fingerprint fields (`profile`,
  `instances`) and the frozen tables' headers keep their names, mapped in the Glossary.

## 2026-09-26: file-verified interactive results pay their protocol's rounds

- Bug: `views.interaction` gave every file-verified result `rounds = 0`, so all 14 B-interactive cells (B-Ligero, a 3-round
  interactive protocol, verified from its replayed transcript) were modelled as non-interactive. The reference network
  undercharged them and RTT sweeps left them flat.
- Root's ruling (2026-09-26): a file-verified result of an interactive protocol (its configuration is not transferable) uses the
  protocol's sequential depth as its rounds: the per-proof depth (`rounds.sequential_depth`, else `rounds.protocol`) times the
  sub-batch proofs that run one after another. With `software.backend.pipeline` proofs in flight, that is
  ceil(`N_subbatches` / pipeline) waves; without a pipeline, all `N_subbatches`. The measured RTT stays null. Fiat–Shamir
  results (A-GKR, SP1, B-Ligero FS) keep 0.
- The interaction record, and the entities render's `interaction`, gain `rounds_source`: `measured`,
  `derived: file-verified` or `fiat-shamir`. Markdown shows `<rounds> (derived)`, and the Table 3 interaction note now
  counts rounds × 1 ms.
- Effect at the reference network: the 14 B-interactive results rise by 0.9–5.5% (12 to 147 rounds each). The legacy
  parity is unchanged.
- Entities: `throughput.instances` is now `throughput.inputs`, following `interaction.inputs`.
- Entities: `interaction.measured_rtt.method` is now an enum (`views.RTT_METHODS`: `tcp-connect`,
  `open-connection-ping`, `unclassified`). The recorded `rtt_method` text names hosts and ports, so it stays in the result's
  own record in the evidence store. The markdown still prints the recorded text and is byte-identical.

## 2026-09-26: "instance set" becomes "input set"

- `census/instance_sets.json` and its schema are now `census/input_sets.json` and `schemas/input_sets.schema.json`. The
  reader's `census.INSTANCE_SETS` is now `census.INPUT_SETS`.
- In the entities render:
  - results, their `key` and subcircuits carry `input_set` (and `input_set_provenance`) instead of `instance_set`;
  - a result's `interaction.instances` (the inputs per timed run) is now `interaction.inputs`;
  - the embedded census section is now `input_sets`;
  - result ids are unchanged in shape: `<backend>@<statement>@<input set>@<prover hardware>`.
- Recorded names keep the old word, because fixtures and results carry them: the `instance-equiv/v1` artifact kind, a
  result's `instances` ref, and the `tier` field. Printed markdown strings such as reason I, "another instance set", are
  also unchanged, so renders stay byte-identical.

## 2026-09-25: `views --format entities`, the docs site's render

- `python -m verity_numerical.bench.views --root <store> --format entities [--published]` prints the render as entities
  (schema `verity/tables-entities/v1`; the site's spec is the ontology's "A result" and "Presentation rules"):
  - `subcircuits`: one per subcircuit, a fully bound subcircuit template. Each has its census id and display name (e.g.
    `gemm-coordinate/k1536/sm90-wgmma-e4m3`, "GemmCoordinate<1536> · sm90 wgmma · FP8 (E4M3)"), its
    `subcircuit_template` with its bound `parameters` (`K`, `datatype`, and `semantics`: the exact instruction),
    `legacy_target`, `fidelity_notes`, the instance set with its provenance (captured or synthetic), the census hardware
    line, the datatype display and native N;
  - `backend_families`: A-GKR, B-Ligero, C-Flock and D-SP1, each with its markdown column header as `column`. The
    markdown itself is unchanged;
  - `backends`: one per backend family and interaction mode (`A-fs`, `B-interactive`, `B-fs`, `C-interactive`, `D-fs`),
    plus composed backends under their own id. Route (a) is `A-route-a`: it renders in the A-GKR column, and its profile
    adds C-Flock's assumptions and the sigma link's. Each backend's claims are written out in full;
  - `statements`: one per subcircuit and commitment scheme, `<subcircuit>+<scheme>`: the full committed relation a backend
    lowers and proves, with its scheme's assumptions. The renderer's configurations are an implementation detail
    (`implementations` names them) and are not emitted. A backend's profile leaves out statement-commitment hashes, which
    belong to the statement;
  - `schemes`: one per scheme, under ids that name the leaf hash (e.g. `frame-v3/blake3-keyed`; `legacy_id` is the
    renderer's). Each has its tree and leaf hashes, a structured `algebraic` flag, its assumptions (taken from the code)
    and the `leaf_circuits` that commit it. `blake3-xob` commits keyed BLAKE3's bytes (same schema and params in
    `ligero-verify`), so it is a circuit of `frame-v3/blake3-keyed`, not a scheme of its own;
  - `results`: the best result of every admissible backend, not only each family's pick (`best` marks that pick), keyed by
    backend × statement × instance set × prover hardware. Each result carries:
    - its `backend`, its `statement` and the `key` of those four;
    - its full assumption set: the backend's plus the statement's;
    - `hardware` (the semantics chip, which supplies N) and `prover_hardware` (where the prover ran, which supplies P);
    - the instance set with its provenance;
    - its soundness: the whole-cell bound, a union over the sub-batch proofs, plus the backend's per-proof bound where the
      result records it;
    - `N` (the native peak: a work-model version plus a census peak id) and `P` (instances per second at the reference
      network) beside the overhead;
    - its `interaction` record, enough to recompute the overhead at any network (`formula`):
      - `rounds`, `rounds_protocol` and `sequential_depth`;
      - `bytes_up` and `bytes_down`;
      - `commit_s`, `prover_compute_s` and `verifier_compute_s`;
      - `measured_rtt` and `loopback_per_round`, each with its method;
    - its phases: stages, fused stages as one span, "Other" clamped to 0 in [−0.5%, 0) and flagged when it is more
      negative, with the raw value kept as `other_raw`, and one-time setup in its own field;
    - its state, artifact and render time;
  - `best_standard_hash`: for each backend family and subcircuit, the lowest-overhead cell on standard hashes only;
  - `tables`: Table 1 lists backends (guarantees, assumptions, unmapped texts). Table 2 and 3 carry every printed cell's
    exact display string, value, marks and footnote; their rows are keyed by statement and prover hardware, and each cell
    names its backend;
  - `claims`: the `verity.claims` entries cited; `census`: the `census/` entries as read at render time;
  - `provenance`: the render time and the verity commit.
- Ids are stable across renders: subcircuits by census id, backends as `<letter>-<mode>` or a composed backend's own id,
  statements as `<subcircuit>+<scheme>`, and results as `<backend>@<statement>@<instance set>@<prover hardware>`.
- The markdown is unchanged byte for byte. Table 2's cells and footnotes and Table 3's rows now come from shared helpers
  (`table2_grid`, `table3_row`) that both renders use.
- The steward's `published = "views"` entry also writes `<stamp>-tables.json` (the entities render).

## 2026-09-25: peaks, networks and instance sets come from the census (`census/`, data only)

- The renderers read these values from `census/` at the repository root:
  - the native peaks;
  - the reference network (1 ms, 100 Gb/s) and the interaction drill-down's sensitivity profiles;
  - K and B;
  - the frozen instance sets.

  `census/` is data only: five JSON files, each with a JSON Schema, and a README. Each entry has a stable id, such as
  `a100-sxm4-80gb/bf16`, `dc-1ms-100gbps` or the subcircuit template `gemm_coordinate` with its bound subcircuits, and
  its sources.
- `verity_numerical.bench.census` reads the files directly (`$VERITY_CENSUS` overrides the location), and
  `tests/bench/test_census_data.py` validates them against their schemas.
- No value changed. Renders are byte-identical, and `--parity` is unchanged.
- Core holds no peak. `verity.proofs.target.Target` names only its `anchor_device`; `native_peak` and `NativePeak`
  are gone.
- `contract.native_peak_of(target)` returns the census line for the target's anchor device and operand dtype.
- `latency.pipeline` takes `native_s_per_vu` as a parameter.

## 2026-09-25: route (a) is its own full-relation A-GKR configuration

- Route (a) (A-GKR's prime proof, the operands as keyed-BLAKE3 rows proved in Flock flock-128-r2 with live coins, joined by the
  sigma link; PROTOCOL 17.5–17.6) gets a drill-down variant and a Table 1 configuration: full committed relation over
  `frame-v3/blake3-row`, NON_ZK_PROOF, 2^-130.19 (red-team-flock's fifth audit), live coins, collision resistance a Table 1
  assumption (decision 57). Its results fill the A-GKR column; plain A-GKR GPU results still read "weaker statement".
- The spec views read a result registered with a lane summary as its meta and the envelope as its JSON payload (as
  `cell.py rederive` registers art:4b52879f), as the drill-down already did. The legacy rules still read the meta only, so
  `--parity` is unchanged (15/15).

## 2026-09-25: live cells count prover compute once, and loopback coin handling as compute

- A live result whose `interaction` block says `t_total_includes_wait` (Flock's live coins) takes its prover compute from
  `live.prover_compute_seconds` (else `t.total` minus the wait). Before this, the network time was counted twice: once
  inside `t.total` and again as rounds × RTT.
- The prover's own per-round live-coin handling, measured on loopback (`live.loopback_round_seconds`), is prover compute:
  rounds × that, at most the measured wait, moves out of the network wait. Only the excess over loopback counts as
  network wait, and the ±10% tolerance check compares the measured wall against compute + rounds × 1 ms + bytes ÷ 100 Gb/s.
  Applies to every live cell (route (a), Flock). The tolerance is unchanged (root's ruling).
- The ±10% check validates the model against the run itself: the formula is evaluated at the run's measured RTT
  (`net.rtt_ms`, which must be the network round trip, not a figure derived from the prover's coin waits) and measured
  bandwidth (`net.bandwidth_bps`, else 100 Gb/s), and compared with the measured wall time. The published cell is still
  the formula at the 1 ms / 100 Gb/s reference. A cell's footnote names any loopback coin-handling figure counted as
  prover compute and how it was measured (`interaction.loopback_method`; a figure from another run says BORROWED).

## 2026-09-25: the switch to the approved tables spec (a change of method)

From the 6:00 PM PT digest on 2026-09-25, the published tables are the approved spec's views (`kb/TABLES.md`),
rendered by `python -m verity_numerical.bench.views --published`. The frozen 2026-09-22 tables
(`verity_numerical.bench.tables`) are still rendered, byte for byte as before, as a drill-down beside them.

- **Method.** Table 1 lists backend configurations and their security. Table 2 is the end-to-end overhead
  N ÷ P per subcircuit, hardware class and commitment scheme. N is the compute-bound datasheet estimate. P is
  instances per second, counting commitment, proving and the reference network (1 ms RTT, 100 Gb/s). A cell must
  prove the full committed relation over a scheme defined in `verity.commitments`. Table 3 splits each cell into
  serving (commitment), interaction and proving overhead. 2^-128 becomes the view's default filter, not a validity
  rule. The fixed B = 4096 and K = 1536 checks give way to the plateau of a recorded batch sweep. Algebraic hashes
  are admitted and flagged `(alg.)`.
- **Old numbers.** None of the 15 frozen cells meets the new rules as it stands. The A-GKR and bare B-Ligero
  cells prove a weaker statement (reason R). The B-Ligero + in-proof hash cells have no recorded protocol or plateau
  sweep (reason M). The switch digest (`python -m verity_numerical.bench.switch`)
  shows each old number beside its replacement, or beside the reason it left.
- **Parity.** Under today's parameters (`--rules legacy`), the new views select exactly the frozen cells, with the
  same artifacts and ratios. `views --parity --take-snapshot --record` checks this against a fresh
  `tables --snapshot` and records the check as a `tables-parity/v1` artifact.
- **Flock column.** Flock is its own backend family and gets its own column in Tables 1–3 (Daniel, 2026-09-25).
  That column holds pure-Flock configurations only: the relation and its row hashes in one binary-field circuit.
  Route (a) (the A-GKR prime side, Flock's row hashes and the sigma link) is an A-GKR configuration and stays in
  the A-GKR column. Until a Flock result qualifies, the column renders `—` with its reason.
- **Steward.** A `[[render]]` entry with `published = "views"` writes the views as `<stamp>-tables.md` and the
  frozen tables as `<stamp>-tables-frozen.md`. Without it, the entry renders the frozen tables as before.
