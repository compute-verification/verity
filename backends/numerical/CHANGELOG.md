# Changelog: `verity_numerical` (the bench renderers and their rules)

## 2026-09-25: `views --format entities`, the docs site's render

- `python -m verity_numerical.bench.views --root <store> --format entities [--published]` prints the render as entities
  (schema `verity/tables-entities/v1`):
  - `subcircuits`: one per subcircuit, a fully bound subcircuit template. Each has its `subcircuit_template`
    (`gemm_coordinate`), its bound `parameters` (`K`, and `semantics`: the exact instruction, e.g.
    `sm90.wgmma.m64n8k32.e4m3`) and `legacy_target` (the internal target id). Each also has the census hardware line (id,
    short name such as "H100 SXM5", full device, peak), the datatype display (the common name, then the format, e.g.
    `FP8 (E4M3)`), the tensor-core step, the instance set and native N;
  - `backend_families`: A-GKR, B-Ligero, C-Flock and D-SP1. Each keeps its markdown column header as `column`; the
    markdown itself is unchanged;
  - `backends`: the primary shape of Table 1's claims. There is one backend per backend family and interaction mode:
    `A-interactive`, `A-fs`, `B-interactive`, `B-fs`, `C-interactive` and `D-fs`, where `fs` means Fiat–Shamir. Each has its
    member configurations and its claims written out in full;
  - `configurations`: the detail behind each backend, from the spec's configurations (route (a) and Flock included). Each
    has its variant, scheme, ZK class (a short enum with the text), soundness bits, hash classes with the algebraic flag,
    and interaction and assumption texts as lists. Its `claims` list each property (soundness with its bound, ZK,
    non-interactive or designated-verifier, transparent setup) with the assumptions it rests on. Each assumption is a
    `verity.claims` id with its role as the code declares it (empty where it declares none) and its text. Text with no id
    yet is kept and flagged `unmapped`;
  - `schemes`: each commitment scheme's id, name, one-line description and the hash assumptions it rests on, taken from
    the configurations that bind it (frame-v3's SHA-256 trees included). A result's `scheme` is one of these ids;
  - `results`: one per Table 2 cell, configuration × subcircuit. Each has its backend and backend family, its overhead,
    the phases that sum to it, one-time setup kept apart, its state (provisional, algebraic, verified, under
    re-verification), the artifact and the render time;
  - `best_standard_hash`: for each backend family and subcircuit, the lowest-overhead cell on standard hashes only;
  - `tables`: every Table 1, 2 and 3 cell's exact display string, its numeric value, its marks and its footnote text;
  - `claims`: the `verity.claims` entries cited; `census`: the `census/` entries as read at render time;
  - `provenance`: the render time and the verity commit.
- Ids are stable across renders: subcircuits by census id, backends as `<letter>-<mode>`, configurations by the slug of
  their name, and results as `<configuration>@<subcircuit>`.
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
