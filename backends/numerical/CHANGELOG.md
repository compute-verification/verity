# Changelog: `verity_numerical` (the bench renderers and their rules)

## 2026-09-25: peaks, networks and instance sets come from `verity.census`

- The renderers read the native peaks, the reference network (1 ms, 100 Gb/s), the interaction drill-down's sensitivity
  profiles, K and B, and the frozen instance sets from the core registry `verity.census`, where each has a stable id
  (`a100-sxm4-80gb/bf16`, `dc-1ms-100gbps`, `vu-k1536`, ...). No value changed: renders are byte-identical and `--parity`
  is unchanged. `verity.proofs.target.NativePeak` is gone; a target's `native_peak` is a `verity.census.Hardware`.

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
