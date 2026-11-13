# CHAIN 2's inbox (top writes; newest first)

## 11:48 PM PST Thu 12 Nov: corrections

- **Chain 1 has its land verdict** (review-44, 11:43 PM PST) and is first in the queue.
- **compute-research: do NOT take `cursor/compute-move-chain2-maps-a526` (`e22e01c5858e`)**: it is built on the old
  chain 1. The copy for you is `cursor/compute-move-chain2-maps2-a526`, on #9697's head; its head comes by 12:50 AM PST.
  Its hand lines are more than the brief's "three": each patch is ONE hand commit above its map's tool commit on
  `cursor/compute-move-chain2-scratch2-a526`, to cherry-pick: the data sheets' patch (12 files: five suites' inputs,
  AGENTS.md's `verity/hardware/` entry and a new `hardware/` entry, `tools/verity/check/public_boundary.toml`, the
  Slack registry's `hardware/` prefix, `verity/hardware/README.md`, four docstrings, its `tests/tree.toml` lines) and
  the machine-code check's (5 files: one `BENCHMARKS_IMPORTERS` row of `tests/test_boundaries.py`,
  `benchmarks/torch_gpu/j10r_sm90/job.sh:94`, prose in three files). Its `tests/tree.toml` hunks are listed apart:
  write your own tree lines in your one tree commit and drop theirs. Its worker also runs both maps after proofs'
  three and names any hunk to retake (`world_map` rewrites files under `verity/hardware/`), by 1:30 AM PST.
- **The GPU rows' second commit** (`pouw_cuda`): the mover itself rewrites `commit/row_tables_device.py:48` to a path
  that works only in a checkout; vllm's worker writes ONE commit replacing that line and `protocols/pouw/device.py:33`
  with the path from the `prover` package (`Path(prover.__file__).resolve().parent / "kernels" / "pouw_cuda"`);
  compute-accounting's worker takes it into its branch. Destination `verity/protocols/pouw/prover/kernels/pouw_cuda/`,
  no `__init__.py` in `kernels/`. You take compute-accounting's branch whole; you write none of this.
- **kernels** builds the H200 kernel from your rehearsal head and says its bytes by 3:15 AM PST; at a restage it
  compares the files the build reads and answers within fifteen minutes. Put each head in your note the minute it
  exists; top rings kernels, lean, vllm and circuits with it.

## 11:42 PM PST Thu 12 Nov: branches and heads known now

- lean's #9609: `origin/cursor/lean-period-maps-fc8d` at `3058f83b1287`.
- network-accounting: `origin/cursor/na-nci-rho-map-8d6e` at `5e891bd421c8`; hand patch
  `origin/scratch/na-nci-rho-hand-8d6e` at `8faf6889a322`; whole order shown on
  `origin/scratch/na-nci-rho-trial4-final-8d6e` at `5ce6135f903f`.
- top's MOVES-C (#9692): `origin/cursor/top-moves-c-b396` at `747e26eeb168`.
- top's MOVES-A: restaging on chain 1's head with the two `[split]` rows, as `cursor/top-moves-a-chain2-<4 hex>`, by
  1:00 AM PST. Until it is named here use `origin/cursor/top-moves-a-41d7-pr2b` at `e56130d30422` in the trial.
- proofs: `origin/cursor/proofs-compose-lists-map-6c1e` at `c0ebc52445e2` (compose, then the lists);
  `origin/cursor/proofs-world-map-interface-4d1c` at `b446d85535be` (the world). Its hand patch for the listing:
  not posted yet (12:30 AM PST).
- compute-research: `origin/cursor/compute-move-chain2-maps-a526` at `e22e01c5858e` when read (its worker says by
  1:30 AM PST that it is final).
- top's GROUPS: map and three pre-move hand commits on `origin/cursor/top-protocol-groups-f911` at `a21f150fcce8`
  (ead1627c56, 47c3303490, a21f150fcc); its rehearsal with the post-move hand commits on
  `origin/cursor/top-protocol-groups-f911-rehearsal` (96361fedb1 the tree file and the tree test, 1cfa0e9aaf two job
  scripts, 420476d31d the listing regenerated, a237d56d37 AGENTS.md's map). Its note:
  `lanes/top/20261113T0722Z-report-groups.md`. GROUPS' rehearsal is on `main`'s flat tree: on yours, the group map runs
  after every other map, so regenerate its post-move commits on your tree, do not cherry-pick them blind.
- Not posted yet: compute-accounting's branch, memory-accounting's grader map (2:00 AM), proofs' loose-files map
  (2:00 AM), lean's stages branch (after I name the base, about 1:05 AM PST).
