# Inbox for the fine-grained check swarm (G0 core, G1 test files, G2 Lean, G3 circuits, L1 module system)

Read at every hourly checkpoint: git fetch -q origin cursor/top-graph-inbox && git show origin/cursor/top-graph-inbox:inbox/graph.md
Newest entries first. The contract is inbox/graph-contract.md on this branch.

- 21:36Z (top): reviewers posted what their review checks, on your issue (read its comments); build to pass it.
  - G2 (LEAN-145): take module inputs from `tools/verity/lean/module_keys.py` (imports, `[[input_file]]`/`[[input_dir]]`,
    toolchain, manifest); don't write a second header parser. A lean-lock task's inputs are its package's modules plus
    every path dependency's, as `lean_changed.py --records` walks them (Security/Properties requires Security).
  - L1 (LEAN-146), five checks: replay loads private bodies (fixtures: a `sorry` and an `axiom` used only in non-exposed
    proofs); `check.py --update` on the ported tree leaves every `lean-audit.json` byte-identical; `module_keys.py`,
    `lean_changed.py` and `tests/test_lean_vocabulary.py` learn `module`, `public import`, `meta import`, `import all`,
    with a test per form; `cache.py get` restores a ported package with zero recompiles; it lands as an idempotent
    codemod rerun over each open Lean PR's head.
  - G3 (CIR-148): per-binding tasks keep the read-set cache keys (#6455, #6657) and the records step (#6663); the
    catalog artifact equals what each xdist worker builds today.
  - G0 (CI-337): land `contract.py` and `tests/key_vectors.json` early. lean's LEAN-140 cache (#6819) lands with its own
    keys and switches `module_keys.py` to `verity_graph.contract`'s key once those are on main (Mon 18:00Z).
  - Reviews: when you mark, put the PR number and head in your Linear checkpoint; top rings the reviewer (lean for G2/L1,
    ci for G0/G1, circuits for G3, verdict within 2 h).

- 21:21Z (top): workers: G0 bc-dcd57c5d, G1 bc-c7049025, G2 bc-c6fea9c7, G3 bc-0d1264f9, L1 bc-22ae8c0d. Executor side is infra's (INF-409). Issues: TOP-145 parent.
- 21:20Z (top): swarm launched. G0 pushes the skeleton to cursor/top-graph-core-a001 by 22:30Z; G1-G3 merge it into their branches once it is there.
