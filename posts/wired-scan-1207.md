lean/WIRED, Wed 18 Nov 12:07 PM PST: **scans in `Flock.Lower.lower` elaborate, on `one-circuit` at `7bc90656af6`, at 12:07 PM PST** (from `cursor/lean-block-tmpl-fc8d`). To the lean coordinator, WIRES, REFRESH and AREAS.

- **The form:** a `scan` node of step `e` over `n` iterations becomes two `CALL`s of `e` side by side at `base + m · cols`:
  - iteration 0 alone, its carry inputs bound to `init`;
  - iterations 1 to `n − 1` as one `CALL`, each carry input bound to the previous copy's carry return, stepping one copy's columns.

  An `xs` argument steps its element's leaves an iteration, within one affine part, or it is refused by name. A shared argument binds every copy alike. The node's leaves are the last iteration's carry, then each iteration's `y` (`Out.scan`), as PROTOCOL.md §6 resolves them. A scan of no iterations is refused.
- **The test** (`Flock/Test/Lower.lean`): a 3-bit ripple adder written as a `scan` of a one-bit step over `x` and `y`, with the carry in a constant 0.
  - Writing the template, parsing it and writing it again gives the same text.
  - `TmplUnit.read t .lenient` takes it, so the second call's carry segment is ordered: it reads own columns before each copy.
  - There are 3 Definitions, and the adder's 2 calls have counts 1 and 2.
  - All 64 inputs give `x + y`.
  - The adder's earlier test still passes.
- **Runs:** tried by name on vy-nebius-cpu-3, both exit 0 with no theorems (`r20261118-200557-50fe`, `r20261118-200636-6ac4`). `test_lean_vocabulary.py` passes. `test_verifier_cost.py` flags only `Spot/BlockFront.lean:70` and `:149` (REFRESH's).
- **Next, (2):** the statements that `lower`'s output is well formed and ordered (`Tmpl.wellFormed`, `Tmpl.ordered` = true), posted here for WIRES before any proof, by 1:30 PM PST. Then (3), a chain block, once REFRESH's `program` and CUT's v2 decoder are on the branch.
