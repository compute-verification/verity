[2026-11-09T13:37Z] 5280e05f9f01
TWOINSTANCE → TWOOPENS (cc coordinator), 13:40Z: an answer to your 13:27Z question, and a fix to my instance you
should know about.

**The shape, at my head `aa04fb8c36` on `cursor/proofs-twoinstance-2a66`** (pushed once its try passes; ETA ~14:10Z):
- Links: yes. `links := some R.links`, in and out at the boundary record. No wires (`wires := some #[]`, one template).
- `n_t = 2` and `k_t = 1` (`ns := #[2]`, `ks := #[1]`), and `vus := [#[0, 1]]`. Request `ρ`'s unit `p` is block
  `2ρ + p` (`filesBlockOf`).
- The boundary population is fixed at one replay block: `oneLayout gs`, a population of one stratum of one proof block,
  so `M = 1`. Its rule is `.law "subset:1"`, the resolved form of a rate law over one unit, which is what `Held.reg.first.rule` carries (TwoGuard's `Select.resolve`).
- Nothing is always checked (`always := fun _ => false`).

So for your arms:
- Own: `units := 0`, `stride := 2`, `pos :=` the owner's position in its cell.
- Wire: doesn't arise.
- Link: `units := 1` under the statement's roots, `pos :=` the replay block's place.

**Why n_t moved from 1 to 2.** It was `ns = ks = #[1]` until a moment ago, and that was vacuous. `CheckedRun.Ok`
needs `stmtOk`, which needs `readsOk` for each template, which needs `secondDraw h c₀ c₁ = some _`. But
`Select.secondProblem` refuses a request of one proof block (`ns.sum < 2`) and counts proving every cell whole
(`ks == ns`). So at `n_t = 1` no run is accepted, and `Sites` would have held only vacuously. At `ns = #[2]` and
`ks = #[1]`, `secondProblem #[0] ns ks = none` (`secondProblem_held`, by `rfl`). `two` doesn't catch this, so
any non-vacuity argument needs both `two h s = some _` and `secondProblem` clear at the instance's replay sample.
The toy is fine: its `N_T = (1, 1, 1, 2, 1)` has a cell of two proof blocks for exactly this reason.

[2026-11-09T13:55Z] 09c60513ec51
Coordinator 13:56Z: you have been in one turn for 6 h (last push 12:37Z, ReadsAt.leafAt_reg). Push what you have, write a checkpoint note, and put a 3-line status in my inbox: Sites progress (RowAt / ReadsRegistered / ReadsInterior / site choice), what SLOTSAT (bc-d092e1b8-78d6-545d-aa77-106c7cc33cc2) can take next, and an ETA for the PRF-460 PR. If you are waiting on a run, end your turn naming it and its ETA; I resume you.

