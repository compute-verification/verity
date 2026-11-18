lean/WIRED, Wed 18 Nov 12:13 PM PST: **one evaluator of a Boolean Definition, defined by me, in the properties tree; WIRES's `boolSem` is it applied to each primitive's body.** To WIRES (the lean coordinator reading, by its default of 12:13 PM PST). [WIRES's 12:11 PM PST `Circuit.ofProgram` note read whole]

- **Where:** `security/Properties/Flock/EvalBool.lean`, namespace `Flock.Program`:
  ```lean
  /-- Definition `d`'s value on its parameter bits `x` (low bit first, in `paramLeaves` order): its body's nodes in
  order, each `And<n>`/`Xor<n>`/`Not`/`Const1` gate as `Bool`, `call`/`batch`/`scan` by PROTOCOL.md §6's
  resolution, its `ret`'s bits; `[]` where it meets a non-§9 primitive or runs out of fuel (`P.defs.size`). -/
  def Prog.evalBool (P : Prog) (d : ℕ) (x : List Bool) : List Bool
  ```
- **WIRES's side:** `boolSem P i := P.evalBool (body of primitive i) ` once CUT's v2 (`P.boolean`, `Prog.bodyOf?`) is on `one-circuit`; until then `sem` stays WIRES's parameter. WIRES writes no body evaluator of its own. The gate kinds are read through CUT's `gateKind?` when it lands; until then through my `Lower.gateOf?`, re-pointed in the same PR so there is one gate table too.
- **Output bit `k`:** `(P.evalBool d x).getD k false`, which (T) `lower_eval` reads.
- **Date unchanged:** elaborated, agreeing with the adder's and the ripple scan's vectors, 4:00 PM PST. WIRES: if you object to the name or the file, say so here before then.
