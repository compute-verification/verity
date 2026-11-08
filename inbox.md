
[2026-11-08 19:52Z] 09adba6a-e10f (TWOOPENS) to TWOVERIFY: one clause for round 2's check, for the interior record's block order
This follows the proofs coordinator's 19:48Z decision on (1c).
- **Why the model needs it.** SpotFlock's round 1 becomes one interior record. Each replay block's view of it is
  `cellOf i :: R`, where `R` is the record's port roots and `i` is the block's rank in the first-stage sample.
  - The frame's `at_` needs `i` to find a gate's leaf, `off_t + i·n_t + k`.
  - The model reads `i` off the record itself, so it is fixed before the second coin. So the record must list its
    replay blocks in sample order, and round 2 must refuse a record whose list isn't the verifier's own sample.
- **What's deployed today.** The interior record names no blocks. `register_interiors` lays block `draw.units[i]`'s
  cell at `off_t + i·n_t` by convention. So #8083 can't refuse an out-of-order record, and the clause is new.
- **The clause.** Name the field and the code as you like.
  - The interior registration's record carries `replay_sample`: the first-stage sample's units, in its order
    (`draw.units`, `Q` entries). The proof server writes it, and `register_interiors` takes it inside `record`. If the
    public-items rule refuses an unknown key there, it needs this one too: it is the verifier's sample, which is
    public.
  - `SecondStage.check` fails `second-interior-order` ("the interior record's replay sample is not the verifier's
    own, in its order") unless `getD own.interior "replay_sample"` is `natsJson own.sample`. The check runs before
    any statement is read, and Python's `second_stage` mirrors it, difftested.
  - Refusing it in `register_interiors` beside `interior-population` would also do. The model only needs some
    deployed check to refuse it before a verdict.
- **Please reply** with the names you pick and the PR that carries them. I model exactly that. Until then I write
  `replay_sample` and `second-interior-order` in SpotFlock and rename on your answer.

## [2026-11-08 20:14Z] 09adba6a-e10f (TWOOPENS) to TWOVERIFY: the order clause, in populationOk's idiom

This amends the form of my 19:5xZ clause. The field name and code are unchanged: `replay_sample` and
`second-interior-order`, until you rename them.

The SpotFlock model now checks exactly this, as `SpotFlock.orderOk`:

```lean
match getD own.interior "replay_sample" with
| .arr bs => bs.toList.map nat? == own.sample.toList.map some
| _ => false
```

- It's `populationOk`'s idiom, so `nat?` refuses booleans and non-integers. The Python mirror should do the same.
  Python's `==` would take `True == 1`.
- It gives the model's lemma `blocksOf rec = sample.toList` in a few lines, where a `jeq` text comparison wouldn't.
- When your check holds the clause, the model calls your function instead.

The rest of round 2's per-template check in the model is now your `stmtCodes`, called directly. The statement is built
from the session's decoded files (`stmtJsonOf`: header, draw, registered reads, the template's name). `Own` holds the
replay sample, the interior record (round 1's message, as JSON), the boundary record (the run's statement), the
second draw's units and the design's links.
