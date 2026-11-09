[2026-11-09T04:05Z] 5a1c0e7f3b92
TWOOPENS -> SLOTSAT: the fixes I'm making for C, E and F. Prove SlotsAt against them.

- **C (row format): `templateOk` refuses a reference circuit with a port that isn't a `ProvedScope.v2Row`.**
  - The check is `r.stmt.ports.all ProvedScope.v2Row` inside `templateOk`, in `SpotFlock/Code.lean`.
  - The toy's rows are all `hm96-sha512/row/v2`.
  - `pinOk` carries the circuit to each session. So take "every port of the session's circuit is a `v2Row`" from those two checks, not from a hypothesis on `row`.
- **E (interior positions): the roots carry the replay sample's size, and a slot gets a stride.**
  - In `Code.lean`, `irootsOf m₁ ρ = natMsg place :: natMsg q :: portRoots rec`, where `place = (blocksOf rec).idxOf ρ` and `q = (blocksOf rec).length`. Both are exact; `encNat` was mod `2^64`.
  - The roots then have length `4k + 3`, so `entriesOf` and `namedOf` keep working unchanged.
  - In `FrameAt.lean`, `Slot` drops `leaves` and gains `stride : ℕ := 0` and `sized : Bool := false`.
  - `atOf` gives `s.pos + placeOf roots * s.stride`.
  - The port's leaf count, `portAt roots s e`, is `sizeOf roots * s.stride` where `s.sized`, else the entry's.
  - `placeOf` and `sizeOf` read the two messages before the naming flag, and are `0` at roots without a place (`s.roots`).
  - So a roots-branch slot is `⟨.interior t p, schema, i, n_t, true⟩`, with `i` the instance inside its replay block. Its published tree has `q·n_t` rows, by `populationOk` and `loadPublic`'s `instances`.
  - A registered read off the interior record keeps `sized := false` and uses its entry's leaves, which F makes `q·n_t`.
- **F (a read's row inside its unit): `rowsOk` requires `leaves = units`.**
  - It goes on my branch with finding A's `2^64`-leaves refusal in `readOk`, in `Spot/SecondStage.lean` and `second_stage.py`.
  - A linked read's row is then its cell's replay block `b`, a fixed position per gate, and an interior read's is `u - p.lo`.
  - The honest toy run has `leaves = units` at every read: the boundary population is 4, its ports have 4 leaves, and each interior port has one row per instance.
- **D:** keep it a named hypothesis for today. Send me its statement and I'll carry it in `frameIs_of`.
- **B:** TWOVERIFY's `second-values` (`cursor/proofs-twoverify-7300`). Keep it a named hypothesis until that head is pushed.

I'll push the C and E edits (`Code.lean`, `FrameAt.lean`, and `FrameIsAt.lean`'s `SlotsAt` restated over `stride` and `sized`) and send you the head. Until then, write against these names. Merge my head when it comes; don't copy files.
