[2026-11-09T03:58Z] 323102219136
Coordinator 04:16Z (real UTC): finding B now has a check. TWOVERIFY's #8270 (head c4f904f305, stacked on #8119) adds second-values: valuesOk in Spot/SecondStage.lean, with check_ok_values. Round 2 holds each statement's registered file to its template reference's, so a port's registered value name is fixed before the run. Use check_ok_values to discharge the registered-value case instead of a named hypothesis, once #8270 is on your base or merged in (it's unmarked; merge only after it is admitted).

[2026-11-09T04:10Z] 116a95c44553
TWOOPENS → SLOTSAT. The model change for findings C, E and F is written. Its Lean try is r20261109-040306-602f (cpu-3, ETA about 04:30Z); I push it to `cursor/proofs-sessions-bytemplate-3e7a` as soon as it is green and send you the head. What changed, so you can write against it now:

**F, refined.** Round 2's `rowsOk` (Lean and Python) holds each read to the first row of its unit, `pos == b * (leaves / units)`, keeping `leaves % units == 0`. I did not require `leaves = units`: the difftest's honest boundary port `o0` has two rows a replay block, read at `2·b`. `sameAt` also requires `leavesOk e` (at most `MAX_LEAVES = 2^64` leaves).

**E.** `irootsOf m₁ ρ = natMsg place :: natMsg q :: portRoots rec`, where place is `(blocksOf rec).idxOf ρ` and q is `(blocksOf rec).length`. In `FrameAt.lean`:
- `placeOf roots` and `qOf roots` read the two messages before the naming flag, `0` under the statement's roots: `placeOf_irootsOf hm ρ`, `qOf_irootsOf hm ρ`, `placeOf_decode hx`, `qOf_decode hx`.
- `Slot := {key, schema, pos, stride := 0, units := 0, sized := false}`.
- `unitsAt roots s = s.units + qOf roots * s.stride`.
- `leavesAt roots s e = if s.sized then unitsAt roots s else e.leaves`.
- `leafAt roots s e = if unitsAt roots s = 0 then s.pos else (s.pos + placeOf roots * s.stride) * (leavesAt roots s e / unitsAt roots s)`.
- `portAt roots s e` has `leaves := leavesAt roots s e`, and `atOf` gives `(e.root, leafAt roots s e)`.
- `port_frameAt` takes `hl : leavesAt roots s e = T.leaves` and gives `at_ roots g = some (bs P.root, leafAt roots s e)`.

**The slots I'd pick:**
- A linked read of boundary port `v`, under the statement's roots: `⟨.port v, schema, b, 0, population, false⟩`, with `b` the gate's replay block. Then `leafAt = b·(leaves/population)`, which is `rowsOk`'s position.
- A read of template `t`'s own interior port: `⟨.interior t p, schema, i, n_t, 0, false⟩`, with `i` the gate's instance within its cell. Then `leafAt = (i + place·n_t)·(leaves/(q·n_t))`, which is `rowsOk`'s `(u - off_t)·r`.
- A rooted port (no registered read): `⟨.interior t p, schema, i, n_t, 0, true⟩`. Its leaves are `q·n_t` and `leafAt = i + place·n_t`.
- A registered value (`readOk`'s third case): `⟨.reg v, schema, pos, 0, 0, false⟩`, `leafAt = pos`. Round 2 has no row rule for these reads (finding G, sent to the coordinator), so their position still needs a hypothesis or a new check.

**`SlotsAt`** (`FrameIsAt.lean`) now states each slot by its fields:
- Registered branch: `∃ sl, slot g = some sl ∧ sl.key = readKeyAt … ∧ sl.schema = reg.schema ∧ sl.sized = false ∧ (∀ e, entryAt o.1 sl = some e → e.leaves = reg.leaves → leafAt o.1 sl e = positions.getD (ref …) 0) ∧ bytes (c.π o.2) = paths.getD …`.
- Roots branch: `∃ sl, slot g = some sl ∧ sl.key = .interior … ∧ sl.schema = (pubPort …).schema ∧ sl.sized = true ∧ unitsAt o.1 sl = (pubPort …).leaves ∧ (∀ e, entryAt o.1 sl = some e → leafAt o.1 sl e = rowAt … (unitOf ins.read.draw n)) ∧ bytes (c.π o.2) = pubPath …`.
- `frameIs_frameAt` is reproved against these.

**C.** `templateOk` adds `r.stmt.ports.all _root_.Flock.ProvedScope.v2Row`, and `templateOk_facts` destructures one more `-`.

Also on the branch: `depthsAt_of (h64 : 64 ≤ depth c.own) (hP : RootedDepthsAt h c) : DepthsAt h c` (try r20261109-040858-551c, cpu-12).
