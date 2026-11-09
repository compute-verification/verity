[2026-11-09T16:59Z] 2f9e9f43bffc
TWOINSTANCE → TWOOPENS, 17:00Z. Two things on ReadsAt's arm interface, and what the instance now has (`cursor/proofs-twoinstance-2a66`, about to push).

**1. `harm`'s premise can't be discharged at any registration.** `harm_own`, `harm_wire`, `harm_link` and `leafAt_reg` take `harm : ∀ st rec draw e, decode … → jsonMsg … → secondDraw … → ReadFields e reg → armOf … e = some …`.
- `armOf` branches on `getD e "value"`, and `ReadFields` doesn't pin it. An `e` with `"value": null` satisfies `ReadFields` but gets `armOf … = none`, so the premise is false for every `h`.
- Fix: add `Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (byTemplate rec) (r.drawnPart h) reg.name e = true` to `harm`. `leafAt_reg` already has it from `regEntry` (`hrd`), so passing it through is one more argument.
- With it, each arm is derivable from acceptance:
  - link: `readOk` with `linkOf = some l` forces a `.str` value and a population `M`;
  - own: `linkOf = none` and `wireOf = none`, plus the interior having the port;
  - wire: likewise.

**2. Link arm: `harm_link` needs `s.units = M` and `M` is per-run.** As I sent earlier, `M` is the run's boundary record's population, and a `Site` is chosen before the run.
- At a zero row this doesn't matter: `s.pos + placeOf·s.stride = 0` and round 2's `unit j = some 0` make both leaves 0, whatever `units`.
- So a variant of `leafAt_reg` whose `harm` gives `s.pos + placeOf·s.stride = 0 ∧ b = 0` in place of `unitsAt = units` would cover a sample of one replay block.
- In general it still needs a population check in round 2, or an `M` fixed by the registration.

**At the instance (InstanceSites.lean), proved by going through `regEntry` directly, not `harm`:**
- `sample_ok`: sample = `#[0]`.
- `rho_ok`: ρ = 0.
- `drawnPart_ok`: part = `⟨R.name, 0, 2⟩`.
- `wireOf_held`: no wires.
- `armOf_linked` (general): a linked read `readOk` accepts has arm `(M, sample[span u]?)` for some `M`.
- `leafAt_link`: at the instance, a linked registered read's slot with units 0, stride 0 and pos 0 opens exactly `positions.getD (ref …) 0`, `ReadsRegistered`'s last conjunct, with no hypothesis on `M`. (try running)
- Also `CheckedRun.count_le_one` (OneTable.lean, general): `r.Ok h → r.ins.read.I.count ≤ 1`. It discharges every `hc1`.

Next on my side is the own arm at the instance, the same way.

