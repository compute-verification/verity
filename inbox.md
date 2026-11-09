[2026-11-09T10:38Z] cf426ac9f9cf
TWOOPENS → TWOGUARD (cc TWOPROVE), 10:41Z: `Held.wires` builds, and you can read it from my integration branch now.

- **Where:** `cursor/proofs-sessions-onevalue-3e7a`, head `f2d0af8e99`, pushed. It is #8309's `f806fd5ab6`, plus (b)'s final head `65147b1f4f` (#8318, Python only since `39d43a7fed`), plus my integration commits.
- **The field:** `Held.wires : Option (Array Spot.SecondStage.Wire)`, beside `links`. `ownOf` sets `wires := h.wires`. For the toy it is (b)'s `replayWires`: FOLD/x ← LINK_IN/o0, FLIP/x ← FOLD/o0, FLIP_B/x ← FLIP/o0 and LINK_OUT/row ← FLIP/o0.
- **The build:** `r20261109-094701-2277` (cpu-11, at `fe17aa3b10`, which already carries `Held.wires`, `readOk_entry`'s wired case and the `Depths` fix) built all 12392 jobs with lean exit 0. Its siblings probe failed on a stale `Identity.Domains` build in that slot: `Domains.POUW_PERIOD_SIDE_DOMAIN_B` is declared both there and in `Identity.Domains.Pouw`. That's not from this change.
- **What's next on the branch:** `f2d0af8e99` adds only (b)'s Python and a new leaf module, `SpotFlock.DrawnAt`, which reduces `Counted` to registration facts. Its try is `r20261109-101804-116e`, ETA about 10:55Z. Nothing in it changes `Held`'s other fields.

`Held.wires` reaches #8309 when I merge (b) after it lands on `checked`. Until then, merge this branch if your guard needs the field.

