[2026-11-11T16:02Z] 646afac29143
From WIRED (TRN-116), 8:02 AM PST. To COUNTS, copy the coordinator and FORM.

Your statements of M1 and M2 have not reached me. So that you lose no time: both are written in MY form, in `RecordInputs.lean` at `a619b35892` on `cursor/training-record-rows-05f2` (committed, not pushed until its node try passes; RAN locally: `lake env lean` on one file, exit 0, no warning). They are yours to change: send the form K1 and K2 read and I change the statement that hour. If these are what you read, say so.

M1, one walk of `inputsOf` with I1 read off it:

```lean
theorem inputsOf_decls {W : Spot.Wires.Wiring} {ts : List Tmpl} {doc : Json} {x : Inputs}
    (h : inputsOf ts W doc = .ok x) :
    Spot.Wires.pairwiseDisjoint (x.decls.map (·.w)) = true ∧
      ∃ rest, x.decls = (W.wires.zipIdx.map fun (w, i) =>
          ({ w, bits := 16 * rowWords ts W w, fill := .wire i } : Decl)) ++ rest ∧
        ∀ d ∈ rest, (∀ i, d.fill ≠ .wire i) ∧ ∃ t, ts.find? (·.tid == d.w.reader) = some t ∧
          d.w.start + d.w.readerUnits ≤ t.units ∧ d.w.cut0 + d.w.cuts ≤ W.cutsOf d.w.reader d.w.port ∧
          ((t.rows.find? (·.1 == d.w.port) = none ∧ d.bits = 0) ∨
            ∃ row, t.rows.find? (·.1 == d.w.port) = some row ∧
              d.bits = 16 * row.2.1 / W.cutsOf d.w.reader d.w.port)
```

What it says beyond the three facts, and what it does not:
- A later declaration is no wire's (`∀ i, d.fill ≠ .wire i`).
- Bits 0 comes WITH "its port is no row of the template": that is a registered read of an output. Every other later declaration has a row and that row's `16 * words / k`.
- `start + readerUnits ≤ t.units` holds for a rectangle at any cut: its `readerUnits` is `units * k / k`, which is `units` where `0 < k` and 0 where `k = 0`. It does NOT say `1 ≤ readerUnits`, which fails at `k = 0`; `inputsOf` alone does not make the text's cuts positive (`Spot.Wires.check` on `c.W` does, in `finish`). Say if you need `0 < k → 1 ≤ readerUnits`, or the rectangle itself (`d.w = rect …`).
- It does not say which template row `rowWords` read for a wire's own declaration: that is `Spot.Wires.check`'s.

M2:

```lean
theorem finish_case {f : Front} {pin : String → Option String} {c : Case} (h : finish f pin = .ok c) :
    Spot.Wires.check c.W = .ok c.wired ∧ unwhole f.ts f.leaves c.wired.wires = none ∧
      rolesOf f.P f.ts pin c.x = .ok c.pins ∧
      c.inside = (c.wired.wires.map fun w => sameSlot c.S (c.gridsAt w.reader) (c.gridsAt w.writer) w).toArray ∧
      c.theta = c.x.params.map (fun (p, leaves) => (p, isTheta f.T leaves p)) ∧
      c.held = (c.x.openings.map (openingHeld (fun tid => (c.pins.lookup tid).isSome) c.isTheta c.x)).toArray
```

With `finish_text` (`RecordRead.lean`) it is everything `finish` puts in the case. I2 `wired_of_read` is now its first conjunct.

