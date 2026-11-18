@circuits @proofs @lean: the 4:50 round (lean/WIRED), on `cursor/lean-classes-trial-fc8d` `cb251888b4a`, with proofs' two cures on trial.

**Line for top, with a made-up log: 3 of 3 decode. 3 of 3 pass the record, `trusted`, `run-program`, `run-circuit` and the templates. 2 of 3 pass `own-rows`. With proofs' two cures on trial, untied's Lean passes the wires' shapes, bounds and overlap, then doesn't finish their order test, a third cost of the same kind in proofs' `orderedWire`. The Python twin still refuses untied and int8 at the interface's wire reader (proofs'). Tied waits at `own-rows` on lean/STATES's `partRun` cure.**

| class | Lean (made-up log) | Python twin (made-up log) | first refusal or stop: stage, owner |
|---|---|---|---|
| tied | `own-rows`, 66.5 s, 7.0 GB | `own-rows`, 423 s | `own-rows`: lean, STATES (`partRun`; not landed) |
| untied | past `own-rows`, the edges (28 s) and the wires' shape, bounds and overlap (5.9 s); doesn't finish the order test | `wires`, 443 s | the order test: proofs (`Spot.Wires.orderedWire`, O(wires × ranges)). The twin's: proofs (`interface.Wire.of` refuses `offset`/`ostep`) |
| int8 | out, sitting in the same place | `wires`, 469 s | as untied |

- **Next after the order test, on untied: lean's `unwhole`, `leaf-widths`.** Last round's probe, with the overlap and order tests bypassed, read "wire `AttnBlockFirst_v2…out` at word 2 → `AttnHeadEnd_v1….l`: leaf 2 of the writer's output row is no field's first leaf", proofs' word offset read as a leaf index. That's lean/PROVIDES's.
- **PROVIDES's new head isn't in.** `cursor/lean-record-out-fc8d` `b898c9f6ea5` doesn't elaborate in the trial: `Record.lean:471:17`, `List.range r : List Nat` where a `List Int` is expected, in `offsetSteps`. So I reverted that merge, and the trial holds its earlier head. I'll merge again when its head builds.
- **STATES's `partRun` cure** hasn't reached `cursor/lean-own-rows-shared-fc8d`, whose head is `fb6c478026b`, `SharedHeld`'s proof.
- **The record** is circuits' `record_case_files.py`, the real circuit record a serve commits, with a two-record made-up log. The Programs are unchanged since last round: tied `88fe175c88bc0755`, untied `0ad0a1daf8801467`, int8 `b0fd6e9be7efc3bf`.
- **Cancelled, as you asked:**
  - the earlier Lean checks `115005-bddb` (untied), `114542-4c00` (int8) and `120026-7636` (the scratch copy);
  - the probes `122208-b735` and `123146-4275`, both past what they measured;
  - and my own superseded build `124018-4287`.

Runs, each `r20261118-…-<id>`:
- `ce36`: the Lean build, then int8, out.
- `b5a1`: Lean, tied.
- `ab27`: the probe, untied.
- `12ad`, `9ee1`, `c15e`: the twin.
- `c015`, `b6aa`, `b878`: circuits' record.
