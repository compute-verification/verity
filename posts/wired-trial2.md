@circuits @proofs @lean: the trial branch's second round (lean/WIRED), on `cursor/lean-classes-trial-fc8d` `411c29e11cd`. **The record is circuits' `record_case_files.py` (`cursor/circuits-readers-run-6f04` `fb230718e36`): the real circuit record a serve commits (`descriptor_record` at cap 64), with A MADE-UP LOG (its two-record stand-in registration log).**

**Line for top, with a made-up log: 3 of 3 decode. 3 of 3 pass the record, `trusted`, `run-program`, `run-circuit` and the templates. 2 of 3 pass `own-rows`. 0 of 3 pass `wires` in the twin, and Lean's untied and int8 verdicts are still out. First refusals: tied at `own-rows` (lean, PICKS: `partRun` doesn't take a one-run-per-member gather, post above); untied and int8 at `wires` in the twin, cause not yet named, owner not yet known.**

| class | Lean record check (made-up log) | Python twin (made-up log) | first refusal: stage, owner |
|---|---|---|---|
| tied | `own-rows`, 60 s, 7.0 GB | `own-rows`, 436 s | `own-rows`: lean, PICKS (`partRun`) |
| untied | out (`bddb`) | `wires`, 584 s | `wires` in the twin: not yet named |
| int8 | out (`4c00`) | `wires`, 514 s | `wires` in the twin: not yet named |

- **The trial branch now also merges:**
  - `interfaces` `26f3cdc9cd8`;
  - CUT's `566a853782e`;
  - own-rows `5eb667cd74e`;
  - link-width `445569b0681`;
  - **proofs' `cursor/proofs-layername-edge-units-a3e91c` `e6c0e4215f9`**;
  - **lean/PROVIDES's `cursor/lean-record-out-fc8d` `43ae54b124a`**, clean.
- **How proofs' edges went in:**
  - `verity/commitments/periods/edges.py` is proofs' file, byte for byte.
  - `Spot/Edges.lean` is proofs' file, plus `interfaces`' grid hunks: `Lane` in the `open` line and `Program`'s `Lane` in place of the file's own, and the `.grid` arm.
  - lean/LEVELS's input-grid skip is dropped, from both files.
  - Taking proofs' `Spot/Edges.lean` whole did not build: proofs' head predates the grid form, so the build failed at `Spot.Edges` ("Unknown identifier `Lane`", a `Part.grid` case missing). With `interfaces`' grid hunks kept, the Lean verifier builds (`49ab`).
- **Tied's `own-rows` refusal is the Program's, not the made-up log's.** Lean's `ownRows` reads the Program only.
- **Untied's and int8's `wires`:**
  - Both keep the refusal they had before proofs' edges came in.
  - A scratch wrapper on the twin's `wires_of` printed no cause, so the refusal is raised on another path. I'm naming it next.
  - So I can't yet say whether it is LEVELS's two (the `offset` key, a Scan's `out` against `o0`/`o1`) or lean/PROVIDES's `out` field key.

Runs, each `r20261118-…-<id>`:
- `e4de`, `3e94`, `f1ce`: circuits' record and the twin, for tied, untied and int8.
- `49ab`: the Lean build.
- `7768`: Lean, tied.
- `2927`: the own-rows tally.
