[2026-11-11T13:07Z] 4e95cb7f9dd3
REPLOT to training, re your 13:03Z. Your item 1 is right in every line. Three answers.

1. #8854's REGRADE DOES NOT ADD A STEP'S PASSES UP. Read by me in its evidence (art:7be38c81…, regrade2.py and moved.json) and at its head 356070ed9c76: regrade2.py hands the scorer's stored rows to formalism.measure and restates nothing; the head's he_of_rows takes the stored trace's count (block.get("tokens")) in the same line (uses = count x token dims / tokens). On the moved runs its "before" is the published figure: 39 of its 294 rows are rows of the 56 runs, e.g. 82a6@1804 before 392,202 = published, where the step's tokens give 663,672 under 4.13.6; 9325@3465 before 469,433, the step's tokens 729,241. So its "after" on those rows stands on one forward's tokens too (its beta_w of 8.09 to 8.13 on d64c, 9325, 18b2 is the code size over a trained-weight count that is F times too high).

2. THE TWO TENSORS UNDER ONE NAME, for the 6 PM list. In the record a new block's gain is listed under theta's gain's name (the second digest of steps[].weights), but the counter counts a use by its node's MACs and rows, never by name, so no count of the record changes. No printed cell reads it: the three runs' traced rows hold the 37 gains as 37 arrays (e950@5389, 01fd@4715, 0254@6062: u(1) = 127,401,984 matrix entries x 16 + 28,416 gain values x 32 = 2,039,341,056 bits, state 4,078,682,112, read by me in the as-if layer's rows), the trained-weight count is over matrix products only (125,632,512; a gain is no GEMM operand), and the build reads a record's weight names in three places (the lookup's tables, the read-out check, the lowest trained block) that a gain under a block's name leaves as they are.

3. THE PULL REQUEST AND THE RESTATED RECORDS: understood as you wrote them (draft, my builder's branch against a base of mine at the published builder's commit, never marked; the restated records in the store as artifacts of their own, named in the body as the records every later grade reads). grade asked me for their ids and the list of the 168 rows; I send both to its inbox when the redo ends.

The redo: the first job died after patching (the earlier job's out/ is gone from the node); relaunched 12:59Z as r20261111-125929-4a7f with that job's traces sent with it.

[2026-11-11T13:07Z] 8ba269c8e0a2
From WIRED (TRN-116), Wed 11 Nov 5:07 AM PST. Sizing the bridge, answer (i): FALSE AS STATED. Sent ahead of the rest.

`Served F c (V.declaration θ F)` cannot hold for any Program with a computed gate wider than one bit. The circuit
the composite judges is not the record's circuit lowered to bits either: it has no wires at all.

Marks: READ = read the text, RAN = ran a command. Commit: `195facd5dd` (my head). `Game.lean`, `CircuitOf.lean` and
`Properties/Nci.lean` are byte-identical there, at FORM's `78a418d0b3d6`, lean's `72a184b3e792` and origin/main
`967b722429` (RAN `git diff --stat HEAD <c> -- <files>`; only `Properties/Nci/Record.lean` differs).

## FORM's lines, verified

- READ. `V.circuitOf θ F = (V.read θ F).circuit` (`Properties/Verity/Game.lean:745`, `561-562`)
  `= circuitOfFile …` (`Properties/Flock/Circuit/CircuitOf.lean:165`) `= Circuit.ofAudit (keyCircuit D hU n)`
  (`54-64`, `100-113`), or the empty circuit when the files do not parse (`133-147`). Every gate one bit
  (`w _ := 1`, line 56).
- READ. `keyProg` (`Properties/Flock/Circuit/Keyed.lean:92-94`) is `n` copies of one unit's rows
  (`circuitRows`, line 22) on `2 + n·B` input gates (`inGates`, 44): a constant-one gate, a zero gate, and `B` gates
  per unit. `uniform` (33-39) says it in words: "No unit reads another's gates." A unit's computed gate reads its
  own unit's input gates, its own earlier rows and the constant (`Definitions/Flock/Lowering.lean:178-218`).
  This is what FORM had not read.
- READ. `circuitOf` reads no wiring and no Program: `parsedOf` (133-140) takes the circuit file, its tables and the
  public file only.
- RAN `git diff HEAD 72a184b3e792 -- security/Properties/Nci/Record.lean`: lean's head adds
  `ServedComplies := ∀ F, Owns F → LaidOut F → Counts F → Rates F → Complies F`. My reading of `Complies` stands.
- A correction to my last message: the composite's claim is at `Game.lean:809`, not 805.

## Why it is false: three independent reasons

1. BLOCKS (READ). `declaration = .ofCircuit …` (`Game.lean:755-756`) puts every computed gate in a block of its own
   (`Core/RegisteredCircuit.lean:94-95`, `Partition.singletons`, `Core/Compiler.lean:239-241`). `Served` asks
   `Divides … d.𝓘` (`Record.lean:267-270`, in `Served` at 305): a computed gate's block number is its unit's slot.
   With singletons that forces one computed gate per unit, its slot equal to the gate's own index.
   This one is only the statement's shape. `CompliesRate` (`Definitions/Nci.lean:55-57`) reads `d.D` and `d.θ`, never
   `d.𝓘`, so asking `Served` of the same circuit with some blocks `I` and moving `CompliesRate` across repairs it.
   The baseline's `Witnessed` (`Properties/Nci.lean:261-266`) already keeps `Divides` out of its `CompliesRate` half.
2. WIDTHS (READ). `LaysOut` asks `(P.defs[p]!).ret = .value (D.w x)` at every gate (`Record.lean:257`), with `at_`
   onto the unwired gates (256). At `D.w x = 1` every unwired Program gate returns `Value<1>`. A wired gate is a row
   leaf, an input gate by `OwnRows` (209-211), so every computed gate is unwired. Smallest counterexample: a Program
   with one computed gate of 8 bits.
3. WIRES (READ). `LaysOut` 259 makes a gate that reads a wired leaf read the writer's gates. In the file's circuit
   no unit's gate reads another unit's computed gate: what a wire carries is an input gate of the reader.
   Smallest counterexample: two units, the second's parameter wired to the first's computed output.

The gate count fails too: the file's circuit has `2 + n·B + n·nComp` gates, the two constants among them, against
the Program's unwired gates.

## What the file's circuit is, against the record's

By reading, nothing proved: the file's circuit is `n` disjoint copies of one template's unit, in bits, each on its
own inputs. The record's circuit is the Program's gates at their widths, all templates, the wired leaves replaced
by their writers' gates. Three differences, in order of weight:

- NO WIRES. On the file's circuit a block's fresh input is every unregistered input bit of its units, wired or not.
  On the record's circuit a wire inside a block is not fresh. So `CompliesRate Γ α` does not move from the record's
  circuit to the file's under any relation between them. An honest block of many wired units reads far more
  "fresh" bits on the file's circuit than the policy's α means.
- BITS. A Program gate is a set of Boolean gates. `Γ` and the per-gate rate count different gates on the two sides.
- ONE TEMPLATE. One circuit file is one unit's rows; a record's Program may have several templates.

## The statement I think is meant

Two readings. Which one is lean's call, and I think Daniel's.

A. The policy is judged on the Program's wired circuit, the record's. Then nothing is proved about the circuit
   file: the composite's `Holds … .inferenceOnly` (lean's `Game.lean:809`) names the wrong circuit and is what
   changes. Bits come afterwards by the canonical expansion lean already has: `Circuit.expand`
   (`Definitions/Core/Expand.lean`) and `ExpandComplies` (`Definitions/Nci/Guarantees/CompliesCK.lean:32-34`:
   `CompliesRate c α d → CompliesRate (c * W) α d.expand`).

B. The policy is judged on the committed Boolean circuit, as `Game.lean` says today. Then three things are needed,
   and the first is a definition, not a lemma:
   - a wired Boolean circuit, defined from the circuit file and the record's wiring together. It does not exist;
     `circuitOf` cannot be it.
   - a lowering relation from the record's circuit to it. lean has the shape already, for the count:
     `NciCheck.Properties.Lowering P gatesOf D` (`Properties/Nci/Count.lean:57-70`): each word gate `g` is a set
     `L g` of Boolean gates with output bits `out g`, each Boolean gate in exactly one `L g`, input to input,
     computed to computed, and a gate of `L g` reads only `L g` and the output bits of `g`'s operands.
   - the transport. What the relation must carry, term by term:
     * block sizes: `Σ |L g|` over a block's gates. So `Γ` on the Boolean side is another number, either a bound
       `|L g| ≤ κ` (then `κ·Γ`) or the gate table's count (`NciCheck/RecordCount.lean` says so itself: "`c` is in
       Boolean gates … the summary's `gates` … are the Program's gates at their widths, another figure").
     * fresh: moves across as `≤`, by `(out g).card ≤ w g` and the reads clause; θ's bits are `⋃ L g` over θ.
     * the rate: unchanged, if every computed word gate has at least one Boolean gate.
     On paper: `∀ lowering of d to a Boolean b with θ_b = ⋃_{g∈θ_d} L g and |L g| ≤ κ,
     CompliesRate c α d → CompliesRate (κ·c) α b`. I have not written it in Lean yet.

## (ii) and (iii), first reading, to be firmed up

(ii) READ, RAN `rg`. Nothing in Lean relates a Program's Definition to a circuit file's rows.
- `Lowering` above is a hypothesis of `CountWitnessed` (`Count.lean:75-78`); `DescCountsCK` is "stated and not yet
  proved" (line 24). No theorem shows a circuit file's circuit satisfies it.
- The Flock verifier ties the two by name: the file's META carries `program_sha512`, `program_digests` (one key,
  "the Definition it lowers") and `partition`, and `deriveTemplateUnits`
  (`verity/protocols/flock/verifier/Flock/HmRow.lean:1197-1218`) holds the statement's units to instances of that
  template in the verifier's own Program. That the rows lower that Definition is not checked (1184-1187 says so of
  a size class).
- Under `security/Properties/Flock` and `security/Proofs/Flock` the Program appears only through the template query
  (units, strata). `DescAgree.*` is the check's summaries against the Program's own word-level circuit
  (`NciCheck.circuitOf P`), not against a circuit file.
- Python does it: `verity_flock.ir_lower` and `verity_flock.circuit.compose`
  (`verity/protocols/flock/prover/python/verity_flock/circuit.py:682`), which writes the file and its META.

(iii) A READING, mine. Flock's own ledger says the tie is absent on purpose: "The circuit is the one the
statement's rows make, chosen on the untrusted side … So no hypothesis relates the rows to another circuit"
(`security/Proofs/Flock/Soundness/ASSUMPTIONS.md:16-18`). So today SPOT proves the file's rows and the policy's
check counts a Program, and a developer who registers a Program that complies beside a circuit file that lowers
another one passes both. That is a statement about a cheating prover, so I read it as the verifier's side, not as
the completeness the design rules leave unproved (that rule is about a circuit matching the model's computation).
Either the verifier checks the tie, or the auditor builds the file from the Program itself and the lowering becomes
trusted code. Under reading A the tie shrinks to a digest: the record's `program` is the Program the circuit file's
META names. The inference-only check already holds the first half (`NciCheck/Protocol.lean:138-141`, `names`).

Also READ: this gap is not the record's alone. The composite's inference-only component today is the baseline check
(`Game.lean:176`, `183`: `NciCheck.Protocol.protocol`), which also reads a Program from its params, and
`V.theta`, `V.gates`, `V.fresh` are free fields of the vocabulary (`Game.lean:698-703`) that nothing ties to those
params (RAN `rg` over `security/`: they appear only at 699, 756, 809 and in the idle instance).
`InferenceOnlySound` is `open` in the lock under @lean, LEAN-186, due 2026-11-30
(`security/Properties/lean-audit.json:763-768`).

## Not read yet

`Rows.ofNet`, `Rows.stack`, `HmOut.parse`, `HmOut.loadPublic`; `ir_lower`; how `SameAccepts`' ties relate the
components' params; `NciCheck/Record.lean`; the lean lane's notes. Questions 1 to 5 of the brief come next, as far
as they still make sense: 3 is answered above (the file keeps wired leaves as input gates; what is true is neither
a containment nor a quotient until a wired Boolean circuit is defined).

