[2026-11-11T13:46Z] 8150ba0fa8e8
WIRED to @training: the page. READ at f41591c9b490 (my branch head) unless a line says RAN; AGENTS.md at origin/main
72e4d46d92. Nothing is proved and no file of lean's, FORM's or COUNTS' is touched. Sent ahead: (i) 5:07 AM, (iii)
5:17 AM, who writes the circuit file 5:31 AM, where the verdict crosses 5:39 AM PST.

RESULT. `Served F c (V.declaration θ F)` is false as stated, and nothing in Lean carries the rate rule from the
record's circuit to the circuit file's. Two ways remain. A judges the policy on the record's circuit and leaves the
circuit file out of the claim. B judges it on a wired Boolean circuit built from the circuit file, which no Lean
defines yet. As I read the three design-rule sentences, they pick the circuit file, and then the inference-only check
reading a descriptor and no circuit file is itself the gap. Which file "the circuit file" names decides what the
guarantee says: that is Daniel's.

## 1. The statement that would close `InferenceOnlySound`

    ∀ o : Exchange, V.Verdict k .inferenceOnly o →
      Nci.CompliesRate (V.gates o.params) (V.fresh o.params) (V.declaration o.params o.files)

Text only, not tried in Lean. `InferenceOnlySound k` (Game.lean 902, `Target` 890-892) follows with error 0 once
every outcome of the play at θ, F has `params = θ` and `files = F`; `finish` writes them (501-503). Whether a lemma
says so: not read.

To get it from `Complies F` (Record.lean 348-353), four ties:
- T1, the component. `component k .inferenceOnly` must be the form's check, so its acceptance is `Accepts F j tr c`.
  Today it is `NciCheck.Protocol.protocol` (Game.lean 176, 183), the baseline, which reads no wiring. MEDIUM: a
  protocol for the form and a decode lemma, as `acceptComplies` is for the baseline
  (`Proofs/Properties/Remote.lean:526`).
- T2, Γ and α. `V.gates θ = (F.inst c).gamma`, `V.fresh θ = num / den`. Free fields of `Vocab` (701, 703), read
  only at 809 (RAN `rg 'V\.gates|V\.fresh|V\.theta'`: lines 756 and 809). SMALL, in lean's file.
- T3, θ. `V.theta θ F` is the circuit's θ. A free field (698-699), read only at 756. SMALL, in lean's file.
- T4, the circuit. The false one: sections 2 to 4.

## 2. Reading A, in five lines

1. Game.lean 809 stops naming `V.declaration θ F` and says: every circuit the record of the policy's case lays out
   (`Served F c d`) complies at the instance's Γ and α. `declaration`, `theta`, `gates`, `fresh` leave the claim.
2. Then `InferenceOnlySound` is `Complies F` plus T1. No lowering enters the theorem.
3. The Program is tied to the circuit file by digests only (the seven ties of my (iii) message). The composite then
   makes two claims about two circuits, sampled proofs' on the file's and the policy's on the record's, and no
   statement says they are one computation.
4. `ExpandComplies` (proved, `Proofs/Nci/CompliesCK.lean:67`; stated `Definitions/Nci/Guarantees/CompliesCK.lean`
   32-34) gives bits: `CompliesRate (c * W) α d.expand` when computed gates are 1 to W bits wide. That is the
   canonical expansion, a w-bit gate as w one-bit gates. It is not the file's gates: a 16-bit product counts 16.
5. vllm's note `20261110T0150Z-report-v0-for-the-final-run.md` 28-31 records this as the condition already set:
   the verifier gains the form only with "`InferenceOnlySound .remote` restated over the record's circuit, @lean's
   guarantee".

## 3. Reading B, on one page

B keeps the claim on the committed Boolean circuit. It needs three things.

S1, THE DEFINITION THAT IS MISSING: a wired Boolean circuit from the circuit file and the record's wiring. Its gates
are the file's unit gates, copy by copy, less every input bit of a wired row; a bit that read such an input reads
the writer's output bit. The file's circuit today is n copies of one unit, each on its own inputs: "No unit reads
another's sources" (`Spot/FilePart.lean` 5-14; `Properties/Flock/Circuit/Keyed.lean` 33-39). `Spot/Wires.lean` 4-5
says the same of the Program: "the program holds no dataflow between units. The wires are that dataflow." The parts
exist: the row maps (`rowBit_keyRowOf`, `rowBit_wire`, `Proofs/Flock/Bridge/BitsRowMap.lean` 148, 197; the
output-row pair named in its header 28-30, lemmas not read) and the wires' order (`check_ranked`,
`Spot/WiresSound.lean:255`). LARGE, three reasons. `Verity.Circuit` asks that a gate read only earlier gates
(`reads_lt`, `Core/Circuit.lean` 46), so units are renumbered by the wires' rank. The game holds one circuit file of
one template (`FileRead`, Game.lean 552-570; `circuitTemplate`, HmRow.lean 1178-1182), and sampled proofs v2 refuses
a windowed registration (a5.py 1146-1148), where a record's wires cross templates. And `Spot.Wires.check` is called
by the inference-only record check alone (RAN `rg` over `verity/protocols`): whether sampled proofs holds a wire's
reader row to its writer's committed row I did not read.

S2, THE LOWERING RELATION: the record's circuit against S1's. lean's `Lowering P gatesOf D`
(`Properties/Nci/Count.lean` 57-70) has the right clauses for one Program: the sets cover the Boolean circuit once
(`cover`); an input gate's bits are input gates and all output bits; a computed gate has the gate table's count of
bits; at most a gate's width of them are output bits; a bit reads its own gate's bits or the output bits of its
gate's operands (`located`). What it lacks for wires: it is indexed by the Program's own gates and operands
(`P.gate g`), where a wired leaf is an input gate, so no clause lets a reader's bit read a writer's output bit.
Indexing by the record's circuit repairs that, since that circuit's reads already land on the writer (Record.lean
259). That is `CircuitLowering D B` in my file. LARGE, and not by its statement: nothing in Lean produces one. The
file carries no map from its gates to the Program's. Python's lowering folds constants, lowers a Boolean
Definition whole by its builder, and splits a template instance into identical units by connected components
(`verity_flock/ir_lower.py` 1-11). So neither relation is what it builds: a folded gate has no bit
(`CircuitLowering.computed` fails) and its count depends on its operands, not on its primitive (`located`'s table
fails).

S3, THE TRANSPORT. Stated, not proved: `LoweredComplies`, `security/Proofs/Properties/Nci/RecordLowered.lean`. RAN
`lean_changed.py --try FILE --siblings` on a CPU node, run r20261111-132310-5b44, first try: lean exit 0 in 17 s,
siblings clean over 2,942 modules, "no theorems in it".

    ∀ c κ α d b (lw : CircuitLowering d.D b.D), 0 ≤ α → (∀ x, x ∈ b.θ ↔ lw.word x ∈ d.θ) →
      (∀ g, (univ.filter fun x => lw.word x = g).card ≤ κ) →
      Nci.CompliesRate c α d → Nci.CompliesRate (κ * c) α b

On paper, mine, checked by nobody: a block's bits are the bits of its gates. Γ becomes κ·Γ, κ the most bits a gate
has; the exact figure is the block's own sum, the gate table's count. Fresh bits move as ≤: an output bit read from
outside belongs to a gate read from outside, and a gate has at most its width of them. α is kept, since a block has
at least as many bits as gates, and is then slack by that ratio. `ExpandComplies` is the case where a gate's bits
are its width. SMALL: that proof is 20 lines over `ExpandSize`.

WHAT `DescCountsCK` COVERS: S3 only, for the count rule, on the Program's own circuit. It says a published
`complies` gives `CompliesAtMost c k n` on every Boolean declaration that lowers the Program by the gate table
(Count.lean 75-83). It takes the lowering as a hypothesis, so no S2. It knows no wiring, so no S1. The lock lists it
"unstated ... not yet proved" (`Properties/lean-audit.json` 11279). `Lowering` has no other reader (RAN `rg`: Count.lean
76 and `Proofs/Properties/Nci/Count.lean:24`).

## 4. The design rules against each way

The three sentences (AGENTS.md, origin/main):
- 236-237: soundness is "one theorem over everything the compiled Lean verifier accepts, for arbitrary Boolean
  circuits, with no hypothesis on the circuit file or the statement's form".
- 167-170: "That a circuit computes what the model computes is completeness, never proved. ... every guarantee is
  about the registered circuit ... Nobody proves it, lists it as an assumption or counts it as a gap."
- 170-171: "Lean holds one description of the circuit, the parsed circuit file the verifier checks, and defines
  every other layer's circuit from it."

THE PREMISE ("the named circuit file is the lowering of the named Program"): not allowed. It is a hypothesis on the
circuit file, which the first sentence excludes. The second does not rescue it: that sentence is about the circuit
against the model, and this tie is between two descriptions of the registered circuit. A wrong tie does not fail an
honest run. It makes an accepting verdict say the rate rule of a circuit the proofs are not about.

THE VERIFIER DERIVING THE CIRCUIT FROM THE DESCRIPTOR: allowed by the first sentence once the lowering's own theorem
is proved, which is S2 as a function: large. It does not fit the third as written, since the descriptor becomes the
description and the file a derived layer. It is what `spot/verifier/one_stage/proofs.py` 9-11 says is coming.

A THIRD WAY fits all three, and it is what I read them to pick: THE CHECK READS THE PARSED CIRCUIT FILE. No premise
and no lowering theorem: whether the file is the Program's lowering becomes the completeness the second sentence
leaves unproved, since a wrong file is judged for what it is. So yes, under these rules the inference-only check
reading a descriptor and no circuit file is itself the gap. It would READ:
- each template's parsed circuit file, once: the unit's computed gates as a count, its input columns with each
  one's row and bit, its output bits. O(one unit), which `Spot.FilePart` reads already (FilePart.lean 13-14);
- each public file's header: the units' count and order, as `FilePart.check` does;
- the record: the wires between units' rows, the held rows with their roots (θ), the division into blocks;
- the registration log, for θ, as today.
A block's gates are its units' counts summed. Its fresh bits are its units' input bits neither held nor written by a
unit of the same block. S1 is still needed, to say which circuit that judges. S2 is not.

THE OTHER READING, which I cannot rule out: "circuit file" means the Program's descriptor. ARCHITECTURE.md 113-117
says "A program is a compressed circuit" and calls Flock's files expanded circuits; 128 says the registered circuit
is the developer's "registered programs on its committed inputs". Under it A is right, the check reads the right
file, and what is out of line is Game.lean's `declaration` and sampled proofs proving a file no Lean defines from the
Program. The tree's Lean leans to the first reading: Game.lean 738-739 and `OneStageZk.lean` 5 call Flock's file
"the circuit file", core's circuit is defined from it (`circuitOfFile`, `Properties/Flock/Circuit/CircuitOf.lean:165`),
and only that file is an arbitrary Boolean circuit.

## 5. The split by `Served`'s clauses (Record.lean 292-305)

- The clauses that do not read `d` (decode, the record's parse, wiring, cuts, wires, slot, held rows,
  `Spot.Wires.check`, `TemplateQuery.evaluate`, `Templates`, `OwnRows`, `Aligned`, `Filled`, no held row has a
  writer): they hold by the record and the Program alone, and `LaidOut F` gives them. No bridge lemma. NONE.
- `LaysOut` 256, a gate of `d` for each unwired Program gate: false on the file. Under B it is
  `CircuitLowering.word`, the map the other way. LARGE (S2).
- `LaysOut` 257, widths: false on the file, every gate one bit (`Properties/Flock/Circuit/CircuitOf.lean:56`). Under
  B it is `bit` and `width`. With S2.
- `LaysOut` 258, input gates: holds bit by bit. Nearest: `rowBit_keyRowOf`, `rowBit_wire`. MEDIUM.
- `LaysOut` 259, reads: false on the file for a wired read. Needs S1. LARGE.
- `Names`, θ: a definition to make (T3), the held rows' bits through the row map. MEDIUM.
- `Divides`, 𝓘: false at `.ofCircuit`, "each computed gate its own block" (`Core/RegisteredCircuit.lean` 92-95).
  `CompliesRate` does not read 𝓘, so the clause can go. SMALL.
- `DescAgree.*` is about the Program's own circuit: it reaches the baseline's `LaysOut` (`Properties/Nci.lean`
  201-203), not the record's or the file's. `CircuitOf.lean` relates the file's circuit to no Program. RAN `rg` for
  `Flock.Program`, `Program.resolve`, `Program.decode`, `Program.gate` under security's three Flock areas: 0, 0 and
  1 file (`Proofs/Flock/Claims/Recursion/SetupCount.lean`).

## 6. The wired leaves

The file keeps them, as input gates (`Keyed.lean` 44, 59-62). What is true is a quotient, not a containment either
way: glue each wired input bit to its writer's output bit, drop the two constant gates, group bits into words, and
you have the record's circuit. Smallest example: two units of a template with a 1-bit input x and an output y, wired
u1.x from u0.y. The record's circuit has 3 gates: u0.x, u0.y, and u1.y reading u0.y. The file's has the constant,
the zero, u0.x, u1.x and each unit's rows, with u1's rows reading u1.x, an input.

## 7. Who has part of it

- lean, as an open point with no owner or date: `20261110T1630Z-report-lean238-property.md` 305, under "Not
  settled": "No statement gives that the composite's declaration (`V.declaration`, the committed files' circuit) is
  a circuit the record lays out".
- The lock: `Verity.Vocab.InferenceOnlySound` is open, @lean, LEAN-186, due 2026-11-30 (`Properties/lean-audit.json`
  763-768, 9300), a premise of `Verity.remote_soundness_of_subprotocols` (882).
- The tree, for the baseline: `Properties/Remote.lean` 52-54. The registered circuit "is the program the policy's
  parameters carry"; that it is the compiler's output explaining the committed trace "is not stated".
- lean, proved: `acceptComplies` (`Proofs/Properties/Remote.lean:526`): acceptance gives `CompliesRate` on every
  circuit the Program lays out. `PolicyMet` (Remote.lean 77-81): its expansion at `(ΓW, ⌊αΓW⌋)`.
- lean's `20261109T1405Z-report-coordinator-mon.md` 186: the policy on a served stretch's record is lean's,
  statement and proof; "the count rule needs `DescCountsCK` first, unproved and undated".
- vllm's `20261110T0732Z-draft-ask-count-or-rate.md` 164: accept is proved "for the description the check reads on
  `main`, not yet for the committed circuit of a served run".
- (ii), firm. Lean ties a Program to its circuit file by name, digest and count only: `HmRow.checkPartition`,
  `HmRow.deriveTemplateUnits`, `ProvedScope.programWhyP`, `Spot.FilePart.check`, `Flock.Registered`'s cover.
  `Definitions/Flock/Lowering.lean` (header 8-30) is the file's rows against the statement, not a Program against
  the file. Python lowers: `verity_flock.ir_lower.lower` (326), `lower_gates` (209), `units` (510);
  `verity_flock.circuit.compose` (circuit.py 682) writes the file.
- (iii), as corrected and then extended: the Program and the file's digest are both the verifier's params; the
  verifier's code that composes the file runs on the served node in every launch the tree has; on `auditor-host`
  sampled proofs' row is read from `summary.json` and `window/audit.json` as that node wrote them
  (`served_integrity.py` 65, 73-81). The period's verdict there is never `accept` (`block_record.py` 180).
- Onsite: the same component and the same tie (Game.lean 176; Composition.lean 86-89). `CoreOnsite.lean` not read.

## 8. Not read

`NciCheck/Record.lean` past its header; `Flock/Program.lean` past line 358; the Bridge's lemmas beyond two
statements; `Proofs/Core/Expand.lean`; `ir_lower.py` past line 12; `benchmarks/prove_path/run.sh` and
`records_audit.py`; `composite.py`; `CoreOnsite.lean`; PRF-487 and LEAN-238 themselves (only notes citing them); a
size class's circuit beyond HmRow.lean 1184-1187, where the file is a class's circuit and not one Definition's;
whether sampled proofs holds a wire to committed values.

My branch `cursor/training-record-wired-05f2` at f41591c9b490, pushed: one new file, two statements, no theorem, 0
retired-terms sites (RAN `tests/test_retired_terms.offenders` on it).

