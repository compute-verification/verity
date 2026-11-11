[2026-11-11T14:25Z] 19cf52d0efb5
WIRED to the coordinator, Wed 11 Nov 6:25 AM PST. The small job: BOTH clauses proved. Nothing lands; no pull request.

COMMIT `bb3494155578` on `cursor/training-record-wired-05f2` (pushed), one new file,
`security/Proofs/Properties/Nci/RecordDeclared.lean`.
RUN `r20261111-142027-e3d1` (RAN, vy-nebius-cpu-6, tree `bb3494155578`,
`python3 tools/verity/lean/lean_changed.py --try security/Proofs/Properties/Nci/RecordDeclared.lean --siblings`):
"lean exit 0 in 96 s"; "axiom probe: none of its 10 theorems uses an axiom beyond propext, Classical.choice and
Quot.sound"; siblings clean (799 imported, 3077 unchecked).

THE TWO STATEMENTS (namespace `NciCheck.Properties.Record.SecurityProofs`)

1. The width clause.

   theorem not_served_declaration_of_wide {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]
       (V : Verity.Vocab d α) (θ : ByteArray) (Fs : Array ByteArray) {F : Form} {c : F.Case} {P : Prog}
       (hP : Flock.Program.decode (F.inst c).program.toUTF8 = .ok P) {g p w : ℕ} {ops : Array ℕ} {sc : Scope}
       (hgate : P.gate g = .ok (p, ops, sc)) (hcomp : Flock.Program.isInputId (P.defs[p]!).key = false)
       (hret : (P.defs[p]!).ret = .value w) (hw : w ≠ 1) : ¬ Served F c (V.declaration θ Fs)

   In words: a record whose Program has one computed gate wider than one bit is not `Served` at the composite's
   declaration. It rests on two lemmas in the file:
   `declaration_w : ∀ x, (V.declaration θ Fs).D.w x = 1` (the parsed circuit file's circuit is `.ofAudit`, `w _ := 1`,
   or the empty circuit), and
   `not_laysOut_of_wide : (∀ x : Fin D.N, D.w x = 1) → g < gates P → ¬ Wired P rs W S g →
   P.gate g = .ok (p, ops, sc) → (P.defs[p]!).ret = .value w → w ≠ 1 → ∀ at_, ¬ LaysOut P rs W S D at_`
   (any circuit of one-bit gates, any unwired gate, input or computed). A computed gate is unwired by `Served`'s own
   `OwnRows` clause, so the theorem needs no hypothesis on the wires.

2. The division.

   theorem not_served_declaration_of_one_slot {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]
       (V : Verity.Vocab d α) (θ : ByteArray) (Fs : Array ByteArray) {F : Form} {c : F.Case} {s₀ : ℕ}
       (hone : ∀ tid u s, F.slot c tid u = some s → s = s₀) {x y : Fin (V.circuitOf θ Fs).N}
       (hx : (V.circuitOf θ Fs).isInput x = false) (hy : (V.circuitOf θ Fs).isInput y = false) (hxy : x ≠ y) :
       ¬ Served F c (V.declaration θ Fs)

   In words: a record whose units all have one slot (one inference block) is not `Served` at the declaration once
   the committed files' circuit has two computed gates. The same for any `.ofCircuit C θ hθ`
   (`not_served_ofCircuit_of_one_slot`). The lemma that does the work:
   `divides_singletons_slot : Divides P rs slot at_ (Verity.Partition.singletons C) → C.isInput x = false →
   ∃ tid u sc, UnitAt P rs tid u sc ∧ sc.off ≤ at_ x ∧ at_ x < sc.off + (P.defs[sc.d]!).gates ∧
   slot tid u = some x.val`: at `.ofCircuit` a computed gate's unit must have the gate's own index as its slot.

WHAT IS NOT IN IT
- No Program written out. The first theorem takes the Program's wide gate as hypotheses (`hgate`, `hcomp`, `hret`,
  `hw`); a literal Program would need `P.gate` evaluated on it (`Steps.run` at fuel 2^62, `gateStep`'s search,
  `operands`' loops), which I have not tried. I try it next as a separate step and send it only if it is short.
- No accepting record: neither theorem says the check accepts such a record. That served Programs have gates wider
  than one bit is READ (the page, section 5), not proved here.
- The division's hypothesis is one slot for all units. Two computed gates in ONE unit would also fail, but that
  needs "a gate lies in one unit", which I did not look for.

ALSO DONE SINCE 5:53 AM PST
- Merged main `72e4d46d9209` into the branch: merge commit `0e4e1553e98b`, clean, pushed.
- RAN `r20261111-135514-cbeb` (vy-nebius-cpu-1, tree `0e4e1553e98b`): the same try on `RecordLowered.lean` ("lean
  exit 0 in 26 s") and `RecordWired.lean` ("lean exit 0 in 304 s", "none of its 10 theorems uses an axiom beyond
  propext, Classical.choice and Quot.sound"); siblings clean for both.
- RAN `research doctor` at `0e4e1553e98b`: rc 1, 10 findings, every one `ssh:` unreachable on a host I do not use
  (vy-dc-h200, -h200-2, -h200-4, vy-dc-l40s, vy-glm53-cpu-2, -eu, -eu2, -h8, -s1, vy-nebius-2); 22 ok.
- Retired-terms sites (RAN `tests/test_retired_terms.offenders([f])` at `bb3494155578`): 0 in each of
  `RecordWired.lean`, `RecordLowered.lean`, `RecordDeclared.lean`.

