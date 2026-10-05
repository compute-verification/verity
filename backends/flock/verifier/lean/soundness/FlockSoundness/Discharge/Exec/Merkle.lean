import FlockSoundness.ExecTemplate
import FlockSoundness.Discharge.Exec.Tables
import FlockSoundness.Protocol.Discharge.Exec.Merkle

/-!
# The leaf scheme of each table flock-verify sets up (`hms`)

`accepts_tables_hm96` refines a table whose leaves are `hm96-sha512/v1` (`hms : sts[j].merkle = MerkleScheme.hm96Sha512`).
This file proves `hms` from the statement's tags: `HmRow.parse` refuses a circuit file whose `leaf_scheme` id is not the
tags' `merkleLeaf`, and `Stmt.setupH` takes its Merkle scheme from that id (`MerkleScheme.find?`).

* `parse_leafScheme_pre`, `parse_leafScheme_tmpl`, `parseTyped_leafScheme`: the circuit that `parse` (untyped, with nets
  `pre` first, or a typed template) and `parseTyped` accept has the tags' leaf scheme. The proofs walk `parse` as
  `parse_facts_pre` and `parse_facts_tmpl` do, keeping the leaf-scheme check.
* `setupH_find`: a statement `setupH` accepts has the tags' leaf scheme, and its Merkle scheme is `find?` of it.
* `setupH_hm96`, `setupH_tagsAt_hm96`: for tags whose `merkleLeaf` is `hm96-sha512/v1` (every `hm96Rows` statement
  pinned today: `merkleLeaf_circuitEb90718f`, `merkleLeaf_circuit`, `merkleLeaf_circuit967b8d06`,
  `merkleLeaf_circuitTypes`), each table's Merkle scheme is `hm96Sha512`.
* `accepts_tables_hm96_of_leaf`: `accepts_tables_hm96` with `hms` discharged.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Exec

theorem merkleLeaf_circuitEb90718f : Tags.circuitEb90718f.merkleLeaf = "hm96-sha512/v1" := rfl
theorem merkleLeaf_circuit : Tags.circuit.merkleLeaf = "hm96-sha512/v1" := rfl
theorem merkleLeaf_circuit967b8d06 : Tags.circuit967b8d06.merkleLeaf = "hm96-sha512/v1" := rfl
theorem merkleLeaf_circuitTypes : Tags.circuitTypes.merkleLeaf = "hm96-sha512/v1" := rfl

section Accepts

open Refine Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128

/-- **flock-verify refines each table of the session it accepts, for tags that implement `hm96-sha512/v1`**:
`accepts_tables_hm96` with `hms` discharged by `setupH_tagsAt_hm96`. -/
theorem accepts_tables_hm96_of_leaf {I : Inputs} {t : Transcript} (hl : I.tags.merkleLeaf = "hm96-sha512/v1")
    (hh : I.tags.hiddenOutputs = false) (h : Accepts I t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupOf I t.record = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧ sts.size = spec.tables.size ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size),
        (∃ d, Stmt.setupH (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables d I.partition
          I.program = .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)) ∧
        sts[j].merkle = MerkleScheme.hm96Sha512 ∧
        ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧ 22 ≤ sts[j].m ∧
          ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m),
          ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (ms : List Msg) (out : OutCL 𝔽 Dig (List UInt8)),
            decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ ∧
            decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ ∧
            sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
            sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
            Plays fOf (tableAfterL execArith H512 enc512 leaf512 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
                PT_LOCAL sts[j].mPts (capDig p₀.cap) (ptsOf sts[j].mPts sess.points)) ms
              (s₀.rounds.toList.map (·.coins) ++ s₁.rounds.toList.map (·.coins)) out ∧
            out.accepted = true ∧ out.opens = opensOf p₀ p₁ sch ∧ out.salts = saltsOf p₀ p₁ sch := by
  obtain ⟨spec, sts, sess, hs, hd, hsz, hpos, hall⟩ := accepts_tables_hm96 hh h
  refine ⟨spec, sts, sess, hs, hd, hsz, hpos, fun j hj => ?_⟩
  obtain ⟨⟨d, hset, hdr⟩, sch, hf, hm22, hrest⟩ := hall j hj
  have hms := setupH_tagsAt_hm96 hl hset
  exact ⟨⟨d, hset, hdr⟩, hms, sch, hf, hm22, fun hwf hr hm => hrest hwf hr hm hms⟩

end Accepts

end FlockSoundness.Discharge.Exec
