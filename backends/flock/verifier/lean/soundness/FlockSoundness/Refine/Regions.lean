import FlockSoundness.Refine.StmtOf
import FlockSoundness.Refine.Table
import FlockSoundness.Protocol.Refine.Regions

/-!
# The statement decoding, part 2: the regions are the model's (PROTOCOL.md §11, §16.2)

`regionOf` reads a region of the executable's statement as the model's: its free bits, the fixed bits the executable's
`regionPoint` sets, and bit `k` of block `b`'s bytes. `ofCircuit_extra`: at two link points of `m_pts` coordinates, the
extra claims `Setup.ofCircuit st` hands the verifier are the model's `extraClaims` of `stmtOf st` with those regions
(`ExtraRealizes`). It needs `RegionsWF`: `m = k_log + nbl`, `m_pts = PT_LOCAL + nbl`, and each region's free bits are
distinct in-block bits, at most `PT_LOCAL` of them.

`verify_refines_ofCircuit` composes this and `ofCircuit_fold` with `verify_refines`: for a verifier set up with
`Setup.ofCircuit st`, its conclusion holds for `stmtOf st`, with nothing assumed of the fold or the extra claims. The
record's link points have the configured shape because `Record.decode` checks it (S9, `decode_points`).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3

local notation "𝔽" => FlockLevel3.GF128

/-! ### `regionValue` -/

/-! ### The regions, as the model's -/

/-! ### `regionPoint` -/

/-! ### `extraClaims`: region-major, then point by point -/

/-! ### The extra claims are the model's -/

/-! ### The record's link points, and the composition -/

open Game Model in
/-- **The verifier refines the model's reps, for the statements it is set up with** (§3, §5, §10, §16). When
`Flock.verify` accepts a one-table session under the unsalted SHA-512 scheme, with the setup `Setup.ofCircuit st` of a
statement whose layout and regions are well formed, the conclusion of `verify_refines` holds for the model statement
`stmtOf st` read off the layout, with the extra claims the model's `extraClaims` at the record's link points. -/
theorem verify_refines_ofCircuit (st : Stmt) (hwf : StmtWF st) (hr : RegionsWF st) (hm : 13 ≤ st.m) (msId : String)
    (hms : st.merkle = MerkleScheme.sha512 msId) (record : String) (proofs : Array ByteArray)
    (h : Flock.verify st.spec #[Setup.ofCircuit st] record proofs = .ok ()) :
    ∃ (sess : Session) (sch : Flock.Schedule) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
      Record.decode st.spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
        (ptsOf st.mPts sess.points)).length) proofs[0]! = .ok p₀ ∧
      decodeProof (shapeOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
        (ptsOf st.mPts sess.points)).length) proofs[1]! = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString st.spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString st.spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (Model.repC execArith 0 (stmtOf st hwf (regionsOf st)) (schedOf sch) hm
          (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points)))
        (repMsgs (Setup.ofCircuit st) sch (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
          (ptsOf st.mPts sess.points)).length) p₀ capDig) (s₀.rounds.toList.map (·.coins)) pend₀ ∧
      Plays fOf (Model.repC execArith 1 (stmtOf st hwf (regionsOf st)) (schedOf sch) hm
          (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points)))
        (repMsgs (Setup.ofCircuit st) sch (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
          (ptsOf st.mPts sess.points)).length) p₁ capDig) (s₁.rounds.toList.map (·.coins)) pend₁ ∧
      pend₀.verdict (opensOf p₀ p₁ sch) = true ∧ pend₁.verdict (opensOf p₀ p₁ sch) = true ∧
      OpensOK H512 enc512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) ∧
      OpensOK H512 enc512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) :=
  verify_refines st.spec (Setup.ofCircuit st) (stmtOf st hwf (regionsOf st)) hm rfl rfl (ofCircuit_fold st hwf _) msId hms
    (fun sess => Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points))
    record proofs
    (fun sess hsess => ofCircuit_extra st hwf hr sess.points (decode_points hsess).1 (decode_points hsess).2) h

end FlockSoundness.Refine
