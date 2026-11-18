module

public import Proofs.Flock.Claims.Steps.NextP
public import Proofs.Flock.Claims.Recursion.NextJ

/-!
# The last message with its proofs decoded (`FinalP`)

`FinalJ` carries the proof files, which the verdict decodes inside `verifyRepWith`, a check of the last step that no
V\* part computes (`NextJ`'s open gap). `FinalP` carries each rep's decoded proof instead, and its decision
(`decisionAtP`) is `decisionAtJ` at the proofs' one encodings (`filesAt`): rep `k`'s at table `k / 2`'s shape on its
`fast100` schedule, with its extra claims at the live first coin's link points (`shapeAt`), the points the record's value
carries once `coinSubJ` substitutes the live coins.

* **The decision is `decisionAtJ`'s** (`decisionAtP_eq`): at proof files that decode at those shapes to the proofs, the
  two decisions are one, decoding being canonical (`decodeProof_read`: a file that decodes to `p` is `p`'s one encoding,
  `filesAt_eq`).
* **What stays**: the verdict still decodes `filesAt`'s files, each a proof's own encoding at `shapeAt`. With
  `Flock.verifyRepWith_split` that decode is `decodeProof` at `Flock.repFileShape`, which accepts a proof's own
  encoding at every proof of the shape (`FlockVBridge.Family.decodeProof_encodeProof`), where `shapeAt` is the verdict's
  shape; the reps' proofs are `RepsRead`'s, read from their step messages.
-/

@[expose] public section

namespace FlockDeploy

open Flock

open Lean (Json)

/-- **Files that decode to the proofs are their files**: decoding is canonical. -/
theorem filesAt_eq {tabs : Array Flock.Setup} {pts : Array (Array F128)} {ps : Array Flock.Proof} {bs : Array ByteArray}
    (hsz : bs.size = ps.size)
    (hdec : ∀ (k : ℕ) (hk : k < ps.size), decodeProof (shapeAt tabs pts k) (bs[k]'(by omega)) = .ok ps[k]) :
    filesAt tabs pts ps = bs := by
  refine Array.ext (by simp only [filesAt, Array.size_mapIdx, hsz]) fun k h₁ h₂ => ?_
  have hk : k < ps.size := by simpa only [filesAt, Array.size_mapIdx] using h₁
  simp only [filesAt, Array.getElem_mapIdx]
  exact (decodeProof_read (hdec k hk)).2

/-- **The decoded proofs' decision at a session is `decisionAtJ`'s** at proof files that decode to them, file by file,
at `shapeAt`. -/
theorem decisionAtP_eq (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {f : FinalJ}
    {ps : Array Flock.Proof} (hsz : f.proofs.size = ps.size)
    (hdec : ∀ (k : ℕ) (hk : k < ps.size),
      decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) (f.proofs[k]'(by omega)) = .ok ps[k]) :
    decisionAtP s spec sts x link c₁ mids ⟨f.stmt, f.j, ps⟩ = decisionAtJ s spec sts x link c₁ mids f := by
  obtain ⟨stmt, j, proofs⟩ := f
  have h := filesAt_eq hsz hdec
  simp only [decisionAtP, h]

end FlockDeploy
