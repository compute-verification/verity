module

public import Proofs.Flock.Claims.Recursion.NextJ

/-! The last message with each rep's proof decoded (`FinalP`), whose decision is `decisionAtJ`'s at the proofs' one encodings. -/

@[expose] public section

namespace FlockDeploy

open Flock
open Lean (Json)

/-- **The prover's last message with the proofs decoded**: the statement, the record's parsed value and each rep's
proof, in `spec.repStreams`' order. -/
structure FinalP where
  stmt : Claims.Msg
  j : Json
  ps : Array Flock.Proof

/-- The shape rep `k`'s proof file is read at, at the link points `pts`: table `k / 2`'s, on its `fast100` schedule,
with its extra claims at `pts` (`Flock.repFileShape`). -/
def shapeAt (tabs : Array Flock.Setup) (pts : Array (Array F128)) (k : ℕ) : Flock.Shape :=
  Flock.repFileShape tabs[k / 2]! ((fast100 tabs[k / 2]!.m).getD default)
    (match tabs[k / 2]!.extra pts with
     | .ok e => e.size
     | .error _ => 0)

/-- The proofs' files: each proof's one encoding at its shape (`Wire.proofFile`). -/
def filesAt (tabs : Array Flock.Setup) (pts : Array (Array F128)) (ps : Array Flock.Proof) : Array ByteArray :=
  ps.mapIdx fun k p => Wire.proofFile (shapeAt tabs pts k) p

theorem filesAt_eq {tabs : Array Flock.Setup} {pts : Array (Array F128)} {ps : Array Flock.Proof} {bs : Array ByteArray}
    (hsz : bs.size = ps.size)
    (hdec : ∀ (k : ℕ) (hk : k < ps.size), decodeProof (shapeAt tabs pts k) (bs[k]'(by omega)) = .ok ps[k]) :
    filesAt tabs pts ps = bs := by
  refine Array.ext (by simp only [filesAt, Array.size_mapIdx, hsz]) fun k h₁ h₂ => ?_
  have hk : k < ps.size := by simpa only [filesAt, Array.size_mapIdx] using h₁
  simp only [filesAt, Array.getElem_mapIdx]
  exact (decodeProof_read (hdec k hk)).2

/-- **The decision at a session with the proofs decoded**: `decisionAtJ` at the proofs' files, at the live first coin's
link points. -/
def decisionAtP (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : FinalP) :
    Claims.OCV Unit :=
  decisionAtJ s spec sts x link c₁ mids ⟨f.stmt, f.j, filesAt (tabsOf sts) (pointsOf spec c₁) f.ps⟩

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
