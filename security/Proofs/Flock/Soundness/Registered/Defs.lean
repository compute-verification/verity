module

public import Proofs.Flock.Soundness.Binding.Rows
public import Proofs.Flock.Verifier.Registered
import Mathlib.Combinatorics.Matroid.Init

/-! Registered reads: two openings of one registered position (`TwoOpenings`) and a layout's registered reads
(`Reads`). -/

@[expose] public section

namespace FlockSoundness.Registered

open Refine Binding Audit

/-- Two openings of one position `pos` under one registered root (`port`): two leaves, each with its path. -/
structure TwoOpenings where
  port : Flock.Registered.Port
  pos : ℕ
  leaf₁ : Dig
  path₁ : ByteArray
  leaf₂ : Dig
  path₂ : ByteArray

namespace TwoOpenings

def Conflict (o : TwoOpenings) : Prop :=
  o.leaf₁ ≠ o.leaf₂ ∧ Flock.Registered.opensLeaf o.port o.leaf₁.1 o.pos o.path₁ = true ∧
    Flock.Registered.opensLeaf o.port o.leaf₂.1 o.pos o.path₂ = true

/-- The extractor's two SHA-512 inputs (`Flock.Registered.openPair`), as byte lists. -/
def pair (o : TwoOpenings) : List UInt8 × List UInt8 :=
  let x := Flock.Registered.openPair o.port o.pos o.leaf₁.1 o.path₁ o.leaf₂.1 o.path₂
  (x.1.data.toList, x.2.data.toList)

end TwoOpenings

variable {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

structure Reads (lay : Layout P plan tab decode) (rs : HmRows lay) where
  isRead : lay.Pos → Prop
  read_row : ∀ p, isRead p → rs.isRow p
  port : lay.Pos → Flock.Registered.Port
  readAt : lay.Pos → ℕ
  wv : lay.Pos → lay.Val
  ws : lay.Pos → List UInt8
  wpath : lay.Pos → ByteArray
  committed : ∀ p, isRead p →
    Flock.Registered.opensLeaf (port p) ((ExplicitLeaf.hm96 rs.enc rs.enc_inj).leaf (wv p) (ws p)).1 (readAt p)
      (wpath p) = true

namespace Reads

variable {lay : Layout P plan tab decode} {rs : HmRows lay} (rd : Reads lay rs)

def leaf (p : lay.Pos) : Dig := (ExplicitLeaf.hm96 rs.enc rs.enc_inj).leaf (rd.wv p) (rd.ws p)

def Checked (R : Reg) (path : lay.Pos → ByteArray) : Prop :=
  ∀ p, rd.isRead p → Flock.Registered.opensLeaf (rd.port p) (lay.regLeaf R p).1 (rd.readAt p) (path p) = true

def openings (R : Reg) (path : lay.Pos → ByteArray) (p : lay.Pos) : TwoOpenings :=
  ⟨rd.port p, rd.readAt p, lay.regLeaf R p, path p, rd.leaf p, rd.wpath p⟩

def Off (R : Reg) : Prop := ∃ p, rd.isRead p ∧ lay.regLeaf R p ≠ rd.leaf p

noncomputable def offAt (R : Reg) : lay.Pos := Classical.epsilon fun p => rd.isRead p ∧ lay.regLeaf R p ≠ rd.leaf p

noncomputable def pick (R : Reg) (path : lay.Pos → ByteArray) : Option (List UInt8 × List UInt8) :=
  some (rd.openings R path (rd.offAt R)).pair

end Reads

end FlockSoundness.Registered
