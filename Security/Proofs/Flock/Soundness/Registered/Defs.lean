import Proofs.Flock.Soundness.Binding.Rows
import Proofs.Flock.Verifier.Registered

/-!
# Registered reads: the model's objects

A session reads some of its rows from values registered once before (PROTOCOL.md §16.10): each such row's leaf opens the
registered root the verifier holds, at the position the verifier's own program reads (`Flock.Registered.check_ok`).

* `TwoOpenings`: two openings of one position under one registered root. They conflict (`Conflict`) when the verifier
  accepts both and their leaves differ, and `pair` is the executable's extractor (`Flock.Registered.openPair`) on them,
  as byte lists: an explicit SHA-512 collision when they conflict (`TwoOpenings.collides`).
* `Reads`: a layout's registered reads. The commit strings whose rows the session reads from a registered value, the root
  the verifier holds and the position its program reads at each, and the registrant's opening there: the row and salt
  it committed, and their path. `Checked` is what the verifier's registered check establishes at a registration, and
  `Off` the event that some registered read's leaf is not the registrant's.
-/

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

/-- The verifier accepts both openings (`Flock.Registered.opensLeaf`), and their leaves differ. -/
def Conflict (o : TwoOpenings) : Prop :=
  o.leaf₁ ≠ o.leaf₂ ∧ Flock.Registered.opensLeaf o.port o.leaf₁.1 o.pos o.path₁ = true ∧
    Flock.Registered.opensLeaf o.port o.leaf₂.1 o.pos o.path₂ = true

/-- The extractor's two SHA-512 inputs (`Flock.Registered.openPair`), as byte lists. -/
def pair (o : TwoOpenings) : List UInt8 × List UInt8 :=
  let x := Flock.Registered.openPair o.port o.pos o.leaf₁.1 o.path₁ o.leaf₂.1 o.path₂
  (x.1.data.toList, x.2.data.toList)

end TwoOpenings

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

/-- **A layout's registered reads.** The commit strings whose rows the session reads from a value registered before
(`isRead`), each an `hm96-sha512/row/v1` row (`read_row`); at each, the registered root the verifier holds (`port`, from
the registration log's held roots) and the position the verifier's program reads there (`readAt`); and the registrant's
opening there: the row and salt it committed (`wv`, `ws`) and their path (`wpath`), which open that root at that
position (`committed`). -/
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

/-- The registrant's leaf at `p`: hm96's leaf of the row and salt it committed there. -/
def leaf (p : lay.Pos) : Dig := (ExplicitLeaf.hm96 rs.enc rs.enc_inj).leaf (rd.wv p) (rd.ws p)

/-- **What the verifier's registered check establishes** at the registration `R` (`Flock.Registered.check_ok`, read at the
model's leaves): at each registered read `p`, the session's leaf there (`Layout.regLeaf`) opens the root the verifier
holds, at the position the verifier's program reads, through `path p`. -/
def Checked (R : Reg) (path : lay.Pos → ByteArray) : Prop :=
  ∀ p, rd.isRead p → Flock.Registered.opensLeaf (rd.port p) (lay.regLeaf R p).1 (rd.readAt p) (path p) = true

/-- The session's opening at `p` under the registration `R`, through `path p`, and the registrant's. -/
def openings (R : Reg) (path : lay.Pos → ByteArray) (p : lay.Pos) : TwoOpenings :=
  ⟨rd.port p, rd.readAt p, lay.regLeaf R p, path p, rd.leaf p, rd.wpath p⟩

/-- Some registered read's leaf under the registration `R` is not the registrant's. -/
def Off (R : Reg) : Prop := ∃ p, rd.isRead p ∧ lay.regLeaf R p ≠ rd.leaf p

/-- A registered read whose leaf under `R` is not the registrant's, if there is one (else any commit string). -/
noncomputable def offAt (R : Reg) : lay.Pos := Classical.epsilon fun p => rd.isRead p ∧ lay.regLeaf R p ≠ rd.leaf p

/-- The registered-path finder's output: the extractor's pair from the session's opening at `offAt R` through `path` and
the registrant's. -/
noncomputable def pick (R : Reg) (path : lay.Pos → ByteArray) : Option (List UInt8 × List UInt8) :=
  some (rd.openings R path (rd.offAt R)).pair

end Reads

end FlockSoundness.Registered
