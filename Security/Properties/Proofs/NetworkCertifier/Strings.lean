import Properties.NetworkCertifier

/-!
# The strings the run hashes, as predicates

What a record head, a frame tree and the run hash, as the predicates the injectivity lemmas take: each is a string of
the list the claims search (`headStrings`, `framesStrings`, `treeStrings`, `checkStrings`; `Extract` shows it).
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open NetworkCertifier (ClockSync)

namespace Properties.NetworkCertifier

/-- What SHA-512 hashes for a record head: `tagged_sha512(RECORD_TAG, encode(h))`'s input. -/
def HeadString (h : RecordHead) (x : ByteArray) : Prop :=
  ∃ enc, h.encode = .ok enc ∧ Properties.Identity.taggedInput recordTag enc = .ok x

/-- What SHA-512 hashes for the tree of `frames` under grid parameters `g`: each frame's leaf, each node of each level
below the root, and every empty subtree up to the depth. -/
def FramesString (g : Grid) (frames : Array ByteArray) (x : ByteArray) : Prop :=
  (∃ f ∈ frames.toList, leafInput f g.frameBytes = .ok x) ∨ x = emptyInput ∨
    (∃ l < depth g, x = nodeInput (empty l) (empty l)) ∨
    ∃ ls, leaves frames g = .ok ls ∧ ∃ lv < depth g, x ∈ pairInputs lv (levelAt lv ls.toList)

/-- What SHA-512 hashes for the frame tree of a grid: `FramesString` of its real frames. -/
def TreeString (g : Grid) (grid : Array (Array (Option ByteArray))) (x : ByteArray) : Prop :=
  ∃ frames, realFrames grid g = .ok frames ∧ FramesString g frames x

/-- The strings the check hashes under egress parameters digest `eg`: each record's head and certificate message, and
the head and frame tree it recomputes of each egress link-window. -/
def CheckString (p : Params) (c : Commitment) (e : Evidence) (eg : ByteArray) (x : ByteArray) : Prop :=
  (∃ r ∈ c.records.toList, HeadString r.head x ∨
    messageInput r.certificate.keyId p.session r.certificate.link r.certificate.direction r.certificate.window
      r.certificate.digest = .ok x) ∨
  ∃ l link, (l, link) ∈ p.egress.toList.zip e.links.toList ∧ ∃ w < p.windows,
    ∃ syncs : Fin p.windows → ClockSync, Declares link syncs ∧ (TreeString p.grid (programGrid p syncs link.frames w) x ∨
      ∃ h, recordHead l .egress w (programGrid p syncs link.frames w) none p.grid eg true = .ok h ∧ HeadString h x)

end Properties.NetworkCertifier
