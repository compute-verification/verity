module

public import Properties.Certifier.Check

/-! The strings a record head, a signed digest, a frame tree and the run hash, as the predicates the injectivity lemmas
take. -/

@[expose] public section

open NetworkCheck hiding Params
open NetworkCheck.Check
open NetworkAttester (ClockSync)

namespace Properties.Certifier

def HeadString (h : RecordHead) (x : ByteArray) : Prop :=
  ∃ enc, h.encode = .ok enc ∧ Properties.Identity.taggedInput recordTag enc = .ok x

def FramesString (g : Grid) (frames : Array ByteArray) (x : ByteArray) : Prop :=
  (∃ f ∈ frames.toList, leafInput f g.frameBytes = .ok x) ∨ x = emptyInput ∨
    (∃ l < depth g, x = nodeInput (empty l) (empty l)) ∨
    ∃ ls, leaves frames g = .ok ls ∧ ∃ lv < depth g, x ∈ pairInputs lv (levelAt lv ls.toList)

def TreeString (g : Grid) (grid : Array (Array (Option ByteArray))) (x : ByteArray) : Prop :=
  ∃ frames, realFrames grid g = .ok frames ∧ FramesString g frames x

def SaltedString (salt : ByteArray) (h : RecordHead) (x : ByteArray) : Prop :=
  ∃ dg, h.digest = .ok dg ∧ x = certificateInput salt dg

def CheckString (p : Params) (c : Commitment) (e : Evidence) (eg : ByteArray) (x : ByteArray) : Prop :=
  (∃ r ∈ c.records.toList, HeadString r.head x ∨ SaltedString r.salt r.head x ∨
    messageInput r.certificate.keyId p.session r.certificate.link r.certificate.direction r.certificate.window
      r.certificate.digest = .ok x) ∨
  ∃ l link, (l, link) ∈ p.egress.toList.zip e.links.toList ∧ ∃ w < p.windows,
    ∃ syncs : Fin p.windows → ClockSync, Declares link syncs ∧ (TreeString p.grid (programGrid p syncs link.frames w) x ∨
      ∃ h, recordHead l .egress w (programGrid p syncs link.frames w) none p.grid eg true = .ok h ∧ HeadString h x)

end Properties.Certifier
