module

public import Properties.Composition
public import VStarRecursive.SampledTwo

/-! The two-stage form accepts nothing while its switch, `FORM`, is off. -/

@[expose] public section

namespace Verity.SampledTwo

open VStar.Recursive.SampledTwo (FORM protocol checkBlobs_refuses)

theorem accepts_never (hF : FORM = false) (bs : Core.Blobs) (p : protocol.Params) (c : protocol.Commitment)
    (e : protocol.Evidence) : ¬ (protocol.onBlobs bs).Accepts p c e := by
  rintro ⟨pub, prof, h⟩
  exact checkBlobs_refuses hF p c e bs pub prof h

theorem decodedAccepted_never (hF : FORM = false) (bs : Core.Blobs) (a b e : ByteArray) :
    ¬ Properties.Composition.DecodedAccepted (protocol.onBlobs bs) a b e := by
  rintro ⟨p, c, ev, -, -, -, hacc⟩
  exact accepts_never hF bs p c ev hacc

end Verity.SampledTwo
