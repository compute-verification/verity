module

public import Properties.Composition
public import VStarRecursive.SampledTwo

/-!
# The two-stage form accepts nothing while its switch is off

The two-stage form (`VStar.Recursive.SampledTwo`) decides last by its switch, `FORM`, which is `SecondStage.FORM` and
off (`form_off`): its check is never an accept while it is off (`checkBlobs_refuses`). Read at the form's protocol on
any blobs:

* `accepts_never`: the protocol's check accepts no parameters, registration and evidence;
* `decodedAccepted_never`: no bytes of parameters, registration and evidence decode to a case it accepts, the
  composition's reading of a component's accept (`Properties.Composition.DecodedAccepted`), which the sampled
  component's refusal of every form but the one-stage one (`Verity.Sampled.paramsAt_of_verdict`) takes its two-stage
  case from.
-/

@[expose] public section

namespace Verity.SampledTwo

open VStar.Recursive.SampledTwo (FORM protocol checkBlobs_refuses)

/-- **The form's protocol accepts nothing while the switch is off**, on any blobs. -/
theorem accepts_never (hF : FORM = false) (bs : Core.Blobs) (p : protocol.Params) (c : protocol.Commitment)
    (e : protocol.Evidence) : ¬ (protocol.onBlobs bs).Accepts p c e := by
  rintro ⟨pub, prof, h⟩
  exact checkBlobs_refuses hF p c e bs pub prof h

/-- **No bytes decode to a case the form accepts while the switch is off**, on any blobs. -/
theorem decodedAccepted_never (hF : FORM = false) (bs : Core.Blobs) (a b e : ByteArray) :
    ¬ Properties.Composition.DecodedAccepted (protocol.onBlobs bs) a b e := by
  rintro ⟨p, c, ev, -, -, -, hacc⟩
  exact accepts_never hF bs p c ev hacc

end Verity.SampledTwo
