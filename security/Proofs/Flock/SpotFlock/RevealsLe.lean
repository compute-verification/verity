module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.SamplingGeneric.Deployed

/-! SPOT[Flock]'s reveal count bounds what a session reveals (`revealsLe`) and so the opening checks' calls
(`openCallsLe`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

theorem reveal_code :
    (code h).reveal = Reveal.reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) := rfl

theorem reveals_code : (code h).reveals = Reveal.reveals (partition h) (parse h) := rfl

theorem revealsLe : RevealsLe (code h) := fun _ _ hx u v wF =>
  Reveal.length_le (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) (parse h) hx u v wF

theorem openCallsLe {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (H : Msg → Msg) : (compiled F (code h) iroots).OpenCallsLe H :=
  openCallsLe_compiled F (code h) iroots H (revealsLe h)

end SpotFlock
