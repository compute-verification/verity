module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.SamplingGeneric.Deployed

/-!
# SPOT[Flock]'s reveal count

SPOT's two-stage code reveals, for a checked unit, one opening per value its proof block's correctness reads
(`Reveal.reveal`), so at a statement that parses a session's witness reveals at most `reveals` openings for a unit
(`revealsLe`), and the opening checks of every deployed record of the code are bounded (`openCallsLe`: at most
`4 + depth own` SHA-512 calls an opening, `openCallsLe_compiled`).

* `reveal_code`, `reveals_code`: the code's reveal and reveal count are the program's;
* `revealsLe`: the reveal count bounds what a session reveals;
* `openCallsLe`: the opening checks' count at a deployed record of the code.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

theorem reveal_code :
    (code h).reveal = Reveal.reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) := rfl

theorem reveals_code : (code h).reveals = Reveal.reveals (partition h) (parse h) := rfl

/-- **SPOT[Flock]'s reveal count bounds what a session reveals**: at a statement that parses, a session's witness
reveals at most `reveals` openings for a unit (`Reveal.length_le`). -/
theorem revealsLe : RevealsLe (code h) := fun _ _ hx u v wF =>
  Reveal.length_le (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) (parse h) hx u v wF

/-- **The opening checks' count** at a deployed record of SPOT[Flock]'s code, from the reveal count
(`openCallsLe_compiled`). -/
theorem openCallsLe {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (H : Msg → Msg) : (compiled F (code h) iroots).OpenCallsLe H :=
  openCallsLe_compiled F (code h) iroots H (revealsLe h)

end SpotFlock
