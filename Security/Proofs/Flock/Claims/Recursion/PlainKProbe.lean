import Proofs.Flock.Claims.Recursion.FlockKSDeployK

namespace FlockDeploy

open Claims.Recursion

example : VK id = V false := rfl

example : nextK id = next false := rfl

example : AccLast sha512 (V false) := accLast_VK id

example (keep : Bool → Bool) (x : Claims.Msg) (f : Fin ((V false).k x) → Coin) : Fin ((VK keep).k x) → Coin := f

example {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin ((V false).k x) → Coin}
    (h : (Claims.run sha512 P (VK id) x ω r).out = .acc) : (Claims.run sha512 P (V false) x ω r).out = .acc := h

end FlockDeploy
