import Proofs.Flock.Claims.Recursion.PlainK

namespace FlockDeploy

open Claims.Recursion

example : VK id = V false := rfl

example : nextK id = next false := rfl

example (rel : Flock.Readable.Parsed → Flock.Readable.Relation) : deployPlainK id rel = deployPlain rel := rfl

example : AccLast sha512 (V false) := accLast_VK id

example {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin ((V false).k x) → Coin} {q : ℕ}
    {u v : List FlockLevel3.GF128} (h : (Claims.run sha512 P (V false) x ω r).out = .acc)
    (hu : (q, u) ∈ KS.plainRawCols (Claims.run sha512 P (V false) x ω r).seen)
    (hv : (q, v) ∈ KS.plainRawCols (Claims.run sha512 P (V false) x ω r).seen) (huv : u ≠ v) :
    Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω r).vLog :=
  KS.plainK_selfClash_tied id h hu hv huv

end FlockDeploy
