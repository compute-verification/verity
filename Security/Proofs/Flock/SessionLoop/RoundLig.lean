import Proofs.Flock.SessionLoop.SchedAt

/-!
# A rep's Ligerito rounds past level 0

`LiveSched`'s equation at rep stream `2j + r`'s Ligerito rounds after level 0's closing round (`Masked.repSlot`
`k0 + 4`) and before the inner proof (`Masked.repSlot` `k0 + 7 + 11R`, `R` the levels past level 0): the levels ≥ 1
(each a fold, its commitment at the tape's salts, its sumcheck) and the queries. Past the algebraic phase passing is
byte equality, so the release is the frame's scheduled round, which the opening's feed computes from the frame's table,
the levels' salts and the coins; the switched session sends the same round from its table (`Masked.swSrcE`).

* **`LeanRelease.release_sched_lig`**: the release of the worker's round there is the switched session's message
  without its leading domain frame.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

namespace LeanRelease

/-- **Ligerito past level 0**: at rep stream `2j + r`'s round `i` past its PIOP (`Masked.repSlot` `i`), from
`k0 + 5` to the last before the inner proof, the release of the worker's round is the switched session's message
without its leading domain frame. -/
theorem release_sched_lig (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {v : Msg}
    (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i) (h1 : (Masked.zkTab x j).sch.k0 + 5 ≤ i)
    (h2 : i < (Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)) :
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) := by
  sorry

end LeanRelease

end Claims.ZKServer
