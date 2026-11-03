import Proofs.Flock.SessionLoop.SchedAt
import Proofs.Flock.SessionLoop.SchedUnrefused

/-!
# A rep's ring-switch rounds

`LiveSched`'s equation at rep stream `2j + r`'s rounds from its rank check's round (`Masked.rankSlot`) to the round
before the opening's binding (`Masked.repSlot` `0`): the opening's first round (the PCS label, where the frame runs the
rank check, and ring-switch claim `0`), claims `1` to `N − 1`, and `γ`'s squeeze, `N = 2 + points · regions` the rep's
claims. Past the algebraic phase passing is byte equality, so the release is the frame's scheduled round, the ring
switch's body at the switch vectors of the frame's table over the claims' points; the switched session sends the same
body at its table's vectors (`Masked.swSrcE_sv`).

* **`LeanRelease.release_sched_switch`**: the release of the worker's round there is the switched session's message
  without its leading domain frame.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

namespace LeanRelease

/-- **The ring switch's rounds**: at rep stream `2j + r`'s round `k` from its rank check's round, before the opening's
binding (`Masked.repSlot` `0`), the release of the worker's round is the switched session's message without its leading
domain frame. -/
theorem release_sched_switch (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
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
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {k : ℕ}
    (hn : ops.length = Masked.rankSlot x j.val r.val + k)
    (hk : ops.length < Masked.repSlot x (2 * j.val + r.val) fun _ => 0) :
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) := by
  sorry

end LeanRelease

end Claims.ZKServer
