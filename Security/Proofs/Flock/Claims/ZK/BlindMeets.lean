import Proofs.Flock.Claims.ZK.Blind

/-!
# `zk_server_blind`'s hypotheses on the values and the server hold together

`Flock.zk_server_blind` takes, beyond the session's five Props, two hypotheses on the workers' values `vs` and the
server: `SwCausal vs`, the switched round encoding's causality, and `hRef`, `RefLive` at the worker's own values (where
the run is live, the server's frame at those values is `swP vs`'s message). Both hold at once:

* `Flock.zk_server_blind_swCausal_meets`: at each table's own values (`RefSession.honestVals`), `SwCausal` is the
  session's `Causal` (`RefSession.swCausal_vals`).
* `Flock.zk_server_blind_ref_meets`: at the server that recomputes everything (`ZkServer.solo`, at `swP vs`'s prover),
  `RefLive` holds at every pad-blind worker and every `vs`: its frame reads no worker value.

So the two hypotheses are not contradictory. That server's workers contribute nothing: the server the deployment runs,
which masks the workers' values, is not defined in the tree, and its `RefLive` is an obligation of its definition.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal

/-- **`SwCausal` at the tables' own values** is the session's `Causal`. -/
theorem Flock.zk_server_blind_swCausal_meets (S : RefSession) (hc : S.Causal) : S.SwCausal S.honestVals :=
  S.swCausal_vals hc

/-- **`RefLive` at the server that recomputes everything**, at every pad-blind worker and every `vs`. -/
theorem Flock.zk_server_blind_ref_meets (H : Msg → Msg) (S : RefSession) (vs : S.SwVals)
    (bw : (ZkServer.solo (W := Flock.Wit) (R := S.refP.Nonce × S.refP.Rand) H (S.swP vs).prover).BWorker)
    (x : Msg) (w : Flock.Wit) :
    let srv := ZkServer.solo (W := Flock.Wit) (R := S.refP.Nonce × S.refP.Rand) H (S.swP vs).prover
    (srv.withHonest (srv.srcOf (srv.lift bw))).RefLive H (S.swP vs).prover (srv.lift bw) x w := by
  intro srv ρ κ h _
  rfl

end Claims.ZKServer
