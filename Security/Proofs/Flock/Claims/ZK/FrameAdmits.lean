import Proofs.Flock.Claims.ZK.FrameZK
import Proofs.Flock.Claims.ZK.FinishRound
import Proofs.Flock.Claims.ZK.AdmitsOf
import Proofs.Flock.Claims.ZK.LinkRecheck

/-!
# `FrameAdmits` at the session loop: the tables' side and the frame's

`LeanRelease.FrameAdmits` (every table admits the coins of a live history the frame takes) splits at `Masked.Refused`,
the coin faults some rep frame refuses at its round:

* `Flock.Guarantees.TablesAdmitUnrefused T`: an obligation on the tables `T`, that where no fault shows by the session's
  last coin at a `Live` statement every table admits the coins. At the deployed tables it is proved
  (**`tablesAdmitUnrefused_dataE`**, `Masked.admits_of_not_refused_refK`; off `Live` they have none).
* `LeanRelease.FrameUnrefused`: at a `Live` statement, on a live history of `refK x` openings whose checks pass and
  whose last coins the frame takes, no fault shows: the rep frames refuse each fault at its round. Narrower than
  `FrameAdmits`: it reads no table. At the session loop's own frames it is proved
  (`LeanRelease.frameUnrefused_ofLoop`).
* **`LeanRelease.frameAdmits_of_unrefused`**: `FrameAdmits` from the two, and **`LeanRelease.finalRoundEq_ofLoop`**:
  `FinalRoundEq` at `ofLoop` from them and `FinishIs` (`finalRoundEq_of_finishIs`, `loop_last`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal

/-- Obligation (the tables admit unrefused coins): at every statement `x`, coins `c` at which, where `x` is `Live`, no
coin fault shows by the session's last coin (`Masked.Refused`: an outer coin `1`, `ρ_0, ρ_1` dependent, a rank-deficient
mask slot, `β = 0`) are coins every table of `T` admits. The deployed tables meet it
(`LeanRelease.tablesAdmitUnrefused_dataE`). -/
def _root_.Flock.Guarantees.TablesAdmitUnrefused (T : CoinLayer.TableData) : Prop :=
  ∀ (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin), (Masked.Live x → ¬ Masked.Refused x c (refK x)) →
    ∀ j, (T.C x j).Admits (T.cz x c j)

namespace LeanRelease

/-- **The deployed tables admit unrefused coins** (`Masked.admits_of_not_refused_refK` at a `Live` statement; at any
other the deployed session has no table). -/
theorem tablesAdmitUnrefused_dataE : _root_.Flock.Guarantees.TablesAdmitUnrefused Masked.dataE := by
  intro x c hn j
  by_cases hx : Masked.Live x
  · exact Masked.admits_of_not_refused_refK Masked.PE hx (hn hx) j
  · have h0 : Masked.jOf x = 0 := by unfold Masked.jOf; simp [hx]
    exact absurd j.isLt (by
      show ¬ j.val < Masked.jOf x
      rw [h0]
      exact Nat.not_lt_zero _)

/-- **The frames refuse every coin fault**: at a `Live` statement, after every live history of `refK x` openings
whose openings pass and whose last coins the frame takes, no coin fault shows at the coins the openings open
(`Masked.Refused` by `refK x`): each rep frame refuses each fault at its round (`Frame.Masker.answer`,
`Frame.maskRank`), on the state and coins the transport hands it (`prog`, `coinsOf`). -/
def FrameUnrefused (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  Masked.Live x → ∀ (ν : (CoinLayer.session nr T).Nonce) (t : (CoinLayer.session nr T).Tape) (κ : L.toSrv.Chk)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    ops.length = refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session nr T).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    ¬ Masked.Refused x ((CoinLayer.session nr T).pad (ops.map fun p => (CoinLayer.session nr T).decOf p.2)) (refK x)

/-- **`FrameAdmits` from the tables admitting unrefused coins and the frames refusing every fault.** -/
theorem frameAdmits_of_unrefused (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hT : _root_.Flock.Guarantees.TablesAdmitUnrefused T) (hU : FrameUnrefused nr T L H wk x w) :
    FrameAdmits (CoinLayer.session nr T) L H wk x w :=
  fun ν t κ m₀ ρ₀ ops hl hlive hok hr => hT x _ fun hx => hU hx ν t κ m₀ ρ₀ ops hl hlive hok hr

/-- **`FinalRoundEq` at the session loop's own frames** from the tables admitting unrefused coins, the frames refusing
every fault and the firewall's `Finish` being the session's (`finalRoundEq_of_finishIs`, at `loop_last`). -/
theorem finalRoundEq_ofLoop (T : CoinLayer.TableData) (vs : (CoinLayer.session Loop.loopNr T).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (H : Msg → Msg) (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hT : _root_.Flock.Guarantees.TablesAdmitUnrefused T)
    (hU : FrameUnrefused Loop.loopNr T (ofLoop T L₀ real) H wk x w)
    (hF : FinishIs (CoinLayer.session Loop.loopNr T) vs (ofLoop T L₀ real) H wk x w) :
    FinalRoundEq (CoinLayer.session Loop.loopNr T) vs (ofLoop T L₀ real) H wk x w :=
  finalRoundEq_of_finishIs _ vs _ H wk x w (loop_last Loop.loopNr T (withFrames T L₀ real) x)
    (frameAdmits_of_unrefused Loop.loopNr T _ H wk x w hT hU) hF

end LeanRelease

end Claims.ZKServer
