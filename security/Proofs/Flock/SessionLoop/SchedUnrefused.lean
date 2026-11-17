module

public import Proofs.Flock.SessionLoop.PcsRank
public import Proofs.Flock.SessionLoop.RhoUnrefused

/-! No refusal shows at a rep's scheduled rounds up to its `β` round (`LeanRelease.unrefused_sched`). -/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V)

namespace LeanRelease

open Loop

theorem rho_unrefused_sched (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) (hN : Masked.rhoSlot x j r + 1 < ops.length) :
    _root_.Flock.Zk.rhoIndependent (Masked.rhoAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r) = true :=
  rho_unrefused_sched_pf T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 hN

/-- Holds where rep `q`'s rank check holds at the opened coins once its round is reached (`hrk`). -/
theorem unrefused_of_rank (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {q : ℕ} (hq : q < 2 * (Masked.stsOf x).size) (hq0 : Masked.repBase x q ≤ ops.length)
    (hq1 : ops.length ≤ Masked.betaSlot x (q / 2) (q % 2))
    (hrk : Masked.rankSlot x (q / 2) (q % 2) ≤ ops.length →
      ¬ Masked.RankBad (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
        (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) (q / 2) (q % 2)) :
    ¬ Masked.Refused x ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) ops.length := by
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (show 0 < (Masked.stsOf x).size by omega)
  have Sq := Masked.slots_order hx hsx (j := q / 2) (r := q % 2) (by omega) (Nat.mod_lt _ (by norm_num))
  rw [Nat.div_add_mod] at Sq
  have hord : ∀ {j r : ℕ}, j < (Masked.stsOf x).size → r < 2 → 2 * j + r ≠ q →
      Masked.betaSlot x j r < ops.length ∨ ops.length < Masked.repBase x (2 * j + r) := by
    intro j r hj hr2 hne
    have Sp := Masked.slots_order hx hsx hj hr2
    rcases Nat.lt_or_ge (2 * j + r) q with h | h
    · left
      have hm := FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x) (show 2 * j + r + 1 ≤ q by omega)
      rw [← Masked.repBase_eq hsx, ← Masked.repBase_eq hsx] at hm
      omega
    · right
      have hm := FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x) (show q + 1 ≤ 2 * j + r by omega)
      rw [← Masked.repBase_eq hsx, ← Masked.repBase_eq hsx] at hm
      omega
  rintro (⟨j, hj, r, hr2, ho, e, he, h1⟩ | ⟨j, hj, r, hr2, hth⟩)
  · exact outer_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 ho he h1
  · have Sp := Masked.slots_order hx hsx hj hr2
    rcases hth with ⟨-, -, -, hrs, hbad⟩ | ⟨hρ, hd⟩ | ⟨hβ, hz⟩
    · by_cases hcur : 2 * j + r = q
      · subst hcur
        have e1 : (2 * j + r) / 2 = j := by omega
        have e2 : (2 * j + r) % 2 = r := by omega
        rw [e1, e2] at hrk
        exact hrk hrs hbad
      · rcases hord hj hr2 hcur with hb | hb
        · exact rank_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 hb hbad
        · omega
    · by_cases hcur : 2 * j + r = q
      · rw [rho_unrefused_sched T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 hρ] at hd
        exact Bool.noConfusion hd
      · rcases hord hj hr2 hcur with hb | hb
        · rw [rho_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 hb] at hd
          exact Bool.noConfusion hd
        · omega
    · by_cases hcur : 2 * j + r = q
      · subst hcur
        have e1 : (2 * j + r) / 2 = j := by omega
        have e2 : (2 * j + r) % 2 = r := by omega
        rw [e1, e2] at hq1
        omega
      · rcases hord hj hr2 hcur with hb | hb
        · rw [beta_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 hb] at hz
          exact Bool.noConfusion hz
        · omega

/-- Not at the rank check's own round, where the frame decides the check on the round it schedules. -/
theorem unrefused_sched (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {q : ℕ} (hq : q < 2 * (Masked.stsOf x).size) (hq0 : Masked.repBase x q ≤ ops.length)
    (hq1 : ops.length ≤ Masked.betaSlot x (q / 2) (q % 2)) (hpcs : ops.length ≠ Masked.rankSlot x (q / 2) (q % 2)) :
    ¬ Masked.Refused x ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) ops.length :=
  unrefused_of_rank T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hq hq0 hq1 fun hN =>
    rank_unrefused_sched T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr (by omega) (Nat.mod_lt _ (by norm_num))
      (by omega)

end LeanRelease

end Claims.ZKServer
