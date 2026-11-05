module

public import Proofs.Flock.SessionLoop.PcsRank
public import Proofs.Flock.SessionLoop.RhoUnrefused

/-!
# No refusal shows at a rep's scheduled rounds

The session sends `[]` at a round by which the coins show a refusal (`Masked.encStop`, `Masked.Refused`).
`LeanRelease.unrefused_prefix` rules one out up to the rep's rank check; past it, the rep's own rank check and its level
0's pair `ρ₀, ρ₁` count too, which the frame decides from the coins at the opening's label (`Masker.label`, the rank
check `LeanRelease.rank_unrefused_sched` reads, in `PcsRank`) and at `ρ₁`'s coins (`Masker.answer`), before the rep's
`β` round:

* **`LeanRelease.rho_unrefused_sched`**: on a live history past rep `2j + r`'s `ρ₁` round, its pair coins are
  independent (`LeanRelease.rho_unrefused_sched_pf`, in `RhoUnrefused`).
* **`LeanRelease.unrefused_of_rank`**: on a live history whose next round is one of rep `q`'s up to its `β` round, no
  refusal shows by that round where rep `q`'s rank check holds once its round is reached: an outer coin, an earlier
  rep's throw and a later rep's are `unrefused_prefix`'s cases, and rep `q`'s pair is `rho_unrefused_sched`'s.
* **`LeanRelease.unrefused_sched`**: the same past rep `q`'s rank check's round, which `rank_unrefused_sched` covers.
  At the round itself the frame decides the check on the round it schedules, which `unrefused_of_rank`'s caller reads.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V)

namespace LeanRelease

open Loop

/-- **The frame refuses dependent `ρ₀, ρ₁` at `ρ₁`'s coins**: on a live history of at most `refK x` openings whose last
coins the frame takes, past table `j`'s rep `r`'s `ρ₁` round (`Masked.rhoSlot` `+ 1`), the rep's pair coins at the
opened coins are independent. The frame took `ρ₁`'s coins (`Masker.answer`, level 0's pair check). -/
theorem rho_unrefused_sched (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) (hN : Masked.rhoSlot x j r + 1 < ops.length) :
    _root_.Flock.Zk.rhoIndependent (Masked.rhoAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r) = true :=
  rho_unrefused_sched_pf T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 hN

/-- **No refusal shows at a rep's scheduled rounds where its rank check holds**: on a live history of at most `refK x`
openings whose last coins the frame takes, whose next round is rep `q`'s, from its first up to its `β` round, where rep
`q`'s rank check holds at the opened coins once its round is reached (`hrk`), no refusal shows by that round. An
answered outer coin is refused (`outer_unrefused`); a rep before `q` ended before its first round, so its throws are
refused (`rank_unrefused`, `rho_unrefused`, `beta_unrefused`); a rep after `q` has none yet (`Masked.slots_order`); and
rep `q`'s pair is refused once shown (`rho_unrefused_sched`), its `β` not yet answered. -/
theorem unrefused_of_rank (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
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

/-- **No refusal shows at a rep's scheduled rounds past its rank check's**: `unrefused_of_rank` at every round of rep
`q`'s up to its `β` round but its rank check's, where the rank check holds once its round is past
(`rank_unrefused_sched`). -/
theorem unrefused_sched (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
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
