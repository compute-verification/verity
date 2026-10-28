import Proofs.Flock.Claims.ZK.FrameRho

/-!
# The session loop's own frames refuse every coin fault (`FrameUnrefused` at `ofLoop`)

At a `Live` statement the loop serves, each fault `Masked.Refused` shows by `refK x` is one a rep frame refuses at its
round: an outer coin `1` (`outer_unrefused`), a rank-deficient mask slot (`rank_unrefused`), `ρ₀, ρ₁` dependent
(`rho_unrefused`) and `β = 0` (`beta_unrefused`).

* `LeanRelease.slots_lt`: a rep's outer and `β` rounds are below `refK x`.
* **`LeanRelease.frameUnrefused_ofLoop`**: `FrameUnrefused` at `ofLoop`, at every statement the loop serves.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Transcript TMW)

namespace LeanRelease

open Loop

/-- **A rep's outer and `β` rounds are the session's**: below `refK x`, at a `Live` statement. -/
theorem slots_lt {x : Msg} (hx : Masked.Live x) {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) :
    Masked.outerSlot x j r < refK x ∧ Masked.betaSlot x j r < refK x := by
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨y⟩ := hx.2 j hj
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hK := Masked.refK_eq hsx
  have hq2 : 2 * j + r < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hst : (Masked.stsOf x)[(2 * j + r) / 2]! = (Masked.stsOf x)[j] := by
    rw [show (2 * j + r) / 2 = j by omega]
    exact getElem!_pos _ j hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]; exact y.hz
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by rw [hst]; exact y.wf.1.hk6
  have hend : FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j + r) / 2]! ≤ refK x := by
    rw [← FlockComplete.ZkTables.repStart_succ, hK, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x) (by omega)
  obtain ⟨N, hN⟩ : ∃ N, N = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) := ⟨_, rfl⟩
  have hR4 : FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j + r) / 2]! = N + 4 := by
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [repRounds_true]
    simp only [hz]
    rw [hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  rw [hR4] at hend
  rw [Masked.outerSlot_eq hsx, Masked.repBase_eq hsx]
  unfold Masked.betaSlot
  rw [Masked.repSlot_eq hsx hz]
  constructor <;> omega

/-- **The session loop's own frames refuse every coin fault** (`FrameUnrefused` at `ofLoop`), at every statement the
loop serves: each fault by `refK x` is one the frames refuse at its round, below `refK x` (`slots_lt`). -/
theorem frameUnrefused_ofLoop (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (H : Msg → Msg) (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hSv : Serves x) :
    FrameUnrefused loopNr T (ofLoop T L₀ real) H wk x w := by
  intro hx ν t κ m₀ ρ₀ ops hl hlive _ hr hR
  rcases hR with ⟨j, hj, r, hr2, -, e, he, h1⟩ | ⟨j, hj, r, hr2, hth⟩
  · exact outer_unrefused T L₀ real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops hl hlive hr hj hr2 (slots_lt hx hj hr2).1 he h1
  · rcases hth with ⟨-, -, -, -, hbad⟩ | ⟨-, hd⟩ | ⟨-, hz⟩
    · exact rank_unrefused T L₀ real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops hl hlive hr hj hr2 hbad
    · rw [rho_unrefused T L₀ real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops hl hlive hr hj hr2] at hd
      exact Bool.noConfusion hd
    · rw [beta_unrefused T L₀ real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops hl hlive hr hj hr2 (slots_lt hx hj hr2).2] at hz
      exact Bool.noConfusion hz

end LeanRelease

end Claims.ZKServer
