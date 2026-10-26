import Proofs.Flock.Claims.KS.BatchSim
import Proofs.Flock.Claims.KS.PlainLive
import Proofs.Flock.Claims.KS.ZkLock

/-!
# The masked simulator's live strategy submits its rounds stream by stream

The masked simulator sends deployed round `k + 1` as a round on stream `zkStream (k + 1)`, table `zkStream / 2`'s rep
`zkStream % 2` (`zkSub_round`), and its streams are table `0`'s rep 0 `repRounds` times, then its rep 1, then table
`1`'s, and so on (`zkStreams`, `|zkRepNsOf| = repRounds`). So its live strategy, read through `liveSession_eq` as its
bytes prover on the rounds' explicit game (`roundsK`), follows `SchedLJ` at `zkStreams`:

* `SchedKJ`, `schedLJ_of_schedKJ`: the schedule on `roundsK`'s strategies of `J` tables, carried to `liveSession`'s
  across `bind_rounds`;
* `zkSim_sub`, `schedKJ_zk`, `zkLive_sched`: the masked simulator's bytes prover keeps it, round by round;
* `zkRepNsOf_length`: `|zkRepNsOf| = repRounds` at the `--zk` schedule;
* `zkStreams_eq`, `zkStreams_length`, `zkStreams_split`: the streams are `fromS` at `repRounds`, as many as the
  deployed rounds, split at table `j` as `front_simRJ` reads them.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

/-! ## The schedule on the rounds' explicit game -/

section Sched

variable {J mPts : ℕ} (dg : ByteArray → ByteArray) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)

/-- **`SchedLJ` on the rounds' explicit game** (`roundsK`): the next rounds are on the streams `ss`, and then the
strategy stops or runs out of rounds. -/
def SchedKJ : (n : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → List ℕ →
    Strategy (roundsK dg rootB pts n rs) → Prop
  | 0, _, ss, _ => ss = []
  | n + 1, rs, ss, σ => match σ with
    | ⟨none, _⟩ => ss = []
    | ⟨some s, τ⟩ => ∃ ss', ss = (2 * s.tab.val + s.rep.val) :: ss' ∧
        ∀ c, SchedKJ n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)) ss' (τ c)

/-- **`SchedKJ` gives `SchedLJ`** at the strategy carried along `bind_rounds`. -/
theorem schedLJ_of_schedKJ : ∀ (n : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (ss : List ℕ)
    (σ : Strategy ((liveRoundsJ dg J n rs).bind (KLJ rootB pts))) (σ' : Strategy (roundsK dg rootB pts n rs)),
    HEq σ σ' → SchedKJ dg rootB pts n rs ss σ' → SchedLJ dg rootB pts n rs ss σ
  | 0, _, _, _, _, _, h => h
  | n + 1, rs, ss, ⟨o, τ⟩, ⟨o', τ'⟩, e, h => by
    have he := heq_send (bind_rounds dg rootB pts (n + 1) rs) e
    have ho : o = o' := he.1
    subst ho
    cases o with
    | none => exact h
    | some s =>
      obtain ⟨ss', h₁, h₂⟩ := h
      exact ⟨ss', h₁, fun c => schedLJ_of_schedKJ n _ ss' (τ c) (τ' c)
        (heq_coin (funext fun c => bind_rounds dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))
          he.2 c) (h₂ c)⟩

theorem schedLJ_of_schedKJ' {rootB' : Array ByteArray} (hB : rootB = rootB') (n : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) (ss : List ℕ) (σ : Strategy ((liveRoundsJ dg J n rs).bind (KLJ rootB pts)))
    (σ' : Strategy (roundsK dg rootB' pts n rs)) (e : HEq σ σ') (h : SchedKJ dg rootB' pts n rs ss σ') :
    SchedLJ dg rootB pts n rs ss σ := by
  subst hB
  exact schedLJ_of_schedKJ dg rootB pts n rs ss σ σ' e h

/-- A bytes prover whose next submission is `s` keeps `SchedKJ` for one more round when `s` is on the schedule's next
stream and it keeps `SchedKJ` after each of the round's coins. -/
theorem schedKJ_some {nMax : ℕ} (w : Wire nMax J mPts) (H : Claims.Msg → Claims.Msg) {Ω : Type}
    (Q : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (n : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (ss : List ℕ)
    (hist : List (Claims.Msg × Claims.Msg)) (a : Claims.Msg) (o : Option (SubmitJ J))
    (hle : ∀ s, o = some s → s.n ≤ nMax) (s : SubmitJ J) (ho : o = some s) (ss' : List ℕ)
    (h₁ : ss = (2 * s.tab.val + s.rep.val) :: ss')
    (h₂ : ∀ c, SchedKJ dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)) ss'
      (toStrat _ (roundsIf w dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))
        (stratOf H Q x ω _ _ (hist ++ [(a, (rdIf w s.n (hle s ho)).enc c)]) none))) :
    SchedKJ dg rootB pts (n + 1) rs ss
      ⟨o, toStrat _ (subIf w rootB pts rs
          (fun s c => roundsK dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))
          (fun s c => roundsIf w dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))) o hle)
        (stratOf H Q x ω _ (subIf w rootB pts rs
          (fun s c => roundsK dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))
          (fun s c => roundsIf w dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))) o hle)
          hist (some a))⟩ := by
  subst ho
  exact ⟨ss', h₁, h₂⟩

end Sched

/-! ## The masked simulator -/

namespace Gen

variable [S : ZkSetup]

section Zk

variable {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} {J : ℕ} (hJ : sts.size ≤ J) (h0 : 0 < sts.size)
  (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (u : zkTape x)

include hJ h0 in
/-- **The simulator's submission after `k + 1` messages**, `k < zkN x`, is on stream `zkStream (k + 1)`. -/
theorem zkSim_sub (hist : List (Claims.Msg × Claims.Msg)) (k : ℕ) (hl : hist.length = k + 1)
    (hk : k + 1 ≤ zkN x) :
    ∃ s : SubmitJ J, (zkW J spec hw x).sub (((zkSim P x spec sts).next x hist (ω, u)).eval sha512).1 = some s ∧
      2 * s.tab.val + s.rep.val = zkStream spec sts (k + 1) := by
  have hlt := zkStream_lt (spec := spec) h0 (k + 1)
  have e : (zkW J spec hw x).sub (((zkSim P x spec sts).next x hist (ω, u)).eval sha512).1 =
      some ⟨⟨zkStream spec sts (k + 1) / 2, by omega⟩,
        ⟨zkStream spec sts (k + 1) % 2, Nat.mod_lt _ (by norm_num)⟩,
        ⟨(zkDepMsg sha512 P x ω (Claims.coinSeq (zkSimCoins x spec sts u hist)) (k + 1)).tail.toArray⟩,
        zkCnt x spec sts (k + 1)⟩ := by
    show zkSub J _ = _
    simp only [zkSim, Claims.OC.eval]
    unfold zkMsg
    rw [hl, ite_eq_right (Nat.succ_ne_zero k), ite_eq_left hk]
    exact zkSub_round J _ _ _ _ (zkCnt_le x spec sts (Nat.succ_ne_zero k))
  exact ⟨_, e, by dsimp only; omega⟩

include hJ h0 in
/-- **The simulator's bytes prover keeps `SchedKJ`** after `k + 1` messages, on the streams of the deployed rounds from
`k + 1` on. -/
theorem schedKJ_zk (rootB : Array ByteArray) (pts : Fin 2 → Fin spec.mPts → 𝔽)
    (hS : (zkStreams spec sts).length = zkN x) :
    ∀ (n : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (hist : List (Claims.Msg × Claims.Msg)) (k : ℕ),
      hist.length = k + 1 → k + n = zkN x →
      SchedKJ spec.dg rootB pts n rs ((zkStreams spec sts).drop k)
        (toStrat _ (roundsIf (zkW J spec hw x) spec.dg rootB pts n rs)
          (stratOf sha512 (zkSim P x spec sts) x (ω, u) _ _ hist none))
  | 0, _, _, k, _, hk => by
    show (zkStreams spec sts).drop k = []
    exact List.drop_eq_nil_of_le (by omega)
  | n + 1, rs, hist, k, hl, hk => by
    obtain ⟨s, ho, hrep⟩ := zkSim_sub hJ h0 hw P ω u hist k hl (by omega)
    refine schedKJ_some spec.dg rootB pts (zkW J spec hw x) sha512 (zkSim P x spec sts) x (ω, u) n rs _ hist
      (((zkSim P x spec sts).next x hist (ω, u)).eval sha512).1 _ ((zkW J spec hw x).sub_le _) s ho
      ((zkStreams spec sts).drop (k + 1)) ?_ fun c => ?_
    · rw [hrep, zkStream, Nat.add_sub_cancel,
        List.getD_eq_getElem (l := zkStreams spec sts) (n := k) (d := 0) (by omega)]
      exact List.drop_eq_getElem_cons (by omega)
    · exact schedKJ_zk rootB pts hS n _ _ (k + 1) (by simp [hl]) (by omega)

include hJ h0 in
/-- **The masked simulator's live strategy submits its rounds on `zkStreams`** at every link point. -/
theorem zkLive_sched (pts : Fin 2 → Fin spec.mPts → 𝔽) (hS : (zkStreams spec sts).length = zkN x) :
    SchedLJ spec.dg (liveStrat (fun _ => J) (fun _ => spec.mPts) zkN spec.dg (zkW J spec hw) sha512
        (zkSim P x spec sts) x (ω, u)).1 pts (zkN x) (fun _ _ => []) (zkStreams spec sts)
      ((liveStrat (fun _ => J) (fun _ => spec.mPts) zkN spec.dg (zkW J spec hw) sha512
        (zkSim P x spec sts) x (ω, u)).2 pts) :=
  schedLJ_of_schedKJ' spec.dg _ pts
    (liveStrat_fst (fun _ => J) (fun _ => spec.mPts) zkN spec.dg (zkW J spec hw) sha512
      (zkSim P x spec sts) x (ω, u)) (zkN x) _ _ _ _
    (liveStrat_snd_heq (fun _ => J) (fun _ => spec.mPts) zkN spec.dg (zkW J spec hw) sha512
      (zkSim P x spec sts) x (ω, u) pts)
    (schedKJ_zk hJ h0 hw P ω u _ pts hS (zkN x) _ _ 0 rfl (Nat.zero_add _))

end Zk

end Gen

section Zk

variable {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} {J : ℕ} (hJ : sts.size ≤ J) (h0 : 0 < sts.size)
  (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (u : zkTape x)

include hJ h0 in
/-- **The simulator's submission after `k + 1` messages**, `k < zkN x`, is on stream `zkStream (k + 1)`. -/
theorem zkSim_sub (hist : List (Claims.Msg × Claims.Msg)) (k : ℕ) (hl : hist.length = k + 1)
    (hk : k + 1 ≤ zkN x) :
    ∃ s : SubmitJ J, (zkW J spec hw x).sub (((zkSim P x spec sts).next x hist (ω, u)).eval sha512).1 = some s ∧
      2 * s.tab.val + s.rep.val = zkStream spec sts (k + 1) :=
  letI : ZkSetup := S₀
  Gen.zkSim_sub hJ h0 hw P ω u hist k hl hk

include hJ h0 in
/-- **The simulator's bytes prover keeps `SchedKJ`** after `k + 1` messages, on the streams of the deployed rounds from
`k + 1` on. -/
theorem schedKJ_zk (rootB : Array ByteArray) (pts : Fin 2 → Fin spec.mPts → 𝔽)
    (hS : (zkStreams spec sts).length = zkN x) :
    ∀ (n : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (hist : List (Claims.Msg × Claims.Msg)) (k : ℕ),
      hist.length = k + 1 → k + n = zkN x →
      SchedKJ spec.dg rootB pts n rs ((zkStreams spec sts).drop k)
        (toStrat _ (roundsIf (zkW J spec hw x) spec.dg rootB pts n rs)
          (stratOf sha512 (zkSim P x spec sts) x (ω, u) _ _ hist none)) :=
  letI : ZkSetup := S₀
  Gen.schedKJ_zk hJ h0 hw P ω u rootB pts hS

include hJ h0 in
/-- **The masked simulator's live strategy submits its rounds on `zkStreams`** at every link point. -/
theorem zkLive_sched (pts : Fin 2 → Fin spec.mPts → 𝔽) (hS : (zkStreams spec sts).length = zkN x) :
    SchedLJ spec.dg (liveStrat (fun _ => J) (fun _ => spec.mPts) zkN spec.dg (zkW J spec hw) sha512
        (zkSim P x spec sts) x (ω, u)).1 pts (zkN x) (fun _ _ => []) (zkStreams spec sts)
      ((liveStrat (fun _ => J) (fun _ => spec.mPts) zkN spec.dg (zkW J spec hw) sha512
        (zkSim P x spec sts) x (ω, u)).2 pts) :=
  letI : ZkSetup := S₀
  Gen.zkLive_sched hJ h0 hw P ω u pts hS

end Zk

/-! ## The streams of the deployed rounds -/

/-- **A masked rep's rounds are `repRounds`** at the `--zk` schedule. -/
theorem zkRepNsOf_length {spec : SessionSpec} {st : Stmt} {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule st.m = .ok (sch, tPad)) : (zkRepNsOf spec st).length = repRounds true spec st := by
  obtain ⟨h25, -, -, -, -, ⟨-, h2, -⟩, -⟩ := ZkExec.zk_schedule_seams _ _ _ hz
  rw [repRounds_zk hz]
  unfold zkRepNsOf zkPre zkLig
  rw [hz]
  obtain ⟨m, hm⟩ : ∃ m, sch.levels.size - 1 = m + 1 := ⟨sch.levels.size - 2, by omega⟩
  simp only [zkRepNs, zcNs, lcNs, rsNs, zkLigNs, innerNs, qNs, List.length_append, List.length_cons,
    List.length_nil, List.length_replicate]
  rw [hm, loopNs_length sch m 0 _ (by omega)]
  omega

theorem flatMap_fromS (Rf : ℕ → ℕ) : ∀ t d,
    (List.range' (2 * t) (2 * d)).flatMap (fun k => List.replicate (Rf (k / 2)) k) = fromS Rf t d
  | _, 0 => rfl
  | t, d + 1 => by
    rw [show 2 * (d + 1) = 2 * d + 1 + 1 by omega, List.range'_succ, List.range'_succ, List.flatMap_cons,
      List.flatMap_cons, show 2 * t + 1 + 1 = 2 * (t + 1) by omega, flatMap_fromS Rf (t + 1) d, fromS,
      show 2 * t / 2 = t by omega, show (2 * t + 1) / 2 = t by omega]

theorem length_fromS (Rf : ℕ → ℕ) : ∀ t d, (fromS Rf t d).length = (nsFrom Rf t d).sum
  | _, 0 => rfl
  | t, d + 1 => by
    simp only [fromS, nsFrom, List.length_append, List.length_replicate, List.sum_cons, length_fromS Rf (t + 1) d]
    omega

theorem sum_nsFrom (Rf : ℕ → ℕ) : ∀ t d, (nsFrom Rf t d).sum = ((List.range' t d).map fun i => 2 * Rf i).sum
  | _, 0 => rfl
  | t, d + 1 => by
    rw [nsFrom, List.sum_cons, sum_nsFrom Rf (t + 1) d, List.range'_succ, List.map_cons, List.sum_cons]
    omega

theorem fromS_add (Rf : ℕ → ℕ) : ∀ t a b, fromS Rf t (a + b) = fromS Rf t a ++ fromS Rf (t + a) b
  | t, 0, b => by simp [fromS]
  | t, a + 1, b => by
    rw [show a + 1 + b = a + b + 1 by omega, fromS, fromS, fromS_add Rf (t + 1) a b,
      show t + 1 + a = t + (a + 1) by omega]
    simp only [List.append_assoc]

theorem mem_fromS (Rf : ℕ → ℕ) : ∀ t d p, p ∈ fromS Rf t d → 2 * t ≤ p
  | _, 0, _, h => by simp [fromS] at h
  | t, d + 1, p, h => by
    simp only [fromS, List.mem_append, List.mem_replicate] at h
    rcases h with ⟨-, rfl⟩ | ⟨-, rfl⟩ | h
    · omega
    · omega
    · have := mem_fromS Rf (t + 1) d p h
      omega

section Deploy

variable {spec : SessionSpec} {sts : Array Stmt}
  (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))

include hz in
/-- **The streams of the deployed rounds** are each table's rep 0 `repRounds` times, then its rep 1, table by table. -/
theorem zkStreams_eq : zkStreams spec sts = fromS (fun t => repRounds true spec sts[t]!) 0 sts.size := by
  rw [← flatMap_fromS, zkStreams, List.range_eq_range', Nat.mul_zero]
  refine List.flatMap_congr fun k hk => ?_
  rw [List.mem_range'_1] at hk
  have ht : k / 2 < sts.size := by omega
  rw [getElem!_pos sts _ ht, zkRepNsOf_length (hz _ ht)]

namespace Gen

variable [S : ZkSetup]

include hz in
/-- **There are as many streams as deployed rounds.** -/
theorem zkStreams_length {x : Claims.Msg} {ins : Ins} (hset : setupOf true x = some (ins, spec, sts)) :
    (zkStreams spec sts).length = zkN x := by
  rw [zkStreams_eq hz, length_fromS, sum_nsFrom, zkN, zkK hset, rounds_sum, List.range_eq_range']
  omega

end Gen

include hz in
/-- **There are as many streams as deployed rounds.** -/
theorem zkStreams_length {x : Claims.Msg} {ins : Ins} (hset : setupOf true x = some (ins, spec, sts)) :
    (zkStreams spec sts).length = zkN x :=
  Gen.zkStreams_length (S := S₀) (x := x) (ins := ins) (hset := hset) (hz := hz)

include hz in
/-- **The streams split at table `j`**: the tables in front's, table `j`'s rep 0, its rep 1, then the tables after. -/
theorem zkStreams_split {j : ℕ} (hj : j < sts.size) :
    zkStreams spec sts = fromS (fun t => repRounds true spec sts[t]!) 0 j ++
      (List.replicate (repRounds true spec sts[j]!) (2 * j) ++
        (List.replicate (repRounds true spec sts[j]!) (2 * j + 1) ++
          fromS (fun t => repRounds true spec sts[t]!) (j + 1) (sts.size - (j + 1)))) := by
  rw [zkStreams_eq hz]
  conv_lhs => rw [show sts.size = j + (1 + (sts.size - (j + 1))) by omega]
  rw [fromS_add, fromS_add, Nat.zero_add]
  simp only [fromS, List.append_nil, List.append_assoc]

end Deploy

end FlockDeploy.KS
