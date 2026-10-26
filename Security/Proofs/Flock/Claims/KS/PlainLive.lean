import Proofs.Flock.Claims.KS.PlainLead
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# The plain simulator's live strategy submits rep 0's rounds first

The plain simulator sends deployed round `k + 1` as a round on rep `plainRep (k + 1)`, rep 0's for the first
`|plainRepNs|` rounds and rep 1's after (`plainSub_round`), and `|plainRepNs| = repRounds` wherever `fast100` has at
least two levels. So its live strategy, read through `liveSession_eq` as its bytes prover on the rounds' explicit game
(`roundsK`), submits `repRounds` rounds on rep 0 first:

* `SchedK`, `schedL_of_schedK`: the schedule on `roundsK`'s strategies, carried to `liveSession`'s across `bind_rounds`;
* `schedK_plain`, `plainLive_sched`: the plain simulator's bytes prover keeps it, round by round;
* `plainRepNs_length`: `|plainRepNs| = repRounds` at two levels or more;
* `plainSchedAlign`: so `PlainSchedAlign` holds at every statement.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

section Casts

variable {β : Type}

/-- Equal strategies of equal games sending a message send the same message, and agree after it. -/
theorem heq_send {M : Type} {k k' : M → Game β} (e : Game.send M k = Game.send M k') {s : Strategy (Game.send M k)}
    {s' : Strategy (Game.send M k')} (h : HEq s s') : s.1 = s'.1 ∧ HEq s.2 s'.2 := by
  obtain rfl : k = k' := eq_of_heq (Game.send.inj e).2
  obtain rfl := eq_of_heq h
  exact ⟨rfl, HEq.rfl⟩

/-- Equal strategies of equal games drawing a coin agree at each coin. -/
theorem heq_coin {C : Type} {iF iF' : Fintype C} {iN : Nonempty C} {k k' : C → Game β} (hk : k = k')
    {s : Strategy (@Game.coin β C iF iN k)} {s' : Strategy (@Game.coin β C iF' iN k')} (h : HEq s s') (c : C) :
    HEq (s c) (s' c) := by
  obtain rfl : iF = iF' := Subsingleton.elim _ _
  subst hk
  obtain rfl := eq_of_heq h
  rfl

end Casts

/-- `liveStrat` at the link points `pts` is the bytes prover's strategy of the session's explicit game there. -/
theorem liveStrat_snd_heq {nMax : ℕ} (J mPts N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Claims.Msg) → Wire nMax (J x) (mPts x)) (H : Claims.Msg → Claims.Msg) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) (ω : Ω) (pts : Fin 2 → Fin (mPts x) → 𝔽) :
    HEq ((liveStrat J mPts N dg w H P x ω).2 pts)
      ((stratG H P (fun x => sessionK dg (J x) (mPts x) (N x)) (fun x => sessionIf (w x) dg (N x)) x ω).2 pts) := by
  have h : HEq (liveStrat J mPts N dg w H P x ω)
      (stratG H P (fun x => sessionK dg (J x) (mPts x) (N x)) (fun x => sessionIf (w x) dg (N x)) x ω) := by
    unfold liveStrat
    exact eqRec_heq_self _ _
  exact (heq_snd_coin (liveSession_eq dg (J x) (mPts x) (N x)).symm h.symm pts).symm

/-! ## The schedule on the rounds' explicit game -/

section Sched

variable {mPts : ℕ} (dg : ByteArray → ByteArray) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)

/-- **`SchedL` on the rounds' explicit game** (`roundsK`): the next `r₀` rounds are rep 0's, and the rest rep 1's. -/
def SchedK : (n : ℕ) → (rs : Fin 1 → Fin 2 → List Flock.Round) → ℕ → Strategy (roundsK dg rootB pts n rs) → Prop
  | 0, _, r₀, _ => r₀ = 0
  | n + 1, rs, r₀, σ => match σ with
    | ⟨none, _⟩ => r₀ = 0
    | ⟨some s, τ⟩ => s.rep = (if r₀ = 0 then 1 else 0) ∧
        ∀ c, SchedK n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)) (r₀ - 1) (τ c)

/-- **`SchedK` gives `SchedL`** at the strategy carried along `bind_rounds`. -/
theorem schedL_of_schedK : ∀ (n : ℕ) (rs : Fin 1 → Fin 2 → List Flock.Round) (r₀ : ℕ)
    (σ : Strategy ((liveRoundsJ dg 1 n rs).bind (KLJ rootB pts))) (σ' : Strategy (roundsK dg rootB pts n rs)),
    HEq σ σ' → SchedK dg rootB pts n rs r₀ σ' → SchedL dg rootB pts n rs r₀ σ
  | 0, _, _, _, _, _, h => h
  | n + 1, rs, r₀, ⟨o, τ⟩, ⟨o', τ'⟩, e, h => by
    have he := heq_send (bind_rounds dg rootB pts (n + 1) rs) e
    have ho : o = o' := he.1
    subst ho
    cases o with
    | none => exact h
    | some s =>
      obtain ⟨h₁, h₂⟩ := h
      exact ⟨h₁, fun c => schedL_of_schedK n _ (r₀ - 1) (τ c) (τ' c)
        (heq_coin (funext fun c => bind_rounds dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))
          he.2 c) (h₂ c)⟩

theorem schedL_of_schedK' {rootB' : Array ByteArray} (hB : rootB = rootB') (n : ℕ)
    (rs : Fin 1 → Fin 2 → List Flock.Round) (r₀ : ℕ) (σ : Strategy ((liveRoundsJ dg 1 n rs).bind (KLJ rootB pts)))
    (σ' : Strategy (roundsK dg rootB' pts n rs)) (e : HEq σ σ') (h : SchedK dg rootB' pts n rs r₀ σ') :
    SchedL dg rootB pts n rs r₀ σ := by
  subst hB
  exact schedL_of_schedK dg rootB pts n rs r₀ σ σ' e h

/-- A bytes prover whose next submission is `s` keeps `SchedK` for one more round when `s` is on the scheduled rep and
it keeps `SchedK` after each of the round's coins. -/
theorem schedK_some {nMax : ℕ} (w : Wire nMax 1 mPts) (H : Claims.Msg → Claims.Msg) {Ω : Type}
    (Q : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (n : ℕ) (rs : Fin 1 → Fin 2 → List Flock.Round) (r₀ : ℕ)
    (hist : List (Claims.Msg × Claims.Msg)) (a : Claims.Msg) (o : Option (SubmitJ 1))
    (hle : ∀ s, o = some s → s.n ≤ nMax) (s : SubmitJ 1) (ho : o = some s)
    (h₁ : s.rep = if r₀ = 0 then 1 else 0)
    (h₂ : ∀ c, SchedK dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)) (r₀ - 1)
      (toStrat _ (roundsIf w dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))
        (stratOf H Q x ω _ _ (hist ++ [(a, (rdIf w s.n (hle s ho)).enc c)]) none))) :
    SchedK dg rootB pts (n + 1) rs r₀
      ⟨o, toStrat _ (subIf w rootB pts rs
          (fun s c => roundsK dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))
          (fun s c => roundsIf w dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))) o hle)
        (stratOf H Q x ω _ (subIf w rootB pts rs
          (fun s c => roundsK dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))
          (fun s c => roundsIf w dg rootB pts n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))) o hle)
          hist (some a))⟩ := by
  subst ho
  exact ⟨h₁, h₂⟩

end Sched

/-! ## The plain simulator -/

section Plain

variable (keep : Bool → Bool) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
  (P : Claims.Prover Ω) (ω : Ω) (u : plainTape x)

/-- **The simulator's submission after `k + 1` messages**, `k < plainN x`, is on rep `plainRep (k + 1)`. -/
theorem plainSim_subK (hist : List (Claims.Msg × Claims.Msg)) (k : ℕ) (hl : hist.length = k + 1)
    (hk : k + 1 ≤ plainN x) :
    ∃ s : SubmitJ 1, (plainW spec hw x).sub (((plainSimK keep P x spec st).next x hist (ω, u)).eval sha512).1 = some s ∧
      s.rep.val = plainRep spec st (k + 1) := by
  have e : (plainW spec hw x).sub (((plainSimK keep P x spec st).next x hist (ω, u)).eval sha512).1 =
      some ⟨0, ⟨plainRep spec st (k + 1), plainRep_lt spec st (k + 1)⟩,
        ⟨(depMsg sha512 P x ω (Claims.coinSeq (simCoins x spec st u hist)) (k + 1)).tail.toArray⟩,
        plainCnt x spec st (k + 1)⟩ := by
    show plainSub _ = _
    simp only [plainSimK, Claims.OC.eval]
    unfold plainMsgK
    rw [hl, ite_eq_right (Nat.succ_ne_zero k), ite_eq_left hk]
    exact plainSub_round _ _ _ (plainRep_lt spec st (k + 1)) (plainCnt_le x spec st (Nat.succ_ne_zero k))
  exact ⟨_, e, rfl⟩

/-- `plainSim_subK` at the deployed plain verifier (`keep := id`). -/
theorem plainSim_sub (hist : List (Claims.Msg × Claims.Msg)) (k : ℕ) (hl : hist.length = k + 1)
    (hk : k + 1 ≤ plainN x) :
    ∃ s : SubmitJ 1, (plainW spec hw x).sub (((plainSim P x spec st).next x hist (ω, u)).eval sha512).1 = some s ∧
      s.rep.val = plainRep spec st (k + 1) :=
  plainSim_subK id (x := x) (spec := spec) (st := st) (hw := hw) (P := P) (ω := ω) (u := u) (hist := hist) (k := k) (hl := hl) (hk := hk)

/-- **The simulator's bytes prover keeps `SchedK`** after `k + 1` messages, with `|plainRepNs| - k` rounds of rep 0
left. -/
theorem schedK_plainK (rootB : Array ByteArray) (pts : Fin 2 → Fin spec.mPts → 𝔽)
    (hR : (plainRepNs spec st).length ≤ plainN x) :
    ∀ (n : ℕ) (rs : Fin 1 → Fin 2 → List Flock.Round) (hist : List (Claims.Msg × Claims.Msg)) (k : ℕ),
      hist.length = k + 1 → k + n = plainN x →
      SchedK (plainDg spec) rootB pts n rs ((plainRepNs spec st).length - k)
        (toStrat _ (roundsIf (plainW spec hw x) (plainDg spec) rootB pts n rs)
          (stratOf sha512 (plainSimK keep P x spec st) x (ω, u) _ _ hist none))
  | 0, _, _, k, _, hk => by
    show (plainRepNs spec st).length - k = 0
    omega
  | n + 1, rs, hist, k, hl, hk => by
    obtain ⟨s, ho, hrep⟩ := plainSim_subK keep x spec st hw P ω u hist k hl (by omega)
    refine schedK_some (plainDg spec) rootB pts (plainW spec hw x) sha512 (plainSimK keep P x spec st) x (ω, u) n rs _ hist
      (((plainSimK keep P x spec st).next x hist (ω, u)).eval sha512).1 _ ((plainW spec hw x).sub_le _) s ho ?_
      fun c => ?_
    · apply Fin.ext
      rw [hrep]
      unfold plainRep
      by_cases hkR : k < (plainRepNs spec st).length
      · rw [ite_eq_left (show k + 1 - 1 < (plainRepNs spec st).length by omega),
          ite_eq_right (show (plainRepNs spec st).length - k ≠ 0 by omega)]
        rfl
      · rw [ite_eq_right (show ¬ k + 1 - 1 < (plainRepNs spec st).length by omega),
          ite_eq_left (show (plainRepNs spec st).length - k = 0 by omega)]
        rfl
    · rw [show (plainRepNs spec st).length - k - 1 = (plainRepNs spec st).length - (k + 1) by omega]
      exact schedK_plainK rootB pts hR n _ _ (k + 1) (by simp [hl]) (by omega)

/-- `schedK_plainK` at the deployed plain verifier (`keep := id`). -/
theorem schedK_plain (rootB : Array ByteArray) (pts : Fin 2 → Fin spec.mPts → 𝔽)
    (hR : (plainRepNs spec st).length ≤ plainN x) :
    ∀ (n : ℕ) (rs : Fin 1 → Fin 2 → List Flock.Round) (hist : List (Claims.Msg × Claims.Msg)) (k : ℕ),
      hist.length = k + 1 → k + n = plainN x →
      SchedK (plainDg spec) rootB pts n rs ((plainRepNs spec st).length - k)
        (toStrat _ (roundsIf (plainW spec hw x) (plainDg spec) rootB pts n rs)
          (stratOf sha512 (plainSim P x spec st) x (ω, u) _ _ hist none)) :=
  schedK_plainK id (x := x) (spec := spec) (st := st) (hw := hw) (P := P) (ω := ω) (u := u) (rootB := rootB) (pts := pts) (hR := hR)

/-- **The plain simulator's live strategy submits `|plainRepNs|` rounds on rep 0 first** at every link point. -/
theorem plainLive_schedK (pts : Fin 2 → Fin spec.mPts → 𝔽) (hR : (plainRepNs spec st).length ≤ plainN x) :
    SchedL (plainDg spec) (plainLiveK keep x spec st hw P ω u).1 pts (plainN x) (fun _ _ => [])
      (plainRepNs spec st).length ((plainLiveK keep x spec st hw P ω u).2 pts) :=
  schedL_of_schedK' (plainDg spec) _ pts
    (liveStrat_fst (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSimK keep P x spec st) x (ω, u)) (plainN x) _ _ _ _
    (liveStrat_snd_heq (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSimK keep P x spec st) x (ω, u) pts)
    (schedK_plainK keep x spec st hw P ω u _ pts hR (plainN x) _ _ 0 rfl (Nat.zero_add _))

/-- `plainLive_schedK` at the deployed plain verifier (`keep := id`). -/
theorem plainLive_sched (pts : Fin 2 → Fin spec.mPts → 𝔽) (hR : (plainRepNs spec st).length ≤ plainN x) :
    SchedL (plainDg spec) (plainLive x spec st hw P ω u).1 pts (plainN x) (fun _ _ => [])
      (plainRepNs spec st).length ((plainLive x spec st hw P ω u).2 pts) :=
  plainLive_schedK id (x := x) (spec := spec) (st := st) (hw := hw) (P := P) (ω := ω) (u := u) (pts := pts) (hR := hR)

end Plain

/-! ## A plain rep's rounds -/

theorem loopNs_length (sch : Flock.Schedule) : ∀ (m a y : ℕ), a + m + 2 = sch.levels.size →
    (loopNs (levelNs sch) (levelDim sch) a (m + 1) y).length = 11 * m + 7
  | 0, a, y, h => by
    simp only [loopNs, levelNs, List.length_append, List.length_nil]
    rw [ite_eq_right (by omega)]
    simp [qNs]
  | m + 1, a, y, h => by
    rw [loopNs, List.length_append, loopNs_length sch m (a + 1) _ (by omega), levelNs, ite_eq_left (by omega)]
    simp only [qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
    omega

theorem ligNs_length (sch : Flock.Schedule) (h2 : 2 ≤ sch.levels.size) :
    (ligNs sch).length = sch.k0 + 5 + 11 * (sch.levels.size - 1) := by
  obtain ⟨m, hm⟩ : ∃ m, sch.levels.size - 1 = m + 1 := ⟨sch.levels.size - 2, by omega⟩
  simp only [ligNs, qNs, List.length_append, List.length_cons, List.length_nil, List.length_replicate]
  rw [hm, loopNs_length sch m 0 _ (by omega)]
  omega

/-- **A plain rep's rounds are `repRounds`** where `fast100` has at least two levels. -/
theorem plainRepNs_length {spec : SessionSpec} {st : Stmt} {s : Flock.Schedule} (hs : fast100 st.m = some s)
    (h2 : 2 ≤ s.levels.size) : (plainRepNs spec st).length = repRounds false spec st := by
  rw [repRounds_plain hs, plainRepNs, Claims.Flock.plainSched, hs, Option.getD_some]
  simp only [repNs, zcNs, lcNs, rsNs, preLig, List.length_append, List.length_cons, List.length_nil,
    List.length_replicate]
  rw [ligNs_length s h2]
  omega

/-! ## At the deployed plain verifier -/

/-- **The plain simulator's live strategy submits `repRounds` rounds on rep 0 first** at a plain statement of one
table whose schedule has at least two levels. -/
theorem plainLive_schedLK (keep : Bool → Bool) {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) (ht : tables false x = [(spec, st)])
    {sch : Accounting.Schedule} (hsch : Accounting.fast100 st.m = some sch) (h2 : 2 ≤ sch.levels.length)
    (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (u : plainTape x)
    (pts : Fin 2 → Fin spec.mPts → 𝔽) :
    SchedL (plainDg spec) (plainLiveK keep x spec st hw P ω u).1 pts (plainN x) (fun _ _ => []) (repRounds false spec st)
      ((plainLiveK keep x spec st hw P ω u).2 pts) := by
  obtain ⟨sF, hF, rfl⟩ := exec_sched hsch
  simp only [FlockSoundness.Refine.schedOf, List.length_map, Array.length_toList] at h2
  obtain ⟨s', sts', hs', hst⟩ := setupOf_head (zk := false) (show (tables false x).head? = some (spec, st) by
    rw [ht]; rfl)
  rw [hset] at hs'
  simp only [Option.some.injEq, Prod.mk.injEq] at hs'
  obtain ⟨-, -, rfl⟩ := hs'
  have hle := rep_le_rounds (zk := false) (spec := spec) hst
  have hk : (V false).k x = 2 + rounds false spec sts := by simp only [V, hset]
  have hlen := plainRepNs_length (spec := spec) hF h2
  rw [← hlen]
  exact plainLive_schedK keep x spec st hw P ω u pts (by unfold plainN; omega)

/-- `plainLive_schedLK` at the deployed plain verifier (`keep := id`). -/
theorem plainLive_schedL {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) (ht : tables false x = [(spec, st)])
    {sch : Accounting.Schedule} (hsch : Accounting.fast100 st.m = some sch) (h2 : 2 ≤ sch.levels.length)
    (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (u : plainTape x)
    (pts : Fin 2 → Fin spec.mPts → 𝔽) :
    SchedL (plainDg spec) (plainLive x spec st hw P ω u).1 pts (plainN x) (fun _ _ => []) (repRounds false spec st)
      ((plainLive x spec st hw P ω u).2 pts) :=
  plainLive_schedLK id (x := x) (ins := ins) (spec := spec) (sts := sts) (st := st) (hset := hset) (ht := ht) (sch := sch) (hsch := hsch) (h2 := h2) (hw := hw) (Ω := Ω) (P := P) (ω := ω) (u := u) (pts := pts)

/-- **`PlainSchedAlign` holds at every statement.** -/
theorem plainSchedAlign (x : Claims.Msg) : PlainSchedAlign x := by
  intro s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω
  have hS := fun u => plainLive_schedL hset ht hsch (by simp) (plain_mPts hs hin hset).1 P ω u
  refine ⟨fun hh C u => ?_, fun hh hl C u => ?_⟩
  · exact leadsAfter_plainSimAt L m k0 l₀ l₁ rest ext hsch hm hlay ins.read.I d x
      (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
      (plain_hdecAt hs hin hset ht hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C u (hS u)
  · exact leadsAfter_plainSimAt L m k0 l₀ l₁ rest ext hsch hm hlay ins.read.I d x
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
      (plain_hdecHAt hs hin hset ht hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C u
      (hS u)

end FlockDeploy.KS
