import Proofs.Flock.Claims.ZK.FirstRefusal

/-!
# The masked session's rounds after a refusal its coins show

`flock-circuit --zk` sends nothing more once it refuses: at an outer coin `1` as soon as it is answered
(`honestPR_stop`), and at `β = 0`, a dependent pair `ρ₀, ρ₁` or a rank-deficient mask slot once `sessionPZk` throws,
right after the round that answers the refused coin's last part. A round plan sees each refusal once every coin it
reads is answered, so its encoding can send `[]` from there on.

* `rankSlot`: table `j`'s rep `r`'s rank check's round, after its zerocheck's and lincheck's `m + k_log − 6` rounds.
  `RankBad`: that check refuses at the opened coins.
* `Thrown x c i`: a throw whose coins are all below round `i` refuses at `c`: a rank check (rep `0`'s and rep `r`'s
  check rounds both reached), a pair `ρ₀, ρ₁` (both answered) or a `β` (answered). `Refused`: that, or an answered outer
  coin `1`. `refused_congr`: `Refused x c i` reads the coins below `i` only; `refused_zero`: round `0` sees none.
* `encStop enc`: `[]` at a round where a refusal is seen, `enc`'s message elsewhere. `causal_encStop`: causal where
  `enc` is.
* `outerHit_of_lt`: an answered outer coin `1` is an outer hit, after any number of openings.
* `honestPreOnPos_stop`: `HonestPreOnPos` at `encStop enc`, from `enc` at the rounds where no refusal is seen and from a
  seen throw stopping `honestPR`.
* `Flock.zk_deployZk_masked_exe_stopped_gap`: `Flock.zk_deployZk_masked_exe_refusals_gap` at `encStop enc`, its rounds
  after the first asked only where no refusal is seen, given that a seen throw stops `honestPR`.
* `honestPR_thrown`: a seen throw does stop `honestPR`. The session on the mask witness stops right after the send whose
  reply it refused (`sessionPZk_rho_stop`, `sessionPZk_beta_stop`, `sessionPZk_rank_stop`), within the openings'
  answers, so the prover sends `[]` (`honestPR_stops`). `slotAt_padOps_lt` and `ptsAt_open` read the opened coins as
  those answers. `Flock.zk_deployZk_masked_exe_live_gap`: `Flock.zk_deployZk_masked_exe_stopped_gap` without that
  premise (`thrown_stops`).
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **Table `j`'s rep `r`'s rank check's round**: after its zerocheck's and lincheck's `m + k_log − 6` rounds from its
first (`repBase`), the round whose message the check guards. -/
def rankSlot (x : Msg) (j r : ℕ) : ℕ := repBase x (2 * j + r) + ((stsOf x)[j]!.m + (stsOf x)[j]!.c.kLog - 6)

/-- **Table `j`'s rep `r`'s rank check refuses at the opened coins `c`** (`rankCheckP`): its claim points, after rep
`0`'s at rep `1` (`ptsOpen`), miss full rank over its mask slot's words. -/
def RankBad {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j r : ℕ) : Prop :=
  FlockComplete.MaskRank.maskRankOk ((if r = 1 then ptsOpen c (stsOf x)[j]! (2 * j) else #[]) ++
    ptsOpen c (stsOf x)[j]! (2 * j + r)) (Composed.maskWordsOf (stsOf x)[j]!) = false

/-- **A throw that the coins below round `i` show** at the opened coins `c`: at some table `j`'s rep `r`, the rank check
with every claim point answered (rep `0`'s and rep `r`'s check rounds reached), the pair `ρ₀, ρ₁` both answered, or `β`
answered, refuses. -/
def Thrown (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (i : ℕ) : Prop :=
  ∃ j < (stsOf x).size, ∃ r < 2,
    (14 ≤ (stsOf x)[j]!.m ∧ 6 ≤ (stsOf x)[j]!.c.kLog ∧ rankSlot x j 0 ≤ i ∧ rankSlot x j r ≤ i ∧ RankBad c j r) ∨
    (rhoSlot x j r + 1 < i ∧ _root_.Flock.Zk.rhoIndependent (rhoAt c j r) = false) ∨
    (betaSlot x j r < i ∧ (betaAt c j r).isZero = true)

/-- **A refusal that the coins below round `i` show**: an answered outer coin `1` (`outerAt`), or a throw
(`Thrown`). -/
def Refused (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (i : ℕ) : Prop :=
  (∃ j < (stsOf x).size, ∃ r < 2, outerSlot x j r < i ∧ ∃ e < (stsOf x)[j]!.m - 13, outerAt c j r e = 1) ∨
    Thrown x c i

theorem slotAt_congr {x : Msg} {c c' : Fin (refK x) → FlockDeploy.Coin} {i : ℕ}
    (h : ∀ k : Fin (refK x), k.val < i → c k = c' k) {k : ℕ} (hk : k < i) (e : ℕ) :
    slotAt c k e = slotAt c' k e := by
  unfold slotAt
  split_ifs with hK
  · rw [h ⟨k, hK⟩ hk]
  · rfl

theorem ptsOpen_congr {x : Msg} {c c' : Fin (refK x) → FlockDeploy.Coin} {i : ℕ}
    (h : ∀ k : Fin (refK x), k.val < i → c k = c' k) {st : _root_.Flock.Stmt} {q : ℕ} (hm : 14 ≤ st.m)
    (hk : 6 ≤ st.c.kLog) (hq : repBase x q + (st.m + st.c.kLog - 6) ≤ i) :
    ptsOpen c st q = ptsOpen c' st q := by
  unfold ptsOpen
  exact FlockComplete.ZkTables.claimPoints_congr_lt (by omega) hk (fun n hn => slotAt_congr h (by omega) _)
    (fun e _ => slotAt_congr h (by omega) _)

theorem refused_of_agree {x : Msg} {c c' : Fin (refK x) → FlockDeploy.Coin} {i : ℕ}
    (h : ∀ k : Fin (refK x), k.val < i → c k = c' k) (hs : Refused x c i) : Refused x c' i := by
  rcases hs with ⟨j, hj, r, hr, ho, e, he, h1⟩ | ⟨j, hj, r, hr, hb⟩
  · exact .inl ⟨j, hj, r, hr, ho, e, he, (slotAt_congr h ho e).symm.trans h1⟩
  · refine .inr ⟨j, hj, r, hr, ?_⟩
    rcases hb with ⟨hm, hk, h0, h1, hbad⟩ | ⟨hρ, hbad⟩ | ⟨hβ, hbad⟩
    · refine .inl ⟨hm, hk, h0, h1, ?_⟩
      have h0' : repBase x (2 * j) + ((stsOf x)[j]!.m + (stsOf x)[j]!.c.kLog - 6) ≤ i := by
        unfold rankSlot at h0
        rwa [Nat.add_zero] at h0
      have e0 := ptsOpen_congr h hm hk h0'
      have e1 := ptsOpen_congr (q := 2 * j + r) h hm hk h1
      unfold RankBad at hbad ⊢
      rw [← e0, ← e1]
      exact hbad
    · refine .inr (.inl ⟨hρ, ?_⟩)
      unfold rhoAt at hbad ⊢
      simp only [slotAt_congr h (show rhoSlot x j r < i by omega),
        slotAt_congr h (show rhoSlot x j r + 1 < i by omega)] at hbad
      exact hbad
    · refine .inr (.inr ⟨hβ, ?_⟩)
      unfold betaAt at hbad ⊢
      rwa [slotAt_congr h hβ] at hbad

/-- **`Refused x c i` reads the coins below `i` only.** -/
theorem refused_congr {x : Msg} {c c' : Fin (refK x) → FlockDeploy.Coin} {i : ℕ}
    (h : ∀ k : Fin (refK x), k.val < i → c k = c' k) : Refused x c i ↔ Refused x c' i :=
  ⟨refused_of_agree h, refused_of_agree fun k hk => (h k hk).symm⟩

/-- **Round `0` sees no refusal**: every refusal's coins are answered after the link. -/
theorem refused_zero {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) : ¬ Refused x c 0 := by
  rintro (⟨j, hj, r, hr, ho, -⟩ | ⟨j, hj, r, hr, ⟨-, -, h0, -⟩ | ⟨hρ, -⟩ | ⟨hβ, -⟩⟩)
  · exact Nat.not_lt_zero _ ho
  · obtain ⟨s, spec, hsx⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
    unfold rankSlot at h0
    rw [repBase_eq hsx] at h0
    unfold FlockComplete.repStartZk at h0
    omega
  · omega
  · omega

open Classical in
/-- **A round encoding that stops where a refusal is seen**: `[]` at a round whose coins below it show a refusal
(`Refused`), `enc`'s message elsewhere. -/
noncomputable def encStop {C : Tabs}
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (C x j).Rest}
    (enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((C x j).T (cz x c j)).tab.Cut) → Msg) :
    (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((C x j).T (cz x c j)).tab.Cut) → Msg :=
  fun x c i cut => if Refused x c i then [] else enc x c i cut

/-- **Stopping where a refusal is seen keeps a plan causal**: `Refused` reads the coins below the round only. -/
theorem causal_encStop {C : Tabs}
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (C x j).Rest}
    {enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((C x j).T (cz x c j)).tab.Cut) → Msg}
    {fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((C x j).T (cz x c j)).tab.View) → Msg}
    (h : (RoundPlan.stop cz enc fin).Causal) : (RoundPlan.stop cz (encStop enc) fin).Causal := by
  intro x wt ω p c c' i hcc
  dsimp only [RoundPlan.stop, encStop]
  by_cases hs : Refused x c i
  · rw [ite_eq_left hs, ite_eq_left ((refused_congr hcc).1 hs)]
  · rw [ite_eq_right hs, ite_eq_right fun h' => hs ((refused_congr hcc).2 h')]
    exact h x wt ω p c c' i hcc

/-- **An answered outer coin `1` is an outer hit**, after any openings `ops` that answer its round: outer coin
`i < m − 13` of table `j`'s rep `r`, read at the opened coins, equal to `1` makes `honestPR` stop (`outerHit`). -/
theorem outerHit_of_lt {x : Msg} (ops : List (Msg × Msg)) {j r i : ℕ} (hj : j < (stsOf x).size) (hr : r < 2)
    (hk : outerSlot x j r < ops.length) (hi : i < (stsOf x)[j]!.m - 13)
    (h1 : outerAt (padOps (x := x) ops) j r i = 1) :
    FlockComplete.ZkTables.outerHit x ((ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd) = true := by
  have h10 : (0 : _root_.Flock.F128) ≠ 1 := by decide
  unfold outerAt at h1
  split_ifs at h1 with hs
  swap
  · exact absurd h1 h10
  have hc : padOps (x := x) ops ⟨outerSlot x j r, hs⟩ = rd ops[outerSlot x j r].2 := by
    unfold padOps
    rw [List.getD_eq_getElem _ _ (by simpa using hk), List.getElem_map]
  rw [hc] at h1
  unfold FlockDeploy.Coin.at at h1
  split_ifs at h1 with hi'
  swap
  · exact absurd h1 h10
  obtain ⟨s, spec, hsx⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  have hsl : outerSlot x j r = FlockComplete.repStartZk spec (stsOf x) (2 * j + r) + 1 := by
    rw [outerSlot_eq hsx, repBase_eq hsx]
  unfold FlockComplete.ZkTables.outerHit
  rw [List.any_eq_true]
  refine ⟨(outerSlot x j r, i), ?_, ?_⟩
  · unfold FlockComplete.outerSlotsZk
    rw [hsx]
    simp only [List.mem_flatMap, List.mem_range, List.mem_map]
    refine ⟨2 * j + r, by omega, i, ?_, ?_⟩
    · rw [show (2 * j + r) / 2 = j by omega]
      exact hi
    · rw [hsl]
  · rw [List.map_map, Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff]
    refine ⟨by simpa using hk, ?_⟩
    rw [List.getD_eq_getElem _ _ (by simpa using hk), List.getElem_map]
    exact h1

/-- **`HonestPreOnPos` at a plan that stops where a refusal is seen** (`encStop enc`): at a round where none is seen,
`enc`'s message is `honestPR`'s (`hlive`); where an outer coin `1` is seen `honestPR` stops (`outerHit_of_lt`); where a
throw is seen it sends `[]` (`hthrow`). -/
theorem honestPreOnPos_stop {P : Composed.LigProver}
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
    {enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg}
    {fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg}
    (U : TapePlan (tabC P)) (nr : Msg → ℕ → ℕ) (H : Msg → Msg)
    (hlive : ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : U.Tape) (ops : List (Msg × Msg)), 0 < ops.length →
      ops.length < refK x → ¬ Refused x (padOps ops) ops.length →
      enc x (padOps ops) ops.length
          ((session nr (tableData P (RoundPlan.stop cz enc fin) U)).cutOf x (padOps ops) (wtOf P x w) τ) =
        (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) (U.tapeZk τ)).eval H).1)
    (hthrow : ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : U.Tape) (ops : List (Msg × Msg)), ops.length < refK x →
      Thrown x (padOps ops) ops.length →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) (U.tapeZk τ)).eval H).1 =
        []) :
    (session nr (tableData P (RoundPlan.stop cz (encStop enc) fin) U)).HonestPreOnPos Live RZk H coinOfD
      fun ω => U.tapeZk ω.2 := by
  intro x hx w hR _ τ ops h0 hl
  change encStop enc x (padOps ops) ops.length _ = _
  dsimp only [encStop]
  by_cases hs : Refused x (padOps ops) ops.length
  · rw [ite_eq_left hs]
    rcases hs with ⟨j, hj, r, hr, ho, e, he, h1⟩ | ht
    · rw [FlockComplete.ZkTables.honestPR_stop (outerHit_of_lt ops hj hr ho he h1)]
      rfl
    · exact (hthrow x hx w hR τ ops hl ht).symm
  · rw [ite_eq_right hs]
    exact hlive x hx w hR τ ops h0 hl hs

/-- **An opened coin's element is its opening's**, at a round the openings answer. -/
theorem slotAt_padOps_lt {x : Msg} (ops : List (Msg × Msg)) {k : ℕ} (hk : k < ops.length) (hK : k < refK x)
    {i : ℕ} (hi : i < 65536) :
    slotAt (padOps (x := x) ops) k i = FlockComplete.coinOf ((ops.map fun p => coinOfD p.2).getD k []) i := by
  unfold slotAt padOps FlockDeploy.Coin.at
  simp only [hK, hi, ↓reduceDIte]
  rw [List.getD_eq_getElem _ _ (by simpa using hk), List.getD_eq_getElem _ _ (by simpa using hk)]
  simp only [List.getElem_map, rd]

/-- **A rep's claim points at the openings' answers are its points at the opened coins** (`ptsOpen`), where the
openings answer its first `m + k_log − 6` rounds. -/
theorem ptsAt_open {x : Msg} {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (ops : List (Msg × Msg)) (hl : ops.length < refK x) {j n : ℕ} (hm : 14 ≤ (stsOf x)[j]!.m)
    (hm35 : (stsOf x)[j]!.m ≤ 35) (hk : 6 ≤ (stsOf x)[j]!.c.kLog)
    (hn : repBase x n + ((stsOf x)[j]!.m + (stsOf x)[j]!.c.kLog - 6) ≤ ops.length) :
    FlockComplete.ZkTables.ptsAt (fun k => (ops.map fun p => coinOfD p.2).getD k []) spec (stsOf x) j n =
      ptsOpen (padOps (x := x) ops) (stsOf x)[j]! n := by
  unfold FlockComplete.ZkTables.ptsAt ptsOpen
  rw [repBase_eq hsx] at hn ⊢
  refine FlockComplete.ZkTables.claimPoints_congr_lt (by omega) hk (fun n' hn' => ?_) (fun e he => ?_)
  · rw [slotAt_padOps_lt ops (by omega) (by omega) (by norm_num)]
  · rw [slotAt_padOps_lt ops (by omega) (by omega) (by omega)]

/-- **A throw the openings show stops the honest prover**: after openings `ops`, fewer than `refK x`, at which a
throw is seen (`Thrown`), the masked session on the mask witness stops within the openings' answers, right after the
send whose reply it refused (`sessionPZk_rho_stop`, `sessionPZk_beta_stop`, `sessionPZk_rank_stop`), so `honestPR`'s
message is `[]` (`honestPR_stops`). -/
theorem honestPR_thrown (H : Msg → Msg) {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk)
    (ops : List (Msg × Msg)) (hl : ops.length < refK x) (ht : Thrown x (padOps ops) ops.length) :
    (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) ω).eval H).1 = [] := by
  obtain ⟨j, hj, r, hr, hth⟩ := ht
  obtain ⟨s, spec, hsx⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hh : (ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd = ops.map fun p => coinOfD p.2 := by
    rw [List.map_map]
    rfl
  have hsts : ∀ i (hi : i < (stsOf x).size), 14 ≤ (stsOf x)[i].m ∧ 6 ≤ (stsOf x)[i].c.kLog :=
    fun i hi => ⟨by have := (hall i hi).1; omega, (hx.2 i hi).some.wf.1.hk6⟩
  have hbl : (ops.map fun p => coinOfD p.2).length = ops.length := List.length_map _
  refine FlockComplete.ZkTables.honestPR_stops H x w ω hsx _ _ hh fun R L hY => ?_
  rw [hbl]
  have hq : 2 * j + r < 2 * (stsOf x).size := by omega
  have hst : (stsOf x)[(2 * j + r) / 2]! = (stsOf x)[j] := by
    rw [show (2 * j + r) / 2 = j by omega]
    exact getElem!_pos (stsOf x) j hj
  obtain ⟨y⟩ := hx.2 j hj
  have hz : _root_.Flock.Zk.schedule (stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]
    exact y.hz
  rcases hth with ⟨hm, hk, h0, h1, hbad⟩ | ⟨hρ, hbad⟩ | ⟨hβ, hbad⟩
  · have hjj : (stsOf x)[j]! = (stsOf x)[j] := getElem!_pos (stsOf x) j hj
    have hm35 : (stsOf x)[j]!.m ≤ 35 := by rw [hjj]; exact (hall j hj).2.1
    have e1 := ptsAt_open hsx ops hl (n := 2 * j + r) hm hm35 hk h1
    have e0 := ptsAt_open hsx ops hl (n := 2 * j) hm hm35 hk (by unfold rankSlot at h0; rwa [Nat.add_zero] at h0)
    have := FlockComplete.ZkTables.sessionPZk_rank_stop _ _ spec (stsOf x) _ _ ω hspec hsts hq ?hb hY
    case hb =>
      rw [show (2 * j + r) % 2 = r by omega, show (2 * j + r) / 2 = j by omega,
        FlockComplete.MaskRank.masksOf_getElem! hj, e1]
      unfold RankBad at hbad
      by_cases hr1 : r = 1
      · subst hr1
        simp only [↓reduceIte] at hbad ⊢
        rw [show 2 * j + 1 - 1 = 2 * j by omega, e0]
        exact hbad
      · simp only [hr1, ↓reduceIte] at hbad ⊢
        exact hbad
    refine ⟨this.1, ?_⟩
    have h2 := this.2
    unfold rankSlot at h1
    rw [repBase_eq hsx] at h1
    rw [show (2 * j + r) / 2 = j by omega] at h2
    omega
  · have hs : rhoSlot x j r = FlockComplete.repStartZk spec (stsOf x) (2 * j + r) +
        FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! + (y.sch.k0 + 2) := by
      unfold rhoSlot
      rw [repSlot_eq hsx hz]
    unfold rhoAt at hbad
    rw [hs] at hbad hρ
    rw [slotAt_padOps_lt ops (by omega) (by omega) (by norm_num : 0 < 65536),
      slotAt_padOps_lt ops (by omega) (by omega) (by norm_num : 1 < 65536),
      slotAt_padOps_lt ops (by omega) (by omega) (by norm_num : 0 < 65536),
      slotAt_padOps_lt ops (by omega) (by omega) (by norm_num : 1 < 65536)] at hbad
    have := FlockComplete.ZkTables.sessionPZk_rho_stop _ _ spec (stsOf x) _ _ ω hspec hsts hq hz ?hb hY
    case hb =>
      unfold FlockComplete.coinKOf
      exact hbad
    have h2 := this.2
    dsimp only at h2
    exact ⟨this.1, by omega⟩
  · have hs : betaSlot x j r = FlockComplete.repStartZk spec (stsOf x) (2 * j + r) +
        FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! +
          (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) + 2 := by
      unfold betaSlot
      rw [repSlot_eq hsx hz]
      omega
    unfold betaAt at hbad
    rw [hs] at hbad hβ
    rw [slotAt_padOps_lt ops hβ (by omega) (by norm_num)] at hbad
    have := FlockComplete.ZkTables.sessionPZk_beta_stop _ _ spec (stsOf x) _ _ ω hspec hsts hq hz ?hb hY
    case hb => exact hbad
    have h2 := this.2
    dsimp only at h2
    exact ⟨this.1, by omega⟩

/-- **`hthrow` at any tape**: `honestPR_thrown`. -/
theorem thrown_stops (H : Msg → Msg) {T : Type} (tz : T → FlockComplete.TapeZk) :
    ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : T) (ops : List (Msg × Msg)), ops.length < refK x →
      Thrown x (padOps ops) ops.length →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) (tz τ)).eval H).1 = [] :=
  fun _ hx w _ τ ops hl ht => honestPR_thrown H hx w (tz τ) ops hl ht

end Masked

open Masked

open Classical in
/-- **Flock's zero knowledge at the deployed masked session, at the executable's layout and a plan that stops where a
refusal is seen**: `Flock.zk_deployZk_masked_exe_refusals_gap` at `encStop enc` (causal and `enc0` at round `0` where
`enc` is), its rounds after the first asked only where no refusal is seen (`honestPreOnPos_stop`). Left: `enc` at those
rounds, a seen throw stopping `honestPR`, the plan's `Causal`, its round `0` and the last round where every table
admits. -/
theorem Flock.zk_deployZk_masked_exe_stopped_gap (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash)
    (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
    {enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg}
    {fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg}
    (h0 : ∀ x c cut, enc x c 0 cut = enc0 P x (cz x c) cut) (t : ℕ) (ht : 0 < t)
    (hc : (RoundPlan.stop cz enc fin).Causal) (ho : OuterReads cz) (hr : CoinReads cz) (hp : PointReads cz)
    (hlive : ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), 0 < ops.length → ops.length < refK x →
      ¬ Refused x (padOps ops) ops.length →
      enc x (padOps ops) ops.length ((session nr (tableData P (RoundPlan.stop cz enc fin)
          (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).cutOf x (padOps ops) (wtOf P x w) τ) =
        (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2))
          ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ)).eval H).1)
    (hthrow : ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), ops.length < refK x →
      Thrown x (padOps ops) ops.length →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2))
        ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ)).eval H).1 = [])
    (hfin : (session nr (tableData P (RoundPlan.stop cz (encStop enc) fin)
      (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestFinOn Live RZk H coinOfD
        fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr
        (tableData P (RoundPlan.stop cz (encStop enc) fin) (TapePlan.ofLayout (TapeLayout.exe P) exeTape)) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_masked_exe_refusals_gap H hH nr P (enc := encStop enc)
    (fun x c cut => by dsimp only [encStop]; rw [ite_eq_right (refused_zero c)]; exact h0 x c cut) t ht
    (causal_encStop hc) ho hr hp (honestPreOnPos_stop _ nr H hlive hthrow) hfin

open Classical in
/-- **Flock's zero knowledge at the deployed masked session, at the executable's layout and a plan that stops where a
refusal is seen**: `Flock.zk_deployZk_masked_exe_stopped_gap` with a seen throw stopping `honestPR`
(`honestPR_thrown`: the session stops right after the round whose reply it refused). Left: `enc` at the rounds after
the first where no refusal is seen, the plan's `Causal`, its round `0` and the last round where every table admits. -/
theorem Flock.zk_deployZk_masked_exe_live_gap (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash)
    (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
    {enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg}
    {fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg}
    (h0 : ∀ x c cut, enc x c 0 cut = enc0 P x (cz x c) cut) (t : ℕ) (ht : 0 < t)
    (hc : (RoundPlan.stop cz enc fin).Causal) (ho : OuterReads cz) (hr : CoinReads cz) (hp : PointReads cz)
    (hlive : ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), 0 < ops.length → ops.length < refK x →
      ¬ Refused x (padOps ops) ops.length →
      enc x (padOps ops) ops.length ((session nr (tableData P (RoundPlan.stop cz enc fin)
          (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).cutOf x (padOps ops) (wtOf P x w) τ) =
        (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2))
          ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ)).eval H).1)
    (hfin : (session nr (tableData P (RoundPlan.stop cz (encStop enc) fin)
      (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestFinOn Live RZk H coinOfD
        fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr
        (tableData P (RoundPlan.stop cz (encStop enc) fin) (TapePlan.ofLayout (TapeLayout.exe P) exeTape)) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_masked_exe_stopped_gap H hH nr P h0 t ht hc ho hr hp hlive (thrown_stops H _) hfin

end Claims.ZKMal
