import Proofs.Flock.Claims.ZK.Refusals
import Proofs.Flock.Claims.Complete.ZkTables.Refuse
import Proofs.Flock.Claims.Complete.ZkTables.Pass

/-!
# The masked session's `β` and pair-coin refusals, as its first refusal

`sessionPZk_pass` reads the masked session forwards: a session that returns passed every rep's `β` and pair-coin checks
at their rounds. Read the other way, it is the first refusal. At a refused `β` (`0`) or a dependent pair `ρ₀, ρ₁` the
session does not return, so `honestPR` sends `[]` (`honestPR_err`). This discharges `honestOffOn_stop_of`'s `hβ` and
`hρ` at any coin plan that reads each rep's `β` and pair coins at the executable's rounds (`CoinReads`; `withCoins` is
one).

* `betaSlot`, `rhoSlot`: rep `2j + r`'s `β` round and its `ρ₀` round (`ρ₁` is the next), as `betaSlotsZk` and
  `rhoSlotsZk` list them; `betaAt` and `rhoAt` read the opened coins there (`slotAt`).
* `honestPR_refused`: after `refK x` openings, a refused `β` or a dependent pair at one of a table's reps makes
  `honestPR` send `[]`.
* `Flock.zk_deployZk_masked_exe_reads_gap`: `Flock.zk_deployZk_masked_exe_stop_gap` without `hβ` and `hρ`, at a plan
  that reads the outer coins (`OuterReads`), `β` and the pair coins (`CoinReads`) at their rounds. Left: the rank
  refusal, the plan's `Causal`, its round `0`, the rounds after the first and the last round.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **Rep `q`'s round `f sch` after its PIOP rounds**: `repStartZk q + piopRoundsZk + f sch` at `x`'s session, `sch`
rep `q`'s table's schedule; `0` where `x` does not set up or the schedule fails. -/
def repSlot (x : Msg) (q : ℕ) (f : _root_.Flock.Schedule → ℕ) : ℕ :=
  match FlockDeploy.setupOf true x with
  | some (_, spec, sts) =>
    match _root_.Flock.Zk.schedule sts[q / 2]!.m with
    | .ok (sch, _) => FlockComplete.repStartZk spec sts q + FlockComplete.piopRoundsZk spec sts[q / 2]! + f sch
    | .error _ => 0
  | none => 0

/-- **Table `j`'s rep `r`'s `ρ₀` round**: Ligerito's round `k₀ + 2` (`rhoSlotsZk`); `ρ₁`'s is the next. -/
def rhoSlot (x : Msg) (j r : ℕ) : ℕ := repSlot x (2 * j + r) fun sch => sch.k0 + 2

/-- **Table `j`'s rep `r`'s `β` round**: the inner proof's round `2`, after Ligerito's `k₀ + 7 + 11R` (`betaSlotsZk`). -/
def betaSlot (x : Msg) (j r : ℕ) : ℕ := repSlot x (2 * j + r) fun sch => sch.k0 + 7 + 11 * (sch.levels.size - 1) + 2

/-- **Element `i` of the opened coin at round `k`**, `0` past the pre-final rounds. -/
def slotAt {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k i : ℕ) : _root_.Flock.F128 :=
  if h : k < refK x then (c ⟨k, h⟩).at i else 0

/-- **Table `j`'s rep `r`'s `β` at the opened coins `c`**: element `0` of the coin at its round. -/
def betaAt {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j r : ℕ) : _root_.Flock.F128 := slotAt c (betaSlot x j r) 0

/-- **Table `j`'s rep `r`'s pair coins at the opened coins `c`**: `ρ₀` is elements `0` and `1` of the coin at its round,
`ρ₁` the next round's. -/
def rhoAt {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j r : ℕ) : _root_.Flock.F256 × _root_.Flock.F256 :=
  (⟨slotAt c (rhoSlot x j r) 0, slotAt c (rhoSlot x j r) 1⟩,
    ⟨slotAt c (rhoSlot x j r + 1) 0, slotAt c (rhoSlot x j r + 1) 1⟩)

/-- **A coin plan reads each rep's `β` and pair coins at the executable's rounds** (`betaAt`, `rhoAt`). -/
def CoinReads {P : Composed.LigProver}
    (cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest) : Prop :=
  ∀ x c (j : Fin (jOf x)) (r : Fin 2), (cz x c j).beta r = betaAt c j.val r.val ∧ (cz x c j).rho r = rhoAt c j.val r.val

/-- **A coin plan with each rep's outer coins, `β` and pair coins read at their rounds**: `withOuter cz`'s, with `β` at
`betaAt` and the pair coins at `rhoAt`. -/
noncomputable def withCoins {P : Composed.LigProver}
    (cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest) :
    (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest :=
  fun x c j => { withOuter cz x c j with beta := fun r => betaAt c j.val r.val, rho := fun r => rhoAt c j.val r.val }

theorem coinReads_withCoins {P : Composed.LigProver}
    (cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest) :
    CoinReads (withCoins cz) :=
  fun _ _ _ _ => ⟨rfl, rfl⟩

theorem outerReads_withCoins {P : Composed.LigProver}
    (cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest) :
    OuterReads (withCoins cz) :=
  outerReads_withOuter cz

theorem repSlot_eq {x : Msg} {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) {q : ℕ}
    {sch : _root_.Flock.Schedule} {tPad : ℕ} (hz : _root_.Flock.Zk.schedule (stsOf x)[q / 2]!.m = .ok (sch, tPad))
    (f : _root_.Flock.Schedule → ℕ) :
    repSlot x q f =
      FlockComplete.repStartZk spec (stsOf x) q + FlockComplete.piopRoundsZk spec (stsOf x)[q / 2]! + f sch := by
  unfold repSlot
  rw [hsx]
  simp only [hz]

theorem refK_eq {x : Msg} {s spec sts} (hsx : FlockDeploy.setupOf true x = some (s, spec, sts)) :
    refK x = 1 + FlockDeploy.rounds true spec sts := by
  unfold refK
  simp only [FlockDeploy.V, hsx]
  omega

/-- **An opened coin's element is its opening's**, at a round of the openings. -/
theorem slotAt_padOps {x : Msg} (ops : List (Msg × Msg)) (hl : ops.length = refK x) {k : ℕ} (hk : k < refK x)
    {i : ℕ} (hi : i < 65536) :
    slotAt (padOps (x := x) ops) k i = FlockComplete.coinOf ((ops.map fun p => coinOfD p.2).getD k []) i := by
  have hk' : k < ops.length := hl ▸ hk
  unfold slotAt padOps FlockDeploy.Coin.at
  simp only [hk, hi, ↓reduceDIte]
  rw [List.getD_eq_getElem _ _ (by simpa using hk'), List.getD_eq_getElem _ _ (by simpa using hk')]
  simp only [List.getElem_map, rd]

/-- **A refused `β` or a dependent pair stops the honest prover**: after `refK x` openings `ops` at which table `j`'s
rep `r` has `β = 0` or dependent pair coins, the masked session on any witness does not return at the openings'
answers (`sessionPZk_pass`), so `honestPR`'s message is `[]` (`honestPR_err`). -/
theorem honestPR_refused (H : Msg → Msg) {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk)
    (ops : List (Msg × Msg)) (hl : ops.length = refK x) {j r : ℕ} (hj : j < (stsOf x).size) (hr : r < 2)
    (hbad : (betaAt (padOps (x := x) ops) j r).isZero = true ∨
      _root_.Flock.Zk.rhoIndependent (rhoAt (padOps (x := x) ops) j r) = false) :
    (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) ω).eval H).1 = [] := by
  obtain ⟨s, spec, hsx⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hK := refK_eq hsx
  have hh : (ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd = ops.map fun p => coinOfD p.2 := by
    rw [List.map_map]
    rfl
  refine FlockComplete.ZkTables.honestPR_err H x w ω hsx hspec (fun i hi => by have := (hall i hi).1; omega) _ _ hh
    (by rw [List.length_map]; omega) ?_
  intro R L hrun
  have hsts : ∀ i (hi : i < (stsOf x).size), 14 ≤ (stsOf x)[i].m ∧ 6 ≤ (stsOf x)[i].c.kLog :=
    fun i hi => ⟨by have := (hall i hi).1; omega, (hx.2 i hi).some.wf.1.hk6⟩
  have hpass := FlockComplete.ZkTables.sessionPZk_pass _ _ spec (stsOf x) _ _ ω hspec hsts hrun (2 * j + r)
    (by omega)
  have hst : (stsOf x)[(2 * j + r) / 2]! = (stsOf x)[j] := by
    rw [show (2 * j + r) / 2 = j by omega]
    exact getElem!_pos (stsOf x) j hj
  obtain ⟨y⟩ := hx.2 j hj
  have hz : _root_.Flock.Zk.schedule (stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]
    exact y.hz
  have hk6 : 6 ≤ (stsOf x)[(2 * j + r) / 2]!.c.kLog := by
    rw [hst]
    exact y.wf.1.hk6
  obtain ⟨hρ, hβ⟩ := hpass (y.sch, y.tPad) hz
  have hend : FlockComplete.repStartZk spec (stsOf x) (2 * j + r) +
      FlockDeploy.repRounds true spec (stsOf x)[(2 * j + r) / 2]! ≤ refK x := by
    rw [← FlockComplete.ZkTables.repStart_succ, hK, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono spec (stsOf x) (by omega)
  have hrr : FlockDeploy.repRounds true spec (stsOf x)[(2 * j + r) / 2]! =
      FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! +
        (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) + 4 := by
    unfold FlockDeploy.repRounds FlockComplete.piopRoundsZk
    simp only [↓reduceIte, hz]
    omega
  rw [hrr] at hend
  rcases hbad with hb | hb
  · have hs : betaSlot x j r = FlockComplete.repStartZk spec (stsOf x) (2 * j + r) +
        FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! +
          (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) + 2 := by
      unfold betaSlot
      rw [repSlot_eq hsx hz]
      omega
    unfold betaAt at hb
    rw [hs, slotAt_padOps ops hl (by omega) (by norm_num)] at hb
    have hβ' : (FlockComplete.coinOf ((ops.map fun p => coinOfD p.2).getD
        (FlockComplete.repStartZk spec (stsOf x) (2 * j + r) +
          FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! +
            (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) + 2) []) 0).isZero = false := hβ
    rw [hβ'] at hb
    cases hb
  · have hs : rhoSlot x j r = FlockComplete.repStartZk spec (stsOf x) (2 * j + r) +
        FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! + (y.sch.k0 + 2) := by
      unfold rhoSlot
      rw [repSlot_eq hsx hz]
    unfold rhoAt at hb
    rw [hs, slotAt_padOps ops hl (by omega) (by norm_num : 0 < 65536),
      slotAt_padOps ops hl (by omega) (by norm_num : 1 < 65536),
      slotAt_padOps ops hl (by omega) (by norm_num : 0 < 65536),
      slotAt_padOps ops hl (by omega) (by norm_num : 1 < 65536)] at hb
    have hρ' : _root_.Flock.Zk.rhoIndependent
        (FlockComplete.coinKOf ((ops.map fun p => coinOfD p.2).getD
          (FlockComplete.repStartZk spec (stsOf x) (2 * j + r) +
            FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! + (y.sch.k0 + 2)) []),
        FlockComplete.coinKOf ((ops.map fun p => coinOfD p.2).getD
          (FlockComplete.repStartZk spec (stsOf x) (2 * j + r) +
            FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! + (y.sch.k0 + 2) + 1) [])) = true := hρ
    unfold FlockComplete.coinKOf at hρ'
    rw [hρ'] at hb
    cases hb

/-- **`hβ` at a plan reading `β` at its rounds**: `honestPR_refused` at a refused `β`. -/
theorem beta_refused {P : Composed.LigProver}
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
    (hr : CoinReads cz) (H : Msg → Msg) {T : Type} (tz : T → FlockComplete.TapeZk) :
    ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), ops.length = refK x → ∀ j r,
      ((cz x (padOps ops) j).beta r).isZero = true →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) (tz τ)).eval H).1 = [] := by
  intro x hx w _ τ ops hl j r hb
  rw [(hr x _ j r).1] at hb
  exact honestPR_refused H hx w _ ops hl (lt_of_jOf j) r.isLt (.inl hb)

/-- **`hρ` at a plan reading the pair coins at their rounds**: `honestPR_refused` at a dependent pair. -/
theorem rho_refused {P : Composed.LigProver}
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
    (hr : CoinReads cz) (H : Msg → Msg) {T : Type} (tz : T → FlockComplete.TapeZk) :
    ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), ops.length = refK x → ∀ j r,
      _root_.Flock.Zk.rhoIndependent ((cz x (padOps ops) j).rho r) = false →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) (tz τ)).eval H).1 = [] := by
  intro x hx w _ τ ops hl j r hb
  rw [(hr x _ j r).2] at hb
  exact honestPR_refused H hx w _ ops hl (lt_of_jOf j) r.isLt (.inr hb)

end Masked

open Masked

open Classical in
/-- **Flock's zero knowledge at the deployed masked session, at the executable's layout and a stopping round plan that
reads the outer coins, `β` and the pair coins at their rounds**: `Flock.zk_deployZk_masked_exe_stop_gap` with its `hβ`
and `hρ` from the session's first refusal (`honestPR_refused`). Left: the plan's `Causal`, its round `0` `enc0`, the
rounds after the first, the last round where every table admits, and the rank refusal. -/
theorem Flock.zk_deployZk_masked_exe_reads_gap (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash)
    (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
    {enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg}
    {fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg}
    (h0 : ∀ x c cut, enc x c 0 cut = enc0 P x (cz x c) cut) (t : ℕ) (ht : 0 < t)
    (hc : (RoundPlan.stop cz enc fin).Causal) (ho : OuterReads cz) (hr : CoinReads cz)
    (hpos : (session nr (tableData P (RoundPlan.stop cz enc fin)
      (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestPreOnPos Live RZk H coinOfD
        fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2)
    (hfin : (session nr (tableData P (RoundPlan.stop cz enc fin)
      (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestFinOn Live RZk H coinOfD
        fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2)
    (hrank : ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), ops.length = refK x → ∀ j,
      ¬ Composed.MaskRankOk ((tabC P x j).points (cz x (padOps ops) j)) (tabC P x j).maskWords →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2))
        ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ)).eval H).1 = []) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr
        (tableData P (RoundPlan.stop cz enc fin) (TapePlan.ofLayout (TapeLayout.exe P) exeTape)) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_masked_exe_stop_gap H hH nr P h0 t ht hc ho hpos hfin (beta_refused hr H _) hrank
    (rho_refused hr H _)

end Claims.ZKMal
