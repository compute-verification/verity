import Proofs.Flock.Claims.ZK.ExeLayout

/-!
# The masked session's last round where a table refuses its coins

At a stopping round plan (`RoundPlan.stop`, `off = []`), `HonestOffOn` asks that `honestPR`'s last message is `[]`
wherever some table refuses the coins (`honestOffOn_stop_iff`). `CoinedSetup.Admits` has four conjuncts, and the
executable refuses at each:

* **an outer coin `1`** (`Masking::refuse_outer`): `honestPR` stops at a history one of whose zerocheck outer coins
  (`outerSlotsZk`) is `1` (`outerHit`, `honestPR_stop`). At a coin plan reading each rep's outer coins at their slots
  (`OuterReads`, `outerAt`; any plan with its `rOuter` so read is one, `withOuter`), a refused outer coin is such a hit
  (`outerHit_of`), so `honestPR` sends `[]`;
* **`β = 0`** (`innerP`), **a rank-deficient mask slot** (`rankCheckP`), **`ρ₀, ρ₁` dependent** (`ligeritoPZk`): throws
  that end `sessionPZk` with `.error`, which `ZkTables.prog` sends as `[]` (`FirstRefusal` discharges all three).

`honestOffOn_stop_of` is `HonestOffOn` at a stopping plan from the outer coins' reading and one premise per throw, and
`Flock.zk_deployZk_masked_exe_stop_gap` is `Flock.zk_deployZk_masked_exe_gap` at a stopping plan with it in place of
`HonestOffOn`.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **Table `j`'s rep `r`'s outer coins' round**: rep `2j + r`'s second round (`outerSlotsZk`), `0` where `x` does not
set up the masked session. -/
def outerSlot (x : Msg) (j r : ℕ) : ℕ :=
  match FlockDeploy.setupOf true x with
  | some (_, spec, sts) => FlockComplete.repStartZk spec sts (2 * j + r) + 1
  | none => 0

/-- **Outer coin `i` of table `j`'s rep `r` at the opened coins `c`**: element `i` of the coin at its round, `0` past
the pre-final rounds. -/
def outerAt {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j r i : ℕ) : _root_.Flock.F128 :=
  if h : outerSlot x j r < refK x then (c ⟨outerSlot x j r, h⟩).at i else 0

/-- **A coin plan reads each rep's outer coins at the executable's slots**: every coin of table `j`'s rep `r`'s
`rOuter` is its outer coin `i` (`outerAt`) for some `i < m − 13`. -/
def OuterReads {P : Composed.LigProver}
    (cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest) : Prop :=
  ∀ x c (j : Fin (jOf x)) (r : Fin 2), ∀ y ∈ (tabC P x j).rOuter (cz x c j) r,
    ∃ i < (stAt x j).m - 13, y = outerAt c j.val r.val i

/-- **A coin plan with each rep's outer coins read at their slots**: `cz`'s, its `rOuter` the opened outer coins
(`outerAt`). -/
noncomputable def withOuter {P : Composed.LigProver}
    (cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest) :
    (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest :=
  fun x c j => { cz x c j with
    rest := ({ ((cz x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) with
      rOuter := fun r i => outerAt c j.val r.val i.val } : Composed.RestOf (zkTab x j).table (stAt x j)) }

theorem outerReads_withOuter {P : Composed.LigProver}
    (cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest) :
    OuterReads (withOuter cz) := by
  intro x c j r y hy
  obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hy
  exact ⟨i.val, i.isLt, rfl⟩

/-- **The coins `refK x` openings open**: opening `k`'s, read as its coin (`rd`); the session's `pad`. -/
def padOps {x : Msg} (ops : List (Msg × Msg)) : Fin (refK x) → FlockDeploy.Coin :=
  fun k => (ops.map fun p => rd p.2).getD k.val default

/-- **An opened outer coin `1` is an outer hit**: after `refK x` openings `ops` whose coins are `c`, outer coin
`i < m − 13` of table `j`'s rep `r` equal to `1` makes `honestPR` stop (`outerHit`). -/
theorem outerHit_of {x : Msg} (ops : List (Msg × Msg)) (hl : ops.length = refK x)
    (c : Fin (refK x) → FlockDeploy.Coin) (hc : ∀ (k : Fin (refK x)) (hk : k.val < ops.length), c k = rd ops[k.val].2)
    (j : Fin (jOf x)) (r : Fin 2) {i : ℕ} (hi : i < (stAt x j).m - 13) (h1 : outerAt c j.val r.val i = 1) :
    FlockComplete.ZkTables.outerHit x ((ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd) = true := by
  have h10 : (0 : _root_.Flock.F128) ≠ 1 := by decide
  unfold outerAt at h1
  split_ifs at h1 with hs
  swap
  · exact absurd h1 h10
  have hk : outerSlot x j.val r.val < ops.length := hl ▸ hs
  rw [hc _ hk] at h1
  unfold FlockDeploy.Coin.at at h1
  split_ifs at h1 with hi'
  swap
  · exact absurd h1 h10
  obtain ⟨s, spec, hsx⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) (lt_of_jOf j))
  have hsl : outerSlot x j.val r.val = FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) + 1 := by
    unfold outerSlot
    rw [hsx]
  unfold FlockComplete.ZkTables.outerHit
  rw [List.any_eq_true]
  refine ⟨(outerSlot x j.val r.val, i), ?_, ?_⟩
  · unfold FlockComplete.outerSlotsZk
    rw [hsx]
    simp only [List.mem_flatMap, List.mem_range, List.mem_map]
    refine ⟨2 * j.val + r.val, by have := lt_of_jOf j; have := r.isLt; omega, i, ?_, ?_⟩
    · have e : (2 * j.val + r.val) / 2 = j.val := by have := r.isLt; omega
      rw [e, getElem!_pos (stsOf x) j.val (lt_of_jOf j)]
      exact hi
    · rw [hsl]
  · rw [List.map_map, Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff]
    refine ⟨by simpa using hk, ?_⟩
    rw [List.getD_eq_getElem _ _ (by simpa using hk), List.getElem_map]
    exact h1

/-- **`HonestOffOn` at a stopping round plan, refusal by refusal**: at a coin plan reading each rep's outer coins at
their slots (`OuterReads`), an outer coin `1` is `honestPR_stop`'s (`outerHit_of`); the other three refusals, `β = 0`,
a rank-deficient mask slot and `ρ₀, ρ₁` dependent, are each a premise that `honestPR`'s last message is `[]`. -/
theorem honestOffOn_stop_of {P : Composed.LigProver} {cz enc fin} (U : TapePlan (tabC P)) (nr : Msg → ℕ → ℕ)
    (A : Msg → Prop) (R : Rel Claims.Flock.Wit) (H : Msg → Msg) (ho : OuterReads cz)
    (hβ : ∀ x, A x → ∀ w, R x w → ∀ t (ops : List (Msg × Msg)), ops.length = refK x → ∀ j r,
      ((cz x (padOps ops) j).beta
        r).isZero = true →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) (U.tapeZk t)).eval H).1 = [])
    (hrank : ∀ x, A x → ∀ w, R x w → ∀ t (ops : List (Msg × Msg)), ops.length = refK x → ∀ j,
      ¬ Composed.MaskRankOk ((tabC P x j).points
          (cz x (padOps ops) j))
        (tabC P x j).maskWords →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) (U.tapeZk t)).eval H).1 = [])
    (hρ : ∀ x, A x → ∀ w, R x w → ∀ t (ops : List (Msg × Msg)), ops.length = refK x → ∀ j r,
      _root_.Flock.Zk.rhoIndependent
        ((cz x (padOps ops) j).rho r) =
          false →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) (U.tapeZk t)).eval H).1 = []) :
    (session nr (tableData P (RoundPlan.stop cz enc fin) U)).HonestOffOn A R H coinOfD (fun ω => U.tapeZk ω.2) := by
  refine (honestOffOn_stop_iff U nr A R H).2 fun x hx w hR t ops hl hA => ?_
  obtain ⟨j, hj⟩ := not_forall.1 hA
  change ¬ (tabC P x j).Admits (cz x (padOps ops) j) at hj
  simp only [Composed.CoinedSetup.Admits, not_and_or, not_forall, not_not, exists_prop, Bool.not_eq_false,
    Bool.not_eq_true] at hj
  rcases hj with ⟨r, hr⟩ | hr | ⟨r, hr⟩ | ⟨r, y, hy, h1⟩
  · exact hβ x hx w hR t ops hl j r hr
  · exact hrank x hx w hR t ops hl j hr
  · exact hρ x hx w hR t ops hl j r hr
  · obtain ⟨i, hi, rfl⟩ := ho x _ j r y hy
    have hc : ∀ (k : Fin (refK x)) (hk : k.val < ops.length), padOps ops k = rd ops[k.val].2 := fun k hk => by
      unfold padOps
      rw [List.getD_eq_getElem _ _ (by simpa using hk), List.getElem_map]
    rw [FlockComplete.ZkTables.honestPR_stop (outerHit_of ops hl _ hc j r hi h1)]
    rfl

end Masked

open Masked

open Classical in
/-- **Flock's zero knowledge at the deployed masked session, at the executable's layout and a stopping round plan**:
`Flock.zk_deployZk_masked_exe_gap` at `RoundPlan.stop cz enc fin`, its `HonestOffOn` from `honestOffOn_stop_of` (an
outer coin `1` discharged at a plan reading the outer coins at their slots, `OuterReads`). Left: the plan's `Causal`,
its round `0` `enc0`, the rounds after the first, the last round where every table admits, and the three throws. -/
theorem Flock.zk_deployZk_masked_exe_stop_gap (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash)
    (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
    {enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg}
    {fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg}
    (h0 : ∀ x c cut, enc x c 0 cut = enc0 P x (cz x c) cut) (t : ℕ) (ht : 0 < t)
    (hc : (RoundPlan.stop cz enc fin).Causal) (ho : OuterReads cz)
    (hpos : (session nr (tableData P (RoundPlan.stop cz enc fin)
      (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestPreOnPos Live RZk H coinOfD
        fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2)
    (hfin : (session nr (tableData P (RoundPlan.stop cz enc fin)
      (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestFinOn Live RZk H coinOfD
        fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2)
    (hβ : ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), ops.length = refK x → ∀ j r,
      ((cz x (padOps ops) j).beta r).isZero = true →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2))
        ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ)).eval H).1 = [])
    (hrank : ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), ops.length = refK x → ∀ j,
      ¬ Composed.MaskRankOk ((tabC P x j).points (cz x (padOps ops) j)) (tabC P x j).maskWords →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2))
        ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ)).eval H).1 = [])
    (hρ : ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), ops.length = refK x → ∀ j r,
      _root_.Flock.Zk.rhoIndependent ((cz x (padOps ops) j).rho r) = false →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2))
        ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ)).eval H).1 = []) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr
        (tableData P (RoundPlan.stop cz enc fin) (TapePlan.ofLayout (TapeLayout.exe P) exeTape)) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_masked_exe_gap H hH nr P (RoundPlan.stop cz enc fin) h0 t ht hc hpos hfin
    (honestOffOn_stop_of _ nr Live RZk H ho hβ hrank hρ)

end Claims.ZKMal
