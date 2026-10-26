import Proofs.Flock.Claims.KS.PlainKSLin
import Proofs.Flock.Claims.KS.PlainDrawRColsE

/-!
# `TableBoundAbortLin` at the deployed reader's plain record

`plain_tableBoundAbortLin` and `plain_tableBoundAbortLinK` are one proof at two verifiers. Here it is once, at any plain
record `D` whose verifier has `V false`'s rounds (`tableBoundAbortLin_of_rdAll`), from what the proof reads of `D`:

* `RdAllAt D lc`: the reader's bound at one tape (`plain_accPr_le_rdAll`'s form), with the run's level columns read by
  `lc`;
* `LevelTopAt D.V lc`: an accepted run has no level columns at or past its schedule's last level
  (`plainRawLevelCols_top`'s form);
* `CapHitAt D.V lc`: each cap position's read sets at `rdAtW D.V lc`, linearly (`Pr_rdAt_le_capHit`'s form);
* the four refuser code facts `CoinsShown`, `AccLast`, `DecideFree` and `Refusable` at `D.V`.

`tableBoundAbortLinK_of_rdAll` checks the form against the K chain: at `deployPlainK keep` all of them hold
(`plain_accPr_le_rdAllK`, `plainRawLevelCols_topK`, `Pr_rdAt_le_capHitK`, `FlockKSDeployK`'s code facts).

At the deployed reader `VR rd` (`nextR rd` at `V false`'s rounds):

* `coinsShown_VR`, `refusable_VR`: a coin is the plain verifier's (`nextR_coin`), and `[0]` is never accepted: a coin
  request where a coin follows it, else no decision (`decR` reads a first coin).
* `tableBoundAbortLin_VR`: the table bound at any plain record whose verifier is `VR rd`, from the reader's three
  facts there and `AccLast` and `DecideFree` at `VR rd`.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-! ## What the table bound reads of a plain record -/

open Classical in
/-- **What an aborting run of `W` reads at rep `r`'s level `ℓ + 1` position `q`** (`rdAt` at `W`, with the level columns
read by `lc`): where it accepts, the rows its level columns hold at `q`; nothing elsewhere. -/
noncomputable def rdAtW (W : Claims.Verifier Coin)
    (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)) {Ω A : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) (keep : Ω → (Fin (W.k x) → Coin) → A → Bool) (r ℓ q lanes : ℕ) (ω : Ω)
    (d : (Fin (W.k x) → Coin) × A) : Finset (Fin (2 ^ lanes) → 𝔽) :=
  if (Claims.Recursion.abortRun sha512 P W x keep ω d).out = .acc then
    (((lc x (Claims.Recursion.abortRun sha512 P W x keep ω d).seen r (ℓ + 1)).filter
      fun e => e.1 = q).map fun e => rowFin lanes e.2).toFinset
  else ∅

open Classical in
/-- **The reader's bound at one tape** (`plain_accPr_le_rdAll`'s form at `D`): wherever some run of `D.V` accepts, the
statement sets up one plain table with a `fast100` schedule of at least two levels, and at every prover, tape and table
of wide columns `D` doesn't decode as good, for any read sets `rdD` holding what an accepted run's level columns
(`lc`) read at each position, and any rows `w` reading only the coins before the slot's fork round, the run accepts
with probability at most `ε_table`, a collision among its calls, the read sets' misses of `w`, and opening a position
of the word off the table at `D`'s columns. -/
def RdAllAt {T : Type} (D : Claims.Flock.Deploy Coin (List 𝔽) T)
    (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)) : Prop :=
  ∀ x, ¬ Claims.Flock.NoAcc sha512 D.V x →
    ∃ (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule) (m k0 : ℕ)
      (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ),
      setupOf false x = some (ins, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      schedOf sch = ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ∧ tables false x = [(spec, st)] ∧
      Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ∧
      ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽),
        (∀ p, wide false x (C p)) → ¬ Claims.Flock.GoodTable D (wordLen false) x C →
        ∀ (rdD : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin (D.V.k x) → Coin) →
            Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
              Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
          (w : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin (D.V.k x) → Coin) →
            Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
              Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽),
        (∀ r ℓ (c c' : Fin (D.V.k x) → Coin),
          (∀ i : Fin (D.V.k x), i.val < forkRound spec st k0 r.val ℓ.val → c i = c' i) →
            w r ℓ c = w r ℓ c') →
        (∀ r ℓ (c : Fin (D.V.k x) → Coin), (Claims.run sha512 P D.V x ω c).out = .acc →
          ∀ q a, (q.val, a) ∈ lc x (Claims.run sha512 P D.V x ω c).seen r.val (ℓ.val + 1) →
            rowFin (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes a ∈ rdD r ℓ c q) →
        Claims.accPr sha512 D.V P x ω ≤ D.εtable x +
          (Claims.Pr fun s : Fin (D.V.k x) → Coin => (Claims.run sha512 P D.V x ω s).out = .acc ∧
            Claims.TiedColl sha512 (Claims.run sha512 P D.V x ω s).vLog) +
          ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length, ∑ q,
            Claims.Pr (fun c : Fin (D.V.k x) → Coin => (rdD r ℓ c q).Nonempty ∧ rdD r ℓ c q ≠ {w r ℓ c q}) +
          Claims.Pr fun r : Fin (D.V.k x) → Coin => (Claims.run sha512 P D.V x ω r).out = .acc ∧
            ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
              Claims.Flock.lookCol (D.cols (Claims.run sha512 P D.V x ω r).seen 0) p.val = some u ∧ u ≠ C p

/-- **An accepted run of `W` has no level columns at or past its schedule's last level** (`plainRawLevelCols_top`'s
form, at the level columns `lc`). -/
def LevelTopAt (W : Claims.Verifier Coin) (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)) : Prop :=
  ∀ {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (c : Fin (W.k x) → Coin),
    (Claims.run sha512 P W x ω c).out = .acc →
    ∀ {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {sch : Flock.Schedule},
      setupOf false x = some (s, spec, sts) → sts[0]? = some st → fast100 st.m = some sch →
      ∀ (r ℓ : ℕ), sch.levels.size ≤ ℓ → lc x (Claims.run sha512 P W x ω c).seen r ℓ = []

open Classical in
/-- **A cap position's read sets at `W`, linearly** (`Pr_rdAt_le_capHit`'s form, at `rdAtW W lc`): at `CpDep`'s
position `i`, the chance an aborting run reads something there other than the plurality single reading after the
position's round is at most its `FC` hits and `1/(M+1)`. -/
def CapHitAt (W : Claims.Verifier Coin) (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)) : Prop :=
  ∀ {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (P : Claims.Prover Ω) (x : Claims.Msg)
    (keep : Ω → (Fin (W.k x) → Coin) → A → Bool) (M : ℕ) (i : Fin (CpDep.width x)) (lanes : ℕ) (ω : Ω),
    (Claims.Pr fun d => (rdAtW W lc P x keep ((Fintype.equivFin (DepIdx x)).symm i).1.val
        ((Fintype.equivFin (DepIdx x)).symm i).2.1.val ((Fintype.equivFin (DepIdx x)).symm i).2.2.val lanes ω d).Nonempty ∧
      rdAtW W lc P x keep ((Fintype.equivFin (DepIdx x)).symm i).1.val ((Fintype.equivFin (DepIdx x)).symm i).2.1.val
          ((Fintype.equivFin (DepIdx x)).symm i).2.2.val lanes ω d ≠
        {Claims.Recursion.capPlurRd W x (CpDep.round x i)
          (rdAtW W lc P x keep ((Fintype.equivFin (DepIdx x)).symm i).1.val ((Fintype.equivFin (DepIdx x)).symm i).2.1.val
            ((Fintype.equivFin (DepIdx x)).symm i).2.2.val lanes ω) d}) ≤
      (Claims.Pr fun s : ((Fin (W.k x) → Coin) × A) × (Fin M → (Fin (W.k x) → Coin) × A) =>
        ∃ u v, (Claims.Recursion.FC sha512 W CpDep P x keep M i.pos).out (i, ω, s.1, s.2) = some (u, v) ∧
          Claims.Collision sha512 u v) + 1 / ((M : ℝ) + 1)

theorem mem_rdAtW {W : Claims.Verifier Coin} {lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)}
    {Ω A : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {keep : Ω → (Fin (W.k x) → Coin) → A → Bool}
    {r ℓ q lanes : ℕ} {ω : Ω} {d : (Fin (W.k x) → Coin) × A} {u : Fin (2 ^ lanes) → 𝔽} :
    u ∈ rdAtW W lc P x keep r ℓ q lanes ω d ↔ keep ω d.1 d.2 = true ∧
      (Claims.run sha512 P W x ω d.1).out = .acc ∧
      ∃ a, (q, a) ∈ lc x (Claims.run sha512 P W x ω d.1).seen r (ℓ + 1) ∧ rowFin lanes a = u := by
  classical
  unfold rdAtW
  by_cases hacc : (Claims.Recursion.abortRun sha512 P W x keep ω d).out = .acc
  · obtain ⟨hk, hr⟩ := (Claims.Recursion.abortRun_acc_iff (H := sha512) (P := P) (V := W) (x := x)
      (keep := keep) (ω := ω) (d := d)).1 hacc
    rw [if_pos hacc, Claims.Recursion.abortRun_of_keep (H := sha512) (P := P) (V := W) (x := x)
      (keep := keep) (ω := ω) (t := d) hk]
    simp only [List.mem_toFinset, List.mem_map, List.mem_filter, decide_eq_true_eq]
    constructor
    · rintro ⟨⟨q', a⟩, ⟨hm, rfl⟩, rfl⟩
      exact ⟨hk, hr, a, hm, rfl⟩
    · rintro ⟨-, -, a, hm, rfl⟩
      exact ⟨(q, a), ⟨hm, rfl⟩, rfl⟩
  · rw [if_neg hacc]
    simp only [Finset.notMem_empty, false_iff, not_and]
    intro hk hr
    exact absurd ((Claims.Recursion.abortRun_acc_iff (H := sha512) (P := P) (V := W) (x := x)
      (keep := keep) (ω := ω) (d := d)).2 ⟨hk, hr⟩) hacc

/-! ## The table bound from what it reads -/

open Classical in
/-- **The table bound at the abort law, linearly, at `CpDep`, at any plain record with `V false`'s rounds**, from the
reader's bound at one tape (`hall`), the top level's emptiness (`htop`), each cap position's read sets (`hcap`) and
the verifier's four refuser code facts: `plain_tableBoundAbortLin`'s proof. Where no run accepts, the term `0`;
elsewhere, on each tape, the chance that the run as it aborts ties a collision among its own calls and, at each cap
position, that it reads there anything but the plurality single reading after the slot's fork round. The bound is
`hall` at the refuser's runs with those read sets and tables, averaged over the refuser's last coin (`Pr_refDraw`). -/
theorem tableBoundAbortLin_of_rdAll {T : Type} (D : Claims.Flock.Deploy Coin (List 𝔽) T)
    (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)) (hk : ∀ x, D.V.k x = (V false).k x)
    (hεt : ∀ x, 0 ≤ D.εtable x) (hcs : Claims.Recursion.CoinsShown sha512 D.V decCoin)
    (hal : Claims.Recursion.AccLast sha512 D.V)
    (hdf : Claims.Recursion.DecideFree sha512 D.V) (hrf : Claims.Recursion.Refusable sha512 D.V)
    (hall : RdAllAt D lc) (htopL : LevelTopAt D.V lc) (hcap : CapHitAt D.V lc) :
    Claims.Recursion.TableBoundAbortLin sha512 D (wordLen false) (wide false) CpDep := by
  classical
  intro x Ω _ _ P A _ _ keep M
  have hB0 := Claims.Recursion.PrBAbort_nonneg sha512 D.V P x keep
  have hFC0 := Claims.Recursion.PrFC_nonneg sha512 D.V CpDep P x keep M
  have hM0 : (0 : ℝ) ≤ 1 / ((M : ℝ) + 1) := by positivity
  by_cases hno : Claims.Flock.NoAcc sha512 D.V x
  · refine ⟨fun _ => 0, fun _ => le_rfl, ?_, fun ω C _ _ => ?_⟩
    · rw [Claims.Ex_const]
      exact add_nonneg (mul_nonneg (Nat.cast_nonneg _) hB0) (mul_nonneg (Nat.cast_nonneg _) (add_nonneg hFC0 hM0))
    · have h0 : Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P D.V x keep ω) = 0 := by
        unfold Claims.Recursion.accPrD
        refine (Claims.Pr_congr (q := fun _ => False) fun d => ⟨fun hd => ?_, False.elim⟩).trans Claims.Pr_false
        exact hno Ω P ω d.1 (Claims.Recursion.abortRun_acc_iff.1 hd).2
      calc Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P D.V x keep ω) = 0 := h0
        _ ≤ _ := add_nonneg (add_nonneg (hεt x) le_rfl) (Claims.Pr_nonneg _)
  obtain ⟨ins, spec, sts, st, sch, m, k0, l₀, l₁, rest, ext, hset, hst, hF, hF', ht, hsch, hall⟩ :=
    hall x hno
  have hVk : 0 < D.V.k x := (hk x).symm ▸ V_k_pos false x
  obtain ⟨m₀, hm₀⟩ := hrf x
  have hrun := Claims.Recursion.refuser_run sha512 P D.V x decCoin hcs hal hdf keep m₀ hm₀
  have hk0' : sch.k0 = k0 := congrArg Accounting.Schedule.k0 hF'
  have hlen : (l₀ :: l₁ :: rest).length = depL x + 1 := by
    have h1 := congrArg (fun S : Accounting.Schedule => S.levels.length) hF'
    simp only [schedOf, List.length_map, Array.length_toList] at h1
    rw [depL_eq hset hst hF]
    simp only [List.length_cons] at h1 ⊢
    omega
  have hlog : ∀ ℓ, depLogLen x ℓ = (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen := fun ℓ => by
    unfold depLogLen
    rw [depSetup_eq hset hst hF]
    exact congrArg (fun S => (lvOf S ℓ).logLen) hF'
  have hne : ∀ ω r q lanes d, ¬ (rdAtW D.V lc P x keep r (depL x) q lanes ω d).Nonempty := by
    rintro ω r q lanes d ⟨u, hu⟩
    obtain ⟨-, hacc, a, ha, -⟩ := mem_rdAtW.1 hu
    rw [htopL P x ω _ hacc hset hst hF r (depL x + 1) (by rw [depL_eq hset hst hF]; omega)] at ha
    simp at ha
  obtain ⟨E, hE⟩ : ∃ E : Ω → ℕ → ℕ → ℕ → (Fin (D.V.k x) → Coin) × A → Prop,
      E = fun ω r ℓ q d => (rdAtW D.V lc P x keep r ℓ q (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes ω d).Nonempty ∧
        rdAtW D.V lc P x keep r ℓ q (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes ω d ≠
          {Claims.Recursion.capPlurRd D.V x (forkRound spec st k0 r ℓ)
            (rdAtW D.V lc P x keep r ℓ q (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes ω) d} := ⟨_, rfl⟩
  have htop : ∀ ω r q, Claims.Pr (E ω r (depL x) q) = 0 := fun ω r q => by
    refine (Claims.Pr_congr (q := fun _ => False) fun d => ⟨fun h => ?_, False.elim⟩).trans Claims.Pr_false
    rw [hE] at h
    exact hne ω r q _ d h.1
  obtain ⟨hit, hhit⟩ : ∃ hit : Ω → Fin (CpDep.width x) → ℝ, hit = fun ω i =>
      Claims.Pr fun s : ((Fin (D.V.k x) → Coin) × A) × (Fin M → (Fin (D.V.k x) → Coin) × A) =>
        ∃ u v, (Claims.Recursion.FC sha512 D.V CpDep P x keep M i.pos).out (i, ω, s.1, s.2) = some (u, v) ∧
          Claims.Collision sha512 u v := ⟨_, rfl⟩
  have hcap : ∀ ω (i : Fin (CpDep.width x)), Claims.Pr (E ω ((Fintype.equivFin (DepIdx x)).symm i).1.val
      ((Fintype.equivFin (DepIdx x)).symm i).2.1.val ((Fintype.equivFin (DepIdx x)).symm i).2.2.val) ≤
        hit ω i + 1 / ((M : ℝ) + 1) := by
    intro ω i
    have h := hcap P x keep M i
      (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (((Fintype.equivFin (DepIdx x)).symm i).2.1.val + 1)).lanes ω
    rw [(CpDep_at hset hst hF i).1, hk0'] at h
    rw [hE, hhit]
    exact h
  have hpt : ∀ ω, (∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
      ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen), Claims.Pr (E ω r.val ℓ.val q.val)) ≤
        (∑ i, hit ω i) + (CpDep.width x : ℝ) * (1 / ((M : ℝ) + 1)) := fun ω =>
    calc (∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
          ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen), Claims.Pr (E ω r.val ℓ.val q.val))
        = ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.range (l₀ :: l₁ :: rest).length,
            ∑ q ∈ Finset.range (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).logLen), Claims.Pr (E ω r ℓ q) :=
          sum3_range (fun ℓ => (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).logLen) fun r ℓ q =>
            Claims.Pr (E ω r ℓ q)
      _ = ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.range (depL x),
            ∑ q ∈ Finset.range (2 ^ depLogLen x (ℓ + 1)), Claims.Pr (E ω r ℓ q) := by
          refine Finset.sum_congr rfl fun r _ => ?_
          rw [hlen, Finset.sum_range_succ, Finset.sum_eq_zero fun q _ => htop ω r q, add_zero]
          exact Finset.sum_congr rfl fun ℓ _ => by rw [hlog]
      _ = ∑ j : DepIdx x, Claims.Pr (E ω j.1.val j.2.1.val j.2.2.val) :=
          (sum_depIdx x fun r ℓ q => Claims.Pr (E ω r ℓ q)).symm
      _ = ∑ i : Fin (CpDep.width x), Claims.Pr (E ω ((Fintype.equivFin (DepIdx x)).symm i).1.val
            ((Fintype.equivFin (DepIdx x)).symm i).2.1.val ((Fintype.equivFin (DepIdx x)).symm i).2.2.val) :=
          (Equiv.sum_comp (Fintype.equivFin (DepIdx x)).symm fun j => Claims.Pr (E ω j.1.val j.2.1.val j.2.2.val)).symm
      _ ≤ ∑ i : Fin (CpDep.width x), (hit ω i + 1 / ((M : ℝ) + 1)) := Finset.sum_le_sum fun i _ => hcap ω i
      _ = _ := by rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hEt : (Claims.Ex fun ω => (Claims.Pr fun d : (Fin (D.V.k x) → Coin) × A =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P D.V x keep ω d).vLog) +
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
        ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen), Claims.Pr (E ω r.val ℓ.val q.val)) ≤
      D.V.k x * Claims.Recursion.PrBAbort sha512 D.V P x keep +
        CpDep.width x * (Claims.Recursion.PrFC sha512 D.V CpDep P x keep M + 1 / ((M : ℝ) + 1)) := by
    rw [Claims.Ex_add]
    refine add_le_add ?_ ?_
    · exact (le_of_eq (Claims.Ex_Pr fun ω d =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P D.V x keep ω d).vLog)).trans
          ((Claims.Recursion.Pr_one_le_PrBAbort sha512 D.V P x keep (fun ω d =>
            Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P D.V x keep ω d).vLog)
              fun _ _ h => h).trans (le_mul_of_one_le_left hB0 (Nat.one_le_cast.2 hVk)))
    · refine (Claims.Ex_mono hpt).trans (le_of_eq ?_)
      subst hhit
      rw [Claims.Ex_add, Claims.Recursion.Ex_capHits sha512 D.V CpDep P x keep M, Claims.Ex_const]
      ring
  refine ⟨fun ω => (Claims.Pr fun d : (Fin (D.V.k x) → Coin) × A =>
      Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P D.V x keep ω d).vLog) +
    ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
      ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen), Claims.Pr (E ω r.val ℓ.val q.val),
    fun ω => add_nonneg (Claims.Pr_nonneg _) (Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ =>
      Finset.sum_nonneg fun _ _ => Claims.Pr_nonneg _), hEt, fun ω C hC hg => ?_⟩
  have hb' : ∀ g : (Fin (D.V.k x) → Coin) → Coin × A,
      Claims.accPr sha512 D.V (Claims.Recursion.refuser P x (D.V.k x) decCoin keep m₀) x (ω, g) ≤
        D.εtable x + (Claims.Pr fun s : Fin (D.V.k x) → Coin =>
          (Claims.run sha512 (Claims.Recursion.refuser P x (D.V.k x) decCoin keep m₀) D.V x (ω, g) s).out = .acc ∧
            Claims.TiedColl sha512 (Claims.run sha512 (Claims.Recursion.refuser P x (D.V.k x) decCoin keep m₀) D.V x (ω, g) s).vLog) +
        ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
          ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen),
            Claims.Pr (fun c : Fin (D.V.k x) → Coin =>
              E ω r.val ℓ.val q.val (Claims.Recursion.refDraw g c)) +
        Claims.Pr fun r : Fin (D.V.k x) → Coin =>
          (Claims.run sha512 (Claims.Recursion.refuser P x (D.V.k x) decCoin keep m₀) D.V x (ω, g)
            r).out = .acc ∧
          ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
            Claims.Flock.lookCol (D.cols (Claims.run sha512
              (Claims.Recursion.refuser P x (D.V.k x) decCoin keep m₀) D.V x (ω, g) r).seen 0) p.val =
                some u ∧ u ≠ C p := by
    intro g
    subst hE
    refine hall (Claims.Recursion.refuser P x (D.V.k x) decCoin keep m₀) (ω, g) C hC hg
      (fun r ℓ c q => rdAtW D.V lc P x keep r.val ℓ.val q.val (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes ω
        (Claims.Recursion.refDraw g c))
      (fun r ℓ c q => Claims.Recursion.capPlurRd D.V x (forkRound spec st k0 r.val ℓ.val)
        (rdAtW D.V lc P x keep r.val ℓ.val q.val (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes ω)
        (Claims.Recursion.refDraw g c)) ?_ ?_
    · intro r ℓ c c' hcc
      funext q
      refine Claims.Recursion.capPlurRd_prefix _ _ _ _ fun i hi => ?_
      have hf : forkRound spec st k0 r.val ℓ.val < D.V.k x :=
        (hk x).symm ▸ forkRound_lt (tables_head_one ht) hsch r.isLt ℓ.isLt
      have hi' : i.val + 1 < D.V.k x := by omega
      simp only [Claims.Recursion.refDraw, Claims.Recursion.drawOf, Claims.Recursion.setLast, hi', ↓reduceIte]
      exact hcc i hi
    · intro r ℓ c hacc q a ha
      have hab := ((hrun ω g c).1).1 hacc
      rw [(hrun ω g c).2 hacc] at ha
      obtain ⟨hk, hr⟩ := Claims.Recursion.abortRun_acc_iff.1 hab
      rw [Claims.Recursion.abortRun_of_keep hk] at ha
      exact (mem_rdAtW (x := x) (keep := keep)).2 ⟨hk, hr, a, ha, rfl⟩
  have hL : (Claims.Ex fun g : (Fin (D.V.k x) → Coin) → Coin × A =>
      Claims.accPr sha512 D.V (Claims.Recursion.refuser P x (D.V.k x) decCoin keep m₀) x (ω, g)) =
        Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P D.V x keep ω) := by
    unfold Claims.accPr Claims.Recursion.accPrD
    rw [← Claims.Recursion.Pr_refDraw hVk]
    exact congrArg Claims.Ex (funext fun g => Claims.Pr_congr fun r => (hrun ω g r).1)
  have hE1 : (Claims.Ex fun g : (Fin (D.V.k x) → Coin) → Coin × A =>
      Claims.Pr fun s : Fin (D.V.k x) → Coin =>
        (Claims.run sha512 (Claims.Recursion.refuser P x (D.V.k x) decCoin keep m₀) D.V x (ω, g) s).out = .acc ∧
            Claims.TiedColl sha512 (Claims.run sha512 (Claims.Recursion.refuser P x (D.V.k x) decCoin keep m₀) D.V x (ω, g) s).vLog) ≤
      Claims.Pr fun d : (Fin (D.V.k x) → Coin) × A =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P D.V x keep ω d).vLog := by
    refine (Claims.Ex_mono fun g => Claims.Pr_mono fun s hs => ?_).trans
      (le_of_eq (Claims.Recursion.Pr_refDraw hVk fun d =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P D.V x keep ω d).vLog))
    have h2 := hs.2
    rw [(hrun ω g s).2 hs.1] at h2
    exact h2
  have hS : (Claims.Ex fun g : (Fin (D.V.k x) → Coin) → Coin × A =>
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
        ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen),
          Claims.Pr (fun c : Fin (D.V.k x) → Coin => E ω r.val ℓ.val q.val (Claims.Recursion.refDraw g c))) =
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
        ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen), Claims.Pr (E ω r.val ℓ.val q.val) := by
    rw [Claims.Bridge.Ex_sum_univ]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [Claims.Bridge.Ex_sum_univ]
    refine Finset.sum_congr rfl fun ℓ _ => ?_
    rw [Claims.Bridge.Ex_sum_univ]
    exact Finset.sum_congr rfl fun q _ => Claims.Recursion.Pr_refDraw hVk (E ω r.val ℓ.val q.val)
  have hO : (Claims.Ex fun g : (Fin (D.V.k x) → Coin) → Coin × A =>
      Claims.Pr fun r : Fin (D.V.k x) → Coin =>
        (Claims.run sha512 (Claims.Recursion.refuser P x (D.V.k x) decCoin keep m₀) D.V x (ω, g)
          r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (D.cols (Claims.run sha512
            (Claims.Recursion.refuser P x (D.V.k x) decCoin keep m₀) D.V x (ω, g) r).seen 0) p.val =
              some u ∧ u ≠ C p) =
      Claims.Pr fun d : (Fin (D.V.k x) → Coin) × A =>
        (Claims.Recursion.abortRun sha512 P D.V x keep ω d).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (D.cols (Claims.Recursion.abortRun sha512 P D.V x keep ω d).seen 0) p.val =
            some u ∧ u ≠ C p := by
    rw [← Claims.Recursion.Pr_refDraw hVk]
    refine congrArg Claims.Ex (funext fun g => Claims.Pr_congr fun r => ?_)
    constructor
    · rintro ⟨ha, rest⟩
      rw [(hrun ω g r).2 ha] at rest
      exact ⟨(hrun ω g r).1.1 ha, rest⟩
    · rintro ⟨ha, rest⟩
      have ha' := (hrun ω g r).1.2 ha
      rw [← (hrun ω g r).2 ha'] at rest
      exact ⟨ha', rest⟩
  have hEx := Claims.Ex_mono hb'
  rw [Claims.Ex_add, Claims.Ex_add, Claims.Ex_add, Claims.Ex_const, hL, hO, hS] at hEx
  have key : Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P D.V x keep ω) ≤ D.εtable x +
      ((Claims.Pr fun d : (Fin (D.V.k x) → Coin) × A =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P D.V x keep ω d).vLog) +
        ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
          ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen),
            Claims.Pr (E ω r.val ℓ.val q.val)) +
      Claims.Pr fun d : (Fin (D.V.k x) → Coin) × A =>
        (Claims.Recursion.abortRun sha512 P D.V x keep ω d).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (D.cols (Claims.Recursion.abortRun sha512 P D.V x keep ω d).seen 0) p.val =
            some u ∧ u ≠ C p := by
    linarith
  exact key

/-- **The form at `VK keep`**: `tableBoundAbortLin_of_rdAll` at `deployPlainK keep`, each input from the K chain. -/
theorem tableBoundAbortLinK_of_rdAll (keepV : Bool → Bool) :
    Claims.Recursion.TableBoundAbortLin sha512 (deployPlainK keepV Readable.relation) (wordLen false) (wide false)
      CpDep :=
  tableBoundAbortLin_of_rdAll (deployPlainK keepV Readable.relation) (fun _ => plainRawLevelCols) (fun _ => rfl)
    (fun x => εtable_nonneg false x) (coinsShown_VK keepV sha512) (accLast_VK keepV) (decideFree_VK keepV sha512)
    (refusable_VK keepV sha512) (fun _ hno => plain_accPr_le_rdAllK keepV hno)
    (fun _ _ _ _ h _ _ _ _ _ hs hst hsch r _ hℓ => plainRawLevelCols_topK keepV h hs hst hsch r hℓ)
    (fun P x keep M i lanes ω => Pr_rdAt_le_capHitK keepV P x keep M i lanes ω)

/-! ## At the deployed reader `VR rd` -/

/-- **`actR` asks a coin only where the plain verifier does.** -/
theorem actR_coin {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {c : Coin} (h : actR rd x as cs = .coin c) : act false x as cs = .coin c := by
  unfold actR at h
  split at h
  · rename_i c' hc'
    cases h
    exact hc'
  · split at h <;> cases h

/-- **`nextR` answers a message only where the plain verifier answers it.** -/
theorem next_msg_of_nextR (rd : Claims.Msg → Option FinalP) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {m : Claims.Msg} (h : ((nextR rd x as cs).eval H).1 = .msg m) :
    ((next false x as cs).eval H).1 = .msg m := by
  cases hact : actR rd x as cs with
  | rej => simp [nextR, hact, Claims.OC.eval] at h
  | coin c =>
    have ha := actR_coin hact
    rw [nextR_coin rd H x as cs c ha] at h
    rw [next_coin H ha]
    exact h
  | decide s spec sts link c₁ mids f =>
    rw [nextR_decide hact] at h
    exact absurd h (Claims.Bridge.outOf_ne_msg _ _ _)

/-- **`VR rd` shows each round's coin**: it answers a message only where the plain verifier does
(`next_msg_of_nextR`). -/
theorem coinsShown_VR (rd : Claims.Msg → Option FinalP) (H : Claims.Msg → Claims.Msg) :
    Claims.Recursion.CoinsShown H (VR rd) decCoin :=
  fun x as cs c m hl h => coinsShown_V H false x as cs c m hl (next_msg_of_nextR (cs := cs ++ [c]) rd H h)

/-- **`VR rd` never accepts `[0]` last**: past a coin it is a coin request, and with no coin `decR` reads none. -/
theorem refusable_VR (rd : Claims.Msg → Option FinalP) (H : Claims.Msg → Claims.Msg) :
    Claims.Recursion.Refusable H (VR rd) := fun x =>
  ⟨[0], fun as cs => by
    show ((nextR rd x (as ++ [[0]]) cs).eval H).1 ≠ .acc
    cases hact : actR rd x (as ++ [[0]]) cs with
    | rej => simp [nextR, hact, Claims.OC.eval]
    | coin c => simp [nextR, hact, Claims.OC.eval]
    | decide s spec sts link c₁ mids f =>
      exfalso
      obtain ⟨⟨s', spec', sts'⟩, c₁', -, -, -, -, -, hv, hc1, -⟩ := decR_some (actR_decide hact)
      rcases hgl : cs.getLast? with _ | cl
      · rw [List.getLast?_eq_none_iff.1 hgl] at hc1
        cases hc1
      · have hcoin : act false x (as ++ [[0]]) cs = .coin cl := by
          unfold act
          rw [hv]
          simp only [List.getLast?_concat, hgl]
        unfold actR at hact
        rw [hcoin] at hact
        cases hact⟩

/-- **The table bound at any plain record whose verifier is `VR rd`**, from the reader's bound, top and cap facts at
`lc` and `VR rd`'s `AccLast` and `DecideFree`: `tableBoundAbortLin_of_rdAll` with `VR rd`'s coins and refusal
(`coinsShown_VR`, `refusable_VR`). -/
theorem tableBoundAbortLin_VR {T : Type} (rd : Claims.Msg → Option FinalP) (D : Claims.Flock.Deploy Coin (List 𝔽) T)
    (hV : D.V = VR rd) (hεt : ∀ x, 0 ≤ D.εtable x) (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽))
    (hal : Claims.Recursion.AccLast sha512 (VR rd)) (hdf : Claims.Recursion.DecideFree sha512 (VR rd))
    (hall : RdAllAt D lc) (htopL : LevelTopAt (VR rd) lc) (hcap : CapHitAt (VR rd) lc) :
    Claims.Recursion.TableBoundAbortLin sha512 D (wordLen false) (wide false) CpDep :=
  tableBoundAbortLin_of_rdAll D lc (fun x => by rw [hV]; rfl) hεt (hV ▸ coinsShown_VR rd sha512) (hV ▸ hal)
    (hV ▸ hdf) (hV ▸ refusable_VR rd sha512) hall (hV ▸ htopL) (hV ▸ hcap)

end FlockDeploy.KS
