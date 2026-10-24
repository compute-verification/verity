import Proofs.Flock.Claims.KS.PlainRdRun
import Proofs.Flock.Claims.KS.Rounds
import Proofs.Flock.Claims.KS.PlainBound
import Proofs.Flock.Claims.KS.OffSim
import Proofs.Flock.Claims.Recursion.FlockKSLin
import Proofs.Flock.Claims.Recursion.FlockKSAcc
import Proofs.Flock.Claims.Recursion.FlockKSRecord
import Proofs.Flock.Claims.KS.PlainRdRunK

/-!
# `KSFlockLin` at the deployed plain record

The plain record's KS chain with the cap terms linear: each cap slot's term is charged to `FC` at the deployed caps
`CpDep` (`Pr_rdAt_le_capHit`) instead of a square root, and the one-run event to `BAbort`.

* `tableBoundAbortLin_mapCols`: `TableBoundAbortLin` through `f`, by padding back, as `tableBoundθOnAcc_mapCols`.
* `plain_tableBoundAbortLin`: `TableBoundAbortLin` at the untrimmed deployed plain instance over full-width tables, at
  `CpDep`. At a statement some run accepts it is `plain_accPr_le_rdAll` at the refuser's runs, with the read sets
  `rdAt` (what each run reads at a cap position) and the tables `capPlurRd` (the plurality single reading after the
  slot's fork round), averaged over the refuser's last coin (`Pr_refDraw`). The term on each tape is the one-run
  event's chance at the abort law and each cap position's read-set chance, whose average is at most
  `k·Pr[BAbort] + W·(Pr[FC] + 1/(M+1))` (`Pr_one_le_PrBAbort`, `Pr_rdAt_le_capHit`, `Ex_capHits`): the schedule's top
  slot reads nothing (`plainRawLevelCols_top`), and the others are `CpDep`'s positions.
* `plain_KSFlockLin`: `KSFlockLin` at SHA-512 at the trimmed record `deployPlainTrim` and `CpDep`, from
  `KSFlockLin_of_tableBoundAbortLin` with the record's code facts and numbers (`plain_KSAbort`'s).
* `plain_tableBoundAbortLinK`: `plain_tableBoundAbortLin` at `deployPlainK keep`, from `plain_accPr_le_rdAllK`.
-/

namespace Claims.Recursion

open Claims Claims.Flock

/-- **`TableBoundAbortLin` through `f`, by padding back**: at a table of columns through `f` of ones `Ok` admits, from
the bound at the table they came from, with the same term, when `f` keeps `0`, accepted runs open only columns `Ok`
admits, and `f` is one to one on them (`tableBoundθOnAcc_mapCols`). -/
theorem tableBoundAbortLin_mapCols {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (f : Col → Col)
    (hf : f 0 = 0) (Ok : Msg → Col → Prop) (Cp : Caps Coin) (hok : ColsOk H D N Ok)
    (hinj : ∀ x u v, Ok x u → Ok x v → f u = f v → u = v) (htab : TableBoundAbortLin H D N Ok Cp) :
    TableBoundAbortLin H (D.mapCols f) N (OkThrough f Ok) Cp := by
  intro x Ω _ _ P A _ _ keep M
  obtain ⟨t, ht0, hEt, hb⟩ := htab x Ω P A keep M
  refine ⟨t, ht0, hEt, fun ω C' hC' hg' => ?_⟩
  choose C hC hfC using hC'
  have hg : ¬ GoodTable D N x C := fun h => hg' (by
    have := goodTable_mapCols D N x f hf C h
    rwa [show (fun p => f (C p)) = C' from funext hfC] at this)
  refine (hb ω C hC hg).trans (add_le_add le_rfl (Pr_mono fun d ⟨ha, p, u, hu, hne⟩ => ⟨ha, p, f u, ?_, ?_⟩))
  · show lookCol ((D.cols (abortRun H P D.V x keep ω d).seen 0).map fun q => (q.1, f q.2)) p.val = some (f u)
    rw [lookCol_map, hu]
    rfl
  · rw [← hfC p]
    obtain ⟨hk, hr⟩ := abortRun_acc_iff.1 ha
    rw [abortRun_of_keep hk] at hu
    exact fun he => hne (hinj x u (C p) (hok x Ω P ω d.1 p.val u hr p.isLt hu) (hC p) he)

end Claims.Recursion

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- A triple sum over `Fin`s, with the last one's size read from the middle index, as one over `range`s. -/
theorem sum3_range {L : ℕ} (lg : ℕ → ℕ) (f : ℕ → ℕ → ℕ → ℝ) :
    (∑ r : Fin 2, ∑ ℓ : Fin L, ∑ q : Fin (2 ^ lg ℓ.val), f r.val ℓ.val q.val) =
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.range L, ∑ q ∈ Finset.range (2 ^ lg ℓ), f r ℓ q :=
  calc _ = ∑ r ∈ Finset.range 2, ∑ ℓ : Fin L, ∑ q : Fin (2 ^ lg ℓ.val), f r ℓ.val q.val :=
        Fin.sum_univ_eq_sum_range (fun r => ∑ ℓ : Fin L, ∑ q : Fin (2 ^ lg ℓ.val), f r ℓ.val q.val) 2
    _ = _ := Finset.sum_congr rfl fun r _ =>
      calc _ = ∑ ℓ ∈ Finset.range L, ∑ q : Fin (2 ^ lg ℓ), f r ℓ q.val :=
            Fin.sum_univ_eq_sum_range (fun ℓ => ∑ q : Fin (2 ^ lg ℓ), f r ℓ q.val) L
        _ = _ := Finset.sum_congr rfl fun ℓ _ => Fin.sum_univ_eq_sum_range (f r ℓ) _

/-- A sum over `CpDep`'s positions of `x`, as one over reps, slots and level positions. -/
theorem sum_depIdx (x : Claims.Msg) (f : ℕ → ℕ → ℕ → ℝ) :
    (∑ j : DepIdx x, f j.1.val j.2.1.val j.2.2.val) =
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.range (depL x), ∑ q ∈ Finset.range (2 ^ depLogLen x (ℓ + 1)), f r ℓ q := by
  rw [Fintype.sum_sigma]
  refine (Finset.sum_congr rfl fun r _ => Fintype.sum_sigma _).trans ?_
  exact sum3_range (fun ℓ => depLogLen x (ℓ + 1)) f

open Classical in
/-- **The deployed plain record's table bound at the abort law, linearly, at `CpDep`**, at the untrimmed instance over
full-width tables: where no run accepts, the term `0`; elsewhere, on each tape, the chance that the run as it aborts
ties a collision among its own calls and, at each cap position, that it reads there anything but the plurality single
reading after the slot's fork round. The first averages to at most `Pr[BAbort]` (`Pr_one_le_PrBAbort`), and each
position's to at most its `FC` hits and `1/(M+1)` (`Pr_rdAt_le_capHit`). The bound itself is `plain_accPr_le_rdAll`
at the refuser's runs with those read sets and tables, averaged over the refuser's last coin (`Pr_refDraw`). -/
theorem plain_tableBoundAbortLin :
    Claims.Recursion.TableBoundAbortLin sha512 (deployPlain Readable.relation) (wordLen false) (wide false) CpDep := by
  classical
  intro x Ω _ _ P A _ _ keep M
  have hB0 := Claims.Recursion.PrBAbort_nonneg sha512 (V false) P x keep
  have hFC0 := Claims.Recursion.PrFC_nonneg sha512 (V false) CpDep P x keep M
  have hM0 : (0 : ℝ) ≤ 1 / ((M : ℝ) + 1) := by positivity
  by_cases hno : Claims.Flock.NoAcc sha512 (V false) x
  · refine ⟨fun _ => 0, fun _ => le_rfl, ?_, fun ω C _ _ => ?_⟩
    · rw [Claims.Ex_const]
      exact add_nonneg (mul_nonneg (Nat.cast_nonneg _) hB0) (mul_nonneg (Nat.cast_nonneg _) (add_nonneg hFC0 hM0))
    · have h0 : Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P (V false) x keep ω) = 0 := by
        unfold Claims.Recursion.accPrD
        refine (Claims.Pr_congr (q := fun _ => False) fun d => ⟨fun hd => ?_, False.elim⟩).trans Claims.Pr_false
        exact hno Ω P ω d.1 (Claims.Recursion.abortRun_acc_iff.1 hd).2
      calc Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P (V false) x keep ω) = 0 := h0
        _ ≤ _ := add_nonneg (add_nonneg (εtable_nonneg false x) le_rfl) (Claims.Pr_nonneg _)
  obtain ⟨ins, spec, sts, st, sch, m, k0, l₀, l₁, rest, ext, hset, hst, hF, hF', ht, hsch, hall⟩ :=
    plain_accPr_le_rdAll hno
  have hVk := V_k_pos false x
  obtain ⟨m₀, hm₀⟩ := refusable_V sha512 false x
  have hrun := Claims.Recursion.refuser_run sha512 P (V false) x decCoin (coinsShown_V sha512 false)
    (accLast_V_sha512 false) (decideFree_V sha512 false) keep m₀ hm₀
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
  have hne : ∀ ω r q lanes d, ¬ (rdAt P x keep r (depL x) q lanes ω d).Nonempty := by
    rintro ω r q lanes d ⟨u, hu⟩
    obtain ⟨-, hacc, a, ha, -⟩ := mem_rdAt.1 hu
    rw [plainRawLevelCols_top (ℓ := depL x + 1) hacc hset hst hF r (by rw [depL_eq hset hst hF]; omega)] at ha
    simp at ha
  obtain ⟨E, hE⟩ : ∃ E : Ω → ℕ → ℕ → ℕ → (Fin ((V false).k x) → Coin) × A → Prop,
      E = fun ω r ℓ q d => (rdAt P x keep r ℓ q (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes ω d).Nonempty ∧
        rdAt P x keep r ℓ q (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes ω d ≠
          {Claims.Recursion.capPlurRd (V false) x (forkRound spec st k0 r ℓ)
            (rdAt P x keep r ℓ q (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes ω) d} := ⟨_, rfl⟩
  have htop : ∀ ω r q, Claims.Pr (E ω r (depL x) q) = 0 := fun ω r q => by
    refine (Claims.Pr_congr (q := fun _ => False) fun d => ⟨fun h => ?_, False.elim⟩).trans Claims.Pr_false
    rw [hE] at h
    exact hne ω r q _ d h.1
  obtain ⟨hit, hhit⟩ : ∃ hit : Ω → Fin (CpDep.width x) → ℝ, hit = fun ω i =>
      Claims.Pr fun s : ((Fin ((V false).k x) → Coin) × A) × (Fin M → (Fin ((V false).k x) → Coin) × A) =>
        ∃ u v, (Claims.Recursion.FC sha512 (V false) CpDep P x keep M i.pos).out (i, ω, s.1, s.2) = some (u, v) ∧
          Claims.Collision sha512 u v := ⟨_, rfl⟩
  have hcap : ∀ ω (i : Fin (CpDep.width x)), Claims.Pr (E ω ((Fintype.equivFin (DepIdx x)).symm i).1.val
      ((Fintype.equivFin (DepIdx x)).symm i).2.1.val ((Fintype.equivFin (DepIdx x)).symm i).2.2.val) ≤
        hit ω i + 1 / ((M : ℝ) + 1) := by
    intro ω i
    have h := Pr_rdAt_le_capHit P x keep M i
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
  have hEt : (Claims.Ex fun ω => (Claims.Pr fun d : (Fin ((V false).k x) → Coin) × A =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).vLog) +
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
        ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen), Claims.Pr (E ω r.val ℓ.val q.val)) ≤
      (V false).k x * Claims.Recursion.PrBAbort sha512 (V false) P x keep +
        CpDep.width x * (Claims.Recursion.PrFC sha512 (V false) CpDep P x keep M + 1 / ((M : ℝ) + 1)) := by
    rw [Claims.Ex_add]
    refine add_le_add ?_ ?_
    · exact (le_of_eq (Claims.Ex_Pr fun ω d =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).vLog)).trans
          ((Claims.Recursion.Pr_one_le_PrBAbort sha512 (V false) P x keep (fun ω d =>
            Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).vLog)
              fun _ _ h => h).trans (le_mul_of_one_le_left hB0 (Nat.one_le_cast.2 hVk)))
    · refine (Claims.Ex_mono hpt).trans (le_of_eq ?_)
      subst hhit
      rw [Claims.Ex_add, Claims.Recursion.Ex_capHits sha512 (V false) CpDep P x keep M, Claims.Ex_const]
      ring
  refine ⟨fun ω => (Claims.Pr fun d : (Fin ((V false).k x) → Coin) × A =>
      Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).vLog) +
    ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
      ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen), Claims.Pr (E ω r.val ℓ.val q.val),
    fun ω => add_nonneg (Claims.Pr_nonneg _) (Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ =>
      Finset.sum_nonneg fun _ _ => Claims.Pr_nonneg _), hEt, fun ω C hC hg => ?_⟩
  have hb' : ∀ g : (Fin ((V false).k x) → Coin) → Coin × A,
      Claims.accPr sha512 (V false) (Claims.Recursion.refuser P x ((V false).k x) decCoin keep m₀) x (ω, g) ≤
        εtable false x + (Claims.Pr fun s : Fin ((V false).k x) → Coin =>
          plainEv₁ (Claims.Recursion.refuser P x ((V false).k x) decCoin keep m₀) x (ω, g) s) +
        ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
          ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen),
            Claims.Pr (fun c : Fin ((V false).k x) → Coin =>
              E ω r.val ℓ.val q.val (Claims.Recursion.refDraw g c)) +
        Claims.Pr fun r : Fin ((V false).k x) → Coin =>
          (Claims.run sha512 (Claims.Recursion.refuser P x ((V false).k x) decCoin keep m₀) (V false) x (ω, g)
            r).out = .acc ∧
          ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
            Claims.Flock.lookCol (cols false (Claims.run sha512
              (Claims.Recursion.refuser P x ((V false).k x) decCoin keep m₀) (V false) x (ω, g) r).seen 0) p.val =
                some u ∧ u ≠ C p := by
    intro g
    subst hE
    refine hall (Claims.Recursion.refuser P x ((V false).k x) decCoin keep m₀) (ω, g) C hC hg
      (fun r ℓ c q => rdAt P x keep r.val ℓ.val q.val (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes ω
        (Claims.Recursion.refDraw g c))
      (fun r ℓ c q => Claims.Recursion.capPlurRd (V false) x (forkRound spec st k0 r.val ℓ.val)
        (rdAt P x keep r.val ℓ.val q.val (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes ω)
        (Claims.Recursion.refDraw g c)) ?_ ?_
    · intro r ℓ c c' hcc
      funext q
      refine Claims.Recursion.capPlurRd_prefix _ _ _ _ fun i hi => ?_
      have hf : forkRound spec st k0 r.val ℓ.val < (V false).k x :=
        forkRound_lt (tables_head_one ht) hsch r.isLt ℓ.isLt
      have hi' : i.val + 1 < (V false).k x := by omega
      simp only [Claims.Recursion.refDraw, Claims.Recursion.drawOf, Claims.Recursion.setLast, hi', ↓reduceIte]
      exact hcc i hi
    · intro r ℓ c hacc q a ha
      have hab := ((hrun ω g c).1).1 hacc
      rw [(hrun ω g c).2 hacc] at ha
      obtain ⟨hk, hr⟩ := Claims.Recursion.abortRun_acc_iff.1 hab
      rw [Claims.Recursion.abortRun_of_keep hk] at ha
      exact mem_rdAt.2 ⟨hk, hr, a, ha, rfl⟩
  have hL : (Claims.Ex fun g : (Fin ((V false).k x) → Coin) → Coin × A =>
      Claims.accPr sha512 (V false) (Claims.Recursion.refuser P x ((V false).k x) decCoin keep m₀) x (ω, g)) =
        Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P (V false) x keep ω) := by
    unfold Claims.accPr Claims.Recursion.accPrD
    rw [← Claims.Recursion.Pr_refDraw hVk]
    exact congrArg Claims.Ex (funext fun g => Claims.Pr_congr fun r => (hrun ω g r).1)
  have hE1 : (Claims.Ex fun g : (Fin ((V false).k x) → Coin) → Coin × A =>
      Claims.Pr fun s : Fin ((V false).k x) → Coin =>
        plainEv₁ (Claims.Recursion.refuser P x ((V false).k x) decCoin keep m₀) x (ω, g) s) ≤
      Claims.Pr fun d : (Fin ((V false).k x) → Coin) × A =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).vLog := by
    refine (Claims.Ex_mono fun g => Claims.Pr_mono fun s hs => ?_).trans
      (le_of_eq (Claims.Recursion.Pr_refDraw hVk fun d =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).vLog))
    have h2 := hs.2
    rw [(hrun ω g s).2 hs.1] at h2
    exact h2
  have hS : (Claims.Ex fun g : (Fin ((V false).k x) → Coin) → Coin × A =>
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
        ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen),
          Claims.Pr (fun c : Fin ((V false).k x) → Coin => E ω r.val ℓ.val q.val (Claims.Recursion.refDraw g c))) =
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
        ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen), Claims.Pr (E ω r.val ℓ.val q.val) := by
    rw [Claims.Bridge.Ex_sum_univ]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [Claims.Bridge.Ex_sum_univ]
    refine Finset.sum_congr rfl fun ℓ _ => ?_
    rw [Claims.Bridge.Ex_sum_univ]
    exact Finset.sum_congr rfl fun q _ => Claims.Recursion.Pr_refDraw hVk (E ω r.val ℓ.val q.val)
  have hO : (Claims.Ex fun g : (Fin ((V false).k x) → Coin) → Coin × A =>
      Claims.Pr fun r : Fin ((V false).k x) → Coin =>
        (Claims.run sha512 (Claims.Recursion.refuser P x ((V false).k x) decCoin keep m₀) (V false) x (ω, g)
          r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512
            (Claims.Recursion.refuser P x ((V false).k x) decCoin keep m₀) (V false) x (ω, g) r).seen 0) p.val =
              some u ∧ u ≠ C p) =
      Claims.Pr fun d : (Fin ((V false).k x) → Coin) × A =>
        (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).seen 0) p.val =
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
  have key : Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P (V false) x keep ω) ≤ εtable false x +
      ((Claims.Pr fun d : (Fin ((V false).k x) → Coin) × A =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).vLog) +
        ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
          ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen),
            Claims.Pr (E ω r.val ℓ.val q.val)) +
      Claims.Pr fun d : (Fin ((V false).k x) → Coin) × A =>
        (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).seen 0) p.val =
            some u ∧ u ≠ C p := by
    linarith
  exact key

/-- **`KSFlockLin` at the deployed plain record**, at SHA-512, at the trimmed record the claims read and the deployed
caps `CpDep`: `KSFlockLin_of_tableBoundAbortLin` from `plain_tableBoundAbortLin` carried to the trimmed columns
(`tableBoundAbortLin_mapCols`), with the record's numbers, one table wherever some run accepts (`plain_hone`), and
`ColsOk` and `ConflictTied` at the trimmed full-width columns. -/
theorem plain_KSFlockLin : Claims.Recursion.KSFlockLin sha512 (deployPlainTrim Readable.relation) CpDep :=
  Claims.Recursion.KSFlockLin_of_tableBoundAbortLin sha512 (deployPlainTrim Readable.relation) (wordLen false)
    (trimWide false) CpDep (wordLen_le_N₀ false) (fun x => εtable_nonneg false x) (fun _ => cap_pos)
    (plain_kSplits Readable.relation (plain_hone Readable.relation)) (plain_hone Readable.relation)
    (trimWide_nil false)
    (Claims.Flock.colsOk_mapCols sha512 (deployPlain Readable.relation) (wordLen false) trimCol (wide false)
      (plain_colsOk sha512 Readable.relation))
    (Claims.Flock.conflictTied_mapCols sha512 (deployPlain Readable.relation) (wordLen false) trimCol
      (plain_conflictTied Readable.relation))
    (Claims.Recursion.tableBoundAbortLin_mapCols sha512 (deployPlain Readable.relation) (wordLen false) trimCol rfl
      (wide false) CpDep (plain_colsOk sha512 Readable.relation) (wide_trimCol_inj false) plain_tableBoundAbortLin)

/-! ## At `VK keep` -/

open Classical in
/-- `plain_tableBoundAbortLin` at `VK keep`. -/
theorem plain_tableBoundAbortLinK (keepV : Bool → Bool) :
    Claims.Recursion.TableBoundAbortLin sha512 (deployPlainK keepV Readable.relation) (wordLen false) (wide false) CpDep := by
  classical
  intro x Ω _ _ P A _ _ keep M
  have hB0 := Claims.Recursion.PrBAbort_nonneg sha512 (VK keepV) P x keep
  have hFC0 := Claims.Recursion.PrFC_nonneg sha512 (VK keepV) CpDep P x keep M
  have hM0 : (0 : ℝ) ≤ 1 / ((M : ℝ) + 1) := by positivity
  by_cases hno : Claims.Flock.NoAcc sha512 (VK keepV) x
  · refine ⟨fun _ => 0, fun _ => le_rfl, ?_, fun ω C _ _ => ?_⟩
    · rw [Claims.Ex_const]
      exact add_nonneg (mul_nonneg (Nat.cast_nonneg _) hB0) (mul_nonneg (Nat.cast_nonneg _) (add_nonneg hFC0 hM0))
    · have h0 : Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω) = 0 := by
        unfold Claims.Recursion.accPrD
        refine (Claims.Pr_congr (q := fun _ => False) fun d => ⟨fun hd => ?_, False.elim⟩).trans Claims.Pr_false
        exact hno Ω P ω d.1 (Claims.Recursion.abortRun_acc_iff.1 hd).2
      calc Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω) = 0 := h0
        _ ≤ _ := add_nonneg (add_nonneg (εtable_nonneg false x) le_rfl) (Claims.Pr_nonneg _)
  obtain ⟨ins, spec, sts, st, sch, m, k0, l₀, l₁, rest, ext, hset, hst, hF, hF', ht, hsch, hall⟩ :=
    plain_accPr_le_rdAllK keepV hno
  have hVk := VK_k_pos keepV x
  obtain ⟨m₀, hm₀⟩ := refusable_VK keepV sha512 x
  have hrun := Claims.Recursion.refuser_run sha512 P (VK keepV) x decCoin (coinsShown_VK keepV sha512)
    (accLast_VK keepV) (decideFree_VK keepV sha512) keep m₀ hm₀
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
  have hne : ∀ ω r q lanes d, ¬ (rdAtK keepV P x keep r (depL x) q lanes ω d).Nonempty := by
    rintro ω r q lanes d ⟨u, hu⟩
    obtain ⟨-, hacc, a, ha, -⟩ := (mem_rdAtK keepV).1 hu
    rw [plainRawLevelCols_topK keepV (ℓ := depL x + 1) hacc hset hst hF r (by rw [depL_eq hset hst hF]; omega)] at ha
    simp at ha
  obtain ⟨E, hE⟩ : ∃ E : Ω → ℕ → ℕ → ℕ → (Fin ((VK keepV).k x) → Coin) × A → Prop,
      E = fun ω r ℓ q d => (rdAtK keepV P x keep r ℓ q (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes ω d).Nonempty ∧
        rdAtK keepV P x keep r ℓ q (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes ω d ≠
          {Claims.Recursion.capPlurRd (VK keepV) x (forkRound spec st k0 r ℓ)
            (rdAtK keepV P x keep r ℓ q (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes ω) d} := ⟨_, rfl⟩
  have htop : ∀ ω r q, Claims.Pr (E ω r (depL x) q) = 0 := fun ω r q => by
    refine (Claims.Pr_congr (q := fun _ => False) fun d => ⟨fun h => ?_, False.elim⟩).trans Claims.Pr_false
    rw [hE] at h
    exact hne ω r q _ d h.1
  obtain ⟨hit, hhit⟩ : ∃ hit : Ω → Fin (CpDep.width x) → ℝ, hit = fun ω i =>
      Claims.Pr fun s : ((Fin ((VK keepV).k x) → Coin) × A) × (Fin M → (Fin ((VK keepV).k x) → Coin) × A) =>
        ∃ u v, (Claims.Recursion.FC sha512 (VK keepV) CpDep P x keep M i.pos).out (i, ω, s.1, s.2) = some (u, v) ∧
          Claims.Collision sha512 u v := ⟨_, rfl⟩
  have hcap : ∀ ω (i : Fin (CpDep.width x)), Claims.Pr (E ω ((Fintype.equivFin (DepIdx x)).symm i).1.val
      ((Fintype.equivFin (DepIdx x)).symm i).2.1.val ((Fintype.equivFin (DepIdx x)).symm i).2.2.val) ≤
        hit ω i + 1 / ((M : ℝ) + 1) := by
    intro ω i
    have h := Pr_rdAt_le_capHitK keepV P x keep M i
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
  have hEt : (Claims.Ex fun ω => (Claims.Pr fun d : (Fin ((VK keepV).k x) → Coin) × A =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).vLog) +
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
        ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen), Claims.Pr (E ω r.val ℓ.val q.val)) ≤
      (VK keepV).k x * Claims.Recursion.PrBAbort sha512 (VK keepV) P x keep +
        CpDep.width x * (Claims.Recursion.PrFC sha512 (VK keepV) CpDep P x keep M + 1 / ((M : ℝ) + 1)) := by
    rw [Claims.Ex_add]
    refine add_le_add ?_ ?_
    · exact (le_of_eq (Claims.Ex_Pr fun ω d =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).vLog)).trans
          ((Claims.Recursion.Pr_one_le_PrBAbort sha512 (VK keepV) P x keep (fun ω d =>
            Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).vLog)
              fun _ _ h => h).trans (le_mul_of_one_le_left hB0 (Nat.one_le_cast.2 hVk)))
    · refine (Claims.Ex_mono hpt).trans (le_of_eq ?_)
      subst hhit
      rw [Claims.Ex_add, Claims.Recursion.Ex_capHits sha512 (VK keepV) CpDep P x keep M, Claims.Ex_const]
      ring
  refine ⟨fun ω => (Claims.Pr fun d : (Fin ((VK keepV).k x) → Coin) × A =>
      Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).vLog) +
    ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
      ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen), Claims.Pr (E ω r.val ℓ.val q.val),
    fun ω => add_nonneg (Claims.Pr_nonneg _) (Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ =>
      Finset.sum_nonneg fun _ _ => Claims.Pr_nonneg _), hEt, fun ω C hC hg => ?_⟩
  have hb' : ∀ g : (Fin ((VK keepV).k x) → Coin) → Coin × A,
      Claims.accPr sha512 (VK keepV) (Claims.Recursion.refuser P x ((VK keepV).k x) decCoin keep m₀) x (ω, g) ≤
        εtable false x + (Claims.Pr fun s : Fin ((VK keepV).k x) → Coin =>
          plainEv₁K keepV (Claims.Recursion.refuser P x ((VK keepV).k x) decCoin keep m₀) x (ω, g) s) +
        ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
          ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen),
            Claims.Pr (fun c : Fin ((VK keepV).k x) → Coin =>
              E ω r.val ℓ.val q.val (Claims.Recursion.refDraw g c)) +
        Claims.Pr fun r : Fin ((VK keepV).k x) → Coin =>
          (Claims.run sha512 (Claims.Recursion.refuser P x ((VK keepV).k x) decCoin keep m₀) (VK keepV) x (ω, g)
            r).out = .acc ∧
          ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
            Claims.Flock.lookCol (cols false (Claims.run sha512
              (Claims.Recursion.refuser P x ((VK keepV).k x) decCoin keep m₀) (VK keepV) x (ω, g) r).seen 0) p.val =
                some u ∧ u ≠ C p := by
    intro g
    subst hE
    refine hall (Claims.Recursion.refuser P x ((VK keepV).k x) decCoin keep m₀) (ω, g) C hC hg
      (fun r ℓ c q => rdAtK keepV P x keep r.val ℓ.val q.val (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes ω
        (Claims.Recursion.refDraw g c))
      (fun r ℓ c q => Claims.Recursion.capPlurRd (VK keepV) x (forkRound spec st k0 r.val ℓ.val)
        (rdAtK keepV P x keep r.val ℓ.val q.val (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes ω)
        (Claims.Recursion.refDraw g c)) ?_ ?_
    · intro r ℓ c c' hcc
      funext q
      refine Claims.Recursion.capPlurRd_prefix _ _ _ _ fun i hi => ?_
      have hf : forkRound spec st k0 r.val ℓ.val < (VK keepV).k x :=
        forkRound_lt (tables_head_one ht) hsch r.isLt ℓ.isLt
      have hi' : i.val + 1 < (VK keepV).k x := by omega
      simp only [Claims.Recursion.refDraw, Claims.Recursion.drawOf, Claims.Recursion.setLast, hi', ↓reduceIte]
      exact hcc i hi
    · intro r ℓ c hacc q a ha
      have hab := ((hrun ω g c).1).1 hacc
      rw [(hrun ω g c).2 hacc] at ha
      obtain ⟨hk, hr⟩ := Claims.Recursion.abortRun_acc_iff.1 hab
      rw [Claims.Recursion.abortRun_of_keep hk] at ha
      exact (mem_rdAtK keepV (x := x) (keep := keep)).2 ⟨hk, hr, a, ha, rfl⟩
  have hL : (Claims.Ex fun g : (Fin ((VK keepV).k x) → Coin) → Coin × A =>
      Claims.accPr sha512 (VK keepV) (Claims.Recursion.refuser P x ((VK keepV).k x) decCoin keep m₀) x (ω, g)) =
        Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω) := by
    unfold Claims.accPr Claims.Recursion.accPrD
    rw [← Claims.Recursion.Pr_refDraw hVk]
    exact congrArg Claims.Ex (funext fun g => Claims.Pr_congr fun r => (hrun ω g r).1)
  have hE1 : (Claims.Ex fun g : (Fin ((VK keepV).k x) → Coin) → Coin × A =>
      Claims.Pr fun s : Fin ((VK keepV).k x) → Coin =>
        plainEv₁K keepV (Claims.Recursion.refuser P x ((VK keepV).k x) decCoin keep m₀) x (ω, g) s) ≤
      Claims.Pr fun d : (Fin ((VK keepV).k x) → Coin) × A =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).vLog := by
    refine (Claims.Ex_mono fun g => Claims.Pr_mono fun s hs => ?_).trans
      (le_of_eq (Claims.Recursion.Pr_refDraw hVk fun d =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).vLog))
    have h2 := hs.2
    rw [(hrun ω g s).2 hs.1] at h2
    exact h2
  have hS : (Claims.Ex fun g : (Fin ((VK keepV).k x) → Coin) → Coin × A =>
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
        ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen),
          Claims.Pr (fun c : Fin ((VK keepV).k x) → Coin => E ω r.val ℓ.val q.val (Claims.Recursion.refDraw g c))) =
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
        ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen), Claims.Pr (E ω r.val ℓ.val q.val) := by
    rw [Claims.Bridge.Ex_sum_univ]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [Claims.Bridge.Ex_sum_univ]
    refine Finset.sum_congr rfl fun ℓ _ => ?_
    rw [Claims.Bridge.Ex_sum_univ]
    exact Finset.sum_congr rfl fun q _ => Claims.Recursion.Pr_refDraw hVk (E ω r.val ℓ.val q.val)
  have hO : (Claims.Ex fun g : (Fin ((VK keepV).k x) → Coin) → Coin × A =>
      Claims.Pr fun r : Fin ((VK keepV).k x) → Coin =>
        (Claims.run sha512 (Claims.Recursion.refuser P x ((VK keepV).k x) decCoin keep m₀) (VK keepV) x (ω, g)
          r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512
            (Claims.Recursion.refuser P x ((VK keepV).k x) decCoin keep m₀) (VK keepV) x (ω, g) r).seen 0) p.val =
              some u ∧ u ≠ C p) =
      Claims.Pr fun d : (Fin ((VK keepV).k x) → Coin) × A =>
        (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).seen 0) p.val =
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
  have key : Claims.Recursion.accPrD (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω) ≤ εtable false x +
      ((Claims.Pr fun d : (Fin ((VK keepV).k x) → Coin) × A =>
        Claims.TiedColl sha512 (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).vLog) +
        ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length,
          ∑ q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen),
            Claims.Pr (E ω r.val ℓ.val q.val)) +
      Claims.Pr fun d : (Fin ((VK keepV).k x) → Coin) × A =>
        (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).seen 0) p.val =
            some u ∧ u ≠ C p := by
    linarith
  exact key

end FlockDeploy.KS
