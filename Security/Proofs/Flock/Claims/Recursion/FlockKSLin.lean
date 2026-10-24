import Proofs.Flock.Claims.Recursion.FlockKSAbort
import Proofs.Flock.Soundness.ExtractAccLin
import Definitions.Claims.RecursionLin

/-!
# Flock meets `KSFlockLin` from the table bound at the abort law, linearly

`KSAbort_of_oneWaitAbort` reads the table bound with escapes under one root: each cap's fork and level 0's conflict
fork, `√(γ·Pr[BAbort])` after `EscapesD.le`. `KSFlockLin` charges each to a finder of its own, so here the table
bound's escapes are any tape term whose average is linear in the finders (`TableBoundAbortLin`), and level 0's
conflict is `F0`'s:

* **The table bound, linearly** (`TableBoundAbortLin`): at every statement, prover, way it may abort and `M`, a term
  `t ≥ 0` on the tapes with `𝔼 t ≤ k(x)·Pr[BAbort] + W(x)·(Pr[FC] + 1/(M+1))`, such that every table of `Ok` columns
  that does not extract leaves a run accepting as it aborts at most `ε_table + t(ω)` more often than accepting so and
  opening a position off the table.
* **One wait over draws, linearly** (`oneWaitD_lin`): `joint_acc_le_lin` at `drawGame`, as `oneWaitD` is
  `joint_acc_le`, and level 0's conflict among the `k + 1` accepted runs as the conflict finder's success
  (`conflict_succ_ge`, `(k+1)/(R+1)` at every `R`).
* **The conflict finder is `F0`** (`expect_conflictG_drawGame`, `Ex_conflictSucc_le_PrF0`): at `drawGame` its runs are
  `F0`'s coins, a wait's first accepted run is `F0`'s pick (`find?_finRange_map`), and two accepted runs that open a
  position to two columns collide in their calls (`ConflictTied`), which `F0`'s `findColl` finds.
* **KS** (`KSFlockLin_of_tableBoundAbortLin`): `PrBadAbort_trunc` with `a = ε_table + t + Pr[conflict finder] +
  (k+1)/(R+1)`, at Flock's own `κ`.
* **The producers' two averages**: one-run events tied to the run's calls average to at most their weight times
  `Pr[BAbort]` (`Ex_oneRun_le`), and the caps' per-position hits of `FC`'s loop sum to `W(x)·Pr[FC]` over the tape
  (`Ex_capHits`).
-/

namespace Claims.Recursion

open Claims Claims.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Extract FlockSoundness.Rewinding

/-! ## Draws as runs of games -/

section DrawRuns

variable {Dr : Type} [Fintype Dr] [Nonempty Dr]

/-- **`k` runs of a game whose runs are uniform draws** are `k` independent draws. -/
theorem expectN_of_expect {α U : Type} [Fintype U] [Nonempty U] (g : Game α) (s : Strategy g) (e : U → α)
    (h : ∀ f : α → ℝ, expect f g s = Ex fun u => f (e u)) :
    ∀ (k : ℕ) (Φ : List α → ℝ),
      expectN g s k Φ = Ex fun σ : Fin k → U => Φ ((List.finRange k).map fun i => e (σ i))
  | 0, Φ => by
    simp only [expectN, List.finRange_zero, List.map_nil]
    exact (Ex_const _).symm
  | k + 1, Φ => by
    rw [expectN, h, Claims.Bridge.Ex_cons]
    refine congrArg Ex (funext fun c => ?_)
    rw [expectN_of_expect g s e h k]
    refine congrArg Ex (funext fun r => ?_)
    simp [List.finRange_succ, List.map_cons, List.map_map, Function.comp_def]

/-- **`R` runs of `drawGame`** are `R` independent draws. -/
theorem expect_runsG_drawGame (R : ℕ) (f : List Dr → ℝ) :
    expect f (runsG (drawGame Dr) R) (runsStrat (drawGame Dr) (drawS Dr) R) =
      Ex fun u : Fin R → Dr => f ((List.finRange R).map u) := by
  rw [expect_runsG, expectN_drawGame]

/-- **The conflict finder at `drawGame`**: a draw, and `k` waits of `R` draws each, all independent. -/
theorem expect_conflictG_drawGame (R k : ℕ) (F : Dr × List (List Dr) → ℝ) :
    expect F (conflictG (drawGame Dr) R k) (conflictStrat (drawGame Dr) (drawS Dr) R k) =
      Ex fun s : Dr × (Fin k → Fin R → Dr) =>
        F (s.1, (List.finRange k).map fun i => (List.finRange R).map (s.2 i)) := by
  rw [expect_conflictG, expect_drawGame, Claims.Bridge.Ex_prod]
  refine congrArg Ex (funext fun a => ?_)
  exact expectN_of_expect _ _ (fun u : Fin R → Dr => (List.finRange R).map u)
    (fun f => expect_runsG_drawGame R f) k _

omit [Fintype Dr] [Nonempty Dr] in
open Classical in
/-- **A wait's first accepted draw is the loop's pick.** -/
theorem find?_finRange_map (p : Dr → Prop) : ∀ (R : ℕ) (us : Fin R → Dr),
    ((List.finRange R).map us).find? (fun x => decide (p x)) = firstPick p R us
  | 0, _ => rfl
  | R + 1, us => by
    rw [List.finRange_succ, List.map_cons, List.find?_cons]
    by_cases h : p (us 0)
    · simp [firstPick, h]
    · simp only [h, decide_false, firstPick, ↓reduceIte]
      rw [← find?_finRange_map p R (Fin.tail us), List.map_map]
      rfl

omit [Fintype Dr] [Nonempty Dr] in
open Classical in
/-- **Each accepted run the waits find** is one wait's first accepted run. -/
theorem mem_of_firsts (acc : Dr → Prop) : ∀ (bl : List (List Dr)) (bs : List Dr), firsts acc bl = some bs →
    ∀ b ∈ bs, ∃ l ∈ bl, l.find? (fun x => decide (acc x)) = some b
  | [], bs, h, b, hb => by
    simp only [firsts, Option.some.injEq] at h
    subst h
    simp at hb
  | l :: bl, bs, h, b, hb => by
    simp only [firsts] at h
    cases hl : l.find? (fun x => decide (acc x)) with
    | none => rw [hl] at h; simp at h
    | some a =>
      rw [hl, Option.bind_some] at h
      cases hr : firsts acc bl with
      | none => rw [hr] at h; simp at h
      | some cs =>
        rw [hr, Option.map_some, Option.some.injEq] at h
        subst h
        rcases List.mem_cons.1 hb with rfl | hb'
        · exact ⟨l, List.mem_cons_self .., hl⟩
        · obtain ⟨l', hl', hf⟩ := mem_of_firsts acc bl cs hr b hb'
          exact ⟨l', List.mem_cons_of_mem _ hl', hf⟩

end DrawRuns

/-! ## One wait over draws, linearly -/

section OneWaitLin

variable {Dr Coin Col T : Type} [Fintype Dr] [Nonempty Dr] [DecidableEq Col] [Zero Col]

/-- **`k + 1` accepted runs conflict at most as often as the conflict finder succeeds, plus `(k+1)/(R+1)`**, at every
`R`: `conflict_succ_ge` where `R ≥ 1`, and at `R = 0` the product is at most `1`. -/
theorem conflict_succ_ge' {β P V : Type} (g : Game β) (s : Strategy g) (acc : β → Prop) (op : β → P → Option V)
    (R k : ℕ) :
    expect (ind acc) g s * expectAcc g s acc (k + 1) (ind (PairConflict op)) ≤
      expect (ind (ConflictSucc acc op)) (conflictG g R k) (conflictStrat g s R k) + (k + 1) / ((R : ℝ) + 1) := by
  have hS0 : 0 ≤ expect (ind (ConflictSucc acc op)) (conflictG g R k) (conflictStrat g s R k) :=
    expect_nonneg (ind_nonneg _) _ _
  rcases Nat.eq_zero_or_pos R with rfl | hR
  · have hε0 : 0 ≤ expect (ind acc) g s := expect_nonneg (ind_nonneg acc) g s
    have hε1 : expect (ind acc) g s ≤ 1 := expect_ind_le_one acc g s
    have hA : expect (ind acc) g s * expectAcc g s acc (k + 1) (ind (PairConflict op)) ≤ 1 := by
      rcases hε0.eq_or_lt with h0 | hpos
      · rw [← h0, zero_mul]; exact zero_le_one
      · have h1 : expectAcc g s acc (k + 1) (ind (PairConflict op)) ≤ 1 := by
          rw [← expectAcc_const g s acc (k + 1) 1 hpos]
          exact expectAcc_mono g s acc (k + 1) (ind_le_one _)
        have h0' : 0 ≤ expectAcc g s acc (k + 1) (ind (PairConflict op)) :=
          expectAcc_nonneg g s acc (k + 1) (ind_nonneg _)
        calc expect (ind acc) g s * expectAcc g s acc (k + 1) (ind (PairConflict op))
            ≤ 1 * expectAcc g s acc (k + 1) (ind (PairConflict op)) := mul_le_mul_of_nonneg_right hε1 h0'
          _ ≤ 1 := by linarith
    have hk : (1 : ℝ) ≤ (k + 1) / (((0 : ℕ) : ℝ) + 1) := by
      rw [Nat.cast_zero, zero_add, div_one]
      linarith [(Nat.cast_nonneg k : (0 : ℝ) ≤ k)]
    linarith
  · have h := conflict_succ_ge g s acc op R k hR
    have hR' : (1 : ℝ) ≤ R := by exact_mod_cast hR
    have he : (R : ℝ) + 1 ≤ Real.exp 1 * R := by
      have := Real.add_one_le_exp (1 : ℝ)
      nlinarith
    have hk0 : (0 : ℝ) ≤ k + 1 := by positivity
    have : (k + 1 : ℝ) / (Real.exp 1 * R) ≤ (k + 1) / ((R : ℝ) + 1) :=
      div_le_div_of_nonneg_left hk0 (by positivity) he
    linarith

/-- **One wait's bound over draws, linearly, from the table's bound**: `oneWaitD` with level 0's conflict among the
`k + 1` accepted runs left linear (`joint_acc_le_lin`) and then charged to the conflict finder at every `R`:
`ε · Q/ε^k ≤ ε_c + N·ε/(e·k) + Pr[conflict finder] + (k+1)/(R+1)`. -/
theorem oneWaitD_lin (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (x : Msg) (tr : Dr → Trace) {s : Flock.Stmt}
    (hs : D.parse x = some s) (h1 : s.tables = 1) (Ok : Msg → Col → Prop) (hz : Ok x 0)
    (hok : ∀ d q u, (tr d).out = .acc → q < N x → lookCol (D.cols (tr d).seen 0) q = some u → Ok x u)
    (εc : ℝ) (hεc : 0 ≤ εc)
    (hb : ∀ C : Fin (N x) → Col, (∀ p, Ok x (C p)) → ¬ GoodTable D N x C →
      accPrD tr ≤ εc + Pr fun d => (tr d).out = .acc ∧
        ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (tr d).seen 0) p.val = some u ∧ u ≠ C p)
    (k : ℕ) (hk : 1 ≤ k) (R : ℕ) :
    accPrD tr * (accFailD tr k (Extracts D x) / accPrD tr ^ k) ≤
      εc + N x * accPrD tr / (Real.exp 1 * k) +
        (expect (ind (ConflictSucc (AccD tr) (opD D N x tr))) (conflictG (drawGame Dr) R k)
          (conflictStrat (drawGame Dr) (drawS Dr) R k) + (k + 1) / ((R : ℝ) + 1)) := by
  classical
  set Xt : List Dr → Prop := fun bs =>
    (∀ b ∈ bs, AccD tr b) → Extracts D x (bs.map fun d => (tr d).seen) with hXt
  have hXb : ∀ bs, ¬ Xt bs → expect (ind (AccD tr)) (drawGame Dr) (drawS Dr) ≤
      εc + expect (ind fun d => AccD tr d ∧
        ∃ p, Off (opD D N x tr) (firstTable (opD D N x tr) (fun _ => 0) bs) d p) (drawGame Dr) (drawS Dr) := by
    intro bs hnX
    simp only [hXt, Classical.not_imp] at hnX
    obtain ⟨hall, hnot⟩ := hnX
    have hft : firstTable (opD D N x tr) (fun _ => 0) bs =
        fun p => (lookCol ((bs.map fun d => (tr d).seen).flatMap fun v => D.cols v 0) p.val).getD 0 :=
      funext (firstTable_opD D N x tr bs hall)
    have hgood : ¬ GoodTable D N x (firstTable (opD D N x tr) (fun _ => 0) bs) := by
      intro hg
      rw [hft] at hg
      exact hnot (extracts_of_good D N x _ hs h1 hg)
    have hOk : ∀ p, Ok x (firstTable (opD D N x tr) (fun _ => 0) bs p) := by
      intro p
      rw [firstTable_opD D N x tr bs hall p]
      cases hl : lookCol ((bs.map fun d => (tr d).seen).flatMap fun v => D.cols v 0) p.val with
      | none => exact hz
      | some u =>
        rw [List.flatMap_map] at hl
        obtain ⟨b, hb', hu⟩ := lookCol_flatMap_some (fun d => D.cols (tr d).seen 0) p.val u bs hl
        exact hok b p.val u (hall b hb') p.isLt hu
    rw [← accPrD_eq, expect_drawGame, ← Claims.Bridge.Pr_eq_Ex]
    refine (hb _ hOk hgood).trans (le_of_eq ?_)
    congr 1
    refine Pr_congr fun d => ?_
    unfold AccD Off opD
    constructor
    · rintro ⟨ha, p, u, hu, hne⟩
      refine ⟨ha, p, ?_⟩
      rw [ite_eq_left ha, hu]
      exact ⟨rfl, fun h => hne (Option.some_injective _ h)⟩
    · rintro ⟨ha, p, h1, h2⟩
      rw [ite_eq_left ha] at h1 h2
      obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 h1
      exact ⟨ha, p, u, hu, fun h => h2 (by rw [hu, h])⟩
  have hj := joint_acc_le_lin (opD D N x tr) (fun _ => 0) (drawGame Dr) (drawS Dr) (AccD tr) Xt εc hεc hXb k hk
  have hc := conflict_succ_ge' (drawGame Dr) (drawS Dr) (AccD tr) (opD D N x tr) R k
  have hlhs : accPrD tr * (accFailD tr k (Extracts D x) / accPrD tr ^ k) =
      expect (ind (AccD tr)) (drawGame Dr) (drawS Dr) *
        expectAcc (drawGame Dr) (drawS Dr) (AccD tr) k (ind fun bs => ¬ Xt bs) := by
    rw [oneWaitD_lhs_eq]
    congr 1
    unfold expectAcc
    congr 1
    refine congrArg (expectN _ _ _) (funext fun bs => ?_)
    rw [prod_map_ind]
    by_cases hall : ∀ b ∈ bs, AccD tr b
    · congr 1
      unfold ind
      have : (¬ Xt bs) ↔ ¬ Extracts D x (bs.map fun d => (tr d).seen) := by
        rw [hXt]
        exact not_congr ⟨fun h => h hall, fun h _ => h⟩
      beta_reduce
      split_ifs with h1 h2 h2 <;> first | rfl | (exfalso; tauto)
    · rw [ite_eq_right hall, zero_mul, zero_mul]
  rw [← accPrD_eq, Fintype.card_fin] at hj
  rw [← accPrD_eq] at hc
  rw [hlhs, ← accPrD_eq]
  linarith

end OneWaitLin

/-! ## Level 0's conflict finder is `F0` -/

section ConflictF0

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- Calls in a sublist collide in the list. -/
theorem tiedColl_of_subset {H : Msg → Msg} {l l' : List (Msg × Msg)} (h : TiedColl H l)
    (hsub : ∀ p ∈ l, p ∈ l') : TiedColl H l' := by
  obtain ⟨u, v, hc, hu, hv⟩ := h
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hu
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hv
  exact ⟨p.1, q.1, hc, List.mem_map.2 ⟨p, hsub p hp, rfl⟩, List.mem_map.2 ⟨q, hsub q hq, rfl⟩⟩

theorem ind_le_of_imp {α β : Type} {p : α → Prop} {q : β → Prop} {a : α} {b : β} (h : p a → q b) :
    ind p a ≤ ind q b := by
  unfold ind
  split_ifs with h1 h2 <;> first | exact absurd (h h1) h2 | norm_num

omit [DecidableEq Col] [Zero Col] in
open Classical in
/-- **The conflict finder over draws succeeds at most as often as `F0` finds a collision**, averaged over the tape:
its `k + 1` accepted runs are `F0`'s run and its waits' picks, two of them that open a position of the word to two
columns collide in their calls (`ConflictTied`), and every call of theirs is in `F0`'s list. -/
theorem Ex_conflictSucc_le_PrF0 (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ)
    (hconf : ConflictTied H D N) (x : Msg) {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A]
    (P : Prover Ω) (keep : Ω → (Fin (D.V.k x) → Coin) → A → Bool) (k R : ℕ) :
    (Ex fun ω => expect (ind (ConflictSucc (AccD (abortRun H P D.V x keep ω))
        (opD D N x (abortRun H P D.V x keep ω))))
      (conflictG (drawGame ((Fin (D.V.k x) → Coin) × A)) R k)
      (conflictStrat (drawGame ((Fin (D.V.k x) → Coin) × A)) (drawS _) R k)) ≤
      PrF0 H D.V P x keep k R := by
  simp only [expect_conflictG_drawGame]
  unfold PrF0 Finder.Pr
  rw [Claims.Bridge.Pr_eq_Ex]
  refine le_of_le_of_eq ?_ (Claims.Bridge.Ex_prod (α := Ω)
    (γ := ((Fin (D.V.k x) → Coin) × A) × (Fin k → Fin R → (Fin (D.V.k x) → Coin) × A))
    (ind fun s => ∃ u v, (F0 H D.V P x keep k R).out s = some (u, v) ∧ Collision H u v)).symm
  refine Ex_mono fun ω => Ex_mono fun s => ind_le_of_imp fun hs => ?_
  obtain ⟨ha, bs, hfs, a', ha', b', hb', p, hp1, hp2, hp3⟩ := hs
  have hacc : accAt H D.V P x keep ω s.1 := ha
  let L : List (Msg × Msg) := (abortRun H P D.V x keep ω s.1).vLog ++ (List.finRange k).flatMap fun i =>
    ((firstPick (accAt H D.V P x keep ω) R (s.2 i)).map fun d => (abortRun H P D.V x keep ω d).vLog).getD []
  have hout : (F0 H D.V P x keep k R).out (ω, s) = findColl L := by
    show (if accAt H D.V P x keep ω s.1 then _ else none) = _
    rw [ite_eq_left hacc]
  have hmem : ∀ b ∈ s.1 :: bs, ∀ q ∈ (abortRun H P D.V x keep ω b).vLog, q ∈ L := by
    intro b hb q hq
    rcases List.mem_cons.1 hb with rfl | hb
    · exact List.mem_append_left _ hq
    · obtain ⟨l, hl, hf⟩ := mem_of_firsts (AccD (abortRun H P D.V x keep ω)) _ bs hfs b hb
      obtain ⟨i, -, rfl⟩ := List.mem_map.1 hl
      rw [find?_finRange_map] at hf
      refine List.mem_append_right _ (List.mem_flatMap.2 ⟨i, List.mem_finRange i, ?_⟩)
      have hf' : firstPick (accAt H D.V P x keep ω) R (s.2 i) = some b := hf
      rw [hf']
      exact hq
  have htied : TiedColl H ((abortRun H P D.V x keep ω a').vLog ++ (abortRun H P D.V x keep ω b').vLog) := by
    simp only [opD] at hp1 hp2 hp3
    split_ifs at hp1 hp2 hp3 with h1 h2
    · obtain ⟨hk1, hr1⟩ := abortRun_acc_iff.1 h1
      obtain ⟨hk2, hr2⟩ := abortRun_acc_iff.1 h2
      obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 hp1
      obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hp2
      have hne : u ≠ v := fun huv => hp3 (by rw [hu, hv, huv])
      rw [abortRun_of_keep hk1] at hu ⊢
      rw [abortRun_of_keep hk2] at hv ⊢
      exact hconf x Ω P ω a'.1 b'.1 p.val u v hr1 hr2 p.isLt hu hv hne
    all_goals simp at hp1 hp2
  have hLt : TiedColl H L := tiedColl_of_subset htied fun q hq => by
    rcases List.mem_append.1 hq with hq | hq
    · exact hmem a' ha' q hq
    · exact hmem b' hb' q hq
  have hlog : ∀ q ∈ L, H q.1 = q.2 := by
    intro q hq
    rcases List.mem_append.1 hq with hq | hq
    · exact abortRun_vLog H P D.V x keep ω _ q hq
    · obtain ⟨i, -, hi⟩ := List.mem_flatMap.1 hq
      cases hpk : firstPick (accAt H D.V P x keep ω) R (s.2 i) with
      | none => rw [hpk] at hi; simp at hi
      | some d =>
        rw [hpk] at hi
        exact abortRun_vLog H P D.V x keep ω d q hi
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (findColl_isSome H hlog hLt)
  exact ⟨u, v, hout.trans huv, findColl_spec H L hlog u v huv⟩

end ConflictF0

/-! ## The table bound at the abort law, linearly, and KS -/

section KSLin

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The fixed-table bound at the abort law, linearly**: at every statement, prover, way it may abort and `M`, a
nonnegative term on the tapes whose average is at most `k(x)·Pr[BAbort] + W(x)·(Pr[FC] + 1/(M+1))`, such that at
every tape and every table at the word's positions whose columns `Ok` admits and which does not extract, a run accepts
as it aborts with probability at most `ε_table` and that term beyond that of accepting so and opening a position of
the word off the table. -/
def TableBoundAbortLin (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop)
    (Cp : Caps Coin) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A]
    (keep : Ω → (Fin (D.V.k x) → Coin) → A → Bool) (M : ℕ),
    ∃ t : Ω → ℝ, (∀ ω, 0 ≤ t ω) ∧
      Ex t ≤ D.V.k x * PrBAbort H D.V P x keep + Cp.width x * (PrFC H D.V Cp P x keep M + 1 / ((M : ℝ) + 1)) ∧
      ∀ (ω : Ω) (C : Fin (N x) → Col), (∀ p, Ok x (C p)) → ¬ GoodTable D N x C →
        accPrD (abortRun H P D.V x keep ω) ≤ D.εtable x + t ω +
          Pr fun d : (Fin (D.V.k x) → Coin) × A => (abortRun H P D.V x keep ω d).out = .acc ∧
            ∃ (p : Fin (N x)) (u : Col),
              lookCol (D.cols (abortRun H P D.V x keep ω d).seen 0) p.val = some u ∧ u ≠ C p

omit [DecidableEq Col] [Zero Col] in
theorem PrFC_nonneg (H : Msg → Msg) (V : Verifier Coin) (Cp : Caps Coin) {Ω A : Type} [Fintype Ω] [Nonempty Ω]
    [Fintype A] [Nonempty A] (P : Prover Ω) (x : Msg) (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (M : ℕ) :
    0 ≤ PrFC H V Cp P x keep M := by
  unfold PrFC
  split_ifs
  · exact Pr_nonneg _
  · exact le_rfl

/-- **Flock as the inner protocol meets `KSFlockLin`, from the table bound at the abort law, linearly**: at every
decoder that meets `ListWithin`, when the word is no longer than `N₀`, `ε_table ≥ 0`, the verifier takes a round,
`K > 0`, `Ok` admits `0` and every column an accepted run opens, with `ConflictTied`, and `k ≥ 8`, the relation splits
by table and the statement has one table wherever the verifier accepts some run. -/
theorem KSFlockLin_of_tableBoundAbortLin (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ)
    (Ok : Msg → Col → Prop) (Cp : Caps Coin) (hN : ∀ x, (N x : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x)
    (hK : ∀ x, 0 < D.cap x) (hk : ∀ x, (8 ≤ D.k x ∧ Splits D x) ∨ NoAcc H D.V x)
    (hone : ∀ x, (∃ s, D.parse x = some s ∧ s.tables = 1) ∨ NoAcc H D.V x) (hz : ∀ x, Ok x 0)
    (hok : ColsOk H D N Ok) (hconf : ConflictTied H D N) (htab : TableBoundAbortLin H D N Ok Cp) :
    KSFlockLin H D Cp := by
  classical
  intro decode hlw x Ω _ _ P A _ _ keep M R
  have hW0 : (0 : ℝ) ≤ Cp.width x := Nat.cast_nonneg _
  have hM0 : (0 : ℝ) ≤ 1 / ((M : ℝ) + 1) := by positivity
  have hR0 : (0 : ℝ) ≤ ((D.k x : ℝ) + 1) / ((R : ℝ) + 1) := by positivity
  have hFC0 := PrFC_nonneg H D.V Cp P x keep M
  have hF00 : 0 ≤ PrF0 H D.V P x keep (D.k x) R := Pr_nonneg _
  have hB0 := PrBAbort_nonneg H D.V P x keep
  have hkx : (0 : ℝ) ≤ D.V.k x := Nat.cast_nonneg _
  rcases hk x with ⟨hk8, hsp⟩ | hno
  swap
  · have h0 : (Inner.ofFlock D decode).PrBadAbort H x P keep = 0 := by
      unfold Inner.PrBadAbort
      rw [Pr_congr (q := fun _ => False) fun s =>
        ⟨fun hs => hno Ω P s.1 s.2.1.1 (abortRun_acc_iff.1 hs.1).2, False.elim⟩, Pr_false]
    rw [h0]
    have := (Flock.κ_pos D x (hεt x)).le
    have : 0 ≤ 4 * (D.V.k x : ℝ) * PrBAbort H D.V P x keep := mul_nonneg (by positivity) hB0
    have : 0 ≤ 4 * (Cp.width x : ℝ) * (PrFC H D.V Cp P x keep M + 1 / ((M : ℝ) + 1)) :=
      mul_nonneg (by positivity) (add_nonneg hFC0 hM0)
    have : 0 ≤ 4 * (PrF0 H D.V P x keep (D.k x) R + ((D.k x : ℝ) + 1) / ((R : ℝ) + 1)) := by linarith
    linarith
  obtain ⟨t, ht0, hEt, hb⟩ := htab x Ω P A keep M
  set cs : Ω → ℝ := fun ω => expect (ind (ConflictSucc (AccD (abortRun H P D.V x keep ω))
      (opD D N x (abortRun H P D.V x keep ω))))
    (conflictG (drawGame ((Fin (D.V.k x) → Coin) × A)) R (D.k x))
    (conflictStrat (drawGame ((Fin (D.V.k x) → Coin) × A)) (drawS _) R (D.k x)) with hcs
  have hcs0 : ∀ ω, 0 ≤ cs ω := fun ω => expect_nonneg (ind_nonneg _) _ _
  set a : Ω → ℝ := fun ω => D.εtable x + t ω + cs ω + ((D.k x : ℝ) + 1) / ((R : ℝ) + 1) with ha
  have ha0 : ∀ ω, 0 ≤ a ω := fun ω => by
    have := hεt x; have := ht0 ω; have := hcs0 ω
    simp only [ha]
    linarith
  have hk1 : 1 ≤ D.k x := by omega
  have hwait : ∀ ω, 0 < accPrD (abortRun H P D.V x keep ω) →
      accPrD (abortRun H P D.V x keep ω) * (accFailD (abortRun H P D.V x keep ω) (D.k x)
        (Assembles (fun v => [D.cols v]) (assemble D.parse D.table decode D.read x) (check D.parse x)) /
          accPrD (abortRun H P D.V x keep ω) ^ D.k x) ≤
        a ω + D.N₀ x / (Real.exp 1 * D.k x) * accPrD (abortRun H P D.V x keep ω) := by
    intro ω hε
    have hacc : ¬ NoAcc H D.V x := fun hno => by
      have : accPrD (abortRun H P D.V x keep ω) = 0 := by
        unfold accPrD
        rw [Pr_congr (q := fun _ => False) fun d =>
          ⟨fun h => hno Ω P ω d.1 (abortRun_acc_iff.1 h).2, False.elim⟩, Pr_false]
      linarith
    obtain ⟨s, hs, h1⟩ := (hone x).resolve_right hacc
    have hokD : ∀ d q u, (abortRun H P D.V x keep ω d).out = .acc → q < N x →
        lookCol (D.cols (abortRun H P D.V x keep ω d).seen 0) q = some u → Ok x u := by
      intro d q u hd hq hu
      obtain ⟨hkeep, hr⟩ := abortRun_acc_iff.1 hd
      rw [abortRun_of_keep hkeep] at hu
      exact hok x Ω P ω d.1 q u hr hq hu
    have hεc : 0 ≤ D.εtable x + t ω := add_nonneg (hεt x) (ht0 ω)
    have hj : accPrD (abortRun H P D.V x keep ω) * (accFailD (abortRun H P D.V x keep ω) (D.k x) (Extracts D x) /
          accPrD (abortRun H P D.V x keep ω) ^ D.k x) ≤
        (D.εtable x + t ω) + N x * accPrD (abortRun H P D.V x keep ω) / (Real.exp 1 * D.k x) +
          (cs ω + ((D.k x : ℝ) + 1) / ((R : ℝ) + 1)) :=
      oneWaitD_lin D N x (abortRun H P D.V x keep ω) hs h1 Ok (hz x) hokD _ hεc (fun C hC hg => hb ω C hC hg)
        (D.k x) hk1 R
    have hrate : (N x : ℝ) * accPrD (abortRun H P D.V x keep ω) / (Real.exp 1 * D.k x) ≤
        D.N₀ x / (Real.exp 1 * D.k x) * accPrD (abortRun H P D.V x keep ω) := by
      rw [div_mul_eq_mul_div]
      exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (hN x) hε.le) (by positivity)
    have hasm : accPrD (abortRun H P D.V x keep ω) * (accFailD (abortRun H P D.V x keep ω) (D.k x)
        (Assembles (fun v => [D.cols v]) (assemble D.parse D.table decode D.read x) (check D.parse x)) /
          accPrD (abortRun H P D.V x keep ω) ^ D.k x) ≤
        accPrD (abortRun H P D.V x keep ω) * (accFailD (abortRun H P D.V x keep ω) (D.k x) (Extracts D x) /
          accPrD (abortRun H P D.V x keep ω) ^ D.k x) := by
      refine mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right ?_ (pow_nonneg hε.le _)) hε.le
      unfold accFailD
      exact Pr_mono fun σ ⟨hacc, hno⟩ => ⟨hacc, fun hex => hno (assembles_of_extracts D decode hlw x _ hsp hex)⟩
    simp only [ha]
    linarith
  have hbad := PrBadAbort_trunc H (Inner.ofFlock D decode) (check_sound D.parse) x P keep (hK x) hk8 a ha0
    (rate_le D x (by omega)) hwait
  have hEx : Ex a = D.εtable x + Ex t + Ex cs + ((D.k x : ℝ) + 1) / ((R : ℝ) + 1) := by
    simp only [ha]
    rw [Ex_add, Ex_add, Ex_add, Ex_const, Ex_const]
  have hcsF : Ex cs ≤ PrF0 H D.V P x keep (D.k x) R := Ex_conflictSucc_le_PrF0 H D N hconf x P keep (D.k x) R
  show (Inner.ofFlock D decode).PrBadAbort H x P keep ≤ _
  have hb' : (Inner.ofFlock D decode).PrBadAbort H x P keep ≤ 4 * Ex a +
      2 ^ (-(D.waits x : ℝ)) + 2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8) := hbad
  rw [hEx] at hb'
  unfold Flock.κ
  linarith

end KSLin

/-! ## The producers' averages -/

section Averages

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **One-run events, linearly**: events on one draw, each tied to a collision among the verifier's calls in the run as
it aborts, weighted by `β ≥ 0`, average over the tape to at most `Σβ · Pr[BAbort]`. -/
theorem Ex_oneRun_le (H : Msg → Msg) (V : Verifier Coin) {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A]
    [Nonempty A] (P : Prover Ω) (x : Msg) (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) {L : Type} [Fintype L]
    (β : L → ℝ) (hβ : ∀ l, 0 ≤ β l) (ev₁ : L → Ω → (Fin (V.k x) → Coin) × A → Prop)
    (tied₁ : ∀ l ω d, ev₁ l ω d → TiedColl H (abortRun H P V x keep ω d).vLog) :
    (Ex fun ω => ∑ l, β l * Pr fun d : (Fin (V.k x) → Coin) × A => ev₁ l ω d) ≤
      (∑ l, β l) * PrBAbort H V P x keep := by
  rw [Ex_sum, Finset.sum_mul]
  refine Finset.sum_le_sum fun l _ => ?_
  rw [Ex_mul_left, Ex_Pr]
  exact mul_le_mul_of_nonneg_left (Pr_one_le_PrBAbort H V P x keep (ev₁ l) (tied₁ l)) (hβ l)

/-- **`FC`'s hits per cap position, summed**: over the tape, the sum over the caps' positions of the probability
that `FC`'s run and loop at that position find a collision is `W(x)·Pr[FC]`, `FC`'s position being uniform. -/
theorem Ex_capHits (H : Msg → Msg) (V : Verifier Coin) (Cp : Caps Coin) {Ω A : Type} [Fintype Ω] [Nonempty Ω]
    [Fintype A] [Nonempty A] (P : Prover Ω) (x : Msg) (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (M : ℕ) :
    (Ex fun ω => ∑ q : Fin (Cp.width x), Pr fun s : ((Fin (V.k x) → Coin) × A) × (Fin M → (Fin (V.k x) → Coin) × A) =>
      ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, s.1, s.2) = some (u, v) ∧ Collision H u v) =
      Cp.width x * PrFC H V Cp P x keep M := by
  unfold PrFC
  split_ifs with hW
  · have hWr : (Cp.width x : ℝ) ≠ 0 := by exact_mod_cast hW.ne'
    have hPr : (FC H V Cp P x keep M hW).Pr H = Ex fun q : Fin (Cp.width x) =>
        Pr fun r : Ω × ((Fin (V.k x) → Coin) × A) × (Fin M → (Fin (V.k x) → Coin) × A) =>
          ∃ u v, (FC H V Cp P x keep M hW).out (q, r.1, r.2.1, r.2.2) = some (u, v) ∧ Collision H u v :=
      (Ex_Pr fun (q : Fin (Cp.width x)) (r : Ω × ((Fin (V.k x) → Coin) × A) × (Fin M → (Fin (V.k x) → Coin) × A)) =>
        ∃ u v, (FC H V Cp P x keep M hW).out (q, r.1, r.2.1, r.2.2) = some (u, v) ∧ Collision H u v).symm
    have hfin : ∀ f : Fin (Cp.width x) → ℝ, (Cp.width x : ℝ) * Ex f = ∑ q, f q := fun f => by
      unfold Ex
      rw [Fintype.card_fin]
      field_simp
    rw [hPr, hfin, Ex_sum]
    refine Finset.sum_congr rfl fun q _ => ?_
    exact Ex_Pr fun (ω : Ω) (s : ((Fin (V.k x) → Coin) × A) × (Fin M → (Fin (V.k x) → Coin) × A)) =>
      ∃ u v, (FC H V Cp P x keep M hW).out (q, ω, s.1, s.2) = some (u, v) ∧ Collision H u v
  · have : IsEmpty (Fin (Cp.width x)) := ⟨fun q => hW q.pos⟩
    simp only [Finset.univ_eq_empty, Finset.sum_empty, mul_zero]
    exact Ex_const 0

end Averages

end Claims.Recursion
