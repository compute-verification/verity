module

public import Proofs.Flock.Claims.KS.JointR
public import Definitions.Claims.SamplingTable
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Star.Real
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.Bridge.Coins
import Proofs.Flock.Claims.Bridge.Transport
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.KS.Deploy
import Proofs.Flock.Claims.KS.FitsR
import Proofs.Flock.Claims.KS.Flock
import Proofs.Flock.Claims.KS.Fork
import Proofs.Flock.Claims.KS.Joint
import Proofs.Flock.Claims.KS.JointJ
import Proofs.Flock.Claims.KS.Range
import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.Prob
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# Flock's KS at one table of a session, from every table's bound

`Flock.KSTable`: the extractor at one table of a session (`Flock.ETable`) is knowledge sound at that table's own
relation
(`Flock.RTable`), with each wait stopping at `k_j = ⌈4·N_j/e⌉` accepted reruns, `N_j` the table's word's length alone.
Every table's bound counted per round (`TableBoundJR`) is already a bound at each table: at a table `j` and a table at
its word's positions that does not extract, the session accepts with probability at most `ε_table + escapes(ω)` beyond
that of accepting and opening a position of table `j`'s word off the table. So the product form (`joint_acc_le`) runs
over table `j`'s positions alone:

* **One wait at any count and event** (`oneWait_le_of_jointAt`): `oneWait_le_of_joint` with its count, its positions'
  bound, its error and its extraction event as parameters; `oneWait_le_of_joint` is its instance at `D.k`, `D.N₀`,
  `D.εtable` and `Extracts`.
* **One wait at table `j`** (`oneWait_le_table`): at table `j`'s positions (`opTable`), where a list of accepted reruns
  that does not extract table `j` fills a table that is not `GoodTableAt` there. The conflict fork at round `0` is at
  table `j`'s positions (`opTable_tied`), at weight `k_j`.
* **The escapes** fit the session's weight: `TableBoundJR`'s fit with one more fork event of weight `D.k`, so with one
  of weight `k_j ≤ D.k` (`Escapes.FitsForkR.mono`), where table `j`'s word is no longer than `N₀`.
* **Extraction** (`assemblesTable_of_extractsAt`): on `ExtractsAt` the decoder lists table `j`'s trace, `assembleTable`
  returns a witness and `checkTable` accepts it. No splitting of the relation by table is read.
* **Assembly** (`ksTable_of_oneWait`, `KSTable_of_tableBoundEachR`, `KSTable_of_tableBoundJR`): `ERep_trunc` at
  `rate = N_j/(e·k_j) ≤ 1/4` (`rateTable_le`) and `Escapes.le`, as `KS_of_oneWait`, at an error for each table
  (`TableBoundEachR`; `TableBoundJR` is its instance at the tables' errors summed). At a table with an empty word
  (`k_j = 0`, every table past the statement's) the error is at least `1`.
-/

@[expose] public section

namespace Claims

section MonoFork

variable {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}

/-- **Escapes that fit with one more fork event of weight `c` fit with a lighter one.** -/
theorem Escapes.FitsForkR.mono {e : Escapes H V P x} {C c c' : ℝ} (h : e.FitsForkR C c) (hc : c' ≤ c) :
    e.FitsForkR C c' := by
  obtain ⟨hmc, hβ⟩ := h
  refine ⟨le_trans (add_le_add (mul_le_mul_of_nonneg_left hc ?_) le_rfl) hmc, hβ⟩
  positivity

end MonoFork

end Claims

namespace Claims.Flock

open FlockSoundness FlockSoundness.Game FlockSoundness.Extract FlockSoundness.Rewinding

/-! ## Extraction at one table -/

theorem checkTable_sound (table : Msg → ℕ → Option Stmt) (j : ℕ) :
    ∀ x w, checkTable table j x w = true → RTable table j x w := by
  intro x w h
  unfold checkTable at h
  split at h
  · rename_i t ht
    exact ⟨t, ht, of_decide_eq_true h⟩
  · simp at h

/-- `E^rep`'s relative rate at table `j`, `N_j/(e·k_j)`, is at most `1/4`, as `k_j = ⌈4·N_j/e⌉`. -/
theorem rateTable_le (N : Msg → ℕ → ℕ) (j : ℕ) (x : Msg) (hk : 0 < kTable N j x) :
    (N x j : ℝ) / (Real.exp 1 * kTable N j x) ≤ 1 / 4 := by
  have hk' : (0 : ℝ) < kTable N j x := by exact_mod_cast hk
  have he := Real.exp_pos 1
  rw [div_le_iff₀ (mul_pos he hk')]
  have hc : 4 * (N x j : ℝ) / Real.exp 1 ≤ kTable N j x := Nat.le_ceil _
  rw [div_le_iff₀ he] at hc
  linarith

theorem κTable_pos {Coin Col T : Type} (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ) (j : ℕ) (x : Msg)
    (hε : 0 ≤ ε x j) : 0 < κTable D N ε j x := by
  unfold κTable
  have h1 : (0 : ℝ) < 2 ^ (-(D.waits x : ℝ)) := by positivity
  have h2 : (0 : ℝ) ≤ 2 * kTable N j x / D.cap x := by positivity
  have h3 : 0 < Real.exp (-(kTable N j x : ℝ) / 8) := Real.exp_pos _
  linarith

/-- At a table with an empty word (`k_j = 0`) the error at the table is at least `1`: its last term is `e^0`. -/
theorem one_le_κTable {Coin Col T : Type} (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ) (j : ℕ) (x : Msg)
    (hε : 0 ≤ ε x j) (hk : kTable N j x = 0) : 1 ≤ κTable D N ε j x := by
  unfold κTable
  have h1 : (0 : ℝ) < 2 ^ (-(D.waits x : ℝ)) := by positivity
  have h2 : (2 : ℝ) * (kTable N j x : ℝ) / D.cap x = 0 := by
    rw [hk]
    simp
  have h3 : Real.exp (-(kTable N j x : ℝ) / 8) = 1 := by
    rw [hk]
    simp
  linarith

section Table

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **Extraction succeeds at table `j`** on its event: the decoder lists the table's trace, `assembleTable` returns a
witness and `checkTable` accepts it. -/
theorem assemblesTable_of_extractsAt (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (hlw : ListWithin D.len D.enc D.θ decode) (x : Msg) (j : ℕ) (vs : List Seen) (h : ExtractsAt D x vs j) :
    Assembles (fun v => [D.cols v]) (assembleTable D.table decode D.read j x) (checkTable D.table j x) vs := by
  obtain ⟨M, t, ht, hR, hag⟩ := h
  have hM : M ∈ decode (vs.flatMap fun v => D.cols v j) := hlw _ M hag
  have hsome : ((decode (vs.flatMap fun v => D.cols v j)).find? fun M => decide (RelS t (D.read [M]))).isSome := by
    rw [List.find?_isSome]
    exact ⟨M, hM, decide_eq_true hR⟩
  obtain ⟨M', hM'⟩ := Option.isSome_iff_exists.1 hsome
  have hq := List.find?_some hM'
  simp only [decide_eq_true_eq] at hq
  refine ⟨D.read [M'], ?_, ?_⟩
  · unfold assembleTable
    rw [ht, Option.bind_some, flatMap_singleton_flatMap, hM', Option.map_some]
  · unfold checkTable
    rw [ht]
    exact decide_eq_true hq

/-! ## One wait, at any count and event -/

/-- **One wait's bound at a tape, from the joint bound at any positions, count and event**: where `op` reads a run's
openings at the positions `Pos`, at most `n` of them, `k ≥ 1`, and at every list of accepted reruns that fails `G` a
run accepts with probability at most `εt + escapes(ω)` beyond that of accepting off the table they fill, the escapes
with the conflict at round `0` added at weight `k` bound one wait of `k` accepted runs at `ω`. -/
theorem oneWait_le_of_jointAt (H : Msg → Msg) (D : Deploy Coin Col T) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) {Pos : Type} [Fintype Pos] (op : Ω → (Fin (D.V.k x) → Coin) → Pos → Option Col)
    (G : List Seen → Prop) (k : ℕ) (n εt : ℝ)
    (hN : (Fintype.card Pos : ℝ) ≤ n) (hεt : 0 ≤ εt) (e₀ : Escapes H D.V P x) (j₀ : Fin (D.V.k x))
    (tied : ∀ ω r r', (∃ p, ConflictAt (op ω) (r, r') p) →
      TiedColl H ((run H P D.V x ω r).vLog ++ (run H P D.V x ω (forkCoins j₀ r r')).vLog))
    (ω : Ω) (hk1 : 1 ≤ k) (hε : 0 < accPr H D.V P x ω)
    (hXb : ∀ bs : List (Fin (D.V.k x) → Coin), (∀ b ∈ bs, Acc H D.V P x ω b) →
      ¬ G (seenOf H D.V P x ω bs) →
      accPr H D.V P x ω ≤ εt + e₀.term ω + Pr fun r : Fin (D.V.k x) → Coin =>
        Acc H D.V P x ω r ∧ ∃ p, Off (op ω) (firstTable (op ω) (fun _ => 0) bs) r p) :
    accPr H D.V P x ω * (accFail H D.V P x ω k G / accPr H D.V P x ω ^ k) ≤
      εt + (e₀.addFork k (Nat.cast_nonneg _) j₀ (fun ω r r' => ∃ p, ConflictAt (op ω) (r, r') p)
        tied).term ω + n / (Real.exp 1 * k) * accPr H D.V P x ω := by
  classical
  set Xt : List (Fin (D.V.k x) → Coin) → Prop := fun bs =>
    (∀ b ∈ bs, Acc H D.V P x ω b) → G (seenOf H D.V P x ω bs) with hXt
  have hXb' : ∀ bs, ¬ Xt bs → expect (ind (Acc H D.V P x ω)) (runGame (D.V.k x)) (runS _) ≤
      (εt + e₀.term ω) + expect (ind fun r => Acc H D.V P x ω r ∧
        ∃ p, Off (op ω) (firstTable (op ω) (fun _ => 0) bs) r p) (runGame (D.V.k x)) (runS _) := by
    intro bs hnX
    simp only [hXt, Classical.not_imp] at hnX
    rw [← accPr_eq, expect_runGame, ← Claims.Bridge.Pr_eq_Ex]
    exact hXb bs hnX.1 hnX.2
  have hεc : 0 ≤ εt + e₀.term ω := add_nonneg hεt (e₀.term_nonneg ω)
  have hj := joint_acc_le (op ω) (fun _ => 0) (runGame (D.V.k x)) (runS _)
    (Acc H D.V P x ω) Xt _ hεc hXb' k hk1
  have hlhs : accPr H D.V P x ω * (accFail H D.V P x ω k G / accPr H D.V P x ω ^ k) =
      expect (ind (Acc H D.V P x ω)) (runGame (D.V.k x)) (runS _) *
        expectAcc (runGame (D.V.k x)) (runS _) (Acc H D.V P x ω) k (ind fun bs => ¬ Xt bs) := by
    rw [oneWait_lhs_eq]
    congr 1
    unfold expectAcc
    congr 1
    refine congrArg (expectN _ _ _) (funext fun bs => ?_)
    rw [prod_map_ind]
    by_cases hall : ∀ b ∈ bs, Acc H D.V P x ω b
    · congr 1
      unfold ind
      have : (¬ Xt bs) ↔ ¬ G (seenOf H D.V P x ω bs) := by
        rw [hXt]
        exact not_congr ⟨fun h => h hall, fun h _ => h⟩
      beta_reduce
      split_ifs with h1 h2 h2 <;> first | rfl | (exfalso; tauto)
    · rw [ite_eq_right hall, zero_mul, zero_mul]
  have hE₂ : expect₂ (fun a a' => ind (fun y : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) =>
        ∃ p, ConflictAt (op ω) y p) (a, a')) (runGame (D.V.k x)) (runS _) =
      Pr fun s : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) => ∃ p, ConflictAt (op ω) (s.1, s.2) p := by
    unfold expect₂
    simp only [expect_runGame]
    rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  rw [hE₂, ← accPr_eq] at hj
  rw [hlhs, ← accPr_eq, Escapes.term_addFork]
  have hrate : (Fintype.card Pos : ℝ) * accPr H D.V P x ω / (Real.exp 1 * k) ≤
      n / (Real.exp 1 * k) * accPr H D.V P x ω := by
    rw [div_mul_eq_mul_div]
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hN hε.le) (by positivity)
  linarith

/-! ## One wait at table `j` -/

open Classical in
/-- **An accepted run's openings at table `j`'s word's positions**, its first column of table `j` at each; none on a
rejected run. -/
noncomputable def opTable (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) {Ω : Type} (P : Prover Ω)
    (x : Msg) (j : ℕ) (ω : Ω) (r : Fin (D.V.k x) → Coin) (p : Fin (N x j)) : Option Col :=
  if (run H P D.V x ω r).out = .acc then lookCol (D.cols (run H P D.V x ω r).seen j) p.val else none

omit [DecidableEq Col] [Fintype Coin] [Nonempty Coin] in
/-- **On accepted reruns the extractor's table at table `j` is its zero-filled word.** -/
theorem firstTable_opTable (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) {Ω : Type} (P : Prover Ω)
    (x : Msg) (j : ℕ) (ω : Ω) (bs : List (Fin (D.V.k x) → Coin)) (hacc : ∀ b ∈ bs, Acc H D.V P x ω b)
    (p : Fin (N x j)) :
    firstTable (opTable H D N P x j ω) (fun _ => 0) bs p =
      (lookCol ((seenOf H D.V P x ω bs).flatMap fun v => D.cols v j) p.val).getD 0 := by
  unfold firstTable
  rw [seenOf, List.flatMap_map,
    findSome?_lookCol _ (fun r => D.cols (run H P D.V x ω r).seen j) p.val bs fun b hb => by
      unfold opTable
      simp only [show (run H P D.V x ω b).out = .acc from hacc b hb, ↓reduceIte]]

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **Two runs at one tape that open a position of table `j`'s word to two columns collide**, with the second run
forked at round `0`: `ConflictTiedJ` at the runs' openings of table `j` (`opTable`). -/
theorem opTable_tied (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (hconf : ConflictTiedJ H D N) {Ω : Type}
    (P : Prover Ω) (x : Msg) (j : ℕ) (hj : j < nTab D x) (j₀ : Fin (D.V.k x)) (hj₀ : j₀.val = 0) (ω : Ω)
    (r r' : Fin (D.V.k x) → Coin) (h : ∃ p, ConflictAt (opTable H D N P x j ω) (r, r') p) :
    TiedColl H ((run H P D.V x ω r).vLog ++ (run H P D.V x ω (forkCoins j₀ r r')).vLog) := by
  classical
  obtain ⟨p, h1, h2, h3⟩ := h
  rw [Claims.Bridge.forkCoins_zero j₀ hj₀]
  simp only [opTable] at h1 h2 h3
  split_ifs at h1 h2 h3 with ha ha'
  · obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 h1
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 h2
    exact hconf x Ω P ω r r' j p.val u v ha ha' hj p.isLt hu hv
      fun huv => h3 (by rw [hu, hv, huv])
  all_goals simp at h1 h2

/-- **One wait's bound at a tape, at table `j`, from the table's bound there**: `oneWait_le_of_jointAt` over table
`j`'s word's positions, at `k_j` accepted reruns and the event that table `j` extracts; the escapes are `e₀` with the
conflict at round `0` at table `j`'s positions (`opTable_tied`), at weight `k_j`. -/
theorem oneWait_le_table (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop)
    (hz : ∀ x j, Ok x j 0) (hok : ColsOkJ H D N Ok) (hconf : ConflictTiedJ H D N) (x : Msg) (j : ℕ)
    (hj : j < nTab D x) (hVk : 0 < D.V.k x) (εt : ℝ) (hεt : 0 ≤ εt) (hk1 : 1 ≤ kTable N j x)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (e₀ : Escapes H D.V P x)
    (hb : ∀ (ω : Ω) (C : Fin (N x j) → Col), (∀ p, Ok x j (C p)) → ¬ GoodTableAt D N x j C →
      accPr H D.V P x ω ≤ εt + e₀.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (N x j)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen j) p.val = some u ∧ u ≠ C p)
    (ω : Ω) (hε : 0 < accPr H D.V P x ω) :
    accPr H D.V P x ω * (accFail H D.V P x ω (kTable N j x) (fun vs => ExtractsAt D x vs j) /
        accPr H D.V P x ω ^ kTable N j x) ≤
      εt + (e₀.addFork (kTable N j x) (Nat.cast_nonneg _) ⟨0, hVk⟩
        (fun ω r r' => ∃ p, ConflictAt (opTable H D N P x j ω) (r, r') p)
        (opTable_tied H D N hconf P x j hj ⟨0, hVk⟩ rfl)).term ω +
        (N x j : ℝ) / (Real.exp 1 * kTable N j x) * accPr H D.V P x ω := by
  classical
  refine oneWait_le_of_jointAt H D P x (opTable H D N P x j) (fun vs => ExtractsAt D x vs j) (kTable N j x) (N x j) εt
    ?_ hεt e₀ ⟨0, hVk⟩ (opTable_tied H D N hconf P x j hj ⟨0, hVk⟩ rfl) ω hk1 hε fun bs hall hnot => ?_
  · exact le_of_eq (by rw [Fintype.card_fin])
  have hCe : ∀ p : Fin (N x j), firstTable (opTable H D N P x j ω) (fun _ => 0) bs p =
      (lookCol ((seenOf H D.V P x ω bs).flatMap fun v => D.cols v j) p.val).getD 0 :=
    fun p => firstTable_opTable H D N P x j ω bs hall p
  have hgood : ¬ GoodTableAt D N x j (firstTable (opTable H D N P x j ω) (fun _ => 0) bs) := fun hg => by
    rw [funext hCe] at hg
    exact hnot (extractsAt_of_goodAt D N x j (seenOf H D.V P x ω bs) hg)
  have hOk : ∀ p, Ok x j (firstTable (opTable H D N P x j ω) (fun _ => 0) bs p) := by
    intro p
    rw [hCe p]
    cases hl : lookCol ((seenOf H D.V P x ω bs).flatMap fun v => D.cols v j) p.val with
    | none => exact hz x j
    | some u =>
      rw [seenOf, List.flatMap_map] at hl
      obtain ⟨b, hb', hu⟩ := lookCol_flatMap_some (fun r => D.cols (run H P D.V x ω r).seen j) p.val u bs hl
      exact hok x Ω P ω b j p.val u (hall b hb') hj p.isLt hu
  refine (hb ω (firstTable (opTable H D N P x j ω) (fun _ => 0) bs) hOk hgood).trans
    (add_le_add le_rfl (Pr_mono fun r ⟨ha, p, u, hu, hne⟩ => ?_))
  have e : opTable H D N P x j ω r p = some u := by
    unfold opTable
    rw [ite_eq_left ha]
    exact hu
  refine ⟨ha, p, ?_⟩
  show (opTable H D N P x j ω r p).isSome ∧ opTable H D N P x j ω r p ≠ some _
  rw [e]
  exact ⟨rfl, fun h => hne (Option.some_injective _ h)⟩

/-! ## KS at table `j` -/

/-- **Flock's KS at table `j`, at one statement and prover, from one wait's bound there**: when `K > 0`, `k_j ≥ 8`
and the error `εt` is nonnegative, and escapes that fit the session's weight bound one wait of `k_j` accepted runs on
the event that table `j` extracts, the bound is `Flock.κTable`'s at `εt`. -/
theorem ksTable_of_oneWait (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (decode : List (ℕ × Col) → List T) (hlw : ListWithin D.len D.enc D.θ decode) (j : ℕ) (x : Msg)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (hK : 0 < D.cap x) (hk8 : 8 ≤ kTable N j x)
    (εt : ℝ) (hεt : 0 ≤ εt) (e : Escapes H D.V P x) (hfit : e.Fits (D.weight x))
    (hw : ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (kTable N j x) (fun vs => ExtractsAt D x vs j) /
          accPr H D.V P x ω ^ kTable N j x) ≤
        εt + e.term ω + (N x j : ℝ) / (Real.exp 1 * kTable N j x) * accPr H D.V P x ω) :
    PrBad H D.V (RTable D.table j) (ETable D N decode j) x P ≤
      4 * εt + 2 ^ (-(D.waits x : ℝ)) + 2 * kTable N j x / D.cap x + Real.exp (-(kTable N j x : ℝ) / 8) +
        Real.sqrt (γ D x * PrB H D.V P x) := by
  have hwait : ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (kTable N j x)
        (Assembles (fun v => [D.cols v]) (assembleTable D.table decode D.read j x) (checkTable D.table j x)) /
          accPr H D.V P x ω ^ kTable N j x) ≤
        (εt + e.term ω) + (N x j : ℝ) / (Real.exp 1 * kTable N j x) * accPr H D.V P x ω := by
    intro ω hε
    refine le_trans ?_ ((hw ω hε).trans (le_of_eq (by ring)))
    refine mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right ?_ (pow_nonneg hε.le _)) hε.le
    unfold accFail
    exact Pr_mono fun σ ⟨hacc, hno⟩ => ⟨hacc, fun hex => hno (assemblesTable_of_extractsAt D decode hlw x j _ hex)⟩
  have hb := ERep_trunc H D.V (RTable D.table j) D.waits (kTable N j) D.cap (fun v => [D.cols v])
    (assembleTable D.table decode D.read j) (checkTable D.table j) (checkTable_sound D.table j) x P hK hk8
    (fun ω => εt + e.term ω) (fun ω => add_nonneg hεt (e.term_nonneg ω)) (rateTable_le N j x (by omega)) hwait
  have hEx : 4 * Ex (fun ω => εt + e.term ω) ≤ 4 * εt + Real.sqrt (γ D x * PrB H D.V P x) := by
    rw [Ex_add, Ex_const]
    have := e.le (D.weight x) hfit
    unfold γ
    linarith
  show PrBad H D.V (RTable D.table j)
      (ERep D.V D.waits (kTable N j) D.cap (fun v => [D.cols v]) (assembleTable D.table decode D.read j)
        (checkTable D.table j)) x P ≤
    4 * εt + 2 ^ (-(D.waits x : ℝ)) + 2 * kTable N j x / D.cap x + Real.exp (-(kTable N j x : ℝ) / 8) +
      Real.sqrt (γ D x * PrB H D.V P x)
  linarith

/-- **Flock: KS at each table of a session, at an error for each table, from each table's bound at its error**
(`TableBoundEachR`): when each of the statement's tables' words is no longer than `N₀`, the errors are nonnegative, the
verifier takes a round, `K > 0`, at each table of the statement `k_j ≥ 8` or the verifier accepts no run, a table past
the statement's has an empty word, and `Ok` admits `0` and every column an accepted run opens. -/
theorem KSTable_of_tableBoundEachR (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (hN : ∀ x j, j < nTab D x → (N x j : ℝ) ≤ D.N₀ x) (hεt : ∀ x j, 0 ≤ ε x j) (hVk : ∀ x, 0 < D.V.k x)
    (hK : ∀ x, 0 < D.cap x)
    (hk : ∀ x j, (j < nTab D x ∧ 8 ≤ kTable N j x) ∨ kTable N j x = 0 ∨ NoAcc H D.V x)
    (Ok : Msg → ℕ → Col → Prop) (hz : ∀ x j, Ok x j 0) (hok : ColsOkJ H D N Ok) (hconf : ConflictTiedJ H D N)
    (htab : TableBoundEachR H D N Ok ε) :
    KSTable H D N ε := by
  intro decode hlw j x Ω _ _ P
  show PrBad H D.V (RTable D.table j) (ETable D N decode j) x P ≤
    κTable D N ε j x + Real.sqrt (γ D x * PrB H D.V P x)
  rcases hk x j with ⟨hj, hk8⟩ | hk0 | hno
  · obtain ⟨e₀, hfit, hb⟩ := htab x Ω P
    have hkle : kTable N j x ≤ D.k x :=
      Nat.ceil_mono (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left (hN x j hj) (by norm_num))
        (Real.exp_pos 1).le)
    have hfit' : e₀.FitsForkR (D.weight x) (kTable N j x) := hfit.mono (Nat.cast_le.2 hkle)
    unfold κTable
    refine ksTable_of_oneWait H D N decode hlw j x P (hK x) hk8 (ε x j) (hεt x j)
      (e₀.addFork (kTable N j x) (Nat.cast_nonneg _) ⟨0, hVk x⟩
        (fun ω r r' => ∃ p, ConflictAt (opTable H D N P x j ω) (r, r') p)
        (opTable_tied H D N hconf P x j hj ⟨0, hVk x⟩ rfl)).mergeR
      (e₀.fitsR_addFork (D.weight x) (kTable N j x) (Nat.cast_nonneg _) ⟨0, hVk x⟩ rfl
        (fun ω r r' => ∃ p, ConflictAt (opTable H D N P x j ω) (r, r') p)
        (opTable_tied H D N hconf P x j hj ⟨0, hVk x⟩ rfl) hfit').fits_mergeR fun ω hε => ?_
    have hm := (e₀.addFork (kTable N j x) (Nat.cast_nonneg _) ⟨0, hVk x⟩
      (fun ω r r' => ∃ p, ConflictAt (opTable H D N P x j ω) (r, r') p)
      (opTable_tied H D N hconf P x j hj ⟨0, hVk x⟩ rfl)).term_le_mergeR ω
    have := oneWait_le_table H D N Ok hz hok hconf x j hj (hVk x) (ε x j) (hεt x j) (by omega) P e₀
      (fun ω C hC hg => hb ω j hj C hC hg) ω hε
    linarith
  · have h1 : PrBad H D.V (RTable D.table j) (ETable D N decode j) x P ≤ 1 := by
      unfold PrBad
      exact Pr_le_one _
    exact h1.trans ((one_le_κTable D N ε j x (hεt x j) hk0).trans (le_add_of_nonneg_right (Real.sqrt_nonneg _)))
  · have h0 : PrBad H D.V (RTable D.table j) (ETable D N decode j) x P = 0 := by
      unfold PrBad
      rw [Pr_congr (q := fun _ => False) fun s => ⟨fun hs => hno Ω P s.1 s.2.1 hs.1, False.elim⟩, Pr_false]
    rw [h0]
    exact add_nonneg (κTable_pos D N ε j x (hεt x j)).le (Real.sqrt_nonneg _)

/-- **Flock: KS at each table of a session, from every table's bound counted per round**, at the tables' errors
summed (`D.εtable`): `KSTable_of_tableBoundEachR` at `TableBoundJR`, which is `TableBoundEachR` at that error. -/
theorem KSTable_of_tableBoundJR (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (hN : ∀ x j, j < nTab D x → (N x j : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x) (hVk : ∀ x, 0 < D.V.k x)
    (hK : ∀ x, 0 < D.cap x)
    (hk : ∀ x j, (j < nTab D x ∧ 8 ≤ kTable N j x) ∨ kTable N j x = 0 ∨ NoAcc H D.V x)
    (Ok : Msg → ℕ → Col → Prop) (hz : ∀ x j, Ok x j 0) (hok : ColsOkJ H D N Ok) (hconf : ConflictTiedJ H D N)
    (htab : TableBoundJR H D N Ok) :
    KSTable H D N (fun x _ => D.εtable x) :=
  KSTable_of_tableBoundEachR H D N (fun x _ => D.εtable x) hN (fun x _ => hεt x) hVk hK hk Ok hz hok hconf htab

end Table

end Claims.Flock
