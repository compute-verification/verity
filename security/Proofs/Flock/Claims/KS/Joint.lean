module

public import Proofs.Flock.Claims.DeployCode
public import Proofs.Flock.Claims.KS.Flock
public import Proofs.Flock.Claims.KS.Range
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
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
import Proofs.Flock.Claims.Bridge.Transport
import Proofs.Flock.Claims.KS.Deploy
import Std.Tactic.BVDecide.Normalize.Prop

/-! One wait's bound from a fixed table's bound (`TableBoundθ`, `ColsOk`, `ConflictTied`), at the verifier's runs at one tape read as a game. -/

@[expose] public section

namespace Claims.Flock

open FlockSoundness FlockSoundness.Game FlockSoundness.Extract FlockSoundness.Rewinding

section RunGame

variable {Coin : Type} [Fintype Coin] [Nonempty Coin]

def runGame (n : ℕ) : Game (Fin n → Coin) := .coin (Fin n → Coin) .ret

def runS (n : ℕ) : Strategy (runGame (Coin := Coin) n) := fun _ => PUnit.unit

theorem expect_runGame (n : ℕ) (f : (Fin n → Coin) → ℝ) : expect f (runGame n) (runS n) = Ex f := rfl

theorem expectN_runGame (n : ℕ) : ∀ (k : ℕ) (F : List (Fin n → Coin) → ℝ),
    expectN (runGame n) (runS n) k F = Ex fun σ : Fin k → Fin n → Coin => F ((List.finRange k).map σ)
  | 0, F => by
    simp only [expectN, List.finRange_zero, List.map_nil]
    exact (Ex_const _).symm
  | k + 1, F => by
    rw [expectN, expect_runGame, Claims.Bridge.Ex_cons]
    refine congrArg Ex (funext fun c => ?_)
    rw [expectN_runGame n k]
    refine congrArg Ex (funext fun r => ?_)
    simp [List.finRange_succ, List.map_cons, List.map_map, Function.comp_def]

open Classical in
theorem prod_map_ind {α : Type} (p : α → Prop) :
    ∀ l : List α, (l.map (ind p)).prod = if ∀ a ∈ l, p a then 1 else 0
  | [] => by simp
  | a :: l => by
    rw [List.map_cons, List.prod_cons, prod_map_ind p l]
    unfold ind
    by_cases ha : p a <;> by_cases hl : ∀ b ∈ l, p b <;> simp [ha, hl]

end RunGame

section SamplingDesigns

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

def Acc (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (V.k x) → Coin) : Prop :=
  (run H P V x ω r).out = .acc

def seenOf (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (bs : List (Fin (V.k x) → Coin)) : List Seen :=
  bs.map fun r => (run H P V x ω r).seen

theorem accPr_eq (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    accPr H V P x ω = expect (ind (Acc H V P x ω)) (runGame (V.k x)) (runS _) := by
  rw [expect_runGame]
  exact Claims.Bridge.Pr_eq_Ex _

theorem accFail_eq (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (k : ℕ)
    (G : List Seen → Prop) :
    accFail H V P x ω k G = expectN (runGame (V.k x)) (runS _) k fun bs =>
      (bs.map (ind (Acc H V P x ω))).prod * ind (fun bs => ¬ G (seenOf H V P x ω bs)) bs := by
  classical
  rw [expectN_runGame]
  unfold accFail
  rw [Claims.Bridge.Pr_eq_Ex]
  refine congrArg Ex (funext fun σ => ?_)
  rw [prod_map_ind]
  have hall : (∀ a ∈ (List.finRange k).map σ, Acc H V P x ω a) ↔ ∀ i, (run H P V x ω (σ i)).out = .acc := by
    simp [Acc]
  have hseen : seenOf H V P x ω ((List.finRange k).map σ) =
      (List.finRange k).map fun i => (run H P V x ω (σ i)).seen := by
    simp [seenOf, List.map_map, Function.comp_def]
  unfold ind
  beta_reduce
  rw [hseen]
  by_cases h1 : ∀ i, (run H P V x ω (σ i)).out = .acc <;>
    by_cases h2 : G ((List.finRange k).map fun i => (run H P V x ω (σ i)).seen) <;> simp [Acc, h1, h2]

theorem oneWait_lhs_eq (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (k : ℕ)
    (G : List Seen → Prop) :
    accPr H V P x ω * (accFail H V P x ω k G / accPr H V P x ω ^ k) =
      expect (ind (Acc H V P x ω)) (runGame (V.k x)) (runS _) *
        expectAcc (runGame (V.k x)) (runS _) (Acc H V P x ω) k
          (ind fun bs => ¬ G (seenOf H V P x ω bs)) := by
  rw [accFail_eq, accPr_eq]
  rfl

end SamplingDesigns

section Table

variable {Col T : Type} [DecidableEq Col]

def lookCol (os : List (ℕ × Col)) (q : ℕ) : Option Col := (os.find? fun p => decide (p.1 = q)).map Prod.snd

omit [DecidableEq Col] in
theorem fill_eq_lookCol [Zero Col] (n : ℕ) (os : List (ℕ × Col)) (i : Fin n) :
    fill n os i = (lookCol os i.val).getD 0 := by
  unfold fill lookCol
  congr 3

def extTab [Zero Col] {n : ℕ} (C : Fin n → Col) (q : ℕ) : Col := if h : q < n then C ⟨q, h⟩ else 0

omit [DecidableEq Col] in
theorem lookCol_append (os os' : List (ℕ × Col)) (q : ℕ) :
    lookCol (os ++ os') q = (lookCol os q).or (lookCol os' q) := by
  unfold lookCol
  rw [List.find?_append]
  cases os.find? fun p => decide (p.1 = q) <;> rfl

omit [DecidableEq Col] in
theorem lookCol_flatMap_some {β : Type} (cs : β → List (ℕ × Col)) (q : ℕ) (u : Col) :
    ∀ bs : List β, lookCol (bs.flatMap cs) q = some u → ∃ b ∈ bs, lookCol (cs b) q = some u
  | [], h => by simp [lookCol] at h
  | b :: bs, h => by
    rw [List.flatMap_cons, lookCol_append] at h
    cases hb : lookCol (cs b) q with
    | some v => rw [hb] at h; exact ⟨b, List.mem_cons_self, hb.trans h⟩
    | none =>
      rw [hb] at h
      obtain ⟨b', hb', h'⟩ := lookCol_flatMap_some cs q u bs h
      exact ⟨b', List.mem_cons_of_mem _ hb', h'⟩

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Zero Col]

open Classical in
noncomputable def opθ (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (r : Fin (D.V.k x) → Coin) (p : Fin (N x)) : Option Col :=
  if (run H P D.V x ω r).out = .acc then lookCol (D.cols (run H P D.V x ω r).seen 0) p.val else none

omit [DecidableEq Col] [Fintype Coin] [Nonempty Coin] [Zero Col] in
theorem findSome?_lookCol {β : Type} (f : β → Option Col) (cs : β → List (ℕ × Col)) (q : ℕ) :
    ∀ bs : List β, (∀ b ∈ bs, f b = lookCol (cs b) q) → bs.findSome? f = lookCol (bs.flatMap cs) q
  | [], _ => rfl
  | b :: bs, h => by
    rw [List.flatMap_cons, lookCol_append, List.findSome?_cons, h b List.mem_cons_self,
      findSome?_lookCol f cs q bs fun b' hb' => h b' (List.mem_cons_of_mem _ hb')]
    cases lookCol (cs b) q <;> rfl

omit [DecidableEq Col] [Fintype Coin] [Nonempty Coin] in
theorem firstTable_opθ (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (bs : List (Fin (D.V.k x) → Coin)) (hacc : ∀ b ∈ bs, Acc H D.V P x ω b) (p : Fin (N x)) :
    firstTable (opθ H D N P x ω) (fun _ => 0) bs p =
      (lookCol ((seenOf H D.V P x ω bs).flatMap fun v => D.cols v 0) p.val).getD 0 := by
  unfold firstTable
  rw [seenOf, List.flatMap_map,
    findSome?_lookCol _ (fun r => D.cols (run H P D.V x ω r).seen 0) p.val bs fun b hb => by
      unfold opθ
      simp only [show (run H P D.V x ω b).out = .acc from hacc b hb, ↓reduceIte]]

def GoodTable (D : Deploy Coin Col T) (N : Msg → ℕ) (x : Msg) (C : Fin (N x) → Col) : Prop :=
  ∃ (M : T) (t : Stmt), D.table x 0 = some t ∧ RelS t (D.read [M]) ∧ D.len M ≤ N x ∧
    (1 - D.θ M) * D.len M ≤ ((Finset.univ : Finset (Fin (D.len M))).filter fun i => D.enc M i.val = extTab C i.val).card

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem extractsAt_of_good (D : Deploy Coin Col T) (N : Msg → ℕ) (x : Msg) (vs : List Seen)
    (h : GoodTable D N x fun p => (lookCol (vs.flatMap fun v => D.cols v 0) p.val).getD 0) :
    ExtractsAt D x vs 0 := by
  obtain ⟨M, t, ht, hrel, hle, hag⟩ := h
  have e : ((Finset.univ : Finset (Fin (D.len M))).filter fun i => D.enc M i.val =
        extTab (fun p : Fin (N x) => (lookCol (vs.flatMap fun v => D.cols v 0) p.val).getD 0) i.val) =
      (Finset.univ.filter fun i => D.enc M i.val = fill (D.len M) (vs.flatMap fun v => D.cols v 0) i) :=
    Finset.filter_congr fun i _ => by rw [fill_eq_lookCol, extTab, dite_eq_left (lt_of_lt_of_le i.isLt hle)]
  exact ⟨M, t, ht, hrel, hag.trans_eq (by rw [e])⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem extracts_of_good (D : Deploy Coin Col T) (N : Msg → ℕ) (x : Msg) (vs : List Seen) {s : Stmt}
    (hs : D.parse x = some s) (h1 : s.tables = 1)
    (h : GoodTable D N x fun p => (lookCol (vs.flatMap fun v => D.cols v 0) p.val).getD 0) : Extracts D x vs :=
  ⟨s, hs, fun j hj => by
    obtain rfl : j = 0 := by omega
    exact extractsAt_of_good D N x vs h⟩

end Table

section Fork

variable {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}

def _root_.Claims.Escapes.addFork (e : Escapes H V P x) (c : ℝ) (hc : 0 ≤ c) (j₀ : Fin (V.k x))
    (ev : Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ ω r r', ev ω r r' →
      TiedColl H ((run H P V x ω r).vLog ++ (run H P V x ω (forkCoins j₀ r r')).vLog)) : Escapes H V P x where
  I := Option e.I
  c i := i.elim c e.c
  j i := i.elim j₀ e.j
  ev i := i.elim ev e.ev
  L := e.L
  β := e.β
  ev₁ := e.ev₁
  c_nonneg i := by cases i; exacts [hc, e.c_nonneg _]
  β_nonneg := e.β_nonneg
  tied i := by cases i; exacts [tied, e.tied _]
  tied₁ := e.tied₁

def _root_.Claims.Escapes.FitsFork (e : Escapes H V P x) (C c : ℝ) : Prop :=
  ∃ mult : ℕ, (∀ r : Fin (V.k x), (Finset.univ.filter fun i => e.j i = r).card + (if r.val = 0 then 1 else 0) ≤ mult) ∧
    mult * (c + ∑ i, e.c i) ≤ 2 * C ∧ ∑ l, e.β l ≤ V.k x

omit [Inhabited Coin] in
theorem sum_option_any {α M : Type} [AddCommMonoid M] [Fintype α] (inst : Fintype (Option α))
    (f : Option α → M) : @Finset.sum _ _ _ (@Finset.univ _ inst) f = f none + ∑ a, f (some a) := by
  rw [Subsingleton.elim inst instFintypeOption]
  exact Fintype.sum_option f

omit [Inhabited Coin] in
theorem card_filter_option {α β : Type} [Fintype α] [DecidableEq β] (inst : Fintype (Option α))
    (g : Option α → β) (r : β) :
    ((@Finset.univ _ inst).filter fun i => g i = r).card =
      (if g none = r then 1 else 0) + (Finset.univ.filter fun a => g (some a) = r).card := by
  rw [Subsingleton.elim inst instFintypeOption, Finset.card_filter, Fintype.sum_option, Finset.card_filter]

variable [Fintype Coin]

theorem _root_.Claims.Escapes.term_addFork (e : Escapes H V P x) (c : ℝ) (hc : 0 ≤ c) (j₀ : Fin (V.k x))
    (ev : Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop) (tied) (ω : Ω) :
    (e.addFork c hc j₀ ev tied).term ω =
      Real.sqrt (c * Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev ω s.1 s.2) + e.term ω := by
  have h1 : (∑ i : (e.addFork c hc j₀ ev tied).I, Real.sqrt ((e.addFork c hc j₀ ev tied).c i *
        Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => (e.addFork c hc j₀ ev tied).ev i ω s.1 s.2)) =
      Real.sqrt (c * Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev ω s.1 s.2) +
        ∑ i, Real.sqrt (e.c i * Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => e.ev i ω s.1 s.2) :=
    sum_option_any _ fun i : Option e.I => Real.sqrt (i.elim c e.c *
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => (i.elim ev e.ev) ω s.1 s.2)
  unfold Escapes.term
  rw [h1, add_assoc]
  rfl

omit [Fintype Coin] in
theorem _root_.Claims.Escapes.fits_addFork (e : Escapes H V P x) (C c : ℝ) (hc : 0 ≤ c) (j₀ : Fin (V.k x))
    (hj₀ : j₀.val = 0) (ev : Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop) (tied) (h : e.FitsFork C c) :
    (e.addFork c hc j₀ ev tied).Fits C := by
  classical
  obtain ⟨mult, hm, hmc, hβ⟩ := h
  refine ⟨mult, fun r => ?_, ?_, hβ⟩
  · refine le_trans (le_of_eq ?_) (hm r)
    refine (card_filter_option _ (fun i : Option e.I => i.elim j₀ e.j) r).trans ?_
    by_cases h0 : r.val = 0
    · rw [ite_eq_left (show (none : Option e.I).elim j₀ e.j = r from Fin.ext (hj₀.trans h0.symm)),
        ite_eq_left h0]
      exact Nat.add_comm _ _
    · rw [ite_eq_right (show ¬ (none : Option e.I).elim j₀ e.j = r from fun h => h0 (by
          rw [← (h : j₀ = r)]; exact hj₀)), ite_eq_right h0]
      exact (Nat.zero_add _).trans (Nat.add_zero _).symm
  · show mult * ∑ i : Option e.I, i.elim c e.c ≤ 2 * C
    rw [sum_option_any]
    exact hmc

end Fork

section OneWait

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

def TableBoundθ (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x,
    e.FitsFork (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (C : Fin (N x) → Col), (∀ p, Ok x (C p)) →
      ¬ GoodTable D N x C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen 0) p.val = some u ∧ u ≠ C p

def ColsOk (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) (q : ℕ) (u : Col),
    (run H P D.V x ω r).out = .acc → q < N x → lookCol (D.cols (run H P D.V x ω r).seen 0) q = some u → Ok x u

def ConflictTied (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r r' : Fin (D.V.k x) → Coin) (q : ℕ) (u v : Col),
    (run H P D.V x ω r).out = .acc → (run H P D.V x ω r').out = .acc → q < N x →
    lookCol (D.cols (run H P D.V x ω r).seen 0) q = some u →
    lookCol (D.cols (run H P D.V x ω r').seen 0) q = some v →
    u ≠ v → TiedColl H ((run H P D.V x ω r).vLog ++ (run H P D.V x ω r').vLog)

theorem oneWait_le_of_joint (H : Msg → Msg) (D : Deploy Coin Col T) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) {Pos : Type} [Fintype Pos] (op : Ω → (Fin (D.V.k x) → Coin) → Pos → Option Col)
    (hN : (Fintype.card Pos : ℝ) ≤ D.N₀ x) (hεt : 0 ≤ D.εtable x) (e₀ : Escapes H D.V P x) (j₀ : Fin (D.V.k x))
    (tied : ∀ ω r r', (∃ p, ConflictAt (op ω) (r, r') p) →
      TiedColl H ((run H P D.V x ω r).vLog ++ (run H P D.V x ω (forkCoins j₀ r r')).vLog))
    (ω : Ω) (hk1 : 1 ≤ D.k x) (hε : 0 < accPr H D.V P x ω)
    (hXb : ∀ bs : List (Fin (D.V.k x) → Coin), (∀ b ∈ bs, Acc H D.V P x ω b) →
      ¬ Extracts D x (seenOf H D.V P x ω bs) →
      accPr H D.V P x ω ≤ D.εtable x + e₀.term ω + Pr fun r : Fin (D.V.k x) → Coin =>
        Acc H D.V P x ω r ∧ ∃ p, Off (op ω) (firstTable (op ω) (fun _ => 0) bs) r p) :
    accPr H D.V P x ω * (accFail H D.V P x ω (D.k x) (Extracts D x) / accPr H D.V P x ω ^ D.k x) ≤
      D.εtable x + (e₀.addFork (D.k x) (Nat.cast_nonneg _) j₀ (fun ω r r' => ∃ p, ConflictAt (op ω) (r, r') p)
        tied).term ω + D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω := by
  classical
  set Xt : List (Fin (D.V.k x) → Coin) → Prop := fun bs =>
    (∀ b ∈ bs, Acc H D.V P x ω b) → Extracts D x (seenOf H D.V P x ω bs) with hXt
  have hXb' : ∀ bs, ¬ Xt bs → expect (ind (Acc H D.V P x ω)) (runGame (D.V.k x)) (runS _) ≤
      (D.εtable x + e₀.term ω) + expect (ind fun r => Acc H D.V P x ω r ∧
        ∃ p, Off (op ω) (firstTable (op ω) (fun _ => 0) bs) r p) (runGame (D.V.k x)) (runS _) := by
    intro bs hnX
    simp only [hXt, Classical.not_imp] at hnX
    rw [← accPr_eq, expect_runGame, ← Claims.Bridge.Pr_eq_Ex]
    exact hXb bs hnX.1 hnX.2
  have hεc : 0 ≤ D.εtable x + e₀.term ω := add_nonneg hεt (e₀.term_nonneg ω)
  have hj := joint_acc_le (op ω) (fun _ => 0) (runGame (D.V.k x)) (runS _)
    (Acc H D.V P x ω) Xt _ hεc hXb' (D.k x) hk1
  have hlhs : accPr H D.V P x ω * (accFail H D.V P x ω (D.k x) (Extracts D x) /
        accPr H D.V P x ω ^ D.k x) =
      expect (ind (Acc H D.V P x ω)) (runGame (D.V.k x)) (runS _) *
        expectAcc (runGame (D.V.k x)) (runS _) (Acc H D.V P x ω) (D.k x) (ind fun bs => ¬ Xt bs) := by
    rw [oneWait_lhs_eq]
    congr 1
    unfold expectAcc
    congr 1
    refine congrArg (expectN _ _ _) (funext fun bs => ?_)
    rw [prod_map_ind]
    by_cases hall : ∀ b ∈ bs, Acc H D.V P x ω b
    · congr 1
      unfold ind
      have : (¬ Xt bs) ↔ ¬ Extracts D x (seenOf H D.V P x ω bs) := by
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
  have hrate : (Fintype.card Pos : ℝ) * accPr H D.V P x ω / (Real.exp 1 * D.k x) ≤
      D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω := by
    rw [div_mul_eq_mul_div]
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hN hε.le) (by positivity)
  linarith

theorem oneWait_of_tableBound (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ)
    (hN : ∀ x, (N x : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x) (hVk : ∀ x, 0 < D.V.k x)
    (hk : ∀ x, 1 ≤ D.k x ∨ NoAcc H D.V x)
    (hone : ∀ x, (∃ s, D.parse x = some s ∧ s.tables = 1) ∨ NoAcc H D.V x) (Ok : Msg → Col → Prop) (hz : ∀ x, Ok x 0)
    (hok : ColsOk H D N Ok) (hconf : ConflictTied H D N) (htab : TableBoundθ H D N Ok) :
    OneWait H D := by
  classical
  intro x Ω _ _ P
  obtain ⟨e₀, hfit, hb⟩ := htab x Ω P
  set j₀ : Fin (D.V.k x) := ⟨0, hVk x⟩
  have tied : ∀ ω r r', (∃ p, ConflictAt (opθ H D N P x ω) (r, r') p) →
      TiedColl H ((run H P D.V x ω r).vLog ++ (run H P D.V x ω (forkCoins j₀ r r')).vLog) := by
    intro ω r r' ⟨p, h1, h2, h3⟩
    rw [Claims.Bridge.forkCoins_zero j₀ rfl]
    simp only [opθ] at h1 h2 h3
    split_ifs at h1 h2 h3 with ha ha'
    · obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 h1
      obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 h2
      exact hconf x Ω P ω r r' p.val u v ha ha' p.isLt hu hv fun huv => h3 (by rw [hu, hv, huv])
    all_goals simp at h1 h2
  refine ⟨e₀.addFork (D.k x) (Nat.cast_nonneg _) j₀ _ tied,
    e₀.fits_addFork (D.weight x) (D.k x) (Nat.cast_nonneg _) j₀ rfl _ tied hfit, fun ω hε => ?_⟩
  have hacc : ¬ NoAcc H D.V x := fun hno => by
    have : accPr H D.V P x ω = 0 := by
      unfold accPr
      rw [Pr_congr (q := fun _ => False) fun r => ⟨fun h => hno Ω P ω r h, False.elim⟩, Pr_false]
    linarith
  obtain ⟨s, hs, h1⟩ := (hone x).resolve_right hacc
  refine oneWait_le_of_joint H D P x (opθ H D N P x) (by rw [Fintype.card_fin]; exact hN x) (hεt x) e₀ j₀ tied ω
    ((hk x).resolve_right hacc) hε fun bs hall hnot => ?_
  have hgood : ¬ GoodTable D N x (firstTable (opθ H D N P x ω) (fun _ => 0) bs) := by
    intro hg
    refine hnot (extracts_of_good D N x _ hs h1 ?_)
    have : firstTable (opθ H D N P x ω) (fun _ => 0) bs =
        fun p => (lookCol ((seenOf H D.V P x ω bs).flatMap fun v => D.cols v 0) p.val).getD 0 :=
      funext (firstTable_opθ H D N P x ω bs hall)
    rwa [this] at hg
  have hOk : ∀ p, Ok x (firstTable (opθ H D N P x ω) (fun _ => 0) bs p) := by
    intro p
    rw [firstTable_opθ H D N P x ω bs hall p]
    cases hl : lookCol ((seenOf H D.V P x ω bs).flatMap fun v => D.cols v 0) p.val with
    | none => exact hz x
    | some u =>
      rw [seenOf, List.flatMap_map] at hl
      obtain ⟨b, hb', hu⟩ := lookCol_flatMap_some (fun r => D.cols (run H P D.V x ω r).seen 0) p.val u bs hl
      exact hok x Ω P ω b p.val u (hall b hb') p.isLt hu
  refine (hb ω _ hOk hgood).trans (le_of_eq ?_)
  congr 1
  refine Pr_congr fun r => ?_
  unfold Acc Off opθ
  constructor
  · rintro ⟨ha, p, u, hu, hne⟩
    refine ⟨ha, p, ?_⟩
    rw [ite_eq_left ha, hu]
    exact ⟨rfl, fun h => hne (Option.some_injective _ h)⟩
  · rintro ⟨ha, p, h1, h2⟩
    rw [ite_eq_left ha] at h1 h2
    obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 h1
    exact ⟨ha, p, u, hu, fun h => h2 (by rw [hu, h])⟩

def orZero (Ok : Msg → Col → Prop) : Msg → Col → Prop := fun x u => u = 0 ∨ Ok x u

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem colsOk_orZero (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop)
    (h : ColsOk H D N Ok) : ColsOk H D N (orZero Ok) :=
  fun x Ω P ω r q u ha hq hu => .inr (h x Ω P ω r q u ha hq hu)

end OneWait

end Claims.Flock

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

def wordLen (zk : Bool) (x : Claims.Msg) : ℕ :=
  match (tables zk x).head? with
  | some (_, st) => 2 ^ (if zk then Claims.Flock.zkLevel0 st.wideLeaf st.m else Claims.Flock.plainLevel0 st.m).d
  | none => 0

noncomputable def zeroCol (zk : Bool) (x : Claims.Msg) : List 𝔽 :=
  match (tables zk x).head? with
  | some (_, st) =>
    List.replicate (2 ^ (if zk then Claims.Flock.zkLevel0 st.wideLeaf st.m else Claims.Flock.plainLevel0 st.m).lanes) 0
  | none => []

theorem wordLen_le_N₀ (zk : Bool) (x : Claims.Msg) : (wordLen zk x : ℝ) ≤ N₀ zk x := by
  unfold wordLen N₀
  cases h : tables zk x with
  | nil => simp
  | cons p l =>
    obtain ⟨_, st⟩ := p
    have hl : 0 ≤ (l.map fun p => tableLen zk p.2).sum :=
      List.sum_nonneg fun a ha => by
        obtain ⟨q, -, rfl⟩ := List.mem_map.1 ha
        unfold tableLen
        positivity
    simp only [List.head?_cons, List.map_cons, List.sum_cons, Nat.cast_pow, Nat.cast_ofNat]
    exact le_add_of_nonneg_right hl

namespace Gen

variable [S : ZkSetup]

theorem V_k_pos (zk : Bool) (x : Claims.Msg) : 0 < (V zk).k x := by
  simp only [V]
  split <;> omega

end Gen

theorem V_k_pos (zk : Bool) (x : Claims.Msg) : 0 < (V zk).k x :=
  Gen.V_k_pos (S := S₀) (zk := zk) (x := x)

theorem k_one_plain (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    1 ≤ (deployPlain rel).k x ∨ Claims.Flock.NoAcc sha512 (deployPlain rel).V x :=
  (k_range_plain rel x).imp_left fun h => by omega

namespace Gen

variable [S : ZkSetup]

theorem k_one_zk (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    1 ≤ (deployZk rel).k x ∨ Claims.Flock.NoAcc sha512 (deployZk rel).V x :=
  (k_range_zk rel x).imp_left fun h => by omega

end Gen

theorem k_one_zk (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    1 ≤ (deployZk rel).k x ∨ Claims.Flock.NoAcc sha512 (deployZk rel).V x :=
  Gen.k_one_zk (S := S₀) (rel := rel) (x := x)

def wide (zk : Bool) (x : Claims.Msg) (u : List 𝔽) : Prop := u.length = (zeroCol zk x).length

theorem wide_ne_zero (zk : Bool) (x : Claims.Msg) (u : List 𝔽) (hN : 0 < wordLen zk x) (h : wide zk x u) :
    u ≠ 0 := by
  unfold wordLen at hN
  unfold wide zeroCol at h
  rintro rfl
  cases hh : (tables zk x).head? with
  | none => rw [hh] at hN; exact absurd hN (lt_irrefl 0)
  | some p =>
    obtain ⟨_, _⟩ := p
    rw [hh] at h
    simp only [List.length_replicate] at h
    exact absurd h.symm (pow_ne_zero _ two_ne_zero)

theorem plain_oneWait_of_tableBound (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployPlain rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployPlain rel).V x)
    (hok : Claims.Flock.ColsOk sha512 (deployPlain rel) (wordLen false) (wide false))
    (hconf : Claims.Flock.ConflictTied sha512 (deployPlain rel) (wordLen false))
    (htab : Claims.Flock.TableBoundθ sha512 (deployPlain rel) (wordLen false) (Claims.Flock.orZero (wide false))) :
    Claims.Flock.OneWait sha512 (deployPlain rel) :=
  Claims.Flock.oneWait_of_tableBound sha512 (deployPlain rel) (wordLen false) (wordLen_le_N₀ false)
    (fun x => εtable_nonneg false x) (V_k_pos false) (k_one_plain rel) hone _ (fun _ => .inl rfl)
    (Claims.Flock.colsOk_orZero _ _ _ _ hok) hconf htab

theorem zk_oneWait_of_tableBound (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (hok : Claims.Flock.ColsOk sha512 (deployZk rel) (wordLen true) (wide true))
    (hconf : Claims.Flock.ConflictTied sha512 (deployZk rel) (wordLen true))
    (htab : Claims.Flock.TableBoundθ sha512 (deployZk rel) (wordLen true) (Claims.Flock.orZero (wide true))) :
    Claims.Flock.OneWait sha512 (deployZk rel) :=
  Claims.Flock.oneWait_of_tableBound sha512 (deployZk rel) (wordLen true) (wordLen_le_N₀ true)
    (fun x => εtable_nonneg true x) (V_k_pos true) (k_one_zk rel) hone _ (fun _ => .inl rfl)
    (Claims.Flock.colsOk_orZero _ _ _ _ hok) hconf htab

theorem plain_KS_of_tableBound (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployPlain rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployPlain rel).V x)
    (hok : Claims.Flock.ColsOk sha512 (deployPlain rel) (wordLen false) (wide false))
    (hconf : Claims.Flock.ConflictTied sha512 (deployPlain rel) (wordLen false))
    (htab : Claims.Flock.TableBoundθ sha512 (deployPlain rel) (wordLen false) (Claims.Flock.orZero (wide false))) :
    Claims.Flock.KS sha512 (deployPlain rel) :=
  plain_of_oneWait_sha512 rel
    (fun x => (hone x).imp_left fun ⟨_, hs, h1⟩ => splits_of_one (deployPlain rel) rfl rfl hs h1)
    (plain_oneWait_of_tableBound rel hone hok hconf htab)

theorem zk_KS_of_tableBound (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (hok : Claims.Flock.ColsOk sha512 (deployZk rel) (wordLen true) (wide true))
    (hconf : Claims.Flock.ConflictTied sha512 (deployZk rel) (wordLen true))
    (htab : Claims.Flock.TableBoundθ sha512 (deployZk rel) (wordLen true) (Claims.Flock.orZero (wide true))) :
    Claims.Flock.KS sha512 (deployZk rel) :=
  zk_of_oneWait_sha512 rel
    (fun x => (hone x).imp_left fun ⟨_, hs, h1⟩ => splits_of_one (deployZk rel) rfl rfl hs h1)
    (zk_oneWait_of_tableBound rel hone hok hconf htab)

end FlockDeploy.KS
