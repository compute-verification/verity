module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.Claims.Sampling.KS
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
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
import Mathlib.RingTheory.WittVector.IsPoly
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Extract

/-!
# The sampled compiler: the extraction's failure event and its probability

The extraction's failure event and its bound, for a compiled sampled verifier `L` over any proof system for proof
blocks `A` (`Claims.SamplingGeneric.Compiled`). The extractor's coins at an accepted inner run are one
`Compiled.BlockEC` per slot of the sample (`Compiled.SlotEC`), of which a session reads its first slot's.

* `first_of_slot`, `leadOf`, `card_first_le`, `sum_first_le`: **a sample's sessions**. A slot holding a block below
  `n_PU` is in a session whose first slot (`leadOf`) holds one too; first slots have distinct keys, so a sample has at
  most `n_sess` sessions (`Sessions.keys`), and their slots are disjoint.
* `Found`: a sampled proof block's session's extraction finds a witness in `A`'s relation at its statement, as
  `Compiled.sessWit` runs it. `GoodAt`: the block, below `n_PU`, is at some slot of the sample, and the block at the
  first slot of that slot's session is found at the first slot's coins.
* `BadOuter`, `Fails`: an outer run is bad for a set `B` of proof blocks when its inner wait comes up short or keeps only
  runs at which no block of `B` is good; the extraction fails when the outer wait comes up short or, for some `B` of
  acceptance probability at most `δ` (`S₁`), keeps only runs bad for `B`. Both read the found event as a parameter.
* `SessionKS`: the sessions' knowledge soundness at the found event, the form `A.KS` gives at the slot a selection
  picks: a weight `β` of each tape and first-stage coins, averaging at most `Pr[B]`, bounds the probability that the run
  accepts and the proof block at the picked slot is not found at the slot's coins by `κ_s + √(k(x)·γ_s·β)`.
* `pr_notGood_le`, `inner_bad_le`: at first-stage coins where the sample misses `B` with probability at most `esc`, the
  union bound at the picked slot, the first slot of the session of the first slot holding a block of `B` (`leadSel`;
  the sample's slots at most `n_PU`, `Sessions.slots`), gives
  `Pr[acc]·Pr[bad for B] ≤ 2k_in/K_in + e^{−k_in/8} + c·(esc + κ)` (`rpow_kIn_le_slack`: one set).
* `outer_bad_le`: over the refused sets (at most `2^{n_PU}`, `n_PU ≤ n_out` by `Sessions.unions`, against the outer
  wait's slack, `rpow_kOut_le_slack`), with the sampling design's weights averaging at most `P_a(B) ≤ δ`.
* `pr_fails_le_of`, `pr_fails_le`: **the composition**, `Pr[acc ∧ fails] ≤ κ_δ + √(γ·Pr[B])` with
  `κ_δ = c²·(δ + κ_s) + (2k_out/K_out + e^{−k_out/8}) + c·(2k_in/K_in + e^{−k_in/8})`: independent of the judged coins,
  the outer bound at each tape, and
  the average of `√(k(x)·γ_s·β)` at most `√(k(x)·γ_s·Pr[B])` (Jensen).
* `KS_of_split`, `KS_of_sessionKS`: **`Compiled.KS`** from the split (an accepted run whose extraction fails is in
  `Fails` or its extraction's own calls collide) and the sessions' bound, which at `Found` is `SessionKS`, so `KSOf`
  (`SamplingGeneric/Split.lean`) needs only the split.
* `κ_le_κs`, `γ_le_γs`: each proof block's error bounds are at most the sessions' (`le_maxBelow`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScope forkOn forkOn_forkOn' accList AllBad wait_bad_le S₁ card_S₁_le
  rpow_le_slack Pr_or_le slack slack_pos Pr_acc_forkOn)
open Claims.Recursion (serve viewing)

theorem foldr_max_nonneg (l : List ℝ) : 0 ≤ l.foldr max 0 := by
  induction l with
  | nil => exact le_rfl
  | cons a l ih => exact ih.trans (le_max_right _ _)

theorem maxBelow_nonneg (n : ℕ) (f : ℕ → ℝ) : 0 ≤ maxBelow n f := foldr_max_nonneg _

theorem le_foldr_max {a : ℝ} : ∀ {l : List ℝ}, a ∈ l → a ≤ l.foldr max 0
  | [], h => by simp at h
  | b :: l, h => by
    rw [List.foldr_cons]
    rcases List.mem_cons.1 h with rfl | h
    · exact le_max_left _ _
    · exact le_max_of_le_right (le_foldr_max h)

theorem le_maxBelow {n : ℕ} (f : ℕ → ℝ) {u : ℕ} (hu : u < n) : f u ≤ maxBelow n f :=
  le_foldr_max (List.mem_map.2 ⟨u, List.mem_range.2 hu, rfl⟩)

theorem getElem?_of_mem_zip_finRange {β : Type} {k : ℕ} {l : List β} {j : Fin k} {a : β}
    (h : (j, a) ∈ (List.finRange k).zip l) : l[j.val]? = some a := by
  obtain ⟨i, hi⟩ := List.mem_iff_getElem?.1 h
  obtain ⟨h1, h2⟩ := List.getElem?_zip_eq_some.1 hi
  obtain ⟨hlt, hj⟩ := List.getElem?_eq_some_iff.1 h1
  have hj' : (List.finRange k)[i] = j := hj
  have hji : j.val = i := by rw [← hj']; simp
  rw [hji]
  exact h2

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The extractor's coins, per outer run -/

/-- The extractor's coins for one outer run: its inner wait's, and at each of its accepted runs every slot's. -/
abbrev E1 (L : Compiled Coin A) (x : Msg) := (Fin (L.capIn x) → Fin (L.V.k x) → Coin) × (Fin (L.kIn x) → L.SlotEC x)

def toE1 (L : Compiled Coin A) (x : Msg) (c : L.ECoins x) (j : Fin (L.kOut x)) : L.E1 x :=
  (c.2.1 j, fun l u => c.2.2 (j, l, u))

/-- The extractor's coins as the outer wait's and one `E1` per outer run. -/
def ecoinsEquiv (L : Compiled Coin A) (x : Msg) :
    L.ECoins x ≃ (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → L.E1 x) where
  toFun c := (c.1, L.toE1 x c)
  invFun p := (p.1, fun j => (p.2 j).1, fun t => (p.2 t.1).2 t.2.1 t.2.2)
  left_inv _ := rfl
  right_inv _ := rfl

/-! ## A sample's sessions -/

section Sessions

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

theorem mem_sessSlots {L : Compiled Coin A} {x : Msg} {l : List ℕ} {i j : ℕ} :
    j ∈ L.sessSlots x l i ↔ ∃ u, l[j]? = some u ∧ u < L.nPU x ∧ L.skey x l j = L.skey x l i := by
  unfold sessSlots
  rw [List.mem_filter, List.mem_range, Bool.and_eq_true, beq_iff_eq]
  constructor
  · rintro ⟨-, h1, h2⟩
    cases hu : l[j]? with
    | none => rw [hu] at h1; exact absurd h1 Bool.false_ne_true
    | some u =>
      rw [hu, Option.any_some] at h1
      exact ⟨u, rfl, of_decide_eq_true h1, h2⟩
  · rintro ⟨u, hu, hlt, hk⟩
    refine ⟨(List.getElem?_eq_some_iff.1 hu).1, ?_, hk⟩
    rw [hu, Option.any_some]
    exact decide_eq_true hlt

theorem sessSlots_congr {L : Compiled Coin A} {x : Msg} {l : List ℕ} {i i' : ℕ} (h : L.skey x l i = L.skey x l i') :
    L.sessSlots x l i = L.sessSlots x l i' := by
  unfold sessSlots
  rw [h]

theorem self_mem_sessSlots {L : Compiled Coin A} {x : Msg} {l : List ℕ} {i u : ℕ} (hi : l[i]? = some u)
    (hu : u < L.nPU x) : i ∈ L.sessSlots x l i :=
  mem_sessSlots.2 ⟨u, hi, hu, rfl⟩

theorem sessSlots_pairwise (L : Compiled Coin A) (x : Msg) (l : List ℕ) (i : ℕ) :
    (L.sessSlots x l i).Pairwise (· < ·) :=
  List.pairwise_lt_range.filter _

/-- **The first slot of a sampled block's session**: slot `i`, its block below `n_PU`, is in a session whose first slot
`i₀ ≤ i` holds a block below `n_PU`, has `i`'s key, and is its own session's first. -/
theorem first_of_slot {L : Compiled Coin A} {x : Msg} {l : List ℕ} {i u : ℕ} (hi : l[i]? = some u)
    (hu : u < L.nPU x) :
    ∃ i₀ u₀, (L.sessSlots x l i).head? = some i₀ ∧ i₀ ≤ i ∧ l[i₀]? = some u₀ ∧ u₀ < L.nPU x ∧
      L.skey x l i₀ = L.skey x l i ∧ (L.sessSlots x l i₀).head? = some i₀ := by
  have hmem := self_mem_sessSlots hi hu
  have hp := L.sessSlots_pairwise x l i
  cases h : L.sessSlots x l i with
  | nil => rw [h] at hmem; exact absurd hmem List.not_mem_nil
  | cons i₀ t =>
    have hi₀ : i₀ ∈ L.sessSlots x l i := by rw [h]; exact List.mem_cons_self
    obtain ⟨u₀, hu₀, hlt, hk⟩ := mem_sessSlots.1 hi₀
    refine ⟨i₀, u₀, rfl, ?_, hu₀, hlt, hk, ?_⟩
    · rw [h] at hmem hp
      rcases List.mem_cons.1 hmem with h' | h'
      · exact h'.ge
      · exact (List.rel_of_pairwise_cons hp h').le
    · rw [sessSlots_congr hk, h, List.head?_cons]

/-- Two sessions' first slots with one key are one slot. -/
theorem first_eq_of_key {L : Compiled Coin A} {x : Msg} {l : List ℕ} {i j : ℕ}
    (hi : (L.sessSlots x l i).head? = some i) (hj : (L.sessSlots x l j).head? = some j)
    (hk : L.skey x l i = L.skey x l j) : i = j := by
  rw [sessSlots_congr hk, hj] at hi
  exact (Option.some.inj hi).symm

/-- A session's first slot holds a block below `n_PU`. -/
theorem of_first {L : Compiled Coin A} {x : Msg} {l : List ℕ} {i : ℕ} (h : (L.sessSlots x l i).head? = some i) :
    ∃ u, l[i]? = some u ∧ u < L.nPU x := by
  obtain ⟨u, hu, hlt, -⟩ := mem_sessSlots.1 (List.mem_of_mem_head? h)
  exact ⟨u, hu, hlt⟩

/-- The first slot of slot `i`'s session in the sample `l`; none when no slot of it holds a block below `n_PU`. -/
def leadOf (L : Compiled Coin A) (x : Msg) (l : List ℕ) (i : Fin (L.nPU x)) : Option (Fin (L.nPU x)) :=
  ((L.sessSlots x l i).head?).bind fun i₀ => if h : i₀ < L.nPU x then some ⟨i₀, h⟩ else none

/-- A selection of the sample's slots, moved to the first slot of the picked slot's session. -/
def leadSel (L : Compiled Coin A) (x : Msg) (sel : List ℕ → Option (Fin (L.nPU x))) (l : List ℕ) :
    Option (Fin (L.nPU x)) :=
  (sel l).bind (L.leadOf x l)

theorem leadOf_some {L : Compiled Coin A} {x : Msg} {l : List ℕ} {i : Fin (L.nPU x)} {v : ℕ}
    (hi : l[i.val]? = some v) (hv : v < L.nPU x) :
    ∃ i₀ u₀ : Fin (L.nPU x), L.leadOf x l i = some i₀ ∧ l[i₀.val]? = some u₀.val := by
  obtain ⟨i₀, u₀, hhead, hle, hu₀, hu₀lt, -, -⟩ := first_of_slot hi hv
  have hi₀ : i₀ < L.nPU x := lt_of_le_of_lt hle i.isLt
  refine ⟨⟨i₀, hi₀⟩, ⟨u₀, hu₀lt⟩, ?_, hu₀⟩
  simp only [leadOf, hhead, Option.bind_some, dite_eq_left hi₀]

/-- The first slot of slot `i`'s session has `i`'s key, so `i`'s session, of which it is the first. -/
theorem of_leadOf {L : Compiled Coin A} {x : Msg} {l : List ℕ} {i i₀ : Fin (L.nPU x)}
    (h : L.leadOf x l i = some i₀) :
    L.skey x l i₀ = L.skey x l i ∧ L.sessSlots x l i₀ = L.sessSlots x l i ∧
      (L.sessSlots x l i₀).head? = some i₀.val := by
  unfold leadOf at h
  obtain ⟨j, hj, h'⟩ := Option.bind_eq_some_iff.1 h
  by_cases hjn : j < L.nPU x
  · rw [dite_eq_left hjn, Option.some.injEq] at h'
    subst h'
    obtain ⟨-, -, -, hk⟩ := mem_sessSlots.1 (List.mem_of_mem_head? hj)
    have hc : L.sessSlots x l j = L.sessSlots x l i := sessSlots_congr hk
    show L.skey x l j = L.skey x l i ∧ L.sessSlots x l j = L.sessSlots x l i ∧ (L.sessSlots x l j).head? = some j
    refine ⟨hk, hc, ?_⟩
    rw [hc]
    exact hj
  · rw [dite_eq_right hjn] at h'
    exact (Option.some_ne_none _ h'.symm).elim

/-- **A sample's sessions hold at most its slots**: the sessions at distinct first slots are disjoint, and each is
among the sample's slots. -/
theorem sum_first_le (L : Compiled Coin A) (x : Msg) (l : List ℕ) :
    ∑ i ∈ (Finset.univ : Finset (Fin (L.nPU x))).filter
        (fun i : Fin (L.nPU x) => (L.sessSlots x l i.val).head? = some i.val),
      (L.sessSlots x l i).length ≤ l.length := by
  classical
  set F := (Finset.univ : Finset (Fin (L.nPU x))).filter fun i : Fin (L.nPU x) =>
    (L.sessSlots x l i.val).head? = some i.val
  have hnd : ∀ i : Fin (L.nPU x), (L.sessSlots x l i).Nodup := fun i =>
    (L.sessSlots_pairwise x l i).imp (fun h => Nat.ne_of_lt h)
  have hdisj : (F : Set (Fin (L.nPU x))).PairwiseDisjoint fun i => (L.sessSlots x l i).toFinset := by
    intro i hi j hj hij
    refine Finset.disjoint_left.2 fun t hti htj => hij (Fin.ext ?_)
    obtain ⟨-, -, -, hki⟩ := mem_sessSlots.1 (List.mem_toFinset.1 hti)
    obtain ⟨-, -, -, hkj⟩ := mem_sessSlots.1 (List.mem_toFinset.1 htj)
    exact first_eq_of_key (Finset.mem_filter.1 hi).2 (Finset.mem_filter.1 hj).2 (hki.symm.trans hkj)
  calc ∑ i ∈ F, (L.sessSlots x l i).length
      = ∑ i ∈ F, (L.sessSlots x l i).toFinset.card :=
        Finset.sum_congr rfl fun i _ => (List.toFinset_card_of_nodup (hnd i)).symm
    _ = (F.biUnion fun i => (L.sessSlots x l i).toFinset).card := (Finset.card_biUnion hdisj).symm
    _ ≤ (Finset.range l.length).card := Finset.card_le_card fun t ht => by
        obtain ⟨i, -, hti⟩ := Finset.mem_biUnion.1 ht
        obtain ⟨u, hu, -⟩ := mem_sessSlots.1 (List.mem_toFinset.1 hti)
        exact Finset.mem_range.2 (List.getElem?_eq_some_iff.1 hu).1
    _ = l.length := Finset.card_range _

/-- **Slots with distinct keys are their own sessions**: slot `i`, its block below `n_PU`, is its session's one slot. -/
theorem sessSlots_eq_singleton {L : Compiled Coin A} {x : Msg} {l : List ℕ} {i u : ℕ}
    (hk : ∀ p q, L.skey x l p = L.skey x l q → p = q) (hi : l[i]? = some u) (hu : u < L.nPU x) :
    L.sessSlots x l i = [i] := by
  have hmem : ∀ j ∈ L.sessSlots x l i, j = i := fun j hj => by
    obtain ⟨-, -, -, h⟩ := mem_sessSlots.1 hj
    exact hk _ _ h
  have hself := self_mem_sessSlots hi hu
  have hp := L.sessSlots_pairwise x l i
  cases h : L.sessSlots x l i with
  | nil => rw [h] at hself; exact absurd hself List.not_mem_nil
  | cons a t =>
    rw [h] at hmem hp
    have ha : a = i := hmem a List.mem_cons_self
    subst ha
    cases t with
    | nil => rfl
    | cons b t' =>
      have hb : b = a := hmem b (List.mem_cons_of_mem _ List.mem_cons_self)
      have hlt : a < b := List.rel_of_pairwise_cons hp List.mem_cons_self
      omega

theorem sessBlocks_eq_singleton {L : Compiled Coin A} {x : Msg} {l : List ℕ} {i u : ℕ}
    (hk : ∀ p q, L.skey x l p = L.skey x l q → p = q) (hi : l[i]? = some u) (hu : u < L.nPU x) :
    L.sessBlocks x l i = [u] := by
  simp only [sessBlocks, sessSlots_eq_singleton hk hi hu, List.filterMap_cons, hi, List.filterMap_nil]

end Sessions

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **A sample has at most `n_sess` sessions**: their first slots have distinct keys (`first_eq_of_key`), each one of
the sample's, which hold at most `n_sess` keys (`hkeys`, a contract's `keys`). -/
theorem card_first_le {L : Compiled Coin A}
    (hkeys : ∀ x r, ((Finset.range (L.checked x r).length).image (L.skey x (L.checked x r))).card ≤ L.nSess x)
    (x : Msg) (r : Fin (L.V.k x) → Coin) :
    ((Finset.univ : Finset (Fin (L.nPU x))).filter fun i : Fin (L.nPU x) =>
      (L.sessSlots x (L.checked x r) i.val).head? = some i.val).card ≤ L.nSess x := by
  refine (Finset.card_le_card_of_injOn (fun i : Fin (L.nPU x) => L.skey x (L.checked x r) i.val) ?_ ?_).trans
    (hkeys x r)
  · intro i hi
    obtain ⟨u, hu, -⟩ := of_first (Finset.mem_filter.1 hi).2
    exact Finset.mem_image.2 ⟨i.val, Finset.mem_range.2 (List.getElem?_eq_some_iff.1 hu).1, rfl⟩
  · intro i hi j hj hij
    exact Fin.ext (first_eq_of_key (Finset.mem_filter.1 hi).2 (Finset.mem_filter.1 hj).2 hij)

/-! ## The failure event -/

/-- **Proof block `u`'s session's extraction finds a witness** at the run `r'`, on the prover's runs `run'`: `r'` shows
`u`'s session `(xS, vS)` at the block's shape, and `A`'s extractor at the session, as `sessWit` serves it, returns a
witness in `A`'s relation at `xS`. -/
def Found (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (u : Fin (L.nPU x)) (r' : Fin (L.V.k x) → Coin) (c : A.EC (L.sh x u)) : Prop :=
  ∃ xS vS, L.session x u (run' r').seen = some (xS, vS) ∧ ∃ h : A.shape xS = L.sh x u, ∃ w,
    ((serve (fun r'' => viewing (L.view x u) (L.place x u xS r' r''))
        (A.ext xS (L.coinsOf x u xS r') (cast (congrArg A.EC h.symm) c))).eval H run').1 = some w ∧ A.R xS w

/-- Proof block `v`, below `n_PU`, is at some slot of the sample at `r'`, and `found` holds of the block at the first
slot of that slot's session (`leadOf`), at the first slot's coins in `uc`. -/
def GoodAt (L : Compiled Coin A) (x : Msg)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) (r' : Fin (L.V.k x) → Coin)
    (v : ℕ) (uc : L.SlotEC x) : Prop :=
  ∃ i : Fin (L.nPU x), v < L.nPU x ∧ (L.checked x r')[i.val]? = some v ∧
    ∃ i₀ u₀ : Fin (L.nPU x), L.leadOf x (L.checked x r') i = some i₀ ∧ (L.checked x r')[i₀.val]? = some u₀.val ∧
      found u₀ r' (uc i₀ u₀)

/-- **An outer run bad for `B`**, its second stage `g`: the inner wait comes up short, or no kept inner run has a
proof block of `B` good. -/
def BadOuter (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) (B : Finset ℕ)
    (g : (Fin (L.V.k x) → Coin) → Fin (L.V.k x) → Coin) (e : L.E1 x) : Prop :=
  (accList (fun d => (run' (g d)).out = .acc) (L.kIn x) ((List.finRange (L.capIn x)).map e.1)).length < L.kIn x ∨
    AllBad (fun d => (run' (g d)).out = .acc) (L.kIn x)
      (fun d uc => ∀ v ∈ B, ¬ L.GoodAt x found (g d) v uc) ((List.finRange (L.capIn x)).map e.1) e.2

/-- **The extraction's failure event**: the statement is in scope, and the outer wait comes up short or, for some set
`B` of acceptance probability at most `δ`, every kept outer run is bad for `B`. -/
def Fails (L : Compiled Coin A) (δ : ℝ) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) (c : L.ECoins x) : Prop :=
  ∃ s Pt, L.parse x = some s ∧ L.partition s = some Pt ∧ InScope Pt s ∧
    ((accList (fun r => (run' r).out = .acc) (L.kOut x) ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x ∨
      ∃ B ∈ S₁ s.design Pt δ, AllBad (fun r => (run' r).out = .acc) (L.kOut x)
        (fun r e => L.BadOuter x run' found B (forkOn (L.stage2 x) r) e) ((List.finRange (L.capOut x)).map c.1)
        (L.toE1 x c))

/-- **The sessions' knowledge soundness** at the sample's slots: on every prover, a nonnegative weight `β` of each tape
and first-stage coins `(ω, r)`, averaging at most `Pr[B]`, such that at `(ω, r)`, over fresh second-stage coins and the
extractor's coins at each slot, the run accepts and the proof block at the slot a selection rule `sel` picks is not
found at that slot's coins with probability at most `κ_s + √(k(x)·γ_s·β(ω, r))` (`Claims.Sampling`'s `SessionKS`). It
is what `A.KS` gives at the session of the block at the picked slot, the sampled run up to it as `A`'s prover. -/
def SessionKS (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    ∃ β : Ω → (Fin (L.V.k x) → Coin) → ℝ, (∀ ω r, 0 ≤ β ω r) ∧
      Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x ∧
      ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x))),
        Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
          (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
            ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
              ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
                ¬ L.Found H x (run H P L.V x ω) u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤
          L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)

/-! ## The numbers -/

section Numbers

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

theorem kOut_pos (L : Compiled Coin A) (x : Msg) : 0 < L.kOut x := by
  unfold Compiled.kOut; positivity

theorem kIn_pos (L : Compiled Coin A) (x : Msg) : 0 < L.kIn x := by
  unfold Compiled.kIn; positivity

theorem capOut_pos (L : Compiled Coin A) (x : Msg) : 0 < L.capOut x := by
  unfold Compiled.capOut; exact Nat.mul_pos (by positivity) (L.kOut_pos x)

theorem capIn_pos (L : Compiled Coin A) (x : Msg) : 0 < L.capIn x := by
  unfold Compiled.capIn; exact Nat.mul_pos (by positivity) (L.kIn_pos x)

theorem nPU_eq {L : Compiled Coin A} {x : Msg} {s : AuditStatement} {Pt : Partition} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) : L.nPU x = Pt.blocks.length := by
  simp only [Compiled.nPU, hs, Compiled.part, hPt, Option.getD_some]

theorem κs_nonneg (L : Compiled Coin A) (x : Msg) : 0 ≤ L.κs x := maxBelow_nonneg _ _

theorem γs_nonneg (L : Compiled Coin A) (x : Msg) : 0 ≤ L.γs x := maxBelow_nonneg _ _

/-- Each proof block's bound on `A`'s `κ` is at most the sessions' (`κs`). -/
theorem κ_le_κs (L : Compiled Coin A) (x : Msg) {u : ℕ} (hu : u < L.nPU x) : L.κb x u ≤ L.κs x :=
  le_maxBelow (L.κb x) hu

/-- Each proof block's bound on `A`'s `γ`, per round, is at most the sessions' (`γs`). -/
theorem γ_le_γs (L : Compiled Coin A) (x : Msg) {u : ℕ} (hu : u < L.nPU x) :
    L.γb x u / A.rounds (L.sh x u) ≤ L.γs x :=
  le_maxBelow (fun u => L.γb x u / A.rounds (L.sh x u)) hu

theorem κ_nonneg (L : Compiled Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (x : Msg) : 0 ≤ L.κ δ x := by
  have hc := slack_pos
  unfold Compiled.κ
  exact add_nonneg (add_nonneg (mul_nonneg (by positivity) (add_nonneg hδ (L.κs_nonneg x))) (by positivity))
    (mul_nonneg hc.le (by positivity))

/-- The slack bound at the outer wait's `k_out`, over at most `2^m` sets, `m ≤ 2·n_out + 1`. -/
theorem rpow_kOut_le_slack (L : Compiled Coin A) (x : Msg) {N m : ℕ} (hN : N ≤ 2 ^ m)
    (hm : m ≤ 2 * L.nOut x + 1) : (N : ℝ) ^ (1 / (L.kOut x : ℝ)) ≤ slack :=
  rpow_le_slack hN (Nat.mul_le_mul_left _ hm)

/-- The slack bound at the inner wait's `k_in`, over at most `2^m` sets, `m ≤ 2·n_in + 1`. -/
theorem rpow_kIn_le_slack (L : Compiled Coin A) (x : Msg) {N m : ℕ} (hN : N ≤ 2 ^ m)
    (hm : m ≤ 2 * L.nIn x + 1) : (N : ℝ) ^ (1 / (L.kIn x : ℝ)) ≤ slack :=
  rpow_le_slack hN (Nat.mul_le_mul_left _ hm)

end Numbers

/-! ## The failure event's probability -/

omit [Inhabited Coin] in
/-- **No block of `B` good at an inner run**: at first-stage coins `r` where the sample misses every proof block of `B`
(each below `n_PU`) with probability at most `esc`, the sample has at most `n_PU` slots, and the block at the slot any
selection picks is not found with probability at most `κr`, the run accepts with no block of `B` good with probability
at most `esc + κr`: the union bound at the first slot of the session of the first slot holding a block of `B`. -/
theorem pr_notGood_le (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) {B : Finset ℕ}
    (hB : ∀ v ∈ B, v < L.nPU x) (hslots : ∀ r', (L.checked x r').length ≤ L.nPU x) (r : Fin (L.V.k x) → Coin)
    {esc κr : ℝ} (hesc : Pr (fun d : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ B) ≤ esc)
    (hsess : ∀ sel : List ℕ → Option (Fin (L.nPU x)),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
              ¬ found u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤ κr) :
    Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
        ∀ v ∈ B, ¬ L.GoodAt x found (forkOn (L.stage2 x) r p.1) v p.2) ≤ esc + κr := by
  let sel : List ℕ → Option (Fin (L.nPU x)) := fun l =>
    (List.finRange (L.nPU x)).find? fun i => decide (∃ v ∈ B, l[i.val]? = some v)
  have hsel : ∀ l : List ℕ, l.length ≤ L.nPU x → ∀ v ∈ l, v ∈ B →
      ∃ i, sel l = some i ∧ ∃ v ∈ B, l[i.val]? = some v := by
    intro l hl v hvl hvB
    obtain ⟨j, hj, hjv⟩ := List.getElem_of_mem hvl
    have hjv' : l[(⟨j, lt_of_lt_of_le hj hl⟩ : Fin (L.nPU x)).val]? = some v := by
      show l[j]? = some v
      rw [List.getElem?_eq_getElem hj, hjv]
    obtain ⟨i, hi⟩ := Option.isSome_iff_exists.1
      ((List.find?_isSome (p := fun i : Fin (L.nPU x) => decide (∃ v ∈ B, l[i.val]? = some v))).2
        ⟨⟨j, lt_of_lt_of_le hj hl⟩, List.mem_finRange _, decide_eq_true ⟨v, hvB, hjv'⟩⟩)
    exact ⟨i, hi, of_decide_eq_true
      (List.find?_some (p := fun i : Fin (L.nPU x) => decide (∃ v ∈ B, l[i.val]? = some v)) hi)⟩
  have hmiss := (Pr_fst (β := L.SlotEC x) (fun d : Fin (L.V.k x) → Coin =>
    ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ B)).trans_le hesc
  refine Pr_or_le (fun p hp => ?_) hmiss (hsess (L.leadSel x sel))
  by_cases hm : ∃ v ∈ L.checked x (forkOn (L.stage2 x) r p.1), v ∈ B
  · obtain ⟨v, hvl, hvB⟩ := hm
    obtain ⟨i, hi, v', hv'B, hv'⟩ := hsel _ (hslots _) v hvl hvB
    obtain ⟨i₀, u₀, hlead, hu₀⟩ := leadOf_some hv' (hB v' hv'B)
    have hsel₀ : L.leadSel x sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i₀ := by
      rw [leadSel, hi, Option.bind_some, hlead]
    exact Or.inr ⟨hp.1, i₀, hsel₀, u₀, hu₀, fun hf => hp.2 v' hv'B ⟨i, hB v' hv'B, hv', i₀, u₀, hlead, hu₀, hf⟩⟩
  · exact Or.inl fun v hvl hvB => hm ⟨v, hvl, hvB⟩

omit [Inhabited Coin] in
/-- **One outer run's inner wait**: at first-stage coins `r` where the sample misses every proof block of `B` (each
below `n_PU`) with probability at most `esc`, the sample has at most `n_PU` slots, and the block at the slot any
selection picks is not found with probability at most `κr`, the inner wait (times its acceptance) comes up short or
keeps only runs at which no block of `B` is good with probability at most `2k_in/K_in + e^{−k_in/8} + c·(esc + κr)`
(`pr_notGood_le` at each inner run). -/
theorem inner_bad_le (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) {B : Finset ℕ}
    (hB : ∀ v ∈ B, v < L.nPU x) (hslots : ∀ r', (L.checked x r').length ≤ L.nPU x) (r : Fin (L.V.k x) → Coin)
    {esc κr : ℝ} (hesc0 : 0 ≤ esc) (hκr0 : 0 ≤ κr)
    (hesc : Pr (fun d : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ B) ≤ esc)
    (hsess : ∀ sel : List ℕ → Option (Fin (L.nPU x)),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
              ¬ found u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤ κr) :
    Pr (fun d : Fin (L.V.k x) → Coin => (run' (forkOn (L.stage2 x) r d)).out = .acc) *
        Pr (fun e : L.E1 x => L.BadOuter x run' found B (forkOn (L.stage2 x) r) e) ≤
      2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (esc + κr) := by
  have hy : ∀ _u ∈ ({()} : Finset Unit),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∀ v ∈ B, ¬ L.GoodAt x found (forkOn (L.stage2 x) r p.1) v p.2) ≤ esc + κr :=
    fun _ _ => L.pr_notGood_le x run' found hB hslots r hesc hsess
  have hw := wait_bad_le (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.kIn x) (L.capIn x)
    (L.capIn_pos x) (L.kIn_pos x) ({()} : Finset Unit)
    (fun _ d uc => ∀ v ∈ B, ¬ L.GoodAt x found (forkOn (L.stage2 x) r d) v uc)
    (add_nonneg hesc0 hκr0) hy
  have hcard : (({()} : Finset Unit).card : ℝ) ^ (1 / (L.kIn x : ℝ)) ≤ slack :=
    L.rpow_kIn_le_slack x (m := 0) (by simp) (by omega)
  have hE : Pr (fun e : L.E1 x => L.BadOuter x run' found B (forkOn (L.stage2 x) r) e) =
      Pr (fun c : L.E1 x =>
        (accList (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.kIn x)
          ((List.finRange (L.capIn x)).map c.1)).length < L.kIn x ∨
        ∃ _u ∈ ({()} : Finset Unit), AllBad (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.kIn x)
          (fun d uc => ∀ v ∈ B, ¬ L.GoodAt x found (forkOn (L.stage2 x) r d) v uc)
          ((List.finRange (L.capIn x)).map c.1) c.2) :=
    Pr_congr fun e => by simp [BadOuter]
  rw [hE]
  exact hw.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hcard (add_nonneg hesc0 hκr0)))

set_option synthInstance.maxSize 1024 in
omit [Inhabited Coin] in
/-- **One tape's outer wait**: on a statement in scope at `(s, Pt)`, where the sampling design holds, the sample has at most
`n_PU ≤ n_out` slots, and the block at the slot any selection picks at first-stage coins `r` is not found with
probability at most `κr(r)`, the outer wait (times its acceptance) comes up short or, for some refused `B`, keeps only
runs bad for `B`, with probability at most `T_out + c·(T_in + c·(δ + E[κr]))`, `T = 2k/K + e^{−k/8}` at each wait's
count. -/
theorem outer_bad_le (L : Compiled Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) (s : AuditStatement)
    (Pt : Partition) (hPU : L.nPU x = Pt.blocks.length) (hun : L.nPU x ≤ L.nOut x)
    (κr : (Fin (L.V.k x) → Coin) → ℝ) (hκr0 : ∀ r, 0 ≤ κr r)
    (hdraw : ∀ B ⊆ range Pt.blocks.length, ∃ esc : (Fin (L.V.k x) → Coin) → ℝ, (∀ r, 0 ≤ esc r) ∧
      (∀ r, Pr (fun d : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ B) ≤ esc r) ∧
      Ex esc ≤ s.design.Pa Pt B)
    (hslots : ∀ r', (L.checked x r').length ≤ L.nPU x)
    (hsess : ∀ r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
              ¬ found u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤
        κr r) :
    Pr (fun r : Fin (L.V.k x) → Coin => (run' r).out = .acc) *
        Pr (fun c : (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → L.E1 x) =>
          (accList (fun r => (run' r).out = .acc) (L.kOut x) ((List.finRange (L.capOut x)).map c.1)).length <
              L.kOut x ∨
            ∃ B ∈ S₁ s.design Pt δ, AllBad (fun r => (run' r).out = .acc) (L.kOut x)
              (fun r e => L.BadOuter x run' found B (forkOn (L.stage2 x) r) e)
              ((List.finRange (L.capOut x)).map c.1) c.2) ≤
      2 * (L.kOut x : ℝ) / L.capOut x + Real.exp (-(L.kOut x : ℝ) / 8) +
        slack * (2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (δ + Ex κr)) := by
  have hc := slack_pos
  have hEκ : 0 ≤ Ex κr := Ex_nonneg hκr0
  have hy : ∀ B ∈ S₁ s.design Pt δ,
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.E1 x =>
        (run' p.1).out = .acc ∧ L.BadOuter x run' found B (forkOn (L.stage2 x) p.1) p.2) ≤
      2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (δ + Ex κr) := by
    intro B hB
    obtain ⟨hBsub, hBδ⟩ := mem_filter.1 hB
    have hBsub' : B ⊆ range Pt.blocks.length := mem_powerset.1 hBsub
    obtain ⟨esc, hesc0, hesc, hEesc⟩ := hdraw B hBsub'
    have hBlt : ∀ v ∈ B, v < L.nPU x := fun v hv => by
      rw [hPU]
      exact mem_range.1 (hBsub' hv)
    refine (Pr_acc_forkOn (L.stage2 x) (fun r => (run' r).out = .acc)
      (fun r e => L.BadOuter x run' found B (forkOn (L.stage2 x) r) e)
      fun r d e => by rw [forkOn_forkOn']).trans_le ?_
    calc Ex (fun r => Pr (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) *
          Pr (fun e : L.E1 x => L.BadOuter x run' found B (forkOn (L.stage2 x) r) e))
        ≤ Ex (fun r => 2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (esc r + κr r)) :=
          Ex_mono fun r => L.inner_bad_le x run' found hBlt hslots r (hesc0 r) (hκr0 r) (hesc r) (hsess r)
      _ = 2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (Ex esc + Ex κr) := by
          rw [Ex_add, Ex_const, Ex_mul_left, Ex_add]
      _ ≤ 2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (δ + Ex κr) := by
          have := hEesc.trans hBδ
          gcongr
  have hy0 : 0 ≤ 2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (δ + Ex κr) := by
    positivity
  have hw := wait_bad_le (fun r => (run' r).out = .acc) (L.kOut x) (L.capOut x) (L.capOut_pos x) (L.kOut_pos x)
    (S₁ s.design Pt δ) (fun B r e => L.BadOuter x run' found B (forkOn (L.stage2 x) r) e) hy0 hy
  have hcard : ((S₁ s.design Pt δ).card : ℝ) ^ (1 / (L.kOut x : ℝ)) ≤ slack :=
    L.rpow_kOut_le_slack x (card_S₁_le _ _ _) (by rw [← hPU]; omega)
  exact hw.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hcard hy0))

set_option synthInstance.maxSize 1024 in
/-- **The failure event's probability, at any found event**: over tapes, judged coins and the extractor's coins, the
judged run accepts and the extraction's failure event at `found` holds with probability at most `κ(δ) + √(γ·Pr[B])`,
when the block at the slot any selection picks is not found with probability at most `κ_s + √(k(x)·γ_s·β(ω, r))` for
a weight `β` averaging at most `Pr[B]`. Out of scope the event is empty. In scope it is independent of the judged
coins; at each tape, the outer wait's bound (`outer_bad_le`, the sampling design from `Sessions.draws` and the slots from
`Sessions.slots`) averages, by Jensen, to `κ(δ) + c²·√(k(x)·γ_s·E[β]) ≤ κ(δ) + √(γ·Pr[B])`. -/
theorem pr_fails_le_of (H : Msg → Msg) (L : Compiled Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (hS : L.Sessions H) (x : Msg)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (found : Ω → (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop)
    (β : Ω → (Fin (L.V.k x) → Coin) → ℝ) (hβ0 : ∀ ω r, 0 ≤ β ω r)
    (hβ : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x)
    (hsess : ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
              ¬ found ω u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤
        L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) :
    Pr (fun s : Ω × (Fin (L.V.k x) → Coin) × L.ECoins x =>
      (run H P L.V x s.1 s.2.1).out = .acc ∧ L.Fails δ x (run H P L.V x s.1) (found s.1) s.2.2) ≤
      L.κ δ x + Real.sqrt (L.γ x * PrB H L.V P x) := by
  refine (Pr_indep (fun ω r => (run H P L.V x ω r).out = .acc)
    (fun ω c => L.Fails δ x (run H P L.V x ω) (found ω) c)).trans_le ?_
  have hc := slack_pos
  have hκ0 := L.κ_nonneg hδ x
  have hγ0 : 0 ≤ (L.V.k x : ℝ) * L.γs x := mul_nonneg (Nat.cast_nonneg _) (L.γs_nonneg x)
  by_cases hsc : ∃ s Pt, L.parse x = some s ∧ L.partition s = some Pt ∧ InScope Pt s
  swap
  · have h0 : ∀ ω, Pr (fun c : L.ECoins x => L.Fails δ x (run H P L.V x ω) (found ω) c) = 0 := fun ω =>
      (Pr_congr fun c => ⟨fun ⟨s, Pt, h1, h2, h3, _⟩ => hsc ⟨s, Pt, h1, h2, h3⟩, False.elim⟩).trans Pr_false
    simp only [h0, mul_zero, Ex_const]
    exact add_nonneg hκ0 (Real.sqrt_nonneg _)
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  have hper : ∀ ω, Pr (fun r => (run H P L.V x ω r).out = .acc) *
      Pr (fun c : L.ECoins x => L.Fails δ x (run H P L.V x ω) (found ω) c) ≤
      L.κ δ x + slack ^ 2 * Ex (fun r => Real.sqrt (L.V.k x * L.γs x * β ω r)) := by
    intro ω
    have hF : Pr (fun c : L.ECoins x => L.Fails δ x (run H P L.V x ω) (found ω) c) =
        Pr (fun c : (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → L.E1 x) =>
          (accList (fun r => (run H P L.V x ω r).out = .acc) (L.kOut x)
            ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x ∨
            ∃ B ∈ S₁ s.design Pt δ, AllBad (fun r => (run H P L.V x ω r).out = .acc) (L.kOut x)
              (fun r e => L.BadOuter x (run H P L.V x ω) (found ω) B (forkOn (L.stage2 x) r) e)
              ((List.finRange (L.capOut x)).map c.1) c.2) := by
      refine Eq.trans ?_ (Claims.Sampling.Pr_equiv (L.ecoinsEquiv x) _)
      refine Pr_congr fun c => ⟨fun ⟨s', Pt', hs', hPt', _, h⟩ => ?_, fun h => ⟨s, Pt, hs, hPt, hin, h⟩⟩
      rw [hs, Option.some.injEq] at hs'
      subst hs'
      rw [hPt, Option.some.injEq] at hPt'
      subst hPt'
      exact h
    rw [hF]
    refine (L.outer_bad_le hδ x (run H P L.V x ω) (found ω) s Pt (nPU_eq hs hPt) (hS.unions x)
      (fun r => L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r))
      (fun r => add_nonneg (L.κs_nonneg x) (Real.sqrt_nonneg _))
      (hS.draws x s Pt hs hPt hin) (hS.slots x) (hsess ω)).trans (le_of_eq ?_)
    rw [Ex_add, Ex_const]
    unfold Compiled.κ
    ring
  calc Ex (fun ω => Pr (fun r => (run H P L.V x ω r).out = .acc) *
        Pr (fun c : L.ECoins x => L.Fails δ x (run H P L.V x ω) (found ω) c))
      ≤ Ex (fun ω => L.κ δ x + slack ^ 2 * Ex (fun r => Real.sqrt (L.V.k x * L.γs x * β ω r))) := Ex_mono hper
    _ = L.κ δ x + slack ^ 2 *
          Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2)) := by
        rw [Ex_add, Ex_const, Ex_mul_left,
          Ex_prod (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2))]
    _ ≤ L.κ δ x + slack ^ 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) := by
        refine add_le_add le_rfl (mul_le_mul_of_nonneg_left ?_ (by positivity))
        refine (Ex_sqrt_le _ fun p => mul_nonneg hγ0 (hβ0 _ _)).trans (Real.sqrt_le_sqrt ?_)
        rw [Ex_mul_left]
        exact mul_le_mul_of_nonneg_left hβ hγ0
    _ = L.κ δ x + Real.sqrt (L.γ x * PrB H L.V P x) := by
        rw [show L.γ x * PrB H L.V P x = (slack ^ 2) ^ 2 * (L.V.k x * L.γs x * PrB H L.V P x) by
            unfold Compiled.γ; ring,
          Real.sqrt_mul (pow_nonneg (pow_nonneg hc.le 2) 2) (L.V.k x * L.γs x * PrB H L.V P x),
          Real.sqrt_sq (pow_nonneg hc.le 2)]

set_option synthInstance.maxSize 1024 in
/-- **The acceptance-probability composition**: from the sessions' knowledge soundness (`SessionKS`) and the sampling design
(`Sessions.draws`), over tapes, judged coins and the extractor's coins, the judged run accepts and the extraction fails
(for some set of proof blocks of acceptance probability at most `δ`, no kept run finds a block of it) with probability
at most `κ(δ) + √(γ·Pr[B])`, `κ(δ) = c²·(δ + κ_s) + (2k_out/K_out + e^{−k_out/8}) + c·(2k_in/K_in + e^{−k_in/8})`.
-/
theorem pr_fails_le (H : Msg → Msg) (L : Compiled Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (hsess : L.SessionKS H)
    (hS : L.Sessions H) (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun s : Ω × (Fin (L.V.k x) → Coin) × L.ECoins x =>
      (run H P L.V x s.1 s.2.1).out = .acc ∧
        L.Fails δ x (run H P L.V x s.1) (L.Found H x (run H P L.V x s.1)) s.2.2) ≤
      L.κ δ x + Real.sqrt (L.γ x * PrB H L.V P x) := by
  obtain ⟨β, hβ0, hβ, hκ⟩ := hsess x Ω P
  exact L.pr_fails_le_of H hδ hS x Ω P (fun ω => L.Found H x (run H P L.V x ω)) β hβ0 hβ hκ

/-! ## `Compiled.KS` from the split -/

set_option synthInstance.maxSize 1024 in
/-- **`Compiled.KS` from the split and the sessions' bound**: if, at every `δ ≥ 0`, an accepted run whose extraction
fails is in the failure event (at each prover's and tape's found event `found`) or its extraction's own calls collide,
and on every prover a weight `β` averaging at most `Pr[B]` bounds the block at the slot any selection picks not being
found by `κ_s + √(k_V·γ_s·β)`, then the compiled verifier is knowledge sound (`Sampling.KSLin_of_ownColl` at
`pr_fails_le_of`). -/
theorem KS_of_split {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H)
    (found : ∀ (x : Msg) (Ω : Type), Prover Ω → Ω → (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) →
      A.EC (L.sh x u) → Prop)
    (hsplit : ∀ δ : ℝ, 0 ≤ δ → ∀ x (Ω : Type) (P : Prover Ω) (s : Ω × (Fin (L.V.k x) → Coin) × L.ECoins x),
      (run H P L.V x s.1 s.2.1).out = .acc →
        ¬ (Sampling.R H L.commit L.parse L.partition δ).ext x (L.E.eval H P x s.1 s.2.1 s.2.2).1 →
        L.Fails δ x (run H P L.V x s.1) (found x Ω P s.1) s.2.2 ∨ Sampling.OwnColl H L.E P x s.1 s.2.1 s.2.2)
    (hsess : ∀ x (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      ∃ β : Ω → (Fin (L.V.k x) → Coin) → ℝ, (∀ ω r, 0 ≤ β ω r) ∧
        Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x ∧
        ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x))),
          Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
            (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
              ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
                ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
                  ¬ found x Ω P ω u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤
            L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) :
    L.KS H := by
  intro δ hδ x _
  exact Sampling.KSLin_of_ownColl H L.V (Sampling.R H L.commit L.parse L.partition δ) L.E (L.κ δ) L.γ
    (fun x Ω P s => (run H P L.V x s.1 s.2.1).out = .acc ∧ L.Fails δ x (run H P L.V x s.1) (found x Ω P s.1) s.2.2)
    (fun x Ω P s hacc hfail => (hsplit δ hδ x Ω P s hacc hfail).imp_left fun h => ⟨hacc, h⟩)
    (fun x Ω _ _ P => by
      obtain ⟨β, hβ0, hβ, hκ⟩ := hsess x Ω P
      exact L.pr_fails_le_of H hδ hS x Ω P (found x Ω P) β hβ0 hβ hκ) x

set_option synthInstance.maxSize 1024 in
/-- **`Compiled.KS` from the sessions' KS and the split**: `SessionKS` is `KS_of_split`'s bound at the found event
`Found`, so `KSOf` at a compiled verifier needs only the split into `Fails` at `Found` or an own-call collision. -/
theorem KS_of_sessionKS {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) (hSK : L.SessionKS H)
    (hsplit : ∀ δ : ℝ, 0 ≤ δ → ∀ x (Ω : Type) (P : Prover Ω) (s : Ω × (Fin (L.V.k x) → Coin) × L.ECoins x),
      (run H P L.V x s.1 s.2.1).out = .acc →
        ¬ (Sampling.R H L.commit L.parse L.partition δ).ext x (L.E.eval H P x s.1 s.2.1 s.2.2).1 →
        L.Fails δ x (run H P L.V x s.1) (L.Found H x (run H P L.V x s.1)) s.2.2 ∨
          Sampling.OwnColl H L.E P x s.1 s.2.1 s.2.2) :
    L.KS H :=
  KS_of_split hS (fun x _ P ω => L.Found H x (run H P L.V x ω)) hsplit (fun x Ω _ _ P => hSK x Ω P)

end Compiled

end Claims.SamplingGeneric
