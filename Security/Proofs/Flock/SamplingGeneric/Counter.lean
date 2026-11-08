module

public import Proofs.Flock.SamplingGeneric.Assemble
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
import Mathlib.Tactic.ENatToNat
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
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Cost

/-!
# Why the contract counts the opening checks

`CompilerEfficient` bounds the compiled extractor's own SHA-512 calls by `k_out·k_in·n_PU·openCalls`, the bound
`compilerBits_of` reads. It takes `Compiled.OpenCallsLe`: without it `openCalls` bounds nothing, since no other part of
the contract relates it to `commit` or `reveal`, and checking what one session's witness reveals can cost more SHA-512
calls than `openCalls` (`not_ownLe_openCalls`).

`L₀ n oc` is the compiled verifier it is read at: one proof block, a verifier of one round that accepts, a proof system
of one round whose extractor replays the judged run and returns its witness, a sample of `n` copies of the block, a
reveal of one opening whose check is one SHA-512 call, and `openCalls = oc`. `L₀ 1 oc` meets the contract
(`sessions₀`); its extractor's own calls are `k_out·k_in` (`ownLog_E₀`), against the bound `k_out·k_in·0` at `oc = 0`.
They are counted at any compiled verifier: at an accepted run, the outer body's at the kept outer runs
(`Compiled.length_ownLog_E`), `k_out` of them when every run accepts (`Compiled.length_outerList_all`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Claims.Sampling
open Claims.Recursion (serve viewing)

theorem ownLog_flatMapRP_const {R β γ : Type} (H : Msg → Msg) (run : R → Trace) (f : β → RP R (List γ)) (m : ℕ)
    (l : List β) (h : ∀ b ∈ l, (ownLog H run (f b)).length = m) :
    (ownLog H run (flatMapRP f l)).length = l.length * m := by
  rw [ownLog_flatMapRP, List.map_congr_left h, List.map_const', List.sum_replicate, smul_eq_mul]

/-- A wait at which every run accepts keeps the first `k` runs. -/
theorem waitAccC_length_all {R : Type} (H : Msg → Msg) (run : R → Trace) (hrun : ∀ r, (run r).out = .acc) (k : ℕ)
    (l : List R) : ((waitAccC k l []).eval H run).1.length = min k l.length := by
  rw [waitAccC_eval_nil, List.length_map, accList, List.filter_eq_self.2 fun r _ => decide_eq_true (hrun r),
    List.length_take]

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem kOut_le_capOut (L : Compiled Coin A) (x : Msg) : L.kOut x ≤ L.capOut x := by
  unfold Compiled.capOut; exact Nat.le_mul_of_pos_left _ (by positivity)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem kIn_le_capIn (L : Compiled Coin A) (x : Msg) : L.kIn x ≤ L.capIn x := by
  unfold Compiled.capIn; exact Nat.le_mul_of_pos_left _ (by positivity)

/-- At an accepted run, the extractor's own calls are its outer body's at the kept outer runs. -/
theorem length_ownLog_E (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (L.V.k x) → Coin) (c : L.E.C x) {s : AuditStatement} (hs : L.parse x = some s)
    (hacc : (run H P L.V x ω r₀).out = .acc) :
    (ownLog H (run H P L.V x ω) (L.E.ext x r₀ c)).length =
      (ownLog H (run H P L.V x ω) (flatMapRP (L.outerBody x s c) (L.outerList H x (run H P L.V x ω) c))).length := by
  have hacc' : (run H P L.V x ω r₀).seen.out = .acc := hacc
  simp only [E, ownLog, hs]
  rw [ite_eq_left hacc', ownLog_bind, ownLog_bind, ownLog_of_noHash _ _ (NoHash.waitAccC _ _ _), List.nil_append,
    List.length_append]
  exact Nat.add_zero _

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- At runs that all accept, the outer wait keeps `k_out` runs. -/
theorem length_outerList_all (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (c : L.ECoins x) (hrun : ∀ r, (run' r).out = .acc) : (L.outerList H x run' c).length = L.kOut x := by
  unfold outerList
  rw [List.length_zip, List.length_finRange, waitAccC_length_all H run' hrun, List.length_map, List.length_finRange,
    Nat.min_eq_left (L.kOut_le_capOut x), min_self]

end Compiled

namespace Counter

/-! ## The compiled verifier -/

/-- A verifier of one round that accepts whatever the prover sends. -/
def V₁ : Verifier Unit := ⟨fun _ => 1, fun _ _ _ => .ret .acc⟩

/-- A program of one input gate and no outputs, sampled `all`. -/
def s₀ : AuditStatement := ⟨[], ⟨[.input], []⟩, ⟨"", 0, []⟩, ⟨[.all]⟩, []⟩

/-- One class of one stratum of one proof block: the input gate. -/
def P₀ : Partition := ⟨[[[[0]]]]⟩

/-- A commitment whose check of an opening is one SHA-512 call, and passes. -/
def Cm₀ : CommitmentScheme Bool := ⟨fun _ _ => .query [] fun _ => .ret true, fun _ _ _ => []⟩

/-- A proof system of one round whose extractor replays the judged run and, if it accepted, makes one wait that keeps no
rerun and returns its one witness; its relation is empty. -/
def A₀ : Arg Unit where
  Sh := Unit
  shape _ := ()
  V := V₁
  rounds _ := 1
  wf _ := rfl
  W := Unit
  R _ _ := False
  Op := Unit
  opens _ := []
  assemble _ _ := (some (), 0)
  check _ _ := true
  waits _ := 1
  k _ := 0
  cap _ := 1
  κ _ := 0
  γ _ := 0

/-- `A₀`'s extractor: the replay, then its witness if the replay accepted. -/
theorem A₀_ext (x : Msg) (r₀ : Fin (A₀.V.k x) → Unit) (c : A₀.EC (A₀.shape x)) :
    A₀.ext x r₀ c = .rerun r₀ fun v₀ => if v₀.out = .acc then .ret (some ()) else .ret none := rfl

/-- **The compiled verifier**: `V₁`, the statement `s₀` and partition `P₀` at every message, a sample of `n` copies of
the one proof block, each session the whole run at its own statement, a reveal of one opening, and `openCalls = oc`. -/
def L₀ (n oc : ℕ) : Compiled Unit A₀ where
  V := V₁
  parse _ := some s₀
  scopeCheck _ := true
  partition _ := some P₀
  commit := Cm₀
  stage2 _ := Finset.univ
  checked _ _ := List.replicate n 0
  sh _ _ := ()
  rounds _ _ := Finset.univ
  session x _ v := some (x, v)
  iroots _ _ _ := []
  reveal _ _ _ _ := [([], ⟨0, true, []⟩)]
  openCalls _ := oc
  nOut _ := 1
  nIn _ := 1
  κb _ _ := 0
  γb _ _ := 0
  Tape := Unit
  P _ := ⟨fun _ _ _ => .ret []⟩
  δc _ _ := 0

/-! ## It meets the contract -/

theorem unitClass₀ : unitClass s₀.prog [0] = some 16 := by
  obtain ⟨t, ht⟩ : ∃ t, sizeClasses = 16 :: t := by
    rw [sizeClasses, List.range_succ_eq_map, List.map_cons]
    exact ⟨_, rfl⟩
  rw [unitClass, ht]
  rfl

theorem inScope₀ : InScope P₀ s₀ := by
  refine ⟨⟨fun g b hb i hi => ?_, fun o ho => by simp [s₀] at ho⟩,
    ⟨by simp [Partition.gates, Partition.blocks, P₀], fun g hg => ?_, fun g hg _ => ?_, fun u hu => ?_,
      fun u hu => ?_⟩, rfl, fun p hp => ?_⟩
  · rcases g with _ | g
    · simp only [s₀, List.getElem?_cons_zero, Option.some.injEq] at hb
      subst hb
      simp [BGate.reads] at hi
    · simp [s₀] at hb
  · simp [Partition.gates, Partition.blocks, P₀] at hg
    simp [hg, s₀]
  · simp [s₀] at hg
    simp [hg, Partition.gates, Partition.blocks, P₀]
  · simp [Partition.blocks, P₀] at hu
    simp [hu]
  · simp [Partition.blocks, P₀] at hu
    subst hu
    exact ⟨16, unitClass₀, by decide⟩
  · simp [s₀, P₀] at hp
    subst hp
    exact trivial

theorem pa₀ {B : Finset ℕ} (hB : B ⊆ Finset.range 1) : s₀.design.Pa P₀ B = OneStage.Pa [1] .all B := by
  have hw : window 0 1 B = B := by
    ext v
    simp only [window, Finset.mem_image, Finset.mem_filter]
    constructor
    · rintro ⟨a, ⟨ha, -⟩, rfl⟩
      simpa using ha
    · intro hv
      exact ⟨v, ⟨hv, Nat.zero_le _, by simpa using Finset.mem_range.1 (hB hv)⟩, Nat.sub_zero v⟩
  have h : s₀.design.Pa P₀ B = OneStage.Pa [1] .all (window 0 1 B) * classesPa [] [] 1 B := rfl
  rw [h, hw]
  exact mul_one _

theorem draws₀ {n : ℕ} (hn : 0 < n) (oc : ℕ) : ∀ x s P, (L₀ n oc).parse x = some s →
    (L₀ n oc).partition s = some P → InScope P s → ∀ B ⊆ Finset.range P.blocks.length,
      ∃ esc : (Fin ((L₀ n oc).V.k x) → Unit) → ℝ, (∀ r, 0 ≤ esc r) ∧
        (∀ r, Pr (fun d : Fin ((L₀ n oc).V.k x) → Unit =>
          ∀ u ∈ (L₀ n oc).checked x (forkOn ((L₀ n oc).stage2 x) r d), u ∉ B) ≤ esc r) ∧
        Ex esc ≤ s.design.Pa P B := by
  intro x s P hs hP _ B hB
  obtain rfl : s₀ = s := Option.some.inj hs
  obtain rfl : P₀ = P := Option.some.inj hP
  have hB1 : B ⊆ Finset.range 1 := hB
  rw [pa₀ hB1]
  refine ⟨fun _ => OneStage.Pa [1] .all B, fun _ => ?_, fun r => ?_, (Ex_const _).le⟩
  · show 0 ≤ OneStage.Pa [1] .all B
    simp only [OneStage.Pa]
    split_ifs <;> norm_num
  · show Pr _ ≤ OneStage.Pa [1] .all B
    by_cases hB0 : B = ∅
    · have h1 : OneStage.Pa [1] .all B = 1 := by simp [OneStage.Pa, hB0]
      rw [h1]
      exact Pr_le_one _
    · obtain ⟨v, hv⟩ := Finset.nonempty_iff_ne_empty.2 hB0
      have hv0 : v = 0 := by have := Finset.mem_range.1 (hB1 hv); omega
      subst hv0
      have h0 : OneStage.Pa [1] .all B = 0 := by simp [OneStage.Pa, hB0]
      rw [h0]
      exact (Pr_congr (q := fun _ => False) fun d =>
        ⟨fun h => h 0 (List.mem_replicate.2 ⟨by omega, rfl⟩) hv, False.elim⟩).trans_le Pr_false.le

/-- **`L₀ 1 oc` meets the contract**: it samples the block once. -/
theorem sessions₀ (H : Msg → Msg) (oc : ℕ) : (L₀ 1 oc).Sessions H where
  draws := draws₀ Nat.one_pos oc
  slots _ _ := le_rfl
  shared _ _ _ _ _ _ _ _ _ _ _ _ _ := rfl
  keys _ _ := Finset.card_image_le.trans (Finset.card_range _).le
  unions _ := le_rfl
  refuses _ _ _ _ _ _ := ⟨s₀, P₀, rfl, rfl, inScope₀⟩
  interior _ _ _ _ _ _ _ := rfl
  fixed _ _ _ _ := ⟨Finset.Subset.refl _, fun _ => rfl⟩
  runs x _ P ω _ _ _ := ⟨x, ⟨fun _ _ _ => P.next x [] ω⟩, rfl, le_rfl, le_rfl, (Finset.card_fin 1).symm,
    fun _ _ _ => rfl, fun _ _ => ⟨rfl, fun q hq => by
      have h : q ∈ ([] : List (Msg × Msg)) := hq
      simp at h⟩⟩
  opens _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hR := False.elim hR

/-! ## Its extractor's own calls -/

/-- At runs that all accept, a session's extraction at its one proof block is the replay, served as one rerun, then the
check of the one opening its witness reveals: one SHA-512 call. -/
theorem ownLog_sessExt₀ (H : Msg → Msg) (n oc : ℕ) (x : Msg) (run' : (Fin ((L₀ n oc).V.k x) → Unit) → Trace)
    (hrun : ∀ r, (run' r).out = .acc) (s : AuditStatement) (u : ℕ) (r : Fin ((L₀ n oc).V.k x) → Unit) (v : Seen)
    (c : A₀.EC ((L₀ n oc).sh x u)) :
    (ownLog H run' ((L₀ n oc).sessExt x s [u] u r v c)).length = 1 := by
  have hacc : ∀ r', (run' r').seen.out = .acc := hrun
  unfold Compiled.sessExt Compiled.sessWit
  have hs : (L₀ n oc).session x u v = some (x, v) := rfl
  have hv : ∀ w, (L₀ n oc).view x u w = w := fun _ => rfl
  simp only [hs]
  rw [dite_eq_left (show A₀.shape x = (L₀ n oc).sh x u from rfl), A₀_ext]
  simp only [serve, viewing, RP.bind, ownLog, hv, hacc, ite_true]
  have hr : (L₀ n oc).reveal s u v () = [([], ⟨0, true, []⟩)] := rfl
  have hc : (L₀ n oc).commit = Cm₀ := rfl
  simp only [flatMapRP, checkRevealed, hr, hc]
  rfl

theorem ownLog_innerBody₀ (H : Msg → Msg) (n oc : ℕ) (x : Msg) (run' : (Fin ((L₀ n oc).V.k x) → Unit) → Trace)
    (hrun : ∀ r, (run' r).out = .acc) (s : AuditStatement) (c : (L₀ n oc).ECoins x) (j : Fin ((L₀ n oc).kOut x))
    (li : Fin ((L₀ n oc).kIn x) × ((Fin ((L₀ n oc).V.k x) → Unit) × Seen)) :
    (ownLog H run' ((L₀ n oc).innerBody x s c j li)).length = min ((L₀ n oc).nPU x) n := by
  unfold Compiled.innerBody Compiled.runExt
  refine Eq.trans (ownLog_flatMapRP_const H run' _ 1 _ fun iu hiu => ?_) ?_
  · obtain ⟨i, u⟩ := iu
    obtain rfl : u = 0 := (List.mem_replicate.1 (List.of_mem_zip hiu).2).2
    have hi : ((L₀ n oc).checked x li.2.1)[i.val]? = some 0 := getElem?_of_mem_zip_finRange hiu
    have hlt : 0 < (L₀ n oc).nPU x := Nat.one_pos
    have hk : ∀ p q, (L₀ n oc).skey x ((L₀ n oc).checked x li.2.1) p =
        (L₀ n oc).skey x ((L₀ n oc).checked x li.2.1) q → p = q := fun _ _ h => h
    dsimp only
    rw [dite_eq_left (show 0 < (L₀ n oc).nPU x ∧
        ((L₀ n oc).sessSlots x ((L₀ n oc).checked x li.2.1) i).head? = some i.val from
          ⟨hlt, by simp only [Compiled.sessSlots_eq_singleton hk hi hlt, List.head?_cons]⟩),
      Compiled.sessBlocks_eq_singleton hk hi hlt]
    exact ownLog_sessExt₀ H n oc x run' hrun s 0 _ _ _
  · show ((List.finRange ((L₀ n oc).nPU x)).zip (List.replicate n 0)).length * 1 = min ((L₀ n oc).nPU x) n
    rw [List.length_zip, List.length_finRange, List.length_replicate, Nat.mul_one]

theorem ownLog_outerBody₀ (H : Msg → Msg) (n oc : ℕ) (x : Msg) (run' : (Fin ((L₀ n oc).V.k x) → Unit) → Trace)
    (hrun : ∀ r, (run' r).out = .acc) (s : AuditStatement) (c : (L₀ n oc).ECoins x)
    (jo : Fin ((L₀ n oc).kOut x) × ((Fin ((L₀ n oc).V.k x) → Unit) × Seen)) :
    (ownLog H run' ((L₀ n oc).outerBody x s c jo)).length = (L₀ n oc).kIn x * min ((L₀ n oc).nPU x) n := by
  unfold Compiled.outerBody
  rw [ownLog_bind, ownLog_of_noHash H run' (NoHash.waitAccC _ _ _), List.nil_append]
  refine Eq.trans (ownLog_flatMapRP_const H run' _ _ _ fun li _ => ownLog_innerBody₀ H n oc x run' hrun s c jo.1 li) ?_
  rw [List.length_zip, List.length_finRange, waitAccC_length_all H run' hrun, List.length_map, List.length_finRange,
    Nat.min_eq_left ((L₀ n oc).kIn_le_capIn x), min_self]

/-- **The extractor's own calls are `k_out·k_in·min(n_PU, n)`**: at every run, each of the `k_out` kept outer runs
keeps `k_in` inner runs, at each of which each of the first `n_PU` slots of the sample costs one call. -/
theorem ownLog_E₀ (H : Msg → Msg) (n oc : ℕ) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin ((L₀ n oc).V.k x) → Unit) (c : (L₀ n oc).E.C x) :
    (ownLog H (run H P (L₀ n oc).V x ω) ((L₀ n oc).E.ext x r₀ c)).length =
      (L₀ n oc).kOut x * ((L₀ n oc).kIn x * min ((L₀ n oc).nPU x) n) := by
  have hrun : ∀ r, (run H P (L₀ n oc).V x ω r).out = .acc := fun _ => rfl
  rw [Compiled.length_ownLog_E H (L₀ n oc) P x ω r₀ c (s := s₀) rfl (hrun r₀),
    ownLog_flatMapRP_const H _ _ _ _ fun jo _ => ownLog_outerBody₀ H n oc x _ hrun s₀ c jo,
    Compiled.length_outerList_all H (L₀ n oc) x _ c hrun]

/-- The prover the bound is read at: it sends the empty message. -/
def P₁ : Prover Unit := ⟨fun _ _ _ => .ret []⟩

/-! ## The bound fails -/

/-- **Without repeats, `openCalls` bounds nothing**: `L₀ 1 0` samples its one block once, and checking its reveal
costs one SHA-512 call while `openCalls = 0`, so its extractor's `k_out·k_in` own calls exceed the bound
`k_out·k_in·0`. -/
theorem not_ownLe_openCalls (H : Msg → Msg) :
    ¬ ExtractOwnLe H (L₀ 1 0).E
      (fun x => (L₀ 1 0).kOut x * (L₀ 1 0).kIn x * (L₀ 1 0).nPU x * (L₀ 1 0).openCalls x) := by
  intro h
  obtain ⟨c⟩ : Nonempty ((L₀ 1 0).E.C []) := inferInstance
  have h3 := (ownLog_E₀ H 1 0 P₁ [] () (fun _ => ()) c).symm.trans_le (h [] Unit P₁ () (fun _ => ()) c)
  have hb : (L₀ 1 0).openCalls [] = 0 := rfl
  have hm : min ((L₀ 1 0).nPU []) 1 = 1 := rfl
  have hk := (L₀ 1 0).kOut_pos []
  have hk' := (L₀ 1 0).kIn_pos []
  simp only [hb, hm, mul_zero, mul_one] at h3
  exact absurd h3 (Nat.not_le.2 (Nat.mul_pos hk hk'))

theorem not_efficient_openCalls (H : Msg → Msg) : ¬ (L₀ 1 0).Efficient H :=
  fun h => not_ownLe_openCalls H h.2.2

/-- `L₀ 1 0`'s sample has no repeats. -/
theorem nodup_checked₀ (x : Msg) (r : Fin ((L₀ 1 0).V.k x) → Unit) : ((L₀ 1 0).checked x r).Nodup :=
  List.nodup_singleton 0

end Counter

end Claims.SamplingGeneric
