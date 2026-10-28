import Proofs.Core.AndSize
import Proofs.Pouw.ServedValues
import Proofs.Pouw.PeriodJ10R.Honest

/-!
# `RightValues` at an honest period, and against a verifier that accepts everything

The two witnesses of `Properties.Pouw.ServedValues.RightValues`, on `Proofs.Pouw.PeriodJ10R.Honest`'s J10-R
period. Its declared circuit has `N = N_p` committed words of no operands, one a block. The honest registered
declaration (`andDecl N`) is a Boolean circuit of two input gates and `N` output gates, each the AND of the two
(`andCircuit N`), and its lowering is itself (`Verity.RegisteredCircuit.D`). The honest map (`blocks N`) has block `t` AND
gate `t + 2` and word `t` its one bit. No other protocol credits a gate (the ledger is empty), so each block's credit
size is `1` (`Verity.SecurityProofs.ToCircuitAnd`: C-Flock's `and a b` of two distinct gates is an AND gate).

* `right_values_honest`: the honest period meets every premise of `RightValues`, `j10r_meets`' without
  `UnitFixed`, so the guarantee's bound holds there. The premises are conditional on `Credit.window`'s verdict on the
  windows' inputs, as `j10r_meets` is (`hacc`, `hsalt`, `hdraw`).
* `right_values_nontrivial`: the conclusion fails when the verifier accepts everything. With every window credited
  (`hacc`), every block live and the play accepted, a prover that skips every block whole has no right block, while
  the right values must cover `(1 − ε(n))₊·N_p·n > 0` AND gates once `Λ·ln(1/δ_s) < n`. So the conclusion's event
  holds on every path, and its probability `1` exceeds `δ_s < 1`.
-/
namespace Properties.Pouw.ServedValues.SecurityProofs.HonestValues

open Properties.Pouw.PeriodJ10R Properties.Pouw.PeriodJ10R.SecurityProofs Properties.Pouw.ServedValues
open Properties.Pouw.ServedValues.SecurityProofs Properties.Pouw.PeriodJ10R.SecurityProofs.HonestJ10R
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposePeriod

/-! ## The declared Boolean circuit and its blocks -/

/-- The first input gate. -/
def in0 {N : ℕ} : Fin (N + 2) := ⟨0, by omega⟩

/-- The second input gate. -/
def in1 {N : ℕ} : Fin (N + 2) := ⟨1, by omega⟩

/-- Two input gates, then `N` AND gates of the two, every one an output. -/
def andCircuit (N : ℕ) : FlockSoundness.Audit.Circuit where
  N := N + 2
  op g := if g.val < 2 then .input else .and in0 in1
  topo g a h := by
    by_cases hg : g.val < 2
    · simp [hg, FlockSoundness.Audit.Op.args] at h
    · simp only [hg, if_false, FlockSoundness.Audit.Op.args, List.mem_cons, List.not_mem_nil, or_false] at h
      rcases h with rfl | rfl
      · exact Fin.lt_def.mpr (show 0 < g.val by omega)
      · exact Fin.lt_def.mpr (show 1 < g.val by omega)
  outputs := Finset.univ.filter fun g => 2 ≤ g.val

/-- The declaration of `andCircuit N`, with nothing registered. -/
def andDecl (N : ℕ) : Verity.RegisteredCircuit := ⟨(andCircuit N).toCircuit, ∅, by simp⟩

/-- Block `t`'s gate, the AND gate `t + 2`. -/
def gateOf {N : ℕ} (t : Fin N) : Fin (andDecl N).D.N := ⟨t.val + 2, show t.val + 2 < N + 2 by omega⟩

theorem op_gateOf {N : ℕ} (t : Fin N) : (andCircuit N).op (gateOf t) = .and in0 in1 := by
  simp only [andCircuit, gateOf]
  split_ifs with h
  · exact absurd h (by omega)
  · rfl

theorem isAnd_gateOf {N : ℕ} (t : Fin N) : (andDecl N).D.IsAnd (gateOf t) :=
  (Verity.SecurityProofs.ToCircuitAnd (andCircuit N) (gateOf t)).2.1 in0 in1 (op_gateOf t)
    fun h => absurd (congrArg Fin.val h) (by simp [in0, in1])

theorem gateOf_output {N : ℕ} (t : Fin N) : gateOf t ∈ (andDecl N).D.outputs :=
  Finset.mem_filter.mpr ⟨Finset.mem_univ _, show 2 ≤ t.val + 2 by omega⟩

theorem andSize_gateOf {N : ℕ} (t : Fin N) : (andDecl N).D.andSize {gateOf t} = 1 := by
  unfold Verity.Circuit.andSize
  rw [Finset.filter_singleton, if_pos (isAnd_gateOf t), Finset.card_singleton]

/-- **The honest map**: block `t` is AND gate `t + 2`, and word `i` is its one bit, that gate. The lowering is the
declaration's own circuit, which is Boolean. -/
def blocks (N : ℕ) : BlockMap (circuit N) N Verity.RegisteredCircuit.D (andDecl N) where
  gates t := {gateOf t}
  bits i := if h : i < N then {gateOf ⟨i, h⟩} else ∅
  disjoint t t' h := Finset.disjoint_singleton.mpr fun he => h (Fin.ext (by
    have := congrArg Fin.val he
    simp only [gateOf] at this
    omega))
  computed t := by
    intro g hg
    rw [Finset.mem_singleton] at hg
    subst hg
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, (isAnd_gateOf t).1⟩
  nonempty t i hi := by
    rw [unitWords_eq, if_pos t.isLt, List.mem_singleton] at hi
    subst hi
    rw [dif_pos t.isLt]
    exact Finset.singleton_nonempty _
  exits t i hi := by
    rw [unitWords_eq, if_pos t.isLt, List.mem_singleton] at hi
    subst hi
    rw [dif_pos t.isLt]
    intro g hg
    rw [Finset.mem_singleton] at hg
    subst hg
    exact Finset.mem_filter.mpr ⟨Finset.mem_singleton_self _, Or.inl (gateOf_output t)⟩

/-- **Every block's credit size is `1`** with no gate credited elsewhere: its one gate is an AND gate. -/
theorem size_blocks {N : ℕ} (t : Fin N) : (blocks N).size ∅ t = 1 := by
  unfold BlockMap.size
  rw [Finset.sdiff_empty]
  exact andSize_gateOf t

/-! ## The honest witness -/

section Witnesses

variable {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB)
  (unitOf : Fin nB → ℕ) (regOf : Bool → Registered)
  (evOf : Bool → Equiv.Perm (Fin (tileCount σ sT)) × Unit → Answered) (enc : Bool → ByteArray)

/-- **`RightValues`'s bound at the honest J10-R period**: `j10r_meets` gives every
premise but `UnitFixed`, which the guarantee doesn't take, at `N_p` one-word blocks, one round, `ρ = 1`, any
`δ_s > 0` and `δ_bind = δ_ks = 0`. -/
theorem right_values_honest (hKd : 1 ≤ k sT) (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β))
    (hsalt : ∀ c, (PouwCheck.hex? (regOf c).salt).getD ByteArray.empty = enc c)
    (hdraw : ∀ c β, ∃ d : Array ℕ, creditDrawn (regOf c) (evOf c β) = .ok d ∧
      ∀ u : ℕ, u ∈ d.toList ↔ ∃ i : Fin (k sT), unitOf (tileAt σ sT (drawOf (hk sT) β i)) = u)
    (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) :
    prob (fun x => (play Unit (tileCount σ sT) (k sT) n).accepts x ∧
      rightSize Honest.scheme (circuit (tileCount σ sT)) ((blocks (tileCount σ sT)).size ∅) (fun w t =>
        (tileLayer (circuit (tileCount σ sT)) (hr1 _) σ k hk sT unitOf (windows σ sT regOf evOf n) opened
          (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) (j10rAtCoin (hr1 _) enc place tile)).Good
          (periodMode (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
            (tileRun σ k hk sT unitOf (windows σ sT regOf evOf n) opened
              (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) (j10rAtCoin (hr1 _) enc place tile)))
          x w t) <
        creditFloor Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)
          ((windows σ sT regOf evOf n).prefixLen x)
          (creditLive Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
            (windows σ sT regOf evOf n) x)
          (creditedSize Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
            (windows σ sT regOf evOf n) ((blocks (tileCount σ sT)).size ∅) x)
          ((blocks (tileCount σ sT)).size ∅) 1 1 (j10rRate (k sT) 1) δs) ≤ δs + n * (0 + 0) := by
  obtain ⟨hN, hM, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hLucks, -, -, -, -⟩ :=
    j10r_meets σ k hk sT unitOf regOf evOf enc hKd hacc hsalt hdraw n hn hδs
  exact rightValues σ k hk sT unitOf Honest.scheme (circuit (tileCount σ sT)) hn hN (hr1 _) hM hKd
    (andDecl (tileCount σ sT)) Verity.RegisteredCircuit.D (blocks (tileCount σ sT)) ∅ (windows σ sT regOf evOf n) opened
    enc place tile (period (Ω1 σ sT) (tileCount σ sT) n) (play Unit (tileCount σ sT) (k sT) n) 1 δs 0 0 hS one_pos
    le_rfl hρN hwmax hlk hG hDD hPre hMoved hReg hDraw hServ hSalt hLucks

end Witnesses

/-! ## Against a verifier that accepts everything -/

/-- A window's stages where every tile is skipped whole, its check passing by luck. -/
def skipStages (N : ℕ) : SplitStages Bool (Fin N) 1 :=
  .stage (fun _ _ => .whole) (fun _ _ => True) (fun _ _ => True) fun _ => .done

/-- The skipping period of `n` windows: every tile live, each window's stages `skipStages`. -/
def skipPeriod (Ω : Type) (N : ℕ) : (n : ℕ) → Period Bool Ω (Fin N) (circuit N).rounds n
  | 0 => .done
  | n + 1 => .window (fun _ => true) (skipStages N) fun _ => skipPeriod Ω N n

theorem liveAt_skipPeriod {Ω : Type} {N : ℕ} :
    ∀ (n : ℕ) (x : Paths Ω n) (w : Fin n), (skipPeriod Ω N n).liveAt x w = fun _ => true
  | 0, _, w => w.elim0
  | n + 1, x, w => by
    refine Fin.cases ?_ (fun w' => ?_) w
    · rfl
    · exact liveAt_skipPeriod n x.2 w'

theorem stages_skipPeriod {Ω : Type} {N : ℕ} :
    ∀ (n : ℕ) (x : Paths Ω n) (w : Fin n), (skipPeriod Ω N n).windowAt x w = skipStages N
  | 0, _, w => w.elim0
  | n + 1, x, w => by
    refine Fin.cases ?_ (fun w' => ?_) w
    · rfl
    · exact stages_skipPeriod n x.2 w'

/-- **No block of the skipping period is paid.** -/
theorem skip_not_paid {B V : Type} {s N n : ℕ} (pr : PeriodRun Bool B (circuit N).rounds s N n V)
    (x : Paths (Paths Bool (circuit N).rounds × B) n) (u : Fin n × Fin N) :
    periodMode (circuit N) (skipPeriod _ N n) pr x u ≠ .paid := by
  unfold periodMode
  split_ifs
  · rw [stages_skipPeriod]
    simp [skipStages, SplitStages.joint, SplitStages.modeAt, tileStage, Declared.stageOf, circuit]
  · decide

theorem paths_nonempty {Ω : Type} [h : Nonempty Ω] : ∀ n, Nonempty (Paths Ω n)
  | 0 => ⟨()⟩
  | n + 1 => ⟨(Classical.choice h, Classical.choice (paths_nonempty n))⟩

theorem zeroCone_eq (N : ℕ) : zeroCone Honest.scheme (circuit N) N = 0 :=
  Finset.sum_eq_zero fun t _ => if_neg (by rw [tileWords_eq]; exact one_ne_zero)

theorem sizePerWord_blocks {N : ℕ} (hN : 0 < N) :
    sizePerWord Honest.scheme (circuit N) N ((blocks N).size ∅) = 1 := by
  have : Nonempty (Fin N) := ⟨⟨0, hN⟩⟩
  unfold sizePerWord
  simp only [tileWords_eq, size_blocks, one_ne_zero, if_false, Nat.cast_one, div_one, ciSup_const]

/-- **No right block, no right size.** -/
theorem rightSize_none {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ} (sz : Fin Np → ℕ)
    (good : Fin n → Fin Np → Prop) (h : ∀ w t, ¬ good w t) : rightSize Sc Cd sz good = 0 :=
  Finset.sum_eq_zero fun w _ => Finset.sum_eq_zero fun t _ => if_neg fun h' => h w t h'.1

/-- **An event on every path exceeds every bound below `1`.** -/
theorem not_prob_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {p : Ω → Prop} (h : ∀ ω, p ω) {δ : ℝ} (hδ : δ < 1) :
    ¬ prob p ≤ δ := by
  intro hle
  have h1 := prob_add_not p
  rw [Honest.prob_none (p := fun ω => ¬ p ω) fun ω h' => h' (h ω), add_zero] at h1
  linarith

/-- **The credit at `σ = 1` and `ε < 1` is positive.** -/
theorem credit_pos {σ ε S : ℝ} (hS : 0 < S) (hσ : σ = 1) (hε : ε < 1) : 0 < max (1 - σ * ε * S / S) 0 * S := by
  rw [hσ, one_mul, mul_div_assoc, div_self hS.ne', mul_one, max_eq_left (by linarith)]
  exact mul_pos (by linarith) hS

section Witnesses

variable {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB)
  (unitOf : Fin nB → ℕ) (regOf : Bool → Registered)
  (evOf : Bool → Equiv.Perm (Fin (tileCount σ sT)) × Unit → Answered) (enc : Bool → ByteArray)

/-- **The skipping period's live words are `N·n`** when every window is credited. -/
theorem creditLive_skip (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β)) (n : ℕ) (x : Paths (Ω1 σ sT) n) :
    creditLive Honest.scheme (circuit (tileCount σ sT)) (skipPeriod (Ω1 σ sT) (tileCount σ sT) n)
      (windows σ sT regOf evOf n) x = tileCount σ sT * n := by
  unfold creditLive
  calc _ = ∑ _w : Fin n, (tileCount σ sT : ℝ) := Finset.sum_congr rfl fun w _ => by
          have hw : (windows σ sT regOf evOf n).AcceptedUpTo x w := fun _ _ => hacc _ _
          rw [if_pos hw, liveAt_skipPeriod n x w]
          simp only [liveWords_true]
          exact sum_tileWords _
    _ = tileCount σ sT * n := by simp [mul_comm]

/-- **The skipping period's credited size is `N·n` AND gates** when every window is credited. -/
theorem creditedSize_skip (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β)) (n : ℕ) (x : Paths (Ω1 σ sT) n) :
    creditedSize Honest.scheme (circuit (tileCount σ sT)) (skipPeriod (Ω1 σ sT) (tileCount σ sT) n)
      (windows σ sT regOf evOf n) ((blocks (tileCount σ sT)).size ∅) x =
        tileCount σ sT * n := by
  unfold creditedSize
  calc _ = ∑ _w : Fin n, (tileCount σ sT : ℝ) := Finset.sum_congr rfl fun w _ => by
          have hw : (windows σ sT regOf evOf n).AcceptedUpTo x w := fun _ _ => hacc _ _
          rw [if_pos hw, liveAt_skipPeriod n x w]
          simp [liveSize, liveWords_true, tileWords_eq, size_blocks]
    _ = tileCount σ sT * n := by simp [mul_comm]

/-- **`RightValues`'s conclusion fails against a verifier that accepts everything.** Every window is credited
(`hacc`), every block is live and the play accepts every path, but the prover skips every block whole
(`skipPeriod`), so no block is right. The right values must cover `(1 − ε(n))₊·N_p·n` AND gates, positive once
`Λ·ln(1/δ_s) < n` at `Λ = j10rRate (k s_T) 1`. So the conclusion's event holds on every path: its probability is `1`,
above `δ_s + n·(0 + 0)` for `δ_s < 1`. -/
theorem right_values_nontrivial (hKd : 1 ≤ k sT) (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β))
    (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs1 : δs < 1) (hΛ : j10rRate (k sT) 1 * Real.log (1 / δs) < n) :
    ¬ prob (fun x => (play Unit (tileCount σ sT) (k sT) n).accepts x ∧
      rightSize Honest.scheme (circuit (tileCount σ sT))
        ((blocks (tileCount σ sT)).size ∅) (fun w t =>
        (tileLayer (circuit (tileCount σ sT)) (hr1 _) σ k hk sT unitOf (windows σ sT regOf evOf n) opened
          (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) (j10rAtCoin (hr1 _) enc place tile)).Good
          (periodMode (circuit (tileCount σ sT)) (skipPeriod (Ω1 σ sT) (tileCount σ sT) n)
            (tileRun σ k hk sT unitOf (windows σ sT regOf evOf n) opened
              (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) (j10rAtCoin (hr1 _) enc place tile)))
          x w t) <
        creditFloor Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)
          ((windows σ sT regOf evOf n).prefixLen x)
          (creditLive Honest.scheme (circuit (tileCount σ sT)) (skipPeriod (Ω1 σ sT) (tileCount σ sT) n)
            (windows σ sT regOf evOf n) x)
          (creditedSize Honest.scheme (circuit (tileCount σ sT)) (skipPeriod (Ω1 σ sT) (tileCount σ sT) n)
            (windows σ sT regOf evOf n) ((blocks (tileCount σ sT)).size ∅) x)
          ((blocks (tileCount σ sT)).size ∅) 1 1 (j10rRate (k sT) 1) δs) ≤
      δs + n * (0 + 0) := by
  have hN : 0 < tileCount σ sT := Nat.lt_of_lt_of_le hKd (hk sT)
  have hNn : (0 : ℝ) < tileCount σ sT * n := mul_pos (Nat.cast_pos.mpr hN) (Nat.cast_pos.mpr hn)
  have hN' : (tileCount σ sT : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  have hε : epsPrefix Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) n (tileCount σ sT * n) 1 1
      (j10rRate (k sT) 1) δs < 1 := by
    unfold epsPrefix rOfPrefix
    rw [zeroCone_eq, wordsOf_eq, min_self, mul_zero, mul_zero, add_zero, one_mul,
      mul_comm (tileCount σ sT : ℝ) (n : ℝ), mul_div_mul_right _ _ hN', div_lt_one (Nat.cast_pos.mpr hn)]
    exact hΛ
  have : Nonempty (Ω1 σ sT) := ⟨(Classical.choice (paths_nonempty _), (1, ()))⟩
  have : Nonempty (Paths (Ω1 σ sT) n) := paths_nonempty n
  refine not_prob_le (fun x => ⟨trivial, ?_⟩) (by rw [add_zero, mul_zero, add_zero]; exact hδs1)
  rw [rightSize_none]
  · unfold creditFloor epsSkip
    rw [prefixLen_all σ sT regOf evOf hacc n x, creditLive_skip σ sT regOf evOf hacc n x,
      creditedSize_skip σ sT regOf evOf hacc n x]
    exact credit_pos hNn (sizePerWord_blocks hN) hε
  · intro w t h
    exact skip_not_paid _ x (w, t) (h (w, t) Relation.ReflTransGen.refl)

end Witnesses

end Properties.Pouw.ServedValues.SecurityProofs.HonestValues
