module

public import Proofs.Flock.Claims.Sampling.KS
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Cost

/-!
# The deployed sampled verifier at a checked unit's session

The compiled sampled verifier's sessions (`SamplingGeneric.Compiled.Sessions`) hold at a frozen record `D`
(`SamplingGeneric.sessions_ofDeploy`) given three facts about the deployed sampled verifier at a checked unit's session,
each a hypothesis to discharge there:

* `DrawFixed`: a checked unit's session is in the second stage, and the checked units don't read its coins.
* `SessionRuns`: at a run that checks `u`, `u`'s session is a Flock run, on its own rounds (an order embedding of
  Flock's), of a Flock prover fixed by the run's coins off them; the session decides the run; and the sampled verifier
  makes the Flock verifier's `H` calls. The session's view needs only the coins before its first round.
* `InnerNumbers`: the inner numbers (`innerWaits`, `innerCap`, `innerN₀`) are Flock's at every session's statement, and
  its `ε_table` and weight at most the inner ones. `InnerNumbersAt D n₀` is the same with each unit's own `N₀`,
  `n₀ x u`, for a two-stage record whose units' sessions set up tables of different shapes; at `n₀ x _ = innerN₀ x` it
  is `InnerNumbers` (`innerNumbersAt_innerN₀`).

The coin lemmas the compiled sessions' proof reads are here too: a set of rounds that resampling keeps averages the
second stage's coins over the session's (`Ex_resample`), coins restricted along an order embedding are uniform
(`Ex_comp_inj`, `Pr_restrict_pair`), and `B`'s collision probability is a logged collision's (`B_Pr_logColl`).

A session may cover several checked units, as the one-stage audit's one session proves every sampled proof block:
`SessionRunsSlots` asks `SessionRuns` only of the units at the slots whose sessions cover the rest (`SlotsCover`,
`Sampling.KS`), and gives it (`sessionRuns_of_slots`); one session per unit is `SessionRuns` itself
(`sessionRunsSlots_units_iff`). `InnerNumbers` at one session over every unit is one statement about it
(`innerNumbers_one_iff`).
-/

@[expose] public section

namespace Claims.Sampling

open Claims Finset

/-! ## Coins restricted along an injective map, and resampled on a set they determine -/

section Coins

variable {Coin : Type} [Fintype Coin] [Nonempty Coin]

/-- **Uniform coins restricted along an injective map are uniform.** -/
theorem Ex_comp_inj {ι ι' : Type} [Fintype ι] [Fintype ι'] [DecidableEq ι] [DecidableEq ι'] (e : ι' → ι)
    (he : Function.Injective e) (h : (ι' → Coin) → ℝ) :
    Ex (fun c : ι → Coin => h fun i => c (e i)) = Ex h := by
  classical
  let eq : (ι → Coin) ≃ (ι' → Coin) × ({t // t ∉ Set.range e} → Coin) :=
    (Equiv.piEquivPiSubtypeProd (fun t => t ∈ Set.range e) fun _ => Coin).trans
      (Equiv.prodCongr (Equiv.arrowCongr (Equiv.ofInjective e he).symm (Equiv.refl Coin)) (Equiv.refl _))
  calc Ex (fun c : ι → Coin => h fun i => c (e i)) = Ex (fun c => h (eq c).1) := rfl
    _ = Ex (fun s : (ι' → Coin) × ({t // t ∉ Set.range e} → Coin) => h s.1) := Ex_equiv eq (fun s => h s.1)
    _ = Ex h := Ex_fst h

theorem Pr_comp_inj {ι ι' : Type} [Fintype ι] [Fintype ι'] [DecidableEq ι] [DecidableEq ι'] (e : ι' → ι)
    (he : Function.Injective e) (q : (ι' → Coin) → Prop) :
    Pr (fun c : ι → Coin => q fun i => c (e i)) = Pr q := by
  classical
  rw [← Ex_ind (fun c : ι → Coin => q fun i => c (e i)), ← Ex_ind q]
  exact Ex_comp_inj e he fun c => if q c then 1 else 0

omit [Fintype Coin] [Nonempty Coin] in
theorem forkOn_empty {k : ℕ} (r d : Fin k → Coin) : forkOn ∅ r d = r :=
  funext fun t => by simp [forkOn]

/-- **Resampling on a set the coins determine**: when resampling `d` on `S d` keeps `S d`, averaging over `d` and then
over fresh coins on `S d` is averaging over `d`: swapping `d` and the fresh coins on `S d` is an involution. -/
theorem Ex_resample {k : ℕ} (S : (Fin k → Coin) → Finset (Fin k)) (hS : ∀ d σ, S (forkOn (S d) d σ) = S d)
    (F : (Fin k → Coin) → ℝ) :
    Ex (fun d => Ex fun σ => F (forkOn (S d) d σ)) = Ex F := by
  have inv : ∀ p : (Fin k → Coin) × (Fin k → Coin),
      (forkOn (S (forkOn (S p.1) p.1 p.2)) (forkOn (S p.1) p.1 p.2) (forkOn (S p.1) p.2 p.1),
        forkOn (S (forkOn (S p.1) p.1 p.2)) (forkOn (S p.1) p.2 p.1) (forkOn (S p.1) p.1 p.2)) = p := by
    intro p
    rw [hS]
    refine Prod.ext (funext fun t => ?_) (funext fun t => ?_) <;> by_cases h : t ∈ S p.1 <;> simp [forkOn, h]
  let sw : (Fin k → Coin) × (Fin k → Coin) ≃ (Fin k → Coin) × (Fin k → Coin) :=
    { toFun := fun p => (forkOn (S p.1) p.1 p.2, forkOn (S p.1) p.2 p.1)
      invFun := fun p => (forkOn (S p.1) p.1 p.2, forkOn (S p.1) p.2 p.1)
      left_inv := inv
      right_inv := inv }
  have h1 := Ex_prod (fun p : (Fin k → Coin) × (Fin k → Coin) => F (forkOn (S p.1) p.1 p.2))
  have h2 : Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F (forkOn (S p.1) p.1 p.2)) =
      Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F p.1) := Ex_equiv sw (fun p => F p.1)
  rw [← Ex_fst (β := Fin k → Coin) F, ← h2, h1]

omit [Fintype Coin] [Nonempty Coin] in
/-- Fresh coins on `J ⊆ I`, then `r` off `I`, are fresh on `J` of the coins fresh on `I`. -/
theorem forkOn_sub {k : ℕ} {I J : Finset (Fin k)} (h : J ⊆ I) (r d σ : Fin k → Coin) :
    forkOn I r (forkOn J d σ) = forkOn J (forkOn I r d) σ :=
  funext fun t => by
    by_cases hJ : t ∈ J
    · simp [forkOn, hJ, h hJ]
    · simp [forkOn, hJ]

end Coins

/-! ## `B`'s collisions, as logged collisions -/

theorem B_Pr_logColl {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) :
    (B H V P x hk).Pr H = Pr fun s : BCoins Ω Coin (V.k x) =>
      LogColl ((run H P V x s.2.1 s.2.2.1).vLog ++ (run H P V x s.2.1 (forkCoins s.1 s.2.2.1 s.2.2.2)).vLog) := by
  rw [B_Pr_eq]
  refine Pr_congr fun s => ⟨fun h => ?_, fun h => findColl_isSome _ h⟩
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 h
  obtain ⟨hc, hu, hv⟩ := B_out_spec H V P x hk s u v huv
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hu
  obtain ⟨p', hp', rfl⟩ := List.mem_map.1 hv
  have hg : ∀ q ∈ (run H P V x s.2.1 s.2.2.1).vLog ++ (run H P V x s.2.1 (forkCoins s.1 s.2.2.1 s.2.2.2)).vLog,
      H q.1 = q.2 := fun q hq => (List.mem_append.1 hq).elim (fun hq => (run_logs H P V x _ _).2 q hq)
        fun hq => (run_logs H P V x _ _).2 q hq
  exact ⟨p, hp, p', hp', hc.1, by rw [← hg p hp, ← hg p' hp', hc.2]⟩

theorem PrB_nonneg {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) : 0 ≤ PrB H V P x := by
  unfold PrB
  split_ifs
  · exact Pr_nonneg _
  · exact le_rfl

theorem Pr_unit {X : Type} [Fintype X] (p : X → Prop) : Pr (fun s : Unit × X => p s.2) = Pr p :=
  Pr_equiv (Equiv.punitProd X) p

section Restrict

variable {Coin : Type} [Fintype Coin] [Nonempty Coin]

/-- Two independent coin vectors restricted along an injective map. -/
theorem Pr_restrict_pair {kS kF : ℕ} (emb : Fin kF → Fin kS) (hemb : Function.Injective emb)
    (Q : (Fin kF → Coin) → (Fin kF → Coin) → Prop) :
    Pr (fun s : (Fin kS → Coin) × (Fin kS → Coin) => Q (fun i => s.1 (emb i)) fun i => s.2 (emb i)) =
      Pr (fun s : (Fin kF → Coin) × (Fin kF → Coin) => Q s.1 s.2) := by
  rw [Pr_prod_eq_Ex, Pr_prod_eq_Ex]
  have h : ∀ a : Fin kF → Coin, Pr (fun R2 : Fin kS → Coin => Q a fun i => R2 (emb i)) = Pr (Q a) :=
    fun a => Pr_comp_inj emb hemb (Q a)
  simp only [h]
  exact Ex_comp_inj emb hemb fun a => Pr (Q a)

end Restrict

/-! ## What the deployed sampled verifier is taken to do at a session -/

section Session

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The sample precedes the sessions**: a unit checked at `r` has its session's rounds in the second stage, and fresh
coins on them keep the checked units. -/
def DrawFixed (D : Deploy Coin Col T) : Prop :=
  ∀ x r u, (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) →
    D.rounds x u ⊆ D.stage2 x ∧ ∀ d, D.checked x (forkOn (D.rounds x u) r d) = D.checked x r

/-- **A checked unit's session is a Flock run**: at coins `r` that check `u`, the session's statement `xS`, an order
embedding `emb` of Flock's rounds onto `u`'s, and a Flock prover `P'` (the sampled prover with `r`'s coins off `u`'s
rounds), such that at coins `r'` agreeing with `r` off `u`'s rounds the sampled run accepts exactly when `P'`'s run on
`r'`'s session coins does (the session decides the run), and at coins agreeing with `r` on the rounds outside `u`'s
before its first, `u`'s session is `P'`'s run and the sampled verifier makes its `H` calls. -/
def SessionRuns (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) (u : ℕ),
    (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) →
    ∃ (xS : Msg) (emb : Fin (D.flock.V.k xS) → Fin (D.V.k x)) (P' : Prover Unit),
      StrictMono emb ∧ (∀ t, t ∈ D.rounds x u ↔ ∃ i, emb i = t) ∧
      (∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, r' t = r t) →
        ((run H P D.V x ω r').out = .acc ↔ (run H P' D.flock.V xS () fun i => r' (emb i)).out = .acc)) ∧
      ∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, (∀ i, t < emb i) → r' t = r t) →
        D.session x u (run H P D.V x ω r').seen = some (xS, (run H P' D.flock.V xS () fun i => r' (emb i)).seen) ∧
          ∀ q ∈ (run H P' D.flock.V xS () fun i => r' (emb i)).vLog, q ∈ (run H P D.V x ω r').vLog

/-- **The inner numbers are Flock's** at every session's statement: its `E^rep`'s waits, truncation and `N₀`, and its
`ε_table` and weight at most the inner ones. -/
def InnerNumbers (D : Deploy Coin Col T) : Prop :=
  ∀ x u v xS vS, D.session x u v = some (xS, vS) →
    D.flock.waits xS = D.innerWaits x ∧ D.flock.cap xS = D.innerCap x ∧ D.flock.N₀ xS = D.innerN₀ x ∧
      D.flock.εtable xS ≤ D.εtable x ∧ D.flock.weight xS ≤ D.weight x

/-- **The inner numbers are Flock's at each unit's `N₀`**: `InnerNumbers` with a session of unit `u` at `x` at `N₀ =
n₀ x u`. -/
def InnerNumbersAt (D : Deploy Coin Col T) (n₀ : Msg → ℕ → ℝ) : Prop :=
  ∀ x u v xS vS, D.session x u v = some (xS, vS) →
    D.flock.waits xS = D.innerWaits x ∧ D.flock.cap xS = D.innerCap x ∧ D.flock.N₀ xS = n₀ x u ∧
      D.flock.εtable xS ≤ D.εtable x ∧ D.flock.weight xS ≤ D.weight x

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem innerNumbersAt_innerN₀ (D : Deploy Coin Col T) : InnerNumbersAt D (fun x _ => D.innerN₀ x) ↔ InnerNumbers D :=
  Iff.rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **`DrawFixed` from the sample's place in the run**: where each checked unit's session is in the second stage and the
checked units read only the coins off the second stage, fresh coins on a session keep them. -/
theorem drawFixed_of_first (D : Deploy Coin Col T)
    (hrounds : ∀ x r u, (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) → D.rounds x u ⊆ D.stage2 x)
    (hfirst : ∀ x r r', (∀ t ∉ D.stage2 x, r t = r' t) → D.checked x r = D.checked x r') : DrawFixed D := by
  intro x r u hu
  refine ⟨hrounds x r u hu, fun d => hfirst x _ _ fun t ht => ?_⟩
  have hn : t ∉ D.rounds x u := fun hm => ht (hrounds x r u hu hm)
  simp [forkOn, hn]

/-- **`SessionRuns` a session at a time**: at coins `r`, the unit checked at a slot below `S x` has its session a Flock
run on its rounds, as `SessionRuns` says of every checked unit. Where the slots cover the checked units
(`SlotsCover`), that is `SessionRuns` (`sessionRuns_of_slots`): one session over every checked unit, as the one-stage
audit runs, is one obligation per run (`S = 1`), and one session per unit is `SessionRuns` itself (`S = nPU D`,
`sessionRunsSlots_units_iff`). -/
def SessionRunsSlots (H : Msg → Msg) (D : Deploy Coin Col T) (S : Msg → ℕ) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) (p : ℕ) (u : ℕ),
    p < S x → (D.checked x r)[p]? = some u →
    ∃ (xS : Msg) (emb : Fin (D.flock.V.k xS) → Fin (D.V.k x)) (P' : Prover Unit),
      StrictMono emb ∧ (∀ t, t ∈ D.rounds x u ↔ ∃ i, emb i = t) ∧
      (∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, r' t = r t) →
        ((run H P D.V x ω r').out = .acc ↔ (run H P' D.flock.V xS () fun i => r' (emb i)).out = .acc)) ∧
      ∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, (∀ i, t < emb i) → r' t = r t) →
        D.session x u (run H P D.V x ω r').seen = some (xS, (run H P' D.flock.V xS () fun i => r' (emb i)).seen) ∧
          ∀ q ∈ (run H P' D.flock.V xS () fun i => r' (emb i)).vLog, q ∈ (run H P D.V x ω r').vLog

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **`SessionRuns` from its slot form**: each checked unit has a slot's rounds and session. -/
theorem sessionRuns_of_slots {H : Msg → Msg} {D : Deploy Coin Col T} {S : Msg → ℕ} (hc : SlotsCover D S)
    (h : SessionRunsSlots H D S) : SessionRuns H D := by
  intro x Ω P ω r u hu
  obtain ⟨p, hp, u', hu', hr, hs⟩ := hc x r u hu
  rw [hr, hs]
  exact h x Ω P ω r p u' hp hu'

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **At one session per unit the slot form is `SessionRuns`.** -/
theorem sessionRunsSlots_units_iff (H : Msg → Msg) (D : Deploy Coin Col T) :
    SessionRunsSlots H D (nPU D) ↔ SessionRuns H D :=
  ⟨sessionRuns_of_slots (slotsCover_units D), fun h x Ω P ω r p u hp hu => h x Ω P ω r u ⟨⟨p, hp⟩, hu⟩⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **`InnerNumbers` at one session over every unit**: one statement about the run's one session. `InnerNumbers` reads
every session a unit has, whichever units share it, so several sessions at `x` must have Flock's numbers alike: the
same tables' shapes give the same `N₀`. -/
theorem innerNumbers_one_iff (D : Deploy Coin Col T) (hs : ∀ x u u', D.session x u = D.session x u') :
    InnerNumbers D ↔ ∀ x v xS vS, D.session x 0 v = some (xS, vS) →
      D.flock.waits xS = D.innerWaits x ∧ D.flock.cap xS = D.innerCap x ∧ D.flock.N₀ xS = D.innerN₀ x ∧
        D.flock.εtable xS ≤ D.εtable x ∧ D.flock.weight xS ≤ D.weight x :=
  ⟨fun h x v xS vS hv => h x 0 v xS vS hv, fun h x u v xS vS hv => h x v xS vS (hs x u 0 ▸ hv)⟩

end Session

end Claims.Sampling
