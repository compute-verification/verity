module

public import Definitions.Core.Guarantees.ValueSalts
public import Proofs.Core.Game.Basic
public import Proofs.Core.Challenger

/-!
# What a registration leaks: proof

`RegisteredLeakage`. Under `ZkServerOwnUser` the stack's choice at every key is its choice at `k₀`. The loop admits the
record it hands and the record meets `A` only if one of its `m` sessions' genuine records meets `A` (the stack registered
that session, or handed a copy of its record) or the record is admitted and none of theirs, the forge event at `k₀`
(`forgeAdvantage`). Each session's genuine record is a uniform session's (`prCoin_coord`), so the union bound over the
sessions gives `m` times `genuine`, which is at most the ideal probability plus the PRF gap.

Its witnesses: `RegisteredLeakage_honest`, a stack that registers its one session and reads no key, where the premise and
the bound hold; and `RegisteredLeakage_nontrivial`, where the bound fails at a stack that makes its record tag from the
key (`RegisteredLeakage_nontrivial_stack`: it reads the key).

`RegisteredStreamLeakage`, the same over a stream. The loop admits the stream the stack hands it and the stream meets `A`
only if some choice `d` of `streamChoices` has genuine records that meet `A`, or the stack hands at some registration an
admitted record none of that epoch's sessions has, the forge event at `k₀`. Each registration's record is the genuine
record of a session of its epoch (one the stack registered, or a copy of its record: a record admitted at registration
`i` carries the number `i`) or none, where the loop passed it over, so the records are a choice's, which passes over at
most `b`. The union bound over the choices gives the sum of their genuine probabilities, each at most the ideal one plus
the PRF gap at that choice. Its witnesses are the single registration's, on a stream of one with none passed over.
-/

@[expose] public section

namespace Verity

open scoped ENNReal

namespace ValueSalts

theorem avg_prod {C D : Type} [Fintype C] [Fintype D] (f : C × D → ℝ≥0∞) :
    Game.avg f = Game.avg fun c => Game.avg fun d => f (c, d) := by
  unfold Game.avg
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul,
    ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun d _ => ?_
  ring

/-- One coordinate of a uniform tuple is uniform. -/
theorem avg_coord {S : Type} [Fintype S] {m : ℕ} (i : Fin m) (g : S → ℝ≥0∞) :
    Game.avg (fun s : Fin m → S => g (s i)) = Game.avg g := by
  rcases isEmpty_or_nonempty S with hS | hS
  · have : IsEmpty (Fin m → S) := ⟨fun s => hS.false (s i)⟩
    simp [Game.avg]
  · rw [← Game.avg_update i (fun s : Fin m → S => g (s i))]
    simp only [Function.update_self]
    exact Game.avg_const _

/-- An event of the key and one coordinate of `m` uniform sessions has that of the key and one uniform session. -/
theorem prCoin_coord {K S : Type} [Fintype K] [Fintype S] {m : ℕ} (i : Fin m) (P : K → S → Prop) :
    Game.prCoin (fun p : K × (Fin m → S) => P p.1 (p.2 i)) = Game.prCoin fun q : K × S => P q.1 q.2 := by
  classical
  unfold Game.prCoin
  rw [avg_prod (C := K) (D := Fin m → S), avg_prod (C := K) (D := S)]
  congr 1
  funext k
  exact avg_coord i fun x => if P k x then 1 else 0

theorem prCoin_or_le {C : Type} [Fintype C] (p q : C → Prop) :
    Game.prCoin (fun c => p c ∨ q c) ≤ Game.prCoin p + Game.prCoin q := by
  classical
  unfold Game.prCoin
  rw [← Game.avg_add]
  refine Game.avg_mono fun c => ?_
  by_cases hp : p c <;> by_cases hq : q c <;> simp [hp, hq]

theorem prCoin_of_forall {C : Type} [Fintype C] [Nonempty C] {p : C → Prop} (h : ∀ c, p c) : Game.prCoin p = 1 := by
  unfold Game.prCoin
  simp only [h, ite_true]
  exact Game.avg_const _

theorem prCoin_of_forall_not {C : Type} [Fintype C] {p : C → Prop} (h : ∀ c, ¬ p c) : Game.prCoin p = 0 :=
  le_antisymm ((Game.prCoin_mono fun c hc => (h c hc).elim).trans_eq Game.prCoin_false) zero_le

end ValueSalts

namespace SecurityProofs

open Verity.ValueSalts Verity.Assumptions

theorem RegisteredLeakage : Verity.Guarantees.RegisteredLeakage := by
  intro K S Y V O T R _ _ _ _ _ sv n m v st A k₀ hown
  have hsub : ∀ p : K × (Fin m → S),
      (sv.admits p.1 (sv.handed p.1 n v p.2 (st p.1 v (sv.views p.1 v p.2))) ∧
        A (sv.handed p.1 n v p.2 (st p.1 v (sv.views p.1 v p.2)))) →
      (∃ i ∈ (Finset.univ : Finset (Fin m)), A (sv.record p.1 n m v (p.2 i))) ∨
        ∃ ρ, st k₀ v (sv.views p.1 v p.2) = .inr ρ ∧ sv.admits p.1 ρ ∧ ∀ i, ρ ≠ sv.record p.1 n m v (p.2 i) := by
    intro p ⟨had, hA⟩
    rw [hown p.1 k₀] at had hA
    cases hc : st k₀ v (sv.views p.1 v p.2) with
    | inl i =>
      rw [hc] at hA
      exact Or.inl ⟨i, Finset.mem_univ _, hA⟩
    | inr ρ =>
      rw [hc] at had hA
      by_cases hgen : ∃ i, ρ = sv.record p.1 n m v (p.2 i)
      · obtain ⟨i, rfl⟩ := hgen
        exact Or.inl ⟨i, Finset.mem_univ _, hA⟩
      · push Not at hgen
        exact Or.inr ⟨ρ, rfl, had, hgen⟩
  have hgen : ∀ i : Fin m,
      Game.prCoin (fun p : K × (Fin m → S) => A (sv.record p.1 n m v (p.2 i))) = sv.genuine n m v A :=
    fun i => prCoin_coord i fun k x => A (sv.record k n m v x)
  calc sv.real n m v st A
      ≤ Game.prCoin fun p : K × (Fin m → S) =>
          (∃ i ∈ (Finset.univ : Finset (Fin m)), A (sv.record p.1 n m v (p.2 i))) ∨
            ∃ ρ, st k₀ v (sv.views p.1 v p.2) = .inr ρ ∧ sv.admits p.1 ρ ∧ ∀ i, ρ ≠ sv.record p.1 n m v (p.2 i) := by
        unfold Service.real
        exact Game.prCoin_mono hsub
    _ ≤ Game.prCoin (fun p : K × (Fin m → S) =>
          ∃ i ∈ (Finset.univ : Finset (Fin m)), A (sv.record p.1 n m v (p.2 i))) +
          sv.forgeAdvantage n m v (st k₀ v) := by
        unfold Service.forgeAdvantage
        exact prCoin_or_le _ _
    _ ≤ (∑ i : Fin m, Game.prCoin fun p : K × (Fin m → S) => A (sv.record p.1 n m v (p.2 i))) +
          sv.forgeAdvantage n m v (st k₀ v) := by
        gcongr
        exact Game.prCoin_exists_le Finset.univ fun i (p : K × (Fin m → S)) => A (sv.record p.1 n m v (p.2 i))
    _ = m * sv.genuine n m v A + sv.forgeAdvantage n m v (st k₀ v) := by
        rw [Finset.sum_congr rfl fun i _ => hgen i, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]
    _ ≤ m * (sv.ideal n m v A + sv.prfAdvantage n m v A) + sv.forgeAdvantage n m v (st k₀ v) := by
        gcongr
        exact le_add_tsub

theorem RegisteredStreamLeakage : Verity.Guarantees.RegisteredStreamLeakage := by
  classical
  intro K S Y V O T R _ _ _ _ _ sv r m b v st A k₀ hown
  have hsub : ∀ p : K × ((i : Fin r) → Fin (m i) → S),
      (sv.streamAdmits p.1 b (sv.streamHanded p.1 v p.2 (st p.1 v (sv.streamViews p.1 v p.2))) ∧
        A (sv.streamHanded p.1 v p.2 (st p.1 v (sv.streamViews p.1 v p.2)))) →
      (∃ d ∈ streamChoices r m b, A fun i => (d i).map fun j => sv.record p.1 i (m i) (v i) (p.2 i j)) ∨
        ∃ i ρ, st k₀ v (sv.streamViews p.1 v p.2) i = some (.inr ρ) ∧ ρ.1 = i ∧ sv.admits p.1 ρ ∧
          ∀ j, ρ ≠ sv.record p.1 i (m i) (v i) (p.2 i j) := by
    intro p ⟨⟨hcard, hadm⟩, hA⟩
    rw [hown p.1 k₀] at hcard hadm hA
    by_cases hf : ∃ i ρ, st k₀ v (sv.streamViews p.1 v p.2) i = some (.inr ρ) ∧ ρ.1 = i ∧ sv.admits p.1 ρ ∧
        ∀ j, ρ ≠ sv.record p.1 i (m i) (v i) (p.2 i j)
    · exact Or.inr hf
    · push Not at hf
      have hd : ∀ i, ∃ e : Option (Fin (m i)),
          sv.streamHanded p.1 v p.2 (st k₀ v (sv.streamViews p.1 v p.2)) i =
            e.map fun j => sv.record p.1 i (m i) (v i) (p.2 i j) := by
        intro i
        rcases hc : st k₀ v (sv.streamViews p.1 v p.2) i with _ | (j | ρ)
        · exact ⟨none, by simp [Service.streamHanded, hc]⟩
        · exact ⟨some j, by simp [Service.streamHanded, hc, Service.handed]⟩
        · have hρ : sv.streamHanded p.1 v p.2 (st k₀ v (sv.streamViews p.1 v p.2)) i = some ρ := by
            simp [Service.streamHanded, hc, Service.handed]
          obtain ⟨h1, h2⟩ := hadm i ρ hρ
          obtain ⟨j, rfl⟩ := hf i ρ hc h1 h2
          exact ⟨some j, hρ⟩
      choose d hd using hd
      refine Or.inl ⟨d, ?_, (funext hd : _ = fun i => (d i).map fun j => sv.record p.1 i (m i) (v i) (p.2 i j)) ▸ hA⟩
      have hP : passed d = Finset.univ.filter fun i =>
          sv.streamHanded p.1 v p.2 (st k₀ v (sv.streamViews p.1 v p.2)) i = none := by
        ext i
        simp [passed, hd]
      simp only [streamChoices, Finset.mem_filter, Finset.mem_univ, true_and]
      rw [hP]
      exact hcard
  calc sv.streamReal b v st A
      ≤ Game.prCoin fun p : K × ((i : Fin r) → Fin (m i) → S) =>
          (∃ d ∈ streamChoices r m b, A fun i => (d i).map fun j => sv.record p.1 i (m i) (v i) (p.2 i j)) ∨
            ∃ i ρ, st k₀ v (sv.streamViews p.1 v p.2) i = some (.inr ρ) ∧ ρ.1 = i ∧ sv.admits p.1 ρ ∧
              ∀ j, ρ ≠ sv.record p.1 i (m i) (v i) (p.2 i j) := by
        unfold Service.streamReal
        exact Game.prCoin_mono hsub
    _ ≤ Game.prCoin (fun p : K × ((i : Fin r) → Fin (m i) → S) =>
          ∃ d ∈ streamChoices r m b, A fun i => (d i).map fun j => sv.record p.1 i (m i) (v i) (p.2 i j)) +
          sv.streamForgeAdvantage v (st k₀ v) := by
        unfold Service.streamForgeAdvantage
        exact prCoin_or_le _ _
    _ ≤ (∑ d ∈ streamChoices r m b, sv.streamGenuine v d A) + sv.streamForgeAdvantage v (st k₀ v) := by
        gcongr
        exact Game.prCoin_exists_le (streamChoices r m b) fun d (p : K × ((i : Fin r) → Fin (m i) → S)) =>
          A fun i => (d i).map fun j => sv.record p.1 i (m i) (v i) (p.2 i j)
    _ ≤ (∑ d ∈ streamChoices r m b, (sv.streamIdeal m v (passed d) A + sv.streamPrfAdvantage v d A)) +
          sv.streamForgeAdvantage v (st k₀ v) := by
        gcongr with d hd
        exact le_add_tsub

-- The witnesses' value service: two keys, one session, salt and tag; every public commitment is `false`, and the record
-- tag is the key.
local notation "toyS" => (Service.mk (fun _ _ => ()) (fun _ _ => false) (fun _ _ _ => ()) (fun k _ _ _ => k) :
  Service Bool Unit Unit Unit Bool Unit Bool)

/-- **The premise and the bound hold at a stack that registers its one session.** At the witnesses' value service, the
stack that commits one session and registers it as registration 0 reads no key, and the bound holds for the event that
the record's public commitment is `false`. -/
theorem RegisteredLeakage_honest :
    ZkServerOwnUser (fun _ _ _ => .inl 0 : Stack Bool Unit Unit Bool Unit Bool 1) ∧
      Service.real toyS 0 1 () (fun _ _ _ => .inl 0) (fun ρ => ρ.2.2.1.2.1 = false) ≤
        ((1 : ℕ) : ℝ≥0∞) * (Service.ideal toyS 0 1 () (fun ρ => ρ.2.2.1.2.1 = false) +
          Service.prfAdvantage toyS 0 1 () (fun ρ => ρ.2.2.1.2.1 = false)) +
          Service.forgeAdvantage toyS 0 1 ()
            ((fun _ _ _ => .inl 0 : Stack Bool Unit Unit Bool Unit Bool 1) false ()) :=
  ⟨fun _ _ => rfl, RegisteredLeakage _ 0 1 () _ _ false fun _ _ => rfl⟩

/-- **The stack of `RegisteredLeakage_nontrivial` reads the key**: it hands a record whose record tag is the key, so it
is not the same under both keys. -/
theorem RegisteredLeakage_nontrivial_stack :
    ¬ ZkServerOwnUser (fun k _ _ => .inr (0, 1, ((), true, ()), k) : Stack Bool Unit Unit Bool Unit Bool 1) := by
  intro h
  have := congrFun (congrFun (h true false) ()) fun _ => ((), false, ())
  simp at this

/-- **The bound fails at a stack that reads the key.** At the witnesses' value service, the stack that hands a record of
its own as registration 0, public commitment `true` and record tag the key, is always admitted with a `true` commitment,
while no genuine or ideal record has one and, run at the key `false`, it forges only when the key is `false`:
`1 ≤ 1 · (0 + 0) + 1/2` is false. -/
theorem RegisteredLeakage_nontrivial :
    ¬ Service.real toyS 0 1 () (fun k _ _ => .inr (0, 1, ((), true, ()), k)) (fun ρ => ρ.2.2.1.2.1 = true) ≤
        ((1 : ℕ) : ℝ≥0∞) * (Service.ideal toyS 0 1 () (fun ρ => ρ.2.2.1.2.1 = true) +
          Service.prfAdvantage toyS 0 1 () (fun ρ => ρ.2.2.1.2.1 = true)) +
          Service.forgeAdvantage toyS 0 1 ()
            ((fun k _ _ => .inr (0, 1, ((), true, ()), k) : Stack Bool Unit Unit Bool Unit Bool 1) false ()) := by
  have hreal : Service.real toyS 0 1 () (fun k _ _ => .inr (0, 1, ((), true, ()), k)) (fun ρ => ρ.2.2.1.2.1 = true) = 1 :=
    prCoin_of_forall fun _ => ⟨rfl, rfl⟩
  have hideal : Service.ideal toyS 0 1 () (fun ρ => ρ.2.2.1.2.1 = true) = 0 :=
    prCoin_of_forall_not fun _ h => Bool.false_ne_true h
  have hgen : Service.genuine toyS 0 1 () (fun ρ => ρ.2.2.1.2.1 = true) = 0 :=
    prCoin_of_forall_not fun _ h => Bool.false_ne_true h
  have hforge : Service.forgeAdvantage toyS 0 1 ()
      ((fun k _ _ => .inr (0, 1, ((), true, ()), k) : Stack Bool Unit Unit Bool Unit Bool 1) false ()) = 2⁻¹ := by
    have : Service.forgeAdvantage toyS 0 1 ()
        ((fun k _ _ => .inr (0, 1, ((), true, ()), k) : Stack Bool Unit Unit Bool Unit Bool 1) false ()) =
        Game.prCoin fun p : Bool × (Fin 1 → Unit) => p.1 = false := by
      unfold Service.forgeAdvantage
      congr 1
      funext p
      apply propext
      constructor
      · rintro ⟨ρ, hρ, had, -⟩
        cases hρ
        exact had.symm
      · intro hk
        refine ⟨_, rfl, hk.symm, fun i h => ?_⟩
        exact Bool.false_ne_true (congrArg (fun ρ : RegistrationRecord Unit Bool Unit Bool => ρ.2.2.1.2.1) h).symm
    rw [this]
    unfold Game.prCoin
    rw [avg_prod]
    simp [Game.avg]
  rw [hreal, Service.prfAdvantage, hideal, hgen, hforge]
  simp

-- The stream witnesses' stacks over one registration of an epoch of one session: the honest one registers it, and the
-- other hands a record of its own, public commitment `true` and record tag the key.
local notation "toyH" => (fun _ _ _ _ => some (Sum.inl 0) : StreamStack Bool Unit Unit Bool Unit Bool 1 fun _ => 1)
local notation "toyF" =>
  (fun k _ _ _ => some (Sum.inr (0, 1, ((), true, ()), k)) : StreamStack Bool Unit Unit Bool Unit Bool 1 fun _ => 1)

/-- **The premise and the bound hold at a stack that registers its one session on a stream of one.** At the witnesses'
value service, the stack that commits one session and registers it reads no key, and the bound holds, with no
registration passed over, for the event that the record's public commitment is `false`. -/
theorem RegisteredStreamLeakage_honest :
    ZkServerOwnUser toyH ∧
      Service.streamReal toyS 0 (fun _ => ()) toyH (fun ρs => ∀ ρ, ρs 0 = some ρ → ρ.2.2.1.2.1 = false) ≤
        (∑ d ∈ streamChoices 1 (fun _ => 1) 0,
          (Service.streamIdeal toyS (fun _ => 1) (fun _ => ()) (passed d)
              (fun ρs => ∀ ρ, ρs 0 = some ρ → ρ.2.2.1.2.1 = false) +
            Service.streamPrfAdvantage toyS (fun _ => ()) d (fun ρs => ∀ ρ, ρs 0 = some ρ → ρ.2.2.1.2.1 = false))) +
          Service.streamForgeAdvantage toyS (fun _ => ()) (toyH false fun _ => ()) :=
  ⟨fun _ _ => rfl, RegisteredStreamLeakage _ 1 _ 0 _ _ _ false fun _ _ => rfl⟩

/-- **The stack of `RegisteredStreamLeakage_nontrivial` reads the key**: it hands a record whose record tag is the key,
so it is not the same under both keys. -/
theorem RegisteredStreamLeakage_nontrivial_stack : ¬ ZkServerOwnUser toyF := by
  intro h
  have := congrFun (congrFun (congrFun (h true false) fun _ => ()) fun _ _ => ((), false, ())) 0
  simp at this

/-- **The stream's bound fails at a stack that reads the key.** At the witnesses' value service, on a stream of one
registration none of which may be passed over, the stack that hands a record of its own, public commitment `true` and
record tag the key, is always admitted with a `true` commitment, while no genuine or ideal stream has one and, run at the
key `false`, it forges only when the key is `false`: `1 ≤ 0 + 1/2` is false. -/
theorem RegisteredStreamLeakage_nontrivial :
    ¬ Service.streamReal toyS 0 (fun _ => ()) toyF (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true) ≤
        (∑ d ∈ streamChoices 1 (fun _ => 1) 0,
          (Service.streamIdeal toyS (fun _ => 1) (fun _ => ()) (passed d)
              (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true) +
            Service.streamPrfAdvantage toyS (fun _ => ()) d (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true))) +
          Service.streamForgeAdvantage toyS (fun _ => ()) (toyF false fun _ => ()) := by
  have hreal : Service.streamReal toyS 0 (fun _ => ()) toyF (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true) = 1 := by
    refine prCoin_of_forall fun p => ⟨⟨?_, ?_⟩, ⟨_, rfl, rfl⟩⟩
    · simp [Service.streamHanded]
    · intro i ρ h
      obtain rfl : i = 0 := Subsingleton.elim _ _
      cases h
      exact ⟨rfl, rfl⟩
  have hsum : (∑ d ∈ streamChoices 1 (fun _ => 1) 0,
      (Service.streamIdeal toyS (fun _ => 1) (fun _ => ()) (passed d)
          (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true) +
        Service.streamPrfAdvantage toyS (fun _ => ()) d (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true))) = 0 := by
    refine Finset.sum_eq_zero fun d _ => ?_
    have hideal : Service.streamIdeal toyS (fun _ => 1) (fun _ => ()) (passed d)
        (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true) = 0 := by
      refine prCoin_of_forall_not fun p h => ?_
      obtain ⟨ρ, h1, h2⟩ := h
      dsimp only at h1
      split_ifs at h1
      cases h1
      exact Bool.false_ne_true h2
    have hgen : Service.streamGenuine toyS (fun _ => ()) d (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true) = 0 := by
      refine prCoin_of_forall_not fun p h => ?_
      obtain ⟨ρ, h1, h2⟩ := h
      dsimp only at h1
      cases hd : d 0 with
      | none => rw [hd] at h1; cases h1
      | some j => rw [hd] at h1; cases h1; exact Bool.false_ne_true h2
    rw [Service.streamPrfAdvantage, hideal, hgen]
    simp
  have hforge : Service.streamForgeAdvantage toyS (fun _ => ()) (toyF false fun _ => ()) = 2⁻¹ := by
    have : Service.streamForgeAdvantage toyS (fun _ => ()) (toyF false fun _ => ()) =
        Game.prCoin fun p : Bool × ((i : Fin 1) → Fin ((fun _ => 1) i) → Unit) => p.1 = false := by
      unfold Service.streamForgeAdvantage
      congr 1
      funext p
      apply propext
      constructor
      · rintro ⟨i, ρ, hρ, -, had, -⟩
        cases hρ
        exact had.symm
      · intro hk
        refine ⟨0, _, rfl, rfl, hk.symm, fun j h => ?_⟩
        exact Bool.false_ne_true (congrArg (fun ρ : RegistrationRecord Unit Bool Unit Bool => ρ.2.2.1.2.1) h).symm
    rw [this]
    unfold Game.prCoin
    rw [avg_prod]
    simp [Game.avg]
  rw [hreal, hsum, hforge]
  simp

end SecurityProofs

end Verity
