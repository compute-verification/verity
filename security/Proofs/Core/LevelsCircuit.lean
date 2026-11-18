module

public import Proofs.Core.Circuit
public import Proofs.Core.LevelsLate
public import Proofs.Core.CircuitLevels

/-!
# Committing late does not help, at a circuit's levels

Two levels drawn over a circuit's nested partitions: the inner blocks are the units of `Pf`, the outer blocks those of
`Pc`, and `Pf` refines `Pc` (`r`). The levels' blocks are sets of gates of some type `G`, and `blk`, `oblk` place the
inner and outer units among them, an inner unit with a gate inside an outer one exactly when the outer is its parent.
The outer trace is `τ₁` at the opened gates `O`. An inner trace extends it when it agrees with `τ₁` on `O`, and its incorrect
inner blocks are its incorrect units with those having an outer committed wire outside `O` (`Partition.unopened`).

* `Levels.late_le_missProb_opened`: after the outer sample is drawn, the prover picks at each outer block, by any rule,
  an inner trace extending `τ₁`, its interior. Every inner sample then misses that trace's incorrect inner blocks with
  probability at most `missProb L.design` of one trace's, a trace fixed in advance that extends `τ₁`. It is
  `Levels.late_le_missProb` at the incorrect sets of the traces extending `τ₁`, which glue
  (`Partition.Refines.glue`).
* `Levels.late_le_missProb_trace`: the same with every outer committed wire opened.
* `Levels.prCoin_accept_le_late`: the sampling step of soundness at a miss `δ` with interiors committed late. A run
  accepted only where the inner samples miss the late interiors' incorrect inner blocks, or a break occurs, is
  accepted while the largest miss over the interiors is at most `δ` with probability at most `δ` plus the break's.
* `Levels.ofPartitions_late_le_missProb`: at the levels of a chain of partitions, any number of them, the prover commits
  a trace after each level's draw, by any rule, agreeing with the one before on that level's committed gates. Every
  sample misses the last trace's incorrect last-level blocks with probability at most `missProb` of one trace's, fixed
  in advance and agreeing with the first on the first level's committed gates. It is `Levels.late_le_missProb_levels`
  with traces near a block when they agree on the gates it holds or reads (`Partition.region`).
* `Levels.ofPartitions_prCoin_accept_le_levels`: the sampling step of soundness at a miss `δ` at those levels.
-/

@[expose] public section

namespace Verity

open Game
open scoped ENNReal

/-- The sampling step of soundness at a miss `δ`, at any event `E` each `p` bounds by `m p`. -/
theorem prCoin_accept_le_of {P Ω : Type} [Fintype P] [Nonempty P] [Fintype Ω] (E acc brk : P → Ω → Prop)
    (m : P → ℝ≥0∞) (hE : ∀ p, prCoin (E p) ≤ m p) (h : ∀ p ω, acc p ω → E p ω ∨ brk p ω) (δ : ℝ≥0∞) :
    prCoin (fun x : P × Ω => acc x.1 x.2 ∧ m x.1 ≤ δ) ≤ δ + prCoin (fun x : P × Ω => brk x.1 x.2) := by
  classical
  have hmiss : prCoin (fun x : P × Ω => E x.1 x.2 ∧ m x.1 ≤ δ) ≤ δ := by
    have hp : ∀ p : P, prCoin (fun ω : Ω => E p ω ∧ m p ≤ δ) ≤ δ := by
      intro p
      by_cases hδ : m p ≤ δ
      · exact (prCoin_mono fun ω hω => hω.1).trans ((hE p).trans hδ)
      · exact (prCoin_mono (q := fun _ => False) fun ω hω => hδ hω.2).trans (prCoin_false.trans_le zero_le)
    calc prCoin (fun x : P × Ω => E x.1 x.2 ∧ m x.1 ≤ δ)
        = avg fun p : P => prCoin fun ω : Ω => E p ω ∧ m p ≤ δ := by
          unfold prCoin
          exact Tape.avg_prod _
      _ ≤ avg fun _ : P => δ := avg_mono hp
      _ = δ := avg_const δ
  calc prCoin (fun x : P × Ω => acc x.1 x.2 ∧ m x.1 ≤ δ)
      ≤ prCoin (fun x : P × Ω => (E x.1 x.2 ∧ m x.1 ≤ δ) ∨ brk x.1 x.2) :=
        prCoin_mono fun x hx => (h x.1 x.2 hx.1).imp (fun hd => ⟨hd, hx.2⟩) id
    _ ≤ _ := (prCoin_or_le _ _).trans (add_le_add hmiss le_rfl)

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **Committing late does not help, at a circuit's two levels, from a partly opened outer trace.** -/
theorem late_le_missProb_opened (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂])
    {C : Circuit} {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (r : Pf.Refines Pc)
    (blk : Fin nf → Finset G) (oblk : Fin nc → Finset G)
    (hsub : ∀ v k, (Pf.gates v).Nonempty → (blk v ⊆ oblk k ↔ r.parent v = k))
    (hparts : ∀ X ∈ l₁.blocks.parts, ∃ k, oblk k = X) (τ₁ : C.Trace) (O : Finset (Fin C.N))
    (R : (l₁.design Finset.univ).Ω → Finset G → C.Trace) (hR : ∀ ω X, ∀ g ∈ O, R ω X g = τ₁ g) :
    ∃ τ₂ : C.Trace, (∀ g ∈ O, τ₂ g = τ₁ g) ∧
      prCoin (fun ω : (l₁.design Finset.univ).Ω × ((X : Finset G) → (l₂.design X).Ω) =>
          ∀ X ∈ (l₁.design Finset.univ).draw ω.1,
            Disjoint ((l₂.design X).draw (ω.2 X)) ((Pf.unopened Pc.committed O ∪ Pf.incorrect (R ω.1 X)).image blk)) ≤
        L.design.missProb ((Pf.unopened Pc.committed O ∪ Pf.incorrect τ₂).image blk) := by
  classical
  set U := Pf.unopened Pc.committed O
  let 𝒯 : Finset (Finset (Finset G)) := Finset.univ.filter fun T =>
    ∃ Y : C.Trace, (∀ g ∈ O, Y g = τ₁ g) ∧ T = (U ∪ Pf.incorrect Y).image blk
  have mem𝒯 : ∀ T, T ∈ 𝒯 ↔ ∃ Y : C.Trace, (∀ g ∈ O, Y g = τ₁ g) ∧ T = (U ∪ Pf.incorrect Y).image blk :=
    fun T => by simp [𝒯]
  have hfilter : ∀ (s : Finset (Fin nf)) (k : Fin nc), (∀ v ∈ s, (Pf.gates v).Nonempty) →
      (s.image blk).filter (· ⊆ oblk k) = (s.filter (r.parent · = k)).image blk := by
    intro s k hs
    rw [Finset.filter_image]
    exact congrArg (Finset.image blk) (Finset.filter_congr fun v hv => hsub v k (hs v hv))
  have hne : ∀ Y : C.Trace, ∀ v ∈ U ∪ Pf.incorrect Y, (Pf.gates v).Nonempty := fun Y v hv =>
    (Finset.mem_union.1 hv).elim Pf.gates_nonempty_of_unopened Pf.gates_nonempty_of_incorrect
  have hglue : ∀ c : Finset G → Finset (Finset G), (∀ X, c X ∈ 𝒯) →
      ∃ T ∈ 𝒯, ∀ X ∈ l₁.blocks.parts, T.filter (· ⊆ X) = (c X).filter (· ⊆ X) := by
    intro c hc
    choose Yc hYc hcY using fun X => (mem𝒯 (c X)).1 (hc X)
    obtain ⟨hres, hfil⟩ := r.glue (X := τ₁) (Y := fun k => Yc (oblk k)) (O := O) fun k => hYc (oblk k)
    refine ⟨(U ∪ Pf.incorrect (Pc.splice τ₁ fun k => Yc (oblk k))).image blk, (mem𝒯 _).2 ⟨_, hres, rfl⟩, ?_⟩
    intro X hX
    obtain ⟨k, rfl⟩ := hparts X hX
    rw [hcY (oblk k), hfilter _ _ (hne _), hfilter _ _ (hne _), hfil k]
  obtain ⟨T, hT, hle⟩ := L.late_le_missProb hL 𝒯 hglue (fun ω X => (U ∪ Pf.incorrect (R ω X)).image blk)
    fun ω X => (mem𝒯 _).2 ⟨R ω X, hR ω X, rfl⟩
  obtain ⟨Y, hY, rfl⟩ := (mem𝒯 T).1 hT
  exact ⟨Y, hY, hle⟩

/-- **Committing late does not help, at a circuit's two levels**, from an outer trace opened on every outer committed
wire. -/
theorem late_le_missProb_trace (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂])
    {C : Circuit} {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (r : Pf.Refines Pc)
    (blk : Fin nf → Finset G) (oblk : Fin nc → Finset G)
    (hsub : ∀ v k, (Pf.gates v).Nonempty → (blk v ⊆ oblk k ↔ r.parent v = k))
    (hparts : ∀ X ∈ l₁.blocks.parts, ∃ k, oblk k = X) (τ₁ : C.Trace)
    (R : (l₁.design Finset.univ).Ω → Finset G → C.Trace) (hR : ∀ ω X, ∀ g ∈ Pc.committed, R ω X g = τ₁ g) :
    ∃ τ₂ : C.Trace, (∀ g ∈ Pc.committed, τ₂ g = τ₁ g) ∧
      prCoin (fun ω : (l₁.design Finset.univ).Ω × ((X : Finset G) → (l₂.design X).Ω) =>
          ∀ X ∈ (l₁.design Finset.univ).draw ω.1,
            Disjoint ((l₂.design X).draw (ω.2 X)) ((Pf.incorrect (R ω.1 X)).image blk)) ≤
        L.design.missProb ((Pf.incorrect τ₂).image blk) := by
  obtain ⟨τ₂, h, hle⟩ := L.late_le_missProb_opened hL r blk oblk hsub hparts τ₁ Pc.committed R hR
  refine ⟨τ₂, h, ?_⟩
  simpa only [Pf.unopened_self, Finset.empty_union] using hle

/-- **The sampling step of soundness at a miss `δ`, interiors committed late.** The prover's coins `p`, drawn before
the samples and independently of them, fix the outer trace `τ₁ p` at the opened gates `O p`. After the outer sample the
prover picks at each outer block, by any rule `R p`, an interior extending it. Where every accepted run's inner samples
miss the late interiors' incorrect inner blocks or a break occurs, the run is accepted while the largest miss over
the interiors extending `τ₁ p` is at most `δ` with probability at most `δ` plus the break's. -/
theorem prCoin_accept_le_late (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂])
    {C : Circuit} {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (r : Pf.Refines Pc)
    (blk : Fin nf → Finset G) (oblk : Fin nc → Finset G)
    (hsub : ∀ v k, (Pf.gates v).Nonempty → (blk v ⊆ oblk k ↔ r.parent v = k))
    (hparts : ∀ X ∈ l₁.blocks.parts, ∃ k, oblk k = X) {P : Type} [Fintype P] [Nonempty P]
    (τ₁ : P → C.Trace) (O : P → Finset (Fin C.N)) (R : P → (l₁.design Finset.univ).Ω → Finset G → C.Trace)
    (hR : ∀ p ω X, ∀ g ∈ O p, R p ω X g = τ₁ p g)
    (acc brk : P → (l₁.design Finset.univ).Ω × ((X : Finset G) → (l₂.design X).Ω) → Prop)
    (h : ∀ p ω, acc p ω → (∀ X ∈ (l₁.design Finset.univ).draw ω.1,
        Disjoint ((l₂.design X).draw (ω.2 X)) ((Pf.unopened Pc.committed (O p) ∪ Pf.incorrect (R p ω.1 X)).image blk)) ∨
      brk p ω) (δ : ℝ≥0∞) :
    prCoin (fun x : P × ((l₁.design Finset.univ).Ω × ((X : Finset G) → (l₂.design X).Ω)) => acc x.1 x.2 ∧
        (⨆ (τ₂ : C.Trace) (_ : ∀ g ∈ O x.1, τ₂ g = τ₁ x.1 g),
          L.design.missProb ((Pf.unopened Pc.committed (O x.1) ∪ Pf.incorrect τ₂).image blk)) ≤ δ) ≤
      δ + prCoin (fun x : P × ((l₁.design Finset.univ).Ω × ((X : Finset G) → (l₂.design X).Ω)) => brk x.1 x.2) := by
  refine prCoin_accept_le_of _ acc brk (fun p => ⨆ (τ₂ : C.Trace) (_ : ∀ g ∈ O p, τ₂ g = τ₁ p g),
    L.design.missProb ((Pf.unopened Pc.committed (O p) ∪ Pf.incorrect τ₂).image blk)) (fun p => ?_) h δ
  obtain ⟨τ₂, hτ₂, hle⟩ := L.late_le_missProb_opened hL r blk oblk hsub hparts (τ₁ p) (O p) (R p) (hR p)
  exact hle.trans (le_iSup₂ (f := fun (τ : C.Trace) (_ : ∀ g ∈ O p, τ g = τ₁ p g) =>
    L.design.missProb ((Pf.unopened Pc.committed (O p) ∪ Pf.incorrect τ).image blk)) τ₂ hτ₂)

end Levels

namespace Partition

/-- A fine unit with a gate lies inside a coarse unit's block exactly when that unit is its parent. -/
theorem Refines.block_subset_iff {C : Circuit} {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc}
    (r : Pf.Refines Pc) (v : Fin nf) (k : Fin nc) (hv : (Pf.gates v).Nonempty) :
    Pf.block v ⊆ Pc.block k ↔ r.parent v = k := by
  refine ⟨fun h => ?_, fun h => h ▸ r.block_subset v⟩
  obtain ⟨g, hg⟩ := hv
  have hu : Pf.unit g = some v := (Finset.mem_filter.1 hg).2
  have hin : C.isInput g = false := by
    cases hi : C.isInput g
    · rfl
    · exact absurd ((Pf.input_iff g).2 hi) (by rw [hu]; exact Option.some_ne_none v)
  have hk := (Finset.mem_filter.1 (h (Finset.mem_filter.2 ⟨Finset.mem_univ ⟨g, hin⟩, hu⟩))).2
  rw [r.sub, hu] at hk
  exact Option.some_inj.1 hk

end Partition

namespace Levels

variable {C : Circuit} {nc nf : ℕ} {Pc : Partition C nc} {Pf : Partition C nf}
  (Rc : Pc.Refines (Partition.whole C)) (rule₁ : Fin 1 → SamplingDesign (Fin nc)) (r : Pf.Refines Pc)
  (rule₂ : Fin nc → SamplingDesign (Fin nf))

/-- **Committing late does not help, at a circuit's two partitions**, from a partly opened outer trace: the levels of
`Pc` refining the whole circuit and `Pf` refining `Pc`, the blocks their units. -/
theorem ofPartitions_late_le_missProb_opened (τ₁ : C.Trace) (O : Finset (Fin C.N))
    (R : (Rc.design rule₁ Finset.univ).Ω → Finset C.Computed → C.Trace) (hR : ∀ ω X, ∀ g ∈ O, R ω X g = τ₁ g) :
    ∃ τ₂ : C.Trace, (∀ g ∈ O, τ₂ g = τ₁ g) ∧
      prCoin (fun ω : (Rc.design rule₁ Finset.univ).Ω × ((X : Finset C.Computed) → (r.design rule₂ X).Ω) =>
          ∀ X ∈ (Rc.design rule₁ Finset.univ).draw ω.1,
            Disjoint ((r.design rule₂ X).draw (ω.2 X))
              ((Pf.unopened Pc.committed O ∪ Pf.incorrect (R ω.1 X)).image Pf.block)) ≤
        (Levels.ofPartitions (.cons Pc Rc rule₁ (.cons Pf r rule₂ .nil))).design.missProb
          ((Pf.unopened Pc.committed O ∪ Pf.incorrect τ₂).image Pf.block) :=
  late_le_missProb_opened (l₁ := ⟨Pc.blocks, Rc.design rule₁⟩) (l₂ := ⟨Pf.blocks, r.design rule₂⟩) _ rfl r
    Pf.block Pc.block (r.block_subset_iff) (fun _ hX => (Partition.mem_blocks.1 hX).2) τ₁ O R hR

/-- **The sampling step of soundness at a miss `δ`, interiors committed late, at a circuit's two partitions.** -/
theorem ofPartitions_prCoin_accept_le_late {P : Type} [Fintype P] [Nonempty P]
    (τ₁ : P → C.Trace) (O : P → Finset (Fin C.N))
    (R : P → (Rc.design rule₁ Finset.univ).Ω → Finset C.Computed → C.Trace)
    (hR : ∀ p ω X, ∀ g ∈ O p, R p ω X g = τ₁ p g)
    (acc brk : P → (Rc.design rule₁ Finset.univ).Ω × ((X : Finset C.Computed) → (r.design rule₂ X).Ω) → Prop)
    (h : ∀ p ω, acc p ω → (∀ X ∈ (Rc.design rule₁ Finset.univ).draw ω.1,
        Disjoint ((r.design rule₂ X).draw (ω.2 X))
          ((Pf.unopened Pc.committed (O p) ∪ Pf.incorrect (R p ω.1 X)).image Pf.block)) ∨ brk p ω)
    (δ : ℝ≥0∞) :
    prCoin (fun x : P × ((Rc.design rule₁ Finset.univ).Ω × ((X : Finset C.Computed) → (r.design rule₂ X).Ω)) =>
        acc x.1 x.2 ∧ (⨆ (τ₂ : C.Trace) (_ : ∀ g ∈ O x.1, τ₂ g = τ₁ x.1 g),
          (Levels.ofPartitions (.cons Pc Rc rule₁ (.cons Pf r rule₂ .nil))).design.missProb
            ((Pf.unopened Pc.committed (O x.1) ∪ Pf.incorrect τ₂).image Pf.block)) ≤ δ) ≤
      δ + prCoin (fun x : P × ((Rc.design rule₁ Finset.univ).Ω × ((X : Finset C.Computed) → (r.design rule₂ X).Ω)) =>
        brk x.1 x.2) :=
  prCoin_accept_le_late (l₁ := ⟨Pc.blocks, Rc.design rule₁⟩) (l₂ := ⟨Pf.blocks, r.design rule₂⟩) _ rfl r
    Pf.block Pc.block (r.block_subset_iff) (fun _ hX => (Partition.mem_blocks.1 hX).2) τ₁ O R hR acc brk h δ

end Levels

/-- The chain's partition at level `i`, the first below the root at `0`, and the last past its end. -/
def Partition.Chain.part {C : Circuit} : {nc : ℕ} → {Pc : Partition C nc} → Partition.Chain C Pc → ℕ →
    Σ n, Partition C n
  | nc, Pc, .nil, _ => ⟨nc, Pc⟩
  | _, _, .cons Pf _ _ .nil, _ => ⟨_, Pf⟩
  | _, _, .cons Pf _ _ (.cons _ _ _ _), 0 => ⟨_, Pf⟩
  | _, _, .cons _ _ _ (.cons Pg Rg rg rest), i + 1 => (Partition.Chain.cons Pg Rg rg rest).part i

/-- Each level's committed gates are committed at every level below it. -/
theorem Partition.Chain.committed_part_subset {C : Circuit} :
    {nc : ℕ} → {Pc : Partition C nc} → (ch : Partition.Chain C Pc) → (k : ℕ) →
      (ch.part k).2.committed ⊆ (ch.part (k + 1)).2.committed
  | _, _, .nil, _ => subset_rfl
  | _, _, .cons _ _ _ .nil, _ => subset_rfl
  | _, _, .cons _ _ _ (.cons _ Rg _ rest), 0 => by
    cases rest <;> exact Rg.committed_subset
  | _, _, .cons _ _ _ (.cons Pg Rg rg rest), i + 1 =>
    Partition.Chain.committed_part_subset (.cons Pg Rg rg rest) i

/-- The chain's `k`-th level's blocks are its `k`-th partition's. -/
theorem Partition.Chain.levels_get_blocks {C : Circuit} :
    {nc : ℕ} → {Pc : Partition C nc} → (ch : Partition.Chain C Pc) → (k : ℕ) → (hk : k < ch.levels.length) →
      (ch.levels.get ⟨k, hk⟩).blocks = (ch.part k).2.blocks
  | _, _, .nil, _, hk => absurd hk (Nat.not_lt_zero _)
  | _, _, .cons _ _ _ .nil, 0, _ => rfl
  | _, _, .cons _ _ _ .nil, _ + 1, hk => absurd hk (by simp [Partition.Chain.levels])
  | _, _, .cons _ _ _ (.cons _ _ _ _), 0, _ => rfl
  | _, _, .cons _ _ _ (.cons Pg Rg rg rest), i + 1, hk =>
    Partition.Chain.levels_get_blocks (.cons Pg Rg rg rest) i (by simpa [Partition.Chain.levels] using hk)

namespace Partition

variable {C : Circuit} {n : ℕ}

/-- The gates a block of computed gates holds or reads. -/
def region (X : Finset C.Computed) : Finset (Fin C.N) :=
  X.image Subtype.val ∪ X.biUnion fun x => C.reads x.1

theorem region_mono {X Y : Finset C.Computed} (h : X ⊆ Y) : region X ⊆ region Y :=
  Finset.union_subset_union (Finset.image_subset_image h) (Finset.biUnion_subset_biUnion_of_subset_left _ h)

theorem unit_of_mem_block {P : Partition C n} {u : Fin n} {x : C.Computed} (hx : x ∈ P.block u) :
    P.unit x.1 = some u :=
  (Finset.mem_filter.1 hx).2

/-- Two blocks of a partition share only committed gates in their regions. -/
theorem mem_committed_of_region (P : Partition C n) {X Y : Finset C.Computed} (hX : X ∈ P.blocks.parts)
    (hY : Y ∈ P.blocks.parts) (hne : X ≠ Y) {g : Fin C.N} (hgX : g ∈ region X) (hgY : g ∈ region Y) :
    g ∈ P.committed := by
  obtain ⟨-, u, rfl⟩ := mem_blocks.1 hX
  obtain ⟨-, v, rfl⟩ := mem_blocks.1 hY
  have hsome : (some u : Option (Fin n)) ≠ some v := fun h => hne (by rw [Option.some_inj.1 h])
  simp only [region, Finset.mem_union, Finset.mem_image, Finset.mem_biUnion] at hgX hgY
  rcases hgX with ⟨x, hx, rfl⟩ | ⟨x, hx, hgx⟩ <;> rcases hgY with ⟨y, hy, hyg⟩ | ⟨y, hy, hgy⟩
  · exact absurd ((unit_of_mem_block hx).symm.trans (by rw [← hyg]; exact unit_of_mem_block hy)) hsome
  · exact P.mem_committed_of_read hgy (by rw [unit_of_mem_block hy, unit_of_mem_block hx]; exact hsome.symm)
  · exact P.mem_committed_of_read hgx (by rw [unit_of_mem_block hx, ← hyg, unit_of_mem_block hy]; exact hsome)
  · by_cases hg : P.unit g = some u
    · exact P.mem_committed_of_read hgy (by rw [unit_of_mem_block hy, hg]; exact hsome.symm)
    · exact P.mem_committed_of_read hgx (by rw [unit_of_mem_block hx]; exact Ne.symm hg)

theorem io_subset_region (P : Partition C n) (u : Fin n) : P.io u ⊆ region (P.block u) := by
  classical
  have hin : ∀ g, P.unit g = some u → C.isInput g = false := fun g hg => by
    cases h : C.isInput g
    · rfl
    · exact absurd ((P.input_iff g).2 h) (by rw [hg]; exact Option.some_ne_none u)
  intro g hg
  simp only [io, Finset.mem_filter] at hg
  simp only [region, Finset.mem_union, Finset.mem_image, Finset.mem_biUnion]
  rcases hg.2 with hu | ⟨h, hu, hgh⟩
  · exact Or.inl ⟨⟨g, hin g hu⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _, hu⟩, rfl⟩
  · exact Or.inr ⟨⟨h, hin h hu⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _, hu⟩, hgh⟩

/-- Whether a block is incorrect reads a trace only in the block's region. -/
theorem mem_image_incorrect_of_region (P : Partition C n) {X : Finset C.Computed} {a b : C.Trace}
    (h : ∀ g ∈ region X, a g = b g) (hX : X ∈ (P.incorrect a).image P.block) : X ∈ (P.incorrect b).image P.block := by
  obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hX
  refine Finset.mem_image.2 ⟨u, P.mem_incorrect.2 fun hb => P.mem_incorrect.1 hu ?_, rfl⟩
  exact P.correct_congr (fun g hg => h g (P.io_subset_region u hg)) hb

end Partition

namespace Levels

variable {C : Circuit}

/-- **Committing late does not help, at any number of levels.** At the levels of a circuit's partitions, the trace
`τ k` is committed after `Q_k` is drawn, by any rule, and agrees with `τ (k - 1)` on the committed gates of level
`k - 1`. Every sample misses the incorrect last-level blocks of the last trace with probability at most the miss of
those of one trace fixed in advance that agrees with `τ 0` on the first level's committed gates. -/
theorem ofPartitions_late_le_missProb (ch : Partition.Chain C (Partition.whole C))
    (τ : ℕ → (Levels.ofPartitions ch).Coins → C.Trace) (hτ : ∀ k, (Levels.ofPartitions ch).ReadsBelow k (τ k))
    (hstep : ∀ k ω, ∀ g ∈ (ch.part k).2.committed, τ (k + 1) ω g = τ k ω g) :
    ∃ τ' : C.Trace, (∀ g ∈ (ch.part 0).2.committed, τ' g = τ 0 (Classical.arbitrary _) g) ∧
      prCoin (fun ω => Disjoint ((Levels.ofPartitions ch).drawTo ω ch.levels.length)
          (((ch.part (ch.levels.length - 1)).2.incorrect (τ (ch.levels.length - 1) ω)).image
            (ch.part (ch.levels.length - 1)).2.block)) ≤
        (Levels.ofPartitions ch).design.missProb
          (((ch.part (ch.levels.length - 1)).2.incorrect τ').image (ch.part (ch.levels.length - 1)).2.block) := by
  classical
  refine (Levels.ofPartitions ch).late_le_missProb_levels
    (fun θ => ((ch.part (ch.levels.length - 1)).2.incorrect θ).image (ch.part (ch.levels.length - 1)).2.block)
    (fun k a b => ∀ g ∈ (ch.part k).2.committed, b g = a g) (fun X a b => ∀ g ∈ Partition.region X, a g = b g)
    (fun _ _ _ _ => rfl) (fun k _ _ _ hab hbc g hg => (hbc g (ch.committed_part_subset k hg)).trans (hab g hg))
    (fun _ _ _ h g hg => (h g hg).symm) (fun _ _ _ _ hXY h g hg => h g (Partition.region_mono hXY hg))
    (fun X a b h => ⟨Partition.mem_image_incorrect_of_region _ h,
      Partition.mem_image_incorrect_of_region _ fun g hg => (h g hg).symm⟩)
    (fun k X a b a' h ha => ⟨fun g => if g ∈ Partition.region X then a' g else b g, fun g hg => ?_,
      fun g hg => by simp only [hg, ite_true]⟩) ?_ τ hτ hstep
  · by_cases hX : g ∈ Partition.region X
    · simp only [hX, ite_true]
      exact (ha g hg).trans (h g hX)
    · simp only [hX, ite_false]
  · intro k hk θ θX hθX
    rw [show ((Levels.ofPartitions ch).levels.get ⟨k, by omega⟩).blocks = (ch.part k).2.blocks from
      ch.levels_get_blocks k _]
    refine ⟨fun g => if h : ∃ X ∈ (ch.part k).2.blocks.parts, g ∈ Partition.region X then θX h.choose g else θ g,
      fun g hg => ?_, fun X hX g hg => ?_⟩
    · by_cases h : ∃ X ∈ (ch.part k).2.blocks.parts, g ∈ Partition.region X
      · simp only [h, dite_true]
        exact hθX _ g hg
      · simp only [h, dite_false]
    · have h : ∃ X ∈ (ch.part k).2.blocks.parts, g ∈ Partition.region X := ⟨X, hX, hg⟩
      simp only [h, dite_true]
      obtain ⟨hX₀, hg₀⟩ := h.choose_spec
      by_cases hXX : h.choose = X
      · rw [hXX]
      · have hc := (ch.part k).2.mem_committed_of_region hX₀ hX hXX hg₀ hg
        rw [hθX _ g hc, hθX X g hc]

/-- **The sampling step of soundness at a miss `δ`, interiors committed late, at any number of levels.** The prover's
coins `p`, drawn before the samples and independently of them, fix its rule `τ p`. Where every accepted run's sample
misses the last trace's incorrect last-level blocks or a break occurs, the run is accepted while the largest miss over
the traces agreeing with the first on the first level's committed gates is at most `δ` with probability at most `δ`
plus the break's. -/
theorem ofPartitions_prCoin_accept_le_levels (ch : Partition.Chain C (Partition.whole C)) {P : Type} [Fintype P]
    [Nonempty P] (τ : P → ℕ → (Levels.ofPartitions ch).Coins → C.Trace)
    (hτ : ∀ p k, (Levels.ofPartitions ch).ReadsBelow k (τ p k))
    (hstep : ∀ p k ω, ∀ g ∈ (ch.part k).2.committed, τ p (k + 1) ω g = τ p k ω g)
    (acc brk : P → (Levels.ofPartitions ch).Coins → Prop)
    (h : ∀ p ω, acc p ω → Disjoint ((Levels.ofPartitions ch).drawTo ω ch.levels.length)
        (((ch.part (ch.levels.length - 1)).2.incorrect (τ p (ch.levels.length - 1) ω)).image
          (ch.part (ch.levels.length - 1)).2.block) ∨ brk p ω) (δ : ℝ≥0∞) :
    prCoin (fun x : P × (Levels.ofPartitions ch).Coins => acc x.1 x.2 ∧
        (⨆ (τ' : C.Trace) (_ : ∀ g ∈ (ch.part 0).2.committed, τ' g = τ x.1 0 (Classical.arbitrary _) g),
          (Levels.ofPartitions ch).design.missProb
            (((ch.part (ch.levels.length - 1)).2.incorrect τ').image (ch.part (ch.levels.length - 1)).2.block)) ≤ δ) ≤
      δ + prCoin (fun x : P × (Levels.ofPartitions ch).Coins => brk x.1 x.2) := by
  refine prCoin_accept_le_of _ acc brk (fun p => ⨆ (τ' : C.Trace)
    (_ : ∀ g ∈ (ch.part 0).2.committed, τ' g = τ p 0 (Classical.arbitrary _) g),
      (Levels.ofPartitions ch).design.missProb
        (((ch.part (ch.levels.length - 1)).2.incorrect τ').image (ch.part (ch.levels.length - 1)).2.block))
    (fun p => ?_) h δ
  obtain ⟨τ', hτ', hle⟩ := ofPartitions_late_le_missProb ch (τ p) (hτ p) (hstep p)
  exact hle.trans (le_iSup₂ (f := fun (τ' : C.Trace)
    (_ : ∀ g ∈ (ch.part 0).2.committed, τ' g = τ p 0 (Classical.arbitrary _) g) =>
      (Levels.ofPartitions ch).design.missProb
        (((ch.part (ch.levels.length - 1)).2.incorrect τ').image (ch.part (ch.levels.length - 1)).2.block)) τ' hτ')

end Levels

end Verity
