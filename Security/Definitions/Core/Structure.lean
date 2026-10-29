import Mathlib.Logic.Relation
import Definitions.Core.Annotation
import Definitions.Core.Formalism

/-!
# Structure: blocks, what they read and write, and least totals over families (definitions only)

The structural vocabulary: every term is a block, a family of blocks, a quantity of a block, or a least total over a
family. It is stated on core's objects and adds only what they lack.

* **Reused.** The computation is a `Declaration` (`D`, registered once as `θ`); a block's size is `Circuit.size`, its
  reads are `Circuit.into` (fresh: `Declaration.unregistered`, whose width is `Nci.novelReads`), its writes come from
  `Circuit.exits`; paths are `Circuit.Feeds`; a partition is a `Partition`, a cover an `Annotation` that `Covers`, and a
  realization is a `Containment`.
* **Provenance** (`Provenance`): a read bit is registered (in `θ`), external (another input gate, tagged with its
  source) or produced (a computed gate, tagged with its producer block). Fresh input is external plus produced
  (`fresh_eq_external_add_produced`).
* **Blocks and prefixes.** A block is any set of computed gates (`IsBlock`); a prefix is a block that reads no computed
  gate outside itself (`IsPrefix`); `downClosure S` is the computed gates that feed `S`.
* **Per block**: size (`Circuit.size`), depth as a number (`depth`), reads by kind in wire bits (`registeredIn`,
  `externalIn`, `producedIn`, by source `externalFrom`, by producer `producedFrom`), and in a time-ordered partition its
  writes by destination (`TimeOrdered.writes`: the same period, a later period, the final outputs).
* **Time** is a time-ordered partition (`TimeOrdered`): blocks tagged by location and period, the blocks before each
  period a prefix (`TimeOrdered.upto_isPrefix`). Clock ticks are the protocol layer's.
* **Least totals.** `total` sums a per-block quantity over a partition. `intrinsic R 𝔉 q a` is the least `q` over the
  members of the family `𝔉` (`partitionsAt`, `timeOrderedAt`, `coversAt`) on every realization of `a` up to the rewrite
  set `R` (`Realizes`). A realized member bounds it from above (`intrinsic_le`).

Reads are counted in wire bits (`Circuit.width`), not information. Memory is not a block quantity here: what wipes and
bandwidth bound is what a time-ordered partition's blocks carry and send.
-/

namespace Verity.Structure

open Finset

/-! ### Provenance -/

/-- **A read bit's provenance**: registered, external from a source in `Λ`, or produced by a block labelled in `L`. -/
inductive Provenance (Λ L : Type) where
  /-- in the baseline `θ`, registered once at the start -/
  | registered
  /-- an input gate outside `θ`, from the source `source` -/
  | external (source : Λ)
  /-- a computed gate, produced by the block `producer` -/
  | produced (producer : L)

/-- **The provenance of gate `a`** of `d` as a read: registered in `θ`, else external from `source a`, else produced by
`producer a`. -/
def provenance (d : Declaration) {Λ L : Type} (source : Fin d.D.N → Λ) (producer : Fin d.D.N → L) (a : Fin d.D.N) :
    Provenance Λ L :=
  if a ∈ d.θ then .registered else if d.D.isInput a then .external (source a) else .produced (producer a)

/-! ### Blocks and prefixes -/

/-- **A block** of `C`: a set of its computed gates. -/
def IsBlock (C : Circuit) (B : Finset (Fin C.N)) : Prop := B ⊆ C.computed

/-- **A prefix** of `C`: a block that reads no computed gate outside itself. -/
def IsPrefix (C : Circuit) (T : Finset (Fin C.N)) : Prop :=
  IsBlock C T ∧ ∀ a ∈ C.into T, C.isInput a = true

open Classical in
/-- **The down-closure** of `S`: the computed gates that feed a gate of `S`. -/
noncomputable def downClosure (C : Circuit) (S : Finset (Fin C.N)) : Finset (Fin C.N) :=
  C.computed.filter fun g => ∃ t ∈ S, C.Feeds g t

/-! ### Per-block quantities -/

/-- The most gates of `B` on a path inside `B` that ends at `g`, and `0` when `g ∉ B`. -/
def depthTo (C : Circuit) (B : Finset (Fin C.N)) (g : Fin C.N) : ℕ :=
  if g ∈ B then (C.reads g).sup (fun a => if _h : a < g then depthTo C B a else 0) + 1 else 0
termination_by g.val
decreasing_by all_goals assumption

/-- **The depth** of `B`: the most gates on a path inside `B`. -/
def depth (C : Circuit) (B : Finset (Fin C.N)) : ℕ := B.sup (depthTo C B)

/-- **The registered reads** of `B`: the gates of `in(B)` in `θ`. -/
def registeredIn (d : Declaration) (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into B ∩ d.θ

/-- **The external reads** of `B`: the input gates of `in(B)` outside `θ`. -/
def externalIn (d : Declaration) (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) :=
  (d.unregistered B).filter fun a => d.D.isInput a = true

/-- **The produced reads** of `B`: the computed gates of `in(B)`. -/
def producedIn (d : Declaration) (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) :=
  (d.unregistered B).filter fun a => ¬ d.D.isInput a = true

open Classical in
/-- **The reads of `B` from the source `s`**: its external reads that `source` tags `s`. -/
noncomputable def externalFrom (d : Declaration) {Λ : Type} (source : Fin d.D.N → Λ) (s : Λ)
    (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) :=
  (externalIn d B).filter fun a => source a = s

/-- **The reads of `B` from the block `B'`**, `r(B' → B)`: the gates of `B'` with a wire into `B`. -/
def producedFrom (C : Circuit) (B' B : Finset (Fin C.N)) : Finset (Fin C.N) := C.into B ∩ B'

/-! ### Families of blocks -/

/-- **Scale `c`**: every block of `P` has at most `c` gates. -/
def AtScale {C : Circuit} {n : ℕ} (P : Partition C n) (c : ℕ) : Prop := ∀ u, C.size (P.gates u) ≤ c

/-- **A cover at scale `c`**: blocks of at most `c` gates, every computed gate in at least one of them. -/
def IsCoverAt {C : Circuit} {L : Type} (α : Annotation C L) (c : ℕ) : Prop :=
  α.Covers C.computed ∧ ∀ ℓ, IsBlock C (α.holds ℓ) ∧ C.size (α.holds ℓ) ≤ c

/-- **The total** of a per-block quantity `q` over the blocks of `P`. -/
def total {C : Circuit} {n : ℕ} (P : Partition C n) (q : Finset (Fin C.N) → ℕ) : ℕ := ∑ u, q (P.gates u)

/-- **A time-ordered partition** of `C` into `n` blocks, each tagged with a location in `Loc` and one of `m` periods: no
block reads a computed gate of a later period. -/
structure TimeOrdered (C : Circuit) (n : ℕ) (Loc : Type) (m : ℕ) extends Partition C n where
  /-- the location each block runs at -/
  loc : Fin n → Loc
  /-- the period each block runs in -/
  period : Fin n → Fin m
  /-- a block reads computed gates of its own period or earlier ones only -/
  ordered : ∀ g, ∀ a ∈ C.reads g, ∀ u v, unit g = some u → unit a = some v → period v ≤ period u

/-- **Where a write goes**: to a block of the same period, to a block of a later period, or to the final outputs. -/
inductive Destination where
  /-- read by another block of the writer's period -/
  | samePeriod
  /-- read by a block of a later period -/
  | laterPeriod
  /-- an output gate of the circuit -/
  | finalOutput

namespace TimeOrdered

variable {C : Circuit} {n m : ℕ} {Loc : Type} (F : TimeOrdered C n Loc m)

open Classical in
/-- The gates of the blocks of the periods before `j`. -/
noncomputable def upto (j : ℕ) : Finset (Fin C.N) :=
  univ.filter fun g => ∃ u, F.unit g = some u ∧ (F.period u : ℕ) < j

open Classical in
/-- **Block `u`'s writes to `δ`**: its gates that another block of its period reads, that a block of a later period
reads, or that are outputs of `C`. -/
noncomputable def writes : Destination → Fin n → Finset (Fin C.N)
  | .samePeriod, u => (F.toPartition.gates u).filter fun g =>
      ∃ h v, g ∈ C.reads h ∧ F.unit h = some v ∧ v ≠ u ∧ F.period v = F.period u
  | .laterPeriod, u => (F.toPartition.gates u).filter fun g =>
      ∃ h v, g ∈ C.reads h ∧ F.unit h = some v ∧ F.period u < F.period v
  | .finalOutput, u => (F.toPartition.gates u).filter fun g => g ∈ C.outputs

/-- **The total writes to `δ`** over the blocks, in bits: communication at `samePeriod`, carried state at
`laterPeriod`. -/
noncomputable def writesTotal (δ : Destination) : ℕ := ∑ u, C.width (F.writes δ u)

end TimeOrdered

/-! ### Realizations and least totals -/

/-- **`d` realizes `a` up to `R`**: `d` contains `a` or a chain of `R`-rewrites of it, registered gates going to
registered gates and external to external. -/
def Realizes (R : Declaration → Declaration → Prop) (a d : Declaration) : Prop :=
  ∃ a', Relation.ReflTransGen R a a' ∧
    ∃ M : Containment a'.D d.D, ∀ g, a'.D.isInput g = true → (M.emb g ∈ d.θ ↔ g ∈ a'.θ)

/-- **A named family** of block structures: on each declaration, its members' type and which members it admits. -/
structure Family where
  /-- the members' type on a declaration -/
  Member : Declaration → Type
  /-- the members the family admits -/
  admits : (d : Declaration) → Member d → Prop

/-- **Partitions at scale `c`**. -/
def partitionsAt (c : ℕ) : Family where
  Member d := Σ n, Partition d.D n
  admits _ x := AtScale x.2 c

/-- **Time-ordered partitions at scale `c`**, located in `Loc`. -/
def timeOrderedAt (Loc : Type) (c : ℕ) : Family where
  Member d := Σ n m, TimeOrdered d.D n Loc m
  admits _ x := AtScale x.2.2.toPartition c

/-- **Covers at scale `c`**. -/
def coversAt (c : ℕ) : Family where
  Member d := Σ n, Annotation d.D (Fin n)
  admits _ x := IsCoverAt x.2 c

/-- **The intrinsic value** of `q` for the method `a`: its least value over the members of `𝔉` on the realizations of
`a` up to `R`, and `⊤` when there is none. -/
noncomputable def intrinsic (R : Declaration → Declaration → Prop) (𝔉 : Family)
    (q : (d : Declaration) → 𝔉.Member d → ℕ) (a : Declaration) : ℕ∞ :=
  ⨅ (d : Declaration) (_ : Realizes R a d) (x : 𝔉.Member d) (_ : 𝔉.admits d x), (q d x : ℕ∞)

/-- **The least total fresh input at scale `c`** (`K*(c)`): `intrinsic` over partitions at scale `c`. -/
noncomputable def leastFresh (R : Declaration → Declaration → Prop) (c : ℕ) (a : Declaration) : ℕ∞ :=
  intrinsic R (partitionsAt c)
    (fun d (x : Σ n, Partition d.D n) => total x.2 fun B => d.D.width (d.unregistered B)) a

/-! ### Sanity lemmas -/

theorem IsBlock.union {C : Circuit} {B₁ B₂ : Finset (Fin C.N)} (h₁ : IsBlock C B₁) (h₂ : IsBlock C B₂) :
    IsBlock C (B₁ ∪ B₂) :=
  Finset.union_subset h₁ h₂

theorem isPrefix_computed (C : Circuit) : IsPrefix C C.computed := by
  refine ⟨Finset.Subset.refl _, fun a ha => ?_⟩
  have h := (Finset.mem_filter.1 ha).2.1
  cases hi : C.isInput a with
  | true => rfl
  | false => exact absurd (show a ∈ C.computed from Finset.mem_filter.2 ⟨Finset.mem_univ _, hi⟩) h

theorem fresh_eq_external_add_produced (d : Declaration) (B : Finset (Fin d.D.N)) :
    d.D.width (d.unregistered B) = d.D.width (externalIn d B) + d.D.width (producedIn d B) :=
  (Finset.sum_filter_add_sum_filter_not (d.unregistered B) (fun a => d.D.isInput a = true) d.D.w).symm

theorem TimeOrdered.upto_isPrefix {C : Circuit} {n m : ℕ} {Loc : Type} (F : TimeOrdered C n Loc m) (j : ℕ) :
    IsPrefix C (F.upto j) := by
  refine ⟨fun g hg => ?_, fun a ha => ?_⟩
  · obtain ⟨u, hu, -⟩ := (Finset.mem_filter.1 hg).2
    refine Finset.mem_filter.2 ⟨Finset.mem_univ _, ?_⟩
    cases hi : C.isInput g with
    | false => rfl
    | true => simp [(F.input_iff g).2 hi] at hu
  · obtain ⟨-, hnot, g, hg, hag⟩ := Finset.mem_filter.1 ha
    obtain ⟨u, hu, hj⟩ := (Finset.mem_filter.1 hg).2
    by_contra hin
    cases hv : F.unit a with
    | none => exact hin ((F.input_iff a).1 hv)
    | some v =>
      exact hnot (Finset.mem_filter.2 ⟨Finset.mem_univ _, v, hv,
        lt_of_le_of_lt (Fin.le_def.1 (F.ordered g a hag u v hu hv)) hj⟩)

theorem intrinsic_le {R : Declaration → Declaration → Prop} {𝔉 : Family} {q : (d : Declaration) → 𝔉.Member d → ℕ}
    {a d : Declaration} (hd : Realizes R a d) {x : 𝔉.Member d} (hx : 𝔉.admits d x) :
    intrinsic R 𝔉 q a ≤ q d x :=
  iInf₂_le_of_le d hd (iInf₂_le_of_le x hx le_rfl)

end Verity.Structure
