module

public import Proofs.Properties.Nci.RecordTakes
public import Proofs.Properties.Nci.RecordGrid
public import Proofs.Properties.Nci.RecordRowBits
public import Mathlib.Tactic

/-!
# The landing bound

A read of a row's leaf lands, in the wired circuit `D`, on the gates the wires carry it to (`Lands`).
`landing_le`: the gates a row's reads land on are together no wider than the row.

`LeafWide` says a gate is as wide as the leaf that names it, for every leaf of a reader row and of an output row.
The bound reads less, `LeafHolds`: three inequalities, two of them only at gates no wire covers. It is proved
from `LeafHolds` (`landing_le_of_holds`), and `landing_le` is that at what `LeafWide` gives
(`leafHolds_of_leafWide`).

The bound is false leaf by leaf: one leaf of a writer may span several leaves of its reader, and then each of them
lands on the whole of it. It holds for a row, and for a leaf of a unit's output row:

* a gate no wire covers lands on itself alone (`lands_unwired`), one gate of `D`, no wider than its leaf;
* a leaf of a row with a writer lands where the writer's leaves that meet it land (`lands_wired_row`), and those
  are leaves of one output row of one unit (`writerLeaf_data`);
* `outLeaf_land_le`, by induction on the rank of the unit: a leaf of a unit's output row lands on gates no wider
  than the leaf. Where the leaf is a leaf of a row of the unit with a writer, the writer's leaves that meet it lie
  inside it (`Aligned`: the unit passes this leaf on), they are apart, each is an output leaf of a unit of
  smaller rank, and the leaf has no more bits as the row's than as the output's;
* `width_charge` adds these up: gates charged to leaves of one row that lie in `[A, B)`, each leaf holding gates
  no wider than itself, are together no wider than `B - A`.

A row with no writer lands on its own leaves, and a row with a writer inside what its writer's output row lands
on, which is no wider than the output row and so than the row (`OutFits`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref Ty Defn)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row Grid fieldsOf outerOf)
open NciCheck.Record (inSlot)
open NciCheck.Record.SecurityProofs (slotOf OneGrid)
open Spot.Bits (bitsOf)

/-! ## What the bound reads of the widths -/

/-- **A leaf holds its gate**: what the landing bound reads of `LeafWide`. A gate no wire covers that is a leaf of
a reader row, or a leaf of a unit's output row, holds a value of no more bits than the leaf; and a leaf of a row
with a writer that the row's unit passes on, as a leaf of its output row, has no more bits than that output
leaf. -/
def LeafHolds (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) : Prop :=
  (∀ r l g lo hi, RowLeaf P rs W r l g lo hi → ¬ Wired P rs W S g →
      ∃ p ops sc w, P.gate g = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value w ∧ w ≤ hi - lo) ∧
    (∀ tid u sc out l g lo hi, UnitAt P rs tid u sc → OutLeaf P sc out l g lo hi → ¬ Wired P rs W S g →
      ∃ p ops sc' w, P.gate g = .ok (p, ops, sc') ∧ (P.defs[p]!).ret = .value w ∧ w ≤ hi - lo) ∧
    ∀ r l g lo hi sc out l' lo' hi', (S.writerOf r.tid r.unit r.port r.row).isSome →
      RowLeaf P rs W r l g lo hi → UnitAt P rs r.tid r.unit sc → OutLeaf P sc out l' g lo' hi' →
      hi - lo ≤ hi' - lo'

variable {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}

/-- A gate as wide as the leaf that names it is held by the leaf. -/
theorem leafHolds_of_leafWide (hW : LeafWide P rs W) : LeafHolds P rs W S := by
  refine ⟨fun r l g lo hi hrl _ => ?_, fun tid u sc out l g lo hi hsc hout _ => ?_,
    fun r l g lo hi sc out l' lo' hi' _ hrl hsc hout => ?_⟩
  · obtain ⟨p, ops, sc, hg, hret⟩ := hW.1 r l g lo hi hrl
    exact ⟨p, ops, sc, hi - lo, hg, hret, Nat.le_refl _⟩
  · obtain ⟨p, ops, sc', hg, hret⟩ := hW.2 tid u sc out l g lo hi hsc hout
    exact ⟨p, ops, sc', hi - lo, hg, hret, Nat.le_refl _⟩
  · obtain ⟨p, ops, sc₁, hg, hret⟩ := hW.1 r l g lo hi hrl
    obtain ⟨p', ops', sc₂, hg', hret'⟩ := hW.2 r.tid r.unit sc out l' g lo' hi' hsc hout
    rw [hg'] at hg
    have hp : p' = p := (Prod.mk.inj (Except.ok.inj hg)).1
    subst hp
    rw [hret'] at hret
    exact Nat.le_of_eq (Ty.value.inj hret).symm

/-! ## The leaves of a row -/

/-- Leaves of a row whose bits meet are one leaf. -/
theorem leaf_eq_of_meet (ws : Array ℕ) (l l' : ℕ) (hl : l < ws.size) (hl' : l' < ws.size)
    (h : bitAt ws l' < bitAt ws l + ws[l]!) (h' : bitAt ws l < bitAt ws l' + ws[l']!) : l = l' := by
  rcases Nat.lt_trichotomy l l' with hlt | heq | hgt
  · have := bitAt_mono ws l l' hlt (Nat.le_of_lt hl')
    omega
  · exact heq
  · have := bitAt_mono ws l' l hgt (Nat.le_of_lt hl)
    omega

/-- A leaf ends at the row's bits, or before. -/
theorem bitAt_leaf_le (ws : Array ℕ) (l : ℕ) (hl : l < ws.size) : bitAt ws l + ws[l]! ≤ bitAt ws ws.size :=
  bitAt_mono ws l ws.size hl (Nat.le_refl _)

/-- A leaf of a reader row is a leaf of the row `inRow` cuts from the parameter its port names. -/
theorem rowLeaf_data (hN : PortsNamed P rs) {r : Row} {sc : Scope} (hsc : UnitAt P rs r.tid r.unit sc) {i : ℕ}
    (hi : i < (P.defs[sc.d]!).params.size) (hp : (P.defs[sc.d]!).params[i]!.1 = r.port) {l g lo hi' : ℕ}
    (h : RowLeaf P rs W r l g lo hi') :
    l < (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.size ∧
      Flock.Program.resolve P sc
        (⟨true, i, (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).1 + l⟩ : Ref) = .ok g ∧
      lo = bitAt (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2 l ∧
      hi' = lo + (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2[l]! := by
  obtain ⟨sc₁, hsc₁, i₁, hi₁, hp₁, _, hl, hres, hlo, hhi⟩ := h
  have e := unitAt_fun P rs r.tid r.unit sc₁ sc hsc₁ hsc
  subst e
  have e := hN r.tid r.unit sc₁ i₁ i hsc hi₁ hi (hp₁.trans hp.symm)
  subst e
  exact ⟨hl, hres, hlo, hhi⟩

/-- A row's leaf index names one gate and one range of bits. -/
theorem rowLeaf_fun (hN : PortsNamed P rs) {r : Row} {l g lo hi g' lo' hi' : ℕ}
    (h : RowLeaf P rs W r l g lo hi) (h' : RowLeaf P rs W r l g' lo' hi') : g = g' ∧ lo = lo' ∧ hi = hi' := by
  obtain ⟨sc, hsc, i, hi₁, hp, _, _, _, _, _⟩ := id h
  have d := rowLeaf_data hN hsc hi₁ hp h
  have d' := rowLeaf_data hN hsc hi₁ hp h'
  refine ⟨Except.ok.inj (d.2.1.symm.trans d'.2.1), d.2.2.1.trans d'.2.2.1.symm, ?_⟩
  rw [d.2.2.2, d'.2.2.2, d.2.2.1, d'.2.2.1]

/-- A leaf of the output row named `"o" ++ toString f` is a leaf of the row `outRow` gives for field `f`. -/
theorem outLeaf_data {sc : Scope} {f l g lo hi : ℕ} (h : OutLeaf P sc ("o" ++ toString f) l g lo hi) :
    l < (outRow (P.defs[sc.d]!) f).2.size ∧
      Flock.Program.resolve P sc ((P.defs[sc.d]!).body.ret.get ((outRow (P.defs[sc.d]!) f).1 + l)) = .ok g ∧
      lo = bitAt (outRow (P.defs[sc.d]!) f).2 l ∧ hi = lo + (outRow (P.defs[sc.d]!) f).2[l]! := by
  obtain ⟨f', hf', _, hl, hres, hlo, hhi⟩ := h
  have e := outName_inj f f' hf'
  subst e
  exact ⟨hl, hres, hlo, hhi⟩

/-- The writer's leaves of a row are leaves of one output row of one unit. -/
theorem writerLeaf_data {r : Row} {l₀ g₀ lo₀ hi₀ : ℕ} (h₀ : WriterLeaf P rs S r l₀ g₀ lo₀ hi₀) :
    ∃ wt wu sc' f, S.writerOf r.tid r.unit r.port r.row = some (wt, wu, "o" ++ toString f) ∧ 0 ≤ wu ∧
      UnitAt P rs wt wu.toNat sc' ∧ f < (fieldsOf (P.defs[sc'.d]!).ret).length ∧
      ∀ l g lo hi, WriterLeaf P rs S r l g lo hi → OutLeaf P sc' ("o" ++ toString f) l g lo hi := by
  obtain ⟨wt, wu, out, sc', hw, h0, hu, f, hout, hf, _⟩ := h₀
  subst hout
  refine ⟨wt, wu, sc', f, hw, h0, hu, hf, ?_⟩
  rintro l g lo hi ⟨wt₁, wu₁, out₁, sc₁, hw₁, _, hu₁, ho⟩
  rw [hw] at hw₁
  have e := Option.some.inj hw₁
  have e1 := (Prod.mk.inj e).1
  have e2 := (Prod.mk.inj (Prod.mk.inj e).2).1
  have e3 := (Prod.mk.inj (Prod.mk.inj e).2).2
  subst e1 e2 e3
  have e4 := unitAt_fun P rs wt wu.toNat sc₁ sc' hu₁ hu
  subst e4
  exact ho

/-! ## Where a read lands -/

/-- A read of a gate no wire covers lands on the gate itself. -/
theorem lands_unwired {o a : ℕ} (hw : ¬ Wired P rs W S o) (h : Lands P rs W S o a) : a = o := by
  cases h with
  | here _ _ => rfl
  | wire _ g _ hf _ =>
    obtain ⟨r, l, l', lo, hi, lo', hi', hrl, ⟨wt, wu, out, sc', hwr, _⟩, _⟩ := hf
    exact absurd ⟨r, l, lo, hi, Option.isSome_iff_exists.2 ⟨_, hwr⟩, hrl⟩ hw

/-- A read of a leaf of a row with a writer lands where a leaf of the writer that meets it lands. -/
theorem lands_wired_row (hO : OwnRows P rs W) (hN : PortsNamed P rs) {r : Row} {l g lo hi : ℕ}
    (hrl : RowLeaf P rs W r l g lo hi) (hw : (S.writerOf r.tid r.unit r.port r.row).isSome) {a : ℕ}
    (h : Lands P rs W S g a) :
    ∃ l' g' lo' hi', WriterLeaf P rs S r l' g' lo' hi' ∧ lo < hi' ∧ lo' < hi ∧ Lands P rs W S g' a := by
  cases h with
  | here _ hn => exact absurd ⟨r, l, lo, hi, hw, hrl⟩ hn
  | wire _ g' _ hf hl =>
    obtain ⟨r₁, l₁, l', lo₁, hi₁, lo', hi', hrl₁, hwl, hm1, hm2⟩ := hf
    obtain ⟨e1, e2⟩ := hO.2 r₁ l₁ lo₁ hi₁ r l lo hi g hrl₁ hrl
    subst e1 e2
    obtain ⟨_, e3, e4⟩ := rowLeaf_fun hN hrl₁ hrl
    subst e3 e4
    exact ⟨l', g', lo', hi', hwl, hm1, hm2, hl⟩

/-! ## Widths in the wired circuit -/

/-- Fewer gates are no wider. -/
theorem width_mono (D : Verity.Circuit) {T T' : Finset (Fin D.N)} (h : T ⊆ T') : D.width T ≤ D.width T' :=
  Finset.sum_le_sum_of_subset h

/-- No gates have no width. -/
theorem width_none_le (D : Verity.Circuit) (T : Finset (Fin D.N)) (c : ℕ) (h : ∀ a ∈ T, False) : D.width T ≤ c :=
  (Nat.le_of_eq (Finset.sum_eq_zero fun a ha => (h a ha).elim)).trans (Nat.zero_le c)

/-- Gates of `D` that are one Program gate are one gate, as wide as any of them. -/
theorem width_one_le (D : Verity.Circuit) (at_ : Fin D.N → ℕ) (hinj : Function.Injective at_)
    (T : Finset (Fin D.N)) (c : ℕ) (h1 : ∀ a ∈ T, ∀ b ∈ T, at_ a = at_ b) (hw : ∀ a ∈ T, D.w a ≤ c) :
    D.width T ≤ c := by
  have hc : T.card ≤ 1 := Finset.card_le_one.2 fun a ha b hb => hinj (h1 a ha b hb)
  have h2 : D.width T ≤ T.card • c := Finset.sum_le_card_nsmul T D.w c hw
  have h3 : T.card * c ≤ 1 * c := Nat.mul_le_mul_right c hc
  rw [smul_eq_mul] at h2
  omega

/-- **Charging gates to the leaves of a row**: gates each charged to a leaf that lies in `[A, B)`, the gates
charged to one leaf no wider than the leaf, are together no wider than `B - A`. -/
theorem width_charge (D : Verity.Circuit) (ws : Array ℕ) (A B : ℕ) (charged : ℕ → Finset (Fin D.N))
    (hC : ∀ l, l < ws.size → D.width (charged l) ≤ ws[l]!) (T : Finset (Fin D.N))
    (hT : ∀ a ∈ T, ∃ l, l < ws.size ∧ A ≤ bitAt ws l ∧ bitAt ws l + ws[l]! ≤ B ∧ a ∈ charged l) :
    D.width T ≤ B - A := by
  classical
  have key : ∀ n, n ≤ ws.size → ∀ T : Finset (Fin D.N),
      (∀ a ∈ T, ∃ l, l < n ∧ A ≤ bitAt ws l ∧ bitAt ws l + ws[l]! ≤ B ∧ a ∈ charged l) →
      D.width T ≤ min B (bitAt ws n) - A := by
    intro n
    induction n with
    | zero =>
      intro _ T hT
      refine width_none_le D T _ fun a ha => ?_
      obtain ⟨l, hl, _⟩ := hT a ha
      exact absurd hl (Nat.not_lt_zero l)
    | succ n ih =>
      intro hn T hT
      have e := bitAt_succ ws n (by omega)
      by_cases q : A ≤ bitAt ws n ∧ bitAt ws n + ws[n]! ≤ B
      · have hsplit : D.width T =
            D.width (T.filter fun a => a ∈ charged n) + D.width (T.filter fun a => ¬ a ∈ charged n) :=
          (Finset.sum_filter_add_sum_filter_not T (fun a => a ∈ charged n) D.w).symm
        have h1 : D.width (T.filter fun a => a ∈ charged n) ≤ ws[n]! :=
          (width_mono D fun a ha => (Finset.mem_filter.1 ha).2).trans (hC n (by omega))
        have h2 : D.width (T.filter fun a => ¬ a ∈ charged n) ≤ min B (bitAt ws n) - A := by
          refine ih (by omega) _ fun a ha => ?_
          obtain ⟨ha1, ha2⟩ := Finset.mem_filter.1 ha
          obtain ⟨l, hl, hA, hB, hx⟩ := hT a ha1
          have hne : l ≠ n := fun hln => ha2 (hln ▸ hx)
          exact ⟨l, by omega, hA, hB, hx⟩
        omega
      · have h2 : D.width T ≤ min B (bitAt ws n) - A := by
          refine ih (by omega) T fun a ha => ?_
          obtain ⟨l, hl, hA, hB, hx⟩ := hT a ha
          have hne : l ≠ n := fun hln => q (hln ▸ ⟨hA, hB⟩)
          exact ⟨l, by omega, hA, hB, hx⟩
        omega
  have := key ws.size (Nat.le_refl _) T hT
  omega

/-! ## The gates a read lands on -/

open Classical in
/-- The gates of `D` on which a read of the Program gate `g` lands. -/
noncomputable def landOf (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary)
    {D : Verity.Circuit} (at_ : Fin D.N → ℕ) (g : ℕ) : Finset (Fin D.N) :=
  Finset.univ.filter fun a => Lands P rs W S g (at_ a)

open Classical in
/-- The gates of `D` on which a read of leaf `l` of the row `r`'s writer lands. -/
noncomputable def writerLand (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary)
    {D : Verity.Circuit} (at_ : Fin D.N → ℕ) (r : Row) (l : ℕ) : Finset (Fin D.N) :=
  Finset.univ.filter fun a => ∃ g lo hi, WriterLeaf P rs S r l g lo hi ∧ Lands P rs W S g (at_ a)

open Classical in
/-- The gates of `D` that are leaf `l` of the row `r`. -/
noncomputable def leafGates (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) {D : Verity.Circuit}
    (at_ : Fin D.N → ℕ) (r : Row) (l : ℕ) : Finset (Fin D.N) :=
  Finset.univ.filter fun a => ∃ lo hi, RowLeaf P rs W r l (at_ a) lo hi

variable {D : Verity.Circuit} {at_ : Fin D.N → ℕ}

/-- A gate of `D` is in `landOf g` when a read of `g` lands on it. -/
theorem mem_landOf {g : ℕ} {a : Fin D.N} : a ∈ landOf P rs W S at_ g ↔ Lands P rs W S g (at_ a) := by
  classical
  simp only [landOf, Finset.mem_filter, Finset.mem_univ, true_and]

/-- A gate of `D` is in `writerLand r l` when a read of leaf `l` of `r`'s writer lands on it. -/
theorem mem_writerLand {r : Row} {l : ℕ} {a : Fin D.N} :
    a ∈ writerLand P rs W S at_ r l ↔ ∃ g lo hi, WriterLeaf P rs S r l g lo hi ∧ Lands P rs W S g (at_ a) := by
  classical
  simp only [writerLand, Finset.mem_filter, Finset.mem_univ, true_and]

/-- A gate of `D` is in `leafGates r l` when it is leaf `l` of `r`. -/
theorem mem_leafGates {r : Row} {l : ℕ} {a : Fin D.N} :
    a ∈ leafGates P rs W at_ r l ↔ ∃ lo hi, RowLeaf P rs W r l (at_ a) lo hi := by
  classical
  simp only [leafGates, Finset.mem_filter, Finset.mem_univ, true_and]

/-- A gate of `D` is in `landing r` when a read of a leaf of `r` lands on it. -/
theorem mem_landing {r : Row} {a : Fin D.N} :
    a ∈ landing P rs W S at_ r ↔ ∃ l o lo hi, RowLeaf P rs W r l o lo hi ∧ Lands P rs W S o (at_ a) := by
  classical
  simp only [landing, Finset.mem_filter, Finset.mem_univ, true_and]

/-- A read of a gate no wire covers lands on one gate of `D`. -/
theorem landOf_unwired_le (hinj : Function.Injective at_) {g c : ℕ} (hw : ¬ Wired P rs W S g)
    (hc : ∀ a, at_ a = g → D.w a ≤ c) : D.width (landOf P rs W S at_ g) ≤ c := by
  have hat : ∀ a ∈ landOf P rs W S at_ g, at_ a = g := fun a ha =>
    lands_unwired hw (mem_landOf.1 ha)
  exact width_one_le D at_ hinj _ c (fun a ha b hb => (hat a ha).trans (hat b hb).symm) fun a ha => hc a (hat a ha)

/-- A gate of `D` is as wide as the value of the Program gate it is. -/
theorem width_of_gate (hL : LaysOut P rs W S D at_) {a : Fin D.N} {p : ℕ} {ops : Array ℕ} {sc : Scope} {w : ℕ}
    (hg : P.gate (at_ a) = .ok (p, ops, sc)) (hret : (P.defs[p]!).ret = .value w) : D.w a = w := by
  obtain ⟨p', ops', sc', hg', hret', _⟩ := hL.2.2 a
  rw [hg] at hg'
  have hp : p = p' := (Prod.mk.inj (Except.ok.inj hg')).1
  subst hp
  rw [hret] at hret'
  exact (Ty.value.inj hret').symm

/-- A leaf of a row's writer lands on gates no wider than the leaf, where each of the writer's leaves does. -/
theorem writerLand_le {r : Row} {sc' : Scope} {f : ℕ}
    (hall : ∀ l g lo hi, WriterLeaf P rs S r l g lo hi → OutLeaf P sc' ("o" ++ toString f) l g lo hi)
    (hstep : ∀ l g lo hi, WriterLeaf P rs S r l g lo hi → D.width (landOf P rs W S at_ g) ≤ hi - lo) (l : ℕ) :
    D.width (writerLand P rs W S at_ r l) ≤ (outRow (P.defs[sc'.d]!) f).2[l]! := by
  by_cases hne : ∃ g lo hi, WriterLeaf P rs S r l g lo hi
  · obtain ⟨g₀, lo₀, hi₀, h₀⟩ := hne
    have d₀ := outLeaf_data (hall l g₀ lo₀ hi₀ h₀)
    have hsub : writerLand P rs W S at_ r l ⊆ landOf P rs W S at_ g₀ := by
      intro a ha
      obtain ⟨g, lo, hi, hwl, hl⟩ := mem_writerLand.1 ha
      have d := outLeaf_data (hall l g lo hi hwl)
      have e : g = g₀ := Except.ok.inj (d.2.1.symm.trans d₀.2.1)
      subst e
      exact mem_landOf.2 hl
    have h1 := (width_mono D hsub).trans (hstep l g₀ lo₀ hi₀ h₀)
    have h2 := d₀.2.2.2
    omega
  · refine width_none_le D _ _ fun a ha => ?_
    obtain ⟨g, lo, hi, hwl, _⟩ := mem_writerLand.1 ha
    exact hne ⟨g, lo, hi, hwl⟩

/-! ## A leaf of an output row -/

/-- **The bound for an output leaf**: a read of a leaf of a unit's output row lands on gates together no wider than
the leaf. -/
theorem outLeaf_land_le (hO : OwnRows P rs W) (hA : Aligned P rs W S) (hL : LaysOut P rs W S D at_)
    (hH : LeafHolds P rs W S) (hR : ReturnsOwn P rs W) (hU : UnitsComputed P rs) (hN : PortsNamed P rs)
    (ρ : String × ℤ → ℕ) (hρ : Ranked S ρ) :
    ∀ (n : ℕ) (tid : String) (u : ℕ) (sc : Scope) (out : String) (l g lo hi : ℕ), ρ (tid, (u : ℤ)) = n →
      UnitAt P rs tid u sc → OutLeaf P sc out l g lo hi → D.width (landOf P rs W S at_ g) ≤ hi - lo := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro tid u sc out l g lo hi hn hsc hout
    have hwd : ¬ Wired P rs W S g → ∀ a, at_ a = g → D.w a ≤ hi - lo := by
      intro hnw a ha
      obtain ⟨p, ops, sc', w, hg, hret, hle⟩ := hH.2.1 tid u sc out l g lo hi hsc hout hnw
      rw [← ha] at hg
      rw [width_of_gate hL hg hret]
      exact hle
    rcases hR tid u sc out l g lo hi hsc hout with hin | ⟨r', l'', lo'', hi'', htid, hunit, hrl⟩
    · -- a gate of the unit: computed, so no row's leaf
      have hnw : ¬ Wired P rs W S g := by
        rintro ⟨r, l₁, lo₁, hi₁, _, hrl⟩
        obtain ⟨p, ops, sc₁, hg, hinp⟩ := hO.1 r l₁ g lo₁ hi₁ hrl
        have hc := hU tid u sc g p ops sc₁ hsc hin.1 hin.2 hg
        rw [hinp] at hc
        exact Bool.noConfusion hc
      exact landOf_unwired_le hL.1 hnw (hwd hnw)
    · -- a leaf of a row of the unit
      subst htid hunit
      by_cases hwr : (S.writerOf r'.tid r'.unit r'.port r'.row).isSome
      · refine Nat.le_trans ?_ (hH.2.2 r' l'' g lo'' hi'' sc out l lo hi hwr hrl hsc hout)
        by_cases hne : ∃ l' g' lo' hi', WriterLeaf P rs S r' l' g' lo' hi'
        · obtain ⟨l₀, g₀, lo₀, hi₀, h₀⟩ := hne
          obtain ⟨wt, wu, sc', f, hw, h0, hu, _, hall⟩ := writerLeaf_data h₀
          obtain ⟨scr, hscr, i, hi₁, hp, _⟩ := id hrl
          have hd := rowLeaf_data hN hscr hi₁ hp hrl
          have hstep : ∀ l g lo hi, WriterLeaf P rs S r' l g lo hi →
              D.width (landOf P rs W S at_ g) ≤ hi - lo := by
            intro l₁ g₁ lo₁ hi₁ h₁
            have hrank := hρ r' wt wu _ hw
            have hwu : ((wu.toNat : ℕ) : ℤ) = wu := Int.toNat_of_nonneg h0
            refine ih (ρ (wt, ((wu.toNat : ℕ) : ℤ))) ?_ wt wu.toNat sc' _ l₁ g₁ lo₁ hi₁ rfl hu
              (hall l₁ g₁ lo₁ hi₁ h₁)
            rw [hwu, ← hn]
            exact hrank
          refine width_charge D (outRow (P.defs[sc'.d]!) f).2 lo'' hi'' (writerLand P rs W S at_ r')
            (fun l' _ => writerLand_le hall hstep l') _ fun a ha => ?_
          obtain ⟨l', g', lo', hi', hwl, hm1, hm2, hl'⟩ := lands_wired_row hO hN hrl hwr (mem_landOf.1 ha)
          have hod := outLeaf_data (hall l' g' lo' hi' hwl)
          rcases hA r' l' g' lo' hi' hwl with ⟨l₂, g₂, lo₂, hi₂, hrl₂, hle1, hle2⟩ | hsec
          · -- the writer's leaf lies inside one leaf of the row, which is this one
            have hd₂ := rowLeaf_data hN hscr hi₁ hp hrl₂
            have e : l'' = l₂ := by
              refine leaf_eq_of_meet _ l'' l₂ hd.1 hd₂.1 ?_ ?_
              · have h1 := hd.2.2.1
                have h2 := hd.2.2.2
                have h3 := hd₂.2.2.1
                omega
              · have h1 := hd.2.2.1
                have h3 := hd₂.2.2.1
                have h4 := hd₂.2.2.2
                omega
            subst e
            obtain ⟨_, e1, e2⟩ := rowLeaf_fun hN hrl₂ hrl
            have h5 := hod.2.2.1
            have h6 := hod.2.2.2
            exact ⟨l', hod.1, by omega, by omega, mem_writerLand.2 ⟨g', lo', hi', hwl, hl'⟩⟩
          · -- else the unit passes on no leaf of the row that meets the writer's, and it passes this one on
            exact absurd ⟨sc, out, l, lo, hi, hsc, hout⟩ (hsec l'' g lo'' hi'' hrl hm1 hm2)
        · refine width_none_le D _ _ fun a ha => ?_
          obtain ⟨l', g', lo', hi', hwl, _⟩ := lands_wired_row hO hN hrl hwr (mem_landOf.1 ha)
          exact hne ⟨l', g', lo', hi', hwl⟩
      · -- the row has no writer: no wire covers the leaf
        have hnw : ¬ Wired P rs W S g := by
          rintro ⟨r, l₁, lo₁, hi₁, hsome, hrl₁⟩
          obtain ⟨e, _⟩ := hO.2 r l₁ lo₁ hi₁ r' l'' lo'' hi'' g hrl₁ hrl
          subst e
          exact hwr hsome
        exact landOf_unwired_le hL.1 hnw (hwd hnw)

/-! ## A row -/

/-- **The landing bound**, from `LeafHolds`: the gates a row's reads land on are together no wider than the
row. -/
theorem landing_le_of_holds (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary)
    (D : Verity.Circuit) (at_ : Fin D.N → ℕ) (hO : OwnRows P rs W) (hA : Aligned P rs W S)
    (hL : LaysOut P rs W S D at_) (hH : LeafHolds P rs W S) (hR : ReturnsOwn P rs W) (hU : UnitsComputed P rs)
    (hN : PortsNamed P rs) (hF : OutFits P rs W S) (ρ : String × ℤ → ℕ) (hρ : Ranked S ρ) (r : Row) (sc : Scope)
    (hsc : UnitAt P rs r.tid r.unit sc) (i : ℕ) (hi : i < (P.defs[sc.d]!).params.size)
    (hp : (P.defs[sc.d]!).params[i]!.1 = r.port) :
    D.width (landing P rs W S at_ r) ≤
      (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.foldl (· + ·) 0 := by
  by_cases hwr : (S.writerOf r.tid r.unit r.port r.row).isSome
  · -- the row has a writer: its reads land where the writer's leaves land
    by_cases hne : ∃ l' g' lo' hi', WriterLeaf P rs S r l' g' lo' hi'
    · obtain ⟨l₀, g₀, lo₀, hi₀, h₀⟩ := hne
      obtain ⟨wt, wu, sc', f, hw, h0, hu, hf, hall⟩ := writerLeaf_data h₀
      have hstep : ∀ l g lo hi, WriterLeaf P rs S r l g lo hi →
          D.width (landOf P rs W S at_ g) ≤ hi - lo := fun l₁ g₁ lo₁ hi₁ h₁ =>
        outLeaf_land_le hO hA hL hH hR hU hN ρ hρ _ wt wu.toNat sc' _ l₁ g₁ lo₁ hi₁ rfl hu
          (hall l₁ g₁ lo₁ hi₁ h₁)
      have h1 : D.width (landing P rs W S at_ r) ≤
          bitAt (outRow (P.defs[sc'.d]!) f).2 (outRow (P.defs[sc'.d]!) f).2.size - 0 := by
        refine width_charge D (outRow (P.defs[sc'.d]!) f).2 0 _ (writerLand P rs W S at_ r)
          (fun l' _ => writerLand_le hall hstep l') _ fun a ha => ?_
        obtain ⟨l, o, lo, hi', hrl, hla⟩ := mem_landing.1 ha
        obtain ⟨l', g', lo', hi'', hwl, _, _, hl'⟩ := lands_wired_row hO hN hrl hwr hla
        have hod := outLeaf_data (hall l' g' lo' hi'' hwl)
        exact ⟨l', hod.1, Nat.zero_le _, bitAt_leaf_le _ l' hod.1, mem_writerLand.2 ⟨g', lo', hi'', hwl, hl'⟩⟩
      have h2 := hF r wt wu _ sc sc' i f hw h0 hsc hu hi hp rfl hf
      rw [Nat.sub_zero, bitAt_size] at h1
      exact h1.trans h2
    · refine width_none_le D _ _ fun a ha => ?_
      obtain ⟨l, o, lo, hi', hrl, hla⟩ := mem_landing.1 ha
      obtain ⟨l', g', lo', hi'', hwl, _⟩ := lands_wired_row hO hN hrl hwr hla
      exact hne ⟨l', g', lo', hi'', hwl⟩
  · -- the row has no writer: its reads land on its own leaves
    have hown : ∀ l o lo hi', RowLeaf P rs W r l o lo hi' → ¬ Wired P rs W S o := by
      rintro l o lo hi' hrl ⟨r₁, l₁, lo₁, hi₁, hsome, hrl₁⟩
      obtain ⟨e, _⟩ := hO.2 r₁ l₁ lo₁ hi₁ r l lo hi' o hrl₁ hrl
      rw [e] at hsome
      exact hwr hsome
    have h1 : D.width (landing P rs W S at_ r) ≤
        bitAt (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2
          (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.size - 0 := by
      refine width_charge D (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2 0 _
        (leafGates P rs W at_ r) (fun l _ => ?_) _ fun a ha => ?_
      · -- the gates at one leaf are one gate, no wider than the leaf
        refine width_one_le D at_ hL.1 _ _ (fun a ha b hb => ?_) fun a ha => ?_
        · obtain ⟨lo, hi', hrl⟩ := mem_leafGates.1 ha
          obtain ⟨lo₂, hi₂, hrl₂⟩ := mem_leafGates.1 hb
          exact (rowLeaf_fun hN hrl hrl₂).1
        · obtain ⟨lo, hi', hrl⟩ := mem_leafGates.1 ha
          have hd := rowLeaf_data hN hsc hi hp hrl
          obtain ⟨p, ops, sc', w, hg, hret, hle⟩ :=
            hH.1 r l (at_ a) lo hi' hrl (hown l (at_ a) lo hi' hrl)
          have hw := width_of_gate hL hg hret
          have h4 := hd.2.2.2
          omega
      · obtain ⟨l, o, lo, hi', hrl, hla⟩ := mem_landing.1 ha
        have e := lands_unwired (hown l o lo hi' hrl) hla
        have hd := rowLeaf_data hN hsc hi hp hrl
        exact ⟨l, hd.1, Nat.zero_le _, bitAt_leaf_le _ l hd.1, mem_leafGates.2 ⟨lo, hi', e ▸ hrl⟩⟩
    rw [Nat.sub_zero, bitAt_size] at h1
    exact h1

/-- **The landing bound**: the gates a row's reads land on are together no wider than the row. -/
theorem landing_le (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary)
    (D : Verity.Circuit) (at_ : Fin D.N → ℕ) (hO : OwnRows P rs W) (hA : Aligned P rs W S)
    (hL : LaysOut P rs W S D at_) (hW : LeafWide P rs W) (hR : ReturnsOwn P rs W) (hU : UnitsComputed P rs)
    (hN : PortsNamed P rs) (hF : OutFits P rs W S) (ρ : String × ℤ → ℕ) (hρ : Ranked S ρ) (r : Row) (sc : Scope)
    (hsc : UnitAt P rs r.tid r.unit sc) (i : ℕ) (hi : i < (P.defs[sc.d]!).params.size)
    (hp : (P.defs[sc.d]!).params[i]!.1 = r.port) :
    D.width (landing P rs W S at_ r) ≤
      (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.foldl (· + ·) 0 := by
  exact landing_le_of_holds P rs W S D at_ hO hA hL (leafHolds_of_leafWide hW) hR hU hN hF ρ hρ r sc hsc i hi hp

end NciCheck.Properties.Record.SecurityProofs
