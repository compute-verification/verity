module

public import Proofs.Properties.Nci.RecordTakes
public import Proofs.Properties.Nci.RecordGrid
public import Proofs.Properties.Nci.RecordLanding
public import Mathlib.Tactic

/-!
# What the landing bound reads of the widths

`LeafWide` says a gate is as wide as the leaf that names it, for every leaf of a reader row and of an output row.
The landing bound reads less, `LeafHolds`: three inequalities, two of them only at gates no wire covers.

* `leafHolds_of_leafWide`: `LeafWide` gives `LeafHolds`.
* `landing_le_of_holds`: the landing bound with `LeafHolds` in place of `LeafWide`, its other hypotheses those of
  `landing_le`, which follows from it by `leafHolds_of_leafWide`.

The proof is `landing_le`'s: a gate no wire covers lands on itself, one gate of `D` no wider than its leaf; a leaf
of an output row that is a leaf of a row with a writer lands inside what the writer's leaves within it land on, and
has no more bits as the row's leaf than as the output's.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref Ty Defn)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row Grid fieldsOf outerOf)
open NciCheck.Record (inSlot)
open NciCheck.Record.SecurityProofs (slotOf OneGrid)
open Spot.Bits (bitsOf)

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

variable {D : Verity.Circuit} {at_ : Fin D.N → ℕ}

/-- A gate of `D` is as wide as the value of the Program gate it is. -/
theorem width_of_gate (hL : LaysOut P rs W S D at_) {a : Fin D.N} {p : ℕ} {ops : Array ℕ} {sc : Scope} {w : ℕ}
    (hg : P.gate (at_ a) = .ok (p, ops, sc)) (hret : (P.defs[p]!).ret = .value w) : D.w a = w := by
  obtain ⟨p', ops', sc', hg', hret', _⟩ := hL.2.2 a
  rw [hg] at hg'
  have hp : p = p' := (Prod.mk.inj (Except.ok.inj hg')).1
  subst hp
  rw [hret] at hret'
  exact (Ty.value.inj hret').symm

/-- **The bound for an output leaf**, from `LeafHolds`: a read of a leaf of a unit's output row lands on gates
together no wider than the leaf. -/
theorem outLeaf_land_le_of_holds (hO : OwnRows P rs W) (hA : Aligned P rs W S) (hL : LaysOut P rs W S D at_)
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
        outLeaf_land_le_of_holds hO hA hL hH hR hU hN ρ hρ _ wt wu.toNat sc' _ l₁ g₁ lo₁ hi₁ rfl hu
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

/-- `landing_le` is `landing_le_of_holds` at what `LeafWide` gives. -/
example (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary)
    (D : Verity.Circuit) (at_ : Fin D.N → ℕ) (hO : OwnRows P rs W) (hA : Aligned P rs W S)
    (hL : LaysOut P rs W S D at_) (hW : LeafWide P rs W) (hR : ReturnsOwn P rs W) (hU : UnitsComputed P rs)
    (hN : PortsNamed P rs) (hF : OutFits P rs W S) (ρ : String × ℤ → ℕ) (hρ : Ranked S ρ) (r : Row) (sc : Scope)
    (hsc : UnitAt P rs r.tid r.unit sc) (i : ℕ) (hi : i < (P.defs[sc.d]!).params.size)
    (hp : (P.defs[sc.d]!).params[i]!.1 = r.port) :
    D.width (landing P rs W S at_ r) ≤
      (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.foldl (· + ·) 0 :=
  landing_le_of_holds P rs W S D at_ hO hA hL (leafHolds_of_leafWide hW) hR hU hN hF ρ hρ r sc hsc i hi hp

end NciCheck.Properties.Record.SecurityProofs
