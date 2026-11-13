module

public import Proofs.Properties.Nci.RecordGates
public import Proofs.Properties.Nci.RecordLanding
public import Proofs.Properties.Nci.RecordTakes

/-!
# Where a block's reads land

A gate that a block reads from outside itself, and that θ does not hold, is a gate on which a read of some row of
the block lands, a row that is neither inside the block nor held (`reads_land`).

The gate `a` is read by a gate `g` of the block, which lies in a unit of the block's slot (`Divides`). The operand
of `g` that lands on `a` is an earlier gate of the unit or a leaf of one of the unit's rows (`OperandsOwn`). An
earlier gate of the unit is computed, so no wire covers it and the read lands on the gate itself, a gate of the
block. A leaf of a row, by the row:

* a row that is held has no writer, so the read lands on the leaf itself, a gate θ holds;
* a row inside the block has a writer in the block's slot, and the read lands where a leaf of the writer's output
  row lands (`lands_wired_row`). That leaf is a gate of the writer's unit, a gate of the block, or a leaf of one of
  the writer's rows (`ReturnsOwn`), and the writer ranks below the reader (`Ranked`): the same again, by induction
  on the rank;
* any other row is the row asked for.

`unit_of_gate`: a gate lies in one unit, of a Program the record check reads.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row)

/-- **A gate lies in one unit**: two units that hold one gate are one unit of one template. -/
theorem unit_of_gate {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid tid' : String}
    {u u' : ℕ} {sc sc' : Scope} (hu : UnitAt P rs tid u sc) (hu' : UnitAt P rs tid' u' sc') {g : ℕ}
    (h1 : sc.off ≤ g) (h2 : g < sc.off + (P.defs[sc.d]!).gates) (h1' : sc'.off ≤ g)
    (h2' : g < sc'.off + (P.defs[sc'.d]!).gates) : tid = tid' ∧ u = u' := by
  obtain ⟨i, m, hi, ht, hm, hue, rfl⟩ := hu
  obtain ⟨i', m', hi', ht', hm', hue', rfl⟩ := hu'
  by_cases hne : i ≠ i' ∨ m ≠ m'
  · rcases units_apart hP hrs hi hi' hm hm' hne with h | h <;> omega
  · have hii : i = i' := by
      by_contra h
      exact hne (Or.inl h)
    have hmm : m = m' := by
      by_contra h
      exact hne (Or.inr h)
    subst hii hmm
    subst ht
    subst ht'
    exact ⟨rfl, hue.trans hue'.symm⟩

variable {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary} {D : Verity.Circuit}
  {at_ : Fin D.N → ℕ}

/-- **Where a block's reads land.** A gate `a` that block `u` reads from outside itself and θ does not hold is a
gate on which a read of a row lands, the row one of a unit of slot `u`, not inside the block (`inside`) and not
held. `inside` is any set of rows each of which has a writer, the writer in the row's slot where both have one;
every unit has a slot, a held row has no writer, and a gate lies in one unit. -/
theorem reads_land (hO : OwnRows P rs W) (hL : LaysOut P rs W S D at_) (hU : UnitsComputed P rs)
    (hOp : OperandsOwn P rs W) (hR : ReturnsOwn P rs W) (hN : PortsNamed P rs) {ρ : String × ℤ → ℕ}
    (hρ : Ranked S ρ)
    (hA : ∀ tid u sc tid' u' sc' g, UnitAt P rs tid u sc → UnitAt P rs tid' u' sc' → sc.off ≤ g →
      g < sc.off + (P.defs[sc.d]!).gates → sc'.off ≤ g → g < sc'.off + (P.defs[sc'.d]!).gates →
      tid = tid' ∧ u = u')
    {slot : String → ℕ → Option ℕ} {n : ℕ} {U : Verity.Partition D n} (hD : Divides P rs slot at_ U)
    (hslot : ∀ tid u sc, UnitAt P rs tid u sc → ∃ s, slot tid u = some s)
    {held : Row → Bool} {θ : Finset (Fin D.N)} (hNm : Names P rs W held at_ θ)
    (hheld : ∀ r, held r = true → S.writerOf r.tid r.unit r.port r.row = none) (inside : Row → Prop)
    (hin : ∀ r, inside r → (S.writerOf r.tid r.unit r.port r.row).isSome ∧ ∀ wt wu out s s',
      S.writerOf r.tid r.unit r.port r.row = some (wt, wu, out) → slot r.tid r.unit = some s →
      slot wt wu.toNat = some s' → s' = s)
    (u : Fin n) (a : Fin D.N) (ha : a ∈ D.into (U.gates u)) (hθ : a ∉ θ) :
    ∃ r, slot r.tid r.unit = some u.val ∧ ¬ inside r ∧ held r = false ∧ a ∈ landing P rs W S at_ r := by
  -- a read of a gate inside a unit lands on the gate itself
  have hland : ∀ tid un sc g, UnitAt P rs tid un sc → sc.off ≤ g → g < sc.off + (P.defs[sc.d]!).gates →
      ∀ x : Fin D.N, Lands P rs W S g (at_ x) → at_ x = g := by
    intro tid un sc g hu h1 h2 x hla
    refine lands_unwired ?_ hla
    rintro ⟨r, l, lo, hi, -, hrl⟩
    obtain ⟨p, ops, sc', hg, hinp⟩ := hO.1 r l g lo hi hrl
    have hc := hU tid un sc g p ops sc' hu h1 h2 hg
    rw [hc] at hinp
    exact Bool.false_ne_true hinp
  -- a gate of `D` inside a unit of slot `s` is in block `s`
  have hblock : ∀ tid un sc, UnitAt P rs tid un sc → ∀ s, slot tid un = some s → ∀ x : Fin D.N,
      sc.off ≤ at_ x → at_ x < sc.off + (P.defs[sc.d]!).gates → (U.unit x).map Fin.val = some s := by
    intro tid un sc hu s hs x h1 h2
    obtain ⟨p, ops, sc', hg, -, hinp, -⟩ := hL.2.2 x
    have hc : D.isInput x = false := by
      rw [hinp]
      exact hU tid un sc _ p ops sc' hu h1 h2 hg
    obtain ⟨tid', u', sc'', s', hu', h1', h2', hs', hx⟩ := hD x hc
    obtain ⟨e1, e2⟩ := hA tid un sc tid' u' sc'' (at_ x) hu hu' h1 h2 h1' h2'
    rw [← e1, ← e2, hs] at hs'
    rw [hx, ← hs']
  have hmem : ∀ x : Fin D.N, (U.unit x).map Fin.val = some u.val → x ∈ U.gates u := by
    intro x hx
    unfold Verity.Partition.gates
    refine Finset.mem_filter.2 ⟨Finset.mem_univ _, ?_⟩
    cases hv : U.unit x with
    | none =>
      rw [hv] at hx
      cases hx
    | some v =>
      rw [hv] at hx
      have hx' : some v.val = some u.val := hx
      exact congrArg some (Fin.ext (Option.some.inj hx'))
  obtain ⟨hnot, g, hg, hread⟩ : a ∉ U.gates u ∧ ∃ g ∈ U.gates u, a ∈ D.reads g := by
    unfold Verity.Circuit.into at ha
    exact (Finset.mem_filter.1 ha).2
  -- the rows, by the rank of their unit
  have key : ∀ k (r : Row) l o lo hi, ρ (r.tid, (r.unit : ℤ)) = k → RowLeaf P rs W r l o lo hi →
      slot r.tid r.unit = some u.val → Lands P rs W S o (at_ a) →
      ∃ r', slot r'.tid r'.unit = some u.val ∧ ¬ inside r' ∧ held r' = false ∧
        a ∈ landing P rs W S at_ r' := by
    intro k
    induction k using Nat.strong_induction_on with
    | _ k ih =>
      intro r l o lo hi hk hrl hs hla
      by_cases hins : inside r
      · -- a row inside the block: the read lands where a leaf of its writer's output row lands
        obtain ⟨hsome, hsame⟩ := hin r hins
        obtain ⟨l', g', lo', hi', hwl, -, -, hla'⟩ := lands_wired_row hO hN hrl hsome hla
        obtain ⟨wt, wu, out, sc', hw, h0, hu', hol⟩ := hwl
        obtain ⟨s', hs'⟩ := hslot wt wu.toNat sc' hu'
        have hss : s' = u.val := hsame wt wu out _ _ hw hs hs'
        rw [hss] at hs'
        rcases hR wt wu.toNat sc' out l' g' lo' hi' hu' hol with ⟨h1, h2⟩ | ⟨r', l'', lo'', hi'', ht, hun, hrl'⟩
        · exfalso
          have e := hland wt wu.toNat sc' g' hu' h1 h2 a hla'
          exact hnot (hmem a (hblock wt wu.toNat sc' hu' u.val hs' a (by omega) (by omega)))
        · have hlt : ρ (r'.tid, (r'.unit : ℤ)) < k := by
            rw [← hk, ht, hun, Int.toNat_of_nonneg h0]
            exact hρ r wt wu out hw
          exact ih _ hlt r' l'' g' lo'' hi'' rfl hrl' (by rw [ht, hun]; exact hs') hla'
      · by_cases hh : held r = true
        · -- a held row: the read lands on the leaf itself, which θ holds
          exfalso
          have hnone := hheld r hh
          have hnw : ¬ Wired P rs W S o := by
            rintro ⟨r₁, l₁, lo₁, hi₁, hsome, hrl₁⟩
            obtain ⟨e, -⟩ := hO.2 r₁ l₁ lo₁ hi₁ r l lo hi o hrl₁ hrl
            rw [e, hnone] at hsome
            cases hsome
          have e := lands_unwired hnw hla
          rw [← e] at hrl
          exact hθ ((hNm a).2 ⟨r, l, lo, hi, hh, hrl⟩)
        · exact ⟨r, hs, hins, Bool.eq_false_iff.2 hh, mem_landing.2 ⟨l, o, lo, hi, hrl, hla⟩⟩
  -- the gate that reads `a`, in a unit of the slot
  have hgu : U.unit g = some u := by
    unfold Verity.Partition.gates at hg
    exact (Finset.mem_filter.1 hg).2
  have hc : D.isInput g = false := by
    cases hb : D.isInput g with
    | false => rfl
    | true =>
      have hn := (U.input_iff g).2 hb
      rw [hgu] at hn
      cases hn
  obtain ⟨tid, un, sc, s, hu, h1, h2, hs, hx⟩ := hD g hc
  have hsu : s = u.val := by
    rw [hgu] at hx
    have hx' : some u.val = some s := hx
    exact (Option.some.inj hx').symm
  rw [hsu] at hs
  obtain ⟨p, ops, sc', hgate, -, -, hreads⟩ := hL.2.2 g
  obtain ⟨o, ho, hla⟩ := (hreads a).1 hread
  rcases hOp tid un sc (at_ g) p ops sc' hu h1 h2 hgate o ho with ⟨h3, h4⟩ | ⟨r, l, lo, hi, ht, hun, hrl⟩
  · exfalso
    have e := hland tid un sc o hu h3 (by omega) a hla
    exact hnot (hmem a (hblock tid un sc hu u.val hs a (by omega) (by omega)))
  · exact key _ r l o lo hi rfl hrl (by rw [ht, hun]; exact hs) hla

end NciCheck.Properties.Record.SecurityProofs
