module

public import Proofs.Properties.InferenceOnly.RecordGates
public import Proofs.Properties.InferenceOnly.RecordLanding
public import Proofs.Properties.InferenceOnly.RecordTakes

/-! A gate a block reads from outside itself that θ does not hold is one a read of a row of the block lands on. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open Spot.RecordText (Row)

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
  have hland : ∀ tid un sc g, UnitAt P rs tid un sc → sc.off ≤ g → g < sc.off + (P.defs[sc.d]!).gates →
      ∀ x : Fin D.N, Lands P rs W S g (at_ x) → at_ x = g := by
    intro tid un sc g hu h1 h2 x hla
    refine lands_unwired ?_ hla
    rintro ⟨r, l, lo, hi, -, hrl⟩
    obtain ⟨p, ops, sc', hg, hinp⟩ := hO.1 r l g lo hi hrl
    have hc := hU tid un sc g p ops sc' hu h1 h2 hg
    rw [hc] at hinp
    exact Bool.false_ne_true hinp
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
  have key : ∀ k (r : Row) l o lo hi, ρ (r.tid, (r.unit : ℤ)) = k → RowLeaf P rs W r l o lo hi →
      slot r.tid r.unit = some u.val → Lands P rs W S o (at_ a) →
      ∃ r', slot r'.tid r'.unit = some u.val ∧ ¬ inside r' ∧ held r' = false ∧
        a ∈ landing P rs W S at_ r' := by
    intro k
    induction k using Nat.strong_induction_on with
    | _ k ih =>
      intro r l o lo hi hk hrl hs hla
      by_cases hins : inside r
      ·
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
        ·
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

end InferenceOnlyCheck.Properties.Record.SecurityProofs
