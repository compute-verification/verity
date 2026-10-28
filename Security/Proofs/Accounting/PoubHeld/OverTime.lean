import Proofs.Accounting.PoubHeld
import Mathlib.Combinatorics.Enumerative.DoubleCounting
import Mathlib.Order.Lattice.Nat
import Mathlib.Tactic.Ring

/-!
# The time a PoUB shows held

`Accounting.Guarantees.Poub.HeldOverTimeLinks` from its premises, and `HeldOverTime` from it at links of rate 0
(`heldOverTime_of_links`). At each outcome where the sketch bound and the live coins' bound hold (`holdsAnd`), each tick
of the credited span `[e_0, s_C)` has its `N` registered blocks in node memory, on a chip (at most `S`) or absent, and
the absent ticks are bounded block by block over `[s_0, s_C) ⊇ [e_0, s_C)` (`Absence.absent_le`):

- a group `g < C` whose window `[s_g, e_g)` the block is absent from throughout charges its segment `[s_g, s_(g+1))`,
  and the closing group `L`;
- an absent tick with a present one less than `L` after it is one of at most `L` before an entry (a tick the block
  comes back at) over `[s_0, s_C + L)` (`Absence.head_le`, by double counting);
- the rest are tails, from a tick to its group's window's end: at most `L` each, one a group, and each group with a
  later present tick has its own entry (`Absence.firstFrom`), so `L` an entry, and one tail more only if the closing
  group's window is absent throughout.

Every pass of a group whose window the block is absent from throughout reads it (`Covers`) and so reads it wrong
(`SketchBound`), so the groups' charges are a point of `fill`'s knapsack within `W` (`LiveCoins`). The entries are what
rebuilds, disk reads and links bring in over `[s_0, s_C + L)` (`Routed`), one interval of at most `Σ + 2·L` ticks: at
most `(s/E + β + βl)·(Σ + 2·L) + (Br + Bd + Bl)` by the token buckets.
-/

namespace Accounting.SecurityProofs.Poub.Absence

open Finset

/-- An absent tick before a present one: some tick between is an entry. -/
theorem exists_entry (pr : ℕ → Prop) {a b : ℕ} (hab : a < b) (ha : ¬ pr a) (hb : pr b) :
    ∃ τ, a ≤ τ ∧ τ < b ∧ ¬ pr τ ∧ pr (τ + 1) := by
  induction b, hab using Nat.le_induction with
  | base => exact ⟨a, le_rfl, Nat.lt_succ_self a, ha, hb⟩
  | succ b hab ih =>
    by_cases hpb : pr b
    · obtain ⟨τ, h1, h2, h3, h4⟩ := ih hpb
      exact ⟨τ, h1, by omega, h3, h4⟩
    · exact ⟨b, by omega, Nat.lt_succ_self b, hpb, hb⟩

/-- The ticks of `[u, v)` at which `pr` turns on: false at `τ`, true at `τ + 1`. -/
def entries (pr : ℕ → Prop) [DecidablePred pr] (u v : ℕ) : Finset ℕ :=
  (Ico u v).filter fun τ => ¬ pr τ ∧ pr (τ + 1)

/-- The absent ticks of `[u, v)` with a present one less than `L` after them: at most `L` for each entry of
`[u, v + L)`, the entry that ends each one's absence lying less than `L` after it. -/
theorem head_le (pr : ℕ → Prop) [DecidablePred pr] (u v L : ℕ) :
    ((Ico u v).filter fun t => ¬ pr t ∧ ∃ w ∈ Ioo t (t + L), pr w).card ≤
      L * (entries pr u (v + L)).card := by
  classical
  let r : ℕ → ℕ → Prop := fun t τ => t ≤ τ ∧ τ < t + L
  calc ((Ico u v).filter fun t => ¬ pr t ∧ ∃ w ∈ Ioo t (t + L), pr w).card
      = ∑ t ∈ (Ico u v).filter fun t => ¬ pr t ∧ ∃ w ∈ Ioo t (t + L), pr w, 1 := card_eq_sum_ones _
    _ ≤ ∑ t ∈ (Ico u v).filter fun t => ¬ pr t ∧ ∃ w ∈ Ioo t (t + L), pr w,
          (bipartiteAbove r (entries pr u (v + L)) t).card := by
        refine sum_le_sum fun t ht => ?_
        obtain ⟨htI, hn, w, hwI, hw⟩ := mem_filter.1 ht
        rw [mem_Ico] at htI
        rw [mem_Ioo] at hwI
        obtain ⟨τ, h1, h2, h3, h4⟩ := exists_entry pr hwI.1 hn hw
        exact card_pos.2 ⟨τ, (mem_bipartiteAbove r).2
          ⟨mem_filter.2 ⟨mem_Ico.2 ⟨by omega, by omega⟩, h3, h4⟩, h1, by omega⟩⟩
    _ = ∑ τ ∈ entries pr u (v + L),
          (bipartiteBelow r ((Ico u v).filter fun t => ¬ pr t ∧ ∃ w ∈ Ioo t (t + L), pr w) τ).card :=
        sum_card_bipartiteAbove_eq_sum_card_bipartiteBelow r
    _ ≤ ∑ τ ∈ entries pr u (v + L), L := by
        refine sum_le_sum fun τ _ => ?_
        calc _ ≤ (Ico (τ + 1 - L) (τ + 1)).card := by
              refine card_le_card fun t ht => ?_
              obtain ⟨-, h1, h2⟩ := (mem_bipartiteBelow r).1 ht
              exact mem_Ico.2 ⟨by omega, by omega⟩
          _ ≤ L := by rw [Nat.card_Ico]; omega
    _ = L * (entries pr u (v + L)).card := by rw [sum_const, smul_eq_mul, mul_comm]

/-- An absent tick of a group's segment with no present one less than `L` after it: from it to the group's
window's end, nothing is present. -/
theorem tail_absent (pr : ℕ → Prop) {L sg eg t : ℕ} (hL : eg ≤ sg + L) (ht : sg ≤ t) (hn : ¬ pr t)
    (hnear : ¬ ∃ w ∈ Ioo t (t + L), pr w) : ∀ x ∈ Ico t eg, ¬ pr x := by
  intro x hx hp
  rw [mem_Ico] at hx
  rcases Nat.eq_or_lt_of_le hx.1 with h | h
  · exact hn (h ▸ hp)
  · exact hnear ⟨x, mem_Ioo.2 ⟨h, by omega⟩, hp⟩

/-- One segment `[sg, sg1)` of a group whose window is `[sg, eg)`: its absent ticks with nothing present less than
`L` after them are none, its length if nothing of the window is present, or at most `L` if the group has a tail
(nothing present from its segment's last tick to its window's end). -/
theorem seg_le (pr : ℕ → Prop) [DecidablePred pr] {L sg sg1 eg : ℕ} (hse : sg1 ≤ eg) (hL : eg ≤ sg + L) :
    ((Ico sg sg1).filter fun t => ¬ pr t ∧ ¬ ∃ w ∈ Ioo t (t + L), pr w).card ≤
      (if ∃ w ∈ Ico sg eg, pr w then 0 else sg1 - sg) +
      (if (∃ w ∈ Ico sg eg, pr w) ∧ sg < sg1 ∧ ∀ x ∈ Ico (sg1 - (1 : ℕ)) eg, ¬ pr x then L else 0) := by
  have hc : ((Ico sg sg1).filter fun t => ¬ pr t ∧ ¬ ∃ w ∈ Ioo t (t + L), pr w).card ≤ sg1 - sg :=
    (card_filter_le _ _).trans (Nat.card_Ico sg sg1).le
  by_cases hw : ∃ w ∈ Ico sg eg, pr w
  · rw [ite_eq_left hw, zero_add]
    by_cases hg : sg < sg1 ∧ ∀ x ∈ Ico (sg1 - (1 : ℕ)) eg, ¬ pr x
    · rw [ite_eq_left ⟨hw, hg⟩]
      omega
    · rw [ite_eq_right fun h => hg h.2]
      refine (card_eq_zero.2 (filter_eq_empty_iff.2 fun t ht h => hg ?_)).le
      rw [mem_Ico] at ht
      refine ⟨by omega, fun x hx => tail_absent pr hL ht.1 h.1 h.2 x ?_⟩
      rw [mem_Ico] at hx ⊢
      omega
  · rw [ite_eq_right hw, ite_eq_right fun h => hw h.1]
    omega

/-- A group with a tail: its segment's last tick is absent, and every present tick of its window `[sg, eg)` is before
it. -/
theorem tail_facts (pr : ℕ → Prop) {sg sg1 eg : ℕ} (hse : sg1 ≤ eg)
    (hg : sg < sg1 ∧ ∀ x ∈ Ico (sg1 - (1 : ℕ)) eg, ¬ pr x) :
    sg ≤ sg1 - 1 ∧ ¬ pr (sg1 - 1) ∧ ∀ w ∈ Ico sg eg, pr w → w < sg1 - 1 := by
  obtain ⟨hlt, hx⟩ := hg
  refine ⟨by omega, hx _ (mem_Ico.2 ⟨le_rfl, by omega⟩), fun w hw hp => ?_⟩
  rw [mem_Ico] at hw
  by_contra h
  exact hx w (mem_Ico.2 ⟨by omega, hw.2⟩) hp

/-- The first element of `En` at or after `a`. -/
noncomputable def firstFrom (En : Finset ℕ) (a : ℕ) : ℕ := sInf {τ | a ≤ τ ∧ τ ∈ En}

theorem firstFrom_spec {En : Finset ℕ} {a τ : ℕ} (h : a ≤ τ) (hτ : τ ∈ En) :
    a ≤ firstFrom En a ∧ firstFrom En a ∈ En ∧ firstFrom En a ≤ τ :=
  ⟨(Nat.sInf_mem (s := {τ | a ≤ τ ∧ τ ∈ En}) ⟨τ, h, hτ⟩).1,
    (Nat.sInf_mem (s := {τ | a ≤ τ ∧ τ ∈ En}) ⟨τ, h, hτ⟩).2, Nat.sInf_le ⟨h, hτ⟩⟩

theorem sum_if_const {ι : Type*} (s : Finset ι) (P : ι → Prop) [DecidablePred P] (L : ℕ) :
    ∑ g ∈ s, (if P g then L else 0) = (s.filter P).card * L := by
  rw [sum_ite, sum_const_zero, sum_const, smul_eq_mul, add_zero]

/-- **One block's absence.** Groups `g ≤ C` with windows `[s g, e g)`, `s` monotone, each window at most `L` long and
reaching the next group's start; `pr t` that the block is present at tick `t`. The block's absent ticks over
`[s 0, s C)` are at most the segments `[s g, s (g + 1))` of the groups `g < C` whose windows it is absent from
throughout, `L` if it is absent from the closing group's window throughout, and `2·L` for each entry over
`[s 0, s C + L)`. -/
theorem absent_le (pr : ℕ → Prop) [DecidablePred pr] (C L : ℕ) (s e : ℕ → ℕ) (hs : Monotone s)
    (hse : ∀ g < C, s (g + 1) ≤ e g) (hL : ∀ g ≤ C, e g ≤ s g + L) :
    ((Ico (s 0) (s C)).filter fun t => ¬ pr t).card ≤
      ∑ g ∈ range C, (if ∃ w ∈ Ico (s g) (e g), pr w then 0 else s (g + 1) - s g) +
      (if ∃ w ∈ Ico (s C) (e C), pr w then 0 else L) + 2 * L * (entries pr (s 0) (s C + L)).card := by
  -- the absent ticks: a present one less than `L` after them, or none
  have hU : ((Ico (s 0) (s C)).filter fun t => ¬ pr t).card ≤
      ((Ico (s 0) (s C)).filter fun t => ¬ pr t ∧ ∃ w ∈ Ioo t (t + L), pr w).card +
      ((Ico (s 0) (s C)).filter fun t => ¬ pr t ∧ ¬ ∃ w ∈ Ioo t (t + L), pr w).card := by
    refine (card_le_card fun t ht => ?_).trans (card_union_le _ _)
    obtain ⟨h1, h2⟩ := mem_filter.1 ht
    by_cases h : ∃ w ∈ Ioo t (t + L), pr w
    · exact mem_union_left _ (mem_filter.2 ⟨h1, h2, h⟩)
    · exact mem_union_right _ (mem_filter.2 ⟨h1, h2, h⟩)
  have hH := head_le pr (s 0) (s C) L
  -- the rest, segment by segment
  have hTl : ((Ico (s 0) (s C)).filter fun t => ¬ pr t ∧ ¬ ∃ w ∈ Ioo t (t + L), pr w).card =
      ∑ g ∈ range C, ((Ico (s g) (s (g + 1))).filter fun t => ¬ pr t ∧ ¬ ∃ w ∈ Ioo t (t + L), pr w).card := by
    simp only [card_filter]
    exact (sum_passes _ s hs C).symm
  have hseg := sum_le_sum fun g (hg : g ∈ range C) =>
    seg_le pr (hse g (mem_range.1 hg)) (hL g (mem_range.1 hg).le)
  rw [sum_add_distrib, sum_if_const] at hseg
  -- the groups with a tail
  set Sg := (range C).filter fun g =>
    (∃ w ∈ Ico (s g) (e g), pr w) ∧ s g < s (g + 1) ∧ ∀ x ∈ Ico (s (g + 1) - (1 : ℕ)) (e g), ¬ pr x with hSg
  set En := entries pr (s 0) (s C + L) with hEn
  have hsplit := card_filter_add_card_filter_not (s := Sg)
    (fun g => ∃ h ∈ Ioc g C, ∃ w ∈ Ico (s h) (e h), pr w)
  -- with no later witnessed group: at most one, and none if the closing group is witnessed
  have hlast : (Sg.filter fun g => ¬ ∃ h ∈ Ioc g C, ∃ w ∈ Ico (s h) (e h), pr w).card ≤
      if ∃ w ∈ Ico (s C) (e C), pr w then 0 else 1 := by
    split_ifs with hC
    · refine (card_eq_zero.2 (filter_eq_empty_iff.2 fun g hg h => h ⟨C, ?_, hC⟩)).le
      exact mem_Ioc.2 ⟨mem_range.1 (mem_filter.1 hg).1, le_rfl⟩
    · refine card_le_one.2 fun a ha b hb => ?_
      obtain ⟨ha1, ha2⟩ := mem_filter.1 ha
      obtain ⟨hb1, hb2⟩ := mem_filter.1 hb
      obtain ⟨haC, haw, -⟩ := mem_filter.1 ha1
      obtain ⟨hbC, hbw, -⟩ := mem_filter.1 hb1
      rw [mem_range] at haC hbC
      rcases lt_trichotomy a b with h | h | h
      · exact absurd ⟨b, mem_Ioc.2 ⟨h, hbC.le⟩, hbw⟩ ha2
      · exact h
      · exact absurd ⟨a, mem_Ioc.2 ⟨h, haC.le⟩, haw⟩ hb2
  -- with one: each to the first entry after its segment's last tick, strictly increasing
  have hlater : (Sg.filter fun g => ∃ h ∈ Ioc g C, ∃ w ∈ Ico (s h) (e h), pr w).card ≤ En.card := by
    have hent : ∀ g ∈ Sg, ∀ w, s (g + 1) ≤ w → w < s C + L → pr w →
        ∃ τ, s (g + 1) - 1 ≤ τ ∧ τ < w ∧ τ ∈ En := by
      intro g hg w hw1 hw2 hw
      obtain ⟨hgC, -, hgt⟩ := mem_filter.1 hg
      rw [mem_range] at hgC
      obtain ⟨h1, h3, -⟩ := tail_facts pr (hse g hgC) hgt
      obtain ⟨τ, hτ1, hτ2, hτ3, hτ4⟩ := exists_entry pr (show s (g + 1) - 1 < w by omega) h3 hw
      have h0 : s 0 ≤ s g := hs (Nat.zero_le g)
      exact ⟨τ, hτ1, hτ2, mem_filter.2 ⟨mem_Ico.2 ⟨by omega, by omega⟩, hτ3, hτ4⟩⟩
    refine card_le_card_of_injOn (fun g => firstFrom En (s (g + 1) - 1)) (fun g hg => ?_) ?_
    · obtain ⟨hgS, h, hh, w, hw, hpw⟩ := mem_filter.1 hg
      obtain ⟨hgh, hhC⟩ := mem_Ioc.1 hh
      rw [mem_Ico] at hw
      have := hs (show g + 1 ≤ h by omega)
      have := hs hhC
      have := hL h hhC
      obtain ⟨τ, hτ1, -, hτ⟩ := hent g hgS w (by omega) (by omega) hpw
      exact (firstFrom_spec hτ1 hτ).2.1
    · refine StrictMonoOn.injOn fun g hg g' hg' hlt => ?_
      obtain ⟨hgS, h, hh, w, hw, hpw⟩ := mem_filter.1 (mem_coe.1 hg)
      obtain ⟨hg'S, h', hh', w', hw', hpw'⟩ := mem_filter.1 (mem_coe.1 hg')
      obtain ⟨hg'C, hg'w, hg't⟩ := mem_filter.1 hg'S
      rw [mem_range] at hg'C
      obtain ⟨-, -, hbefore⟩ := tail_facts pr (hse g' hg'C) hg't
      obtain ⟨v, hv, hpv⟩ := hg'w
      have hv' := hbefore v hv hpv
      rw [mem_Ico] at hv
      have := hs (show g + 1 ≤ g' by omega)
      have := hs hg'C.le
      have := hL g' hg'C.le
      obtain ⟨τ, hτ1, hτ2, hτ⟩ := hent g hgS v (by omega) (by omega) hpv
      -- the first entry after `g'`'s segment's last tick
      obtain ⟨hhg', hh'C⟩ := mem_Ioc.1 hh'
      rw [mem_Ico] at hw'
      have := hs (show g' + 1 ≤ h' by omega)
      have := hs hh'C
      have := hL h' hh'C
      obtain ⟨τ', hτ'1, -, hτ'⟩ := hent g' hg'S w' (by omega) (by omega) hpw'
      have h1 := (firstFrom_spec hτ1 hτ).2.2
      have h2 := (firstFrom_spec hτ'1 hτ').1
      show firstFrom En (s (g + 1) - 1) < firstFrom En (s (g' + 1) - 1)
      omega
  have hlast' : (Sg.filter fun g => ¬ ∃ h ∈ Ioc g C, ∃ w ∈ Ico (s h) (e h), pr w).card * L ≤
      if ∃ w ∈ Ico (s C) (e C), pr w then 0 else L := by
    split_ifs at hlast ⊢ with hC
    · rw [Nat.le_zero.1 hlast, zero_mul]
    · exact (Nat.mul_le_mul_right L hlast).trans (one_mul L).le
  have hlater' := Nat.mul_le_mul_right L hlater
  rw [← hsplit, add_mul] at hseg
  rw [hTl] at hU
  calc _ ≤ _ := hU
    _ ≤ L * En.card + (∑ g ∈ range C, (if ∃ w ∈ Ico (s g) (e g), pr w then 0 else s (g + 1) - s g) +
          (En.card * L + if ∃ w ∈ Ico (s C) (e C), pr w then 0 else L)) := by
        refine Nat.add_le_add hH (hseg.trans ?_)
        exact Nat.add_le_add_left (Nat.add_le_add hlater' hlast') _
    _ = _ := by ring

end Accounting.SecurityProofs.Poub.Absence

namespace Accounting.SecurityProofs.Poub

open Finset Accounting.Poub.Held Accounting.Poub.Held.Assumptions Pouw Absence

variable {Ω : Type} {T : Trace Ω}

/-- Steps of `f` up to `n`: `f` is monotone on `[0, n]`. -/
theorem le_of_steps (f : ℕ → ℕ) {n : ℕ} (h : ∀ g < n, f g ≤ f (g + 1)) {a b : ℕ} (hab : a ≤ b) (hb : b ≤ n) :
    f a ≤ f b := by
  induction b, hab using Nat.le_induction with
  | base => exact le_rfl
  | succ b hab ih => exact (ih (by omega)).trans (h b (by omega))

/-- `sum_passes` for a sequence that steps up only to `n`. -/
theorem sum_groups (f : ℕ → ℕ) (t : ℕ → ℕ) {n : ℕ} (ht : ∀ g < n, t g ≤ t (g + 1)) :
    ∑ g ∈ range n, ∑ τ ∈ Ico (t g) (t (g + 1)), f τ = ∑ τ ∈ Ico (t 0) (t n), f τ := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_range_succ, ih fun g hg => ht g (by omega),
      sum_Ico_consecutive _ (le_of_steps t ht (Nat.zero_le n) (Nat.le_succ n)) (ht n (Nat.lt_succ_self n))]

theorem div_mul_cancel_mid {a n k : ℝ} (hn : n ≠ 0) : a / n * (n * k) = k * a := by
  rw [div_mul_eq_mul_div, mul_left_comm, mul_div_cancel_left₀ _ hn, mul_comm]

/-- `HeldOverTimeLinks` at one outcome. -/
theorem heldOverLinks_of [Fintype Ω] (K : Pace Ω) (ω : Ω) {S : ℕ} {s E β βl Br Bd Bl W : ℝ} (hE : 0 < E)
    (hs : 0 ≤ s) (hβ : 0 ≤ β) (hβl : 0 ≤ βl) (hl : Links T βl Bl) (hsq : SquaringCeiling T s E Br)
    (hd : Disks T β Bd) (hS : ChipCeiling T S) (hr : Routed T) (hp : Paced T K) (hcov : Covers T)
    (ha : T.accepts ω)
    (hk : ∀ p < T.passes, ∀ j ∈ T.correct ω p,
      ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t)
    (hw : (T.wrong ω : ℝ) ≤ W) :
    ((K.stop ω - K.start ω : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω *
              ((s / E + β + βl) * (((K.stop ω - K.start ω : ℕ) : ℝ) + 2 * K.longest ω) + (Br + Bd + Bl))
          - K.fill ω T.blocks W
        ≤ T.heldOver ω (K.start ω) (K.stop ω) := by
  obtain ⟨-, hf0, hfP, hfg, hrel, hre, hpass⟩ := hp ω ha
  have hplan := hcov ω ha
  rw [show K.stop ω = K.released ω K.credited from rfl, show K.start ω = K.ended ω 0 from rfl]
  set C := K.credited with hCdef
  set L := K.longest ω with hLdef
  set rel := K.released ω with hreldef
  set en := K.ended ω with hendef
  set N := T.blocks with hNdef
  have hL : ∀ g ≤ C, en g ≤ rel g + L := by
    intro g hg
    have h1 : en g - rel g ≤ L :=
      le_sup (f := fun g => K.ended ω g - K.released ω g) (mem_range.2 (Nat.lt_succ_of_le hg))
    have h2 := (hrel (Nat.le_succ g)).trans (hre g hg)
    omega
  have hse : ∀ g < C, rel (g + 1) ≤ en g := fun g hg => hre g hg.le
  have h01 : rel 0 ≤ en 0 := (hrel (Nat.zero_le 1)).trans (hre 0 (Nat.zero_le C))
  have h0C : rel 0 ≤ rel C := hrel (Nat.zero_le C)
  -- each block's absence
  have hsum := sum_le_sum fun j (_ : j ∈ (univ : Finset (Fin N))) =>
    absent_le (fun t => j ∈ T.present ω t) C L rel en hrel hse hL
  rw [sum_add_distrib, sum_add_distrib, ← mul_sum] at hsum
  -- the entries, summed over the blocks
  have hent : ∑ j : Fin N, (entries (fun t => j ∈ T.present ω t) (rel 0) (rel C + L)).card =
      ∑ τ ∈ Ico (rel 0) (rel C + L), T.entered ω τ := by
    simp only [entries, card_filter]
    rw [sum_comm]
    refine sum_congr rfl fun τ _ => ?_
    have : T.present ω (τ + 1) \ T.present ω τ =
        univ.filter fun j => ¬ j ∈ T.present ω τ ∧ j ∈ T.present ω (τ + 1) := by
      ext j
      simp [and_comm]
    rw [Trace.entered, this, card_filter]
  have hin : ∑ τ ∈ Ico (rel 0) (rel C + L), T.entered ω τ ≤
      over T.rebuilt ω (rel 0) (rel C + L) + over T.disk ω (rel 0) (rel C + L) +
        over T.link ω (rel 0) (rel C + L) := by
    have h := sum_le_sum fun τ (_ : τ ∈ Ico (rel 0) (rel C + L)) => hr ω τ
    simp only [sum_add_distrib] at h
    exact h
  -- the groups whose windows a block is absent from throughout
  set nw : ℕ → ℕ := fun g =>
    (univ.filter fun j : Fin N => ¬ ∃ w ∈ Ico (rel g) (en g), j ∈ T.present ω w).card with hnw
  have hcost1 : ∑ j : Fin N, ∑ g ∈ range C,
      (if ∃ w ∈ Ico (rel g) (en g), j ∈ T.present ω w then 0 else rel (g + 1) - rel g) =
      ∑ g ∈ range C, nw g * (rel (g + 1) - rel g) := by
    rw [sum_comm]
    refine sum_congr rfl fun g _ => ?_
    rw [sum_ite, sum_const_zero, zero_add, sum_const, smul_eq_mul]
  have hcost2 : ∑ j : Fin N, (if ∃ w ∈ Ico (rel C) (en C), j ∈ T.present ω w then 0 else L) = nw C * L := by
    rw [sum_ite, sum_const_zero, zero_add, sum_const, smul_eq_mul]
  rw [hcost1, hcost2, hent] at hsum
  -- their wrong receipts
  have hnw_le : ∀ g ≤ C, ∀ p, K.first ω g ≤ p → p < K.first ω (g + 1) →
      nw g ≤ (T.plan ω p \ T.correct ω p).card := by
    intro g hg p hp1 hp2
    have hpP : p < T.passes :=
      lt_of_lt_of_le hp2 ((le_of_steps (K.first ω) (fun g hg => (hfg g (by omega)).le) (by omega) le_rfl).trans hfP)
    refine card_le_card fun j hj => ?_
    have hj' := (mem_filter.1 hj).2
    rw [hplan p hpP]
    refine mem_sdiff.2 ⟨mem_univ j, fun hc => hj' ?_⟩
    obtain ⟨t, ht1, ht2, ht⟩ := hk p hpP j hc
    obtain ⟨hp3, hp4⟩ := hpass g hg p hp1 hp2
    exact ⟨t, mem_Ico.2 ⟨by omega, by omega⟩, ht⟩
  have hwrong : ∑ g ∈ range (C + 1), K.size ω g * nw g ≤ T.wrong ω := by
    calc ∑ g ∈ range (C + 1), K.size ω g * nw g
        ≤ ∑ g ∈ range (C + 1), ∑ p ∈ Ico (K.first ω g) (K.first ω (g + 1)), (T.plan ω p \ T.correct ω p).card := by
          refine sum_le_sum fun g hg => ?_
          rw [Pace.size, ← Nat.card_Ico, ← smul_eq_mul, ← sum_const]
          exact sum_le_sum fun p hp =>
            hnw_le g (Nat.lt_succ_iff.1 (mem_range.1 hg)) p (mem_Ico.1 hp).1 (mem_Ico.1 hp).2
      _ = ∑ p ∈ Ico (K.first ω 0) (K.first ω (C + 1)), (T.plan ω p \ T.correct ω p).card :=
          sum_groups _ _ fun g hg => (hfg g (by omega)).le
      _ ≤ T.wrong ω := by
          rw [hf0, ← range_eq_Ico]
          exact sum_le_sum_of_subset fun p hp => mem_range.2 ((mem_range.1 hp).trans_le hfP)
  -- the knapsack's point
  have hrate : ∀ g, 0 ≤ K.rate ω g := fun g => by
    unfold Pace.rate
    split_ifs <;> positivity
  have hsize : ∀ g ≤ C, (K.size ω g : ℝ) ≠ 0 := fun g hg => by
    have := hfg g hg
    rw [Pace.size, Nat.cast_ne_zero]
    omega
  have hfill : ((∑ g ∈ range C, nw g * (rel (g + 1) - rel g) + nw C * L : ℕ) : ℝ) ≤ K.fill ω N W := by
    refine le_csSup ⟨∑ g ∈ range (C + 1), K.rate ω g * (K.size ω g * N : ℕ), ?_⟩
      ⟨fun g => ((K.size ω g * nw g : ℕ) : ℝ), ⟨fun g => ⟨Nat.cast_nonneg _, ?_⟩, ?_⟩, ?_⟩
    · rintro v ⟨y, ⟨hy, -⟩, rfl⟩
      exact sum_le_sum fun g _ => mul_le_mul_of_nonneg_left (hy g).2 (hrate g)
    · exact Nat.cast_le.2 (Nat.mul_le_mul_left _ ((card_filter_le _ _).trans (card_fin N).le))
    · rw [← Nat.cast_sum]
      exact (Nat.cast_le.2 hwrong).trans hw
    · simp only
      rw [sum_range_succ]
      push_cast
      congr 1
      · refine sum_congr rfl fun g hg => ?_
        have hgC := mem_range.1 hg
        rw [Pace.rate, ite_eq_left hgC]
        exact div_mul_cancel_mid (hsize g hgC.le)
      · rw [Pace.rate, ite_eq_right (lt_irrefl C)]
        exact div_mul_cancel_mid (hsize C le_rfl)
  -- each tick: its blocks in node memory, on a chip and absent
  have hpt : ∀ t, N ≤ (T.mem ω t).card + S + (univ \ T.present ω t).card := by
    intro t
    have h1 : (univ : Finset (Fin N)).card ≤ (univ \ T.present ω t).card + (T.present ω t).card :=
      card_le_card_sdiff_add_card
    have h2 : (T.present ω t).card ≤ (T.mem ω t).card + (T.chip ω t).card := card_union_le _ _
    have h3 := hS ω t
    rw [card_fin] at h1
    omega
  have hheld : (rel C - en 0) * N ≤ T.heldOver ω (en 0) (rel C) + (rel C - en 0) * S +
      ∑ t ∈ Ico (en 0) (rel C), (univ \ T.present ω t).card := by
    have h := sum_le_sum fun t (_ : t ∈ Ico (en 0) (rel C)) => hpt t
    rw [sum_add_distrib, sum_add_distrib, sum_const, sum_const, Nat.card_Ico, smul_eq_mul, smul_eq_mul] at h
    exact h
  have habs : ∑ t ∈ Ico (en 0) (rel C), (univ \ T.present ω t).card ≤
      ∑ j : Fin N, ((Ico (rel 0) (rel C)).filter fun t => ¬ j ∈ T.present ω t).card := by
    refine (sum_le_sum_of_subset (Ico_subset_Ico h01 le_rfl)).trans_eq ?_
    simp only [card_filter]
    rw [sum_comm]
    refine sum_congr rfl fun t _ => ?_
    rw [show univ \ T.present ω t = univ.filter fun j => ¬ j ∈ T.present ω t by ext j; simp, card_filter]
  -- the window the entries are counted over
  have hlen : ((rel C + L - rel 0 : ℕ) : ℝ) ≤ ((rel C - en 0 : ℕ) : ℝ) + 2 * L := by
    have := hL 0 (Nat.zero_le C)
    have h : rel C + L - rel 0 ≤ (rel C - en 0) + 2 * L := by omega
    exact_mod_cast h
  have h0L : rel 0 ≤ rel C + L := h0C.trans (Nat.le_add_right _ _)
  have hreb := hsq ω (rel 0) (rel C + L) h0L
  have hdisk := hd ω (rel 0) (rel C + L) h0L
  have hlink := hl ω (rel 0) (rel C + L) h0L
  -- in the reals
  have hheldR : ((rel C - en 0 : ℕ) : ℝ) * N ≤ T.heldOver ω (en 0) (rel C) + ((rel C - en 0 : ℕ) : ℝ) * S +
      ((∑ t ∈ Ico (en 0) (rel C), (univ \ T.present ω t).card : ℕ) : ℝ) := by
    exact_mod_cast hheld
  have hA : ((∑ t ∈ Ico (en 0) (rel C), (univ \ T.present ω t).card : ℕ) : ℝ) ≤
      ((∑ g ∈ range C, nw g * (rel (g + 1) - rel g) + nw C * L : ℕ) : ℝ) +
        2 * L * ((over T.rebuilt ω (rel 0) (rel C + L) : ℝ) + over T.disk ω (rel 0) (rel C + L) +
          over T.link ω (rel 0) (rel C + L)) := by
    have h := habs.trans (hsum.trans (Nat.add_le_add_left (Nat.mul_le_mul_left _ hin) _))
    exact_mod_cast h
  have hX : 0 ≤ s / E + β + βl := add_nonneg (add_nonneg (div_nonneg hs hE.le) hβ) hβl
  have hL0 : (0 : ℝ) ≤ 2 * L := by positivity
  have hO : (over T.rebuilt ω (rel 0) (rel C + L) : ℝ) + over T.disk ω (rel 0) (rel C + L) +
      over T.link ω (rel 0) (rel C + L) ≤
      (s / E + β + βl) * (((rel C - en 0 : ℕ) : ℝ) + 2 * L) + (Br + Bd + Bl) := by
    have := mul_le_mul_of_nonneg_left hlen hX
    nlinarith
  have hO2 := mul_le_mul_of_nonneg_left hO hL0
  linarith

/-- **Held over time with the links charged** (`Accounting.Guarantees.Poub.HeldOverTimeLinks`). -/
theorem HeldOverTimeLinks : Accounting.Guarantees.Poub.HeldOverTimeLinks := by
  intro Ω _ T K S s E β βl Br Bd Bl W δs δc hE hs hβ hβl _ hl hsq hd hS hr hp hcov hk hw
  refine holdsMono (fun ω ha h => ?_) (holdsAnd hk hw)
  exact heldOverLinks_of K ω hE hs hβ hβl hl hsq hd hS hr hp hcov ha h.1 h.2

/-- **Held over time, under Identity** (`Accounting.Guarantees.Poub.HeldOverTime`). -/
theorem HeldOverTime : Accounting.Guarantees.Poub.HeldOverTime :=
  heldOverTime_of_links HeldOverTimeLinks

end Accounting.SecurityProofs.Poub
