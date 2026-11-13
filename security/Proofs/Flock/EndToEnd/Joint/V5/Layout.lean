module

public import Proofs.Flock.EndToEnd.Joint.V5.Bound

/-!
# The registered member layout of a window's sample

A window's audit (`JointAuditV5`) samples once over the whole window's population of proof blocks and hands each
member its part of the sample (`MemberV5.part`, through the member's place `MemberV5.emb` in the population). This
file states the layout that fixes those places before the sample and proves what the members' samples are under it:

* `MemberLayout.base cnt k`: under the registered counts `cnt`, the members' proof-block counts in order, member `k`'s
  base is the sum of the counts before it. A unit `u` below the total belongs to exactly one member, the one with
  `base ≤ u < base + n` (`MemberLayout.exists_member`, `MemberLayout.member_unique`), and
  `MemberLayout.memberUnits cnt k us` is member `k`'s part of a sampled list `us` in its own indices, in the list's
  order (`MemberLayout.mem_memberUnits`);
* `JointAuditV5.Laid`: the window's members are laid out in order: their counts sum to the population and member `k`'s
  proof block `v` is the population's `base k + v`. Both sides are fixed by the registration, before any coin;
* under it the members partition the population (`Laid.exists_member`, `Laid.member_unique`, `Laid.mem_iff`), a sample
  is its members' parts and nothing else (`Laid.parts`, an equivalence; `Laid.card_parts`), and member `k`'s part of the
  window's sample is read off it by the layout alone (`Laid.mem_design_draw`, `Laid.mem_draw_iff`;
  `Laid.design_draw_ok` on the sampled list the verifier's bytes give);
* **the members' sampling designs are the window's marginals**: an event of member `k`'s sample has, under `design k`
  and under `memberLaw τ k` (the design its session is played alone at), the probability the window's one sample gives
  the event at its part (`prCoin_design`, `prCoin_memberLaw`); the members' samples together are the window's
  (`Laid.prCoin_parts`); and member `k`'s design misses a set of its proof blocks, or samples one, exactly when the
  window's misses or samples their places in the population (`design_escape`, `memberLaw_escape`, `design_incl`), so at
  most `P_a` of those places under A3 (`design_escape_le`);
* the window's incorrect proof blocks are its members' (`Laid.part_wrongSet`).
-/

@[expose] public section

namespace MemberLayout

/-- **A member's base** under the registered counts `cnt`: the sum of the counts before it. -/
def base (cnt : List ℕ) (k : ℕ) : ℕ := (cnt.take k).sum

theorem base_zero (cnt : List ℕ) : base cnt 0 = 0 := by simp [base]

theorem base_succ (cnt : List ℕ) {k : ℕ} (h : k < cnt.length) : base cnt (k + 1) = base cnt k + cnt[k] :=
  List.sum_take_succ cnt k h

theorem base_length (cnt : List ℕ) : base cnt cnt.length = cnt.sum := by
  simp [base]

theorem base_mono (cnt : List ℕ) {k k' : ℕ} (h : k ≤ k') : base cnt k ≤ base cnt k' :=
  (List.take_sublist_take_left h).sum_le_sum fun _ _ => Nat.zero_le _

theorem base_le_sum (cnt : List ℕ) (k : ℕ) : base cnt k ≤ cnt.sum :=
  (List.take_sublist k cnt).sum_le_sum fun _ _ => Nat.zero_le _

theorem base_add_le (cnt : List ℕ) {k k' : ℕ} (h : k < k') (hk : k < cnt.length) :
    base cnt k + cnt[k] ≤ base cnt k' :=
  (base_succ cnt hk).symm.trans_le (base_mono cnt h)

/-- **Every unit below the total belongs to a member**: the one with `base ≤ u < base + n`. -/
theorem exists_member (cnt : List ℕ) {u : ℕ} (hu : u < cnt.sum) :
    ∃ k, ∃ hk : k < cnt.length, base cnt k ≤ u ∧ u < base cnt k + cnt[k] := by
  have key : ∀ j, j ≤ cnt.length → u < base cnt j →
      ∃ k, ∃ hk : k < cnt.length, base cnt k ≤ u ∧ u < base cnt k + cnt[k] := by
    intro j
    induction j with
    | zero => intro _ h; rw [base_zero] at h; exact absurd h (Nat.not_lt_zero _)
    | succ j ih =>
      intro hj h
      by_cases hlt : u < base cnt j
      · exact ih (Nat.le_of_succ_le hj) hlt
      · exact ⟨j, hj, Nat.le_of_not_gt hlt, by rwa [base_succ cnt hj] at h⟩
  exact key cnt.length le_rfl (by rwa [base_length])

theorem base_add_le_sum (cnt : List ℕ) {k : ℕ} (hk : k < cnt.length) : base cnt k + cnt[k] ≤ cnt.sum :=
  (base_succ cnt hk).symm.trans_le (base_le_sum cnt _)

/-- **A unit belongs to one member only.** -/
theorem member_unique (cnt : List ℕ) {u k k' : ℕ} (hk : k < cnt.length) (hk' : k' < cnt.length)
    (h₁ : base cnt k ≤ u) (h₂ : u < base cnt k + cnt[k]) (h₁' : base cnt k' ≤ u)
    (h₂' : u < base cnt k' + cnt[k']) :
    k = k' := by
  rcases Nat.lt_trichotomy k k' with h | h | h
  · exact absurd ((base_add_le cnt h hk).trans h₁') (Nat.not_le.2 h₂)
  · exact h
  · exact absurd ((base_add_le cnt h hk').trans h₁) (Nat.not_le.2 h₂')

/-- **Member `k`'s part of a sampled list**, in its own indices and the list's order: the units `u` with
`base ≤ u < base + n`, each less its base. -/
def memberUnits (cnt : List ℕ) (k : ℕ) (n : ℕ) (us : List ℕ) : List ℕ :=
  (us.filter fun u => base cnt k ≤ u ∧ u < base cnt k + n).map (· - base cnt k)

theorem mem_memberUnits (cnt : List ℕ) (k n : ℕ) (us : List ℕ) (v : ℕ) :
    v ∈ memberUnits cnt k n us ↔ v < n ∧ base cnt k + v ∈ us := by
  simp only [memberUnits, List.mem_map, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro ⟨u, ⟨hu, h₁, h₂⟩, rfl⟩
    refine ⟨by omega, ?_⟩
    rwa [Nat.add_sub_cancel' h₁]
  · rintro ⟨hv, hu⟩
    exact ⟨base cnt k + v, ⟨hu, Nat.le_add_right _ _, Nat.add_lt_add_left hv _⟩, Nat.add_sub_cancel_left ..⟩

/-- A member's part of a sampled list is as long as the list at most. -/
theorem length_memberUnits_le (cnt : List ℕ) (k n : ℕ) (us : List ℕ) :
    (memberUnits cnt k n us).length ≤ us.length := by
  rw [memberUnits, List.length_map]
  exact List.length_filter_le _ _

end MemberLayout

namespace Flock.Guarantees.JointAuditV5

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

open Classical

variable (w : JointAuditV5)

/-- **The registered counts**: each member's number of proof blocks, in the members' order. -/
def counts : List ℕ := w.ms.map (·.n)

theorem counts_length : w.counts.length = w.ms.length := List.length_map _

theorem counts_get (k : Fin w.ms.length) : w.counts[k.1]'(by rw [counts_length]; exact k.2) = (w.ms.get k).n := by
  simp [counts]

/-- **Member `k`'s base**: the sum of the counts of the members before it. -/
def base (k : Fin w.ms.length) : ℕ := MemberLayout.base w.counts k.1

/-- **The window's members are laid out in order**: their counts sum to the population, and member `k`'s proof block
`v` is the population's `base k + v`. The counts, the bases and the places are the registration's, fixed before the
sample. -/
structure Laid : Prop where
  total : w.counts.sum = w.nJ
  emb : ∀ (k : Fin w.ms.length) (v : Fin (w.ms.get k).n), ((w.ms.get k).emb v).val = w.base k + v.val

variable {w}

namespace Laid

/-- **A population unit is member `k`'s** exactly when `base k ≤ u < base k + n_k`. -/
theorem mem_iff (h : w.Laid) (k : Fin w.ms.length) (u : Fin w.nJ) :
    (∃ v, (w.ms.get k).emb v = u) ↔ w.base k ≤ u.val ∧ u.val < w.base k + (w.ms.get k).n := by
  constructor
  · rintro ⟨v, rfl⟩
    rw [h.emb]
    exact ⟨Nat.le_add_right _ _, Nat.add_lt_add_left v.2 _⟩
  · rintro ⟨h₁, h₂⟩
    refine ⟨⟨u.val - w.base k, by omega⟩, Fin.ext ?_⟩
    rw [h.emb]
    show w.base k + (u.val - w.base k) = u.val
    omega

/-- **Every population unit belongs to a member.** -/
theorem exists_member (h : w.Laid) (u : Fin w.nJ) :
    ∃ k : Fin w.ms.length, w.base k ≤ u.val ∧ u.val < w.base k + (w.ms.get k).n := by
  obtain ⟨k, hk, h₁, h₂⟩ := MemberLayout.exists_member w.counts (u := u.val) (by rw [h.total]; exact u.2)
  refine ⟨⟨k, w.counts_length ▸ hk⟩, h₁, ?_⟩
  rw [← w.counts_get ⟨k, w.counts_length ▸ hk⟩]
  exact h₂

/-- **A unit belongs to one member only.** -/
theorem member_unique (_h : w.Laid) {u : ℕ} {k k' : Fin w.ms.length} (h₁ : w.base k ≤ u)
    (h₂ : u < w.base k + (w.ms.get k).n) (h₁' : w.base k' ≤ u) (h₂' : u < w.base k' + (w.ms.get k').n) :
    k = k' := by
  rw [← w.counts_get k] at h₂
  rw [← w.counts_get k'] at h₂'
  exact Fin.ext (MemberLayout.member_unique w.counts _ _ h₁ h₂ h₁' h₂')

theorem exists_emb (h : w.Laid) (u : Fin w.nJ) : ∃ k v, (w.ms.get k).emb v = u := by
  obtain ⟨k, hk⟩ := h.exists_member u
  exact ⟨k, (h.mem_iff k u).2 hk⟩

theorem emb_injective (h : w.Laid) (k : Fin w.ms.length) : Function.Injective (w.ms.get k).emb := fun v v' e => by
  have e' := congrArg Fin.val e
  rw [h.emb, h.emb] at e'
  exact Fin.ext (Nat.add_left_cancel e')

theorem member_eq (h : w.Laid) {k k' : Fin w.ms.length} {v : Fin (w.ms.get k).n} {v' : Fin (w.ms.get k').n}
    (e : (w.ms.get k).emb v = (w.ms.get k').emb v') : k = k' :=
  h.member_unique ((h.mem_iff k _).1 ⟨v, rfl⟩).1 ((h.mem_iff k _).1 ⟨v, rfl⟩).2
    ((h.mem_iff k' _).1 ⟨v', e.symm⟩).1 ((h.mem_iff k' _).1 ⟨v', e.symm⟩).2

end Laid

variable (w)

theorem mem_part (k : Fin w.ms.length) (S : Finset (Fin w.nJ)) (v : Fin (w.ms.get k).n) :
    v ∈ (w.ms.get k).part S ↔ (w.ms.get k).emb v ∈ S := by
  simp [MemberV5.part]

/-- **The sample with the parts `T`**: every member's part at its place in the population. -/
noncomputable def join (T : (k : Fin w.ms.length) → Finset (Fin (w.ms.get k).n)) : Finset (Fin w.nJ) :=
  Finset.univ.biUnion fun k => (T k).image (w.ms.get k).emb

variable {w}

namespace Laid

theorem join_part (h : w.Laid) (S : Finset (Fin w.nJ)) : w.join (fun k => (w.ms.get k).part S) = S := by
  ext u
  simp only [join, Finset.mem_biUnion, Finset.mem_univ, true_and, Finset.mem_image, mem_part]
  constructor
  · rintro ⟨k, v, hv, rfl⟩
    exact hv
  · intro hu
    obtain ⟨k, v, rfl⟩ := h.exists_emb u
    exact ⟨k, v, hu, rfl⟩

theorem part_join (h : w.Laid) (T : (k : Fin w.ms.length) → Finset (Fin (w.ms.get k).n)) (k : Fin w.ms.length) :
    (w.ms.get k).part (w.join T) = T k := by
  ext v
  simp only [join, mem_part, Finset.mem_biUnion, Finset.mem_univ, true_and, Finset.mem_image]
  constructor
  · rintro ⟨k', v', hv', e⟩
    obtain rfl := h.member_eq e
    rwa [← h.emb_injective _ e]
  · intro hv
    exact ⟨k, v, hv, rfl⟩

/-- **A sample is its members' parts**: under the layout, the samples of the population are the families of one sample
per member. -/
noncomputable def parts (h : w.Laid) :
    Finset (Fin w.nJ) ≃ ((k : Fin w.ms.length) → Finset (Fin (w.ms.get k).n)) where
  toFun S k := (w.ms.get k).part S
  invFun := w.join
  left_inv := h.join_part
  right_inv T := funext (h.part_join T)

/-- **The members' parts have the sample's units between them.** -/
theorem card_parts (h : w.Laid) (S : Finset (Fin w.nJ)) : ∑ k, ((w.ms.get k).part S).card = S.card := by
  conv_rhs => rw [← h.join_part S]
  rw [join, Finset.card_biUnion]
  · exact Finset.sum_congr rfl fun k _ => (Finset.card_image_of_injective _ (h.emb_injective k)).symm
  · intro k _ k' _ hkk'
    rw [Function.onFun, Finset.disjoint_left]
    intro u hu hu'
    obtain ⟨v, -, rfl⟩ := Finset.mem_image.1 hu
    obtain ⟨v', -, e⟩ := Finset.mem_image.1 hu'
    exact hkk' (h.member_eq e).symm

end Laid

theorem disjoint_part_iff (k : Fin w.ms.length) (S : Finset (Fin w.nJ)) (B : Finset (Fin (w.ms.get k).n)) :
    Disjoint ((w.ms.get k).part S) B ↔ Disjoint S (B.image (w.ms.get k).emb) := by
  simp only [Finset.disjoint_left, mem_part, Finset.mem_image, not_exists, not_and]
  constructor
  · rintro h u hu v hv rfl
    exact h hu hv
  · intro h v hv hB
    exact h hv v hB rfl

/-! ### Member `k`'s sample, read off the window's by the layout -/

variable (w)

/-- **The window's sample on the verifier's bytes**: where `drawUnits` samples the list `r.1`, the units of that
list. -/
theorem draw_ok (ω : w.L.Ω) {r : Array ℕ × ℕ}
    (hr : _root_.Spot.Draw.drawUnits w.law w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) = .ok r) (u : Fin w.nJ) :
    u ∈ w.L.draw ω ↔ u.val ∈ r.1 := by
  dsimp only [JointAuditV5.L, Bridge.execSel]
  rw [hr]
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]

/-- **The window's sample where the verifier's bytes run out**: every unit. -/
theorem draw_error (ω : w.L.Ω) {e : String}
    (hr : _root_.Spot.Draw.drawUnits w.law w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) = .error e)
    (u : Fin w.nJ) :
    u ∈ w.L.draw ω := by
  dsimp only [JointAuditV5.L, Bridge.execSel]
  rw [hr]
  exact Finset.mem_univ u

variable {w}

namespace Laid

/-- **Member `k`'s sample is the window's at its place**: its proof block `v` is sampled exactly when the population's
`base k + v` is. -/
theorem mem_design_draw (h : w.Laid) (k : Fin w.ms.length) (ω : w.L.Ω) (v : Fin (w.ms.get k).n) (u : Fin w.nJ)
    (hu : u.val = w.base k + v.val) : v ∈ (w.design k).draw ω ↔ u ∈ w.L.draw ω := by
  have e : (w.ms.get k).emb v = u := Fin.ext ((h.emb k v).trans hu.symm)
  rw [← e]
  exact w.mem_part k _ v

/-- **A sampled proof block `u` belongs to the member with `base ≤ u < base + n`**, as its proof block `u - base`. -/
theorem mem_draw_iff (h : w.Laid) (ω : w.L.Ω) (u : Fin w.nJ) :
    u ∈ w.L.draw ω ↔ ∃ k v, u.val = w.base k + v.val ∧ v ∈ (w.design k).draw ω := by
  constructor
  · intro hu
    obtain ⟨k, v, rfl⟩ := h.exists_emb u
    exact ⟨k, v, h.emb k v, (w.mem_part k _ v).2 hu⟩
  · rintro ⟨k, v, hu, hv⟩
    exact (h.mem_design_draw k ω v u hu).1 hv

/-- **Member `k`'s sample on the verifier's bytes**: where `drawUnits` samples the list `r.1` over the window's
population, member `k`'s sample is its part of that list (`MemberLayout.memberUnits`): the members' samples are read
off the one sampled list in order, by the registered bases and counts. -/
theorem design_draw_ok (h : w.Laid) (k : Fin w.ms.length) (ω : w.L.Ω) {r : Array ℕ × ℕ}
    (hr : _root_.Spot.Draw.drawUnits w.law w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) = .ok r)
    (v : Fin (w.ms.get k).n) :
    v ∈ (w.design k).draw ω ↔ v.val ∈ MemberLayout.memberUnits w.counts k.1 (w.ms.get k).n r.1.toList := by
  have hlt : w.base k + v.val < w.nJ := by
    have := ((h.mem_iff k _).1 ⟨v, rfl⟩).2
    rw [← h.emb k v]
    exact ((w.ms.get k).emb v).2
  rw [h.mem_design_draw k ω v ⟨w.base k + v.val, hlt⟩ rfl, w.draw_ok ω hr, MemberLayout.mem_memberUnits,
    Array.mem_toList_iff]
  exact ⟨fun hu => ⟨v.2, hu⟩, fun hu => hu.2⟩

/-- **Member `k`'s sample where the verifier's bytes run out**: every one of its proof blocks. -/
theorem design_draw_error (k : Fin w.ms.length) (ω : w.L.Ω) {e : String}
    (hr : _root_.Spot.Draw.drawUnits w.law w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) = .error e) :
    (w.design k).draw ω = Finset.univ :=
  Finset.eq_univ_iff_forall.2 fun v => (w.mem_part k _ v).2 (w.draw_error ω hr _)

end Laid

/-! ### The members' sampling designs are the window's marginals -/

variable (w)

/-- **Member `k`'s sampling design is the window's marginal**: an event of its sample has the probability the window's
one sample gives the event at member `k`'s part. -/
theorem prCoin_design (k : Fin w.ms.length) (E : Finset (Fin (w.ms.get k).n) → Prop) :
    prCoin (fun ω : (w.design k).Ω => E ((w.design k).draw ω)) =
      prCoin (fun ω : w.L.Ω => E ((w.ms.get k).part (w.L.draw ω))) :=
  rfl

/-- **Member `k`'s design misses a set of its proof blocks** exactly when the window's sample misses their places in
the population. -/
theorem design_escape (k : Fin w.ms.length) (B : Finset (Fin (w.ms.get k).n)) :
    (w.design k).escape B = w.L.escape (B.image (w.ms.get k).emb) :=
  congrArg (fun p : w.L.Ω → Prop => prCoin p) (funext fun ω => propext (w.disjoint_part_iff k (w.L.draw ω) B))

/-- **Member `k`'s design samples its proof block `v`** exactly when the window's samples its place. -/
theorem design_incl (k : Fin w.ms.length) (v : Fin (w.ms.get k).n) :
    (w.design k).incl v = w.L.incl ((w.ms.get k).emb v) :=
  congrArg (fun p : w.L.Ω → Prop => prCoin p) (funext fun ω => propext (w.mem_part k (w.L.draw ω) v))

variable {w}

/-- **The members' samples together are the window's**: an event of the window's sample is the event of the members'
samples at the sample they make up. -/
theorem Laid.prCoin_parts (h : w.Laid) (E : Finset (Fin w.nJ) → Prop) :
    prCoin (fun ω : w.L.Ω => E (w.L.draw ω)) =
      prCoin (fun ω : w.L.Ω => E (w.join fun k => (w.design k).draw ω)) :=
  congrArg (fun p : w.L.Ω → Prop => prCoin p) (funext fun ω => congrArg E (h.join_part (w.L.draw ω)).symm)

variable (w)

/-- **The design member `k`'s session is played alone at is the window's marginal too**: the earlier sessions' coins it
samples with the window's don't move its sample. -/
theorem prCoin_memberLaw (τ : Strategy w.game) (k : Fin w.ms.length) (E : Finset (Fin (w.ms.get k).n) → Prop) :
    prCoin (fun x : (w.memberLaw τ k).Ω => E ((w.memberLaw τ k).draw x)) =
      prCoin (fun ω : w.L.Ω => E ((w.ms.get k).part (w.L.draw ω))) :=
  w.avg_memberLaw τ k fun S => if E S then 1 else 0

/-- Member `k`'s design played alone misses a set of its proof blocks exactly when the window's sample misses their
places in the population. -/
theorem memberLaw_escape (τ : Strategy w.game) (k : Fin w.ms.length) (B : Finset (Fin (w.ms.get k).n)) :
    (w.memberLaw τ k).escape B = w.L.escape (B.image (w.ms.get k).emb) :=
  (w.prCoin_memberLaw τ k fun S => Disjoint S B).trans (w.design_escape k B)

/-- **Member `k`'s design misses a set of its proof blocks** with probability at most the window's sampling design's
own miss `P_a` of their places in the population, under A3. -/
theorem design_escape_le (hA3 : Assumptions.UniformRandomBytes w.Lb w.src) (k : Fin w.ms.length)
    (B : Finset (Fin (w.ms.get k).n)) :
    (w.design k).escape B ≤ ENNReal.ofReal (w.pa (B.image (w.ms.get k).emb)) :=
  (w.design_escape k B).trans_le (Bridge.execSel_escape_le w.src hA3 w.hlaw w.hns _)

/-! ### The window's incorrect proof blocks are its members' -/

theorem wrongSet_eq_join (τ : Strategy w.game) : w.wrongSet τ = w.join (w.memberWrong τ) := by
  ext u
  simp only [JointAuditV5.wrongSet, JointAuditV5.join, Finset.mem_biUnion, Finset.mem_univ, true_and,
    Finset.mem_image]

variable {w}

/-- **Member `k`'s part of the window's incorrect proof blocks is its own incorrect proof blocks.** -/
theorem Laid.part_wrongSet (h : w.Laid) (τ : Strategy w.game) (k : Fin w.ms.length) :
    (w.ms.get k).part (w.wrongSet τ) = w.memberWrong τ k := by
  rw [w.wrongSet_eq_join τ]
  exact h.part_join (w.memberWrong τ) k

/-- **A population unit is incorrect** exactly when it is an incorrect proof block of the member it belongs to. -/
theorem Laid.mem_wrongSet (h : w.Laid) (τ : Strategy w.game) (u : Fin w.nJ) :
    u ∈ w.wrongSet τ ↔ ∃ k v, u.val = w.base k + v.val ∧ v ∈ w.memberWrong τ k := by
  constructor
  · intro hu
    obtain ⟨k, v, rfl⟩ := h.exists_emb u
    refine ⟨k, v, h.emb k v, ?_⟩
    rw [← h.part_wrongSet τ k]
    exact (w.mem_part k _ v).2 hu
  · rintro ⟨k, v, hu, hv⟩
    have e : (w.ms.get k).emb v = u := Fin.ext ((h.emb k v).trans hu.symm)
    rw [← e, ← w.mem_part k, h.part_wrongSet τ k]
    exact hv

end Flock.Guarantees.JointAuditV5
