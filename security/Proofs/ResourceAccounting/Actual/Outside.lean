module

public import Proofs.ResourceAccounting.Actual.Physical
public import Definitions.Registration.Server
public import Proofs.ResourceAccounting.Actual.Device
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# What the limits leave outside a set of gates

A proof of resource use shows that a set `S` of gates, a declared computation, uses at least `x` of a resource. The
physical constraints then leave the rest of the run at most the limit less `x`:

* `work_outside`: under `ComputeRate`, the gates of `A[X, [a, b))` outside `S` do at most `Φ_X (b − a) − x` work;
* `state_outside`: where a run's state of `X` at a tick, what its leaves hold and what is in flight between them, is
  at most `M` bits and at least `x` of them are of `S`, `X` carries at most `M − x` bits outside `S` across the tick;
* `delivered_le`: under `Links`, over `[a, b)` the links deliver at most `β_X (b − a)` bits to `X`;
* `delivered_outside`: under `Links`, where the links deliver to `X` over `[a, b)` at least `x` bits of `S` and at most
  `y` in all, `A[X, [a, b))` reads at most `min (β_X (b − a)) y − x` bits outside `S` from outside `X`.

A capacity certificate is in bit-ticks over a window, an average: `card_short_ticks_mul_le` bounds the ticks at which
the account holds fewer than `x` bits.

`feasible_ofLeaves_iff` is the feasible set at limits per leaf and per link (`Hardware.ofLeaves`), and
`server_compute`, `server_memory` and `server_link` are a server's limits: its work a tick, the bits it carries and
the bits a tick across its boundary. `server_gpu_fits`: each GPU of a server is a device of the device model,
its die the chip, its device memory the memory, and its links to the host and the switch the side channels, so the
joint audit's guarantees hold of each GPU.
-/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Actual

open Verity Finset

variable {L : Type*} (A : Placed L)

theorem work_inter_add_sdiff (π : Fin A.N → ℕ) (R S : Finset (Fin A.N)) :
    A.work π (R ∩ S) + A.work π (R \ S) = A.work π R := by
  unfold Placed.work
  exact sum_inter_add_sum_sdiff R S π

/-- A set's width in `G` is at most its width outside `S` plus its width in `S ∩ G`. -/
theorem width_inter_le_split (R S G : Finset (Fin A.N)) :
    A.width (R ∩ G) ≤ A.width (R ∩ Sᶜ) + A.width (R ∩ (S ∩ G)) := by
  classical
  refine (width_mono A fun g hg => ?_).trans (width_union_le A _ _)
  obtain ⟨h1, h2⟩ := mem_inter.1 hg
  by_cases hs : g ∈ S
  · exact mem_union_right _ (mem_inter.2 ⟨h1, mem_inter.2 ⟨hs, h2⟩⟩)
  · exact mem_union_left _ (mem_inter.2 ⟨h1, mem_compl.2 hs⟩)

/-- **Work outside `S`**: if the gates of `A[X, [a, b))` in `S` do at least `x` work, those outside `S` do at most
`Φ_X (b − a) − x`. -/
theorem work_outside {𝒳 : Set (Finset L)} {π : Fin A.N → ℕ} {compute : Finset L → ℕ} (h : A.ComputeRate 𝒳 π compute)
    {X : Finset L} (hX : X ∈ 𝒳) (a b : ℕ) (S : Finset (Fin A.N)) {x : ℕ}
    (hx : x ≤ A.work π (A.region X (Set.Ico a b) ∩ S)) :
    A.work π (A.region X (Set.Ico a b) \ S) ≤ compute X * (b - a) - x := by
  have h1 := h X hX a b
  have h2 := work_inter_add_sdiff A π (A.region X (Set.Ico a b)) S
  omega

/-- **State outside `S`**: if the run's state of `X` at `t`, what its leaves hold and what is in
flight between them, is at most `M` bits and at least `x` of them are of `S`, then `X` carries at most `M − x` bits
outside `S` across `t`. A gate of `S` counts when it is held, whether or not it is read again. -/
theorem state_outside {R : Actual.Run L} {X : Finset L} {t M x : ℕ} (hM : R.mem X t ≤ M)
    (S : Finset (Fin R.A.N)) (hx : x ≤ R.A.width (R.state X t ∩ S)) : R.A.stateIn X t Sᶜ ≤ M - x := by
  classical
  have h1 : R.A.stateIn X t Sᶜ ≤ R.A.width (R.state X t ∩ Sᶜ) := by
    unfold Placed.stateIn
    refine width_mono R.A (inter_subset_inter (fun a ha => ?_) (Finset.Subset.refl _))
    obtain ⟨hai, hxa⟩ := Finset.mem_filter.1 ha
    simp only [Actual.Run.state, Finset.mem_filter, Finset.mem_univ, true_and]
    exact Or.inr ⟨hai, hxa⟩
  have h2 := width_inter_add_compl R.A (R.state X t) S
  unfold Actual.Run.mem at hM
  omega

/-- **From a capacity certificate to the ticks that fall short**: if account `a` holds at most `m` bits of device memory
at each tick of the window `w` and at least `C` bit-ticks over it, the ticks at which it holds fewer than `x ≤ m` bits
number at most `(m · len − C) / (m − x)`. -/
theorem card_short_ticks_mul_le {ι α : Type} (sc : Schedule ι α) (w : Window) (a : α) {m x C : ℕ}
    (hm : ∀ t ∈ w.ticks, (sc t).mem a ≤ m) (hx : x ≤ m) (hC : C ≤ sc.held w a) :
    (w.ticks.filter fun t => (sc t).mem a < x).card * (m - x) ≤ m * w.len - C := by
  unfold Schedule.held at hC
  have hsplit := sum_filter_add_sum_filter_not w.ticks (fun t => (sc t).mem a < x) fun t => (sc t).mem a
  have hbad : ∑ t ∈ w.ticks.filter (fun t => (sc t).mem a < x), (sc t).mem a ≤
      (w.ticks.filter fun t => (sc t).mem a < x).card * x :=
    (sum_le_card_nsmul _ _ x fun t ht => (mem_filter.1 ht).2.le).trans (by rw [smul_eq_mul])
  have hgood : ∑ t ∈ w.ticks.filter (fun t => ¬ (sc t).mem a < x), (sc t).mem a ≤
      (w.ticks.filter fun t => ¬ (sc t).mem a < x).card * m :=
    (sum_le_card_nsmul _ _ m fun t ht => hm t (mem_filter.1 ht).1).trans (by rw [smul_eq_mul])
  have hcard := card_filter_add_card_filter_not (s := w.ticks) fun t => (sc t).mem a < x
  have hlen : w.ticks.card = w.len := by simp [Window.ticks]
  have h1 : (w.ticks.filter fun t => (sc t).mem a < x).card * (m - x) +
      (w.ticks.filter fun t => (sc t).mem a < x).card * x =
      (w.ticks.filter fun t => (sc t).mem a < x).card * m := by
    rw [← Nat.mul_add, Nat.sub_add_cancel hx]
  have h2 : m * w.len = (w.ticks.filter fun t => (sc t).mem a < x).card * m +
      (w.ticks.filter fun t => ¬ (sc t).mem a < x).card * m := by
    rw [← hlen, ← hcard]
    ring
  exact Nat.le_sub_of_add_le (by linarith)

open Classical in
/-- **What the links deliver**: under `Links`, over the ticks of `[a, b)` the links deliver at most `β_X (b − a)` bits
to `X`, a bit counted at each tick `X` reads it. -/
theorem delivered_le {E : Type*} [Fintype E] {ends : E → L × L} {β : E → ℕ} {link : Fin A.N → Fin A.N → E}
    (h : A.Links ends β link) (X : Finset L) (a b : ℕ) :
    ∑ t ∈ Finset.Ico a b, A.width ((A.into (A.region X (Set.Ico t (t + 1)))).filter fun g => A.x g ∉ X) ≤
      linkCapacity ends β X * (b - a) := by
  refine (sum_le_sum fun t _ => ?_).trans (by rw [sum_const, Nat.card_Ico, smul_eq_mul, Nat.mul_comm])
  have h4 := ingress_le A h X t (t + 1)
  unfold Placed.ingress at h4
  rw [Finset.filter_congr_decidable, Nat.add_sub_cancel_left, Nat.mul_one] at h4
  exact h4

open Classical in
/-- **What a window reads outside `S`, by what the links deliver**: if over the ticks of `[a, b)` the links deliver
to `X` at least `x` bits of `S` and at most `y` bits in all, a bit counted at each tick `X` reads it, then
`A[X, [a, b))` reads at most `min (β_X (b − a)) y − x` bits outside `S` from outside `X`. -/
theorem delivered_outside {E : Type*} [Fintype E] {ends : E → L × L} {β : E → ℕ} {link : Fin A.N → Fin A.N → E}
    (h : A.Links ends β link) (X : Finset L) (a b : ℕ) (S : Finset (Fin A.N)) {x y : ℕ}
    (hx : x ≤ ∑ t ∈ Finset.Ico a b,
      A.width (((A.into (A.region X (Set.Ico t (t + 1)))).filter fun g => A.x g ∉ X) ∩ S))
    (hy : ∑ t ∈ Finset.Ico a b,
      A.width (((A.into (A.region X (Set.Ico t (t + 1)))).filter fun g => A.x g ∉ X) ∩ univ) ≤ y) :
    A.width (((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ Sᶜ) ≤
      min (linkCapacity ends β X * (b - a)) y - x := by
  set D : ℕ → Finset (Fin A.N) := fun t => (A.into (A.region X (Set.Ico t (t + 1)))).filter fun g => A.x g ∉ X
  have hsub : ((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ Sᶜ ⊆
      (Finset.Ico a b).biUnion fun t => D t ∩ Sᶜ := by
    intro s hs
    obtain ⟨hs1, hsS⟩ := mem_inter.1 hs
    obtain ⟨hin, hsx⟩ := mem_filter.1 hs1
    obtain ⟨-, g, hg, hsg⟩ := (mem_into A).1 hin
    obtain ⟨hc, hgx, hgt⟩ := (mem_region A).1 hg
    exact mem_biUnion.2 ⟨A.t g, Finset.mem_Ico.2 hgt, mem_inter.2 ⟨mem_filter.2 ⟨(mem_into A).2
      ⟨fun hr => hsx ((mem_region A).1 hr).2.1, g, (mem_region A).2 ⟨hc, hgx, le_rfl, Nat.lt_succ_self _⟩, hsg⟩,
      hsx⟩, hsS⟩⟩
  have h1 := (width_mono A hsub).trans (width_biUnion_le A _ _)
  have h2 : ∑ t ∈ Finset.Ico a b, A.width (D t) =
      ∑ t ∈ Finset.Ico a b, A.width (D t ∩ S) + ∑ t ∈ Finset.Ico a b, A.width (D t ∩ Sᶜ) := by
    rw [← sum_add_distrib]
    exact sum_congr rfl fun t _ => width_inter_add_compl A (D t) S
  have h3 : ∑ t ∈ Finset.Ico a b, A.width (D t) ≤ linkCapacity ends β X * (b - a) := delivered_le A h X a b
  have h5 : ∑ t ∈ Finset.Ico a b, A.width (D t) ≤ y := by
    simpa only [inter_univ] using hy
  have h6 : ∑ t ∈ Finset.Ico a b, A.width (D t) ≤ min (linkCapacity ends β X * (b - a)) y := le_min h3 h5
  have h7 : x ≤ ∑ t ∈ Finset.Ico a b, A.width (D t ∩ S) := hx
  have h8 : ∑ t ∈ Finset.Ico a b, A.width (D t ∩ Sᶜ) ≤ min (linkCapacity ends β X * (b - a)) y - x :=
    Nat.le_sub_of_add_le (by rw [Nat.add_comm]; exact (Nat.add_le_add_right h7 _).trans (h2 ▸ h6))
  exact h1.trans h8

/-- What `X` carries across `t` in `G` is at most what it carries outside `S` plus what it carries in `S ∩ G`. -/
theorem stateIn_le_split (X : Finset L) (t : ℕ) (S G : Finset (Fin A.N)) :
    A.stateIn X t G ≤ A.stateIn X t Sᶜ + A.stateIn X t (S ∩ G) := by
  unfold Placed.stateIn
  exact width_inter_le_split A _ S G

/-- Gates each priced at least 1 number at most the work of the region they are in. -/
theorem card_filter_le_work {α : Type*} (π : Fin A.N → ℕ) (R : Finset (Fin A.N)) (s : Finset α) (f : α ↪ Fin A.N)
    (hπ : ∀ g ∈ s, 1 ≤ π (f g)) [DecidablePred fun g => f g ∈ R] :
    (s.filter fun g => f g ∈ R).card ≤ A.work π R := by
  unfold Placed.work
  calc (s.filter fun g => f g ∈ R).card = ∑ _g ∈ s.filter (fun g => f g ∈ R), 1 := by
        rw [sum_const, smul_eq_mul, mul_one]
    _ ≤ ∑ g ∈ s.filter (fun g => f g ∈ R), π (f g) := sum_le_sum fun g hg => hπ g (mem_filter.1 hg).1
    _ = ∑ a ∈ (s.filter fun g => f g ∈ R).map f, π a := (sum_map _ f π).symm
    _ ≤ ∑ a ∈ R, π a := sum_le_sum_of_subset fun a ha => by
        obtain ⟨g, hg, rfl⟩ := mem_map.1 ha
        exact (mem_filter.1 hg).2

/-- Compute rate at every leaf is compute rate at every set of leaves, at the sum of its leaves'. -/
theorem computeRate_of_leaves [DecidableEq L] {π : Fin A.N → ℕ} {c : L → ℕ}
    (h : ∀ p a b, A.work π (A.region {p} (Set.Ico a b)) ≤ c p * (b - a)) :
    A.ComputeRate Set.univ π (fun X => ∑ p ∈ X, c p) := by
  intro X _ a b
  have hmaps : ∀ g ∈ A.region X (Set.Ico a b), A.x g ∈ X := fun g hg => ((mem_region A).1 hg).2.1
  unfold Placed.work
  change _ ≤ (∑ p ∈ X, c p) * (b - a)
  rw [sum_mul, ← sum_fiberwise_of_maps_to hmaps]
  refine sum_le_sum fun p hp => ?_
  have hfib : (A.region X (Set.Ico a b)).filter (fun g => A.x g = p) = A.region {p} (Set.Ico a b) := by
    ext g
    simp only [mem_filter, mem_region A, mem_singleton]
    constructor
    · rintro ⟨⟨hi, -, ht⟩, rfl⟩
      exact ⟨hi, rfl, ht⟩
    · rintro ⟨hi, rfl, ht⟩
      exact ⟨⟨hi, hp, ht⟩, rfl⟩
  rw [hfib]
  exact h p a b

/-- **The feasible set at limits per leaf and per link is four clauses**: a gate reads gates placed no later than
itself, each leaf does at most its work a tick, every set of leaves carries at most its leaves' bits, and the links
hold. `Feasible`'s sequential-speed clause holds of every run there. -/
theorem feasible_ofLeaves_iff [DecidableEq L] {E : Type*} (c M : L → ℕ) (ends : E → L × L) (β : E → ℕ) (u : Usage A E) :
    Feasible (Hardware.ofLeaves c M ends β) A u ↔
      (A.Causal fun _ _ => 0) ∧ (∀ p a b, A.work u.π (A.region {p} (Set.Ico a b)) ≤ c p * (b - a)) ∧
        (∀ X t, A.state X t ≤ ∑ p ∈ X, M p) ∧ A.Links ends β u.link := by
  constructor
  · rintro ⟨hc, hw, -, hm, hl⟩
    refine ⟨hc, fun p a b => ?_, fun X t => hm X (Set.mem_univ _) t, hl⟩
    have h := hw {p} (Set.mem_univ _) a b
    simpa [Hardware.ofLeaves] using h
  · rintro ⟨hc, hw, hm, hl⟩
    refine ⟨hc, computeRate_of_leaves A hw, fun k p _ hD => ?_, fun X _ t => hm X t, hl⟩
    simp [Hardware.ofLeaves] at hD

section Server

variable (n die host fabric chip dram ram queue world store bus peer net : ℕ)

/-- A server does its `n` dies' work, its host's and its switch's a tick. -/
theorem server_compute :
    (server n die host fabric chip dram ram queue world store bus peer net).compute (Loc.inside n) =
      n * die + host + fabric := by
  calc (server n die host fabric chip dram ram queue world store bus peer net).compute (Loc.inside n)
      = ∑ p : Loc n, ((∑ i : Fin n, if p = Loc.die i then die else 0) + (if p = Loc.host then host else 0)
          + if p = Loc.switch then fabric else 0) := by
        simp only [server, Hardware.ofLeaves, Loc.inside, ← filter_ne' univ Loc.outside, sum_filter]
        exact sum_congr rfl fun p _ => by cases p <;> simp
    _ = n * die + host + fabric := by
        rw [sum_add_distrib, sum_add_distrib, sum_comm]
        simp

/-- A server carries its `n` dies' and device memories' bits, its host's and its switch's. -/
theorem server_memory :
    (server n die host fabric chip dram ram queue world store bus peer net).M (Loc.inside n) =
      n * (chip + dram) + ram + queue := by
  calc (server n die host fabric chip dram ram queue world store bus peer net).M (Loc.inside n)
      = ∑ p : Loc n, ((∑ i : Fin n, ((if p = Loc.die i then chip else 0) + if p = Loc.mem i then dram else 0))
          + (if p = Loc.host then ram else 0) + if p = Loc.switch then queue else 0) := by
        simp only [server, Hardware.ofLeaves, Loc.inside, ← filter_ne' univ Loc.outside, sum_filter]
        exact sum_congr rfl fun p _ => by cases p <;> simp
    _ = n * (chip + dram) + ram + queue := by
        rw [sum_add_distrib, sum_add_distrib, sum_comm]
        simp [sum_add_distrib, mul_add]

/-- The one link across a server's boundary is its host's to the world outside. -/
theorem server_link :
    linkCapacity (server n die host fabric chip dram ram queue world store bus peer net).ends
      (server n die host fabric chip dram ram queue world store bus peer net).β (Loc.inside n) = net := by
  classical
  have hb : boundary (server n die host fabric chip dram ram queue world store bus peer net).ends (Loc.inside n) =
      {Wire.net} := by
    ext e
    cases e <;> simp [boundary, server, Hardware.ofLeaves, Wire.ends, Loc.inside]
  rw [linkCapacity, hb, sum_singleton]
  simp [server, Hardware.ofLeaves]

open Classical in
/-- **Each GPU of a server is a device.** The schedule a run feasible on the server induces at GPU `i` fits the device
with the die's state, the device memory's bits, the memory link's bandwidth and the host and switch links as side
channels, when each op class runs at most at its rate there. -/
theorem server_gpu_fits {α : Type} [Fintype α] (R : Actual.Run (Loc n))
    {usage : Usage R.A (Wire n)}
    (hF : Feasible (server n die host fabric chip dram ram queue world store bus peer net) R.A usage)
    (hM : R.Memory Set.univ (server n die host fabric chip dram ram queue world store bus peer net).M) (i : Fin n)
    (rate : OpClass → ℕ) (cls : Fin R.A.N → Option OpClass) (acct : Fin R.A.N → α)
    (hΦ : ∀ o, R.A.ComputeRate {{Loc.die i} ∪ {Loc.mem i}} (R.A.classPrice cls o) (fun _ => rate o)) :
    (R.schedule {Loc.die i} {Loc.mem i} cls acct usage.link (Wire.store i, Loc.die i)
      (fun b : Bool => (if b then Wire.bus i else Wire.peer i, Loc.die i))).Fits
      ⟨chip, dram, rate, store, fun b => if b then bus else peer⟩ := by
  refine fits_of_feasible R hF hM _ _ _ cls acct _ _ (by convert hΦ) (Set.mem_univ _) (Set.mem_univ _) ?_ ?_ ?_ ?_
  · simp [server, Hardware.ofLeaves]
  · simp [server, Hardware.ofLeaves]
  · simp [server, Hardware.ofLeaves]
  · intro b
    cases b <;> simp [server, Hardware.ofLeaves]

end Server

end ResourceAccounting.SecurityProofs.Actual
