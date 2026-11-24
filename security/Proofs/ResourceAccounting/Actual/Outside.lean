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
* `state_outside`: under `MemoryAt`, `X` carries at most `M_X − x` bits outside `S` across the tick;
* `ingress_outside`: under `Links`, `A[X, [a, b))` reads at most `β_X (b − a) − x` bits outside `S` from outside `X`.

So no run within the limits both uses `x` in `S` and more than the rest outside it (`work_excludes`,
`state_excludes`, `ingress_excludes`).

`feasible_ofLeaves` is the feasible set at limits per leaf and per link (`Hardware.ofLeaves`), and `server_dies`,
`server_drams` and `server_store` are a server's limits over its dies, its device memories and their links, and
`server_all` the whole server's work a tick. `server_gpu_fits`: each GPU of a server is a device of the device model,
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

/-- **Work outside `S`**: if the gates of `A[X, [a, b))` in `S` do at least `x` work, those outside `S` do at most
`Φ_X (b − a) − x`. -/
theorem work_outside {𝒳 : Set (Finset L)} {π : Fin A.N → ℕ} {compute : Finset L → ℕ} (h : A.ComputeRate 𝒳 π compute)
    {X : Finset L} (hX : X ∈ 𝒳) (a b : ℕ) (S : Finset (Fin A.N)) {x : ℕ}
    (hx : x ≤ A.work π (A.region X (Set.Ico a b) ∩ S)) :
    A.work π (A.region X (Set.Ico a b) \ S) ≤ compute X * (b - a) - x := by
  have h1 := h X hX a b
  have h2 := work_inter_add_sdiff A π (A.region X (Set.Ico a b)) S
  omega

/-- **State outside `S`**: if `X` carries at least `x` bits of `S` across the tick `t`, it carries at most `M_X − x`
outside `S`. -/
theorem state_outside {𝒳 : Set (Finset L)} {M : Finset L → ℕ} {T : Set ℕ} (h : A.MemoryAt 𝒳 M T) {X : Finset L}
    (hX : X ∈ 𝒳) {t : ℕ} (ht : t ∈ T) (S : Finset (Fin A.N)) {x : ℕ} (hx : x ≤ A.stateIn X t S) :
    A.stateIn X t Sᶜ ≤ M X - x := by
  have h1 := h X hX t ht
  have h2 := state_eq_stateIn_add_compl A X t S
  omega

open Classical in
/-- **Ingress outside `S`**: if `A[X, [a, b))` reads at least `x` bits of `S` from outside `X`, it reads at most
`β_X (b − a) − x` bits outside `S` from there. -/
theorem ingress_outside {E : Type*} [Fintype E] {ends : E → L × L} {β : E → ℕ} {link : Fin A.N → Fin A.N → E}
    (h : A.Links ends β link) (X : Finset L) (a b : ℕ) (S : Finset (Fin A.N)) {x : ℕ}
    (hx : x ≤ A.width (((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ S)) :
    A.width (((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ Sᶜ) ≤
      linkCapacity ends β X * (b - a) - x := by
  have h1 := ingress_le A h X a b
  unfold Placed.ingress at h1
  have h2 := width_inter_add_compl A ((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) S
  omega

/-- **Work excluded**: no run at compute rate `Φ_X` both does `x` work in `S` and more than `Φ_X (b − a) − x` outside
it, in `A[X, [a, b))`. -/
theorem work_excludes {𝒳 : Set (Finset L)} {π : Fin A.N → ℕ} {compute : Finset L → ℕ} (h : A.ComputeRate 𝒳 π compute)
    {X : Finset L} (hX : X ∈ 𝒳) (a b : ℕ) (S : Finset (Fin A.N)) (x : ℕ) :
    ¬ (x ≤ A.work π (A.region X (Set.Ico a b) ∩ S) ∧
      compute X * (b - a) - x < A.work π (A.region X (Set.Ico a b) \ S)) :=
  fun hh => absurd (work_outside A h hX a b S hh.1) (not_le.mpr hh.2)

/-- **Memory excluded**: no run within memory `M_X` at the tick `t` carries both `x` bits of `S` and more than
`M_X − x` bits outside it across `t`. -/
theorem state_excludes {𝒳 : Set (Finset L)} {M : Finset L → ℕ} {T : Set ℕ} (h : A.MemoryAt 𝒳 M T) {X : Finset L}
    (hX : X ∈ 𝒳) {t : ℕ} (ht : t ∈ T) (S : Finset (Fin A.N)) (x : ℕ) :
    ¬ (x ≤ A.stateIn X t S ∧ M X - x < A.stateIn X t Sᶜ) :=
  fun hh => absurd (state_outside A h hX ht S hh.1) (not_le.mpr hh.2)

open Classical in
/-- **Bandwidth excluded**: no run on links of capacity `β_X` reads both `x` bits of `S` and more than
`β_X (b − a) − x` bits outside it into `A[X, [a, b))` from outside `X`. -/
theorem ingress_excludes {E : Type*} [Fintype E] {ends : E → L × L} {β : E → ℕ} {link : Fin A.N → Fin A.N → E}
    (h : A.Links ends β link) (X : Finset L) (a b : ℕ) (S : Finset (Fin A.N)) (x : ℕ) :
    ¬ (x ≤ A.width (((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ S) ∧
      linkCapacity ends β X * (b - a) - x <
        A.width (((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ Sᶜ)) :=
  fun hh => absurd (ingress_outside A h X a b S hh.1) (not_le.mpr hh.2)

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

/-- **The feasible set at limits per leaf and per link**: a gate reads gates placed no later than itself, each leaf
does at most its work a tick, every set of leaves carries at most its leaves' bits, and the links hold. -/
theorem feasible_ofLeaves [DecidableEq L] {E : Type*} (c M : L → ℕ) (ends : E → L × L) (β : E → ℕ) (u : Usage A E)
    (hc : A.Causal fun _ _ => 0) (hw : ∀ p a b, A.work u.π (A.region {p} (Set.Ico a b)) ≤ c p * (b - a))
    (hm : ∀ X t, A.state X t ≤ ∑ p ∈ X, M p) (hl : A.Links ends β u.link) :
    Feasible (Hardware.ofLeaves c M ends β) A u := by
  refine ⟨hc, computeRate_of_leaves A hw, fun k p _ hD => ?_, fun X _ t => hm X t, hl⟩
  simp [Hardware.ofLeaves] at hD

section Server

variable (n die host fabric chip dram ram queue world store bus peer net : ℕ)

/-- A server's dies together do `n` times one die's work a tick. -/
theorem server_dies :
    (server n die host fabric chip dram ram queue world store bus peer net).compute (dies n) = n * die := by
  simp [server, Hardware.ofLeaves, dies, sum_image (f := fun p : Loc n => _) fun i _ j _ h => Loc.die.inj h]

/-- A server's device memories together carry `n` times one's bits. -/
theorem server_drams :
    (server n die host fabric chip dram ram queue world store bus peer net).M (drams n) = n * dram := by
  simp [server, Hardware.ofLeaves, drams, sum_image (f := fun p : Loc n => _) fun i _ j _ h => Loc.mem.inj h]

/-- The links into a server's device memories together carry `n` times one's bits a tick. -/
theorem server_store :
    linkCapacity (server n die host fabric chip dram ram queue world store bus peer net).ends
      (server n die host fabric chip dram ram queue world store bus peer net).β (drams n) = n * store := by
  classical
  have hb : boundary (server n die host fabric chip dram ram queue world store bus peer net).ends (drams n) =
      univ.image Wire.store := by
    ext e
    cases e <;> simp [boundary, server, Hardware.ofLeaves, Wire.ends, drams]
  rw [linkCapacity, hb, sum_image fun i _ j _ h => Wire.store.inj h]
  simp [server, Hardware.ofLeaves]

/-- A whole server does its `n` dies' work, its host's and its switch's a tick. -/
theorem server_all :
    (server n die host fabric chip dram ram queue world store bus peer net).compute univ = n * die + host + fabric := by
  calc (server n die host fabric chip dram ram queue world store bus peer net).compute univ
      = ∑ p : Loc n, ((∑ i : Fin n, if p = Loc.die i then die else 0) + (if p = Loc.host then host else 0)
          + if p = Loc.switch then fabric else 0) := by
        simp only [server, Hardware.ofLeaves]
        exact sum_congr rfl fun p _ => by cases p <;> simp
    _ = n * die + host + fabric := by
        rw [sum_add_distrib, sum_add_distrib, sum_comm]
        simp

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
