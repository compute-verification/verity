import Definitions.Accounting.Bridge
import Proofs.Accounting.Actual.Physical

/-!
# The actual circuit's schedule fits the device

`fits`: the schedule a run induces (`Actual.Run.schedule`) fits the device when each op class's gates in the device
(`chip ∪ mem`) run at most at its rate, the states of `chip` and `mem` (what their leaves hold, and what is in flight
between them) are at most the device's state and memory (`Run.Memory`), and the memory link and side channels carry at
most the device's bandwidth and side capacities. Summed over accounts, each charge is
the device's whole (each gate has one account, `sum_width_filter`), and that is what compute rate, memory and links
bound. `fits_of_feasible` takes links from the feasible set and memory from the run (`Feasible`'s `Memory` bounds the
circuit's `state` only); the per-class rates stay a hypothesis, as `Feasible` prices compute one way only.

The converse, that the schedule charges everything an account's gates use: `covers_ingress`, when every wire into
`chip` from outside it runs on the memory link or a side channel toward its device end, the chip's ingress from an
account's gates at a tick is at most that account's `read` and `side` charges (a union bound); `covers_work`, when the
work's pricing is `p` on each op class and nothing on gates of none, an account's work over a window is the schedule's
`work p` (the window's gates split by tick, `sum_fiberwise_of_maps_to`). State needs nothing more: the device is
`chip ∪ mem`, whose states the schedule charges.
-/

namespace Accounting.SecurityProofs.Actual

open Finset

variable {L : Type*} (A : Placed L)

open Classical in
theorem sum_width_filter {α : Type} [Fintype α] (S : Finset (Fin A.N)) (acct : Fin A.N → α) :
    ∑ a, A.width (S.filter fun s => acct s = a) = A.width S := by
  unfold Verity.Circuit.width
  convert Finset.sum_fiberwise S acct A.w

open Classical in
theorem sum_card_filter {α : Type} [Fintype α] (S : Finset (Fin A.N)) (acct : Fin A.N → α) :
    ∑ a, (S.filter fun s => acct s = a).card = S.card := by
  convert Finset.sum_fiberwise S acct (fun _ => 1) <;> simp

open Classical in
/-- **The induced schedule fits the device.** -/
theorem fits {ι α E : Type} [Fintype α] (R : Actual.Run L) (d : Device ι) (chip mem : Finset L)
    (cls : Fin R.A.N → Option Op) (acct : Fin R.A.N → α) {𝒳 : Set (Finset L)} {M : Finset L → ℕ} {ends : E → L × L}
    {β : E → ℕ} {link : Fin R.A.N → Fin R.A.N → E} (hbm : E × L) (side : ι → E × L)
    (hΦ : ∀ o, R.A.ComputeRate {chip ∪ mem} (R.A.classPrice cls o) (fun _ => d.rate o))
    (hM : R.Memory 𝒳 M) (hchip : chip ∈ 𝒳) (hmem : mem ∈ 𝒳) (hMc : M chip ≤ d.chip) (hMm : M mem ≤ d.mem)
    (hL : R.A.Links ends β link) (hbw : β hbm.1 ≤ d.bw) (hcap : ∀ i, β (side i).1 ≤ d.cap i) :
    (R.schedule chip mem cls acct link hbm side).Fits d := by
  classical
  intro t
  have one : t + 1 - t = 1 := by omega
  refine ⟨fun o => ?_, ?_, fun i => ?_, ?_, ?_⟩
  · simp only [Actual.Run.schedule]
    rw [sum_card_filter, Finset.card_filter]
    simpa [Placed.work, Placed.classPrice, one] using hΦ o (chip ∪ mem) rfl t (t + 1)
  · simp only [Actual.Run.schedule]
    rw [sum_width_filter]
    exact ((hL.2 hbm.1 hbm.2 t (t + 1)).1.trans (by rw [one, mul_one])).trans hbw
  · simp only [Actual.Run.schedule]
    rw [sum_width_filter]
    exact ((hL.2 (side i).1 (side i).2 t (t + 1)).1.trans (by rw [one, mul_one])).trans (hcap i)
  · simp only [Actual.Run.schedule]
    rw [sum_width_filter]
    exact (hM chip hchip (t + 1)).trans hMc
  · simp only [Actual.Run.schedule]
    rw [sum_width_filter]
    exact (hM mem hmem (t + 1)).trans hMm

open Classical in
/-- **A feasible run's schedule fits the device**, when its locations `chip` and `mem` and its links are the device's,
its memory per run is the hardware's and each op class runs at most at its rate. `Feasible`'s `Memory` bounds only the
circuit's `state` (`memory_of_mem` derives it from the run's), so the run's memory is a hypothesis of its own. -/
theorem fits_of_feasible {ι α E : Type} [Fintype α] (R : Actual.Run L) {H : Hardware L E} {ann : Annotation R.A E}
    (hF : Feasible H R.A ann) (hM : R.Memory H.units H.M) (d : Device ι) (chip mem : Finset L)
    (cls : Fin R.A.N → Option Op) (acct : Fin R.A.N → α) (hbm : E × L) (side : ι → E × L)
    (hΦ : ∀ o, R.A.ComputeRate {chip ∪ mem} (R.A.classPrice cls o) (fun _ => d.rate o))
    (hchip : chip ∈ H.units) (hmem : mem ∈ H.units) (hMc : H.M chip ≤ d.chip) (hMm : H.M mem ≤ d.mem)
    (hbw : H.β hbm.1 ≤ d.bw) (hcap : ∀ i, H.β (side i).1 ≤ d.cap i) :
    (R.schedule chip mem cls acct ann.link hbm side).Fits d :=
  fits R d chip mem cls acct hbm side hΦ hM hchip hmem hMc hMm hF.2.2.2.2.1 hbw hcap

open Classical in
/-- **The schedule charges the chip's ingress**: when every wire into `chip` from outside it runs on the memory link
toward `hbm.2` or on a side channel toward its device end, what `chip` reads at `t` from outside it, from account `a`'s
gates, is at most `a`'s `read` and `side` charges at `t`. -/
theorem covers_ingress {ι α E : Type} [Fintype ι] (R : Actual.Run L) (chip mem : Finset L)
    (cls : Fin R.A.N → Option Op) (acct : Fin R.A.N → α) (link : Fin R.A.N → Fin R.A.N → E) (hbm : E × L)
    (side : ι → E × L)
    (hcov : ∀ g, ∀ s ∈ R.A.reads g, R.A.x g ∈ chip → R.A.x s ∉ chip →
      (link g s = hbm.1 ∧ R.A.x g = hbm.2) ∨ ∃ i, link g s = (side i).1 ∧ R.A.x g = (side i).2) (t : ℕ) (a : α) :
    R.A.width (((R.A.into (R.A.region chip (Set.Ico t (t + 1)))).filter fun s => R.A.x s ∉ chip).filter
        fun s => acct s = a) ≤
      (R.schedule chip mem cls acct link hbm side t).read a +
        ∑ i, (R.schedule chip mem cls acct link hbm side t).side a i := by
  simp only [Actual.Run.schedule]
  set D := (R.A.readOn link hbm.1 hbm.2 (Set.Ico t (t + 1))).filter fun s => acct s = a
  set Q := fun i => (R.A.readOn link (side i).1 (side i).2 (Set.Ico t (t + 1))).filter fun s => acct s = a
  unfold Verity.Circuit.width
  have point : ∀ s ∈ ((R.A.into (R.A.region chip (Set.Ico t (t + 1)))).filter fun s => R.A.x s ∉ chip).filter
      (fun s => acct s = a), R.A.w s ≤ (if s ∈ D then R.A.w s else 0) + ∑ i, if s ∈ Q i then R.A.w s else 0 := by
    intro s hs
    simp only [Finset.mem_filter] at hs
    obtain ⟨⟨hin, hx⟩, ha⟩ := hs
    obtain ⟨-, g, hg, hsg⟩ := (mem_into R.A).1 hin
    obtain ⟨-, hgx, hgt⟩ := (mem_region R.A).1 hg
    have hne : R.A.x s ≠ R.A.x g := fun h => hx (h ▸ hgx)
    rcases hcov g s hsg hgx hx with ⟨hl, hy⟩ | ⟨i, hl, hy⟩
    · have : s ∈ D := by
        simp only [D, Placed.readOn, Finset.mem_filter, Finset.mem_univ, true_and]
        exact ⟨⟨g, hsg, hne, hl, hy, hgt⟩, ha⟩
      simp [this]
    · have : s ∈ Q i := by
        simp only [Q, Placed.readOn, Finset.mem_filter, Finset.mem_univ, true_and]
        exact ⟨⟨g, hsg, hne, hl, hy, hgt⟩, ha⟩
      exact le_add_left (Finset.single_le_sum (f := fun i => if s ∈ Q i then R.A.w s else 0) (fun _ _ => Nat.zero_le _)
        (Finset.mem_univ i) |>.trans' (by simp [this]))
  refine (Finset.sum_le_sum point).trans ?_
  rw [Finset.sum_add_distrib, Finset.sum_comm]
  refine add_le_add ?_ (Finset.sum_le_sum fun i _ => ?_) <;> rw [Finset.sum_ite_mem] <;>
    exact Finset.sum_le_sum_of_subset Finset.inter_subset_right

open Classical in
/-- **The schedule charges the work**: when the pricing `π` is `p` on each op class and nothing on gates of none,
account `a`'s work over a window of the device's gates is the schedule's `work p` for `a`. -/
theorem covers_work {ι α E : Type} (R : Actual.Run L) (chip mem : Finset L) (cls : Fin R.A.N → Option Op)
    (acct : Fin R.A.N → α) [DecidableEq α] (link : Fin R.A.N → Fin R.A.N → E) (hbm : E × L) (side : ι → E × L)
    {p : Price} {π : Fin R.A.N → ℕ} (hπ : ∀ g, R.A.x g ∈ chip ∪ mem → π g = (cls g).elim 0 p) (w : Window) (a : α) :
    R.A.work π ((R.A.region (chip ∪ mem) (Set.Ico w.start (w.start + w.len))).filter fun g => acct g = a) =
      (R.schedule chip mem cls acct link hbm side).work w a p := by
  classical
  set D := (R.A.region (chip ∪ mem) (Set.Ico w.start (w.start + w.len))).filter fun g => acct g = a
  have hR : ∀ g ∈ D, R.A.x g ∈ chip ∪ mem ∧ R.A.t g ∈ w.ticks := fun g hg => by
    obtain ⟨hg, -⟩ := Finset.mem_filter.1 hg
    obtain ⟨-, hx, ht⟩ := (mem_region R.A).1 hg
    exact ⟨hx, Finset.mem_Ico.2 ht⟩
  have split : ∀ g ∈ D, π g = ∑ o, p o * if cls g = some o then 1 else 0 := fun g hg => by
    rw [hπ g (hR g hg).1]
    cases cls g <;> simp
  unfold Placed.work Schedule.work Schedule.ops
  rw [Finset.sum_congr rfl split, Finset.sum_comm]
  refine Finset.sum_congr rfl fun o _ => ?_
  rw [← Finset.mul_sum]
  congr 1
  rw [← Finset.sum_fiberwise_of_maps_to (g := R.A.t) (t := w.ticks) (fun g hg => (hR g hg).2)]
  refine Finset.sum_congr rfl fun u hu => ?_
  simp only [Actual.Run.schedule, Finset.sum_boole, Nat.cast_id]
  congr 1
  ext g
  simp only [D, Finset.mem_filter, mem_region, Set.mem_Ico, Window.ticks, Finset.mem_Ico] at hu ⊢
  constructor
  · rintro ⟨⟨⟨⟨hc, hx, -⟩, ha⟩, rfl⟩, ho⟩
    exact ⟨⟨⟨hc, hx, le_refl _, Nat.lt_succ_self _⟩, ho⟩, ha⟩
  · rintro ⟨⟨⟨hc, hx, h1, h2⟩, ho⟩, ha⟩
    have : R.A.t g = u := by omega
    subst this
    exact ⟨⟨⟨⟨hc, hx, hu⟩, ha⟩, rfl⟩, ho⟩

end Accounting.SecurityProofs.Actual
