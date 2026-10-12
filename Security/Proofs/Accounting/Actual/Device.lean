import Definitions.Accounting.Bridge
import Proofs.Accounting.Actual.Physical

/-!
# The actual circuit's schedule fits the device

`fits`: the schedule `A` induces (`Placed.schedule`) fits the device when each op class's gates in `dev` run at most at
its rate, `chip` and `mem` carry at most the device's state and memory, and the memory link and side channels carry at
most the device's bandwidth and side capacities. Summed over accounts, each charge is the device's whole (each gate has
one account, `sum_width_filter`), and that is what compute rate, memory and links bound. `fits_of_feasible` takes memory
and links from the feasible set; the per-class rates stay a hypothesis, as `Feasible` prices compute one way only.
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

/-- **The induced schedule fits the device.** -/
theorem fits {ι α E : Type} [Fintype α] (d : Device ι) (dev chip mem : Finset L) (cls : Fin A.N → Option Op)
    (acct : Fin A.N → α) {𝒳 : Set (Finset L)} {M : Finset L → ℕ} {ends : E → L × L} {β : E → ℕ}
    {link : Fin A.N → Fin A.N → E} (hbm : E × L) (side : ι → E × L)
    (hΦ : ∀ o, A.ComputeRate {dev} (A.classPrice cls o) (fun _ => d.rate o))
    (hM : A.Memory 𝒳 M) (hchip : chip ∈ 𝒳) (hmem : mem ∈ 𝒳) (hMc : M chip ≤ d.chip) (hMm : M mem ≤ d.mem)
    (hL : A.Links ends β link) (hbw : β hbm.1 ≤ d.bw) (hcap : ∀ i, β (side i).1 ≤ d.cap i) :
    (A.schedule dev chip mem cls acct link hbm side).Fits d := by
  classical
  intro t
  have one : t + 1 - t = 1 := by omega
  refine ⟨fun o => ?_, ?_, fun i => ?_, ?_, ?_⟩
  · simp only [Placed.schedule]
    rw [sum_card_filter, Finset.card_filter]
    simpa [Placed.work, Placed.classPrice, one] using hΦ o dev rfl t (t + 1)
  · simp only [Placed.schedule]
    rw [sum_width_filter]
    exact ((hL.2 hbm.1 hbm.2 t (t + 1)).1.trans (by rw [one, mul_one])).trans hbw
  · simp only [Placed.schedule]
    rw [sum_width_filter]
    exact ((hL.2 (side i).1 (side i).2 t (t + 1)).1.trans (by rw [one, mul_one])).trans (hcap i)
  · simp only [Placed.schedule]
    rw [sum_width_filter]
    exact (hM chip hchip (t + 1)).trans hMc
  · simp only [Placed.schedule]
    rw [sum_width_filter]
    exact (hM mem hmem (t + 1)).trans hMm

/-- **A feasible actual circuit's schedule fits the device**, when its locations `chip` and `mem` and its links are the
device's and each op class runs at most at its rate. -/
theorem fits_of_feasible {ι α E : Type} [Fintype α] {H : Hardware L E} {ann : Annotation A E} (hF : Feasible H A ann)
    (d : Device ι) (dev chip mem : Finset L) (cls : Fin A.N → Option Op) (acct : Fin A.N → α) (hbm : E × L)
    (side : ι → E × L) (hΦ : ∀ o, A.ComputeRate {dev} (A.classPrice cls o) (fun _ => d.rate o))
    (hchip : chip ∈ H.units) (hmem : mem ∈ H.units) (hMc : H.M chip ≤ d.chip) (hMm : H.M mem ≤ d.mem)
    (hbw : H.β hbm.1 ≤ d.bw) (hcap : ∀ i, H.β (side i).1 ≤ d.cap i) :
    (A.schedule dev chip mem cls acct ann.link hbm side).Fits d :=
  fits A d dev chip mem cls acct hbm side hΦ hF.2.2.2.1 hchip hmem hMc hMm hF.2.2.2.2.1 hbw hcap

end Accounting.SecurityProofs.Actual
