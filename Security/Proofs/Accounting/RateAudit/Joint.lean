import Mathlib
import Proofs.Accounting.Device
import Proofs.Accounting.RateAudit.Audit

/-!
# Proof that the rate audit is a joint run's bandwidth guarantee, with the rebuilds charged as compute

At a value of the other audits' coins, the run's bandwidth audit is a device of `RateAudit`'s class at `S = S_chip`,
`L = ∑ cap_i · T_bw` and `q = rate_oracle · T_bw`. On a bandwidth coin where the compute account did at least `W`, the
game's compute, covered by the other accounts' work, is within `F·T − W` (`Run.work_add_othersWork`), and the game's
footprint is at most its ingress, which the bandwidth account's reads cover. So the joint event is inside `RateAudit`'s
at `F = F·T − W`.
-/

namespace Accounting.SecurityProofs

open Accounting Accounting.Rate

theorem RateOfAudit : Accounting.Guarantees.RateOfAudit := by
  intro N b h ρ s k cTag cEnc lam W p ι α ΩW ΩS _ _ _ d r real x s'
  refine (Bandwidth.pr_mono (fun y hy => ?_)).trans
    (RateAudit N b h ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len) (d.rate .oracle * (r.win .bandwidth).len) s k
      (d.priced p * (r.win .compute).len - W) cTag cEnc lam (real.dev (x, s')))
  obtain ⟨hacc, hW, hread⟩ := hy
  refine ⟨(real.accepts_iff x y s').1 hacc, ?_, ?_⟩
  · have h1 := real.compute_le x y s'
    have h2 := r.work_add_othersWork p (x, y, s')
    omega
  · have h1 := Chip.card_addrs_le ((real.dev (x, s')).world y.1 y.2.1.1 y.2.1.2)
      ((real.dev (x, s')).chip ((real.dev (x, s')).state y.1))
    have h3 : (real.dev (x, s')).footprint y.1 y.2.1.1 y.2.1.2 * cEnc ≤ r.read (x, y, s') * cEnc :=
      Nat.mul_le_mul_right _ (h1.trans (real.ingress_le x y s'))
    exact lt_of_le_of_lt h3 hread

end Accounting.SecurityProofs
