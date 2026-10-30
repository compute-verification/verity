import Definitions.Accounting.Guarantees.BandwidthAudit
import Proofs.Accounting.Prob
import Proofs.Accounting.Bandwidth.AuditRO

/-!
# Proof that the bandwidth audit's guarantee is a joint run's bandwidth guarantee

At a value of the other audits' coins, the game's ingress is covered by the bandwidth account's reads, so on a bandwidth
coin where the audit accepts with those reads (or a bound `B` on them on every bandwidth coin) below the certificate,
the game's device is accepted with its ingress, plus `S_chip + c + G·b + λ`, short of `k·b` bits. AuditRO bounds that
event in `ℝ≥0∞`, and with `N > 0` its bound is finite and is the certificate's error as a real number
(`pr_short_le_error`). The sure bound gives `BandwidthOfAudit`, the reads at the coins drawn `BandwidthOfAuditPerRun`.
-/

namespace Accounting.SecurityProofs

open Accounting Pouw
open scoped ENNReal

theorem pr_short_le_error {N b h s G k lam : ℕ} {ι α ΩW ΩS : Type} [Fintype ι] [Fintype α] {d : Device ι}
    {r : Run d α ΩW (BandwidthCoins N b h s) ΩS} (hN : 0 < N) (real : RealizesBandwidth G r) (x : ΩW) (s' : ΩS) :
    pr (fun y : BandwidthCoins N b h s => Bandwidth.Accepts (real.dev (x, s')) y.1 y.2.1 y.2.2 ∧
        (real.dev (x, s')).ingress y.1 y.2.1 + d.chip + ∑ i, d.cap i * (r.win .bandwidth).len + G * b + lam <
          k * b) ≤
      (((k - 1 : ℕ) : ℝ) / N) ^ s + (s : ℝ) / 2 ^ h +
        (((d.rate .oracle * (r.win .bandwidth).len).choose k : ℕ) : ℝ) / 2 ^ lam := by
  set c := ∑ i, d.cap i * (r.win .bandwidth).len
  set q := d.rate .oracle * (r.win .bandwidth).len with hq
  have hN0 : (N : ℝ≥0∞) ≠ 0 := by exact_mod_cast hN.ne'
  have h2 : ∀ n : ℕ, (2 : ℝ≥0∞) ^ n ≠ 0 := fun n => pow_ne_zero _ two_ne_zero
  have hfin : (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam ≠ ⊤ :=
    ENNReal.add_ne_top.2 ⟨ENNReal.add_ne_top.2 ⟨ENNReal.pow_ne_top (ENNReal.div_ne_top (ENNReal.natCast_ne_top _) hN0),
      ENNReal.div_ne_top (ENNReal.natCast_ne_top _) (h2 h)⟩, ENNReal.div_ne_top (ENNReal.natCast_ne_top _) (h2 lam)⟩
  calc _ = (Pous.pr (fun y : BandwidthCoins N b h s => Bandwidth.Accepts (real.dev (x, s')) y.1 y.2.1 y.2.2 ∧
          (real.dev (x, s')).ingress y.1 y.2.1 + d.chip + c + G * b + lam < k * b)).toReal := pr_eq_toReal _
    _ ≤ ((((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam).toReal :=
        ENNReal.toReal_mono hfin
          (Accounting.SecurityProofs.Remote.AuditRO N b h d.chip c G q s k lam (real.dev (x, s')))
    _ = _ := by
        rw [ENNReal.toReal_add (ENNReal.add_ne_top.1 hfin).1 (ENNReal.add_ne_top.1 hfin).2,
          ENNReal.toReal_add (ENNReal.add_ne_top.1 (ENNReal.add_ne_top.1 hfin).1).1
            (ENNReal.add_ne_top.1 (ENNReal.add_ne_top.1 hfin).1).2]
        simp only [ENNReal.toReal_div, ENNReal.toReal_pow, ENNReal.toReal_natCast, ENNReal.toReal_ofNat, hq]

theorem BandwidthOfAudit : Accounting.Guarantees.BandwidthOfAudit := by
  intro N b h s G k lam ι α ΩW ΩS _ _ _ _ d r hN real x s' B _ hread
  set c := ∑ i, d.cap i * (r.win .bandwidth).len
  refine le_trans (pr_mono ?_) (pr_short_le_error hN real x s')
  rintro y ⟨hacc, hBQ⟩
  refine ⟨(real.accepts_iff x y s').1 hacc, ?_⟩
  have hin : ((real.dev (x, s')).ingress y.1 y.2.1 : ℝ) ≤ B :=
    (Nat.cast_le.2 (real.ingress_le x y s')).trans (hread y)
  have : (((real.dev (x, s')).ingress y.1 y.2.1 + d.chip + c + G * b + lam : ℕ) : ℝ) < ((k * b : ℕ) : ℝ) := by
    rw [Nat.cast_add, Nat.cast_add, Nat.cast_add, Nat.cast_add]
    linarith
  exact_mod_cast this

theorem BandwidthOfAuditPerRun : Accounting.Guarantees.BandwidthOfAuditPerRun := by
  intro N b h s G k lam ι α ΩW ΩS _ _ _ _ d r hN real x s'
  set c := ∑ i, d.cap i * (r.win .bandwidth).len
  refine le_trans (pr_mono ?_) (pr_short_le_error hN real x s')
  rintro y ⟨hacc, hQ⟩
  refine ⟨(real.accepts_iff x y s').1 hacc, ?_⟩
  have hin : ((real.dev (x, s')).ingress y.1 y.2.1 : ℝ) ≤ r.read (x, y, s') :=
    Nat.cast_le.2 (real.ingress_le x y s')
  have : (((real.dev (x, s')).ingress y.1 y.2.1 + d.chip + c + G * b + lam : ℕ) : ℝ) < ((k * b : ℕ) : ℝ) := by
    rw [Nat.cast_add, Nat.cast_add, Nat.cast_add, Nat.cast_add]
    linarith
  exact_mod_cast this

end Accounting.SecurityProofs
