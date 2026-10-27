import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe on one server: its threshold, and the 5 GB point

What the per-server statements of the streamed-label wipe read
(`Security/Definitions/Pous/Guarantees/StreamedWipeServer.lean`). In the physical model
(`Security/Definitions/Accounting/Physical.lean`) a server `s` is a location whose state is at most its *assumed memory
capacity* `M_s` bits at every tick (`Placed.Memory`), a public parameter over every memory tier it has: each GPU's HBM,
the host's RAM, its disks. The memory challenger fills `n` labels of 512 bits (64 bytes) into it, and
`R = M_s − 512·n` is every bit of `M_s` the fill doesn't reach, on any tier: the boot disk, the GPU contexts, a
driver's reserve, the host OS. At ratio `r` the fill's share of what the server keeps through the wipe is about
`(n/r)·(512 + log₂(e·r))` bits (`fillShare`); `R` and the egress count in full.

The claims (`Security/Definitions/Pous/Claims/StreamedWipeServer.lean`) take the verdict as a parameter (`Verdict`,
with `winRealBy` and `wipeDistinguisherBy` the game's `winReal` and `wipeDistinguisher` under it), so they can be
stated false at `acceptAll`; and they are met at a deployment (`Deployment`), the H200 as measured (`h200`).
-/

namespace Pous.StreamedWipe

open scoped ENNReal

/-- `m = ⌈n/r⌉ − 1`: the most labels of the fill that a prover answering more than `n − ⌈n/r⌉` of them can lack. One
that answers at most `n − ⌈n/r⌉` is caught by `k` reads with `(1 − 1/r)^k ≤ 2^-129`. -/
def missing (n r : ℕ) : ℕ := (n + r - 1) / r - 1

/-- `m·(⌊log₂⌊3n/m⌋⌋ + 1)`, at least `log₂ C(n, m)`: the bits that say which `m` labels of `n` are missing
(`C(n, m) ≤ (e·n/m)^m < 2^(m·(⌊log₂⌊3n/m⌋⌋ + 1))`). It is `0` at `m = 0`. -/
def countBits (n m : ℕ) : ℕ := m * (Nat.log2 (3 * n / m) + 1)

/-- **The fill's share** of what a server keeps through a wipe of `n` labels at ratio `r` with `k` reads:
`512·m + m·(⌊log₂⌊3n/m⌋⌋ + 1) + ⌊log₂ k⌋ + 130` bits, `m = ⌈n/r⌉ − 1` (`missing`). That is the missing labels' bits,
the bits that name them (`countBits`), `⌊log₂ k⌋ + 1` for the read that catches the prover, and `129` for the error
`2^-129`. -/
def fillShare (n r k : ℕ) : ℕ := 512 * missing n r + countBits n (missing n r) + Nat.log2 k + 130

/-- **`B(n, r, k, R)`**: what a server of assumed memory capacity `M_s = 512·n + R` keeps at most through a wipe of `n`
labels at ratio `r` with `k` reads, before its egress: the fill's share plus `R`, every bit outside the fill. -/
def serverBound (n r k R : ℕ) : ℕ := fillShare n r k + R

/-- **The 5 GB point**, the wipe's operating point: `B(n, r, k, R) ≤ 4·10^10` bits, so the server
keeps at most 5 GB of its assumed memory capacity plus its egress. -/
def OperatingPoint (n r k R : ℕ) : Prop := serverBound n r k R ≤ 40000000000

/-- A verdict of the wipe: whether the memory challenger accepts the prover `P` on prior content `z`, labels `L` and
challenges `c`. The wipe's own is `Accepts` (every answer is its label); `acceptAll` accepts everything. -/
abbrev Verdict : Type :=
  ∀ {N k b q M E : ℕ}, Prover N k b q M E → Bits q → (Fin N → Bits b) → (Fin k → Fin N) → Prop

/-- The verdict that accepts every prover, against which the wipe's guarantees must fail. -/
def acceptAll : Verdict := fun _ _ _ _ => True

/-- `wipeDistinguisher` under the verdict `V`: coins `(z, c)`, labels `L`, and it outputs whether `V` accepts `P` and
`P` keeps `z`. At `V = Accepts` it is `wipeDistinguisher P`. -/
def wipeDistinguisherBy (V : Verdict) {N k b q M E : ℕ} (P : Prover N k b q M E) :
    (Bits q × (Fin k → Fin N)) × (Fin N → Bits b) → Prop :=
  fun x => V P x.1.1 x.2 x.1.2 ∧ Keeps P x.1.1 x.2 x.1.2

/-- `winReal` under the verdict `V`: the probability that `V` accepts `P` and `P` keeps `z`, with the labels `prf K`
under a uniform key `K`. At `V = Accepts` it is `winReal prf P`. -/
noncomputable def winRealBy (V : Verdict) {N k b q M E κ : ℕ} (prf : Bits κ → Fin N → Bits b)
    (P : Prover N k b q M E) : ℝ≥0∞ :=
  pr (fun x : (Bits q × (Fin k → Fin N)) × Bits κ => wipeDistinguisherBy V P (x.1, prf x.2))

/-- A wipe as deployed on one server: the fill's `n` labels, the server's assumed memory capacity `Ms` in bits over
every tier it declares, and the read count `k` and ratio `r` the memory challenger publishes. -/
structure Deployment where
  n : ℕ
  Ms : ℕ
  k : ℕ
  r : ℕ

/-- `R = M_s − 512·n`: every bit of the deployment's capacity outside the fill. -/
def Deployment.R (d : Deployment) : ℕ := d.Ms - 512 * d.n

/-- **One H200 on vy-dc-h200-2, as measured.** An honest wipe accepted on the whole card measured both numbers. The
card (NVIDIA H200, GPU-7d104033) holds 143,771 MiB: `M_s = 143,771 · 2^20 · 8 = 1,206,038,560,768` bits. The fill
was 4,425 segments of 32 MiB, `n = 2,319,974,400` labels (148,478,361,600 bytes). So `R` is 18,211,667,968 bits
(2.276 GB), the driver's reserve and every context included. The read count is the deployed `k = 5500`, the
`wipe_party --reads 5500` pinned for this card on vy-auditor-2 (the measuring wipe read 2152), at its published
ratio `r = 62`, the largest `r` with `(r − 1)^k · 2^129 ≤ r^k`. -/
def h200 : Deployment where
  n := 2319974400
  Ms := 1206038560768
  k := 5500
  r := 62

end Pous.StreamedWipe
