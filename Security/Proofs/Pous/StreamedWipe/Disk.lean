import Definitions.Pous.Guarantees.StreamedWipeDisk
import Proofs.Pous.StreamedWipe.Server
import Proofs.Pous.StreamedWipe.Witnesses

/-!
# The streamed-label wipe with a server's disks in the fill

`StreamedWipeMeetsDisk` is `streamed_wipe_meets_server` on the one session over the server's tiers and its disks:
`n + f` labels on `M_s = M + 8·D` bits, whose `R` is the other tiers' `M − 512·n` and the disks' `8·D − 512·f`. Its
honest and nontrivial lemmas are at the session on vy-dc-h200-2's card (`StreamedWipe.h200`) and its disks as measured
(`StreamedWipe.h200Disks`), in the pattern of `Security/Proofs/Pous/StreamedWipe/Witnesses.lean`.
-/

namespace PousStreamedWipe

open Pous Pous.StreamedWipe

/-- **`StreamedWipeMeetsDisk`**: `streamed_wipe_meets_server` at `M_s = M + 8·D = 512·(n + f) + R` with
`R = (M − 512·n) + (8·D − 512·f)`. -/
theorem streamed_wipe_meets_disk : Pous.Guarantees.StreamedWipeMeetsDisk :=
  fun M n D f hM hf hn r k hr hk q E κ prf P hq =>
    streamed_wipe_meets_server (M + 8 * D) (n + f) (M - 512 * n + (8 * D - 512 * f)) (by omega) hn r k hr hk
      q E κ prf P hq

end PousStreamedWipe

namespace Pous.Claims

open Pous Pous.StreamedWipe

/-- **`StreamedWipeMeetsDisk`'s honest lemma**: its antecedents hold at the session on vy-dc-h200-2's card
(`h200`: `n` labels in `M_s` bits, `k` reads at `r`) and its disks as measured (`h200Disks`): the card's fill fits it,
the disks' fill fits them, the session has labels, and `(r − 1)^k · 2^129 ≤ r^k`. -/
theorem StreamedWipeMeetsDisk_meets :
    512 * h200.n ≤ h200.Ms ∧ 64 * h200Disks.labels ≤ h200Disks.bytes ∧ 0 < h200.n + h200Disks.labels ∧
      0 < h200.r ∧ (h200.r - 1) ^ h200.k * 2 ^ 129 ≤ h200.r ^ h200.k := by
  decide +kernel

/-- **`StreamedWipeMeetsDisk`'s nontrivial lemma**: it is false at `acceptAll`, at the session on vy-dc-h200-2's card
and disks, for `keepAll` keeping `B(n + f, r, k, R)` bits. -/
theorem StreamedWipeMeetsDisk_fails : ¬ StreamedWipeMeetsDisk acceptAll := fun h =>
  keepAll_not_within (by decide +kernel) (by decide +kernel) (noKey (h200.n + h200Disks.labels))
    (h h200.Ms h200.n h200Disks.bytes h200Disks.labels (by decide +kernel) (by decide +kernel) (by decide +kernel)
      h200.r h200.k (by decide +kernel) (by decide +kernel)
      (serverBound (h200.n + h200Disks.labels) h200.r h200.k
        (h200.Ms - 512 * h200.n + (8 * h200Disks.bytes - 512 * h200Disks.labels))) 0 0
      (noKey (h200.n + h200Disks.labels)) (keepAll _ _ _ _ _ _ (by decide +kernel)) le_rfl)

end Pous.Claims
