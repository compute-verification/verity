import Proofs.Flock.VBridge.ResLayer
import Proofs.Flock.Verifier.HmNets

/-!
# The opening unit reads its row and sibling columns (`ReadCol`)

`OpenCols`' second half asks that a computed row of each level's opening unit read each of its first
`128 · LANES + 512 · H` columns. A unit the verifier accepted as `rec-open/l<LANES>-h<H>` is the net `netOf` builds from
`RecOpen.unitLaid LANES H` (`RecOpen.check_ok`), so the statements are at that net, with `unitLog ≤ 32`, under which its
rows are `Rows.ofNet`'s:

* `OpenReadCol LANES H`: every column `c < 128 · LANES + 512 · H` is read (`ReadCol`), at `valid LANES H`. It is its
  two halves (`openReadCol_of`):
* `OpenRowRead`: the row columns, `c < 128 · LANES`. Residual 0's Karatsuba leaves include `(coef_l[j], row_l[j])`, an
  AND of two input bits, so a row reads row bit `128 l + j` in B.
* `OpenSibRead`: the sibling columns, `128 · LANES + 512 · h + i` for `h < H` and `i < 512`. At level `h`, `swap` ANDs
  direction bit `h` (an input bit, as `H ≤ 32`) with `cur ⊕ s`, whose support holds sibling bit `i`, since `cur` reads
  no sibling of level `h`.
-/

namespace FlockC3.Reps

open FlockSoundness FlockSoundness.Audit
open FlockVBridge.OfVStar (ReadCol)

/-- **The opening unit reads its row columns**: at the net of `unitLaid LANES H`, a computed row reads each of the
first `128 · LANES` columns. -/
def OpenRowRead (LANES H : ℕ) : Prop :=
  ∀ {net : Flock.Net} (hU : Built net),
    Flock.HmNets.netOf (Flock.RecOpen.unitName LANES H) (Flock.RecOpen.unitLaid LANES H) = .ok net →
      net.unitLog ≤ 32 → ∀ c < 128 * LANES, ReadCol (Rows.ofNet hU) c

/-- **The opening unit reads its sibling columns**: at the net of `unitLaid LANES H` with `valid LANES H`, a computed
row reads each of the `512` columns of each of the `H` sibling blocks of `aux`. -/
def OpenSibRead (LANES H : ℕ) : Prop :=
  ∀ {net : Flock.Net} (hU : Built net),
    Flock.HmNets.netOf (Flock.RecOpen.unitName LANES H) (Flock.RecOpen.unitLaid LANES H) = .ok net →
      Flock.RecOpen.valid LANES H = true → net.unitLog ≤ 32 →
        ∀ h < H, ∀ i < 512, ReadCol (Rows.ofNet hU) (128 * LANES + 512 * h + i)

/-- **The opening unit reads its row and sibling columns** (`OpenCols`' `ReadCol` half, at one unit). -/
def OpenReadCol (LANES H : ℕ) : Prop :=
  ∀ {net : Flock.Net} (hU : Built net),
    Flock.HmNets.netOf (Flock.RecOpen.unitName LANES H) (Flock.RecOpen.unitLaid LANES H) = .ok net →
      Flock.RecOpen.valid LANES H = true → net.unitLog ≤ 32 →
        ∀ c < 128 * LANES + 512 * H, ReadCol (Rows.ofNet hU) c

/-- **`OpenReadCol` is its two halves**: a column past the row's is `128 · LANES + 512 · h + i`. -/
theorem openReadCol_of {LANES H : ℕ} (hr : OpenRowRead LANES H) (hs : OpenSibRead LANES H) : OpenReadCol LANES H := by
  intro net hU hn hv hu c hc
  by_cases hcr : c < 128 * LANES
  · exact hr hU hn hu c hcr
  · have h1 : (c - 128 * LANES) / 512 < H := by omega
    have h2 : (c - 128 * LANES) % 512 < 512 := Nat.mod_lt _ (by norm_num)
    have h3 : c = 128 * LANES + 512 * ((c - 128 * LANES) / 512) + (c - 128 * LANES) % 512 := by omega
    have := hs hU hn hv hu _ h1 _ h2
    rwa [← h3] at this

end FlockC3.Reps
