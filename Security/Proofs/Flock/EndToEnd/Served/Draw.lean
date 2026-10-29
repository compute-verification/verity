import Proofs.Flock.Soundness.ExecDrawOS

/-!
# The served stratified draw's tapes: the challenger's coins, cut where each stratum stops reading (trusted)

The sampled-proofs service draws a sampling block's units once, from the challenger's coins: `Select.select` runs
`SampledProofs.Draw.drawUnits`, which is `Flock.Draw.drawWith` reading one stream of coins in order through
`SampledProofs.Draw.take`. A stratified law runs its strata in turn, stratum `i`'s `subset` on `need` bytes at a time
from where stratum `i − 1` stopped, extending and rerunning until it draws (`Flock.Draw.onWith`). The live draw of the
model (`Law.runOS`, `Law.execOS`, which `StratifiedBlock` reads) reads one tape per stratum instead. These are the tapes
the served draw reads, one segment of the coins each:

* `Draws`, `used`: a stratum's sampler draws at its `j`-th read of `need` bytes from a stream's start; the bytes it
  reads up to its first draw, or the whole stream when it never draws.
* `starts`, `seg`: where each stratum starts reading the one stream of coins, and how many bytes it reads.
* `tapes`: stratum `i`'s tape is its segment of the coins `c`, then its own spare bytes `pad i`, which no draw that ends
  on the coins reads.
-/

namespace FlockSoundness.Audit.Served

open Classical

/-- A stratum's sampler draws at its `j`-th read of `need` bytes from the start of `a`: `subset` on `a`'s first
`j · need` bytes draws. -/
def Draws (s : Flock.Draw.Stratum) (a : ByteArray) (j : ℕ) : Prop :=
  1 ≤ j ∧ j * s.need ≤ a.size ∧ (Flock.Draw.subset s.k s.size (a.extract 0 (j * s.need))).isSome = true

/-- **The bytes a stratum reads from the start of `a`**: `j · need` at the first read `j` that draws (`Draws`), or all
of `a` when none does. -/
noncomputable def used (s : Flock.Draw.Stratum) (a : ByteArray) : ℕ :=
  if h : ∃ j, Draws s a j then Nat.find h * s.need else a.size

/-- **Where each stratum starts reading** the one stream `src`, the strata reading it in order: stratum `i + 1` where
stratum `i` stopped. -/
noncomputable def starts (strata : Array Flock.Draw.Stratum) (src : ByteArray) : ℕ → ℕ
  | 0 => 0
  | i + 1 => starts strata src i + used strata[i]! (src.extract (starts strata src i) src.size)

/-- The bytes stratum `i` reads of `src`, from `starts i`. -/
noncomputable def seg (strata : Array Flock.Draw.Stratum) (src : ByteArray) (i : ℕ) : ℕ :=
  used strata[i]! (src.extract (starts strata src i) src.size)

/-- Byte `j` of `c`, `0` past its end. -/
def byteAt {L : ℕ} (c : Fin L → Fin 256) (j : ℕ) : Fin 256 := if h : j < L then c ⟨j, h⟩ else 0

/-- The model's spare bytes: `L` of them for each of `m` strata. -/
abbrev Pads (m L : ℕ) := Fin m → Fin L → Fin 256

/-- **The served draw's tapes**: stratum `i`'s tape is the segment of the coins `c` it reads, `seg i` bytes from
`starts i`, then its spare bytes `pad i`. -/
noncomputable def tapes {m L : ℕ} (St : Fin m → Flock.Draw.Stratum) (c : Fin L → Fin 256) (pad : Pads m L) :
    (i : Fin m) → Fin L → Fin 256 :=
  fun i x => if x.val < seg (Array.ofFn St) (Law.streamOf c) i then
    byteAt c (starts (Array.ofFn St) (Law.streamOf c) i + x.val) else pad i x

end FlockSoundness.Audit.Served
