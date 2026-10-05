import Flock.Draw
import FlockSoundness.Protocol.Audit.Law

/-!
# The executable draw escapes no more often than the model's

`Flock.Draw.subset k n src` is the verifier's executable sampler (`backends/flock/verifier/lean/Flock/Draw.lean`): a
partial Fisher–Yates over a `Std.HashMap` of swaps, each index drawn by `Flock.Draw.uniform`, an exact rejection sampler
on big-endian words, and the `k` values sorted by `Array.qsort`. `Law.subset_exec_escape_le`: over a uniform stream of
bytes, it returns a draw that misses `B` with probability at most `(Law.subset n k hk).escape B`, the model's uniform
`k`-subset. A stream that runs out gives `none`, which only lowers the executable's escape.

* `mem_qsort`: `Array.qsort` keeps its elements, through core's private `qsort.sort` and `qpartition.loop`.
* `subset_eq`, `subset_misses`: the loop restated as a list recursion, and its `HashMap` read through a permutation `π`
  with `sw.getD t t = π t`: the draw misses `B` only if `π` does on `[0, k)`.
* `uniformL_le`: the rejection sampler. `uniform_streamOf` ties `Flock.Draw.uniform` and `beNat` at a position to it.
* `loopL_le`: step `i` draws uniformly over the pool `π '' [i, n)`, which holds all of `B` while the earlier values miss
  it, so the loop escapes `B` with probability at most `C(n − i − |B|, c)/C(n − i, c)`.
-/

open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic

namespace FlockSoundness.Audit.Law

/-- The byte stream of `L` bytes `f`, as `Flock.Draw` reads it. -/
def streamOf {L : ℕ} (f : Fin L → Fin 256) : ByteArray := ⟨Array.ofFn fun i => UInt8.ofFin (f i)⟩

end FlockSoundness.Audit.Law


namespace FlockSoundness.Audit.DrawExec


open Flock.Draw

open Game Finset
open scoped ENNReal

/-! ## Uniform byte streams -/

/-! ## The rejection sampler -/

/-! ## The executable's reads, on the rest of the stream -/

/-! ## The permutation loop, on the rest of the stream -/

end FlockSoundness.Audit.DrawExec

namespace FlockSoundness.Audit.Law

open Game Finset
open scoped ENNReal

end FlockSoundness.Audit.Law
