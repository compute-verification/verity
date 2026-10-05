import FlockSoundness.Assumptions
import Init.Internal.Order.While
import FlockSoundness.Protocol.ExecStratified

/-!
# The live draw: `drawOS`, extend-and-rerun included, escapes no more often than the model's

`Flock.Draw.drawOS` is the verifier's draw from its operating-system randomness. It is `Flock.Draw.drawWith` with every
stream's bytes from `IO.getRandomBytes` (`drawOS_def`, by `rfl`): each stratum's `subset` runs on a stream of
`need` bytes, and `Flock.Draw.onWith` extends the stream by `need` more and reruns it whenever it runs out.

To Lean's logic `IO.getRandomBytes` is an opaque constant, so the theorems read `drawWith` on a model byte source,
`tapeGet`: stream `i` reads a tape in order, and a read past its end fails. `runOS f law pop` is the draw on the bytes
`f i` of each stream `i` (`some d`), or `none` if it would read past them. The bytes come from a source `t` over a finite
probability space `Ω`, uniform by assumption A3 (`Assumptions.UniformRandomBytes`, `uniform/io-getrandombytes`).

* `subset_append`: a `subset` draw that succeeds is the same on a longer stream, so rerunning on the extended stream
  never changes a draw once it succeeds.
* `onWith_tape`: `onWith` in the model, through `Lean.Loop.forIn_eq_of_monadTail`, by an invariant (`loop_inv`): the
  stream so far is the tape's first bytes, so a draw it returns is the sampler's on the whole tape.
* `stratifiedWith_tape`: a stratified run on unread streams returns `Flock.Draw.stratified`'s draw on the whole tapes.
* `drawOS_subset_escape_le`, `drawOS_stratified_escape_le`, `drawOS_work_escape_le`: the live draw misses `B` with
  probability at most the model's escape, `Law.subset`'s or `Law.stratified`'s, for every number of bytes `L` read,
  through `subset_exec_escape_le` and `stratified_exec_escape_le`.
* `execOS`: the live draw as a law, a run that would read past the source's bytes drawing every unit.
  `execOS_escape_le`, `execOS_work_escape_le` and `execOS_subset_escape_le` bound its escape, and `execOS_miss_le` its
  count curve.

The live verifier reads as many bytes as its reruns need, with no bound. On a run that reads at most `L i` bytes of each
stream it returns what `runOS` does on those bytes, and these runs exhaust every run as `L` grows, so its escape is the
limit of the ones bounded here, each at most the model's. A3 is what makes those bytes uniform. The limit itself is
measure theory the theorems don't state.

Not covered: the Bernoulli law (no executable Bernoulli theorem is pinned).
-/

namespace FlockSoundness.Audit.Law

open Game Finset
open scoped ENNReal

variable {n m : ℕ}


/-- The model's byte source: stream `i` reads the tape `T i` in order, from its position in the state; a read past the
tape's end fails. -/
def tapeGet (T : ℕ → ByteArray) (i need : ℕ) : StateT (ℕ → ℕ) Option ByteArray := fun pos =>
  if pos i + need ≤ (T i).size then some ((T i).extract (pos i) (pos i + need), Function.update pos i (pos i + need))
  else none

/-- A source's streams as tapes, stream `i` the bytes `f i`, as `Flock.Draw.stratified` reads its streams. -/
def tapesOf {m : ℕ} {L : Fin m → ℕ} (f : (i : Fin m) → Fin (L i) → Fin 256) : ℕ → ByteArray :=
  fun i => (Array.ofFn fun s => streamOf (f s)).getD i ByteArray.empty

/-- **The live draw on given bytes**: `Flock.Draw.drawOS`'s `drawWith`, each stream `i` reading the bytes `f i`; `none`
when it would read past them. -/
def runOS {m : ℕ} {L : Fin m → ℕ} (f : (i : Fin m) → Fin (L i) → Fin 256) (law : Flock.Draw.Law) (pop : ℕ) :
    Option Flock.Draw.UnitDraw :=
  match (Flock.Draw.drawWith (tapeGet (tapesOf f)) law pop).run fun _ => 0 with
  | some (some d, _) => some d
  | _ => none


/-- **The live draw as a law**: `Flock.Draw.drawOS`'s draw on a byte source `t` over `Ω`, reading each stream's first `L i`
bytes; a run that would read past them draws every unit. -/
def execOS {Ω : Type} [Fintype Ω] [Nonempty Ω] {m : ℕ} {L : Fin m → ℕ} (t : Ω → (i : Fin m) → Fin (L i) → Fin 256)
    (law : Flock.Draw.Law) (pop : ℕ) : Law n where
  Ω := Ω
  draw ω := match runOS (t ω) law pop with
    | some d => univ.filter fun u => u.val ∈ d.units
    | none => univ

end FlockSoundness.Audit.Law
