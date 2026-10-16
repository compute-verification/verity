import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.BigOperators.Fin

/-!
# The deployed bound receipt's sketch: a relay as a two-party protocol (trusted)

The remote protocol's bound receipt (`ro-bound-sketch-blake3/v0`, lane ra-bound-receipt) hashes a sketch of the block's
bytes against a multiplier matrix drawn from the matmul: for a tile of `r` vectors `e_i` of `m` bytes and a matrix `B`
of `r` columns of `m` bytes, the sketch is `v_i[c] = Σ_q e_i[q]·B[c][q]` and the receipt `H(σ_p, j, v)`. The matrix is
`B = R ⊕ M`: `M` the pass's fresh sign mask (one buffer for every tile of the pass, derived from the chained seed), `R`
bytes of the matmul's own values (the bound tile's partial accumulators). A location makes the right receipt only
knowing `v`, so a relay between the block side (the memory, holding the tile's bytes) and the tile side (holding `R`)
ends with one of them knowing the sketch.

This file states that relay in the communication model: a deterministic two-party protocol (`Protocol`) over the two
inputs, each step one side sending one bit of its input and the bits before it, and at the end one side outputting a
function of its input and the transcript. Public coins (the seed, the mask `M`) are fixed: a randomized protocol is
an average of these. The bytes are read in `𝔽_p` (`p = 257` for the deployed sketch, where a byte embeds
injectively); the sketch is the integer sketch's image mod `p`, which a location knowing the integer sketch knows. `ip` is the inner product on `𝔽_p^ι`, and the sketch's trace
`Σ_i v_i[i]` is `ip` of the tile's bytes and the matrix (`sketch`, with `ι = Fin r × Fin m`).
-/

namespace Accounting.BoundSketch

/-- A deterministic two-party protocol between the block side (input `x : X`) and the tile side (input `y : Y`),
ending with an output in `O`. `sendX s t f` is the block side sending the bit `s x` (the tree so far is the transcript,
so `s` reads the bits before it), continuing as `t` on `true` and `f` on `false`; `sendY` the same for the tile side;
`outX o` and `outY o` end with the block side or the tile side outputting `o` of its input. -/
inductive Protocol (X Y O : Type) : Type where
  | outX : (X → O) → Protocol X Y O
  | outY : (Y → O) → Protocol X Y O
  | sendX : (X → Bool) → Protocol X Y O → Protocol X Y O → Protocol X Y O
  | sendY : (Y → Bool) → Protocol X Y O → Protocol X Y O → Protocol X Y O

namespace Protocol

variable {X Y O O' : Type}

/-- The protocol's output on inputs `x` and `y`. -/
def eval : Protocol X Y O → X → Y → O
  | outX o, x, _ => o x
  | outY o, _, y => o y
  | sendX s t f, x, y => if s x then t.eval x y else f.eval x y
  | sendY s t f, x, y => if s y then t.eval x y else f.eval x y

/-- The bits the protocol exchanges, both directions, on its longest run. -/
def bits : Protocol X Y O → ℕ
  | outX _ => 0
  | outY _ => 0
  | sendX _ t f => max t.bits f.bits + 1
  | sendY _ t f => max t.bits f.bits + 1

/-- The protocol with its output read through `g`. -/
def map (g : O → O') : Protocol X Y O → Protocol X Y O'
  | outX o => outX (g ∘ o)
  | outY o => outY (g ∘ o)
  | sendX s t f => sendX s (t.map g) (f.map g)
  | sendY s t f => sendY s (t.map g) (f.map g)

end Protocol

variable {p : ℕ}

/-- The inner product on `𝔽_p^ι`. -/
def ip {ι : Type} [Fintype ι] (x y : ι → ZMod p) : ZMod p := ∑ i, x i * y i

/-- A sketch tile: `r` vectors of `m` bytes against a matrix of `r` columns of `m` bytes, `v_i[c] = Σ_q e_i[q]·B[c][q]`
in `𝔽_p`. -/
def sketch {r m : ℕ} (E B : Fin r × Fin m → ZMod p) : Fin r → Fin r → ZMod p :=
  fun i c => ∑ q, E (i, q) * B (c, q)

end Accounting.BoundSketch
