import Definitions.Pouw.PearlC.Instance
import Definitions.Pouw.PearlC.Sha512

/-!
# Pearl-C's noise as SHA-512 gives it (trusted; definitions only)

`PearlCSem.noise` is abstract: the named assumptions take the noise as a function of the oracle `H`. This module is the
noise Pearl-C's h3 hashing actually draws, as a function of the salt, the weight's commitment root and the committed
activations, through the spec's own SHA-512 (`Sha512.digest`), not a random oracle. A statement that wants the noise of
the deployment instantiates `PearlCSem` with `sha512Sem`; the oracle `H` is then unused, and what is uniform is the salt.

The rules, op for op, are `verity/ml/protocols/pouw/schemes/pearl_c.py` (h3, row-seeded, forming v1):
- `s512 role msg` (`pearl_c.s512`): the first 32 bytes of SHA-512 over the role's label block
  (`"pearl-c/h3s/<role>\0"` zero-padded to 128 bytes) and `msg`.
- `seedB` (`pearl_c.seed_b`): `s512 "seed-B" (salt ‖ root_B)`, one per weight and salt.
- `rowSeed` (`pearl_c.row_seed`): `s512 "seed-A" (salt ‖ leaf ‖ root_B ‖ u64le (u·2^32 + i))` for row `i` of unit `u`,
  with `leaf` the row's `sha512/row-seg/v1` digest (`rowDigest`, `rowleaf.sha512_row_seg_digest`) over its FP32 words
  little-endian. So each row of A seeds its own `E_A` line from its own values: the activations may be chosen after the
  salt.
- `sampleLine seed side factor line norm` (`pearl_c.sample_line` for an `S512Seed`): 32 bytes
  `s512 "line" (s512 "noise-line" seed ‖ [side, factor] ‖ u32le line)`, each byte `b` the integer
  `±((b mod 128) + 1)` (minus when bit 7 is set), scaled in BF16 to L2 norm `norm` and cast to E4M3:
  `scale = bf16(rn(bf16(32·norm) / bf16(isqrt(1024·Σx²))))` and `code = e4m3(bf16(rn(x·scale)))`, where `bf16` is
  `cvt.rn.bf16.f32` (`bf16Rn`), `rn` FP32's rounding, and `e4m3` Pearl's saturating cast (`Defs.e4m3`).
- The lines (`pearl_c.PearlC.encode`, `_side`, `_basis`): `E_A` row `i` is `sampleLine (rowSeed i) 0 0 0 256`, `E_B` row
  `j` is `sampleLine seedB 1 0 j 256`, `F_A` row `t` is `sampleLine seedB 0 1 t 16` and `F_B` row `t` is
  `sampleLine seedB 1 1 t 16` (`LINE_NORM_V1 = 16`; E keeps Pearl's 256).

**What is a parameter.**
- The salt's bytes, `salt : S → List ℕ`: the 32-byte epoch salt (`identifier.SALT_BYTES`, `coins.COIN_BYTES`) is
  `Salt32`, uniform when the game draws `s : Salt32` uniformly.
- `rootB w`, the 64-byte root of weight `w`'s hiding commitment. It is fixed with the weights before the salt, and its
  opening (the leaves' salts) is the committer's randomness, so the noise does not determine it from the weight codes.
- `fc`, `alpha`, `beta` and `floor` are the base `PearlCSem`'s: only the noise is pinned here.

The Python's refusals (a unit or row index of `2^32` or more, a non-finite BF16 value) are total here: the indices are
taken mod `2^64` as `u64le` bytes, and no line reaches a non-finite value (every `|x| ≥ 1`, so the scale and the codes are
finite and normal).
-/

namespace Pouw.PearlC

open Pouw.Fp8Atom Pouw.PearlC.Sha512

namespace NoiseSha512

/-- The 32-byte salt of an epoch. -/
abbrev Salt32 : Type := Fin 32 → Fin 256

/-- A salt's bytes, in order. -/
def Salt32.bytes (s : Salt32) : List ℕ := List.ofFn fun i => (s i).val

/-- A role's label block: `"pearl-c/h3s/<role>\0"` zero-padded to 128 bytes (`pearl_c.S512_LABELS`). -/
def label (role : String) : List ℕ := padTo 128 (ascii ("pearl-c/h3s/" ++ role) ++ [0])

/-- `H_role(msg)`: the first 32 bytes of SHA-512 of the role's label block and `msg` (`pearl_c.s512`). -/
def s512 (role : String) (msg : List ℕ) : List ℕ := (digest (label role ++ msg)).take 32

/-- `sha512/row-seg/v1`'s segment lengths in bytes for a row of `len` bytes: 1,024 each, the last the rest. -/
def rowSegs (len : ℕ) : List ℕ := List.replicate (len / 1024) 1024 ++ (if len % 1024 = 0 then [] else [len % 1024])

/-- **A row's `sha512/row-seg/v1` digest** (`rowleaf.sha512_row_seg_digest`, a whole number of bytes):
`SHA-512(Q ‖ d₀ ‖ … ‖ d_(s−1))`, `Q = "verity/sha512-row-seg/v1\0" ‖ u64be bits ‖ u64be s ‖ u32be 1024` and
`d_j = SHA-512(S_j ‖ segment j)`, `S_j = "verity/sha512-row-seg/v1/segment\0" ‖ u64be j ‖ u64be s ‖ u64be bits_j`, each
prefix zero-padded to 128 bytes. -/
def rowDigest (row : List ℕ) : List ℕ :=
  let segs := rowSegs row.length
  let s := segs.length
  let q := padTo 128 (ascii "verity/sha512-row-seg/v1" ++ [0] ++ beBytes 8 (8 * row.length) ++ beBytes 8 s ++ beBytes 4 1024)
  let d := (List.range s).flatMap fun j =>
    digest (padTo 128 (ascii "verity/sha512-row-seg/v1/segment" ++ [0] ++ beBytes 8 j ++ beBytes 8 s ++
      beBytes 8 (8 * segs.getD j 0)) ++ (row.drop (1024 * j)).take (segs.getD j 0))
  digest (q ++ d)

/-- Row `i` of a matrix of FP32 words with `k` columns, as bytes: each word little-endian (`pearl_c.f32_row`). -/
def rowBytes (W : Codes) (k i : ℕ) : List ℕ := (List.range k).flatMap fun l => leBytes 4 (W i l)

/-- `seed_B` of weight root `rootB` at the salt's bytes `salt` (`pearl_c.seed_b`, h3). -/
def seedB (salt rootB : List ℕ) : List ℕ := s512 "seed-B" (salt ++ rootB)

/-- The seed of row `i` of unit `u` with activation row bytes `row` (`pearl_c.row_seed`). -/
def rowSeed (salt row rootB : List ℕ) (u i : ℕ) : List ℕ :=
  s512 "seed-A" (salt ++ rowDigest row ++ rootB ++ leBytes 8 (u * 2 ^ 32 + i))

/-- A line's 32 signed integers from its 32 bytes: `±((b mod 128) + 1)`, minus when bit 7 is set. -/
def lineInts (raw : List ℕ) : List ℤ :=
  raw.map fun b => (if b / 128 % 2 = 1 then -1 else 1) * (((b % 128 : ℕ) : ℤ) + 1)

/-- The BF16 scale that takes a line of integers `xs` to L2 norm `norm`: `bf16(rn(bf16(32·norm) / bf16(isqrt(1024·Σx²))))`. -/
def lineScale (norm : ℚ) (xs : List ℤ) : ℚ :=
  bf16Rn (rn (bf16Rn (norm * 32) / bf16Rn (Nat.sqrt ((xs.map fun x => x.natAbs ^ 2).sum * 1024) : ℚ)))

/-- One noise line's codes as a list: `e4m3(bf16(rn(x·scale)))` per integer. -/
def lineList (seed : List ℕ) (side factor line : ℕ) (norm : ℚ) : List Code :=
  let xs := lineInts (s512 "line" (s512 "noise-line" seed ++ [side, factor] ++ leBytes 4 line))
  let sc := lineScale norm xs
  xs.map fun x => e4m3 (bf16Rn (rn (bf16Rn x * sc)))

/-- **One noise line** (`pearl_c.sample_line` for an `S512Seed`): 32 E4M3 codes. -/
def sampleLine (seed : List ℕ) (side factor line : ℕ) (norm : ℚ) (t : Fin Params.pi.r) : Code :=
  (lineList seed side factor line norm).getD t.val 0

/-- **The unit's noise from SHA-512**: `E_A`'s rows from their own row seeds, and `E_B`, `F_A`, `F_B` from the weight's
`seed_B`, at the salt's bytes `salt` and the weight root `rootB` of the unit's weight. -/
def unitNoise (salt rootB : List ℕ) (sh : Shape) (u : ℕ) (act : Codes) : Noise sh.m sh.n sh.k Params.pi.r where
  EA := fun i t => sampleLine (rowSeed salt (rowBytes act sh.k i) rootB u i) 0 0 0 256 t
  FA := fun l t => sampleLine (seedB salt rootB) 0 1 l 16 t
  EB := fun j t => sampleLine (seedB salt rootB) 1 0 j 256 t
  FB := fun l t => sampleLine (seedB salt rootB) 1 1 l 16 t

end NoiseSha512

open NoiseSha512

variable {Q R S : Type}

/-- **Pearl-C's semantics with SHA-512's noise**: `base` with its noise replaced by `unitNoise` at the salt's bytes
`salt s` and the root `rootB (wid u)` of the unit's weight. The oracle is unused. -/
def sha512Sem (base : PearlCSem Q R S) (salt : S → List ℕ) (rootB : ℕ → List ℕ) : PearlCSem Q R S :=
  { base with noise := fun _ s U u act => unitNoise (salt s) (rootB (U.layout.wid u)) (U.layout.shape u) u act }

end Pouw.PearlC
