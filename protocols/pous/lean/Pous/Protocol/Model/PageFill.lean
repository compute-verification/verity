import Mathlib.Data.Fintype.Perm
import Mathlib.Logic.Equiv.Fin.Basic
import Pous.Protocol.Model.Scheme

/-!
# The keyed page fill, with each page's permutation ideal

The fill `F_K` that `verity_pous.erasure.KeyedFill` computes and `ErasureVerifier` audits: `P` pages of `pb` bytes
(`PAGE_BYTES = 2^20`), page `p` the ten-round Feistel-SHAKE256 permutation `Feistel10` of the zero page under the
40-byte tweak `K ‖ u64(p)`, where `K` is the verifier's fresh 32-byte fill key. Byte `i` of the fill is byte `i mod pb`
of page `⌊i / pb⌋` (`KeyedFill.at`).

**The idealization.** Each page's `Feistel10` is an ideal permutation: the model `pageModel` draws one independent
uniform permutation of `8·pb`-bit pages per tweak, with forward and inverse queries, as the P3 model draws `Π₂` and the
tweakable `P` (`Pous/Protocol/Model/P3Chain.lean`, `p3ChainModel`). Nothing else is idealized and nothing is assumed:
pages are independent, keyed by `K` and the page index, with no dependency graph between them. One query is one whole
page's permutation, forward or inverse; a responder bounded by `D = 0` rounds makes none, which is what a read deadline
`T_read` below one page's recomputation (`KeyedFill.recompute_ns`, `page_keccak_f(PAGE_BYTES) · KECCAK_F_NS` =
1.7352 ms for 1 MiB pages) gives.

**The scheme.** `pageScheme pb P K` has no data (`n = 0`, no public parameters, `t = 0`, no coins) and a code of
`B = P·pb` blocks of `ℓ = 8` bits: a read is one byte. `K` is a parameter, so every statement about the scheme holds
for every key, with the prover chosen after it.

**What a reviewer checks.** That `tweak K p` is `K ‖ u64(p)` for `p < 2^64`; that byte `i` is byte `i mod pb` of
page `⌊i / pb⌋` (`finProdFinEquiv.symm i = (⌊i / pb⌋, i mod pb)`); and that the model is the ideal-permutation
idealization of P3 and nothing more.
-/

namespace Pous.PageFill

open Pous

/-- A page of `pb` bytes, as `8·pb` bits. -/
abbrev Page (pb : ℕ) : Type := Bits (pb * 8)

/-- A page's tweak: 40 bytes. -/
abbrev Tweak : Type := Bits 320

/-- The tweak `K ‖ u64(p)`: the 256 bits of the key, then the page index `p` in 64 bits, least significant first. -/
def tweak (K : Bits 256) (p : ℕ) : Tweak :=
  fun j => if h : j.val < 256 then K ⟨j.val, h⟩ else p.testBit (j.val - 256)

/-- The primitive: one permutation of pages per tweak. -/
abbrev PageOmega (pb : ℕ) : Type := Tweak → Equiv.Perm (Page pb)

/-- **The page model**: an ideal tweakable permutation of `pb`-byte pages, one independent uniform permutation per
tweak, with queries `π_T` and `π_T⁻¹`. -/
noncomputable def pageModel (pb : ℕ) : IdealModel where
  Ω := PageOmega pb
  Qry := Tweak × (Page pb ⊕ Page pb)
  Ans := Page pb
  answer ω q := match q with
    | (T, .inl x) => ω T x
    | (T, .inr y) => (ω T).symm y

/-- The zero page, every page's input (`Feistel10.forward(0, tweak)`). -/
def zeroPage (pb : ℕ) : Page pb := fun _ => false

/-- Page `p` of the fill under key `K`: `π_{K ‖ u64(p)}(0)`. -/
def page {pb : ℕ} (K : Bits 256) (ω : PageOmega pb) (p : ℕ) : Page pb := ω (tweak K p) (zeroPage pb)

/-- Byte `b` of a page: its bits `8b, …, 8b + 7`. -/
def byte {pb : ℕ} (x : Page pb) (b : Fin pb) : Bits 8 := fun k => x (finProdFinEquiv (b, k))

/-- **The keyed page fill as a scheme**: `P` pages of `pb` bytes under key `K`, read a byte at a time. Byte `i` is
byte `i mod pb` of page `⌊i / pb⌋`. -/
noncomputable def pageScheme (pb P : ℕ) [NeZero pb] [NeZero P] (K : Bits 256) : Scheme (pageModel pb) where
  n := 0
  t := 0
  B := P * pb
  ℓ := 8
  B_pos := Nat.mul_pos (Nat.pos_of_ne_zero (NeZero.ne P)) (Nat.pos_of_ne_zero (NeZero.ne pb))
  Coins := Unit
  enc _ ω _ := (fun _ => false, fun i => byte (page K ω (finProdFinEquiv.symm i).1) (finProdFinEquiv.symm i).2)
  dec _ _ _ := fun _ => false

end Pous.PageFill
