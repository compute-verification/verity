import FlockSoundness.Discharge.Hidden.Wide.RowCommit

set_option Elab.async false

/-!
# A row's commit string at any length: the partial last block

`Hidden.row_commitP` hashes a row whose `B` data bytes are whole blocks (`B = 128 n`): `n` `sha512x3` compressions read
the data, and the `hm96` slot reads the padding block `HmRow.padWords`, all constants. A canonical row (`PROTOCOL.md`
§16.15) has `B = ⌈ℓ / 8⌉` bytes for any `ℓ < 2^64`. After the 128-byte prefix `P(ℓ)` its message is
`data ‖ padTail(128 + B)` (SHA-512's own padding), `n + 1 = ⌈(B + 17) / 128⌉` blocks: the first `n = chunks` are the
`sha512x3` slots', and the last, the `hm96` slot's, holds the row's last `B mod 128` bytes (if any) then the padding, or
the padding alone. `row_commitC` is `row_commitP` at that message: the same chain, the last block read bit by bit from
`data ‖ padTail` instead of as `padWords`, and `n = 0` allowed (a row of fewer than 111 bytes is hashed by its `hm96`
slot alone, `row_commit0P`'s case with data).

This is the one genuinely new hashing fact the canonical format needs; everything else generalizes existing lemmas.

**Proof work** (the bulk of the format's new proof): `padData (rowMsgP pre data)` is `pre ‖ data ‖ padTail(128 + B)`
(`Hm.padData_data` and `padTail`'s unfolding: the same `0x80`, `zeroPad` zeros and 16 length bytes); then
`blockV_rowP`'s block lemma for an unaligned `data ‖ tail` (block `k + 1` is bytes `128 k …` of `data ‖ tail`, by
`getD_append` on either side of `B`), `chainV_one_rowP` unchanged, and `lastIdx_rowMsgP` at `n + 1` blocks from the
`zeroPad` arithmetic (`(B + 17 + 127) / 128`, an `omega` over `B % 128`). The compression induction is `row_commitP`'s
verbatim with `hmsgW` from the new block lemma; the last step is `hm96_commit` with `hpad` read through `wordBit` like
`hmsgW`, and `hcv` from `hcvh0` (`n = 0`, `chainV_one_rowP`) or `hcvh` (`n > 0`).
-/

namespace FlockSoundness.Discharge.Canonical

open Flock.HmNets hiding and
open Flock Hm

/-- **A row's message after its prefix**: its `B` bytes, then SHA-512's padding of a message of `128 + B` bytes. -/
def rowTail (data : List UInt8) : List UInt8 := data ++ (HmRow.padTail (128 + data.length)).data.toList

/-- **`row_commitP` at any length**: a row of `B` data bytes in `n + 1 = ⌈(B + 17) / 128⌉` message blocks after the
128-byte prefix `pre`, whose `n` `sha512x3` slots hold Δ's copies (the first `cv` the prefix's midstate, each next `cv`
the previous output, `msg` block `k` of `rowTail data`) and whose `hm96` slot reads the last output (or, at `n = 0`, the
prefix's midstate) and the last block of `rowTail data`, outputs the commit string of `pre ‖ data` under the salt `s`.

*Route*: the module docstring's. -/
theorem row_commitC {pre : ByteArray} (hpre : pre.size = 128) {data : List UInt8} {n : ℕ}
    (hn : n + 1 = (data.length + 17 + 127) / 128) (zs : ℕ → Var → Bool) (sub : ℕ → ℕ) (hsub : ∀ k < n, sub k < 3)
    (hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2)
    (hcv0 : 0 < n → ∀ q : ℕ, q < 512 → zs 0 (1536 * sub 0 + q) = wordsBit (Sha512.compress Sha512.iv pre 0) q)
    (hchain : ∀ k, k + 1 < n → ∀ q : ℕ, q < 512 →
      zs (k + 1) (1536 * sub (k + 1) + q) = Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q)
    (hmsg : ∀ k < n, ∀ o < 128, ∀ t < 8,
      zs k (1536 * sub k + 512 + wordBit (8 * o + t)) = bitL (rowTail data) (8 * (128 * k + o) + t))
    (zh : Var → Bool) (hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2)
    (hcvh0 : n = 0 → ∀ q : ℕ, q < 512 → zh q = wordsBit (Sha512.compress Sha512.iv pre 0) q)
    (hcvh : 0 < n → ∀ q : ℕ, q < 512 →
      zh q = Hm.bit (zs (n - 1)) ((compression3.run St.empty).1.2.getD (sub (n - 1)) []) q)
    (hpadh : ∀ o < 128, ∀ t < 8, zh (512 + wordBit (8 * o + t)) = bitL (rowTail data) (8 * (128 * n + o) + t))
    {s : List UInt8} (hs : s.length = 192) (hy : ∀ q : ℕ, q < 1536 → zh (1536 + q) = bitL s q) :
    ∀ t < 1024, Hm.bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t =
      bitL (Binding.commitString (pre.data.toList ++ data) s) t := by
  sorry

end FlockSoundness.Discharge.Canonical
