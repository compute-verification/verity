import Proofs.Flock.Soundness.Discharge.Hidden.Wide.RowCommit

/-!
# A segment's chain: SHA-512 of its message in its compressions alone

A `sha512/row-seg/v1` row hashes each segment `j` in compressions of its own (`HmRow.segShape`): the first `cv` the
midstate of its first block `S(j, s, m)`, each later `cv` the previous compression's output, and the messages its bytes
and then SHA-512's padding, all of it in the `sha512x3` slots, since the segment's digest is a message of the outer hash,
not a leaf. `chain_digestP` is that chain under any 128-byte first block `pre`: when the `n` compressions' messages are
the blocks after `pre` of `pre ‖ data` padded (`128 (n - 1) ≤ |data|`, `|data| + 17 ≤ 128 n`), the last compression's
output is `SHA-512(pre ‖ data)` in word order. The outer message is then `row_commitP`'s, with `Q(l)` its prefix and the
segments' digests its data.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Hidden

open Flock.HmNets hiding and

open Flock Hm

section Pre

variable {pre : ByteArray} (hpre : pre.size = 128)
include hpre

/-- Block `k + 1` of `pre ‖ data` padded, bit `q`: bit `8 (128 k + o) + q % 8` of the data and its padding, at the
block's byte `o = 8 (q / 64) + 7 - q % 64 / 8`. -/
theorem wbit_blockV_rowP (data : List UInt8) (k : ℕ) {q : ℕ} (hq : q < 1024) :
    wbit (blockV (padData (rowMsgP pre data)) (k + 1)) q =
      bitL (data ++ (HmRow.padTail (128 + data.length)).data.toList)
        (8 * (128 * k + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) := by
  rw [wbit_blockV _ _ hq, rowMsgP_pad hpre, List.append_assoc]
  unfold bitL
  rw [List.getD_append_right _ _ _ _ (by rw [length_pre hpre]; omega), length_pre hpre]
  have e1 : (8 * (128 * (k + 1) + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) / 8 - 128 =
      (8 * (128 * k + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) / 8 := by omega
  have e2 : (8 * (128 * (k + 1) + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) % 8 =
      (8 * (128 * k + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) % 8 := by omega
  rw [e1, e2]

/-- **A chain of `n > 0` compressions hashes `pre ‖ data`**: the first `cv` the prefix's midstate, each later one the
previous compression's output, and the messages the blocks after `pre` of `pre ‖ data` padded, which end with the `n`-th
(`128 (n - 1) ≤ |data|`, `|data| + 17 ≤ 128 n`): the last compression's output, in word order, is the digest. -/
theorem chain_digestP {n : ℕ} (hn : 0 < n) {data : List UInt8} (h1 : 128 * (n - 1) ≤ data.length)
    (h2 : data.length + 17 ≤ 128 * n) (zs : ℕ → Var → Bool) (sub : ℕ → ℕ) (hsub : ∀ k < n, sub k < 3)
    (hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2)
    (hcv0 : ∀ q : ℕ, q < 512 → zs 0 (1536 * sub 0 + q) = wordsBit (Sha512.compress Sha512.iv pre 0) q)
    (hchain : ∀ k, k + 1 < n → ∀ q : ℕ, q < 512 →
      zs (k + 1) (1536 * sub (k + 1) + q) = Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q)
    (hmsg : ∀ k < n, ∀ o < 128, ∀ t < 8, zs k (1536 * sub k + 512 + wordBit (8 * o + t)) =
      bitL (data ++ (HmRow.padTail (128 + data.length)).data.toList) (8 * (128 * k + o) + t)) :
    ∀ t < 512, Hm.bit (zs (n - 1)) ((compression3.run St.empty).1.2.getD (sub (n - 1)) []) (wordBit t) =
      bitL (Sha512.hash (rowMsgP pre data)).data.toList t := by
  set P := padData (rowMsgP pre data) with hP
  have hmsgW : ∀ k < n, ∀ q : ℕ, q < 1024 → zs k (1536 * sub k + 512 + q) = wbit (blockV P (k + 1)) q := by
    intro k hk q hq
    have ho : 8 * (q / 64) + (7 - q % 64 / 8) < 128 := by omega
    have hq' : wordBit (8 * (8 * (q / 64) + (7 - q % 64 / 8)) + q % 8) = q := by unfold wordBit; omega
    have e := hmsg k hk (8 * (q / 64) + (7 - q % 64 / 8)) ho (q % 8) (Nat.mod_lt _ (by norm_num))
    rw [hq'] at e
    rw [e, wbit_blockV_rowP hpre data k hq]
  have hlen8 : ∀ k, (chainV P k).length = 8 := fun k => by rw [chainV, List.length_map, List.length_range]
  have hlen16 : ∀ k, (blockV P k).length = 16 := fun k => by rw [blockV, List.length_map, List.length_range]
  have hout : ∀ k < n, ∀ q < 512,
      Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q = wbit (chainV P (k + 2)) q := by
    intro k
    induction k with
    | zero =>
      intro hk q hq
      refine (compression3_bits (zs 0) (hrows 0 hk) (hsub 0 hk) (hlen8 1) (hlen16 1) (fun q hq => ?_)
        (hmsgW 0 hk) q hq).trans (by rw [show 0 + 2 = 1 + 1 from rfl, chainV_succ P 1])
      rw [hcv0 q hq, wordsBit_eq _ (show q < 64 * 8 by omega), chainV_one_rowP hpre]
    | succ k ih =>
      intro hk q hq
      refine (compression3_bits (zs (k + 1)) (hrows _ hk) (hsub _ hk) (hlen8 (k + 2)) (hlen16 (k + 2))
        (fun q hq => ?_) (hmsgW _ hk) q hq).trans (by rw [chainV_succ P (k + 1 + 1)])
      rw [hchain k hk q hq, ih (by omega) q hq]
  have hlast : P.size / 128 - 1 = n := by
    have := lastIdx_rowMsgP hpre (n := n - 1) h1 (by omega)
    rw [← hP] at this
    omega
  intro t ht
  rw [hout (n - 1) (by omega) _ (wordBit_lt ht), bitL_hash _ ht, lastCv_eq, lastBlock_eq, ← hP, hlast, ← chainV_succ,
    show n - 1 + 2 = n + 1 by omega]

end Pre

end FlockSoundness.Discharge.Hidden
