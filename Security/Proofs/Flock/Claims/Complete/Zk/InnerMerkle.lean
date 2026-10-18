import Proofs.Flock.Claims.Complete.Zk.Inner
import Proofs.Flock.Claims.Complete.Replay.LigeritoMerkle

/-!
# The pads' commitment opens at every column

`innerCommitP_tree`: the pads' commitment (`innerCommitP`) is its Merkle tree (`InnerTree`): `n` columns, `n` a power
of two, leaf `j` the `hm96-sha512/v1` leaf of `SHA-512(C_h[j] ‖ C_μ[j])` under its salt, layer `h + 1` the nodes of
layer `h`'s pairs, and the root layer `log₂ n`'s entry. So each column's siblings, as `innerP` opens them, climb from
its leaf to the root, which is the Merkle check `Zk.verifyInnerWith` makes at each query (`inner_column_opens`, from
the plain Ligerito's `climb_ok`).
-/

namespace FlockComplete

open Flock Transcript FlockComplete.Replay.Lg

/-- The inner code's length, `nextPow2(s) · 2^INNER_LOG_INV_RATE` columns for `s` symbols, is a power of two. -/
theorem innerCols_pow (s : ℕ) : s.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE =
    2 ^ Nat.log2 (s.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE) := by
  obtain ⟨k, hk⟩ := Nat.isPowerOfTwo_nextPowerOfTwo s
  rw [hk, ← Nat.pow_add, Nat.log2_two_pow]

/-- **The pads' commitment as its Merkle tree**, through `hs`: `n` columns and salts, leaf `j` the leaf of column `j`
under its salt, layer `h + 1` the nodes of layer `h`'s pairs, and the root layer `log₂ n`'s first entry. -/
structure InnerTree (hs : Hashes V) (n : ℕ) (ic : InnerCommit) : Prop where
  ch : ic.ch.size = n
  cm : ic.cm.size = n
  salts : ic.salts.size = n
  leaf : ∀ j < n, (do Hm96.Default512.leafWith hs (← hs.sha512 (ic.ch[j]!.toBytes ++ ic.cm[j]!.toBytes)) ic.salts[j]!)
    = .ok (ic.layers[0]!)[j]!
  node : ∀ h < Nat.log2 n, ∀ i < 2 ^ (Nat.log2 n - h - 1),
    (MerkleOps.hm96With hs).node (ic.layers[h]!)[2 * i]! (ic.layers[h]!)[2 * i + 1]! = .ok (ic.layers[h + 1]!)[i]!
  root : ic.root = (ic.layers[Nat.log2 n]!)[0]!

/-- **The pads' commitment is its Merkle tree**, at `n = nextPow2(s + K_PAD) · 2^INNER_LOG_INV_RATE` columns for the
`s + K_PAD` symbols of `h ‖ pad_h`. -/
theorem innerCommitP_tree {hs : Hashes V} {p : PadsZk} {salt : ℕ → ByteArray} {ic : InnerCommit}
    (h : innerCommitP hs p salt = .ok ic) :
    InnerTree hs ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE) ic := by
  unfold innerCommitP at h
  obtain ⟨leaves, hlv, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨r, hr, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h
  obtain ⟨hls, hleaf⟩ := mapM_get hlv
  simp only [Array.size_range] at hls
  set n := (p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE with hN
  have hpow : n = 2 ^ Nat.log2 n := innerCols_pow _
  have hI := forIn_inv (Nat.log2 n) (#[leaves], leaves)
    (fun i (q : Array (Array ByteArray) × Array ByteArray) => q.1.size = i + 1 ∧ q.2 = q.1[i]! ∧
      q.1[0]! = leaves ∧ (∀ j ≤ i, q.1[j]!.size = 2 ^ (Nat.log2 n - j)) ∧
      ∀ j < i, ∀ k < 2 ^ (Nat.log2 n - j - 1),
        (MerkleOps.hm96With hs).node (q.1[j]!)[2 * k]! (q.1[j]!)[2 * k + 1]! = .ok (q.1[j + 1]!)[k]!)
    ⟨rfl, rfl, rfl, fun j hj => by
      obtain rfl : j = 0 := by omega
      simpa [hls] using hpow, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
    (fun i hi q r' hq e => by
      obtain ⟨cur', hc, e⟩ := FlockSoundness.Refine.ebind_ok e
      cases e
      obtain ⟨h1, h2, h3, h4, h5⟩ := hq
      obtain ⟨hcs, hcn⟩ := mapM_get hc
      simp only [Array.size_range] at hcs
      have hq2 : q.2.size = 2 ^ (Nat.log2 n - i) := by rw [h2]; exact h4 i le_rfl
      have hd : Nat.log2 n - i = Nat.log2 n - (i + 1) + 1 := by omega
      have hsz : cur'.size = 2 ^ (Nat.log2 n - (i + 1)) := by
        rw [hcs, hq2, hd, Nat.pow_succ, Nat.mul_div_cancel _ (by norm_num)]
      have hget : ∀ j ≤ i, (q.1.push cur')[j]! = q.1[j]! := fun j hj => push_get_lt _ (by omega)
      have hlast : (q.1.push cur')[i + 1]! = cur' := by rw [← h1]; exact push_get_eq _ _
      refine ⟨_, rfl, by simp [h1], hlast.symm, by rw [hget 0 (Nat.zero_le _)]; exact h3, fun j hj => ?_,
        fun j hj k hk => ?_⟩
      · rcases Nat.lt_or_eq_of_le hj with hj | rfl
        · rw [hget j (by omega)]
          exact h4 j (by omega)
        · rw [hlast]
          exact hsz
      · rcases Nat.lt_or_eq_of_le (Nat.lt_succ_iff.1 hj) with hj | rfl
        · rw [hget j (by omega), hget (j + 1) (by omega)]
          exact h5 j hj k hk
        · rw [hget j le_rfl, hlast, ← h2]
          have := hcn k (by
            rw [Array.size_range, hq2, hd, Nat.pow_succ, Nat.mul_div_cancel _ (by norm_num)]
            rwa [Nat.sub_sub] at hk)
          simpa using this) hr
  obtain ⟨-, hcur, h0, -, hN'⟩ := hI
  refine ⟨?_, ?_, ?_, fun j hj => ?_, hN', ?_⟩
  · exact Array.size_map .. |>.trans (Array.size_range)
  · exact Array.size_map .. |>.trans (Array.size_range)
  · exact Array.size_map .. |>.trans (Array.size_range)
  · have := hleaf j (by rw [Array.size_range]; exact hj)
    rw [h0]
    simp only [Array.getElem_range] at this
    exact this
  · exact congrArg (·[0]!) hcur

/-- **Each column's siblings climb to the root**: at a column below `n`, the siblings `innerP` opens (layer `k`'s entry
beside the column's ancestor, for `k < log₂ n`) climb from the column's leaf to the root. -/
theorem innerTree_climb {hs : Hashes V} {n : ℕ} {ic : InnerCommit} (ht : InnerTree hs n ic)
    (hn : n = 2 ^ Nat.log2 n) {j : ℕ} (hj : j < n) :
    climbWith (MerkleOps.hm96With hs) (ic.layers[0]!)[j]! j
      ((Array.range (Nat.log2 n)).map fun k => (ic.layers[k]!)[(j >>> k) ^^^ 1]!).toList = .ok ic.root := by
  have hc := climb_ok (ops := MerkleOps.hm96With hs) (layers := ic.layers) (d := Nat.log2 n) (D := Nat.log2 n) le_rfl
    (fun h hh i hi => ht.node h hh i hi) (p := j) (by rw [← hn]; exact hj) (Nat.log2 n) 0 (by omega)
  have h0 : j >>> Nat.log2 n = 0 := by
    rw [Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt (by rw [← hn]; exact hj)]
  rw [Nat.shiftRight_zero, h0] at hc
  rw [Array.toList_map, Array.toList_range, List.range_eq_range', ht.root]
  exact hc

/-- **The pads' commitment opens at every column** (`Zk.verifyInnerWith`'s Merkle check): at a column `j` below `n`,
the column's leaf, from its two codeword entries and its salt, climbs through the siblings `innerP` opens to the
root. -/
theorem inner_column_opens {hs : Hashes V} {p : PadsZk} {salt : ℕ → ByteArray} {ic : InnerCommit}
    (h : innerCommitP hs p salt = .ok ic) {j : ℕ} (hj : j < ic.ch.size) :
    (do
      let x ← hs.sha512 (ic.ch[j]!.toBytes ++ ic.cm[j]!.toBytes)
      let leaf ← (Hm96.Default512.leafWith hs x ic.salts[j]! : V ByteArray)
      climbWith (MerkleOps.hm96With hs) leaf j
        ((Array.range (Nat.log2 ic.ch.size)).map fun k => (ic.layers[k]!)[(j >>> k) ^^^ 1]!).toList) =
      .ok ic.root := by
  have ht := innerCommitP_tree h
  rw [ht.ch] at hj ⊢
  have hl := ht.leaf j hj
  obtain ⟨x, hx, hl⟩ := FlockSoundness.Refine.ebind_ok hl
  rw [hx]
  show (do let leaf ← Hm96.Default512.leafWith hs x ic.salts[j]!; _) = _
  rw [hl]
  exact innerTree_climb ht (innerCols_pow _) hj

/-- A rep's pads have `len + K_PAD` symbols, so their commitment has the verifier's `n` columns. -/
theorem padsOf_symbols (tape : TapeZk) (i r : ℕ) (lay : Zk.PadLayout) :
    ((padsOf tape i r lay).h ++ (padsOf tape i r lay).padH).size = lay.len + Zk.K_PAD := by
  simp [padsOf]

/-- `τ`'s salt is `TAU_SALT_BYTES` long, as the verifier checks. -/
theorem padsOf_tauSalt_size (tape : TapeZk) (i r : ℕ) (lay : Zk.PadLayout) :
    (padsOf tape i r lay).tauSalt.size = Zk.TAU_SALT_BYTES :=
  fitBytes_size' _ _

end FlockComplete
