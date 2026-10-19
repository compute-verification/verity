import Proofs.Flock.Claims.ZK.Split

/-!
# A rep's Ligerito opening from the cut (`SimLig`)

`SimLig` (`Split`) compares `repMix 1`, whose Ligerito opening reads level 0's masked messages, `y₁'`, `ȳ` and levels
≥ 1's salts from the cut (`ligeritoCut`), with `repMix 2`, the session's (`ligeritoPZk`).

* **The salt roundtrip** (`blobSalt_saltBlob`): level `l ≥ 1`'s salt at position `p`, read back from the salts' string,
  is the salt fitted to 192 bytes; a 192-byte salt comes back whole (`fit192_of_size`).
-/

namespace Claims.ZKMal.Masked

namespace LigPf

/-! ## The salt roundtrip -/

theorem fit192_size (b : ByteArray) : (fit192 b).size = 192 := by
  unfold fit192
  rw [ByteArray.size_append, ByteArray.size_extract]
  show _ + (Array.replicate (192 - b.size) (0 : UInt8)).size = 192
  rw [Array.size_replicate]
  omega

theorem fit192_of_size {b : ByteArray} (h : b.size = 192) : fit192 b = b := by
  unfold fit192
  rw [h, Nat.sub_self, ← h, ByteArray.extract_zero_size]
  exact ByteArray.append_empty

/-- Level `l`'s salts, fitted, one after another. -/
def levelBlob (s : ℕ → ℕ → ByteArray) (l N : ℕ) : ByteArray :=
  (List.range N).foldl (fun acc p => acc ++ fit192 (s l p)) ByteArray.empty

theorem foldl_app_eq {α : Type} (g : α → ByteArray) :
    ∀ (xs : List α) (acc : ByteArray),
      xs.foldl (fun acc a => acc ++ g a) acc = acc ++ xs.foldl (fun acc a => acc ++ g a) ByteArray.empty
  | [], acc => by simp [ByteArray.append_empty]
  | a :: xs, acc => by
    simp only [List.foldl_cons]
    rw [foldl_app_eq g xs (acc ++ g a), foldl_app_eq g xs (ByteArray.empty ++ g a), ByteArray.empty_append,
      ByteArray.append_assoc]

theorem levelBlob_succ (s : ℕ → ℕ → ByteArray) (l N : ℕ) :
    levelBlob s l (N + 1) = levelBlob s l N ++ fit192 (s l N) := by
  unfold levelBlob
  rw [List.range_succ, List.foldl_append]
  rfl

theorem levelBlob_size (s : ℕ → ℕ → ByteArray) (l : ℕ) : ∀ N, (levelBlob s l N).size = 192 * N
  | 0 => rfl
  | N + 1 => by
    rw [levelBlob_succ, ByteArray.size_append, levelBlob_size s l N, fit192_size]
    ring

theorem extract_append_left_of_le {a b : ByteArray} {i j : ℕ} (hj : j ≤ a.size) :
    (a ++ b).extract i j = a.extract i j := by
  rw [ByteArray.extract_append, show j - a.size = 0 by omega,
    show b.extract (i - a.size) 0 = ByteArray.empty from ByteArray.extract_eq_empty_iff.2 (by omega),
    ByteArray.append_empty]

theorem levelBlob_extract (s : ℕ → ℕ → ByteArray) (l : ℕ) :
    ∀ N p, p < N → (levelBlob s l N).extract (192 * p) (192 * (p + 1)) = fit192 (s l p)
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | N + 1, p, h => by
    rw [levelBlob_succ]
    by_cases hp : p = N
    · subst hp
      refine ByteArray.extract_append_eq_right (by rw [levelBlob_size]) ?_
      rw [levelBlob_size, fit192_size]
      ring
    · rw [extract_append_left_of_le (by rw [levelBlob_size]; nlinarith [show p + 1 ≤ N by omega])]
      exact levelBlob_extract s l N p (by omega)

/-- The salts' string through level `k`. -/
def blobUpTo (ls : List FlockSoundness.Accounting.Level) (s : ℕ → ℕ → ByteArray) (k : ℕ) : ByteArray :=
  (List.range' 1 k).foldl (fun acc l =>
    (List.range (2 ^ (ls.getD l default).logLen)).foldl (fun acc p => acc ++ fit192 (s l p)) acc) ByteArray.empty

theorem blobUpTo_succ (ls : List FlockSoundness.Accounting.Level) (s : ℕ → ℕ → ByteArray) (k : ℕ) :
    blobUpTo ls s (k + 1) = blobUpTo ls s k ++ levelBlob s (1 + k) (2 ^ (ls.getD (1 + k) default).logLen) := by
  unfold blobUpTo
  rw [List.range'_1_concat, List.foldl_append]
  simp only [List.foldl_cons, List.foldl_nil]
  exact foldl_app_eq (fun p => fit192 (s (1 + k) p)) _ _

theorem blobUpTo_size (ls : List FlockSoundness.Accounting.Level) (s : ℕ → ℕ → ByteArray) :
    ∀ k, k + 1 ≤ ls.length → (blobUpTo ls s k).size = 192 * hiOff ls (k + 1)
  | 0, _ => by simp [blobUpTo, hiOff, ByteArray.size_empty]
  | k + 1, h => by
    rw [blobUpTo_succ, ByteArray.size_append, blobUpTo_size ls s k (by omega), levelBlob_size,
      hiOff_succ ls (show 1 ≤ k + 1 by omega) (by omega), show 1 + k = k + 1 by omega]
    ring

theorem blobUpTo_prefix (ls : List FlockSoundness.Accounting.Level) (s : ℕ → ℕ → ByteArray) (k : ℕ) :
    ∀ d, ∃ rest, blobUpTo ls s (k + d) = blobUpTo ls s k ++ rest
  | 0 => ⟨ByteArray.empty, by rw [Nat.add_zero, ByteArray.append_empty]⟩
  | d + 1 => by
    obtain ⟨rest, hX⟩ := blobUpTo_prefix ls s k d
    exact ⟨rest ++ levelBlob s (1 + (k + d)) (2 ^ (ls.getD (1 + (k + d)) default).logLen), by
      rw [← Nat.add_assoc, blobUpTo_succ, hX, ByteArray.append_assoc]⟩

theorem saltBlob_eq (ls : List FlockSoundness.Accounting.Level) (s : ℕ → ℕ → ByteArray) :
    saltBlob ls s = blobUpTo ls s (ls.length - 1) := rfl

/-- **The salt roundtrip**: level `l ≥ 1`'s salt at position `p` of its tree, read back from the salts' string, is the
salt fitted to 192 bytes. -/
theorem blobSalt_saltBlob {ls : List FlockSoundness.Accounting.Level} {s : ℕ → ℕ → ByteArray} {l p : ℕ}
    (h : HiAt ls l p) : blobSalt ls (saltBlob ls s) l p = fit192 (s l p) := by
  obtain ⟨h1, hl, hp⟩ := h
  unfold blobSalt
  rw [ite_eq_left_of_eq_true _ _ (eq_true (show HiAt ls l p from ⟨h1, hl, hp⟩)), saltBlob_eq]
  obtain ⟨rest, hX⟩ := blobUpTo_prefix ls s l (ls.length - 1 - l)
  rw [show ls.length - 1 = l + (ls.length - 1 - l) by omega, hX]
  have hl' : l = (l - 1) + 1 := by omega
  have hsz : (blobUpTo ls s (l - 1)).size = 192 * hiOff ls l := by
    rw [blobUpTo_size ls s (l - 1) (by omega), ← hl']
  have hU : blobUpTo ls s l = blobUpTo ls s (l - 1) ++ levelBlob s l (2 ^ (ls.getD l default).logLen) := by
    conv_lhs => rw [hl']
    rw [blobUpTo_succ, show 1 + (l - 1) = l by omega]
  have hUsz : (blobUpTo ls s l).size = 192 * hiOff ls l + 192 * 2 ^ (ls.getD l default).logLen := by
    rw [hU, ByteArray.size_append, hsz, levelBlob_size]
  rw [extract_append_left_of_le (by rw [hUsz]; nlinarith), hU,
    show 192 * (hiOff ls l + p) = 192 * hiOff ls l + 192 * p by ring,
    show 192 * (hiOff ls l + p + 1) = 192 * hiOff ls l + 192 * (p + 1) by ring,
    ByteArray.extract_append_size_add' hsz.symm]
  exact levelBlob_extract s l _ p hp

/-- A 192-byte salt comes back whole. -/
theorem blobSalt_saltBlob_of_size {ls : List FlockSoundness.Accounting.Level} {s : ℕ → ℕ → ByteArray} {l p : ℕ}
    (h : HiAt ls l p) (hs : (s l p).size = 192) : blobSalt ls (saltBlob ls s) l p = s l p := by
  rw [blobSalt_saltBlob h, fit192_of_size hs]

end LigPf

end Claims.ZKMal.Masked
