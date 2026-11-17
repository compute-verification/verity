module

public import Proofs.Flock.Claims.ZK.LinkRecheck
public import Flock.ZkServer.Finish

/-! The ZK server's commitments are the masked prover's. -/

@[expose] public section

namespace Flock.ZkServer.Finish

open FlockSoundness.Refine

theorem rowsFft_size (lv : Level) (g : Array F128) : (rowsFft lv g).size = 2 ^ lv.d := by
  simp [rowsFft]

theorem rowsFft_get_size (lv : Level) (g : Array F128) {q : ℕ} (hq : q < 2 ^ lv.d) :
    ((rowsFft lv g)[q]!).size = 2 ^ lv.lanes := by
  simp only [rowsFft]
  rw [FlockSoundness.Refine.getElem!_range_map _ hq]
  simp

theorem encodeRows_size (lv : Level) (g : Array F128) : (FlockComplete.encodeRows lv g).size = 2 ^ lv.d := by
  simp [FlockComplete.encodeRows]

theorem encodeRows_get_size (lv : Level) (g : Array F128) {q : ℕ} (hq : q < 2 ^ lv.d) :
    ((FlockComplete.encodeRows lv g)[q]!).size = 2 ^ lv.lanes := by
  rw [FlockComplete.Replay.Lg.encodeRows_get lv g hq]
  simp

theorem rowsFft_get (lv : Level) (g : Array F128) (hc : lv.cols < lv.d) (hd : lv.d ≤ 64) {q l : ℕ}
    (hq : q < 2 ^ lv.d) (hl : l < 2 ^ lv.lanes) :
    fOf ((rowsFft lv g)[q]!)[l]! = Link.fftSum 0 (lv.cols + 1)
      (fun c => fOf (Link.coeffs lv.cols (fun c => g[c * 2 ^ lv.lanes + l]!) #[] c)) (FlockLevel3.omega q) := by
  have h := Link.fft_get (lv.cols + 1) lv.d (by omega) hd (lv.cols + 1) 0
    (Link.coeffs lv.cols (fun c => g[c * 2 ^ lv.lanes + l]!) #[]) (by omega) q (by simpa using hq)
  rw [Nat.pow_zero, Nat.mul_one] at h
  simp only [rowsFft]
  rw [FlockSoundness.Refine.getElem!_range_map _ hq, FlockSoundness.Refine.getElem!_range_map _ hl, Array.map_map,
    FlockSoundness.Refine.getElem!_range_map _ hl]
  exact h

theorem encodeRows_entry (lv : Level) (g : Array F128) {q l : ℕ} (hq : q < 2 ^ lv.d) (hl : l < 2 ^ lv.lanes) :
    fOf ((FlockComplete.encodeRows lv g)[q]!)[l]! = ∑ c ∈ Finset.range (2 ^ lv.cols),
      fOf g[c * 2 ^ lv.lanes + l]! * FlockSoundness.execArith.Xhat c (lv.cols + 1) (FlockSoundness.execArith.omega q) := by
  rw [FlockComplete.Replay.Lg.encodeRows_get lv g hq, FlockSoundness.Refine.getElem!_range_map _ hl]
  simp only [FlockComplete.fOf_sumF]
  refine Finset.sum_congr rfl fun c hc => ?_
  rw [fOf_mul, FlockComplete.Replay.Lg.fOf_xhats q lv.cols c (Finset.mem_range.1 hc),
    FlockSoundness.Xhat_succ_low FlockSoundness.execArith lv.cols c (Finset.mem_range.1 hc)]

theorem rowsFft_eq (lv : Level) (g : Array F128) (hc : lv.cols < lv.d) (hd : lv.d ≤ 64) :
    rowsFft lv g = FlockComplete.encodeRows lv g := by
  refine Link.arr2_ext (2 ^ lv.d) (2 ^ lv.lanes) (rowsFft_size lv g) (encodeRows_size lv g)
    (fun q hq => rowsFft_get_size lv g hq) (fun q hq => encodeRows_get_size lv g hq) fun q hq l hl => ?_
  have hv := Link.fftSum_coeffs lv.cols q 0 (fun c => g[c * 2 ^ lv.lanes + l]!) (fun _ => 0) (Nat.zero_le _)
  have he : (Array.range 0).map (fun _ : ℕ => (0 : F128)) = #[] := by simp
  simp only [he, Finset.sum_range_zero, add_zero] at hv
  apply fOf_injective
  rw [rowsFft_get lv g hc hd hq hl, encodeRows_entry lv g hq hl]
  exact hv

theorem rowsOf_eq (lv : Level) (g : Array F128) : rowsOf lv g = FlockComplete.encodeRows lv g := by
  unfold rowsOf
  split
  · rename_i h
    exact rowsFft_eq lv g h.1 h.2
  · rfl

def Commit.toC (c : Commit) : FlockComplete.Commit := ⟨c.rows, c.salts, c.layers, c.cap⟩

theorem openP_eq (ms : MerkleScheme) (cm : Commit) (lv : Level) (pos : Array ℕ) :
    FlockComplete.openP ms cm.toC lv pos = openP ms cm lv pos := rfl

variable {M : Type → Type} [Monad M]

theorem hm96_ops (hs : Hashes M) : MerkleScheme.hm96Sha512.opsWith hs = MerkleOps.hm96With hs := rfl

theorem hm96_node (l r : ByteArray) :
    (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 Sha512.hash : Hashes M)).node l r =
      pure (Sha512.hash (l ++ r)) := by
  rw [hm96_ops]
  rfl

variable [LawfulMonad M]

theorem hm96_leaf (row : Array F128) (s : ByteArray) :
    (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 Sha512.hash : Hashes M)).leaf row s =
      pure (MerkleScheme.hm96Sha512.leaf row s) := by
  rw [hm96_ops]
  simp only [MerkleOps.hm96With, Hm96.Default512.leafWith, Hashes.ofSha512, pure_bind]
  rfl

theorem forIn_range_pure {β : Type} (n : ℕ) (b : β) (f : ℕ → β → β) :
    (forIn [0:n] b (fun i r => (pure (ForInStep.yield (f i r)) : M (ForInStep β)))) =
      pure (Id.run (forIn [0:n] b (fun i r => (pure (ForInStep.yield (f i r)) : Id (ForInStep β))))) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    List.forIn_pure_yield_eq_foldl]
  rfl

theorem commitRowsP_eq (lv : Level) (rows : Array (Array F128)) (salt : ℕ → ByteArray) :
    (FlockComplete.commitRowsP (Hashes.ofSha512 Sha512.hash : Hashes M) MerkleScheme.hm96Sha512 lv rows salt) =
      pure (commitRows lv rows salt).toC := by
  have hk : 2 ^ (lv.d - min lv.d Link.CHUNK) * 2 ^ min lv.d Link.CHUNK = 2 ^ lv.d := by
    rw [← Nat.pow_add, Nat.sub_add_cancel (Nat.min_le_left _ _)]
  unfold FlockComplete.commitRowsP commitRows treeOf
  simp only [hm96_leaf, hm96_node, Array.mapM_pure, pure_bind, Link.parRange_eq, hk, forIn_range_pure (M := M)]
  rfl

theorem commitP_eq (lv : Level) (g : Array F128) (salt : ℕ → ByteArray) :
    (FlockComplete.commitP (Hashes.ofSha512 Sha512.hash : Hashes M) MerkleScheme.hm96Sha512 lv g salt) =
      pure (commitFw lv g salt).toC := by
  rw [show FlockComplete.commitP (Hashes.ofSha512 Sha512.hash : Hashes M) MerkleScheme.hm96Sha512 lv g salt =
      FlockComplete.commitRowsP (Hashes.ofSha512 Sha512.hash) MerkleScheme.hm96Sha512 lv
        (FlockComplete.encodeRows lv g) salt from rfl, commitRowsP_eq, commitFw, rowsOf_eq]

theorem commitPZk_eq (lv : Level) (tPad : ℕ) (i : Link.In) :
    (FlockComplete.commitPZk (Hashes.ofSha512 Sha512.hash : Hashes M) MerkleScheme.hm96Sha512 lv tPad i.g i.extra
      i.lanePad i.salt) = pure (commit0 lv tPad i).toC := by
  rw [FlockComplete.commitPZk, commitRowsP_eq, commit0, Link.rows_eq]

end Flock.ZkServer.Finish
