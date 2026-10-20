import Proofs.Flock.Claims.ZK.FrameTape
import Proofs.Flock.Claims.ZK.Round1Root

/-!
# The frame's pads root is the masked prover's

At a rep's start the Lean frame commits the pads it drew (`Flock.Firewall.Frame.padsRoot`): `treeRoot` of the leaves
`H(H(c_h(j) ‖ c_μ(j)), salt j)`, a level at a time. The masked prover commits its tape's pads with `innerCommitP`, whose
root is `layerOf`'s of the same leaves at its own salts (`FlockComplete.innerCommitP_root_layerOf`). On the tape of the
frames' draws they are one root:

* `treeRoot_layerOf`: `treeRoot` of `2^d` leaves is `layerOf`'s root, so `padsRoot` is `layerOf`'s root of the frame's
  leaves (`padsRoot_layerOf`), over `codeLen` columns (`ofBytes_cols`).
* `padsRoot_tapeOfDraws`: at frame rep `rep i r`'s draws of `drawBytes` bytes, the frame's `padsRoot` is the root
  `innerCommitP` commits table `i`'s rep `r`'s pads under on `tapeOfDraws`: the pads are the frame's
  (`padsOf_tapeOfDraws`) and so are the leaf salts (`innerSalt_fit`).
-/

namespace Claims.ZKServer

open _root_.Flock.Firewall.Frame (Draws drawBytes codeLen SALT treeRoot padsRoot)

/-- `treeRoot`'s loop from level `h` of `layerOf`'s tree of depth `d`, with fuel for the `d - h` levels above it, is the
tree's root. -/
theorem treeRoot_go_layerOf (lf : ℕ → ByteArray) (d : ℕ) :
    ∀ k h, h + k = d → ∀ fuel, k ≤ fuel →
      treeRoot.go (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) d lf h) fuel =
        (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) d lf d)[0]!
  | 0, h, hk, fuel, _ => by
    obtain rfl : h = d := by omega
    have hs := FlockComplete.layerOf_size (fun l r => _root_.Flock.Sha512.hash (l ++ r)) h lf h (Nat.le_refl _)
    rw [Nat.sub_self, Nat.pow_zero] at hs
    cases fuel with
    | zero => rfl
    | succ f => rw [treeRoot.go, ite_cond_eq_true _ _ (eq_true (by omega))]
  | k + 1, h, hk, fuel, hf => by
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
    have hs := FlockComplete.layerOf_size (fun l r => _root_.Flock.Sha512.hash (l ++ r)) d lf h (by omega)
    have h2 : 2 ≤ 2 ^ (d - h) := by
      rw [show d - h = (k + 1) by omega, Nat.pow_succ]
      have := Nat.one_le_two_pow (n := k)
      omega
    rw [treeRoot.go, ite_cond_eq_false _ _ (eq_false (by omega))]
    exact treeRoot_go_layerOf lf d k (h + 1) (by omega) f (by omega)

/-- **`treeRoot` of `2^d` leaves is `layerOf`'s root.** -/
theorem treeRoot_layerOf (lf : ℕ → ByteArray) (d : ℕ) :
    treeRoot ((Array.range (2 ^ d)).map lf) =
      (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) d lf d)[0]! :=
  treeRoot_go_layerOf lf d d 0 (by omega) _ (by simpa using Nat.le_of_lt (Nat.lt_two_pow_self (n := d)))

/-- `layerOf` reads its leaves only below `2^d`. -/
theorem layerOf_congr (nd : ByteArray → ByteArray → ByteArray) (d : ℕ) {lf lf' : ℕ → ByteArray}
    (h : ∀ j < 2 ^ d, lf j = lf' j) : ∀ k, FlockComplete.layerOf nd d lf k = FlockComplete.layerOf nd d lf' k
  | 0 => by
    simp only [FlockComplete.layerOf]
    exact Array.ext (by simp) fun j h1 _ => by
      simp only [Array.getElem_map, Array.getElem_range]
      exact h j (by simpa using h1)
  | k + 1 => by simp only [FlockComplete.layerOf, layerOf_congr nd d h k]

/-- The rows the frame's pads tree encodes, `h ‖ pad_h`, take `codeLen` columns. -/
theorem ofBytes_cols (l : _root_.Flock.Zk.PadLayout) (b : ByteArray) :
    ((Draws.ofBytes l b).h ++ (Draws.ofBytes l b).padH).size.nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE =
      codeLen l := by
  simp [Draws.ofBytes, _root_.Flock.Firewall.Frame.words, codeLen]

/-- **The frame's pads root is `layerOf`'s root** of its leaves: column `j`'s leaf under the draws' `j`-th salt. -/
theorem padsRoot_layerOf (l : _root_.Flock.Zk.PadLayout) (d : Draws) :
    padsRoot l d = (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) (Nat.log2 (codeLen l))
      (fun j => _root_.Flock.Hm96.Default512.leaf (_root_.Flock.Sha512.hash
        ((_root_.Flock.Zk.codeAt (d.h ++ d.padH) j).toBytes ++ (_root_.Flock.Zk.codeAt (d.mu ++ d.padMu) j).toBytes))
        (d.salts.extract (SALT * j) (SALT * j + SALT))) (Nat.log2 (codeLen l)))[0]! := by
  obtain ⟨k, hk⟩ := Nat.isPowerOfTwo_nextPowerOfTwo (l.len + _root_.Flock.Zk.K_PAD)
  have hD : codeLen l = 2 ^ (k + _root_.Flock.Zk.INNER_LOG_INV_RATE) := by
    rw [Nat.pow_add, ← hk]
    rfl
  unfold padsRoot
  rw [hD, Nat.log2_two_pow, treeRoot_layerOf]

/-- **The frame's pads root is the masked prover's**: at frame rep `rep i r`'s draws of `drawBytes` bytes, the frame's
`padsRoot` is the root `innerCommitP` commits table `i`'s rep `r`'s pads under, on the tape of the frames' draws. -/
theorem padsRoot_tapeOfDraws (t : FlockComplete.TapeZk) (lay : ℕ → _root_.Flock.Zk.PadLayout) (B : ℕ → ByteArray)
    (rep : ℕ → ℕ → ℕ) (i r : ℕ) (hs : (B (rep i r)).size = drawBytes (lay (rep i r)))
    {ic : FlockComplete.InnerCommit}
    (h : (FlockComplete.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (FlockComplete.padsOf (tapeOfDraws t lay B rep) i r (lay (rep i r)))
      ((tapeOfDraws t lay B rep).innerSalt i r) : _root_.Flock.V FlockComplete.InnerCommit) = .ok ic) :
    padsRoot (lay (rep i r)) (Draws.ofBytes (lay (rep i r)) (B (rep i r))) = ic.root := by
  rw [FlockComplete.innerCommitP_root_layerOf h,
    padsOf_tapeOfDraws t lay B rep i r (by rw [hs]; exact tau_le_drawBytes _), padsRoot_layerOf]
  simp only [drawsPads]
  rw [ofBytes_cols]
  refine congrArg (fun a => a[0]!) (layerOf_congr _ _ (fun j hj => ?_) _)
  have hj' : j < codeLen (lay (rep i r)) := by
    rw [show codeLen (lay (rep i r)) = 2 ^ Nat.log2 (codeLen (lay (rep i r))) from FlockComplete.innerCols_pow _]
    exact hj
  beta_reduce
  rw [innerSalt_fit t lay B rep i r j hs hj']

end Claims.ZKServer
