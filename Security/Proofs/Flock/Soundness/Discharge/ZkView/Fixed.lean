import Proofs.Flock.Soundness.Discharge.ZkView.Leaf

/-!
# Lemma B at the `--zk` wire

`Coined.hm96` is `Table.Hm96` at a `--zk` table: its leaves are `H(x ⊕ M·y, c(y))` by `leafSplit`, so `viewExe` and
`simExe` are `Hm96.viewR` and `Hm96.simR`, and `CROnly.session_shvzk_gap` bounds any event of their encoding.
-/

namespace FlockSoundness.Discharge.ZkView

open Flock ZK

variable {K W Hi VL : Type} [Field K] {z : Sizes}

/-- **`Table.Hm96` at a `--zk` table**: the leaves under `My`, `saltHash` and `leafH`, level 0's digest of a column the
SHA-512 of its row, a pads column's the SHA-512 of its two elements. The hash of the salt is taken as any `f = My`, so
that `(T.hm96At f hf).My = f` holds by unfolding the structure rather than `My`. -/
noncomputable def Coined.hm96At (T : Coined K W Hi VL z) (f : Salt → Bits) (hf : f = My) : T.tab.Hm96 ByteArray where
  My := f
  cs := saltHash
  H := leafH
  dig0 col := bitsOf (Sha512.hash (MerkleScheme.rowBytes (row0 col)))
  digP p := bitsOf (Sha512.hash (F128.toBytes p.1 ++ F128.toBytes p.2))
  leaf0_eq _ _ := by subst hf; exact leafSplit _ _ (Sha512.hash_size _)
  leafP_eq _ _ := by subst hf; exact leafSplit _ _ (Sha512.hash_size _)
  ideal_eq _ _ := rfl
  EU := T.EU
  GpU := T.GpU
  ChU := T.ChU
  CpU := T.CpU

theorem Coined.hm96At_My (T : Coined K W Hi VL z) (f : Salt → Bits) (hf : f = My) : (T.hm96At f hf).My = f := rfl

theorem Coined.hm96At_cs (T : Coined K W Hi VL z) (f : Salt → Bits) (hf : f = My) : (T.hm96At f hf).cs = saltHash :=
  rfl

/-- `Coined.hm96At` at `My` itself. -/
noncomputable abbrev Coined.hm96 (T : Coined K W Hi VL z) : T.tab.Hm96 ByteArray := T.hm96At My rfl

theorem Coined.hidExe_eq (T : Coined K W Hi VL z) (w : W) (ω : T.tab.Rand) (ys : T.tab.Hid → Salt) :
    T.hidExe w ω ys = fun i => T.hm96.H (T.hm96.dig w ω i + T.hm96.My (ys i), T.hm96.cs (ys i)) := by
  funext i
  rcases i with q | ⟨r, k⟩
  · exact T.hm96.leaf0_eq _ _
  · exact T.hm96.leafP_eq _ _

theorem Coined.hidExeS_eq (T : Coined K W Hi VL z) (w0 : W) (σ : T.tab.SimRand) (ys : T.tab.Hid → Salt) :
    T.hidExeS w0 σ ys = fun i => T.hm96.H (T.hm96.digS w0 σ i + T.hm96.My (ys i), T.hm96.cs (ys i)) := by
  funext i
  rcases i with q | ⟨r, k⟩
  · exact T.hm96.leaf0_eq _ _
  · exact T.hm96.leafP_eq _ _

theorem Coined.viewExe_eq (T : Coined K W Hi VL z) (w : W) (ω : T.tab.Rand) (ys : T.tab.Hid → Salt) :
    T.viewExe w ω ys = T.hm96.viewR w ω ys := by
  rw [Coined.viewExe, Table.Hm96.viewR, T.hidExe_eq]

theorem Coined.simExe_eq (T : Coined K W Hi VL z) (w0 : W) (σ : T.tab.SimRand) (ys : T.tab.Hid → Salt) :
    T.simExe w0 σ ys = T.hm96.simR w0 σ ys := by
  rw [Coined.simExe, Table.Hm96.simR, T.hidExeS_eq]

theorem zk_wire_fixed : ZkWireFixed := by
  intro K W Hi VL _ _ _ _ _ _ J zs T Out enc hrank hpad hW hpads hβ w hreg hinner w0 E
  simp only [Coined.viewExe_eq, Coined.simExe_eq]
  exact CROnly.session_shvzk_gap (fun j => (T j).tab) (fun j => (T j).hm96) My saltHash (fun j => (T j).hm96At_My My rfl) (fun j => (T j).hm96At_cs My rfl)
    hrank hpad hW hpads hβ w hreg hinner w0 fun v => E (enc v)

end FlockSoundness.Discharge.ZkView
