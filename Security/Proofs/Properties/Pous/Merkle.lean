import Properties.Pous
import Proofs.Properties.NetworkCertifier.Bytes

/-!
# The verification tree's binding, for the PoUS run's proofs

Two trees over leaves of 64 bytes with one root have the same leaves unless two of their inner nodes' inputs collide
(`mth_binds`, over `nodeInputs`); with the byte-level facts the run's proofs read (`hash_size`, `append_cancel`,
`leaves_entry`, `root_mth`). `Proofs.Properties.Pous.Run` and `RunAt` use them.
-/

namespace PousCheck.Properties.SecurityProofs

open Flock PousCheck.Properties

local instance instLawfulBEqByteArray_properties : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

theorem hash_size (m : ByteArray) : (Sha512.hash m).size = 64 := Properties.NetworkCertifier.Bytes.hash_size m

theorem append_cancel {a b c d : ByteArray} (h : a ++ b = c ++ d) (hs : a.size = c.size) : a = c ∧ b = d := by
  have := congrArg ByteArray.data h
  simp only [ByteArray.data_append] at this
  obtain ⟨h1, h2⟩ := Array.append_inj this hs
  exact ⟨ByteArray.ext h1, ByteArray.ext h2⟩

theorem mth_size (L : Array ByteArray) (hL : ∀ i (h : i < L.size), L[i].size = 64) :
    ∀ n lo, 1 ≤ n → lo + n ≤ L.size → (Merkle.mth L lo n).size = 64 := by
  intro n lo h1 h2
  rw [Merkle.mth]
  split
  · have : n = 1 := by omega
    subst this
    rw [Array.getElem?_eq_getElem (by omega)]
    exact hL lo (by omega)
  · exact hash_size _

theorem nodeInputs_eq (L : Array ByteArray) (lo n : Nat) (h : 2 ≤ n) :
    nodeInputs L lo n = (ByteArray.mk #[1] ++ Merkle.mth L lo (Merkle.split n) ++
        Merkle.mth L (lo + Merkle.split n) (n - Merkle.split n)) ::
      (nodeInputs L lo (Merkle.split n) ++ nodeInputs L (lo + Merkle.split n) (n - Merkle.split n)) := by
  rw [nodeInputs]
  simp only [show ¬ n ≤ 1 by omega, ↓reduceDIte]

theorem mth_node (L : Array ByteArray) (lo n : Nat) (h : 2 ≤ n) :
    Merkle.mth L lo n = Sha512.hash (ByteArray.mk #[1] ++ Merkle.mth L lo (Merkle.split n) ++
        Merkle.mth L (lo + Merkle.split n) (n - Merkle.split n)) := by
  rw [Merkle.mth]
  simp only [show ¬ n ≤ 1 by omega, ↓reduceDIte]
  rfl

/-- Two trees of `n` leaves from `lo` with one root: their leaves agree, or two of their inner nodes' inputs collide. -/
theorem mth_binds (L1 L2 : Array ByteArray) (h1 : ∀ i (h : i < L1.size), L1[i].size = 64)
    (h2 : ∀ i (h : i < L2.size), L2[i].size = 64) :
    ∀ n lo, lo + n ≤ L1.size → lo + n ≤ L2.size → Merkle.mth L1 lo n = Merkle.mth L2 lo n →
      (∀ i, lo ≤ i → i < lo + n → L1[i]! = L2[i]!) ∨
        ∃ x ∈ nodeInputs L1 lo n, ∃ y ∈ nodeInputs L2 lo n, x ≠ y ∧ Sha512.hash x = Sha512.hash y := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hl1 hl2 he
  by_cases hn : n ≤ 1
  · left
    intro i hi1 hi2
    have hi : i = lo := by omega
    subst hi
    unfold Merkle.mth at he
    simp only [hn, ↓reduceDIte] at he
    rw [Array.getElem?_eq_getElem (by omega), Array.getElem?_eq_getElem (by omega)] at he
    simp only [Option.getD_some] at he
    rw [getElem!_pos L1 i (by omega), getElem!_pos L2 i (by omega)]
    exact he
  have hn2 : 2 ≤ n := by omega
  have hk := Merkle.split_lt hn2
  have hk0 := Merkle.split_pos n
  rw [mth_node L1 lo n hn2, mth_node L2 lo n hn2] at he
  rw [nodeInputs_eq L1 lo n hn2, nodeInputs_eq L2 lo n hn2]
  set k := Merkle.split n
  by_cases hx : ByteArray.mk #[1] ++ Merkle.mth L1 lo k ++ Merkle.mth L1 (lo + k) (n - k) =
      ByteArray.mk #[1] ++ Merkle.mth L2 lo k ++ Merkle.mth L2 (lo + k) (n - k)
  · have s1 : (ByteArray.mk #[1] ++ Merkle.mth L1 lo k).size = (ByteArray.mk #[1] ++ Merkle.mth L2 lo k).size := by
      simp only [ByteArray.size_append, mth_size L1 h1 k lo (by omega) (by omega),
        mth_size L2 h2 k lo (by omega) (by omega)]
    obtain ⟨ha, hb⟩ := append_cancel hx s1
    obtain ⟨-, ha⟩ := append_cancel ha rfl
    rcases ih k hk lo (by omega) (by omega) ha with hl | ⟨x, hx1, y, hy1, hxy⟩
    · rcases ih (n - k) (by omega) (lo + k) (by omega) (by omega) hb with hr | ⟨x, hx1, y, hy1, hxy⟩
      · left
        intro i hi1 hi2
        by_cases hik : i < lo + k
        · exact hl i hi1 hik
        · exact hr i (by omega) (by omega)
      · exact .inr ⟨x, by simp [hx1], y, by simp [hy1], hxy⟩
    · exact .inr ⟨x, by simp [hx1], y, by simp [hy1], hxy⟩
  · exact .inr ⟨_, List.mem_cons_self, _, List.mem_cons_self, hx, he⟩

theorem leaves_entry (salt : ByteArray) (C : Array (Array ByteArray)) :
    ∀ i (hi : i < (leaves salt C).size), (leaves salt C)[i].size = 64 := by
  intro i hi
  have hm := Array.getElem_mem hi
  generalize (leaves salt C)[i] = x at hm ⊢
  simp only [leaves, List.mem_toArray, List.mem_flatten, List.mem_mapIdx] at hm
  obtain ⟨_, ⟨_, _, rfl⟩, hx⟩ := hm
  obtain ⟨_, _, rfl⟩ := List.mem_mapIdx.1 hx
  exact hash_size _

theorem root_mth {L : Array ByteArray} {root : ByteArray} (h : Merkle.root L = some root) :
    Merkle.mth L 0 L.size = root := by
  unfold Merkle.root at h
  split at h
  · contradiction
  · exact Option.some.inj h

end PousCheck.Properties.SecurityProofs
