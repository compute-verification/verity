module

public import Proofs.Properties.Verity.Sampled.TreeMask
import Proofs.Flock.Soundness.CROnly.Hiding
import Proofs.Flock.Soundness.CROnly.Hm96Sha512
import Proofs.Flock.Soundness.CROnly.ZKKey
import Proofs.Flock.Soundness.Discharge.Hm.Commit

/-! A coin tree's hiding gap, averaged over its key. -/

@[expose] public section

namespace Verity.Game

open Finset
open FlockSoundness.CROnly (hidingGap hankel)
open FlockSoundness.Discharge.ZkView (Bits bitsOf)
open FlockSoundness.Discharge.Hm (bitL eq_of_bitL)

/-- A bit as an element of `ZMod 2`: `true` is `1`. -/
def boolZ : Bool ≃ ZMod 2 where
  toFun b := if b then 1 else 0
  invFun z := decide (z = 1)
  left_inv b := by cases b <;> rfl
  right_inv := by decide

def bitsZ (n : ℕ) : (Fin n → Bool) ≃ (Fin n → ZMod 2) := Equiv.piCongrRight fun _ => boolZ

theorem hidingGap_comp_equiv {Salt Salt' B C : Type} [Fintype Salt] [Fintype Salt'] [Fintype B] [DecidableEq B]
    [DecidableEq C] (e : Salt' ≃ Salt) (My : Salt → B) (c : Salt → C) :
    hidingGap (My ∘ e) (c ∘ e) = hidingGap My c := by
  have himg : univ.image (c ∘ e) = univ.image c := by
    classical
    rw [← image_image, image_univ_equiv]
  unfold hidingGap
  rw [himg, Fintype.card_congr e, Fintype.card_congr ((Equiv.refl B).prodCongr e)]
  congr 1
  refine sum_congr rfl fun v _ => ?_
  have h1 : #{y : Salt' | ((My ∘ e) y, (c ∘ e) y) = v} = #{y : Salt | (My y, c y) = v} :=
    card_equiv e fun y => by simp
  have h2 : #{p : B × Salt' | (p.1, (c ∘ e) p.2) = v} = #{p : B × Salt | (p.1, c p.2) = v} :=
    card_equiv ((Equiv.refl B).prodCongr e) fun p => by simp
  rw [h1, h2]

theorem treeMy_treeKey_eq_comp (κ : Fin 2047 → Bool) :
    treeMy (treeKey κ) = hankel (n := 512) (l := 1536) (bitsZ 2047 κ) ∘ bitsZ 1536 :=
  funext fun y => by
    rw [Function.comp_apply]
    exact treeMy_treeKey κ y

theorem hidingGap_tree (ν : ByteArray) (κ : Fin 2047 → Bool) :
    hidingGap (treeMy (treeKey κ)) (treeC ν) =
      hidingGap (hankel (n := 512) (l := 1536) (bitsZ 2047 κ)) (treeC ν ∘ (bitsZ 1536).symm) := by
  have hc : (treeC ν ∘ (bitsZ 1536).symm) ∘ bitsZ 1536 = treeC ν := by
    rw [Function.comp_assoc, Equiv.symm_comp_self, Function.comp_id]
  rw [← hidingGap_comp_equiv (bitsZ 1536) (hankel (n := 512) (l := 1536) (bitsZ 2047 κ))
    (treeC ν ∘ (bitsZ 1536).symm), hc, treeMy_treeKey_eq_comp]

theorem sum_hidingGap_tree (ν : ByteArray) :
    ∑ κ : Fin 2047 → Bool, hidingGap (treeMy (treeKey κ)) (treeC ν) =
      ∑ κ : Fin (512 + 1536 - 1) → ZMod 2,
        hidingGap (hankel (n := 512) (l := 1536) κ) (treeC ν ∘ (bitsZ 1536).symm) :=
  Fintype.sum_equiv (bitsZ 2047) _ _ (hidingGap_tree ν)

theorem card_image_le_of_size64 {Dom : Type} [Fintype Dom] (c : Dom → ByteArray) (hc : ∀ y, (c y).size = 64) :
    #(univ.image c) ≤ 2 ^ 512 := by
  have hlen : ∀ y, (c y).data.toList.length = 64 := fun y => by rw [Array.length_toList, ByteArray.size_data, hc]
  have hinj : Set.InjOn (bitsOf : ByteArray → Bits) (univ.image c : Finset ByteArray) := by
    rintro _ ha _ hb hab
    obtain ⟨y, -, rfl⟩ := mem_image.1 (mem_coe.1 ha)
    obtain ⟨y', -, rfl⟩ := mem_image.1 (mem_coe.1 hb)
    refine ByteArray.ext (Array.toList_inj.mp (eq_of_bitL ((hlen y).trans (hlen y').symm) fun t ht => ?_))
    rw [hlen y] at ht
    have h := congrFun hab ⟨t, ht⟩
    simp only [bitsOf] at h
    revert h
    cases bitL (c y).data.toList t <;> cases bitL (c y').data.toList t <;> decide
  calc #(univ.image c) = #((univ.image c).image (bitsOf : ByteArray → Bits)) := (card_image_of_injOn hinj).symm
    _ ≤ Fintype.card Bits := card_le_univ _
    _ = 2 ^ 512 := by rw [Fintype.card_fun, ZMod.card, Fintype.card_fin]

theorem card_image_treeC (ν : ByteArray) : #(univ.image (treeC ν)) ≤ 2 ^ 512 :=
  card_image_le_of_size64 (treeC ν) fun _ => Flock.Sha512.hash_size _

theorem sum_hidingGap_tree_le (ν : ByteArray) :
    ∑ κ : Fin 2047 → Bool, hidingGap (treeMy (treeKey κ)) (treeC ν) ≤
      Fintype.card (Fin 2047 → Bool) * (1 / 2 ^ 257) := by
  rw [sum_hidingGap_tree, Fintype.card_congr (bitsZ 2047)]
  exact FlockSoundness.CROnly.hm96_sha512_sum (ZMod.card 2) _
    (card_image_le_of_size64 _ fun _ => Flock.Sha512.hash_size _)

theorem avg_hidingGap_tree_le (ν : ByteArray) :
    avg (fun κ : Fin 2047 → Bool => ENNReal.ofReal (hidingGap (treeMy (treeKey κ)) (treeC ν))) ≤
      ENNReal.ofReal (1 / 2 ^ 257) :=
  FlockSoundness.CROnly.avg_ofReal_le _ (fun _ => FlockSoundness.CROnly.hidingGap_nonneg _ _) (sum_hidingGap_tree_le ν)

end Verity.Game
