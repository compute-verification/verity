module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.SamplingGeneric.TwoAt
public import Proofs.Flock.Claims.Bridge.Hm96Tree
public import Proofs.Flock.Claims.Complete.Json.Goals
public import Proofs.Flock.Level3.ArithFacts
public import Proofs.Flock.Soundness.ZK.Hiding

/-! SPOT[Flock]'s stage bytes are uniform on its coins: A3 at each stage's call (`callsUniform`, `stageUniform`), a
theorem with no premise on the operating system. -/

@[expose] public section

namespace SpotFlock

theorem coinByte_injective : Function.Injective fun (c : FlockDeploy.Coin) (_ : Fin 1) => coinByte c := by
  intro c d h
  have hb : ∀ i : Fin L, coinByte c i = coinByte d i := fun i => congrFun (congrFun h 0) i
  apply Claims.Bridge.coin_bytes_injective
  have hc := FlockComplete.JsonRT.coin_bytes_length c
  have hd := FlockComplete.JsonRT.coin_bytes_length d
  refine List.ext_getElem (hc.trans hd.symm) fun n h1 h2 => ?_
  have hn : n < L := by
    rw [hc] at h1
    show n < 65536 * 16
    omega
  have e := congrArg Fin.val (hb ⟨n, hn⟩)
  simp only [coinByte, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h1, List.getElem?_eq_getElem h2,
    Option.getD_some] at e
  exact UInt8.toNat_inj.mp e

theorem card_coin : Fintype.card FlockDeploy.Coin = Fintype.card (Fin 1 → Fin L → Fin 256) := by
  have h₁ : Fintype.card FlockDeploy.Coin = (2 ^ 128) ^ 65536 := by
    rw [show Fintype.card FlockDeploy.Coin = Fintype.card (Fin 65536 → FlockLevel3.GF128) from
      Fintype.card_congr (Equiv.refl _), Fintype.card_fun, FlockLevel3.card_F, Fintype.card_fin]
  have h₂ : Fintype.card (Fin 1 → Fin L → Fin 256) = 256 ^ L := by
    rw [Fintype.card_fun, Fintype.card_fun, Fintype.card_fin 256, Fintype.card_fin L, Fintype.card_fin 1, pow_one]
  rw [h₁, h₂, ← pow_mul, show (256 : ℕ) = 2 ^ 8 by norm_num, ← pow_mul]
  exact congrArg (2 ^ ·) (by norm_num [L])

noncomputable def coinEquiv : FlockDeploy.Coin ≃ (Fin 1 → Fin L → Fin 256) :=
  Equiv.ofBijective (fun c _ => coinByte c)
    ((Fintype.bijective_iff_injective_and_card _).2 ⟨coinByte_injective, card_coin⟩)

theorem callsUniform : SamplingDeploy.CallsUniform (fun _ => L) fun _ => coinByte := fun _ E =>
  FlockSoundness.ZK.prCoin_congr_equiv coinEquiv E

noncomputable def stageEquiv (R : ℕ) : (Fin R → FlockDeploy.Coin) ≃ (Fin 1 → Fin (R * L) → Fin 256) :=
  ((Equiv.piCongrRight fun _ : Fin R => coinEquiv.trans (Equiv.funUnique (Fin 1) (Fin L → Fin 256))).trans
    ((Equiv.curry (Fin R) (Fin L) (Fin 256)).symm.trans
      (Equiv.arrowCongr finProdFinEquiv (Equiv.refl (Fin 256))))).trans
    (Equiv.funUnique (Fin 1) (Fin (R * L) → Fin 256)).symm

theorem stageEquiv_apply (R : ℕ) (cs : Fin R → FlockDeploy.Coin) : stageEquiv R cs = fun _ => stageByte cs := rfl

theorem stageUniform (R : ℕ) : SamplingDeploy.StageUniform (fun _ => R) (fun _ => R * L) fun _ => stageByte :=
  fun _ E => FlockSoundness.ZK.prCoin_congr_equiv (stageEquiv R) E

end SpotFlock
