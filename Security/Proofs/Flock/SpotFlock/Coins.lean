module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.SamplingGeneric.TwoAt
public import Proofs.Flock.Claims.Bridge.Hm96Tree
public import Proofs.Flock.Claims.Complete.Json.Goals
public import Proofs.Flock.Level3.ArithFacts
public import Proofs.Flock.Soundness.ZK.Hiding

/-!
# SPOT[Flock]'s stage bytes are uniform on its coins

Each stage of SPOT[Flock] reads its `L = 2^20` bytes off its round's coin (`coinByte`): the coin's `65536` field elements'
`16` bytes each, in order (`FlockDeploy.Coin.bytes`). Those bytes determine the coin (`Claims.Bridge.coin_bytes_injective`)
and there are as many coins as byte strings of length `L` (`2^128` elements, `65536` times), so the map is a bijection
(`coinEquiv`) and carries the uniform law on coins to the uniform law on the bytes. That is `CallsUniform` at SPOT's
byte maps (`callsUniform`), A3 at each stage's call, as a theorem with no premise on the operating system.
-/

@[expose] public section

namespace SpotFlock

/-- **A coin's stage bytes determine the coin.** -/
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

/-- **As many coins as stage byte strings**: `(2^128)^65536 = 256^(2^20)`. -/
theorem card_coin : Fintype.card FlockDeploy.Coin = Fintype.card (Fin 1 → Fin L → Fin 256) := by
  have h₁ : Fintype.card FlockDeploy.Coin = (2 ^ 128) ^ 65536 := by
    rw [show Fintype.card FlockDeploy.Coin = Fintype.card (Fin 65536 → FlockLevel3.GF128) from
      Fintype.card_congr (Equiv.refl _), Fintype.card_fun, FlockLevel3.card_F, Fintype.card_fin]
  have h₂ : Fintype.card (Fin 1 → Fin L → Fin 256) = 256 ^ L := by
    rw [Fintype.card_fun, Fintype.card_fun, Fintype.card_fin 256, Fintype.card_fin L, Fintype.card_fin 1, pow_one]
  rw [h₁, h₂, ← pow_mul, show (256 : ℕ) = 2 ^ 8 by norm_num, ← pow_mul]
  exact congrArg (2 ^ ·) (by norm_num [L])

/-- **The stage bytes and the coins, one for one** (`coinByte`). -/
noncomputable def coinEquiv : FlockDeploy.Coin ≃ (Fin 1 → Fin L → Fin 256) :=
  Equiv.ofBijective (fun c _ => coinByte c)
    ((Fintype.bijective_iff_injective_and_card _).2 ⟨coinByte_injective, card_coin⟩)

/-- **A3 at SPOT[Flock]'s stage bytes**: the bytes each stage reads off a uniform coin are uniform bytes, at every
statement. -/
theorem callsUniform : SamplingDeploy.CallsUniform (fun _ => L) fun _ => coinByte := fun _ E =>
  FlockSoundness.ZK.prCoin_congr_equiv coinEquiv E

end SpotFlock
