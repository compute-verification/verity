module

public import Proofs.Properties.Verity.SampledTwo.SecondCoins
public import Proofs.Flock.Claims.Recursion.RestFitsLift

/-!
# Any bytes are coins' bytes

Every byte string of one coin's length is a coin's bytes as `Select` reads them (`exists_src`), since any 16 bytes are a
field element's (`FlockDeploy.toBytes_surj`). So the reading's coins are well defined wherever the stages' bytes fit:
`coinsOfBytes` and `stagesOfBytes` prove the premises `CoinsOfBytes` and `StagesOfBytes` of the stages' coins, and
`coinsAgree_first_of` and `coinsAgree_second_of` state the stages' coins without them.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- Byte `t` of a coin as `Select` reads it is byte `t` of its elements' bytes. -/
theorem src_getElem? (c : FlockDeploy.Coin) (t : ℕ) :
    (SpotFlock.src c).data[t]? = if t < SpotFlock.L then some (c.bytes.getD t 0) else none := by
  rw [show (SpotFlock.src c).data = Array.ofFn fun i => UInt8.ofFin (SpotFlock.coinByte c i) from rfl,
    Array.getElem?_ofFn]
  by_cases ht : t < SpotFlock.L
  · rw [dif_pos ht, if_pos ht]
    refine congrArg some ?_
    apply UInt8.toNat_inj.1
    simp [SpotFlock.coinByte]
  · rw [dif_neg ht, if_neg ht]

/-- **Any coin-long bytes are a coin's.** -/
theorem exists_src (B : ByteArray) (hB : B.size = SpotFlock.L) : ∃ c : FlockDeploy.Coin, SpotFlock.src c = B := by
  obtain ⟨vs, hlen, hv⟩ := FlockDeploy.colBytes_surj 65536 B.data.toList (by
    rw [Array.length_toList, ByteArray.size_data, hB]
    rfl)
  have hmap : (List.finRange 65536).map (fun i : Fin 65536 => vs[i.val]'(by rw [hlen]; exact i.isLt)) = vs := by
    apply List.ext_getElem
    · rw [List.length_map, List.length_finRange, hlen]
    · intro n h1 h2
      rw [List.getElem_map, List.getElem_finRange]
      rfl
  refine ⟨fun i => vs[i.val]'(by rw [hlen]; exact i.isLt), ?_⟩
  have hc : FlockDeploy.Coin.bytes (fun i => vs[i.val]'(by rw [hlen]; exact i.isLt)) = B.data.toList := by
    rw [← hv]
    unfold FlockDeploy.Coin.bytes FlockSoundness.Refine.colBytes
    conv_rhs => rw [← hmap]
    rw [List.flatMap_map]
  ext1
  apply Array.ext_getElem?
  intro t
  rw [src_getElem? (fun i => vs[i.val]'(by rw [hlen]; exact i.isLt)) t, hc]
  by_cases ht : t < SpotFlock.L
  · rw [if_pos ht, List.getD_eq_getElem?_getD, Array.getElem?_toList]
    have : t < B.data.size := by rw [ByteArray.size_data, hB]; exact ht
    rw [Array.getElem?_eq_getElem this]
    rfl
  · rw [if_neg ht]
    symm
    apply Array.getElem?_eq_none
    rw [ByteArray.size_data, hB]
    omega

/-- **Bytes are a coin's prefix** (`CoinsOfBytes`). -/
theorem coinsOfBytes : CoinsOfBytes := by
  intro b hb
  apply exists_src
  rw [ByteArray.size_append]
  show b.size + (Array.replicate (SpotFlock.L - b.size) (0 : UInt8)).size = SpotFlock.L
  rw [Array.size_replicate]
  omega

theorem div_lt_of {R x : ℕ} (hx : x < R * SpotFlock.L) : x / SpotFlock.L < R :=
  Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hx)

/-- Byte `x` of `R` rounds' coins, `x` below `R · L`, is byte `x % L` of coin `x / L`. -/
theorem stageSrc_getElem? {R : ℕ} (cs : Fin R → FlockDeploy.Coin) {x : ℕ} (hx : x < R * SpotFlock.L) :
    (SpotFlock.stageSrc cs).data[x]? =
      (SpotFlock.src (cs ⟨x / SpotFlock.L, div_lt_of hx⟩)).data[x % SpotFlock.L]? := by
  have hL : x % SpotFlock.L < SpotFlock.L := Nat.mod_lt _ (by unfold SpotFlock.L; omega)
  rw [src_getElem?, if_pos hL,
    show (SpotFlock.stageSrc cs).data = Array.ofFn fun i => UInt8.ofFin (SpotFlock.stageByte cs i) from rfl,
    Array.getElem?_ofFn, dif_pos hx,
    stageByte_at cs ⟨x / SpotFlock.L, div_lt_of hx⟩ ⟨x % SpotFlock.L, hL⟩ _ (Nat.div_add_mod' x _).symm]
  refine congrArg some ?_
  apply UInt8.toNat_inj.1
  simp [SpotFlock.coinByte]

/-- **Bytes are coins' prefix** (`StagesOfBytes`). -/
theorem stagesOfBytes : StagesOfBytes := by
  intro R b hb
  let B : ByteArray := b ++ ⟨Array.replicate (R * SpotFlock.L - b.size) 0⟩
  have hB : B.size = R * SpotFlock.L := by
    show (b ++ _).size = _
    rw [ByteArray.size_append]
    show b.size + (Array.replicate (R * SpotFlock.L - b.size) (0 : UInt8)).size = _
    rw [Array.size_replicate]
    omega
  have hchunk : ∀ j : Fin R, (B.extract (j * SpotFlock.L) (j * SpotFlock.L + SpotFlock.L)).size = SpotFlock.L := by
    intro j
    have := Nat.mul_le_mul_right SpotFlock.L (Nat.succ_le_of_lt j.isLt)
    rw [Nat.succ_mul] at this
    rw [ByteArray.size_extract, hB]
    generalize j.val * SpotFlock.L = m at *
    omega
  choose cs hcs using fun j : Fin R => exists_src _ (hchunk j)
  refine ⟨cs, ?_⟩
  ext1
  apply Array.ext_getElem?
  intro x
  by_cases hx : x < R * SpotFlock.L
  · rw [stageSrc_getElem? cs hx, hcs, ByteArray.data_extract, Array.getElem?_extract]
    have hL : x % SpotFlock.L < SpotFlock.L := Nat.mod_lt _ (by unfold SpotFlock.L; omega)
    have hlt : x % SpotFlock.L < min (x / SpotFlock.L * SpotFlock.L + SpotFlock.L) B.data.size -
        x / SpotFlock.L * SpotFlock.L := by
      have := Nat.div_add_mod' x SpotFlock.L
      have h2 : x / SpotFlock.L * SpotFlock.L + SpotFlock.L ≤ R * SpotFlock.L := by
        have := Nat.mul_le_mul_right SpotFlock.L (Nat.succ_le_of_lt (div_lt_of hx))
        rwa [Nat.succ_mul] at this
      rw [ByteArray.size_data, hB]
      generalize x / SpotFlock.L * SpotFlock.L = m at *
      omega
    rw [if_pos hlt, Nat.div_add_mod']
  · have h1 : (SpotFlock.stageSrc cs).data[x]? = none := by
      apply Array.getElem?_eq_none
      rw [show (SpotFlock.stageSrc cs).data = Array.ofFn fun i => UInt8.ofFin (SpotFlock.stageByte cs i) from rfl,
        Array.size_ofFn]
      omega
    have h2 : B.data[x]? = none := by
      apply Array.getElem?_eq_none
      rw [ByteArray.size_data, hB]
      omega
    rw [h1, h2]

/-- **The first stage's coins are the device's**, with no premise on the coins' bytes. -/
theorem coinsAgree_first_of (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop)
    (hShow : FirstShown C Good) (hDev : DevFirst C dev) :
    CoinsAgree A (formReading A C S dev) (FormAccepts C) Good .first :=
  coinsAgree_first A C S dev Good hShow hDev coinsOfBytes

/-- **The second stage's coins are the device's**, with no premise on the coins' bytes. -/
theorem coinsAgree_second_of (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop)
    (hShow : SecondShown C Good) (hDev : DevSecond A C dev) :
    CoinsAgree A (formReading A C S dev) (FormAccepts C) Good .second :=
  coinsAgree_second A C S dev Good hShow hDev stagesOfBytes

end Verity.SampledTwo
