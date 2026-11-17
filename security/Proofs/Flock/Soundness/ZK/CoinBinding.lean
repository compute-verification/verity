module

public import Definitions.Flock.Merkle

/-! The model of the verifier's coin tree `coin-tree/hm96-sha512/v1` (T6), the deployed keyed v2 tree without its
nonce; `mix` stands for `x ⊕ M(key)·y`, of which binding needs only that it is injective in `x` for each salt. -/

@[expose] public section

namespace FlockSoundness.ZK

open FlockSoundness.Merkle

variable {D Coin Salt : Type}

/-- The byte encodings the coin tree uses: a digest, a coin (16 bytes), a salt (192 bytes), each injective, and each of one
length. -/
structure CoinEnc (D Coin Salt : Type) where
  dig : D → List UInt8
  coin : Coin → List UInt8
  salt : Salt → List UInt8
  dig_inj : Function.Injective dig
  coin_inj : Function.Injective coin
  salt_inj : Function.Injective salt
  dig_len : ∀ a b : D, (dig a).length = (dig b).length
  coin_len : ∀ a b : Coin, (coin a).length = (coin b).length

variable (H : List UInt8 → D) (E : CoinEnc D Coin Salt)

def chainTop (tag : List UInt8) (Tn : D) (cs : List Coin) : D :=
  cs.foldr (fun c t => H (tag ++ E.coin c ++ E.dig t)) Tn

def coinLeaf (mix : D → Salt → D) (chainTag roundTag pos lp sp : List UInt8) (Tn : D) (cs : List Coin) (y : Salt) : D :=
  H (lp ++ E.dig (mix (H (roundTag ++ pos ++ E.dig (chainTop H E chainTag Tn cs))) y) ++ E.dig (H (sp ++ E.salt y)))

theorem append3_inj {p : List UInt8} {a a' : List UInt8} {b b' : List UInt8} (hlen : a.length = a'.length)
    (h : p ++ a ++ b = p ++ a' ++ b') : a = a' ∧ b = b' := by
  rw [List.append_assoc, List.append_assoc] at h
  exact List.append_inj (List.append_cancel_left h) hlen

end FlockSoundness.ZK
