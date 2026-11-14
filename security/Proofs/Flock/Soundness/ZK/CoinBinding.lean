module

public import Definitions.Flock.Merkle

/-!
# Zero knowledge: the verifier's coin tree (T6's model)

The model of the one computational step of the public-circuit zero-knowledge proof (`docs/zk-proof-public.md` in the
Project store, §4.1 T6). The verifier commits to every coin of a session with a coin tree
(`verity/ml/flock/live/src/coin_tree.rs`, Flock ZK), today the keyed `coin-tree/hm96-sha512/v2`: v1's tree with every hash
under the session's nonce. This file's model is v1's: per (stream, round), a block of coins is chained,
`T_i = H(CHAIN ‖ c_i ‖ T_{i+1})` (`chainTop`); the round's digest is `x = H(ROUND ‖ stream ‖ round ‖ T_0)`; the leaf is
the hm96 leaf of `x` under a salt, `H(lp ‖ b ‖ c)` with `b = mix(x, y)` and `c = H(sp ‖ y)` (`coinLeaf`); the leaves sit
in one SHA-512 tree. An opening of `n` coins is the coins, `T_n`, the salt and the path.

**Binding** is in `Proofs.Flock.Soundness.ZK.GK.Extract`: from two openings of one (stream, round) position with the same
number of coins that both reach the root and disagree on a coin, `coinColl` computes two byte strings that collide under
`H` (`coinColl_isColl`), and `coinCollKeyed` does so for the keyed tree (`coinCollKeyed_isColl`). No theorem here
concludes only that a collision exists.

The model is `FlockSoundness.Merkle`'s: one hash on byte strings, injective fixed-length encodings. `mix` stands for
`x ⊕ M(key)·y`, of which the binding proof needs only that it is injective in `x` for each salt.
-/

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

/-- `T_0` of a round: the chain from `T_n` down through the coins, `T_i = H(CHAIN ‖ c_i ‖ T_{i+1})`. -/
def chainTop (tag : List UInt8) (Tn : D) (cs : List Coin) : D :=
  cs.foldr (fun c t => H (tag ++ E.coin c ++ E.dig t)) Tn

/-- The round's hm96 leaf: `x = H(ROUND ‖ pos ‖ T_0)`, then `H(lp ‖ mix(x, y) ‖ H(sp ‖ y))`. -/
def coinLeaf (mix : D → Salt → D) (chainTag roundTag pos lp sp : List UInt8) (Tn : D) (cs : List Coin) (y : Salt) : D :=
  H (lp ++ E.dig (mix (H (roundTag ++ pos ++ E.dig (chainTop H E chainTag Tn cs))) y) ++ E.dig (H (sp ++ E.salt y)))

theorem append3_inj {p : List UInt8} {a a' : List UInt8} {b b' : List UInt8} (hlen : a.length = a'.length)
    (h : p ++ a ++ b = p ++ a' ++ b') : a = a' ∧ b = b' := by
  rw [List.append_assoc, List.append_assoc] at h
  exact List.append_inj (List.append_cancel_left h) hlen

end FlockSoundness.ZK
