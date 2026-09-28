import FlockSoundness.Merkle

/-!
# Zero knowledge: the verifier's coin commitment is binding (T6)

The one computational step of the public-circuit zero-knowledge proof (`docs/zk-proof-public.md` in the Project store, §4.1
T6). The verifier commits to every coin of a session with `coin-tree/hm96-sha512/v1` (`backends/flock/live/src/coin_tree.rs`,
M2): per (stream, round), a block of coins is chained, `T_i = H(CHAIN ‖ c_i ‖ T_{i+1})`; the round's digest is
`x = H(ROUND ‖ stream ‖ round ‖ T_0)`; the leaf is the hm96 leaf of `x` under a salt, `H(lp ‖ b ‖ c)` with `b = mix(x, y)`
and `c = H(sp ‖ y)`; the leaves sit in one SHA-512 tree. An opening of `n` coins is the coins, `T_n`, the salt and the path.

`coin_opening_binding`: two openings of one (stream, round) position with the same number of coins that both reach the
root and disagree on a coin give an explicit collision of `H`. So a rewound verifier that opens other coins than it did in
the extraction run has found a SHA-512 collision, which is how the proof's simulator turns `⊥_bind` into A2's bound.

The model is `FlockSoundness.Merkle`'s: one hash on byte strings, injective fixed-length encodings. `mix` stands for
`x ⊕ M(key)·y`, of which the proof needs only that it is injective in `x` for each salt.
-/

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

/-- Two chains of equally many coins that disagree on a coin but reach one `T_0` give a collision. -/
theorem chain_binding (tag : List UInt8) :
    ∀ (cs cs' : List Coin) (Tn Tn' : D), cs.length = cs'.length → cs ≠ cs' →
      chainTop H E tag Tn cs = chainTop H E tag Tn' cs' → Collision H
  | [], [], _, _, _, hne, _ => absurd rfl hne
  | [], _ :: _, _, _, hlen, _, _ => by simp at hlen
  | _ :: _, [], _, _, hlen, _, _ => by simp at hlen
  | c :: rest, c' :: rest', Tn, Tn', hlen, hne, heq => by
      simp only [chainTop, List.foldr_cons] at heq
      by_cases hin : tag ++ E.coin c ++ E.dig (chainTop H E tag Tn rest)
          = tag ++ E.coin c' ++ E.dig (chainTop H E tag Tn' rest')
      · obtain ⟨hc, ht⟩ := append3_inj (E.coin_len c c') hin
        have hc' : c = c' := E.coin_inj hc
        have hrest : rest ≠ rest' := fun h => hne (by rw [hc', h])
        exact chain_binding tag rest rest' Tn Tn' (by simpa using hlen) hrest (E.dig_inj ht)
      · exact ⟨_, _, hin, heq⟩

/-- **T6: the coin commitment is binding.** Two openings of one (stream, round) position, with equally many coins and a
coin on which they disagree, that both walk from their leaves to one root give an explicit collision of `H`. -/
theorem coin_opening_binding (mix : D → Salt → D) (hmix : ∀ y, Function.Injective (fun x => mix x y))
    (chainTag roundTag pos lp sp : List UInt8) (root : D) (idx : ℕ)
    (cs cs' : List Coin) (Tn Tn' : D) (y y' : Salt) (sibs sibs' : List D)
    (hlen : cs.length = cs'.length) (hne : cs ≠ cs') (hsibs : sibs.length = sibs'.length)
    (MerkleE : Enc Unit D)
    (h : walk H MerkleE idx (coinLeaf H E mix chainTag roundTag pos lp sp Tn cs y) sibs = root)
    (h' : walk H MerkleE idx (coinLeaf H E mix chainTag roundTag pos lp sp Tn' cs' y') sibs' = root) :
    Collision H := by
  by_cases hleaf : coinLeaf H E mix chainTag roundTag pos lp sp Tn cs y
      = coinLeaf H E mix chainTag roundTag pos lp sp Tn' cs' y'
  · -- the leaves agree: split the hm96 leaf, then the round digest, then the chain
    unfold coinLeaf at hleaf
    set T0 := chainTop H E chainTag Tn cs
    set T0' := chainTop H E chainTag Tn' cs'
    set x := H (roundTag ++ pos ++ E.dig T0)
    set x' := H (roundTag ++ pos ++ E.dig T0')
    by_cases hbc : lp ++ E.dig (mix x y) ++ E.dig (H (sp ++ E.salt y))
        = lp ++ E.dig (mix x' y') ++ E.dig (H (sp ++ E.salt y'))
    · obtain ⟨hb, hc⟩ := append3_inj (E.dig_len _ _) hbc
      have hc' := E.dig_inj hc
      by_cases hy : y = y'
      · subst hy
        have hx : x = x' := hmix y (E.dig_inj hb)
        by_cases hT : T0 = T0'
        · exact chain_binding H E chainTag cs cs' Tn Tn' hlen hne hT
        · refine ⟨_, _, fun hin => hT ?_, hx⟩
          have := List.append_cancel_left hin
          exact E.dig_inj this
      · refine ⟨_, _, fun hin => hy ?_, hc'⟩
        exact E.salt_inj (List.append_cancel_left hin)
    · exact ⟨_, _, hbc, hleaf⟩
  · -- the leaves differ but the walks meet: a node collides
    exact walk_binding H MerkleE sibs sibs' idx _ _ hsibs hleaf (h.trans h'.symm)

end FlockSoundness.ZK
