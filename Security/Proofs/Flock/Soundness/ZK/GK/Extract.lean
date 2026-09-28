import Proofs.Flock.Soundness.ZK.CoinBinding

/-!
# Zero knowledge: T6's extraction as an algorithm

`coin_opening_binding` (T6) proves that two openings of one coin position that disagree on a coin and reach one root give
a collision of `H`; its conclusion is that a collision exists. The simulator's `⊥_bind` term needs the collision computed,
so that a finder can output it and `SHA512CRStrict` bound its probability. This file computes it.

The extraction lines up the hash inputs of the two openings node by node, `coinPairs`: the Merkle path's steps
(`walkPairs`), the hm96 leaf, the salt hash, the round digest, and the chain from `T_0` down (`chainPairs`, bottom up, one
pass). `findColl` returns the first pair whose inputs differ and whose hashes agree. T6's case split shows such a pair
exists (`coinPairs_coll`), so the output is a collision (`coinColl_isColl`). It hashes each listed input once: at most
`2·(|path| + 3 + n)` evaluations of `H` for openings of `n` coins (`coinPairs_length_le`).

* `findColl`, `findColl_isColl`: the first colliding pair of a list;
* `walkPairs_coll`, `chainPairs_coll`: `walk_binding` and `chain_binding` with the collision located among the aligned
  inputs;
* `coinColl`, `coinColl_isColl`: T6's collision, computed; `coinCollKeyed`, `coinCollKeyed_isColl` for the keyed tree
  (`coin-tree/hm96-sha512/v2`, the hash `fun m => H (ν ++ m)`);
* `CoinTree`, `CoinOpening`, `Conflict`, `conflicts`, `bindColl`, `bindColl_isColl`: the same at the level of two views'
  lists of openings, the form the simulator's `⊥_bind` outcome takes.
-/

namespace FlockSoundness.ZK.GK

open FlockSoundness.Merkle

/-- A finder's output: two byte strings, or nothing (as `StrictCR.Pair`). -/
abbrev Pair : Type := Option (List UInt8 × List UInt8)

/-- The output is a collision of `H`: `SHA512CRStrict`'s event with `out = id`. -/
def IsColl {D : Type} (H : List UInt8 → D) (o : Pair) : Prop := ∃ x y, o = some (x, y) ∧ x ≠ y ∧ H x = H y

section Find

variable {D : Type} [DecidableEq D] (H : List UInt8 → D)

/-- The first pair of the list whose inputs differ and whose hashes agree. -/
def findColl (ps : List (List UInt8 × List UInt8)) : Pair :=
  ps.find? fun p => decide (p.1 ≠ p.2 ∧ H p.1 = H p.2)

theorem findColl_isColl {ps : List (List UInt8 × List UInt8)} (h : ∃ p ∈ ps, p.1 ≠ p.2 ∧ H p.1 = H p.2) :
    IsColl H (findColl H ps) := by
  obtain ⟨p, hp, hc⟩ := h
  have hs : (findColl H ps).isSome := List.find?_isSome.2 ⟨p, hp, by simpa using hc⟩
  obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 hs
  have hq' := List.find?_some hq
  simp only [decide_eq_true_eq] at hq'
  exact ⟨q.1, q.2, hq, hq'⟩

end Find

section Walk

variable {D Col : Type} (H : List UInt8 → D) (E : Enc Col D)

/-- The input of one step up the tree: `acc ‖ sib` at an even index, `sib ‖ acc` at an odd one. -/
def upIn (idx : ℕ) (acc sib : D) : List UInt8 :=
  if idx % 2 = 0 then E.dig acc ++ E.dig sib else E.dig sib ++ E.dig acc

theorem up_eq (idx : ℕ) (acc sib : D) : up H E idx acc sib = H (upIn E idx acc sib) := by
  unfold up upIn
  split_ifs <;> rfl

/-- The inputs of two walks' steps, side by side, from the leaves up. -/
def walkPairs : ℕ → D → D → List D → List D → List (List UInt8 × List UInt8)
  | idx, a, a', s :: r, s' :: r' =>
      (upIn E idx a s, upIn E idx a' s') :: walkPairs (idx / 2) (up H E idx a s) (up H E idx a' s') r r'
  | _, _, _, _, _ => []

theorem walkPairs_length_le : ∀ (idx : ℕ) (a a' : D) (sibs sibs' : List D),
    (walkPairs H E idx a a' sibs sibs').length ≤ sibs.length
  | _, _, _, [], _ => by simp [walkPairs]
  | _, _, _, _ :: _, [] => by simp [walkPairs]
  | idx, a, a', _ :: r, _ :: r' => by
      simp only [walkPairs, List.length_cons]
      exact Nat.succ_le_succ (walkPairs_length_le _ _ _ r r')

/-- **`walk_binding`, located**: two walks from different nodes that meet collide at a step, and the colliding inputs are
one of the aligned pairs. -/
theorem walkPairs_coll : ∀ (sibs sibs' : List D) (idx : ℕ) (a a' : D),
    sibs.length = sibs'.length → a ≠ a' → walk H E idx a sibs = walk H E idx a' sibs' →
      ∃ p ∈ walkPairs H E idx a a' sibs sibs', p.1 ≠ p.2 ∧ H p.1 = H p.2
  | [], [], _, a, a', _, hne, heq => absurd heq hne
  | [], _ :: _, _, _, _, hlen, _, _ => by simp at hlen
  | _ :: _, [], _, _, _, hlen, _, _ => by simp at hlen
  | s :: rest, s' :: rest', idx, a, a', hlen, hne, heq => by
      simp only [walk] at heq
      simp only [walkPairs, List.mem_cons]
      by_cases hup : up H E idx a s = up H E idx a' s'
      · refine ⟨_, Or.inl rfl, ?_, ?_⟩
        · intro hin
          unfold upIn at hin
          split_ifs at hin
          · exact hne (append_inj E hin).1
          · exact hne (append_inj E hin).2
        · rw [← up_eq H E idx a s, ← up_eq H E idx a' s']
          exact hup
      · obtain ⟨p, hp, hc⟩ := walkPairs_coll rest rest' (idx / 2) _ _ (by simpa using hlen) hup heq
        exact ⟨p, Or.inr hp, hc⟩

end Walk

section Chain

variable {D Coin Salt : Type} (H : List UInt8 → D) (E : CoinEnc D Coin Salt)

/-- Two chains, bottom up in one pass: their tops and the inputs of their links side by side, top first. -/
def chainPairs (tag : List UInt8) : List Coin → List Coin → D → D → D × D × List (List UInt8 × List UInt8)
  | c :: r, c' :: r', Tn, Tn' =>
      let below := chainPairs tag r r' Tn Tn'
      let x := tag ++ E.coin c ++ E.dig below.1
      let x' := tag ++ E.coin c' ++ E.dig below.2.1
      (H x, H x', (x, x') :: below.2.2)
  | _, _, Tn, Tn' => (Tn, Tn', [])

theorem chainPairs_tops (tag : List UInt8) : ∀ (cs cs' : List Coin) (Tn Tn' : D), cs.length = cs'.length →
    (chainPairs H E tag cs cs' Tn Tn').1 = chainTop H E tag Tn cs ∧
      (chainPairs H E tag cs cs' Tn Tn').2.1 = chainTop H E tag Tn' cs'
  | [], [], _, _, _ => ⟨rfl, rfl⟩
  | [], _ :: _, _, _, hlen => by simp at hlen
  | _ :: _, [], _, _, hlen => by simp at hlen
  | c :: r, c' :: r', Tn, Tn', hlen => by
      obtain ⟨h1, h2⟩ := chainPairs_tops tag r r' Tn Tn' (by simpa using hlen)
      simp only [chainPairs, chainTop, List.foldr_cons] at h1 h2 ⊢
      rw [h1, h2]
      exact ⟨rfl, rfl⟩

theorem chainPairs_length_le (tag : List UInt8) : ∀ (cs cs' : List Coin) (Tn Tn' : D),
    (chainPairs H E tag cs cs' Tn Tn').2.2.length ≤ cs.length
  | [], _, _, _ => by simp [chainPairs]
  | _ :: _, [], _, _ => by simp [chainPairs]
  | _ :: r, _ :: r', Tn, Tn' => by
      simp only [chainPairs, List.length_cons]
      exact Nat.succ_le_succ (chainPairs_length_le tag r r' Tn Tn')

/-- **`chain_binding`, located**: two chains of equally many coins that disagree on a coin and reach one `T_0` collide at
a link, and the colliding inputs are one of the aligned pairs. -/
theorem chainPairs_coll (tag : List UInt8) : ∀ (cs cs' : List Coin) (Tn Tn' : D), cs.length = cs'.length → cs ≠ cs' →
    chainTop H E tag Tn cs = chainTop H E tag Tn' cs' →
      ∃ p ∈ (chainPairs H E tag cs cs' Tn Tn').2.2, p.1 ≠ p.2 ∧ H p.1 = H p.2
  | [], [], _, _, _, hne, _ => absurd rfl hne
  | [], _ :: _, _, _, hlen, _, _ => by simp at hlen
  | _ :: _, [], _, _, hlen, _, _ => by simp at hlen
  | c :: rest, c' :: rest', Tn, Tn', hlen, hne, heq => by
      have hlen' : rest.length = rest'.length := by simpa using hlen
      obtain ⟨h1, h2⟩ := chainPairs_tops H E tag rest rest' Tn Tn' hlen'
      simp only [chainTop, List.foldr_cons] at heq
      simp only [chainPairs, List.mem_cons]
      rw [h1, h2]
      by_cases hin : tag ++ E.coin c ++ E.dig (chainTop H E tag Tn rest)
          = tag ++ E.coin c' ++ E.dig (chainTop H E tag Tn' rest')
      · obtain ⟨hc, ht⟩ := append3_inj (E.coin_len c c') hin
        have hc' : c = c' := E.coin_inj hc
        have hrest : rest ≠ rest' := fun h => hne (by rw [hc', h])
        obtain ⟨p, hp, hcol⟩ := chainPairs_coll tag rest rest' Tn Tn' hlen' hrest (E.dig_inj ht)
        exact ⟨p, Or.inr hp, hcol⟩
      · exact ⟨_, Or.inl rfl, hin, heq⟩

end Chain

section Opening

variable {D Coin Salt : Type} (H : List UInt8 → D) (E : CoinEnc D Coin Salt)

/-- **The aligned hash inputs of two openings of one coin position**: the Merkle path's steps from the leaves up, the hm96
leaves, the salt hashes, the round digests, and the chains from `T_0` down. -/
def coinPairs (mix : D → Salt → D) (chainTag roundTag pos lp sp : List UInt8) (MerkleE : Enc Unit D) (idx : ℕ)
    (cs cs' : List Coin) (Tn Tn' : D) (y y' : Salt) (sibs sibs' : List D) : List (List UInt8 × List UInt8) :=
  let ch := chainPairs H E chainTag cs cs' Tn Tn'
  let r := roundTag ++ pos ++ E.dig ch.1
  let r' := roundTag ++ pos ++ E.dig ch.2.1
  let s := sp ++ E.salt y
  let s' := sp ++ E.salt y'
  let l := lp ++ E.dig (mix (H r) y) ++ E.dig (H s)
  let l' := lp ++ E.dig (mix (H r') y') ++ E.dig (H s')
  walkPairs H MerkleE idx (H l) (H l') sibs sibs' ++ (l, l') :: (s, s') :: (r, r') :: ch.2.2

/-- The extraction's size: at most `|path| + 3 + n` aligned pairs, so at most `2·(|path| + 3 + n)` evaluations of `H`. -/
theorem coinPairs_length_le (mix : D → Salt → D) (chainTag roundTag pos lp sp : List UInt8) (MerkleE : Enc Unit D)
    (idx : ℕ) (cs cs' : List Coin) (Tn Tn' : D) (y y' : Salt) (sibs sibs' : List D) :
    (coinPairs H E mix chainTag roundTag pos lp sp MerkleE idx cs cs' Tn Tn' y y' sibs sibs').length ≤
      sibs.length + 3 + cs.length := by
  unfold coinPairs
  simp only [List.length_append, List.length_cons]
  have h1 := walkPairs_length_le H MerkleE idx
    (H (lp ++ E.dig (mix (H (roundTag ++ pos ++ E.dig (chainPairs H E chainTag cs cs' Tn Tn').1)) y) ++
      E.dig (H (sp ++ E.salt y))))
    (H (lp ++ E.dig (mix (H (roundTag ++ pos ++ E.dig (chainPairs H E chainTag cs cs' Tn Tn').2.1)) y') ++
      E.dig (H (sp ++ E.salt y')))) sibs sibs'
  have h2 := chainPairs_length_le H E chainTag cs cs' Tn Tn'
  omega

/-- **T6, located**: under `coin_opening_binding`'s hypotheses, one of the aligned pairs is a collision of `H`. -/
theorem coinPairs_coll (mix : D → Salt → D) (hmix : ∀ y, Function.Injective (fun x => mix x y))
    (chainTag roundTag pos lp sp : List UInt8) (root : D) (idx : ℕ)
    (cs cs' : List Coin) (Tn Tn' : D) (y y' : Salt) (sibs sibs' : List D)
    (hlen : cs.length = cs'.length) (hne : cs ≠ cs') (hsibs : sibs.length = sibs'.length)
    (MerkleE : Enc Unit D)
    (h : walk H MerkleE idx (coinLeaf H E mix chainTag roundTag pos lp sp Tn cs y) sibs = root)
    (h' : walk H MerkleE idx (coinLeaf H E mix chainTag roundTag pos lp sp Tn' cs' y') sibs' = root) :
    ∃ p ∈ coinPairs H E mix chainTag roundTag pos lp sp MerkleE idx cs cs' Tn Tn' y y' sibs sibs',
      p.1 ≠ p.2 ∧ H p.1 = H p.2 := by
  obtain ⟨hT1, hT2⟩ := chainPairs_tops H E chainTag cs cs' Tn Tn' hlen
  unfold coinPairs
  simp only [List.mem_append, List.mem_cons]
  rw [hT1, hT2]
  unfold coinLeaf at h h'
  set T0 := chainTop H E chainTag Tn cs
  set T0' := chainTop H E chainTag Tn' cs'
  set x := H (roundTag ++ pos ++ E.dig T0)
  set x' := H (roundTag ++ pos ++ E.dig T0')
  set l := lp ++ E.dig (mix x y) ++ E.dig (H (sp ++ E.salt y))
  set l' := lp ++ E.dig (mix x' y') ++ E.dig (H (sp ++ E.salt y'))
  by_cases hleaf : H l = H l'
  · by_cases hbc : l = l'
    · obtain ⟨hb, hc⟩ := append3_inj (E.dig_len _ _) hbc
      have hc' := E.dig_inj hc
      by_cases hy : y = y'
      · subst hy
        have hx : x = x' := hmix y (E.dig_inj hb)
        by_cases hT : T0 = T0'
        · obtain ⟨p, hp, hcol⟩ := chainPairs_coll H E chainTag cs cs' Tn Tn' hlen hne hT
          exact ⟨p, Or.inr (Or.inr (Or.inr (Or.inr hp))), hcol⟩
        · refine ⟨_, Or.inr (Or.inr (Or.inr (Or.inl rfl))), fun hin => hT ?_, hx⟩
          exact E.dig_inj (List.append_cancel_left hin)
      · refine ⟨_, Or.inr (Or.inr (Or.inl rfl)), fun hin => hy ?_, hc'⟩
        exact E.salt_inj (List.append_cancel_left hin)
    · exact ⟨_, Or.inr (Or.inl rfl), hbc, hleaf⟩
  · obtain ⟨p, hp, hcol⟩ := walkPairs_coll H MerkleE sibs sibs' idx _ _ hsibs hleaf (h.trans h'.symm)
    exact ⟨p, Or.inl hp, hcol⟩

variable [DecidableEq D]

/-- **T6's collision, computed**: the first aligned pair of the two openings that collides. -/
def coinColl (mix : D → Salt → D) (chainTag roundTag pos lp sp : List UInt8) (MerkleE : Enc Unit D) (idx : ℕ)
    (cs cs' : List Coin) (Tn Tn' : D) (y y' : Salt) (sibs sibs' : List D) : Pair :=
  findColl H (coinPairs H E mix chainTag roundTag pos lp sp MerkleE idx cs cs' Tn Tn' y y' sibs sibs')

/-- **T6 as an algorithm**: under `coin_opening_binding`'s hypotheses, `coinColl` outputs a collision of `H`. -/
theorem coinColl_isColl (mix : D → Salt → D) (hmix : ∀ y, Function.Injective (fun x => mix x y))
    (chainTag roundTag pos lp sp : List UInt8) (root : D) (idx : ℕ)
    (cs cs' : List Coin) (Tn Tn' : D) (y y' : Salt) (sibs sibs' : List D)
    (hlen : cs.length = cs'.length) (hne : cs ≠ cs') (hsibs : sibs.length = sibs'.length)
    (MerkleE : Enc Unit D)
    (h : walk H MerkleE idx (coinLeaf H E mix chainTag roundTag pos lp sp Tn cs y) sibs = root)
    (h' : walk H MerkleE idx (coinLeaf H E mix chainTag roundTag pos lp sp Tn' cs' y') sibs' = root) :
    IsColl H (coinColl H E mix chainTag roundTag pos lp sp MerkleE idx cs cs' Tn Tn' y y' sibs sibs') :=
  findColl_isColl H (coinPairs_coll H E mix hmix chainTag roundTag pos lp sp root idx cs cs' Tn Tn' y y' sibs sibs'
    hlen hne hsibs MerkleE h h')

/-- T6's collision for the keyed tree (`coin-tree/hm96-sha512/v2`, every hash `fun m => H (ν ++ m)`), as a collision of
`H` itself: the colliding inputs with the prefix `ν`. -/
def coinCollKeyed (ν : List UInt8) (mix : D → Salt → D) (chainTag roundTag pos lp sp : List UInt8) (MerkleE : Enc Unit D)
    (idx : ℕ) (cs cs' : List Coin) (Tn Tn' : D) (y y' : Salt) (sibs sibs' : List D) : Pair :=
  (coinColl (fun m => H (ν ++ m)) E mix chainTag roundTag pos lp sp MerkleE idx cs cs' Tn Tn' y y' sibs sibs').map
    fun p => (ν ++ p.1, ν ++ p.2)

/-- **T6 as an algorithm, keyed tree**: under `coin_opening_binding_keyed`'s hypotheses, `coinCollKeyed` outputs a
collision of `H`. -/
theorem coinCollKeyed_isColl (ν : List UInt8) (mix : D → Salt → D) (hmix : ∀ y, Function.Injective (fun x => mix x y))
    (chainTag roundTag pos lp sp : List UInt8) (root : D) (idx : ℕ)
    (cs cs' : List Coin) (Tn Tn' : D) (y y' : Salt) (sibs sibs' : List D)
    (hlen : cs.length = cs'.length) (hne : cs ≠ cs') (hsibs : sibs.length = sibs'.length)
    (MerkleE : Enc Unit D)
    (h : walk (fun m => H (ν ++ m)) MerkleE idx
      (coinLeaf (fun m => H (ν ++ m)) E mix chainTag roundTag pos lp sp Tn cs y) sibs = root)
    (h' : walk (fun m => H (ν ++ m)) MerkleE idx
      (coinLeaf (fun m => H (ν ++ m)) E mix chainTag roundTag pos lp sp Tn' cs' y') sibs' = root) :
    IsColl H (coinCollKeyed H E ν mix chainTag roundTag pos lp sp MerkleE idx cs cs' Tn Tn' y y' sibs sibs') := by
  obtain ⟨a, b, hab, hne', heq⟩ := coinColl_isColl (fun m => H (ν ++ m)) E mix hmix chainTag roundTag pos lp sp root idx
    cs cs' Tn Tn' y y' sibs sibs' hlen hne hsibs MerkleE h h'
  refine ⟨ν ++ a, ν ++ b, ?_, fun h => hne' (List.append_cancel_left h), heq⟩
  unfold coinCollKeyed
  rw [hab]
  rfl

end Opening

/-! ## Two views' openings -/

/-- The coin tree's fixed parameters (`coin-tree/hm96-sha512/v1`): the encodings, the Merkle tree's, the salt mix with
its injectivity, and the domain tags. -/
structure CoinTree (D Coin Salt : Type) where
  E : CoinEnc D Coin Salt
  ME : Enc Unit D
  mix : D → Salt → D
  mix_inj : ∀ y, Function.Injective (fun x => mix x y)
  chainTag : List UInt8
  roundTag : List UInt8
  lp : List UInt8
  sp : List UInt8

/-- An opening of one (stream, round) position: its bytes and leaf index, the coins, `T_n`, the salt and the path. -/
structure CoinOpening (D Coin Salt : Type) where
  pos : List UInt8
  idx : ℕ
  cs : List Coin
  Tn : D
  y : Salt
  sibs : List D

variable {D Coin Salt : Type} (H : List UInt8 → D) (T : CoinTree D Coin Salt)

/-- The root an opening reaches. -/
def CoinTree.reach (op : CoinOpening D Coin Salt) : D :=
  walk H T.ME op.idx (coinLeaf H T.E T.mix T.chainTag T.roundTag op.pos T.lp T.sp op.Tn op.cs op.y) op.sibs

/-- **Two openings conflict** under `root`: one position, equally many coins and path nodes, different coins, and both
reach `root`. -/
def Conflict (root : D) (op op' : CoinOpening D Coin Salt) : Prop :=
  op.pos = op'.pos ∧ op.idx = op'.idx ∧ op.cs.length = op'.cs.length ∧ op.sibs.length = op'.sibs.length ∧
    op.cs ≠ op'.cs ∧ T.reach H op = root ∧ T.reach H op' = root

instance instDecidableConflictOfDecidableEq [DecidableEq D] [DecidableEq Coin] (root : D) (op op' : CoinOpening D Coin Salt) :
    Decidable (Conflict H T root op op') := by
  unfold Conflict
  infer_instance

/-- One opening of each list, every pair. -/
def openPairs (os os' : List (CoinOpening D Coin Salt)) : List (CoinOpening D Coin Salt × CoinOpening D Coin Salt) :=
  os.flatMap fun op => os'.map fun op' => (op, op')

variable [DecidableEq D] [DecidableEq Coin]

/-- **`⊥_bind`**: some opening of the first list and some of the second conflict under `root`. -/
def conflicts (root : D) (os os' : List (CoinOpening D Coin Salt)) : Bool :=
  (openPairs os os').any fun q => decide (Conflict H T root q.1 q.2)

/-- **The `⊥_bind` collision**: T6's collision from the first conflicting pair of openings, nothing if none conflict. -/
def bindColl (root : D) (os os' : List (CoinOpening D Coin Salt)) : Pair :=
  match (openPairs os os').find? (fun q => decide (Conflict H T root q.1 q.2)) with
  | some (op, op') => coinColl H T.E T.mix T.chainTag T.roundTag op.pos T.lp T.sp T.ME op.idx op.cs op'.cs op.Tn op'.Tn
      op.y op'.y op.sibs op'.sibs
  | none => none

/-- **T6 as an algorithm, on views**: when two lists of openings conflict under a root, `bindColl` outputs a collision of
`H`. -/
theorem bindColl_isColl (root : D) (os os' : List (CoinOpening D Coin Salt)) (h : conflicts H T root os os' = true) :
    IsColl H (bindColl H T root os os') := by
  unfold conflicts at h
  have hs : ((openPairs os os').find? (fun q => decide (Conflict H T root q.1 q.2))).isSome :=
    List.find?_isSome.2 (List.any_eq_true.1 h)
  obtain ⟨⟨op, op'⟩, hq⟩ := Option.isSome_iff_exists.1 hs
  have hc := List.find?_some hq
  simp only [decide_eq_true_eq] at hc
  obtain ⟨hpos, hidx, hlen, hsibs, hne, h1, h2⟩ := hc
  unfold bindColl
  rw [hq]
  unfold CoinTree.reach at h1 h2
  rw [← hpos, ← hidx] at h2
  exact coinColl_isColl H T.E T.mix T.mix_inj T.chainTag T.roundTag op.pos T.lp T.sp root op.idx op.cs op'.cs op.Tn op'.Tn
    op.y op'.y op.sibs op'.sibs hlen hne hsibs T.ME h1 h2

/-- **The `⊥_bind` collision for the keyed tree** (`coin-tree/hm96-sha512/v2`, every hash `fun m => H (ν ++ m)`), as a
collision of `H`: the colliding inputs with the prefix `ν`. -/
def bindCollKeyed (ν : List UInt8) (root : D) (os os' : List (CoinOpening D Coin Salt)) : Pair :=
  (bindColl (fun m => H (ν ++ m)) T root os os').map fun p => (ν ++ p.1, ν ++ p.2)

/-- **T6 as an algorithm, on views, keyed tree**: when two lists of openings conflict under a root of the keyed tree,
`bindCollKeyed` outputs a collision of `H`. -/
theorem bindCollKeyed_isColl (ν : List UInt8) (root : D) (os os' : List (CoinOpening D Coin Salt))
    (h : conflicts (fun m => H (ν ++ m)) T root os os' = true) : IsColl H (bindCollKeyed H T ν root os os') := by
  obtain ⟨a, b, hab, hne, heq⟩ := bindColl_isColl (fun m => H (ν ++ m)) T root os os' h
  refine ⟨ν ++ a, ν ++ b, ?_, fun h => hne (List.append_cancel_left h), heq⟩
  unfold bindCollKeyed
  rw [hab]
  rfl

end FlockSoundness.ZK.GK
