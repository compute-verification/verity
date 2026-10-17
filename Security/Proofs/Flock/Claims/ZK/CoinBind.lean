import Proofs.Flock.Claims.ZK.CoinCheck

/-!
# The coin-tree check binds the whole opening

`treeChk_binding` binds an opening's coins. The deployed opening (`coin-tree/hm96-sha512/v2`, `Resp::Coins`) is bare
words: which of them are coins, `T_n`, salt and path is the statement's schedule, so the coin the prover hands on is read
from the whole opening. Every part of an opening is a hash input on its walk to the root, so two openings of one position
that pass and differ anywhere still collide:

* `walkPairs_coll_any`, `chainPairs_coll_any`, `coinPairs_coll_any`: `walkPairs_coll`, `chainPairs_coll` and
  `coinPairs_coll` with the two sides differing anywhere (a node or a sibling; a coin or `T_n`; a coin, `T_n`, the salt
  or the path);
* `treeChk_binding_any`: two openings of one position that pass and differ leave a collision among the two checks' `H`
  calls;
* `Proto.binding_of_open`: a committed-coin protocol whose check is the walk on an opening read at the round's context,
  and whose coin is a function of the opening's message, meets `Binding`, with no hypothesis on `H`.
-/

namespace FlockSoundness.ZK.GK

open FlockSoundness.Merkle

section Walk

variable {D Col : Type} (H : List UInt8 → D) (E : Enc Col D)

/-- **`walkPairs_coll` with the walks differing anywhere**: two walks of equal length that start at different nodes or
take different siblings, and meet, collide at an aligned step. -/
theorem walkPairs_coll_any : ∀ (sibs sibs' : List D) (idx : ℕ) (a a' : D),
    sibs.length = sibs'.length → (a ≠ a' ∨ sibs ≠ sibs') → walk H E idx a sibs = walk H E idx a' sibs' →
      ∃ p ∈ walkPairs H E idx a a' sibs sibs', p.1 ≠ p.2 ∧ H p.1 = H p.2
  | [], [], _, _, _, _, hne, heq => by
      rcases hne with hne | hne
      · exact absurd heq hne
      · exact absurd rfl hne
  | [], _ :: _, _, _, _, hlen, _, _ => by simp at hlen
  | _ :: _, [], _, _, _, hlen, _, _ => by simp at hlen
  | s :: rest, s' :: rest', idx, a, a', hlen, hne, heq => by
      simp only [walk] at heq
      simp only [walkPairs, List.mem_cons]
      have hlen' : rest.length = rest'.length := by simpa using hlen
      by_cases hup : up H E idx a s = up H E idx a' s'
      · by_cases hin : upIn E idx a s = upIn E idx a' s'
        · have hss : a = a' ∧ s = s' := by
            unfold upIn at hin
            split_ifs at hin
            · exact append_inj E hin
            · exact ⟨(append_inj E hin).2, (append_inj E hin).1⟩
          obtain ⟨rfl, rfl⟩ := hss
          have hr : rest ≠ rest' := fun h => by
            subst h
            rcases hne with hne | hne <;> exact hne rfl
          obtain ⟨p, hp, hc⟩ := walkPairs_coll_any rest rest' (idx / 2) _ _ hlen' (Or.inr hr) heq
          exact ⟨p, Or.inr hp, hc⟩
        · refine ⟨_, Or.inl rfl, hin, ?_⟩
          rw [← up_eq H E idx a s, ← up_eq H E idx a' s']
          exact hup
      · obtain ⟨p, hp, hc⟩ := walkPairs_coll_any rest rest' (idx / 2) _ _ hlen' (Or.inl hup) heq
        exact ⟨p, Or.inr hp, hc⟩

end Walk

section Chain

variable {D Coin Salt : Type} (H : List UInt8 → D) (E : CoinEnc D Coin Salt)

/-- **`chainPairs_coll` with the chains differing anywhere**: two chains of equally many coins that differ on a coin or
on `T_n`, and reach one `T_0`, collide at an aligned link. -/
theorem chainPairs_coll_any (tag : List UInt8) : ∀ (cs cs' : List Coin) (Tn Tn' : D), cs.length = cs'.length →
    (cs ≠ cs' ∨ Tn ≠ Tn') → chainTop H E tag Tn cs = chainTop H E tag Tn' cs' →
      ∃ p ∈ (chainPairs H E tag cs cs' Tn Tn').2.2, p.1 ≠ p.2 ∧ H p.1 = H p.2
  | [], [], _, _, _, hne, heq => by
      rcases hne with hne | hne
      · exact absurd rfl hne
      · exact absurd heq hne
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
        obtain rfl : c = c' := E.coin_inj hc
        have hrest : rest ≠ rest' ∨ Tn ≠ Tn' := by
          rcases hne with hne | hne
          · exact Or.inl fun h => hne (by rw [h])
          · exact Or.inr hne
        obtain ⟨p, hp, hcol⟩ := chainPairs_coll_any tag rest rest' Tn Tn' hlen' hrest (E.dig_inj ht)
        exact ⟨p, Or.inr hp, hcol⟩
      · exact ⟨_, Or.inl rfl, hin, heq⟩

end Chain

section Opening

variable {D Coin Salt : Type} (H : List UInt8 → D) (E : CoinEnc D Coin Salt)

/-- **T6, located, with the openings differing anywhere**: two openings of one position, of equally many coins and path
nodes, that differ on a coin, `T_n`, the salt or the path, and reach one root, leave a colliding aligned pair. -/
theorem coinPairs_coll_any (mix : D → Salt → D) (hmix : ∀ y, Function.Injective (fun x => mix x y))
    (chainTag roundTag pos lp sp : List UInt8) (root : D) (idx : ℕ)
    (cs cs' : List Coin) (Tn Tn' : D) (y y' : Salt) (sibs sibs' : List D)
    (hlen : cs.length = cs'.length) (hne : cs ≠ cs' ∨ Tn ≠ Tn' ∨ y ≠ y' ∨ sibs ≠ sibs')
    (hsibs : sibs.length = sibs'.length) (MerkleE : Enc Unit D)
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
  by_cases hw : H l = H l' ∧ sibs = sibs'
  · obtain ⟨hleaf, hs⟩ := hw
    by_cases hbc : l = l'
    · obtain ⟨hb, hc⟩ := append3_inj (E.dig_len _ _) hbc
      have hc' := E.dig_inj hc
      by_cases hy : y = y'
      · subst hy
        have hx : x = x' := hmix y (E.dig_inj hb)
        by_cases hT : T0 = T0'
        · have hne' : cs ≠ cs' ∨ Tn ≠ Tn' := by
            rcases hne with e | e | e | e
            · exact Or.inl e
            · exact Or.inr e
            · exact absurd rfl e
            · exact absurd hs e
          obtain ⟨p, hp, hcol⟩ := chainPairs_coll_any H E chainTag cs cs' Tn Tn' hlen hne' hT
          exact ⟨p, Or.inr (Or.inr (Or.inr (Or.inr hp))), hcol⟩
        · refine ⟨_, Or.inr (Or.inr (Or.inr (Or.inl rfl))), fun hin => hT ?_, hx⟩
          exact E.dig_inj (List.append_cancel_left hin)
      · refine ⟨_, Or.inr (Or.inr (Or.inl rfl)), fun hin => hy ?_, hc'⟩
        exact E.salt_inj (List.append_cancel_left hin)
    · exact ⟨_, Or.inr (Or.inl rfl), hbc, hleaf⟩
  · obtain ⟨p, hp, hcol⟩ := walkPairs_coll_any H MerkleE sibs sibs' idx _ _ hsibs (not_and_or.mp hw)
      (h.trans h'.symm)
    exact ⟨p, Or.inl hp, hcol⟩

end Opening

end FlockSoundness.ZK.GK

namespace Claims.ZKMal

open Claims FlockSoundness.Merkle FlockSoundness.ZK

section Walk

variable {Coin Salt : Type} (H : Msg → Msg) (ν : List UInt8)

theorem opening_eq {op op' : GK.CoinOpening Dg Coin Salt} (h1 : op.pos = op'.pos) (h2 : op.idx = op'.idx)
    (h3 : op.cs = op'.cs) (h4 : op.Tn = op'.Tn) (h5 : op.y = op'.y) (h6 : op.sibs = op'.sibs) : op = op' := by
  cases op
  cases op'
  simp only at h1 h2 h3 h4 h5 h6
  subst h1 h2 h3 h4 h5 h6
  rfl

/-- **The coin-tree check binds the whole opening in its own calls**: two different openings of one position that pass
leave a collision among the two checks' `H` calls. -/
theorem treeChk_binding_any (T : GK.CoinTree Dg Coin Salt) (root : Dg) (pos : List UInt8) (idx n depth : ℕ)
    (op op' : GK.CoinOpening Dg Coin Salt)
    (h : ((treeChk T ν root pos idx n depth (some op)).eval H).1 = true)
    (h' : ((treeChk T ν root pos idx n depth (some op')).eval H).1 = true) (hne : op ≠ op') :
    Claims.findColl (((treeChk T ν root pos idx n depth (some op)).eval H).2 ++
      ((treeChk T ν root pos idx n depth (some op')).eval H).2) ≠ none := by
  simp only [treeChk] at h h' ⊢
  by_cases hs : op.pos = pos ∧ op.idx = idx ∧ op.cs.length = n ∧ op.sibs.length = depth
  swap
  · rw [ite_eq_right hs] at h; simp [OC.eval] at h
  by_cases hs' : op'.pos = pos ∧ op'.idx = idx ∧ op'.cs.length = n ∧ op'.sibs.length = depth
  swap
  · rw [ite_eq_right hs'] at h'; simp [OC.eval] at h'
  rw [ite_eq_left hs] at h ⊢
  rw [ite_eq_left hs'] at h' ⊢
  obtain ⟨hA, hr, hsub⟩ := reachK_passes H ν T op _ h
  obtain ⟨hA', hr', hsub'⟩ := reachK_passes H ν T op' _ h'
  simp only [OC.eval, decide_eq_true_eq] at hr hr'
  obtain ⟨hp, hi, hn, hd⟩ := hs
  obtain ⟨hp', hi', hn', hd'⟩ := hs'
  have hpos : op'.pos = op.pos := hp'.trans hp.symm
  have hidx : op'.idx = op.idx := hi'.trans hi.symm
  have hlen : op.cs.length = op'.cs.length := hn.trans hn'.symm
  have hsibs : op.sibs.length = op'.sibs.length := hd.trans hd'.symm
  have hne' : op.cs ≠ op'.cs ∨ op.Tn ≠ op'.Tn ∨ op.y ≠ op'.y ∨ op.sibs ≠ op'.sibs := by
    by_contra hc
    simp only [not_or, not_not] at hc
    exact hne (opening_eq hpos.symm hidx.symm hc.1 hc.2.1 hc.2.2.1 hc.2.2.2)
  have e2 : walk (keyedDg H ν) T.ME op.idx (coinLeaf (keyedDg H ν) T.E T.mix T.chainTag T.roundTag op.pos T.lp T.sp
      op'.Tn op'.cs op'.y) op'.sibs = root := by
    rw [← hpos, ← hidx]; exact hr'
  obtain ⟨p, hpm, hpne, hpeq⟩ := GK.coinPairs_coll_any (keyedDg H ν) T.E T.mix T.mix_inj T.chainTag T.roundTag op.pos
    T.lp T.sp root op.idx op.cs op'.cs op.Tn op'.Tn op.y op'.y op.sibs op'.sibs hlen hne' hsibs T.ME hr e2
  obtain ⟨m1, m2⟩ := coinPairs_mem (keyedDg H ν) T op op' hpos hidx hlen p hpm
  obtain ⟨l1, len1⟩ := hA p.1 m1
  obtain ⟨l2, len2⟩ := hA' p.2 m2
  rw [keyedDg_of_len len1, keyedDg_of_len len2] at hpeq
  refine (findColl_ne_none_iff _).mpr ⟨_, List.mem_append_left _ l1, _, List.mem_append_right _ l2,
    fun e => hpne (List.append_cancel_left e), congrArg Subtype.val hpeq⟩

end Walk

/-- **`Binding` for a protocol whose check is the coin-tree walk on an opening read in context**: at statement `x`,
nonce `ν` and commitment `ρ`, coin `j`'s message is read as an opening at that round (`parse x ρ j`) and checked by
`treeChk` against the tree `T x ρ` and root `root x ρ`; and two messages that read as one opening there read as one
coin. -/
theorem Proto.binding_of_open {W : Type} (G : Proto W) (H : Msg → Msg) {Coin Salt : Type}
    (T : Msg → Msg → GK.CoinTree Dg Coin Salt) (nb : G.Nonce → List UInt8) (root : Msg → Msg → Dg)
    (parse : Msg → Msg → ℕ → Msg → Option (GK.CoinOpening Dg Coin Salt)) (pos : Msg → Msg → ℕ → List UInt8)
    (idx n depth : Msg → Msg → ℕ → ℕ)
    (hchk : ∀ x ν ρ j m, G.chk x ν ρ j m =
      treeChk (T x ρ) (nb ν) (root x ρ) (pos x ρ j) (idx x ρ j) (n x ρ j) (depth x ρ j) (parse x ρ j m))
    (hdec : ∀ x ρ j m m' op, parse x ρ j m = some op → parse x ρ j m' = some op → G.dec m = G.dec m') :
    G.Binding H := by
  intro x ν ρ j m m' h h' hne
  rw [hchk] at h h' ⊢
  rw [hchk]
  cases hm : parse x ρ j m with
  | none => rw [hm] at h; simp [treeChk, OC.eval] at h
  | some op =>
    cases hm' : parse x ρ j m' with
    | none => rw [hm'] at h'; simp [treeChk, OC.eval] at h'
    | some op' =>
      rw [hm] at h
      rw [hm'] at h'
      refine treeChk_binding_any H (nb ν) (T x ρ) (root x ρ) (pos x ρ j) (idx x ρ j) (n x ρ j) (depth x ρ j) op op'
        h h' fun e => hne ?_
      subst e
      exact hdec x ρ j m m' op hm hm'

end Claims.ZKMal
