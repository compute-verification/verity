import Proofs.Flock.Claims.ZK.GK
import Proofs.Flock.Soundness.ZK.GK.Extract

/-!
# The coin-tree check as an oracle program (`coin-tree/hm96-sha512/v2`)

The deployed masked prover reads the verifier's opening of one (stream, round) position, its coins, `T_n`, salt and
Merkle path, and walks the keyed coin tree from it, every hash `SHA-512(ν ‖ ·)` under the prover's session nonce `ν`,
to the root the verifier committed at `Hello` (`verity/ml/flock/live/PROTOCOL.md` §9). `treeChk` is that walk as an
`OC` program, so its `H` calls are logged: the chain from `T_n` up, the round digest, the salt hash, the hm96 leaf,
then the path (`reachK`). An answer that is not 64 bytes fails the check, as no SHA-512 output is.

**`Proto.binding_of_tree`**: a committed-coin protocol whose check is this walk, and whose coin is read from the
opening's coins, meets `Binding`, with no hypothesis on `H`. Two openings that pass and read as different coins
conflict under the root, so T6 located (`coinPairs_coll`) gives an aligned pair of hash inputs that collide under
`H(ν ‖ ·)`. Each is an input the walk asked (`coinPairs_mem`, `reachK_passes`), so the two checks' logs hold a
collision.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Merkle FlockSoundness.ZK

/-- A SHA-512 digest: 64 bytes. -/
abbrev Dg : Type := {m : Msg // m.length = 64}

/-- `H(ν ‖ ·)` as a digest: its answer when that is 64 bytes, zeros otherwise. -/
def keyedDg (H : Msg → Msg) (ν m : List UInt8) : Dg :=
  if h : (H (ν ++ m)).length = 64 then ⟨H (ν ++ m), h⟩ else ⟨List.replicate 64 0, by simp⟩

theorem keyedDg_of_len {H : Msg → Msg} {ν m : List UInt8} (h : (H (ν ++ m)).length = 64) :
    keyedDg H ν m = ⟨H (ν ++ m), h⟩ := by
  simp [keyedDg, h]

/-- One call of `H`, its answer read as a digest; an answer that is not one fails the check. -/
def hq (q : Msg) (k : Dg → OC Bool) : OC Bool :=
  .query q fun a => if h : a.length = 64 then k ⟨a, h⟩ else .ret false

section Walk

variable {Coin Salt : Type}

/-- The chain from `T_n` up, `T_i = H(ν ‖ CHAIN ‖ c_i ‖ T_{i+1})`, then `k` at `T_0`. -/
def chainK (ν tag : List UInt8) (E : CoinEnc Dg Coin Salt) : List Coin → Dg → (Dg → OC Bool) → OC Bool
  | [], Tn, k => k Tn
  | c :: r, Tn, k => chainK ν tag E r Tn fun t => hq (ν ++ (tag ++ E.coin c ++ E.dig t)) k

/-- The Merkle path from a node up, then `k` at the node it reaches. -/
def walkK (ν : List UInt8) (ME : Enc Unit Dg) : ℕ → Dg → List Dg → (Dg → OC Bool) → OC Bool
  | _, a, [], k => k a
  | idx, a, s :: r, k => hq (ν ++ GK.upIn ME idx a s) fun b => walkK ν ME (idx / 2) b r k

/-- **An opening's walk**: the chain, the round digest `H(ν ‖ ROUND ‖ pos ‖ T_0)`, the salt hash `H(ν ‖ sp ‖ y)`, the
hm96 leaf, the path; then `k` at the root it reaches. -/
def reachK (ν : List UInt8) (T : GK.CoinTree Dg Coin Salt) (op : GK.CoinOpening Dg Coin Salt) (k : Dg → OC Bool) :
    OC Bool :=
  chainK ν T.chainTag T.E op.cs op.Tn fun t0 =>
  hq (ν ++ (T.roundTag ++ op.pos ++ T.E.dig t0)) fun xd =>
  hq (ν ++ (T.sp ++ T.E.salt op.y)) fun cd =>
  hq (ν ++ (T.lp ++ T.E.dig (T.mix xd op.y) ++ T.E.dig cd)) fun l =>
  walkK ν T.ME op.idx l op.sibs k

/-- **The coin-tree check**: an opening (`none` if the message does not parse as one) of the position `pos` at leaf
`idx`, with `n` coins and `depth` path nodes, that reaches `root`. -/
def treeChk (T : GK.CoinTree Dg Coin Salt) (ν : List UInt8) (root : Dg) (pos : List UInt8) (idx n depth : ℕ) :
    Option (GK.CoinOpening Dg Coin Salt) → OC Bool
  | none => .ret false
  | some op =>
      if op.pos = pos ∧ op.idx = idx ∧ op.cs.length = n ∧ op.sibs.length = depth then
        reachK ν T op fun r => .ret (decide (r = root))
      else .ret false

/-! ## The inputs a walk hashes -/

/-- The chain's hash inputs, `T_0`'s first. -/
def chainIns (Hd : List UInt8 → Dg) (tag : List UInt8) (E : CoinEnc Dg Coin Salt) : List Coin → Dg → List (List UInt8)
  | [], _ => []
  | c :: r, Tn => (tag ++ E.coin c ++ E.dig (chainTop Hd E tag Tn r)) :: chainIns Hd tag E r Tn

/-- The path's hash inputs, from the leaf up. -/
def walkIns (Hd : List UInt8 → Dg) (ME : Enc Unit Dg) : ℕ → Dg → List Dg → List (List UInt8)
  | _, _, [] => []
  | idx, a, s :: r => GK.upIn ME idx a s :: walkIns Hd ME (idx / 2) (up Hd ME idx a s) r

/-- The round digest's input. -/
def rIn (Hd : List UInt8 → Dg) (T : GK.CoinTree Dg Coin Salt) (op : GK.CoinOpening Dg Coin Salt) : List UInt8 :=
  T.roundTag ++ op.pos ++ T.E.dig (chainTop Hd T.E T.chainTag op.Tn op.cs)

/-- The salt hash's input. -/
def sIn (T : GK.CoinTree Dg Coin Salt) (op : GK.CoinOpening Dg Coin Salt) : List UInt8 := T.sp ++ T.E.salt op.y

/-- The hm96 leaf's input. -/
def lIn (Hd : List UInt8 → Dg) (T : GK.CoinTree Dg Coin Salt) (op : GK.CoinOpening Dg Coin Salt) : List UInt8 :=
  T.lp ++ T.E.dig (T.mix (Hd (rIn Hd T op)) op.y) ++ T.E.dig (Hd (sIn T op))

/-- **Every input an opening's walk hashes.** -/
def opIns (Hd : List UInt8 → Dg) (T : GK.CoinTree Dg Coin Salt) (op : GK.CoinOpening Dg Coin Salt) :
    List (List UInt8) :=
  chainIns Hd T.chainTag T.E op.cs op.Tn ++
    rIn Hd T op :: sIn T op :: lIn Hd T op :: walkIns Hd T.ME op.idx (Hd (lIn Hd T op)) op.sibs

/-! ## A passing walk asks every input -/

/-- **`p` passing asks every input of `ins` under `ν` and gets a digest, and `k` then passes, within `p`'s calls.** -/
def Passes (H : Msg → Msg) (ν : List UInt8) (p : OC Bool) (ins : List (List UInt8)) (k : OC Bool) : Prop :=
  (p.eval H).1 = true → (∀ i ∈ ins, (ν ++ i, H (ν ++ i)) ∈ (p.eval H).2 ∧ (H (ν ++ i)).length = 64) ∧
    (k.eval H).1 = true ∧ (k.eval H).2 ⊆ (p.eval H).2

variable (H : Msg → Msg) (ν : List UInt8)

theorem Passes.refl (p : OC Bool) : Passes H ν p [] p := fun h => ⟨by simp, h, List.Subset.refl _⟩

theorem Passes.trans {p q k : OC Bool} {ins ins' : List (List UInt8)} (h₁ : Passes H ν p ins q)
    (h₂ : Passes H ν q ins' k) : Passes H ν p (ins ++ ins') k := by
  intro hp
  obtain ⟨a₁, hq₁, s₁⟩ := h₁ hp
  obtain ⟨a₂, hk, s₂⟩ := h₂ hq₁
  refine ⟨fun i hi => ?_, hk, List.Subset.trans s₂ s₁⟩
  rcases List.mem_append.mp hi with hi | hi
  · exact a₁ i hi
  · exact ⟨s₁ (a₂ i hi).1, (a₂ i hi).2⟩

theorem Passes.mono {p k : OC Bool} {ins ins' : List (List UInt8)} (h : Passes H ν p ins k) (hs : ins' ⊆ ins) :
    Passes H ν p ins' k := fun hp => ⟨fun i hi => (h hp).1 i (hs hi), (h hp).2⟩

theorem hq_passes (i : List UInt8) (k : Dg → OC Bool) : Passes H ν (hq (ν ++ i) k) [i] (k (keyedDg H ν i)) := by
  intro hp
  by_cases hl : (H (ν ++ i)).length = 64
  · rw [keyedDg_of_len hl]
    simp only [hq, OC.eval, hl, ↓reduceDIte] at hp ⊢
    refine ⟨fun j hj => ?_, hp, List.subset_cons_self _ _⟩
    rw [List.mem_singleton.mp hj]
    exact ⟨List.mem_cons_self .., hl⟩
  · simp [hq, OC.eval, hl] at hp

theorem chainK_passes (tag : List UInt8) (E : CoinEnc Dg Coin Salt) : ∀ (cs : List Coin) (Tn : Dg) (k : Dg → OC Bool),
    Passes H ν (chainK ν tag E cs Tn k) (chainIns (keyedDg H ν) tag E cs Tn) (k (chainTop (keyedDg H ν) E tag Tn cs))
  | [], Tn, k => Passes.refl H ν _
  | c :: r, Tn, k => by
      have h := (chainK_passes tag E r Tn fun t => hq (ν ++ (tag ++ E.coin c ++ E.dig t)) k).trans H ν
        (hq_passes H ν (tag ++ E.coin c ++ E.dig (chainTop (keyedDg H ν) E tag Tn r)) k)
      exact h.mono H ν fun i hi => by
        simp only [chainIns, List.mem_cons] at hi
        rcases hi with rfl | hi
        · exact List.mem_append_right _ (List.mem_singleton_self _)
        · exact List.mem_append_left _ hi

theorem walkK_passes (ME : Enc Unit Dg) : ∀ (idx : ℕ) (a : Dg) (sibs : List Dg) (k : Dg → OC Bool),
    Passes H ν (walkK ν ME idx a sibs k) (walkIns (keyedDg H ν) ME idx a sibs) (k (walk (keyedDg H ν) ME idx a sibs))
  | _, a, [], k => Passes.refl H ν _
  | idx, a, s :: r, k => by
      have e : keyedDg H ν (GK.upIn ME idx a s) = up (keyedDg H ν) ME idx a s :=
        (GK.up_eq (keyedDg H ν) ME idx a s).symm
      have h := (hq_passes H ν (GK.upIn ME idx a s) fun b => walkK ν ME (idx / 2) b r k).trans H ν
        (walkK_passes ME (idx / 2) (keyedDg H ν (GK.upIn ME idx a s)) r k)
      rw [e] at h
      exact h

/-- **A passing walk asks every input it hashes**, and its continuation passes at the root the opening reaches. -/
theorem reachK_passes (T : GK.CoinTree Dg Coin Salt) (op : GK.CoinOpening Dg Coin Salt) (k : Dg → OC Bool) :
    Passes H ν (reachK ν T op k) (opIns (keyedDg H ν) T op) (k (T.reach (keyedDg H ν) op)) := by
  have h1 := chainK_passes H ν T.chainTag T.E op.cs op.Tn fun t0 =>
    hq (ν ++ (T.roundTag ++ op.pos ++ T.E.dig t0)) fun xd =>
    hq (ν ++ (T.sp ++ T.E.salt op.y)) fun cd =>
    hq (ν ++ (T.lp ++ T.E.dig (T.mix xd op.y) ++ T.E.dig cd)) fun l =>
    walkK ν T.ME op.idx l op.sibs k
  have h2 := hq_passes H ν (rIn (keyedDg H ν) T op) fun xd =>
    hq (ν ++ (T.sp ++ T.E.salt op.y)) fun cd =>
    hq (ν ++ (T.lp ++ T.E.dig (T.mix xd op.y) ++ T.E.dig cd)) fun l =>
    walkK ν T.ME op.idx l op.sibs k
  have h3 := hq_passes H ν (sIn T op) fun cd =>
    hq (ν ++ (T.lp ++ T.E.dig (T.mix (keyedDg H ν (rIn (keyedDg H ν) T op)) op.y) ++ T.E.dig cd)) fun l =>
    walkK ν T.ME op.idx l op.sibs k
  have h4 := hq_passes H ν (lIn (keyedDg H ν) T op) fun l => walkK ν T.ME op.idx l op.sibs k
  have h5 := walkK_passes H ν T.ME op.idx (keyedDg H ν (lIn (keyedDg H ν) T op)) op.sibs k
  have h := (((h1.trans H ν h2).trans H ν h3).trans H ν h4).trans H ν h5
  refine h.mono H ν fun i hi => ?_
  simp only [opIns, List.mem_append, List.mem_cons] at hi ⊢
  tauto

/-! ## T6's aligned inputs are the walks' -/

theorem walkPairs_mem (Hd : List UInt8 → Dg) (ME : Enc Unit Dg) : ∀ (idx : ℕ) (a a' : Dg) (sibs sibs' : List Dg),
    ∀ p ∈ GK.walkPairs Hd ME idx a a' sibs sibs', p.1 ∈ walkIns Hd ME idx a sibs ∧ p.2 ∈ walkIns Hd ME idx a' sibs'
  | _, _, _, [], _ => by simp [GK.walkPairs]
  | _, _, _, _ :: _, [] => by simp [GK.walkPairs]
  | idx, a, a', s :: r, s' :: r' => by
      intro p hp
      simp only [GK.walkPairs, List.mem_cons] at hp
      rcases hp with rfl | hp
      · simp [walkIns]
      · obtain ⟨h1, h2⟩ := walkPairs_mem Hd ME _ _ _ r r' p hp
        exact ⟨List.mem_cons_of_mem _ h1, List.mem_cons_of_mem _ h2⟩

theorem chainPairs_mem (Hd : List UInt8 → Dg) (E : CoinEnc Dg Coin Salt) (tag : List UInt8) :
    ∀ (cs cs' : List Coin) (Tn Tn' : Dg), cs.length = cs'.length →
      ∀ p ∈ (GK.chainPairs Hd E tag cs cs' Tn Tn').2.2, p.1 ∈ chainIns Hd tag E cs Tn ∧ p.2 ∈ chainIns Hd tag E cs' Tn'
  | [], _, _, _, _ => by simp [GK.chainPairs]
  | _ :: _, [], _, _, h => by simp at h
  | c :: r, c' :: r', Tn, Tn', hlen => by
      intro p hp
      have hlen' : r.length = r'.length := by simpa using hlen
      obtain ⟨h1, h2⟩ := GK.chainPairs_tops Hd E tag r r' Tn Tn' hlen'
      simp only [GK.chainPairs, List.mem_cons] at hp
      rcases hp with rfl | hp
      · simp only [chainIns, List.mem_cons]
        rw [h1, h2]
        exact ⟨Or.inl rfl, Or.inl rfl⟩
      · obtain ⟨e1, e2⟩ := chainPairs_mem Hd E tag r r' Tn Tn' hlen' p hp
        exact ⟨List.mem_cons_of_mem _ e1, List.mem_cons_of_mem _ e2⟩

/-- **The inputs of T6's aligned pairs are the two walks'**: for openings of one position and leaf with equally many
coins, each pair's first input is one the first opening's walk hashes, and its second one the second's. -/
theorem coinPairs_mem (Hd : List UInt8 → Dg) (T : GK.CoinTree Dg Coin Salt) (o o' : GK.CoinOpening Dg Coin Salt)
    (hpos : o'.pos = o.pos) (hidx : o'.idx = o.idx) (hlen : o.cs.length = o'.cs.length) :
    ∀ p ∈ GK.coinPairs Hd T.E T.mix T.chainTag T.roundTag o.pos T.lp T.sp T.ME o.idx o.cs o'.cs o.Tn o'.Tn o.y o'.y
      o.sibs o'.sibs, p.1 ∈ opIns Hd T o ∧ p.2 ∈ opIns Hd T o' := by
  obtain ⟨h1, h2⟩ := GK.chainPairs_tops Hd T.E T.chainTag o.cs o'.cs o.Tn o'.Tn hlen
  have er : rIn Hd T o = T.roundTag ++ o.pos ++ T.E.dig (GK.chainPairs Hd T.E T.chainTag o.cs o'.cs o.Tn o'.Tn).1 := by
    rw [h1]; rfl
  have er' : rIn Hd T o' =
      T.roundTag ++ o.pos ++ T.E.dig (GK.chainPairs Hd T.E T.chainTag o.cs o'.cs o.Tn o'.Tn).2.1 := by
    rw [h2, ← hpos]; rfl
  intro p hp
  unfold GK.coinPairs at hp
  simp only [List.mem_append, List.mem_cons] at hp
  have el : lIn Hd T o = T.lp ++ T.E.dig (T.mix (Hd (T.roundTag ++ o.pos ++
      T.E.dig (GK.chainPairs Hd T.E T.chainTag o.cs o'.cs o.Tn o'.Tn).1)) o.y) ++ T.E.dig (Hd (T.sp ++ T.E.salt o.y)) := by
    rw [lIn, er]; rfl
  have el' : lIn Hd T o' = T.lp ++ T.E.dig (T.mix (Hd (T.roundTag ++ o.pos ++
      T.E.dig (GK.chainPairs Hd T.E T.chainTag o.cs o'.cs o.Tn o'.Tn).2.1)) o'.y) ++
        T.E.dig (Hd (T.sp ++ T.E.salt o'.y)) := by
    rw [lIn, er']; rfl
  rw [← el, ← el', ← er, ← er'] at hp
  unfold opIns
  rw [hidx]
  simp only [List.mem_append, List.mem_cons]
  rcases hp with hw | rfl | rfl | rfl | hc
  · obtain ⟨a, b⟩ := walkPairs_mem Hd T.ME o.idx _ _ o.sibs o'.sibs p hw
    exact ⟨Or.inr (Or.inr (Or.inr (Or.inr a))), Or.inr (Or.inr (Or.inr (Or.inr b)))⟩
  · exact ⟨Or.inr (Or.inr (Or.inr (Or.inl rfl))), Or.inr (Or.inr (Or.inr (Or.inl rfl)))⟩
  · exact ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inr (Or.inl rfl))⟩
  · exact ⟨Or.inr (Or.inl rfl), Or.inr (Or.inl rfl)⟩
  · obtain ⟨a, b⟩ := chainPairs_mem Hd T.E T.chainTag o.cs o'.cs o.Tn o'.Tn hlen p hc
    exact ⟨Or.inl a, Or.inl b⟩

/-! ## Binding -/

/-- **The coin-tree check binds in its own calls**: two openings of one position that pass and carry different coins
leave a collision among the two checks' `H` calls. -/
theorem treeChk_binding (T : GK.CoinTree Dg Coin Salt) (root : Dg) (pos : List UInt8) (idx n depth : ℕ)
    (op op' : GK.CoinOpening Dg Coin Salt)
    (h : ((treeChk T ν root pos idx n depth (some op)).eval H).1 = true)
    (h' : ((treeChk T ν root pos idx n depth (some op')).eval H).1 = true) (hne : op.cs ≠ op'.cs) :
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
  have e2 : walk (keyedDg H ν) T.ME op.idx (coinLeaf (keyedDg H ν) T.E T.mix T.chainTag T.roundTag op.pos T.lp T.sp
      op'.Tn op'.cs op'.y) op'.sibs = root := by
    rw [← hpos, ← hidx]; exact hr'
  obtain ⟨p, hpm, hpne, hpeq⟩ := GK.coinPairs_coll (keyedDg H ν) T.E T.mix T.mix_inj T.chainTag T.roundTag op.pos T.lp
    T.sp root op.idx op.cs op'.cs op.Tn op'.Tn op.y op'.y op.sibs op'.sibs hlen hne hsibs T.ME hr e2
  obtain ⟨m1, m2⟩ := coinPairs_mem (keyedDg H ν) T op op' hpos hidx hlen p hpm
  obtain ⟨l1, len1⟩ := hA p.1 m1
  obtain ⟨l2, len2⟩ := hA' p.2 m2
  rw [keyedDg_of_len len1, keyedDg_of_len len2] at hpeq
  refine (findColl_ne_none_iff _).mpr ⟨_, List.mem_append_left _ l1, _, List.mem_append_right _ l2,
    fun e => hpne (List.append_cancel_left e), congrArg Subtype.val hpeq⟩

end Walk

/-- **`Binding` for a protocol whose check is the coin-tree walk** (`coin-tree/hm96-sha512/v2`): at statement `x`, nonce
`ν` (its bytes `nb ν` key the tree) and commitment `ρ`, coin `j`'s opening is checked by `treeChk` against the tree
`T x ρ` and root `root x ρ`, at the position, leaf, coin count and depth the round fixes; and the coin an opening reads as
is a function of its coins. -/
theorem Proto.binding_of_tree {W : Type} (G : Proto W) (H : Msg → Msg) {Coin Salt : Type}
    (T : Msg → Msg → GK.CoinTree Dg Coin Salt) (nb : G.Nonce → List UInt8) (root : Msg → Msg → Dg)
    (parse : Msg → Option (GK.CoinOpening Dg Coin Salt)) (pos : Msg → Msg → ℕ → List UInt8)
    (idx n depth : Msg → Msg → ℕ → ℕ) (dc : List Coin → G.Coin)
    (hchk : ∀ x ν ρ j m, G.chk x ν ρ j m =
      treeChk (T x ρ) (nb ν) (root x ρ) (pos x ρ j) (idx x ρ j) (n x ρ j) (depth x ρ j) (parse m))
    (hdec : ∀ m op, parse m = some op → G.dec m = dc op.cs) : G.Binding H := by
  intro x ν ρ j m m' h h' hne
  rw [hchk] at h h' ⊢
  rw [hchk]
  cases hm : parse m with
  | none => rw [hm] at h; simp [treeChk, OC.eval] at h
  | some op =>
    cases hm' : parse m' with
    | none => rw [hm'] at h'; simp [treeChk, OC.eval] at h'
    | some op' =>
      rw [hm] at h
      rw [hm'] at h'
      refine treeChk_binding H (nb ν) (T x ρ) (root x ρ) (pos x ρ j) (idx x ρ j) (n x ρ j) (depth x ρ j) op op' h h'
        fun e => hne ?_
      rw [hdec m op hm, hdec m' op' hm', e]

end Claims.ZKMal
