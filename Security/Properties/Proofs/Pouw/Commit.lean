import Properties.Pouw.Commit

/-!
# Proofs of PoUW's commitment properties (`Properties.Pouw.Commit`)

The sizes and numbers by unfolding and evaluation in the kernel, the refusals and the accepted coins by unfolding the
port's definitions case by case. `CoinBinding` and `FrameInjective` are open.
-/

namespace Properties.Pouw.Commit

open Flock PouwCommit

/-- `ByteArray`'s `==` is its arrays', which is lawful. -/
instance lawfulBEqByteArray : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

theorem sha512_size (m : ByteArray) : (Sha512.hash m).size = 64 := by
  unfold Sha512.hash; simp only [Id.run]; exact Array.size_ofFn

theorem sha256_size (m : ByteArray) : (Sha256.hash m).size = 32 := by
  unfold Sha256.hash; simp only [Id.run]; exact Array.size_ofFn

theorem run_size (h : Hash) (m : ByteArray) : (h.run m).size = h.bytes := by
  cases h
  · exact sha256_size m
  · exact sha512_size m

theorem frameHash_size (h : Hash) (tag : String) (parts : List ByteArray) : (frameHash h tag parts).size = h.bytes :=
  run_size h _

theorem leaf_size (x y : ByteArray) : (Hm96.Default512.leaf x y).size = 64 := sha512_size _

theorem depth2 : depth 2 = 1 := by decide

theorem taggedSize : TaggedSize := fun _ _ => sha512_size _

theorem frameSizes : FrameSizes := by
  intro h binding owner count d hd
  have hid : d.id.size = h.bytes ∧ d.hash = h := by
    unfold Domain.mk? at hd
    by_cases h32 : binding.size = 32 <;> by_cases h64 : binding.size = 64 <;> by_cases ho : owner < -2 <;>
      simp [h32, h64, ho, pure, Except.pure, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at hd <;>
      first | (cases hd; exact ⟨frameHash_size .., rfl⟩) | (cases hd) | skip
  obtain ⟨hid, hh⟩ := hid
  rcases d with ⟨dh, dc, di⟩
  dsimp only at hh
  subst hh
  exact ⟨hid, frameHash_size .., fun _ => frameHash_size .., fun _ _ _ => frameHash_size ..,
    fun _ _ _ _ => frameHash_size ..⟩

theorem coinNumbers : CoinNumbers :=
  ⟨rfl, rfl, depth2, fun _ => rfl, fun _ _ => leaf_size _ _⟩

theorem depthServed : DepthServed := by
  unfold DepthServed depth; decide +kernel

theorem domainRefusals : DomainRefusals := by
  intro h binding owner count
  have e : (-2 ≤ owner) ↔ ¬ owner < -2 := by omega
  unfold Domain.mk?
  by_cases h32 : binding.size = 32 <;> by_cases h64 : binding.size = 64 <;> by_cases ho : owner < -2 <;>
    simp [h32, h64, ho, e] <;> rfl

theorem leafRefusals : LeafRefusals := by
  intro d rank schema v
  have e : 192 ≤ v.size ↔ ¬ v.size < 192 := by omega
  rcases d with ⟨frame, hm, dg⟩
  unfold RowsDomain.leaf
  cases hm <;> cases dg <;> by_cases h1 : v.size < 192 <;> by_cases h2 : v.size = 256 <;>
    simp [h1, h2, HM96_SALT, e] <;> first | rfl | omega

theorem idx_labels (label : String) (rank : Nat) :
    List.idxOf? label ["salt", "draw"] = some rank ↔ (label = "salt" ∧ rank = 0) ∨ (label = "draw" ∧ rank = 1) := by
  by_cases hs : label = "salt"
  · subst hs
    have : List.idxOf? "salt" ["salt", "draw"] = some 0 := by decide
    rw [this]; simp [eq_comm]
  · by_cases hd : label = "draw"
    · subst hd
      have : List.idxOf? "draw" ["salt", "draw"] = some 1 := by decide
      rw [this]; simp [eq_comm]
    · have hs' : ¬"salt" = label := fun h => hs h.symm
      have hd' : ¬"draw" = label := fun h => hd h.symm
      simp [List.idxOf?, List.findIdx?, List.findIdx?.go, hs, hd, hs', hd']

theorem idx_none (label : String) : List.idxOf? label ["salt", "draw"] = none ↔ label ∉ ["salt", "draw"] := by
  by_cases hs : label = "salt"
  · subst hs; decide
  · by_cases hd : label = "draw"
    · subst hd; decide
    · have hs' : ¬"salt" = label := fun h => hs h.symm
      have hd' : ¬"draw" = label := fun h => hd h.symm
      simp [List.idxOf?, List.findIdx?, List.findIdx?.go, hs, hd, hs', hd']

theorem coinAccepts : CoinAccepts := by
  intro c o label after
  have hc : ∀ e, (coinDomain e).count = 2 := fun _ => rfl
  have hh : ∀ e, (coinDomain e).hash = .sha512 := fun _ => rfl
  unfold coinReason CoinOpens
  simp only [LABELS]
  rcases hl : List.idxOf? label ["salt", "draw"] with _ | rank
  · simp [throw, throwThe, MonadExceptOf.throw]
  · have hr : rank < 2 := by rcases (idx_labels label rank).1 hl with ⟨_, h⟩ | ⟨_, h⟩ <;> omega
    rcases o with ⟨epoch, olabel, oafter, coin, salt, path⟩
    rcases path with _ | ⟨sib, _ | ⟨s2, rest⟩⟩
    · simp [Domain.verify, hc, hh, depth2, checkPath, Hash.bytes]
      repeat' split
      all_goals simp [pure, Except.pure]
    · dsimp only
      simp only [Domain.verify, hc, hh, depth2, checkPath, Hash.bytes, coinRoot, leafValue, leaf_size, COIN_BYTES]
      by_cases h1 : epoch = c.epoch
      case neg => simp [h1, pure, Except.pure]
      by_cases h2 : olabel = label
      case neg => simp [h1, h2, pure, Except.pure]
      by_cases h3 : coin.size = 32
      case neg => simp [h1, h2, h3, pure, Except.pure]
      by_cases h4 : salt.size = 192
      case neg => simp [h1, h2, h3, h4, pure, Except.pure]
      have hr2 : ¬ 2 ≤ rank := by omega
      simp [h1, h2, h3, h4]
      by_cases h5 : sib.size = 64
      case neg =>
        by_cases h6 : sib.size = 32 <;>
          simp [h5, h6, hr2, pure, Except.pure, bind, Except.bind, throw, throwThe, MonadExceptOf.throw]
      by_cases h7 : c.root.size = 64
      case neg => simp [h5, h7, pure, Except.pure, bind, Except.bind]
      simp [h5, h7, hr2, pure, Except.pure, bind, Except.bind]
      by_cases ha : oafter = after <;> split <;> simp_all [eq_comm]
      all_goals (split <;> simp)
    · dsimp only
      simp only [Domain.verify, hc, hh, depth2, checkPath]
      by_cases h1 : epoch = c.epoch
      case neg => simp [h1, pure, Except.pure]
      by_cases h2 : olabel = label
      case neg => simp [h1, h2, pure, Except.pure]
      simp [h1, h2, pure, Except.pure, bind, Except.bind, throw, throwThe, MonadExceptOf.throw]
      repeat' split
      all_goals simp_all

theorem coinRaises : CoinRaises := by
  intro c o label after
  have hc : ∀ e, (coinDomain e).count = 2 := fun _ => rfl
  have hh : ∀ e, (coinDomain e).hash = .sha512 := fun _ => rfl
  unfold coinReason
  simp only [LABELS]
  rcases hl : List.idxOf? label ["salt", "draw"] with _ | rank
  · simp [(idx_none label).1 hl, throw, throwThe, MonadExceptOf.throw, Except.isOk, Except.toBool]
  · have hm : label ∈ ["salt", "draw"] := by
      rcases (idx_labels label rank).1 hl with ⟨h, _⟩ | ⟨h, _⟩ <;> simp [h]
    have hr2 : ¬ 2 ≤ rank := by rcases (idx_labels label rank).1 hl with ⟨_, h⟩ | ⟨_, h⟩ <;> omega
    rcases o with ⟨epoch, olabel, oafter, coin, salt, path⟩
    dsimp only
    simp only [Domain.verify, hc, hh, depth2, checkPath, Hash.bytes, leafValue, leaf_size, COIN_BYTES]
    by_cases h1 : epoch = c.epoch
    case neg => simp [h1, hm, pure, Except.pure, Except.isOk, Except.toBool]
    by_cases h2 : olabel = label
    case neg => simp [h1, h2, hm, pure, Except.pure, Except.isOk, Except.toBool]
    by_cases h3 : coin.size = 32
    case neg => simp [h1, h2, h3, hm, pure, Except.pure, Except.isOk, Except.toBool]
    by_cases h4 : salt.size = 192
    case neg => simp [h1, h2, h3, h4, hm, pure, Except.pure, Except.isOk, Except.toBool]
    simp [h1, h2, h3, h4, hm, hr2]
    by_cases hp : path.any (fun p => p.size != 32 && p.size != 64) = true
    · simp at hp
      simp [hp, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, Except.isOk, Except.toBool]
    · simp at hp
      have hn : ¬ ∃ x, x ∈ path ∧ ¬x.size = 32 ∧ ¬x.size = 64 := fun ⟨x, hx, a, b⟩ => b (hp x hx a)
      simp [hn, bind, Except.bind, pure, Except.pure, Except.isOk, Except.toBool]
      repeat' split
      all_goals (try simp_all)
      all_goals (repeat' split at *)
      all_goals simp_all

end Properties.Pouw.Commit
