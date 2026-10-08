import Properties.Pouw.Commit

/-!
# Proofs of PoUW's commitment properties (`Properties.Pouw.Commit`)

The sizes and numbers by unfolding and evaluation in the kernel, the refusals and the accepted coins by unfolding the
port's definitions case by case, and the draws' shape from `Flock.Draw.uniform`'s range and the tickets' running sums.
The binding statements from the frame's injectivity (its lengths are big-endian, so a frame parses back to its tag and
parts) and, down an accepted coin's path, one step per hash: each pair of hashed messages is equal, or it is the collision.
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

/-! ## Salts, keys and draws -/

open FlockSoundness.Randomness in
theorem derive_size (source : ByteArray) (domain : String) (context : List (String × Part)) :
    (derive source domain context).size = 32 := by
  unfold derive
  rw [ByteArray.size_extract, sha512_size]
  rfl

theorem keyRefusals : KeyRefusals := by
  intro source domain context
  unfold key?
  by_cases h : source.size = 0
  · rw [ite_eq_left_of_eq_true _ _ (eq_true h)]; simp only [h, ne_eq, not_true_eq_false, decide_false]; rfl
  · rw [ite_eq_right_of_eq_false _ _ (eq_false h)]; simp only [h, ne_eq, not_false_eq_true, decide_true]; rfl

theorem keySize : KeySize := by
  intro source domain context k h
  unfold key? at h
  split at h
  · cases h
  · rw [← Except.ok.inj h]; exact derive_size ..

open FlockSoundness.Randomness in
theorem blocks_size (key : ByteArray) (index : List Part) (l : List Nat) (acc : ByteArray) :
    (l.foldl (fun acc c => acc ++ block key index c) acc).size = acc.size + 64 * l.length := by
  induction l generalizing acc with
  | nil => simp
  | cons c l ih =>
    rw [List.foldl_cons, ih, ByteArray.size_append, block, sha512_size]
    simp only [List.length_cons]
    omega

open FlockSoundness.Randomness in
/-- A read of `n` bytes of a stream gives `n` bytes. -/
theorem stream_size (key : ByteArray) (index : List Part) (n : Nat) : (stream key index n).size = n := by
  simp only [stream, ByteArray.size_extract, blocks_size, List.length_range, ByteArray.size_empty]
  omega

theorem saltSize : SaltSize := by
  refine ⟨fun c coin wr s h => ?_, fun seed wr s h => ?_⟩
  · unfold saltOf at h
    simp only [bind, Except.bind, pure, Except.pure] at h
    split at h
    · cases h
    · rw [← Except.ok.inj h]; exact stream_size ..
  · unfold unverifiedSalt at h
    simp only [bind, Except.bind, pure, Except.pure] at h
    split at h
    · cases h
    · rw [← Except.ok.inj h]; exact stream_size ..

theorem saltRefusals : SaltRefusals := by
  refine ⟨fun c coin wr => ?_, fun seed wr => ?_⟩
  · unfold saltOf key?
    by_cases h : coin.size = 0
    · rw [ite_eq_left_of_eq_true _ _ (eq_true h)]; simp only [h, ne_eq, not_true_eq_false, decide_false]; rfl
    · rw [ite_eq_right_of_eq_false _ _ (eq_false h)]; simp only [h, ne_eq, not_false_eq_true, decide_true]; rfl
  · unfold unverifiedSalt key?
    by_cases h : seed.size = 0
    · rw [ite_eq_left_of_eq_true _ _ (eq_true h)]; simp only [h, ne_eq, not_true_eq_false, decide_false]; rfl
    · rw [ite_eq_right_of_eq_false _ _ (eq_false h)]; simp only [h, ne_eq, not_false_eq_true, decide_true]; rfl

theorem epochSaltAccepts : EpochSaltAccepts := by
  intro c o wr s
  rw [← coinAccepts c o "salt" wr]
  unfold epochSalt
  cases h : coinReason c o "salt" wr with
  | error e => simp [bind, Except.bind]
  | ok r =>
    cases r with
    | some r => simp [bind, Except.bind, pure, Except.pure]
    | none =>
      cases hs : saltOf c o.coin wr <;> simp [bind, Except.bind, pure, Except.pure]

theorem drawCoinAccepts : DrawCoinAccepts := by
  refine ⟨fun c o after coin => ?_, fun o after coin => ?_⟩
  · rw [← coinAccepts c o "draw" after]
    unfold drawCoin
    cases h : coinReason c o "draw" after with
    | error e => simp [h, bind, Except.bind]
    | ok r => cases r <;> simp [h, bind, Except.bind, pure, Except.pure, eq_comm]
  · simp [drawCoin, pure, Except.pure]

theorem mapM_shape {α β : Type} (f : α → Except String β) :
    ∀ (l : List α) (ys : List β), l.mapM f = .ok ys → ys.length = l.length ∧ ∀ y ∈ ys, ∃ x ∈ l, f x = .ok y := by
  intro l
  induction l with
  | nil => intro ys h; simp [pure, Except.pure] at h; subst h; simp
  | cons a l ih =>
    intro ys h
    simp only [List.mapM_cons, bind, Except.bind] at h
    cases ha : f a with
    | error e => simp [ha] at h
    | ok b =>
      cases hl : l.mapM f with
      | error e => simp [ha, hl] at h
      | ok bs =>
        simp [ha, hl, pure, Except.pure] at h
        subst h
        obtain ⟨hlen, hmem⟩ := ih bs hl
        refine ⟨by simp [hlen], ?_⟩
        intro y hy
        simp at hy
        rcases hy with rfl | hy
        · exact ⟨a, by simp, ha⟩
        · obtain ⟨x, hx, hfx⟩ := hmem y hy
          exact ⟨x, by simp [hx], hfx⟩

theorem draw_uniform_lt (n : Nat) (hn : 0 < n) (src : ByteArray) :
    ∀ fuel pos r, Draw.uniform n src fuel pos = some r → r.1 < n := by
  intro fuel
  induction fuel with
  | zero => intro pos r h; simp [Draw.uniform] at h
  | succ fuel ih =>
    intro pos r h
    simp only [Draw.uniform] at h
    split at h
    · cases h
    · split at h
      · cases h; exact Nat.mod_lt _ hn
      · exact ih _ _ h

theorem sums_shape (tickets : List Nat) (acc : Array Nat) (s : Nat) :
    let r := tickets.foldl (fun (acc : Array Nat × Nat) w => (acc.1.push (acc.2 + w), acc.2 + w)) (acc, s)
    r.1.size = acc.size + tickets.length ∧ (tickets ≠ [] → r.1.toList.getLast? = some r.2) := by
  induction tickets generalizing acc s with
  | nil => simp
  | cons w ts ih =>
    simp only [List.foldl_cons]
    obtain ⟨h1, h2⟩ := ih (acc.push (s + w)) (s + w)
    refine ⟨by simp [h1]; omega, fun _ => ?_⟩
    by_cases hts : ts = []
    · subst hts; simp
    · exact h2 hts

theorem takeWhile_length_lt (p : Nat → Bool) :
    ∀ (l : List Nat) (x : Nat), x ∈ l → p x = false → (l.takeWhile p).length < l.length
  | [], _, h, _ => by simp at h
  | a :: l, x, hx, hp => by
    cases ha : p a
    · simp [ha]
    · have hl : x ∈ l := by
        rcases List.mem_cons.1 hx with rfl | h
        · simp [ha] at hp
        · exact h
      have := takeWhile_length_lt p l x hl hp
      simp [ha]
      omega

theorem bisectRight_lt (l : List Nat) (x : Nat) (last : Nat) (h : l.getLast? = some last) (hx : x < last) :
    bisectRight l x < l.length :=
  takeWhile_length_lt _ l last (List.mem_of_getLast? h) (by simp; omega)

theorem drawShape : DrawShape := by
  intro key tickets t ids h
  unfold weightedDraw at h
  by_cases ht : t = 0
  · subst ht; simp [pure, Except.pure] at h; subst h; simp
  simp only [ht, ite_false] at h
  obtain ⟨hsize, hlast⟩ := sums_shape tickets #[] 0
  generalize hr : tickets.foldl _ (#[], 0) = r at hsize hlast h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  · rename_i total hne hl
    simp only [bind, Except.bind, pure, Except.pure] at h
    have hpos : 0 < total := Nat.pos_of_ne_zero hne
    obtain ⟨hlen, hmem⟩ := mapM_shape _ _ _ h
    refine ⟨by simpa using hlen, fun i hi => ?_⟩
    obtain ⟨j, _, hj⟩ := hmem i hi
    split at hj
    · rename_i rr hu
      cases hj
      have hrr : rr < total := by
        simp only [FlockSoundness.Randomness.uniform, FlockSoundness.Randomness.uniformOn, Option.map_eq_some_iff] at hu
        obtain ⟨p, hp, rfl⟩ := hu
        exact draw_uniform_lt total hpos _ _ _ _ hp
      have := bisectRight_lt _ rr total hl hrr
      simp at hsize
      simpa [hsize] using this
    · cases hj

/-! ## Binding -/

theorem be8_data (x : Nat) : (be8 x).data.toList =
    [(x >>> 56).toUInt8, (x >>> 48).toUInt8, (x >>> 40).toUInt8, (x >>> 32).toUInt8,
     (x >>> 24).toUInt8, (x >>> 16).toUInt8, (x >>> 8).toUInt8, (x >>> 0).toUInt8] := by
  simp [be8, List.range_succ]

theorem be32_data (x : UInt32) : (be32 x).data.toList =
    [(x >>> 24).toUInt8, (x >>> 16).toUInt8, (x >>> 8).toUInt8, (x >>> 0).toUInt8] := by
  simp only [be32, Id.run]
  simp [List.range', List.foldl]
  rfl

theorem be8_inj (x y : Nat) (hx : x < 2 ^ 64) (hy : y < 2 ^ 64)
    (h : (be8 x).data.toList = (be8 y).data.toList) : x = y := by
  simp only [be8_data, List.cons.injEq] at h
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, -⟩ := h
  simp only [← UInt8.toNat_inj] at h0 h1 h2 h3 h4 h5 h6 h7
  simp [Nat.shiftRight_eq_div_pow] at h0 h1 h2 h3 h4 h5 h6 h7
  omega

theorem be32_inj (x y : Nat) (hx : x < 2 ^ 32) (hy : y < 2 ^ 32)
    (h : (be32 x.toUInt32).data.toList = (be32 y.toUInt32).data.toList) : x = y := by
  simp only [be32_data, List.cons.injEq] at h
  obtain ⟨h0, h1, h2, h3, -⟩ := h
  simp only [← UInt8.toNat_inj] at h0 h1 h2 h3
  simp [Nat.shiftRight_eq_div_pow] at h0 h1 h2 h3
  omega

/-- The framed parts, as bytes. -/
def framed (parts : List ByteArray) : List UInt8 :=
  parts.flatMap fun p => (be8 p.size).data.toList ++ p.data.toList

theorem foldl_frame (parts : List ByteArray) (acc : ByteArray) :
    (parts.foldl (fun acc p => acc ++ be8 p.size ++ p) acc).data.toList = acc.data.toList ++ framed parts := by
  induction parts generalizing acc with
  | nil => simp [framed]
  | cons p ps ih => simp [ih, framed, List.append_assoc]

theorem frameBytes_data (tag : String) (parts : List ByteArray) :
    (frameBytes tag parts).data.toList = FRAME.data.toList ++ (be32 tag.toUTF8.size.toUInt32).data.toList ++
      tag.toUTF8.data.toList ++ framed parts := by
  simp [frameBytes, foldl_frame, List.append_assoc]

theorem be8_length (x : Nat) : (be8 x).data.toList.length = 8 := by simp [be8_data]

theorem be32_length (x : UInt32) : (be32 x).data.toList.length = 4 := by simp [be32_data]

theorem framed_inj : ∀ (ps ps' : List ByteArray), (∀ p ∈ ps, p.size < 2 ^ 64) → (∀ p ∈ ps', p.size < 2 ^ 64) →
    framed ps = framed ps' → ps = ps'
  | [], [], _, _, _ => rfl
  | [], p' :: ps', _, _, h => by
    have := congrArg List.length h
    simp only [framed, List.flatMap_nil, List.flatMap_cons, List.length_append, be8_length, List.length_nil] at this
    omega
  | p :: ps, [], _, _, h => by
    have := congrArg List.length h
    simp only [framed, List.flatMap_nil, List.flatMap_cons, List.length_append, be8_length, List.length_nil] at this
    omega
  | p :: ps, p' :: ps', hb, hb', h => by
    simp only [framed, List.flatMap_cons, List.append_assoc] at h
    obtain ⟨h8, h⟩ := List.append_inj h (by rw [be8_length, be8_length])
    have hs := be8_inj _ _ (hb p (by simp)) (hb' p' (by simp)) h8
    obtain ⟨hp, h⟩ := List.append_inj h (by simp [hs])
    have hp : p = p' := ByteArray.ext (Array.toList_inj.1 hp)
    rw [hp, framed_inj ps ps' (fun q hq => hb q (by simp [hq])) (fun q hq => hb' q (by simp [hq])) h]

/-- A frame-v3 message determines its tag and parts. -/
theorem frameBytes_inj (tag tag' : String) (parts parts' : List ByteArray) (ht : tag.toUTF8.size < 2 ^ 32)
    (ht' : tag'.toUTF8.size < 2 ^ 32) (hb : ∀ p ∈ parts, p.size < 2 ^ 64) (hb' : ∀ p ∈ parts', p.size < 2 ^ 64)
    (h : frameBytes tag parts = frameBytes tag' parts') : tag = tag' ∧ parts = parts' := by
  have h := congrArg (fun b => b.data.toList) h
  simp only [frameBytes_data, List.append_assoc] at h
  obtain ⟨-, h⟩ := List.append_inj h rfl
  obtain ⟨h4, h⟩ := List.append_inj h (by rw [be32_length, be32_length])
  have hs := be32_inj _ _ ht ht' h4
  obtain ⟨htag, h⟩ := List.append_inj h (by simp only [Array.length_toList, ByteArray.size_data]; exact hs)
  have htag : tag = tag' := String.toByteArray_inj.1 (ByteArray.ext (Array.toList_inj.1 htag))
  exact ⟨htag, framed_inj parts parts' hb hb' h⟩

theorem frameInjective : FrameInjective := by
  intro h tag tag' parts parts' ht ht' hb hb' hf
  by_cases he : frameBytes tag parts = frameBytes tag' parts'
  · exact .inl (frameBytes_inj tag tag' parts parts' ht ht' hb hb' he)
  · exact .inr ⟨he, hf⟩

/-- Two lists of messages hash two different messages, one from each, to one SHA-512 digest. -/
def Break (L L' : List ByteArray) : Prop := ∃ x ∈ L, ∃ y ∈ L', x ≠ y ∧ Sha512.hash x = Sha512.hash y

theorem step {L L' : List ByteArray} {x y : ByteArray} (hx : x ∈ L) (hy : y ∈ L')
    (h : Sha512.hash x = Sha512.hash y) : x = y ∨ Break L L' := by
  by_cases e : x = y
  · exact .inl e
  · exact .inr ⟨x, hx, y, hy, e, h⟩

theorem framed_sizes : ∀ (ps ps' : List ByteArray), ps.map ByteArray.size = ps'.map ByteArray.size →
    framed ps = framed ps' → ps = ps'
  | [], [], _, _ => rfl
  | [], _ :: _, hs, _ => by simp at hs
  | _ :: _, [], hs, _ => by simp at hs
  | p :: ps, p' :: ps', hs, h => by
    simp only [List.map_cons, List.cons.injEq] at hs
    simp only [framed, List.flatMap_cons, List.append_assoc, hs.1] at h
    obtain ⟨-, h⟩ := List.append_inj h rfl
    obtain ⟨hp, h⟩ := List.append_inj h (by simp [hs.1])
    rw [ByteArray.ext (Array.toList_inj.1 hp), framed_sizes ps ps' hs.2 h]

/-- One tag, and parts of one size each: the frame determines the parts, with no bound on the sizes. -/
theorem frameBytes_sizes (tag : String) (parts parts' : List ByteArray)
    (hs : parts.map ByteArray.size = parts'.map ByteArray.size) (h : frameBytes tag parts = frameBytes tag parts') :
    parts = parts' := by
  have h := congrArg (fun b => b.data.toList) h
  simp only [frameBytes_data, List.append_assoc] at h
  exact framed_sizes parts parts' hs (List.append_cancel_left (List.append_cancel_left (List.append_cancel_left h)))

theorem commitString_size (x salt : ByteArray) :
    (ByteArray.mk ((Array.range 64).map fun i => x.get! i ^^^ (Hm96.Default512.mask salt).get! i)).size = 64 := by
  simp only [ByteArray.size, Array.size_map, Array.size_range]

theorem xor_cancel (a b m : UInt8) (h : a ^^^ m = b ^^^ m) : a = b := by
  have := congrArg (· ^^^ m) h
  simpa [UInt8.xor_assoc] using this

/-- Two 64-byte strings that agree under one mask agree. -/
theorem masked_inj (x x' m : ByteArray) (hx : x.size = 64) (hx' : x'.size = 64)
    (h : ByteArray.mk ((Array.range 64).map fun i => x.get! i ^^^ m.get! i) =
      ByteArray.mk ((Array.range 64).map fun i => x'.get! i ^^^ m.get! i)) : x = x' := by
  rcases x with ⟨xs⟩
  rcases x' with ⟨xs'⟩
  have hx : xs.size = 64 := hx
  have hx' : xs'.size = 64 := hx'
  congr 1
  apply Array.ext (hx.trans hx'.symm)
  intro i hi hi'
  have hi64 : i < 64 := hx ▸ hi
  have e := (Array.ext_iff.1 (congrArg ByteArray.data h)).2 i (by simp [hi64]) (by simp [hi64])
  simp only [Array.getElem_map, Array.getElem_range] at e
  have e := xor_cancel _ _ _ e
  simpa [ByteArray.get!, getElem!_pos, hi, hi'] using e

/-- An hm96 leaf binds its coin: equal leaf values give the same coin, or a collision among the coin-tag, salt-digest and
leaf inputs. -/
theorem leafValue_binding (L L' : List ByteArray) (coin salt coin' salt' : ByteArray)
    (h1 : COIN_TAG ++ coin ∈ L) (h1' : COIN_TAG ++ coin' ∈ L')
    (h2 : Hm96.Default512.saltPrefix ++ salt ∈ L) (h2' : Hm96.Default512.saltPrefix ++ salt' ∈ L')
    (h3 : Hm96.Default512.leafPrefix ++ Hm96.Default512.commitString (Sha512.hash (COIN_TAG ++ coin)) salt ∈ L)
    (h3' : Hm96.Default512.leafPrefix ++ Hm96.Default512.commitString (Sha512.hash (COIN_TAG ++ coin')) salt' ∈ L')
    (hv : leafValue coin salt = leafValue coin' salt') : coin = coin' ∨ Break L L' := by
  rcases step h3 h3' hv with e | b
  case inr => exact .inr b
  have e := (ByteArray.append_right_inj _).1 e
  simp only [Hm96.Default512.commitString] at e
  have hb := ByteArray.append_inj_left e (by rw [commitString_size, commitString_size])
  rw [hb, ByteArray.append_right_inj] at e
  rcases step h2 h2' e with e | b
  case inr => exact .inr b
  have hs := (ByteArray.append_right_inj _).1 e
  rw [hs] at hb
  have hx := masked_inj _ _ _ (sha512_size _) (sha512_size _) hb
  rcases step h1 h1' hx with e | b
  case inr => exact .inr b
  exact .inl ((ByteArray.append_right_inj _).1 e)

theorem pathFrames_one (d : Domain) (rank : Nat) (leaf sib : ByteArray) :
    pathFrames d rank leaf [sib] =
      [if rank % 2 == 0 then frameBytes "node" [d.id, natBytes 0, natBytes (rank / 2), leaf, sib]
        else frameBytes "node" [d.id, natBytes 0, natBytes (rank / 2), sib, leaf]] := rfl

theorem coinBinding : CoinBinding := by
  intro c o o' label after after' h h'
  obtain ⟨rank, sib, hl, -, -, -, -, hp, hsib, -, hroot, -⟩ := (coinAccepts c o label after).1 h
  obtain ⟨rank', sib', hl', -, -, -, -, hp', hsib', -, hroot', -⟩ := (coinAccepts c o' label after').1 h'
  rw [hl] at hl'
  cases hl'
  have hidx : LABELS.idxOf label = rank := by
    rcases (idx_labels label rank).1 hl with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> decide
  rw [hidx]
  have hd : (coinDomain c.epoch).hash = .sha512 := rfl
  have hlf : ∀ v, ((coinDomain c.epoch).leaf rank SCHEME v).size = 64 := fun _ => frameHash_size ..
  have mem : ∀ (o : CoinOpening) x, x ∈ [COIN_TAG ++ o.coin, Hm96.Default512.saltPrefix ++ o.salt,
      Hm96.Default512.leafPrefix ++ Hm96.Default512.commitString (Sha512.hash (COIN_TAG ++ o.coin)) o.salt,
      frameBytes "leaf" [(coinDomain c.epoch).id, natBytes rank, natBytes rank, SCHEME.toUTF8,
        leafValue o.coin o.salt]] → x ∈ coinHashed c.epoch rank o :=
    fun _ _ hx => List.mem_append_left _ hx
  have memN : ∀ (o : CoinOpening) sib x, o.path = [sib] →
      x ∈ pathFrames (coinDomain c.epoch) rank ((coinDomain c.epoch).leaf rank SCHEME (leafValue o.coin o.salt)) [sib] →
      x ∈ coinHashed c.epoch rank o := by
    intro o sib x hp hx
    unfold coinHashed
    rw [hp]
    exact List.mem_append_right _ hx
  have leaves : (coinDomain c.epoch).leaf rank SCHEME (leafValue o.coin o.salt) =
      (coinDomain c.epoch).leaf rank SCHEME (leafValue o'.coin o'.salt) →
      o.coin = o'.coin ∨ Break (coinHashed c.epoch rank o) (coinHashed c.epoch rank o') := by
    intro e
    rcases step (mem o _ (.tail _ (.tail _ (.tail _ (.head _)))))
      (mem o' _ (.tail _ (.tail _ (.tail _ (.head _))))) e with e | b
    case inr => exact .inr b
    have e := frameBytes_sizes _ _ _ (by simp only [List.map_cons, List.map_nil, leafValue, leaf_size]) e
    simp only [List.cons.injEq, and_true, true_and] at e
    exact leafValue_binding _ _ _ _ _ _ (mem o _ (.head _)) (mem o' _ (.head _)) (mem o _ (.tail _ (.head _)))
      (mem o' _ (.tail _ (.head _))) (mem o _ (.tail _ (.tail _ (.head _)))) (mem o' _ (.tail _ (.tail _ (.head _)))) e
  have node : ∀ x y, x ∈ coinHashed c.epoch rank o → y ∈ coinHashed c.epoch rank o' →
      Sha512.hash x = c.root → Sha512.hash y = c.root →
        x = y ∨ Break (coinHashed c.epoch rank o) (coinHashed c.epoch rank o') :=
    fun x y hx hy e e' => step hx hy (e.trans e'.symm)
  by_cases hr : rank % 2 = 0
  · simp only [coinRoot, hr, ite_true, Domain.node, frameHash, hd, Hash.run] at hroot hroot'
    rcases node _ _ (memN o sib _ hp (by simp only [pathFrames_one, hr]; exact List.mem_singleton_self _))
      (memN o' sib' _ hp' (by simp only [pathFrames_one, hr]; exact List.mem_singleton_self _))
      hroot.symm hroot'.symm with e | b
    case inr => exact .inr b
    have e := frameBytes_sizes _ _ _ (by simp only [List.map_cons, List.map_nil, hlf, hsib, hsib']) e
    simp only [List.cons.injEq, and_true, true_and] at e
    exact leaves e.1
  · simp only [coinRoot, hr, ite_false, Domain.node, frameHash, hd, Hash.run] at hroot hroot'
    rcases node _ _
      (memN o sib _ hp (by simp only [pathFrames_one, hr, beq_iff_eq, ite_false]; exact List.mem_singleton_self _))
      (memN o' sib' _ hp' (by simp only [pathFrames_one, hr, beq_iff_eq, ite_false]; exact List.mem_singleton_self _))
      hroot.symm hroot'.symm with e | b
    case inr => exact .inr b
    have e := frameBytes_sizes _ _ _ (by simp only [List.map_cons, List.map_nil, hlf, hsib, hsib']) e
    simp only [List.cons.injEq, and_true, true_and] at e
    exact leaves e.2

end Properties.Pouw.Commit
