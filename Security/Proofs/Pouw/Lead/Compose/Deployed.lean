import Definitions.Pouw.Lead.Deployed
import Proofs.Pouw.Lead.Compose.Honest

/-!
# The lead's honest lemma at the deployed instance

At the deployed commitment (`l1wCommit`) and form classifier (`l1wIsA`), `CommitL1w` holds (`commitL1w_l1w`) when the
call's rows and columns are whole MMA tiles and the forward's words leaves and A's segments number at most `2^64`
(`range-u64` domains refuse more): each of the call's words chunks is its forward's words leaf at its own rank, after
the forward's earlier calls' (`value_call`), and each segment of A is A's leaf at its own rank, so
`commitL1w_of_leaves` applies over the words leaves up to the call's last. A words leaf is no message of A's instance
(`leafMsg_not_mem_aMsgs`): A's leaves are under a schema of another length (`leafMsg_schema_ne`), and its pads, nodes,
empty root and hm96 messages differ from a leaf in a header byte. The stream's tiles are the call's
`S · rowBlocks · colBlocks` (`wordsTiles_length`), so the call's chunks are `wordChunks` and cover its words' bytes
(`sum_chunks`), and A's segments its `rows · K` bytes.

`honestL1w` is the honest lemma at the deployed instance: `honestOfCommit` at `l1wCommit`, `l1wIsA` and the certifier's
stream `certStream`, at every admitted call whose A is at most `2^64` bytes, in every forward whose words leaves number
at most `2^64`. `honest` is the guarantee, `honestL1w` beside `certifierCoins`.
-/

namespace Pouw.SecurityProofs.Lead.Compose

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100

/-! ## A whole tree's messages -/

/-- A leaf at a rank below the count is one of the tree's messages. -/
theorem leaf_mem_msgs (H : Hash.Msg → Hash.Digest) (D : Hash.Tree.Domain) (s : String) (v : ℕ → Hash.Msg) {r : ℕ}
    (hr : r < D.count) : Hash.Tree.leafMsg D r s (v r) ∈ Hash.Tree.msgs H D s v := by
  unfold Hash.Tree.msgs
  split_ifs
  · omega
  · exact List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨r, List.mem_range.2 hr, rfl⟩))

/-- A level's messages are nodes. -/
theorem mem_pairs {D : Hash.Tree.Domain} {lv : ℕ} :
    ∀ {i : ℕ} {xs : List Hash.Digest} {y : Hash.Msg}, y ∈ Hash.Tree.pairs D lv i xs →
      ∃ j l r, y = Hash.Tree.nodeMsg D lv j l r
  | i, l :: r :: xs, y, h => by
    simp only [Hash.Tree.pairs, List.mem_cons] at h
    rcases h with h | h
    · exact ⟨i, l, r, h⟩
    · exact mem_pairs h
  | _, [], _, h => by simp [Hash.Tree.pairs] at h
  | _, [_], _, h => by simp [Hash.Tree.pairs] at h

/-- The levels' messages are nodes. -/
theorem mem_build {H : Hash.Msg → Hash.Digest} {D : Hash.Tree.Domain} :
    ∀ {lv d : ℕ} {xs : List Hash.Digest} {y : Hash.Msg}, y ∈ (Hash.Tree.build H D lv d xs).1 →
      ∃ lv j l r, y = Hash.Tree.nodeMsg D lv j l r
  | _, 0, _, _, h => by simp [Hash.Tree.build] at h
  | lv, _ + 1, _, _, h => by
    simp only [Hash.Tree.build, List.mem_append] at h
    rcases h with h | h
    · obtain ⟨j, l, r, rfl⟩ := mem_pairs h
      exact ⟨lv, j, l, r, rfl⟩
    · exact mem_build h

/-- A tree's messages are its empty root's, its leaves' at ranks below the count, its pads' and its nodes'. -/
theorem mem_msgs {H : Hash.Msg → Hash.Digest} {D : Hash.Tree.Domain} {s : String} {v : ℕ → Hash.Msg} {y : Hash.Msg}
    (h : y ∈ Hash.Tree.msgs H D s v) :
    y = Hash.Tree.emptyMsg D ∨ (∃ r < D.count, y = Hash.Tree.leafMsg D r s (v r)) ∨
      (∃ r, y = Hash.Tree.padMsg D r) ∨ ∃ lv j l r, y = Hash.Tree.nodeMsg D lv j l r := by
  unfold Hash.Tree.msgs at h
  split_ifs at h
  · exact Or.inl (List.mem_singleton.1 h)
  · simp only [Hash.Tree.leaves, List.mem_append, List.mem_map] at h
    rcases h with (⟨r, hr, rfl⟩ | ⟨r, -, rfl⟩) | h
    · exact Or.inr (Or.inl ⟨r, List.mem_range.1 hr, rfl⟩)
    · exact Or.inr (Or.inr (Or.inl ⟨r, rfl⟩))
    · exact Or.inr (Or.inr (Or.inr (mem_build h)))

/-! ## A leaf is no other message -/

/-- A frame's header: the frame string, its tag's length and the tag. -/
def frameHead (t : String) : Hash.Msg := Hash.Tree.FRAME ++ Hash.be 4 (Hash.utf8 t).length ++ Hash.utf8 t

/-- A frame's bytes begin with its header's. -/
theorem frame_getElem? (t : String) (ps : List Hash.Msg) {k : ℕ} (hk : k < (frameHead t).length) :
    (Hash.Tree.frame t ps)[k]? = (frameHead t)[k]? :=
  List.getElem?_append_left hk

theorem leafMsg_ne_padMsg {D D' : Hash.Tree.Domain} {r r' : ℕ} {s : String} {v : Hash.Msg} :
    Hash.Tree.leafMsg D r s v ≠ Hash.Tree.padMsg D' r' := fun h => by
  have h := congrArg (·[36]?) h
  simp only [Hash.Tree.leafMsg, Hash.Tree.padMsg] at h
  rw [frame_getElem? _ _ (by decide), frame_getElem? _ _ (by decide)] at h
  exact absurd h (by decide)

theorem leafMsg_ne_emptyMsg {D D' : Hash.Tree.Domain} {r : ℕ} {s : String} {v : Hash.Msg} :
    Hash.Tree.leafMsg D r s v ≠ Hash.Tree.emptyMsg D' := fun h => by
  have h := congrArg (·[36]?) h
  simp only [Hash.Tree.leafMsg, Hash.Tree.emptyMsg] at h
  rw [frame_getElem? _ _ (by decide), frame_getElem? _ _ (by decide)] at h
  exact absurd h (by decide)

theorem leafMsg_ne_nodeMsg {D D' : Hash.Tree.Domain} {r lv i : ℕ} {s : String} {v : Hash.Msg} {l r' : Hash.Digest} :
    Hash.Tree.leafMsg D r s v ≠ Hash.Tree.nodeMsg D' lv i l r' := fun h => by
  have h := congrArg (·[37]?) h
  simp only [Hash.Tree.leafMsg, Hash.Tree.nodeMsg] at h
  rw [frame_getElem? _ _ (by decide), frame_getElem? _ _ (by decide)] at h
  exact absurd h (by decide)

/-- A frame leaf is neither of an hm96 instance's messages: they begin `verity/`, a frame `veritor/`. -/
theorem leafMsg_not_mem_hm96 {D : Hash.Tree.Domain} {r : ℕ} {s : String} {v : Hash.Msg}
    (H : Hash.Msg → Hash.Digest) (key : Hash.Msg) (x : Hash.Digest) (y : Hash.Msg) :
    Hash.Tree.leafMsg D r s v ∉ Hash.Hm96.hashed H key x y := by
  have hl : (Hash.Tree.leafMsg D r s v)[5]? = some 111 := by
    rw [Hash.Tree.leafMsg, frame_getElem? _ _ (by decide)]
    decide
  have hs : (Hash.Hm96.saltPrefix ++ y)[5]? = some 121 := by
    have h27 : (Hash.cstr "verity/hm96-sha512/salt/v1").length = 27 := by decide
    rw [List.getElem?_append_left (by rw [Hash.Hm96.saltPrefix, Hash.BindingProofs.padTo_length]; omega),
      Hash.Hm96.saltPrefix, Hash.padTo, List.getElem?_append_left (by rw [h27]; omega)]
    decide
  have hp : (Hash.Hm96.leafPrefix H key ++ Hash.Hm96.commitString H key x y)[5]? = some 121 := by
    have h27 : (Hash.cstr "verity/hm96-sha512/leaf/v1").length = 27 := by decide
    have hm : 5 < (Hash.cstr "verity/hm96-sha512/leaf/v1" ++ (H key).bytes).length := by
      rw [List.length_append, h27]
      omega
    rw [List.getElem?_append_left (by rw [Hash.Hm96.leafPrefix, Hash.BindingProofs.padTo_length]; omega),
      Hash.Hm96.leafPrefix, Hash.padTo, List.getElem?_append_left hm, List.getElem?_append_left (by rw [h27]; omega)]
    decide
  intro h
  simp only [Hash.Hm96.hashed, List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with h | h
  · rw [h, hs] at hl
    exact absurd hl (by decide)
  · rw [h, hp] at hl
    exact absurd hl (by decide)

/-- A `uint` part below `2^64` determines what follows it. -/
theorem uint_part_cancel {a b : ℕ} {X Y : Hash.Msg} (ha : a < 2 ^ 64) (hb : b < 2 ^ 64)
    (h : Hash.be 8 (Hash.uint a).length ++ (Hash.uint a ++ X) = Hash.be 8 (Hash.uint b).length ++ (Hash.uint b ++ Y)) :
    X = Y := by
  obtain ⟨hlen, h⟩ := List.append_inj h (by simp)
  exact (List.append_inj h (Hash.BindingProofs.be_inj_lt ((uint_length_le ha).trans_lt (by norm_num))
    ((uint_length_le hb).trans_lt (by norm_num)) hlen)).2

/-- **Frame leaves under schemas of different lengths are distinct**, at ranks and positions below `2^64`, whatever
their domains and values. -/
theorem leafMsg_schema_ne {D D' : Hash.Tree.Domain} {r r' : ℕ} {s s' : String} {v v' : Hash.Msg} (hr : r < 2 ^ 64)
    (hr' : r' < 2 ^ 64) (hp : D.position r < 2 ^ 64) (hp' : D'.position r' < 2 ^ 64)
    (hs : Hash.be 8 (Hash.utf8 s).length ≠ Hash.be 8 (Hash.utf8 s').length) :
    Hash.Tree.leafMsg D r s v ≠ Hash.Tree.leafMsg D' r' s' v' := fun h => by
  have h := Hash.BindingProofs.frame_parts h
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.append_assoc, Hash.BindingProofs.bytes_length,
    List.append_cancel_left_eq] at h
  exact hs (List.append_inj (uint_part_cancel hp hp' (uint_part_cancel hr hr' (List.append_inj h (by simp)).2))
    (by simp)).1

/-! ## `CommitL1w` at the deployed commitment -/

/-- **A words leaf is no message of A's instance**: A's leaves are under another schema at ranks below `2^64`, and its
pads, nodes, empty root and hm96 messages differ from a leaf in a header byte. -/
theorem leafMsg_not_mem_aMsgs (L : Pouw.Lead.Layout) (P : L1w) (F : CallForward) (H H₀ : Hash.Msg → Hash.Digest)
    (A : Fin L.rows → Fin L.K → UInt8) (hka : aLeaves L ≤ 2 ^ 64) {r : ℕ} (hr : r < 2 ^ 64) (v : Hash.Msg) :
    Hash.Tree.leafMsg (P.wordsDom H F L) r wordsSchema v ∉ (l1wCommit L P F H₀ A).aMsgs := by
  intro h
  simp only [l1wCommit, List.mem_append] at h
  rcases h with h | h
  · rcases mem_msgs h with h | ⟨r', hr', h⟩ | ⟨_, h⟩ | ⟨_, _, _, _, h⟩
    · exact leafMsg_ne_emptyMsg h
    · exact leafMsg_schema_ne hr (hr'.trans_le hka) hr (hr'.trans_le hka) (by decide) h
    · exact leafMsg_ne_padMsg h
    · exact leafMsg_ne_nodeMsg h
  · exact leafMsg_not_mem_hm96 _ _ _ _ h

theorem wordBytes_length (x : ℤ) : (wordBytes x).length = 4 := by
  simp [wordBytes, Hash.BindingProofs.be_length]

theorem tileFrag_length (L : Pouw.Lead.Layout) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) (x : ℕ × ℕ × ℕ) :
    (tileFrag L w x).length = tileWords := by
  obtain ⟨rb, cb, t⟩ := x
  simp [tileFrag]

theorem wordsBytes_length (L : Pouw.Lead.Layout) (nb : ℕ) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) :
    (wordsBytes L nb w).length = 4 * (tileWords * (wordsTiles L nb).length) := by
  simp only [wordsBytes, List.length_flatMap, wordBytes_length, tileFrag_length, List.map_const', List.sum_replicate,
    smul_eq_mul]
  ring

/-- Chunks of `b` from the start cover `min |xs| (b k)` of a list's elements. -/
theorem sum_chunks {α : Type} (xs : List α) (b : ℕ) :
    ∀ k, ((List.range k).map fun r => ((xs.drop (b * r)).take b).length).sum = min xs.length (b * k)
  | 0 => by simp
  | k + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_chunks xs b k]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_take, List.length_drop,
      Nat.mul_succ]
    omega

/-- The values below `k` that stay below `M` past `a`, while `Q` holds. -/
theorem count_filter (a M k : ℕ) (Q : Prop) [Decidable Q] :
    ((List.range k).filter fun w => decide (a + w < M ∧ Q)).length = if Q then min k (M - a) else 0 := by
  induction k with
  | zero => split_ifs <;> simp
  | succ k ih =>
    rw [List.range_succ, List.filter_append, List.length_append, ih]
    by_cases hQ : Q <;> by_cases h : a + k < M <;> simp [hQ, h] <;> omega

theorem sum_ite_min (c a M : ℕ) :
    ∀ k, ((List.range k).map fun i => if a + i < M then c else 0).sum = c * min k (M - a)
  | 0 => by simp
  | k + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_ite_min c a M k]
    by_cases h : a + k < M
    · simp only [h, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, ite_true, add_zero]
      rw [show min (k + 1) (M - a) = min k (M - a) + 1 by omega, Nat.mul_succ]
    · simp only [h, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, ite_false, add_zero]
      rw [show min (k + 1) (M - a) = min k (M - a) by omega]

theorem sum_min (N C : ℕ) : ∀ B, ((List.range B).map fun i => min N (C - N * i)).sum = min (N * B) C
  | 0 => by simp
  | B + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_min N C B]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.mul_succ]
    omega

/-- **The stream's tiles are the call's**: `S · rowBlocks · colBlocks` of them, for a column tile of `nb > 0` blocks. -/
theorem wordsTiles_length (L : Pouw.Lead.Layout) {nb : ℕ} (hnb : 0 < nb) :
    (wordsTiles L nb).length = L.S * (L.rowBlocks * L.colBlocks) := by
  simp only [wordsTiles, List.length_flatMap, List.length_map, count_filter, sum_ite_min, List.map_const',
    List.sum_replicate, List.length_range, smul_eq_mul, ← mul_assoc, List.sum_map_mul_left, List.sum_map_mul_right,
    sum_min]
  have hA : min (4 * ((L.rowBlocks + 3) / 4)) L.rowBlocks = L.rowBlocks := by omega
  have hB : min (nb * ((L.colBlocks + nb - 1) / nb)) L.colBlocks = L.colBlocks := by
    have := Nat.div_add_mod (L.colBlocks + nb - 1) nb
    have := Nat.mod_lt (L.colBlocks + nb - 1) hnb
    omega
  rw [hA, hB]

theorem aBytes_length (L : Pouw.Lead.Layout) (A : Fin L.rows → Fin L.K → UInt8) :
    (aBytes L A).length = L.rows * L.K := by
  simp only [aBytes, List.length_flatten, List.map_ofFn, Function.comp_def, List.length_ofFn, List.sum_ofFn,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]

/-- **The call's chunks follow its earlier calls'**: the forward's words leaf at rank `before + i` is the call's
chunk `i`. -/
theorem value_call (F : CallForward) (nb : ℕ) (L : Pouw.Lead.Layout) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) {i : ℕ}
    (hi : i < wordsLeaves L nb) : F.value nb L w (F.before nb + i) = wordsChunk L nb w i := by
  have hpre : (F.pre.flatMap (callChunks nb)).length = F.before nb := by
    simp [CallForward.before, callChunks, List.length_flatMap]
  have hcall : (callChunks nb ⟨L, w⟩).length = wordsLeaves L nb := by simp [callChunks]
  rw [CallForward.value, CallForward.values, List.getD_eq_getElem?_getD,
    List.getElem?_append_left (by rw [List.length_append, hpre, hcall]; omega),
    List.getElem?_append_right (by rw [hpre]; omega), hpre, Nat.add_sub_cancel_left]
  simp [callChunks, hi]

/-- **`CommitL1w` at the deployed commitment**: for all parameters whose column tile is a positive number of blocks,
every forward and every A, at every hash and data, when the call's rows and columns are whole MMA tiles and the
forward's words leaves and A's segments number at most `2^64`. -/
theorem commitL1w_l1w (L : Pouw.Lead.Layout) (hm : 16 ∣ L.m) (hn : 8 ∣ L.n) (hcw : 0 < L.chunkWords) (P : L1w)
    (hnb : 0 < P.nb) (F : CallForward) (hkw : F.leaves P.nb L ≤ 2 ^ 64) (hka : aLeaves L ≤ 2 ^ 64)
    (H₀ : Hash.Msg → Hash.Digest) (A : Fin L.rows → Fin L.K → UInt8) (H : Hash.Msg → Hash.Digest)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) :
    CommitL1w H (l1wCommit L P F H₀ A) (l1wIsA L P F H₀ A) κ := by
  have hw : tileWords * (wordsTiles L P.nb).length = L.m * L.n * L.S := by
    rw [wordsTiles_length L hnb]
    obtain ⟨a, ha⟩ := hm
    obtain ⟨b, hb⟩ := hn
    simp only [Pouw.Lead.Layout.rowBlocks, Pouw.Lead.Layout.colBlocks, ha, hb, tileWords]
    rw [show (16 * a + 15) / 16 = a by omega, show (8 * b + 7) / 8 = b by omega]
    ring
  have hch : wordsLeaves L P.nb = wordChunks L := by
    rw [wordsLeaves, hw]
    rfl
  have hk : F.before P.nb + wordsLeaves L P.nb ≤ F.leaves P.nb L := by
    unfold CallForward.leaves
    omega
  have hwc : ∀ w, 4 * (L.m * L.n * L.S) ≤
      ((List.range (wordsLeaves L P.nb)).map fun i => (wordsChunk L P.nb w i).length).sum := by
    intro w
    simp only [wordsChunk]
    rw [sum_chunks, wordsBytes_length, hw, hch]
    have hq := Nat.div_add_mod (L.m * L.n * L.S + L.chunkWords - 1) L.chunkWords
    have hr := Nat.mod_lt (L.m * L.n * L.S + L.chunkWords - 1) hcw
    unfold wordChunks
    generalize (L.m * L.n * L.S + L.chunkWords - 1) / L.chunkWords = q at hq ⊢
    generalize (L.m * L.n * L.S + L.chunkWords - 1) % L.chunkWords = e at hq hr
    generalize L.m * L.n * L.S = a at hq ⊢
    have : 4 * L.chunkWords * q = 4 * (L.chunkWords * q) := by ring
    omega
  refine commitL1w_of_leaves H _ _ κ hch (P.wordsDom H F L) (P.aDom H₀ F L) wordsSchema aSchema
    (F.before P.nb + wordsLeaves L P.nb) (aLeaves L) (hk.trans hkw) hka (Nat.le_add_left _ _)
    (fun ω => F.value P.nb L (word L κ ω)) (aSeg L A) (fun ω => (hwc (word L κ ω)).trans ?_) ?_ (fun ω r hr => ?_)
    (fun ω r hr => ?_) (fun ω r hr => ?_) (fun r hr => ?_)
  · rw [List.range_add, List.map_append, List.sum_append, List.map_map]
    refine le_add_left (le_of_eq (congrArg List.sum (List.map_congr_left fun i hi => ?_)))
    simp only [Function.comp_apply, value_call F P.nb L _ (List.mem_range.1 hi)]
  · simp only [aSeg]
    rw [sum_chunks, aBytes_length]
    unfold aLeaves aSegment
    omega
  · exact List.mem_append_left _ (List.mem_append_left _ (leaf_mem_msgs H _ _ _ (hr.trans_le hk)))
  · exact List.mem_append_right _ (List.mem_append_left _ (leaf_mem_msgs H₀ _ _ _ hr))
  · exact decide_eq_false (leafMsg_not_mem_aMsgs L P F H H₀ A hka (hr.trans_le (hk.trans hkw)) _)
  · exact decide_eq_true (List.mem_append_left _ (leaf_mem_msgs H₀ _ _ _ hr))

/-! ## The certifier's stream holds every coin the feed carries -/

/-- `a · i + x` with `x < a` determines `i` and `x`. -/
theorem radix_inj {a i i' x x' : ℕ} (hx : x < a) (hx' : x' < a) (h : a * i + x = a * i' + x') : i = i' ∧ x = x' := by
  have hi : i = i' := by
    have h1 : (a * i + x) / a = i := by rw [Nat.mul_add_div (by omega), Nat.div_eq_of_lt hx, add_zero]
    have h2 : (a * i' + x') / a = i' := by rw [Nat.mul_add_div (by omega), Nat.div_eq_of_lt hx', add_zero]
    rw [← h1, ← h2, h]
  subst hi
  exact ⟨rfl, by omega⟩

theorem radix_lt {a i m x : ℕ} (hi : i < m) (hx : x < a) : a * i + x < a * m := by
  have := Nat.mul_le_mul_left a (Nat.succ_le_of_lt hi)
  rw [Nat.mul_succ] at this
  omega

/-- **The feed's coins start at distinct bits**: no two coins share one. -/
theorem feedSlot_inj (L : Pouw.Lead.Layout) {x y : CoinIdx L} {k : ℕ} (hx : feedSlot L x = some k)
    (hy : feedSlot L y = some k) : x = y := by
  set W := L.feedRowWords with hW
  have hS : L.S + 3 ≤ 32 * W := by rw [hW, Layout.feedRowWords]; omega
  have fb : ∀ {i : Fin L.m} {t : Fin L.S} {k}, feedSlot L (.inl (i, t)) = some k →
      (t : ℕ) ≠ 0 ∧ 32 * (W * i) + t + 3 = k := by
    intro i t k h
    simp only [feedSlot] at h
    split_ifs at h with h0
    exact ⟨h0, Option.some.inj h⟩
  have fa : ∀ {i : Fin L.m} {e : Fin 4} {k}, feedSlot L (.inr (.inl (i, e))) = some k → 32 * (W * i) + e = k :=
    fun h => by simpa only [feedSlot, Option.some.injEq] using h
  have fg : ∀ {e : Fin 4} {j : Fin L.n} {k}, feedSlot L (.inr (.inr (.inl (e, j)))) = some k →
      32 * (W * L.m + j) + 8 * e = k := fun h => by simpa only [feedSlot, Option.some.injEq] using h
  have fl : ∀ {t : Fin L.S} {k' : Fin L.tiles} {e : Fin 3} {j : Fin L.n} {k},
      feedSlot L (.inr (.inr (.inr (t, k', e, j)))) = some k →
        32 * (W * L.m + L.n + (k' * L.S + t) * L.n + j) + 8 * e = k :=
    fun h => by simpa only [feedSlot, Option.some.injEq] using h
  have hrow : ∀ (i : Fin L.m) p, p < 32 * W → 32 * (W * i) + p < 32 * (W * L.m) := fun i p hp => by
    have := Nat.mul_le_mul_left W (Nat.succ_le_of_lt i.2)
    rw [Nat.mul_succ] at this
    omega
  have rowInj : ∀ (i i' : Fin L.m) p p', p < 32 * W → p' < 32 * W → 32 * (W * i) + p = 32 * (W * i') + p' →
      i = i' ∧ p = p' := fun i i' p p' hp hp' h => by
    obtain ⟨hi, hpp⟩ := radix_inj (i := i) (i' := i') hp hp' (by rw [Nat.mul_assoc, Nat.mul_assoc]; exact h)
    exact ⟨Fin.ext hi, hpp⟩
  rcases x with ⟨i, t⟩ | ⟨i, e⟩ | ⟨e, j⟩ | ⟨t, k₁, e, j⟩ <;>
    rcases y with ⟨i', t'⟩ | ⟨i', e'⟩ | ⟨e', j'⟩ | ⟨t', k₂, e', j'⟩
  · obtain ⟨-, hx⟩ := fb hx
    obtain ⟨-, hy⟩ := fb hy
    obtain ⟨rfl, h⟩ := rowInj i i' (t + 3) (t' + 3) (by omega) (by omega) (by omega)
    rw [Fin.ext (by omega : (t : ℕ) = t')]
  · obtain ⟨h0, hx⟩ := fb hx
    have := rowInj i i' (t + 3) e' (by omega) (by omega) (by have := fa hy; omega)
    omega
  · obtain ⟨-, hx⟩ := fb hx
    have := hrow i (t + 3) (by omega)
    have := fg hy
    omega
  · obtain ⟨-, hx⟩ := fb hx
    have := hrow i (t + 3) (by omega)
    have := fl hy
    omega
  · obtain ⟨h0, hy⟩ := fb hy
    have := rowInj i i' e (t' + 3) (by omega) (by omega) (by have := fa hx; omega)
    omega
  · obtain ⟨rfl, h⟩ := rowInj i i' e e' (by omega) (by omega) ((fa hx).trans (fa hy).symm)
    rw [Fin.ext h]
  · have := hrow i e (by omega)
    have := fa hx
    have := fg hy
    omega
  · have := hrow i e (by omega)
    have := fa hx
    have := fl hy
    omega
  · obtain ⟨-, hy⟩ := fb hy
    have := hrow i' (t' + 3) (by omega)
    have := fg hx
    omega
  · have := hrow i' e' (by omega)
    have := fa hy
    have := fg hx
    omega
  · have := fg hx
    have := fg hy
    rw [Fin.ext (by omega : (e : ℕ) = e'), Fin.ext (by omega : (j : ℕ) = j')]
  · have := fg hx
    have := fl hy
    omega
  · obtain ⟨-, hy⟩ := fb hy
    have := hrow i' (t' + 3) (by omega)
    have := fl hx
    omega
  · have := hrow i' e' (by omega)
    have := fa hy
    have := fl hx
    omega
  · have := fl hx
    have := fg hy
    omega
  · have h1 := fl hx
    have h2 := fl hy
    have he : (e : ℕ) = e' := by omega
    obtain ⟨hk, hj⟩ := radix_inj j.2 j'.2
      (by rw [Nat.mul_comm, Nat.mul_comm L.n]; omega : L.n * (k₁ * L.S + t) + j = L.n * (k₂ * L.S + t') + j')
    obtain ⟨hk, ht⟩ := radix_inj t.2 t'.2
      (by rw [Nat.mul_comm, Nat.mul_comm L.S]; exact hk : L.S * k₁ + t = L.S * k₂ + t')
    rw [Fin.ext ht, Fin.ext hk, Fin.ext he, Fin.ext hj]

/-- **The certifier's stream holds each coin at its feed bit**, within the node's storage: every coin but step 0's tag
bits starts at a bit of the feed (`feedSlot`), and no other coin starts there. -/
theorem certStream_ι (L : Pouw.Lead.Layout) {x : CoinIdx L} {c : ℕ} {p : Fin tileWords} (hc : c < L.mem)
    (hx : feedSlot L x = some (tileWords * c + p)) : (certStream L).ι c p = some x := by
  have h : c < L.mem ∧ ∃ x, feedSlot L x = some (tileWords * c + p) := ⟨hc, x, hx⟩
  show (if h : c < L.mem ∧ ∃ x, feedSlot L x = some (tileWords * c + p) then some h.2.choose else none) = some x
  split_ifs
  exact congrArg some (feedSlot_inj L h.2.choose_spec hx)

/-- An A of at most `2^64` bytes has at most `2^64` segments. -/
theorem aLeaves_le (T m n K : ℕ) (hA : m * K ≤ 2 ^ 64) : aLeaves (lead T m n K) ≤ 2 ^ 64 := by
  simp only [aLeaves, aSegment, lead]
  omega

/-- **The honest lemma at the deployed instance**: at every admitted call of the lead whose A is at most `2^64` bytes,
every hash and data, all commitment parameters whose column tile is a positive number of blocks, every forward of the
call whose words leaves number at most `2^64`, and every A, committed under the same hash, some set of strategies
meets every premise of `G` at the deployed commitment `l1wCommit`, its classifier `l1wIsA` and the certifier's stream
`certStream`, and holds an honest strategy accepted at every coin value. -/
theorem honestL1w (T m n K : ℕ) (hadm : (lead T m n K).Admits) (hA : m * K ≤ 2 ^ 64) (H : Hash.Msg → Hash.Digest)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ) (P : L1w) (hnb : 0 < P.nb)
    (F : CallForward) (hF : F.leaves P.nb (lead T m n K) ≤ 2 ^ 64)
    (A : Fin (lead T m n K).rows → Fin (lead T m n K).K → UInt8) :
    ∃ (Real : Strategy (lead T m n K) → Prop) (σ : Strategy (lead T m n K)),
      CommitL1w H (l1wCommit _ P F H A) (l1wIsA _ P F H A) κ ∧ H32 h100Nominal ∧ W1 h100Nominal Real ∧
      TileGrain H (l1wCommit _ P F H A) κ (certStream _) Real ∧ OneInstructionMinimal Real ∧
      CardWithinAccelW1 Real Real ∧ ShaWrites (l1wIsA _ P F H A) Real ∧
      HonestHashRuled H (l1wCommit _ P F H A) κ Real ∧
      OutsourceWithinEgressLead h100Nominal (betaLead (lead T m n K) (l1wCommit _ P F H A)) H
        (l1wCommit _ P F H A) κ Real ∧
      Real σ ∧ ∀ ω, Accepted (lead T m n K) H (l1wCommit _ P F H A) κ σ ω :=
  honestOfCommit T m n K H κ (certStream _) (l1wCommit _ P F H A) (l1wIsA _ P F H A)
    (commitL1w_l1w _ (Dvd.intro _ rfl) (Dvd.intro _ rfl) (show 0 < 65536 by norm_num) P hnb F hF
      (aLeaves_le T m n K hA) H A H κ) hadm

/-- **The honest lemma** (`Pouw.Guarantees.Lead.Honest`): `certifierCoins` beside `honestL1w`, at every admitted
call. -/
theorem honest : Pouw.Guarantees.Lead.Honest := fun T m n K H κ hadm =>
  ⟨⟨Unit, _, (), _, _, certifierCoins _⟩, fun P F A hnb hA hF => honestL1w T m n K hadm hA H κ P hnb F hF A⟩

end Pouw.SecurityProofs.Lead.Compose
