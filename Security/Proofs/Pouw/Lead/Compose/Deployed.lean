import Definitions.Pouw.Lead.Deployed
import Proofs.Pouw.Lead.Compose.Honest

/-!
# The lead's honest lemma at the deployed instance

At the deployed commitment (`l1wCommit`) and form classifier (`l1wIsA`), `CommitL1w` holds (`commitL1w_l1w`) when the
two domains' ids differ and the words chunks and A's rows number at most `2^64` (`range-u64` domains refuse more): each
words chunk and each row of A is its tree's frame leaf at its own rank, so `commitL1w_of_leaves` applies. A words leaf
is no message of A's instance (`leafMsg_not_mem_aMsgs`): A's leaves carry A's id (`leafMsg_id_inj`), and its pads,
nodes, empty root and hm96 messages differ from a leaf in a header byte. The chunks cover the words' bytes
(`sum_chunks`), and A's rows its `rows · K` bytes (`sum_aRow`).

`honestL1w` is the honest lemma at the deployed instance: `honestOfCommit` at `l1wCommit`, `l1wIsA` and the certifier's
stream `certStream`, at every admitted call of at most `2^64` rows; the chunks' bound follows from the call's
admission.
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

/-- A tree's messages are its empty root's, its leaves', its pads' and its nodes'. -/
theorem mem_msgs {H : Hash.Msg → Hash.Digest} {D : Hash.Tree.Domain} {s : String} {v : ℕ → Hash.Msg} {y : Hash.Msg}
    (h : y ∈ Hash.Tree.msgs H D s v) :
    y = Hash.Tree.emptyMsg D ∨ (∃ r, y = Hash.Tree.leafMsg D r s (v r)) ∨ (∃ r, y = Hash.Tree.padMsg D r) ∨
      ∃ lv j l r, y = Hash.Tree.nodeMsg D lv j l r := by
  unfold Hash.Tree.msgs at h
  split_ifs at h
  · exact Or.inl (List.mem_singleton.1 h)
  · simp only [Hash.Tree.leaves, List.mem_append, List.mem_map] at h
    rcases h with (⟨r, -, rfl⟩ | ⟨r, -, rfl⟩) | h
    · exact Or.inr (Or.inl ⟨r, rfl⟩)
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

/-! ## `CommitL1w` at the deployed commitment -/

/-- **A words leaf is no message of A's instance**, when the two domains' ids differ. -/
theorem leafMsg_not_mem_aMsgs (L : Pouw.Lead.Layout) (P : L1w) (H₀ : Hash.Msg → Hash.Digest)
    (A : Fin L.rows → Fin L.K → UInt8) (hid : P.idw ≠ P.ida) (r : ℕ) (v : Hash.Msg) :
    Hash.Tree.leafMsg (P.wordsDom L) r P.sw v ∉ (l1wCommit L P H₀ A).aMsgs := by
  intro h
  simp only [l1wCommit, List.mem_append] at h
  rcases h with h | h
  · rcases mem_msgs h with h | ⟨_, h⟩ | ⟨_, h⟩ | ⟨_, _, _, _, h⟩
    · exact leafMsg_ne_emptyMsg h
    · exact hid (leafMsg_id_inj h)
    · exact leafMsg_ne_padMsg h
    · exact leafMsg_ne_nodeMsg h
  · exact leafMsg_not_mem_hm96 _ _ _ _ h

theorem wordsBytes_length (L : Pouw.Lead.Layout) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) :
    (wordsBytes L w).length = 4 * (L.m * L.n * L.S) := by
  simp only [wordsBytes, wordBytes, List.length_flatten, List.map_ofFn, Function.comp_def, List.length_reverse,
    Hash.BindingProofs.be_length, List.sum_ofFn, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
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

/-- A's rows below `k ≤ rows` are `k · K` bytes. -/
theorem sum_aRow (L : Pouw.Lead.Layout) (A : Fin L.rows → Fin L.K → UInt8) :
    ∀ k ≤ L.rows, ((List.range k).map fun r => (aRow L A r).length).sum = k * L.K
  | 0, _ => by simp
  | k + 1, hk => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_aRow L A k (by omega)]
    simp [aRow, show k < L.rows by omega, Nat.succ_mul]

/-- **`CommitL1w` at the deployed commitment**: for all parameters whose two domain ids differ and every A, at
every hash and data, when the call's words chunks and A's rows number at most `2^64`. -/
theorem commitL1w_l1w (L : Pouw.Lead.Layout) (hcw : 0 < L.chunkWords) (hkw : wordChunks L ≤ 2 ^ 64)
    (hka : L.rows ≤ 2 ^ 64) (P : L1w) (hid : P.idw ≠ P.ida) (H₀ : Hash.Msg → Hash.Digest)
    (A : Fin L.rows → Fin L.K → UInt8) (H : Hash.Msg → Hash.Digest) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) :
    CommitL1w H (l1wCommit L P H₀ A) (l1wIsA L P H₀ A) κ := by
  refine commitL1w_of_leaves H _ _ κ rfl (P.wordsDom L) (P.aDom L) P.sw P.sa (wordChunks L) L.rows hkw hka le_rfl
    (fun ω => wordsChunk L (word L κ ω)) (aRow L A) (fun ω => ?_) (sum_aRow L A _ le_rfl).ge
    (fun ω r hr => ?_) (fun ω r hr => ?_) (fun ω r _ => ?_) (fun r hr => ?_)
  · simp only [wordsChunk]
    rw [sum_chunks, wordsBytes_length]
    have hq := Nat.div_add_mod (L.m * L.n * L.S + L.chunkWords - 1) L.chunkWords
    have hr := Nat.mod_lt (L.m * L.n * L.S + L.chunkWords - 1) hcw
    unfold wordChunks
    generalize (L.m * L.n * L.S + L.chunkWords - 1) / L.chunkWords = q at hq ⊢
    generalize (L.m * L.n * L.S + L.chunkWords - 1) % L.chunkWords = e at hq hr
    generalize L.m * L.n * L.S = a at hq ⊢
    have : 4 * L.chunkWords * q = 4 * (L.chunkWords * q) := by ring
    omega
  · exact List.mem_append_left _ (List.mem_append_left _ (leaf_mem_msgs H _ _ _ hr))
  · exact List.mem_append_right _ (List.mem_append_left _ (leaf_mem_msgs H₀ _ _ _ hr))
  · exact decide_eq_false (leafMsg_not_mem_aMsgs L P H₀ A hid r _)
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

/-- **The certifier's stream holds each coin at its feed bit**, within the node's memory: every coin but step 0's tag
bits starts at a bit of the feed (`feedSlot`), and no other coin starts there. -/
theorem certStream_ι (L : Pouw.Lead.Layout) {x : CoinIdx L} {c : ℕ} {p : Fin tileWords} (hc : c < L.mem)
    (hx : feedSlot L x = some (tileWords * c + p)) : (certStream L).ι c p = some x := by
  have h : c < L.mem ∧ ∃ x, feedSlot L x = some (tileWords * c + p) := ⟨hc, x, hx⟩
  show (if h : c < L.mem ∧ ∃ x, feedSlot L x = some (tileWords * c + p) then some h.2.choose else none) = some x
  split_ifs
  exact congrArg some (feedSlot_inj L h.2.choose_spec hx)

/-- An admitted call of the lead has at most `2^64` words chunks: its words are at most `2^47`. -/
theorem wordChunks_le (T m n K : ℕ) (hadm : (lead T m n K).Admits) : wordChunks (lead T m n K) ≤ 2 ^ 64 := by
  have h := hadm.2
  have hw : (lead T m n K).m * (lead T m n K).n * (lead T m n K).S =
      128 * ((lead T m n K).rowBlocks * (lead T m n K).colBlocks * (lead T m n K).S) := by
    simp only [lead, Layout.rowBlocks, Layout.colBlocks, padRows, padCols]
    rw [show (16 * ((m + 15) / 16) + 15) / 16 = (m + 15) / 16 by omega,
      show (8 * ((n + 7) / 8) + 7) / 8 = (n + 7) / 8 by omega]
    ring
  unfold wordChunks
  rw [hw]
  simp only [lead] at h ⊢
  omega

/-- **The honest lemma at the deployed instance**: at every admitted call of the lead of at most `2^64` rows, every
hash and data, all commitment parameters whose two domain ids differ and every A, committed under the same hash,
some set of strategies meets every premise of `G` at the deployed commitment `l1wCommit`, its classifier `l1wIsA` and
the certifier's stream `certStream`, and holds an honest strategy accepted at every coin value. -/
theorem honestL1w (T m n K : ℕ) (hadm : (lead T m n K).Admits) (hm : m ≤ 2 ^ 64) (H : Hash.Msg → Hash.Digest)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ) (P : L1w) (hid : P.idw ≠ P.ida)
    (A : Fin (lead T m n K).rows → Fin (lead T m n K).K → UInt8) :
    ∃ (Real : Strategy (lead T m n K) → Prop) (σ : Strategy (lead T m n K)),
      CommitL1w H (l1wCommit _ P H A) (l1wIsA _ P H A) κ ∧ H32 h100Nominal ∧ W1 h100Nominal Real ∧
      TileGrain H (l1wCommit _ P H A) κ (certStream _) Real ∧ OneInstructionMinimal Real ∧ CardWithinAccelW1 Real Real ∧
      ShaWrites (l1wIsA _ P H A) Real ∧ HonestHashRuled H (l1wCommit _ P H A) κ Real ∧
      OutsourceWithinEgressLead h100Nominal (betaLead (lead T m n K) (l1wCommit _ P H A)) H (l1wCommit _ P H A) κ Real ∧
      Real σ ∧ ∀ ω, Accepted (lead T m n K) H (l1wCommit _ P H A) κ σ ω :=
  honestOfCommit T m n K H κ (certStream _) (l1wCommit _ P H A) (l1wIsA _ P H A)
    (commitL1w_l1w _ (show 0 < 65536 by norm_num) (wordChunks_le T m n K hadm) hm P hid H A H κ) hadm

end Pouw.SecurityProofs.Lead.Compose
