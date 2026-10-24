import Proofs.Flock.Claims.ZK.FrameRounds
import Proofs.Flock.Claims.ZK.FrameOuter
import Proofs.Flock.Claims.ZK.FramePass

/-!
# The values a round carries freely, read back off its bytes

`Frame.roundFree` returns, beside the round's release, the offsets in the round's bytes of the values it absorbs in the
algebraic phase, and the loop reads each free value as the 16-byte word there (`Loop.maskedOf`). At the bytes of frames
`fs` and the round's squeeze, those words are the frames' values, in order (`fvals`):

* **The folds** (`foldlM_absorb1_vals`, `foldlM_stepAt_vals`): inside a masked phase below the algebraic phase's end,
  each absorbed value's offset is freed, and the word there is the value.
* **The rounds** (`roundFree_vals`, `roundFree_zcFree`): a round inside the algebraic phase frees its frames' values;
  the zerocheck's first round, which opens the phase, frees none.
* **The coins** (`answer_ok`): where none of a round's coins at a zerocheck outer coin's place is `1`, the frame takes
  them and keeps them.
* **A value set at its place** (`foldl_set_at`): the fold that sets each pair's place, read at a place one pair alone
  names.
-/

namespace Claims.ZKServer

open _root_.Flock (V F128)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame
  (Cfg Masker labelFrame bytesFrame roundFree LABEL_ZC K_SKIP N_INNER INNER_ROUNDS)
open FlockSoundness.Refine (Frame encs colBytes toBytes_length)

/-- The values the frames `fs` absorb, in order. -/
def fvals : List Frame → List F128
  | [] => []
  | .dom _ :: fs => fvals fs
  | .label _ :: fs => fvals fs
  | .bytes _ :: fs => fvals fs
  | .one v :: fs => v :: fvals fs
  | .vec vs :: fs => vs ++ fvals fs
  | .sq _ _ :: fs => fvals fs

/-- The word at byte `o` of `L`, as the loop reads a free value (`Frame.words`). -/
def wd (L : List UInt8) (o : ℕ) : F128 := (_root_.Flock.Firewall.Frame.words (ByteArray.mk L.toArray) o 1)[0]!

theorem wd_drop {L : List UInt8} {q : ℕ} {v : F128} {rest : List UInt8}
    (h : L.drop q = v.toBytes.data.toList ++ rest) : wd L q = v := by
  have hq : q ≤ L.length := by
    by_contra hc
    rw [List.drop_eq_nil_of_le (by omega)] at h
    have e := congrArg List.length h
    rw [List.length_append, toBytes_length] at e
    simp only [List.length_nil] at e
    omega
  obtain ⟨pre, hpre, hlen⟩ : ∃ pre, L = pre ++ v.toBytes.data.toList ++ rest ∧ pre.length = q :=
    ⟨L.take q, by rw [List.append_assoc, ← h, List.take_append_drop],
      by rw [List.length_take]; exact Nat.min_eq_left hq⟩
  subst hpre hlen
  unfold wd _root_.Flock.Firewall.Frame.words
  rw [getElem!_pos _ 0 (by simp), Array.getElem_map, Array.getElem_range]
  simp only [Nat.mul_zero, Nat.add_zero]
  rw [word_lo, word_hi]

/-- **An observe's values below the algebraic phase's end are freed**, each at its offset in `L`. -/
theorem foldlM_absorb1_vals (c : Cfg) (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    (hN : c.nPiop ≤ c.order.size) (L : List UInt8) :
    ∀ (vs : List F128) (s : Masker) (o : ByteArray) (fr : Array ℕ) (q : ℕ) (post : List UInt8),
      s.on = true → s.pos + vs.length ≤ c.nPiop → L.drop q = colBytes vs ++ post →
      ∃ r, vs.foldlM (absorb1 c) (s, o, fr, q) = .ok r ∧ r.1 = { s with pos := s.pos + vs.length } ∧
        r.2.2.1.toList.map (wd L) = fr.toList.map (wd L) ++ vs
  | [], s, o, fr, q, _, _, _, _ => ⟨(s, o, fr, q), rfl, by simp, by simp⟩
  | v :: vs, s, o, fr, q, post, hon, hp, hd => by
    simp only [List.length_cons] at hp
    rw [List.foldlM_cons]
    have ha : absorb1 c (s, o, fr, q) v = .ok ({ s with pos := s.pos + 1 },
        F128.pushBytes o (maskVal c s.stopped s.pos v), fr.push q, q + 16) := by
      unfold absorb1
      rw [absorb_on c s v hon (by omega) (hc _ (by omega))]
      simp only [hon, Bool.true_and, decide_eq_true (show s.pos < c.nPiop by omega), ↓reduceIte]
      rfl
    rw [ha]
    have hcb : colBytes (v :: vs) = v.toBytes.data.toList ++ colBytes vs := by simp [colBytes]
    rw [hcb, List.append_assoc] at hd
    have hd' : L.drop (q + 16) = colBytes vs ++ post := by
      rw [← List.drop_drop, hd, List.drop_left' (toBytes_length v)]
    obtain ⟨r, hr, h1, h2⟩ := foldlM_absorb1_vals c hc hN L vs { s with pos := s.pos + 1 } _ (fr.push q) (q + 16)
      post hon (by simp; omega) hd'
    refine ⟨r, hr, ?_, ?_⟩
    · rw [h1]; simp only [List.length_cons]; congr 1; omega
    · rw [h2, Array.toList_push, List.map_append, List.map_singleton, wd_drop hd]
      simp

/-- The position after a fold of the frame's step is past the frames' bytes. -/
theorem foldlM_stepAt_pos (c : Cfg) : ∀ (fs : List Frame) (r r' : Masker × ByteArray × Array ℕ × ℕ),
    fs.foldlM (stepAt c) r = .ok r' → r'.2.2.2 = r.2.2.2 + (encs fs).length
  | [], r, r', h => by cases h; simp [encs]
  | f :: fs, r, r', h => by
    rw [List.foldlM_cons] at h
    cases hs : stepAt c r f with
    | error e => rw [hs] at h; cases h
    | ok q =>
      rw [hs] at h
      rw [foldlM_stepAt_pos c fs q r' h, stepAt_pos c r q f hs, encs_cons, List.length_append]
      omega

/-- **A masked phase's frames below the algebraic phase's end free their values**, each at its offset in `L`, from a
state inside the phase whose frames sit at `q` in `L`. -/
theorem foldlM_stepAt_vals (c : Cfg) (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    (hN : c.nPiop ≤ c.order.size) (L : List UInt8) :
    ∀ (fs : List Frame) (s : Masker) (o : ByteArray) (fr : Array ℕ) (q : ℕ) (post : List UInt8),
      s.on = true → (∀ f ∈ fs, plainFrame f) → s.pos + nAbs fs ≤ c.nPiop → L.drop q = encs fs ++ post →
      ∃ r, fs.foldlM (stepAt c) (s, o, fr, q) = .ok r ∧ r.1 = { s with pos := s.pos + nAbs fs } ∧
        r.2.2.1.toList.map (wd L) = fr.toList.map (wd L) ++ fvals fs
  | [], s, o, fr, q, _, _, _, _, _ => ⟨(s, o, fr, q), rfl, by simp [nAbs], by simp [fvals]⟩
  | f :: fs, s, o, fr, q, post, hon, hp, hb, hd => by
    rw [List.foldlM_cons]
    have hp' : ∀ g ∈ fs, plainFrame g := fun g hg => hp g (List.mem_cons_of_mem _ hg)
    rw [encs_cons, List.append_assoc] at hd
    have hnext : ∀ r', stepAt c (s, o, fr, q) f = .ok r' → L.drop r'.2.2.2 = encs fs ++ post := by
      intro r' h
      rw [stepAt_pos c _ r' f h]
      dsimp only
      rw [← List.drop_drop, hd, List.drop_left]
    cases f with
    | dom b =>
      have hs : stepAt c (s, o, fr, q) (.dom b) = .ok (s, o, fr, q + 16 + (b.size + 15) / 16 * 16) := rfl
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_vals c hc hN L fs s o fr _ post hon hp'
        (by simpa [nAbs] using hb) (hnext _ hs)
      exact ⟨r, by rw [hs]; exact hr, by rw [h1]; rfl, by rw [h2]; rfl⟩
    | label b =>
      have hl := label_plain c s b (hp _ List.mem_cons_self)
      have hs : stepAt c (s, o, fr, q) (.label b) =
          .ok (s, o ++ labelFrame b, fr, q + 16 + (b.size + 15) / 16 * 16) := by
        simp only [stepAt, hl] <;> rfl
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_vals c hc hN L fs s _ fr _ post hon hp'
        (by simpa [nAbs] using hb) (hnext _ hs)
      exact ⟨r, by rw [hs]; exact hr, by rw [h1]; rfl, by rw [h2]; rfl⟩
    | bytes b =>
      have hs : stepAt c (s, o, fr, q) (.bytes b) =
          .ok (s, o ++ bytesFrame b, fr, q + 16 + (b.size + 15) / 16 * 16) := rfl
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_vals c hc hN L fs s _ fr _ post hon hp'
        (by simpa [nAbs] using hb) (hnext _ hs)
      exact ⟨r, by rw [hs]; exact hr, by rw [h1]; rfl, by rw [h2]; rfl⟩
    | one v =>
      simp only [nAbs] at hb
      have hd1 : L.drop (q + 16) = colBytes [v] ++ (encs fs ++ post) := by
        rw [← List.drop_drop, hd, enc_one, List.append_assoc,
          List.drop_left' (FlockSoundness.Refine.header_length 3 1 1)]
        simp [colBytes]
      obtain ⟨r₁, hr₁, h1₁, h2₁⟩ := foldlM_absorb1_vals c hc hN L [v] s (o ++ header 3 1 1) fr (q + 16) _ hon
        (by simp; omega) hd1
      have hs : stepAt c (s, o, fr, q) (.one v) = .ok r₁ := hr₁
      obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
      simp only at h1₁ h2₁
      subst h1₁
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_vals c hc hN L fs { s with pos := s.pos + 1 } o₁ fr₁ q₁ post hon hp'
        (by simp; omega) (hnext _ hs)
      refine ⟨r, by rw [hs]; exact hr, ?_, ?_⟩
      · rw [h1]; simp only [nAbs]; congr 1; omega
      · rw [h2, h2₁]; simp [fvals]
    | vec vs =>
      simp only [nAbs] at hb
      have hd1 : L.drop (q + 16) = colBytes vs ++ (encs fs ++ post) := by
        rw [← List.drop_drop, hd, enc_vec, List.append_assoc,
          List.drop_left' (FlockSoundness.Refine.header_length 3 2 vs.length)]
      obtain ⟨r₁, hr₁, h1₁, h2₁⟩ := foldlM_absorb1_vals c hc hN L vs s (o ++ header 3 2 vs.length) fr (q + 16) _
        hon (by omega) hd1
      have hs : stepAt c (s, o, fr, q) (.vec vs) = .ok r₁ := hr₁
      obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
      simp only at h1₁ h2₁
      subst h1₁
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_vals c hc hN L fs { s with pos := s.pos + vs.length } o₁ fr₁ q₁ post
        hon hp' (by simp; omega) (hnext _ hs)
      refine ⟨r, by rw [hs]; exact hr, ?_, ?_⟩
      · rw [h1]; simp only [nAbs]; congr 1; omega
      · rw [h2, h2₁]; simp [fvals]
    | sq k n =>
      obtain ⟨rfl, rfl⟩ : k = 2 ∧ n = 0 := hp _ List.mem_cons_self
      have hs : stepAt c (s, o, fr, q) (.sq 2 0) = .ok (s, o ++ header 4 2 0, fr, q + 16) := rfl
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_vals c hc hN L fs s _ fr _ post hon hp'
        (by simpa [nAbs] using hb) (hnext _ hs)
      exact ⟨r, by rw [hs]; exact hr, by rw [h1]; rfl, by rw [h2]; rfl⟩

/-- **A round inside the algebraic phase frees its frames' values**: at the bytes of frames `fs` inside the phase, then
the squeeze of `n` coins of kind `k`, the words at the offsets the frame frees are the frames' values. -/
theorem roundFree_vals (c : Cfg) (s : Masker) (fs : List Frame) (k : UInt8) (n : ℕ)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hN : c.nPiop ≤ c.order.size)
    (hon : s.on = true) (hi : s.inner = none) (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f)
    (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) (hb : s.pos + nAbs fs ≤ c.nPiop) :
    ∃ out fr, roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n =
        .ok (out, { s with pos := s.pos + nAbs fs, owed := some n }, fr) ∧
      fr.toList.map (wd (encs fs ++ (Frame.sq k n).enc)) = fvals fs := by
  obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_vals c hc hN (encs fs ++ (Frame.sq k n).enc) fs s ByteArray.empty #[] 0 _
    hon hp hb (List.drop_zero _)
  rw [roundFree_encs c s fs k n (by rw [hi]; simp) hm hk h0 hn, hr]
  obtain ⟨m, o, fr, q⟩ := r
  simp only at h1 h2
  subst h1
  refine ⟨o ++ header 4 k n, fr, ?_, by simpa using h2⟩
  simp only [bind, Except.bind, hi]
  rfl

/-- **The zerocheck's first round frees nothing**: at frames `pre` outside a masked phase, the zerocheck's label, then
frames `post` that absorb nothing, and the squeeze, the frame starts the algebraic phase and frees no offset. -/
theorem roundFree_zcFree (c : Cfg) (s : Masker) (pre post : List Frame) (k : UInt8) (n : ℕ)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hN : c.nPiop ≤ c.order.size)
    (hon : s.on = false) (hd : s.done = false) (hi : s.inner = none)
    (hm : ∀ f ∈ pre ++ Frame.label LABEL_ZC.toUTF8 :: post, midFrame f)
    (hpre : ∀ f ∈ pre, plainFrame f) (hpost : ∀ f ∈ post, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) (hb : s.pos ≤ c.nPiop) (hv : fvals post = [])
    (ha : nAbs post = 0) :
    ∃ out, roundFree c s
        (ByteArray.mk (encs (pre ++ Frame.label LABEL_ZC.toUTF8 :: post) ++ (Frame.sq k n).enc).toArray) n =
        .ok (out, { s with on := true, owed := some n }, #[]) := by
  obtain ⟨r₁, hr₁, h1₁, -⟩ := foldlM_stepAt_off c pre s ByteArray.empty #[] 0 0 hon hpre
  have hf₁ := foldlM_stepAt_off_fr c pre s ByteArray.empty #[] 0 r₁ hon hpre hr₁
  have hq₁ := foldlM_stepAt_pos c pre _ r₁ hr₁
  obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
  simp only at h1₁ hf₁ hq₁
  subst h1₁ hf₁ hq₁
  have hs₂ : stepAt c (m, o₁, #[], 0 + (encs pre).length) (Frame.label LABEL_ZC.toUTF8) =
      .ok ({ m with on := true },
        o₁ ++ (labelFrame LABEL_ZC.toUTF8 ++ labelFrame _root_.Flock.Zk.LABEL_PADS.toUTF8 ++ bytesFrame c.root), #[],
        0 + (encs pre).length + 16 + (LABEL_ZC.toUTF8.size + 15) / 16 * 16) := by
    simp only [stepAt, label_zc c m hon hd] <;> rfl
  have hp₂ := stepAt_pos c _ _ _ hs₂
  simp only at hp₂
  have he : encs (pre ++ Frame.label LABEL_ZC.toUTF8 :: post) ++ (Frame.sq k n).enc =
      encs pre ++ ((Frame.label LABEL_ZC.toUTF8).enc ++ (encs post ++ (Frame.sq k n).enc)) := by
    simp [encs, List.append_assoc]
  have hd₂ : (encs (pre ++ Frame.label LABEL_ZC.toUTF8 :: post) ++ (Frame.sq k n).enc).drop
      (0 + (encs pre).length + 16 + (LABEL_ZC.toUTF8.size + 15) / 16 * 16) = encs post ++ (Frame.sq k n).enc := by
    rw [hp₂, Nat.zero_add, ← List.drop_drop, he, List.drop_left, List.drop_left]
  obtain ⟨r₂, hr₂, h1₂, h2₂⟩ := foldlM_stepAt_vals c hc hN _ post { m with on := true } _ #[] _ _ rfl hpost
    (by rw [ha]; simpa using hb) hd₂
  have hfold : (pre ++ Frame.label LABEL_ZC.toUTF8 :: post).foldlM (stepAt c) (m, ByteArray.empty, #[], 0) =
      .ok r₂ := by
    rw [List.foldlM_append, hr₁]
    show (Frame.label LABEL_ZC.toUTF8 :: post).foldlM (stepAt c) (m, o₁, #[], 0 + (encs pre).length) = .ok r₂
    rw [List.foldlM_cons, hs₂]
    exact hr₂
  rw [roundFree_encs c m _ k n (by rw [hi]; simp) hm hk h0 hn, hfold]
  obtain ⟨m₂, o₂, fr₂, q₂⟩ := r₂
  simp only [ha, Nat.add_zero] at h1₂ h2₂
  subst h1₂
  have hfr : fr₂ = #[] := by
    rw [hv] at h2₂
    simpa using h2₂
  subst hfr
  refine ⟨o₂ ++ header 4 k n, ?_⟩
  simp only [bind, Except.bind, hi]
  rfl

/-- A loop each of whose steps yields returns. -/
theorem forIn_yield_all {g : ℕ → PUnit → V (ForInStep PUnit)} :
    ∀ (l : List ℕ), (∀ i ∈ l, g i PUnit.unit = .ok (.yield PUnit.unit)) → forIn l PUnit.unit g = .ok PUnit.unit
  | [], _ => rfl
  | a :: l, h => by
    rw [List.forIn_cons, h a List.mem_cons_self]
    exact forIn_yield_all l fun i hi => h i (List.mem_cons_of_mem _ hi)

/-- **The frame takes a round's coins in the algebraic phase** where none at a zerocheck outer coin's place is `1`,
and keeps them. -/
theorem answer_ok {cfg : Cfg} {s : Masker} {cs : Array F128} {n : ℕ} (ho : s.owed = some n) (hc : cs.size = n)
    (hi : s.inner = none) (hl : s.l0 = false) (hon : s.on = true)
    (hz : ∀ e < n, K_SKIP ≤ s.coins.size + e → s.coins.size + e < K_SKIP + (cfg.lay.nMlv - N_INNER) →
      (cs[e]! + 1).isZero = false) :
    s.answer cfg cs = .ok { s with owed := none, coins := s.coins ++ cs } := by
  unfold Masker.answer
  have hne : (cs.size != n) = false := by simp [hc]
  simp only [ho, hi, hl, hne, hon, Bool.false_eq_true, ↓reduceIte, Bool.not_true]
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  rw [forIn_yield_all _ fun i hi' => ?_]
  · rfl
  · rw [List.mem_range'] at hi'
    obtain ⟨e, he, hie⟩ := hi'
    simp only [Std.Legacy.Range.size, Array.size_append] at he
    have he' : e < n := by omega
    simp only [Nat.one_mul] at hie
    subst hie
    try dsimp only
    split
    · rename_i hcond
      exfalso
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hcond
      obtain ⟨⟨h1, h2⟩, h3⟩ := hcond
      rw [Loop.getElem!_append_size (by omega)] at h3
      rw [hz e he' h1 h2] at h3
      cases h3
    · rfl

/-! ## A value set at its place -/

theorem set!_ne {α : Type} [Inhabited α] (v : Array α) {j i : ℕ} (a : α) (h : j ≠ i) :
    (v.set! j a)[i]! = v[i]! := by
  simp [getElem!_def, h]

theorem set!_hit {α : Type} [Inhabited α] (v : Array α) {j : ℕ} (a : α) (h : j < v.size) :
    (v.set! j a)[j]! = a := by
  simp [h]

theorem foldl_set_size {α β : Type} (f : Array α → β → Array α) (hf : ∀ v e, (f v e).size = v.size) :
    ∀ (Z : List β) (v : Array α), (Z.foldl f v).size = v.size
  | [], _ => rfl
  | e :: Z, v => by rw [List.foldl_cons, foldl_set_size f hf Z, hf]

theorem foldl_set_miss {α : Type} [Inhabited α] (g : ℕ → α → α) (f : Array α → ℕ × α → Array α)
    (hf : ∀ v e, f v e = v.set! e.1 (g e.1 e.2)) :
    ∀ (Z : List (ℕ × α)) (v : Array α) (i : ℕ), (∀ e ∈ Z, e.1 ≠ i) → (Z.foldl f v)[i]! = v[i]!
  | [], _, _, _ => rfl
  | e :: Z, v, i, h => by
    rw [List.foldl_cons, foldl_set_miss g f hf Z _ i (fun e' he' => h e' (List.mem_cons_of_mem _ he')), hf]
    exact set!_ne v _ (h e List.mem_cons_self)

/-- **The fold that sets each pair's place, at a place pair `p` alone names**, holds `g` of pair `p`. -/
theorem foldl_set_at {α : Type} [Inhabited α] (g : ℕ → α → α) (f : Array α → ℕ × α → Array α)
    (hf : ∀ v e, f v e = v.set! e.1 (g e.1 e.2)) (Z : List (ℕ × α)) (v : Array α) {i p : ℕ}
    (hp : p < Z.length) (hi : Z[p].1 = i) (hu : ∀ q (hq : q < Z.length), Z[q].1 = i → q = p) (hv : i < v.size) :
    (Z.foldl f v)[i]! = g i Z[p].2 := by
  have hZ : Z = Z.take p ++ Z[p] :: Z.drop (p + 1) := by
    rw [← List.drop_eq_getElem_cons hp, List.take_append_drop]
  generalize hZp : Z[p] = zp at hi ⊢
  rw [hZp] at hZ
  rw [hZ, List.foldl_append, List.foldl_cons]
  rw [foldl_set_miss g f hf _ _ i ?_, hf, hi]
  · apply set!_hit
    rw [foldl_set_size f (fun v e => by rw [hf]; simp) _ v]; exact hv
  · intro e he heq
    obtain ⟨q, hq, rfl⟩ := List.getElem_of_mem he
    rw [List.getElem_drop] at heq
    have := hu _ (by simp at hq; omega) heq
    omega

end Claims.ZKServer
