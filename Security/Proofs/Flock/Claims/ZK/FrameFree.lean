import Proofs.Flock.Claims.ZK.FrameRounds
import Proofs.Flock.Claims.ZK.FrameOuter
import Proofs.Flock.Claims.ZK.FramePass
import Proofs.Flock.SessionLoop.Statement

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
  (Cfg Masker labelFrame bytesFrame roundFree LABEL_ZC K_SKIP N_INNER INNER_ROUNDS absorbed)
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
          List.drop_left' (FlockSoundness.Refine.Frame.header_length 3 1 1)]
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
          List.drop_left' (FlockSoundness.Refine.Frame.header_length 3 2 vs.length)]
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
    hon hp hb List.drop_zero
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
    simp only [stepAt, label_zc c m hon hd]; rfl
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
    ∀ (zs : List β) (v : Array α), (zs.foldl f v).size = v.size
  | [], _ => rfl
  | e :: zs, v => by rw [List.foldl_cons, foldl_set_size f hf zs, hf]

theorem foldl_set_miss {α : Type} [Inhabited α] (g : ℕ → α → α) (f : Array α → ℕ × α → Array α)
    (hf : ∀ v e, f v e = v.set! e.1 (g e.1 e.2)) :
    ∀ (zs : List (ℕ × α)) (v : Array α) (i : ℕ), (∀ e ∈ zs, e.1 ≠ i) → (zs.foldl f v)[i]! = v[i]!
  | [], _, _, _ => rfl
  | e :: zs, v, i, h => by
    rw [List.foldl_cons, foldl_set_miss g f hf zs _ i (fun e' he' => h e' (List.mem_cons_of_mem _ he')), hf]
    exact set!_ne v _ (h e List.mem_cons_self)

/-- **The fold that sets each pair's place, at a place pair `p` alone names**, holds `g` of pair `p`. -/
theorem foldl_set_at {α : Type} [Inhabited α] (g : ℕ → α → α) (f : Array α → ℕ × α → Array α)
    (hf : ∀ v e, f v e = v.set! e.1 (g e.1 e.2)) (zs : List (ℕ × α)) (v : Array α) {i p : ℕ}
    (hp : p < zs.length) (hi : zs[p].1 = i) (hu : ∀ q (hq : q < zs.length), zs[q].1 = i → q = p) (hv : i < v.size) :
    (zs.foldl f v)[i]! = g i zs[p].2 := by
  have hZ : zs = zs.take p ++ zs[p] :: zs.drop (p + 1) := by
    rw [← List.drop_eq_getElem_cons hp, List.take_append_drop]
  generalize hZp : zs[p] = zp at hi ⊢
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

/-! ## The PIOP's masked values, read back off its rounds -/

theorem nAbs_fvals : ∀ fs : List Frame, nAbs fs = (fvals fs).length
  | [] => rfl
  | .dom _ :: fs | .label _ :: fs | .bytes _ :: fs | .sq _ _ :: fs => by
    simp only [nAbs, fvals]; exact nAbs_fvals fs
  | .one _ :: fs => by simp only [nAbs, fvals, List.length_cons]; rw [nAbs_fvals fs]
  | .vec vs :: fs => by simp only [nAbs, fvals, List.length_append]; rw [nAbs_fvals fs]; omega

/-- **The PIOP's round `jj`'s frames** at layout `lay`, the masked values `mk` and the binding `digest cap root`: the
zerocheck's binding, `r_outer`'s round (none), `round1`'s values, each residual round's two, the finals with the
lincheck's label, `β`'s round (none), each lincheck round's two, and `z_partial`'s values. -/
def piopFr (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (jj : ℕ) :
    List Frame :=
  if jj = 0 then FlockSoundness.Refine.Frame.label "flock-r1cs-v0".toUTF8 :: .bytes digest :: .bytes cap ::
    .label LABEL_ZC.toUTF8 :: .label _root_.Flock.Zk.LABEL_PADS.toUTF8 :: .bytes root :: []
  else if jj = 1 then []
  else if jj = 2 then [.vec ((List.range 64).map fun i => mk (lay.r1ab i)), .vec ((List.range 64).map fun i => mk (lay.r1c i))]
  else if jj < 3 + (m - 6) then [.one (mk (lay.ml (jj - 3) 0)), .one (mk (lay.ml (jj - 3) 1))]
  else if jj = 3 + (m - 6) then [.one (mk lay.fa), .one (mk lay.fb), .label "flock-lincheck-v0".toUTF8]
  else if jj = 4 + (m - 6) then []
  else if jj < 5 + (m - 6) + (kLog - 6) then
    [.one (mk (lay.lc (jj - (5 + (m - 6))) 0)), .one (mk (lay.lc (jj - (5 + (m - 6))) 1))]
  else [.vec ((List.range 64).map fun i => mk (lay.zp i))]

/-- The PIOP's round `jj`'s squeeze kind and count. -/
def piopK (jj : ℕ) : UInt8 := if jj ≤ 1 then 2 else 1

def piopN (m jj : ℕ) : ℕ := if jj = 0 then 6 else if jj = 1 then m - 13 else 1

/-- The PIOP's rounds. -/
def piopNR (m kLog : ℕ) : ℕ := 6 + (m - 6) + (kLog - 6)

/-- The values the PIOP's first `a` rounds carry. -/
def piopV (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (a : ℕ) :
    List F128 :=
  (List.range a).flatMap fun jj => fvals (piopFr m kLog digest cap root lay mk jj)

theorem piopV_succ (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (a : ℕ) : piopV m kLog digest cap root lay mk (a + 1) =
      piopV m kLog digest cap root lay mk a ++ fvals (piopFr m kLog digest cap root lay mk a) := by
  simp [piopV, List.range_succ, List.flatMap_append]

theorem piopV_mono (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    {a b : ℕ} (h : a ≤ b) :
    (piopV m kLog digest cap root lay mk a).length ≤ (piopV m kLog digest cap root lay mk b).length := by
  induction b with
  | zero =>
    obtain rfl : a = 0 := by omega
    exact le_rfl
  | succ b ih =>
    rcases Nat.eq_or_lt_of_le h with rfl | h'
    · exact le_rfl
    · rw [piopV_succ, List.length_append]
      exact Nat.le_trans (ih (by omega)) (Nat.le_add_right _ _)

theorem piopV_two (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) :
    piopV m kLog digest cap root lay mk 2 = [] := by
  show piopV m kLog digest cap root lay mk (0 + 1 + 1) = []
  rw [piopV_succ, piopV_succ]
  simp [piopV, piopFr, fvals]

theorem flatMap_range'_congr {β : Type} (f g : ℕ → List β) (s n : ℕ) (h : ∀ i < n, f (s + i) = g i) :
    (List.range' s n).flatMap f = (List.range n).flatMap g := by
  rw [List.range'_eq_map_range, List.flatMap_map]
  exact List.flatMap_congr fun i hi => h i (List.mem_range.mp hi)

/-- **The PIOP's values are the masked values at the absorbed order's pads**, through `z_partial`'s. -/
theorem piopV_all (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (hm : 14 ≤ m) (hk : 6 ≤ kLog) (hn : lay.nMlv = m - 6) (hl : lay.nLc = kLog - 6) :
    piopV m kLog digest cap root lay mk (piopNR m kLog) =
      (List.range (lay.fa + 2) ++ List.range' (lay.fa + 3) (2 * lay.nLc + 64)).map mk := by
  have hsplit : List.range (piopNR m kLog) = List.range' 0 1 ++ (List.range' 1 1 ++ (List.range' 2 1 ++
      (List.range' 3 (m - 6) ++ (List.range' (3 + (m - 6)) 1 ++ (List.range' (4 + (m - 6)) 1 ++
      (List.range' (5 + (m - 6)) (kLog - 6) ++ List.range' (5 + (m - 6) + (kLog - 6)) 1)))))) := by
    rw [range'_join' (5 + (m - 6)) (kLog - 6) _ _ rfl, range'_join' (4 + (m - 6)) 1 _ (5 + (m - 6)) (by omega),
      range'_join' (3 + (m - 6)) 1 _ (4 + (m - 6)) (by omega), range'_join' 3 (m - 6) _ (3 + (m - 6)) rfl,
      range'_join' 2 1 _ 3 rfl, range'_join' 1 1 _ 2 rfl, range'_join' 0 1 _ 1 rfl, List.range_eq_range']
    congr 1
    unfold piopNR
    omega
  have e := absorbed_piop lay []
  simp only [List.append_nil] at e
  rw [← e]
  unfold piopV
  rw [hsplit]
  simp only [List.flatMap_append]
  have h0 : (List.range' 0 1).flatMap (fun jj => fvals (piopFr m kLog digest cap root lay mk jj)) = [] := by
    simp [piopFr, fvals]
  have h1 : (List.range' 1 1).flatMap (fun jj => fvals (piopFr m kLog digest cap root lay mk jj)) = [] := by
    simp [piopFr, fvals]
  have h2 : (List.range' 2 1).flatMap (fun jj => fvals (piopFr m kLog digest cap root lay mk jj)) =
      (List.range' 0 64).map (fun i => mk (lay.r1ab i)) ++ (List.range' 0 64).map (fun i => mk (lay.r1c i)) := by
    simp [piopFr, fvals, List.range_eq_range']
  have h3 : (List.range' 3 (m - 6)).flatMap (fun jj => fvals (piopFr m kLog digest cap root lay mk jj)) =
      (List.range' 0 lay.nMlv).flatMap fun a => [mk (lay.ml a 0), mk (lay.ml a 1)] := by
    rw [flatMap_range'_congr _ (fun a => [mk (lay.ml a 0), mk (lay.ml a 1)]) 3 (m - 6), hn, List.range_eq_range']
    intro i hi
    rw [piopFr, if_neg (show ¬ 3 + i = 0 by omega), if_neg (show ¬ 3 + i = 1 by omega),
      if_neg (show ¬ 3 + i = 2 by omega), if_pos (show 3 + i < 3 + (m - 6) by omega)]
    simp [fvals]
  have h4 : (List.range' (3 + (m - 6)) 1).flatMap (fun jj => fvals (piopFr m kLog digest cap root lay mk jj)) =
      [mk lay.fa, mk lay.fb] := by
    simp only [List.range'_one, List.flatMap_cons, List.flatMap_nil, List.append_nil]
    rw [piopFr, if_neg (show ¬ 3 + (m - 6) = 0 by omega), if_neg (show ¬ 3 + (m - 6) = 1 by omega),
      if_neg (show ¬ 3 + (m - 6) = 2 by omega), if_neg (show ¬ 3 + (m - 6) < 3 + (m - 6) by omega), if_pos rfl]
    simp [fvals]
  have h5 : (List.range' (4 + (m - 6)) 1).flatMap (fun jj => fvals (piopFr m kLog digest cap root lay mk jj)) =
      [] := by
    simp only [List.range'_one, List.flatMap_cons, List.flatMap_nil, List.append_nil]
    rw [piopFr, if_neg (show ¬ 4 + (m - 6) = 0 by omega), if_neg (show ¬ 4 + (m - 6) = 1 by omega),
      if_neg (show ¬ 4 + (m - 6) = 2 by omega), if_neg (show ¬ 4 + (m - 6) < 3 + (m - 6) by omega),
      if_neg (show ¬ 4 + (m - 6) = 3 + (m - 6) by omega), if_pos rfl]
    rfl
  have h6 : (List.range' (5 + (m - 6)) (kLog - 6)).flatMap
      (fun jj => fvals (piopFr m kLog digest cap root lay mk jj)) =
      (List.range' 0 lay.nLc).flatMap fun a => [mk (lay.lc a 0), mk (lay.lc a 1)] := by
    rw [flatMap_range'_congr _ (fun a => [mk (lay.lc a 0), mk (lay.lc a 1)]) _ (kLog - 6), hl, List.range_eq_range']
    intro i hi
    rw [piopFr, if_neg (show ¬ 5 + (m - 6) + i = 0 by omega), if_neg (show ¬ 5 + (m - 6) + i = 1 by omega),
      if_neg (show ¬ 5 + (m - 6) + i = 2 by omega), if_neg (show ¬ 5 + (m - 6) + i < 3 + (m - 6) by omega),
      if_neg (show ¬ 5 + (m - 6) + i = 3 + (m - 6) by omega), if_neg (show ¬ 5 + (m - 6) + i = 4 + (m - 6) by omega),
      if_pos (show 5 + (m - 6) + i < 5 + (m - 6) + (kLog - 6) by omega)]
    simp [fvals]
  have h7 : (List.range' (5 + (m - 6) + (kLog - 6)) 1).flatMap
      (fun jj => fvals (piopFr m kLog digest cap root lay mk jj)) = (List.range' 0 64).map (fun i => mk (lay.zp i)) := by
    simp only [List.range'_one, List.flatMap_cons, List.flatMap_nil, List.append_nil]
    rw [piopFr, if_neg (show ¬ 5 + (m - 6) + (kLog - 6) = 0 by omega),
      if_neg (show ¬ 5 + (m - 6) + (kLog - 6) = 1 by omega), if_neg (show ¬ 5 + (m - 6) + (kLog - 6) = 2 by omega),
      if_neg (show ¬ 5 + (m - 6) + (kLog - 6) < 3 + (m - 6) by omega),
      if_neg (show ¬ 5 + (m - 6) + (kLog - 6) = 3 + (m - 6) by omega),
      if_neg (show ¬ 5 + (m - 6) + (kLog - 6) = 4 + (m - 6) by omega),
      if_neg (show ¬ 5 + (m - 6) + (kLog - 6) < 5 + (m - 6) + (kLog - 6) by omega)]
    simp [fvals, List.range_eq_range']
  rw [h0, h1, h2, h3, h4, h5, h6, h7]
  simp [List.map_flatMap, Function.comp_def]

theorem piopV_length (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (hm : 14 ≤ m) (hk : 6 ≤ kLog) (hn : lay.nMlv = m - 6) (hl : lay.nLc = kLog - 6) :
    (piopV m kLog digest cap root lay mk (piopNR m kLog)).length = lay.fa + 2 + (2 * lay.nLc + 64) := by
  rw [piopV_all m kLog digest cap root lay mk hm hk hn hl]
  simp

/-- The rep's state inside the algebraic phase at absorbed position `P`, past the zerocheck's outer coins. -/
structure Inv (c : Cfg) (P : ℕ) (s : Masker) : Prop where
  on : s.on = true
  inner : s.inner = none
  l0 : s.l0 = false
  owed : s.owed = none
  pos : s.pos = P
  coins : K_SKIP + (c.lay.nMlv - N_INNER) ≤ s.coins.size

theorem maskedOf_cons_ok {c : Cfg} {s s₁ s₂ : Masker} {L : List UInt8} {n : ℕ} {cs : Array F128} {out : ByteArray}
    {fr : Array ℕ} (rest : List (ByteArray × ℕ × Array F128))
    (h : roundFree c s (ByteArray.mk L.toArray) n = .ok (out, s₁, fr)) (ha : s₁.answer c cs = .ok s₂) :
    (Loop.maskedOf c s ((ByteArray.mk L.toArray, n, cs) :: rest)).toList =
      fr.toList.map (wd L) ++ (Loop.maskedOf c s₂ rest).toList := by
  simp only [Loop.maskedOf, h, ha, Array.toList_append, Array.toList_map]
  rfl

theorem maskedOf_cons_last {c : Cfg} {s s₁ : Masker} {L : List UInt8} {n : ℕ} {cs : Array F128} {out : ByteArray}
    {fr : Array ℕ} (rest : List (ByteArray × ℕ × Array F128))
    (h : roundFree c s (ByteArray.mk L.toArray) n = .ok (out, s₁, fr)) :
    ∃ tail, (Loop.maskedOf c s ((ByteArray.mk L.toArray, n, cs) :: rest)).toList = fr.toList.map (wd L) ++ tail := by
  simp only [Loop.maskedOf, h]
  cases s₁.answer c cs with
  | ok s₂ => exact ⟨_, by simp only [Array.toList_append, Array.toList_map]; rfl⟩
  | error e => exact ⟨[], by simp only [List.append_nil, Array.toList_map]; rfl⟩

/-- **A round inside the algebraic phase past the outer coins**: the frame frees its frames' values, takes its one coin,
and the state moves on by the frames' values. -/
theorem maskedOf_mid (c : Cfg) (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hN : c.nPiop ≤ c.order.size)
    {P : ℕ} {s : Masker} (hs : Inv c P s) (fs : List Frame) (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f)
    (hb : P + nAbs fs ≤ c.nPiop) {cs : Array F128} (hcs : cs.size = 1) (rest : List (ByteArray × ℕ × Array F128)) :
    ∃ s', Inv c (P + nAbs fs) s' ∧
      (Loop.maskedOf c s ((ByteArray.mk (encs fs ++ (Frame.sq 1 1).enc).toArray, 1, cs) :: rest)).toList =
        fvals fs ++ (Loop.maskedOf c s' rest).toList := by
  obtain ⟨out, fr, hr, hfr⟩ := roundFree_vals c s fs 1 1 hc hN hs.on hs.inner hm hp (.inr ⟨rfl, rfl⟩) (by decide)
    (by decide) (by rw [hs.pos]; exact hb)
  have ha := answer_ok (cfg := c) (s := { s with pos := s.pos + nAbs fs, owed := some 1 }) (cs := cs) (n := 1) rfl hcs
    hs.inner hs.l0 hs.on (fun e he h1 h2 => by
      have := hs.coins
      simp only at h2
      omega)
  refine ⟨{ ({ s with pos := s.pos + nAbs fs, owed := some 1 } : Masker) with
      owed := none, coins := s.coins ++ cs }, ⟨hs.on, hs.inner, hs.l0, rfl, by simp [hs.pos], ?_⟩, ?_⟩
  · simp only [Array.size_append]
    exact Nat.le_trans hs.coins (Nat.le_add_right _ _)
  · rw [maskedOf_cons_ok rest hr ha, hfr]

theorem maskedOf_end (c : Cfg) (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hN : c.nPiop ≤ c.order.size)
    {P : ℕ} {s : Masker} (hs : Inv c P s) (fs : List Frame) (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f)
    (hb : P + nAbs fs ≤ c.nPiop) (cs : Array F128) (rest : List (ByteArray × ℕ × Array F128)) :
    ∃ tail, (Loop.maskedOf c s ((ByteArray.mk (encs fs ++ (Frame.sq 1 1).enc).toArray, 1, cs) :: rest)).toList =
        fvals fs ++ tail := by
  obtain ⟨out, fr, hr, hfr⟩ := roundFree_vals c s fs 1 1 hc hN hs.on hs.inner hm hp (.inr ⟨rfl, rfl⟩) (by decide)
    (by decide) (by rw [hs.pos]; exact hb)
  obtain ⟨tail, ht⟩ := maskedOf_cons_last (cs := cs) rest hr
  exact ⟨tail, by rw [ht, hfr]⟩

theorem piopFr_frames (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    {jj : ℕ} (h2 : 2 ≤ jj) :
    (∀ f ∈ piopFr m kLog digest cap root lay mk jj, midFrame f) ∧
      (∀ f ∈ piopFr m kLog digest cap root lay mk jj, plainFrame f) := by
  have hl : plainLabel "flock-lincheck-v0".toUTF8 := by unfold plainLabel; decide
  have hs : "flock-lincheck-v0".toUTF8.size < 2 ^ 64 := by decide
  unfold piopFr
  rw [if_neg (show ¬ jj = 0 by omega), if_neg (show ¬ jj = 1 by omega)]
  split_ifs <;> simp [midFrame, plainFrame]
  all_goals exact ⟨by simpa using hs, by simpa using hl⟩

/-- **The masked values a rep's frame takes over the PIOP's rounds**: from the rep's start, at rounds whose bytes are the
PIOP's frames and squeezes (`piopFr`, the zerocheck's binding without the stream's domain) with coin counts `piopN`,
answered by coins of those counts with no zerocheck outer coin `1`, the frame takes the masked values at the absorbed
order's pads through `z_partial`'s, in order. -/
theorem maskedOf_piop (lay : _root_.Flock.Zk.PadLayout) (b : ByteArray) (ms : Option _root_.Flock.Firewall.Frame.MaskSlot)
    (m kLog : ℕ) (hm : 14 ≤ m) (hk : 6 ≤ kLog) (hm64 : m < 2 ^ 64) (hn : lay.nMlv = m - 6) (hl : lay.nLc = kLog - 6)
    (digest cap root : ByteArray) (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hroot : root.size < 2 ^ 64)
    (mk : ℕ → F128) (C : ℕ → Array F128) (rs : List (ByteArray × ℕ × Array F128))
    (hrs : ∀ jj < piopNR m kLog, rs[jj]? = some (ByteArray.mk (encs (piopFr m kLog digest cap root lay mk jj) ++
      (Frame.sq (piopK jj) (piopN m jj)).enc).toArray, piopN m jj, C jj))
    (hC : ∀ jj, jj + 1 < piopNR m kLog → (C jj).size = piopN m jj)
    (hout : ∀ e < m - 13, ((C 1)[e]! + 1).isZero = false) :
    ∃ tail, (Loop.maskedOf (Cfg.ofDraws lay (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b) ms) {} rs).toList =
      (List.range (lay.fa + 2) ++ List.range' (lay.fa + 3) (2 * lay.nLc + 64)).map mk ++ tail := by
  set c := Cfg.ofDraws lay (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b) ms with hcdef
  have hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size := ofDraws_order lay b ms
  have hN : c.nPiop ≤ c.order.size := Nat.sub_le _ _
  have hNP : c.nPiop = lay.fa + 2 + (2 * lay.nLc + 64) := by
    show (absorbed lay).size - _root_.Flock.Firewall.Frame.nLig lay = _
    rw [absorbed_size]
    omega
  have hlen := piopV_length m kLog digest cap root lay mk hm hk hn hl
  have hNR : 7 ≤ piopNR m kLog := by unfold piopNR; omega
  have hdrop : ∀ a < piopNR m kLog, rs.drop a = (ByteArray.mk (encs (piopFr m kLog digest cap root lay mk a) ++
      (Frame.sq (piopK a) (piopN m a)).enc).toArray, piopN m a, C a) :: rs.drop (a + 1) := by
    intro a ha
    have e := hrs a ha
    rw [List.getElem?_eq_some_iff] at e
    obtain ⟨hlt, e⟩ := e
    rw [List.drop_eq_getElem_cons hlt, e]
  -- the zerocheck's binding round and `r_outer`'s
  have hbase : ∃ s, Inv c 0 s ∧ (Loop.maskedOf c {} rs).toList = (Loop.maskedOf c s (rs.drop 2)).toList := by
    have e0 := hdrop 0 (by omega)
    rw [List.drop_zero] at e0
    have e1 := hdrop 1 (by omega)
    have hF0 : piopFr m kLog digest cap root lay mk 0 =
        [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap] ++ Frame.label LABEL_ZC.toUTF8 ::
          [.label _root_.Flock.Zk.LABEL_PADS.toUTF8, .bytes root] := rfl
    have hF1 : piopFr m kLog digest cap root lay mk 1 = [] := rfl
    have hK0 : piopK 0 = 2 := rfl
    have hK1 : piopK 1 = 2 := rfl
    have hN0 : piopN m 0 = 6 := rfl
    have hN1 : piopN m 1 = m - 13 := rfl
    rw [hF0, hK0, hN0] at e0
    rw [hF1, hK1, hN1] at e1
    have hpl : plainLabel _root_.Flock.Zk.LABEL_PADS.toUTF8 := by unfold plainLabel; decide
    have hr1 : plainLabel "flock-r1cs-v0".toUTF8 := by unfold plainLabel; decide
    obtain ⟨out₀, hr₀⟩ := roundFree_zcFree c {} [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap]
      [.label _root_.Flock.Zk.LABEL_PADS.toUTF8, .bytes root] 2 6 hc hN rfl rfl rfl
      (by
        intro f hf
        simp only [List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hf
        rcases hf with rfl | rfl | rfl | rfl | rfl | rfl
        · show "flock-r1cs-v0".toUTF8.size < 2 ^ 64; decide
        · exact hdg
        · exact hcap
        · show LABEL_ZC.toUTF8.size < 2 ^ 64; decide
        · show _root_.Flock.Zk.LABEL_PADS.toUTF8.size < 2 ^ 64; decide
        · exact hroot)
      (by
        intro f hf
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
        rcases hf with rfl | rfl | rfl
        · exact hr1
        · trivial
        · trivial)
      (by
        intro f hf
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
        rcases hf with rfl | rfl
        · exact hpl
        · trivial)
      (.inl rfl) (by decide) (by decide) (Nat.zero_le _) rfl rfl
    have hc0 : (C 0).size = 6 := hC 0 (by omega)
    have hc1 : (C 1).size = m - 13 := hC 1 (by omega)
    have ha₀ : Masker.answer c { ({} : Masker) with on := true, owed := some 6 } (C 0) =
        .ok { ({} : Masker) with on := true, coins := #[] ++ C 0 } :=
      answer_ok rfl hc0 rfl rfl rfl (fun e he h1 _ => by simp [K_SKIP] at h1; omega)
    obtain ⟨out₁, fr₁, hr₁, hfr₁⟩ := roundFree_vals c { ({} : Masker) with on := true, coins := #[] ++ C 0 }
      [] 2 (m - 13) hc hN rfl rfl (by simp) (by simp) (.inl rfl) (fun h => absurd h.1 (by omega)) (by omega)
      (by simp [nAbs])
    have ha₁ : Masker.answer c { ({ ({} : Masker) with on := true, coins := #[] ++ C 0 } : Masker) with
        pos := ({ ({} : Masker) with on := true, coins := #[] ++ C 0 } : Masker).pos + nAbs [],
        owed := some (m - 13) } (C 1) = .ok { ({} : Masker) with on := true, coins := #[] ++ C 0 ++ C 1 } :=
      answer_ok rfl hc1 rfl rfl rfl (fun e he _ _ => hout e he)
    have hcn : c.lay.nMlv = m - 6 := hn
    refine ⟨{ ({} : Masker) with on := true, coins := #[] ++ C 0 ++ C 1 }, ⟨rfl, rfl, rfl, rfl, rfl, ?_⟩, ?_⟩
    · show K_SKIP + (c.lay.nMlv - N_INNER) ≤ (#[] ++ C 0 ++ C 1).size
      rw [Array.size_append, Array.size_append, Array.size_empty, hc0, hc1, hcn]
      unfold K_SKIP N_INNER
      omega
    · conv_lhs => rw [e0]
      rw [maskedOf_cons_ok _ hr₀ ha₀, e1, maskedOf_cons_ok _ hr₁ ha₁, hfr₁]
      simp [fvals]
  -- the rounds from `round1` on
  have hwalk : ∀ a, 2 ≤ a → a + 1 ≤ piopNR m kLog → ∃ s,
      Inv c (piopV m kLog digest cap root lay mk a).length s ∧
      (Loop.maskedOf c {} rs).toList =
        piopV m kLog digest cap root lay mk a ++ (Loop.maskedOf c s (rs.drop a)).toList := by
    intro a
    induction a with
    | zero => intro h; omega
    | succ a ih =>
      intro h2 ha
      rcases Nat.eq_or_lt_of_le h2 with h | h
      · obtain rfl : a = 1 := by omega
        obtain ⟨s, hs, he⟩ := hbase
        refine ⟨s, ?_, ?_⟩
        · rw [piopV_two]; exact hs
        · rw [he, piopV_two]
          rfl
      · obtain ⟨s, hs, he⟩ := ih (by omega) (by omega)
        obtain ⟨hmf, hpf⟩ := piopFr_frames m kLog digest cap root lay mk (jj := a) (by omega)
        have hK : piopK a = 1 := by unfold piopK; rw [if_neg (by omega)]
        have hNa : piopN m a = 1 := by unfold piopN; rw [if_neg (by omega), if_neg (by omega)]
        have hb : (piopV m kLog digest cap root lay mk a).length + nAbs (piopFr m kLog digest cap root lay mk a) ≤
            c.nPiop := by
          rw [nAbs_fvals, ← List.length_append, ← piopV_succ, hNP, ← hlen]
          exact piopV_mono m kLog digest cap root lay mk (by omega)
        have e := hdrop a (by omega)
        rw [hK, hNa] at e
        obtain ⟨s', hs', he'⟩ := maskedOf_mid c hc hN hs _ hmf hpf hb (by rw [← hNa]; exact hC a (by omega))
          (rs.drop (a + 1))
        refine ⟨s', ?_, ?_⟩
        · rw [piopV_succ, List.length_append, ← nAbs_fvals]; exact hs'
        · rw [he, e, he', piopV_succ, List.append_assoc]
  obtain ⟨s, hs, he⟩ := hwalk (piopNR m kLog - 1) (by omega) (by omega)
  have a1 : piopNR m kLog - 1 + 1 = piopNR m kLog := by omega
  obtain ⟨hmf, hpf⟩ := piopFr_frames m kLog digest cap root lay mk (jj := piopNR m kLog - 1) (by omega)
  have hK : piopK (piopNR m kLog - 1) = 1 := by unfold piopK; rw [if_neg (by omega)]
  have hNa : piopN m (piopNR m kLog - 1) = 1 := by unfold piopN; rw [if_neg (by omega), if_neg (by omega)]
  have hb : (piopV m kLog digest cap root lay mk (piopNR m kLog - 1)).length +
      nAbs (piopFr m kLog digest cap root lay mk (piopNR m kLog - 1)) ≤ c.nPiop :=
    le_of_eq (by rw [nAbs_fvals, ← List.length_append, ← piopV_succ, a1, hNP, ← hlen])
  have e := hdrop (piopNR m kLog - 1) (by omega)
  rw [hK, hNa] at e
  obtain ⟨tail, ht⟩ := maskedOf_end c hc hN hs _ hmf hpf hb (C (piopNR m kLog - 1)) (rs.drop (piopNR m kLog - 1 + 1))
  refine ⟨tail, ?_⟩
  rw [he, e, ht, ← List.append_assoc, ← piopV_succ, a1, piopV_all m kLog digest cap root lay mk hm hk hn hl]

/-- **The value a rep's frame took at pad `i`** (`valsOf`) below `z_partial`'s end, but `final_c`'s: the masked value
at `i`'s absorbed position plus pad `i`. -/
theorem valsOf_at (lay : _root_.Flock.Zk.PadLayout) (b : ByteArray) (ms : Option _root_.Flock.Firewall.Frame.MaskSlot)
    (M : Array F128) (fc : F128) {i : ℕ} (hi : i < lay.zp 64) (hne : i ≠ lay.fc)
    (hM : lay.fa + 2 + (2 * lay.nLc + 64) ≤ M.size) :
    (Loop.valsOf (Cfg.ofDraws lay (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b) ms) M fc)[i]! =
      M[absPos lay i]! + (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h[i]! := by
  have hcl : (Cfg.ofDraws lay (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b) ms).lay = lay := rfl
  have hco : (Cfg.ofDraws lay (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b) ms).order = absorbed lay := rfl
  have hch : (Cfg.ofDraws lay (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b) ms).h =
      (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h := rfl
  have hfc : lay.fc = lay.fa + 2 := rfl
  have hzp : lay.zp 64 = lay.fa + 3 + 2 * lay.nLc + 64 := by
    show lay.fc + 1 + 2 * lay.nLc + 0 + 64 = _
    omega
  have hsz : (absorbed lay).size = lay.fa + 2 + (2 * lay.nLc + 64 + _root_.Flock.Firewall.Frame.nLig lay) :=
    absorbed_size lay
  have hp : absPos lay i < (absorbed lay).size := by unfold absPos; split_ifs <;> omega
  have hpM : absPos lay i < M.size := by unfold absPos; split_ifs <;> omega
  have hord : (absorbed lay)[absPos lay i]! = i := by
    rw [absorbed_getElem! lay hp]
    unfold absPos
    split_ifs <;> omega
  have hih : i < (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h.size := by
    have := ofDraws_order lay b ms (absPos lay i) hp
    rwa [hco, hord, hch] at this
  have hzl : ((absorbed lay).zip M).toList.length = min (absorbed lay).size M.size := by simp
  have hpz : absPos lay i < ((absorbed lay).zip M).toList.length := by rw [hzl]; omega
  have hz : ∀ q (hq : q < ((absorbed lay).zip M).toList.length),
      ((absorbed lay).zip M).toList[q] = ((absorbed lay)[q]!, M[q]!) := by
    intro q hq
    rw [hzl] at hq
    rw [Array.getElem_toList, Array.getElem_zip, getElem!_pos (absorbed lay) q (by omega), getElem!_pos M q (by omega)]
  unfold Loop.valsOf
  simp only [hcl, hco, hch]
  rw [set!_ne _ _ (Ne.symm hne), ← Array.foldl_toList]
  refine (foldl_set_at (fun i m => m + (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h.getD i 0) _
    (by intro v e; cases e; rfl) _ _ hpz ?_ ?_ ?_).trans ?_
  · rw [hz _ hpz]; exact hord
  · intro q hq he
    rw [hz _ hq] at he
    have hq' : q < (absorbed lay).size := by rw [hzl] at hq; omega
    dsimp only at he
    rw [absorbed_getElem! lay hq'] at he
    unfold absPos
    split_ifs at he ⊢ <;> omega
  · rw [Array.size_replicate]; exact hi
  · rw [hz _ hpz]
    dsimp only
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hih, Option.getD_some,
      getElem!_pos (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h i hih]

end Claims.ZKServer
