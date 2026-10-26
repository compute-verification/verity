import Proofs.Flock.Claims.ZK.FrameOpen

/-!
# Level 0's phase labels: the rounds that open and close its masked phase

Level 0's masked phase starts at the opening's label (`LABEL_OPEN`) and ends at level 0's closing label
(`LABEL_CLOSE`), and neither is a plain label, so neither round is a round inside or outside one phase
(`roundFree_on`, `roundFree_off`). The frame takes each as the masked Ligerito's code sends it:

* **The opening's label** (`label_open`, `roundFree_open`): after the algebraic phase's opening and before level 0
  closes, at a Ligerito rep outside a masked phase, the frame releases the frames before it as they came and the label,
  then the squeeze, and starts level 0's masked phase (`on`, `l0`) at its position.
* **The opening's binding** (`openBind`, `roundFree_openBind`, `LeanRelease.release_openBind`): `ligeritoPZk`'s first
  round (the basis label, `T`, level 0's cap and the opening's label, with `z`'s squeeze) leaves as it came.
* **Level 0's closing label** (`label_close`, `roundFree_close`, `LeanRelease.release_close`): inside level 0's phase
  with every masked value absorbed and its four `ρ` coins taken, the frame releases the label, then the frames after it
  as they came and the squeeze, and ends level 0's phase.
-/

namespace Claims.ZKServer

open _root_.Flock (F128)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker labelFrame roundFree LABEL_PCS LABEL_ZC RHO_COINS)
open FlockSoundness.Refine (Frame encs)

/-! ## The opening's label -/

/-- **The opening's label starts level 0's masked phase**: outside a masked phase, after the algebraic phase's opening
and before level 0 closes, at a Ligerito rep, the frame releases it as it came. -/
theorem label_open (c : Cfg) (s : Masker) (hon : s.on = false) (hd : s.done = true) (hlig : c.lig = true)
    (hl0 : s.doneL0 = false) :
    s.label c _root_.Flock.Zk.LABEL_OPEN.toUTF8 =
      .ok ({ s with on := true, l0 := true }, labelFrame _root_.Flock.Zk.LABEL_OPEN.toUTF8) := by
  simp [Masker.label, hon, hd, hlig, hl0] <;> rfl

/-- **The opening's round**: at frames `pre` outside a masked phase, then the opening's label and the squeeze, the frame
releases `pre` as they came, the label and the squeeze, and starts level 0's masked phase at the state's position. -/
theorem roundFree_open (c : Cfg) (s : Masker) (pre : List Frame) (k : UInt8) (n : ℕ) (hon : s.on = false)
    (hd : s.done = true) (hlig : c.lig = true) (hl0 : s.doneL0 = false) (hi : s.inner = none)
    (hm : ∀ f ∈ pre ++ [Frame.label _root_.Flock.Zk.LABEL_OPEN.toUTF8], midFrame f) (hpre : ∀ f ∈ pre, plainFrame f)
    (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) :
    ∃ out, roundFree c s
        (ByteArray.mk (encs (pre ++ [Frame.label _root_.Flock.Zk.LABEL_OPEN.toUTF8]) ++ (Frame.sq k n).enc).toArray)
          n =
        .ok (out, { s with on := true, l0 := true, owed := some n }, #[]) ∧
      out.data.toList = encs (maskFrames (fun _ v => v) 0 pre ++ [Frame.label _root_.Flock.Zk.LABEL_OPEN.toUTF8]) ++
        (Frame.sq k n).enc := by
  obtain ⟨r₁, hr₁, h1₁, h2₁⟩ := foldlM_stepAt_off c pre s ByteArray.empty #[] 0 0 hon hpre
  have hfr := foldlM_stepAt_off_fr c pre s ByteArray.empty #[] 0 r₁ hon hpre hr₁
  obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
  simp only at h1₁ h2₁ hfr
  subst h1₁
  have h3 : [Frame.label _root_.Flock.Zk.LABEL_OPEN.toUTF8].foldlM (stepAt c) (m, o₁, fr₁, q₁) =
      .ok ({ m with on := true, l0 := true }, o₁ ++ labelFrame _root_.Flock.Zk.LABEL_OPEN.toUTF8, fr₁,
        q₁ + 16 + (_root_.Flock.Zk.LABEL_OPEN.toUTF8.size + 15) / 16 * 16) := by
    rw [List.foldlM_cons]
    simp only [stepAt, label_open c m hon hd hlig hl0] <;> rfl
  have hf : (pre ++ [Frame.label _root_.Flock.Zk.LABEL_OPEN.toUTF8]).foldlM (stepAt c) (m, ByteArray.empty, #[], 0) =
      .ok ({ m with on := true, l0 := true }, o₁ ++ labelFrame _root_.Flock.Zk.LABEL_OPEN.toUTF8, fr₁,
        q₁ + 16 + (_root_.Flock.Zk.LABEL_OPEN.toUTF8.size + 15) / 16 * 16) := by
    rw [List.foldlM_append, hr₁]
    exact h3
  rw [roundFree_encs c m _ k n (by rw [hi]; simp) hm hk h0 hn, hf]
  subst hfr
  refine ⟨o₁ ++ labelFrame _root_.Flock.Zk.LABEL_OPEN.toUTF8 ++ header 4 k n, ?_, ?_⟩
  · simp only [bind, Except.bind, hi]
    rfl
  · rw [data_append', data_append', h2₁, sq_enc, labelFrame_data]
    simp [encs]

/-- **The opening's binding**: `ligeritoPZk`'s frames before `z`'s squeeze, the basis label, the table's claimed value
`T`, level 0's cap and the opening's label. -/
def openBind (T : F128) (cap : ByteArray) : List Frame :=
  [.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, .bytes cap, .label _root_.Flock.Zk.LABEL_OPEN.toUTF8]

/-- **The opening's binding round**: outside a masked phase, after the algebraic phase's opening and before level 0
closes, at a Ligerito rep, the frame releases the binding and `z`'s squeeze of `n` coins as they came, and starts level
0's masked phase at the state's position. -/
theorem roundFree_openBind (c : Cfg) (s : Masker) (T : F128) (cap : ByteArray) (n : ℕ) (hon : s.on = false)
    (hd : s.done = true) (hlig : c.lig = true) (hl0 : s.doneL0 = false) (hi : s.inner = none)
    (hcap : cap.size < 2 ^ 64) (hn0 : n ≠ 0) (hn : n < 2 ^ 64) :
    ∃ out, roundFree c s (ByteArray.mk (encs (openBind T cap) ++ (Frame.sq 2 n).enc).toArray) n =
        .ok (out, { s with on := true, l0 := true, owed := some n }, #[]) ∧
      out.data.toList = encs (openBind T cap) ++ (Frame.sq 2 n).enc := by
  obtain ⟨out, hr, ho⟩ := roundFree_open c s
    [.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, .bytes cap] 2 n hon hd hlig hl0 hi
    (by
      intro f hf
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl | rfl
      · show "flock-ligerito-basis-f256-split-v0".toUTF8.size < 2 ^ 64; decide
      · trivial
      · exact hcap
      · show _root_.Flock.Zk.LABEL_OPEN.toUTF8.size < 2 ^ 64; decide)
    (by
      intro f hf
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl
      · show plainLabel "flock-ligerito-basis-f256-split-v0".toUTF8; unfold plainLabel; decide
      · trivial
      · trivial)
    (.inl rfl) (fun h => hn0 h.1) hn
  rw [show openBind T cap = [.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, .bytes cap] ++
      [.label _root_.Flock.Zk.LABEL_OPEN.toUTF8] from rfl]
  refine ⟨out, hr, ?_⟩
  rw [ho]
  simp only [maskFrames]

/-! ## Level 0's closing label -/

/-- **Level 0's closing label ends its masked phase**: inside level 0's phase with every masked value of the layout
absorbed and its `ρ` coins taken, the frame releases it as it came. -/
theorem label_close (c : Cfg) (s : Masker) (hl0 : s.l0 = true) (hpos : s.pos = c.order.size)
    (hrho : s.rho.size = RHO_COINS) :
    s.label c _root_.Flock.Zk.LABEL_CLOSE.toUTF8 =
      .ok ({ s with l0 := false, on := false, doneL0 := true }, labelFrame _root_.Flock.Zk.LABEL_CLOSE.toUTF8) := by
  simp [Masker.label, hl0, hpos, hrho] <;> rfl

/-- **Level 0's closing round**: inside level 0's phase with every masked value absorbed and its `ρ` coins taken, at
the closing label, then frames `post` and the squeeze, the frame releases the label, `post` as they came and the
squeeze, and ends level 0's phase. -/
theorem roundFree_close (c : Cfg) (s : Masker) (post : List Frame) (k : UInt8) (n : ℕ) (hl0 : s.l0 = true)
    (hpos : s.pos = c.order.size) (hrho : s.rho.size = RHO_COINS) (hi : s.inner = none)
    (hm : ∀ f ∈ Frame.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: post, midFrame f)
    (hpost : ∀ f ∈ post, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) :
    ∃ out fr, roundFree c s
        (ByteArray.mk (encs (Frame.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: post) ++ (Frame.sq k n).enc).toArray)
          n =
        .ok (out, { s with l0 := false, on := false, doneL0 := true, owed := some n }, fr) ∧
      out.data.toList = encs (Frame.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: maskFrames (fun _ v => v) 0 post) ++
        (Frame.sq k n).enc := by
  have h1 : (Frame.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: post).foldlM (stepAt c) (s, ByteArray.empty, #[], 0) =
      post.foldlM (stepAt c) ({ s with l0 := false, on := false, doneL0 := true },
        ByteArray.empty ++ labelFrame _root_.Flock.Zk.LABEL_CLOSE.toUTF8, #[],
        0 + 16 + (_root_.Flock.Zk.LABEL_CLOSE.toUTF8.size + 15) / 16 * 16) := by
    rw [List.foldlM_cons]
    simp only [stepAt, label_close c s hl0 hpos hrho] <;> rfl
  obtain ⟨r, hr, h1r, h2r⟩ := foldlM_stepAt_off c post { s with l0 := false, on := false, doneL0 := true }
    (ByteArray.empty ++ labelFrame _root_.Flock.Zk.LABEL_CLOSE.toUTF8) #[]
    (0 + 16 + (_root_.Flock.Zk.LABEL_CLOSE.toUTF8.size + 15) / 16 * 16) 0 rfl hpost
  obtain ⟨m, o, fr, q⟩ := r
  simp only at h1r h2r
  subst h1r
  rw [roundFree_encs c s _ k n (by rw [hi]; simp) hm hk h0 hn, h1, hr]
  refine ⟨o ++ header 4 k n, fr, ?_, ?_⟩
  · simp only [bind, Except.bind, hi]
    rfl
  · rw [data_append', h2r, sq_enc, data_append', labelFrame_data]
    simp [encs]

/-! ## At the release -/

namespace LeanRelease

/-- **The release of the opening's binding**: where the frame takes the coins before it at a state outside a masked
phase, after the algebraic phase's opening and before level 0 closes, at a Ligerito rep, and the round squeezes `z`'s
`n` coins, the release is the binding and the squeeze as they came. -/
theorem release_openBind {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (T : F128) (cap : ByteArray) {m : Masker} (hA : L.answered x ρ h = .ok m) (hon : m.on = false)
    (hd : m.done = true) (hlig : (L.cfg x ρ h).lig = true) (hl0 : m.doneL0 = false) (hi : m.inner = none)
    (hcap : cap.size < 2 ^ 64) (hn0 : L.coins x h.length ≠ 0) (hn : L.coins x h.length < 2 ^ 64) :
    L.release x ρ h st (encs (openBind T cap) ++ (Frame.sq 2 (L.coins x h.length)).enc) =
      0 :: (encs (openBind T cap) ++ (Frame.sq 2 (L.coins x h.length)).enc) := by
  obtain ⟨out, hr, ho⟩ := roundFree_openBind (L.cfg x ρ h) { m with stopped := st } T cap _ hon hd hlig hl0 hi
    hcap hn0 hn
  rw [release_of_roundFree L x ρ h st _ hA rfl hr, ho]

/-- **The release of level 0's closing round**: where the frame takes the coins before it inside level 0's phase with
every masked value absorbed and its `ρ` coins taken, the release is the closing label, the frames `post` after it as
they came, then the squeeze. -/
theorem release_close {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (post : List Frame) (k : UInt8) {m : Masker} (hA : L.answered x ρ h = .ok m) (hl0 : m.l0 = true)
    (hpos : m.pos = (L.cfg x ρ h).order.size) (hrho : m.rho.size = RHO_COINS) (hi : m.inner = none)
    (hm : ∀ f ∈ Frame.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: post, midFrame f)
    (hpost : ∀ f ∈ post, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ L.coins x h.length = 1))
    (h0 : ¬ (L.coins x h.length = 0 ∧ k = 2)) (hn : L.coins x h.length < 2 ^ 64) :
    L.release x ρ h st (encs (Frame.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: post) ++
        (Frame.sq k (L.coins x h.length)).enc) =
      0 :: (encs (Frame.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: maskFrames (fun _ v => v) 0 post) ++
        (Frame.sq k (L.coins x h.length)).enc) := by
  obtain ⟨out, fr, hr, ho⟩ := roundFree_close (L.cfg x ρ h) { m with stopped := st } post k _ hl0 hpos hrho hi hm
    hpost hk h0 hn
  rw [release_of_roundFree L x ρ h st _ hA rfl hr, ho]

end LeanRelease

end Claims.ZKServer
