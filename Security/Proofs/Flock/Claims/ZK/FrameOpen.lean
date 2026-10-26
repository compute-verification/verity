import Proofs.Flock.Claims.ZK.FrameRounds

/-!
# Level 0's masked rounds: the frame's release of the zero-pad rounds is the rounds at the pads

The pads `h` enter the masked Ligerito (`FlockComplete.ligeritoPZk`) only at level 0, and only additively: its
out-of-domain value `y + h[lgOod 0]`, its first `k_0` lane messages `maskMsg` (`lgMsg j 0 … 3`) and the pair's sums
`e_i + ⟨h[lgE i 0], h[lgE i 1]⟩`. The firewall's opening (`openLive`) runs the same code at zero pads, so each of its
masked rounds is the rep's round at `h` with every masked value's pad added back (characteristic 2), and the frame,
which adds the rep's pads at those values' absorbed positions, releases it as the round at `h`:

* **The rounds** (`oodFr`, `pairFr`): the out-of-domain round's value and grinding nonce, and a round of two `F256`
  values (a lane message, or the pair's sums), as `absorbF`, `pow` and `absorbK` frame them.
* **Value by value** (`oodFr_unmask`, `pairFr_unmask`): where the frame's pad at each value's position is the one the
  rep's round carries, the round at the pads with each pad added back is the round at zero pads (`add_pad_pad`).
* **The frame's pads at level 0** (`l0_pad`, `l0_padAt`): at a masked level 0, each of its pads' absorbed positions
  lies past the algebraic phase's, and the frame's pad there is the draws' (`padAt_ofDraws_absPos`). So a stopped rep
  masks them as a live one does (`maskFrames_ge`): `maskVal` zeroes only the algebraic phase's values.
* **A level-0 round** (`L0Round`, `l0Round_ood`, `l0Round_msg`, `l0Round_e`): the zero-pad round, the round at the
  rep's pads, and the facts the frame's round reads; through the frame's masks the zero-pad round is the round at the
  pads (`L0Round.maskFrames_eq`, by `maskFrames_unmask`).
* **At the release** (`LeanRelease.release_l0`, `roundHonest_l0`, `roundFrames_l0`): where the frame takes the coins
  before such a round at level 0's masked phase and position, it releases the zero-pad round as the round at the pads,
  for a stopped rep or not. That is `RoundHonest` at a live round and `RoundFrames` at a stand-in, the two forms
  `liveRoundEq_of_honest` and `fillRoundEq_of_frames` read, at each of level 0's masked rounds.
-/

namespace Claims.ZKServer

open _root_.Flock (F128 F256)
open _root_.Flock.Firewall.Frame (Cfg Masker absorbed nLig Draws MaskSlot)
open FlockSoundness.Refine (Frame encs)

/-! ## Level 0's masked rounds, value by value -/

/-- **Level 0's out-of-domain round**: its value `y` (`absorbF`), then the grinding nonce's bytes `b` (`powP`). -/
def oodFr (y : F128) (b : ByteArray) : List Frame := [.one y, .bytes b]

/-- **A round of two `F256` values** as `absorbK` frames them: a lane round's sumcheck message (`sendMsg`), or the
pair's sums `e_0`, `e_1`. -/
def pairFr (a b : F256) : List Frame := [.vec [a.c0, a.c1], .vec [b.c0, b.c1]]

theorem f128_add_zero (v : F128) : v + 0 = v := by
  obtain ⟨a, b⟩ := v
  show F128.mk (a ^^^ 0) (b ^^^ 0) = F128.mk a b
  rw [UInt64.xor_zero, UInt64.xor_zero]

/-- Zero pads: every index of `Array.replicate n 0` reads `0`. -/
theorem zeros_get! (n q : ℕ) : (Array.replicate n (0 : F128))[q]! = 0 := by
  by_cases hq : q < n
  · rw [getElem!_pos (Array.replicate n (0 : F128)) q (by simpa using hq)]
    simp
  · rw [getElem!_neg (Array.replicate n (0 : F128)) q (by simpa using hq)]
    rfl

/-- **The out-of-domain round, unmasked**: with the frame's pad at the value's position the round's pad `x`, the round
at the pads with its pad added back is the round at zero pads. -/
theorem oodFr_unmask (c : Cfg) (p : ℕ) (y x z : F128) (b : ByteArray) (hx : padAt c p = x) (hz : z = 0) :
    unmaskFrames c p (oodFr (y + x) b) = oodFr (y + z) b := by
  show [Frame.one (y + x + padAt c p), .bytes b] = [Frame.one (y + z), .bytes b]
  rw [hx, hz]
  simp only [add_pad_pad, f128_add_zero]

/-- **A round of two `F256` values, unmasked**: with the frame's pads at its four values' positions `p, …, p + 3` the
round's pads `x_0, …, x_3`, the round at the pads with each pad added back is the round at zero pads. -/
theorem pairFr_unmask (c : Cfg) (p : ℕ) (a b : F256) (x0 x1 x2 x3 z0 z1 z2 z3 : F128)
    (h0 : padAt c p = x0) (h1 : padAt c (p + 1) = x1) (h2 : padAt c (p + 2) = x2) (h3 : padAt c (p + 3) = x3)
    (g0 : z0 = 0) (g1 : z1 = 0) (g2 : z2 = 0) (g3 : z3 = 0) :
    unmaskFrames c p (pairFr (a + ⟨x0, x1⟩) (b + ⟨x2, x3⟩)) = pairFr (a + ⟨z0, z1⟩) (b + ⟨z2, z3⟩) := by
  show [Frame.vec [a.c0 + x0 + padAt c p, a.c1 + x1 + padAt c (p + 1)],
      .vec [b.c0 + x2 + padAt c (p + 2), b.c1 + x3 + padAt c (p + 3)]] =
    [Frame.vec [a.c0 + z0, a.c1 + z1], .vec [b.c0 + z2, b.c1 + z3]]
  rw [h0, h1, h2, h3, g0, g1, g2, g3]
  simp only [add_pad_pad, f128_add_zero]

/-! ## The frame's pads at level 0 -/

theorem maskVals_ge (c : Cfg) (st : Bool) : ∀ (p : ℕ) (vs : List F128), c.nPiop ≤ p →
    maskVals (maskVal c st) p vs = maskVals (maskVal c false) p vs
  | _, [], _ => rfl
  | p, v :: vs, hp => by
    have hv : maskVal c st p v = maskVal c false p v := by
      unfold maskVal
      rw [show decide (p < c.nPiop) = false from decide_eq_false (by omega)]
      simp
    simp only [maskVals, hv, maskVals_ge c st (p + 1) vs (by omega)]

/-- **Past the algebraic phase a stopped rep masks as a live one**: `maskVal` zeroes only the algebraic phase's
values. -/
theorem maskFrames_ge (c : Cfg) (st : Bool) : ∀ (p : ℕ) (F : List Frame), c.nPiop ≤ p →
    maskFrames (maskVal c st) p F = maskFrames (maskVal c false) p F
  | _, [], _ => rfl
  | p, f :: F, hp => by
    cases f with
    | dom b => exact maskFrames_ge c st p F hp
    | label b => exact congrArg (List.cons _) (maskFrames_ge c st p F hp)
    | bytes b => exact congrArg (List.cons _) (maskFrames_ge c st p F hp)
    | one v =>
      have hv : maskVal c st p v = maskVal c false p v := by
        unfold maskVal
        rw [show decide (p < c.nPiop) = false from decide_eq_false (by omega)]
        simp
      simp only [maskFrames, hv]
      exact congrArg (List.cons _) (maskFrames_ge c st (p + 1) F (by omega))
    | vec vs =>
      simp only [maskFrames, maskVals_ge c st p vs hp]
      exact congrArg (List.cons _) (maskFrames_ge c st (p + vs.length) F (by omega))
    | sq k n => exact congrArg (List.cons _) (maskFrames_ge c st p F hp)

theorem l0_ge (l : _root_.Flock.Zk.PadLayout) {q : ℕ} (h : l.lgOod 0 ≤ q) : l.fa + 3 ≤ q := by
  simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc,
    _root_.Flock.Zk.PadLayout.fc] at h
  omega

theorem absPos_add (l : _root_.Flock.Zk.PadLayout) {q : ℕ} (hq : l.fa + 2 ≤ q) (t : ℕ) :
    absPos l (q + t) = absPos l q + t := by
  rw [absPos_ge l (j := q + t) (by omega), absPos_ge l hq]
  omega

/-- **Level 0's pads in the frame**: at a masked level 0 (`k_0 > 0`), each of its pads `q` (from `lgOod 0`, below
`h_ab`) sits at an absorbed position past the algebraic phase's, where the frame's pad is the draws' pad `q`. -/
theorem l0_pad (l : _root_.Flock.Zk.PadLayout) (d : Draws) (mk : Option MaskSlot) (hk : 0 < l.k0) {q : ℕ}
    (h1 : l.lgOod 0 ≤ q) (h2 : q < l.hab) :
    (Cfg.ofDraws l d mk).nPiop ≤ absPos l q ∧ padAt (Cfg.ofDraws l d mk) (absPos l q) = d.h[q]! := by
  have hq := l0_ge l h1
  have hlig : nLig l = l.nOod0 + 4 * l.k0 + 4 := by
    unfold nLig
    rw [if_pos hk]
  refine ⟨?_, (padAt_ofDraws_absPos l d mk (j := q) ?_ ?_).2⟩
  · show (absorbed l).size - nLig l ≤ absPos l q
    rw [absorbed_size, absPos_ge l (j := q) (by omega)]
    simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc,
      _root_.Flock.Zk.PadLayout.fc] at h1
    omega
  · show q ≠ l.fa + 2
    omega
  · rw [hlig]
    simp only [_root_.Flock.Zk.PadLayout.hab, _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc,
      _root_.Flock.Zk.PadLayout.fc] at h2
    omega

/-- `l0_pad` at the positions after pad `q`'s. -/
theorem l0_padAt (l : _root_.Flock.Zk.PadLayout) (d : Draws) (mk : Option MaskSlot) (hk : 0 < l.k0) {q : ℕ}
    (h1 : l.lgOod 0 ≤ q) {t : ℕ} (h2 : q + t < l.hab) :
    padAt (Cfg.ofDraws l d mk) (absPos l q + t) = d.h[q + t]! := by
  rw [← absPos_add l (q := q) (by have := l0_ge l h1; omega) t]
  exact (l0_pad l d mk hk (by omega) h2).2

/-! ## A level-0 round -/

/-- **A level-0 masked round** at the frame's configuration `c`, from absorbed position `p`: the round `H` at the
rep's pads and the round `G` at zero pads, `G` being `H` with each of its `n` values' pads added back, its frames none
that opens or closes a phase, squeezes coins or carries a domain, and its values past the algebraic phase. -/
structure L0Round (c : Cfg) (p n : ℕ) (G H : List Frame) : Prop where
  unmask : unmaskFrames c p H = G
  mid : ∀ f ∈ H, midFrame f
  plain : ∀ f ∈ H, plainFrame f
  nd : ∀ f ∈ H, noDom f
  abs : nAbs H = n
  piop : c.nPiop ≤ p

/-- **Through the frame's masks the zero-pad round is the round at the pads**, for a stopped rep or not
(`maskFrames_unmask`, characteristic 2). -/
theorem L0Round.maskFrames_eq {c : Cfg} {p n : ℕ} {G H : List Frame} (R : L0Round c p n G H) (st : Bool) :
    maskFrames (maskVal c st) p G = H := by
  rw [← R.unmask, maskFrames_ge c st p _ R.piop, maskFrames_unmask, dropDoms_noDom H R.nd]

theorem L0Round.mid_G {c : Cfg} {p n : ℕ} {G H : List Frame} (R : L0Round c p n G H) : ∀ f ∈ G, midFrame f := by
  rw [← R.unmask]
  exact unmask_mid c p H R.mid

theorem L0Round.plain_G {c : Cfg} {p n : ℕ} {G H : List Frame} (R : L0Round c p n G H) :
    ∀ f ∈ G, plainFrame f := by
  rw [← R.unmask]
  exact unmask_plain c p H R.plain

theorem L0Round.abs_G {c : Cfg} {p n : ℕ} {G H : List Frame} (R : L0Round c p n G H) : nAbs G = n := by
  rw [← R.unmask, nAbs_unmask, R.abs]

/-- **Level 0's out-of-domain round** (`y` at its pad `lgOod 0`, then the nonce's bytes `b`) at a masked level 0. -/
theorem l0Round_ood (l : _root_.Flock.Zk.PadLayout) (d : Draws) (mk : Option MaskSlot) (hk : 0 < l.k0) (y : F128)
    (z : Array F128) (hz : z[l.lgOod 0]! = 0) (b : ByteArray) (hb : b.size < 2 ^ 64) :
    L0Round (Cfg.ofDraws l d mk) (absPos l (l.lgOod 0)) 1 (oodFr (y + z[l.lgOod 0]!) b)
      (oodFr (y + d.h[l.lgOod 0]!) b) := by
  have hlt : l.lgOod 0 < l.hab := by
    simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.hab]
    omega
  obtain ⟨hP, hpad⟩ := l0_pad l d mk hk (le_refl _) hlt
  refine ⟨oodFr_unmask _ _ _ _ _ _ hpad hz, ?_, ?_, ?_, rfl, hP⟩
  · intro f hf
    simp only [oodFr, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl
    · trivial
    · exact hb
  · intro f hf
    simp only [oodFr, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl <;> trivial
  · intro f hf
    simp only [oodFr, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl <;> trivial

theorem pairFr_mid (a b : F256) : ∀ f ∈ pairFr a b, midFrame f := by
  intro f hf
  simp only [pairFr, List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl <;> simp [midFrame]

theorem pairFr_plain (a b : F256) : ∀ f ∈ pairFr a b, plainFrame f := by
  intro f hf
  simp only [pairFr, List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl <;> trivial

theorem pairFr_nd (a b : F256) : ∀ f ∈ pairFr a b, noDom f := by
  intro f hf
  simp only [pairFr, List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl <;> trivial

/-- **Lane round `j`'s message** (`maskMsg`, at the pads `lgMsg j 0 … 3`) at a masked level 0, for `j < k_0`. -/
theorem l0Round_msg (l : _root_.Flock.Zk.PadLayout) (d : Draws) (mk : Option MaskSlot) (hk : 0 < l.k0) {j : ℕ}
    (hj : j < l.k0) (m : F256 × F256) (z : Array F128) (hz : ∀ t < 4, z[l.lgMsg j t]! = 0) :
    L0Round (Cfg.ofDraws l d mk) (absPos l (l.lgMsg j 0)) 4
      (pairFr (FlockComplete.maskMsg l z j m).1 (FlockComplete.maskMsg l z j m).2)
      (pairFr (FlockComplete.maskMsg l d.h j m).1 (FlockComplete.maskMsg l d.h j m).2) := by
  have h1 : l.lgOod 0 ≤ l.lgMsg j 0 := by
    simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.lgMsg]
    omega
  have hq : ∀ t < 4, l.lgMsg j 0 + t = l.lgMsg j t := by
    intro t _
    simp only [_root_.Flock.Zk.PadLayout.lgMsg]
    omega
  have hpad : ∀ t < 4, padAt (Cfg.ofDraws l d mk) (absPos l (l.lgMsg j 0) + t) = d.h[l.lgMsg j t]! := by
    intro t ht
    rw [l0_padAt l d mk hk h1 (t := t) (by
      simp only [_root_.Flock.Zk.PadLayout.lgMsg, _root_.Flock.Zk.PadLayout.hab]
      have : 4 * j + 4 ≤ 4 * l.k0 := by omega
      omega), hq t ht]
  have hP := (l0_pad l d mk hk (le_refl _) (q := l.lgOod 0) (by
    simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.hab]
    omega)).1
  refine ⟨pairFr_unmask (Cfg.ofDraws l d mk) (absPos l (l.lgMsg j 0)) m.1 m.2 _ _ _ _ _ _ _ _ (hpad 0 (by omega)) (hpad 1 (by omega)) (hpad 2 (by omega))
    (hpad 3 (by omega)) (hz 0 (by omega)) (hz 1 (by omega)) (hz 2 (by omega)) (hz 3 (by omega)),
    pairFr_mid _ _, pairFr_plain _ _, pairFr_nd _ _, rfl, ?_⟩
  have hmono : absPos l (l.lgOod 0) ≤ absPos l (l.lgMsg j 0) := by
    rw [absPos_ge l (j := l.lgOod 0) (by have := l0_ge l (le_refl (l.lgOod 0)); omega),
      absPos_ge l (j := l.lgMsg j 0) (by have := l0_ge l h1; omega)]
    omega
  omega

/-- **The pair's sums** `e_0`, `e_1` (at the pads `lgE 0 0`, `lgE 0 1`, `lgE 1 0`, `lgE 1 1`) at a masked level 0. -/
theorem l0Round_e (l : _root_.Flock.Zk.PadLayout) (d : Draws) (mk : Option MaskSlot) (hk : 0 < l.k0) (e0 e1 : F256)
    (z : Array F128) (hz : ∀ i < 2, ∀ c < 2, z[l.lgE i c]! = 0) :
    L0Round (Cfg.ofDraws l d mk) (absPos l (l.lgE 0 0)) 4
      (pairFr (e0 + ⟨z[l.lgE 0 0]!, z[l.lgE 0 1]!⟩) (e1 + ⟨z[l.lgE 1 0]!, z[l.lgE 1 1]!⟩))
      (pairFr (e0 + ⟨d.h[l.lgE 0 0]!, d.h[l.lgE 0 1]!⟩) (e1 + ⟨d.h[l.lgE 1 0]!, d.h[l.lgE 1 1]!⟩)) := by
  have h1 : l.lgOod 0 ≤ l.lgE 0 0 := by
    simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.lgE]
    omega
  have hhab : l.lgE 0 0 + 3 < l.hab := by
    simp only [_root_.Flock.Zk.PadLayout.lgE, _root_.Flock.Zk.PadLayout.hab]
    omega
  have hp : ∀ t i c, t < 4 → l.lgE 0 0 + t = l.lgE i c →
      padAt (Cfg.ofDraws l d mk) (absPos l (l.lgE 0 0) + t) = d.h[l.lgE i c]! := by
    intro t i c ht he
    rw [l0_padAt l d mk hk h1 (t := t) (by omega), he]
  have hP := (l0_pad l d mk hk (le_refl _) (q := l.lgOod 0) (by
    simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.hab]
    omega)).1
  refine ⟨pairFr_unmask (Cfg.ofDraws l d mk) (absPos l (l.lgE 0 0)) e0 e1 _ _ _ _ _ _ _ _
    (hp 0 0 0 (by omega) (by unfold _root_.Flock.Zk.PadLayout.lgE; omega))
    (hp 1 0 1 (by omega) (by unfold _root_.Flock.Zk.PadLayout.lgE; omega))
    (hp 2 1 0 (by omega) (by unfold _root_.Flock.Zk.PadLayout.lgE; omega))
    (hp 3 1 1 (by omega) (by unfold _root_.Flock.Zk.PadLayout.lgE; omega))
    (hz 0 (by omega) 0 (by omega)) (hz 0 (by omega) 1 (by omega)) (hz 1 (by omega) 0 (by omega))
    (hz 1 (by omega) 1 (by omega)),
    pairFr_mid _ _, pairFr_plain _ _, pairFr_nd _ _, rfl, ?_⟩
  have hmono : absPos l (l.lgOod 0) ≤ absPos l (l.lgE 0 0) := by
    rw [absPos_ge l (j := l.lgOod 0) (by have := l0_ge l (le_refl (l.lgOod 0)); omega),
      absPos_ge l (j := l.lgE 0 0) (by have := l0_ge l h1; omega)]
    omega
  omega

/-! ## At the release -/

namespace LeanRelease

/-- **The frame releases a zero-pad level-0 round as the round at the pads**: where it takes the coins before the
round at a state inside level 0's masked phase at the round's position, a stopped rep or not, the release of the
zero-pad round `G` and its squeeze is the round at the pads `H` and the squeeze, tagged `0`. -/
theorem release_l0 {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    {p n : ℕ} {G H : List Frame} (hR : L0Round (L.cfg x ρ h) p n G H) (k : UInt8) {m : Masker}
    (hA : L.answered x ρ h = .ok m)
    (hc : ∀ q, q < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[q]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hpos : m.pos = p)
    (hk : k = 2 ∨ (k = 1 ∧ L.coins x h.length = 1)) (h0 : ¬ (L.coins x h.length = 0 ∧ k = 2))
    (hn : L.coins x h.length < 2 ^ 64) (hb : p + n ≤ (L.cfg x ρ h).order.size) :
    L.release x ρ h st (encs G ++ (Frame.sq k (L.coins x h.length)).enc) =
      0 :: (encs H ++ (Frame.sq k (L.coins x h.length)).enc) := by
  rw [L.release_on x ρ h st G k hA hc hon hi hR.mid_G hR.plain_G hk h0 hn (by rw [hR.abs_G, hpos]; exact hb), hpos,
    hR.maskFrames_eq st]

/-- **A live level-0 round of an honest device** (`RoundHonest`): the device sends the zero-pad round, the session the
round at the pads. -/
theorem roundHonest_l0 {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    {p n : ℕ} {G H : List Frame} (hR : L0Round (L.cfg x ρ h) p n G H) (k : UInt8) {m : Masker}
    (hA : L.answered x ρ h = .ok m)
    (hc : ∀ q, q < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[q]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hpos : m.pos = p)
    (hk : k = 2 ∨ (k = 1 ∧ L.coins x h.length = 1)) (h0 : ¬ (L.coins x h.length = 0 ∧ k = 2))
    (hn : L.coins x h.length < 2 ^ 64) (hb : p + n ≤ (L.cfg x ρ h).order.size) :
    L.RoundHonest x ρ h (encs G ++ (Frame.sq k (L.coins x h.length)).enc)
      (0 :: (encs H ++ (Frame.sq k (L.coins x h.length)).enc)) := by
  subst hpos
  refine ⟨m, [], H, k, hA, hc, hon, hi, hR.mid, hR.plain, hR.nd, hk, h0, hn, by rw [hR.abs]; exact hb, .inl rfl, ?_,
    rfl⟩
  rw [List.nil_append, hR.unmask]

/-- **A level-0 round in frames** (`RoundFrames`), for a stopped rep or not: the stand-in the firewall schedules is
the zero-pad round, and through the frame's masks it is the session's round at the pads. -/
theorem roundFrames_l0 {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    {p n : ℕ} {G H : List Frame} (hR : L0Round (L.cfg x ρ h) p n G H) (k : UInt8) {m : Masker}
    (hA : L.answered x ρ h = .ok m)
    (hc : ∀ q, q < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[q]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hpos : m.pos = p)
    (hk : k = 2 ∨ (k = 1 ∧ L.coins x h.length = 1)) (h0 : ¬ (L.coins x h.length = 0 ∧ k = 2))
    (hn : L.coins x h.length < 2 ^ 64) (hb : p + n ≤ (L.cfg x ρ h).order.size) :
    L.RoundFrames x ρ h st (encs G ++ (Frame.sq k (L.coins x h.length)).enc)
      (0 :: (encs H ++ (Frame.sq k (L.coins x h.length)).enc)) := by
  subst hpos
  refine ⟨m, G, [], k, hA, hc, hon, hi, hR.mid_G, hR.plain_G, hk, h0, hn, by rw [hR.abs_G]; exact hb, .inl rfl, rfl,
    ?_⟩
  rw [List.nil_append, hR.maskFrames_eq st]

end LeanRelease

end Claims.ZKServer
