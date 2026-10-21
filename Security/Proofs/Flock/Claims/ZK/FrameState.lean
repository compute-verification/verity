import Proofs.Flock.Claims.ZK.FramePhase

/-!
# The frame's state across level 0's opening

The releases of level 0's rounds (`release_openBind`, `release_l0`, `release_close`) read the frame's state before each
round (`LeanRelease.answered`): its phase (`on`, `l0`), its position (`pos`) and its `ρ` coins. That state comes from the
transport (`LeanRelease.prog`) and the coins the stream answered (`coinsOf`). This file carries it from round to round:

* **The transport's trust** (`LeanRelease.ProgFollows`): within a rep, the frame's state after a round's release is the
  frame's own step's, but for `stopped`, which the release does not read.
* **A round's state** (`LeanRelease.answered_of_roundFree`): under it, where the frame takes a round and the transport's
  coin check passes the history with its release, the state after it is the frame's taking of the stream's coins at the
  step's state (`Masker.answer`).
* **The coins level 0 takes** (`answer_l0_mid`, `answer_rho_first`): before its last masked value the frame only marks
  the round answered; at it, `ρ_0`'s words are its first `ρ` coins (`answer_rho_independent` takes `ρ_1`'s).
* **Across the opening** (`LeanRelease.answered_openBind`, `answered_l0`, `answered_l0_last`, `answered_rho1`): the
  binding starts level 0's phase at the algebraic phase's position; each masked round (`L0Round`) moves the position by
  its values; the pair's sums reach the layout's last value and take `ρ_0`; `ρ_1`'s round takes `ρ_1`, after which the
  closing round's state (`release_close`) holds.
-/

namespace Claims.ZKServer

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree)
open FlockSoundness.Refine (Frame encs)

/-! ## The coins level 0 takes -/

/-- **Before level 0's last masked value the frame only marks the round answered**: at level 0's phase, outside the
inner proof, with a value of the layout still unabsorbed, it takes the round's coins and keeps nothing of them. -/
theorem answer_l0_mid (c : Cfg) (s : Masker) (cs : Array F128) (hn : s.owed = some cs.size) (hi : s.inner = none)
    (hl : s.l0 = true) (hp : s.pos ≠ c.order.size) :
    s.answer c cs = .ok { s with owed := none } := by
  unfold _root_.Flock.Firewall.Frame.Masker.answer
  simp [hn, hi, hl, hp] <;> rfl

/-- **At level 0's last masked value, `ρ_0`'s words are its first `ρ` coins**: with every value of the layout absorbed
and no `ρ` coin yet, the frame keeps the round's two coins as `ρ`. -/
theorem answer_rho_first (c : Cfg) (s : Masker) (a b : F128) (hn : s.owed = some 2) (hi : s.inner = none)
    (hl : s.l0 = true) (hp : s.pos = c.order.size) (hr : s.rho = #[]) :
    s.answer c #[a, b] = .ok { s with owed := none, rho := #[a, b] } := by
  unfold _root_.Flock.Firewall.Frame.Masker.answer
  simp [hn, hi, hl, hp, _root_.Flock.Firewall.Frame.rhoCoins, hr, _root_.Flock.Firewall.Frame.RHO_COINS] <;> rfl

namespace LeanRelease

/-! ## The transport's trust, and a round's state -/

/-- **The transport's frame state follows the frame's own steps** at `x`: where the frame takes the device round `v`
after `h` (`step`), of a stopped rep or not, and the next round is the same rep's, the frame's state after the history
with that round's release and the stream's answer `a` is the step's, but for `stopped`, which the release does not read
(`prog`). A named hypothesis, one of the transport's trusts beside `ProgReads`: the frame's state is what Rust's requests
to it make it (`LeanFrame`, `firewall.rs`), and within a rep the frame's state after a round is the one its round
returned. -/
def ProgFollows {W R : Type} (L : LeanRelease W R) (x : Msg) : Prop :=
  ∀ (ρ : R) (h : List (Msg × Msg)) (st : Bool) (v a : Msg) (out : ByteArray) (s' : Masker),
    L.step x ρ h st v = .ok (out, s') → L.rep x (h.length + 1) = L.rep x h.length →
      L.prog x (h ++ [(0 :: out.data.toList, a)]) =
        { s' with stopped := (L.prog x (h ++ [(0 :: out.data.toList, a)])).stopped }

/-- Within a rep the configuration is the rep's, after one more round as before it. -/
theorem cfg_snoc {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (e : Msg × Msg)
    (hrep : L.rep x (h.length + 1) = L.rep x h.length) : L.cfg x ρ (h ++ [e]) = L.cfg x ρ h := by
  unfold cfg
  rw [List.length_append, List.length_singleton, hrep]

/-- Where the frame takes the round, its step is the round's. -/
theorem step_of_roundFree {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (v : Msg) {m : Masker} {n : ℕ} {out : ByteArray} {s' : Masker} {fr : Array ℕ} (hA : L.answered x ρ h = .ok m)
    (hn : L.coins x h.length = n)
    (hr : roundFree (L.cfg x ρ h) { m with stopped := st } ⟨v.toArray⟩ n = .ok (out, s', fr)) :
    L.step x ρ h st v = .ok (out, s') := by
  unfold step
  rw [hA]
  show _root_.Flock.Firewall.Frame.round _ _ _ _ = _
  unfold _root_.Flock.Firewall.Frame.round
  rw [hn, hr]
  rfl

/-- **A round's state**: where the frame takes the device round `v` after `h` at the state `m` to the state `s'`, the next
round is the same rep's and the transport's coin check passes the history with the round's release and the answer `a`,
the frame's state after that history is its taking of the stream's coins at `s'`, at some `stopped`. -/
theorem answered_of_roundFree {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (v a : Msg) {m : Masker} {n : ℕ} {out : ByteArray} {s' : Masker} {fr : Array ℕ} (hF : L.ProgFollows x)
    (hA : L.answered x ρ h = .ok m) (hn : L.coins x h.length = n)
    (hr : roundFree (L.cfg x ρ h) { m with stopped := st } ⟨v.toArray⟩ n = .ok (out, s', fr))
    (hrep : L.rep x (h.length + 1) = L.rep x h.length)
    (ho : L.opens x ρ (h ++ [(0 :: out.data.toList, a)]) = true) :
    ∃ b, L.answered x ρ (h ++ [(0 :: out.data.toList, a)]) =
      { s' with stopped := b }.answer (L.cfg x ρ h) (L.coinsOf x (h ++ [(0 :: out.data.toList, a)])) := by
  have hs := L.step_of_roundFree x ρ h st v hA hn hr
  refine ⟨(L.prog x (h ++ [(0 :: out.data.toList, a)])).stopped, ?_⟩
  rw [L.answered_of_opens x ρ _ ho]
  unfold frameAnswer
  rw [show (h ++ [(0 :: out.data.toList, a)]).getLast? = some (0 :: out.data.toList, a) by simp]
  dsimp only
  rw [L.cfg_snoc x ρ h _ hrep, ← hF ρ h st v a out s' hs hrep]

/-! ## Across the opening -/

/-- **The binding starts level 0's phase**: where the frame takes the coins before the opening's binding at a state
outside a masked phase, after the algebraic phase's opening and before level 0 closes, short of the layout's last
value, at a Ligerito rep, and the next round is the rep's, the frame's state after the binding's release and the
stream's answer to `z`'s squeeze is level 0's phase at the same position. -/
theorem answered_openBind {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (T : F128) (cap : ByteArray) (a : Msg) {m : Masker} (hF : L.ProgFollows x) (hA : L.answered x ρ h = .ok m)
    (hon : m.on = false) (hd : m.done = true) (hlig : (L.cfg x ρ h).lig = true) (hl0 : m.doneL0 = false)
    (hi : m.inner = none) (hcap : cap.size < 2 ^ 64) (hn0 : L.coins x h.length ≠ 0)
    (hn : L.coins x h.length < 2 ^ 64) (hp : m.pos ≠ (L.cfg x ρ h).order.size)
    (hrep : L.rep x (h.length + 1) = L.rep x h.length)
    (ho : L.opens x ρ (h ++ [(0 :: (encs (openBind T cap) ++ (Frame.sq 2 (L.coins x h.length)).enc), a)]) = true)
    (hcs : (L.coinsOf x (h ++ [(0 :: (encs (openBind T cap) ++ (Frame.sq 2 (L.coins x h.length)).enc), a)])).size =
      L.coins x h.length) :
    ∃ b, L.answered x ρ (h ++ [(0 :: (encs (openBind T cap) ++ (Frame.sq 2 (L.coins x h.length)).enc), a)]) =
      .ok { m with stopped := b, on := true, l0 := true, owed := none } := by
  obtain ⟨out, fr, hr, hout⟩ := roundFree_openBind (L.cfg x ρ h) { m with stopped := st } T cap
    (L.coins x h.length) hon hd hlig hl0 hi hcap hn0 hn
  obtain ⟨b, hb⟩ := L.answered_of_roundFree x ρ h st _ a hF hA rfl hr hrep (by rw [hout]; exact ho)
  rw [hout] at hb
  exact ⟨b, hb.trans (answer_l0_mid _ _ _ (by rw [hcs]; try rfl) (by exact hi) (by rfl) (by exact hp))⟩

/-- **A masked round of level 0 moves the position by its values**: where the frame takes the coins before a level-0
round (`L0Round`) at level 0's phase and the round's position, its values short of the layout's last, and the next round
is the rep's, the frame's state after the round's release and the stream's answer is level 0's phase past the round's
values. -/
theorem answered_l0 {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    {p n : ℕ} {G H : List Frame} (hR : L0Round (L.cfg x ρ h) p n G H) (k : UInt8) (a : Msg) {m : Masker}
    (hF : L.ProgFollows x) (hA : L.answered x ρ h = .ok m)
    (hc : ∀ q, q < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[q]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hl0 : m.l0 = true) (hpos : m.pos = p)
    (hk : k = 2 ∨ (k = 1 ∧ L.coins x h.length = 1)) (h0 : ¬ (L.coins x h.length = 0 ∧ k = 2))
    (hn : L.coins x h.length < 2 ^ 64) (hb : p + n < (L.cfg x ρ h).order.size)
    (hrep : L.rep x (h.length + 1) = L.rep x h.length)
    (ho : L.opens x ρ (h ++ [(0 :: (encs H ++ (Frame.sq k (L.coins x h.length)).enc), a)]) = true)
    (hcs : (L.coinsOf x (h ++ [(0 :: (encs H ++ (Frame.sq k (L.coins x h.length)).enc), a)])).size =
      L.coins x h.length) :
    ∃ b, L.answered x ρ (h ++ [(0 :: (encs H ++ (Frame.sq k (L.coins x h.length)).enc), a)]) =
      .ok { m with stopped := b, pos := p + n, owed := none } := by
  subst hpos
  obtain rfl : nAbs G = n := hR.abs_G
  obtain ⟨out, fr, hr, hout⟩ := roundFree_on (L.cfg x ρ h) { m with stopped := st } G k (L.coins x h.length) hc hon
    hi hR.mid_G hR.plain_G hk h0 hn (by show m.pos + nAbs G ≤ _; omega)
  have hout' : out.data.toList = encs H ++ (Frame.sq k (L.coins x h.length)).enc := by
    rw [hout]
    show encs (maskFrames (maskVal (L.cfg x ρ h) st) m.pos G) ++ _ = _
    rw [hR.maskFrames_eq st]
  obtain ⟨b, hb'⟩ := L.answered_of_roundFree x ρ h st _ a hF hA rfl hr hrep (by rw [hout']; exact ho)
  rw [hout'] at hb'
  exact ⟨b, hb'.trans (answer_l0_mid _ _ _ (by rw [hcs]; try rfl) (by exact hi) (by exact hl0)
    (by show m.pos + nAbs G ≠ _; omega))⟩

/-- **The pair's sums reach the layout's last value and take `ρ_0`**: where the frame takes the coins before a level-0
round at level 0's phase and the round's position, with no `ρ` coin yet, whose values end the layout's and which squeezes
`ρ_0`'s two coins, and the next round is the rep's, the frame's state after the round's release and the stream's answer
`r0, r1` is level 0's phase at the layout's end with `ρ = (r0, r1)`. -/
theorem answered_l0_last {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    {p n : ℕ} {G H : List Frame} (hR : L0Round (L.cfg x ρ h) p n G H) (a : Msg) (r0 r1 : F128) {m : Masker}
    (hF : L.ProgFollows x) (hA : L.answered x ρ h = .ok m)
    (hc : ∀ q, q < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[q]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hl0 : m.l0 = true) (hpos : m.pos = p) (hrho : m.rho = #[])
    (hn2 : L.coins x h.length = 2) (hb : p + n = (L.cfg x ρ h).order.size)
    (hrep : L.rep x (h.length + 1) = L.rep x h.length)
    (ho : L.opens x ρ (h ++ [(0 :: (encs H ++ (Frame.sq 2 (L.coins x h.length)).enc), a)]) = true)
    (hcs : L.coinsOf x (h ++ [(0 :: (encs H ++ (Frame.sq 2 (L.coins x h.length)).enc), a)]) = #[r0, r1]) :
    ∃ b, L.answered x ρ (h ++ [(0 :: (encs H ++ (Frame.sq 2 (L.coins x h.length)).enc), a)]) =
      .ok { m with stopped := b, pos := p + n, owed := none, rho := #[r0, r1] } := by
  subst hpos
  obtain rfl : nAbs G = n := hR.abs_G
  obtain ⟨out, fr, hr, hout⟩ := roundFree_on (L.cfg x ρ h) { m with stopped := st } G 2 (L.coins x h.length) hc hon
    hi hR.mid_G hR.plain_G (.inl rfl) (fun h' => absurd (h'.1.symm.trans hn2) (by decide)) (by rw [hn2]; decide)
    (by show m.pos + nAbs G ≤ _; omega)
  have hout' : out.data.toList = encs H ++ (Frame.sq 2 (L.coins x h.length)).enc := by
    rw [hout]
    show encs (maskFrames (maskVal (L.cfg x ρ h) st) m.pos G) ++ _ = _
    rw [hR.maskFrames_eq st]
  obtain ⟨b, hb'⟩ := L.answered_of_roundFree x ρ h st _ a hF hA rfl hr hrep (by rw [hout']; exact ho)
  rw [hout', hcs] at hb'
  exact ⟨b, hb'.trans (answer_rho_first _ _ r0 r1 (by show some (L.coins x h.length) = _; rw [hn2])
    (by exact hi) (by exact hl0) (by show m.pos + nAbs G = _; omega) (by exact hrho))⟩

/-- **`ρ_1`'s round takes `ρ_1`**: where the frame takes the coins before level 0's last round, its squeeze of `ρ_1`'s
two coins alone, at level 0's phase with every value of the layout absorbed and `ρ_0`'s words `r0, r1` taken, the next
round is the rep's and the stream answers words `a0, a1` independent of them over F128, the frame's state after the
round's release and the answer is level 0's phase with its four `ρ` coins: the state the closing round reads
(`release_close`). -/
theorem answered_rho1 {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (a : Msg) (r0 r1 a0 a1 : F128) {m : Masker} (hF : L.ProgFollows x) (hA : L.answered x ρ h = .ok m)
    (hc : ∀ q, q < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[q]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hl0 : m.l0 = true) (hpos : m.pos = (L.cfg x ρ h).order.size)
    (hrho : m.rho = #[r0, r1]) (hn2 : L.coins x h.length = 2) (hrep : L.rep x (h.length + 1) = L.rep x h.length)
    (ho : L.opens x ρ (h ++ [(0 :: (Frame.sq 2 (L.coins x h.length)).enc, a)]) = true)
    (hcs : L.coinsOf x (h ++ [(0 :: (Frame.sq 2 (L.coins x h.length)).enc, a)]) = #[a0, a1])
    (hd : _root_.Flock.Zk.rhoIndependent (⟨r0, r1⟩, ⟨a0, a1⟩) = true) :
    ∃ b, L.answered x ρ (h ++ [(0 :: (Frame.sq 2 (L.coins x h.length)).enc, a)]) =
      .ok { m with stopped := b, owed := none, rho := #[r0, r1, a0, a1] } := by
  obtain ⟨out, fr, hr, hout⟩ := roundFree_on (L.cfg x ρ h) { m with stopped := st } [] 2 (L.coins x h.length) hc
    hon hi (by simp) (by simp) (.inl rfl) (fun h' => absurd (h'.1.symm.trans hn2) (by decide))
    (by rw [hn2]; decide) (by show m.pos + nAbs [] ≤ _; simp [nAbs, hpos])
  have hout' : out.data.toList = (Frame.sq 2 (L.coins x h.length)).enc := by
    rw [hout]
    simp only [maskFrames, encs, List.flatMap_nil, List.nil_append]
  obtain ⟨b, hb'⟩ := L.answered_of_roundFree x ρ h st _ a hF hA rfl hr hrep (by rw [hout']; exact ho)
  rw [hout', hcs] at hb'
  exact ⟨b, hb'.trans (_root_.Claims.ZKServer.answer_rho_independent _ _ r0 r1 a0 a1
    (by show some (L.coins x h.length) = _; rw [hn2]) (by exact hi) (by exact hl0)
    (by show m.pos + nAbs [] = _; simp [nAbs, hpos]) (by exact hrho) (by exact hd))⟩

end LeanRelease

end Claims.ZKServer
