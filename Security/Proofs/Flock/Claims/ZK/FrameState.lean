import Proofs.Flock.Claims.ZK.FramePhase

/-!
# The frame's state across level 0's opening

The releases of level 0's rounds (`release_openBind`, `release_l0`, `release_close`) read the frame's state before each
round (`LeanRelease.answered`): its phase (`on`, `l0`), its position (`pos`) and its `ρ` coins. That state comes from the
transport (`LeanRelease.prog`) and the coins the stream answered (`coinsOf`). This file carries it from round to round:

* **The frame's check at level 0** (`check_l0`): where the scheduled round carries no free value (`roundFree`'s free
  offsets are empty, as past the algebraic phase: `roundFree_on_fr`, `roundFree_openBind`), the state `Frame.check`
  returns, whatever the device round and whether it passes, is the scheduled round's (`roundFree`'s), but for `stopped`.
  A device round that passes is the scheduled round byte for byte (`passes_eq`), and one that fails leaves as the
  scheduled round of a stopped rep (`check_some`).
* **The transport's trust** (`LeanRelease.ProgChecks`): after a released frame round, the state of the frame of the
  rep that released it is the state the frame's own `check` returned for a device round and a `stopped` that release
  those bytes, but for `stopped`, which the release does not read.
* **A round's state** (`LeanRelease.answered_of_check`, `answered_of_scheduled`): under it, where the frame's check of
  every device round returns the scheduled round's state and the transport's coin check passes the history with its
  release, the state after it is the frame's taking of the stream's coins at that state (`Masker.answer`).
* **The coins level 0 takes** (`answer_l0_mid`, `answer_rho_first`): before its last masked value the frame only marks
  the round answered; at it, `ρ_0`'s words are its first `ρ` coins (`answer_rho_independent` takes `ρ_1`'s).
* **Across the opening** (`LeanRelease.answered_openBind`, `answered_l0`, `answered_l0_last`, `answered_rho1`): the
  binding starts level 0's phase at the algebraic phase's position; each masked round (`L0Round`) moves the position by
  its values; the pair's sums reach the layout's last value and take `ρ_0`; `ρ_1`'s round takes `ρ_1`, after which the
  closing round's state (`release_close`) holds.
-/

namespace Claims.ZKServer

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree check passes firstDiff inFree)
open FlockSoundness.Refine (Frame encs)

/-! ## The frame's check at level 0 -/

theorem round_of_roundFree {c : Cfg} {s : Masker} {r : ByteArray} {n : ℕ} {o : ByteArray} {t : Masker}
    {fr : Array ℕ} (h : roundFree c s r n = .ok (o, t, fr)) :
    _root_.Flock.Firewall.Frame.round c s r n = .ok (o, t) := by
  unfold _root_.Flock.Firewall.Frame.round
  rw [h]
  rfl

/-- Rounds of one size that differ at no byte are one round. -/
theorem eq_of_firstDiff {v r : ByteArray} (hsz : v.size = r.size) (h : firstDiff v r #[] = none) : v = r := by
  unfold firstDiff at h
  by_cases he : (v == r) = true
  · exact ByteArray.ext (eq_of_beq (show (v.data == r.data) = true from he))
  · rw [if_neg he, List.find?_eq_none] at h
    apply ByteArray.ext
    apply Array.ext hsz
    intro i h1 h2
    have hf : inFree #[] i = false := by simp [inFree]
    have hi : v.get! i = r.get! i := by simpa [hf] using h i (List.mem_range.mpr h1)
    have e : v.data[i]! = r.data[i]! := hi
    rwa [getElem!_pos v.data i h1, getElem!_pos r.data i h2] at e

/-- **Where the scheduled round carries no free value, a device round that passes is it**, byte for byte. -/
theorem passes_eq {c : Cfg} {s : Masker} {v r : ByteArray} {n : ℕ} {o : ByteArray} {t : Masker}
    (hr : roundFree c s r n = .ok (o, t, #[])) (hp : passes c s v r n = .ok none) : v = r := by
  unfold passes at hp
  rw [hr] at hp
  simp only [bind, Except.bind] at hp
  by_cases hsz : v.size = r.size
  · simp only [show (v.size != r.size) = false by simp [hsz], Bool.false_eq_true, ↓reduceIte] at hp
    apply eq_of_firstDiff hsz
    cases hfd : firstDiff v r #[] with
    | none => rfl
    | some i => rw [hfd] at hp; cases hp
  · simp only [show (v.size != r.size) = true by simpa using hsz, ↓reduceIte] at hp
    cases hp

/-- The frame's `check` passes the device round only where `passes` does. -/
theorem check_none_passes {c : Cfg} {s : Masker} {v r : ByteArray} {n : ℕ} {out : ByteArray} {s' : Masker}
    (h : check c s v r n = .ok (out, s', none)) : passes c s v r n = .ok none := by
  unfold check _root_.Flock.Firewall.Frame.standIn at h
  cases hs : s.stopped <;> simp only [hs, Bool.false_eq_true, ↓reduceIte] at h
  · cases hp : passes c s v r n with
    | error e => simp only [hp] at h; cases h
    | ok w =>
      simp only [hp] at h
      cases w with
      | some w =>
        cases hr : _root_.Flock.Firewall.Frame.round c { s with stopped := true } r n with
        | error e => simp only [hr] at h; cases h
        | ok x => simp only [hr] at h; cases h
      | none => rfl
  · cases hr : _root_.Flock.Firewall.Frame.round c { s with stopped := true } r n with
    | error e => simp only [hr] at h; cases h
    | ok x => simp only [hr] at h; cases h

/-- **The frame's check returns the scheduled round's state** where the scheduled round carries no free value: the state
`round` returns on it, of the rep as it is or as a stopped rep's. -/
theorem check_state {c : Cfg} {s : Masker} {v r : ByteArray} {n : ℕ} {o out : ByteArray} {t s' : Masker}
    {why : Option String} (hr : roundFree c s r n = .ok (o, t, #[])) (h : check c s v r n = .ok (out, s', why)) :
    (out, s') = (o, t) ∨ _root_.Flock.Firewall.Frame.round c { s with stopped := true } r n = .ok (out, s') := by
  cases why with
  | some w => exact .inr (check_some h)
  | none =>
    left
    have hv := check_none h
    rw [passes_eq hr (check_none_passes h), round_of_roundFree hr] at hv
    cases hv
    rfl

/-- **The frame's check at level 0**: where the scheduled round `r` carries no free value at any `stopped` and its state
is `t`'s but for `stopped`, the state the frame's check returns on any device round is `t`'s but for `stopped`. -/
theorem check_l0 {c : Cfg} {m t : Masker} {v r : ByteArray} {n : ℕ} {st : Bool} {out : ByteArray} {s' : Masker}
    {why : Option String}
    (hr : ∀ b, ∃ o, roundFree c { m with stopped := b } r n = .ok (o, { t with stopped := b }, #[]))
    (h : check c { m with stopped := st } v r n = .ok (out, s', why)) : ∃ b, s' = { t with stopped := b } := by
  obtain ⟨o, ho⟩ := hr st
  rcases check_state ho h with he | he
  · exact ⟨st, by cases he; rfl⟩
  · obtain ⟨o', ho'⟩ := hr true
    have he' : _root_.Flock.Firewall.Frame.round c { m with stopped := true } r n = .ok (out, s') := he
    rw [round_of_roundFree ho'] at he'
    cases he'
    exact ⟨true, rfl⟩

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

/-- The frame's check after `h` at the state `m`, of a stopped rep (`st`) or not, of the device round `v` against the
firewall's scheduled round at witness `w` (`pass`, `check_release`). -/
def checkAt {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) (m : Masker)
    (st : Bool) (v : Msg) : Except String (ByteArray × Masker × Option String) :=
  check (L.cfg x ρ h) { m with stopped := st } ⟨v.toArray⟩
    ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length)

/-- **The transport's frame state is the frame's check's** at `x` and witness `w`: where the frame takes the coins
after `h` at the state `m`, round `h.length` is a frame round and the frame's check of some device round releases `out`,
the state of the frame of the rep that released it, after the history with that release and the stream's answer `a`,
is the state the frame's check of a device round `v`, of a stopped rep or not, returned with that release, but for
`stopped`, which the release does not read. A named hypothesis, one of the transport's trusts beside `ProgReads` and of
its kind: `prog` is the state the session loop's requests leave the rep's frame in (`Flock.Firewall.Frames`, on the
schedule `Drive.reqs` fixes), which a release takes as given. The frame's step (`Frame.step`) sets its state, at a round
that leaves, to what `check` returns at the statement's scheduled round (`Place.sched`), to `round`'s of a stopped rep at
the rep's stop (`Place.fail`: `check`'s of a stopped rep, where the loop sends the scheduled round) and to `round`'s at
an inner proof's round (`Place.own`: `check`'s with the round in its own place, `check_self`, where the loop sends the
firewall's own round); the Prop adds that the loop's requests are those. The frame's own code is `check_l0`, proved. At
the loop's own frames (`LeanRelease.ofLoop`) the Prop is a theorem (`ofLoop_progChecks`). -/
def ProgChecks {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) : Prop :=
  ∀ (ρ : R) (h : List (Msg × Msg)) (a : Msg) (m : Masker) (out : ByteArray),
    L.answered x ρ h = .ok m → L.kindAt x h.length = .masked →
    (∃ v s' why, L.checkAt x w ρ h m false v = .ok (out, s', why)) →
    ∃ v st s' why, L.checkAt x w ρ h m st v = .ok (out, s', why) ∧
      L.prog x (h ++ [(0 :: out.data.toList, a)]) =
        { s' with stopped := (L.prog x (h ++ [(0 :: out.data.toList, a)])).stopped }

/-- Within a rep the configuration is the rep's, after one more round as before it. -/
theorem cfg_snoc {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (e : Msg × Msg)
    (hrep : L.rep x (h.length + 1) = L.rep x h.length) : L.cfg x ρ (h ++ [e]) = L.cfg x ρ h := by
  unfold cfg
  rw [List.length_append, List.length_singleton, hrep]

/-- **A round's state**: where the frame takes the coins after `h` at the state `m`, round `h.length` is a frame
round, the next round is the same rep's,
the frame's check of some device round releases `out`, every check that does returns `t`'s state but for `stopped`, and
the transport's coin check passes the history with the release and the answer `a`, the frame's state after that history
is its taking of the stream's coins at `t`, at some `stopped`. -/
theorem answered_of_check {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg))
    (a : Msg) {m : Masker} (t : Masker) {out : ByteArray} (hP : L.ProgChecks x w) (hA : L.answered x ρ h = .ok m)
    (hkd : L.kindAt x h.length = .masked) (hrep : L.rep x (h.length + 1) = L.rep x h.length)
    (ho : L.opens x ρ (h ++ [(0 :: out.data.toList, a)]) = true)
    (hex : ∃ v s' why, L.checkAt x w ρ h m false v = .ok (out, s', why))
    (hst : ∀ v st s' why, L.checkAt x w ρ h m st v = .ok (out, s', why) → ∃ b, s' = { t with stopped := b }) :
    ∃ b, L.answered x ρ (h ++ [(0 :: out.data.toList, a)]) =
      { t with stopped := b }.answer (L.cfg x ρ h) (L.coinsOf x (h ++ [(0 :: out.data.toList, a)])) := by
  obtain ⟨v, st, s', why, hc, hp⟩ := hP ρ h a m out hA hkd hex
  obtain ⟨b, rfl⟩ := hst v st s' why hc
  refine ⟨(L.prog x (h ++ [(0 :: out.data.toList, a)])).stopped, ?_⟩
  have hp' : L.prog x (h ++ [(0 :: out.data.toList, a)]) =
      { t with stopped := (L.prog x (h ++ [(0 :: out.data.toList, a)])).stopped } := hp
  have hm := kindAt_masked hkd
  have hn : ¬ (h ++ [(0 :: out.data.toList, a)]).length ≤ 2 := by
    rw [List.length_append, List.length_singleton]
    omega
  have hs : decide (L.rep x (h ++ [(0 :: out.data.toList, a)]).length =
      L.rep x ((h ++ [(0 :: out.data.toList, a)]).length - 1)) = true := by
    rw [List.length_append, List.length_singleton, Nat.add_sub_cancel]
    exact decide_eq_true hrep
  rw [L.answered_of_opens x ρ _ ho]
  unfold frameAnswer
  rw [show (h ++ [(0 :: out.data.toList, a)]).getLast? = some (0 :: out.data.toList, a) by simp]
  dsimp only
  unfold takeAt
  rw [if_neg hn, hs, if_pos rfl, L.cfg_snoc x ρ h _ hrep]
  exact congrArg (fun s => s.answer (L.cfg x ρ h) (L.coinsOf x (h ++ [(0 :: out.data.toList, a)]))) hp'

/-- **A round's state at level 0**: where the firewall's scheduled round after `h` carries no free value at any
`stopped`, with `t`'s state but for `stopped`, and releases `out` at the state `m`, the frame's state after the history
with that release and the stream's answer is its taking of the stream's coins at `t`, at some `stopped`. -/
theorem answered_of_scheduled {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg))
    (a : Msg) {m : Masker} (t : Masker) {out : ByteArray} (hP : L.ProgChecks x w) (hA : L.answered x ρ h = .ok m)
    (hkd : L.kindAt x h.length = .masked) (hrep : L.rep x (h.length + 1) = L.rep x h.length)
    (ho : L.opens x ρ (h ++ [(0 :: out.data.toList, a)]) = true)
    (hr : ∀ b, ∃ o, roundFree (L.cfg x ρ h) { m with stopped := b }
      ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) =
        .ok (o, { t with stopped := b }, #[]))
    (h0 : roundFree (L.cfg x ρ h) { m with stopped := false }
      ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) =
        .ok (out, { t with stopped := false }, #[])) :
    ∃ b, L.answered x ρ (h ++ [(0 :: out.data.toList, a)]) =
      { t with stopped := b }.answer (L.cfg x ρ h) (L.coinsOf x (h ++ [(0 :: out.data.toList, a)])) :=
  L.answered_of_check x w ρ h a t hP hA hkd hrep ho
    ⟨L.scheduled x w ρ h (computedOf (L.computed x w ρ h)), _, none, check_self rfl (round_of_roundFree h0)⟩
    fun _ _ _ _ hc => check_l0 hr hc

/-! ## Across the opening -/

/-- **The binding starts level 0's phase**: where the frame takes the coins before the opening's binding at a state
outside a masked phase, after the algebraic phase's opening and before level 0 closes, short of the layout's last
value, at a Ligerito rep, the firewall's scheduled round is the binding with `z`'s squeeze, and the next round is the
rep's, the frame's state after the binding's release and the stream's answer to `z`'s squeeze is level 0's phase at the
same position. -/
theorem answered_openBind {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg))
    (T : F128) (cap : ByteArray) (a : Msg) {m : Masker} (hP : L.ProgChecks x w) (hA : L.answered x ρ h = .ok m)
    (hsch : L.scheduled x w ρ h (computedOf (L.computed x w ρ h)) =
      encs (openBind T cap) ++ (Frame.sq 2 (L.coins x h.length)).enc)
    (hon : m.on = false) (hd : m.done = true) (hlig : (L.cfg x ρ h).lig = true) (hl0 : m.doneL0 = false)
    (hi : m.inner = none) (hcap : cap.size < 2 ^ 64) (hn0 : L.coins x h.length ≠ 0)
    (hn : L.coins x h.length < 2 ^ 64) (hp : m.pos ≠ (L.cfg x ρ h).order.size)
    (hkd : L.kindAt x h.length = .masked) (hrep : L.rep x (h.length + 1) = L.rep x h.length)
    (ho : L.opens x ρ (h ++ [(0 :: (encs (openBind T cap) ++ (Frame.sq 2 (L.coins x h.length)).enc), a)]) = true)
    (hcs : (L.coinsOf x (h ++ [(0 :: (encs (openBind T cap) ++ (Frame.sq 2 (L.coins x h.length)).enc), a)])).size =
      L.coins x h.length) :
    ∃ b, L.answered x ρ (h ++ [(0 :: (encs (openBind T cap) ++ (Frame.sq 2 (L.coins x h.length)).enc), a)]) =
      .ok { m with stopped := b, on := true, l0 := true, owed := none } := by
  have hr : ∀ b, ∃ o, roundFree (L.cfg x ρ h) { m with stopped := b }
      ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) =
        .ok (o, { { m with on := true, l0 := true, owed := some (L.coins x h.length) } with stopped := b }, #[]) ∧
      o.data.toList = encs (openBind T cap) ++ (Frame.sq 2 (L.coins x h.length)).enc := by
    intro b
    obtain ⟨o, hr, hout⟩ := roundFree_openBind (L.cfg x ρ h) { m with stopped := b } T cap (L.coins x h.length) hon
      hd hlig hl0 hi hcap hn0 hn
    rw [hsch]
    exact ⟨o, hr, hout⟩
  obtain ⟨out, hr0, hout⟩ := hr false
  obtain ⟨b, hb⟩ := L.answered_of_scheduled x w ρ h a
    { m with on := true, l0 := true, owed := some (L.coins x h.length) } hP hA hkd hrep (by rw [hout]; exact ho)
    (fun b => (hr b).imp fun _ hb => hb.1) hr0
  rw [hout] at hb
  exact ⟨b, hb.trans (answer_l0_mid _ _ _ (by rw [hcs]; try rfl) (by exact hi) (by rfl) (by exact hp))⟩

/-- **A masked round of level 0 moves the position by its values**: where the frame takes the coins before a level-0
round (`L0Round`) at level 0's phase and the round's position, its values short of the layout's last, the firewall's
scheduled round is the round at the pads (`G`) with its squeeze, and the next round is the rep's, the frame's state
after the round's release and the stream's answer is level 0's phase past the round's values. -/
theorem answered_l0 {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg))
    {p n : ℕ} {G H : List Frame} (hR : L0Round (L.cfg x ρ h) p n G H) (k : UInt8) (a : Msg) {m : Masker}
    (hP : L.ProgChecks x w) (hA : L.answered x ρ h = .ok m)
    (hsch : L.scheduled x w ρ h (computedOf (L.computed x w ρ h)) = encs G ++ (Frame.sq k (L.coins x h.length)).enc)
    (hc : ∀ q, q < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[q]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hl0 : m.l0 = true) (hpos : m.pos = p)
    (hk : k = 2 ∨ (k = 1 ∧ L.coins x h.length = 1)) (h0 : ¬ (L.coins x h.length = 0 ∧ k = 2))
    (hn : L.coins x h.length < 2 ^ 64) (hb : p + n < (L.cfg x ρ h).order.size)
    (hkd : L.kindAt x h.length = .masked) (hrep : L.rep x (h.length + 1) = L.rep x h.length)
    (ho : L.opens x ρ (h ++ [(0 :: (encs H ++ (Frame.sq k (L.coins x h.length)).enc), a)]) = true)
    (hcs : (L.coinsOf x (h ++ [(0 :: (encs H ++ (Frame.sq k (L.coins x h.length)).enc), a)])).size =
      L.coins x h.length) :
    ∃ b, L.answered x ρ (h ++ [(0 :: (encs H ++ (Frame.sq k (L.coins x h.length)).enc), a)]) =
      .ok { m with stopped := b, pos := p + n, owed := none } := by
  have hpi := hR.piop
  subst hpos
  obtain rfl : nAbs G = n := hR.abs_G
  have hr : ∀ b, ∃ o, roundFree (L.cfg x ρ h) { m with stopped := b }
      ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) =
        .ok (o, { { m with pos := m.pos + nAbs G, owed := some (L.coins x h.length) } with stopped := b }, #[]) ∧
      o.data.toList = encs H ++ (Frame.sq k (L.coins x h.length)).enc := by
    intro b
    obtain ⟨o, hr, hout⟩ := roundFree_on_fr (L.cfg x ρ h) { m with stopped := b } G k (L.coins x h.length) hc hon
      hi hR.mid_G hR.plain_G hk h0 hn (by show m.pos + nAbs G ≤ _; omega) hpi
    rw [hsch]
    refine ⟨o, hr, ?_⟩
    rw [hout]
    show encs (maskFrames (maskVal (L.cfg x ρ h) b) m.pos G) ++ _ = _
    rw [hR.maskFrames_eq b]
  obtain ⟨out, hr0, hout⟩ := hr false
  obtain ⟨b, hb'⟩ := L.answered_of_scheduled x w ρ h a
    { m with pos := m.pos + nAbs G, owed := some (L.coins x h.length) } hP hA hkd hrep (by rw [hout]; exact ho)
    (fun b => (hr b).imp fun _ hb => hb.1) hr0
  rw [hout] at hb'
  exact ⟨b, hb'.trans (answer_l0_mid _ _ _ (by rw [hcs]; try rfl) (by exact hi) (by exact hl0)
    (by show m.pos + nAbs G ≠ _; omega))⟩

/-- **The pair's sums reach the layout's last value and take `ρ_0`**: where the frame takes the coins before a level-0
round at level 0's phase and the round's position, with no `ρ` coin yet, whose values end the layout's and which squeezes
`ρ_0`'s two coins, the firewall's scheduled round is the round at the pads with its squeeze, and the next round is the
rep's, the frame's state after the round's release and the stream's answer `r0, r1` is level 0's phase at the layout's
end with `ρ = (r0, r1)`. -/
theorem answered_l0_last {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg))
    {p n : ℕ} {G H : List Frame} (hR : L0Round (L.cfg x ρ h) p n G H) (a : Msg) (r0 r1 : F128) {m : Masker}
    (hP : L.ProgChecks x w) (hA : L.answered x ρ h = .ok m)
    (hsch : L.scheduled x w ρ h (computedOf (L.computed x w ρ h)) = encs G ++ (Frame.sq 2 (L.coins x h.length)).enc)
    (hc : ∀ q, q < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[q]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hl0 : m.l0 = true) (hpos : m.pos = p) (hrho : m.rho = #[])
    (hn2 : L.coins x h.length = 2) (hb : p + n = (L.cfg x ρ h).order.size)
    (hkd : L.kindAt x h.length = .masked) (hrep : L.rep x (h.length + 1) = L.rep x h.length)
    (ho : L.opens x ρ (h ++ [(0 :: (encs H ++ (Frame.sq 2 (L.coins x h.length)).enc), a)]) = true)
    (hcs : L.coinsOf x (h ++ [(0 :: (encs H ++ (Frame.sq 2 (L.coins x h.length)).enc), a)]) = #[r0, r1]) :
    ∃ b, L.answered x ρ (h ++ [(0 :: (encs H ++ (Frame.sq 2 (L.coins x h.length)).enc), a)]) =
      .ok { m with stopped := b, pos := p + n, owed := none, rho := #[r0, r1] } := by
  have hpi := hR.piop
  subst hpos
  obtain rfl : nAbs G = n := hR.abs_G
  have hr : ∀ b, ∃ o, roundFree (L.cfg x ρ h) { m with stopped := b }
      ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) =
        .ok (o, { { m with pos := m.pos + nAbs G, owed := some (L.coins x h.length) } with stopped := b }, #[]) ∧
      o.data.toList = encs H ++ (Frame.sq 2 (L.coins x h.length)).enc := by
    intro b
    obtain ⟨o, hr, hout⟩ := roundFree_on_fr (L.cfg x ρ h) { m with stopped := b } G 2 (L.coins x h.length) hc hon
      hi hR.mid_G hR.plain_G (.inl rfl) (fun h' => absurd (h'.1.symm.trans hn2) (by decide)) (by rw [hn2]; decide)
      (by show m.pos + nAbs G ≤ _; omega) hpi
    rw [hsch]
    refine ⟨o, hr, ?_⟩
    rw [hout]
    show encs (maskFrames (maskVal (L.cfg x ρ h) b) m.pos G) ++ _ = _
    rw [hR.maskFrames_eq b]
  obtain ⟨out, hr0, hout⟩ := hr false
  obtain ⟨b, hb'⟩ := L.answered_of_scheduled x w ρ h a
    { m with pos := m.pos + nAbs G, owed := some (L.coins x h.length) } hP hA hkd hrep (by rw [hout]; exact ho)
    (fun b => (hr b).imp fun _ hb => hb.1) hr0
  rw [hout, hcs] at hb'
  exact ⟨b, hb'.trans (answer_rho_first _ _ r0 r1 (by show some (L.coins x h.length) = _; rw [hn2])
    (by exact hi) (by exact hl0) (by show m.pos + nAbs G = _; omega) (by exact hrho))⟩

/-- **`ρ_1`'s round takes `ρ_1`**: where the frame takes the coins before level 0's last round, its squeeze of `ρ_1`'s
two coins alone, at level 0's phase with every value of the layout absorbed past the algebraic phase's and `ρ_0`'s
words `r0, r1` taken, the firewall's scheduled round is that squeeze, the next round is the rep's and the stream answers
words `a0, a1` independent of them over F128, the frame's state after the round's release and the answer is level 0's
phase with its four `ρ` coins: the state the closing round reads (`release_close`). -/
theorem answered_rho1 {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg))
    (a : Msg) (r0 r1 a0 a1 : F128) {m : Masker} (hP : L.ProgChecks x w) (hA : L.answered x ρ h = .ok m)
    (hsch : L.scheduled x w ρ h (computedOf (L.computed x w ρ h)) = (Frame.sq 2 (L.coins x h.length)).enc)
    (hc : ∀ q, q < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[q]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hl0 : m.l0 = true) (hpos : m.pos = (L.cfg x ρ h).order.size)
    (hpi : (L.cfg x ρ h).nPiop ≤ m.pos)
    (hrho : m.rho = #[r0, r1]) (hn2 : L.coins x h.length = 2) (hkd : L.kindAt x h.length = .masked)
    (hrep : L.rep x (h.length + 1) = L.rep x h.length)
    (ho : L.opens x ρ (h ++ [(0 :: (Frame.sq 2 (L.coins x h.length)).enc, a)]) = true)
    (hcs : L.coinsOf x (h ++ [(0 :: (Frame.sq 2 (L.coins x h.length)).enc, a)]) = #[a0, a1])
    (hd : _root_.Flock.Zk.rhoIndependent (⟨r0, r1⟩, ⟨a0, a1⟩) = true) :
    ∃ b, L.answered x ρ (h ++ [(0 :: (Frame.sq 2 (L.coins x h.length)).enc, a)]) =
      .ok { m with stopped := b, owed := none, rho := #[r0, r1, a0, a1] } := by
  have hr : ∀ b, ∃ o, roundFree (L.cfg x ρ h) { m with stopped := b }
      ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) =
        .ok (o, { { m with pos := m.pos + nAbs [], owed := some (L.coins x h.length) } with stopped := b }, #[]) ∧
      o.data.toList = (Frame.sq 2 (L.coins x h.length)).enc := by
    intro b
    obtain ⟨o, hr, hout⟩ := roundFree_on_fr (L.cfg x ρ h) { m with stopped := b } [] 2 (L.coins x h.length) hc
      hon hi (by simp) (by simp) (.inl rfl) (fun h' => absurd (h'.1.symm.trans hn2) (by decide))
      (by rw [hn2]; decide) (by show m.pos + nAbs [] ≤ _; simp [nAbs, hpos]) hpi
    rw [hsch]
    refine ⟨o, hr, ?_⟩
    rw [hout]
    simp only [maskFrames, encs, List.flatMap_nil, List.nil_append]
  obtain ⟨out, hr0, hout⟩ := hr false
  obtain ⟨b, hb'⟩ := L.answered_of_scheduled x w ρ h a
    { m with pos := m.pos + nAbs [], owed := some (L.coins x h.length) } hP hA hkd hrep (by rw [hout]; exact ho)
    (fun b => (hr b).imp fun _ hb => hb.1) hr0
  rw [hout, hcs] at hb'
  exact ⟨b, hb'.trans (_root_.Claims.ZKServer.answer_rho_independent _ _ r0 r1 a0 a1
    (by show some (L.coins x h.length) = _; rw [hn2]) (by exact hi) (by exact hl0)
    (by show m.pos + nAbs [] = _; simp [nAbs, hpos]) (by exact hrho) (by exact hd))⟩

end LeanRelease

end Claims.ZKServer
