import Proofs.Flock.Claims.ZK.Blind
import Flock.Firewall.Frame
import Flock.Firewall.Assumptions

/-!
# The ZK server's release is the Lean frame, at the Lean frame's draws

The outer firewall releases only what each rep's Lean frame returns (`flock-firewall frame`, PROTOCOL.md §10.1 and
§10.4), and the frame masks with randomness it drew itself. At the rep's start it reads bytes from the operating system,
parses them into the rep's draws (`Flock.Firewall.Frame.Draws.ofBytes`: the pads `h`, `μ`, the inner rows' padding, `τ`'s
salt and the pads tree's salts) and commits them (`padsRoot`). `LeanRelease` is the ZK server whose release is that frame
at the draws on the server's tape (`LeanRelease.draws`):
* the round's rep's configuration (`LeanRelease.cfg`) is `Cfg.ofDraws` of its draws on the tape, the function
  `flock-firewall frame` starts the rep with: the layout, the pads, the absorbed order, the root it computes and the mask
  slot (`cfg_pads`, `cfg_root`). The frame's steps take it and return only the rep's state (`Masker`), so none changes
  the pads or the root. Only that state comes from the transport (`LeanRelease.prog`): where the rep's stream is, a
  function of the statement and the history, never of the tape;
* each round's release (`LeanRelease.release`) is the frame's step: it takes the coins the stream answered the round
  before (`Masker.answer`, which stops at a zerocheck outer coin `1`, at level 0's `ρ_0, ρ_1` dependent over F128 and
  at the inner proof's `β = 0`: `answer_rho_dependent`, `answer_beta_zero`), then the device round (`Frame.round`,
  which masks every absorbed value under the next pad, inserts the pads' label and root, and stops at a mask slot short
  of full rank). Every stop the coins decide is the frame's, from the coins alone, and where the frame refuses the
  coins the release is the stop whatever the round (`LeanRelease.release_of_refusesCoins`);
* a value carries whether it is the firewall's stand-in (`Val := Msg × Bool`), and the frame of a stopped rep releases the
  algebraic phase's values as their pads alone (`Masker.stopped`). The firewall's scheduled round is a stand-in. The
  worker's values are not, and the check refuses a worker value that says it is;
* the pass decision is the frame's (`Frame.check`, `LeanRelease.pass`): the firewall hands it each device round with
  the scheduled round in its place, and the device round passes where it equals the scheduled round byte for byte
  outside the algebraic phase's masked values and the frame takes it. A round that passes leaves as the frame's release
  of it, and one that fails as the frame's release of the scheduled round, as a stopped rep's
  (`LeanRelease.check_release`). Rust's own checks run first and can only fail a round more (`LeanRelease.cross`).

`LeanRelease.toSrv` is it as a `ZkServer` with one masked slot per round: the worker's value is its device round, the mask
is the frame's release, and the check is Rust's checks, the frame's pass decision and the frame's own refusals.

The firewall stops a session only at its `Finish` (§10.1): once a round fails, it sends its own scheduled round in the
device's place (`LeanRelease.scheduled`), and the last round is the stop. That is the deferred server `ZkServer.nextD` at
`L.toSrv` with the stand-in `LeanRelease.fillOf`. Every way a worker fails is a round that fails:
* an error, output that does not decode, a panic or silence is no value (`LeanRelease.pass_of_none`);
* a tree at another size than the shadow's, or any round Rust's checks, the frame's pass decision or the frame refuses, is
  a value that fails.

From that round on, below the last round, the release is the frame's release of the scheduled round as a stopped rep's,
and at the last round it is the stop (`LeanRelease.nextD_of_fail`), as the fixed firewall does.

The device round and the scheduled round read, by name, the values the firewall fixes by its own computation, as its
transcript reports them (`computedOf`): its level-0 recheck, its opening, `final_c'` and the inner proof. Five Props give
`ZkServer.RefD` at the deferred release (`LeanRelease.refD`):
* `FirewallComputes` (D8 (a), still a named assumption): each reported value is the specified one;
* `Reports`: the transcript reports every specified value;
* `LeanRelease.LiveRefine`: on every history whose rounds all passed and whose last coins the frame takes, the frame's
  release of the worker's round is the prover's message;
* `LeanRelease.FillRefine`: on every history below the last round on which a round failed and whose last coins the
  frame takes, the frame's release of the firewall's scheduled round, with the specified values, is the prover's
  message;
* `LeanRelease.StopRefine`: on every history whose last coins the frame refuses, the prover's message is the stop. It
  reads no worker value and nothing the firewall computes.

So `Flock.zk_server_blind`'s bound holds at the deployed release, at any worker: one bit a session
(`Flock.zk_server_lean`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock.Firewall.Frame (Cfg Masker Draws MaskSlot padsRoot)

/-- The value the firewall's transcript `t` reports fixing for `what` by its own computation (`By.recheck`, `By.shadow`,
`By.own`): the first such item's. -/
def computedOf (t : _root_.Flock.Firewall.Contract.Transcript) (what : String) : Option ByteArray :=
  t.items.toList.findSome? fun
    | .fixed w .recheck want _ _ | .fixed w .shadow want _ _ | .fixed w .own want _ _ =>
      if w = what then some want else none
    | _ => none

/-- Under D8 (a), every value the transcript reports fixing by the firewall's own computation is `spec`'s. -/
theorem computedOf_spec {spec : String → Option ByteArray} {t : _root_.Flock.Firewall.Contract.Transcript}
    (hc : _root_.Flock.Firewall.Contract.FirewallComputes spec t) {what : String} {v : ByteArray}
    (hv : computedOf t what = some v) : spec what = some v := by
  obtain ⟨a, ha, hf⟩ := List.exists_of_findSome?_eq_some hv
  cases a with
  | fixed w src want sent ps =>
    cases src
    case recheck | shadow | own =>
      dsimp only at hf
      split at hf
      · cases hf
        subst_vars
        exact hc _ _ _ sent ps ha (by decide)
      · cases hf
    all_goals cases hf
  | _ => cases hf

/-- The transcript `t` reports a value for every name `spec` fixes. -/
def Reports (spec : String → Option ByteArray) (t : _root_.Flock.Firewall.Contract.Transcript) : Prop :=
  ∀ what v, spec what = some v → (computedOf t what).isSome

/-- Under D8 (a), a transcript that reports every name `spec` fixes reports exactly `spec`. -/
theorem computedOf_eq {spec : String → Option ByteArray} {t : _root_.Flock.Firewall.Contract.Transcript}
    (hc : _root_.Flock.Firewall.Contract.FirewallComputes spec t) (hr : Reports spec t) : computedOf t = spec := by
  funext what
  cases hs : spec what with
  | none =>
    cases hv : computedOf t what with
    | none => rfl
    | some v =>
      have := computedOf_spec hc hv
      rw [hs] at this
      cases this
  | some v =>
    obtain ⟨v', hv'⟩ := Option.isSome_iff_exists.mp (hr what v hs)
    have := computedOf_spec hc hv'
    rw [hs] at this
    rw [hv', this]

/-! ## The frame's coin stops

The frame decides the two coin stops the prover's own checks once decided, from the coins alone (`Masker.answer`): the
inner proof's `β = 0`, and level 0's `ρ_0, ρ_1` dependent over F128 (`Flock.Zk.rhoIndependent`). Each is a stop, and any
other value of them is taken. -/

section CoinStops
open _root_.Flock.Firewall.Frame (INNER_BETA RHO_COINS STOP innerCoins rhoCoins)

/-- At the inner proof's `β` round, the coin `β = 0` is a stop. -/
theorem answer_beta_zero (c : Cfg) (s : Masker) (b : _root_.Flock.F128) (hn : s.owed = some 1)
    (hi : s.inner = some INNER_BETA) (hb : b.isZero = true) :
    s.answer c #[b] = .error s!"{STOP}degenerate coin β = 0" := by
  unfold _root_.Flock.Firewall.Frame.Masker.answer
  simp [hn, hi, innerCoins, hb, INNER_BETA]
  rfl

/-- At the inner proof's `β` round, any other coin is taken. -/
theorem answer_beta_ne (c : Cfg) (s : Masker) (b : _root_.Flock.F128) (hn : s.owed = some 1)
    (hi : s.inner = some INNER_BETA) (hb : b.isZero = false) :
    s.answer c #[b] = .ok { s with owed := none, inner := some (INNER_BETA + 1) } := by
  unfold _root_.Flock.Firewall.Frame.Masker.answer
  simp [hn, hi, innerCoins, hb, INNER_BETA]
  rfl

/-- After level 0's last masked value and `ρ_0`'s words `r0, r1`, `ρ_1`'s words `a, b` dependent with them over F128
are a stop. -/
theorem answer_rho_dependent (c : Cfg) (s : Masker) (r0 r1 a b : _root_.Flock.F128) (hn : s.owed = some 2)
    (hi : s.inner = none) (hl : s.l0 = true) (hp : s.pos = c.order.size) (hr : s.rho = #[r0, r1])
    (hd : _root_.Flock.Zk.rhoIndependent (⟨r0, r1⟩, ⟨a, b⟩) = false) :
    s.answer c #[a, b] = .error s!"{STOP}degenerate coins: rho_0, rho_1 dependent over F128" := by
  unfold _root_.Flock.Firewall.Frame.Masker.answer
  simp [hn, hi, hl, hp, rhoCoins, hr, RHO_COINS, hd]
  rfl

/-- After level 0's last masked value and `ρ_0`'s words `r0, r1`, `ρ_1`'s words `a, b` independent of them are
taken. -/
theorem answer_rho_independent (c : Cfg) (s : Masker) (r0 r1 a b : _root_.Flock.F128) (hn : s.owed = some 2)
    (hi : s.inner = none) (hl : s.l0 = true) (hp : s.pos = c.order.size) (hr : s.rho = #[r0, r1])
    (hd : _root_.Flock.Zk.rhoIndependent (⟨r0, r1⟩, ⟨a, b⟩) = true) :
    s.answer c #[a, b] = .ok { s with owed := none, rho := #[r0, r1, a, b] } := by
  unfold _root_.Flock.Firewall.Frame.Masker.answer
  simp [hn, hi, hl, hp, rhoCoins, hr, RHO_COINS, hd]
  rfl

end CoinStops

/-! ## The frame's pass decision

The firewall hands the frame each device round with its scheduled round in its place, and the frame's `check` decides:
a round it passes leaves as the frame's release of it, and one it fails as the frame's release of the scheduled round, as
a stopped rep's. -/

section PassDecision
open _root_.Flock.Firewall.Frame (check standIn passes)

/-- The frame's `check` passes the device round only through `Frame.round`: its release and state are `round`'s on it. -/
theorem check_none {c : Cfg} {s : Masker} {v r : ByteArray} {n : Nat} {out : ByteArray} {s' : Masker}
    (h : check c s v r n = .ok (out, s', none)) : _root_.Flock.Firewall.Frame.round c s v n = .ok (out, s') := by
  unfold check standIn at h
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
      | none =>
        cases hv : _root_.Flock.Firewall.Frame.round c s v n with
        | ok x => simp only [hv] at h; cases h; rfl
        | error e =>
          simp only [hv] at h
          split at h
          · cases h
          · cases hr : _root_.Flock.Firewall.Frame.round c { s with stopped := true } r n with
            | error e => simp only [hr] at h; cases h
            | ok x => simp only [hr] at h; cases h
  · cases hr : _root_.Flock.Firewall.Frame.round c { s with stopped := true } r n with
    | error e => simp only [hr] at h; cases h
    | ok x => simp only [hr] at h; cases h

/-- Where the frame's `check` fails the device round, what leaves is the scheduled round `r` as a stopped rep's. -/
theorem check_some {c : Cfg} {s : Masker} {v r : ByteArray} {n : Nat} {out : ByteArray} {s' : Masker} {w : String}
    (h : check c s v r n = .ok (out, s', some w)) :
    _root_.Flock.Firewall.Frame.round c { s with stopped := true } r n = .ok (out, s') := by
  unfold check standIn at h
  cases hs : s.stopped <;> simp only [hs, Bool.false_eq_true, ↓reduceIte] at h
  · cases hp : passes c s v r n with
    | error e => simp only [hp] at h; cases h
    | ok w' =>
      simp only [hp] at h
      cases w' with
      | some w' =>
        cases hr : _root_.Flock.Firewall.Frame.round c { s with stopped := true } r n with
        | error e => simp only [hr] at h; cases h
        | ok x => simp only [hr] at h; cases h; rfl
      | none =>
        cases hv : _root_.Flock.Firewall.Frame.round c s v n with
        | ok x => simp only [hv] at h; cases h
        | error e =>
          simp only [hv] at h
          split at h
          · cases h
          · cases hr : _root_.Flock.Firewall.Frame.round c { s with stopped := true } r n with
            | error e => simp only [hr] at h; cases h
            | ok x => simp only [hr] at h; cases h; rfl
  · cases hr : _root_.Flock.Firewall.Frame.round c { s with stopped := true } r n with
    | error e => simp only [hr] at h; cases h
    | ok x => simp only [hr] at h; cases h; rfl

end PassDecision

/-- **The deployed release**, for witnesses `W` on the server's tape `R`: at statement `x`, after the history `h`, round
`h.length` leaves as its rep's Lean frame's release of the device round, at the rep's draws on the tape. -/
structure LeanRelease (W R : Type) where
  /-- round `j`'s rep, by the statement's schedule -/
  rep : Msg → ℕ → ℕ
  /-- rep `i`'s pad layout -/
  lay : Msg → ℕ → _root_.Flock.Zk.PadLayout
  /-- rep `i`'s mask slot, whose rank its frame checks at the opening (none: no check) -/
  slot : Msg → ℕ → Option MaskSlot
  /-- the bytes rep `i`'s frame reads from the operating system at its start (`IO.getRandomBytes`), on the server's
  tape -/
  draws : R → ℕ → ByteArray
  /-- the transport: the round's rep's frame state after the rounds of `h` (`Masker`: where its stream is, owing the
  coins of the last round); a function of the statement and the history, never of the tape. Its `stopped` is not
  read. -/
  prog : Msg → List (Msg × Msg) → Masker
  /-- the transport: the coins of the stream's answer to a round, as the frame takes them (`coins N`) -/
  coinsOf : Msg → Array _root_.Flock.F128
  /-- the device round at honest worker values, given the values the firewall fixes by its own computation, by name -/
  device : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg
  /-- the firewall's transcript after the rounds of `h`, whose own values the device round reads -/
  computed : Msg → W → R → List (Msg × Msg) → _root_.Flock.Firewall.Contract.Transcript
  /-- Rust's own checks on a worker's device round (`firewall::Checker` against the algebraic phase's skeleton,
  `ShadowLink::check` against the opening's shadow), which run before the Lean frame's pass decision (`pass`) and can only
  fail a round more: any decision at all -/
  cross : Msg → ℕ → W → R → List (Msg × Msg) → Msg → Bool
  /-- the device round the firewall sends in the device's place once a round failed: the skeleton's round with zero
  values in the algebraic phase, the shadow's round in the opening, given the values the firewall fixes by its own
  computation, by name -/
  scheduled : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg
  /-- round `j`'s coin count -/
  coins : Msg → ℕ → ℕ
  /-- round `j`'s release time -/
  sched : Msg → ℕ → ℕ
  stop : Msg

namespace LeanRelease

/-- Rep `i`'s draws on the tape `ρ`: its frame's bytes, parsed as the frame parses them. -/
def drawsAt {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (i : ℕ) : Draws :=
  Draws.ofBytes (L.lay x i) (L.draws ρ i)

/-- **The round's rep's configuration**: `Cfg.ofDraws` of its draws on the tape, as `flock-firewall frame` starts the
rep. -/
def cfg {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Cfg :=
  Cfg.ofDraws (L.lay x (L.rep x h.length)) (L.drawsAt x ρ (L.rep x h.length)) (L.slot x (L.rep x h.length))

/-- The configuration's pads are the frame's draws on the tape. -/
theorem cfg_pads {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) :
    (L.cfg x ρ h).h = (Draws.ofBytes (L.lay x (L.rep x h.length)) (L.draws ρ (L.rep x h.length))).h :=
  rfl

/-- The pads' root the frame inserts is its own commitment to its draws on the tape. -/
theorem cfg_root {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) :
    (L.cfg x ρ h).root = padsRoot (L.lay x (L.rep x h.length)) (L.drawsAt x ρ (L.rep x h.length)) :=
  rfl

/-- **The frame takes the stream's coins** for the last round of `h`, at the round's rep's configuration
(`Masker.answer`): the rep's state after them, or the frame's refusal of them. -/
def answered {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Except String Masker :=
  match h.getLast? with
  | some (_, cs) => (L.prog x h).answer (L.cfg x ρ h) (L.coinsOf cs)
  | none => pure (L.prog x h)

/-- **The frame's step** after `h` on the device round `v`, of a stopped rep (`st`) or not: at the rep's configuration,
it takes the stream's coins for the last round (`answered`), then the round (`Frame.round`), in the order
`flock-firewall frame` takes them (`coins`, then `round`, whose `fail` stops the rep). -/
def step {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool) (v : Msg) :
    Except String (ByteArray × Masker) := do
  let m ← L.answered x ρ h
  _root_.Flock.Firewall.Frame.round (L.cfg x ρ h) { m with stopped := st } ⟨v.toArray⟩ (L.coins x h.length)

/-- **The Lean frame's release** after `h` of the device round `v`, of a stopped rep (`st`) or not: the bytes the frame's
step returns, or `stop` where it refuses. -/
def release {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool) (v : Msg) : Msg :=
  match L.step x ρ h st v with
  | .ok (out, _) => out.data.toList
  | .error _ => L.stop

/-- **Whether the frame refuses the stream's coins** for the last round of `h` (`Masker.answer`): a zerocheck outer coin
`1`, level 0's `ρ_0, ρ_1` dependent over F128, the inner proof's `β = 0`, or coins for no round or of another count. -/
def refusesCoins {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Bool :=
  match L.answered x ρ h with
  | .ok _ => false
  | .error _ => true

/-- **Where the frame refuses the coins, the release is the stop**, whatever the round and whether the rep is stopped:
the coin stops are the frame's, from the coins alone. -/
theorem release_of_refusesCoins {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (v : Msg) (hr : L.refusesCoins x ρ h = true) : L.release x ρ h st v = L.stop := by
  unfold refusesCoins at hr
  cases ha : L.answered x ρ h with
  | ok m => rw [ha] at hr; cases hr
  | error e =>
    unfold release step
    rw [ha]
    rfl

/-- Whether the Lean frame takes the device round `v` after `h`. -/
def takes {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (v : Msg) : Bool :=
  match L.step x ρ h false v with
  | .ok _ => true
  | .error _ => false

/-- **The pass decision**, the Lean frame's (`Frame.check`, which `flock-firewall frame` answers each device round
with): after `h`, once the frame took the stream's last coins, the device round `v` against the firewall's scheduled
round in its place, with the values its transcript reports. -/
def pass {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) (v : Msg) : Bool :=
  match L.answered x ρ h with
  | .ok m =>
    match _root_.Flock.Firewall.Frame.check (L.cfg x ρ h) { m with stopped := false } ⟨v.toArray⟩
        ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) with
    | .ok (_, _, none) => true
    | _ => false
  | .error _ => false

/-- **The frame's check is the server's release**: after `h`, where the frame takes the stream's coins, the deployed
`Frame.check` of the device round `v` against the firewall's scheduled round decides `pass`, and releases the frame's
release of `v` where it passes and of the scheduled round as a stopped rep's where it fails: `toSrv`'s mask of the worker's
value `(v, false)`, and of the stand-in `fillOf`. -/
theorem check_release {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) (v : Msg)
    {m : Masker} (hm : L.answered x ρ h = .ok m) {out : ByteArray} {s' : Masker} {why : Option String}
    (hc : _root_.Flock.Firewall.Frame.check (L.cfg x ρ h) { m with stopped := false } ⟨v.toArray⟩
      ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) = .ok (out, s', why)) :
    L.pass x w ρ h v = why.isNone ∧
      out.data.toList = if why.isNone then L.release x ρ h false v
        else L.release x ρ h true (L.scheduled x w ρ h (computedOf (L.computed x w ρ h))) := by
  refine ⟨?_, ?_⟩
  · unfold pass
    rw [hm]
    simp only
    rw [hc]
    cases why <;> rfl
  · unfold release step
    rw [hm]
    cases why with
    | none =>
      show _ = match _root_.Flock.Firewall.Frame.round (L.cfg x ρ h) { m with stopped := false } ⟨v.toArray⟩
        (L.coins x h.length) with
        | .ok (out, _) => out.data.toList
        | .error _ => L.stop
      rw [check_none hc]
    | some e =>
      show _ = match _root_.Flock.Firewall.Frame.round (L.cfg x ρ h) { m with stopped := true }
        ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) with
        | .ok (out, _) => out.data.toList
        | .error _ => L.stop
      rw [check_some hc]

/-- **The deployed release as a ZK server**: one masked slot per round, whose worker value is the device round (with
whether it is the firewall's stand-in), whose mask is the Lean frame's release and whose check is Rust's checks, the
Lean frame's pass decision and the frame's own refusals, refusing a worker value that says it is a stand-in. -/
def toSrv {W R : Type} (L : LeanRelease W R) : ZkServer W R where
  Chk := Unit
  Val := Msg × Bool
  nS := fun _ _ => 1
  kind := fun _ _ _ => .masked
  cst := fun _ _ _ => ([], false)
  own := fun _ _ _ _ _ => ([], false)
  recomp := fun _ _ _ _ _ _ _ => ([], false)
  mask := fun x _ _ ρ h v => (L.release x ρ h v.2 v.1, v.2)
  rel := fun x j _ w ρ h _ _ v => !v.2 && L.cross x j w ρ h v.1 && L.pass x w ρ h v.1 && L.takes x ρ h v.1
  honest := fun x _ _ w ρ h => (L.device x w ρ h (computedOf (L.computed x w ρ h)), false)
  frame := fun _ _ vs => (vs ⟨0, Nat.zero_lt_one⟩).1
  stop := L.stop
  sched := L.sched

/-- **The server's frame at honest workers is the Lean frame's release** of the honest device round, by definition. -/
theorem refMsg_toSrv {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) :
    L.toSrv.refMsg x w ρ h = L.release x ρ h false (L.device x w ρ h (computedOf (L.computed x w ρ h))) :=
  rfl

/-- **The deployed release's stand-in**: once a round failed, the scheduled round, with the values the firewall's
transcript reports, as a stand-in. -/
def fillOf {W R : Type} (L : LeanRelease W R) : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val :=
  fun x _ _ w ρ h => (L.scheduled x w ρ h (computedOf (L.computed x w ρ h)), true)

/-- **A worker that sends no value for a round fails it**: an error, output that does not decode, a panic or silence. -/
theorem pass_of_none {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hn : wk x w ρ h 0 = none) : L.toSrv.pass wk x w ρ κ h = false := by
  show (List.range 1).all _ = false
  simp only [List.range_one, List.all_cons, List.all_nil, Bool.and_true]
  unfold ZkServer.check
  rw [hn]
  rfl

/-- **While the worker's run is live it sent its own value for the round**, which Rust's checks, the frame's pass decision
and the frame took. -/
theorem live_some {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : L.toSrv.live wk x w ρ κ h = true) :
    ∃ v, wk x w ρ h 0 = some (v, false) ∧ L.cross x h.length w ρ h v = true ∧ L.pass x w ρ h v = true ∧
      L.takes x ρ h v = true := by
  have hp := L.toSrv.pass_of_live wk x w ρ κ h hl (le_refl h.length)
  rw [List.take_length] at hp
  change (List.range 1).all _ = true at hp
  simp only [List.range_one, List.all_cons, List.all_nil, Bool.and_true] at hp
  unfold ZkServer.check at hp
  cases hw : wk x w ρ h 0 with
  | none =>
    rw [hw] at hp
    change false = true at hp
    cases hp
  | some u =>
    obtain ⟨v, b⟩ := u
    rw [hw] at hp
    change (!b && L.cross x h.length w ρ h v && L.pass x w ρ h v && L.takes x ρ h v) = true at hp
    simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at hp
    obtain ⟨⟨⟨hb, hcv⟩, hpv⟩, htv⟩ := hp
    subst hb
    exact ⟨v, rfl, hcv, hpv, htv⟩

/-- The deferred frame at the release is the Lean frame's release of the value `wkD` masks. -/
theorem frameD_toSrv {W R : Type} (L : LeanRelease W R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val) (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) {u : L.toSrv.Val}
    (hu : L.toSrv.wkD fill wk κ x w ρ h 0 = some u) :
    L.toSrv.frameD fill wk x w ρ κ h = L.release x ρ h u.2 u.1 := by
  show (L.toSrv.value x h.length 0 w ρ h (L.toSrv.copy (L.toSrv.wkD fill wk κ) x w ρ h)
    (L.toSrv.wkD fill wk κ x w ρ h 0)).1 = _
  unfold ZkServer.value
  rw [hu]
  rfl

/-- **A worker that fails at round `k` ends its live run there**: on every history through round `k`, the deferred release
is, below the last round, the Lean frame's release of the stand-in the server sends, as a stopped rep's, and at the last
round the stop. -/
theorem nextD_of_fail {W R : Type} (L : LeanRelease W R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val) (n : Msg → ℕ) (wk : L.toSrv.Worker) (x : Msg)
    (w : W) (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) {k : ℕ} (hk : k ≤ h.length)
    (hf : L.toSrv.pass wk x w ρ κ (h.take k) = false) :
    L.toSrv.nextD fill n wk x w ρ κ h =
      if h.length + 1 < n x then L.release x ρ h (fill x h.length 0 w ρ h).2 (fill x h.length 0 w ρ h).1
      else L.stop := by
  have hl : L.toSrv.live wk x w ρ κ h = false := by
    cases hl : L.toSrv.live wk x w ρ κ h with
    | false => rfl
    | true => rw [L.toSrv.pass_of_live wk x w ρ κ h hl hk] at hf; cases hf
  have hu : L.toSrv.wkD fill wk κ x w ρ h 0 = some (fill x h.length 0 w ρ h) := by
    simp only [ZkServer.wkD, hl, Bool.false_eq_true, ↓reduceIte]
  unfold ZkServer.nextD
  rw [hl, L.frameD_toSrv fill wk x w ρ κ h hu]
  simp only [Bool.false_eq_true, or_false]
  rfl

/-- **The rounds the worker passed**: after every history on which every round so far passed and whose last coins the
frame takes, the Lean frame's release of the worker's round is `P`'s message. -/
def LiveRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (wk : L.toSrv.Worker)
    (x : Msg) (w : W) : Prop :=
  ∀ ρ κ h v, L.toSrv.live wk x w ρ κ h = true → L.refusesCoins x ρ h = false → wk x w ρ h 0 = some (v, false) →
    (((P w).next x h ρ).eval H).1 = L.release x ρ h false v

/-- **The stand-in rounds**: after every history below the last round `n x - 1` on which a round failed and whose last
coins the frame takes, the Lean frame's release of the firewall's scheduled round, with the values `spec`, as a stopped
rep's, is `P`'s message. -/
def FillRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) (spec : String → Option ByteArray) : Prop :=
  ∀ ρ κ h, h.length + 1 < n x → L.toSrv.live wk x w ρ κ h = false → L.refusesCoins x ρ h = false →
    (((P w).next x h ρ).eval H).1 = L.release x ρ h true (L.scheduled x w ρ h spec)

/-- **The coin stops**: after every history whose last coins the frame refuses (`refusesCoins`), at a round the
deferred release releases, `P`'s message is the stop. It reads no worker value and nothing the firewall computes: the
frame's release there is the stop whatever the round (`release_of_refusesCoins`), and the frame decides it from the
coins alone. -/
def StopRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) : Prop :=
  ∀ ρ κ h, (h.length + 1 < n x ∨ L.toSrv.live wk x w ρ κ h = true) → L.refusesCoins x ρ h = true →
    (((P w).next x h ρ).eval H).1 = L.stop

/-- **The item-stream refinement** at `(x, w)`, the worker `wk`, the last round `n` and the specified values `spec`: after
every history the firewall's transcript reports every value `spec` fixes, and wherever the deferred release releases a
frame it is `P`'s message: where the frame takes the stream's last coins, at the worker's rounds while they pass
(`LiveRefine`) and at the firewall's scheduled rounds after one failed (`FillRefine`), and where it refuses them, the
stop (`StopRefine`). -/
def ItemsRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) (spec : String → Option ByteArray) : Prop :=
  (∀ ρ h, Reports spec (L.computed x w ρ h)) ∧ L.LiveRefine H P wk x w ∧ L.FillRefine H P n wk x w spec ∧
    L.StopRefine H P n wk x w

/-- **`hRef` at the deferred Lean release**: D8 (a) on every transcript the firewall keeps, and the item-stream
refinement, give `ZkServer.RefD` at the stand-in `fillOf`. -/
theorem refD {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) (spec : String → Option ByteArray)
    (hc : ∀ ρ h, _root_.Flock.Firewall.Contract.FirewallComputes spec (L.computed x w ρ h))
    (hI : L.ItemsRefine H P n wk x w spec) : L.toSrv.RefD H P L.fillOf n wk x w := fun ρ κ h hcond => by
  cases hl : L.toSrv.live wk x w ρ κ h with
  | true =>
    obtain ⟨v, hv, -, -, -⟩ := L.live_some wk x w ρ κ h hl
    have hu : L.toSrv.wkD L.fillOf wk κ x w ρ h 0 = some (v, false) := by
      simp only [ZkServer.wkD, hl, ↓reduceIte, hv]
      rfl
    rw [L.frameD_toSrv L.fillOf wk x w ρ κ h hu]
    cases hs : L.refusesCoins x ρ h with
    | true =>
      rw [L.release_of_refusesCoins x ρ h _ _ hs]
      exact hI.2.2.2 ρ κ h hcond hs
    | false => exact hI.2.1 ρ κ h v hl hs hv
  | false =>
    have hlt : h.length + 1 < n x := hcond.resolve_right (by rw [hl]; exact Bool.false_ne_true)
    have hu : L.toSrv.wkD L.fillOf wk κ x w ρ h 0 = some (L.fillOf x h.length 0 w ρ h) := by
      simp only [ZkServer.wkD, hl, Bool.false_eq_true, ↓reduceIte]
    rw [L.frameD_toSrv L.fillOf wk x w ρ κ h hu]
    cases hs : L.refusesCoins x ρ h with
    | true =>
      rw [L.release_of_refusesCoins x ρ h _ _ hs]
      exact hI.2.2.2 ρ κ h hcond hs
    | false =>
      show _ = L.release x ρ h true (L.scheduled x w ρ h (computedOf (L.computed x w ρ h)))
      rw [computedOf_eq (hc ρ h) (hI.1 ρ h)]
      exact hI.2.2.1 ρ κ h hlt hl hs

end LeanRelease

open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The ZK server at `flock-circuit --zk`'s reference session, released by the Lean frame at its own draws**: the
deployed release `L.toSrv`, whose masks are the frames' draws on the server's tape and whose coin stops and pass decision
are the frames', and which stops a session only at its `Finish` (`ZkServer.nextD` with the stand-in `LeanRelease.fillOf`; every worker
failure is a failed round, `LeanRelease.nextD_of_fail`), at any worker `wk` and any values `vs` whose round encoding is
causal, releases, on every event, at most twice the probability at the simulator that guesses whether the session stops
(`leaky1`, at `Flock.zk_server_ref`'s simulator, `refP`'s), plus `ε_zk + γ·Fork`, with no term for the firewall's checks.
It holds under `Claims.Hm96Hiding`, the session's six Props, D8 (a) on the firewall's transcripts (`FirewallComputes`)
and the item-stream refinement at `swP vs` (`LeanRelease.ItemsRefine`: `Reports`, `LiveRefine`, `FillRefine` and
`StopRefine`). The
stop leaks one bit a session; over a stream of `m` sessions with at most `b` stops, `log₂ ∑_{i ≤ b} C(m, i)` bits
(`stream_leak_deferred`). -/
theorem Flock.zk_server_lean (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (vs : S.SwVals) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hsc : S.SwCausal vs) (hT : S.TapeOK) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hR : Flock.R (FlockDeploy.deployZk rel).parse x w)
    (spec : String → Option ByteArray)
    (hFC : ∀ ρ h, _root_.Flock.Firewall.Contract.FirewallComputes spec (L.computed x w ρ h))
    (hI : L.ItemsRefine H (S.swP vs).prover (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk x w spec) {Θ : Type}
    (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : (S.refP.Nonce × S.refP.Rand) × L.toSrv.Chk =>
        Ev (L.toSrv.timed x (runStar H (L.toSrv.proverD L.fillOf (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).C x =>
          Ev (L.toSrv.timed x ((leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).out H Vs θ x c))) +
          (8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t * (Fork H ((S.swP vs).prover w) Vs θ x).Pr H) := by
  have h := L.toSrv.zk_deferred L.fillOf H (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (S.swP vs).prover _ _ _ _
    (Flock.zk_deployZk_sw H rel S vs t ht hT1 hδ₁ hS hV hD hc hsc hT).2 x w hR wk
    (L.refD H _ _ wk x w spec hFC hI) Vs θ hk Ev
  exact h

end Claims.ZKServer
