import Proofs.Flock.Claims.ZK.Blind
import Flock.Firewall.Frame
import Flock.Firewall.Assumptions
import Flock.Firewall.Coins

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
  the pads or the root. Only that state comes from the transport (`LeanRelease.prog`, `LeanRelease.fresh`): where the
  rep's stream is, a function of the statement and the history, never of the tape;
* each round's release (`LeanRelease.release`) is the frame's step: the frame of the rep that released the round before
  takes the coins the stream answered it (`LeanRelease.takeAt`: none at `Hello` and the link, whose coins are the
  session's; `Masker.answer`, which stops at a zerocheck outer coin `1`, at level 0's `ρ_0, ρ_1` dependent over F128
  and at the inner proof's `β = 0`: `answer_rho_dependent`, `answer_beta_zero`), then the device round at its own
  rep's state, at the rep's first round its start (`Frame.round`,
  which masks every absorbed value under the next pad, inserts the pads' label and root, and stops at a mask slot short
  of full rank), tagged `0` as the session sends a frame round. Every stop the coins decide is the frame's, from the
  coins alone, and where the frame refuses the coins the release is the stop whatever the round
  (`LeanRelease.release_of_refusesCoins`);
* the coins reach the frame only past the transport's coin check (`LeanRelease.opens`): `flock-firewall coins` on the
  server's own `Hello` at the tape's nonce, then `Hello`'s answer, then each opening at the transport's request
  (`LeanRelease.req`), as `coin_tree::Checked` asks it. Where it refuses, the release is the stop
  (`LeanRelease.answered`, `refusesCoins_of_not_opens`);
* a value carries whether it is the firewall's stand-in (`Val := Msg × Bool`), and the frame of a stopped rep releases the
  algebraic phase's values as their pads alone (`Masker.stopped`). The firewall's scheduled round is a stand-in. The
  worker's values are not, and the check refuses a worker value that says it is;
* the pass decision is the frame's (`Frame.check`, `LeanRelease.pass`): the firewall hands it each device round with
  the scheduled round in its place, and the device round passes where it equals the scheduled round byte for byte
  outside the algebraic phase's masked values and the frame takes it. A round that passes leaves as the frame's release
  of it, and one that fails as the frame's release of the scheduled round, as a stopped rep's
  (`LeanRelease.check_release`). Rust's own checks run first and can only fail a round more (`LeanRelease.cross`). The
  inner proof's rounds, which the firewall sends alone, are `check`'s against themselves (`check_self`).

`LeanRelease.toSrv` is it as a `ZkServer` with one slot per round, of the deployed session's kinds
(`LeanRelease.kindAt`):
* round `0`, the `Hello`, is the server's own at the tape's nonce (`LeanRelease.hello`), whatever the worker sends;
* the link (round `1`) and the `Finish` (round `LeanRelease.last`) are the firewall's own (`recomputed`,
  `LeanRelease.ownMsg`, the stop where the frame refuses the coins): the link's level-0 roots, which the firewall
  computes from the leaves under salts the worker never sees (`Firewall::new`'s recheck, sent by `Firewall::root` in
  `flock-circuit`), and `Final.encode` of the record and proof files the firewall builds itself (the session's Lean
  finish, `Flock.Firewall.Finish`, whose files `flock-circuit` sends).
  The worker's value there passes where it is the firewall's: a device root other than the recheck's fails the rep, and
  a worker whose proof the firewall's checks fail sends any other value;
* every other round is a frame round: the worker's value is its device round, the mask is the frame's release, and the
  check is Rust's checks, the frame's pass decision and the frame's own refusals.

The firewall stops a session only at its `Finish` (§10.1): once a round fails, it sends its own scheduled round in the
device's place (`LeanRelease.scheduled`), and the last round is the stop. That is the deferred server `ZkServer.nextD` at
`L.toSrv` with the stand-in `LeanRelease.fillOf`. Every way a worker fails is a round that fails:
* an error, output that does not decode, a panic or silence is no value (`LeanRelease.pass_of_none`);
* a tree at another size than the Lean finish's opening's, or any round Rust's checks, the frame's pass decision or the frame refuses, is
  a value that fails.

From that round on, below the last round, the release is the frame's release of the scheduled round as a stopped rep's,
and at the last round it is the stop (`LeanRelease.nextD_of_fail`), as the fixed firewall does. The `Finish` also stops
where the firewall's own `Finish` does, on workers' values that fail a constraint at the rep's drawn pads
(`LeanRelease.finStops`, `LeanRelease.refusesFinish`): the server refuses that round, so the session's one stop bit is
whether a round failed or its `Finish` stopped, `0` for honest workers, and a live run's `Finish` is one the firewall's
`Finish` takes (`LeanRelease.finStops_of_live`).

The device round and the scheduled round read, by name, the values the firewall fixes by its own computation, as its
release log reports them (`computedOf`): its level-0 recheck, its opening, `final_c'` and the inner proof. Five Props
give `ZkServer.RefD` at the deferred release (`LeanRelease.refD`, by the round's kind), with no assumption on what the
firewall computes:
* `LeanRelease.HelloRefine`: at every tape, the server's `Hello` is the prover's first message;
* `LeanRelease.OwnRefine`: at the link and the `Finish`, on every history where the deferred release releases a frame,
  the firewall's own message is the prover's;
* `LeanRelease.LiveRefine`: on every history at a frame round whose rounds all passed and whose last coins the frame
  takes, the frame's release of the worker's round is the prover's message;
* `LeanRelease.FillRefine`: on every history at a frame round below the last round on which a round failed and whose
  last coins the frame takes, the frame's release of the firewall's scheduled round, at the values the release log
  reports there, is the prover's message;
* `LeanRelease.StopRefine`: on every history at a frame round whose last coins the frame refuses, the prover's message
  is the stop. It reads no worker value and nothing the firewall computes.

So `Flock.zk_server_blind`'s bound holds at the deployed release, at any worker: one bit a session
(`Flock.zk_server_lean`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock.Firewall.Frame (Cfg Masker Draws MaskSlot padsRoot)

/-- The value the firewall's release log `t` reports fixing for `what` by its own computation (`By.recheck`, `By.shadow`,
`By.own`): the first such item's. -/
def computedOf (t : _root_.Flock.Firewall.Contract.ReleaseLog) (what : String) : Option ByteArray :=
  t.items.toList.findSome? fun
    | .fixed w .recheck want _ _ | .fixed w .shadow want _ _ | .fixed w .own want _ _ =>
      if w = what then some want else none
    | _ => none

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

/-- At the inner proof's `β` round, any other coin is taken, and kept as `β`. -/
theorem answer_beta_ne (c : Cfg) (s : Masker) (b : _root_.Flock.F128) (hn : s.owed = some 1)
    (hi : s.inner = some INNER_BETA) (hb : b.isZero = false) :
    s.answer c #[b] = .ok { s with owed := none, beta := b, inner := some (INNER_BETA + 1) } := by
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
open _root_.Flock.Firewall.Frame (check standIn passes firstDiff roundFree)

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

/-- A round checked against itself passes wherever the frame takes it: the inner proof's rounds, which the firewall sends
alone (`round N LEN`), are `check`'s with the round in its own place. -/
theorem check_self {c : Cfg} {s : Masker} {v : ByteArray} {n : Nat} {out : ByteArray} {s' : Masker}
    (hs : s.stopped = false) (h : _root_.Flock.Firewall.Frame.round c s v n = .ok (out, s')) :
    check c s v v n = .ok (out, s', none) := by
  have hr : ∃ fr, roundFree c s v n = .ok (out, s', fr) := by
    unfold _root_.Flock.Firewall.Frame.round at h
    cases hf : roundFree c s v n with
    | error e => rw [hf] at h; cases h
    | ok x =>
      obtain ⟨o, s'', fr⟩ := x
      rw [hf] at h
      cases h
      exact ⟨fr, rfl⟩
  obtain ⟨fr, hfr⟩ := hr
  have hp : passes c s v v n = .ok none := by
    unfold passes
    rw [hfr]
    simp only [firstDiff, bne_self_eq_false, Bool.and_false]
    have hn : (if (v == v) = true then (none : Option Nat) else List.find? (fun _ => false) (List.range v.size)) = none := by
      split
      · rfl
      · exact List.find?_eq_none.mpr fun _ _ => Bool.false_ne_true
    simp only [hn, Option.map_none]
    rfl
  unfold check
  simp only [hs, Bool.false_eq_true, ↓reduceIte]
  rw [hp]
  show (match _root_.Flock.Firewall.Frame.round c s v n with
    | .ok (out, s') => (Except.ok (out, s', none) : Except String _)
    | .error e => if e.startsWith _root_.Flock.Firewall.Frame.STOP then throw e else standIn c s v n e) = _
  rw [h]

end PassDecision

/-! ## The transport's coin check

The coin check (`Flock.Firewall.Coins`) reads the session's `Hello` (its tree, link points and nonce), checks `Hello`'s
answer and each opening, and hands the frame no coins past its first refusal: the session loop (`flock-firewall session`)
at the `Hello` its setup's values give (`Flock.Firewall.Session.helloOf`), `coin_tree::Checked` asking
`flock-firewall coins` at the parse of the `Hello`'s bytes (`coinHello`). -/

/-- `flock-firewall coins`' reading of the server's `Hello` `hb`, `Req::Hello`'s frame (its tag, its `u32le` length, the
bytes): the bytes as UTF-8, parsed (`Hello.parse`). -/
def coinHello (hb : Msg) : Except String _root_.Flock.Firewall.Coins.Hello :=
  match String.fromUTF8? ⟨(hb.drop 5).toArray⟩ with
  | some s => _root_.Flock.Firewall.Coins.Hello.parse s
  | none => .error "a Hello that is not UTF-8"

/-- **The coin check's verdict** on a session: its reading of the `Hello`, `hl`, then `Hello`'s answer `a`
(`Session.start`), then the openings `os` in order (`Session.openAll`). -/
def coinVerdict (hl : Except String _root_.Flock.Firewall.Coins.Hello) (a : Msg) (os : List (String × ℕ × Msg)) :
    Except String _root_.Flock.Firewall.Coins.Session :=
  hl.bind fun hl => (_root_.Flock.Firewall.Coins.Session.start hl a).bind fun S => S.openAll os

/-- The openings `ms` as `flock-firewall coins` takes them, from the `j`-th on: each with the stream and coin count of
its request (`req`). -/
def reqsFrom (req : ℕ → String × ℕ) : ℕ → List Msg → List (String × ℕ × Msg)
  | _, [] => []
  | j, m :: ms => ((req j).1, (req j).2, m) :: reqsFrom req (j + 1) ms

theorem reqsFrom_map (req : ℕ → String × ℕ) :
    ∀ (j : ℕ) (ms : List Msg), (reqsFrom req j ms).map (fun o => o.2.2) = ms
  | _, [] => rfl
  | j, m :: ms => congrArg (m :: ·) (reqsFrom_map req (j + 1) ms)

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
  /-- the transport: the frame state of the rep that released the last round of `h`, after it (`Masker`: where its
  stream is, owing that round's coins); a function of the statement and the history, never of the tape. Its `stopped`
  is not read. -/
  prog : Msg → List (Msg × Msg) → Masker
  /-- the transport: the coins the frame takes (`coins N`) after the rounds of `h`, for the stream's last answer (the
  session loop's `Frames.feedOf` of it, `LeanRelease.loop`) -/
  coinsOf : Msg → List (Msg × Msg) → Array _root_.Flock.F128
  /-- the transport: the frame state of the rep of the round after `h` before that round, where it is the rep's first:
  no coins of its own, after an earlier rep's algebraic-phase coins (`Frame.step`'s `prior`, rep 1's, as
  `Frames.roundPlace` hands them), or the frame's refusal of them; a function of the statement and the history, never of
  the tape -/
  fresh : Msg → List (Msg × Msg) → Except String Masker
  /-- the device round at honest worker values, given the values the firewall fixes by its own computation, by name -/
  device : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg
  /-- the firewall's release log after the rounds of `h`, whose own values the device round reads -/
  computed : Msg → W → R → List (Msg × Msg) → _root_.Flock.Firewall.Contract.ReleaseLog
  /-- Rust's own checks on a worker's rep beside the Lean frame's pass decision (`pass`): the worker failing, the
  device's level-0 root, the constraints on the pads, the end check on the returned proof and Rust's masking
  cross-checks. Each can only fail a round more, the failed rep's rounds reaching the loop as no device round, which
  the frame fails: any decision at all -/
  cross : Msg → ℕ → W → R → List (Msg × Msg) → Msg → Bool
  /-- the device round the firewall sends in the device's place once a round failed: the skeleton's round with zero
  values in the algebraic phase, the Lean finish's round in the opening, and at the inner proof's rounds, which the firewall
  sends alone, the firewall's own round itself (`check_self`), given the values the firewall fixes by its own
  computation, by name -/
  scheduled : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg
  /-- round `j`'s coin count -/
  coins : Msg → ℕ → ℕ
  /-- round `j`'s release time -/
  sched : Msg → ℕ → ℕ
  stop : Msg
  /-- round `0`, the server's own `Hello` at the tape's nonce (`CoinLayer.helloD` at the deployed coin layer) -/
  hello : Msg → R → Msg
  /-- the transport: what its coin check reads of the session's `Hello`, the tree, the session stream's coin count and the
  nonce (`Flock.Firewall.Session.helloOf` from the setup's values in the session loop; `coinHello` of `hello`, a parse of
  its bytes, under `coin_tree::Checked`), or why it reads none -/
  helloChk : Msg → R → Except String _root_.Flock.Firewall.Coins.Hello
  /-- the transport: the stream and coin count it requests the `j`-th opening at, by the statement's schedule
  (`coin_tree::Checked`'s request at `Req::Commit` and `Req::Round`) -/
  req : Msg → ℕ → String × ℕ
  /-- round `1`, the link, the firewall's own: level 0's roots as its recheck commits them, from the leaves under the
  salts on its tape (`Firewall::root`, `Firewall::new` in `flock-circuit`), never the worker's -/
  link : Msg → W → R → List (Msg × Msg) → Msg
  /-- the record and proof files the firewall builds itself at the session's `Finish` (the session's Lean finish,
  `Flock.Firewall.Finish`) -/
  files : Msg → W → R → List (Msg × Msg) → String × Array ByteArray
  /-- whether the firewall's own `Finish` stops (on the constraints the Lean finish replays: `repFw`'s `failingOn`, and
  `flock-circuit`'s `constraints_failing_on_pads` on those `innerLive` answers): where a
  constraint fails at the rep's drawn pads, a function of the witness and the workers' values, which the server's tape
  fixes -/
  finStops : Msg → W → R → List (Msg × Msg) → Bool
  /-- the session's last round, its `Finish` -/
  last : Msg → ℕ

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

/-- **Whether the transport's coin check passes the history `h`** at the tape `ρ`: the coin check at its reading of the
session's `Hello` at the tape (`helloChk`, at the tape's nonce), then `Hello`'s answer, then each later answer as the
opening of the transport's request (`coinVerdict`). A history with no opening passes. The deployed transport checks `Hello`'s
answer as it arrives (`coin_tree::Checked`) and stops before the link where it is malformed; the committed-coin
prover's link (`strat` at no opening) reads nothing of the answer, so a release that stopped there could not be its
message (`OwnRefine`), and here the stop comes at the first opening instead, which a malformed answer fails
(`CoinOpen.verdict_isOk`). On such a history the deployed release is this one's cut after the answer, with the stop in
the link's place: a function of the verifier's own answer, which is the coin layer's and not the witness's, so it adds
nothing for the stop-leak bound to charge. -/
def opens {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) : List (Msg × Msg) → Bool
  | (_, a) :: o :: ops => (coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 ((o :: ops).map Prod.snd))).isOk
  | _ => true

/-- Round `j`'s rep's configuration: `Cfg.ofDraws` of its draws on the tape. -/
def cfgAt {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (j : ℕ) : Cfg :=
  Cfg.ofDraws (L.lay x (L.rep x j)) (L.drawsAt x ρ (L.rep x j)) (L.slot x (L.rep x j))

/-- **Who takes the coins of round `n - 1`**, as the session loop routes them (`Session.relay`): `Hello`'s and the
`Commit`'s go to the session, not to a frame (`Sessions.link`), so round `n` starts at its rep's first-round state
`fresh`; else the frame of the rep that released round `n - 1`, at its state `prog` and configuration `cP`
(`Frames.coins`), whose refusal stops the session. Where round `n` is that rep's (`same`), it is checked at the state
the coins leave, at its configuration `cN`; else it is the first round of its rep, which takes no coins, at `fresh`. -/
def takeAt (n : ℕ) (same : Bool) (cN cP : Cfg) (prog : Masker) (fresh : Except String Masker)
    (cs : Array _root_.Flock.F128) : Except String Masker :=
  if n ≤ 2 then fresh else if same then prog.answer cN cs else prog.answer cP cs >>= fun _ => fresh

/-- **The frame takes the stream's coins** for the last round of `h` (`takeAt`): the state the round after `h` is
checked at, or the frame's refusal of the coins. -/
def frameAnswer {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Except String Masker :=
  match h.getLast? with
  | some _ =>
    takeAt h.length (decide (L.rep x h.length = L.rep x (h.length - 1))) (L.cfg x ρ h) (L.cfgAt x ρ (h.length - 1))
      (L.prog x h) (L.fresh x h) (L.coinsOf x h)
  | none => pure (L.prog x h)

/-- The transport's stop where its coin check refuses. -/
def coinRefused : String := _root_.Flock.Firewall.Coins.STOP ++ "the coin check refused an opening"

/-- **The coins the frame is handed and takes** after `h`: where the transport's coin check passes the history
(`opens`), the frame's taking of the stream's last coins (`frameAnswer`); else the transport's stop, as the transport
hands the frame no coins past the check's refusal. -/
def answered {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Except String Masker :=
  if L.opens x ρ h then L.frameAnswer x ρ h else .error coinRefused

theorem answered_of_opens {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (ho : L.opens x ρ h = true) : L.answered x ρ h = L.frameAnswer x ρ h := by
  unfold answered
  exact if_pos ho

theorem answered_of_not_opens {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (ho : L.opens x ρ h = false) : L.answered x ρ h = .error coinRefused := by
  unfold answered
  exact if_neg (by rw [ho]; exact Bool.false_ne_true)

/-- Past `Hello`'s answer and one opening, where the coin check reads the session's `Hello` as `hl`, the check is
`Session.start` at the answer, then `Session.openAll` over the openings. -/
theorem opens_of_hello {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (m₀ a : Msg) (o : Msg × Msg)
    (ops : List (Msg × Msg)) {hl : _root_.Flock.Firewall.Coins.Hello} (hp : L.helloChk x ρ = .ok hl) :
    L.opens x ρ ((m₀, a) :: o :: ops) =
      ((_root_.Flock.Firewall.Coins.Session.start hl a).bind fun S =>
        S.openAll (reqsFrom (L.req x) 0 ((o :: ops).map Prod.snd))).isOk := by
  show (coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 ((o :: ops).map Prod.snd))).isOk = _
  unfold coinVerdict
  rw [hp]
  try rfl

/-- **The frame's step** after `h` on the device round `v`, of a stopped rep (`st`) or not: at the rep's configuration,
it takes the stream's coins for the last round (`answered`), then the round (`Frame.round`), in the order
`flock-firewall frame` takes them (`coins`, then `round`, whose `fail` stops the rep). -/
def step {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool) (v : Msg) :
    Except String (ByteArray × Masker) := do
  let m ← L.answered x ρ h
  _root_.Flock.Firewall.Frame.round (L.cfg x ρ h) { m with stopped := st } ⟨v.toArray⟩ (L.coins x h.length)

/-- **The Lean frame's release** after `h` of the device round `v`, of a stopped rep (`st`) or not: the bytes the frame's
step returns, tagged `0` as a round message, or `stop` where it refuses. -/
def release {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool) (v : Msg) : Msg :=
  match L.step x ρ h st v with
  | .ok (out, _) => 0 :: out.data.toList
  | .error _ => L.stop

/-- **Whether the frame refuses the stream's coins** for the last round of `h`: the transport's coin check refuses the
history (`opens`), or the frame refuses the coins (`Masker.answer`: a zerocheck outer coin `1`, level 0's `ρ_0, ρ_1`
dependent over F128, the inner proof's `β = 0`, or coins for no round or of another count). -/
def refusesCoins {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Bool :=
  match L.answered x ρ h with
  | .ok _ => false
  | .error _ => true

/-- **Where the transport's coin check refuses the history, so does the release.** -/
theorem refusesCoins_of_not_opens {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (ho : L.opens x ρ h = false) : L.refusesCoins x ρ h = true := by
  have ha := L.answered_of_not_opens x ρ h ho
  unfold refusesCoins
  cases hb : L.answered x ρ h with
  | ok m => rw [ha] at hb; cases hb
  | error e => rfl

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
round in its place, with the values its release log reports. -/
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
      0 :: out.data.toList = if why.isNone then L.release x ρ h false v
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
        | .ok (out, _) => 0 :: out.data.toList
        | .error _ => L.stop
      rw [check_none hc]
    | some e =>
      show _ = match _root_.Flock.Firewall.Frame.round (L.cfg x ρ h) { m with stopped := true }
        ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) with
        | .ok (out, _) => 0 :: out.data.toList
        | .error _ => L.stop
      rw [check_some hc]

/-- **Round `j`'s kind**: round `0`, the `Hello`, is the server's own; the link (round `1`) and the `Finish` (the last
round) are the firewall's own, which a worker's value passes only where it is the firewall's byte for byte; every other
round is a frame round, the worker's device round masked by the Lean frame. -/
def kindAt {W R : Type} (L : LeanRelease W R) (x : Msg) (j : ℕ) : Kind :=
  if j = 0 then .own else if j = 1 ∨ j = L.last x then .recomputed else .masked

theorem kindAt_own {W R : Type} {L : LeanRelease W R} {x : Msg} {j : ℕ} (h : L.kindAt x j = .own) : j = 0 := by
  unfold kindAt at h
  split_ifs at h with h0 <;> first | exact h0 | cases h

theorem kindAt_zero {W R : Type} (L : LeanRelease W R) (x : Msg) : L.kindAt x 0 = .own := rfl

theorem kindAt_recomputed {W R : Type} {L : LeanRelease W R} {x : Msg} {j : ℕ} (h : L.kindAt x j = .recomputed) :
    j = 1 ∨ j = L.last x := by
  unfold kindAt at h
  split_ifs at h with h0 h1
  exact h1

theorem kindAt_masked {W R : Type} {L : LeanRelease W R} {x : Msg} {j : ℕ} (h : L.kindAt x j = .masked) :
    j ≠ 0 ∧ j ≠ 1 ∧ j ≠ L.last x := by
  unfold kindAt at h
  split_ifs at h with h0 h1
  exact ⟨h0, fun e => h1 (Or.inl e), fun e => h1 (Or.inr e)⟩

/-- **The firewall's own message** after `h`, at its link and its `Finish`: the stop where the frame refuses the
stream's last coins; at the last round `Final.encode` of the record and proof files it builds itself; else its link. -/
def ownMsg {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) : Msg :=
  if L.refusesCoins x ρ h then L.stop
  else if h.length = L.last x then FlockDeploy.Final.encode ⟨x, (L.files x w ρ h).1, (L.files x w ρ h).2⟩
  else L.link x w ρ h

/-- **The model's stop at the `Finish`**, its one definition: the server refuses the last round where the firewall's own
`Finish` stops (`finStops`). With it the session stops at its `Finish` where a round failed or the workers' values fail a
constraint at the drawn pads, still one bit a session (`ZkServer.zk_deferred`'s factor `2`), `0` for honest workers. A
premise on the workers' values instead would make this `false`. -/
def refusesFinish {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) : Bool :=
  decide (h.length = L.last x) && L.finStops x w ρ h

/-- **The deployed release as a ZK server**: one slot per round. Round `0` is the server's own `Hello`; the link and
the `Finish` are the firewall's own (`ownMsg`), which a worker's value passes only where it is the firewall's; at a
frame round the worker's value is the device round (with whether it is the firewall's stand-in), the mask is the Lean
frame's release and the check is Rust's checks, the Lean frame's pass decision and the frame's own refusals, refusing a
worker value that says it is a stand-in. The session ends at its `Finish` (§10.1), so the firewall takes no frame round
past it: a worker's value there fails. The server refuses the `Finish` where the firewall's own `Finish` stops
(`refusesFinish`). -/
def toSrv {W R : Type} (L : LeanRelease W R) : ZkServer W R where
  Chk := Unit
  Val := Msg × Bool
  nS := fun _ _ => 1
  kind := fun x j _ => L.kindAt x j
  cst := fun _ _ _ => ([], false)
  own := fun x _ _ ρ _ => (L.hello x ρ, false)
  recomp := fun x _ _ w ρ h _ => (L.ownMsg x w ρ h, false)
  mask := fun x _ _ ρ h v => (L.release x ρ h v.2 v.1, v.2)
  rel := fun x j _ w ρ h _ _ v =>
    !v.2 && decide (h.length < L.last x) && L.cross x j w ρ h v.1 && L.pass x w ρ h v.1 && L.takes x ρ h v.1
  honest := fun x j _ w ρ h => if L.kindAt x j = .masked then
    (L.device x w ρ h (computedOf (L.computed x w ρ h)), false) else (L.ownMsg x w ρ h, false)
  frame := fun _ _ vs => (vs ⟨0, Nat.zero_lt_one⟩).1
  stop := L.stop
  sched := L.sched
  refuses := L.refusesFinish

/-- A live run's `Finish` is one the firewall's own `Finish` does not stop. -/
theorem finStops_of_live {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : L.toSrv.live wk x w ρ κ h = true) (he : h.length = L.last x) :
    L.finStops x w ρ h = false := by
  have hp := L.toSrv.pass_of_live wk x w ρ κ h hl (le_refl h.length)
  rw [List.take_length] at hp
  have hr : L.toSrv.refuses x w ρ h = false := by simpa using ((Bool.and_eq_true _ _).mp hp).1
  change (decide (h.length = L.last x) && L.finStops x w ρ h) = false at hr
  simpa [he] using hr

/-- **The server's frame at honest workers is the Lean frame's release** of the honest device round at a frame round. -/
theorem refMsg_toSrv {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg))
    (hk : L.kindAt x h.length = .masked) :
    L.toSrv.refMsg x w ρ h = L.release x ρ h false (L.device x w ρ h (computedOf (L.computed x w ρ h))) := by
  show (L.toSrv.value x h.length 0 w ρ h (L.toSrv.hcopy x w ρ h) (some (L.toSrv.honest x h.length 0 w ρ h))).1 = _
  unfold ZkServer.value
  rw [show L.toSrv.kind x h.length 0 = .masked from hk]
  have he : L.toSrv.honest x h.length 0 w ρ h = (L.device x w ρ h (computedOf (L.computed x w ρ h)), false) :=
    ite_eq_left hk
  show L.release x ρ h (L.toSrv.honest x h.length 0 w ρ h).2 (L.toSrv.honest x h.length 0 w ρ h).1 = _
  rw [he]

/-- **The deployed release's stand-in**: once a round failed, the scheduled round, with the values the firewall's
release log reports, as a stand-in. -/
def fillOf {W R : Type} (L : LeanRelease W R) : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val :=
  fun x _ _ w ρ h => (L.scheduled x w ρ h (computedOf (L.computed x w ρ h)), true)

/-- **A worker that sends no value for a round past the `Hello` fails it**: an error, output that does not decode, a
panic or silence. -/
theorem pass_of_none {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length ≠ .own) (hn : wk x w ρ h 0 = none) :
    L.toSrv.pass wk x w ρ κ h = false := by
  show (!L.toSrv.refuses x w ρ h && (List.range 1).all _) = false
  simp only [List.range_one, List.all_cons, List.all_nil, Bool.and_true]
  refine Bool.and_eq_false_iff.mpr (Or.inr ?_)
  unfold ZkServer.check
  rw [hn, show L.toSrv.kind x h.length 0 = L.kindAt x h.length from rfl]
  cases hkk : L.kindAt x h.length with
  | own => exact absurd hkk hk
  | fixed => rfl
  | recomputed => rfl
  | masked => rfl

/-- **While the worker's run is live it sent its own value for each frame round** below the `Finish`, which Rust's
checks, the frame's pass decision and the frame took. -/
theorem live_some_lt {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked)
    (hl : L.toSrv.live wk x w ρ κ h = true) :
    ∃ v, wk x w ρ h 0 = some (v, false) ∧ h.length < L.last x ∧ L.cross x h.length w ρ h v = true ∧
      L.pass x w ρ h v = true ∧ L.takes x ρ h v = true := by
  have hp := L.toSrv.pass_of_live wk x w ρ κ h hl (le_refl h.length)
  rw [List.take_length] at hp
  replace hp := ZkServer.slots_of_pass hp
  change (List.range 1).all _ = true at hp
  simp only [List.range_one, List.all_cons, List.all_nil, Bool.and_true] at hp
  unfold ZkServer.check at hp
  rw [show L.toSrv.kind x h.length 0 = .masked from hk] at hp
  cases hw : wk x w ρ h 0 with
  | none =>
    rw [hw] at hp
    change false = true at hp
    cases hp
  | some u =>
    obtain ⟨v, b⟩ := u
    rw [hw] at hp
    change (!b && decide (h.length < L.last x) && L.cross x h.length w ρ h v && L.pass x w ρ h v &&
      L.takes x ρ h v) = true at hp
    simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_true_eq] at hp
    obtain ⟨⟨⟨⟨hb, hlt⟩, hcv⟩, hpv⟩, htv⟩ := hp
    subst hb
    exact ⟨v, rfl, hlt, hcv, hpv, htv⟩

/-- **While the worker's run is live it sent its own value for each frame round**, which Rust's checks, the frame's pass
decision and the frame took. -/
theorem live_some {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked)
    (hl : L.toSrv.live wk x w ρ κ h = true) :
    ∃ v, wk x w ρ h 0 = some (v, false) ∧ L.cross x h.length w ρ h v = true ∧ L.pass x w ρ h v = true ∧
      L.takes x ρ h v = true := by
  obtain ⟨v, hv, -, hc, hp, ht⟩ := L.live_some_lt wk x w ρ κ h hk hl
  exact ⟨v, hv, hc, hp, ht⟩

/-- **A live run's frame rounds are below the `Finish`.** -/
theorem live_lt_last {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked)
    (hl : L.toSrv.live wk x w ρ κ h = true) : h.length < L.last x := by
  obtain ⟨-, -, hlt, -⟩ := L.live_some_lt wk x w ρ κ h hk hl
  exact hlt

/-- The deferred frame at a frame round is the Lean frame's release of the value `wkD` masks. -/
theorem frameD_toSrv {W R : Type} (L : LeanRelease W R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val) (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked) {u : L.toSrv.Val}
    (hu : L.toSrv.wkD fill wk κ x w ρ h 0 = some u) :
    L.toSrv.frameD fill wk x w ρ κ h = L.release x ρ h u.2 u.1 := by
  show (L.toSrv.value x h.length 0 w ρ h (L.toSrv.copy (L.toSrv.wkD fill wk κ) x w ρ h)
    (L.toSrv.wkD fill wk κ x w ρ h 0)).1 = _
  unfold ZkServer.value
  rw [show L.toSrv.kind x h.length 0 = .masked from hk, hu]
  rfl

/-- **The deferred frame at the `Hello` is the server's own**, whatever the worker sends. -/
theorem frameD_own {W R : Type} (L : LeanRelease W R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val) (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .own) :
    L.toSrv.frameD fill wk x w ρ κ h = L.hello x ρ := by
  show (L.toSrv.value x h.length 0 w ρ h (L.toSrv.copy (L.toSrv.wkD fill wk κ) x w ρ h)
    (L.toSrv.wkD fill wk κ x w ρ h 0)).1 = _
  unfold ZkServer.value
  rw [show L.toSrv.kind x h.length 0 = .own from hk]
  rfl

/-- **The deferred frame at the link and the `Finish` is the firewall's own message**, whatever the worker sends. -/
theorem frameD_recomp {W R : Type} (L : LeanRelease W R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val) (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .recomputed) :
    L.toSrv.frameD fill wk x w ρ κ h = L.ownMsg x w ρ h := by
  show (L.toSrv.value x h.length 0 w ρ h (L.toSrv.copy (L.toSrv.wkD fill wk κ) x w ρ h)
    (L.toSrv.wkD fill wk κ x w ρ h 0)).1 = _
  unfold ZkServer.value
  rw [show L.toSrv.kind x h.length 0 = .recomputed from hk]
  rfl

/-- **A worker that fails at round `k` ends its live run there**: on every history through round `k`, the deferred release
at a frame round is, below the last round, the Lean frame's release of the stand-in the server sends, as a stopped
rep's, and at the last round the stop. -/
theorem nextD_of_fail {W R : Type} (L : LeanRelease W R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val) (n : Msg → ℕ) (wk : L.toSrv.Worker) (x : Msg)
    (w : W) (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hm : L.kindAt x h.length = .masked) {k : ℕ}
    (hk : k ≤ h.length) (hf : L.toSrv.pass wk x w ρ κ (h.take k) = false) :
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
  rw [hl, L.frameD_toSrv fill wk x w ρ κ h hm hu]
  simp only [Bool.false_eq_true, or_false]
  rfl

/-- **The `Hello`**: at every tape, the server's own `Hello` is `P`'s first message. -/
def HelloRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (x : Msg) (w : W) : Prop :=
  ∀ ρ, (((P w).next x [] ρ).eval H).1 = L.hello x ρ

/-- **The firewall's own rounds**: after every history of the deferred release's own run (`ZkServer.OwnHist`) at the
link or the `Finish` on which it releases a frame, the firewall's own message there (`ownMsg`) is `P`'s message. -/
def OwnRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) : Prop :=
  ∀ ρ κ h, L.kindAt x h.length = .recomputed → (h.length + 1 < n x ∨ L.toSrv.live wk x w ρ κ h = true) →
    L.toSrv.OwnHist L.fillOf wk x w ρ κ h → (((P w).next x h ρ).eval H).1 = L.ownMsg x w ρ h

/-- **The frame rounds the worker passed**: after every history at a frame round on which every round so far passed and
whose last coins the frame takes, the Lean frame's release of the worker's round is `P`'s message. -/
def LiveRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (wk : L.toSrv.Worker)
    (x : Msg) (w : W) : Prop :=
  ∀ ρ κ h v, L.kindAt x h.length = .masked → L.toSrv.live wk x w ρ κ h = true → L.refusesCoins x ρ h = false →
    wk x w ρ h 0 = some (v, false) → (((P w).next x h ρ).eval H).1 = L.release x ρ h false v

/-- **The stand-in rounds**: after every history at a frame round below the last round `n x - 1` on which a round failed
and whose last coins the frame takes, the Lean frame's release of the firewall's scheduled round, at the values the
firewall's release log reports fixing there (`computedOf`), as a stopped rep's, is `P`'s message. -/
def FillRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) : Prop :=
  ∀ ρ κ h, L.kindAt x h.length = .masked → h.length + 1 < n x → L.toSrv.live wk x w ρ κ h = false →
    L.refusesCoins x ρ h = false →
    (((P w).next x h ρ).eval H).1 = L.release x ρ h true (L.scheduled x w ρ h (computedOf (L.computed x w ρ h)))

/-- **The coin stops**: after every history at a frame round whose last coins the frame refuses (`refusesCoins`), at a
round the deferred release releases, `P`'s message is the stop. It reads no worker value and nothing the firewall
computes: the frame's release there is the stop whatever the round (`release_of_refusesCoins`), and the frame decides it
from the coins alone. -/
def StopRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) : Prop :=
  ∀ ρ κ h, L.kindAt x h.length = .masked → (h.length + 1 < n x ∨ L.toSrv.live wk x w ρ κ h = true) →
    L.refusesCoins x ρ h = true → (((P w).next x h ρ).eval H).1 = L.stop

/-- **The item-stream refinement** at `(x, w)`, the worker `wk` and the last round `n`: wherever the deferred release
releases a frame it is `P`'s message: the server's `Hello` (`HelloRefine`), the firewall's own link and `Finish`
(`OwnRefine`), and at a frame round, where the frame takes the stream's last coins, the worker's rounds while they pass
(`LiveRefine`) and the firewall's scheduled rounds after one failed, at the values its release log reports
(`FillRefine`), and where it refuses them, the stop (`StopRefine`). -/
def ItemsRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) : Prop :=
  L.HelloRefine H P x w ∧ L.OwnRefine H P n wk x w ∧ L.LiveRefine H P wk x w ∧ L.FillRefine H P n wk x w ∧
    L.StopRefine H P n wk x w

/-- **`hRef` at the deferred Lean release**: the item-stream refinement gives `ZkServer.RefDOwn` at the stand-in
`fillOf`. -/
theorem refD {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (hI : L.ItemsRefine H P n wk x w) : L.toSrv.RefDOwn H P L.fillOf n wk x w := fun ρ κ h hown hcond => by
  cases hk : L.kindAt x h.length with
  | fixed => unfold kindAt at hk; split_ifs at hk
  | own =>
    rw [L.frameD_own L.fillOf wk x w ρ κ h hk]
    have h0 : h = [] := List.eq_nil_of_length_eq_zero (kindAt_own hk)
    subst h0
    exact hI.1 ρ
  | recomputed =>
    rw [L.frameD_recomp L.fillOf wk x w ρ κ h hk]
    exact hI.2.1 ρ κ h hk hcond hown
  | masked =>
    cases hl : L.toSrv.live wk x w ρ κ h with
    | true =>
      obtain ⟨v, hv, -, -, -⟩ := L.live_some wk x w ρ κ h hk hl
      have hu : L.toSrv.wkD L.fillOf wk κ x w ρ h 0 = some (v, false) := by
        simp only [ZkServer.wkD, hl, ↓reduceIte, hv]
        rfl
      rw [L.frameD_toSrv L.fillOf wk x w ρ κ h hk hu]
      cases hs : L.refusesCoins x ρ h with
      | true =>
        rw [L.release_of_refusesCoins x ρ h _ _ hs]
        exact hI.2.2.2.2 ρ κ h hk hcond hs
      | false => exact hI.2.2.1 ρ κ h v hk hl hs hv
    | false =>
      have hlt : h.length + 1 < n x := hcond.resolve_right (by rw [hl]; exact Bool.false_ne_true)
      have hu : L.toSrv.wkD L.fillOf wk κ x w ρ h 0 = some (L.fillOf x h.length 0 w ρ h) := by
        simp only [ZkServer.wkD, hl, Bool.false_eq_true, ↓reduceIte]
      rw [L.frameD_toSrv L.fillOf wk x w ρ κ h hk hu]
      cases hs : L.refusesCoins x ρ h with
      | true =>
        rw [L.release_of_refusesCoins x ρ h _ _ hs]
        exact hI.2.2.2.2 ρ κ h hk hcond hs
      | false => exact hI.2.2.2.1 ρ κ h hk hlt hl hs

end LeanRelease

open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The ZK server at `flock-circuit --zk`'s reference session, released by the Lean frame at its own draws**: the
deployed release `L.toSrv`, whose masks are the frames' draws on the server's tape and whose coin stops and pass decision
are the frames', and which stops a session only at its `Finish` (`ZkServer.nextD` with the stand-in `LeanRelease.fillOf`; every worker
failure is a failed round, `LeanRelease.nextD_of_fail`; the server refuses the `Finish` where the firewall's own `Finish`
stops, `LeanRelease.refusesFinish`), at any worker `wk` and any values `vs` whose round encoding is
causal, releases, on every event, at most twice the probability at the simulator that guesses whether the session stops
(`leaky1`, at `Flock.zk_server_ref`'s simulator, `refP`'s), plus `ε_zk + γ·Fork`, with no term for the firewall's checks.
It holds under `Claims.Hm96Hiding`, the session's six Props and the item-stream refinement at `swP vs`
(`LeanRelease.ItemsRefine`: `HelloRefine`, `OwnRefine`, `LiveRefine`, `FillRefine` at the values the firewall's
release log reports, and `StopRefine`),
with no assumption on what the firewall computes. The stop leaks one bit a session, the factor `2`:
`b = 1[a round failed ∨ L.finStops]`, whether the session stopped at its `Finish`, where `L.finStops` is the firewall's
own `Finish` stopping on workers' values that fail a constraint at the rep's drawn pads; `b = 0` for honest workers.
Over a stream of `m` sessions with at most `b` stops, `log₂ ∑_{i ≤ b} C(m, i)` bits (`stream_leak_deferred`).

Its rounds are the deployed session's: the server's own `Hello`, the firewall's own link and `Finish`, and the frame
rounds tagged `0` (`LeanRelease.kindAt`). `Flock.zk_server_lean_stops` proves `ItemsRefine` at the session with the
frame's stops, whose frame rounds drop a rep's leading domain frame, from per-round equalities: the frame rounds'
(`LiveRoundEq`, `FillRoundEq`) hold wherever the device's frames are the session's (`LeanRelease.liveRoundEq_of_frames`,
`liveRoundEq_of_honest`, `fillRoundEq_of_frames`), and that the server's `Hello` and the firewall's link and `Finish`
are the session's stays a hypothesis there (`HelloEq`, `LinkRoundEq`, `FinalRoundEq`). -/
theorem Flock.zk_server_lean (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (vs : S.SwVals) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hsc : S.SwCausal vs) (hT : S.TapeOK) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hR : Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hI : L.ItemsRefine H (S.swP vs).prover (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk x w) {Θ : Type}
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
    (L.refD H _ _ wk x w hI) Vs θ hk Ev
  exact h

end Claims.ZKServer
