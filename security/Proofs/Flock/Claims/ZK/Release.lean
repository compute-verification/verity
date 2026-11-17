module

public import Proofs.Flock.Claims.ZK.Blind
public import Flock.ZkServer.Frame
public import Flock.ZkServer.Assumptions
public import Flock.ZkServer.Coins

/-! The ZK server's release is the Lean frame, at the Lean frame's samples. -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock.ZkServer.Frame (Cfg Masker Draws MaskSlot padsRoot)

/-- The value the ZK server's release log `t` reports fixing for `what` by its own computation (`By.recheck`, `By.shadow`,
`By.own`): the first such item's. -/
def computedOf (t : _root_.Flock.ZkServer.Contract.ReleaseLog) (what : String) : Option ByteArray :=
  t.items.toList.findSome? fun
    | .fixed w .recheck want _ _ | .fixed w .shadow want _ _ | .fixed w .own want _ _ =>
      if w = what then some want else none
    | _ => none

section CoinStops
open _root_.Flock.ZkServer.Frame (INNER_BETA RHO_COINS STOP innerCoins rhoCoins)

theorem answer_beta_zero (c : Cfg) (s : Masker) (b : _root_.Flock.F128) (hn : s.owed = some 1)
    (hi : s.inner = some INNER_BETA) (hb : b.isZero = true) :
    s.answer c #[b] = .error s!"{STOP}degenerate coin β = 0" := by
  unfold _root_.Flock.ZkServer.Frame.Masker.answer
  simp [hn, hi, innerCoins, hb, INNER_BETA]
  rfl

theorem answer_beta_ne (c : Cfg) (s : Masker) (b : _root_.Flock.F128) (hn : s.owed = some 1)
    (hi : s.inner = some INNER_BETA) (hb : b.isZero = false) :
    s.answer c #[b] = .ok { s with owed := none, beta := b, inner := some (INNER_BETA + 1) } := by
  unfold _root_.Flock.ZkServer.Frame.Masker.answer
  simp [hn, hi, innerCoins, hb, INNER_BETA]
  rfl

theorem answer_rho_dependent (c : Cfg) (s : Masker) (r0 r1 a b : _root_.Flock.F128) (hn : s.owed = some 2)
    (hi : s.inner = none) (hl : s.l0 = true) (hp : s.pos = c.order.size) (hr : s.rho = #[r0, r1])
    (hd : _root_.Flock.Zk.rhoIndependent (⟨r0, r1⟩, ⟨a, b⟩) = false) :
    s.answer c #[a, b] = .error s!"{STOP}degenerate coins: rho_0, rho_1 dependent over F128" := by
  unfold _root_.Flock.ZkServer.Frame.Masker.answer
  simp [hn, hi, hl, hp, rhoCoins, hr, RHO_COINS, hd]
  rfl

theorem answer_rho_independent (c : Cfg) (s : Masker) (r0 r1 a b : _root_.Flock.F128) (hn : s.owed = some 2)
    (hi : s.inner = none) (hl : s.l0 = true) (hp : s.pos = c.order.size) (hr : s.rho = #[r0, r1])
    (hd : _root_.Flock.Zk.rhoIndependent (⟨r0, r1⟩, ⟨a, b⟩) = true) :
    s.answer c #[a, b] = .ok { s with owed := none, rho := #[r0, r1, a, b] } := by
  unfold _root_.Flock.ZkServer.Frame.Masker.answer
  simp [hn, hi, hl, hp, rhoCoins, hr, RHO_COINS, hd]
  rfl

end CoinStops

section PassDecision
open _root_.Flock.ZkServer.Frame (check standIn passes firstDiff roundFree)

theorem check_none {c : Cfg} {s : Masker} {v r : ByteArray} {n : Nat} {out : ByteArray} {s' : Masker}
    (h : check c s v r n = .ok (out, s', none)) : _root_.Flock.ZkServer.Frame.round c s v n = .ok (out, s') := by
  unfold check standIn at h
  cases hs : s.stopped <;> simp only [hs, Bool.false_eq_true, ↓reduceIte] at h
  · cases hp : passes c s v r n with
    | error e => simp only [hp] at h; cases h
    | ok w =>
      simp only [hp] at h
      cases w with
      | some w =>
        cases hr : _root_.Flock.ZkServer.Frame.round c { s with stopped := true } r n with
        | error e => simp only [hr] at h; cases h
        | ok x => simp only [hr] at h; cases h
      | none =>
        cases hv : _root_.Flock.ZkServer.Frame.round c s v n with
        | ok x => simp only [hv] at h; cases h; rfl
        | error e =>
          simp only [hv] at h
          split at h
          · cases h
          · cases hr : _root_.Flock.ZkServer.Frame.round c { s with stopped := true } r n with
            | error e => simp only [hr] at h; cases h
            | ok x => simp only [hr] at h; cases h
  · cases hr : _root_.Flock.ZkServer.Frame.round c { s with stopped := true } r n with
    | error e => simp only [hr] at h; cases h
    | ok x => simp only [hr] at h; cases h

theorem check_some {c : Cfg} {s : Masker} {v r : ByteArray} {n : Nat} {out : ByteArray} {s' : Masker} {w : String}
    (h : check c s v r n = .ok (out, s', some w)) :
    _root_.Flock.ZkServer.Frame.round c { s with stopped := true } r n = .ok (out, s') := by
  unfold check standIn at h
  cases hs : s.stopped <;> simp only [hs, Bool.false_eq_true, ↓reduceIte] at h
  · cases hp : passes c s v r n with
    | error e => simp only [hp] at h; cases h
    | ok w' =>
      simp only [hp] at h
      cases w' with
      | some w' =>
        cases hr : _root_.Flock.ZkServer.Frame.round c { s with stopped := true } r n with
        | error e => simp only [hr] at h; cases h
        | ok x => simp only [hr] at h; cases h; rfl
      | none =>
        cases hv : _root_.Flock.ZkServer.Frame.round c s v n with
        | ok x => simp only [hv] at h; cases h
        | error e =>
          simp only [hv] at h
          split at h
          · cases h
          · cases hr : _root_.Flock.ZkServer.Frame.round c { s with stopped := true } r n with
            | error e => simp only [hr] at h; cases h
            | ok x => simp only [hr] at h; cases h; rfl
  · cases hr : _root_.Flock.ZkServer.Frame.round c { s with stopped := true } r n with
    | error e => simp only [hr] at h; cases h
    | ok x => simp only [hr] at h; cases h; rfl

theorem check_self {c : Cfg} {s : Masker} {v : ByteArray} {n : Nat} {out : ByteArray} {s' : Masker}
    (hs : s.stopped = false) (h : _root_.Flock.ZkServer.Frame.round c s v n = .ok (out, s')) :
    check c s v v n = .ok (out, s', none) := by
  have hr : ∃ fr, roundFree c s v n = .ok (out, s', fr) := by
    unfold _root_.Flock.ZkServer.Frame.round at h
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
  show (match _root_.Flock.ZkServer.Frame.round c s v n with
    | .ok (out, s') => (Except.ok (out, s', none) : Except String _)
    | .error e => if e.startsWith _root_.Flock.ZkServer.Frame.STOP then throw e else standIn c s v n e) = _
  rw [h]

end PassDecision

def coinHello (hb : Msg) : Except String _root_.Flock.ZkServer.Coins.Hello :=
  match String.fromUTF8? ⟨(hb.drop 5).toArray⟩ with
  | some s => _root_.Flock.ZkServer.Coins.Hello.parse s
  | none => .error "a Hello that is not UTF-8"

/-- **The coin check's verdict** on a session: its reading of the `Hello`, `hl`, then `Hello`'s answer `a`
(`Release.start`), then the openings `os` in order (`Release.openAll`). -/
def coinVerdict (hl : Except String _root_.Flock.ZkServer.Coins.Hello) (a : Msg) (os : List (String × ℕ × Msg)) :
    Except String _root_.Flock.ZkServer.Coins.Release :=
  hl.bind fun hl => (_root_.Flock.ZkServer.Coins.Release.start hl a).bind fun S => S.openAll os

/-- The openings `ms` as `flock-zk-server coins` takes them, from the `j`-th on: each with the stream and coin count of
its request (`req`). -/
def reqsFrom (req : ℕ → String × ℕ) : ℕ → List Msg → List (String × ℕ × Msg)
  | _, [] => []
  | j, m :: ms => ((req j).1, (req j).2, m) :: reqsFrom req (j + 1) ms

theorem reqsFrom_map (req : ℕ → String × ℕ) :
    ∀ (j : ℕ) (ms : List Msg), (reqsFrom req j ms).map (fun o => o.2.2) = ms
  | _, [] => rfl
  | j, m :: ms => congrArg (m :: ·) (reqsFrom_map req (j + 1) ms)

/-- **The deployed release**, for witnesses `W` on the server's tape `R`: at statement `x`, after the history `h`, round
`h.length` leaves as its rep's Lean frame's release of the device round, at the rep's samples on the tape. -/
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
  /-- The transport: the frame state of the rep that released the last round of `h`; a function of the
  statement and the history, never of the tape, and its `stopped` is not read. -/
  prog : Msg → List (Msg × Msg) → Masker
  /-- The transport: the coins the frame takes after the rounds of `h`, for the stream's last answer. -/
  coinsOf : Msg → List (Msg × Msg) → Array _root_.Flock.F128
  /-- The transport: the frame state of the rep of the round after `h` before that round, where it is the
  rep's first; a function of the statement and the history, never of the tape. -/
  fresh : Msg → List (Msg × Msg) → Except String Masker
  /-- the device round at honest worker values, given the values the ZK server fixes by its own computation, by name -/
  device : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg
  /-- the ZK server's release log after the rounds of `h`, whose own values the device round reads -/
  computed : Msg → W → R → List (Msg × Msg) → _root_.Flock.ZkServer.Contract.ReleaseLog
  /-- Rust's own checks on a worker's rep beside the Lean frame's pass decision: any decision at all, as each
  can only fail a round more. -/
  cross : Msg → ℕ → W → R → List (Msg × Msg) → Msg → Bool
  /-- The device round the ZK server sends in the device's place once a round failed, given the values the
  ZK server fixes by its own computation, by name. -/
  scheduled : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg
  /-- round `j`'s coin count -/
  coins : Msg → ℕ → ℕ
  /-- round `j`'s release time -/
  sched : Msg → ℕ → ℕ
  stop : Msg
  /-- round `0`, the server's own `Hello` at the tape's nonce (`CoinLayer.helloD` at the deployed coin layer) -/
  hello : Msg → R → Msg
  /-- The transport: what its coin check reads of the session's `Hello` (the tree, the session stream's coin
  count and the nonce), or why it reads none. -/
  helloChk : Msg → R → Except String _root_.Flock.ZkServer.Coins.Hello
  /-- the transport: the stream and coin count it requests the `j`-th opening at, by the statement's schedule
  (`coin_tree::Checked`'s request at `Req::Commit` and `Req::Round`) -/
  req : Msg → ℕ → String × ℕ
  /-- Round `1`, the link, the ZK server's own: level 0's roots as its recheck commits them, never the
  worker's. -/
  link : Msg → W → R → List (Msg × Msg) → Msg
  /-- the record and proof files the ZK server builds itself at the session's `Finish` (the session's Lean finish,
  `Flock.ZkServer.Finish`) -/
  files : Msg → W → R → List (Msg × Msg) → String × Array ByteArray
  /-- Whether the ZK server's own `Finish` stops: where a constraint fails at the rep's sampled pads, a
  function of the witness and the workers' values. -/
  finStops : Msg → W → R → List (Msg × Msg) → Bool
  /-- the session's last round, its `Finish` -/
  last : Msg → ℕ

namespace LeanRelease

/-- Rep `i`'s samples on the tape `ρ`: its frame's bytes, parsed as the frame parses them. -/
def samplesAt {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (i : ℕ) : Draws :=
  Draws.ofBytes (L.lay x i) (L.draws ρ i)

/-- **The round's rep's configuration**: `Cfg.ofSamples` of its samples on the tape, as `flock-zk-server frame` starts the
rep. -/
def cfg {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Cfg :=
  Cfg.ofSamples (L.lay x (L.rep x h.length)) (L.samplesAt x ρ (L.rep x h.length)) (L.slot x (L.rep x h.length))

theorem cfg_pads {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) :
    (L.cfg x ρ h).h = (Draws.ofBytes (L.lay x (L.rep x h.length)) (L.draws ρ (L.rep x h.length))).h :=
  rfl

theorem cfg_root {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) :
    (L.cfg x ρ h).root = padsRoot (L.lay x (L.rep x h.length)) (L.samplesAt x ρ (L.rep x h.length)) :=
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

/-- Round `j`'s rep's configuration: `Cfg.ofSamples` of its samples on the tape. -/
def cfgAt {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (j : ℕ) : Cfg :=
  Cfg.ofSamples (L.lay x (L.rep x j)) (L.samplesAt x ρ (L.rep x j)) (L.slot x (L.rep x j))

/-- The frame's stop at a mask slot short of full rank. -/
def rankRefused : String :=
  _root_.Flock.ZkServer.Frame.STOP ++ "degenerate coins: the mask slot's weights are short of full rank"

/-- **Whether the coins so far leave the rep's mask slot short of full rank** at the state `s` a round is checked at:
in the algebraic phase (`on`, before the opening's label and outside level 0) at its last position, the rank check the
opening's label runs there (`Masker.label`) on the claim points of the table's earlier reps and of the coins so far
(`claimPoints`, `maskRank`) fails. It reads the coins, the layout and the slot alone. -/
def rankFails (c : Cfg) (s : Masker) : Bool :=
  s.on && !s.done && !s.l0 && s.pos == c.nPiop &&
    match c.mask with
    | none => false
    | some ms =>
      match _root_.Flock.ZkServer.Frame.claimPoints c.lay s.coins with
      | .ok P => _root_.Flock.ZkServer.Frame.maskRank (s.prior ++ P) ms
          (c.lay.nLc + _root_.Flock.ZkServer.Frame.K_SKIP) != 128 * (s.prior ++ P).size
      | .error _ => false

/-- **The rank stop**: the state the coins leave, or the stop where they leave the mask slot short of full rank
(`rankFails`). -/
def rankStop (c : Cfg) (s : Masker) : Except String Masker :=
  if rankFails c s then .error rankRefused else .ok s

theorem rankStop_ok {c : Cfg} {s m : Masker} (h : rankStop c s = .ok m) : m = s ∧ rankFails c s = false := by
  unfold rankStop at h
  cases hf : rankFails c s with
  | true => rw [hf] at h; cases h
  | false => rw [hf] at h; cases h; exact ⟨rfl, rfl⟩

theorem rankStop_of_not {c : Cfg} {s : Masker} (h : rankFails c s = false) : rankStop c s = .ok s := by
  unfold rankStop
  rw [h]
  rfl

theorem bind_rankStop_ok {c : Cfg} {e : Except String Masker} {m : Masker} (h : (e >>= rankStop c) = .ok m) :
    e = .ok m ∧ rankFails c m = false := by
  cases e with
  | error _ => cases h
  | ok s =>
    obtain ⟨rfl, hf⟩ := rankStop_ok (c := c) (s := s) h
    exact ⟨rfl, hf⟩

theorem rankFails_of_l0 (c : Cfg) {s : Masker} (h : s.l0 = true) : rankFails c s = false := by
  simp [rankFails, h]

theorem bind_rankStop_l0 {c : Cfg} {e : Except String Masker} {s : Masker} (he : e = .ok s) (hl : s.l0 = true) :
    (e >>= rankStop c) = .ok s := by
  rw [he]
  exact rankStop_of_not (rankFails_of_l0 c hl)

theorem rankFails_congr {c c' : Cfg} (s : Masker) (hl : c.lay = c'.lay) (hn : c.nPiop = c'.nPiop)
    (hm : c.mask = c'.mask) : rankFails c s = rankFails c' s := by
  unfold rankFails
  rw [hl, hn, hm]

theorem rankStop_congr {c c' : Cfg} (s : Masker) (hl : c.lay = c'.lay) (hn : c.nPiop = c'.nPiop)
    (hm : c.mask = c'.mask) : rankStop c s = rankStop c' s := by
  unfold rankStop
  rw [rankFails_congr s hl hn hm]

/-- **Who takes the coins of round `n - 1`**, as the session loop routes them (`Session.relay`): `Hello`'s and the
`Commit`'s go to the session, not to a frame (`Sessions.link`), so round `n` starts at its rep's first-round state
`fresh`; else the frame of the rep that released round `n - 1`, at its state `prog` and configuration `cP`
(`Frames.coins`), whose refusal stops the session. Where round `n` is that rep's (`same`), it is checked at the state
the coins leave, at its configuration `cN`, which the rank stop refuses where the coins leave its mask slot short of
full rank (`rankStop`: the opening's label, the round after the algebraic phase's last coins, would stop at them); else
it is the first round of its rep, which takes no coins, at `fresh`. -/
def takeAt (n : ℕ) (same : Bool) (cN cP : Cfg) (prog : Masker) (fresh : Except String Masker)
    (cs : Array _root_.Flock.F128) : Except String Masker :=
  if n ≤ 2 then fresh else if same then prog.answer cN cs >>= rankStop cN else prog.answer cP cs >>= fun _ => fresh

/-- **The frame takes the stream's coins** for the last round of `h` (`takeAt`): the state the round after `h` is
checked at, or the frame's refusal of the coins. -/
def frameAnswer {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Except String Masker :=
  match h.getLast? with
  | some _ =>
    takeAt h.length (decide (L.rep x h.length = L.rep x (h.length - 1))) (L.cfg x ρ h) (L.cfgAt x ρ (h.length - 1))
      (L.prog x h) (L.fresh x h) (L.coinsOf x h)
  | none => pure (L.prog x h)

/-- The transport's stop where its coin check refuses. -/
def coinRefused : String := _root_.Flock.ZkServer.Coins.STOP ++ "the coin check refused an opening"

/-- **Whether the frame took the coins after every earlier round** of `h`, the rounds of each proper prefix
(`frameAnswer`). -/
def tookEarlier {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Bool :=
  (List.range h.length).all fun j => (L.frameAnswer x ρ (h.take j)).isOk

/-- **The coins the frame is handed and takes** after `h`: where the transport's coin check passes the history
(`opens`) and the frame took the coins after every earlier round (`tookEarlier`), the frame's taking of the stream's last
coins (`frameAnswer`); else the transport's stop. The transport hands the frame no coins past the check's refusal, and a
refusal stops the session for good: the session loop answers every request after it with it (`Session.State.refuse`),
so the coins after a round reach a frame only where every earlier round's were taken (`refusesCoins_of_take`). -/
def answered {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Except String Masker :=
  if L.opens x ρ h && L.tookEarlier x ρ h then L.frameAnswer x ρ h else .error coinRefused

theorem answered_of_opens {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (ho : L.opens x ρ h = true) (ht : L.tookEarlier x ρ h = true) : L.answered x ρ h = L.frameAnswer x ρ h := by
  unfold answered
  rw [ho, ht]
  rfl

theorem answered_of_not_opens {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (ho : L.opens x ρ h = false) : L.answered x ρ h = .error coinRefused := by
  unfold answered
  rw [ho]
  rfl

theorem answered_of_not_tookEarlier {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (ht : L.tookEarlier x ρ h = false) : L.answered x ρ h = .error coinRefused := by
  unfold answered
  rw [ht, Bool.and_false]
  rfl

theorem answered_ok {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)} {m : Masker}
    (hA : L.answered x ρ h = .ok m) :
    L.opens x ρ h = true ∧ L.tookEarlier x ρ h = true ∧ L.frameAnswer x ρ h = .ok m := by
  unfold answered at hA
  split at hA
  · rename_i hc
    exact ⟨((Bool.and_eq_true _ _).mp hc).1, ((Bool.and_eq_true _ _).mp hc).2, hA⟩
  · cases hA

theorem frameAnswer_of_answered {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    {m : Masker} (hA : L.answered x ρ h = .ok m) : L.frameAnswer x ρ h = .ok m :=
  (answered_ok hA).2.2

theorem answered_rank {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)} {m : Masker}
    (hA : L.answered x ρ h = .ok m) (hn : 3 ≤ h.length) (hs : L.rep x h.length = L.rep x (h.length - 1)) :
    rankFails (L.cfg x ρ h) m = false := by
  have hf := frameAnswer_of_answered hA
  unfold frameAnswer at hf
  split at hf
  · unfold takeAt at hf
    rw [if_neg (by omega), if_pos (decide_eq_true hs)] at hf
    exact (bind_rankStop_ok hf).2
  · rename_i hg
    rw [List.getLast?_eq_none_iff] at hg
    subst hg
    simp at hn

theorem frameAnswer_nil {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) :
    L.frameAnswer x ρ [] = pure (L.prog x []) :=
  rfl

theorem tookEarlier_single {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (e : Msg × Msg) :
    L.tookEarlier x ρ [e] = true :=
  rfl

theorem tookEarlier_append {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (e : Msg × Msg) :
    L.tookEarlier x ρ (h ++ [e]) = (L.tookEarlier x ρ h && (L.frameAnswer x ρ h).isOk) := by
  unfold tookEarlier
  rw [List.length_append, List.length_singleton, List.range_succ, List.all_append, List.all_cons, List.all_nil,
    Bool.and_true, List.take_left' rfl]
  congr 1
  refine Bool.eq_iff_iff.mpr ?_
  simp only [List.all_eq_true, List.mem_range]
  refine forall_congr' fun j => imp_congr_right fun hj => ?_
  rw [List.take_append_of_le_length (by omega)]

theorem tookEarlier_of_answered {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    {m : Masker} (hA : L.answered x ρ h = .ok m) (e : Msg × Msg) : L.tookEarlier x ρ (h ++ [e]) = true := by
  obtain ⟨-, ht, hf⟩ := answered_ok hA
  rw [L.tookEarlier_append x ρ h e, ht, hf]
  rfl

theorem answered_append {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)} {m : Masker}
    (hA : L.answered x ρ h = .ok m) (e : Msg × Msg) (ho : L.opens x ρ (h ++ [e]) = true) :
    L.answered x ρ (h ++ [e]) = L.frameAnswer x ρ (h ++ [e]) :=
  L.answered_of_opens x ρ _ ho (tookEarlier_of_answered hA e)

theorem answered_single {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (e : Msg × Msg) :
    L.answered x ρ [e] = L.frameAnswer x ρ [e] :=
  L.answered_of_opens x ρ [e] rfl (L.tookEarlier_single x ρ e)

theorem opens_of_hello {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (m₀ a : Msg) (o : Msg × Msg)
    (ops : List (Msg × Msg)) {hl : _root_.Flock.ZkServer.Coins.Hello} (hp : L.helloChk x ρ = .ok hl) :
    L.opens x ρ ((m₀, a) :: o :: ops) =
      ((_root_.Flock.ZkServer.Coins.Release.start hl a).bind fun S =>
        S.openAll (reqsFrom (L.req x) 0 ((o :: ops).map Prod.snd))).isOk := by
  show (coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 ((o :: ops).map Prod.snd))).isOk = _
  unfold coinVerdict
  rw [hp]
  try rfl

theorem reqsFrom_take (req : ℕ → String × ℕ) :
    ∀ (j : ℕ) (ms : List Msg) (k : ℕ), reqsFrom req j (ms.take k) = (reqsFrom req j ms).take k
  | _, [], _ => by simp only [List.take_nil, reqsFrom]
  | _, _ :: _, 0 => rfl
  | j, m :: ms, k + 1 => by
    show ((req j).1, (req j).2, m) :: reqsFrom req (j + 1) (ms.take k) =
      ((req j).1, (req j).2, m) :: (reqsFrom req (j + 1) ms).take k
    rw [reqsFrom_take req (j + 1) ms k]

theorem openAll_take (S : _root_.Flock.ZkServer.Coins.Release) :
    ∀ (os : List (String × ℕ × Msg)) (k : ℕ), (S.openAll os).isOk = true → (S.openAll (os.take k)).isOk = true
  | [], k, h => by rw [List.take_nil]; exact h
  | _ :: _, 0, _ => rfl
  | (st, n, m) :: os, k + 1, h => by
    rw [List.take_succ_cons]
    unfold _root_.Flock.ZkServer.Coins.Release.openAll at h ⊢
    cases hop : S.open st n m with
    | error e => simp [hop, Except.isOk, Except.toBool] at h
    | ok p =>
      obtain ⟨S', c⟩ := p
      simp only [hop] at h
      exact openAll_take S' os k h

theorem coinVerdict_take (hl : Except String _root_.Flock.ZkServer.Coins.Hello) (a : Msg)
    (os : List (String × ℕ × Msg)) (k : ℕ) (h : (coinVerdict hl a os).isOk = true) :
    (coinVerdict hl a (os.take k)).isOk = true := by
  unfold coinVerdict at h ⊢
  cases hl with
  | error e => exact h
  | ok hl =>
    cases hS : _root_.Flock.ZkServer.Coins.Release.start hl a with
    | error e =>
      change (((_root_.Flock.ZkServer.Coins.Release.start hl a).bind fun S => S.openAll os)).isOk = true at h
      simp [Except.bind, hS, Except.isOk, Except.toBool] at h
    | ok S =>
      change (((_root_.Flock.ZkServer.Coins.Release.start hl a).bind fun S => S.openAll os)).isOk = true at h
      show (((_root_.Flock.ZkServer.Coins.Release.start hl a).bind fun S => S.openAll (os.take k))).isOk = true
      rw [hS] at h ⊢
      exact openAll_take S os k h

theorem opens_short {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) :
    ∀ (h : List (Msg × Msg)), h.length ≤ 1 → L.opens x ρ h = true
  | [], _ => rfl
  | [_], _ => rfl
  | _ :: _ :: _, hl => by simp only [List.length_cons] at hl; omega

theorem opens_take {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (i : ℕ)
    (ho : L.opens x ρ h = true) : L.opens x ρ (h.take i) = true := by
  rcases Nat.lt_or_ge i 2 with hi | hi
  · exact L.opens_short x ρ _ (by rw [List.length_take]; omega)
  obtain ⟨i, rfl⟩ : ∃ k, i = k + 2 := ⟨i - 2, by omega⟩
  match h, ho with
  | [], _ => exact L.opens_short x ρ _ (by rw [List.length_take]; simp only [List.length_nil]; omega)
  | [_], _ => exact L.opens_short x ρ _ (by rw [List.length_take]; simp only [List.length_cons, List.length_nil]; omega)
  | (_, a) :: o :: ops, ho =>
    show (coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 ((o :: ops.take i).map Prod.snd))).isOk = true
    have e : (o :: ops.take i).map Prod.snd = ((o :: ops).map Prod.snd).take (i + 1) := by
      show o.2 :: (ops.take i).map Prod.snd = o.2 :: (ops.map Prod.snd).take i
      rw [List.map_take]
    rw [e, reqsFrom_take]
    exact coinVerdict_take _ a _ (i + 1) ho

/-- **The frame's step** after `h` on the device round `v`, of a stopped rep (`st`) or not: at the rep's configuration,
it takes the stream's coins for the last round (`answered`), then the round (`Frame.round`), in the order
`flock-zk-server frame` takes them (`coins`, then `round`, whose `fail` stops the rep). -/
def step {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool) (v : Msg) :
    Except String (ByteArray × Masker) := do
  let m ← L.answered x ρ h
  _root_.Flock.ZkServer.Frame.round (L.cfg x ρ h) { m with stopped := st } ⟨v.toArray⟩ (L.coins x h.length)

/-- **The Lean frame's release** after `h` of the device round `v`, of a stopped rep (`st`) or not: the bytes the frame's
step returns, tagged `0` as a round message, or `stop` where it refuses. -/
def release {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool) (v : Msg) : Msg :=
  match L.step x ρ h st v with
  | .ok (out, _) => 0 :: out.data.toList
  | .error _ => L.stop

/-- **Whether the frame refuses the stream's coins** for the last round of `h`, the session stopped: the transport's
coin check refuses the history (`opens`), or the frame refuses the coins of the last round or of an earlier one
(`Masker.answer`: a zerocheck outer coin `1`, level 0's `ρ_0, ρ_1` dependent over F128, the inner proof's `β = 0`, or
coins for no round or of another count; `rankStop`: the algebraic phase's coins leaving the mask slot short of full
rank). Once true it stays true (`refusesCoins_of_take`). -/
def refusesCoins {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Bool :=
  match L.answered x ρ h with
  | .ok _ => false
  | .error _ => true

theorem refusesCoins_of_not_opens {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (ho : L.opens x ρ h = false) : L.refusesCoins x ρ h = true := by
  have ha := L.answered_of_not_opens x ρ h ho
  unfold refusesCoins
  cases hb : L.answered x ρ h with
  | ok m => rw [ha] at hb; cases hb
  | error e => rfl

theorem refusesCoins_eq_false {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) :
    L.refusesCoins x ρ h = false ↔
      L.opens x ρ h = true ∧ ∀ j, j ≤ h.length → (L.frameAnswer x ρ (h.take j)).isOk = true := by
  constructor
  · intro hr
    unfold refusesCoins at hr
    cases hA : L.answered x ρ h with
    | error e => rw [hA] at hr; cases hr
    | ok m =>
      obtain ⟨ho, ht, hf⟩ := answered_ok hA
      refine ⟨ho, fun j hj => ?_⟩
      rcases Nat.lt_or_ge j h.length with hlt | hge
      · exact List.all_eq_true.1 ht j (List.mem_range.2 hlt)
      · rw [List.take_of_length_le hge, hf]
        rfl
  · rintro ⟨ho, hall⟩
    have ht : L.tookEarlier x ρ h = true :=
      List.all_eq_true.2 fun j hj => hall j (Nat.le_of_lt (List.mem_range.1 hj))
    have hf := hall h.length (le_refl _)
    rw [List.take_length] at hf
    unfold refusesCoins
    rw [L.answered_of_opens x ρ h ho ht]
    cases hF : L.frameAnswer x ρ h with
    | error e => rw [hF] at hf; simp [Except.isOk, Except.toBool] at hf
    | ok m => rfl

theorem refusesCoins_of_take {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (i : ℕ)
    (hr : L.refusesCoins x ρ (h.take i) = true) : L.refusesCoins x ρ h = true := by
  cases hb : L.refusesCoins x ρ h with
  | true => rfl
  | false =>
    obtain ⟨ho, hall⟩ := (L.refusesCoins_eq_false x ρ h).1 hb
    have hi : L.refusesCoins x ρ (h.take i) = false := by
      refine (L.refusesCoins_eq_false x ρ (h.take i)).2 ⟨L.opens_take x ρ h i ho, fun j hj => ?_⟩
      rw [List.length_take] at hj
      rw [List.take_take]
      exact hall (min j i) (by omega)
    rw [hi] at hr
    cases hr

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

/-- **The pass decision**, the Lean frame's (`Frame.check`, which `flock-zk-server frame` answers each device round
with): after `h`, once the frame took the stream's last coins, the device round `v` against the ZK server's scheduled
round in its place, with the values its release log reports. -/
def pass {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) (v : Msg) : Bool :=
  match L.answered x ρ h with
  | .ok m =>
    match _root_.Flock.ZkServer.Frame.check (L.cfg x ρ h) { m with stopped := false } ⟨v.toArray⟩
        ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) with
    | .ok (_, _, none) => true
    | _ => false
  | .error _ => false

theorem check_release {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) (v : Msg)
    {m : Masker} (hm : L.answered x ρ h = .ok m) {out : ByteArray} {s' : Masker} {why : Option String}
    (hc : _root_.Flock.ZkServer.Frame.check (L.cfg x ρ h) { m with stopped := false } ⟨v.toArray⟩
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
      show _ = match _root_.Flock.ZkServer.Frame.round (L.cfg x ρ h) { m with stopped := false } ⟨v.toArray⟩
        (L.coins x h.length) with
        | .ok (out, _) => 0 :: out.data.toList
        | .error _ => L.stop
      rw [check_none hc]
    | some e =>
      show _ = match _root_.Flock.ZkServer.Frame.round (L.cfg x ρ h) { m with stopped := true }
        ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) with
        | .ok (out, _) => 0 :: out.data.toList
        | .error _ => L.stop
      rw [check_some hc]

/-- **Round `j`'s kind**: round `0`, the `Hello`, is the server's own; the link (round `1`) and the `Finish` (the last
round) are the ZK server's own, which a worker's value passes only where it is the ZK server's byte for byte; every other
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

/-- **The ZK server's own message** after `h`, at its link and its `Finish`: the stop where the frame refuses the
stream's last coins; at the last round `Final.encode` of the record and proof files it builds itself; else its link. -/
def ownMsg {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) : Msg :=
  if L.refusesCoins x ρ h then L.stop
  else if h.length = L.last x then FlockDeploy.Final.encode ⟨x, (L.files x w ρ h).1, (L.files x w ρ h).2⟩
  else L.link x w ρ h

/-- **The model's stop at the `Finish`**, its one definition: the server refuses the last round where the ZK server's own
`Finish` stops (`finStops`). With it the session stops at its `Finish` where a round failed or the workers' values fail a
constraint at the sampled pads, still one bit a session (`ZkServer.zk_deferred`'s factor `2`), `0` for honest workers. A
premise on the workers' values instead would make this `false`. -/
def refusesFinish {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) : Bool :=
  decide (h.length = L.last x) && L.finStops x w ρ h

/-- **The deployed release as a ZK server**: one slot per round. Round `0` is the server's own `Hello`; the link and
the `Finish` are the ZK server's own (`ownMsg`), which a worker's value passes only where it is the ZK server's; at a
frame round the worker's value is the device round (with whether it is the ZK server's stand-in), the mask is the Lean
frame's release and the check is Rust's checks, the Lean frame's pass decision and the frame's own refusals, refusing a
worker value that says it is a stand-in. The session ends at its `Finish` (§10.1), so the ZK server takes no frame round
past it: a worker's value there fails. The server refuses the `Finish` where the ZK server's own `Finish` stops
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

theorem finStops_of_live {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : L.toSrv.live wk x w ρ κ h = true) (he : h.length = L.last x) :
    L.finStops x w ρ h = false := by
  have hp := L.toSrv.pass_of_live wk x w ρ κ h hl (le_refl h.length)
  rw [List.take_length] at hp
  have hr : L.toSrv.refuses x w ρ h = false := by simpa using ((Bool.and_eq_true _ _).mp hp).1
  change (decide (h.length = L.last x) && L.finStops x w ρ h) = false at hr
  simpa [he] using hr

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

/-- **The deployed release's stand-in**: once a round failed, the scheduled round, with the values the ZK server's
release log reports, as a stand-in. -/
def fillOf {W R : Type} (L : LeanRelease W R) : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val :=
  fun x _ _ w ρ h => (L.scheduled x w ρ h (computedOf (L.computed x w ρ h)), true)

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

theorem live_some {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked)
    (hl : L.toSrv.live wk x w ρ κ h = true) :
    ∃ v, wk x w ρ h 0 = some (v, false) ∧ L.cross x h.length w ρ h v = true ∧ L.pass x w ρ h v = true ∧
      L.takes x ρ h v = true := by
  obtain ⟨v, hv, -, hc, hp, ht⟩ := L.live_some_lt wk x w ρ κ h hk hl
  exact ⟨v, hv, hc, hp, ht⟩

theorem live_lt_last {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked)
    (hl : L.toSrv.live wk x w ρ κ h = true) : h.length < L.last x := by
  obtain ⟨-, -, hlt, -⟩ := L.live_some_lt wk x w ρ κ h hk hl
  exact hlt

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

theorem frameD_own {W R : Type} (L : LeanRelease W R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val) (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .own) :
    L.toSrv.frameD fill wk x w ρ κ h = L.hello x ρ := by
  show (L.toSrv.value x h.length 0 w ρ h (L.toSrv.copy (L.toSrv.wkD fill wk κ) x w ρ h)
    (L.toSrv.wkD fill wk κ x w ρ h 0)).1 = _
  unfold ZkServer.value
  rw [show L.toSrv.kind x h.length 0 = .own from hk]
  rfl

theorem frameD_recomp {W R : Type} (L : LeanRelease W R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val) (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .recomputed) :
    L.toSrv.frameD fill wk x w ρ κ h = L.ownMsg x w ρ h := by
  show (L.toSrv.value x h.length 0 w ρ h (L.toSrv.copy (L.toSrv.wkD fill wk κ) x w ρ h)
    (L.toSrv.wkD fill wk κ x w ρ h 0)).1 = _
  unfold ZkServer.value
  rw [show L.toSrv.kind x h.length 0 = .recomputed from hk]
  rfl

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

def HelloRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (x : Msg) (w : W) : Prop :=
  ∀ ρ, (((P w).next x [] ρ).eval H).1 = L.hello x ρ

def OwnRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) : Prop :=
  ∀ ρ κ h, L.kindAt x h.length = .recomputed → (h.length + 1 < n x ∨ L.toSrv.live wk x w ρ κ h = true) →
    L.toSrv.OwnHist L.fillOf wk x w ρ κ h → (((P w).next x h ρ).eval H).1 = L.ownMsg x w ρ h

def LiveRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (wk : L.toSrv.Worker)
    (x : Msg) (w : W) : Prop :=
  ∀ ρ κ h v, L.kindAt x h.length = .masked → L.toSrv.live wk x w ρ κ h = true → L.refusesCoins x ρ h = false →
    L.toSrv.OwnHist L.fillOf wk x w ρ κ h →
    wk x w ρ h 0 = some (v, false) → (((P w).next x h ρ).eval H).1 = L.release x ρ h false v

def FillRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) : Prop :=
  ∀ ρ κ h, L.kindAt x h.length = .masked → h.length + 1 < n x → L.toSrv.live wk x w ρ κ h = false →
    L.refusesCoins x ρ h = false →
    (((P w).next x h ρ).eval H).1 = L.release x ρ h true (L.scheduled x w ρ h (computedOf (L.computed x w ρ h)))

def StopRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) : Prop :=
  ∀ ρ κ h, L.kindAt x h.length = .masked → (h.length + 1 < n x ∨ L.toSrv.live wk x w ρ κ h = true) →
    L.refusesCoins x ρ h = true → (((P w).next x h ρ).eval H).1 = L.stop

def ItemsRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) : Prop :=
  L.HelloRefine H P x w ∧ L.OwnRefine H P n wk x w ∧ L.LiveRefine H P wk x w ∧ L.FillRefine H P n wk x w ∧
    L.StopRefine H P n wk x w

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
      | false => exact hI.2.2.1 ρ κ h v hk hl hs hown hv
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

theorem Flock.zk_server_lean_of (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (vs : S.SwVals) (t : ℕ) (δ₁ : ENNReal)
    (hzk : Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) (S.swP vs).prover
      ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
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
    hzk.2 x w hR wk (L.refD H _ _ wk x w hI) Vs θ hk Ev
  exact h

/-- Leaks one bit a session, `b = 1[a round failed ∨ L.finStops]`, `0` for honest workers; over a stream
of `m` sessions with at most `b` stops, `log₂ ∑_{i ≤ b} C(m, i)` bits (`stream_leak_deferred`). -/
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
          4 * t * 3 ^ t * (Fork H ((S.swP vs).prover w) Vs θ x).Pr H) :=
  Flock.zk_server_lean_of H rel S vs t δ₁ (Flock.zk_deployZk_sw H rel S vs t ht hT1 hδ₁ hS hV hD hc hsc hT) L wk x w
    hR hI Vs θ hk Ev

end Claims.ZKServer
