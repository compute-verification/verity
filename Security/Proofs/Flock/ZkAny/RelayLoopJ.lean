module

public import Proofs.Flock.ZkAny.RelayLoopLink
public import Proofs.Flock.ZkAny.RelaySame
public import Proofs.Flock.ZkAny.RelayTook
public import Proofs.Flock.Claims.DeployK
public import Proofs.Flock.Claims.ZK.FrameZK

/-!
# The relay's invariant at the deployed loop, kept at the control slots and the `Hello`

`LoopP`'s invariant `J` of the loop's frames and the session, against the model's run, is `LoopJ`:

* `StreamAt`: what the loop holds of a rep stream's frame at the model's history, on the session's entry and the
  table's level 0. Where the next round is the stream's, past its first, the frame keeps `SameInv` there, on the
  session's link, the table's statement and the table its level 0 gives, at the rep's salts in the loop's frames once
  its opening has begun. Where the next round is the next stream's first, the frame holds the history's words for
  each of the stream's rounds (`roundPlace_switch1`'s `hco`). `streamAt_stay`: it is kept where the rep's salts stay.
  `find?_of_saltsAt`, `saltsAt_of_find?`: the rep's salts are the loop's salts' entry for it.
* `RepJ`: a rep's frame, the session's entry and the rep's table's level 0 bound for the session, with `StreamAt` on
  them at the model's history so far.
* `nextAt`: the model's next round's rep stream and its index there. `SeenAt`: the rep streams seen are those with a
  round before the model's next. `seenAt_nil`: none before the model's first round.
* `BuiltAt`: every level 0 bound to the session the one the loop drew for its table, `Level0.ofDraws` on what the
  host and the operating system handed it for the table's rep 0 (`LoopModel.src`). `builtAt_stays`: kept where the
  session's tables' level 0 stay. `build_of_ofDraws`, `cap_of_drawn`: such a level 0 is a `build`, so its table's
  commitment's cap is its root. `builtAt_commit`: past the session's `Commit`, where the loop drew those level 0s.
* `LoopJ`: the session stopped in the loop's frames exactly where its `Hello` left and an earlier round of the model's
  run failed; no stream seen before the `Hello` left; `RepJ` at each rep stream seen; the session's plan the setup's
  (`PlanOk`); `SeenAt` at the model's history so far; no stream seen before the model's link; `BuiltAt` past the
  `Commit`; the session's entry the setup's once its `Hello` left (`SessOn`); its `Hello` and coins at the setup's
  link points (`LinksOk`); past the `Commit`, at a level 0 bound to it, its entry linked to the history's link
  coins (`LinkOn`); and, while the model's run has a message there, what the loop took of the coins so far against
  the model's release (`LoopModel.TookAt`).
* `repJ_stays`: updates that keep the rep's frame and what the session's rounds read of it (`Stays`) keep `RepJ`, the
  model's history as it was.
* `answer_head`, `start_byte`, `answer_hello_start`, `sendHello_hello`, `stepK_hello_start`: past the `Hello` the
  session's coin tree is the one the auditor's answer, which is coins, begins under the setup's `Hello` at the
  request's nonce.
  `coins_open`, `stepK_coin_open`: past a `Commit` or a frame round it is the one its tree opens
  the auditor's coins to as the schedule's next opening.
* **`loopKept_ctl`**, `loopKept_ctl_all`: `LoopJ` is kept at every slot that reads no model message.
  **`loopKept_hello`**: and at the `Hello`, whose round the model's run passes, where the model's coin check reads
  the setup's `Hello` at the relay's nonce. `liveBefore_one`: past the `Hello` every earlier round passed.
  **`loopKept_commit`**: and at the `Commit`, where the worker's link is the ZK server's own, below the `Finish`, the
  loop's level 0s for the session being those `Session.level0` drew (`LoopRead`), the model's requests the loop's and
  its frame taking the coins after the `Hello`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed byte? helloOf reqAt)
open _root_.Flock (SessionSpec Stmt F128)

/-- **What the loop holds of rep stream `k`'s frame** `p` (table `k / 2`'s rep `k % 2`) at the model `M`'s history
`h`, on the session's entry `rel`, the table's name `name` and its level 0 `l0`, in the loop's frames `R`, the next
round being at `h.length - 2` of the setup's rounds (`roundsOf`). Where that round is stream `k`'s, past its first,
the frame keeps `SameInv` there, on the session's link `c`, the table's statement and the table its level 0 gives,
whose commitment's cap is the level-0 root, at salts `hi` that are the rep's in `R` once its opening has begun
(`saltsAt`), which it has not where its coins are as many as its algebraic phase's rounds. Where it is stream
`k + 1`'s first, the frame holds the history's words for each of stream `k`'s rounds. -/
def StreamAt {W Rr : Type} (M : LoopModel W Rr) (spec : SessionSpec) (sts : Array Stmt) (k : Nat)
    (h : List (Claims.Msg × Claims.Msg)) (rel : _root_.Flock.Firewall.Frames.Release) (name : String)
    (l0 : _root_.Flock.Firewall.Level0.Level0) (p : _root_.Flock.Firewall.Frames.Rep)
    (R : _root_.Flock.Firewall.Frames.Sessions) : Prop :=
  ((ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).1 = k →
    ∃ (j : Nat) (c : Array F128) (stm : Stmt) (t : _root_.Flock.Firewall.Finish.Tab) (xs : Array (Array F128))
      (hi : Array ByteArray) (pre : List _root_.Flock.Firewall.Frame.Req)
      (ev : List (Nat × ByteArray × _root_.Flock.Firewall.Frame.Place × Array F128)),
      (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).2 = j + 1 ∧ rel.link = some c ∧
        rel.sts[k / 2]? = some stm ∧
        _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t ∧
        _root_.Flock.Firewall.Finish.flat t.cm0.cap = l0.root ∧
        xs = _root_.Flock.Firewall.Drive.extras stm (_root_.Flock.Firewall.Drive.linkPoints stm c) ∧
        SameInv M.L M.x M.ρ h (h.tail.map fun e => ZKMal.CoinLayer.rd e.2) (h.length - 2) j t
          (rel.spec.domain name (k % 2)).toUTF8 xs hi pre ev p ∧
        (p.alg.size < p.coins.size → saltsAt R l0.root (k % 2) = some hi) ∧
        (p.alg.size = p.coins.size → p.opening = none ∧ p.opened = false)) ∧
  ((ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).1 = k + 1 →
    (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).2 = 0 →
    p.coins = (Array.range ((Loop.roundsOf spec sts).getD k 0)).map fun j =>
      Loop.wordsAt M.x (h.tail.map fun e => ZKMal.CoinLayer.rd e.2)
        (h.length - 2 - (Loop.roundsOf spec sts).getD k 0 + j + 1))

/-- **The rep's salts are an entry of the loop's salts.** -/
theorem find?_of_saltsAt {R : _root_.Flock.Firewall.Frames.Sessions} {root : ByteArray} {r : Nat}
    {hi : Array ByteArray} (h : saltsAt R root r = some hi) :
    ∃ r0 q0, R.salts.find? (fun x => x.1 == root && x.2.1 == r) = some (r0, q0, hi) := by
  unfold saltsAt at h
  rcases hf : R.salts.find? (fun x => x.1 == root && x.2.1 == r) with _ | ⟨r0, q0, hi'⟩
  · rw [hf] at h
    simp at h
  · rw [hf] at h
    have h' : hi' = hi := by simpa using h
    subst h'
    first | exact ⟨r0, q0, rfl⟩ | exact ⟨r0, q0, hf⟩

/-- **An entry of the loop's salts is the rep's salts.** -/
theorem saltsAt_of_find? {R : _root_.Flock.Firewall.Frames.Sessions} {root : ByteArray} {r : Nat}
    {r0 : ByteArray} {q0 : Nat} {hi : Array ByteArray}
    (h : R.salts.find? (fun x => x.1 == root && x.2.1 == r) = some (r0, q0, hi)) : saltsAt R root r = some hi := by
  unfold saltsAt
  rw [h]
  rfl

/-- **`StreamAt` is kept where the rep's salts stay**: of the loop's frames it reads only them. -/
theorem streamAt_stay {W Rr : Type} {M : LoopModel W Rr} {spec : SessionSpec} {sts : Array Stmt} {k : Nat}
    {h : List (Claims.Msg × Claims.Msg)} {rel : _root_.Flock.Firewall.Frames.Release} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {p : _root_.Flock.Firewall.Frames.Rep} {sid : Nat}
    {R R' : _root_.Flock.Firewall.Frames.Sessions} (hs : Stay sid l0.root R R')
    (hA : StreamAt M spec sts k h rel name l0 p R) : StreamAt M spec sts k h rel name l0 p R' := by
  refine ⟨fun hk => ?_, hA.2⟩
  obtain ⟨j, c, stm, t, xs, hi, pre, ev, hj, hlink, hstm, htab, hcap, hxs, hI, hsa, hord⟩ := hA.1 hk
  refine ⟨j, c, stm, t, xs, hi, pre, ev, hj, hlink, hstm, htab, hcap, hxs, hI, fun hlt => ?_, hord⟩
  rw [saltsAt_stay hs (k % 2) (by rw [hsa hlt]; rfl)]
  exact hsa hlt

/-- **Table `i`'s rep `r` in the loop at `σ`**, against the model's run `M`: its frame, the session's entry, the table's
name and its level 0 bound for the session, with `StreamAt` at rep stream `2 * i + r` on them at the model's history
so far, in the loop's frames. -/
def RepJ {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) (i r : Nat)
    (σ : LoopAt) : Prop :=
  ∃ (p : _root_.Flock.Firewall.Frames.Rep) (rel : _root_.Flock.Firewall.Frames.Release) (name : String)
    (l0 : _root_.Flock.Firewall.Level0.Level0),
    frameAt σ.R sid i r = some p ∧ sessAt σ.R sid = some rel ∧ rel.spec.tables[i]? = some name ∧
      _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 ∧
      StreamAt M spec sts (2 * i + r) (M.hist σ.c) rel name l0 p σ.R

/-- **The model's next round** at its history `h`: its rep stream and its index in the stream, among the setup's
rounds (`roundsOf`). -/
def nextAt (spec : SessionSpec) (sts : Array Stmt) (h : List (Claims.Msg × Claims.Msg)) : Nat × Nat :=
  ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)

/-- **The rep streams `seen` are those with a round before the model's next**, at its history `h`: a rep stream seen
is before the next round's stream or is that stream past its first round, and that stream past its first round is
seen. -/
def SeenAt (spec : SessionSpec) (sts : Array Stmt) (h : List (Claims.Msg × Claims.Msg)) (seen : List String) : Prop :=
  ∀ i r s, (i, r, s) ∈ spec.repStreams.toList →
    (seen.contains s = true → 2 * i + r < (nextAt spec sts h).1 ∨
      (2 * i + r = (nextAt spec sts h).1 ∧ 0 < (nextAt spec sts h).2)) ∧
    (2 * i + r = (nextAt spec sts h).1 → 0 < (nextAt spec sts h).2 → seen.contains s = true)

/-- Before the model's first round no rep stream is seen. -/
theorem seenAt_nil {spec : SessionSpec} {sts : Array Stmt} {h : List (Claims.Msg × Claims.Msg)}
    (hl : h.length ≤ 2) : SeenAt spec sts h [] := by
  intro i r s _
  refine ⟨fun hc => by simp at hc, fun _ h0 => ?_⟩
  unfold nextAt at h0
  rw [show h.length - 2 = 0 by omega, Loop.locate_zero] at h0
  exact absurd h0 (Nat.lt_irrefl 0)

/-- **Every level 0 the slots `L0` bind to session `sid` is the one the loop drew for its table**, against the model's
run `M`: bound for the table `name`, it is `Level0.ofDraws` at table `j`'s statement's `m` on what the host and the
operating system handed the loop for rep stream `2j` (`LoopModel.src`), `name` being table `j`'s. -/
def BuiltAt {W Rr : Type} (M : LoopModel W Rr) (spec : SessionSpec) (sts : Array Stmt)
    (L0 : _root_.Flock.Firewall.Level0.Slots) (sid : Nat) : Prop :=
  ∀ name l0, _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0 →
    ∃ j g d hi, spec.tables[j]? = some name ∧ M.src (2 * j) = .ok (g, d, hi) ∧
      _root_.Flock.Firewall.Level0.ofDraws sts[j]!.m g d = .ok l0

/-- `BuiltAt` is kept where the session's tables' level 0 stay (`Stays`). -/
theorem builtAt_stays {W Rr : Type} {M : LoopModel W Rr} {spec : SessionSpec} {sts : Array Stmt} {sid : Nat}
    {σ σ' : LoopAt} (hs : Stays sid σ σ') (h : BuiltAt M spec sts σ.L0 sid) : BuiltAt M spec sts σ'.L0 sid := by
  intro name l0 hl
  rw [hs.1 name] at hl
  exact h name l0 hl

/-- **A level 0 `Level0.ofDraws` gives is a `Level0.build`.** -/
theorem build_of_ofDraws {m : Nat} {g : Array F128} {d : ByteArray} {l : _root_.Flock.Firewall.Level0.Level0}
    (h : _root_.Flock.Firewall.Level0.ofDraws m g d = .ok l) :
    ∃ lv tPad extra pads salts, l = _root_.Flock.Firewall.Level0.build m lv tPad g extra pads salts := by
  unfold _root_.Flock.Firewall.Level0.ofDraws at h
  cases hz : _root_.Flock.Zk.schedule m with
  | error e => rw [hz] at h; cases h
  | ok p =>
    obtain ⟨s, tPad⟩ := p
    rw [hz] at h
    simp only [bind, Except.bind, pure, Except.pure] at h
    repeat (split at h; · cases h)
    injection h with h
    exact ⟨_, _, _, _, _, h.symm⟩

/-- **The table of a level 0 `Level0.ofDraws` gave has its commitment's cap at the level 0's root**:
`Finish.tabOfLevel0` commits it as `commitOf` does, and a `build`'s cap is its root (`cap_root_build`). -/
theorem cap_of_drawn {m : Nat} {g : Array F128} {d : ByteArray} {l : _root_.Flock.Firewall.Level0.Level0}
    {st : _root_.Flock.Setup} {t : _root_.Flock.Firewall.Finish.Tab}
    (hl : _root_.Flock.Firewall.Level0.ofDraws m g d = .ok l)
    (ht : _root_.Flock.Firewall.Finish.tabOfLevel0 st l = .ok t) :
    _root_.Flock.Firewall.Finish.flat t.cm0.cap = l.root := by
  have hcm : t.cm0 = _root_.Flock.Firewall.Finish.commitOf l := by
    unfold _root_.Flock.Firewall.Finish.tabOfLevel0 at ht
    simp only [bind, Except.bind] at ht
    split at ht
    · cases ht
    split at ht
    · cases ht
    rcases hs : _root_.Flock.Zk.schedule st.m with e | ⟨sch, tPad⟩
    · rw [hs] at ht; cases ht
    · rw [hs] at ht
      simp only at ht
      split at ht
      · cases ht
      simp only [pure, Except.pure, Except.ok.injEq] at ht
      subst ht
      rfl
  obtain ⟨lv, tPad, extra, pads, salts, hb⟩ := build_of_ofDraws hl
  rw [hcm]
  exact cap_root_build hb

/-- **Past the session's `Commit` each level 0 bound to it is the one the loop drew for its table**: where the loop drew
`ls` for the session (`DrawInv`: none bound to it yet, and the level 0s sampled for it `ls`'s), table `j`'s being
`Level0.ofDraws` at its statement's `m` on what the host and the operating system handed the loop for rep stream `2j`,
the level 0 the slots `commitRead` leaves bind to the session's table `name` is the one sampled for the table the
`Commit` binds as `name`, the setup's table of that name (`commit_inv`). -/
theorem builtAt_commit {W Rr : Type} {M : LoopModel W Rr} {spec : SessionSpec} {sts : Array Stmt}
    {R : _root_.Flock.Firewall.Level0.Slots} {sid : Nat} {ls : Array _root_.Flock.Firewall.Level0.Level0}
    {f : ByteArray} (hd : DrawInv sid ls R)
    (hdr : ∀ j (hj : j < ls.size), ∃ g d hi, M.src (2 * j) = .ok (g, d, hi) ∧
      _root_.Flock.Firewall.Level0.ofDraws sts[j]!.m g d = .ok ls[j]) :
    BuiltAt M spec sts (commitRead R sid (spec.tables.zip (sts.map (·.m))) f).2 sid := by
  have hR : ∀ name l0, _root_.Flock.Firewall.Level0.ofSession R sid name = some l0 → False := by
    intro name l0 h
    unfold _root_.Flock.Firewall.Level0.ofSession at h
    obtain ⟨s, hs, -⟩ := Option.map_eq_some_iff.mp h
    exact hd.unbound s (Array.mem_of_find?_eq_some hs) name (by simpa using Array.find?_some hs)
  intro name l0 h
  unfold commitRead at h
  rcases hr : _root_.Flock.Firewall.Session.commitRoots f with _ | roots
  · rw [hr] at h
    exact (hR name l0 h).elim
  · rw [hr] at h
    dsimp only at h
    rcases hc : _root_.Flock.Firewall.Level0.commit R sid (spec.tables.zip (sts.map (·.m))) roots with e | R'
    · rw [hc] at h
      exact (hR name l0 h).elim
    · rw [hc] at h
      have h' : _root_.Flock.Firewall.Level0.ofSession R' sid name = some l0 := h
      obtain ⟨hmap, hI⟩ := commit_inv hc
      unfold _root_.Flock.Firewall.Level0.ofSession at h'
      obtain ⟨s, hs, rfl⟩ := Option.map_eq_some_iff.mp h'
      have hb : s.bound = some (sid, name) := by simpa using Array.find?_some hs
      have hsm := Array.mem_of_find?_eq_some hs
      obtain ⟨x, hx, hxn, hown, htx⟩ := hI.bound hd.unbound s hsm name hb
      obtain ⟨s₀, hs₀, hl, hown₀, ht₀, -⟩ := hI.same s hsm
      obtain ⟨hj, hl0⟩ := hd.draws s₀ hs₀ (by rw [← hown₀]; exact hown)
      obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hx
      simp only [List.length_zip, List.length_range] at hk
      have hk1 : k < roots.length := by omega
      have hk2 : k < (spec.tables.zip (sts.map (·.m))).toList.length := by omega
      simp only [List.getElem_zip, List.getElem_range] at hxn htx
      have hname : spec.tables[k]? = some name := by
        have e := congrArg (·[k]?) hmap
        simp only [List.getElem?_map, List.getElem?_eq_getElem hk1, List.getElem?_eq_getElem hk2, Option.map_some,
          Option.some.injEq, hxn] at e
        rw [Array.getElem_toList, Array.getElem_zip] at e
        rw [e]
        exact Array.getElem?_eq_getElem _
      have htk : s₀.table = k := by rw [← ht₀, htx]
      obtain ⟨g, d, hi, hsrc, hod⟩ := hdr s₀.table hj
      exact ⟨s₀.table, g, d, hi, by rw [htk]; exact hname, hsrc, by rw [hl, hl0]; exact hod⟩

/-- **The relay's invariant at the deployed loop**, against the model's run `M`, at the session `st`, the streams whose
rounds came being `seen`, the loop at `σ`: the session stopped in the loop's frames exactly where its `Hello` left and
an earlier round of the model's run failed (`liveBefore`); no stream seen before the `Hello` left; `RepJ` at each rep
stream seen; the session's plan the setup's once its `Hello` left; the streams seen those with a round before the
model's next (`SeenAt`); none seen while the session's slots took at most the model's `Hello`; past the `Commit`
every level 0 bound to the session the one the loop drew for its table (`BuiltAt`); once its `Hello` left the
session's entry the setup's (`SessOn`); the session's `Hello` and coins at the setup's link points (`LinksOk`); past
the `Commit`, where a level 0 is bound to the session, its entry linked to the history's link coins (`LinkOn`); once
its `Hello` left the setup's sizes within its frames' (`Session.sizesOk`); and, while the model's run has a message
there, what the loop took of the coins so far against the model's release (`LoopModel.TookAt`). -/
structure LoopJ {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt)
    (seen : List String) (st : State) (σ : LoopAt) : Prop where
  stop : σ.R.stopped sid = (st.sent && !M.L.toSrv.liveBefore M.wk M.x M.w M.ρ M.κ (M.hist σ.c))
  unsent : st.sent = false → seen = []
  reps : ∀ i r s, (i, r, s) ∈ spec.repStreams.toList → seen.contains s = true → RepJ M sid spec sts i r σ
  plan : PlanOk spec sts st
  seenAt : SeenAt spec sts (M.hist σ.c) seen
  early : σ.c ≤ 1 → seen = []
  built : 2 ≤ σ.c → BuiltAt M spec sts σ.L0 sid
  sess : st.sent = true → SessOn spec sts σ.R sid
  links : LinksOk spec st
  link : 2 ≤ σ.c → ∀ name l0, _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 → LinkOn M sid σ
  sizes : st.sent = true → _root_.Flock.Firewall.Session.sizesOk spec sts = true
  took : σ.c < M.τ.length → M.TookAt st σ.c

/-- **`RepJ` is kept from one of the session's requests to the next** where the updates between keep the rep's frame
(`Kept`) and what the session's rounds read of it (`Stays`), and the slot took no model message. -/
theorem repJ_stays {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {i r : Nat} {σ σ' : LoopAt} (hk : Kept sid i r σ.R σ'.R) (hs : Stays sid σ σ') (hc : σ'.c = σ.c)
    (h : RepJ M sid spec sts i r σ) : RepJ M sid spec sts i r σ' := by
  obtain ⟨p, rel, name, l0, hf, hss, hn, hl0, hA⟩ := h
  refine ⟨p, rel, name, l0, ?_, ?_, hn, ?_, ?_⟩
  · rw [frameAt_kept hk (by rw [hf]; rfl)]
    exact hf
  · rw [sessAt_stay (hs.2 name l0 hl0) (by rw [hss]; rfl)]
    exact hss
  · rw [hs.1 name]
    exact hl0
  · rw [hc]
    exact streamAt_stay (hs.2 name l0 hl0) hA

/-- **`Hello`'s answer, where it checks, is `Resp::Coins`' frame**, so its first byte is the coins' tag. -/
theorem answer_head {l : List UInt8} {p : ByteArray × List UInt8}
    (h : _root_.Flock.Firewall.Coins.answer l = .ok p) : l.head? = some 2 := by
  unfold _root_.Flock.Firewall.Coins.answer at h
  cases hb : (l.length == 325 && l.take 5 == _root_.Flock.Firewall.Coins.frame 20 [])
  · rw [hb] at h
    simp at h
  · simp only [Bool.and_eq_true, beq_iff_eq] at hb
    have e := congrArg List.head? hb.2
    simpa [List.head?_take, _root_.Flock.Firewall.Coins.frame] using e

/-- **The auditor's answer that begins the session's coin tree is coins.** -/
theorem start_byte {h : _root_.Flock.Firewall.Coins.Hello} {a : ByteArray} {C : _root_.Flock.Firewall.Coins.Release}
    (hs : _root_.Flock.Firewall.Coins.Release.start h a.data.toList = .ok C) : byte? a 0 = some 2 := by
  have hh : a.data.toList.head? = some 2 := by
    unfold _root_.Flock.Firewall.Coins.Release.start at hs
    rcases ha : _root_.Flock.Firewall.Coins.answer a.data.toList with e | p
    · rw [ha] at hs
      cases hs
    · exact answer_head ha
  obtain ⟨⟨l⟩⟩ := a
  cases l with
  | nil => cases hh
  | cons b l =>
    cases hh
    rfl

/-- **Past `Hello`'s answer, where the session goes on, its coin tree is the one the answer begins** under the
session's `Hello`. -/
theorem answer_hello_start {S : State} {r id : ByteArray} (h : S.stop = none)
    (hn : (S.answer .hello r id).1.stop = none) :
    ∃ hh C, S.hello = some hh ∧ _root_.Flock.Firewall.Coins.Release.start hh r.data.toList = .ok C ∧
      (S.answer .hello r id).1.session = some C := by
  revert hn
  unfold State.answer
  simp only [h]
  rcases hhel : S.hello with _ | hh
  · dsimp only
    split <;> intro hn <;> exact absurd hn (Option.isSome_iff_ne_none.mp rfl)
  · dsimp only
    rcases hs : _root_.Flock.Firewall.Coins.Release.start hh r.data.toList with e | C
    · intro hn
      exact absurd hn (Option.isSome_iff_ne_none.mp rfl)
    · intro _
      exact ⟨hh, C, rfl, hs, rfl⟩

/-- **The `Hello` that leaves at the setup `spec`, `sts`** leaves the session reading the setup's `Hello` at its
nonce. -/
theorem sendHello_hello {S S' : State} {spec : SessionSpec} {sts : Array Stmt} {n o : ByteArray}
    (he : S.sendHello (.ok (spec, sts)) n = .ok (S', o)) : S'.hello = helloOf spec sts n := by
  unfold State.sendHello at he
  rcases hst : S.stop with _ | why
  · rw [hst] at he
    dsimp only at he
    split at he
    · cases he
    · split at he
      · cases he
      · cases he
        rfl
  · rw [hst] at he
    cases he

/-- **Past the `Hello`, where the session goes on, its coin tree is the one the auditor's answer begins** under the
setup's `Hello` at the request's nonce. -/
theorem stepK_hello_start {st : State} {pl : Placed} {e : Env} {spec : SessionSpec} {sts : Array Stmt}
    (h : st.stop = none) (he : e.setup = .ok (spec, sts))
    (hgo : ((stepK true st pl (some (tagOf .hello)) (.ok .hello) e).go &&
      (stepK true st pl (some (tagOf .hello)) (.ok .hello) e).st.stop.isNone) = true) :
    ∃ hh a C, helloOf spec sts e.nonce = some hh ∧ e.ans = some a ∧
      _root_.Flock.Firewall.Coins.Release.start hh a.data.toList = .ok C ∧
      (stepK true st pl (some (tagOf .hello)) (.ok .hello) e).st.session = some C := by
  obtain ⟨hg, hn⟩ := go_live hgo
  rcases hr : st.request pl .hello with _ | ⟨st', o⟩
  · rw [stepK_hello hr] at hg hn ⊢
    rcases hh : st.sendHello e.setup e.nonce with ⟨st', o⟩ | ⟨st', o⟩
    · rw [hh] at hn
      exact absurd hn (Option.isSome_iff_ne_none.mp (sendHello_error h hh))
    · obtain ⟨_, _, hstop, -⟩ := sendHello_ok hh
      have hhel : st'.hello = helloOf spec sts e.nonce := by
        rw [he] at hh
        exact sendHello_hello hh
      rcases ha : e.ans with _ | a
      · rw [hh, ha] at hg
        simp at hg
      · rw [hh, ha] at hn
        obtain ⟨hh', C, hs1, hs2, hs3⟩ := answer_hello_start (r := a) (id := e.id)
          (show st'.stop = none by rw [hstop, h]) hn
        exact ⟨hh', a, C, hhel.symm.trans hs1, rfl, hs2, hs3⟩
  · rw [stepK_refused (by decide) (by decide) (by decide) hr] at hn
    exact absurd hn (Option.isSome_iff_ne_none.mp (request_stop h hr))

/-- **Coins the session goes on past are opened by its coin tree** as the schedule's next opening (`reqAt`), and the
session after holds the tree after it. -/
theorem coins_open {st : State} {r : ByteArray} {C : _root_.Flock.Firewall.Coins.Release} (h : st.stop = none)
    (hs : st.session = some C) (hn : (st.coins r).1.stop = none) :
    ∃ C' c, C.open (reqAt st.plan C.hello.links C.opened.length).1 (reqAt st.plan C.hello.links C.opened.length).2
      r.data.toList = .ok (C', c) ∧ (st.coins r).1.session = some C' := by
  revert hn
  unfold State.coins
  rw [h, hs]
  dsimp only
  rcases ho : C.open (reqAt st.plan C.hello.links C.opened.length).1 (reqAt st.plan C.hello.links C.opened.length).2
      r.data.toList with e | ⟨C', c⟩
  · intro hn
    exact absurd hn (Option.isSome_iff_ne_none.mp rfl)
  · intro _
    exact ⟨C', c, rfl, rfl⟩

/-- **Past a `Commit` or a frame round, where the session goes on, its coin tree is the one its tree before opens the
auditor's coins to** as the schedule's next opening (`reqAt`). -/
theorem stepK_coin_open {st : State} {pl : Placed} {k : _root_.Flock.Firewall.Session.Kind} {e : Env}
    {C : _root_.Flock.Firewall.Coins.Release} (h : st.stop = none) (hk : k = .commit ∨ ∃ s n, k = .round s n)
    (hC : st.session = some C)
    (hgo : ((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
      (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true) :
    ∃ a C' c, e.ans = some a ∧ byte? a 0 = some 2 ∧
      C.open (reqAt st.plan C.hello.links C.opened.length).1 (reqAt st.plan C.hello.links C.opened.length).2
        a.data.toList = .ok (C', c) ∧
      ∀ D, (stepK true st pl (some (tagOf k)) (.ok k) e).st.session = some D → D = C' := by
  obtain ⟨hg, hn⟩ := go_live hgo
  have hk' : k ≠ .hello ∧ k ≠ .finish ∧ k ≠ .ping ∧ k ≠ .proof := by
    rcases hk with rfl | ⟨s, n, rfl⟩ <;> exact ⟨by simp, by simp, by simp, by simp⟩
  rcases hr : st.request pl k with _ | ⟨st', o⟩
  · rcases stepK_frame_ans hk'.1 hk'.2.1 hk'.2.2.1 hk'.2.2.2 hr hg with hst | ⟨a, ha, hst, -⟩
    · exact absurd hn (Option.isSome_iff_ne_none.mp hst)
    · rw [hst] at hn ⊢
      have haa := afterAnswer_coin hn
      rw [haa] at hn ⊢
      have hb : byte? a 0 = some 2 := by
        by_cases hb : byte? a 0 = some 2
        · exact hb
        · rcases hk with rfl | ⟨s, n, rfl⟩
          · rw [answer_commit, if_neg hb] at hn
            exact absurd hn (Option.isSome_iff_ne_none.mp (noCoins_stop _ _ _))
          · rw [answer_round, if_neg hb] at hn
            exact absurd hn (Option.isSome_iff_ne_none.mp (noCoins_stop _ _ _))
      have hco : (st.answer k a).1 = (st.coins a).1 := by
        rcases hk with rfl | ⟨s, n, rfl⟩
        · rw [answer_commit, if_pos hb]
        · rw [answer_round, if_pos hb]
      rw [hco] at hn ⊢
      obtain ⟨C', c, ho, hs'⟩ := coins_open h hC hn
      exact ⟨a, C', c, ha, hb, ho, fun D hD => Option.some.inj (hD.symm.trans hs')⟩
  · obtain ⟨h9, h5, h6⟩ := tagOf_ne hk'.2.2.1 hk'.2.2.2 hk'.2.1
    rw [stepK_refused h9 h5 h6 hr] at hn
    exact absurd hn (Option.isSome_iff_ne_none.mp (request_stop h hr))

/-- **`LoopJ` is kept at a slot that reads no model message**: where the loop's reading of it brings only updates that
keep every rep's frame, the session's stop and what its rounds read of it, and takes no model message; such a slot
keeps the session's coin tree (`stepK_ctl_session`), so what the loop took stays. -/
theorem loopKept_ctl {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {k : _root_.Flock.Firewall.Session.Kind} (hm : isModel k = false)
    (hs : ∀ seen, seenAfter seen k = seen)
    (hr : ∀ st pl e σ σ', LoopRead M sid spec sts nonce k st pl e σ σ' →
      Between sid σ.R σ'.R ∧ σ'.c = σ.c ∧ Stays sid σ σ') :
    LoopKept M sid spec sts nonce (LoopJ M sid spec sts) k := by
  intro seen st pl e σ σ' hsl he hF hJ hra hgo
  obtain ⟨hbt, hcc, hst⟩ := hr _ _ _ _ _ hra
  have hk : (Except.ok k : Except String _root_.Flock.Firewall.Session.Kind) ≠ .ok .hello := by
    intro h
    cases h
    cases hm
  have hsent := stepK_sent (hold := true) (st := st) (pl := pl) (b := some (tagOf k)) (e := e) hk
  rw [hs seen]
  refine ⟨?_, fun h => hJ.unsent (by rw [hsent] at h; exact h),
    fun i r s hmem hseen => repJ_stays (hbt.1 i r) hst hcc (hJ.reps i r s hmem hseen), stepK_plan he hJ.plan, ?_,
    fun h => hJ.early (by rwa [hcc] at h), fun h => builtAt_stays hst (hJ.built (by rwa [hcc] at h)),
    fun h => sessOn_sessKept hbt.2 (hJ.sess (by rw [hsent] at h; exact h)), stepK_links he hJ.links,
    fun h => linkOn_stays (hJ.sess (sent_of_two hF (by rwa [hcc] at h))) hst (by rwa [hcc] at h) (by omega)
      (hJ.link (by rwa [hcc] at h)), fun h => hJ.sizes (by rw [hsent] at h; exact h), ?_⟩
  · rw [hsent, hcc, stopped_sessKept hbt.2]
    exact hJ.stop
  · rw [hcc]
    exact hJ.seenAt
  · intro h
    rw [hcc] at h ⊢
    exact LoopModel.TookAt.congr (stepK_ctl_session hF.live hsl hm hgo) (hJ.took h)

/-- **`LoopJ` is kept at every slot that reads no model message**: each such slot's reading is only the updates between
the session's requests, keeping what its rounds read of it. -/
theorem loopKept_ctl_all {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} : ∀ k, isModel k = false → LoopKept M sid spec sts nonce (LoopJ M sid spec sts) k := by
  intro k hm
  cases k with
  | register => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | link => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | openStream s => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | proof => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | ping => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | other => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | hello => cases hm
  | commit => cases hm
  | round _ _ => cases hm
  | finish => cases hm

/-- **Where the session goes on past its `Hello`, the setup's sizes are within its frames'**: the `Hello` left, which
`State.sendHello` lets only at a setup passing `Session.sizesOk` (`Loop.sizesOk_of_sendHello`). -/
theorem stepK_hello_sizes {st : State} {pl : Placed} {e : Env} {spec : SessionSpec} {sts : Array Stmt}
    (he : e.setup = .ok (spec, sts))
    (hgo : ((stepK true st pl (some (tagOf .hello)) (.ok .hello) e).go &&
      (stepK true st pl (some (tagOf .hello)) (.ok .hello) e).st.stop.isNone) = true) :
    _root_.Flock.Firewall.Session.sizesOk spec sts = true := by
  rcases hr : st.request pl .hello with _ | ⟨st', o⟩
  · have hs : st.stop = none := by
      revert hr
      unfold State.request
      split
      · intro h
        cases h
      · intro _
        assumption
    rw [stepK_hello hr] at hgo
    rcases hh : st.sendHello e.setup e.nonce with ⟨st', o⟩ | ⟨st', o⟩
    · rw [hh] at hgo
      have hst := sendHello_error hs hh
      simp only [Bool.true_and] at hgo
      rw [Option.isSome_iff_ne_none] at hst
      exact absurd (Option.isNone_iff_eq_none.mp hgo) hst
    · rw [he] at hh
      exact Loop.sizesOk_of_sendHello hh
  · rw [stepK_refused (by decide) (by decide) (by decide) hr] at hgo
    have hst : st'.stop.isSome := by
      rcases hs : st.stop with _ | why
      · exact request_stop hs hr
      · revert hr
        unfold State.request
        rw [hs]
        intro h
        cases h
        rw [hs]
        rfl
    simp only [Bool.true_and] at hgo
    rw [Option.isSome_iff_ne_none] at hst
    exact absurd (Option.isNone_iff_eq_none.mp hgo) hst

/-- **`LoopJ` is kept at the `Hello`**: before it no stream was seen and the session had not stopped; past it the
session has sent its `Hello`, the model's `Hello` round passed, and the updates between keep the session's stop; and,
where the model's release reads the setup's `Hello` at the nonce (`hchk`), the session's coin tree is the one the
model's first answer begins (`LoopModel.TookAt.hello`). -/
theorem loopKept_hello {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} (hchk : ∀ hl, helloOf spec sts nonce = some hl → M.L.helloChk M.x M.ρ = .ok hl) :
    LoopKept M sid spec sts nonce (LoopJ M sid spec sts) .hello := by
  intro seen st pl e σ σ' hsl he hF hJ hra hgo
  obtain ⟨hans, hnon, hbt, hcc, hlb, hso⟩ := hra
  have hs0 : st.sent = false := (slot_hello hsl).2
  have hseen : seen = [] := hJ.unsent hs0
  have hsent := stepK_hello_sent hgo
  refine ⟨?_, fun h => ?_, fun i r s _ hc => ?_, stepK_plan he hJ.plan, ?_, fun _ => hseen,
    fun h => absurd h (by rw [hcc, hF.start hs0]; omega), fun _ => hso he, stepK_links he hJ.links,
    fun h => absurd h (by rw [hcc, hF.start hs0]; omega), fun _ => stepK_hello_sizes he hgo, ?_⟩
  · rw [hsent, hlb, stopped_sessKept hbt.2, hJ.stop, hs0]
    rfl
  · rw [hsent] at h
    cases h
  · simp [seenAfter, hseen] at hc
  · show SeenAt spec sts (M.hist σ'.c) seen
    rw [hseen]
    exact seenAt_nil (by rw [hcc, hF.start hs0, LoopModel.hist, length_histOf_take]; omega)
  · intro _
    obtain ⟨hh, a, C, hho, hae, hst, hsC⟩ := stepK_hello_start hF.live he hgo
    obtain ⟨hj, hA⟩ := hans
    have key : ∀ c, c = 0 → ∀ hj : c < M.τ.length, (M.τ[c]'hj).2 = .msg a.data.toList →
        ∃ hlt : 0 < M.τ.length, (M.τ[0]'hlt).2 = .msg a.data.toList := by
      intro c hc hj h
      subst hc
      exact ⟨hj, h⟩
    obtain ⟨hlt, hm⟩ := key σ.c (hF.start hs0) hj (hA a hae (start_byte hst))
    rw [hcc, hF.start hs0]
    refine LoopModel.TookAt.hello hlt (hchk hh (by rw [← hnon]; exact hho)) fun C' hC' => ?_
    rw [hsC] at hC'
    cases hC'
    rw [hm]
    exact hst

/-- **Past the `Hello` every earlier round passed**, below the `Finish`: the `Hello` passes (`toSrv_pass_own`). -/
theorem liveBefore_one {W Rr : Type} (L : LeanRelease W Rr) (wk : L.toSrv.Worker) (x : Claims.Msg) (w : W) (ρ : Rr)
    (κ : L.toSrv.Chk) (h : List (Claims.Msg × Claims.Msg)) (hl : h.length = 1) (hlt : 0 < L.last x) :
    L.toSrv.liveBefore wk x w ρ κ h = true := by
  unfold ZkServer.liveBefore
  rw [hl]
  simp only [List.all_eq_true, List.mem_range]
  intro j hj
  obtain rfl : j = 0 := by omega
  refine L.toSrv_pass_own wk x w ρ κ (h.take 0) ?_ ?_
  · rw [List.take_zero, List.length_nil]
    exact L.kindAt_zero x
  · rw [List.take_zero, List.length_nil]
    exact hlt

/-- **`LoopJ` is kept at the `Commit`**: the session's slots took only the model's `Hello` before it, so no stream was
seen; the updates between keep the session's stop; and every round before the model's next passed, before the link
and after it, where the worker's link is the ZK server's own, below the `Finish` (`liveBefore_one`, `liveBefore_two`);
and past it, at a level 0 bound to the session, the session's entry is linked to the history's link coins
(`linkOn_commit`): that level 0 is a table's (`BuiltAt`), so the setup has a rep stream, whose sizes (`LoopJ.sizes`)
put the setup's link points within `Frames.COIN_MAX` words; and, where the model's release requests the loop's
openings (`hreq`) and frames its first answer (`hf1`), the session's coin tree past it is the one its tree before
opens the model's answer to at the schedule's first opening (`LoopModel.TookAt.succ`). -/
theorem loopKept_commit {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {s0 : FlockDeploy.Ins} (hx : FlockDeploy.setupOf true M.x = some (s0, spec, sts))
    (hlast : 1 < M.L.last M.x)
    (hwk : M.wk M.x M.w M.ρ (M.hist 1) 0 = some (M.L.ownMsg M.x M.w M.ρ (M.hist 1), false))
    (hreq : ∀ j, M.L.req M.x j = Loop.reqOf M.x j)
    (hf1 : 0 < M.τ.length → (M.L.frameAnswer M.x M.ρ (M.hist 1)).isOk = true) :
    LoopKept M sid spec sts nonce (LoopJ M sid spec sts) .commit := by
  intro seen st pl e σ σ' hsl he hF hJ hra hgo
  have hlk : spec.points * spec.mPts ≤ _root_.Flock.Firewall.Frames.COIN_MAX → LinkOn M sid σ' :=
    fun hpm => linkOn_commit hx hpm hsl hF hJ.links hra hgo
  obtain ⟨⟨hlt, hA⟩, ⟨_, -, hL0⟩, -, ⟨_, _, hda, hdr, -⟩, -, hbt, hcc⟩ := hra
  obtain ⟨hs1, -, hop⟩ := slot_commit hsl
  have hbuilt : BuiltAt M spec sts σ'.L0 sid := by
    intro name l0 hl
    rw [hL0 name] at hl
    exact builtAt_commit (afterDraw_inv hda) hdr name l0 hl
  have hc1 : σ.c = 1 := by
    have := hF.count hs1
    omega
  have hseen : seen = [] := hJ.early (by omega)
  have hk : (Except.ok .commit : Except String _root_.Flock.Firewall.Session.Kind) ≠ .ok .hello := by
    intro h
    cases h
  have hsent := stepK_sent (hold := true) (st := st) (pl := pl) (b := some (tagOf .commit)) (e := e) hk
  have hl1 : (M.hist σ.c).length = 1 := by
    rw [LoopModel.hist, length_histOf_take]
    omega
  have hl2 : (M.hist σ'.c).length = 2 := by
    rw [LoopModel.hist, length_histOf_take]
    omega
  have ht : (M.hist σ'.c).take 1 = M.hist 1 := histOf_take_take M.τ (i := σ'.c) (i' := 1) (by omega)
  have hlb1 := liveBefore_one M.L M.wk M.x M.w M.ρ M.κ _ hl1 (by omega)
  have hlb2 := liveBefore_two M.L M.wk M.x M.w M.ρ M.κ _ hl2 hlast (by rw [ht]; exact hwk)
  refine ⟨?_, fun _ => hseen, fun i r s _ hc => ?_, stepK_plan he hJ.plan, ?_, fun _ => hseen, fun _ => hbuilt,
    fun _ => sessOn_sessKept hbt.2 (hJ.sess hs1), stepK_links he hJ.links, fun _ name l0 hl => ?_,
    fun _ => hJ.sizes hs1, ?_⟩
  · rw [hsent, stopped_sessKept hbt.2, hJ.stop, hlb1, hlb2]
  · simp [seenAfter, hseen] at hc
  · show SeenAt spec sts (M.hist σ'.c) seen
    rw [hseen]
    exact seenAt_nil (by omega)
  · obtain ⟨j, -, -, -, hj, -⟩ := hbuilt name l0 hl
    have hk : 0 < spec.repStreams.size := by
      have := (Array.getElem?_eq_some_iff.mp hj).1
      rw [Claims.Bridge.Custody.repStreams_size]
      omega
    exact hlk (Loop.sizesOk_at (hJ.sizes hs1) hk).1.2.2
  · intro _
    obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hF.reach.sess hs1)
    obtain ⟨a, C', c, hae, hb, ho, hs'⟩ := stepK_coin_open hF.live (.inl rfl) hC hgo
    have hm := hA a hae hb
    have hlen : C.opened.length = 0 := by
      rw [← opened_of hC]
      exact hop
    have hq : reqAt st.plan C.hello.links C.opened.length = M.L.req M.x (σ.c - 1) := by
      rw [hreq, Loop.reqOf_eq hx, hJ.plan hs1, hJ.links.2 C hC, hlen, show σ.c - 1 = 0 by omega]
    rw [hcc]
    refine LoopModel.TookAt.succ hlt (hJ.took hlt) (by omega) hC (by rw [← hq, hm]; exact ho) hs'
      (by rw [hc1]; exact hf1 (by omega)) (M.L.fed_of_le M.x M.ρ ?_)
    rw [LoopModel.hist, length_histOf_take]
    omega

end Claims.ZKServer.Relay
