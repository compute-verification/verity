import Proofs.Flock.Claims.ZK.FrameAdmits
import Proofs.Flock.Claims.ZK.FrameTape
import Proofs.Flock.Claims.ZK.PiopSrcVals

/-!
# C-Flock's zero knowledge at the session loop's own frames (trusted)

`SessionLoopZK` is the session loop's zero-knowledge bound as a
guarantee: at the deployed relation (`Flock.Readable.relation`), at the release whose transport state is the loop's own
frames (`LeanRelease.ofLoop`, each rep's frame driven on `Drive.reqs` from the decoded coins), at every statement `x` the
loop serves (`Loop.Serves`: its `Hello` leaves at `x`'s setup, which it refuses where the setup's sizes are past its
frames', `Session.sizesOk`), and at the hiding gap of hm96-sha512's pinned key (`hidingGap My saltHash`, with no hiding
premise). It concludes that the deployed frame's state is the loop's but for `stopped` (`ProgChecks`) and the bound.

Each rep's table is the loop's own (`Loop.realOf`: `tabOfLevel0` at the level 0 `Level0.ofDraws` draws), at every
level-0 source `src` (the host's packed witness, the bytes drawn for level 0 and the salts for levels ≥ 1). What it
takes of the loop beyond the code is named, each an obligation:

* `LoopInnerRound0`: `L₀`'s round at the inner proof's round 0 is the firewall's own at some 16-byte `final_c'` (Rust's
  firewall sends that round, `check_self`, at the inner proof's label).
* `TablesAdmitUnrefused`, on the tables `T` alone: where, at a `Live` statement, no coin fault shows (`Masked.Refused`),
  every table admits the coins. `SessionLoopZK` takes it; at the deployed tables it is proved
  (`LeanRelease.tablesAdmitUnrefused_dataE`), so `SessionLoopZKDeployed` does not.

`SessionLoopZKDeployed` is `SessionLoopZKAt Masked.dataE` at the loop's own link and `Finish` (`LeanRelease.ofLoopLink`:
`withLink`, the roots of the level 0 the loop drew, `Loop.linkOf`, as its `Commit` carries them; `withFiles`, the record
and proof files of the loop's `Finish`, `Loop.filesE`). In place of `LinkRoundEq` it
takes the masked verifier's scope (`Claims.Flock.Scope`) and `Loop.Src` (`flock-firewall-os-src`): every input the loop
takes from its host and operating system is the tape's, its level-0 inputs (`Loop.Level0Src`) and its rep draws
(`Loop.RepSrc`).

The transport's other trusts (`LeanRelease.LinkRoundEq`, `FinishIs`, `LiveRoundEq`, `FillRoundEq`) and the tables' side
(`CoinLayer` `Tables`, `Complete`, `DummyReg`, `Causal`, `SwCausal`, `TapeOK`) are the statement's premises, with
`FinishIs` in place of the `FinalRoundEq` `Flock.zk_server_lean_loop` takes (`LeanRelease.finalRoundEq_ofLoop`, with the
loop's frames refusing every coin fault, `LeanRelease.frameUnrefused_ofLoop`). The proofs are
`Flock.SecurityProofs.SessionLoopZK` and `SessionLoopZKDeployed` (`Proofs.Flock.SessionLoop`),
`Flock.zk_server_lean_ofLoop` at the gap.

The loop's own `Finish` (`LeanRelease.withFiles`) is its record and proof files, as functions of the statement, the
witness, the tape and the history, and whether it stops:

* **The proof files** (`Loop.proofsOf`): rep stream `k`'s is `Finish.proofFile` on its table and salts for levels ≥ 1
  (`Loop.realOf src`, the loop's own level 0) and on the rep's request (`Loop.repInOf`), which is what the loop holds of the
  rep (`Frames.ownOf`): the frame's draws, the extra claims at the session's link points (`Drive.extras`), the coins the
  stream answered each of its rounds, and the values its frame took. Those values are read back off the rounds the frame
  released (`Loop.maskedOf`): each leaves masked, `v + h_i` under the rep's pad `h_i` (`Frame.Masker.absorb`), so at a pad
  of the absorbed order the value is the released word plus the pad (`Loop.valsOf`). `final_c` is never absorbed: it is
  the masked word of the inner proof's round 0 (`final_c'`, `LoopInnerRound0`) plus its pad, which is the value the loop's
  finish computes its file from where the `inner` request that computed it carried it (`Finish.serve`'s `liveOnly`: the
  file the loop serves is the one its `inner` request computed live).
* **The record** (`Loop.recordOf`): the session loop sends the auditor its proofs and then `Finish`, no record of its own,
  so the record is the one the auditor assembles from what it saw, `recordJson` of each rep stream's rounds as they left
  the loop (`Loop.seenRounds`: each round's bytes, coin count and the coins the stream answered; `Loop.streamOf`: round
  `0`'s bytes after the stream's domain frame, which the loop sends before the frame's release), the link's roots (the
  level 0 the loop drew, `Loop.level0Of`) and points (the `Commit`'s coins), and the proof files. This is a modelling
  choice: the record the analysis's `Final.encode` reads is the auditor's own of the streams it saw.
* **The stop**: the loop's `Finish` stops where any of this throws (`Loop.filesE`), a rep's proof file above all
  (`Finish.proofFile`: its `repFw`, whose constraints fail at a `final_c` the passed values do not fix).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker labelFrame)
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- **The session loop's own tables** at the level-0 source `src`: rep stream `k`'s table is table `k / 2`'s
`tabOfLevel0` at the level 0 `Level0.ofDraws` draws at its `m` from `src`'s packed witness and drawn bytes, with `src`'s
salts for levels ≥ 1 (`Frames.placeOf` on the loop's `Level0`, which `Level0.draw` drew on the host's packed witness and
the operating system's bytes, and `Level0.commit` binds at the table's `m`; `Finish.serveLoop`'s salts). -/
def Loop.realOf {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) (k : ℕ) : V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray) :=
  match FlockDeploy.setupOf true x with
  | some (_, _, sts) => do
    let (g, d, hi) ← src x w ρ k
    let l ← _root_.Flock.Firewall.Level0.ofDraws sts[k / 2]!.m g d
    let t ← _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit sts[k / 2]!) l
    pure (t, hi)
  | none => .error "no session"

/-- **The level 0 the loop draws for table `j`** at the source `src`: `Level0.ofDraws` at table `j`'s `m` on the packed
witness and drawn bytes `src` gives rep stream `2j` (one level 0 a table, as `Loop.realOf` reads it). -/
def Loop.level0Of {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) (sts : Array Stmt) (j : ℕ) : V _root_.Flock.Firewall.Level0.Level0 := do
  let (g, d, _) ← src x w ρ (2 * j)
  _root_.Flock.Firewall.Level0.ofDraws sts[j]!.m g d

/-- **The loop's own link** at the source `src`: each table's drawn level-0 root, with no public words, as the loop's
`Commit` carries them (`Session.relay`'s `commitFrame` of the roots `Level0.ofSession` bound); the empty root where the
draw is refused, which the loop's `Commit` check refuses. -/
def Loop.linkOf {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) : Msg :=
  match FlockDeploy.setupOf true x with
  | some (_, _, sts) =>
    0 :: FlockDeploy.linkBody
      (List.ofFn fun j : Fin sts.size => ((Loop.level0Of src x w ρ sts j.val).toOption.map (·.root)).getD .empty)
      (List.replicate sts.size ByteArray.empty)
  | none => []

/-- `L₀` with the loop's own link at the source `src` (`Loop.linkOf`): the link the loop's `Commit` carries, which it
builds itself from the level 0 it drew. -/
def LeanRelease.withLink {R : Type} (L₀ : LeanRelease Flock.Wit R)
    (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)) : LeanRelease Flock.Wit R :=
  { L₀ with link := fun x w ρ _ => Loop.linkOf src x w ρ }

namespace Loop

/-- Rep stream `k`'s first rep round, at the rep streams' round counts `rs`: the rounds of the streams before it. -/
def streamAt (rs : List ℕ) (k : ℕ) : ℕ := (rs.take k).sum

/-- **Rep stream `k`'s rounds as they left the loop** on the history `h` at `x`, at the rep streams' round counts `rs`: its
round `j`, rep round `streamAt rs k + j`, the history's entry past `Hello` and the link that carries it (`progOf`'s
placing), as its bytes (the frame's release past the round message's tag `0`, `LeanRelease.release`), its coin count
(`loopNr`) and the words of the coins the stream answered it (`wordsAt`). -/
noncomputable def seenRounds (x : Msg) (h : List (Msg × Msg)) (rs : List ℕ) (k : ℕ) :
    List (ByteArray × ℕ × Array F128) :=
  let cs := h.tail.map fun p => CoinLayer.rd p.2
  (List.range (rs.getD k 0)).map fun j =>
    let q := streamAt rs k + j + 1
    (⟨((h.tail.getD q ([], [])).1).tail.toArray⟩, loopNr x q, wordsAt x cs q)

/-- **The masked values a rep's released rounds carry**, in the order its frame absorbed them, at its configuration `c`
from the frame's state `s`: at each round, the words at the offsets `Frame.roundFree` frees (the algebraic phase's
absorbed values), then the frame takes the round's coins (`Masker.answer`). It ends at a round the frame does not take
or coins it refuses. -/
def maskedOf (c : Cfg) : Masker → List (ByteArray × ℕ × Array F128) → Array F128
  | _, [] => #[]
  | s, (r, n, cs) :: rest =>
    match _root_.Flock.Firewall.Frame.roundFree c s r n with
    | .error _ => #[]
    | .ok (_, s', free) =>
      let here := free.map fun o => (_root_.Flock.Firewall.Frame.words r o 1)[0]!
      match s'.answer c cs with
      | .ok s'' => here ++ maskedOf c s'' rest
      | .error _ => here

/-- `final_c'`, the masked `final_c` the inner proof's round 0 `r` carries between its headers: the 16 bytes past its
label frame and its observe's header (`LoopInnerRound0`'s shape). -/
def finalCOf (r : ByteArray) : F128 :=
  let o := (_root_.Flock.Firewall.Frame.labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16
  (_root_.Flock.Firewall.Frame.words r o 1)[0]!

/-- **The values a rep's frame took, by pad index**, at its configuration `c`: at the `p`-th pad of the absorbed order,
the `p`-th masked value plus that pad; at `final_c`'s, `final_c'` plus its pad; `0` elsewhere (`Frames.ownOf`'s array of
`zp 64` values). -/
def valsOf (c : Cfg) (masked : Array F128) (fc : F128) : Array F128 :=
  let v := (c.order.zip masked).foldl (fun v (i, m) => v.set! i (m + c.h.getD i 0))
    (Array.replicate (c.lay.zp 64) 0)
  v.set! c.lay.fc (fc + c.h.getD c.lay.fc 0)

/-- **What the loop holds of rep stream `k`** on the history `h` at `x`, at the frame draws `draws` and salts for levels ≥ 1
`hi`, as its finish's request (`Finish.RepIn`): the draws at the rep's layout (`Drive.start`), the values its frame took
(`valsOf` of the rounds it released, `final_c'` from the inner proof's round 0, the stream's round past its scheduled
ones), the extra claims at the session's link points, `hi`, and the coins the stream answered each round. -/
noncomputable def repInOf (x : Msg) (h : List (Msg × Msg)) (spec : SessionSpec) (sts : Array Stmt) (k : ℕ)
    (draws : ByteArray) (hi : Array ByteArray) : V _root_.Flock.Firewall.Finish.RepIn := do
  let st := sts[k / 2]!
  let (lay, _) ← _root_.Flock.Firewall.Drive.start st
  let d := _root_.Flock.Firewall.Frame.Draws.ofBytes lay draws
  let c := Cfg.ofDraws lay d none
  let rounds := seenRounds x h (roundsOf spec sts) k
  let nSched := rounds.length - _root_.Flock.Firewall.Frame.INNER_ROUNDS
  let fc := finalCOf ((rounds.getD nSched (.empty, 0, #[])).1)
  let link := wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) 0
  return { draws := d, vals := valsOf c (maskedOf c {} rounds) fc,
           xs := _root_.Flock.Firewall.Drive.extras st (_root_.Flock.Firewall.Drive.linkPoints st link),
           hiSalts := hi, coins := (rounds.map fun r => r.2.2).toArray }

/-- **The loop's proof files** on the history `h` at `x`, the witness `w` and the tape `ρ`, at the level-0 source `src` and
the frame draws `draws`: rep stream `k`'s, in `spec.repStreams` order, `Finish.proofFile` on its table and salts
(`realOf src`) and its request (`repInOf`), at its domain. -/
noncomputable def proofsOf {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (draws : R → ℕ → ByteArray) (x : Msg) (w : Flock.Wit) (ρ : R) (h : List (Msg × Msg)) (spec : SessionSpec)
    (sts : Array Stmt) : V (Array ByteArray) :=
  (Array.range spec.repStreams.size).mapM fun k => do
    let (t, hi) ← realOf src x w ρ k
    let q ← repInOf x h spec sts k (draws ρ k) hi
    _root_.Flock.Firewall.Finish.proofFile t (k % 2) (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 q

/-- **The roots the loop's `Commit` carries**: each table's drawn level-0 root (`level0Of`), the empty one where the
draw is refused. -/
def rootsOf {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) (sts : Array Stmt) : Array ByteArray :=
  (Array.range sts.size).map fun j => ((level0Of src x w ρ sts j).toOption.map (·.root)).getD .empty

/-- **The link points the auditor saw** on the history `h` at `x`: the `Commit`'s coins, `spec.points` points of
`spec.mPts` words each. -/
noncomputable def pointsOf (x : Msg) (h : List (Msg × Msg)) (spec : SessionSpec) : Array (Array F128) :=
  (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j =>
    (wordsAt x (h.tail.map fun e => CoinLayer.rd e.2) 0).getD (p * spec.mPts + j) 0

/-- **Rep stream `k`'s rounds as the auditor records them**: `seenRounds`, each with its coin count, no digest, its
coins and its bytes (the prover transcript's shape, `squeezeP`), round `0`'s after the stream's domain frame
(`Frames.domainFrame`, `Transcript.init`'s), which the frame releases it without and the session loop sends before it
(`Frames.roundStep`). -/
noncomputable def streamOf (x : Msg) (h : List (Msg × Msg)) (spec : SessionSpec) (sts : Array Stmt) (k : ℕ) :
    Array _root_.Flock.Round :=
  ((seenRounds x h (roundsOf spec sts) k).mapIdx fun j (b, n, cs) =>
    ({ n, sha := .empty, coins := cs,
       bytes := some (if j = 0 then
         _root_.Flock.Firewall.Frames.domainFrame (spec.domain spec.tables[k / 2]! (k % 2)) ++ b else b) } :
      _root_.Flock.Round)).toArray

/-- **The record the auditor assembles** on the history `h` at `x`, the witness `w` and the tape `ρ`, at the level-0 source
`src` and the proof files `proofs`: `recordJson` at SHA-512 of each rep stream's rounds as they left the loop
(`seenRounds`, each round's bytes retained), the roots of the level 0 the loop drew (`level0Of`, as its `Commit` carries
them) and the link points (the `Commit`'s coins). -/
noncomputable def recordOf {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) (h : List (Msg × Msg)) (spec : SessionSpec) (sts : Array Stmt)
    (proofs : Array ByteArray) : V String :=
  FlockComplete.recordJson (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) spec (rootsOf src x w ρ sts)
    (pointsOf x h spec) ((Array.range spec.repStreams.size).map (streamOf x h spec sts)) proofs

/-- **The loop's `Finish`** on the history `h` at `x`, the witness `w` and the tape `ρ`: the record and the proof files
(`recordOf`, `proofsOf`), or why it stops. -/
noncomputable def filesE {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (draws : R → ℕ → ByteArray) (x : Msg) (w : Flock.Wit) (ρ : R) (h : List (Msg × Msg)) :
    V (String × Array ByteArray) :=
  match FlockDeploy.setupOf true x with
  | some (_, spec, sts) => do
    let proofs ← proofsOf src draws x w ρ h spec sts
    return (← recordOf src x w ρ h spec sts proofs, proofs)
  | none => .error "no session"

end Loop

/-- `L₀` with the session loop's own `Finish` at the level-0 source `src`: its files `Loop.filesE` at `L₀`'s frame draws
(the empty record and no proof files where it stops), and its stop where `Loop.filesE` throws. -/
noncomputable def LeanRelease.withFiles {R : Type} (L₀ : LeanRelease Flock.Wit R)
    (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)) : LeanRelease Flock.Wit R :=
  { L₀ with
    files := fun x w ρ h => (Loop.filesE src L₀.draws x w ρ h).toOption.getD ("", #[])
    finStops := fun x w ρ h => !(Loop.filesE src L₀.draws x w ρ h).isOk }


/-- **The release at the session loop's own frames, link and `Finish`**, at the deployed tables: `ofLoop` at
`withFiles (withLink L₀ src) src`. -/
noncomputable def LeanRelease.ofLoopLink
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) :
    LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) :=
  LeanRelease.ofLoop Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real

/-- **The level 0 the loop draws for rep stream `k`** at the source `src`: `Level0.ofDraws` at table `k / 2`'s `m` on the
packed witness and drawn bytes `src` gives rep stream `k` (`Loop.realOf`'s). -/
def Loop.level0At {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) (sts : Array Stmt) (k : ℕ) : V _root_.Flock.Firewall.Level0.Level0 := do
  let (g, d, _) ← src x w ρ k
  _root_.Flock.Firewall.Level0.ofDraws sts[k / 2]!.m g d

/-- **The link's level 0 is the tape's** at `x`: at every nonce and tape, table `j`'s drawn level 0 (`Loop.level0Of`) has
`Masked.linkIn`'s inputs. `Loop.Src`'s level-0 half gives it (`Loop.Src.linkLevel0`). -/
def Loop.LinkLevel0 (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (j : Fin (Masked.jOf x)),
    ∃ l, Loop.level0Of src x w (ν, t) (Masked.stsOf x) j.val = .ok l ∧ l.toIn = Masked.linkIn x w t j

/-- **The level-0 inputs are the tape's** at `x`: at every nonce and tape, table `j`'s rep `r`'s drawn level 0
(`Loop.level0At` at rep stream `2j + r`) has `Masked.linkIn`'s inputs: the witness's packed table at the tape's mask
words (what the host hands the loop) and the tape's extra lanes, lane paddings and level-0 salts (the bytes
`Level0.draw` reads from `IO.getRandomBytes`). It says nothing of how a root is committed: the root is `Link.root` of
those inputs by the code (`Level0.ofDraws_ok`). -/
def Loop.Level0Src (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (j : Fin (Masked.jOf x)) (r : Fin 2),
    ∃ l, Loop.level0At src x w (ν, t) (Masked.stsOf x) (2 * j.val + r.val) = .ok l ∧ l.toIn = Masked.linkIn x w t j

/-- **The rep draws are the tape's** at `x`: at every nonce and tape, table `j`'s rep `r`, rep stream `2j + r`, at the
tape's layout (`Masked.layE`):
* the frame's draws (`L₀.draws`, the bytes `Frames.startRep` reads from `IO.getRandomBytes`) are `drawBytes` long, and
  read as `Draws.ofBytes`, their pads are the tape's (`padsOf`) and their pads tree's leaf salts the tape's (each of the
  `codeLen` leaves, at the `192` bytes the masked prover fits it to);
* where `src` hands the rep its salts for levels ≥ 1 (`hi`, the bytes `Finish.serve`'s `hiSaltsOf` reads from
  `IO.getRandomBytes`), each leaf's salt at each such level of the table's schedule is the tape's. -/
def Loop.RepSrc
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (j : Fin (Masked.jOf x)) (r : Fin 2),
    (L₀.draws (ν, t) (2 * j.val + r.val)).size = _root_.Flock.Firewall.Frame.drawBytes (Masked.layE x j) ∧
    FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j) =
      drawsPads (_root_.Flock.Firewall.Frame.Draws.ofBytes (Masked.layE x j) (L₀.draws (ν, t) (2 * j.val + r.val))) ∧
    (∀ q < _root_.Flock.Firewall.Frame.codeLen (Masked.layE x j),
      FlockComplete.fitBytes 192 ((Masked.UE.tapeZk t).innerSalt j.val r.val q) =
        (_root_.Flock.Firewall.Frame.Draws.ofBytes (Masked.layE x j) (L₀.draws (ν, t) (2 * j.val + r.val))).salts.extract
          (_root_.Flock.Firewall.Frame.SALT * q) (_root_.Flock.Firewall.Frame.SALT * q + _root_.Flock.Firewall.Frame.SALT)) ∧
    ∀ (g : Array F128) (d : ByteArray) (hi : Array ByteArray),
      src x w (ν, t) (2 * j.val + r.val) = .ok (g, d, hi) →
      ∀ l, 1 ≤ l → l < (Masked.zkTab x j).sch.levels.size →
        ∀ p < 2 ^ ((Masked.zkTab x j).sch.levels.getD l default).d,
          _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty) p = (Masked.UE.tapeZk t).salt j.val r.val l p

/-- `flock-firewall-os-src` (a trust in the host's inputs and in the operating system's bytes, which the loop reads and
does not check). **Every input the session loop takes from its host and its operating system is the tape's** at `x`, in
two halves: the level-0 inputs (`Loop.Level0Src`: the packed witness the host hands it and the bytes `Level0.draw`
reads) and the rep draws (`Loop.RepSrc`: the bytes `Frames.startRep` and `Finish.serve` read for each rep). Why it is
reasonable: the loop runs on the auditor-trusted host (`trusted-host`), whose operating system's random bytes are
the uniform draws the tape models, and which hands the loop the witness `w` the statement is served at. -/
def Loop.Src
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) : Prop :=
  Loop.Level0Src src x w ∧ Loop.RepSrc L₀ src x w

/-- Obligation (the inner proof's round 0): where history `h` stands at a rep stream `k`'s first round past its
scheduled rounds `rsched` (the opening's, on `real`'s table and salts), `L₀`'s round there is the firewall's own at some
16-byte `final_c'`: the inner proof's label frame, then `final_c'` between its headers (Rust's firewall sends that round,
`check_self`). -/
def _root_.Flock.Guarantees.LoopInnerRound0 (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
    (h : List (Msg × Msg)) (vals : _) (s0 : _) (spec : SessionSpec) (sts : Array Stmt) (q k j : ℕ)
    (tr : _root_.Flock.Firewall.Finish.Tab) (hr : Array ByteArray) (rsched : Array (ByteArray × ℕ)),
    FlockDeploy.setupOf true x = some (s0, spec, sts) → h.length = q + 2 →
    CoinLayer.locate (Loop.roundsOf spec sts) 0 q = (k, j + 1) → real x w ρ k = .ok (tr, hr) →
    Loop.schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (Loop.wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) 0))) hr
      ((Array.range (j + 1)).map
        fun i => Loop.wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) (q - (j + 1) + i + 1)) = .ok rsched →
    rsched.size = j + 1 →
    ∃ w' : ByteArray, w'.size = 16 ∧ L₀.scheduled x w ρ h vals =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1).data.toList

/-- **C-Flock's zero knowledge at the session loop's own frames, at the tables `T`.** At the deployed relation, where
`T`'s session meets the tables' side (`Tables`, `Complete` at the relation, `DummyReg`, `Causal`, `SwCausal` at `vs`,
`TapeOK`), at every repetition count `0 < t`, every `L₀` (the release's inner rounds), every level-0 source `src` (the
rep's tables `real` are the loop's own, `Loop.realOf src`) and worker `wk`, at every statement `x` the loop serves
(`Loop.Serves`) with a witness `w` under the relation, where the loop's obligation (`LoopInnerRound0`) and the
transport's round trusts hold: the deployed frame's state
is the loop's but for `stopped` (`ProgChecks`), and against every `V*` of one more round than the verifier, the
probability of any event of the real view at `ofLoop` is at most twice the simulator's, plus
$$8\,n_L(x)\,\delta_1 + 8\,t\,n_L(x)\,\delta_1 + 2 \cdot 3^{-t}$$
with `n_L(x)` the session's leaf count and `δ₁` the hm96 salted-leaf hiding gap at the pinned key, plus `4·t·3^t` times
the fork probability. -/
def _root_.Flock.Guarantees.SessionLoopZKAt (T : CoinLayer.TableData) : Prop :=
  ∀ (vs : (CoinLayer.session Loop.loopNr T).SwVals) (t : ℕ), 0 < t →
    (CoinLayer.session Loop.loopNr T).Tables →
    (CoinLayer.session Loop.loopNr T).Complete
      (Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) →
    (CoinLayer.session Loop.loopNr T).DummyReg → (CoinLayer.session Loop.loopNr T).Causal →
    (CoinLayer.session Loop.loopNr T).SwCausal vs → (CoinLayer.session Loop.loopNr T).TapeOK →
    ∀ (L₀ : LeanRelease Flock.Wit
        ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
      (src : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
        ℕ → V (Array F128 × ByteArray × Array ByteArray))
      (real : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
        ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)),
    real = Loop.realOf src →
    ∀ (wk : (LeanRelease.ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit),
    Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w → Loop.Serves x →
    _root_.Flock.Guarantees.LoopInnerRound0 T L₀ real x w →
    LeanRelease.LinkRoundEq (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) x w →
    LeanRelease.FinishIs (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH wk x w →
    LeanRelease.LiveRoundEq (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH wk x w →
    LeanRelease.FillRoundEq (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH wk x w →
    ∀ {Θ : Type} (Vs : VStar Θ) (θ : Θ),
    Vs.k x = (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1 →
    ∀ (Ev : List ((Msg × Out) × ℕ) → Prop),
    (LeanRelease.ofLoop T L₀ real).ProgChecks x w ∧
    Pr (fun ω : ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand) ×
          (LeanRelease.ofLoop T L₀ real).toSrv.Chk =>
        Ev ((LeanRelease.ofLoop T L₀ real).toSrv.timed x
          (runStar CoinOpen.shaH ((LeanRelease.ofLoop T L₀ real).toSrv.proverD (LeanRelease.ofLoop T L₀ real).fillOf
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1) wk w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1)
            (LeanRelease.ofLoop T L₀ real).toSrv.stop).C x =>
          Ev ((LeanRelease.ofLoop T L₀ real).toSrv.timed x ((leaky1 ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1)
            (LeanRelease.ofLoop T L₀ real).toSrv.stop).out CoinOpen.shaH Vs θ x c))) +
          (8 * ((CoinLayer.session Loop.loopNr T).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T)
              (LeanRelease.ofLoop T L₀ real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoop T L₀ real).stop).nLeaves x *
              FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
            8 * t * ((CoinLayer.session Loop.loopNr T).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T)
              (LeanRelease.ofLoop T L₀ real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoop T L₀ real).stop).nLeaves x *
              FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t *
            (Fork CoinOpen.shaH ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).swP vs).prover w) Vs θ x).Pr
              CoinOpen.shaH)

/-- **C-Flock's zero knowledge at the session loop's own frames**, at every table data `T` that admits unrefused coins
(`TablesAdmitUnrefused`): `SessionLoopZKAt T`. -/
def _root_.Flock.Guarantees.SessionLoopZK : Prop :=
  ∀ T : CoinLayer.TableData, _root_.Flock.Guarantees.TablesAdmitUnrefused T → _root_.Flock.Guarantees.SessionLoopZKAt T

/-- **C-Flock's zero knowledge at the session loop's own frames, at the deployed tables** (`Masked.dataE`), at the
loop's own link and `Finish`: `SessionLoopZKAt Masked.dataE` at the release `withFiles (withLink L₀ src) src`, whose link
is the one the loop's `Commit` carries (`Loop.linkOf src`) and whose files and stop are its `Finish`'s (`Loop.filesE`),
with no premise on the tables beyond `SessionLoopZKAt`'s. In place of
`LinkRoundEq` it takes that `x` is in the masked verifier's scope (`Claims.Flock.Scope`, the statement check
`flock-verify verify --zk` runs, `ScopeMatches`) and that the loop's inputs from its host and operating system are the
tape's (`Loop.Src`). The scope
is a premise because the loop does not run that check, and cannot run all of it: the check reads the verifier's own
`--public-inputs` and `--registered`, which the loop never sees. At a statement outside it the model's session has no
tables, so its link carries no root while the loop's `Commit` carries each table's. -/
def _root_.Flock.Guarantees.SessionLoopZKDeployed : Prop :=
  ∀ (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (t : ℕ), 0 < t →
    (CoinLayer.session Loop.loopNr Masked.dataE).Tables →
    (CoinLayer.session Loop.loopNr Masked.dataE).Complete
      (Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) →
    (CoinLayer.session Loop.loopNr Masked.dataE).DummyReg → (CoinLayer.session Loop.loopNr Masked.dataE).Causal →
    (CoinLayer.session Loop.loopNr Masked.dataE).SwCausal vs → (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK →
    ∀ (L₀ : LeanRelease Flock.Wit
        ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
      (src : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → V (Array F128 × ByteArray × Array ByteArray))
      (real : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)),
    real = Loop.realOf src →
    ∀ (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit),
    Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w → Loop.Serves x →
    _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ real x w →
    Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x → Loop.Src L₀ src x w →
    LeanRelease.FinishIs (CoinLayer.session Loop.loopNr Masked.dataE) vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w →
    LeanRelease.LiveRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w →
    LeanRelease.FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w →
    ∀ {Θ : Type} (Vs : VStar Θ) (θ : Θ),
    Vs.k x = (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1 →
    ∀ (Ev : List ((Msg × Out) × ℕ) → Prop),
    (LeanRelease.ofLoopLink L₀ src real).ProgChecks x w ∧
    Pr (fun ω : ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
          (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk =>
        Ev ((LeanRelease.ofLoopLink L₀ src real).toSrv.timed x
          (runStar CoinOpen.shaH ((LeanRelease.ofLoopLink L₀ src real).toSrv.proverD (LeanRelease.ofLoopLink L₀ src real).fillOf
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1) wk w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1)
            (LeanRelease.ofLoopLink L₀ src real).toSrv.stop).C x =>
          Ev ((LeanRelease.ofLoopLink L₀ src real).toSrv.timed x ((leaky1 ((((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1)
            (LeanRelease.ofLoopLink L₀ src real).toSrv.stop).out CoinOpen.shaH Vs θ x c))) +
          (8 * ((CoinLayer.session Loop.loopNr Masked.dataE).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE)
              (LeanRelease.ofLoopLink L₀ src real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoopLink L₀ src real).stop).nLeaves x *
              FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
            8 * t * ((CoinLayer.session Loop.loopNr Masked.dataE).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE)
              (LeanRelease.ofLoopLink L₀ src real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoopLink L₀ src real).stop).nLeaves x *
              FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t *
            (Fork CoinOpen.shaH ((((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).swP vs).prover w) Vs θ x).Pr
              CoinOpen.shaH)

end Claims.ZKServer
