import Proofs.Flock.SessionLoop.Statement

/-!
# The session loop's own `Finish` files

`LeanRelease.files` and `finStops` are the record and proof files the firewall builds at the session's `Finish`, and
whether that `Finish` stops. Here they are the session loop's own, as functions of the statement, the witness, the tape
and the history, at a level-0 source `src` and the release `L₀` whose frame draws they read:

* **The proof files** (`Loop.proofsOf`): rep stream `k`'s is `Finish.proofFile` on its table and salts for levels ≥ 1
  (`Loop.realOf src`, the loop's own level 0) and on the rep's request (`Loop.repInOf`), which is what the loop holds of the
  rep (`Frames.ownOf`): the frame's draws (`L₀.draws`), the extra claims at the session's link points (`Drive.extras`), the
  coins the stream answered each of its rounds, and the values its frame took. Those values are read back off the
  rounds the frame released (`Loop.maskedOf`): each leaves masked, `v + h_i` under the rep's pad `h_i` (`Frame.Masker.absorb`),
  so at a pad of the absorbed order the value is the released word plus the pad (`Loop.valsOf`). `final_c` is never
  absorbed: it is the masked word of the inner proof's round 0 (`final_c'`, `LoopInnerRound0`) plus its pad, which is
  the value the loop's finish computes its file from where the `inner` request that computed it carried it
  (`Finish.serve`'s `liveOnly`: the file the loop serves is the one its `inner` request computed live).
* **The record** (`Loop.recordOf`): the session loop sends the auditor its proofs and then `Finish`, no record of its own,
  so the record is the one the auditor assembles from what it saw, `recordJson` of each rep stream's rounds as they left
  the loop (`Loop.seenRounds`: each round's bytes, coin count and the coins the stream answered), the link's roots (the
  level 0 the loop drew, `Loop.level0Of`) and points (the `Commit`'s coins), and the proof files. This is a modelling
  choice: the record the analysis's `Final.encode` reads is the auditor's own of the streams it saw.
* **The stop** (`Loop.finStopsOf`): the loop's `Finish` stops where any of this throws, a rep's proof file above all
  (`Finish.proofFile`: its `repFw`, whose constraints fail at a `final_c` the passed values do not fix).

`LeanRelease.withFiles` puts them in a release.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Frame (Cfg Masker Draws)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- Rep stream `k`'s first rep round, at the rep streams' round counts `rs`: the rounds of the streams before it. -/
def streamAt (rs : List ℕ) (k : ℕ) : ℕ := (rs.take k).sum

/-- **Rep stream `k`'s rounds as they left the loop** on the history `h` at `x`, at the rep streams' round counts `rs`: its
round `j`, rep round `streamAt rs k + j`, the history's entry past `Hello` and the link that carries it (`progOf`'s
placing), as its bytes, its coin count (`loopNr`) and the words of the coins the stream answered it (`wordsAt`). -/
noncomputable def seenRounds (x : Msg) (h : List (Msg × Msg)) (rs : List ℕ) (k : ℕ) :
    List (ByteArray × ℕ × Array F128) :=
  let cs := h.tail.map fun p => CoinLayer.rd p.2
  (List.range (rs.getD k 0)).map fun j =>
    let q := streamAt rs k + j + 1
    (⟨((h.tail.getD q ([], [])).1).toArray⟩, loopNr x q, wordsAt x cs q)

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
  let d := Draws.ofBytes lay draws
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

/-- **The record the auditor assembles** on the history `h` at `x`, the witness `w` and the tape `ρ`, at the level-0 source
`src` and the proof files `proofs`: `recordJson` at SHA-512 of each rep stream's rounds as they left the loop
(`seenRounds`, each round's bytes retained), the roots of the level 0 the loop drew (`level0Of`, as its `Commit` carries
them) and the link points (the `Commit`'s coins). -/
noncomputable def recordOf {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) (h : List (Msg × Msg)) (spec : SessionSpec) (sts : Array Stmt)
    (proofs : Array ByteArray) : V String :=
  let c1 := wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) 0
  let points := (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j => c1.getD (p * spec.mPts + j) 0
  let rootB := (Array.range sts.size).map fun j => ((level0Of src x w ρ sts j).toOption.map (·.root)).getD .empty
  let streams := (Array.range spec.repStreams.size).map fun k =>
    ((seenRounds x h (roundsOf spec sts) k).map fun (b, n, cs) =>
      ({ n, sha := .empty, coins := cs, bytes := some b } : _root_.Flock.Transcript.Round)).toArray
  FlockComplete.recordJson (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) spec rootB points streams proofs

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

namespace LeanRelease

/-- `L₀` with the session loop's own `Finish` at the level-0 source `src`: its files `Loop.filesE` at `L₀`'s frame draws
(the empty record and no proof files where it stops), and its stop where `Loop.filesE` throws. -/
noncomputable def withFiles {R : Type} (L₀ : LeanRelease Flock.Wit R)
    (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)) : LeanRelease Flock.Wit R :=
  { L₀ with
    files := fun x w ρ h => (Loop.filesE src L₀.draws x w ρ h).toOption.getD ("", #[])
    finStops := fun x w ρ h => !(Loop.filesE src L₀.draws x w ρ h).isOk }

end LeanRelease

end Claims.ZKServer
