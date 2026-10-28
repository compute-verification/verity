import CoreProtocol.Live
import SampledProofs.ChallengerDraw
import SampledProofs.OneStage

/-!
# Coins from the device's log: what a draw's coins are, stated against the code that reads them

A component whose draw reads live coins (`Core.Live`) decides at the coins it reads from the challenger's signed record in its evidence. These properties say
what that record shows, for every input: the coins the challenger device drew, or a break the run carries.

* `LiveAnswer`: what `Protocol.answer` accepts, `Protocol.answerAt` accepts at the coins its evidence's record shows.
  It follows from `Live.factors` and reads no device.
* `CoinsFromLog`, the target: what a component that reads its coins through that reader (`ReadsChallenger`)
  accepts, it accepts at coins that are what the device released, or a `Break`; one-stage reads so (`oneStageReads`).
* `DrawnByDevice`: what `SampledProofs.Challenger.drawCoins` reads, under the verifier's key, window and family, is the
  concatenation of the coins of that family that a session of the device released after `after`, a session the device's
  own ledger opens withheld on the window (`OpensOn`), or the fixed finder outputs a break from the run (`Break`).
  `drawnByDevice_nontrivial` (security_proofs) shows it false for a reader that accepts every record.

**The device** (`Device`) is the proof service's challenger as `challenger.py` runs it. Under its key it runs sessions
(`Session`): it draws each coin and its salt by position, commits to the frame-v3 root over their live leaves
(`Session.commitment`), writes its release log (its head the commitment, then each root it received and the coin it
released after it), and signs its record before any coin opens and its close over the log once the session ends. A
session its ledger opens withheld (`OpensOn`, `remote.open_session(..., withhold=True)`) releases, while it runs, each
coin's raw bytes and nothing else, and its commitment, salts and paths only once its close is signed, after the prover's
commitments are in: nothing the prover sees before a coin depends on a coin not yet released, so no hiding premise on
the root enters soundness. It also signs the auditor's ledger, line by line. `Signed` is all it signs under a key.
One role per key: a key also used for another role is outside the model, and a `forgery` witness is then an EUF-CMA
break only under that one-role reading.

**The coins are raw bits.** A coin is `Challenger.COIN_BYTES` bytes the device drew, and a draw reads the concatenation
of the coins it was released (`take`, in `SampledProofs.Draw.drawUnits`), with no hash between: no SHA-512-expanded
stream is read as uniform. In the causal game each coin is one release of `8 * COIN_BYTES` bits (`bitsOf`), after the
bytes it answers (for one-stage, the registration's digest), at most `MAX_COINS` of them (`oneStageRounds`); a withheld
session's releases are those bits alone.

**The break** (`Break`) is one adversary form's: it is no event of the run but what one fixed finder
outputs. `findBreak` is a computable function of the run (the record the evidence carries and the device's log), the
same for every prover; its SHA-512 calls are one per string in `readHashed` and `deviceHashed` and an LMS verification
per signature `drawCoins` checks, so at most a constant times the reader's and the device's own. It outputs a `Witness`,
and `Break` is that it outputs a valid one (`Witness.Valid`):
* `forgery`: an LMS signature among those `drawCoins` checks that verifies under the verifier's key over a message the
  device never signed under it (`Signed`), the LMS term (`euf-cma/lms-sha512-n64` asked of the finder);
* `collision`: two different strings, each hashed in reading the coins (`readHashed`) or by the device
  (`deviceHashed`), with one SHA-512 digest (`ecr/sha-512` asked of the finder).

Nothing is assumed of SHA-512 or LMS here. What the device releases when, and that it opens each window once, are the
device's behaviour on `trusted-host`, which these properties don't take: a soundness argument reads them off the device
model.
-/

namespace Properties.Challenger

open Lean (Json)
open SampledProofs SampledProofs.Challenger

/-! ## What `answer` accepts, at the coins -/

/-- **What `answer` accepts, `answerAt` accepts at the coins its evidence's record shows.** For every protocol whose
check reads its coins (`Live`), an accepted answer's three byte strings decode, its coins read, and its verdict at those
coins accepts. -/
def LiveAnswer : Prop :=
  ∀ (P : Core.Protocol) (L : Core.Live P) (a b e : ByteArray), (P.answer a b e).verdict = .accept →
    ∃ p c ev ξ, (Core.canonicalJson a >>= P.params) = .ok p ∧ P.commitment b = .ok c ∧ P.evidence e = .ok ev ∧
      L.coins p c ev = .ok ξ ∧ (P.answerAt L a b ξ e).verdict = .accept

/-! ## The challenger, as it runs -/

/-- One session the challenger ran under its key: its id and host, its layout, its coins and their salts by position,
the lines of the release log it wrote, and how it ended (`none` when its client closed it). -/
structure Session where
  id : String
  host : String
  scheme : String
  epoch : Nat
  families : Families
  coins : Nat → ByteArray
  salts : Nat → ByteArray
  log : List String
  ended : Option String

/-- The session's tree's leaves under its domain `dom`: position `p`'s frame-v3 leaf over its coin's live leaf. -/
def Session.leaves (s : Session) (dom : ByteArray) : Array ByteArray :=
  (Array.range (total s.families)).map fun p =>
    Flock.FV3.leaf dom p s.scheme (liveLeaf s.scheme (s.coins p) (s.salts p)) Flock.Sha512.hash

/-- The coin commitment the session publishes before any coin opens: its layout and the frame-v3 root over its leaves
under its live domain; `none` when its layout has none. -/
def Session.commitment (s : Session) : Option LiveCommitment :=
  match liveDomain s.scheme s.epoch s.families with
  | .ok dom => some ⟨s.scheme, s.epoch, s.families, Flock.FV3.root dom (s.leaves dom) Flock.Sha512.hash⟩
  | .error _ => none

/-- The coins of family `label` the session released after it received `after`, concatenated in order: coin `i` of the
family for each `i` whose root its log shows received is `after`. -/
def Session.released (s : Session) (label : String) (after : ByteArray) : ByteArray :=
  match s.commitment with
  | none => .empty
  | some c =>
    match releasedRootsOf c s.log with
    | .error _ => .empty
    | .ok roots =>
      let rs := ((roots.find? (·.1 == label)).map (·.2)).getD []
      (rs.zipIdx.filter (·.1 == after)).foldl (fun acc x => match c.position label x.2 with
        | some p => acc ++ s.coins p
        | none => acc) .empty

/-- The challenger devices: under each key, the sessions its holder ran and the auditor's ledger lines it signed, in
order. -/
structure Device where
  sessions : ByteArray → List Session
  ledger : ByteArray → List Json

/-- The key's LMS identifier, its public key's first 16 bytes. -/
def ident (pk : ByteArray) : ByteArray := pk.extract 0 16

/-- Everything the device signs under the key `pk`, in order: each session's record and close messages, and each of
its ledger lines' messages. -/
def signedBy (dev : Device) (pk : ByteArray) : List ByteArray :=
  ((dev.sessions pk).flatMap fun s => match s.commitment with
    | none => []
    | some c => (recordMessage (ident pk) s.host c).toOption.toList ++
        (closeMessage (ident pk) s.host c s.id s.log.length (logChain s.log) s.ended).toOption.toList) ++
  (dev.ledger pk).filterMap fun line => do
    let e ← ledgerEntry? line
    (ledgerMessage (ident pk) e.entry).toOption

/-- The device signed `m` under the key `pk` (`signedBy`). -/
def Signed (dev : Device) (pk m : ByteArray) : Prop := m ∈ signedBy dev pk

/-- The device's own ledger under `pk` opens window `w` (its open root) for the session `id`, under the ledger's v2
schema, as a withheld session (`withheld: true`). -/
def OpensOn (dev : Device) (pk : ByteArray) (w id : String) : Prop :=
  ∃ line ∈ dev.ledger pk, ∃ e, ledgerEntry? line = some e ∧ e.kind = "open" ∧ e.schema = LEDGER_SCHEMA ∧
    e.session = id ∧ getD (getD e.body "window") "root" = .str w ∧ getD e.body "withheld" = .bool true

/-! ## What the reader checks and hashes -/

/-- Whether the LMS verifier accepts `sig`, once decoded, under the key over the message `m`. -/
def lmsVerifies (key : ChallengerKey) (m sig : ByteArray) : Bool :=
  match Lms.Signature.decode key.params sig with
  | .ok σ => match Lms.verify key.params key.pub m σ with
    | .ok b => b
    | .error _ => false
  | .error _ => false

/-- The LMS verifier accepts `sig`, once decoded, under the key over the message `m`. -/
def LmsVerifies (key : ChallengerKey) (m sig : ByteArray) : Prop := lmsVerifies key m sig = true

/-- The signatures `drawCoins` checks under `key`, each with the message it signs: the session's record and close, and
each line of the ledger. -/
def signatures (key : ChallengerKey) (coins : Json) : List (ByteArray × ByteArray) :=
  let I := ident key.pub
  let session := getD coins "session"
  let ofRecord : List (ByteArray × ByteArray) :=
    match ChallengerRecord.ofJson (getD session "record") with
    | .error _ => []
    | .ok r =>
      (match recordMessage I r.host r.commitment with
        | .ok m => [(m, r.signature)]
        | .error _ => []) ++
      (match ChallengerClose.ofJson (getD session "close") with
        | .error _ => []
        | .ok c => match closeMessage I r.host r.commitment c.session c.lines c.chain c.ended with
          | .ok m => [(m, c.signature)]
          | .error _ => [])
  let ofLedger : List (ByteArray × ByteArray) :=
    match getD coins "ledger" with
    | .arr ls => ls.toList.filterMap fun l => do
      let e ← ledgerEntry? l
      let m ← (ledgerMessage I e.entry).toOption
      pure (m, e.sig)
    | _ => []
  ofRecord ++ ofLedger

/-- What `Flock.FV3.hash tag parts` hashes. -/
def fv3Message (tag : String) (parts : List ByteArray) : ByteArray :=
  let t := tag.toUTF8
  parts.foldl (fun acc p => acc ++ Flock.be64 p.size.toUInt64 ++ p) (Flock.FV3.FRAME ++ Flock.be32 t.size.toUInt32 ++ t)

/-- What `taggedSha512 tag manifest` hashes, when the manifest has a canonical form. -/
def taggedMessage (tag : String) (manifest : Json) : List ByteArray :=
  match Flock.canon manifest with
  | .ok s =>
    let t := tag.toUTF8
    let b := s.toUTF8
    ["verity/tagged-sha512/v1".toUTF8.push 0 ++ Flock.be32 t.size.toUInt32 ++ t ++ Flock.be64 b.size.toUInt64 ++ b]
  | .error _ => []

/-- What SHA-512 hashes in `liveDomain scheme epoch families`: the binding, the positions and the domain's frame. -/
def domainHashed (scheme : String) (epoch : Nat) (fs : Families) : List ByteArray :=
  let binding := Json.mkObj [("epoch", Lean.toJson epoch), ("families", familiesJson fs), ("scheme", Json.str scheme)]
  let positions := Json.mkObj [("start", Lean.toJson (0 : Nat)), ("step", Lean.toJson (1 : Nat)),
    ("stop", Lean.toJson (total fs))]
  taggedMessage BINDING binding ++ taggedMessage "veritor/indexed-domain/range/v1" positions ++
    match taggedSha512 BINDING binding, taggedSha512 "veritor/indexed-domain/range/v1" positions with
    | .ok b, .ok p => [fv3Message "domain" [b, Flock.FV3.uint 1, p, Flock.FV3.uint (total fs)]]
    | _, _ => []

/-- What SHA-512 hashes in a coin's live leaf and its frame-v3 leaf at position `pos`: the coin's inner digest's input,
hm96's salt-digest and leaf inputs, and the leaf's frame. -/
def leafHashed (scheme : String) (dom : ByteArray) (pos : Nat) (coin salt : ByteArray) : List ByteArray :=
  let inner := scheme.toUTF8.push 0 ++ coin
  [inner, Flock.Hm96.Default512.saltPrefix ++ salt,
    Flock.Hm96.Default512.leafPrefix ++ Flock.Hm96.Default512.commitString (Flock.Sha512.hash inner) salt,
    fv3Message "leaf" [dom, Flock.FV3.uint pos, Flock.FV3.uint pos, scheme.toUTF8, liveLeaf scheme coin salt]]

/-- The node frames `Flock.Registered.climbL dom cur idx d sibs` hashes, bottom first. -/
def climbFrames (dom : ByteArray) : ByteArray → Nat → Nat → List ByteArray → List ByteArray
  | _, _, _, [] => []
  | cur, idx, d, sib :: rest =>
    let m := if idx % 2 = 0 then fv3Message "node" [dom, Flock.FV3.uint d, Flock.FV3.uint (idx / 2), cur, sib]
      else fv3Message "node" [dom, Flock.FV3.uint d, Flock.FV3.uint (idx / 2), sib, cur]
    m :: climbFrames dom (Flock.Sha512.hash m) (idx / 2) (d + 1) rest

/-- What SHA-512 hashes in checking opening `o` as coin `index` of family `label` of `c`: `c`'s domain, the coin's
leaf, and its climb to the root. -/
def openingHashed (c : LiveCommitment) (o : LiveOpening) (label : String) (index : Int) : List ByteArray :=
  domainHashed c.scheme c.epoch c.families ++
    match liveDomain c.scheme c.epoch c.families, c.position label index with
    | .ok dom, some pos =>
      leafHashed c.scheme dom pos o.coin o.salt ++
        climbFrames dom (Flock.FV3.leaf dom pos c.scheme (liveLeaf c.scheme o.coin o.salt) Flock.Sha512.hash) pos 0 o.path
    | _, _ => []

/-- What SHA-512 hashes in `recordMessage I host c`: the commitment's tagged digest's input, and the message's. -/
def recordHashed (I : ByteArray) (host : String) (c : LiveCommitment) : List ByteArray :=
  taggedMessage COMMITMENT_TAG c.toJson ++
    match taggedSha512 COMMITMENT_TAG c.toJson with
    | .ok d => [field RECORD_ROLE.toUTF8 ++ I ++ field (field host.toUTF8) ++ field d]
    | .error _ => []

/-- What SHA-512 hashes in `closeMessage I host c session lines chain ended`: the record message's, and the close's. -/
def closeHashed (I : ByteArray) (host : String) (c : LiveCommitment) (session : String) (lines : Nat)
    (chain : ByteArray) (ended : Option String) : List ByteArray :=
  recordHashed I host c ++
    match recordMessage I host c with
    | .ok r =>
      let fin := match ended with
        | none => ByteArray.mk #[0]
        | some e => ByteArray.mk #[1] ++ e.toUTF8
      [field CLOSE_ROLE.toUTF8 ++ I ++ field r ++ field session.toUTF8 ++ field (Flock.be64 lines.toUInt64) ++
        field chain ++ field fin]
    | .error _ => []

/-- What SHA-512 hashes in `logChain lines`, and in `logDigest lines`. -/
def logHashed (lines : List String) : List ByteArray :=
  let start := field LOG_TAG.toUTF8
  let steps := (lines.foldl (fun (acc : ByteArray × List ByteArray) l =>
      let x := acc.1 ++ field l.toUTF8
      (Flock.Sha512.hash x, acc.2 ++ [x])) (Flock.Sha512.hash start, [])).2
  [start] ++ steps ++ [(String.join (lines.map (· ++ "\n"))).toUTF8]

/-- What SHA-512 hashes in `ledgerMessage I entry`: the entry's tagged digest's input, and the message's. -/
def ledgerHashed (I : ByteArray) (line : Json) : List ByteArray :=
  match ledgerEntry? line with
  | none => []
  | some e =>
    let body := Json.mkObj ((entries e.entry).toList.filter (·.1 != "sig"))
    match ledgerTag? e.schema with
    | none => []
    | some tag =>
      taggedMessage tag body ++
        match taggedSha512 tag body with
        | .ok d => [field LEDGER_ROLE.toUTF8 ++ I ++ field d]
        | .error _ => []

/-- Everything SHA-512 hashes as `drawCoins key _ label after coins` reads `coins`: the session's record, close and
release log, each ledger line, the host's pins log and the verifier's copy of main's history as files, and each of the
segment's openings, checked as its family's coins from `start`. -/
def readHashed (key : ChallengerKey) (label : String) (coins : Json) : List ByteArray :=
  let I := ident key.pub
  let session := getD coins "session"
  let lines (j : Json) : List String := match j with
    | .arr ls => ls.toList.filterMap fun | .str s => some s | _ => none
    | _ => []
  let ofSession := match ChallengerRecord.ofJson (getD session "record") with
    | .error _ => []
    | .ok r =>
      recordHashed I r.host r.commitment ++
        (match ChallengerClose.ofJson (getD session "close") with
          | .ok c => closeHashed I r.host r.commitment c.session c.lines c.chain c.ended
          | .error _ => []) ++
        logHashed (lines (getD session "log")) ++ logHashed (lines (getD session "hellos")) ++
        let segment := getD coins "segment"
        match nat? (getD segment "start"), getD segment "openings" with
        | some start, .arr os => (os.toList.zipIdx.filterMap fun (o, i) =>
            (LiveOpening.ofJson o).toOption.map fun o => openingHashed r.commitment o label ((start + i : Nat) : Int)).flatten
        | _, _ => []
  let ofLedger := match getD coins "ledger" with
    | .arr ls => ls.toList.flatMap (ledgerHashed I)
    | _ => []
  let ofPins := match getD coins "pins" with
    | .arr _ => logHashed (lines (getD coins "pins"))
    | _ => []
  ofSession ++ ofLedger ++ ofPins

/-- Everything SHA-512 hashes as the device runs under `pk`: for each session its domain, its tree (every leaf and every
node of each level below the root), its record, close and release log, and each of its ledger lines. -/
def deviceHashed (dev : Device) (pk : ByteArray) : List ByteArray :=
  let I := ident pk
  ((dev.sessions pk).flatMap fun s =>
    domainHashed s.scheme s.epoch s.families ++
      match liveDomain s.scheme s.epoch s.families, s.commitment with
      | .ok dom, some c =>
        let n := total s.families
        let width := Flock.nextPow2 n
        let pads := (List.range (width - n)).map fun k => fv3Message "pad" [dom, Flock.FV3.uint (n + k)]
        let level0 := (s.leaves dom).toList ++ pads.map Flock.Sha512.hash
        let nodes := ((List.range (Flock.log2ceil width)).foldl (fun (acc : List ByteArray × List ByteArray) depth =>
            let frames := (List.range (acc.1.length / 2)).map fun i =>
              fv3Message "node" [dom, Flock.FV3.uint depth, Flock.FV3.uint i, acc.1[2 * i]!, acc.1[2 * i + 1]!]
            (frames.map Flock.Sha512.hash, acc.2 ++ frames)) (level0, [])).2
        ((List.range n).flatMap fun p => leafHashed s.scheme dom p (s.coins p) (s.salts p)) ++ pads ++ nodes ++
          closeHashed I s.host c s.id s.log.length (logChain s.log) s.ended ++ logHashed s.log
      | _, _ => []) ++
  (dev.ledger pk).flatMap (ledgerHashed I)

/-! ## The finder -/

/-- What the finder outputs: two strings it hashed, or a signature it checked with the message it signs. -/
inductive Witness where
  | collision (x y : ByteArray)
  | forgery (m sig : ByteArray)

/-- **A witness is a break**: two different strings with one SHA-512 digest, or an LMS signature that verifies under the
key over a message the device never signed under it. -/
def Witness.Valid (dev : Device) (key : ChallengerKey) : Witness → Prop
  | .collision x y => x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y
  | .forgery m sig => LmsVerifies key m sig ∧ ¬ Signed dev key.pub m

/-- The first string of `xs` whose SHA-512 digest another, different string of `xs` has, with that string: each string
hashed once. -/
def collisionIn (xs : List ByteArray) : Option (ByteArray × ByteArray) :=
  let hs := xs.map fun x => (x, Flock.Sha512.hash x)
  hs.findSome? fun (x, hx) => (hs.find? fun (y, hy) => hy == hx && !(y == x)).map fun (y, _) => (x, y)

/-- **The finder**, one fixed computable function of the run for every prover: the first signature `drawCoins` checks
under `key` in `coins` that verifies over a message the device didn't sign, else the first collision among the strings
the reader hashes in `coins` and the device hashes under `key`. -/
def findBreak (dev : Device) (key : ChallengerKey) (label : String) (coins : Json) : Option Witness :=
  match (signatures key coins).find? fun x => lmsVerifies key x.1 x.2 && !((signedBy dev key.pub).contains x.1) with
  | some (m, sig) => some (.forgery m sig)
  | none => (collisionIn (readHashed key label coins ++ deviceHashed dev key.pub)).map fun (x, y) => .collision x y

/-- **The break**: the finder, run on the record `coins` and the device, outputs a valid witness. -/
def Break (dev : Device) (key : ChallengerKey) (label : String) (coins : Json) : Prop :=
  ∃ w, findBreak dev key label coins = some w ∧ w.Valid dev key

/-! ## The coins, as raw bits -/

/-- A byte string's bits in order, bit `t` being bit `t mod 8` of byte `⌊t/8⌋`: a coin's raw bits, as the device drew them
and a draw reads them. -/
def bitsOf (b : ByteArray) : List Bool :=
  b.toList.flatMap fun x => (List.range 8).map fun i => x.toNat.testBit i

/-- The bits of the coins of family `label` the session released after `after`, in order. -/
def Session.releasedBits (s : Session) (label : String) (after : ByteArray) : List Bool :=
  bitsOf (s.released label after)

/-- **The device's rounds in the causal game**: one round per coin it may release, each `8 * COIN_BYTES` raw bits
released after the bytes the round answers, at most `MAX_COINS` of them. -/
def oneStageRounds : List Nat := List.replicate MAX_COINS (8 * COIN_BYTES)

/-! ## The property -/

/-- `DrawnByDevice` for the reader `read` in `drawCoins`' place. -/
def DrawnByDeviceAt (read : ChallengerKey → String → String → ByteArray → Json → Flock.V ByteArray) : Prop :=
  ∀ (dev : Device) (key : ChallengerKey) (w label : String) (after : ByteArray) (coins : Json) (ξ : ByteArray),
    read key w label after coins = .ok ξ →
      (∃ s ∈ dev.sessions key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after) ∨
        Break dev key label coins

/-- **The coins a draw reads are the ones its challenger drew.** For every challenger device, if `drawCoins` reads `ξ`
from `coins` under the verifier's key `key`, window `w` (its open root, lowercase hex) and family `label`, released
after `after`, then `ξ` is the coins of `label` that a session of the device under `key` released after `after`, a
session the device's own ledger opens on `w`; or the fixed finder outputs a break from the run (`Break`). -/
def DrawnByDevice : Prop := DrawnByDeviceAt drawCoins

/-! ## The target: a component's coins, from the device's log -/

/-- What a draw's coins are read under, all of it the verifier's: the challenger's key, the window (its open root,
lowercase hex), the family, and what the coins were released after. -/
structure Draws where
  key : ChallengerKey
  window : String
  label : String
  after : ByteArray

/-- `L` reads its coins through the challenger's reader wherever the parameters and commitment name a challenger
(`source`, read from them alone, never from the evidence): the coins it reads are what `drawCoins` reads from the record
its evidence carries (`record`), decoded (`decode`). -/
structure ReadsChallenger (P : Core.Protocol) (L : Core.Live P) where
  source : P.Params → P.Commitment → Option Draws
  record : P.Evidence → Json
  decode : ByteArray → L.Coins
  reads : ∀ p c d, source p c = some d → ∀ e ξ, L.coins p c e = .ok ξ →
    ∃ raw, drawCoins d.key d.window d.label d.after (record e) = .ok raw ∧ decode raw = ξ

/-- **What a component accepts, it accepts at the coins its challenger device drew, or the run carries a
break.** For every protocol whose check reads its coins through the challenger's reader (`ReadsChallenger`) and every
challenger device: if `answer` accepts `a`, `b`, `e`, they decode to `p`, `c`, `ev`, and `answerAt` accepts them at
the coins `ξ` the evidence's record shows; and where `p` and `c` name a challenger (`source`), `ξ` is the decoding of
what a session of the device under its key released of the family after what they name, a session the device's own
ledger opens on their window, or the fixed finder outputs a break from the record in `ev` and the device (`Break`). -/
def CoinsFromLog : Prop :=
  ∀ (P : Core.Protocol) (L : Core.Live P) (R : ReadsChallenger P L) (dev : Device) (a b e : ByteArray),
    (P.answer a b e).verdict = .accept →
      ∃ p c ev ξ, (Core.canonicalJson a >>= P.params) = .ok p ∧ P.commitment b = .ok c ∧ P.evidence e = .ok ev ∧
        (P.answerAt L a b ξ e).verdict = .accept ∧
        ∀ d, R.source p c = some d →
          (∃ s ∈ dev.sessions d.key.pub, OpensOn dev d.key.pub d.window s.id ∧
            R.decode (s.released d.label d.after) = ξ) ∨
          Break dev d.key d.label (R.record ev)

/-! ## One-stage reads its coins through the challenger -/

/-- What one-stage's coins are read under (`OneStage.coinsOf`): with the parameters' challenger and window, its key, the
window, its family, and the registration's digest. -/
def oneStageSource (p : SampledProofs.OneStage.Params) (c : SampledProofs.OneStage.Commitment) : Option Draws :=
  match p.challenger, p.verifier.windowRoot with
  | some (key, label), some w => some ⟨key, w, label, c.digest⟩
  | _, _ => none

theorem oneStage_reads (p : SampledProofs.OneStage.Params) (c : SampledProofs.OneStage.Commitment) (d : Draws)
    (h : oneStageSource p c = some d) (e : SampledProofs.OneStage.Evidence) (ξ : Option ByteArray)
    (hξ : SampledProofs.OneStage.coinsOf p c e = .ok ξ) :
    ∃ raw, drawCoins d.key d.window d.label d.after e.coins = .ok raw ∧ some raw = ξ := by
  unfold oneStageSource at h
  unfold SampledProofs.OneStage.coinsOf at hξ
  rcases hch : p.challenger with _ | ⟨key, label⟩ <;> rcases hw : p.verifier.windowRoot with _ | w <;>
    simp only [hch, hw] at h hξ <;> try cases h
  cases hr : drawCoins key w label c.digest e.coins with
  | error r => simp [hr, Except.mapError, Functor.map, Except.map] at hξ
  | ok raw =>
    simp [hr, Except.mapError, Functor.map, Except.map] at hξ
    exact ⟨raw, rfl, by first | exact hξ | exact hξ.symm⟩

/-- **One-stage reads its coins through the challenger's reader** (`OneStage.live`), wherever its parameters name a
challenger and a window. -/
def oneStageReads : ReadsChallenger SampledProofs.OneStage.protocol SampledProofs.OneStage.live where
  source := oneStageSource
  record := SampledProofs.OneStage.Evidence.coins
  decode := fun b => (some b : Option ByteArray)
  reads := oneStage_reads

end Properties.Challenger
