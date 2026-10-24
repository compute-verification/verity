import PouwCheck

/-!
# PoUW's Pearl-C live salt: what the check of a pass salted by a live challenger session must establish

A Pearl-C device run (`PearlCDeviceRun`, manifest `MANIFEST_SCHEMA`) can take its salt coin from a live session of the
auditor's challenger instead of from seeded epoch coins (`pouw_pearl_c_host.epoch_of`). The session is reserved with
Pearl-C's layout, its salt coin alone (`SALT_FAMILIES`), and opened on the pass's weights' root `W`
(`weightsRootAt 64 "verity/pouw/weights/v1"` of the manifest's weights). H1's sessions are opened on a window-open root
instead. The session releases its one coin after `W`. `Check` is the signature of the check `core_check` gains for such a
pass. It applies `saltCheck`'s rule with `W` in place of the window-open root and returns the pass's salt. Its port will be
`PouwCheck.liveSaltCheck` (`core_check.live_salt`, the live branch of `_epoch_salt`). This file states its properties
before the check exists, so its proof shows `AcceptMeans PouwCheck.liveSaltCheck` and `OnePair PouwCheck.liveSaltCheck`.

What the check accepts (`Accepted`) is a pass of `MANIFEST_SCHEMA` that names its session and meets these conditions:

* The check reads the pass's coins only through the challenger's own records, never the manifest's word. The session is
  the one that the auditor's signed v2 ledger opens on `{root: W}` under the window's pinned challenger key and ends
  closed, run from a commit the host's pins log records on main's first-parent line (`pinnedSession`).
* The session's files are that session's (`sessionNamed`), under the commitment the session signed.
* The session counts (`Counts`): the commitment is Pearl-C's salt-only layout, and the session's log shows `W`, alone, as
  the one root it received, each received root followed by its coin's release.
* The ledger shows the session in its signed order (`LedgerShows`):
  - a reserve of the salt-only layout;
  - the open on `W` that binds that reserve;
  - the end, closed and complete, having released one salt coin and nothing else, its record's leaf strictly between
    the open's and the end's;
  - no other open on `W` or of the session.
* The manifest's coins are the session's commitment, and its salt opening is the commitment's `pouw/salt` coin 0, opened
  after `W` (`liveReason`).
* The manifest's salt is `audit.salt_of` of that coin over `W` (`liveSalt`).

The session counts exactly when its end is closed and complete, as the gate already computes it (`complete` is every
family of the reserved layout fully released) and as `PHC.held` and the period's close already read it. Under the
salt-only layout, complete means that the salt coin was released. A session reserved with H1's layout (`liveFamilies 1`)
that releases only its salt ends incomplete, and the check refuses it.

The three facts PoUW's argument reads from an acceptance:

* **The salt is fixed only after `W`.** The open naming `W` is signed before the session's record, which holds the
  challenger's commitment to the coins (`leafOf o.sig < recordLeaf`). The coin is opened after `W` (`liveReason`), and
  `W` is the root the log shows the coin's release followed.
* **The salt belongs to one (`W`, session) pair.** The ledger opens `W` once and the session once (`LedgerShows`'s last
  clause). The session is a function of the ledger and `W` (`OnePair`). Its commitment opens to one salt coin, up to a
  SHA-512 collision among the messages hashed (`LiveCoinBinding`). The salt is a function of that commitment, that coin
  and `W`.
* **No work before the salt is credited.** The session released one salt coin, once, and ended closed and complete. A
  session abandoned after its coin ends incomplete and doesn't count, so the prover can't grind sessions for a salt. The
  manifest's salt is that coin's (`liveSalt`), and PoUW's existing pass check binds every unit and record of the pass to
  it.

That the coin stays hidden until its release (the challenger's secret, hm96's hiding) and that the auditor's devices keep
their order (`trusted-nebius-host`) are the protocol's named assumptions, not this check's properties.
-/

namespace Properties.Pouw.LiveSalt

open Lean (Json)
open PouwCheck (R at? checkWorkload workloadOf weightsRootAt MANIFEST_SCHEMA)
open SampledProofs.Challenger (Audited ChallengerKey LedgerEntry LiveCommitment LiveOpening Roots ledgerEntry? leafOf
  liveReason pinnedSession sessionNamed)

/-! ## The layout, the weights' root and the salt -/

/-- Pearl-C's live layout: the salt coin and nothing else. A session reserved with it is complete exactly when it
released its salt coin. -/
def SALT_FAMILIES : List (String × Nat) := [(PouwWindow.SALT, 1)]

/-- `salt_epoch_reason`'s rule: `c` is a Pearl-C salt session's coin commitment, a live epoch's scheme over the salt-only
layout. -/
def SaltEpoch (c : LiveCommitment) : Prop := c.scheme = PouwWindow.LIVE_SCHEME ∧ c.families = SALT_FAMILIES

/-- `w` is the weights' root of the pass's manifest `m`: `audit.weights_root` of the roots of a workload whose every weight
is committed, and only those (`_check_workload`). -/
def WeightsRoot (m : Json) (w : ByteArray) : Prop :=
  ∃ us roots, workloadOf m = .ok (us, roots) ∧ checkWorkload us roots = .ok () ∧
    w = weightsRootAt 64 "verity/pouw/weights/v1" roots

/-- The window identity that a Pearl-C salt session is opened on: `{root}`, the weights' root in lowercase hex. -/
def weightsWindow (w : ByteArray) : Json := Json.mkObj [("root", .str (Flock.hexOf w))]

/-- `audit.salt_of` over a live commitment: the pass's 32-byte salt from its opened salt coin and the weights' root. -/
def liveSalt (c : LiveCommitment) (coin w : ByteArray) : Except String ByteArray := do
  let k ← PouwCommit.key? coin PouwCommit.EPOCH_SALT_DOMAIN
    [("epoch", .int c.epoch), ("coins", .bytes c.root), ("weights", .bytes w)]
  pure (FlockSoundness.Randomness.stream k [.str "salt"] 32)

/-! ## What the session and the ledger show -/

/-- When a Pearl-C salt session counts: its commitment is the salt-only layout's, and its log shows the challenger
received `w`, alone, as the salt coin's root, each received root followed by its coin's release
(`releasedRootsOf`). -/
def Counts (c : LiveCommitment) (received : Roots) (w : ByteArray) : Prop :=
  SaltEpoch c ∧ received = [(PouwWindow.SALT, [w])]

/-- The auditor's ledger's entries: each of its lines that `ledgerEntry?` reads. -/
def entries (ledger : Json) : List LedgerEntry :=
  (match ledger with | .arr ls => ls.toList | _ => []).filterMap ledgerEntry?

/-- The root that an open entry names for its window. -/
def openRoot (e : LedgerEntry) : Json := SampledProofs.getD (SampledProofs.getD e.body "window") "root"

/-- What the auditor's ledger shows of session `s` and the weights' root `w`, in its signed order:
* a v2 reserve of `s`, of the salt-only layout;
* after it, a v2 open of `s` on `{root: w}` that binds that reserve;
* after the open, an end of `s`, closed and complete, which released one salt coin and nothing else and names
  `recordLeaf`, the leaf of the session's record, strictly between the open's leaf and its own;
* no other open on `w` or of `s`. -/
def LedgerShows (ledger : Json) (s : String) (w : ByteArray) (recordLeaf : Nat) : Prop :=
  ∃ r o e, r ∈ entries ledger ∧ o ∈ entries ledger ∧ e ∈ entries ledger ∧
    (r.kind = "reserve" ∧ r.session = s ∧ r.schema = SampledProofs.Challenger.LEDGER_SCHEMA ∧
      SampledProofs.jeq (SampledProofs.getD r.body "families")
        (SampledProofs.Challenger.familiesJson SALT_FAMILIES) = true) ∧
    (o.kind = "open" ∧ o.session = s ∧ o.schema = SampledProofs.Challenger.LEDGER_SCHEMA ∧ r.seq < o.seq ∧
      SampledProofs.int? (SampledProofs.getD o.body "reserve") = some (r.seq : Int) ∧
      openRoot o = .str (Flock.hexOf w)) ∧
    (e.kind = "end" ∧ e.session = s ∧ o.seq < e.seq ∧
      SampledProofs.getD e.body "closed" = .bool true ∧ SampledProofs.getD e.body "complete" = .bool true ∧
      SampledProofs.jeq (SampledProofs.getD e.body "released")
        (Json.mkObj [(PouwWindow.SALT, Lean.toJson (1 : Nat))]) = true ∧
      SampledProofs.nat? (SampledProofs.getD e.body "record_leaf") = some recordLeaf ∧
      leafOf o.sig < recordLeaf ∧ recordLeaf < leafOf e.sig) ∧
    ∀ o' ∈ entries ledger, o'.kind = "open" → (openRoot o' = .str (Flock.hexOf w) ∨ o'.session = s) → o' = o

/-- The manifest's `session`, the gate's open answer as the run carries it (`PHC.carried`), names the ledger's session
`a`, the root it was opened on and the commitment its end names. -/
def ManifestNames (m : Json) (a : Audited) (w : ByteArray) : Prop :=
  SampledProofs.getD (at? m "session") "session" = .str a.session ∧
    SampledProofs.getD (at? m "session") "root" = .str (Flock.hexOf w) ∧
    SampledProofs.getD (at? m "session") "commitment" = .str a.commitment

/-! ## The check -/

/-- The check's signature. Its inputs are `saltCheck`'s first five: the verifier's challenger key, the auditor's ledger (its
lines), the host's pins log, the verifier's copy of main's first-parent history and the session's files
`{close, hellos, log, record}`. In place of the window, its two roots and the opening, it takes the pass's manifest `m`
(an object). On acceptance it returns the pass's salt. -/
abbrev Check := (key ledger pins main session m : Json) → R ByteArray

/-- The check accepted the pass `m` with salt `salt`, salted by session `a`'s commitment `c` and its salt coin `o` over the
weights' root `w`. -/
def Accepted (key ledger pins main session m : Json) (w : ByteArray) (a : Audited) (c : LiveCommitment)
    (o : LiveOpening) (salt : ByteArray) : Prop :=
  ∃ (k : ChallengerKey) (received : Roots),
    ChallengerKey.ofJson key = .ok k ∧
    at? m "schema" = .str MANIFEST_SCHEMA ∧ WeightsRoot m w ∧
    pinnedSession ledger (weightsWindow w) k pins main = .ok a ∧
    sessionNamed a k session = .ok (c, received) ∧
    Counts c received w ∧
    LedgerShows ledger a.session w a.recordLeaf ∧
    ManifestNames m a w ∧
    LiveCommitment.ofJson (at? m "coins") = .ok c ∧
    LiveOpening.ofJson (at? m "salt_opening") = .ok o ∧
    liveReason c o PouwWindow.SALT 0 w = .ok none ∧
    liveSalt c o.coin w = .ok salt ∧
    at? m "salt" = .str (Flock.hexOf salt)

/-- What the check accepts is a pass salted by a live session that counts, opened on its weights' root, and the salt it
returns is that session's salt coin's over the weights' root (`Accepted`). -/
def AcceptMeans (check : Check) : Prop :=
  ∀ key ledger pins main session m salt, check key ledger pins main session m = .ok salt →
    ∃ w a c o, Accepted key ledger pins main session m w a c o salt

/-- One pair: two passes that the check accepts against one ledger, under one key, pins log and history, whose weights'
roots agree, were salted by one session `a` of the ledger, under commitments with one root. Their salts agree whenever
their commitments and salt coins do (`LiveCoinBinding` gives the coins, up to a SHA-512 collision). -/
def OnePair (check : Check) : Prop :=
  ∀ key ledger pins main session session' m m' salt salt' w,
    check key ledger pins main session m = .ok salt → check key ledger pins main session' m' = .ok salt' →
    WeightsRoot m w → WeightsRoot m' w →
    ∃ a c c' o o', Accepted key ledger pins main session m w a c o salt ∧
      Accepted key ledger pins main session' m' w a c' o' salt' ∧
      c.root = c'.root ∧ (c = c' → o.coin = o'.coin → salt = salt')

/-! ## A live commitment's binding -/

/-- The frame-v3 message that `Flock.FV3.hash tag parts` hashes. -/
def fv3Message (tag : String) (parts : List ByteArray) : ByteArray :=
  let t := tag.toUTF8
  parts.foldl (fun acc p => acc ++ Flock.be64 p.size.toUInt64 ++ p) (Flock.FV3.FRAME ++ Flock.be32 t.size.toUInt32 ++ t)

/-- The node frames `Flock.Registered.climbL dom cur idx d sibs` hashes, bottom first. -/
def climbFrames (dom : ByteArray) : ByteArray → Nat → Nat → List ByteArray → List ByteArray
  | _, _, _, [] => []
  | cur, idx, d, sib :: rest =>
    let m := if idx % 2 = 0 then fv3Message "node" [dom, Flock.FV3.uint d, Flock.FV3.uint (idx / 2), cur, sib]
      else fv3Message "node" [dom, Flock.FV3.uint d, Flock.FV3.uint (idx / 2), sib, cur]
    m :: climbFrames dom (Flock.Sha512.hash m) (idx / 2) (d + 1) rest

/-- What SHA-512 hashes when `liveReason` checks `o` as the coin at position `pos` of `c`'s tree, whose domain is `dom`,
from the coin up: the input of the coin's inner digest, the hm96 leaf's salt-digest and leaf inputs, the tree leaf's
frame, and the frame of each node on the path. -/
def liveHashed (c : LiveCommitment) (dom : ByteArray) (pos : Nat) (o : LiveOpening) : List ByteArray :=
  let inner := c.scheme.toUTF8.push 0 ++ o.coin
  let leaf := fv3Message "leaf" [dom, Flock.FV3.uint pos, Flock.FV3.uint pos, c.scheme.toUTF8,
    SampledProofs.Challenger.liveLeaf c.scheme o.coin o.salt]
  [inner, Flock.Hm96.Default512.saltPrefix ++ o.salt,
    Flock.Hm96.Default512.leafPrefix ++ Flock.Hm96.Default512.commitString (Flock.Sha512.hash inner) o.salt, leaf] ++
    climbFrames dom (Flock.Sha512.hash leaf) pos 0 o.path

/-- A live commitment opens to one coin at each position. Take two openings that `liveReason` accepts for one coin
(`label`, `index`) of `c`. Either they carry the same coin, or two different messages, one hashed in checking each opening,
have one SHA-512 digest. -/
def LiveCoinBinding : Prop :=
  ∀ (c : LiveCommitment) (o o' : LiveOpening) label index after after' dom pos,
    liveReason c o label index after = .ok none → liveReason c o' label index after' = .ok none →
    SampledProofs.Challenger.liveDomain c.scheme c.epoch c.families = .ok dom → c.position label index = some pos →
    o.coin = o'.coin ∨
      ∃ x ∈ liveHashed c dom pos o, ∃ y ∈ liveHashed c dom pos o', x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y

end Properties.Pouw.LiveSalt
