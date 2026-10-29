import PouwCheck

/-!
# PoUW's Pearl-C live salt: what the check of a pass salted by a live challenger session must establish

A Pearl-C device run (`PearlCDeviceRun`, manifest `MANIFEST_SCHEMA`) can take its salt coin from a live session of the
auditor's challenger instead of from seeded epoch coins (`pouw_pearl_c_host.epoch_of`). The salt's unit is the *salt
epoch*: one install, or one measured interval of an install, over the weights' root `W` (`weightsRootAt 64
"verity/pouw/weights/v1"` of the manifest's weights). The epochs over one `W` are numbered `k` = 0, 1, 2, … in the order
their sessions open, a new number for each reinstall or interval. Epoch `k` over `W` is salted by a session reserved
with Pearl-C's layout, its salt coin alone (`SALT_FAMILIES`), and opened on the epoch's root `epochOpenRoot W k` =
SHA-512(`EPOCH_OPEN_TAG` ‖ `W` ‖ u64be(`k`)). H1's sessions are opened on a window-open root instead. The session releases
its one coin after the epoch's root, and the epoch's salt is `audit.salt_of` of that coin over `W`, as H1 keys a window's
salt by its weights' root. A salt epoch is neither `LiveCommitment.epoch`, the challenger's own number, nor `audit.Epoch`.

Many passes run in one epoch under its one salt. A pass is one engine step's forward, cut so that the tile
caps fit, and the salt fixes seed_B, so B~ is formed once per epoch. Each pass's manifest names the epoch's session, and
its `pass_index` is the epoch's `k`.

`Check` is the signature of the check `core_check` gains for such a pass: it applies `saltCheck`'s rule with the epoch's
root in place of the window-open root and returns the pass's salt. Its port will be `PouwCheck.liveSaltCheck`
(`core_check.live_salt`, the live branch of `_epoch_salt`, taken when the manifest's coins are of `LIVE_SCHEME`). This file
states its properties before the check exists, so its proof shows `AcceptMeans`, `OnePair` and `OneRoot` of
`PouwCheck.liveSaltCheck`, with `LiveCoinBinding` and `EpochRootBinding`. It also states what the Pearl-C period's credit
must show of the passes it credits (`CreditDistinct`), with `RowSeedBinding`.

What the check accepts (`Accepted`) is a pass of `MANIFEST_SCHEMA` that names its session and meets these conditions:

* The check reads the pass's coins only through the challenger's own records, never the manifest's word. It computes the
  epoch's root `R` from the manifest's weights and `pass_index` (`EpochRoot`). The session is the one that the auditor's
  signed v2 ledger opens on `{root: R}` under the window's pinned challenger key and ends closed, run from a commit the
  host's pins log records on main's first-parent line (`pinnedSession`).
* The session's files are that session's (`sessionNamed`), under the commitment the session signed.
* The session counts (`Counts`): the commitment is Pearl-C's salt-only layout, and the session's log shows `R`, alone, as
  the one root it received, each received root followed by its coin's release.
* The ledger shows the session in its signed order (`LedgerShows`):
  - a reserve of the salt-only layout;
  - the open on `R` that binds that reserve;
  - the end, closed and complete, having released one salt coin and nothing else, its record's leaf strictly between
    the open's and the end's;
  - no other open on `R` or of the session.
* The manifest's `session`, the gate's open answer, names that session, the open's `seq`, `R` and the commitment, and
  carries no `secret` (`ManifestNames`).
* The manifest's coins are the session's commitment, and its salt opening is the commitment's `pouw/salt` coin 0, opened
  after `R` (`liveReason`).
* The manifest's salt is `audit.salt_of` of that coin over `W` (`liveSalt`).

The facts PoUW's argument reads from an acceptance:

* **The salt is fixed only after the epoch's root.** The open naming `R` is signed before the session's record, which
  holds the challenger's commitment to the coins (`leafOf o.sig < recordLeaf`). The coin is opened after `R`
  (`liveReason`), and `R` is the root the log shows the coin's release followed. That the gate and the challenger run as
  written, in that order, is S5 (live coins on `trusted-host`); that the coin stays hidden from the hello to its
  release is N1 (X-RO commitments, hm96's hiding leaves). Both are named assumptions of PoUW's model, not this check's
  properties.
* **One epoch, one session, one salt.** The ledger opens `R` once and the session once (`LedgerShows`' last clause), so
  two passes of one epoch have one session and one salt, up to a SHA-512 collision (`OnePair`, from `LiveCoinBinding`).
  One session salts one root (`OneRoot`), and a root names one weights' root and one epoch, up to a SHA-512 collision
  (`EpochRootBinding`).
* **No credit for work before the salt.** The session released one salt coin, once, and ended closed and complete, and
  PoUW's existing pass check binds every unit and record of the pass to the salt.

**Every pass of an epoch does distinct work.** -h3's seed of row `i` of call `u` is
H_"seed-A"(salt ‖ row digest ‖ root_B ‖ u64le(`u`·2^32 + `i`)) (`rowSeed`, `pearl_c.row_seed`), and nothing in it names the
pass. The passes of one epoch share its salt and root_B, so a call index repeated in two of them, over equal rows, gives
equal row seeds, noise and tickets, credited twice. On main the device's call index restarts at 0 every pass
(`pouw_pearl_c_device`: `index = len(cur.matmuls)`), and the verifier checks only that indices are unique within a pass
(`_check_workload`), so nothing refuses that. The rule: the call index runs across the epoch, every index is below 2^32,
and the Pearl-C period refuses to credit two passes of one epoch that share a call index (`PassesDistinct`,
`CreditDistinct`). The row seeds of two different (call index, row) pairs hash messages that differ in their last 8
bytes, so they are different unless the first 32 bytes of SHA-512 collide (`RowSeedBinding`). An epoch's passes are
credited in one period, the one whose close lists the epoch's root.

Under the salt-only layout an abandoned session ends complete too, so a prover can open `n` sessions on `n` epoch roots
and present the epoch whose salt it likes best. The salt decides seed_B, the row seeds and `rotation_key`, not which units
are proven (the draw coin, released after registration). Each epoch's deadline counts from its own session's release, and
every pass of the epoch registers by it, so abandoning buys no time, and an install that outlasts the deadline is cut into
intervals, each its own epoch. Best-of-`n` gains at most `n·ε`, where `ε` is the chance that one salt lets a dishonest
pass through, while nothing keyed by the salt is rewarded. A Pearl-C period refuses a complete salt-only session that no
accepted pass presents, which makes the gain zero within credited periods. That rule is the period checker's, not this
check's.
-/

namespace Properties.Pouw.LiveSalt

open Lean (Json)
open PouwCheck (R at? checkWorkload workloadOf weightsRootAt MANIFEST_SCHEMA)
open SampledProofs.Challenger (Audited ChallengerKey Families LedgerEntry LiveCommitment LiveOpening Roots ledgerEntry?
  leafOf liveReason pinnedSession sessionNamed)

/-! ## The layout, the epoch's root and the salt -/

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

/-- `salt.epoch_open_root`'s domain tag, 25 ASCII bytes. -/
def EPOCH_OPEN_TAG : ByteArray := "verity/pouw-epoch-open/v1".toUTF8

/-- What `epochOpenRoot w k` hashes: `EPOCH_OPEN_TAG` ‖ `w` ‖ u64be(`k`), concatenated with no separator, as
`PouwCheck.windowOpenRoot` concatenates its parts. With `w` 64 bytes and `k` below 2^64 it is 97 bytes, every part at a
fixed offset. -/
def epochMessage (w : ByteArray) (k : Nat) : ByteArray := EPOCH_OPEN_TAG ++ w ++ PouwCommit.be8 k

/-- `salt.epoch_open_root(W, k)`: the root epoch `k` over the weights' root `w` opens its salt session on. -/
def epochOpenRoot (w : ByteArray) (k : Nat) : ByteArray := Flock.Sha512.hash (epochMessage w k)

/-- The manifest `m` is a pass of epoch `k` over the weights' root `w`: its weights' root is `w`, and its `pass_index` is
`k`, an integer below 2^64. -/
def EpochRoot (m : Json) (w : ByteArray) (k : Nat) : Prop :=
  WeightsRoot m w ∧ PouwWindow.natOf? (at? m "pass_index") = some k ∧ k < 2 ^ 64

/-- The window identity that a Pearl-C salt session is opened on: `{root}`, the epoch's root in lowercase hex. -/
def rootWindow (r : ByteArray) : Json := Json.mkObj [("root", .str (Flock.hexOf r))]

/-- `audit.salt_of` over a live commitment: the epoch's 32-byte salt from its opened salt coin and the weights' root. -/
def liveSalt (c : LiveCommitment) (coin w : ByteArray) : Except String ByteArray := do
  let k ← PouwCommit.key? coin PouwCommit.EPOCH_SALT_DOMAIN
    [("epoch", .int c.epoch), ("coins", .bytes c.root), ("weights", .bytes w)]
  pure (FlockSoundness.Randomness.stream k [.str "salt"] 32)

/-! ## What the session and the ledger show -/

/-- When a Pearl-C salt session counts: its commitment is the salt-only layout's, and its log shows the challenger
received `r`, the epoch's root, alone, as the salt coin's root, each received root followed by its coin's release
(`releasedRootsOf`). -/
def Counts (c : LiveCommitment) (received : Roots) (r : ByteArray) : Prop :=
  SaltEpoch c ∧ received = [(PouwWindow.SALT, [r])]

/-- The auditor's ledger's entries: each of its lines that `ledgerEntry?` reads. -/
def entries (ledger : Json) : List LedgerEntry :=
  (match ledger with | .arr ls => ls.toList | _ => []).filterMap ledgerEntry?

/-- The root an open entry names for its window, as `verifyLedger` reads it: `window.root` under v2, `window.commit`
under v1. -/
def openRoot (e : LedgerEntry) : Json :=
  SampledProofs.getD (SampledProofs.getD e.body "window")
    (if e.schema == SampledProofs.Challenger.LEDGER_SCHEMA then "root" else "commit")

/-- What the auditor's ledger shows of session `s` and the epoch's root `r`, in its signed order:
* a v2 reserve of `s`, of the salt-only layout;
* after it, a v2 open of `s` on `{root: r}`, at `openSeq`, that binds that reserve;
* after the open, an end of `s`, closed and complete, which released one salt coin and nothing else and names
  `recordLeaf`, the leaf of the session's record, strictly between the open's leaf and its own;
* no other open, v1 or v2, on `r` or of `s`. -/
def LedgerShows (ledger : Json) (s : String) (r : ByteArray) (recordLeaf openSeq : Nat) : Prop :=
  ∃ rv o e : LedgerEntry, rv ∈ entries ledger ∧ o ∈ entries ledger ∧ e ∈ entries ledger ∧
    (rv.kind = "reserve" ∧ rv.session = s ∧ rv.schema = SampledProofs.Challenger.LEDGER_SCHEMA ∧
      SampledProofs.jeq (SampledProofs.getD rv.body "families")
        (SampledProofs.Challenger.familiesJson SALT_FAMILIES) = true) ∧
    (o.kind = "open" ∧ o.session = s ∧ o.schema = SampledProofs.Challenger.LEDGER_SCHEMA ∧ rv.seq < o.seq ∧
      o.seq = openSeq ∧ SampledProofs.int? (SampledProofs.getD o.body "reserve") = some (rv.seq : Int) ∧
      openRoot o = .str (Flock.hexOf r)) ∧
    (e.kind = "end" ∧ e.session = s ∧ o.seq < e.seq ∧
      SampledProofs.getD e.body "closed" = .bool true ∧ SampledProofs.getD e.body "complete" = .bool true ∧
      SampledProofs.jeq (SampledProofs.getD e.body "released")
        (Json.mkObj [(PouwWindow.SALT, Lean.toJson (1 : Nat))]) = true ∧
      SampledProofs.nat? (SampledProofs.getD e.body "record_leaf") = some recordLeaf ∧
      leafOf o.sig < recordLeaf ∧ recordLeaf < leafOf e.sig) ∧
    ∀ o' ∈ entries ledger, o'.kind = "open" → (openRoot o' = .str (Flock.hexOf r) ∨ o'.session = s) → o' = o

/-- The manifest's `session`, the gate's open answer as the run carries it (`PHC.carried`), names the ledger's session
`a`, the `seq` of its open, the root it was opened on and the commitment its end names, and carries no `secret`. -/
def ManifestNames (m : Json) (a : Audited) (r : ByteArray) (openSeq : Nat) : Prop :=
  SampledProofs.getD (at? m "session") "session" = .str a.session ∧
    SampledProofs.int? (SampledProofs.getD (at? m "session") "seq") = some (openSeq : Int) ∧
    SampledProofs.getD (at? m "session") "root" = .str (Flock.hexOf r) ∧
    SampledProofs.getD (at? m "session") "commitment" = .str a.commitment ∧
    SampledProofs.field? (at? m "session") "secret" = none

/-! ## The check -/

/-- The check's signature. Its inputs are `saltCheck`'s first five: the verifier's challenger key, the auditor's ledger (its
lines), the host's pins log, the verifier's copy of main's first-parent history and the session's files
`{close, hellos, log, record}`. In place of the window, its two roots and the opening, it takes the pass's manifest `m`
(an object). On acceptance it returns the pass's salt, its epoch's. -/
abbrev Check := (key ledger pins main session m : Json) → R ByteArray

/-- The check accepted the pass `m`, a pass of epoch `k` over the weights' root `w`, with salt `salt`, salted by session
`a`'s commitment `c` and its salt coin `o`, opened after the epoch's root. -/
def Accepted (key ledger pins main session m : Json) (w : ByteArray) (k : Nat) (a : Audited) (c : LiveCommitment)
    (o : LiveOpening) (salt : ByteArray) : Prop :=
  ∃ (ck : ChallengerKey) (openSeq : Nat) (received : Roots),
    ChallengerKey.ofJson key = .ok ck ∧
    at? m "schema" = .str MANIFEST_SCHEMA ∧ EpochRoot m w k ∧
    pinnedSession ledger (rootWindow (epochOpenRoot w k)) ck pins main = .ok a ∧
    sessionNamed a ck session = .ok (c, received) ∧
    Counts c received (epochOpenRoot w k) ∧
    LedgerShows ledger a.session (epochOpenRoot w k) a.recordLeaf openSeq ∧
    ManifestNames m a (epochOpenRoot w k) openSeq ∧
    LiveCommitment.ofJson (at? m "coins") = .ok c ∧
    LiveOpening.ofJson (at? m "salt_opening") = .ok o ∧
    liveReason c o PouwWindow.SALT 0 (epochOpenRoot w k) = .ok none ∧
    liveSalt c o.coin w = .ok salt ∧
    at? m "salt" = .str (Flock.hexOf salt)

/-- What the check accepts is a pass salted by a live session that counts, opened on its epoch's root, and the salt it
returns is that session's salt coin's over the weights' root (`Accepted`). -/
def AcceptMeans (check : Check) : Prop :=
  ∀ key ledger pins main session m salt, check key ledger pins main session m = .ok salt →
    ∃ w k a c o, Accepted key ledger pins main session m w k a c o salt

/-! ## Collisions and the messages hashed -/

/-- Two different messages, one from each list, with one SHA-512 digest. -/
def Collides (xs ys : List ByteArray) : Prop :=
  ∃ x ∈ xs, ∃ y ∈ ys, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y

/-- Two different messages, one from each list, whose SHA-512 digests agree in their first 32 bytes. -/
def Collides32 (xs ys : List ByteArray) : Prop :=
  ∃ x ∈ xs, ∃ y ∈ ys, x ≠ y ∧ (Flock.Sha512.hash x).extract 0 32 = (Flock.Sha512.hash y).extract 0 32

/-- The frame-v3 message that `Flock.FV3.hash tag parts` hashes. -/
def fv3Message (tag : String) (parts : List ByteArray) : ByteArray :=
  let t := tag.toUTF8
  parts.foldl (fun acc p => acc ++ Flock.be64 p.size.toUInt64 ++ p) (Flock.FV3.FRAME ++ Flock.be32 t.size.toUInt32 ++ t)

/-- The message `taggedSha512 tag manifest` hashes, when the manifest has identity bytes (`Flock.canonUtf8`). -/
def taggedMessage (tag : String) (manifest : Json) : List ByteArray :=
  match Flock.canonUtf8 manifest with
  | .ok s =>
    let t := tag.toUTF8
    let b := s.toUTF8
    ["verity/tagged-sha512/v1".toUTF8.push 0 ++ Flock.be32 t.size.toUInt32 ++ t ++ Flock.be64 b.size.toUInt64 ++ b]
  | .error _ => []

/-- What SHA-512 hashes when `liveDomain c.scheme c.epoch c.families` computes `c`'s tree domain: the binding of the
scheme, epoch and layout, the positions, and the domain's frame. -/
def domainHashed (c : LiveCommitment) : List ByteArray :=
  let binding := Json.mkObj [("epoch", Lean.toJson c.epoch),
    ("families", SampledProofs.Challenger.familiesJson c.families), ("scheme", Json.str c.scheme)]
  let positions := Json.mkObj [("start", Lean.toJson (0 : Nat)), ("step", Lean.toJson (1 : Nat)),
    ("stop", Lean.toJson (SampledProofs.Challenger.total c.families))]
  let tagged := taggedMessage SampledProofs.Challenger.BINDING binding ++
    taggedMessage "veritor/indexed-domain/range/v1" positions
  match SampledProofs.Challenger.taggedSha512 SampledProofs.Challenger.BINDING binding,
      SampledProofs.Challenger.taggedSha512 "veritor/indexed-domain/range/v1" positions with
  | .ok b, .ok p =>
    tagged ++ [fv3Message "domain" [b, Flock.FV3.uint 1, p, Flock.FV3.uint (SampledProofs.Challenger.total c.families)]]
  | _, _ => tagged

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

/-- Everything SHA-512 hashes when `liveReason c o label index` checks `o` against `c`'s root: `c`'s domain, then the
coin's climb from its position. -/
def hashedBy (c : LiveCommitment) (o : LiveOpening) (label : String) (index : Int) : List ByteArray :=
  match SampledProofs.Challenger.liveDomain c.scheme c.epoch c.families, c.position label index with
  | .ok dom, some pos => domainHashed c ++ liveHashed c dom pos o
  | _, _ => domainHashed c

/-! ## The row seeds and the passes' call indices -/

/-- -h3's label block for seed_A (`pearl_c.S512_LABELS["seed-A"]`): `pearl-c/h3s/seed-A` and a zero byte, zero-padded to
128 bytes. -/
def SEED_A_LABEL : ByteArray :=
  let t := "pearl-c/h3s/seed-A".toUTF8.push 0
  t ++ ByteArray.mk (List.replicate (128 - t.size) (0 : UInt8)).toArray

/-- `x` as 8 bytes, little-endian. -/
def le64 (x : Nat) : ByteArray := ⟨(Array.range 8).map fun j => (x >>> (8 * j)).toUInt8⟩

/-- What `pearl_c.row_seed(salt, leaf, root_b, u, i)` hashes: seed_A's label block, then salt ‖ the row's digest ‖ root_B
‖ u64le(`u`·2^32 + `i`). -/
def rowSeedMessage (salt leaf rootB : ByteArray) (u i : Nat) : ByteArray :=
  SEED_A_LABEL ++ salt ++ leaf ++ rootB ++ le64 (u * 2 ^ 32 + i)

/-- `pearl_c.row_seed` under -h3: the seed of row `i` of call `u`, the first 32 bytes of SHA-512 of `rowSeedMessage`. -/
def rowSeed (salt leaf rootB : ByteArray) (u i : Nat) : ByteArray :=
  (Flock.Sha512.hash (rowSeedMessage salt leaf rootB u i)).extract 0 32

/-- The call indices of the pass `m`'s matmuls, in its manifest's order (none when its workload doesn't parse). -/
def callIndices (m : Json) : List Int :=
  match workloadOf m with
  | .ok (us, _) => us.map (·.index)
  | .error _ => []

/-- The epoch the pass `m` names: its manifest's session's `root`. An accepted pass of epoch `k` over `w` names
`epochOpenRoot w k` there (`ManifestNames`). -/
def epochOf (m : Json) : Json := SampledProofs.getD (at? m "session") "root"

/-- The signature of the Pearl-C period's credit: the verifier's challenger key, the auditor's ledger, the host's pins
log, the verifier's copy of main's first-parent history and the period (its name, its close and the passes it presents,
each a manifest with its session's files). On acceptance it returns the manifests of the passes it credits. -/
abbrev PeriodCredit := (key ledger pins main period : Json) → R (List Json)

/-! ## The properties -/

/-- A live root binds its commitment and its coins. Take two commitments with one root `r`, `⟨s, e, fs, r⟩` and
`⟨s', e', fs', r⟩`, and an opening of each that `liveReason` accepts for one coin (`label`, `index`). Either the
commitments are one (scheme, epoch and layout) and the openings carry the same coin, or two different messages, one hashed
in checking each opening, have one SHA-512 digest. It doesn't bound the salt's choice among a weights' root's epochs: the
prover picks `pass_index`, so it can open several epochs over one `W` and present the salt it likes, and that choice rests
on the grinding term over every epoch the prover can open, at most n·ε for n epochs, until the period check refuses a
complete salt-only session that no accepted pass presents. -/
def LiveCoinBinding : Prop :=
  ∀ (s s' : String) (e e' : Nat) (fs fs' : Families) (r : ByteArray) (o o' : LiveOpening) label index after after',
    liveReason ⟨s, e, fs, r⟩ o label index after = .ok none →
    liveReason ⟨s', e', fs', r⟩ o' label index after' = .ok none →
    ((⟨s, e, fs, r⟩ : LiveCommitment) = ⟨s', e', fs', r⟩ ∧ o.coin = o'.coin) ∨
      Collides (hashedBy ⟨s, e, fs, r⟩ o label index) (hashedBy ⟨s', e', fs', r⟩ o' label index)

/-- An epoch's root names one weights' root and one epoch: two 64-byte weights' roots and two epoch numbers below 2^64
with one epoch root are one pair, or their two messages, different, have one SHA-512 digest. -/
def EpochRootBinding : Prop :=
  ∀ (w w' : ByteArray) (k k' : Nat), w.size = 64 → w'.size = 64 → k < 2 ^ 64 → k' < 2 ^ 64 →
    epochOpenRoot w k = epochOpenRoot w' k' → (w = w' ∧ k = k') ∨ Collides [epochMessage w k] [epochMessage w' k']

/-- One epoch, one session, one salt: two passes that the check accepts against one ledger, under one key, pins log and
history, both passes of epoch `k` over the weights' root `w`, were salted by one session `a` of the ledger, and their
salts agree unless two different messages hashed in checking their salt coins have one SHA-512 digest. It doesn't bound
the salt's choice among a weights' root's epochs: the prover picks `pass_index`, so it can open several epochs over one
`W` and present the salt it likes, and that choice rests on the grinding term over every epoch the prover can open, at
most n·ε for n epochs, until the period check refuses a complete salt-only session that no accepted pass presents. -/
def OnePair (check : Check) : Prop :=
  ∀ key ledger pins main session session' m m' salt salt',
    check key ledger pins main session m = .ok salt → check key ledger pins main session' m' = .ok salt' →
    ∀ w k, EpochRoot m w k → EpochRoot m' w k →
    ∃ a c c' o o', Accepted key ledger pins main session m w k a c o salt ∧
      Accepted key ledger pins main session' m' w k a c' o' salt' ∧
      (salt = salt' ∨ Collides (hashedBy c o PouwWindow.SALT 0) (hashedBy c' o' PouwWindow.SALT 0))

/-- One session salts one epoch: of two passes that the check accepts against one ledger, under one key, pins log and
history, any two acceptances by one session were opened on one epoch root. It doesn't bound the salt's choice among a
weights' root's epochs: the prover picks `pass_index`, so it can open several epochs over one `W` and present the salt it
likes, and that choice rests on the grinding term over every epoch the prover can open, at most n·ε for n epochs, until
the period check refuses a complete salt-only session that no accepted pass presents. -/
def OneRoot (check : Check) : Prop :=
  ∀ key ledger pins main session session' m m' salt salt',
    check key ledger pins main session m = .ok salt → check key ledger pins main session' m' = .ok salt' →
    ∀ w k a c o w' k' a' c' o', Accepted key ledger pins main session m w k a c o salt →
      Accepted key ledger pins main session' m' w' k' a' c' o' salt' →
      a.session = a'.session → epochOpenRoot w k = epochOpenRoot w' k'

/-- A row seed binds its call index and row: the seeds of two different (call index, row) pairs, each part below 2^32,
under any salts, row digests and root_Bs, are different, or their two messages, which differ in their last 8 bytes,
agree in the first 32 bytes of their SHA-512 digests. -/
def RowSeedBinding : Prop :=
  ∀ (salt leaf rootB salt' leaf' rootB' : ByteArray) (u i u' i' : Nat),
    u < 2 ^ 32 → i < 2 ^ 32 → u' < 2 ^ 32 → i' < 2 ^ 32 → (u, i) ≠ (u', i') →
    rowSeed salt leaf rootB u i = rowSeed salt' leaf' rootB' u' i' →
    Collides32 [rowSeedMessage salt leaf rootB u i] [rowSeedMessage salt' leaf' rootB' u' i']

/-- The passes `credited` do distinct work within each epoch: every call index is below 2^32, `row_seed`'s bound, and
two passes that name one epoch share no call index, so no (call index, row) pair, and by `RowSeedBinding` no row seed,
is credited twice. A pass with a call, listed twice, shares that call's index with itself, so it is refused too. -/
def PassesDistinct (credited : List Json) : Prop :=
  (∀ m ∈ credited, ∀ u ∈ callIndices m, 0 ≤ u ∧ u < 2 ^ 32) ∧
    credited.Pairwise fun m m' => epochOf m = epochOf m' → ∀ u ∈ callIndices m, u ∉ callIndices m'

/-- The Pearl-C period credits only passes that do distinct work within each epoch: whatever it credits meets
`PassesDistinct`. -/
def CreditDistinct (credit : PeriodCredit) : Prop :=
  ∀ key ledger pins main period credited,
    credit key ledger pins main period = .ok credited → PassesDistinct credited

end Properties.Pouw.LiveSalt
