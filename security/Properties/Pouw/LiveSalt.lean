module

public import PouwCheck
public import PouwCheck.Pass

/-! What the check of a Pearl-C pass salted by a live challenger session must establish, stated before that check
exists, and what the period's credit must show of the passes it credits. -/

@[expose] public section

namespace Properties.Pouw.LiveSalt

open Lean (Json)
open PouwCheck (R at? checkWorkload workloadOf weightsRootAt MANIFEST_SCHEMA)
open Spot.Challenger (Audited ChallengerKey Families LedgerEntry LiveCommitment LiveOpening Roots ledgerEntry?
  leafOf liveReason pinnedChallengePhase challengePhaseNamed)

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
`PouwCheck.executionPhaseOpenRoot` concatenates its parts. With `w` 64 bytes and `k` below 2^64 it is 97 bytes, every part at a
fixed offset. -/
def epochMessage (w : ByteArray) (k : Nat) : ByteArray := EPOCH_OPEN_TAG ++ w ++ PouwCommit.be8 k

/-- `salt.epoch_open_root(W, k)`: the root epoch `k` over the weights' root `w` opens its salt session on. -/
def epochOpenRoot (w : ByteArray) (k : Nat) : ByteArray := Flock.Sha512.hash (epochMessage w k)

/-- The manifest `m` is a pass of epoch `k` over the weights' root `w`: its weights' root is `w`, and its `pass_index` is
`k`, an integer below 2^64. -/
def EpochRoot (m : Json) (w : ByteArray) (k : Nat) : Prop :=
  WeightsRoot m w ∧ PouwWindow.natOf? (at? m "pass_index") = some k ∧ k < 2 ^ 64

/-- The window identity that a Pearl-C salt session is opened on: `{root}`, the epoch's root in lowercase hex. -/
def rootExecutionPhase (r : ByteArray) : Json := Json.mkObj [("root", .str (Flock.hexOf r))]

/-- `audit.salt_of` over a live commitment: the epoch's 32-byte salt from its opened salt coin and the weights' root. -/
def liveSalt (c : LiveCommitment) (coin w : ByteArray) : Except String ByteArray := do
  let k ← PouwCommit.key? coin PouwCommit.EPOCH_SALT_DOMAIN
    [("epoch", .int c.epoch), ("coins", .bytes c.root), ("weights", .bytes w)]
  pure (FlockSoundness.Randomness.stream k [.str "salt"] 32)

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
  Spot.getD (Spot.getD e.body "window")
    (if e.schema == Spot.Challenger.LEDGER_SCHEMA then "root" else "commit")

/-- What the auditor's ledger shows of session `s` and the epoch's root `r`, in its signed order:
* a v2 reserve of `s`, of the salt-only layout;
* after it, a v2 open of `s` on `{root: r}`, at `openSeq`, that binds that reserve;
* after the open, an end of `s`, closed and complete, which released one salt coin and nothing else and names
  `recordLeaf`, the leaf of the session's record, strictly between the open's leaf and its own;
* no other open, v1 or v2, on `r` or of `s`. -/
def LedgerShows (ledger : Json) (s : String) (r : ByteArray) (recordLeaf openSeq : Nat) : Prop :=
  ∃ rv o e : LedgerEntry, rv ∈ entries ledger ∧ o ∈ entries ledger ∧ e ∈ entries ledger ∧
    (rv.kind = "reserve" ∧ rv.session = s ∧ rv.schema = Spot.Challenger.LEDGER_SCHEMA ∧
      Spot.jeq (Spot.getD rv.body "families")
        (Spot.Challenger.familiesJson SALT_FAMILIES) = true) ∧
    (o.kind = "open" ∧ o.session = s ∧ o.schema = Spot.Challenger.LEDGER_SCHEMA ∧ rv.seq < o.seq ∧
      o.seq = openSeq ∧ Spot.int? (Spot.getD o.body "reserve") = some (rv.seq : Int) ∧
      openRoot o = .str (Flock.hexOf r)) ∧
    (e.kind = "end" ∧ e.session = s ∧ o.seq < e.seq ∧
      Spot.getD e.body "closed" = .bool true ∧ Spot.getD e.body "complete" = .bool true ∧
      Spot.jeq (Spot.getD e.body "released")
        (Json.mkObj [(PouwWindow.SALT, Lean.toJson (1 : Nat))]) = true ∧
      Spot.nat? (Spot.getD e.body "record_leaf") = some recordLeaf ∧
      leafOf o.sig < recordLeaf ∧ recordLeaf < leafOf e.sig) ∧
    ∀ o' ∈ entries ledger, o'.kind = "open" → (openRoot o' = .str (Flock.hexOf r) ∨ o'.session = s) → o' = o

/-- The manifest's `session`, the gate's open answer as the run carries it (`PHC.carried`), names the ledger's session
`a`, the `seq` of its open, the root it was opened on and the commitment its end names, and carries no `secret`. -/
def ManifestNames (m : Json) (a : Audited) (r : ByteArray) (openSeq : Nat) : Prop :=
  Spot.getD (at? m "session") "session" = .str a.session ∧
    Spot.int? (Spot.getD (at? m "session") "seq") = some (openSeq : Int) ∧
    Spot.getD (at? m "session") "root" = .str (Flock.hexOf r) ∧
    Spot.getD (at? m "session") "commitment" = .str a.commitment ∧
    Spot.field? (at? m "session") "secret" = none

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
    pinnedChallengePhase ledger (rootExecutionPhase (epochOpenRoot w k)) ck pins main = .ok a ∧
    challengePhaseNamed a ck session = .ok (c, received) ∧
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
    ("families", Spot.Challenger.familiesJson c.families), ("scheme", Json.str c.scheme)]
  let positions := Json.mkObj [("start", Lean.toJson (0 : Nat)), ("step", Lean.toJson (1 : Nat)),
    ("stop", Lean.toJson (Spot.Challenger.total c.families))]
  let tagged := taggedMessage Spot.Challenger.BINDING binding ++
    taggedMessage "veritor/indexed-domain/range/v1" positions
  match Spot.Challenger.taggedSha512 Spot.Challenger.BINDING binding,
      Spot.Challenger.taggedSha512 "veritor/indexed-domain/range/v1" positions with
  | .ok b, .ok p =>
    tagged ++ [fv3Message "domain" [b, Flock.FV3.uint 1, p, Flock.FV3.uint (Spot.Challenger.total c.families)]]
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
    Spot.Challenger.liveLeaf c.scheme o.coin o.salt]
  [inner, Flock.Hm96.Default512.saltPrefix ++ o.salt,
    Flock.Hm96.Default512.leafPrefix ++ Flock.Hm96.Default512.commitString (Flock.Sha512.hash inner) o.salt, leaf] ++
    climbFrames dom (Flock.Sha512.hash leaf) pos 0 o.path

/-- Everything SHA-512 hashes when `liveReason c o label index` checks `o` against `c`'s root: `c`'s domain, then the
coin's climb from its position. -/
def hashedBy (c : LiveCommitment) (o : LiveOpening) (label : String) (index : Int) : List ByteArray :=
  match Spot.Challenger.liveDomain c.scheme c.epoch c.families, c.position label index with
  | .ok dom, some pos => domainHashed c ++ liveHashed c dom pos o
  | _, _ => domainHashed c

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
def epochOf (m : Json) : Json := Spot.getD (at? m "session") "root"

/-- The signature of the Pearl-C period's credit: the verifier's challenger key, the auditor's ledger, the host's pins
log, the verifier's copy of main's first-parent history and the period (its name, its close and the passes it presents,
each a manifest with its session's files). On acceptance it returns the manifests of the passes it credits. -/
abbrev PeriodCredit := (key ledger pins main period : Json) → R (List Json)

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

def CreditDistinct (credit : PeriodCredit) : Prop :=
  ∀ key ledger pins main period credited,
    credit key ledger pins main period = .ok credited → PassesDistinct credited

end Properties.Pouw.LiveSalt
