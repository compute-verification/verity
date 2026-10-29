import Flock.Draw
import PouwCommit.Bindings
import PouwCommit.Draws
import PouwWindow.Rounds
import PouwCheck.Inputs
import SampledProofs.ChallengerRecord
import SampledProofs.ChallengerSession
import Identity.Domains

/-!
# The pass's epoch and the window's draw (`core_check.py`, "the pass's epoch and the window's draw")

`passEpoch` is `core_check.pass_epoch`: the window's pass's manifest re-starts its epoch under the challenger's own coin
commitment, and the window's context names the pass by its SHA-512. A pass salted by epoch coins (`MANIFEST_SCHEMA`)
carries the epoch's commitment, which must be the challenger's, and its salt coin's opening, opened after the weights'
root (`coinReasonAt`); a live Pearl-C pass, whose manifest's coins are a live session's, took its salt coin from a
salt-only session of the auditor host's challenger opened on its salt epoch's root, which its verifier reads only through
the auditor's ledger and the session's own files (`{key, ledger, main, pins, session}`, `livePassSalt`), never the
pass's word; a pass recorded before epoch coins (`MANIFEST_RECORDED`, read only in a replay) carries a beacon. Either
way the manifest's salt must be the epoch's. A certified window's verifier holds the network certifier's salt for
the PoUW window its context names (`windowOf`), the window's stream and prover, and what PoUW recorded with the certifier
before its salt (`pouwRecord`, `salt.pouw_record`), and passes only a pass salted by them (`MANIFEST_CERTIFIED`, the salt
`pouwSalt`, `salt.pouw_salt`); no other verifier passes such a pass. A refusal is `coin` (the coin rule's, Python's
`DomainError`) or `manifest` (a manifest it can't read, or not the window's pass), with Python's words.

`drawCheck` is `core_check.draw`: the epoch's draw coin opens after the registration, and the drawn units are
`Flock.Draw.subset` of the stream `"draw"` of the key the coin derives under the stream, the registration and its
receipt.

`liveSaltCheck` is `core_check.live_salt` (LIVESALT2, stated in `Properties.Pouw.LiveSalt`), the Pearl-C form of
`saltCheck` (`PouwCheck.Rounds`): the session's root is the pass's salt epoch's root (`epochOpenRoot` of its weights'
root W and its `pass_index`), not a window's open root, and the session counts only when it is reserved for the salt coin alone (`PouwWindow.SALT_FAMILIES`) and the ledger ends it
complete, having released that coin. It sits here, not beside `saltCheck`, because `passEpoch` reads it and
`PouwCheck.Rounds` imports this module. `saltSession` is the session both read.

Both read the coin versions of `coinVersion`: the current one, whose binding and weights' root are `tagged`'s whole
SHA-512, and, only in a replay of a recorded pass, the e2e-service passes', which keep its first 32 bytes. A replay of a
recorded draw also reads the e2e-service's first coin commitments, named as the current one but on SHA-256 as core was
then: bound by `taggedSha256`, after the current binding, and drawn by `deriveV1` and `streamV1`.
-/

namespace PouwCheck

open Lean (Json)
open PouwCommit (CoinOpening pyFromHex pyRepr tagged leafValue Domain)
open SampledProofs.Challenger (Audited ChallengerKey LedgerEntry LiveCommitment LiveOpening Roots ledgerEntry?
  pinnedSession sessionNamed)

def MANIFEST_SCHEMA : String := "verity-vllm/pouw-pearl-c-run/v2"
def MANIFEST_RECORDED : String := Domains.VLLM_PEARL_C_RUN_V1
def MANIFEST_CERTIFIED : String := Domains.VLLM_PEARL_C_RUN_CERTIFIED
def WINDOW_SEGMENT : String := "pouw-window"
def SALT_BYTES : Nat := 64
def SALT_TAG : ByteArray := Domains.POUW_PERIOD_SALT.toUTF8
def RECORD_TAG : ByteArray := Domains.POUW_RECORD.toUTF8
def WINDOW_OPEN_TAG : ByteArray := "verity/pouw-window-open/v1".toUTF8
def EPOCH_OPEN_TAG : ByteArray := "verity/pouw-epoch-open/v1".toUTF8
def WEIGHTS_ROOT_BYTES : Nat := 64
def STREAM_BYTES : Nat := 64
def PROVER_BYTES : Nat := 64
def COINS_RECORDED : String := Domains.POUW_EPOCH_COINS_RECORDED
def DRAW_DOMAIN : String := "verity/pouw/served-zk-draw/v1"
def U64 : Int := 2 ^ 64
def OPENING_FORM : String :=
  "a coin opening is {epoch, label, after, coin, salt, path}: an integer, a string, hex, and SHA-512 digests"
def MATMULS_FORM : String := "the pass's matmuls are not a list of objects with integer index, m, k, n and weight"
def WEIGHTS_FORM : String := "the pass's weights are not an object from each weight, in decimal, to an object with a hex root"

/-- `COIN_VERSIONS`: the bytes of `tagged` a coin scheme's binding and weights' root keep, and the weights' root's tag. -/
def coinVersion (scheme : String) : Option (Nat × String) :=
  if scheme == PouwCommit.SCHEME then some (64, Domains.POUW_WEIGHTS_ROOT)
  else if scheme == COINS_RECORDED then some (32, Domains.POUW_WEIGHTS_ROOT_RECORDED)
  else none

structure Coins where
  scheme : String
  epoch : Nat
  root : ByteArray
  keep : Nat
  weightsTag : String

def Coins.same (a b : Coins) : Bool := a.scheme == b.scheme && a.epoch == b.epoch && a.root == b.root

/-- `_hex`: `bytes.fromhex` of a string. -/
def hex? : Json → Option ByteArray
  | .str s => (pyFromHex s).toOption
  | _ => none

def at? (j : Json) (k : String) : Json := (PouwWindow.field? j k).getD .null

/-- `j == .str s`, by a match the kernel reduces: `BEq Json` is a `partial def`, opaque to the kernel, so a property of a
check that compares with it can't be proved. -/
def jsonStrIs (j : Json) (s : String) : Bool := match j with | .str t => t == s | _ => false

/-- `j == .bool true`, by a match the kernel reduces (`jsonStrIs`). -/
def jsonTrue (j : Json) : Bool := match j with | .bool true => true | _ => false

example : jsonStrIs (.str MANIFEST_SCHEMA) MANIFEST_SCHEMA = true := by decide
example : jsonStrIs (.str MANIFEST_RECORDED) MANIFEST_SCHEMA = false := by decide
example : jsonStrIs .null "" = false := by decide
example : jsonTrue (.bool true) = true := by decide
example : jsonTrue (.bool false) = false := by decide
example : jsonTrue .null = false := by decide
example : jsonStrIs (at? (Json.mkObj [("salt", .str "ab"), ("schema", .null)]) "salt") "ab" = true := by decide
example : jsonTrue (at? (Json.mkObj [("closed", .bool true), ("complete", .bool true)]) "complete") = true := by decide

/-- `coins_of`. -/
def coinsOf (j : Json) (recorded : Bool) : Except String Coins := do
  let bad := s!"a coin commitment's scheme is {pyRepr PouwCommit.SCHEME}"
  let some (.str scheme) := PouwWindow.field? j "scheme" | throw bad
  unless scheme == PouwCommit.SCHEME || (recorded && scheme == COINS_RECORDED) do throw bad
  let some epoch := PouwWindow.natOf? (at? j "epoch") | throw "an epoch is a nonnegative integer"
  let some root := (hex? (at? j "root")).filter (·.size == 64) | throw "a coin commitment's root is 64 bytes"
  let some (keep, tag) := coinVersion scheme | throw bad
  pure ⟨scheme, epoch, root, keep, tag⟩

/-- `opening_of`. -/
def openingOf (j : Json) : Except String CoinOpening := do
  let some epoch := PouwWindow.intOf? (at? j "epoch") | throw OPENING_FORM
  let .str label := at? j "label" | throw OPENING_FORM
  let some after := hex? (at? j "after") | throw OPENING_FORM
  let some coin := hex? (at? j "coin") | throw OPENING_FORM
  let some salt := hex? (at? j "salt") | throw OPENING_FORM
  let .arr ps := at? j "path" | throw OPENING_FORM
  let some path := ps.toList.mapM fun p => (hex? p).filter (·.size == 64) | throw OPENING_FORM
  pure ⟨epoch, label, after, coin, salt, path⟩

/-- `core_check.tagged_sha256`: `tagged` as it was on SHA-256, the binding of `COIN_EARLY`. -/
def taggedSha256 (tag : String) (parts : List PouwCommit.Part) : ByteArray :=
  Flock.Sha256.hash (parts.foldl (fun acc p => acc ++ PouwCommit.frame p) (tag.toUTF8.push 0))

/-- `core_check.derive_v1`: `derive` as it was (`verity/randomness/derive/v1`, SHA-256), `COIN_EARLY`'s draw key. -/
def deriveV1 (source : ByteArray) (domain : String) (context : List (String × PouwCommit.Part)) : ByteArray :=
  let ctx := ((context.toArray.qsort fun a b => a.1 < b.1).toList.foldl
    (fun acc kv => acc ++ PouwCommit.frame (.str kv.1) ++ PouwCommit.frame kv.2) ByteArray.empty)
  Flock.Sha256.hash (Domains.RANDOMNESS_DERIVE_V1.toUTF8.push 0 ++ PouwCommit.frame (.bytes source) ++
    PouwCommit.frame (.str domain) ++ PouwCommit.frame (.bytes ctx))

/-- `core_check.stream_v1`: `Key.stream` as it was (`verity/randomness/stream/v1`, SHA-256 blocks), `COIN_EARLY`'s draw
stream. -/
def streamV1 (key : ByteArray) (index : List PouwCommit.Part) (n : Nat) : ByteArray :=
  let head := Domains.RANDOMNESS_STREAM_V1.toUTF8.push 0 ++ key ++ FlockSoundness.Randomness.frames index
  ((List.range ((n + 31) / 32)).foldl (fun acc c => acc ++ Flock.Sha256.hash (head ++ PouwCommit.be8 c))
    ByteArray.empty).extract 0 n

/-- `core_check.coin_version`: `coins.coin_reason` under the commitment's scheme, and whether the coin opened under the
binding of the e2e-service's first coin commitments (`COIN_EARLY`: `PouwCommit.SCHEME` bound by `taggedSha256`), which
`early` also tries when the current one doesn't open. -/
def coinVersionAt (c : Coins) (o : CoinOpening) (label : String) (after : ByteArray) (early : Bool) :
    Except String (Option String × Bool) := do
  let rank := if label == "salt" then 0 else 1
  if o.epoch != c.epoch then return (some s!"coin is epoch {o.epoch}'s, not epoch {c.epoch}'s", false)
  if o.label != label then return (some s!"coin is the epoch's {pyRepr o.label} coin, not its {pyRepr label} coin", false)
  if o.coin.size != PouwCommit.COIN_BYTES || o.salt.size != 192 then return (some "coin opening is malformed", false)
  let opens (b : ByteArray) : Except String Bool := do
    let d ← Domain.mk? .sha512 b (-1) 2
    d.verify c.root rank (leafValue o.coin o.salt) o.path fun v => pure (d.leaf rank c.scheme v)
  let old ← if ← opens ((tagged c.scheme [.int c.epoch]).extract 0 c.keep) then pure false
    else if early && c.scheme == PouwCommit.SCHEME && (← opens (taggedSha256 c.scheme [.int c.epoch])) then pure true
    else return (some "coin does not open against the epoch's coin commitment", false)
  if o.after != after then return (some "coin was not opened after what the verifier recorded", old)
  return (none, old)

/-- `core_check.coin_reason`: `coins.coin_reason` under the commitment's scheme. -/
def coinReasonAt (c : Coins) (o : CoinOpening) (label : String) (after : ByteArray) : Except String (Option String) :=
  (·.1) <$> coinVersionAt c o label after false

/-- `PouwCommit.foldRoots` at `tag`, its first `keep` bytes: `core_check.weights_root`
(`audit.weights_root`, an epoch's, under a scheme's version) and `Credit.weightsRoot` (`weights.weights_root`, the
weights' root a live H1 window's open root binds). -/
def weightsRootAt (keep : Nat) (tag : String) (roots : List (Nat × ByteArray)) : ByteArray :=
  (PouwCommit.foldRoots tag roots).extract 0 keep

/-- A refusal: its code (`coin` or `manifest`) and why. -/
abbrev R := Except (String × String)

def coin {α : Type} (why : String) : R α := throw ("coin", why)
def manifest {α : Type} (why : String) : R α := throw ("manifest", why)

/-- An error of the port's own (a reading Python raises on) as a refusal it can't make. -/
def port {α : Type} (x : Except String α) : R α := x.mapError fun e => ("error", e)

structure Matmul where
  index : Int
  m : Int
  k : Int
  n : Int
  weight : Int

def isDecimal (s : String) : Bool :=
  !s.isEmpty && s.all Char.isDigit && (s.length == 1 || s.front != '0')

/-- `_workload`: the matmuls (each a shape and 64-bit ids) and the weights' roots. -/
def workloadOf (m : Json) : R (List Matmul × List (Nat × ByteArray)) := do
  let .arr us := at? m "matmuls" | manifest MATMULS_FORM
  let some us := us.toList.mapM (fun u => match u with
      | .obj _ => do
        let f := fun k => PouwWindow.intOf? (at? u k)
        pure (Matmul.mk (← f "index") (← f "m") (← f "k") (← f "n") (← f "weight"))
      | _ => none) | manifest MATMULS_FORM
  for u in us do
    if min u.m (min u.k u.n) < 1 then coin "a shape has m, k, n >= 1"
    if !(0 ≤ u.index && u.index < U64 && 0 ≤ u.weight && u.weight < U64) then
      coin "a matmul's index and weight id are 64-bit"
  let .obj ws := at? m "weights" | manifest WEIGHTS_FORM
  let roots ← ws.toArray.toList.mapM fun (w, v) => do
    let some r := (match v with | .obj _ => hex? (at? v "root") | _ => none) | manifest WEIGHTS_FORM
    if !isDecimal w then manifest WEIGHTS_FORM
    pure (w.toNat!, r)
  pure (us, roots)

/-- `_check_workload`. -/
def checkWorkload (us : List Matmul) (roots : List (Nat × ByteArray)) : R Unit := do
  if (us.map (·.index)).eraseDups.length != us.length then coin "matmul indices are unique"
  let mut dims : List (Int × (Int × Int)) := []
  for u in us do
    match dims.lookup u.weight with
    | some d => if d != (u.n, u.k) then coin s!"every matmul on weight {u.weight} has the weight's (n, k)"
    | none => dims := dims ++ [(u.weight, (u.n, u.k))]
  let held := (roots.map fun (w, _) => (w : Int)).mergeSort (· ≤ ·)
  let want := (dims.map (·.1)).mergeSort (· ≤ ·)
  if held != want then coin "every weight of the workload is committed, and only those"

/-- `salt.window_of`: the PoUW window a context names, `<prefix>/pouw-window/<p>/<manifest SHA-512>`. -/
def windowOf (context : String) : Option Nat :=
  let parts := context.splitOn "/"
  let n := parts.length
  let d := parts.getD (n - 2) ""
  if 3 ≤ n && parts.getD (n - 3) "" == WINDOW_SEGMENT && isDecimal d && d.toNat! < 2 ^ 64 then some d.toNat! else none

/-- `salt.pouw_salt`: the salt PoUW applies in window `p` for the registration of `stream` and `prover`,
SHA-512(`SALT_TAG` ‖ reveal ‖ u64be(p) ‖ stream ‖ prover), every part fixed-width (64, 8, 64 and 64 bytes). -/
def pouwSalt (reveal : ByteArray) (p : Nat) (stream prover : ByteArray) : ByteArray :=
  Flock.Sha512.hash (SALT_TAG ++ reveal ++ PouwCommit.be8 p ++ stream ++ prover)

/-- `salt.pouw_record`: what PoUW records with the certifier for a window before its salt,
SHA-512(`RECORD_TAG` ‖ weights' root ‖ stream ‖ prover), every part 64 bytes. -/
def pouwRecord (weights stream prover : ByteArray) : ByteArray :=
  Flock.Sha512.hash (RECORD_TAG ++ weights ++ stream ++ prover)

/-- `salt.SESSION`: a challenger session's id as the auditor's gate assigns it, 1 to 64 of `[A-Za-z0-9._:-]`, the first
alphanumeric. -/
def sessionId? (s : String) : Bool :=
  let ok := fun (c : Char) => c.isAlphanum || c == '.' || c == '_' || c == ':' || c == '-'
  1 ≤ s.length && s.length ≤ 64 && s.all ok && s.front.isAlphanum

/-- `salt.window_open_root`: a live window's open root, what its session opens on and its salt coin opens after,
SHA-512(`WINDOW_OPEN_TAG` ‖ weights' root ‖ round-0 request table's root ‖ SHA-512(the session's ASCII id)), every part
64 bytes. -/
def windowOpenRoot (weights requests : ByteArray) (session : String) : ByteArray :=
  Flock.Sha512.hash (WINDOW_OPEN_TAG ++ weights ++ requests ++ Flock.Sha512.hash session.toUTF8)

/-- `salt.epoch_open_root`: a Pearl-C salt epoch's open root, what its salt-only session opens on and its salt coin opens
after, SHA-512(`EPOCH_OPEN_TAG` ‖ the weights' root ‖ u64be(`k`)), `k` the epoch's number among the weights' root's
epochs (the manifest's `pass_index`). -/
def epochOpenRoot (weights : ByteArray) (k : Nat) : ByteArray :=
  Flock.Sha512.hash (EPOCH_OPEN_TAG ++ weights ++ PouwCommit.be8 k)

/-- `_certified_salt`: a certified window's pass under the window's `stream` and `prover`: `committed`, what the
certifier recorded before it drew the window's salt, is `pouwRecord` of the pass's weights' root, the stream and the
prover, and the pass's salt is `pouwSalt` of the certifier's reveal for the PoUW window the window's context names. -/
def certifiedSalt (m : Json) (reveal : ByteArray) (context : String) (committed stream prover : Option ByteArray) :
    R (ByteArray × String) := do
  let schema := at? m "schema"
  if schema != .str MANIFEST_CERTIFIED then
    let .str s := schema | manifest "the pass's manifest names no schema"
    coin s!"schema {pyRepr s}, not {MANIFEST_CERTIFIED}: the window is certified"
  let (us, roots) ← workloadOf m
  if reveal.size != SALT_BYTES then coin s!"the verifier's reveal is not a {SALT_BYTES}-byte salt"
  let some p := windowOf context | coin "the window's context names no PoUW window, whose certifier salt the pass's is"
  checkWorkload us roots
  let (some s, some v) := (stream.filter (·.size == STREAM_BYTES), prover.filter (·.size == PROVER_BYTES))
    | coin s!"the verifier holds no {STREAM_BYTES}-byte stream digest and {PROVER_BYTES}-byte prover id for the window"
  let some c := committed.filter (·.size == WEIGHTS_ROOT_BYTES)
    | coin "the verifier holds no record the certifier made before it drew the window's salt"
  if pouwRecord (weightsRootAt 64 Domains.POUW_WEIGHTS_ROOT roots) s v != c then
    coin "the pass's weights, stream and prover are not those the certifier recorded before it drew the window's salt"
  let salt := pouwSalt reveal p s v
  if at? m "salt" != .str (Flock.hexOf salt) then
    coin "the pass's salt is not the certifier's for its window, stream and prover"
  pure (salt, "certifier")

/-- `core_check.NO_RESERVE`: a salt rule's refusal of a session opened without a reserve. -/
def NO_RESERVE : String :=
  "the auditor's ledger opens the window's session without reserving it: a window's open root binds the session a " ++
    "signed reserve entry names"

/-- `core_check._salt_session`: the session the salt rules read, the one the auditor's signed v2 ledger opens and ends
closed for `window` under the pinned challenger key, run from a commit the host's pins log records on main's
first-parent line (`pinnedSession`), and its files (`sessionNamed`): the ledger's session, its signed commitment and the
roots its log shows received. -/
def saltSession (key ledger pins main window session : Json) : R (Audited × LiveCommitment × Roots) := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  (pinnedSession ledger window k pins main >>= fun a => (a, ·) <$> sessionNamed a k session).mapError
    fun e => ("coin", s!"the challenger's session is not trusted: {e}")

/-- `core_check._named`: session `sid`'s entries in the auditor's ledger (which `pinnedSession` has read). -/
def sessionEntries (ledger : Json) (sid : String) : List LedgerEntry :=
  ((match ledger with | .arr ls => ls.toList | _ => []).filterMap ledgerEntry?).filter (·.session == sid)

/-- `core_check.RESERVE_LAYOUT`, `INCOMPLETE`, `LIVE_NOT_RECEIVED`, `LIVE_NOT_SIGNED`, `LIVE_SALT` and `LIVE_SESSION`:
the reasons of a `live_salt` refusal. -/
def RESERVE_LAYOUT : String :=
  "the auditor's ledger reserves the pass's session for another layout than a Pearl-C salt session's, its salt coin " ++
    "alone"
def INCOMPLETE : String :=
  "the auditor's ledger does not end the pass's session complete, having released its salt coin and nothing else"
def LIVE_NOT_RECEIVED : String :=
  "the challenger's log does not show the epoch's root as the one root its salt coin followed"
def LIVE_NOT_SIGNED : String := "the pass's coin commitment is not the one its session's challenger signed"
def LIVE_SALT : String := "the pass's salt is not its live coin's and the weights'"
def LIVE_SESSION : String :=
  "the pass's session is not the gate's answer, without its secret, for the session the auditor's ledger opened on " ++
    "its epoch's root: its session, seq, root and commitment"

/-- `core_check.PASS_FORM` and `CALL_RANGE`: a live pass's manifest that names no salt epoch, or a call index that -h3's
row seed can't take. -/
def PASS_FORM : String :=
  "the pass's manifest names no epoch: `pass_index` is an integer in [0, 2^64), its epoch's number among its " ++
    "weights' root's epochs"
def CALL_BOUND : Int := 2 ^ 32
def CALL_RANGE : String := "a call index is 2^32 or more: -h3's row seed takes call indices below 2^32"

/-- `core_check._carries`: `s`, the session a pass's manifest carries, is the gate's open answer for session `sid`,
opened at ledger entry `seq` on `root` under the commitment whose root is `commitment` (hex), without its hello's
`secret`. -/
def carries (s : Json) (sid : String) (seq : Nat) (root : ByteArray) (commitment : String) : Bool :=
  match s with
  | .obj _ => (PouwWindow.field? s "secret").isNone && jsonStrIs (at? s "session") sid &&
      PouwWindow.intOf? (at? s "seq") == some (seq : Int) && jsonStrIs (at? s "root") (Flock.hexOf root) &&
      jsonStrIs (at? s "commitment") commitment
  | _ => false

example : carries (Json.mkObj [("session", .str "s1"), ("seq", (4 : Nat)), ("root", .str "0102"),
    ("commitment", .str "ab")]) "s1" 4 ⟨#[1, 2]⟩ "ab" = true := by decide
example : carries (Json.mkObj [("session", .str "s1"), ("seq", (4 : Nat)), ("root", .str "0102"),
    ("commitment", .str "ab"), ("secret", .str "00")]) "s1" 4 ⟨#[1, 2]⟩ "ab" = false := by decide
example : carries (Json.mkObj [("session", .str "s1"), ("seq", (4 : Nat)), ("root", .str "0103"),
    ("commitment", .str "ab")]) "s1" 4 ⟨#[1, 2]⟩ "ab" = false := by decide

/-- `claimed == c` for two live coin commitments, field by field: what comparing their `toJson`'s `compress` decided,
since `toJson` is injective, by functions the kernel reduces (`Json.compress` is a `partial def`). -/
def sameCommitment (a b : LiveCommitment) : Bool :=
  a.scheme == b.scheme && a.epoch == b.epoch && a.families == b.families && a.root == b.root

example : sameCommitment ⟨"verity/live", 3, [("pouw/salt", 1)], ⟨#[1, 2]⟩⟩ ⟨"verity/live", 3, [("pouw/salt", 1)], ⟨#[1, 2]⟩⟩
    = true := by decide
example : sameCommitment ⟨"verity/live", 3, [("pouw/salt", 1)], ⟨#[1, 2]⟩⟩ ⟨"verity/live", 3, [("pouw/salt", 1)], ⟨#[1, 3]⟩⟩
    = false := by decide
example : sameCommitment ⟨"verity/live", 3, [("pouw/salt", 1)], ⟨#[1, 2]⟩⟩ ⟨"verity/live", 4, [("pouw/salt", 1)], ⟨#[1, 2]⟩⟩
    = false := by decide

/-- `core_check.live_salt` (LIVESALT2, `Properties.Pouw.LiveSalt`): a live Pearl-C pass's salt, `saltCheck`'s rule with
the pass's epoch's root R in place of a window's open root, its session and commitment read from the auditor's ledger
and the session's own files, never the manifest `m`. The manifest is `MANIFEST_SCHEMA`'s, its every call index below
2^32 (`CALL_RANGE`), W the weights' root of its workload and k its `pass_index`, a natural below 2^64 (`PASS_FORM`):
R is `epochOpenRoot W k`, computed here, never read from the manifest. The session is the one `saltSession` reads for
`{root: R}`, and it counts: its commitment is a Pearl-C salt session's (`PouwWindow.saltEpochReason`), the ledger
reserved it for that layout, opened it binding that reserve and ended it complete, having released its salt coin and
nothing else, and its log shows R, alone, as the root its salt coin followed. Then the manifest's coins are that
commitment, its salt opening the commitment's `pouw/salt` coin 0 opened after R (`liveReason`), its salt that coin's
over W, and its session the gate's answer for the session without its secret (`carries`). Else the refusal
(`manifest`, `coin` or `salt`). -/
def liveSaltCheck (key ledger pins main session m : Json) : R ByteArray := do
  let .obj _ := m | manifest "the pass's manifest is not a JSON object"
  let schema := at? m "schema"
  unless jsonStrIs schema MANIFEST_SCHEMA do
    let .str s := schema | manifest "the pass's manifest names no schema"
    coin s!"schema {pyRepr s}, not {MANIFEST_SCHEMA}"
  let (us, roots) ← workloadOf m
  checkWorkload us roots
  if us.any (·.index ≥ CALL_BOUND) then manifest CALL_RANGE
  let some k := (PouwWindow.natOf? (at? m "pass_index")).filter (· < 2 ^ 64) | manifest PASS_FORM
  let w := weightsRootAt 64 "verity/pouw/weights/v1" roots
  let r := epochOpenRoot w k
  let (a, c, received) ← saltSession key ledger pins main (Json.mkObj [("root", .str (Flock.hexOf r))]) session
  if let some why := PouwWindow.saltEpochReason c then throw ("salt", why)
  let named := sessionEntries ledger a.session
  let (some reserve, some opened, some ended) :=
      (named.find? (·.kind == "reserve"), named.find? (·.kind == "open"), named.find? (·.kind == "end"))
    | throw ("salt", NO_RESERVE)
  unless SampledProofs.jeq (SampledProofs.getD reserve.body "families")
      (SampledProofs.Challenger.familiesJson PouwWindow.SALT_FAMILIES) do throw ("salt", RESERVE_LAYOUT)
  unless SampledProofs.int? (SampledProofs.getD opened.body "reserve") == some (reserve.seq : Int) do
    throw ("salt", NO_RESERVE)
  unless jsonTrue (SampledProofs.getD ended.body "complete") &&
      SampledProofs.jeq (SampledProofs.getD ended.body "released")
        (Json.mkObj [(PouwWindow.SALT, Lean.toJson (1 : Nat))]) do throw ("salt", INCOMPLETE)
  unless received == [(PouwWindow.SALT, [r])] do throw ("salt", LIVE_NOT_RECEIVED)
  let claimed ← (LiveCommitment.ofJson (at? m "coins")).mapError fun e => ("coin", s!"the pass's coins are malformed: {e}")
  unless sameCommitment claimed c do coin LIVE_NOT_SIGNED
  let o ← (LiveOpening.ofJson (at? m "salt_opening")).mapError fun e => ("coin", s!"the salt's {e}")
  if let some why ← port (SampledProofs.Challenger.liveReason c o PouwWindow.SALT 0 r) then coin s!"the salt's {why}"
  let seed ← port (PouwCommit.key? o.coin PouwCommit.EPOCH_SALT_DOMAIN
    [("epoch", .int c.epoch), ("coins", .bytes c.root), ("weights", .bytes w)])
  let salt := FlockSoundness.Randomness.stream seed [.str "salt"] 32
  unless jsonStrIs (at? m "salt") (Flock.hexOf salt) do coin LIVE_SALT
  unless carries (at? m "session") a.session opened.seq r a.commitment do coin LIVE_SESSION
  pure salt

/-- `core_check.LIVE_FORM` and `LIVE_UNHELD`: the reasons a live Pearl-C pass's verifier refuses it before `live_salt`. -/
def LIVE_FORM : String :=
  "a live pass's verifier holds {key, ledger, main, pins, session}: the pinned challenger key, the auditor's ledger, " ++
    "the host's pins log, main's first-parent history and the session's files"
def LIVE_UNHELD : String :=
  "the pass's coins are a live session's: only a verifier holding the auditor's ledger reads a pass salted by a live " ++
    "session"

/-- `core_check._live_salt`: a live Pearl-C pass's salt, `liveSaltCheck` under the verifier's `live` inputs
`{key, ledger, main, pins, session}`, its `salt` refusals `coin`'s. -/
def livePassSalt (m live : Json) : R (ByteArray × String) := do
  if live == .null then coin LIVE_UNHELD
  unless SampledProofs.keys? live == some #["key", "ledger", "main", "pins", "session"] do coin LIVE_FORM
  let g := SampledProofs.getD live
  let salt ← (liveSaltCheck (g "key") (g "ledger") (g "pins") (g "main") (g "session") m).mapError
    fun (code, e) => (if code == "manifest" || code == "error" then code else "coin", e)
  pure (salt, "live coin")

/-- `core_check._coin_salt`: a pass salted by epoch coins the verifier holds as a commitment (`coinsOf`), its salt coin
opened against them after the weights' root. -/
def coinPassSalt (m coins : Json) (recorded : Bool) (us : List Matmul) (roots : List (Nat × ByteArray)) :
    R (ByteArray × String) := do
  let c ← (coinsOf coins recorded).mapError fun e => ("coin", s!"the verifier's coins are malformed: {e}")
  let (o, claimed) ← (do pure (← openingOf (at? m "salt_opening"), ← coinsOf (at? m "coins") recorded))
    |>.mapError fun e => ("coin", s!"the pass's coins are malformed: {e}")
  if !claimed.same c then coin "the pass's coin commitment is not the epoch's"
  checkWorkload us roots
  let root := weightsRootAt c.keep c.weightsTag roots
  if let some why := (← port (coinReasonAt c o "salt" root)) then coin s!"the salt's {why}"
  let k ← port (PouwCommit.key? o.coin PouwCommit.EPOCH_SALT_DOMAIN
    [("epoch", .int c.epoch), ("coins", .bytes c.root), ("weights", .bytes root)])
  pure (FlockSoundness.Randomness.stream k [.str "salt"] 32, "coin")

/-- `_epoch_salt`: the pass's salt re-started from its manifest, and how it was had. -/
def epochSalt (m : Json) (coins : Json) (recorded : Bool) (reveal : Option ByteArray) (context : String)
    (committed stream prover : Option ByteArray) (live : Json) : R (ByteArray × String) := do
  if let some r := reveal then return (← certifiedSalt m r context committed stream prover)
  let schema := at? m "schema"
  if schema == .str MANIFEST_CERTIFIED then
    coin (s!"the pass's manifest is {MANIFEST_CERTIFIED}, salted by the network certifier: only a certified window's " ++
      "verifier, which holds the certifier's reveal, reads it")
  if schema == .str MANIFEST_RECORDED && !recorded then
    coin s!"the pass's manifest is {MANIFEST_RECORDED}, recorded before epoch coins: only a replay of a recorded pass reads it"
  if schema != .str MANIFEST_SCHEMA && schema != .str MANIFEST_RECORDED then
    let .str s := schema | manifest "the pass's manifest names no schema"
    coin (s!"schema {pyRepr s}, not {MANIFEST_SCHEMA}" ++ if recorded then s!" or {MANIFEST_RECORDED}" else "")
  let (us, roots) ← workloadOf m
  let (salt, how) ← if schema == .str MANIFEST_SCHEMA then do
      if (PouwWindow.field? m "salt_opening").isNone then coin "the pass has no salt coin"
      if at? (at? m "coins") "scheme" == .str PouwWindow.LIVE_SCHEME then livePassSalt m live
      else do
        if coins == .null then
          coin "the pass's salt coin opens against the epoch's published coin commitment, which the verifier holds"
        coinPassSalt m coins recorded us roots
    else if (PouwWindow.field? m "beacon").isSome then do
      let some beacon := (hex? (at? m "beacon")).filter (·.size > 0)
        | manifest "the recorded pass's beacon is not nonempty hex"
      checkWorkload us roots
      let k ← port (PouwCommit.key? beacon PouwCommit.SALT_DOMAIN
        [("weights", .bytes (weightsRootAt 64 Domains.POUW_WEIGHTS_ROOT roots))])
      pure (FlockSoundness.Randomness.stream k [.str "salt"] 32, "recorded beacon")
    else coin "the recorded pass has no beacon"
  if at? m "salt" != .str (Flock.hexOf salt) then coin s!"the pass's salt is not its {how}'s and the weights'"
  pure (salt, how)

/-- `core_check.pass_epoch`: how the pass's salt was had, or the refusal. A live Pearl-C pass, whose manifest's coins are
a live session's, is read only through the verifier's `live` inputs (`livePassSalt`), never `coins`. A certified
window's verifier holds `reveal`, `committed`, what PoUW recorded with the certifier before it drew that salt, and the
window's `stream` and `prover`. -/
def passEpoch (coins : Json) (raw : Option ByteArray) (context : String) (recorded : Bool)
    (reveal committed stream prover : Option ByteArray) (live : Json) : R String := do
  let some raw := raw | coin "the window names no pass, whose epoch the challenger re-starts from its manifest"
  let m ← match String.fromUTF8? raw with
    | some s => match Json.parse s with
      | .ok j@(.obj _) => pure j
      | _ => manifest "the pass's manifest is not a JSON object"
    | none => manifest "the pass's manifest is not a JSON object"
  let (_, how) ← epochSalt m coins recorded reveal context committed stream prover live
  if Flock.hexOf (Flock.Sha512.hash raw) != ((context.splitOn "/").getLast?.getD "") then
    manifest "the pass is not the one the window's parameters name"
  pure how

def plain (s : String) : Bool := s.all fun c => ' ' ≤ c && c ≤ '~' && c != '"' && c != '\\'

/-- `draw_key_context`'s receipt: the SHA-512 of its canonical JSON, an object of plain strings. -/
def receiptDigest (kvs : List (String × String)) : String :=
  let sorted := kvs.mergeSort fun a b => a.1 ≤ b.1
  let body := "{" ++ ",".intercalate (sorted.map fun (k, v) => s!"\"{k}\":\"{v}\"") ++ "}"
  Flock.hexOf (Flock.Sha512.hash body.toUTF8)

/-- `core_check.SIGNED_LIVE`. -/
def SIGNED_LIVE : String :=
  "a live window's coins are its challenger's signed record and the verifier's key for it, {key, record} " ++
    "(`challenger.trusted_commitment`), not a bare commitment"

/-- `core_check.NO_PINNED_KEY`, `OTHER_KEY` and `SEEDED_PINNED`: the reasons of a `coin-key` refusal. -/
def NO_PINNED_KEY : String := "the window's parameters pin no challenger key for its draw's record"
def OTHER_KEY : String :=
  "the challenger record is signed under a key other than the challenger key the window's parameters pin"
def SEEDED_PINNED : String :=
  "the window's parameters pin a challenger key, so its draw's coins are that challenger's signed record, " ++
    "not seeded coins"

/-- `core_check.draw`: none when `units` are the window's draw, else the refusal. Seeded coins (`coinsOf`) have their
`draw` coin, and are refused (`coin-key`) when the window's parameters pin a challenger key; live ones
(`PouwWindow.LIVE_SCHEME`, a certified window's) are read only from the challenger's signed record with the key it is
signed under, `{key, record}`: that key must be `key`, the challenger key the window's parameters pin (`coin-key`, so a
record the prover signed under a key of its own, over coins it holds, is refused), this check verifies the record under
it (`SampledProofs.Challenger.recordReason`; a bare live commitment is refused, `SIGNED_LIVE`), and they have their
`pouw/draw` coin 0 (`SampledProofs.Challenger.liveReason`). -/
def drawCheck (coins : Json) (recorded : Bool) (opening stream registration receipt k population units key : Json) :
    R Unit := do
  let liveScheme := match PouwWindow.field? coins "scheme" with
    | some (.str s) => s == PouwWindow.LIVE_SCHEME
    | _ => false
  let live := (PouwWindow.field? coins "record").isSome || liveScheme
  let malformed := fun (e : String) => ("coin", s!"the verifier's coins are malformed: {e}")
  let c : Coins ⊕ LiveCommitment ← if live then do
      unless SampledProofs.keys? coins == some #["key", "record"] do coin SIGNED_LIVE
      let want ← (SampledProofs.Challenger.ChallengerKey.ofJson key).mapError
        fun e => ("coin-key", s!"{NO_PINNED_KEY}: {e}")
      let got ← (SampledProofs.Challenger.ChallengerKey.ofJson (SampledProofs.getD coins "key")).mapError
        fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
      unless got.toJson.compress == want.toJson.compress do throw ("coin-key", OTHER_KEY)
      let record := SampledProofs.getD coins "record"
      if let some why ← port (SampledProofs.Challenger.recordReason want record) then coin why
      let c := (← port (SampledProofs.Challenger.ChallengerRecord.ofJson record)).commitment
      if let some why := PouwWindow.liveEpochReason c then coin why
      pure (.inr c)
    else match key with
      | .null => .inl <$> (coinsOf coins recorded).mapError malformed
      | _ => throw ("coin-key", SEEDED_PINNED)
  let (.str reg, .str st) := (registration, stream)
    | throw ("draw", "the registration is not a SHA-512 digest, or the stream's digest not a string")
  if !isHexDigest reg then throw ("draw", "the registration is not a SHA-512 digest, or the stream's digest not a string")
  let plainPair : String × Json → Option (String × String)
    | (x, Json.str s) => if plain x && plain s then some (x, s) else none
    | _ => none
  let some kvs := (match receipt with
      | .obj kv => kv.toArray.toList.mapM plainPair
      | _ => none) | throw ("draw", "the receipt is not an object of printable ASCII strings")
  let (some k, some n) := (PouwWindow.natOf? k, PouwWindow.natOf? population)
    | throw ("draw", "the law is not subset:k over N with k <= N and N >= 1")
  if !(1 ≤ n && k ≤ n) then throw ("draw", "the law is not subset:k over N with k <= N and N >= 1")
  let opened := fun (e : String) => ("coin", s!"the draw's {e}")
  let after ← port (pyFromHex reg)
  let (why, old, drawn) ← match c with
    | .inl c => do
        let o ← (openingOf opening).mapError opened
        let (why, old) ← port (coinVersionAt c o "draw" after recorded)
        pure (why, old, o.coin)
    | .inr c => do
        let o ← (LiveOpening.ofJson opening).mapError opened
        pure ((← port (SampledProofs.Challenger.liveReason c o PouwWindow.DRAW 0 after)), false, o.coin)
  if let some why := why then coin s!"the draw's {why}"
  let ctx := [("stream", FlockSoundness.Randomness.Part.str st), ("registration", FlockSoundness.Randomness.Part.str reg),
    ("receipt", FlockSoundness.Randomness.Part.str (receiptDigest kvs))]
  let bytes := 2 * k * Flock.Draw.width n + 64
  let src ← if old then pure (streamV1 (deriveV1 drawn DRAW_DOMAIN ctx) [.str "draw"] bytes)
    else do pure (FlockSoundness.Randomness.stream (← port (PouwCommit.key? drawn DRAW_DOMAIN ctx)) [.str "draw"] bytes)
  let some got := Flock.Draw.subset k n src
    | throw ("sampler", "the draw's stream ran out")
  let same := match units with
    | .arr us => us.toList.mapM PouwWindow.natOf? == some got.toList
    | _ => false
  if !same then throw ("draw", s!"the units are not the draw coin's subset:{k} over {n}")

/-- `core_check.work_units`: `drawCheck`'s rule under the law `work:k`, the units the draw coin `coin` draws over the
work strata `strata`, ascending. Stratum s is `Flock.Draw.subset` of its `k_s` over its units on the stream
`["draw", template]` of the key the coin derives under the stream's digest, the registration and the receipt's
(`receiptDigest`), its local index j its j-th unit (`Flock.Draw.stratified`); `none` when a stream runs out. -/
def workUnits (coin : ByteArray) (stream registration : String) (receipt : List (String × String))
    (strata : Array Flock.Draw.Stratum) : Except String (Option (Array Nat)) := do
  let ctx := [("stream", FlockSoundness.Randomness.Part.str stream),
    ("registration", FlockSoundness.Randomness.Part.str registration),
    ("receipt", FlockSoundness.Randomness.Part.str (receiptDigest receipt))]
  let key ← PouwCommit.key? coin DRAW_DOMAIN ctx
  pure (Flock.Draw.stratified strata (strata.map fun s =>
    FlockSoundness.Randomness.stream key [.str "draw", .str s.template] s.need))

/-! Each separator below keeps its literal while a lock reads its definition, whose record hashes the value as
written: tied here to its entry of the table (`Identity.Domains`), it reads the entry at its next record change. -/
example : MANIFEST_SCHEMA = Domains.VLLM_PEARL_C_RUN := rfl
example : WINDOW_OPEN_TAG = Domains.POUW_PERIOD_OPEN.toUTF8 := rfl
example : EPOCH_OPEN_TAG = Domains.POUW_EPOCH_OPEN.toUTF8 := rfl
example : DRAW_DOMAIN = Domains.POUW_SERVED_ZK_SAMPLING := rfl
example : ("verity/pouw/weights/v1" : String) = Domains.POUW_WEIGHTS_ROOT := rfl

end PouwCheck
