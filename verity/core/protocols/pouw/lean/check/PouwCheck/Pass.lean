import Flock.Draw
import PouwCommit.Draws
import PouwCheck.Inputs

/-!
# The pass's epoch and the window's draw (`core_check.py`, "the pass's epoch and the window's draw")

`passEpoch` is `core_check.pass_epoch`: the window's pass's manifest re-starts its epoch under the challenger's own coin
commitment, and the window's context names the pass by its SHA-512. A pass salted by epoch coins (`MANIFEST_SCHEMA`)
carries the epoch's commitment, which must be the challenger's, and its salt coin's opening, opened after the weights'
root (`coinReasonAt`); a pass recorded before epoch coins (`MANIFEST_RECORDED`, read only in a replay) carries a beacon.
Either way the manifest's salt must be the epoch's. A refusal is `coin` (the coin rule's, Python's `DomainError`) or
`manifest` (a manifest it can't read, or not the window's pass), with Python's words.

`drawCheck` is `core_check.draw`: the epoch's draw coin opens after the registration, and the drawn units are
`Flock.Draw.subset` of the stream `"draw"` of the key the coin derives under the stream, the registration and its
receipt.

Both read the coin versions of `coinVersion`: the current one, whose binding and weights' root are `tagged`'s whole
SHA-512, and, only in a replay of a recorded pass, the e2e-service passes', which keep its first 32 bytes. A replay of a
recorded draw also reads the e2e-service's first coin commitments, named as the current one but on SHA-256 as core was
then: bound by `taggedSha256`, after the current binding, and drawn by `deriveV1` and `streamV1`.
-/

namespace PouwCheck

open Lean (Json)
open PouwCommit (CoinOpening pyFromHex pyRepr tagged leafValue Domain)

def MANIFEST_SCHEMA : String := "verity-vllm/pouw-pearl-c-run/v2"
def MANIFEST_RECORDED : String := "verity-vllm/pouw-pearl-c-run/v1"
def COINS_RECORDED : String := "verity/pouw/epoch-coins/v2"
def DRAW_DOMAIN : String := "verity/pouw/served-zk-draw/v1"
def U64 : Int := 2 ^ 64
def OPENING_FORM : String :=
  "a coin opening is {epoch, label, after, coin, salt, path}: an integer, a string, hex, and SHA-512 digests"
def MATMULS_FORM : String := "the pass's matmuls are not a list of objects with integer index, m, k, n and weight"
def WEIGHTS_FORM : String := "the pass's weights are not an object from each weight, in decimal, to an object with a hex root"

/-- `COIN_VERSIONS`: the bytes of `tagged` a coin scheme's binding and weights' root keep, and the weights' root's tag. -/
def coinVersion (scheme : String) : Option (Nat × String) :=
  if scheme == PouwCommit.SCHEME then some (64, "verity/pouw/weights/v1")
  else if scheme == COINS_RECORDED then some (32, "verity/pouw/weights/v2")
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
  Flock.Sha256.hash ("verity/randomness/derive/v1".toUTF8.push 0 ++ PouwCommit.frame (.bytes source) ++
    PouwCommit.frame (.str domain) ++ PouwCommit.frame (.bytes ctx))

/-- `core_check.stream_v1`: `Key.stream` as it was (`verity/randomness/stream/v1`, SHA-256 blocks), `COIN_EARLY`'s draw
stream. -/
def streamV1 (key : ByteArray) (index : List PouwCommit.Part) (n : Nat) : ByteArray :=
  let head := "verity/randomness/stream/v1".toUTF8.push 0 ++ key ++ FlockSoundness.Randomness.frames index
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

/-- `core_check.weights_root`: `audit.weights_root` under a scheme's version. -/
def weightsRootAt (keep : Nat) (tag : String) (roots : List (Nat × ByteArray)) : ByteArray :=
  let sorted := roots.mergeSort fun a b => a.1 ≤ b.1
  (tagged tag (sorted.flatMap fun (w, r) => [.int w, .bytes r])).extract 0 keep

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

/-- `_epoch_salt`: the pass's salt re-started from its manifest, and how it was had. -/
def epochSalt (m : Json) (coins : Json) (recorded : Bool) : R (ByteArray × String) := do
  let schema := at? m "schema"
  if schema == .str MANIFEST_RECORDED && !recorded then
    coin s!"the pass's manifest is {MANIFEST_RECORDED}, recorded before epoch coins: only a replay of a recorded pass reads it"
  if schema != .str MANIFEST_SCHEMA && schema != .str MANIFEST_RECORDED then
    let .str s := schema | manifest "the pass's manifest names no schema"
    coin (s!"schema {pyRepr s}, not {MANIFEST_SCHEMA}" ++ if recorded then s!" or {MANIFEST_RECORDED}" else "")
  let (us, roots) ← workloadOf m
  let (salt, how) ← if schema == .str MANIFEST_SCHEMA then do
      if (PouwWindow.field? m "salt_opening").isNone then coin "the pass has no salt coin"
      if coins == .null then
        coin "the pass's salt coin opens against the epoch's published coin commitment, which the verifier holds"
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
    else if (PouwWindow.field? m "beacon").isSome then do
      let some beacon := (hex? (at? m "beacon")).filter (·.size > 0)
        | manifest "the recorded pass's beacon is not nonempty hex"
      checkWorkload us roots
      let k ← port (PouwCommit.key? beacon PouwCommit.SALT_DOMAIN
        [("weights", .bytes (weightsRootAt 64 "verity/pouw/weights/v1" roots))])
      pure (FlockSoundness.Randomness.stream k [.str "salt"] 32, "recorded beacon")
    else coin "the recorded pass has no beacon"
  if at? m "salt" != .str (Flock.hexOf salt) then coin s!"the pass's salt is not its {how}'s and the weights'"
  pure (salt, how)

/-- `core_check.pass_epoch`: how the pass's salt was had, or the refusal. -/
def passEpoch (coins : Json) (raw : Option ByteArray) (context : String) (recorded : Bool) : R String := do
  let some raw := raw | coin "the window names no pass, whose epoch the challenger re-starts from its manifest"
  let m ← match String.fromUTF8? raw with
    | some s => match Json.parse s with
      | .ok j@(.obj _) => pure j
      | _ => manifest "the pass's manifest is not a JSON object"
    | none => manifest "the pass's manifest is not a JSON object"
  let (_, how) ← epochSalt m coins recorded
  if Flock.hexOf (Flock.Sha512.hash raw) != ((context.splitOn "/").getLast?.getD "") then
    manifest "the pass is not the one the window's parameters name"
  pure how

def plain (s : String) : Bool := s.all fun c => ' ' ≤ c && c ≤ '~' && c != '"' && c != '\\'

/-- `draw_key_context`'s receipt: the SHA-512 of its canonical JSON, an object of plain strings. -/
def receiptDigest (kvs : List (String × String)) : String :=
  let sorted := kvs.mergeSort fun a b => a.1 ≤ b.1
  let body := "{" ++ ",".intercalate (sorted.map fun (k, v) => s!"\"{k}\":\"{v}\"") ++ "}"
  Flock.hexOf (Flock.Sha512.hash body.toUTF8)

/-- `core_check.draw`: none when `units` are the window's draw, else the refusal. -/
def drawCheck (coins : Json) (recorded : Bool) (opening stream registration receipt k population units : Json) :
    R Unit := do
  let c ← (coinsOf coins recorded).mapError fun e => ("coin", s!"the verifier's coins are malformed: {e}")
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
  let o ← (openingOf opening).mapError fun e => ("coin", s!"the draw's {e}")
  let after ← port (pyFromHex reg)
  let (why, old) ← port (coinVersionAt c o "draw" after recorded)
  if let some why := why then coin s!"the draw's {why}"
  let ctx := [("stream", FlockSoundness.Randomness.Part.str st), ("registration", FlockSoundness.Randomness.Part.str reg),
    ("receipt", FlockSoundness.Randomness.Part.str (receiptDigest kvs))]
  let bytes := 2 * k * Flock.Draw.width n + 64
  let src ← if old then pure (streamV1 (deriveV1 o.coin DRAW_DOMAIN ctx) [.str "draw"] bytes)
    else do pure (FlockSoundness.Randomness.stream (← port (PouwCommit.key? o.coin DRAW_DOMAIN ctx)) [.str "draw"] bytes)
  let some got := Flock.Draw.subset k n src
    | throw ("sampler", "the draw's stream ran out")
  let same := match units with
    | .arr us => us.toList.mapM PouwWindow.natOf? == some got.toList
    | _ => false
  if !same then throw ("draw", s!"the units are not the draw coin's subset:{k} over {n}")

end PouwCheck
