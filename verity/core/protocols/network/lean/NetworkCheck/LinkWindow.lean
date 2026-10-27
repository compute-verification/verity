import NetworkCheck.Certificate
import Lms.Verify
import Flock.Hm96
import Flock.Canon
import Identity.Tagged

/-!
# N2's link-window verdict of record

`verdict` decides one link-window of N2 (`verity/core/protocols/certifier/service.py`, `LinkWindows.verify`) from the
proof as the verifier receives it and the link's public fields, and nothing else: it is accepted exactly when

* the verifier's own run of C-Flock's Lean verifier (`proofs.verify`, whose findings `Case.findings` carries) refused
  nothing, every gate of the link's unit names a pinned circuit digest (`pinned`), one or more sessions each accepted
  and named the registered Program, and each proved exactly the drawn units;
* the statement's public file holds, for the drawn unit, a ``pub`` row commitment that opens under the delivered salt to
  the verifier's own public row (`pubRow`: the registered root, the link-window's served domain (`domainId`), the link,
  the window, each end's declared grid and its profile's row, both certificates' signed digests, and on a served link the
  window's declared sync, on an advised link the window's view of the advice) and an ``o0`` row commitment that opens to
  the delivered output word, whose verdict and deadline bits are both 1 (`opened`);
* both certificates verify under the network certifier's LMS key in its session over their public fields, and name the
  link, their end (egress first) and the window (`certified`);
* the window's declared sync is one of Σ_sync.

A warden link (`Kind.warden`, `service.WardenProfile`) is the plain unit under the active warden's records: both ends'
grids are the warden's one grid, with no profile, and both carry the warden's parameters digest (`gridBytes`) in place
of a certifier's; the warden runs no shaper clock, so a window declares no sync and nothing is read of the syncs
(`synced`). Its outputs are a plain link's.

Its public outputs are `verdict` and the proved `deadline` bit (on an advised link the advice it read, as given), and
`digest` binds them to the registration, the sessions and the accept bit (`verdictDigest`). An advised link's window view
is an input here (`Case.view`), the Python's `advised.view`: that derivation is not ported, so an advised link's verdict
of record stays the Python's (`service.OF_RECORD`).

`linkWindowFn` is the `link-window` function of `network-records verdicts` (`Difftest.serve`), which `service.py` calls
through `lean_audit.SERVER`. Code: it imports only Lean, C-Flock's verifier (`Flock.Hash`, `Flock.Bytes`, `Flock.Canon`,
`Flock.Hm96`), the identity hash, the LMS port and the certificate's encodings.
-/

namespace NetworkCheck.LinkWindow

open Lean (Json toJson)
open Flock (V)

def verdictFormat : String := "verity-network-certifier/verdict/v0"
/-- `verity.ml.commitments.window.WINDOW_DOMAIN`, one-stage's served domain. -/
def windowDomainTag : String := "verity/one-stage/served-domain/v1"
def value : String := "certificates"
def schema : String := "hm96-sha512/row/v2"
def leaves : Nat := 4
/-- The output row: one u16 word, bit 0 the verdict, bit 1 the deadline. -/
def outBits : Nat := 16
def saltBytes : Nat := 192
/-- A row commitment in a public file, `b ‖ c`. -/
def bcBytes : Nat := 128
/-- Each instance's row commitments, in this order: ``pub``, ``wit``, ``o0``. -/
def ports : List String := ["pub", "wit", "o0"]

/-! ## The case -/

/-- An end's declared grid (`grid.GridParams`) and its occupancy profile, a row of T counts a window, or none. -/
structure End where
  grid : NetworkCheck.Params
  profile : Option (Array (Array Nat))

/-- A certificate's public fields (`certificate.Certificate`), direction by its code. The public row reads each
certificate's signed digest from `Case.digests`, as the Python reads it from any object (`public_arguments`). -/
structure Cert where
  keyId : ByteArray
  link : String
  direction : Direction
  window : Nat
  digest : ByteArray
  signature : ByteArray

/-- One session's finding from the verifier's own Lean run (`proofs.Finding`). -/
structure Finding where
  accepted : Bool
  program : Option String
  units : Array Nat

/-- A declared clock sync: δ and ρ = num/den, both at least 1. -/
structure Sync where
  delta : Nat
  num : Nat
  den : Nat

/-- An advised window's view of the advice (`advised.WindowView`). -/
structure View where
  released : Array Nat
  floor : Array Nat
  received : Nat

inductive Kind where
  | plain
  | served
  /-- the window's view (none where `advised.view` refuses) and the advice output (`advised.digest`, none without one) -/
  | advised (view : Option View) (advice : Option String)
  /-- the warden's parameters digest (`service.WardenProfile.digest`), which both ends' heads bind -/
  | warden (digest : ByteArray)

structure Case where
  link : String
  run : String
  window : Nat
  root : ByteArray
  registration : String
  units : Array Nat
  program : String
  partition : String
  gates : Array String
  findings : Option (Array Finding)
  /-- the statement's public file -/
  statement : Option ByteArray
  saltPub : Option ByteArray
  saltOut : Option ByteArray
  output : Option Nat
  sessions : Array String
  certificates : Array (Option Cert)
  /-- each certificate's signed digest as the public row reads it, whatever its other fields (empty for none) -/
  digests : Array ByteArray
  syncs : Array Sync
  deltas : Array Nat
  rhos : Array (Nat × Nat)
  lms : Lms.Params
  key : ByteArray
  session : ByteArray
  egress : End
  ingress : End
  kind : Kind

/-! ## Decoding -/

section Decode

open Flock.J

private def opt (j : Json) (k : String) : Option Json :=
  match j.getObjVal? k with
  | .ok .null => none
  | .ok v => some v
  | .error _ => none

private def bytesOf (j : Json) (ctx : String) : V ByteArray := do Flock.unhex (← asStr j ctx)

private def natsOf (j : Json) (ctx : String) : V (Array Nat) := do (← asArr j ctx).mapM (asNat · ctx)

def gridOf (j : Json) : V NetworkCheck.Params := do
  let field (k : String) : V Nat := do
    let v ← nat j k "grid"
    unless 1 ≤ v ∧ v < 2 ^ 32 do throw s!"grid.{k} is a u32 of at least 1, not {v}"
    pure v
  pure ⟨← field "buckets", ← field "allocation", ← field "frame_bytes", ← field "bucket_ms", ← nat j "queue_capacity" "grid"⟩

def endOf (j : Json) (ctx : String) : V End := do
  let grid ← gridOf (← get j "grid" ctx)
  let profile ← (opt j "profile").mapM fun p => do (← asArr p s!"{ctx}.profile").mapM (natsOf · s!"{ctx}.profile")
  pure ⟨grid, profile⟩

private def certOf (j : Json) : V (Option Cert) := do
  if j.isNull then return none
  let direction ← match ← nat j "direction" "certificate" with
    | 0 => pure Direction.egress
    | 1 => pure Direction.ingress
    | c => throw s!"a direction code is 0 or 1, not {c}"
  pure (some { keyId := ← bytesOf (← get j "key_id" "certificate") "key_id", link := ← str j "link" "certificate",
               direction, window := ← nat j "window" "certificate",
               digest := ← bytesOf (← get j "digest" "certificate") "digest",
               signature := ← bytesOf (← get j "signature" "certificate") "signature" })

private def findingOf (j : Json) : V Finding := do
  pure { accepted := ← bool j "accepted" "finding",
         program := ← (opt j "program").mapM (asStr · "finding.program"),
         units := ← natsOf (← get j "units" "finding") "finding.units" }

private def ratOf (j : Json) (ctx : String) : V (Nat × Nat) := do
  let #[n, d] ← asArr j ctx | throw s!"{ctx}: not [numerator, denominator]"
  let n ← asNat n ctx
  let d ← asNat d ctx
  unless 1 ≤ n ∧ 1 ≤ d do throw s!"{ctx}: a numerator and a denominator of at least 1"
  pure (n, d)

private def syncOf (j : Json) : V Sync := do
  let #[d, n, m] ← asArr j "a sync" | throw "a sync is [delta, num, den]"
  let (n, m) ← ratOf (.arr #[n, m]) "a sync's rho"
  pure ⟨← asNat d "a sync's delta", n, m⟩

private def viewOf (j : Json) : V View := do
  pure { released := ← natsOf (← get j "released" "view") "view.released",
         floor := ← natsOf (← get j "floor" "view") "view.floor", received := ← nat j "received" "view" }

def caseOf (j : Json) : V Case := do
  let kind ← match ← str j "kind" "case" with
    | "plain" => pure Kind.plain
    | "served" => pure Kind.served
    | "advised" => pure (Kind.advised (← (opt j "view").mapM viewOf) (← (opt j "advice").mapM (asStr · "advice")))
    | "warden" => do
      let d ← bytesOf (← get j "warden" "case") "warden"
      unless d.size = digestBytes do throw s!"a warden's parameters digest is {digestBytes} bytes, not {d.size}"
      pure (Kind.warden d)
    | k => throw s!"a link is plain, served, advised or a warden's, not {k}"
  let ss ← get j "sync_set" "case"
  let deltas ← natsOf (← get ss "deltas" "sync_set") "sync_set.deltas"
  let rhos ← (← arr ss "rhos" "sync_set").mapM (ratOf · "sync_set.rhos")
  unless deltas.size ≥ 1 ∧ rhos.size ≥ 1 do throw "sync_set: non-empty"
  let l ← get j "lms" "case"
  let lms ← Lms.Params.make (← nat l "N" "lms") (← nat l "W" "lms") (← nat l "H" "lms") (← bool l "toy" "lms")
  let key ← bytesOf (← get j "key" "case") "key"
  unless key.size = 16 + lms.N do throw s!"an LMS public key is {16 + lms.N} bytes, not {key.size}"
  let salts ← get j "salts" "case"
  let egress ← endOf (← get j "egress" "case") "egress"
  let ingress ← endOf (← get j "ingress" "case") "ingress"
  if let .warden _ := kind then
    let fields (g : NetworkCheck.Params) := (g.buckets, g.allocation, g.frameBytes, g.bucketMs, g.queueCapacity)
    unless egress.profile.isNone && ingress.profile.isNone && fields egress.grid == fields ingress.grid do
      throw "a warden link's two ends are on the warden's one grid, with no occupancy profile"
  pure { link := ← str j "link" "case", run := ← str j "run" "case", window := ← nat j "window" "case",
         root := ← bytesOf (← get j "root" "case") "root", registration := ← str j "registration" "case",
         units := ← natsOf (← get j "units" "case") "units", program := ← str j "program" "case",
         partition := ← str j "partition" "case",
         gates := ← (← arr j "gates" "case").mapM (asStr · "gates"),
         findings := ← (opt j "findings").mapM fun f => do (← asArr f "findings").mapM findingOf,
         statement := ← (opt j "public").mapM (bytesOf · "public"),
         saltPub := ← (opt salts "pub").mapM (bytesOf · "salts.pub"),
         saltOut := ← (opt salts "o0").mapM (bytesOf · "salts.o0"),
         output := ← (opt j "output").mapM (asNat · "output"),
         sessions := ← (← arr j "sessions" "case").mapM (asStr · "sessions"),
         certificates := ← (← arr j "certificates" "case").mapM certOf,
         digests := ← (← arr j "digests" "case").mapM (bytesOf · "digests"),
         syncs := ← (← arr j "syncs" "case").mapM syncOf, deltas, rhos, lms, key,
         session := ← bytesOf (← get j "session" "case") "session", egress, ingress, kind }

end Decode

/-! ## The public row -/

/-- A 64-byte digest as the unit reads it (`head_check._digest_bits`, packed): each 8-byte big-endian word's bits least
significant first, so each word's bytes reversed. -/
def digestRow (d : ByteArray) : ByteArray :=
  ⟨(Array.range d.size).map fun i => d.get! (8 * (i / 8) + 7 - i % 8)⟩

/-- `x` as `width` little-endian bytes (`head_check._int_bits`, packed). -/
def le (width x : Nat) : ByteArray := ⟨(Array.range width).map fun i => (x >>> (8 * i)).toUInt8⟩

/-- `x` as `width` big-endian bytes, `x < 2^(8·width)`. -/
def be (width x : Nat) : ByteArray := ⟨(Array.range width).map fun i => (x >>> (8 * (width - 1 - i))).toUInt8⟩

/-- frame-v3's framed hash through SHA-512 (`verity.ml.commitments.merkle._message`): the frame tag, the tag, then each
part length-prefixed. -/
def frameHash (tag : String) (parts : List ByteArray) : ByteArray :=
  let t := tag.toUTF8
  Flock.Sha512.hash (parts.foldl (fun acc p => acc ++ Flock.be64 p.size.toUInt64 ++ p)
    ("veritor/protocol/merkle/frame/v3".toUTF8.push 0 ++ Flock.be32 t.size.toUInt32 ++ t))

/-- `merkle._uint`: the minimal big-endian bytes, one zero byte for 0. -/
def uint (x : Nat) : ByteArray :=
  if x = 0 then ⟨#[0]⟩ else be ((Nat.log2 x) / 8 + 1) x

/-- The link-window's served domain id (`LinkWindows.domain(window).domain_id`): one-stage's window domain over four
leaves, its binding the identity digest of the registered Program, the partition, the value, its schema and leaf count,
and the run `<run>/<link>/<window>`; the boundary's owner, positions `RangeIndexedDomain(0, 4)`. -/
def domainId (c : Case) : V ByteArray := do
  let binding ← Identity.identityDigestSha512 windowDomainTag (Json.mkObj [
    ("program", .str c.program), ("partition", .str c.partition), ("port", .str value), ("schema", .str schema),
    ("leaves", toJson leaves), ("run", .str s!"{c.run}/{c.link}/{c.window}")])
  let positions ← Identity.identityDigestSha512 "veritor/indexed-domain/range/v1"
    (Json.mkObj [("start", toJson (0 : Nat)), ("step", toJson (1 : Nat)), ("stop", toJson leaves)])
  return frameHash "domain" [binding, uint 1, positions, uint leaves]

/-- Σ_sync as `parameters_digest` reads it, each ρ in lowest terms. -/
def manifestSet (c : Case) : NetworkCheck.SyncSet :=
  ⟨c.deltas.map Int.ofNat, c.rhos.map fun (n, d) => ((n / Nat.gcd n d : Nat), d / Nat.gcd n d)⟩

/-- An end's grid as its head encodes it (`head_check.grid_bytes`): T, r, B and τ_b as u32 big-endian, then the grid's
parameters digest, which binds Σ_sync on egress; on a warden link, both ends' the warden's (`LinkWindows.grid`). -/
def gridBytes (c : Case) (d : Nat) : V ByteArray := do
  let e := if d = 0 then c.egress else c.ingress
  let g := e.grid
  let digest ← match c.kind with
    | .warden w => pure w
    | _ => parametersDigest g (if d = 0 then some (manifestSet c) else none)
  return be 4 g.buckets ++ be 4 g.allocation ++ be 4 g.frameBytes ++ be 4 g.bucketMs ++ digest

/-- The served unit's `sync` input (`served.sync_words`): δ, then ρ's numerator and denominator in lowest terms, each a
u16; none past 2^16. -/
def syncWords (s : Sync) : Option (Nat × Nat × Nat) :=
  let g := Nat.gcd s.num s.den
  let w := (s.delta, s.num / g, s.den / g)
  if w.1 < 2 ^ 16 ∧ w.2.1 < 2 ^ 16 ∧ w.2.2 < 2 ^ 16 then some w else none

/-- An advised count as the unit reads it (`advised._count_bits`, packed): four bytes big-endian. -/
def countRow (xs : Array Nat) : Option ByteArray :=
  if xs.all (· < 2 ^ 32) then some (xs.foldl (fun acc x => acc ++ be 4 x) ByteArray.empty) else none

/-- The unit's public row as the verifier derives it from public fields alone (`LinkWindows.public_arguments`, then
`LinkWindows.row`): both certificates' signed digests, then each end's registered root, served domain, link, window,
grid and (with a profile) its row for the window, then on a served link the window's declared sync and on an advised
link its view. None where the unit takes no instance: not two certificates, a signed digest of another length, a window
past an end's profile, no sync a served unit can hold, or no view. -/
def pubRow (c : Case) : V (Option ByteArray) := do
  let #[d0, d1] := c.digests | return none
  let digests := [d0, d1]
  if c.certificates.size ≠ 2 ∨ digests.any (·.size ≠ 64) then return none
  let tail : Option ByteArray ← match c.kind with
    | .plain | .warden _ => pure (some ByteArray.empty)
    | .served => pure do
        let s ← c.syncs[c.window]?
        let (d, p, q) ← syncWords s
        pure (le 2 d ++ le 2 p ++ le 2 q)
    | .advised view _ => pure do
        let v ← view
        pure ((← countRow v.released) ++ (← countRow v.floor) ++ (← countRow #[v.received]))
  let some tail := tail | return none
  let dom ← domainId c
  let link := c.link.toUTF8
  let mut row := digestRow d0 ++ digestRow d1
  for d in [0, 1] do
    let e := if d = 0 then c.egress else c.ingress
    let profile ← match e.profile with
      | none => pure ByteArray.empty
      | some p => match p[c.window]? with
        | none => return none
        | some r => pure (r.foldl (fun acc n => acc ++ le 4 n) ByteArray.empty)
    row := row ++ digestRow c.root ++ digestRow dom ++ link ++ le 8 c.window ++ (← gridBytes c d) ++ profile
  return some (row ++ tail)

/-! ## The proof -/

/-- Whether a gate id names a pinned circuit digest (`head_check.pinned`): `Gate[<definition>@<128 lowercase hex>]_v2`,
the definition non-empty and on one line. -/
def pinned (g : String) : Bool :=
  let cs := g.toList
  let hex (ch : Char) : Bool := ch.isDigit || ('a' ≤ ch && ch ≤ 'f')
  if g.startsWith "Gate[" && g.endsWith "]_v2" && cs.length ≥ 5 + 1 + 1 + 128 + 4 then
    let inner := (cs.drop 5).take (cs.length - 9)
    let defn := inner.take (inner.length - 129)
    let rest := inner.drop (inner.length - 129)
    !defn.isEmpty && !defn.contains '\n' && rest.head? == some '@' && (rest.drop 1).all hex
  else false

/-- Whether the findings prove this link's unit and the draw (`LinkWindows.proved`, then each session's units): nothing
refused, every gate pinned, one or more sessions, each accepted, naming the registered Program and proving exactly the
drawn units. -/
def proved (c : Case) : Bool :=
  match c.findings with
  | none => false
  | some fs => c.gates.all pinned && !fs.isEmpty &&
      fs.all fun f => f.accepted && f.program == some c.program && f.units == c.units

/-- The row commitments the public file holds for its one instance whose unit index is `unit`, by port
(`service._commitments`): the header line's `units.indices` names it once; then the instances' commitments, three
each, instance-major, or in a shared-row file (`shared_rows`, one table a port, each at least one row) each port's table
then every instance's refs (u32 little-endian, ports in order). None when no one instance is, or the file is short. -/
def commitments (file : ByteArray) (unit : Nat) : Option (List ByteArray) := do
  let nl ← (List.range file.size).find? fun i => file.get! i == 10
  let line ← String.fromUTF8? (file.extract 0 nl)
  let head ← (Json.parse line).toOption
  let indices ← ((head.getObjVal? "units").toOption.bind fun u => (u.getObjVal? "indices").toOption).bind
    fun x => x.getArr?.toOption
  let matches_ := indices.toList.map fun x => x.getNat?.toOption == some unit
  if matches_.count true ≠ 1 then none
  let i ← matches_.findIdx? id
  let body := file.extract (nl + 1) file.size
  let k := ports.length
  let cut (b : ByteArray) (j : Nat) : ByteArray := b.extract (j * bcBytes) ((j + 1) * bcBytes)
  match (head.getObjVal? "shared_rows").toOption.filter (!·.isNull) with
  | none =>
    let data := body.extract (i * k * bcBytes) ((i + 1) * k * bcBytes)
    if data.size ≠ k * bcBytes then none
    pure ((List.range k).map (cut data))
  | some shared =>
    let obj ← shared.getObj?.toOption
    let keys := obj.toArray.toList.map (·.1)
    unless keys.length = k ∧ ports.all keys.contains do none
    let counts ← ports.mapM fun p => (shared.getObjValAs? Nat p).toOption
    unless counts.all (1 ≤ ·) do none
    let total := counts.foldl (· + ·) 0
    let need := total * bcBytes + indices.size * k * 4
    if body.size < need then none
    let refs := body.extract (total * bcBytes + i * k * 4) (total * bcBytes + (i + 1) * k * 4)
    let (out, _) ← (List.range k).foldlM (init := ([], 0)) fun (acc, base) j => do
      let r := (List.range 4).foldl (fun a b => a + (refs.get! (4 * j + b)).toNat * 256 ^ b) 0
      let n := counts[j]!
      if r ≥ n then none
      pure (acc ++ [cut body (base + r)], base + n)
    pure out

/-- `sha512/row/v2`'s digest of a row of `bits` bits (`rowleaf.sha512_row_v2_digest`): SHA-512 of its first block
`"verity/sha512-row/v2\0" ‖ u64be bits`, zero-padded to 128 bytes, then the row's bytes. -/
def rowDigest (row : ByteArray) (bits : Nat) : ByteArray :=
  let p := "verity/sha512-row/v2".toUTF8.push 0 ++ be 8 bits
  Flock.Sha512.hash (p ++ ByteArray.mk (Array.replicate (128 - p.size) 0) ++ row)

/-- Whether `bc` commits to the `bits`-bit row under `salt` (`service._opens`): hm96-sha512 under its default key over
the row's `sha512/row/v2` digest, the salt 192 bytes. -/
def opens (row : ByteArray) (bits : Nat) (salt : Option ByteArray) (bc : ByteArray) : Bool :=
  match salt with
  | some s => s.size == saltBytes && Flock.Hm96.Default512.commitString (rowDigest row bits) s == bc
  | none => false

/-- The drawn unit's output word, once its public row commitment opens to `row` and its output row commitment to the
delivered word, each under the salt delivered (`service._opened`); else none. -/
def opened (c : Case) (row : Option ByteArray) : Option Nat := do
  let row ← row
  let file ← c.statement
  let unit ← c.units[0]?
  let bcs ← commitments file unit
  let word ← c.output
  if word ≥ 4 then none
  let [bcPub, _, bcOut] := bcs | none
  if opens row (8 * row.size) c.saltPub bcPub && opens (le 2 word) outBits c.saltOut bcOut then some word else none

/-! ## The certificates and the sync -/

/-- `Certificate.verify` under the link's LMS key in its certifier session: the key's identifier, and the signature
verifying over the certificate's message (`certificateMessage`); a message or signature that doesn't decode is false. -/
def verifies (c : Case) (x : Cert) : Bool :=
  x.keyId == c.key.extract 0 16 &&
    match certificateMessage x.keyId c.session x.link x.direction x.window x.digest,
        Lms.Signature.decode c.lms x.signature with
    | .ok m, .ok sig => (Lms.verify c.lms c.key m sig).toOption == some true
    | _, _ => false

/-- The certificates' check (`LinkWindows.certified`): two, each naming this link, its end (egress first) and the window,
and verifying. -/
def certified (c : Case) : Bool :=
  c.certificates.size == 2 && (List.range 2).all fun d =>
    match c.certificates[d]! with
    | some x => x.link == c.link && x.direction.code == d && x.window == c.window && verifies c x
    | none => false

/-- The window's declared sync is one of Σ_sync: its δ one of the deltas and its ρ one of the rhos. A warden link
declares none (the warden runs no shaper clock), so nothing is read of the syncs there. -/
def synced (c : Case) : Bool :=
  match c.kind with
  | .warden _ => true
  | _ => match c.syncs[c.window]? with
    | some s => c.deltas.contains s.delta && c.rhos.any fun (n, d) => s.num * d == n * s.den
    | none => false

/-! ## The verdict -/

structure Verdict where
  accepted : Bool
  /-- the verdict's public outputs -/
  outputs : Json
  digest : String
  proved : Bool
  opened : Option Nat
  certified : Bool
  synced : Bool

/-- `SHA-512(canonical_json({format, registration, sessions, accepted, public}))`, hex. -/
def verdictDigest (c : Case) (accepted : Bool) (outputs : Json) : V String := do
  let body := Json.mkObj [("format", .str verdictFormat), ("registration", .str c.registration),
    ("sessions", .arr (c.sessions.map Json.str)), ("accepted", .bool accepted), ("public", outputs)]
  return Flock.hexOf (Flock.Sha512.hash (← Flock.canon body).toUTF8)

/-- The verdict's public outputs: the accept bit as `verdict`, the proved word's deadline bit (0 where nothing opened),
and on an advised link the advice it read. -/
def outputsOf (c : Case) (accepted : Bool) (word : Nat) : Json :=
  Json.mkObj ([("verdict", toJson (if accepted then 1 else 0 : Nat)), ("deadline", toJson (word / 2 % 2))] ++
    match c.kind with
    | .advised _ advice => [("advice", match advice with | some a => .str a | none => .null)]
    | _ => [])

/-- Whether the link-window is accepted, from its checks: proved, its rows opened to the word 3 (both bits 1), both
certificates verifying, and the window's sync one of Σ_sync. -/
def acceptedOf (p : Bool) (o : Option Nat) (cert sync : Bool) : Bool :=
  p && o == some 3 && cert && sync

/-- The verdict of record on the link-window (the module docstring). -/
def verdict (c : Case) : V Verdict := do
  let p := proved c
  let row ← pubRow c
  let o := if p then opened c row else none
  let cert := certified c
  let sync := synced c
  let accepted := acceptedOf p o cert sync
  let outputs := outputsOf c accepted (o.getD 0)
  return { accepted, outputs, digest := ← verdictDigest c accepted outputs, proved := p, opened := o, certified := cert,
           synced := sync }

def verdictJson (v : Verdict) : Json :=
  Json.mkObj [("accepted", .bool v.accepted), ("public", v.outputs), ("digest", .str v.digest),
    ("checks", Json.mkObj [("proved", .bool v.proved), ("opened", match v.opened with | some w => toJson w | none => .null),
      ("certified", .bool v.certified), ("synced", .bool v.synced)])]

/-- `network-records verdicts`' `link-window`: the verdict on a case, or its refusal of a case it can't read. -/
def linkWindowFn (j : Json) : Except String Json := do
  return verdictJson (← verdict (← caseOf j))

end NetworkCheck.LinkWindow
