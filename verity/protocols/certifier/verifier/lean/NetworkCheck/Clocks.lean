module

public import NetworkCheck.Check

/-!
# The clock openings

`clocksFn`, `network-records verdicts`' `clocks`, reads the syncs the certifier signed for a link through its salted
clock certificates' openings, each sync and salt against the digest its certificate signs (`certificate.SignedClocks`,
`signedSyncs`). The certifier chain's record takes each window's sync from it, with `certificate.SignedClocks` beside
it as the reference (`lean_audit.signed_clocks`).

With them, the decoders and the clock digest they share with the period's check (`NetworkCheck.Period`), in its
namespace: `exactly`, `hexAt`, `certificateOf` and `clockDigest`. Code: it imports only the certificate check
(`NetworkCheck.Check`) and what that imports, and nothing of the inference-only policy's.
-/

@[expose] public section

namespace NetworkCheck.Period

open Lean (Json toJson)
open Flock (V)
open NetworkCertifier (ClockSync)
open NetworkCheck.Check (Certificate)

def clockTag : String := "verity-network-certifier/clock/v1"

def keysOf : Json → List String
  | .obj kvs => kvs.keys
  | _ => []

def exactly (j : Json) (want : List String) (ctx : String) : V Unit :=
  match j with
  | .obj _ => if keysOf j == want then pure () else throw s!"{ctx}: the keys must be {want}, not {keysOf j}"
  | _ => throw s!"{ctx}: not an object"

def certificateOf (j : Json) (ctx : String) : V Certificate := do
  exactly j ["digest", "direction", "key_id", "link", "signature", "window"] ctx
  let direction ← Direction.ofValue (← Flock.J.str j "direction" ctx)
  pure { keyId := ← Flock.unhex (← Flock.J.str j "key_id" ctx), link := ← Flock.J.str j "link" ctx, direction,
         window := ← Flock.J.nat j "window" ctx, digest := ← Flock.unhex (← Flock.J.str j "digest" ctx),
         signature := ← Flock.unhex (← Flock.J.str j "signature" ctx) }

def hexAt (j : Json) (k ctx : String) : V ByteArray := do Flock.unhex (← Flock.J.str j k ctx)

/-- `clock_digest`: `tagged_sha512("verity-network-certifier/clock/v1", u64be δ ‖ u64be p ‖ u64be q)` of the sync, ρ = p/q
in lowest terms; `none` past a u64. -/
def clockDigest (s : ClockSync) : Option ByteArray :=
  (do
    let d ← uBE 8 s.delta "δ"
    let n ← uBE 8 s.rho.num "p"
    let q ← uBE 8 s.rho.den "q"
    Identity.taggedSha512 clockTag (d ++ n ++ q)).toOption

/-! ## The clock openings -/

/-- A salted clock certificate's opening (`certificate.CertifiedClock`): the sync the certifier ran the window under and
its salt `salt_c`, both witness, and the certificate, which signs the sync's clock digest salted by it. -/
structure ClockOpening where
  sync : ClockSync
  salt : ByteArray
  certificate : Certificate

/-- Two openings are one: the same sync, salt and certificate, field by field. -/
def ClockOpening.same (a b : ClockOpening) : Bool :=
  let x := a.certificate
  let y := b.certificate
  a.sync.delta == b.sync.delta && a.sync.rho == b.sync.rho && a.salt == b.salt && x.keyId == y.keyId &&
    x.link == y.link && x.direction == y.direction && x.window == y.window && x.digest == y.digest &&
    x.signature == y.signature

/-- One certifier session's clock openings (`certificate.SignedClocks`): its LMS key and session, the openings, and the
link whose syncs are read. -/
structure Openings where
  lms : Lms.Params
  key : ByteArray
  session : ByteArray
  link : String
  openings : Array ClockOpening

/-- `{"key", "link", "lms", "openings", "session"}`, each opening `{"certificate", "salt", "sync"}`, its certificate as
the evidence's and its sync `[δ, num, den]`. -/
def openingsOf (j : Json) : V Openings := do
  exactly j ["key", "link", "lms", "openings", "session"] "the object"
  let l ← Flock.J.get j "lms" "the object"
  exactly l ["H", "N", "W", "toy"] "lms"
  let lms ← Lms.Params.make (← Flock.J.nat l "N" "lms") (← Flock.J.nat l "W" "lms") (← Flock.J.nat l "H" "lms")
    (← Flock.J.bool l "toy" "lms")
  let key ← Flock.unhex (← Flock.J.str j "key" "the object")
  unless key.size = 16 + lms.N do throw s!"an LMS public key is {16 + lms.N} bytes, not {key.size}"
  let openings ← (← Flock.J.arr j "openings" "the object").mapM fun o => do
    exactly o ["certificate", "salt", "sync"] "an opening"
    pure { sync := ← Check.syncOf (← Flock.J.get o "sync" "an opening"), salt := ← hexAt o "salt" "an opening",
           certificate := ← certificateOf (← Flock.J.get o "certificate" "an opening") "an opening's certificate" }
  pure { lms, key, session := ← hexAt j "session" "the object", link := ← Flock.J.str j "link" "the object",
         openings }

/-- Why an opening gives no sync (`CertifiedClock.refusal`), or none when it holds: its certificate isn't an egress one
(`egress`), isn't the key's (`key`), signs another digest than its sync's clock digest salted by its 64-byte salt
(`clock_commitment`, `opening`), or its signature doesn't verify in the session (`Certificate.signs`: `signature`). -/
def openingRefusal (s : Openings) (o : ClockOpening) : Option String :=
  let c := o.certificate
  if c.direction != .egress then some "egress"
  else if c.keyId != s.key.extract 0 16 then some "key"
  else if !(o.salt.size == saltBytes && (clockDigest o.sync).any (c.digest == certificateDigest o.salt ·)) then
    some "opening"
  else if !c.signs s.lms s.key s.session then some "signature"
  else none

/-- The openings that hold by `first` (none), by salt: the first of each salt, and whether another of that salt isn't
it. -/
def bySalt (s : Openings) (first : Array (Option String)) : Std.TreeMap.Raw (Array UInt8) (ClockOpening × Bool) :=
  (s.openings.zip first).foldl (init := ∅) fun m (o, r) =>
    if r.isSome then m
    else m.alter o.salt.data fun
      | some (x, another) => some (x, another || !x.same o)
      | none => some (o, false)

/-- Each opening's refusal: `openingRefusal`'s, else `salt` where another opening that holds, not this one, has its
salt, under which two windows of one sync would sign one digest; none where it holds. The openings are keyed by salt
once (`bySalt`), so past their signatures this takes n log n steps in the openings. -/
def openingRefusals (s : Openings) : Array (Option String) :=
  let first := s.openings.map (openingRefusal s)
  let salts := bySalt s first
  (s.openings.zip first).map fun (o, r) =>
    if r.isSome then r else if (salts.get? o.salt.data).any (·.2) then some "salt" else none

/-- Link-windows in order: by link, then by window. -/
def linkWindowOrder (a b : String × Nat) : Ordering := (compare a.1 b.1).then (compare a.2 b.2)

/-- What the openings that hold sign, by link-window: the first one's sync, and whether another signs another sync. -/
abbrev Signed := Std.TreeMap.Raw (String × Nat) (ClockSync × Bool) linkWindowOrder

/-- The openings that hold (`codes` none), keyed by link-window once, so that `signedSyncs` and `split` take n log n
steps in the openings. -/
def signedBy (s : Openings) (codes : Array (Option String)) : Signed :=
  (s.openings.zip codes).foldl (init := ∅) fun m (o, r) =>
    if r.isSome then m
    else m.alter (o.certificate.link, o.certificate.window) fun
      | some (x, two) => some (x, two || !(x.delta == o.sync.delta && x.rho == o.sync.rho))
      | none => some (o.sync, false)

/-- `SignedClocks.syncs(link)`: window 0's sync on, each the one sync the holding openings sign for the window, up to
the first window without exactly one. -/
def signedSyncs (s : Openings) (signed : Signed) : Array ClockSync := Id.run do
  let mut out := #[]
  for w in [0:s.openings.size] do
    match signed.get? (s.link, w) with
    | some (x, false) => out := out.push x
    | _ => break
  return out

/-- The link-windows, of any link, that the holding openings sign two or more syncs for, which have none, in
`linkWindowOrder`. -/
def split (signed : Signed) : Array (String × Nat) :=
  signed.foldl (init := #[]) fun acc k (_, two) => if two then acc.push k else acc

/-- `network-records verdicts`' `clocks`: on `openingsOf`'s object, each opening's refusal (`openingRefusals`, null
where it holds), the link-windows with two or more signed syncs (`split`, each `[link, window]`), and the link's signed
syncs (`signedSyncs`, each `[δ, num, den]`), which `lean_audit.signed_clocks` reads with `certificate.SignedClocks`
beside it as the reference, each refusal and each such link-window by identity. -/
def clocksFn (j : Json) : Except String Json := do
  let s ← openingsOf j
  let codes := openingRefusals s
  let signed := signedBy s codes
  let sync (x : ClockSync) : Json := .arr #[toJson x.delta, toJson x.rho.num, toJson x.rho.den]
  return Json.mkObj [("refused", .arr (codes.map fun | some r => .str r | none => .null)),
    ("split", .arr ((split signed).map fun (l, w) => .arr #[.str l, toJson w])),
    ("syncs", .arr ((signedSyncs s signed).map sync))]

end NetworkCheck.Period
