import CertifierDevice.Issuer
import CertifierDevice.Difftest

/-!
# The issuer difftest: `Issuer.certify` against `certificate.Issuer`, on generated records

`scripts/issuer_vectors.py` writes instances (an LMS key from a seed, and calls to one issuer: a record, its direction,
grid parameters, Σ_sync, parameters digest and deadline bit, the leaf the host reserves for it or a refusal, and a
tamper) and what the checkout's real `certificate.Issuer` does with each, its signer signing at the call's leaf.
`CertifierDevice/IssuerDifftestMain.lean` reads both files and fails unless `evaluate`, which makes the key with
`Lms.keygen` and threads one `Issuer` through the calls, gives every result exactly: each call's parameters digest
(`parameters`), and `null` where `Issuer.certify` refuses the record, or the certified record (`null` when the
reservation is refused) with its head's encoding and digest, its certificate's fields, `CertifiedRecord.verify` under
the key and after the tamper; and at the end the issuer's `refused`. The Lean audit runs it (`runs` in
`lean-audit.json`): compiled evaluation, a test and not a proof.
-/

namespace CertifierDevice.IssuerDifftest

open Lean Flock
open NetworkCertifier (Reason Status)
open NetworkCheck (Direction)
open CertifierDevice.Difftest (hexJson bytesOf arrOf paramsOf syncSetOf)

def reasons : List Reason :=
  [.monitorUnavailable, .overflow, .malformed, .dropped, .deliveryRefused, .coverageNeverEstablished]

def statusOf : Json → Except String Status
  | .null => pure .complete
  | j => do
    let s ← j.getStr?
    match reasons.find? (reasonValue · == s) with
    | some r => pure (.notEstablished r)
    | none => throw s!"not a reason: {s}"

def slotOf : Json → Except String (Option ByteArray)
  | .null => pure none
  | j => some <$> bytesOf j

def recordOf (j : Json) : Except String Record := do
  let grid ← (← j.getObjValAs? (Array Json) "grid").mapM fun row => do (← arrOf row).mapM slotOf
  pure { link := ← j.getObjValAs? String "link", window := ← j.getObjValAs? Nat "window", grid,
         status := ← statusOf (← j.getObjVal? "status") }

/-- Byte `pos mod size` of `b` with its low bit flipped. -/
def flip (b : ByteArray) (pos : Nat) : ByteArray :=
  if b.size = 0 then b else
    let i := pos % b.size
    b.set! i (b.get! i ^^^ 1)

/-- `verify` after the tamper: the certificate's signature (a byte, or its last byte dropped), link, direction, window or
digest changed, the head's deadline bit flipped, or the public key's identifier or root changed; or the certificate's
link, direction, window or key identifier changed and signed again by `k` at leaf `q`. -/
def tampered (c : CertifiedRecord) (k : Lms.PrivateKey) (q : Nat) (pub : ByteArray) : String × Nat → Except String Bool
  | (kind, pos) => do
    let x := c.certificate
    let h := c.head
    let id := pub.extract 0 16
    let root := pub.extract 16 pub.size
    let other := if x.direction == .egress then Direction.ingress else .egress
    let resign (y : Certificate) : Except String CertifiedRecord := do
      let m ← NetworkCheck.certificateMessage y.keyId y.link y.direction y.window y.digest
      pure { c with certificate := { y with signature := (← Lms.signAt k q m).encode } }
    let (c, pub) ← match kind with
      | "signature" => pure ({ c with certificate := { x with signature := flip x.signature pos } }, pub)
      | "signature-length" => pure ({ c with certificate := { x with signature := x.signature.extract 0 (x.signature.size - 1) } }, pub)
      | "link" => pure ({ c with certificate := { x with link := x.link ++ "x" } }, pub)
      | "direction" => pure ({ c with certificate := { x with direction := other } }, pub)
      | "window" => pure ({ c with certificate := { x with window := x.window ^^^ 1 } }, pub)
      | "digest" => pure ({ c with certificate := { x with digest := flip x.digest pos } }, pub)
      | "deadline" => pure ({ c with head := { h with deadline := !h.deadline } }, pub)
      | "public-id" => pure (c, flip id pos ++ root)
      | "public-root" => pure (c, id ++ flip root pos)
      | "resign-link" => pure (← resign { x with link := x.link ++ "x" }, pub)
      | "resign-direction" => pure (← resign { x with direction := other }, pub)
      | "resign-window" => pure (← resign { x with window := x.window ^^^ 1 }, pub)
      | "resign-key-id" => pure (← resign { x with keyId := flip x.keyId pos }, pub)
      | t => throw s!"no tamper {t}"
    pure (c.verify k.params pub)

def certifiedJson (c : CertifiedRecord) (verified : Bool) (tampered : Json) : Except String Json := do
  let x := c.certificate
  pure <| Json.mkObj [
    ("head", .arr #[hexJson (← c.head.encode), hexJson (← c.head.digest)]),
    ("certificate", .arr #[hexJson x.keyId, .str x.link, .str x.direction.value, toJson x.window, hexJson x.digest,
                           hexJson x.signature]),
    ("verify", toJson verified), ("tampered", tampered)]

/-- What `Issuer`'s definitions give on an instance, in the shape of the Python's results, with each call's outcome. -/
def evaluate (j : Json) : Except String (Json × List String) := do
  let key ← j.getObjVal? "key"
  let #[n, w, h, toy] ← key.getObjValAs? (Array Json) "params" | throw "not [N, W, H, toy]"
  let p ← Lms.Params.make (← n.getNat?) (← w.getNat?) (← h.getNat?) (← toy.getBool?)
  let k ← Lms.keygen p (← bytesOf (← key.getObjVal? "seed"))
  let pub := k.publicKey
  let mut issuer : Issuer := { key := k }
  let mut calls := #[]
  let mut outcomes := []
  for c in ← j.getObjValAs? (Array Json) "calls" do
    let params ← paramsOf (← c.getObjVal? "params")
    let ss ← match c.getObjValD "sync_set" with
      | .null => pure none
      | s => some <$> syncSetOf s
    let digest ← match c.getObjValD "parameters" with
      | .null => parameters params ss
      | b => bytesOf b
    let leaf : Except String Nat ← match c.getObjValD "leaf" with
      | .null => pure (.error "the leaf state refused the reservation")
      | q => .ok <$> q.getNat?
    let tamper ← match c.getObjValD "tamper" with
      | .null => pure none
      | t => do
        let #[kind, pos] ← arrOf t | throw s!"not [kind, at]: {t.compress}"
        pure (some ((← kind.getStr?), (← pos.getNat?)))
    let record ← recordOf (← c.getObjVal? "record")
    let direction ← Direction.ofValue (← c.getObjValAs? String "direction")
    let deadline ← c.getObjValAs? Bool "deadline"
    let mut certify := Json.null
    match issuer.certify leaf record direction params digest deadline with
    | .error _ => outcomes := "refused record" :: outcomes
    | .ok (i, none) =>
      issuer := i
      certify := Json.mkObj [("certified", .null)]
      outcomes := "refused leaf" :: outcomes
    | .ok (i, some cr) =>
      issuer := i
      let verified := cr.verify p pub
      let t ← match tamper with
        | none => pure Json.null
        | some t => toJson <$> tampered cr k (← leaf) pub t
      certify := Json.mkObj [("certified", ← certifiedJson cr verified t)]
      outcomes := s!"certified, {if verified then "verified" else "NOT verified"}" :: outcomes
      if let some (kind, _) := tamper then
        outcomes := s!"tampered {kind}, {if t == toJson true then "verified" else "rejected"}" :: outcomes
    calls := calls.push (Json.mkObj [("parameters", hexJson digest), ("certify", certify)])
  let refused := issuer.refused.map fun (link, d, window) => Json.arr #[.str link, .str d.value, toJson window]
  pure (Json.mkObj [("public", hexJson pub), ("calls", .arr calls), ("refused", .arr refused)], outcomes)

/-- How the Lean side's results for instance line `i` differ from Python's, one line per call or field that differs, and
the Lean side's outcomes. -/
def differences (i : Nat) (inst python : String) : List String × List String :=
  match Json.parse inst >>= evaluate, Json.parse python with
  | .error e, _ => ([s!"instance {i}: the Lean side rejects it: {e}"], [])
  | _, .error e => ([s!"instance {i}: Python's results don't parse: {e}"], [])
  | .ok (lean, outcomes), .ok py =>
    let a := (lean.getObjValD "calls").getArr?.toOption.getD #[]
    let b := (py.getObjValD "calls").getArr?.toOption.getD #[]
    let calls := if a.size != b.size then [s!"instance {i}: {a.size} calls, Python {b.size}"] else
      (a.zip b).toList.zipIdx.filterMap fun ((x, y), k) =>
        if x == y then none else some s!"instance {i} call {k}: Lean {x.compress.take 300}, Python {y.compress.take 300}"
    let fields := ["public", "refused"].filterMap fun f =>
      let x := lean.getObjValD f
      let y := py.getObjValD f
      if x == y then none else some s!"instance {i}: {f}: Lean {x.compress.take 300}, Python {y.compress.take 300}"
    let found := calls ++ fields
    (if found.isEmpty && lean != py then [s!"instance {i}: Python's results have fields Lean doesn't give"] else found,
     outcomes)

/-- `certifier-issuer-difftest INSTANCES PYTHON N`: every instance line of `INSTANCES` evaluated, each compared with the
same line of `PYTHON`. Fails unless both files have `N` lines and none differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: certifier-issuer-difftest INSTANCES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"issuer difftest: {a.size} instances and {b.size} results, not {n}"
    return 1
  let tasks := (a.zip b).toList.zipIdx.map fun ((x, y), i) => Task.spawn fun _ => differences i x y
  let results := tasks.map Task.get
  let found := results.flatMap (·.1)
  let counts := results.foldl (fun m (_, ks) => ks.foldl (fun (m : Std.HashMap String Nat) k => m.insert k (m.getD k 0 + 1)) m) {}
  for (k, v) in counts.toList.mergeSort (fun x y => x.1 ≤ y.1) do IO.println s!"  {v} {k}"
  for d in found.take 50 do IO.eprintln d
  IO.println s!"issuer difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end CertifierDevice.IssuerDifftest
