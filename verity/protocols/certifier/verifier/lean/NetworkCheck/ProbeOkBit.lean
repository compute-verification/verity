/-
A probe, not for landing: the rule the period's rows form would hold a scan's registration to if each scan member's
public word `o0` were committed under public salts (`scan_zk.ok_salt`). A member's `o0` tree is then the tree of
`leaves` rows `OK_ROW`, which anyone computes, so a registration whose `<member>/o0` root is that root names no unit
whose word is another. Three forms of one rule, each over every `roots` entry whose port ends `/o0` and is no
entries port:

* `okWalk`: `Flock.Registered.checkKnown` on each entry, which hashes every row of every member on every check;
* `okCached`: the root of n such rows at a domain (`okRoot`) computed once for each distinct (leaves, domain) the
  registration names, and each entry's root compared to it;
* `okCarried`: the constants carried (`Const`), each computed once by whoever reads the table (`checkCarried`, which
  is `checkKnown` on each), and each entry's root only compared.

Each also holds the entries' leaves to sum to the scan's 2T proof blocks.
-/
module

public import Flock.RegKnown

@[expose] public section

namespace NetworkCheck
namespace ProbeOkBit

open Lean (Json)
open Flock

/-- `scan_zk.OK_SALT_TAG`: the tag of the public salts a public `o0`'s rows are committed under. -/
def OK_SALT_TAG : ByteArray := "verity-network-certifier/scan-ok-row-salt/v1".toUTF8.push 0

/-- `scan_zk.OK_ROW`: a public `o0`'s one row the verifier accepts, the ok bit set, in its one word. -/
def OK_ROW : Array Nat := #[1]

/-- `scan_zk.ok_salt`: `HmRow.SALT_BYTES` bytes of SHA-512 in counter mode, `SHA-512(msg ‖ u32be c)` for
`c < HmRow.SALT_BYTES / 64`, each output whole, over msg = `OK_SALT_TAG` ‖ u64be(position): the row's index in its
member's tree, and nothing of the link, the execution phase or the registration. -/
def okSalt (position : Nat) : ByteArray :=
  let msg := OK_SALT_TAG ++ HmRow.be64 position
  (List.range (HmRow.SALT_BYTES / 64)).foldl (fun acc c => acc ++ Sha512.hash (msg ++ be32 c.toUInt32)) ByteArray.empty

/-- `scan_zk.ok_known`: `n` rows `row`, row `i` under `okSalt i`. -/
def okKnown (n : Nat) (row : Array Nat := OK_ROW) : Array (Array Nat × ByteArray) :=
  (Array.range n).map fun i => (row, okSalt i)

/-- A member's public `o0` as a registered port: `leaves` `hm96-sha512/row/v2` rows of one 16-bit word at `domain`. -/
def okPort (port : String) (root domain : ByteArray) (leaves : Nat) : Registered.Port :=
  { name := port, value := port, root, leaves, words := some 1, wordBits := 16, domain, readsRel := #[],
    schema := HmRow.ROW_SCHEMA_V2 }

/-- The root of `leaves` rows `row` under the public salts at `domain`: `checkKnown`'s own computation. -/
def okRoot (leaves : Nat) (domain : ByteArray) (row : Array Nat := OK_ROW) : ByteArray :=
  FV3.root domain (Registered.knownLeaves (okPort "" ByteArray.empty domain leaves) (okKnown leaves row)) Sha512.hash

/-- A `roots` entry of a registration. -/
structure Entry where
  port : String
  root : ByteArray
  domain : ByteArray
  leaves : Nat

/-- A registration's `roots` entries. -/
def entries (reg : Json) : V (Array Entry) := do
  let .arr xs := reg.getObjValD "roots" | throw "the registration has no roots"
  xs.mapM fun e => do
    let .str port := e.getObjValD "port" | throw "a root names no port"
    let .str root := e.getObjValD "root" | throw s!"{port}: no root"
    let .str domain := e.getObjValD "domain" | throw s!"{port}: no domain"
    let .ok leaves := (e.getObjValD "leaves").getNat? | throw s!"{port}: no leaves"
    return { port, root := ← unhex root, domain := ← unhex domain, leaves }

/-- Whether a port is a member's public word: it ends `/o0` and is no entries port (`entriesPort`: the period check's
`entriesPortName`). -/
def publicO0 (entriesPort : String → Bool) (port : String) : Bool :=
  "/o0".toList.isSuffixOf port.toList && !entriesPort port

/-- A registration's public `o0` entries, and that their leaves are the scan's 2T proof blocks. -/
def publicEntries (reg : Json) (T : Nat) (entriesPort : String → Bool) : V (Array Entry) := do
  let es := (← entries reg).filter fun e => publicO0 entriesPort e.port
  let sum := es.foldl (fun n e => n + e.leaves) 0
  unless sum == 2 * T do throw s!"the scan's public o0 rows are {sum}, not its 2T = {2 * T} proof blocks"
  return es

/-- **The rule, hashing every row**: each public `o0` entry is the root of its `leaves` rows `OK_ROW` under the public
salts at its own domain (`Flock.Registered.checkKnown`), and their leaves sum to 2T. -/
def okWalk (reg : Json) (T : Nat) (entriesPort : String → Bool := fun _ => false) : V Unit := do
  let es ← publicEntries reg T entriesPort
  es.toList.forM fun e => Registered.checkKnown (okPort e.port e.root e.domain e.leaves) (okKnown e.leaves)

/-- The root of `leaves` rows `OK_ROW` under the public salts at `domain`. -/
structure Const where
  leaves : Nat
  domain : ByteArray
  root : ByteArray

/-- Whether a constant is of an entry's leaves and domain. -/
def Const.names (c : Const) (e : Entry) : Bool := c.leaves == e.leaves && c.domain == e.domain

def refused (e : Entry) : String :=
  s!"registered reads: port {e.port}: its registered root is not the root of {e.leaves} rows {OK_ROW} under the public salts"

/-- **The rule, hashing once for each distinct (leaves, domain)**: the constants it computed, in order. -/
def okCached (reg : Json) (T : Nat) (entriesPort : String → Bool := fun _ => false) : V (Array Const) := do
  let es ← publicEntries reg T entriesPort
  let mut cs : Array Const := #[]
  for e in es do
    unless cs.any (fun c => c.names e) do
      cs := cs.push { leaves := e.leaves, domain := e.domain, root := okRoot e.leaves e.domain }
    unless cs.any (fun c => c.names e && c.root == e.root) do throw (refused e)
  return cs

/-- **A carried table's one check**: each constant is the root of its rows (`checkKnown`). -/
def checkCarried (cs : Array Const) : V Unit :=
  cs.toList.forM fun c => Registered.checkKnown (okPort "a carried constant" c.root c.domain c.leaves) (okKnown c.leaves)

/-- **The rule, comparing only**: each public `o0` entry's root is the carried constant of its leaves and domain. -/
def okCarried (cs : Array Const) (reg : Json) (T : Nat) (entriesPort : String → Bool := fun _ => false) : V Unit := do
  let es ← publicEntries reg T entriesPort
  es.toList.forM fun e => unless cs.any (fun c => c.names e && c.root == e.root) do throw (refused e)

def verdict {α : Type} (v : V α) : String :=
  match v with
  | .ok _ => "ACCEPT"
  | .error e => s!"REFUSE {e}"

def accepts {α : Type} (v : V α) : Bool :=
  match v with
  | .ok _ => true
  | .error _ => false

/-- SHA-512 calls of `okRoot` at `leaves` rows: each row's salt (`HmRow.SALT_BYTES / 64`), row digest, salt digest,
hm96 leaf and frame leaf, then the frame's pads and nodes. -/
def okRootCalls (leaves : Nat) : Nat :=
  let width := nextPow2 leaves
  (HmRow.SALT_BYTES / 64 + 4) * leaves + (width - leaves) + (width - 1)

end ProbeOkBit
end NetworkCheck
