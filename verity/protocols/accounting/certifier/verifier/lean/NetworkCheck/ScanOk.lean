module

public import Flock.RegKnown
public import Identity.Domains
public import Identity.Tagged
meta import Flock.RegKnown
meta import Identity.Domains
meta import Identity.Tagged

/-!
# The scan's ok bits at public content

A proof block of the network certifier's scan outputs one public word, `o0`, its ok bit: 1 when every check of its
bucket held. A member's `o0` rows are committed under public salts (`salt`), each row's by its index in the member's
tree, so the tree of a member whose every proof block is ok is one anyone computes: its `n` rows `OK_ROW` under salts
`0 … n − 1`, at the member's `o0` frame domain (`domain`), which is its template's and its proof blocks' count's and
names no run, round or execution phase. One execution phase's scan has the members `members` gives, a function of the
schedule's `T` alone, so the roots are constants of `T` (`consts`). A scan has at least two ticks: at `T = 1` its one
bucket would be `last` and `ingress-0` a member of no proof block, a shape no scan is staged in, so `consts` refuses
a `T` below 2.

The scan's start unit, the member `start`, is none of them: it has no ok bit, and its `o0` (`START`) is the state it
copies, a row of the scan's state, which the period check holds by the scan's state reads (`Period.stateRead`).

The period check (its rows form, `okChecks`) computes them once a period, from `T`, and holds every execution
phase's scan's registration to them by comparison alone (`held`): its `roots` name each member's `<member>/o0` at the
constant's root, leaves and domain, and no other port ending `/o0` but the start unit's. A member's statement reads
its `o0` rows at the root its registration names, so a proof block's ok word opens a row `OK_ROW` or SHA-512
collides: the bit is the verifier's own computation from public parameters, never a row the prover's registered file
states.

**Cost.** `consts T` is `calls T` SHA-512 calls, of which `rootCalls` are the trees' (511 at T = 32): each row's salt,
row digest, salt digest, hm96 leaf and frame leaf, then the frame's pads and nodes, and three for each member's domain.
`held` makes none: it reads the registration's `roots` once and compares strings and counts.

`verity.protocols.certifier.verifier.scan_ok` mirrors it. Code: it imports Flock's registered reads at known content
(`Flock.RegKnown`) and core's identity hash and table of domain separators.
-/

@[expose] public section

namespace NetworkCheck.ScanOk

open Lean (Json toJson)
open Flock (V)

/-- `scan_ok.OK_ROW`: the ok row the verifier accepts, a proof block's one output bit, 1, in its one word. -/
def OK_ROW : Array Nat := #[1]

def WORD_BITS : Nat := 16

/-- `scan_ok.PORT`: a scan proof block's public word, its first output. -/
def PORT : String := "o0"

/-- `scan_ok.START`: the start unit's first output, as a registration's `roots` name it: the state it copies, no ok
bit. -/
def START : String := "start/o0"

/-- `scan_ok.SALT_TAG`: the ok rows' hm96 salt tag. -/
def SALT_TAG : ByteArray := Domains.CERTIFIER_SCAN_OK_ROW_SALT.toUTF8.push 0

/-- `scan_ok.salt`: ok row `position`'s hm96 salt, its index in its member's tree and nothing of the link, the
execution phase or the registration: `HmRow.SALT_BYTES` bytes of SHA-512 in counter mode, `SHA-512(msg ‖ u32be c)` for
`c < HmRow.SALT_BYTES / 64`, each output whole, over msg = `SALT_TAG` ‖ u64be(position). -/
def salt (position : Nat) : ByteArray :=
  let msg := SALT_TAG ++ Flock.HmRow.be64 position
  (List.range (Flock.HmRow.SALT_BYTES / 64)).foldl
    (fun acc c => acc ++ Flock.Sha512.hash (msg ++ Flock.be32 c.toUInt32)) ByteArray.empty

/-- `scan_ok.known`: `n` rows `OK_ROW`, row `i` under `salt i`. -/
def known (n : Nat) : Array (Array Nat × ByteArray) := (Array.range n).map fun i => (OK_ROW, salt i)

/-- `scan_ok.set_of`: a scan template's statements' input set, which their frame domains bind, the template's alone. -/
def setOf (template : String) : String := s!"unit-scan-{template}"

/-- The `o0` frame domain of a statement over `n` proof blocks of `template`: Flock's statement's, at instances
`0 … n` of the input set `setOf template` (`scan_ok.binding`, then `advice.domain`). -/
def domain (template : String) (n : Nat) : V ByteArray := do
  let binding ← Identity.identityDigestSha512 Domains.FLOCK_CIRCUIT_BINDING (Json.mkObj [
    ("set", .str (setOf template)), ("lo", toJson (0 : Nat)), ("hi", toJson n), ("port", .str PORT),
    ("schema", .str Flock.HmRow.ROW_SCHEMA_V2)])
  pure (Flock.FV3.Sha512.Domain.make binding (-1) 0 n).id

/-- The root of `n` rows `OK_ROW` under the public salts at the frame domain whose id is `dom`: the computation of
`Flock.Registered.checkKnown`, on rows the verifier names. -/
def root (dom : ByteArray) (n : Nat) : ByteArray :=
  Flock.FV3.root dom (Flock.Registered.knownLeaves
    { name := PORT, value := PORT, root := ByteArray.empty, leaves := n, words := some 1, wordBits := WORD_BITS,
      domain := dom, readsRel := #[], schema := Flock.HmRow.ROW_SCHEMA_V2 } (known n)) Flock.Sha512.hash

/-- `scan_ok.members`: one execution phase's scan's members with an ok bit, (name, template, proof blocks), in its
Program's order after the start unit: `ingress-0`, buckets `0 … T − 2`, and `last`, bucket `T − 1`, whose next state
no state leaf holds, so it is a member of its own; and `egress-0`, the `T` egress buckets. `2T` proof blocks. Read at
`T ≥ 2`, where `ingress-0` has a proof block (`consts`). -/
def members (T : Nat) : Array (String × String × Nat) :=
  #[("ingress-0", "ingress", T - 1), ("last", "ingress", 1), ("egress-0", "egress", T)]

/-- A member's `o0` as a registration's `roots` names it when every proof block of the member is ok: its port, its
leaves, and its domain and root in hex. -/
structure Const where
  port : String
  leaves : Nat
  domain : String
  root : String
  deriving Inhabited

/-- The constant of the member `m` (name, template, proof blocks): its port `<name>/o0`, its proof blocks' count,
and the frame domain (`domain`) and root (`root`) of its ok rows, in hex. -/
def constOf (m : String × String × Nat) : V Const := do
  let dom ← domain m.2.1 m.2.2
  pure { port := m.1 ++ ("/" ++ PORT), leaves := m.2.2, domain := Flock.hexOf dom,
         root := Flock.hexOf (root dom m.2.2) }

/-- `scan_ok.constants`: each member's constant at the schedule's `T` (`members`, `constOf`), in order. A `T` below 2
is refused: `ingress-0` would be a member of no proof block, and no scan is staged so. -/
def consts (T : Nat) : V (Array Const) :=
  if T < 2 then throw s!"T = {T}: a scan has at least two ticks, its first bucket and its last"
  else (members T).mapM constOf

/-- `scan_ok.ok_port`: whether a `roots` entry's port is a proof block's public word, read over its characters: it
ends `/o0`. -/
def okPort (port : String) : Bool := ("/" ++ PORT).toList.isSuffixOf port.toList

/-- A registration's `roots` entries, none when it names none. -/
def rootsOf (reg : Json) : Array Json := match reg.getObjValD "roots" with | .arr xs => xs | _ => #[]

/-- Whether a `roots` entry's field `k` is the text `n`. -/
def named (k n : String) (e : Json) : Bool := match e.getObjValD k with | .str x => x == n | _ => false

/-- Whether a `roots` entry's `leaves` is the count `n`. -/
def counted (n : Nat) (e : Json) : Bool := match (e.getObjValD "leaves").getNat? with
  | .ok l => l == n
  | .error _ => false

/-- Whether a `roots` entry is no ok bits' or is at a constant of `cs`: an entry whose port ends `/o0` (`okPort`),
but the start unit's (`START`), is at that port's constant, in leaves, domain and root. -/
def entryHeld (cs : Array Const) (e : Json) : Bool := match e.getObjValD "port" with
  | .str q => !okPort q || q == START || cs.any fun c =>
      c.port == q && counted c.leaves e && named "domain" c.domain e && named "root" c.root e
  | _ => true

/-- `scan_ok.held`: a scan's registration names its ok bits at the constants `cs`: every constant's port is in its
`roots` (`rootsOf`, `named`), and every `roots` entry whose port ends `/o0`, but the start unit's, is at its port's
constant, in leaves, domain and root (`entryHeld`). It compares and hashes nothing. -/
def held (cs : Array Const) (reg : Json) : Bool :=
  let roots := rootsOf reg
  cs.all (fun c => roots.any (named "port" c.port)) && roots.all (entryHeld cs)

/-- SHA-512 calls of `root` at `n` rows: each row's salt (`HmRow.SALT_BYTES / 64`), row digest, salt digest, hm96
leaf and frame leaf, then the frame's pads and nodes. -/
def rootCalls (n : Nat) : Nat :=
  let width := Flock.nextPow2 n
  (Flock.HmRow.SALT_BYTES / 64 + 4) * n + (width - n) + (width - 1)

/-- SHA-512 calls of `consts T`: each member's domain (its binding, its range's identity and its id) and root. -/
def calls (T : Nat) : Nat := (members T).foldl (fun acc m => acc + 3 + rootCalls m.2.2) 0

-- the salt rule and the roots are `scan_ok`'s: `scan_ok.salt(0)[:8]`, `scan_ok.salt(7)[184:]`, and `scan_ok.constants`
-- at T = 16 and T = 3, and at T = 2, where `ingress-0` and `last` are one proof block of one template each, at one
-- domain and root
#guard Flock.hexOf ((salt 0).extract 0 8) == "f61f80289735b3b6"
#guard (salt 7).size == 192 && Flock.hexOf ((salt 7).extract 184 192) == "27e9fc9788b62bae"
#guard ((consts 16).map fun cs => cs.map fun c => (c.port, c.leaves, c.domain, c.root)).toOption == some #[
  ("ingress-0/o0", 15,
    "d85575918fa7f47e3539edee110443d3b8a548c64c291c23b45b3043ce7b9283\
      11f74c668f91d782ee01485a04c20797e0001cf41ab5942bded8aa4dc566deef",
    "970dca820db7995e32ba5fd9cf48f02c8cafe07b9b6a6d8d28f64d35a67f7478\
      669096a154227e1a8f3b1b8773ac2e887626ec60301aecce4535f32790376fa3"),
  ("last/o0", 1,
    "fd35975d6bafaf188590e3dd6a0648bc174821e81a7593ee9346bd01fc5546fe\
      b2e3d4a45bb7cf9fe6a363b31a5e5c8088ec201d487a39d25f07c0a288236a91",
    "a343d2d50d845bb751143f81fe6b59d8532cdb3238cb04903b63e2362c8dde6b\
      f12a60a5c5d18a83f7c0b7d961de2c1a9228e2f31894d796a19a3cc5b4da1f8f"),
  ("egress-0/o0", 16,
    "8190bf324e277745d8364f69d58f89ea895eb371ea9f19ce571e4b6b2f9ab181\
      8b1721da96991ce2600c6346e63535aa87ed9a63b589aa81425f1f53b1055a31",
    "fed250918695d50d116e2e5f263a898fbf82c26d392c02c1cf26ded06c2ef588\
      0c6b93cc47827632e333a23251c142ebf53526fc904e3a7395b48e8b8ed42419")]
#guard ((consts 3).map fun cs => cs.map fun c => (c.port, c.leaves, c.root)).toOption == some #[
  ("ingress-0/o0", 2,
    "3ff491826e963c3a7caa145c82f587b7591900c650757544118ca82b11fc9254\
      8e258b7a12cf2d3794ca09d48abc4b6b4bf1b7f2cac48389f63699a17b111a4e"),
  ("last/o0", 1,
    "a343d2d50d845bb751143f81fe6b59d8532cdb3238cb04903b63e2362c8dde6b\
      f12a60a5c5d18a83f7c0b7d961de2c1a9228e2f31894d796a19a3cc5b4da1f8f"),
  ("egress-0/o0", 3,
    "f9b478b82e457b566f32ac528e447b185d1df3ba42eb792fed539e61867b9272\
      e2161f725b83ad09aeb6fd947d5a14c9e5caacff5129b9c4f6ef73e5cd788a53")]
#guard ((consts 2).map fun cs => cs.map fun c => (c.port, c.leaves, c.root)).toOption == some #[
  ("ingress-0/o0", 1,
    "a343d2d50d845bb751143f81fe6b59d8532cdb3238cb04903b63e2362c8dde6b\
      f12a60a5c5d18a83f7c0b7d961de2c1a9228e2f31894d796a19a3cc5b4da1f8f"),
  ("last/o0", 1,
    "a343d2d50d845bb751143f81fe6b59d8532cdb3238cb04903b63e2362c8dde6b\
      f12a60a5c5d18a83f7c0b7d961de2c1a9228e2f31894d796a19a3cc5b4da1f8f"),
  ("egress-0/o0", 2,
    "91d104a3734809e12a186ef53f3d0769e82b5f3e93b47a5c34869b516ca14ce2\
      7c58be5d867f8da5a0e285dde551707f2c2c4829dab29fc7f58019e46a577856")]
-- the members are the schedule's 2T proof blocks with an ok bit; a schedule of fewer than two ticks has no constants,
-- and its refusal names its T
#guard (members 2).map (·.1) == #["ingress-0", "last", "egress-0"] &&
  [2, 3, 16, 32].all fun T => (members T).foldl (fun n m => n + m.2.2) 0 == 2 * T
#guard [0, 1].all fun T => match consts T with
  | .error e => e == s!"T = {T}: a scan has at least two ticks, its first bucket and its last"
  | .ok _ => false
-- the cost: at T = 16 the three trees are 121 + 7 + 127 = 255 calls and at T = 32 they are 249 + 7 + 255 = 511, and
-- each member's domain three more
#guard rootCalls 1 == 7 && rootCalls 2 == 15 && rootCalls 3 == 25 && rootCalls 15 == 121 && rootCalls 16 == 127 &&
  rootCalls 31 == 249 && rootCalls 32 == 255
#guard calls 16 == 255 + 9 && calls 32 == 511 + 9 && calls 3 == 15 + 7 + 25 + 9
-- a registration is held at its schedule's constants and at no other's; not with a member's root, leaves or domain
-- another, a member's port missing (`ingress-0`'s, or `last`'s) or named twice at two roots, or a port ending `/o0`
-- that is no member's; the start unit's `o0`, a row of the scan's state, is held to none
#guard
  let entry := fun (c : Const) => Json.mkObj [
    ("port", .str c.port), ("leaves", toJson c.leaves), ("domain", .str c.domain), ("root", .str c.root)]
  let reg := fun (es : Array Json) => Json.mkObj [("roots", Json.arr (es.push (Json.mkObj [
    ("port", .str "scan/state"), ("leaves", toJson (3 : Nat)), ("domain", .str "d"), ("root", .str "r")])))]
  match consts 3, consts 4 with
  | .ok cs, .ok other =>
    let es := cs.map entry
    held cs (reg es) && !held other (reg es) && !held cs (reg (other.map entry)) &&
      !held cs (reg (es.set! 2 (entry { cs[2]! with root := cs[0]!.root }))) &&
      !held cs (reg (es.set! 2 (entry { cs[2]! with leaves := 4 }))) &&
      !held cs (reg (es.set! 2 (entry { cs[2]! with domain := cs[0]!.domain }))) &&
      !held cs (reg (es.extract 1 3)) && !held cs (reg (es.extract 0 1 ++ es.extract 2 3)) &&
      !held cs (reg (es.push (entry { cs[2]! with root := cs[0]!.root }))) &&
      !held cs (reg (es.push (entry { cs[0]! with port := "other/o0" }))) &&
      held cs (reg (es.push (entry { cs[0]! with port := START }))) &&
      held cs (reg (es.push (entry { cs[0]! with port := "other/o1" }))) && held cs (reg (es ++ es)) &&
      !held cs (Json.mkObj [])
  | _, _ => false

end NetworkCheck.ScanOk
