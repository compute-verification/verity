import CoreProtocol.Protocol
import Flock.Hash

/-!
# The registration log

A deployment's registration is one append-only log of entries, each naming the head of the log before it
(`Entry.previous`): the digest of the log's last entry, or of its composite's genesis when the log is empty. So the head
names the whole log. An entry is one of two formats, each under its own versioned id:

* **hardware** (`verity/registration/hardware/v1`): the cards, locations and links it adds to the site. The site
  (`Site`) is what the log's hardware entries add, in order: the cards its devices name by digest, a tree of locations
  (the site at the root, its hosts, a host's devices, memory leaves by tier and bytes, and at the site the auditor's
  network certifiers and memory challengers, with their keys), and the links between locations and the rest of the
  world (`outside`), each with its capacity `β` in bits a tick each way and its latency in ticks.
* **information** (`verity/registration/information/v1`): the items it registers, each once in the log (`Item.key`):
  θ, each registered value (weights, initial state, tables) by its committed root, and the builder `B` that compiles
  advice into the circuit (its digest, its advice's format and its public cap `ℓ` on one declaration's advice, in
  bits). Registration registers the site, θ and `B`, not the circuit `B` builds; a subprotocol's Program and
  parameters, and the proof system, are that protocol's public verifier parameters, not registration.

An entry's digest is `SHA-512(format ‖ 0 ‖ canon(entry))` over the entry's canonical JSON, its format and `previous`
among its keys; a card's is the same under `verity/registration/card/v1`, and the genesis of a composite's log is the
same under `verity/registration/genesis/v1` over `{"composite": NAME}`. Registration is ASCII-only: a log or
parameters holding any other character is refused as `ascii` before any digest is taken.

The check (`check`) holds each entry to the log before it, and refuses the first entry that fails by the first check it
fails, in this order:

| Refusal | The check |
| --- | --- |
| `previous` | the entry follows the log's head |
| `duplicate-id` | no id is registered twice, `outside`'s included, and no card twice |
| `unknown-ref` | each new location's parent and each new link's ends are registered |
| `kind` | each new location sits under a parent its kind allows and has a tier exactly when it is memory; each new link joins two different ends of kinds a link may join, at a positive capacity; each certifier splits exactly one uplink |
| `card-digest` | each new card is the auditor's data sheet for its SKU, and each new device names a registered card |
| `capacity` | each new memory leaf holds something, and each device's memory leaves fit in its card's memory |
| `owner` | each new certifier or challenger is one of the auditor's own devices with its key, and every other new location is the prover's |
| `uncertified` | onsite: every boundary link is certified |
| `challenger-reach` | onsite: the host of every memory leaf is at an end of a memory challenger's link |
| `duplicate-information` | no information item is registered twice |

The hardware checks are an information entry's too, and hold of it whenever they held of the log before it, which adds
no hardware; so an information entry is held to `previous` and `duplicate-information` alone.

A component reads its view of the log by projection (`View`): the site, θ's values and the builder, at the head. As a
core protocol (`protocol`, `verity/registration/log/v1`) the log is the commitment, its head the
digest, and the parameters the auditor's own: the composite and, for the hardware checks, its data sheets and its
devices' keys.
-/

namespace Core.Registration

open Lean (Json)
open Flock (V)

def HARDWARE : String := "verity/registration/hardware/v1"
def INFORMATION : String := "verity/registration/information/v1"
def CARD : String := "verity/registration/card/v1"
def GENESIS : String := "verity/registration/genesis/v1"
def protocolId : String := "verity/registration/log/v1"

/-- `SHA-512(tag ‖ 0 ‖ canon(j))` as hex. Every JSON the log's types write holds integers only, which `Flock.canon`
always writes, so the fallback is never taken. -/
def tagged (tag : String) (j : Json) : String :=
  Flock.hexOf (Flock.Sha512.hash (tag.toUTF8.push 0 ++ ((Flock.canon j).toOption.getD "").toUTF8))

/-- The rest of the world: an end every site has, whose id no location may take. -/
def outside : String := "outside"

/-- What a location is. `outside` is the kind of the rest of the world, which no location's kind allows a parent of. -/
inductive Kind
  | site
  | host
  | device
  | memory
  | certifier
  | challenger
  | outside
  deriving DecidableEq, Repr, Inhabited

def Kind.name : Kind → String
  | .site => "site"
  | .host => "host"
  | .device => "device"
  | .memory => "memory"
  | .certifier => "certifier"
  | .challenger => "challenger"
  | .outside => "outside"

def Kind.all : List Kind := [.site, .host, .device, .memory, .certifier, .challenger, .outside]

/-- The kinds of parent a location of each kind may have, `none` being the root. -/
def Kind.parents : Kind → List (Option Kind)
  | .site => [none]
  | .host => [some .site]
  | .device => [some .host]
  | .memory => [some .device, some .host]
  | .certifier | .challenger => [some .site]
  | .outside => []

/-- The pairs of kinds a link may join, in either order. -/
def Kind.links : List (Kind × Kind) :=
  [(.host, .host), (.host, .outside), (.certifier, .host), (.certifier, .outside), (.challenger, .host)]

/-- The auditor's devices: the network certifiers and the memory challengers. -/
def Kind.auditor : Kind → Bool
  | .certifier | .challenger => true
  | _ => false

inductive Tier
  | hbm
  | dram
  | disk
  deriving DecidableEq, Repr, Inhabited

def Tier.name : Tier → String
  | .hbm => "hbm"
  | .dram => "dram"
  | .disk => "disk"

def Tier.all : List Tier := [.hbm, .dram, .disk]

inductive Owner
  | prover
  | auditor
  deriving DecidableEq, Repr, Inhabited

def Owner.name : Owner → String
  | .prover => "prover"
  | .auditor => "auditor"

def Owner.all : List Owner := [.prover, .auditor]

/-- An SKU's data-sheet card: tensor-core multiply-adds a tick (`tc`), bits from device memory to the chip a tick
(`bw`), bits of on-chip state (`chip`) and bytes of device memory (`mem`). -/
structure Card where
  sku : String
  tc : Nat
  bw : Nat
  chip : Nat
  mem : Nat
  deriving DecidableEq, Repr, Inhabited

/-- A location and its owner. A device names its card's digest, a memory leaf its tier and bytes, and the auditor's
devices their keys; a field the location's kind doesn't use is carried as written. -/
structure Loc where
  id : String
  kind : Kind
  parent : Option String := none
  owner : Owner := .prover
  card : String := ""
  tier : Option Tier := none
  bytes : Nat := 0
  key : String := ""
  deriving DecidableEq, Repr, Inhabited

structure Link where
  id : String
  ends : String × String
  β : Nat
  latency : Nat
  deriving DecidableEq, Repr, Inhabited

/-- **The site**: the cards, locations and links the log's hardware entries add, in order. -/
structure Site where
  cards : List Card := []
  locs : List Loc := []
  links : List Link := []
  deriving DecidableEq, Repr, Inhabited

/-- One of θ's registered values (weights, initial state, tables) by its committed root: its name, the root as hex, its
rows (`leaves`) of `words` words of `wordBits` bits (16 or 32), the hex id of the domain its leaves are hashed under,
which binds no window, and for public weights their checkpoint's SHA-512. -/
structure Theta where
  value : String
  root : String
  leaves : Nat
  words : Nat
  wordBits : Nat := 16
  domain : String
  checkpoint : Option String := none
  deriving DecidableEq, Repr

/-- The builder `B`: its digest, the format of the advice it compiles, and its public cap `ℓ` on one declaration's
advice, in bits. -/
structure Builder where
  digest : String
  advice : String
  cap : Nat
  deriving DecidableEq, Repr

inductive Item
  | theta (t : Theta)
  | builder (b : Builder)
  deriving DecidableEq, Repr

/-- What an item registers, which the log registers once: each θ value by its name, and one builder. -/
def Item.key : Item → String
  | .theta t => "theta " ++ t.value
  | .builder _ => "builder"

inductive Body
  | hardware (cards : List Card) (locs : List Loc) (links : List Link)
  | information (items : List Item)
  deriving DecidableEq, Repr

/-- **An entry of the registration log**: the head of the log it follows, and what it registers. -/
structure Entry where
  previous : String
  body : Body
  deriving DecidableEq, Repr

/-- **The registration log**, oldest entry first. -/
abbrev Log := List Entry

/-- The composite a log is registered for: onsite, whose boundary links the auditor's certifiers split and whose memory
its challengers reach, or remote, where neither need hold. -/
inductive Composite
  | onsite
  | remote
  deriving DecidableEq, Repr

def Composite.name : Composite → String
  | .onsite => "onsite"
  | .remote => "remote"

/-! ## Canonical JSON -/

def optJson {α} (f : α → Json) : Option α → Json
  | none => .null
  | some x => f x

def Card.toJson (k : Card) : Json :=
  Json.mkObj [("bw", Lean.toJson k.bw), ("chip", Lean.toJson k.chip), ("mem", Lean.toJson k.mem), ("sku", .str k.sku),
    ("tc", Lean.toJson k.tc)]

def Card.digest (k : Card) : String := tagged CARD k.toJson

def Loc.toJson (x : Loc) : Json :=
  Json.mkObj [("bytes", Lean.toJson x.bytes), ("card", .str x.card), ("id", .str x.id), ("key", .str x.key),
    ("kind", .str x.kind.name), ("owner", .str x.owner.name), ("parent", optJson .str x.parent),
    ("tier", optJson (fun t : Tier => .str t.name) x.tier)]

def Link.toJson (k : Link) : Json :=
  Json.mkObj [("beta", Lean.toJson k.β), ("ends", .arr #[.str k.ends.1, .str k.ends.2]), ("id", .str k.id),
    ("latency", Lean.toJson k.latency)]

def Item.toJson : Item → Json
  | .theta t => Json.mkObj ((t.checkpoint.map fun c => [("checkpoint_sha512", Json.str c)]).getD [] ++
      [("domain", .str t.domain), ("kind", .str "theta"), ("leaves", Lean.toJson t.leaves), ("root", .str t.root),
       ("value", .str t.value)] ++ (if t.wordBits == 16 then [] else [("word_bits", Lean.toJson t.wordBits)]) ++
      [("words", Lean.toJson t.words)])
  | .builder b => Json.mkObj [("advice", .str b.advice), ("cap", Lean.toJson b.cap), ("digest", .str b.digest),
      ("kind", .str "builder")]

def Entry.format (e : Entry) : String :=
  match e.body with
  | .hardware .. => HARDWARE
  | .information _ => INFORMATION

def Entry.toJson (e : Entry) : Json :=
  match e.body with
  | .hardware cards locs links =>
    Json.mkObj [("cards", .arr (cards.map Card.toJson).toArray), ("format", .str HARDWARE),
      ("links", .arr (links.map Link.toJson).toArray), ("locs", .arr (locs.map Loc.toJson).toArray),
      ("previous", .str e.previous)]
  | .information items =>
    Json.mkObj [("format", .str INFORMATION), ("items", .arr (items.map Item.toJson).toArray),
      ("previous", .str e.previous)]

/-- **An entry's digest**: `SHA-512(format ‖ 0 ‖ canon(entry))`. -/
def Entry.digest (e : Entry) : String := tagged e.format e.toJson

/-- The head of a composite's empty log. -/
def genesis (c : Composite) : String := tagged GENESIS (Json.mkObj [("composite", .str c.name)])

/-- **The log's head**, which names it: its last entry's digest, or its composite's genesis when it is empty. -/
def head (c : Composite) (log : Log) : String :=
  match log.getLast? with
  | none => genesis c
  | some e => e.digest

/-- The log of `bodies` in order, each entry following the head of the log before it. -/
def chain (c : Composite) (bodies : List Body) : Log :=
  (bodies.foldl (fun (acc : Log × String) b =>
    let e : Entry := { previous := acc.2, body := b }
    (acc.1 ++ [e], e.digest)) ([], genesis c)).1

/-! ## Decoding: exactly what `toJson` writes -/

private def keysOf : Json → Option (List String)
  | .obj kvs => some (kvs.toList.map (·.1))
  | _ => none

/-- `j` is an object with exactly the keys `ks`, which are sorted. -/
def exactly (j : Json) (ks : List String) (what : String) : V Unit :=
  if keysOf j == some ks then pure () else throw s!"{what}: the keys must be [{", ".intercalate ks}]"

/-- Registration is ASCII-only: `j`'s canonical JSON holds no other character, so its digest is the same however a
canonical JSON escapes the rest. -/
def ascii (j : Json) (what : String) : V Unit :=
  if ((Flock.canon j).toOption.getD "").all (fun (c : Char) => c.toNat < 128) then pure ()
  else throw s!"ascii: {what} holds a character outside ASCII"

def hex64 (j : Json) (what : String) : V String := do
  let s ← Flock.J.asStr j what
  let b ← (Flock.unhex s).mapError (s!"{what}: " ++ ·)
  if b.size != 64 then throw s!"{what}: not a SHA-512 digest (64 bytes)"
  pure s

private def byName {α} (all : List α) (name : α → String) (j : Json) (what : String) : V α := do
  let s ← Flock.J.asStr j what
  match all.find? (name · == s) with
  | some x => pure x
  | none => throw s!"{what}: {s} is not one of [{", ".intercalate (all.map name)}]"

private def optOf {α} (f : Json → V α) : Json → V (Option α)
  | .null => pure none
  | j => some <$> f j

def Card.ofJson (j : Json) : V Card := do
  exactly j ["bw", "chip", "mem", "sku", "tc"] "a card"
  pure { sku := ← Flock.J.str j "sku" "a card", tc := ← Flock.J.nat j "tc" "a card", bw := ← Flock.J.nat j "bw" "a card",
         chip := ← Flock.J.nat j "chip" "a card", mem := ← Flock.J.nat j "mem" "a card" }

def Loc.ofJson (j : Json) : V Loc := do
  exactly j ["bytes", "card", "id", "key", "kind", "owner", "parent", "tier"] "a location"
  let get := fun k => Flock.J.get j k "a location"
  pure { id := ← Flock.J.str j "id" "a location", kind := ← byName Kind.all Kind.name (← get "kind") "a location's kind",
         parent := ← optOf (Flock.J.asStr · "a location's parent") (← get "parent"),
         owner := ← byName Owner.all Owner.name (← get "owner") "a location's owner",
         card := ← Flock.J.str j "card" "a location",
         tier := ← optOf (byName Tier.all Tier.name · "a location's tier") (← get "tier"),
         bytes := ← Flock.J.nat j "bytes" "a location", key := ← Flock.J.str j "key" "a location" }

def Link.ofJson (j : Json) : V Link := do
  exactly j ["beta", "ends", "id", "latency"] "a link"
  let ends ← Flock.J.arr j "ends" "a link"
  let #[a, b] := ends | throw "a link: ends: not two ends"
  pure { id := ← Flock.J.str j "id" "a link", ends := (← Flock.J.asStr a "a link's end", ← Flock.J.asStr b "a link's end"),
         β := ← Flock.J.nat j "beta" "a link", latency := ← Flock.J.nat j "latency" "a link" }

def Item.ofJson (j : Json) : V Item := do
  match ← Flock.J.str j "kind" "an item" with
  | "theta" =>
    let has := fun k => (j.getObjVal? k).isOk
    exactly j ((if has "checkpoint_sha512" then ["checkpoint_sha512"] else []) ++
      ["domain", "kind", "leaves", "root", "value"] ++ (if has "word_bits" then ["word_bits"] else []) ++ ["words"])
      "a theta item"
    let wordBits ← if has "word_bits" then Flock.J.nat j "word_bits" "a theta item" else pure 16
    if has "word_bits" && wordBits != 32 then throw "a theta item: word_bits is 32 when present (16 is left out)"
    pure (.theta { value := ← Flock.J.str j "value" "a theta item",
                   root := ← hex64 (← Flock.J.get j "root" "a theta item") "a theta item's root",
                   leaves := ← Flock.J.nat j "leaves" "a theta item", words := ← Flock.J.nat j "words" "a theta item",
                   wordBits, domain := ← hex64 (← Flock.J.get j "domain" "a theta item") "a theta item's domain",
                   checkpoint := ← if has "checkpoint_sha512" then
                     some <$> hex64 (← Flock.J.get j "checkpoint_sha512" "a theta item") "a theta item's checkpoint"
                     else pure none })
  | "builder" =>
    exactly j ["advice", "cap", "digest", "kind"] "a builder item"
    pure (.builder { digest := ← hex64 (← Flock.J.get j "digest" "the builder") "the builder's digest",
                     advice := ← Flock.J.str j "advice" "the builder", cap := ← Flock.J.nat j "cap" "the builder" })
  | k => throw s!"an item: kind {k} is not one of [theta, builder]"

def Entry.ofJson (j : Json) : V Entry := do
  let format ← Flock.J.str j "format" "an entry"
  if format == HARDWARE then
    exactly j ["cards", "format", "links", "locs", "previous"] "a hardware entry"
    let previous ← hex64 (← Flock.J.get j "previous" "an entry") "an entry's previous"
    let cards ← (← Flock.J.arr j "cards" "a hardware entry").toList.mapM Card.ofJson
    let locs ← (← Flock.J.arr j "locs" "a hardware entry").toList.mapM Loc.ofJson
    let links ← (← Flock.J.arr j "links" "a hardware entry").toList.mapM Link.ofJson
    pure { previous, body := .hardware cards locs links }
  else if format == INFORMATION then
    exactly j ["format", "items", "previous"] "an information entry"
    let previous ← hex64 (← Flock.J.get j "previous" "an entry") "an entry's previous"
    pure { previous, body := .information (← (← Flock.J.arr j "items" "an information entry").toList.mapM Item.ofJson) }
  else throw s!"an entry: format {format} is not one of [{HARDWARE}, {INFORMATION}]"

def Composite.ofJson (j : Json) (what : String) : V Composite := byName (α := Composite) [.onsite, .remote] Composite.name j what

/-- The log as committed: its composite and its entries, `{"composite": NAME, "entries": [...]}`. -/
structure Committed where
  composite : Composite
  log : Log

def Committed.toJson (c : Committed) : Json :=
  Json.mkObj [("composite", .str c.composite.name), ("entries", .arr (c.log.map Entry.toJson).toArray)]

def Committed.ofJson (j : Json) : V Committed := do
  exactly j ["composite", "entries"] "the log"
  pure { composite := ← Composite.ofJson (← Flock.J.get j "composite" "the log") "the log's composite",
         log := ← (← Flock.J.arr j "entries" "the log").toList.mapM Entry.ofJson }

/-! ## The site's reading -/

namespace Site

variable (S : Site)

def add (cards : List Card) (locs : List Loc) (links : List Link) : Site :=
  ⟨S.cards ++ cards, S.locs ++ locs, S.links ++ links⟩

/-- The location an id names: the first registered under it. -/
def loc? (i : String) : Option Loc := S.locs.find? (·.id == i)

/-- The kind of what an id names: the outside's, or its location's; `none` when nothing is registered under it. -/
def kind? (i : String) : Option Kind := if i == outside then some .outside else (S.loc? i).map (·.kind)

/-- An id names something: the outside, or a registered location. -/
def known (i : String) : Bool := i == outside || S.locs.any (·.id == i)

/-- Every registered id: the locations', the links' and the outside's. -/
def ids : List String := S.locs.map (·.id) ++ S.links.map (·.id) ++ [outside]

/-- A device's card: the registered card under the digest it names. -/
def card? (x : Loc) : Option Card := S.cards.find? (·.digest == x.card)

/-- The bytes of the memory leaves directly under `i`. -/
def held (i : String) : Nat := ((S.locs.filter fun x => x.kind == .memory && x.parent == some i).map (·.bytes)).sum

/-- Link `k` joins kinds `a` and `b`, in either order. -/
def joins (k : Link) (a b : Kind) : Bool :=
  (S.kind? k.ends.1, S.kind? k.ends.2) == (some a, some b) || (S.kind? k.ends.1, S.kind? k.ends.2) == (some b, some a)

/-- A boundary link of the prover's: a host's link to the outside, or to the certifier that splits its uplink. -/
def boundary (k : Link) : Bool := S.joins k .host .outside || S.joins k .certifier .host

/-- A certified link: a certifier is at one of its ends. -/
def certified (k : Link) : Bool := S.kind? k.ends.1 == some .certifier || S.kind? k.ends.2 == some .certifier

/-- The far ends of `i`'s links: for each link with an end at `i`, its other end. -/
def far (i : String) : List String :=
  (S.links.filter fun k => k.ends.1 == i || k.ends.2 == i).flatMap fun k => [k.ends.1, k.ends.2].filter (· != i)

/-- The host a memory leaf is in: its parent when that is a host, else its parent's parent. -/
def host? (x : Loc) : Option String :=
  x.parent.bind fun p => if S.kind? p == some .host then some p else (S.loc? p).bind (·.parent)

/-- What the memory challengers reach: the ends of their links. -/
def reached : List String :=
  (S.links.filter fun k => S.kind? k.ends.1 == some .challenger || S.kind? k.ends.2 == some .challenger).flatMap
    fun k => [k.ends.1, k.ends.2]

def toJson : Json :=
  Json.mkObj [("cards", .arr (S.cards.map Card.toJson).toArray), ("links", .arr (S.links.map Link.toJson).toArray),
    ("locs", .arr (S.locs.map Loc.toJson).toArray)]

end Site

/-! ## The check -/

/-- What the auditor brings to the check: its data sheet for each SKU, and the key of each of its own devices. -/
structure Auditor where
  sheets : List Card := []
  keys : List (String × String) := []

def Auditor.sheet (A : Auditor) (sku : String) : Option Card := A.sheets.find? (·.sku == sku)

def Auditor.key (A : Auditor) (i : String) : Option String := (A.keys.find? (·.1 == i)).map (·.2)

inductive Refusal
  | previous
  | duplicateId
  | unknownRef
  | kind
  | cardDigest
  | capacity
  | owner
  | uncertified
  | challengerReach
  | duplicateInformation
  deriving DecidableEq, Repr

def Refusal.name : Refusal → String
  | .previous => "previous"
  | .duplicateId => "duplicate-id"
  | .unknownRef => "unknown-ref"
  | .kind => "kind"
  | .cardDigest => "card-digest"
  | .capacity => "capacity"
  | .owner => "owner"
  | .uncertified => "uncertified"
  | .challengerReach => "challenger-reach"
  | .duplicateInformation => "duplicate-information"

def nodup : List String → Bool
  | [] => true
  | x :: xs => !xs.contains x && nodup xs

/-- A certifier splits exactly one uplink: the far ends of its links are one host and the outside. -/
def splitsOne (S : Site) (x : Loc) : Bool :=
  let ks := (S.far x.id).map S.kind?
  ks == [some .host, some .outside] || ks == [some .outside, some .host]

/-- The first check a hardware entry adding `cards`, `locs` and `links` to the site `S` fails, for the composite `c`
and the auditor `A`, or `none`. -/
def hardwareRefusal (A : Auditor) (c : Composite) (S : Site) (cards : List Card) (locs : List Loc)
    (links : List Link) : Option Refusal :=
  let T := S.add cards locs links
  if !(nodup T.ids && nodup (T.cards.map (·.digest))) then some .duplicateId
  else if !(locs.all (fun x => x.parent.all T.known) && links.all fun k => T.known k.ends.1 && T.known k.ends.2) then
    some .unknownRef
  else if !(locs.all (fun x => x.kind.parents.contains (x.parent.bind T.kind?) && (x.kind == .memory) == x.tier.isSome) &&
      links.all (fun k => k.ends.1 != k.ends.2 && Kind.links.any (fun p => T.joins k p.1 p.2) && 0 < k.β) &&
      T.locs.all fun x => x.kind != .certifier || splitsOne T x) then some .kind
  else if !(cards.all (fun k => A.sheet k.sku == some k) &&
      locs.all fun x => x.kind != .device || (T.cards.map (·.digest)).contains x.card) then some .cardDigest
  else if !(locs.all (fun x => x.kind != .memory || 0 < x.bytes) &&
      T.locs.all fun x => x.kind != .device || (T.card? x).all (T.held x.id ≤ ·.mem)) then some .capacity
  else if !(locs.all fun x =>
      if x.kind.auditor then x.owner == .auditor && A.key x.id == some x.key else x.owner == .prover) then some .owner
  else if c == .onsite && !(T.links.all fun k => !T.boundary k || T.certified k) then some .uncertified
  else if c == .onsite && !(T.locs.all fun x => x.kind != .memory || (T.host? x).all T.reached.contains) then
    some .challengerReach
  else none

/-- The log so far, folded: its site, its information items in order, its head and its length. -/
structure State where
  site : Site
  items : List Item
  head : String
  entries : Nat

def State.empty (c : Composite) : State := { site := {}, items := [], head := genesis c, entries := 0 }

/-- The first check the entry `e` fails against the log folded to `s`, or `none`. -/
def Entry.refusal (A : Auditor) (c : Composite) (s : State) (e : Entry) : Option Refusal :=
  if e.previous != s.head then some .previous
  else match e.body with
    | .hardware cards locs links => hardwareRefusal A c s.site cards locs links
    | .information items =>
      if nodup ((s.items ++ items).map Item.key) then none else some .duplicateInformation

def State.add (s : State) (e : Entry) : State :=
  match e.body with
  | .hardware cards locs links => { s with site := s.site.add cards locs links, head := e.digest, entries := s.entries + 1 }
  | .information items => { s with items := s.items ++ items, head := e.digest, entries := s.entries + 1 }

/-- **The check**: the log folded entry by entry, each held to the log before it; or the first entry that fails, by its
index and the first check it fails. -/
def check (A : Auditor) (c : Composite) (log : Log) : Except (Nat × Refusal) State :=
  log.foldlM (fun s e => match e.refusal A c s with
    | some r => .error (s.entries, r)
    | none => .ok (s.add e)) (State.empty c)

/-! ## Views: what a component reads of the log -/

/-- What a component reads of an accepted log, at its head: the site, θ's registered values in order, and the builder. -/
structure View where
  site : Site
  theta : List Theta
  builder : Option Builder
  head : String

def State.view (s : State) : View :=
  { site := s.site, head := s.head,
    theta := s.items.filterMap fun | .theta t => some t | _ => none,
    builder := s.items.findSome? fun | .builder b => some b | _ => none }

/-- θ's registered value named `value`. -/
def View.value? (v : View) (value : String) : Option Theta := v.theta.find? (·.value == value)

def View.toJson (v : View) : Json :=
  Json.mkObj [("builder", optJson (Item.toJson ∘ .builder) v.builder), ("head", .str v.head),
    ("site", v.site.toJson), ("theta", .arr (v.theta.map (Item.toJson ∘ .theta)).toArray)]

/-! ## The core protocol -/

structure Params where
  composite : Composite
  auditor : Auditor

def Params.ofJson (j : Json) : V Params := do
  ascii j "the params"
  exactly j ["composite", "keys", "sheets"] "the params"
  let sheets ← (← Flock.J.arr j "sheets" "the params").toList.mapM Card.ofJson
  let keys ← (← Flock.J.arr j "keys" "the params").toList.mapM fun k => do
    exactly k ["id", "key"] "a key"
    pure (← Flock.J.str k "id" "a key", ← Flock.J.str k "key" "a key")
  if !nodup (sheets.map (·.sku)) then throw "the params: a data sheet's SKU is named twice"
  if !nodup (keys.map (·.1)) then throw "the params: a device's key is named twice"
  pure { composite := ← Composite.ofJson (← Flock.J.get j "composite" "the params") "the params' composite",
         auditor := { sheets, keys } }

/-- The log as committed, and its head as bytes. -/
structure Commitment where
  committed : Committed
  head : ByteArray

def commitmentOf (b : ByteArray) : V Commitment := do
  let j ← canonicalJson b
  ascii j "the log"
  let committed ← Committed.ofJson j
  pure { committed, head := ← Flock.unhex (head committed.composite committed.log) }

def evidenceOf (b : ByteArray) : V Unit := do
  exactly (← canonicalJson b) [] "the evidence"

def checkLog (p : Params) (c : Commitment) (_ : Unit) : Outcome Json Json :=
  if c.committed.composite != p.composite then
    .reject s!"composite: the log is {c.committed.composite.name}'s, not {p.composite.name}'s"
  else match check p.auditor p.composite c.committed.log with
    | .error (i, r) => .reject s!"entry {i}: {r.name}"
    | .ok s => .accept s.view.toJson (Json.mkObj [("entries", Lean.toJson s.entries)])

/-- **The registration log's check** as a core protocol, `verity/registration/log/v1`: the parameters are the auditor's
(`{"composite", "keys": [{"id", "key"}], "sheets": [CARD]}`), the commitment is the log
(`{"composite", "entries": [ENTRY]}`) with its head as digest, the evidence is `{}`. It accepts a log the check passes,
making public its view (`View.toJson`) and keeping its length, and rejects the first entry the check refuses as
`entry I: REFUSAL`, and a log or parameters outside ASCII as `ascii`. It is never undecided. -/
def protocol : Protocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Unit
  Public := Json
  Profile := Json
  params := Params.ofJson
  commitment := commitmentOf
  evidence := evidenceOf
  digest c := c.head
  check := checkLog
  publicJson := id
  profileJson := id

/-! ## An example: one node's onsite site, then its information -/

namespace Example

def card : Card := { sku := "sim-a", tc := 64, bw := 3200, chip := 1 <<< 25, mem := 64 <<< 30 }

def auditor : Auditor := { sheets := [card], keys := [("cert0", "k0"), ("chal0", "k1")] }

def hex (c : Char) : String := String.ofList (List.replicate 128 c)

def locs : List Loc :=
  [{ id := "site0", kind := .site }, { id := "node0", kind := .host, parent := some "site0" },
   { id := "gpu0", kind := .device, parent := some "node0", card := card.digest },
   { id := "hbm0", kind := .memory, parent := some "gpu0", tier := some .hbm, bytes := 1 <<< 30 },
   { id := "cert0", kind := .certifier, parent := some "site0", owner := .auditor, key := "k0" },
   { id := "chal0", kind := .challenger, parent := some "site0", owner := .auditor, key := "k1" }]

def links : List Link :=
  [{ id := "up0", ends := ("node0", "cert0"), β := 100, latency := 1 },
   { id := "up1", ends := ("cert0", outside), β := 100, latency := 1 },
   { id := "mc0", ends := ("chal0", "node0"), β := 10, latency := 1 }]

def weights : Theta :=
  { value := "weights", root := hex 'a', leaves := 4, words := 64, domain := hex 'b', checkpoint := some (hex 'c') }

def items : List Item :=
  [.theta weights, .theta { value := "state", root := hex 'd', leaves := 1, words := 8, wordBits := 32, domain := hex 'e' },
   .builder { digest := hex 'f', advice := "verity/advice/v1", cap := 32 }]

def bodies : List Body := [.hardware [card] locs links, .information items]

def refusal (c : Composite) (bodies : List Body) (A : Auditor := auditor) : Option (Nat × Refusal) :=
  match check A c (chain c bodies) with
  | .ok _ => none
  | .error r => some r

#guard refusal .onsite bodies == none
#guard ((check auditor .onsite (chain .onsite bodies)).toOption.map (·.view.value? "weights")) == some (some weights)
#guard (chain .onsite bodies).all fun e => match Entry.ofJson e.toJson with | .ok e' => e' == e | .error _ => false
#guard head .onsite (chain .onsite bodies) != head .remote (chain .remote bodies)
-- a log chained for remote is not onsite's
#guard (match check auditor .onsite (chain .remote bodies) with | .error r => some r | .ok _ => none) ==
  some (0, .previous)
-- onsite: an uplink no certifier splits; remote takes it
private def bare : List Body :=
  [.hardware [card] (locs.take 4) [{ id := "up0", ends := ("node0", outside), β := 100, latency := 1 }]]
#guard refusal .onsite bare == some (0, .uncertified)
#guard refusal .remote bare == none
-- onsite: memory no challenger reaches
#guard refusal .onsite [.hardware [card] (locs.take 5) (links.take 2)] == some (0, .challengerReach)
-- a card that isn't the auditor's data sheet, a device's memory over its card's, a certifier without its key
#guard refusal .remote [.hardware [{ card with mem := 1 }] [] []] == some (0, .cardDigest)
#guard refusal .remote bare { auditor with sheets := [] } == some (0, .cardDigest)
private def over : Loc := { id := "hbm0", kind := .memory, parent := some "gpu0", tier := some .hbm, bytes := (64 <<< 30) + 1 }
#guard refusal .remote [.hardware [card] (locs.take 3 ++ [over]) []] == some (0, .capacity)
#guard refusal .onsite bodies { auditor with keys := [("cert0", "k1"), ("chal0", "k1")] } == some (0, .owner)
-- an id twice, a parent nobody registered, a host under a host
#guard refusal .remote [.hardware [card] (locs.take 2) [], .hardware [] (locs.drop 1 |>.take 1) []] == some (1, .duplicateId)
#guard refusal .remote [.hardware [] [{ id := "node1", kind := .host, parent := some "site9" }] []] ==
  some (0, .unknownRef)
#guard refusal .remote [.hardware [] (locs.take 2 ++ [{ id := "node1", kind := .host, parent := some "node0" }]) []] ==
  some (0, .kind)
-- θ's weights registered twice, and a second builder; another θ value is a new item
#guard refusal .onsite (bodies ++ [.information (items.take 1)]) == some (2, .duplicateInformation)
#guard refusal .onsite (bodies ++ [.information (items.drop 2)]) == some (2, .duplicateInformation)
#guard refusal .onsite (bodies ++ [.information [.theta { weights with value := "tables" }]]) == none

private def bytes (j : Json) : ByteArray := ((Flock.canon j).toOption.getD "").toUTF8

private def params (c : Composite) : Json :=
  Json.mkObj [("composite", .str c.name), ("keys", .arr #[Json.mkObj [("id", "cert0"), ("key", "k0")],
    Json.mkObj [("id", "chal0"), ("key", "k1")]]), ("sheets", .arr #[card.toJson])]

private def answer (c : Composite) (log : Committed) : Answer :=
  protocol.answer (bytes (params c)) (bytes log.toJson) "{}".toUTF8

#guard (answer .onsite { composite := .onsite, log := chain .onsite bodies }).verdict == .accept
#guard (answer .onsite { composite := .onsite, log := chain .onsite bodies }).digest ==
  (Flock.unhex (head .onsite (chain .onsite bodies))).toOption
#guard (answer .onsite { composite := .remote, log := chain .remote bodies }).reason ==
  some "composite: the log is remote's, not onsite's"
#guard (answer .onsite { composite := .onsite, log := chain .onsite bare }).reason == some "entry 0: uncertified"
private def accented : Committed :=
  { composite := .onsite, log := chain .onsite [.information [.theta { weights with value := "poids\u00e9" }]] }
#guard ((answer .onsite accented).reason.map (·.startsWith "commitment: ascii")) == some true

end Example

end Core.Registration
