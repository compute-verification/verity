module

public import CoreProtocol.Protocol
public import CoreProtocol.Site
public import Flock.Hash
meta import CoreProtocol.Protocol
meta import CoreProtocol.Site
meta import Flock.Hash

/-!
# The registration log

A deployment's registration is one append-only log of entries, each naming the head of the log before it
(`Entry.previous`): the digest of the log's last entry, or of its composite's genesis when the log is empty. So the head
<<<<<<< ours
names the whole log. An entry is one of three kinds, each under its own versioned format:

* **site** (`verity/registration/site/v1`): the Site's own registration entry (`Core.Site.Registration.Entry`), the
  sheets, locations and links it adds to the Site (`Core.Site`), its JSON, digest and check the Site's.
||||||| base
names the whole log. An entry is one of two formats, each under its own versioned id:

* **hardware** (`verity/registration/hardware/v1`): the cards, locations and links it adds to the site. The site
  (`Site`) is what the log's hardware entries add, in order: the cards its devices name by digest, a tree of locations
  (the site at the root, its hosts, a host's devices, memory leaves by tier and bytes, and at the site the auditor's
  network certifiers and memory challengers, with their keys), and the links between locations and the rest of the
  world (`outside`), each with its capacity `β` in bits a tick each way and its latency in ticks. A card is a device's
  registered figures: `DeviceFigures`' SKU, tensor clock and peak rate of tensor-core multiply-adds for each operand
  class (int8 operands uniform over one of four ranges, or all zero), and its memory figures.
=======
names the whole log. An entry is one of three formats, each under its own versioned id:

* **hardware** (`verity/registration/hardware/v1`): the cards, locations and links it adds to the site. The site
  (`Site`) is what the log's hardware entries add, in order: the cards its devices name by digest, a tree of locations
  (the site at the root, its hosts, a host's devices, memory leaves by tier and bytes, and at the site the auditor's
  network certifiers and memory challengers, with their keys), and the links between locations and the rest of the
  world (`outside`), each with its capacity `β` in bits a tick each way and its latency in ticks. A card is a device's
  registered figures: `DeviceFigures`' SKU, tensor clock and peak rate of tensor-core multiply-adds for each operand
  class (int8 operands uniform over one of four ranges, or all zero), and its memory figures.
>>>>>>> theirs
* **information** (`verity/registration/information/v1`): the items it registers, each once in the log (`Item.key`):
  θ, each registered value (weights, initial state, tables) by its committed root, and the builder `B` that compiles
  advice into the circuit (its digest, its advice's format and its public cap `ℓ` on one declaration's advice, in
  bits). Registration registers the site, θ and `B`, not the circuit `B` builds; a subprotocol's Program and
  parameters, and the proof system, are that protocol's public verifier parameters, not registration.
* **declaration** (`verity/registration/declaration/v1`): a day's circuit (`Declared`, the spec's
  `Verity.Registration.Declared`): the day, by its index among the monitoring period's days, its program's digest, and
  its block Definitions by descriptor key. The prover declares one circuit a day, before the day's first coin, so a
  second declaration of a day is refused; a window's program is its day's (`View.declared?`).

An entry's digest is `SHA-512(format ‖ 0 ‖ canon(entry))` over the entry's canonical JSON, its format and `previous`
among its keys, and the genesis of a composite's log is the Site's (`Core.Site.Registration.sha512`), so a log of site
entries alone has the Site's head. Registration is ASCII-only without DEL: a log or parameters holding any other
character, as itself or escaped, is refused (`ascii`, in `Core.canonicalJson`'s words) before any entry is read.

The check (`check`) holds each entry to the log before it, and refuses the first entry that fails by the first check it
<<<<<<< ours
fails: a site entry by the Site's checks (`Core.Site.Registration.Entry.refusal`, `previous` first), and an information
entry by `previous` and then `duplicate-information`, no item registered twice, and a declaration by `previous` and
then `duplicate-declaration`, no day declared twice. A site entry's checks hold of an information entry or a declaration
whenever they held of the log before it, which it adds no hardware to.

A component reads its view of the log by projection (`View`): the Site, θ's values, the builder and the days' declared
circuits, at the head. As a
core protocol (`protocol`, `verity/registration/log/v1`) the log is the commitment, its head the digest, and the
parameters the auditor's own: the composite and the Site's auditor (`Core.Site.Registration.Auditor`: its data sheets,
the tolerance it allows a sheet's figures around their estimates, and its devices' keys).
||||||| base
fails, in this order:

| Refusal | The check |
| --- | --- |
| `previous` | the entry follows the log's head |
| `duplicate-id` | no id is registered twice, `outside`'s included, and no card twice |
| `unknown-ref` | each new location's parent and each new link's ends are registered |
| `kind` | each new location sits under a parent its kind allows and has a tier exactly when it is memory; each new link joins two different ends of kinds a link may join, at a positive capacity; each certifier splits exactly one uplink |
| `card-digest` | each new card's SKU has an auditor's data sheet, and each new device names a registered card |
| `figures` | each new card's clock and peak rates are within the auditor's tolerance of its sheet's in both directions, its peak rates are positive, and its memory figures are its sheet's |
| `capacity` | each new memory leaf holds something, and each device's memory leaves fit in its card's memory |
| `owner` | each new certifier or challenger is one of the auditor's own devices with its key, and every other new location is the prover's |
| `uncertified` | onsite: every boundary link is certified |
| `challenger-reach` | onsite: the host of every memory leaf is at an end of a memory challenger's link |
| `duplicate-information` | no information item is registered twice |

The hardware checks are an information entry's too, and hold of it whenever they held of the log before it, which adds
no hardware; so an information entry is held to `previous` and `duplicate-information` alone.

A component reads its view of the log by projection (`View`): the site, θ's values and the builder, at the head. As a
core protocol (`protocol`, `verity/registration/log/v1`) the log is the commitment, its head the
digest, and the parameters the auditor's own: the composite and, for the hardware checks, its data sheets, the tolerance
it allows a card's clock and peak rates around its sheet's, and its devices' keys. `figures` is registration's check of
a device's registered figures against the auditor's estimates, `DeviceFigures.Accepted` over the SKU, the clock and the
peak rates, in integers.
=======
fails, in this order:

| Refusal | The check |
| --- | --- |
| `previous` | the entry follows the log's head |
| `duplicate-id` | no id is registered twice, `outside`'s included, and no card twice |
| `unknown-ref` | each new location's parent and each new link's ends are registered |
| `kind` | each new location sits under a parent its kind allows and has a tier exactly when it is memory; each new link joins two different ends of kinds a link may join, at a positive capacity; each certifier splits exactly one uplink |
| `card-digest` | each new card's SKU has an auditor's data sheet, and each new device names a registered card |
| `figures` | each new card's clock and peak rates are within the auditor's tolerance of its sheet's in both directions, its peak rates are positive, and its memory figures are its sheet's |
| `capacity` | each new memory leaf holds something, and each device's memory leaves fit in its card's memory |
| `owner` | each new certifier or challenger is one of the auditor's own devices with its key, and every other new location is the prover's |
| `uncertified` | onsite: every boundary link is certified |
| `challenger-reach` | onsite: the host of every memory leaf is at an end of a memory challenger's link |
| `duplicate-information` | no information item is registered twice |
| `duplicate-declaration` | no day is declared twice |

The hardware checks are an information entry's and a declaration's too, and hold of it whenever they held of the log
before it, which adds no hardware; so an information entry is held to `previous` and `duplicate-information` alone, and a
declaration to `previous` and `duplicate-declaration`.

A component reads its view of the log by projection (`View`): the site, θ's values, the builder and the days' declared
circuits, at the head. As a
core protocol (`protocol`, `verity/registration/log/v1`) the log is the commitment, its head the
digest, and the parameters the auditor's own: the composite and, for the hardware checks, its data sheets, the tolerance
it allows a card's clock and peak rates around its sheet's, and its devices' keys. `figures` is registration's check of
a device's registered figures against the auditor's estimates, `DeviceFigures.Accepted` over the SKU, the clock and the
peak rates, in integers.
>>>>>>> theirs
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Core.Registration

open Lean (Json)
open Flock (V)
open Core.Site.Registration (Composite Auditor)

def INFORMATION : String := "verity/registration/information/v1"
<<<<<<< ours
def DECLARATION : String := "verity/registration/declaration/v1"
||||||| base
def CARD : String := "verity/registration/card/v1"
def GENESIS : String := "verity/registration/genesis/v1"
=======
def DECLARATION : String := "verity/registration/declaration/v1"
def CARD : String := "verity/registration/card/v1"
def GENESIS : String := "verity/registration/genesis/v1"
>>>>>>> theirs
def protocolId : String := "verity/registration/log/v1"

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

<<<<<<< ours
/-- A day's declared circuit: the day, by its index among the monitoring period's days, the digest of the circuit's
program as hex, and its block Definitions by descriptor key. -/
structure Declared where
  day : Nat
  program : String
  blocks : List String
||||||| base
inductive Body
  | hardware (cards : List Card) (locs : List Loc) (links : List Link)
  | information (items : List Item)
=======
/-- A day's declared circuit: the day, by its index among the monitoring period's days, the digest of the circuit's
program as hex, and its block Definitions by descriptor key. -/
structure Declared where
  day : Nat
  program : String
  blocks : List String
  deriving DecidableEq, Repr

inductive Body
  | hardware (cards : List Card) (locs : List Loc) (links : List Link)
  | information (items : List Item)
  | declaration (declared : Declared)
>>>>>>> theirs
  deriving DecidableEq, Repr

/-- **An entry of the registration log**: the Site's site entry, an information entry naming the head of the log it
follows and the items it registers, or a declaration naming that head and the day's circuit. -/
inductive Entry
  | site (e : Site.Registration.Entry)
  | information (previous : String) (items : List Item)
  | declaration (previous : String) (declared : Declared)
  deriving DecidableEq

/-- The head of the log the entry follows. -/
def Entry.previous : Entry → String
  | .site e => e.previous
  | .information p _ => p
  | .declaration p _ => p

/-- **The registration log**, oldest entry first. -/
abbrev Log := List Entry

/-! ## Canonical JSON -/

def Item.toJson : Item → Json
  | .theta t => Json.mkObj ((t.checkpoint.map fun c => [("checkpoint_sha512", Json.str c)]).getD [] ++
      [("domain", .str t.domain), ("kind", .str "theta"), ("leaves", Lean.toJson t.leaves), ("root", .str t.root),
       ("value", .str t.value)] ++ (if t.wordBits == 16 then [] else [("word_bits", Lean.toJson t.wordBits)]) ++
      [("words", Lean.toJson t.words)])
  | .builder b => Json.mkObj [("advice", .str b.advice), ("cap", Lean.toJson b.cap), ("digest", .str b.digest),
      ("kind", .str "builder")]

<<<<<<< ours
def Declared.toJson (d : Declared) : Json :=
  Json.mkObj [("blocks", .arr (d.blocks.map Json.str).toArray), ("day", Lean.toJson d.day), ("program", .str d.program)]

def Entry.format : Entry → String
  | .site _ => Site.SITE
  | .information .. => INFORMATION
  | .declaration .. => DECLARATION
||||||| base
def Entry.format (e : Entry) : String :=
  match e.body with
  | .hardware .. => HARDWARE
  | .information _ => INFORMATION
=======
def Declared.toJson (d : Declared) : Json :=
  Json.mkObj [("blocks", .arr (d.blocks.map Json.str).toArray), ("day", Lean.toJson d.day), ("program", .str d.program)]

def Entry.format (e : Entry) : String :=
  match e.body with
  | .hardware .. => HARDWARE
  | .information _ => INFORMATION
  | .declaration _ => DECLARATION
>>>>>>> theirs

<<<<<<< ours
def Entry.toJson : Entry → Json
  | .site e => e.toJson
  | .information p items =>
    Json.mkObj [("format", .str INFORMATION), ("items", .arr (items.map Item.toJson).toArray), ("previous", .str p)]
  | .declaration p d =>
    Json.mkObj [("blocks", .arr (d.blocks.map Json.str).toArray), ("day", Lean.toJson d.day),
      ("format", .str DECLARATION), ("previous", .str p), ("program", .str d.program)]
||||||| base
def Entry.toJson (e : Entry) : Json :=
  match e.body with
  | .hardware cards locs links =>
    Json.mkObj [("cards", .arr (cards.map Card.toJson).toArray), ("format", .str HARDWARE),
      ("links", .arr (links.map Link.toJson).toArray), ("locs", .arr (locs.map Loc.toJson).toArray),
      ("previous", .str e.previous)]
  | .information items =>
    Json.mkObj [("format", .str INFORMATION), ("items", .arr (items.map Item.toJson).toArray),
      ("previous", .str e.previous)]
=======
def Entry.toJson (e : Entry) : Json :=
  match e.body with
  | .hardware cards locs links =>
    Json.mkObj [("cards", .arr (cards.map Card.toJson).toArray), ("format", .str HARDWARE),
      ("links", .arr (links.map Link.toJson).toArray), ("locs", .arr (locs.map Loc.toJson).toArray),
      ("previous", .str e.previous)]
  | .information items =>
    Json.mkObj [("format", .str INFORMATION), ("items", .arr (items.map Item.toJson).toArray),
      ("previous", .str e.previous)]
  | .declaration d =>
    Json.mkObj [("blocks", .arr (d.blocks.map Json.str).toArray), ("day", Lean.toJson d.day),
      ("format", .str DECLARATION), ("previous", .str e.previous), ("program", .str d.program)]
>>>>>>> theirs

/-- **An entry's digest**: `SHA-512(format ‖ 0 ‖ canon(entry))`, a site entry's the Site's. -/
def Entry.digest (e : Entry) : String := Site.tagged e.format e.toJson

/-- The head of a composite's empty log, the Site's. -/
def genesis (c : Composite) : String := Site.Registration.sha512.genesis c

/-- **The log's head**, which names it: its last entry's digest, or its composite's genesis when it is empty. -/
def head (c : Composite) (log : Log) : String :=
  match log.getLast? with
  | none => genesis c
  | some e => e.digest

/-- A site entry adding `s`, to chain. -/
def site (s : Site.Entry) : String → Entry := fun p => .site (.site p s)

/-- An information entry registering `items`, to chain. -/
def information (items : List Item) : String → Entry := fun p => .information p items

/-- A declaration of `d`, to chain. -/
def declaration (d : Declared) : String → Entry := fun p => .declaration p d

/-- The log of `entries` in order, each following the head of the log before it. -/
def chain (c : Composite) (entries : List (String → Entry)) : Log :=
  (entries.foldl (fun (acc : Log × String) f =>
    let e := f acc.2
    (acc.1 ++ [e], e.digest)) ([], genesis c)).1

/-! ## Decoding: exactly what `toJson` writes -/

open Core.Site (exactly)

def hex64 (j : Json) (what : String) : V String := do
  let s ← Flock.J.asStr j what
  let b ← (Flock.unhex s).mapError (s!"{what}: " ++ ·)
  if b.size != 64 then throw s!"{what}: not a SHA-512 digest (64 bytes)"
  pure s

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
  if format == Site.SITE then Entry.site <$> Site.Registration.Entry.ofJson j
  else if format == INFORMATION then
    exactly j ["format", "items", "previous"] "an information entry"
<<<<<<< ours
    pure (.information (← Flock.J.str j "previous" "an entry")
      (← (← Flock.J.arr j "items" "an information entry").toList.mapM Item.ofJson))
  else if format == DECLARATION then
    exactly j ["blocks", "day", "format", "previous", "program"] "a declaration"
    let previous ← Flock.J.str j "previous" "an entry"
    let day ← Flock.J.nat j "day" "a declaration"
    let program ← hex64 (← Flock.J.get j "program" "a declaration") "a declaration's program"
    let blocks ← (← Flock.J.arr j "blocks" "a declaration").toList.mapM (Flock.J.asStr · "a declaration's block")
    pure (.declaration previous { day, program, blocks })
  else throw s!"an entry: format {format} is not one of [{Site.SITE}, {INFORMATION}, {DECLARATION}]"
||||||| base
    let previous ← hex64 (← Flock.J.get j "previous" "an entry") "an entry's previous"
    pure { previous, body := .information (← (← Flock.J.arr j "items" "an information entry").toList.mapM Item.ofJson) }
  else throw s!"an entry: format {format} is not one of [{HARDWARE}, {INFORMATION}]"

def Composite.ofJson (j : Json) (what : String) : V Composite := byName (α := Composite) [.onsite, .remote] Composite.name j what
=======
    let previous ← hex64 (← Flock.J.get j "previous" "an entry") "an entry's previous"
    pure { previous, body := .information (← (← Flock.J.arr j "items" "an information entry").toList.mapM Item.ofJson) }
  else if format == DECLARATION then
    exactly j ["blocks", "day", "format", "previous", "program"] "a declaration"
    let previous ← hex64 (← Flock.J.get j "previous" "an entry") "an entry's previous"
    let day ← Flock.J.nat j "day" "a declaration"
    let program ← hex64 (← Flock.J.get j "program" "a declaration") "a declaration's program"
    let blocks ← (← Flock.J.arr j "blocks" "a declaration").toList.mapM (Flock.J.asStr · "a declaration's block")
    pure { previous, body := .declaration { day, program, blocks } }
  else throw s!"an entry: format {format} is not one of [{HARDWARE}, {INFORMATION}, {DECLARATION}]"

def Composite.ofJson (j : Json) (what : String) : V Composite := byName (α := Composite) [.onsite, .remote] Composite.name j what
>>>>>>> theirs

/-- The log as committed: its composite and its entries, `{"composite": NAME, "entries": [...]}`. -/
structure Committed where
  composite : Composite
  log : Log

def Committed.toJson (c : Committed) : Json :=
  Json.mkObj [("composite", .str c.composite.name), ("entries", .arr (c.log.map Entry.toJson).toArray)]

def Committed.ofJson (j : Json) : V Committed := do
  exactly j ["composite", "entries"] "the log"
  pure { composite := ← Site.Registration.Composite.ofJson (← Flock.J.get j "composite" "the log"),
         log := ← (← Flock.J.arr j "entries" "the log").toList.mapM Entry.ofJson }

/-! ## The check -/

/-- A refusal: a site entry's, by the Site's check that fails, `duplicate-information` or `duplicate-declaration`. -/
inductive Refusal
  | site (r : Site.Registration.Refusal)
  | duplicateInformation
  | duplicateDeclaration
  deriving DecidableEq, Repr

def Refusal.name : Refusal → String
  | .site r => r.name
  | .duplicateInformation => "duplicate-information"
  | .duplicateDeclaration => "duplicate-declaration"

def nodup : List String → Bool
  | [] => true
  | x :: xs => !xs.contains x && nodup xs

<<<<<<< ours
/-- The log so far, folded: its Site, its information items in order, its declarations in order, its head and its
length. -/
||||||| base
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
  else if !(cards.all (fun k => (A.sheet k.sku).isSome) &&
      locs.all fun x => x.kind != .device || (T.cards.map (·.digest)).contains x.card) then some .cardDigest
  else if !(cards.all fun k => (A.sheet k.sku).all (A.accepts · k)) then some .figures
  else if !(locs.all (fun x => x.kind != .memory || 0 < x.bytes) &&
      T.locs.all fun x => x.kind != .device || (T.card? x).all (T.held x.id ≤ ·.mem)) then some .capacity
  else if !(locs.all fun x =>
      if x.kind.auditor then x.owner == .auditor && A.key x.id == some x.key else x.owner == .prover) then some .owner
  else if c == .onsite && !(T.links.all fun k => !T.boundary k || T.certified k) then some .uncertified
  else if c == .onsite && !(T.locs.all fun x => x.kind != .memory || (T.host? x).all T.reached.contains) then
    some .challengerReach
  else none

/-- The log so far, folded: its site, its information items in order, its head and its length. -/
=======
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
  else if !(cards.all (fun k => (A.sheet k.sku).isSome) &&
      locs.all fun x => x.kind != .device || (T.cards.map (·.digest)).contains x.card) then some .cardDigest
  else if !(cards.all fun k => (A.sheet k.sku).all (A.accepts · k)) then some .figures
  else if !(locs.all (fun x => x.kind != .memory || 0 < x.bytes) &&
      T.locs.all fun x => x.kind != .device || (T.card? x).all (T.held x.id ≤ ·.mem)) then some .capacity
  else if !(locs.all fun x =>
      if x.kind.auditor then x.owner == .auditor && A.key x.id == some x.key else x.owner == .prover) then some .owner
  else if c == .onsite && !(T.links.all fun k => !T.boundary k || T.certified k) then some .uncertified
  else if c == .onsite && !(T.locs.all fun x => x.kind != .memory || (T.host? x).all T.reached.contains) then
    some .challengerReach
  else none

/-- The log so far, folded: its site, its information items in order, its declarations in order, its head and its
length. -/
>>>>>>> theirs
structure State where
  site : Site
  items : List Item
  declared : List Declared := []
  head : String
  entries : Nat

def State.empty (c : Composite) : State := { site := ⟨[], [], []⟩, items := [], head := genesis c, entries := 0 }

<<<<<<< ours
/-- The first check the entry `e` fails against the log folded to `s`, or `none`: a site entry's the Site's, under its
deployed digests. -/
def Entry.refusal (A : Auditor) (c : Composite) (s : State) : Entry → Option Refusal
  | .site e => (e.refusal Site.Registration.sha512 A c s.site s.head).map .site
  | .information p items =>
    if p ≠ s.head then some (.site .previous)
    else if nodup ((s.items ++ items).map Item.key) then none else some .duplicateInformation
  | .declaration p d =>
    if p ≠ s.head then some (.site .previous)
    else if s.declared.any (·.day == d.day) then some .duplicateDeclaration else none
||||||| base
/-- The first check the entry `e` fails against the log folded to `s`, or `none`. -/
def Entry.refusal (A : Auditor) (c : Composite) (s : State) (e : Entry) : Option Refusal :=
  if e.previous != s.head then some .previous
  else match e.body with
    | .hardware cards locs links => hardwareRefusal A c s.site cards locs links
    | .information items =>
      if nodup ((s.items ++ items).map Item.key) then none else some .duplicateInformation
=======
/-- The first check the entry `e` fails against the log folded to `s`, or `none`. -/
def Entry.refusal (A : Auditor) (c : Composite) (s : State) (e : Entry) : Option Refusal :=
  if e.previous != s.head then some .previous
  else match e.body with
    | .hardware cards locs links => hardwareRefusal A c s.site cards locs links
    | .information items =>
      if nodup ((s.items ++ items).map Item.key) then none else some .duplicateInformation
    | .declaration d => if s.declared.any (·.day == d.day) then some .duplicateDeclaration else none
>>>>>>> theirs

def State.add (s : State) (e : Entry) : State :=
<<<<<<< ours
  match e with
  | .site x => { s with site := s.site.add x.added, head := e.digest, entries := s.entries + 1 }
  | .information _ items => { s with items := s.items ++ items, head := e.digest, entries := s.entries + 1 }
  | .declaration _ d => { s with declared := s.declared ++ [d], head := e.digest, entries := s.entries + 1 }
||||||| base
  match e.body with
  | .hardware cards locs links => { s with site := s.site.add cards locs links, head := e.digest, entries := s.entries + 1 }
  | .information items => { s with items := s.items ++ items, head := e.digest, entries := s.entries + 1 }
=======
  match e.body with
  | .hardware cards locs links => { s with site := s.site.add cards locs links, head := e.digest, entries := s.entries + 1 }
  | .information items => { s with items := s.items ++ items, head := e.digest, entries := s.entries + 1 }
  | .declaration d => { s with declared := s.declared ++ [d], head := e.digest, entries := s.entries + 1 }
>>>>>>> theirs

/-- **The check**: the log folded entry by entry, each held to the log before it; or the first entry that fails, by its
index and the first check it fails. -/
def check (A : Auditor) (c : Composite) (log : Log) : Except (Nat × Refusal) State :=
  log.foldlM (fun s e => match e.refusal A c s with
    | some r => .error (s.entries, r)
    | none => .ok (s.add e)) (State.empty c)

/-! ## Views: what a component reads of the log -/

<<<<<<< ours
/-- What a component reads of an accepted log, at its head: the Site, θ's registered values in order, the builder, and
the days' declared circuits in order. -/
||||||| base
/-- What a component reads of an accepted log, at its head: the site, θ's registered values in order, and the builder. -/
=======
/-- What a component reads of an accepted log, at its head: the site, θ's registered values in order, the builder, and
the days' declared circuits in order. -/
>>>>>>> theirs
structure View where
  site : Site
  theta : List Theta
  builder : Option Builder
  declarations : List Declared := []
  head : String

def State.view (s : State) : View :=
  { site := s.site, head := s.head,
    theta := s.items.filterMap fun | .theta t => some t | _ => none,
    builder := s.items.findSome? fun | .builder b => some b | _ => none,
    declarations := s.declared }

/-- θ's registered value named `value`. -/
def View.value? (v : View) (value : String) : Option Theta := v.theta.find? (·.value == value)

/-- **Day `day`'s declared circuit**, the spec's `Verity.Registration.declared`: the one declaration of the day in an
accepted log, or `none` when the log declares no circuit for it. A window of that day runs its program. -/
def View.declared? (v : View) (day : Nat) : Option Declared := v.declarations.find? (·.day == day)

def View.toJson (v : View) : Json :=
<<<<<<< ours
  Json.mkObj [("builder", match v.builder with | some b => Item.toJson (.builder b) | none => .null),
    ("declarations", .arr (v.declarations.map Declared.toJson).toArray), ("head", .str v.head),
||||||| base
  Json.mkObj [("builder", optJson (Item.toJson ∘ .builder) v.builder), ("head", .str v.head),
=======
  Json.mkObj [("builder", optJson (Item.toJson ∘ .builder) v.builder),
    ("declarations", .arr (v.declarations.map Declared.toJson).toArray), ("head", .str v.head),
>>>>>>> theirs
    ("site", v.site.toJson), ("theta", .arr (v.theta.map (Item.toJson ∘ .theta)).toArray)]

/-! ## The core protocol -/

structure Params where
  composite : Composite
  auditor : Auditor

/-- Registration is ASCII without DEL, refused in `Core.canonicalJson`'s words: `j`'s strings, unescaped
(`Flock.canonUtf8`), hold no character from 0x7f up, so its digest is the same however a canonical JSON escapes the rest.
`Flock.canon` escapes such a character, so the bytes can't be the test: a `\u00e9` in the committed text is refused as
`é` is. -/
def ascii (j : Json) : V Unit :=
  if ((Flock.canonUtf8 j).toOption.getD "").all (fun (c : Char) => c.toNat < 0x7f) then pure ()
  else throw "not ASCII without DEL (a byte at or above 0x7f)"

def Params.ofJson (j : Json) : V Params := do
  ascii j
  exactly j ["auditor", "composite"] "the params"
  pure { composite := ← Site.Registration.Composite.ofJson (← Flock.J.get j "composite" "the params"),
         auditor := ← Site.Registration.Auditor.ofJson (← Flock.J.get j "auditor" "the params") }

/-- The log as committed, and its head as bytes. -/
structure Commitment where
  committed : Committed
  head : ByteArray

def commitmentOf (b : ByteArray) : V Commitment := do
  let j ← canonicalJson b
  ascii j
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
<<<<<<< ours
(`{"auditor": AUDITOR, "composite"}`, the auditor as `Core.Site.Registration.Auditor.ofJson` reads it), the commitment
is the log (`{"composite", "entries": [ENTRY]}`) with its head as digest, the evidence is `{}`. It accepts a log the
check passes, making public its view (`View.toJson`: its Site, θ, builder, declarations and head) and keeping its length, and rejects the first entry the check
refuses as `entry I: REFUSAL`, and a log or parameters outside ASCII without DEL as not ASCII. It is never undecided. -/
||||||| base
(`{"composite", "keys": [{"id", "key"}], "sheets": [CARD], "tolerance": {"den", "num"}}`), the commitment is the log
(`{"composite", "entries": [ENTRY]}`) with its head as digest, the evidence is `{}`. It accepts a log the check passes,
making public its view (`View.toJson`) and keeping its length, and rejects the first entry the check refuses as
`entry I: REFUSAL`, and a log or parameters outside ASCII as `ascii`. It is never undecided. -/
=======
(`{"composite", "keys": [{"id", "key"}], "sheets": [CARD], "tolerance": {"den", "num"}}`), the commitment is the log
(`{"composite", "entries": [ENTRY]}`) with its head as digest, the evidence is `{}`. It accepts a log the check passes,
making public its view (`View.toJson`: its site, θ, builder, declarations and head) and keeping its length, and rejects
the first entry the check refuses as
`entry I: REFUSAL`, and a log or parameters outside ASCII as `ascii`. It is never undecided. -/
>>>>>>> theirs
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

open Core.Site (Sheet Loc Link Figure outside)

/-- The figure `n / d`. -/
def fig (n : Nat) (d : Nat := 1) : Figure := if h : 0 ≤ mkRat n d then ⟨_, h⟩ else ⟨0, by decide⟩

def sheet : Sheet :=
  { figures := { sku := "sim-a", clock := fig 2000000000,
                 peakRate := fun
                   | .uniformS8 => fig 1000000000000000 | .uniformS2 => fig 2000000000000000
                   | .uniformU2 => fig 2000000000000000 | .uniformU4 => fig 1500000000000000
                   | .zero => fig 4000000000000000 },
    bw := 3200, chip := 1 <<< 25, mem := 64 <<< 30 }

/-- `sheet` with its clock at `n` Hz. -/
def clocked (n : Nat) : Sheet := { sheet with figures := { sheet.figures with clock := fig n } }

/-- `sheet` with its peak rate of class `c` at `n`. -/
def rated (c : Site.OperandClass) (n : Nat) : Sheet :=
  { sheet with figures := { sheet.figures with peakRate := fun x => if x = c then fig n else sheet.figures.peakRate x } }

def keys : List (String × String) := [("cert0", "k0"), ("chal0", "k1")]

def auditor : Auditor :=
  { sheet := fun sku => [sheet].find? (·.sku == sku), tolerance := fig 1 20, key := fun i => (keys.find? (·.1 == i)).map (·.2) }

def hex (c : Char) : String := String.ofList (List.replicate 128 c)

def locs : List Loc :=
  [{ id := "site0", kind := .site }, { id := "node0", kind := .host, parent := some "site0" },
   { id := "gpu0", kind := .device, parent := some "node0", sheet := Site.Registration.sha512.sheet sheet },
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

<<<<<<< ours
/-- Day 0's circuit: its program's digest and its one block Definition. -/
def day0 : Declared := { day := 0, program := hex '1', blocks := ["block0"] }

def entries : List (String → Entry) := [site ⟨[sheet], locs, links⟩, information items, declaration day0]
||||||| base
def bodies : List Body := [.hardware [card] locs links, .information items]
=======
/-- Day 0's circuit: its program's digest and its one block Definition. -/
def day0 : Declared := { day := 0, program := hex '1', blocks := ["block0"] }

def bodies : List Body := [.hardware [card] locs links, .information items, .declaration day0]
>>>>>>> theirs

/-- A site entry adding only `k`. -/
def stating (k : Sheet) : List (String → Entry) := [site ⟨[k], [], []⟩]

def refusal (c : Composite) (entries : List (String → Entry)) (A : Auditor := auditor) : Option (Nat × Refusal) :=
  match check A c (chain c entries) with
  | .ok _ => none
  | .error r => some r

/-- A tolerance of `n / d` around the auditor's sheets. -/
def at_ (n : Nat) (d : Nat := 1) : Auditor := { auditor with tolerance := fig n d }

#guard refusal .onsite entries == none
#guard ((check auditor .onsite (chain .onsite entries)).toOption.map (·.view.value? "weights")) == some (some weights)
#guard (chain .onsite entries).all fun e => match Entry.ofJson e.toJson with | .ok e' => e' == e | .error _ => false
#guard head .onsite (chain .onsite entries) != head .remote (chain .remote entries)
-- a log of site entries alone is the Site's log, with the Site's head
#guard head .onsite (chain .onsite [site ⟨[sheet], locs, links⟩]) ==
  Site.Registration.head Site.Registration.sha512 .onsite [.site (genesis .onsite) ⟨[sheet], locs, links⟩]
-- a log chained for remote is not onsite's
#guard (match check auditor .onsite (chain .remote entries) with | .error r => some r | .ok _ => none) ==
  some (0, .site .previous)
-- onsite: an uplink no certifier splits; remote takes it
private def bare : List (String → Entry) :=
  [site ⟨[sheet], locs.take 4, [{ id := "up0", ends := ("node0", outside), β := 100, latency := 1 }]⟩]
#guard refusal .onsite bare == some (0, .site .uncertified)
#guard refusal .remote bare == none
-- onsite: memory no challenger reaches
#guard refusal .onsite [site ⟨[sheet], locs.take 5, links.take 2⟩] == some (0, .site .challengerReach)
-- a sheet with no data sheet, a device's memory over its sheet's, a certifier without its key
#guard refusal .remote (stating { sheet with figures := { sheet.figures with sku := "sim-b" } }) ==
  some (0, .site .sheetDigest)
#guard refusal .remote bare { auditor with sheet := fun _ => none } == some (0, .site .sheetDigest)
-- a clock or a peak rate stated five percent off is within a twentieth either way; past it, slow or fast, is refused
#guard refusal .remote (stating (clocked 1900000000)) == none
private def fast : Sheet :=
  { sheet with figures := { (clocked 2100000000).figures with peakRate := (rated .zero 3800000000000000).figures.peakRate } }
#guard refusal .remote (stating fast) == none
#guard refusal .remote (stating (clocked 1899999999)) == some (0, .site .figures)
#guard refusal .remote (stating (rated .uniformS2 2100000000000001)) == some (0, .site .figures)
#guard refusal .remote (stating { sheet with mem := 1 }) == some (0, .site .figures)
-- at tolerance 0 only the sheet's figures; at 1 or more nothing below is refused
#guard refusal .remote (stating (clocked 1999999999)) (at_ 0) == some (0, .site .figures)
#guard refusal .remote (stating (clocked 0)) (at_ 1) == none
#guard refusal .remote (stating (clocked 4000000001)) (at_ 1) == some (0, .site .figures)
-- a zero peak rate is refused even where the tolerance allows it
#guard refusal .remote (stating (rated .uniformU4 0)) (at_ 1) == some (0, .site .figures)
private def over : Loc := { id := "hbm0", kind := .memory, parent := some "gpu0", tier := some .hbm, bytes := (64 <<< 30) + 1 }
#guard refusal .remote [site ⟨[sheet], locs.take 3 ++ [over], []⟩] == some (0, .site .capacity)
#guard refusal .onsite entries { auditor with key := fun _ => some "k1" } == some (0, .site .owner)
-- an id twice, a parent nobody registered, a host under a host
#guard refusal .remote [site ⟨[sheet], locs.take 2, []⟩, site ⟨[], locs.drop 1 |>.take 1, []⟩] == some (1, .site .duplicateId)
#guard refusal .remote [site ⟨[], [{ id := "node1", kind := .host, parent := some "site9" }], []⟩] ==
  some (0, .site .unknownRef)
#guard refusal .remote [site ⟨[], locs.take 2 ++ [{ id := "node1", kind := .host, parent := some "node0" }], []⟩] ==
  some (0, .site .kind)
-- θ's weights registered twice, and a second builder; another θ value is a new item
<<<<<<< ours
#guard refusal .onsite (entries ++ [information (items.take 1)]) == some (3, .duplicateInformation)
#guard refusal .onsite (entries ++ [information (items.drop 2)]) == some (3, .duplicateInformation)
#guard refusal .onsite (entries ++ [information [.theta { weights with value := "tables" }]]) == none
-- day 0 declared twice, even with the same circuit; day 1 is a new day; the view reads each day's circuit
#guard refusal .onsite (entries ++ [declaration { day0 with program := hex '2' }]) == some (3, .duplicateDeclaration)
#guard refusal .onsite (entries ++ [declaration day0]) == some (3, .duplicateDeclaration)
#guard refusal .onsite (entries ++ [declaration { day0 with day := 1, program := hex '2' }]) == none
#guard ((check auditor .onsite (chain .onsite entries)).toOption.map fun s => (s.view.declared? 0, s.view.declared? 1)) ==
  some (some day0, none)
#guard (match Entry.ofJson (Json.mkObj [("blocks", .arr #[]), ("day", Lean.toJson (0 : Nat)),
    ("format", .str DECLARATION), ("previous", .str (hex 'a')), ("program", .str "1")]) with
  | .ok _ => false | .error _ => true)
||||||| base
#guard refusal .onsite (bodies ++ [.information (items.take 1)]) == some (2, .duplicateInformation)
#guard refusal .onsite (bodies ++ [.information (items.drop 2)]) == some (2, .duplicateInformation)
#guard refusal .onsite (bodies ++ [.information [.theta { weights with value := "tables" }]]) == none
=======
#guard refusal .onsite (bodies ++ [.information (items.take 1)]) == some (3, .duplicateInformation)
#guard refusal .onsite (bodies ++ [.information (items.drop 2)]) == some (3, .duplicateInformation)
#guard refusal .onsite (bodies ++ [.information [.theta { weights with value := "tables" }]]) == none
-- day 0 declared twice, even with the same circuit; day 1 is a new day; the view reads each day's circuit
#guard refusal .onsite (bodies ++ [.declaration { day0 with program := hex '2' }]) == some (3, .duplicateDeclaration)
#guard refusal .onsite (bodies ++ [.declaration day0]) == some (3, .duplicateDeclaration)
#guard refusal .onsite (bodies ++ [.declaration { day0 with day := 1, program := hex '2' }]) == none
#guard ((check auditor .onsite (chain .onsite bodies)).toOption.map fun s => (s.view.declared? 0, s.view.declared? 1)) ==
  some (some day0, none)
#guard (match Entry.ofJson (Json.mkObj [("blocks", .arr #[]), ("day", Lean.toJson (0 : Nat)),
    ("format", .str DECLARATION), ("previous", .str (hex 'a')), ("program", .str "1")]) with
  | .ok _ => false | .error _ => true)
>>>>>>> theirs

private def bytes (j : Json) : ByteArray := ((Flock.canon j).toOption.getD "").toUTF8

private def params (c : Composite) : Json :=
  Json.mkObj [("auditor", Json.mkObj [("keys", .arr #[Json.mkObj [("id", "cert0"), ("key", "k0")],
      Json.mkObj [("id", "chal0"), ("key", "k1")]]), ("sheets", .arr #[sheet.toJson]), ("tolerance", "1/20")]),
    ("composite", .str c.name)]

private def answer (c : Composite) (log : Committed) : Answer :=
  protocol.answer (bytes (params c)) (bytes log.toJson) "{}".toUTF8

#guard (answer .onsite { composite := .onsite, log := chain .onsite entries }).verdict == .accept
#guard (answer .onsite { composite := .onsite, log := chain .onsite entries }).digest ==
  (Flock.unhex (head .onsite (chain .onsite entries))).toOption
#guard (answer .onsite { composite := .remote, log := chain .remote entries }).reason ==
  some "composite: the log is remote's, not onsite's"
#guard (answer .onsite { composite := .onsite, log := chain .onsite bare }).reason == some "entry 0: uncertified"
private def accented : Committed :=
  { composite := .onsite, log := chain .onsite [information [.theta { weights with value := "poids\u00e9" }]] }
-- a log or parameters outside ASCII without DEL is refused before any entry is read (`ascii`)
#guard (protocol.answer (bytes (params .onsite)) accented.toJson.compress.toUTF8 "{}".toUTF8).reason ==
  some "commitment: not ASCII without DEL (a byte at or above 0x7f)"
private def deleted : Committed :=
  { composite := .onsite, log := chain .onsite [information [.theta { weights with value := "poids\x7f" }]] }
#guard (protocol.answer (bytes (params .onsite)) deleted.toJson.compress.toUTF8 "{}".toUTF8).reason ==
  some "commitment: not ASCII without DEL (a byte at or above 0x7f)"
-- escaped (`Flock.canon`, `\u00e9`) or as itself in UTF-8 (`Flock.canonUtf8`), it is refused the same
private def raw (j : Json) : ByteArray := ((Flock.canonUtf8 j).toOption.getD "").toUTF8
#guard [accented, deleted].all fun l => [bytes l.toJson, raw l.toJson].all fun b =>
  (protocol.answer (bytes (params .onsite)) b "{}".toUTF8).reason ==
    some "commitment: not ASCII without DEL (a byte at or above 0x7f)"

end Example

end Core.Registration
