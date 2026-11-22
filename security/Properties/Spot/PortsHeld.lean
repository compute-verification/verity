module

public import Spot.OneStageZk
public import Properties.Spot

/-!
# The per-member form holds every port of every member's statement (`Spot.OneStageZk.headerHeldAt`)

What the per-member form's check accepts (`Spot.OneStageZk.checkMembersAt`) has every member's files read, sampled or
not (`OneStageZk.membersRun_files`), each at its slice of the registration (`OneStageZk.Member.slice`, at the roots
`OneStageZk.membersRootsOf` derives from the parameters and the record), and the population check of its statement
holds its public file's header there (`OneStageZk.headerHeldAt`). `MembersPortsHeld`: every port of that header's
`frame_v3` roots is held (`PortHeld`), in one of three ways:

* **at a shared tree, by a registered read**: the slice holds the port at a tree (its entry's `tree`) and the member's
  registered reads name the port at that tree;
* **at a registered value**: the slice holds no root of the port, and the member's registered reads name the port at a
  value the verifier's registered file holds at it as a registered value (`OneStageZk.registeredAt`);
* **at the slice's root**: the one root of the slice naming the port is the header's root there, and its domain the
  header's domain there. That the slice names each port at most once is the members check's refusal
  (`OneStageZk.Member.portsOnce`, `OneStageZk.membersRootsOf_portsOnce`).

It names the read that holds a port of the first two kinds and does not say that the read opened: that each row of it
opens under the registration's root at the leaf C gives is the session's (`Flock.Registered.checkPort`), which runs
only for a statement the sample holds. It holds a port to the slice's root, which the verifier computes from the
parameters and the committed record alone (`OneStageZk.membersRootsOf p ms c.record`), reading nothing of the
evidence; no theorem yet states which record root each one is.
-/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot

/-- **A port `p` of a member's public file's header `h`, at the root `x` there, is held** at the member's slice `slice`
of the registration, at its registered reads `reads` and the verifier's registered file's ports `held`: by a registered
read at the slice's tree of the port, at a registered value where the slice holds no root of it, or at the one root
the slice names it at and that root's domain. -/
def PortHeld (reads : Array (String × String)) (held : Array Flock.Registered.Port) (slice h : Json) (p : String)
    (x : Json) : Prop :=
  (∃ r ∈ (match getD slice "roots" with | .arr rs => rs | _ => #[]), getD r "port" = .str p ∧
    ∃ t, getD r "tree" = .str t ∧ (p, t) ∈ reads) ∨
  ((∀ q ∈ Record.byPort slice, q.1 ≠ p) ∧ ∃ v, (p, v) ∈ reads ∧ OneStageZk.registeredAt held p v = true) ∨
  ∃ r ∈ (match getD slice "roots" with | .arr rs => rs | _ => #[]), getD r "port" = .str p ∧
    (∀ r' ∈ (match getD slice "roots" with | .arr rs => rs | _ => #[]), getD r' "port" = .str p → r' = r) ∧
    jeq (getD r "root") x = true ∧ jeq (getD (getD (getD h "frame_v3") "domain_ids") p) (getD r "domain") = true

/-- **Every port of the header `h`'s `frame_v3` roots is held** (`PortHeld`). -/
def PortsHeld (reads : Array (String × String)) (held : Array Flock.Registered.Port) (slice h : Json) : Prop :=
  ∀ pr ∈ entries (getD (getD h "frame_v3") "roots"), PortHeld reads held slice h pr.1 pr.2

/-- **What the per-member form's check accepts holds every port of every member's statement**: every port of every
member's statement is held to a root the verifier computes from the committed record (`membersRootsOf`); no theorem
yet states which record root each one is. The verifier's roots of the members (`OneStageZk.membersRootsOf`) returned,
and every member of the parameters, sampled or not, has its statement read at its slice of the registration, the
header of that statement's public file parsed, its registered reads and the verifier's registered file's ports read,
and every port of the header held there (`PortsHeld`). -/
def MembersPortsHeld : Prop :=
  ∀ (p : OneStageZk.Params) (ms : Array OneStageZk.Member) (c : OneStage.Commitment) (ξ : ByteArray)
    (e : OneStageZk.Evidence) (sts : Array OneStageZk.StatementFiles) (bs : Core.Blobs) (pub : OneStage.Public)
    (prof : Core.Integrity),
    OneStageZk.checkMembersAt p ms c ξ e sts bs = .accept pub prof →
      ∃ roots, OneStageZk.membersRootsOf p ms c.record = .ok roots ∧
        ∀ m ∈ ms, ∃ a hd reads held, m.statement (m.slice c.record roots) bs = .ok a ∧
          FilePart.header a.I.publicFile = .ok hd ∧ m.reads = .ok reads ∧ m.held = .ok held ∧
          PortsHeld reads held (m.slice c.record roots) hd

/-- Parameters naming no members, a default program and no ports: what the per-member form reads of them before
the registration's roots is nothing. -/
def bareVerifier : Docket.Verifier where
  program := default
  query := .null
  samplingDesign := .null
  workTable := none
  scheme := ""
  ports := #[]
  leafLayer := none
  anchors := #[]
  log := #[]
  partition := none

/-- The parameters `MembersPortsHeld`'s nontrivial lemma refuses at: `bareVerifier`'s, with no own program and no
members. -/
def bareParams : OneStageZk.Params where
  base := { verifier := bareVerifier, delta := default }
  own := { programJ := .null, program := ByteArray.empty, partition := .null, pub := "", registered := none }
  members := none

/-- **`r` is the committed record `rec`'s root for the port `q`**: the record's `roots` hold exactly one root naming `q`
(R3: no port twice), and `r` has its root, domain and leaves. -/
def IsRecordRoot (rec : Json) (q : String) (r : Json) : Prop :=
  ∃ r0 ∈ Properties.Spot.rootsOf rec, getD r0 "port" = .str q ∧ (∀ r1 ∈ Properties.Spot.rootsOf rec, getD r1 "port" = .str q → r1 = r0) ∧
    getD r "root" = getD r0 "root" ∧ getD r "domain" = getD r0 "domain" ∧ getD r "leaves" = getD r0 "leaves"

/-- **The wire `w` of C's wiring under the verifier's partition is one into the member `m`**, as the verifier's own
check has it (`OneStageZk.wireReaches`): its reader is the template of a range of the partition's ranges `rs` that holds
`m`'s units, and its unit window, in that template's own units (`OneStageZk.localBases`), overlaps `m`'s. -/
def WireInto (rs : Array Flock.TemplateQuery.TRange) (m : OneStageZk.Member) (w : Spot.Wires.Wire) : Prop :=
  let rs' := rs.map fun r => (r.template, r.first, r.count)
  OneStageZk.wireReaches rs' (OneStageZk.localBases rs') m w = true

/-- **What the per-member form's check accepts reads each member's slice root as the record's root for its port**:
every root of every member's slice (`OneStageZk.Member.slice`, at the roots `OneStageZk.membersRootsOf` returned) is
the committed record's root for a port `q` (`IsRecordRoot`): the member's own port `<member>/<port>`, the tree the
slice root names, or, for a wired port, `<writer>/<field>`, the window's tree of the exact output field that a wire
of C's wiring under the verifier's partition (`OneStageZk.derivedWiring`) into the member (`WireInto`) at that port
names. Only the verifier writes a slice root's `tree`, and only two kinds: the window's tree `<template>/<field>` of an
output field of the member's own template (`OneStageZk.fieldTreesOf`, as `OneStageZk.ownFieldsOf` adds it), and
`input/n<w>` of a run input node `w` the member's Definition reads in C (`Flock.Registered.freshValue`, as
`OneStageZk.runInputRootsAt` adds it); the member's own roots and its wired roots carry none (`OneStageZk.rootStep`,
`OneStageZk.fieldRootIn`). -/
def MembersRootsRecord : Prop :=
  ∀ (p : OneStageZk.Params) (ms : Array OneStageZk.Member) (c : OneStage.Commitment) (ξ : ByteArray)
    (e : OneStageZk.Evidence) (sts : Array OneStageZk.StatementFiles) (bs : Core.Blobs) (pub : OneStage.Public)
    (prof : Core.Integrity),
    OneStageZk.checkMembersAt p ms c ξ e sts bs = .accept pub prof →
      ∃ roots, OneStageZk.membersRootsOf p ms c.record = .ok roots ∧
        ∀ m ∈ ms, ∀ r ∈ (match getD (m.slice c.record roots) "roots" with | .arr rs => rs | _ => #[]),
          ∃ q, IsRecordRoot c.record q r ∧
            ((∃ b, getD r "port" = .str b ∧ q = m.name ++ "/" ++ b) ∨ getD r "tree" = .str q ∨
              ∃ P rs W, Program.parse p.own.program = .ok P ∧
                Partition.evaluate P (getD p.own.partition "query") = .ok rs ∧
                OneStageZk.derivedWiring P.prog rs = .ok W ∧
                ∃ w ∈ W.wires, WireInto rs m w ∧ getD r "port" = .str w.port ∧
                  q = w.writer ++ "/" ++ w.out)

/-- **On an accepted run of the per-member variant, every untied run input a member reads is at the record's one
tree**: `OneStageZk.memberInputs` succeeds for every member, and for each parameter `port` member `m`'s units read
there at a run input node `w` of C outside the tied values' gates (`Docket.tiedGates`), `m`'s slice holds a root under `port` at the tree `input/n<w>`
(`Flock.Registered.freshValue`) that is the record's one root of that tree (`IsRecordRoot`). So every proof block that
reads an untied run input node opens one root of it. -/
def RunInputsAtOneTree : Prop :=
  ∀ (p : OneStageZk.Params) (ms : Array OneStageZk.Member) (c : OneStage.Commitment) (ξ : ByteArray)
    (e : OneStageZk.Evidence) (sts : Array OneStageZk.StatementFiles) (bs : Core.Blobs) (pub : OneStage.Public)
    (prof : Core.Integrity),
    OneStageZk.checkMembersAt p ms c ξ e sts bs = .accept pub prof →
      ∃ roots, OneStageZk.membersRootsOf p ms c.record = .ok roots ∧
        ∃ P rs, Program.parse p.own.program = .ok P ∧ Partition.evaluate P (getD p.own.partition "query") = .ok rs ∧
          ∀ m ∈ ms, ∃ ins, OneStageZk.memberInputs P.prog rs m = .ok ins ∧
            ∀ (port : String) (w len : Nat), (port, w, len) ∈ ins → Flock.Registered.runNode P.prog w = true →
              ((Docket.tiedGates p.base.verifier).any fun (first, count) => Partition.within P.prog first count w) =
                false →
              ∃ r ∈ (match getD (m.slice c.record roots) "roots" with | .arr rs => rs | _ => #[]),
                getD r "port" = .str port ∧ getD r "tree" = .str (Flock.Registered.freshValue w) ∧
                IsRecordRoot c.record (Flock.Registered.freshValue w) r

/-- **Where `OneStageZk.runInputRoots` passes, every reader of a tied run input node brings a root, and those roots are
one root**: for members of distinct names and a run input node `w` of C inside the tied values' gates, there is one
root `r0` such that each member holds a root of its own (`roots`, as `runInputRoots` is handed it) at each port it reads
`w` at (`OneStageZk.memberInputs`; the first such entry, as `OneStageZk.tiedRootsAt` reads it;
`OneStageZk.TIED_NO_ROOT_WHY` otherwise), and that root is `r0` or `==` to it (`OneStageZk.TIED_ROOTS_WHY` otherwise;
core's `==` on `Json` is opaque to the kernel, so not `=`). That `r0` is the right root is not said here. -/
def TiedInputsAtOneRoot : Prop :=
  ∀ (v : Docket.Verifier) (P : Flock.Program.Prog) (rs : Array Flock.TemplateQuery.TRange)
    (ms : Array OneStageZk.Member) (rec : Json) (roots out : Std.HashMap String (Array Json)),
    (ms.toList.map (·.name)).Nodup → OneStageZk.runInputRoots v P rs ms rec roots = .ok out →
    ∀ w, Flock.Registered.runNode P w = true →
      ((Docket.tiedGates v).any fun (first, count) => Partition.within P first count w) = true →
      ∃ r0 : Json, ∀ m ∈ ms, ∀ ins, OneStageZk.memberInputs P rs m = .ok ins →
        ∀ (port : String) (len : Nat), (port, w, len) ∈ ins → ∃ r : Json,
          ((roots.getD m.name #[]).find? (getD · "port" == .str port)).map (getD · "root") = some r ∧
            (r0 = r ∨ (r0 == r) = true)

end Properties.Spot
