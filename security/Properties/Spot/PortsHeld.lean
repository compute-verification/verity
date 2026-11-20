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

/-- **`r` is the committed record `rec`'s root for the port `q`**: the record's `roots` hold exactly one root naming `q`
(R3: no port twice), and `r` has its root, domain and leaves. -/
def IsRecordRoot (rec : Json) (q : String) (r : Json) : Prop :=
  ∃ r0 ∈ Properties.Spot.rootsOf rec, getD r0 "port" = .str q ∧ (∀ r1 ∈ Properties.Spot.rootsOf rec, getD r1 "port" = .str q → r1 = r0) ∧
    getD r "root" = getD r0 "root" ∧ getD r "domain" = getD r0 "domain" ∧ getD r "leaves" = getD r0 "leaves"

/-- **What the per-member form's check accepts reads each member's slice root as the record's root for its port**:
every root of every member's slice (`OneStageZk.Member.slice`, at the roots `OneStageZk.membersRootsOf` returned) is
the committed record's root for a port `q` (`IsRecordRoot`): the member's own port `<member>/<port>`, the tree the
slice root names, or a window's tree of a template's field `<template>/<field>`, which a wired port reads. -/
def MembersRootsRecord : Prop :=
  ∀ (p : OneStageZk.Params) (ms : Array OneStageZk.Member) (c : OneStage.Commitment) (ξ : ByteArray)
    (e : OneStageZk.Evidence) (sts : Array OneStageZk.StatementFiles) (bs : Core.Blobs) (pub : OneStage.Public)
    (prof : Core.Integrity),
    OneStageZk.checkMembersAt p ms c ξ e sts bs = .accept pub prof →
      ∃ roots, OneStageZk.membersRootsOf p ms c.record = .ok roots ∧
        ∀ m ∈ ms, ∀ r ∈ (match getD (m.slice c.record roots) "roots" with | .arr rs => rs | _ => #[]),
          ∃ q, IsRecordRoot c.record q r ∧
            ((∃ b, getD r "port" = .str b ∧ q = m.name ++ "/" ++ b) ∨ getD r "tree" = .str q ∨
              ∃ t f, OneStageZk.templateOf q = some (t, f))

end Properties.Spot
