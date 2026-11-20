module

public import Spot.OneStageZk

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
  header's domain there. That the slice names each port at most once is `MembersPortsOnce`, which the proof takes as
  a hypothesis.

It names the read that holds a port of the first two kinds and does not say that the read opened: that each row of it
opens under the registration's root at the leaf C gives is the session's (`Flock.Registered.checkPort`), which runs
only for a statement the sample holds. It holds a port to the slice's root. The slice's roots are
`OneStageZk.membersRootsOf p ms c.record`'s, a function of the parameters and the record that reads nothing of the
evidence. Each of them copies one root of the record (its `root`, `domain` and `leaves`) under a port name: `rootStep`
copies a member's own, `fieldTreesOf` the window's field trees (`ownFieldsOf`, `fieldRoots`), and `runInputRootsAt` a
run input node's tree (`rootsByPort`). No theorem states that copy; the code of `membersRootsOf` is the only tie.
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

/-- The ports the slice `slice`'s roots name, as strings, in its roots' order. -/
def slicePortsOf (slice : Json) : List String :=
  ((match getD slice "roots" with | .arr rs => rs | _ => #[]).filterMap fun r =>
    match getD r "port" with
    | .str q => some q
    | _ => none).toList

/-- **Where the verifier's roots of the members return, each member's slice names each port at most once**, ports
compared as strings. -/
def MembersPortsOnce : Prop :=
  ∀ (p : OneStageZk.Params) (ms : Array OneStageZk.Member) (rec : Json) (roots : Std.HashMap String (Array Json)),
    OneStageZk.membersRootsOf p ms rec = .ok roots → ∀ m ∈ ms, (slicePortsOf (m.slice rec roots)).Nodup

/-- **What the per-member form's check accepts holds every port of every member's statement**: the verifier's roots
of the members (`OneStageZk.membersRootsOf`) returned, and every member of the parameters, sampled or not, has its
statement read at its slice of the registration, the header of that statement's public file parsed, its registered
reads and the verifier's registered file's ports read, and every port of the header held there (`PortsHeld`). -/
def MembersPortsHeld : Prop :=
  ∀ (p : OneStageZk.Params) (ms : Array OneStageZk.Member) (c : OneStage.Commitment) (ξ : ByteArray)
    (e : OneStageZk.Evidence) (sts : Array OneStageZk.StatementFiles) (bs : Core.Blobs) (pub : OneStage.Public)
    (prof : Core.Integrity),
    OneStageZk.checkMembersAt p ms c ξ e sts bs = .accept pub prof →
      ∃ roots, OneStageZk.membersRootsOf p ms c.record = .ok roots ∧
        ∀ m ∈ ms, ∃ a hd reads held, m.statement (m.slice c.record roots) bs = .ok a ∧
          FilePart.header a.I.publicFile = .ok hd ∧ m.reads = .ok reads ∧ m.held = .ok held ∧
          PortsHeld reads held (m.slice c.record roots) hd

end Properties.Spot
