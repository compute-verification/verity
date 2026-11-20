module

public import Spot.OneStageZkStatement

/-!
# One-stage v2: the compiled files' pass, the members' names and slice ports once, and a wire into a member

Between `Spot/OneStageZkStatement.lean` and `Spot/OneStageZk.lean`, in the namespace `Spot.OneStageZk`:

* the files' pass with each circuit file parsed once (`membersFilesKept`), compiled in `membersFiles`' place
  (`membersFiles_eq_kept`);
* a member's slice naming each port once (`Member.portsOnce`, `portsOnceAll`), and the members naming each name once
  (`namesOnce`, at `portTwice`);
* whether a wire of C's wiring enters a member (`wireReaches`, at each range's base among its template's units,
  `localBases`), which `fieldRoots` reads.
-/

@[expose] public section

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

/-- **Every member's files, a circuit file read and parsed once for the members that name it**, and the program's
strata evaluated once for the members that read it: `keptFiles` from no file and no program kept, at the statement's
tags and the blobs' lookup tables, each read once. It returns what `membersFiles` returns, a refusal included
(`membersFiles_eq_kept`), and is compiled in its place. -/
def membersFilesKept (slice : Member → Json) (bs : Core.Blobs) (ms : List Member) : V (List (String × StepAt)) :=
  match Flock.Tags.byName? STATEMENT with
  | some tags0 => keptFiles slice bs tags0 (tablesOf bs) none ms
  | none => membersFiles slice bs ms

/-- **The files' pass with each circuit file read and parsed once is the files' pass**: `membersFilesKept` returns
exactly what `membersFiles` returns, a statement for each member or the first refusal, so every statement about
`membersFiles` is one about the compiled pass. -/
@[csimp] theorem membersFiles_eq_kept : @membersFiles = @membersFilesKept := by
  funext slice bs ms
  unfold membersFilesKept
  cases ht : Flock.Tags.byName? STATEMENT with
  | none => rfl
  | some tags0 => exact (keptFiles_eq ht slice bs ms none (fun _ h => nomatch h)).symm

/-- The refusal of a member's slice holding a root that names no port. -/
def SLICE_NO_PORT_WHY (member : String) : String :=
  s!"member {member}: a root of its slice names no port"

/-- The refusal of a member's slice whose roots name its port `p` twice (`Member.portsOnce`). -/
def PORT_TWICE_WHY (member p : String) : String :=
  s!"member {member}: its slice names port {p} twice: a member's slice holds one root of each port, the one its " ++
    "public file's root there is held to"

/-- The ports of the slice `s`'s roots, each its string, in the slice's order; none where the roots are not a list or
one names no port as a string. -/
def slicePorts (s : Json) : Option (List String) :=
  match getD s "roots" with
  | .arr rs => rs.toList.mapM fun r => match getD r "port" with
    | .str p => some p
    | _ => none
  | _ => none

/-- The first port of `ps` a later one names again, the two compared as strings; none where each is named once. -/
def portTwice : List String → Option String
  | [] => none
  | p :: ps => if p ∈ ps then some p else portTwice ps

/-- **A member's slice `s` names each port once**, its ports compared as strings (`slicePorts`): a root that names no
port refuses (`SLICE_NO_PORT_WHY`), and so does a port named twice (`PORT_TWICE_WHY`), whatever the two roots are. -/
def Member.portsOnce (m : Member) (s : Json) : V Unit :=
  match slicePorts s with
  | none => throw (SLICE_NO_PORT_WHY m.name)
  | some ps =>
    match portTwice ps with
    | some p => throw (PORT_TWICE_WHY m.name p)
    | none => pure ()

/-- `Member.portsOnce` on each member of `ms` at its slice `slice m`, in order. -/
def portsOnceAll (slice : Member → Json) : List Member → V Unit
  | [] => pure ()
  | m :: ms => m.portsOnce (slice m) >>= fun _ => portsOnceAll slice ms

theorem portTwice_nodup : ∀ {ps : List String}, portTwice ps = none → ps.Nodup
  | [], _ => List.nodup_nil
  | p :: ps, h => by
    unfold portTwice at h
    split at h
    · cases h
    · rename_i hp
      exact List.nodup_cons.2 ⟨hp, portTwice_nodup h⟩

/-- Where `Member.portsOnce` passes on the slice `s`, every root of `s` names a port as a string, and no two the same. -/
theorem Member.portsOnce_ok {m : Member} {s : Json} (h : m.portsOnce s = .ok ()) :
    ∃ ps, slicePorts s = some ps ∧ ps.Nodup := by
  unfold Member.portsOnce at h
  split at h
  · cases h
  · rename_i ps hps
    split at h
    · cases h
    · rename_i hn
      exact ⟨ps, hps, portTwice_nodup hn⟩

theorem portsOnceAll_ok {slice : Member → Json} :
    ∀ {ms : List Member}, portsOnceAll slice ms = .ok () → ∀ m ∈ ms, ∃ ps, slicePorts (slice m) = some ps ∧ ps.Nodup
  | [], _, _, hm => by cases hm
  | m0 :: ms, h, m, hm => by
    rw [portsOnceAll] at h
    obtain ⟨⟨⟩, h0, h⟩ := Flock.HmNets.bind_ok h
    rcases List.mem_cons.1 hm with rfl | hm
    · exact Member.portsOnce_ok h0
    · exact portsOnceAll_ok h m hm

/-- The refusal of members of which two are named `name` (`membersStructure`): the members' roots, slices and
statements are each keyed by a member's name. -/
def NAME_TWICE_WHY (name : String) : String :=
  s!"proofs: two members are named {name}: a member's roots, slice and statement are each read by its name"

/-- **The members `ms` name each name once**, compared as strings (`portTwice`); two of one name refuse by it
(`NAME_TWICE_WHY`). -/
def namesOnce (ms : Array Member) : V Unit :=
  match portTwice (ms.toList.map (·.name)) with
  | some n => throw (NAME_TWICE_WHY n)
  | none => pure ()

theorem namesOnce_ok {ms : Array Member} (h : namesOnce ms = .ok ()) : (ms.toList.map (·.name)).Nodup := by
  unfold namesOnce at h
  split at h
  · cases h
  · exact portTwice_nodup (by assumption)

/-- Each range of `rs` (template, first unit, count) at its base among its template's own units: the counts of its
template's earlier ranges. -/
def localBases (rs : Array (String × Nat × Nat)) : Array Nat :=
  (rs.foldl (fun (acc : Array Nat × Std.HashMap String Nat) (t, _, c) =>
    (acc.1.push (acc.2.getD t 0), acc.2.insert t (acc.2.getD t 0 + c))) (#[], {})).1

/-- **The wire `w` enters the member `m`** over the ranges `rs` at their bases `lbase` (`localBases`): a range of
`w`'s reader holds `m`'s units, and `w`'s unit window, in its template's own units, meets `m`'s there. -/
def wireReaches (rs : Array (String × Nat × Nat)) (lbase : Array Nat) (m : Member) (w : Spot.Wires.Wire) : Bool :=
  (List.range rs.size).any fun j =>
    let (t, a, c) := rs[j]!
    let lo := lbase[j]! + (m.first - a)
    t == w.reader && decide (a ≤ m.first) && decide (m.first + m.count ≤ a + c) &&
      decide (lo < w.start + w.units / max w.cuts 1) && decide (w.start < lo + m.count)

end Spot.OneStageZk
