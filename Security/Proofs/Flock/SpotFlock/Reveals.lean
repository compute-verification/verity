module

public import Proofs.Flock.Claims.Sampling.Code
public import Proofs.Flock.Claims.Sampling.ProgReads
public import Definitions.Claims.SamplingTwo

/-!
# What a SPOT[Flock] session reveals

The openings a session's witness reveals for a checked unit are the program's: one for each value its proof block's
correctness reads (`Prog.Needs`, as `Prog.needsList` lists them), at that gate, under the roots that value is committed
under: the interior roots of the unit's replay block where it is one of that block's interior values, and the run's
roots otherwise. What each opening holds, its value and its data, is an opener's (`op`), so the reveal count is the
program's whatever the opener: at most the largest number of gates one of the partition's proof blocks needs opened
(`length_le`).

* `needs`: the gates unit `u`'s proof block needs opened, each once, none off the partition;
* `rootsAt`: the roots gate `g`'s opening is under;
* `openAt`: the opening at gate `g`, under its roots, holding the opener's;
* `reveal`: the opening at each gate `needs` lists;
* `reveals`: the largest number of gates one of the partition's proof blocks needs opened;
* `frameAt`: the opener that reads a code's reveal, its first opening at the gate under the roots;
* `mem_needs`, `mem_reveal`, `openAt_mem`, `rootsAt_eq`: what `reveal` holds, at a unit in the partition and the
  two-stage design: the opening at each gate `Prog.Needs` names, under `rootsAt`, and nothing else.
-/

@[expose] public section

namespace SpotFlock.Reveal

open Claims Claims.Sampling SamplingDeploy

variable (part : AuditStatement → Option Partition) (two : AuditStatement → Option Design.Two)
  (ir : Seen → ℕ → List Msg)

/-- **The gates unit `u`'s proof block needs opened** at the statement `s` (`Prog.needsList`), each once; none where
the partition doesn't evaluate or has no block `u`. -/
def needs (s : AuditStatement) (u : ℕ) : List ℕ :=
  match part s with
  | some P => (P.blocks[u]?.map fun b => (s.prog.needsList b).dedup).getD []
  | none => []

/-- **The roots gate `g`'s opening is under**, for unit `u` at the transcript `v`: the interior roots `ir v ρ` of `u`'s
replay block `ρ` where `g` is one of that block's gates and none of its boundary, and the run's roots otherwise. -/
def rootsAt (s : AuditStatement) (u : ℕ) (v : Seen) (g : ℕ) : List Msg :=
  match part s, two s with
  | some P, some d =>
    match replayOf P d.part u with
    | some ρ =>
      match d.part.blocks[ρ]? with
      | some rb => if g ∈ rb ∧ s.prog.Boundary rb g = false then ir v ρ else s.roots
      | none => s.roots
    | none => s.roots
  | _, _ => s.roots

variable (op : AuditStatement → ℕ → Seen → Claims.Flock.Wit → ℕ → List Msg → Opening Bool)

/-- **The opening at gate `g`** for unit `u` of a session's witness `wF`: under `g`'s roots (`rootsAt`), at `g`, holding
the opener's value and data there. -/
def openAt (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) (g : ℕ) : List Msg × Opening Bool :=
  (rootsAt part two ir s u v g, { op s u v wF g (rootsAt part two ir s u v g) with pos := g })

/-- **What a session's witness `wF` reveals for unit `u`**: the opening at each gate its proof block needs opened. -/
def reveal (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) : List (List Msg × Opening Bool) :=
  (needs part s u).map (openAt part two ir op s u v wF)

/-- **The reveal count** at the run's statement `x`: the largest number of gates one of the partition's proof blocks
needs opened, at the statement `x` parses to; `0` where it doesn't parse or the partition doesn't evaluate. -/
def reveals (parse : Msg → Option AuditStatement) (x : Msg) : ℕ :=
  match parse x with
  | some s => (((part s).map fun P => P.blocks.map fun b => (s.prog.needsList b).dedup.length).getD []).foldr max 0
  | none => 0

/-- **The opener that reads a code's reveal**: `f`'s first opening at gate `g` under the roots `r`, and an empty
opening at `g` where it reveals none. -/
def frameAt (f : Code) (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) (g : ℕ) (r : List Msg) :
    Opening Bool :=
  (((f.reveal s u v wF).find? fun q => q.2.pos == g && q.1 == r).map Prod.snd).getD ⟨g, false, []⟩

theorem le_foldr_max {l : List ℕ} {a : ℕ} (h : a ∈ l) : a ≤ l.foldr max 0 := by
  induction l with
  | nil => cases h
  | cons b l ih =>
    show a ≤ max b (l.foldr max 0)
    rcases List.mem_cons.1 h with rfl | h
    · exact Nat.le_max_left _ _
    · exact (ih h).trans (Nat.le_max_right _ _)

theorem needs_eq {s : AuditStatement} {u : ℕ} {P : Partition} {b : List ℕ} (hP : part s = some P)
    (hb : P.blocks[u]? = some b) : needs part s u = (s.prog.needsList b).dedup := by
  simp [needs, hP, hb]

/-- **`needs` lists exactly what the proof block's correctness reads** (`Prog.Needs`). -/
theorem mem_needs {s : AuditStatement} {u g : ℕ} {P : Partition} {b : List ℕ} (hP : part s = some P)
    (hb : P.blocks[u]? = some b) : g ∈ needs part s u ↔ s.prog.Needs b g := by
  rw [needs_eq part hP hb, List.mem_dedup, ← Prog.needs_iff]

theorem length_needs_le (parse : Msg → Option AuditStatement) {x : Msg} {s : AuditStatement} (hx : parse x = some s)
    (u : ℕ) : (needs part s u).length ≤ reveals part parse x := by
  have hr : reveals part parse x =
      (((part s).map fun P => P.blocks.map fun b => (s.prog.needsList b).dedup.length).getD []).foldr max 0 := by
    simp only [reveals, hx]
  rw [hr]
  cases hP : part s with
  | none => simp [needs, hP]
  | some P =>
    cases hb : P.blocks[u]? with
    | none => simp [needs, hP, hb]
    | some b =>
      rw [needs_eq part hP hb]
      exact le_foldr_max (List.mem_map.2 ⟨b, List.mem_of_getElem? hb, rfl⟩)

/-- **The reveal count bounds the reveal**, whatever the opener: at a statement that parses, what a session's witness
reveals for a unit is at most `reveals` openings. -/
theorem length_le (parse : Msg → Option AuditStatement) {x : Msg} {s : AuditStatement} (hx : parse x = some s)
    (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) : (reveal part two ir op s u v wF).length ≤ reveals part parse x := by
  unfold reveal
  rw [List.length_map]
  exact length_needs_le part parse hx u

theorem mem_reveal {s : AuditStatement} {u : ℕ} {v : Seen} {wF : Claims.Flock.Wit} {q : List Msg × Opening Bool} :
    q ∈ reveal part two ir op s u v wF ↔ ∃ g ∈ needs part s u, openAt part two ir op s u v wF g = q := by
  unfold reveal
  exact List.mem_map

/-- **The reveal holds the opening at each gate the proof block needs** (`Prog.Needs`). -/
theorem openAt_mem {s : AuditStatement} {u g : ℕ} {v : Seen} {wF : Claims.Flock.Wit} {P : Partition} {b : List ℕ}
    (hP : part s = some P) (hb : P.blocks[u]? = some b) (hg : s.prog.Needs b g) :
    openAt part two ir op s u v wF g ∈ reveal part two ir op s u v wF :=
  (mem_reveal part two ir op).2 ⟨g, (mem_needs part hP hb).2 hg, rfl⟩

theorem rootsAt_eq {s : AuditStatement} {u ρ g : ℕ} {v : Seen} {P : Partition} {d : Design.Two} {rb : List ℕ}
    (hP : part s = some P) (hd : two s = some d) (hρ : replayOf P d.part u = some ρ)
    (hrb : d.part.blocks[ρ]? = some rb) :
    rootsAt part two ir s u v g = if g ∈ rb ∧ s.prog.Boundary rb g = false then ir v ρ else s.roots := by
  simp only [rootsAt, hP, hd, hρ, hrb]

end SpotFlock.Reveal
