import FlockSoundness.Protocol.ExecTemplatePlace

/-!
# A typed template's layout, from `Stmt.setupH` (1e for templates, T3)

`ExecTemplatePlace.lean` places a template's block rows from `TemplateLayout`, the accepted statement's layout facts. This
file derives them from what `Stmt.setupH` accepted for a template (`setupH_spec_typed`, `parseTyped_spec`,
`parse_facts_tmpl`, `check_facts_typed`, `delta_typed`) and what `Typed.read` built it from (`read_template`,
`templateOf_spec`, `blockOf_spec`).
-/

open Flock FlockLevel3

namespace FlockSoundness

open Compose Flock.DeriveAll

/-! ## The class's unit is one of its layouts -/

/-! ## The tables the class holds -/

/-! ## The rows' wires are into the rows' circuits -/

/-! ## A template circuit's nets -/

/-! ## A placed layout's net -/


section template

variable {name : String} {done : Array D} {ls : Array Flock.Layout} {unit : ℕ} {ty : CircuitType}
  {held : Array Typed.Held} {t : Typed.Template}

/-! ## The block's parts -/

/-! ## An input's message bit is a column of the block -/

theorem starts_size (c : Circuit) : (Typed.starts c).size = c.ports.size := by
  unfold Typed.starts
  rw [← Array.foldl_toList]
  suffices h : ∀ (L : List Port) (acc : Array ℕ × ℕ),
      (L.foldl (fun (acc : Array ℕ × ℕ) p => (acc.1.push acc.2, acc.2 + p.words)) acc).1.size = acc.1.size + L.length by
    simpa using h c.ports.toList (#[], 0)
  intro L
  induction L with
  | nil => intro acc; simp
  | cons p L ih => intro acc; rw [List.foldl_cons, ih]; simp only [Array.size_push, List.length_cons]; omega

/-! ## The accepted template statement's layout -/

end template

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}


/-! ## The unit's inputs copy their message bits -/

namespace TemplateLayout

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

end TemplateLayout

end FlockSoundness
