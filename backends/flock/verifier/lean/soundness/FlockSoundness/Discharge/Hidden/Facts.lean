import FlockSoundness.Discharge.Hidden.Spec
import FlockSoundness.Protocol.Discharge.Hidden.Facts
import FlockSoundness.Discharge.Exec.Merkle
import FlockSoundness.Discharge.Hidden.Defs

/-!
# The facts read off `Spec`

What [wave2-hidden-exec] reads of `Stmt.setupHidden`, from `setupHidden_spec`: its tags, its leaf scheme
(`setupHidden_hm96`), its well-formedness (`setupHidden_wf`: `StmtWF`, `RegionsWF`) and the drawn public file; and of
`HmOut.parse`, that an untyped one is `HmRow.parse`'s circuit then `HmOut.checkRows` (`parse_untyped`).
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

open Refine

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}

/-! ## The parse -/

/-! ## The regions -/

/-! ## The facts read off `Spec` -/

/-- `Spec.tagsEq`. -/
theorem setupHidden_tags (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.tags = tags :=
  (setupHidden_spec' h).tagsEq

/-- `Spec.pub`, at a draw. -/
theorem setupHidden_drawn {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) :
    ∃ pop, HmOut.loadPublic tags st.c pf = .ok pop ∧ HmOut.drawn st.c pop d = .ok st.pub := by
  obtain ⟨pop, hl, -, hd⟩ := (setupHidden_spec' h).pub
  exact ⟨pop, hl, hd d rfl⟩

end FlockSoundness.Discharge.Hidden
