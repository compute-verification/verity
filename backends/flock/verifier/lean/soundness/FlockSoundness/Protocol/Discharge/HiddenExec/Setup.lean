import FlockSoundness.Protocol.Discharge.HiddenExec.Tables
import FlockSoundness.Protocol.Discharge.Hidden.Theorems

/-!
# What `Stmt.setupHidden` returns, as this area reads it

`setupOne_hidden`, `setupOne_public`: `setupOne` is `setupHidden` under hidden outputs and `setupH` otherwise.

The rest are what this area reads of `Stmt.setupHidden`, all from [wave2-hidden]'s `Hidden.setupHidden_spec`
(`Discharge/Hidden/`). `setupHidden_tags`, `setupHidden_drawn`, `setupHidden_parseC` and `setupHidden_n` are projections
of `Spec` (`tagsEq`, `pub`, `parse`, and `n`, `blocksReal`, `nbl`); `setupHidden_hm96`, `setupHidden_wf` and
`parse_untyped` are [wave2-hidden]'s (`Hidden/Facts.lean`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock Refine
open FlockSoundness.Discharge.Exec
open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Flock.Stmt}

/-- `Spec.parse`: the statement's circuit is the one `HmOut.parse` accepts. -/
theorem setupHidden_parseC (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    HmOut.parse tags cf tables = .ok st.c :=
  (Hidden.setupHidden_spec h).parse

/-- An untyped `HmOut.parse` is `HmRow.parse`'s circuit, then `HmOut.checkRows` ([wave2-hidden]'s
`Hidden.parse_untyped`). -/
theorem parse_untyped {c : Circuit} (ht : tags.typed = false) (h : HmOut.parse tags cf tables = .ok c) :
    HmRow.parse tags cf tables = .ok c ∧ HmOut.checkRows c = .ok () :=
  Hidden.parse_untyped ht h

end FlockSoundness.Discharge.HiddenExec
