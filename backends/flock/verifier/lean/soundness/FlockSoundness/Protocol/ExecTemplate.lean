import FlockSoundness.ExecCircuit
import FlockSoundness.ExecCheck
import FlockSoundness.Protocol.ExecTyped
import FlockSoundness.Protocol.ExecSetup

/-!
# A typed template's statement, as `HmRow.parse` and `Stmt.setupH` accept it (1e for templates, T2)

`HmRow.parse … (tmpl := some t)` is the path `parseTyped` takes for a typed template (#277, #307):
- `parse_facts_tmpl`: its nets are the template's root and placed layouts' circuits, then the file's own text nets and
  lookup slots; its unit is the root (`unitNet = 0`), one per VU; its block is `blockOf t` over those nets; its wires are
  the rows' own chains (`rowWires`), with no leaf map or cut; and `HmRow.check` passed.
-/

open Flock FlockLevel3

namespace FlockSoundness

/-- **What `parseTyped` accepted** (#307): `Typed.read`'s class `k` and block, then `parse` on the block, a template's
(`tmpl`) or a flat class's unit net (`pre`), with the class recorded as `cls`. -/
theorem parseTyped_spec {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmRow.parseTyped tags bytes tables = .ok c) :
    ∃ k r, Typed.read tags bytes tables = .ok (k, r) ∧
      ((∃ t c', r = .inr t ∧ HmRow.parse tags bytes tables (tmpl := some t) = .ok c' ∧ c = { c' with cls := some k }) ∨
        ∃ net c', r = .inl net ∧ HmRow.parse tags bytes tables (pre := #[("unit", net)]) = .ok c' ∧
          c = { c' with cls := some k }) := by
  unfold HmRow.parseTyped at h
  obtain ⟨⟨k, r⟩, hr, h⟩ := except_bind_ok h
  refine ⟨k, r, hr, ?_⟩
  cases r with
  | inr t =>
    obtain ⟨c', hc', h⟩ := except_bind_ok h
    exact Or.inl ⟨t, c', rfl, hc', (Except.ok.inj h).symm⟩
  | inl net =>
    obtain ⟨c', hc', h⟩ := except_bind_ok h
    exact Or.inr ⟨net, c', rfl, hc', (Except.ok.inj h).symm⟩


end FlockSoundness
