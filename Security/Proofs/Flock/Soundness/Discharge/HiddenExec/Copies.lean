import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
import Proofs.Flock.Soundness.Discharge.Zero

/-!
# `Layout.Aliased` at a hidden-output statement, from its flat twin's copy positions

`Aliased.Copies.lean` and `Zero.Uniform.lean`'s block lemmas, at the hidden statement's block R1CS (`L.model regs`, its
level-0 matrices `placedA st`, `placedB st`), with the copy positions of its flat twin (`Aliased.copyPos Lf regs F`, at
the twin's layout `Lf` and unit facts `F`, which read only the circuit). Each unit slot's row reads the same in the
hidden statement as in its twin (`UnitRowsFlat`), so:
- `zeroPos_blockH`: each unit slot's forced-zero row carries 0 (`Layout.placed_zero`, at the hidden layout);
- `copyAt_blockH`, `copyPos_blockH`: each of template instance `g`'s stacked unit inputs carries what its copy position
  carries (`Layout.placed_port`, at the twin, whose Δ is `HmRow.delta`'s);
- `aliased_of_copiesH`: an instance that wires the inputs as Δ copies them (`Aliased.Copies` at the twin) is aliased in
  the hidden statement (`Layout.Aliased`);
- `stack_zero_blockH`: a stacked input that reads the zero carries 0.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3
open FlockSoundness.Discharge.Aliased (portBits_nodup port_self mem_portBits)
open FlockSoundness.Discharge.Zero (ZeroIn)

section copies

variable {tags : Tags} {st : Stmt} (L : Layout st) (Lf : Layout (Hidden.flatTwin st))
  (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
  (hR : UnitRowsFlat st)

/-- **A unit slot's forced-zero row carries 0** in the hidden statement, in every block of a satisfying witness. -/
theorem zeroPos_blockH {s₀ : ℕ} (hs₀ : s₀ < st.c.unitRange.count) {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o (Aliased.zeroPos Lf regs F hs₀) = 0 := by
  have huse := (built_spec L.unit_built).2.1
  have hra : st.c.unit.a.rows ≤ st.c.unit.useful := F.rowsA
  have hrb : st.c.unit.b.rows ≤ st.c.unit.useful := F.rowsB
  have hA : ∀ j', sparseEntry st.c.unit.a st.c.unit.useful j' = 0 := fun j' => by
    unfold sparseEntry
    rw [ite_eq_right (by omega)]
  have hB : ∀ j', sparseEntry st.c.unit.b st.c.unit.useful j' + lookupEntry st.c.unit.lookup st.c.unit.useful j' = 0 :=
    fun j' => by
      rw [L.unit_lookup]
      unfold sparseEntry
      rw [ite_eq_right (by omega)]
      rfl
  have hZ : ZeroRow (L.model regs) (Aliased.zeroPos Lf regs F hs₀) :=
    ⟨fun j => (L.placed_zero hs₀ F.check.unit_zero huse hA hB _ j rfl).1,
      fun j => (L.placed_zero hs₀ F.check.unit_zero huse hA hB _ j rfl).2⟩
  exact hZ.block_eq hz o

include hR in
/-- **A unit slot's input carries what its copy position carries** in the hidden statement, in every block of a
satisfying witness. -/
theorem copyAt_blockH {s₀ off : ℕ} (hs₀ : s₀ < st.c.unitRange.count) (hoff : off < st.c.unit.inWords * WORD)
    (r : Fin (2 ^ (L.model regs).kLog)) (hr : r.val = st.c.unitRange.slot s₀ + off) {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o r = (L.model regs).block z o (Aliased.copyAt Lf regs F hs₀ off) := by
  unfold Aliased.copyAt
  by_cases hm : off ∈ portBits st.c.unit.inGroups
  · rw [ite_eq_left hm]
    have hnd := portBits_nodup L.unit_built
    obtain ⟨p, t, hp, ht, hpt⟩ := mem_portBits hm
    have hsrc : flatSrcAt st.c s₀ off = Flat.flatSrc st.c s₀ p t := by
      unfold flatSrcAt
      rw [hpt, portOf_eq hnd hp ht]
      congr 1
      omega
    have hself := port_self L.unit_built (L.unitLog_le hs₀) hm
    rw [hpt] at hself
    have hb : (portCols st.c.unit.inGroups)[p]!.1 + t < 2 ^ st.c.unit.unitLog :=
      lt_of_lt_of_le (L.port_lt hp ht) (L.fit st.c.unitNet).2
    have hrow := fun j => Lf.placed_port F.parse F.check F.delta F.sha F.hm hnd hs₀ hp ht (fun j' => (hself j').1)
      (fun j' => by rw [Lf.unit_lookup]; exact (add_zero _).trans (hself j').2) r j (by rw [hr, hpt])
    have hrow' : ∀ j, placedA st r j = (if j.val = Flat.flatSrc st.c s₀ p t then 1 else 0) ∧
        placedB st r j = if j.val = Flat.flatSrc st.c s₀ p t then 1 else 0 := fun j => by
      obtain ⟨eA, eB⟩ := hR s₀ hs₀ _ hb r j (by rw [hr, hpt])
      rw [eA, eB]
      exact hrow j
    rw [hsrc]
    split
    · rename_i hlt
      have hC : CopyRow (L.model regs) r ⟨_, hlt⟩ :=
        ⟨fun j => by rw [← ite_fin_val j hlt]; exact (hrow' j).1, fun j => by rw [← ite_fin_val j hlt]; exact (hrow' j).2⟩
      exact hC.block_eq hz o
    · rename_i hlt
      have hZ : ZeroRow (L.model regs) r := by
        refine ⟨fun j => ?_, fun j => ?_⟩
        · rw [show (L.model regs).A₀ = placedA st from rfl, (hrow' j).1,
            ite_eq_right (by have := j.isLt; intro he; exact hlt (he ▸ this))]
        · rw [show (L.model regs).B₀ = placedB st from rfl, (hrow' j).2,
            ite_eq_right (by have := j.isLt; intro he; exact hlt (he ▸ this))]
      rw [hZ.block_eq hz o, zeroPos_blockH L Lf regs F hs₀ hz o]
  · rw [ite_eq_right hm]
    have hlt : st.c.unitRange.slot s₀ + off < 2 ^ st.c.kLog :=
      L.unit_slot_lt hs₀ (lt_of_lt_of_le hoff (L.fit st.c.unitNet).2)
    rw [dite_eq_left hlt]
    exact congrArg _ (Fin.ext hr)

include hR in
/-- **Each of template instance `g`'s stacked unit inputs carries what its copy position carries** in the hidden
statement, at its place (`stackPlace`, which `Layout.Aliased` reads), in every block of a satisfying witness. -/
theorem copyPos_blockH {g : ℕ} (hg : g < st.c.g) (x : Fin ((unitRows st L.unit_built).stackW st.c.upv))
    (hx : x.val < ((unitRows st L.unit_built).stack st.c.upv).nIn) {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o (stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) x) =
      (L.model regs).block z o (Aliased.copyPos Lf regs F hg x.val) := by
  have hn := L.unitRows_nIn
  have hx' : x.val < st.c.upv * (unitRows st L.unit_built).nIn := hx
  have hx'' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by rw [← hn]; exact hx'
  unfold Aliased.copyPos
  rw [dite_eq_left hx'']
  refine copyAt_blockH L Lf regs F hR _ (Nat.mod_lt _ (by
    exact Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx''; omega)) _ ?_ hz o
  unfold stackPlace
  rw [dite_eq_left hx']
  show st.c.unitRange.slot (g * st.c.upv + x.val / (unitRows st L.unit_built).nIn) +
    x.val % (unitRows st L.unit_built).nIn = _
  rw [hn]

include hR in
/-- **The unit's inputs on one gate carry one value in the hidden statement** (`Layout.Aliased`), at template instance
`g`'s stacked unit slots in every block, for an instance that wires them as Δ copies them in the flat twin
(`Aliased.Copies`). -/
theorem aliased_of_copiesH {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st L.unit_built).stack st.c.upv)) {g : ℕ} (hg : g < st.c.g)
    (hcp : Aliased.Copies Lf regs F inst hg) (o : Fin (2 ^ (L.model regs).nbl)) : L.Aliased regs inst o hg := by
  intro c c' hw hc hc' z hz
  rw [copyPos_blockH L Lf regs F hR hg c hc hz o, copyPos_blockH L Lf regs F hR hg c' hc' hz o]
  rcases hcp c c' hw hc hc' with he | ⟨⟨s, hs, he⟩, ⟨s', hs', he'⟩⟩
  · rw [he]
  · rw [he, he', zeroPos_blockH L Lf regs F hs hz o, zeroPos_blockH L Lf regs F hs' hz o]

include Lf F hR in
/-- **A stacked input that reads the zero carries 0** in the hidden statement: at template instance `g`'s place (`stackPlace`), in every
block of a satisfying witness. -/
theorem stack_zero_blockH {g : ℕ} (hg : g < st.c.g) (x : Fin ((unitRows st L.unit_built).stackW st.c.upv))
    (hx : x.val < ((unitRows st L.unit_built).stack st.c.upv).nIn) (hz : ZeroIn st.c x.val)
    {z : Model.Witness (L.model regs).m} (hsat : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o (stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) x) = 0 := by
  have hx' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    have : x.val < st.c.upv * (unitRows st L.unit_built).nIn := hx
    rwa [L.unitRows_nIn] at this
  rw [copyPos_blockH L Lf regs F hR hg x hx hsat o]
  obtain ⟨s, hs, he⟩ := Zero.copyPos_zero Lf regs F hg hx' hz
  rw [he]
  exact zeroPos_blockH L Lf regs F hs hsat o

end copies

end FlockSoundness.Discharge.HiddenExec
