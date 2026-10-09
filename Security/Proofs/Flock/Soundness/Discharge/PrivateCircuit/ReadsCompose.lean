import Proofs.Flock.Soundness.Discharge.PrivateCircuit.ReadsDefs

/-!
# Wrong units against the topology table and the row store

What the verifier is owed is the registered circuit's outputs end to end on the registered cells: `evalR` at the
registered table `T` on the store's registered cells. The units are judged against the same objects: unit `j` is correct
when its block, at the dimensions of `T`'s entry `j`, on the store's values at the cells that entry names, gives the
store's values at `j`'s output rows (`UnitOk`).

* `store_cone`: if every unit feeding unit `i` under `T` is correct against the store, `i`'s output rows in the store are
  `evalR`'s. No wiring hypothesis enters: a reader and its producer meet at one store cell, so the store itself is the
  wiring.
* `owedWrong_cone`, `owedWrong_bad`: so wrong owed outputs (`OwedWrongR`) put a unit wrong against the store in the cone
  of an owed unit, and `bad T w` is not empty.
* `classRel_bad`: at a unit wrong against the store, the class relation's reads (`ClassRel`) cannot all be the registered
  ones: given the registered entry, some input read or output write holds a value other than the store's. The reads are
  the layout's (`ReadLayout.tabOf`, `xOf`, `yOf`), so this is the step `read_binding` turns into a collision.
* `classWrongOn_of_computes`: where the class net computes the class relation (`ClassComputes`), its wrong units are the
  relation's failures (`ClassWrongOn`).
-/

namespace FlockSoundness.Discharge.PrivateCircuit.Reads

open Flock Audit Binding
open scoped ENNReal

variable {V : Type} [Inhabited V] (P : Pop V)

/-- **The units feeding unit `i`, correct against the store, give its owed outputs**: `i`'s output rows in the store are
the circuit's end to end on the store's registered cells. -/
theorem store_cone (T : Table P) (w : Cell → V) (i : Fin P.N) (h : ∀ j, Reaches P T j i → UnitOk P T w j) :
    ∀ l : Fin P.o, w (.out i l) = evalR P T (fun m => w (.reg m)) i l := by
  suffices H : ∀ n (i : Fin P.N), i.val = n → (∀ j, Reaches P T j i → UnitOk P T w j) →
      ∀ l : Fin P.o, w (.out i l) = evalR P T (fun m => w (.reg m)) i l from H i.val i rfl h
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro i hi h l
  obtain ⟨hac, hblk⟩ := h i (.refl i)
  rw [← hblk l, evalR]
  congr 2
  funext k
  cases hs : (T i).src k with
  | reg m => rfl
  | out i' l' =>
    have ha := hac k i' l' hs
    have hi' : i' < P.N := lt_trans ha.1 i.isLt
    dsimp only
    rw [dite_eq_left_of_eq_true (eq_true ha)]
    exact ih i' (hi ▸ ha.1) ⟨i', hi'⟩ rfl (fun j hj => h j (.step k ⟨i', hi'⟩ l' hs hj)) ⟨l', ha.2⟩

/-- **Wrong owed outputs are a unit wrong against the store in an owed unit's cone.** -/
theorem owedWrong_cone (T : Table P) (w : Cell → V) (owed : Finset (Fin P.N × Fin P.o)) (h : OwedWrongR P T w owed) :
    ∃ p ∈ owed, ∃ j ∈ cone P T p.1, j ∈ bad P T w := by
  classical
  obtain ⟨p, hp, hne⟩ := h
  refine ⟨p, hp, ?_⟩
  by_contra hno
  refine hne (store_cone P T w p.1 (fun j hj => ?_) p.2)
  by_contra hj'
  exact hno ⟨j, by unfold cone; exact Finset.mem_filter.2 ⟨Finset.mem_univ _, hj⟩,
    by unfold bad; exact Finset.mem_filter.2 ⟨Finset.mem_univ _, hj'⟩⟩

/-- **Wrong owed outputs leave a unit wrong against the store.** -/
theorem owedWrong_bad (T : Table P) (w : Cell → V) (owed : Finset (Fin P.N × Fin P.o)) (h : OwedWrongR P T w owed) :
    (bad P T w).Nonempty := by
  obtain ⟨_, _, j, _, hj⟩ := owedWrong_cone P T w owed h
  exact ⟨j, hj⟩

variable {P}

omit [Inhabited V] in
/-- **A unit wrong against the store reads or writes off the store.** Where the class relation holds at unit `u` and `u`'s
entry read is `T`'s entry, but `u` is wrong against the store `w`, some input read holds a value other than `w`'s at the
cell `T`'s entry names, or some output write a value other than `w`'s at `u`'s output row. -/
theorem classRel_bad {G M : ℕ} (L : ReadLayout P G) (lfT : ExplicitLeaf (Entry P)) (lfS : ExplicitLeaf V)
    (idx : Cell → Fin M) (rtT rtS : Flock.Registered.Port) (v : Fin G → Bool) (u : Fin P.N)
    (hrel : ClassRel L lfT lfS idx rtT rtS v u) (T : Table P) (w : Cell → V) (hT : L.tabOf v u = T u)
    (hbad : u ∈ bad P T w) :
    (∃ k, L.xOf v u k ≠ w ((T u).src k)) ∨ ∃ l : Fin P.o, L.yOf v u l ≠ w (.out u l) := by
  classical
  by_contra hno
  have hx : L.xOf v u = fun k => w ((T u).src k) := funext fun k => by
    by_contra h
    exact hno (Or.inl ⟨k, h⟩)
  have hy : ∀ l, L.yOf v u l = w (.out u l) := fun l => by
    by_contra h
    exact hno (Or.inr ⟨l, h⟩)
  obtain ⟨-, hac, -, -, hblk⟩ := hrel
  have hok : UnitOk P T w u := ⟨by rw [← hT]; exact hac, fun l => by rw [← hy l, ← hblk l, hT, hx]⟩
  unfold bad at hbad
  exact (Finset.mem_filter.1 hbad).2 hok

/-- **The class net's wrong units are the class relation's**: where the class net computes `rel` (`ClassComputes`), a
unit is wrong at a good assignment exactly when `rel` fails there, so `ClassWrongOn` is a fact about the net and not a
free hypothesis. A unit's only committed gate of its own is its check, set at the good assignments, so the unit is
correct exactly when its gates evaluate the check to `true`. -/
theorem classWrongOn_of_computes {C : Audit.Circuit} {n : ℕ} {Reg : Type} {Pt : Audit.Partition C n}
    {good : Reg → (Fin C.N → Bool) → Prop} {rel : Reg → (Fin C.N → Bool) → Fin n → Prop} {ok : Fin n → Fin C.N}
    (h : ClassComputes Pt good rel ok) : ClassWrongOn Pt good rel := by
  intro R v hg u
  rw [Audit.Partition.mem_wrong]
  refine not_congr ⟨fun hc => ?_, fun hr g hg' hu => ?_⟩
  · have e := hc (ok u) (Pt.mem_committed_of_output (h.out_ok u)) (h.unit_ok u)
    rw [h.set_ok R v hg u] at e
    exact (h.computes R v hg u).1 e.symm
  · obtain rfl := h.only_ok g hg' u hu
    rw [h.set_ok R v hg u]
    exact ((h.computes R v hg u).2 hr).symm

end FlockSoundness.Discharge.PrivateCircuit.Reads
