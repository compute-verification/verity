import FlockSoundness.Discharge.PrivateCircuit.ReadsSound

/-!
# The hidden-reads soundness headline is not vacuous

A hypothesis about the class circuit can make a soundness headline trivially true when nothing satisfies it together
with the others. `private_circuit_sound_reads_nonvacuous` exhibits one instance at which every hypothesis of
`private_circuit_sound_reads` holds at once, and whose session accepts with the registrant's roots with probability one:

* the population is one unit (`nvP`), reading no cells and writing no output rows, so its class relation is the entry
  read alone;
* the class net (`nvC`, `nvPt`) is two gates of that unit, its check `¬(row [] [])`, which is `true`; the relation holds
  at every assignment, so the net computes it (`ClassComputes`);
* the table and the store are registered under one-leaf roots (`onePort`) of real hm96 leaves (`ExplicitLeaf.hm96`),
  and the registrant's openings check against them;
* the session (`nvS`) is the one-outcome game that accepts, draws its one unit, and whose committed values set every
  gate; the finders the two `cr/sha-512` hypotheses name find nothing, because no read can differ from the registrant's.

The `example` after it applies the headline at the instance, so the conjuncts are its hypotheses as stated.
-/

namespace FlockSoundness.Discharge.PrivateCircuit.Reads.NonVacuous

open Flock FlockSoundness.Game FlockSoundness.Registered Binding Audit
open scoped ENNReal

/-- One unit, no read slots, no output rows, one dimension. -/
abbrev nvP : Pop Unit where
  N := 1
  r := 0
  o := 0
  Dm := Unit
  blk := fun _ _ _ l => l.elim0

instance : Subsingleton (Entry nvP) :=
  ⟨fun ⟨d, s⟩ ⟨d', s'⟩ => by
    have hd : d = d' := Subsingleton.elim (α := Unit) d d'
    have hs : s = s' := funext fun k => k.elim0
    subst hd hs
    rfl⟩

/-- The class net: gate 0 is `row [] []` (`false`), gate 1, the check and the one output, is its negation. -/
abbrev nvC : Audit.Circuit where
  N := 2
  op g := if g.val = 0 then .row [] [] else .not 0
  topo g a ha := by
    by_cases h : g.val = 0
    · simp [h, Op.args] at ha
    · simp only [h, ↓reduceIte, Op.args, List.mem_singleton] at ha
      subst ha
      exact Fin.pos_iff_ne_zero.2 fun e => h (by rw [e]; rfl)
  outputs := {1}

/-- Both gates are the one unit's. -/
def nvPt : Audit.Partition nvC 1 where
  unit _ := some 0
  input_iff g := by
    refine ⟨fun h => absurd h (Option.some_ne_none _), fun h => ?_⟩
    exfalso
    by_cases hg : g.val = 0 <;> simp [hg, Op.isInput] at h

/-- The check gate. -/
def nvOk : Fin 1 → Fin nvC.N := fun _ => ⟨1, by decide⟩

/-- The entry every opening shows. -/
def nvRead : ReadVal (Entry nvP) := ⟨⟨(), Fin.elim0⟩, [], ByteArray.empty⟩

/-- The registrant's store opening. -/
def nvRow : ReadVal Unit := ⟨(), [], ByteArray.empty⟩

/-- The layout: the entry read decodes no gates; there are no input reads or output writes. -/
def nvL : ReadLayout nvP nvC.N where
  ent _ := ⟨0, Fin.elim0, fun _ => nvRead⟩
  inp _ k := k.elim0
  out _ l := l.elim0

/-- The table's hm96 leaves. -/
def nvLfT : ExplicitLeaf (Entry nvP) := ExplicitLeaf.hm96 (fun _ => []) fun a b _ => Subsingleton.elim a b

/-- The store's hm96 leaves. -/
def nvLfS : ExplicitLeaf Unit := ExplicitLeaf.hm96 (fun _ => []) fun a b _ => Subsingleton.elim a b

/-- **A one-leaf registered root**: one leaf, the payload's, at position 0, and no path. -/
def onePort (payload : ByteArray) : Flock.Registered.Port where
  name := ""
  value := ""
  root := Flock.Registered.climb ByteArray.empty (Flock.Registered.rowLeaf ByteArray.empty 0 payload) 0 0 ByteArray.empty
  leaves := 1
  words := none
  wordBits := 16
  domain := ByteArray.empty
  reads := #[]

theorem opensLeaf_onePort (payload : ByteArray) :
    Flock.Registered.opensLeaf (onePort payload) payload 0 ByteArray.empty = true := by
  simp [Flock.Registered.opensLeaf, onePort, Flock.log2ceil]

/-- The table root. -/
def nvRtT : Flock.Registered.Port := onePort (nvLfT.leaf nvRead.val nvRead.salt).1

/-- The store root. -/
def nvRtS : Flock.Registered.Port := onePort (nvLfS.leaf nvRow.val nvRow.salt).1

def nvRootT : Unit → Flock.Registered.Port := fun _ => nvRtT
def nvRootS : Unit → Flock.Registered.Port := fun _ => nvRtS
def nvIdx : Cell → Fin 1 := fun _ => 0
def nvTr : Fin nvP.N → ReadVal (Entry nvP) := fun _ => nvRead
def nvWo : Fin 1 → ReadVal Unit := fun _ => nvRow

/-- The good assignments: the check is set. -/
def nvGood : Unit → (Fin nvC.N → Bool) → Prop := fun _ v => v (nvOk 0) = true

/-- The law draws the one unit. -/
def nvLaw : Audit.Law 1 where
  Ω := Unit
  draw _ := Finset.univ

/-- **The session**: one outcome, which accepts; the committed values set every gate; no strata. -/
def nvS : UnitSession nvPt Unit where
  Out := Unit
  G := .ret ()
  Accepts _ := True
  regOf _ := ()
  Lm := nvLaw
  vals _ _ := fun _ => true
  εks _ := 0
  δlink _ := 0
  CR _ := True
  m := 0
  Ω := (i : Fin 0) → Fin (i.elim0 : ℕ) → Fin 256
  Lb i := i.elim0
  src ω := ω
  A6 := True

def nvDrawn : nvS.Out → Finset (Fin nvP.N) := fun _ => Finset.univ
def nvYD : nvS.Out → Fin nvP.N × Fin nvP.o → ReadVal Unit := fun _ _ => nvRow
def nvXD : nvS.Out → ℕ → ReadVal Unit := fun _ _ => nvRow

theorem nvRel (v : Fin nvC.N → Bool) (u : Fin 1) :
    ClassRel nvL nvLfT nvLfS nvIdx nvRtT nvRtS v u := by
  obtain rfl := Fin.fin_one_eq_zero u
  exact ⟨opensLeaf_onePort _, fun k => k.elim0, fun k => k.elim0, fun l => l.elim0, fun l => l.elim0⟩

theorem nvComputes :
    ClassComputes nvPt nvGood (fun R v u => ClassRel nvL nvLfT nvLfS nvIdx (nvRootT R) (nvRootS R) v u) nvOk where
  unit_ok u := by obtain rfl := Fin.fin_one_eq_zero u; rfl
  out_ok _ := Finset.mem_singleton.2 rfl
  only_ok g hg u _ := by
    classical
    rcases g with ⟨_ | _ | k, hk⟩
    · exfalso
      simp only [Partition.committed, Finset.mem_filter, Finset.mem_univ, true_and] at hg
      rcases hg with h | h | ⟨h, -, hu⟩
      · simp [Op.isInput] at h
      · simp at h
      · exact hu rfl
    · rfl
    · have : k + 1 + 1 < 2 := hk
      omega
  set_ok _ _ h _ := h
  computes R v _ u := by
    obtain rfl := Fin.fin_one_eq_zero u
    refine iff_of_true ?_ (nvRel v 0)
    rw [Partition.localEval_of_mem _ _ (rfl : nvPt.unit (nvOk 0) = some 0)]
    show (!(nvPt.localEval v 0 ⟨0, by decide⟩)) = true
    rw [Partition.localEval_of_mem _ _ (rfl : nvPt.unit ⟨0, by decide⟩ = some 0)]
    rfl

/-- **Every hypothesis of `private_circuit_sound_reads` holds at once**, at the instance of this file, and its session
accepts with the registrant's roots with probability one: the class net computes the class relation (`ClassComputes`),
the registrant's openings check, protocol 1's drawn bound and the draw's escape hold, A3, A6 and the knowledge finders'
condition hold, accepted outcomes are good, the delivered values check (none are owed), and both `cr/sha-512`
hypotheses hold at budget 0, for every prover. -/
theorem private_circuit_sound_reads_nonvacuous :
    ClassComputes nvPt nvGood (fun R v u => ClassRel nvL nvLfT nvLfS nvIdx (nvRootT R) (nvRootS R) v u) nvOk ∧
    (∀ j : Fin nvP.N, ReadOk nvLfT nvRtT j (nvTr j)) ∧ (∀ i : Fin 1, ReadOk nvLfS nvRtS i (nvWo i)) ∧
    ZkSessionDrawn nvS nvDrawn ∧ DrawEscapes nvS nvDrawn ∧ Assumptions.UniformRandomBytes nvS.Lb nvS.src ∧ nvS.A6 ∧
    (∀ τ, nvS.CR τ) ∧ (∀ o, nvS.Accepts o → ∀ τ, nvGood (nvS.regOf o) (nvS.vals τ o)) ∧
    (∀ o, nvS.Accepts o →
      (∀ p ∈ (∅ : Finset (Fin nvP.N × Fin nvP.o)), ReadOk nvLfS (nvRootS (nvS.regOf o)) (nvIdx (.out p.1 p.2)) (nvYD o p)) ∧
        ∀ m ∈ (∅ : Finset ℕ), ReadOk nvLfS (nvRootS (nvS.regOf o)) (nvIdx (.reg m)) (nvXD o m)) ∧
    (∀ τ, Assumptions.SHA512CRStrict Refine.H512 nvS.G τ
      (fun o => some (readPick nvLfS nvRtS Fin.val nvWo (storeReads nvL nvIdx nvS τ nvDrawn ∅ ∅ nvYD nvXD) o))
      (fun _ => 0) 0) ∧
    (∀ τ, Assumptions.SHA512CRStrict Refine.H512 nvS.G τ
      (fun o => some ((regCommit nvLfT nvP.N).ext (nvRtT, tabFill nvL nvS τ nvDrawn nvTr o) (nvRtT, nvTr)))
      (fun _ => 0) 0) ∧
    ∀ τ, prob (fun o => nvS.Accepts o ∧ nvRootT (nvS.regOf o) = nvRtT ∧ nvRootS (nvS.regOf o) = nvRtS) nvS.G τ = 1 := by
  classical
  have hwr := classWrongOn_of_computes nvComputes
  have hnone : ∀ u : Fin 1, u ∉ nvPt.wrong (fun _ => true) := fun u hu =>
    (hwr () (fun _ => true) rfl u).1 hu (nvRel _ u)
  have hfind : ∀ (τ : Strategy nvS.G) (f : nvS.Out → List UInt8 × List UInt8), (∀ o, f o = ([], [])) →
      (prob (fun b => ∃ x y, some (f b) = some (x, y) ∧ x ≠ y ∧ Refine.H512 x = Refine.H512 y) nvS.G τ).toReal
        ≤ (0 : ℝ) ^ 2 / 2 ^ (513 : ℝ) := by
    intro τ f hf
    have h0 : prob (fun b => ∃ x y, some (f b) = some (x, y) ∧ x ≠ y ∧ Refine.H512 x = Refine.H512 y) nvS.G τ ≤ 0 := by
      refine (prob_mono (q := fun _ => False) (fun b ⟨x, y, he, hne, _⟩ => hne ?_) _ _).trans (prob_false _ _).le
      rw [hf b] at he
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj he).symm
      rfl
    refine (ENNReal.toReal_mono ENNReal.zero_ne_top h0).trans ?_
    simp
  refine ⟨nvComputes, fun j => ?_, fun i => ?_, ?_, ?_, fun _ => rfl, trivial, fun _ => trivial, fun _ _ _ => rfl,
    fun _ _ => by simp, fun τ _ => hfind τ _ fun o => ?_, fun τ _ => hfind τ _ fun o => ?_, fun τ => ?_⟩
  · obtain rfl := Fin.fin_one_eq_zero j
    exact opensLeaf_onePort _
  · obtain rfl := Fin.fin_one_eq_zero i
    exact opensLeaf_onePort _
  · intro _ _ τ _
    refine (prob_mono (q := fun _ => False) (fun o ⟨_, u, _, hu⟩ => hnone u hu) _ _).trans ?_
    rw [prob_false]
    simp
  · intro τ B
    by_cases hB : B () = ∅
    · exact (prob_le_one _ _ _).trans (le_iSup_of_le () (by rw [hB, Law.escape_empty]))
    · refine (prob_mono (q := fun _ => False) (fun o ⟨_, hd⟩ => ?_) _ _).trans ?_
      · rw [prob_false]
        simp
      · obtain ⟨x, hx⟩ := Finset.nonempty_iff_ne_empty.2 hB
        exact Finset.disjoint_left.1 hd (Finset.mem_univ x) hx
  · unfold readPick
    split
    · next h =>
      obtain ⟨_, _, _, _, hne⟩ := h
      exact (hne (Subsingleton.elim _ _)).elim
    · rfl
  · unfold regCommit
    dsimp only
    split
    · next h =>
      obtain ⟨_, hne⟩ := h
      exact (hne (Subsingleton.elim _ _)).elim
    · rfl
  · exact le_antisymm (prob_le_one _ _ _)
      ((prob_true _ _).symm.le.trans (prob_mono (fun _ _ => ⟨trivial, rfl, rfl⟩) _ _))

example (τ : Strategy nvS.G) :=
  have h := private_circuit_sound_reads_nonvacuous
  private_circuit_sound_reads nvL nvLfT nvLfS nvIdx nvRootT nvRootS nvGood nvOk h.1 nvRtT nvRtS nvTr nvWo h.2.1 h.2.2.1
    nvS nvDrawn h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2.1 h.2.2.2.2.2.2.1 τ (h.2.2.2.2.2.2.2.1 τ)
    (fun o ho => h.2.2.2.2.2.2.2.2.1 o ho τ) ∅ ∅ nvYD nvXD h.2.2.2.2.2.2.2.2.2.1 (fun _ => 0) (fun _ => 0) 0 0
    (fun _ => le_rfl) (fun _ => le_rfl) (h.2.2.2.2.2.2.2.2.2.2.1 τ) (h.2.2.2.2.2.2.2.2.2.2.2.1 τ)

end FlockSoundness.Discharge.PrivateCircuit.Reads.NonVacuous
