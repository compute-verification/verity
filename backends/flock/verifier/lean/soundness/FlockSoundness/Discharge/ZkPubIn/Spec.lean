import FlockSoundness.Discharge.ZkPubIn.Defs
import FlockSoundness.Discharge.Hidden.Spec

/-!
# What `Stmt.setupHidden` accepts with the verifier's public inputs

`setupHidden_specP`: every statement `Stmt.setupHidden … (some own)` returns on a circuit with public input ports is
its `SpecP`, as `Hidden.setupHidden_spec'` is `Spec` at `publicInputs := none` (where a circuit with public input
ports is refused, `Hidden.checkOwn_none`).

* `checkOwn_some`: `HmIn.checkOwn` accepting `own` means the circuit names a public input port and the population's
  public words are `own`'s (`OwnWords`).
* `parse_leavesPublic`: `HmIn.parse`'s checks on a circuit with public input ports (`LeavesPublic`).
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.ZkPubIn

open Hidden Refine

/-- **`checkOwn` accepting the verifier's own recomputation**: the circuit names a public input port, and every
population public word is `own`'s word at its index. -/
theorem checkOwn_some {pi : HmIn.Ports} {pop : Public} {own : ByteArray}
    (h : HmIn.checkOwn pi pop (some own) = .ok ()) : pi.ports.isEmpty = false ∧ OwnWords own pi pop := by
  unfold HmIn.checkOwn at h
  cases he : pi.ports.isEmpty
  · simp only [he] at h
    refine ⟨rfl, ?_⟩
    cases hnl : findByte own 10 with
    | none => simp [hnl] at h
    | some nl =>
      simp only [hnl] at h
      cases hu : String.fromUTF8? (own.extract 0 nl) with
      | none => simp only [hu, throw_bind'] at h; exact absurd h (throw_ne_ok _ _)
      | some t =>
        cases hj : Json.parse t with
        | error e => simp only [hu, hj, throw_bind'] at h; exact absurd h (throw_ne_ok _ _)
        | ok j =>
          simp only [hu, hj, pure_bind] at h
          repeat' (first | walk_step | exact absurd ‹(throw _ : V _) = Except.ok _› (throw_ne_ok _ _))
          rename_i hne
          intro x hx
          have hx' : ¬ (((pop.population.map (·.2.2)).getD pop.outputs).getD x 0 !=
              ((own.extract (nl + 1) own.size).get! (2 * x)).toNat +
                256 * ((own.extract (nl + 1) own.size).get! (2 * x + 1)).toNat) = true := by
            cases hf : List.find? (fun x => ((pop.population.map (·.2.2)).getD pop.outputs).getD x 0 !=
                ((own.extract (nl + 1) own.size).get! (2 * x)).toNat +
                  256 * ((own.extract (nl + 1) own.size).get! (2 * x + 1)).toNat)
                (List.range ((pop.population.map (·.1)).getD pop.n * pi.words)) with
            | some y => exact absurd hf (hne y)
            | none => exact List.find?_eq_none.1 hf x (List.mem_range.2 hx)
          unfold ownWord
          rw [hnl]
          simpa using hx'
  · simp [he] at h

/-- **`HmIn.parse`'s checks on a circuit with public input ports.** -/
theorem parse_leavesPublic {c : Circuit} {pi : HmIn.Ports} (h : HmIn.parse c = .ok pi)
    (he : pi.ports.isEmpty = false) : LeavesPublic c pi := by
  sorry

set_option maxHeartbeats 4000000 in
/-- **What `Stmt.setupHidden` accepts with the verifier's public inputs**: a statement it returns on the verifier's own
recomputation `own`, on a circuit whose public input ports `HmIn.parse` reads as `pi`, is its `SpecP`; and the
circuit names a public input port, else `checkOwn` refuses `own`. -/
theorem setupHidden_specP {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {own : ByteArray} {st : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st) :
    ∃ pi, SpecP tags cf pf coins tables draw own pi st := by
  cases draw <;> unfold Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hk := klog_of ‹¬ (_ || _) = true›
    have hown := checkOwn_some ‹HmIn.checkOwn _ _ (some _) = _›
    exact ⟨_, {
      tagsEq := rfl
      leaf := rfl
      parse := by assumption
      inParse := by assumption
      ports := hown.1
      leavesPublic := parse_leavesPublic ‹HmIn.parse _ = _› hown.1
      merkle := by assumption
      pub := by
        refine ⟨_, ‹HmIn.loadPublic _ _ _ _ = _›, ‹HmIn.checkOwn _ _ (some _) = _›, hown.2, fun hd => ?_,
          fun d' hd => ?_⟩
        · first | exact absurd hd (Option.some_ne_none _) | rfl
        · first | exact absurd hd.symm (Option.some_ne_none _) | (cases hd; assumption)
      checkPublic := by assumption
      n := rfl
      blocksReal := rfl
      nbl := rfl
      m := rfl
      kLog := hk.1
      m35 := hk.2
      mPts := rfl
      pin := by assumption
      da := rfl
      db := rfl
      regions := ⟨_, _, ‹HmOut.regions _ _ = _›, ‹HmIn.regions _ _ _ = _›, rfl⟩
      regionWords := by assumption
      inRange := by assumption
      sigma := rfl
      hello := by assumption }⟩

end FlockSoundness.Discharge.ZkPubIn
