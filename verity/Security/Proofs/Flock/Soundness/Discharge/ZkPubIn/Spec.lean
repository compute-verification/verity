import Proofs.Flock.Soundness.Discharge.ZkPubIn.Defs
import Proofs.Flock.Soundness.Discharge.Hidden.Spec

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

private theorem getD_eq_getElem! {α : Type} [Inhabited α] (a : Array α) (i : ℕ) (d : α) (h : i < a.size) :
    a.getD i d = a[i]! := by
  simp [Array.getD, h, getElem!_pos]

set_option maxHeartbeats 4000000 in
/-- **`HmIn.parse`'s checks on a circuit with public input ports.** -/
theorem parse_leavesPublic {c : Circuit} {pi : HmIn.Ports} (h : HmIn.parse c = .ok pi)
    (he : pi.ports.isEmpty = false) : LeavesPublic c pi := by
  unfold HmIn.parse at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' (first | walk_step | exact absurd ‹(throw _ : V _) = Except.ok _› (throw_ne_ok _ _))
  · rename_i s _ _ _ _ _ _ w3 _ _ _ lv _ _ _ _ hmap hpk hsl hws _ hblk _ _ hrow
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq, List.any_eq_true,
      List.mem_range, not_or, not_exists, not_and, not_not, not_lt, not_le, ge_iff_le] at hmap
    obtain ⟨hsz, hall⟩ := hmap
    refine ⟨hsz, ?_, by simpa using hpk, by omega, by simpa using hws, ?_, ?_⟩
    · intro u hu
      obtain ⟨hs, hj⟩ := hall u hu
      rw [getD_eq_getElem! _ _ _ (hsz ▸ hu)]
      refine ⟨hs, fun j hjl => ?_⟩
      obtain ⟨⟨h1, h2⟩, h3⟩ := hj j hjl
      rw [getD_eq_getElem! _ _ _ hjl]
      refine ⟨h1, h2, fun h0 => ?_⟩
      rw [getD_eq_getElem! _ _ _ (hs ▸ hjl)]
      exact h3 h0
    · rw [← Array.forIn_toList] at hblk
      refine list_forIn_each _ (fun b => _ ∧ _) (fun a s y hy => ?_) _ _ _ hblk
      split at hy
      · exact absurd hy (throw_ne_ok _ _)
      · rename_i hc
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, not_lt, gt_iff_lt] at hc
        exact ⟨⟨hc.1, hc.2⟩, _, hy.symm⟩
    · intro u hu j hj hin hlp
      rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hrow
      have hP := list_forIn_each _ (fun u => ∀ j < (c.leavesIn.getD u #[]).size,
          (HmIn.words c ⟨s, w3, lv⟩).binSearchContains ((c.unit.inGroups.getD 0 default).col + j / 8) (· < ·) = true →
            (lv[u]!)[j]! < 0 → (c.leavesIn.getD u #[])[j]! < 0) (fun a s y hy => ?_) _ _ _ hrow u
        (by simpa [Std.Legacy.Range.size, List.mem_range'_1] using hu)
      · have hl : j < (lv[u]!).size := (hall u hu).1 ▸ hj
        dsimp only at hlp
        rw [getD_eq_getElem! lv u _ (by omega), getD_eq_getElem! _ _ _ hl] at hlp
        rw [getD_eq_getElem! _ _ _ hj]
        exact hP j hj hin hlp
      · split at hy
        · exact absurd hy (throw_ne_ok _ _)
        · rename_i _ hx
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨fun j hj hin hlp => ?_, _, hy.symm⟩
          have := List.find?_eq_none.1 (Option.eq_none_iff_forall_ne_some.2 hx) j (List.mem_range.2 hj)
          simp only [Bool.and_eq_true, decide_eq_true_eq, not_and, ge_iff_le, not_le] at this
          exact this ⟨hin, hlp⟩
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact absurd he (by decide)

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
