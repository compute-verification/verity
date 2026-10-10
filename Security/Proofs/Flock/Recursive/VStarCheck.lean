import Flock.RecVStar
import Proofs.Flock.Recursive.Loads
import Proofs.Flock.Soundness.Refine.Rep

/-!
# Recursion: the verifier's check that a session is one of V*'s (gap 8, check 1 in the model)

`flock-verify verify --rec-inner F` runs `RecVStar.checkStmt` on each statement after `buildSession` (`FlockVerify.lean`'s
`setupOf`): the circuit's META binds the verifier's own partition object and program, its unit is one of V*'s statements
at the inner shape, its population is that statement's instance count, and the circuit file is byte for byte the one
`vstar stage` writes for that statement under the file's own bindings. `AcceptsZKR` doesn't read that check. This file
states it and what it gives. Nothing in `RecursiveAudit` reads it yet: which of V*'s statements a session must be, the
bindings' `program_digests` value, and how `Vs` is tied to V*'s files wait on gap 8's call.

* `innerAt spec st`: the verifier's own inner shape at the inner statement, the value `RecVStar.innerOf` returns.
* `AcceptsZKV i I own t`: flock-verify `--zk --registered F --rec-inner F` accepts: `AcceptsZKR`, and each statement
  passes `checkStmt` at the inner shape `i` against the inputs' circuit file, partition object and program and `own`.
* `firstDiff?_eq_none`: byte strings without a first difference are equal (`forIn_none_all`, the loop running through).
* `checkFile_ok`, `check_ok`, `checkCircuit_ok`, `checkStmt_ok`: what each check gives when it passes.
* `vstarFile_of_accepts`: at a raw session that `AcceptsZKV` accepts and that passes the statement check, the inputs'
  circuit file is `circuitFile s b` for one of V*'s statements `s` at `i`, named as the session's unit, with `b` the
  bindings its META names; the setup's statement has `s`'s instances as its population, and the verifier's reads hold
  `s`'s message-round ports.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FailClosed

/-- The verifier's own inner shape at an inner statement `st` set up with `spec`: `RecVStar.innerOf`'s value, its `m`,
`k_log`, two claims and one per region and link point, and its regions. -/
def innerAt (spec : Flock.SessionSpec) (st : Flock.Stmt) : RecVStar.Inner :=
  ⟨st.m, st.c.kLog, 2 + spec.points * st.regions.size, st.regions.size⟩

/-- **flock-verify `--zk --registered F --rec-inner F` accepts the transcript**, at the inner shape `i`: `AcceptsZKR`,
and each statement the session sets up passes `RecVStar.checkStmt` at `i`, against the inputs' circuit file, partition
object and program, and the verifier's own registered reads `own` (`FlockVerify.lean`'s `setupOf`, after
`buildSession`). -/
def AcceptsZKV (i : RecVStar.Inner) (I : Inputs) (own : Array Flock.Registered.Port) (t : Exec.Transcript) : Prop :=
  AcceptsZKR I own t ∧ ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
    sts.forM (RecVStar.checkStmt (some i) · I.circuitFile I.partition I.program (some own)) = .ok ()

/-- **A loop that returns early or runs through**, over a state whose first component is the early return: where each
step from a state with no return either returns a value (`.done` with `some (some y)`) or, at a `k` where `P k` holds,
continues with no return, a run that ends with no return had `P` at every `k`, and one that ends with a return returned
a value. -/
theorem forIn_none_all {β γ : Type} (P : ℕ → Prop)
    (f : ℕ → Option (Option γ) × β → Id (ForInStep (Option (Option γ) × β)))
    (hf : ∀ k s, s.1 = none →
      (∃ x, f k s = .done x ∧ ∃ y, x.1 = some (some y)) ∨ (P k ∧ ∃ x, f k s = .yield x ∧ x.1 = none)) :
    ∀ (L : List ℕ) (s : Option (Option γ) × β), s.1 = none →
      ((Id.run (forIn L s f)).1 = none → ∀ k ∈ L, P k) ∧ ∀ r, (Id.run (forIn L s f)).1 = some r → r.isSome
  | [], s, hs => ⟨fun _ => by simp, fun r h => by simp [hs] at h⟩
  | k :: L, s, hs => by
    rw [List.forIn_cons]
    rcases hf k s hs with ⟨x, hx, y, hy⟩ | ⟨hp, x, hx, hq⟩
    · rw [hx]
      refine ⟨fun h => ?_, fun r h => ?_⟩
      · change x.1 = none at h
        rw [hy] at h
        cases h
      · change x.1 = some r at h
        rw [hy] at h
        cases h
        rfl
    · rw [hx]
      obtain ⟨h1, h2⟩ := forIn_none_all P f hf L x hq
      refine ⟨fun h k' hk' => ?_, fun r h => h2 r h⟩
      rcases List.mem_cons.1 hk' with rfl | hk'
      · exact hp
      · exact h1 h k' hk'

theorem byteArray_getElem!_eq {a : ByteArray} {i : ℕ} (h : i < a.data.size) : a[i]! = a.data[i] := by
  have hlt : i < a.size := h
  rw [getElem!_pos a i hlt, ByteArray.getElem_eq_getElem_data]

/-- **Byte strings without a first difference are equal** (`RecVStar.firstDiff?`): its loop over the common length runs
through only where every byte agrees (`forIn_none_all`), and then it returns `none` only where the lengths agree. -/
theorem firstDiff?_eq_none {a b : ByteArray} (h : RecVStar.firstDiff? a b = none) : a = b := by
  unfold RecVStar.firstDiff? at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Id.run_bind] at h
  have hf : ∀ (k : ℕ) (s : Option (Option (ℕ × ℕ)) × ℕ), s.1 = none →
      (∃ x, (if (a[k]! != b[k]!) = true then pure (ForInStep.done (some (some (k, s.2)), s.2))
          else if (a[k]! == 10) = true then pure (ForInStep.yield (none, s.2 + 1))
          else pure (ForInStep.yield (none, s.2)) : Id _) = .done x ∧ ∃ y, x.1 = some (some y)) ∨
        (a[k]! = b[k]! ∧ ∃ x, (if (a[k]! != b[k]!) = true then pure (ForInStep.done (some (some (k, s.2)), s.2))
          else if (a[k]! == 10) = true then pure (ForInStep.yield (none, s.2 + 1))
          else pure (ForInStep.yield (none, s.2)) : Id _) = .yield x ∧ x.1 = none) := by
    intro k s _
    by_cases hk : a[k]! = b[k]!
    · right
      refine ⟨hk, ?_⟩
      simp only [hk, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
      split <;> exact ⟨_, rfl, rfl⟩
    · left
      have hne : (a[k]! != b[k]!) = true := by simpa using hk
      simp only [hne, ↓reduceIte]
      exact ⟨_, rfl, _, rfl⟩
  split at h
  · rename_i r heq
    have := (forIn_none_all (fun k => a[k]! = b[k]!) _ hf _ _ rfl).2 r heq
    subst h
    simp at this
  · rename_i heq
    have hall := (forIn_none_all (fun k => a[k]! = b[k]!) _ hf _ _ rfl).1 heq
    split at h
    · rename_i hsz
      have hs : a.size = b.size := by simpa using hsz
      apply ByteArray.ext
      apply Array.ext
      · simpa using hs
      · intro i h1 h2
        have hi := hall i (by simp [List.mem_range'_1, Std.Legacy.Range.size]; simp at h1 h2; omega)
        rw [byteArray_getElem!_eq h1, byteArray_getElem!_eq h2] at hi
        exact hi
    · cases h

/-- **The file check**: the circuit file is V*'s for `s` under `b`. -/
theorem checkFile_ok {s : VStar.Stmt} {b : VStar.Bindings} {circuitB : ByteArray}
    (h : RecVStar.checkFile s b circuitB = .ok ()) : RecVStar.circuitFile s b = .ok circuitB := by
  unfold RecVStar.checkFile at h
  cases hc : RecVStar.circuitFile s b with
  | error e =>
    rw [hc] at h
    cases h
  | ok f =>
    rw [hc] at h
    dsimp only at h
    rw [pure_bind] at h
    cases hd : RecVStar.firstDiff? circuitB f with
    | none => rw [firstDiff?_eq_none hd]
    | some p =>
      obtain ⟨k, l⟩ := p
      rw [hd] at h
      cases h

/-- **The unit check**: a unit named `name` over `n` instances is one of V*'s statements at `i`, by its name, and `n` is
its instance count. -/
theorem check_ok {i : RecVStar.Inner} {name : String} {n : ℕ} {s : VStar.Stmt}
    (h : RecVStar.check i name n = .ok s) :
    ∃ ss, RecVStar.statements i = .ok ss ∧ s ∈ ss.toList ∧ s.name = name ∧ n = s.instances := by
  unfold RecVStar.check at h
  dsimp only at h
  split at h
  all_goals try (split at h; (obtain ⟨_, h1, -⟩ := ebind_ok h); cases h1)
  all_goals
    obtain ⟨ss, hss, h⟩ := ebind_ok h
    split at h
    · cases h
    · rename_i s' hfl
      split at h
      · obtain ⟨_, h1, -⟩ := ebind_ok h
        cases h1
      · rename_i hn
        cases h
        have hm : s ∈ ss.toList.filter (·.name == name) := by rw [hfl]; simp
        rw [List.mem_filter] at hm
        exact ⟨ss, hss, hm.1, by simpa using hm.2, by simpa using hn⟩
    · cases h

/-- **The circuit check**: the unit check, and the file check under the bindings META names. -/
theorem checkCircuit_ok {i : RecVStar.Inner} {c : Flock.Circuit} {n : ℕ} {circuitB : ByteArray} {s : VStar.Stmt}
    (h : RecVStar.checkCircuit i c n circuitB = .ok s) :
    RecVStar.check i c.unit.name n = .ok s ∧
      ∃ b, RecVStar.bindingsOf c = .ok b ∧ RecVStar.checkFile s b circuitB = .ok () := by
  unfold RecVStar.checkCircuit at h
  obtain ⟨s', hs', h⟩ := ebind_ok h
  obtain ⟨b, hb, h⟩ := ebind_ok h
  obtain ⟨u, hu, h⟩ := ebind_ok h
  cases h
  exact ⟨hs', b, hb, hu⟩

/-- **`checkStmt` at an inner shape and the verifier's own reads**: the statement's circuit file is `circuitFile s b`
for one of V*'s statements `s` at `i`, named as its unit, with the statement's population as its instances, under the
bindings `b` its META names, and the verifier's reads hold `s`'s message-round ports. -/
theorem checkStmt_ok {i : RecVStar.Inner} {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json}
    {program : Option ByteArray} {own : Array Flock.Registered.Port}
    (h : RecVStar.checkStmt (some i) st circuitB partition program (some own) = .ok ()) :
    ∃ ss s b, RecVStar.statements i = .ok ss ∧ s ∈ ss.toList ∧ s.name = st.c.unit.name ∧
      RecVStar.population st = s.instances ∧ RecVStar.bindingsOf st.c = .ok b ∧
      RecVStar.circuitFile s b = .ok circuitB ∧ RecVStar.checkOwn s own = .ok () := by
  unfold RecVStar.checkStmt at h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨s, hs, h⟩ := ebind_ok h
  obtain ⟨hck, b, hb, hf⟩ := checkCircuit_ok hs
  obtain ⟨ss, hss, hmem, hname, hn⟩ := check_ok hck
  exact ⟨ss, s, b, hss, hmem, hname, hn, hb, checkFile_ok hf, h⟩

/-- **A raw session that `AcceptsZKV` accepts loads one of V*'s files**, where its inputs pass the statement check: its
one statement is the setup's (`Exec.setupTables_ok`, one table), whose circuit is the inputs' (`setupH_spec`,
`ScopeZ.parse`), and `checkStmt_ok` at it. Which of V*'s statements, and the bindings' `program_digests` value, are the
file's. -/
theorem vstarFile_of_accepts {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (t : Exec.Transcript)
    {i : RecVStar.Inner} (hacc : AcceptsZKV i x.I x.own t) (hck : ScopeZR x.I) :
    ∃ spec st ss s b, setupOf (zkInputs x.I) t.record = .ok (spec, #[st]) ∧ st.c = circZ x.I ∧
      RecVStar.statements i = .ok ss ∧ s ∈ ss.toList ∧ s.name = (circZ x.I).unit.name ∧
      RecVStar.population st = s.instances ∧ RecVStar.bindingsOf (circZ x.I) = .ok b ∧
      RecVStar.circuitFile s b = .ok x.I.circuitFile ∧ RecVStar.checkOwn s x.own = .ok () := by
  obtain ⟨-, spec, sts, hs, hv⟩ := hacc
  have hZ := hck.scopeZ x.hTags.1
  have hh' : (zkTags x.I).hiddenOutputs = false := (zkTags_hiddenOutputs x.I).trans x.hTags.1
  rcases Exec.setupTables_ok hh' hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
  swap
  · exact absurd x.hTags.2.2.2 (by change 1 < x.I.count at hc'; omega)
  have hcst : st.c = circZ x.I := by
    have h1 : HmRow.parse (zkTags x.I) x.I.circuitFile x.I.tables = .ok st.c :=
      (setupH_spec ((zkTags_typed x.I).trans hZ.typed) hst).1
    rw [hZ.parse x.hTags.1] at h1
    exact (Except.ok.inj h1).symm
  have hck1 := array_forM_ok hv 0 (by simp)
  simp only [List.getElem_toArray, List.getElem_cons_zero] at hck1
  obtain ⟨ss, s, b, hss, hmem, hname, hn, hb, hf, ho⟩ := checkStmt_ok hck1
  rw [hcst] at hname hb
  exact ⟨spec, st, ss, s, b, hs, hcst, hss, hmem, hname, hn, hb, hf, ho⟩

end FlockSoundness.Discharge.Recursion
