module

public import Flock.RecVStar
public import Proofs.Flock.Recursive.Raw
public import Proofs.Flock.Soundness.Discharge.FailClosed.One
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk

/-! The verifier's check that a session is one of V*'s statements (`RecVStar.checkStmt`), and what it gives. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FailClosed

def innerAt (spec : Flock.SessionSpec) (st : Flock.Stmt) : RecVStar.Inner :=
  { m := st.m, kLog := st.c.kLog, claims := 2 + spec.points * st.regions.size, regions := st.regions.size }

/-- **verity-verify-proof `--zk --registered F --rec-inner F --rec-part J --rec-bindings F --rec-rows F` accepts the
transcript**, at the inner shape `i`, as V*'s statement `j` under the verifier's own bindings `f`: `AcceptsZKR`, and each
statement the session sets up passes `RecVStar.checkStmt` at `i`, against the inputs' tags and lookup tables, circuit
file, partition object and program, the verifier's own registered reads `own`, `j`, `f` and its shared rows `rows`
(`FlockVerify.lean`'s `setupOf`, after `buildSession`). -/
def AcceptsZKV (i : RecVStar.Inner) (j : ℕ) (f : VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    (I : Inputs) (own : Array Flock.Registered.Port) (t : Exec.Transcript) : Prop :=
  AcceptsZKR I own t ∧ ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
    sts.forM (RecVStar.checkStmt (some i) I.tags I.tables · I.circuitFile I.partition I.program (some own) (some j)
      (some f) rows) = .ok ()

/-- verity-verify-proof `--zk --registered F --rec-inner F --rec-part J --rec-bindings F --rec-rows F` accepts the live audit's
transcript (`AcceptsZKV`). -/
def LiveAcceptsZKV {n : ℕ} {Reg : Type} (i : RecVStar.Inner) (j : ℕ) (f : VStar.Bindings)
    (rows : Option (Array Flock.Registered.Port)) (I : Inputs) (own : Array Flock.Registered.Port)
    (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop :=
  AcceptsZKV i j f rows I own o.2.2

section Sessions

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ}

/-- **Every session accepts as V*'s statement of its index**, as the prover supplies them: session `j` at the inner
shape `i`, `--rec-part j`, the bindings `bind j` and the shared rows `rows` (`LiveAcceptsZKV`). -/
def AcceptsRV (i : RecVStar.Inner) (bind : Fin S → VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    (zr : (j : Fin S) → (Vs j).OuterR Reg') (b : VRec Vs Reg' dg N) : Prop :=
  ∀ j : Fin S, LiveAcceptsZKV i j (bind j) rows (zr j).I (zr j).own (b j).2

/-- **Every session accepts as V*'s statement of its index**, at the registration the outcome records. -/
def AcceptsVRV (i : RecVStar.Inner) (bind : Fin S → VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg') (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  AcceptsRV i bind rows (zr o.1) o.2

theorem acceptsVR_of_V {i : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)} {zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg'}
    {o : Reg₂ × VRec Vs Reg' dg N} (h : AcceptsVRV i bind rows zr o) : AcceptsVR zr o :=
  fun j => (h j).1

end Sessions

/-- Obligation: open until gap 4 builds `Vs j` from the partition, as this parse of V*'s file.
**V*'s statement `j` at the inner shape `i`, written under its bindings, parses to `Vs j`'s circuit, and `Vs j` has one
unit per instance.** At V*'s tags `t₀` (the `--statement` the verifier requires of a V* session), where `i` has
statements and the `j`th is `s`, the file `circuitFile s (bind j) i.holo` that `vstar stage` writes parses, under `--zk`'s tags
for one table (`Zk.tags` on the coin tree of `Zk.coinSpec #[t₀.table]`, at the file's `mask_log` and count 1) and with
no lookup tables, to `(Vs j).c`; and `(Vs j).nv` is `s`'s instance count. A fact about our code, not a premise: the tags
are a constant of the file, so no session input enters it. -/
def VStarParses {S : ℕ} (t₀ : Flock.Tags) (i : RecVStar.Inner) (bind : Fin S → VStar.Bindings) (Vs : Fin S → VStmt) :
    Prop :=
  ∀ (j : Fin S) (ss : Array VStar.Stmt) (s : VStar.Stmt) (cf : ByteArray), RecVStar.statements i = .ok ss →
    ss[(j : ℕ)]? = some s → RecVStar.circuitFile s (bind j) i.holo = .ok cf →
    HmRow.parse (Flock.Zk.tags t₀ (Flock.Zk.maskLogOf cf) 1) cf
      [] = .ok (Vs j).c ∧ (Vs j).nv = s.instances

/-- Obligation: a fact about our code, not a premise. **The inputs' tags are the statement their name names**
(`Tags.find?`): `verifyCmd` sets every statement up at `stmtTags`, the tags `Tags.find?` returns for `--statement`, so
every input the verifier forms carries the tags of its own name. -/
def TagsFound (I : Inputs) : Prop :=
  Flock.Tags.find? I.tags.name = some I.tags

theorem find?_circuit : Flock.Tags.find? Flock.Tags.circuit.name = some Flock.Tags.circuit := rfl

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

theorem checkFile_ok {s : VStar.Stmt} {b : VStar.Bindings} {circuitB : ByteArray} {holo : Bool}
    (h : RecVStar.checkFile s b circuitB holo = .ok ()) : RecVStar.circuitFile s b holo = .ok circuitB := by
  unfold RecVStar.checkFile at h
  cases hc : RecVStar.circuitFile s b holo with
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

theorem checkParsed_ok {s : VStar.Stmt} {c : Flock.Circuit} (h : RecVStar.checkParsed s c = .ok ()) :
    ∃ u, s.unit = .ok u ∧ c.upv = 1 ∧ c.leafCuts = #[] ∧
      c.leavesIn.getD 0 #[] = (Array.range u.inWords).map Int.ofNat ∧
      c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = RecVStar.vstarRows s.sub u := by
  unfold RecVStar.checkParsed at h
  dsimp only at h
  split at h
  · cases h
  split at h
  · rename_i u hs
    refine ⟨u, hs, ?_⟩
    rw [pure_bind] at h
    split at h
    · cases h
    · rename_i h1
      split at h
      · cases h
      · rename_i h2
        split at h
        · cases h
        · rename_i h3
          simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, Bool.not_eq_true', Array.isEmpty_iff, not_or,
            Classical.not_not, Bool.not_eq_false] at h1 h2
          exact ⟨h1.1, h1.2, h2, Classical.byContradiction fun hne =>
            h3 (by rw [Bool.or_eq_true]; exact Or.inl (bne_iff_ne.2 hne))⟩
  · cases h

theorem checkParsed_v1 {s : VStar.Stmt} {c : Flock.Circuit} (h : RecVStar.checkParsed s c = .ok ()) :
    ∀ p ∈ c.ports.toList, p.v2 = false := by
  unfold RecVStar.checkParsed at h
  dsimp only at h
  split at h
  · cases h
  split at h
  · rw [pure_bind] at h
    split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · rename_i h3
          intro p hp
          cases hv : p.v2
          · rfl
          · exact absurd (by rw [Bool.or_eq_true]; exact Or.inr (List.any_eq_true.2 ⟨p, hp, hv⟩)) h3
  · cases h

theorem checkParsed_flat {s : VStar.Stmt} {c : Flock.Circuit} (h : RecVStar.checkParsed s c = .ok ()) :
    c.tmpl = none := by
  unfold RecVStar.checkParsed at h
  dsimp only at h
  split at h
  · cases h
  · rename_i h0
    cases ht : c.tmpl with
    | none => rfl
    | some _ => simp [ht] at h0

theorem checkParsed_leavesOut {s : VStar.Stmt} {c : Flock.Circuit} (h : RecVStar.checkParsed s c = .ok ()) :
    ∃ u, s.unit = .ok u ∧ c.leavesOut.getD 0 #[] = Array.range u.outWords := by
  unfold RecVStar.checkParsed at h
  dsimp only at h
  split at h
  · cases h
  split at h
  · rename_i u hs
    refine ⟨u, hs, ?_⟩
    rw [pure_bind] at h
    split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · split at h
          · cases h
          · rename_i h4
            exact Classical.byContradiction fun hne => h4 (bne_iff_ne.2 hne)
  · cases h

theorem check_ok {i : RecVStar.Inner} {name : String} {n j : ℕ} {s : VStar.Stmt}
    (h : RecVStar.check i name n (some j) = .ok s) :
    ∃ ss, RecVStar.statements i = .ok ss ∧ ss[j]? = some s ∧ s.name = name ∧ n = s.instances ∧
      ss.toList.filter (·.name == name) = [s] := by
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
        split at h
        · obtain ⟨_, h1, -⟩ := ebind_ok h
          cases h1
        · rename_i t ht
          split at h
          · obtain ⟨_, h1, -⟩ := ebind_ok h
            cases h1
          · rename_i htn
            cases h
            have hm : s ∈ ss.toList.filter (·.name == name) := by rw [hfl]; simp
            have htm : t ∈ ss.toList.filter (·.name == name) := by
              rw [List.mem_filter]
              exact ⟨Array.mem_toList_iff.mpr (Array.mem_of_getElem? ht), by simpa using htn⟩
            rw [hfl, List.mem_singleton] at htm
            rw [List.mem_filter] at hm
            exact ⟨ss, hss, htm ▸ ht, by simpa using hm.2, by simpa using hn, hfl⟩
    · cases h

theorem checkPart_ok {s : VStar.Stmt} {c : Flock.Circuit} (h : RecVStar.checkPart s c = .ok ()) {p : ℕ}
    {nm : String} {pt : VStar.Part} (hs : s = .inner p nm pt) {m k cl j : ℕ}
    (hp : RecResiduals.params c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) {holo : Bool} (hh : RecResiduals.holoOf c = .ok holo)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : RecResiduals.part { sh with holo } j = .ok (S, ns, nv)) :
    pt.S.ports.map (·.2) = ns ∧ pt.S.gf.nv = nv ∧ ∀ P, RecVStar.partPorts { sh with holo } j = .ok P →
      pt.S.ports.map (·.1) = P.toList.map ((RecVStar.portNames { sh with holo })[·]!) := by
  subst hs
  unfold RecVStar.checkPart at h
  rw [hp] at h
  dsimp only at h
  rw [hsh] at h
  dsimp only at h
  rw [hh] at h
  dsimp only at h
  rw [hpart] at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hne
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Classical.not_not] at hne
    refine ⟨hne.1, hne.2, fun P hP => ?_⟩
    rw [hP] at h
    dsimp only at h
    split at h
    · cases h
    · rename_i hn
      simp only [bne_iff_ne, ne_eq, Classical.not_not] at hn
      exact hn

theorem checkCircuit_ok {i : RecVStar.Inner} {j : ℕ} {c : Flock.Circuit} {n : ℕ} {circuitB : ByteArray}
    {f : VStar.Bindings} {s : VStar.Stmt} (h : RecVStar.checkCircuit i j c n circuitB f = .ok s) :
    RecVStar.check i c.unit.name n (some j) = .ok s ∧ RecVStar.checkFile s f circuitB i.holo = .ok () ∧
      RecVStar.checkParsed s c = .ok () ∧ RecVStar.checkPart s c = .ok () := by
  unfold RecVStar.checkCircuit at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨s', hs', h⟩ := ebind_ok h
  obtain ⟨u, hu, h⟩ := ebind_ok h
  obtain ⟨u', hp, h⟩ := ebind_ok h
  obtain ⟨u'', hq, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  cases h
  exact ⟨hs', hu, hp, hq⟩

theorem checkHolo_ok {s : VStar.Stmt} {c : Flock.Circuit} {holo : Bool} (h : RecVStar.checkHolo s c holo = .ok ()) :
    RecResiduals.holoOf c = .ok (RecVStar.partHolo s holo) := by
  unfold RecVStar.checkHolo at h
  obtain ⟨b, hb, h⟩ := ebind_ok h
  rw [hb]
  by_cases he : b = RecVStar.partHolo s holo
  · rw [he]
  · have hne : (b != RecVStar.partHolo s holo) = true := bne_iff_ne.2 he
    simp only [hne, ↓reduceIte] at h
    first | cases h | simp only [throw, throwThe, MonadExceptOf.throw, reduceCtorEq] at h

theorem checkCircuit_holo {i : RecVStar.Inner} {j : ℕ} {c : Flock.Circuit} {n : ℕ} {circuitB : ByteArray}
    {f : VStar.Bindings} {s : VStar.Stmt} (h : RecVStar.checkCircuit i j c n circuitB f = .ok s) :
    RecResiduals.holoOf c = .ok (RecVStar.partHolo s i.holo) := by
  unfold RecVStar.checkCircuit at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, hh, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  cases h
  exact checkHolo_ok hh

theorem checkCircuit_plain {i : RecVStar.Inner} {j : ℕ} {c : Flock.Circuit} {n : ℕ} {circuitB : ByteArray}
    {f : VStar.Bindings} {s : VStar.Stmt} (h : RecVStar.checkCircuit i j c n circuitB f = .ok s)
    (hi : i.holo = false) : RecResiduals.holoOf c = .ok false := by
  have h' := checkCircuit_holo h
  rw [hi] at h'
  exact h'

theorem checkStmt_ok {i : RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)} {st : Flock.Stmt}
    {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f) rows =
      .ok ()) :
    tags.name = Flock.Tags.circuit.name ∧ tables = [] ∧
      ∃ ss s, RecVStar.statements i = .ok ss ∧ ss[j]? = some s ∧ s.name = st.c.unit.name ∧
        RecVStar.population st = s.instances ∧ RecVStar.circuitFile s f i.holo = .ok circuitB ∧
        RecVStar.checkOwn s own = .ok () ∧ ss.toList.filter (·.name == st.c.unit.name) = [s] := by
  unfold RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := ebind_ok h
    cases h1
  · rename_i hn
    split at h
    · obtain ⟨_, h1, -⟩ := ebind_ok h
      cases h1
    · rename_i ht
      split at h
      · split at h
        · obtain ⟨_, h1, -⟩ := ebind_ok h; cases h1
        obtain ⟨_, -, h⟩ := ebind_ok h
        obtain ⟨s, hs, h⟩ := ebind_ok h
        obtain ⟨hck, hf, -⟩ := checkCircuit_ok hs
        obtain ⟨ss, hss, hj, hname, hpop, hfl⟩ := check_ok hck
        obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
        obtain ⟨⟨⟩, ho, -⟩ := ebind_ok h
        exact ⟨by simpa using hn, by simpa using ht, ss, s, hss, hj, hname, hpop, checkFile_ok hf, ho, hfl⟩
      · cases h

-- walking `HmRow.loadPublic`'s do block branch by branch (`walk_step`, as `Layout.setupH_sampled`) exceeds the default
set_option maxHeartbeats 2000000 in
theorem loadPublic_population {tags : Tags} {c : Flock.Circuit} {file : ByteArray} {pop : Flock.Public}
    (h : HmRow.loadPublic tags c file = .ok pop) : pop.population = none := by
  unfold HmRow.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

-- walking `HmRow.drawn`'s do block branch by branch (`walk_step`, as `Layout.setupH_sampled`) exceeds the default
set_option maxHeartbeats 2000000 in
theorem sampled_population {c : Flock.Circuit} {pop pub : Flock.Public} {d : Lean.Json}
    (h : HmRow.drawn c pop d = .ok pub) :
    pub.population.map (·.1) = some pop.n := by
  unfold HmRow.drawn at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

theorem population_of_setupHidden {tags : Tags} {cf pf : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray}
    {st : Flock.Stmt} (h : Flock.Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    ∃ pop, HmOut.loadPublic tags st.c pf = .ok pop ∧ RecVStar.population st = pop.n := by
  have hs := Hidden.setupHidden_spec' h
  obtain ⟨pop, hl, hnone, hsome⟩ := hs.pub
  refine ⟨pop, hl, ?_⟩
  unfold RecVStar.population
  cases draw with
  | none =>
    have hp : st.pub = pop := hnone rfl
    simp [hp, loadPublic_population (Hidden.loadPublic_nil hl), hs.n]
  | some d =>
    rw [sampled_population (hsome d rfl)]
    rfl

theorem setup_of_accepts {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (t : Exec.Transcript)
    {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf (zkInputs x.I) t.record = .ok (spec, sts))
    (hck : ScopeZR x.I) (hi : x.I.tags.isTmpl = false) :
    ∃ h0 : 0 < sts.size, sts[0].c = circZ x.I ∧ RecVStar.population sts[0] = (pubZ x.I).n := by
  have hh' : (zkTags x.I).hiddenOutputs = true := (zkTags_hiddenOutputs x.I).trans x.hTags.1
  obtain ⟨h0, hc0⟩ := setup_zero_c hck.typed x.hTags.1 (hck.parseHF x.hTags.1 hi) hs
  refine ⟨h0, hc0, ?_⟩
  rcases HiddenExec.setupTables_okH hh' hs with ⟨-, st, hst, -, rfl⟩ | ⟨-, -, hj, -, -⟩
  · obtain ⟨pop, hl, hp⟩ := population_of_setupHidden hst
    have hl' : HmOut.loadPublic (zkTags x.I) st.c x.I.publicFile = .ok pop := hl
    rw [show st.c = circZ x.I from hc0, hck.loadH x.hTags.1] at hl'
    cases hl'
    exact hp
  · obtain ⟨d, hd⟩ := hj 0 h0
    obtain ⟨pop, hl, hp⟩ := population_of_setupHidden hd
    have hl' : HmOut.loadPublic (zkTags x.I) sts[0].c x.I.publicFile = .ok pop := hl
    rw [hc0, hck.loadH x.hTags.1] at hl'
    cases hl'
    exact hp

theorem vstarFile_of_accepts {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (t : Exec.Transcript)
    {i : RecVStar.Inner} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (hacc : AcceptsZKV i j f rows x.I x.own t) (hck : ScopeZR x.I) :
    x.I.tags.name = Flock.Tags.circuit.name ∧ x.I.tables = [] ∧
      ∃ ss s, RecVStar.statements i = .ok ss ∧ ss[j]? = some s ∧ s.name = (circZ x.I).unit.name ∧
        (pubZ x.I).n = s.instances ∧ RecVStar.circuitFile s f i.holo = .ok x.I.circuitFile ∧
        RecVStar.checkOwn s x.own = .ok () := by
  obtain ⟨-, spec, sts, hs, hv⟩ := hacc
  have h0 := setup_size_pos x.hTags.1 hs
  have hck1 := array_forM_ok hv 0 h0
  obtain ⟨htag, htab, ss, s, hss, hj, hname, hn, hf, ho, -⟩ := checkStmt_ok hck1
  obtain ⟨_, hcst, hpop⟩ := setup_of_accepts x t hs hck (isTmpl_of_name htag)
  rw [hcst] at hname
  exact ⟨htag, htab, ss, s, hss, hj, hname, hpop ▸ hn, hf, ho⟩

theorem flat_of_acceptsV {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (t : Exec.Transcript)
    {i : RecVStar.Inner} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (hacc : AcceptsZKV i j f rows x.I x.own t) (hck : ScopeZR x.I) : x.I.tags.isTmpl = false :=
  isTmpl_of_name (vstarFile_of_accepts x t hacc hck).1

theorem checkStmt_scope {i : RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)} {st : Flock.Stmt}
    {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f) rows =
      .ok ()) :
    Flock.ProvedScope.scopeOk st.c = true := by
  unfold RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := ebind_ok h
    cases h1
  · split at h
    · obtain ⟨_, h1, -⟩ := ebind_ok h
      cases h1
    · split at h
      · split at h
        · obtain ⟨_, h1, -⟩ := ebind_ok h; cases h1
        rename_i hsc
        simpa using hsc
      · cases h

theorem scope_of_accepts {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (t : Exec.Transcript)
    {i : RecVStar.Inner} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (hacc : AcceptsZKV i j f rows x.I x.own t) (hck : ScopeZR x.I) : Layout.scopeOk (circZ x.I) = true := by
  have hi := flat_of_acceptsV x t hacc hck
  obtain ⟨-, spec, sts, hs, hv⟩ := hacc
  obtain ⟨h0, hcst, -⟩ := setup_of_accepts x t hs hck hi
  rw [← hcst, ← FlockSoundness.Discharge.FailClosed.scopeOk_eq]
  exact checkStmt_scope (array_forM_ok hv 0 h0)

/-- An obligation on our code, no claim: the V* statement files `vstar stage` writes parse to the circuits the
guarantee names. `VStarParses` gives it and asks more. -/
def VStarParsesScoped {S : ℕ} (t₀ : Flock.Tags) (i : RecVStar.Inner) (bind : Fin S → VStar.Bindings)
    (Vs : Fin S → VStmt) : Prop :=
  ∀ (j : Fin S) (ss : Array VStar.Stmt) (s : VStar.Stmt) (cf : ByteArray) (c : Flock.Circuit),
    RecVStar.statements i = .ok ss → ss[(j : ℕ)]? = some s → RecVStar.circuitFile s (bind j) i.holo = .ok cf →
    HmRow.parse (Flock.Zk.tags t₀ (Flock.Zk.maskLogOf cf) 1) cf
      [] = .ok c → Layout.scopeOk c = true → c = (Vs j).c ∧ (Vs j).nv = s.instances

theorem vstarParsesScoped_of {S : ℕ} {t₀ : Flock.Tags} {i : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    {Vs : Fin S → VStmt} (h : VStarParses t₀ i bind Vs) : VStarParsesScoped t₀ i bind Vs :=
  fun j ss s cf _ hss hs hcf hc _ =>
    ⟨Except.ok.inj (hc.symm.trans (h j ss s cf hss hs hcf).1), (h j ss s cf hss hs hcf).2⟩

end FlockSoundness.Discharge.Recursion
