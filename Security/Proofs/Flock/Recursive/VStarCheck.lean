import Flock.RecVStar
import Proofs.Flock.Recursive.Raw
import Proofs.Flock.Soundness.Discharge.FailClosed.One
import Proofs.Flock.Soundness.Discharge.Integrate.Accepted
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Refine.Rep

/-!
# Recursion: the verifier's check that a session is one of V*'s (gap 8, check 1 in the model)

`flock-verify verify --rec-inner F --rec-part J --rec-bindings F` runs `RecVStar.checkStmt` on each statement after
`buildSession` (`FlockVerify.lean`'s `setupOf`): the verifier's own bindings are its partition object's and program's,
the circuit's META binds exactly those, its unit is V*'s statement `J` at the inner shape, its population is that
statement's instance count, and the circuit file is byte for byte the one `vstar stage` writes for that statement under
the verifier's bindings. `AcceptsZKR` doesn't read that check. This file states it and what it gives; `Loads` takes it
to `VStmt.Loads` (`loads_of_verifiesV`), and `RecursiveAudit`'s (i) reads it in its event (`AcceptsVRV`).

* `innerAt spec st`: the verifier's own inner shape at the inner statement, the value `RecVStar.innerOf` returns.
* `AcceptsZKV i j f rows I own t`: flock-verify `--zk --registered F --rec-inner F --rec-part J --rec-bindings F
  --rec-rows F` accepts: `AcceptsZKR`, and each statement passes `checkStmt` at the inner shape `i`, as statement `j`
  under the bindings `f`, against the inputs' tags, lookup tables, circuit file, partition object and program, `own`
  and the shared rows `rows`.
* `LiveAcceptsZKV`, `AcceptsRV`, `AcceptsVRV`: the same of the live audit's transcript, and of every one of V*'s
  sessions as the prover supplies them, session `j` as statement `j` under `bind j` (`acceptsVR_of_V`: they accept).
* `VStarParses` (an obligation, open until gap 4 builds `Vs j` from the partition): at the inner shape `i`, V*'s
  statement `j`'s file under `bind j` parses, at V*'s constant tags and no lookup tables, to `Vs j`'s circuit, and
  `Vs j` has its instances. `VStarParsesScoped` asks it only where that parse is in `Layout.scopeOk`, as every
  accepted session's is (`checkStmt_scope`, `scope_of_accepts`); `VStarParses` gives it (`vstarParsesScoped_of`).
* `TagsFound` (an obligation on our code): the inputs' tags are the statement their name names (`Tags.find?`), as
  `verifyCmd` forms them; `find?_circuit`: `Tags.circuit`'s name names `Tags.circuit`. With refusal (5)'s name
  equation they give V*'s tags.
* `firstDiff?_eq_none`: byte strings without a first difference are equal (`forIn_none_all`, the loop running through).
* `checkFile_ok`, `checkParsed_ok`, `checkParsed_leavesOut`, `checkParsed_v1`, `check_ok`, `checkCircuit_ok`,
  `checkStmt_ok`: what each check gives when it passes.
* `checkHolo_ok`, `checkCircuit_holo`, `checkCircuit_plain`: the circuit check pins the parsed circuit's META `holo`
  (`RecResiduals.holoOf`) to `i`'s holo flag at a part's statement and to `false` elsewhere, so a plain session's
  circuits are plain.
* `population_of_setupHidden`: the set-up statement's population (`RecVStar.population`) is the size of the public
  file `loadPublic` reads, drawn or not (`drawn` records it, `loadPublic` records none and leaves `n` at it).
* `setup_of_accepts`: a raw session that passes the statement check sets up a first table, with `circZ I` as its
  circuit and `(pubZ I).n` as its population.
* `vstarFile_of_accepts`: at a raw session that `AcceptsZKV` accepts as statement `j` under `f` and that passes the
  statement check, the inputs' circuit file is `circuitFile s f i.holo` for V*'s statement `s` at index `j` at `i`, named as the
  session's unit; `(pubZ I).n` is `s`'s instance count, and the verifier's reads hold `s`'s message-round ports.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FailClosed

/-- The verifier's own inner shape at an inner statement `st` set up with `spec`: `RecVStar.innerOf`'s value, its `m`,
`k_log`, two claims and one per region and link point, and its regions. -/
def innerAt (spec : Flock.SessionSpec) (st : Flock.Stmt) : RecVStar.Inner :=
  { m := st.m, kLog := st.c.kLog, claims := 2 + spec.points * st.regions.size, regions := st.regions.size }

/-- **flock-verify `--zk --registered F --rec-inner F --rec-part J --rec-bindings F --rec-rows F` accepts the
transcript**, at the inner shape `i`, as V*'s statement `j` under the verifier's own bindings `f`: `AcceptsZKR`, and each
statement the session sets up passes `RecVStar.checkStmt` at `i`, against the inputs' tags and lookup tables, circuit
file, partition object and program, the verifier's own registered reads `own`, `j`, `f` and its shared rows `rows`
(`FlockVerify.lean`'s `setupOf`, after `buildSession`). -/
def AcceptsZKV (i : RecVStar.Inner) (j : ℕ) (f : VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    (I : Inputs) (own : Array Flock.Registered.Port) (t : Exec.Transcript) : Prop :=
  AcceptsZKR I own t ∧ ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
    sts.forM (RecVStar.checkStmt (some i) I.tags I.tables · I.circuitFile I.partition I.program (some own) (some j)
      (some f) rows) = .ok ()

/-- flock-verify `--zk --registered F --rec-inner F --rec-part J --rec-bindings F --rec-rows F` accepts the live audit's
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

/-- **Sessions accepted as V*'s statements are accepted** (`AcceptsZKV`'s first conjunct). -/
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
    HmRow.parse (Flock.Zk.tags (t₀.withCoins (.tree (Flock.Zk.coinSpec #[t₀.table]))) (Flock.Zk.maskLogOf cf) 1) cf
      [] = .ok (Vs j).c ∧ (Vs j).nv = s.instances

/-- Obligation: a fact about our code, not a premise. **The inputs' tags are the statement their name names**
(`Tags.find?`): `verifyCmd` sets every statement up at `stmtTags`, the tags `Tags.find?` returns for `--statement`, so
every input the verifier forms carries the tags of its own name. -/
def TagsFound (I : Inputs) : Prop :=
  Flock.Tags.find? I.tags.name = some I.tags

/-- **V*'s statement name names V*'s tags**: `Tags.find?` at `Tags.circuit`'s name, evaluated. -/
theorem find?_circuit : Flock.Tags.find? Flock.Tags.circuit.name = some Flock.Tags.circuit := rfl

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

/-- **The file check**: the circuit file is V*'s for `s` under `b`, over an inner session holo or not. -/
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

/-- **The parse check**: the circuit has one unit per VU and no leaf cuts, its input leaves are its unit's input words
in order, and its row ports are V*'s rows of `s`'s unit, by name, words and bits. -/
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

/-- **The parse check's rows are `sha512/row/v1`'s**: no row port of the circuit is a bit row (`v2`). -/
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

/-- **A circuit V*'s parse check passed carries no template.** -/
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

/-- **The parse check's output leaves**: the circuit's output leaves are its unit's output words in order, as V*'s
`leaves_out` (`VStar.Compose`'s META) writes them. -/
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

/-- **The unit check, as V*'s statement `j`**: a unit named `name` over `n` instances is statement `j` of V*'s
statements at `i` (the one statement so named, and statement `j`'s name is it), and `n` is its instance count. -/
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

/-- **The part check**: at a part's statement whose unit's name, shape, META and part are the verifier's part's
(`RecResiduals.check`'s), V*'s part's row ports' elements and v elements are that part's, and its row ports are
named the whole S's (`portNames`) at the part's ports (`partPorts`). -/
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

/-- **The circuit check, as V*'s statement `j` under the verifier's bindings `f`**: the unit check as statement `j`,
the file check under `f` at `i`'s holo flag, the parse check and the part check. -/
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
  cases h
  exact ⟨hs', hu, hp, hq⟩

/-- **The holo check**: the parsed circuit's META `holo` is V*'s file's (`RecVStar.partHolo`). -/
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

/-- **The circuit's META `holo`, as the circuit check pins it**: `i`'s holo flag at a part's statement, `false`
elsewhere (`RecVStar.partHolo`). -/
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
  cases h
  exact checkHolo_ok hh

/-- **A plain session's circuits are plain**: at an inner session whose holo flag is off, every statement the circuit
check passes has no META `holo`. -/
theorem checkCircuit_plain {i : RecVStar.Inner} {j : ℕ} {c : Flock.Circuit} {n : ℕ} {circuitB : ByteArray}
    {f : VStar.Bindings} {s : VStar.Stmt} (h : RecVStar.checkCircuit i j c n circuitB f = .ok s)
    (hi : i.holo = false) : RecResiduals.holoOf c = .ok false := by
  have h' := checkCircuit_holo h
  rw [hi] at h'
  exact h'

/-- **`checkStmt` at an inner shape, the verifier's own reads, statement `j` and bindings `f`**: the statement is set
up under tags named V*'s `Tags.circuit` with no lookup table (refusals 5 and 6), and its circuit file is
`circuitFile s f i.holo` for V*'s statement `s` at index `j` at `i`, named as its unit (the one statement so named), with the
statement's population as its instances, and the verifier's reads hold `s`'s message-round ports. -/
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

-- walking `HmRow.loadPublic`'s do block branch by branch (`walk_step`, as `Layout.setupH_drawn`) exceeds the default
set_option maxHeartbeats 2000000 in
/-- **A loaded public file is a population, not a draw**: `loadPublic` records no population. -/
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

-- walking `HmRow.drawn`'s do block branch by branch (`walk_step`, as `Layout.setupH_drawn`) exceeds the default
set_option maxHeartbeats 2000000 in
/-- **A drawn public file records its population's size**: the registered file's instance count. -/
theorem drawn_population {c : Flock.Circuit} {pop pub : Flock.Public} {d : Lean.Json}
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

/-- **The set-up statement's population is the loaded public file's size**, drawn or not: `HmOut.drawn` records it,
and an undrawn statement's `n` is the file's own (`Spec.pub`, `Spec.n`). -/
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
    rw [drawn_population (hsome d rfl)]
    rfl

/-- **A raw session that passes the statement check at untemplated tags sets up a first table**, whatever the record's
draw: its circuit is the inputs' (`setup_zero_c`, `ScopeZR.parseHF`) and its population is `(pubZ I).n`
(`population_of_setupHidden`, `ScopeZR.loadH`; at several tables, table 0's tags differ from the session's in the
identity alone, which `HmOut.loadPublic` does not read). -/
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

/-- **A raw session that `AcceptsZKV` accepts as V*'s statement `j` under the bindings `f` loads that statement's
file**, where its inputs pass the statement check: its first table's statement has the inputs' circuit and
`(pubZ I).n` as its population (`setup_of_accepts`), and `checkStmt_ok` at it: the inputs' tags are named the
circuit's and they carry no lookup tables, and the inputs' circuit file is `circuitFile s f i.holo` for V*'s statement
`s` at index `j`, whose instance count is `(pubZ I).n`. -/
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

/-- **A raw session that `AcceptsZKV` accepts is at untemplated tags**: refusal (5) names its tags as `Tags.circuit`
(`vstarFile_of_accepts`), and `Tags.isTmpl` reads the name alone (`isTmpl_of_name`). -/
theorem flat_of_acceptsV {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (t : Exec.Transcript)
    {i : RecVStar.Inner} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (hacc : AcceptsZKV i j f rows x.I x.own t) (hck : ScopeZR x.I) : x.I.tags.isTmpl = false :=
  isTmpl_of_name (vstarFile_of_accepts x t hacc hck).1

/-- **`checkStmt` at an inner shape passes only at a circuit in `ProvedScope.scopeOk`**: under `--rec-inner` it refuses
a V* statement whose circuit has a row outside `sha512/row/v1`. -/
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

/-- **A raw session that `AcceptsZKV` accepts has its circuit in `Layout.scopeOk`**, where its inputs pass the
statement check: `checkStmt` passes at its first table's statement (`setup_of_accepts`), whose circuit is the inputs',
so that circuit is in `ProvedScope.scopeOk` (`checkStmt_scope`), which is `Layout.scopeOk` (`scopeOk_eq`). -/
theorem scope_of_accepts {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (t : Exec.Transcript)
    {i : RecVStar.Inner} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (hacc : AcceptsZKV i j f rows x.I x.own t) (hck : ScopeZR x.I) : Layout.scopeOk (circZ x.I) = true := by
  have hi := flat_of_acceptsV x t hacc hck
  obtain ⟨-, spec, sts, hs, hv⟩ := hacc
  obtain ⟨h0, hcst, -⟩ := setup_of_accepts x t hs hck hi
  rw [← hcst, ← FlockSoundness.Discharge.FailClosed.scopeOk_eq]
  exact checkStmt_scope (array_forM_ok hv 0 h0)

/-- **V*'s statement `j`'s file, where the verifier parses it into scope, parses to `Vs j`'s circuit, and `Vs j` has
its instances**: `VStarParses` at the files whose parse, at V*'s constant tags and no lookup tables, is a circuit in
`Layout.scopeOk`. An accepted session's file is one (`ScopeZR.parseH`, `scope_of_accepts`), so this asks nothing of a
file the verifier refuses; `VStarParses` gives it (`vstarParsesScoped_of`).

Assumption record. No claim: an obligation, a fact about our code, open until gap 4 builds `Vs` from the partition. It
assumes that the V* statement files `vstar stage` writes parse to the circuits the guarantee names, which is a
deterministic computation on public inputs. It follows from `VStarParses` (`vstarParsesScoped_of`), which the lock
already lists, and it asks less: `VStarParses` also asks every listed file to parse. At every inner shape and binding,
`FlockC3.VStarList.vstarParsesScoped_of_list` builds a family `Vs` that satisfies it. So it fixes which `Vs` the
guarantee speaks of, and taking `VStarParses` instead would weaken the guarantee. It is no hash premise and no premise
about the world. -/
def VStarParsesScoped {S : ℕ} (t₀ : Flock.Tags) (i : RecVStar.Inner) (bind : Fin S → VStar.Bindings)
    (Vs : Fin S → VStmt) : Prop :=
  ∀ (j : Fin S) (ss : Array VStar.Stmt) (s : VStar.Stmt) (cf : ByteArray) (c : Flock.Circuit),
    RecVStar.statements i = .ok ss → ss[(j : ℕ)]? = some s → RecVStar.circuitFile s (bind j) i.holo = .ok cf →
    HmRow.parse (Flock.Zk.tags (t₀.withCoins (.tree (Flock.Zk.coinSpec #[t₀.table]))) (Flock.Zk.maskLogOf cf) 1) cf
      [] = .ok c → Layout.scopeOk c = true → c = (Vs j).c ∧ (Vs j).nv = s.instances

theorem vstarParsesScoped_of {S : ℕ} {t₀ : Flock.Tags} {i : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    {Vs : Fin S → VStmt} (h : VStarParses t₀ i bind Vs) : VStarParsesScoped t₀ i bind Vs :=
  fun j ss s cf _ hss hs hcf hc _ =>
    ⟨Except.ok.inj (hc.symm.trans (h j ss s cf hss hs hcf).1), (h j ss s cf hss hs hcf).2⟩

end FlockSoundness.Discharge.Recursion
