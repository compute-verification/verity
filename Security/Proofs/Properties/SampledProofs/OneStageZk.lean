import Properties.SampledProofs.OneStageZk
import Proofs.Properties.SampledProofs
import Proofs.Flock.Soundness.Discharge.ZkExec.Event
import Proofs.Flock.Recursive.Witness

/-! # Proofs of one-stage v2's property (`Properties.SampledProofs.OneStageZk`)

`checkBlobs` accepts only when `run` returned a run whose every session accepted and v1's check accepted its verdicts.
`run` is a `foldlM` of `step` over the evidence's sessions, so `fold_ok` reads each session's acceptance off the fold, one
`step` at a time (`step_ok`): its record is the blob its digest names (`get_ofArray`), and its setup (`setupAt_ok`,
`buildSessionOf_ok`) is `Flock.Witness.setupOf` at its record's draw, the event `AcceptsZK` sets up. v1's proved
`acceptMeans` gives the verdicts' acceptance, so at least one session ran. `acceptsZK_iff` is the bridge to the
model's `ZkExec.AcceptsZK` at `Witness.toExec I`.
-/

open Lean (Json)
open SampledProofs SampledProofs.OneStageZk

namespace Properties.SampledProofs.OneStageZk

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem map_ok {ε α β : Type} {x : Except ε α} {f : α → β} {b : β} (h : x.map f = .ok b) :
    ∃ a, x = .ok a ∧ f a = b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, Except.ok.inj h⟩

/-! `Except`'s evaluation, for `simp only` on a step whose every discriminant is fixed. -/

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl
theorem error_bind {ε α β : Type} (e : ε) (f : α → Except ε β) : (Except.error e >>= f) = Except.error e := rfl
theorem throw_bind {ε α β : Type} (e : ε) (f : α → Except ε β) : (throw e >>= f : Except ε β) = throw e := rfl
theorem pure_bind' {ε α β : Type} (a : α) (f : α → Except ε β) : (pure a >>= f : Except ε β) = f a := rfl
theorem map_ok' {ε α β : Type} (a : α) (f : α → β) : (f <$> (Except.ok a : Except ε α)) = Except.ok (f a) := rfl
theorem throw_eq_ok {ε α : Type} (e : ε) (a : α) : ((throw e : Except ε α) = Except.ok a) ↔ False :=
  ⟨fun h => nomatch h, False.elim⟩
theorem error_eq_ok {ε α : Type} (e : ε) (a : α) : ((Except.error e : Except ε α) = Except.ok a) ↔ False :=
  ⟨fun h => nomatch h, False.elim⟩
theorem pure_eq_ok {ε α : Type} (a b : α) : ((pure a : Except ε α) = Except.ok b) ↔ a = b :=
  ⟨fun h => Except.ok.inj h, fun h => h ▸ rfl⟩

theorem beq_bytes {a b : ByteArray} (h : (a == b) = true) : a = b := by
  have h' : (a.data == b.data) = true := h
  exact ByteArray.ext (beq_iff_eq.mp h')

/-- A blob found under a digest is bytes whose SHA-512 the digest names. -/
theorem get_ofArray {blobs : Array ByteArray} {d w : String} {b : ByteArray}
    (h : (Core.Blobs.ofArray blobs).get d w = .ok b) : Names d b := by
  unfold Core.Blobs.get at h
  obtain ⟨d', hu, h⟩ := bind_ok h
  by_cases hc : (d'.size != 64 || Flock.hexOf d' != d) = true
  · rw [if_pos hc] at h
    cases h
  · rw [if_neg hc] at h
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hc
    split at h
    · rename_i fst snd hf
      have hb : snd = b := Except.ok.inj h
      have hp := Array.find?_some hf
      have hm := Array.mem_of_find?_eq_some hf
      simp only [Core.Blobs.ofArray, Array.mem_map] at hm
      obtain ⟨b0, -, hb0⟩ := hm
      simp only [Prod.mk.injEq] at hb0
      obtain ⟨h1, h2⟩ := hb0
      have hd : fst = d' := beq_bytes hp
      unfold Names
      rw [← hb, ← h2, h1, hd, hc.2]
    · cases h

/-- What `buildSessionOf` returns is `Stmt.setupTables`'s session at its arguments, its circuit parsed as `circuitOf`
parses it under the tags. -/
theorem buildSessionOf_ok {tags : Flock.Tags} {circuitB pubB : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {count : Nat} {draw partition : Option Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {r : Flock.SessionSpec × Array Flock.Stmt}
    (h : buildSessionOf (Flock.ProvedScope.circuitOf tags circuitB tables) tags circuitB pubB coins tables count draw
      partition program registered = .ok r) :
    Flock.Stmt.setupTables tags circuitB pubB coins count tables draw partition program = .ok r := by
  unfold buildSessionOf at h
  by_cases hrow : (!tags.hm96Rows) = true
  · rw [if_pos hrow] at h
    cases h
  · rw [if_neg hrow] at h
    obtain ⟨⟨spec, sts⟩, hs, h⟩ := bind_ok h
    have hs' : Flock.Stmt.setupTables tags circuitB pubB coins count tables draw partition program = .ok (spec, sts) := by
      by_cases hh : tags.hiddenOutputs = true
      · rw [if_pos hh] at hs
        rw [Flock.Stmt.setupTables_eq]
        unfold Flock.ProvedScope.circuitOf at hs
        rw [if_pos hh] at hs
        exact hs
      · rw [if_neg hh] at hs
        exact hs
    obtain ⟨_, -, h⟩ := bind_ok h
    obtain ⟨_, -, h⟩ := bind_ok h
    rw [← Except.ok.inj h]
    exact hs'

/-- What `setupAt` sets up at `draw` is `Stmt.setupTables`'s session under `--zk`'s tags and coins, every table's
statement passing `Zk.shapeOk`, with its verifier-side setups. -/
theorem setupAt_ok {I : Flock.Witness.Inputs} {rd : Reads} {draw : Option Json} {spec : Flock.SessionSpec}
    {tabs : Array Flock.Setup} {rest : Json × Json} (h : setupAt I rd draw = .ok (spec, tabs, rest)) :
    ∃ sts, Flock.Stmt.setupTables (Flock.Witness.zkTags I) I.circuitFile I.publicFile (Flock.Witness.zkCoins I) I.count
        I.tables draw I.partition I.program = .ok (spec, sts) ∧ sts.forM Flock.Zk.shapeOk = .ok () ∧
      tabs = sts.map Flock.Setup.ofCircuit := by
  unfold setupAt at h
  split at h
  · next spec' sts' hb =>
    obtain ⟨_, -, hb⟩ := bind_ok hb
    obtain ⟨_, -, hb⟩ := bind_ok hb
    obtain ⟨r, hr, hb⟩ := bind_ok hb
    obtain ⟨_, -, hb⟩ := bind_ok hb
    obtain ⟨_, hshape, hb⟩ := bind_ok hb
    obtain ⟨_, -, hb⟩ := bind_ok hb
    have hr' : r = (spec', sts') := Except.ok.inj hb
    subst hr'
    obtain ⟨_, -, hf⟩ := map_ok h
    simp only [Prod.mk.injEq] at hf
    obtain ⟨rfl, rfl, -⟩ := hf
    exact ⟨sts', buildSessionOf_ok hr, hshape, rfl⟩
  · cases h

/-! ## One step, its pieces named

`step'` is `step` with the session's setup, rep files, coin-tree key and verdict as functions of their own, so a proof
cases on the verdict as one term; `step_eq` is `rfl`. -/

/-- The session's setup at its record's own draw. -/
def stOf (I : Flock.Witness.Inputs) (rd : Reads) (record : String) :
    Flock.V (Flock.SessionSpec × Array Flock.Setup × Json × Json) :=
  setupAt I rd (Flock.Witness.recordDraw record)

/-- The session's rep files, the blobs its `reps` name. -/
def filesOf (I : Flock.Witness.Inputs) (rd : Reads) (bs : Core.Blobs) (s : Session) (record : String) :
    Flock.V (Array ByteArray) := do
  let (spec, _, _) ← stOf I rd record
  spec.repStreams.mapM fun (_, _, w) => do
    let n := s!"{w.replace "/" "."}.bin"
    let some (_, d) := s.reps.find? (·.1 == n) | throw s!"session files: the session names no rep file {n}"
    (bs.get d s!"session files: {n}")

/-- The session's coin-tree key. -/
def coinKeyOf (I : Flock.Witness.Inputs) (rd : Reads) (record : String) : Option String :=
  match stOf I rd record with
  | .ok (spec, _, _) =>
    if spec.coinTree.isNone then none
    else (Json.parse record).toOption.bind fun j =>
      (j.getObjVal? "coin_tree").toOption.bind fun ct => (ct.getObjValAs? String "key").toOption
  | .error _ => none

/-- The session's verdict. -/
def verdictOf (I : Flock.Witness.Inputs) (rd : Reads) (bs : Core.Blobs) (scope : Flock.V Unit) (r : Run) (s : Session)
    (record : String) : Flock.V Unit := do
  let proofs ← filesOf I rd bs s record
  if let some k := coinKeyOf I rd record then Flock.CoinTree.checkFresh r.coinKeys k
  let (spec, tabs, _) ← stOf I rd record
  Flock.Zk.verify spec tabs record proofs
  scope
  Flock.ProvedScope.sizesOk true (tabs.map (·.m))
  if (Flock.Witness.recordDraw record).isNone then
    Flock.ProvedScope.drawLaw true (Flock.Witness.zkTags I) false (Flock.Witness.recordDraw record)
  Flock.RecSetup.fullDraw none false (Flock.Witness.recordDraw record) record
  Flock.RecSetup.innerDrawn none
  Flock.ProvedScope.programHeld I.program
  Flock.RecordText.check record

def step' (I : Flock.Witness.Inputs) (rd : Reads) (bs : Core.Blobs) (scope : Flock.V Unit) (programSha : Option String)
    (want : Json) (r : Run) (s : Session) : Flock.V Run := do
  let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
    | throw s!"session {s.name}'s record is not UTF-8"
  let coinKeys := match coinKeyOf I rd record with | some k => r.coinKeys.push k | none => r.coinKeys
  let (ok, why) := match verdictOf I rd bs scope r s record with | .ok () => (true, "") | .error e => (false, e)
  if ok && (programSha.map Json.str) != some want then
    throw s!"session {s.name}'s verdict names program {programSha.getD "none"}, not the partition object's {want.compress}"
  let units ← if ok then provedUnits I.publicFile record else pure #[]
  let (part, opened) := match stOf I rd record with | .ok (_, _, p, o) => (p, o) | .error _ => (Json.null, Json.null)
  pure { coinKeys, all := r.all && ok, proved := r.proved ++ units,
         out := r.out.push (s!"flock-verify/{s.name}", Json.mkObj ([("accepted", Json.bool ok),
           ("of_record", Json.bool true), ("session", Json.str s.name), ("why", Json.str why), ("units", natsJson units)] ++
           (if part.isNull then [] else [("partition", part)]) ++
           (match programSha with | some h => [("program_sha512", Json.str h)] | none => []) ++
           (match rd.registered with | some r => [("registered", Flock.Registered.finding r)] | none => []) ++
           (if opened.isNull then [] else [("openings", opened)]))) }

theorem step_eq (I : Flock.Witness.Inputs) (rd : Reads) (bs : Core.Blobs) (scope : Flock.V Unit)
    (programSha : Option String) (want : Json) (r : Run) (s : Session) :
    step I rd bs scope programSha want r s = step' I rd bs scope programSha want r s := rfl

/-- An accepted verdict is flock-verify `--zk`'s acceptance of the record and the rep files at the record's own draw. -/
theorem verdictOf_ok {I : Flock.Witness.Inputs} {rd : Reads} {bs : Core.Blobs} {scope : Flock.V Unit} {r : Run}
    {s : Session} {record : String} (h : verdictOf I rd bs scope r s record = .ok ()) :
    ∃ proofs, AcceptsZK I record proofs := by
  unfold verdictOf at h
  obtain ⟨proofs, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨⟨spec, tabs, rest⟩, hst, h⟩ := bind_ok h
  obtain ⟨_, hv, -⟩ := bind_ok h
  obtain ⟨sts, hset, hshape, rfl⟩ := setupAt_ok hst
  exact ⟨proofs, spec, sts, hset, hshape, hv⟩

/-- A step that leaves every session accepted, on an accepted verdict: it pushes one verdict and adds the units the
session proved, and the program is the partition object's. -/
theorem step'_ok {I : Flock.Witness.Inputs} {rd : Reads} {bs : Core.Blobs} {scope : Flock.V Unit}
    {programSha : Option String} {want : Json} {r r' : Run} {s : Session}
    (h : step' I rd bs scope programSha want r s = .ok r') (hall : r'.all = true) :
    r.all = true ∧ (∃ h, programSha = some h ∧ (Json.str h == want) = true) ∧ r'.out.size = r.out.size + 1 ∧
      ∃ b record units, bs.get s.record s!"session {s.name}'s record" = .ok b ∧ String.fromUTF8? b = some record ∧
        verdictOf I rd bs scope r s record = .ok () ∧ provedUnits I.publicFile record = .ok units ∧
        r'.proved = r.proved ++ units := by
  unfold step' at h
  obtain ⟨b, hb, h⟩ := bind_ok h
  rcases hrec : String.fromUTF8? b with _ | record
  · simp only [hrec, throw_eq_ok] at h
  rcases hv : verdictOf I rd bs scope r s record with e | ⟨⟨⟩⟩
  · rcases hst : stOf I rd record with e' | ⟨spec, tabs, part, opened⟩ <;>
    (simp only [hrec, hv, hst, ok_bind, error_bind, throw_bind, pure_bind', map_ok', throw_eq_ok, error_eq_ok,
        pure_eq_ok, Bool.false_and, Bool.false_eq_true, ↓reduceIte] at h
     subst h
     simp at hall)
  rcases programSha with _ | ph
  · simp only [hrec, hv, ok_bind, error_bind, throw_bind, pure_bind', throw_eq_ok, error_eq_ok, pure_eq_ok,
      Option.map_none, bne, Option.none_beq_some, Bool.not_false, Bool.true_and, eq_self_iff_true, ↓reduceIte] at h
  rcases hj : (Json.str ph == want) with _ | _
  · simp only [hrec, hv, hj, ok_bind, error_bind, throw_bind, pure_bind', throw_eq_ok, error_eq_ok, pure_eq_ok,
      Option.map_some, bne, Option.some_beq_some, Bool.not_false, Bool.true_and, eq_self_iff_true, ↓reduceIte] at h
  rcases hu : provedUnits I.publicFile record with eu | units
  · simp only [hrec, hv, hj, hu, ok_bind, error_bind, throw_bind, pure_bind', throw_eq_ok, error_eq_ok, pure_eq_ok,
      Option.map_some, bne, Option.some_beq_some, Bool.not_true, Bool.true_and, Bool.false_eq_true, eq_self_iff_true,
      ↓reduceIte] at h
  rcases hst : stOf I rd record with e' | ⟨spec, tabs, part, opened⟩ <;>
  (simp only [hrec, hv, hj, hu, hst, ok_bind, error_bind, throw_bind, pure_bind', map_ok', throw_eq_ok, error_eq_ok,
      pure_eq_ok, Option.map_some, bne, Option.some_beq_some, Bool.not_true, Bool.true_and, Bool.false_eq_true,
      eq_self_iff_true, ↓reduceIte] at h
   subst h
   exact ⟨by simpa using hall, ⟨ph, rfl, hj⟩, by simp, b, record, units, hb, hrec, hv, hu, rfl⟩)

/-- One session's step, when the run it starts goes on to accept every session: the run so far accepted every one,
this session's record and proof files are accepted by flock-verify `--zk` at its record's own draw, its program is the
partition object's, it pushes one verdict, and it adds the units it proved. -/
theorem step_ok {I : Flock.Witness.Inputs} {rd : Reads} {bs : Core.Blobs} {scope : Flock.V Unit}
    {programSha : Option String} {want : Json} {r r' : Run} {s : Session}
    (hbs : ∀ d w b, bs.get d w = .ok b → Names d b)
    (h : step I rd bs scope programSha want r s = .ok r') (hall : r'.all = true) :
    r.all = true ∧ (∃ h, programSha = some h ∧ (Json.str h == want) = true) ∧ r'.out.size = r.out.size + 1 ∧
      ∃ record proofs units, SessionAccepted I s record proofs units ∧ r'.proved = r.proved ++ units := by
  rw [step_eq] at h
  obtain ⟨hall0, hprog, hsize, b, record, units, hb, hrec, hv, hunits, hproved⟩ := step'_ok h hall
  obtain ⟨proofs, hacc⟩ := verdictOf_ok hv
  exact ⟨hall0, hprog, hsize, record, proofs, units, ⟨⟨b, hbs _ _ _ hb, hrec⟩, hacc, hunits⟩, hproved⟩

/-- The fold of `step` over the sessions `l`, when it ends in a run that accepted every session. -/
theorem fold_ok {I : Flock.Witness.Inputs} {rd : Reads} {bs : Core.Blobs} {scope : Flock.V Unit}
    {programSha : Option String} {want : Json} (hbs : ∀ d w b, bs.get d w = .ok b → Names d b) :
    ∀ (l : List Session) (r0 r : Run), l.foldlM (step I rd bs scope programSha want) r0 = .ok r → r.all = true →
      r0.all = true ∧ (l ≠ [] → ∃ h, programSha = some h ∧ (Json.str h == want) = true) ∧
        r.out.size = r0.out.size + l.length ∧
        ∃ ss : List (String × Array ByteArray × Array Nat), ss.length = l.length ∧
          (∀ x ∈ l.zip ss, SessionAccepted I x.1 x.2.1 x.2.2.1 x.2.2.2) ∧
          r.proved = ss.foldl (fun acc x => acc ++ x.2.2) r0.proved
  | [], r0, r, h, hall => by
    have hr : r0 = r := Except.ok.inj h
    subst hr
    exact ⟨hall, fun hn => absurd rfl hn, rfl, [], rfl, fun x hx => by simp at hx, rfl⟩
  | s :: l, r0, r, h, hall => by
    rw [List.foldlM_cons] at h
    obtain ⟨r1, h1, h⟩ := bind_ok h
    obtain ⟨hall1, -, hsize, ss, hlen, hss, hproved⟩ := fold_ok hbs l r1 r h hall
    obtain ⟨hall0, hprog, hsize1, record, proofs, units, hacc, hp1⟩ := step_ok hbs h1 hall1
    refine ⟨hall0, fun _ => hprog, ?_, (record, proofs, units) :: ss, by simp [hlen], ?_, ?_⟩
    · rw [hsize, hsize1, List.length_cons]
      omega
    · intro x hx
      simp only [List.zip_cons_cons, List.mem_cons] at hx
      rcases hx with rfl | hx
      · exact hacc
      · exact hss x hx
    · rw [hproved, hp1]
      rfl

/-- What `run` returns, when every session accepted. -/
theorem run_ok {p : Own} {pub : String} {sessions : Array Session} {want : Array Nat} {blobs : Array ByteArray} {r : Run}
    (h : run p pub sessions (some want) (Core.Blobs.ofArray blobs) = .ok r) (hall : r.all = true) :
    ∃ (tags : Flock.Tags) (circuitB pubB : ByteArray), Flock.Tags.find? STATEMENT = some tags ∧ Names p.circuit circuitB ∧
      Names pub pubB ∧ r.out.size = sessions.size ∧
      (0 < sessions.size → ∃ h, programDigest p.program = some h ∧ (Json.str h == getD p.partition "program") = true) ∧
      ∃ ss : List (String × Array ByteArray × Array Nat), ss.length = sessions.size ∧
        (∀ x ∈ sessions.toList.zip ss,
          SessionAccepted (inputsOf tags p circuitB pubB (Core.Blobs.ofArray blobs)) x.1 x.2.1 x.2.2.1 x.2.2.2) ∧
        (ss.foldl (fun acc x => acc ++ x.2.2) #[]).qsort (· < ·) = want := by
  unfold run at h
  rcases htags : Flock.Tags.find? STATEMENT with _ | tags
  · simp only [htags, throw_eq_ok] at h
  rcases hpl : tags.plainLeaves with _ | _ <;>
  rcases hreg : (p.registered.isNone && readsRegistered p.programJ) with _ | _ <;>
  rcases hc : (Core.Blobs.ofArray blobs).get p.circuit "circuit" with _ | circuitB <;>
  rcases hp : (Core.Blobs.ofArray blobs).get pub "public file" with _ | pubB <;>
  rcases hrd : readsOf p.registered with _ | rd <;>
  simp only [htags, hpl, hreg, hc, hp, hrd, ok_bind, error_bind, throw_bind, pure_bind', throw_eq_ok, error_eq_ok,
    Bool.false_eq_true, eq_self_iff_true, ↓reduceIte] at h
  obtain ⟨r1, hf, h⟩ := bind_ok h
  rw [← Array.foldlM_toList] at hf
  rcases ha1 : r1.all with _ | _ <;>
  rcases hq : (r1.proved.qsort (· < ·) != want) with _ | _ <;>
  simp only [ha1, hq, ok_bind, error_bind, throw_bind, pure_bind', throw_eq_ok, error_eq_ok, pure_eq_ok,
    Bool.true_and, Bool.false_and, Bool.false_eq_true, eq_self_iff_true, ↓reduceIte] at h <;>
  subst h <;>
  simp only [ha1, Bool.false_eq_true] at hall
  obtain ⟨-, hprog, hsize, ss, hlen, hss, hproved⟩ :=
    fold_ok (bs := Core.Blobs.ofArray blobs) (fun _ _ _ hb => get_ofArray hb) sessions.toList _ _ hf ha1
  have hunits := bne_eq_false_iff_eq.mp hq
  rw [hproved] at hunits
  refine ⟨tags, circuitB, pubB, by first | exact htags | rfl, get_ofArray hc, get_ofArray hp, by rw [hsize]; simp,
    fun hpos => ?_,
    ss, by simpa using hlen, hss, hunits⟩
  exact hprog (List.ne_nil_of_length_pos (by simpa using hpos))

theorem acceptMeans : AcceptMeans := by
  intro p c e blobs pub prof h
  unfold checkBlobs at h
  split at h
  · cases h
  · next r hrun =>
    split at h
    · next pub' prof' hchk =>
      by_cases hall : r.all = true
      · rw [if_pos hall] at h
        cases h
        obtain ⟨tags, circuitB, pubB, htags, hcirc, hpub, hsize, hprog, ss, hlen, hss, hunits⟩ := run_ok hrun hall
        obtain ⟨-, -, -, -, -, hva, -⟩ := Properties.SampledProofs.acceptMeans _ _ _ _ _ hchk
        obtain ⟨x, hx, -⟩ := hva.2.1
        have hpos : 0 < e.sessions.size := hsize ▸ Array.size_pos_of_mem hx
        exact ⟨tags, circuitB, pubB, ss, r.out, htags, hcirc, hpub, hpos, hlen, hss, hunits, hprog hpos, hchk⟩
      · rw [if_neg hall] at h
        cases h
    · next o hne hchk =>
      rw [hchk] at h
      exact absurd h (hne _ _)

/-- **`AcceptsZK` is the model's `ZkExec.AcceptsZK`** at the witness's inputs as the model reads them (`toExec`), on the
transcript of the session's record and proof files. -/
theorem acceptsZK_iff (I : Flock.Witness.Inputs) (record : String) (proofs : Array ByteArray) :
    AcceptsZK I record proofs ↔
      FlockSoundness.Discharge.ZkExec.AcceptsZK (FlockSoundness.Discharge.Recursion.Witness.toExec I) ⟨record, proofs⟩ :=
  Iff.rfl

end Properties.SampledProofs.OneStageZk
