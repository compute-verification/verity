import Properties.SampledProofs.OneStageZk
import Proofs.Properties.SampledProofs
import Proofs.Flock.Soundness.Discharge.ZkExec.Event
import Proofs.Flock.Recursive.Witness

/-! # Proofs of one-stage v2's property (`Properties.SampledProofs.OneStageZk`)

`checkBlobs` accepts only when `run` returned a run whose every session accepted and v1's check accepted its verdicts.
`run` is a `foldlM` of `step` over the evidence's sessions, so `fold_ok` reads each session's acceptance off the fold, one
`step` at a time (`step_ok`): its record is the blob its digest names (`get_ofArray`), its setup (`setupAt_ok`, through
`sessionAt_ok` and `buildSessionOf_ok`) is `Flock.Witness.setupOf` at its record's draw with every table's shape, the
event `AcceptsZK` sets up, `Zk.verify` accepted its proof files (`verdictOf_ok`), and it proved its record's units
(`push_ok`). `run_ok` adds the files, each one of the blobs, and the proved units; v1's proved `acceptMeans` gives the
verdicts' acceptance, so at least one session ran. `acceptMeans_nontrivial`: an accept-everything check would fail
`AcceptMeans`. `acceptsZK_iff` is the bridge to the model's `ZkExec.AcceptsZK` at `Witness.toExec I`.
-/

open Lean (Json)
open Flock (V)
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
  ⟨(fun h => nomatch h), False.elim⟩
theorem error_eq_ok {ε α : Type} (e : ε) (a : α) : ((Except.error e : Except ε α) = Except.ok a) ↔ False :=
  ⟨(fun h => nomatch h), False.elim⟩
theorem pure_eq_ok {ε α : Type} (a b : α) : ((pure a : Except ε α) = Except.ok b) ↔ a = b :=
  ⟨fun h => Except.ok.inj h, fun h => h ▸ rfl⟩

theorem beq_bytes {a b : ByteArray} (h : (a == b) = true) : a = b := by
  have h' : (a.data == b.data) = true := h
  exact ByteArray.ext (beq_iff_eq.mp h')

/-- A blob found under a digest is bytes whose SHA-512 the digest names. -/
theorem get_ofArray {blobs : Array ByteArray} {d w : String} {b : ByteArray}
    (h : (Core.Blobs.ofArray blobs).get d w = .ok b) : Blob blobs d b := by
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
      obtain ⟨b0, hb0m, hb0⟩ := hm
      simp only [Prod.mk.injEq] at hb0
      obtain ⟨h1, h2⟩ := hb0
      have hd : fst = d' := beq_bytes hp
      refine ⟨?_, ?_⟩
      · rw [← hb, ← h2]
        exact hb0m
      · unfold Names
        rw [← hb, ← h2, h1, hd, hc.2]
    · cases h

/-- `buildSessionOf` on `ProvedScope.circuitOf`'s circuit sets up what `Stmt.setupTables` sets up. -/
theorem buildSessionOf_ok {tags : Flock.Tags} {cB pB : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {count : Nat} {draw partition : Option Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (h : buildSessionOf (Flock.ProvedScope.circuitOf tags cB tables) tags cB pB coins tables count draw partition program
      registered = .ok (spec, sts)) :
    Flock.Stmt.setupTables tags cB pB coins count tables draw partition program = .ok (spec, sts) := by
  unfold buildSessionOf at h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨r, h1, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  cases h
  cases hh : tags.hiddenOutputs
  · simpa [hh] using h1
  · first
    | simpa [Flock.ProvedScope.circuitOf, hh, Flock.Stmt.setupTables_eq] using h1
    | (rw [Flock.Stmt.setupTables_eq]; simpa [Flock.ProvedScope.circuitOf, hh] using h1)

/-- `sessionAt` sets up `Witness.setupOf`'s statement at the draw, and every table passes `Zk.shapeOk`. -/
theorem sessionAt_ok {I : Flock.Witness.Inputs} {rd : Reads} {draw : Option Json} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} (h : sessionAt I rd draw = .ok (spec, sts)) :
    Flock.Stmt.setupTables (Flock.Witness.zkTags I) I.circuitFile I.publicFile (Flock.Witness.zkCoins I) I.count
        I.tables draw I.partition I.program = .ok (spec, sts) ∧ sts.forM Flock.Zk.shapeOk = .ok () := by
  unfold sessionAt at h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨r, hr, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨u, hs, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  cases h
  cases u
  exact ⟨buildSessionOf_ok hr, hs⟩

theorem setupAt_ok {I : Flock.Witness.Inputs} {rd : Reads} {draw : Option Json} {spec : Flock.SessionSpec}
    {tabs : Array Flock.Setup} {x : Json × Json} (h : setupAt I rd draw = .ok (spec, tabs, x)) :
    ∃ sts, Flock.Stmt.setupTables (Flock.Witness.zkTags I) I.circuitFile I.publicFile (Flock.Witness.zkCoins I) I.count
        I.tables draw I.partition I.program = .ok (spec, sts) ∧ sts.forM Flock.Zk.shapeOk = .ok () ∧
      tabs = sts.map Flock.Setup.ofCircuit := by
  unfold setupAt at h
  obtain ⟨⟨spec', sts⟩, h1, h⟩ := bind_ok h
  obtain ⟨hset, hshape⟩ := sessionAt_ok (mapError_ok h1)
  obtain ⟨_, _, h⟩ := bind_ok h
  cases h
  exact ⟨sts, hset, hshape, rfl⟩

/-- A verdict that accepts ran `Zk.verify` on the session's setup and proof files, and accepted. -/
theorem verdictOf_ok {I : Flock.Witness.Inputs} {scope : V Unit} {ck : Array String} {record : String}
    {draw : Option Json} {st : SetUp} {files : V (Array ByteArray)} {coinKey : Option String}
    (h : verdictOf I scope ck record draw st files coinKey = .ok ()) :
    ∃ proofs spec tabs x, files = .ok proofs ∧ st = .ok (spec, tabs, x) ∧
      Flock.Zk.verify spec tabs record proofs = .ok () := by
  unfold verdictOf at h
  obtain ⟨proofs, hf, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨⟨spec, tabs, x⟩, hst, h⟩ := bind_ok h
  obtain ⟨u, hv, _⟩ := bind_ok h
  cases u
  exact ⟨proofs, spec, tabs, x, hf, hst, hv⟩

/-- A push that leaves the run accepting pushed an accepted verdict, naming the partition object's program, with the
units the record drew. -/
theorem push_ok {I : Flock.Witness.Inputs} {rd : Reads} {programSha : Option String} {want : Json} {r r' : Run}
    {s : Session} {record : String} {st : SetUp} {coinKey : Option String} {verdict : V Unit}
    (h : push I rd programSha want r s record st coinKey verdict = .ok r') (hall : r'.all = true) :
    r.all = true ∧ verdict = .ok () ∧ (∃ h, programSha = some h ∧ (Json.str h == want) = true) ∧
      ∃ units, provedUnits I.publicFile record = .ok units ∧ r'.proved = r.proved ++ units ∧
        r'.out.size = r.out.size + 1 := by
  unfold push at h
  cases verdict with
  | error why =>
    cases h
    simp at hall
  | ok u =>
    cases u
    dsimp only at h
    split at h
    · cases h
    · rename_i hne
      obtain ⟨units, hu, h⟩ := bind_ok h
      cases h
      refine ⟨hall, rfl, ?_, units, hu, rfl, by simp⟩
      cases programSha with
      | none => exact (hne rfl).elim
      | some p =>
        refine ⟨p, rfl, ?_⟩
        cases hb : (Json.str p == want)
        · exact (hne (show (!(Json.str p == want)) = true by simp [hb])).elim
        · rfl

/-- A step that leaves the run accepting verified its session. -/
theorem step_ok {I : Flock.Witness.Inputs} {rd : Reads} {blobs : Array ByteArray} {scope : V Unit}
    {programSha : Option String} {want : Json} {r r' : Run} {s : Session}
    (h : step I rd (Core.Blobs.ofArray blobs) scope programSha want r s = .ok r') (hall : r'.all = true) :
    r.all = true ∧ (∃ h, programSha = some h ∧ (Json.str h == want) = true) ∧
      ∃ record proofs units, SessionAccepted I s record proofs units ∧ r'.proved = r.proved ++ units ∧
        r'.out.size = r.out.size + 1 := by
  unfold step at h
  obtain ⟨b, hb, h⟩ := bind_ok h
  cases hrec : String.fromUTF8? b with
  | none => (first | rw [hrec] at h | simp only [hrec] at h) <;> cases h
  | some record =>
    first | rw [hrec] at h | simp only [hrec] at h
    try dsimp only at h
    obtain ⟨hr, hv, hp, units, hu, hpr, hsz⟩ := push_ok h hall
    obtain ⟨proofs, spec, tabs, x, _, hst, hz⟩ := verdictOf_ok hv
    obtain ⟨sts, hset, hshape, htabs⟩ := setupAt_ok hst
    subst htabs
    exact ⟨hr, hp, record, proofs, units, ⟨⟨b, (get_ofArray hb).2, hrec⟩, ⟨spec, sts, hset, hshape, hz⟩, hu⟩, hpr, hsz⟩

/-- The fold of `step` over sessions, from `r0` to an accepting `r`. -/
theorem fold_ok {I : Flock.Witness.Inputs} {rd : Reads} {blobs : Array ByteArray} {scope : V Unit}
    {programSha : Option String} {want : Json} :
    ∀ (l : List Session) (r0 r : Run),
      l.foldlM (step I rd (Core.Blobs.ofArray blobs) scope programSha want) r0 = .ok r → r.all = true →
      r0.all = true ∧ ∃ ss : List (String × Array ByteArray × Array Nat), ss.length = l.length ∧
        (∀ x ∈ l.zip ss, SessionAccepted I x.1 x.2.1 x.2.2.1 x.2.2.2) ∧
        r.proved = ss.foldl (fun acc x => acc ++ x.2.2) r0.proved ∧
        r.out.size = r0.out.size + l.length ∧
        (l ≠ [] → ∃ h, programSha = some h ∧ (Json.str h == want) = true)
  | [], r0, r, h, hall => by
    cases h
    exact ⟨hall, [], rfl, by simp, rfl, rfl, fun hl => (hl rfl).elim⟩
  | s :: l, r0, r, h, hall => by
    rw [List.foldlM_cons] at h
    obtain ⟨r1, h1, h2⟩ := bind_ok h
    obtain ⟨hr1, ss, hlen, hss, hpr, hsz, _⟩ := fold_ok l r1 r h2 hall
    obtain ⟨hr0, hp, record, proofs, units, hacc, hpr1, hsz1⟩ := step_ok h1 hr1
    refine ⟨hr0, (record, proofs, units) :: ss, by simp [hlen], ?_, ?_, ?_, fun _ => hp⟩
    · intro x hx
      simp only [List.zip_cons_cons, List.mem_cons] at hx
      rcases hx with rfl | hx
      · exact hacc
      · exact hss x hx
    · (rw [hpr, hpr1]) <;> rfl
    · (rw [hsz, hsz1, List.length_cons]) <;> omega

/-- What `run` returns, when every session accepted. -/
theorem run_ok {p : Own} {pub : String} {sessions : Array Session} {want : Array Nat} {blobs : Array ByteArray} {r : Run}
    (h : run p pub sessions (some want) (Core.Blobs.ofArray blobs) = .ok r) (hall : r.all = true) :
    ∃ (tags : Flock.Tags) (circuitB pubB : ByteArray), Flock.Tags.find? STATEMENT = some tags ∧
      Blob blobs p.circuit circuitB ∧ Blob blobs pub pubB ∧ r.out.size = sessions.size ∧
      (0 < sessions.size → ∃ h, programDigest p.program = some h ∧ (Json.str h == getD p.partition "program") = true) ∧
      ∃ ss : List (String × Array ByteArray × Array Nat), ss.length = sessions.size ∧
        (∀ x ∈ sessions.toList.zip ss,
          SessionAccepted (inputsOf tags p circuitB pubB (Core.Blobs.ofArray blobs)) x.1 x.2.1 x.2.2.1 x.2.2.2) ∧
        (ss.foldl (fun acc x => acc ++ x.2.2) #[]).qsort (· < ·) = want := by
  unfold run at h
  cases ht : Flock.Tags.find? STATEMENT with
  | none => simp only [ht] at h <;> cases h
  | some tags0 =>
    simp only [ht] at h
    obtain ⟨_, _, h⟩ := bind_ok h
    obtain ⟨_, _, h⟩ := bind_ok h
    obtain ⟨circuitB, hc, h⟩ := bind_ok h
    obtain ⟨pubB, hpb, h⟩ := bind_ok h
    obtain ⟨rd, _, h⟩ := bind_ok h
    obtain ⟨r1, hf, h⟩ := bind_ok h
    rw [← Array.foldlM_toList] at hf
    try dsimp only at h
    split at h
    · cases h
    · rename_i hne
      cases h
      obtain ⟨_, ss, hlen, hss, hpr, hsz, hprog⟩ := fold_ok _ _ _ hf hall
      have hq : r.proved.qsort (· < ·) = want := by simpa [hall] using hne
      refine ⟨tags0, circuitB, pubB, rfl, get_ofArray hc, get_ofArray hpb, by simpa using hsz, fun hpos => ?_, ss, hlen,
        hss, ?_⟩
      · exact hprog (List.ne_nil_of_length_pos (by simpa using hpos))
      · (rw [← hq, hpr]) <;> rfl

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
        obtain ⟨_, _, _, -, -, -, hva, -⟩ := Properties.SampledProofs.acceptMeans _ _ _ _ _ hchk
        obtain ⟨x, hx, -⟩ := hva.2.1
        have hx' : x ∈ r.out := hx
        have hpos : 0 < e.sessions.size := by rw [← hsize]; exact Array.size_pos_of_mem hx'
        exact ⟨tags, circuitB, pubB, ss, r.out, htags, hcirc, hpub, hpos, hlen, hss, hunits, hprog hpos, hchk⟩
      · rw [if_neg hall] at h
        cases h
    · next o hne hchk =>
      first
      | exact (hchk _ _ h).elim
      | exact (hchk _ _ (hne ▸ h)).elim
      | exact (hchk _ _ (hne.symm ▸ h)).elim
      | (subst hne; exact (hchk _ _ h).elim)

/-- **`AcceptMeans` fails for an accept-everything check**: such a check accepts evidence with no sessions, which
`Accepted` refuses. -/
theorem acceptMeans_nontrivial (p : Params) (c : OneStage.Commitment) (e : Evidence) (blobs : Array ByteArray)
    (pub : OneStage.Public) (prof : Core.Integrity) (h : e.sessions = #[]) : ¬ Accepted p c e blobs pub prof := by
  rintro ⟨_, _, _, _, _, _, _, _, hpos, _⟩
  simp [h] at hpos

/-- **`AcceptsZK` is the model's `ZkExec.AcceptsZK`** at the witness's inputs as the model reads them (`toExec`), on the
transcript of the session's record and proof files. -/
theorem acceptsZK_iff (I : Flock.Witness.Inputs) (record : String) (proofs : Array ByteArray) :
    AcceptsZK I record proofs ↔
      FlockSoundness.Discharge.ZkExec.AcceptsZK (FlockSoundness.Discharge.Recursion.Witness.toExec I) ⟨record, proofs⟩ :=
  Iff.rfl

end Properties.SampledProofs.OneStageZk
