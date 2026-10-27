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
(`push_ok`). `runFrom_ok` adds the files, each one of the blobs, and the proved units, from any run before;
`run_ok` is it from none. v1's proved `acceptMeans` gives the verdicts' acceptance, so at least one session ran
(`acceptMeansOne`). The group form folds `runFrom` over the members (`group_fold_ok`, `runGroup_ok`), each on its own
inputs, and adds the leaf layer (`leafLayer_ok`) and the units each once (`acceptMeansGroup`). `acceptMeans_nontrivial`:
an accept-everything check would fail `AcceptMeans`. `circuitsHeld` reads R6 off either form. `acceptsZK_iff` is the
bridge to the model's `ZkExec.AcceptsZK` at `Witness.toExec I`.
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

theorem mapError_ok {ε ε' A : Type} {x : Except ε A} {f : ε → ε'} {a : A} (h : x.mapError f = .ok a) : x = .ok a := by
  cases x with
  | error e => cases h
  | ok a' => cases h; rfl

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

/-- What `runFrom` returns from the run `r0`, when every session accepted. -/
theorem runFrom_ok {r0 : Run} {p : Own} {pub : String} {sessions : Array Session} {pu : Option (Array Nat)}
    {blobs : Array ByteArray} {r : Run}
    (h : runFrom r0 p pub sessions pu (Core.Blobs.ofArray blobs) = .ok r) (hall : r.all = true) :
    ∃ (tags : Flock.Tags) (circuitB pubB : ByteArray), Flock.Tags.find? STATEMENT = some tags ∧
      Blob blobs p.circuit circuitB ∧ Blob blobs pub pubB ∧ r0.all = true ∧
      r.out.size = r0.out.size + sessions.size ∧
      (0 < sessions.size → ∃ h, programDigest p.program = some h ∧ (Json.str h == getD p.partition "program") = true) ∧
      ∃ ss : List (String × Array ByteArray × Array Nat), ss.length = sessions.size ∧
        (∀ x ∈ sessions.toList.zip ss,
          SessionAccepted (inputsOf tags p circuitB pubB (Core.Blobs.ofArray blobs)) x.1 x.2.1 x.2.2.1 x.2.2.2) ∧
        r.proved = ss.foldl (fun acc x => acc ++ x.2.2) r0.proved ∧
        ∀ want, pu = some want → r.proved.qsort (· < ·) = want := by
  unfold runFrom at h
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
    have key : r1 = r ∧ ∀ want, pu = some want → r1.all = true → r1.proved.qsort (· < ·) = want := by
      cases pu with
      | none =>
        try dsimp only at h
        cases h
        exact ⟨rfl, fun _ hw => by cases hw⟩
      | some want =>
        try dsimp only at h
        split at h
        · cases h
        · rename_i hne
          cases h
          refine ⟨rfl, fun w hw hall' => ?_⟩
          cases hw
          simpa [hall'] using hne
    obtain ⟨rfl, hq⟩ := key
    obtain ⟨hr0, ss, hlen, hss, hpr, hsz, hprog⟩ := fold_ok _ _ _ hf hall
    exact ⟨tags0, circuitB, pubB, rfl, get_ofArray hc, get_ofArray hpb, hr0, by simpa using hsz,
      fun hpos => hprog (List.ne_nil_of_length_pos (by simpa using hpos)), ss, hlen, hss, hpr,
      fun want hw => hq want hw hall⟩

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
  have h' : runFrom ⟨#[], #[], #[], true⟩ p pub sessions (some want) (Core.Blobs.ofArray blobs) = .ok r := h
  obtain ⟨tags, circuitB, pubB, ht, hc, hp, -, hsz, hprog, ss, hlen, hss, hpr, hq⟩ := runFrom_ok h' hall
  refine ⟨tags, circuitB, pubB, ht, hc, hp, by simpa using hsz, hprog, ss, hlen, hss, ?_⟩
  (rw [← hq want rfl, hpr]) <;> rfl

/-- A member's step that returned a run ran its sessions from the run before. -/
theorem memberStep_ok {bs : Core.Blobs} {r r' : Run} {y : (String × Own) × Member} (h : memberStep bs r y = .ok r') :
    runFrom r y.1.2 y.2.pub y.2.sessions none bs = .ok r' := by
  unfold memberStep at h
  exact mapError_ok h

/-- The fold of `memberStep` over the members, from `r0` to an accepting `r`: each member accepted on its own inputs,
the units its sessions proved appended member by member, one verdict a session. -/
theorem group_fold_ok {blobs : Array ByteArray} :
    ∀ (l : List ((String × Own) × Member)) (r0 r : Run),
      l.foldlM (memberStep (Core.Blobs.ofArray blobs)) r0 = .ok r → r.all = true →
      r0.all = true ∧ (l ≠ [] → ∃ tags, Flock.Tags.find? STATEMENT = some tags) ∧
      ∃ sss : List (List (String × Array ByteArray × Array Nat)), sss.length = l.length ∧
        (∀ y ∈ l.zip sss, ∀ tags, Flock.Tags.find? STATEMENT = some tags →
          MemberAccepted tags y.1.1.2 y.1.2 blobs y.2) ∧
        r.proved = sss.foldl (fun acc ss => ss.foldl (fun acc x => acc ++ x.2.2) acc) r0.proved ∧
        r.out.size = r0.out.size + (l.map (·.2.sessions.size)).sum
  | [], r0, r, h, hall => by
    cases h
    exact ⟨hall, fun hl => (hl rfl).elim, [], rfl, by simp, rfl, by simp⟩
  | y :: l, r0, r, h, hall => by
    rw [List.foldlM_cons] at h
    obtain ⟨r1, h1, h2⟩ := bind_ok h
    obtain ⟨hr1, -, sss, hlen, hsss, hpr, hsz⟩ := group_fold_ok l r1 r h2 hall
    obtain ⟨tags', circuitB, pubB, ht', hc, hp, hr0, hsz1, hprog, ss, hlen1, hss, hpr1, -⟩ :=
      runFrom_ok (memberStep_ok h1) hr1
    refine ⟨hr0, fun _ => ⟨tags', ht'⟩, ss :: sss, by simp [hlen], ?_, ?_, ?_⟩
    · intro z hz tags ht
      obtain rfl : tags' = tags := Option.some.inj (ht'.symm.trans ht)
      simp only [List.zip_cons_cons, List.mem_cons] at hz
      rcases hz with rfl | hz
      · exact ⟨circuitB, pubB, hc, hp, hlen1, hss, hprog⟩
      · exact hsss z hz tags' ht
    · (rw [hpr, hpr1]) <;> rfl
    · rw [hsz, hsz1]
      simp only [List.map_cons, List.sum_cons]
      omega

/-- Mapping a function of the right component over a zip of lists of one length maps it over the right list. -/
theorem map_zip_snd {α β γ : Type} (f : β → γ) :
    ∀ (l1 : List α) (l2 : List β), l1.length = l2.length → (l1.zip l2).map (fun y => f y.2) = l2.map f
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | _ :: l1, _ :: l2, h => by
    simp only [List.zip_cons_cons, List.map_cons, map_zip_snd f l1 l2 (by simpa using h)]

/-- Each element of a list is paired in its zip with a list at least as long. -/
theorem exists_mem_zip {α β : Type} :
    ∀ {l1 : List α} {l2 : List β} {a : α}, l1.length ≤ l2.length → a ∈ l1 → ∃ b, (a, b) ∈ l1.zip l2
  | [], _, _, _, ha => by cases ha
  | _ :: _, [], _, hl, _ => by simp at hl
  | a' :: l1, b :: l2, a, hl, ha => by
    rcases List.mem_cons.1 ha with rfl | ha
    · exact ⟨b, by simp⟩
    · obtain ⟨b', hb⟩ := exists_mem_zip (l2 := l2) (by simpa using hl) ha
      exact ⟨b', by rw [List.zip_cons_cons]; exact List.mem_cons_of_mem _ hb⟩

/-- What `runGroup` returns, when every session accepted: the evidence's members are the parameters', each accepted
on its own inputs. -/
theorem runGroup_ok {ms : Array (String × Own)} {xs : Array Member} {blobs : Array ByteArray} {r : Run}
    (h : runGroup ms xs (Core.Blobs.ofArray blobs) = .ok r) (hall : r.all = true) :
    xs.map (·.name) = ms.map (·.1) ∧ (0 < ms.size → ∃ tags, Flock.Tags.find? STATEMENT = some tags) ∧
      ∃ sss : List (List (String × Array ByteArray × Array Nat)), sss.length = ms.size ∧
        (∀ y ∈ (ms.toList.zip xs.toList).zip sss, ∀ tags, Flock.Tags.find? STATEMENT = some tags →
          MemberAccepted tags y.1.1.2 y.1.2 blobs y.2) ∧
        r.proved = sss.foldl (fun acc ss => ss.foldl (fun acc x => acc ++ x.2.2) acc) #[] ∧
        r.out.size = (xs.toList.map (·.sessions.size)).sum := by
  unfold runGroup at h
  obtain ⟨u, hu, h⟩ := bind_ok h
  have hn : xs.map (·.name) = ms.map (·.1) := by
    split at hu
    · cases hu
    · rename_i hne
      exact Classical.byContradiction fun hc => hne (bne_iff_ne.mpr hc)
  have hsize : xs.size = ms.size := by simpa using congrArg Array.size hn
  rw [← Array.foldlM_toList, Array.toList_zip] at h
  obtain ⟨-, htags, sss, hlen, hsss, hpr, hsz⟩ := group_fold_ok _ _ _ h hall
  have hzip : (ms.toList.zip xs.toList).length = ms.size := by simp [hsize]
  have hmap : (ms.toList.zip xs.toList).map (·.2.sessions.size) = xs.toList.map (·.sessions.size) :=
    map_zip_snd (fun x : Member => x.sessions.size) _ _ (by simp [hsize])
  refine ⟨hn, fun hpos => htags ?_, sss, hlen.trans hzip, hsss, hpr, ?_⟩
  · intro h0
    have h1 : ms.size = 0 := by
      have := congrArg List.length h0
      rw [hzip] at this
      simpa using this
    omega
  · rw [hsz, hmap]
    simp

/-- The members' leaf layer the registration names is `LeafLayer`. -/
theorem leafLayer_ok {c : OneStage.Commitment} {xs : Array Member} {l : String} (h : leafLayer xs = .ok l)
    (hl : ¬ (getD c.record "leaf_layer" != .str l) = true) : LeafLayer c xs := by
  unfold leafLayer at h
  obtain ⟨t, ht, h⟩ := bind_ok h
  have hl' : (getD c.record "leaf_layer" == .str l) = true := by
    cases hb : (getD c.record "leaf_layer" == .str l)
    · exact absurd (by simp [bne, hb]) hl
    · rfl
  exact ⟨t, l, ht, hl', Except.ok.inj h⟩

/-- What the single form accepts is `AcceptedOne`. -/
theorem acceptMeansOne (p : Params) (c : OneStage.Commitment) (e : Evidence) (blobs : Array ByteArray)
    (pub : OneStage.Public) (prof : Core.Integrity)
    (h : checkOne p c e (Core.Blobs.ofArray blobs) = .accept pub prof) : AcceptedOne p c e blobs pub prof := by
  unfold checkOne at h
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

/-- What the group form accepts is `AcceptedGroup`. -/
theorem acceptMeansGroup (p : Params) (c : OneStage.Commitment) (e : Evidence) (blobs : Array ByteArray)
    (pub : OneStage.Public) (prof : Core.Integrity)
    (h : checkGroup p c e (Core.Blobs.ofArray blobs) = .accept pub prof) : AcceptedGroup p c e blobs pub prof := by
  unfold checkGroup at h
  split at h
  · cases h
  · next layer hlayer =>
    split at h
    · cases h
    · next hl =>
      split at h
      · cases h
      · next r hrun =>
        split at h
        · cases h
        · next hunits =>
          split at h
          · next pub' prof' hchk =>
            by_cases hall : r.all = true
            · rw [if_pos hall] at h
              cases h
              have hq : r.proved.qsort (· < ·) = e.base.provedUnits ∧ increasing e.base.provedUnits.toList = true := by
                first
                | simpa [hall] using hunits
                | (simp only [hall, Bool.true_and, Bool.or_eq_true, bne_iff_ne, ne_eq, Bool.not_eq_true',
                    not_or, Decidable.not_not, Bool.not_eq_false] at hunits; exact hunits)
              obtain ⟨hn, htags, sss, hlen, hsss, hpr, hsz⟩ := runGroup_ok hrun hall
              obtain ⟨_, _, _, -, -, -, hva, -⟩ := Properties.SampledProofs.acceptMeans _ _ _ _ _ hchk
              obtain ⟨x, hx, -⟩ := hva.2.1
              have hx' : x ∈ r.out := hx
              have hpos : 0 < (e.members.toList.map (·.sessions.size)).sum := by
                rw [← hsz]; exact Array.size_pos_of_mem hx'
              have hsize : e.members.size = p.members.size := by simpa using congrArg Array.size hn
              have hms : 0 < p.members.size := by
                rw [← hsize]
                refine Nat.pos_of_ne_zero fun h0 => ?_
                rw [Array.eq_empty_of_size_eq_zero h0] at hpos
                simp at hpos
              obtain ⟨tags, ht⟩ := htags hms
              refine ⟨tags, sss, r.out, ht, hms, hn, leafLayer_ok hlayer hl, hlen, fun y hy => hsss y hy tags ht, hpos,
                ?_, hq.2, hchk⟩
              rw [← hpr]
              exact hq.1
            · rw [if_neg hall] at h
              cases h
          · next o hne hchk =>
            first
            | exact (hchk _ _ h).elim
            | exact (hchk _ _ (hne ▸ h)).elim
            | exact (hchk _ _ (hne.symm ▸ h)).elim
            | (subst hne; exact (hchk _ _ h).elim)

/-- **What v2 accepts is `Accepted`**: the single form's acceptance with no member on either side, else the group
form's. -/
theorem acceptMeans : AcceptMeans := by
  intro p c e blobs pub prof h
  unfold checkBlobs at h
  split at h
  · next hone =>
    have hone' : p.members = #[] ∧ e.members = #[] := by
      first
      | simpa using hone
      | (simp only [Bool.and_eq_true, Array.isEmpty_iff] at hone; exact hone)
    exact .inl ⟨hone'.1, hone'.2, acceptMeansOne p c e blobs pub prof h⟩
  · exact .inr (acceptMeansGroup p c e blobs pub prof h)

/-- **`AcceptMeans` fails for an accept-everything check**: such a check accepts parameters with no member and
evidence with no sessions, which `Accepted` refuses. -/
theorem acceptMeans_nontrivial (p : Params) (c : OneStage.Commitment) (e : Evidence) (blobs : Array ByteArray)
    (pub : OneStage.Public) (prof : Core.Integrity) (hp : p.members = #[]) (h : e.sessions = #[]) :
    ¬ Accepted p c e blobs pub prof := by
  rintro (⟨-, -, _, _, _, _, _, _, _, _, hpos, _⟩ | ⟨_, _, _, _, hm, _⟩)
  · simp [h] at hpos
  · simp [hp] at hm

/-- **What v2 accepted holds its committed circuit files**: the single form's circuit, and each member's, is one of
the blobs under its committed digest (R6). -/
theorem circuitsHeld {p : Params} {c : OneStage.Commitment} {e : Evidence} {blobs : Array ByteArray}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : Accepted p c e blobs pub prof) : CircuitsHeld p blobs := by
  rcases h with ⟨hp, -, _, circuitB, _, _, _, _, hcirc, _⟩ | ⟨_, sss, _, _, hms, hn, _, hlen, hmem, _⟩
  · refine ⟨fun _ => ⟨circuitB, hcirc⟩, fun m hm => ?_⟩
    rw [hp] at hm
    simp at hm
  · refine ⟨fun h0 => by simp [h0] at hms, fun m hm => ?_⟩
    have hsize : e.members.size = p.members.size := by simpa using congrArg Array.size hn
    obtain ⟨x, hx⟩ := exists_mem_zip (l2 := e.members.toList) (by simp [hsize]) (Array.mem_toList_iff.2 hm)
    obtain ⟨ss, hy⟩ := exists_mem_zip (l2 := sss) (by simp [hsize, hlen]) hx
    obtain ⟨circuitB, _, hc, _⟩ := hmem _ hy
    exact ⟨circuitB, hc⟩

/-- **`AcceptsZK` is the model's `ZkExec.AcceptsZK`** at the witness's inputs as the model reads them (`toExec`), on the
transcript of the session's record and proof files. -/
theorem acceptsZK_iff (I : Flock.Witness.Inputs) (record : String) (proofs : Array ByteArray) :
    AcceptsZK I record proofs ↔
      FlockSoundness.Discharge.ZkExec.AcceptsZK (FlockSoundness.Discharge.Recursion.Witness.toExec I) ⟨record, proofs⟩ :=
  Iff.rfl

end Properties.SampledProofs.OneStageZk
