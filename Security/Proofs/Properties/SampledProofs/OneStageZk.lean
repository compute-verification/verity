import Properties.SampledProofs.OneStageZk
import Proofs.Properties.SampledProofs
import Proofs.Flock.Recursive.Witness

/-! # Proof of the one-stage audit on the proofs' property (`Properties.SampledProofs.OneStageZk`)

`checkBlobs` accepts only when its run of the sessions (`run`) returns with every session accepted and v1's check
accepts the verdicts it computed. The run is a fold of `step` over the sessions; a step that leaves the run accepting
has its record in the blob its digest names (`get_names`), its setup `Witness.setupOf`'s with every table's shape
(`setupAt_ok`, through `buildSessionOf_ok`), `Zk.verify` accepting its proof files (`verdictOf_ok`), and the units it
proved (`push_ok`). `fold_ok` gathers the steps, `run_ok` adds the files and the proved units, and v1's
`Properties.SampledProofs.acceptMeans` gives at least one verdict, so at least one session. `acceptsZK_iff` is the
model's event.
-/

open Lean (Json)
open SampledProofs SampledProofs.OneStageZk

namespace Properties.SampledProofs.OneStageZk

open FlockSoundness.Discharge.Recursion.Witness (bind_ok)

local instance instLawfulBEqByteArray_oneStageZk : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

/-- **`AcceptsZK` is the model's event** (`ZkExec.AcceptsZK`) at the witness's inputs. -/
theorem acceptsZK_iff (I : Flock.Witness.Inputs) (record : String) (proofs : Array ByteArray) :
    AcceptsZK I record proofs ↔
      FlockSoundness.Discharge.ZkExec.AcceptsZK (FlockSoundness.Discharge.Recursion.Witness.toExec I) ⟨record, proofs⟩ :=
  Iff.rfl

theorem mapError_ok {ε ε' α : Type} {f : ε → ε'} {x : Except ε α} {a : α} (h : x.mapError f = .ok a) : x = .ok a := by
  cases x with
  | error e => cases h
  | ok b => cases h; rfl

/-- A blob found under a digest is one whose SHA-512 the digest names. -/
theorem get_names {blobs : Array ByteArray} {d what : String} {b : ByteArray}
    (h : (Core.Blobs.ofArray blobs).get d what = .ok b) : Names d b := by
  unfold Core.Blobs.get at h
  obtain ⟨d', _, h⟩ := bind_ok h
  by_cases hb : (d'.size != 64 || Flock.hexOf d' != d) = true
  · exfalso
    try dsimp only at h
    rw [if_pos hb] at h
    cases h
  · have hhex : Flock.hexOf d' = d := by
      simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hb
      exact hb.2
    try dsimp only at h
    rw [if_neg hb] at h
    try (obtain ⟨_, _, h⟩ := bind_ok h)
    rcases hf : (Core.Blobs.ofArray blobs).entries.find? (·.1 == d') with _ | ⟨x, b'⟩
    · rw [hf] at h; cases h
    · rw [hf] at h
      have hb' : b' = b := by cases h; rfl
      subst hb'
      have hp : (x == d') = true := Array.find?_some hf
      have hm := Array.mem_of_find?_eq_some hf
      simp only [Core.Blobs.ofArray, Array.mem_map, Prod.mk.injEq] at hm
      obtain ⟨c, _, hx, hcb⟩ := hm
      subst hcb
      have hxd : x = d' := eq_of_beq hp
      subst hxd
      subst hx
      exact hhex

/-- `buildSessionOf` on `ProvedScope.circuitOf`'s circuit sets up what `Stmt.setupTables` sets up. -/
theorem buildSessionOf_ok {tags : Flock.Tags} {cB pB : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {count : Nat} {draw partition : Option Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (h : buildSessionOf (Flock.ProvedScope.circuitOf tags cB tables) tags cB pB coins tables count draw partition program
      registered = .ok (spec, sts)) :
    Flock.Stmt.setupTables tags cB pB coins count tables draw partition program = .ok (spec, sts) := by
  unfold buildSessionOf at h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨⟨spec', sts'⟩, h1, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  cases h
  cases hh : tags.hiddenOutputs
  · simpa [hh] using h1
  · simpa [Flock.ProvedScope.circuitOf, hh, Flock.Stmt.setupTables_eq] using h1

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
        · exact (hne (show (!(Json.str p == want)) = true by rw [hb]; rfl)).elim
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
  | none => rw [hrec] at h; cases h
  | some record =>
    rw [hrec] at h
    dsimp only at h
    obtain ⟨hr, hv, hp, units, hu, hpr, hsz⟩ := push_ok h hall
    obtain ⟨proofs, spec, tabs, x, _, hst, hz⟩ := verdictOf_ok hv
    obtain ⟨sts, hset, hshape, htabs⟩ := setupAt_ok hst
    subst htabs
    exact ⟨hr, hp, record, proofs, units, ⟨⟨b, get_names hb, hrec⟩, ⟨spec, sts, hset, hshape, hz⟩, hu⟩, hpr, hsz⟩

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
    · rw [hpr, hpr1]; rfl
    · rw [hsz, hsz1, List.length_cons]; omega

/-- A run that returns accepting, on proved units `want`. -/
theorem run_ok {p : Own} {pub : String} {sessions : Array Session} {want : Array Nat} {blobs : Array ByteArray}
    {r : Run} (h : run p pub sessions (some want) (Core.Blobs.ofArray blobs) = .ok r) (hall : r.all = true) :
    ∃ (tags : Flock.Tags) (circuitB pubB : ByteArray) (ss : List (String × Array ByteArray × Array Nat)),
      Flock.Tags.find? STATEMENT = some tags ∧ Names p.circuit circuitB ∧ Names pub pubB ∧
      ss.length = sessions.size ∧
      (∀ x ∈ sessions.toList.zip ss,
        SessionAccepted (inputsOf tags p circuitB pubB (Core.Blobs.ofArray blobs)) x.1 x.2.1 x.2.2.1 x.2.2.2) ∧
      (ss.foldl (fun acc x => acc ++ x.2.2) #[]).qsort (· < ·) = want ∧
      r.out.size = sessions.size ∧
      (sessions.size ≠ 0 → ∃ h, programDigest p.program = some h ∧ (Json.str h == getD p.partition "program") = true) := by
  unfold run at h
  cases ht : Flock.Tags.find? STATEMENT with
  | none => simp only [ht] at h; cases h
  | some tags0 =>
    simp only [ht] at h
    obtain ⟨_, _, h⟩ := bind_ok h
    obtain ⟨_, _, h⟩ := bind_ok h
    obtain ⟨circuitB, hc, h⟩ := bind_ok h
    obtain ⟨pubB, hpb, h⟩ := bind_ok h
    obtain ⟨rd, _, h⟩ := bind_ok h
    obtain ⟨r1, hf, h⟩ := bind_ok h
    rw [← Array.foldlM_toList] at hf
    dsimp only at h
    split at h
    · cases h
    · rename_i hne
      cases h
      obtain ⟨_, ss, hlen, hss, hpr, hsz, hprog⟩ := fold_ok _ _ _ hf hall
      have hq : r1.proved.qsort (· < ·) = want := by simpa [hall] using hne
      refine ⟨tags0, circuitB, pubB, ss, rfl, get_names hc, get_names hpb, hlen, hss, ?_, ?_, ?_⟩
      · rw [← hq, hpr]
      · simpa using hsz
      · intro hn
        apply hprog
        intro hl
        apply hn
        rw [← Array.length_toList, hl]; rfl

/-- **What v2 accepts on blobs** (`AcceptMeans`). -/
theorem acceptMeans : AcceptMeans := by
  intro p c e blobs pub prof h
  unfold checkBlobs at h
  cases hr : run p.own e.pub e.sessions (some e.base.provedUnits) (Core.Blobs.ofArray blobs) with
  | error err => simp [hr] at h
  | ok r =>
    cases hv : OneStage.check p.base c { e.base with verdicts := r.out } with
    | reject why => simp [hr, hv] at h
    | undecided why => simp [hr, hv] at h
    | accept pub' prof' =>
      cases hall : r.all with
      | false => simp [hr, hv, hall] at h
      | true =>
        simp only [hr, hv, hall] at h
        simp at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨tags, circuitB, pubB, ss, ht, hcB, hpB, hlen, hss, hq, hsize, hprog⟩ := run_ok hr hall
        obtain ⟨_, _, _, _, _, _, hva, _⟩ := _root_.Properties.SampledProofs.acceptMeans _ _ _ _ _ hv
        have hpos : 0 < e.sessions.size := by
          obtain ⟨x, hx, _⟩ := hva.2.1
          have : 0 < r.out.size := Array.size_pos_of_mem hx
          omega
        exact ⟨tags, circuitB, pubB, ss, r.out, ht, hcB, hpB, hpos, hlen, hss, hq, hprog (by omega), hv⟩

end Properties.SampledProofs.OneStageZk
