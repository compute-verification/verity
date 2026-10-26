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

/-- One session's step, when the run it starts goes on to accept every session: the run so far accepted every one,
this session's record and proof files are accepted by flock-verify `--zk` at its record's own draw, its program is the
partition object's, it pushes one verdict, and it adds the units it proved. -/
theorem step_ok {I : Flock.Witness.Inputs} {rd : Reads} {bs : Core.Blobs} {scope : Flock.V Unit}
    {programSha : Option String} {want : Json} {r r' : Run} {s : Session}
    (hbs : ∀ d w b, bs.get d w = .ok b → Names d b)
    (h : step I rd bs scope programSha want r s = .ok r') (hall : r'.all = true) :
    r.all = true ∧ (∃ h, programSha = some h ∧ (Json.str h == want) = true) ∧ r'.out.size = r.out.size + 1 ∧
      ∃ record proofs units, SessionAccepted I s record proofs units ∧ r'.proved = r.proved ++ units := by
  unfold step at h
  obtain ⟨b, hb, h⟩ := bind_ok h
  split at h
  · next record hrec =>
    trace_state
    sorry
  · cases h

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
  split at h
  · next tags0 htags =>
    trace_state
    sorry
  · cases h

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
