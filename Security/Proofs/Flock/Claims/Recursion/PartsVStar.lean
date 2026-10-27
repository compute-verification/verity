import Proofs.Flock.Claims.Recursion.VStarLinks
import Proofs.Flock.Claims.Recursion.DeployMargin

/-!
# `readX` holds each part's circuit to V\*'s, as each part's session does

`readX` refuses a statement unless each part's circuit is V\*'s statement's of its index (`partsVStar`): its circuit
file is V\*'s byte for byte under the verifier's own bindings (`RecVStar.checkFile`), and the circuit the setup parses
off it carries V\*'s row ports and part (`checkParsed`, `checkPart`), V\*'s unit name (`RecVStar.check`) and V\*'s
META holo (`checkHolo`). The deployed verifier makes the same check in
each part's session: `RecIns.setup` runs `RecSetup.checks` (`verity/core/service/flock/Flock/RecSetup.lean`), which runs
`RecVStar.checkStmt` on every table, which runs `RecVStar.checkCircuit` (`Flock/RecVStar.lean`) on the table's circuit
and the session's `--circuit` file as `--rec-part s` under `--rec-bindings`.

* `partVStar_of_setup`: where part `s`'s session sets up at `r` and its scope check passes, its circuit passes
  `partVStar`. So `readX` refuses no statement every part's session sets up at (`readX_of_sessions`), and a statement
  it refuses is one some part's session refuses at every staging.
* `readX_partVStar`: what `readX` read of each part, for the proofs that read the parts' ports.
* `readX_partNamed`: each part's circuit's part, unit name and META holo are V\*'s statement's of its index.
-/

namespace RecDeploy

open Flock
open FlockDeploy (jsonOf)

theorem RecRaw.ins_bindings {w : RecRaw} {s : RecIns} (h : w.ins = .ok s) :
    w.recBindings.mapM (fun b => do RecVStar.ownBindingsOfJson (← jsonOf "rec-bindings" b)) = .ok s.bindings := by
  unfold RecRaw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

/-- A raw statement whose public file is swapped reads where it does: `Raw.ins` reads no public file. -/
theorem rawIns_ok_of_publicFile {w : FlockDeploy.Raw} {pf : ByteArray} {i' : FlockDeploy.Ins}
    (h : ({ w with publicFile := pf } : FlockDeploy.Raw).ins = .ok i') : ∃ i, w.ins = .ok i := by
  unfold FlockDeploy.Raw.ins at h ⊢
  simp only [bind, Except.bind, pure, Except.pure] at h ⊢
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals simp_all

/-- What `readXPre` read: the verifier's own inner shape and V*'s list at it. -/
theorem readXPre_spec {x : Claims.Msg} {r : XR} (h : readXPre x = some r) :
    (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)).toOption = some r.shape ∧
      ∃ vs, RecVStar.statements r.shape = .ok vs ∧ r.w.parts.length = vs.size := by
  unfold readXPre at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨w, -, ri, -, sh, hsh, pa, -, dims, -, vs, hvs, h⟩ := h
  split_ifs at h with hl
  · simp at h
  have h' : (⟨w, ri, pa, sh, dims⟩ : XR) = r := by simpa using h
  subst h'
  exact ⟨hsh, vs, toOption_eq_some hvs, not_not.1 (not_or.1 hl).1⟩

theorem checkHolo_of_holoOf {s : VStar.Stmt} {c : Flock.Circuit} {holo : Bool}
    (h : RecResiduals.holoOf c = .ok (RecVStar.partHolo s holo)) : RecVStar.checkHolo s c holo = .ok () := by
  unfold RecVStar.checkHolo
  rw [h]
  simp [bind, Except.bind, pure, Except.pure]

/-- **A part whose session sets up has V\*'s circuit**: where part `s`'s session at `r` (`partRaw`, at any staging)
reads its inputs, sets up and passes its scope check, the session's `RecVStar.checkCircuit` holds its circuit to
V\*'s statement `s` under the verifier's own bindings, and that is `partVStar`. -/
theorem partVStar_of_setup {r : XR} {st : Stage} {x : Claims.Msg} {ms : List (Claims.Msg × Claims.Out)} {s : ℕ}
    {p : PartX} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hsh : (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)).toOption = some r.shape)
    (hi : (partRaw r st x ms s p).ins = .ok i) (hs : i.setup = .ok (spec, sts)) (hck : i.base.read.check = .ok ()) :
    ∃ f vs sv, ownBindings p.bindings = some f ∧ RecVStar.statements r.shape = .ok vs ∧ vs[s]? = some sv ∧
      partVStar r.shape f sv p = true := by
  obtain ⟨hin, -⟩ := RecRaw.ins_inner_src hi
  have hinner : i.inner = some r.shape := by
    dsimp only [partRaw] at hin
    simp only [Option.mapM_some] at hin
    cases he : (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)) with
    | error e => rw [he] at hin; cases hin
    | ok v =>
      rw [he] at hin hsh
      simp only [Functor.map, Except.map, Except.ok.injEq] at hin
      simp only [Except.toOption, Option.some.injEq] at hsh
      rw [← hin, hsh]
  have hpart : i.part = some s := RecRaw.ins_part hi
  have hb := RecRaw.ins_bindings hi
  have hbase := RecRaw.ins_base hi
  have hpi := publicInputs_none_of_scope hck
  have hty : i.base.read.tags.typed = false := ((Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck).1.2.1
  have htab : i.base.read.tablesOf = some sts := by
    unfold Flock.Readable.Read.tablesOf
    rw [RecIns.setup_read hs]
    cases hz : i.base.read.zk
    · simp
    · simp [RecIns.setup_shapeOk hs hz]
  obtain ⟨st0, hst0, hc0, -⟩ := Flock.Readable.table0 htab hty hpi
  have hpos : 0 < sts.size := by
    by_contra h0
    rw [Array.getElem?_eq_none (by omega)] at hst0
    cases hst0
  rw [Array.getElem?_eq_getElem hpos, Option.some.injEq] at hst0
  subst hst0
  have hC := checks_stmt (RecIns.setup_checks hs) 0 hpos
  rw [hinner, hpart] at hC
  obtain ⟨j, f, sv, hj, hf, hcc⟩ := checkStmt_circuit hC
  cases hj
  obtain ⟨hck', hfile, hparsed, hpt⟩ := FlockSoundness.Discharge.Recursion.checkCircuit_ok hcc
  obtain ⟨vs, hvs, hsv, hn, -⟩ := FlockSoundness.Discharge.Recursion.check_ok hck'
  have hho : RecVStar.checkHolo sv sts[0].c r.shape.holo = .ok () :=
    checkHolo_of_holoOf (FlockSoundness.Discharge.Recursion.checkCircuit_holo hcc)
  obtain ⟨-, hcf, -⟩ := rawIns_fields hbase
  obtain ⟨ib, hib⟩ := rawIns_ok_of_publicFile (w := p.base) (pf := st.pub x ms s) hbase
  obtain ⟨htg, hcf2, htb⟩ := rawIns_publicFile hib hbase
  obtain ⟨-, hcf3, htb3, -⟩ := rawIns_fields hib
  have hpc : partCircuit p = some sts[0].c := by
    unfold partCircuit
    rw [hib]
    simp only [Except.toOption, Option.bind_eq_bind, Option.bind_some]
    rw [← htg, ← hcf3, ← htb3, ← hcf2, ← htb, hc0]
  have hown : ownBindings p.bindings = some f := by
    dsimp only [partRaw] at hb
    simp only [Option.mapM_some] at hb
    rw [hf] at hb
    unfold ownBindings
    cases he : (do RecVStar.ownBindingsOfJson (← jsonOf "rec-bindings" p.bindings)) with
    | error e => rw [he] at hb; cases hb
    | ok v =>
      rw [he] at hb
      simp only [Functor.map, Except.map, Except.ok.injEq, Option.some.injEq] at hb
      simp [Except.toOption, hb]
  refine ⟨f, vs, sv, hown, hvs, hsv, ?_⟩
  have hcb : p.base.circuit = i.base.read.I.circuitFile := hcf.symm
  unfold partVStar
  rw [hpc, hcb]
  simp [hfile, hparsed, hpt, hn, hho, Except.isOk, Except.toBool]

/-- **`readX` refuses no statement every part's session sets up at**: where `readXPre` reads `x` and each part's
session (`partRaw`, at some staging) reads its inputs, sets up and passes its scope check, `readX` reads `x` the
same. -/
theorem readX_of_sessions {x : Claims.Msg} {r : XR} (hr : readXPre x = some r)
    (h : ∀ s p, r.w.parts[s]? = some p → ∃ (st : Stage) (ms : List (Claims.Msg × Claims.Out)) (i : RecIns)
      (spec : SessionSpec) (sts : Array Stmt),
      (partRaw r st x ms s p).ins = .ok i ∧ i.setup = .ok (spec, sts) ∧ i.base.read.check = .ok ()) :
    readX x = some r := by
  obtain ⟨hsh, vs, hvs, hl⟩ := readXPre_spec hr
  refine readX_of_pre hr hvs ?_
  unfold partsVStar
  rw [List.all_eq_true]
  intro s hs
  have hs' : s < r.w.parts.length := by rw [hl]; exact List.mem_range.1 hs
  obtain ⟨st, ms, i, spec, sts, hi, hset, hck⟩ := h s r.w.parts[s] (List.getElem?_eq_getElem hs')
  obtain ⟨f, vs', sv, hown, hvs', hsv, hp⟩ := partVStar_of_setup hsh hi hset hck
  rw [hvs] at hvs'
  cases hvs'
  simp only [List.getElem?_eq_getElem hs', hsv, hown]
  exact hp

/-- **What `readX` read of each part**: the verifier's own bindings of it read, and its circuit is V\*'s statement's
of its index (`partVStar`). -/
theorem readX_partVStar {x : Claims.Msg} {r : XR} (h : readX x = some r) :
    ∃ vs, RecVStar.statements r.shape = .ok vs ∧ r.w.parts.length = vs.size ∧
      ∀ s (hs : s < vs.size), ∃ f p c, ownBindings p.bindings = some f ∧ r.w.parts[s]? = some p ∧
        partCircuit p = some c ∧ RecVStar.checkFile vs[s] f p.base.circuit r.shape.holo = .ok () ∧
        RecVStar.checkParsed vs[s] c = .ok () ∧ RecVStar.checkPart vs[s] c = .ok () := by
  obtain ⟨vs, hvs, hp⟩ := readX_parts h
  obtain ⟨-, vs', hvs', hl⟩ := readXPre_spec (readXPre_of_readX h)
  rw [hvs] at hvs'
  cases hvs'
  refine ⟨vs, hvs, hl, fun s hs => ?_⟩
  have hall := List.all_eq_true.1 hp s (List.mem_range.2 hs)
  have hs' : s < r.w.parts.length := by omega
  rw [List.getElem?_eq_getElem hs', Array.getElem?_eq_getElem hs] at hall
  simp only at hall
  cases hf : ownBindings r.w.parts[s].bindings with
  | none => rw [hf] at hall; simp at hall
  | some f =>
    rw [hf] at hall
    simp only at hall
    unfold partVStar at hall
    cases hc : partCircuit r.w.parts[s] with
    | none => rw [hc] at hall; cases hall
    | some c =>
      rw [hc] at hall
      simp only [Bool.and_eq_true] at hall
      obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, -⟩, -⟩ := hall
      refine ⟨f, r.w.parts[s], c, hf, List.getElem?_eq_getElem hs', hc, ?_, ?_, ?_⟩
      · revert h1; cases RecVStar.checkFile vs[s] f r.w.parts[s].base.circuit r.shape.holo <;>
          simp [Except.isOk, Except.toBool]
      · revert h2; cases RecVStar.checkParsed vs[s] c <;> simp [Except.isOk, Except.toBool]
      · revert h3; cases RecVStar.checkPart vs[s] c <;> simp [Except.isOk, Except.toBool]

/-- **Each part's circuit is named and flagged as V\*'s statement of its index**: besides its part (`checkPart`), the
circuit's unit name is the statement's (`RecVStar.check`) and its META holo is V\*'s file's (`checkHolo`). -/
theorem readX_partNamed {x : Claims.Msg} {r : XR} (h : readX x = some r) :
    ∃ vs, RecVStar.statements r.shape = .ok vs ∧ ∀ s (hs : s < vs.size), ∃ p c, r.w.parts[s]? = some p ∧
      partCircuit p = some c ∧ RecVStar.checkPart vs[s] c = .ok () ∧ c.unit.name = vs[s].name ∧
      RecVStar.checkHolo vs[s] c r.shape.holo = .ok () := by
  obtain ⟨vs, hvs, hp⟩ := readX_parts h
  obtain ⟨-, vs', hvs', hl⟩ := readXPre_spec (readXPre_of_readX h)
  rw [hvs] at hvs'
  cases hvs'
  refine ⟨vs, hvs, fun s hs => ?_⟩
  have hall := List.all_eq_true.1 hp s (List.mem_range.2 hs)
  have hs' : s < r.w.parts.length := by omega
  rw [List.getElem?_eq_getElem hs', Array.getElem?_eq_getElem hs] at hall
  simp only at hall
  cases hf : ownBindings r.w.parts[s].bindings with
  | none => rw [hf] at hall; simp at hall
  | some f =>
    rw [hf] at hall
    simp only at hall
    unfold partVStar at hall
    cases hc : partCircuit r.w.parts[s] with
    | none => rw [hc] at hall; cases hall
    | some c =>
      rw [hc] at hall
      simp only [Bool.and_eq_true] at hall
      obtain ⟨⟨⟨⟨-, -⟩, h3⟩, h4⟩, h5⟩ := hall
      refine ⟨r.w.parts[s], c, List.getElem?_eq_getElem hs', hc, ?_, beq_iff_eq.1 h4, ?_⟩
      · revert h3; cases RecVStar.checkPart vs[s] c <;> simp [Except.isOk, Except.toBool]
      · revert h5; cases RecVStar.checkHolo vs[s] c r.shape.holo <;> simp [Except.isOk, Except.toBool]

end RecDeploy
