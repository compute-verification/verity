import Proofs.Flock.Claims.Recursion.Deploy

/-!
# The compiled verifier's scope refusals

At `deployRec` the inner session's record and the V* sessions' records are run messages, so the recursion's shape and
draw conditions are not on the statement: the compiled verifier checks them when each V* session decides, on the
statement the prover staged. This file proves those refusals from the shipped code.

* **Shape.** A staged V* statement holds the verifier's own inner shape (`--rec-inner`) and the inner statement's files
  as `x` carries them (`partRaw`), so its recursive setup (`RecIns.setup`) runs `RecSetup.checkInner`, hence
  `RecCoins.checkAtC3`, on what `x` carries (`checkAtC3_staged`); `readX` has already run `RecVStar.statements` on that
  shape. So a session decision the compiled verifier doesn't refuse has the setup's checks accept the statement's inner
  shape (`RecScope.accepts`, on `readOf`).
* **Draw.** At a decision, a record without a draw fails the custody checks (`checks_recordDraw`: `drawEq` against the
  statement's draw, and `ProvedScope.drawLaw` refuses a draw-less statement), and the verdict's full-draw refusal
  (`fullDrawO`, `RecSetup.fullDraw`) runs under the staged statement's inner shape, which is `x`'s. So the record's
  draw passes `RecScope.fullDrawOk`, hence is `Draw.Full` (`RecScope.fullDrawOk_iff`).

`decide_scope` is the two at one session decision; `decide_inScope` puts them, with the dense form below, in
`RecScope`'s statement: `InScope` of what the verifier reads with the decided records (`RecScope.inScope_iff`).
`acc_parse` is the last step
of an accepted run: the last session's decision (`next_acc`), so `parse` reads the statement and it is in `InScope`.
`DeployScopeRuns` puts the refusals on every session of an accepted run.

* **Dense.** The verdict's last refusal (`RecSetup.denseOnly`, in `fullDrawO`) refuses a holo or sparse inner shape, so
  a decision the compiled verifier doesn't refuse is at a dense one (`decide_dense`), which `InScope` asks; every
  statement it accepts parses at a dense form (`acc_dense`: `DenseAt`, the class `RecursiveKS` is stated at).
-/

namespace RecDeploy

open Flock
open FlockDeploy (Coin Final Ins checks jsonOf)

/-! ## What a staged statement's setup reads -/

theorem jsonOf_toOption (a b : String) (q : ByteArray) : (jsonOf a q).toOption = (jsonOf b q).toOption := by
  unfold jsonOf
  cases String.fromUTF8? q with
  | none => rfl
  | some t =>
    simp only
    cases Lean.Json.parse t <;> rfl

theorem toOption_eq_some {ε α : Type} {e : Except ε α} {a : α} (h : e.toOption = some a) : e = .ok a := by
  cases e <;> simp_all [Except.toOption]

theorem decodeRec_nil : decodeRec [] = none := rfl

theorem insO_nil : insO [] = none := by
  simp [insO, decodeRec_nil]

/-- What `readX` read: the inner statement, the verifier's own inner shape, the partition, and V*'s list at the
shape. -/
theorem readX_spec {x : Claims.Msg} {r : XR} (h : readX x = some r) :
    (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)).toOption = some r.shape ∧
      (r.innerRaw.partition.mapM (jsonOf "partition")).toOption = some r.partition ∧
      (RecVStar.statements r.shape).isOk = true := by
  replace h := readXPre_of_readX h
  unfold readXPre at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨w, -, ri, -, sh, hsh, pa, hpa, dims, -, vs, hvs, h⟩ := h
  split_ifs at h with hl
  · simp at h
  have h' : (⟨w, ri, pa, sh, dims⟩ : XR) = r := by simpa using h
  subst h'
  refine ⟨hsh, hpa, ?_⟩
  cases he : RecVStar.statements sh with
  | error e => rw [he] at hvs; cases hvs
  | ok v => rfl

/-- The verifier's readers on a statement give its inner shape and check 5's source as `--rec-inner` and the inner
files read. -/
theorem RecRaw.ins_inner_src {w : RecRaw} {s : RecIns} (h : w.ins = .ok s) :
    w.recInner.mapM (fun b => do RecVStar.Inner.ofJson (← jsonOf "rec-inner" b)) = .ok s.inner ∧
      sourceOf w = .ok s.src := by
  unfold RecRaw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨by assumption, by assumption⟩)

/-- **The recursive setup runs candidate 3's checks first** (`RecSetup.checkInner`). -/
theorem RecIns.setup_checkInner {s : RecIns} {p : SessionSpec × Array Stmt} (h : s.setup = .ok p) :
    RecSetup.checkInner s.inner s.src s.bound = .ok () := by
  unfold RecIns.setup at h
  cases hc : RecSetup.checkInner s.inner s.src s.bound with
  | ok u => rfl
  | error e =>
    exfalso
    simp only [bind, Except.bind, hc] at h
    split at h <;> simp_all [throw, throwThe, MonadExceptOf.throw]

theorem checkInner_checkAtC3 {i : Option RecVStar.Inner} {src : Option RecCoins.Source}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} (h : RecSetup.checkInner i src bound = .ok ()) :
    RecCoins.checkAtC3 i src = .ok () := by
  unfold RecSetup.checkInner at h
  cases hc : RecCoins.checkAtC3 i src with
  | ok u => rfl
  | error e => simp [hc, bind, Except.bind] at h

theorem checkInner_checkShape {i : Option RecVStar.Inner} {src : Option RecCoins.Source}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} (h : RecSetup.checkInner i src bound = .ok ()) :
    RecSetup.checkShape i src = .ok () := by
  unfold RecSetup.checkInner at h
  cases hc : RecCoins.checkAtC3 i src with
  | error e => simp [hc, bind, Except.bind] at h
  | ok u =>
    cases hs : RecSetup.checkShape i src with
    | ok u' => rfl
    | error e => simp [hc, hs, bind, Except.bind] at h

/-- Part `s`'s staged statement, where it reads back, is `partRaw`'s. -/
theorem staged_decode {st : Stage} {x : Claims.Msg} {ms : List (Claims.Msg × Claims.Out)} {s : ℕ}
    (hne : staged st x ms s ≠ []) :
    ∃ r p, readX x = some r ∧ r.w.parts[s]? = some p ∧
      decodeRec (staged st x ms s) = some (partRaw r st x ms s p) := by
  rcases hr : readX x with _ | r
  · exact absurd (by simp [staged, hr]) hne
  rcases hp : r.w.parts[s]? with _ | p
  · exact absurd (by simp [staged, hr, hp]) hne
  by_cases hd : decodeRec (partRaw r st x ms s p).encode = some (partRaw r st x ms s p)
  · refine ⟨r, p, rfl, hp, ?_⟩
    simp only [staged, hr, hp, hd, ↓reduceIte]
  · exact absurd (by simp [staged, hr, hp, hd]) hne

/-- A staged statement says the inner statement is set up as `x`'s inner statement is (`innerFormOf`). -/
theorem formOf_partRaw (r : XR) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ)
    (p : PartX) : formOf (partRaw r st x ms s p) = .ok (some (innerFormOf r.innerRaw)) := rfl

/-- **A staged statement's inner shape and check 5 are `x`'s**: where it reads, the verifier's inner shape is the one
`x` carries, `RecCoins.checkAtC3` and `RecSetup.checkShape` at its source are the same checks at the files `x`
carries (`filesOf`), neither of which reads the record, and `x` carries them: a staged statement carries the inner
statement's circuit and public file, so `sourceOf` refuses one without its partition. -/
theorem insO_staged {st : Stage} {x : Claims.Msg} {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {i : RecIns}
    (h : insO (staged st x ms s) = some i) :
    ∃ r, readX x = some r ∧ i.inner = some r.shape ∧
      RecCoins.checkAtC3 i.inner i.src = RecCoins.checkAtC3 (some r.shape) (filesOf r) ∧
      RecSetup.checkShape i.inner i.src = RecSetup.checkShape (some r.shape) (filesOf r) ∧
      (filesOf r).isSome = true := by
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at h; cases h
  obtain ⟨r, p, hr, -, hd⟩ := staged_decode hne
  obtain ⟨hsh, hpa, -⟩ := readX_spec hr
  unfold insO at h
  rw [hd] at h
  simp only [Option.bind_eq_bind, Option.bind_some] at h
  split_ifs at h
  · simp at h
  have h' : (partRaw r st x ms s p).ins.toOption = some i := by simpa using h
  obtain ⟨hin, hsrc⟩ := RecRaw.ins_inner_src (toOption_eq_some h')
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
  refine ⟨r, hr, hinner, ?_⟩
  rw [hinner]
  unfold sourceOf at hsrc
  rw [formOf_partRaw] at hsrc
  dsimp only [partRaw] at hsrc
  cases hq : r.innerRaw.partition with
  | none => rw [hq] at hsrc; simp at hsrc
  | some q =>
    rw [hq] at hsrc hpa
    simp only [Option.mapM_some] at hpa
    simp only [bind, Except.bind] at hsrc
    cases hj : jsonOf "--rec-inner-partition" q with
    | error e => rw [hj] at hsrc; cases hsrc
    | ok pj =>
      rw [hj] at hsrc
      have hpj : r.partition = some pj := by
        have := jsonOf_toOption "partition" "--rec-inner-partition" q
        rw [hj] at this
        cases hj' : jsonOf "partition" q with
        | error e => rw [hj'] at this; cases this
        | ok pj' =>
          rw [hj'] at this hpa
          simp only [Except.toOption, Option.some.injEq] at this
          simp only [Functor.map, Except.map, Except.toOption, Option.some.injEq] at hpa
          rw [← hpa, this]
      cases ht : textOf "rec inner session" (st.record x ms) with
      | error e => rw [ht] at hsrc; cases hsrc
      | ok rec =>
        rw [ht] at hsrc
        simp only [pure, Except.pure, Except.ok.injEq] at hsrc
        rw [← hsrc, filesOf, hpj]
        exact ⟨rfl, rfl, rfl⟩

/-! ## The draw at a decision -/

/-- **The custody checks refuse a record without a draw**: its draw must be the statement's (`drawEq`), and a
statement without one is refused (`ProvedScope.drawLaw`). -/
theorem checks_recordDraw {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) : (Witness.recordDraw f.record).isSome = true := by
  have he : Witness.recordDraw f.record = FlockSoundness.Discharge.Exec.recordDraw f.record := rfl
  rw [he]
  unfold checks at h
  cases hr : FlockSoundness.Discharge.Exec.recordDraw f.record with
  | some d => rfl
  | none =>
    exfalso
    cases hd : s.read.draw with
    | some d =>
      simp [hr, hd, FlockDeploy.drawEq, Record.check, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
      split at h <;> simp_all
    | none =>
      simp only [hr, hd, FlockDeploy.drawEq, Option.isNone_none, ite_true, bind, Except.bind] at h
      repeat' split at h
      all_goals simp_all [ProvedScope.drawLaw]

/-! ## The refusals -/

variable {IV : Claims.Verifier Coin} {st : Stage} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}

/-- **The compiled verifier refuses a session decision out of scope.** Where session `s`'s outer verifier decides on
the final message `f`, the compiled verifier answers other than `.rej` only where the setup's checks accept the inner
shape `x` carries (`RecScope.accepts` on `readOf`: `RecVStar.statements` in `readX`, `RecCoins.checkAtC3` and
`RecSetup.checkShape` in the staged statement's setup, whose readers refuse it without the inner statement's files) and `f`'s record draws every unit of its population (`RecScope.fullDrawOk` under that shape:
`ProvedScope.drawLaw` in the custody checks, then `RecSetup.fullDraw`). -/
theorem decide_scope (H : Claims.Msg → Claims.Msg) {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg}
    {cs' : List Coin} {i : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final} (h : ract IV st x as cs = .sess s j xO as' cs')
    (ha : actO xO as' cs' = .decide i spec sts link c₁ mids f) (hn : ((next IV st x as cs).eval H).1 ≠ .rej) :
    ∃ r, readX x = some r ∧ RecScope.accepts (readOf r) = true ∧ RecScope.fullDrawOk r.shape f.record = true := by
  rw [next_sess H h] at hn
  have hacc : ((nextO xO as' cs').eval H).1 = .acc := by
    rcases nextO_decide_out H ha with ho | ho
    · exact ho
    · rw [ho] at hn; exact absurd rfl hn
  obtain ⟨⟨spec', sts', -, -, hc⟩, hf⟩ := (nextO_decide H ha).1 hacc
  obtain ⟨-, -, hxO⟩ := ract_sess h
  obtain ⟨i', hi', -⟩ := actO_decide ha
  obtain ⟨hins, hsu⟩ := setupO_some hi'
  rw [hxO] at hins
  obtain ⟨r, hr, hinner, hc3, hcs, hheld⟩ := insO_staged hins
  refine ⟨r, hr, ?_, ?_⟩
  · unfold RecScope.accepts readOf
    rw [(readX_spec hr).2.2, Bool.true_and, ← hc3,
      checkInner_checkAtC3 (RecIns.setup_checkInner hsu)]
    change ((Except.ok () : Except String Unit).isOk && (RecSetup.checkShape (some r.shape) (filesOf r)).isOk &&
      (filesOf r).isSome) = true
    rw [← hcs, checkInner_checkShape (RecIns.setup_checkInner hsu), hheld]
    rfl
  · have hio : innerOfO xO = some r.shape := by
      rw [innerOfO, hxO, hins]
      exact hinner
    replace hf := fullDrawO_fullDraw hf
    rw [hio, RecScope.fullDraw_some_rec] at hf
    unfold RecScope.fullDrawOk
    have hd := checks_recordDraw hc
    cases hrd : Witness.recordDraw f.record with
    | none => rw [hrd] at hd; cases hd
    | some d => rw [hrd] at hf; exact hf

/-! ## The dense form -/

/-- **The compiled verifier refuses a session decision at a holo or sparse inner shape**: the verdict's last refusal
(`RecSetup.denseOnly`, in `fullDrawO`) runs under the staged statement's inner shape, which is `x`'s, so a decision the
compiled verifier doesn't refuse has `x`'s inner shape neither holo nor sparse. -/
theorem decide_dense (H : Claims.Msg → Claims.Msg) {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg}
    {cs' : List Coin} {i : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final} (h : ract IV st x as cs = .sess s j xO as' cs')
    (ha : actO xO as' cs' = .decide i spec sts link c₁ mids f) (hn : ((next IV st x as cs).eval H).1 ≠ .rej) :
    ∃ r, readX x = some r ∧ r.shape.holo = false ∧ r.shape.sparse = none := by
  rw [next_sess H h] at hn
  have hacc : ((nextO xO as' cs').eval H).1 = .acc := by
    rcases nextO_decide_out H ha with ho | ho
    · exact ho
    · rw [ho] at hn; exact absurd rfl hn
  obtain ⟨-, hf⟩ := (nextO_decide H ha).1 hacc
  obtain ⟨-, -, hxO⟩ := ract_sess h
  obtain ⟨i', hi', -⟩ := actO_decide ha
  obtain ⟨hins, -⟩ := setupO_some hi'
  rw [hxO] at hins
  obtain ⟨r, hr, hinner, -⟩ := insO_staged hins
  have hio : innerOfO xO = some r.shape := by
    rw [innerOfO, hxO, hins]
    exact hinner
  have hd := fullDrawO_denseOnly hf
  rw [hio] at hd
  exact ⟨r, hr, RecSetup.denseOnly_some hd⟩

/-! ## In scope -/

/-- A session decision of `x` that the compiled verifier doesn't refuse, on the final record `rec`. -/
def Decided (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (rec : String) :
    Prop :=
  ∃ (as : List Claims.Msg) (cs : List Coin) (s j : ℕ) (xO : Claims.Msg) (as' : List Claims.Msg) (cs' : List Coin)
    (i : Ins) (spec : SessionSpec) (sts : Array Stmt) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : Final), ract IV st x as cs = .sess s j xO as' cs' ∧
      actO xO as' cs' = .decide i spec sts link c₁ mids f ∧ f.record = rec ∧ ((next IV st x as cs).eval H).1 ≠ .rej

/-- **The decided records are in scope**: for records `recs`, at least one, each the final record of a session
decision of `x` that the compiled verifier doesn't refuse, what the verifier reads of `x` with those records is in
`InScope`: the inner shape candidate 3's and dense, and every draw `Draw.Full` (`RecScope.inScope_iff`). -/
theorem decide_inScope (H : Claims.Msg → Claims.Msg) {r : XR} (hr : readX x = some r) {recs : List String}
    (hne : recs ≠ []) (hd : ∀ rec ∈ recs, Decided H IV st x rec) :
    Claims.Recursion.InScope (RecScope.toStmt { readOf r with records := recs }) := by
  have hrec : ∀ rec ∈ recs, RecScope.accepts (readOf r) = true ∧ RecScope.fullDrawOk r.shape rec = true ∧
      r.shape.holo = false ∧ r.shape.sparse = none := by
    intro rec hm
    obtain ⟨as, cs, s, j, xO, as', cs', i, spec, sts, link, c₁, mids, f, h, ha, rfl, hn⟩ := hd rec hm
    obtain ⟨r', hr', hacc, hf⟩ := decide_scope H h ha hn
    obtain ⟨r'', hr'', hdn⟩ := decide_dense H h ha hn
    rw [hr] at hr' hr''
    cases hr'
    cases hr''
    exact ⟨hacc, hf, hdn⟩
  obtain ⟨rec₀, hm₀⟩ := List.exists_mem_of_ne_nil recs hne
  exact (RecScope.inScope_iff (r := { readOf r with records := recs }) (hrec rec₀ hm₀).1).2
    ⟨List.all_eq_true.2 fun rec hm => (hrec rec hm).2.1, (RecScope.dense_iff r.shape).2 (hrec rec₀ hm₀).2.2⟩

/-- **An accepting step parses in scope**: where the compiled verifier accepts (at the last session's decision,
`next_acc`), `parse` reads `x` as what the verifier reads of it, which is in `InScope`. -/
theorem acc_parse (H : Claims.Msg → Claims.Msg) (h : ((next IV st x as cs).eval H).1 = .acc) :
    ∃ r, readX x = some r ∧ parse x = some (RecScope.toStmt (readOf r)) ∧
      Claims.Recursion.InScope (RecScope.toStmt (readOf r)) := by
  obtain ⟨s, j, xO, as', cs', hr, -, -, ho⟩ := next_acc H h
  cases ha : actO xO as' cs' with
  | rej => rw [nextO_rej H ha] at ho; cases ho
  | coin c => rw [nextO_coin H ha] at ho; cases ho
  | decide i spec sts link c₁ mids f =>
    have hn : ((next IV st x as cs).eval H).1 ≠ .rej := by rw [h]; exact fun e => by cases e
    obtain ⟨r, hrx, hacc, -⟩ := decide_scope H hr ha hn
    obtain ⟨r', hrx', hdn⟩ := decide_dense H hr ha hn
    rw [hrx] at hrx'
    cases hrx'
    refine ⟨r, hrx, ?_, (RecScope.inScope_iff hacc).2 ⟨rfl, (RecScope.dense_iff r.shape).2 hdn⟩⟩
    simp [parse, RecScope.parseOf, RecScope.parseBytes, decodeR, hrx, hacc]

/-- **An accepted statement is dense**: where the compiled verifier accepts, every parse of `x` has a dense inner form
(`Form.Dense`: not holo, no sparse phase), so `DenseAt x`, the class `RecursiveKS` is stated at, holds at every
statement the compiled verifier accepts. -/
theorem acc_dense (H : Claims.Msg → Claims.Msg) (h : ((next IV st x as cs).eval H).1 = .acc) :
    ∀ p, parse x = some p → p.form.holo = false ∧ p.form.sparse = none := by
  obtain ⟨r, -, he, hin⟩ := acc_parse H h
  intro p hp
  rw [he] at hp
  cases hp
  unfold Claims.Recursion.InScope at hin
  obtain ⟨-, -, -, -, -, ⟨hho, hsp, -⟩, -⟩ := hin
  exact ⟨hho, hsp⟩

end RecDeploy
