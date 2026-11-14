module

public import Proofs.Flock.SpotFlock.HeldOfCheck
public import Proofs.Flock.SessionLoop.Concrete
public import VStarRecursive.SampledTwoDesc

/-!
# The form's descriptor check gives the held copies (`heldOkDesc_of_descOk`)

The two-stage form's descriptor check (`VStar.Recursive.SampledTwoDesc.descOk`) is the verifier's own code, so it can't
read `heldOkDesc`, which is stated over held copies. It decides the same things on what the form holds, once per
template. Where it passes, `heldOkDesc` holds at the form's params, blobs and the run of record's input.

* `refOf_ok`: a reference statement the check reads, field by field.
* `setupOf_rawOf`: its bytes (`rawOf`) set up under `--zk` as the check set it up, with Flock's parsers' reading
  (`insOfRef`), where each field fits its length word (`fits_of`).
* `refStmtsOf_of_refsOf`: the form's reference statements are those bytes.
* `decided_of`: `decided` from its clauses, each of which the check gives at the copies it builds.
* `heldOkDesc_of_descOk`: the check gives `heldOkDesc`.
* `heldOf_of_descOk`: so where the check passes, `heldOf` builds a held copy at the reference statements the blobs
  give.
-/

@[expose] public section

namespace SpotFlock.HeldOf

open Claims Claims.Sampling SamplingDeploy Properties.Spot SpotFlock.RunsAs
open Lean (Json)
open VStar.Recursive

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

theorem ok_map {ε α β : Type} (a : α) (f : α → β) : (f <$> (Except.ok a : Except ε α)) = .ok (f a) := rfl

theorem error_bind {ε α β : Type} (e : ε) (f : α → Except ε β) : (Except.error e >>= f) = .error e := rfl

/-! ## A reference statement, read -/

/-- **A reference statement's bytes**, from what the check read of it. -/
def rawOf (R : SampledTwoDesc.Ref) : FlockDeploy.Raw :=
  { statement := Spot.OneStageZk.STATEMENT, circuit := R.circuit, publicFile := R.pub, tables := R.tables, zk := true,
    osCoins := false, count := 1, draw := none, partition := some R.partitionB, program := some R.program,
    work := none, workTable := none, stratified := none, closure := none, registered := R.registeredB,
    publicInputs := none }

/-- **What Flock's parsers read of those bytes.** -/
def insOfRef (R : SampledTwoDesc.Ref) : FlockDeploy.Ins :=
  { stmtTags := R.tags, registered := R.registered, read :=
    { I := { tags := R.tags, circuitFile := R.circuit, publicFile := R.pub,
             coins := .tree (_root_.Flock.Zk.coinSpec (R.tags.tableNames 1)), count := 1, tables := R.tables,
             partition := some R.partition, program := some R.program, work := none, stratified := none,
             closure := none }
      zk := true, draw := none, publicInputs := none, registered := R.registered.isSome } }

theorem memberOf_eq (p : SampledTwo.Params) (t : ℕ) : SampledTwoDesc.memberOf p t = memberOf p t := rfl

theorem jsonOf_eq : SampledTwoDesc.jsonOf = FlockDeploy.jsonOf := rfl

/-- **A reference statement the check reads**, field by field. -/
theorem refOf_ok {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) :
    ∃ m, memberOf p t = some m ∧ bs.get m.own.circuit "circuit" = .ok R.circuit ∧
      bs.get m.own.pub "public file" = .ok R.pub ∧
      _root_.Flock.Tags.byName? Spot.OneStageZk.STATEMENT = some R.tags ∧ R.tags.plainLeaves = false ∧
      R.partitionB = (_root_.Flock.RecordText.printT m.own.partition).toUTF8 ∧
      SampledTwoDesc.jsonOf "partition" R.partitionB = .ok R.partition ∧
      R.registeredB = m.own.registered.map (fun j => (_root_.Flock.RecordText.printT j).toUTF8) ∧
      R.registeredB.mapM (fun b => do _root_.Flock.Registered.ofJson (← SampledTwoDesc.jsonOf "registered reads" b)) =
        .ok R.registered ∧
      R.tables = Spot.OneStageZk.tablesOf bs ∧
      R.tz = _root_.Flock.Zk.tags R.tags (_root_.Flock.Zk.maskLogOf R.circuit) 1 ∧ R.program = m.own.program ∧
      SampledTwoDesc.setupAt R.tz R.tags R.circuit R.pub (.tree (_root_.Flock.Zk.coinSpec (R.tags.tableNames 1)))
        R.tables (some R.partition) (some R.program) R.registered = .ok (R.spec, R.sts) := by
  unfold SampledTwoDesc.refOf at h
  rw [memberOf_eq] at h
  rcases hm : memberOf p t with _ | m
  · simp only [hm] at h; cases h
  simp only [hm] at h
  rcases hc : bs.get m.own.circuit "circuit" with e | circuit
  · simp only [hc, error_bind] at h; cases h
  simp only [hc, ok_bind] at h
  rcases hp : bs.get m.own.pub "public file" with e | pub
  · simp only [hp, error_bind] at h; cases h
  simp only [hp, ok_bind] at h
  rcases ht : _root_.Flock.Tags.byName? Spot.OneStageZk.STATEMENT with _ | tags
  · simp only [ht] at h; cases h
  simp only [ht] at h
  cases hpl : tags.plainLeaves
  · simp only [hpl, Bool.false_eq_true, ↓reduceIte] at h
    rcases hj : SampledTwoDesc.jsonOf "partition" (_root_.Flock.RecordText.printT m.own.partition).toUTF8 with
      e | partition
    · simp only [hj, error_bind] at h; cases h
    simp only [hj, ok_bind] at h
    rcases hr : (m.own.registered.map fun j => (_root_.Flock.RecordText.printT j).toUTF8).mapM
        (fun b => do _root_.Flock.Registered.ofJson (← SampledTwoDesc.jsonOf "registered reads" b)) with
      e | registered
    · simp only [hr, error_bind] at h; cases h
    simp only [hr, ok_bind] at h
    rcases hs : SampledTwoDesc.setupAt (_root_.Flock.Zk.tags tags (_root_.Flock.Zk.maskLogOf circuit) 1) tags circuit
        pub (.tree (_root_.Flock.Zk.coinSpec (tags.tableNames 1))) (Spot.OneStageZk.tablesOf bs) (some partition)
        (some m.own.program) registered with e | ⟨spec, sts⟩
    · simp only [hs, error_bind] at h; cases h
    simp only [hs, ok_bind] at h
    cases h
    exact ⟨m, rfl, hc, hp, rfl, hpl, rfl, hj, rfl, hr, rfl, rfl, rfl, hs⟩
  · simp only [hpl] at h; cases h

/-- **Each field of a reference statement the check passes fits its length word.** -/
theorem fits_of {R : SampledTwoDesc.Ref} (hf : SampledTwoDesc.fits R = true) : (rawOf R).Fits := by
  unfold SampledTwoDesc.fits at hf
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hf
  obtain ⟨⟨⟨⟨⟨⟨⟨hs, hc⟩, hp⟩, hl⟩, hi⟩, hpa⟩, hpr⟩, hre⟩ := hf
  refine ⟨hs, hc, hp, hl, fun q hq => hi q hq,
    (by show 1 < 2 ^ 64; decide), ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro b hb; cases hb
  · intro b hb; cases hb; exact hpa
  · intro b hb; cases hb; exact hpr
  · intro n hn; cases hn
  · intro b hb; cases hb
  · intro n hn; cases hn
  · intro b hb; cases hb
  · intro b hb
    have hb' : R.registeredB = some b := hb
    rw [hb'] at hre
    simpa using hre
  · intro b hb; cases hb

/-- **Flock's parsers read the check's reading of a reference statement.** -/
theorem ins_rawOf {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) : (rawOf R).ins = .ok (insOfRef R) := by
  obtain ⟨m, -, -, -, ht, hpl, -, hj, -, hr, -, -, -, -⟩ := refOf_ok h
  rw [jsonOf_eq] at hj hr
  unfold FlockDeploy.Raw.ins
  simp only [rawOf, ht, hpl, Bool.and_false, Bool.false_eq_true, ↓reduceIte, pure_bind, Option.mapM_none,
    Option.mapM_some, hj, ok_map, ok_bind]
  rw [hr]
  rfl

/-- **The setup Flock's verifier runs is the check's.** -/
theorem setup_insOfRef {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) : (insOfRef R).setup = .ok (R.spec, R.sts) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, htz, -, hs⟩ := refOf_ok h
  rw [← hs, htz]
  rfl

theorem insOf_rawOf {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) (hf : SampledTwoDesc.fits R = true) :
    FlockDeploy.insOf true (rawOf R).encode = some (insOfRef R) := by
  unfold FlockDeploy.insOf
  rw [FlockDeploy.decode_encode_of_fits (fits_of hf)]
  simp only [Option.bind_eq_bind, Option.bind_some]
  rw [ins_rawOf h]
  rfl

/-- **A reference statement's bytes set up under `--zk`** as the check set it up. -/
theorem setupOf_rawOf {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) (hf : SampledTwoDesc.fits R = true) :
    FlockDeploy.setupOf true (rawOf R).encode = some (insOfRef R, R.spec, R.sts) := by
  unfold FlockDeploy.setupOf
  rw [insOf_rawOf h hf]
  simp only [Option.bind_eq_bind, Option.bind_some, setup_insOfRef h, Except.toOption]
  rfl

theorem readOf_rawOf {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) (hf : SampledTwoDesc.fits R = true) :
    FlockDeploy.readOf true (rawOf R).encode = some (insOfRef R).read := by
  unfold FlockDeploy.readOf
  rw [insOf_rawOf h hf]
  rfl

/-- **The read's tags are `--zk`'s**, the check's. -/
theorem tags_insOfRef {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) : (insOfRef R).read.tags = R.tz := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, htz, -, -⟩ := refOf_ok h
  rw [htz]
  rfl

theorem stmt_insOfRef {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) : (insOfRef R).read.stmt = R.stmt := by
  unfold _root_.Flock.Readable.Read.stmt _root_.Flock.Readable.Read.circ
  rw [tags_insOfRef h]
  rfl

theorem stmtOf_rawOf {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) (hf : SampledTwoDesc.fits R = true) :
    stmtOf (rawOf R).encode = R.stmt := by
  unfold stmtOf
  rw [readOf_rawOf h hf]
  exact stmt_insOfRef h

theorem shapeOf_rawOf {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) (hf : SampledTwoDesc.fits R = true) :
    shapeOf (rawOf R).encode = R.sts.toList.map (·.m) := by
  unfold shapeOf
  rw [setupOf_rawOf h hf]

theorem zkOk_rawOf {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) (hf : SampledTwoDesc.fits R = true) :
    zkOk (rawOf R).encode = SampledTwoDesc.zkOk R.spec R.sts := by
  unfold zkOk
  rw [setupOf_rawOf h hf]
  rfl

/-- **A reference statement the check reads reads as a member's** where the check says so. -/
theorem templateOk_of {h : Held} {t : ℕ} {p : SampledTwo.Params} {bs : Core.Blobs} {j : ℕ} {R : SampledTwoDesc.Ref}
    (hR : SampledTwoDesc.refOf p bs j = .ok R) (hf : SampledTwoDesc.fits R = true)
    (hx : h.xS₀ t = (rawOf R).encode) (hT : SampledTwoDesc.templateOk R = true) : templateOk h t = true := by
  unfold templateOk
  rw [hx, readOf_rawOf hR hf, shapeOf_rawOf hR hf]
  simp only [tags_insOfRef hR, stmt_insOfRef hR]
  exact hT

/-! ## The reference statements -/

theorem mapM_of_mapM {α β γ : Type} {f : α → Option β} {g : α → Option γ} {k : β → γ}
    (hfg : ∀ a b, f a = some b → g a = some (k b)) :
    ∀ (l : List α) {bs : List β}, l.mapM f = some bs → l.mapM g = some (bs.map k) := by
  intro l
  induction l with
  | nil =>
    intro bs h
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
    subst h
    rfl
  | cons a l ih =>
    intro bs h
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨b, hb, bs', hbs, rfl⟩ := h
    simp [List.mapM_cons, hfg a b hb, ih hbs]

theorem refRawOf_of_refOf {p : SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {R : SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refOf p bs t = .ok R) : refRawOf p bs t = some (rawOf R) := by
  obtain ⟨m, hm, hc, hp, -, -, hpb, -, hrb, -, htb, -, hpr, -⟩ := refOf_ok h
  unfold refRawOf
  simp only [hm, hc, hp, Option.bind_eq_bind, Option.bind_some, Except.toOption, Option.pure_def]
  unfold rawOf memberRaw
  rw [hpb, hrb, htb, hpr]

/-- **The form's reference statements are the bytes of the ones the check reads.** -/
theorem refStmtsOf_of_refsOf {p : SampledTwo.Params} {bs : Core.Blobs} {rs : List SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refsOf p bs = some rs) : refStmtsOf p bs = some (rs.map fun R => (rawOf R).encode) := by
  unfold refStmtsOf
  unfold SampledTwoDesc.refsOf at h
  refine mapM_of_mapM (k := fun R => (rawOf R).encode) (fun t R hR => ?_) _ h
  rw [refRawOf_of_refOf (ok_of_toOption hR)]
  rfl

theorem length_refsOf {p : SampledTwo.Params} {bs : Core.Blobs} {rs : List SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refsOf p bs = some rs) : rs.length = p.circuits.size := by
  unfold SampledTwoDesc.refsOf at h
  rw [length_of_mapM _ h, List.length_range]

theorem refOf_of_refsOf {p : SampledTwo.Params} {bs : Core.Blobs} {rs : List SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refsOf p bs = some rs) {t : ℕ} (ht : t < p.circuits.size) :
    ∃ R, SampledTwoDesc.refOf p bs t = .ok R ∧ rs[t]? = some R := by
  have h' := h
  unfold SampledTwoDesc.refsOf at h'
  obtain ⟨R, hR, hx⟩ := getElem?_of_mapM _ h' (t := t) (by simpa using ht)
  simp only [List.getElem_range] at hR
  exact ⟨R, ok_of_toOption hR, hx⟩

/-- **Every reference statement the check holds is one it read.** -/
theorem refOf_of_mem {p : SampledTwo.Params} {bs : Core.Blobs} {rs : List SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refsOf p bs = some rs) {R : SampledTwoDesc.Ref} (hR : R ∈ rs) :
    ∃ j, SampledTwoDesc.refOf p bs j = .ok R := by
  obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hR
  obtain ⟨R', hR', hx⟩ := refOf_of_refsOf h (t := j) (by rw [← length_refsOf h]; exact hj)
  rw [List.getElem?_eq_getElem hj, Option.some.injEq] at hx
  rw [hx]
  exact ⟨j, hR'⟩

theorem circAt_of {rs : List SampledTwoDesc.Ref} {t : ℕ} {R : SampledTwoDesc.Ref} (h : rs[t]? = some R) :
    SampledTwoDesc.circAt (SampledTwoDesc.circsOf rs) t = R.stmt := by
  unfold SampledTwoDesc.circAt SampledTwoDesc.circsOf
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, h]
  rfl

/-- **The templates' circuits are the check's**: template `t`'s reference statement's circuit is the check's
(`circAt` of `circsOf`), at every `t`. -/
theorem stmtOf_refStmt {p : SampledTwo.Params} {bs : Core.Blobs} {rs : List SampledTwoDesc.Ref}
    (h : SampledTwoDesc.refsOf p bs = some rs) (hf : rs.all SampledTwoDesc.fits = true) (hne : rs ≠ [])
    (t : ℕ) : stmtOf (refStmt (rs.map fun R => (rawOf R).encode) t) =
      SampledTwoDesc.circAt (SampledTwoDesc.circsOf rs) t := by
  have hall : ∀ R ∈ rs, stmtOf (rawOf R).encode = R.stmt := by
    intro R hR
    obtain ⟨j, hj⟩ := refOf_of_mem h hR
    exact stmtOf_rawOf hj (List.all_eq_true.1 hf R hR)
  unfold refStmt SampledTwoDesc.circAt SampledTwoDesc.circsOf
  cases rs with
  | nil => exact absurd rfl hne
  | cons R₀ rs' =>
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_map]
    cases hj : (R₀ :: rs')[t]? with
    | none => exact hall R₀ (by simp)
    | some R => exact hall R (List.mem_of_getElem? hj)

/-! ## The first stage -/

theorem firstOf_of {i : Spot.Run.Input} {f : SampledTwoDesc.First} (h : SampledTwoDesc.firstOf i = some f) :
    firstOf i = some ⟨f.verifier, f.rule, f.closure⟩ := by
  unfold SampledTwoDesc.firstOf at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨n, hn, rule, hr, law, hl, wt, hwt, res, hres, rfl⟩ := h
  simp [firstOf, hn, hr, hl, hwt, hres]

theorem cellsOf_eq (i : Spot.Run.Input) (it : Spot.Run.Interior) : SampledTwoDesc.cellsOf i it = cellsOf i it := rfl

theorem oneStageOk_lawOk (ns : List ℕ) (law : Option _root_.Flock.Draw.Law) :
    oneStageOk ns ((law.map oneStageOf).getD .all) = SampledTwoDesc.lawOk ns law := by
  rcases law with _ | (_ | _ | _ | _) <;> rfl

/-- **The design `Select.select` samples**, from the check's first-stage law. -/
theorem slotDesign_of_firstLaw {rule : Spot.Select.Rule} {d : Spot.Program} {q : Json} {tb : Option Json}
    {law : Option _root_.Flock.Draw.Law} (h : SampledTwoDesc.firstLaw rule d q tb = .ok law) :
    slotDesign rule d q tb = .ok ⟨[(law.map oneStageOf).getD .all, .all]⟩ := by
  unfold SampledTwoDesc.firstLaw at h
  unfold slotDesign
  rcases h1 : Spot.Partition.units d q with e | n
  · simp only [h1, error_bind] at h; cases h
  simp only [h1, ok_bind] at h ⊢
  rcases h2 : rule.lawOver n with e | named
  · simp only [h2, error_bind] at h; cases h
  simp only [h2, ok_bind] at h ⊢
  rcases h3 : Spot.Draw.deriveLaw d q named tb with e | derived
  · simp only [h3, error_bind] at h; cases h
  simp only [h3, ok_bind] at h ⊢
  cases rule with
  | all => cases h; rfl
  | _ =>
    simp only at h ⊢
    rcases h4 : Spot.Draw.flockLaw ((derived.map (·.toJson)).getD named) n with e | parsed
    · simp only [h4, error_bind] at h; cases h
    simp only [h4, ok_bind] at h ⊢
    cases h
    rfl

theorem all_zip_two {α β : Type} (a b : α) (a' b' : β) (S : List (List ℕ)) {F : List ℕ → α → Bool}
    {G : List ℕ → β → Bool} (ha : ∀ s, F s a = G s a') (hb : ∀ s, F s b = G s b') :
    ([a, b].zip S).all (fun p => F p.2 p.1) = ([a', b'].zip S).all (fun p => G p.2 p.1) := by
  rcases S with _ | ⟨s₀, _ | ⟨s₁, S⟩⟩ <;> simp [ha, hb]

theorem designOk_eq (law : Option _root_.Flock.Draw.Law) (l : Spot.Slots.SlotOrder) :
    designOk ⟨[(law.map oneStageOf).getD .all, .all]⟩ l = SampledTwoDesc.designOk law l := by
  unfold designOk SampledTwoDesc.designOk
  congr 1
  exact all_zip_two _ _ _ _ _ (fun s => oneStageOk_lawOk s law) (fun s => oneStageOk_lawOk s none)

/-! ## The clauses, at free held copies -/

theorem namesOnce_eq (h : Held) : namesOnce h = SampledTwoDesc.namesOnce h.ms.length h.names := rfl

theorem writersFirst_eq (h : Held) : writersFirst h = SampledTwoDesc.writersFirst h.ms.length h.names h.wires := rfl

theorem refReadsOwn_eq (h : Held) :
    refReadsOwn h = SampledTwoDesc.refReadsOwn h.ms.length h.names h.links h.wires h.refReads
      h.interiorNames.ports := rfl

theorem wiresAlike_of {h : Held} {c : ℕ → _root_.Flock.Circuit} (hc : ∀ t, memberCircuit h t = c t) :
    wiresAlike h = SampledTwoDesc.wiresAlike h.ms.length h.names h.wires c := by
  have e : memberCircuit h = c := funext hc
  show SampledTwoDesc.wiresAlike h.ms.length h.names h.wires (memberCircuit h) = _
  rw [e]

theorem sourceRowsNamed_eq (h : Held) :
    sourceRowsNamed h = SampledTwoDesc.sourceRowsNamed h.ms.length h.names h.links h.wires (memberCircuit h) := rfl

theorem writtenRowsUnwired_eq (h : Held) :
    writtenRowsUnwired h = SampledTwoDesc.writtenRowsUnwired h.ms.length h.names h.links h.wires (memberCircuit h) :=
  rfl

theorem writtenRowsRead_eq (h : Held) :
    writtenRowsRead h = SampledTwoDesc.writtenRowsRead h.ms.length h.names h.links h.wires (memberCircuit h) := rfl

/-- **The units `unitsOf` gives a template**, counted: the check's `unitsAt`. -/
theorem size_unitsOf (r : ℕ) (ns : Array ℕ) (T t : ℕ) :
    ((unitsOf r ns T).getD t #[]).size = SampledTwoDesc.unitsAt r ns T t := by
  unfold unitsOf SampledTwoDesc.unitsAt
  by_cases ht : t < T
  · simp [ht, List.getD_eq_getElem?_getD]
  · rw [List.getD_eq_default _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp [ht]

theorem wiresMoveRows_of {h : Held} {c : ℕ → _root_.Flock.Circuit} {u : ℕ → ℕ}
    (hc : ∀ t, memberCircuit h t = c t) (hu : ∀ t, (h.vus.getD t #[]).size = u t) :
    wiresMoveRows h = SampledTwoDesc.wiresMoveRows h.ms.length h.names h.links h.wires h.rows c h.reg.ns u := by
  obtain rfl : memberCircuit h = c := funext hc
  obtain rfl : (fun t => (h.vus.getD t #[]).size) = u := funext hu
  rfl

theorem portsOnce_eq (h : Held) : portsOnce h = SampledTwoDesc.portsOnce h.ms.length (memberCircuit h) := rfl

theorem outsOk_eq (h : Held) : outsOk h = SampledTwoDesc.outsOk h.ms.length (memberCircuit h) := rfl

theorem memberOk_eq (c : _root_.Flock.Circuit) (us : Array ℕ) : memberOk (c, us) = SampledTwoDesc.memberOk c := rfl

/-- **`decided`, from its clauses.** -/
theorem decided_of {u : Unread} {it : Spot.Run.Interior} {H : Held} {l : Spot.Slots.SlotOrder} {d : Design}
    (h1 : (u.xs.length == it.circuits.size) = true) (h2 : namesOnce H = true)
    (h3 : H.interiorNames.values.isEmpty = true)
    (h4 : (Spot.Slots.layout H.reg.first.program H.reg.first.query).toOption = some l)
    (h5 : replayDesign H = some d) (h6 : H.ms.all memberOk = true) (h7 : designOk d l = true)
    (h8 : (H.k.length == H.ms.length) = true) (h9 : H.k.all (1 ≤ ·) = true)
    (h10 : (H.reg.ns.size == H.ms.length) = true) (h11 : (List.range H.ms.length).all (templateOk H) = true)
    (h12 : membersScoped H = true) (h13 : wiringOk H = true) : decided u it H = true := by
  unfold decided descCore
  rw [h4, h5]
  simp only [h1, h3, h6, h7, h8, h9, h10, h11, h12, h13, Bool.and_true, Bool.true_and]
  exact h2

/-! ## The check gives `heldOkDesc` -/

/-- **The two-stage form's descriptor check gives `heldOkDesc`**, whatever the registered copies `u₀` gives of the
rest. -/
theorem heldOkDesc_of_descOk {p : SampledTwo.Params} {bs : Core.Blobs} {i : Spot.Run.Input}
    {it : Spot.Run.Interior} (u₀ : Unread) (hd : SampledTwoDesc.descOk p bs i it = true) :
    heldOkDesc p bs i it u₀ = true := by
  unfold SampledTwoDesc.descOk at hd
  rcases hf : SampledTwoDesc.firstOf i with _ | f
  · rw [hf] at hd; cases hd
  rcases hc : SampledTwoDesc.cellsOf i it with _ | ⟨ns, ks⟩
  · rw [hf, hc] at hd; cases hd
  rcases hrs : SampledTwoDesc.refsOf p bs with _ | rs
  · rw [hf, hc, hrs] at hd; cases hd
  rw [hf, hc, hrs] at hd
  simp only at hd
  rcases hl : (Spot.Slots.layout f.verifier.program f.verifier.query).toOption with _ | l
  · rw [hl] at hd; cases hd
  rcases hlaw : (SampledTwoDesc.firstLaw f.rule f.verifier.program f.verifier.query f.verifier.workTable).toOption
    with _ | law
  · rw [hl, hlaw] at hd; cases hd
  rw [hl, hlaw] at hd
  simp only [Bool.and_eq_true] at hd
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨hT, hne⟩, hR⟩, hmem⟩, hdes⟩, hks⟩, hks1⟩, hns⟩, hwir⟩ := hd
  have hT' : p.circuits.size = it.circuits.size := beq_iff_eq.1 hT
  have hne' : rs ≠ [] := by
    intro h0
    rw [h0] at hne
    cases hne
  have hR' : ∀ R ∈ rs, SampledTwoDesc.fits R = true ∧ SampledTwoDesc.zkOk R.spec R.sts = true ∧
      SampledTwoDesc.templateOk R = true ∧ _root_.Flock.ProvedScope.scopeOkH R.stmt = true := by
    intro R hRm
    have h' := List.all_eq_true.1 hR R hRm
    simp only [Bool.and_eq_true] at h'
    exact ⟨h'.1.1.1, h'.1.1.2, h'.1.2, h'.2⟩
  have hfits : rs.all SampledTwoDesc.fits = true := List.all_eq_true.2 fun R hRm => (hR' R hRm).1
  have hlen := length_refsOf hrs
  have hat : ∀ t < it.circuits.size, ∃ R, SampledTwoDesc.refOf p bs t = .ok R ∧ rs[t]? = some R ∧ R ∈ rs := by
    intro t ht
    obtain ⟨R, hR₁, hR₂⟩ := refOf_of_refsOf hrs (hT' ▸ ht)
    exact ⟨R, hR₁, hR₂, List.mem_of_getElem? hR₂⟩
  have hxsOk : xsOk (rs.map fun R => (rawOf R).encode) = true := by
    unfold xsOk
    rw [Bool.and_eq_true, List.all_eq_true]
    refine ⟨by simpa using hne', fun x hx => ?_⟩
    obtain ⟨R, hRm, rfl⟩ := List.mem_map.1 hx
    obtain ⟨j, hj⟩ := refOf_of_mem hrs hRm
    rw [zkOk_rawOf hj (hR' R hRm).1]
    exact (hR' R hRm).2.1
  have hfirst := firstOf_of hf
  have hw' := hwir
  unfold SampledTwoDesc.wiringOk at hw'
  simp only [Bool.and_eq_true] at hw'
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hno, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩, hval⟩, -⟩, -⟩ := hw'
  unfold heldOkDesc
  rw [refStmtsOf_of_refsOf hrs]
  simp only
  unfold heldOk
  rw [hfirst, show cellsOf i it = some (ns, ks) from hc]
  simp only [Reg.program, Reg.query, hl]
  rw [dif_pos hxsOk]
  refine decided_of (l := l) (d := ⟨[(law.map oneStageOf).getD .all, .all]⟩) ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · -- as many reference statements as templates
    simp only [List.length_map, hlen]
    exact hT
  · rw [namesOnce_eq, ms_length_held₀]
    exact hno
  · exact hval
  · exact hl
  · show (slotDesign f.rule f.verifier.program f.verifier.query f.verifier.workTable).toOption = _
    rw [slotDesign_of_firstLaw (ok_of_toOption hlaw)]
    rfl
  · -- every member fits
    unfold Held.ms
    rw [List.all_eq_true]
    intro x hx
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hx
    have ht' : t < it.circuits.size := by simpa [held₀, unitsOf] using ht
    show memberOk (stmtOf (refStmt (rs.map fun R => (rawOf R).encode) t),
      (unitsOf (requests l) ns it.circuits.size).getD t #[]) = true
    rw [memberOk_eq, stmtOf_refStmt hrs hfits hne' t]
    exact List.all_eq_true.1 hmem t (List.mem_range.2 ht')
  · rw [designOk_eq]
    exact hdes
  · rw [ms_length_held₀]
    show (ks.toList.length == it.circuits.size) = true
    rw [Array.length_toList]
    exact hks
  · exact hks1
  · rw [ms_length_held₀]
    exact hns
  · -- every template's reference statement reads as a member's
    rw [List.all_eq_true]
    intro t ht
    rw [ms_length_held₀] at ht
    obtain ⟨R, hR₁, hR₂, hRm⟩ := hat t (List.mem_range.1 ht)
    refine templateOk_of hR₁ (hR' R hRm).1 ?_ (hR' R hRm).2.2.1
    show refStmt (rs.map fun R => (rawOf R).encode) t = _
    simp only [refStmt, List.getD_eq_getElem?_getD, List.getElem?_map, hR₂] <;> rfl
  · -- every member's circuit is in the hidden-output proof's scope
    unfold membersScoped
    rw [ms_length_held₀, List.all_eq_true]
    intro t ht
    obtain ⟨R, -, hR₂, hRm⟩ := hat t (List.mem_range.1 ht)
    show _root_.Flock.ProvedScope.scopeOkH (stmtOf (refStmt (rs.map fun R => (rawOf R).encode) t)) = true
    rw [stmtOf_refStmt hrs hfits hne' t, circAt_of hR₂]
    exact (hR' R hRm).2.2.2
  · -- the registered wiring
    have hmc : ∀ t, memberCircuit (held₀ i it { u₀ with xs := rs.map fun R => (rawOf R).encode }
        ⟨f.verifier, f.rule, f.closure⟩ ns ks (requests l) (zkOk_refStmt hxsOk)) t =
        SampledTwoDesc.circAt (SampledTwoDesc.circsOf rs) t :=
      fun t => stmtOf_refStmt hrs hfits hne' t
    have hu : ∀ t, ((held₀ i it { u₀ with xs := rs.map fun R => (rawOf R).encode }
        ⟨f.verifier, f.rule, f.closure⟩ ns ks (requests l) (zkOk_refStmt hxsOk)).vus.getD t #[]).size =
        SampledTwoDesc.unitsAt (requests l) ns it.circuits.size t :=
      fun t => size_unitsOf (requests l) ns it.circuits.size t
    unfold wiringOk
    rw [namesOnce_eq, writersFirst_eq, sourceRowsNamed_eq, wiresMoveRows_of hmc hu, wiresAlike_of hmc,
      writtenRowsUnwired_eq, writtenRowsRead_eq, refReadsOwn_eq, portsOnce_eq, outsOk_eq, funext hmc,
      ms_length_held₀]
    exact hwir

/-! ## The check gives a held copy -/

/-- **The two-stage form's descriptor check gives a held copy**: where it passes at the form's params and blobs and
the run of record's input and interior, the blobs give the templates' reference statements (`refStmtsOf`) and
`heldOf` builds a held copy at them, whatever `u₀` gives of the rest of what the run does not read. It reads no
acceptance: the check alone decides what `heldOf` asks. -/
theorem heldOf_of_descOk {p : SampledTwo.Params} {bs : Core.Blobs} {i : Spot.Run.Input} {it : Spot.Run.Interior}
    (u₀ : Unread) (hd : SampledTwoDesc.descOk p bs i it = true) :
    ∃ xs h, refStmtsOf p bs = some xs ∧ heldOf i it { u₀ with xs } = some h := by
  have hk := heldOkDesc_of_descOk u₀ hd
  unfold heldOkDesc at hk
  cases hx : refStmtsOf p bs with
  | none => rw [hx] at hk; simp at hk
  | some xs =>
    rw [hx] at hk
    obtain ⟨h, hh⟩ := heldOf_of_ok { u₀ with xs } hk
    exact ⟨xs, h, rfl, hh⟩

end SpotFlock.HeldOf
