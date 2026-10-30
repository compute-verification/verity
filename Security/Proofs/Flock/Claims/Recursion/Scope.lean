import Definitions.Claims.Recursion
import Proofs.Flock.Claims.DeployCode
import Flock.RecSetup

/-!
# `Claims.Recursion.ScopeMatches` at the compiled verifier's parser and scope check

The compiled verifier (`flock-verify verify --rec-inner`) reads, of a recursion statement, the inner statement's bytes,
the inner statement's files where it holds them (`RecCoins.Source`), its own inner shape (`RecVStar.Inner`,
`--rec-inner`) and each V* session's record (`Read`). Its setup refuses an inner shape whose V* list
`RecVStar.statements` refuses, at candidate 3's sparse shape one whose inner statement `RecCoins.checkAtC3`
refuses (`checkInnerRange`), and where it holds the inner statement's files one that is not the inner statement's, or an
inner statement set up otherwise than its files set it up (`RecSetup.checkShape`, `RecSetup.checkForm`); every V*
statement carries those files (`RecCoins.filesInput` refuses them in part). Its verdict refuses a V* session whose record holds no draw (`ProvedScope.drawLaw`) or
whose draw is not every unit of its population (`RecSetup.fullDraw`), and a session over an inner session whose
record holds no draw (`RecSetup.innerDrawn`); the compiled statement's reader refuses an inner statement without a draw
(`RecDeploy.innerOk`). There is no other recursion-level refusal.

* `parseOf`: the deployed parser, what the verifier reads where its setup accepts the inner shape, readable (`toStmt`);
* `scopeCheckOf`: the parser accepts, the inner scope check takes the inner statement, and `RecSetup.fullDraw` takes
  every V* session's draw at the verifier's inner shape (`fullDrawOk`);
* `inScope_iff`: at a statement the parser accepts, `InScope` holds exactly when every V* session's draw is full, the
  setup's checks giving every condition on the inner shape (`statements_ok`, `candidate3_ok`, `checkAtC3_ok`,
  `checkInnerRange_ok`);
* `scopeMatches`: `ScopeMatches` at every inner protocol whose scope its check matches; `scopeMatches_flock` with
  Flock inside, and `scopeMatches_deployed` at the deployed plain Flock inside.

`InScope` states the inner shape's numbers that `statements` and `checkAtC3` check. `statements` also builds V*'s list
at the shape (V*'s parts as many as the verifier's groups, a holo shape's `h` round in a group, R's opening's parts),
and the parser refuses a shape where that fails, as the verifier's setup does.
-/

namespace Flock.RecScope

/-- **What the compiled verifier reads** of a recursion statement: the inner statement's bytes, the inner statement's
files where it holds them, its own inner shape, and each V* session's record. -/
structure Read where
  inner : Claims.Msg
  src : Option RecCoins.Source
  shape : RecVStar.Inner
  records : List String

/-- The inner statement as the verifier sets it up from its files (`RecCoins.Files.stmt`), where it holds them. -/
def innerStmt : Option RecCoins.Source → Option Stmt
  | some (.files f) => (RecCoins.Files.stmt "the inner session" f).toOption.map (·.2)
  | _ => none

/-- **The unit range's slots lie inside the block** at the RecSparse shape `sp`: the layout factors at the inner
circuit's unit range are computed, at the inner points' `k_log − 6` coordinates, as `checkInnerRange` computes them. -/
def rangeOk (sp : RecSparse.Shape) (st : Stmt) : Bool :=
  match RecCoins.unitRangeOf st with
  | .ok rng => (RecCoins.layoutFactors sp (Array.replicate (st.c.kLog - 6) (0 : F128))
      (Array.replicate (st.c.kLog - 6) (0 : F128)) rng).isOk
  | .error _ => false

/-- A draw, readable: its `subset` law's `k`, population, units and closure. -/
def drawOf (D : Draw.UnitDraw) : Claims.Recursion.Draw where
  subset := match D.law with
    | .subset k => some k
    | _ => none
  population := D.population
  units := D.units.toList
  closure := D.closure.toList

/-- A V* session's draw as its record holds it, readable; `none` where the record holds none, or one that doesn't
read. -/
def recordDrawOf (record : String) : Option Claims.Recursion.Draw :=
  (Witness.recordDraw record).bind fun d => match Draw.UnitDraw.ofJson d with
    | .ok D => some (drawOf D)
    | .error _ => none

/-- **The recursion's statement, readable**, from what the verifier reads: no circuit commitment (the compiled verifier
has no private-circuit variant), the inner statement's dimensions as it sets it up, its inner shape as the form, and
each V* session's draw. -/
def toStmt (r : Read) : Claims.Recursion.Stmt where
  inner := r.inner
  innerDims := (innerStmt r.src).map fun st => (st.m, st.c.kLog, st.regions.size)
  circ := none
  form :=
    { m := r.shape.m, kLog := r.shape.kLog, regions := r.shape.regions, claims := r.shape.claims
      holo := r.shape.holo
      sparse := r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)
      stacked := r.shape.stacked.map fun s => (s.m, s.claims)
      unitRangeInBlock := match r.shape.sparse, innerStmt r.src with
        | some sp, some st => rangeOk sp st
        | _, _ => false }
  draws := r.records.map recordDrawOf

/-- **The verifier's setup accepts the inner shape**: V*'s list at it (`RecVStar.statements`), candidate 3's checks
on the inner statement (`RecCoins.checkAtC3`), the inner shape the inner statement's and the inner statement set up as
its files set it up where the verifier holds its files (`RecSetup.checkShape`, with `RecSetup.checkForm`), and the
verifier holds them: `RecCoins.filesInput` refuses the inner statement's circuit and public file without its partition,
and every V* statement carries those two (`RecDeploy.partRaw`). -/
def accepts (r : Read) : Bool :=
  (RecVStar.statements r.shape).isOk && (RecCoins.checkAtC3 (some r.shape) r.src).isOk &&
    (RecSetup.checkShape (some r.shape) r.src).isOk && r.src.isSome

/-- What the verifier reads of a statement's bytes (`decode`), where its setup accepts the inner shape. -/
def parseBytes (decode : Claims.Msg → Option Read) (x : Claims.Msg) : Option Read := (decode x).filter accepts

/-- **The deployed parser**: what the verifier reads where its setup accepts, readable. -/
def parseOf (decode : Claims.Msg → Option Read) (x : Claims.Msg) : Option Claims.Recursion.Stmt :=
  (parseBytes decode x).map toStmt

/-- **A V* session's draw passes the verdict's refusals** under the verifier's inner shape `i`: its record holds a draw
(`ProvedScope.drawLaw` refuses a session without one), and `RecSetup.fullDraw` takes it, as `verify` calls it in a
recursive session (under an inner shape its `rec` flag changes nothing, `fullDraw_some`). -/
def fullDrawOk (i : RecVStar.Inner) (record : String) : Bool :=
  match Witness.recordDraw record with
  | some d => (RecSetup.fullDraw (some i) true (some d) record).isOk
  | none => false

/-- Under an inner shape, `RecSetup.fullDraw`'s `noInner` refusal never fires, so its `rec` flag changes nothing. -/
theorem fullDraw_some (i : RecVStar.Inner) (rec rec' : Bool) (draw : Option Lean.Json) (record : String) :
    RecSetup.fullDraw (some i) rec draw record = RecSetup.fullDraw (some i) rec' draw record := by
  simp [RecSetup.fullDraw]

/-- **The deployed scope check**: the parser accepts, the inner scope check `ivScope` takes the inner statement, and
every V* session's draw passes. -/
def scopeCheckOf (ivScope : Claims.Msg → Bool) (decode : Claims.Msg → Option Read) (x : Claims.Msg) : Bool :=
  match parseBytes decode x with
  | some r => ivScope r.inner && r.records.all (fullDrawOk r.shape)
  | none => false

/-! ## The shape the setup accepts -/

theorem isOk_iff {ε α : Type} (e : Except ε α) : e.isOk = true ↔ ∃ a, e = .ok a := by
  cases e <;> simp [Except.isOk, Except.toBool]

theorem fast100_bounds {m : ℕ} {s : Schedule} (h : fast100 m = some s) : 22 ≤ m ∧ m ≤ 35 := by
  by_contra hc
  have hc' : (m < 22 || m > 35) = true := by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega
  simp [fast100, hc'] at h

theorem shape?_ok {m k c : ℕ} {sh : Shape} (h : RecResiduals.shape? m k c = some sh) :
    22 ≤ m ∧ m ≤ 35 ∧ 6 ≤ k := by
  unfold RecResiduals.shape? at h
  obtain ⟨s, hs, h⟩ := Option.bind_eq_some_iff.1 h
  refine ⟨(fast100_bounds hs).1, (fast100_bounds hs).2, ?_⟩
  simp at h
  exact h.1.1

theorem pLog_eq (sp : RecSparse.Shape) : sp.pLog = Claims.Recursion.pLog sp.ell := by
  unfold RecSparse.Shape.pLog RecSparse.Shape.P Claims.Recursion.pLog
  rw [Nat.log2_eq_log_two, show 2 * sp.ell + 2 - 1 = 2 * sp.ell + 1 by omega]

/-- **Candidate 3's list at the inner shape**: no RecSparse phase and no R, or a valid RecSparse shape and R at its
`table_m`, with a fast100 schedule and the two reps' claims. -/
theorem candidate3_ok {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.candidate3Statements i = .ok ss) :
    i.sparse = none ∧ i.stacked = none ∨
      ∃ sp r, i.sparse = some sp ∧ sp.valid = true ∧ i.stacked = some r ∧ r.claims = 2 ∧
        r.m = RecVStar.tableM sp ∧ 22 ≤ r.m ∧ r.m ≤ 35 := by
  unfold RecVStar.candidate3Statements at h
  rcases hs : i.sparse with _ | sp <;> rcases hr : i.stacked with _ | r <;> simp only [hs, hr] at h
  · exact Or.inl ⟨rfl, rfl⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  · by_cases hv : sp.valid = true <;> simp [hv, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  · right
    by_cases hv : sp.valid = true
    · by_cases hc : r.claims = 2
      · by_cases hm : r.m = RecVStar.tableM sp
        · rcases hf : fast100 r.m with _ | sch
          · rw [hm] at hf
            simp [RecVStar.STACKED_CLAIMS, hv, hc, hm, hf, throw, throwThe,
              MonadExceptOf.throw] at h
          · exact ⟨sp, r, rfl, hv, rfl, hc, hm, (fast100_bounds hf).1, (fast100_bounds hf).2⟩
        · simp [RecVStar.STACKED_CLAIMS, hv, hc, hm, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
      · simp [RecVStar.STACKED_CLAIMS, hv, hc, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    · simp [hv, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h

/-- **V*'s list at the inner shape**: a fast100 `m`, `k_log ≥ 6`, two claims and two per region, and candidate 3's
list. -/
theorem statements_ok {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    22 ≤ i.m ∧ i.m ≤ 35 ∧ 6 ≤ i.kLog ∧ i.claims = 2 + 2 * i.regions ∧
      ∃ ss', RecVStar.candidate3Statements i = .ok ss' := by
  unfold RecVStar.statements at h
  rcases hsh : RecResiduals.shape? i.m i.kLog i.claims with _ | sh0
  · simp [hsh, throw, throwThe, MonadExceptOf.throw] at h
  obtain ⟨h1, h2, h3⟩ := shape?_ok hsh
  by_cases hc : i.claims = 2 + 2 * i.regions
  · refine ⟨h1, h2, h3, hc, ?_⟩
    rcases hc3 : RecVStar.candidate3Statements i with e | ss'
    · exfalso
      simp only [hc, hc3, bne_self_eq_false, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
      repeat' split at h
      all_goals simp_all
    · exact ⟨ss', rfl⟩
  · simp [hsh, hc, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h

/-- **Candidate 3's checks on the inner statement** (`checkInnerRange`), at a RecSparse shape `sp`: `T_LOG = 0`, the
inner shape's `(m, k_log, regions)` the inner statement's, a valid shape, the inner points not short of `ELL − 6`
coordinates, and the layout factors at the unit range computed. -/
theorem checkInnerRange_ok {i : RecVStar.Inner} {st : Stmt} {sp : RecSparse.Shape} (hsp : i.sparse = some sp)
    (h : RecCoins.checkInnerRange i st = .ok ()) :
    sp.tLog = 0 ∧ i.m = st.m ∧ i.kLog = st.c.kLog ∧ i.regions = st.regions.size ∧ sp.valid = true ∧
      ¬ st.c.kLog - 6 < sp.ell - RecSparse.PRE_ROUNDS ∧ rangeOk sp st = true := by
  unfold RecCoins.checkInnerRange RecVStar.checkInnerStmt RecVStar.checkInnerShape at h
  simp only [hsp, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  by_cases ht : sp.tLog = 0
  · by_cases hd : i.m = st.m ∧ i.kLog = st.c.kLog ∧ i.regions = st.regions.size
    · obtain ⟨hd1, hd2, hd3⟩ := hd
      by_cases hv : sp.valid = true
      · by_cases hx : st.c.kLog - 6 < sp.ell - RecSparse.PRE_ROUNDS
        · simp [ht, hd1, hd2, hd3, hv, hx, pure, Except.pure] at h
        · refine ⟨ht, hd1, hd2, hd3, hv, hx, ?_⟩
          simp only [ht, hd1, hd2, hd3, hv, hx, bne_self_eq_false, Bool.or_self, Bool.false_eq_true, ↓reduceIte,
            Bool.not_true, Array.size_replicate] at h
          unfold rangeOk
          rcases hu : RecCoins.unitRangeOf st with e | rng <;> simp only [hu, Except.mapError] at h
          · cases h
          · rcases hl : RecCoins.layoutFactors sp (Array.replicate (st.c.kLog - 6) (0 : F128))
              (Array.replicate (st.c.kLog - 6) (0 : F128)) rng with e | v
            · simp [hl, pure, Except.pure] at h
            · simp [hl, Except.isOk, Except.toBool]
      · simp [ht, hd1, hd2, hd3, hv, pure, Except.pure] at h
    · simp only [not_and_or] at hd
      rcases hd with hd | hd | hd <;> simp [ht, hd] at h
  · simp [ht] at h

/-- **`checkAtC3` at a RecSparse shape**: the verifier holds the inner statement's files, and `checkInnerRange` passes
on the inner statement they set up. -/
theorem checkAtC3_ok {i : RecVStar.Inner} {src : Option RecCoins.Source} {sp : RecSparse.Shape}
    (hsp : i.sparse = some sp) (h : RecCoins.checkAtC3 (some i) src = .ok ()) :
    ∃ st, innerStmt src = some st ∧ RecCoins.checkInnerRange i st = .ok () := by
  unfold RecCoins.checkAtC3 at h
  simp only [hsp, Option.isNone_some, Bool.false_eq_true, ↓reduceIte, bind, Except.bind, throw, throwThe,
    MonadExceptOf.throw] at h
  rcases src with _ | ⟨f⟩ | ⟨j, rec⟩
  · cases h
  · rcases hf : RecCoins.Files.stmt "the inner session" f with e | ⟨spec, st⟩ <;> simp only [hf] at h
    · cases h
    · exact ⟨st, by simp [innerStmt, hf, Except.toOption], h⟩
  · cases h

/-! ## The full draw -/

/-- Under an inner shape, `RecSetup.fullDraw` doesn't read whether the session is recursive: its `noInner` refusal is of
a recursive session without one. -/
theorem fullDraw_some_rec (i : RecVStar.Inner) (b : Bool) (d : Option Lean.Json) (record : String) :
    RecSetup.fullDraw (some i) b d record = RecSetup.fullDraw (some i) true d record := by
  simp [RecSetup.fullDraw]

theorem fullDrawOk_iff (i : RecVStar.Inner) (record : String) :
    fullDrawOk i record = true ↔ ∃ e, recordDrawOf record = some e ∧ e.Full := by
  unfold fullDrawOk recordDrawOf RecSetup.fullDraw
  simp only [Option.isNone_some, Bool.and_false, Bool.false_eq_true, ↓reduceIte]
  rcases hd : Witness.recordDraw record with _ | d
  · simp
  rcases hD : Draw.UnitDraw.ofJson d with e | D
  · simp [hD, bind, Except.bind, Except.isOk, Except.toBool]
  simp only [hD, Option.bind_some, Option.some.injEq, exists_eq_left', bind, Except.bind]
  unfold Witness.fullDraw
  simp only [hd, hD, bind, Except.bind]
  unfold Claims.Recursion.Draw.Full drawOf
  rcases hl : D.law with k | _ | _ | _
  · simp only [Option.some.injEq, true_and]
    rw [← Array.toList_range, Array.toList_inj, Array.toList_eq_nil_iff]
    by_cases hc : k = D.population ∧ D.units = Array.range D.population ∧ D.closure = #[]
    · simp [hc, pure, Except.pure, Except.mapError, Except.isOk, Except.toBool]
    · simp only [hc, ↓reduceIte, throw, throwThe, MonadExceptOf.throw, Except.mapError, Except.isOk, Except.toBool,
        Bool.false_eq_true]
  all_goals simp [throw, throwThe, MonadExceptOf.throw, Except.mapError, Except.isOk, Except.toBool]

/-! ## The scope -/

/-- **`InScope` at a statement the setup accepts** holds exactly when every V* session's draw passes: the setup's
checks give every condition on the inner shape. -/
theorem inScope_iff {r : Read} (hr : accepts r = true) :
    Claims.Recursion.InScope (toStmt r) ↔ r.records.all (fullDrawOk r.shape) = true := by
  unfold accepts at hr
  simp only [Bool.and_eq_true, isOk_iff] at hr
  obtain ⟨⟨⟨⟨ss, hs⟩, ⟨u, hc⟩⟩, -⟩, -⟩ := hr
  obtain ⟨h1, h2, h3, h4, ss', hc3⟩ := statements_ok hs
  have hshape : (toStmt r).form.sparse = none ∧ (toStmt r).form.stacked = none ∨
      ∃ nu ell sLog, (toStmt r).form.sparse = some (nu, ell, sLog, 0) ∧ 1 ≤ nu ∧ 6 ≤ ell ∧
        ell ≤ (toStmt r).form.kLog ∧
        (toStmt r).innerDims = some ((toStmt r).form.m, (toStmt r).form.kLog, (toStmt r).form.regions) ∧
        (toStmt r).form.unitRangeInBlock = true ∧
        (toStmt r).form.stacked = some (max 22 (nu + Claims.Recursion.pLog ell), 2) ∧
        max 22 (nu + Claims.Recursion.pLog ell) ≤ 35 := by
    rcases candidate3_ok hc3 with ⟨hsp, hst⟩ | ⟨sp, R, hsp, hv, hst, hcl, hm, hm1, hm2⟩
    · left
      simp [toStmt, hsp, hst]
    · right
      obtain ⟨st, hst', hcr⟩ := checkAtC3_ok hsp hc
      obtain ⟨ht, hd1, hd2, hd3, -, hx, hrg⟩ := checkInnerRange_ok hsp hcr
      have hv' : 1 ≤ sp.nu ∧ 6 ≤ sp.ell ∧ sp.tLog ≤ 16 := by
        simp only [RecSparse.Shape.valid, RecSparse.PRE_ROUNDS, Bool.and_eq_true, decide_eq_true_eq] at hv
        exact ⟨hv.1.1, of_decide_eq_true hv.1.2, hv.2⟩
      have htm : R.m = max 22 (sp.nu + Claims.Recursion.pLog sp.ell) := by
        rw [hm, RecVStar.tableM, pLog_eq, ht, RecVStar.PAD_TO, Nat.add_zero]
      refine ⟨sp.nu, sp.ell, sp.sLog, by simp [toStmt, hsp, ht], hv'.1, hv'.2.1, ?_, ?_, ?_, ?_, ?_⟩
      · simp only [toStmt]
        simp only [RecSparse.PRE_ROUNDS] at hx
        omega
      · simp [toStmt, hst', hd1, hd2, hd3]
      · simp [toStmt, hsp, hst', hrg]
      · simp [toStmt, hst, hcl, htm]
      · omega
  unfold Claims.Recursion.InScope
  constructor
  · rintro ⟨-, -, -, -, -, -, hd⟩
    rw [List.all_eq_true]
    intro rec hrec
    exact (fullDrawOk_iff r.shape rec).2 (hd _ (List.mem_map_of_mem hrec))
  · intro hall
    refine ⟨rfl, h1, h2, h3, h4, hshape, ?_⟩
    intro d hd
    obtain ⟨rec, hrec, rfl⟩ := List.mem_map.1 hd
    exact (fullDrawOk_iff r.shape rec).1 (List.all_eq_true.1 hall rec hrec)

/-- **The readable scope at the verifier's parser**: the setup accepts what the verifier reads, the inner statement is
in the inner scope, and every V* session's draw passes. -/
theorem scope_iff {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin W) (decode : Claims.Msg → Option Read)
    (hparse : D.parse = parseOf decode) (x : Claims.Msg) :
    Claims.Recursion.Scope D I x ↔
      ∃ r, parseBytes decode x = some r ∧ I.Scope r.inner ∧ r.records.all (fullDrawOk r.shape) = true := by
  unfold Claims.Recursion.Scope
  rw [hparse]
  constructor
  · rintro ⟨s, hs, hI, hin⟩
    obtain ⟨r, hr, rfl⟩ := Option.map_eq_some_iff.1 hs
    have ha : accepts r = true := (Option.filter_eq_some_iff.1 hr).2
    exact ⟨r, hr, hI, (inScope_iff ha).1 hin⟩
  · rintro ⟨r, hr, hI, hd⟩
    have ha : accepts r = true := (Option.filter_eq_some_iff.1 hr).2
    exact ⟨toStmt r, by simp [parseOf, hr], hI, (inScope_iff ha).2 hd⟩

/-- **`ScopeMatches` at the compiled verifier's parser and scope check**, at every inner protocol whose readable scope
its scope check `ivScope` matches. -/
theorem scopeMatches {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin W) (decode : Claims.Msg → Option Read)
    (ivScope : Claims.Msg → Bool) (hparse : D.parse = parseOf decode)
    (hcheck : D.scopeCheck = scopeCheckOf ivScope decode) (hI : Claims.ScopeMatches I.Scope ivScope) :
    Claims.Recursion.ScopeMatches D I := by
  intro x
  rw [scope_iff D I decode hparse, hcheck, scopeCheckOf]
  constructor
  · rintro ⟨r, hr, hIr, hd⟩
    rw [hr, Bool.and_eq_true]
    exact ⟨(hI r.inner).1 hIr, hd⟩
  · intro hx
    split at hx
    · rename_i r hr
      rw [Bool.and_eq_true] at hx
      exact ⟨r, hr, (hI r.inner).2 hx.1, hx.2⟩
    · cases hx

/-- **`ScopeMatches` with Flock inside**: at the verifier's parser and scope check over the inner Flock's scope check,
wherever the inner Flock's readable scope is its scope check (`Claims.Flock.ScopeMatches`, which the deployed records
meet: `FlockDeploy.scopeMatches_plain`, `scopeMatches_zk`). -/
theorem scopeMatches_flock {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (D : Claims.Recursion.Deploy Coin) (DI : Claims.Flock.Deploy Coin Col T) (decI : List (ℕ × Col) → List T)
    (decode : Claims.Msg → Option Read) (hparse : D.parse = parseOf decode)
    (hcheck : D.scopeCheck = scopeCheckOf DI.scopeCheck decode) (hI : Claims.Flock.ScopeMatches DI) :
    Claims.Recursion.ScopeMatches D (Claims.Recursion.Inner.ofFlock DI decI) :=
  scopeMatches D _ decode DI.scopeCheck hparse hcheck hI

/-- **`ScopeMatches` at the deployed inner Flock**: the inner session is a plain one (`RecVStar.innerTags`), so its
scope check is the deployed plain verifier's (`FlockDeploy.deployPlain`), whose readable scope it is
(`FlockDeploy.scopeMatches_plain`), at every inner decoder. -/
theorem scopeMatches_deployed (rel : Readable.Parsed → Readable.Relation) (D : Claims.Recursion.Deploy FlockDeploy.Coin)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (decode : Claims.Msg → Option Read)
    (hparse : D.parse = parseOf decode)
    (hcheck : D.scopeCheck = scopeCheckOf (FlockDeploy.deployPlain rel).scopeCheck decode) :
    Claims.Recursion.ScopeMatches D (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) :=
  scopeMatches_flock D _ decI decode hparse hcheck (FlockDeploy.scopeMatches_plain rel)

end Flock.RecScope
