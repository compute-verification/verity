import Proofs.Flock.C3.Sparse.MsgsLayers
import Proofs.Flock.C3.PubRows.Shape
import Proofs.Flock.C3.Opening.SessionCoins
import Proofs.Flock.C3.Compose.RsFacts
import Proofs.Flock.C3.Reps.WRead

/-!
# `SparseSessionAt` from the sparse session's acceptance

`SparseSessionAt` at the rec-sparse session `s` of V*'s list, at copy 0 of a unit `u` with `u.val = i.val`, at any
coins record `C` every hold's `c3Of` is (`C3Held`'s form: W's `c3Held_heldC3` gives it at `heldC3 ish sp (held (wr t))`,
with `held x := innerCommitsAt x.1 x.2 st reg`), the words copy 0 holds at the message columns (`wSparse`, `rdOf`'s
array) and coins whose phase coins are check 5's (`csOf`).

* `checkSparse_coins`, `check_sparseCoins`: check 5 (`RecCoins.check`) at a `rec-sparse/…` unit holds the coins record
  (`c3`) and computes both reps' sparse coins (`sparseCoins`) under the unit's name. `stackedParams_sparse`,
  `openingParams_sparse`: a RecSparse name is no stacked level's or opening part's, so check 5 reaches `checkSparse`.
  `sparseParams_of_check`: a unit `RecSparse.check` accepted under `unitName sh` has `params` `sh`.
* `c3_eq_of_held`: the record check 5 holds is `C` where every hold's `c3Of` is `C`.
* `nw_eq_msum`: w's elements are the message ports' words exactly (`claim` is the last port, and no earlier one).
* `CoinsAccepted`, `coinsAccepted_of_keep`, `coinsAccepted_checked`: where a checked session accepts, a transcript it
  accepts passes `ChecksC3`. `stmt0_of_setup`: an accepted session's setup has a first statement, of its circuit.
  `sparseCoins_of_accepted`: at the session V*'s list names RecSparse, its circuit is `unitName` of the session's
  shape (`unitShape`, the inner shape's `sparse` entry, `SparseIs`) and check 5 gives both reps' coins.
* `sparseSessionAt_of_regs`: `SparseSessionAt`, and its copy with `un.val = i.val` (the h-half tie's pin, K's
  `h_of_sessionAt`), from that, the constant 1 (`hone`), the message columns' registrations
  (`hkeys`, which `bitAt_of_regBit` makes `wSparse`'s bits), and the registrations of `v` (`hv`), `tau` (`htau`) and the
  zero outputs (`hz`), which the verifier does not check against the coins (`rec-sparse-tau` and `rec-sparse-zero` are
  the prover's registrations, and no check reads their rows).
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge (wbit)
open FlockVBridge.Decode (RegBit)
open FlockVBridge.OfVStar (ReadCol rowCol)
open FlockVBridge.AccChain (RegOut)
open FlockVBridge.Family (bitAt_of_regBit ioColAt_of_readCol)
open FlockVBridge.Sparse (SCoins nw verifierRow tauBase vBaseS sparseBase shOf)

/-! ## Check 5 at a rec-sparse unit -/

/-- **`checkSparse` computes both reps' sparse coins** under the unit's name. -/
theorem checkSparse_coins {inn : RecCoins.Inner} {c3 : RecCoins.C3} {c : Flock.Circuit} {pub : Public}
    {own : Array Registered.Port} {sh : RecSparse.Shape} (h : RecCoins.checkSparse inn c3 c pub own sh = .ok ()) :
    ∀ r < 2, ∃ sc, RecCoins.sparseCoins c3 c.unit.name r = .ok sc := by
  unfold RecCoins.checkSparse at h
  by_cases hs : (sh != c3.sparse) = true
  · simp only [hs, ↓reduceIte] at h
    cases h
  · simp only [hs, Bool.false_eq_true, ↓reduceIte] at h
    obtain ⟨x, -, h⟩ := FlockVBridge.PartS.bind_ok h
    obtain ⟨rng, -, h⟩ := FlockVBridge.PartS.bind_ok h
    obtain ⟨rows, hrows, -⟩ := FlockVBridge.PartS.bind_ok h
    obtain ⟨a, b, h0, h1, -⟩ := FlockC3.Opening.mapM_two hrows
    intro r hr
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · obtain ⟨rep, -, h0⟩ := FlockVBridge.PartS.bind_ok h0
      obtain ⟨sc, hsc, -⟩ := FlockVBridge.PartS.bind_ok h0
      exact ⟨sc, hsc⟩
    · obtain ⟨rep, -, h1⟩ := FlockVBridge.PartS.bind_ok h1
      obtain ⟨sc, hsc, -⟩ := FlockVBridge.PartS.bind_ok h1
      exact ⟨sc, hsc⟩

/-- **Check 5 at a `rec-sparse/…` unit** that is no stacked level and no opening part: the verifier holds the coins
record, and `checkSparse` computes both reps' sparse coins. -/
theorem check_sparseCoins {st : Flock.Stmt} {own : Array Registered.Port} {files : Option RecCoins.Source}
    {c3o : Option RecCoins.C3} {sh : RecSparse.Shape} (h : RecCoins.check st (some own) files c3o = .ok ())
    (hp : RecSparse.params st.c.unit.name = some sh) (hso : RecCoins.stackedParams st.c.unit.name = none)
    (hop : RecOpening.params st.c.unit.name = none) :
    ∃ c3, c3o = some c3 ∧ ∀ r < 2, ∃ sc, RecCoins.sparseCoins c3 st.c.unit.name r = .ok sc := by
  unfold RecCoins.check at h
  simp only [hp, hso, hop, Option.isNone_some, Option.isNone_none, Bool.and_false, Bool.false_and,
    Bool.false_eq_true, ↓reduceIte] at h
  cases files with
  | none => cases h
  | some f =>
    obtain ⟨inn, -, h⟩ := FlockVBridge.PartS.bind_ok h
    cases c3o with
    | none => cases h
    | some c3 =>
      obtain ⟨u, hu, h⟩ := FlockVBridge.PartS.bind_ok h
      dsimp only at hu
      cases hu
      exact ⟨c3, rfl, checkSparse_coins h⟩

/-- **Check 5 at a `rec-sparse/…` unit holds a source** (`Inner.load`'s files): without one it refuses. -/
theorem files_of_check {st : Flock.Stmt} {own : Array Registered.Port} {files : Option RecCoins.Source}
    {c3o : Option RecCoins.C3} {sh : RecSparse.Shape} (h : RecCoins.check st (some own) files c3o = .ok ())
    (hp : RecSparse.params st.c.unit.name = some sh) (hso : RecCoins.stackedParams st.c.unit.name = none)
    (hop : RecOpening.params st.c.unit.name = none) : ∃ f, files = some f := by
  unfold RecCoins.check at h
  simp only [hp, hso, hop, Option.isNone_some, Option.isNone_none, Bool.and_false, Bool.false_and,
    Bool.false_eq_true, ↓reduceIte] at h
  cases files with
  | none => cases h
  | some f => exact ⟨f, rfl⟩

theorem stackedParams_sparse (sh : RecSparse.Shape) : RecCoins.stackedParams (RecSparse.unitName sh) = none := by
  show RecOpen.stackedParams _ = none
  cases hq : RecOpen.stackedParams (RecSparse.unitName sh) with
  | none => rfl
  | some p =>
    obtain ⟨L, H⟩ := p
    exact (((FlockC3.PubRows.sparse_ne_names sh).2.2.1 L H) (RecOpen.stackedName_of_params hq).symm).elim

theorem openingParams_sparse (sh : RecSparse.Shape) : RecOpening.params (RecSparse.unitName sh) = none := by
  cases hq : RecOpening.params (RecSparse.unitName sh) with
  | none => rfl
  | some p =>
    obtain ⟨m, cl, j⟩ := p
    exact (((FlockC3.PubRows.sparse_ne_names sh).2.2.2 m cl j) (openingName_params hq).symm).elim

theorem startsWith_sparseName (sh : RecSparse.Shape) : (RecSparse.unitName sh).startsWith "rec-sparse/" = true := by
  have : RecSparse.unitName sh = "rec-sparse/" ++ s!"n{sh.nu}-e{sh.ell}-s{sh.sLog}-t{sh.tLog}" := by
    simp [RecSparse.unitName, ← String.toList_inj, List.append_assoc, FlockVBridge.toString_str]
  rw [this]
  simp [String.startsWith_string_iff]

/-- **A unit `RecSparse.check` accepted under the name `unitName sh` has `params` `sh`**: where `params` is `none`,
the check refuses a name beginning `rec-sparse/`. -/
theorem sparseParams_of_check {c : Flock.Circuit} (h : RecSparse.check c = .ok ()) {sh : RecSparse.Shape}
    (hn : c.unit.name = RecSparse.unitName sh) : RecSparse.params c.unit.name = some sh := by
  cases hp : RecSparse.params c.unit.name with
  | some sh' => rw [FlockC3.PubRows.sparse_unitName_inj ((FlockVBridge.Sparse.unitName_params hp).trans hn)]
  | none =>
    unfold RecSparse.check at h
    rw [hp] at h
    dsimp only at h
    rw [hn, startsWith_sparseName] at h
    cases h

/-- **w's elements are the message ports' words**: `claim`, w's last element (`mp_offsets`), is the last port, and no
earlier port is named `claim`. -/
theorem nw_eq_msum (sh : RecSparse.Shape) : nw (shOf sh) = ((RecSparse.messagePorts sh).map (·.2)).sum := by
  obtain ⟨l, hl, hne⟩ : ∃ l, RecSparse.messagePorts sh = l ++ [("claim", 1)] ∧ ∀ p ∈ l, p.1 ≠ "claim" := by
    refine ⟨_, rfl, fun p hp => ?_⟩
    simp only [List.mem_append, List.mem_singleton, List.mem_map] at hp
    rcases hp with ((((hp | ⟨j, -, rfl⟩) | hp) | ⟨⟨d, j⟩, -, rfl⟩) | hp) | hp
    · subst hp; decide
    · exact FlockVBridge.Sparse.pre_ne j (by decide)
    · subst hp; decide
    · exact FlockVBridge.Sparse.sc_ne j (by decide)
    · subst hp; show "u" ≠ "claim"; decide
    · split at hp
      · simp only [List.mem_singleton] at hp
        subst hp; show "vt" ≠ "claim"; decide
      · simp at hp
  have h1 : ((RecSparse.messagePorts sh).map (·.2)).sum = RecSparse.offset (RecSparse.messagePorts sh) "claim" + 1 := by
    rw [hl, FlockVBridge.Sparse.offset_append _ _ _ hne]
    simp [RecSparse.offset]
  rw [h1, (FlockVBridge.Sparse.mp_offsets sh).2.2.2.2.2.2]
  rfl

/-! ## The coins record and the words -/

/-- **The coins record check 5 holds is `C`** where every hold's `c3Of` is `C` (`C3Held`'s form, which W's
`c3Held_heldC3` gives at `heldC3 ish sp (held (wr t))`). -/
theorem c3_eq_of_held {ish : RecVStar.Inner} {r : Except String RecVStar.InnerCommits} {c3 C : RecCoins.C3}
    (h : Flock.Guarantees.c3Of ish r.toOption = some c3)
    (hC : ∀ cs, r = .ok cs → Flock.Guarantees.c3Of ish (some cs) = some C) : c3 = C := by
  cases r with
  | error e =>
    unfold Flock.Guarantees.c3Of at h
    cases h
  | ok cs => exact Option.some.inj (h.symm.trans (hC cs rfl))

theorem c3Of_sparse {ish : RecVStar.Inner} {cs : Option RecVStar.InnerCommits} {c3 : RecCoins.C3}
    (h : Flock.Guarantees.c3Of ish cs = some c3) {sp : RecSparse.Shape} (hsp : ish.sparse = some sp) :
    c3.sparse = sp := by
  unfold Flock.Guarantees.c3Of at h
  cases cs with
  | none => cases h
  | some x =>
    rw [hsp] at h
    cases h
    rfl

/-- Rep `r`'s phase coins as check 5 computes them under the name `unit` (`sparseCs`), none where it refuses. -/
def csOf (c3 : RecCoins.C3) (unit : String) (r : ℕ) : Array F128 :=
  match RecCoins.sparseCoins c3 unit r with
  | .ok sc => sparseCs sc
  | .error _ => #[]

theorem csOf_eq {c3 : RecCoins.C3} {unit : String} {r : ℕ} {sc : RecCoins.SparseCoins}
    (h : RecCoins.sparseCoins c3 unit r = .ok sc) : csOf c3 unit r = sparseCs sc := by
  unfold csOf
  rw [h]

/-- The words copy 0 of unit `u` of session `s` holds at the message columns (`msgWord`), `rdOf`'s array. -/
def wSparse {S : ℕ} (Vs : Fin S → VStmt) (sh : SparseShape) (mp : List (String × ℕ)) (Lb : (j : Fin S) → (Vs j).Bits)
    (s : Fin S) (u : Fin (Vs s).nv) : Array F128 :=
  Array.ofFn fun x : Fin (nw sh) => msgWord Vs mp Lb s u x.val

theorem unitShape_holoAt {st : Flock.Stmt} {hp : HoloParams} (hell : hp.EllOk (execClassAt st).cls.kLog) :
    unitShape (holoAt st hp) = { nu := hp.nu, ell := hp.ell, sLog := hp.sLog, tLog := hp.tLog } := by
  have hs : (holoAt st hp).cls.slotLog = hp.ell := ite_eq_left hell
  unfold unitShape
  rw [hs]
  rfl

/-! ## From the session's acceptance -/

/-- **An accepted session's setup has a first statement, of the session's circuit** (`setupTables_okH`: one table's
statement or several, each `setupHidden`'s, whose circuit is the session's file's, `hcH`). -/
theorem stmt0_of_setup {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv) {record : String}
    {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf (zkInputs zo.I) record = .ok (spec, sts)) :
    ∃ h0 : 0 < sts.size, sts[0].c = c := by
  have hh : (zkTags zo.I).hiddenOutputs = true := (zkTags_hiddenOutputs zo.I).trans zo.hTags.1
  rcases HiddenExec.setupTables_okH hh hs with ⟨-, st₀, hst, -, rfl⟩ | ⟨hc1, hsz, hall, -, -⟩
  · have h1 : HmOut.parse (zkTags zo.I) zo.I.circuitFile zo.I.tables = .ok st₀.c :=
      (FlockSoundness.Discharge.Hidden.setupHidden_spec' hst).parse
    rw [zo.hcH] at h1
    exact ⟨by simp, (Except.ok.inj h1).symm⟩
  · have h0 : 0 < sts.size := by omega
    obtain ⟨d, hd⟩ := hall 0 h0
    have h1 := (FlockSoundness.Discharge.Hidden.setupHidden_spec' hd).parse
    rw [ZkHidden.parseH_identity ((zkTags_typed zo.I).trans zo.ht)] at h1
    have h2 : HmOut.parse (zkTags zo.I) zo.I.circuitFile zo.I.tables = .ok sts[0].c := h1
    rw [zo.hcH] at h2
    exact ⟨h0, (Except.ok.inj h2).symm⟩

section Family

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Where a session accepts, a transcript it accepts passes the verifier's checks at candidate 3** (`ChecksC3`, check
5 among them), at its hold of the inner transcript and check 5's source. -/
def CoinsAccepted (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (rows : Option (Array Flock.Registered.Port))
    (wr : List (Cm × (ℐ).Coin) → Lean.Json × String) (reg : Lean.Json)
    (coins : String → Option Flock.RecCoins.Source) : Prop :=
  ∀ R ω t R₂ j, (∃ tr', AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr') →
    ∃ tr', AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr' ∧
      Flock.Guarantees.ChecksC3 F.ish (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption (coins (wr t).2) j (F.bind j)
        rows (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr'

/-- **Check 5 where a session accepts, at a family whose sessions keep their own list only under `ChecksC3`**
(`hzs`, `checks_of_keepOwn`). -/
theorem coinsAccepted_of_keep (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    {coins : String → Option Flock.RecCoins.Source}
    (z : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount F.ish)) → (F.Vs j).Outer Reg')
    (hzs : ∀ R ω t R₂ j, F.zs R ω t R₂ j = Flock.Guarantees.keepOwn (z R ω t R₂ j) fun J o tr =>
      (ℐ).r ≤ t.length ∧ Flock.Guarantees.ChecksC3 F.ish (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption
        (coins (wr t).2) j (F.bind j) rows J o tr) :
    CoinsAccepted F rows wr reg coins := by
  intro R ω t R₂ j hex
  rw [hzs] at hex ⊢
  exact FlockC3.PubRows.checks_of_keepOwn (Pc := fun J o tr =>
    Flock.Guarantees.ChecksC3 F.ish (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption (coins (wr t).2) j (F.bind j)
      rows J o tr) (fun _ _ _ hP => hP.2) hex

/-- **The checked family passes check 5 where a session accepts**: a checked session keeps its own list only where a
transcript it accepts passes `ChecksC3` (`coinsAccepted_of_keep`). -/
theorem coinsAccepted_checked {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
    {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
    {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
    {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
    {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
    {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}
    {coins : String → Option Flock.RecCoins.Source} {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} :
    CoinsAccepted (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg)
      rows wr reg coins :=
  coinsAccepted_of_keep _ (fun R ω t R₂ j => Flock.Guarantees.custodied zr hCust h R ω t R₂ j) fun _ _ _ _ _ => by
    simp only [C3Family.checked, C3Family.honest, Flock.Guarantees.checkedC3, Flock.Guarantees.innerHeld]

/-- **Check 5 at the sparse session, as it ran**: where session `s` accepts and V*'s list names RecSparse at `s`, at
the inner shape's `sparse` entry `hp`'s (`SparseIs`), the session's first statement `sj` is of its circuit, which is
the verifier's at `unitShape (holoAt st hp)` (`nameChecked`, `sparseParams_of_check`), and check 5 accepted `sj` at
the session's reads file, a source `f` and the coins record `c3` the hold gives (`check_sparseCoins`,
`files_of_check`). -/
theorem sparseCheck_at (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    {coins : String → Option Flock.RecCoins.Source} (hchk : CoinsAccepted F rows wr reg coins)
    (hsp : FlockC3.VStarList.SparseIs F.ish hp) (hell : hp.EllOk (execClassAt st).cls.kLog)
    {s : Fin (vstarCount F.ish)} {sh : Flock.RecSparse.Shape} {hR : ℕ} (hs : F.stmts[(s : ℕ)]? = some (.sparse sh hR))
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂}
    (hex : ∃ tr, AcceptsZKR (F.zs R ω t R₂ s).I (F.zs R ω t R₂ s).own tr) :
    ∃ (sj : Flock.Stmt) (f : Flock.RecCoins.Source) (c3 : Flock.RecCoins.C3), sj.c = (F.Vs s).c ∧
      coins (wr t).2 = some f ∧
      Flock.Guarantees.c3Of F.ish (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption = some c3 ∧
      Flock.RecCoins.check sj (some (F.zs R ω t R₂ s).own) (some f) (some c3) = .ok () ∧
      Flock.RecSparse.params (F.Vs s).c.unit.name = some (unitShape (holoAt st hp)) := by
  obtain ⟨tr, -, spec, sts, hset, hstm, hcoin, -⟩ := hchk R ω t R₂ s hex
  obtain ⟨h0, hc0⟩ := stmt0_of_setup (F.zs R ω t R₂ s) hset
  have hck0 := FlockSoundness.Discharge.ZkExec.array_forM_ok hstm 0 h0
  have hco0 := FlockSoundness.Discharge.ZkExec.array_forM_ok hcoin 0 h0
  obtain ⟨s', hs', hn'⟩ := FlockC3.PubRows.nameChecked F s sts[0] _ _ _ _ _ _ _ hc0 hck0
  rw [hs] at hs'
  cases hs'
  have hname : (F.Vs s).c.unit.name = Flock.RecSparse.unitName sh := hn'.symm
  have hpar : Flock.RecSparse.params sts[0].c.unit.name = some sh := by
    rw [hc0]
    exact sparseParams_of_check (parse_facts (F.zs R ω t R₂ s).hc).recSparse hname
  have hish : F.ish.sparse = some sh := FlockC3.PubRows.sparse_of_checkStmt hck0 hpar
  have hsh : sh = unitShape (holoAt st hp) := by
    rw [unitShape_holoAt hell]
    exact Option.some.inj (hish.symm.trans hsp)
  subst hsh
  have hso : Flock.RecCoins.stackedParams sts[0].c.unit.name = none := by
    rw [hc0, hname]; exact stackedParams_sparse _
  have hop : Flock.RecOpening.params sts[0].c.unit.name = none := by
    rw [hc0, hname]; exact openingParams_sparse _
  obtain ⟨c3, hc3, -⟩ := check_sparseCoins hco0 hpar hso hop
  obtain ⟨f, hf⟩ := files_of_check hco0 hpar hso hop
  rw [hc3, hf] at hco0
  exact ⟨sts[0], f, c3, hc0, hf, hc3, hco0, by rw [← hc0]; exact hpar⟩

/-- **Check 5 at the sparse session**: where session `s` accepts and V*'s list names RecSparse at `s`, at the inner
shape's `sparse` entry `hp`'s (`SparseIs`), its circuit is named `unitName (unitShape (holoAt st hp))`
(`nameChecked`), and check 5 there holds the coins record and computes both reps' coins (`check_sparseCoins`). -/
theorem sparseCoins_of_accepted (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    {coins : String → Option Flock.RecCoins.Source} (hchk : CoinsAccepted F rows wr reg coins)
    (hsp : FlockC3.VStarList.SparseIs F.ish hp) (hell : hp.EllOk (execClassAt st).cls.kLog)
    {s : Fin (vstarCount F.ish)} {sh : Flock.RecSparse.Shape} {hR : ℕ} (hs : F.stmts[(s : ℕ)]? = some (.sparse sh hR))
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂}
    (hex : ∃ tr, AcceptsZKR (F.zs R ω t R₂ s).I (F.zs R ω t R₂ s).own tr) :
    ∃ c3, Flock.Guarantees.c3Of F.ish (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption = some c3 ∧
      (∀ r < 2, ∃ sc, Flock.RecCoins.sparseCoins c3 (Flock.RecSparse.unitName (unitShape (holoAt st hp))) r = .ok sc) ∧
      (F.Vs s).c.unit.name = Flock.RecSparse.unitName (unitShape (holoAt st hp)) := by
  obtain ⟨sj, f, c3, hc, -, hc3, hck, hpar⟩ := sparseCheck_at F hchk hsp hell hs hex
  have hname : (F.Vs s).c.unit.name = Flock.RecSparse.unitName (unitShape (holoAt st hp)) :=
    (FlockVBridge.Sparse.unitName_params hpar).symm
  obtain ⟨c3', hc3', hco⟩ := check_sparseCoins hck (by rw [hc]; exact hpar)
    (by rw [hc, hname]; exact stackedParams_sparse _) (by rw [hc, hname]; exact openingParams_sparse _)
  cases hc3'
  rw [hc, hname] at hco
  exact ⟨c3, hc3, hco, hname⟩

/-- **(n7) Rep `i`'s sparse phase in V*'s sessions, from the sparse session's acceptance** (`SparseSessionAt`): at an
accepted outer proof, at session `s`, which V*'s list names RecSparse, and copy 0 of its unit `u` (`u.val = i.val`,
which the second conjunct carries as the copy's pin), at a coins record `C` every hold's `c3Of` is (`hC`), the words
copy 0 holds at the message columns (`wSparse`) and coins `c` whose phase coins are check 5's (`csOf`). The coins are
check 5's (`sparseCoins_of_accepted`, at `C` by `c3_eq_of_held`) at the inner shape's RecSparse shape
(`c3Of_sparse`); the unit is the verifier's at that shape (`ParseFacts.recSparse`, `sparseParams_of_check`); its message
columns a row reads register `wSparse`'s bits where they register any (`hkeys`, `bitAt_of_regBit` in the session's
layer, which agrees with the registrant, and `nw_eq_msum`). The constant 1 (`hone`), `v`'s registered verifier row
(`hv`), `tau`'s registered slots' types (`htau`) and the registered zero outputs (`hz`) are taken as given. -/
theorem sparseSessionAt_of_regs (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    {coins : String → Option Flock.RecCoins.Source} (hchk : CoinsAccepted F rows wr reg coins)
    (hsp : FlockC3.VStarList.SparseIs F.ish hp) (hell : hp.EllOk (execClassAt st).cls.kLog)
    {s : Fin (vstarCount F.ish)} {sh : Flock.RecSparse.Shape} {hR : ℕ} (hs : F.stmts[(s : ℕ)]? = some (.sparse sh hR))
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂}
    {τ : (j : Fin (vstarCount F.ish)) → Strategy ((F.zs R ω t R₂ j).game dg N)}
    {o : (j : Fin (vstarCount F.ish)) → Reg' × Finset (Fin (F.Vs j).nv) × Exec.Transcript}
    (hA : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (i : Fin 2) (u : Fin (F.Vs s).nv) (hu : u.val = i.val)
    {C : Flock.RecCoins.C3}
    (hC : ∀ cs, innerCommitsAt (wr t).1 (wr t).2 st reg = .ok cs → Flock.Guarantees.c3Of F.ish (some cs) = some C)
    {c : SCoins} (hcs : c.cs = csOf C (Flock.RecSparse.unitName (unitShape (holoAt st hp))) i.val)
    (hone : layers F.zs dg N k Rw R ω t R₂ τ s
      (Const.Prog.oneAt (Placed.keyProg (F.Vs s).c (F.Vs s).hU (F.Vs s).nv) u) = true)
    (hkeys : ∀ q < 128 * ((Flock.RecSparse.messagePorts (unitShape (holoAt st hp))).map (·.2)).sum,
      ReadCol (Rows.ofNet (F.Vs s).hU) (rowCol (Flock.RecSparse.messagePorts (unitShape (holoAt st hp))) 0 q) →
        ∃ b, RegBit (F.zs R ω t R₂ s) u 0 (rowCol (Flock.RecSparse.messagePorts (unitShape (holoAt st hp))) 0 q) b)
    (hv : ∀ q < 128 * Flock.RecSparse.nv (unitShape (holoAt st hp)),
      RegBit (F.zs R ω t R₂ s) u 0 (vBaseS (unitShape (holoAt st hp)) + q)
        (wbit ((verifierRow (holoAt st hp).cls.shape c)[·]!) q))
    (htau : (unitShape (holoAt st hp)).stacked = true → ∀ x < (unitShape (holoAt st hp)).S, ∀ b < 16,
      RegBit (F.zs R ω t R₂ s) u 0 (tauBase (unitShape (holoAt st hp)) + (16 * x + b))
        (((tauArr (holoAt st hp))[x]!).testBit b))
    (hz : ∀ x < 128 * (Flock.RecSparse.residualNames (unitShape (holoAt st hp))).length,
      RegOut (F.zs R ω t R₂ s) u (FlockC3.Reps.upv_of_outer (F.zs R ω t R₂ s))
        (sparseBase (unitShape (holoAt st hp)) + x) false) :
    SparseSessionAt (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) (holoAt st hp) C
        (Flock.RecSparse.unitName (unitShape (holoAt st hp))) i
        (wSparse F.Vs (holoAt st hp).cls.shape (Flock.RecSparse.messagePorts (unitShape (holoAt st hp)))
          (layers F.zs dg N k Rw R ω t R₂ τ) s u) c ∧
      ∃ (s' : Fin (vstarCount F.ish)) (un : Fin (F.Vs s').nv) (j : ℕ) (hj : j < (F.Vs s').c.upv), un.val = i.val ∧
        SparseCopyAt (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
          (wSparse F.Vs (holoAt st hp).cls.shape (Flock.RecSparse.messagePorts (unitShape (holoAt st hp)))
            (layers F.zs dg N k Rw R ω t R₂ τ) s u) c s' un j hj := by
  obtain ⟨c3, hc3, hco, hname⟩ := sparseCoins_of_accepted F hchk hsp hell hs ⟨(o s).2.2, (hA.1 s).1⟩
  have hck := (parse_facts (F.zs R ω t R₂ s).hc).recSparse
  have hpar := sparseParams_of_check hck hname
  obtain rfl := c3_eq_of_held hc3 hC
  obtain ⟨sc, hsc⟩ := hco i.val i.isLt
  have hcopy : SparseCopyAt (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
      (wSparse F.Vs (holoAt st hp).cls.shape (Flock.RecSparse.messagePorts (unitShape (holoAt st hp)))
        (layers F.zs dg N k Rw R ω t R₂ τ) s u) c s u 0 (FlockC3.Reps.upv_of_outer (F.zs R ω t R₂ s)) := by
    refine ⟨hck, hpar, hone, FlockVBridge.Family.unitLog_le (F.Vs s).hscope, ?_, hv, htau, hz⟩
    intro q hq hrd
    obtain ⟨b, hb⟩ := hkeys q hq hrd
    obtain ⟨-, -, -, hn⟩ := FlockVBridge.Sparse.check_sparse hck hpar
    have hn' : Flock.HmNets.netOf (Flock.RecSparse.unitName (unitShape (holoAt st hp)))
        (Flock.RecSparse.unitLaid (unitShape (holoAt st hp))) = .ok (F.Vs s).c.unit := hn
    have hlt := FlockVBridge.Sparse.rowCol_lt_nInS (F.Vs s).hU hn' hq
    have hbit := bitAt_of_regBit (F.zs R ω t R₂ s) (bits := layers F.zs dg N k Rw R ω t R₂ τ s)
      (hA.1 s).2.2.1 hb
      (ioColAt_of_readCol u hlt hrd) (FlockC3.Reps.upv_of_outer (F.zs R ω t R₂ s))
    have hm : nw (holoAt st hp).cls.shape =
        ((Flock.RecSparse.messagePorts (unitShape (holoAt st hp))).map (·.2)).sum :=
      nw_eq_msum (unitShape (holoAt st hp))
    have hq' : q / 128 < nw (holoAt st hp).cls.shape := by omega
    have hw : wbit (fun e => (wSparse F.Vs (holoAt st hp).cls.shape
        (Flock.RecSparse.messagePorts (unitShape (holoAt st hp))) (layers F.zs dg N k Rw R ω t R₂ τ) s u)[e]!) q =
        b := by
      unfold FlockVBridge.wbit wSparse
      beta_reduce
      rw [getElem!_rd hq', bitOf_msgWord (Nat.mod_lt _ (by omega)), Nat.div_add_mod]
      exact hbit
    rw [hw]
    exact hb
  exact ⟨⟨⟨sc, hsc, by rw [c3Of_sparse hc3 hsp, unitShape_holoAt hell], by rw [hcs]; exact csOf_eq hsc⟩,
    Array.size_ofFn, s, u, 0, _, hcopy⟩, s, u, 0, _, hu, hcopy⟩

end Family

end

end FlockC3.Sparse
