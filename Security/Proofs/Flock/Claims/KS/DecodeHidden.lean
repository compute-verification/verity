import Proofs.Flock.Claims.KS.DecodeDeploy
import Proofs.Flock.Soundness.Discharge.HiddenDec

/-!
# The deployed plain decoders at hidden outputs

`DecodeSC` and `DecodeDeploy` where the statement hides its outputs (`hiddenOutputs = true`), at the hidden tables
`HiddenExec.tablesAtH` and their openings `HiddenDec.opnAtH`:

* `decodesTable_wrP_scH`: `decodesTable_wrP_sc` at the hidden tables, from `accepts_repsH` and `tableAtH_of_acceptsI`;
  `accepts_opens₀L` holds at hidden outputs too.
* `plain_setupHiddenT`, `plain_setupHidden`: a plain statement in scope with hidden outputs sets its one table up at its
  draw under setupHidden, with `HiddenDec.TagsOkH`'s tags at any leaf tag and at `hm96-sha512/v1` leaves.

None of them reads which net returns the outputs: `InScope`'s general instance form (hidden outputs and `wiresAcyclic`)
may have tail stages, which the verifier evaluates and the table's statement (`stmtOf` of setupHidden's) carries, and
the decoders read only the table's committed rows.
* `tableAtH_plainTab`: the hidden table of a draw that sets up `st` is `plainTab` at `st`, its statement read off any
  layout (`Integrate.model_eq`).
* `plain_hdecH`: so at the draw `d` and no drawn unit, `decodesTable_wrP_scH` is the decoders' fact `PlainLiveAccAt`
  asks, at `plainTab`.
-/

namespace Claims.Bridge.DecodePlain

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

open DecodeZL HiddenExec HiddenDec in
/-- **The deployed plain decoders are right at `wrP` for accepting with no opening off `C` without a self-clash, at
hidden outputs**: `decodesTable_wrP_sc` at the hidden tables and their openings `opnAtH`, wherever the outputs are
returned (`tableAtH_of_acceptsI`). -/
theorem decodesTable_wrP_scH {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (j : Fin [tablesAtH (Reg := Reg) dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 leaf512 (tablesAtH (Reg := Reg) dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 leaf512 (tablesAtH (Reg := Reg) dj x₀ S R).sch o))
      (fun o => Exec.Accepts I (wrP I dj (fun _ _ => 1) S R o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOff I C (wrP I dj (fun _ _ => 1) S R o))
      (fun ps _ => opnAtH (tablesAtH dj x₀) I dj S R ps)
      (fun _ => decAt (tablesAtH dj x₀) wOP wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([tablesAtH (Reg := Reg) dj x₀ S R].get j)) j := by
  have hj : j.val = 0 := by
    have h := j.isLt
    simp only [List.length_cons, List.length_nil] at h
    omega
  obtain rfl : j = ⟨0, Nat.one_pos⟩ := Fin.ext hj
  rintro o ⟨hacc, hr₀, hno⟩
  rw [← wrP1_oneOf] at hacc hno
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  have h := PlaysCoin.decodesOfRuns (fun cap pend os => tableOutL H512 enc512 leaf512 cap pend os)
    (fun o' => o' = oneOf o)
    (fun out => out.accepted = true ∧ ¬ (Off₀L H512 enc512 leaf512 (tablesAtH (Reg := Reg) dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) out ∧
        ¬ SelfClashL H512 enc512 leaf512 (tablesAtH (Reg := Reg) dj x₀ S R).sch out))
    (opnAtH (tablesAtH dj x₀) I dj S R) (decAt (tablesAtH dj x₀) wOP wLIG S R) (fun _ _ => []) capOfRoot
    (fun pts r => repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) pts r)
    (fun pts _ => PlaysCoin.repSpine Dig (tablesAtH (Reg := Reg) dj x₀ S R) pts)
    (PlaysCoin.repsOf_spine Dig (tablesAtH (Reg := Reg) dj x₀ S R)) ?_ (oneOf o) rfl
  · obtain ⟨pend, h0, h1, hok⟩ := h
    exact ⟨pend, h0, h1, hok⟩
  rintro o' rfl
  obtain ⟨hd, hprf, hrec⟩ := recordsLive_wrP1 (Reg := Reg) (mPts := mPts) I dj S R (oneOf o) hacc
  obtain ⟨hSD, hmPts⟩ := hS
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAtH_of_acceptsI ht hc hh hc1 dj x₀ hacc hd hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ := accepts_repsH hh hrr (Or.inl hl) hc1 hacc
  rw [hd] at hst
  obtain rfl : st = st₁ := Except.ok.inj (hst.symm.trans hst₁)
  obtain ⟨hroot, hpts, hstreams⟩ := hrec spec #[st] sess hsu hdec
  have hmP : st.mPts = mPts := hmPts st hst
  obtain ⟨-, hr⟩ := setupHidden_wf hst
  have hm : 13 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hms : st.merkle = MerkleScheme.hm96Sha512 := setupHidden_hm96 hl hst
  obtain ⟨pend, hreps, hv⟩ := hk hwf hr hm
  rw [hl, leafOfId_hm96] at hv
  have hn : ∀ (q : ℕ) (pts : Fin 2 → Fin q → 𝔽),
      (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL q pts).length = 2 * st.regions.size := by
    intro q pts
    rw [DecodeZL.extraClaims_length]
    simp [stmtOf, regionsOf]
  obtain ⟨s₀, p₀, -, hs₀, hp₀, hpl₀, -⟩ := hreps 0
  obtain ⟨s₁, p₁, -, hs₁, hp₁, hpl₁, -⟩ := hreps 1
  have hcap : sess.rootB[0]! = rootB₀[0]! := hroot.trans hr₀
  have hv' := hv
  rw [hn, hcap] at hv'
  obtain ⟨spec', sts', sess', hsu', hdec', hall⟩ := accepts_opens₀L hacc
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  rw [hdec, Except.ok.injEq] at hdec'
  subst hdec'
  obtain ⟨sch', hf', hk'⟩ := hall 0 (by simp)
  obtain rfl : sch = sch' := Option.some.inj (hf.symm.trans hf')
  have hn' : ∀ (q : ℕ) (pts : Fin 2 → Fin q → 𝔽),
      (Model.extraClaims (stmtOf #[st][0] hwf (regionsOf #[st][0])) PT_LOCAL q pts).length = 2 * st.regions.size :=
    hn
  obtain ⟨hver, hent⟩ := hk' hwf hr hm p₀ p₁ s₀ s₁ (pend 0) (pend 1) (capOfRoot rootB₀[0]!)
    (tableOutL H512 enc512 leaf512 (capOfRoot rootB₀[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrP1 I dj S R (oneOf o)).proofs[0]!, (wrP1 I dj S R (oneOf o)).proofs[1]!]))
    hs₀ hs₁ hp₀ hp₁ hpl₀ hpl₁ (by rw [hn']; rfl) hv'
  rw [hn'] at hent
  rw [hmP, hpts] at hreps
  have hS' : (tablesAtH (Reg := Reg) dj x₀ S R).S = stmtOf st hwf (regionsOf st) := hTS
  have hpl : (tablesAtH (Reg := Reg) dj x₀ S R).ptLocal = PT_LOCAL := tableAtH_ptLocal dj x₀ _
  have hsch : (tablesAtH (Reg := Reg) dj x₀ S R).sch = schedOf sch := by
    have h1 : Accounting.fast100 st.m = some (schedOf sch) := by rw [← fast100_schedOf, hf]; rfl
    have h2 := (tablesAtH (Reg := Reg) dj x₀ S R).hsch
    rw [hS'] at h2
    exact Option.some.inj (h2.symm.trans h1)
  have hrep : ∀ r : Fin 2, PlaysCoin.RepRun (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R)
      (oneOf o).pts r) (PlaysCoin.repSpine Dig (tablesAtH (Reg := Reg) dj x₀ S R) (oneOf o).pts)
      (decAt (tablesAtH dj x₀) wOP wLIG S R) (pick ((oneOf o).rep0, (oneOf o).rep1) r) [] (pend r) := by
    intro r
    obtain ⟨s, p, Tv, hs, hp, hplay, hlen, hret⟩ := hreps r
    obtain ⟨s', hs', hrs⟩ := hstreams r
    obtain rfl : s' = s := Option.some.inj (hs'.symm.trans hs)
    rw [← hrs]
    exact repRun_at op_carries lig_carries decodeProof_sizes _ st hwf sch hf (by rw [hms]; rfl) hS' hsch hpl
      (oneOf o).pts r hm rfl _ p hp s'.rounds (PlaysCoin.roundsWF_mem (decode_roundsWF hdec s'
        (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  refine ⟨(pend 0, pend 1), hrep 0, hrep 1, ?_⟩
  have hopn : opnAtH (tablesAtH (Reg := Reg) dj x₀) I dj S R (oneOf o).proofs =
      opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrP1 I dj S R (oneOf o)).proofs[0]!, (wrP1 I dj S R (oneOf o)).proofs[1]!] := by
    rw [← hprf]
    unfold opnAtH
    rw [hst]
    dsimp only
    rw [hf]
    dsimp only
    congr 1
    unfold nX
    rw [DecodeZL.extraClaims_length, hS']
    simp [stmtOf, regionsOf]
  have e₀ : (oneOf o).rootB = rootB₀[0]! := hr₀
  rw [hopn, e₀, hsch]
  refine ⟨hv', fun ⟨hoff, hsc⟩ => hno spec st sess sch s₀ s₁ hsu hdec hf hs₀ hs₁ ?_⟩
  rw [hsch] at hN
  exact Claims.Flock.exists_ne_of_entries H512 enc512 leaf512 (schedOf sch) (capOfRoot rootB₀[0]!) _ hN C _ _ rfl
    hver hent hoff hsc

end Claims.Bridge.DecodePlain

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A statement whose form is not templated is on flat tags**, once its session commits a circuit: a templated
session's is `HmOut.parseTmpl`'s, which carries its template, and the check parses that circuit (`Read.circ`). -/
theorem isTmpl_of_form {r : Readable.Read} {c : Circuit} (hh : r.tags.hiddenOutputs = true)
    (hc : HmOut.parseCommit r.tags r.I.circuitFile r.I.tables = .ok c) (htm : r.form.templated = false) :
    r.tags.isTmpl = false := by
  cases hi : r.tags.isTmpl
  · rfl
  · have hcirc : r.circ = c := by
      unfold Readable.Read.circ
      rw [ProvedScope.circuitOf_hidden hh, hc]
    rw [parseCommit_tmpl hi] at hc
    obtain ⟨tu, c₀, -, -, -, rfl⟩ := TmplValue.parseTmpl_cases hc
    simp [Readable.Read.form, hcirc] at htm

/-- **A plain statement in scope with hidden outputs sets up its one table at its draw, under any leaf tag**:
`plain_setupH` with setupHidden for setupH and `HmOut.parse` for `HmRow.parse`, in the general instance form (any input
groups, any tail stages the verifier evaluates, whichever net returns the outputs), with `TagsOkH`'s tags at the tag
`merkleLeaf` names. -/
theorem plain_setupHiddenT {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) {tag : String} (hl : ins.read.I.tags.merkleLeaf = tag) {d : Json}
    (hd : ins.read.draw = some d) :
    ins.read.I.tags.typed = false ∧ HmOut.parse ins.read.I.tags ins.read.I.circuitFile ins.read.I.tables = .ok st.c ∧
      (ins.read.I.tags.hiddenOutputs = true ∧ ins.read.I.tags.retainRounds = true ∧ ins.read.I.tags.merkleLeaf = tag ∧
        ins.read.I.count ≤ 1) ∧
      Stmt.setupHidden ins.read.I.tags ins.read.I.circuitFile ins.read.I.publicFile ins.read.I.coins
        ins.read.I.tables (some d) ins.read.I.partition ins.read.I.program = .ok st ∧ spec = st.spec ∧ 22 ≤ st.m ∧
      st.m ≤ 35 := by
  have hm₁ := hin.1
  have hm₂ := hin.2.1
  obtain ⟨p, hp, rfl, -, -, -⟩ := parsed_of_parse hs
  obtain ⟨-, hpr, hsr⟩ := parseBytes_setup hp hset
  have hzk : ins.read.zk = false := readOf_zk (x := x) (by unfold readOf; rw [(setupOf_some hset).1]; rfl)
  have htags : ins.read.tags = ins.read.I.tags := by simp [Readable.Read.tags, hzk]
  have hform : (Readable.toReadable Readable.relation p).form = ins.read.form := by rw [← hpr]; rfl
  obtain ⟨-, -, -, -, -, ht, hrr, -, hc1, hpi, -, -, -, -, -, htm⟩ := hin
  rw [hform] at ht hrr hc1 hpi htm
  have htm' : ins.read.form.templated = false := by
    cases e : ins.read.form.templated
    · rfl
    · exact absurd (htm e).1 (by decide)
  simp only [Readable.Read.form, htags] at ht hrr hc1 hpi
  simp only [Bool.not_eq_true] at ht
  simp only [Bool.false_eq_true, ↓reduceIte] at hc1
  have hpi' : ins.read.publicInputs = none := Option.not_isSome_iff_eq_none.1 hpi
  have hcoins : ins.read.coins = ins.read.I.coins := by simp [Readable.Read.coins, hzk]
  unfold Readable.Read.setup at hsr
  rw [htags, hcoins, hpi', hd] at hsr
  rcases HiddenExec.setupTables_okH hh hsr with ⟨-, st', hst, hsp, rfl⟩ | ⟨hc, -⟩
  swap
  · omega
  have e : st' = st := by
    unfold tables at h
    rw [hset] at h
    simpa using h
  subst e
  have hi : ins.read.I.tags.isTmpl = false := by
    rw [← htags]
    exact isTmpl_of_form (by rw [htags]; exact hh) (by rw [htags]; exact Hidden.setupHidden_parseCommit hst) htm'
  have hc := HiddenExec.setupHidden_parseC hi hst
  have hm := parse_m hs hset (by simp)
  simp only [Bool.false_eq_true, ↓reduceIte] at hm₁
  refine ⟨ht, hc, ⟨hh, hrr, hl, hc1⟩, hst, hsp, ?_, ?_⟩
  · simpa [hm] using hm₁
  · simpa [hm] using hm₂

/-- **A plain statement in scope with hidden outputs and `hm96-sha512/v1` leaves sets up its one table at its draw**:
`plain_setupH` with setupHidden for setupH and `HmOut.parse` for `HmRow.parse`, in the general instance form (any input
groups, any tail stages the verifier evaluates, whichever net returns the outputs). The scope also admits plain SHA-512
leaves here, which `TagsOkH` does not: `plain_setupHiddenT` at `hm96-sha512/v1`. -/
theorem plain_setupHidden {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) :
    ins.read.I.tags.typed = false ∧ HmOut.parse ins.read.I.tags ins.read.I.circuitFile ins.read.I.tables = .ok st.c ∧
      HiddenDec.TagsOkH ins.read.I ∧
      Stmt.setupHidden ins.read.I.tags ins.read.I.circuitFile ins.read.I.publicFile ins.read.I.coins
        ins.read.I.tables (some d) ins.read.I.partition ins.read.I.program = .ok st ∧ spec = st.spec ∧ 22 ≤ st.m ∧
      st.m ≤ 35 :=
  plain_setupHiddenT hs hin hset h hh hl hd

/-- **The hidden table of a draw that sets up `st` is `plainTab` at `st`**: its statement `stmtOf st` is the one any
layout reads (`Integrate.model_eq`), and the schedule is `fast100`'s at `st.m`. -/
theorem tableAtH_plainTab {I : Exec.Inputs} (ht : I.tags.typed = false) (hi : I.tags.isTmpl = false) {n : ℕ}
    (dj : Finset (Fin n) → Json)
    {spec : SessionSpec} {S₀ : Finset (Fin n)} (x₀ : HiddenExec.DrawSetupH I spec.mPts S₀ (dj S₀))
    {S : Finset (Fin n)} (x : HiddenExec.DrawSetupH I spec.mPts S (dj S)) {st : Stmt} (hx : x.st = st)
    (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) :
    (HiddenExec.tableAtH dj x₀ S).T = plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay := by
  obtain ⟨y, hy, e⟩ := HiddenExec.tableAtH_stmt dj x₀ x
  rw [e]
  rw [hx] at hy
  obtain ⟨yst, yh, D, hD, hS, hg, hblk, ym, yk0, yl₀, yl₁, yrest, yext, yhsch, yhm, yhmPts, ywf⟩ := y
  dsimp only at hy
  subst hy
  have hs := Option.some.inj (yhsch.symm.trans hsch)
  simp only [Accounting.Schedule.mk.injEq, List.cons.injEq] at hs
  obtain ⟨rfl, rfl, ⟨rfl, rfl, rfl⟩, rfl⟩ := hs
  have hS' : L.model (Refine.regionsOf yst) = Refine.stmtOf yst ywf.1 (Refine.regionsOf yst) :=
    Integrate.model_eq (parse_facts (Hidden.parse_untyped ht (HiddenExec.setupHidden_parseC hi yh)).1) L
      (Hidden.setupHidden_spec yh).leaf ywf.1 _
  unfold plainTab
  simp only [HiddenExec.DrawSetupH.table, HiddenExec.DrawSetupH.specAt]
  congr 1
  exact hS'.symm

open Classical Discharge.Tables Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain in
/-- **The deployed plain decoders are right at `plainTab` at hidden outputs and `hm96-sha512/v1` leaves**: `plain_hdec`
with `decodesTable_wrP_scH` and the hidden openings `HiddenDec.opnAtH`. -/
theorem plain_hdecH {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) :
    ∀ j, DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (fun o => Exec.Accepts ins.read.I (wrP ins.read.I (fun _ : Finset (Fin 0) => d) (fun _ _ => 1) ∅ () o) ∧
        o.rootB[0]! = rootB₀[0]! ∧
        TxNoOff ins.read.I C (wrP ins.read.I (fun _ : Finset (Fin 0) => d) (fun _ _ => 1) ∅ () o))
      (fun ps _ => HiddenDec.opnAtH (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j := by
  obtain ⟨ht, hc, hT, hst, rfl, h22, h35⟩ := plain_setupHidden hs hin hset h hh hl hd
  obtain ⟨sch, hf⟩ : ∃ sch, Flock.fast100 st.m = some sch := by
    have e := Refine.fast100_schedOf st.m
    rw [hsch] at e
    cases hf : Flock.fast100 st.m with
    | none => rw [hf] at e; cases e
    | some sch => exact ⟨sch, rfl⟩
  obtain ⟨pop, -, hdr⟩ := HiddenExec.setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { st.c with outPorts := #[] }) hdr
  obtain ⟨x₀, hx₀⟩ := HiddenExec.drawSetupH_of_setupHiddenI (S := (∅ : Finset (Fin 0))) (mPts := st.spec.mPts) ht hc
    hst hf hD (fun u hu => absurd hu (Finset.notMem_empty u)) rfl
  have e : HiddenExec.tablesAtH (Reg := Unit) (fun _ : Finset (Fin 0) => d) x₀ =
      fun _ _ => plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay := by
    funext S R
    obtain rfl : S = ∅ := Finset.eq_empty_of_isEmpty S
    exact tableAtH_plainTab ht (HmOut.parse_isTmpl hc) (fun _ : Finset (Fin 0) => d) (spec := st.spec) (S := ∅) x₀ x₀
      hx₀ L m k0 l₀ l₁
      rest ext hsch hm hlay
  have hS : HiddenDec.DrawOkH ins.read.I (fun _ : Finset (Fin 0) => d) st.spec.mPts ∅ :=
    ⟨fun _ _ u hu => absurd hu (Finset.notMem_empty u), fun st' hst' => by
      obtain rfl := Except.ok.inj (hst.symm.trans hst')
      rfl⟩
  have hN : wordLen false x =
      2 ^ (lvOf (HiddenExec.tablesAtH (Reg := Unit) (fun _ : Finset (Fin 0) => d) x₀ ∅ ()).sch 0).logLen := by
    rw [e]
    exact plain_wordLen h h22 h35 hsch
  have hdec := decodesTable_wrP_scH ht hc (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () hN C rootB₀
  rw [e] at hdec
  exact hdec

end FlockDeploy.KS
