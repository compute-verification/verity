import Proofs.Flock.Claims.KS.DecodeSC
import Proofs.Flock.Claims.KS.PlainAcc

/-!
# The deployed plain decoders at the plain table

`PlainLiveAccAt` asks for decoders right at `plainTab`, table 0's model statement. `decodesTable_wrP_sc` gives them at
`Integrate.tablesAt`, the table of the draw a session's inputs set up. At a plain statement in scope with public
outputs whose draw is `d`, the two are one table:

* `plain_setupH`: the statement's inputs are untyped, have the tags `DecodeZL.TagsOk` asks, parse its one table's
  circuit, which returns the outputs, and set that table up at `d`, with `22 ≤ m ≤ 35`.
* `tableAt_plainTab`: the table of a draw that sets up `st` is `plainTab` at `st`, whatever its layout.
* `plain_wordLen`: the plain codeword length is `2^logLen` at level 0 of the table's schedule.
* `plain_hdec`: so at the draw `d` and no drawn unit, `decodesTable_wrP_sc` is the decoders' fact `PlainLiveAccAt`
  asks, at `plainTab`.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A plain statement in scope with public outputs sets up its one table at its draw**: its inputs are untyped and
have the tags `DecodeZL.TagsOk` asks, they parse the table's circuit, which returns the outputs, setupH sets the
table up at the draw `d`, the session's spec is the table's, and `22 ≤ m ≤ 35`. -/
theorem plain_setupH {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = false) {d : Json} (hd : ins.read.draw = some d) :
    ins.read.I.tags.typed = false ∧ HmRow.parse ins.read.I.tags ins.read.I.circuitFile ins.read.I.tables = .ok st.c ∧
      st.c.outNet = st.c.unitNet ∧ DecodeZL.TagsOk ins.read.I ∧
      Stmt.setupH ins.read.I.tags ins.read.I.circuitFile ins.read.I.publicFile ins.read.I.coins ins.read.I.tables
        (some d) ins.read.I.partition ins.read.I.program = .ok st ∧ spec = st.spec ∧ 22 ≤ st.m ∧ st.m ≤ 35 := by
  have hm₁ := hin.1
  have hm₂ := hin.2.1
  obtain ⟨p, hp, rfl, -, -, -⟩ := parsed_of_parse hs
  obtain ⟨-, hpr, hsr⟩ := parseBytes_setup hp hset
  have hzk : ins.read.zk = false := readOf_zk (x := x) (by unfold readOf; rw [(setupOf_some hset).1]; rfl)
  have htags : ins.read.tags = ins.read.I.tags := by simp [Readable.Read.tags, hzk]
  have hform : (Readable.toReadable Readable.relation p).form = ins.read.form := by rw [← hpr]; rfl
  obtain ⟨-, -, -, -, -, ht, hrr, hl, hc1, hpi, -, hg, -⟩ := hin
  rw [hform] at ht hrr hl hc1 hpi hg
  simp only [Readable.Read.form, htags] at ht hrr hl hc1 hpi hg
  simp only [hh, Bool.false_eq_true, false_and, or_false] at hg
  obtain ⟨-, ho⟩ := hg
  simp only [Bool.not_eq_true] at ht
  simp only [Bool.false_eq_true, ↓reduceIte] at hc1
  have hpi' : ins.read.publicInputs = none := Option.not_isSome_iff_eq_none.1 hpi
  have hcoins : ins.read.coins = ins.read.I.coins := by simp [Readable.Read.coins, hzk]
  unfold Readable.Read.setup at hsr
  rw [htags, hcoins, hpi', hd] at hsr
  rcases setupTables_ok hh hsr with ⟨-, st', hst, hsp, rfl⟩ | ⟨hc, -⟩
  swap
  · omega
  have e : st' = st := by
    unfold tables at h
    rw [hset] at h
    simpa using h
  subst e
  have hc := (setupH_spec ht hst).1
  have hcirc : ins.read.circ = st'.c := by
    simp only [Readable.Read.circ, ProvedScope.circuitOf, htags, hh, Bool.false_eq_true, ↓reduceIte, hc]
  have hstmt : ins.read.stmt = st'.c := by
    rw [Readable.Read.stmt, hcirc]
    exact FlockSoundness.unrolled_flat (FlockSoundness.parse_facts hc).tmpl
  rw [hstmt] at ho
  have hm := parse_m hs hset (by simp)
  simp only [Bool.false_eq_true, ↓reduceIte] at hm₁
  have hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1" := by
    rcases hl with hl | ⟨-, hho, -⟩
    · exact hl
    · simp [hh] at hho
  refine ⟨ht, hc, by simpa using ho, ⟨hh, hrr, hl, hc1⟩, hst, hsp, ?_, ?_⟩
  · simpa [hm] using hm₁
  · simpa [hm] using hm₂

/-- **The table of a draw that sets up `st` is `plainTab` at `st`**: the layout is a proposition, and the schedule is
`fast100`'s at `st.m`. -/
theorem tableAt_plainTab {I : Exec.Inputs} (ht : I.tags.typed = false) {n : ℕ} (dj : Finset (Fin n) → Json)
    {spec : SessionSpec} {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I spec.mPts S₀ (dj S₀))
    {S : Finset (Fin n)} (x : Integrate.DrawSetup I spec.mPts S (dj S)) {st : Stmt} (hx : x.st = st) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) :
    (Integrate.tableAt ht dj x₀ S).T = plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay := by
  obtain ⟨y, hy, e⟩ := Integrate.tableAt_stmt ht dj x₀ x
  rw [e]
  rw [hx] at hy
  obtain ⟨yst, yh, D, hD, hS, hg, hblk, ym, yk0, yl₀, yl₁, yrest, yext, yhsch, yhm, yhmPts⟩ := y
  dsimp only at hy
  subst hy
  have hs := Option.some.inj (yhsch.symm.trans hsch)
  simp only [Accounting.Schedule.mk.injEq, List.cons.injEq] at hs
  obtain ⟨rfl, rfl, ⟨rfl, rfl, rfl⟩, rfl⟩ := hs
  rfl

/-- **The plain codeword length is `2^logLen` at level 0**, at the one table's `fast100` schedule. -/
theorem plain_wordLen {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)])
    (h22 : 22 ≤ st.m) (h35 : st.m ≤ 35) {m k0 : ℕ} {l₀ l₁ : Accounting.Level} {rest : List Accounting.Level}
    {ext : ℕ} (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) :
    wordLen false x = 2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0).logLen := by
  obtain ⟨i, hi⟩ : ∃ i : Claims.Flock.PlainIdx, i.m = st.m :=
    ⟨⟨st.m - 22, by omega⟩, by simp only [Claims.Flock.PlainIdx.m]; omega⟩
  unfold wordLen
  simp only [h, List.head?_cons, Bool.false_eq_true, ↓reduceIte]
  rw [← hi] at hsch ⊢
  have e := Option.some.inj ((Claims.Flock.fast100_plainSched i).symm.trans hsch)
  have hl := Claims.Flock.level₀_plainSched i
  rw [e] at hl
  have hl' : l₀ = Refine.levelOf (Claims.Flock.plainLevel0 i.m) := hl
  show 2 ^ (Claims.Flock.plainLevel0 i.m).d = 2 ^ l₀.logLen
  rw [hl', Claims.Flock.logLen_plain]
  rfl

open Classical Discharge.Tables Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain in
/-- **The deployed plain decoders are right at `plainTab`**: at a plain statement in scope with public outputs whose
draw is `d`, `decodesTable_wrP_sc` at `d` and no drawn unit, on the outcomes whose `wrP` transcript flock-verify
accepts, whose root is `rootB₀`'s and on which the reader looks up no column off `C`, with the verdict
`PlainLiveAccAt` reads. -/
theorem plain_hdec {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = false) {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
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
      (fun ps _ => DecodeZL.opnAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j := by
  obtain ⟨ht, hc, ho, hT, hst, rfl, h22, h35⟩ := plain_setupH hs hin hset h hh hd
  obtain ⟨sch, hf⟩ : ∃ sch, Flock.fast100 st.m = some sch := by
    have e := Refine.fast100_schedOf st.m
    rw [hsch] at e
    cases hf : Flock.fast100 st.m with
    | none => rw [hf] at e; cases e
    | some sch => exact ⟨sch, rfl⟩
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x₀, hx₀⟩ := Integrate.drawSetup_of_setupH (S := (∅ : Finset (Fin 0))) (mPts := st.spec.mPts) ht hc ho hst
    hf hD (fun u hu => absurd hu (Finset.notMem_empty u)) rfl
  have e : Integrate.tablesAt (Reg := Unit) ht (fun _ : Finset (Fin 0) => d) x₀ =
      fun _ _ => plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay := by
    funext S R
    obtain rfl : S = ∅ := Finset.eq_empty_of_isEmpty S
    exact tableAt_plainTab ht (fun _ : Finset (Fin 0) => d) (spec := st.spec) (S := ∅) x₀ x₀ hx₀ L m k0 l₀ l₁ rest ext
      hsch hm hlay
  have hS : DecodeZL.DrawOk ins.read.I (fun _ : Finset (Fin 0) => d) st.spec.mPts ∅ :=
    ⟨fun _ _ u hu => absurd hu (Finset.notMem_empty u), fun st' hst' => by
      obtain rfl := Except.ok.inj (hst.symm.trans hst')
      rfl⟩
  have hN : wordLen false x =
      2 ^ (lvOf (Integrate.tablesAt (Reg := Unit) ht (fun _ : Finset (Fin 0) => d) x₀ ∅ ()).sch 0).logLen := by
    rw [e]
    exact plain_wordLen h h22 h35 hsch
  have hdec := decodesTable_wrP_sc ht hc ho (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () hN C rootB₀
  rw [e] at hdec
  exact hdec

end FlockDeploy.KS
