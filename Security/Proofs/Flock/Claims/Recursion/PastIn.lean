import Proofs.Flock.Claims.Recursion.MoreSlots
import Proofs.Flock.Soundness.Discharge.HiddenExec.Ret
import Proofs.Flock.VBridge.InnerUnit

/-!
# A port V\* outputs is an output row (`PastIn`)

At a staged part, `RecVStar.checkParsed` holds the circuit's ports to V\*'s rows (`vstarRows`: the statement's inputs,
then one row per output) and its first output leaf map to the unit's output words in order, `HmRow.check` each output
leaf below the circuit's output words, and `HmOut.checkRows` the rows past `HmOut.nIn` to one per output port, each
holding its port's words. Every V\* unit outputs at least one word per output (`unit_outs`), so the circuit has
at least as many output ports as V\* has outputs: otherwise its output words would fall short of the unit's
(`pastIn_core`). `checkMsgNames` names each port once, so a port named as an output is none of the input rows, and
sits at or past `HmOut.nIn`.
-/

namespace RecDeploy.MoreSlots

open Flock
open Claims (Msg)
open RecDeploy.ResLower (staged_hidden)

/-! ## What V\*'s units output -/

theorem sparse_outWords_pos (sh : RecSparse.Shape) : 0 < RecSparse.outWords sh := by
  unfold RecSparse.outWords RecSparse.residualNames RecSparse.F_WORDS RecSparse.ROW_BLOCK
  cases sh.stacked <;> simp <;> omega

/-- **Each V\* unit outputs a positive number of words per output**: one output `out` of `a > 0` words, or a level's
`out` and `acc_out`, 64 words each. -/
theorem unit_outs {sv : VStar.Stmt} {u : VStar.UnitC} (h : sv.unit = .ok u) :
    (∃ a, 0 < a ∧ u.outPorts = [a] ∧ sv.sub.outputs = ["out"]) ∨
      (u.outPorts = [64, 64] ∧ sv.sub.outputs = ["out", "acc_out"]) := by
  cases sv with
  | recOpen l L H n =>
    exact .inr ⟨(FlockVBridge.OfVStar.recOpenUnit_ok (h : VStar.recOpenUnit L H = .ok u)).2.2.2, rfl⟩
  | recOpenStacked l L H n =>
    have h' : (VStar.recOpenUnit L H >>= fun U => pure { U with name := VStar.recOpenStackedName L H }) =
        (.ok u : Except String VStar.UnitC) := h
    obtain ⟨U, hU, h2⟩ := FlockSoundness.Refine.ebind_ok h'
    cases h2
    exact .inr ⟨(FlockVBridge.OfVStar.recOpenUnit_ok hU).2.2.2, rfl⟩
  | inner j name p =>
    exact .inl ⟨_, by decide,
      (FlockVBridge.OfVStar.innerUnit_ok (h : VStar.innerUnit p.S name = .ok u)).2.2.2.1, rfl⟩
  | sparse sh x =>
    refine .inl ⟨RecSparse.outWords sh, sparse_outWords_pos sh, ?_, rfl⟩
    unfold VStar.Stmt.unit VStar.sparseUnit at h
    try dsimp only at h
    split at h
    · cases h
    · cases h
      rfl
  | opening m c j S ports nv =>
    have h' : (Except.ok (VStar.openingUnit _ S ports nv) : Except String VStar.UnitC) = .ok u := h
    obtain rfl := Except.ok.inj h'
    exact .inl ⟨_, by decide, rfl, rfl⟩
  | cap l N =>
    have h' : (Except.ok (VStar.capUnit l N) : Except String VStar.UnitC) = .ok u := h
    obtain rfl := Except.ok.inj h'
    exact .inl ⟨_, by decide, rfl, rfl⟩

/-! ## The circuit's output words -/

theorem outWords_zero {C : Circuit} (h : C.outPorts.size = 0) : C.outWords = 0 := by
  unfold Circuit.outWords
  rw [← Array.foldl_toList,
    List.eq_nil_of_length_eq_zero (l := C.outPorts.toList) (by rw [Array.length_toList]; exact h)]
  rfl

theorem outWords_one {C : Circuit} (h : C.outPorts.size = 1) : C.outWords = (C.outPorts.getD 0 default).2 := by
  unfold Circuit.outWords
  have h0 : 0 < C.outPorts.size := by omega
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h0, Option.getD_some, ← Array.foldl_toList]
  have hl : C.outPorts.toList = [C.outPorts[0]] := by
    apply List.ext_getElem
    · simp [h]
    · intro i h1 h2
      have : i = 0 := by simp at h2; omega
      subst this
      simp
  rw [hl]
  simp

/-! ## A port named as an output is past the input rows -/

/-- Where a circuit's rows are `A ++ B`, each port named once, a port named as one of `B`'s is at or past `A`. -/
theorem past_inputs {C : Circuit} {A B : List (String × ℕ × ℕ)}
    (hr : C.ports.toList.map (fun p => (p.name, p.words, p.bits)) = A ++ B)
    (hnd : (C.ports.toList.map (·.name)).Nodup) {nm : String} (hm : nm ∈ B.map Prod.fst) {q : ℕ}
    (hq : C.ports.findIdx? (·.name == nm) = some q) : A.length ≤ q := by
  obtain ⟨hqs, hname, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hn : C.ports.toList.map (·.name) = A.map Prod.fst ++ B.map Prod.fst := by
    rw [← List.map_append, ← hr, List.map_map]
    try rfl
  rw [hn] at hnd
  obtain ⟨-, -, hdis⟩ := List.nodup_append.1 hnd
  apply Classical.byContradiction
  intro hlt
  have hlt' : q < (A.map Prod.fst).length := by rw [List.length_map]; omega
  have hg : (C.ports.toList.map (·.name))[q]? = some nm := by
    simp [hqs, beq_iff_eq.1 hname]
  rw [hn, List.getElem?_append_left hlt'] at hg
  exact hdis nm (List.mem_of_getElem? hg) nm hm rfl

/-- **A port V\* outputs is at or past the input rows**, at a circuit holding V\*'s rows, each name once, its output
rows one per output port, each holding its port's words, and its output leaves the unit's output words, each below the
circuit's output words. -/
theorem pastIn_core {C : Circuit} {sub : VStar.Sub} {u : VStar.UnitC}
    (hrows : C.ports.toList.map (fun p => (p.name, p.words, p.bits)) = Flock.RecVStar.vstarRows sub u)
    (hnd : (C.ports.toList.map (·.name)).Nodup) (hcr : HmOut.checkRows C = .ok ())
    (hrr : C.outRowPorts = Array.replicate C.outPorts.size 1) (hb : ∀ p ∈ C.ports.toList, p.bits = 16 * p.words)
    (hlo : C.leavesOut.getD 0 #[] = Array.range u.outWords)
    (hlt : ∀ o ∈ (C.leavesOut.getD 0 #[]).toList, o < C.outWords)
    (hU : (∃ a, 0 < a ∧ u.outPorts = [a] ∧ sub.outputs = ["out"]) ∨
      (u.outPorts = [64, 64] ∧ sub.outputs = ["out", "acc_out"]))
    {nm : String} (hm : nm ∈ sub.outputs) {q : ℕ} (hq : C.ports.findIdx? (·.name == nm) = some q) :
    HmOut.nIn C ≤ q := by
  have hr' : C.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      (sub.inputs.zip u.inPorts).map (fun p => (p.1, p.2, 16 * p.2)) ++
        (sub.outputs.zip u.outPorts).map fun p => (p.1, (p.2 + 63) / 64 * 64, 16 * ((p.2 + 63) / 64 * 64)) := hrows
  have hnI : HmOut.nIn C = C.ports.size - C.outPorts.size := by
    unfold HmOut.nIn
    rw [hrr, Array.size_replicate]
  rcases hU with ⟨a, ha, hp, ho⟩ | ⟨hp, ho⟩
  · have hr2 : C.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
        (sub.inputs.zip u.inPorts).map (fun p => (p.1, p.2, 16 * p.2)) ++
          [("out", (a + 63) / 64 * 64, 16 * ((a + 63) / 64 * 64))] := by
      rw [hr', hp, ho]
      try rfl
    generalize (sub.inputs.zip u.inPorts).map (fun p => (p.1, p.2, 16 * p.2)) = A at hr2
    have hA := past_inputs hr2 hnd (by rw [ho] at hm; simpa using hm) hq
    have hlen := congrArg List.length hr2
    simp only [List.length_map, Array.length_toList, List.length_append, List.length_cons, List.length_nil] at hlen
    have hw : u.outWords = a := by simp [VStar.UnitC.outWords, hp]
    have h1 : 1 ≤ C.outPorts.size := by
      apply Classical.byContradiction
      intro hlt1
      have h0w := hlt 0 (by rw [hlo, hw]; simpa using ha)
      rw [outWords_zero (by omega)] at h0w
      omega
    omega
  · have hr2 : C.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
        (sub.inputs.zip u.inPorts).map (fun p => (p.1, p.2, 16 * p.2)) ++
          [("out", 64, 1024), ("acc_out", 64, 1024)] := by
      rw [hr', hp, ho]
      try rfl
    generalize (sub.inputs.zip u.inPorts).map (fun p => (p.1, p.2, 16 * p.2)) = A at hr2
    have hA := past_inputs hr2 hnd (by rw [ho] at hm; simpa using hm) hq
    have hlen := congrArg List.length hr2
    simp only [List.length_map, Array.length_toList, List.length_append, List.length_cons, List.length_nil] at hlen
    have hw : u.outWords = 128 := by simp [VStar.UnitC.outWords, hp]
    have h2 : 2 ≤ C.outPorts.size := by
      apply Classical.byContradiction
      intro hlt2
      have hw127 := hlt 127 (by rw [hlo, hw]; simp)
      rcases Nat.lt_or_ge C.outPorts.size 1 with h0 | h1
      · rw [outWords_zero (by omega)] at hw127
        omega
      · have hone : C.outPorts.size = 1 := by omega
        have hwr := FlockSoundness.Discharge.HiddenExec.checkRows_words hcr hrr hb (j := 0) (by omega)
        have hn1 : HmOut.nIn C + 0 = A.length + 1 := by rw [hnI]; omega
        have hpw : (C.ports.getD (A.length + 1) default).words = 64 := by
          have hg : (C.ports.toList.map (fun p => (p.name, p.words, p.bits)))[A.length + 1]? =
              some ("acc_out", 64, 1024) := by
            rw [hr2, List.getElem?_append_right (by omega)]
            simp
          have hl : A.length + 1 < C.ports.size := by omega
          rw [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hl, Option.map_some,
            Option.some.injEq] at hg
          rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hl, Option.getD_some]
          exact congrArg (fun e => e.2.1) hg
        rw [hn1, hpw] at hwr
        rw [outWords_one hone] at hw127
        omega
    omega

/-! ## At a staged part -/

/-- With an inner shape, `RecVStar.checkStmt` names each of the table's circuit's ports once (`checkMsgNames`). -/
theorem checkStmt_names {i : Flock.RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {part : Option ℕ} {bindings : Option VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    (h : Flock.RecVStar.checkStmt (some i) tags tables st circuitB partition program registered part bindings rows =
      .ok ()) :
    (st.c.ports.toList.map (·.name)).Nodup := by
  unfold Flock.RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
    cases h1
  · split at h
    · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
      cases h1
    · split at h
      · split at h
        · split at h
          · split at h
            · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
              cases h1
            obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
            obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
            obtain ⟨_, hn, -⟩ := FlockSoundness.Refine.ebind_ok h
            unfold Flock.RecVStar.checkMsgNames at hn
            split at hn
            · cases hn
            · rename_i hnd
              simpa using hnd
          · cases h
        · cases h
      · cases h

/-- A staged part's tables' circuits name each port once. -/
theorem staged_names {st : Stage} {s : ℕ} {x xO : Msg} (h : StagedPart st s x xO) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) {t : ℕ} (ht : t < p.sts.size) :
    (p.sts[t].c.ports.toList.map (·.name)).Nodup := by
  obtain ⟨ms, i, spec, sts, rfl, hsu, -⟩ := h
  have e : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  subst e
  obtain ⟨hins, hset⟩ := setupO_some hsu
  obtain ⟨r, hr, hinner, -⟩ := insO_staged hins
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins] at hC
  exact checkStmt_names (checks_stmt hC t ht)

/-- A hidden-output table's output rows pass `HmOut.checkRows`, one per output port. -/
theorem checkRows_setupOne {tags : Flock.Tags} {cf pf : ByteArray} {coins : _} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Flock.Stmt} (ht : tags.typed = false)
    (hh : tags.hiddenOutputs = true)
    (hd : FlockSoundness.Discharge.HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks) :
    HmOut.checkRows st.c = .ok () ∧ st.c.outRowPorts = Array.replicate st.c.outPorts.size 1 := by
  have hd' := hd
  unfold FlockSoundness.Discharge.HiddenExec.setupOne at hd'
  rw [hh] at hd'
  simp only [↓reduceIte] at hd'
  have s := FlockSoundness.Discharge.Hidden.setupHidden_spec' hd'
  obtain ⟨hp, hcr⟩ := FlockSoundness.Discharge.Hidden.parse_untyped ht s.parse
  exact ⟨hcr, (FlockSoundness.Discharge.Hidden.out_rows_v1 hp hcr fun p hp => (hv1 p hp).1).2.1⟩

/-- **`PastIn` holds at every stage**: at a staged part, its circuit holds V\*'s rows and output leaves
(`checkParsed_ok`, `checkParsed_leavesOut`), names each port once (`staged_names`), has one output row per output
port, each holding its port's words (`checkRows_setupOne`, `checkRows_words`), and each output leaf below its output
words (`Layout.check_out`), so `pastIn_core`. -/
theorem pastIn (st : Stage) : PastIn st := by
  intro s x xO r ss sv c nm q hS hr hss hsv hpc hm hq
  obtain ⟨p, r', sv', hx, ht, hpi, hr', ⟨ss', hss', hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨spec0, hset⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall 0 h0
  have htd : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  have hh : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).hiddenOutputs = true := by
    unfold FlockSoundness.Discharge.Exec.tagsAt
    split <;> exact staged_hidden hS hx h0
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0
  have hs := (FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports hs
  have ho : p.sts[0].c.outNet = p.sts[0].c.unitNet := by
    unfold FlockSoundness.Discharge.Layout.scopeOk at hs
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hs
    exact hs.1.1.1.2
  obtain ⟨hcr, hrr⟩ := checkRows_setupOne htd hh hd hv1
  obtain ⟨u, hu, hupv, -, -, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  obtain ⟨u', hu', hlo⟩ := FlockSoundness.Discharge.Recursion.checkParsed_leavesOut (htab 0 h0).2.2
  rw [hu] at hu'
  cases hu'
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨-, -, hlv⟩ := FlockSoundness.Discharge.Layout.check_out P.typed ho hls
  have hget0 : p.sts[0].c.leavesOut.getD 0 #[] = p.sts[0].c.leavesOut[0]! := by
    rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD']; rfl
  have hnd := staged_names hS hx h0
  subst hc0
  exact pastIn_core hrows hnd hcr hrr (fun p hp => (hv1 p hp).2.1) hlo
    (fun o ho => by rw [hget0] at ho; exact (hlv 0 (by omega)).2 o ho) (unit_outs hu) hm hq

end RecDeploy.MoreSlots
