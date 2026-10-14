import Proofs.Flock.C3.PubRows.Honest
import Proofs.Flock.C3.VStarList.Shape
import Proofs.Flock.VBridge.Sparse.Unit
import Proofs.Flock.VBridge.Pinned

/-!
# n1: a rec-sparse unit the verifier checked is at its RecSparse shape

`SparseAtShape` does not follow from `hob`: `bind` is free, and at a binding `VStar.metaHead` refuses, `hob` says
nothing of `Vs j`. The verifier's statement check does: `RecVStar.checkStmt` at the inner shape, `j` and `j`'s binding
finds the checked circuit's unit name in the list (`checkStmt_ok`), and the list names RecSparse only at the inner
shape's `sparse` entry (`sparse_of_name`: the other statements' names start `rec-o` or `rec-r`, `RecSparse.unitName` is
injective, and candidate 3's statements after RecSparse are R's levels and opening, `candidate3_rest`).

* `SparseAtShapeChecked`: `SparseAtShape` at a statement of `Vs j`'s circuit that `checkStmt` passes;
  `sparseAtShapeChecked` proves it of every family. `NameChecked` (`nameChecked`, the same way): such a circuit is
  named as the family's list's statement `j`.
* `ChecksAccepted`: where a session accepts, a transcript it accepts passes `checkPubRows` and `checkLabels`
  (`PubRowsChecked`) and
  `checkStmt` (`StmtsChecked`, `AcceptsZKV`'s second conjunct), n1b's checks; the checked family meets it
  (`checksAccepted_checked`, by `checks_of_keepOwn`). `ChecksReached` is it at the transcripts a session writes, and
  `hPubRows_of_checked` is `hPubRows_of_check'` with it, `rowsHeld` and `SparseAtShapeChecked`.
* `ownIn_of_checks`: where `ChecksAccepted` holds, the accepted sessions' reads hold the verifier's message-round ports
  of their statement (`OwnIn` at `ownAt`), by `checkOwn`'s second and third loops (`checkOwn_ok`,
  `ownIn_of_checkStmt`); `ownIn_checked` is it at the checked family.
-/

namespace FlockC3.PubRows

section List

open Flock

theorem take_five {p : String} {t : List Char} (hp : 5 ≤ p.toList.length) :
    (p.toList ++ t).take 5 = p.toList.take 5 := List.take_append_of_le_length hp

/-- **The name `rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>` determines the shape.** -/
theorem sparse_unitName_inj {sh sh' : RecSparse.Shape} (h : RecSparse.unitName sh = RecSparse.unitName sh') :
    sh = sh' := by
  have h := congrArg String.toList h
  simp only [RecSparse.unitName, String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  have h := List.append_cancel_left h
  have dig : ∀ n, ∀ x ∈ Nat.toDigits 10 n, x.isDigit := fun n x hx =>
    Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  obtain ⟨h1, h⟩ := FlockVBridge.digits_dash (dig _) (dig _) (by simpa using h)
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨h2, h⟩ := FlockVBridge.digits_dash (dig _) (dig _) (by simpa using h)
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨h3, h⟩ := FlockVBridge.digits_dash (dig _) (dig _) (by simpa using h)
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨_, _, _, _⟩ := sh
  obtain ⟨_, _, _, _⟩ := sh'
  simp only [RecSparse.Shape.mk.injEq]
  exact ⟨FlockVBridge.eq_of_toDigits h1, FlockVBridge.eq_of_toDigits h2, FlockVBridge.eq_of_toDigits h3,
    FlockVBridge.eq_of_toDigits h⟩

/-- **No other statement's name is a RecSparse unit's**: V*'s levels and parts, R's levels and R's opening are named
`rec-o…` or `rec-r…`, RecSparse `rec-s…`. -/
theorem sparse_ne_names (sh : RecSparse.Shape) :
    (∀ L H : ℕ, RecSparse.unitName sh ≠ s!"rec-open/l{L}-h{H}") ∧
    (∀ m k c j : ℕ, RecSparse.unitName sh ≠ s!"rec-residuals/m{m}-k{k}-c{c}-p{j}") ∧
    (∀ L H, RecSparse.unitName sh ≠ VStar.recOpenStackedName L H) ∧
    (∀ m c j, RecSparse.unitName sh ≠ RecOpening.unitName m c j) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> intros <;> intro h <;>
    have h := congrArg (fun s : String => s.toList.take 5) h <;>
    simp only [RecSparse.unitName, VStar.recOpenStackedName, RecOpening.unitName, String.toList_append,
      List.append_assoc, FlockVBridge.toString_str] at h <;>
    rw [take_five (by decide), take_five (by decide)] at h <;>
    exact absurd h (by decide)

theorem forIn_all {α : Type} (P : VStar.Stmt → Prop)
    {f : α → Array VStar.Stmt → Except String (ForInStep (Array VStar.Stmt))}
    (hf : ∀ a acc r, f a acc = .ok r → (∀ s ∈ acc, P s) → ∀ s ∈ r.value, P s) :
    ∀ (l : List α) (acc : Array VStar.Stmt) {out : Array VStar.Stmt}, forIn l acc f = .ok out →
      (∀ s ∈ acc, P s) → ∀ s ∈ out, P s
  | [], acc, out, h, hacc => by
    simp only [List.forIn_nil] at h
    cases h
    exact hacc
  | a :: l, acc, out, h, hacc => by
    rw [List.forIn_cons] at h
    obtain ⟨r, hr, h⟩ := FlockC3.VStarList.eok_bind h
    have hr' := hf a acc r hr hacc
    cases r with
    | done b => cases h; exact hr'
    | yield b => exact forIn_all P hf l b h hr'

/-- **R's opening is `rec-opening` statements.** -/
theorem openingParts_kind {r : RecVStar.StackedShape} {x : Array VStar.Stmt} (h : RecVStar.openingParts r = .ok x) :
    ∀ s ∈ x, ∃ m c k S ports nv, s = VStar.Stmt.opening m c k S ports nv := by
  unfold RecVStar.openingParts at h
  split at h
  rotate_left
  · cases h
  obtain ⟨S, -, h⟩ := FlockC3.VStarList.eok_bind h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := FlockC3.VStarList.eok_bind h
    cases h1
  obtain ⟨gs, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨out, hloop, h⟩ := FlockC3.VStarList.eok_bind h
  cases h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  refine forIn_all _ (fun j acc r hr hacc => ?_) _ _ hloop (by simp)
  try dsimp only at hr
  obtain ⟨S', -, hr⟩ := FlockC3.VStarList.eok_bind hr
  cases hr
  intro s hs
  rcases Array.mem_push.1 hs with hs | rfl
  · exact hacc s hs
  · exact ⟨_, _, _, _, _, _, rfl⟩

/-- **Candidate 3's statements after RecSparse** are R's levels and the parts of R's opening. -/
theorem candidate3_rest {i : RecVStar.Inner} {C : Array VStar.Stmt} (h : RecVStar.candidate3Statements i = .ok C)
    {sp : RecSparse.Shape} (hsp : i.sparse = some sp) :
    ∃ rest, C = #[.sparse sp (RecVStar.hRound i.m i.kLog)] ++ rest ∧ ∀ s ∈ rest,
      (∃ l L H n, s = .recOpenStacked l L H n) ∨ ∃ m c k S ports nv, s = .opening m c k S ports nv := by
  unfold RecVStar.candidate3Statements at h
  rw [hsp] at h
  dsimp only at h
  split at h
  · cases h
  cases hr : i.stacked with
  | none =>
    rw [hr] at h
    cases h
  | some r =>
    rw [hr] at h
    dsimp only at h
    split at h
    · cases h
    split at h
    · cases h
    split at h
    rotate_left
    · cases h
    obtain ⟨x, hx, h⟩ := FlockC3.VStarList.eok_bind h
    cases h
    refine ⟨_, Array.append_assoc, fun s hs => ?_⟩
    rcases Array.mem_append.1 hs with hs | hs
    · simp only [Array.mem_map, List.mem_toArray, List.mem_range] at hs
      obtain ⟨l, -, rfl⟩ := hs
      exact .inl ⟨_, _, _, _, rfl⟩
    · exact .inr (openingParts_kind hx s hs)

/-- **The list names RecSparse only at the inner shape's `sparse` entry**: a statement of the verifier's list named
`unitName sh` is candidate 3's RecSparse, at `sh`. -/
theorem sparse_of_name {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    {s : VStar.Stmt} (hs : s ∈ ss) {sh : RecSparse.Shape} (hn : RecSparse.unitName sh = s.name) :
    i.sparse = some sh := by
  obtain ⟨n1, n2, n3, n4⟩ := sparse_ne_names sh
  obtain ⟨vsh, A, C, hA, hC, rfl, -⟩ := FlockC3.VStarList.statements_ok h
  rcases Array.mem_append.1 hs with hs | hs
  · obtain ⟨_, _, _, -, -, -, rfl⟩ := FlockVBridge.Family.statements_eq hA
    simp only [Array.mem_append, List.mem_toArray, List.mem_map, List.mem_range] at hs
    rcases hs with ⟨l, -, rfl⟩ | ⟨j, -, rfl⟩
    · exact absurd hn (n1 _ _)
    · exact absurd hn (n2 _ _ _ _)
  · cases hsp : i.sparse with
    | none =>
      rw [(FlockC3.VStarList.candidate3_ok hC).1 hsp] at hs
      simp at hs
    | some sp =>
      obtain ⟨rest, rfl, hrest⟩ := candidate3_rest hC hsp
      rcases Array.mem_append.1 hs with hs | hs
      · simp only [List.mem_toArray, List.mem_singleton] at hs
        subst hs
        rw [sparse_unitName_inj hn]
      · rcases hrest s hs with ⟨l, L, H, n, rfl⟩ | ⟨m, c, k, S, ports, nv, rfl⟩
        · exact absurd hn (n3 _ _)
        · exact absurd hn (n4 _ _ _)

/-- **A circuit the verifier checked as V*'s statement `j` is RecSparse only at the inner shape's `sparse` entry.** -/
theorem sparse_of_checkStmt {i : RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f) rows =
      .ok ()) {sh : RecSparse.Shape} (hp : RecSparse.params st.c.unit.name = some sh) : i.sparse = some sh := by
  obtain ⟨-, -, ss, s, hss, hj, hname, -⟩ := FlockSoundness.Discharge.Recursion.checkStmt_ok h
  exact sparse_of_name hss (Array.mem_of_getElem? hj)
    ((FlockVBridge.Sparse.unitName_params hp).trans hname.symm)

/-- **`checkOwn` accepting, its second and third loops**: each of the statement's message-round ports is one of the
reads file's, and each of its `sparseMsgs` is one of the file's ports. -/
theorem checkOwn_ok {s : VStar.Stmt} {own : Array Flock.Registered.Port} (h : RecVStar.checkOwn s own = .ok ()) :
    (∀ q ∈ RecVStar.ownPorts s, q ∈ RecVStar.msgPorts own) ∧
      ∀ q ∈ RecVStar.sparseMsgs s, q ∈ own.toList.map fun p => (p.name, p.value, p.reads) := by
  unfold RecVStar.checkOwn at h
  simp only at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hnone
      split at h
      · cases h
      · rename_i hnone'
        refine ⟨fun q hq => ?_, fun q hq => ?_⟩
        · have hfind : (RecVStar.ownPorts s).find? (fun x => !(RecVStar.msgPorts own).contains x) = none := by
            cases hf : (RecVStar.ownPorts s).find? (fun x => !(RecVStar.msgPorts own).contains x) with
            | none => rfl
            | some q' => exact absurd hf (hnone q')
          simpa using List.find?_eq_none.1 hfind q hq
        · have hfind : (RecVStar.sparseMsgs s).find?
              (fun x => !(own.toList.map fun p => (p.name, p.value, p.reads)).contains x) = none := by
            cases hf : (RecVStar.sparseMsgs s).find?
                (fun x => !(own.toList.map fun p => (p.name, p.value, p.reads)).contains x) with
            | none => rfl
            | some q' => exact absurd hf (hnone' q')
          simpa using List.find?_eq_none.1 hfind q hq

/-- **`checkOwn` accepting, its second loop, at the reads file's ports**: each of the statement's message-round ports is
a port of the reads file, by name, value and reads. -/
theorem own_of_checkOwn {s : VStar.Stmt} {own : Array Flock.Registered.Port} (h : RecVStar.checkOwn s own = .ok ()) :
    ∀ q ∈ RecVStar.ownPorts s, ∃ p ∈ own, (p.name, p.value, p.reads) = q := by
  intro q hq
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 ((checkOwn_ok h).1 q hq)
  exact ⟨p, Array.mem_toList_iff.1 (List.mem_filter.1 hp).1, rfl⟩

/-- **A RecSparse statement's reads file names every one of its message ports** (`RecSparse.messagePorts`): `h` and
`claim` by `checkOwn`'s second loop (`ownPorts`), the others by its third (`sparseMsgs`, `checkOwn_ok`). -/
theorem sparseOwnNames_of_checkOwn {sh : RecSparse.Shape} {hr : ℕ} {own : Array Flock.Registered.Port}
    (h : RecVStar.checkOwn (.sparse sh hr) own = .ok ()) :
    ∀ nm ∈ (RecSparse.messagePorts sh).map (·.1), ∃ p ∈ own, p.name = nm := by
  obtain ⟨rest, hY⟩ : ∃ rest, RecSparse.messagePorts sh = ("h", RecSparse.SKIP) :: (rest ++ [("claim", 1)]) :=
    ⟨_, rfl⟩
  have hown := own_of_checkOwn h
  have h3 := (checkOwn_ok h).2
  simp only [RecVStar.sparseMsgs, hY, List.drop_succ_cons, List.drop_zero, List.dropLast_concat] at h3
  intro nm hnm
  rw [hY] at hnm
  simp only [List.map_cons, List.map_append, List.map_nil, List.mem_cons, List.mem_append, List.not_mem_nil,
    or_false] at hnm
  rcases hnm with rfl | hnm | rfl
  · obtain ⟨p, hp, he⟩ := hown ("h", s!"rec-c{hr}", #[some 0, some 1]) (by simp [RecVStar.ownPorts])
    exact ⟨p, hp, (Prod.mk.inj he).1⟩
  · obtain ⟨⟨a, b⟩, hx, rfl⟩ := List.mem_map.1 hnm
    obtain ⟨p, hp, he⟩ := List.mem_map.1 (h3 _ (List.mem_map.2 ⟨(a, b), hx, rfl⟩))
    exact ⟨p, Array.mem_toList_iff.1 hp, (Prod.mk.inj he).1⟩
  · obtain ⟨p, hp, he⟩ := hown ("claim", RecVStar.CLAIM_VALUE, #[some 0, some 1]) (by simp [RecVStar.ownPorts])
    exact ⟨p, hp, (Prod.mk.inj he).1⟩

/-- **A statement the verifier checked as V*'s statement `j` has its message ports in the reads file**: each port of
`ownAt ss j` is one of `own`'s by name, value and reads (`checkStmt_ok`'s `checkOwn`, its second and third loops). -/
theorem ownIn_of_checkStmt {i : RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f) rows =
      .ok ()) {ss : Array VStar.Stmt} (hss : RecVStar.statements i = .ok ss) :
    ∀ p ∈ ownAt ss j, ∃ p' ∈ own, p'.name = p.name ∧ p'.value = p.value ∧ p'.reads = p.reads := by
  obtain ⟨-, -, ss', s, hss', hj, -, -, -, ho, -⟩ := FlockSoundness.Discharge.Recursion.checkStmt_ok h
  obtain rfl := Except.ok.inj (hss'.symm.trans hss)
  intro p hp
  simp only [ownAt, hj, Option.map_some, Option.getD_some, List.map_toArray, List.mem_toArray, List.mem_map] at hp
  obtain ⟨q, hq, rfl⟩ := hp
  obtain ⟨h2, h3⟩ := checkOwn_ok ho
  rcases List.mem_append.1 hq with hq | hq
  · obtain ⟨p', hp', rfl⟩ := List.mem_map.1 (h2 q hq)
    exact ⟨p', Array.mem_toList_iff.1 (List.mem_filter.1 hp').1, rfl, rfl, rfl⟩
  · obtain ⟨p', hp', rfl⟩ := List.mem_map.1 (h3 q hq)
    exact ⟨p', Array.mem_toList_iff.1 hp', rfl, rfl, rfl⟩

end List

section Session

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockSoundness.Discharge.ZkReg FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
open FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.Composed (drawJ partOf)
open FlockC3.Compose

/-- **The verifier's check of a session's statements** (`AcceptsZKV`'s second conjunct): `RecVStar.checkStmt` passes
on each statement the session's record sets up, at the inner shape `i` as V*'s statement `j` under the bindings `f`,
against the inputs' tags, tables, circuit file, partition object and program, the reads `own` and the shared rows
`rows`. -/
def StmtsChecked (i : RecVStar.Inner) (j : ℕ) (f : VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    (I : Inputs) (own : Array Flock.Registered.Port) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
    sts.forM (RecVStar.checkStmt (some i) I.tags I.tables · I.circuitFile I.partition I.program (some own) (some j)
      (some f) rows) = .ok ()

/-- **A checked session's statement of its circuit**: where the session accepts and its record draws, the statement
setting up a drawn unit's part has the session's circuit and passes `checkStmt`. -/
theorem stmt_of_session {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)
    {tr : Exec.Transcript} (hacc : AcceptsZKR zo.I zo.own tr) {i : RecVStar.Inner} {j : ℕ} {f : VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)} (hck : StmtsChecked i j f rows zo.I zo.own tr) {d : Lean.Json}
    (hd : recordDraw tr.record = some d) (u : Fin nv) :
    ∃ sj : Flock.Stmt, sj.c = c ∧ RecVStar.checkStmt (some i) zo.I.tags zo.I.tables sj zo.I.circuitFile
      zo.I.partition zo.I.program (some zo.own) (some j) (some f) rows = .ok () := by
  obtain ⟨spec, sts, hs, -⟩ := hacc
  obtain ⟨spec', sts', hs', hfm⟩ := hck
  have hst : sts' = sts := congrArg Prod.snd (Except.ok.inj (hs'.symm.trans hs))
  rw [hst] at hfm
  obtain ⟨hsz, hstJ⟩ := ZkHidden.setupOf_tableJH zo.hTags.1 hd hs
  have hjs : (partOf zo.I d u.val).val < sts.size := by rw [hsz]; exact (partOf zo.I d u.val).isLt
  have s := Hidden.setupHidden_spec' (hstJ _ hjs)
  have hcst : sts[(partOf zo.I d u.val).val].c = c := by
    have h1 := s.parse
    rw [ZkHidden.parseH_tagsAt ((zkTags_typed zo.I).trans zo.ht), zo.hcH] at h1
    exact (Except.ok.inj h1).symm
  exact ⟨_, hcst, array_forM_ok hfm _ hjs⟩

/-- **A kept session that accepts kept its own list** (`Flock.Guarantees.keepOwn`), so some transcript it accepts
passes the keep's predicate `P`, and so any `Pc` that `P` implies. -/
theorem checks_of_keepOwn {Q : VStmt} {Reg' : Type} {z : Q.Outer Reg'} {P Pc : Inputs → Array Flock.Registered.Port →
    Exec.Transcript → Prop} (hP : ∀ J o t, P J o t → Pc J o t)
    (h : ∃ t, AcceptsZKR (Flock.Guarantees.keepOwn z P).I (Flock.Guarantees.keepOwn z P).own t) :
    ∃ t, AcceptsZKR (Flock.Guarantees.keepOwn z P).I (Flock.Guarantees.keepOwn z P).own t ∧
      Pc (Flock.Guarantees.keepOwn z P).I (Flock.Guarantees.keepOwn z P).own t := by
  obtain ⟨t, ht⟩ := h
  by_cases hp : ∃ t, AcceptsZKR z.I z.own t ∧ P z.I z.own t
  · have ho : (Flock.Guarantees.keepOwn z P).own = z.own := ite_eq_left hp
    obtain ⟨t', ha, hpt⟩ := hp
    rw [ho]
    exact ⟨t', ha, hP _ _ _ hpt⟩
  · have hf : (Flock.Guarantees.keepOwn z P).own = freshOwn Q.c := ite_eq_right hp
    rw [hf] at ht
    exact (not_acceptsZKR_fresh z.I z.ht z.hTags z.hcH t ht).elim

/-- **`checkMsgNames` passing names each port of the circuit once** (its first refusal). -/
theorem names_of_msgNames {s : VStar.Stmt} {c : Flock.Circuit} (h : RecVStar.checkMsgNames s c = .ok ()) :
    ∀ q q' : Fin c.ports.size, c.ports[q].name = c.ports[q'].name → q = q' := by
  unfold RecVStar.checkMsgNames at h
  split at h
  · cases h
  · rename_i hn
    have hnd : (c.ports.toList.map (·.name)).Nodup := by simpa using hn
    intro q q' he
    rcases Nat.lt_trichotomy q.val q'.val with hl | he' | hl
    · exact absurd he (by simpa using List.pairwise_iff_getElem.1 hnd q.val q'.val (by simp) (by simp) hl)
    · exact Fin.ext he'
    · exact absurd he.symm (by simpa using List.pairwise_iff_getElem.1 hnd q'.val q.val (by simp) (by simp) hl)

/-- **`checkStmt` accepting passes `checkMsgNames`** at statement `j` of the list, on the circuit. -/
theorem msgNames_of_checkStmt {i : RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Option (Array Flock.Registered.Port)} {j : ℕ} {f : VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    (h : RecVStar.checkStmt (some i) tags tables st circuitB partition program own (some j) (some f) rows = .ok ()) :
    ∃ ss s, RecVStar.statements i = .ok ss ∧ ss[j]? = some s ∧ RecVStar.checkMsgNames s st.c = .ok () := by
  unfold RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := ebind_ok h
    cases h1
  · split at h
    · obtain ⟨_, h1, -⟩ := ebind_ok h
      cases h1
    · split at h
      · split at h
        · obtain ⟨_, h1, -⟩ := ebind_ok h
          cases h1
        obtain ⟨_, -, h⟩ := ebind_ok h
        obtain ⟨s, hs, h⟩ := ebind_ok h
        obtain ⟨ss, hss, hj, -⟩ := FlockSoundness.Discharge.Recursion.check_ok
          (FlockSoundness.Discharge.Recursion.checkCircuit_ok hs).1
        obtain ⟨⟨⟩, hm, -⟩ := ebind_ok h
        exact ⟨ss, s, hss, hj, hm⟩
      · cases h

/-- **A circuit `checkStmt` passes as one of V*'s statements names each port once** (`checkMsgNames`' first
refusal), so the verifier's find of a port by name (`Registered.checkPort`) finds that port. -/
theorem names_of_checkStmt {i : RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Option (Array Flock.Registered.Port)} {j : ℕ} {f : VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    (h : RecVStar.checkStmt (some i) tags tables st circuitB partition program own (some j) (some f) rows = .ok ())
    {c : Flock.Circuit} (hc : st.c = c) : ∀ q q' : Fin c.ports.size, c.ports[q].name = c.ports[q'].name → q = q' := by
  subst hc
  obtain ⟨_, _, -, -, hm⟩ := msgNames_of_checkStmt h
  exact names_of_msgNames hm

end Session

section Family

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(n1) A rec-sparse unit the verifier checked is at the verifier's RecSparse shape**: `SparseAtShape` at `j` where
`RecVStar.checkStmt` at the inner shape, `j` and `j`'s binding passes on a statement of `Vs j`'s circuit. -/
def SparseAtShapeChecked (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) : Prop :=
  ∀ j (sj : Flock.Stmt) tags tables circuitB partition program own rows, sj.c = (F.Vs j).c →
    RecVStar.checkStmt (some F.ish) tags tables sj circuitB partition program (some own) (some (j : ℕ))
      (some (F.bind j)) rows = .ok () →
    ∀ sh sh₀, RecSparse.params (F.Vs j).c.unit.name = some sh → F.ish.sparse = some sh₀ → sh = sh₀

/-- **(n1) Every family's checked rec-sparse units are at its shape.** -/
theorem sparseAtShapeChecked (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) : SparseAtShapeChecked F := by
  intro j sj _ _ _ _ _ _ _ hc h sh sh₀ hsh h₀
  rw [← hc] at hsh
  exact Option.some.inj ((sparse_of_checkStmt h hsh).symm.trans h₀)

/-- **A circuit the verifier checked as V*'s statement `j` is named as the list's statement `j`**: where
`RecVStar.checkStmt` at the inner shape, `j` and `j`'s binding passes on a statement of `Vs j`'s circuit, the family's
list has a statement at `j`, and its name is `Vs j`'s unit's. -/
def NameChecked (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) : Prop :=
  ∀ j (sj : Flock.Stmt) tags tables circuitB partition program own rows, sj.c = (F.Vs j).c →
    RecVStar.checkStmt (some F.ish) tags tables sj circuitB partition program (some own) (some (j : ℕ))
      (some (F.bind j)) rows = .ok () →
    ∃ s, F.stmts[(j : ℕ)]? = some s ∧ s.name = (F.Vs j).c.unit.name

/-- **Every family's checked circuits are named as its list's statements** (`checkStmt_ok`, as `sparse_of_checkStmt`;
the list `checkStmt` reads is the family's, `hstmts`). -/
theorem nameChecked (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) : NameChecked F := by
  intro j sj _ _ _ _ _ _ _ hc h
  obtain ⟨-, -, ss, s, hss, hj, hname, -⟩ := FlockSoundness.Discharge.Recursion.checkStmt_ok h
  obtain rfl := Except.ok.inj (hss.symm.trans F.hstmts)
  exact ⟨s, hj, by rw [hname, hc]⟩

/-- **(n1b) Where a session accepts, a transcript it accepts passes the verifier's checks** of its public rows against
its hold of the inner transcript (`PubRowsChecked`) and of its statements as V*'s statement of its index
(`StmtsChecked`), as the checked sessions keep it (`checksAccepted_checked`). -/
def ChecksAccepted (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (rows : Option (Array Flock.Registered.Port))
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json) : Prop :=
  ∀ R ω t R₂ j, (∃ tr', AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr') →
    ∃ tr', AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr' ∧
      PubRowsChecked (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption F.ish.sparse (F.zs R ω t R₂ j).I
        (F.zs R ω t R₂ j).own tr' ∧
      StmtsChecked F.ish j (F.bind j) rows (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr'

/-- **(n1b) Where a session accepts a transcript it writes, a transcript it writes and accepts passes the verifier's
checks** of its public rows (`PubRowsChecked`) and statements (`StmtsChecked`): `ChecksAccepted` at the outcomes the
session's game reaches (`ZkOuter.Writes`), the transcripts whose records the custody reads (`rowsHeld`). The verifier
checks the transcript it is sent; the checked family keeps a session's list where some transcript it accepts passes
the checks, written or not (`Flock.Guarantees.keepOwn`), so it gives `ChecksAccepted` (`checksAccepted_checked`) and not
this. -/
def ChecksReached (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (rows : Option (Array Flock.Registered.Port))
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json) : Prop :=
  ∀ R ω t R₂ j, (∃ R' tr', (F.zs R ω t R₂ j).Writes R' tr' ∧ AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr') →
    ∃ R' tr', (F.zs R ω t R₂ j).Writes R' tr' ∧ AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr' ∧
      PubRowsChecked (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption F.ish.sparse (F.zs R ω t R₂ j).I
        (F.zs R ω t R₂ j).own tr' ∧
      StmtsChecked F.ish j (F.bind j) rows (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr'

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).Outer Reg'}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}

/-- **(n1b) The checked family passes the verifier's checks where a session accepts**: a checked session that accepts
kept its own list, so a transcript it accepts passes `ChecksC3`, whose public-row and statement checks these are. -/
theorem checksAccepted_checked {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} :
    ChecksAccepted (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg)
      rows wr reg := by
  intro R ω t R₂ j hex
  refine checks_of_keepOwn (z := Flock.Guarantees.custodied zr hCust h R ω t R₂ j)
    (P := fun J o tr' => (holoInnerAt T CT st hp).r ≤ t.length ∧
      Flock.Guarantees.ChecksC3 ish (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption (coins (wr t).2) j (bind j)
        rows J o tr')
    (Pc := fun J o tr' => PubRowsChecked (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption ish.sparse J o tr' ∧
      StmtsChecked ish j (bind j) rows J o tr') ?_ hex
  intro J o tr' hp
  obtain ⟨-, spec, sts, hs, hst, -, hpr, hlb, -⟩ := hp
  exact ⟨⟨spec, sts, hs, hpr, hlb⟩, spec, sts, hs, hst⟩

/-- **(n1b) The accepted sessions' reads hold the verifier's message-round ports** (`OwnIn` at `ownAt`) where a
transcript they accept passes the statement check (`ChecksAccepted`): the check of the session's first statement
(`setup_zero_c`) is `checkOwn` at the family's list's statement `j` (`ownIn_of_checkStmt`). -/
theorem ownIn_of_checks {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hca : ChecksAccepted F rows wr reg) : OwnIn F fun j => ownAt F.stmts j := by
  intro R ω t R₂ j hex
  obtain ⟨tr, -, -, spec, sts, hs, hfm⟩ := hca R ω t R₂ j hex
  obtain ⟨h0, -⟩ := setup_zero_c (F.zs R ω t R₂ j).ht (F.zs R ω t R₂ j).hTags.1 (F.zs R ω t R₂ j).hcH hs
  exact ownIn_of_checkStmt (array_forM_ok hfm 0 h0) F.hstmts

/-- **(n1b) The checked family's accepted sessions' reads hold the verifier's message-round ports**, at whatever `own`
the family is (`checksAccepted_checked`, `ownIn_of_checks`); at `own := fun j => ownAt ss j` it is `hbr_honest`'s
`hown`. -/
theorem ownIn_checked {coins : String → Option Flock.RecCoins.Source} {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} :
    OwnIn (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg)
      fun j => ownAt ss j :=
  ownIn_of_checks (checksAccepted_checked (coins := coins) (rows := rows) (zr := zr) (hCust := hCust) (h := h))

local notation "𝓗" => C3Family.honest (xR := xR) ish hss bind hob zs own sel bc dec wr reg

/-- **(n1) The public rows at the honest family, from the verifier's checks**: `hPubRows_of_check'` with the
rec-sparse shape read at a transcript the session writes and accepts whose public rows and statements the verifier
checks (`ChecksReached`), where `SparseAtShapeChecked` holds (`sparseAtShapeChecked`) and every unit is drawn
(`rowsHeld`). The port the check finds by name is the session's: the statement check refuses a circuit naming a port
twice (`names_of_checkStmt`), so `PortsNamed` holds at every accepted session. -/
theorem hPubRows_of_checked {rows : Option (Array Flock.Registered.Port)} (hchk : ChecksReached 𝓗 rows wr reg)
    (hsp : SparseAtShapeChecked 𝓗) : HPubRows 𝓗 := by
  intro R ω t R₂ j hex rg hrg u q x hname hread b hrow
  obtain ⟨R', tr', hw, hacc, hchk', hst⟩ := hchk R ω t R₂ j hex
  obtain ⟨cs, w, c, hcs, hc, hb⟩ :=
    heldRow_some (show heldRow (innerCommitsAt (wr t).1 (wr t).2 st reg) ish.sparse rg.value x = some b from hrow)
  have hfix : (RecVStar.fixedRow { reps := #[] } rg.value 0).isSome = true := by
    rw [fixedRow_isSome { reps := #[] } cs rg.value 0 x none ish.sparse]
    simp [hc]
  obtain ⟨d, D, dp, idx, hd, hD, hdp, hidx, hu⟩ := rowsHeld 𝓗 R ω t R₂ j R' tr' hw hacc rg hrg hfix u x hread
  obtain ⟨sj, hsj, hck⟩ := stmt_of_session ((𝓗).zs R ω t R₂ j) hacc hst hd u
  have hfind := port_find_of_unique (names_of_checkStmt hck hsj) q
  rw [hname] at hfind
  rw [hcs] at hchk'
  obtain ⟨spec₀, sts₀, hs₀, hpr₀, hlb₀⟩ := hchk'
  have hlab := label_of_checked ((𝓗).zs R ω t R₂ j) hacc ⟨spec₀, sts₀, hs₀, hlb₀⟩ hd hD hdp hidx hu
  obtain ⟨w', c', hc', hdig⟩ := pubRow_of_checked ((𝓗).zs R ω t R₂ j) hacc (cs := cs) ⟨spec₀, sts₀, hs₀, hpr₀⟩ hrg
    hfix hfind hd hD hdp hidx hu (by rw [hlab]; exact hread)
  have he := (fixedRow_or (hsp j sj _ _ _ _ _ _ _ hsj hck) hc).symm.trans hc'
  simp only [Option.some.injEq, Except.ok.injEq, Prod.mk.injEq] at he
  exact hdig.trans (he.2.symm.trans hb.symm)

end Family

end FlockC3.PubRows
