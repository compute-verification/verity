import Proofs.Flock.VBridge.Family
import Proofs.Flock.VBridge.Pinned
import Proofs.Flock.VBridge.ResLayer

/-!
# A rep's values in V*'s layers: `HStruct`, `HShape`, `HResL`

`repOf` is the `rep₀` of a family, read as V*'s Python reads it (`rec_vstage.py:352-366`, `409-412`):

* `w`, from the wires: element `i`'s bit `e` is the layer's bit at the input column of a part's `InnerRepCheck` copy
  (copy 0 of the rep's instance, `units_per_vu` 1) that holds bit `e` of `i` in one of the part's row ports (`rowCol`)
  and that a row of the unit reads (`IoColAt`, which a column a computed row reads is: `ioColAt_of_readCol`), where the
  layer agrees with the registrant (`RegAgree`);
* `rounds`, `extras`, `comb`, from the verifier's coins at the history: the rep's stream's coins and the session's extra
  claims in a table's files `tf` (the family's `files`, or any), and the circuit fold the lincheck computes at the coins'
  points (`Piop.lean:143`), 64 elements.

`hStruct_of`, `hShape_of` and `hResL_of` prove `HStruct`, `HShape` and `HResL` for every family whose `rep₀` is `repOf`,
from facts about V*'s statements, circuits and registrations, each a named `Prop` below saying which check or lane
establishes it. `HResL` asks each part's row ports for the rep's `w` only at the columns some computed row reads
(`ResZeroAtL`), so each such bit of `w` is read where it is asked, and `repOf`'s `w` needs no column beyond those.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el)
open FlockVBridge.OfVStar (ResZeroAtL ReadCol rowCol vBase partRep partBase)
open FlockVBridge.Decode (RegBit sub vOf)
open FlockVBridge.PartS (wIdx)

/-! ## The verifier's names and sizes -/

/-- **The name `rec-residuals/m<m>-k<k>-c<c>-p<j>` determines `m`, `k`, `c` and `j`.** -/
theorem resUnitName_inj {m k c j m' k' c' j' : ℕ}
    (h : RecResiduals.unitName m k c j = RecResiduals.unitName m' k' c' j') : m = m' ∧ k = k' ∧ c = c' ∧ j = j' := by
  have h := congrArg String.toList h
  simp only [RecResiduals.unitName, String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  have h := List.append_cancel_left h
  have dig : ∀ n, ∀ x ∈ Nat.toDigits 10 n, x.isDigit := fun n x hx =>
    Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  obtain ⟨h1, h2⟩ := digits_dash (dig m) (dig m') (by simpa using h)
  simp only [List.cons.injEq, true_and] at h2
  obtain ⟨h3, h4⟩ := digits_dash (dig k) (dig k') (by simpa using h2)
  simp only [List.cons.injEq, true_and] at h4
  obtain ⟨h5, h6⟩ := digits_dash (dig c) (dig c') (by simpa using h4)
  simp only [List.cons.injEq, true_and] at h6
  exact ⟨eq_of_toDigits h1, eq_of_toDigits h3, eq_of_toDigits h5, eq_of_toDigits h6⟩

/-- `params` gives `(m, k, c, j)` only for the name `unitName m k c j`. -/
theorem resUnitName_params {name : String} {m k c j : ℕ} (h : RecResiduals.params name = some (m, k, c, j)) :
    RecResiduals.unitName m k c j = name := by
  unfold RecResiduals.params at h
  repeat' split at h
  all_goals first
    | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl, rfl, rfl⟩ := h; assumption)
    | cases h

theorem startsWith_resUnitName (m k c j : ℕ) :
    (RecResiduals.unitName m k c j).startsWith "rec-residuals/" = true := by
  have : RecResiduals.unitName m k c j = "rec-residuals/" ++ s!"m{m}-k{k}-c{c}-p{j}" := by
    simp [RecResiduals.unitName, ← String.toList_inj, List.append_assoc, toString_str]
  rw [this]
  simp [String.startsWith_string_iff]

/-- **A unit the verifier accepted under the name `unitName m k c j` gives back `(m, k, c, j)`.** -/
theorem resParams_of_check {c : Flock.Circuit} (h : RecResiduals.check c = .ok ()) {m k cl j : ℕ}
    (hn : c.unit.name = RecResiduals.unitName m k cl j) :
    RecResiduals.params c.unit.name = some (m, k, cl, j) := by
  cases hp : RecResiduals.params c.unit.name with
  | some p =>
    obtain ⟨m', k', c', j'⟩ := p
    obtain ⟨rfl, rfl, rfl, rfl⟩ := resUnitName_inj ((resUnitName_params hp).trans hn)
    rfl
  | none =>
    unfold RecResiduals.check at h
    rw [hp] at h
    dsimp only at h
    rw [hn, startsWith_resUnitName] at h
    cases h

/-- A parsed circuit V* is drawn over has a unit per VU: its unit range counts `g · upv` slots, a power of two when the
unit returns the outputs (`scopeOk`). -/
theorem upv_pos {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) (hs : Layout.scopeOk c = true) : 0 < c.upv := by
  have P := parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have ho : c.outNet = c.unitNet := by
    unfold Layout.scopeOk at hs
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hs
    exact hs.1.1.1.2
  have hp := C.unit_pow2 ho
  rw [C.unit_range.2] at hp
  unfold Flock.isPow2 at hp
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
  exact Nat.pos_of_ne_zero fun h0 => by simp [h0] at hp

theorem unitLog_le {c : Flock.Circuit} (hs : Layout.scopeOk c = true) : c.unit.unitLog ≤ 32 := by
  unfold Layout.scopeOk at hs
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hs
  exact hs.1.1.2

theorem foldl_push_all {α β : Type} (P : α → Prop) (f : β → α) (hf : ∀ b, P (f b)) :
    ∀ (l : List β) (init : Array α), (∀ x ∈ init.toList, P x) →
      ∀ x ∈ (l.foldl (fun out b => out.push (f b)) init).toList, P x
  | [], _, h => h
  | b :: l, init, h => by
    rw [List.foldl_cons]
    refine foldl_push_all P f hf l _ fun x hx => ?_
    rw [Array.toList_push, List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | rfl
    · exact h x hx
    · exact hf b

/-- **V*'s statement of part `p` is named as the verifier's unit of part `p`** at V*'s shape (`Stage.lean:63-64`). -/
theorem statements_names {vsh : VStar.Algebra.Shape} {stmts : Array VStar.Stmt} (h : VStar.statements vsh = .ok stmts) :
    ∀ s ∈ stmts.toList, ∀ p nm pt, s = .inner p nm pt → nm = RecResiduals.unitName vsh.m vsh.kLog vsh.claims p := by
  unfold VStar.statements at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind, bind_pure] at h
  cases hs : vsh.sched with
  | error e => simp [hs, Bind.bind, Except.bind] at h
  | ok s =>
    cases hsp : VStar.Algebra.spec vsh with
    | error e => simp [hs, hsp, Bind.bind, Except.bind] at h
    | ok sp =>
      cases hps : VStar.parts sp with
      | error e => simp [hs, hsp, hps, Bind.bind, Except.bind] at h
      | ok ps =>
        simp only [hs, hsp, hps, Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq] at h
        subst h
        refine foldl_push_all _ _ ?_ _ _ (foldl_push_all _ _ ?_ _ _ (by simp))
        all_goals
          intro b p nm pt he
          cases he
          try rfl

theorem statements_name {vsh : VStar.Algebra.Shape} {stmts : Array VStar.Stmt} (h : VStar.statements vsh = .ok stmts)
    (j : Fin stmts.size) {p : ℕ} {nm : String} {pt : VStar.Part} (hj : stmts[j] = .inner p nm pt) :
    nm = RecResiduals.unitName vsh.m vsh.kLog vsh.claims p :=
  statements_names h _ (Array.getElem_mem_toList j.isLt) p nm pt hj

theorem foldl_add_eq : ∀ (l : List ℕ) (a : ℕ), l.foldl (· + ·) a = a + l.sum
  | [], a => by simp
  | x :: l, a => by rw [List.foldl_cons, foldl_add_eq l, List.sum_cons]; omega

theorem foldl_push_lt {f : Array ℕ → ℕ → ℕ} :
    ∀ (l : List ℕ) (init : Array ℕ) {i : ℕ}, i < init.size → (l.foldl (fun s n => s.push (f s n)) init)[i]! = init[i]!
  | [], _, _, _ => rfl
  | x :: l, init, i, hi => by
    rw [List.foldl_cons, foldl_push_lt l (init.push (f init x)) (i := i) (by rw [Array.size_push]; omega),
      getElem!_pos (init.push (f init x)) i (by rw [Array.size_push]; omega), getElem!_pos init i hi,
      Array.getElem_push_lt (xs := init) (x := f init x) hi]

theorem starts_aux : ∀ (l : List ℕ) (init : Array ℕ), 0 < init.size → ∀ i ≤ l.length,
    (l.foldl (fun s n => s.push (s.back! + n)) init)[init.size - 1 + i]! = init.back! + (l.take i).sum
  | [], init, h0, i, hi => by
    obtain rfl : i = 0 := by simpa using hi
    simp [Array.back!]
  | x :: l, init, h0, i, hi => by
    rw [List.foldl_cons]
    cases i with
    | zero =>
      rw [foldl_push_lt l (init.push (init.back! + x)) (i := init.size - 1 + 0) (by rw [Array.size_push]; omega)]
      rw [getElem!_pos (init.push (init.back! + x)) (init.size - 1 + 0) (by rw [Array.size_push]; omega),
        Array.getElem_push_lt (xs := init) (x := init.back! + x) (i := init.size - 1 + 0) (by omega), Array.back!,
        getElem!_pos init (init.size - 1) (by omega)]
      simp
    | succ i =>
      have := starts_aux l (init.push (init.back! + x)) (by simp) i (by simp at hi; omega)
      rw [Array.size_push, show init.size + 1 - 1 + i = init.size - 1 + (i + 1) by omega] at this
      have hb : (init.push (init.back! + x)).back! = init.back! + x := by
        rw [Array.back!, Array.size_push, Nat.add_sub_cancel,
          getElem!_pos (init.push (init.back! + x)) init.size (by simp), Array.getElem_push_eq]
      rw [this, hb, List.take_succ_cons, List.sum_cons]
      omega

/-- `starts sizes` at `p` is the sum of the sizes before `p`. -/
theorem starts_get (sizes : Array ℕ) {p : ℕ} (hp : p ≤ sizes.size) :
    (RecResiduals.starts sizes)[p]! = (sizes.toList.take p).sum := by
  have := starts_aux sizes.toList #[0] (by simp) p (by simpa using hp)
  unfold RecResiduals.starts
  rw [← Array.foldl_toList]
  simpa [Array.back!] using this

/-- **A part's `w` index points into `w`**: below the row ports' total size. -/
theorem wIdx_lt (sizes P : Array ℕ) {a : ℕ} (ha : a < (wIdx sizes P).size) :
    (wIdx sizes P)[a] < sizes.foldl (· + ·) 0 := by
  have key : ∀ x ∈ wIdx sizes P, x < sizes.foldl (· + ·) 0 := by
    intro x hx
    unfold wIdx at hx
    rw [Array.mem_flatMap] at hx
    obtain ⟨p, -, hx⟩ := hx
    rw [Array.mem_map] at hx
    obtain ⟨b, hb, rfl⟩ := hx
    rw [Array.mem_range] at hb
    have hp : p < sizes.size := by
      by_contra hn
      simp [getElem!_neg sizes p hn] at hb
    rw [getElem!_pos sizes p hp] at hb
    rw [starts_get sizes hp.le, ← Array.foldl_toList, foldl_add_eq]
    have h1 := List.sum_take_add_sum_drop sizes.toList (p + 1)
    have h2 : (sizes.toList.take (p + 1)).sum = (sizes.toList.take p).sum + sizes[p] := by
      rw [List.sum_take_succ _ _ (by simpa using hp)]
      simp
    omega
  exact key _ (Array.getElem_mem ha)

/-- A row port's total size is S's `nw`, once `part` accepted. -/
theorem part_nw {sh : Shape} {S : Flock.RecAlgebra.GfS} (hS : structureOf sh = .ok S) {p : ℕ}
    {x : Flock.RecOpen.GfS × List ℕ × ℕ} (h : RecResiduals.part sh p = .ok x) :
    (portSizes sh).foldl (· + ·) 0 = S.nw := by
  unfold RecResiduals.part at h
  obtain ⟨S0, hS0, h⟩ := FlockVBridge.PartS.bind_ok h
  rw [hS] at hS0
  cases hS0
  dsimp only at h
  split at h
  · cases h
  · rename_i hne
    simpa [bne_iff_ne] using hne

/-- **`verifierRows`' `v` has S's `nv` elements** when the comb has 64: the coefficient slots, then the comb's. -/
theorem build_nv {sh : Shape} {lins : Array Flock.RecAlgebra.Lin} {b : Flock.RecAlgebra.Built}
    (hb : Flock.RecAlgebra.build sh lins = .ok b) : b.S.nv = b.vVals.size + 64 := by
  unfold Flock.RecAlgebra.build at hb
  generalize hK : sh.sched.levels.map (fun lv => Array.replicate 2 (Array.replicate (2 ^ lv.lanes) (0 : F128))) = K
    at hb
  generalize hc0 : (lins.toList.map Flock.RecAlgebra.slotsOf).sum = comb0 at hb
  simp only at hb
  cases hB : (Flock.RecAlgebra.buildAll sh (2 + sh.claims) comb0 lins.toList 0).run { kappa := K } with
  | error e => rw [hB] at hb; cases hb
  | ok p =>
    obtain ⟨res, st⟩ := p
    rw [hB] at hb
    simp only [Bind.bind, Except.bind] at hb
    split at hb
    · cases hb
    · rename_i hne
      cases hb
      simp only [not_or, not_not, ne_eq] at hne
      simp only [Array.size_append, Array.size_map, Array.size_range]
      omega

theorem verifierRows_v_size {sh : Shape} {GS : Flock.RecAlgebra.GfS} (hS : structureOf sh = .ok GS)
    {rounds : Array (Array F128)} {extras : Array Claim} {comb v : Array F128} {info : Flock.RecAlgebra.Info Flock.RecAlgebra.Lin}
    {kappa : Array (Array (Array F128))} (h : verifierRows sh rounds extras comb = .ok (v, info, kappa))
    (hc : comb.size = 64) : v.size = GS.nv := by
  unfold verifierRows at h
  cases hl : Flock.RecAlgebra.residualLins sh (Flock.RecAlgebra.Coins.ofRounds rounds) extras with
  | error e => rw [hl] at h; cases h
  | ok q =>
    obtain ⟨lins, info'⟩ := q
    rw [hl] at h
    simp only [Bind.bind, Except.bind] at h
    cases hb : Flock.RecAlgebra.build sh lins with
    | error e => rw [hb] at h; cases h
    | ok b =>
      rw [hb] at h
      simp only at h
      split at h
      · cases h
      · rename_i hne
        cases h
        rw [hS] at hne
        simp only [Except.toOption, ne_eq, Option.some.injEq, not_not] at hne
        rw [Array.size_append, hc, ← hne, build_nv hb]

/-! ## The parts and the decoders -/

/-- The verifier's parts at a shape (`groups` of S), none where it refuses. -/
def partsOf (sh : Shape) : Array (Array ℕ × Array ℕ × Array ℕ) :=
  match structureOf sh with
  | .ok S => match groups S (portSizes sh) with
    | .ok gs => gs
    | .error _ => #[]
  | .error _ => #[]

/-- S's `nw` at a shape, zero where the verifier refuses. -/
def nwOf (sh : Shape) : ℕ :=
  match structureOf sh with
  | .ok S => S.nw
  | .error _ => 0

/-- Part `p`'s `w` index (`wIdx` of its row ports). -/
def wIOf (sh : Shape) (p : ℕ) : Array ℕ := wIdx (portSizes sh) (partsOf sh)[p]!.2.1

/-- Part `p`'s row ports' sizes. -/
def nsOf (sh : Shape) (p : ℕ) : List ℕ := (partsOf sh)[p]!.2.1.toList.map ((portSizes sh)[·]!)

/-- Part `p`'s number of `v` elements. -/
def nvOf (sh : Shape) (p : ℕ) : ℕ := (partsOf sh)[p]!.2.2.size

/-- The input column of part `p`'s unit holding bit `q` of its row ports. -/
def rowColOf (sh : Shape) (p q : ℕ) : ℕ := rowCol (partRep (nsOf sh p) (nvOf sh p)).ports 0 q

theorem partsOf_eq {sh : Shape} {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf sh = .ok GS) (hg : groups GS (portSizes sh) = .ok gs) : partsOf sh = gs := by
  unfold partsOf
  rw [hS]
  dsimp only
  rw [hg]

theorem nwOf_eq {sh : Shape} {GS : Flock.RecAlgebra.GfS} (hS : structureOf sh = .ok GS) : nwOf sh = GS.nw := by
  unfold nwOf
  rw [hS]

/-- Copy 0 of unit `u` of statement `Q` reads input column `col` at a row: the column's wire is one of the unit's. -/
def IoColAt (Q : VStmt) (u : Fin Q.nv) (col : ℕ) : Prop :=
  ∀ (hj : 0 < Q.c.upv) (hc : col < (Rows.ofNet Q.hU).w),
    (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u).wire ((Rows.ofNet Q.hU).stackCol Q.c.upv hj ⟨col, hc⟩) ∈
      ((Placed.keyProg Q.c Q.hU Q.nv).partition (Layout.outsOf Q.c Q.hU Q.nv)).io u

/-- The bit a layer holds at column `col` of copy 0 of unit `u` of statement `Q`. -/
noncomputable def bitAt (Q : VStmt) (bits : Q.Bits) (u : Fin Q.nv) (col : ℕ) : Bool :=
  if hj : 0 < Q.c.upv then copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj col else false

/-- **A registered bit a row reads is the layer's**: in a layer agreeing with the registrant, copy 0's column at a
registered bit `b` (`RegBit`) whose wire is one of the unit's holds `b`. -/
theorem bitAt_of_regBit {Q : VStmt} {Reg' : Type} (zo : Q.Outer Reg') {bits : Q.Bits} (hA : zo.RegAgree bits)
    {u : Fin Q.nv} {col : ℕ} {b : Bool} (hb : RegBit zo u 0 col b) (hio : IoColAt Q u col) (hj : 0 < Q.c.upv) :
    bitAt Q bits u col = b := by
  unfold bitAt
  rw [dite_eq_left hj]
  have h1 := FlockVBridge.Decode.copyVal_regBit zo bits hj hb
  obtain ⟨hcol, -⟩ := hb
  rw [← h1]
  refine (copyVal_input _ _ hj hcol).trans (Eq.trans ?_ (copyVal_input _ _ hj hcol).symm)
  exact (FlockVBridge.Decode.fill_io zo hA (hio hj _)).symm

/-- **A column a computed row reads is read at copy 0**: copy 0's computed row `i` of the stack reads the copy's columns
of row `i` (`stack_a`, `stack_b`), and an input column is no gate of the unit (`outside`), so its wire is one of the
unit's inputs (`mem_io_of_read`). -/
theorem ioColAt_of_readCol {Q : VStmt} (u : Fin Q.nv) {col : ℕ} (hc : col < (Rows.ofNet Q.hU).nIn)
    (h : ReadCol (Rows.ofNet Q.hU) col) : IoColAt Q u col := by
  obtain ⟨i, x, hx, rfl⟩ := h
  intro hj hcw
  refine (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u).mem_io_of_read
    (i := ⟨0 * ((Rows.ofNet Q.hU).nComp + 1) + (1 + i.val), Rows.mul_add_lt hj (by omega)⟩) ?_ ?_
  · rw [stack_a _ _ hj i, stack_b _ _ hj i]
    rcases List.mem_append.1 hx with hx | hx
    · exact List.mem_append_left _ (List.mem_map_of_mem hx)
    · exact List.mem_append_right _ (List.mem_map_of_mem hx)
  · refine (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u).outside _ fun i' h' => ?_
    have he := congrArg Fin.val h'
    have hle : (Rows.ofNet Q.hU).nIn ≤ Q.c.upv * (Rows.ofNet Q.hU).nIn := Nat.le_mul_of_pos_left _ hj
    rw [Rows.stackCol, dite_eq_left hc] at he
    simp only [Rows.comp, Rows.stack] at he
    omega

/-- The circuit fold the lincheck computes at a rep's coins (`Piop.lean:143-146`), 64 zeros where it gives no 64
elements. -/
def combOf (sh : Shape) {cls : HClass} (TF : TableFiles cls) (rounds : Array (Array F128)) : Array F128 :=
  let c : Array F128 :=
    match Flock.RecAlgebra.residualLins sh (Flock.RecAlgebra.Coins.ofRounds rounds) TF.extras with
    | .ok (_, info) =>
      match TF.st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn
          info.lc.fold.ts with
      | .ok v => v
      | .error _ => #[]
    | .error _ => #[]
  if c.size = 64 then c else Array.replicate 64 0

theorem pad_size (c : Array F128) : (if c.size = 64 then c else Array.replicate 64 0).size = 64 := by
  split
  · assumption
  · simp

theorem combOf_size (sh : Shape) {cls : HClass} (TF : TableFiles cls) (rounds : Array (Array F128)) :
    (combOf sh TF rounds).size = 64 :=
  pad_size _

/-- Rep `r`'s coins in a table's files: its stream's rounds' coins. -/
def roundsOf {cls : HClass} (TF : TableFiles cls) (r : Fin 2) : Array (Array F128) :=
  (TF.stream r.val).rounds.map (·.coins)

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- Column `col` of rep `r`'s copy of part `p` is read by a row of its unit. -/
def IoCol (p : ℕ) (r : Fin 2) (col : ℕ) : Prop := IoColAt (F.Vs (F.sess p)) (F.un p r) col

/-- The layer's bit at column `col` of rep `r`'s copy of part `p`. -/
noncomputable def colBit (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) (p col : ℕ) : Bool :=
  bitAt (F.Vs (F.sess p)) (l (F.sess p)) (F.un p r) col

/-- Bit `e` of element `i` of `w` is at position `a` of part `p`'s `w` index, in a column rep `r`'s copy reads. -/
def Seen (r : Fin 2) (i e : ℕ) (pk : ℕ × ℕ) : Prop :=
  pk.1 < (partsOf F.sh).size ∧ ∃ hk : pk.2 < (wIOf F.sh pk.1).size, (wIOf F.sh pk.1)[pk.2] = i ∧
    F.IoCol pk.1 r (rowColOf F.sh pk.1 (128 * pk.2 + e))

open Classical in
/-- Where rep `r` reads bit `e` of element `i` of `w`: a part and position some row reads it at. -/
noncomputable def locOf (r : Fin 2) (i e : ℕ) : ℕ × ℕ := if h : ∃ pk, F.Seen r i e pk then Classical.choose h else (0, 0)

theorem locOf_spec {r : Fin 2} {i e : ℕ} (h : ∃ pk, F.Seen r i e pk) : F.Seen r i e (F.locOf r i e) := by
  unfold locOf
  rw [dite_eq_left h]
  exact Classical.choose_spec h

/-- **Rep `r`'s `w` in V*'s layers**: element `i`'s bit `e` is the layer's at the column `locOf` reads it at. -/
noncomputable def wOf (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) : Array F128 :=
  Array.ofFn (n := nwOf F.sh) fun i => ofBits fun e =>
    F.colBit l r (F.locOf r i.val e).1 (rowColOf F.sh (F.locOf r i.val e).1 (128 * (F.locOf r i.val e).2 + e))

theorem el_wOf (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) {i : ℕ} (hi : i < nwOf F.sh) :
    el (F.wOf l r) i = ofBits fun e =>
      F.colBit l r (F.locOf r i e).1 (rowColOf F.sh (F.locOf r i e).1 (128 * (F.locOf r i e).2 + e)) := by
  have hs : i < (F.wOf l r).size := by rw [wOf, Array.size_ofFn]; exact hi
  unfold el
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hs, Option.getD_some]
  simp only [wOf, Array.getElem_ofFn]

/-- **`rep₀` from V*'s layers and a table's files**: `w` from the wires (`wOf`), the coins and extra claims from the
files (`roundsOf`, `TableFiles.extras`), and the comb at those coins (`combOf`). -/
noncomputable def repOf (tf : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → TableFiles cls) :
    Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → RepData :=
  fun R ω t l r =>
    { w := F.wOf l r, comb := combOf F.sh (tf R ω t l) (roundsOf (tf R ω t l) r), cons := fun _ _ => 0,
      rounds := roundsOf (tf R ω t l) r, extras := (tf R ω t l).extras }

/-! ## What the proofs take -/

/-- **Some statement is a part's.** `VStar.statements` lists one `InnerRepCheck` per part of `parts (spec vsh)`
(`Stage.lean:63-64`); established by evaluating `statements` at the family's `vsh`, or by any accepted V* proof's part
sessions. -/
def HSomePart : Prop := ∃ (j : Fin F.stmts.size) (p : ℕ) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt

/-- **Every verifier part has V*'s statement.** V*'s parts (`rec_residuals.parts`, `VStar.parts`) are the verifier's
groups (`RecResiduals.groups`) at the shape; established by a lemma `VStar.parts (spec vsh)` = `groups` (V*'s algebra
lane), or by a verifier check that a session per part `p<j>` of `groups` is present. -/
def HCover : Prop :=
  ∀ p < (partsOf F.sh).size, ∃ (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt

/-- Obligation: discharged by `circuitFile`'s emit and parse round trip.
**The circuit of a part's statement names its unit as the statement.** `circuitFile` writes the unit of
`innerUnit S nm` under `nm` (`metaHead`, `emitSections`, `Compose.lean:217,233`) and `HmRow.parse` reads it as the
circuit's unit (`hfile`); established by the emit/parse round trip of `circuitFile` (the lane that proves `hfile`). -/
def HUnitName : Prop :=
  ∀ (j : Fin F.stmts.size) p nm pt, F.stmts[j] = .inner p nm pt → (F.Vs j).c.unit.name = nm

variable (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- **Each part's row-port columns are registered reads.** At an accepted outer proof, every row-port column of rep
`r`'s copy of part `p` has the Δ key of a bit of a registered row (`RegBit`): V* registers each part's row ports as reads
of `regs[nm]`'s row `r` (`rec_vstage.py:356-357`); established by the session's own reads (`ZkOuter.hown`, `OwnReads`)
at V*'s registered ports, with the keys of the unit's inputs (`Aliased.srcKey`). -/
def HRowKeys : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r p, p < (partsOf F.sh).size →
    ∀ q < 128 * (nsOf F.sh p).sum, ∃ b, RegBit (F.zs R ω t R₂ (F.sess p)) (F.un p r) 0 (rowColOf F.sh p q) b

/-- **Two parts read the same registered bit of an element of `w`.** At an accepted outer proof, where parts `p` and
`p'` hold the same element of `w` at positions `a` and `a'`, the registrant's bit `e` there is the same in both sessions:
both read row `r` of the same registered row (`rec_vstage.py:356-357`); established by V*'s registrations of
`regs[nm]` across the part sessions (the lane that proves `HCommits`). -/
def HRowsAgree : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (r : Fin 2) (p p' a a' e : ℕ) (b : Bool),
    p < (partsOf F.sh).size → p' < (partsOf F.sh).size → ∀ (ha : a < (wIOf F.sh p).size)
    (ha' : a' < (wIOf F.sh p').size), (wIOf F.sh p)[a] = (wIOf F.sh p')[a'] → e < 128 →
    RegBit (F.zs R ω t R₂ (F.sess p)) (F.un p r) 0 (rowColOf F.sh p (128 * a + e)) b →
      RegBit (F.zs R ω t R₂ (F.sess p')) (F.un p' r) 0 (rowColOf F.sh p' (128 * a' + e)) b

/-- **The unit's constant is 1.** At an accepted outer proof, the layer carries 1 on the constant gate of rep `r`'s
instance of part `p`; established by the pinned constant of an accepted V* session (as `OpensAt`'s for the openings). -/
def HOne : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r p, p < (partsOf F.sh).size →
    layers F.zs dg N k Rw R ω t R₂ τ (F.sess p)
      (Const.Prog.oneAt (Placed.keyProg (F.Vs (F.sess p)).c (F.Vs (F.sess p)).hU (F.Vs (F.sess p)).nv) (F.un p r)) = true

/-- Obligation: discharged by check 5's `RecCoins.checkResiduals` (in the event as `ChecksC3`'s `RecCoins.check`),
which holds the `v` port's registration to `verifierRows` at the inner session's coins, extra claims and comb
(`checkValue` at `rec-v`).
**The registered `v` is the verifier's at the rep's coins.** At an accepted outer proof, the `v` port of rep `r`'s copy
of part `p` is registered bits of `verifierRows`' coefficients at the rep's coins, extra claims and comb (`vOf`), at the
part's `V`; established from that check's rows (the registrant's rows at `rec-v`'s root, as `AccRowsAre` reads
`rec-acc`'s) and the inner session's coins being the rep's. -/
def HVCoins : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r p, p < (partsOf F.sh).size →
    ∀ q < 128 * nvOf F.sh p,
      RegBit (F.zs R ω t R₂ (F.sess p)) (F.un p r) 0 (vBase (partRep (nsOf F.sh p) (nvOf F.sh p)) + q)
        (FlockVBridge.wbit (sub (partsOf F.sh)[p]!.2.2
          (vOf F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).rounds
            (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).extras
            (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb)) q)

/-- Obligation: to be discharged by acceptance, now that the layout keys a returned output gate to its output row
(`HiddenExec.rowOfH`), which makes `RegOut` hold at the parts' `out` columns where they are registered reads.
**The parts' outputs are 0 in the layer.** At an accepted outer proof, every output column of rep `r`'s copy of
part `p` is 0 in the layer (`copyVal`). V* registers each part's `out` as reads of the zero row `rec-residuals-zero`
(`rec_vstage.py`), and `ZkOuter.Good` asks the layer to agree with the registrant at registered reads only
(`RegAgree`), so acceptance (`Accepted`) reaches these columns only through those reads. -/
def HZeroL : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r p, p < (partsOf F.sh).size →
    ∀ S ns nv, RecResiduals.part F.sh p = .ok (S, ns, nv) → ∀ hj : cp p r < (F.Vs (F.sess p)).c.upv,
      ∀ e < 16 * RecResiduals.RES_OUT_WORDS,
        copyVal (keyInst (F.Vs (F.sess p)).hU (F.Vs (F.sess p)).nv
            (Layout.outsOf (F.Vs (F.sess p)).c (F.Vs (F.sess p)).hU (F.Vs (F.sess p)).nv) (F.un p r))
          (layers F.zs dg N k Rw R ω t R₂ τ (F.sess p)) hj (partBase S ns nv + e) = false

/-- **The rep's coins fit the verifier's schedule.** At an accepted outer proof, `verifierRows` accepts each rep's
coins and extra claims: its rounds have the sizes the shape's schedule draws and there are `claims - 2` extra claims
(`verifierRows_schedule`); established from the table's files (`TableFiles.Valid`: the stream's `RepTape` and the
session's extra claims, the lane that proves `HFiles`). -/
def HFit : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r,
    ∃ x, verifierRows F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).rounds
      (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).extras
      (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb = .ok x

/-! ## The obligations -/

/-- **`HStruct` from a part's statement**: the verifier parses each session's circuit and checks it
(`RecResiduals.check`, `ParseFacts.recResiduals`); a part's unit is named `rec-residuals/m…-p<p>` at V*'s shape
(`statements_name`, `HUnitName`), whose verifier shape is `sh` (`hsh`), so `hStruct_of_check` applies. -/
theorem hStruct_of (hne : Nonempty (Reg × L.Ω × Reg₂)) (hsome : F.HSomePart) (hname : F.HUnitName) :
    ∃ GS gs, F.HStruct GS gs := by
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  obtain ⟨j, p, nm, pt, hj⟩ := hsome
  have hchk := (parse_facts (F.zs R ω [] R₂ j).hc).recResiduals
  have hnm : (F.Vs j).c.unit.name = RecResiduals.unitName F.vsh.m F.vsh.kLog F.vsh.claims p :=
    (hname j p nm pt hj).trans (statements_name F.hstmts j hj)
  exact F.hStruct_of_check hchk (resParams_of_check hchk hnm) F.hsh (F.hplain j)

/-- **`HShape` at `repOf`**: the verifier rows at the coins (`HFit`) have S's `nv` elements, since the comb has 64
(`combOf_size`, `verifierRows_v_size`), and `w` has S's `nw` (`wOf`). -/
theorem hShape_of {tf : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → TableFiles cls}
    (hrep : F.rep₀ = F.repOf tf) {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hst : F.HStruct GS gs) (hfit : F.HFit dg N k Rw) : F.HShape dg N k Rw GS := by
  intro R ω t R₂ τ o h r
  have hc : (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb.size = 64 := by
    show (F.rep₀ R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb.size = 64
    rw [hrep]
    exact combOf_size _ _ _
  obtain ⟨⟨v, info, kappa⟩, hx⟩ := hfit R ω t R₂ τ o h r
  refine ⟨⟨v, info, kappa, hx, verifierRows_v_size hst.1 hx hc⟩, ?_, hc⟩
  show (F.rep₀ R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w.size = GS.nw
  rw [hrep]
  show (F.wOf _ r).size = GS.nw
  rw [wOf, Array.size_ofFn, nwOf_eq hst.1]

/-- **`hres`, its row ports at the read columns and its outputs in the layer**: at every accepted outer proof, rep
`r`'s instance of part `p` holds part `p` at the rep's values, its outputs 0 in the layer (`ResZeroAtL`). -/
def HResL (gs : Array (Array ℕ × Array ℕ × Array ℕ)) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ p < gs.size,
    ResZeroAtL (F.zs R ω t R₂ (F.sess p)) (layers F.zs dg N k Rw R ω t R₂ τ (F.sess p)) (F.un p r) (cp p r) F.sh gs p
      (el (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w)
      (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).rounds (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).extras
      (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb

/-- **`HResL` at `repOf`'s `w`**: rep `r`'s copy of part `p` is the verifier's part `p` (`ParseFacts.recResiduals`,
`HUnitName`, `check_part`), its unit's constant 1 (`HOne`), each row-port column a computed row reads a registered bit
of the decoded `w` (the layer's bit there, `ioColAt_of_readCol`, `bitAt_of_regBit`, carried to every part by
`HRowsAgree`), its `v` the verifier's (`HVCoins`) and its outputs 0 in the layer (`HZeroL`). -/
theorem hResL_of {tf : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → TableFiles cls}
    (hrep : F.rep₀ = F.repOf tf) {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hst : F.HStruct GS gs) (hcov : F.HCover) (hname : F.HUnitName) (hkeys : F.HRowKeys dg N k Rw)
    (hagree : F.HRowsAgree dg N k Rw) (hone : F.HOne dg N k Rw) (hv : F.HVCoins dg N k Rw)
    (hz : F.HZeroL dg N k Rw) : F.HResL dg N k Rw gs := by
  intro R ω t R₂ τ o h r p hp
  have hP : partsOf F.sh = gs := partsOf_eq hst.1 hst.2
  have hpP : p < (partsOf F.sh).size := by rw [hP]; exact hp
  obtain ⟨j, nm, pt, hj⟩ := hcov p hpP
  have hsj : F.sess p = j := F.hsess p j ⟨nm, pt, hj⟩
  have hjs : F.stmts[F.sess p] = .inner p nm pt := by subst hsj; exact hj
  have hnm : (F.Vs (F.sess p)).c.unit.name = RecResiduals.unitName F.vsh.m F.vsh.kLog F.vsh.claims p :=
    (hname _ p nm pt hjs).trans (statements_name F.hstmts _ hjs)
  have hchk : RecResiduals.check (F.Vs (F.sess p)).c = .ok () :=
    (parse_facts (F.zs R ω t R₂ (F.sess p)).hc).recResiduals
  have hpar := resParams_of_check hchk hnm
  have hupv : ∀ p', 0 < (F.Vs (F.sess p')).c.upv := fun p' =>
    upv_pos (F.zs R ω t R₂ (F.sess p')).hc (F.Vs (F.sess p')).hscope
  obtain ⟨S, ns, nv, hpart, -⟩ := FlockVBridge.OfVStar.check_part hchk hpar F.hsh (F.hplain _)
  obtain ⟨-, hns, hnv⟩ := FlockVBridge.OfVStar.part_eq hst.1 hst.2 hp hpart
  have hgp : (partsOf F.sh)[p]! = gs[p] := by rw [hP, getElem!_pos gs p hp]
  have hnsp : nsOf F.sh p = ns := by rw [nsOf, hgp, hns]
  have hnvp : nvOf F.sh p = nv := by rw [nvOf, hgp, hnv]
  have hwI : wIOf F.sh p = wIdx (portSizes F.sh) gs[p].2.1 := by rw [wIOf, hgp]
  have hW : (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w = F.wOf (layers F.zs dg N k Rw R ω t R₂ τ) r := by
    show (F.rep₀ R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w = _
    rw [hrep]
    rfl
  refine ⟨hp, hupv p, hone R ω t R₂ τ o h r p hpP, hchk, F.hplain _, unitLog_le (F.Vs (F.sess p)).hscope, _, _, _,
    hpar, F.hsh, S, ns, nv, hpart, ?_, ?_, ?_⟩
  · intro q hq hrd
    show RegBit (F.zs R ω t R₂ (F.sess p)) (F.un p r) 0 _ _
    rw [hW, ← hwI]
    have hsz : (wIOf F.sh p).size = ns.sum := by rw [hwI, FlockVBridge.OfVStar.wIdx_size, hns]
    have ha : q / 128 < (wIOf F.sh p).size := by rw [hsz]; omega
    have he : q % 128 < 128 := Nat.mod_lt _ (by norm_num)
    have hi : (wIOf F.sh p)[q / 128] < nwOf F.sh := by
      rw [nwOf_eq hst.1, ← part_nw hst.1 hpart]
      exact wIdx_lt (portSizes F.sh) ((partsOf F.sh)[p]!.2.1) ha
    obtain ⟨-, hc1, -⟩ := hkeys R ω t R₂ τ o h r p hpP q (by rw [hnsp]; exact hq)
    have hio : F.IoCol p r (rowColOf F.sh p (128 * (q / 128) + q % 128)) := by
      show IoColAt (F.Vs (F.sess p)) (F.un p r) _
      rw [Nat.div_add_mod q 128]
      refine ioColAt_of_readCol _ hc1 ?_
      rw [rowColOf, hnsp, hnvp]
      exact hrd
    have hex : ∃ pk, F.Seen r (wIOf F.sh p)[q / 128] (q % 128) pk := ⟨(p, q / 128), hpP, ha, rfl, hio⟩
    obtain ⟨hp0, ha0, hi0, hio0⟩ := F.locOf_spec hex
    have hsz0 : (wIOf F.sh (F.locOf r (wIOf F.sh p)[q / 128] (q % 128)).1).size =
        (nsOf F.sh (F.locOf r (wIOf F.sh p)[q / 128] (q % 128)).1).sum := by
      rw [wIOf, nsOf, FlockVBridge.OfVStar.wIdx_size]
    obtain ⟨b0, hb0⟩ := hkeys R ω t R₂ τ o h r _ hp0
      (128 * (F.locOf r (wIOf F.sh p)[q / 128] (q % 128)).2 + q % 128) (by rw [← hsz0]; omega)
    have hdec := bitAt_of_regBit (F.zs R ω t R₂ _) (h.1 _).2.2 hb0 hio0 (hupv _)
    have hb := hagree R ω t R₂ τ o h r _ p _ (q / 128) (q % 128) b0 hp0 hpP ha0 ha hi0 he hb0
    have hval : FlockVBridge.wbit (sub (wIOf F.sh p) (el (F.wOf (layers F.zs dg N k Rw R ω t R₂ τ) r))) q = b0 := by
      unfold FlockVBridge.wbit
      rw [FlockVBridge.Decode.sub_lt _ _ _ ha, F.el_wOf _ r hi, FlockVBridge.bitOf_ofBits _ he]
      exact hdec
    rw [hval]
    have hq : 128 * (q / 128) + q % 128 = q := Nat.div_add_mod q 128
    rw [hq, rowColOf, hnsp, hnvp] at hb
    exact hb
  · intro q hq
    have hq' : q < 128 * nvOf F.sh p := by rw [hnvp]; exact hq
    have := hv R ω t R₂ τ o h r p hpP q hq'
    rw [hnsp, hnvp, hgp] at this
    exact this
  · intro e he
    exact hz R ω t R₂ τ o h r p hpP S ns nv hpart (hupv p) e he

end VStarFamily

end family

end FlockVBridge.Family
