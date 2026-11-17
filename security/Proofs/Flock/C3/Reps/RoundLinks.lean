module

public import Proofs.Flock.C3.Compose.HoloEncoded
public import Proofs.Flock.C3.Compose.RsFacts
public import Proofs.Flock.C3.Reps.SentKeys
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.C3.VStarList.Shape
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.VBridge.Pinned
import Std.Data.String.ToNat

/-! The round reads' links to the verifier and the map, at a circuit port named `m<i>`. -/

@[expose] public section

namespace FlockC3.Reps

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open Flock.RecVStar Flock.Guarantees FlockC3.Compose

theorem index_pre (p : String) (k : ℕ) {n : ℕ} (hn : p.length = n) :
    (if (p ++ toString k).startsWith p = true then ((p ++ toString k).drop n).toString.toNat? else none) = some k := by
  have hs : (p ++ toString k).startsWith p = true := by
    rw [String.startsWith_string_iff]
    exact ⟨(toString k).toList, by simp⟩
  have hd : ((p ++ toString k).drop n).toString = toString k := by
    apply String.toList_inj.1
    rw [String.Slice.toString_eq, String.toList_copy_drop, String.toList_append, ← hn, ← String.length_toList,
      List.drop_left]
  simp only [hs, ↓reduceIte, hd]
  exact Nat.toNat?_repr k

theorem mIndex_m (k : ℕ) : mIndex s!"m{k}" = some k := index_pre "m" k rfl

theorem cIndex_c (k : ℕ) : cIndex s!"rec-c{k}" = some k := index_pre "rec-c" k rfl

theorem openingIndex_c (k : ℕ) : openingIndex s!"rec-opening-c{k}" = some k := index_pre "rec-opening-c" k rfl

theorem cIndex_opening (k : ℕ) : cIndex s!"rec-opening-c{k}" = none := by
  have hn : ¬ (s!"rec-opening-c{k}".startsWith "rec-c" = true) := fun h => by
    obtain ⟨t, ht⟩ := String.startsWith_string_iff.1 h
    have := congrArg (List.take 5) ht
    simp only [String.toList_append] at this
    rw [List.take_append_of_le_length (by decide), List.take_append_of_le_length (by decide)] at this
    exact absurd this (by decide)
  unfold cIndex
  simp only [hn, Bool.false_eq_true, ↓reduceIte]

theorem inner_round_lt_of_checkMsgNames
    {p m kLog claims partNo i : ℕ} {name : String} {part : VStar.Part} {c : Flock.Circuit} {holo : Bool}
    (hp : RecResiduals.params name = some (m, kLog, claims, partNo))
    (h : checkMsgNames (.inner p name part) c holo = .ok ())
    {q : Fin c.ports.size} (hq : c.ports[q].name = s!"m{i}") :
    ∃ rs, (if holo then holoRepRounds else repRounds) m kLog claims = some rs ∧ i < (commitRounds rs).length := by
  generalize hB : (((if holo then holoRepRounds else repRounds) m kLog claims).map
    fun rs => (commitRounds rs).length).getD 0 = B
  have hr : innerMsgRounds (.inner p name part) holo = some B := by
    simp [innerMsgRounds, hp, hB]
  have hn : msgRoundOutOfRange B c = none := by
    cases hx : msgRoundOutOfRange B c with
    | none => rfl
    | some nm =>
      simp only [checkMsgNames] at h
      split at h
      · cases h
      · split at h
        · cases h
        · simp [hr, hx] at h
  have hmem : s!"m{i}" ∈ c.ports.toList.map (·.name) :=
    List.mem_map.2 ⟨c.ports[q], Array.mem_toList_iff.2 (Array.getElem_mem q.isLt), hq⟩
  have hi := List.find?_eq_none.1 hn _ hmem
  rw [mIndex_m] at hi
  have hi' : ¬ B ≤ i := by
    intro hle
    apply hi
    simp [hle]
  cases hrs : (if holo then holoRepRounds else repRounds) m kLog claims with
  | none => rw [hrs] at hB; simp at hB; omega
  | some rs => rw [hrs] at hB; exact ⟨rs, rfl, by simp at hB; omega⟩

theorem ownName_of_msgNames {s : VStar.Stmt} {c : Flock.Circuit} {holo : Bool}
    (hs : (∃ j nm p, s = .inner j nm p) ∨ ∃ m cl j S ps nv, s = .opening m cl j S ps nv)
    (h : checkMsgNames s c holo = .ok ()) {q : Fin c.ports.size} {k : ℕ}
    (hq : c.ports[q].name = s!"m{k}") : ∃ e ∈ ownPorts s, e.1 = s!"m{k}" := by
  have hn : msgUnnamed s c = none := by
    rcases hs with ⟨j, nm, p, rfl⟩ | ⟨m, cl, j, S, ps, nv, rfl⟩ <;>
    · cases hm : msgUnnamed _ c with
      | none => rfl
      | some x =>
        simp only [checkMsgNames, hm] at h
        split at h <;> cases h
  have hmem : s!"m{k}" ∈ c.ports.toList.map (·.name) :=
    List.mem_map.2 ⟨c.ports[q], Array.mem_toList_iff.2 (Array.getElem_mem q.isLt), hq⟩
  have hp := List.find?_eq_none.1 hn _ hmem
  rw [mIndex_m] at hp
  have hany : (ownPorts s).any (fun x => x.1 == s!"m{k}") = true := by
    cases ha : (ownPorts s).any (fun x => x.1 == s!"m{k}")
    · rw [ha] at hp
      exact absurd rfl hp
    · rfl
  obtain ⟨e, he, he1⟩ := List.any_eq_true.1 hany
  exact ⟨e, he, beq_iff_eq.1 he1⟩

theorem inner_port_of_msgNames {j : ℕ} {nm : String} {p : VStar.Part} {c : Flock.Circuit} {holo : Bool}
    (h : checkMsgNames (.inner j nm p) c holo = .ok ()) {q : Fin c.ports.size} {k : ℕ} (hq : c.ports[q].name = s!"m{k}") :
    (s!"m{k}", s!"rec-c{k}", #[some 0, some 1]) ∈ ownPorts (.inner j nm p) := by
  obtain ⟨e, he, he1⟩ := ownName_of_msgNames (.inl ⟨_, _, _, rfl⟩) h hq
  have he' := he
  unfold ownPorts at he'
  obtain ⟨⟨nm', n'⟩, -, hae⟩ := List.mem_filterMap.1 he'
  dsimp only at hae
  cases hk : mIndex nm' with
  | none => rw [hk] at hae; cases hae
  | some k' =>
    rw [hk] at hae
    cases hae
    have ha1 : nm' = s!"m{k}" := he1
    rw [ha1, mIndex_m] at hk
    obtain rfl := Option.some.inj hk
    rw [← ha1]
    exact he

theorem opening_port_of_msgNames {m cl j nv : ℕ} {S : Flock.RecOpen.GfS} {ps : List (String × ℕ)} {c : Flock.Circuit}
    {holo : Bool} (h : checkMsgNames (.opening m cl j S ps nv) c holo = .ok ()) {q : Fin c.ports.size} {k : ℕ}
    (hq : c.ports[q].name = s!"m{k}") :
    (s!"m{k}", s!"rec-opening-c{k}", #[some 0]) ∈ ownPorts (.opening m cl j S ps nv) := by
  obtain ⟨e, he, he1⟩ := ownName_of_msgNames (.inr ⟨_, _, _, _, _, _, rfl⟩) h hq
  unfold ownPorts at he ⊢
  obtain ⟨⟨nm', n'⟩, ha, hae⟩ := List.mem_filterMap.1 he
  have hnm : nm' = s!"m{k}" := by
    rw [← he1]
    dsimp only at hae
    split at hae <;> (cases hae <;> rfl)
  subst hnm
  refine List.mem_filterMap.2 ⟨_, ha, ?_⟩
  dsimp only
  rw [mIndex_m]

theorem opening_of_name {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    {s : VStar.Stmt} (hs : s ∈ ss) {m c j : ℕ} (hn : RecOpening.unitName m c j = s.name) :
    ∃ m' c' j' S' ps nv, s = .opening m' c' j' S' ps nv := by
  obtain ⟨n1, n2, n3, n4, n5⟩ := openingName_ne m c j
  obtain ⟨vsh, A, C, K, L, N, hA, hC, hK, -, hL, -, rfl, -⟩ := FlockC3.VStarList.statements_ok h
  rcases Array.mem_append.1 hs with hs | hs
  rotate_left
  · rcases FlockC3.VStarList.tail_cases hK hL N _ hs with ⟨_, _, rfl⟩ | ⟨_, _, _, rfl⟩
    · exact absurd hn (n5 _ _)
    · exact absurd hn (openingName_ne_link _ _ _ _ _)
  rcases Array.mem_append.1 hs with hs | hs
  · obtain ⟨_, _, _, -, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
    simp only [Array.mem_append, List.mem_toArray, List.mem_map, List.mem_range] at hs
    rcases hs with ⟨l, -, rfl⟩ | ⟨j', -, rfl⟩
    · exact absurd hn (n1 _ _)
    · exact absurd hn (n2 _ _ _ _)
  · cases hsp : i.sparse with
    | none =>
      rw [(FlockC3.VStarList.candidate3_ok hC).1 hsp] at hs
      simp at hs
    | some sp =>
      obtain ⟨rest, rfl, hrest⟩ := FlockC3.PubRows.candidate3_rest hC hsp
      rcases Array.mem_append.1 hs with hs | hs
      · simp only [List.mem_toArray, List.mem_singleton] at hs
        subst hs
        exact absurd hn (n4 _)
      · rcases hrest s hs with ⟨l, L, H, n', rfl⟩ | ⟨m', c', k, S', ports, nv, rfl⟩
        · exact absurd hn (n3 _ _)
        · exact ⟨_, _, _, _, _, _, rfl⟩

theorem resName_ne (m k c j : ℕ) :
    (∀ L H : ℕ, RecResiduals.unitName m k c j ≠ s!"rec-open/l{L}-h{H}") ∧
    (∀ L H, RecResiduals.unitName m k c j ≠ VStar.recOpenStackedName L H) ∧
    (∀ sh, RecResiduals.unitName m k c j ≠ RecSparse.unitName sh) ∧
    (∀ l N, RecResiduals.unitName m k c j ≠ VStar.capName l N) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> intros <;> intro h <;>
    have h := congrArg (fun s : String => s.toList.take 9) h <;>
    simp only [RecSparse.unitName, VStar.recOpenStackedName, RecResiduals.unitName, VStar.capName, String.toList_append,
      List.append_assoc, FlockVBridge.toString_str] at h <;>
    rw [take_nine (by decide), take_nine (by decide)] at h <;>
    exact absurd h (by decide)

theorem resName_ne_link (m k c j UB PB : ℕ) : RecResiduals.unitName m k c j ≠ VStar.linkName UB PB := by
  intro h
  have h := congrArg (fun s : String => s.toList.take 9) h
  simp only [RecResiduals.unitName, VStar.linkName, String.toList_append, List.append_assoc,
    FlockVBridge.toString_str] at h
  rw [take_nine (by decide), take_nine (by decide)] at h
  exact absurd h (by decide)

theorem inner_of_resName {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    {s : VStar.Stmt} (hs : s ∈ ss) {m k c j : ℕ} (hn : RecResiduals.unitName m k c j = s.name) :
    ∃ j' nm p, s = .inner j' nm p := by
  obtain ⟨n1, n2, n3, n4⟩ := resName_ne m k c j
  obtain ⟨vsh, A, C, K, L, N, hA, hC, hK, -, hL, -, rfl, -⟩ := FlockC3.VStarList.statements_ok h
  rcases Array.mem_append.1 hs with hs | hs
  rotate_left
  · rcases FlockC3.VStarList.tail_cases hK hL N _ hs with ⟨_, _, rfl⟩ | ⟨_, _, _, rfl⟩
    · exact absurd hn (n4 _ _)
    · exact absurd hn (resName_ne_link _ _ _ _ _ _)
  rcases Array.mem_append.1 hs with hs | hs
  · obtain ⟨_, _, _, -, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
    simp only [Array.mem_append, List.mem_toArray, List.mem_map, List.mem_range] at hs
    rcases hs with ⟨l, -, rfl⟩ | ⟨j', -, rfl⟩
    · exact absurd hn (n1 _ _)
    · exact ⟨_, _, _, rfl⟩
  · cases hsp : i.sparse with
    | none =>
      rw [(FlockC3.VStarList.candidate3_ok hC).1 hsp] at hs
      simp at hs
    | some sp =>
      obtain ⟨rest, rfl, hrest⟩ := FlockC3.PubRows.candidate3_rest hC hsp
      rcases Array.mem_append.1 hs with hs | hs
      · simp only [List.mem_toArray, List.mem_singleton] at hs
        subst hs
        exact absurd hn (n3 _)
      · rcases hrest s hs with ⟨l, L, H, n', rfl⟩ | ⟨m', c', k', S', ports, nv, rfl⟩
        · exact absurd hn (n2 _ _)
        · exact absurd hn.symm ((openingName_ne m' c' k').2.1 m k c j)

theorem valueRound_rep {rep : Array SchedRound} {sp : Option RecSparse.Shape} {stk : Option (Array SchedRound)}
    (ρ : Fin 2) {m : ℕ} (hm : m < (msgRounds rep).length) :
    valueRound rep sp stk s!"rec-c{m}" ρ.val = some (.rep ρ, (msgRounds rep)[m]) := by
  unfold valueRound
  rw [cIndex_c]
  simp [hm, Nat.le_of_lt_succ ρ.isLt]

theorem valueRound_opening {rep rsR : Array SchedRound} {sp : Option RecSparse.Shape}
    {stk : Option (Array SchedRound)} (hR : stk = some rsR) {i : ℕ} (hi : i < (msgRounds rsR).length) :
    valueRound rep sp stk s!"rec-opening-c{i}" 0 = some (.stacked, (msgRounds rsR)[i]) := by
  unfold valueRound
  rw [cIndex_opening, openingIndex_c]
  simp [hR, hi]

variable {st : Flock.Stmt} {hp : HoloParams} {Cm : Type} {ish : RecVStar.Inner} {tbl : String}
  {bc : Cm → String → ℕ → List UInt8}

theorem holoSel_rep {rs : Array SchedRound} {L : List (HoloStream × Array SchedRound)}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some ((.rep 0, rs) :: (.rep 1, rs) :: L))
    (hr : holoRounds ish.m ish.kLog ish.claims = some rs) (ρ : Fin 2) {m : ℕ} (hm : m < (msgRounds rs).length) :
    (holoMap st hp Cm ish tbl bc).sel (repStart (heldBefore (.rep 0) rs rs.size) ρ +
      heldBefore (.rep ρ) rs (msgRounds rs)[m]) s!"rec-c{m}" ρ.val :=
  ⟨_, rs, .rep ρ, _, hS, hr, valueRound_rep ρ hm, roundAt_reps rs L ρ _⟩

theorem holoSel_stageOf {rs : Array SchedRound} {L : List (HoloStream × Array SchedRound)}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some ((.rep 0, rs) :: (.rep 1, rs) :: L))
    (hr : holoRounds ish.m ish.kLog ish.claims = some rs) (ρ : Fin 2) {m : ℕ} (hm : m < (msgRounds rs).length)
    {σ : List FlockSoundness.Discharge.Exec.Kind} {c₀ : ℕ} (hc₀ : c₀ = heldBefore (.rep 0) rs rs.size)
    (hσ : FlockC3.Sparse.roundOf σ m = heldBefore (.rep ρ) rs (msgRounds rs)[m]) :
    (holoMap st hp Cm ish tbl bc).sel (FlockC3.Reps.stageOf σ c₀ ρ m) s!"rec-c{m}" ρ.val := by
  have he : FlockC3.Reps.stageOf σ c₀ ρ m = repStart c₀ ρ + FlockC3.Sparse.roundOf σ m := by
    unfold FlockC3.Reps.stageOf repStart
    have := ρ.isLt
    split
    · rename_i h0
      rw [h0]
      omega
    · rename_i h0
      have h1 : ρ.val = 1 := by omega
      rw [h1]
      omega
  rw [he, hσ, hc₀]
  exact holoSel_rep hS hr ρ hm

theorem holoSel_opening {L : List (HoloStream × Array SchedRound)} {rep rsR : Array SchedRound}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some L)
    (hr : holoRounds ish.m ish.kLog ish.claims = some rep) (hR : ish.stacked.bind stackedRounds = some rsR) {i : ℕ}
    (hi : i < (msgRounds rsR).length) :
    (holoMap st hp Cm ish tbl bc).sel (roundAt L .stacked (msgRounds rsR)[i]) s!"rec-opening-c{i}" 0 :=
  ⟨L, rep, .stacked, _, hS, hr, valueRound_opening hR hi, rfl⟩

def OwnHas (own : Array Flock.Registered.Port) (sv : VStar.Stmt) : Prop :=
  ∀ e ∈ ownPorts sv, ∃ pt ∈ own, (pt.name, pt.value, pt.reads) = e

theorem read_of_port {own : Array Flock.Registered.Port} {sv : VStar.Stmt} (ho : OwnHas own sv)
    {e : String × String × Array (Option ℕ)} (he : e ∈ ownPorts sv) (u x : ℕ) (hx : e.2.2[u]? = some (some x))
    (P : String → ℕ → Prop) (hP : P e.2.1 x) :
    ∃ pt ∈ own, pt.name = e.1 ∧ pt.value = e.2.1 ∧ ∃ x, pt.reads[u]? = some (some x) ∧ P pt.value x := by
  obtain ⟨pt, hpt, rfl⟩ := ho e he
  exact ⟨pt, hpt, rfl, rfl, x, hx, hP⟩

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {n : ℕ} {L : SamplingDesign n} {Reg Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

def MsgNamesAt (ish : RecVStar.Inner) (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ s : Fin S,
    ∃ (ss : Array VStar.Stmt) (sv : VStar.Stmt), RecVStar.statements ish = .ok ss ∧ ss[(s : ℕ)]? = some sv ∧
      RecVStar.checkMsgNames sv (Vs s).c ish.holo = .ok ()

def AcceptedAt (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∃ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o

theorem msgNamesAt_of_checks {bind : Fin S → VStar.Bindings}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {rows : Option (Array Flock.Registered.Port)}
    (hck : ∀ R ω t R₂ (s : Fin S), (∃ tr, AcceptsZKR (zs R ω t R₂ s).I (zs R ω t R₂ s).own tr) →
      ∃ tr, FlockC3.PubRows.StmtsChecked ish s (bind s) rows (zs R ω t R₂ s).I (zs R ω t R₂ s).own tr) :
    MsgNamesAt ish zs dg N k Rw := by
  intro R ω t R₂ _ _ hA s
  obtain ⟨hacc, -⟩ := hA
  obtain ⟨_, _, sts, hs, hfm⟩ := hck R ω t R₂ s ⟨_, (hacc s).1⟩
  obtain ⟨h0, hc0⟩ := setup_zero_c (zs R ω t R₂ s).ht (zs R ω t R₂ s).hTags.1 (zs R ω t R₂ s).hcH hs
  obtain ⟨ss, sv, hss, hj, hm⟩ := FlockC3.PubRows.msgNames_of_checkStmt (array_forM_ok hfm 0 h0)
  rw [hc0] at hm
  exact ⟨ss, sv, hss, hj, hm⟩

theorem msgNamesAt_of_accepted {xR : HoloReg GF128 Dig} {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hca : FlockC3.PubRows.ChecksAccepted F rows wr reg) {dg : ByteArray → ByteArray} {N k Rw : ℕ} :
    MsgNamesAt F.ish F.zs dg N k Rw :=
  msgNamesAt_of_checks (rows := rows) (bind := F.bind) (fun R ω t R₂ s h => by
    obtain ⟨tr, -, -, hs⟩ := hca R ω t R₂ s h
    exact ⟨tr, hs⟩)

theorem roundPort_inner {bind : Fin S → VStar.Bindings}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs) (hst : SessStmt ish bind zs dg N k Rw)
    (hmn : MsgNamesAt ish zs dg N k Rw) (hA : AcceptedAt zs dg N k Rw) (s : Fin S) {m kk cl j : ℕ}
    (hpar : RecResiduals.params (Vs s).c.unit.name = some (m, kk, cl, j)) (q : Fin (Vs s).c.ports.size) {i : ℕ}
    (hq : (Vs s).c.ports[q].name = s!"m{i}") :
    ∃ ss sv, RecVStar.statements ish = .ok ss ∧ ss[(s : ℕ)]? = some sv ∧
      (s!"m{i}", s!"rec-c{i}", #[some 0, some 1]) ∈ ownPorts sv ∧ (Vs s).nv = 2 := by
  obtain ⟨R, ω, t, R₂, τ, o, hA⟩ := hA
  obtain ⟨ss, sv, cf, hss, hj, hf, hcf, hc⟩ := hst R ω t R₂ τ o hA s
  obtain ⟨ss', sv', hss', hj', hm⟩ := hmn R ω t R₂ τ o hA s
  obtain rfl : ss' = ss := Except.ok.inj (hss'.symm.trans hss)
  obtain rfl : sv' = sv := Option.some.inj (hj'.symm.trans hj)
  obtain ⟨j', nm, pt, rfl⟩ := inner_of_resName hss (Array.mem_of_getElem? hj)
    ((FlockVBridge.Family.resUnitName_params hpar).trans (name_of_filter hf).2.symm)
  exact ⟨_, _, hss, hj, inner_port_of_msgNames hm hq, (hob s _ _ cf _ hss hj hcf hc (Vs s).hscope).2⟩

theorem wRoundRead_inner {bind : Fin S → VStar.Bindings}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    (hst : SessStmt ish bind zs dg N k Rw) (hmn : MsgNamesAt ish zs dg N k Rw) (hA : AcceptedAt zs dg N k Rw)
    {own : Fin S → Array Flock.Registered.Port}
    (hown : ∀ (s : Fin S) ss sv, RecVStar.statements ish = .ok ss → ss[(s : ℕ)]? = some sv → OwnHas (own s) sv)
    {sel : ℕ → String → ℕ → Prop} {stage : Fin 2 → ℕ → ℕ} {sess : ℕ → Fin S}
    {un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv} (hun : ∀ p (r : Fin 2), r.val < (Vs (sess p)).nv → (un p r).val = r.val)
    {p m mm kk cl j : ℕ} (hpar : RecResiduals.params (Vs (sess p)).c.unit.name = some (mm, kk, cl, j)) (ρ : Fin 2)
    (hsel : sel (stage ρ m) s!"rec-c{m}" ρ.val) (q : Fin (Vs (sess p)).c.ports.size)
    (hq : (Vs (sess p)).c.ports[q].name = s!"m{m}") :
    ∃ pt ∈ own (sess p), pt.name = s!"m{m}" ∧ pt.value = s!"rec-c{m}" ∧
      ∃ x, pt.reads[(un p ρ).val]? = some (some x) ∧ sel (stage ρ m) pt.value x := by
  obtain ⟨ss, sv, hss, hj, hmem, hnv⟩ := roundPort_inner hob hst hmn hA (sess p) hpar q hq
  rw [hun p ρ (by rw [hnv]; exact ρ.isLt)]
  refine read_of_port (hown _ _ _ hss hj) hmem ρ.val ρ.val ?_ (sel (stage ρ m)) hsel
  rcases ρ with ⟨_ | _ | n, hρ⟩
  · rfl
  · rfl
  · omega

theorem roundRead_opening {bind : Fin S → VStar.Bindings}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    (hst : SessStmt ish bind zs dg N k Rw) (hmn : MsgNamesAt ish zs dg N k Rw) {own : Fin S → Array Flock.Registered.Port}
    (hown : ∀ (s : Fin S) ss sv, RecVStar.statements ish = .ok ss → ss[(s : ℕ)]? = some sv → OwnHas (own s) sv)
    {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128} {i : ℕ}
    (hsel : sel (c₀ + i) s!"rec-opening-c{i}" 0) {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {s : Fin S}
    {un : Fin (Vs s).nv} {j p : ℕ} (hO : OpenCopy zs dg N k Rw sh w gs s un j p) (q : Fin (Vs s).c.ports.size)
    (hq : (Vs s).c.ports[q].name = s!"m{i}") :
    ∃ pt ∈ own s, pt.name = s!"m{i}" ∧ pt.value = s!"rec-opening-c{i}" ∧
      ∃ x, pt.reads[un.val]? = some (some x) ∧ sel (c₀ + i) pt.value x := by
  have hu : un.val = 0 := Nat.lt_one_iff.mp (lt_of_lt_of_eq un.isLt (nv_of_copy hob hst hO))
  obtain ⟨R, ω, t, R₂, τ, o, _, hA, _, _, -, -, -, m, cl, hpm, -⟩ := hO
  obtain ⟨ss, sv, cf, hss, hj, hf, -, -⟩ := hst R ω t R₂ τ o hA s
  obtain ⟨ss', sv', hss', hj', hm⟩ := hmn R ω t R₂ τ o hA s
  obtain rfl : ss' = ss := Except.ok.inj (hss'.symm.trans hss)
  obtain rfl : sv' = sv := Option.some.inj (hj'.symm.trans hj)
  obtain ⟨m', c', j', S', ps, nv, rfl⟩ := opening_of_name hss (Array.mem_of_getElem? hj)
    ((openingName_params hpm).trans (name_of_filter hf).2.symm)
  rw [hu]
  exact read_of_port (hown _ _ _ hss hj) (opening_port_of_msgNames hm hq) 0 0 rfl (sel (c₀ + i)) hsel

end FlockC3.Reps
