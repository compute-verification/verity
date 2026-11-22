module

public import Proofs.Flock.Recursive.HPortTie
public import Proofs.Flock.VBridge.Family.PartKeys
public import Proofs.Flock.C3.Reps.WithReads
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
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.C3.Reps.AtList
import Proofs.Flock.C3.Reps.RoundLinks
import Proofs.Flock.C3.VStarList.Shape
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.VBridge.Sparse.Unit

/-!
# The algebra unit's `h` round `m<hr>`: the list, the layout and the part

What the verifier's checks of V*'s holo list and each algebra part give about rep `i`'s `h`:

* The list (`hRound_held`, `sparse_stmt_of_name`): at a holo shape `RecVStar.statements` refuses a list none of whose
  groups holds `h`'s round `hRound m kLog` as a row port, and its RecSparse statement is `.sparse _ (hRound m kLog)`.
* The layout (`rounds_holo`, `wi_h`, `portSizes_h`, `starts_h`, `portNames_h`): at a holo shape `h`'s round is the
  64-element round `m<hRound>` after the zerocheck's and lincheck's (`zlRounds`), `h l` its `l`-th element of `w`.
* The part (`readBits_of_hUnread`, `rowBit_at`): `RecResiduals.check` refuses a holo part that reads an element of `h`
  in fewer than 128 bits (`hUnread`), so each bit of the part's `m<hRound>` row is a computed row's input, which
  `ResZeroAtLH`'s `hw` registers at the bit of `w` the part's `wIdx` puts there, `h l`'s.
-/

@[expose] public section

namespace FlockC3.Reps

namespace HAlg

open Flock FlockSoundness Spot.Sampling FlockSoundness.Discharge FlockSoundness.Discharge.Recursion

/-! ## The list -/

/-- **A holo list holds `h`'s round in a group**: `RecVStar.statements` refuses a holo shape none of whose groups holds
`hRound m kLog` as a row port. -/
theorem hRound_held {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    (hh : i.holo = true) :
    ∃ sh0, RecResiduals.shape? i.m i.kLog i.claims = some sh0 ∧
      ∃ GS gs, RecAlgebra.structureOf { sh0 with holo := i.holo } = .ok GS ∧
        RecResiduals.groups GS (RecResiduals.portSizes { sh0 with holo := i.holo }) = .ok gs ∧
        ∃ p, ∃ _ : p < gs.size, RecVStar.hRound i.m i.kLog ∈ gs[p].2.1 := by
  unfold RecVStar.statements at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i sh0 hsh
  split at h
  · cases h
  obtain ⟨ps, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨gs, hgs, h⟩ := FlockC3.VStarList.eok_bind h
  split at h
  · cases h
  split at h
  · cases h
  rename_i hany
  obtain ⟨GS, hGS, hgs⟩ := FlockC3.VStarList.eok_bind (ok_of_mapError hgs)
  rw [hh] at hany
  simp only [Bool.true_and, Bool.not_eq_true', Bool.not_eq_false] at hany
  obtain ⟨p, hp, hc⟩ : ∃ p, ∃ _ : p < gs.size, RecVStar.hRound i.m i.kLog ∈ gs[p].2.1 := by simpa using hany
  exact ⟨sh0, hsh, GS, gs, hGS, hgs, p, hp, hc⟩

/-- **`checkStmt`'s statement and its parse and part checks**: the session's unit is the list's statement `j`, by name,
and `checkParsed` and `checkPart` pass on it. -/
theorem parsed_of_checkStmt {i : RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f)
      rows = .ok ()) :
    ∃ ss s, RecVStar.statements i = .ok ss ∧ ss[j]? = some s ∧ s.name = st.c.unit.name ∧
      RecVStar.checkParsed s st.c = .ok () ∧ RecVStar.checkPart s st.c = .ok () := by
  unfold RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := Refine.ebind_ok h
    cases h1
  · split at h
    · obtain ⟨_, h1, -⟩ := Refine.ebind_ok h
      cases h1
    · split at h
      · split at h
        · cases h
        obtain ⟨_, -, h⟩ := Refine.ebind_ok h
        obtain ⟨s, hs, -⟩ := Refine.ebind_ok h
        obtain ⟨hck, -, hpa, hpt⟩ := checkCircuit_ok hs
        obtain ⟨ss, hss, hj, hname, -⟩ := check_ok hck
        exact ⟨ss, s, hss, hj, hname, hpa, hpt⟩
      · cases h

/-- **The list's statement named as a RecSparse unit is its RecSparse statement**, at `h`'s round (`sparse_of_name`). -/
theorem sparse_stmt_of_name {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    {s : VStar.Stmt} (hs : s ∈ ss) {sh : RecSparse.Shape} (hn : RecSparse.unitName sh = s.name) :
    ∃ sp, s = .sparse sp (RecVStar.hRound i.m i.kLog) := by
  obtain ⟨n1, n2, n3, n4, n5⟩ := FlockC3.PubRows.sparse_ne_names sh
  obtain ⟨vsh, A, C, K, L, N, hA, hC, hK, -, hL, -, rfl, -⟩ := FlockC3.VStarList.statements_ok h
  rcases Array.mem_append.1 hs with hs | hs
  rotate_left
  · rcases FlockC3.VStarList.tail_cases hK hL N _ hs with ⟨_, _, rfl⟩ | ⟨_, _, _, rfl⟩
    · exact absurd hn (n5 _ _)
    · exact absurd hn (FlockC3.PubRows.sparse_ne_linkName _ _ _)
  rcases Array.mem_append.1 hs with hs | hs
  · obtain ⟨_, _, _, -, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
    simp only [Array.mem_append, List.mem_toArray, List.mem_map, List.mem_range] at hs
    rcases hs with ⟨l, -, rfl⟩ | ⟨j, -, rfl⟩
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
        exact ⟨sp, hs⟩
      · rcases hrest s hs with ⟨l, L, H, n, rfl⟩ | ⟨m, c, k, S, ports, nv, rfl⟩
        · exact absurd hn (n3 _ _)
        · exact absurd hn (n4 _ _ _)

/-! ## The layout of `h`'s round -/

/-- The rounds before `h`'s at a holo shape: `RecAlgebra.preRounds` without its `h` round. -/
def zlRounds (sh : Flock.Shape) : Array (Array RecAlgebra.Key) :=
  #[(Array.range 64).map .ab ++ (Array.range 64).map .c] ++
    (Array.range (sh.m - 6)).map (fun i => #[.g1 i, .ginf i]) ++ #[#[.fa, .fb]] ++
    (Array.range (sh.kLog - 6)).map (fun j => #[.q1 j, .qinf j]) ++ #[(Array.range 64).map .zp]

theorem rounds_holo {sh : Flock.Shape} (hh : sh.holo = true) :
    RecAlgebra.rounds sh = zlRounds sh ++ #[(Array.range 64).map .h] ++
      ((Array.range sh.claims).map (fun k => (Array.range 128).map (.rs k)) ++ RecAlgebra.ligRounds sh.sched) := by
  rw [RecAlgebra.rounds_split]
  simp only [RecAlgebra.preRounds, RecAlgebra.hRound, hh, ite_true, zlRounds, Array.append_assoc]

/-- `h`'s round is round `hRound m kLog`. -/
theorem zlRounds_size (sh : Flock.Shape) (hm : 6 ≤ sh.m) (hk : 6 ≤ sh.kLog) :
    (zlRounds sh).size = RecVStar.hRound sh.m sh.kLog := by
  simp only [zlRounds, RecVStar.hRound, Array.size_append, Array.size_map, Array.size_range]
  simp
  omega

theorem h_not_zl (sh : Flock.Shape) (l : ℕ) : RecAlgebra.Key.h l ∉ (zlRounds sh).flatten := by
  simp [zlRounds, Array.mem_flatten]

theorem idxOf?_mid {xs hs ys : Array RecAlgebra.Key} {a : RecAlgebra.Key} {i : ℕ} (hx : a ∉ xs)
    (hh : hs[i]? = some a) (hlt : ∀ j < i, hs[j]? ≠ some a) : (xs ++ hs ++ ys).idxOf? a = some (xs.size + i) := by
  have hi := (Array.getElem?_eq_some_iff.1 hh).1
  apply RecAlgebra.idxOf?_of
  · rw [Array.getElem?_append_left (by simp only [Array.size_append]; omega),
      Array.getElem?_append_right (by omega), Nat.add_sub_cancel_left]
    exact hh
  · intro j hj he
    rw [Array.getElem?_append_left (by simp only [Array.size_append]; omega)] at he
    by_cases hjx : j < xs.size
    · rw [Array.getElem?_append_left hjx] at he
      exact hx (Array.mem_of_getElem? he)
    · rw [Array.getElem?_append_right (by omega)] at he
      exact hlt (j - xs.size) (by omega) he

/-- **`h l` is `w`'s element `l` of `h`'s round**. -/
theorem wi_h {sh : Flock.Shape} (hh : sh.holo = true) {l : ℕ} (hl : l < 64) :
    RecAlgebra.wi sh (.h l) = some ((zlRounds sh).flatten.size + l) := by
  unfold RecAlgebra.wi RecAlgebra.wNames
  rw [rounds_holo hh]
  simp only [Array.flatten_append, Array.append_assoc]
  rw [← Array.append_assoc]
  refine idxOf?_mid (h_not_zl sh l) ?_ fun j hj he => ?_
  · simp [hl]
  · simp only [Array.flatten_singleton, Array.getElem?_map, Array.getElem?_range] at he
    split at he <;> simp at he
    omega

theorem portSizes_h {sh : Flock.Shape} (hh : sh.holo = true) :
    (RecResiduals.portSizes sh)[(zlRounds sh).size]! = 64 := by
  unfold RecResiduals.portSizes
  rw [rounds_holo hh]
  simp [getElem!_pos, Array.getElem_push]

theorem starts_h {sh : Flock.Shape} (hh : sh.holo = true) :
    (RecResiduals.starts (RecResiduals.portSizes sh))[(zlRounds sh).size]! = (zlRounds sh).flatten.size := by
  rw [FlockVBridge.Family.starts_get _ (by unfold RecResiduals.portSizes; rw [rounds_holo hh]; simp; omega)]
  unfold RecResiduals.portSizes
  rw [rounds_holo hh, Array.size_flatten]
  simp [← Array.sum_toList]

theorem portNames_h {sh : Flock.Shape} (hh : sh.holo = true) :
    (RecVStar.portNames sh)[(zlRounds sh).size]! = s!"m{(zlRounds sh).size}" := by
  unfold RecVStar.portNames
  rw [rounds_holo hh, getElem!_pos _ _ (by simp; omega), Array.getElem_append_left (by simp; omega)]
  simp

/-! ## The part's row of `h` -/

theorem readBits_of_hUnread {wn : Array RecAlgebra.Key} {W : Array ℕ} {rd : Array Bool} {ns : List ℕ}
    (h : RecResiduals.hUnread wn W rd ns = none) {q : ℕ} (hq : q < W.size) {l : ℕ}
    (hk : wn[W[q]!]? = some (.h l)) : RecResiduals.readBits rd ns q = 128 := by
  unfold RecResiduals.hUnread at h
  have := List.filterMap_eq_nil_iff.1 (List.head?_eq_none_iff.1 h) q (List.mem_range.2 hq)
  simp only [hk] at this
  by_cases h128 : RecResiduals.readBits rd ns q = 128
  · exact h128
  · simp only [bne_iff_ne, ne_eq, h128, not_false_eq_true, ↓reduceIte, reduceCtorEq] at this

/-- **A part's rows are its statement's row ports, in order** (`srcKey_laid`), from its parse and `checkParsed`. -/
theorem repKeys_of {tags : Flock.Tags} {c : Flock.Circuit} (hU : Built c.unit) (P : ParseFacts tags c)
    {p : ℕ} {nm : String} {pt : VStar.Part} (hpa : RecVStar.checkParsed (.inner p nm pt) c = .ok ())
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {name : String}
    (hn : Flock.HmNets.netOf name (RecResiduals.unitLaid S ns nv) = .ok c.unit) :
    FlockVBridge.OfVStar.RepKeys c pt.S := by
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨u, hu, hupv, -, hlv, hports⟩ := checkParsed_ok hpa
  obtain ⟨-, -, hin, -⟩ := FlockVBridge.OfVStar.innerUnit_ok (show VStar.innerUnit pt.S nm = .ok u from hu)
  unfold RecVStar.vstarRows at hports
  rw [VStar.UnitC.inWords, hin] at hlv
  rw [hin] at hports
  intro q hq
  exact FlockVBridge.OfVStar.srcKey_laid hU hn (check_facts P.typed hls).leaves hupv hlv hports
    (by
      show (pt.S.ports.map (·.1) ++ ["v"]).length = (FlockVBridge.OfVStar.repPorts pt.S).length
      rw [FlockVBridge.OfVStar.repPorts_length, List.length_append, List.length_map, List.length_singleton]) hq

/-- **The row key of bit `e` of element `l` of row port `ih`** is the circuit's port `ih`, named as the port, at bit
`128 l + e` (`srcKey_row`, `take_sum_inj`). -/
theorem rowBit_at {c : Flock.Circuit} {S : VStar.RepS} (h : FlockVBridge.OfVStar.RepKeys c S) (n : ℕ)
    {sizes P : Array ℕ} {names : Array String} (hns : S.ports.map (·.2) = P.toList.map (sizes[·]!))
    (hnm : S.ports.map (·.1) = P.toList.map (names[·]!)) {ih l e : ℕ} (hih : ih < P.size)
    (hl : l < sizes[P[ih]]!) (he : e < 128) {q : Fin c.ports.size} {b : ℕ}
    (hk : Aliased.srcKey c (0 * n + FlockVBridge.OfVStar.rowCol S.ports 0
      (128 * (((P.toList.map (sizes[·]!)).take ih).sum + l) + e)) = .msg q.val (b / 8) (b % 8)) :
    q.val = ih ∧ b = 128 * l + e ∧ c.ports[q].name = names[P[ih]]! := by
  have hih' : ih < (P.toList.map (sizes[·]!)).length := by simpa using hih
  have hnsi : (P.toList.map (sizes[·]!))[ih] = sizes[P[ih]]! := by simp
  have hq : 128 * (((P.toList.map (sizes[·]!)).take ih).sum + l) + e < 128 * (S.ports.map (·.2)).sum := by
    rw [hns]
    have := FlockVBridge.OfVStar.take_sum_add_le (P.toList.map (sizes[·]!)) hih'
    omega
  obtain ⟨i, hi, e0, he0, hqe, hname, -, hk0⟩ := FlockVBridge.OfVStar.srcKey_row h n hq
  rw [hk0] at hk
  simp only [Aliased.Src.msg.injEq] at hk
  obtain ⟨hiq, h1, h2⟩ := hk
  have hb : b = e0 := by omega
  have hlen : S.ports.length = P.size := by simpa using congrArg List.length hns
  have hi' : i.val < (P.toList.map (sizes[·]!)).length := by simp; omega
  have hpi : S.ports[i.val].2 = (P.toList.map (sizes[·]!))[i.val] := by
    have h' := List.getElem_of_eq hns (by rw [List.length_map]; exact hi)
    rw [List.getElem_map] at h'
    exact h'
  rw [hpi] at he0
  rw [hns] at hqe
  obtain ⟨hii, hle⟩ := FlockVBridge.OfVStar.take_sum_inj (P.toList.map (sizes[·]!)) hih' hi'
    (by rw [hnsi]; exact hl) (show e0 / 128 < _ by omega) (show _ = _ by omega)
  have hqi : q = i := Fin.ext hiq.symm
  subst hqi
  refine ⟨hii.symm, by omega, ?_⟩
  have hni : S.ports[q.val].1 = names[P[ih]]! := by
    have h' := List.getElem_of_eq hnm (by rw [List.length_map]; exact hi)
    rw [List.getElem_map, List.getElem_map, Array.getElem_toList] at h'
    rw [h']
    congr 2
    exact hii.symm
  exact hname.trans hni

end HAlg

section family

open Flock FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Spot.Sampling Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg FlockLevel3 Accounting ZkBind Integrate Placed
open StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.PubRows (ChecksAccepted)
open FlockC3.Compose
open HAlg

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Each accepted session's statement passes the parse and part checks**: `ChecksAccepted`'s statement check at the
session's circuit (`setup_zero_c`, `parsed_of_checkStmt`). -/
theorem stmtAt_checked {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} (hchk : ChecksAccepted F rows wr reg)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (s : Fin (vstarCount F.ish)) :
    ∃ sv, F.stmts[(s : ℕ)]? = some sv ∧ sv.name = (F.Vs s).c.unit.name ∧
      RecVStar.checkParsed sv (F.Vs s).c = .ok () ∧ RecVStar.checkPart sv (F.Vs s).c = .ok () := by
  obtain ⟨_, -, -, _, _, hset, hfm⟩ := hchk R ω t R₂ s ⟨_, (h.1 s).1⟩
  obtain ⟨h0, hc0⟩ := setup_zero_c (F.zs R ω t R₂ s).ht (F.zs R ω t R₂ s).hTags.1 (F.zs R ω t R₂ s).hcH hset
  obtain ⟨ss, sv, hss, hj, hname, hpa, hpt⟩ := parsed_of_checkStmt (array_forM_ok hfm 0 h0)
  rw [hc0] at hname hpa hpt
  obtain rfl : ss = F.stmts := Except.ok.inj (hss.symm.trans F.hstmts)
  exact ⟨sv, hj, hname, hpa, hpt⟩

end family

end FlockC3.Reps
