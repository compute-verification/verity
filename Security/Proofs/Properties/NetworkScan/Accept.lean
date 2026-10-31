import Properties.NetworkScan
import Proofs.Properties.Nci.BodiesOk

/-!
# What `NetworkCheck.Period.scanOf`'s acceptance gives of the scan's Program

`scanOf_accepted` reads `scanOf`'s steps one by one (each check a `need`, so each step a bind) and keeps what
`NetTiming.Layout.Fits`'s first two parts rest on (`Accepted`): the description's layout and walk checks, the egress
template's node (a batch of `T`, returning a u16 and the row, every leaf its own, all `T` instances' leaves returned by
the root in one run), each step's ingress bucket (`Ingress`: a batch member, its first parameter the row, its first
argument one input node's leaves), and the Program topological (`Prog.orderViolation`), which part (3) reads.
-/

namespace Properties.NetworkScan.Fits

open Flock.Program (Prog Node Part Ty Defn)
open NciCheck.Properties.SecurityProofs
open NetworkCheck.Period

/-- Root node `k`. -/
abbrev node (P : Prog) (k : ℕ) : Node := (P.defs[P.root]!).body.nodes[k]!

/-- The Definition root node `k` calls. -/
abbrev callee (P : Prog) (k : ℕ) : Defn := P.defs[(node P k).fn]!

/-- The egress template's root node: the last template. -/
abbrev egressNode (P : Prog) : ℕ := (templateNodes P).getLast?.getD 0

/-- Step `t`'s ingress bucket, as `scanLayout` reads it: the root node and member whose `row` is step `t`'s ingress. -/
abbrev bucket (P : Prog) (t : ℕ) : ℕ × ℕ :=
  if t = 0 then ((templateNodes P).headD 0, 0) else ((templateNodes P).getD 1 0, t - 1)

/-- The schedule's row: `S · B` bytes of 8 bits. -/
abbrev rowTy (s : Schedule) : Ty := .array (s.allocation * s.cellBytes) (.array 8 (.value 1))

/-- What `scanOf` checks of an ingress bucket `km`: member `km.2` of a template's batch, the template's first parameter
the schedule's row, the node's first argument one input node's leaves. -/
def Ingress (P : Prog) (s : Schedule) (km : ℕ × ℕ) : Prop :=
  km.1 ∈ templateNodes P ∧ (node P km.1).form = .batch ∧ km.2 < (node P km.1).n ∧
    0 < (callee P km.1).params.size ∧ ((callee P km.1).params[0]!).2 = rowTy s ∧
    ∃ kr, ((node P km.1).args[0]?).bind (inputArg P) = some kr

/-- **What an accepted scan's Program is**, as `NetTiming.Layout.Fits`'s first two parts read it. -/
structure Accepted (P : Prog) (s : Schedule) : Prop where
  laid : NciCheck.Desc.rootLaidOut P = true ∧ NciCheck.Desc.inputsAtRoot P = true ∧ NciCheck.Desc.walkable P = true
  egress_mem : egressNode P ∈ templateNodes P
  egress_batch : (node P (egressNode P)).form = .batch ∧ (node P (egressNode P)).n = s.window
  egress_ret : (callee P (egressNode P)).ret = .tuple #[.array outBits (.value 1), rowTy s]
  egress_own : (ownReturns P)[(node P (egressNode P)).fn]! = true
  returned : ∃ q ∈ (P.defs[P.root]!).body.ret.parts, ∃ c, q = .affine false (egressNode P) 0 c ∧
    s.window * (callee P (egressNode P)).outLeaves ≤ c
  ingress : ∀ t < s.window, Ingress P s (bucket P t)
  topo : P.orderViolation = none

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {y : β} (h : (x >>= f) = .ok y) :
    ∃ a, x = .ok a ∧ f a = .ok y := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem need_ok {b : Bool} {why : String} {u : Unit} (h : need b why = .ok u) : b = true := by
  cases b
  · cases h
  · rfl

theorem needSome_ok {α : Type} {o : Option α} {why : String} {a : α} (h : needSome o why = .ok a) : o = some a := by
  cases o with
  | none => cases h
  | some x => cases h; rfl

theorem ingressTemplates_ok {l : List (Node × Defn)} {x : (Node × Defn) × Option (Node × Defn)}
    (h : ingressTemplates l = .ok x) : (∃ f, l = [f] ∧ x = (f, none)) ∨ ∃ f b, l = [f, b] ∧ x = (f, some b) := by
  match l, h with
  | [f], h => exact Or.inl ⟨f, rfl, (Except.ok.inj h).symm⟩
  | [f, b], h => exact Or.inr ⟨f, b, rfl, (Except.ok.inj h).symm⟩
  | [], h => cases h
  | _ :: _ :: _ :: _, h => cases h

theorem form_batch {f : Flock.Program.Form} (h : (f == .batch) = true) : f = .batch := by
  cases f <;> first | rfl | exact absurd h (by decide)

theorem split_two {β : Type} {g : ℕ → β} {l : List ℕ} {f : β} (h : (l.map g).dropLast = [f]) :
    ∃ a e, l = [a, e] ∧ g a = f := by
  match l, h with
  | [], h => simp at h
  | [_], h => simp at h
  | [a, e], h => exact ⟨a, e, rfl, by simpa using h⟩
  | _ :: _ :: _ :: _, h => simp at h

theorem split_three {β : Type} {g : ℕ → β} {l : List ℕ} {f b : β} (h : (l.map g).dropLast = [f, b]) :
    ∃ a a' e, l = [a, a', e] ∧ g a = f ∧ g a' = b := by
  match l, h with
  | [], h => simp at h
  | [_], h => simp at h
  | [_, _], h => simp at h
  | [a, a', e], h =>
    have h' : g a = f ∧ g a' = b := by simpa using h
    exact ⟨a, a', e, rfl, h'.1, h'.2⟩
  | _ :: _ :: _ :: _ :: _, h => simp at h

/-- **What `scanOf`'s acceptance gives**: an accepted scan's canonical text decodes to a Program of the generator
form, as `Accepted` states it. -/
theorem scanOf_accepted {j : Lean.Json} {program : String} {s : Schedule} {sc : Scan} {c : String} {P : Prog}
    (h : scanOf j program s = .ok sc) (hc : Flock.canon j = .ok c) (hd : Flock.Program.decode c.toUTF8 = .ok P) :
    Accepted P s := by
  unfold scanOf at h
  obtain ⟨c', hc', h⟩ := bind_ok h
  obtain rfl : c' = c := by rw [hc] at hc'; exact (Except.ok.inj hc').symm
  obtain ⟨P', hP', h⟩ := bind_ok h
  rw [hd] at hP'
  obtain rfl : P = P' := by cases hP'; rfl
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨root, hroot, h⟩ := bind_ok h
  replace hroot := needSome_ok hroot
  obtain rfl : root = P.defs[P.root]! := by
    obtain ⟨hlt, e⟩ := Array.getElem?_eq_some_iff.1 hroot
    rw [getElem!_pos P.defs _ hlt, e]
  obtain ⟨⟨xe, de⟩, hback, h⟩ := bind_ok h
  replace hback := needSome_ok hback
  obtain ⟨_, h2, h⟩ := bind_ok h
  obtain ⟨⟨⟨xf, df⟩, later⟩, hm, h⟩ := bind_ok h
  obtain ⟨_, h3, h⟩ := bind_ok h
  obtain ⟨_, h4, h⟩ := bind_ok h
  obtain ⟨_, h5, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, h7, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, h9, h⟩ := bind_ok h
  obtain ⟨_, h10, h⟩ := bind_ok h
  obtain ⟨_, h11, h⟩ := bind_ok h
  obtain ⟨_, h12, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, h15, -⟩ := bind_ok h
  replace h15 := Option.isNone_iff_eq_none.1 (need_ok h15)
  replace h3 := need_ok h3
  replace h4 := need_ok h4
  replace h5 := need_ok h5
  replace h7 := need_ok h7
  replace h9 := need_ok h9
  replace h10 := need_ok h10
  replace h11 := need_ok h11
  replace h12 := need_ok h12
  clear h2
  have hxe : xe = node P (egressNode P) ∧ de = callee P (egressNode P) ∧ egressNode P ∈ templateNodes P := by
    rw [Array.back?_map, List.back?_toArray, Option.map_eq_some_iff] at hback
    obtain ⟨e, he, hge⟩ := hback
    have hen : egressNode P = e := by simp [egressNode, he]
    rw [hen]
    simp only [Prod.mk.injEq] at hge
    exact ⟨hge.1.symm, hge.2.symm, List.mem_of_getLast? he⟩
  obtain ⟨rfl, rfl, hemem⟩ := hxe
  have hpick : (∃ a e, templateNodes P = [a, e] ∧ (xf, df) = (node P a, callee P a) ∧ later = none) ∨
      ∃ a b e, templateNodes P = [a, b, e] ∧ (xf, df) = (node P a, callee P a) ∧
        later = some (node P b, callee P b) := by
    rcases ingressTemplates_ok hm with ⟨f, hf, hx⟩ | ⟨f, b, hf, hx⟩
    · simp only [Array.toList_pop, Array.toList_map] at hf
      obtain ⟨a, e, hts, hga⟩ := split_two hf
      simp only [Prod.mk.injEq] at hx
      refine Or.inl ⟨a, e, hts, ?_, hx.2⟩
      rw [hx.1, ← hga]
    · simp only [Array.toList_pop, Array.toList_map] at hf
      obtain ⟨a, a', e, hts, hga, hgb⟩ := split_three hf
      simp only [Prod.mk.injEq] at hx
      refine Or.inr ⟨a, a', e, hts, ?_, ?_⟩
      · rw [hx.1, ← hga]
      · rw [hx.2, ← hgb]
  have hret : ∃ q ∈ (P.defs[P.root]!).body.ret.parts, ∃ c, q = .affine false (egressNode P) 0 c ∧
      s.window * (callee P (egressNode P)).outLeaves ≤ c := by
    rw [Array.any_eq_true] at h10
    obtain ⟨i, hi, hq⟩ := h10
    refine ⟨_, Array.getElem_mem hi, ?_⟩
    generalize (P.defs[P.root]!).body.ret.parts[i] = q at hq ⊢
    split at hq
    · simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hq
      refine ⟨_, ?_, hq.2⟩
      rw [hq.1]
    · cases hq
  have hrows : ∀ x ∈ #[xf] ++ (later.map (#[·.1])).getD #[], (x.args[0]?.bind (inputArg P)).isSome = true := by
    intro x hx
    rw [Array.all_eq_true_iff_forall_mem] at h12
    exact h12 _ (Array.mem_map.2 ⟨x, hx, rfl⟩)
  simp only [Bool.and_eq_true, beq_iff_eq] at h3 h4 h5 h9
  obtain ⟨⟨hn1, hnl⟩, -⟩ := h3
  obtain ⟨⟨⟨⟨hxf, hxn⟩, hln⟩, hxeb⟩, hxen⟩ := h4
  obtain ⟨hr0, hrl⟩ := h5
  have hdf : 0 < df.params.size := by
    have := congrArg Array.size hn1
    rw [Array.size_map] at this
    rw [this]; simp [firstParams]
  refine ⟨⟨h9.1.1, h9.1.2, h9.2⟩, hemem, ⟨form_batch hxeb, hxen⟩, BodiesOk.ty_beq h7, h11, hret, ?_, h15⟩
  rcases hpick with ⟨a, e, hts, hfa, rfl⟩ | ⟨a, b, e, hts, hfa, rfl⟩
  · simp only [Prod.mk.injEq] at hfa
    obtain ⟨rfl, rfl⟩ := hfa
    try dsimp only at hln
    simp only [Option.some.injEq] at hln
    intro t ht
    have t0 : t = 0 := by omega
    subst t0
    have hb0 : bucket P 0 = (a, 0) := by simp [bucket, hts]
    obtain ⟨kr, hkr⟩ := Option.isSome_iff_exists.1 (hrows (node P a) (by simp))
    rw [hb0]; unfold Ingress; try dsimp only
    exact ⟨by simp [hts], form_batch hxf, by omega, by omega, BodiesOk.ty_beq hr0, kr, hkr⟩
  · simp only [Prod.mk.injEq] at hfa
    obtain ⟨rfl, rfl⟩ := hfa
    simp only [Option.all_some] at hnl hrl
    have hbs : 0 < (callee P b).params.size := by
      have := congrArg Array.size (beq_iff_eq.1 hnl)
      rw [Array.size_map] at this
      rw [this]; simp [ingressParams]
    try dsimp only at hln
    split at hln
    · rename_i hcond
      simp only [Option.some.injEq] at hln
      simp only [decide_eq_true_eq] at hcond
      intro t ht
      by_cases t0 : t = 0
      · subst t0
        have hb0 : bucket P 0 = (a, 0) := by simp [bucket, hts]
        obtain ⟨kr, hkr⟩ := Option.isSome_iff_exists.1 (hrows (node P a) (by simp))
        rw [hb0]; unfold Ingress; try dsimp only
        exact ⟨by simp [hts], form_batch hxf, by omega, by omega, BodiesOk.ty_beq hr0, kr, hkr⟩
      · have hbt : bucket P t = (b, t - 1) := by simp [bucket, t0, hts]
        obtain ⟨kr, hkr⟩ := Option.isSome_iff_exists.1 (hrows (node P b) (by simp))
        rw [hbt]; unfold Ingress; try dsimp only
        exact ⟨by simp [hts], form_batch hcond.1, by omega, by omega, BodiesOk.ty_beq hrl, kr, hkr⟩
    · cases hln

end Properties.NetworkScan.Fits
