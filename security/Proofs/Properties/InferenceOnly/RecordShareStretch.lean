module

public import Proofs.Properties.InferenceOnly.ResolveRows
public import Proofs.Properties.InferenceOnly.RecordOwn

/-!
# The reference a declared node's reader reads

`Record.stretchOf` reads how one argument of a root node reads an input node, member by member: `(node, b, q, os, e)`,
member `m`'s leaf `t < e` the node's leaf `b + (m / q) · os + (m % q) · e + t`. `stretch_ref` proves it of the
reference the member's parameter leaf `t` is: where the argument is the one part `stretchOf` takes, the reference is
that leaf of the node, and `e` is the parameter's leaves.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open Flock.Program
open InferenceOnlyVerifier.Record (stretchOf)
open InferenceOnlyVerifier.Properties.SecurityProofs.Resolve (argIdx argIdx_batch argIdx_call memberIdx bs_members prim_members)

/-- The leaves of a type are those of the type with an axis removed, as many times as the axis is long. -/
theorem leaves_removeAxis : ∀ (ax : Nat) (t : Ty) {n : Nat} {mt : Ty}, t.removeAxis ax = some (n, mt) →
    t.leaves = n * mt.leaves
  | 0, .array k e, n, mt, h => by
    simp only [Ty.removeAxis, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk.leaves_array k e
  | ax + 1, .array k e, n, mt, h => by
    simp only [Ty.removeAxis, Option.map_eq_some_iff] at h
    obtain ⟨⟨m, inner⟩, hm, heq⟩ := h
    simp only [Prod.mk.injEq] at heq
    obtain ⟨rfl, rfl⟩ := heq
    rw [InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk.leaves_array,
      InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk.leaves_array, leaves_removeAxis ax e hm]
    ring
  | 0, .value _, _, _, h => by simp [Ty.removeAxis] at h
  | 0, .tuple _, _, _, h => by simp [Ty.removeAxis] at h
  | 0, .record _, _, _, h => by simp [Ty.removeAxis] at h
  | _ + 1, .value _, _, _, h => by simp [Ty.removeAxis] at h
  | _ + 1, .tuple _, _, _, h => by simp [Ty.removeAxis] at h
  | _ + 1, .record _, _, _, h => by simp [Ty.removeAxis] at h

/-- How many leaves and whether mapped `stretchOf` takes of an argument: a batch's argument mapped over its members,
or a call's or a batch's argument whole. -/
def stretchE (fm : Flock.Program.Form) (dm : Option (Nat × Nat)) (cnt : Nat) : Option (Nat × Bool) :=
  match fm, dm with
  | .batch, some (d, L) => if d * L == cnt then some (L, true) else none
  | .batch, none | .call, none => some (cnt, false)
  | _, _ => none

/-- `stretchOf`, on what it reads of the node and argument. -/
def stretchS (fm : Flock.Program.Form) (dm : Option (Nat × Nat)) (cnt : Nat) (ps : List Part) :
    Option (Nat × Int × Nat × Int × Nat) :=
  match stretchE fm dm cnt, ps with
  | some (e, mapped), [.affine false i b _] => some (i, (b : Int), 1, if mapped then (e : Int) else 0, e)
  | some (e, mapped), [.strided false i b inner os is _] =>
    if e ≥ 1 && inner % e == 0 && inner ≥ e && is == 1 && (mapped || inner == e) then
      some (i, b, inner / e, if mapped then os else 0, e)
    else none
  | _, _ => none

theorem stretchOf_eq (nd : Node) (a : Arg) :
    stretchOf nd a = stretchS nd.form a.dims a.seq.count a.seq.parts.toList := rfl

theorem stretchE_spec {fm : Flock.Program.Form} {dm : Option (Nat × Nat)} {cnt e : Nat} {mp : Bool}
    (h : stretchE fm dm cnt = some (e, mp)) :
    (fm = .batch ∧ ∃ d L, dm = some (d, L) ∧ d * L = cnt ∧ e = L ∧ mp = true) ∨
      ((fm = .batch ∨ fm = .call) ∧ dm = none ∧ e = cnt ∧ mp = false) := by
  unfold stretchE at h
  cases fm
  · cases dm <;> simp at h
  · cases dm
    · have h' := Prod.mk.inj (Option.some.inj h)
      exact Or.inr ⟨Or.inr rfl, rfl, h'.1.symm, h'.2.symm⟩
    · simp at h
  · cases dm with
    | none =>
      have h' := Prod.mk.inj (Option.some.inj h)
      exact Or.inr ⟨Or.inl rfl, rfl, h'.1.symm, h'.2.symm⟩
    | some p =>
      obtain ⟨d, L⟩ := p
      change (if d * L == cnt then some (L, true) else none) = some (e, mp) at h
      split at h
      · rename_i hc
        have h' := Prod.mk.inj (Option.some.inj h)
        exact Or.inl ⟨rfl, d, L, rfl, by simpa using hc, h'.1.symm, h'.2.symm⟩
      · cases h
  · cases dm <;> simp at h

theorem stretchS_spec {fm : Flock.Program.Form} {dm : Option (Nat × Nat)} {cnt : Nat} {ps : List Part} {n : Nat} {b : Int}
    {q : Nat} {os : Int} {e : Nat} (h : stretchS fm dm cnt ps = some (n, b, q, os, e)) :
    ∃ mapped : Bool,
      ((fm = .batch ∧ ∃ d L, dm = some (d, L) ∧ d * L = cnt ∧ e = L ∧ mapped = true) ∨
        ((fm = .batch ∨ fm = .call) ∧ dm = none ∧ e = cnt ∧ mapped = false)) ∧
      ((∃ b0 c0, ps = [.affine false n b0 c0] ∧ b = (b0 : Int) ∧ q = 1 ∧ os = if mapped then (e : Int) else 0) ∨
        (∃ inner os' is c, ps = [.strided false n b inner os' is c] ∧ 1 ≤ e ∧ inner % e = 0 ∧ e ≤ inner ∧
          is = 1 ∧ (mapped = true ∨ inner = e) ∧ q = inner / e ∧ os = if mapped then os' else 0)) := by
  unfold stretchS at h
  rcases hE : stretchE fm dm cnt with _ | ⟨e0, mp⟩
  · rw [hE] at h
    simp at h
  · rw [hE] at h
    rcases ps with _ | ⟨p, _ | ⟨p2, ps'⟩⟩
    · simp at h
    · rcases p with ⟨pp, i, base, c⟩ | ⟨pp, i, base, inner, os', is, c⟩ | items | ⟨run, lanes⟩
      · cases pp
        · have h' := Prod.mk.inj (Option.some.inj h)
          obtain ⟨rfl, h2⟩ := h'
          obtain ⟨rfl, h3⟩ := Prod.mk.inj h2
          obtain ⟨rfl, h4⟩ := Prod.mk.inj h3
          obtain ⟨rfl, rfl⟩ := Prod.mk.inj h4
          exact ⟨mp, stretchE_spec hE, Or.inl ⟨base, c, rfl, rfl, rfl, rfl⟩⟩
        · simp at h
      · cases pp
        · change (if e0 ≥ 1 && inner % e0 == 0 && inner ≥ e0 && is == 1 && (mp || inner == e0) then
            some (i, base, inner / e0, if mp then os' else 0, e0) else none) = some (n, b, q, os, e) at h
          split at h
          · rename_i hc
            simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
            obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := hc
            have h' := Prod.mk.inj (Option.some.inj h)
            obtain ⟨rfl, h2'⟩ := h'
            obtain ⟨rfl, h3'⟩ := Prod.mk.inj h2'
            obtain ⟨rfl, h4'⟩ := Prod.mk.inj h3'
            obtain ⟨rfl, rfl⟩ := Prod.mk.inj h4'
            exact ⟨mp, stretchE_spec hE, Or.inr ⟨inner, os', is, c, rfl, h1, h2, h3, h4, h5, rfl, rfl⟩⟩
          · cases h
        · simp at h
      · simp at h
      · simp at h
    · simp at h

theorem seq_get_single (s : Seq) (p : Part) (h : s.parts.toList = [p]) (k : Nat) : s.get k = p.get k := by
  have hs : s.parts.size = 1 := by simpa using congrArg List.length h
  have hp : s.parts[0]'(by omega) = p := by
    have h0 : s.parts[0]? = some p := by
      rw [← Array.getElem?_toList, h]
      rfl
    rw [Array.getElem?_eq_getElem (by omega)] at h0
    exact Option.some.inj h0
  have hp' : s.parts[0]! = p := by
    rw [getElem!_pos s.parts 0 (by omega)]
    exact hp
  unfold Seq.get
  simp only [hs, beq_self_eq_true, if_true, hp']
  rfl

/-- Dividing a member's stretch by the block of `q` members' leaves. -/
theorem div_mod_stretch (m q e t : Nat) (hq : 0 < q) (hte : t < e) :
    (m * e + t) / (q * e) = m / q ∧ (m * e + t) % (q * e) = (m % q) * e + t := by
  have he : 0 < e := by omega
  obtain ⟨a, r, hr, rfl⟩ : ∃ a r, r < q ∧ m = q * a + r :=
    ⟨m / q, m % q, Nat.mod_lt _ hq, (Nat.div_add_mod m q).symm⟩
  have hlt : r * e + t < q * e := by
    calc r * e + t < r * e + e := by omega
      _ = (r + 1) * e := by ring
      _ ≤ q * e := Nat.mul_le_mul_right e hr
  have hsum : (q * a + r) * e + t = (r * e + t) + (q * e) * a := by ring
  rw [hsum, Nat.mul_add_div hq a r, Nat.div_eq_of_lt hr, Nat.add_zero, Nat.mul_add_mod q a r,
    Nat.mod_eq_of_lt hr]
  refine ⟨?_, ?_⟩
  · rw [Nat.add_mul_div_left _ _ (Nat.mul_pos hq he), Nat.div_eq_of_lt hlt, Nat.zero_add]
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt]

/-- **A member's parameter leaf, through the stretch `stretchOf` reads, is that leaf of the declared node**: where the
argument is the one part that `stretchOf` takes, `e` is the parameter's leaves, and the reference the member `m`'s leaf
`t` is is leaf `b + (m / q) · os + (m % q) · e + t` of node `n`, the leaf `hlo` keeps from below 0. -/
theorem stretch_ref {c : Defn} {nd : Node} (hno : InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk.NodeOk c nd)
    {i : Nat} (hi : i < nd.args.size) (hf : nd.form = .call ∨ nd.form = .batch) {n : Nat} {b : Int} {q : Nat}
    {os : Int} {e : Nat} (hs : stretchOf nd (nd.args[i]!) = some (n, b, q, os, e)) {m t : Nat}
    (hm : m < nd.members) (ht : t < (c.params[i]!).2.leaves) (hlo : 0 ≤ b + ((m / q : Nat) : Int) * os) :
    e = (c.params[i]!).2.leaves ∧ 0 < q ∧ ∃ z : Nat, (nd.args[i]!).seq.get (argIdx nd i m t) = ⟨false, n, z⟩ ∧
      (z : Int) = b + ((m / q : Nat) : Int) * os + ((m % q : Nat) : Int) * e + t := by
  rw [stretchOf_eq] at hs
  obtain ⟨mapped, hmap, hpart⟩ := stretchS_spec hs
  obtain ⟨hsz, hcnt, -, -, hsh, -⟩ := hno
  have hcnt' := hcnt i hi
  have key : e = (c.params[i]!).2.leaves ∧ argIdx nd i m t = if mapped = true then m * e + t else t := by
    rcases hmap with ⟨hb, d, L, hd, hdL, he, hmp⟩ | ⟨hfm, hd, he, hmp⟩
    · have hmem : nd.members = nd.n := bs_members nd (Or.inl hb)
      have hsh' := hsh
      rw [hb] at hsh'
      rcases hsh'.2 i hi with ⟨hnone, -⟩ | ⟨ax, mt, hrem, hmt, hdax⟩
      · rw [hnone] at hd
        cases hd
      · obtain ⟨L', hL'⟩ := InferenceOnlyVerifier.Record.SecurityProofs.axisDims_of_removeAxis ax _ hrem
        rw [hL', hd] at hdax
        obtain ⟨hdn, hLL⟩ := Prod.mk.inj (Option.some.inj hdax)
        have hl := leaves_removeAxis ax _ hrem
        rw [← hcnt', hmt] at hl
        have hnpos : 0 < d := by omega
        have hLeq : d * L = d * (c.params[i]!).2.leaves := by rw [hdL, hl, hdn]
        have hLe : L = (c.params[i]!).2.leaves := Nat.eq_of_mul_eq_mul_left hnpos hLeq
        have hlt : t < L := by omega
        have hx : argIdx nd i m t = m * L + t := by
          rw [argIdx_batch hb i m t]
          simp only [memberIdx, hd]
          rw [Nat.div_eq_of_lt hlt, Nat.mod_eq_of_lt hlt]
          simp
        refine ⟨by rw [he, hLe], ?_⟩
        rw [hx, hmp, he]
        simp
    · have hel : e = (c.params[i]!).2.leaves := by
        rw [he, hcnt']
        rcases hfm with hb | hc
        · have hsh' := hsh
          rw [hb] at hsh'
          rcases hsh'.2 i hi with ⟨-, hty⟩ | ⟨ax, mt, hrem, hmt, hdax⟩
          · rw [hty]
          · obtain ⟨L', hL'⟩ := InferenceOnlyVerifier.Record.SecurityProofs.axisDims_of_removeAxis ax _ hrem
            rw [hL', hd] at hdax
            cases hdax
        · have hsh' := hsh
          rw [hc] at hsh'
          rw [hsh'.2 i hi]
      refine ⟨hel, ?_⟩
      rw [hmp]
      simp only [Bool.false_eq_true, if_false]
      rcases hfm with hb | hc
      · rw [argIdx_batch hb i m t]
        simp [memberIdx, hd]
      · rw [argIdx_call hc]
  obtain ⟨hel, hx⟩ := key
  have het : t < e := by omega
  refine ⟨hel, ?_⟩
  rcases hpart with ⟨b0, c0, hps, hb0, hq1, hos⟩ | ⟨inner, os', is, c0, hps, he1, hdiv, hle, his, hmo, hqd, hos⟩
  · subst hq1
    refine ⟨Nat.one_pos, b0 + argIdx nd i m t, ?_, ?_⟩
    · rw [seq_get_single _ _ hps]
      rfl
    · rw [hx, hb0, hos]
      by_cases hmp : mapped = true
      · simp [hmp]
        ring
      · simp [hmp]
  · have hqe : inner = q * e := by
      rw [hqd]
      exact (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hdiv)).symm.trans (Nat.mul_comm _ _)
    have hqpos : 0 < q := by
      rw [hqd]
      exact Nat.div_pos hle (by omega)
    refine ⟨hqpos, ?_⟩
    have hxd : argIdx nd i m t / inner = (if mapped = true then m / q else 0) ∧
        argIdx nd i m t % inner = (if mapped = true then (m % q) * e + t else t) := by
      by_cases hmp : mapped = true
      · simp only [hx, hmp, if_true, hqe]
        exact div_mod_stretch m q e t hqpos het
      · have hie : inner = e := hmo.resolve_left hmp
        simp only [hx, hmp, if_false, hie]
        exact ⟨Nat.div_eq_of_lt het, Nat.mod_eq_of_lt het⟩
    rw [seq_get_single _ _ hps]
    refine ⟨(b + ((argIdx nd i m t / inner : Nat) : Int) * os' + ((argIdx nd i m t % inner : Nat) : Int) * is).toNat,
      rfl, ?_⟩
    rw [hxd.1, hxd.2, his, hos]
    by_cases hmp : mapped = true
    · simp only [hmp, if_true]
      have hnn : 0 ≤ b + ((m / q : Nat) : Int) * os' + (((m % q) * e + t : Nat) : Int) * 1 := by
        have : (0 : Int) ≤ (((m % q) * e + t : Nat) : Int) * 1 := by positivity
        have hlo' : 0 ≤ b + ((m / q : Nat) : Int) * os' := by
          rw [hos] at hlo
          simpa [hmp] using hlo
        linarith
      rw [Int.toNat_of_nonneg hnn]
      push_cast
      ring
    · have hie : inner = e := hmo.resolve_left hmp
      have hqq : q = 1 := by
        rw [hqd, hie]
        exact Nat.div_self (by omega)
      simp only [hmp, if_false, Bool.false_eq_true]
      have hnn : 0 ≤ b + ((0 : Nat) : Int) * os' + ((t : Nat) : Int) * 1 := by
        have : (0 : Int) ≤ ((t : Nat) : Int) * 1 := by positivity
        have hb' : 0 ≤ b := by
          rw [hos] at hlo
          simpa [hmp, hqq] using hlo
        have h0 : ((0 : Nat) : Int) * os' = 0 := by simp
        linarith
      rw [Int.toNat_of_nonneg hnn]
      subst hqq
      push_cast
      simp

/-- **The node a declared argument's stretch reads is a declared node**: where the argument has a part over a node of
`D`, the argument is the one part `stretchOf` takes, over the node it names. -/
theorem stretch_declared {nd : Node} {a : Arg} {D : List Nat} {part : Part}
    (hfind : a.seq.parts.toList.find? (InferenceOnlyVerifier.Record.overDeclared D) = some part) {n : Nat} {b : Int}
    {q : Nat} {os : Int} {e : Nat} (hs : stretchOf nd a = some (n, b, q, os, e)) : D.contains n = true := by
  rw [stretchOf_eq] at hs
  obtain ⟨mapped, -, hpart⟩ := stretchS_spec hs
  rcases hpart with ⟨b0, c0, hps, -⟩ | ⟨inner, os', is, c0, hps, -⟩
  · rw [hps] at hfind
    have hc : InferenceOnlyVerifier.Record.overDeclared D (Part.affine false n b0 c0) = true := by
      by_contra hne
      have : InferenceOnlyVerifier.Record.overDeclared D (Part.affine false n b0 c0) = false := by simpa using hne
      simp [List.find?, this] at hfind
    exact hc
  · rw [hps] at hfind
    have hc : InferenceOnlyVerifier.Record.overDeclared D (Part.strided false n b inner os' is c0) = true := by
      by_contra hne
      have : InferenceOnlyVerifier.Record.overDeclared D (Part.strided false n b inner os' is c0) = false := by
        simpa using hne
      simp [List.find?, this] at hfind
    exact hc

end InferenceOnlyVerifier.Properties.Record.SecurityProofs
