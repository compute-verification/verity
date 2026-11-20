module

public import Proofs.Properties.Flock.Circuit.LowerEval.Nested

/-! The class of templates whose batches map on any axis (`BatchNested`), and how removing a batch axis acts on types:
on the bit-level type, on the axis' dimensions, and on a member's leaf widths. -/

@[expose] public section

namespace Flock.Chain

open Flock Flock.Lower Flock.Tmpl Flock.TmplValue Flock.Chain.LowerDef
open Flock.Program hiding Form

theorem foldl_append_copies (w : Array ℕ) : ∀ (l : List ℕ) (acc : Array ℕ),
    (l.foldl (fun a _ => a ++ w) acc).toList = acc.toList ++ (List.replicate l.length w.toList).flatten
  | [], acc => by simp
  | _ :: l, acc => by
    rw [List.foldl_cons, foldl_append_copies w l, List.length_cons, List.replicate_succ, List.flatten_cons]
    simp [List.append_assoc]

theorem widths_array_flat (n : ℕ) (e : Ty) :
    (Ty.array n e).widths.toList = (List.replicate n e.widths.toList).flatten := by
  simp only [Ty.widths]
  rw [← Array.foldl_toList, Array.toList_range, foldl_append_copies, List.length_range]
  simp

theorem flatten_getElem?_sum {α : Type} : ∀ (L : List (List α)) {i j : ℕ} (hi : i < L.length)
    (hj : j < L[i].length), L.flatten[((L.take i).map List.length).sum + j]? = some L[i][j]
  | [], _, _, hi, _ => by simp at hi
  | a :: L, 0, j, _, hj => by simp [List.getElem?_append_left (show j < a.length from hj)]
  | a :: L, i + 1, j, hi, hj => by
    simp only [List.take_succ_cons, List.map_cons, List.sum_cons, List.flatten_cons, List.getElem_cons_succ]
    rw [Nat.add_assoc, List.getElem?_append_right (by omega), Nat.add_sub_cancel_left]
    exact flatten_getElem?_sum L (by simpa using hi) (by simpa using hj)

section BatchAxes
open InferenceOnlyVerifier.Properties.SecurityProofs

/-- **A template of batches on any axes over word primitives and earlier such templates**: `Batch0Nested`, but a batch
node may map each argument along any axis of its type (`AxesOk`: the argument's type with axis `ax` removed is the
node's length and its parameter's type). -/
inductive BatchNested (P : Prog) : ℕ → Prop
  | mk {d : ℕ} : (P.defs[d]!).prim = false → 0 < (P.defs[d]!).body.nodes.size →
      (∀ nd ∈ (P.defs[d]!).body.nodes, nd.form = .prim ∨ nd.form = .call ∨ nd.form = .batch) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key = false → nd.fn < d) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key = false →
        BatchNested P nd.fn) →
      BatchNested P d

theorem batchNested_of_batch0Nested {P : Prog} {d : ℕ} (h : Batch0Nested P d) : BatchNested P d := by
  induction h with
  | mk hp hn hform hlt _ ih =>
    refine .mk hp hn (fun nd hnd => ?_) hlt ih
    rcases hform nd hnd with h | h | ⟨h, -⟩
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr h)

theorem bitType_array (n : ℕ) (e : Ty) : Ty.bitType (.array n e) = .array n (Ty.bitType e) := by
  rw [Ty.bitType]

/-- **Removing a batch axis commutes with the bit-level type**: where axis `ax` of `t` has length `n` over `E`, axis `ax`
of `Ty.bitType t` has length `n` over `Ty.bitType E`. -/
theorem removeAxis_bitType : ∀ (ax : ℕ) {t E : Ty} {n : ℕ}, t.removeAxis ax = some (n, E) →
    (Ty.bitType t).removeAxis ax = some (n, Ty.bitType E) := by
  intro ax
  induction ax with
  | zero =>
    intro t E n h
    cases t with
    | array n' e =>
      simp only [Ty.removeAxis, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      rw [bitType_array]; rfl
    | _ => simp [Ty.removeAxis] at h
  | succ ax ih =>
    intro t E n h
    cases t with
    | array n' e =>
      simp only [Ty.removeAxis] at h
      cases he : e.removeAxis ax with
      | none => rw [he] at h; cases h
      | some p =>
        obtain ⟨m, inner⟩ := p
        rw [he] at h
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        rw [bitType_array, bitType_array]
        simp only [Ty.removeAxis, ih he, Option.map_some]
    | _ => simp [Ty.removeAxis] at h

/-- Where axis `ax` of `t` has length `n` over `E`, `axisDims` gives that length and the leaves `L` below the axis, which
divide `E`'s, and `t` has `n` times `E`'s leaves. -/
theorem axisDims_of_removeAxis : ∀ (ax : ℕ) {t E : Ty} {n : ℕ}, t.removeAxis ax = some (n, E) →
    ∃ L, t.axisDims ax = some (n, L) ∧ L ∣ E.leaves ∧ t.leaves = n * E.leaves := by
  intro ax
  induction ax with
  | zero =>
    intro t E n h
    cases t with
    | array n' e =>
      simp only [Ty.removeAxis, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨_, rfl, dvd_refl _, BodiesOk.leaves_array _ _⟩
    | _ => simp [Ty.removeAxis] at h
  | succ ax ih =>
    intro t E n h
    cases t with
    | array n' e =>
      simp only [Ty.removeAxis] at h
      cases he : e.removeAxis ax with
      | none => rw [he] at h; cases h
      | some p =>
        obtain ⟨m, inner⟩ := p
        rw [he] at h
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨L, hL, hdvd, hlv⟩ := ih he
        refine ⟨L, hL, ?_, ?_⟩
        · rw [BodiesOk.leaves_array]; exact Dvd.dvd.mul_left hdvd _
        · rw [BodiesOk.leaves_array, BodiesOk.leaves_array, hlv, Nat.mul_left_comm]
    | _ => simp [Ty.removeAxis] at h

/-- **A member of a batch argument is a strided slice of it**, in widths: where axis `ax` of `t` has length `n` over `E`
with `L` leaves below it, leaf `l` of member `m` has the width of leaf `((l / L) · n + m) · L + l % L` of `t`
(`Resolve.memberIdx`). -/
theorem removeAxis_widths : ∀ (ax : ℕ) {t E : Ty} {n L : ℕ}, t.removeAxis ax = some (n, E) →
    t.axisDims ax = some (n, L) → ∀ m < n, ∀ l < E.leaves,
      t.widths.toList[(l / L * n + m) * L + l % L]? = E.widths.toList[l]? := by
  intro ax
  induction ax with
  | zero =>
    intro t E n L h hd m hm l hl
    cases t with
    | array n' e =>
      simp only [Ty.removeAxis, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      simp only [Ty.axisDims, Option.some.injEq, Prod.mk.injEq] at hd
      obtain ⟨-, rfl⟩ := hd
      have hlen := Flock.Sigma.widths_length e
      rw [Nat.div_eq_of_lt hl, Nat.mod_eq_of_lt hl, Nat.zero_mul, Nat.zero_add, widths_array_flat, ← hlen,
        Nat.add_comm]
      exact replicate_flatten_getElem? _ hm (by rw [hlen]; exact hl)
    | _ => simp [Ty.removeAxis] at h
  | succ ax ih =>
    intro t E n L h hd m hm l hl
    cases t with
    | array n' e =>
      simp only [Ty.removeAxis] at h
      cases he : e.removeAxis ax with
      | none => rw [he] at h; cases h
      | some p =>
        obtain ⟨n0, inner⟩ := p
        rw [he] at h
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        have hde : e.axisDims ax = some (n0, L) := hd
        obtain ⟨L', hL', ⟨c, hc⟩, hlv⟩ := axisDims_of_removeAxis ax he
        rw [hde, Option.some.injEq, Prod.mk.injEq] at hL'
        obtain ⟨-, rfl⟩ := hL'
        rw [BodiesOk.leaves_array] at hl
        have hIL : 0 < inner.leaves := Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hl; omega
        have hLp : 0 < L := Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.zero_mul] at hc; omega
        have hq : l / inner.leaves < n' := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hl)
        have hr : l % inner.leaves < inner.leaves := Nat.mod_lt _ hIL
        have hl' : l = l / inner.leaves * inner.leaves + l % inner.leaves := (Nat.div_add_mod' l _).symm
        generalize l / inner.leaves = q at hq hl'
        generalize l % inner.leaves = r at hr hl'
        subst hl'
        have hrc : r / L < c := Nat.div_lt_of_lt_mul (by rw [← hc]; exact hr)
        have hdiv : (q * inner.leaves + r) / L = q * c + r / L := by
          rw [hc, Nat.mul_comm L c, ← Nat.mul_assoc, Nat.add_comm, Nat.add_mul_div_right _ _ hLp, Nat.add_comm]
        have hmod : (q * inner.leaves + r) % L = r % L := by
          rw [hc, Nat.mul_comm L c, ← Nat.mul_assoc, Nat.add_comm, Nat.add_mul_mod_self_right]
        have hidx : ((q * inner.leaves + r) / L * n0 + m) * L + (q * inner.leaves + r) % L =
            ((r / L * n0 + m) * L + r % L) + q * e.leaves := by
          rw [hdiv, hmod, hlv, hc]
          ring
        have hin : (r / L * n0 + m) * L + r % L < e.leaves := by
          have h1 := BodiesOk.lt_mul_of (r / L) c n0 m hrc hm
          have h2 := BodiesOk.lt_mul_of (r / L * n0 + m) (c * n0) L (r % L) h1 (Nat.mod_lt _ hLp)
          rw [hlv, hc]
          calc _ < c * n0 * L := h2
            _ = n0 * (L * c) := by rw [Nat.mul_comm L c, Nat.mul_comm c n0, Nat.mul_assoc]
        have hle := Flock.Sigma.widths_length e
        have hli := Flock.Sigma.widths_length inner
        rw [hidx, widths_array_flat, widths_array_flat, ← hle, ← hli, Nat.add_comm (q * _) r,
          replicate_flatten_getElem? _ hq (by rw [hle]; exact hin), replicate_flatten_getElem? _ hq (by rw [hli]; exact hr)]
        exact ih he hde m hm r hr
    | _ => simp [Ty.removeAxis] at h

theorem flatten_replicate_replicate {α : Type} (X : List α) (a b : ℕ) :
    (List.replicate a (List.replicate b X).flatten).flatten = (List.replicate (a * b) X).flatten := by
  induction a with
  | zero => simp
  | succ a ih =>
    rw [List.replicate_succ, List.flatten_cons, ih, show (a + 1) * b = b + a * b by ring, List.replicate_add,
      List.flatten_append]

/-- The bits before word leaf `j · |V| + c` of `k` copies of the widths `V`. -/
theorem sum_take_replicate (V : List ℕ) : ∀ (j k c : ℕ), j < k → c ≤ V.length →
    ((List.replicate k V).flatten.take (j * V.length + c)).sum = j * V.sum + (V.take c).sum := by
  intro j
  induction j with
  | zero =>
    intro k c hk hc
    obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    rw [List.replicate_succ, List.flatten_cons, Nat.zero_mul, Nat.zero_add, List.take_append_of_le_length hc]
    simp
  | succ j ih =>
    intro k c hk hc
    obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    rw [List.replicate_succ, List.flatten_cons, show (j + 1) * V.length + c = V.length + (j * V.length + c) by ring,
      List.take_length_add_append, List.sum_append, ih k c (by omega) hc]
    ring

/-- **Removing a batch axis, as blocks**: the widths of `t` are `o` blocks of `n` copies of `F`'s, the widths with the axis
removed `o` copies of `F`'s, and the axis' dimensions are `(n, F`'s leaves`)`. -/
theorem removeAxis_blocks : ∀ (ax : ℕ) {t E : Ty} {n : ℕ}, t.removeAxis ax = some (n, E) →
    ∃ (o : ℕ) (F : Ty), t.widths.toList = (List.replicate o (List.replicate n F.widths.toList).flatten).flatten ∧
      E.widths.toList = (List.replicate o F.widths.toList).flatten ∧ t.axisDims ax = some (n, F.leaves) := by
  intro ax
  induction ax with
  | zero =>
    intro t E n h
    match t, h with
    | .array n' e, h =>
      simp only [Ty.removeAxis, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨1, e, by rw [widths_array_flat]; simp, by simp, rfl⟩
    | .value _, h => simp [Ty.removeAxis] at h
    | .tuple _, h => simp [Ty.removeAxis] at h
    | .record _, h => simp [Ty.removeAxis] at h
  | succ ax ih =>
    intro t E n h
    match t, h with
    | .array n0 e, h =>
      simp only [Ty.removeAxis, Option.map_eq_some_iff, Prod.exists, Prod.mk.injEq] at h
      obtain ⟨m, inner, he, rfl, rfl⟩ := h
      obtain ⟨o, F, hw, hE, hd⟩ := ih he
      refine ⟨n0 * o, F, ?_, ?_, by simpa [Ty.axisDims] using hd⟩
      · rw [widths_array_flat, hw, flatten_replicate_replicate]
      · rw [widths_array_flat, hE, flatten_replicate_replicate]
    | .value _, h => simp [Ty.removeAxis] at h
    | .tuple _, h => simp [Ty.removeAxis] at h
    | .record _, h => simp [Ty.removeAxis] at h

/-- **A batch member's bits**: with `o` blocks of `n` copies of the widths `V` (`L` leaves, `S` bits), bit `t` of member
`m`'s word leaf `a · L + c` sits, among the whole's bits, at member `m`'s place for its bit `b` of the member's
`o` copies of `V`: `(b / S · n + m) · S + b % S`. -/
theorem member_bit (V : List ℕ) (o n a m c t : ℕ) (ha : a < o) (hm : m < n) (hc : c < V.length)
    (ht : t < V[c]) :
    ((List.replicate o (List.replicate n V).flatten).flatten.take ((a * n + m) * V.length + c)).sum + t =
      ((((List.replicate o V).flatten.take (a * V.length + c)).sum + t) / V.sum * n + m) * V.sum +
        (((List.replicate o V).flatten.take (a * V.length + c)).sum + t) % V.sum := by
  rw [flatten_replicate_replicate, sum_take_replicate V _ _ _ (BodiesOk.lt_mul_of a o n m ha hm) hc.le,
    sum_take_replicate V _ _ _ ha hc.le]
  have hlt : (V.take c).sum + t < V.sum := by
    have h1 : (V.take (c + 1)).sum = (V.take c).sum + V[c] := by
      rw [List.take_add_one, List.sum_append, List.getElem?_eq_getElem hc]; simp
    have h2 : (V.take (c + 1)).sum ≤ V.sum := by
      conv_rhs => rw [← List.take_append_drop (c + 1) V]
      rw [List.sum_append]; omega
    omega
  have hS : 0 < V.sum := by omega
  have hd : (a * V.sum + (V.take c).sum + t) / V.sum = a := by
    rw [Nat.add_assoc, Nat.add_comm, Nat.add_mul_div_right _ _ hS, Nat.div_eq_of_lt hlt, Nat.zero_add]
  have hmd : (a * V.sum + (V.take c).sum + t) % V.sum = (V.take c).sum + t := by
    rw [Nat.add_assoc, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hlt]
  rw [hd, hmd]
  ring

end BatchAxes

end Flock.Chain
