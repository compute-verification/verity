module

public import Proofs.Properties.Flock.Circuit.LowerEval.PlacesAll

/-! How removing a batch axis acts on types (on the bit-level type, on the axis' dimensions, and on a member's leaf
widths), and what a member of a batch on any axis reads on bits (`batchBits_ax`). -/

@[expose] public section

namespace Flock.Chain

open Flock Flock.Lower Flock.Tmpl Flock.TmplValue Flock.Chain.LowerDef
open Flock.Program hiding Form

theorem replicate_flatten_getElem? {α : Type} (ew : List α) {n m t : ℕ} (hm : m < n) (ht : t < ew.length) :
    (List.replicate n ew).flatten[t + m * ew.length]? = ew[t]? := by
  obtain ⟨k, hk⟩ : ∃ k, n - m = k + 1 := ⟨n - m - 1, by omega⟩
  rw [replicate_flatten_split ew (Nat.le_of_lt hm), hk, List.replicate_succ, List.flatten_cons,
    List.getElem?_append_right (by rw [length_replicate_flatten]; omega), length_replicate_flatten,
    show t + m * ew.length - m * ew.length = t by omega, List.getElem?_append_left ht]

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
      E.widths.toList = (List.replicate o F.widths.toList).flatten ∧ t.axisDims ax = some (n, F.leaves) ∧
      (Ty.bitType t).axisDims ax = some (n, (Ty.bitType F).leaves) := by
  intro ax
  induction ax with
  | zero =>
    intro t E n h
    match t, h with
    | .array n' e, h =>
      simp only [Ty.removeAxis, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨1, e, by rw [widths_array_flat]; simp, by simp, rfl, by simp [Ty.bitType, Ty.axisDims]⟩
    | .value _, h => simp [Ty.removeAxis] at h
    | .tuple _, h => simp [Ty.removeAxis] at h
    | .record _, h => simp [Ty.removeAxis] at h
  | succ ax ih =>
    intro t E n h
    match t, h with
    | .array n0 e, h =>
      simp only [Ty.removeAxis, Option.map_eq_some_iff, Prod.exists, Prod.mk.injEq] at h
      obtain ⟨m, inner, he, rfl, rfl⟩ := h
      obtain ⟨o, F, hw, hE, hd, hbd⟩ := ih he
      refine ⟨n0 * o, F, ?_, ?_, by simpa [Ty.axisDims] using hd, by simpa [Ty.bitType, Ty.axisDims] using hbd⟩
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

theorem map_range_mul {β : Type} (f : ℕ → β) (K : ℕ) : ∀ o : ℕ,
    (List.range (o * K)).map f = (List.range o).flatMap fun a => (List.range K).map fun c => f (a * K + c)
  | 0 => by simp
  | o + 1 => by
    rw [show (o + 1) * K = o * K + K by ring, List.range_add, List.map_append, map_range_mul f K o, List.range_succ,
      List.flatMap_append, List.flatMap_singleton, List.map_map]
    rfl

theorem block_index {K a c : ℕ} (n m : ℕ) (hc : c < K) :
    (a * K + c) / K * n * K + (a * K + c) % K + m * K = c + (a * n + m) * K := by
  have hK : 0 < K := by omega
  rw [Nat.add_comm (a * K), Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hc, Nat.zero_add, Nat.add_mul_mod_self_right,
    Nat.mod_eq_of_lt hc]
  ring

/-- **`block_bits` on any axis**: with the references' widths `o` blocks of `n` copies of `V`, member `m` reads, on bits,
the bits of the words it reads, at the member's places `(t / K · n) · K + t % K + m · K`. -/
theorem block_bits_ax {α β : Type} [Inhabited α] (refs : List α) (G : α → List α) (Vb : α → β) (Vw : α → List β)
    (wid : α → ℕ) (V : List ℕ) (o n m : ℕ) (hm : m < n)
    (hG : ∀ r ∈ refs, (G r).map Vb = Vw r ∧ (Vw r).length = wid r)
    (hw : refs.map wid = (List.replicate o (List.replicate n V).flatten).flatten) :
    (List.range (o * V.sum)).map (fun t =>
        Vb ((refs.flatMap G).getD (t / V.sum * n * V.sum + t % V.sum + m * V.sum) default)) =
      ((List.range (o * V.length)).map fun t =>
        Vw (refs.getD (t / V.length * n * V.length + t % V.length + m * V.length) default)).flatten := by
  rw [map_range_mul, map_range_mul, flatten_flatMap]
  apply List.flatMap_congr
  intro a ha
  have hb := block_bits refs G Vb Vw wid V (o * n) (a * n + m)
    (BodiesOk.lt_mul_of a o n m (List.mem_range.1 ha) hm) hG (by rw [hw, flatten_replicate_replicate])
  refine (List.map_congr_left fun c hc => ?_).trans (hb.trans (congrArg List.flatten
    (List.map_congr_left fun c hc => ?_)))
  · rw [block_index n m (List.mem_range.1 hc)]
  · rw [block_index n m (List.mem_range.1 hc)]

/-- **A batch argument's bits, on any axis**: member `m` of an argument whose references' widths are `o` blocks of `n`
copies of `V` reads on bits the bits of the words it reads. -/
theorem axisBits_ax {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {s : Seq}
    (hr : ∀ r ∈ s.toArray.toList, RefIn D outs.size r) {o n m : ℕ} (hm : m < n) {V : List ℕ}
    (hw : s.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
      (List.replicate o (List.replicate n V).flatten).flatten) :
    (List.range (o * V.sum)).map (fun t =>
      Flock.Program.refVal x.flatten (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t)))
        (outs.map List.flatten)
        ((Flock.Lift.liftSeq (atOf D) (homoOf D) s).toArray.getD (t / V.sum * n * V.sum + t % V.sum + m * V.sum)
          default)) =
      ((List.range (o * V.length)).map fun t =>
        Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs
          (s.toArray.getD (t / V.length * n * V.length + t % V.length + m * V.length) default)).flatten := by
  have hA : ∀ (A : Array Ref) (i : ℕ), A.getD i default = A.toList.getD i default := by
    intro A i
    rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, List.getD_eq_getElem?_getD]
  have hG : ∀ r ∈ s.toArray.toList,
      ((List.range (atOf D r.p r.i r.j).2).map fun t => ({ p := r.p, i := r.i, j := (atOf D r.p r.i r.j).1 + t } : Ref)).map
        (Flock.Program.refVal x.flatten (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t)))
          (outs.map List.flatten)) = Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs r ∧
      (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs r).length = (tyAt D r).widths[r.j]! := by
    intro r h
    rw [List.map_map, (atOf_in (hr r h)).1]
    exact refBits hx ho (hr r h)
  have hls := liftSeqRefs (atOf D) (homoOf D) s (fun r h w hw j => by
    rw [(atOf_in (hr r h)).1]; rw [(atOf_in (hr r h)).2] at hw; exact bitAt_homo hw j)
  have key := block_bits_ax s.toArray.toList _ _ _ (fun r => (tyAt D r).widths[r.j]!) V o n m hm hG hw
  rw [← hls] at key
  refine (List.map_congr_left (fun t _ => ?_)).trans (key.trans ?_)
  · rw [hA]
  · congr 1
    exact List.map_congr_left (fun t _ => by rw [hA])

theorem liftDef_arg_ax (D : Defn) (fnOf : ℕ → ℕ) (primOf : ℕ → Bool) (gatesOf : ℕ → ℕ) {k i ax : ℕ}
    (hk' : k < (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.size) (hi : i < (D.body.nodes[k]'(by
      simpa [Flock.Lift.liftDef] using hk')).args.size)
    (hf : (D.body.nodes[k]'(by simpa [Flock.Lift.liftDef] using hk')).form = .batch)
    (hax : (D.body.nodes[k]'(by simpa [Flock.Lift.liftDef] using hk')).axes[i]? = some (some ax)) :
    (((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).args.getD i default).dims =
      (Ty.bitType ((D.body.nodes[k]'(by simpa [Flock.Lift.liftDef] using hk')).args.getD i default).ty).axisDims ax := by
  have hsz : ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).args.size =
      (D.body.nodes[k]'(by simpa [Flock.Lift.liftDef] using hk')).args.size := by
    simp [Flock.Lift.liftDef]
  have hi' : i < ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).args.size := by rw [hsz]; exact hi
  have hg : ∀ (A : Array Arg) (h : i < A.size), A.getD i default = A[i] := fun A h => by simp [Array.getD, h]
  rw [hg _ hi', hg _ hi]
  simp [Flock.Lift.liftDef, hf, hax]

/-- Iteration `m` of a scan's sequence argument on bits, the slice `[m·L, (m+1)·L)` of its lifted references, is the
bits of the slice of its word references: `axisBits` at `t < L`. -/
theorem scanSliceBits {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {s : Seq}
    (hr : ∀ r ∈ s.toArray.toList, RefIn D outs.size r) {n m : ℕ} (hm : m < n) {ew : List ℕ}
    (hw : s.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) = (List.replicate n ew).flatten) :
    (List.range ew.sum).map (fun t =>
      Flock.Program.refVal x.flatten (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t)))
        (outs.map List.flatten) ((Flock.Lift.liftSeq (atOf D) (homoOf D) s).toArray.getD (t + m * ew.sum) default)) =
      ((List.range ew.length).map fun t =>
        Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs
          (s.toArray.getD (t + m * ew.length) default)).flatten := by
  have key : ∀ L t : ℕ, t < L → t / L * n * L + t % L + m * L = t + m * L := fun L t h => by
    rw [Nat.div_eq_of_lt h, Nat.mod_eq_of_lt h, Nat.zero_mul, Nat.zero_mul, Nat.zero_add]
  refine (List.map_congr_left fun t ht => ?_).trans ((axisBits hx ho hr hm hw).trans ?_)
  · simp only [key _ t (List.mem_range.1 ht)]
  · congr 1
    exact List.map_congr_left fun t ht => by simp only [key _ t (List.mem_range.1 ht)]

theorem flatten_take_drop {α : Type} : ∀ (o : List (List α)) (k : ℕ),
    o.flatten.take ((o.take k).map List.length).sum = (o.take k).flatten ∧
      o.flatten.drop ((o.take k).map List.length).sum = (o.drop k).flatten
  | [], _ => by simp
  | a :: o, 0 => by simp
  | a :: o, k + 1 => by
    obtain ⟨h1, h2⟩ := flatten_take_drop o k
    simp only [List.take_succ_cons, List.drop_succ_cons, List.map_cons, List.sum_cons, List.flatten_cons]
    exact ⟨by rw [List.take_append, List.take_of_length_le (Nat.le_add_right _ _), Nat.add_sub_cancel_left, h1],
      by rw [List.drop_append, List.drop_of_length_le (Nat.le_add_right _ _), List.nil_append,
        Nat.add_sub_cancel_left, h2]⟩

/-- **A scan's fold on words and on bits agree step by step**: where each iteration's step, from a carry of the
widths `cw`, returns words of the widths `ow` (the carry's first) and on bits their bits, the fold on bits from the
bits of a word carry and outputs ends at the bits of the fold on words, its carry of the widths `cw`. -/
theorem scanFold (cw ow : List ℕ) (hk : ow.take cw.length = cw) (N : ℕ)
    (stepW : List (List Bool) → ℕ → Option (List (List Bool))) (stepB : List Bool → ℕ → Option (List Bool))
    (h : ∀ carry m, m < N → carry.map List.length = cw → ∃ o, stepW carry m = some o ∧ o.map List.length = ow ∧
      stepB carry.flatten m = some o.flatten) :
    ∀ (ms : List ℕ) (carry ys : List (List Bool)), (∀ m ∈ ms, m < N) → carry.map List.length = cw →
      ∃ cr ys', ms.foldlM (fun (st : List (List Bool) × List (List Bool)) m => do
          let o ← stepW st.1 m; pure (o.take cw.length, st.2 ++ o.drop cw.length)) (carry, ys) = some (cr, ys') ∧
        cr.map List.length = cw ∧
        ys'.map List.length = ys.map List.length ++ (List.replicate ms.length (ow.drop cw.length)).flatten ∧
        ms.foldlM (fun (st : List Bool × List Bool) m => do
          let o ← stepB st.1 m; pure (o.take cw.sum, st.2 ++ o.drop cw.sum)) (carry.flatten, ys.flatten) =
          some (cr.flatten, ys'.flatten)
  | [], carry, ys, _, hc => ⟨carry, ys, rfl, hc, by simp, rfl⟩
  | m :: ms, carry, ys, hms, hc => by
    obtain ⟨o, hw, hl, hb⟩ := h carry m (hms m (List.mem_cons_self ..)) hc
    have hc' : (o.take cw.length).map List.length = cw := by rw [List.map_take, hl, hk]
    obtain ⟨cr, ys', h1, h2, h4, h3⟩ := scanFold cw ow hk N stepW stepB h ms (o.take cw.length)
      (ys ++ o.drop cw.length) (fun m' hm' => hms m' (List.mem_cons_of_mem _ hm')) hc'
    have hs : cw.sum = ((o.take cw.length).map List.length).sum := by rw [hc']
    obtain ⟨ht, hd⟩ := flatten_take_drop o cw.length
    rw [← hs] at ht hd
    refine ⟨cr, ys', ?_, h2, ?_, ?_⟩
    · simp only [List.foldlM_cons, hw, Option.bind_eq_bind, Option.bind_some]
      exact h1
    · rw [h4, List.map_append, List.map_drop, hl, List.length_cons, List.replicate_succ, List.flatten_cons,
        List.append_assoc]
    · simp only [List.foldlM_cons, hb, Option.bind_eq_bind, Option.bind_some]
      rw [ht, hd, ← List.flatten_append]
      exact h3

/-- A scan node's argument `i ≥ 1` at iteration `m`, as `nodeStep` reads it: a sequence argument's slice `m`, any
other whole, each at its callee parameter's leaf count. -/
def scanArg {W : Type} [Inhabited W] (v : Ref → W) (callee : Defn) (nd : Node) (m i : ℕ) : List W :=
  if i ≤ nd.nxs then Flock.Program.argVals v (nd.args.getD i default) (callee.params.getD i default).2.leaves
      fun t => t + m * (callee.params.getD i default).2.leaves
  else Flock.Program.argVals v (nd.args.getD i default) (callee.params.getD i default).2.leaves id

/-- `nodeStep`'s scan arm, its fold over the pair's projections. -/
theorem nodeStep_scan {W : Type} [Inhabited W] (I : Flock.Program.Wires Option W)
    (hI : ∀ {α : Type} (o : Option α), I.lift o = o) (P : Prog) (ev : ℕ → List W → Option (List W)) (d : ℕ)
    (x : List W) (defn : Defn) (outs : Array (List W)) (nd : Node) {callee : Defn} (hf : nd.form = .scan)
    (hlt : nd.fn < d) (hc : P.defs[nd.fn]? = some callee) :
    nodeStep I P ev d x defn outs nd =
      ((List.range nd.n).foldlM (fun (st : List W × List W) m => do
          let o ← ev nd.fn (st.1 ++ ((List.range nd.args.size).drop 1).flatMap
            (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts defn.params) outs) callee nd m))
          pure (o.take (callee.params.getD 0 default).2.leaves, st.2 ++ o.drop (callee.params.getD 0 default).2.leaves))
        (Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts defn.params) outs)
          (nd.args.getD 0 default) (callee.params.getD 0 default).2.leaves id, [])) >>=
        fun r => pure (outs.push (r.1 ++ r.2)) := by
  unfold nodeStep
  rw [if_neg (by omega), show I.lift P.defs[nd.fn]? = some callee from by rw [hI, hc]]
  simp only [Option.bind_eq_bind, Option.bind_some, hf]
  rfl

theorem paramBit {C C' : Defn} (hpar : C'.params.map (·.2) = C.params.map (·.2.bitType)) {i : ℕ}
    (hip : i < C.params.size) : (C'.params.getD i default).2 = Ty.bitType (C.params.getD i default).2 := by
  have hsb : C'.params.size = C.params.size := by simpa using congrArg Array.size hpar
  have hib : i < C'.params.size := by rw [hsb]; exact hip
  have h1 := congrArg (·[i]?) hpar
  simp only [Array.getElem?_map] at h1
  rw [Array.getElem?_eq_getElem hib, Array.getElem?_eq_getElem hip] at h1
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hib,
    Array.getElem?_eq_getElem hip]
  simpa using h1

/-- Slice `m` of an argument whose references' widths repeat `E`'s `n` times, on bits, at the lift's leaf count. -/
theorem argSliceBits {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {a a' : Arg}
    (ha' : a'.seq = Flock.Lift.liftSeq (atOf D) (homoOf D) a.seq)
    (hr : ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r) {n m : ℕ} (hm : m < n) {E : Ty}
    (hw : a.seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) = (List.replicate n E.widths.toList).flatten) :
    Flock.Program.argVals (Flock.Program.refVal x.flatten
        (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t))) (outs.map List.flatten)) a'
        (Ty.bitType E).leaves (fun t => t + m * (Ty.bitType E).leaves) =
      (Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) a E.leaves
        (fun t => t + m * E.leaves)).flatten := by
  simp only [Flock.Program.argVals]
  rw [ha', ← sum_widths E, show E.leaves = E.widths.toList.length by rw [Array.length_toList, hb_widths_size]]
  exact scanSliceBits hx ho hr hm hw

/-- An argument whose references have `E`'s widths, whole, on bits, at the lift's leaf count. -/
theorem argSharedBits {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {a a' : Arg}
    (ha' : a'.seq = Flock.Lift.liftSeq (atOf D) (homoOf D) a.seq)
    (hr : ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r) {E : Ty}
    (hw : a.seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) = E.widths.toList) :
    Flock.Program.argVals (Flock.Program.refVal x.flatten
        (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t))) (outs.map List.flatten)) a'
        (Ty.bitType E).leaves id =
      (Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) a E.leaves
        id).flatten := by
  have h := argSliceBits hx ho ha' hr (n := 1) (m := 0) Nat.one_pos (E := E) (by rw [hw]; simp)
  simp only [Nat.zero_mul, Nat.add_zero] at h
  exact h

/-- **A scan's argument `i ≥ 1` at iteration `m`, on bits**, `C` its callee on words and `C'` on bits: the lifted
node's reads the bits of the word node's. -/
theorem scanArgBits {D : Defn} {fnOf : ℕ → ℕ} {primOf : ℕ → Bool} {gatesOf : ℕ → ℕ} {x : List (List Bool)}
    (hx : x.map List.length = leafWidths D) {outs : Array (List (List Bool))} (ho : OutsOk D outs) {n : ℕ}
    (hnD : n < D.body.nodes.size) (hnD' : n < (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.size)
    (hsz : outs.size = n) {C C' : Defn} (hpar : C'.params.map (·.2) = C.params.map (·.2.bitType))
    (hargs : ∀ a ∈ (D.body.nodes[n]).args.toList,
      a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D n r)
    {i : ℕ} (hi : i < (D.body.nodes[n]).args.size) (hip : i < C.params.size)
    (hw : ((D.body.nodes[n]).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
      if i ≤ (D.body.nodes[n]).nxs then
        (List.replicate (D.body.nodes[n]).n (C.params.getD i default).2.widths.toList).flatten
      else (C.params.getD i default).2.widths.toList)
    {m : ℕ} (hm : m < (D.body.nodes[n]).n) :
    scanArg (Flock.Program.refVal x.flatten
        (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t))) (outs.map List.flatten))
        C' ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]) m i =
      (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]) m i).flatten := by
  obtain ⟨hseqi, -, -, -⟩ := liftDef_arg D fnOf primOf gatesOf hnD hnD' hi
  have hmemA : (D.body.nodes[n]).args.getD i default ∈ (D.body.nodes[n]).args.toList := by
    simp [Array.getD, hi]
  have hrefs' : ∀ r ∈ ((D.body.nodes[n]).args.getD i default).seq.toArray.toList, RefIn D outs.size r :=
    fun r h => by rw [hsz]; exact (hargs _ hmemA).2 r h
  have hnx : ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).nxs = (D.body.nodes[n]).nxs := by
    simp [Flock.Lift.liftDef]
  unfold scanArg
  rw [hnx, paramBit hpar hip]
  by_cases hle : i ≤ (D.body.nodes[n]).nxs
  · rw [if_pos hle] at hw
    rw [if_pos hle, if_pos hle]
    exact argSliceBits hx ho hseqi hrefs' hm hw
  · rw [if_neg hle] at hw
    rw [if_neg hle, if_neg hle]
    exact argSharedBits hx ho hseqi hrefs' hw

theorem mem_drop_range {i k : ℕ} (h : i ∈ (List.range k).drop 1) : 1 ≤ i ∧ i < k := by
  obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 h
  simp only [List.length_drop, List.length_range] at hj
  simp only [List.getElem_drop, List.getElem_range]
  omega

/-- **One scan iteration's inputs after the carry, on bits** (step 2b): the lifted node's arguments `i ≥ 1` at
iteration `m` read the bits of the word node's. -/
theorem scanRestBits {D : Defn} {fnOf : ℕ → ℕ} {primOf : ℕ → Bool} {gatesOf : ℕ → ℕ} {x : List (List Bool)}
    (hx : x.map List.length = leafWidths D) {outs : Array (List (List Bool))} (ho : OutsOk D outs) {n : ℕ}
    (hnD : n < D.body.nodes.size) (hnD' : n < (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.size)
    (hsz : outs.size = n) {C C' : Defn} (hpar : C'.params.map (·.2) = C.params.map (·.2.bitType))
    (hargs : ∀ a ∈ (D.body.nodes[n]).args.toList,
      a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D n r)
    (hcs : (D.body.nodes[n]).args.size = C.params.size)
    (hw : ∀ i < (D.body.nodes[n]).args.size, 1 ≤ i →
      ((D.body.nodes[n]).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
        if i ≤ (D.body.nodes[n]).nxs then
          (List.replicate (D.body.nodes[n]).n (C.params.getD i default).2.widths.toList).flatten
        else (C.params.getD i default).2.widths.toList)
    {m : ℕ} (hm : m < (D.body.nodes[n]).n) :
    ((List.range ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).args.size).drop 1).flatMap
        (scanArg (Flock.Program.refVal x.flatten
          (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t))) (outs.map List.flatten))
          C' ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]) m) =
      (((List.range (D.body.nodes[n]).args.size).drop 1).flatMap
        (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]) m)).flatten := by
  have hasz : ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).args.size = (D.body.nodes[n]).args.size := by
    simp [Flock.Lift.liftDef]
  rw [hasz, flatten_flatMap]
  apply List.flatMap_congr
  intro i hi
  obtain ⟨h1, h2⟩ := mem_drop_range hi
  exact scanArgBits hx ho hnD hnD' hsz hpar hargs h2 (by omega) (hw i h2 h1) hm

/-- **A batch member's arguments' bits, on any axis**, `C` its callee on words and `C'` on bits: member `m` of a lifted
batch node reads, on bits, the bits of the words its word node's member reads. -/
theorem batchBits_ax {D : Defn} {fnOf : ℕ → ℕ} {primOf : ℕ → Bool} {gatesOf : ℕ → ℕ} {x : List (List Bool)}
    (hx : x.map List.length = leafWidths D) {outs : Array (List (List Bool))} (ho : OutsOk D outs) {n : ℕ}
    (hnD : n < D.body.nodes.size) (hnD' : n < (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.size)
    (hsz : outs.size = n) (hf : (D.body.nodes[n]).form = .batch) {C C' : Defn}
    (hpar : C'.params.map (·.2) = C.params.map (·.2.bitType))
    (hargs : ∀ a ∈ (D.body.nodes[n]).args.toList,
      a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D n r)
    (hbat : ∀ i < (D.body.nodes[n]).args.size,
      (∀ ax, (D.body.nodes[n]).axes[i]? = some (some ax) →
        ((D.body.nodes[n]).args.getD i default).ty.removeAxis ax = some ((D.body.nodes[n]).n, (C.params.getD i default).2) ∧
        ((D.body.nodes[n]).args.getD i default).dims = ((D.body.nodes[n]).args.getD i default).ty.axisDims ax ∧
        i < C.params.size ∧
        ((D.body.nodes[n]).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
          ((D.body.nodes[n]).args.getD i default).ty.widths.toList) ∧
      ((∀ ax, (D.body.nodes[n]).axes[i]? ≠ some (some ax)) → ((D.body.nodes[n]).args.getD i default).dims = none))
    {m : ℕ} (hm' : m < (D.body.nodes[n]).n) :
    ((List.range ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).args.size).flatMap fun i =>
        batchArg (Flock.Program.refVal x.flatten
          (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t))) (outs.map List.flatten))
          C' ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]) m i) =
      ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
          batchArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]) m i).flatten := by
  have hasz : ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).args.size = (D.body.nodes[n]).args.size := by
    simp [Flock.Lift.liftDef]
  rw [hasz, flatten_flatMap]
  apply List.flatMap_congr
  intro i hi
  have hi' := List.mem_range.1 hi
  obtain ⟨hseqi, -, haxn, -⟩ := liftDef_arg D fnOf primOf gatesOf hnD hnD' hi'
  have hmemA : (D.body.nodes[n]).args.getD i default ∈ (D.body.nodes[n]).args.toList := by
    simp [Array.getD, hi']
  obtain ⟨hcnt, hrefs⟩ := hargs _ hmemA
  have hrefs' : ∀ r ∈ ((D.body.nodes[n]).args.getD i default).seq.toArray.toList, RefIn D outs.size r :=
    fun r h => by rw [hsz]; exact hrefs r h
  by_cases hax : ∃ ax, (D.body.nodes[n]).axes[i]? = some (some ax)
  · obtain ⟨ax, hax⟩ := hax
    obtain ⟨hrm, hdm, hip, hw⟩ := (hbat i hi').1 ax hax
    have hsb : C'.params.size = C.params.size := by simpa using congrArg Array.size hpar
    have hib : i < C'.params.size := by rw [hsb]; exact hip
    have hpb : (C'.params.getD i default).2 = Ty.bitType (C.params.getD i default).2 := by
      have h1 := congrArg (·[i]?) hpar
      simp only [Array.getElem?_map] at h1
      rw [Array.getElem?_eq_getElem hib, Array.getElem?_eq_getElem hip] at h1
      rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hib,
        Array.getElem?_eq_getElem hip]
      simpa using h1
    obtain ⟨o, F, htw, hEw, hdF, hbdF⟩ := removeAxis_blocks ax hrm
    have hd' : (((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).args.getD i default).dims =
        some ((D.body.nodes[n]).n, (Ty.bitType F).leaves) := by
      rw [liftDef_arg_ax D fnOf primOf gatesOf hnD' hi' hf hax, hbdF]
    rw [batchArg_some _ _ _ _ _ hd', batchArg_some _ _ _ _ _ (hdm.trans hdF), hpb]
    simp only [Flock.Program.argVals]
    rw [hseqi]
    generalize (C.params.getD i default).2 = E at hEw ⊢
    rw [← sum_widths E, ← sum_widths F, hEw, sum_replicate_flatten,
      show E.leaves = E.widths.toList.length by rw [Array.length_toList, hb_widths_size],
      show F.leaves = F.widths.toList.length by rw [Array.length_toList, hb_widths_size], hEw,
      length_replicate_flatten]
    exact axisBits_ax hx ho hrefs' hm' (hw.trans htw)
  · have hn0 : ∀ ax, (D.body.nodes[n]).axes[i]? ≠ some (some ax) := fun ax h => hax ⟨ax, h⟩
    rw [batchArg_none _ _ _ _ _ (haxn hf hn0), batchArg_none _ _ _ _ _ ((hbat i hi').2 hn0)]
    simp only [Flock.Program.argVals]
    rw [hseqi]
    exact seqBits hx ho hcnt hrefs'

end BatchAxes

end Flock.Chain
