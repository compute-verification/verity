module

public import Proofs.Properties.Flock.Circuit.LowerEval

/-! The class of templates whose batches map on any axis (`BatchNested`), and how removing a batch axis acts on types:
on the bit-level type, on the axis' dimensions, and on a member's leaf widths. -/

@[expose] public section

namespace Flock.Chain

open Flock Flock.Lower Flock.Tmpl Flock.TmplValue Flock.Chain.LowerDef
open Flock.Program hiding Form

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

end BatchAxes

end Flock.Chain
