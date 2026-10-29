import Proofs.Flock.Soundness.TmplFoldDef

/-!
# The template fold's reference over the call tree

`Tmpl.rows` writes each copy's rows through `place`, one column at a time and never one twice. `foldRef_tree`: entry `s`
of `foldRef` is the sum over the call tree (`placeSum`) of every copy's own rows, each reference read through the copy's
binding (`Def.ref`) and uncancelled (`sum_canon`), plus the constant's row.
-/

open Flock Flock.Tmpl FlockLevel3 GF128

namespace FlockSoundness.TmplFold

/-- Row `x`'s share, its `A` columns `a` read by `FA x` and its `B` columns `b` by `FB x`. -/
noncomputable def rowShare (FA FB : ℕ → ℕ → GF128) (x : ℕ) (a b : Array ℕ) : GF128 :=
  (a.toList.map (FA x)).sum + (b.toList.map (FB x)).sum

theorem rowShare_empty (FA FB : ℕ → ℕ → GF128) (x : ℕ) : rowShare FA FB x #[] #[] = 0 := by
  simp [rowShare]

theorem rowShare_canon (FA FB : ℕ → ℕ → GF128) (x : ℕ) (a b : Array ℕ) :
    rowShare FA FB x (canon a) (canon b) = rowShare FA FB x a b := by
  simp only [rowShare, sum_canon]

/-- Every row written so far, its share summed. -/
noncomputable def accSum (FA FB : ℕ → ℕ → GF128) (acc : Acc) : GF128 :=
  ∑ x ∈ Finset.range acc.ra.size, rowShare FA FB x (acc.ra.getD x #[]) (acc.rb.getD x #[])

/-- The three arrays have one size, and a column not yet written has empty rows. -/
def AccOk (acc : Acc) : Prop :=
  acc.rb.size = acc.ra.size ∧ acc.set.size = acc.ra.size ∧
    ∀ c, acc.set.getD c true = false → acc.ra.getD c #[] = #[] ∧ acc.rb.getD c #[] = #[]

/-- **A write adds its row's share**, and keeps the unwritten columns empty. -/
theorem write_sum (FA FB : ℕ → ℕ → GF128) {acc acc' : Acc} {c : ℕ} {a b : Array ℕ} (hok : AccOk acc)
    (h : write acc c a b = .ok acc') :
    AccOk acc' ∧ accSum FA FB acc' = accSum FA FB acc + rowShare FA FB c a b := by
  obtain ⟨hrb, hset, hempty⟩ := hok
  unfold write at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  rename_i hs
  cases h
  simp only [Array.set!_eq_setIfInBounds]
  have hf : acc.set.getD c true = false := by simpa using hs
  have hc : c < acc.ra.size := by
    rw [← hset]
    by_contra hn
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)] at hf
    simp at hf
  obtain ⟨hea, heb⟩ := hempty c hf
  have hga : ∀ x, x ≠ c → (acc.ra.setIfInBounds c a).getD x #[] = acc.ra.getD x #[] := fun x hx => by
    simp [Array.getElem?_setIfInBounds_ne (Ne.symm hx)]
  have hgb : ∀ x, x ≠ c → (acc.rb.setIfInBounds c b).getD x #[] = acc.rb.getD x #[] := fun x hx => by
    simp [Array.getElem?_setIfInBounds_ne (Ne.symm hx)]
  have hca : (acc.ra.setIfInBounds c a).getD c #[] = a := by simp [hc]
  have hcb : (acc.rb.setIfInBounds c b).getD c #[] = b := by simp [hrb, hc]
  refine ⟨⟨by simp [hrb], by simp [hset], fun x hx => ?_⟩, ?_⟩
  · have hxc : x ≠ c := by
      rintro rfl
      simp [hset, hc] at hx
    have hx' : acc.set.getD x true = false := by
      simpa [Array.getElem?_setIfInBounds_ne (Ne.symm hxc)] using hx
    rw [hga x hxc, hgb x hxc]
    exact hempty x hx'
  · unfold accSum
    simp only [Array.size_setIfInBounds]
    have he : ∑ x ∈ (Finset.range acc.ra.size).erase c,
          rowShare FA FB x ((acc.ra.setIfInBounds c a).getD x #[]) ((acc.rb.setIfInBounds c b).getD x #[]) =
        ∑ x ∈ (Finset.range acc.ra.size).erase c, rowShare FA FB x (acc.ra.getD x #[]) (acc.rb.getD x #[]) :=
      Finset.sum_congr rfl fun x hx => by
        rw [hga x (Finset.ne_of_mem_erase hx), hgb x (Finset.ne_of_mem_erase hx)]
    rw [← Finset.add_sum_erase _ _ (Finset.mem_range.mpr hc), ← Finset.add_sum_erase _ _ (Finset.mem_range.mpr hc),
      he, hca, hcb, hea, heb, rowShare_empty, zero_add, add_comm]

theorem foldlM_sum {α : Type} (S : Acc → GF128) (Q : Acc → Prop) (f : Acc → α → V Acc) (g : α → GF128)
    (hf : ∀ acc a acc', f acc a = .ok acc' → Q acc → Q acc' ∧ S acc' = S acc + g a) :
    ∀ (l : List α) (acc acc' : Acc), l.foldlM f acc = .ok acc' → Q acc → Q acc' ∧ S acc' = S acc + (l.map g).sum
  | [], acc, acc', h, hq => by
    rw [List.foldlM_nil] at h
    cases h
    exact ⟨hq, by simp⟩
  | a :: l, acc, acc', h, hq => by
    rw [List.foldlM_cons] at h
    obtain ⟨b, hb, h⟩ := except_bind_ok h
    obtain ⟨hq1, hs1⟩ := hf acc a b hb hq
    obtain ⟨hq2, hs2⟩ := foldlM_sum S Q f g hf l b acc' h hq1
    refine ⟨hq2, ?_⟩
    rw [hs2, hs1, List.map_cons, List.sum_cons, add_assoc]

/-- **The call tree's rows**: Definition `d`'s copy at offset `o`, its inputs bound to `inp`, its own rows' shares, each
reference read through `Def.ref`, and every call's copies', as `place` writes them. -/
noncomputable def placeSum (FA FB : ℕ → ℕ → GF128) (defs : Array Def) (k : ℕ) : ℕ → ℕ → ℕ → List ℕ → GF128
  | 0, _, _, _ => 0
  | fuel + 1, d, o, inp =>
    match defs[d]? with
    | none => 0
    | some df =>
      (df.rows.toList.map fun r =>
          rowShare FA FB (o + r.1) (r.2.1.map (df.ref o inp k)) (r.2.2.map (df.ref o inp k))).sum +
        (df.calls.toList.map fun c => ((List.range c.count).map fun j =>
          placeSum FA FB defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j)).sum).sum

theorem place_sum (FA FB : ℕ → ℕ → GF128) (defs : Array Def) (k : ℕ) :
    ∀ (fuel d o : ℕ) (inp : List ℕ) (acc acc' : Acc), place defs k fuel d o inp acc = .ok acc' → AccOk acc →
      AccOk acc' ∧ accSum FA FB acc' = accSum FA FB acc + placeSum FA FB defs k fuel d o inp
  | 0, d, o, inp, acc, acc', h, _ => by
    rw [place] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h
  | fuel + 1, d, o, inp, acc, acc', h, hok => by
    rw [place] at h
    cases hd : defs[d]? with
    | none => simp [hd, throw, throwThe, MonadExceptOf.throw] at h
    | some df =>
      simp only [hd] at h
      obtain ⟨acc1, h1, h2⟩ := except_bind_ok h
      obtain ⟨hok1, hs1⟩ := foldlM_sum (accSum FA FB) AccOk
        (fun (acc : Acc) (r : ℕ × Array ℕ × Array ℕ) =>
          write acc (o + r.1) (canon (r.2.1.map (df.ref o inp k))) (canon (r.2.2.map (df.ref o inp k))))
        (fun (r : ℕ × Array ℕ × Array ℕ) =>
          rowShare FA FB (o + r.1) (r.2.1.map (df.ref o inp k)) (r.2.2.map (df.ref o inp k)))
        (fun a _ b hab ha => by
          obtain ⟨hb, hs⟩ := write_sum FA FB ha hab
          exact ⟨hb, by rw [hs, rowShare_canon]⟩) _ acc acc1 h1 hok
      obtain ⟨hok2, hs2⟩ := foldlM_sum (accSum FA FB) AccOk
        (fun (acc : Acc) (c : Flock.Tmpl.Call) => (List.range c.count).foldlM (fun acc j =>
          place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc) acc)
        (fun (c : Flock.Tmpl.Call) => ((List.range c.count).map fun j =>
          placeSum FA FB defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j)).sum)
        (fun a c b hab ha => foldlM_sum (accSum FA FB) AccOk
          (fun (acc : Acc) (j : ℕ) =>
            place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc)
          (fun (j : ℕ) =>
            placeSum FA FB defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j))
          (fun a' j b' hab' ha' => place_sum FA FB defs k fuel _ _ _ a' b' hab' ha') _ a b hab ha)
        _ acc1 acc' h2 hok1
      refine ⟨hok2, ?_⟩
      rw [hs2, hs1]
      simp only [placeSum, hd]
      ring

/-- `foldRef`'s summand at entry `s`, `A` side: reference `v` of row `x`, at their positions under `π`. -/
noncomputable def termA (pt : FoldPt) (π : Array ℕ) (s x v : ℕ) : GF128 :=
  if π.getD v 0 % 64 = s then of (pt.alpha * pt.e (π.getD x 0) * pt.g (π.getD v 0)) else 0

/-- The same, `B` side (no `α`). -/
noncomputable def termB (pt : FoldPt) (π : Array ℕ) (s x v : ℕ) : GF128 :=
  if π.getD v 0 % 64 = s then of (pt.e (π.getD x 0) * pt.g (π.getD v 0)) else 0

theorem replicate_getD (n c : ℕ) : (Array.replicate n (#[] : Array ℕ)).getD c #[] = #[] := by
  by_cases hc : c < n <;> simp [hc]

/-- **`foldRef` over the call tree**: entry `s < 64` is the root copy's `placeSum` plus the constant's row
`[const] · [const]`. -/
theorem foldRef_tree {t : Tmpl} {lays : Array Lay} {alpha : F128} {lw rho r P : Array F128}
    (hP : foldRef t lays alpha lw rho r = .ok P) {s : ℕ} (hs : s < 64) :
    of P[s]! =
      placeSum (termA { alpha, lw, rho, r, qc := 0 } (posMap t lays) s)
          (termB { alpha, lw, rho, r, qc := 0 } (posMap t lays) s) t.defs t.constPos (t.defs.size + 1) t.root 0 [] +
        rowShare (termA { alpha, lw, rho, r, qc := 0 } (posMap t lays) s)
          (termB { alpha, lw, rho, r, qc := 0 } (posMap t lays) s) t.constPos #[t.constPos] #[t.constPos] := by
  rw [foldRef_eq_rows] at hP
  obtain ⟨acc, hR, hP⟩ := except_bind_ok hP
  cases hP
  rw [getElem!_getD _ (by rw [foldRefRows_size]; exact hs), foldRefRows_get _ _ _ _ hs, list_range_sum]
  unfold Tmpl.rows at hR
  obtain ⟨a, ha, hw⟩ := except_bind_ok hR
  have h0 : AccOk (Acc.empty (t.defs.getD t.root default).cols) := by
    exact ⟨by simp [Acc.empty], by simp [Acc.empty], fun c _ => ⟨replicate_getD _ c, replicate_getD _ c⟩⟩
  obtain ⟨hok1, hs1⟩ := place_sum (termA { alpha, lw, rho, r, qc := 0 } (posMap t lays) s)
    (termB { alpha, lw, rho, r, qc := 0 } (posMap t lays) s) _ _ _ _ _ _ _ _ ha h0
  obtain ⟨-, hs2⟩ := write_sum (termA { alpha, lw, rho, r, qc := 0 } (posMap t lays) s)
    (termB { alpha, lw, rho, r, qc := 0 } (posMap t lays) s) hok1 hw
  have hz : accSum (termA { alpha, lw, rho, r, qc := 0 } (posMap t lays) s)
      (termB { alpha, lw, rho, r, qc := 0 } (posMap t lays) s) (Acc.empty (t.defs.getD t.root default).cols) = 0 := by
    unfold accSum
    exact Finset.sum_eq_zero fun x _ =>
      (congrArg₂ (rowShare _ _ x) (replicate_getD _ x) (replicate_getD _ x)).trans (rowShare_empty _ _ x)
  show accSum (termA { alpha, lw, rho, r, qc := 0 } (posMap t lays) s)
    (termB { alpha, lw, rho, r, qc := 0 } (posMap t lays) s) acc = _
  rw [hs2, hs1, hz, zero_add]

end FlockSoundness.TmplFold
