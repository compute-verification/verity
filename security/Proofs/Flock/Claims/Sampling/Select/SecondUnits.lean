module

public import Proofs.Flock.Claims.Sampling.Select.SecondCount
public import Proofs.Flock.Soundness.Discharge.Exec.LawDraw
import Spot.Draw
import Proofs.Core.QSort
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Steps
import Proofs.Flock.Claims.Sampling.Select.Bounds
import Proofs.Flock.Soundness.ExecDraw
import Proofs.Flock.Soundness.ExecStrataOf

/-!
# The units a stratified or work sample returns, stratum by stratum

`Flock.Draw.stratifiedWith` samples each stratum's `k` local indices (`Flock.Draw.subset`), maps them to the stratum's
units, and returns the strata's units together, sorted. Over strata whose slots hold each unit once, the sample it
returns is therefore strictly ascending, lies in the strata's slots, and holds exactly each stratum's `k` units in that
stratum's slots: what `Draw.checkDraw` asks of a sample's units.

* `subset_nodup`: a `subset` sample that succeeds holds each index once, each below `n`.
* `pieces_count`: lists taken one from each stratum's slots, each without repeats, are together without repeats, and
  the ones in a stratum's slots are that stratum's list.
* `stratifiedWith_parts`: a stratified sample that ends is the strata's `subset` samples, as units, sorted.
* `stratifiedWith_units`, `drawUnits_work_units`: the three facts above, of `stratifiedWith` and of `Draw.drawUnits`
  under a work sampling design.
-/

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- A `subset` sample that succeeds holds each index once, each below `n`. -/
theorem subset_nodup {k n : ℕ} {src : ByteArray} {D : Array ℕ} (h : Flock.Draw.subset k n src = some D) :
    D.toList.Nodup ∧ ∀ x ∈ D.toList, x < n := by
  obtain ⟨π, hkn, hπ, hmem⟩ := FlockSoundness.Discharge.Exec.subset_spec h
  have hsz := Flock.Bridge.subset_size h
  constructor
  · have hfin : D.toList.toFinset = (Finset.range k).image π := by
      ext x
      rw [List.mem_toFinset, Array.mem_toList_iff, hmem, Finset.mem_image]
      constructor
      · rintro ⟨i, hi, rfl⟩
        exact ⟨i, Finset.mem_range.2 hi, rfl⟩
      · rintro ⟨i, hi, rfl⟩
        exact ⟨i, Finset.mem_range.1 hi, rfl⟩
    have hcard : D.toList.toFinset.card = D.toList.length := by
      rw [hfin, Finset.card_image_of_injective _ π.injective, Finset.card_range, Array.length_toList, hsz]
    exact Multiset.coe_nodup.1 (Multiset.toFinset_card_eq_card_iff_nodup.1 (by rw [Multiset.coe_card]; exact hcard))
  · intro x hx
    obtain ⟨i, hi, rfl⟩ := (hmem x).1 (Array.mem_toList_iff.1 hx)
    exact hπ i (by omega)

/-- A relation between two lists' entries that gives a property of the first's holds of each of its entries. -/
theorem forall₂_left {α β : Type} {R : α → β → Prop} {Q : α → Prop} (hR : ∀ a b, R a b → Q a) :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ a ∈ l₁, Q a
  | _, _, .nil, _, ha => nomatch ha
  | _, _, .cons h hs, a, ha => by
    rcases List.mem_cons.1 ha with rfl | ha
    · exact hR _ _ h
    · exact forall₂_left hR hs a ha

theorem forall₂_and {α β : Type} {R S : α → β → Prop} : ∀ {l₁ : List α} {l₂ : List β},
    List.Forall₂ R l₁ l₂ → List.Forall₂ S l₁ l₂ → List.Forall₂ (fun a b => R a b ∧ S a b) l₁ l₂
  | _, _, .nil, .nil => .nil
  | _, _, .cons h hs, .cons h' hs' => .cons ⟨h, h'⟩ (forall₂_and hs hs')

theorem forall₂_zipWith {α β γ : Type} {R : α → β → Prop} {R' : α → γ → Prop} (g : α → β → γ)
    (hR : ∀ a b, R a b → R' a (g a b)) : ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ →
      List.Forall₂ R' l₁ (List.zipWith g l₁ l₂)
  | _, _, .nil => .nil
  | _, _, .cons h hs => .cons (hR _ _ h) (forall₂_zipWith g hR hs)

/-- **Lists taken one from each stratum's slots**: over strata whose slots `E s` hold each unit once, lists `P`, one
from each stratum's slots and each without repeats, are together without repeats and in the strata's slots, and after
any units outside the strata's slots, the ones in a stratum's slots are as many as that stratum's list. -/
theorem pieces_count {α : Type} (E : α → List ℕ) {St : List α} {Ps : List (List ℕ)}
    (hF : List.Forall₂ (fun s P => (E s).Nodup → P.Nodup ∧ ∀ x ∈ P, x ∈ E s) St Ps) :
    (St.flatMap E).Nodup → Ps.flatten.Nodup ∧ (∀ x ∈ Ps.flatten, x ∈ St.flatMap E) ∧
      ∀ pre : List ℕ, (∀ x ∈ pre, x ∉ St.flatMap E) →
        List.Forall₂ (fun s P => ((pre ++ Ps.flatten).filter fun u => decide (u ∈ E s)).length = P.length) St Ps := by
  induction hF with
  | nil => exact fun _ => ⟨List.nodup_nil, fun _ hx => by simp at hx, fun _ _ => .nil⟩
  | @cons s P St Ps hsP hF ih =>
    intro hnd
    rw [List.flatMap_cons, List.nodup_append] at hnd
    obtain ⟨hEs, hnd', hdis⟩ := hnd
    obtain ⟨hP, hPE⟩ := hsP hEs
    obtain ⟨ih1, ih2, ih3⟩ := ih hnd'
    refine ⟨?_, ?_, fun pre hpre => .cons ?_ ?_⟩
    · rw [List.flatten_cons, List.nodup_append]
      exact ⟨hP, ih1, fun a ha b hb => hdis a (hPE a ha) b (ih2 b hb)⟩
    · intro x hx
      rw [List.flatten_cons, List.mem_append] at hx
      rw [List.flatMap_cons, List.mem_append]
      exact hx.imp (hPE x) (ih2 x)
    · have h1 : (pre.filter fun u => decide (u ∈ E s)) = [] :=
        List.filter_eq_nil_iff.2 fun u hu => by
          have hn := hpre u hu
          rw [List.flatMap_cons, List.mem_append, not_or] at hn
          simpa using hn.1
      have h2 : (P.filter fun u => decide (u ∈ E s)) = P :=
        List.filter_eq_self.2 fun u hu => by simpa using hPE u hu
      have h3 : (Ps.flatten.filter fun u => decide (u ∈ E s)) = [] :=
        List.filter_eq_nil_iff.2 fun u hu => by
          have hn : u ∉ E s := fun hs => hdis u hs u (ih2 u hu) rfl
          simpa using hn
      rw [List.flatten_cons, List.filter_append, List.filter_append, h1, h2, h3, List.nil_append, List.append_nil]
    · have hrest := ih3 (pre ++ P) fun x hx => by
        rcases List.mem_append.1 hx with hx | hx
        · have hn := hpre x hx
          rw [List.flatMap_cons, List.mem_append, not_or] at hn
          exact hn.2
        · exact fun hc => hdis x (hPE x hx) x hc rfl
      rw [List.flatten_cons, ← List.append_assoc]
      exact hrest

/-- **A stratified sample that ends is the strata's `subset` samples, as units, sorted**: one sample `Dl` per stratum,
of its `k` among its `size` local indices, the units `out` those index in order, and `D` those units sorted. -/
theorem stratifiedWith_parts (src : ByteArray) (strata : Array Flock.Draw.Stratum) (p p' : ℕ) (D : Array ℕ)
    (h : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) strata).run p = .ok (some D, p')) :
    ∃ (Ds : List (Array ℕ)) (out : Array ℕ),
      List.Forall₂ (fun (s : Flock.Draw.Stratum) (Dl : Array ℕ) => ∃ b, Flock.Draw.subset s.k s.size b = some Dl)
        strata.toList Ds ∧
      out.toList = (List.zipWith (fun (s : Flock.Draw.Stratum) (Dl : Array ℕ) => Dl.toList.map s.unit)
        strata.toList Ds).flatten ∧
      D = out.qsort (fun x1 x2 => decide (x1 < x2)) := by
  unfold Flock.Draw.stratifiedWith at h
  dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have hsize : ([:strata.size] : Std.Legacy.Range).size = strata.size := by
    show (strata.size - 0 + 1 - 1) / 1 = _
    omega
  rw [hsize, show ([:strata.size] : Std.Legacy.Range).start = 0 from rfl,
    show ([:strata.size] : Std.Legacy.Range).step = 1 from rfl] at h
  have loop : ∀ (l : List ℕ) (out : Array ℕ) (pos : ℕ) (r : Option (Option (Array ℕ))) (out' : Array ℕ) (pos₁ : ℕ),
      (forIn (m := StateT ℕ (Except String)) l ((none : Option (Option (Array ℕ))), out) (fun i __s => do
          let __x ← Flock.Draw.onWith (fun need => Draw.take src need) strata[i]!.need
            (Flock.Draw.subset strata[i]!.k strata[i]!.size)
          match __x with
          | some local_ => pure (ForInStep.yield (none, __s.2 ++ Array.map strata[i]!.unit local_))
          | _ => pure (ForInStep.done (some none, __s.2)))).run pos = .ok ((r, out'), pos₁) →
      r = none ∧ ∃ Ds : List (Array ℕ),
        List.Forall₂ (fun (s : Flock.Draw.Stratum) (Dl : Array ℕ) => ∃ b, Flock.Draw.subset s.k s.size b = some Dl)
          (l.map fun i => strata[i]!) Ds ∧
        out'.toList = out.toList ++ (List.zipWith
          (fun (s : Flock.Draw.Stratum) (Dl : Array ℕ) => Dl.toList.map s.unit) (l.map fun i => strata[i]!)
          Ds).flatten := by
    intro l
    induction l with
    | nil =>
      intro out pos r out' pos₁ hl
      simp only [List.forIn_nil, StateT.run_pure] at hl
      obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj hl)
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj h1
      exact ⟨rfl, [], List.Forall₂.nil, (List.append_nil _).symm⟩
    | cons i is ih =>
      intro out pos r out' pos₁ hl
      rw [List.forIn_cons, StateT.run_bind, StateT.run_bind] at hl
      cases ho : (Flock.Draw.onWith (fun need => Draw.take src need) strata[i]!.need
          (Flock.Draw.subset strata[i]!.k strata[i]!.size)).run pos with
      | error e => rw [ho] at hl; exact absurd hl (by rw [bind_error, bind_error]; simp)
      | ok x =>
        obtain ⟨v, pos₂⟩ := x
        obtain ⟨j, Dl, rfl, rfl, hst⟩ := onWith_take src _ (by unfold Flock.Draw.Stratum.need; omega) _ pos _ _ ho
        rw [ho, bind_ok] at hl
        dsimp only at hl
        rw [StateT.run_pure, pure_bind] at hl
        dsimp only at hl
        obtain ⟨hr, Ds, hF, hout⟩ := ih _ _ r out' pos₁ hl
        refine ⟨hr, Dl :: Ds, List.Forall₂.cons ⟨_, hst.2.2.1⟩ hF, ?_⟩
        rw [hout, Array.toList_append, Array.toList_map, List.map_cons, List.zipWith_cons_cons, List.flatten_cons,
          List.append_assoc]
  rw [StateT.run_bind] at h
  cases hl : (forIn (m := StateT ℕ (Except String)) (List.range' 0 strata.size 1)
      ((none : Option (Option (Array ℕ))), (#[] : Array ℕ)) _).run p with
  | error e => rw [hl] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨⟨r, out'⟩, pos₁⟩ := y
    rw [hl, bind_ok] at h
    dsimp only at h
    obtain ⟨rfl, Ds, hF, hout⟩ := loop _ _ _ _ _ _ hl
    rw [StateT.run_pure] at h
    obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj h)
    have hD : D = out'.qsort (fun x1 x2 => decide (x1 < x2)) := (Option.some.inj h1).symm
    have hmap : (List.range' 0 strata.size).map (fun i => strata[i]!) = strata.toList := by
      refine List.ext_getElem (by simp) fun i h₁ _ => ?_
      have hi : i < strata.size := by simpa using h₁
      simp [getElem!_pos strata i hi]
    rw [hmap] at hF hout
    exact ⟨Ds, out', hF, by simpa using hout, hD⟩

/-- The units of a `subset` sample of a stratum's local indices are without repeats and in the stratum's slots, where
those hold each unit once. -/
theorem piece_of_subset {s : Flock.Draw.Stratum} {b : ByteArray} {Dl : Array ℕ}
    (h : Flock.Draw.subset s.k s.size b = some Dl) (hnd : (expand s.units.toList).Nodup) :
    (Dl.toList.map s.unit).Nodup ∧ ∀ x ∈ Dl.toList.map s.unit, x ∈ expand s.units.toList := by
  obtain ⟨hD, hlt⟩ := subset_nodup h
  have hget : ∀ j ∈ Dl.toList, ∃ hj : j < (expand s.units.toList).length, s.unit j = (expand s.units.toList)[j] := by
    intro j hj
    have hj' : j < (expand s.units.toList).length := by rw [← stratum_size']; exact hlt j hj
    exact ⟨hj', by rw [stratum_unit, List.getD_eq_getElem _ _ hj']⟩
  constructor
  · refine List.Nodup.map_on (fun x hx y hy hxy => ?_) hD
    obtain ⟨hx', ex⟩ := hget x hx
    obtain ⟨hy', ey⟩ := hget y hy
    rw [ex, ey] at hxy
    exact (List.Nodup.getElem_inj_iff hnd).1 hxy
  · intro x hx
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hx
    obtain ⟨hj', ej⟩ := hget j hj
    rw [ej]
    exact List.getElem_mem hj'

/-- **A stratified sample over strata whose slots hold each unit once**: the units it returns are strictly ascending,
each in some stratum's slots, and each stratum's `k` of them in that stratum's. -/
theorem stratifiedWith_units (src : ByteArray) (strata : Array Flock.Draw.Stratum) (p p' : ℕ) (D : Array ℕ)
    (h : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) strata).run p = .ok (some D, p'))
    (hnd : (strata.toList.flatMap fun s => expand s.units.toList).Nodup) :
    D.toList.Pairwise (· < ·) ∧ (∀ u ∈ D.toList, u ∈ strata.toList.flatMap fun s => expand s.units.toList) ∧
      ∀ s ∈ strata.toList, (D.toList.filter fun u => decide (u ∈ expand s.units.toList)).length = s.k := by
  obtain ⟨Ds, out, hF, hout, rfl⟩ := stratifiedWith_parts src strata p p' D h
  have hR := forall₂_zipWith (R' := fun (s : Flock.Draw.Stratum) (P : List ℕ) =>
      (expand s.units.toList).Nodup → P.Nodup ∧ ∀ x ∈ P, x ∈ expand s.units.toList)
    (fun (s : Flock.Draw.Stratum) (Dl : Array ℕ) => Dl.toList.map s.unit)
    (fun s Dl hb hE => by obtain ⟨b, hb⟩ := hb; exact piece_of_subset hb hE) hF
  have hL := forall₂_zipWith (R' := fun (s : Flock.Draw.Stratum) (P : List ℕ) => P.length = s.k)
    (fun (s : Flock.Draw.Stratum) (Dl : Array ℕ) => Dl.toList.map s.unit)
    (fun s Dl hb => by
      obtain ⟨b, hb⟩ := hb
      rw [List.length_map, Array.length_toList]
      exact Flock.Bridge.subset_size hb) hF
  obtain ⟨h1, h2, h3⟩ := pieces_count (fun s : Flock.Draw.Stratum => expand s.units.toList) hR hnd
  have hperm : (out.qsort (fun x1 x2 => decide (x1 < x2))).toList.Perm
      (List.zipWith (fun (s : Flock.Draw.Stratum) (Dl : Array ℕ) => Dl.toList.map s.unit) strata.toList Ds).flatten :=
    hout ▸ StrataExec.qsort_perm _ out
  refine ⟨?_, fun u hu => h2 u (hperm.mem_iff.1 hu), fun s hs => ?_⟩
  · have hle : (out.qsort (fun x1 x2 => decide (x1 < x2))).toList.Pairwise (fun x y => x ≤ y) :=
      Verity.QSort.ByKey.qsort_sorted (fun x : ℕ => x) out
    have hne : (out.qsort (fun x1 x2 => decide (x1 < x2))).toList.Pairwise (fun x y => x ≠ y) :=
      hperm.nodup_iff.2 h1
    exact (hle.and hne).imp fun hab => lt_of_le_of_ne hab.1 hab.2
  · have hc := forall₂_left (Q := fun s : Flock.Draw.Stratum =>
        ((List.zipWith (fun (s : Flock.Draw.Stratum) (Dl : Array ℕ) => Dl.toList.map s.unit) strata.toList
          Ds).flatten.filter fun u => decide (u ∈ expand s.units.toList)).length = s.k)
      (fun s P hsP => hsP.1.trans hsP.2) (forall₂_and (h3 [] fun _ hx => nomatch hx) hL) s hs
    rw [← hc]
    exact (hperm.filter _).length_eq

/-- A work sampling design's sample that ends is its strata's stratified sample. -/
theorem drawUnits_work_run {K n read : ℕ} {ws : Array Flock.Draw.WorkStratum} {src : ByteArray} {units : Array ℕ}
    (h : Draw.drawUnits (.work K ws) n src = .ok (units, read)) :
    (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) (ws.map (·.toStratum))).run 0 =
      .ok (some units, read) := by
  unfold Draw.drawUnits at h
  cases hw : (Flock.Draw.drawWith (fun _ need => Draw.take src need) (.work K ws) n).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok x =>
    obtain ⟨r, read'⟩ := x
    rw [hw, bind_ok] at h
    obtain ⟨d, rfl⟩ : ∃ d, r = some d := by
      cases r with
      | none => exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
      | some d => exact ⟨d, rfl⟩
    dsimp only at h
    obtain ⟨hu, hr⟩ := Prod.mk.inj (Except.ok.inj h)
    subst hu
    subst hr
    unfold Flock.Draw.drawWith at hw
    dsimp only at hw
    rw [StateT.run_bind] at hw
    cases ho : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) (ws.map (·.toStratum))).run 0 with
    | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
    | ok y =>
      obtain ⟨v, p⟩ := y
      rw [ho, bind_ok, StateT.run_pure] at hw
      obtain ⟨h1, h2⟩ := Prod.mk.inj (Except.ok.inj hw)
      cases v with
      | none => exact absurd h1 (by simp)
      | some D =>
        obtain rfl := (Option.some.inj h1).symm
        subst h2
        rfl

/-- **A work sampling design's sample over strata whose slots are the units `[0, m)` in order**: the units it returns
are strictly ascending, below `m`, and each stratum's `k` of them in that stratum's slots. -/
theorem drawUnits_work_units {K n m read : ℕ} {ws : Array Flock.Draw.WorkStratum} {src : ByteArray}
    {units : Array ℕ} (h : Draw.drawUnits (.work K ws) n src = .ok (units, read))
    (hon : (ws.map (·.toStratum)).toList.flatMap (fun s => expand s.units.toList) = List.range m) :
    units.toList.Pairwise (· < ·) ∧ (∀ u ∈ units.toList, u < m) ∧
      ∀ s ∈ ws.toList, (units.toList.filter fun u => decide (u ∈ expand s.units.toList)).length = s.k := by
  obtain ⟨h1, h2, h3⟩ := stratifiedWith_units src _ 0 read units (drawUnits_work_run h)
    (hon ▸ List.nodup_range)
  refine ⟨h1, fun u hu => ?_, fun s hs => ?_⟩
  · have hu' := h2 u hu
    rw [hon] at hu'
    exact List.mem_range.1 hu'
  · exact h3 s.toStratum (by rw [Array.toList_map]; exact List.mem_map_of_mem hs)

end Properties.Spot.SecurityProofs
