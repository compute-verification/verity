module

public import Proofs.Flock.Soundness.Discharge.ZkLink.DrawOk
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import Batteries.Tactic.OpenPrivate
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
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk
import Std.Data.TreeMap.Raw.Lemmas

/-!
# The sample file `flock-verify draw` prints decodes to the sample it printed

`flock-verify draw` prints the verifier's sample as `UnitDraw.toJson` (`Main.lean`'s `DRAW` line), and `verify` decodes the
record's `unit_draw` with `UnitDraw.ofJson`. For a stratified sample without a closure, the live law's, decoding what
`toJson` printed gives back the sample whenever `ofJson` accepts it (`ofJson_toJson_drawOf`), and likewise for a subset
sample, the `J`-table sessions' law (`ofJson_toJson_drawOf_subset`), and for a work sample without a closure
(`ofJson_toJson_drawOf_work`). `ofJson` also checks the sample against its sampling design (the
strata's tiling, each stratum's count, a subset sample's size, the units in the population), so it accepts only a sample the
sampling design can make; the round trip claims nothing about the others.

So `DrawFileZK`, the second half of `RecordCustodyZK` and of `RecordCustodyZKJ`, follows from each sample file being the
JSON value `flock-verify draw` printed for the verifier's own sample (`drawFileZK_of_toJson`, `drawFileZK_of_toJson_subset`,
`drawFileZK_of_toJson_work`), under every sampling design but a Bernoulli law (`drawFileZK_of_toJson_law`).
That equality is of Lean `Json` values, whose objects are trees shaped by the order their keys were inserted in:
`Flock.JsonParse.parse` of the keys in sorted order, as `compress` prints them, gives `toJson`'s trees, and other orders give other
trees that `ofJson` reads alike.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace FlockSoundness.Discharge.ZkLink

open Flock FlockSoundness.Refine Lean

open private stratumOfJson workStratumOfJson from Commitments.Draw

private theorem getNat?_toJson (x : ℕ) : (Lean.toJson x).getNat? = .ok x := rfl

private theorem asNat_toJson (x : ℕ) (c : String) : J.asNat (Lean.toJson x) c = .ok x := rfl

private theorem fromJson?_str (t : String) : (fromJson? (Json.str t) : Except String String) = .ok t := rfl

private theorem ok_bind {α β : Type} (a : α) (f : α → V β) : (Except.ok a >>= f) = f a := rfl

private theorem getObjVal?_mkObj {l : List (String × Json)} (hd : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq))
    {k : String} {v : Json} (hm : (k, v) ∈ l) : (Json.mkObj l).getObjVal? k = .ok v := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hd hm]
  rfl

private theorem getObjVal?_mkObj_not {l : List (String × Json)} {k : String} (hk : (l.map Prod.fst).contains k = false) :
    ∃ e, (Json.mkObj l).getObjVal? k = .error e := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false hk]
  exact ⟨_, rfl⟩

section
variable (t : String) (us : Array (ℕ × ℕ)) (k : ℕ)

private theorem stratum_template : (Draw.Stratum.toJson ⟨t, us, k⟩).getObjVal? "template" = .ok (.str t) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem stratum_k : (Draw.Stratum.toJson ⟨t, us, k⟩).getObjVal? "k" = .ok (Lean.toJson k) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem stratum_units : (Draw.Stratum.toJson ⟨t, us, k⟩).getObjVal? "units" =
    .ok (.arr (us.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])) :=
  getObjVal?_mkObj (by simp) (by simp)

end

private theorem mapM_map_ok {α β : Type} (f : β → V α) (g : α → β) (hf : ∀ a r, f (g a) = .ok r → r = a) :
    ∀ (l : List α) (r : List α), (l.map g).mapM f = .ok r → r = l
  | [], r, h => by simpa [pure, Except.pure, eq_comm] using h
  | a :: l, r, h => by
      simp only [List.map_cons, List.mapM_cons] at h
      obtain ⟨b, hb, h⟩ := ebind_ok h
      obtain ⟨bs, hbs, h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      rw [← h, hf a b hb, mapM_map_ok f g hf l bs hbs]

private theorem arr_mapM_map_ok {α β : Type} (f : β → V α) (g : α → β) (hf : ∀ a r, f (g a) = .ok r → r = a)
    (us r : Array α) (h : (us.map g).mapM f = .ok r) : r = us := by
  obtain ⟨l⟩ := us
  rw [List.map_toArray, List.mapM_toArray] at h
  cases hl : (l.map g).mapM f with
  | error e => rw [hl] at h; cases h
  | ok bs =>
      rw [hl] at h
      simp only [Functor.map, Except.map, Except.ok.injEq] at h
      rw [← h, mapM_map_ok f g hf l bs hl]

private theorem stratumOfJson_toJson (s s' : Draw.Stratum) (h : stratumOfJson s.toJson = .ok s') : s' = s := by
  obtain ⟨t, us, k⟩ := s
  unfold stratumOfJson at h
  simp only [Json.getObjValAs?, Json.getObjValD, stratum_template, stratum_k, stratum_units, Except.toOption,
    Option.getD, fromJson?_str, asNat_toJson] at h
  simp only [throw_bind'] at h
  repeat' walk_step
  have hk' := ‹Except.ok k = Except.ok _›
  have hm := ‹Array.mapM _ (Array.map _ us) = Except.ok _›
  simp only [pure, Except.pure, Except.ok.injEq] at h hk'
  subst h hk'
  rw [arr_mapM_map_ok _ _ (fun a r hr => ?_) us _ hm]
  obtain ⟨lo, hi⟩ := a
  simp only [Array.size, List.length, bne_self_eq_false, Bool.false_eq_true, ite_false] at hr
  rw [show (#[Lean.toJson lo, Lean.toJson hi] : Array Json)[0]! = Lean.toJson lo from rfl,
    show (#[Lean.toJson lo, Lean.toJson hi] : Array Json)[1]! = Lean.toJson hi from rfl, asNat_toJson, asNat_toJson] at hr
  change (if lo ≥ hi then throw _ else pure (lo, hi)) = Except.ok r at hr
  split at hr
  · exact absurd hr (throw_ne_ok _ _)
  · simpa [pure, Except.pure, eq_comm] using hr

section
variable (t : String) (us : Array (ℕ × ℕ)) (k w f : ℕ)

private theorem wstratum_floor : (Draw.WorkStratum.toJson ⟨⟨t, us, k⟩, w, f⟩).getObjVal? "floor" = .ok (Lean.toJson f) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem wstratum_k : (Draw.WorkStratum.toJson ⟨⟨t, us, k⟩, w, f⟩).getObjVal? "k" = .ok (Lean.toJson k) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem wstratum_template : (Draw.WorkStratum.toJson ⟨⟨t, us, k⟩, w, f⟩).getObjVal? "template" = .ok (.str t) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem wstratum_units : (Draw.WorkStratum.toJson ⟨⟨t, us, k⟩, w, f⟩).getObjVal? "units" =
    .ok (.arr (us.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem wstratum_work : (Draw.WorkStratum.toJson ⟨⟨t, us, k⟩, w, f⟩).getObjVal? "work" = .ok (Lean.toJson w) :=
  getObjVal?_mkObj (by simp) (by simp)

end

private theorem workStratumOfJson_toJson (s s' : Draw.WorkStratum) (h : workStratumOfJson s.toJson = .ok s') : s' = s := by
  obtain ⟨⟨t, us, k⟩, w, f⟩ := s
  unfold workStratumOfJson at h
  simp only [Json.getObjValAs?, Json.getObjValD, wstratum_floor, wstratum_k, wstratum_template, wstratum_units,
    wstratum_work, Except.toOption, Option.getD, fromJson?_str, asNat_toJson, ok_bind] at h
  simp only [throw_bind'] at h
  repeat' walk_step
  have hs := ‹stratumOfJson _ = Except.ok _›
  rw [stratumOfJson_toJson ⟨t, us, k⟩ _ hs] at h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  exact h.symm

section
variable (K pop : ℕ) (strata : Array Draw.Stratum) (us : Array ℕ)

private abbrev dS : Draw.UnitDraw := { law := .stratified K strata, population := pop, units := us }

private theorem draw_population : (dS K pop strata us).toJson.getObjVal? "population" = .ok (Lean.toJson pop) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem draw_law : (dS K pop strata us).toJson.getObjVal? "law" = .ok (.str "stratified") :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem draw_k : (dS K pop strata us).toJson.getObjVal? "k" = .ok (Lean.toJson K) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem draw_strata : (dS K pop strata us).toJson.getObjVal? "strata" = .ok (.arr (strata.map (·.toJson))) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem draw_units : (dS K pop strata us).toJson.getObjVal? "units" = .ok (.arr (us.map fun x => Lean.toJson x)) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem draw_closure : ∃ e, (dS K pop strata us).toJson.getObjVal? "closure" = .error e :=
  getObjVal?_mkObj_not (by simp)

private theorem ofJson_toJson_stratified {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson (dS K pop strata us).toJson = .ok D) :
    D = dS K pop strata us := by
  obtain ⟨e, he⟩ := draw_closure K pop strata us
  unfold Draw.UnitDraw.ofJson at h
  simp only [J.nat, J.arr, J.str, J.get, J.asStr, J.asArr, draw_population, draw_law, draw_k, draw_strata, draw_units,
    he, asNat_toJson, ok_bind, Json.getArr?, Json.getStr?, pure, Except.pure] at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    have hu := ‹Array.mapM (fun x => J.asNat x _) (Array.map _ us) = Except.ok _›
    have hs := ‹Array.mapM stratumOfJson (Array.map _ strata) = Except.ok _›
    rw [arr_mapM_map_ok _ _ (fun a r hr => by rw [asNat_toJson] at hr; exact (Except.ok.inj hr).symm) us _ hu,
      arr_mapM_map_ok _ _ (fun a r hr => stratumOfJson_toJson a r hr) strata _ hs] at h
    exact (Except.ok.inj h).symm

end

section
variable (k pop : ℕ) (us : Array ℕ)

private abbrev dSub : Draw.UnitDraw := { law := .subset k, population := pop, units := us }

private theorem sub_population : (dSub k pop us).toJson.getObjVal? "population" = .ok (Lean.toJson pop) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem sub_law : (dSub k pop us).toJson.getObjVal? "law" = .ok (.str "subset") :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem sub_k : (dSub k pop us).toJson.getObjVal? "k" = .ok (Lean.toJson k) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem sub_units : (dSub k pop us).toJson.getObjVal? "units" = .ok (.arr (us.map fun x => Lean.toJson x)) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem sub_closure : ∃ e, (dSub k pop us).toJson.getObjVal? "closure" = .error e :=
  getObjVal?_mkObj_not (by simp)

private theorem ofJson_toJson_subset {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson (dSub k pop us).toJson = .ok D) :
    D = dSub k pop us := by
  obtain ⟨e, he⟩ := sub_closure k pop us
  unfold Draw.UnitDraw.ofJson at h
  simp only [J.nat, J.arr, J.str, J.get, J.asStr, J.asArr, sub_population, sub_law, sub_k, sub_units,
    he, asNat_toJson, ok_bind, Json.getArr?, Json.getStr?, pure, Except.pure] at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    have hu := ‹Array.mapM (fun x => J.asNat x _) (Array.map _ us) = Except.ok _›
    rw [arr_mapM_map_ok _ _ (fun a r hr => by rw [asNat_toJson] at hr; exact (Except.ok.inj hr).symm) us _ hu] at h
    exact (Except.ok.inj h).symm

end

section
variable (K pop : ℕ) (strata : Array Draw.WorkStratum) (us : Array ℕ)

private abbrev dW : Draw.UnitDraw := { law := .work K strata, population := pop, units := us }

private theorem work_population : (dW K pop strata us).toJson.getObjVal? "population" = .ok (Lean.toJson pop) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem work_law : (dW K pop strata us).toJson.getObjVal? "law" = .ok (.str "work") :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem work_k : (dW K pop strata us).toJson.getObjVal? "k" = .ok (Lean.toJson K) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem work_strata : (dW K pop strata us).toJson.getObjVal? "strata" = .ok (.arr (strata.map (·.toJson))) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem work_units : (dW K pop strata us).toJson.getObjVal? "units" = .ok (.arr (us.map fun x => Lean.toJson x)) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem work_closure : ∃ e, (dW K pop strata us).toJson.getObjVal? "closure" = .error e :=
  getObjVal?_mkObj_not (by simp)

private theorem ofJson_toJson_work {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson (dW K pop strata us).toJson = .ok D) :
    D = dW K pop strata us := by
  obtain ⟨e, he⟩ := work_closure K pop strata us
  unfold Draw.UnitDraw.ofJson at h
  simp only [J.nat, J.arr, J.str, J.get, J.asStr, J.asArr, work_population, work_law, work_k, work_strata, work_units,
    he, asNat_toJson, ok_bind, Json.getArr?, Json.getStr?, pure, Except.pure] at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    have hu := ‹Array.mapM (fun x => J.asNat x _) (Array.map _ us) = Except.ok _›
    have hs := ‹Array.mapM workStratumOfJson (Array.map _ strata) = Except.ok _›
    rw [arr_mapM_map_ok _ _ (fun a r hr => by rw [asNat_toJson] at hr; exact (Except.ok.inj hr).symm) us _ hu,
      arr_mapM_map_ok _ _ (fun a r hr => workStratumOfJson_toJson a r hr) strata _ hs] at h
    exact (Except.ok.inj h).symm

end

variable {n : ℕ}

/-- **The round trip of the live law's sample file**: when `UnitDraw.ofJson` accepts what `UnitDraw.toJson` prints for the
verifier's own sample of `S` under a stratified sampling design, it decodes to that sample. -/
theorem ofJson_toJson_drawOf (K : ℕ) (strata : Array Draw.Stratum) (pop : ℕ) (S : Finset (Fin n)) {D : Draw.UnitDraw}
    (h : Draw.UnitDraw.ofJson (drawOf (.stratified K strata) pop S).toJson = .ok D) :
    D = drawOf (.stratified K strata) pop S :=
  ofJson_toJson_stratified K pop strata _ h

/-- **`DrawFileZK` from the printed sample**: a record whose sample file at every sample `S` is what `flock-verify draw`
printed for the verifier's own sample of `S` (`UnitDraw.toJson` of `drawOf`) meets `DrawFileZK`. -/
theorem drawFileZK_of_toJson (dj : Finset (Fin n) → Lean.Json) (K : ℕ) (strata : Array Draw.Stratum) (pop : ℕ)
    (hdj : ∀ S, dj S = (drawOf (.stratified K strata) pop S).toJson) : DrawFileZK dj (.stratified K strata) pop :=
  fun S _ hD => ofJson_toJson_drawOf K strata pop S (hdj S ▸ hD)

/-- **The round trip of the subset sampling design's sample file**: when `UnitDraw.ofJson` accepts what `UnitDraw.toJson` prints for
the verifier's own sample of `S` under `subset k`, it decodes to that sample. -/
theorem ofJson_toJson_drawOf_subset (k pop : ℕ) (S : Finset (Fin n)) {D : Draw.UnitDraw}
    (h : Draw.UnitDraw.ofJson (drawOf (.subset k) pop S).toJson = .ok D) :
    D = drawOf (.subset k) pop S :=
  ofJson_toJson_subset k pop _ h

/-- **`DrawFileZK` from the printed subset sample**: a record whose sample file at every sample `S` is what `flock-verify draw`
printed for the verifier's own sample of `S` under `subset k` meets `DrawFileZK` at that sampling design. -/
theorem drawFileZK_of_toJson_subset (dj : Finset (Fin n) → Lean.Json) (k pop : ℕ)
    (hdj : ∀ S, dj S = (drawOf (.subset k) pop S).toJson) : DrawFileZK dj (.subset k) pop :=
  fun S _ hD => ofJson_toJson_drawOf_subset k pop S (hdj S ▸ hD)

/-- **The round trip of the work sampling design's sample file**: when `UnitDraw.ofJson` accepts what `UnitDraw.toJson`
prints for the verifier's own sample of `S` under `work K strata`, with no closure, it decodes to that sample. -/
theorem ofJson_toJson_drawOf_work (K : ℕ) (strata : Array Draw.WorkStratum) (pop : ℕ) (S : Finset (Fin n))
    {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson (drawOf (.work K strata) pop S).toJson = .ok D) :
    D = drawOf (.work K strata) pop S :=
  ofJson_toJson_work K pop strata _ h

/-- **`DrawFileZK` from the printed work sample**: a record whose sample file at every sample `S` is what `flock-verify
draw` printed for the verifier's own sample of `S` under `work K strata` meets `DrawFileZK` at that sampling design. -/
theorem drawFileZK_of_toJson_work (dj : Finset (Fin n) → Lean.Json) (K : ℕ) (strata : Array Draw.WorkStratum) (pop : ℕ)
    (hdj : ∀ S, dj S = (drawOf (.work K strata) pop S).toJson) : DrawFileZK dj (.work K strata) pop :=
  fun S _ hD => ofJson_toJson_drawOf_work K strata pop S (hdj S ▸ hD)

/-- **`DrawFileZK` from the printed sample under any sampling design but a Bernoulli law**: the subset, stratified and
work round trips (`drawFileZK_of_toJson_subset`, `drawFileZK_of_toJson`, `drawFileZK_of_toJson_work`). -/
theorem drawFileZK_of_toJson_law (dj : Finset (Fin n) → Lean.Json) (law : Draw.Law)
    (hb : ∀ a b, law ≠ .bernoulli a b) (pop : ℕ) (hdj : ∀ S, dj S = (drawOf law pop S).toJson) :
    DrawFileZK dj law pop := by
  cases law with
  | subset k => exact drawFileZK_of_toJson_subset dj k pop hdj
  | bernoulli a b => exact absurd rfl (hb a b)
  | stratified K strata => exact drawFileZK_of_toJson dj K strata pop hdj
  | work K strata => exact drawFileZK_of_toJson_work dj K strata pop hdj

end FlockSoundness.Discharge.ZkLink
