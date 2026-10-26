import Std.Data.String.ToNat
import Properties.Accounting.Compose
import Proofs.Accounting.HMu.Luck
import Proofs.Accounting.Served
import Proofs.Flock.Claims.Sampling.Select.Design
import Proofs.Flock.VBridge.Family.Names

/-!
# A declared circuit `declaredOf` accepts is the service's (`Properties.Compose.DeclaredUnits`)

* **units**: `declaredOf` accepted, so `Slots.layout` and `Partition.units` did, and `unitsChecked` holds of the layout
  and population it read (`declaredOf_ok`). Each non-input gate's unit is `unitOf`'s, the first population replay unit
  with a span holding it, which the first check finds; the second gives each input slot's gate an input and each
  population unit a number below `N_p`, which is `Cd.units`.
* **the draw** (`draw_bound`): `Select.parseLaw` reads `subset:K_d` as `{law: subset, k: K_d}` and refuses it at
  `K_d = 0` (`parseLaw_subset`: `subset:` and `toString K_d` split at the one `:` (`splitOn_colon`, the legacy
  `String.splitOn ":"` as `List.splitOn ':'` on the characters, as `FlockVBridge.Names.splitOn_nl` is for `"\n"`), and
  `toString K_d` is digits that read back as `K_d`). A subset law has no expansion (`deriveLaw_subset`), and
  `flockLaw` takes it as `subset K_d` with `K_d ≤ N_p` (`flockLaw_subset`). So an accepted draw that proves no unit of
  `U` is an accepted `drawUnits (subset K_d) N_p` that draws none (`select_ok`: the draw proves the units it draws),
  which `Properties.Served.SecurityProofs.drawMissesSubset` bounds.
-/

open SampledProofs SampledProofs.Select
open private split1 digits from SampledProofs.Select

namespace Properties.Compose.SecurityProofs

open Lean (Json)
open Accounting.Compose
open Pouw.RelativeSecurity (prob)
open Properties.SampledProofs.SecurityProofs (select_ok getD_k nat_toJson bind_eq_ok throw_ne throw_bind_ne)
open FlockVBridge.Names (ulen ulen_append ulen_cons utf8ByteSize_eq_ulen getAux_at extract_at)

/-! ## `parseLaw` of `subset:K` -/

theorem splitOnAux_colon (s : String) (pre mid post : List Char) (hs : s.toList = pre ++ mid ++ post)
    (hmid : ':' ∉ mid) (r : List String) :
    String.splitOnAux s ":" ⟨ulen pre⟩ ⟨ulen (pre ++ mid)⟩ 0 r =
      r.reverse ++ ((mid ++ post).splitOn ':').map String.ofList := by
  induction post generalizing pre mid r with
  | nil =>
    rw [String.splitOnAux, if_pos]
    · rw [extract_at s pre mid [] hs, List.reverse_cons, List.append_nil, List.splitOn_eq_splitOnP,
        List.splitOnP_eq_singleton]
      · rfl
      · intro x hx
        simp only [beq_eq_false_iff_ne, ne_eq]
        rintro rfl
        exact hmid hx
    · simp only [String.Pos.Raw.atEnd, utf8ByteSize_eq_ulen, hs, List.append_nil, ge_iff_le, decide_eq_true_eq]
      exact Nat.le_refl _
  | cons c post ih =>
    have hget : String.Pos.Raw.get s ⟨ulen (pre ++ mid)⟩ = c := by
      unfold String.Pos.Raw.get
      rw [hs, List.append_assoc]
      have := getAux_at (pre ++ mid) post c 0
      rw [show (0 : String.Pos.Raw).byteIdx + ulen (pre ++ mid) = ulen (pre ++ mid) from Nat.zero_add _] at this
      rw [← List.append_assoc]
      exact this
    have hnot : ¬ (String.Pos.Raw.atEnd s ⟨ulen (pre ++ mid)⟩ = true) := by
      simp only [String.Pos.Raw.atEnd, utf8ByteSize_eq_ulen, hs, ge_iff_le, decide_eq_true_eq, ulen_append, ulen_cons]
      have := c.utf8Size_pos
      omega
    rw [String.splitOnAux, if_neg hnot]
    have hcol : String.Pos.Raw.get ":" 0 = ':' := rfl
    have hsz : ':'.utf8Size = 1 := rfl
    have hmid' : ∀ x ∈ mid, (x == ':') = false := by
      intro x hx
      simp only [beq_eq_false_iff_ne, ne_eq]
      rintro rfl
      exact hmid hx
    by_cases hc : c = ':'
    · subst hc
      rw [if_pos (by rw [hget, hcol]; rfl)]
      dsimp only
      have hend : String.Pos.Raw.atEnd ":" (String.Pos.Raw.next ":" 0) = true := rfl
      rw [if_pos hend]
      have hnext : String.Pos.Raw.next s ⟨ulen (pre ++ mid)⟩ = ⟨ulen (pre ++ mid ++ [':'])⟩ := by
        unfold String.Pos.Raw.next
        rw [hget, String.Pos.Raw.add_char_eq, ulen_append (pre ++ mid), ulen_cons, hsz]
        rfl
      have hun : (String.Pos.Raw.unoffsetBy ⟨ulen (pre ++ mid ++ [':'])⟩ (String.Pos.Raw.next ":" 0)) =
          ⟨ulen (pre ++ mid)⟩ := by
        have : String.Pos.Raw.next ":" 0 = ⟨1⟩ := rfl
        rw [this]
        show (⟨ulen (pre ++ mid ++ [':']) - 1⟩ : String.Pos.Raw) = _
        rw [ulen_append (pre ++ mid), ulen_cons, hsz]
        rfl
      rw [hnext, hun, extract_at s pre mid _ hs]
      have := ih (pre ++ mid ++ [':']) [] (by rw [hs]; simp) (by simp) (String.ofList mid :: r)
      rw [List.append_nil] at this
      rw [this, List.nil_append, show (mid ++ ':' :: post).splitOn ':' = mid :: post.splitOn ':' from
        List.splitOnP_append_cons_of_forall_mem hmid' ':' rfl post]
      simp
    · rw [if_neg (by rw [hget, hcol, beq_iff_eq]; exact hc)]
      have hnext : (String.Pos.Raw.unoffsetBy ⟨ulen (pre ++ mid)⟩ 0).next s = ⟨ulen (pre ++ (mid ++ [c]))⟩ := by
        show String.Pos.Raw.next s ⟨ulen (pre ++ mid) - 0⟩ = _
        rw [Nat.sub_zero]
        unfold String.Pos.Raw.next
        rw [hget, String.Pos.Raw.add_char_eq, ← List.append_assoc, ulen_append (pre ++ mid), ulen_cons]
        simp [ulen]
      rw [hnext, ih pre (mid ++ [c]) (by rw [hs]; simp) (by
        simp only [List.mem_append, List.mem_singleton, not_or]; exact ⟨hmid, fun h => hc h.symm⟩) r]
      simp

/-- **The legacy `String.splitOn ":"` is `List.splitOn ':'` on the characters.** -/
theorem splitOn_colon (s : String) : s.splitOn ":" = (s.toList.splitOn ':').map String.ofList := by
  unfold String.splitOn
  rw [if_neg (by decide)]
  exact splitOnAux_colon s [] [] s.toList (by simp) (by simp) []

theorem digit_of_mem_toString (n : ℕ) (c : Char) (h : c ∈ (toString n).toList) : c.isDigit = true := by
  rw [show toString n = Nat.repr n from rfl, Nat.toList_repr] at h
  exact Nat.isDigit_of_mem_toDigits (by decide) (by decide) h

theorem colon_not_mem_toString (n : ℕ) : ':' ∉ (toString n).toList := fun h =>
  absurd (digit_of_mem_toString n ':' h) (by decide)

theorem splitOn_subset (n : ℕ) : ("subset:" ++ toString n).splitOn ":" = ["subset", toString n] := by
  have hpre : "subset:".toList = ['s', 'u', 'b', 's', 'e', 't'] ++ [':'] := by simp
  have hs6 : ∀ x ∈ ['s', 'u', 'b', 's', 'e', 't'], (x == ':') = false := by
    intro x hx
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl | rfl <;> rfl
  rw [splitOn_colon, String.toList_append, hpre, List.append_assoc, List.singleton_append, List.splitOn_eq_splitOnP,
    List.splitOnP_append_cons_of_forall_mem hs6 ':' rfl, List.splitOnP_eq_singleton]
  · have h6 : String.ofList ['s', 'u', 'b', 's', 'e', 't'] = "subset" := by decide
    simp only [List.map_cons, List.map_nil, String.ofList_toList, h6]
  · intro x hx
    simp only [beq_eq_false_iff_ne, ne_eq]
    rintro rfl
    exact colon_not_mem_toString n hx

theorem split1_subset (n : ℕ) : split1 ("subset:" ++ toString n) ":" = ("subset", toString n) := by
  unfold split1
  rw [splitOn_subset]
  rfl

theorem digits_toString (n : ℕ) : digits (toString n) = true := by
  unfold digits
  rw [Bool.and_eq_true, Bool.not_eq_true', String.isEmpty_eq_false_iff, String.all_eq_all_toSlice,
    String.Slice.all_bool_eq, String.copy_toSlice, List.all_eq_true]
  exact ⟨Nat.repr_ne_empty, fun c hc => digit_of_mem_toString n c hc⟩

theorem toNat!_toString (n : ℕ) : (toString n).toNat! = n := by
  have h := Nat.toNat?_repr n
  change (Nat.repr n).toNat! = n
  unfold String.toNat? String.Slice.toNat? at h
  unfold String.toNat! String.Slice.toNat!
  split at h
  · rename_i hn
    rw [if_pos hn]
    exact Option.some.inj h
  · exact absurd h (by simp)

/-- `subset:K`'s law object. -/
def subsetObj (k : ℕ) : Json := Json.mkObj [("law", "subset"), ("k", Lean.toJson k)]

/-- **`parseLaw` of `subset:K`** is `{law: subset, k: K}`, and a refusal at `K = 0`. -/
theorem parseLaw_subset (n : ℕ) (named : Json) (h : Select.parseLaw ("subset:" ++ toString n) = .ok named) :
    named = subsetObj n ∧ 1 ≤ n := by
  unfold Select.parseLaw at h
  rw [split1_subset] at h
  dsimp only at h
  rw [digits_toString, toNat!_toString] at h
  by_cases hn : 1 ≤ n
  · rw [if_pos (by simp [hn])] at h
    exact ⟨(Except.ok.inj h).symm, hn⟩
  · rw [if_neg (by simp [hn])] at h
    repeat' split at h
    all_goals first
      | exact absurd h (throw_ne _ _)
      | exact absurd h (throw_bind_ne _ _ _)
      | (exfalso; simp_all)

/-- **A subset law has no expansion.** -/
theorem deriveLaw_subset (d : Program) (query : Json) (k : ℕ) (table : Option Json) (derived : Option Flock.Draw.Law)
    (h : Draw.deriveLaw d query (subsetObj k) table = .ok derived) : derived = none := by
  unfold Draw.deriveLaw at h
  have hk : getD (subsetObj k) "law" = .str "subset" := rfl
  simp only [hk] at h
  exact (Except.ok.inj h).symm

/-- **`flockLaw` takes `subset:K`'s law object as `subset K`**, where `K` is at most the population. -/
theorem flockLaw_subset (k n : ℕ) (parsed : Flock.Draw.Law) (h : Draw.flockLaw (subsetObj k) n = .ok parsed) :
    parsed = .subset k ∧ k ≤ n := by
  have hk : getD (subsetObj k) "law" = .str "subset" := rfl
  unfold Draw.flockLaw at h
  simp only [hk] at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  split at h
  · rename_i k' hnat
    split at h
    · exact absurd h (throw_bind_ne _ _ _)
    · obtain rfl := (Except.ok.inj h).symm
      rw [subsetObj, getD_k, nat_toJson] at hnat
      cases hnat
      exact ⟨rfl, by omega⟩
  · exact absurd h (throw_ne _ _)

/-! ## The draw -/

theorem prob_false {Ω : Type} [Fintype Ω] : prob (fun _ : Ω => False) = 0 := by
  classical
  rw [Accounting.HMuProofs.prob_eq_card]
  simp

/-- **The draw conjunct**: the code's `subset:K_d` draw over a population of `N_p` units, accepted and proving no unit
of `U`, is no likelier than a uniform permutation's `K_d`-subset missing `U`. -/
theorem draw_bound (p : Program) (query : Json) (Np Kd : ℕ) (hNp : Partition.units p query = .ok Np)
    (cm : Option (Array (Int × Array Int))) (L : ℕ) (U : Finset ℕ) :
    prob (fun src : Fin L → Fin 256 => ∃ r,
        Select.select (.law ("subset:" ++ toString Kd)) p query none cm
          (Properties.SampledProofs.bytesOf src) = .ok r ∧ ∀ u ∈ U, u ∉ r.2.1) ≤
      prob (fun β : Equiv.Perm (Fin Np) => ∀ t ∈ Pouw.Window.subsetDraw Kd β, t.val ∉ U) := by
  have hev : ∀ src : Fin L → Fin 256, (∃ r, Select.select (.law ("subset:" ++ toString Kd)) p query none cm
      (Properties.SampledProofs.bytesOf src) = .ok r ∧ ∀ u ∈ U, u ∉ r.2.1) →
      Kd ≤ Np ∧ ∃ r, Draw.drawUnits (.subset Kd) Np (Properties.SampledProofs.bytesOf src) = .ok r ∧
        ∀ u ∈ U, u ∉ r.1 := by
    rintro src ⟨⟨draw, proved, read⟩, hsel, hmiss⟩
    obtain ⟨n, named, derived, hn, hnamed, hderived, units, hsub, hcase⟩ := select_ok _ _ _ _ _ _ _ _ _ hsel
    obtain rfl : n = Np := Except.ok.inj (hn.symm.trans hNp)
    obtain ⟨rfl, -⟩ := parseLaw_subset Kd named hnamed
    obtain rfl := deriveLaw_subset _ _ _ _ _ hderived
    rcases hcase with ⟨h, -⟩ | ⟨-, parsed, read', hparsed, hdraw⟩
    · cases h
    obtain ⟨rfl, hK⟩ := flockLaw_subset Kd n parsed hparsed
    exact ⟨hK, ⟨units, read'⟩, hdraw, fun u hu hmem => hmiss u hu (hsub u hmem)⟩
  by_cases hK : Kd ≤ Np
  · exact le_trans (Accounting.HMuProofs.prob_mono fun src h => (hev src h).2)
      (Properties.Served.SecurityProofs.drawMissesSubset Kd Np L U hK)
  · refine le_trans (Accounting.HMuProofs.prob_mono fun src h => hK (hev src h).1) ?_
    rw [prob_false]
    exact Accounting.HMuProofs.prob_nonneg _

/-! ## The units -/

theorem declaredOf_ok {ι : Type} {Cd₀ Cd : Declared ι} {p : Program} {query : Json}
    (h : declaredOf Cd₀ p query = .ok Cd) :
    ∃ l Np, Slots.layout p query = .ok l ∧ Partition.units p query = .ok Np ∧ unitsChecked Cd₀ l Np = true ∧
      Cd = withUnits Cd₀ l Np := by
  unfold declaredOf at h
  obtain ⟨l, hl, h⟩ := bind_eq_ok.1 h
  obtain ⟨Np, hNp, h⟩ := bind_eq_ok.1 h
  split at h
  · exact ⟨l, Np, hl, hNp, by assumption, (Except.ok.inj h).symm⟩
  · exact absurd h (throw_ne _ _)

theorem declaredUnits : DeclaredUnits := by
  intro ι Cd₀ Cd p query h
  obtain ⟨l, Np₀, hl, hNp₀, hc, rfl⟩ := declaredOf_ok h
  simp only [unitsChecked, Bool.and_eq_true, List.all_eq_true] at hc
  obtain ⟨hgates, hrus⟩ := hc
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, fun Np hNp => ?_⟩
  obtain rfl : Np₀ = Np := Except.ok.inj (hNp₀.symm.trans hNp)
  refine ⟨rfl, fun Kd l' hl' => ?_⟩
  obtain rfl : l = l' := Except.ok.inj (hl.symm.trans hl')
  refine ⟨fun i hi hb => ?_, fun r hr hu g hg => ?_, fun r hr u hu => ?_, fun _ => draw_bound p query Np₀ Kd hNp₀⟩
  · have hi' := hgates i (List.mem_range.2 hi)
    have hb' : Cd₀.binding i = none := hb
    rw [hb'] at hi'
    simp only [Option.isSome_none, Bool.false_or, Option.isSome_iff_exists] at hi'
    obtain ⟨u, hu⟩ := hi'
    obtain ⟨r, hr, hru⟩ := List.exists_of_findSome?_eq_some hu
    rw [Option.filter_eq_some_iff] at hru
    obtain ⟨hru1, hru2⟩ := hru
    obtain ⟨g, hg, hgi⟩ := List.any_eq_true.1 hru2
    refine ⟨r, hr, ?_, g, hg, ?_⟩
    · show r.unit = some ((unitOf l i).getD 0)
      rw [hu, hru1]
      rfl
    · simpa using hgi
  · have hr' := hrus r hr
    rw [hu] at hr'
    simp only [List.all_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hr'
    exact hr' g hg
  · have hr' := hrus r hr
    rw [hu] at hr'
    show u < Np₀
    simpa using hr'

end Properties.Compose.SecurityProofs
