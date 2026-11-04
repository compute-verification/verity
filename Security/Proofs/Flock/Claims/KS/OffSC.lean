import Proofs.Flock.Claims.KS.Refit
import Proofs.Flock.Claims.KS.ConflictTied

/-!
# Off the table without a self-clash, at the deployed reader

The plain decoder's verdict is right when the decoded output opens no level-0 position off `C` unless it self-clashes
(`PlainBound`). That needs, at a deployed run accepting with every looked-up column on `C`, no such opening. The
deployed reader keeps one column per position below `2^d` (`distinctBelow`); a position it is given is looked up, to a
column it was given there (`lookCol_distinctBelow_of_mem`). So a salted output whose level-0 openings all verify
against its own cap, sit below `2^d`, and are, with their rows, the reader's entries, opens a position off the oracle of
`C` without a self-clash only where the reader looks up a column off `C` (`exists_ne_of_off₀L_sc`): two verifying
openings of one position with different rows are a self-clash, so the opening off `C` and the column looked up there
have one row. Only which entries the reader is given matters, not their order. An accepted salted table's level-0
openings verify against its cap when both reps' level-0 requests have level 0's lanes and path length
(`verified_slotOpensL`, `verified_opens₀L`).
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Discharge.Exec

local notation "𝔽" => FlockLevel3.GF128

/-- **A position the reader is given below `2^d` is looked up**, to a column it was given there. -/
theorem lookCol_distinctBelow_of_mem {d : ℕ} {os : List (ℕ × List 𝔽)} {e : ℕ × List 𝔽} (he : e ∈ os)
    (hlt : e.1 < 2 ^ d) : ∃ u, lookCol (distinctBelow d os) e.1 = some u ∧ (e.1, u) ∈ os := by
  have hex : ∃ b ∈ distinctBelow d os, b.1 = e.1 := by
    by_contra hn
    push Not at hn
    have hall := (List.forall_mem_pwFilter (R := fun a b : ℕ × List 𝔽 => a.1 ≠ b.1)
      (fun {x y z} h => by
        by_cases hxy : x.1 = y.1
        · exact Or.inr (hxy ▸ h)
        · exact Or.inl hxy) e (os.filter fun p => decide (p.1 < 2 ^ d))).1
      (fun b hb => (hn b hb).symm)
    exact hall e (List.mem_filter.2 ⟨he, decide_eq_true hlt⟩) rfl
  obtain ⟨b, hb, hbe⟩ := hex
  obtain ⟨c, hc⟩ : ∃ c, (distinctBelow d os).find? (fun p => decide (p.1 = e.1)) = some c :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨b, hb, decide_eq_true hbe⟩)
  have hce : c.1 = e.1 := by simpa using List.find?_some hc
  refine ⟨c.2, by unfold lookCol; rw [hc]; rfl, ?_⟩
  have hcm : c ∈ distinctBelow d os := List.mem_of_find?_eq_some hc
  have := (List.mem_filter.1 ((List.pwFilter_sublist _).subset hcm)).1
  rw [← hce]
  exact this

/-- **The salted level-0 event without a self-clash is a looked-up column off the word**: when every level-0 opening
of the output verifies against its own cap `cap₀` and sits below `2^d`, every entry the reader is given is a level-0
opening's position and row, and every opening's is one, an output that opens a position off the oracle of `C` and does
not self-clash has the reader look up a column off `C`. -/
theorem exists_ne_of_off₀L_sc {D Sl : Type} (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (sch : FlockSoundness.Accounting.Schedule) (cap₀ : Model.Cap D)
    (out : Model.OutCL 𝔽 D Sl) {N : ℕ} (hN : N = 2 ^ (lvOf sch 0).logLen) (C : Fin N → List 𝔽) (d : ℕ)
    (os : List (ℕ × List 𝔽)) (hcap : out.cap₀ = cap₀)
    (hver : ∀ x ∈ opens₀L sch out, VerifiedL H E Lf cap₀ (lvOf sch 0).lanes (Model.pathLen (lvOf sch 0)) x)
    (hlt : ∀ x ∈ opens₀L sch out, x.1 < 2 ^ d)
    (hos : ∀ e ∈ os, ∃ x ∈ opens₀L sch out, (x.1, x.2.1) = e)
    (hin : ∀ x ∈ opens₀L sch out, (x.1, x.2.1) ∈ os)
    (hoff : Off₀L H E Lf sch cap₀ (oracleOf C) out) (hsc : ¬ SelfClashL H E Lf sch out) :
    ∃ (p : Fin N) (u : List 𝔽), lookCol (distinctBelow d os) p.val = some u ∧ u ≠ C p := by
  classical
  refine exists_ne_of_off₀L H E Lf sch cap₀ out hN C _ (fun p v hv => ?_) hoff
  unfold op₀L opOfL at hv
  obtain ⟨x, hx, rfl⟩ := Option.map_eq_some_iff.1 hv
  have hxm := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq] at hxm
  obtain ⟨hxo, hxp, -⟩ := hxm
  obtain ⟨u, hu, hum⟩ := lookCol_distinctBelow_of_mem (hin x hxo) (hlt x hxo)
  obtain ⟨y, hyo, hye⟩ := hos _ hum
  simp only [Prod.mk.injEq] at hye
  obtain ⟨hy1, hy2⟩ := hye
  refine ⟨u, by rw [← hxp]; exact hu, ?_⟩
  by_contra hne
  apply hsc
  left
  rw [hcap]
  refine ⟨y, hyo, x, hxo, hy1, hver y hyo, hver x hxo, fun h => hne ?_⟩
  rw [← hy2, h]

/-- **A rep's level-0 openings verify** against the output's cap when its openings are `OpensOKL` and its level-0
requests have `lanes` lanes and paths of length `depth`. -/
theorem verified_slotOpensL {D Sl : Type} (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (out : Model.OutCL 𝔽 D Sl) (r Q lanes depth : ℕ)
    (hok : Model.OpensOKL H E Lf out.cap₀ r (if r = 0 then out.pend.1 else out.pend.2) out.opens out.salts)
    (h0 : ∀ R ∈ (if r = 0 then out.pend.1 else out.pend.2).reqs, R.level = 0 → R.lanes = lanes ∧ R.depth = depth) :
    ∀ x ∈ slotOpensL out r 0 Q, VerifiedL H E Lf out.cap₀ lanes depth x := by
  intro x hx
  unfold slotOpensL at hx
  split at hx
  · simp at hx
  · rename_i req hreq
    have hlv : req.level = 0 := by simpa using List.find?_some hreq
    have hmem := List.mem_of_find?_eq_some hreq
    obtain ⟨hl, hd⟩ := h0 req hmem hlv
    obtain ⟨cap, hcap, hall⟩ := hok req hmem
    rw [hlv] at hcap hall
    obtain rfl : out.cap₀ = cap := by simpa [Model.capOf] using hcap
    obtain ⟨pj, hpj, rfl⟩ := List.mem_map.1 hx
    have hg := List.mem_zipIdx_iff_getElem?.1 hpj
    rw [List.getElem?_take] at hg
    split_ifs at hg with hq
    have hjl : pj.2 < req.positions.length := (List.getElem?_eq_some_iff.1 hg).1
    have hpos : req.positions.getD pj.2 0 = pj.1 := by simp [List.getD, hg]
    obtain ⟨h1, h2⟩ := hall pj.2 hjl
    rw [hl] at h1
    rw [hd, hpos] at h2
    exact ⟨h1, h2⟩

/-- **An accepted salted table's level-0 openings verify** against its cap, at level 0's lanes and path length, when
both reps' level-0 requests have them: acceptance is both reps' `OpensOKL` at the cap. -/
theorem verified_opens₀L {D Sl : Type} (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (sch : FlockSoundness.Accounting.Schedule) (cap : Model.Cap D)
    (pend : Model.PendL 𝔽 D × Model.PendL 𝔽 D) (os : Model.Opens 𝔽 D × Model.Salts Sl)
    (hacc : (tableOutL H E Lf cap pend os).accepted = true)
    (h0 : ∀ r < 2, ∀ R ∈ (if r = 0 then pend.1 else pend.2).reqs, R.level = 0 →
      R.lanes = (lvOf sch 0).lanes ∧ R.depth = Model.pathLen (lvOf sch 0)) :
    ∀ x ∈ opens₀L sch (tableOutL H E Lf cap pend os),
      VerifiedL H E Lf cap (lvOf sch 0).lanes (Model.pathLen (lvOf sch 0)) x := by
  have hok := hacc
  simp only [tableOutL, Bool.and_eq_true, decide_eq_true_eq] at hok
  intro x hx
  rcases List.mem_append.1 hx with hx | hx
  · exact verified_slotOpensL H E Lf _ 0 _ _ _ hok.2.1 (h0 0 (by norm_num)) x hx
  · exact verified_slotOpensL H E Lf _ 1 _ _ _ hok.2.2 (h0 1 (by norm_num)) x hx

end Claims.Flock
