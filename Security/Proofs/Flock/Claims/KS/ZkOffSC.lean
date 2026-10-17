import Proofs.Flock.Claims.KS.ZkSessionW
import Proofs.Flock.Claims.KS.OffSC

/-!
# Off the masked table on a witness lane without a self-clash, at the deployed reader

`OffSC` at `--zk`, with the level-0 event on the witness lanes (`OffW₀Z`, which `session_sound_of_table_accZ_w` reads).
The masked reader keeps, of each opened level-0 row, its first `2^k0` symbols (`zkRepColsOf`), one column per position
below `2^d` (`distinctBelow`). So a masked output whose level-0 openings all verify against its own cap at `width₀`, sit
below `2^d`, and are, with their rows' witness lanes, the reader's entries, opens a position off the oracle of `C` on a
witness lane without a self-clash only where the reader looks up a column off `C` (`exists_ne_of_offW₀Z_sc`): two
verifying openings of one position with different rows are a self-clash, so the opening off `C` and the entry looked up
there have one row. An accepted masked table's level-0 openings verify against its cap at `width₀` when both reps'
level-0 requests have level 0's path length (`verified_slotOpensZ`, `verified_opens₀Z`).
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession

local notation "𝔽" => FlockLevel3.GF128

/-- **The masked level-0 event on a witness lane without a self-clash is a looked-up column off the word**: when every
level-0 opening of the output verifies against its own cap `cap₀` at `width₀` and sits below `2^d`, every entry the
reader is given is a level-0 opening's position and its row's first `2^k0` symbols, and every opening's is one, an
output that opens a position off the oracle of `C` on a witness lane and does not self-clash has the reader look up a
column off `C`. -/
theorem exists_ne_of_offW₀Z_sc {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap₀ : Model.Cap D)
    (out : OutZC 𝔽 D Sl) {N : ℕ} (hN : N = 2 ^ (lvOf T.sch 0).logLen) (C : Fin N → List 𝔽) (d : ℕ)
    (os : List (ℕ × List 𝔽)) (hcap : out.cap₀ = cap₀)
    (hver : ∀ x ∈ opens₀Z T.sch out, VerifiedZ Hs.H Hs.E Hs.Lf cap₀ (width₀ T) (Model.pathLen (lvOf T.sch 0)) x)
    (hlt : ∀ x ∈ opens₀Z T.sch out, x.1 < 2 ^ d)
    (hos : ∀ e ∈ os, ∃ x ∈ opens₀Z T.sch out, (x.1, x.2.1.take (2 ^ T.sch.k0)) = e)
    (hin : ∀ x ∈ opens₀Z T.sch out, (x.1, x.2.1.take (2 ^ T.sch.k0)) ∈ os)
    (hoff : OffW₀Z Hs T cap₀ (oracleOf C) out) (hsc : ¬ SelfClashZ Hs T out) :
    ∃ (p : Fin N) (u : List 𝔽), lookCol (distinctBelow d os) p.val = some u ∧ u ≠ C p := by
  classical
  subst hN
  obtain ⟨p, v, hv, i, hi, hne⟩ := hoff
  unfold op₀Z at hv
  obtain ⟨x, hx, rfl⟩ := Option.map_eq_some_iff.1 hv
  have hxm := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq] at hxm
  obtain ⟨hxo, hxp, -⟩ := hxm
  obtain ⟨u, hu, hum⟩ := lookCol_distinctBelow_of_mem (hin x hxo) (hlt x hxo)
  obtain ⟨y, hyo, hye⟩ := hos _ hum
  simp only [Prod.mk.injEq] at hye
  obtain ⟨hy1, hy2⟩ := hye
  refine ⟨p, u, by rw [← hxp]; exact hu, fun hu' => ?_⟩
  by_cases hrow : y.2.1 = x.2.1
  · apply hne
    show x.2.1.getD i.val 0 = oracleOf C i.val p.val
    unfold oracleOf
    simp only [p.isLt, ↓reduceDIte]
    rw [← hu', ← hy2, hrow, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hi]
  · apply hsc
    left
    rw [hcap]
    exact ⟨y, hyo, x, hxo, hy1, hver y hyo, hver x hxo, hrow⟩

/-- **A rep's level-0 openings verify** against the output's cap at `width₀` when its openings are `OpensOKZ` and its
level-0 requests have paths of length `depth`. -/
theorem verified_slotOpensZ {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (out : OutZC 𝔽 D Sl) (r Q depth : ℕ)
    (hok : OpensOKZ Hs.H Hs.E Hs.Lf (width₀ T) out.cap₀ r (if r = 0 then out.pend.1 else out.pend.2).lig
      out.opens.rows out.opens.salts)
    (h0 : ∀ R ∈ (if r = 0 then out.pend.1 else out.pend.2).lig.reqs, R.level = 0 → R.depth = depth) :
    ∀ x ∈ slotOpensZ out r 0 Q, VerifiedZ Hs.H Hs.E Hs.Lf out.cap₀ (width₀ T) depth x := by
  intro x hx
  unfold slotOpensZ at hx
  split at hx
  · simp at hx
  · rename_i req hreq
    have hlv : req.level = 0 := by simpa using List.find?_some hreq
    have hmem := List.mem_of_find?_eq_some hreq
    have hd := h0 req hmem hlv
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
    rw [ite_eq_left_of_eq_true _ _ (eq_true rfl)] at h1
    rw [hd, hpos] at h2
    exact ⟨h1, h2⟩

/-- **An accepted masked table's level-0 openings verify** against its cap at `width₀` and level 0's path length, when
both reps' level-0 requests have it: acceptance is both reps' `OpensOKZ` at the cap. -/
theorem verified_opens₀Z {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap : Model.Cap D)
    (pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl) (os : OpensZ 𝔽 D Sl) (hacc : (outZC Hs T cap pend os).accepted = true)
    (h0 : ∀ r < 2, ∀ R ∈ (if r = 0 then pend.1 else pend.2).lig.reqs, R.level = 0 →
      R.depth = Model.pathLen (lvOf T.sch 0)) :
    ∀ x ∈ opens₀Z T.sch (outZC Hs T cap pend os),
      VerifiedZ Hs.H Hs.E Hs.Lf cap (width₀ T) (Model.pathLen (lvOf T.sch 0)) x := by
  have hok := hacc
  simp only [outZC, Bool.and_eq_true, decide_eq_true_eq] at hok
  intro x hx
  rcases List.mem_append.1 hx with hx | hx
  · exact verified_slotOpensZ Hs T _ 0 _ _ hok.2.2.2.1 (h0 0 (by norm_num)) x hx
  · exact verified_slotOpensZ Hs T _ 1 _ _ hok.2.2.2.2 (h0 1 (by norm_num)) x hx

end Claims.Flock
