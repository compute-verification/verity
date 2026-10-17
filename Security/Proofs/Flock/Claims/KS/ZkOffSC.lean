import Proofs.Flock.Claims.KS.ZkSessionW
import Proofs.Flock.Claims.KS.OffSC
import Proofs.Flock.Claims.KS.ZkEntries

/-!
# Off the masked table on a witness lane without a self-clash, at the deployed reader

`OffSC` at `--zk`, with the level-0 event on the witness lanes (`OffW₀Z`, which `session_sound_of_table_accZ_w` reads).
The masked reader keeps, of each opened level-0 row, its first `2^k0` symbols (`zkRepColsOf`), one column per position
below `2^d` (`distinctBelow`). So a masked output whose level-0 openings all verify against its own cap at `width₀`, sit
below `2^d`, and are, with their rows' witness lanes, the reader's entries, opens a position off the oracle of `C` on a
witness lane without a self-clash only where the reader looks up a column off `C` (`exists_ne_of_offW₀Z_sc`): two
verifying openings of one position with different rows are a self-clash, so the opening off `C` and the entry looked up
there have one row. `exists_ne_of_zk_entries` is it with the entries given as one list equation, which `ZkEntries`
gives at `Zk.verifyRep`'s acceptance (`zk_table_opens₀Z`).
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

/-- **`exists_ne_of_offW₀Z_sc` at a list of entries**: when the level-0 openings verify at `width₀` below `2^d` and
their positions and rows' witness lanes are `os`, an output that opens a position off the oracle of `C` on a witness
lane without a self-clash has the reader look up a column off `C`. -/
theorem exists_ne_of_zk_entries {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (cap₀ : Model.Cap D) (out : OutZC 𝔽 D Sl) {N : ℕ} (hN : N = 2 ^ (lvOf T.sch 0).logLen) (C : Fin N → List 𝔽)
    (d : ℕ) (os : List (ℕ × List 𝔽)) (hcap : out.cap₀ = cap₀)
    (hver : ∀ x ∈ opens₀Z T.sch out,
      VerifiedZ Hs.H Hs.E Hs.Lf cap₀ (width₀ T) (Model.pathLen (lvOf T.sch 0)) x ∧ x.1 < 2 ^ d)
    (hent : (opens₀Z T.sch out).map (fun x => (x.1, x.2.1.take (2 ^ T.sch.k0))) = os)
    (hoff : OffW₀Z Hs T cap₀ (oracleOf C) out) (hsc : ¬ SelfClashZ Hs T out) :
    ∃ (p : Fin N) (u : List 𝔽), lookCol (distinctBelow d os) p.val = some u ∧ u ≠ C p :=
  exists_ne_of_offW₀Z_sc Hs T cap₀ out hN C d os hcap (fun x hx => (hver x hx).1) (fun x hx => (hver x hx).2)
    (fun e he => by rw [← hent] at he; exact List.mem_map.1 he)
    (fun x hx => by rw [← hent]; exact List.mem_map_of_mem hx) hoff hsc

end Claims.Flock
