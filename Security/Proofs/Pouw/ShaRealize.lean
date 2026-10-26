import Definitions.Pouw.Guarantees.ShaRealize
import Definitions.Pouw.Assumptions.Dither
import Proofs.Pouw.MerkleExtract
import Proofs.Pouw.RelativeSecurity

/-!
# Commit-and-open as a theorem: `Extract` from SHA-512 as a random oracle and its realization

* `prob_bad_le`: over a period, an event that on every oracle is an extraction failure on the answer stream the
  oracle gives the log has probability at most `(q² + ∑_w q (2 τ_w + 1)) / |D|` over the oracle and the verifier's
  coins (`card_fails_oracle_le` at each value of the coins, then averaged).
* `link_noFold`: on the no-fold round, an accepted run on which some round's check fails on the walk's words is an
  extraction failure (Lemma 1's use: the opened word verifies against the logged root, hashes only logged points, and
  isn't the walk's word at its cell).
* `ExtractOfRealize`: the guarantee `Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`.
* `extractAdv_of_extract`: `Extract` implies `ExtractAdv`, so the advice form is the weaker premise.
-/

namespace Pouw.SecurityProofs.ShaRealize

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.MerkleExtract

/-- **The probability half over a period.** If on every oracle and every answer stream it gives the log at the coins
`x`, the event `Bad` implies an extraction failure at some round, then `Bad` has probability at most
`(q² + ∑_w q (2 τ_w + 1)) / |D|` over the oracle and the coins. -/
theorem prob_bad_le {P : RoundProto} {k : ℕ} {X D W : Type} [Fintype X] [DecidableEq X] [Fintype D]
    [DecidableEq D] [Inhabited D] (E : Enc X D W) (eo : P.Orc ≃ (X → D)) {q : ℕ} {τ : Fin k → ℕ}
    (L : QueryLog P k X D q τ) (Bad : P.Orc → (Fin k → P.Draw × P.Chk) → Prop)
    (hlink : ∀ o x u, Consistent (L.qry x) (eo o) u → Bad o x → ∃ w, ExtractFails E (L.prover x w) u) :
    prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => Bad ox.1 ox.2) ≤
      ((q * q + ∑ w, q * (2 * τ w + 1) : ℕ) : ℝ) / Fintype.card D := by
  classical
  refine prob_le_of_slices (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => Bad ox.1 ox.2) (by positivity)
    fun x => ?_
  show prob (fun o : P.Orc => Bad o x) ≤ _
  rw [prob_equiv eo]
  refine (prob_mono (q := fun h : X → D => ∃ u, Consistent (L.qry x) h u ∧ ∃ w, ExtractFails E (L.prover x w) u)
    fun h hb => ?_).trans ?_
  · obtain ⟨u, hu⟩ := exists_consistent (L.qry x) (L.qry_dep x) h
    exact ⟨u, hu, hlink _ x u (by rwa [Equiv.apply_symm_apply]) hb⟩
  · rw [prob_eq_card]
    have hc := card_fails_oracle_le E (L.qry x) (L.fresh x) (L.prover x)
    have hD : (0 : ℝ) < Fintype.card D := by exact_mod_cast Fintype.card_pos
    have hXD : (0 : ℝ) < Fintype.card (X → D) := by exact_mod_cast Fintype.card_pos
    rw [div_le_div_iff₀ hXD hD]
    convert (Nat.cast_le (α := ℝ)).mpr hc using 1 <;> push_cast <;> rfl

/-- **The deterministic link on the no-fold round.** Where the verifier's check is the Merkle path check under the
oracle along the cell's position (`hopens`), the rounds' roots are the log's (`hroot`), an accepted opening hashes only
logged points (`hN2`) and the words program is the walk over the log (`hwalk`): on every oracle, an accepted run on
which some round's check fails on the walk's words is an extraction failure on the answer stream. -/
theorem link_noFold {G : Accel} {nf : NoFold} {k : ℕ} {X D : Type} [Inhabited D] (E : Enc X D Word)
    (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D))
    (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) {q : ℕ} {τ : Fin k → ℕ}
    (hopens : ∀ o r c v p, nf.opens o r c v p → Opens (eo o) E (rD r) (pl p) v ∧ (pl p).map Prod.fst = dirs c)
    (A : Prover G (noFoldRound nf) k) (L : QueryLog (noFoldRound nf) k X D q τ) {Adv : Type}
    (adv : nf.Orc → (Fin k → nf.Draw × nf.Chk) → Fin k → Adv)
    (Mw : Fin k → G.Mc.Alg (Adv × nf.Draw) (Fin nf.n → Fin nf.n' → Fin nf.nb → Word))
    (hroot : ∀ o x u, Consistent (L.qry x) (eo o) u → ∀ w, rD (A.root o x w) = L.root x w u)
    (hN2 : ∀ o x u, Consistent (L.qry x) (eo o) u → A.Accepted o x → ∀ w (j : Fin nf.s),
      ∀ y ∈ pts E (pl ((A.ans o x w : Fin nf.s → Word × nf.Path) j).2) ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1,
        ∃ i, L.qry x i u = y)
    (hwalk : ∀ o x u, Consistent (L.qry x) (eo o) u → ∀ w (c : Fin nf.n × Fin nf.n' × Fin nf.nb),
      G.Mc.eval (Mw w) (adv o x w, (x w).1) c.1 c.2.1 c.2.2 =
        (ext ((L.prover x w).oracle u) E ((L.prover x w).pre u) (L.root x w u) (dirs c)).getD 0)
    (o : nf.Orc) (x : Fin k → nf.Draw × nf.Chk) (u : Fin q → D) (hu : Consistent (L.qry x) (eo o) u)
    (hacc : A.Accepted o x)
    (hbad : ∃ w, ¬ (noFoldRound nf).ok (A.instAt o x w) (x w).1 (x w).2 (G.Mc.eval (Mw w) (adv o x w, (x w).1))) :
    ∃ w, ExtractFails E (L.prover x w) u := by
  obtain ⟨w, hw⟩ := hbad
  obtain ⟨j, hj⟩ := not_forall.mp hw
  have hA : nf.opens o (A.root o x w) (nf.cells (x w).2 j) ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1
        ((A.ans o x w : Fin nf.s → Word × nf.Path) j).2 ∧
      ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1 = noFoldWord nf (nf.noise (A.instAt o x w) (x w).1)
        (nf.cells (x w).2 j).1 (nf.cells (x w).2 j).2.1 (nf.cells (x w).2 j).2.2 := hacc w j
  obtain ⟨hO, hdirs⟩ := hopens _ _ _ _ _ hA.1
  rw [hroot o x u hu w] at hO
  have hcov := hN2 o x u hu hacc w j
  refine ⟨w, pl ((A.ans o x w : Fin nf.s → Word × nf.Path) j).2, ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1,
    opens_congr E (fun y hy => ?_) hO, hcov, ?_⟩
  · obtain ⟨i, rfl⟩ := hcov y hy
    rw [show L.qry x i u = (L.prover x w).qry i u from rfl, Prover.oracle_qry]
    exact (hu i).symm
  · rw [hdirs]
    intro hext
    have hwc := hwalk o x u hu w (nf.cells (x w).2 j)
    rw [show (L.prover x w).root u = L.root x w u from rfl] at hext
    rw [hext, Option.getD_some] at hwc
    exact hj (hwc.trans hA.2)

/-- `Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`. -/
theorem ExtractOfRealize : Pouw.Guarantees.ShaRealize.ExtractOfRealize := by
  intro G nf k X D _ _ _ _ _ E eo rD pl dirs q τ hopens hR hG1 hG2 hP1 hP2 A
  obtain ⟨L, Adv, adv, Mw, Mh, hadv, hMh, huse, hext, hroot, hN2, hwalk⟩ := hR hG1 hG2 hP1 hP2 A
  refine ⟨Adv, adv, Mw, Mh, hadv, hMh, huse, hext, ?_⟩
  exact prob_bad_le E eo L
    (fun o x => A.Accepted o x ∧
      ∃ w, ¬ (noFoldRound nf).ok (A.instAt o x w) (x w).1 (x w).2 (G.Mc.eval (Mw w) (adv o x w, (x w).1)))
    fun o x u hu hb => link_noFold E eo rD pl dirs hopens A L adv Mw hroot hN2 hwalk o x u hu hb.1 hb.2

/-- A prover's state at the start of round `n` reads only the coins of the rounds before `n`. -/
theorem state_congr {G : Accel} {P : RoundProto} {k : ℕ} (A : Prover G P k) (o : P.Orc)
    {x x' : Fin k → P.Draw × P.Chk} :
    ∀ n : ℕ, (∀ v : Fin k, (v : ℕ) < n → x v = x' v) → A.state o x n = A.state o x' n
  | 0, _ => rfl
  | n + 1, h => by
    have ih := state_congr A o n fun v hv => h v (by omega)
    simp only [Prover.state]
    split_ifs with hn
    · rw [ih, h ⟨n, hn⟩ (by simp)]
    · exact ih

/-- **`Extract` implies its advice form**: the advice is the prover's state after the round's instance. -/
theorem extractAdv_of_extract {G : Accel} {P : RoundProto} {k : ℕ} {κx : ℝ} (hE : Extract G P k κx) :
    ExtractAdv G P k κx := by
  intro hG1 hG2 hP1 hP2 A
  obtain ⟨Mw, Mh, hMh, huse, hext, hprob⟩ := hE hG1 hG2 hP1 hP2 A
  refine ⟨A.St, fun o x w => (G.Mc.eval (A.inst w) (o, A.state o x w)).1, Mw, Mh, ?_, hMh, huse, hext, hprob⟩
  intro o x x' w hx
  simp only
  rw [state_congr A o (w : ℕ) fun v hv => hx v hv]

end Pouw.SecurityProofs.ShaRealize
