module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.Claims.Sampling.Reference.Complete

/-!
# The sampled compiler preserves completeness

`Compiled.Complete H L` from the proof system's completeness at its honest prover, the contract (`Sessions`: its
`fixed` and `slots`) and the completeness contract (`CompleteSessions`), as `Sampling.Reference.complete_of_sessions`
proves it at the deployed record. Each sampled proof block's session is fixed by the coins off its rounds (`fixed`), so
resampling a slot's session rounds (`Ex_resample`) leaves the slot's block and statement fixed, and the session's
coins are then uniform (`Pr_comp_inj`, along `emb`), so the proof system's completeness bounds the slot's rejection by
its error, capped at `1`. The union bound over the slots below `K x` gives `K x` times that, which the compiled error
covers (`CompleteSessions.err`).

* `Compiled.Pr_slot_le`: a sampled slot rejects with probability at most any bound on its sessions' errors.
* `Compiled.complete_of`: **`Compiled.Complete H L`**.
* `compilerComplete`: **`CompilerComplete`**.
-/

@[expose] public section

namespace Claims.SamplingGeneric.Compiled

open Claims Claims.Sampling Finset Properties.Spot Properties.Spot.SecurityProofs

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- The rounds of the proof block sampled at slot `p`, none when no block is. -/
def slotRounds (L : Compiled Coin A) (x : Msg) (p : ℕ) (r : Fin (L.V.k x) → Coin) : Finset (Fin (L.V.k x)) :=
  match (L.checked x r)[p]? with
  | some u => L.rounds x u
  | none => ∅

omit [Nonempty Coin] in
theorem slotRounds_fork {H : Msg → Msg} (L : Compiled Coin A) (hS : L.Sessions H) (x : Msg) (p : ℕ)
    (d σ : Fin (L.V.k x) → Coin) :
    L.slotRounds x p (forkOn (L.slotRounds x p d) d σ) = L.slotRounds x p d := by
  unfold slotRounds
  cases hu : (L.checked x d)[p]? with
  | none => simp only [forkOn_empty, hu]
  | some u => simp only [(hS.fixed x d u (List.mem_of_getElem? hu)).2 σ, hu]

omit [Fintype Coin] [Nonempty Coin] in
theorem coinsOf_forkOn (L : Compiled Coin A) {x : Msg} {u : ℕ} {xS : Msg}
    (hk : A.V.k xS = (L.rounds x u).card) (d σ : Fin (L.V.k x) → Coin) :
    L.coinsOf x u xS (forkOn (L.rounds x u) d σ) = fun i => σ (L.emb x u xS hk i) := by
  funext i
  simp only [coinsOf, dite_eq_left hk, forkOn, emb, Finset.orderEmbOfFin_mem, ite_true]

/-- **A sampled slot rejects with at most its sessions' error**: the event that the proof block sampled at slot `p` has
a session that shows a rejection, at the reference prover on an honest witness, has probability at most any `e ≥ 0`
that bounds, capped at `1`, the proof system's error at the statements of the sessions there. -/
theorem Pr_slot_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Nonempty Ω] {PA : A.W → Prover Ω}
    {ScopeA : Msg → Prop} {δA : Msg → ℝ} {K : Msg → ℕ} (hA : Claims.Complete H A.V PA A.R ScopeA δA)
    (hS : L.Sessions H) (hC : L.CompleteSessions H PA ScopeA δA K) (h64 : Out64 H) (x : Msg) (w : Wit)
    (hx : L.Scope x) (hw : Rhon H L.commit L.parse L.partition x w) (ω : L.Tape) (p : ℕ) (e : ℝ) (he : 0 ≤ e)
    (hle : ∀ (r : Fin (L.V.k x) → Coin) u xS vS, u ∈ L.checked x r →
      L.session x u (run H (L.P w) L.V x ω r).seen = some (xS, vS) → min 1 (δA xS) ≤ e) :
    Pr (fun r : Fin (L.V.k x) → Coin => ∃ u, (L.checked x r)[p]? = some u ∧
      ∃ xS vS, L.session x u (run H (L.P w) L.V x ω r).seen = some (xS, vS) ∧ vS.out ≠ .acc) ≤ e := by
  classical
  set Bad := fun r : Fin (L.V.k x) → Coin => ∃ u, (L.checked x r)[p]? = some u ∧
      ∃ xS vS, L.session x u (run H (L.P w) L.V x ω r).seen = some (xS, vS) ∧ vS.out ≠ .acc with hBad
  have hinner : ∀ d : Fin (L.V.k x) → Coin,
      Ex (fun σ => if Bad (forkOn (L.slotRounds x p d) d σ) then (1 : ℝ) else 0) ≤ e := by
    intro d
    rw [Ex_ind]
    unfold slotRounds
    cases hu : (L.checked x d)[p]? with
    | none =>
      simp only [forkOn_empty]
      refine le_trans (le_of_eq ?_) he
      refine (Pr_congr fun _ => ⟨fun ⟨u, h, _⟩ => ?_, False.elim⟩).trans Pr_false
      rw [hu] at h
      simp at h
    | some u =>
      simp only
      have hmem := List.mem_of_getElem? hu
      obtain ⟨xS, wS, ωS, hsc, hR, hk, hsess⟩ := hC.honest x w hx hw ω d u hmem
      have hagree : ∀ σ, ∀ t ∉ L.rounds x u, forkOn (L.rounds x u) d σ t = d t := fun σ t ht => by
        simp [forkOn, ht]
      have hbad : ∀ σ, Bad (forkOn (L.rounds x u) d σ) →
          (run H (PA wS) A.V xS ωS fun i => σ (L.emb x u xS hk i)).out ≠ .acc := by
        rintro σ ⟨u', hu', xS', vS, hs, hrej⟩
        rw [(hS.fixed x d u hmem).2 σ, hu] at hu'
        cases hu'
        rw [hsess _ (hagree σ), coinsOf_forkOn L hk d σ] at hs
        cases hs
        exact hrej
      have hd := hsess d fun _ _ => rfl
      calc Pr (fun σ => Bad (forkOn (L.rounds x u) d σ))
          ≤ Pr (fun σ : Fin (L.V.k x) → Coin =>
              (run H (PA wS) A.V xS ωS fun i => σ (L.emb x u xS hk i)).out ≠ .acc) := Pr_mono hbad
        _ = Pr (fun c : Fin (A.V.k xS) → Coin => (run H (PA wS) A.V xS ωS c).out ≠ .acc) :=
          Pr_comp_inj (L.emb x u xS hk) (L.emb x u xS hk).injective
            fun c => (run H (PA wS) A.V xS ωS c).out ≠ .acc
        _ = 1 - Pr (fun c : Fin (A.V.k xS) → Coin => (run H (PA wS) A.V xS ωS c).out = .acc) :=
          Claims.Sampling.Pr_not _
        _ ≤ min 1 (δA xS) :=
          le_min (by linarith [Pr_nonneg (fun c : Fin (A.V.k xS) → Coin =>
              (run H (PA wS) A.V xS ωS c).out = .acc)])
            (by linarith [hA h64 xS wS hsc hR ωS])
        _ ≤ e := hle d u xS _ hmem hd
  calc Pr Bad = Ex (fun r => if Bad r then (1 : ℝ) else 0) := (Ex_ind Bad).symm
    _ = Ex (fun d => Ex fun σ => if Bad (forkOn (L.slotRounds x p d) d σ) then (1 : ℝ) else 0) :=
      (Ex_resample (L.slotRounds x p) (L.slotRounds_fork hS x p) _).symm
    _ ≤ Ex (fun _ : Fin (L.V.k x) → Coin => e) := Ex_mono hinner
    _ = e := Ex_const _

/-- **A compiled verifier is complete** at its error, from the proof system's completeness, the contract and the
completeness contract: the reference prover is rejected only where a slot below `K x` shows a rejecting session, each
with probability at most `δ_c / K x` (`Pr_slot_le`), so with probability at most `δ_c` by the union bound. -/
theorem complete_of (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Nonempty Ω] {PA : A.W → Prover Ω}
    {ScopeA : Msg → Prop} {δA : Msg → ℝ} {K : Msg → ℕ} (hA : Claims.Complete H A.V PA A.R ScopeA δA)
    (hS : L.Sessions H) (hC : L.CompleteSessions H PA ScopeA δA K) : L.Complete H := by
  intro h64 x w hx hw ω
  classical
  obtain ⟨hnn, herr⟩ := hC.err x
  have hrej : ∀ r : Fin (L.V.k x) → Coin, ¬ (run H (L.P w) L.V x ω r).out = .acc →
      ∃ p ∈ (univ : Finset (Fin (K x))), ∃ u, (L.checked x r)[p.val]? = some u ∧
        ∃ xS vS, L.session x u (run H (L.P w) L.V x ω r).seen = some (xS, vS) ∧ vS.out ≠ .acc := by
    intro r hr
    by_contra hno
    refine hr (hC.accepts x w hx hw ω r fun u hu xS vS hs => ?_)
    by_contra hout
    obtain ⟨p, hp, u', hu', heq⟩ := hC.among x w hx hw ω r u hu
    exact hno ⟨⟨p, hp⟩, mem_univ _, u', hu', xS, vS, heq ▸ hs, hout⟩
  have hsum : ∑ p : Fin (K x), Pr (fun r : Fin (L.V.k x) → Coin => ∃ u, (L.checked x r)[p.val]? = some u ∧
      ∃ xS vS, L.session x u (run H (L.P w) L.V x ω r).seen = some (xS, vS) ∧ vS.out ≠ .acc) ≤ L.δc H x := by
    calc _ ≤ ∑ _p : Fin (K x), L.δc H x / K x :=
          sum_le_sum fun p _ => by
            have hK : (0 : ℝ) < K x := Nat.cast_pos.2 (Nat.zero_lt_of_lt p.isLt)
            exact Pr_slot_le H L hA hS hC h64 x w hx hw ω p.val _ (div_nonneg hnn hK.le)
              fun r u xS vS hu hs => (le_div_iff₀ hK).2 (by rw [mul_comm]; exact herr w ω r u xS vS hw hu hs)
      _ ≤ L.δc H x := by
          rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
          rcases Nat.eq_zero_or_pos (K x) with hK | hK
          · simp [hK, hnn]
          · have hK' : (K x : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hK.ne'
            exact le_of_eq (by field_simp)
  have hbad := (Pr_mono hrej).trans ((Claims.Sampling.Pr_exists_le univ _).trans hsum)
  rw [Claims.Sampling.Pr_not] at hbad
  linarith

end Claims.SamplingGeneric.Compiled

namespace Claims.SamplingGeneric

/-- **The sampled compiler preserves completeness** (`Compiled.complete_of`). -/
theorem compilerComplete : CompilerComplete := by
  intro Coin _ _ _ A L H Ω _ PA ScopeA δA K hA hS hC
  exact L.complete_of H hA hS hC

end Claims.SamplingGeneric
