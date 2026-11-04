import Proofs.Flock.Claims.Recursion.KSLin2
import Proofs.Flock.Claims.Sampling.Bext

/-!
# Route 2C's bits from its KS

`bitsLin2_of_ksLin2`: `KSLin2` gives `BitsLin2` at every compiled record where an accepted run shows each V* session
accepting (`SessionAccepts`, a fact of the record's reader; `RecDeploy.sessionAccepts` at the deployed records, which
`recBound` builds with `deployRec`); `bitsLin2On_of_ksLin2On` is the same at the statements a predicate `G` admits. The
proof reads `KSLin2` at `M = 2^256` and `R = 2^257` and bounds its six terms:

* one draw (the compiled run and `E_O`) costs at most `t·c_Be/2` SHA-512 calls in expectation (`Ex_drawCost_le`): the
  run at most `t`, and `E_O`, which makes no call of its own (`noHash_EO`), at most `t` per rerun
  (`Claims.Sampling.calls_le_of_noHash`) over at most `1 + Σ_s m′_s·k_s` reruns (`EO_runs_exp`);
* `B_I` costs two draws (`Ex_Binner_cost_le`), `FCI` and `Fdev` at most two in expectation (`Ex_FCI_cost_le`, Wald's
  bound on the gated loop over forked draws `Ex_gate_loop_fork_le`; `Ex_Fdev_cost_le`), and `F0I` at most `k_I + 1`
  (`Ex_F0I_cost_le`), each searching the inner verifier's own calls too (`Deploy.callsAt`: its setup's `|S(x_I)|`
  and at most `n` on each draw's inner messages at its coins), `B_I` and `Fdev` the compiled verifier's own at each draw's V\*
  statements (`BinnerV`, `FdevV`: at most `v`, check 5's row commitments among them; `Ex_BinnerV_cost_le`), and charged
  them (`Finder.charge`, `Ex_charge`: `|S(x_I)|` once and `n + v` per draw it reads), so under `ExpectedCR` each finds a
  collision with probability at most its cost over `2^256`, and `c_Be` becomes `c_S = c_Be + |S(x_I)| + 2(n + v)` in
  their terms;
* `B` costs at most `2t` (`B_cost_le`), so under `StrictBirthday` its term is `t·√γ·2^{-255.5}`
  (`Claims.Sampling.sqrt_birthday`);
* the tails `1/(M+1)` and `(k_I + 1)/(R+1)` are below the collision terms at `t·c_Be ≥ 1`;
* each term is at most `t·2^{-L_i}` at `λ₂`'s `i`-th entry, and six of them at most `t·2^{-(min L_i - log₂ 6)}`
  (`sum_six_le`).

This is a bound on a number under named premises, not a proved security level. `BitsLin2` reads `StrictBirthday` at
`B` and `ExpectedCR` at `B_I`, `Fdev`, `FCI` and `F0I`, the named SHA-512 collision assumptions. At the deployed records
`λ₂ ≥ 100` (`RecDeploy.lamLin2_deployed_ge_dep`), so at the deployment's dense statements `Pr[acc ∧ ¬R] ≤ t·2^{-100}`
holds once `KSLin2On` at `RecDeploy.DenseAt` does, which only follows from premises still open:
`PartChecksBV`, `PartOpensB` and `DecompB` (`RecDeploy.ksLin2_deployed_full`, at the coin server's record, which reads
back). Holo statements are candidate 3's own model.
-/

namespace Claims.Recursion

open Claims

/-! ## `E_O` makes no SHA-512 call of its own -/

section NoHash

theorem noHash_viewing {R : Type} (f : Seen → Seen) (r : R) : Claims.Sampling.NoHash (viewing f r) :=
  Claims.Sampling.NoHash.rerun r _ fun _ => Claims.Sampling.NoHash.ret _

theorem noHash_serve {R₁ R₂ α : Type} {q : R₁ → RP R₂ Seen} (hq : ∀ r, Claims.Sampling.NoHash (q r))
    {p : RP R₁ α} (hp : Claims.Sampling.NoHash p) : Claims.Sampling.NoHash (serve q p) := by
  induction hp with
  | ret a =>
    simp only [serve]
    exact Claims.Sampling.NoHash.ret a
  | rerun r k _ ih =>
    simp only [serve]
    exact Claims.Sampling.NoHash.bind (hq r) fun v => ih v

theorem noHash_collect {R α : Type} :
    ∀ ps : List (RP R (Option α)), (∀ p ∈ ps, Claims.Sampling.NoHash p) → Claims.Sampling.NoHash (collect ps)
  | [], _ => by
    simp only [collect]
    exact Claims.Sampling.NoHash.ret _
  | p :: ps, h => by
    simp only [collect]
    refine Claims.Sampling.NoHash.bind (h p (by simp)) fun o => ?_
    cases o with
    | none => exact Claims.Sampling.NoHash.ret _
    | some a =>
      exact Claims.Sampling.NoHash.bind (noHash_collect ps fun q hq => h q (by simp [hq]))
        fun _ => Claims.Sampling.NoHash.ret _

/-- **`E_O` makes no SHA-512 call of its own**: it reruns the judged run, then serves each session's waits by reruns
seen through the session's view. -/
theorem noHash_EO {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (D : Deploy Coin)
    (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (x : Msg)
    (r₀ : Fin (D.V.k x) → Coin) (c : (EO D I DO decO).C x) :
    Claims.Sampling.NoHash ((EO D I DO decO).ext x r₀ c) := by
  refine Claims.Sampling.NoHash.rerun r₀ _ fun v₀ => ?_
  split
  · exact noHash_collect _ fun p hp => by
      obtain ⟨s, -, rfl⟩ := List.mem_map.1 hp
      exact noHash_serve (fun r => noHash_viewing _ r) (Claims.Sampling.NoHash.firstWait _ _ _ _ _)
  · exact Claims.Sampling.NoHash.ret _

/-- `E_O` calls SHA-512 at most `t` times per rerun, when every run makes at most `t` calls. -/
theorem EO_calls_le {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (x : Msg)
    {Ω : Type} (P : Prover Ω) (t : ℕ)
    (ht : ∀ ω r, (run H P D.V x ω r).pLog.length + (run H P D.V x ω r).vLog.length ≤ t) (ω : Ω)
    (r : Fin (D.V.k x) → Coin) (c : (EO D I DO decO).C x) :
    ((EO D I DO decO).eval H P x ω r c).2.1 ≤ t * ((EO D I DO decO).eval H P x ω r c).2.2 :=
  Claims.Sampling.calls_le_of_noHash H (run H P D.V x ω) t (ht ω) (noHash_EO D I DO decO x r c)

end NoHash

/-! ## The finders' expected costs -/

open Classical in
/-- **Wald's bound on a gated loop over forked draws, in expectation**: the loop's draws, forked at `j` from the first,
cost at most one draw in expectation. -/
theorem Ex_gate_loop_fork_le {Coin A : Type} [Fintype Coin] [Nonempty Coin] [Fintype A] [Nonempty A] {n : ℕ}
    (j : ℕ) (g : (Fin n → Coin) × A → Prop) (c : (Fin n → Coin) × A → ℕ) (M : ℕ) :
    Ex (fun p : ((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A) =>
        if g p.1 then
          (loopCost (fun v => g (fork j p.1.1 v.1, v.2)) (fun v => c (fork j p.1.1 v.1, v.2)) M p.2 : ℝ)
        else 0) ≤
      Ex (fun u => (c u : ℝ)) := by
  let e : (Fin n → Coin) × (((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A)) ≃
      ((Fin n → Coin) × ((Fin n → Coin) × A)) × (Fin M → (Fin n → Coin) × A) :=
    { toFun := fun a => ((a.1, a.2.1), a.2.2)
      invFun := fun b => (b.1.1, b.1.2, b.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have hΨ : UnifMap (fun a : (Fin n → Coin) × (((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A)) =>
      (((fork j a.1 a.2.1.1, a.2.1.2), a.2.2) : ((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A))) :=
    (UnifMap.equiv e).comp ((unifMap_forkPair j).prod UnifMap.self)
  have h1 : ∀ a : (Fin n → Coin) × (((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A)),
      (if g (fork j a.1 a.2.1.1, a.2.1.2) then
        (loopCost (fun v => g (fork j (fork j a.1 a.2.1.1) v.1, v.2))
          (fun v => c (fork j (fork j a.1 a.2.1.1) v.1, v.2)) M a.2.2 : ℝ) else 0) =
      (if g (fork j a.1 a.2.1.1, a.2.1.2) then
        (loopCost (fun v => g (fork j a.1 v.1, v.2)) (fun v => c (fork j a.1 v.1, v.2)) M a.2.2 : ℝ) else 0) :=
    fun a => by simp only [fork_fork]
  calc Ex (fun p : ((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A) =>
        if g p.1 then
          (loopCost (fun v => g (fork j p.1.1 v.1, v.2)) (fun v => c (fork j p.1.1 v.1, v.2)) M p.2 : ℝ)
        else 0)
      = Ex (fun a : (Fin n → Coin) × (((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A)) =>
        if g (fork j a.1 a.2.1.1, a.2.1.2) then
          (loopCost (fun v => g (fork j a.1 v.1, v.2)) (fun v => c (fork j a.1 v.1, v.2)) M a.2.2 : ℝ) else 0) :=
        (hΨ _).symm.trans (congrArg Ex (funext h1))
    _ = Ex fun ρ : Fin n → Coin => Ex fun b : ((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A) =>
        if g (fork j ρ b.1.1, b.1.2) then
          (loopCost (fun v => g (fork j ρ v.1, v.2)) (fun v => c (fork j ρ v.1, v.2)) M b.2 : ℝ) else 0 :=
        Ex_prod _
    _ ≤ Ex fun ρ : Fin n → Coin => Ex fun u : (Fin n → Coin) × A => (c (fork j ρ u.1, u.2) : ℝ) :=
        Ex_mono fun ρ => Ex_gate_loopCost_le (U := (Fin n → Coin) × A) (fun v => g (fork j ρ v.1, v.2))
          (fun v => c (fork j ρ v.1, v.2)) M
    _ = Ex (fun u => (c u : ℝ)) :=
        (Ex_prod (fun q : (Fin n → Coin) × ((Fin n → Coin) × A) => (c (fork j q.1 q.2.1, q.2.2) : ℝ))).symm.trans
          (unifMap_forkPair j (fun u => (c u : ℝ)))

section Costs

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **One draw costs at most `t·c_Be/2` in expectation**: the run at most `t`, and `E_O` at most `t` per rerun over at
most `1 + Σ_s m′_s·k_s` reruns in expectation. -/
theorem Ex_drawCost_le [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (hS : SessionAccepts D) (x : Msg) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ)
    (ht : ∀ ω r, (run H P D.V x ω r).pLog.length + (run H P D.V x ω r).vLog.length ≤ t) (ω : Ω) :
    Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => (drawCost H D I DO decO P x ω d : ℝ)) ≤
      t * cBe D x / 2 := by
  have hpt : ∀ d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x, (drawCost H D I DO decO P x ω d : ℝ) ≤
      t + t * (((EO D I DO decO).eval H P x ω d.1 d.2).2.2 : ℝ) := by
    intro d
    have h1 := ht ω d.1
    have h2 := EO_calls_le H D I DO decO x P t ht ω d.1 d.2
    have : drawCost H D I DO decO P x ω d ≤ t + t * ((EO D I DO decO).eval H P x ω d.1 d.2).2.2 := by
      unfold drawCost
      omega
    exact_mod_cast this
  have hr : Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (((EO D I DO decO).eval H P x ω d.1 d.2).2.2 : ℝ)) ≤
        1 + ∑ s : Fin (D.nS x), (((D.nums x s).waits * (D.nums x s).k : ℕ) : ℝ) :=
    EO_runs_exp H D I DO decO hS x Ω P ω
  have hc : (cBe D x : ℝ) = 4 + 2 * ∑ s : Fin (D.nS x), (((D.nums x s).waits * (D.nums x s).k : ℕ) : ℝ) := by
    simp only [cBe, Nat.cast_add, Nat.cast_mul, Nat.cast_sum, Nat.cast_ofNat]
  have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg _
  calc Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => (drawCost H D I DO decO P x ω d : ℝ))
      ≤ Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
          (t : ℝ) + t * (((EO D I DO decO).eval H P x ω d.1 d.2).2.2 : ℝ)) := Ex_mono hpt
    _ = t + t * Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
          (((EO D I DO decO).eval H P x ω d.1 d.2).2.2 : ℝ)) := by
        rw [Ex_add, Ex_const, Ex_mul_left]
    _ ≤ t + t * (1 + ∑ s : Fin (D.nS x), (((D.nums x s).waits * (D.nums x s).k : ℕ) : ℝ)) :=
        add_le_add le_rfl (mul_le_mul_of_nonneg_left hr ht0)
    _ = t * cBe D x / 2 := by rw [hc]; ring

/-- A draw at a uniform tape costs at most what it costs at every tape. -/
theorem Ex_draws_le (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) {e : ℝ}
    (hd : ∀ ω, Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (drawCost H D I DO decO P x ω d : ℝ)) ≤ e) :
    Ex (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => (drawCost H D I DO decO P x t.1 t.2 : ℝ)) ≤
      e :=
  (Ex_prod _).trans_le ((Ex_mono hd).trans_eq (Ex_const _))

/-- **`B_I` costs two draws**: the judged draw and one forked at the guessed inner round, each uniform. -/
theorem Ex_Binner_cost_le (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (hk : 0 < I.V.k (xI D x)) {e : ℝ}
    (hd : ∀ ω, Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (drawCost H D I DO decO P x ω d : ℝ)) ≤ e) :
    Ex (fun s : (Binner H D I DO decO P x hk).S => ((Binner H D I DO decO P x hk).cost s : ℝ)) ≤ 2 * e := by
  have : Nonempty (Fin (I.V.k (xI D x))) := ⟨⟨0, hk⟩⟩
  have hpt : ∀ (j : Fin (I.V.k (xI D x))) (ω : Ω),
      Ex (fun q : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x =>
        ((Binner H D I DO decO P x hk).cost (j, ω, q) : ℝ)) ≤ 2 * e := by
    intro j ω
    have hn : ∀ q : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x,
        (Binner H D I DO decO P x hk).cost (j, ω, q) = drawCost H D I DO decO P x ω (q.1, q.2.2.1) +
          drawCost H D I DO decO P x ω (fork (j : ℕ) q.1 q.2.1, q.2.2.2) := fun q => by
      simp only [Binner, drawCost]
      omega
    have h1 : Ex (fun q : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
        (EO D I DO decO).C x => (drawCost H D I DO decO P x ω (q.1, q.2.2.1) : ℝ)) =
        Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => (drawCost H D I DO decO P x ω d : ℝ)) :=
      ((UnifMap.self (A := Fin (D.V.k x) → Coin)).prod
        ((UnifMap.snd (A := Fin (D.V.k x) → Coin) (B := (EO D I DO decO).C x × (EO D I DO decO).C x)).comp
          (UnifMap.fst (A := (EO D I DO decO).C x) (B := (EO D I DO decO).C x))))
        (fun d => (drawCost H D I DO decO P x ω d : ℝ))
    have h2 : Ex (fun q : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
        (EO D I DO decO).C x => (drawCost H D I DO decO P x ω (fork (j : ℕ) q.1 q.2.1, q.2.2.2) : ℝ)) =
        Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => (drawCost H D I DO decO P x ω d : ℝ)) :=
      ((unifMap_forkPair (A := (EO D I DO decO).C x × (EO D I DO decO).C x) (n := D.V.k x) (j : ℕ)).comp
        ((UnifMap.self (A := Fin (D.V.k x) → Coin)).prod
          (UnifMap.snd (A := (EO D I DO decO).C x) (B := (EO D I DO decO).C x))))
        (fun d => (drawCost H D I DO decO P x ω d : ℝ))
    simp only [hn, Nat.cast_add]
    rw [Ex_add, h1, h2]
    linarith [hd ω]
  calc Ex (fun s : (Binner H D I DO decO P x hk).S => ((Binner H D I DO decO P x hk).cost s : ℝ))
      = Ex fun j : Fin (I.V.k (xI D x)) => Ex fun ω : Ω => Ex fun q : (Fin (D.V.k x) → Coin) ×
          (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x =>
            ((Binner H D I DO decO P x hk).cost (j, ω, q) : ℝ) :=
        (Ex_prod (fun s : Fin (I.V.k (xI D x)) × Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) ×
          (EO D I DO decO).C x × (EO D I DO decO).C x => ((Binner H D I DO decO P x hk).cost s : ℝ))).trans
          (congrArg Ex (funext fun _ => Ex_prod _))
    _ ≤ Ex fun _ : Fin (I.V.k (xI D x)) => Ex fun _ : Ω => 2 * e := Ex_mono fun j => Ex_mono fun ω => hpt j ω
    _ = 2 * e := by rw [Ex_const, Ex_const]

open Classical in
/-- **`FCI` costs at most two draws in expectation**: the first draw, and the loop over draws forked at the position's
inner round, which Wald's bound holds to one draw (`Ex_gate_loop_fork_le`). -/
theorem Ex_FCI_cost_le (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (Cp : Caps Coin) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (M : ℕ) (hW : 0 < Cp.width (xI D x)) {e : ℝ}
    (hd : ∀ ω, Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (drawCost H D I DO decO P x ω d : ℝ)) ≤ e) :
    Ex (fun s : (FCI H D I DO decO Cp P x M hW).S => ((FCI H D I DO decO Cp P x M hW).cost s : ℝ)) ≤ 2 * e := by
  have : Nonempty (Fin (Cp.width (xI D x))) := ⟨⟨0, hW⟩⟩
  have hpt : ∀ (q : Fin (Cp.width (xI D x))) (ω : Ω),
      Ex (fun p : ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
          (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
        ((FCI H D I DO decO Cp P x M hW).cost (q, ω, p) : ℝ)) ≤ 2 * e := by
    intro q ω
    have hc : ∀ p : ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
        (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x),
        ((FCI H D I DO decO Cp P x M hW).cost (q, ω, p) : ℝ) = (drawCost H D I DO decO P x ω p.1 : ℝ) +
          if opensAtI H D I DO decO Cp P x q ω p.1 then
            (loopCost
              (fun v => opensAtI H D I DO decO Cp P x q ω
                (fork (min (Cp.round (xI D x) q) (I.V.k (xI D x))) p.1.1 v.1, v.2))
              (fun v => drawCost H D I DO decO P x ω
                (fork (min (Cp.round (xI D x) q) (I.V.k (xI D x))) p.1.1 v.1, v.2)) M p.2 : ℝ)
          else 0 := by
      intro p
      by_cases h : opensAtI H D I DO decO Cp P x q ω p.1 <;> simp [FCI, capForkI, h]
    simp only [hc]
    rw [Ex_add]
    have h1 := (UnifMap.fst (A := (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
      (B := Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
      (fun d => (drawCost H D I DO decO P x ω d : ℝ)))
    have h2 := Ex_gate_loop_fork_le (min (Cp.round (xI D x) q) (I.V.k (xI D x)))
      (opensAtI H D I DO decO Cp P x q ω) (drawCost H D I DO decO P x ω) M
    have h3 := hd ω
    rw [h1]
    linarith
  calc Ex (fun s : (FCI H D I DO decO Cp P x M hW).S => ((FCI H D I DO decO Cp P x M hW).cost s : ℝ))
      = Ex fun q : Fin (Cp.width (xI D x)) => Ex fun ω : Ω => Ex fun p : ((Fin (D.V.k x) → Coin) ×
          (EO D I DO decO).C x) × (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
            ((FCI H D I DO decO Cp P x M hW).cost (q, ω, p) : ℝ) :=
        (Ex_prod (fun s : Fin (Cp.width (xI D x)) × Ω × (((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
          (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)) =>
            ((FCI H D I DO decO Cp P x M hW).cost s : ℝ))).trans
          (congrArg Ex (funext fun _ => Ex_prod _))
    _ ≤ Ex fun _ : Fin (Cp.width (xI D x)) => Ex fun _ : Ω => 2 * e := Ex_mono fun q => Ex_mono fun ω => hpt q ω
    _ = 2 * e := by rw [Ex_const, Ex_const]

open Classical in
/-- **`F0I` costs at most `k_I + 1` draws in expectation**: the first draw, and each of the `k_I` waits, which Wald's
bound holds to one draw (`Ex_gate_loopCost_le`). -/
theorem Ex_F0I_cost_le [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (R : ℕ) {e : ℝ}
    (hd : ∀ ω, Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (drawCost H D I DO decO P x ω d : ℝ)) ≤ e) :
    Ex (fun s : (F0I H D I DO decO P x R).S => ((F0I H D I DO decO P x R).cost s : ℝ)) ≤
      ((I.k (xI D x) : ℝ) + 1) * e := by
  have hpt : ∀ ω : Ω,
      Ex (fun p : ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
          (Fin (I.k (xI D x)) → Fin R → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
        ((F0I H D I DO decO P x R).cost (ω, p) : ℝ)) ≤ ((I.k (xI D x) : ℝ) + 1) * e := by
    intro ω
    have hc : ∀ p : ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
        (Fin (I.k (xI D x)) → Fin R → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x),
        ((F0I H D I DO decO P x R).cost (ω, p) : ℝ) = (drawCost H D I DO decO P x ω p.1 : ℝ) +
          ∑ i, if accAtI H D I DO decO P x ω p.1 then
            (loopCost (accAtI H D I DO decO P x ω) (drawCost H D I DO decO P x ω) R (p.2 i) : ℝ) else 0 := by
      intro p
      by_cases h : accAtI H D I DO decO P x ω p.1 <;> simp [F0I, h]
    have hi : ∀ i : Fin (I.k (xI D x)),
        Ex (fun p : ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
            (Fin (I.k (xI D x)) → Fin R → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
          if accAtI H D I DO decO P x ω p.1 then
            (loopCost (accAtI H D I DO decO P x ω) (drawCost H D I DO decO P x ω) R (p.2 i) : ℝ) else 0) ≤ e :=
      fun i => (Ex_pair_coord (C := fun _ : Fin (I.k (xI D x)) => Fin R → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
        i (fun d us => if accAtI H D I DO decO P x ω d then
          (loopCost (accAtI H D I DO decO P x ω) (drawCost H D I DO decO P x ω) R us : ℝ) else 0)).trans_le
        ((Ex_gate_loopCost_le (accAtI H D I DO decO P x ω) (drawCost H D I DO decO P x ω) R).trans (hd ω))
    have h1 := (UnifMap.fst (A := (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
      (B := Fin (I.k (xI D x)) → Fin R → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
      (fun d => (drawCost H D I DO decO P x ω d : ℝ)))
    simp only [hc]
    rw [Ex_add, Ex_sum, h1]
    have hs := Finset.sum_le_sum fun (i : Fin (I.k (xI D x))) (_ : i ∈ Finset.univ) => hi i
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hs
    linarith [hd ω]
  calc Ex (fun s : (F0I H D I DO decO P x R).S => ((F0I H D I DO decO P x R).cost s : ℝ))
      = Ex fun ω : Ω => Ex fun p : ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
          (Fin (I.k (xI D x)) → Fin R → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
            ((F0I H D I DO decO P x R).cost (ω, p) : ℝ) :=
        Ex_prod (fun s : Ω × ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
          (Fin (I.k (xI D x)) → Fin R → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
            ((F0I H D I DO decO P x R).cost s : ℝ))
    _ ≤ Ex fun _ : Ω => ((I.k (xI D x) : ℝ) + 1) * e := Ex_mono hpt
    _ = ((I.k (xI D x) : ℝ) + 1) * e := Ex_const _

end Costs

/-! ## The bits -/

section Bits

/-- `ExpectedCR` at a finder whose expected cost is at most `b`. -/
theorem ecr_le {H : Msg → Msg} {F : Finder} {b : ℝ} (h : ExpectedCR H F)
    (hb : Ex (fun s => (F.cost s : ℝ)) ≤ b) : F.Pr H ≤ b / 2 ^ (256 : ℝ) :=
  h.trans (div_le_div_of_nonneg_right hb (by positivity))

theorem le_two_rpow_of_div {t y a : ℝ} (ht : 0 ≤ t) (hy : 0 ≤ y) :
    t * y / 2 ^ a ≤ t * 2 ^ (-(a - Real.logb 2 y)) := by
  rcases hy.lt_or_eq with hy | rfl
  · refine le_of_eq ?_
    rw [show -(a - Real.logb 2 y) = Real.logb 2 y + -a by ring, Real.rpow_add (by norm_num),
      Real.rpow_logb (by norm_num) (by norm_num) hy, Real.rpow_neg (by norm_num), div_eq_mul_inv, mul_assoc]
  · rw [mul_zero, zero_div]
    exact mul_nonneg ht (Real.rpow_nonneg (by norm_num) _)

theorem sqrt_mul_two_rpow_le {γ : ℝ} (hγ : 0 ≤ γ) :
    Real.sqrt γ * 2 ^ (-(255.5 : ℝ)) ≤ 2 ^ (-(255.5 - Real.logb 2 γ / 2)) := by
  rcases hγ.lt_or_eq with hγ | rfl
  · refine le_of_eq ?_
    have : (2 : ℝ) ^ (-(255.5 - Real.logb 2 γ / 2)) = 2 ^ (Real.logb 2 γ * (1 / 2)) * 2 ^ (-(255.5 : ℝ)) := by
      rw [← Real.rpow_add (by norm_num)]; congr 1; ring
    rw [this, Real.rpow_mul (by norm_num), Real.rpow_logb (by norm_num) (by norm_num) hγ, Real.sqrt_eq_rpow]
  · simp only [Real.sqrt_zero, zero_mul]; positivity

theorem le_two_rpow_logb_inv {κ t : ℝ} (ht : 1 ≤ t) : κ ≤ t * 2 ^ (-Real.logb 2 (1 / κ)) := by
  rcases le_or_gt κ 0 with h | h
  · exact h.trans (mul_nonneg (by linarith) (Real.rpow_nonneg (by norm_num) _))
  · have hκ : (2 : ℝ) ^ (-Real.logb 2 (1 / κ)) = κ := by
      rw [one_div, Real.logb_inv, neg_neg, Real.rpow_logb (by norm_num) (by norm_num) h]
    rw [hκ]
    exact le_mul_of_one_le_left h.le ht

/-- Six terms, each at most `t·2^{-L_i}`, are at most `t·2^{-(min L_i - log₂ 6)}`. -/
theorem sum_six_le {t a₁ a₂ a₃ a₄ a₅ a₆ L₁ L₂ L₃ L₄ L₅ L₆ : ℝ} (ht : 0 ≤ t)
    (h₁ : a₁ ≤ t * 2 ^ (-L₁)) (h₂ : a₂ ≤ t * 2 ^ (-L₂)) (h₃ : a₃ ≤ t * 2 ^ (-L₃)) (h₄ : a₄ ≤ t * 2 ^ (-L₄))
    (h₅ : a₅ ≤ t * 2 ^ (-L₅)) (h₆ : a₆ ≤ t * 2 ^ (-L₆)) :
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ ≤
      t * 2 ^ (-(min (min (min (min (min L₁ L₂) L₃) L₄) L₅) L₆ - Real.logb 2 6)) := by
  have h6 : ∀ m : ℝ, (2 : ℝ) ^ (-(m - Real.logb 2 6)) = 6 * 2 ^ (-m) := fun m => by
    rw [show -(m - Real.logb 2 6) = Real.logb 2 6 + -m by ring, Real.rpow_add (by norm_num),
      Real.rpow_logb (by norm_num) (by norm_num) (by norm_num)]
  have hM : ∀ L, min (min (min (min (min L₁ L₂) L₃) L₄) L₅) L₆ ≤ L →
      t * (2 : ℝ) ^ (-L) ≤ t * 2 ^ (-min (min (min (min (min L₁ L₂) L₃) L₄) L₅) L₆) := fun L h =>
    mul_le_mul_of_nonneg_left (Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)) ht
  have e₁ := hM L₁ (by simp only [min_le_iff, le_refl, true_or, or_true])
  have e₂ := hM L₂ (by simp only [min_le_iff, le_refl, true_or, or_true])
  have e₃ := hM L₃ (by simp only [min_le_iff, le_refl, true_or, or_true])
  have e₄ := hM L₄ (by simp only [min_le_iff, le_refl, true_or, or_true])
  have e₅ := hM L₅ (by simp only [min_le_iff, le_refl, true_or, or_true])
  have e₆ := hM L₆ (by simp only [min_le_iff, le_refl, true_or, or_true])
  rw [h6, mul_left_comm]
  linarith

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **A finder charged `n` calls costs `n` more in expectation.** -/
theorem Ex_charge (F : Finder) (n : ℕ) :
    Ex (fun s => ((F.charge n).cost s : ℝ)) = Ex (fun s => (F.cost s : ℝ)) + n := by
  show Ex (fun s : F.S => ((F.cost s + n : ℕ) : ℝ)) = _
  simp only [Nat.cast_add]
  rw [Ex_add, Ex_const]

theorem ecr_charge_le {H : Msg → Msg} {F : Finder} {n : ℕ} {b : ℝ} (h : ExpectedCR H (F.charge n))
    (hb : Ex (fun s => (F.cost s : ℝ)) + n ≤ b) : F.Pr H ≤ b / 2 ^ (256 : ℝ) :=
  ecr_le (F := F.charge n) h (by rw [Ex_charge]; exact hb)

/-- **`B_I` over the draws' evaluations costs what `B_I` does**: only its output differs. -/
theorem Ex_BinnerV_cost_le [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (hk : 0 < I.V.k (xI D x)) {e : ℝ}
    (hd : ∀ ω, Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (drawCost H D I DO decO P x ω d : ℝ)) ≤ e) :
    Ex (fun s : (BinnerV H D I DO decO P x hk).S => ((BinnerV H D I DO decO P x hk).cost s : ℝ)) ≤ 2 * e :=
  Ex_Binner_cost_le H D I DO decO x P hk hd

/-- **`Fdev` over the draws' evaluations costs what `Fdev` does**: only its output differs. -/
theorem Ex_FdevV_cost_le [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (M : ℕ) (hk : 0 < I.V.k (xI D x)) :
    Ex (fun s : (FdevV H D I DO decO P x M hk).S => ((FdevV H D I DO decO P x M hk).cost s : ℝ)) ≤
      2 * Ex (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
        (drawCost H D I DO decO P x t.1 t.2 : ℝ)) :=
  Ex_Fdev_cost_le H D I DO decO x P M hk

/-- **`KSLin2`'s right side at `M = 2^256` and `R = 2^257` is at most `t·2^{-λ₂}`**, at a compiled record where an
accepted run shows each V* session accepting, under `StrictBirthday` at `B` and `ExpectedCR` at `B_I`, `Fdev`, `FCI` and
`F0I`, each searching the inner verifier's own calls at each draw's coins (`Deploy.callsAt`) and charged their
evaluations (`Finder.charge`): `|S(x_I)|` once (`setupLen`) and `n + v` per draw it reads, for a prover making at most
`t ≥ 1` calls per run. Each finder's expected cost is its draws' cost (`B_I` and `Fdev` over the draws' evaluations,
`BinnerV` and `FdevV`, differ from `Binner` and `Fdev` only in their output) plus its charge, at most
`t·(c_Be + |S(x_I)| + 2(n + v))` times its draws' count. -/
theorem lin2_bound_le [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (Cp : Caps Coin) (hS : SessionAccepts D)
    (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ) (ht1 : 1 ≤ t)
    (hrun : ∀ ω r, (run H P D.V x ω r).pLog.length + (run H P D.V x ω r).vLog.length ≤ t) (n₀ v : ℕ)
    (hSB : ∀ hk : 0 < D.V.k x, StrictBirthday H (B H D.V P x hk))
    (hBI : ∀ hk : 0 < I.V.k (xI D x),
      ExpectedCR H ((BinnerV H D I DO decO P x hk).charge (setupLen D x + 2 * (n₀ + v))))
    (hFdev : ∀ (M : ℕ) (hk : 0 < I.V.k (xI D x)),
      ExpectedCR H ((FdevV H D I DO decO P x M hk).charge (setupLen D x + 2 * (n₀ + v))))
    (hFCI : ∀ (M : ℕ) (hW : 0 < Cp.width (xI D x)),
      ExpectedCR H ((FCI H D I DO decO Cp P x M hW).charge (setupLen D x + 2 * (n₀ + v))))
    (hF0I : ∀ R : ℕ,
      ExpectedCR H ((F0I H D I DO decO P x R).charge
        (setupLen D x + (I.k (xI D x) + 1) * (n₀ + v)))) :
    κ D I x + Real.sqrt (γ D I x * PrB H D.V P x) +
        (4 * I.V.k (xI D x) + 1) * PrBinnerV H D I DO decO P x +
        4 * Cp.width (xI D x) *
          (PrFCI H D I DO decO Cp P x (2 ^ 256) + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1)) +
        4 * (PrF0I H D I DO decO P x (2 ^ 257) +
          ((I.k (xI D x) : ℝ) + 1) / (((2 ^ 257 : ℕ) : ℝ) + 1)) +
        βdev2 D I Cp x * (PrFdevV H D I DO decO P x (2 ^ 256) + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1)) ≤
      t * (2 : ℝ) ^ (-lamLin2 D I Cp x n₀ v) := by
  unfold lamLin2
  generalize n₀ + v = n at hBI hFdev hFCI hF0I ⊢
  have ht1' : (1 : ℝ) ≤ t := by exact_mod_cast ht1
  have ht0 : (0 : ℝ) ≤ t := by linarith
  have hc4 : (4 : ℝ) ≤ cBe D x := by
    have : 4 ≤ cBe D x := by unfold cBe; omega
    exact_mod_cast this
  have hnn : (0 : ℝ) ≤ n := Nat.cast_nonneg _
  have hn0 : (0 : ℝ) ≤ setupLen D x + 2 * n := by positivity
  have htcB : (1 : ℝ) ≤ t * cBe D x := by nlinarith
  have htc : (1 : ℝ) ≤ t * (cBe D x + setupLen D x + 2 * n) := by nlinarith
  have hc0 : (0 : ℝ) ≤ cBe D x + setupLen D x + 2 * n := by linarith
  have htn : (setupLen D x : ℝ) + 2 * n ≤ t * (setupLen D x + 2 * n) := le_mul_of_one_le_left hn0 ht1'
  have hd : ∀ ω, Ex (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (drawCost H D I DO decO P x ω d : ℝ)) ≤ t * cBe D x / 2 :=
    Ex_drawCost_le H D I DO decO hS x P t hrun
  have hp256 : (0 : ℝ) < 2 ^ (256 : ℝ) := by positivity
  have h256 : (2 : ℝ) ^ (256 : ℝ) = 2 * 2 ^ (255 : ℝ) := by
    rw [show (256 : ℝ) = 1 + 255 by norm_num, Real.rpow_add (by norm_num), Real.rpow_one]
  have h257 : (2 : ℝ) ^ (257 : ℝ) = 2 * 2 ^ (256 : ℝ) := by
    rw [show (257 : ℝ) = 1 + 256 by norm_num, Real.rpow_add (by norm_num), Real.rpow_one]
  have hMc : (((2 ^ 256 : ℕ) : ℝ)) = (2 : ℝ) ^ (256 : ℝ) := by
    rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  have hRc : (((2 ^ 257 : ℕ) : ℝ)) = (2 : ℝ) ^ (257 : ℝ) := by
    rw [show (257 : ℝ) = ((257 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  have hM : 1 / (((2 ^ 256 : ℕ) : ℝ) + 1) ≤ t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) := by
    rw [hMc]
    calc 1 / ((2 : ℝ) ^ (256 : ℝ) + 1) ≤ 1 / (2 : ℝ) ^ (256 : ℝ) :=
          one_div_le_one_div_of_le hp256 (by linarith)
      _ ≤ t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) := div_le_div_of_nonneg_right htc hp256.le
  have hB : PrB H D.V P x ≤ (2 * (t : ℝ)) ^ 2 / 2 ^ (513 : ℝ) := by
    unfold PrB
    split
    · rename_i hk
      refine hSB hk (2 * t) fun s => ?_
      exact_mod_cast B_cost_le H D.V P x hk t hrun s
    · positivity
  have hγ0 : 0 ≤ γ D I x := by unfold γ; positivity
  have hβ0 : 0 ≤ βdev2 D I Cp x := by unfold βdev2; positivity
  refine sum_six_le ht0 (le_two_rpow_logb_inv ht1') ?_ ?_ ?_ ?_ ?_
  · calc Real.sqrt (γ D I x * PrB H D.V P x)
        ≤ Real.sqrt (γ D I x * ((2 * (t : ℝ)) ^ 2 / 2 ^ (513 : ℝ))) :=
          Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hB hγ0)
      _ = t * (Real.sqrt (γ D I x) * 2 ^ (-(255.5 : ℝ))) := Claims.Sampling.sqrt_birthday hγ0 ht0
      _ ≤ _ := mul_le_mul_of_nonneg_left (sqrt_mul_two_rpow_le hγ0) ht0
  · have hP : PrBinnerV H D I DO decO P x ≤ t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) := by
      unfold PrBinnerV
      split_ifs with hk
      · refine ecr_charge_le (hBI hk) ?_
        have := Ex_BinnerV_cost_le H D I DO decO x P hk hd
        push_cast
        linarith
      · positivity
    calc (4 * (I.V.k (xI D x) : ℝ) + 1) * PrBinnerV H D I DO decO P x
        ≤ (4 * (I.V.k (xI D x) : ℝ) + 1) * (t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ)) :=
          mul_le_mul_of_nonneg_left hP (by positivity)
      _ = t * ((4 * (I.V.k (xI D x) : ℝ) + 1) * (cBe D x + setupLen D x + 2 * n)) / 2 ^ (256 : ℝ) := by ring
      _ ≤ _ := le_two_rpow_of_div ht0 (by positivity)
  · have hP : PrFCI H D I DO decO Cp P x (2 ^ 256) ≤
        t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) := by
      unfold PrFCI
      split_ifs with hW
      · refine ecr_charge_le (hFCI (2 ^ 256) hW) ?_
        have := Ex_FCI_cost_le H D I DO decO Cp x P (2 ^ 256) hW hd
        linarith [show ((setupLen D x + 2 * n : ℕ) : ℝ) = setupLen D x + 2 * n by
          simp only [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]]
      · positivity
    calc 4 * (Cp.width (xI D x) : ℝ) *
          (PrFCI H D I DO decO Cp P x (2 ^ 256) + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1))
        ≤ 4 * (Cp.width (xI D x) : ℝ) * (t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) +
            t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ)) :=
          mul_le_mul_of_nonneg_left (add_le_add hP hM) (by positivity)
      _ = t * (4 * (Cp.width (xI D x) : ℝ) * (cBe D x + setupLen D x + 2 * n)) / 2 ^ (255 : ℝ) := by
          rw [h256]; ring
      _ ≤ _ := le_two_rpow_of_div ht0 (by positivity)
  · have hP : PrF0I H D I DO decO P x (2 ^ 257) ≤
        (((I.k (xI D x) : ℝ) + 1) * (t * cBe D x / 2) + (setupLen D x + ((I.k (xI D x) : ℝ) + 1) * n)) /
          2 ^ (256 : ℝ) := by
      refine ecr_charge_le (hF0I (2 ^ 257)) ?_
      have : Ex (fun s => ((F0I H D I DO decO P x (2 ^ 257)).cost s : ℝ)) ≤
          ((I.k (xI D x) : ℝ) + 1) * (t * cBe D x / 2) := Ex_F0I_cost_le H D I DO decO x P (2 ^ 257) hd
      push_cast
      linarith
    have hk0 : (0 : ℝ) ≤ (I.k (xI D x) : ℝ) + 1 := by positivity
    have hk1 : (1 : ℝ) ≤ (I.k (xI D x) : ℝ) + 1 := by
      have : (0 : ℝ) ≤ I.k (xI D x) := Nat.cast_nonneg _
      linarith
    have hs1 : (setupLen D x : ℝ) ≤ ((I.k (xI D x) : ℝ) + 1) * (t * setupLen D x) :=
      (le_mul_of_one_le_left (Nat.cast_nonneg _) ht1').trans (le_mul_of_one_le_left (by positivity) hk1)
    have hn2 : ((I.k (xI D x) : ℝ) + 1) * n ≤ ((I.k (xI D x) : ℝ) + 1) * (t * (2 * n)) :=
      mul_le_mul_of_nonneg_left (by nlinarith [mul_nonneg (sub_nonneg.mpr ht1') hnn]) hk0
    have hR : ((I.k (xI D x) : ℝ) + 1) / (((2 ^ 257 : ℕ) : ℝ) + 1) ≤
        ((I.k (xI D x) : ℝ) + 1) * (t * cBe D x) / 2 ^ (257 : ℝ) := by
      rw [hRc]
      calc ((I.k (xI D x) : ℝ) + 1) / ((2 : ℝ) ^ (257 : ℝ) + 1) ≤ ((I.k (xI D x) : ℝ) + 1) / (2 : ℝ) ^ (257 : ℝ) :=
            div_le_div_of_nonneg_left hk0 (by positivity) (by linarith)
        _ ≤ _ := div_le_div_of_nonneg_right (le_mul_of_one_le_right hk0 htcB) (by positivity)
    calc 4 * (PrF0I H D I DO decO P x (2 ^ 257) +
          ((I.k (xI D x) : ℝ) + 1) / (((2 ^ 257 : ℕ) : ℝ) + 1))
        ≤ 4 * ((((I.k (xI D x) : ℝ) + 1) * (t * cBe D x / 2) + (setupLen D x + ((I.k (xI D x) : ℝ) + 1) * n)) /
              2 ^ (256 : ℝ) + ((I.k (xI D x) : ℝ) + 1) * (t * cBe D x) / 2 ^ (257 : ℝ)) :=
          mul_le_mul_of_nonneg_left (add_le_add hP hR) (by norm_num)
      _ = (t * (4 * ((I.k (xI D x) : ℝ) + 1) * cBe D x) + 4 * (setupLen D x + ((I.k (xI D x) : ℝ) + 1) * n)) /
            2 ^ (256 : ℝ) := by
          rw [h257]; ring
      _ ≤ t * (4 * ((I.k (xI D x) : ℝ) + 1) * (cBe D x + setupLen D x + 2 * n)) / 2 ^ (256 : ℝ) := by
          refine div_le_div_of_nonneg_right ?_ hp256.le
          nlinarith
      _ ≤ _ := le_two_rpow_of_div ht0 (by positivity)
  · have hP : PrFdevV H D I DO decO P x (2 ^ 256) ≤
        t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) := by
      unfold PrFdevV
      split_ifs with hk
      · have := Ex_draws_le H D I DO decO x P hd
        refine ecr_charge_le (hFdev (2 ^ 256) hk) ?_
        have := Ex_FdevV_cost_le H D I DO decO x P (2 ^ 256) hk
        linarith [show ((setupLen D x + 2 * n : ℕ) : ℝ) = setupLen D x + 2 * n by
          simp only [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]]
      · positivity
    calc βdev2 D I Cp x * (PrFdevV H D I DO decO P x (2 ^ 256) + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1))
        ≤ βdev2 D I Cp x * (t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) +
            t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ)) :=
          mul_le_mul_of_nonneg_left (add_le_add hP hM) hβ0
      _ = t * (βdev2 D I Cp x * (cBe D x + setupLen D x + 2 * n)) / 2 ^ (255 : ℝ) := by rw [h256]; ring
      _ ≤ _ := le_two_rpow_of_div ht0 (mul_nonneg hβ0 hc0)

end Bits

section Route

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- **Route 2C: `BitsLin2` from `KSLin2`**, at a compiled record where an accepted run shows each V* session accepting:
`KSLin2` at `M = 2^256` and `R = 2^257`, and `lin2_bound_le`. `BitsLin2` takes `StrictBirthday` at `B` and `ExpectedCR`
at `B_I`, `Fdev`, `FCI` and `F0I` as premises, the named SHA-512 collision assumptions. At the deployed records, where
`λ₂ ≥ 100` (`RecDeploy.lamLin2_deployed_ge_dep`) and `SessionAccepts` holds (`RecDeploy.sessionAccepts`), this makes
`Pr[acc ∧ ¬R] ≤ t·2^{-100}` follow from `KSLin2` there. At the deployment `KSLin2` is shown only at the dense
statements, from premises still open (`RecDeploy.ksLin2_deployed_open`), so the deployed form is
`bitsLin2On_of_ksLin2On` (`RecDeploy.bitsLin2_deployed_open`): a bound on a number under named premises, not a proved
security level. -/
theorem bitsLin2_of_ksLin2 [Zero Col] [Zero Col'] {H : Msg → Msg} {D : RecAt Coin} {Bd : Bound Coin Col T}
    {DO : Flock.Deploy Coin Col' T'} {Cp : Caps Coin} (hS : SessionAccepts (D.record Bd.IVStar))
    (h : KSLin2 H D Bd DO Cp) : BitsLin2 H D Bd DO Cp := by
  intro decI hdecI decO hdecO x Ω _ _ P t ht1 hrun n _ v _ hSB hBI hFdev hFCI hF0I
  exact (h decI hdecI decO hdecO x Ω P (2 ^ 256) (2 ^ 257)).trans
    (lin2_bound_le H _ _ DO decO Cp hS x P t ht1 hrun n v hSB hBI hFdev hFCI hF0I)

/-- **`BitsLin2` at the statements `G` admits**: its bound at every statement `x` with `G x`, as `KSLin2On` is
`KSLin2`'s. -/
def BitsLin2On [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) : Prop :=
  ∀ decI, decOK Bd.deploy decI → ∀ decO, decOK DO decO → ∀ (x : Msg), G x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (t : ℕ), 1 ≤ t →
    (∀ ω r, (run H P (D.record Bd.IVStar).V x ω r).pLog.length +
      (run H P (D.record Bd.IVStar).V x ω r).vLog.length ≤ t) →
    ∀ n : ℕ, (∀ cs ws, ((D.record Bd.IVStar).innerCalls H x cs ws).length ≤ n) →
    ∀ v : ℕ, (∀ ps, (setupCalls H (D.record Bd.IVStar) ((D.record Bd.IVStar).vstar x ps)).length ≤ v) →
    (∀ hk : 0 < (D.record Bd.IVStar).V.k x, StrictBirthday H (B H (D.record Bd.IVStar).V P x hk)) →
    (∀ hk : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x),
      ExpectedCR H ((BinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x
        hk).charge (setupLen (D.record Bd.IVStar) x + 2 * (n + v)))) →
    (∀ (M : ℕ) (hk : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x)),
      ExpectedCR H ((FdevV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x M
        hk).charge (setupLen (D.record Bd.IVStar) x + 2 * (n + v)))) →
    (∀ (M : ℕ) (hW : 0 < Cp.width (xI (D.record Bd.IVStar) x)),
      ExpectedCR H ((FCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M
        hW).charge (setupLen (D.record Bd.IVStar) x + 2 * (n + v)))) →
    (∀ R : ℕ, ExpectedCR H ((F0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x
      R).charge (setupLen (D.record Bd.IVStar) x +
        ((Inner.ofFlock Bd.deploy decI).k (xI (D.record Bd.IVStar) x) + 1) * (n + v)))) →
    PrBad H (D.record Bd.IVStar).V (Recursion.R (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
        (E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO) x P ≤
      t * (2 : ℝ) ^ (-lamLin2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x n v)

/-- **Route 2C: `BitsLin2On` from `KSLin2On`**, at the same statements: `bitsLin2_of_ksLin2`'s proof, which reads
`KSLin2` only at the statement it bounds. -/
theorem bitsLin2On_of_ksLin2On [Zero Col] [Zero Col'] {H : Msg → Msg} {D : RecAt Coin} {Bd : Bound Coin Col T}
    {DO : Flock.Deploy Coin Col' T'} {Cp : Caps Coin} {G : Msg → Prop} (hS : SessionAccepts (D.record Bd.IVStar))
    (h : KSLin2On H D Bd DO Cp G) : BitsLin2On H D Bd DO Cp G := by
  intro decI hdecI decO hdecO x hx Ω _ _ P t ht1 hrun n _ v _ hSB hBI hFdev hFCI hF0I
  exact (h decI hdecI decO hdecO x hx Ω P (2 ^ 256) (2 ^ 257)).trans
    (lin2_bound_le H _ _ DO decO Cp hS x P t ht1 hrun n v hSB hBI hFdev hFCI hF0I)

end Route

end Claims.Recursion
