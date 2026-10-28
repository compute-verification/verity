import Proofs.Flock.Claims.Sampling.KS

/-!
# The sampling extractor's run bounds

`Efficient`'s first and third bounds on the claims block's extractor `E` (`Definitions.Claims.Sampling`), at every
decoder, from the rerun programs' laws (`Sampling.Cost`):

* `E_runs_le`: **`ExtractRunsLe`** at `1 + K + k·(K + k·n_PU·m′·K′)`. The replay is one rerun; the outer wait holds `K`
  draws; at each of its at most `k` accepted runs, the inner wait holds `K`; at each of its at most `k`, every one of at
  most `n_PU` checked units makes Flock's `m′` waits of `K′` draws (`unitExt_runs_le`); checking what a witness reveals
  reruns nothing (`checkRevealed_runs`).
* `E_own_le`: **`ExtractOwnLe`** at `k²·n_PU·openCalls`, given `OpenCallsLe`: the waits make no SHA-512 call of their own
  (`NoHash`), so the extractor's own calls are the checks of what each of at most `k·k·n_PU` unit extractions reveals.
* `OpenCallsLe`: checking the openings one checked unit's Flock witness reveals, each under its roots, makes at most
  `openCalls` SHA-512 calls, at a statement that parses. It is the deployed opening check's count, the sampled-proofs
  verifier's.
-/

namespace Claims.Sampling

open Claims

/-! ## Reruns and own calls of a list's programs -/

theorem flatMapRP_runs_le {R β γ : Type} (H : Msg → Msg) (run : R → Trace) (f : β → RP R (List γ)) (B : ℕ)
    (l : List β) (h : ∀ b ∈ l, ((f b).eval H run).2.2 ≤ B) : ((flatMapRP f l).eval H run).2.2 ≤ l.length * B := by
  rw [flatMapRP_runs]
  refine (List.sum_le_length_nsmul _ B fun n hn => ?_).trans_eq (by simp)
  obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hn
  exact h b hb

theorem ownLog_flatMapRP_le {R β γ : Type} (H : Msg → Msg) (run : R → Trace) (f : β → RP R (List γ)) (B : ℕ)
    (l : List β) (h : ∀ b ∈ l, (ownLog H run (f b)).length ≤ B) :
    (ownLog H run (flatMapRP f l)).length ≤ l.length * B := by
  rw [ownLog_flatMapRP]
  refine (List.sum_le_length_nsmul _ B fun n hn => ?_).trans_eq (by simp)
  obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hn
  exact h b hb

theorem length_zip_finRange_le {α : Type} (k : ℕ) (l : List α) : ((List.finRange k).zip l).length ≤ k := by
  rw [List.length_zip, List.length_finRange]
  exact min_le_left _ _

/-- `checkRevealed`'s own calls: the commitment's checks of each opening, under its roots. -/
theorem ownLog_checkRevealed_length {R : Type} (H : Msg → Msg) (run : R → Trace) (Cm : Commitment Bool)
    (rs : List (List Msg × Opening Bool)) :
    (ownLog H run (checkRevealed (R := R) Cm rs)).length = (rs.map fun p => ((Cm.opens p.1 p.2).eval H).2.length).sum := by
  unfold checkRevealed
  rw [ownLog_flatMapRP]
  congr 1
  refine List.map_congr_left fun p _ => ?_
  rw [ownLog_bind, ownLog_ofOC, List.length_append]
  simp [ownLog]

section Runs

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **`openCalls` bounds checking one session's reveal**: at a statement that parses, checking the openings a checked
unit's Flock witness reveals, each under its roots, makes at most `openCalls` SHA-512 calls. -/
def OpenCallsLe (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x s, D.parse x = some s → ∀ u v wF,
    ((D.reveal s u v wF).map fun p => ((D.commit.opens p.1 p.2).eval H).2.length).sum ≤ D.openCalls x

omit [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- The extractor's program, with its outer body named. -/
theorem E_ext_eq (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg) (r₀ : Fin (D.V.k x) → Coin)
    (c : (E D decode).C x) :
    (E D decode).ext x r₀ c = .rerun r₀ fun v₀ =>
      match D.parse x with
      | none => .ret none
      | some s =>
        if v₀.out = .acc then
          (waitAccC (D.k x) ((List.finRange (D.cap x)).map c.1) []).bind fun outer =>
            (flatMapRP (outerBody D decode x s c) ((List.finRange (D.k x)).zip outer)).bind fun found =>
              .ret (some (assemble s found))
        else .ret none :=
  rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- A checked unit's extraction reruns at most `m′·K′` times: Flock's waits on its session. -/
theorem unitExt_runs_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : AuditStatement) (u : ℕ) (r : Fin (D.V.k x) → Coin) (v : Seen) (ρ : D.UnitCoins x)
    (run' : (Fin (D.V.k x) → Coin) → Trace) :
    ((unitExt D decode x s u r v ρ).eval H run').2.2 ≤ D.innerWaits x * D.innerCap x := by
  unfold unitExt
  split
  · simp [RP.eval]
  · rename_i xS _ _
    rw [(RP.eval_bind H run' _ _).2]
    have hw := waits_runs_le H run' (D.innerWaits x) (D.innerK x) (D.innerCap x) (D.sessionCols x u)
      (Flock.assemble D.flock.parse D.flock.table decode D.flock.read xS) (Flock.check D.flock.parse xS)
      fun i j => forkOn (D.rounds x u) r (ρ i j)
    unfold waitsProg at hw
    refine le_of_le_of_eq (Nat.add_le_add_right hw _) ?_
    split
    · simp [RP.eval]
    · rw [checkRevealed_runs, Nat.add_zero]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem innerBody_runs_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : AuditStatement) (c : ECoins D x) (j : Fin (D.k x)) (li : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen))
    (run' : (Fin (D.V.k x) → Coin) → Trace) :
    ((innerBody D decode x s c j li).eval H run').2.2 ≤ nPU D x * (D.innerWaits x * D.innerCap x) :=
  (flatMapRP_runs_le H run' _ _ _ fun iu _ => unitExt_runs_le H D decode x s iu.2 li.2.1 li.2.2 _ run').trans
    (Nat.mul_le_mul_right _ (length_zip_finRange_le _ _))

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem outerBody_runs_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : AuditStatement) (c : ECoins D x) (jo : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen))
    (run' : (Fin (D.V.k x) → Coin) → Trace) :
    ((outerBody D decode x s c jo).eval H run').2.2 ≤
      D.cap x + D.k x * (nPU D x * (D.innerWaits x * D.innerCap x)) := by
  unfold outerBody
  rw [(RP.eval_bind H run' _ _).2, waitAccC_runs_eq]
  refine Nat.add_le_add ((waitRuns_le_length _ _ _ _).trans_eq (by simp)) ?_
  exact (flatMapRP_runs_le H run' _ _ _ fun li _ => innerBody_runs_le H D decode x s c jo.1 li run').trans
    (Nat.mul_le_mul_right _ (length_zip_finRange_le _ _))

/-- **The outer wait and what follows it** rerun at most `K + k·B` times: the wait over `K` draws, and at most `k`
bodies of at most `B` reruns each. Stated at every `k`, so that the kernel never meets the deployed `k`'s
`List.finRange`. -/
theorem outer_runs_le {R γ β : Type} (H : Msg → Msg) (run : R → Trace) (k K B : ℕ) (rs : List R)
    (hrs : rs.length = K) (body : Fin k × (R × Seen) → RP R (List γ)) (fin : List γ → β)
    (hB : ∀ jo, ((body jo).eval H run).2.2 ≤ B) :
    (((waitAccC k rs []).bind fun outer =>
        (flatMapRP body ((List.finRange k).zip outer)).bind fun found => RP.ret (fin found)).eval H run).2.2 ≤
      K + k * B := by
  rw [(RP.eval_bind _ _ _ _).2, (RP.eval_bind _ _ _ _).2, waitAccC_runs_eq]
  have h1 := waitRuns_le_length run k rs 0
  have h2 := (flatMapRP_runs_le H run body B _ fun jo _ => hB jo).trans
    (Nat.mul_le_mul_right B (length_zip_finRange_le k ((waitAccC k rs []).eval H run).1))
  simp only [RP.eval, List.length_nil, Nat.add_zero]
  rw [hrs] at h1
  exact Nat.add_le_add h1 h2

omit [DecidableEq Col] [Zero Col] in
/-- **`ExtractRunsLe` at the sampling extractor**, at every decoder: at most `1 + K + k·(K + k·n_PU·m′·K′)` reruns. -/
theorem E_runs_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    ExtractRunsLe H (E D decode)
      (fun x => 1 + D.cap x + D.k x * (D.cap x + D.k x * (nPU D x * (D.innerWaits x * D.innerCap x)))) := by
  intro x Ω P ω r₀ c
  simp only [Extractor.eval]
  rw [E_ext_eq]
  simp only [RP.eval]
  split
  · simp only [RP.eval, Nat.zero_add]
    rw [Nat.add_assoc]
    exact Nat.le_add_right 1 _
  · rename_i s _
    split
    · have h := outer_runs_le H (run H P D.V x ω) (D.k x) (D.cap x)
        (D.cap x + D.k x * (nPU D x * (D.innerWaits x * D.innerCap x))) ((List.finRange (D.cap x)).map c.1)
        (by rw [List.length_map, List.length_finRange])
        (outerBody D decode x s c) (fun found => some (assemble s found))
        (fun jo => outerBody_runs_le H D decode x s c jo (run H P D.V x ω))
      exact (Nat.add_le_add_right h 1).trans_eq (by ring)
    · simp only [RP.eval, Nat.zero_add]
      rw [Nat.add_assoc]
      exact Nat.le_add_right 1 _

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- A checked unit's extraction's own calls are its reveal's checks, at most `openCalls`. -/
theorem unitExt_own_le {H : Msg → Msg} {D : Deploy Coin Col T} (hopen : OpenCallsLe H D)
    (decode : List (ℕ × Col) → List T) {x : Msg} {s : AuditStatement} (hs : D.parse x = some s) (u : ℕ)
    (r : Fin (D.V.k x) → Coin) (v : Seen) (ρ : D.UnitCoins x) (run' : (Fin (D.V.k x) → Coin) → Trace) :
    (ownLog H run' (unitExt D decode x s u r v ρ)).length ≤ D.openCalls x := by
  unfold unitExt
  split
  · simp [ownLog]
  · rw [ownLog_bind, ownLog_of_noHash H run' (NoHash.firstWait _ _ _ _ _), List.nil_append]
    split
    · simp [ownLog]
    · rw [ownLog_checkRevealed_length]
      exact hopen x s hs u v _

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem outerBody_own_le {H : Msg → Msg} {D : Deploy Coin Col T} (hopen : OpenCallsLe H D)
    (decode : List (ℕ × Col) → List T) {x : Msg} {s : AuditStatement} (hs : D.parse x = some s) (c : ECoins D x)
    (jo : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen)) (run' : (Fin (D.V.k x) → Coin) → Trace) :
    (ownLog H run' (outerBody D decode x s c jo)).length ≤ D.k x * (nPU D x * D.openCalls x) := by
  unfold outerBody
  rw [ownLog_bind, ownLog_of_noHash H run' (NoHash.waitAccC _ _ _), List.nil_append]
  refine (ownLog_flatMapRP_le H run' _ _ _ fun li _ => ?_).trans
    (Nat.mul_le_mul_right _ (length_zip_finRange_le _ _))
  exact (ownLog_flatMapRP_le H run' _ _ _ fun iu _ =>
    unitExt_own_le hopen decode hs iu.2 li.2.1 li.2.2 _ run').trans
    (Nat.mul_le_mul_right _ (length_zip_finRange_le _ _))

omit [DecidableEq Col] [Zero Col] in
/-- **`ExtractOwnLe` at the sampling extractor**, at every decoder: at most `k²·n_PU·openCalls` own calls, when
`openCalls` bounds checking one session's reveal (`OpenCallsLe`). -/
theorem E_own_le {H : Msg → Msg} {D : Deploy Coin Col T} (hopen : OpenCallsLe H D)
    (decode : List (ℕ × Col) → List T) :
    ExtractOwnLe H (E D decode) (fun x => D.k x ^ 2 * nPU D x * D.openCalls x) := by
  intro x Ω P ω r₀ c
  rw [E_ext_eq]
  simp only [ownLog]
  split
  · simp [ownLog]
  · rename_i s hs
    split
    · rw [ownLog_bind, ownLog_of_noHash _ _ (NoHash.waitAccC _ _ _), List.nil_append, ownLog_bind]
      simp only [ownLog, List.append_nil]
      refine ((ownLog_flatMapRP_le H _ _ _ _ fun jo _ =>
        outerBody_own_le hopen decode hs c jo (run H P D.V x ω)).trans
        (Nat.mul_le_mul_right _ (length_zip_finRange_le _ _))).trans_eq ?_
      ring
    · simp [ownLog]

end Runs

end Claims.Sampling
