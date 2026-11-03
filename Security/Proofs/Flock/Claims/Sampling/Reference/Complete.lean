import Proofs.Flock.Claims.Sampling.Select.DrawLaw
import Proofs.Flock.Claims.Sampling.Reference.Prover

/-!
# Sampling completeness from Flock's, session by session

`Claims.Sampling.Complete H D` at a record whose reference prover runs, in each checked unit's session, Flock's
reference prover on a statement Flock's verifier serves and a witness in its relation. Four facts about the record's
verifier and reference prover, each a hypothesis to discharge at the deployment:

* `CheckedLe`: on a statement in scope, the draw checks at most `n_PU` units; from `DrawSlots`
  (`checkedLe_of_slots`).
* `HonestSessions`: at an honest witness and coins that check `u`, `u`'s session is the run of Flock's reference prover
  (`D.flock.P`) on a statement in Flock's scope and a witness in its relation, on `u`'s own rounds (an injective map of
  Flock's) with the coins off them fixed.
* `AcceptsSessions`: at an honest witness, the run accepts when every checked unit's session accepts.
* `SessionsAmong K`: every checked unit's session is the session of the unit checked at a slot below `K x`; at
  `K = n_PU` from `CheckedLe` (`sessionsAmong_nPU`), and at `K = 1` when one session proves every drawn unit.

With `DrawFixed` (the checked units don't read their sessions' coins), each slot `p` rejects with probability at most
its session's Flock error: resampling the slot's session rounds (`Ex_resample`) leaves the slot's unit and statement
fixed, and the session's coins are then uniform (`Pr_comp_inj`), so Flock's completeness bounds the rejection. The union
bound over the slots below `K x` gives the error `K · sessErr`, `sessErr` the largest Flock error, capped at `1`, over
the statements of the sessions at `x`.

* `complete_of_sessions`: **`Complete H D`** from Flock's completeness at `D.flock`, `DrawFixed`, the four facts,
  Flock's errors nonnegative, and `D.δc H x ≥ K · sessErr`.
-/

namespace Claims.Sampling.Reference

open Claims Claims.Sampling Finset Properties.SampledProofs Properties.SampledProofs.SecurityProofs

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The draw checks at most the proof units**: on a statement in scope, at every coins, at most `n_PU` checked
units. -/
def CheckedLe (D : Deploy Coin Col T) : Prop := ∀ x, Scope D x → ∀ r, (D.checked x r).length ≤ nPU D x

omit [Nonempty Coin] [Inhabited Coin] in
/-- **The deployed draw checks at most the proof units**: `Slots.select`'s checked units are among the layout's, and a
refused draw checks each once (`size_checked_le`, `size_allSlots`). -/
theorem checkedLe_of_slots (D : Deploy Coin Col T) (h : DrawSlots D) : CheckedLe D := by
  rintro x ⟨s, Pt, hx, hP, -⟩ r
  obtain ⟨rule, d, query, table, cm, l, L, bytes, -, hl, hshape, -, -, hch⟩ := h x s Pt hx hP
  have hnPU : nPU D x = (slotPartition l).blocks.length := by
    simp [nPU, hx, Deploy.part, hP, Partition.blocks_length_of_shape hshape]
  rw [hch r, hnPU, Array.length_toList]
  unfold drawChecked
  split
  · rename_i c hs
    obtain ⟨hl', draw, proved, read, -, rfl⟩ := slots_select_ok _ _ _ _ _ _ _ _ hs
    rw [hl] at hl'
    obtain rfl := Except.ok.inj hl'
    exact size_checked_le l proved
  · exact (size_allSlots l).le

/-- **At the reference prover, a checked unit's session is Flock's reference prover's run**: on a statement in scope,
at a witness with no faulty unit and coins `r` that check `u`, a statement `xS` in Flock's scope, a witness `wF` in its
relation, a Flock tape `ωF` and an injective map `emb` of Flock's rounds into `u`'s, such that at coins agreeing with
`r` off `u`'s rounds, `u`'s session is `D.flock.P wF`'s run on `xS` at the coins `emb` reads. -/
def HonestSessions (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x w, Scope D x → Rhon H D.commit D.parse D.partition x w → ∀ (ω : D.Tape) (r : Fin (D.V.k x) → Coin) u,
    u ∈ D.checked x r →
    ∃ (xS : Msg) (wF : Claims.Flock.Wit) (ωF : D.flock.Tape) (emb : Fin (D.flock.V.k xS) → Fin (D.V.k x)),
      Function.Injective emb ∧ (∀ i, emb i ∈ D.rounds x u) ∧ Claims.Flock.Scope D.flock xS ∧
        Claims.Flock.R D.flock.parse xS wF ∧
        ∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, r' t = r t) →
          D.session x u (run H (D.P w) D.V x ω r').seen =
            some (xS, (run H (D.flock.P wF) D.flock.V xS ωF fun i => r' (emb i)).seen)

/-- **The run accepts when its sessions do**: on a statement in scope, at a witness with no faulty unit, the reference
prover's run accepts when every checked unit's session that shows accepts. -/
def AcceptsSessions (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x w, Scope D x → Rhon H D.commit D.parse D.partition x w → ∀ (ω : D.Tape) (r : Fin (D.V.k x) → Coin),
    (∀ u ∈ D.checked x r, ∀ xS vS, D.session x u (run H (D.P w) D.V x ω r).seen = some (xS, vS) → vS.out = .acc) →
      (run H (D.P w) D.V x ω r).out = .acc

/-- **A session's completeness error at `x`**: the largest Flock error (`D.flock.δc H`), capped at `1`, over the
statements of the reference prover's sessions at `x`, at an honest witness and coins that check the session's unit; `0`
where there are none. -/
noncomputable def sessErr (H : Msg → Msg) (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  sessErrAt H (Rhon H D.commit D.parse D.partition) D.P D.V D.checked D.session (D.flock.δc H) x

omit [Fintype Coin] [Nonempty Coin] in
theorem sessErr_nonneg (H : Msg → Msg) (D : Deploy Coin Col T) (hnn : ∀ xS, 0 ≤ D.flock.δc H xS) (x : Msg) :
    0 ≤ sessErr H D x :=
  Real.iSup_nonneg fun xS => le_min zero_le_one (hnn xS.1)

omit [Fintype Coin] [Nonempty Coin] in
theorem le_sessErr (H : Msg → Msg) (D : Deploy Coin Col T) {x xS : Msg} {w : Wit} {ω : D.Tape}
    {r : Fin (D.V.k x) → Coin} {u : ℕ} {vS : Seen} (hw : Rhon H D.commit D.parse D.partition x w)
    (hu : u ∈ D.checked x r) (h : D.session x u (run H (D.P w) D.V x ω r).seen = some (xS, vS)) :
    min 1 (D.flock.δc H xS) ≤ sessErr H D x := by
  unfold sessErr sessErrAt
  refine le_ciSup (f := fun xS : {xS : Msg // ∃ w ω r u vS, Rhon H D.commit D.parse D.partition x w ∧
    u ∈ D.checked x r ∧ D.session x u (run H (D.P w) D.V x ω r).seen = some (xS, vS)} =>
    min 1 (D.flock.δc H xS.1)) ⟨1, ?_⟩ ⟨xS, w, ω, r, u, vS, hw, hu, h⟩
  rintro _ ⟨y, rfl⟩
  exact min_le_left _ _

/-- The rounds of the unit checked at slot `p`, none when no unit is. -/
def slotRounds (D : Deploy Coin Col T) (x : Msg) (p : ℕ) (r : Fin (D.V.k x) → Coin) : Finset (Fin (D.V.k x)) :=
  match (D.checked x r)[p]? with
  | some u => D.rounds x u
  | none => ∅

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- A slot that checks a unit is below `n_PU`. -/
theorem slot_lt (D : Deploy Coin Col T) (hle : CheckedLe D) {x : Msg} (hx : Scope D x) {d : Fin (D.V.k x) → Coin}
    {p : ℕ} {u : ℕ} (hu : (D.checked x d)[p]? = some u) :
    ∃ q : Fin (nPU D x), (D.checked x d)[q.val]? = some u :=
  ⟨⟨p, (List.getElem?_eq_some_iff.1 hu).1.trans_le (hle x hx d)⟩, hu⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem slotRounds_fork (D : Deploy Coin Col T) (hfix : DrawFixed D) (hle : CheckedLe D) {x : Msg} (hx : Scope D x)
    (p : ℕ) (d σ : Fin (D.V.k x) → Coin) :
    slotRounds D x p (forkOn (slotRounds D x p d) d σ) = slotRounds D x p d := by
  unfold slotRounds
  cases hu : (D.checked x d)[p]? with
  | none => simp only [forkOn_empty, hu]
  | some u => simp only [(hfix x d u (slot_lt D hle hx hu)).2 σ, hu]

/-- **A checked slot rejects with at most its session's error**: the event that the unit checked at slot `p` has a
session that shows a rejection. -/
theorem Pr_slot_le (H : Msg → Msg) (D : Deploy Coin Col T) (hF : Claims.Flock.Complete H D.flock)
    (hfix : DrawFixed D) (hle : CheckedLe D) (hS : HonestSessions H D) (hnn : ∀ xS, 0 ≤ D.flock.δc H xS)
    (h64 : Out64 H) (x : Msg) (w : Wit) (hx : Scope D x) (hw : Rhon H D.commit D.parse D.partition x w) (ω : D.Tape)
    (p : ℕ) :
    Pr (fun r : Fin (D.V.k x) → Coin => ∃ u, (D.checked x r)[p]? = some u ∧
      ∃ xS vS, D.session x u (run H (D.P w) D.V x ω r).seen = some (xS, vS) ∧ vS.out ≠ .acc) ≤ sessErr H D x := by
  classical
  set Bad := fun r : Fin (D.V.k x) → Coin => ∃ u, (D.checked x r)[p]? = some u ∧
      ∃ xS vS, D.session x u (run H (D.P w) D.V x ω r).seen = some (xS, vS) ∧ vS.out ≠ .acc with hBad
  have hinner : ∀ d : Fin (D.V.k x) → Coin,
      Ex (fun σ => if Bad (forkOn (slotRounds D x p d) d σ) then (1 : ℝ) else 0) ≤ sessErr H D x := by
    intro d
    rw [Ex_ind]
    unfold slotRounds
    cases hu : (D.checked x d)[p]? with
    | none =>
      simp only [forkOn_empty]
      refine le_trans (le_of_eq ?_) (sessErr_nonneg H D hnn x)
      refine (Pr_congr fun _ => ⟨fun ⟨u, h, _⟩ => ?_, False.elim⟩).trans Pr_false
      rw [hu] at h
      simp at h
    | some u =>
      simp only
      obtain ⟨xS, wF, ωF, emb, hinj, hemb, hsc, hR, hsess⟩ :=
        hS x w hx hw ω d u (List.mem_of_getElem? hu)
      have hagree : ∀ σ, ∀ t ∉ D.rounds x u, forkOn (D.rounds x u) d σ t = d t := fun σ t ht => by
        simp [forkOn, ht]
      have hcoins : ∀ σ, (fun i => forkOn (D.rounds x u) d σ (emb i)) = fun i => σ (emb i) := fun σ =>
        funext fun i => by simp [forkOn, hemb i]
      have hbad : ∀ σ, Bad (forkOn (D.rounds x u) d σ) →
          (run H (D.flock.P wF) D.flock.V xS ωF fun i => σ (emb i)).out ≠ .acc := by
        rintro σ ⟨u', hu', xS', vS, hs, hrej⟩
        rw [(hfix x d u (slot_lt D hle hx hu)).2 σ, hu] at hu'
        cases hu'
        rw [hsess _ (hagree σ), hcoins σ] at hs
        cases hs
        exact hrej
      have hd := hsess d fun _ _ => rfl
      calc Pr (fun σ => Bad (forkOn (D.rounds x u) d σ))
          ≤ Pr (fun σ : Fin (D.V.k x) → Coin =>
              (run H (D.flock.P wF) D.flock.V xS ωF fun i => σ (emb i)).out ≠ .acc) := Pr_mono hbad
        _ = Pr (fun c : Fin (D.flock.V.k xS) → Coin => (run H (D.flock.P wF) D.flock.V xS ωF c).out ≠ .acc) :=
          Pr_comp_inj emb hinj fun c => (run H (D.flock.P wF) D.flock.V xS ωF c).out ≠ .acc
        _ = 1 - Pr (fun c : Fin (D.flock.V.k xS) → Coin => (run H (D.flock.P wF) D.flock.V xS ωF c).out = .acc) :=
          Claims.Sampling.Pr_not _
        _ ≤ min 1 (D.flock.δc H xS) :=
          le_min (by linarith [Pr_nonneg (fun c : Fin (D.flock.V.k xS) → Coin =>
              (run H (D.flock.P wF) D.flock.V xS ωF c).out = .acc)])
            (by linarith [hF h64 xS wF hsc hR ωF])
        _ ≤ sessErr H D x := le_sessErr H D hw (List.mem_of_getElem? hu) hd
  calc Pr Bad = Ex (fun r => if Bad r then (1 : ℝ) else 0) := (Ex_ind Bad).symm
    _ = Ex (fun d => Ex fun σ => if Bad (forkOn (slotRounds D x p d) d σ) then (1 : ℝ) else 0) :=
      (Ex_resample (slotRounds D x p) (slotRounds_fork D hfix hle hx p) _).symm
    _ ≤ Ex (fun _ : Fin (D.V.k x) → Coin => sessErr H D x) := Ex_mono hinner
    _ = sessErr H D x := Ex_const _

/-- **The checked units' sessions are the first `K x` slots'**: on a statement in scope, at an honest witness, every
checked unit's session at the reference prover is the session of the unit checked at a slot below `K x`. At `K = n_PU`
it holds of every draw that checks at most `n_PU` units (`sessionsAmong_nPU`); one session that proves every drawn unit
has it at `K = 1`. -/
def SessionsAmong (H : Msg → Msg) (D : Deploy Coin Col T) (K : Msg → ℕ) : Prop :=
  ∀ x w, Scope D x → Rhon H D.commit D.parse D.partition x w → ∀ (ω : D.Tape) (r : Fin (D.V.k x) → Coin) u,
    u ∈ D.checked x r → ∃ p < K x, ∃ u', (D.checked x r)[p]? = some u' ∧
      D.session x u (run H (D.P w) D.V x ω r).seen = D.session x u' (run H (D.P w) D.V x ω r).seen

omit [Fintype Coin] [Nonempty Coin] in
theorem sessionsAmong_nPU (H : Msg → Msg) (D : Deploy Coin Col T) (hle : CheckedLe D) :
    SessionsAmong H D (nPU D) := by
  intro x w hx _ ω r u hu
  obtain ⟨i, hi, hget⟩ := List.getElem_of_mem hu
  exact ⟨i, hi.trans_le (hle x hx r), u, List.getElem?_eq_some_iff.2 ⟨hi, hget⟩, rfl⟩

/-- **Sampling completeness from Flock's, session by session**: at Flock's completeness at the record's sessions,
`DrawFixed`, `CheckedLe`, `HonestSessions`, `AcceptsSessions` and `SessionsAmong` at `K`, with Flock's errors
nonnegative, the reference prover is accepted with probability at least `1 − K · sessErr`, so `Complete` holds at any
`δc` at least that. -/
theorem complete_of_sessions (H : Msg → Msg) (D : Deploy Coin Col T) (hF : Claims.Flock.Complete H D.flock)
    (hfix : DrawFixed D) (hle : CheckedLe D) (hS : HonestSessions H D) (hA : AcceptsSessions H D) (K : Msg → ℕ)
    (hK : SessionsAmong H D K) (hnn : ∀ xS, 0 ≤ D.flock.δc H xS) (hδ : ∀ x, K x * sessErr H D x ≤ D.δc H x) :
    Complete H D := by
  intro h64 x w hx hw ω
  classical
  have hrej : ∀ r : Fin (D.V.k x) → Coin, ¬ (run H (D.P w) D.V x ω r).out = .acc →
      ∃ p ∈ (univ : Finset (Fin (K x))), ∃ u, (D.checked x r)[p.val]? = some u ∧
        ∃ xS vS, D.session x u (run H (D.P w) D.V x ω r).seen = some (xS, vS) ∧ vS.out ≠ .acc := by
    intro r hr
    by_contra hno
    refine hr (hA x w hx hw ω r fun u hu xS vS hs => ?_)
    by_contra hout
    obtain ⟨p, hp, u', hu', heq⟩ := hK x w hx hw ω r u hu
    exact hno ⟨⟨p, hp⟩, mem_univ _, u', hu', xS, vS, heq ▸ hs, hout⟩
  have hsum : ∑ p : Fin (K x), Pr (fun r : Fin (D.V.k x) → Coin => ∃ u, (D.checked x r)[p.val]? = some u ∧
      ∃ xS vS, D.session x u (run H (D.P w) D.V x ω r).seen = some (xS, vS) ∧ vS.out ≠ .acc) ≤
      K x * sessErr H D x := by
    calc _ ≤ ∑ _p : Fin (K x), sessErr H D x :=
          sum_le_sum fun p _ => Pr_slot_le H D hF hfix hle hS hnn h64 x w hx hw ω p.val
      _ = K x * sessErr H D x := by simp
  have hbad := (Pr_mono hrej).trans ((Claims.Sampling.Pr_exists_le univ _).trans hsum)
  rw [Claims.Sampling.Pr_not] at hbad
  linarith [hδ x]

end Claims.Sampling.Reference
