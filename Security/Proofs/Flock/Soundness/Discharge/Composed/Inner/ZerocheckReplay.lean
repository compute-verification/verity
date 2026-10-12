import Proofs.Flock.Soundness.Refine.Exec
import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckMasked

/-!
# `Zk.replay`'s zerocheck constraints are the `ZerocheckMasked` terms

`replay_zerocheck`: when `Zk.replay` succeeds from the transcript `t`, its first two constraints are
`zerocheck c_eval = computedC + fc` and `zerocheck final a·b = running (m − 6) + prod` on the proof's (masked) zerocheck,
at the coins it draws from `t`: the outer coins `rOuterAt t` (round `t.k + 1`; `r = pinned ++ rOuterAt t`), the skip
coin `zAt t` (round `t.k + 2`) and the residual coins `rhoAt t (m − 6)` (rounds `t.k + 3 …`, the loop's pushed coins).
Its `c` claim sits at `(L_s(z), r)` and its `c` value is the form `fc`.
-/

namespace FlockSoundness.Discharge.Composed.ZerocheckMasked

open Flock Flock.Zk FlockSoundness.Refine

/-- The zerocheck's outer coins `r_outer`, drawn at round `t.k + 1` of the transcript the replay starts on. -/
def rOuterAt (t : Transcript) : Array F128 := (t.rounds[t.k + 1]!).coins

/-- The skip coin `z`, drawn at round `t.k + 2`. -/
def zAt (t : Transcript) : F128 := (t.rounds[t.k + 2]!).coins.getD 0 0

/-- The `n` residual coins `ρ`, drawn at rounds `t.k + 3 + i`, in the order the loop pushes them. -/
def rhoAt (t : Transcript) (n : ℕ) : Array F128 :=
  ((List.range n).map fun i => (t.rounds[t.k + 3 + i]!).coins.getD 0 0).toArray

/-- Residual round `j`'s update at the coin `rr` (`roundL` reads it as `rho[j]!`). -/
def roundAt (lay : PadLayout) (zc : ZerocheckProof) (r : Array F128) (j : ℕ) (rr : F128) (running : Lin) : Lin :=
  let (m1, minf) := zc.rounds[j]!
  let req := r[j]!
  let g1 := Lin.masked m1 (lay.ml j 0) lay.len
  let ginf := Lin.masked minf (lay.ml j 1) lay.len
  let g0 := (running.add (g1.scale req)).scale (1 + req).inv
  ((g0.scale (1 + rr)).add (g1.scale rr)).add (ginf.scale (rr * (1 + rr)))

theorem roundL_eq (lay : PadLayout) (zc : ZerocheckProof) (r rho : Array F128) (j : ℕ) (x : Lin) :
    roundL lay zc r rho j x = roundAt lay zc r j rho[j]! x := rfl

/-- The residual loop's state after `n` steps on the coins `c 0, c 1, …`: the running form, read at any coin array
`R` that agrees with them, and the coins pushed. -/
theorem fold_rounds (lay : PadLayout) (zc : ZerocheckProof) (r : Array F128) (z : F128) (c : ℕ → F128)
    (R : Array F128) (N : ℕ) (hR : ∀ i < N, R[i]! = c i) :
    ∀ n ≤ N, (List.range' 0 n).foldl (fun (s : Lin × Array F128) i => (roundAt lay zc r i (c i) s.1, s.2.push (c i)))
        (running0 lay zc z, #[]) = (running lay zc r R z n, ((List.range n).map c).toArray)
  | 0, _ => rfl
  | n + 1, hn => by
      rw [List.range'_concat, List.foldl_append, fold_rounds lay zc r z c R N hR n (by omega), List.foldl_cons,
        List.foldl_nil, running, roundL_eq, hR n (by omega)]
      simp [List.range_succ]

/-- **The replay's zerocheck constraints are `ZerocheckMasked`'s terms**, at the coins it draws. -/
theorem replay_zerocheck (st : Setup) (lay : PadLayout) (p : Proof) (root : ByteArray) {t t' : Transcript}
    {rp : Replayed} (hm : 14 ≤ st.m) (h : Zk.replay st lay p root t = .ok (rp, t')) :
    rp.cons[0]! = ("zerocheck c_eval", (computedC lay p.zerocheck (zAt t)).add (fcL lay p.zerocheck)) ∧
      rp.cons[1]! = ("zerocheck final a·b",
        (running lay p.zerocheck (pinned ++ rOuterAt t) (rhoAt t (st.m - 6)) (zAt t) (st.m - 6)).add
          (prodL lay p.zerocheck)) ∧
      rp.claims[1]! = { value := 0, skip := skipWeights (zAt t), x := pinned ++ rOuterAt t } ∧
      rp.cValue = fcL lay p.zerocheck := by
  unfold Zk.replay at h
  dsimp only at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨t₅, k₅, h⟩ := label_bind h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨-, -, t₇, r₇, kk₇, h⟩ := sampleVec_bind (by decide) h
  obtain ⟨-, -, t₈, r₈, kk₈, h⟩ := sampleVec_bind (by omega) h
  obtain ⟨t₉, k₉, h⟩ := absorbSlice_bind h
  obtain ⟨t₁₀, k₁₀, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₁₁, r₁₁, kk₁₁, h⟩ := sample_bind h
  obtain ⟨s, t₁₂, hloop, h⟩ := bind_ok h
  obtain ⟨t₁₃, -, h⟩ := absorbF_bind h
  obtain ⟨t₁₄, -, h⟩ := absorbF_bind h
  obtain ⟨t₁₅, -, h⟩ := label_bind h
  obtain ⟨-, -, t₁₆, -, -, h⟩ := sample_bind h
  obtain ⟨-, -, t₁₇, -, -, h⟩ := sample_bind h
  obtain ⟨s', t₁₈, -, h⟩ := bind_ok h
  rcases hf : st.fold.folded ((t₁₅.rounds[t₁₅.k]!).coins.getD 0 0) ((t₁₆.rounds[t₁₆.k]!).coins.getD 0 0)
      (skipWeights ((t₁₀.rounds[t₁₀.k]!).coins.getD 0 0)) (s.2.extract 0 (st.kLog - 6)) s'.2 with err | v
  · rw [hf] at h
    obtain ⟨_, _, h₀, _⟩ := bind_ok h
    cases h₀
  rw [hf] at h
  simp only [pure_bind] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₂₀, -, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₂₁, -, -, h⟩ := sample_bind h
  obtain ⟨hrp, -⟩ := pure_ok h
  subst hrp
  have hr₆ : t₆.rounds = t.rounds := by rw [k₆.2, k₅.2, k₄.2, k₃.2, k₂.2, k₁.2]
  have hk₆ : t₆.k = t.k := by rw [k₆.1, k₅.1, k₄.1, k₃.1, k₂.1, k₁.1]
  have hr₇ : t₇.rounds = t.rounds := by rw [r₇, hr₆]
  have hk₇ : t₇.k = t.k + 1 := by rw [kk₇, hk₆]
  have hr₁₀ : t₁₀.rounds = t.rounds := by rw [k₁₀.2, k₉.2, r₈, hr₇]
  have hk₁₀ : t₁₀.k = t.k + 2 := by rw [k₁₀.1, k₉.1, kk₈, hk₇]
  have hr₁₁ : t₁₁.rounds = t.rounds := by rw [r₁₁, hr₁₀]
  have hk₁₁ : t₁₁.k = t.k + 3 := by rw [kk₁₁, hk₁₀]
  have hO : (t₇.rounds[t₇.k]!).coins = rOuterAt t := by rw [hr₇, hk₇]; rfl
  have hZ : (t₁₀.rounds[t₁₀.k]!).coins.getD 0 0 = zAt t := by rw [hr₁₀, hk₁₀]; rfl
  rw [hO, hZ] at hloop ⊢
  rw [forIn_range_eq] at hloop
  obtain ⟨-, -, -, hs⟩ := forIn_rounds _
    (fun i (x : Lin × Array F128) cs => (roundAt lay p.zerocheck (pinned ++ rOuterAt t) i (cs.getD 0 0) x.1,
      x.2.push (cs.getD 0 0))) 1 (fun i x tt b tt' hb => by
        obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
        obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
        obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
        obtain ⟨rfl, rfl⟩ := pure_ok hb
        have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
        have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
        refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
        rw [← e1, ← e2]
        rfl) (st.m - 6) 0 t₁₁.k _ t₁₁ t₁₂ s rfl hloop
  have hrho : ∀ i < st.m - 6, (rhoAt t (st.m - 6))[i]! =
      (fun i => (t₁₁.rounds[t₁₁.k + i]!).coins.getD 0 0) i := fun i hi => by
    rw [getElem!_pos _ i (by simp [rhoAt]; omega), hr₁₁, hk₁₁]
    simp [rhoAt]
  have hs' := hs.trans
    (fold_rounds lay p.zerocheck (pinned ++ rOuterAt t) (zAt t) _ (rhoAt t (st.m - 6)) (st.m - 6) hrho _ le_rfl)
  subst hs'
  refine ⟨rfl, rfl, rfl, rfl⟩

end FlockSoundness.Discharge.Composed.ZerocheckMasked
