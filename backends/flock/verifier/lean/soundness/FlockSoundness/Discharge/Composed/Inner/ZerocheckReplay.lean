import FlockSoundness.Refine.Exec
import FlockSoundness.Discharge.Composed.Inner.ZerocheckMasked

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
  sorry

end FlockSoundness.Discharge.Composed.ZerocheckMasked
