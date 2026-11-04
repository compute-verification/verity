import Proofs.Properties.Verity.Sampled.Record
import Proofs.Flock.Claims.Sampling.Prefix

/-!
# Every checked unit's session is the record's one Flock session (`SessionRuns`)

At the record's statement the verifier runs `DRAW_ROUNDS` draw rounds and the draw file round, then one session of
flock-verify `--zk` at the statement the draw file sets up (`prefixAt`: `SamplingDeploy.PrefixAt` with the draw file
round's check `okAt`); there is no unit anywhere else (`eq_xOf_of_nPU`). Every unit's rounds are the session's, and
`sessionOf` reads a view whose draw file round passed as the session's statement and the rounds after it, and one whose
draw file round refused as `refused`, so `SessionRuns` is `SamplingDeploy.sessionRuns_prefix` at flock-verify `--zk`
against SHA-512 (`flockAt_sha512`).
-/

namespace Verity.Sampled.Record

open Claims Claims.Sampling

variable (A : Live) (o : Own) (e : Ext)

open Classical in
/-- **The draw file round's check**, on the messages and coins so far: `drawOk` at the draw file and the draw coins. -/
noncomputable def okAt (as : List Msg) (cs : List FlockDeploy.Coin) : Bool :=
  decide (drawOk A o (as.getD DRAW_ROUNDS []) (cs.take DRAW_ROUNDS))

theorem drawOk_of_okAt {as : List Msg} {cs : List FlockDeploy.Coin} (h : okAt A o as cs = true) :
    drawOk A o (as.getD DRAW_ROUNDS []) (cs.take DRAW_ROUNDS) :=
  of_decide_eq_true h

/-- **Units are only at the record's statement**: elsewhere the statement doesn't parse. -/
theorem eq_xOf_of_nPU {x : Msg} (hx : 0 < nPU (record A o e) x) : x = xOf A o := by
  by_contra h
  have hp : (record A o e).parse x = none := by
    show (code A o e).parse x = none
    simp only [code, h, ↓reduceIte]
  simp [nPU, hp] at hx

theorem record_session_xOf (u : ℕ) (v : Seen) : (record A o e).session (xOf A o) u v = some (sessionOf A o v) := by
  show (code A o e).session (xOf A o) u v = _
  simp only [code, ↓reduceIte, ite_true]

/-- **The record's verifier is a prefix, then one session**: the draw rounds answer a coin, the draw file round answers
`[]` where `okAt` passes and refuses elsewhere, and past it each round is flock-verify `--zk`'s at the session's
statement, with `kS` rounds where `okAt` passed (`numsOk`). -/
theorem prefixAt :
    SamplingDeploy.PrefixAt FlockDeploy.sha512 (V A o) (FlockDeploy.V true) (xOf A o) (DRAW_ROUNDS + 1) (kS A o)
      (okAt A o) (sessStmt A o) where
  pos := Nat.succ_pos _
  k := V_k_xOf A o
  pre as cs _ hl _ := by
    rw [V_next_xOf]
    unfold nextAt
    rw [ite_eq_left (show as.length ≤ DRAW_ROUNDS by omega)]
    exact ⟨_, rfl⟩
  gate as cs hl _ := by
    rw [V_next_xOf]
    unfold nextAt
    rw [ite_eq_right (show ¬as.length ≤ DRAW_ROUNDS by omega), ite_eq_left hl]
    by_cases hd : drawOk A o (as.getD DRAW_ROUNDS []) (cs.take DRAW_ROUNDS)
    · have ho : okAt A o as cs = true := decide_eq_true hd
      rw [ite_eq_left hd, ho]
      all_goals rfl
    · have ho : okAt A o as cs = false := decide_eq_false hd
      rw [ite_eq_right hd, ho]
      all_goals rfl
  sess as cs hl _ _ := by
    rw [V_next_xOf]
    unfold nextAt
    rw [ite_eq_right (show ¬as.length ≤ DRAW_ROUNDS by omega),
      ite_eq_right (show ¬as.length = DRAW_ROUNDS + 1 by omega)]
    all_goals rfl
  nOk as cs h := by
    have h2 := congrArg Prod.fst (drawOk_of_okAt A o h).2
    exact h2

/-- **`SessionRuns` at the record**: every checked unit's session is the record's one flock-verify `--zk` session, on
the rounds after the draw file round. -/
theorem sessionRuns : SessionRuns FlockDeploy.sha512 (record A o e) :=
  SamplingDeploy.sessionRuns_prefix FlockDeploy.sha512 (record A o e) SamplingDeploy.flockAt_sha512
    (fun _ => DRAW_ROUNDS + 1) (fun _ => kS A o) (fun _ => okAt A o) (fun _ => sessStmt A o) (fun _ => numsOk A o)
    (fun _ => xS₀ A o)
    (fun x hx => by
      obtain rfl := eq_xOf_of_nPU A o e hx
      exact prefixAt A o)
    (fun _ _ _ h => (drawOk_of_okAt A o h).2)
    (fun _ => rfl)
    (fun x u t => by
      rw [record_rounds]
      simp [after])
    (fun x u hx v d hd hok => by
      obtain rfl := eq_xOf_of_nPU A o e hx
      have hd' : v.msgs[DRAW_ROUNDS]? = some (d, .msg []) := hd
      have hok' : numsOk A o d := hok
      rw [record_session_xOf]
      unfold sessionOf
      simp only [hd', hok', ↓reduceIte]
      all_goals rfl)
    (fun x u hx v d hd => by
      obtain rfl := eq_xOf_of_nPU A o e hx
      have hd' : v.msgs[DRAW_ROUNDS]? = some (d, .rej) := hd
      rw [record_session_xOf]
      unfold sessionOf
      simp only [hd']
      all_goals rfl)

end Verity.Sampled.Record
