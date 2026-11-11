module

public import Proofs.Properties.Verity.Sampled.Record.Facts
public import Proofs.Flock.Claims.Sampling.Prefix

/-!
# Every checked unit's session is the record's one Flock session (`SessionRuns`)

At the record's statement the verifier runs `DRAW_ROUNDS` draw rounds and the draw file round, then one session of
flock-verify `--zk` at the statement the draw file sets up (`prefixAt`: `SamplingDeploy.PrefixAt` with the draw file
round's check `okAt`); there is no unit anywhere else (`eq_xOf_of_nPU`). Every unit's rounds are the session's, and
`sessionOf` reads a view whose draw file round passed as the session's statement and the rounds after it, and one whose
draw file round refused as `refused`, so `SessionRuns` is `SamplingDeploy.sessionRuns_prefix` at flock-verify `--zk`
against SHA-512 (`flockAt_sha512`).
-/

@[expose] public section

namespace Verity.Sampled.Record

open Claims Claims.Sampling

variable (A : Live) (o : Own) (e : Ext)

open Classical in
/-- **The draw file round's check**, on the messages and coins so far: `drawOk` at the draw file and the draw coins. -/
noncomputable def okAt (as : List Msg) (cs : List FlockDeploy.Coin) : Bool :=
  decide (drawOk A o (as.getD DRAW_ROUNDS []) (cs.take DRAW_ROUNDS))

open Classical in
theorem drawOk_of_okAt {as : List Msg} {cs : List FlockDeploy.Coin} (h : okAt A o as cs = true) :
    drawOk A o (as.getD DRAW_ROUNDS []) (cs.take DRAW_ROUNDS) :=
  of_decide_eq_true h

/-! The record's fields at a statement `x`, by `rfl` only where `x` is a variable: at `xOf A o` itself,
`ite_eq_left rfl` takes the branch, since unfolding the record there would evaluate the statement's bytes. The parse
is `record_parse_eq`. -/

theorem record_session (x : Msg) (u : ℕ) (v : Seen) :
    (record A o e).session x u v = if x = xOf A o then some (sessionOf A o v) else none := rfl

theorem record_flockV : (record A o e).flock.V = FlockDeploy.V true := rfl

theorem flockV_next : (FlockDeploy.V true).next = FlockDeploy.next true := rfl

theorem nums_fst (xS : Msg) : (Nums xS).1 = (FlockDeploy.V true).k xS := rfl

/-- **Units are only at the record's statement**: elsewhere the statement doesn't parse. -/
theorem eq_xOf_of_nPU {x : Msg} (hx : 0 < nPU (record A o e) x) : x = xOf A o := by
  by_contra h
  have hp : (record A o e).parse x = none := by
    show (code A o e).parse x = none
    simp only [code, h, ↓reduceIte]
  simp [nPU, hp] at hx

theorem record_session_xOf (u : ℕ) (v : Seen) : (record A o e).session (xOf A o) u v = some (sessionOf A o v) :=
  (record_session A o e (xOf A o) u v).trans (ite_eq_left rfl)

open Classical in
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
      ite_eq_right (show ¬as.length = DRAW_ROUNDS + 1 by omega), flockV_next, Nat.add_sub_cancel]
  nOk as cs h := by
    have h2 : Nums (sessStmt A o (as.getD DRAW_ROUNDS [])) = Nums (xS₀ A o) := (drawOk_of_okAt A o h).2.1
    have h3 := congrArg Prod.fst h2
    rw [nums_fst, nums_fst] at h3
    rw [Nat.add_sub_cancel, kS]
    exact h3

/-- **`SessionRuns` at the record**: every checked unit's session is the record's one flock-verify `--zk` session, on
the rounds after the draw file round. -/
theorem sessionRuns : SessionRuns FlockDeploy.sha512 (record A o e) :=
  SamplingDeploy.sessionRuns_prefix FlockDeploy.sha512 (record A o e) SamplingDeploy.flockAt_sha512
    (fun _ => DRAW_ROUNDS + 1) (fun _ => kS A o) (fun _ => okAt A o) (fun _ => sessStmt A o)
    (fun _ d => numsOk A o d ∧ Fits A o d) (fun _ => xS₀ A o)
    (fun x hx => by
      obtain rfl := eq_xOf_of_nPU A o e hx
      rw [record_V, record_flockV]
      exact prefixAt A o)
    (fun _ _ _ h => (drawOk_of_okAt A o h).2)
    (fun _ => by rw [record_flockV, kS])
    (fun x u t => by
      rw [record_rounds]
      simp [after])
    (fun x u hx v d hd hok => by
      obtain rfl := eq_xOf_of_nPU A o e hx
      have hd' : v.msgs[DRAW_ROUNDS]? = some (d, .msg []) := hd
      have hok' : numsOk A o d ∧ Fits A o d := hok
      rw [record_session_xOf]
      unfold sessionOf
      simp only [hd', hok', and_self, ↓reduceIte])
    (fun x u hx v d hd => by
      obtain rfl := eq_xOf_of_nPU A o e hx
      have hd' : v.msgs[DRAW_ROUNDS]? = some (d, .rej) := hd
      rw [record_session_xOf]
      unfold sessionOf
      simp only [hd']
      all_goals rfl)

end Verity.Sampled.Record
