module

public import Proofs.Properties.Verity.Sampled.Record.Facts
public import Proofs.Flock.Claims.Sampling.Prefix

/-! Every checked unit's session is the record's one Flock session (`SessionRuns`). -/

@[expose] public section

namespace Verity.Sampled.Record

open Claims Claims.Sampling

variable (A : Live) (o : Own) (e : Ext)

open Classical in
noncomputable def okAt (as : List Msg) (cs : List FlockDeploy.Coin) : Bool :=
  decide (sampleOk A o (as.getD SAMPLE_ROUNDS []) (cs.take SAMPLE_ROUNDS))

open Classical in
theorem sampleOk_of_okAt {as : List Msg} {cs : List FlockDeploy.Coin} (h : okAt A o as cs = true) :
    sampleOk A o (as.getD SAMPLE_ROUNDS []) (cs.take SAMPLE_ROUNDS) :=
  of_decide_eq_true h

theorem record_session (x : Msg) (u : ℕ) (v : Seen) :
    (record A o e).session x u v = if x = xOf A o then some (sessionOf A o v) else none := rfl

theorem record_flockV : (record A o e).flock.V = FlockDeploy.V true := rfl

theorem flockV_next : (FlockDeploy.V true).next = FlockDeploy.next true := rfl

theorem nums_fst (xS : Msg) : (Nums xS).1 = (FlockDeploy.V true).k xS := rfl

theorem eq_xOf_of_nPU {x : Msg} (hx : 0 < nPU (record A o e) x) : x = xOf A o := by
  by_contra h
  have hp : (record A o e).parse x = none := by
    show (code A o e).parse x = none
    simp only [code, h, ↓reduceIte]
  simp [nPU, hp] at hx

theorem record_session_xOf (u : ℕ) (v : Seen) : (record A o e).session (xOf A o) u v = some (sessionOf A o v) :=
  (record_session A o e (xOf A o) u v).trans (ite_eq_left rfl)

open Classical in
theorem prefixAt :
    SamplingDeploy.PrefixAt FlockDeploy.sha512 (V A o) (FlockDeploy.V true) (xOf A o) (SAMPLE_ROUNDS + 1) (kS A o)
      (okAt A o) (sessStmt A o) where
  pos := Nat.succ_pos _
  k := V_k_xOf A o
  pre as cs _ hl _ := by
    rw [V_next_xOf]
    unfold nextAt
    rw [ite_eq_left (show as.length ≤ SAMPLE_ROUNDS by omega)]
    exact ⟨_, rfl⟩
  gate as cs hl _ := by
    rw [V_next_xOf]
    unfold nextAt
    rw [ite_eq_right (show ¬as.length ≤ SAMPLE_ROUNDS by omega), ite_eq_left hl]
    by_cases hd : sampleOk A o (as.getD SAMPLE_ROUNDS []) (cs.take SAMPLE_ROUNDS)
    · have ho : okAt A o as cs = true := decide_eq_true hd
      rw [ite_eq_left hd, ho]
      all_goals rfl
    · have ho : okAt A o as cs = false := decide_eq_false hd
      rw [ite_eq_right hd, ho]
      all_goals rfl
  sess as cs hl _ _ := by
    rw [V_next_xOf]
    unfold nextAt
    rw [ite_eq_right (show ¬as.length ≤ SAMPLE_ROUNDS by omega),
      ite_eq_right (show ¬as.length = SAMPLE_ROUNDS + 1 by omega), flockV_next, Nat.add_sub_cancel]
  nOk as cs h := by
    have h2 : Nums (sessStmt A o (as.getD SAMPLE_ROUNDS [])) = Nums (xS₀ A o) := (sampleOk_of_okAt A o h).2.1
    have h3 := congrArg Prod.fst h2
    rw [nums_fst, nums_fst] at h3
    rw [Nat.add_sub_cancel, kS]
    exact h3

theorem sessionRuns : SessionRuns FlockDeploy.sha512 (record A o e) :=
  SamplingDeploy.sessionRuns_prefix FlockDeploy.sha512 (record A o e) SamplingDeploy.flockAt_sha512
    (fun _ => SAMPLE_ROUNDS + 1) (fun _ => kS A o) (fun _ => okAt A o) (fun _ => sessStmt A o)
    (fun _ d => numsOk A o d ∧ Fits A o d) (fun _ => xS₀ A o)
    (fun x hx => by
      obtain rfl := eq_xOf_of_nPU A o e hx
      rw [record_V, record_flockV]
      exact prefixAt A o)
    (fun _ _ _ h => (sampleOk_of_okAt A o h).2)
    (fun _ => by rw [record_flockV, kS])
    (fun x u t => by
      rw [record_rounds]
      simp [after])
    (fun x u hx v d hd hok => by
      obtain rfl := eq_xOf_of_nPU A o e hx
      have hd' : v.msgs[SAMPLE_ROUNDS]? = some (d, .msg []) := hd
      have hok' : numsOk A o d ∧ Fits A o d := hok
      rw [record_session_xOf]
      unfold sessionOf
      simp only [hd', hok', and_self, ↓reduceIte])
    (fun x u hx v d hd => by
      obtain rfl := eq_xOf_of_nPU A o e hx
      have hd' : v.msgs[SAMPLE_ROUNDS]? = some (d, .rej) := hd
      rw [record_session_xOf]
      unfold sessionOf
      simp only [hd']
      all_goals rfl)

end Verity.Sampled.Record
