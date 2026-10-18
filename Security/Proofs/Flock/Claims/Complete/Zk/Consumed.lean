import Proofs.Flock.Claims.Complete.Zk.Json.Files
import Proofs.Flock.Claims.Complete.Zk.Count
import Proofs.Flock.Claims.Complete.Zk.Discharge
import Proofs.Flock.Claims.DeployRoundsZk

/-!
# Every recorded round of a masked rep is consumed (`ConsumedZk`)

`consumedZk_goal`: where the masked rep's steps return, the inner proof ends at its stream's last round.

* `verifyInnerWith_end`: the inner proof ends within its rounds, its last coins read from a recorded round.
* `RunZk.inner_k`: where the masked replay, the ring switch, the masked Ligerito and the inner proof return, the inner
  proof ends at the rep's round count (as `zkVerifyRepWith_rounds` counts it), within the rep's stream.
* The two streams together are the sends after the link (`Recs`), at most `rounds true` (`sendCountZk_goal`), so each
  holds exactly its rep's round count.
-/

namespace FlockComplete

open Flock Transcript FlockSoundness.Refine FlockSoundness.Discharge.ZkExec FlockDeploy

/-- **The inner proof ends within its rounds**: its last coins are read from a recorded round. -/
theorem verifyInnerWith_end (hs : Hashes V) {lay : Zk.PadLayout} {pr : Zk.InnerProof} {finalC : F128}
    {cons : Array (String × Zk.Lin)} {t t' : Transcript} {u : Unit}
    (h : Zk.verifyInnerWith hs lay pr finalC cons t = .ok (u, t')) : t'.k ≤ t'.rounds.size := by
  unfold Zk.verifyInnerWith at h
  simp only [liftTM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, -, h⟩ := label_bind h
  obtain ⟨t₂, -, h⟩ := absorbF_bind h
  obtain ⟨-, -, t₃, -, -, h⟩ := sampleWith_bind hs h
  obtain ⟨t₄, -, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₅, -, -, h⟩ := sampleVecWith_bind hs (by simp [Zk.triple]) h
  obtain ⟨t₆, -, h⟩ := absorbBytes_bind h
  obtain ⟨-, -, t₇, -, -, h⟩ := sampleWith_bind hs h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₈, -, h⟩ := absorbSlice_bind h
  obtain ⟨hlt, -, t₉, r₉, kk₉, h⟩ := sampleVecWith_bind hs (by decide) h
  obtain ⟨tc, -, h⟩ := vlift_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₁₀, hk₁₀⟩ := forIn_keeps _ (by
      intro i s tt b tt' hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨rfl, rfl⟩) _ _ _ _ _ hloop
  rw [hr₁₀, r₉, hk₁₀, kk₉]
  omega

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **Where a masked rep's steps return, the inner proof ends at the rep's round count**, within the rep's stream: the
replay's `m + (kLog − 6)`, the ring switch's `claims + 1`, Ligerito's `k0 + 7 + 11 R` and the inner proof's `4`. -/
theorem RunZk.inner_k (hRp : Replay H rel) (hRS : RingSwitchZk H rel) (hLg : LigeritoZk H rel) (hIn : Inner H rel)
    (o : RunZk H rel) {st : Flock.Stmt} (htabs : o.tabs = #[Setup.ofCircuit st]) {v : Schedule × ℕ}
    (hv : Zk.schedule st.m = .ok v) (hsch : o.sch 0 = v.1) {r : ℕ} (hr : r < 2) :
    (o.inner 0 r).2.k = (st.m + ((Setup.ofCircuit st).kLog - 6)) + ((2 + (o.extras 0).size) + 1) +
        (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)) + 4 ∧
      (o.inner 0 r).2.k ≤ (o.stream 0 r).rounds.size := by
  have h0 : 0 < o.tabs.size := by rw [htabs]; simp
  have ht0 : o.tabs[0]! = Setup.ofCircuit st := by rw [htabs]; rfl
  obtain ⟨h25, -, hsm, hk0, -, hdim, hlv, -, -⟩ := zk_schedule_seams st.m v.1 v.2 hv
  obtain ⟨hn7, h2, -, -, -, -, -⟩ := hdim
  have e₃ := hLg o 0 h0 r hr
  rw [hsch] at e₃
  obtain ⟨hr₁, hk₁⟩ := replayWith_rounds o.hs (t' := (o.rp 0 r).2) (rp := (o.rp 0 r).1)
    (by rw [ht0]; exact (by omega : 14 ≤ st.m)) (hRp o 0 h0 r hr)
  obtain ⟨hr₂, hk₂⟩ := ringSwitchWith_rounds o.hs (t' := (o.sw 0 r).2) (out := (o.sw 0 r).1)
    (by simp [ringClaimsZk]) (hRS o 0 h0 r hr)
  obtain ⟨hr₃, hk₃⟩ := zkLigeritoWith_rounds o.hs _ v.1 _ _ _ _ _ (t' := (o.lg 0 r).2) (rc := (o.lg 0 r).1)
    (by omega) hk0 h2 (fun i hi => (hlv i hi).q_ne_zero) e₃
  obtain ⟨hr₄, hk₄⟩ := verifyInnerWith_rounds o.hs (t' := (o.inner 0 r).2) (u := (o.inner 0 r).1) (hIn o 0 h0 r hr)
  have hend := verifyInnerWith_end o.hs (t' := (o.inner 0 r).2) (u := (o.inner 0 r).1) (hIn o 0 h0 r hr)
  refine ⟨?_, ?_⟩
  · rw [hk₄, hk₃, hk₂, hk₁, ht0]
    simp [ringClaimsZk, RunZk.t₀, Transcript.init, Setup.ofCircuit]
  · rw [hr₄, hr₃, hr₂, hr₁] at hend
    exact hend

/-- **Every recorded round of a masked rep is consumed** wherever its steps return: the goal `ConsumedZk`. -/
theorem consumedZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : JsonRT.StmtOkZk H rel)
    (hP : JsonRT.ProverOkZk H rel) (hRp : Replay H rel) (hRS : RingSwitchZk H rel) (hLg : LigeritoZk H rel)
    (hIn : Inner H rel) : ConsumedZk H rel := by
  intro o i hi r hr
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, -, -, -, ⟨bs, hbs, hrec⟩, -, -, hsess, -, -⟩ :=
    JsonRT.RunZk.json S h64 hS hP o
  rw [htabs] at hi
  obtain rfl : i = 0 := by simpa using hi
  obtain ⟨v, hv⟩ := schedulesZk_goal H rel o (Setup.ofCircuit st) (by simp [htabs])
  have hsch : o.sch 0 = v.1 := by simp [RunZk.sch, zkSchedsOf, schedZkOf, htabs, hv]
  have hrs1 := JsonRT.repStreams_one ht
  obtain ⟨s0, s1, rfl⟩ := JsonRT.two_of (xs := streams) (by rw [hss, hrs1]; rfl)
  have hstream : ∀ r < 2, (o.stream 0 r).rounds.size = (if r = 0 then s0 else s1).size := by
    intro r hr
    unfold RunZk.stream
    rw [hsess]
    simp only [JsonRT.sessOf, hrs1, ht, List.zip_toArray, List.zip_cons_cons, List.zip_nil_right, List.map_toArray,
      List.map_cons, List.map_nil, List.find?_toArray, JsonRT.streamRec]
    rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl <;> simp
  have hlen := JsonRT.Recs.length_eq hrec
  have hb : o.bodies.size = 1 + bs.length := by
    rw [← Array.length_toList, hbs]
    simp only [List.length_cons]
    omega
  have hc := sendCountZk_goal H rel o
  rw [hst] at hc
  simp only [FlockDeploy.rounds, List.foldl_toArray', List.foldl_cons, List.foldl_nil] at hc
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.length_append, Array.length_toList] at hlen
  obtain ⟨k0, le0⟩ := RunZk.inner_k hRp hRS hLg hIn o htabs hv hsch (r := 0) (by omega)
  obtain ⟨k1, le1⟩ := RunZk.inner_k hRp hRS hLg hIn o htabs hv hsch (r := 1) (by omega)
  have hx : (o.extras 0).size = o.spec.points * st.regions.size := by
    have he := JsonRT.repExtrasZk_goal S h64 hS hP o 0 (by simp [htabs])
    simp only [htabs, List.getElem!_toArray, List.getElem!_cons_zero] at he
    rw [FlockDeploy.extraClaims_size (st := st) he, hsess]
    simp [JsonRT.sessOf, pointsOf, Nat.mul_comm]
  have hrep : repRounds true o.spec st = (o.inner 0 0).2.k := by
    have hv' : Zk.schedule st.m = .ok v := hv
    rw [k0, hx]
    simp [repRounds, hv', Setup.ofCircuit]
  have e0 := hstream 0 (by omega)
  have e1 := hstream 1 (by omega)
  simp only [↓reduceIte, one_ne_zero] at e0 e1
  rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl
  · simp only [bne_eq_false_iff_eq]
    omega
  · simp only [bne_eq_false_iff_eq]
    omega

end FlockComplete
