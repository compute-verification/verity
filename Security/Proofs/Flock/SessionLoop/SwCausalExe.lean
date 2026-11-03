import Proofs.Flock.SessionLoop.VsOfBody

/-!
# The switched session is causal at a pad-blind worker's values

`RefSession.SwCausal` asks that the switched session's message at prefix `i` read the coins only below `i`. At the
values a pad-blind worker of the algebraic phase supplies (`Loop.vsOf`), a pad's value is the word the frame frees there,
along the coins up to the round that frees it:

* `Loop.freedAt_place`: a pad the frame frees is at a place of the absorbed order before the Ligerito values.
* `Loop.freedAt_le`: a pad at places of the absorbed order below the values of PIOP rounds `0` to `jj` is freed only after
  a history of at most `repBase + jj` openings (`freedAt_le_body`, `piopV_length_eq`).
* `Loop.freeWord_congr`: where every round that frees a pad does so after at most `t` openings, the word `freeWord`
  reads there is one at two coin vectors that agree below `t` (`freedAt_unique`).
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Claims.ZKServer

open _root_.Flock (V F128)
open Claims Claims.ZKMal
open FlockSoundness.Discharge
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (Stmt SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)
open _root_.Flock.Firewall.Frame (Cfg Masker absorbed nLig)

namespace Loop

section Freed

variable (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))

/-- **A pad the frame frees is one the PIOP absorbs**: at a place of the rep's absorbed order before the Ligerito
values (`Free.roundFree_pos`, at the rep's layout: `lay_ofLoopLink`). -/
theorem freedAt_place {x : Msg} {w : Flock.Wit} (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker)
    {ν : CoinLayer.Nonce} {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    {cs : List FlockDeploy.Coin} {j : Fin (jOf x)} {r : Fin 2} {i : ℕ} {a : F128}
    (hF : FreedAt (LeanRelease.ofLoopLink L₀ src real) wk x w ν s cs j r i a) :
    ∃ P, P < (absorbed (layE x j)).size - nLig (layE x j) ∧ (absorbed (layE x j))[P]! = i := by
  obtain ⟨t, κ, m₀, ρ₀, ops, v, m, out, s', fr, k, -, -, hpos, -, -, -, -, ha, hf, hrep, hk, hi, -⟩ := hF
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨q, hq⟩ : ∃ q, ops.length = q + 1 := ⟨ops.length - 1, by omega⟩
  have hst := start_of_answered Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real x
    (ν, t) (m₀, ρ₀) ops ha hx hq
  have hk2 : (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1 = 2 * j.val + r.val := by
    rw [← LeanRelease.repOf_add_two hx, show q + 2 = ops.length + 1 by omega]
    exact hrep
  rw [hk2, show (2 * j.val + r.val) / 2 = j.val by omega] at hst
  obtain ⟨lay, mask, hst⟩ := hst
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := L₀) src real j r hst
  have hord := cfg_order_of_rep (LeanRelease.ofLoopLink L₀ src real) (ρ := (ν, t)) (h := (m₀, ρ₀) :: ops)
    (k := frepOf j.val r.val) hrep
  rw [hlay] at hord
  have hP0 : m.pos + k < ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).nPiop :=
    (Free.roundFree_pos hf).2 k hk
  rw [hord.2] at hP0
  rw [hord.1] at hi
  exact ⟨m.pos + k, hP0, hi⟩

/-- **A pad below PIOP round `jj`'s values is freed by then**: where every place of the absorbed order that holds pad
`i` is below the values of PIOP rounds `0` to `jj` (`piopLen`), every round that frees it does so after at most
`repBase + jj` openings. -/
theorem freedAt_le {x : Msg} {w : Flock.Wit} (hSv : Serves x)
    (hP : (LeanRelease.ofLoopLink L₀ src real).ProgChecks x w)
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) {ν : CoinLayer.Nonce}
    {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z} {cs : List FlockDeploy.Coin}
    {j : Fin (jOf x)} {r : Fin 2} {i : ℕ} {a : F128}
    (hF : FreedAt (LeanRelease.ofLoopLink L₀ src real) wk x w ν s cs j r i a) {jj : ℕ}
    (hjj : jj < piopNR (stAt x j).m (stAt x j).c.kLog)
    (hi : ∀ P, P < (absorbed (layE x j)).size - nLig (layE x j) → (absorbed (layE x j))[P]! = i →
      P < piopLen (stAt x j).m (stAt x j).c.kLog (jj + 1)) :
    cs.length ≤ repBase x (2 * j.val + r.val) + jj := by
  obtain ⟨P, hPs, hPi⟩ := freedAt_place L₀ src real wk hF
  refine freedAt_le_body L₀ src real hSv hP wk hF (P := P) (by omega) hPi hjj ?_
  rw [piopV_length_eq _ _ _ _ _ _ _ _ (show jj + 1 ≤ piopNR (stAt x j).m (stAt x j).c.kLog from hjj)]
  exact hi P hPs hPi

end Freed

/-- **The word a pad-blind worker frees at a pad is one along coins agreeing up to its round**: where every round that
frees pad `i` does so after at most `t` openings, `freeWord` is one at two coin vectors that agree below `t`; the rounds
that free it read the same openings (`ofFn_take_congr`), and free it as one word (`freedAt_unique`). -/
theorem freeWord_congr
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hP : (LeanRelease.ofLoop Masked.dataE L₀ real).ProgChecks x w)
    (bx : AlgWorker (LeanRelease.ofLoop Masked.dataE L₀ real)) {ν : CoinLayer.Nonce}
    {c c' : Fin (refK x) → FlockDeploy.Coin} {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    {j : Fin (jOf x)} {r : Fin 2} {i t : ℕ}
    (hb : ∀ cs a, FreedAt _ (blind _ (bwOf _ bx)) x w ν s cs j r i a → cs.length ≤ t)
    (hcc : ∀ k : Fin (refK x), k.val < t → c k = c' k) :
    freeWord _ bx x w ν c s j r i = freeWord _ bx x w ν c' s j r i := by
  have hT : ∀ (d : Fin (refK x) → FlockDeploy.Coin) n, n < refK x → ∀ a,
      FreedAt _ (blind _ (bwOf _ bx)) x w ν s ((List.ofFn d).take n) j r i a → n ≤ t := fun d n hn a h => by
    have e := hb _ _ h
    rw [List.length_take, List.length_ofFn, Nat.min_eq_left hn.le] at e
    exact e
  have he : ∀ n, n ≤ t → (List.ofFn c).take n = (List.ofFn c').take n := fun n hn =>
    RefSession.ofFn_take_congr c c' n fun k hk => hcc k (by omega)
  have hiff : (∃ n a, n < refK x ∧ FreedAt _ (blind _ (bwOf _ bx)) x w ν s ((List.ofFn c).take n) j r i a) ↔
      (∃ n a, n < refK x ∧ FreedAt _ (blind _ (bwOf _ bx)) x w ν s ((List.ofFn c').take n) j r i a) :=
    ⟨fun ⟨n, a, hn, h⟩ => ⟨n, a, hn, by rw [← he n (hT c n hn a h)]; exact h⟩,
      fun ⟨n, a, hn, h⟩ => ⟨n, a, hn, by rw [he n (hT c' n hn a h)]; exact h⟩⟩
  have key : ∀ n a n' a', n < refK x → n' < refK x →
      FreedAt _ (blind _ (bwOf _ bx)) x w ν s ((List.ofFn c).take n) j r i a →
      FreedAt _ (blind _ (bwOf _ bx)) x w ν s ((List.ofFn c').take n') j r i a' → a = a' := by
    intro n a n' a' hn hn' hc hc'
    rw [he _ (hT c _ hn _ hc)] at hc
    rcases le_total n n' with hle | hle
    · exact freedAt_unique L₀ real hP bx hle (by rw [List.length_ofFn]; exact hn'.le) hc hc'
    · exact (freedAt_unique L₀ real hP bx hle (by rw [List.length_ofFn]; exact hn.le) hc' hc).symm
  unfold freeWord
  by_cases h : ∃ n a, n < refK x ∧ FreedAt _ (blind _ (bwOf _ bx)) x w ν s ((List.ofFn c).take n) j r i a
  · have h' := hiff.mp h
    rw [dite_eq_left h, dite_eq_left h']
    obtain ⟨hn, hc⟩ := Classical.choose_spec (Classical.choose_spec h)
    obtain ⟨hn', hc'⟩ := Classical.choose_spec (Classical.choose_spec h')
    exact congrArg some (key _ _ _ _ hn hn' hc hc')
  · rw [dite_eq_right h, dite_eq_right fun h' => h (hiff.mpr h')]

end Loop

end Claims.ZKServer
