module

public import Proofs.Flock.SessionLoop.VsOfBody
public import Proofs.Flock.SessionLoop.L0FixedHonest
public import Proofs.Flock.Claims.ZK.FinishSwMkAny
public import Proofs.Flock.SessionLoop.RepSrc

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
* `Loop.vsWord_vsOf_l0`: no round frees a pad from level 0's on (`freeWord_ligAt`), so there `vsOf` is the honest values,
  and `Loop.L0Fixed` holds at `vsOf` where it holds at them (`l0Fixed_vsOf_of`).
* **`Loop.l0Fixed_vsOf`**: `Loop.L0Fixed (vsOf bx) x w` at every `bx`, live statement and witness of the masked relation
  (`l0Fixed_vsOf_of` at `l0Fixed_honestVals`).
* **`Loop.swCausal_oneTape_vsOf`**: one-tape `SwCausal` at `vsOf` on `Loop.SwOk`'s coin vectors
  (`RefSession.SwCausalOn`), at every `bx`: where `vsOf` is `vsServed` its rows hold at both (`RowsOK`,
  `rOuter_ne_one_of_swOk`), and elsewhere it is the honest values (`swCausal_rows`).
-/

@[expose] public section

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

/-- **The places of pad `i` in the absorbed order**: `i` itself up to `final_b`'s, `i - 1` after. -/
theorem absorbed_place (l : _root_.Flock.Zk.PadLayout) {P i : ℕ} (hP : P < (absorbed l).size)
    (h : (absorbed l)[P]! = i) : P ≤ i ∧ (l.fa + 2 ≤ i → P + 1 = i) := by
  rw [absorbed_getElem! l hP] at h
  split_ifs at h <;> omega

/-- Pad `i`'s places in the absorbed order are below `b` where its last is. -/
theorem place_hi (l : _root_.Flock.Zk.PadLayout) {i b : ℕ} (h : (if i < l.fa + 2 then i else i - 1) < b) :
    ∀ P, P < (absorbed l).size - nLig l → (absorbed l)[P]! = i → P < b := fun P hP hPi => by
  obtain ⟨a1, a2⟩ := absorbed_place l (by omega) hPi
  split_ifs at h with hi
  · omega
  · have := a2 (by omega)
    omega

/-- A pad past the PIOP's values in the absorbed order is at none of its places. -/
theorem place_lig (l : _root_.Flock.Zk.PadLayout) {i : ℕ} (h : (absorbed l).size - nLig l + 1 ≤ i) :
    ∀ P, P < (absorbed l).size - nLig l → (absorbed l)[P]! ≠ i := fun P hP hPi => by
  obtain ⟨a1, a2⟩ := absorbed_place l (by omega) hPi
  have hs := absorbed_size l
  have := a2 (by omega)
  omega

section Words

variable (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))

/-- **`freeWord` at a pad below PIOP round `jj`'s values reads the coins below `repBase + jj`** (`freedAt_le`). -/
theorem freeWord_le {x : Msg} {w : Flock.Wit} (hSv : Serves x)
    (hP : (LeanRelease.ofLoopLink L₀ src real).ProgChecks x w) (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real))
    {ν : CoinLayer.Nonce} {c c' : Fin (refK x) → FlockDeploy.Coin}
    {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z} {j : Fin (jOf x)} {r : Fin 2}
    {i jj : ℕ} (hjj : jj < piopNR (stAt x j).m (stAt x j).c.kLog)
    (hi : ∀ P, P < (absorbed (layE x j)).size - nLig (layE x j) → (absorbed (layE x j))[P]! = i →
      P < piopLen (stAt x j).m (stAt x j).c.kLog (jj + 1))
    (hcc : ∀ k : Fin (refK x), k.val < repBase x (2 * j.val + r.val) + jj → c k = c' k) :
    freeWord _ bx x w ν c s j r i = freeWord _ bx x w ν c' s j r i :=
  freeWord_congr (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real hP bx
    (fun _ _ hF => freedAt_le L₀ src real hSv hP _ hF hjj hi) hcc

/-- **`freeWord` at a pad the PIOP never absorbs is none at every coins**: no round frees it (`freedAt_place`). -/
theorem freeWord_lig {x : Msg} {w : Flock.Wit} (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real))
    {ν : CoinLayer.Nonce} {c : Fin (refK x) → FlockDeploy.Coin}
    {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z} {j : Fin (jOf x)} {r : Fin 2} {i : ℕ}
    (hi : ∀ P, P < (absorbed (layE x j)).size - nLig (layE x j) → (absorbed (layE x j))[P]! ≠ i) :
    freeWord _ bx x w ν c s j r i = none := by
  unfold freeWord
  exact dite_eq_right fun ⟨_, _, _, hF⟩ => by
    obtain ⟨P, hPs, hPi⟩ := freedAt_place L₀ src real _ hF
    exact hi P hPs hPi

end Words

section Vals

variable (L : LeanRelease Flock.Wit
  ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))

/-- **`vsOfWord` is one at two coin vectors** where `freeWord` is and the table's own value is. -/
theorem vsOfWord_congr (bx : AlgWorker L) {x : Msg} {w : Flock.Wit} {ν : CoinLayer.Nonce}
    {c c' : Fin (refK x) → FlockDeploy.Coin} {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    {j : Fin (jOf x)} {r : Fin 2} {wt : FlockSoundness.Model.Witness (tabC PE x j).S.m}
    (b : ZkView.Base (tabC PE x j).z (tabC PE x j).Hi) {i : ℕ}
    (hF : freeWord L bx x w ν c s j r i = freeWord L bx x w ν c' s j r i)
    (hT : ∀ h : i < (tabC PE x j).z.nE, ((tabC PE x j).T (czExe PE x c j)).tab.vals r wt b.1 b.2.1 ⟨i, h⟩ =
      ((tabC PE x j).T (czExe PE x c' j)).tab.vals r wt b.1 b.2.1 ⟨i, h⟩) :
    vsOfWord L bx x w ν c s j r wt b.1 b.2.1 i = vsOfWord L bx x w ν c' s j r wt b.1 b.2.1 i := by
  unfold vsOfWord
  rw [hF]
  rcases freeWord L bx x w ν c' s j r i with _ | a
  · dsimp only
    split_ifs with h1 h2 h2
    · exact congrArg FlockLevel3.GF128.val (hT h1)
    · exact absurd (show i < (tabC PE x j).z.nE from h1) (show ¬ i < (tabC PE x j).z.nE from h2)
    · exact absurd (show i < (tabC PE x j).z.nE from h2) (show ¬ i < (tabC PE x j).z.nE from h1)
    · rfl
  · rfl

/-- **`vsServed` at a pad other than `final_c`'s is one at two coin vectors** where `freeWord` is and the table's own
value is. -/
theorem vsServed_congr (bx : AlgWorker L) {x : Msg} {w : Flock.Wit} {ν : CoinLayer.Nonce}
    {c c' : Fin (refK x) → FlockDeploy.Coin} {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    {j : Fin (jOf x)} {r : Fin 2} {wt : FlockSoundness.Model.Witness (tabC PE x j).S.m}
    (b : ZkView.Base (tabC PE x j).z (tabC PE x j).Hi) {i : ℕ} (hi : i < (tabC PE x j).z.nE)
    (hfc : i ≠ (layE x j).fc) (hF : freeWord L bx x w ν c s j r i = freeWord L bx x w ν c' s j r i)
    (hT : ((tabC PE x j).T (czExe PE x c j)).tab.vals r wt b.1 b.2.1 ⟨i, hi⟩ =
      ((tabC PE x j).T (czExe PE x c' j)).tab.vals r wt b.1 b.2.1 ⟨i, hi⟩) :
    vsServed L bx x w ν c s j r wt b.1 b.2.1 ⟨i, hi⟩ = vsServed L bx x w ν c' s j r wt b.1 b.2.1 ⟨i, hi⟩ := by
  have e : ∀ c₀ : Fin (refK x) → FlockDeploy.Coin, vsServed L bx x w ν c₀ s j r wt b.1 b.2.1 ⟨i, hi⟩ =
      (vsOfWord L bx x w ν c₀ s j r wt b.1 b.2.1 i : FlockLevel3.GF128) := fun _ => ite_eq_right hfc
  rw [e, e, vsOfWord_congr L bx b hF fun _ => hT]

/-- **`vsServed` at `final_c`'s pad is one at two coin vectors** where the zerocheck's point and the round-1c words
are. -/
theorem vsServed_fc_congr (bx : AlgWorker L) {x : Msg} {w : Flock.Wit} {ν : CoinLayer.Nonce}
    {c c' : Fin (refK x) → FlockDeploy.Coin} {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    {j : Fin (jOf x)} {r : Fin 2} {wt : FlockSoundness.Model.Witness (tabC PE x j).S.m}
    (b : ZkView.Base (tabC PE x j).z (tabC PE x j).Hi) (hi : (layE x j).fc < (tabC PE x j).z.nE)
    (hz : (zcE c j r).z = (zcE c' j r).z)
    (hw : ∀ k < 64, vsOfWord L bx x w ν c s j r wt b.1 b.2.1 ((layE x j).r1c k) =
      vsOfWord L bx x w ν c' s j r wt b.1 b.2.1 ((layE x j).r1c k)) :
    vsServed L bx x w ν c s j r wt b.1 b.2.1 ⟨(layE x j).fc, hi⟩ =
      vsServed L bx x w ν c' s j r wt b.1 b.2.1 ⟨(layE x j).fc, hi⟩ := by
  have e : ((Array.range 64).map fun k => vsOfWord L bx x w ν c s j r wt b.1 b.2.1 ((layE x j).r1c k)) =
      (Array.range 64).map fun k => vsOfWord L bx x w ν c' s j r wt b.1 b.2.1 ((layE x j).r1c k) :=
    Array.ext (by simp) fun k h1 _ => by
      simp only [Array.getElem_map, Array.getElem_range]
      exact hw k (by simpa using h1)
  have e2 : ∀ c₀ : Fin (refK x) → FlockDeploy.Coin,
      vsServed L bx x w ν c₀ s j r wt b.1 b.2.1 ⟨(layE x j).fc, hi⟩ =
        (_root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (zcE c₀ j r).z)
          ((Array.range 64).map fun k => vsOfWord L bx x w ν c₀ s j r wt b.1 b.2.1 ((layE x j).r1c k)) :
          FlockLevel3.GF128) :=
    fun _ => ite_eq_left rfl
  rw [e2, e2, hz, e]

end Vals

/-- The zerocheck's point of rep `2j + r` reads its coins below `repBase + 3`. -/
theorem zcE_z_congr {x : Msg} {c c' : Fin (refK x) → FlockDeploy.Coin} (j : Fin (jOf x)) (r : Fin 2) {n : ℕ}
    (hcc : ∀ k : Fin (refK x), k.val < n → c k = c' k) (h : repBase x (2 * j.val + r.val) + 2 < n) :
    (zcE c j r).z = (zcE c' j r).z :=
  slotAt_congr hcc h 0

section Cut

variable {x : Msg}

/-- **The switched source's masked value** at a sent index is the worker's clear value plus the base's pad. -/
theorem swCut_masked (cz : ∀ j, Composed.CoinsZK (tabC PE x j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (v : ∀ j, ((tabC PE x j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (lv : ZkView.Leaves (fun j => (tabC PE x j).z) → ByteArray) (j : Fin (jOf x)) (r : Fin 2) (k : ℕ) :
    (repSrcCut PE x cz (swCutAt (tabC PE x) cz wt v ω lv) j r).masked k =
      if h : k < (tabC PE x j).z.nE then
        FlockLevel3.GF128.val (v j r (wt j) (ω j).1 (ω j).2.1 ⟨k, h⟩ + ((ω j).2.2.1 r).1 ⟨k, h⟩) else 0 := by
  rw [repSrcCut_masked]
  by_cases h : k < (tabC PE x j).z.nE
  · rw [dite_eq_left h, dite_eq_left h, swCutAt_masked, FlockSoundness.ZK.Table.swPre_masked_any]
    rfl
  · rw [dite_eq_right h, dite_eq_right h]

/-- **The honest sources' masked values agree only where the tables' own values do**: the pad is the base's at both. -/
theorem vals_of_masked (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m)
    (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray) {c c' : Fin (refK x) → FlockDeploy.Coin}
    (j : Fin (jOf x)) (r : Fin 2) {k : ℕ} (hk : k < (tabC PE x j).z.nE)
    (h : (repSrcCut PE x (czExe PE x c) (cutC x wt ω p c) j r).masked k =
      (repSrcCut PE x (czExe PE x c') (cutC x wt ω p c') j r).masked k) :
    ((tabC PE x j).T (czExe PE x c j)).tab.vals r (wt j) (ω j).1 (ω j).2.1 ⟨k, hk⟩ =
      ((tabC PE x j).T (czExe PE x c' j)).tab.vals r (wt j) (ω j).1 (ω j).2.1 ⟨k, hk⟩ := by
  rw [repSrcCut_masked, repSrcCut_masked, dite_eq_left hk, dite_eq_left hk] at h
  have h' : ((tabC PE x j).T (czExe PE x c j)).tab.vals r (wt j) (ω j).1 (ω j).2.1 ⟨k, hk⟩ +
        ((ω j).2.2.1 r).1 ⟨k, hk⟩ =
      ((tabC PE x j).T (czExe PE x c' j)).tab.vals r (wt j) (ω j).1 (ω j).2.1 ⟨k, hk⟩ + ((ω j).2.2.1 r).1 ⟨k, hk⟩ :=
    congrArg FlockLevel3.GF128.of h
  exact add_right_cancel h'

end Cut

/-- **The switched sources' masked value at a pad other than `final_c`'s is one at two coin vectors** where the honest
sources' is and `freeWord` is: the pad is the base's at both, and the worker's value `freeWord`'s or the table's own. -/
theorem swMasked_congr (L : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (bx : AlgWorker L) {x : Msg} {w : Flock.Wit} (hSv : Serves x) (hRo : RowsOK (vsServed L bx) x w)
    {ν : CoinLayer.Nonce}
    {c c' : Fin (refK x) → FlockDeploy.Coin} {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray)
    (lv lv' : ZkView.Leaves (fun j => (tabC PE x j).z) → ByteArray) (j : Fin (jOf x)) (r : Fin 2) {k : ℕ}
    (hfc : k ≠ (layE x j).fc)
    (hH : (repSrcCut PE x (czExe PE x c) (cutC x wt ω p c) j r).masked k =
      (repSrcCut PE x (czExe PE x c') (cutC x wt ω p c') j r).masked k)
    (hF : freeWord L bx x w ν c s j r k = freeWord L bx x w ν c' s j r k) :
    (repSrcCut PE x (czExe PE x c) (swCutAt (tabC PE x) (czExe PE x c) wt (vsOf L bx x w ν c s) ω lv) j r).masked k =
      (repSrcCut PE x (czExe PE x c') (swCutAt (tabC PE x) (czExe PE x c') wt (vsOf L bx x w ν c' s) ω lv') j
        r).masked k := by
  have e : ∀ (c₀ : Fin (refK x) → FlockDeploy.Coin) (lv₀ : ZkView.Leaves (fun j => (tabC PE x j).z) → ByteArray),
      (repSrcCut PE x (czExe PE x c₀) (swCutAt (tabC PE x) (czExe PE x c₀) wt (vsOf L bx x w ν c₀ s) ω lv₀) j
        r).masked k =
        if h : k < (tabC PE x j).z.nE then
          FlockLevel3.GF128.val (vsOf L bx x w ν c₀ s j r (wt j) (ω j).1 (ω j).2.1 ⟨k, h⟩ +
            ((ω j).2.2.1 r).1 ⟨k, h⟩) else 0 :=
    fun c₀ lv₀ => swCut_masked (czExe PE x c₀) wt (vsOf L bx x w ν c₀ s) ω lv₀ j r k
  rw [e, e]
  by_cases hk : k < (tabC PE x j).z.nE
  · rw [dite_eq_left hk, dite_eq_left hk]
    refine congrArg (fun a => FlockLevel3.GF128.val (a + ((ω j).2.2.1 r).1 ⟨k, hk⟩)) ?_
    rw [vsOf_of_serves L bx hSv hRo]
    exact vsServed_congr L bx (ω j) hk hfc hF (vals_of_masked wt ω p j r hk hH)
  · rw [dite_eq_right hk, dite_eq_right hk]

section Uniform

variable (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))

/-- **`freeWord` at any pad reads the coins below the PIOP's last round** (`freeWord_le` at `piopNR − 1`): every place
before the Ligerito values is among the PIOP's (`piopLen_full`). -/
theorem freeWord_all {x : Msg} {w : Flock.Wit} (hSv : Serves x)
    (hP : (LeanRelease.ofLoopLink L₀ src real).ProgChecks x w) (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real))
    {ν : CoinLayer.Nonce} {c c' : Fin (refK x) → FlockDeploy.Coin}
    {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z} {j : Fin (jOf x)} {r : Fin 2} (i : ℕ)
    (hcc : ∀ k : Fin (refK x), k.val < repBase x (2 * j.val + r.val) + (piopNR (stAt x j).m (stAt x j).c.kLog - 1) →
      c k = c' k) :
    freeWord _ bx x w ν c s j r i = freeWord _ bx x w ν c' s j r i := by
  have e1 : piopNR (stAt x j).m (stAt x j).c.kLog - 1 + 1 = piopNR (stAt x j).m (stAt x j).c.kLog := by
    unfold piopNR
    omega
  refine freeWord_le L₀ src real hSv hP bx (by unfold piopNR; omega) (fun P hP' _ => ?_) hcc
  rw [e1, piopLen_full]
  have hs := absorbed_size (layE x j)
  have hfa : (layE x j).fa = 128 + 2 * ((stAt x j).m - 6) := rfl
  have hlc : (layE x j).nLc = (stAt x j).c.kLog - 6 := rfl
  rw [hfa, hlc] at hs
  omega

end Uniform

/-- **`vsServed` is one at two coin vectors at every index** where every `freeWord` is, the zerocheck's point is, and
the table's own values are. -/
theorem vsServed_all (L : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (bx : AlgWorker L) {x : Msg} {w : Flock.Wit} {ν : CoinLayer.Nonce}
    {c c' : Fin (refK x) → FlockDeploy.Coin} {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    {j : Fin (jOf x)} {r : Fin 2} {wt : FlockSoundness.Model.Witness (tabC PE x j).S.m}
    (b : ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (hF : ∀ i, freeWord L bx x w ν c s j r i = freeWord L bx x w ν c' s j r i)
    (hz : (zcE c j r).z = (zcE c' j r).z)
    (hT : ∀ i : Fin (tabC PE x j).z.nE, ((tabC PE x j).T (czExe PE x c j)).tab.vals r wt b.1 b.2.1 i =
      ((tabC PE x j).T (czExe PE x c' j)).tab.vals r wt b.1 b.2.1 i) (i : Fin (tabC PE x j).z.nE) :
    vsServed L bx x w ν c s j r wt b.1 b.2.1 i = vsServed L bx x w ν c' s j r wt b.1 b.2.1 i := by
  obtain ⟨i, hi⟩ := i
  by_cases hfc : i = (layE x j).fc
  · subst hfc
    exact vsServed_fc_congr L bx b hi hz fun k _ => vsOfWord_congr L bx b (hF _) fun h => hT ⟨_, h⟩
  · exact vsServed_congr L bx b hi hfc (hF i) (hT ⟨i, hi⟩)

section Bridges

variable {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
  (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
  (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (lv lv' : ZkView.Leaves (fun j => (C j).z) → ByteArray) (j : Fin J)

theorem swCutAt_sHat : (swCutAt C cz wt v ω lv j).sHat = (cutAt C cz wt ω lv' j).sHat := by
  unfold swCutAt cutAt
  exact FlockSoundness.ZK.Table.cut_sHat_congr _ _ (ZkView.Coined.swPreWith_sHat _ _ _ _ _ _)

theorem swCutAt_sReg : (swCutAt C cz wt v ω lv j).sReg = (cutAt C cz wt ω lv' j).sReg := by
  unfold swCutAt cutAt
  simp only [FlockSoundness.ZK.Table.cut_sReg]
  exact ZkView.Coined.swPreWith_sReg _ _ _ _ _ _

theorem swCutAt_y1 : (swCutAt C cz wt v ω lv j).y1 = (cutAt C cz wt ω lv' j).y1 := by
  unfold swCutAt cutAt
  exact FlockSoundness.ZK.Table.cut_y1_congr _ _ (ZkView.Coined.swPreWith_y1 _ _ _ _ _ _)

theorem swCutAt_ybar : (swCutAt C cz wt v ω lv j).ybar = (cutAt C cz wt ω lv' j).ybar := by
  unfold swCutAt cutAt
  simp only [FlockSoundness.ZK.Table.cut_ybar]
  exact ZkView.Coined.swPreWith_ybar _ _ _ _ _ _

theorem swCutAt_lig : (swCutAt C cz wt v ω lv j).lig = (cutAt C cz wt ω lv' j).lig := by
  unfold swCutAt cutAt
  exact FlockSoundness.ZK.Table.cut_lig_congr _ _ (ZkView.Coined.swPreWith_lig _ _ _ _ _ _)

theorem swCutAt_inner : (swCutAt C cz wt v ω lv j).inner = (cutAt C cz wt ω lv' j).inner := by
  unfold swCutAt cutAt
  simp only [FlockSoundness.ZK.Table.cut_inner]
  exact ZkView.Coined.swPreWith_inner _ _ _ _ _ _

theorem swCutAt_Y2 : (swCutAt C cz wt v ω lv j).Y2 = (cutAt C cz wt ω lv' j).Y2 := by
  unfold swCutAt cutAt
  simp only [FlockSoundness.ZK.Table.cut_Y2]
  exact ZkView.Coined.swPreWith_Y2 _ _ _ _ _ _

theorem swCutAt_leavesP : (swCutAt C cz wt v ω lv j).leavesP = (cutAt C cz wt ω lv j).leavesP := rfl

/-- Where table `j`'s rows hold at its base, the switched cut's `Y₁` is the honest cut's. -/
theorem swCutAt_Y1 (hr : ((C j).T (cz j)).tab.onVals.RowsAt (wt j, v j) (((C j).T (cz j)).randOf (ω j) 0).1
      (((C j).T (cz j)).randOf (ω j) 0).2.1) :
    (swCutAt C cz wt v ω lv j).Y1 = (cutAt C cz wt ω lv' j).Y1 := by
  unfold swCutAt cutAt
  exact FlockSoundness.ZK.Table.cut_Y1_congr _ _ (ZkView.Coined.swPreWith_Y1 _ _ _ _ _ _ hr)

end Bridges

section RepBridges

variable {x : Msg} (cz : ∀ j, Composed.CoinsZK (tabC PE x j).Rest)
  (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (v : ∀ j, ((tabC PE x j).T (cz j)).tab.Vals)
  (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
  (lv lv' : ZkView.Leaves (fun j => (tabC PE x j).z) → ByteArray) (j : Fin (jOf x)) (r : Fin 2)

theorem swRep_sv : (repSrcCut PE x cz (swCutAt (tabC PE x) cz wt v ω lv) j r).sv =
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv') j r).sv :=
  repSrcCut_sv_congr PE x cz _ _ j r (swCutAt_sHat _ _ _ _ _ _ _ _) (swCutAt_sReg _ _ _ _ _ _ _ _)

theorem swRep_restart :
    (repSrcCut PE x cz (swCutAt (tabC PE x) cz wt v ω lv) j r).y1 =
        (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv') j r).y1 ∧
      (repSrcCut PE x cz (swCutAt (tabC PE x) cz wt v ω lv) j r).ybar =
        (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv') j r).ybar ∧
      (repSrcCut PE x cz (swCutAt (tabC PE x) cz wt v ω lv) j r).hiSalt =
        (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv') j r).hiSalt :=
  repSrcCut_restart_congr PE x cz _ _ j r (swCutAt_y1 _ _ _ _ _ _ _ _) (swCutAt_ybar _ _ _ _ _ _ _ _)
    (swCutAt_lig _ _ _ _ _ _ _ _)

theorem swRep_inner :
    (repSrcCut PE x cz (swCutAt (tabC PE x) cz wt v ω lv) j r).rhoIn =
        (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv') j r).rhoIn ∧
      (repSrcCut PE x cz (swCutAt (tabC PE x) cz wt v ω lv) j r).sigmaIn =
        (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv') j r).sigmaIn := by
  have h := swCutAt_inner (tabC PE x) cz wt v ω lv lv' j
  constructor <;> (simp only [repSrcCut]; rw [h])

theorem swRep_y (hr : ((tabC PE x j).T (cz j)).tab.onVals.RowsAt (wt j, v j)
      (((tabC PE x j).T (cz j)).randOf (ω j) 0).1 (((tabC PE x j).T (cz j)).randOf (ω j) 0).2.1) :
    (repSrcCut PE x cz (swCutAt (tabC PE x) cz wt v ω lv) j r).y =
      (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv') j r).y := by
  rw [repSrcCut_y, repSrcCut_y, swCutAt_Y2 (tabC PE x) cz wt v ω lv lv' j,
    swCutAt_Y1 (tabC PE x) cz wt v ω lv lv' j hr]

end RepBridges

set_option smartUnfolding false in
theorem onVals_colPAt {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (b : ZkView.Base z Hi) (r : Fin 2) (p : Fin (z.nQ + z.nPU)) :
    T.onVals.colPAt (w, v) b r p = T.colPAt w b r p :=
  rfl

/-- Where the rows hold at the base, the switched prover's pads leaves commit the honest prover's columns: they read the
base's pads alone. -/
theorem swContent_pads {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (b : ZkView.Base z Hi) (hr : T.tab.onVals.RowsAt (w, v) b.1 b.2.1) (r : Fin 2)
    (p : Fin (z.nQ + z.nPU)) : T.swContent w v b (.inr (.inl (r, p))) = T.content w b (.inr (.inl (r, p))) := by
  unfold ZkView.Coined.swContent
  rw [ite_eq_left hr]
  simp only [ZkView.Coined.content, onVals_colPAt]

/-- The switched prover's `τ` is `λ`'s batch against `μ` at its own pre-final view. -/
theorem swTau_congr {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T T' : ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (v' : T'.tab.Vals) (b : ZkView.Base z Hi) (r : Fin 2)
    (h : T.onVals.lam r (T.onVals.tab.view (w, v) (T.onVals.randOf b 0)).pre =
      T'.onVals.lam r (T'.onVals.tab.view (w, v') (T'.onVals.randOf b 0)).pre) :
    (T.onVals.tab.view (w, v) (T.onVals.randOf b 0)).tau r =
      (T'.onVals.tab.view (w, v') (T'.onVals.randOf b 0)).tau r := by
  show ∑ i, T.onVals.lam r (T.onVals.tab.view (w, v) (T.onVals.randOf b 0)).pre i * b.2.2.2.2.2.1 r i =
    ∑ i, T'.onVals.lam r (T'.onVals.tab.view (w, v') (T'.onVals.randOf b 0)).pre i * b.2.2.2.2.2.1 r i
  rw [h]

/-- A switched cut's pads leaf at tree position `q` is the leaf at `q`, whatever the coins' tree. -/
theorem padLeafCut_swCutAt (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (tabC P x j).S.m) (v : ∀ j, ((tabC P x j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi)
    (lv : ZkView.Leaves (fun j => (tabC P x j).z) → ByteArray) (j : Fin (jOf x)) (r : Fin 2) {q : ℕ}
    (hq : q < (tabC P x j).z.nQ + (tabC P x j).z.nPU) :
    padLeafCut P x cz (swCutAt (tabC P x) cz wt v ω lv) j r q = lv ⟨j, .inr (.inl (r, ⟨q, hq⟩))⟩ := by
  have e : padLeafCut P x cz (swCutAt (tabC P x) cz wt v ω lv) j r q =
      padLeafCut P x cz (cutAt (tabC P x) cz wt ω lv) j r q := by
    unfold padLeafCut
    simp only [swCutAt_leavesP (tabC P x) cz wt v ω lv j]
  rw [e]
  exact Cx.padLeafCut_cutAt P x cz wt ω lv j r hq

section Leaves

attribute [local irreducible] Composed.treeEquiv

variable {x : Msg} (cz : ∀ j, Composed.CoinsZK (tabC PE x j).Rest)
  (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (v : ∀ j, ((tabC PE x j).T (cz j)).tab.Vals)
  (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
  (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray)

/-- The switched session's level-0 leaves are the honest session's (`Coined.swContent_inl`). -/
theorem leafAt_sw (j : Fin (jOf x)) :
    leafAt PE x cz (swCutAt (tabC PE x) cz wt v ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) cz wt v ω l + (p l).1, (p l).2)) j =
      leafAt PE x cz (cutAt (tabC PE x) cz wt ω fun l =>
        ZkView.leafH (digOf (tabC PE x) cz wt ω l + (p l).1, (p l).2)) j :=
  funext fun q => by
    by_cases hq : q < (tabC PE x j).z.nS + (tabC PE x j).z.nU
    · rw [leafAt_swCutAt PE x cz wt v ω _ j hq, leafAt_cutAt PE x cz wt ω _ j hq]
      simp only [swDigOf, digOf, ZkView.Coined.swContent_inl]
    · simp only [leafAt, hq, ↓reduceDIte]

/-- **Where table `j`'s rows hold at its base, the switched session's pads leaves are the honest session's**
(`swContent_pads`). -/
theorem padLeaf_sw (j : Fin (jOf x)) (r : Fin 2)
    (hr : ((tabC PE x j).T (cz j)).tab.onVals.RowsAt (wt j, v j) (ω j).1 (ω j).2.1) :
    padLeafCut PE x cz (swCutAt (tabC PE x) cz wt v ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) cz wt v ω l + (p l).1, (p l).2)) j r =
      padLeafCut PE x cz (cutAt (tabC PE x) cz wt ω fun l =>
        ZkView.leafH (digOf (tabC PE x) cz wt ω l + (p l).1, (p l).2)) j r :=
  funext fun q => by
    by_cases hq : q < (tabC PE x j).z.nQ + (tabC PE x j).z.nPU
    · rw [padLeafCut_swCutAt PE x cz wt v ω _ j r hq, Cx.padLeafCut_cutAt PE x cz wt ω _ j r hq]
      simp only [swDigOf, digOf, swContent_pads _ _ _ _ hr]
    · simp only [padLeafCut, hq, ↓reduceDIte]

set_option smartUnfolding false in
/-- **Where table `j`'s rows hold at its base, the switched cut's `τ` commitment is the worker's view's `τ`'s leaf.** -/
theorem swCut_tauCom (j : Fin (jOf x)) (r : Fin 2)
    (hr : ((tabC PE x j).T (cz j)).tab.onVals.RowsAt (wt j, v j) (ω j).1 (ω j).2.1) :
    (swCutAt (tabC PE x) cz wt v ω (fun l =>
        ZkView.leafH (swDigOf (tabC PE x) cz wt v ω l + (p l).1, (p l).2)) j).tauCom r =
      ZkView.leafH (ZkView.bitsOf (_root_.Flock.Sha512.hash (_root_.Flock.Zk.TAU_TAG ++ _root_.Flock.F128.toBytes
        ((((tabC PE x j).T (cz j)).onVals.tab.view (wt j, v j)
          (((tabC PE x j).T (cz j)).onVals.randOf (ω j) 0)).tau r))) + (p ⟨j, .inr (.inr r)⟩).1,
        (p ⟨j, .inr (.inr r)⟩).2) := by
  have e : (swCutAt (tabC PE x) cz wt v ω (fun l =>
        ZkView.leafH (swDigOf (tabC PE x) cz wt v ω l + (p l).1, (p l).2)) j).tauCom r =
      ZkView.leafH (ZkView.bitsOf (((tabC PE x j).T (cz j)).swContent (wt j) (v j) (ω j) (.inr (.inr r))) +
        (p ⟨j, .inr (.inr r)⟩).1, (p ⟨j, .inr (.inr r)⟩).2) := rfl
  rw [e, ZkView.Coined.swContent, ite_eq_left hr]
  rfl

end Leaves

/-- **The switched view's `τ` at rep `r` reads its coins below the inner proof's `β`**, its own masked claims' points and
the worker's values at its base (`view_tau_congr` at the table at clear values). -/
theorem swView_tau_congr {x : Msg} (cz cz' : ∀ j, Composed.CoinsZK (tabC PE x j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (v : ∀ j, ((tabC PE x j).T (cz j)).tab.Vals)
    (v' : ∀ j, ((tabC PE x j).T (cz' j)).tab.Vals) (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (j : Fin (jOf x)) (r : Fin 2) (hA : Composed.Cx.InAgree (zkTab x j).table (stAt x j) (cz j) (cz' j) r)
    (hpt : ∀ k (h : k < (tabC PE x j).z.nC), (k = 2 * r.val ∨ k = 2 * r.val + 1) →
      Composed.pointsOf ((cz j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) ⟨k, h⟩ =
        Composed.pointsOf ((cz' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) ⟨k, h⟩)
    (hv : ∀ i : Fin (tabC PE x j).z.nE, v j r (wt j) (ω j).1 (ω j).2.1 i = v' j r (wt j) (ω j).1 (ω j).2.1 i) :
    (((tabC PE x j).T (cz j)).onVals.tab.view (wt j, v j) (((tabC PE x j).T (cz j)).onVals.randOf (ω j) 0)).tau r =
      (((tabC PE x j).T (cz' j)).onVals.tab.view (wt j, v' j)
        (((tabC PE x j).T (cz' j)).onVals.randOf (ω j) 0)).tau r := by
  have hrbar := Composed.Cx.rbar_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j) r hA.lane
  have hρ := Composed.Cx.ρ_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j) r hA.rhoP
  have he := Composed.Cx.embed_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  have hla := Composed.Cx.lanes_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  have hζ : ((tabC PE x j).T (cz j)).ζ r = ((tabC PE x j).T (cz' j)).ζ r :=
    Composed.Cx.ζ_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j) r hA.zeta
  have hia : ((tabC PE x j).T (cz j)).ia = ((tabC PE x j).T (cz' j)).ia :=
    Composed.Cx.ia_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  have hib : ((tabC PE x j).T (cz j)).ib = ((tabC PE x j).T (cz' j)).ib :=
    Composed.Cx.ib_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  refine swTau_congr ((tabC PE x j).T (cz j)) ((tabC PE x j).T (cz' j)) (wt j) (v j) (v' j) (ω j) r
    (Composed.Cx.lamOf_congr (zkTab x j).table (stAt x j) (cz j) (cz' j) r _ _ hA ?_ ?_ ?_ ?_ ?_)
  · funext i
    show v j r (wt j) (ω j).1 (ω j).2.1 i + ((ω j).2.2.1 r).1 i = v' j r (wt j) (ω j).1 (ω j).2.1 i + ((ω j).2.2.1 r).1 i
    exact congrArg (fun a => a + ((ω j).2.2.1 r).1 i) (hv i)
  · unfold Composed.sHatAt
    split_ifs with h
    · exact Composed.Cx.coined_val_congr ((tabC PE x j).T (cz j)) ((tabC PE x j).T (cz' j)) (wt j)
        ⟨2 * r.val, h⟩ (ω j).1 (Composed.Cx.nu_congr PE _ _ _ _ (wt j) ⟨2 * r.val, h⟩ (hpt _ h (.inl rfl)))
        (Composed.Cx.eqw_congr PE _ _ _ _ ⟨2 * r.val, h⟩ (hpt _ h (.inl rfl)))
    · rfl
  · unfold Composed.sHatAt
    split_ifs with h
    · exact Composed.Cx.coined_val_congr ((tabC PE x j).T (cz j)) ((tabC PE x j).T (cz' j)) (wt j)
        ⟨2 * r.val + 1, h⟩ (ω j).1 (Composed.Cx.nu_congr PE _ _ _ _ (wt j) ⟨2 * r.val + 1, h⟩ (hpt _ h (.inr rfl)))
        (Composed.Cx.eqw_congr PE _ _ _ _ ⟨2 * r.val + 1, h⟩ (hpt _ h (.inr rfl)))
    · rfl
  · show ∑ col : Fin (tabC PE x j).z.L, ((tabC PE x j).T (cz j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r col *
        Composed.bFold (zkTab x j).table (stAt x j) (cz j) r col.val =
      ∑ col : Fin (tabC PE x j).z.L, ((tabC PE x j).T (cz' j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r col *
        Composed.bFold (zkTab x j).table (stAt x j) (cz' j) r col.val
    rw [y1'_congr (tabC PE x) cz wt ω j r cz' hrbar hρ he hla,
      Composed.Cx.bFold_congr _ _ _ _ r hA.toB0Agree hA.lane]
  · show (((tabC PE x j).T (cz j)).ζ r * ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz j)).ia + ((ω j).2.2.1 r).2.1,
        ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz j)).ib + ((ω j).2.2.1 r).2.2) =
      (((tabC PE x j).T (cz' j)).ζ r * ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz' j)).ia + ((ω j).2.2.1 r).2.1,
        ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz' j)).ib + ((ω j).2.2.1 r).2.2)
    rw [hζ, hia, hib]

/-- **Where table `j`'s rows hold at its base at both coin vectors, the switched sources' `τ` commitments agree** where
the switched views' `τ` do (`swCut_tauCom`). -/
theorem swRep_tauCom_congr {x : Msg} (cz cz' : ∀ j, Composed.CoinsZK (tabC PE x j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (v : ∀ j, ((tabC PE x j).T (cz j)).tab.Vals)
    (v' : ∀ j, ((tabC PE x j).T (cz' j)).tab.Vals) (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray) (j : Fin (jOf x)) (r : Fin 2)
    (hr : ((tabC PE x j).T (cz j)).tab.onVals.RowsAt (wt j, v j) (ω j).1 (ω j).2.1)
    (hr' : ((tabC PE x j).T (cz' j)).tab.onVals.RowsAt (wt j, v' j) (ω j).1 (ω j).2.1)
    (htau : (((tabC PE x j).T (cz j)).onVals.tab.view (wt j, v j)
        (((tabC PE x j).T (cz j)).onVals.randOf (ω j) 0)).tau r =
      (((tabC PE x j).T (cz' j)).onVals.tab.view (wt j, v' j)
        (((tabC PE x j).T (cz' j)).onVals.randOf (ω j) 0)).tau r) :
    (repSrcCut PE x cz (swCutAt (tabC PE x) cz wt v ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) cz wt v ω l + (p l).1, (p l).2)) j r).tauCom =
      (repSrcCut PE x cz' (swCutAt (tabC PE x) cz' wt v' ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) cz' wt v' ω l + (p l).1, (p l).2)) j r).tauCom :=
  (swCut_tauCom cz wt v ω p j r hr).trans
    ((congrArg (fun t => ZkView.leafH (ZkView.bitsOf (_root_.Flock.Sha512.hash (_root_.Flock.Zk.TAU_TAG ++
      _root_.Flock.F128.toBytes t)) + (p ⟨j, .inr (.inr r)⟩).1, (p ⟨j, .inr (.inr r)⟩).2)) htau).trans
      (swCut_tauCom cz' wt v' ω p j r hr').symm)

section TauAgree

variable {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
  (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
  {c c' : Fin (refK x) → FlockDeploy.Coin} {n : ℕ} (hcc : ∀ k : Fin (refK x), k.val < n → c k = c' k)

include hsx hcc in
/-- **What rep `r`'s `τ` reads below its commitment's body agrees**: the inner proof's coins and its masked claims'
points (`repOk`'s `tc` case). -/
theorem inAgree_tc {e : Array _root_.Flock.Claim} (he : e.size = (stAt x j).regions.size * spec.points)
    (hn : FlockComplete.CutEq.innerAt (FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val))
      (_root_.Flock.Setup.ofCircuit (stAt x j)) (zkTab x j).sch e + 2 ≤ n) :
    Composed.Cx.InAgree (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r ∧
      ∀ k (h : k < (tabC PE x j).z.nC), (k = 2 * r.val ∨ k = 2 * r.val + 1) →
        Composed.pointsOf ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) ⟨k, h⟩ =
          Composed.pointsOf ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) ⟨k, h⟩ := by
  have hp2 := FlockComplete.ZkTables.points_of_setup (live_of_lt j).1 hsx
  have hm14 : 14 ≤ (stAt x j).m := (schOk_of (zkTab x j)).m
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hbase := repBase_eq hsx (2 * j.val + r.val)
  have hout := outerSlot_at hsx j r
  have hlig : FlockComplete.CutEq.ligAt (FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val))
      (_root_.Flock.Setup.ofCircuit (stAt x j)) e = FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
        FlockComplete.piopRoundsZk spec (stAt x j) := by
    unfold FlockComplete.CutEq.ligAt FlockComplete.piopRoundsZk
    rw [he, hp2]
    show _ + 3 + ((stAt x j).m - 6) + (3 + ((stAt x j).c.kLog - 6)) + _ + 1 = _
    omega
  simp only [FlockComplete.CutEq.innerAt] at hn
  generalize hB : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) = B at hbase hout hlig hn
  have hpiop : FlockComplete.piopRoundsZk spec (stAt x j) =
      (stAt x j).m + (stAt x j).c.kLog - 6 + (2 + 2 * (stAt x j).regions.size + 1) := by
    unfold FlockComplete.piopRoundsZk
    rw [hp2]
  have ek0 : (zkTab x j).table.sch.k0 = (zkTab x j).sch.k0 := rfl
  have hg : gateAt spec x j r .opening ≤ n := by simp only [gateAt]; rw [hB]; omega
  have hl : ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r :=
    funext fun a => lane_agree_lt hsx j r hcc (i := (zkTab x j).sch.k0) (by omega) a a.isLt
  have h0 : Composed.Cx.B0Agree (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r :=
    ⟨link_agree j hcc (by omega), rOuter_agree j r hcc (by omega), rho_agree j r hcc (by omega),
      ts_agree j r hcc (by omega), rpp_agree j r hcc (by omega), rsGamma_agree hsx j r hcc (by omega),
      oodPt_agree hsx j r hcc (by omega), ood_agree hsx j r hcc (by omega)⟩
  refine ⟨{ toB0Agree := h0
            zc := zc_agree j r hcc (by omega)
            lcAlpha := lcAlpha_agree j r hcc (by omega)
            lcBeta := lcBeta_agree j r hcc (by omega)
            rPrime := rPrime_agree j r hcc (by omega)
            lane := hl
            rhoP := rhoPair_agree hsx j r hcc (by omega)
            zeta := zeta_agree hsx j r hcc (by simp only [innerOff]; omega)
            gammas := gammas_agree hsx j r hcc (by simp only [innerOff]; omega) }, fun k hk hk2 => ?_⟩
  refine points_agree hsx j r hcc ⟨k, hk⟩ ?_ hg
  rcases hk2 with rfl | rfl
  · exact le_of_eq (rep_two r (k := 0) (by omega))
  · exact le_of_eq (rep_two r (k := 1) (by omega))

end TauAgree

/-- **The switched sources' masked value at `final_c`'s pad is one at two coin vectors** where the zerocheck's point is
and, at each round-1c pad, the honest sources' value and `freeWord` are. -/
theorem swMasked_fc_congr (L : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (bx : AlgWorker L) {x : Msg} {w : Flock.Wit} (hSv : Serves x) (hRo : RowsOK (vsServed L bx) x w)
    {ν : CoinLayer.Nonce}
    {c c' : Fin (refK x) → FlockDeploy.Coin} {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray)
    (lv lv' : ZkView.Leaves (fun j => (tabC PE x j).z) → ByteArray) (j : Fin (jOf x)) (r : Fin 2)
    (hz : (zcE c j r).z = (zcE c' j r).z)
    (hH : ∀ k < 64, (repSrcCut PE x (czExe PE x c) (cutC x wt ω p c) j r).masked ((layE x j).r1c k) =
      (repSrcCut PE x (czExe PE x c') (cutC x wt ω p c') j r).masked ((layE x j).r1c k))
    (hF : ∀ k < 64, freeWord L bx x w ν c s j r ((layE x j).r1c k) = freeWord L bx x w ν c' s j r ((layE x j).r1c k)) :
    (repSrcCut PE x (czExe PE x c) (swCutAt (tabC PE x) (czExe PE x c) wt (vsOf L bx x w ν c s) ω lv) j r).masked
        (layE x j).fc =
      (repSrcCut PE x (czExe PE x c') (swCutAt (tabC PE x) (czExe PE x c') wt (vsOf L bx x w ν c' s) ω lv') j
        r).masked (layE x j).fc := by
  have e : ∀ (c₀ : Fin (refK x) → FlockDeploy.Coin) (lv₀ : ZkView.Leaves (fun j => (tabC PE x j).z) → ByteArray),
      (repSrcCut PE x (czExe PE x c₀) (swCutAt (tabC PE x) (czExe PE x c₀) wt (vsOf L bx x w ν c₀ s) ω lv₀) j
        r).masked (layE x j).fc =
        if h : (layE x j).fc < (tabC PE x j).z.nE then
          FlockLevel3.GF128.val (vsOf L bx x w ν c₀ s j r (wt j) (ω j).1 (ω j).2.1 ⟨(layE x j).fc, h⟩ +
            ((ω j).2.2.1 r).1 ⟨(layE x j).fc, h⟩) else 0 :=
    fun c₀ lv₀ => swCut_masked (czExe PE x c₀) wt (vsOf L bx x w ν c₀ s) ω lv₀ j r (layE x j).fc
  rw [e, e]
  by_cases hk : (layE x j).fc < (tabC PE x j).z.nE
  · rw [dite_eq_left hk, dite_eq_left hk]
    refine congrArg (fun a => FlockLevel3.GF128.val (a + ((ω j).2.2.1 r).1 ⟨(layE x j).fc, hk⟩)) ?_
    rw [vsOf_of_serves L bx hSv hRo]
    exact vsServed_fc_congr L bx (ω j) hk hz fun k hk' =>
      vsOfWord_congr L bx (ω j) (hF k hk') fun h => vals_of_masked wt ω p j r h (hH k hk')
  · rw [dite_eq_right hk, dite_eq_right hk]

theorem ofCircuit_m (st : Stmt) : (_root_.Flock.Setup.ofCircuit st).m = st.m := rfl
theorem ofCircuit_kLog (st : Stmt) : (_root_.Flock.Setup.ofCircuit st).kLog = st.c.kLog := rfl
theorem ofCircuit_fold (st : Stmt) : (_root_.Flock.Setup.ofCircuit st).fold.kLog = st.c.kLog := rfl

/-! A pad's place in the absorbed order (`place_hi`'s) is below the values of the PIOP round that frees it: round 2
for round 1's pads, residual round `a` for its pair, `final_a`/`final_b`'s round, lincheck round `a`, `z_partial`'s. -/

theorem place_r1 {m k i : ℕ} (h : i < 128) :
    (if i < 130 + 2 * (m - 6) then i else i - 1) < piopLen m k (2 + 1) := by
  unfold piopLen
  split_ifs <;> omega

theorem place_ml {m k i a : ℕ} (ha : a < m - 6) (h : i < 130 + 2 * a) :
    (if i < 130 + 2 * (m - 6) then i else i - 1) < piopLen m k (3 + a + 1) := by
  unfold piopLen
  split_ifs <;> omega

theorem place_f {m k i : ℕ} (h : i < 130 + 2 * (m - 6)) :
    (if i < 130 + 2 * (m - 6) then i else i - 1) < piopLen m k (3 + (m - 6) + 1) := by
  unfold piopLen
  split_ifs <;> omega

theorem place_lc {m k i a : ℕ} (ha : a < k - 6) (h : i ≤ 132 + 2 * (m - 6) + 2 * a) :
    (if i < 130 + 2 * (m - 6) then i else i - 1) < piopLen m k (5 + (m - 6) + a + 1) := by
  unfold piopLen
  split_ifs <;> omega

theorem place_zp {m k i : ℕ} (h : i ≤ 194 + 2 * (m - 6) + 2 * (k - 6)) :
    (if i < 130 + 2 * (m - 6) then i else i - 1) < piopLen m k (5 + (m - 6) + (k - 6) + 1) := by
  unfold piopLen
  split_ifs <;> omega

section SwRep

variable (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))

/-- `freeWord_le` at a pad whose places in the absorbed order are below PIOP round `jj`'s values (`place_hi`). -/
theorem freeWord_at {x : Msg} {w : Flock.Wit} (hSv : Serves x)
    (hP : (LeanRelease.ofLoopLink L₀ src real).ProgChecks x w) (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real))
    {ν : CoinLayer.Nonce} {c c' : Fin (refK x) → FlockDeploy.Coin}
    {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z} {j : Fin (jOf x)} {r : Fin 2}
    {i : ℕ} (jj : ℕ) (hjj : jj < piopNR (stAt x j).m (stAt x j).c.kLog)
    (hi : (if i < 130 + 2 * ((stAt x j).m - 6) then i else i - 1) <
      piopLen (stAt x j).m (stAt x j).c.kLog (jj + 1))
    (hcc : ∀ k : Fin (refK x), k.val < repBase x (2 * j.val + r.val) + jj → c k = c' k) :
    freeWord _ bx x w ν c s j r i = freeWord _ bx x w ν c' s j r i := by
  have e : (layE x j).fa + 2 = 130 + 2 * ((stAt x j).m - 6) := by
    show 128 + 2 * ((stAt x j).m - 6) + 2 = _
    omega
  exact freeWord_le L₀ src real hSv hP bx hjj (place_hi (layE x j) (by rw [e]; exact hi)) hcc

/-- `freeWord_lig` at a pad past the PIOP's values (`place_lig`). -/
theorem freeWord_ligAt {x : Msg} {w : Flock.Wit} (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real))
    {ν : CoinLayer.Nonce} {c : Fin (refK x) → FlockDeploy.Coin}
    {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z} {j : Fin (jOf x)} {r : Fin 2} {i : ℕ}
    (hi : 195 + 2 * ((stAt x j).m - 6) + 2 * ((stAt x j).c.kLog - 6) ≤ i) :
    freeWord _ bx x w ν c s j r i = none := by
  refine freeWord_lig L₀ src real bx (place_lig (layE x j) ?_)
  have hs := absorbed_size (layE x j)
  have hfa : (layE x j).fa = 128 + 2 * ((stAt x j).m - 6) := rfl
  have hlc : (layE x j).nLc = (stAt x j).c.kLog - 6 := rfl
  rw [hfa, hlc] at hs
  omega

/-- **`vsOf` is the honest values from level 0's pads on**: no round frees a pad past the PIOP's values
(`freeWord_ligAt`), so where `vsOf` is `vsServed` it takes the table's own value there, as the honest values do, and
elsewhere `vsOf` is the honest values. -/
theorem vsWord_vsOf_l0 (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real)) (x : Msg) (w : Flock.Wit)
    (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) {i : ℕ}
    (hi : (layE x j).zp 64 ≤ i) :
    vsWord (vsOf _ bx) x w ν c τ j r i =
      vsWord (CoinLayer.session loopNr Masked.dataE).honestVals x w ν c τ j r i := by
  by_cases hg : Serves x ∧ RowsOK (vsServed _ bx) x w
  · have hzp : (layE x j).zp 64 = 131 + 2 * ((stAt x j).m - 6) + 2 * ((stAt x j).c.kLog - 6) + 64 :=
      pl_zp _ _ _ 64
    have hfc : (layE x j).fc = 130 + 2 * ((stAt x j).m - 6) := pl_fc _ _ _
    have hne : i ≠ (layE x j).fc := by omega
    have hl : 195 + 2 * ((stAt x j).m - 6) + 2 * ((stAt x j).c.kLog - 6) ≤ i := by omega
    by_cases hE : i < (tabC PE x j).z.nE
    · have hE' : i < (((CoinLayer.session loopNr Masked.dataE).C x j).T
          ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.nE := hE
      rw [vsWord_vsOf _ bx hg.1 w hg.2 ν c τ j r hE, ite_eq_right hne]
      refine ((vsOfWord_of_none _ bx _ _ _ (freeWord_ligAt L₀ src real bx hl)).trans (dite_eq_left hE')).trans ?_
      unfold vsWord
      rw [dite_eq_left hE]
      rfl
    · unfold vsWord
      rw [dite_eq_right hE, dite_eq_right hE]
  · unfold vsWord
    rw [vsOf_of_not _ bx hg]

/-- **`vsOf`'s level 0 is the ZK server's where the honest values' is**: `L0Fixed` reads `vsWord` only from `zp 64` on,
where `vsOf` is the honest values (`vsWord_vsOf_l0`). -/
theorem l0Fixed_vsOf_of (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real)) {x : Msg} {w : Flock.Wit}
    (h : L0Fixed (CoinLayer.session loopNr Masked.dataE).honestVals x w) : L0Fixed (vsOf _ bx) x w :=
  fun ν τ c n hR s spec hsx j r extras he k hk hk' =>
    (vsWord_vsOf_l0 L₀ src real bx x w ν c τ j r hk).trans (h ν τ c n hR s spec hsx j r extras he k hk hk')

/-- **`vsOf`'s level 0 is the ZK server's at every worker**: at every live statement and witness of the masked relation,
`L0Fixed` holds at the honest values (`l0Fixed_honestVals`), so at `vsOf bx` (`l0Fixed_vsOf_of`). -/
theorem l0Fixed_vsOf (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real)) (x : Msg) (w : Flock.Wit)
    (hx : Masked.Live x) (hw : Masked.RZk x w) : L0Fixed (vsOf _ bx) x w :=
  l0Fixed_vsOf_of L₀ src real bx (l0Fixed_honestVals x w hx hw)

/-- **The switched sources agree where read**, at coins agreeing below `n`, where table `j`'s rows hold at its base at
both: each field the honest sources' (`repOk`) or the worker's values along coins that agree up to the PIOP round that
frees each (`freeWord_at`, `freeWord_ligAt`); `τ`'s commitment the switched views' `τ` (`swView_tau_congr`). -/
theorem swRepAgree {x : Msg} {w : Flock.Wit} (hSv : Serves x)
    (hP : (LeanRelease.ofLoopLink L₀ src real).ProgChecks x w) (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real))
    (hRo : RowsOK (vsServed _ bx) x w) {ν : CoinLayer.Nonce} {sn : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray)
    {c c' : Fin (refK x) → FlockDeploy.Coin} {n : ℕ} (hcc : ∀ k : Fin (refK x), k.val < n → c k = c' k)
    {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (j : Fin (jOf x)) (r : Fin 2) {e : Array _root_.Flock.Claim} (he : e.size = (stAt x j).regions.size * spec.points)
    (hr : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wt j, vsOf _ bx x w ν c sn j) (ω j).1 (ω j).2.1)
    (hr' : ((tabC PE x j).T (czExe PE x c' j)).tab.onVals.RowsAt (wt j, vsOf _ bx x w ν c' sn j) (ω j).1 (ω j).2.1) :
    FlockComplete.CutEq.RepAgree n (FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val))
      (_root_.Flock.Setup.ofCircuit (stAt x j))
      (zkTab x j).sch e
      (repSrcCut PE x (czExe PE x c) (swCutAt (tabC PE x) (czExe PE x c) wt (vsOf _ bx x w ν c sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c) wt (vsOf _ bx x w ν c sn) ω l + (p l).1, (p l).2)) j r)
      (repSrcCut PE x (czExe PE x c') (swCutAt (tabC PE x) (czExe PE x c') wt (vsOf _ bx x w ν c' sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c') wt (vsOf _ bx x w ν c' sn) ω l + (p l).1, (p l).2)) j r) := by
  have hA := repOk x (live_of_lt j) wt ω p c c' n hcc s spec hsx j r e he
  have hm14 : 14 ≤ (stAt x j).m := (schOk_of (zkTab x j)).m
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hbase := repBase_eq hsx (2 * j.val + r.val)
  have em := ofCircuit_m (stAt x j)
  have ef := ofCircuit_fold (stAt x j)
  have ek := ofCircuit_kLog (stAt x j)
  have hM : ∀ {k : ℕ}, k ≠ (layE x j).fc →
      (repSrcCut PE x (czExe PE x c) (cutC x wt ω p c) j r).masked k =
        (repSrcCut PE x (czExe PE x c') (cutC x wt ω p c') j r).masked k →
      freeWord _ bx x w ν c sn j r k = freeWord _ bx x w ν c' sn j r k →
      (repSrcCut PE x (czExe PE x c) (swCutAt (tabC PE x) (czExe PE x c) wt (vsOf _ bx x w ν c sn) ω fun l =>
          ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c) wt (vsOf _ bx x w ν c sn) ω l + (p l).1, (p l).2)) j
          r).masked k =
        (repSrcCut PE x (czExe PE x c') (swCutAt (tabC PE x) (czExe PE x c') wt (vsOf _ bx x w ν c' sn) ω fun l =>
          ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c') wt (vsOf _ bx x w ν c' sn) ω l + (p l).1, (p l).2)) j
          r).masked k :=
    fun hfc hH hF => swMasked_congr _ bx hSv hRo wt ω p _ _ j r hfc hH hF
  have hL : ∀ {k : ℕ}, 195 + 2 * ((stAt x j).m - 6) + 2 * ((stAt x j).c.kLog - 6) ≤ k →
      freeWord _ bx x w ν c sn j r k = freeWord _ bx x w ν c' sn j r k := fun h =>
    (freeWord_ligAt L₀ src real bx h).trans (freeWord_ligAt L₀ src real bx h).symm
  have bsv : ∀ c₀ : Fin (refK x) → FlockDeploy.Coin,
      (repSrcCut PE x (czExe PE x c₀) (swCutAt (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω l + (p l).1, (p l).2)) j
          r).sv =
        (repSrcCut PE x (czExe PE x c₀) (cutC x wt ω p c₀) j r).sv := fun _ => swRep_sv _ _ _ _ _ _ _ _
  have brs : ∀ c₀ : Fin (refK x) → FlockDeploy.Coin,
      (repSrcCut PE x (czExe PE x c₀) (swCutAt (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω l + (p l).1, (p l).2)) j
          r).y1 = (repSrcCut PE x (czExe PE x c₀) (cutC x wt ω p c₀) j r).y1 ∧
      (repSrcCut PE x (czExe PE x c₀) (swCutAt (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω l + (p l).1, (p l).2)) j
          r).ybar = (repSrcCut PE x (czExe PE x c₀) (cutC x wt ω p c₀) j r).ybar ∧
      (repSrcCut PE x (czExe PE x c₀) (swCutAt (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω l + (p l).1, (p l).2)) j
          r).hiSalt = (repSrcCut PE x (czExe PE x c₀) (cutC x wt ω p c₀) j r).hiSalt :=
    fun _ => swRep_restart _ _ _ _ _ _ _ _
  have bin : ∀ c₀ : Fin (refK x) → FlockDeploy.Coin,
      (repSrcCut PE x (czExe PE x c₀) (swCutAt (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω l + (p l).1, (p l).2)) j
          r).rhoIn = (repSrcCut PE x (czExe PE x c₀) (cutC x wt ω p c₀) j r).rhoIn ∧
      (repSrcCut PE x (czExe PE x c₀) (swCutAt (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω l + (p l).1, (p l).2)) j
          r).sigmaIn = (repSrcCut PE x (czExe PE x c₀) (cutC x wt ω p c₀) j r).sigmaIn :=
    fun _ => swRep_inner _ _ _ _ _ _ _ _
  have byy : ∀ c₀ : Fin (refK x) → FlockDeploy.Coin,
      ((tabC PE x j).T (czExe PE x c₀ j)).tab.onVals.RowsAt (wt j, vsOf _ bx x w ν c₀ sn j) (ω j).1 (ω j).2.1 →
      (repSrcCut PE x (czExe PE x c₀) (swCutAt (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω l + (p l).1, (p l).2)) j
          r).y = (repSrcCut PE x (czExe PE x c₀) (cutC x wt ω p c₀) j r).y :=
    fun _ h => swRep_y _ _ _ _ _ _ _ _ h
  refine ⟨fun hn i hi => ?_, fun i hi hn => ?_, fun hn => ?_, fun i hi hn => ?_, fun hn i hi => ?_,
    fun k hk hn => ?_, fun hn => ?_, fun i hi hn t ht => ?_, fun hn a ha t ht => ?_, fun hn => ?_, fun hn => ?_,
    fun hn => ?_, fun hn => ?_, fun _ => ?_⟩
  · obtain ⟨h1, h2⟩ := hA.ab hn i hi
    refine ⟨hM ?_ h1 (freeWord_at L₀ src real hSv hP bx 2 ?_ (place_r1 ?_) fun k hk => hcc k ?_),
      hM ?_ h2 (freeWord_at L₀ src real hSv hP bx 2 ?_ (place_r1 ?_) fun k hk => hcc k ?_)⟩
    all_goals (try simp only [FlockComplete.CutEq.layOf, layE, pl_r1ab, pl_r1c, pl_fc, ofCircuit_m,
      ofCircuit_kLog, ofCircuit_fold, piopNR] at *); omega
  · obtain ⟨h1, h2⟩ := hA.ml i hi hn
    rw [em] at hi
    refine ⟨hM ?_ h1 (freeWord_at L₀ src real hSv hP bx (3 + i) ?_ (place_ml hi ?_) fun k hk => hcc k ?_),
      hM ?_ h2 (freeWord_at L₀ src real hSv hP bx (3 + i) ?_ (place_ml hi ?_) fun k hk => hcc k ?_)⟩
    all_goals (try simp only [FlockComplete.CutEq.layOf, layE, pl_ml, pl_fc, ofCircuit_m, ofCircuit_kLog,
      ofCircuit_fold, piopNR] at *); omega
  · obtain ⟨h1, h2, -⟩ := hA.f hn
    rw [em] at hn
    refine ⟨hM ?_ h1 (freeWord_at L₀ src real hSv hP bx (3 + ((stAt x j).m - 6)) ?_ (place_f ?_) fun k hk => hcc k ?_),
      hM ?_ h2 (freeWord_at L₀ src real hSv hP bx (3 + ((stAt x j).m - 6)) ?_ (place_f ?_) fun k hk => hcc k ?_),
      swMasked_fc_congr _ bx hSv hRo wt ω p _ _ j r (zcE_z_congr j r hcc ?_) (fun k hk => (hA.ab ?_ k hk).2)
        fun k hk => freeWord_at L₀ src real hSv hP bx 2 ?_ (place_r1 ?_) fun k hk => hcc k ?_⟩
    all_goals (try simp only [FlockComplete.CutEq.layOf, layE, pl_r1c, pl_fa, pl_fb, pl_fc,
      ofCircuit_m, ofCircuit_kLog, ofCircuit_fold, piopNR] at *); omega
  · obtain ⟨h1, h2⟩ := hA.lc i hi hn
    rw [em] at hn
    rw [ef] at hi
    refine ⟨hM ?_ h1 (freeWord_at L₀ src real hSv hP bx (5 + ((stAt x j).m - 6) + i) ?_ (place_lc hi ?_)
        fun k hk => hcc k ?_),
      hM ?_ h2 (freeWord_at L₀ src real hSv hP bx (5 + ((stAt x j).m - 6) + i) ?_ (place_lc hi ?_)
        fun k hk => hcc k ?_)⟩
    all_goals (try simp only [FlockComplete.CutEq.layOf, layE, pl_fc, pl_lc, ofCircuit_m, ofCircuit_kLog,
      ofCircuit_fold, piopNR] at *); omega
  · have h1 := hA.zp hn i hi
    rw [em, ef] at hn
    refine hM ?_ h1 (freeWord_at L₀ src real hSv hP bx (5 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6)) ?_
      (place_zp ?_) fun k hk => hcc k ?_)
    all_goals (try simp only [FlockComplete.CutEq.layOf, layE, pl_fc, pl_zp, ofCircuit_m, ofCircuit_kLog,
      ofCircuit_fold, piopNR] at *); omega
  · rw [bsv c, bsv c']
    exact hA.sv k hk hn
  · refine hM ?_ (hA.ood hn) (hL ?_)
    all_goals (try simp only [FlockComplete.CutEq.layOf, layE, pl_fc, pl_lgOod, ofCircuit_m, ofCircuit_kLog,
      ofCircuit_fold] at *); omega
  · refine hM ?_ (hA.msg i hi hn t ht) (hL ?_)
    all_goals (try simp only [FlockComplete.CutEq.layOf, layE, pl_fc, pl_lgMsg, ofCircuit_m, ofCircuit_kLog,
      ofCircuit_fold] at *); omega
  · refine hM ?_ (hA.E hn a ha t ht) (hL ?_)
    all_goals (try simp only [FlockComplete.CutEq.layOf, layE, pl_fc, pl_lgE, ofCircuit_m, ofCircuit_kLog,
      ofCircuit_fold] at *); omega
  · obtain ⟨h1, h2, h3⟩ := hA.y1 hn
    obtain ⟨b1, b2, b3⟩ := brs c
    obtain ⟨b1', b2', b3'⟩ := brs c'
    exact ⟨b1.trans (h1.trans b1'.symm), b2.trans (h2.trans b2'.symm), b3.trans (h3.trans b3'.symm)⟩
  · obtain ⟨h1, h2⟩ := hA.rs hn
    obtain ⟨b1, b2⟩ := bin c
    obtain ⟨b1', b2'⟩ := bin c'
    exact ⟨b1.trans (h1.trans b1'.symm), b2.trans (h2.trans b2'.symm)⟩
  · obtain ⟨hIA, hpt⟩ := inAgree_tc hsx j r hcc he hn
    simp only [FlockComplete.CutEq.innerAt, FlockComplete.CutEq.ligAt, em, ef] at hn
    have hT : ∀ i : Fin (tabC PE x j).z.nE,
        ((tabC PE x j).T (czExe PE x c j)).tab.vals r (wt j) (ω j).1 (ω j).2.1 i =
          ((tabC PE x j).T (czExe PE x c' j)).tab.vals r (wt j) (ω j).1 (ω j).2.1 i := fun i => by
      show Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c j) r
          (Composed.zWith (stAt x j) (wt j) (ω j).1) ((ω j).2.1.2.1 r) i.val =
        Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c' j) r
          (Composed.zWith (stAt x j) (wt j) (ω j).1) ((ω j).2.1.2.1 r) i.val
      rw [Composed.Cx.honestAt_all _ _ _ _ r _ _ hIA.toB0Agree hIA.zc hIA.lcAlpha hIA.lcBeta hIA.lane]
    have hv : ∀ i : Fin (tabC PE x j).z.nE, vsOf _ bx x w ν c sn j r (wt j) (ω j).1 (ω j).2.1 i =
        vsOf _ bx x w ν c' sn j r (wt j) (ω j).1 (ω j).2.1 i := fun i => by
      rw [vsOf_of_serves _ bx hSv hRo]
      exact vsServed_all _ bx (ω j)
        (fun i => freeWord_all L₀ src real hSv hP bx i fun k hk => hcc k (by unfold piopNR at hk; omega))
        (zcE_z_congr j r hcc (by omega)) hT i
    exact swRep_tauCom_congr _ _ wt _ _ ω p j r hr hr' (swView_tau_congr _ _ wt _ _ ω j r hIA hpt hv)
  · rw [byy c hr, byy c' hr']
    exact hA.y hn
  · exact ⟨(repSrcCut_tau x _ j r _).1.trans (repSrcCut_tau x _ j r _).1.symm,
      (repSrcCut_tau x _ j r _).2.trans (repSrcCut_tau x _ j r _).2.symm⟩

/-- **Where every table's rows hold at its base at both coin vectors, the switched round encodings agree** at coins
agreeing below the round (`encExe_congr` at the switched cuts: `leafAt_sw`, `padLeaf_sw`, `swRepAgree`). -/
theorem swEncExe_congr {x : Msg} {w : Flock.Wit} (hSv : Serves x)
    (hP : (LeanRelease.ofLoopLink L₀ src real).ProgChecks x w) (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real))
    (hRo : RowsOK (vsServed _ bx) x w) {ν : CoinLayer.Nonce} {sn : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray)
    {c c' : Fin (refK x) → FlockDeploy.Coin} {n : ℕ} (hcc : ∀ k : Fin (refK x), k.val < n → c k = c' k)
    (hr : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wt j, vsOf _ bx x w ν c sn j) (ω j).1 (ω j).2.1)
    (hr' : ∀ j,
      ((tabC PE x j).T (czExe PE x c' j)).tab.onVals.RowsAt (wt j, vsOf _ bx x w ν c' sn j) (ω j).1 (ω j).2.1) :
    encExe PE x c n (swCutAt (tabC PE x) (czExe PE x c) wt (vsOf _ bx x w ν c sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c) wt (vsOf _ bx x w ν c sn) ω l + (p l).1, (p l).2)) =
      encExe PE x c' n (swCutAt (tabC PE x) (czExe PE x c') wt (vsOf _ bx x w ν c' sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c') wt (vsOf _ bx x w ν c' sn) ω l + (p l).1, (p l).2)) :=
  encExe_congr (fun c₀ => swCutAt (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω fun l =>
      ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c₀) wt (vsOf _ bx x w ν c₀ sn) ω l + (p l).1, (p l).2)) hcc
    (fun j => (leafAt_sw _ wt _ ω p j).trans ((leaf0_agree x wt ω p c c' j).trans (leafAt_sw _ wt _ ω p j).symm))
    (fun j r => (padLeaf_sw _ wt _ ω p j r (hr j)).trans
      ((padLeaf_agree x wt ω p c c' j r).trans (padLeaf_sw _ wt _ ω p j r (hr' j)).symm))
    fun _ _ _ hsx j r _ he => swRepAgree L₀ src real hSv hP bx hRo wt ω p hcc hsx j r he (hr j) (hr' j)

/-- **The switched session's round encoding at `vsOf` reads the coins before the round**, at coins where, if `vsOf` is
`vsServed`, every table's rows hold at its base: `swEncExe_congr` stopped where a refusal is seen (`refused_congr`);
elsewhere `vsOf` is the honest values and the encoding the honest one's (`causalExe`, `causal_encStop`). -/
theorem swEnc_congr (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real)) {x : Msg} {w : Flock.Wit}
    (hP : Serves x → (LeanRelease.ofLoopLink L₀ src real).ProgChecks x w) {ν : CoinLayer.Nonce}
    (sn : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z)
    (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray)
    {c c' : Fin (refK x) → FlockDeploy.Coin} {n : ℕ} (hcc : ∀ k : Fin (refK x), k.val < n → c k = c' k)
    (hr : Serves x → RowsOK (vsServed _ bx) x w → ∀ j,
      ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wt j, vsOf _ bx x w ν c sn j) (ω j).1 (ω j).2.1)
    (hr' : Serves x → RowsOK (vsServed _ bx) x w → ∀ j,
      ((tabC PE x j).T (czExe PE x c' j)).tab.onVals.RowsAt (wt j, vsOf _ bx x w ν c' sn j) (ω j).1 (ω j).2.1) :
    encStop (encExe PE) x c n (swCutAt (tabC PE x) (czExe PE x c) wt (vsOf _ bx x w ν c sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c) wt (vsOf _ bx x w ν c sn) ω l + (p l).1, (p l).2)) =
      encStop (encExe PE) x c' n (swCutAt (tabC PE x) (czExe PE x c') wt (vsOf _ bx x w ν c' sn) ω fun l =>
        ZkView.leafH (swDigOf (tabC PE x) (czExe PE x c') wt (vsOf _ bx x w ν c' sn) ω l + (p l).1, (p l).2)) := by
  by_cases hg : Serves x ∧ RowsOK (vsServed _ bx) x w
  · obtain ⟨hSv, hRo⟩ := hg
    unfold encStop
    by_cases hf : Refused x c n
    · rw [ite_eq_left hf, ite_eq_left ((refused_congr hcc).1 hf)]
    · rw [ite_eq_right hf, ite_eq_right (mt (refused_congr hcc).2 hf)]
      exact swEncExe_congr L₀ src real hSv (hP hSv) bx hRo wt ω p hcc (hr hSv hRo) (hr' hSv hRo)
  · have hv : ∀ c₀ : Fin (refK x) → FlockDeploy.Coin,
        vsOf _ bx x w ν c₀ sn = fun j => ((tabC PE x j).T (czExe PE x c₀ j)).tab.vals := fun c₀ => by
      rw [vsOf_of_not _ bx hg]
      rfl
    rw [hv c, hv c', swDigOf_vals, swDigOf_vals, swCutAt_vals, swCutAt_vals]
    exact causal_encStop causalExe x wt ω p c c' n hcc

end SwRep

/-- **`SwCausal`'s round at `vsOf` where the tables' rows hold**: at `SessionLoopZKDeployed`'s release with the loop's
own inner rounds (`withInner`) and its tables `realOf src`, the switched round encoding at `vsOf bx` agrees at coins
agreeing before the round, at every witness, nonce, base, seen values and leaf pairs where, if `vsOf` is `vsServed`
there, every table's rows hold at its base at both coin vectors (`swEnc_congr`; its `ProgChecks` as `takesVals_vsOf`
has it). -/
theorem swCausal_rows
    (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray)
    (hreal : real = realOf src)
    (bx : AlgWorker (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real))
    (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce) (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray) (c c' : Fin (refK x) → FlockDeploy.Coin)
    (i : ℕ) (hcc : ∀ k : Fin (refK x), k.val < i → c k = c' k)
    (hr : Serves x → RowsOK (vsServed _ bx) x w → ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt
      ((CoinLayer.session loopNr Masked.dataE).wt x w j, vsOf _ bx x w ν c s j) (ω j).1 (ω j).2.1)
    (hr' : Serves x → RowsOK (vsServed _ bx) x w → ∀ j, ((tabC PE x j).T (czExe PE x c' j)).tab.onVals.RowsAt
      ((CoinLayer.session loopNr Masked.dataE).wt x w j, vsOf _ bx x w ν c' s j) (ω j).1 (ω j).2.1) :
    (CoinLayer.session loopNr Masked.dataE).enc x c i
        (swCutAt ((CoinLayer.session loopNr Masked.dataE).C x) ((CoinLayer.session loopNr Masked.dataE).cz x c)
          ((CoinLayer.session loopNr Masked.dataE).wt x w) (vsOf _ bx x w ν c s) ω fun l =>
          ZkView.leafH (swDigOf ((CoinLayer.session loopNr Masked.dataE).C x)
            ((CoinLayer.session loopNr Masked.dataE).cz x c) ((CoinLayer.session loopNr Masked.dataE).wt x w)
            (vsOf _ bx x w ν c s) ω l + (p l).1, (p l).2)) =
      (CoinLayer.session loopNr Masked.dataE).enc x c' i
        (swCutAt ((CoinLayer.session loopNr Masked.dataE).C x) ((CoinLayer.session loopNr Masked.dataE).cz x c')
          ((CoinLayer.session loopNr Masked.dataE).wt x w) (vsOf _ bx x w ν c' s) ω fun l =>
          ZkView.leafH (swDigOf ((CoinLayer.session loopNr Masked.dataE).C x)
            ((CoinLayer.session loopNr Masked.dataE).cz x c') ((CoinLayer.session loopNr Masked.dataE).wt x w)
            (vsOf _ bx x w ν c' s) ω l + (p l).1, (p l).2)) :=
  swEnc_congr _ src real bx (fun hSv => LeanRelease.ofLoop_progChecks_sizes Masked.dataE
      (LeanRelease.withFiles (LeanRelease.withLink (LeanRelease.withInner Masked.dataE L₁ real fin) src) src) real x w
      (sizesOk_of_serves hSv) (fun ρ _ _ _ hs => by rw [hreal]; exact realTabs_of src x w ρ hs)
      (_root_.Flock.SecurityProofs.LoopInnerRound0Own Masked.dataE L₁ real fin x w))
    s _ ω p hcc hr hr'

/-- **`SwOk`'s coin vectors meet `RowsOK`'s outer-coin condition** at a statement the loop serves: none of their outer
coins is `1` (`outerReads_czExe`), or every table admits them. -/
theorem rOuter_ne_one_of_swOk {x : Msg} {c : Fin (refK x) → FlockDeploy.Coin} (h : SwOk Masked.dataE x c)
    (hSv : Serves x) (j : Fin (jOf x)) : ∀ (r : Fin 2), ∀ y ∈ (tabC PE x j).rOuter (czExe PE x c j) r, y ≠ 1 := by
  rcases h with h | h
  · intro r y hy
    obtain ⟨e, he, rfl⟩ := outerReads_czExe PE x c j r y hy
    have hj := lt_of_jOf j
    exact h (live_of_lt j) hSv j.val hj r.val r.isLt e (by rw [getElem!_pos (stsOf x) j.val hj]; exact he)
  · exact (h j).2.2.2

/-- **One-tape `SwCausal` at `vsOf`, at `SwOk`'s coin vectors** (`RefSession.SwCausalOn`), at every `bx`:
`swCausal_rows` at the tape's bases, what the workers see of it and its level-0 leaves' pairs; where `vsOf` is
`vsServed` its rows hold there (`RowsOK`) at both coin vectors (`rOuter_ne_one_of_swOk`). -/
theorem swCausal_oneTape_vsOf
    (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray)
    (hreal : real = realOf src)
    (bx : AlgWorker (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real)) :
    (CoinLayer.session loopNr Masked.dataE).SwCausalOn (SwOk Masked.dataE) (vsOf _ bx) :=
  fun x w ν t p1 c c' i hcc hc hc' =>
    swCausal_rows L₁ src real fin hreal bx x w ν ((CoinLayer.session loopNr Masked.dataE).eP x t).1
      ((CoinLayer.session loopNr Masked.dataE).seenOf x t) ((CoinLayer.session loopNr Masked.dataE).tapePairs x t p1)
      c c' i hcc
      (fun hSv hRo j => (rowsOK_congr (vsOf_of_serves _ bx hSv hRo)).2 hRo ν c t j (rOuter_ne_one_of_swOk hc hSv j))
      fun hSv hRo j => (rowsOK_congr (vsOf_of_serves _ bx hSv hRo)).2 hRo ν c' t j (rOuter_ne_one_of_swOk hc' hSv j)

end Loop

end Claims.ZKServer
