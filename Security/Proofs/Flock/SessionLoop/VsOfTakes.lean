module

public import Proofs.Flock.SessionLoop.VsOf
public import Proofs.Flock.SessionLoop.LiveInnerAt
public import Proofs.Flock.SessionLoop.FinL0Vals
public import Proofs.Flock.SessionLoop.RepSrc

/-!
# A pad-blind worker meets `TakesVals` at its own values

* **`Loop.takesVals_vsOf`**: at `SessionLoopZKDeployed`'s release with the loop's own inner rounds (`withInner`), at
  every statement the loop serves, `phased bx post` meets `TakesVals` at `vsOf bx`, at every witness at which
  `vsServed bx`'s rows hold (`RowsOK`, where `vsOf bx` is `vsServed bx`), every `bx` and whatever `post` reads.
* `Loop.freedAt_unique`: two rounds that free pad `i` of rep stream `2j + r` along one coin vector free it as one word.
  At the same prefix the worker's round (`bwOf` hands `bx` the answers after `Hello`'s), the frame's state and the free
  list are the same (`frameAnswer_congr`, `Free.roundFree_agree`), so the place in the absorbed order is the same; at a
  longer prefix the frame's state has moved past every place the shorter one frees (`Free.roundFree_pos`, the rep's
  later rounds each leaving the state their check returns: `ProgChecks`, `step_pos`), and the absorbed order is
  injective (`absorbed_inj`).
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)
open _root_.Flock.Firewall.Frame (Cfg Masker absorbed nLig)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

theorem locate_fst_mono : ∀ (rs : List ℕ) (k i i' : ℕ), i ≤ i' →
    (CoinLayer.locate rs k i).1 ≤ (CoinLayer.locate rs k i').1
  | [], _, _, _, _ => Nat.le_refl _
  | c :: cs, k, i, i', h => by
    by_cases h1 : i' < c
    · rw [locate_cons_lt h1, locate_cons_lt (Nat.lt_of_le_of_lt h h1)]
    · rw [locate_cons_ge h1]
      by_cases h2 : i < c
      · rw [locate_cons_lt h2]
        exact Nat.le_of_succ_le (locate_fst_ge cs (k + 1) (i' - c))
      · rw [locate_cons_ge h2]
        exact locate_fst_mono cs (k + 1) (i - c) (i' - c) (by omega)

/-- **Past a rep stream's first round its rounds are not first**: where two rounds `i < i'` are of one rep stream,
`i'` is not the stream's round `0`. -/
theorem locate_snd_ne_zero : ∀ (rs : List ℕ) (k i i' : ℕ), i < i' →
    (CoinLayer.locate rs k i).1 = (CoinLayer.locate rs k i').1 → (CoinLayer.locate rs k i').2 ≠ 0
  | [], _, i, i', h, _ => by
    show i' ≠ 0
    omega
  | c :: cs, k, i, i', h, he => by
    by_cases h1 : i' < c
    · rw [locate_cons_lt h1]
      show i' ≠ 0
      omega
    · rw [locate_cons_ge h1] at he ⊢
      by_cases h2 : i < c
      · rw [locate_cons_lt h2] at he
        have he' : k = (CoinLayer.locate cs (k + 1) (i' - c)).1 := he
        have := locate_fst_ge cs (k + 1) (i' - c)
        omega
      · rw [locate_cons_ge h2] at he
        exact locate_snd_ne_zero cs (k + 1) (i - c) (i' - c) (by omega) he

/-- **The frame's check never moves the state back.** -/
theorem check_pos {c : Cfg} {m : Masker} {st : Bool} {v r : ByteArray} {n : ℕ} {out : ByteArray} {s' : Masker}
    {why : Option String} (h : _root_.Flock.Firewall.Frame.check c { m with stopped := st } v r n = .ok (out, s', why)) :
    m.pos ≤ s'.pos := by
  cases why with
  | none =>
    obtain ⟨fr, hf⟩ := Free.roundFree_of_round (check_none h)
    exact le_trans (Nat.le_add_right _ _) (Free.roundFree_pos hf).1
  | some w =>
    obtain ⟨fr, hf⟩ := Free.roundFree_of_round (check_some h)
    exact le_trans (Nat.le_add_right _ _) (Free.roundFree_pos hf).1

/-- **Two checks against one scheduled round leave the state at one place.** -/
theorem check_pos_eq {c : Cfg} {m : Masker} {r : ByteArray} {n : ℕ} {st₁ st₂ : Bool} {v₁ v₂ o₁ o₂ : ByteArray}
    {s₁ s₂ : Masker} {w₁ w₂ : Option String}
    (hr : ∃ q, _root_.Flock.Firewall.Frame.roundFree c { m with stopped := false } r n = .ok q)
    (h₁ : _root_.Flock.Firewall.Frame.check c { m with stopped := st₁ } v₁ r n = .ok (o₁, s₁, w₁))
    (h₂ : _root_.Flock.Firewall.Frame.check c { m with stopped := st₂ } v₂ r n = .ok (o₂, s₂, w₂)) :
    s₁.pos = s₂.pos := by
  obtain ⟨⟨o, t, f⟩, hr⟩ := hr
  have e : unstop s₁ = unstop s₂ := (Free.check_unstop hr h₁).trans (Free.check_unstop hr h₂).symm
  show (unstop s₁).pos = (unstop s₂).pos
  exact congrArg Masker.pos e

theorem bind_const_ok {ε α β : Type} {e : Except ε α} {f : Except ε β} {b : β} (h : (e >>= fun _ => f) = .ok b) :
    f = .ok b := by
  cases e with
  | error _ => cases h
  | ok _ => exact h

/-- Two rounds of one rep stream are checked at alike configurations, whatever the tapes. -/
theorem cfgAlike_rep {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ ρ' : R) (h h' : List (Msg × Msg))
    (he : L.rep x h.length = L.rep x h'.length) : CfgAlike (L.cfg x ρ h) (L.cfg x ρ' h') := by
  unfold LeanRelease.cfg LeanRelease.drawsAt
  rw [he]
  exact cfgAlike_ofBytes _ _ _ _

theorem cfg_order_of_rep {W R : Type} (L : LeanRelease W R) {x : Msg} {ρ : R} {h : List (Msg × Msg)} {k : ℕ}
    (hk : L.rep x h.length = k) :
    (L.cfg x ρ h).order = absorbed (L.lay x k) ∧
      (L.cfg x ρ h).nPiop = (absorbed (L.lay x k)).size - nLig (L.lay x k) := by
  unfold LeanRelease.cfg
  rw [hk]
  exact ⟨rfl, rfl⟩

/-- **The absorbed order is injective.** -/
theorem absorbed_inj (l : _root_.Flock.Zk.PadLayout) {P P' : ℕ} (h : P < (absorbed l).size)
    (h' : P' < (absorbed l).size) (he : (absorbed l)[P]! = (absorbed l)[P']!) : P = P' := by
  rw [absorbed_getElem! l h, absorbed_getElem! l h'] at he
  split_ifs at he <;> omega

/-- **The algebraic phase's pads are below `n_E` and not `final_c`'s.** -/
theorem absorbed_lt_hab (l : _root_.Flock.Zk.PadLayout) {P : ℕ} (h : P < (absorbed l).size - nLig l) :
    (absorbed l)[P]! < l.hab ∧ (absorbed l)[P]! ≠ l.fc := by
  have hs := absorbed_size l
  rw [absorbed_getElem! l (p := P) (by omega)]
  simp only [_root_.Flock.Zk.PadLayout.hab, _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc,
    _root_.Flock.Zk.PadLayout.fc]
  split_ifs <;> constructor <;> omega

theorem tail_snoc_take {α : Type} (e0 : Msg × Msg) (ops : List (Msg × Msg)) {q : ℕ} (hq : q < ops.length) (o : Msg)
    (f : Msg → α) :
    ((e0 :: ops.take q) ++ [(o, ops[q].2)]).tail.map (fun p => f p.2) = (ops.take (q + 1)).map (fun p => f p.2) := by
  rw [List.cons_append, List.tail_cons, ← List.take_append_getElem hq, List.map_append, List.map_append]
  rfl

/-- **The frame takes the coins after two histories with the same answers alike**: the loop's frame reads the history
past `Hello` only through the decoded coins. -/
theorem frameAnswer_congr (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (ρ : (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand)
    (e0 e0' : Msg × Msg) {h1 h1' : List (Msg × Msg)}
    (hd : h1.map (fun p => CoinLayer.rd p.2) = h1'.map (fun p => CoinLayer.rd p.2)) :
    (LeanRelease.ofLoop T L₀ real).frameAnswer x ρ (e0 :: h1) =
      (LeanRelease.ofLoop T L₀ real).frameAnswer x ρ (e0' :: h1') := by
  have hl : h1.length = h1'.length := by simpa using congrArg List.length hd
  obtain ⟨m0, r0⟩ := e0
  obtain ⟨m0', r0'⟩ := e0'
  have hc : coinsAt x ((m0, r0) :: h1) = readCoins x (h1.map fun p => CoinLayer.rd p.2) :=
    LeanRelease.loop_coinsRead loopNr T (LeanRelease.withFrames T L₀ real) x m0 r0 h1
  have hc' : coinsAt x ((m0', r0') :: h1') = readCoins x (h1'.map fun p => CoinLayer.rd p.2) :=
    LeanRelease.loop_coinsRead loopNr T (LeanRelease.withFrames T L₀ real) x m0' r0' h1'
  rw [LeanRelease.ofLoop_frameAnswer, LeanRelease.ofLoop_frameAnswer, hc, hc', hd, LeanRelease.cfg_eq_cfgAt,
    LeanRelease.cfg_eq_cfgAt]
  simp only [List.length_cons, hl]

/-- **At a rep's round past its first the frame takes the coins at the state the last round left**: the loop's
state at the answers so far (`Loop.progOf`), its place unmoved by the answer (`Free.answer_pos`). -/
theorem answered_pos_same (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (ρ : (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {m : Masker} {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    {q j : ℕ} (hA : (LeanRelease.ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m)
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    (hj : (CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1) :
    m.pos = (progOf x (h1.map fun p => CoinLayer.rd p.2)).pos := by
  have hF := (LeanRelease.answered_ok hA).2.2
  rw [LeanRelease.ofLoop_frameAnswer_same T L₀ real x ρ e0 h1 hx hq hj] at hF
  exact Free.answer_pos (LeanRelease.bind_rankStop_ok hF).1

/-- **A round's check leaves the frame's next state at the check's place**: where the frame takes the coins after `h`
and the check of the worker's round `v` passes, the loop's state one round on is at the place the check left. -/
theorem step_pos {W R : Type} {L : LeanRelease W R} {x : Msg} {w : W} (hP : L.ProgChecks x w) (ρ : R)
    (h : List (Msg × Msg)) (a : Msg) {m : Masker} {out : ByteArray} {s' : Masker} {v : Msg}
    (hA : L.answered x ρ h = .ok m) (hk : L.kindAt x h.length = .masked)
    (hc : L.checkAt x w ρ h m false v = .ok (out, s', none)) :
    (L.prog x (h ++ [(0 :: out.data.toList, a)])).pos = s'.pos := by
  obtain ⟨v', st, s'', why, hc', hp⟩ := hP ρ h a m out hA hk ⟨v, s', none, hc⟩
  unfold LeanRelease.checkAt at hc hc'
  obtain ⟨q, hq, -⟩ := passes_none (check_none_passes hc)
  have e : (L.prog x (h ++ [(0 :: out.data.toList, a)])).pos = s''.pos := (congrArg Masker.pos hp).trans rfl
  exact e.trans (check_pos_eq ⟨q, hq⟩ hc' hc)

/-- **Where the frame takes the coins, the next round's rep starts**: at a rep's first round the frame's fresh state,
past it the rep's own (`ofLoop_start_at`). -/
theorem start_of_answered (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (ρ : (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {m : Masker}
    (hA : (LeanRelease.ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1) :
    ∃ lay mask, _root_.Flock.Firewall.Drive.start sts[(CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]! =
      .ok (lay, mask) := by
  rcases hsn : (CoinLayer.locate (roundsOf spec sts) 0 q).2 with _ | e
  · have hF := (LeanRelease.answered_ok hA).2.2
    rw [LeanRelease.ofLoop_frameAnswer] at hF
    unfold LeanRelease.takeAt at hF
    have hfr : freshOf x (h1.map fun p => CoinLayer.rd p.2) = .ok m := by
      by_cases h2 : (e0 :: h1).length ≤ 2
      · rwa [ite_eq_left h2] at hF
      · have hs : decide (repOf x (e0 :: h1).length = repOf x ((e0 :: h1).length - 1)) = false := by
          rw [decide_eq_false_iff_not]
          intro he
          obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := ⟨q - 1, by simp only [List.length_cons, hq] at h2; omega⟩
          have e1 : (e0 :: h1).length - 1 = q' + 2 := by
            simp only [List.length_cons, hq]
            omega
          have e2 : (e0 :: h1).length = q' + 1 + 2 := by simp only [List.length_cons, hq]
          rw [e1, e2, LeanRelease.repOf_add_two hx, LeanRelease.repOf_add_two hx] at he
          exact locate_snd_ne_zero _ 0 q' (q' + 1) (Nat.lt_succ_self _) he.symm hsn
        rw [ite_eq_right h2, hs, ite_eq_right Bool.false_ne_true] at hF
        exact bind_const_ok hF
    obtain ⟨c', t, dom, xs, hi, alg, hph, -⟩ :=
      freshOf_ok x _ hx (q := q) (by rw [List.length_map, hq]) hfr
    obtain ⟨lay, mask, hst, -⟩ := placeholder_ok hph
    exact ⟨lay, mask, hst⟩
  · exact LeanRelease.ofLoop_start_at T L₀ real x ρ e0 h1 hA rfl hx hq
      (k := (CoinLayer.locate (roundsOf spec sts) 0 q).1) (j := e) (Prod.ext rfl hsn)

section Takes

variable (L : LeanRelease Flock.Wit
  ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))

theorem vsOfWord_of_free (bx : AlgWorker L) {x : Msg} {w : Flock.Wit} {ν : CoinLayer.Nonce}
    {c : Fin (refK x) → FlockDeploy.Coin} {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    {j : Fin (jOf x)} {r : Fin 2} {i : ℕ} {a : F128}
    (wt : FlockSoundness.Model.Witness ((CoinLayer.session loopNr Masked.dataE).C x j).S.m)
    (u : (((CoinLayer.session loopNr Masked.dataE).C x j).T
      ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.Mask)
    (V : (((CoinLayer.session loopNr Masked.dataE).C x j).T
      ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.Vis)
    (h : freeWord L bx x w ν c s j r i = some a) : vsOfWord L bx x w ν c s j r wt u V i = a := by
  unfold vsOfWord
  split
  · rename_i a' h'
    rw [h] at h'
    exact (Option.some.inj h').symm
  · rename_i h'
    rw [h] at h'
    cases h'

theorem vsOfWord_of_none (bx : AlgWorker L) {x : Msg} {w : Flock.Wit} {ν : CoinLayer.Nonce}
    {c : Fin (refK x) → FlockDeploy.Coin} {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    {j : Fin (jOf x)} {r : Fin 2} {i : ℕ}
    (wt : FlockSoundness.Model.Witness ((CoinLayer.session loopNr Masked.dataE).C x j).S.m)
    (u : (((CoinLayer.session loopNr Masked.dataE).C x j).T
      ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.Mask)
    (V : (((CoinLayer.session loopNr Masked.dataE).C x j).T
      ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.Vis)
    (h : freeWord L bx x w ν c s j r i = none) :
    vsOfWord L bx x w ν c s j r wt u V i =
      if hi : i < (((CoinLayer.session loopNr Masked.dataE).C x j).T
          ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.nE then
        FlockLevel3.GF128.val ((((CoinLayer.session loopNr Masked.dataE).C x j).T
          ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.vals r wt u V ⟨i, hi⟩)
      else 0 := by
  unfold vsOfWord
  split
  · rename_i a' h'
    rw [h] at h'
    cases h'
  · rfl

end Takes

/-- **A live history's round `q` passes the frame's check**: at its first `q ≥ 1` openings past `Hello`, the worker's
round passes the check from the state the frame takes the coins at, and the loop's state one opening on is at the place
the check leaves (`step_pos`). -/
theorem round_at
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hP : (LeanRelease.ofLoop Masked.dataE L₀ real).ProgChecks x w)
    (wk : (LeanRelease.ofLoop Masked.dataE L₀ real).toSrv.Worker)
    (ρ : (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand)
    (κ : (LeanRelease.ofLoop Masked.dataE L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hl : (LeanRelease.ofLoop Masked.dataE L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hlt : ops.length < refK x) {q : ℕ} (h1 : 1 ≤ q) (h2 : q < ops.length) :
    ∃ vq M o sq,
      wk x w ρ ((m₀, ρ₀) :: ops.take q) 0 = some (vq, false) ∧
      (LeanRelease.ofLoop Masked.dataE L₀ real).answered x ρ ((m₀, ρ₀) :: ops.take q) = .ok M ∧
      (LeanRelease.ofLoop Masked.dataE L₀ real).checkAt x w ρ ((m₀, ρ₀) :: ops.take q) M false vq =
        .ok (o, sq, none) ∧
      (progOf x ((ops.take (q + 1)).map fun p => CoinLayer.rd p.2)).pos = sq.pos := by
  have hlen : ((m₀, ρ₀) :: ops.take q).length = q + 1 := by
    rw [List.length_cons, List.length_take, Nat.min_eq_left h2.le]
  have hlast : (LeanRelease.ofLoop Masked.dataE L₀ real).last x = refK x + 1 :=
    LeanRelease.loop_last loopNr Masked.dataE _ x
  have hlq := LeanRelease.live_take (LeanRelease.ofLoop Masked.dataE L₀ real) _ x w ρ κ _ hl (q + 1)
  rw [List.take_succ_cons] at hlq
  have hkq : (LeanRelease.ofLoop Masked.dataE L₀ real).kindAt x ((m₀, ρ₀) :: ops.take q).length = .masked :=
    LeanRelease.kindAt_masked_of _ (by rw [hlen]; omega) (by rw [hlen, hlast]; omega)
  obtain ⟨vq, hvq, -⟩ := LeanRelease.live_some (LeanRelease.ofLoop Masked.dataE L₀ real) _ x w ρ κ _ hkq hlq
  obtain ⟨M, o, sq, hM, hc⟩ :=
    LeanRelease.check_of_live (LeanRelease.ofLoop Masked.dataE L₀ real) _ x w ρ κ _ vq hlq hkq hvq
  refine ⟨vq, M, o, sq, hvq, hM, hc, ?_⟩
  have e := step_pos hP ρ ((m₀, ρ₀) :: ops.take q) (ops[q]).2 hM hkq hc
  rw [← tail_snoc_take (m₀, ρ₀) ops h2 (0 :: o.data.toList) CoinLayer.rd]
  exact e

/-- **Through a rep's rounds past its first the loop's place only moves on**: along a live history, from its first
`n₁ ≥ 1` openings to its first `n₂`, where each round between is a rep stream's round past its first, the loop's state
at the answers is no earlier at `n₂` (`round_at`, `answered_pos_same`, `check_pos`). -/
theorem progOf_pos_mono
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hP : (LeanRelease.ofLoop Masked.dataE L₀ real).ProgChecks x w)
    (wk : (LeanRelease.ofLoop Masked.dataE L₀ real).toSrv.Worker)
    (ρ : (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand)
    (κ : (LeanRelease.ofLoop Masked.dataE L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hl : (LeanRelease.ofLoop Masked.dataE L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hlt : ops.length < refK x) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) {n₁ n₂ : ℕ} (h1 : 1 ≤ n₁) (h12 : n₁ ≤ n₂)
    (h2 : n₂ ≤ ops.length)
    (hin : ∀ p, n₁ ≤ p → p < n₂ → ∃ j', (CoinLayer.locate (roundsOf spec sts) 0 (p - 1)).2 = j' + 1) :
    (progOf x ((ops.take n₁).map fun p => CoinLayer.rd p.2)).pos ≤
      (progOf x ((ops.take n₂).map fun p => CoinLayer.rd p.2)).pos := by
  have hwalk : ∀ d, n₁ + d ≤ n₂ → (progOf x ((ops.take n₁).map fun p => CoinLayer.rd p.2)).pos ≤
      (progOf x ((ops.take (n₁ + d)).map fun p => CoinLayer.rd p.2)).pos := by
    intro d
    induction d with
    | zero => intro _; exact le_rfl
    | succ d ih =>
      intro hd
      have ih' := ih (by omega)
      obtain ⟨vq, M, o, sq, -, hM, hc, hp⟩ :=
        round_at L₀ real hP wk ρ κ m₀ ρ₀ ops hl hlt (q := n₁ + d) (by omega) (by omega)
      obtain ⟨j', hj'⟩ := hin (n₁ + d) (by omega) (by omega)
      have hpos := answered_pos_same Masked.dataE L₀ real x ρ (m₀, ρ₀) (ops.take (n₁ + d)) hM hx
        (q := n₁ + d - 1) (by simp only [List.length_take]; omega) hj'
      unfold LeanRelease.checkAt at hc
      have hcp := check_pos (m := M) (st := false) hc
      have hp' : (progOf x ((ops.take (n₁ + (d + 1))).map fun p => CoinLayer.rd p.2)).pos = sq.pos := hp
      rw [hp']
      omega
  have h := hwalk (n₂ - n₁) (by omega)
  rwa [show n₁ + (n₂ - n₁) = n₂ by omega] at h

/-- **Two rounds that free one pad along one coin vector free it as one word**: where pad `i` of rep stream `2j + r` is
freed after a history opening the coins' first `n₁` and after one opening their first `n₂ ≥ n₁` (`FreedAt`, at the
same nonce and what the workers see), by a worker of the algebraic phase, the two words are one. At `n₁ = n₂` the
worker's round (it reads the answers after `Hello`'s, `bwOf`), the frame's state (`frameAnswer_congr`) and its free
list (`Free.roundFree_agree`) are the same, and so is the word's place in the absorbed order (`absorbed_inj`). At
`n₁ < n₂` the rep's rounds `n₁ + 1, …, n₂` each start where the last left the frame (`answered_pos_same`, `step_pos`),
past the places round `n₁` frees (`Free.roundFree_pos`), so no place is freed twice. -/
theorem freedAt_unique
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hP : (LeanRelease.ofLoop Masked.dataE L₀ real).ProgChecks x w)
    (bx : AlgWorker (LeanRelease.ofLoop Masked.dataE L₀ real))
    {ν : CoinLayer.Nonce} {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z}
    {C : List FlockDeploy.Coin} {j : Fin (jOf x)} {r : Fin 2} {i n₁ n₂ : ℕ} {a₁ a₂ : F128}
    (hn : n₁ ≤ n₂) (hC : n₂ ≤ C.length)
    (h₁ : FreedAt _ (blind _ (bwOf _ bx)) x w ν s (C.take n₁) j r i a₁)
    (h₂ : FreedAt _ (blind _ (bwOf _ bx)) x w ν s (C.take n₂) j r i a₂) : a₁ = a₂ := by
  obtain ⟨t₁, κ₁, m₀₁, ρ₀₁, ops₁, v₁, m₁, out₁, s₁, fr₁, k₁, hs₁, hd₁, hpos₁, hlt₁, hl₁, -, hv₁, ha₁, hf₁, hr₁,
    hk₁, hi₁, hw₁⟩ := h₁
  obtain ⟨t₂, κ₂, m₀₂, ρ₀₂, ops₂, v₂, m₂, out₂, s₂, fr₂, k₂, hs₂, hd₂, hpos₂, hlt₂, hl₂, -, hv₂, ha₂, hf₂, hr₂,
    hk₂, hi₂, hw₂⟩ := h₂
  subst hw₁ hw₂
  have hd₁' : ops₁.map (fun p => CoinLayer.rd p.2) = C.take n₁ := hd₁
  have hd₂' : ops₂.map (fun p => CoinLayer.rd p.2) = C.take n₂ := hd₂
  have hl₁' : ops₁.length = n₁ := by
    have e := congrArg List.length hd₁'
    simp only [List.length_map, List.length_take] at e
    omega
  have hl₂' : ops₂.length = n₂ := by
    have e := congrArg List.length hd₂'
    simp only [List.length_map, List.length_take] at e
    omega
  have hpre : ops₁.map (fun p => CoinLayer.rd p.2) = (ops₂.map (fun p => CoinLayer.rd p.2)).take n₁ := by
    rw [hd₁', hd₂', List.take_take, Nat.min_eq_left hn]
  obtain ⟨s0, spec, hx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) (Masked.lt_of_jOf j))
  have hR₁ : (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (n₁ - 1)).1 = 2 * j.val + r.val := by
    rw [← LeanRelease.repOf_add_two hx, show n₁ - 1 + 2 = ops₁.length + 1 by omega]
    exact hr₁
  have hR₂ : (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (n₂ - 1)).1 = 2 * j.val + r.val := by
    rw [← LeanRelease.repOf_add_two hx, show n₂ - 1 + 2 = ops₂.length + 1 by omega]
    exact hr₂
  have hin : ∀ q', n₁ ≤ q' → q' < n₂ →
      ∃ j', (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q').2 = j' + 1 := by
    intro q' h1 h2
    have e1 := locate_fst_mono (roundsOf spec (Masked.stsOf x)) 0 (n₁ - 1) q' (by omega)
    have e2 := locate_fst_mono (roundsOf spec (Masked.stsOf x)) 0 q' (n₂ - 1) (by omega)
    have hne := locate_snd_ne_zero (roundsOf spec (Masked.stsOf x)) 0 (n₁ - 1) q' (by omega) (by omega)
    exact ⟨(CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q').2 - 1, by omega⟩
  have ho₁ := cfg_order_of_rep (LeanRelease.ofLoop Masked.dataE L₀ real) (ρ := (ν, t₁))
    (h := (m₀₁, ρ₀₁) :: ops₁) (k := 2 * j.val + r.val) hr₁
  have ho₂ := cfg_order_of_rep (LeanRelease.ofLoop Masked.dataE L₀ real) (ρ := (ν, t₂))
    (h := (m₀₂, ρ₀₂) :: ops₂) (k := 2 * j.val + r.val) hr₂
  have hq₁ : m₁.pos + fr₁.size ≤ s₁.pos ∧ ∀ k < fr₁.size,
      m₁.pos + k < ((LeanRelease.ofLoop Masked.dataE L₀ real).cfg x (ν, t₁) ((m₀₁, ρ₀₁) :: ops₁)).nPiop :=
    Free.roundFree_pos hf₁
  have hq₂ : m₂.pos + fr₂.size ≤ s₂.pos ∧ ∀ k < fr₂.size,
      m₂.pos + k < ((LeanRelease.ofLoop Masked.dataE L₀ real).cfg x (ν, t₂) ((m₀₂, ρ₀₂) :: ops₂)).nPiop :=
    Free.roundFree_pos hf₂
  have hP₁ := hq₁.2 k₁ hk₁
  have hP₂ := hq₂.2 k₂ hk₂
  rw [ho₁.2] at hP₁
  rw [ho₂.2] at hP₂
  have hi₁' := hi₁
  have hi₂' := hi₂
  rw [ho₁.1] at hi₁'
  rw [ho₂.1] at hi₂'
  have hE := absorbed_inj _ (by omega) (by omega) (hi₁'.trans hi₂'.symm)
  rw [hl₁'] at hf₁
  rw [hl₂'] at hf₂
  rcases Nat.eq_or_lt_of_le hn with heq | hlt
  · -- one prefix: one round, one state, one free list
    have hD : ops₂.map (fun p => CoinLayer.rd p.2) = ops₁.map (fun p => CoinLayer.rd p.2) := by
      have e : (ops₂.map fun p => CoinLayer.rd p.2).length ≤ n₁ := by
        rw [List.length_map]
        omega
      rw [hpre, List.take_of_length_le e]
    have hm : m₂ = m₁ := by
      have e1 := (LeanRelease.answered_ok ha₁).2.2
      have e2 := (LeanRelease.answered_ok ha₂).2.2
      rw [LeanRelease.frameAnswer_tape (LeanRelease.ofLoop Masked.dataE L₀ real) x (ν, t₂) (ν, t₁) ((m₀₂, ρ₀₂) :: ops₂),
        frameAnswer_congr Masked.dataE L₀ real x (ν, t₁) (m₀₂, ρ₀₂) (m₀₁, ρ₀₁) hD] at e2
      exact Except.ok.inj (e2.symm.trans e1)
    have hv : v₂ = v₁ := by
      have e1 : bx x w ν ((CoinLayer.session loopNr Masked.dataE).seenOf x t₁)
          (some (ops₁.map fun p => CoinLayer.rd p.2)) 0 = some (v₁, false) := hv₁
      have e2 : bx x w ν ((CoinLayer.session loopNr Masked.dataE).seenOf x t₂)
          (some (ops₂.map fun p => CoinLayer.rd p.2)) 0 = some (v₂, false) := hv₂
      rw [hs₁] at e1
      rw [hs₂, hD] at e2
      exact (Prod.mk.inj (Option.some.inj (e2.symm.trans e1))).1
    rw [hm, hv, ← heq] at hf₂
    have hfr : fr₂ = fr₁ := (Free.roundFree_agree
      (cfgAlike_rep (LeanRelease.ofLoop Masked.dataE L₀ real) x (ν, t₁) (ν, t₂) ((m₀₁, ρ₀₁) :: ops₁)
        ((m₀₂, ρ₀₂) :: ops₂) (hr₁.trans hr₂.symm))
      (by exact rfl) (by exact rfl) hf₁ hf₂ (fun _ _ _ => rfl)).1
    have hk : k₂ = k₁ := by
      rw [hm] at hE
      omega
    rw [hv, hfr, hk]
  · -- a longer prefix: the rep's later rounds start past every place round `n₁` frees
    exfalso
    obtain ⟨vG, MG, oG, sG, hvG, hMG, hcG, hpG⟩ :=
      round_at L₀ real hP _ (ν, t₂) κ₂ m₀₂ ρ₀₂ ops₂ hl₂ hlt₂ (q := n₁) (by omega) (by omega)
    have hD : (ops₂.take n₁).map (fun p => CoinLayer.rd p.2) = ops₁.map (fun p => CoinLayer.rd p.2) := by
      rw [List.map_take, hpre]
    have hMm : MG = m₁ := by
      have e1 := (LeanRelease.answered_ok ha₁).2.2
      have e2 := (LeanRelease.answered_ok hMG).2.2
      rw [LeanRelease.frameAnswer_tape (LeanRelease.ofLoop Masked.dataE L₀ real) x (ν, t₂) (ν, t₁)
          ((m₀₂, ρ₀₂) :: ops₂.take n₁),
        frameAnswer_congr Masked.dataE L₀ real x (ν, t₁) (m₀₂, ρ₀₂) (m₀₁, ρ₀₁) hD] at e2
      exact Except.ok.inj (e2.symm.trans e1)
    have hvv : vG = v₁ := by
      have e1 : bx x w ν ((CoinLayer.session loopNr Masked.dataE).seenOf x t₁)
          (some (ops₁.map fun p => CoinLayer.rd p.2)) 0 = some (v₁, false) := hv₁
      have e2 : bx x w ν ((CoinLayer.session loopNr Masked.dataE).seenOf x t₂)
          (some ((ops₂.take n₁).map fun p => CoinLayer.rd p.2)) 0 = some (vG, false) := hvG
      rw [hs₁] at e1
      rw [hs₂, hD] at e2
      exact (Prod.mk.inj (Option.some.inj (e2.symm.trans e1))).1
    unfold LeanRelease.checkAt at hcG
    rw [hMm, hvv] at hcG
    obtain ⟨frG, hfG⟩ := Free.roundFree_of_round (check_none hcG)
    have hGl : ((m₀₂, ρ₀₂) :: ops₂.take n₁).length = n₁ + 1 := by
      rw [List.length_cons, List.length_take, Nat.min_eq_left (by omega)]
    rw [hGl] at hfG
    have hfr : frG = fr₁ := (Free.roundFree_agree
      (cfgAlike_rep (LeanRelease.ofLoop Masked.dataE L₀ real) x (ν, t₁) (ν, t₂) ((m₀₁, ρ₀₁) :: ops₁)
        ((m₀₂, ρ₀₂) :: ops₂.take n₁)
        (by rw [hGl, List.length_cons, hl₁']))
      (by exact rfl) (by exact rfl) hf₁ hfG (fun _ _ _ => rfl)).1
    have hbase : m₁.pos + fr₁.size ≤ (progOf x ((ops₂.take (n₁ + 1)).map fun p => CoinLayer.rd p.2)).pos := by
      have hb : m₁.pos + frG.size ≤ sG.pos := (Free.roundFree_pos hfG).1
      rw [hpG, ← hfr]
      exact hb
    have hw := progOf_pos_mono L₀ real hP _ (ν, t₂) κ₂ m₀₂ ρ₀₂ ops₂ hl₂ hlt₂ hx (n₁ := n₁ + 1) (n₂ := n₂)
      (by omega) (by omega) (by omega) (fun p h1 h2 => hin (p - 1) (by omega) (by omega))
    obtain ⟨j₂, hj₂⟩ := hin (n₂ - 1) (by omega) (by omega)
    have hm₂ := answered_pos_same Masked.dataE L₀ real x (ν, t₂) (m₀₂, ρ₀₂) ops₂ ha₂ hx (by omega) hj₂
    rw [← hl₂', List.take_length] at hw
    omega

/-- **A pad-blind worker of the algebraic phase meets `TakesVals` at its own values**: at `SessionLoopZKDeployed`'s
release with the loop's own inner rounds (`withInner`), its tables `realOf src`, at every statement the loop serves, every
algebraic phase's worker `bx`, every witness at which `vsServed bx`'s rows hold (`RowsOK`, so `vsOf bx` is
`vsServed bx` there) and any worker `post` after the phase, `phased bx post` meets `TakesVals` at `vsOf bx`. Each word
the frame frees in the worker's round is at a pad
below `n_E` other than `final_c`'s (`absorbed_lt_hab`, at the rep's layout: `lay_ofLoopLink`), where `vsOf` is the word
some round frees there along the coins (`freeWord`), and that round frees it as this word (`freedAt_unique`). -/
theorem takesVals_vsOf
    (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray)
    (hreal : real = Loop.realOf src) (x : Msg) (w : Flock.Wit) (hSv : Serves x)
    (bx : AlgWorker (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real))
    (hRo : RowsOK (vsServed _ bx) x w)
    (post : (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real).toSrv.Worker) :
    TakesVals (vsOf _ bx) (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real)
      (phased _ bx post) x w := by
  have hP : (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real).ProgChecks x w :=
    LeanRelease.ofLoop_progChecks_sizes Masked.dataE
      (LeanRelease.withFiles (LeanRelease.withLink (LeanRelease.withInner Masked.dataE L₁ real fin) src) src) real x w
      (sizesOk_of_serves hSv) (fun ρ _ _ _ hs => by rw [hreal]; exact realTabs_of src x w ρ hs)
      (_root_.Flock.SecurityProofs.LoopInnerRound0Own Masked.dataE L₁ real fin x w)
  intro ν t κ m₀ ρ₀ ops v m out s fr j r hpos hlt hl hok hv ha hfr hrep k hk
  have hle : ((m₀, ρ₀) :: ops).length ≤ refK x := by
    rw [List.length_cons]
    omega
  have he : ∀ h' : List (Msg × Msg), h'.length ≤ ((m₀, ρ₀) :: ops).length →
      phased _ bx post x w (ν, t) h' = blind _ (bwOf _ bx) x w (ν, t) h' :=
    fun h' hh => phased_alg _ bx post x w (ν, t) h' (hh.trans hle)
  have hl' := (ZkServer.live_congr _ κ he).symm.trans hl
  have hv' := (congrFun (he ((m₀, ρ₀) :: ops) le_rfl) 0).symm.trans hv
  have hcs := RefSession.ofFn_pad_take (CoinLayer.session loopNr Masked.dataE) (K := refK x)
    (ops.map fun p => (CoinLayer.session loopNr Masked.dataE).decOf p.2) (by rw [List.length_map]; exact hlt.le)
  rw [List.length_map, Nat.min_eq_left hlt.le] at hcs
  have hr2 := r.isLt
  obtain ⟨s0, spec, hx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) (Masked.lt_of_jOf j))
  obtain ⟨q, hq⟩ : ∃ q, ops.length = q + 1 := ⟨ops.length - 1, by omega⟩
  have hst := start_of_answered Masked.dataE
    (LeanRelease.withFiles (LeanRelease.withLink (LeanRelease.withInner Masked.dataE L₁ real fin) src) src) real x
    (ν, t) (m₀, ρ₀) ops ha hx hq
  have hk2 : (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1 = 2 * j.val + r.val := by
    rw [← LeanRelease.repOf_add_two hx, show q + 2 = ops.length + 1 by omega]
    exact hrep
  rw [hk2, show (2 * j.val + r.val) / 2 = j.val by omega] at hst
  obtain ⟨lay, mask, hst⟩ := hst
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := LeanRelease.withInner Masked.dataE L₁ real fin) src real j r hst
  have hord := cfg_order_of_rep (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real)
    (ρ := (ν, t)) (h := (m₀, ρ₀) :: ops) (k := frepOf j.val r.val) hrep
  rw [hlay] at hord
  have hP0 : m.pos + k < ((LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real).cfg x
      (ν, t) ((m₀, ρ₀) :: ops)).nPiop := (Free.roundFree_pos hfr).2 k hk
  rw [hord.2] at hP0
  have hab := absorbed_lt_hab (layE x j) hP0
  have hiE : (absorbed (layE x j))[m.pos + k]! < (tabC PE x j).z.nE := by
    rw [Masked.nE_eq_hab]
    exact hab.1
  have hF : FreedAt _ (blind _ (bwOf _ bx)) x w ν ((CoinLayer.session loopNr Masked.dataE).seenOf x t)
      ((List.ofFn ((CoinLayer.session loopNr Masked.dataE).pad (K := refK x)
        (ops.map fun p => (CoinLayer.session loopNr Masked.dataE).decOf p.2))).take ops.length) j r
      (absorbed (layE x j))[m.pos + k]! ((_root_.Flock.Firewall.Frame.words ⟨v.toArray⟩ fr[k]! 1)[0]!) :=
    ⟨t, κ, m₀, ρ₀, ops, v, m, out, s, fr, k, rfl, hcs.symm, hpos, hlt, hl', hok, hv', ha, hfr, hrep, hk,
      by rw [hord.1], rfl⟩
  have hex : ∃ n a, n < refK x ∧ FreedAt _ (blind _ (bwOf _ bx)) x w ν
      ((CoinLayer.session loopNr Masked.dataE).seenOf x t)
      ((List.ofFn ((CoinLayer.session loopNr Masked.dataE).pad (K := refK x)
        (ops.map fun p => (CoinLayer.session loopNr Masked.dataE).decOf p.2))).take n) j r
      (absorbed (layE x j))[m.pos + k]! a :=
    ⟨ops.length, _, hlt, hF⟩
  have hfw : freeWord _ bx x w ν ((CoinLayer.session loopNr Masked.dataE).pad
      (ops.map fun p => (CoinLayer.session loopNr Masked.dataE).decOf p.2))
      ((CoinLayer.session loopNr Masked.dataE).seenOf x t) j r (absorbed (layE x j))[m.pos + k]! =
      some ((_root_.Flock.Firewall.Frame.words ⟨v.toArray⟩ fr[k]! 1)[0]!) := by
    have e : freeWord _ bx x w ν ((CoinLayer.session loopNr Masked.dataE).pad
        (ops.map fun p => (CoinLayer.session loopNr Masked.dataE).decOf p.2))
        ((CoinLayer.session loopNr Masked.dataE).seenOf x t) j r (absorbed (layE x j))[m.pos + k]! =
        some (Classical.choose (Classical.choose_spec hex)) := by
      unfold freeWord
      exact dite_eq_left hex
    rw [e]
    obtain ⟨hn, hc⟩ := Classical.choose_spec (Classical.choose_spec hex)
    refine congrArg some ?_
    rcases le_total (Classical.choose hex) ops.length with h | h
    · exact freedAt_unique
        (LeanRelease.withFiles (LeanRelease.withLink (LeanRelease.withInner Masked.dataE L₁ real fin) src) src) real
        hP bx h (Nat.le_of_lt (Nat.lt_of_lt_of_eq hlt List.length_ofFn.symm)) hc hF
    · exact (freedAt_unique
        (LeanRelease.withFiles (LeanRelease.withLink (LeanRelease.withInner Masked.dataE L₁ real fin) src) src) real
        hP bx h (Nat.le_of_lt (Nat.lt_of_lt_of_eq hn List.length_ofFn.symm)) hF hc).symm
  rw [hord.1]
  refine Eq.symm ((vsWord_vsOf _ bx hSv w hRo ν _ t j r hiE).trans ?_)
  rw [ite_eq_right hab.2]
  exact vsOfWord_of_free _ bx _ _ _ hfw

end Loop

end Claims.ZKServer
