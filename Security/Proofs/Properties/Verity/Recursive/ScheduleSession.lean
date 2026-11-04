import Proofs.Properties.Verity.Recursive.ScheduleInner
import Proofs.Flock.Claims.Recursion.RecRecord

/-!
# The recursive verifier's schedule: one V* session

The converse of `RecDeploy.session_core` and `RecDeploy.sim`, on a run of the compiled recursive verifier
`RecDeploy.V IV st` that need not accept. Take V* session `s` of the run, at its V* statement `vstarAt` read off the
run's inner and staging rounds, and suppose:

* part `s`'s staged numbers are its numbers (`numsOf (staged …) = nums …`, the check the compiled verifier makes at each
  of the session's rounds);
* the outer verifier `VO` accepts the session's view as the run shows it (`session`, with each answer's coin
  `outCoin`, as `LiveCoins` reads it), at an oracle where it accepts only at its last round (`OuterAtEnd`, which
  SHA-512 is).

Then the run reaches the session's last round, and the compiled verifier answers it as the outer verifier's
acceptance (`sessOut … .acc`: `.acc` at the last session, else `.msg []`), and before the last session the run goes on
to session `s + 1` (`session_end`). The session's earlier rounds are answered by their coins (`session_mid`), so the
coins its view shows are the run's own, but for the last one, which no decision reads (`nextO_last_coin`).
-/

namespace Verity.Recursive.Schedule

open Claims (Msg Out Prover Trace stepWith run coinSeq)
open FlockDeploy (Coin)
open RecDeploy

/-- **Session `s`'s first round** in a run whose inner and staging rounds are `ms`: past them and the earlier V*
sessions' rounds, at the V* statements read off `ms` (`session`'s `off`). -/
noncomputable def sessOff (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (ms : List (Msg × Out)) (s : ℕ) : ℕ :=
  IV.k (xIOf x) + kS x + (((vstar st x ms).take s).map VO.k).sum

/-! ## The outer verifier's last coin -/

/-- The outer verifier answers a coin only as its coins' last. -/
theorem actO_coin_last {xO : Msg} {as : List Msg} {cs : List Coin} {c : Coin} (h : actO xO as cs = .coin c) :
    cs.getLast? = some c := by
  unfold actO at h
  split at h
  · cases h
  · unfold actAt at h
    split at h
    · rename_i _ hc
      cases h
      exact hc
    · repeat' split at h
      all_goals cases h
    · cases h

/-- A decision is not on a last message of tag `0`, which the outer verifier answers with a coin. -/
theorem actO_decide_tag {xO : Msg} {as : List Msg} {cs : List Coin} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Msg × Coin)}
    {f : FlockDeploy.Final} (h : actO xO as cs = .decide s spec sts link c₁ mids f) (hcs : cs ≠ []) (t : Msg) :
    as.getLast? ≠ some (0 :: t) := by
  intro ha
  cases hc : cs.getLast? with
  | none => exact hcs (List.getLast?_eq_none_iff.1 hc)
  | some c =>
    unfold actO at h
    split at h
    · cases h
    · unfold actAt at h
      rw [ha, hc] at h
      simp at h

/-- `actAt` reads its last coin only at a last message of tag `0`. -/
theorem actAt_last (s : FlockDeploy.Ins) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt) (as : List Msg)
    (c₀ : Coin) (cs : List Coin) (c c' : Coin) (ha : ∀ t, as.getLast? ≠ some (0 :: t)) :
    actAt s spec sts as (c₀ :: cs ++ [c']) = actAt s spec sts as (c₀ :: cs ++ [c]) := by
  unfold actAt
  rw [List.getLast?_concat, List.getLast?_concat]
  rcases hl : as.getLast? with _ | a
  · rfl
  · have ha' : ∀ t, a ≠ 0 :: t := fun t h => ha t (by rw [hl, h])
    split
    · rename_i heq _
      exact absurd (Option.some.inj heq) (ha' _)
    · rename_i a₂ heq _
      obtain rfl := Option.some.inj heq
      symm
      split
      · rename_i heq' _
        exact absurd (Option.some.inj heq') (ha' _)
      · rename_i a₃ heq' _
        obtain rfl := Option.some.inj heq'
        rw [List.cons_append, List.cons_append]
        rcases hf : FlockDeploy.finalOf a with _ | f
        · rfl
        · rcases as with _ | ⟨a₁, rest⟩
          · rfl
          · simp only [List.dropLast_concat]
      · rename_i heq'
        cases heq'
    · rename_i heq
      cases heq

/-- **The outer verifier's step doesn't read its last coin** where the last message is not of tag `0`: coins that
agree but for the last give the same step. -/
theorem nextO_last {xO : Msg} {as : List Msg} {cs cs' : List Coin} (ha : ∀ t, as.getLast? ≠ some (0 :: t))
    (h2 : 2 ≤ cs.length) (hl : cs.length = cs'.length) (hd : cs.dropLast = cs'.dropLast) :
    nextO xO as cs = nextO xO as cs' := by
  rcases List.eq_nil_or_concat cs with rfl | ⟨L, b, rfl⟩
  · simp at h2
  rcases List.eq_nil_or_concat cs' with rfl | ⟨L', b', rfl⟩
  · simp at hl
  simp only [List.concat_eq_append, List.dropLast_concat] at hd h2
  subst hd
  rcases L with _ | ⟨c₀, m⟩
  · simp at h2
  have he : actO xO as (c₀ :: m ++ [b]) = actO xO as (c₀ :: m ++ [b']) := by
    unfold actO
    split
    · rfl
    · exact actAt_last _ _ _ as c₀ m b' b ha
  simp only [List.concat_eq_append]
  unfold nextO
  rw [he]

/-- The coins a run's round `off + j` reads past `off` end on its own. -/
theorem coins_getLast {k : ℕ} (r : Fin k → Coin) (off j : ℕ) :
    (((List.range (off + j + 1)).map (coinSeq r)).drop off).getLast? = some (coinSeq r (off + j)) := by
  rw [List.getLast?_drop, if_neg (by simp only [List.length_map, List.length_range]; omega), List.getLast?_map,
    List.getLast?_range, if_neg (by omega)]
  simp

/-- `fixLast` changes only a last answer, never a message. -/
theorem fixLast_spec (last : Bool) (k : ℕ) (seg : List (Msg × Out)) :
    (fixLast last k seg).map Prod.fst = seg.map Prod.fst ∧ (fixLast last k seg).dropLast = seg.dropLast := by
  unfold fixLast
  split_ifs
  · split
    · rename_i a h
      obtain ⟨ys, rfl⟩ := List.getLast?_eq_some_iff.1 h
      simp
    · exact ⟨rfl, rfl⟩
  · exact ⟨rfl, rfl⟩

section Session

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV st).k x) → Coin)

open Classical in
/-- **A session's round in the run**: the compiled verifier's step at session `s`'s round `j` is the outer verifier's
at the session's messages and coins, but at its first round where it refuses. -/
theorem ract_sess {s : ℕ} (hs : s < nS x) {xs : List Msg} {off : ℕ} {xO : Msg}
    (hxs : xs = vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))) (hoff : off = (IV.k (xIOf x) + kS x) + ((xs.take s).map VO.k).sum)
    (hxO : xO = xs.getD s []) (hnum : numsOf (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) = nums st x s)
    {j : ℕ} (hj : j < VO.k xO) (hl : off + j < (run H P (V IV st) x ω r).msgs.length) (pa : Msg) :
    (j = 0 ∧ ract IV st x (((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa])
        ((List.range (off + j + 1)).map (coinSeq r)) = .rej) ∨
      ract IV st x (((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa]) ((List.range (off + j + 1)).map (coinSeq r)) =
        .sess s j xO ((((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa]).drop off)
          (((List.range (off + j + 1)).map (coinSeq r)).drop off) := by
  obtain ⟨xr, hx⟩ := readX_of_nS x (by omega)
  have hxl : xs.length = nS x := by rw [hxs, vstar_length]
  have hml : (((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst).length = off + j := by
    simp only [List.length_map, List.length_take]
    omega
  have hlen : (((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa]).length = off + j + 1 := by
    simp only [List.length_append, hml, List.length_singleton]
  have hip : innerPairs (IV.k (xIOf x) + kS x) (((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa])
      ((List.range (off + j + 1)).map (coinSeq r)) = (run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x) :=
    innerPairs_reach H IV st x P ω r hx (by omega) hl.le [pa] (range_map_take _ (by omega))
  have hloc : locate ((vstar st x (innerPairs (IV.k (xIOf x) + kS x) (((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa])
      ((List.range (off + j + 1)).map (coinSeq r)))).map VO.k)
      ((((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa]).length - 1 - (IV.k (xIOf x) + kS x)) = some (s, j) := by
    rw [hip, ← hxs, hlen, show off + j + 1 - 1 - (IV.k (xIOf x) + kS x) = ((xs.map VO.k).take s).sum + j by
      rw [← List.map_take]; omega]
    exact locate_eq _ (by simp only [List.length_map]; omega)
      (by rw [getD_map_k xs (s := s) (by omega), ← hxO]; exact hj)
  rw [ract_eq (st := st) hx (by rw [hlen]; omega) hloc, hip, ← hxs, hlen,
    show off + j + 1 - 1 - j = off by omega, ← hxO]
  split_ifs with h1 h2
  · exact absurd hnum h1
  · exact Or.inl ⟨h2.1, rfl⟩
  · exact Or.inr rfl

/-- **A session's round in the run**, as `run_get` and `ract_sess` give it. -/
theorem round_at {s : ℕ} (hs : s < nS x) {xs : List Msg} {off : ℕ} {xO : Msg}
    (hxs : xs = vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))) (hoff : off = (IV.k (xIOf x) + kS x) + ((xs.take s).map VO.k).sum)
    (hxO : xO = xs.getD s []) (hnum : numsOf (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) = nums st x s)
    {j : ℕ} (hj : j < VO.k xO) (hl : off + j < (run H P (V IV st) x ω r).msgs.length) :
    ∃ pa, (run H P (V IV st) x ω r).msgs[off + j]? = some (pa, ((next IV st x (((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa])
        ((List.range (off + j + 1)).map (coinSeq r))).eval H).1) ∧
      ((j = 0 ∧ ract IV st x (((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa])
          ((List.range (off + j + 1)).map (coinSeq r)) = .rej) ∨
        ract IV st x (((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa]) ((List.range (off + j + 1)).map (coinSeq r)) =
          .sess s j xO ((((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa]).drop off)
            (((List.range (off + j + 1)).map (coinSeq r)).drop off)) := by
  obtain ⟨new, hnew, hget⟩ := run_get H P (V IV st) x ω r (off + j) hl
  subst hnew
  exact ⟨_, hget, ract_sess H IV st x P ω r hs hxs hoff hxO hnum hj hl _⟩

/-- **A session's round before its last is answered by its coin**, where the run goes past it. -/
theorem round_mid {s : ℕ} (hs : s < nS x) {xs : List Msg} {off : ℕ} {xO : Msg}
    (hxs : xs = vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))) (hoff : off = (IV.k (xIOf x) + kS x) + ((xs.take s).map VO.k).sum)
    (hxO : xO = xs.getD s []) (hnum : numsOf (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) = nums st x s)
    {j : ℕ} (hj : j + 1 < VO.k xO) (hl : off + j + 1 < (run H P (V IV st) x ω r).msgs.length) :
    ∃ a, (run H P (V IV st) x ω r).msgs[off + j]? = some (a, .msg (coinSeq r (off + j)).bytes) := by
  obtain ⟨pa, hget, hr⟩ := round_at H IV st x P ω r hs hxs hoff hxO hnum (j := j) (by omega) (by omega)
  obtain ⟨a, m, hm⟩ := run_mid H IV st x P ω r hl
  have hans : ((next IV st x (((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa])
      ((List.range (off + j + 1)).map (coinSeq r))).eval H).1 = .msg m := by
    rw [hget] at hm
    simp only [Option.some.injEq, Prod.mk.injEq] at hm
    exact hm.2
  refine ⟨a, ?_⟩
  rw [hm]
  rcases hr with ⟨-, hrej⟩ | hsess
  · rw [next_rej H hrej] at hans
    cases hans
  · rw [next_sess H hsess, show (j + 1 == VO.k xO) = false from beq_eq_false_iff_ne.2 (by omega)] at hans
    cases ho : ((nextO xO ((((run H P (V IV st) x ω r).msgs.take (off + j)).map Prod.fst ++ [pa]).drop off)
        (((List.range (off + j + 1)).map (coinSeq r)).drop off)).eval H).1 with
    | msg m' =>
      rw [ho] at hans
      have hmm : m' = m := by simpa [sessOut] using hans
      obtain ⟨c, hc, hmc⟩ := nextO_msg H ho
      have hlast := actO_coin_last hc
      rw [coins_getLast] at hlast
      have hceq : coinSeq r (off + j) = c := Option.some.inj hlast
      rw [← hmm, hmc, hceq]
    | acc =>
      rw [ho] at hans
      simp [sessOut] at hans
    | rej =>
      rw [ho] at hans
      simp [sessOut] at hans

/-- **F2's core**, at session `s`'s first round `off` and V* statement `xO`: the run reaches the session's last round,
answers each earlier round by its coin, and answers the last as the outer verifier's acceptance. -/
theorem session_facts (hO : OuterAtEnd H) {s : ℕ} (hs : s < nS x) {xs : List Msg} {off : ℕ} {xO : Msg}
    (hxs : xs = vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))) (hoff : off = (IV.k (xIOf x) + kS x) + ((xs.take s).map VO.k).sum)
    (hxO : xO = xs.getD s []) (hnum : numsOf (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) = nums st x s)
    (hacc : ((nextO xO ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map Prod.fst)
        ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map fun m => outCoin m.2)).eval H).1 = .acc) :
    off + VO.k xO ≤ (run H P (V IV st) x ω r).msgs.length ∧
      (∀ j, j + 1 < VO.k xO → ∃ a, (run H P (V IV st) x ω r).msgs[off + j]? = some (a, .msg (coinSeq r (off + j)).bytes)) ∧
      ∃ a, (run H P (V IV st) x ω r).msgs[off + VO.k xO - 1]? = some (a, sessOut (s + 1 == nS x) true .acc) := by
  have hsm : (session IV st x s (run H P (V IV st) x ω r).seen).msgs =
      fixLast (s + 1 == nS x) (VO.k xO) (((run H P (V IV st) x ω r).msgs.drop off).take (VO.k xO)) := by
    subst hxO hoff hxs
    rfl
  rw [hsm] at hacc
  set seg := ((run H P (V IV st) x ω r).msgs.drop off).take (VO.k xO) with hseg
  obtain ⟨hsvf, hsvd⟩ := fixLast_spec (s + 1 == nS x) (VO.k xO) seg
  set sv := fixLast (s + 1 == nS x) (VO.k xO) seg with hsv
  have hkpos := VO_k_pos xO
  have hlenSv : sv.length = VO.k xO := by
    have := hO xO (sv.map Prod.fst) (sv.map fun m => outCoin m.2) (by simp) hacc
    simpa using this
  have hseglen : seg.length = VO.k xO := by
    have := congrArg List.length hsvf
    simp only [List.length_map] at this
    omega
  have hle : off + VO.k xO ≤ (run H P (V IV st) x ω r).msgs.length := by
    have : seg.length = min (VO.k xO) ((run H P (V IV st) x ω r).msgs.length - off) := by simp [hseg]
    omega
  have hmid : ∀ j, j + 1 < VO.k xO → ∃ a, (run H P (V IV st) x ω r).msgs[off + j]? = some (a, .msg (coinSeq r (off + j)).bytes) :=
    fun j hj => round_mid H IV st x P ω r hs hxs hoff hxO hnum hj (by omega)
  refine ⟨hle, hmid, ?_⟩
  obtain ⟨s', spec, sts, link, c₁, mids, f, hd⟩ : ∃ s' spec sts link c₁ mids f,
      actO xO (sv.map Prod.fst) (sv.map fun m => outCoin m.2) = .decide s' spec sts link c₁ mids f := by
    cases ha : actO xO (sv.map Prod.fst) (sv.map fun m => outCoin m.2) with
    | rej => rw [nextO_rej H ha] at hacc; cases hacc
    | coin c => rw [nextO_coin H ha] at hacc; cases hacc
    | decide s' spec sts link c₁ mids f => exact ⟨_, _, _, _, _, _, _, rfl⟩
  obtain ⟨i, hi, -⟩ := actO_decide hd
  have hk2 : 2 ≤ VO.k xO := by
    have : VO.k xO = 2 + FlockDeploy.rounds true spec sts := by
      show (match setupO xO with | some (_, spec, sts) => 2 + FlockDeploy.rounds true spec sts | none => 1) = _
      rw [hi]
    omega
  have htag : ∀ t, (sv.map Prod.fst).getLast? ≠ some (0 :: t) := by
    refine actO_decide_tag hd ?_
    intro h0
    have := congrArg List.length h0
    simp only [List.length_map, List.length_nil, hlenSv] at this
    omega
  obtain ⟨pa, hget, hr⟩ := round_at H IV st x P ω r hs hxs hoff hxO hnum (j := VO.k xO - 1) (by omega) (by omega)
  have hsess : ract IV st x (((run H P (V IV st) x ω r).msgs.take (off + (VO.k xO - 1))).map Prod.fst ++ [pa])
      ((List.range (off + (VO.k xO - 1) + 1)).map (coinSeq r)) =
      .sess s (VO.k xO - 1) xO ((((run H P (V IV st) x ω r).msgs.take (off + (VO.k xO - 1))).map Prod.fst ++ [pa]).drop off)
        (((List.range (off + (VO.k xO - 1) + 1)).map (coinSeq r)).drop off) := by
    rcases hr with ⟨hj0, -⟩ | h
    · omega
    · exact h
  have hA : (((run H P (V IV st) x ω r).msgs.take (off + (VO.k xO - 1))).map Prod.fst ++ [pa]).drop off = sv.map Prod.fst := by
    have ht : ((run H P (V IV st) x ω r).msgs.take (off + VO.k xO)).map Prod.fst =
        ((run H P (V IV st) x ω r).msgs.take (off + (VO.k xO - 1))).map Prod.fst ++ [pa] := by
      rw [show off + VO.k xO = off + (VO.k xO - 1) + 1 by omega, List.take_add_one, hget]
      simp
    rw [← ht, ← List.map_drop, List.drop_take, Nat.add_sub_cancel_left, hsvf]
  have hClen : (((List.range (off + (VO.k xO - 1) + 1)).map (coinSeq r)).drop off).length = VO.k xO := by
    simp only [List.length_drop, List.length_map, List.length_range]
    omega
  have hsvclen : (sv.map fun m => outCoin m.2).length = VO.k xO := by
    rw [List.length_map, hlenSv]
  have hD : (((List.range (off + (VO.k xO - 1) + 1)).map (coinSeq r)).drop off).dropLast =
      (sv.map fun m => outCoin m.2).dropLast := by
    apply List.ext_getElem?
    intro i
    rw [List.getElem?_dropLast, List.getElem?_dropLast, hClen, hsvclen]
    split_ifs with hi
    · have hsvi : sv[i]? = seg[i]? := by
        have := congrArg (·[i]?) hsvd
        simp only [List.getElem?_dropLast, hlenSv, hseglen, hi, ↓reduceIte] at this
        exact this
      obtain ⟨a, ha⟩ := hmid i (by omega)
      rw [List.getElem?_drop, List.getElem?_map, List.getElem?_range (by omega), List.getElem?_map, hsvi, hseg,
        List.getElem?_take_of_lt (by omega), List.getElem?_drop, ha]
      simp [outCoin, coinOf_bytes]
    · rfl
  have hN := nextO_last (xO := xO) htag (by omega) (by rw [hClen, hsvclen]) hD
  rw [next_sess H hsess, hA, hN, hacc, show (VO.k xO - 1 + 1 == VO.k xO) = true from beq_iff_eq.2 (by omega)]
    at hget
  exact ⟨pa, by rw [show off + VO.k xO - 1 = off + (VO.k xO - 1) by omega]; exact hget⟩

/-- **F2, the session's rounds before its last**: under F2's hypotheses, each is answered by its coin's bytes. -/
theorem session_mid (hO : OuterAtEnd H) {s : ℕ} (hs : s < nS x)
    (hnum : numsOf (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) = nums st x s)
    (hacc : ((nextO (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s)
        ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map Prod.fst)
        ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map fun m => outCoin m.2)).eval H).1 = .acc)
    {j : ℕ} (hj : j + 1 < VO.k (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s)) :
    ∃ a, (run H P (V IV st) x ω r).msgs[sessOff IV st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s + j]? =
      some (a, .msg (coinSeq r (sessOff IV st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s + j)).bytes) :=
  (session_facts H IV st x P ω r hO hs rfl rfl (vstar_getD st x _ hs).symm hnum hacc).2.1 j hj

/-- **F2: one V\* session.** Where part `s`'s staged numbers are its numbers and the outer verifier, at an oracle where
it accepts only at its last round, accepts session `s`'s view of the run, the run reaches the session's last round, the
compiled verifier answers it as the outer verifier's acceptance (`.acc` at the last session, else `.msg []`), and before
the last session the run goes on to session `s + 1`. -/
theorem session_end (hO : OuterAtEnd H) {s : ℕ} (hs : s < nS x)
    (hnum : numsOf (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) = nums st x s)
    (hacc : ((nextO (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s)
        ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map Prod.fst)
        ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map fun m => outCoin m.2)).eval H).1 = .acc) :
    sessOff IV st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s + VO.k (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) ≤ (run H P (V IV st) x ω r).msgs.length ∧
    (∃ a, (run H P (V IV st) x ω r).msgs[sessOff IV st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s + VO.k (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) - 1]? =
      some (a, sessOut (s + 1 == nS x) true .acc)) ∧
    (s + 1 < nS x → sessOff IV st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s + VO.k (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) <
      (run H P (V IV st) x ω r).msgs.length) := by
  obtain ⟨hle, -, a, ha⟩ := session_facts H IV st x P ω r hO hs (xs := vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)))
    (off := sessOff IV st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) rfl rfl (vstar_getD st x _ hs).symm hnum hacc
  refine ⟨hle, ⟨a, ha⟩, fun hs1 => ?_⟩
  rw [show (s + 1 == nS x) = false from beq_eq_false_iff_ne.2 (by omega)] at ha
  have ha' : (run H P (V IV st) x ω r).msgs[sessOff IV st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s + VO.k (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) - 1]? =
      some (a, .msg []) := ha
  have hxl : (vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))).length = nS x := vstar_length st x _
  have hK := k_eq IV st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))
  have ht0 := rounds_through (vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))) (s := s) (by omega)
  have ht1 := rounds_through (vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))) (s := s + 1) (by omega)
  have hsum := take_sum_le ((vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))).map VO.k) (s + 1 + 1)
  rw [← List.map_take] at hsum
  have hpos1 := VO_k_pos ((vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))).getD (s + 1) [])
  rw [vstar_getD st x _ hs] at ht0
  have hpos0 := VO_k_pos (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s)
  have hoff : sessOff IV st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s =
      (IV.k (xIOf x) + kS x) + (((vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))).take s).map VO.k).sum := rfl
  rcases run_full H IV st x P ω r ha' with h | h <;> omega

end Session

end Verity.Recursive.Schedule
