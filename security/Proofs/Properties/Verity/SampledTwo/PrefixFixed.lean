module

public import Proofs.Properties.Verity.SampledTwo.RoundsRise
public import Proofs.Properties.Verity.SampledTwo.SessionDev
public import Proofs.Properties.Verity.SampledTwo.CoinPrefix
public import Proofs.Properties.Verity.SampledTwo.MessagesShown

/-! The play before a session's round is fixed by the coins its history shows; `Messages.SessionMsgsAsk`
is a premise here, not proved. -/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Verity.Sampled.Binding (Read)

theorem forall₂_split_right {α β : Type} {R : α → β → Prop} {a : List β} {y : β} {rest : List β} :
    ∀ {l : List α}, List.Forall₂ R l (a ++ y :: rest) →
      ∃ la x lrest, l = la ++ x :: lrest ∧ List.Forall₂ R la a ∧ R x y ∧ List.Forall₂ R lrest rest := by
  induction a with
  | nil =>
    intro l h
    rw [List.nil_append] at h
    cases h with
    | cons hx hrest => exact ⟨[], _, _, rfl, List.Forall₂.nil, hx, hrest⟩
  | cons z a ih =>
    intro l h
    rw [List.cons_append] at h
    cases h with
    | cons hw h₁ =>
      obtain ⟨la, x, lrest, rfl, hla, hx, hlr⟩ := ih h₁
      exact ⟨_ :: la, x, lrest, rfl, List.Forall₂.cons hw hla, hx, hlr⟩

theorem same_append_frames {l l' : List Release}
    (hl : List.Forall₂ (fun r r' : Release => r.frame = r'.frame) l l') :
    ∀ {pre pre' : List Release}, Same pre pre' → Same (pre ++ l) (pre' ++ l') := by
  induction hl with
  | nil =>
    intro pre pre' h
    rw [List.append_nil, List.append_nil]
    exact h
  | @cons r r' l l' hr _ ih =>
    intro pre pre' h
    rw [show pre ++ r :: l = pre ++ [r] ++ l by simp, show pre' ++ r' :: l' = pre' ++ [r'] ++ l' by simp]
    exact ih (h.snoc hr)

theorem openTree_of_locOf {A : Live₂} {tab : ℕ → String} {pre : List Release} {fr : Verity.Frame} {n : ℕ}
    (h : locOf A tab pre fr = some n) : ∃ i, openTree fr = some i := by
  cases fr with
  | fresh _ => simp [locOf, sessLoc] at h
  | hello _ => simp [locOf, sessLoc] at h
  | opening i s c => exact ⟨i, rfl⟩

theorem split_past {q l' a rest : List Release} {r : Release} (h : q ++ l' = a ++ r :: rest)
    (hq : ∀ x ∈ q, openTree x.frame = none) {i : ℕ} (hr : openTree r.frame = some i) :
    ∃ a', a = q ++ a' ∧ l' = a' ++ r :: rest := by
  rcases List.append_eq_append_iff.1 h with ⟨a', ha, hl⟩ | ⟨c', hq', hc⟩
  · exact ⟨a', ha, hl⟩
  · cases c' with
    | nil =>
      rw [List.append_nil] at hq'
      rw [List.nil_append] at hc
      exact ⟨[], by rw [List.append_nil, hq'], hc.symm⟩
    | cons y c'' =>
      rw [List.cons_append] at hc
      have hy : r = y := (List.cons.inj hc).1
      have hn : openTree y.frame = none := hq y (by rw [hq']; simp)
      rw [← hy, hr] at hn
      cases hn

def LogRising (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ a r b r' c, o.log = a ++ r :: (b ++ r' :: c) →
    ∀ n n', locOf A tab a r.frame = some n → locOf A tab (a ++ r :: b) r'.frame = some n' → n < n'

theorem logRising_of {A : Live₂} {C : CaseOf} {Good : Exchange → Prop} {tab : ℕ → String}
    (hR : LocatedRising A C tab) (hI : InteriorLogged C Good) (hRd : InteriorReads C) :
    LogRising A C Good tab := by
  intro o hacc hgood a r b r' c hsplit n n' h1 h2
  obtain ⟨l, hl⟩ := hRd o hacc
  obtain ⟨q, l', hlog, hq, hF⟩ := hI o hacc hgood l hl
  have hF' : List.Forall₂ (fun x y : Release => x.frame = y.frame) l l' :=
    (List.forall₂_map_left_iff.1 hF).imp fun _ _ h => (of_ofV2 h).1
  obtain ⟨i, hi⟩ := openTree_of_locOf h1
  rw [hlog] at hsplit
  obtain ⟨a', rfl, hl'⟩ := split_past hsplit (fun x hx => (hq x hx).1) hi
  subst hl'
  obtain ⟨la, x, lrest, rfl, hla, hx, hrest⟩ := forall₂_split_right hF'
  obtain ⟨lb, x', lc, rfl, hlb, hx', -⟩ := forall₂_split_right hrest
  have hxf : x.frame = r.frame := hx
  have hxf' : x'.frame = r'.frame := hx'
  have S0 : Same [] q := by simpa using same_append (pre := []) hq
  have S1 : Same la (q ++ a') := by simpa using same_append_frames hla S0
  have S2 : Same (la ++ [x]) (q ++ a' ++ [r]) := S1.snoc hxf
  have S3 : Same (la ++ [x] ++ lb) (q ++ a' ++ [r] ++ b) := same_append_frames hlb S2
  have e1 : locOf A tab la x.frame = some n := by
    unfold locOf at h1 ⊢
    rw [sessLoc_congr S1, hxf]
    exact h1
  have e2 : locOf A tab (la ++ x :: lb) x'.frame = some n' := by
    unfold locOf at h2 ⊢
    rw [show la ++ x :: lb = la ++ [x] ++ lb by simp, sessLoc_congr S3, hxf']
    rw [show q ++ a' ++ r :: b = q ++ a' ++ [r] ++ b by simp] at h2
    exact h2
  exact hR o hacc _ hl la x lb x' lc rfl n n' e1 e2

theorem sessions_of_lt {A : Live₂} {n : ℕ} (h : A.h.stage + 1 < n) : groupOf A n = .sessions := by
  unfold groupOf
  rw [ite_eq_right (show ¬ n = 0 by omega), ite_eq_right (show ¬ n ≤ A.h.stage by omega),
    ite_eq_right (show ¬ n = A.h.stage + 1 by omega)]

theorem exists_of_usedOf {L : CoinMap.Locate} {j : ℕ} : ∀ (q pre : List Release),
    CoinMap.usedOf L j pre q ≠ 0 →
      ∃ a r b, q = a ++ r :: b ∧ CoinMap.locAt L (pre ++ a) r.frame r.answers = some j
  | [], pre, h => by simp [CoinMap.usedOf] at h
  | rl :: q, pre, h => by
    by_cases hl : CoinMap.locAt L pre rl.frame rl.answers = some j
    · exact ⟨[], rl, q, rfl, by rw [List.append_nil]; exact hl⟩
    · have h' : CoinMap.usedOf L j (pre ++ [rl]) q ≠ 0 := by
        intro e
        apply h
        show (if CoinMap.locAt L pre rl.frame rl.answers = some j then CoinMap.widthOf rl.frame else 0) +
          CoinMap.usedOf L j (pre ++ [rl]) q = 0
        simp only [ite_eq_right hl, e, Nat.add_zero]
      obtain ⟨a, r, b, rfl, hr⟩ := exists_of_usedOf q (pre ++ [rl]) h'
      exact ⟨rl :: a, r, b, rfl, by rw [show pre ++ rl :: a = pre ++ [rl] ++ a by simp]; exact hr⟩

theorem used_below {A : Live₂} {C : CaseOf} {Good : Exchange → Prop} {tab : ℕ → String} {L : CoinMap.Locate}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) (hRise : LogRising A C Good tab) {o : Exchange}
    (hacc : FormAccepts C o) (hgood : Good o) {q rest : List Release} {rl : Release} {i : ℕ}
    (hlog : o.log = q ++ rl :: rest) (hi : CoinMap.locAt L q rl.frame rl.answers = some i)
    (hg : groupOf A i = .sessions) {j : ℕ} (hj : CoinMap.usedOf L j [] q ≠ 0) : j < i := by
  obtain ⟨a, r, b, rfl, hr⟩ := exists_of_usedOf q [] hj
  rw [List.nil_append] at hr
  have hsi := stage_lt_of_sessions hg
  by_cases hgj : groupOf A j = .sessions
  · have h1 := (hL a r.frame r.answers j hgj).1 hr
    have h2 := (hL (a ++ r :: b) rl.frame rl.answers i hg).1 hi
    exact hRise o hacc hgood a r b rl rest (by rw [hlog]; simp) j i h1 h2
  · have hnj : ¬ A.h.stage + 1 < j := fun h => hgj (sessions_of_lt h)
    omega

theorem locAt_run_ne_stmt (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String) (pre : List Release)
    (fr : Verity.Frame) (b : ByteArray) :
    CoinMap.locAt (runLocate A Ls tab) pre fr b ≠ some (A.h.stage + 1) := by
  intro h
  cases fr with
  | hello sp => cases (show (none : Option ℕ) = some (A.h.stage + 1) from h)
  | opening i s c =>
    have hp := locOf_past (show locOf A tab pre (.opening i s c) = some (A.h.stage + 1) from h)
    omega
  | fresh l =>
    have h' : (Ls pre (.fresh l) b).filter (fun n => decide (n ≤ A.h.stage)) = some (A.h.stage + 1) := h
    obtain ⟨-, hle⟩ := Option.filter_eq_some_iff.1 h'
    have hd := of_decide_eq_true hle
    omega

theorem locAt_run_session (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String) {pre : List Release}
    {fr : Verity.Frame} {b : ByteArray} {n : ℕ} (h : CoinMap.locAt (runLocate A Ls tab) pre fr b = some n)
    (hn : A.h.stage + 1 < n) :
    ∃ t k, t < A.h.ms.length ∧ k < SpotFlock.k₀ A.h t ∧ n = SpotFlock.off A.h t + k := by
  cases fr with
  | hello sp => cases (show (none : Option ℕ) = some n from h)
  | fresh l =>
    have h' : (Ls pre (.fresh l) b).filter (fun m => decide (m ≤ A.h.stage)) = some n := h
    obtain ⟨-, hle⟩ := Option.filter_eq_some_iff.1 h'
    have hd := of_decide_eq_true hle
    omega
  | opening i s c =>
    have h' : locOf A tab pre (.opening i s c) = some n := h
    simp only [locOf, sessLoc] at h'
    obtain ⟨j, -, hif⟩ := Option.bind_eq_some_iff.1 h'
    have hc : (treesOpened pre).idxOf i < A.h.ms.length ∧ j < SpotFlock.k₀ A.h ((treesOpened pre).idxOf i) := by
      by_contra hc
      rw [ite_eq_right hc] at hif
      cases hif
    rw [ite_eq_left hc] at hif
    exact ⟨_, j, hc.1, hc.2, (Option.some.inj hif).symm⟩

end Verity.SampledTwo.SessionCoins

namespace Verity.SampledTwo.Messages

open Claims

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem shownAt_range_map (m g : ℕ → Msg) {i j : ℕ} {c : FlockDeploy.Coin} (hji : j < i) (h : g j = c.bytes) :
    shownAt ((List.range i).map fun l => (m l, g l)) j = some c := by
  unfold shownAt
  rw [List.getElem?_map, List.getElem?_range hji, Option.map_some, Option.bind_some]
  show shownCoin (g j) = some c
  rw [h]
  exact shownCoin_of c

theorem shownAt_playHist (A : Live₂) (x : Msg) (r : Coins A x) (ms : ℕ → Msg) {i : ℕ}
    (j : Fin (A.L.V.k x)) (hji : j.val < i)
    (h : msgOf ((A.L.V.next x ((List.range (j.val + 1)).map ms)
      ((List.range (j.val + 1)).map (coinSeq r))).eval FlockDeploy.sha512).1 = (r j).bytes) :
    shownAt (playHist A x r ms i) j.val = some (r j) := by
  unfold playHist
  exact shownAt_range_map ms (fun l => msgOf ((A.L.V.next x ((List.range (l + 1)).map ms)
    ((List.range (l + 1)).map (coinSeq r))).eval FlockDeploy.sha512).1) hji h

theorem prefix_shown (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (ω : TapeOf V θ F c σ) (r r' : Coins A x)
    (hist : List (Msg × Msg)) {q rest : List Verity.Release} {rl : Verity.Release}
    (h : (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)).log = q ++ rl :: rest)
    (hq : ∀ j : Fin (A.L.V.k x), CoinMap.usedOf L j.val [] q ≠ 0 → shownAt hist j.val = some (r j))
    (hs : Shows hist r') :
    ∃ cs rest', (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r' ω)).log =
      q ++ ⟨rl.frame, rl.answers, cs⟩ :: rest' :=
  CoinMap.place_prefix_used L V θ F c σ r r' ω h fun j hj => (hs j _ (hq j hj)).symm

theorem V_next (h : SpotFlock.Held) : (SpotFlock.V h).next = SpotFlock.next h := rfl

theorem next_stage_coin (h : SpotFlock.Held) (x : Msg) {as : List Msg} {cs : List FlockDeploy.Coin} (h0 : as ≠ [])
    (hcs : cs.length = as.length) (hle : as.length ≤ h.stage + 1)
    (hc : as.length ≤ 2 ∨ as.getD (as.length - 1) [] = []) :
    msgOf ((SpotFlock.next h x as cs).eval FlockDeploy.sha512).1 = (cs.getD (as.length - 1) default).bytes := by
  rw [SpotFlock.next_coin h x h0 hcs hle, ite_eq_left hc]
  simp only [OC.eval, msgOf]

theorem next_session_coin (h : SpotFlock.Held) (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) {t : ℕ}
    (has : h.stage + 2 < as.length) (hcs : cs.length = as.length) (hslot : SpotFlock.slotOf h as.length = some t)
    (hund : ∀ j, 0 < j → j < as.length - SpotFlock.off h t →
      SpotFlock.decidedAt (SpotFlock.stmtAt h (as.getD (h.stage + 1) []) t) (as.drop (SpotFlock.off h t))
        (cs.drop (SpotFlock.off h t)) j = false)
    {c : FlockDeploy.Coin}
    (hc : FlockDeploy.act true (SpotFlock.stmtAt h (as.getD (h.stage + 1) []) t) (as.drop (SpotFlock.off h t))
      (cs.drop (SpotFlock.off h t)) = .coin c) :
    msgOf ((SpotFlock.next h x as cs).eval FlockDeploy.sha512).1 = (cs.getD (as.length - 1) default).bytes := by
  rw [SpotFlock.next_long h x as cs has hcs, hslot]
  dsimp only
  rw [SpotFlock.sessStep_eval h _ t _ as cs hund]
  dsimp only
  unfold FlockDeploy.next
  rw [hc]
  simp only [OC.eval, Claims.Sampling.markOut, msgOf]
  rw [← hcs, getD_of_getLast?_drop default (Claims.Bridge.act_coin hc)]

theorem getLast?_take_drop_range_map {β : Type} (f : ℕ → β) {N o j : ℕ} (hj0 : 0 < j) (hj : o + j ≤ N) :
    ((((List.range N).map f).drop o).take j).getLast? = some (f (o + j - 1)) := by
  rw [List.getLast?_eq_getElem?, List.length_take, List.length_drop, List.length_map, List.length_range,
    Nat.min_eq_left (by omega), List.getElem?_take_of_lt (by omega), List.getElem?_drop, List.getElem?_map,
    List.getElem?_range (by omega), Option.map_some, show o + (j - 1) = o + j - 1 by omega]

theorem getLast?_drop_range_map {β : Type} (f : ℕ → β) {N o : ℕ} (h : o < N) :
    (((List.range N).map f).drop o).getLast? = some (f (N - 1)) := by
  rw [List.getLast?_eq_getElem?, List.length_drop, List.length_map, List.length_range, List.getElem?_drop,
    List.getElem?_map, List.getElem?_range (by omega), Option.map_some, show o + (N - o - 1) = N - 1 by omega]

def CoinAnsweredAt (Vr : Verifier FlockDeploy.Coin) (x : Msg) (ms : ℕ → Msg) (j : ℕ) : Prop :=
  ∀ (r : Fin (Vr.k x) → FlockDeploy.Coin) (hj : j < Vr.k x),
    msgOf ((Vr.next x ((List.range (j + 1)).map ms) ((List.range (j + 1)).map (coinSeq r))).eval
      FlockDeploy.sha512).1 = (r ⟨j, hj⟩).bytes

def CoinAnswered (A : Live₂) (x : Msg) (ms : ℕ → Msg) (j : ℕ) : Prop := CoinAnsweredAt A.L.V x ms j

theorem coinAnswered_of (A : Live₂) (x : Msg) (ms : ℕ → Msg) (j : ℕ)
    (h : CoinAnsweredAt (SpotFlock.V A.h) x ms j) : CoinAnswered A x ms j := by
  unfold CoinAnswered
  rw [L_V]
  exact h

theorem coinAnsweredAt_stage (h : SpotFlock.Held) (x : Msg) (ms : ℕ → Msg) {j : ℕ} (hj : j ≤ h.stage)
    (hm : j ≤ 1 ∨ ms j = []) : CoinAnsweredAt (SpotFlock.V h) x ms j := by
  intro r hjk
  have hl : ((List.range (j + 1)).map ms).length = j + 1 := by simp
  have hne : (List.range (j + 1)).map ms ≠ [] := List.ne_nil_of_length_pos (by omega)
  have hcl : ((List.range (j + 1)).map (coinSeq r)).length = ((List.range (j + 1)).map ms).length := by simp
  have hc : ((List.range (j + 1)).map ms).length ≤ 2 ∨
      ((List.range (j + 1)).map ms).getD (((List.range (j + 1)).map ms).length - 1) [] = [] := by
    rw [hl, Nat.add_sub_cancel, getD_range_map ms (Nat.lt_add_one j)]
    exact hm.imp (fun h => by omega) id
  rw [V_next, next_stage_coin h x hne hcl (by rw [hl]; omega) hc, hl, Nat.add_sub_cancel,
    getD_range_map (coinSeq r) (Nat.lt_add_one j), coinSeq_lt r hjk]

theorem coinAnswered_stage (A : Live₂) (x : Msg) (ms : ℕ → Msg) {j : ℕ} (hj : j ≤ A.h.stage)
    (hm : j ≤ 1 ∨ ms j = []) : CoinAnswered A x ms j :=
  coinAnswered_of A x ms j (coinAnsweredAt_stage A.h x ms hj hm)

theorem coinAnsweredAt_session (h : SpotFlock.Held) (x : Msg) (ms : ℕ → Msg) {t n : ℕ} (ht : t < h.ms.length)
    (h1 : SpotFlock.off h t ≤ n) (h2 : n < SpotFlock.off h t + SpotFlock.k₀ h t)
    (hset : (FlockDeploy.setupOf true (SpotFlock.stmtAt h (ms (h.stage + 1)) t)).isSome = true)
    (htag : ∀ l, SpotFlock.off h t ≤ l → l ≤ n → ∃ u, ms l = 0 :: u) : CoinAnsweredAt (SpotFlock.V h) x ms n := by
  intro r hnk
  have hso := SpotFlock.stage_le_off h t
  have hl : ((List.range (n + 1)).map ms).length = n + 1 := by simp
  have hlc : ((List.range (n + 1)).map (coinSeq r)).length = n + 1 := by simp
  have hcl : ((List.range (n + 1)).map (coinSeq r)).length = ((List.range (n + 1)).map ms).length := by
    rw [hl, hlc]
  have hxS : SpotFlock.stmtAt h (((List.range (n + 1)).map ms).getD (h.stage + 1) []) t =
      SpotFlock.stmtAt h (ms (h.stage + 1)) t := by
    rw [getD_range_map ms (show h.stage + 1 < n + 1 by omega)]
  have hslot : SpotFlock.slotOf h ((List.range (n + 1)).map ms).length = some t := by
    rw [hl]
    exact SpotFlock.slotOf_eq h ht (by omega) (by omega)
  have hund : ∀ j, 0 < j → j < ((List.range (n + 1)).map ms).length - SpotFlock.off h t →
      SpotFlock.decidedAt (SpotFlock.stmtAt h (((List.range (n + 1)).map ms).getD (h.stage + 1) []) t)
        (((List.range (n + 1)).map ms).drop (SpotFlock.off h t))
        (((List.range (n + 1)).map (coinSeq r)).drop (SpotFlock.off h t)) j = false := by
    intro j hj0 hj
    rw [hl] at hj
    obtain ⟨u, hu⟩ := htag (SpotFlock.off h t + j - 1) (by omega) (by omega)
    refine decidedAt_false_iff.2 (act_coin_iff.2 ⟨by rw [hxS]; exact hset, ⟨u, ?_⟩, ?_⟩)
    · rw [getLast?_take_drop_range_map ms hj0 (by omega), hu]
    · exact List.ne_nil_of_length_pos (by
        rw [List.length_take, List.length_drop, hlc]
        omega)
  obtain ⟨u, hu⟩ := htag n h1 (Nat.le_refl n)
  obtain ⟨c, hc⟩ := act_coin_iff.2 ⟨show (FlockDeploy.setupOf true (SpotFlock.stmtAt h
      (((List.range (n + 1)).map ms).getD (h.stage + 1) []) t)).isSome = true by rw [hxS]; exact hset,
    ⟨u, show (((List.range (n + 1)).map ms).drop (SpotFlock.off h t)).getLast? = some (0 :: u) by
      rw [getLast?_drop_range_map ms (show SpotFlock.off h t < n + 1 by omega), Nat.add_sub_cancel, hu]⟩,
    show ((List.range (n + 1)).map (coinSeq r)).drop (SpotFlock.off h t) ≠ [] from
      List.ne_nil_of_length_pos (by rw [List.length_drop, hlc]; omega)⟩
  rw [V_next, next_session_coin h x _ _ (by rw [hl]; omega) hcl hslot hund hc, hl, Nat.add_sub_cancel,
    getD_range_map (coinSeq r) (Nat.lt_add_one n), coinSeq_lt r hnk]

theorem coinAnswered_session (A : Live₂) (x : Msg) (ms : ℕ → Msg) {t n : ℕ} (ht : t < A.h.ms.length)
    (h1 : SpotFlock.off A.h t ≤ n) (h2 : n < SpotFlock.off A.h t + SpotFlock.k₀ A.h t)
    (hset : (FlockDeploy.setupOf true (SpotFlock.stmtAt A.h (ms (A.h.stage + 1)) t)).isSome = true)
    (htag : ∀ l, SpotFlock.off A.h t ≤ l → l ≤ n → ∃ u, ms l = 0 :: u) : CoinAnswered A x ms n :=
  coinAnswered_of A x ms n (coinAnsweredAt_session A.h x ms ht h1 h2 hset htag)

theorem playMsgs_stage (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (o : Verity.Exchange) {j : ℕ} (h1 : 1 < j)
    (hj : j ≤ A.h.stage) : playMsgs (formReading A C S dev) o j = [] := by
  simp only [playMsgs, formReading, ite_eq_right (show ¬ j = 0 by omega), ite_eq_right (show ¬ j = 1 by omega),
    ite_eq_left hj, Option.getD_some]

theorem coinAnswered_formStage (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (o : Verity.Exchange) {j : ℕ} (hj : j ≤ A.h.stage) :
    CoinAnswered A x (playMsgs (formReading A C S dev) o) j := by
  refine coinAnswered_stage A x _ hj ?_
  by_cases h1 : j ≤ 1
  · exact Or.inl h1
  · exact Or.inr (playMsgs_stage A C S dev o (by omega) hj)

def ShowsPlaced (A : Live₂) (x : Msg) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (L : CoinMap.Locate) :
    Prop :=
  ∀ o, Acc o → Good o → ∀ a rl b, o.log = a ++ rl :: b → ∀ j, CoinMap.locAt L a rl.frame rl.answers = some j →
    CoinAnswered A x (playMsgs Rd o) j

def SessionsAsk (A : Live₂) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o, Acc o → Good o → ∀ a rl b, o.log = a ++ rl :: b → ∀ j, CoinMap.locAt L a rl.frame rl.answers = some j →
    A.h.stage + 1 < j →
      ∃ t, t < A.h.ms.length ∧ SpotFlock.off A.h t ≤ j ∧ j < SpotFlock.off A.h t + SpotFlock.k₀ A.h t ∧
        (FlockDeploy.setupOf true (SpotFlock.stmtAt A.h (playMsgs Rd o (A.h.stage + 1)) t)).isSome = true ∧
        ∀ l, SpotFlock.off A.h t ≤ l → l ≤ j → ∃ u, playMsgs Rd o l = 0 :: u

theorem showsPlaced_run (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (Ls : CoinMap.Locate)
    (tab : ℕ → String)
    (hAsk : SessionsAsk A (formReading A C S dev) (FormAccepts C) Good (SessionCoins.runLocate A Ls tab)) :
    ShowsPlaced A x (formReading A C S dev) (FormAccepts C) Good (SessionCoins.runLocate A Ls tab) := by
  intro o hacc hgood a rl b hlog j hj
  by_cases h1 : j ≤ A.h.stage
  · exact coinAnswered_formStage A x C S dev o h1
  by_cases h2 : j = A.h.stage + 1
  · subst h2
    exact absurd hj (SessionCoins.locAt_run_ne_stmt A Ls tab a rl.frame rl.answers)
  obtain ⟨t, ht, ho, hk, hset, htag⟩ := hAsk o hacc hgood a rl b hlog j hj (by omega)
  exact coinAnswered_session A x _ ht ho hk hset htag

theorem playMsgs_stmt (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (o : Verity.Exchange) :
    playMsgs (formReading A C S dev) o (A.h.stage + 1) = (S.stmts o).getD [] := by
  have hpos : 0 < A.h.stage := Nat.succ_pos _
  simp only [playMsgs, formReading, ite_eq_right (show ¬ A.h.stage + 1 = 0 by omega),
    ite_eq_right (show ¬ A.h.stage + 1 = 1 by omega), ite_eq_right (show ¬ A.h.stage + 1 ≤ A.h.stage by omega),
    ↓reduceIte]

theorem playMsgs_session (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (o : Verity.Exchange) {t l : ℕ}
    (ht : t < A.h.ms.length) (h1 : SpotFlock.off A.h t ≤ l) (h2 : l < SpotFlock.off A.h t + SpotFlock.k₀ A.h t) :
    playMsgs (formReading A C S dev) o l = (S.msg o t (l - SpotFlock.off A.h t)).getD [] := by
  have hso := SpotFlock.stage_le_off A.h t
  have hslot : SpotFlock.slotOf A.h (l + 1) = some t := SpotFlock.slotOf_eq A.h ht (by omega) (by omega)
  simp only [playMsgs, formReading, ite_eq_right (show ¬ l = 0 by omega), ite_eq_right (show ¬ l = 1 by omega),
    ite_eq_right (show ¬ l ≤ A.h.stage by omega), ite_eq_right (show ¬ l = A.h.stage + 1 by omega), hslot]

def SessionMsgsAsk (A : Live₂) (S : SessionReading) (Acc Good : Verity.Exchange → Prop) (L : CoinMap.Locate) :
    Prop :=
  ∀ o, Acc o → Good o → ∀ a rl b, o.log = a ++ rl :: b → ∀ t k, t < A.h.ms.length → k < SpotFlock.k₀ A.h t →
    CoinMap.locAt L a rl.frame rl.answers = some (SpotFlock.off A.h t + k) →
      (FlockDeploy.setupOf true (SpotFlock.stmtAt A.h ((S.stmts o).getD []) t)).isSome = true ∧
        ∀ l, l ≤ k → ∃ u, (S.msg o t l).getD [] = 0 :: u

theorem sessionsAsk_run (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (Ls : CoinMap.Locate)
    (tab : ℕ → String) (h : SessionMsgsAsk A S (FormAccepts C) Good (SessionCoins.runLocate A Ls tab)) :
    SessionsAsk A (formReading A C S dev) (FormAccepts C) Good (SessionCoins.runLocate A Ls tab) := by
  intro o hacc hgood a rl b hlog j hj hpast
  obtain ⟨t, k, ht, hk, rfl⟩ := SessionCoins.locAt_run_session A Ls tab hj hpast
  obtain ⟨hset, htag⟩ := h o hacc hgood a rl b hlog t k ht hk hj
  refine ⟨t, ht, Nat.le_add_right _ _, Nat.add_lt_add_left hk _, ?_, fun l hl1 hl2 => ?_⟩
  · rw [playMsgs_stmt]
    exact hset
  · obtain ⟨u, hu⟩ := htag (l - SpotFlock.off A.h t) (by omega)
    exact ⟨u, by rw [playMsgs_session A C S dev o ht hl1 (by omega)]; exact hu⟩

theorem showsPlaced_form (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (Ls : CoinMap.Locate)
    (tab : ℕ → String) (h : SessionMsgsAsk A S (FormAccepts C) Good (SessionCoins.runLocate A Ls tab)) :
    ShowsPlaced A x (formReading A C S dev) (FormAccepts C) Good (SessionCoins.runLocate A Ls tab) :=
  showsPlaced_run A x C S dev Good Ls tab (sessionsAsk_run A C S dev Good Ls tab h)

theorem session_prefix_fixed (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (C : CaseOf)
    (Good : Verity.Exchange → Prop) (tab : ℕ → String) (Rd : Reading A)
    (hL : SessionCoins.LocatesSessions A tab (CoinMap.locAt L)) (hRise : SessionCoins.LogRising A C Good tab)
    (hShow : ShowsPlaced A x Rd (FormAccepts C) Good L) (ω : TapeOf V θ F c σ) (r r' : Coins A x)
    (hacc : FormAccepts C (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)))
    (hgood : Good (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)))
    {q rest : List Verity.Release} {rl : Verity.Release} {i : ℕ}
    (h : (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)).log = q ++ rl :: rest)
    (hi : CoinMap.locAt L q rl.frame rl.answers = some i) (hg : groupOf A i = .sessions)
    (hs : Shows (playHist A x r
      (playMsgs Rd (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω))) i) r') :
    ∃ cs rest', (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r' ω)).log =
      q ++ ⟨rl.frame, rl.answers, cs⟩ :: rest' := by
  refine prefix_shown V A x θ F c σ L ω r r' _ h (fun j hj => ?_) hs
  have hlt : j.val < i := SessionCoins.used_below hL hRise hacc hgood h hi hg hj
  obtain ⟨a, r₀, b, hqe, hr₀⟩ := SessionCoins.exists_of_usedOf q [] hj
  rw [List.nil_append] at hr₀
  have hans := hShow _ hacc hgood a r₀ (b ++ rl :: rest) (by rw [h, hqe]; simp) j.val hr₀
  exact shownAt_playHist A x r _ j hlt (hans r j.isLt)

end Verity.SampledTwo.Messages
