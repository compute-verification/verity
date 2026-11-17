module

public import Definitions.Claims.RecursionGeneric.Server
public import Proofs.Flock.Claims.Recursion.BoundBridge

/-! `Open(V, S)`'s runs on replayed messages in closed form, and what a run it accepts checks, at any server. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

namespace Server

variable {Coin : Type} [Inhabited Coin]

theorem ivstar_eval_ne_msg (S : Server) (V : Verifier Coin) (ans : Coin → Msg) (H : Msg → Msg) (x : Msg)
    (as : List Msg) (cs : List Coin) (h : ¬ as.length < V.k x) (m : Msg) :
    (((S.IVStar V ans).next x as cs).eval H).1 ≠ .msg m := by
  simp only [Server.IVStar, h, ↓reduceIte]
  split
  · simp [OC.eval]
  · split
    · rw [eval_bindOC]
      split
      · rw [eval_bindOC]
        exact Bound.eval_ret_ne_msg H _ m
      · simp [OC.eval]
    · simp [OC.eval]

def vnAt (S : Server) (V : Verifier Coin) (ans : Coin → Msg) (x : Msg) (r : Fin ((S.IVStar V ans).k x) → Coin) :
    List Msg → OC Out :=
  fun as => (S.IVStar V ans).next x as ((List.range as.length).map (coinSeq r))

def decAt (S : Server) (V : Verifier Coin) (ans : Coin → Msg) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin ((S.IVStar V ans).k x) → Coin) : Out × List (Msg × Msg) :=
  ((S.IVStar V ans).next x ((List.range (V.k x)).map fun i => ms.getD i [])
    ((List.range (V.k x)).map (coinSeq r))).eval H

theorem stepWith_ivstar (S : Server) (V : Verifier Coin) (ans : Coin → Msg) (H : Msg → Msg) (x : Msg)
    (ms : List Msg) (r : Fin ((S.IVStar V ans).k x) → Coin) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), hist.length = t.msgs.length →
      as.length = t.msgs.length → t.msgs.length + (n + 1) = V.k x →
      stepWith H (replay ms) x () (S.vnAt V ans x r) (n + 1) hist as t =
        { msgs := t.msgs ++ (List.range' t.msgs.length n).map (fun i => (ms.getD i [], Out.msg (ans (coinSeq r i)))) ++
            [(ms.getD (V.k x - 1) [], ((S.vnAt V ans x r (as ++
              (List.range' t.msgs.length (n + 1)).map fun i => ms.getD i [])).eval H).1)],
          pLog := t.pLog,
          vLog := t.vLog ++
            ((S.vnAt V ans x r (as ++ (List.range' t.msgs.length (n + 1)).map fun i => ms.getD i [])).eval H).2,
          out := ((S.vnAt V ans x r (as ++ (List.range' t.msgs.length (n + 1)).map fun i => ms.getD i [])).eval H).1 }
      := by
  intro n
  induction n with
  | zero =>
    intro hist as t hh ha hk
    have hp : ((replay ms).next x hist ()).eval H = (ms.getD t.msgs.length [], []) := by
      show (OC.ret (ms.getD hist.length [])).eval H = _
      rw [hh]
      rfl
    have hne : ∀ m, ((S.vnAt V ans x r (as ++ [ms.getD t.msgs.length []])).eval H).1 ≠ .msg m :=
      S.ivstar_eval_ne_msg V ans H x _ _ (by simp only [List.length_append, List.length_singleton]; omega)
    have hk1 : V.k x - 1 = t.msgs.length := by omega
    simp only [stepWith, hp, hk1, Nat.zero_add, List.range'_zero, List.map_nil, List.append_nil, List.range'_one,
      List.map_cons]
    rcases hv : (S.vnAt V ans x r (as ++ [ms.getD t.msgs.length []])).eval H with ⟨o, l⟩
    rw [hv] at hne
    rcases o with m | _ | _
    · exact absurd rfl (hne m)
    · rfl
    · rfl
  | succ n ih =>
    intro hist as t hh ha hk
    have hp : ((replay ms).next x hist ()).eval H = (ms.getD t.msgs.length [], []) := by
      show (OC.ret (ms.getD hist.length [])).eval H = _
      rw [hh]
      rfl
    have hv : (S.vnAt V ans x r (as ++ [ms.getD t.msgs.length []])).eval H =
        (.msg (ans (coinSeq r t.msgs.length)), []) := by
      have hlt : t.msgs.length + 1 < V.k x := by omega
      simp only [Server.vnAt, Server.IVStar, List.length_append, List.length_singleton, hlt, ↓reduceIte,
        List.range_succ, List.map_append, List.map_singleton, List.getLastD_concat, ha]
      rfl
    have hv' : ((S.vnAt V ans x r (as ++ [(((replay ms).next x hist ()).eval H).1])).eval H).1 =
        .msg (ans (coinSeq r t.msgs.length)) := by
      rw [hp]
      exact congrArg Prod.fst hv
    rw [stepWith_msg H (replay ms) x () (S.vnAt V ans x r) (n + 1) hist as t hv']
    simp only [hp, hv]
    refine (ih _ _ _ ?_ ?_ ?_).trans ?_
    · simp only [List.length_append, List.length_singleton, hh]
    · simp only [List.length_append, List.length_singleton, ha]
    · simp only [List.length_append, List.length_singleton]
      omega
    · simp only [List.length_append, List.length_singleton, List.append_nil, List.range'_succ, List.map_cons,
        List.append_assoc, List.cons_append, List.nil_append]

theorem run_ivstar (S : Server) (V : Verifier Coin) (ans : Coin → Msg) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin ((S.IVStar V ans).k x) → Coin) (hk : 0 < V.k x) :
    run H (replay ms) (S.IVStar V ans) x () r =
      { msgs := (List.range (V.k x - 1)).map (fun i => (ms.getD i [], Out.msg (ans (coinSeq r i)))) ++
          [(ms.getD (V.k x - 1) [], (S.decAt V ans H x ms r).1)],
        pLog := [], vLog := (S.decAt V ans H x ms r).2, out := (S.decAt V ans H x ms r).1 } := by
  obtain ⟨n, hn⟩ : ∃ n, V.k x = n + 1 := ⟨V.k x - 1, by omega⟩
  have h := S.stepWith_ivstar V ans H x ms r n [] [] ⟨[], [], [], .rej⟩ rfl rfl (by simp [hn])
  have e : run H (replay ms) (S.IVStar V ans) x () r =
      stepWith H (replay ms) x () (S.vnAt V ans x r) (n + 1) [] [] ⟨[], [], [], .rej⟩ := by
    show stepWith H (replay ms) x () (S.vnAt V ans x r) (V.k x) [] [] ⟨[], [], [], .rej⟩ = _
    rw [hn]
  rw [e, h]
  simp only [List.length_nil, List.nil_append, ← List.range_eq_range', Server.decAt, Server.vnAt, List.length_map,
    List.length_range, hn, Nat.add_sub_cancel]

theorem run_ivstar_zero (S : Server) (V : Verifier Coin) (ans : Coin → Msg) (H : Msg → Msg) (x : Msg)
    (ms : List Msg) (r : Fin ((S.IVStar V ans).k x) → Coin) (hk : V.k x = 0) :
    run H (replay ms) (S.IVStar V ans) x () r = ⟨[], [], [], .rej⟩ := by
  show stepWith H (replay ms) x () (S.vnAt V ans x r) (V.k x) [] [] ⟨[], [], [], .rej⟩ = _
  rw [hk]
  rfl

theorem opensRound_true (S : Server) (H : Msg → Msg) {a : Msg} {os : List (Msg × S.Salt)}
    (h : ((S.opensRound a os).eval H).1 = true) :
    a.length = S.width * os.length ∧ ∀ (i : ℕ) (hi : i < os.length),
      ((S.opens ((a.drop (S.width * i)).take S.width) os[i].1 os[i].2).eval H).1 = true ∧
        ∀ e ∈ ((S.opens ((a.drop (S.width * i)).take S.width) os[i].1 os[i].2).eval H).2,
          e ∈ ((S.opensRound a os).eval H).2 := by
  unfold Server.opensRound at h ⊢
  by_cases hl : a.length = S.width * os.length
  · rw [ite_eq_left hl] at h ⊢
    refine ⟨hl, fun i hi => ?_⟩
    have hm : S.opens ((a.drop (S.width * i)).take S.width) os[i].1 os[i].2 ∈
        os.mapIdx fun i o => S.opens ((a.drop (S.width * i)).take S.width) o.1 o.2 := by
      have := List.getElem_mem (l := os.mapIdx fun i o => S.opens ((a.drop (S.width * i)).take S.width) o.1 o.2)
        (by simpa using hi)
      rwa [List.getElem_mapIdx] at this
    exact allOC_true H _ h _ hm
  · rw [ite_eq_right hl] at h
    simp [OC.eval] at h

theorem agrees_true (V : Verifier Coin) (ans : Coin → Msg) (H : Msg → Msg) (x : Msg) {ms : List Msg}
    {cs : List Coin} (h : ((Server.agrees V ans x ms cs).eval H).1 = true) (j : ℕ) (hj : j < ms.length) :
    ((V.next x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
        (if j + 1 < ms.length then .msg (ans (cs.getD j default)) else .acc) ∧
      ∀ e ∈ ((V.next x (ms.take (j + 1)) (cs.take (j + 1))).eval H).2,
        e ∈ ((Server.agrees V ans x ms cs).eval H).2 := by
  have hm : (bindOC (V.next x (ms.take (j + 1)) (cs.take (j + 1))) fun b =>
      OC.ret (decide (b = if j + 1 < ms.length then .msg (ans (cs.getD j default)) else .acc))) ∈
      (List.range ms.length).map fun j => bindOC (V.next x (ms.take (j + 1)) (cs.take (j + 1))) fun b =>
        OC.ret (decide (b = if j + 1 < ms.length then .msg (ans (cs.getD j default)) else .acc)) :=
    List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩
  obtain ⟨h1, h2⟩ := allOC_true H _ h _ hm
  rw [eval_bindOC_full] at h1 h2
  simp only [OC.eval, decide_eq_true_eq, List.append_nil] at h1 h2
  exact ⟨h1, h2⟩

theorem ivstar_acc (S : Server) (V : Verifier Coin) (ans : Coin → Msg) (H : Msg → Msg) (x : Msg) (as : List Msg)
    (cs : List Coin) (hl : ¬ as.length < V.k x) (h : (((S.IVStar V ans).next x as cs).eval H).1 = .acc) :
    ∃ os last, S.final (as.getLastD []) = some (os, last) ∧ os.length + 1 = as.length ∧
      (∀ j < os.length, (os.getD j []).length = S.nAt x j) ∧
      ((allOC ((as.dropLast.zip os).map fun p => S.opensRound p.1 p.2)).eval H).1 = true ∧
      (∀ e ∈ ((allOC ((as.dropLast.zip os).map fun p => S.opensRound p.1 p.2)).eval H).2,
        e ∈ (((S.IVStar V ans).next x as cs).eval H).2) ∧
      ((Server.agrees V ans x (S.opened os last) cs).eval H).1 = true ∧
      ∀ e ∈ ((Server.agrees V ans x (S.opened os last) cs).eval H).2,
        e ∈ (((S.IVStar V ans).next x as cs).eval H).2 := by
  rcases hf : S.final (as.getLastD []) with _ | ⟨os, last⟩
  · exfalso
    have : (S.IVStar V ans).next x as cs = .ret .rej := by simp only [Server.IVStar, hl, ↓reduceIte, hf]
    rw [this] at h
    simp [OC.eval] at h
  · by_cases hc : os.length + 1 = as.length ∧ ∀ j < os.length, (os.getD j []).length = S.nAt x j
    · have e : (S.IVStar V ans).next x as cs =
          bindOC (allOC ((as.dropLast.zip os).map fun p => S.opensRound p.1 p.2)) fun ok =>
            if ok then
              bindOC (Server.agrees V ans x (S.opened os last) cs) fun acc => .ret (if acc then .acc else .rej)
            else .ret .rej := by
        simp only [Server.IVStar, hl, ↓reduceIte, hf, ite_eq_left hc]
      rw [e] at h ⊢
      rw [eval_bindOC_full] at h ⊢
      rcases hok : ((allOC ((as.dropLast.zip os).map fun p => S.opensRound p.1 p.2)).eval H).1 with _ | _
      · rw [hok] at h
        simp [OC.eval] at h
      · rw [hok] at h ⊢
        rw [ite_eq_left rfl] at h ⊢
        rw [eval_bindOC_full] at h ⊢
        rcases hacc : ((Server.agrees V ans x (S.opened os last) cs).eval H).1 with _ | _
        · rw [hacc] at h
          simp [OC.eval] at h
        · exact ⟨os, last, rfl, hc.1, hc.2, hok, fun e he => List.mem_append_left _ he, hacc,
            fun e he => List.mem_append_right _ (List.mem_append_left _ he)⟩
    · exfalso
      have : (S.IVStar V ans).next x as cs = .ret .rej := by
        simp only [Server.IVStar, hl, ↓reduceIte, hf, ite_eq_right hc]
      rw [this] at h
      simp [OC.eval] at h

theorem decAt_acc (S : Server) (V : Verifier Coin) (ans : Coin → Msg) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin ((S.IVStar V ans).k x) → Coin) (hk : 0 < V.k x) (h : (S.decAt V ans H x ms r).1 = .acc) :
    ∃ os last, S.final (ms.getD (V.k x - 1) []) = some (os, last) ∧ os.length + 1 = V.k x ∧
      (∀ j < os.length, (os.getD j []).length = S.nAt x j ∧
        ((S.opensRound (ms.getD j []) (os.getD j [])).eval H).1 = true ∧
        ∀ e ∈ ((S.opensRound (ms.getD j []) (os.getD j [])).eval H).2, e ∈ (S.decAt V ans H x ms r).2) ∧
      ((Server.agrees V ans x (S.opened os last) ((List.range (V.k x)).map (coinSeq r))).eval H).1 = true ∧
      ∀ e ∈ ((Server.agrees V ans x (S.opened os last) ((List.range (V.k x)).map (coinSeq r))).eval H).2,
        e ∈ (S.decAt V ans H x ms r).2 := by
  obtain ⟨n, hn⟩ : ∃ n, V.k x = n + 1 := ⟨V.k x - 1, by omega⟩
  have hl : ((List.range (V.k x)).map fun i => ms.getD i []).getLastD [] = ms.getD (V.k x - 1) [] := by
    rw [hn, List.range_succ, List.map_append, List.map_singleton, List.getLastD_concat, Nat.add_sub_cancel]
  have hd : ((List.range (V.k x)).map fun i => ms.getD i []).dropLast =
      (List.range (V.k x - 1)).map fun i => ms.getD i [] := by
    rw [hn, List.range_succ, List.map_append, List.map_singleton, List.dropLast_concat, Nat.add_sub_cancel]
  obtain ⟨os, last, hf, hlen, hn', hok, hokl, hag, hagl⟩ := S.ivstar_acc V ans H x _ _ (by simp) h
  rw [hl] at hf
  rw [hd] at hok hokl
  simp only [List.length_map, List.length_range] at hlen
  refine ⟨os, last, hf, hlen, fun j hj => ⟨hn' j hj, ?_⟩, hag, hagl⟩
  have hmem : S.opensRound (ms.getD j []) (os.getD j []) ∈
      (((List.range (V.k x - 1)).map fun i => ms.getD i []).zip os).map fun p => S.opensRound p.1 p.2 := by
    refine List.mem_map.2 ⟨(ms.getD j [], os.getD j []), List.mem_iff_getElem.2 ⟨j, by simp; omega, ?_⟩, rfl⟩
    simp [List.getElem?_eq_getElem hj]
  obtain ⟨h1, h2⟩ := allOC_true H _ hok _ hmem
  exact ⟨h1, fun e he => hokl e (h2 e he)⟩

theorem coinSeq_iv (S : Server) (V : Verifier Coin) (ans : Coin → Msg) (x : Msg)
    (r : Fin ((S.IVStar V ans).k x) → Coin) : @coinSeq Coin _ ((S.IVStar V ans).k x) r = @coinSeq Coin _ (V.k x) r :=
  rfl

def ansAt (V : Verifier Coin) (ans : Coin → Msg) (x : Msg) {k : ℕ} (r : Fin k → Coin) (j : ℕ) : Out :=
  if j + 1 < V.k x then .msg (ans (coinSeq r j)) else .acc

theorem accFacts (S : Server) (V : Verifier Coin) (ans : Coin → Msg) (H : Msg → Msg) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (r : Fin ((S.IVStar V ans).k x) → Coin)
    (hacc : (run H P (S.IVStar V ans) x ω r).out = .acc) :
    0 < V.k x ∧ ∃ os last,
      S.final (((run H P (S.IVStar V ans) x ω r).msgs.map Prod.fst).getD (V.k x - 1) []) = some (os, last) ∧
      os.length + 1 = V.k x ∧
      (∀ j < os.length, (os.getD j []).length = S.nAt x j ∧
        ((S.opensRound (((run H P (S.IVStar V ans) x ω r).msgs.map Prod.fst).getD j []) (os.getD j [])).eval H).1 =
          true ∧
        ∀ e ∈ ((S.opensRound (((run H P (S.IVStar V ans) x ω r).msgs.map Prod.fst).getD j [])
          (os.getD j [])).eval H).2, e ∈ (run H P (S.IVStar V ans) x ω r).vLog) ∧
      (∀ j < V.k x,
        ((V.next x ((List.range (j + 1)).map fun i => (S.opened os last).getD i [])
          ((List.range (j + 1)).map (coinSeq r))).eval H).1 = ansAt V ans x r j ∧
        ∀ e ∈ ((V.next x ((List.range (j + 1)).map fun i => (S.opened os last).getD i [])
          ((List.range (j + 1)).map (coinSeq r))).eval H).2, e ∈ (run H P (S.IVStar V ans) x ω r).vLog) ∧
      S.openView (run H P (S.IVStar V ans) x ω r).seen =
        ⟨(List.range (V.k x)).map fun j => ((S.opened os last).getD j [], ansAt V ans x r j), .acc⟩ := by
  obtain ⟨hm, hv, ho⟩ := run_replay_same H P (S.IVStar V ans) x ω r
  have hk : 0 < V.k x := by
    by_contra h0
    rw [S.run_ivstar_zero V ans H x _ r (by omega), hacc] at ho
    simp at ho
  refine ⟨hk, ?_⟩
  rw [S.run_ivstar V ans H x _ r hk] at hm hv ho
  dsimp only at hm hv ho
  have hd : (S.decAt V ans H x ((run H P (S.IVStar V ans) x ω r).msgs.map Prod.fst) r).1 = .acc := ho.trans hacc
  obtain ⟨os, last, hf, hlen, hrounds, hag, hagl⟩ := S.decAt_acc V ans H x _ r hk hd
  have hol : (S.opened os last).length = V.k x := by simp [Server.opened, hlen]
  refine ⟨os, last, hf, hlen, fun j hj => ?_, fun j hj => ?_, ?_⟩
  · obtain ⟨h1, h2, h3⟩ := hrounds j hj
    exact ⟨h1, h2, fun e he => hv ▸ h3 e he⟩
  · obtain ⟨h1, h2⟩ := agrees_true V ans H x hag j (by omega)
    rw [take_eq_range_getD _ (by omega), take_range_map _ (by omega)] at h1 h2
    refine ⟨?_, fun e he => hv ▸ hagl e (h2 e he)⟩
    rw [h1, hol]
    unfold Server.ansAt
    split_ifs with hj1
    · rw [getD_range_map _ _ hj]
    · rfl
  · have hlast : (run H P (S.IVStar V ans) x ω r).msgs.getLastD ([], .rej) =
        (((run H P (S.IVStar V ans) x ω r).msgs.map Prod.fst).getD (V.k x - 1) [], .acc) := by
      have e := congrArg (fun l : List (Msg × Out) => l.getLastD ([], .rej)) hm
      simp only [List.getLastD_concat, hd] at e
      exact e.symm
    have hsnd : (run H P (S.IVStar V ans) x ω r).msgs.map Prod.snd =
        (List.range (V.k x)).map (ansAt V ans x r) := by
      rw [← hm, hd]
      obtain ⟨n, hn⟩ : ∃ n, V.k x = n + 1 := ⟨V.k x - 1, by omega⟩
      have hr : List.range (V.k x) = List.range (V.k x - 1) ++ [V.k x - 1] := by
        rw [hn, Nat.add_sub_cancel, List.range_succ]
      rw [hr, List.map_append, List.map_append, List.map_map, List.map_singleton, List.map_singleton]
      congr 1
      · refine List.map_congr_left fun i hi => ?_
        rw [List.mem_range] at hi
        simp only [Function.comp, Server.ansAt, show i + 1 < V.k x by omega, ↓reduceIte]
      · simp only [Server.ansAt, show ¬ (V.k x - 1 + 1 < V.k x) by omega, ↓reduceIte]
    unfold Server.openView
    simp only [Transcript.seen]
    rw [hlast]
    simp only [hf, hsnd, zip_range_map _ _ hol, hacc]

theorem openView_out (S : Server) (v : Seen) : (S.openView v).out = v.out := by
  unfold Server.openView
  split <;> rfl

theorem openView_eq_of_acc (S : Server) {v : Seen} (h : v.out = .acc) :
    S.openView v = ⟨(S.openView v).msgs, .acc⟩ := by
  rw [← h, ← S.openView_out v]

theorem opened_getD_lt (S : Server) {os : List (List (Msg × S.Salt))} {last : Msg} {j : ℕ} (h : j < os.length) :
    (S.opened os last).getD j [] = S.join ((os.getD j []).map Prod.fst) := by
  have h' : j < (os.map fun o => S.join (o.map Prod.fst)).length := by simpa using h
  unfold Server.opened
  rw [List.getD_eq_getElem _ _ (by simp; omega), List.getElem_append_left h', List.getElem_map,
    List.getD_eq_getElem _ _ h]

end Server

end Claims.RecGeneric
