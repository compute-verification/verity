module

public import Proofs.Flock.Claims.Recursion.DeploySessions
public import Proofs.Flock.Claims.Recursion.VStarCross
public import Proofs.Flock.Claims.Sampling.Session

/-!
# A sampled verifier that runs its own rounds, then one Flock session (`SessionRuns`)

`sessionRuns_prefix` proves `SessionRuns` at a sampled record whose verifier, at each statement with units, runs `p`
rounds of its own and then one session of its Flock verifier (`PrefixAt`):

* the rounds before the `p`-th answer a message whatever the prover sends;
* the `p`-th answers `.msg []` where a check `ok` passes on the messages and coins so far, else refuses;
* past it, where `ok` passed, each round is the Flock verifier's at the statement `stmt` reads off the `p`-th message,
  on the messages and coins after the `p`-th, with `n` rounds.

Every unit's session is that one, on the rounds from `p` on. Where `ok` passed, the session's prover is the sampled
prover after the first `p` rounds (`sessProver`), and `RecDeploy.sim` makes its run the sampled run's from round `p`
on (`run_session`). Where `ok` refused, the sampled run ends refused at round `p`, and the session is the run of the
prover that sends `[2]`, which the Flock verifier refuses at once (`twoP`). The Flock verifier's facts this reads
(`FlockAt`) hold at flock-verify `--zk` against SHA-512 (`flockAt_sha512`).
-/

@[expose] public section

namespace SamplingDeploy

open Claims (Msg Out Seen Trace Prover Verifier stepWith run coinSeq)
open Claims.Sampling (Deploy nPU SessionRuns)
open FlockDeploy (Coin)
open RecDeploy (histOf sview)
open Claims.Recursion (sessProver sessCoins outMsg)

/-! ## Runs -/

section Steps

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)

/-- **A run that accepts a round ends accepted.** -/
theorem stepWith_acc_out (vn : List Msg → Claims.OC Out) :
    ∀ (f i : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (a : Msg),
      (stepWith H P x ω vn f hist as t).msgs[t.msgs.length + i]? = some (a, .acc) →
      (stepWith H P x ω vn f hist as t).out = .acc
  | 0, i, _, _, t, a, h => by simp [stepWith] at h
  | f + 1, i, hist, as, t, a, h => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg mm =>
      rw [RecDeploy.stepWith_succ_msg H P x ω vn f hist as t hvb] at h ⊢
      cases i with
      | zero =>
        obtain ⟨rest, hrest⟩ := (RecDeploy.stepWith_mono H P x ω vn f _ _ _).1
        rw [← hrest] at h
        simp at h
      | succ i =>
        exact stepWith_acc_out vn f i _ _ _ a (by simpa [Nat.add_assoc, Nat.add_comm 1 i] using h)
    | acc => rw [RecDeploy.stepWith_succ_acc H P x ω vn f hist as t hvb]
    | rej =>
      rw [RecDeploy.stepWith_succ_rej H P x ω vn f hist as t hvb] at h
      cases i <;> simp at h

/-- **A run goes past the next `m` rounds** where its verifier answers a message at each, whatever the prover sends,
and rounds are left. -/
theorem stepWith_long (vn : List Msg → Claims.OC Out) :
    ∀ (m f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), m < f →
      (∀ bs : List Msg, as.length < bs.length → bs.length ≤ as.length + m → ∃ mm, ((vn bs).eval H).1 = .msg mm) →
      t.msgs.length + m < (stepWith H P x ω vn f hist as t).msgs.length
  | 0, 0, _, _, _, h, _ => absurd h (Nat.lt_irrefl 0)
  | 0, f + 1, hist, as, t, _, _ => by
    simpa using RecDeploy.stepWith_length_succ H P x ω vn f hist as t
  | m + 1, 0, _, _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | m + 1, f + 1, hist, as, t, hmf, hv => by
    obtain ⟨mm, hmm⟩ := hv (as ++ [((P.next x hist ω).eval H).1]) (by simp) (by simp)
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    rw [hvb] at hmm
    simp only at hmm
    subst hmm
    rw [RecDeploy.stepWith_succ_msg H P x ω vn f hist as t hvb]
    have := stepWith_long vn m f (hist ++ [(((P.next x hist ω).eval H).1, mm)]) (as ++ [((P.next x hist ω).eval H).1])
      { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .msg mm)],
               pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo } (by omega)
      (fun bs h1 h2 => hv bs (by simp at h1; omega) (by simp at h2; omega))
    simp only [List.length_append, List.length_singleton] at this
    omega

/-- **A run with an accepted round accepts.** -/
theorem run_acc_of_mem {C : Type} [Inhabited C] (V : Verifier C) (r : Fin (V.k x) → C) {a : Msg}
    (h : (a, Out.acc) ∈ (run H P V x ω r).msgs) : (run H P V x ω r).out = .acc := by
  obtain ⟨i, hi, he⟩ := List.getElem_of_mem h
  have h2 : (run H P V x ω r).msgs[0 + i]? = some (a, .acc) := by
    rw [Nat.zero_add, List.getElem?_eq_getElem hi, he]
  exact stepWith_acc_out H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r))) (V.k x) i [] []
    ⟨[], [], [], .rej⟩ a h2

/-- **A run ends accepted or refused.** -/
theorem stepWith_out_dec (vn : List Msg → Claims.OC Out) :
    ∀ (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      (stepWith H P x ω vn f hist as t).out = .acc ∨ (stepWith H P x ω vn f hist as t).out = .rej
  | 0, _, _, _ => by simp [stepWith]
  | f + 1, hist, as, t => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg _ =>
      rw [RecDeploy.stepWith_succ_msg H P x ω vn f hist as t hvb]
      exact stepWith_out_dec vn f _ _ _
    | acc =>
      rw [RecDeploy.stepWith_succ_acc H P x ω vn f hist as t hvb]
      exact Or.inl rfl
    | rej =>
      rw [RecDeploy.stepWith_succ_rej H P x ω vn f hist as t hvb]
      exact Or.inr rfl

theorem run_out_dec {C : Type} [Inhabited C] (V : Verifier C) (r : Fin (V.k x) → C) :
    (run H P V x ω r).out = .acc ∨ (run H P V x ω r).out = .rej :=
  stepWith_out_dec H P x ω _ (V.k x) [] [] ⟨[], [], [], .rej⟩

end Steps

/-- The first `m` coins of two coin vectors that agree below `p ≥ m`. -/
theorem coins_agree {C : Type} [Inhabited C] {k : ℕ} {r r' : Fin k → C} {p : ℕ}
    (hr : ∀ t : Fin k, t.val < p → r' t = r t) {m : ℕ} (hm : m ≤ p) :
    (List.range m).map (coinSeq r') = (List.range m).map (coinSeq r) :=
  List.map_congr_left fun i hi => by
    rw [List.mem_range] at hi
    unfold coinSeq
    split_ifs with h
    · exact hr ⟨i, h⟩ (show i < p by omega)
    · rfl

/-- **Runs at coins that agree on the first `p` rounds have the same first `p` rounds.** -/
theorem run_take_agree {C : Type} [Inhabited C] (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (V : Verifier C)
    (x : Msg) (ω : Ω) (r r' : Fin (V.k x) → C) (p : ℕ) (hr : ∀ t : Fin (V.k x), t.val < p → r' t = r t) :
    (run H P V x ω r').msgs.take p = (run H P V x ω r).msgs.take p := by
  have := (RecDeploy.stepWith_agree H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r')))
    (fun as => V.next x as ((List.range as.length).map (coinSeq r))) (V.k x) p [] [] ⟨[], [], [], .rej⟩
    ⟨[], [], [], .rej⟩ rfl (fun bs hb => by
      simp only [List.length_nil, Nat.zero_add] at hb
      rw [coins_agree hr hb])).1
  simp only [List.length_nil, Nat.zero_add] at this
  exact this

/-! ## The verifier's shape -/

/-- **The Flock verifier a session runs**: it accepts only at its last round (one coin per message), answers no empty
message, has a round at every statement, and refuses the message `[2]` with no query. -/
structure FlockAt (H : Msg → Msg) (FV : Verifier Coin) : Prop where
  acc : ∀ xS as cs, cs.length = as.length → ((FV.next xS as cs).eval H).1 = .acc → as.length = FV.k xS
  msg : ∀ xS as cs m, ((FV.next xS as cs).eval H).1 = .msg m → m ≠ []
  pos : ∀ xS, 0 < FV.k xS
  two : ∀ xS cs, (FV.next xS [[2]] cs).eval H = (.rej, [])

/-- **A verifier that runs `p` rounds of its own, then one session of `FV`**, at `x`, with `n` rounds after the
`p`-th: the rounds before the `p`-th answer a message; the `p`-th answers `.msg []` where `ok` passes on the messages
and coins so far, else refuses; past it, where `ok` passed, each round is `FV`'s at the statement `stmt` reads off the
`p`-th message, on the messages and coins after the `p`-th, which has `n` rounds. -/
structure PrefixAt (H : Msg → Msg) (V FV : Verifier Coin) (x : Msg) (p n : ℕ) (ok : List Msg → List Coin → Bool)
    (stmt : Msg → Msg) : Prop where
  pos : 0 < p
  k : V.k x = p + n
  pre : ∀ as cs, 0 < as.length → as.length < p → cs.length = as.length →
    ∃ m, ((V.next x as cs).eval H).1 = .msg m
  gate : ∀ as cs, as.length = p → cs.length = p →
    ((V.next x as cs).eval H).1 = if ok as cs then .msg [] else .rej
  sess : ∀ as cs, p < as.length → cs.length = as.length → ok (as.take p) (cs.take p) = true →
    (V.next x as cs).eval H = (FV.next (stmt (as.getD (p - 1) [])) (as.drop p) (cs.drop p)).eval H
  nOk : ∀ as cs, ok as cs = true → FV.k (stmt (as.getD (p - 1) [])) = n

/-- The session's statement, read off the `p`-th of the first rounds `M`. -/
def stmtAt (stmt : Msg → Msg) (p : ℕ) (M : List (Msg × Out)) : Msg := stmt ((M.map Prod.fst).getD (p - 1) [])

/-- **A list whose first `p` entries are `L ++ [e]`** has `p` of them, `e` its `p`-th, and `e`'s message the `p`-th
of their messages. -/
theorem take_last {l L : List (Msg × Out)} {e : Msg × Out} {p : ℕ} (hp : 0 < p) (hL : L.length = p - 1)
    (h : l.take p = L ++ [e]) :
    (l.take p).length = p ∧ l[p - 1]? = some e ∧ ((l.take p).map Prod.fst).getD (p - 1) [] = e.1 := by
  have hg : (L ++ [e])[p - 1]? = some e := by
    rw [List.getElem?_append_right (by omega), hL, Nat.sub_self, List.getElem?_cons_zero]
  refine ⟨?_, ?_, ?_⟩
  · rw [h, List.length_append, hL, List.length_singleton]
    omega
  · rw [← List.getElem?_take_of_lt (l := l) (show p - 1 < p by omega), h, hg]
  · (rw [h, List.getD_eq_getElem?_getD, List.getElem?_map, hg]) <;> rfl

section Prefix

variable {H : Msg → Msg} {V FV : Verifier Coin} {x : Msg} {p n : ℕ} {ok : List Msg → List Coin → Bool}
  {stmt : Msg → Msg}

/-- **The run's first `p` rounds**: `p` of them, the `p`-th answered `.msg []` where `ok` passed on them and their
coins, else refused; where it refused, the run ends refused, and where it passed, the run goes on from them with `n`
rounds left. -/
theorem run_pre (hS : PrefixAt H V FV x p n ok stmt) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin) :
    ((run H P V x ω r).msgs.take p).length = p ∧
    (run H P V x ω r).msgs[p - 1]? = some ((((run H P V x ω r).msgs.take p).map Prod.fst).getD (p - 1) [],
      if ok (((run H P V x ω r).msgs.take p).map Prod.fst) ((List.range p).map (coinSeq r)) then .msg [] else .rej) ∧
    (ok (((run H P V x ω r).msgs.take p).map Prod.fst) ((List.range p).map (coinSeq r)) = false →
      (run H P V x ω r).out = .rej) ∧
    (ok (((run H P V x ω r).msgs.take p).map Prod.fst) ((List.range p).map (coinSeq r)) = true →
      ∃ t : Trace, t.msgs = (run H P V x ω r).msgs.take p ∧
        run H P V x ω r = stepWith H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r))) n
          (((run H P V x ω r).msgs.take p).map histOf) (((run H P V x ω r).msgs.take p).map Prod.fst) t) := by
  have hp := hS.pos
  have hrun : run H P V x ω r = stepWith H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r)))
      (V.k x) [] [] ⟨[], [], [], .rej⟩ := rfl
  have hlong := stepWith_long H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r))) (p - 1)
    (V.k x) [] [] ⟨[], [], [], .rej⟩ (by rw [hS.k]; omega)
    (fun bs h1 h2 => hS.pre bs _ h1 (by simp only [List.length_nil, Nat.zero_add] at h2; omega) (by simp))
  obtain ⟨t', -, ht'm, -, hR⟩ := RecDeploy.stepWith_reach H P x ω
    (fun as => V.next x as ((List.range as.length).map (coinSeq r))) (p - 1) (V.k x) [] [] ⟨[], [], [], .rej⟩ hlong
  rw [← hrun] at ht'm hR hlong
  simp only [List.length_nil, List.drop_zero, List.nil_append, Nat.zero_add] at ht'm hR hlong
  rw [show V.k x - (p - 1) = n + 1 by rw [hS.k]; omega, ← ht'm] at hR
  have hl1 : t'.msgs.length = p - 1 := by rw [ht'm, List.length_take]; omega
  generalize run H P V x ω r = R at hR ht'm hlong ⊢
  set pa := ((P.next x (t'.msgs.map histOf) ω).eval H).1 with hpa
  have hbl : (t'.msgs.map Prod.fst ++ [pa]).length = p := by simp [hl1]; omega
  have hg : ((V.next x (t'.msgs.map Prod.fst ++ [pa])
      ((List.range (t'.msgs.map Prod.fst ++ [pa]).length).map (coinSeq r))).eval H).1 =
      if ok (t'.msgs.map Prod.fst ++ [pa]) ((List.range p).map (coinSeq r)) then .msg [] else .rej := by
    rw [hS.gate (t'.msgs.map Prod.fst ++ [pa]) ((List.range (t'.msgs.map Prod.fst ++ [pa]).length).map (coinSeq r))
      hbl (by simp [hbl]), hbl]
  rcases hvb : (V.next x (t'.msgs.map Prod.fst ++ [pa])
      ((List.range (t'.msgs.map Prod.fst ++ [pa]).length).map (coinSeq r))).eval H with ⟨o, lo⟩
  rw [hvb] at hg
  simp only at hg
  cases hb : ok (t'.msgs.map Prod.fst ++ [pa]) ((List.range p).map (coinSeq r)) with
  | false =>
    rw [hb] at hg
    simp only [Bool.false_eq_true, ↓reduceIte] at hg
    subst hg
    have hR2 := hR.trans (RecDeploy.stepWith_succ_rej H P x ω
      (fun as => V.next x as ((List.range as.length).map (coinSeq r))) n (t'.msgs.map histOf) (t'.msgs.map Prod.fst)
      t' hvb)
    have hRm : R.msgs = t'.msgs ++ [(pa, Out.rej)] := congrArg Trace.msgs hR2
    have hRo : R.out = .rej := congrArg Trace.out hR2
    have htk : R.msgs.take p = t'.msgs ++ [(pa, Out.rej)] := by
      rw [hRm]
      exact List.take_of_length_le (by rw [List.length_append, hl1, List.length_singleton]; omega)
    obtain ⟨hlen, hget, hgd⟩ := take_last hp hl1 htk
    have hfst : (R.msgs.take p).map Prod.fst = t'.msgs.map Prod.fst ++ [pa] := by
      rw [htk]
      simp
    refine ⟨hlen, ?_, fun _ => hRo, fun h => ?_⟩
    · (rw [hget, hgd, hfst, hb]) <;> rfl
    · rw [hfst, hb] at h
      cases h
  | true =>
    rw [hb] at hg
    simp only [↓reduceIte] at hg
    subst hg
    have hR2 : R = stepWith H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r))) n
        (t'.msgs.map histOf ++ [(pa, [])]) (t'.msgs.map Prod.fst ++ [pa])
        { t' with msgs := t'.msgs ++ [(pa, .msg [])], pLog := t'.pLog ++ ((P.next x (t'.msgs.map histOf) ω).eval H).2,
                  vLog := t'.vLog ++ lo } :=
      hR.trans (RecDeploy.stepWith_succ_msg H P x ω
        (fun as => V.next x as ((List.range as.length).map (coinSeq r))) n (t'.msgs.map histOf) (t'.msgs.map Prod.fst)
        t' hvb)
    obtain ⟨rest, hrest⟩ := (RecDeploy.stepWith_mono H P x ω
      (fun as => V.next x as ((List.range as.length).map (coinSeq r))) n
      (t'.msgs.map histOf ++ [(pa, [])]) (t'.msgs.map Prod.fst ++ [pa])
      { t' with msgs := t'.msgs ++ [(pa, .msg [])], pLog := t'.pLog ++ ((P.next x (t'.msgs.map histOf) ω).eval H).2,
                vLog := t'.vLog ++ lo }).1
    have htk : R.msgs.take p = t'.msgs ++ [(pa, .msg [])] := by
      rw [hR2, ← hrest]
      exact List.take_left' (by simp [hl1]; omega)
    obtain ⟨hlen, hget, hgd⟩ := take_last hp hl1 htk
    have hfst : (R.msgs.take p).map Prod.fst = t'.msgs.map Prod.fst ++ [pa] := by
      rw [htk]
      simp
    refine ⟨hlen, ?_, fun h => ?_, fun _ =>
      ⟨{ t' with msgs := t'.msgs ++ [(pa, .msg [])],
                 pLog := t'.pLog ++ ((P.next x (t'.msgs.map histOf) ω).eval H).2, vLog := t'.vLog ++ lo },
        htk.symm, ?_⟩⟩
    · (rw [hget, hgd, hfst, hb]) <;> rfl
    · rw [hfst, hb] at h
      cases h
    · rw [htk, hR2]
      simp [histOf, outMsg]

/-- **After a passed `p`-th round the run is `FV`'s session**: at coins `r'` that agree with `r` on the first `p`
rounds, where `ok` passed at `r`, the session's prover (`sessProver`: `P` after the first `p` rounds) against `FV` at
the statement read off the `p`-th message, on the coins from round `p` on, has `sview`'s view of the run, and its
oracle calls are among the run's. -/
theorem run_session (hS : PrefixAt H V FV x p n ok stmt) (hF : FlockAt H FV) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r r' : Fin (V.k x) → Coin) (hr : ∀ t : Fin (V.k x), t.val < p → r' t = r t)
    (hok : ok (((run H P V x ω r).msgs.take p).map Prod.fst) ((List.range p).map (coinSeq r)) = true) :
    (run H (sessProver P x ω ((run H P V x ω r).msgs.take p)) FV (stmtAt stmt p ((run H P V x ω r).msgs.take p)) ()
        (sessCoins p r' (FV.k (stmtAt stmt p ((run H P V x ω r).msgs.take p))))).seen =
      sview true p n (run H P V x ω r').seen ∧
    ∀ l ∈ (run H (sessProver P x ω ((run H P V x ω r).msgs.take p)) FV
        (stmtAt stmt p ((run H P V x ω r).msgs.take p)) ()
        (sessCoins p r' (FV.k (stmtAt stmt p ((run H P V x ω r).msgs.take p))))).vLog,
      l ∈ (run H P V x ω r').vLog := by
  have hp := hS.pos
  have hM := run_take_agree H P V x ω r r' p hr
  rw [← hM]
  rw [← hM, ← coins_agree hr (le_refl p)] at hok
  obtain ⟨hlen, -, -, hdec⟩ := run_pre hS P ω r'
  obtain ⟨t, htm, hR⟩ := hdec hok
  generalize hMg : (run H P V x ω r').msgs.take p = M at hlen hok htm hR
  have hk : FV.k (stmtAt stmt p M) = n := hS.nOk _ _ hok
  have hml : (M.map Prod.fst).length = p := by simp [hlen]
  have hV' : ∀ bs : List Msg, 0 < bs.length → bs.length ≤ FV.k (stmtAt stmt p M) →
      (V.next x (M.map Prod.fst ++ bs) ((List.range (M.map Prod.fst ++ bs).length).map (coinSeq r'))).eval H =
        (RecDeploy.sessOut true (bs.length == FV.k (stmtAt stmt p M))
          ((FV.next (stmtAt stmt p M) bs ((List.range bs.length).map
            (coinSeq (sessCoins p r' (FV.k (stmtAt stmt p M)))))).eval H).1,
          ((FV.next (stmtAt stmt p M) bs ((List.range bs.length).map
            (coinSeq (sessCoins p r' (FV.k (stmtAt stmt p M)))))).eval H).2) := by
    intro bs h0 hbn
    have hl : (M.map Prod.fst ++ bs).length = p + bs.length := by simp [hml]
    rw [hS.sess _ _ (by rw [hl]; omega) (by simp) (by
      rw [List.take_left' hml, hl, RecDeploy.range_map_take _ (Nat.le_add_right p bs.length)]
      exact hok)]
    have hgd : (M.map Prod.fst ++ bs).getD (p - 1) [] = (M.map Prod.fst).getD (p - 1) [] := by
      simp only [List.getD_eq_getElem?_getD]
      rw [List.getElem?_append_left (show p - 1 < (M.map Prod.fst).length by omega)]
    have hcs : ((List.range (M.map Prod.fst ++ bs).length).map (coinSeq r')).drop p =
        (List.range bs.length).map (coinSeq (sessCoins p r' (FV.k (stmtAt stmt p M)))) := by
      rw [hl, RecDeploy.range_map_drop]
      exact List.map_congr_left fun i hi => by
        rw [List.mem_range] at hi
        rw [RecDeploy.coinSeq_sessCoins p r' _ (by omega)]
    rw [hgd, List.drop_left' hml, hcs]
    show (FV.next (stmtAt stmt p M) bs ((List.range bs.length).map
      (coinSeq (sessCoins p r' (FV.k (stmtAt stmt p M)))))).eval H = _
    rcases he : (FV.next (stmtAt stmt p M) bs ((List.range bs.length).map
      (coinSeq (sessCoins p r' (FV.k (stmtAt stmt p M)))))).eval H with ⟨o, lo⟩
    cases o with
    | msg m => rfl
    | rej => rfl
    | acc =>
      have hacc := hF.acc (stmtAt stmt p M) bs ((List.range bs.length).map
        (coinSeq (sessCoins p r' (FV.k (stmtAt stmt p M))))) (by simp) (by simp only [he])
      simp [RecDeploy.sessOut, hacc]
  have key := RecDeploy.sim H P x ω (sessProver P x ω M) (stmtAt stmt p M)
    (fun bs => FV.next (stmtAt stmt p M) bs ((List.range bs.length).map
      (coinSeq (sessCoins p r' (FV.k (stmtAt stmt p M))))))
    (fun as => V.next x as ((List.range as.length).map (coinSeq r'))) true (FV.k (stmtAt stmt p M)) 0
    (M.map histOf) (M.map Prod.fst) M (fun _ => rfl) hV'
    (fun bs h => hF.acc _ bs _ (by simp) h) (fun bs m h => hF.msg _ bs _ m h) (fun _ => rfl)
    (fun h => absurd h (Nat.lt_irrefl 0))
    (FV.k (stmtAt stmt p M)) [] [] ⟨[], [], [], .rej⟩ t (hF.pos _) (by simp) (by rw [htm]; simp) rfl rfl (by simp)
    (by simp)
  simp only [List.append_nil, Nat.add_zero, hlen] at key
  have hR' : stepWith H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r')))
      (FV.k (stmtAt stmt p M)) (M.map histOf) (M.map Prod.fst) t = run H P V x ω r' := by
    rw [hk]
    exact hR.symm
  rw [hR'] at key
  refine ⟨?_, key.2⟩
  rw [← hk]
  exact key.1

/-- **`sview`'s view of a run is its rounds from the `p`-th on, with its decision**: the run plays at most `p + n`
rounds, and where it refuses, none of its rounds accepted (`run_acc_of_mem`). -/
theorem sview_run (hS : PrefixAt H V FV x p n ok stmt) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin) :
    sview true p n (run H P V x ω r).seen = ⟨(run H P V x ω r).msgs.drop p, (run H P V x ω r).out⟩ := by
  have hl := Claims.Recursion.run_length_le H P V x ω r
  have hk := hS.k
  have ht : ((run H P V x ω r).msgs.drop p).take n = (run H P V x ω r).msgs.drop p :=
    List.take_of_length_le (by rw [List.length_drop]; omega)
  unfold sview
  simp only [Trace.seen, RecDeploy.fixLast_true, ht]
  rcases run_out_dec H P x ω V r with h | h
  · simp [h]
  · have hseg : RecDeploy.segOut ((run H P V x ω r).msgs.drop p) = .rej := by
      unfold RecDeploy.segOut
      rcases hlast : ((run H P V x ω r).msgs.drop p).getLast? with _ | ⟨a, o⟩
      · rfl
      · cases o with
        | acc =>
          have := run_acc_of_mem H P x ω V r (a := a) ((List.drop_subset _ _) (List.mem_of_getLast? hlast))
          rw [h] at this
          cases this
        | msg _ => rfl
        | rej => rfl
    simp [h, hseg]

end Prefix

/-! ## The refused session -/

/-- The prover that sends `[2]`. -/
def twoP : Prover Unit := ⟨fun _ _ _ => .ret [2]⟩

/-- **The prover that sends `[2]`** is refused at once with no query, by a Flock verifier as `FlockAt` reads it. -/
theorem twoP_run {H : Msg → Msg} {FV : Verifier Coin} (hF : FlockAt H FV) (xS : Msg) (cs : Fin (FV.k xS) → Coin) :
    (run H twoP FV xS () cs).seen = ⟨[([2], .rej)], .rej⟩ ∧ (run H twoP FV xS () cs).vLog = [] := by
  have key : ∀ f, 0 < f →
      (stepWith H twoP xS () (fun as => FV.next xS as ((List.range as.length).map (coinSeq cs))) f [] []
        ⟨[], [], [], .rej⟩).seen = ⟨[([2], .rej)], .rej⟩ ∧
      (stepWith H twoP xS () (fun as => FV.next xS as ((List.range as.length).map (coinSeq cs))) f [] []
        ⟨[], [], [], .rej⟩).vLog = [] := by
    intro f hf
    obtain ⟨g, rfl⟩ : ∃ g, f = g + 1 := ⟨f - 1, by omega⟩
    have h2 : (FV.next xS ([] ++ [((twoP.next xS [] ()).eval H).1])
        ((List.range ([] ++ [((twoP.next xS [] ()).eval H).1]).length).map (coinSeq cs))).eval H = (.rej, []) :=
      hF.two xS _
    rw [RecDeploy.stepWith_succ_rej H twoP xS () _ g [] [] ⟨[], [], [], .rej⟩ h2]
    exact ⟨rfl, rfl⟩
  exact key (FV.k xS) (hF.pos xS)

/-! ## `SessionRuns` -/

/-- **`SessionRuns` at a prefix-then-session record**: at each statement with units the sampled verifier runs `p x`
rounds of its own, then one session of its Flock verifier (`PrefixAt`), every unit's rounds are those from `p x` on,
and `session`, at each statement with units, reads a view whose `p x`-th round passed at the message `d` (with
`okD x d`, which a passed round has) as the statement `stmt x d` and the view's rounds from the `p x`-th on with its
decision (at a run, `sview`'s view: `sview_run`), and one whose `p x`-th round refused as `xS₀ x` and the view of the
prover that sends `[2]`. -/
theorem sessionRuns_prefix {Col T : Type} [DecidableEq Col] [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T)
    (hF : FlockAt H D.flock.V) (p n : Msg → ℕ) (ok : Msg → List Msg → List Coin → Bool) (stmt : Msg → Msg → Msg)
    (okD : Msg → Msg → Prop) (xS₀ : Msg → Msg)
    (hV : ∀ x, 0 < nPU D x → PrefixAt H D.V D.flock.V x (p x) (n x) (ok x) (stmt x))
    (hokD : ∀ x as cs, ok x as cs = true → okD x (as.getD (p x - 1) []))
    (hn₀ : ∀ x, D.flock.V.k (xS₀ x) = n x)
    (hrounds : ∀ x u (t : Fin (D.V.k x)), t ∈ D.rounds x u ↔ p x ≤ t.val)
    (hpass : ∀ x u, 0 < nPU D x → ∀ (v : Seen) d, v.msgs[p x - 1]? = some (d, .msg []) → okD x d →
      D.session x u v = some (stmt x d, ⟨v.msgs.drop (p x), v.out⟩))
    (hrefuse : ∀ x u, 0 < nPU D x → ∀ (v : Seen) d, v.msgs[p x - 1]? = some (d, .rej) →
      D.session x u v = some (xS₀ x, ⟨[([2], .rej)], .rej⟩)) :
    SessionRuns H D := by
  intro x Ω P ω r u hu
  have hx : 0 < nPU D x := by
    obtain ⟨q, -⟩ := hu
    exact Nat.lt_of_le_of_lt (Nat.zero_le _) q.isLt
  have hS := hV x hx
  have hagree : ∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, r' t = r t) →
      ∀ t : Fin (D.V.k x), t.val < p x → r' t = r t := fun r' h t ht =>
    h t (fun hm => absurd ((hrounds x u t).1 hm) (by omega))
  have hemb : ∀ xS, D.flock.V.k xS = n x → ∀ i : Fin (D.flock.V.k xS), p x + i.val < D.V.k x := fun xS hk i => by
    rw [hS.k]
    have := i.isLt
    omega
  have hround : ∀ xS (hk : D.flock.V.k xS = n x) (t : Fin (D.V.k x)),
      t ∈ D.rounds x u ↔ ∃ i : Fin (D.flock.V.k xS), (⟨p x + i.val, hemb xS hk i⟩ : Fin (D.V.k x)) = t := by
    intro xS hk t
    rw [hrounds x u t]
    constructor
    · intro ht
      have h1 := t.isLt
      have h2 := hS.k
      exact ⟨⟨t.val - p x, by omega⟩, Fin.ext (show p x + (t.val - p x) = t.val by omega)⟩
    · rintro ⟨i, rfl⟩
      exact Nat.le_add_right _ _
  have hmono : ∀ xS (hk : D.flock.V.k xS = n x),
      StrictMono fun i : Fin (D.flock.V.k xS) => (⟨p x + i.val, hemb xS hk i⟩ : Fin (D.V.k x)) :=
    fun xS hk i j hij => by
      show p x + i.val < p x + j.val
      have : i.val < j.val := hij
      omega
  have hbefore : ∀ xS (hk : D.flock.V.k xS = n x) (r' : Fin (D.V.k x) → Coin),
      (∀ t ∉ D.rounds x u, (∀ i : Fin (D.flock.V.k xS), t < ⟨p x + i.val, hemb xS hk i⟩) → r' t = r t) →
      ∀ t : Fin (D.V.k x), t.val < p x → r' t = r t := fun xS hk r' h t ht =>
    h t (fun hm => absurd ((hrounds x u t).1 hm) (by omega)) (fun i => by
      show t.val < p x + i.val
      omega)
  cases hok : ok x (((run H P D.V x ω r).msgs.take (p x)).map Prod.fst) ((List.range (p x)).map (coinSeq r)) with
  | false =>
    refine ⟨xS₀ x, fun i => ⟨p x + i.val, hemb _ (hn₀ x) i⟩, twoP, hmono _ (hn₀ x), hround _ (hn₀ x), ?_, ?_⟩
    · intro r' hr'
      have hr := hagree r' hr'
      obtain ⟨-, -, hrej', -⟩ := run_pre hS P ω r'
      rw [run_take_agree H P D.V x ω r r' (p x) hr, coins_agree hr (le_refl _)] at hrej'
      have h1 := hrej' hok
      have h2 := (twoP_run hF (xS₀ x) fun i => r' ⟨p x + i.val, hemb _ (hn₀ x) i⟩).1
      have h3 : (run H twoP D.flock.V (xS₀ x) () fun i => r' ⟨p x + i.val, hemb _ (hn₀ x) i⟩).out = .rej :=
        congrArg Seen.out h2
      rw [h1, h3]
    · intro r' hr'
      have hr := hbefore _ (hn₀ x) r' hr'
      obtain ⟨-, hlast', -, -⟩ := run_pre hS P ω r'
      rw [run_take_agree H P D.V x ω r r' (p x) hr, coins_agree hr (le_refl _), hok] at hlast'
      simp only [Bool.false_eq_true, ↓reduceIte] at hlast'
      obtain ⟨h2, h3⟩ := twoP_run hF (xS₀ x) fun i => r' ⟨p x + i.val, hemb _ (hn₀ x) i⟩
      refine ⟨?_, fun q hq => ?_⟩
      · rw [h2]
        exact hrefuse x u hx (run H P D.V x ω r').seen _ hlast'
      · rw [h3] at hq
        cases hq
  | true =>
    have hk : D.flock.V.k (stmtAt (stmt x) (p x) ((run H P D.V x ω r).msgs.take (p x))) = n x := hS.nOk _ _ hok
    have hcoins : ∀ r' : Fin (D.V.k x) → Coin,
        sessCoins (p x) r' (D.flock.V.k (stmtAt (stmt x) (p x) ((run H P D.V x ω r).msgs.take (p x)))) =
          fun i => r' ⟨p x + i.val, hemb _ hk i⟩ := by
      intro r'
      funext i
      show coinSeq r' (p x + i.val) = r' ⟨p x + i.val, hemb _ hk i⟩
      unfold coinSeq
      rw [dite_eq_left (hemb _ hk i)]
    refine ⟨stmtAt (stmt x) (p x) ((run H P D.V x ω r).msgs.take (p x)), fun i => ⟨p x + i.val, hemb _ hk i⟩,
      sessProver P x ω ((run H P D.V x ω r).msgs.take (p x)), hmono _ hk, hround _ hk, ?_, ?_⟩
    · intro r' hr'
      have hs := (run_session hS hF P ω r r' (hagree r' hr') hok).1
      rw [hcoins r'] at hs
      have ho := congrArg Seen.out hs
      simp only [sview, Trace.seen] at ho
      rw [ho]
      constructor
      · intro h
        simp [h]
      · intro h
        by_cases hacc : (run H P D.V x ω r').out = Out.acc
        · exact hacc
        · simp only [hacc, ↓reduceIte] at h
          unfold RecDeploy.segOut at h
          split at h
          · rename_i a hl
            apply run_acc_of_mem H P x ω D.V r' (a := a)
            have := List.mem_of_getLast? hl
            rw [RecDeploy.fixLast_true] at this
            exact (List.drop_subset _ _) ((List.take_subset _ _) this)
          · cases h
    · intro r' hr'
      have hr := hbefore _ hk r' hr'
      obtain ⟨hs, hlog⟩ := run_session hS hF P ω r r' hr hok
      rw [hcoins r'] at hs hlog
      refine ⟨?_, hlog⟩
      rw [hs, sview_run hS P ω r']
      obtain ⟨-, hlast', -, -⟩ := run_pre hS P ω r'
      rw [run_take_agree H P D.V x ω r r' (p x) hr, coins_agree hr (le_refl _), hok] at hlast'
      simp only [↓reduceIte] at hlast'
      exact hpass x u hx (run H P D.V x ω r').seen _ hlast' (hokD x _ _ hok)

/-! ## At flock-verify `--zk` -/

/-- **flock-verify `--zk` against SHA-512 is a session's Flock verifier**: it accepts only at its last round
(`next_acc_length_sha512`), answers a message only at a coin, with the coin's bytes (`next_msg_coin`, never empty:
`coin_bytes_ne_nil`), has a round, and refuses `[2]`, which is neither a coin's tag `0` nor `finalOf`'s tag `1`. -/
theorem flockAt_sha512 : FlockAt FlockDeploy.sha512 (FlockDeploy.V true) where
  acc _ _ _ hcs h := FlockDeploy.next_acc_length_sha512 true hcs h
  msg xS as cs m h := by
    obtain ⟨c, hc⟩ := RecDeploy.next_msg_coin FlockDeploy.sha512 h
    have h' : ((FlockDeploy.next true xS as cs).eval FlockDeploy.sha512).1 = .msg m := h
    rw [FlockDeploy.next_coin FlockDeploy.sha512 hc] at h'
    have hm : c.bytes = m := Out.msg.inj h'
    rw [← hm]
    exact RecDeploy.coin_bytes_ne_nil c
  pos xS := by
    show 0 < (match FlockDeploy.setupOf true xS with
      | some (_, spec, sts) => 2 + FlockDeploy.rounds true spec sts
      | none => 1)
    split
    · exact Nat.add_pos_left Nat.two_pos _
    · exact Nat.one_pos
  two xS cs := by
    apply FlockDeploy.next_rej
    unfold FlockDeploy.act
    split
    · rfl
    · simp [FlockDeploy.finalOf]

end SamplingDeploy
