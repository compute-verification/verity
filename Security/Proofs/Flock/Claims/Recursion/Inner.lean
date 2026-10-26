import Proofs.Flock.Claims.Recursion.OuterKS

/-!
# The inner stage: its code facts and binding

`KS_of` reads the inner protocol's KS at the served runs (`InnerKS`). `Recursion/InnerAbort.lean` derives it from the
inner protocol's KS against provers that may abort (`Inner.KSAbort`), at an inner prover that sends, at each inner-coin
prefix, the message most extractions open there; `E*`'s view agrees with that prover's run unless two extractions open
one commit string differently. This file states the code facts about the inner session that derivation reads and
proves its binding step. `Inner.KS` alone doesn't reach `InnerKS`: the outer sessions' coins also decide whether `E*`
sees an inner rerun accepted, which `Inner.KS` doesn't cover, and `Recursion/Gap.lean` gives a deployment with every
code fact here where `InnerKS` fails (`innerKS_gap`).

* Code facts (`Prop`s about `D`): the inner view `E_O`'s witnesses give on an accepted run is the inner verifier's run
  of its own messages at the compiled verifier's first `k_I` coins (`InnerViewRun`, `innerCoins`); the witnesses record
  the inner verifier's SHA-512 calls in that run, every recorded call a true evaluation (`CallsRecord`), and at each
  inner round the calls that open the run's commit string to the inner view's message, under the deployment's binding
  opening (`CallsOpen`, `Opening`; one call is `Opening.ofCall`); that inner view is accepted (`InnerAccepts`); no
  witnesses record only true evaluations (`CallsNil`); and the inner verifier's messages show its coins (`PublicCoin`).
* The weaker forms the chain reads, which V*'s parts give at the deployed readers: every recorded call a true
  evaluation (`CallsTrue`, `CallsRecord`'s second half); the inner verifier's calls recorded, or two recorded calls a
  collision (`CallsLogColl`, its first half off a collision: where two parts hold different messages at a round, the
  run's calls on the readers' message aren't the other part's); and an opening at each round the inner view plays
  (`CallsOpenPlayed`: an inner view that rejects early has message `[]` at the rounds it doesn't play, which no part
  opens). `CallsRecord` gives the first two (`CallsRecord.toTrue`, `CallsRecord.toLogColl`).
* Binding: two inner views that differ at round `j` first differ at a round both play (`run_replay_plays`), where both
  open their message to the runs' one commit string: a collision among their recorded calls, at a fork at `j` one of
  `Be`'s (`forkColl_of_opening`). So the inner rounds' differing openings cost at most `k(x)·Pr[Be]`
  (`sum_opensDiffer_le`).
-/

namespace Claims.Recursion

open Claims

/-! ## Openings -/

/-- **How the deployment opens a commit string**: `opens a m cs` when the SHA-512 evaluations `cs` open the commit
string `a` to the message `m`. It binds: two openings of one commit string to different messages, by true evaluations
of any `H`, hold two queries with one answer. -/
structure Opening where
  opens : Msg → Msg → List (Msg × Msg) → Prop
  bind : ∀ (H : Msg → Msg) (a m m' : Msg) (cs cs' : List (Msg × Msg)), opens a m cs → opens a m' cs' → m ≠ m' →
    (∀ e ∈ cs ++ cs', H e.1 = e.2) → ∃ q ∈ (cs ++ cs').map Prod.fst, ∃ q' ∈ (cs ++ cs').map Prod.fst, Collision H q q'

/-- **One call opens**: an evaluation answers the commit string at a query that `f` reads as the message. -/
def Opening.ofCall (f : Msg → Msg) : Opening where
  opens a m cs := ∃ q, (q, a) ∈ cs ∧ f q = m
  bind H a m m' cs cs' := by
    rintro ⟨q, hq, rfl⟩ ⟨q', hq', rfl⟩ hne hall
    exact ⟨q, List.mem_map.2 ⟨(q, a), List.mem_append_left _ hq, rfl⟩,
      q', List.mem_map.2 ⟨(q', a), List.mem_append_right _ hq', rfl⟩, fun h => hne (congrArg f h),
      (hall _ (List.mem_append_left _ hq)).trans (hall _ (List.mem_append_right _ hq')).symm⟩

/-! ## The inner session's code facts -/

section CodeFacts

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `E_O` gives the witnesses `ws` on an accepted run at the tape `ω` and coins `r`, at its coins `c`. -/
def EOGives (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit) : Prop :=
  (run H P D.V x ω r).out = .acc ∧ ((EO D I DO decO).eval H P x ω r c).1 = some ws

/-- The inner coins: the compiled verifier's first `k_I`, whose rounds are the inner session's. -/
def innerCoins (D : Deploy Coin) (I : Inner Coin W) (x : Msg) (r : Fin (D.V.k x) → Coin) :
    Fin (I.V.k (xI D x)) → Coin :=
  fun i => coinSeq r i.val

/-- The inner view's run: the inner verifier's run, at the inner coins, of the prover that sends the inner view's
messages. -/
def innerRun (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (x : Msg) (r : Fin (D.V.k x) → Coin)
    (ws : List Flock.Wit) : Trace :=
  run H (replay ((D.innerView H x r ws).msgs.map Prod.fst)) I.V (xI D x) () (innerCoins D I x r)

/-- **The inner view at `r` and `E_O`'s witnesses is the inner verifier's run** (a code fact): wherever `E_O` gives
witnesses on an accepted run, the inner view read off them is `I.V`'s run of its own messages at the compiled
verifier's first `k_I` coins, so its answers and its decision are the inner verifier's. -/
def InnerViewRun (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    D.innerView H x r ws = (innerRun H D I x r ws).seen

/-- **`D.calls` records the inner verifier's SHA-512 calls** (a code fact): wherever `E_O` gives witnesses on an
accepted run, `I.V`'s calls in the inner view's run are among the calls the witnesses record, and every recorded call
is a true evaluation. -/
def CallsRecord (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    (∀ e ∈ (innerRun H D I x r ws).vLog, e ∈ D.calls x ws) ∧ ∀ e ∈ D.calls x ws, H e.1 = e.2

/-- **`D.calls` records each inner message's opening** (a code fact): wherever `E_O` gives witnesses on an accepted
run, at each inner round `j` some recorded calls open the run's message `j`, its commit string, to the inner view's
message `j` under `opening`. -/
def CallsOpen (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (opening : Opening) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    ∀ j < I.V.k (xI D x), ∃ cs, cs ⊆ D.calls x ws ∧
      opening.opens (((run H P D.V x ω r).msgs.map Prod.fst).getD j [])
        (((D.innerView H x r ws).msgs.map Prod.fst).getD j []) cs

/-- **Every recorded call is a true evaluation** (`CallsRecord`'s second half, a code fact): wherever `E_O` gives
witnesses on an accepted run, every call they record is a true evaluation of `H`. -/
def CallsTrue (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    ∀ e ∈ D.calls x ws, H e.1 = e.2

/-- **`CallsRecord`'s first half off a collision** (a code fact): wherever `E_O` gives witnesses on an accepted run,
the inner verifier's calls in the inner view's run are among the calls the witnesses record, or two recorded calls are
a collision. Where two parts of the witnesses hold different messages at one round, the run's calls on the message the
readers take aren't the other part's, so the first half alone holds only off a collision. -/
def CallsLogColl (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    (∀ e ∈ (innerRun H D I x r ws).vLog, e ∈ D.calls x ws) ∨
      ∃ u v, findColl (D.calls x ws) = some (u, v) ∧ Collision H u v

/-- **`CallsLogColl` over the calls a finder searches at a draw** (a code fact): wherever `E_O` gives witnesses `ws` on
an accepted run at the coins `r`, the inner verifier's calls in the inner view's run are among `cl x r ws`, or two of
those collide. `CallsLogColl` is it at the recorded calls; `CallsLogCollAt` at `Deploy.callsAt`, which searches the
inner verifier's own calls at the draw's coins too. -/
def CallsLogCollBy (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (cl : (x : Msg) → (Fin (D.V.k x) → Coin) → List Flock.Wit → List (Msg × Msg)) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    (∀ e ∈ (innerRun H D I x r ws).vLog, e ∈ cl x r ws) ∨
      ∃ u v, findColl (cl x r ws) = some (u, v) ∧ Collision H u v

/-- **`D.calls` records each played inner message's opening** (`CallsOpen` at the rounds the inner view plays, a code
fact): wherever `E_O` gives witnesses on an accepted run, at each round the inner view plays, some recorded calls open
the run's message there, its commit string, to the inner view's message there under `opening`. An inner view that
stops early has message `[]` at the rounds it doesn't play, which no recorded call need open. -/
def CallsOpenPlayed (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (opening : Opening) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    ∀ j < (D.innerView H x r ws).msgs.length, ∃ cs, cs ⊆ D.calls x ws ∧
      opening.opens (((run H P D.V x ω r).msgs.map Prod.fst).getD j [])
        (((D.innerView H x r ws).msgs.map Prod.fst).getD j []) cs

theorem CallsRecord.toTrue {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (h : CallsRecord H D I DO) : CallsTrue H D I DO := fun decO x Ω P ω r c ws hg => (h decO x Ω P ω r c ws hg).2

theorem CallsRecord.toLogColl {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (h : CallsRecord H D I DO) : CallsLogColl H D I DO := fun decO x Ω P ω r c ws hg =>
  Or.inl (h decO x Ω P ω r c ws hg).1

/-- **The compiled verifier accepts only where the inner view does** (a code fact): wherever `E_O` gives witnesses on
an accepted run, the inner view read off them is accepted. -/
def InnerAccepts (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    (D.innerView H x r ws).out = .acc

/-- **The inner verifier is public-coin** (a code fact about `I`): its message at a round shows that round's coin,
which `decC` reads off it. -/
def PublicCoin (H : Msg → Msg) (I : Inner Coin W) (decC : Msg → Msg → Coin) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (c : Coin) (m : Msg), as.length = cs.length + 1 →
    ((I.V.next x as (cs ++ [c])).eval H).1 = .msg m → decC x m = c

/-- **No witnesses record only true evaluations** (a code fact): the calls `D.calls` reads off no witnesses, what `Be`
reads where `E_O` gives up, are true evaluations of `H`. -/
def CallsNil (H : Msg → Msg) (D : Deploy Coin) : Prop := ∀ x, ∀ e ∈ D.calls x [], H e.1 = e.2

end CodeFacts

/-! ## Binding -/

/-- **`findColl` finds a collision wherever true evaluations hold one**: two of their queries with one answer. -/
theorem findColl_of_coll (H : Msg → Msg) {l : List (Msg × Msg)} (hl : ∀ e ∈ l, H e.1 = e.2) {q q' : Msg}
    (hq : q ∈ l.map Prod.fst) (hq' : q' ∈ l.map Prod.fst) (hc : Collision H q q') :
    ∃ u v, findColl l = some (u, v) ∧ Collision H u v := by
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (findColl_isSome_mem H l hl q q' hq hq' hc)
  exact ⟨u, v, huv, findColl_spec H _ hl u v huv⟩

/-! ## A run's prover message at round `j` reads only the coins before it -/

/-- A round's prover messages so far are the history's and the prover's next one, whatever the verifier answers. -/
theorem stepWith_succ_fst_take {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out)
    (N : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) :
    ((stepWith H P x ω vn (N + 1) hist as t).msgs.map Prod.fst).take (t.msgs.length + 1) =
      t.msgs.map Prod.fst ++ [((P.next x hist ω).eval H).1] := by
  have hp : t.msgs ++ [(((P.next x hist ω).eval H).1, ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1)] <+:
      (stepWith H P x ω vn (N + 1) hist as t).msgs := by
    simp only [stepWith]
    split
    · exact stepWith_prefix H P x ω vn N _ _ { t with msgs := _, pLog := _, vLog := _ }
    · exact List.prefix_refl _
    · exact List.prefix_refl _
  obtain ⟨rest, h⟩ := hp
  rw [← h]
  simp [List.take_append]

theorem stepWith_fst_take {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn vn' : List Msg → OC Out) :
    ∀ (N n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (L : ℕ), L = t.msgs.length + n →
      (∀ bs : List Msg, bs.length ≤ as.length + n → vn bs = vn' bs) →
      ((stepWith H P x ω vn N hist as t).msgs.map Prod.fst).take (L + 1) =
        ((stepWith H P x ω vn' N hist as t).msgs.map Prod.fst).take (L + 1)
  | 0, _, _, _, _, _, _, _ => rfl
  | N + 1, 0, hist, as, t, L, hL, _ => by
    rw [hL, Nat.add_zero, stepWith_succ_fst_take, stepWith_succ_fst_take]
  | N + 1, n + 1, hist, as, t, L, hL, h => by
    have hv : vn (as ++ [((P.next x hist ω).eval H).1]) = vn' (as ++ [((P.next x hist ω).eval H).1]) :=
      h _ (by simp)
    simp only [stepWith, hv]
    split
    · exact stepWith_fst_take H P x ω vn vn' N n _ _ _ L
        (by simp only [List.length_append, List.length_singleton]; omega)
        (fun bs hb => h bs (by simp only [List.length_append, List.length_singleton] at hb; omega))
    · rfl
    · rfl

/-- **A run's prover message at round `j` reads only its first `j` coins.** -/
theorem run_fst_getD {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (ω : Ω) (j : ℕ) {r r' : Fin (V.k x) → Coin} (h : ∀ i : Fin (V.k x), i.val < j → r i = r' i) :
    ((run H P V x ω r).msgs.map Prod.fst).getD j [] = ((run H P V x ω r').msgs.map Prod.fst).getD j [] := by
  have := stepWith_fst_take H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r)))
    (fun as => V.next x as ((List.range as.length).map (coinSeq r'))) (V.k x) j [] [] ⟨[], [], [], .rej⟩ j
    (by simp) (fun bs hb => by
      simp only [List.length_nil, Nat.zero_add] at hb
      congr 1
      refine List.map_congr_left fun i hi => ?_
      rw [List.mem_range] at hi
      unfold coinSeq
      split_ifs with hk
      · exact h ⟨i, hk⟩ (by simp only; omega)
      · rfl)
  have e : ∀ l : List Msg, l.getD j [] = (l.take (j + 1)).getD j [] := fun l => by
    simp only [List.getD_eq_getElem?_getD, List.getElem?_take, Nat.lt_succ_self, ↓reduceIte]
  rw [e, e (List.map _ _)]
  exact congrArg (fun l => l.getD j []) this

/-! ## Two runs play a round alike when they agree before it -/

/-- A round the verifier answers with a message: the rounds go on from the history and the trace it extends. -/
theorem stepWith_msg {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) (n : ℕ)
    (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) {m : Msg}
    (hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 = .msg m) :
    stepWith H P x ω vn (n + 1) hist as t =
      stepWith H P x ω vn n (hist ++ [(((P.next x hist ω).eval H).1, m)]) (as ++ [((P.next x hist ω).eval H).1])
        { t with
          msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1)],
          pLog := t.pLog ++ ((P.next x hist ω).eval H).2,
          vLog := t.vLog ++ ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).2 } := by
  simp only [stepWith]
  rw [hv]

/-- A run plays at most its rounds. -/
theorem stepWith_length_le {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      (stepWith H P x ω vn n hist as t).msgs.length ≤ t.msgs.length + n
  | 0, _, _, _ => le_refl _
  | n + 1, hist, as, t => by
    rcases hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with m | _ | _
    · rw [stepWith_msg H P x ω vn n hist as t hv]
      refine (stepWith_length_le H P x ω vn n _ _ _).trans ?_
      simp only [List.length_append, List.length_singleton]
      omega
    · simp [stepWith, hv]
    · simp [stepWith, hv]

/-- A run with a round left plays one. -/
theorem stepWith_length_gt {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out)
    {n : ℕ} (hn : 0 < n) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) :
    t.msgs.length < (stepWith H P x ω vn n hist as t).msgs.length := by
  obtain ⟨n, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  have := congrArg List.length (stepWith_succ_fst_take H P x ω vn n hist as t)
  simp only [List.length_take, List.length_map, List.length_append, List.length_singleton] at this
  omega

/-- A round before a run's last is answered with a message. -/
theorem stepWith_msg_of_lt {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (i : ℕ), t.msgs.length ≤ i →
      i + 1 < (stepWith H P x ω vn n hist as t).msgs.length →
        ∃ m, ((stepWith H P x ω vn n hist as t).msgs[i]?).map Prod.snd = some (.msg m)
  | 0, _, _, t, i, hi, h => by simp only [stepWith] at h; omega
  | n + 1, hist, as, t, i, hi, h => by
    rcases hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with m | _ | _
    · rw [stepWith_msg H P x ω vn n hist as t hv] at h ⊢
      rcases Nat.lt_or_ge i (t.msgs.length + 1) with hlt | hge
      · obtain rfl : i = t.msgs.length := by omega
        obtain ⟨l, hl⟩ := stepWith_prefix H P x ω vn n _ _ _
        refine ⟨m, ?_⟩
        rw [← hl]
        simp [hv]
      · exact stepWith_msg_of_lt H P x ω vn n _ _ _ i (by simpa using hge) h
    · simp only [stepWith, hv, List.length_append, List.length_singleton] at h
      omega
    · simp only [stepWith, hv, List.length_append, List.length_singleton] at h
      omega

/-- A run that stops before its rounds run out is decided at its last round. -/
theorem stepWith_last_of_lt {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      (stepWith H P x ω vn n hist as t).msgs.length < t.msgs.length + n →
        ∀ m, ((stepWith H P x ω vn n hist as t).msgs.getLast?).map Prod.snd ≠ some (.msg m)
  | 0, _, _, t, h => by simp only [stepWith] at h; omega
  | n + 1, hist, as, t, h => by
    rcases hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with m | _ | _
    · rw [stepWith_msg H P x ω vn n hist as t hv] at h ⊢
      exact stepWith_last_of_lt H P x ω vn n _ _ _ (by simp only [List.length_append, List.length_singleton]; omega)
    · simp [stepWith, hv]
    · simp [stepWith, hv]

/-- **Two replayed runs agree on their first `N` rounds** where their messages agree before round `N` and their
verifiers answer alike on up to `N` messages. -/
theorem stepWith_replay_take (H : Msg → Msg) (x : Msg) (vn vn' : List Msg → OC Out) (ms ms' : List Msg) (N : ℕ) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), hist.length = t.msgs.length →
      as.length = t.msgs.length → (∀ i, t.msgs.length ≤ i → i < N → ms.getD i [] = ms'.getD i []) →
      (∀ bs : List Msg, bs.length ≤ N → vn bs = vn' bs) →
      (stepWith H (replay ms) x () vn n hist as t).msgs.take N =
        (stepWith H (replay ms') x () vn' n hist as t).msgs.take N
  | 0, _, _, _, _, _, _, _ => rfl
  | n + 1, hist, as, t, hl, ha, hm, hv => by
    rcases Nat.lt_or_ge t.msgs.length N with hN | hN
    · have e1 : ((replay ms').next x hist ()).eval H = ((replay ms).next x hist ()).eval H := by
        simp only [replay, OC.eval, hm hist.length hl.ge (hl ▸ hN)]
      have e2 : vn' (as ++ [(((replay ms).next x hist ()).eval H).1]) =
          vn (as ++ [(((replay ms).next x hist ()).eval H).1]) :=
        (hv _ (by simp only [List.length_append, List.length_singleton]; omega)).symm
      simp only [stepWith]
      rw [e1, e2]
      split
      · exact stepWith_replay_take H x vn vn' ms ms' N n _ _ _ (by simp [hl]) (by simp [ha])
          (fun i hi1 hi2 => hm i (by simp only [List.length_append, List.length_singleton] at hi1; omega) hi2) hv
      · rfl
      · rfl
    · have key : ∀ (P : Prover Unit) (v : List Msg → OC Out),
          (stepWith H P x () v (n + 1) hist as t).msgs.take N = t.msgs.take N := fun P v => by
        obtain ⟨l, hl'⟩ := stepWith_prefix H P x () v (n + 1) hist as t
        rw [← hl', List.take_append_of_le_length hN]
      rw [key, key]

/-- **Two replayed runs agree on their first `N` rounds** where their messages and their coins agree before `N`. -/
theorem run_replay_take {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg)
    (ms ms' : List Msg) (c c' : Fin (V.k x) → Coin) (N : ℕ) (hm : ∀ i < N, ms.getD i [] = ms'.getD i [])
    (hc : ∀ i < N, coinSeq c i = coinSeq c' i) :
    (run H (replay ms) V x () c).msgs.take N = (run H (replay ms') V x () c').msgs.take N :=
  stepWith_replay_take H x _ _ ms ms' N (V.k x) [] [] _ rfl rfl (fun i _ hi => hm i hi) fun bs hb => by
    congr 1
    exact List.map_congr_left fun i hi => hc i (by rw [List.mem_range] at hi; omega)

/-- **Two replayed runs that agree before round `j` play round `j` alike**: a run plays round `j` unless it decided
before it or ran out of rounds, and both are read off the rounds before `j`. -/
theorem run_replay_plays {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg)
    (ms ms' : List Msg) (c c' : Fin (V.k x) → Coin) (j : ℕ) (hm : ∀ i < j, ms.getD i [] = ms'.getD i [])
    (hc : ∀ i < j, coinSeq c i = coinSeq c' i) (hj : j < (run H (replay ms) V x () c).msgs.length) :
    j < (run H (replay ms') V x () c').msgs.length := by
  by_contra hn
  push Not at hn
  have ht := run_replay_take H V x ms ms' c c' j hm hc
  rw [List.take_of_length_le hn] at ht
  have hlen : (run H (replay ms') V x () c').msgs.length = j := by
    rw [← ht, List.length_take]
    omega
  have hk : j < V.k x := by
    have := stepWith_length_le H (replay ms) x ()
      (fun as => V.next x as ((List.range as.length).map (coinSeq c))) (V.k x) [] [] ⟨[], [], [], .rej⟩
    simp only [List.length_nil, Nat.zero_add] at this
    exact lt_of_lt_of_le hj this
  have hlast := stepWith_last_of_lt H (replay ms') x ()
    (fun as => V.next x as ((List.range as.length).map (coinSeq c'))) (V.k x) [] [] ⟨[], [], [], .rej⟩
    (by simp only [List.length_nil, Nat.zero_add]; exact hlen ▸ hk)
  rcases j with _ | j
  · have := stepWith_length_gt H (replay ms') x ()
      (fun as => V.next x as ((List.range as.length).map (coinSeq c'))) hk [] [] ⟨[], [], [], .rej⟩
    simp only [List.length_nil] at this
    unfold run at hlen
    omega
  · obtain ⟨m, hm'⟩ := stepWith_msg_of_lt H (replay ms) x ()
      (fun as => V.next x as ((List.range as.length).map (coinSeq c))) (V.k x) [] [] ⟨[], [], [], .rej⟩ j
      (Nat.zero_le _) hj
    have hg : (run H (replay ms') V x () c').msgs.getLast? = (run H (replay ms) V x () c).msgs[j]? := by
      rw [List.getLast?_eq_getElem?, hlen, Nat.add_sub_cancel, ← ht, List.getElem?_take]
      simp
    exact hlast m (by unfold run at hg; rw [hg]; exact hm')

/-- A run plays at most its verifier's rounds. -/
theorem run_length_le {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (ω : Ω) (r : Fin (V.k x) → Coin) : (run H P V x ω r).msgs.length ≤ V.k x := by
  have := stepWith_length_le H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r))) (V.k x) [] []
    ⟨[], [], [], .rej⟩
  simp only [List.length_nil, Nat.zero_add] at this
  exact this

/-! ## Differing openings are `Be`'s collisions -/

section Forks

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **Two extractions open inner round `j` differently**: at a draw `(ω, r, d, c, c')` of `Be`'s fork at `j`, `E_O`
gives witnesses on both accepted runs, and the inner views read off them differ at message `j`. -/
def OpensDiffer (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (j : ℕ)
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x) :
    Prop :=
  ∃ ws ws', EOGives H D I DO decO P x t.1 t.2.1 t.2.2.2.1 ws ∧
    EOGives H D I DO decO P x t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' ∧
    ((D.innerView H x t.2.1 ws).msgs.map Prod.fst).getD j [] ≠
      ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst).getD j []

/-- **Binding at a fork, at the played rounds**: two inner views that differ at round `j` first differ at some round
`i ≤ j`. Both play round `i` (`run_replay_plays`: each is the inner verifier's run of its own messages, `InnerViewRun`,
and the two agree on their messages before `i` and on their coins before `j`), so both open their message `i` to the
runs' message `i` (`CallsOpenPlayed`), which is one commit string since the runs share their coins before `j`
(`run_fst_getD`): a collision among the calls both record, which `Be` finds at its fork at `j`. -/
theorem forkColl_of_opening {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} (hV : InnerViewRun H D I DO) (hO : CallsOpenPlayed H D I DO opening)
    (hT : CallsTrue H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} (P : Prover Ω) {j : ℕ}
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x)
    (h : OpensDiffer H D I DO decO P x j t) : forkColl H D I DO decO P x j t := by
  classical
  obtain ⟨ws, ws', hg, hg', hd⟩ := h
  have hex : ∃ i, ((D.innerView H x t.2.1 ws).msgs.map Prod.fst).getD i [] ≠
      ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst).getD i [] := ⟨j, hd⟩
  have hi := Nat.find_spec hex
  have hmin : ∀ i < Nat.find hex, ((D.innerView H x t.2.1 ws).msgs.map Prod.fst).getD i [] =
      ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst).getD i [] :=
    fun i h => not_not.1 (Nat.find_min hex h)
  have hij : Nat.find hex ≤ j := Nat.find_min' hex hd
  generalize Nat.find hex = i at hi hmin hij
  have hc : ∀ i' < i, coinSeq (innerCoins D I x t.2.1) i' =
      coinSeq (innerCoins D I x (fork j t.2.1 t.2.2.1)) i' := fun i' hi' => by
    unfold coinSeq innerCoins coinSeq
    split_ifs with h1 h2
    · exact fork_agree j t.2.1 t.2.2.1 ⟨i', h2⟩ (by simp only; omega)
    · rfl
    · rfl
  have hv := congrArg Seen.msgs (hV decO x Ω P t.1 t.2.1 _ ws hg)
  have hv' := congrArg Seen.msgs (hV decO x Ω P t.1 (fork j t.2.1 t.2.2.1) _ ws' hg')
  simp only [innerRun, Trace.seen] at hv hv'
  have hboth : i < (D.innerView H x t.2.1 ws).msgs.length ∧
      i < (D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.length := by
    by_cases h1 : i < (D.innerView H x t.2.1 ws).msgs.length
    · refine ⟨h1, ?_⟩
      rw [hv']
      exact run_replay_plays H I.V (xI D x) _ _ _ _ _ hmin hc (hv ▸ h1)
    · by_cases h2 : i < (D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.length
      · refine absurd ?_ h1
        rw [hv]
        exact run_replay_plays H I.V (xI D x) _ _ _ _ _ (fun i h => (hmin i h).symm) (fun i h => (hc i h).symm)
          (hv' ▸ h2)
      · refine absurd ?_ hi
        push Not at h1 h2
        rw [List.getD_eq_default _ _ (by simpa using h1), List.getD_eq_default _ _ (by simpa using h2)]
  obtain ⟨cs, hcs, ho⟩ := hO decO x Ω P t.1 t.2.1 _ ws hg _ hboth.1
  obtain ⟨cs', hcs', ho'⟩ := hO decO x Ω P t.1 (fork j t.2.1 t.2.2.1) _ ws' hg' _ hboth.2
  rw [run_fst_getD H P D.V x t.1 i fun i' hi' => fork_agree j t.2.1 t.2.2.1 i' (by omega)] at ho
  have hsub : cs ++ cs' ⊆ D.calls x ws ++ D.calls x ws' := fun e he =>
    (List.mem_append.1 he).elim (fun he => List.mem_append_left _ (hcs he)) fun he => List.mem_append_right _ (hcs' he)
  have hall : ∀ e ∈ D.calls x ws ++ D.calls x ws', H e.1 = e.2 := fun e he =>
    (List.mem_append.1 he).elim (hT decO x Ω P t.1 t.2.1 _ ws hg e) (hT decO x Ω P t.1 _ _ ws' hg' e)
  obtain ⟨q, hq, q', hq', hcol⟩ := opening.bind H _ _ _ cs cs' ho ho' hi fun e he => hall e (hsub he)
  obtain ⟨u, v, huv, hcuv⟩ := findColl_of_coll H hall (List.map_subset _ hsub hq) (List.map_subset _ hsub hq') hcol
  refine ⟨u, v, ?_, hcuv⟩
  rw [hg.2, hg'.2]
  exact huv

/-- **The inner rounds' differing openings cost at most `k(x)·Pr[Be]`**: each is a collision at `Be`'s fork at its
round (`forkColl_of_opening`), and the inner rounds are among the compiled verifier's (`InnerRounds`). -/
theorem sum_opensDiffer_le {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} (hV : InnerViewRun H D I DO) (hO : CallsOpenPlayed H D I DO opening)
    (hT : CallsTrue H D I DO) (hR : InnerRounds D I)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j) ≤ D.V.k x * PrBe H D I DO decO P x := by
  have hq : ∀ j, 0 ≤ Pr (forkColl H D I DO decO P x j) := fun j => Pr_nonneg _
  have hle : ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j) ≤
      ∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j) := by
    calc ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j)
        ≤ ∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j) :=
          Finset.sum_le_sum fun j _ => Pr_mono fun t => forkColl_of_opening hV hO hT decO x P t
      _ ≤ ∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j) := by
          rw [Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j)),
            Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j))]
          exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono (hR x)) fun j _ _ => hq j
  by_cases hk : 0 < D.V.k x
  · rw [PrBe_eq_avg H D I DO decO P x hk, mul_div_cancel₀ _ (by exact_mod_cast hk.ne')]
    exact hle
  · have h0 : D.V.k x = 0 := Nat.eq_zero_of_not_pos hk
    have hI : I.V.k (xI D x) = 0 := Nat.eq_zero_of_le_zero (h0 ▸ hR x)
    have hS : ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j) = 0 := by
      rw [Fin.sum_univ_eq_sum_range (fun j => Pr (OpensDiffer H D I DO decO P x j)), hI, Finset.sum_range_zero]
    rw [hS]
    exact mul_nonneg (Nat.cast_nonneg _) (PrBe_nonneg H D I DO decO P x)

end Forks

end Claims.Recursion
