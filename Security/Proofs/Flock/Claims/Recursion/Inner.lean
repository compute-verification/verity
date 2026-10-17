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
* Binding: two extractions whose runs agree on inner round `j`'s commit string and whose inner views differ there give
  a collision among their recorded calls (`opening_coll`), at a fork at `j` one of `Be`'s (`forkColl_of_opening`). So
  the inner rounds' differing openings cost at most `k(x)·Pr[Be]` (`sum_opensDiffer_le`).
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

section Binding

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **Binding**: two extractions on accepted runs whose messages at inner round `j` agree (one commit string) and whose
inner views differ there give a collision among the calls their witnesses record. -/
theorem opening_coll {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening}
    (hO : CallsOpen H D I DO opening) (hC : CallsRecord H D I DO) (decO : List (ℕ × Col) → List T)
    (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) {r r' : Fin (D.V.k x) → Coin} {c c' : (EO D I DO decO).C x}
    {ws ws' : List Flock.Wit} (h : EOGives H D I DO decO P x ω r c ws) (h' : EOGives H D I DO decO P x ω r' c' ws')
    {j : ℕ} (hj : j < I.V.k (xI D x))
    (hm : ((run H P D.V x ω r).msgs.map Prod.fst).getD j [] = ((run H P D.V x ω r').msgs.map Prod.fst).getD j [])
    (hd : ((D.innerView H x r ws).msgs.map Prod.fst).getD j [] ≠ ((D.innerView H x r' ws').msgs.map Prod.fst).getD j []) :
    ∃ u v, findColl (D.calls x ws ++ D.calls x ws') = some (u, v) ∧ Collision H u v := by
  obtain ⟨cs, hcs, ho⟩ := hO decO x Ω P ω r c ws h j hj
  obtain ⟨cs', hcs', ho'⟩ := hO decO x Ω P ω r' c' ws' h' j hj
  rw [hm] at ho
  have hsub : cs ++ cs' ⊆ D.calls x ws ++ D.calls x ws' := fun e he =>
    (List.mem_append.1 he).elim (fun he => List.mem_append_left _ (hcs he)) fun he => List.mem_append_right _ (hcs' he)
  have hall : ∀ e ∈ D.calls x ws ++ D.calls x ws', H e.1 = e.2 := fun e he =>
    (List.mem_append.1 he).elim ((hC decO x Ω P ω r c ws h).2 e) ((hC decO x Ω P ω r' c' ws' h').2 e)
  obtain ⟨q, hq, q', hq', hc⟩ := opening.bind H _ _ _ cs cs' ho ho' hd fun e he => hall e (hsub he)
  exact findColl_of_coll H hall (List.map_subset _ hsub hq) (List.map_subset _ hsub hq') hc

end Binding

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

/-- **Binding at a fork**: a differing opening at inner round `j` is a collision `Be` finds at its fork at `j`, since
the two runs share their coins before `j` and so their message `j`. -/
theorem forkColl_of_opening {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening}
    (hO : CallsOpen H D I DO opening) (hC : CallsRecord H D I DO) (decO : List (ℕ × Col) → List T)
    (x : Msg) {Ω : Type} (P : Prover Ω) {j : ℕ} (hj : j < I.V.k (xI D x))
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x)
    (h : OpensDiffer H D I DO decO P x j t) : forkColl H D I DO decO P x j t := by
  obtain ⟨ws, ws', hg, hg', hd⟩ := h
  obtain ⟨u, v, huv, hc⟩ := opening_coll hO hC decO x P t.1 hg hg' hj
    (run_fst_getD H P D.V x t.1 j (fork_agree j t.2.1 t.2.2.1)) hd
  refine ⟨u, v, ?_, hc⟩
  rw [hg.2, hg'.2]
  exact huv

/-- **The inner rounds' differing openings cost at most `k(x)·Pr[Be]`**: each is a collision at `Be`'s fork at its
round (`forkColl_of_opening`), and the inner rounds are among the compiled verifier's (`InnerRounds`). -/
theorem sum_opensDiffer_le {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening}
    (hO : CallsOpen H D I DO opening) (hC : CallsRecord H D I DO) (hR : InnerRounds D I)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j) ≤ D.V.k x * PrBe H D I DO decO P x := by
  have hq : ∀ j, 0 ≤ Pr (forkColl H D I DO decO P x j) := fun j => Pr_nonneg _
  have hle : ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j) ≤
      ∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j) := by
    calc ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j)
        ≤ ∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j) :=
          Finset.sum_le_sum fun j _ => Pr_mono fun t => forkColl_of_opening hO hC decO x P j.isLt t
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
