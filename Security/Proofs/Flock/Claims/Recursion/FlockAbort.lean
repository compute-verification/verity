import Proofs.Flock.Claims.Recursion.Gap

/-!
# `Claims.Flock.KS` does not give `Inner.KSAbort`

A Flock deployment `qDI` meets `Claims.Flock.KS` (`qFlockKS`), and Flock as the inner protocol over it, at a decoder
within its radius, fails `Inner.KSAbort` (`qNotKSAbort`). So `Inner.KSAbort` at Flock doesn't follow from
`Claims.Flock.KS` (`flockKS_not_KSAbort`).

`qDI`'s verifier shows the prover a query position, `0` or `1` at its coin 0, after the first message, and accepts
after the second, whatever either message says (`qV_out`). A run opens one column, `1` at the position shown
(`qCols_run`), and the code is the one word `(1, 1)` at radius `0`.

* **`qDI` meets `Claims.Flock.KS`.** No prover makes the verifier reject, so no prover's acceptance depends on the query
  position. `E^rep`'s first wait sees both positions opened unless its first three reruns draw one position, so
  `PrBad ≤ 1/4 ≤ κ` at every prover and every decoder within the radius.
* **It fails `Inner.KSAbort`.** A prover that aborts its run at position `0` reads only the run's own coin 0 and no
  fresh coins (`A = Unit`, `qKeep`). `E^rep`'s accepted reruns then open position `1` alone, from which the decoder
  lists nothing, so `E^rep` fails whenever the judged run is kept: half the time, while `κ < 1/2` and `B` finds nothing
  (`qKSAbort_fails`).

So `keep`'s dependence on the run's coins folds into the prover only where the verifier lets the prover make the run
reject after those coins, and `qV` gives no such move after its coin 0. `qDI` meets `Claims.Flock.KS` only because no
prover can thin acceptance on the query position: its `ε_table = 0` is no table bound, as a table out of the code, wrong
at position `0` alone, is opened consistently at every run that queries position `1`, half the runs.
-/

namespace Claims.Recursion

open Claims

/-! ## The deployment -/

/-- The query position, as the length of the verifier's message: `1` at coin `true`, `0` at `false`. -/
def qPos (b : Bool) : Msg := if b then [0] else []

/-- **The toy verifier**: two rounds. It answers the first message with the query position its coin 0 draws and accepts
after the second, whatever either message says. -/
def qV : Verifier Bool :=
  ⟨fun _ => 2, fun _ as cs => .ret (if as.length < 2 then .msg (qPos (cs.getD 0 false)) else .acc)⟩

/-- A run's opened column: `1` at the query position the verifier showed. -/
def qCols (v : Seen) : List (ℕ × ℕ) :=
  match v.msgs with
  | (_, .msg b) :: _ => [(b.length, 1)]
  | _ => []

/-- **The toy deployment**: the code is the one word `(1, 1)` at radius `0`; `k = 400` accepted reruns per wait,
`m′ = 2` waits, `K = 204800`, so `κ = 2^{−2} + 2^{−8} + e^{−50}`. -/
noncomputable def qDI : Flock.Deploy Bool ℕ Unit where
  zk := true
  V := qV
  parse := fun _ => some gs₀
  scopeCheck := fun _ => false
  table := fun _ _ => some gs₀
  cols := fun v _ => qCols v
  enc := fun _ _ => 1
  len := fun _ => 2
  θ := fun _ => 0
  dec := fun _ => ([], 0)
  read := fun _ => ⟨[]⟩
  εtable := fun _ => 0
  N₀ := fun _ => 100 * Real.exp 1
  weight := fun _ => 0
  waits := fun _ => 2
  cap := fun _ => 204800

/-- A decoder within the radius: the word, where both positions read `1`. -/
def qDec (os : List (ℕ × ℕ)) : List Unit :=
  if Flock.fill 2 os 0 = 1 ∧ Flock.fill 2 os 1 = 1 then [()] else []

/-! ## Its runs -/

/-- **Every run is accepted**, at every prover, tape and coin: no prover makes `qV` reject, so none aborts at the query
position. -/
theorem qV_out (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (qV.k x) → Bool) :
    (run H P qV x ω r).out = .acc := by
  simp [run, stepWith, qV, OC.eval]

theorem qV_vLog (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (qV.k x) → Bool) :
    (run H P qV x ω r).vLog = [] := by
  simp [run, stepWith, qV, OC.eval]

/-- A run opens `1` at the query position its coin 0 draws. -/
theorem qCols_run (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (qV.k x) → Bool) :
    qCols (run H P qV x ω r).seen = [(if coinSeq r 0 then 1 else 0, 1)] := by
  cases h : coinSeq r 0 <;> simp [run, stepWith, qV, OC.eval, Trace.seen, qCols, qPos]
  · exact h
  · exact congrArg List.length (ite_eq_left h)

theorem qCols_runD (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (qDI.V.k x) → Bool)
    (j : ℕ) : qDI.cols (run H P qDI.V x ω r).seen j = [(if coinSeq r 0 then 1 else 0, 1)] :=
  qCols_run H P x ω r

/-- Each run's one opening, read at table `j`. -/
theorem flatMap_open {α β : Type} (f : α → ℕ → List β) (j : ℕ) :
    ∀ vs : List α, ((vs.flatMap fun v => [f v]).flatMap fun o => o j) = vs.flatMap fun v => f v j
  | [] => rfl
  | v :: vs => by
    simp only [List.flatMap_cons, List.cons_append, List.nil_append]
    rw [flatMap_open f j vs]

/-- `qDI`'s assembly at its one table: the decoder's list at the openings, its first entry read. -/
theorem qAssemble (decode : List (ℕ × ℕ) → List Unit) (x : Msg) (vs : List Seen) (a : Unit) (l : List Unit)
    (hdl : decode (vs.flatMap fun v => qDI.cols v 0) = a :: l) :
    Flock.assemble qDI.parse qDI.table decode qDI.read x (vs.flatMap fun v => [qDI.cols v]) = some ⟨[]⟩ := by
  have h1 : gs₀.tables = 1 := rfl
  simp only [Flock.assemble, show qDI.parse x = some gs₀ from rfl, h1, List.range_one, List.mapM_cons,
    List.mapM_nil, show qDI.table x 0 = some gs₀ from rfl, Option.bind_some, flatMap_open, hdl]
  simp [qDI, gRelS]

/-- `qDI`'s assembly gives up where the decoder lists nothing. -/
theorem qAssemble_none (decode : List (ℕ × ℕ) → List Unit) (x : Msg) (vs : List Seen)
    (hdl : decode (vs.flatMap fun v => qDI.cols v 0) = []) :
    Flock.assemble qDI.parse qDI.table decode qDI.read x (vs.flatMap fun v => [qDI.cols v]) = none := by
  have h1 : gs₀.tables = 1 := rfl
  simp only [Flock.assemble, show qDI.parse x = some gs₀ from rfl, h1, List.range_one, List.mapM_cons,
    List.mapM_nil, show qDI.table x 0 = some gs₀ from rfl, Option.bind_some, flatMap_open, hdl]
  rfl

theorem qDI_k (x : Msg) : qDI.k x = 400 := by
  unfold Flock.Deploy.k
  have : 4 * qDI.N₀ x / Real.exp 1 = ((400 : ℕ) : ℝ) := by
    show 4 * (100 * Real.exp 1) / Real.exp 1 = _
    field_simp
    norm_num
  rw [this, Nat.ceil_natCast]

/-- `κ = 2^{−2} + 2^{−8} + e^{−50}`. -/
theorem qκ (x : Msg) : Flock.κ qDI x = 1 / 4 + 1 / 256 + Real.exp (-50) := by
  unfold Flock.κ
  rw [qDI_k]
  have h2 : (2 : ℝ) ^ (-(qDI.waits x : ℝ)) = 1 / 4 := by
    show (2 : ℝ) ^ (-((2 : ℕ) : ℝ)) = 1 / 4
    rw [Real.rpow_neg (by norm_num), Real.rpow_natCast]
    norm_num
  have hε : qDI.εtable x = 0 := rfl
  have hK : (qDI.cap x : ℝ) = 204800 := by norm_num [qDI]
  rw [h2, hε, hK, show -((400 : ℕ) : ℝ) / 8 = -50 by norm_num]
  norm_num

theorem qκ_ge (x : Msg) : 1 / 4 ≤ Flock.κ qDI x := by
  rw [qκ]
  linarith [Real.exp_pos (-50)]

theorem qκ_lt (x : Msg) : Flock.κ qDI x < 1 / 2 := by
  rw [qκ]
  have he : Real.exp (-50) ≤ 1 / 51 := by
    rw [Real.exp_neg, inv_eq_one_div]
    exact one_div_le_one_div_of_le (by norm_num) (by linarith [Real.add_one_le_exp (50 : ℝ)])
  linarith

/-! ## Waits -/

/-- **A wait where every rerun accepts** collects the first reruns it needs. -/
theorem waitAcc_all {R : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (hacc : ∀ r, (run r).out = .acc) :
    ∀ (rs : List R) (acc : List Seen),
      ((waitAcc k rs acc).eval H run).1 = acc ++ (rs.take (k - acc.length)).map fun r => (run r).seen
  | [], acc => by simp [waitAcc, RP.eval]
  | r :: rs, acc => by
    unfold waitAcc
    split_ifs with hk
    · simp [RP.eval, Nat.sub_eq_zero_of_le hk]
    · have hv : (run r).seen.out = .acc := hacc r
      simp only [RP.eval, hv, ↓reduceIte]
      rw [waitAcc_all H run k hacc rs (acc ++ [(run r).seen])]
      obtain ⟨n, hn⟩ : ∃ n, k - acc.length = n + 1 := ⟨k - acc.length - 1, by omega⟩
      have hn' : k - (acc ++ [(run r).seen]).length = n := by
        rw [List.length_append, List.length_singleton]
        omega
      rw [hn, hn', List.take_succ_cons, List.map_cons, List.append_assoc, List.singleton_append]

/-- **A wait collects only accepted reruns**: where every accepted rerun is `good`, so is everything a wait collects
from `good` runs. -/
theorem waitAcc_good {R : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (good : Seen → Prop)
    (hgood : ∀ r, (run r).out = .acc → good (run r).seen) :
    ∀ (rs : List R) (acc : List Seen), (∀ v ∈ acc, good v) → ∀ v ∈ ((waitAcc k rs acc).eval H run).1, good v
  | [], acc, h => by simpa [waitAcc, RP.eval] using h
  | r :: rs, acc, h => by
    unfold waitAcc
    split_ifs
    · simpa [RP.eval] using h
    · simp only [RP.eval]
      refine waitAcc_good H run k good hgood rs _ ?_
      split_ifs with ho
      · intro v hv
        rcases List.mem_append.1 hv with hv | hv
        · exact h v hv
        · rw [List.mem_singleton.1 hv]
          exact hgood r ho
      · exact h

/-- **`E^rep`'s waits where every rerun accepts**: if the first wait's first `k` draws assemble a candidate `check`
accepts, `firstWait` returns it. -/
theorem firstWait_finRange_some {R W Op : Type} (H : Msg → Msg) (run : R → Trace) (k m K : ℕ)
    (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool) (hacc : ∀ r, (run r).out = .acc)
    (hm : 0 < m) (ρ : Fin m → Fin K → R) (w : W)
    (ha : assemble ((((List.finRange K).map (ρ ⟨0, hm⟩)).take k).map (fun r => (run r).seen) |>.flatMap opens) =
      some w)
    (hc : check w = true) :
    ((firstWait k opens assemble check ((List.finRange m).map fun i => (List.finRange K).map (ρ i))).eval H
      run).1 = some w := by
  obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  rw [List.finRange_succ, List.map_cons]
  simp only [firstWait]
  rw [(RP.eval_bind H run _ _).1, waitAcc_all H run k hacc _ [], List.nil_append, List.length_nil, Nat.sub_zero]
  rw [show (⟨0, hm⟩ : Fin (m' + 1)) = 0 from rfl] at ha
  rw [ha]
  simp [hc, RP.eval]

/-- **`firstWait` gives up** where nothing the waits can collect assembles. -/
theorem firstWait_none_of {R W Op : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (good : Seen → Prop)
    (hgood : ∀ r, (run r).out = .acc → good (run r).seen)
    (ha : ∀ vs : List Seen, (∀ v ∈ vs, good v) → assemble (vs.flatMap opens) = none) :
    ∀ rss : List (List R), ((firstWait k opens assemble check rss).eval H run).1 = none
  | [] => rfl
  | rs :: rss => by
    simp only [firstWait]
    rw [(RP.eval_bind H run _ _).1,
      ha _ (waitAcc_good H run k good hgood rs [] (fun _ h => absurd h List.not_mem_nil))]
    exact firstWait_none_of H run k opens assemble check good hgood ha rss

theorem mem_take_map_finRange {R : Type} {K : ℕ} (f : Fin K → R) (k : ℕ) (j : Fin K) (hj : j.val < k) :
    f j ∈ ((List.finRange K).map f).take k := by
  rw [List.mem_iff_getElem]
  refine ⟨j.val, by simp; omega, by simp⟩

/-! ## Decoding -/

/-- Where every opened column reads `c` and one was opened at `i`, the zero-filled word reads `c` at `i`. -/
theorem fill_of_all {n : ℕ} (os : List (ℕ × ℕ)) (c : ℕ) (i : Fin n) (hall : ∀ p ∈ os, p.2 = c)
    (hi : ∃ p ∈ os, p.1 = i.val) : Flock.fill n os i = c := by
  obtain ⟨p, hp, hpi⟩ := hi
  unfold Flock.fill
  obtain ⟨q, hq⟩ : ∃ q, os.find? (fun p => p.1 == i.val) = some q := by
    rw [← Option.isSome_iff_exists, List.find?_isSome]
    exact ⟨p, hp, by simp [hpi]⟩
  rw [hq]
  simpa using hall q (List.mem_of_find?_eq_some hq)

/-- Where no column was opened at `i`, the zero-filled word reads `0` there. -/
theorem fill_of_none {n : ℕ} (os : List (ℕ × ℕ)) (i : Fin n) (h : ∀ p ∈ os, p.1 ≠ i.val) :
    Flock.fill n os i = 0 := by
  unfold Flock.fill
  rw [List.find?_eq_none.2 fun p hp => by simpa using h p hp]
  rfl

/-- **A decoder within the radius lists the word** where both positions read `1`. -/
theorem qDecodes (decode : List (ℕ × ℕ) → List Unit) (hdec : Flock.ListWithin qDI.len qDI.enc qDI.θ decode)
    (os : List (ℕ × ℕ)) (h0 : Flock.fill 2 os 0 = 1) (h1 : Flock.fill 2 os 1 = 1) : () ∈ decode os := by
  refine hdec os () ?_
  have hf : (Finset.univ.filter fun i : Fin 2 => (1 : ℕ) = Flock.fill 2 os i) = Finset.univ :=
    Finset.filter_true_of_mem fun i _ =>
      (Fin.forall_fin_two (p := fun i => (1 : ℕ) = Flock.fill 2 os i)).2 ⟨h0.symm, h1.symm⟩ i
  show (1 - 0) * ((2 : ℕ) : ℝ) ≤ ((Finset.univ.filter fun i : Fin 2 => (1 : ℕ) = Flock.fill 2 os i).card : ℝ)
  rw [hf]
  simp

theorem qDec_ok : decOK qDI qDec := by
  intro os M h
  have hle := Finset.card_filter_le (Finset.univ : Finset (Fin 2)) fun i : Fin 2 => (1 : ℕ) = Flock.fill 2 os i
  rw [Finset.card_univ, Fintype.card_fin] at hle
  have h' : (1 - 0) * ((2 : ℕ) : ℝ) ≤
      ((Finset.univ.filter fun i : Fin 2 => (1 : ℕ) = Flock.fill 2 os i).card : ℝ) := h
  have h2 : (Finset.univ.filter fun i : Fin 2 => (1 : ℕ) = Flock.fill 2 os i).card =
      (Finset.univ : Finset (Fin 2)).card := by
    rw [Finset.card_univ, Fintype.card_fin]
    have : (2 : ℝ) ≤ ((Finset.univ.filter fun i : Fin 2 => (1 : ℕ) = Flock.fill 2 os i).card : ℝ) := by
      simpa using h'
    exact le_antisymm hle (by exact_mod_cast this)
  have hall : ∀ i : Fin 2, (1 : ℕ) = Flock.fill 2 os i := fun i => Finset.filter_card_eq h2 i (Finset.mem_univ i)
  cases M
  simp [qDec, ← hall 0, ← hall 1]

/-! ## `Claims.Flock.KS` holds -/

/-- The query coins of `E^rep`'s first three draws in its first wait. -/
def q3 {n m K : ℕ} (hm : 0 < m) (hK : 3 ≤ K) (ρ : Fin m → Fin K → Fin n → Bool) : Fin 3 → Bool :=
  fun i => coinSeq (ρ ⟨0, hm⟩ (Fin.castLE hK i)) 0

/-- Three query coins not all one draw both positions. -/
theorem bool3 : ∀ f : Fin 3 → Bool, ¬ (f 0 = f 1 ∧ f 1 = f 2) → ∀ b, ∃ j, f j = b := by decide

/-- **`E^rep` succeeds** unless its first wait's first three reruns draw one query position: two that differ open both
positions, so a decoder within the radius lists the word. -/
theorem qE_some (H : Msg → Msg) (decode : List (ℕ × ℕ) → List Unit)
    (hdec : Flock.ListWithin qDI.len qDI.enc qDI.θ decode) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (qDI.V.k x) → Bool) (ρ : (Flock.E qDI decode).C x) (hw : 0 < qDI.waits x) (hK : 3 ≤ qDI.cap x)
    (hne : ¬ (q3 hw hK ρ 0 = q3 hw hK ρ 1 ∧ q3 hw hK ρ 1 = q3 hw hK ρ 2)) :
    ((Flock.E qDI decode).eval H P x ω r₀ ρ).1 = some ⟨[]⟩ := by
  have hacc : ∀ r, (run H P qDI.V x ω r).out = .acc := fun r => qV_out H P x ω r
  simp only [Flock.E, ERep, Extractor.eval, RP.eval, Trace.seen, hacc, ↓reduceIte]
  refine firstWait_finRange_some H (run H P qDI.V x ω) (qDI.k x) (qDI.waits x) (qDI.cap x)
    (fun v => [qDI.cols v]) _ _ hacc hw ρ (⟨[]⟩ : Flock.Wit) ?_ (by simp [Flock.check, qDI, gRelS])
  have hall : ∀ p ∈ ((((List.finRange (qDI.cap x)).map (ρ ⟨0, hw⟩)).take (qDI.k x)).map
      (fun r => (run H P qDI.V x ω r).seen)).flatMap (fun v => qDI.cols v 0), p.2 = 1 := by
    intro p hp
    obtain ⟨v, hv, hpv⟩ := List.mem_flatMap.1 hp
    obtain ⟨r, -, rfl⟩ := List.mem_map.1 hv
    rw [qCols_runD H P x ω r, List.mem_singleton] at hpv
    rw [hpv]
  have hpos : ∀ b : Bool, ∃ p ∈ ((((List.finRange (qDI.cap x)).map (ρ ⟨0, hw⟩)).take (qDI.k x)).map
      (fun r => (run H P qDI.V x ω r).seen)).flatMap (fun v => qDI.cols v 0), p.1 = if b then 1 else 0 := by
    intro b
    obtain ⟨j, hj⟩ := bool3 (q3 hw hK ρ) hne b
    have hj' : coinSeq (ρ ⟨0, hw⟩ (Fin.castLE hK j)) 0 = b := hj
    have hmem : ρ ⟨0, hw⟩ (Fin.castLE hK j) ∈ ((List.finRange (qDI.cap x)).map (ρ ⟨0, hw⟩)).take (qDI.k x) :=
      mem_take_map_finRange _ _ _ (by rw [qDI_k]; simp; omega)
    generalize ρ ⟨0, hw⟩ (Fin.castLE hK j) = r at hj' hmem
    refine ⟨(if b then 1 else 0, 1),
      List.mem_flatMap.2 ⟨(run H P qDI.V x ω r).seen, List.mem_map.2 ⟨r, hmem, rfl⟩, ?_⟩, rfl⟩
    rw [qCols_runD H P x ω r, hj']
    exact List.mem_singleton_self _
  have hm : () ∈ decode _ := qDecodes decode hdec _ (fill_of_all _ 1 0 hall (hpos false))
    (fill_of_all _ 1 1 hall (hpos true))
  obtain ⟨a, l, hdl⟩ := List.exists_cons_of_ne_nil (List.ne_nil_of_mem hm)
  exact qAssemble decode x _ a l hdl

/-- `E^rep`'s first wait's first three query coins are uniform. -/
theorem unifMap_q3 (decode : List (ℕ × ℕ) → List Unit) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (hw : 0 < qDI.waits x) (hK : 3 ≤ qDI.cap x) :
    UnifMap (fun s : Ω × (Fin (qDI.V.k x) → Bool) × (Flock.E qDI decode).C x => q3 hw hK s.2.2) := by
  have h0 : 0 < qDI.V.k x := by show 0 < 2; omega
  have hc : UnifMap (fun r : Fin (qDI.V.k x) → Bool => coinSeq r 0) := by
    simp only [show ∀ r : Fin (qDI.V.k x) → Bool, coinSeq r 0 = r ⟨0, h0⟩ from fun r => dite_eq_left h0]
    exact UnifMap.coord _
  exact UnifMap.snd.comp (UnifMap.snd.comp ((UnifMap.coord (⟨0, hw⟩ : Fin (qDI.waits x))).comp
    ((UnifMap.comp_emb (γ := Fin (qDI.V.k x) → Bool) (Fin.castLEEmb hK)).comp (UnifMap.pi hc 3))))

theorem Pr_bool3 : Pr (fun f : Fin 3 → Bool => f 0 = f 1 ∧ f 1 = f 2) = 1 / 4 := by
  rw [Pr_eq_sum]
  simp
  have : ({f | f 0 = f 1 ∧ f 1 = f 2} : Finset (Fin 3 → Bool)).card = 2 := by decide
  rw [this]
  norm_num

/-- **`qDI` meets `Claims.Flock.KS`**: at every prover and every decoder within the radius, `E^rep` fails only where its
first wait's first three reruns draw one query position, a quarter of the draws, and `κ ≥ 1/4`. -/
theorem qFlockKS (H : Msg → Msg) : Flock.KS H qDI := by
  intro decode hdec x Ω _ _ P
  have hw : 0 < qDI.waits x := by show 0 < 2; omega
  have hK : 3 ≤ qDI.cap x := by show 3 ≤ 204800; omega
  have hbad : PrBad H qDI.V (Flock.R qDI.parse) (Flock.E qDI decode) x P ≤ 1 / 4 := by
    calc PrBad H qDI.V (Flock.R qDI.parse) (Flock.E qDI decode) x P
        ≤ Pr (fun s : Ω × (Fin (qDI.V.k x) → Bool) × (Flock.E qDI decode).C x =>
            q3 hw hK s.2.2 0 = q3 hw hK s.2.2 1 ∧ q3 hw hK s.2.2 1 = q3 hw hK s.2.2 2) :=
          Pr_mono fun s hs => by
            by_contra hne
            refine hs.2 ?_
            rw [qE_some H decode hdec P x s.1 s.2.1 s.2.2 hw hK hne]
            exact ⟨gs₀, rfl, gRelS _⟩
      _ = 1 / 4 := by
          rw [(unifMap_q3 decode x hw hK).Pr (fun f => f 0 = f 1 ∧ f 1 = f 2), Pr_bool3]
  linarith [qκ_ge x, Real.sqrt_nonneg (Flock.γ qDI x * PrB H qDI.V P x)]

/-! ## `Inner.KSAbort` fails -/

/-- The prover: `[1]`, every round. -/
def qP : Prover Unit := ⟨fun _ _ _ => .ret [1]⟩

/-- **The abort**: keep a run at query position `1`, reading the run's coin 0 and nothing else. -/
def qKeep : Unit → (Fin (qV.k []) → Bool) → Unit → Bool := fun _ r _ => coinSeq r 0

theorem qPrBAbort (H : Msg → Msg) {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (P : Prover Ω)
    (x : Msg) (keep : Ω → (Fin (qV.k x) → Bool) → A → Bool) : PrBAbort H qV P x keep = 0 := by
  have hk : 0 < qV.k x := by show 0 < 2; omega
  unfold PrBAbort
  rw [dite_eq_left hk]
  refine le_antisymm ((Pr_mono fun s h => ?_).trans_eq Pr_false) (Pr_nonneg _)
  obtain ⟨u, v, huv, -⟩ := h
  have h0 : ∀ ω t, (abortRun H P qV x keep ω t).vLog = [] := fun ω t => by
    unfold abortRun
    split
    · exact qV_vLog H P x ω t.1
    · rfl
  simp only [BAbort, h0] at huv
  simp [findColl] at huv

/-- **`E^rep` gives up at the kept runs**: they open position `1` alone, so the decoder lists nothing. -/
theorem qE_abort_none (H : Msg → Msg) (ω : Unit)
    (ρ : Fin (qDI.waits []) → Fin (qDI.cap []) → (Fin (qV.k []) → Bool) × Unit) :
    ((firstWait (qDI.k []) (fun v => [qDI.cols v]) (Flock.assemble qDI.parse qDI.table qDec qDI.read [])
      (Flock.check qDI.parse [])
      ((List.finRange (qDI.waits [])).map fun i => (List.finRange (qDI.cap [])).map (ρ i))).eval H
        (abortRun H qP qV [] qKeep ω)).1 = none := by
  refine firstWait_none_of H _ _ _ _ _ (fun v => qCols v = [(1, 1)]) (fun t ht => ?_) (fun vs hvs => ?_) _
  · unfold abortRun at ht ⊢
    split at ht
    · rename_i hkeep
      rw [ite_eq_left hkeep, qCols_run H qP [] ω t.1, show coinSeq t.1 0 = true from hkeep]
      rfl
    · exact absurd ht (by simp [abortTrace])
  · have hos : ∀ p ∈ vs.flatMap fun v => qDI.cols v 0, p = (1, 1) := fun p hp => by
      obtain ⟨v, hv, hpv⟩ := List.mem_flatMap.1 hp
      rw [show qDI.cols v 0 = qCols v from rfl, hvs v hv, List.mem_singleton] at hpv
      exact hpv
    have h0 : Flock.fill 2 (vs.flatMap fun v => qDI.cols v 0) 0 = 0 :=
      fill_of_none _ _ fun p hp => by rw [hos p hp]; decide
    refine qAssemble_none qDec [] vs ?_
    simp [qDec, h0]

/-- The judged run's coin 0 is `true` half the time, whatever else is drawn. -/
theorem Pr_coin0 {n : ℕ} (h0 : 0 < n) {Y : Type} [Fintype Y] [Nonempty Y] :
    Pr (fun s : Unit × ((Fin n → Bool) × Unit) × Y => coinSeq s.2.1.1 0 = true) = 1 / 2 := by
  have hc : UnifMap (fun r : Fin n → Bool => coinSeq r 0) := by
    simp only [show ∀ r : Fin n → Bool, coinSeq r 0 = r ⟨0, h0⟩ from fun r => dite_eq_left h0]
    exact UnifMap.coord _
  have hφ : UnifMap (fun s : Unit × ((Fin n → Bool) × Unit) × Y => coinSeq s.2.1.1 0) :=
    UnifMap.snd.comp (UnifMap.fst.comp (UnifMap.fst.comp hc))
  rw [hφ.Pr (fun b => b = true), Pr_eq_sum]
  simp

/-- **The abort beats Flock's errors**: kept at query position `1`, the judged run is accepted and `E^rep` fails, half
the time, while `κ < 1/2` and `B` finds nothing in runs that make no SHA-512 calls. -/
theorem qKSAbort_fails (H : Msg → Msg) :
    (Inner.ofFlock qDI qDec).κ [] +
        Real.sqrt ((Inner.ofFlock qDI qDec).γ [] * PrBAbort H (Inner.ofFlock qDI qDec).V qP [] qKeep) <
      (Inner.ofFlock qDI qDec).PrBadAbort H [] qP qKeep := by
  have hr : (Inner.ofFlock qDI qDec).κ [] +
      Real.sqrt ((Inner.ofFlock qDI qDec).γ [] * PrBAbort H (Inner.ofFlock qDI qDec).V qP [] qKeep) =
        Flock.κ qDI [] := by
    have hB : PrBAbort H (Inner.ofFlock qDI qDec).V qP [] qKeep = 0 := qPrBAbort H qP [] qKeep
    rw [hB, mul_zero, Real.sqrt_zero, add_zero]
    rfl
  rw [hr]
  have h0 : 0 < qV.k [] := by show 0 < 2; omega
  have hq : (1 : ℝ) / 2 ≤ (Inner.ofFlock qDI qDec).PrBadAbort H [] qP qKeep := by
    calc (1 : ℝ) / 2 = Pr (fun s : Unit × ((Fin (qV.k []) → Bool) × Unit) ×
          (Fin (qDI.waits []) → Fin (qDI.cap []) → (Fin (qV.k []) → Bool) × Unit) => coinSeq s.2.1.1 0 = true) :=
          (Pr_coin0 h0).symm
      _ ≤ _ := Pr_mono fun s hs => by
          change coinSeq s.2.1.1 0 = true at hs
          refine ⟨?_, ?_⟩
          · show (abortRun H qP qV [] qKeep s.1 s.2.1).out = .acc
            simp only [abortRun, qKeep, hs, ↓reduceIte]
            exact qV_out H qP [] s.1 s.2.1.1
          · have hacc : (abortRun H qP qV [] qKeep s.1 s.2.1).seen.out = .acc := by
              show (if qKeep s.1 s.2.1.1 s.2.1.2 then run H qP qV [] s.1 s.2.1.1 else abortTrace).out = .acc
              simp only [qKeep, hs, ↓reduceIte]
              exact qV_out H qP [] s.1 s.2.1.1
            show ¬ (Flock.R qDI.parse).ext [] ((erep (qDI.waits []) (qDI.k []) (qDI.cap [])
              (fun v => [qDI.cols v]) (Flock.assemble qDI.parse qDI.table qDec qDI.read [])
                (Flock.check qDI.parse []) s.2.1 s.2.2).eval H (abortRun H qP qV [] qKeep s.1)).1
            simp only [erep, RP.eval, hacc, ↓reduceIte]
            rw [qE_abort_none H s.1 s.2.2]
            exact id
  linarith [qκ_lt []]

/-- **Flock as the inner protocol fails `Inner.KSAbort`** at `qDI` and `qDec`, a decoder within the radius. -/
theorem qNotKSAbort (H : Msg → Msg) : ¬ (Inner.ofFlock qDI qDec).KSAbort H := fun h =>
  absurd (h [] Unit qP Unit qKeep) (not_le.2 (qKSAbort_fails H))

/-- **`Inner.KSAbort` at Flock doesn't follow from `Claims.Flock.KS`.** At every hash, a Flock deployment meets
`Claims.Flock.KS`, and Flock as the inner protocol over it, at a decoder within its radius, fails `Inner.KSAbort`, at a
prover whose abort reads only the run's own coin 0 and no fresh coins (`qKSAbort_fails`). -/
theorem flockKS_not_KSAbort (H : Msg → Msg) :
    ∃ (DI : Flock.Deploy Bool ℕ Unit) (decI : List (ℕ × ℕ) → List Unit),
      Flock.KS H DI ∧ decOK DI decI ∧ ¬ (Inner.ofFlock DI decI).KSAbort H :=
  ⟨qDI, qDec, qFlockKS H, qDec_ok, qNotKSAbort H⟩

end Claims.Recursion
