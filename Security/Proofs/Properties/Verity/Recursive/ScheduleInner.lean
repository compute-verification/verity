import Proofs.Flock.Claims.Recursion.DeploySessions

/-!
# The recursive verifier's schedule: the inner and staging rounds

The converse of `RecDeploy.ract_inner`, on a run of the compiled recursive verifier `RecDeploy.V IV st` that need not
accept. Where the statement reads (`readX`), and so wherever it has a V* part (`0 < nS x`):

* every inner and staging round the run reaches is answered by its coin's bytes (`RecDeploy.run_inner`), so the run
  reaches session 0's first round (`reach_sessions`);
* every later round reads the run's own inner and staging rounds (`innerPairs_reach`), whose last message is the
  staging message that `Stage.ofMsg` reads each part's public file, the inner session's record and V*'s shared rows
  off, with the coin server's transcript of the inner rounds before it (`stage_reads`).

The run lemmas under them hold for any verifier: a round before a run's last is answered by a message
(`stepWith_mid`), a run whose round is answered by a message runs on or has used its rounds up (`stepWith_full`), and
a run that accepts a round accepts (`stepWith_acc_out`).
-/

namespace Verity.Recursive.Schedule

open Claims (Msg Out Prover Trace stepWith run coinSeq)
open FlockDeploy (Coin)
open RecDeploy

/-! ## Runs -/

section Steps

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → Claims.OC Out)

/-- **A round before a run's last is answered by a message**: a run stops at the first round it accepts or refuses. -/
theorem stepWith_mid :
    ∀ (f i : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      t.msgs.length + i + 1 < (stepWith H P x ω vn f hist as t).msgs.length →
      ∃ a m, (stepWith H P x ω vn f hist as t).msgs[t.msgs.length + i]? = some (a, .msg m)
  | 0, i, hist, as, t, h => by
    have := (stepWith_mono H P x ω vn 0 hist as t).2.2
    omega
  | f + 1, i, hist, as, t, h => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg mm =>
      rw [stepWith_succ_msg H P x ω vn f hist as t hvb] at h ⊢
      set a₀ := ((P.next x hist ω).eval H).1
      set t₁ : Trace :=
        { t with msgs := t.msgs ++ [(a₀, .msg mm)], pLog := t.pLog ++ ((P.next x hist ω).eval H).2,
                 vLog := t.vLog ++ lo }
      have hl₁ : t₁.msgs.length = t.msgs.length + 1 := by simp [t₁]
      cases i with
      | zero =>
        obtain ⟨rest, hrest⟩ := (stepWith_mono H P x ω vn f (hist ++ [(a₀, mm)]) (as ++ [a₀]) t₁).1
        refine ⟨a₀, mm, ?_⟩
        rw [← hrest]
        simp [t₁]
      | succ i =>
        obtain ⟨a, m, hm⟩ := stepWith_mid f i (hist ++ [(a₀, mm)]) (as ++ [a₀]) t₁ (by omega)
        refine ⟨a, m, ?_⟩
        rw [show t.msgs.length + (i + 1) = t₁.msgs.length + i by omega]
        exact hm
    | acc =>
      rw [stepWith_succ_acc H P x ω vn f hist as t hvb] at h
      simp only [List.length_append, List.length_singleton] at h
      omega
    | rej =>
      rw [stepWith_succ_rej H P x ω vn f hist as t hvb] at h
      simp only [List.length_append, List.length_singleton] at h
      omega

/-- **A run whose round is answered by a message runs on, or has used its rounds up.** -/
theorem stepWith_full :
    ∀ (f i : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (a m : Msg),
      (stepWith H P x ω vn f hist as t).msgs[t.msgs.length + i]? = some (a, .msg m) →
      t.msgs.length + i + 1 < (stepWith H P x ω vn f hist as t).msgs.length ∨
        (stepWith H P x ω vn f hist as t).msgs.length = t.msgs.length + f
  | 0, i, _, _, t, a, m, h => by simp [stepWith] at h
  | f + 1, i, hist, as, t, a, m, h => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg mm =>
      rw [stepWith_succ_msg H P x ω vn f hist as t hvb] at h ⊢
      set a₀ := ((P.next x hist ω).eval H).1
      set t₁ : Trace :=
        { t with msgs := t.msgs ++ [(a₀, .msg mm)], pLog := t.pLog ++ ((P.next x hist ω).eval H).2,
                 vLog := t.vLog ++ lo }
      have hl₁ : t₁.msgs.length = t.msgs.length + 1 := by simp [t₁]
      have hge := (stepWith_mono H P x ω vn f (hist ++ [(a₀, mm)]) (as ++ [a₀]) t₁).1.length_le
      have hle := (stepWith_mono H P x ω vn f (hist ++ [(a₀, mm)]) (as ++ [a₀]) t₁).2.2
      cases i with
      | zero =>
        cases f with
        | zero =>
          right
          omega
        | succ g =>
          left
          have := stepWith_length_succ H P x ω vn g (hist ++ [(a₀, mm)]) (as ++ [a₀]) t₁
          omega
      | succ i =>
        rcases stepWith_full f i (hist ++ [(a₀, mm)]) (as ++ [a₀]) t₁ a m
          (by rw [show t₁.msgs.length + i = t.msgs.length + (i + 1) by omega]; exact h) with h' | h'
        · left
          omega
        · right
          omega
    | acc =>
      rw [stepWith_succ_acc H P x ω vn f hist as t hvb] at h
      cases i <;> simp at h
    | rej =>
      rw [stepWith_succ_rej H P x ω vn f hist as t hvb] at h
      cases i <;> simp at h

/-- **A run that accepts a round accepts.** -/
theorem stepWith_acc_out :
    ∀ (f i : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (a : Msg),
      (stepWith H P x ω vn f hist as t).msgs[t.msgs.length + i]? = some (a, .acc) →
      (stepWith H P x ω vn f hist as t).out = .acc
  | 0, i, _, _, t, a, h => by simp [stepWith] at h
  | f + 1, i, hist, as, t, a, h => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg mm =>
      rw [stepWith_succ_msg H P x ω vn f hist as t hvb] at h ⊢
      cases i with
      | zero =>
        obtain ⟨rest, hrest⟩ := (stepWith_mono H P x ω vn f _ _ _).1
        rw [← hrest] at h
        simp at h
      | succ i =>
        exact stepWith_acc_out f i _ _ _ a (by simpa [Nat.add_assoc, Nat.add_comm 1 i] using h)
    | acc =>
      rw [stepWith_succ_acc H P x ω vn f hist as t hvb]
    | rej =>
      rw [stepWith_succ_rej H P x ω vn f hist as t hvb] at h
      cases i <;> simp at h

end Steps

section Runs

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV st).k x) → Coin)

/-- A round before the run's last is answered by a message. -/
theorem run_mid {i : ℕ} (h : i + 1 < (run H P (V IV st) x ω r).msgs.length) :
    ∃ a m, (run H P (V IV st) x ω r).msgs[i]? = some (a, .msg m) := by
  have := stepWith_mid H P x ω (vnOf IV st x r) ((V IV st).k x) i [] [] ⟨[], [], [], .rej⟩
  simp only [List.length_nil, Nat.zero_add] at this
  rw [← run_eq H IV st x P ω r] at this
  exact this h

/-- A round answered by a message is followed by another, or is the verifier's last. -/
theorem run_full {i : ℕ} {a m : Msg} (h : (run H P (V IV st) x ω r).msgs[i]? = some (a, .msg m)) :
    i + 1 < (run H P (V IV st) x ω r).msgs.length ∨ (run H P (V IV st) x ω r).msgs.length = (V IV st).k x := by
  have := stepWith_full H P x ω (vnOf IV st x r) ((V IV st).k x) i [] [] ⟨[], [], [], .rej⟩ a m
  simp only [List.length_nil, Nat.zero_add] at this
  rw [← run_eq H IV st x P ω r] at this
  exact this h

/-- **A run that accepts a round accepts.** -/
theorem run_acc_out {i : ℕ} {a : Msg} (h : (run H P (V IV st) x ω r).msgs[i]? = some (a, .acc)) :
    (run H P (V IV st) x ω r).out = .acc := by
  have := stepWith_acc_out H P x ω (vnOf IV st x r) ((V IV st).k x) i [] [] ⟨[], [], [], .rej⟩ a
  simp only [List.length_nil, Nat.zero_add] at this
  rw [← run_eq H IV st x P ω r] at this
  exact this h

/-- A statement with a V* part reads. -/
theorem readX_of_nS (hn : 0 < nS x) : ∃ xr, readX x = some xr := by
  unfold nS at hn
  cases h : readX x with
  | none => simp [h] at hn
  | some xr => exact ⟨xr, rfl⟩

/-- **The compiled verifier's rounds**: the inner and staging rounds, then each V* session's, at any inner rounds. -/
theorem k_eq (ms : List (Msg × Out)) :
    (V IV st).k x = IV.k (xIOf x) + kS x + ((vstar st x ms).map VO.k).sum := by
  show _ + _ = _
  rw [vstar_map_k]

/-- **F1: the run reaches session 0.** Where the statement has a V* part, every inner and staging round is answered
by its coin's bytes, so the run goes past them: its message at session 0's first round is there. -/
theorem reach_sessions (hn : 0 < nS x) :
    IV.k (xIOf x) + kS x < (run H P (V IV st) x ω r).msgs.length := by
  obtain ⟨xr, hx⟩ := readX_of_nS x hn
  have hK := k_eq IV st x []
  have hsum : 0 < ((vstar st x []).map VO.k).sum := by
    obtain ⟨y, ys, hy⟩ := List.exists_cons_of_length_pos
      (show 0 < (vstar st x []).length by rw [vstar_length]; exact hn)
    rw [hy, List.map_cons, List.sum_cons]
    have := VO_k_pos y
    omega
  have hpos : 0 < (run H P (V IV st) x ω r).msgs.length := by
    have key : ∀ f, 0 < f → 0 < (stepWith H P x ω (vnOf IV st x r) f [] [] ⟨[], [], [], .rej⟩).msgs.length := by
      intro f hf
      obtain ⟨g, rfl⟩ : ∃ g, f = g + 1 := ⟨f - 1, by omega⟩
      exact stepWith_length_succ H P x ω _ g [] [] ⟨[], [], [], .rej⟩
    rw [run_eq H IV st x P ω r]
    exact key _ (by omega)
  rcases Nat.lt_or_ge (IV.k (xIOf x) + kS x) (run H P (V IV st) x ω r).msgs.length with h | hle
  · exact h
  · obtain ⟨a, ha⟩ := run_inner H IV st x P ω r hx (i := (run H P (V IV st) x ω r).msgs.length - 1)
      (by omega) (by omega)
    rcases run_full H IV st x P ω r ha with h | h <;> omega

/-- **F1: every later round reads the run's own inner and staging rounds**: messages that extend the run's first
`m ≥ k_I + k_S` and coins that extend its first `k_I + k_S` show them as `innerPairs`. `RecDeploy.innerPairs_run`
without the run's acceptance. -/
theorem innerPairs_reach {xr : XR} (hx : readX x = some xr) {m : ℕ} (hk : IV.k (xIOf x) + kS x ≤ m)
    (hm : m ≤ (run H P (V IV st) x ω r).msgs.length) (bs : List Msg) {cs : List Coin}
    (hc : cs.take (IV.k (xIOf x) + kS x) = (List.range (IV.k (xIOf x) + kS x)).map (coinSeq r)) :
    innerPairs (IV.k (xIOf x) + kS x) (((run H P (V IV st) x ω r).msgs.take m).map Prod.fst ++ bs) cs =
      (run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x) := by
  unfold innerPairs
  rw [hc, List.take_append_of_le_length (by simp only [List.length_map, List.length_take]; omega),
    ← List.map_take, List.take_take, Nat.min_eq_left hk, List.map_map]
  conv_rhs => rw [← List.zip_unzip ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x))]
  rw [List.unzip_fst, List.unzip_snd]
  congr 1
  apply List.ext_getElem (by simp only [List.length_map, List.length_range, List.length_take]; omega)
  intro i h₁ h₂
  simp only [List.length_map, List.length_range] at h₁
  obtain ⟨a, ha⟩ := run_inner H IV st x P ω r hx h₁ (by omega)
  rw [List.getElem?_eq_getElem (by omega)] at ha
  simp [Option.some.inj ha]

/-- **F1: the staging round as `Stage.ofMsg` reads it.** Where the statement has a V* part, the run's staging round
(round `k_I`) is its message `a`, answered by its coin's bytes, and `Stage.ofMsg tr` reads each part's public file,
the inner session's record and V*'s shared rows off `a`, and the coin server's transcript `tr` of the inner rounds
before it. -/
theorem stage_reads (tr : Msg → List (Msg × Out) → ByteArray) (hn : 0 < nS x) :
    ∃ a, (run H P (V IV st) x ω r).msgs[IV.k (xIOf x)]? = some (a, .msg (coinSeq r (IV.k (xIOf x))).bytes) ∧
      (∀ s, (Stage.ofMsg tr).pub x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s =
        ((decodeStage a).bind fun p => p.1[s]?).getD ByteArray.empty) ∧
      (Stage.ofMsg tr).record x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) =
        ((decodeStage a).map (·.2.1)).getD ByteArray.empty ∧
      (Stage.ofMsg tr).rows x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) =
        ((decodeStage a).map (·.2.2)).getD ByteArray.empty ∧
      (Stage.ofMsg tr).transcript x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) =
        tr x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x))) := by
  obtain ⟨xr, hx⟩ := readX_of_nS x hn
  have hkS : kS x = 1 := rfl
  have hreach := reach_sessions H IV st x P ω r hn
  obtain ⟨a, ha⟩ := run_inner H IV st x P ω r hx (i := IV.k (xIOf x)) (by omega) (by omega)
  have htake : (run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x) =
      (run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x)) ++ [(a, .msg (coinSeq r (IV.k (xIOf x))).bytes)] := by
    rw [hkS, List.take_add_one, ha]
    rfl
  have hst : stageOf ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) = decodeStage a := by
    unfold stageOf
    rw [htake, List.getLast?_concat]
    rfl
  refine ⟨a, ha, fun s => ?_, ?_, ?_, ?_⟩
  · show ((stageOf _).bind fun p => p.1[s]?).getD ByteArray.empty = _
    rw [hst]
  · show ((stageOf _).map (·.2.1)).getD ByteArray.empty = _
    rw [hst]
  · show ((stageOf _).map (·.2.2)).getD ByteArray.empty = _
    rw [hst]
  · show tr x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)).dropLast = _
    rw [htake, List.dropLast_concat]

end Runs

end Verity.Recursive.Schedule
