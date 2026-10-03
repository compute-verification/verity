import Pouw.Protocol.PearlC.RowSem
import Pouw.SecurityProofs.PearlC.TileCircuit

/-!
# The row tables compute a row's statistics, scales and forming (proofs)

For every `ops` (core's primitives as a parameter), a row table's honest values meet its gates run by run
(`Rows.holds`, `RunsHold`), the inputs being the transcript's (`Rows.val_input`), and a transcript whose one unit is
correct holds them on every output (`Spec.out_eq`). On the runs `pc8`'s two row units share (`pre8Runs`), those values
are `RowSem`'s formulas on the row's inputs:

* `s` and `ρ` (`stats_holds`): the running max by induction over the positions (`selRun_holds`), each FMA partial by
  strong induction over its accumulator (`fmaRun_holds`), the butterfly round by round (`bfly_holds`);
* `α`, `β` and the floor (`pre8_alpha`, `pre8_beta`, `pre8_floor`), the rescaled noise' codes (`pre8_eTilde`) and each
  position's code (`pre8_code`), so `row8A_code` and `row8B_code`: a transcript correct on RowA or RowB has on position
  `t`'s output `code8W` at `alpha8`'s and `beta8`'s words of its row.
-/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit

variable (ops : RowOps)

/-! ## Values that meet a table's gates -/

/-- The values `V` meet the run `R`'s gates from gate `o`. -/
def RunHolds (V : ℕ → ℕ) (R : Run) (o : ℕ) : Prop :=
  ∀ r < R.n, V (o + r) = (R.gate r).prim.eval ops ((R.gate r).args.map (Arg.eval ops V))

/-- The values `V` meet every run's gates, the runs laid out from gate `o`. -/
def RunsHold (V : ℕ → ℕ) : List Run → ℕ → Prop
  | [], _ => True
  | R :: L, o => RunHolds ops V R o ∧ RunsHold V L (o + R.n)

theorem runsHold_append (V : ℕ → ℕ) : ∀ {L₁ L₂ : List Run} {o : ℕ}, RunsHold ops V (L₁ ++ L₂) o →
    RunsHold ops V L₁ o ∧ RunsHold ops V L₂ (o + runsLen L₁)
  | [], _, o, h => ⟨trivial, by simpa [runsLen] using h⟩
  | R :: L₁, L₂, o, h => by
    obtain ⟨h1, h2⟩ := h
    obtain ⟨h3, h4⟩ := runsHold_append V h2
    exact ⟨⟨h1, h3⟩, by simpa [runsLen, Nat.add_assoc] using h4⟩

theorem fill_reads (V : ℕ → ℕ) : ∀ args : List Arg,
    fill ops args ((args.flatMap Arg.reads).map V) = args.map (Arg.eval ops V)
  | [] => rfl
  | .c v :: as => by simp [fill, Arg.reads, Arg.eval, fill_reads V as]
  | .g a w :: as => by simp [fill, Arg.reads, Arg.eval, fill_reads V as]

theorem runsAt_lt : ∀ (L : List Run) (o g : ℕ) (G : RGate), runsAt L o g = some G → g < o + runsLen L
  | [], _, _, _, h => by simp [runsAt] at h
  | R :: L, o, g, G, h => by
    unfold runsAt at h
    split_ifs at h with hg
    · simp only [runsLen]; omega
    · have := runsAt_lt L _ g G h; simp only [runsLen]; omega

theorem runsHold_of (V : ℕ → ℕ) : ∀ (L : List Run) (o : ℕ),
    (∀ g G, o ≤ g → runsAt L o g = some G → V g = G.prim.eval ops (G.args.map (Arg.eval ops V))) →
    RunsHold ops V L o
  | [], _, _ => trivial
  | R :: L, o, h => by
    refine ⟨fun r hr => h (o + r) (R.gate r) (by omega) ?_,
      runsHold_of V L (o + R.n) fun g G hg hG => h g G (by omega) ?_⟩
    · simp [runsAt, hr]
    · have hn : ¬ g < o + R.n := by omega
      simp only [runsAt, hn, ↓reduceIte]; exact hG

@[simp] theorem Wire.eval_id (x : ℕ) : Wire.eval ops .id x = x := rfl

theorem Prim.eval_mod (q : Prim) (l : List ℕ) : q.eval ops l % 2 ^ q.w = q.eval ops l := by
  unfold Prim.eval; exact Nat.mod_mod _ _

/-- **A table's honest values meet its gates**: the circuit's evaluation on any inputs. -/
theorem Rows.holds (T : Rows) (X : (T.spec ops).toCircuit.Transcript) :
    RunsHold ops (val ((T.spec ops).toCircuit.eval X)) T.runs T.nIn := by
  apply runsHold_of
  intro g G hg hG
  have hlt : g < (T.spec ops).N := by
    have := runsAt_lt _ _ _ _ hG
    show g < T.nIn + runsLen T.runs
    omega
  have hgate : (T.spec ops).gate g = G.toGate ops := by
    show T.gate ops g = _
    unfold Rows.gate
    have hn : ¬ g < T.nIn := by omega
    simp only [hn, ↓reduceIte, hG]
  rw [Spec.val_gate _ X hlt (by rw [hgate]; rfl), hgate]
  simp only [RGate.toGate, RGate.reads]
  rw [fill_reads, Prim.eval_mod]

/-- A table's honest values on its inputs are the transcript's. -/
theorem Rows.val_input (T : Rows) (X : (T.spec ops).toCircuit.Transcript) {l : ℕ} (hl : l < T.nIn) :
    val ((T.spec ops).toCircuit.eval X) l = val X l := by
  have hlt : l < (T.spec ops).N := by
    show l < T.nIn + runsLen T.runs
    omega
  rw [Spec.val_input _ X hlt (by show (T.gate ops l).input = true; unfold Rows.gate; simp only [hl, ↓reduceIte]),
    val_of_lt _ hlt]

/-- A transcript whose one unit is correct holds the circuit's values on every output (core's
`Verity.Partition.compose`). -/
theorem Spec.out_eq (S : Spec) (X : S.toCircuit.Transcript) (hX : S.part.Correct X 0) {g : ℕ} (hg : g ∈ S.outs)
    (hlt : g < S.N) : val X g = val (S.toCircuit.eval X) g := by
  have hw : S.part.wrong X = ∅ := by
    ext u
    simp only [Verity.Partition.mem_wrong, Finset.notMem_empty, iff_false, not_not]
    rwa [Subsingleton.elim u 0]
  have hg' : (⟨g, hlt⟩ : Fin S.toCircuit.N) ∈ S.part.committed :=
    Verity.Partition.mem_committed_of_output _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hg⟩)
  rw [val_of_lt _ hlt, val_of_lt _ hlt, S.part.compose X X (fun _ _ => rfl) hw _ hg']

/-! ## The statistics -/

section Stats

variable {ops} {V : ℕ → ℕ}

theorem selRun_holds {ab : Wire} {K b : ℕ} (h : RunHolds ops V (selRun ab K b) b) :
    ∀ t, t + 1 < K → V (b + 2 * t + 1) = rowMax ops ab V (t + 1) := by
  intro t
  induction t with
  | zero =>
    intro hK
    have h0 := h 0 (by simp [selRun]; omega)
    have h1 := h 1 (by simp [selRun]; omega)
    simp only [selRun] at h0 h1
    simp at h0 h1
    simp only [rowMax, Arg.eval, Wire.eval_id] at h0 h1 ⊢
    rw [h1, h0]
  | succ t ih =>
    intro hK
    have h0 := h (2 * t + 2) (by simp [selRun]; omega)
    have h1 := h (2 * t + 3) (by simp [selRun]; omega)
    have e0 : (2 * t + 2) % 2 = 0 := by omega
    have e1 : (2 * t + 3) % 2 = 1 := by omega
    have d0 : (2 * t + 2) / 2 + 1 = t + 2 := by omega
    have d1 : (2 * t + 3) / 2 + 1 = t + 2 := by omega
    simp only [selRun, e0, e1, d0, d1, ↓reduceIte, show 2 * t + 2 ≠ 0 by omega, show 2 * t + 3 ≠ 1 by omega,
      show (1 : ℕ) ≠ 0 by omega, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at h0 h1
    rw [show b + (2 * t + 3) = b + 2 * (t + 1) + 1 by omega] at h1
    rw [h1, show b + 2 * (t + 1) + 1 - 1 = b + (2 * t + 2) by omega, h0,
      show b + (2 * t + 2) - 1 = b + 2 * t + 1 by omega, show b + 2 * (t + 1) + 1 - 2 = b + 2 * t + 1 by omega,
      ih (by omega)]
    rfl

theorem fmaRun_holds {w : Wire} {K b : ℕ} (h : RunHolds ops V (fmaRun w K b) (fmaAt K b)) :
    ∀ i < K / 8, V (fmaAt K b + i) = fmaAcc ops w V i := by
  intro i
  induction i using Nat.strong_induction_on with
  | _ i ih =>
    intro hi
    have hg := h i (by simpa [fmaRun] using hi)
    simp only [fmaRun] at hg
    rw [hg, fmaAcc]
    by_cases h32 : i < 32
    · simp [h32, Arg.eval]
    · simp only [h32, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval]
      rw [show fmaAt K b + i - 32 = fmaAt K b + (i - 32) by omega, ih (i - 32) (by omega) (by omega)]

theorem bfly_congr {h n j : ℕ} {v v' : ℕ → ℕ} (hj : v j = v' j) (hh : j + h < n → v (j + h) = v' (j + h)) :
    bfly ops h n v j = bfly ops h n v' j := by
  unfold bfly
  by_cases hc : j + h < n
  · simp only [hc, ↓reduceIte, hj, hh hc]
  · simp only [hc, ↓reduceIte, hj]

theorem bfly_holds {h K pb o : ℕ} (hR : RunHolds ops V (bflyRun h K pb) o) :
    ∀ j < min h (K / 8), V (o + j) = bfly ops h (K / 8) (fun j => V (pb + j)) j := by
  intro j hj
  have hg := hR j (by simpa [bflyRun] using hj)
  simp only [bflyRun] at hg
  rw [hg, bfly]
  by_cases hc : j + h < K / 8
  · simp [hc, Arg.eval, Nat.add_assoc]
  · simp [hc, Arg.eval]

theorem accLast_lt {n j : ℕ} (hj : j < n) : accLast n j < n := by
  unfold accLast
  have := Nat.div_mul_le_self (n - 1 - j) 32
  omega

/-- **The statistics**: on values meeting `statsRuns`, the running max's last gate is `rowMax` and `ρ`'s gate is
`rowRho`, on the values of `x`'s gates. -/
theorem stats_holds {ab w : Wire} {K b : ℕ} (hK : 8 ≤ K) (h : RunsHold ops V (statsRuns ab w K b) b) :
    V (fmaAt K b - 1) = rowMax ops ab V (K - 1) ∧ V (rhoAt K b + 1) = rowRho ops w V K := by
  obtain ⟨hs, hf, h0, h8, h4, h2, h1, hr, -⟩ := h
  have hf : RunHolds ops V (fmaRun w K b) (fmaAt K b) := hf
  have h0 : RunHolds ops V (bfly0Run K b) (bf0At K b) := h0
  have h8 : RunHolds ops V (bflyRun 8 K (bf0At K b)) (bf1At K b) := h8
  have h4 : RunHolds ops V (bflyRun 4 K (bf1At K b)) (bf2At K b) := h4
  have h2 : RunHolds ops V (bflyRun 2 K (bf2At K b)) (bf3At K b) := h2
  have h1 : RunHolds ops V (bflyRun 1 K (bf3At K b)) (bf4At K b) := h1
  have hr : RunHolds ops V (rhoRun K b) (rhoAt K b) := hr
  refine ⟨?_, ?_⟩
  · rw [show fmaAt K b - 1 = b + 2 * (K - 2) + 1 by unfold fmaAt; omega,
      selRun_holds hs (K - 2) (by omega), show K - 2 + 1 = K - 1 by omega]
  have hF := fmaRun_holds hf
  set A := fun j => fmaAcc ops w V (accLast (K / 8) j) with hA
  have ha : ∀ j' < K / 8, V (lastFma K b j') = A j' := by
    intro j' hj'
    rw [show lastFma K b j' = fmaAt K b + accLast (K / 8) j' by unfold lastFma accLast; omega]
    exact hF _ (accLast_lt hj')
  have q0 : ∀ j < min 16 (K / 8), V (bf0At K b + j) = bfly ops 16 (K / 8) A j := by
    intro j hj
    have hg := h0 j (by simpa [bfly0Run] using hj)
    simp only [bfly0Run] at hg
    rw [hg, bfly]
    by_cases hc : j + 16 < K / 8
    · simp [hc, Arg.eval, ha j (by omega), ha (j + 16) hc]
    · simp [hc, Arg.eval, ha j (by omega)]
  have r1 := bfly_holds h8
  have r2 := bfly_holds h4
  have r3 := bfly_holds h2
  have r4 := bfly_holds h1
  have q1 : ∀ j < min 8 (K / 8), V (bf1At K b + j) = bfly ops 8 (K / 8) (bfly ops 16 (K / 8) A) j := by
    intro j hj
    rw [r1 j hj]
    exact bfly_congr (q0 j (by omega)) fun hc => q0 _ (by omega)
  have q2 : ∀ j < min 4 (K / 8), V (bf2At K b + j) =
      bfly ops 4 (K / 8) (bfly ops 8 (K / 8) (bfly ops 16 (K / 8) A)) j := by
    intro j hj
    rw [r2 j hj]
    exact bfly_congr (q1 j (by omega)) fun hc => q1 _ (by omega)
  have q3 : ∀ j < min 2 (K / 8), V (bf3At K b + j) =
      bfly ops 2 (K / 8) (bfly ops 4 (K / 8) (bfly ops 8 (K / 8) (bfly ops 16 (K / 8) A))) j := by
    intro j hj
    rw [r3 j hj]
    exact bfly_congr (q2 j (by omega)) fun hc => q2 _ (by omega)
  have q4 : V (bf4At K b) = sumSq ops w V (K / 8) := by
    have := r4 0 (by omega)
    rw [Nat.add_zero] at this
    rw [this]
    exact bfly_congr (q3 0 (by omega)) fun hc => q3 _ (by omega)
  have g0 := hr 0 (by simp [rhoRun, Run.ofList])
  have g1 := hr 1 (by simp [rhoRun, Run.ofList])
  simp [rhoRun, Run.ofList, Arg.eval] at g0 g1
  rw [g1, g0, show rhoAt K b - 1 = bf4At K b by unfold rhoAt; omega, q4]
  rfl

end Stats

/-! ## The formulas read only the row's inputs -/

section Congr

variable {ops}

theorem rowMax_congr {ab : Wire} {x y : ℕ → ℕ} : ∀ t, (∀ l ≤ t, x l = y l) → rowMax ops ab x t = rowMax ops ab y t
  | 0, h => by simp only [rowMax, h 0 le_rfl]
  | t + 1, h => by
    simp only [rowMax, rowMax_congr t fun l hl => h l (by omega), h (t + 1) le_rfl]

theorem fmaAcc_congr {w : Wire} {x y : ℕ → ℕ} (i : ℕ) (h : ∀ l ≤ 8 * i, x l = y l) :
    fmaAcc ops w x i = fmaAcc ops w y i := by
  induction i using Nat.strong_induction_on with
  | _ i ih =>
    rw [fmaAcc.eq_1 ops w x i, fmaAcc.eq_1 ops w y i, h (8 * i) le_rfl]
    by_cases h32 : i < 32
    · simp only [h32, ↓reduceIte]
    · simp only [h32, ↓reduceIte, ih (i - 32) (by omega) fun l hl => h l (by omega)]

theorem bfly_agree {h n : ℕ} {v v' : ℕ → ℕ} (hv : ∀ j < n, v j = v' j) :
    ∀ j < n, bfly ops h n v j = bfly ops h n v' j :=
  fun j hj => bfly_congr (hv j hj) fun hc => hv _ hc

theorem rowRho_congr {w : Wire} {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) :
    rowRho ops w x K = rowRho ops w y K := by
  have hA : ∀ j < K / 8, fmaAcc ops w x (accLast (K / 8) j) = fmaAcc ops w y (accLast (K / 8) j) := by
    intro j hj
    have := accLast_lt hj
    exact fmaAcc_congr _ fun l hl => h l (by omega)
  unfold rowRho sumSq
  rw [bfly_agree (bfly_agree (bfly_agree (bfly_agree (bfly_agree hA))))  0 (by omega)]

theorem alpha8_congr {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) :
    alpha8 ops x K = alpha8 ops y K := by
  unfold alpha8
  rw [rowRho_congr hK h, rowMax_congr (K - 1) fun l hl => h l (by omega)]

theorem beta8_congr {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) :
    beta8 ops x K = beta8 ops y K := by
  unfold beta8
  rw [rowRho_congr hK h, alpha8_congr hK h]

theorem floor8_congr {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) :
    floor8 ops x K = floor8 ops y K := by
  unfold floor8
  rw [rowRho_congr hK h, alpha8_congr hK h]

end Congr

/-! ## `pc8`'s shared runs -/

section Pre8

variable {ops} {V : ℕ → ℕ} {p : Pipeline} {K b : ℕ}

theorem pre8_runs (h : RunsHold ops V (pre8Runs p K b) b) :
    RunsHold ops V (statsRuns .fabs .id K b) b ∧ RunHolds ops V (alpha8Run K b) (al8At K b) ∧
      RunHolds ops V (ok8Run K b) (ok8At K b) ∧ RunHolds ops V (ep8Run K b) (ep8At K b) ∧
      RunHolds ops V (code8Run p K b) (code8At K b) := by
  obtain ⟨hs, hr⟩ := runsHold_append ops V h
  rw [statsRuns_len] at hr
  obtain ⟨ha, -, -, -, ho, he, -, hc, -⟩ := hr
  exact ⟨hs, ha, ho, he, hc⟩

/-- **`α` and `β`** on values meeting `pc8`'s shared runs: `alpha8` and `beta8` of the values of `x`'s gates. -/
theorem pre8_alpha (hK : 8 ≤ K) (h : RunsHold ops V (pre8Runs p K b) b) :
    V (al8At K b + 3) = alpha8 ops V K ∧ V (al8At K b + 5) = beta8 ops V K := by
  obtain ⟨hs, ha, -⟩ := pre8_runs h
  obtain ⟨hsm, hrho⟩ := stats_holds hK hs
  have g0 := ha 0 (by simp [alpha8Run, Run.ofList])
  have g1 := ha 1 (by simp [alpha8Run, Run.ofList])
  have g2 := ha 2 (by simp [alpha8Run, Run.ofList])
  have g3 := ha 3 (by simp [alpha8Run, Run.ofList])
  have g4 := ha 4 (by simp [alpha8Run, Run.ofList])
  have g5 := ha 5 (by simp [alpha8Run, Run.ofList])
  simp only [alpha8Run, Run.ofList, List.getD_cons_zero, List.getD_cons_succ, List.map_cons, List.map_nil, Arg.eval,
    Wire.eval_id, Nat.add_zero] at g0 g1 g2 g3 g4 g5
  have hα : V (al8At K b + 3) = alpha8 ops V K := by
    rw [g3, g2, g1, g0, hsm, hrho]; rfl
  refine ⟨hα, ?_⟩
  rw [g5, g4, hα, hrho]; rfl

/-- **The floor**: the flag run's first gate is `floor8`. -/
theorem pre8_floor (hK : 8 ≤ K) (h : RunsHold ops V (pre8Runs p K b) b) : V (ok8At K b) = floor8 ops V K := by
  obtain ⟨hs, -, ho, -⟩ := pre8_runs h
  obtain ⟨-, hrho⟩ := stats_holds hK hs
  have g0 := ho 0 (by simp [ok8Run, Run.ofList])
  simp only [ok8Run, Run.ofList, List.getD_cons_zero, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id,
    Nat.add_zero] at g0
  rw [g0, hrho, (pre8_alpha hK h).1]; rfl

/-- **The rescaled noise' codes**: `eTilde8` at `β`'s word of each of `e`'s codes. -/
theorem pre8_eTilde (hK : 8 ≤ K) (h : RunsHold ops V (pre8Runs p K b) b) :
    ∀ l < 32, V (ep8At K b + 3 * l + 2) = eTilde8 ops (beta8 ops V K) (V (K + l)) := by
  intro l hl
  obtain ⟨-, -, -, he, -⟩ := pre8_runs h
  have g0 := he (3 * l) (by simp [ep8Run]; omega)
  have g1 := he (3 * l + 1) (by simp [ep8Run]; omega)
  have g2 := he (3 * l + 2) (by simp [ep8Run]; omega)
  have e0 : 3 * l % 3 = 0 := by omega
  have e1 : (3 * l + 1) % 3 = 1 := by omega
  have e2 : (3 * l + 2) % 3 = 2 := by omega
  have d0 : 3 * l / 3 = l := by omega
  simp only [ep8Run, e0, e1, e2, d0, ↓reduceIte, show (2 : ℕ) ≠ 0 by omega, show (2 : ℕ) ≠ 1 by omega,
    show (1 : ℕ) ≠ 0 by omega, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at g0 g1 g2
  rw [show ep8At K b + 3 * l + 2 = ep8At K b + (3 * l + 2) by omega, g2, show ep8At K b + (3 * l + 2) - 1 = ep8At K b + (3 * l + 1) by omega, g1,
    show ep8At K b + (3 * l + 1) - 1 = ep8At K b + 3 * l by omega, g0, (pre8_alpha hK h).2]
  rfl

/-- **Position `t`'s code**: `code8W` at `α`'s word, `x_t`, the rescaled noise' codes and the form basis' row `t`. -/
theorem pre8_code (hK : 8 ≤ K) (h : RunsHold ops V (pre8Runs p K b) b) :
    ∀ t < K, V (code8 K b t) = code8W ops p (alpha8 ops V K) (V t)
      (fun l => eTilde8 ops (beta8 ops V K) (V (K + l))) (fun l => V (K + 32 + 32 * t + l)) := by
  intro t ht
  obtain ⟨-, -, -, -, hc⟩ := pre8_runs h
  have g0 := hc (3 * t) (by simp [code8Run]; omega)
  have g1 := hc (3 * t + 1) (by simp [code8Run]; omega)
  have g2 := hc (3 * t + 2) (by simp [code8Run]; omega)
  have e0 : 3 * t % 3 = 0 := by omega
  have e1 : (3 * t + 1) % 3 = 1 := by omega
  have e2 : (3 * t + 2) % 3 = 2 := by omega
  have d0 : 3 * t / 3 = t := by omega
  have d1 : (3 * t + 1) / 3 = t := by omega
  simp only [code8Run, e0, e1, e2, d0, d1, ↓reduceIte, show (2 : ℕ) ≠ 0 by omega, show (2 : ℕ) ≠ 1 by omega,
    show (1 : ℕ) ≠ 0 by omega, List.map_cons, List.map_nil, List.map_append, List.map_map, Arg.eval,
    Wire.eval_id] at g0 g1 g2
  have hE : List.map ((Arg.eval ops V) ∘ fun l => Arg.g (ep8At K b + 3 * l + 2)) (List.range 32) =
      List.map (fun l => eTilde8 ops (beta8 ops V K) (V (K + l))) (List.range 32) := by
    apply List.map_congr_left
    intro l hl
    exact pre8_eTilde hK h l (List.mem_range.1 hl)
  rw [hE] at g1
  unfold code8 code8W form8
  rw [show code8At K b + 3 * t + 2 = code8At K b + (3 * t + 2) by omega, g2,
    show code8At K b + (3 * t + 2) - 1 = code8At K b + (3 * t + 1) by omega, g1,
    show code8At K b + (3 * t + 1) - 1 = code8At K b + 3 * t by omega, g0, (pre8_alpha hK h).1]
  rfl

end Pre8

/-! ## `pc8`'s row units -/

section Rows8

variable {ops}

theorem code8W_congr {p : Pipeline} {α x : ℕ} {et et' f f' : ℕ → ℕ} (he : ∀ l < 32, et l = et' l)
    (hf : ∀ l < 32, f l = f' l) : code8W ops p α x et f = code8W ops p α x et' f' := by
  unfold code8W form8
  rw [List.map_congr_left fun l hl => he l (List.mem_range.1 hl),
    List.map_congr_left fun l hl => hf l (List.mem_range.1 hl)]

/-- **A `pc8` row unit computes its codes**: a table whose runs start with `pc8`'s shared runs, in a transcript whose one
unit is correct, holds on position `t`'s code output `code8W` at `alpha8`'s and `beta8`'s words of the row's inputs `x`,
`x_t`, `e`'s codes and the form basis' row `t`. -/
theorem rows8_code (T : Rows) (p : Pipeline) {K : ℕ} (hK : 8 ≤ K) (hb : 65 * K + 32 ≤ T.nIn) {rest : List Run}
    (hr : T.runs = pre8Runs p K T.nIn ++ rest) (X : (T.spec ops).toCircuit.Transcript)
    (hX : (T.spec ops).part.Correct X 0) {t : ℕ} (ht : t < K) (hout : code8 K T.nIn t ∈ T.outs) :
    val X (code8 K T.nIn t) = code8W ops p (alpha8 ops (val X) K) (val X t)
      (fun l => eTilde8 ops (beta8 ops (val X) K) (val X (K + l))) (fun l => val X (K + 32 + 32 * t + l)) := by
  have hlen := pre8Runs_len p K T.nIn
  have hlt : code8 K T.nIn t < (T.spec ops).N := by
    show code8 K T.nIn t < T.nIn + runsLen T.runs
    rw [hr, runsLen_append]
    unfold code8
    unfold post8At at hlen
    omega
  rw [Spec.out_eq (T.spec ops) X hX (List.mem_toFinset.2 hout) hlt]
  have hh := Rows.holds ops T X
  rw [hr] at hh
  have hp := (runsHold_append ops _ hh).1
  have hin : ∀ l < T.nIn, val ((T.spec ops).toCircuit.eval X) l = val X l := fun l hl => Rows.val_input ops T X hl
  rw [pre8_code hK hp t ht, hin t (by omega), alpha8_congr hK fun l hl => hin l (by omega),
    beta8_congr hK fun l hl => hin l (by omega)]
  exact code8W_congr (fun l hl => by rw [hin _ (by omega)]) fun l hl => hin _ (by omega)

/-- **RowA computes `A′`'s codes** (`rows8_code` at `Pc8RowA_v1{K}`). -/
theorem row8A_code (p : Pipeline) {K : ℕ} (hK : 8 ≤ K) (X : ((row8A p K).spec ops).toCircuit.Transcript)
    (hX : ((row8A p K).spec ops).part.Correct X 0) {t : ℕ} (ht : t < K) :
    val X (code8 K (rowIn K false) t) = code8W ops p (alpha8 ops (val X) K) (val X t)
      (fun l => eTilde8 ops (beta8 ops (val X) K) (val X (K + l))) (fun l => val X (K + 32 + 32 * t + l)) :=
  rows8_code (row8A p K) p hK (by simp [row8A, rowIn]) rfl X hX ht
    (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨t, List.mem_range.2 ht,
      rfl⟩))))

/-- **RowB computes `B̃`'s codes** (`rows8_code` at `Pc8RowB_v1{K}`). -/
theorem row8B_code (p : Pipeline) {K : ℕ} (hK : 8 ≤ K) (X : ((row8B p K).spec ops).toCircuit.Transcript)
    (hX : ((row8B p K).spec ops).part.Correct X 0) {t : ℕ} (ht : t < K) :
    val X (code8 K (rowIn K true) t) = code8W ops p (alpha8 ops (val X) K) (val X t)
      (fun l => eTilde8 ops (beta8 ops (val X) K) (val X (K + l))) (fun l => val X (K + 32 + 32 * t + l)) :=
  rows8_code (row8B p K) p hK (by simp [row8B, rowIn]) rfl X hX ht
    (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨t, List.mem_range.2 ht,
      rfl⟩))))

end Rows8

end Pouw.PearlC.RowCircuit
