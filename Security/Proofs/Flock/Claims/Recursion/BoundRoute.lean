import Proofs.Flock.Claims.Recursion.BoundAccepts
import Proofs.Flock.Claims.Recursion.BoundForks

/-!
# The normalized record and the readers' played rounds (option (d))

Pieces of `KSBoundDeployed`'s route from the bridge (`RecDeploy.ksBoundRoute_of_bridge_cols`, `BoundRouteCols`):

* A record whose readers read the last message as its representative under `final` (`Bound.rep`,
  `RecDeploy.nfBrings`). `IVStar` and the opened view read the last message only through `final`, so the two records'
  inner views give `E*` the same decisions, calls and openings (`Bound.run_nf`). `E*` reads an inner view only through
  its decision and its openings (`served_eval_congr`), so `E*`'s output and `KSA` are the same at both
  (`RecDeploy.ksA_of_nf`), and so are `CallsLogColl` and `InnerAcceptsColl` (`RecDeploy.callsLogColl_nf`,
  `RecDeploy.innerAcceptsColl_nf`).
* The readers' rounds before the last are the run's, off a collision (`RecDeploy.readersRun_played`, from
  `ReadersRunPlayed`).
-/

namespace Claims.Recursion

open Claims

/-! ## `E*` reads an inner view only through its decision and its openings -/

theorem rp_eval_bind_fst {R α β : Type} (H : Msg → Msg) (run : R → Trace) (p : RP R α) (f : α → RP R β) :
    ((p.bind f).eval H run).1 = ((f (p.eval H run).1).eval H run).1 :=
  (RP.eval_bind H run p f).1

theorem waitAcc_rel {R Op : Type} (H : Msg → Msg) (run₁ run₂ : R → Trace) (opens : Seen → List Op) (k : ℕ)
    (h : ∀ r, (run₁ r).seen.out = (run₂ r).seen.out ∧ opens (run₁ r).seen = opens (run₂ r).seen) :
    ∀ (rs : List R) (acc₁ acc₂ : List Seen), acc₁.length = acc₂.length → acc₁.flatMap opens = acc₂.flatMap opens →
      ((waitAcc k rs acc₁).eval H run₁).1.length = ((waitAcc k rs acc₂).eval H run₂).1.length ∧
        ((waitAcc k rs acc₁).eval H run₁).1.flatMap opens = ((waitAcc k rs acc₂).eval H run₂).1.flatMap opens
  | [], _, _, hl, hf => ⟨hl, hf⟩
  | r :: rs, acc₁, acc₂, hl, hf => by
    obtain ⟨ho, hop⟩ := h r
    by_cases hk : k ≤ acc₁.length
    · have hk' : k ≤ acc₂.length := hl ▸ hk
      simp only [waitAcc, hk, hk', ↓reduceIte, RP.eval]
      exact ⟨hl, hf⟩
    · have hk' : ¬ k ≤ acc₂.length := hl ▸ hk
      simp only [waitAcc, hk, hk', ↓reduceIte, RP.eval]
      by_cases hv : (run₁ r).seen.out = .acc
      · have hv' : (run₂ r).seen.out = .acc := ho ▸ hv
        simp only [hv, hv', ↓reduceIte]
        exact waitAcc_rel H run₁ run₂ opens k h rs _ _ (by simp [hl]) (by simp [List.flatMap_append, hf, hop])
      · have hv' : ¬ (run₂ r).seen.out = .acc := ho ▸ hv
        simp only [hv, hv', ↓reduceIte]
        exact waitAcc_rel H run₁ run₂ opens k h rs _ _ hl hf

theorem firstWait_rel {R W Op : Type} (H : Msg → Msg) (run₁ run₂ : R → Trace) (opens : Seen → List Op) (k : ℕ)
    (assemble : List Op → Option W) (check : W → Bool)
    (h : ∀ r, (run₁ r).seen.out = (run₂ r).seen.out ∧ opens (run₁ r).seen = opens (run₂ r).seen) :
    ∀ rss : List (List R), ((firstWait k opens assemble check rss).eval H run₁).1 =
      ((firstWait k opens assemble check rss).eval H run₂).1
  | [] => rfl
  | rs :: rss => by
    simp only [firstWait, rp_eval_bind_fst]
    rw [(waitAcc_rel H run₁ run₂ opens k h rs [] [] rfl rfl).2]
    split
    · split
      · rfl
      · exact firstWait_rel H run₁ run₂ opens k assemble check h rss
    · exact firstWait_rel H run₁ run₂ opens k assemble check h rss

theorem erep_rel {D W Op : Type} (H : Msg → Msg) (run₁ run₂ : D → Trace) (m k K : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (d₀ : D) (ρ : Fin m → Fin K → D)
    (h : ∀ r, (run₁ r).seen.out = (run₂ r).seen.out ∧ opens (run₁ r).seen = opens (run₂ r).seen) :
    ((erep m k K opens assemble check d₀ ρ).eval H run₁).1 =
      ((erep m k K opens assemble check d₀ ρ).eval H run₂).1 := by
  simp only [erep, RP.eval]
  rw [(h d₀).1]
  split_ifs
  · exact firstWait_rel H run₁ run₂ opens k assemble check h _
  · rfl

/-- **`E^rep` served by `E₂` reads what the inner stage sees only through its decision and its openings**: two readings
`dec`, `dec'` of `E₂`'s output that agree there give one output. -/
theorem served_eval_congr {Coin W W₂ Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {Ω : Type}
    (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (E₂ : Extractor Coin W₂ V)
    (dec dec' : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg)
    (h : ∀ r o, (dec x r o).out = (dec' x r o).out ∧ opens (dec x r o) = opens (dec' x r o))
    (ω : Ω) (r₀ : Fin (V.k x) → Coin) (c : (served V E₂ dec m k K opens assemble check).C x) :
    ((served V E₂ dec m k K opens assemble check).eval H P x ω r₀ c).1 =
      ((served V E₂ dec' m k K opens assemble check).eval H P x ω r₀ c).1 := by
  simp only [Extractor.eval, served, serve_eq]
  rw [(RP.serve_eval H _ _ _).1, (RP.serve_eval H _ _ _).1]
  refine erep_rel H _ _ _ _ _ _ _ _ _ _ fun d => ?_
  simp only [servedRun_seen, outerStage, RP.eval_bind_ret]
  exact h d.1 _

theorem coinSeq_congr {Coin : Type} [Inhabited Coin] {k : ℕ} {r r' : Fin k → Coin} {i : ℕ}
    (h : ∀ hi : i < k, r ⟨i, hi⟩ = r' ⟨i, hi⟩) : coinSeq r i = coinSeq r' i := by
  unfold coinSeq
  split_ifs with hi
  · exact h hi
  · rfl

/-! ## The last message, read as its representative -/

section Bound

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **Messages with the last read as its representative** (`Bound.rep`): `IVStar`'s rounds, each the message `ms` holds
there, the last one's representative. -/
noncomputable def Bound.nfMsgs (Bd : Bound Coin Col T) (x : Msg) (ms : List Msg) : List Msg :=
  (List.range (Bd.DE.V.k x)).map fun i => if i + 1 = Bd.DE.V.k x then Bd.rep (ms.getD i []) else ms.getD i []

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem Bound.nfMsgs_getD_lt (Bd : Bound Coin Col T) (x : Msg) (ms : List Msg) {i : ℕ}
    (hi : i + 1 < Bd.DE.V.k x) : (Bd.nfMsgs x ms).getD i [] = ms.getD i [] := by
  have h1 : i < Bd.DE.V.k x := by omega
  have h2 : i + 1 ≠ Bd.DE.V.k x := by omega
  simp [Bound.nfMsgs, List.getD_eq_getElem?_getD, h1, h2]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem Bound.nfMsgs_getD_last (Bd : Bound Coin Col T) (x : Msg) (ms : List Msg) (hk : 0 < Bd.DE.V.k x) :
    (Bd.nfMsgs x ms).getD (Bd.DE.V.k x - 1) [] = Bd.rep (ms.getD (Bd.DE.V.k x - 1) []) := by
  have h1 : Bd.DE.V.k x - 1 < Bd.DE.V.k x := by omega
  have h2 : Bd.DE.V.k x - 1 + 1 = Bd.DE.V.k x := by omega
  simp [Bound.nfMsgs, List.getD_eq_getElem?_getD, h1, h2]

/-- **`IVStar` decides alike on the representative**: one decision and one log. -/
theorem Bound.decAt_nf (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) : Bd.decAt H x (Bd.nfMsgs x ms) r = Bd.decAt H x ms r := by
  rcases Nat.eq_zero_or_pos (Bd.DE.V.k x) with hk | hk
  · unfold Bound.decAt
    simp only [hk, List.range_zero, List.map_nil]
  · exact Bd.run_ivstar_congr H x _ _ r (fun i hi => Bd.nfMsgs_getD_lt x ms hi)
      (by rw [Bd.nfMsgs_getD_last x ms hk, Bd.final_rep])

/-- **`IVStar`'s run on the representative**: the same decision, the same calls, and the same opened view. -/
theorem Bound.run_nf (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) :
    (run H (replay (Bd.nfMsgs x ms)) Bd.IVStar x () r).out = (run H (replay ms) Bd.IVStar x () r).out ∧
      (run H (replay (Bd.nfMsgs x ms)) Bd.IVStar x () r).vLog = (run H (replay ms) Bd.IVStar x () r).vLog ∧
      Bd.openView (run H (replay (Bd.nfMsgs x ms)) Bd.IVStar x () r).seen =
        Bd.openView (run H (replay ms) Bd.IVStar x () r).seen := by
  rcases Nat.eq_zero_or_pos (Bd.DE.V.k x) with hk | hk
  · rw [Bd.run_ivstar_zero H x _ r hk, Bd.run_ivstar_zero H x _ r hk]
    exact ⟨rfl, rfl, rfl⟩
  rw [Bd.run_ivstar H x _ r hk, Bd.run_ivstar H x _ r hk, Bd.decAt_nf]
  refine ⟨rfl, rfl, ?_⟩
  have hpre : (List.range (Bd.DE.V.k x - 1)).map
      (fun i => ((Bd.nfMsgs x ms).getD i [], Out.msg (Bd.ans (coinSeq r i)))) =
      (List.range (Bd.DE.V.k x - 1)).map (fun i => (ms.getD i [], Out.msg (Bd.ans (coinSeq r i)))) :=
    List.map_congr_left fun i hi => by
      rw [List.mem_range] at hi
      rw [Bd.nfMsgs_getD_lt x ms (by omega)]
  simp only [Trace.seen, hpre, Bd.nfMsgs_getD_last x ms hk]
  rcases hf : Bd.final (ms.getD (Bd.DE.V.k x - 1) []) with _ | p
  · rw [Bd.rep_of_none hf]
  · exact Bd.openView_congr_final _ _ _ _ _ p ((Bd.final_rep _).trans hf) hf

end Bound

end Claims.Recursion

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin)
open Claims.Recursion (Bound Inner)

section Normal

variable {Col T : Type} [DecidableEq Col]

/-- **The deployment's brings, its readers reading the last inner message as its representative** (`Bound.nfMsgs`). -/
noncomputable def nfBrings (Bd : Bound Coin Col T) (b : Brings) : Brings :=
  { b with rd := ⟨fun x ws => Bd.nfMsgs (xIOf x) (b.rd.msgs x ws), b.rd.calls⟩ }

theorem innerView_nf (H : Msg → Msg) (Bd : Bound Coin Col T) (b : Brings) (x : Msg) {k : ℕ} (r : Fin k → Coin)
    (ws : List Claims.Flock.Wit) :
    (innerView H Bd.IVStar (nfBrings Bd b).rd x r ws).out = (innerView H Bd.IVStar b.rd x r ws).out ∧
      Bd.openView (innerView H Bd.IVStar (nfBrings Bd b).rd x r ws) = Bd.openView (innerView H Bd.IVStar b.rd x r ws) :=
  let h := Bd.run_nf H (xIOf x) (b.rd.msgs x ws) fun i => coinSeq r i.val
  ⟨h.1, h.2.2⟩

omit [DecidableEq Col] in
/-- The inner view's run, at `IVStar`, is `IVStar`'s run of the readers' messages. -/
theorem innerRun_bound (H : Msg → Msg) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (ivScope : Msg → Bool) (b : Brings) (x : Msg) (r : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin)
    (ws : List Claims.Flock.Wit) :
    Claims.Recursion.innerRun H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) x r ws =
      run H (replay (b.rd.msgs x ws)) Bd.IVStar (xIOf x) () fun i => coinSeq r i.val := by
  show run H (replay ((innerView H Bd.IVStar b.rd x r ws).msgs.map Prod.fst)) Bd.IVStar (xIOf x) ()
    (fun i => coinSeq r i.val) = _
  unfold innerView
  rw [run_replay_seen]

variable {Col' T' : Type} [DecidableEq Col']

omit [DecidableEq Col'] in
/-- **`E*` gives one output at the deployed record and the normalized one.** -/
theorem E_eval_nf (H : Msg → Msg) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool)
    (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (r : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin)
    (c : (Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO).C x) :
    ((Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO).eval H P x ω r
        c).1 =
      ((Claims.Recursion.E H (deployRec Bd.IVStar ivScope (nfBrings Bd b)) (Inner.ofFlock Bd.deploy decI) DO
        decO).eval H P x ω r c).1 := by
  refine Claims.Recursion.served_eval_congr H P _ _ _
    (Claims.Recursion.dec H (deployRec Bd.IVStar ivScope (nfBrings Bd b))) _ _ _ _ _ _ x (fun r o => ?_) ω r c
  cases o with
  | none => exact ⟨rfl, rfl⟩
  | some ws =>
    obtain ⟨h1, h2⟩ := innerView_nf H Bd b x r ws
    exact ⟨h1.symm, congrArg (fun v => [Bd.DE.cols v]) h2.symm⟩

/-- **`KSA` at the normalized record is `KSA` at the deployed one**: `E*`'s output is the same (`E_eval_nf`), and the
bound reads no inner view. -/
theorem ksA_of_nf [Zero Col'] (H : Msg → Msg) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (h : Claims.Recursion.KSA H (deployRec Bd.IVStar ivScope (nfBrings Bd b)) (Inner.ofFlock Bd.deploy decI) DO) :
    Claims.Recursion.KSA H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO := by
  intro decO hdec x Ω _ _ P
  have hE : Claims.PrBad H (deployRec Bd.IVStar ivScope b).V
      (Claims.Recursion.R (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P =
    Claims.PrBad H (deployRec Bd.IVStar ivScope (nfBrings Bd b)).V
      (Claims.Recursion.R (deployRec Bd.IVStar ivScope (nfBrings Bd b)) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRec Bd.IVStar ivScope (nfBrings Bd b)) (Inner.ofFlock Bd.deploy decI) DO decO)
      x P := by
    unfold Claims.PrBad
    exact Claims.Pr_congr fun s => by
      rw [E_eval_nf H Bd decI ivScope b DO decO x P s.1 s.2.1 s.2.2]
      try exact Iff.rfl
  exact hE.trans_le (h decO hdec x Ω P)

omit [DecidableEq Col'] in
theorem innerAcceptsColl_nf (H : Msg → Msg) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (hA : Claims.Recursion.InnerAcceptsColl H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO) :
    Claims.Recursion.InnerAcceptsColl H (deployRec Bd.IVStar ivScope (nfBrings Bd b)) (Inner.ofFlock Bd.deploy decI)
      DO := by
  intro decO x Ω P ω r c ws hg
  rcases hA decO x Ω P ω r c ws hg with h | h
  · exact Or.inl ((innerView_nf H Bd b x r ws).1.trans h)
  · exact Or.inr h

omit [DecidableEq Col'] in
theorem callsLogColl_nf (H : Msg → Msg) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (hL : Claims.Recursion.CallsLogColl H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO) :
    Claims.Recursion.CallsLogColl H (deployRec Bd.IVStar ivScope (nfBrings Bd b)) (Inner.ofFlock Bd.deploy decI)
      DO := by
  intro decO x Ω P ω r c ws hg
  rcases hL decO x Ω P ω r c ws hg with h | h
  · left
    intro e he
    rw [innerRun_bound] at he
    apply h
    rw [innerRun_bound H Bd decI ivScope b x r ws]
    have key : ∀ c : Fin (Bd.IVStar.k (xIOf x)) → Coin,
        e ∈ (run H (replay (Bd.nfMsgs (xIOf x) (b.rd.msgs x ws))) Bd.IVStar (xIOf x) () c).vLog →
          e ∈ (run H (replay (b.rd.msgs x ws)) Bd.IVStar (xIOf x) () c).vLog := fun c hc => by
      rw [← (Bd.run_nf H (xIOf x) (b.rd.msgs x ws) c).2.1]
      exact hc
    exact key _ he
  · exact Or.inr h

end Normal

/-- **The readers' rounds before the last are the run's, off a collision**, where `E_O` gives witnesses on an accepted
run (`ReadersRunPlayed` through the staging, as `vstarAccepts_bound` reads it), at an inner verifier that plays all its
rounds on replayed messages. -/
theorem readersRun_played {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings)
    (DO : Claims.Flock.Deploy Coin Col T)
    (hplay : ∀ x ms (c : Fin (I.V.k (xIOf x)) → Coin),
      (run H (replay ms) I.V (xIOf x) () c).msgs.length = I.V.k (xIOf x))
    (hT : ReadersTrue H I.V b DO) (hRRP : ReadersRunPlayed H I.V b DO) {decO : List (ℕ × Col) → List T} {x : Msg}
    {Ω : Type} {P : Prover Ω} {ω : Ω} {r : Fin ((deployRec I.V ivScope b).V.k x) → Coin}
    {c : (Claims.Recursion.EO (deployRec I.V ivScope b) I DO decO).C x} {ws : List Claims.Flock.Wit}
    (hg : Claims.Recursion.EOGives H (deployRec I.V ivScope b) I DO decO P x ω r c ws) :
    (∀ j, j + 1 < I.V.k (xIOf x) → (b.rd.msgs x ws).getD j [] =
        ((run H P (deployRec I.V ivScope b).V x ω r).msgs.map Prod.fst).getD j []) ∨
      ∃ u v, findColl ((deployRec I.V ivScope b).calls x ws) = some (u, v) ∧ Collision H u v := by
  have ho := Claims.Recursion.eoGives_outerHolds hg
  rw [runVstar_staged H I ivScope b P ω r hg.1] at ho
  have hl := runAs_length H I.V b x P ω r hg.1
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  have hsp := stagedAll_run H I.V b x P ω r hg.1
  rcases hRRP x _ _ ws hl hc hsp ho (hT x _ _ ws hl hc hsp ho) with hag | hcoll
  · left
    intro j hj
    have hj' : j < (innerRunAt H I.V b.rd x ws (runCs (I.V.k (xIOf x) + kS x) r)).msgs.length := by
      unfold innerRunAt
      rw [hplay]
      omega
    have h1 := hag j hj' hj
    unfold innerRunAt at h1 hj'
    rw [Claims.Recursion.VStar.run_replay_fst H I.V (xIOf x) (b.rd.msgs x ws) _ hj'] at h1
    show _ = ((run H P (V I.V b.st) x ω r).msgs.map Prod.fst).getD j []
    rw [h1]
    simp only [runAs, List.getD_eq_getElem?_getD, List.map_take,
      List.getElem?_take_of_lt (show j < I.V.k (xIOf x) + kS x by omega)]
  · exact Or.inr hcoll

/-! ## The bound record's answers -/

theorem bdDeployed_ans (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (c : Coin) :
    (bdDeployed rel nextE colsE wE join rowIn final w).ans c = c.bytes := by
  simp only [bdDeployed]

end RecDeploy
