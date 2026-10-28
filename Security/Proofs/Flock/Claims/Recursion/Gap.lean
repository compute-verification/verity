import Proofs.Flock.Claims.Recursion.Inner
import Proofs.Flock.Claims.Recursion.Abort
import Proofs.Flock.Claims.Recursion.Registered

/-!
# `InnerKS` does not follow from `Inner.KS` and the code facts; `Inner.KSAbort` excludes the deployment

A compiled verifier `gD` over an inner protocol `gI` and an outer `--zk` Flock deployment `gDO` with every code fact
`KS_of_flockKS` reads, `Inner.KS` at `κ_I = γ_I = 0` and `Claims.Flock.KS` at `gDO`, whose recursion's `KS` fails at
the errors its proof gives (`KSA`) and at the draft's (`KS`), so `InnerKS` fails (`innerKS_gap`).

The inner verifier accepts whatever it is sent, in one round; the inner `E^rep` makes one wait of one rerun and needs it
accepted. The compiled verifier's one outer session is a round that rejects at its own coin half the time. `E*` reruns
the inner stage at fresh outer coins, so its one inner rerun is seen rejected half the time, whatever the inner
verifier decides, and `E*` fails on a quarter of the draws while `κ = 2·κ_O < 1/4` and `B`, `Be` find nothing. No
inner prover makes the inner verifier reject on a coin it never reads, so `Inner.KS` cannot bound that failure:
`Inner.KS` bounds `E^rep` against runs whose acceptance is the inner verifier's alone. A prover that aborts at a
fresh coin of its own does make `gI`'s one rerun rejected half the time, so `gI` fails `Inner.KSAbort`
(`gI_not_KSAbort`), the one hypothesis of `KS_of_flockKS` the deployment lacks.
-/

namespace Claims.Recursion

open Claims

/-! ## The objects -/

/-- The inner protocol: one round that accepts whatever it is sent; `E^rep` makes one wait of one rerun and assembles
a witness from it only if it accepted. -/
def gI : Inner Bool Unit where
  V := ⟨fun _ => 1, fun _ _ _ => .ret .acc⟩
  R := fun _ _ => True
  Scope := fun _ => True
  Op := Unit
  opens := fun _ => [()]
  assemble := fun _ l => l.head?
  check := fun _ _ => true
  waits := fun _ => 1
  k := fun _ => 1
  cap := fun _ => 1
  κ := fun _ => 0
  γ := fun _ => 0

/-- The outer statement: a circuit with no gates and no outputs, so every witness is in its relation. -/
def gs₀ : Flock.Stmt :=
  ⟨0, 0, ⟨0, [], []⟩, fun _ => [], fun _ => [], [],
    ⟨false, false, false, "", 0, false, false, false, false, 0, false, 0, false, false, false, false⟩⟩

/-- The outer verifier: one round, accepting at its coin `false`. -/
def gVO : Verifier Bool := ⟨fun _ => 1, fun _ _ cs => .ret (if cs.getD 0 true = false then .acc else .rej)⟩

/-- The outer deployment: `k = 400` accepted reruns per wait, `m′ = 8` waits, `K = 204800`, so
`κ_O = 2^{−8} + 2^{−8} + e^{−50}`. -/
noncomputable def gDO : Flock.Deploy Bool ℕ Unit where
  zk := true
  V := gVO
  parse := fun _ => some gs₀
  scopeCheck := fun _ => false
  table := fun _ _ => some gs₀
  cols := fun _ _ => []
  enc := fun _ _ => 0
  len := fun _ => 0
  θ := fun _ => 0
  dec := fun _ => ([], 0)
  read := fun _ => ⟨[]⟩
  εtable := fun _ => 0
  N₀ := fun _ => 100 * Real.exp 1
  weight := fun _ => 0
  waits := fun _ => 8
  cap := fun _ => 204800
  Tape := Unit
  P := fun _ => ⟨fun _ _ _ => .ret []⟩
  δc := fun _ _ => 0

/-- The outer session's numbers, `gDO`'s. -/
noncomputable def gNums : Nums := ⟨0, 100 * Real.exp 1, 0, 1, 8, 204800, 0, 0⟩

/-- The compiled verifier's answer: the first message must be `H []` (the commit string `D.calls` opens), the second
round decides at the outer session's coin. -/
def gVerdict (h : Msg) (as : List Msg) (cs : List Bool) : Out :=
  if as.head? = some h then (if as.length < 2 then .msg [] else if cs.getD 1 true then .rej else .acc) else .rej

/-- The compiled verifier: the inner round, then the outer session's. -/
def gV : Verifier Bool := ⟨fun _ => 2, fun _ as cs => .query [] fun h => .ret (gVerdict h as cs)⟩

/-- The recursion's statement: the empty inner statement, no circuit commitment. -/
def gStmt : CompiledStatement := ⟨[], none, none, ⟨0, 0, 0, 0, false, none, none, false⟩, []⟩

/-- The compiled verifier's deployment: one outer session, its view the run's second round. -/
noncomputable def gD (H : Msg → Msg) : Deploy Bool where
  V := gV
  parse := fun _ => some gStmt
  scopeCheck := fun _ => false
  kS := fun _ => 0
  nC := fun _ => 0
  stage := fun _ _ _ => .ret []
  vstar := fun _ _ => [[]]
  nS := fun _ => 1
  session := fun _ _ v => ⟨v.msgs.drop 1, v.out⟩
  innerView := fun _ _ _ _ => ⟨[([], .acc)], .acc⟩
  calls := fun _ _ => [([], H [])]
  nums := fun _ _ => gNums
  saltLen := 0
  saltPrefix := []
  mask := fun _ _ => 0
  Tape := Unit
  P := fun _ => ⟨fun _ _ _ => .ret []⟩
  δc := fun _ _ => 1

/-! ## The compiled verifier's runs -/

/-- **The compiled verifier's run**: the prover's first message must be `H []`; the second round decides at coin 1. -/
theorem gV_run (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (gV.k x) → Bool) :
    run H P gV x ω r =
      if ((P.next x [] ω).eval H).1 = H [] then
        ⟨[(((P.next x [] ω).eval H).1, .msg []),
          (((P.next x [(((P.next x [] ω).eval H).1, [])] ω).eval H).1, if coinSeq r 1 then .rej else .acc)],
          ((P.next x [] ω).eval H).2 ++ ((P.next x [(((P.next x [] ω).eval H).1, [])] ω).eval H).2,
          [([], H []), ([], H [])], if coinSeq r 1 then .rej else .acc⟩
      else ⟨[(((P.next x [] ω).eval H).1, .rej)], ((P.next x [] ω).eval H).2, [([], H [])], .rej⟩ := by
  by_cases h : ((P.next x [] ω).eval H).1 = H []
  · simp only [h, ↓reduceIte]
    simp only [run, stepWith, gV, gVerdict, OC.eval, h]
    split_ifs <;> simp_all
  · simp only [h, ↓reduceIte]
    simp [run, stepWith, gV, gVerdict, OC.eval, h]

theorem gD_run (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((gD H).V.k x) → Bool) :
    run H P (gD H).V x ω r =
      if ((P.next x [] ω).eval H).1 = H [] then
        ⟨[(((P.next x [] ω).eval H).1, .msg []),
          (((P.next x [(((P.next x [] ω).eval H).1, [])] ω).eval H).1, if coinSeq r 1 then .rej else .acc)],
          ((P.next x [] ω).eval H).2 ++ ((P.next x [(((P.next x [] ω).eval H).1, [])] ω).eval H).2,
          [([], H []), ([], H [])], if coinSeq r 1 then .rej else .acc⟩
      else ⟨[(((P.next x [] ω).eval H).1, .rej)], ((P.next x [] ω).eval H).2, [([], H [])], .rej⟩ :=
  gV_run H P x ω r

theorem gD_out (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((gD H).V.k x) → Bool) :
    (run H P (gD H).V x ω r).out = .acc ↔ ((P.next x [] ω).eval H).1 = H [] ∧ coinSeq r 1 = false := by
  rw [gD_run]
  by_cases h : ((P.next x [] ω).eval H).1 = H [] <;> by_cases hc : coinSeq r 1 = true <;> simp [h, hc]

/-- The compiled verifier's only SHA-512 call is `H []`. -/
theorem gD_vLog (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((gD H).V.k x) → Bool) :
    ∀ e ∈ (run H P (gD H).V x ω r).vLog, e.1 = [] := by
  rw [gD_run]
  split_ifs <;> simp

theorem gDO_run (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (gDO.V.k x) → Bool) :
    run H P gDO.V x ω r =
      ⟨[(((P.next x [] ω).eval H).1, if coinSeq r 0 = false then .acc else .rej)], ((P.next x [] ω).eval H).2, [],
        if coinSeq r 0 = false then .acc else .rej⟩ := by
  simp only [run, stepWith, gDO, gVO, OC.eval]
  split_ifs <;> simp_all

theorem gI_run_out (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (gI.V.k x) → Bool) :
    (run H P gI.V x ω r).out = .acc := by
  simp [run, stepWith, gI, OC.eval]

theorem coinSeq_fork {Coin : Type} [Inhabited Coin] {k : ℕ} (o : ℕ) (r d : Fin k → Coin) (i : ℕ) :
    coinSeq (fork o r d) i = if i < o then coinSeq r i else coinSeq d i := by
  unfold coinSeq fork
  split_ifs <;> simp_all

/-- `findColl` finds nothing among evaluations at one query. -/
theorem findColl_fst_const (q : Msg) : ∀ l : List (Msg × Msg), (∀ e ∈ l, e.1 = q) → findColl l = none
  | [] => fun _ => rfl
  | (q', a) :: l => fun h => by
    have hq' : q' = q := h _ List.mem_cons_self
    have hn : l.find? (fun p => decide (p.1 ≠ q') && decide (p.2 = a)) = none := by
      rw [List.find?_eq_none]
      intro p hp
      simp [h p (List.mem_cons_of_mem _ hp), hq']
    simp only [findColl, hn]
    exact findColl_fst_const q l fun e he => h e (List.mem_cons_of_mem _ he)

/-! ## Uniform coordinates -/

theorem UnifMap.fst {A B : Type} [Fintype A] [Fintype B] [Nonempty B] : UnifMap (fun p : A × B => p.1) := fun G => by
  rw [Ex_prod]
  simp only [Ex_const]

theorem UnifMap.snd {A B : Type} [Fintype A] [Fintype B] [Nonempty A] : UnifMap (fun p : A × B => p.2) := fun G => by
  rw [Ex_prod]
  simp only [Ex_const]

theorem UnifMap.coord {ι A : Type} [Fintype ι] [DecidableEq ι] [Fintype A] [Nonempty A] (i : ι) :
    UnifMap (fun f : ι → A => f i) := fun G => Ex_pi_coord (C := fun _ => A) i G

/-! ## The code facts hold -/

theorem gRefuses (H : Msg → Msg) : Refuses H (gD H) := fun _ _ _ _ _ _ => ⟨gStmt, rfl, rfl⟩

theorem gInnerRounds (H : Msg → Msg) : InnerRounds (gD H) gI := fun _ => by show 1 ≤ 2; omega

theorem gSessionNums (H : Msg → Msg) : SessionNums (gD H) gDO := fun _ _ _ => ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem gTableErrNonneg (H : Msg → Msg) : TableErrNonneg (gD H) := fun _ _ => le_refl (0 : ℝ)

theorem gSessionAccepts (H : Msg → Msg) : SessionAccepts (gD H) := fun _ _ _ _ h => h

theorem gSessionRounds (H : Msg → Msg) : SessionRounds (gD H) gI gDO := by
  intro x s ms
  have hs : (s : ℕ) = 0 := Nat.lt_one_iff.1 s.isLt
  simp [hs, gI, gD, gDO, gVO, gV, xI]

theorem gInnerViewRun (H : Msg → Msg) : InnerViewRun H (gD H) gI gDO := by
  intro decO x Ω P ω r c ws _
  simp [innerRun, run, stepWith, replay, gI, gD, OC.eval, Trace.seen]

theorem gCallsRecord (H : Msg → Msg) : CallsRecord H (gD H) gI gDO := by
  intro decO x Ω P ω r c ws _
  refine ⟨fun e he => ?_, fun e he => ?_⟩
  · simp [innerRun, run, stepWith, replay, gI, OC.eval] at he
  · simp only [gD, List.mem_singleton] at he
    subst he
    rfl

theorem gInnerAccepts (H : Msg → Msg) : InnerAccepts H (gD H) gI gDO := fun _ _ _ _ _ _ _ _ _ => rfl

/-- The inner verifier never answers with a message, so it is public-coin at any reading of its messages. -/
theorem gPublicCoin (H : Msg → Msg) : PublicCoin H gI fun _ _ => false := by
  intro x as cs c m _ h
  simp [gI, OC.eval] at h

theorem gCallsNil (H : Msg → Msg) : CallsNil H (gD H) := fun _ e he => by
  simp only [gD, List.mem_singleton] at he
  subst he
  rfl

theorem gCallsOpen (H : Msg → Msg) : CallsOpen H (gD H) gI gDO (Opening.ofCall id) := by
  intro decO x Ω P ω r c ws hg j hj
  have hj0 : j = 0 := Nat.lt_one_iff.1 hj
  subst hj0
  have h1 := ((gD_out H P x ω r).1 hg.1).1
  refine ⟨(gD H).calls x ws, List.Subset.refl _, [], ?_, rfl⟩
  rw [gD_run, ite_eq_left h1]
  simp [h1, gD]

theorem gSessionRuns (H : Msg → Msg) : SessionRuns H (gD H) gI gDO := by
  intro x s Ω P ω r d hacc
  have hs : (s : ℕ) = 0 := Nat.lt_one_iff.1 s.isLt
  have hoff : offO H (gD H) gI gDO P x ω s r = 1 := by simp [offO, hs, gI, gD]
  have h1 := ((gD_out H P x ω r).1 hacc).1
  rw [hoff]
  refine ⟨?_, fun e he => ?_⟩
  · rw [gDO_run, gD_run H P x ω (fork 1 r d), ite_eq_left h1]
    simp only [sessP, sessProver, hoff]
    rw [gD_run H P x ω r, ite_eq_left h1]
    have hc : coinSeq (sessCoins 1 d (gDO.V.k (sessStmt H (gD H) gI P x ω (↑s) r))) 0 = coinSeq d 1 := by
      simp [coinSeq, sessCoins, gDO, gVO]
    have hf : coinSeq (fork 1 r d) 1 = coinSeq d 1 := by rw [coinSeq_fork]; simp
    rw [hc, hf]
    cases coinSeq d 1 <;> simp [outMsg, Trace.seen, gD]
  · rw [gDO_run] at he
    simp at he

/-! ## The inner and outer KS hold -/

/-- One wait of one rerun, accepted, assembles what its openings give. -/
theorem firstWait_one_acc {R W Op : Type} (H : Msg → Msg) (run : R → Trace) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (ρ : Fin 1 → Fin 1 → R) (w : W) (h : (run (ρ 0 0)).out = .acc)
    (ha : ∀ v, assemble (opens v) = some w) (hc : check w = true) :
    ((firstWait 1 opens assemble check ((List.finRange 1).map fun i => (List.finRange 1).map (ρ i))).eval H run).1 =
      some w := by
  simp [firstWait, waitAcc, RP.bind, RP.eval, Trace.seen, h, ha, hc, List.finRange_succ]

theorem gI_E_some (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (gI.V.k x) → Bool)
    (ρ : gI.E.C x) : (gI.E.eval H P x ω r₀ ρ).1 = some () := by
  simp only [Inner.E, ERep, Extractor.eval, RP.eval, Trace.seen, gI_run_out, ↓reduceIte]
  exact firstWait_one_acc H _ gI.opens (gI.assemble x) (gI.check x) ρ () (gI_run_out H P x ω _) (fun _ => rfl) rfl

theorem gInnerKS (H : Msg → Msg) : Inner.KS H gI := by
  intro x Ω _ _ P
  calc PrBad H gI.V gI.R gI.E x P ≤ Pr (fun _ : Ω × (Fin (gI.V.k x) → Bool) × gI.E.C x => False) :=
        Pr_mono fun s h => h.2 (by rw [gI_E_some]; trivial)
    _ = 0 := Pr_false
    _ ≤ _ := by simp [gI]

theorem firstWait_const_some {R W Op : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (w : W)
    (ha : ∀ vs : List Seen, assemble (vs.flatMap opens) = some w) (hc : check w = true) (rs : List R)
    (rss : List (List R)) :
    ((firstWait k opens assemble check (rs :: rss)).eval H run).1 = some w := by
  simp only [firstWait]
  rw [(RP.eval_bind H run _ _).1, ha]
  simp [hc, RP.eval]

theorem gRelS (w : Flock.Wit) : Flock.RelS gs₀ w := fun _ _ => rfl

theorem gDO_E_some (H : Msg → Msg) (decode : List (ℕ × ℕ) → List Unit)
    (hdec : Flock.ListWithin gDO.len gDO.enc gDO.θ decode) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (gDO.V.k x) → Bool) (ρ : (Flock.E gDO decode).C x) (hacc : (run H P gDO.V x ω r₀).out = .acc) :
    ((Flock.E gDO decode).eval H P x ω r₀ ρ).1 = some ⟨[]⟩ := by
  simp only [Flock.E, ERep, Extractor.eval, RP.eval, Trace.seen, hacc, ↓reduceIte]
  obtain ⟨rs, rss, hl⟩ := List.exists_cons_of_ne_nil (l := (List.finRange (gDO.waits x)).map fun i =>
    (List.finRange (gDO.cap x)).map (ρ i)) (by simp [gDO])
  rw [hl]
  have hm : () ∈ decode [] := hdec [] () (by simp [gDO])
  obtain ⟨a, l, hdl⟩ := List.exists_cons_of_ne_nil (List.ne_nil_of_mem hm)
  refine firstWait_const_some H _ _ _ _ _ (⟨[]⟩ : Flock.Wit) (fun vs => ?_) ?_ rs rss
  · have hf : ((vs.flatMap fun v => [gDO.cols v]).flatMap fun o => o 0) = [] :=
      List.flatMap_eq_nil_iff.2 fun o ho => by
        obtain ⟨v, -, hv⟩ := List.mem_flatMap.1 ho
        rw [List.mem_singleton.1 hv]
        rfl
    have h1 : gs₀.tables = 1 := rfl
    simp only [Flock.assemble, show gDO.parse x = some gs₀ from rfl, h1, List.range_one, List.mapM_cons,
      List.mapM_nil, show gDO.table x 0 = some gs₀ from rfl, Option.bind_some, hf, hdl]
    simp [gDO, gRelS]
  · simp [Flock.check, gDO, gRelS]

theorem gFlockKS (H : Msg → Msg) : Flock.KS H gDO := by
  intro decode hdec x Ω _ _ P
  have h0 : PrBad H gDO.V (Flock.R gDO.parse) (Flock.E gDO decode) x P ≤ 0 :=
    (Pr_mono fun s h => h.2 (by
      rw [gDO_E_some H decode hdec P x s.1 s.2.1 s.2.2 h.1]
      exact ⟨gs₀, rfl, gRelS _⟩)).trans_eq Pr_false
  have hκ : 0 ≤ Flock.κ gDO x := by
    unfold Flock.κ
    simp only [gDO]
    positivity
  exact h0.trans (add_nonneg hκ (Real.sqrt_nonneg _))

/-! ## The recursion's KS fails -/

/-- `E^rep` at one wait of one rerun gives up when that rerun is rejected and nothing assembles from no openings. -/
theorem erep_one_none {Dr W Op : Type} (H : Msg → Msg) (run : Dr → Trace) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (ha : assemble [] = none) (d₀ : Dr) (ρ : Fin 1 → Fin 1 → Dr)
    (h : (run (ρ 0 0)).out ≠ .acc) : ((erep 1 1 1 opens assemble check d₀ ρ).eval H run).1 = none := by
  simp only [erep, RP.eval]
  split_ifs
  · simp [firstWait, waitAcc, RP.bind, RP.eval, Trace.seen, h, ha, List.finRange_succ]
  · rfl

/-- The prover: `H []`, every round. -/
def gP (H : Msg → Msg) : Prover Unit := ⟨fun _ _ _ => .ret (H [])⟩

/-- An outer decoder within the radius: every trace, at every opening. -/
def gDecO : List (ℕ × ℕ) → List Unit := fun _ => [()]

theorem gDecOK : decOK gDO gDecO := fun _ M _ => by cases M; simp [gDecO]

/-- `E_O` gives up at a rejected run. -/
theorem EO_ext_none {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) (h : (run H P D.V x ω d.1).out ≠ .acc) :
    (((EO D I DO decO).ext x d.1 d.2).eval H (run H P D.V x ω)).1 = none := by
  simp [EO, RP.eval, Trace.seen, h]

/-- A draw where the outer stage gives up is served as a rejected run. -/
theorem servedRun_out_none {Coin W₂ : Type} {V : Verifier Coin} (H : Msg → Msg) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (x : Msg) (run : (Fin (V.k x) → Coin) → Trace)
    (d : (Fin (V.k x) → Coin) × E₂.C x) (h : ((E₂.ext x d.1 d.2).eval H run).1 = none) :
    (servedRun H (outerStage E₂ dec x) run d).out = (dec x d.1 none).out := by
  simp only [servedRun, outerStage]
  rw [RP.eval_bind_ret, h]

/-- The one draw of one wait of one rerun. -/
def at00 {X : Type} (ρ : Fin 1 → Fin 1 → X) : X := ρ 0 0

/-- `E*`'s one inner rerun's draw: the compiled verifier's coins and `E_O`'s. -/
def gDraw (H : Msg → Msg) (decO : List (ℕ × ℕ) → List Unit) (x : Msg) (c : (E H (gD H) gI gDO decO).C x) :
    (Fin ((gD H).V.k x) → Bool) × (EO (gD H) gI gDO decO).C x :=
  at00 c.2

/-- **`E*` gives up when its inner rerun's outer coin rejects**: `E_O` gives up at a rejected run, so the inner stage
sees its one rerun rejected. -/
theorem gE_none (H : Msg → Msg) (decO : List (ℕ × ℕ) → List Unit) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin ((gD H).V.k x) → Bool) (c : (E H (gD H) gI gDO decO).C x) (h : coinSeq (gDraw H decO x c).1 1 = true) :
    ((E H (gD H) gI gDO decO).eval H P x ω r₀ c).1 = none := by
  have hrej : (run H P (gD H).V x ω (gDraw H decO x c).1).out ≠ .acc := fun h' => by
    have := ((gD_out H P x ω _).1 h').2
    rw [h] at this
    exact Bool.noConfusion this
  have hserved := servedRun_out_none H (EO (gD H) gI gDO decO) (dec H (gD H)) x (run H P (gD H).V x ω)
    (gDraw H decO x c) (EO_ext_none H (gD H) gI gDO decO P x ω _ hrej)
  simp only [Extractor.eval, E, served]
  rw [serve_eq, (RP.serve_eval H _ _ _).1]
  refine erep_one_none H _ gI.opens (gI.assemble _) (gI.check _) rfl (r₀, c.1) c.2 ?_
  change (servedRun H (outerStage (EO (gD H) gI gDO decO) (dec H (gD H)) x) (run H P (gD H).V x ω)
    (gDraw H decO x c)).out ≠ .acc
  rw [hserved]
  simp [dec]

theorem gPr_quarter : Pr (fun b : Bool × Bool => b.1 = false ∧ b.2 = true) = 1 / 4 := by
  rw [Pr_eq_sum]
  simp
  have : ({x | x.1 = false ∧ x.2 = true} : Finset (Bool × Bool)).card = 1 := by decide
  rw [this]
  norm_num

/-- **`E*` fails on a quarter of the draws**: the judged run accepts at its coin `false`, and the inner rerun's outer
coin `true` rejects it. -/
theorem gPrBad_ge (H : Msg → Msg) (decO : List (ℕ × ℕ) → List Unit) :
    1 / 4 ≤ PrBad H (gD H).V (R (gD H) gI) (E H (gD H) gI gDO decO) [] (gP H) := by
  have h1 : (1 : ℕ) < (gD H).V.k [] := by show 1 < 2; omega
  have hc : ∀ r : Fin ((gD H).V.k []) → Bool, coinSeq r 1 = r ⟨1, h1⟩ := fun r => dite_eq_left h1
  let φ : Unit × (Fin ((gD H).V.k []) → Bool) × (E H (gD H) gI gDO decO).C [] → Bool × Bool :=
    fun s => (coinSeq s.2.1 1, coinSeq (gDraw H decO [] s.2.2).1 1)
  have hcoin : UnifMap (fun r : Fin ((gD H).V.k []) → Bool => coinSeq r 1) := by
    simp only [hc]
    exact UnifMap.coord _
  have hd : UnifMap (fun c : (E H (gD H) gI gDO decO).C [] => (gDraw H decO [] c).1) :=
    ((UnifMap.snd.comp (UnifMap.coord (0 : Fin 1))).comp (UnifMap.coord (0 : Fin 1))).comp UnifMap.fst
  have hp := hcoin.prod (hd.comp hcoin)
  have hφ : UnifMap φ := fun G => by
    rw [Ex_prod]
    exact (congrArg Ex (funext fun _ => hp G)).trans (Ex_const _)
  calc (1 : ℝ) / 4 = Pr (fun s => (φ s).1 = false ∧ (φ s).2 = true) := by
        rw [hφ.Pr (fun b => b.1 = false ∧ b.2 = true), gPr_quarter]
    _ ≤ _ := Pr_mono fun s hs =>
        ⟨(gD_out H (gP H) [] s.1 s.2.1).2 ⟨rfl, hs.1⟩, fun hR => by rw [gE_none H decO _ _ _ _ _ hs.2] at hR; exact hR⟩

theorem gPrB (H : Msg → Msg) : PrB H (gD H).V (gP H) [] = 0 := by
  have hk : 0 < (gD H).V.k [] := by show 0 < 2; omega
  unfold PrB
  rw [dite_eq_left hk]
  refine le_antisymm ((Pr_mono fun s h => ?_).trans_eq Pr_false) (Pr_nonneg _)
  obtain ⟨u, v, huv, -⟩ := h
  simp only [B] at huv
  rw [findColl_fst_const [] _ fun e he => (List.mem_append.1 he).elim (gD_vLog H _ _ _ _ e)
    (gD_vLog H _ _ _ _ e)] at huv
  simp at huv

theorem gPrBe (H : Msg → Msg) (decO : List (ℕ × ℕ) → List Unit) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) : PrBe H (gD H) gI gDO decO P x = 0 := by
  unfold PrBe
  split
  · refine le_antisymm ((Pr_mono fun s h => ?_).trans_eq Pr_false) (Pr_nonneg _)
    obtain ⟨u, v, huv, -⟩ := h
    simp only [Be] at huv
    rw [findColl_fst_const [] _ fun e he => by simp [gD] at he; simp [he]] at huv
    simp at huv
  · rfl

theorem gPrBr (H : Msg → Msg) (decO : List (ℕ × ℕ) → List Unit) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) : PrBr H (gD H) gI gDO decO P x = 0 := by
  unfold PrBr
  split
  · refine le_antisymm ((Pr_mono fun s h => ?_).trans_eq Pr_false) (Pr_nonneg _)
    obtain ⟨u, v, huv, -⟩ := h
    simp only [Br, setupCalls_eq_nil H (gD H) (fun _ => rfl), List.append_nil, findCollIn, reduceCtorEq] at huv
  · rfl

theorem gNums_k : gNums.k = 400 := by
  unfold Nums.k
  have : 4 * gNums.N₀ / Real.exp 1 = ((400 : ℕ) : ℝ) := by
    show 4 * (100 * Real.exp 1) / Real.exp 1 = _
    field_simp
    norm_num
  rw [this, Nat.ceil_natCast]

/-- `κ_O = 2^{−8} + 2^{−8} + e^{−50} < 1/8`. -/
theorem gNums_κ_lt : gNums.κ < 1 / 8 := by
  unfold Nums.κ
  rw [gNums_k]
  have h2 : (2 : ℝ) ^ (-(gNums.waits : ℝ)) = 1 / 256 := by
    show (2 : ℝ) ^ (-((8 : ℕ) : ℝ)) = 1 / 256
    rw [Real.rpow_neg (by norm_num), Real.rpow_natCast]
    norm_num
  have he : Real.exp (-((400 : ℕ) : ℝ) / 8) ≤ 1 / 51 := by
    rw [show -((400 : ℕ) : ℝ) / 8 = -50 by norm_num, Real.exp_neg, inv_eq_one_div]
    exact one_div_le_one_div_of_le (by norm_num) (by linarith [Real.add_one_le_exp (50 : ℝ)])
  have hε : gNums.εtable = 0 := rfl
  have hK : (gNums.cap : ℝ) = 204800 := by norm_num [gNums]
  rw [h2, hε, hK]
  norm_num
  linarith

theorem gκ (H : Msg → Msg) (x : Msg) : κ (gD H) gI x = 2 * gNums.κ := by
  unfold κ NI
  show gI.κ (xI (gD H) x) + ((1 + 1 * 1 : ℕ) : ℝ) * ∑ _s : Fin 1, gNums.κ = 2 * gNums.κ
  rw [Fin.sum_univ_one]
  simp only [gI]
  norm_num

/-- **The recursion's KS fails** at the prover that sends `H []` every round. -/
theorem gNotKS (H : Msg → Msg) : ¬ KS H (gD H) gI gDO := by
  intro hK
  have h := hK gDecO gDecOK [] Unit (gP H)
  rw [gPrB, gPrBe, gPrBr, mul_zero, mul_zero, Real.sqrt_zero, add_zero, add_zero, add_zero, gκ] at h
  linarith [gPrBad_ge H gDecO, gNums_κ_lt]

/-- **The recursion's KS fails at the errors its proof gives**, as `B`, `Be` and `Br` find nothing there either. -/
theorem gNotKSA (H : Msg → Msg) : ¬ KSA H (gD H) gI gDO := by
  intro hK
  have h := hK gDecO gDecOK [] Unit (gP H)
  rw [gPrB, gPrBe, gPrBr, mul_zero, mul_zero, Real.sqrt_zero, add_zero, add_zero, add_zero, gκ] at h
  linarith [gPrBad_ge H gDecO, gNums_κ_lt]

/-! ## `Inner.KSAbort` fails -/

/-- A prover's aborts: each run is kept at its fresh coin `false`. -/
def gKeep : Unit → (Fin (gI.V.k []) → Bool) → Bool → Bool := fun _ _ a => !a

/-- **`gI` fails `Inner.KSAbort`**: at the prover that keeps each run at its own fresh coin `false`, the judged run is
kept and accepted, and `E^rep`'s one rerun aborted, on a quarter of the draws, while `κ_I = γ_I = 0`. -/
theorem gI_not_KSAbort (H : Msg → Msg) : ¬ gI.KSAbort H := by
  intro h
  have hx := h [] Unit (gP H) Bool gKeep
  have hr : gI.κ [] + Real.sqrt (gI.γ [] * PrBAbort H gI.V (gP H) [] gKeep) = 0 := by simp [gI]
  rw [hr] at hx
  let φ : Unit × ((Fin (gI.V.k []) → Bool) × Bool) ×
      (Fin (gI.waits []) → Fin (gI.cap []) → (Fin (gI.V.k []) → Bool) × Bool) → Bool × Bool :=
    fun s => (s.2.1.2, (at00 s.2.2).2)
  have hφ : UnifMap φ := UnifMap.snd.comp (UnifMap.prod UnifMap.snd
    (((UnifMap.coord (0 : Fin 1)).comp (UnifMap.coord (0 : Fin 1))).comp UnifMap.snd))
  have hq : (1 : ℝ) / 4 ≤ gI.PrBadAbort H [] (gP H) gKeep := by
    calc (1 : ℝ) / 4 = Pr (fun s => (φ s).1 = false ∧ (φ s).2 = true) := by
          rw [hφ.Pr (fun b => b.1 = false ∧ b.2 = true), gPr_quarter]
      _ ≤ _ := Pr_mono fun s hs => by
          obtain ⟨h1, h2⟩ := hs
          change s.2.1.2 = false at h1
          change (at00 s.2.2).2 = true at h2
          refine ⟨?_, ?_⟩
          · show (abortRun H (gP H) gI.V [] gKeep s.1 s.2.1).out = .acc
            simp [abortRun, gKeep, h1, gI_run_out]
          · have hrej : (abortRun H (gP H) gI.V [] gKeep s.1 (at00 s.2.2)).out ≠ .acc := by
              simp [abortRun, gKeep, h2, abortTrace]
            have hnone : ((erep (gI.waits []) (gI.k []) (gI.cap []) gI.opens (gI.assemble []) (gI.check [])
                s.2.1 s.2.2).eval H (abortRun H (gP H) gI.V [] gKeep s.1)).1 = none :=
              erep_one_none H _ gI.opens (gI.assemble []) (gI.check []) rfl s.2.1 s.2.2 hrej
            show ¬ gI.R.ext [] ((erep (gI.waits []) (gI.k []) (gI.cap []) gI.opens (gI.assemble []) (gI.check [])
                s.2.1 s.2.2).eval H (abortRun H (gP H) gI.V [] gKeep s.1)).1
            rw [hnone]
            exact id
  linarith

/-! ## The gap -/

/-- **`InnerKS` does not follow from `Inner.KS` and the code facts, and `Inner.KSAbort` excludes the deployment.** At
every hash, a compiled verifier with every code fact `KS_of_flockKS` reads, over an inner protocol with `Inner.KS` and
an outer deployment with `Claims.Flock.KS`, whose recursion's KS fails at the errors its proof gives (`KSA`) and at the
draft's (`KS`): so `InnerKS` fails there, as `KS_of` with `outerKS_of_flockKS` would otherwise prove `KSA`. Its inner
protocol fails `Inner.KSAbort`, the one hypothesis of `KS_of_flockKS` it lacks. -/
theorem innerKS_gap (H : Msg → Msg) :
    ∃ (D : Deploy Bool) (I : Inner Bool Unit) (DO : Flock.Deploy Bool ℕ Unit) (opening : Opening)
      (decC : Msg → Msg → Bool),
      Inner.KS H I ∧ ¬ I.KSAbort H ∧ Flock.KS H DO ∧ Refuses H D ∧ InnerRounds D I ∧ SessionNums D DO ∧
        TableErrNonneg D ∧ SessionRounds D I DO ∧ SessionAccepts D ∧ SessionRuns H D I DO ∧
        InnerViewRun H D I DO ∧ CallsRecord H D I DO ∧ CallsOpen H D I DO opening ∧ InnerAccepts H D I DO ∧
        PublicCoin H I decC ∧ CallsNil H D ∧ ¬ KSA H D I DO ∧ ¬ KS H D I DO ∧ ¬ InnerKS H D I DO :=
  ⟨gD H, gI, gDO, Opening.ofCall id, fun _ _ => false, gInnerKS H, gI_not_KSAbort H, gFlockKS H, gRefuses H,
    gInnerRounds H, gSessionNums H, gTableErrNonneg H, gSessionRounds H, gSessionAccepts H, gSessionRuns H, gInnerViewRun H,
    gCallsRecord H, gCallsOpen H, gInnerAccepts H, gPublicCoin H, gCallsNil H, gNotKSA H, gNotKS H,
    fun hI => gNotKSA H (KS_of H (gD H) gI gDO (gRefuses H) (gInnerRounds H)
      (outerKS_of_flockKS (gFlockKS H) (gSessionNums H) (gTableErrNonneg H) (gSessionRounds H) (gSessionAccepts H)
        (gSessionRuns H)) hI)⟩

end Claims.Recursion
