import Definitions.NetworkCertifier.TapRows
import Proofs.NetworkCertifier.SampledLeak
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.Claims.Finder

/-!
# The leak bound at the network certifier's rows

The network certifier registers its records at the transfer gates (`TapOpens`). On the sampling relation (`R δ`), every
unit outside the faulty set is correct on the trace's committed values, so the trace opens every transfer gate there
(each is committed: an output or an input). Where the trace opens a transfer gate it agrees with the network certifier,
unless the two openings collide (below). Fill in the network certifier's value at every gate the trace leaves unopened
(`fill`): a correct unit stays correct (`correct_mono`, it reads only opened values), and the filled trace carries the
network certifier's records at every transfer gate (`carries_fill`), the faulty units' included, since the trace's own
value there, where it has one, is the network certifier's. So `accepted_of` gives `Accepted` with the faulty units'
committed gates held at the filled values, the advice the count charges (`2^β` over `faultySets`), and since a held
gate's input value is never read, the inputs move back to the trace's own values (`valOver_congr_held`): the network
certifier's records are among the sequences `AccSeqsSampled` counts at the trace's values (`rLeakTap_of_r`).

An extracted trace that disagrees with the network certifier opens a transfer gate to a value the network certifier's
opening there doesn't. The extractor checked that opening with its own calls (`Passes.found`, `ownLog_E`), so under
`Commitment.Binds` its calls and those that check the network certifier's opening collide, and `B_tap` finds the
collision (`tapColl_of_disagree`, `Pr_tapColl_le_Btap`). `sampledLeak_tap` is the bound:
`Pr[acc ∧ ¬RLeakTap] ≤ δ + ε_ks + ε_bind + Pr[B_tap]`.

The registered-input and collision steps follow the registration's (proofs' P1b), at the network certifier's openings.
-/

namespace NetTiming.TapProof

open Claims Claims.Sampling NetTiming NetTiming.SampledAuditProof

/-! ### A correct unit stays correct when unopened values are filled in -/

theorem mapM_mono {rd rd' : ℕ → Option Bool} :
    ∀ {ws : List ℕ}, (∀ i ∈ ws, ∀ b, rd i = some b → rd' i = some b) → ∀ {bs : List Bool},
      ws.mapM rd = some bs → ws.mapM rd' = some bs
  | [], _, bs, h => h
  | i :: ws, hrd, bs, h => by
    rcases hri : rd i with _ | c
    · simp [List.mapM_cons, hri] at h
    · rcases hws : ws.mapM rd with _ | cs
      · simp [List.mapM_cons, hri, hws] at h
      · have h1 := hrd i (by simp) c hri
        have h2 := mapM_mono (ws := ws) (fun j hj => hrd j (by simp [hj])) hws
        simp only [List.mapM_cons, hri, hws, Option.bind_eq_bind, Option.bind_some, Option.pure_def,
          Option.some.injEq] at h
        simp [List.mapM_cons, h1, h2, ← h]

theorem val_mono {rd rd' : ℕ → Option Bool} {own own' : Option Bool} (x : BGate)
    (hrd : ∀ i ∈ x.reads, ∀ b, rd i = some b → rd' i = some b) (hown : ∀ b, own = some b → own' = some b) {b : Bool}
    (h : x.val rd own = some b) : x.val rd' own' = some b := by
  cases x with
  | input => exact hown b h
  | const c => exact h
  | not i =>
    simp only [BGate.val] at h ⊢
    obtain ⟨c, hc, rfl⟩ := Option.map_eq_some_iff.1 h
    rw [hrd i (by simp [BGate.reads]) c hc]
    rfl
  | and ws =>
    simp only [BGate.val] at h ⊢
    obtain ⟨bs, hbs, rfl⟩ := Option.map_eq_some_iff.1 h
    rw [mapM_mono (fun i hi => hrd i (by simpa [BGate.reads] using hi)) hbs]
    rfl
  | xor ws =>
    simp only [BGate.val] at h ⊢
    obtain ⟨bs, hbs, rfl⟩ := Option.map_eq_some_iff.1 h
    rw [mapM_mono (fun i hi => hrd i (by simpa [BGate.reads] using hi)) hbs]
    rfl

theorem read_mono {C : Prog} {ref ref' : ℕ → Option Bool} (hext : ∀ g b, ref g = some b → ref' g = some b) {i : ℕ}
    {b : Bool} (h : C.read ref i = some b) : C.read ref' i = some b := by
  by_cases hc : ∃ c, C.gates[i]? = some (.const c)
  · obtain ⟨c, hc⟩ := hc
    rw [read_const hc] at h ⊢
    exact h
  · have hc' : ∀ c, C.gates[i]? ≠ some (.const c) := fun c h' => hc ⟨c, h'⟩
    rw [read_of_not_const hc'] at h ⊢
    exact hext i b h

theorem runIn_mono {C : Prog} {ref ref' : ℕ → Option Bool} (hext : ∀ g b, ref g = some b → ref' g = some b)
    (u : List ℕ) : ∀ g b, C.runIn ref u g = some b → C.runIn ref' u g = some b := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
  intro b h
  rw [Prog.runIn] at h ⊢
  rcases hx : C.gates[g]? with _ | x
  · rw [hx] at h
    cases h
  · rw [hx] at h
    simp only [Option.bind_some] at h ⊢
    refine val_mono x (fun i _ c hc => ?_) (hext g) h
    by_cases hiu : i ∈ u
    · simp only [hiu, ite_true] at hc ⊢
      by_cases hig : i < g
      · rw [dite_eq_left hig] at hc ⊢
        exact ih i hig c hc
      · rw [dite_eq_right hig] at hc
        cases hc
    · simp only [hiu, ite_false] at hc ⊢
      exact read_mono hext hc

theorem ruRef_mono {C : Prog} {r : RU} {ref ref' : ℕ → Option Bool} {ι : ℕ → Bool}
    (hext : ∀ g b, ref g = some b → ref' g = some b) :
    ∀ g b, RU.ref C r ref ι g = some b → RU.ref C r ref' ι g = some b := by
  intro g b h
  unfold RU.ref at h ⊢
  split_ifs at h ⊢
  · exact h
  · exact hext g b h

theorem correct_mono {C : Prog} {ref ref' : ℕ → Option Bool} (hext : ∀ g b, ref g = some b → ref' g = some b)
    {u : List ℕ} (hc : C.Correct ref u) : C.Correct ref' u := by
  intro g hg
  obtain ⟨h1, h2⟩ := hc g hg
  refine ⟨fun x hx i hi hiu => ?_, fun hbd => ?_⟩
  · obtain ⟨c, hc⟩ := Option.ne_none_iff_exists'.1 (h1 x hx i hi hiu)
    rw [read_mono hext hc]
    exact Option.some_ne_none _
  · obtain ⟨b, hb1, hb2⟩ := h2 hbd
    exact ⟨b, hext g b hb1, runIn_mono hext u g b hb2⟩

/-! ### The trace filled in with the network certifier's values -/

/-- The committed values `ref`, with `tv`'s value at every gate `ref` leaves unopened. -/
def fill (ref : ℕ → Option Bool) (tv : ℕ → Bool) (g : ℕ) : Option Bool := some ((ref g).getD (tv g))

theorem fill_ext {ref : ℕ → Option Bool} {tv : ℕ → Bool} : ∀ g b, ref g = some b → fill ref tv g = some b :=
  fun g b h => by simp [fill, h]

theorem traceVal_fill {C : Prog} {ref : ℕ → Option Bool} {tv : ℕ → Bool} {g : ℕ}
    (hnc : ∀ b, C.gates[g]? ≠ some (.const b)) (hag : ∀ b, ref g = some b → b = tv g) :
    traceVal C (fill ref tv) g = tv g := by
  rw [traceVal, read_of_not_const hnc]
  rcases hr : ref g with _ | b
  · simp [fill, hr]
  · simp [fill, hr, hag b hr]

theorem traceVal_fill_of {C : Prog} {ref : ℕ → Option Bool} {tv : ℕ → Bool} {g : ℕ}
    (hc : ∀ b, C.gates[g]? = some (.const b) → tv g = b) (hag : ∀ b, ref g = some b → b = tv g) :
    traceVal C (fill ref tv) g = tv g := by
  by_cases hk : ∃ b, C.gates[g]? = some (.const b)
  · obtain ⟨b, hb⟩ := hk
    rw [traceVal, read_const hb, hc b hb]
    rfl
  · exact traceVal_fill (fun b h => hk ⟨b, h⟩) hag

theorem rowOf_congr {S : ℕ} {val val' : ℕ → Bool} {p : Fin S → ℕ × List ℕ} {σ : Fin S}
    (h : ∀ g ∈ slotGates (p σ), val g = val' g) : rowOf val p σ = rowOf val' p σ := by
  simp only [rowOf]
  rw [h (p σ).1 (by simp [slotGates]), List.map_congr_left fun g hg => h g (by simp [slotGates, hg])]

/-- **The filled trace carries the network certifier's records**: where the trace opens a transfer gate the network
certifier opens, at the network certifier's value, the trace filled in with the network certifier's values carries every
record the network certifier registers. -/
theorem carries_fill {L K S : ℕ} {H : Msg → Msg} {Cm : Commitment Bool} {lay : BoundaryGates L S} {k : ℕ}
    {op : ℕ → List (Obs L K S) → List Bool} {s : AuditStatement} {ops : List (Opening Bool)} {o : List (Obs L K S)}
    {ref : ℕ → Option Bool} (hfit : lay.Fits s.prog k) (htap : TapOpens H Cm lay k op s ops o)
    (hag : ∀ g, (∃ q ∈ ops, q.pos = g) → ∀ b, ref g = some b → b = tapVal ops g) :
    Carries lay s.prog (fill ref (tapVal ops)) op k o := by
  intro t ht
  obtain ⟨hE, hI, ho, hrows, hop⟩ := htap.2 t ht
  refine ⟨ho, fun ℓ => ?_, ?_⟩
  · rw [hrows ℓ]
    funext σ
    exact rowOf_congr fun g hg => (traceVal_fill_of (hE ℓ σ g hg).1 (hag g (hE ℓ σ g hg).2)).symm
  · rw [hop]
    refine List.map_congr_left fun g hg => (traceVal_fill (fun b h => ?_) (hag g (hI g hg))).symm
    rw [hfit.2.1 t ht g hg] at h
    cases h

/-! ### A registered input of a correct unit is opened, and its input value is the trace's -/

theorem val_some_of_input {H : Msg → Msg} {Cm : Commitment Bool} {s : AuditStatement} {P : Partition} {w : Wit}
    (hP : P.Partitions s.prog) {g : ℕ} (hg : s.prog.gates[g]? = some .input)
    (hnf : ¬ FaultyGate s.prog P (faulty H Cm s P w) g) : ∃ b, w.val H Cm s g = some b := by
  obtain ⟨ι, hι⟩ := correct_of_not_faulty H Cm s P w
  have hlt : g < s.prog.gates.length := (List.getElem?_eq_some_iff.1 hg).1
  have hmem : g ∈ P.gates := hP.2.2.1 g hlt fun b h => by rw [hg] at h; cases h
  obtain ⟨v, r, u, hr, hu, hgu⟩ := exists_of_mem_gates hmem
  have hv : v ∉ faulty H Cm s P w := fun hv => hnf ⟨v, hv, r, hr, mem_ru_gates hu hgu, boundary_of_input hg⟩
  obtain ⟨b, hb, -⟩ := (hι v r hr hv u hu g hgu).2 (boundary_of_input hg)
  have hc : ¬ (g ∈ RU.gates r ∧ s.prog.Boundary (RU.gates r) g = false) := fun h => by
    rw [boundary_of_input hg] at h
    exact Bool.noConfusion h.2
  unfold RU.ref at hb
  rw [ite_eq_right hc] at hb
  exact ⟨b, hb⟩

theorem traceVal_of_val {C : Prog} {ref : ℕ → Option Bool} {g : ℕ} {b : Bool} (hg : C.gates[g]? = some .input)
    (h : ref g = some b) : traceVal C ref g = b := by
  simp [traceVal, Prog.read, hg, h]

theorem inputOf_congr_at {L S : ℕ} (lay : BoundaryGates L S) (k : ℕ) (ins : List (List Bool)) {a a' : ℕ → Bool} {g : ℕ}
    (h : a g = a' g) : lay.inputOf k ins a g = lay.inputOf k ins a' g := by
  unfold BoundaryGates.inputOf
  split
  · rfl
  · rw [h]

theorem valOver_congr_held (C : Prog) {inp inp' : ℕ → Bool} (free : ℕ → Option Bool)
    (h : ∀ i, C.gates[i]? = some .input → free i = none → inp i = inp' i) (g : ℕ) :
    valOver C inp free g = valOver C inp' free g := by
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    rcases hf : free g with _ | b
    · rcases hx : C.gates[g]? with _ | x
      · rw [valOver_of_none_none C inp hf hx, valOver_of_none_none C inp' hf hx]
      · rw [valOver_of_none_some C inp hf hx, valOver_of_none_some C inp' hf hx]
        have hrd : ∀ i ∈ x.reads, (if _h : i < g then some (valOver C inp free i) else some false) =
            if _h : i < g then some (valOver C inp' free i) else some false := by
          intro i _
          split_ifs with hig
          · rw [ih i hig]
          · rfl
        rw [bgate_val_congr x _ hrd]
        cases x <;> first | rfl | rw [h g hx hf]
    · rw [valOver_of_some C inp hf, valOver_of_some C inp' hf]

/-! ### The sampling relation gives the leak relation at the network certifier's rows -/

section Deploy

variable {Coin Col T : Type}

/-- **At the network certifier's rows**: on the audit's scope at `δ ≥ 0`, the sampling relation at a trace that agrees
with the network certifier wherever both open a gate gives the leak relation at the network certifier's rows, at any
inputs `As` that suffice (`AdviceIn`). -/
theorem rLeakTap_of_r (H : Msg → Msg) (D : Deploy Coin Col T) {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    {δ : ℝ} (hδ : 0 ≤ δ) {As : AuditStatement → (ℕ → Bool) → Set (ℕ → Bool)} {Dj : ℕ} (tops : AuditStatement → List (Opening Bool))
    (hin : ∀ s P, D.partition s = some P → ∀ ρ, AdviceIn K s.prog P s.design (lay s) (k s) ρ (As s ρ) δ) {x : Msg}
    (hx : ScopeAcc D lay k x) {w : Wit} (hR : R H D.commit D.parse D.partition δ x w)
    (hagree : ∀ s, D.parse x = some s → (∀ q ∈ tops s, D.commit.Opens H s.roots q = true) →
      ∀ g, (∃ q ∈ tops s, q.pos = g) → ∀ b, w.val H D.commit s g = some b → b = tapVal (tops s) g) :
    RLeakTap H D.commit D.parse D.partition L K S lay k δ As Dj tops x w := by
  obtain ⟨s, P, hs, hP, hM⟩ := hR
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hsc⟩, s₂, hs₂, hfit⟩ := hx
  rw [hs] at hs₁ hs₂
  obtain rfl := Option.some.inj hs₁
  obtain rfl := Option.some.inj hs₂
  rw [hP] at hP₁
  obtain rfl := Option.some.inj hP₁
  have hF := faulty_mem_faultySets hM (disjoint_allSlots hδ hM)
  obtain ⟨ι, hι⟩ := correct_of_not_faulty H D.commit s P w
  refine ⟨s, P, hs, hP, fun op o htap hcov hl hD => ?_⟩
  have hext : ∀ g b, w.val H D.commit s g = some b → fill (w.val H D.commit s) (tapVal (tops s)) g = some b :=
    fill_ext
  have hcar := carries_fill hfit htap (hagree s hs htap.1)
  have hacc := accepted_of hsc.1 hsc.2.1 hfit ι
    (fun v r hr hv u hu => correct_mono (ruRef_mono hext) (hι v r hr hv u hu)) hcar hcov
  have hacc' : Accepted (circuitProgram s.prog (lay s) (k s)) op (k s) o
      (traceVal s.prog (w.val H D.commit s),
        heldAdv s.prog P (faulty H D.commit s P w) (fill (w.val H D.commit s) (tapVal (tops s)))) := by
    intro t ht
    obtain ⟨h, hr, hst⟩ := hacc t ht
    refine ⟨h, ?_, hst⟩
    have hv : valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t)
          (traceVal s.prog (fill (w.val H D.commit s) (tapVal (tops s)))))
          (heldAdv s.prog P (faulty H D.commit s P w) (fill (w.val H D.commit s) (tapVal (tops s)))) =
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) (traceVal s.prog (w.val H D.commit s)))
          (heldAdv s.prog P (faulty H D.commit s P w) (fill (w.val H D.commit s) (tapVal (tops s)))) :=
      funext (valOver_congr_held s.prog _ fun i hi hfi => inputOf_congr_at (lay s) (k s) _ (by
        have hnf : ¬ FaultyGate s.prog P (faulty H D.commit s P w) i := fun hfg => by
          rw [heldAdv_of hfg] at hfi
          cases hfi
        obtain ⟨b, hb⟩ := val_some_of_input hsc.2.1 hi hnf
        rw [traceVal_of_val hi hb, traceVal_of_val hi (hext i b hb)]))
    rw [hr]
    simp only [circuitProgram, hv]
  exact ⟨hl, hD, _, hF, hin s P hP _ _ hF op o
    ⟨_, Set.mem_singleton _, _, fun g b h => faultyGate_of_heldAdv h, hacc'⟩⟩

/-! ### A disagreement with the network certifier is a collision -/

theorem mem_assemble_of_val {H : Msg → Msg} {Cm : Commitment Bool} {s : AuditStatement} {found : List (List Msg × Opening Bool)}
    {g : ℕ} {b : Bool} (h : (assemble s found).val H Cm s g = some b) :
    ∃ q ∈ found, q.1 = s.roots ∧ q.2.pos = g ∧ q.2.val = b := by
  unfold Wit.val at h
  obtain ⟨o, ho, rfl⟩ := Option.map_eq_some_iff.1 h
  have hpo := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hpo
  obtain ⟨q, hq, hqo⟩ := List.mem_filterMap.1 (List.mem_of_find?_eq_some ho)
  by_cases hr : q.1 = s.roots
  · simp only [hr, ite_true, Option.some.injEq] at hqo
    subst hqo
    exact ⟨q, hq, hr, hpo.1, rfl⟩
  · simp [hr] at hqo

theorem tapVal_mem {ops : List (Opening Bool)} {g : ℕ} (h : ∃ o ∈ ops, o.pos = g) :
    ∃ o ∈ ops, o.pos = g ∧ o.val = tapVal ops g := by
  obtain ⟨o, ho⟩ : ∃ o, ops.find? (fun o => o.pos == g) = some o :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 (by obtain ⟨o, hm, hp⟩ := h; exact ⟨o, hm, by simp [hp]⟩))
  refine ⟨o, List.mem_of_find?_eq_some ho, by simpa using List.find?_some ho, ?_⟩
  simp [tapVal, ho]

theorem tapLog_valid (H : Msg → Msg) (Cm : Commitment Bool) (roots : List Msg) (ops : List (Opening Bool)) :
    ∀ p ∈ tapLog H Cm roots ops, H p.1 = p.2 := by
  intro p hp
  obtain ⟨o, -, hpo⟩ := List.mem_flatMap.1 hp
  exact OC.eval_log H _ p hpo

variable [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- A collision against the network certifier's openings: the extraction's own calls and the checks of the network
certifier's openings hold two different inputs with one output. -/
def TapColl (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (tops : AuditStatement → List (Opening Bool)) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x) : Prop :=
  LogColl (ownLog H (run H P D.V x s.1) ((E D decode).ext x s.2.1 s.2.2) ++
    ((D.parse x).map fun st => tapLog H D.commit st.roots (tops st)).getD [])

omit [DecidableEq Col] [Zero Col] in
/-- A kept opening of an accepted run passes, and its check's calls are the extraction's own. -/
theorem passes_of_mem_found (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (c : ECoins D x) {st : AuditStatement}
    (hst : D.parse x = some st) (hacc : (run H P D.V x ω r₀).out = .acc) {q : List Msg × Opening Bool}
    (hq : q ∈ found H D decode x st (run H P D.V x ω) c (outerList H D x (run H P D.V x ω) c)) :
    D.commit.Opens H q.1 q.2 = true ∧
      ∀ t ∈ ((D.commit.opens q.1 q.2).eval H).2, t ∈ ownLog H (run H P D.V x ω) ((E D decode).ext x r₀ c) := by
  rw [found_eq] at hq
  obtain ⟨h1, h2⟩ := Passes.found H D decode x (run H P D.V x ω) st c _ q hq
  exact ⟨h1, fun t ht => ownLog_E H D decode P x ω r₀ c hst hacc t (h2 t ht)⟩

omit [DecidableEq Col] [Zero Col] in
/-- **A disagreement with the network certifier is a collision**: at an accepted run, an extracted trace that opens a
gate the network certifier opens to a value other than the network certifier's makes, with the network certifier's
opening there, calls that collide among the extraction's own and the checks of the network certifier's openings. -/
theorem tapColl_of_disagree (H : Msg → Msg) (D : Deploy Coin Col T) (hb : D.commit.Binds H)
    (decode : List (ℕ × Col) → List T) (tops : AuditStatement → List (Opening Bool)) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x) {st : AuditStatement} (hst : D.parse x = some st)
    (hpass : ∀ q ∈ tops st, D.commit.Opens H st.roots q = true) (hacc : (run H P D.V x s.1 s.2.1).out = .acc)
    {w : Wit} (hw : ((E D decode).eval H P x s.1 s.2.1 s.2.2).1 = some w) {g : ℕ} (hq : ∃ q ∈ tops st, q.pos = g)
    {b : Bool} (hwb : w.val H D.commit st g = some b) (hne : b ≠ tapVal (tops st) g) : TapColl H D decode tops P x s := by
  rw [E_eval H D decode P x s.1 s.2.1 s.2.2 hst hacc, Option.some.injEq] at hw
  subst hw
  obtain ⟨q, hq', hqr, hqp, hqv⟩ := mem_assemble_of_val hwb
  obtain ⟨hqo, hql⟩ := passes_of_mem_found H D decode P x s.1 s.2.1 s.2.2 hst hacc hq'
  obtain ⟨o', ho', ho'p, ho'v⟩ := tapVal_mem hq
  have hqo' : D.commit.Opens H st.roots q.2 = true := by rw [← hqr]; exact hqo
  have hql' : ∀ t ∈ ((D.commit.opens st.roots q.2).eval H).2,
      t ∈ ownLog H (run H P D.V x s.1) ((E D decode).ext x s.2.1 s.2.2) := by
    intro t ht
    rw [← hqr] at ht
    exact hql t ht
  have htl : ((D.parse x).map fun st => tapLog H D.commit st.roots (tops st)).getD [] =
      tapLog H D.commit st.roots (tops st) := by
    rw [hst]
    rfl
  unfold TapColl
  rw [htl]
  exact disagree_collision hb (hqp.trans ho'p.symm) (by rw [hqv, ho'v]; exact hne) hqo' (hpass o' ho')
    (fun t ht => List.mem_append_left _ (hql' t ht))
    (fun t ht => List.mem_append_right _ (List.mem_flatMap.2 ⟨o', ho', ht⟩))

omit [DecidableEq Col] [Zero Col] in
/-- **The reduction's probability step**: the collisions against the network certifier's openings have probability at
most `B_tap`'s. -/
theorem Pr_tapColl_le_Btap (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (tops : AuditStatement → List (Opening Bool)) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x => TapColl H D decode tops P x s) ≤
      (Btap H D.V (E D decode) D.commit D.parse tops Ω P x).Pr H := by
  unfold Btap Finder.Pr
  refine Pr_mono fun s h => ?_
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (findColl_isSome _ h)
  refine ⟨u, v, huv, findColl_spec H _ (fun p hp => ?_) u v huv⟩
  rcases List.mem_append.1 hp with hp | hp
  · exact ownLog_valid H _ _ p hp
  · cases hx : D.parse x with
    | none => rw [hx] at hp; exact absurd hp (by simp)
    | some st => rw [hx] at hp; exact tapLog_valid H D.commit st.roots (tops st) p hp

/-- **The leak bound at the network certifier's rows**, at any deployment `D` of the sampled verifier where `KS` holds
and the trace's commitment binds. Take a public layout `lay` and horizon `k`, `δ ≥ 0`, a decoder that lists every trace
within Flock's radius, inputs `As` that suffice (`AdviceIn`), and the network certifier's openings `tops s` at each
statement. On the statements in the audit's scope: `Pr[acc ∧ ¬RLeakTap] ≤ δ + ε_ks + ε_bind + Pr[B_tap]`. -/
theorem sampledLeak_tap (H : Msg → Msg) (D : Deploy Coin Col T) (hks : KS H D) (hb : D.commit.Binds H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T)
    (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode)
    (As : AuditStatement → (ℕ → Bool) → Set (ℕ → Bool)) (Dj : ℕ)
    (hin : ∀ s P, D.partition s = some P → ∀ ρ, AdviceIn K s.prog P s.design (lay s) (k s) ρ (As s ρ) δ)
    (tops : AuditStatement → List (Opening Bool)) (x : Msg) (hx : ScopeAcc D lay k x) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    PrBad H D.V (RLeakTap H D.commit D.parse D.partition L K S lay k δ As Dj tops) (E D decode) x P ≤
      κ D δ x + Real.sqrt (γ D x * PrB H D.V P x) + (Bext H D.V (E D decode) Ω P x).Pr H +
        (Btap H D.V (E D decode) D.commit D.parse tops Ω P x).Pr H := by
  have hks' := hks δ decode hδ hdec x hx.1 Ω P
  rw [one_mul] at hks'
  unfold PrBad at hks' ⊢
  calc Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ¬ (RLeakTap H D.commit D.parse D.partition L K S lay k δ As Dj tops).ext x
            ((E D decode).eval H P x s.1 s.2.1 s.2.2).1)
      ≤ Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
          ((run H P D.V x s.1 s.2.1).out = .acc ∧
            ¬ (R H D.commit D.parse D.partition δ).ext x ((E D decode).eval H P x s.1 s.2.1 s.2.2).1) ∨
            TapColl H D decode tops P x s) := by
        refine Pr_mono fun s ⟨hac, hR⟩ => ?_
        rcases hw : ((E D decode).eval H P x s.1 s.2.1 s.2.2).1 with _ | w
        · exact Or.inl ⟨hac, by simp [Rel.ext]⟩
        · rw [hw] at hR
          by_cases hA : R H D.commit D.parse D.partition δ x w
          · refine Or.inr ?_
            by_contra hnc
            refine hR (rLeakTap_of_r H D lay k hδ tops hin hx hA fun st hst hpass g hq b hwb => ?_)
            by_contra hne
            exact hnc (tapColl_of_disagree H D hb decode tops P x s hst hpass hac hw hq hwb hne)
          · exact Or.inl ⟨hac, hA⟩
    _ ≤ _ := Pr_union_le _ _
    _ ≤ _ := add_le_add hks' (Pr_tapColl_le_Btap H D decode tops P x)

end Deploy

end NetTiming.TapProof
