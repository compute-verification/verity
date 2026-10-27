import Properties.NetworkCertifier.PerServerIngress
import Proofs.NetworkCertifier.TapRows
import Proofs.NetworkCertifier.Ingress
import Proofs.NetworkCertifier.Horizon
import Proofs.Flock.Claims.Sampling.Wait
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The server's ingress over a rows-form period: the proof

The proof of `Properties.NetworkCertifier.PerServerIngress`.

**Soundness** (`periodSound_check`). An accepted period answers every window (`windowAnswered_of_accept`: `check`
accepts only when `firstFinding` finds nothing, so every window's `rowsWindowCheck` is `.ok`), so by `AnswersOfRecord`
each window's sampled verifier accepted on its coins. On the sampling relation at `δ`, the trace filled in with the
openings the verifier holds (`NetTiming.TapProof.fill`) keeps every unit outside the faulty set correct. Hold the faulty
set's committed gates at the filled values, a checked one at its opening: a checked gate of the faulty set is read only
inside its unit (`ChecksOwned`), so the rest of the scan's Program runs as with the trace's values (`valOver_agree`)
and returns the trace's value at every gate it returns (`NetTiming.SampledAuditProof.valOver_out`). A checked gate
outside the faulty set is its correct unit's, so its trace value is its opening, a constant one included
(`Prog.Correct`). The opened `ok` gates are 1 and each link's ends open to one value, and the registered rows are the
inputs (`inputOf_rows`), so the window's ingress is in `ScanDeclared` (`scanDeclared_of`, `ingressDeclared_of_r`).
Where the trace disagrees with an opening it is a collision `B_tap` finds, as in `NetTiming.TapProof.sampledLeak_tap`
(`ingress_sound`). Each window's prover and statement read the period's coins through its tape (`tapeOf`), which with
the window's own coins is the period's coins rearranged (`windowSplit`); at each tape they are fixed and the window's
coins uniform, so each window's term is its sampled verifier's averaged over the tape (`tape_term_le`,
`window_term_le`), and the union bound over the windows gives `periodErr`.

**The count** (`perServerIngress_count`). The frames the warden releases on the requests' link are its windows' frame
streams in order (`linkFrames_eq`: each record row's cells are its frames, `RecordsCells`), each a stream the scan
accepts (`IngressDeclared` at the registered rows), so the content is one of the products of `ScanCountsRequests`'
sets, and `NetTiming.encard_ingress_decodable_le_contents` adds the timing. `perServerIngress_drawn` is the count where
the draw takes every ingress unit (`IngressDrawn`): no faulty set holds one, and each window's term is
`#faultySets · 2^windowBits`.

* honest: `perServerIngress_honest`, every window's and audit's answer accepted at one `δ` makes `check` accept;
* nontrivial: `perServerIngress_nontrivial`, a check that accepts every period has an error of 1 where some window's
  ingress isn't one the scan accepts.
-/

namespace Properties.NetworkCertifier

open NetTiming Claims Claims.Sampling NetTiming.SampledAuditProof NetTiming.TapProof
open NetworkCheck.Period (Schedule Scan Params Evidence Public Profile rowsWindowCheck auditCheck check findingOf
  firstFinding)
open Properties.NetworkScan (layOf)

/-! ## The scan's Program at the trace's values -/

/-- **Inputs that hold the rows are the rows**: where the values `v` are the rows `ins` at every ingress gate of their
steps, `inputOf` at `ins` and `v` is `v`. -/
theorem inputOf_rows {L S : ℕ} {lay : Layout L S} {k : ℕ} {ins : List (List Bool)} (hlen : ins.length = k)
    {v : ℕ → Bool} (hrows : ∀ t < k, ∀ j g, (lay.ingress t)[j]? = some g → v g = (ins.getD t []).getD j false)
    (g : ℕ) : lay.inputOf k ins v g = v g := by
  unfold Layout.inputOf
  split
  · rename_i i j h
    obtain ⟨i', hi', he⟩ := List.exists_of_findSome?_eq_some h
    obtain ⟨j', hj', hij⟩ := Option.map_eq_some_iff.1 he
    cases hij
    have hi : i < k := by rw [← hlen]; simpa using hi'
    obtain ⟨hjl, hjg, -⟩ := List.findIdx?_eq_some_iff_getElem.1 hj'
    exact (hrows i hi j g (by rw [List.getElem?_eq_getElem hjl, beq_iff_eq.1 hjg])).symm
  · rename_i h
    rw [ite_eq_right]
    simp only [List.any_eq_true, List.mem_range, List.contains_iff_mem, not_exists, not_and]
    intro i hik hg
    have := List.findSome?_eq_none_iff.1 h i (by simpa [hlen] using hik)
    rw [Option.map_eq_none_iff, List.findIdx?_eq_none_iff] at this
    simpa using this g hg

theorem valOver_absent {C : Prog} {inp : ℕ → Bool} {free : ℕ → Option Bool} {g : ℕ} (h : free g = none)
    (hx : C.gates[g]? = none) : valOver C inp free g = false := by
  rw [valOver]
  simp only [h, hx]

/-- **Holdings that agree off a set closed under reading agree off it**: where `f₁` and `f₂` agree at every gate off
`N`, and a gate off `N` that `f₁` doesn't hold reads only gates off `N`, `C` run with either has one value at every
gate off `N`. -/
theorem valOver_agree {C : Prog} {inp : ℕ → Bool} {f₁ f₂ : ℕ → Option Bool} (N : ℕ → Prop)
    (hf : ∀ g, ¬ N g → f₁ g = f₂ g)
    (hN : ∀ g x, C.gates[g]? = some x → ¬ N g → f₁ g = none → ∀ i ∈ x.reads, ¬ N i) :
    ∀ g, ¬ N g → valOver C inp f₁ g = valOver C inp f₂ g := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
  intro hg
  rcases h1 : f₁ g with _ | b
  · have h2 : f₂ g = none := (hf g hg).symm.trans h1
    rcases hx : C.gates[g]? with _ | x
    · rw [valOver_absent h1 hx, valOver_absent h2 hx]
    rw [valOver_gate h1 hx, valOver_gate h2 hx]
    by_cases hxi : x = .input
    · subst hxi
      rfl
    congr 1
    refine val_congr hxi fun i hi => ?_
    by_cases hlt : i < g
    · rw [dif_pos hlt, dif_pos hlt, ih i hlt (hN g x hx hg h1 i hi)]
    · rw [dif_neg hlt, dif_neg hlt]
  · rw [valOver_held h1, valOver_held ((hf g hg).symm.trans h1)]

theorem runIn_const {C : Prog} {ref : ℕ → Option Bool} {u : List ℕ} {g : ℕ} {b : Bool}
    (h : C.gates[g]? = some (.const b)) : C.runIn ref u g = some b := by
  rw [Prog.runIn]
  simp only [h, Option.bind_some, BGate.val]

theorem slot_of_mem {P : Partition} {r : RU} (h : r ∈ P.rus) : ∃ v : ℕ, P.rus[v]? = some r := by
  obtain ⟨v, hv, rfl⟩ := List.getElem_of_mem h
  exact ⟨v, List.getElem?_eq_getElem hv⟩

theorem fill_of_agree {ref : ℕ → Option Bool} {tv : ℕ → Bool} {g : ℕ} (hag : ∀ b, ref g = some b → b = tv g) :
    TapProof.fill ref tv g = some (tv g) := by
  rcases h : ref g with _ | b
  · simp [TapProof.fill, h]
  · simp [TapProof.fill, h, hag b h]

/-- **A window whose committed values the openings hold is one the scan accepts**: at a topological program the
partition partitions, every unit outside `F` correct at the committed values `ref`, `F` faulty at `δ`, the checked
gates committed and their units' own, and `ref` holding the openings `ops` wherever they open a gate, rows `ins` the
openings hold at the ingress gates, every `ok` gate opened to 1 and every link's two ends opened to one value are in
`ScanDeclared`. The advice is the trace's values; `F`'s committed gates are held at them, except a checked one, held at
its opening. A checked gate of `F` is read only inside its faulty proof block, whose gates `C` doesn't compute past its held
ones (`valOver_agree`), so the rest of `C` runs as with the trace's values (`valOver_out`); and a checked gate outside
`F` has the trace's value, its opening, a constant one included, since its unit commits it and is correct. -/
theorem scanDeclared_of {S : ℕ} {C : Prog} {P : Partition} {d : Design} {lay : Layout 1 S} {k : ℕ} {ok : List ℕ}
    {links : List (ℕ × ℕ)} {δ : ℝ} (hT : C.Topological) (hP : P.Partitions C)
    (hin : ∀ t < k, ∀ g ∈ lay.ingress t, C.gates[g]? = some .input) {ref : ℕ → Option Bool} {F : Finset ℕ}
    (ι : ℕ → ℕ → Bool) (hι : ∀ v r, P.rus[v]? = some r → v ∉ F → ∀ u ∈ r, C.Correct (RU.ref C r ref (ι v)) u)
    (hF : F ∈ faultySets d P δ) (hcomm : ChecksCommitted C ok links)
    (hown : ChecksOwned C P (ok ++ links.map Prod.snd)) {ops : List (Opening Bool)}
    (hval : ∀ g, (∃ q ∈ ops, q.pos = g) → ref g = some (tapVal ops g))
    {ins : List (List Bool)} (hlen : ins.length = k)
    (hrows : ∀ t < k, ∀ j g, (lay.ingress t)[j]? = some g →
      (∃ q ∈ ops, q.pos = g) ∧ tapVal ops g = (ins.getD t []).getD j false)
    (hok : ∀ g ∈ ok, (∃ q ∈ ops, q.pos = g) ∧ tapVal ops g = true)
    (hchain : ∀ e ∈ links, (∃ q ∈ ops, q.pos = e.1) ∧ (∃ q ∈ ops, q.pos = e.2) ∧ tapVal ops e.1 = tapVal ops e.2) :
    ins ∈ ScanDeclared C P d lay k ok links δ := by
  classical
  have hN := hP.1
  have hread : ∀ g, (∃ q ∈ ops, q.pos = g) → (∀ b, C.gates[g]? ≠ some (.const b)) →
      traceVal C ref g = tapVal ops g := fun g hq hnc => by
    rw [traceVal, read_of_not_const hnc, hval g hq, Option.getD_some]
  have hinp : lay.inputOf k ins (traceVal C ref) = traceVal C ref := funext (inputOf_rows hlen fun t ht j g hg => by
    have hgi : C.gates[g]? = some .input := hin t ht g (List.mem_of_getElem? hg)
    rw [hread g (hrows t ht j g hg).1 fun b hb => by rw [hgi] at hb; cases hb]
    exact (hrows t ht j g hg).2)
  have hchk : ∀ g ∈ ok ++ links.map Prod.snd, (∃ q ∈ ops, q.pos = g) ∧ g ∈ C.outs := by
    intro g hg
    rcases List.mem_append.1 hg with hg | hg
    · exact ⟨(hok g hg).1, hcomm.1 g hg⟩
    · obtain ⟨e, he, rfl⟩ := List.mem_map.1 hg
      exact ⟨(hchain e he).2.1, (hcomm.2 e he).1⟩
  let M : ℕ → Prop := fun g => g ∈ ok ++ links.map Prod.snd ∧ FaultyGate C P F g
  let free : ℕ → Option Bool := fun g => if M g then some (tapVal ops g) else heldAdv C P F ref g
  let N : ℕ → Prop := fun g =>
    M g ∨ ∃ v ∈ F, ∃ r, P.rus[v]? = some r ∧ g ∈ RU.gates r ∧ C.Boundary (RU.gates r) g = false
  have hint : ∀ g, ¬ FaultyGate C P F g → ∀ v ∈ F, ∀ r, P.rus[v]? = some r → g ∈ RU.gates r → N g := by
    intro g hfg v hv r hr hg
    refine Or.inr ⟨v, hv, r, hr, hg, ?_⟩
    cases hb : C.Boundary (RU.gates r) g
    · rfl
    · exact (hfg ⟨v, hv, r, hr, hg, hb⟩).elim
  have hfree : ∀ g, ¬ N g → free g = heldAdv C P F ref g := fun g hg => if_neg fun hM => hg (Or.inl hM)
  have hagree : ∀ g, ¬ N g →
      valOver C (traceVal C ref) free g = valOver C (traceVal C ref) (heldAdv C P F ref) g := by
    refine valOver_agree N hfree fun g x hx hg hnone i hi hNi => ?_
    have hfg : ¬ FaultyGate C P F g := fun hfg => by
      rw [hfree g hg, heldAdv_of hfg] at hnone
      cases hnone
    rcases hNi with ⟨hic, v, hv, r, hr, hir, -⟩ | ⟨v, hv, r, hr, hir, hbi⟩
    · obtain ⟨r', hr', hir', hrd⟩ := hown i hic
      obtain ⟨v', hv'⟩ := slot_of_mem hr'
      have hvv : v' = v := slot_unique hN hv' hr hir' hir
      rw [hvv, hr, Option.some.injEq] at hv'
      rw [← hv'] at hrd
      exact hg (hint g hfg v hv r hr (hrd g x hx hi))
    · by_cases hgr : g ∈ RU.gates r
      · exact hg (hint g hfg v hv r hr hgr)
      · rw [boundary_of_read hx hi hgr] at hbi
        cases hbi
  have hval_chk : ∀ g ∈ ok ++ links.map Prod.snd, valOver C (traceVal C ref) free g = tapVal ops g := by
    intro g hg
    obtain ⟨hq, hout⟩ := hchk g hg
    by_cases hfg : FaultyGate C P F g
    · exact valOver_held (show free g = some (tapVal ops g) from if_pos ⟨hg, hfg⟩)
    have hNg : ¬ N g := by
      rintro (⟨-, hf⟩ | ⟨v, -, r, -, -, hb⟩)
      · exact hfg hf
      · rw [boundary_of_out hout] at hb
        cases hb
    rw [hagree g hNg,
      valOver_out hT hP ι hι (fun _ => True) (fun _ _ _ _ _ _ => trivial) (fun _ _ _ => rfl) hout trivial]
    by_cases hk : ∃ b, C.gates[g]? = some (.const b)
    · obtain ⟨b, hb⟩ := hk
      obtain ⟨r, hr, hgr, -⟩ := hown g hg
      obtain ⟨v, hv⟩ := slot_of_mem hr
      have hvF : v ∉ F := fun hvF => hfg ⟨v, hvF, r, hv, hgr, boundary_of_out hout⟩
      obtain ⟨u, hu, hgu⟩ := List.mem_flatten.1 hgr
      obtain ⟨b', hb', hrun⟩ := (hι v r hv hvF u hu g hgu).2 (boundary_of_out hout)
      rw [ref_of_boundary (boundary_of_out hout), hval g hq] at hb'
      rw [runIn_const hb] at hrun
      rw [traceVal, read_const hb, Option.getD_some]
      exact (Option.some.inj hrun).trans (Option.some.inj hb').symm
    · push Not at hk
      exact hread g hq hk
  simp only [ScanDeclared, Set.mem_ofPred_eq]
  refine ⟨hlen, F, hF, traceVal C ref, free, fun g b h => ?_, fun e he => ?_, fun g hg => ?_⟩
  · by_cases hM : M g
    · exact hM.2
    · exact faultyGate_of_heldAdv ((if_neg hM).symm.trans h)
  · obtain ⟨h1, -, heq⟩ := hchain e he
    have he1 : C.gates[e.1]? = some .input := (hcomm.2 e he).2
    rw [hinp, hval_chk e.2 (List.mem_append_right _ (List.mem_map_of_mem he)),
      hread e.1 h1 (fun b hb => by rw [he1] at hb; cases hb), heq]
  · rw [hinp, hval_chk g (List.mem_append_left _ hg), (hok g hg).2]

section Deploy

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **On the sampling relation the registered ingress is one the scan accepts**: at a scan statement and `δ ≥ 0`, the
sampling relation at a trace that agrees with the openings `ops` wherever both open a gate gives `IngressDeclared`. -/
theorem ingressDeclared_of_r (H : Msg → Msg) (D : Deploy Coin Col T) (s : Schedule) {δ : ℝ} (hδ : 0 ≤ δ)
    (ops : Stmt → List (Opening Bool)) {x : Msg} (hx : ScanScope D s x) {w : Wit}
    (hR : R H D.commit D.parse D.partition δ x w)
    (hagree : ∀ st, D.parse x = some st → (∀ q ∈ ops st, D.commit.Opens H st.roots q = true) →
      ∀ g, (∃ q ∈ ops st, q.pos = g) → ∀ b, w.val H D.commit st g = some b → b = tapVal (ops st) g) :
    IngressDeclared H D s ops δ x := by
  obtain ⟨st, P, hs, hP, hM⟩ := hR
  obtain ⟨⟨⟨s₁, P₁, hs₁, hP₁, hsc⟩, s₂, hs₂, hfit⟩, hcomm⟩ := hx
  rw [hs] at hs₁ hs₂
  obtain rfl := Option.some.inj hs₁
  obtain rfl := Option.some.inj hs₂
  rw [hP] at hP₁
  obtain rfl := Option.some.inj hP₁
  have hF := faulty_mem_faultySets hM (disjoint_allSlots hδ hM)
  obtain ⟨ι, hι⟩ := correct_of_not_faulty H D.commit st P w
  have hext : ∀ g b, w.val H D.commit st g = some b →
      TapProof.fill (w.val H D.commit st) (tapVal (ops st)) g = some b := fill_ext
  refine ⟨st, P, hs, hP, fun ins hio => ?_⟩
  obtain ⟨hpass, ⟨hlen, hrows⟩, hok, hchain⟩ := hio
  obtain ⟨hcc, hown⟩ := hcomm st hs
  exact scanDeclared_of hsc.1 hsc.2.1 hfit.2.1 ι
    (fun v r hr hv u hu => correct_mono (ruRef_mono hext) (hι v r hr hv u hu)) hF hcc (hown P hP)
    (fun g hq => fill_of_agree (hagree st hs hpass g hq)) hlen (fun t ht => (hrows t ht).2) hok hchain

/-- **The scan's soundness at one window**: at a deployment where `KS` holds and the commitment binds, on a scan
statement `x`, `Pr[acc ∧ ¬IngressDeclared x] ≤ δ + ε_ks + ε_bind + Pr[B_tap]` over the tape, the run's coins and the
extractor's. -/
theorem ingress_sound (H : Msg → Msg) (D : Deploy Coin Col T) (hks : KS H D) (hb : D.commit.Binds H) (s : Schedule)
    (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (ops : Stmt → List (Opening Bool))
    (x : Msg) (hx : ScanScope D s x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun u : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
        (run H P D.V x u.1 u.2.1).out = .acc ∧ ¬ IngressDeclared H D s ops δ x) ≤
      κ D δ x + Real.sqrt (γ D x * PrB H D.V P x) + (Bext H D.V (E D decode) Ω P x).Pr H +
        (Btap H D.V (E D decode) D.commit D.parse ops Ω P x).Pr H := by
  have hks' := hks δ decode hδ hdec x hx.1.1 Ω P
  rw [one_mul] at hks'
  unfold PrBad at hks'
  calc Pr (fun u : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
        (run H P D.V x u.1 u.2.1).out = .acc ∧ ¬ IngressDeclared H D s ops δ x)
      ≤ Pr (fun u : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
          ((run H P D.V x u.1 u.2.1).out = .acc ∧
            ¬ (R H D.commit D.parse D.partition δ).ext x ((E D decode).eval H P x u.1 u.2.1 u.2.2).1) ∨
            TapColl H D decode ops P x u) := by
        refine Claims.Pr_mono fun u ⟨hac, hID⟩ => ?_
        rcases hw : ((E D decode).eval H P x u.1 u.2.1 u.2.2).1 with _ | w
        · exact Or.inl ⟨hac, by simp [Rel.ext]⟩
        · by_cases hA : R H D.commit D.parse D.partition δ x w
          · refine Or.inr ?_
            by_contra hnc
            refine hID (ingressDeclared_of_r H D s hδ ops hx hA fun st hst hpass g hq b hwb => ?_)
            by_contra hne
            exact hnc (tapColl_of_disagree H D hb decode ops P x u hst hpass hac hw hq hwb hne)
          · exact Or.inl ⟨hac, hA⟩
    _ ≤ _ := Claims.Sampling.Pr_union_le _ _
    _ ≤ _ := add_le_add hks' (Pr_tapColl_le_Btap H D decode ops P x)

theorem stepWith_comap_tape {Ω Ω' : Type} (H : Msg → Msg) (P : Prover Ω) (f : Ω' → Ω) (x : Msg) (ω : Ω')
    (vnext : List Msg → Claims.OC Claims.Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Claims.Trace),
      stepWith H (P.comap f) x ω vnext n hist as t = stepWith H P x (f ω) vnext n hist as t
  | 0, _, _, _ => rfl
  | n + 1, hist, as, t => by
      have e : (P.comap f).next x hist ω = P.next x hist (f ω) := rfl
      simp only [stepWith, e]
      split
      · exact stepWith_comap_tape H P f x ω vnext n _ _ _
      · rfl
      · rfl

omit [Fintype Coin] [Nonempty Coin] in
/-- The run reads a prover on other tapes at the tape it reads. -/
theorem run_comap_tape {Ω Ω' : Type} (H : Msg → Msg) (P : Prover Ω) (f : Ω' → Ω) (V : Claims.Verifier Coin) (x : Msg)
    (ω : Ω') (r : Fin (V.k x) → Coin) : run H (P.comap f) V x ω r = run H P V x (f ω) r :=
  stepWith_comap_tape H P f x ω _ _ _ _ _

/-- **The scan's soundness at one tape**: at a scan statement `m` of `kw` rounds and a prover `P` at its tape `t`,
`Pr[acc ∧ ¬IngressDeclared m] ≤ windowErr` over the run's `kw` coins: `ingress_sound` on the empty tape, at `P` read at
`t`. -/
theorem tape_term_le (H : Msg → Msg) (D : Deploy Coin Col T) (hks : KS H D) (hb : D.commit.Binds H) (s : Schedule)
    (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (ops : Stmt → List (Opening Bool))
    (m : Msg) (hm : ScanScope D s m) {Ω' : Type} (P : Prover Ω') (t : Ω') {kw : ℕ} (hk : D.V.k m = kw) :
    Pr (fun r : Fin kw → Coin => (run H P D.V m t fun j => r (Fin.cast hk j)).out = .acc ∧
        ¬ IngressDeclared H D s ops δ m) ≤
      windowErr H D δ decode ops m (P.comap fun _ => t) := by
  subst hk
  refine le_trans (le_of_eq ?_) (ingress_sound H D hks hb s δ decode hδ hdec ops m hm Unit (P.comap fun _ => t))
  calc _ = Pr (fun r : Fin (D.V.k m) → Coin => (run H P D.V m t r).out = .acc ∧ ¬ IngressDeclared H D s ops δ m) :=
        Claims.Pr_congr fun _ => Iff.rfl
    _ = Pr (fun a : Unit × (Fin (D.V.k m) → Coin) =>
          (run H P D.V m t a.2).out = .acc ∧ ¬ IngressDeclared H D s ops δ m) :=
        (NetTiming.Pr_equiv (Equiv.punitProd _) fun r =>
          (run H P D.V m t r).out = .acc ∧ ¬ IngressDeclared H D s ops δ m).symm
    _ = Pr (fun a : Unit × (Fin (D.V.k m) → Coin) =>
          (run H (P.comap fun _ => t) D.V m a.1 a.2).out = .acc ∧ ¬ IngressDeclared H D s ops δ m) :=
        Claims.Pr_congr fun a => by rw [run_comap_tape]
    _ = Pr (fun u : Unit × (Fin (D.V.k m) → Coin) × (E D decode).C m =>
          (run H (P.comap fun _ => t) D.V m u.1 u.2.1).out = .acc ∧ ¬ IngressDeclared H D s ops δ m) :=
        (Claims.Pr_drop fun (u : Unit) (r : Fin (D.V.k m) → Coin) =>
          (run H (P.comap fun _ => t) D.V m u r).out = .acc ∧ ¬ IngressDeclared H D s ops δ m).symm

/-- Window `i`'s coins split off the period's: its prover's tape (`tapeOf`) and its own coins. -/
def windowSplit (Coin : Type) {n : ℕ} (kw : Fin n → ℕ) (Ω : Type) (i : Fin n) :
    PeriodCoins Coin kw Ω ≃ WindowTape Coin kw Ω i × (Fin (kw i) → Coin) :=
  (Equiv.prodCongr (Equiv.refl Ω) (Equiv.piSplitAt i fun j => Fin (kw j) → Coin)).trans
    ((Equiv.prodCongr (Equiv.refl Ω) (Equiv.prodComm _ _)).trans (Equiv.prodAssoc _ _ _).symm)

/-- **Window `i`'s term**: over the period's coins, window `i`'s sampled verifier accepts on its statement with its
prover at its tape while its ingress isn't one the scan accepts with probability at most its error averaged over the
tape. The period's coins are the window's tape and its own coins (`windowSplit`); at each tape the statement and the
prover are fixed, and the window's coins are uniform (`tape_term_le`). -/
theorem window_term_le (H : Msg → Msg) (D : Deploy Coin Col T) (hks : KS H D) (hb : D.commit.Binds H) (s : Schedule)
    (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (ops : Stmt → List (Opening Bool))
    {n : ℕ} {kw : Fin n → ℕ} {Ω : Type} [Fintype Ω] (xw : (i : Fin n) → WindowTape Coin kw Ω i → Msg)
    (hk : ∀ i t, D.V.k (xw i t) = kw i) (hx : ∀ i t, ScanScope D s (xw i t))
    (P : (i : Fin n) → Prover (WindowTape Coin kw Ω i)) (i : Fin n) :
    Pr (fun ω : PeriodCoins Coin kw Ω =>
        (run H (P i) D.V (xw i (tapeOf ω i)) (tapeOf ω i) fun j => ω.2 i (Fin.cast (hk i (tapeOf ω i)) j)).out =
          .acc ∧ ¬ IngressDeclared H D s ops δ (xw i (tapeOf ω i))) ≤
      Ex fun t : WindowTape Coin kw Ω i => windowErr H D δ decode ops (xw i t) ((P i).comap fun _ => t) := by
  have key : ∀ q : WindowTape Coin kw Ω i → (Fin (kw i) → Coin) → Prop,
      Pr (fun ω : PeriodCoins Coin kw Ω => q (tapeOf ω i) (ω.2 i)) = Ex fun t => Pr fun r => q t r :=
    fun q => (NetTiming.Pr_equiv (windowSplit Coin kw Ω i) fun u => q u.1 u.2).trans (NetTiming.Pr_prod_eq_Ex _)
  exact (key fun t r => (run H (P i) D.V (xw i t) t fun j => r (Fin.cast (hk i t) j)).out = .acc ∧
      ¬ IngressDeclared H D s ops δ (xw i t)).trans_le
    (Claims.Ex_mono fun t => tape_term_le H D hks hb s δ decode hδ hdec ops (xw i t) (hx i t) (P i) t (hk i t))

/-- **The period's error under a bound at every tape**: where window `i`'s error is at most `ε i` at every tape, the
period's is at most `Σ_i ε i`. -/
theorem periodErr_le (H : Msg → Msg) (D : Deploy Coin Col T) (δ : ℝ) (decode : List (ℕ × Col) → List T)
    (ops : Stmt → List (Opening Bool)) {n : ℕ} {kw : Fin n → ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (xw : (i : Fin n) → WindowTape Coin kw Ω i → Msg) (P : (i : Fin n) → Prover (WindowTape Coin kw Ω i))
    (ε : Fin n → ℝ) (h : ∀ i t, windowErr H D δ decode ops (xw i t) ((P i).comap fun _ => t) ≤ ε i) :
    periodErr H D δ decode ops xw P ≤ ∑ i, ε i := by
  unfold periodErr
  exact Finset.sum_le_sum fun i _ => (Claims.Ex_mono (h i)).trans (Claims.Ex_const (ε i)).le

end Deploy

/-! ## The check of record -/

theorem firstFinding_eq_none {fs : List (Option (Core.Verdict × String))} (h : firstFinding fs = none) :
    ∀ f ∈ fs, f = none := by
  unfold firstFinding at h
  split at h
  · cases h
  · exact List.findSome?_eq_none_iff.1 h

theorem firstFinding_of_none {fs : List (Option (Core.Verdict × String))} (h : ∀ f ∈ fs, f = none) :
    firstFinding fs = none := by
  unfold firstFinding
  rw [List.findSome?_eq_none_iff.2 fun f hf => by rw [h f hf]; rfl]
  exact List.findSome?_eq_none_iff.2 h

/-- Window `i`'s finding is among the findings of the windows `check` maps. -/
theorem mem_findings {α β γ : Type} (f : (α × β) × ℕ → NetworkCheck.Period.Found γ) (cw : Array α) (ew : Array β)
    {i : ℕ} (hc : i < cw.size) (he : i < ew.size) :
    findingOf (f ((cw[i], ew[i]), i)) ∈ ((cw.zip ew).zipIdx.map f).toList.map findingOf := by
  refine List.mem_map_of_mem ?_
  have hz : i < ((cw.zip ew).zipIdx.map f).size := by simp [Array.size_zipIdx, Array.size_zip]; omega
  have := Array.getElem_mem_toList hz
  simpa [Array.getElem_map, Array.getElem_zipIdx, Array.getElem_zip] using this

/-- **An accepted rows-form period answers every window**: `check` accepts only when no finding is left, so each
window's `rowsWindowCheck` is `.ok`. -/
theorem windowAnswered_of_accept {p : Params} {s : Schedule} {sd : ByteArray} {sc : Scan}
    {c : NetworkCheck.Period.Commitment} {e : Evidence} (hform : p.form = .rows s sd sc)
    (h : PeriodAccepts (check p c e)) : ∀ i < p.windows, WindowAnswered p s sd c e i := by
  obtain ⟨pub, pr, h⟩ := h
  unfold check at h
  split_ifs at h with h1 h2 h3 h4
  rw [hform] at h
  dsimp only at h
  split at h
  · cases h
  · cases h
  rename_i hff
  have hall := firstFinding_eq_none hff
  intro i hi
  have hc : i < c.windows.size := by omega
  have he : i < e.windows.size := by omega
  have hw := hall _ (List.mem_append_left _ (List.mem_append_left _ (mem_findings _ c.windows e.windows hc he)))
  dsimp only at hw
  rcases hr : rowsWindowCheck p s sd (p.first + i) c.windows[i] e.windows[i] with err | f
  · rw [hr] at hw
    cases hw
  · exact ⟨c.windows[i], e.windows[i], f, Array.getElem?_eq_getElem hc, Array.getElem?_eq_getElem he, hr⟩

section Deploy

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The period's soundness at the check of record**: the union over the windows of `window_term_le`, each accepted
window's answer its sampled verifier's acceptance (`AnswersOfRecord`). -/
theorem periodSound_check (H : Msg → Msg) (D : Deploy Coin Col T) (hks : KS H D) (hb : D.commit.Binds H)
    (s : Schedule) (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (p : Params) (sd : ByteArray)
    (sc : Scan) (hform : p.form = .rows s sd sc) (ops : Stmt → List (Opening Bool)) {kw : Fin p.windows → ℕ}
    {Ω : Type} [Fintype Ω] (xw : (i : Fin p.windows) → WindowTape Coin kw Ω i → Msg)
    (hk : ∀ i t, D.V.k (xw i t) = kw i) (hx : ∀ i t, ScanScope D s (xw i t))
    (P : (i : Fin p.windows) → Prover (WindowTape Coin kw Ω i))
    (c : PeriodCoins Coin kw Ω → NetworkCheck.Period.Commitment) (ev : PeriodCoins Coin kw Ω → Evidence)
    (hrec : AnswersOfRecord H D p s sd xw hk P c ev) :
    PeriodSound check H D s δ decode ops p xw P c ev := by
  unfold PeriodSound periodErr
  let W : Fin p.windows → PeriodCoins Coin kw Ω → Prop := fun i ω =>
    (run H (P i) D.V (xw i (tapeOf ω i)) (tapeOf ω i) fun j => ω.2 i (Fin.cast (hk i (tapeOf ω i)) j)).out = .acc ∧
      ¬ IngressDeclared H D s ops δ (xw i (tapeOf ω i))
  calc Pr (fun ω : PeriodCoins Coin kw Ω =>
        PeriodAccepts (check p (c ω) (ev ω)) ∧ ∃ i, ¬ IngressDeclared H D s ops δ (xw i (tapeOf ω i)))
      ≤ Pr (fun ω => ∃ i ∈ (Finset.univ : Finset (Fin p.windows)), W i ω) :=
        Claims.Pr_mono fun ω ⟨hacc, i, hi⟩ =>
          ⟨i, Finset.mem_univ _, hrec ω i (windowAnswered_of_accept hform hacc i.val i.isLt), hi⟩
    _ ≤ ∑ i, Pr (W i) := Claims.Sampling.Pr_exists_le Finset.univ W
    _ ≤ _ := Finset.sum_le_sum fun i _ => window_term_le H D hks hb s δ decode hδ hdec ops xw hk hx P i

end Deploy

/-! ## The count -/

theorem cellChunks {n : ℕ} (hn : 0 < n) : ∀ cs : List (List Bool), (∀ c ∈ cs, c.length = n) →
    (List.range (cs.flatten.length / n)).map (fun k => (cs.flatten.drop (k * n)).take n) = cs
  | [], _ => by simp
  | c :: cs, h => by
    have hc : c.length = n := h c (by simp)
    have ih := cellChunks hn cs fun c' hc' => h c' (by simp [hc'])
    rw [List.flatten_cons, List.length_append, hc, Nat.add_comm, Nat.add_div_right _ hn, List.range_succ_eq_map,
      List.map_cons, List.map_map]
    congr 1
    · simp [← hc]
    · conv_rhs => rw [← ih]
      refine List.map_congr_left fun k _ => ?_
      simp only [Function.comp, Nat.succ_mul]
      rw [Nat.add_comm, ← List.drop_drop, ← hc, List.drop_left]

theorem filter_cells {n : ℕ} : ∀ (l : List (Option (List Bool))), (∀ c, some c ∈ l → c.any id = true) →
    (l.map fun c => c.getD (List.replicate n false)).filter (fun c => c.any id) = l.filterMap id
  | [], _ => rfl
  | none :: l, h => by
    rw [List.map_cons, List.filter_cons_of_neg (by simp), filter_cells l fun c hc => h c (by simp [hc])]
    rfl
  | some c :: l, h => by
    rw [List.map_cons, List.filter_cons_of_pos (by simpa using h c (by simp)),
      filter_cells l fun c hc => h c (by simp [hc])]
    rfl

/-- **A record row's cells are its frames**: where every frame is a cell of `8B` bits, not all zero, the row's bits cut
into cells, padding dropped, are its frames. -/
theorem cells_rowBits {S : ℕ} (s : Schedule) (r : Row S (List Bool))
    (h : ∀ σ c, r σ = some c → c.length = 8 * s.cellBytes ∧ c.any id = true) :
    (cellsOf s (rowBits s r)).filter (fun c => c.any id) = rowFrames r := by
  have hmem : ∀ c, some c ∈ List.ofFn r → c.length = 8 * s.cellBytes ∧ c.any id = true := by
    intro c hc
    obtain ⟨σ, hσ⟩ := List.mem_ofFn.1 hc
    exact h σ c hσ
  rcases Nat.eq_zero_or_pos (8 * s.cellBytes) with h0 | hpos
  · have hnone : ∀ c, some c ∉ List.ofFn r := fun c hc => by
      have := hmem c hc
      rw [h0, List.length_eq_zero_iff] at this
      rw [this.1] at this
      cases this.2
    have hf : rowFrames r = [] := by
      rw [rowFrames, List.filterMap_eq_nil_iff]
      intro a ha
      cases a with
      | none => rfl
      | some c => exact absurd ha (hnone c)
    rw [hf, cellsOf, h0]
    simp
  · unfold cellsOf rowBits
    rw [List.flatMap_def, cellChunks hpos _ (by
      intro c hc
      obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hc
      cases o with
      | none => simp
      | some c => exact (hmem c ho).1)]
    exact filter_cells _ fun c hc => (hmem c hc).2

theorem frameStream_windowIns {K S : ℕ} (s : Schedule) (o : List (Obs 1 K S)) (hcells : RecordsCells s o) (i : ℕ) :
    frameStream s (windowIns s o i) =
      (List.range s.window).flatMap fun t => ((o[i * s.window + t]?).map fun y => rowFrames (y.rows 0)).getD [] := by
  unfold frameStream windowIns
  rw [List.flatMap_map]
  refine List.flatMap_congr fun t _ => ?_
  rcases hy : o[i * s.window + t]? with _ | y
  · simp [cellsOf]
  · exact cells_rowBits s _ (hcells y (List.mem_of_getElem? hy))

theorem flatMap_eq_range {α β : Type} (h : α → List β) : ∀ l : List α,
    l.flatMap h = (List.range l.length).flatMap fun j => ((l[j]?).map h).getD []
  | [] => rfl
  | a :: l => by
    rw [List.flatMap_cons, List.length_cons, List.range_succ_eq_map, List.flatMap_cons, List.flatMap_map,
      flatMap_eq_range h l]
    rfl

theorem range_mul (T : ℕ) : ∀ W : ℕ,
    List.range (W * T) = (List.range W).flatMap fun i => (List.range T).map (i * T + ·)
  | 0 => by simp
  | W + 1 => by
    rw [Nat.succ_mul, List.range_add, range_mul T W, List.range_succ, List.flatMap_append, List.flatMap_singleton]

theorem flatMap_blocks {α β : Type} (h : α → List β) (l : List α) (W T : ℕ) (hl : l.length = W * T) :
    l.flatMap h = (List.range W).flatMap fun i =>
      (List.range T).flatMap fun t => ((l[i * T + t]?).map h).getD [] := by
  rw [flatMap_eq_range h l, hl, range_mul, List.flatMap_assoc]
  refine List.flatMap_congr fun i _ => ?_
  rw [List.flatMap_map]

theorem ofFn_flatten {β : Type} {W : ℕ} (F : ℕ → List β) :
    (List.ofFn fun i : Fin W => F i.val).flatten = (List.range W).flatMap F := by
  rw [List.flatMap_def]
  congr 1
  exact List.ext_getElem (by simp) fun i _ _ => by simp

/-- **The requests' link carries its windows' frame streams in order**: over `W` windows of `T` steps whose record
holds cells, the frames the warden releases are each window's frame stream, one after another. -/
theorem linkFrames_eq {K S : ℕ} (s : Schedule) (o : List (Obs 1 K S)) (W : ℕ) (hl : o.length = W * s.window)
    (hcells : RecordsCells s o) :
    linkFrames o 0 = (List.ofFn fun i : Fin W => frameStream s (windowIns s o i)).flatten := by
  rw [ofFn_flatten (fun i => frameStream s (windowIns s o i)), linkFrames,
    flatMap_blocks (fun x => rowFrames (x.rows 0)) o W s.window hl]
  refine List.flatMap_congr fun i _ => ?_
  rw [frameStream_windowIns s o hcells i]

theorem encard_contents {W : ℕ} (A : Fin W → Set (List (List Bool))) (b : Fin W → ℕ)
    (hb : ∀ i, (A i).encard ≤ b i) :
    ((fun f : Fin W → List (List Bool) => fun _ : Fin 1 => (List.ofFn f).flatten) '' Set.univ.pi A).encard ≤
      ((∏ i, b i : ℕ) : ℕ∞) := by
  refine (Set.encard_image_le _ _).trans ?_
  rw [Set.encard_pi_eq_prod_encard, Nat.cast_prod]
  exact Finset.prod_le_prod fun i _ => hb i

theorem layOf_congr (s : Schedule) {x y : Stmt} (h : x.desc = y.desc) : layOf s x = layOf s y := by
  unfold layOf
  rw [h]

theorem checksOf_congr (s : Schedule) {x y : Stmt} (h : x.desc = y.desc) : checksOf s x = checksOf s y := by
  unfold checksOf
  rw [h]

/-- **The count**: an outside sender's secrets whose every window's registered ingress the scan accepts number at most
`∏_i scanBound · (∏_ℓ (r ℓ + 1))^n · Dj^n`. -/
theorem perServerIngress_count {Coin Col T : Type} (H : Msg → Msg) (D : Deploy Coin Col T) (δ : ℝ) (p : Params)
    (s : Schedule)
    (cls : ScanClass) (ops : Stmt → List (Opening Bool)) {K : ℕ} {Z In Act Wire : Type}
    (I : Interaction 1 K s.allocation Z In Act Wire) (Dj : ℕ) (r : Fin 1 → ℕ)
    (view : ℕ → Obs 1 K s.allocation → Wire) (hW2 : Regenerates I.certifier view) (hW4 : Packed I.certifier)
    (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier Dj)
    (sw : Fin p.windows → Stmt) (Pw : Fin p.windows → Partition)
    (hcount : ∀ i, ScanCountsRequests s cls (sw i) (Pw i) δ)
    (Ms : Set Z) (xm : Z → Fin p.windows → Msg)
    (hsw : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st →
      st.prog = (sw i).prog ∧ st.desc = (sw i).desc ∧ st.design = (sw i).design ∧ D.partition st = some (Pw i))
    (hacc : ∀ m ∈ Ms, StatusComplete (p.windows * s.window) (I.obsRun m (p.windows * s.window)))
    (hcells : ∀ m ∈ Ms, RecordsCells s (I.obsRun m (p.windows * s.window)))
    (hpass : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st → OpensPass H D.commit st (ops st))
    (hrows : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st →
      WardenRowsOpen s st (ops st) (windowIns s (I.obsRun m (p.windows * s.window)) i.val))
    (hok : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st → OkOpens s st (ops st))
    (hchain : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st → ChainOpens s st (ops st))
    (dec : List Wire → Z) (hdec : ∀ m ∈ Ms, dec (I.wireRun m (p.windows * s.window)) = m) :
    {m | m ∈ Ms ∧ ∀ i, IngressDeclared H D s ops δ (xm m i)}.encard ≤
      (((∏ i, scanBound s cls (sw i) (Pw i) δ) *
        ((∏ ℓ, (r ℓ + 1)) ^ (p.windows * s.window) * Dj ^ (p.windows * s.window)) : ℕ) : ℕ∞) := by
  let A : Fin p.windows → Set (List (List Bool)) := fun i =>
    frameStream s '' ScanDeclared (sw i).prog (Pw i) (sw i).design (layOf s (sw i)) s.window (checksOf s (sw i)).1
      (checksOf s (sw i)).2 δ
  let Cs : Set (Fin 1 → List (List Bool)) :=
    (fun f : Fin p.windows → List (List Bool) => fun _ : Fin 1 => (List.ofFn f).flatten) '' Set.univ.pi A
  have hCs : Cs.encard ≤ ((∏ i, scanBound s cls (sw i) (Pw i) δ : ℕ) : ℕ∞) := encard_contents A _ hcount
  have hc : ∀ m ∈ {m | m ∈ Ms ∧ ∀ i, IngressDeclared H D s ops δ (xm m i)}, ∃ c ∈ Cs,
      ∀ ℓ, linkFrames (I.obsRun m (p.windows * s.window)) ℓ <+: c ℓ := by
    intro m hm
    obtain ⟨hm, hID⟩ := hm
    refine ⟨fun _ => (List.ofFn fun i : Fin p.windows =>
        frameStream s (windowIns s (I.obsRun m (p.windows * s.window)) i)).flatten,
      ⟨fun i => frameStream s (windowIns s (I.obsRun m (p.windows * s.window)) i), fun i _ => ?_, rfl⟩,
      fun ℓ => ?_⟩
    · obtain ⟨st, P, hst, hP, hdecl⟩ := hID i
      obtain ⟨hprog, hdesc, hdesign, hPw⟩ := hsw m hm i st hst
      rw [hP, Option.some.injEq] at hPw
      subst hPw
      have hins := hdecl (windowIns s (I.obsRun m (p.windows * s.window)) i)
        ⟨hpass m hm i st hst, hrows m hm i st hst, hok m hm i st hst, hchain m hm i st hst⟩
      rw [hprog, hdesign, layOf_congr s hdesc, checksOf_congr s hdesc] at hins
      exact ⟨_, hins, rfl⟩
    · simp only [Fin.fin_one_eq_zero ℓ, linkFrames_eq s _ p.windows (I.length_obsRun m _) (hcells m hm),
        List.prefix_refl]
  refine (encard_ingress_decodable_le_contents I _ Dj r view hW2 hW4 hr hJ Cs _ (fun m hm => hacc m hm.1) hc dec
    (fun m hm => hdec m hm.1)).trans ?_
  rw [Nat.cast_mul (∏ i, scanBound s cls (sw i) (Pw i) δ)]
  exact mul_le_mul_left hCs _

/-! ## The guarantee -/

theorem perServerIngress : PerServerIngress := by
  unfold PerServerIngress PerServerIngressBy
  intro Coin Col T _ _ _ _ _ H D hks hb hflat δ decode hδ hdec p s sd sc hform cls ops
  refine ⟨?_, ?_⟩
  · intro kw Ω _ _ xw hk hx P c ev hrec
    exact periodSound_check H D hks hb s δ decode hδ hdec p sd sc hform ops xw hk hx P c ev hrec
  · intro K Z In Act Wire I Dj r view hW2 hW4 hr hJ sw Pw hcount Ms xm hsw hacc hcells hpass hrows hok hchain dec
      hdecm
    exact perServerIngress_count H D δ p s cls ops I Dj r view hW2 hW4 hr hJ sw Pw hcount Ms xm hsw hacc hcells hpass
      hrows hok hchain dec hdecm

/-! ## Every ingress unit drawn -/

theorem faultyIngress_drawn {s : Schedule} {st : Stmt} {P : Partition} {δ : ℝ} {F : Finset ℕ}
    (h : IngressDrawn s st P) (hF : F ∈ faultySets st.design P δ) :
    faultyIngress P (layOf s st) s.window (checksOf s st).1 F = 0 := by
  classical
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset] at hF
  obtain ⟨hsub, -, hdis⟩ := hF
  unfold faultyIngress
  rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
  intro v hv hing
  exact Finset.disjoint_left.1 hdis hv (h v (Finset.mem_range.1 (hsub hv)) hing)

/-- **Where the draw takes every ingress unit, a window's term is `#faultySets · 2^windowBits`.** -/
theorem scanBound_drawn {s : Schedule} (cls : ScanClass) {st : Stmt} {P : Partition} (δ : ℝ)
    (h : IngressDrawn s st P) : scanBound s cls st P δ = (faultySets st.design P δ).card * 2 ^ windowBits s cls := by
  unfold scanBound
  calc ∑ F ∈ faultySets st.design P δ,
        2 ^ (faultyIngress P (layOf s st) s.window (checksOf s st).1 F * faultyIngressBits s cls + windowBits s cls)
      = ∑ F ∈ faultySets st.design P δ, 2 ^ windowBits s cls :=
        Finset.sum_congr rfl fun F hF => by rw [faultyIngress_drawn h hF, zero_mul, zero_add]
    _ = _ := by rw [Finset.sum_const, smul_eq_mul]

/-- **The count where the draw takes every ingress unit** (`IngressDrawn` at every window): `perServerIngress_count`
with no faulty ingress unit. -/
theorem perServerIngress_drawn : PerServerIngressDrawn := by
  unfold PerServerIngressDrawn
  intro Coin Col T H D δ p s cls ops K Z In Act Wire I Dj r view hW2 hW4 hr hJ sw Pw hcount hdrawn Ms xm hsw hacc
    hcells hpass hrows hok hchain dec hdec
  have h := perServerIngress_count H D δ p s cls ops I Dj r view hW2 hW4 hr hJ sw Pw hcount Ms xm hsw hacc hcells
    hpass hrows hok hchain dec hdec
  rwa [Finset.prod_congr rfl fun i _ => scanBound_drawn cls δ (hdrawn i)] at h

/-! ## Its honest and nontrivial statements -/

theorem ratio_bne_self (r : Core.Ratio) : (r != r) = false := by
  cases r
  show (!((_ : Nat) == _ && (_ : Nat) == _)) = false
  simp

theorem perServerIngress_honest : PerServerIngressHonest := by
  unfold PerServerIngressHonest
  intro p s sd sc c e hform hh
  obtain ⟨hcw, hca, hew, hea, ⟨δ0, hwin⟩, haud⟩ := hh
  refine ⟨?_, fun i hi => ?_⟩
  · unfold PeriodAccepts check
    rw [ite_eq_right (not_not.2 hcw), ite_eq_right (not_not.2 hca), ite_eq_right (not_not.2 (hew.trans hcw.symm)),
      ite_eq_right (not_not.2 (hea.trans hca.symm)), hform]
    dsimp only
    rw [firstFinding_of_none]
    · exact ⟨_, _, rfl⟩
    have hws : ∀ w ∈ (c.windows.zip e.windows).zipIdx.map (fun x : (NetworkCheck.Period.Registration ×
        NetworkCheck.Period.Window) × ℕ => some <$> rowsWindowCheck p s sd (p.first + x.2) x.1.1 x.1.2),
        ∃ f, w = .ok (some f) ∧ f.delta = δ0 := by
      intro w hw
      obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hw
      have hi' : i < p.windows := by simp [Array.size_zipIdx, Array.size_zip] at hi; omega
      obtain ⟨reg, x, f, hreg, hx, hf, hd⟩ := hwin i hi'
      rw [Array.getElem?_eq_getElem (by omega), Option.some.injEq] at hreg hx
      simp only [Array.getElem_map, Array.getElem_zipIdx, Array.getElem_zip, Nat.zero_add]
      rw [hreg, hx, hf]
      exact ⟨f, rfl, hd⟩
    have hscan : ∀ f ∈ ((c.windows.zip e.windows).zipIdx.map (fun x : (NetworkCheck.Period.Registration ×
        NetworkCheck.Period.Window) × ℕ => some <$> rowsWindowCheck p s sd (p.first + x.2) x.1.1 x.1.2)).filterMap
        (·.toOption.join), f.delta = δ0 := by
      intro f hf
      obtain ⟨w, hw, hwf⟩ := Array.mem_filterMap.1 hf
      obtain ⟨f', rfl, hd⟩ := hws w hw
      cases hwf
      exact hd
    intro f hf
    simp only [List.mem_append, List.mem_map, List.mem_singleton] at hf
    rcases hf with (⟨w, hw, rfl⟩ | rfl) | ⟨w, hw, rfl⟩
    · obtain ⟨f, rfl, -⟩ := hws w (Array.mem_def.2 hw)
      rfl
    · rcases h0 : ((((c.windows.zip e.windows).zipIdx.map (fun x : (NetworkCheck.Period.Registration ×
          NetworkCheck.Period.Window) × ℕ => some <$> rowsWindowCheck p s sd (p.first + x.2) x.1.1 x.1.2)).filterMap
          (·.toOption.join))[0]?) with _ | f0
      · simp only [h0, Option.bind_none]
      · simp only [h0, Option.bind_some, Option.map_eq_none_iff]
        rw [Array.find?_eq_none]
        intro f hf
        rw [hscan f hf, hscan f0 (Array.mem_of_getElem? h0), ratio_bne_self]
        exact Bool.false_ne_true
    · obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem (Array.mem_def.2 hw)
      have hi' : i < p.audits.size := by simp [Array.size_zipIdx, Array.size_zip] at hi; omega
      obtain ⟨a, reg, x, f, ha, hreg, hx, hf⟩ := haud i hi'
      rw [Array.getElem?_eq_getElem (by omega), Option.some.injEq] at ha hreg hx
      simp only [Array.getElem_map, Array.getElem_zipIdx, Array.getElem_zip, Nat.zero_add]
      rw [ha, hreg, hx, hf]
      rfl
  · obtain ⟨reg, x, f, hreg, hx, hf, -⟩ := hwin i hi
    exact ⟨reg, x, f, hreg, hx, hf⟩

theorem perServerIngress_nontrivial : PerServerIngressNontrivial := by
  unfold PerServerIngressNontrivial
  intro chk hall Coin Col T _ _ _ _ _ H D s δ decode ops p kw Ω _ _ xw P c ev hbad herr hsound
  have hsure : Pr (fun ω : PeriodCoins Coin kw Ω => PeriodAccepts (chk p (c ω) (ev ω)) ∧
      ∃ i, ¬ IngressDeclared H D s ops δ (xw i (tapeOf ω i))) = 1 :=
    (Claims.Pr_congr fun ω => ⟨fun _ => trivial, fun _ => ⟨hall p (c ω) (ev ω), hbad ω⟩⟩).trans
      Claims.Sampling.Pr_true
  unfold PeriodSound at hsound
  linarith

end Properties.NetworkCertifier
