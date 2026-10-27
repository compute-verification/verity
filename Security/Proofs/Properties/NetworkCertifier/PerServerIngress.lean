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
openings the verifier holds (`NetTiming.TapProof.fill`) keeps every unit outside the faulty set correct, and at the
faulty set's committed gates held at the filled values the scan's Program returns the trace's value at every gate it
returns (`NetTiming.SampledAuditProof.valOver_out`): the opened `ok` gates are 1 and each link's ends open to one value,
and the registered rows are the inputs (`inputOf_rows`), so the window's ingress is in `ScanDeclared`
(`scanDeclared_of`, `ingressDeclared_of_r`). Where the trace disagrees with an opening it is a collision `B_tap` finds,
as in `NetTiming.TapProof.sampledLeak_tap` (`ingress_sound`). Each window's prover reads the period's coins through its
tape (`tapeOf`), which with the window's own coins is the period's coins rearranged (`windowSplit`), so each window's
term is its sampled verifier's (`window_term_le`), and the union bound over the windows gives `periodErr`.

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

/-- **A window whose committed values the openings hold is one the scan accepts**: at a topological program the
partition partitions, every unit outside `F` correct at the committed values `ref`, `F` faulty at `δ`, and `ref`
holding the openings `ops` wherever they open a gate (a constant where the opening says so), rows `ins` the openings
hold at the ingress gates, every `ok` gate opened to 1 and every link's two ends opened to one value are in
`ScanDeclared`: the advice is the trace's values, and `F`'s committed gates are held at them. -/
theorem scanDeclared_of {S : ℕ} {C : Prog} {P : Partition} {d : Design} {lay : Layout 1 S} {k : ℕ} {ok : List ℕ}
    {links : List (ℕ × ℕ)} {δ : ℝ} (hT : C.Topological) (hP : P.Partitions C)
    (hin : ∀ t < k, ∀ g ∈ lay.ingress t, C.gates[g]? = some .input) {ref : ℕ → Option Bool} {F : Finset ℕ}
    (ι : ℕ → ℕ → Bool) (hι : ∀ v r, P.rus[v]? = some r → v ∉ F → ∀ u ∈ r, C.Correct (RU.ref C r ref (ι v)) u)
    (hF : F ∈ faultySets d P δ) (hcomm : ChecksCommitted C ok links) {ops : List (Opening Bool)}
    (hval : ∀ g, (∃ q ∈ ops, q.pos = g) → (∀ b, C.gates[g]? = some (.const b) → tapVal ops g = b) →
      traceVal C ref g = tapVal ops g)
    {ins : List (List Bool)} (hlen : ins.length = k)
    (hrows : ∀ t < k, ∀ j g, (lay.ingress t)[j]? = some g →
      (∃ q ∈ ops, q.pos = g) ∧ tapVal ops g = (ins.getD t []).getD j false)
    (hok : ∀ g ∈ ok, (∀ b, C.gates[g]? = some (.const b) → b = true) ∧ (∃ q ∈ ops, q.pos = g) ∧
      tapVal ops g = true)
    (hchain : ∀ e ∈ links, (∃ q ∈ ops, q.pos = e.1) ∧ (∃ q ∈ ops, q.pos = e.2) ∧ tapVal ops e.1 = tapVal ops e.2 ∧
      ∀ b, C.gates[e.2]? = some (.const b) → tapVal ops e.2 = b) :
    ins ∈ ScanDeclared C P d lay k ok links δ := by
  have hinp : lay.inputOf k ins (traceVal C ref) = traceVal C ref := funext (inputOf_rows hlen fun t ht j g hg => by
    have hgi : C.gates[g]? = some .input := hin t ht g (List.mem_of_getElem? hg)
    rw [hval g (hrows t ht j g hg).1 fun b hb => by rw [hgi] at hb; cases hb]
    exact (hrows t ht j g hg).2)
  have hout : ∀ g ∈ C.outs,
      valOver C (lay.inputOf k ins (traceVal C ref)) (heldAdv C P F ref) g = traceVal C ref g := by
    intro g hg
    rw [hinp]
    exact valOver_out hT hP ι hι (fun _ => True) (fun _ _ _ _ _ _ => trivial) (fun _ _ _ => rfl) hg trivial
  simp only [ScanDeclared, Set.mem_ofPred_eq]
  refine ⟨hlen, F, hF, traceVal C ref, heldAdv C P F ref, fun g b h => faultyGate_of_heldAdv h, fun e he => ?_,
    fun g hg => ?_⟩
  · obtain ⟨h1, h2, heq, hc⟩ := hchain e he
    have he1 : C.gates[e.1]? = some .input := (hcomm.2 e he).2
    rw [hout e.2 (hcomm.2 e he).1, hval e.1 h1 (fun b hb => by rw [he1] at hb; cases hb), hval e.2 h2 hc, heq]
  · obtain ⟨hc, hq, hv⟩ := hok g hg
    rw [hout g (hcomm.1 g hg), hval g hq (fun b hb => hv.trans (hc b hb).symm), hv]

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
      fill (w.val H D.commit st) (tapVal (ops st)) g = some b := fill_ext
  refine ⟨st, P, hs, hP, fun ins hio => ?_⟩
  obtain ⟨hpass, ⟨hlen, hrows⟩, hok, hchain⟩ := hio
  exact scanDeclared_of hsc.1 hsc.2.1 hfit.2.1 ι
    (fun v r hr hv u hu => correct_mono (ruRef_mono hext) (hι v r hr hv u hu)) hF (hcomm st hs)
    (fun g hq hc => traceVal_fill_of hc (hagree st hs hpass g hq)) hlen (fun t ht => (hrows t ht).2) hok hchain

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

/-- Window `i`'s coins split off the period's: its prover's tape (`tapeOf`) and its own coins. -/
def windowSplit (D : Deploy Coin Col T) {n : ℕ} (x : Fin n → Msg) (Ω : Type) (i : Fin n) :
    PeriodCoins D x Ω ≃ WindowTape D x Ω i × (Fin (D.V.k (x i)) → Coin) :=
  (Equiv.prodCongr (Equiv.refl Ω) (Equiv.piSplitAt i fun j => Fin (D.V.k (x j)) → Coin)).trans
    ((Equiv.prodCongr (Equiv.refl Ω) (Equiv.prodComm _ _)).trans (Equiv.prodAssoc _ _ _).symm)

/-- **Window `i`'s term**: over the period's coins, window `i`'s sampled verifier accepts with its prover at its tape
while its ingress isn't one the scan accepts with probability at most its `δ + ε_ks + ε_bind + Pr[B_tap]`. -/
theorem window_term_le (H : Msg → Msg) (D : Deploy Coin Col T) (hks : KS H D) (hb : D.commit.Binds H) (s : Schedule)
    (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (ops : Stmt → List (Opening Bool))
    {n : ℕ} (x : Fin n → Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : (i : Fin n) → Prover (WindowTape D x Ω i))
    (i : Fin n) (hx : ScanScope D s (x i)) :
    Pr (fun ω : PeriodCoins D x Ω => (run H (P i) D.V (x i) (tapeOf ω i) (ω.2 i)).out = .acc ∧
        ¬ IngressDeclared H D s ops δ (x i)) ≤
      κ D δ (x i) + Real.sqrt (γ D (x i) * PrB H D.V (P i) (x i)) +
        (Bext H D.V (E D decode) (WindowTape D x Ω i) (P i) (x i)).Pr H +
        (Btap H D.V (E D decode) D.commit D.parse ops (WindowTape D x Ω i) (P i) (x i)).Pr H := by
  have key : ∀ q : WindowTape D x Ω i → (Fin (D.V.k (x i)) → Coin) → Prop,
      Pr (fun ω : PeriodCoins D x Ω => q (tapeOf ω i) (ω.2 i)) =
        Pr (fun u : WindowTape D x Ω i × (Fin (D.V.k (x i)) → Coin) × (E D decode).C (x i) => q u.1 u.2.1) :=
    fun q => (NetTiming.Pr_equiv (windowSplit D x Ω i) fun u => q u.1 u.2).trans (Claims.Pr_drop q).symm
  exact (key fun t r => (run H (P i) D.V (x i) t r).out = .acc ∧ ¬ IngressDeclared H D s ops δ (x i)).trans_le
    (ingress_sound H D hks hb s δ decode hδ hdec ops (x i) hx (WindowTape D x Ω i) (P i))

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
    (sc : Scan) (hform : p.form = .rows s sd sc) (ops : Stmt → List (Opening Bool))
    (c : NetworkCheck.Period.Commitment) (x : Fin p.windows → Msg) (hx : ∀ i, ScanScope D s (x i)) (Ω : Type)
    [Fintype Ω] [Nonempty Ω] (P : (i : Fin p.windows) → Prover (WindowTape D x Ω i))
    (ev : PeriodCoins D x Ω → Evidence) (hrec : AnswersOfRecord H D p s sd c x P ev) :
    PeriodSound check H D s δ decode ops p c x Ω P ev := by
  unfold PeriodSound periodErr
  calc Pr (fun ω : PeriodCoins D x Ω =>
        PeriodAccepts (check p c (ev ω)) ∧ ∃ i, ¬ IngressDeclared H D s ops δ (x i))
      ≤ Pr (fun ω : PeriodCoins D x Ω => ∃ i ∈ (Finset.univ : Finset (Fin p.windows)),
          (run H (P i) D.V (x i) (tapeOf ω i) (ω.2 i)).out = .acc ∧ ¬ IngressDeclared H D s ops δ (x i)) :=
        Claims.Pr_mono fun ω ⟨hacc, i, hi⟩ =>
          ⟨i, Finset.mem_univ _, hrec ω i (windowAnswered_of_accept hform hacc i.val i.isLt), hi⟩
    _ ≤ ∑ i, Pr (fun ω : PeriodCoins D x Ω =>
          (run H (P i) D.V (x i) (tapeOf ω i) (ω.2 i)).out = .acc ∧ ¬ IngressDeclared H D s ops δ (x i)) :=
        Claims.Sampling.Pr_exists_le Finset.univ fun i (ω : PeriodCoins D x Ω) =>
          (run H (P i) D.V (x i) (tapeOf ω i) (ω.2 i)).out = .acc ∧ ¬ IngressDeclared H D s ops δ (x i)
    _ ≤ _ := Finset.sum_le_sum fun i _ => window_term_le H D hks hb s δ decode hδ hdec ops x Ω P i (hx i)

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
  · intro c x hx Ω _ _ P ev hrec
    exact periodSound_check H D hks hb s δ decode hδ hdec p sd sc hform ops c x hx Ω P ev hrec
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
  intro chk hall Coin Col T _ _ _ _ _ H D s δ decode ops p c x Ω _ _ P ev hbad herr hsound
  have hsure : Pr (fun ω : PeriodCoins D x Ω =>
      PeriodAccepts (chk p c (ev ω)) ∧ ∃ i, ¬ IngressDeclared H D s ops δ (x i)) = 1 :=
    (Claims.Pr_congr fun ω => ⟨fun _ => trivial, fun _ => ⟨hall p c (ev ω), hbad⟩⟩).trans Claims.Sampling.Pr_true
  unfold PeriodSound at hsound
  linarith

end Properties.NetworkCertifier
