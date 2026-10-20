import Proofs.Flock.Claims.Recursion.BoundRouteSet
import Proofs.Flock.Claims.Recursion.LastColsDeployed

/-!
# (P4) with the calls compared as sets, at the deployed bound record (draft)

`Bound.LastBindsECSet` (`BoundRouteSet`) at the deployed bound record, by `LastBindsDeployed`'s route with every
comparison of calls made between sets:

* `Bound.lastBindsECSet_of_lastStep`: at any bound record, `LastBindsECSet` from `Binds`, `LastStepFree` and
  `Bound.LastStepBindsSet`, the element-level verifier's binding of its last message with its last step's calls
  compared as sets. Where the openings agree, `IVStar`'s calls are the opening checks' then `agrees`' (`OC.eval` appends
  logs through `bindOC`), so `DE`'s last-step calls being one set makes `IVStar`'s one set (`Bound.ivstar_evalS_snoc`).
* `RecDeploy.lastBindsECSet_deployed_of`: at the deployed bound record, from `RecDeploy.CanonicalProofFiles` and the
  calls half with the calls compared as sets (`RecDeploy.PlainCallsBindSetOrSlack`); the columns half is
  `RecDeploy.plainColsBind_sha512`.

`PlainCallsBindSetOrSlack` is open. Its core is two reps accepted on one stream: `rep_frames_bind` (`RepFramesBind`)
gives them one rep's frames, and the openings and the fields no frame carries remain
(`note:proofs/20261020T0132Z-finding-rec2-calls-as-lists`).
-/

namespace Claims.Recursion

open Claims

/-- **Two oracle programs alike but for their calls' order and multiplicity**: one result, one set of calls. -/
def OCSet (H : Msg → Msg) {α : Type} (p p' : OC α) : Prop :=
  (p.eval H).1 = (p'.eval H).1 ∧ (p.eval H).2.toFinset = (p'.eval H).2.toFinset

theorem ocSet_of_eval (H : Msg → Msg) {α : Type} {p p' : OC α} (h : p.eval H = p'.eval H) : OCSet H p p' :=
  ⟨by rw [h], by rw [h]⟩

theorem ocSet_bindOC (H : Msg → Msg) {α β : Type} {p p' : OC α} {f f' : α → OC β} (hp : OCSet H p p')
    (hf : ∀ a, OCSet H (f a) (f' a)) : OCSet H (bindOC p f) (bindOC p' f') := by
  unfold OCSet at hp ⊢
  rw [eval_bindOC_full, eval_bindOC_full, hp.1]
  exact ⟨(hf _).1, by rw [List.toFinset_append, List.toFinset_append, hp.2, (hf _).2]⟩

theorem ocSet_allOC_map (H : Msg → Msg) {ι : Type} (f g : ι → OC Bool) :
    ∀ l : List ι, (∀ j ∈ l, OCSet H (f j) (g j)) → OCSet H (allOC (l.map f)) (allOC (l.map g))
  | [], _ => ⟨rfl, rfl⟩
  | j :: l, h => by
    simp only [List.map_cons, allOC]
    refine ocSet_bindOC H (h j List.mem_cons_self) fun b => ?_
    cases b
    · exact ⟨rfl, rfl⟩
    · exact ocSet_allOC_map H f g l fun i hi => h i (List.mem_cons_of_mem _ hi)

variable {Coin Col T : Type} [Inhabited Coin]

/-- **(P4) at the element level, the calls compared as sets**: `Bound.LastStepBinds` where the last step's calls count
as different only if some call of one is not a call of the other. -/
def Bound.LastStepBindsSet (H : Msg → Msg) (Bd : Bound Coin Col T) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = Bd.DE.V.k x →
    cs.length = ms.length + 1 →
    (∀ j < ms.length, ((Bd.DE.V.next x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (Bd.ans (cs.getD j default))) →
    ((Bd.DE.V.next x (ms ++ [l]) cs).eval H).1 = .acc → ((Bd.DE.V.next x (ms ++ [l']) cs).eval H).1 = .acc →
    (Bd.DE.cols (Bd.lastView (ms ++ [l]) cs) ≠ Bd.DE.cols (Bd.lastView (ms ++ [l']) cs) ∨
      ((Bd.DE.V.next x (ms ++ [l]) cs).eval H).2.toFinset ≠ ((Bd.DE.V.next x (ms ++ [l']) cs).eval H).2.toFinset) →
    ∃ q ∈ (((Bd.DE.V.next x (ms ++ [l]) cs).eval H).2 ++ ((Bd.DE.V.next x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((Bd.DE.V.next x (ms ++ [l]) cs).eval H).2 ++
        ((Bd.DE.V.next x (ms ++ [l']) cs).eval H).2).map Prod.fst, Collision H q q'

/-- `agrees` on messages that share every round before the last reads the last only in its last step, up to the calls'
order. -/
theorem Bound.agrees_evalS_snoc (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) {ms : List Msg} {l l' : Msg}
    {cs : List Coin}
    (h : OCSet H (Bd.DE.V.next x (ms ++ [l']) (cs.take (ms.length + 1)))
      (Bd.DE.V.next x (ms ++ [l]) (cs.take (ms.length + 1)))) :
    OCSet H (Bd.agrees x (ms ++ [l']) cs) (Bd.agrees x (ms ++ [l]) cs) := by
  unfold Bound.agrees
  simp only [List.length_append, List.length_singleton]
  refine ocSet_allOC_map H _ _ _ fun j hj => ?_
  rw [List.mem_range] at hj
  by_cases hj' : j < ms.length
  · rw [List.take_append_of_le_length (show j + 1 ≤ ms.length by omega),
      List.take_append_of_le_length (show j + 1 ≤ ms.length by omega)]
    exact ⟨rfl, rfl⟩
  · obtain rfl : j = ms.length := by omega
    rw [List.take_of_length_le (show (ms ++ [l']).length ≤ ms.length + 1 by simp),
      List.take_of_length_le (show (ms ++ [l]).length ≤ ms.length + 1 by simp)]
    exact ocSet_bindOC H h fun _ => ⟨rfl, rfl⟩

/-- Two last messages `final` reads to the same openings give one `IVStar` decision with one set of calls where the
element-level verifier's acceptance of the opened messages is one, with one set of calls. -/
theorem Bound.ivstar_evalS_snoc (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) {pre : List Msg} {a a' : Msg}
    {cs : List Coin} {os : List (List (Msg × (Fin Bd.saltLen → UInt8)))} {last last' : Msg}
    (hk : pre.length + 1 = Bd.DE.V.k x) (hfin : Bd.final a = some (os, last)) (hfin' : Bd.final a' = some (os, last'))
    (hag : OCSet H (Bd.agrees x (Bd.opened os last') cs) (Bd.agrees x (Bd.opened os last) cs)) :
    OCSet H (Bd.IVStar.next x (pre ++ [a']) cs) (Bd.IVStar.next x (pre ++ [a]) cs) := by
  rw [Bd.ivstar_last x pre a' cs hk hfin', Bd.ivstar_last x pre a cs hk hfin]
  by_cases hc : os.length + 1 = pre.length + 1 ∧ ∀ j < os.length, (os.getD j []).length = Bd.nAt x j
  · rw [ite_eq_left hc, ite_eq_left hc]
    refine ocSet_bindOC H ⟨rfl, rfl⟩ fun ok => ?_
    cases ok
    · exact ⟨rfl, rfl⟩
    · exact ocSet_bindOC H hag fun _ => ⟨rfl, rfl⟩
  · rw [ite_eq_right hc, ite_eq_right hc]
    exact ⟨rfl, rfl⟩

/-- **(P4) modulo what `E*` reads with the calls as sets, from the element-level verifier's binding of its last
message with its last step's calls as sets**: `Bound.lastBindsEC_of_lastStep` at sets. -/
theorem Bound.lastBindsECSet_of_lastStep {H : Msg → Msg} {Bd : Bound Coin Col T} (hB : Bd.Binds)
    (hF : Bd.LastStepFree H) (hS : Bd.LastStepBindsSet H) : Bd.LastBindsECSet H := by
  intro x as as' cs cs' hk hl hc hc' hd hcd hne ha ha'
  have hev : (Bd.IVStar.next x as' cs').eval H = (Bd.IVStar.next x as' cs).eval H :=
    Bound.lastCoinFree_of_step hF x as' cs' cs (hl.trans hk) (hc'.trans hl.symm) (hc.trans hl.symm) hcd.symm
  have hvw : Bd.viewAt H x as' cs' = Bd.viewAt H x as' cs := by
    unfold Bound.viewAt
    rw [hev, ← hcd]
  rw [hvw, hev] at hne
  rw [hev] at ha' ⊢
  rcases as with _ | ⟨a₀, as₀⟩
  · cases as' with
    | cons _ _ => simp at hl
    | nil => exact (hne.elim (fun h => h rfl) fun h => h rfl).elim
  obtain ⟨pre, a, has⟩ : ∃ pre a, a₀ :: as₀ = pre ++ [a] :=
    ⟨(a₀ :: as₀).dropLast, (a₀ :: as₀).getLast (List.cons_ne_nil _ _),
      (List.dropLast_concat_getLast (List.cons_ne_nil _ _)).symm⟩
  rw [has] at hk hl hc hd hne ha ⊢
  obtain ⟨pre', a', rfl⟩ : ∃ pre' a', as' = pre' ++ [a'] := by
    cases as' with
    | nil => simp at hl
    | cons b bs =>
      exact ⟨(b :: bs).dropLast, (b :: bs).getLast (List.cons_ne_nil _ _),
        (List.dropLast_concat_getLast (List.cons_ne_nil _ _)).symm⟩
  rw [List.dropLast_concat, List.dropLast_concat] at hd
  subst hd
  have hpk : pre.length + 1 = Bd.DE.V.k x := by simpa using hk
  have hpc : cs.length = pre.length + 1 := by simpa using hc
  have hlt : ¬ (pre ++ [a]).length < Bd.DE.V.k x := by
    simp only [List.length_append, List.length_singleton]
    omega
  have hlt' : ¬ (pre ++ [a']).length < Bd.DE.V.k x := by
    simp only [List.length_append, List.length_singleton]
    omega
  obtain ⟨os, last, hfin, hlen, -, hok, hokc, hag, hagc⟩ := Bd.ivstar_acc H x (pre ++ [a]) cs hlt ha
  obtain ⟨os', last', hfin', hlen', -, hok', hokc', hag', hagc'⟩ := Bd.ivstar_acc H x (pre ++ [a']) cs hlt' ha'
  rw [List.getLastD_concat] at hfin
  rw [List.getLastD_concat] at hfin'
  rw [List.dropLast_concat] at hok
  rw [List.dropLast_concat] at hokc
  rw [List.dropLast_concat] at hok'
  rw [List.dropLast_concat] at hokc'
  have hlo : os.length = pre.length := by
    simp only [List.length_append, List.length_singleton] at hlen
    omega
  have hlo' : os'.length = pre.length := by
    simp only [List.length_append, List.length_singleton] at hlen'
    omega
  by_cases hos : os = os'
  · subst hos
    obtain ⟨P, hP⟩ : ∃ P, P = os.map (fun o => Bd.join (o.map Prod.fst)) := ⟨_, rfl⟩
    have hop : ∀ l, Bd.opened os l = P ++ [l] := fun l => by
      rw [hP]
      rfl
    have hPl : P.length = pre.length := by
      rw [hP, List.length_map]
      exact hlo
    have hcs : cs.take (P.length + 1) = cs := List.take_of_length_le (by omega)
    rw [hop last] at hag hagc
    rw [hop last'] at hag' hagc'
    have hstepA : ∀ {l : Msg}, ((Bd.agrees x (P ++ [l]) cs).eval H).1 = true →
        ((Bd.DE.V.next x (P ++ [l]) cs).eval H).1 = .acc ∧
          ∀ e ∈ ((Bd.DE.V.next x (P ++ [l]) cs).eval H).2, e ∈ ((Bd.agrees x (P ++ [l]) cs).eval H).2 := by
      intro l h
      have h1 := Bd.agrees_true H x h P.length (by simp)
      rw [List.take_of_length_le (show (P ++ [l]).length ≤ P.length + 1 by simp), hcs,
        ite_eq_right (show ¬ P.length + 1 < (P ++ [l]).length by simp)] at h1
      exact h1
    have hearly : ∀ j < P.length, ((Bd.DE.V.next x (P.take (j + 1)) (cs.take (j + 1))).eval H).1 =
        .msg (Bd.ans (cs.getD j default)) := by
      intro j hj
      have h1 := (Bd.agrees_true H x hag j (by simp only [List.length_append, List.length_singleton]; omega)).1
      rw [List.take_append_of_le_length (show j + 1 ≤ P.length by omega),
        ite_eq_left (show j + 1 < (P ++ [last]).length by simp only [List.length_append, List.length_singleton]; omega)]
        at h1
      exact h1
    have hv := Bd.openView_viewAt H x hpc hfin ha
    have hv' := Bd.openView_viewAt H x hpc hfin' ha'
    rw [hv, hv', hop last, hop last'] at hne
    have hdiff : Bd.DE.cols (Bd.lastView (P ++ [last]) cs) ≠ Bd.DE.cols (Bd.lastView (P ++ [last']) cs) ∨
        ((Bd.DE.V.next x (P ++ [last]) cs).eval H).2.toFinset ≠
          ((Bd.DE.V.next x (P ++ [last']) cs).eval H).2.toFinset := by
      refine hne.imp_right fun h heq => h ?_
      have hA : OCSet H (Bd.agrees x (Bd.opened os last') cs) (Bd.agrees x (Bd.opened os last) cs) := by
        rw [hop last, hop last']
        exact Bd.agrees_evalS_snoc H x (by
          rw [hcs]
          exact ⟨(hstepA hag').1.trans (hstepA hag).1.symm, heq.symm⟩)
      exact (Bd.ivstar_evalS_snoc H x hpk hfin hfin' hA).2.symm
    obtain ⟨q, hq, q', hq', hcol⟩ :=
      hS x P last last' cs (by omega) (by omega) hearly (hstepA hag).1 (hstepA hag').1 hdiff
    have hsub : ∀ e ∈ ((Bd.DE.V.next x (P ++ [last]) cs).eval H).2,
        e ∈ ((Bd.IVStar.next x (pre ++ [a]) cs).eval H).2 :=
      fun e he => hagc e ((hstepA hag).2 e he)
    have hsub' : ∀ e ∈ ((Bd.DE.V.next x (P ++ [last']) cs).eval H).2,
        e ∈ ((Bd.IVStar.next x (pre ++ [a']) cs).eval H).2 :=
      fun e he => hagc' e ((hstepA hag').2 e he)
    exact ⟨q, mem_fst_append_mono hsub hsub' hq, q', mem_fst_append_mono hsub hsub' hq', hcol⟩
  · obtain ⟨j, hj, hjne⟩ : ∃ j, ∃ hj : j < os.length, os[j] ≠ os'[j]'(by omega) := by
      by_contra hcon
      push Not at hcon
      exact hos (List.ext_getElem (by omega) fun j h1 _ => hcon j h1)
    have hjp : j < pre.length := by omega
    have hj' : j < os'.length := by omega
    have hmem : Bd.opensRound pre[j] os[j] ∈ (pre.zip os).map fun p => Bd.opensRound p.1 p.2 :=
      List.mem_map.2 ⟨(pre[j], os[j]), List.mem_iff_getElem.2
        ⟨j, by simp only [List.length_zip, lt_min_iff]; exact ⟨hjp, hj⟩, by simp⟩, rfl⟩
    have hmem' : Bd.opensRound pre[j] os'[j] ∈ (pre.zip os').map fun p => Bd.opensRound p.1 p.2 :=
      List.mem_map.2 ⟨(pre[j], os'[j]), List.mem_iff_getElem.2
        ⟨j, by simp only [List.length_zip, lt_min_iff]; exact ⟨hjp, hj'⟩, by simp⟩, rfl⟩
    obtain ⟨hr1, hr2⟩ := allOC_true H _ hok _ hmem
    obtain ⟨hr1', hr2'⟩ := allOC_true H _ hok' _ hmem'
    obtain ⟨hal, hro⟩ := Bd.opensRound_true H hr1
    obtain ⟨hal', hro'⟩ := Bd.opensRound_true H hr1'
    have hlj : os[j].length = os'[j].length := by omega
    obtain ⟨i, hi, hine⟩ : ∃ i, ∃ hi : i < os[j].length, os[j][i] ≠ os'[j][i]'(by omega) := by
      by_contra hcon
      push Not at hcon
      exact hjne (List.ext_getElem hlj fun i h1 _ => hcon i h1)
    obtain ⟨ho1, ho2⟩ := hro i hi
    obtain ⟨ho1', ho2'⟩ := hro' i (by omega)
    obtain ⟨u, hu, v, hv, hcol⟩ := Bd.opens_coll H hB ho1 ho1'
      (fun h => hine (Prod.ext (congrArg Prod.fst h) (congrArg Prod.snd h)))
    obtain ⟨eu, heu, rfl⟩ := List.mem_map.1 hu
    obtain ⟨ev, hev', rfl⟩ := List.mem_map.1 hv
    exact ⟨eu.1, List.mem_map.2 ⟨eu, List.mem_append_left _ (hokc eu (hr2 eu (ho2 eu heu))), rfl⟩,
      ev.1, List.mem_map.2 ⟨ev, List.mem_append_right _ (hokc' ev (hr2' ev (ho2' ev hev'))), rfl⟩, hcol⟩

end Claims.Recursion

namespace RecDeploy

open Claims (Msg Out OC)
open Claims.Recursion (Bound)
open FlockDeploy (Coin)

/-- **The deployed plain verifier binds its last message, the calls compared as sets**: `PlainLastBinds` where two
decisions' calls count as different only if some call of one is not a call of the other. -/
def PlainLastBindsSet (H : Msg → Msg) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = (FlockDeploy.V false).k x →
    cs.length = ms.length + 1 →
    (∀ j < ms.length, ((FlockDeploy.next false x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (cs.getD j default).bytes) →
    ((FlockDeploy.next false x (ms ++ [l]) cs).eval H).1 = .acc →
    ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).1 = .acc →
    (FlockDeploy.cols false (plainView (ms ++ [l]) cs) 0 ≠ FlockDeploy.cols false (plainView (ms ++ [l']) cs) 0 ∨
      ((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2.toFinset ≠
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2.toFinset) →
    ∃ q ∈ (((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2).map Prod.fst, Claims.Collision H q q'

/-- **The deployed plain verifier binds its last message's calls as a set, up to the decoder's slack**:
`PlainCallsBindOrSlack` with the calls compared as sets. Open: its core is `rep_frames_bind`, with the openings and the
fields no frame carries still to bind. -/
def PlainCallsBindSetOrSlack (H : Msg → Msg) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = (FlockDeploy.V false).k x →
    cs.length = ms.length + 1 →
    (∀ j < ms.length, ((FlockDeploy.next false x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (cs.getD j default).bytes) →
    ((FlockDeploy.next false x (ms ++ [l]) cs).eval H).1 = .acc →
    ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).1 = .acc →
    ((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2.toFinset ≠
      ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2.toFinset →
    (∃ q ∈ (((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2).map Prod.fst, Claims.Collision H q q') ∨
    ProofSlack l l'

/-- **`PlainLastBindsSet` from the columns half, the calls half as sets and a canonical decoder.** -/
theorem plainLastBindsSet_of {H : Msg → Msg} (hC : CanonicalProofFiles) (hP : PlainColsBind H)
    (hK : PlainCallsBindSetOrSlack H) : PlainLastBindsSet H := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  rcases hne with hne | hne
  · exact hP x ms l l' cs hk hc hacc hacc' hne
  · rcases hK x ms l l' cs hk hc hearly hacc hacc' hne with hcol | ⟨_, _, _, b, b', -, -, -, -, hbb, sh, p, hd, hd'⟩
    · exact hcol
    · exact absurd (hC sh b b' p hd hd') hbb

/-- **`LastStepBindsSet` at every bound record whose `DE` steps as the deployed plain verifier**, answers a round
with its coin's bytes and reads a view's columns only through table 0's, from `PlainLastBindsSet`. -/
theorem lastStepBindsSet_of_V {T : Type} (H : Msg → Msg) (Bd : Bound Coin (List FlockLevel3.GF128) T)
    (hV : Bd.DE.V = FlockDeploy.V false) (hans : ∀ c, Bd.ans c = c.bytes)
    (hcols : ∀ v v' : Claims.Seen, FlockDeploy.cols false v 0 = FlockDeploy.cols false v' 0 →
      Bd.DE.cols v = Bd.DE.cols v') (h : PlainLastBindsSet H) : Bd.LastStepBindsSet H := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  have hview : ∀ ms' : List Msg, Bd.lastView ms' cs = plainView ms' cs := fun ms' => by
    unfold Bound.lastView plainView
    simp only [hans]
  rw [hV] at hk hacc hacc' hne ⊢
  simp only [hV, hans] at hearly
  rw [hview, hview] at hne
  exact h x ms l l' cs hk hc hearly hacc hacc' (hne.imp_left fun hc0 heq => hc0 (hcols _ _ heq))

/-- **(P4) modulo what `E*` reads with the calls as sets, at the deployed bound record**, from the deployed plain
verifier's binding of its last message with its calls as sets. -/
theorem lastBindsECSet_deployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (H : Msg → Msg) (h : PlainLastBindsSet H) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final
      w).LastBindsECSet H :=
  Bound.lastBindsECSet_of_lastStep (binds_deployed rel _ _ wE join final w) (lastStepFree_of_V H _ rfl)
    (lastStepBindsSet_of_V H _ rfl (bdDeployed_ans rel _ _ wE join rowInDeployed final w) (fun _ _ heq => by
      rw [bdDeployed_cols]
      exact plainTrim_cols_congr rel heq) h)

/-- **(P4) with the calls as sets at the deployed bound record, at SHA-512**, from the decoder's canonicity and the
calls half as sets; the columns half is `plainColsBind_sha512`. -/
theorem lastBindsECSet_deployed_of (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hC : CanonicalProofFiles) (hK : PlainCallsBindSetOrSlack FlockDeploy.sha512) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final
      w).LastBindsECSet FlockDeploy.sha512 :=
  lastBindsECSet_deployed rel wE join final w FlockDeploy.sha512 (plainLastBindsSet_of hC plainColsBind_sha512 hK)

end RecDeploy
