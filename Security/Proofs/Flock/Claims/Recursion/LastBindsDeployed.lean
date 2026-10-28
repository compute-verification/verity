import Proofs.Flock.Claims.Recursion.BoundBridge
import Proofs.Flock.Claims.Recursion.LastBindsEC

/-!
# (P4) at the deployed bound record

`Bound.LastBindsEC`, (P4) modulo what `E*` reads, at the deployed bound record: `RecDeploy.bdDeployed`
at the deployed plain verifier's steps and columns and the deployed rows' digest input. It comes down to the deployed
plain verifier binding its own last message.

* `Bound.lastBindsEC_of_lastStep`: at any bound record, `LastBindsEC` from `Binds`, `LastStepFree` and
  `Bound.LastStepBinds`, the element-level verifier's binding of its last message. Where the two decisions' openings
  differ, some earlier round's commit string opens two ways, a collision by `Binds` (`Bound.opens_coll`). Where they
  agree, the opened messages share every round before the last, so the two decisions differ in `DE`'s columns of the
  opened view or in the set of `DE`'s last step's calls, which `LastStepBinds` charges to a collision. `IVStar`'s
  calls there are the opening checks' then `agrees`', so `DE`'s last-step calls being one set makes `IVStar`'s one set
  (`OCSet`, `Bound.ivstar_eval_snoc`).
* `RecDeploy.lastBindsEC_deployed`: at the deployed bound record, from `RecDeploy.PlainLastBinds`, `LastStepBinds` read
  at the deployed plain verifier. Two last messages it accepts after the same rounds and coins, that differ in table
  0's opened columns or in the set of the decision's calls, show a collision among those calls.
* `RecDeploy.CanonicalProofFiles`, what (P4) needs of V*'s proof-file decoder: `decodeProof` accepts a file only if it
  is the one encoding of what it decodes to, so two files it decodes alike are one file. `PlainLastBinds` follows from
  it and `RecDeploy.PlainLastBindsOrSlack` (`RecDeploy.plainLastBinds_of_canonical`), which allows, beside a collision,
  two proof files at one position that differ in their bytes and decode alike (`RecDeploy.ProofSlack`).

**Where the decoder stands.** The verifier hashes each proof file whole and compares the digest with the record's
`proof_sha` (S17), and no custody check pins `proof_sha`, so two files with one decoding would give two decisions with
different calls that need not collide. `decodeProof` (`verity/core/service/flock/Flock/Proof.lean`) accepts a file
only if it is `Flock.Wire.proofFile` of what it decodes to, so there are no such two files
(`Flock.decodeProof_canonical`): `CanonicalProofFiles` holds (`RecDeploy.canonicalProofFiles`), and (P4) at the
deployed bound record holds at SHA-512 with no premise (`RecDeploy.lastBindsEC_deployed_sha512_closed`,
`LastBindsClosed`).

**Where the record's text stands.** The verifier reads the record's text through `Json.parse`, and S0 accepts one text
per parse (`LastGap.checks_record_eq`), so of two records that parse alike at most one is accepted: the trailing-space
pair that refuted `LastBinds` before S0 is refused (`LastGap.check_space`). Two records that
parse differently can still give the same accepted decision, with calls that are the same set but not the same list:
`Record.decodeWith` hashes the rounds in the record's own stream order, and the digest of a table's empty publics only
when the record lists it. So the calls are compared as sets throughout: with them compared as lists,
`PlainLastBinds` is false at this record decoder, whatever `decodeProof` does
(`note:proofs/20261020T0132Z-finding-rec2-calls-as-lists`).

`PlainLastBindsOrSlack` is the collision charge through the plain verifier's steps. `LastColsDeployed` splits it into
its columns half, proved at SHA-512 (`RecDeploy.plainColsBind_sha512`), and its calls half
(`RecDeploy.PlainCallsBindOrSlack`), proved at SHA-512 too (`RecDeploy.plainCallsBindOrSlack_sha512`).
-/

namespace Claims.Recursion

open Claims

variable {Coin Col T : Type} [Inhabited Coin]

theorem mem_fst_append_mono {A B C D : List (Msg × Msg)} (h1 : ∀ e ∈ A, e ∈ C) (h2 : ∀ e ∈ B, e ∈ D) {q : Msg}
    (hq : q ∈ (A ++ B).map Prod.fst) : q ∈ (C ++ D).map Prod.fst := by
  obtain ⟨e, he, rfl⟩ := List.mem_map.1 hq
  rcases List.mem_append.1 he with he | he
  · exact List.mem_map.2 ⟨e, List.mem_append_left _ (h1 e he), rfl⟩
  · exact List.mem_map.2 ⟨e, List.mem_append_right _ (h2 e he), rfl⟩

/-- **Two oracle programs alike but for their calls' order and multiplicity**: one result, one set of calls. -/
def OCSet (H : Msg → Msg) {α : Type} (p p' : OC α) : Prop :=
  (p.eval H).1 = (p'.eval H).1 ∧ (p.eval H).2.toFinset = (p'.eval H).2.toFinset

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

/-- **The opened view of an accepted last round**: the element-level messages `ms`, each before the last answered by its
coin (`ans`), the last by acceptance. -/
def Bound.lastView (Bd : Bound Coin Col T) (ms : List Msg) (cs : List Coin) : Seen :=
  ⟨ms.zip (cs.dropLast.map (fun c => Out.msg (Bd.ans c)) ++ [Out.acc]), Out.acc⟩

/-- **(P4) at the element level**: two last messages that the element-level verifier accepts after the same rounds and
coins, where it answered each earlier round with that round's coin, and that differ in its columns of the opened view or
in the set of its last step's calls, show a SHA-512 collision among those calls. -/
def Bound.LastStepBinds (H : Msg → Msg) (Bd : Bound Coin Col T) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = Bd.DE.V.k x →
    cs.length = ms.length + 1 →
    (∀ j < ms.length, ((Bd.DE.V.next x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (Bd.ans (cs.getD j default))) →
    ((Bd.DE.V.next x (ms ++ [l]) cs).eval H).1 = .acc → ((Bd.DE.V.next x (ms ++ [l']) cs).eval H).1 = .acc →
    (Bd.DE.cols (Bd.lastView (ms ++ [l]) cs) ≠ Bd.DE.cols (Bd.lastView (ms ++ [l']) cs) ∨
      ((Bd.DE.V.next x (ms ++ [l]) cs).eval H).2.toFinset ≠
        ((Bd.DE.V.next x (ms ++ [l']) cs).eval H).2.toFinset) →
    ∃ q ∈ (((Bd.DE.V.next x (ms ++ [l]) cs).eval H).2 ++ ((Bd.DE.V.next x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((Bd.DE.V.next x (ms ++ [l]) cs).eval H).2 ++
        ((Bd.DE.V.next x (ms ++ [l']) cs).eval H).2).map Prod.fst, Collision H q q'

/-- `agrees` on messages that share every round before the last reads the last only in its last step, up to the calls'
order. -/
theorem Bound.agrees_eval_snoc (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) {ms : List Msg} {l l' : Msg}
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

/-- `IVStar` at its last round, where `final` reads the last message. -/
theorem Bound.ivstar_last (Bd : Bound Coin Col T) (x : Msg) (pre : List Msg) (a : Msg) (cs : List Coin)
    {os : List (List (Msg × (Fin Bd.saltLen → UInt8)))} {last : Msg} (hk : pre.length + 1 = Bd.DE.V.k x)
    (hfin : Bd.final a = some (os, last)) :
    Bd.IVStar.next x (pre ++ [a]) cs =
      if os.length + 1 = pre.length + 1 ∧ ∀ j < os.length, (os.getD j []).length = Bd.nAt x j then
        bindOC (allOC ((pre.zip os).map fun p => Bd.opensRound p.1 p.2)) fun ok =>
          if ok then bindOC (Bd.agrees x (Bd.opened os last) cs) fun acc => .ret (if acc then .acc else .rej)
          else .ret .rej
      else .ret .rej := by
  have hlt : ¬ pre.length + 1 < Bd.DE.V.k x := by omega
  simp only [Bound.IVStar, hlt, ↓reduceIte, List.getLastD_concat, hfin, List.dropLast_concat,
    List.length_append, List.length_singleton]

/-- Two last messages `final` reads to the same openings give one `IVStar` decision with one set of calls where the
element-level verifier's acceptance of the opened messages is one, with one set of calls. -/
theorem Bound.ivstar_eval_snoc (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) {pre : List Msg} {a a' : Msg}
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

/-- The opened view of an accepted decision is the opened messages, answered as `lastView` answers them. -/
theorem Bound.openView_viewAt (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) {pre : List Msg} {a : Msg}
    {cs : List Coin} {os : List (List (Msg × (Fin Bd.saltLen → UInt8)))} {last : Msg}
    (hc : cs.length = pre.length + 1) (hfin : Bd.final a = some (os, last))
    (hacc : ((Bd.IVStar.next x (pre ++ [a]) cs).eval H).1 = .acc) :
    Bd.openView (Bd.viewAt H x (pre ++ [a]) cs) = Bd.lastView (Bd.opened os last) cs := by
  have hlen : pre.length = (cs.dropLast.map fun c => Out.msg (Bd.ans c)).length := by
    rw [List.length_map, List.length_dropLast, hc, Nat.add_sub_cancel]
  have hv : Bd.viewAt H x (pre ++ [a]) cs =
      ⟨pre.zip (cs.dropLast.map fun c => Out.msg (Bd.ans c)) ++ [(a, Out.acc)], Out.acc⟩ := by
    unfold Bound.viewAt
    rw [hacc, List.zip_append hlen]
    rfl
  have hs : (pre.zip (cs.dropLast.map fun c => Out.msg (Bd.ans c)) ++ [(a, Out.acc)]).map Prod.snd =
      cs.dropLast.map (fun c => Out.msg (Bd.ans c)) ++ [Out.acc] := by
    rw [List.map_append,
      List.map_snd_zip (show (cs.dropLast.map fun c => Out.msg (Bd.ans c)).length ≤ pre.length by omega)]
    rfl
  rw [hv]
  unfold Bound.openView Bound.lastView
  simp only [List.getLastD_concat, hfin, hs]

/-- **(P4) modulo what `E*` reads, from the element-level verifier's binding of its last message**: where `rowIn` is
injective (`Binds`) and `DE`'s last step reads no last coin (`LastStepFree`), `LastStepBinds` gives `LastBindsEC`. -/
theorem Bound.lastBindsEC_of_lastStep {H : Msg → Msg} {Bd : Bound Coin Col T} (hB : Bd.Binds)
    (hF : Bd.LastStepFree H) (hS : Bd.LastStepBinds H) : Bd.LastBindsEC H := by
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
        exact Bd.agrees_eval_snoc H x (by
          rw [hcs]
          exact ⟨(hstepA hag').1.trans (hstepA hag).1.symm, heq.symm⟩)
      exact (Bd.ivstar_eval_snoc H x hpk hfin hfin' hA).2.symm
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

/-- **The opened view of an accepted last round at the deployed plain verifier**: `Bound.lastView` at the deployed
bound record, whose answer to a round is its coin's bytes. -/
def plainView (ms : List Msg) (cs : List Coin) : Claims.Seen :=
  ⟨ms.zip (cs.dropLast.map (fun c => Out.msg c.bytes) ++ [Out.acc]), Out.acc⟩

/-- **The deployed plain verifier binds its last message**: `Bound.LastStepBinds` at its steps and its table-0 columns.
Two last messages it accepts after the same rounds and coins, where it answered each earlier round with that round's
coin, that differ in table 0's opened columns or in the set of the decision's calls, show a SHA-512 collision among
those calls. -/
def PlainLastBinds (H : Msg → Msg) : Prop :=
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

/-- The trimmed plain record's columns read a view only through table 0's opened columns. -/
theorem plainTrim_cols_congr (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    {v v' : Claims.Seen} (h : FlockDeploy.cols false v 0 = FlockDeploy.cols false v' 0) :
    (FlockDeploy.deployPlainTrim rel).cols v = (FlockDeploy.deployPlainTrim rel).cols v' := by
  funext j
  show ((FlockDeploy.plainTableCols v j).map fun p => (p.1, FlockDeploy.trimCol p.2)) =
    (FlockDeploy.plainTableCols v' j).map fun p => (p.1, FlockDeploy.trimCol p.2)
  unfold FlockDeploy.plainTableCols
  rw [h]

/-- **`LastStepBinds` at every bound record whose `DE` steps as the deployed plain verifier**, answers a round with its
coin's bytes and reads a view's columns only through table 0's, from `PlainLastBinds`. -/
theorem lastStepBinds_of_V {T : Type} (H : Msg → Msg) (Bd : Bound Coin (List FlockLevel3.GF128) T)
    (hV : Bd.DE.V = FlockDeploy.V false) (hans : ∀ c, Bd.ans c = c.bytes)
    (hcols : ∀ v v' : Claims.Seen, FlockDeploy.cols false v 0 = FlockDeploy.cols false v' 0 →
      Bd.DE.cols v = Bd.DE.cols v') (h : PlainLastBinds H) : Bd.LastStepBinds H := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  have hview : ∀ ms' : List Msg, Bd.lastView ms' cs = plainView ms' cs := fun ms' => by
    unfold Bound.lastView plainView
    simp only [hans]
  rw [hV] at hk hacc hacc' hne ⊢
  simp only [hV, hans] at hearly
  rw [hview, hview] at hne
  exact h x ms l l' cs hk hc hearly hacc hacc' (hne.imp_left fun hc0 heq => hc0 (hcols _ _ heq))

/-- The deployed bound record's `DE` reads a view's columns as `colsE`. Stated at a variable `colsE`, so that it is
read off the record without unfolding `colsE`'s own record. -/
theorem bdDeployed_cols (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → OC Out) (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128))
    (wE : Msg → ℝ) (join : List Msg → Msg) (rowIn : Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    (bdDeployed rel nextE colsE wE join rowIn final w).DE.cols = colsE :=
  rfl

/-- **`LastStepBinds` at the deployed bound record**, from `PlainLastBinds`, whatever the code side's other choices. -/
theorem lastStepBinds_deployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (wE : Msg → ℝ) (join : List Msg → Msg) (rowIn : Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (H : Msg → Msg) (h : PlainLastBinds H) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowIn final
      w).LastStepBinds H := by
  refine lastStepBinds_of_V H _ rfl (bdDeployed_ans rel _ _ wE join rowIn final w) (fun _ _ heq => ?_) h
  rw [bdDeployed_cols]
  exact plainTrim_cols_congr rel heq

/-- **(P4) modulo what `E*` reads, at the deployed bound record** (`Bound.LastBindsEC`), from the deployed plain
verifier's binding of its last message (`PlainLastBinds`): at the deployed plain verifier's steps and columns and the
deployed rows' digest input, whatever the code side's join, `final` and weights. -/
theorem lastBindsEC_deployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (H : Msg → Msg) (h : PlainLastBinds H) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final
      w).LastBindsEC H :=
  Bound.lastBindsEC_of_lastStep (binds_deployed rel _ _ wE join final w) (lastStepFree_of_V H _ rfl)
    (lastStepBinds_deployed rel wE join rowInDeployed final w H h)

/-- **What (P4) needs of V*'s proof-file decoder**: `decodeProof` accepts a file only if it is the one encoding of what
it decodes to, so two files it decodes alike at one shape are one file. -/
def CanonicalProofFiles : Prop :=
  ∀ (sh : _root_.Flock.Shape) (b b' : ByteArray) (p : _root_.Flock.Proof),
    _root_.Flock.decodeProof sh b = .ok p → _root_.Flock.decodeProof sh b' = .ok p → b = b'

/-- **Two last messages whose proof files decode alike**: at some position their proof files differ in their bytes, and
`decodeProof` reads both to one proof at some shape. -/
def ProofSlack (l l' : Msg) : Prop :=
  ∃ (f f' : FlockDeploy.Final) (i : ℕ) (b b' : ByteArray), FlockDeploy.finalOf l = some f ∧
    FlockDeploy.finalOf l' = some f' ∧ f.proofs[i]? = some b ∧ f'.proofs[i]? = some b' ∧ b ≠ b' ∧
    ∃ (sh : _root_.Flock.Shape) (p : _root_.Flock.Proof),
      _root_.Flock.decodeProof sh b = .ok p ∧ _root_.Flock.decodeProof sh b' = .ok p

/-- **`PlainLastBinds` up to the decoder's slack**: its hypotheses give a collision among the decision's calls, or two
proof files at one position that differ in their bytes and decode alike (`ProofSlack`). It is the collision charge
through the plain verifier's steps: its columns half and its calls half, with the calls compared as sets, hold at
SHA-512 (`LastColsDeployed`, `CallsOfProofs`). -/
def PlainLastBindsOrSlack (H : Msg → Msg) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = (FlockDeploy.V false).k x →
    cs.length = ms.length + 1 →
    (∀ j < ms.length, ((FlockDeploy.next false x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (cs.getD j default).bytes) →
    ((FlockDeploy.next false x (ms ++ [l]) cs).eval H).1 = .acc →
    ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).1 = .acc →
    (FlockDeploy.cols false (plainView (ms ++ [l]) cs) 0 ≠ FlockDeploy.cols false (plainView (ms ++ [l']) cs) 0 ∨
      ((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2.toFinset ≠
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2.toFinset) →
    (∃ q ∈ (((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2).map Prod.fst, Claims.Collision H q q') ∨
    ProofSlack l l'

/-- **A canonical decoder leaves no slack**: `PlainLastBinds` from `CanonicalProofFiles` and `PlainLastBindsOrSlack`. -/
theorem plainLastBinds_of_canonical {H : Msg → Msg} (hC : CanonicalProofFiles) (hT : PlainLastBindsOrSlack H) :
    PlainLastBinds H := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  rcases hT x ms l l' cs hk hc hearly hacc hacc' hne with hcol | ⟨_, _, _, b, b', -, -, -, -, hbb, sh, p, hd, hd'⟩
  · exact hcol
  · exact absurd (hC sh b b' p hd hd') hbb

/-- **(P4) modulo what `E*` reads at the deployed bound record, from the decoder's canonicity and the collision charge
through the plain verifier's steps.** -/
theorem lastBindsEC_deployed_of_canonical (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (H : Msg → Msg) (hC : CanonicalProofFiles) (hT : PlainLastBindsOrSlack H) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final
      w).LastBindsEC H :=
  lastBindsEC_deployed rel wE join final w H (plainLastBinds_of_canonical hC hT)

end RecDeploy
