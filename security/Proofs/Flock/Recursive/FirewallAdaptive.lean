module

public import Proofs.Flock.Recursive.FirewallAny

/-!
# The live firewall at a worker that reads its answers

`FirewallAny.liveOut_any` fixes the inner worker's requests, and how many salts are read for each, before any salt. The
inner worker sends each request after reading what `flock-firewall live` answered it (`Flock.Firewall.liveWith`): what
it sent the challenger's live-coin part for each request and the salts it drew for them, `{"sent", "salts"}`. Here the worker's next
request is any function of those answers, its tape, its randomness and the challenger's live-coin part's being its parameters, so the
live coins it reads are too. Whatever it is, the live firewall sends the schedule's transcript filled with commit
strings `strs dg ys`, the `i`-th that of a 64-byte digest `dg i (ys.take i)` under salt `i`, cut after one of its
messages: digest `i` reads the salts before it, never salt `i` or a later one.

* `adaptOut`: the live loop at a worker `W`, for at most `N` requests, reading the salts from the list `ys` in turn.
* `liveOut_prefix`, `adaptOut_prefix`: the sends on a prefix of the requests are a prefix of the sends.
* `adaptOut_liveOut`: each such session is a live session (`FirewallAny.liveOut`) on the requests it took.
* `adaptOut_rel`: two sessions whose salts agree before `i` send the commit strings of the same digests up to salt
  `i`: they take the same requests and sample the same salts until one samples salt `i`.
* `Flock.SecurityProofs.adaptOut_any`: the theorem.

The model reads the salts from a list, and a session that reads more salts than the list holds is given those left;
`liveWith` reads them from the operating system, which never runs out, so the two agree at a list long enough for the
salts the session reads. A line that isn't hex stops `liveWith`'s session as a refused request does, and the worker
model sends bytes: the empty request, which `decode` refuses, sends the same (`adaptOut_empty`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.FirewallAdaptive

open Flock Flock.Firewall FirewallRun FirewallTemplate FirewallLive FirewallAny

/-- **What the live firewall answered so far**: per request, what it sent the challenger's live-coin part and the salts it drew. -/
abbrev History := List (List Sent × List ByteArray)

/-- **An inner worker**: its next request, read from the answers so far, or `none` when it closes the session. Its
tape and randomness, and the challenger's live-coin part's, are its parameters. -/
abbrev Worker := History → Option ByteArray

/-- How many fresh salts `liveWith` reads before a request runs. -/
def reads (b : ByteArray) : Nat := b.size / NODE_BYTES + 1

/-- **A live session at the worker `W`**, for at most `N` requests: from the state `st` and the answers `h`, the salts
read in turn from `ys`, what the ZK server sends the challenger's live-coin part, in order, up to the first refusal, and how the session
ends, as `FirewallAny.liveOut`. -/
def adaptOut (s : Schedule) (W : Worker) : Nat → State → History → List ByteArray → List Sent × Option State
  | 0, st, _, _ => ([], some st)
  | N + 1, st, h, ys =>
    match W h with
    | none => ([], some st)
    | some b =>
      match liveStep s st b (ys.take (reads b)) with
      | .ok (sent, used, st') =>
        (sent ++ (adaptOut s W N st' (h ++ [(sent, used)]) (ys.drop (reads b))).1,
          (adaptOut s W N st' (h ++ [(sent, used)]) (ys.drop (reads b))).2)
      | .error _ => ([], none)

/-- **The session finished**: no request was refused and `Finish` was taken. -/
def adaptFinished (s : Schedule) (W : Worker) (N : Nat) (ys : List ByteArray) : Bool :=
  match (adaptOut s W N { salts := [] } [] ys).2 with
  | some t => t.finished
  | none => false

/-- **The commit strings at digests that read the salts before them**: string `i` is digest `dg i (ys.take i)` under
salt `i`. -/
def strs (dg : Nat → List ByteArray → ByteArray) (ys : List ByteArray) : List ByteArray :=
  List.zipWith Hm96.Default512.commitString ((List.range ys.length).map fun i => dg i (ys.take i)) ys

theorem length_strs (dg : Nat → List ByteArray → ByteArray) (ys : List ByteArray) :
    (strs dg ys).length = ys.length := by
  simp only [strs, List.length_zipWith, List.length_map, List.length_range, Nat.min_self]

theorem getElem_strs (dg : Nat → List ByteArray → ByteArray) (ys : List ByteArray) (i : Nat)
    (hi : i < (strs dg ys).length) :
    (strs dg ys)[i] = Hm96.Default512.commitString (dg i (ys.take i)) (ys[i]'(length_strs dg ys ▸ hi)) := by
  simp only [strs, List.getElem_zipWith, List.getElem_map, List.getElem_range]

theorem adaptOut_none {s : Schedule} {W : Worker} {N : Nat} {st : State} {h : History} {ys : List ByteArray}
    (hW : W h = none) : adaptOut s W (N + 1) st h ys = ([], some st) := by
  simp only [adaptOut, hW]

theorem adaptOut_some {s : Schedule} {W : Worker} {N : Nat} {st : State} {h : History} {ys : List ByteArray}
    {b : ByteArray} (hW : W h = some b) :
    adaptOut s W (N + 1) st h ys = match liveStep s st b (ys.take (reads b)) with
      | .ok (sent, used, st') =>
        (sent ++ (adaptOut s W N st' (h ++ [(sent, used)]) (ys.drop (reads b))).1,
          (adaptOut s W N st' (h ++ [(sent, used)]) (ys.drop (reads b))).2)
      | .error _ => ([], none) := by
  simp only [adaptOut, hW]

/-! ## A prefix of the requests -/

theorem liveOut_append (s : Schedule) : ∀ (st : State) (L₁ L₂ : List (ByteArray × List ByteArray)),
    ∃ r : List Sent, (liveOut s st (L₁ ++ L₂)).1 = (liveOut s st L₁).1 ++ r
  | st, [], L₂ => ⟨(liveOut s st L₂).1, rfl⟩
  | st, (b, f) :: rest, L₂ => by
    rcases hm : (stepOf s b).run { st with salts := st.salts ++ f, sent := #[] } with e | ⟨⟨⟩, t⟩
    · rw [List.cons_append, liveOut_error hm, liveOut_error hm]
      exact ⟨[], rfl⟩
    · obtain ⟨r, hr⟩ := liveOut_append s { t with sent := #[] } rest L₂
      rw [List.cons_append, liveOut_ok hm, liveOut_ok hm]
      exact ⟨r, by simp only [hr, List.append_assoc]⟩

/-- **The sends on a prefix of the requests are a prefix of the sends.** -/
theorem liveOut_prefix (s : Schedule) (st : State) (L₁ L₂ : List (ByteArray × List ByteArray)) :
    (liveOut s st L₁).1 <+: (liveOut s st (L₁ ++ L₂)).1 := by
  obtain ⟨r, hr⟩ := liveOut_append s st L₁ L₂
  rw [hr]
  exact List.prefix_append _ _

/-- **The sends on a worker's first requests are a prefix of its sends.** -/
theorem adaptOut_prefix (s : Schedule) (W : Worker) : ∀ (N M : Nat) (st : State) (h : History) (ys : List ByteArray),
    (adaptOut s W N st h ys).1 <+: (adaptOut s W (N + M) st h ys).1
  | 0, _, _, _, _ => List.nil_prefix
  | N + 1, M, st, h, ys => by
    rw [show N + 1 + M = N + M + 1 by omega]
    rcases hW : W h with _ | b
    · rw [adaptOut_none hW]
      exact List.nil_prefix
    rw [adaptOut_some hW, adaptOut_some hW]
    rcases liveStep s st b (ys.take (reads b)) with e | ⟨sent, used, st'⟩
    · exact List.prefix_refl _
    · exact (List.prefix_append_right_inj sent).mpr (adaptOut_prefix s W N M st' _ _)

/-! ## The requests a session took -/

/-- **A worker's session is a live session on the requests it took.** -/
theorem adaptOut_liveOut (s : Schedule) (W : Worker) : ∀ (N : Nat) (st : State) (h : History) (ys : List ByteArray),
    ∃ L : List (ByteArray × List ByteArray), adaptOut s W N st h ys = liveOut s st L
  | 0, st, _, _ => ⟨[], rfl⟩
  | N + 1, st, h, ys => by
    rcases hW : W h with _ | b
    · exact ⟨[], adaptOut_none hW⟩
    rw [adaptOut_some hW]
    rcases hl : liveStep s st b (ys.take (reads b)) with e | ⟨sent, used, st'⟩
    · refine ⟨[(b, ys.take (reads b))], ?_⟩
      simp only [liveOut, hl]
    · obtain ⟨L, hL⟩ := adaptOut_liveOut s W N st' (h ++ [(sent, used)]) (ys.drop (reads b))
      refine ⟨(b, ys.take (reads b)) :: L, ?_⟩
      simp only [liveOut, hl, hL]

/-! ## Two sessions whose salts agree before `i` -/

theorem adaptOut_ok {s : Schedule} {W : Worker} {N : Nat} {st t : State} {h : History} {ys : List ByteArray}
    {b : ByteArray} (hW : W h = some b)
    (hm : (stepOf s b).run { st with salts := st.salts ++ ys.take (reads b), sent := #[] } = .ok ((), t)) :
    adaptOut s W (N + 1) st h ys =
      (t.sent.toList ++ (adaptOut s W N { t with sent := #[] } (h ++ [(t.sent.toList,
          (st.salts ++ ys.take (reads b)).take ((st.salts ++ ys.take (reads b)).length - t.salts.length))])
          (ys.drop (reads b))).1,
        (adaptOut s W N { t with sent := #[] } (h ++ [(t.sent.toList,
          (st.salts ++ ys.take (reads b)).take ((st.salts ++ ys.take (reads b)).length - t.salts.length))])
          (ys.drop (reads b))).2) := by
  rw [adaptOut_some hW, liveStep_eq, hm]

theorem adaptOut_error {s : Schedule} {W : Worker} {N : Nat} {st : State} {h : History} {ys : List ByteArray}
    {b : ByteArray} {e : String} (hW : W h = some b)
    (hm : (stepOf s b).run { st with salts := st.salts ++ ys.take (reads b), sent := #[] } = .error e) :
    adaptOut s W (N + 1) st h ys = ([], none) := by
  rw [adaptOut_some hW, liveStep_eq, hm]

/-- **The empty request stops the session and sends nothing**, as a line that isn't hex does. -/
theorem adaptOut_empty {s : Schedule} {W : Worker} {N : Nat} {st : State} {h : History} {ys : List ByteArray}
    (hW : W h = some ByteArray.empty) : adaptOut s W (N + 1) st h ys = ([], none) :=
  adaptOut_error (e := "empty frame") hW rfl

theorem zipWith_take_length {α β γ : Type} (f : α → β → γ) :
    ∀ (a : List α) (l : List β), List.zipWith f a l = List.zipWith f a (l.take a.length)
  | [], _ => by simp
  | _ :: _, [] => by simp
  | x :: a, y :: l => by
    simp only [List.zipWith_cons_cons, List.length_cons, List.take_succ_cons]
    rw [← zipWith_take_length f a l]

/-- A request's commit strings, then the rest of the session's, at the salts in order. -/
theorem holes_append {ds E p q L R : List ByteArray} {m r : List Sent} (hL : L = p ++ q)
    (hR : R = L.drop ds.length) (hm : m.flatMap holes = List.zipWith Hm96.Default512.commitString ds p)
    (hr : r.flatMap holes = List.zipWith Hm96.Default512.commitString E R) (hd : ds.length ≤ p.length) :
    (m ++ r).flatMap holes = List.zipWith Hm96.Default512.commitString (ds ++ E) L := by
  subst hL hR
  rw [List.flatMap_append, hm, hr, zipWith_append_left, zipWith_append_of_le _ _ _ _ hd]

/-- **Two sessions whose salts agree before `i`**, from states equal but for unsampled salts of one length and the same
answers: their commit strings are those of digests `D₁ ++ D₂` and `D₁ ++ D₂'` under each session's salts, the state's
then those of `ys`, with `D₁` past salt `i` or nothing after it. Until a request samples salt `i`, the two take the same
requests, sample the same salts and send the same; that request samples the same digests in both
(`FirewallRun.stepOf_rel`). -/
theorem adaptOut_rel (s : Schedule) (W : Worker) : ∀ (N : Nat) (st : State) (z' : List ByteArray) (h : History)
    (ys ys' : List ByteArray) (i : Nat), st.salts.length = z'.length → ys.length = ys'.length →
    (st.salts ++ ys).take i = (z' ++ ys').take i →
    ∃ D₁ D₂ D₂' : List ByteArray, (∀ d ∈ D₁ ++ D₂ ++ D₂', d.size = 64) ∧
      (D₁ ++ D₂).length ≤ (st.salts ++ ys).length ∧ (D₁ ++ D₂').length ≤ (z' ++ ys').length ∧
      (adaptOut s W N st h ys).1.flatMap holes =
        List.zipWith Hm96.Default512.commitString (D₁ ++ D₂) (st.salts ++ ys) ∧
      (adaptOut s W N { st with salts := z' } h ys').1.flatMap holes =
        List.zipWith Hm96.Default512.commitString (D₁ ++ D₂') (z' ++ ys') ∧
      (i < D₁.length ∨ (D₂ = [] ∧ D₂' = []))
  | 0, _, _, _, _, _, _, _, _, _ =>
    ⟨[], [], [], by simp, Nat.zero_le _, Nat.zero_le _, rfl, rfl, Or.inr ⟨rfl, rfl⟩⟩
  | N + 1, st, z', h, ys, ys', i, hz, hy, ha => by
    rcases hW : W h with _ | b
    · rw [adaptOut_none hW, adaptOut_none hW]
      exact ⟨[], [], [], by simp, Nat.zero_le _, Nat.zero_le _, rfl, rfl, Or.inr ⟨rfl, rfl⟩⟩
    have hpl : (st.salts ++ ys.take (reads b)).length = (z' ++ ys'.take (reads b)).length := by
      simp only [List.length_append, List.length_take, hz, hy]
    have H := stepOf_rel (ys := st.salts ++ ys.take (reads b)) (ys' := z' ++ ys'.take (reads b)) s b
      { st with salts := st.salts ++ ys.take (reads b), sent := #[] } (z' ++ ys'.take (reads b)) #[]
      ⟨rfl, hpl, [], (fun _ h => nomatch h), Nat.zero_le _, rfl, rfl, rfl, rfl⟩
    change Both _ ((stepOf s b).run _)
      ((stepOf s b).run { st with salts := z' ++ ys'.take (reads b), sent := #[] }) at H
    rcases hm : (stepOf s b).run { st with salts := st.salts ++ ys.take (reads b), sent := #[] } with
        e | ⟨⟨⟩, t⟩ <;>
      rcases hm' : (stepOf s b).run { st with salts := z' ++ ys'.take (reads b), sent := #[] } with
        e' | ⟨⟨⟩, t'⟩ <;>
      rw [hm, hm'] at H
    · rw [adaptOut_error hW hm, adaptOut_error (st := { st with salts := z' }) hW hm']
      exact ⟨[], [], [], by simp, Nat.zero_le _, Nat.zero_le _, rfl, rfl, Or.inr ⟨rfl, rfl⟩⟩
    · exact H.elim
    · exact H.elim
    obtain ⟨q, q', ⟨rfl, rfl⟩, z₁, x₁, ht', hb, -, ds₁, hds₁, hle₁, hz₁, hz₁', hh₁, hh₁'⟩ := H
    subst ht'
    rw [adaptOut_ok hW hm, adaptOut_ok (st := { st with salts := z' }) hW hm']
    have hle₁' : ds₁.length ≤ (z' ++ ys'.take (reads b)).length := hpl ▸ hle₁
    have hp : st.salts ++ ys = (st.salts ++ ys.take (reads b)) ++ ys.drop (reads b) := by
      rw [List.append_assoc, List.take_append_drop]
    have hp' : z' ++ ys' = (z' ++ ys'.take (reads b)) ++ ys'.drop (reads b) := by
      rw [List.append_assoc, List.take_append_drop]
    have hcat : t.salts ++ ys.drop (reads b) = (st.salts ++ ys).drop ds₁.length := by
      rw [hz₁, hp, List.drop_append_of_le_length hle₁]
    have hcat' : z₁ ++ ys'.drop (reads b) = (z' ++ ys').drop ds₁.length := by
      rw [hz₁', hp', List.drop_append_of_le_length hle₁']
    have hm1 : t.sent.toList.flatMap holes =
        List.zipWith Hm96.Default512.commitString ds₁ (st.salts ++ ys.take (reads b)) :=
      (List.append_nil _).symm.trans hh₁
    have hm2 : x₁.toList.flatMap holes =
        List.zipWith Hm96.Default512.commitString ds₁ (z' ++ ys'.take (reads b)) :=
      (List.append_nil _).symm.trans hh₁'
    have hdL : ds₁.length ≤ (st.salts ++ ys).length := by
      rw [hp, List.length_append]
      omega
    have hdL' : ds₁.length ≤ (z' ++ ys').length := by
      rw [hp', List.length_append]
      omega
    have hc1 : (t.salts ++ ys.drop (reads b)).length = (st.salts ++ ys).length - ds₁.length := by
      rw [hcat, List.length_drop]
    have hc2 : (z₁ ++ ys'.drop (reads b)).length = (z' ++ ys').length - ds₁.length := by
      rw [hcat', List.length_drop]
    rcases Nat.lt_or_ge i ds₁.length with hi | hi
    · obtain ⟨E₁, E₂, E₃, hE, hEl, -, hr1, -, -⟩ := adaptOut_rel s W N { t with sent := #[] } t.salts
        (h ++ [(t.sent.toList, (st.salts ++ ys.take (reads b)).take
          ((st.salts ++ ys.take (reads b)).length - t.salts.length))]) (ys.drop (reads b)) (ys.drop (reads b)) 0
        rfl rfl rfl
      obtain ⟨F₁, F₂, F₃, hF, hFl, -, hr2, -, -⟩ := adaptOut_rel s W N { { t with sent := #[] } with salts := z₁ } z₁
        (h ++ [(x₁.toList, (z' ++ ys'.take (reads b)).take ((z' ++ ys'.take (reads b)).length - z₁.length))])
        (ys'.drop (reads b)) (ys'.drop (reads b)) 0 rfl rfl rfl
      have hEl' : (E₁ ++ E₂).length ≤ (t.salts ++ ys.drop (reads b)).length := hEl
      have hFl' : (F₁ ++ F₂).length ≤ (z₁ ++ ys'.drop (reads b)).length := hFl
      refine ⟨ds₁, E₁ ++ E₂, F₁ ++ F₂, fun x hx => ?_, ?_, ?_, holes_append hp hcat hm1 hr1 hle₁,
        holes_append hp' hcat' hm2 hr2 hle₁', Or.inl hi⟩
      · simp only [List.mem_append] at hx hE hF
        rcases hx with (hx | hx | hx) | hx | hx
        · exact hds₁ x hx
        · exact hE x (Or.inl (Or.inl hx))
        · exact hE x (Or.inl (Or.inr hx))
        · exact hF x (Or.inl (Or.inl hx))
        · exact hF x (Or.inl (Or.inr hx))
      · simp only [List.length_append] at hEl' hc1 hdL ⊢
        omega
      · simp only [List.length_append] at hFl' hc2 hdL' ⊢
        omega
    · have hpre : (st.salts ++ ys.take (reads b)).take ds₁.length = (z' ++ ys'.take (reads b)).take ds₁.length :=
        calc (st.salts ++ ys.take (reads b)).take ds₁.length = (st.salts ++ ys).take ds₁.length := by
              rw [hp, List.take_append_of_le_length hle₁]
          _ = ((st.salts ++ ys).take i).take ds₁.length := by rw [List.take_take, Nat.min_eq_left hi]
          _ = ((z' ++ ys').take i).take ds₁.length := by rw [ha]
          _ = (z' ++ ys').take ds₁.length := by rw [List.take_take, Nat.min_eq_left hi]
          _ = (z' ++ ys'.take (reads b)).take ds₁.length := by rw [hp', List.take_append_of_le_length hle₁']
      have hholes : t.sent.toList.flatMap holes = x₁.toList.flatMap holes := by
        rw [hm1, hm2, zipWith_take_length _ ds₁ (st.salts ++ ys.take (reads b)),
          zipWith_take_length _ ds₁ (z' ++ ys'.take (reads b)), hpre]
      have hsent : t.sent.toList = x₁.toList := by
        have e := fillAll_blank_holes t.sent.toList []
        have e' := fillAll_blank_holes x₁.toList []
        rw [List.append_nil] at e e'
        rw [hb, hholes] at e
        exact e.symm.trans e'
      have hused : (st.salts ++ ys.take (reads b)).take ((st.salts ++ ys.take (reads b)).length - t.salts.length) =
          (z' ++ ys'.take (reads b)).take ((z' ++ ys'.take (reads b)).length - z₁.length) := by
        have e1 : (st.salts ++ ys.take (reads b)).length - t.salts.length = ds₁.length := by
          rw [hz₁, List.length_drop]
          omega
        have e2 : (z' ++ ys'.take (reads b)).length - z₁.length = ds₁.length := by
          rw [hz₁', List.length_drop]
          omega
        rw [e1, e2, hpre]
      have hz₂ : ({ t with sent := #[] } : State).salts.length = z₁.length := by
        show t.salts.length = z₁.length
        rw [hz₁, hz₁', List.length_drop, List.length_drop, hpl]
      have hy₂ : (ys.drop (reads b)).length = (ys'.drop (reads b)).length := by
        simp only [List.length_drop, hy]
      have ha₂ : (({ t with sent := #[] } : State).salts ++ ys.drop (reads b)).take (i - ds₁.length) =
          (z₁ ++ ys'.drop (reads b)).take (i - ds₁.length) := by
        show (t.salts ++ ys.drop (reads b)).take (i - ds₁.length) = _
        rw [hcat, hcat', ← List.drop_take, ← List.drop_take, ha]
      obtain ⟨E₁, E₂, E₂', hE, hEl, hEl', hr1, hr2, hor⟩ := adaptOut_rel s W N { t with sent := #[] } z₁
        (h ++ [(t.sent.toList, (st.salts ++ ys.take (reads b)).take
          ((st.salts ++ ys.take (reads b)).length - t.salts.length))]) (ys.drop (reads b)) (ys'.drop (reads b))
        (i - ds₁.length) hz₂ hy₂ ha₂
      rw [hsent, hused] at hr2
      have hEl₂ : (E₁ ++ E₂).length ≤ (t.salts ++ ys.drop (reads b)).length := hEl
      refine ⟨ds₁ ++ E₁, E₂, E₂', fun x hx => ?_, ?_, ?_, ?_, ?_, ?_⟩
      · simp only [List.mem_append] at hx hE
        rcases hx with ((hx | hx) | hx) | hx
        · exact hds₁ x hx
        · exact hE x (Or.inl (Or.inl hx))
        · exact hE x (Or.inl (Or.inr hx))
        · exact hE x (Or.inr hx)
      · simp only [List.length_append] at hEl₂ hc1 hdL ⊢
        omega
      · simp only [List.length_append] at hEl' hc2 hdL' ⊢
        omega
      · rw [List.append_assoc]
        exact holes_append hp hcat hm1 hr1 hle₁
      · rw [List.append_assoc]
        exact holes_append hp' hcat' hm2 hr2 hle₁'
      · rcases hor with hor | hor
        · left
          rw [List.length_append]
          omega
        · exact Or.inr hor

/-! ## Digest `i` reads the salts before `i` -/

/-- **A session's digests**: its commit strings are those of the 64-byte digests `D` under the salts in order. -/
def Digests (s : Schedule) (W : Worker) (N : Nat) (ys D : List ByteArray) : Prop :=
  (∀ d ∈ D, d.size = 64) ∧ D.length ≤ ys.length ∧
    (adaptOut s W N { salts := [] } [] ys).1.flatMap holes = List.zipWith Hm96.Default512.commitString D ys

theorem digests_exist (s : Schedule) (W : Worker) (N : Nat) (ys : List ByteArray) : ∃ D, Digests s W N ys D := by
  obtain ⟨D₁, D₂, D₃, hD, hl, -, hh, -, -⟩ := adaptOut_rel s W N { salts := [] } [] [] ys ys 0 rfl rfl rfl
  refine ⟨D₁ ++ D₂, fun d hd => hD d (List.mem_append_left _ hd), hl, hh⟩

/-- **Digest `i` is the same at salt lists that agree before `i`.** -/
theorem digests_agree (s : Schedule) (W : Worker) (N : Nat) {ys ys' D D' : List ByteArray} (i : Nat)
    (hy : ys.length = ys'.length) (ha : ys.take i = ys'.take i) (hD : Digests s W N ys D) (hD' : Digests s W N ys' D')
    (hi : i < D.length) (hi' : i < D'.length) : D[i] = D'[i] := by
  obtain ⟨D₁, D₂, D₂', h64, hl, hl', hh, hh', hor⟩ :=
    adaptOut_rel s W N { salts := [] } [] [] ys ys' i rfl hy ha
  have e1 : D = D₁ ++ D₂ := digests_eq hD.1 (fun d hd => h64 d (List.mem_append_left _ hd)) hD.2.1 hl
    (hD.2.2.symm.trans hh)
  have e2 : D' = D₁ ++ D₂' := digests_eq hD'.1 (fun d hd => h64 d (by
      rcases List.mem_append.mp hd with hd | hd
      · exact List.mem_append_left _ (List.mem_append_left _ hd)
      · exact List.mem_append_right _ hd)) hD'.2.1 hl' (hD'.2.2.symm.trans hh')
  subst e1 e2
  rcases hor with hi₁ | ⟨rfl, rfl⟩
  · rw [List.getElem_append_left hi₁, List.getElem_append_left hi₁]
  · rfl

theorem digest_fn (s : Schedule) (W : Worker) (N n : Nat) :
    ∃ dg : Nat → List ByteArray → ByteArray, ∀ i pre, (dg i pre).size = 64 ∧
      ∀ ys D, ys.length = n → ys.take i = pre → Digests s W N ys D → ∀ hi : i < D.length, D[i] = dg i pre := by
  have key : ∀ i (pre : List ByteArray), ∃ d : ByteArray, d.size = 64 ∧
      ∀ ys D, ys.length = n → ys.take i = pre → Digests s W N ys D → ∀ hi : i < D.length, D[i] = d := by
    intro i pre
    rcases Classical.em (∃ ys D, ys.length = n ∧ ys.take i = pre ∧ Digests s W N ys D ∧ i < D.length) with
      ⟨ys₀, D₀, hn₀, hp₀, hD₀, hi₀⟩ | hex
    · exact ⟨D₀[i], hD₀.1 _ (List.getElem_mem _), fun ys D hn hp hD hi =>
        digests_agree s W N i (hn.trans hn₀.symm) (hp.trans hp₀.symm) hD hD₀ hi hi₀⟩
    · exact ⟨⟨Array.replicate 64 0⟩, Array.size_replicate .., fun ys D hn hp hD hi =>
        absurd ⟨ys, D, hn, hp, hD, hi⟩ hex⟩
  exact ⟨fun i pre => Classical.choose (key i pre), fun i pre => Classical.choose_spec (key i pre)⟩

end FlockSoundness.Discharge.Recursion.FirewallAdaptive

namespace Flock.SecurityProofs

open Flock.Firewall FlockSoundness.Discharge.Recursion
open FirewallRun FirewallTemplate FirewallAny FirewallAdaptive

/-- **The live firewall at any worker that reads its answers sends the schedule's transcript, filled at commit strings
whose digests read only the salts before them, or stops at a public place.** For any inner worker `W` whose next request
reads the answers so far, the sends and the salts sampled, for at most `N` requests, and any number `n` of salts, there are
digests `dg`, `dg i pre` 64 bytes, and a stop `σ` such that at every salt list `ys` of length `n`, the live session
(`adaptOut`, `flock-firewall live`'s loop) sends the challenger's live-coin part the schedule's transcript (`tmpl s s.tables`, Commit
naming the schedule's tables in its order, the only order `step` takes), filled with `strs dg ys`, string `i` the commit
string of `dg i (ys.take i)` under salt `i`, cut after its first `(σ ys).1` messages, and finishes exactly when
`(σ ys).2`. The stop lies in `Fin ((tmpl s s.tables).length + 1) × Bool`, and a session that finishes sends the whole
transcript. Digest `i` reads the salts before `i` alone: `W` may read every salt it was answered. -/
theorem adaptOut_any (s : Schedule) (W : Worker) (N n : Nat) :
    ∃ (dg : Nat → List ByteArray → ByteArray) (σ : List ByteArray → Nat × Bool),
      (∀ i pre, (dg i pre).size = 64) ∧
      ∀ ys : List ByteArray, ys.length = n →
        (σ ys).1 ≤ (tmpl s s.tables).length ∧ ((σ ys).2 = true → (σ ys).1 = (tmpl s s.tables).length) ∧
        (adaptOut s W N { salts := [] } [] ys).1 = (fillAll (tmpl s s.tables) (strs dg ys)).take (σ ys).1 ∧
        adaptFinished s W N ys = (σ ys).2 := by
  obtain ⟨dg, hdg⟩ := digest_fn s W N n
  have key : ∀ ys : List ByteArray, ∃ σ : Nat × Bool, ys.length = n →
      σ.1 ≤ (tmpl s s.tables).length ∧ (σ.2 = true → σ.1 = (tmpl s s.tables).length) ∧
      (adaptOut s W N { salts := [] } [] ys).1 = (fillAll (tmpl s s.tables) (strs dg ys)).take σ.1 ∧
      adaptFinished s W N ys = σ.2 := by
    intro ys
    obtain ⟨D, hD⟩ := digests_exist s W N ys
    obtain ⟨L, hL⟩ := adaptOut_liveOut s W N { salts := [] } [] ys
    obtain ⟨k, hk, hfin, hbl⟩ := liveOut_blank s L
    refine ⟨(k, liveFinished s L), fun hn => ⟨hk, hfin, ?_, ?_⟩⟩
    · have hpre : ((List.range ys.length).map fun i => dg i (ys.take i)).take D.length = D := by
        apply List.ext_getElem
        · simp only [List.length_take, List.length_map, List.length_range]
          exact Nat.min_eq_left hD.2.1
        · intro j _ h₂
          simp only [List.getElem_take, List.getElem_map, List.getElem_range]
          exact ((hdg j (ys.take j)).2 ys D hn rfl hD h₂).symm
      have hstrs : strs dg ys = (adaptOut s W N { salts := [] } [] ys).1.flatMap holes ++
          List.zipWith Hm96.Default512.commitString
            (((List.range ys.length).map fun i => dg i (ys.take i)).drop D.length) (ys.drop D.length) := by
        rw [hD.2.2, strs, ← List.take_append_drop D.length ((List.range ys.length).map fun i => dg i (ys.take i)),
          zipWith_append_left, hpre, List.drop_left]
      have e3 := fillAll_blank_holes (adaptOut s W N { salts := [] } [] ys).1
        (List.zipWith Hm96.Default512.commitString
          (((List.range ys.length).map fun i => dg i (ys.take i)).drop D.length) (ys.drop D.length))
      rw [← hstrs, hL, hbl, fillAll_take] at e3
      rw [hL]
      exact e3.symm
    · unfold adaptFinished liveFinished
      rw [hL]
      generalize (liveOut s { salts := [] } L).2 = r
      cases r <;> rfl
  exact ⟨dg, fun ys => Classical.choose (key ys), fun i pre => (hdg i pre).1,
    fun ys hn => Classical.choose_spec (key ys) hn⟩

end Flock.SecurityProofs
