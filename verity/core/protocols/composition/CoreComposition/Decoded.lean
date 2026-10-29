/-
A composition's check, `checkSame`, as decoding followed by a check of the decoded values.

`checkSame` hands each component's case, as bytes, to that component's `answer` (`answersOf`), and its tie decodes each
commitment again (`committedBy`). Here:

* `decodeCases`: each case decoded under the component in its place, with the parameters the composition names for it
  (`Protocol.decode`). It is the only part that reads bytes.
* `runOf`, `committedOf`, `straysOf`, `checkSameOf`: `run`, `committedBy`, `strays` and `checkSame` on the decoded
  cases, each component's answer its `answerOf`. The fit reads the cases' ids and parameters (`fitOf`), text it compares
  and never parses.
* `checkSame_decode`: on every input, `checkSame` is `checkSameOf` of `fitOf` and `decodeCases`. `sameB_decode` is the
  same on the caller's blobs.

A component reads its blobs inside its own check. So on blobs what is decoded here is each component's three inputs; a
blob a component's check decodes (a session's files, at the one-stage audit) stays in that check.
-/
import CoreComposition.Blobs
import CoreProtocol.Decoded

namespace Core.Composition

open Flock (V)

/-- A component with its case decoded under its own decoders. -/
structure Decoded where
  P : Protocol
  d : P.Decoded

/-- Each case of `ev` decoded under the component of `cs` in its place, with the parameters `ps` names for it: what
`answersOf` hands each component's `answer`. -/
def decodeCases (cs : List Protocol) (ps : List Named) (ev : List Case) : List Decoded :=
  (cs.zip (ps.zip ev)).map fun (P, n, k) => ⟨P, P.decode n.params.toUTF8 k.commitment k.evidence⟩

/-- The cases' ids and parameters, the text `run`'s fit compares. -/
def fitOf (ev : List Case) : List (String × String) := ev.map fun k => (k.id, k.params)

/-- `run` on the cases' ids and parameters `ks` and the decoded cases `ds`. -/
def runOf (cs : List Protocol) (ps : List Named) (ks : List (String × String)) (ds : List Decoded) : V Answers :=
  let want := cs.map (·.id)
  if ps.map (·.id) ≠ want then .error (misfit "params" want (ps.map (·.id)))
  else if ks.map (·.1) ≠ want then .error (misfit "evidence" want (ks.map (·.1)))
  else
    match (cs.zip (ps.zip ks)).find? (fun (_, n, k) => k.2 ≠ n.params) with
    | some (P, _, _) => .error s!"evidence: {P.id}'s case carries params other than the composition's"
    | none => .ok (ds.map fun x => (x.P.id, x.P.answerOf x.d))

/-- `committedBy` on the decoded cases. -/
def committedOf (s : String) (ds : List Decoded) : Option ByteArray :=
  (ds.find? fun x => x.P.id == s).bind fun x => x.d.commitment.toOption.map x.P.digest

/-- `strays` on the decoded cases. -/
def straysOf (ss : List String) (ds : List Decoded) : Option String :=
  match ss.mapM fun s => (committedOf s ds).map (s, ·) with
  | some ((s, d) :: rest) => (rest.find? (·.2 != d)).map fun (t, d') =>
    s!"record: {s} committed {Flock.hexOf d}, {t} committed {Flock.hexOf d'}"
  | _ => none

/-- `checkSame` on the cases' ids and parameters `ks` and the decoded cases `ds`. -/
def checkSameOf (id : String) (ss : List String) (cs : List Protocol) (ps : List Named) (c : ByteArray)
    (ks : List (String × String)) (ds : List Decoded) : Outcome Keyed Keyed :=
  match runOf cs ps ks ds with
  | .error r => .reject r
  | .ok answers => match straysOf ss ds with
    | some r => .reject r
    | none => judge id c answers

private theorem find_params (cs : List Protocol) (ps : List Named) (ev : List Case) :
    ((cs.zip (ps.zip ev)).find? (fun (_, n, k) => decide (k.params ≠ n.params))).map (·.1) =
      ((cs.zip (ps.zip (fitOf ev))).find? (fun (_, n, k) => decide (k.2 ≠ n.params))).map (·.1) := by
  induction cs generalizing ps ev with
  | nil => rfl
  | cons P cs ih =>
    cases ps with
    | nil => rfl
    | cons n ps =>
      cases ev with
      | nil => rfl
      | cons k ev =>
        have ih' := ih ps ev
        simp only [fitOf, List.map_cons, List.zip_cons_cons, List.find?_cons] at ih' ⊢
        by_cases hk : k.params = n.params
        · simpa [hk] using ih'
        · simp [hk]

private theorem answers_decode (cs : List Protocol) (ps : List Named) (ev : List Case) :
    answersOf cs ps ev = (decodeCases cs ps ev).map fun x => (x.P.id, x.P.answerOf x.d) := by
  simp only [answersOf, decodeCases, List.map_map]
  rfl

private theorem run_decode (cs : List Protocol) (ps : List Named) (ev : List Case) :
    run cs ps ev = runOf cs ps (fitOf ev) (decodeCases cs ps ev) := by
  have hids : (fitOf ev).map (·.1) = ev.map (·.id) := by simp [fitOf]
  unfold run runOf
  rw [hids]
  by_cases hp : ps.map (·.id) = cs.map (·.id)
  · by_cases he : ev.map (·.id) = cs.map (·.id)
    · have hf := find_params cs ps ev
      simp only [hp, he, ne_eq, not_true_eq_false, ite_false]
      simp only [ne_eq] at hf
      split
      next P n k h1 =>
        split
        next P' n' k' h2 => rw [h1, h2] at hf; cases hf; rfl
        next h2 => rw [h1, h2] at hf; cases hf
      next h1 =>
        split
        next P' n' k' h2 => rw [h1, h2] at hf; cases hf
        next h2 => rw [answers_decode]
    · simp only [hp, he, ne_eq, not_true_eq_false, not_false_eq_true, ite_false, ite_true]
  · simp only [hp, ne_eq, not_false_eq_true, ite_true]

private theorem committed_decode (s : String) (cs : List Protocol) (ps : List Named) (ev : List Case)
    (h : ps.length = cs.length) : committedBy s cs ev = committedOf s (decodeCases cs ps ev) := by
  induction cs generalizing ps ev with
  | nil => rfl
  | cons P cs ih =>
    cases ps with
    | nil => simp at h
    | cons n ps =>
      cases ev with
      | nil => rfl
      | cons k ev =>
        have ih' := ih ps ev (by simpa using h)
        simp only [committedBy, committedOf, decodeCases, List.zip_cons_cons, List.map_cons, List.find?_cons]
          at ih' ⊢
        cases hP : P.id == s
        · simpa [hP] using ih'
        · rfl

private theorem strays_decode (ss : List String) (cs : List Protocol) (ps : List Named) (ev : List Case)
    (h : ps.length = cs.length) : strays ss cs ev = straysOf ss (decodeCases cs ps ev) := by
  have e : (fun s => (committedBy s cs ev).map (s, ·)) = fun s => (committedOf s (decodeCases cs ps ev)).map (s, ·) :=
    funext fun s => by rw [committed_decode s cs ps ev h]
  simp only [strays, straysOf, e]
  cases List.mapM (fun s => (committedOf s (decodeCases cs ps ev)).map (s, ·)) ss with
  | none => rfl
  | some l =>
    cases l with
    | nil => rfl
    | cons x rest => obtain ⟨s, d⟩ := x; rfl

/-- **`checkSame` is decoding followed by a check of the decoded values**, on every input. -/
theorem checkSame_decode (id : String) (ss : List String) (cs : List Protocol) (ps : List Named) (c : ByteArray)
    (ev : List Case) :
    checkSame id ss cs ps c ev = checkSameOf id ss cs ps c (fitOf ev) (decodeCases cs ps ev) := by
  unfold checkSame checkSameOf
  rw [run_decode]
  by_cases hp : ps.map (·.id) = cs.map (·.id)
  · rw [strays_decode ss cs ps ev (by simpa using congrArg List.length hp)]
    cases runOf cs ps (fitOf ev) (decodeCases cs ps ev) with
    | error r => rfl
    | ok a => cases straysOf ss (decodeCases cs ps ev) <;> rfl
  · have hr : runOf cs ps (fitOf ev) (decodeCases cs ps ev) =
        .error (misfit "params" (cs.map (·.id)) (ps.map (·.id))) := by
      simp only [runOf, hp, ne_eq, not_false_eq_true, ite_true]
    rw [hr]

/-- **On the caller's blobs, `sameB`'s check is decoding followed by a check of the decoded values**, each component on
those blobs. -/
theorem sameB_decode (id : String) (ss : List String) (cs : List BlobProtocol) (blobs : Array ByteArray)
    (ps : List Named) (c : ByteArray) (ev : List Case) :
    ((sameB id ss cs).on blobs).check ps c ev =
      checkSameOf id ss (cs.map (·.onBlobs (Blobs.ofArray blobs))) ps c (fitOf ev)
        (decodeCases (cs.map (·.onBlobs (Blobs.ofArray blobs))) ps ev) :=
  checkSame_decode id ss _ ps c ev

end Core.Composition
