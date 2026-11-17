module

public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.Verity.SampledRead
public import Proofs.Properties.Verity.Sampled.Walk

/-! Every round a session's coin tree opens is a round of its record. -/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)

/-- **The rounds a session's record gives stream `stream` of its coin tree**: one of
`Spot.Challenger.TREE_SESSION`, its round 0 the link points; of another stream, the size of `rounds` of the
record's first stream so named (`recordRound`'s); none of a stream the record doesn't name. -/
def recordRounds (record : Json) (stream : String) : ℕ :=
  if stream == _root_.Spot.Challenger.TREE_SESSION then 1
  else
    match _root_.Spot.getD record "streams" with
    | .arr ss =>
      match ss.find? (fun x => match _root_.Spot.getD x "stream" with | .str n => n == stream | _ => false) with
      | some x =>
        match _root_.Spot.getD x "rounds" with
        | .arr rs => rs.size
        | _ => 0
      | none => 0
    | _ => 0

def openedRecorded (t : _root_.Spot.Challenger.LoggedTree) (record : Json) : Bool :=
  t.opened.all fun o => o.round < recordRounds record o.stream

def OpenedRecorded (record : Json) (log : List Verity.Release) : Prop :=
  ∀ pre r post, log = pre ++ r :: post → ∀ i s n, r.frame = .opening i s n →
    ∀ name, ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name →
      openedOf pre i s < recordRounds record name

theorem openedOf_congr {i s : ℕ} :
    ∀ {l l' : List Verity.Release}, List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l' →
      openedOf l i s = openedOf l' i s
  | _, _, .nil => rfl
  | _ :: _, _ :: _, .cons h t => by
    have e := openedOf_congr (i := i) (s := s) t
    simp only [openedOf, List.countP_cons, h.1] at e ⊢
    rw [e]

theorem openedRecorded_congr {record : Json} {l l' : List Verity.Release}
    (hl : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l')
    (hP : OpenedRecorded record l) : OpenedRecorded record l' := by
  intro pre' r' post' heq i s n hr' name hname
  subst heq
  obtain ⟨pre, r, post, rfl, hpre, hr, -⟩ := forall₂_split hl
  rw [← helloSpecs_congr hpre] at hname
  rw [← openedOf_congr hpre]
  exact hP pre r post rfl i s n (hr.1.trans hr') name hname

theorem openedRecorded_logged {record : Json} {o : Verity.Exchange} (hL : Binding.Logged Binding.readsV2 o)
    {l : List Verity.Release} (hl : coinsRead o.sent = some l) (hP : OpenedRecorded record l) :
    OpenedRecorded record o.log := by
  have h := hL (l.map Binding.Read.ofV2) (by simp [Binding.readsV2, hl])
  refine openedRecorded_congr ?_ hP
  exact (List.forall₂_map_left_iff.1 h).imp fun _ _ hab => ⟨hab.1, hab.2.1⟩

end Verity.Sampled
