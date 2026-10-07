import CoreComposition.Product

/-!
# Compositions of core protocols: what their Lean code (`CoreComposition.Product`) does

The properties of `Core.Composition.product id cs`, the core protocol that composes the components `cs`
(`verity/experimental/core/protocols/composition/`), stated against the functions that run, for every id and every list
of components, so they hold of onsite and remote Verity (`Properties.Onsite`, `Properties.Remote`) whatever their
components become. Each is a `Prop`; `Properties.Proofs.Composition` proves them all.

The rule: the composition runs each component's own `answer` on that component's case, and

* rejects when any component rejects, whatever the others answer (`RejectDominates`), and when the cases don't fit the
  composition (`MisfitRejects`);
* is undecided exactly when the cases fit, no component rejects, the commitment binds their digests, and some component
  is undecided (`UndecidedIff`);
* accepts exactly when the cases fit, every component accepts, and the commitment binds their digests (`AcceptsIff`);
  it then makes public each component's public outputs and keeps each one's profile, keyed by the component's id
  (`PublishesEach`), and nothing is public otherwise (`NothingPublicUnlessAccepted`).

`AnswerIsCheck` ties the check to what a caller runs: `answer`'s verdict is the check's on the decoded inputs, and a
rejection when they don't decode.

None of these needs an assumption: each is about the code as it runs. In particular none says the commitment binds the
components to one set of digests, which would need SHA-512's collision resistance (`cr/sha-512`); they say only that the
commitment accepted is `binding`'s value on the digests the components' answers report.
-/

namespace Properties.Composition

open Core Core.Composition

/-- The verdict of an outcome. -/
def verdictOf {α β : Type} : Outcome α β → Verdict
  | .accept .. => .accept
  | .reject _ => .reject
  | .undecided _ => .undecided

/-- A component's answer on its case, under the parameters the composition names for it. -/
def answerOn (P : Protocol) (n : Named) (k : Case) : Answer :=
  P.answer n.params.toUTF8 k.commitment k.evidence

/-- The cases fit the composition of `cs`: the parameters and the evidence name exactly its components, in its order,
and each case carries the parameters the composition names for its component. -/
def Fits (cs : List Protocol) (ps : List Named) (ev : List Case) : Prop :=
  ps.map (·.id) = cs.map (·.id) ∧ ev.map (·.id) = cs.map (·.id) ∧
    ∀ x ∈ cs.zip (ps.zip ev), x.2.2.params = x.2.1.params

/-- When the cases fit, every component of `cs` is paired with the parameters and the case named by its own id, and its
answer is listed under that id. -/
def Covers : Prop :=
  ∀ (cs : List Protocol) (ps : List Named) (ev : List Case), Fits cs ps ev →
    (cs.zip (ps.zip ev)).map (·.1) = cs ∧ (cs.zip (ps.zip ev)).map (·.2.1.id) = cs.map (·.id) ∧
      (cs.zip (ps.zip ev)).map (·.2.2.id) = cs.map (·.id) ∧ (answersOf cs ps ev).map (·.1) = cs.map (·.id)

/-- Reject dominance: a component that rejects its case rejects the composition, whatever the others answer. -/
def RejectDominatesAt (id : String) (cs : List Protocol) : Prop :=
  ∀ (ps : List Named) (c : ByteArray) (ev : List Case) (P : Protocol) (n : Named) (k : Case),
    (P, n, k) ∈ cs.zip (ps.zip ev) → (answerOn P n k).verdict = .reject →
      verdictOf ((product id cs).check ps c ev) = .reject

def RejectDominates : Prop := ∀ id cs, RejectDominatesAt id cs

/-- Cases that don't fit (a component unknown, missing, repeated or out of order, or a case under other parameters)
reject the composition. -/
def MisfitRejectsAt (id : String) (cs : List Protocol) : Prop :=
  ∀ (ps : List Named) (c : ByteArray) (ev : List Case), ¬ Fits cs ps ev →
    verdictOf ((product id cs).check ps c ev) = .reject

def MisfitRejects : Prop := ∀ id cs, MisfitRejectsAt id cs

/-- The composition accepts exactly when its cases fit, every component accepts its case, and the commitment is the
binding of the digests the components' answers report. -/
def AcceptsIffAt (id : String) (cs : List Protocol) : Prop :=
  ∀ (ps : List Named) (c : ByteArray) (ev : List Case),
    (product id cs).Accepts ps c ev ↔
      Fits cs ps ev ∧ (∀ x ∈ cs.zip (ps.zip ev), (answerOn x.1 x.2.1 x.2.2).verdict = .accept) ∧
        binding id (answersOf cs ps ev) = some c

def AcceptsIff : Prop := ∀ id cs, AcceptsIffAt id cs

/-- The composition is undecided exactly when its cases fit, no component rejects, the commitment binds their digests,
and some component is undecided: an undecided component never hides a rejection. -/
def UndecidedIffAt (id : String) (cs : List Protocol) : Prop :=
  ∀ (ps : List Named) (c : ByteArray) (ev : List Case),
    verdictOf ((product id cs).check ps c ev) = .undecided ↔
      Fits cs ps ev ∧ (∀ x ∈ cs.zip (ps.zip ev), (answerOn x.1 x.2.1 x.2.2).verdict ≠ .reject) ∧
        binding id (answersOf cs ps ev) = some c ∧
          ∃ x ∈ cs.zip (ps.zip ev), (answerOn x.1 x.2.1 x.2.2).verdict = .undecided

def UndecidedIff : Prop := ∀ id cs, UndecidedIffAt id cs

/-- On acceptance the composition makes public each component's own public outputs and keeps each one's profile, keyed
by the component's id; every component has both, since every one accepted. -/
def PublishesEachAt (id : String) (cs : List Protocol) : Prop :=
  ∀ (ps : List Named) (c : ByteArray) (ev : List Case) (pub prof : Keyed),
    (product id cs).check ps c ev = .accept pub prof →
      pub = (answersOf cs ps ev).map (fun x => (x.1, x.2.published.getD .null)) ∧
        prof = (answersOf cs ps ev).map (fun x => (x.1, x.2.profile.getD .null)) ∧
          ∀ x ∈ answersOf cs ps ev, x.2.published.isSome ∧ x.2.profile.isSome

def PublishesEach : Prop := ∀ id cs, PublishesEachAt id cs

/-- What a caller runs, `answer`, gives the check's verdict on the decoded inputs, and rejects inputs that don't
decode. -/
def AnswerIsCheck : Prop :=
  ∀ (P : Protocol) (params commitment evidence : ByteArray),
    (P.answer params commitment evidence).verdict =
      match canonicalJson params >>= P.params, P.commitment commitment, P.evidence evidence with
      | .ok p, .ok c, .ok e => verdictOf (P.check p c e)
      | _, _, _ => .reject

/-- Nothing is public unless the answer accepts: no public outputs and no profile on a rejection or an undecided
answer. -/
def NothingPublicUnlessAccepted : Prop :=
  ∀ (P : Protocol) (params commitment evidence : ByteArray),
    (P.answer params commitment evidence).verdict ≠ .accept →
      (P.answer params commitment evidence).published = none ∧ (P.answer params commitment evidence).profile = none

end Properties.Composition
