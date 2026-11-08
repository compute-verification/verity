module

public import Definitions.NetworkCertifier.Ingress

/-!
# Continuations: a response open at a run edge, continued in the next run

Definitions only. A monitoring period's runs are cut at public edges, the bucket `b = w·T` that starts run `w` of `T`
buckets, inside one certifier session: the schedule, the connections' ordinals, the release order and the watermark run
on across an edge, and of the served state only the pool root crosses it. A request released before `b` that the ids it
released before `b` don't finish is open at `b`. It continues in the next run as an ordinary arrival (`Continuation`):
its token row is its prompt, then its released ids, `e` of them, its frames released before `b`, and `max_tokens − e` is
left. The certifier stamps each open request's `(q, N, e)` in the next run's ingress, in release order, and the
reference recomputes the stamps from the session's records (`stampsAt`), so a stamp missing, extra or out of order is
refused. `e` is read off the certified egress, never the engine's count of what it emitted, which would be advice.

`ContinuationCertified` is what the leak bounds need of a stamp rule: its stamps at `b` are a function of the certified
ingress and of the observations before `b`, which hold the ids `y` released and their timing. The session's ingress as
deployed (`StampedIngress`), each edge's stamps recomputed from the whole session's records, is then the base
operator's at each step, joined at each edge with the stamps (`contIngress`), each step's a function of the observations
before it; and run `i`'s is `opCont` of the earlier runs' observations and its own so far, the chained operator the
carried pool's count reads. A continuation's prompt and earlier released ids are so certified inputs of its run, never
advice and never a free input; they count once, in the observations of the run that released them, and the connection
keeps its ordinal, so its cap is one factor over the session.
-/

@[expose] public section

namespace NetTiming

/-- **A release**, as a step's certified ingress records it: the ordinal `q` of the request's connection in the
certifier session (`conn`), its release order `N` (`order`), its prompt's ids and its `max_tokens`. -/
structure Release (Tok : Type) where
  conn : ℕ
  order : ℕ
  prompt : List Tok
  maxTokens : ℕ

/-- **A continuation**, as the reference composes it at an edge from the certified records: the request's connection and
release order, `e` its frames released before the edge (`emitted`), its token row (its prompt, then its released ids)
and what is left of its `max_tokens`. -/
structure Continuation (Tok : Type) where
  conn : ℕ
  order : ℕ
  emitted : ℕ
  row : List Tok
  maxTokens : ℕ

/-- **The ids connection `q` released before step `b`**, in the order released: step by step, link by link and slot by
slot, the frames of the observations before `b` that `dec` reads as `q`'s. -/
def idsBefore {L K S : ℕ} {Tok : Type} (dec : List Bool → Option (ℕ × Tok)) (o : List (Obs L K S)) (b q : ℕ) :
    List Tok :=
  (o.take b).flatMap fun x => (List.finRange L).flatMap fun ℓ =>
    (rowFrames (x.rows ℓ)).filterMap fun p => (dec p).bind fun c => if c.1 = q then some c.2 else none

/-- **The releases before step `b`**: those the certified ingress of the steps before `b` records, in release order. -/
def releasesBefore {In Tok : Type} (rel : In → List (Release Tok)) (x : List In) (b : ℕ) : List (Release Tok) :=
  (x.take b).flatMap rel

/-- **Release `r`'s continuation at the edge `b`**: its ids released before `b` follow its prompt. -/
def continuationAt {L K S : ℕ} {Tok : Type} (dec : List Bool → Option (ℕ × Tok)) (o : List (Obs L K S)) (b : ℕ)
    (r : Release Tok) : Continuation Tok :=
  ⟨r.conn, r.order, (idsBefore dec o b r.conn).length, r.prompt ++ idsBefore dec o b r.conn,
    r.maxTokens - (idsBefore dec o b r.conn).length⟩

/-- **The stamps at the edge `b`** (cut and continue): the continuation of each release before `b` that its ids released
before `b` don't finish (`done`: its first end token, its `max_tokens`, the compaction's last), in release order. -/
def stampsAt {L K S : ℕ} {In Tok : Type} (rel : In → List (Release Tok)) (dec : List Bool → Option (ℕ × Tok))
    (done : Release Tok → List Tok → Bool) (x : List In) (o : List (Obs L K S)) (b : ℕ) : List (Continuation Tok) :=
  ((releasesBefore rel x b).filter fun r => !done r (idsBefore dec o b r.conn)).map (continuationAt dec o b)

/-- **A certified stamp rule** (`ContinuationCertified`): its stamps at each edge `b` are a function of the certified
ingress and of the observations before `b`, which hold the ids `y` released and their timing. Two sessions whose records
agree before `b` get one stamp list at `b`, so a continuation is no advice and its stamps cost no bit. -/
def ContinuationCertified {L K S : ℕ} {In Out : Type} (stamp : List In → List (Obs L K S) → ℕ → Out) : Prop :=
  ∀ (x x' : List In) (o o' : List (Obs L K S)) (b : ℕ), x.take b = x'.take b → o.take b = o'.take b →
    stamp x o b = stamp x' o' b

/-- **The session's ingress as deployed** over its first `n` steps: at each step the base operator's (the new arrivals),
joined at each edge, a positive multiple of the run length `T`, with the stamps the rule computes there from the whole
session's records `x` and `o`, the steps at and after the edge included. -/
def StampedIngress {L K S : ℕ} {In Out : Type} (op : ℕ → List (Obs L K S) → In)
    (stamp : List In → List (Obs L K S) → ℕ → Out) (join : In → Out → In) (T n : ℕ) (x : List In)
    (o : List (Obs L K S)) : Prop :=
  x.length = n ∧ ∀ t (ht : t < x.length),
    x[t] = if 0 < t ∧ T ∣ t then join (op t (o.take t)) (stamp x o t) else op t (o.take t)

/-- **Step `t`'s ingress with continuations**, from the ingress `xs` of the steps before it: the base operator's, joined
at each edge with the stamps there, each read off the records before `t`. -/
def contStep {L K S : ℕ} {In Out : Type} (op : ℕ → List (Obs L K S) → In)
    (stamp : List In → List (Obs L K S) → ℕ → Out) (join : In → Out → In) (T : ℕ) (o : List (Obs L K S))
    (xs : List In) (t : ℕ) : In :=
  if 0 < t ∧ T ∣ t then join (op t (o.take t)) (stamp xs (o.take t) t) else op t (o.take t)

/-- **The session's ingress with continuations** over its first `t` steps, each step's from the steps before it
(`contStep`). -/
def contIngress {L K S : ℕ} {In Out : Type} (op : ℕ → List (Obs L K S) → In)
    (stamp : List In → List (Obs L K S) → ℕ → Out) (join : In → Out → In) (T : ℕ) (o : List (Obs L K S)) :
    ℕ → List In
  | 0 => []
  | t + 1 => contIngress op stamp join T o t ++ [contStep op stamp join T o (contIngress op stamp join T o t) t]

/-- **A run's ingress operator with continuations** (`opCont`), in the chained form: at the run's step `t`, from the
earlier runs' observations `prev` and its own before `t` (`cur`), the session's ingress at its step
`prev.flatten.length + t`. -/
def opCont {L K S : ℕ} {In Out : Type} (op : ℕ → List (Obs L K S) → In)
    (stamp : List In → List (Obs L K S) → ℕ → Out) (join : In → Out → In) (T : ℕ)
    (prev : List (List (Obs L K S))) (t : ℕ) (cur : List (Obs L K S)) : In :=
  contStep op stamp join T (prev.flatten ++ cur)
    (contIngress op stamp join T (prev.flatten ++ cur) (prev.flatten.length + t)) (prev.flatten.length + t)

end NetTiming
