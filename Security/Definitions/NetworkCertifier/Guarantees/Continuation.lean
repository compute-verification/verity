module

public import Definitions.NetworkCertifier.Continuation

/-!
# Continuations' guarantees: certified inputs of the run that continues them, explained by the run that released them

Statements only. Over a session cut into runs of `T` steps whose ingress is deployed (`StampedIngress`) under a certified
stamp rule (`ContinuationCertified`):
* `ContinuationIngress`: a continuation's prompt and earlier released ids are certified inputs of its run. The ingress is
  `contIngress`, and run `i`'s step `t` gets `opCont` of the earlier runs' observations and its own before `t`, the
  chained operator, which the carried pool's count charges nothing for;
* `ExplainedByRun`: each output is explained by the run that released it. Changing `y` at or after a step changes no
  run's ingress before that step, so a run reads of `y` only the ids released before its edge, which the earlier runs'
  egress rows hold and their acceptance explains, and no run reads an output it releases.
-/

@[expose] public section

namespace NetTiming.Guarantees

/-- **A continuation's inputs are certified inputs of its run** (`ContinuationIngress`). Take a session of `m` runs of
`T` steps (`o i`, run `i`'s observations) whose ingress `x` is deployed under a certified stamp rule: each edge's stamps
are recomputed from the whole session's records. Then `x` is `contIngress`, each step's a function of the observations
before it, and run `i`'s step `t` gets `opCont` of the earlier runs' observations and its own before `t`. -/
def ContinuationIngress : Prop :=
  ∀ {L K S : ℕ} {In Out : Type} (op : ℕ → List (Obs L K S) → In) (stamp : List In → List (Obs L K S) → ℕ → Out)
    (join : In → Out → In) (T m : ℕ) (o : Fin m → List (Obs L K S)) (x : List In),
    ContinuationCertified stamp → (∀ i, (o i).length = T) →
    StampedIngress op stamp join T (m * T) x (List.ofFn o).flatten →
    x = contIngress op stamp join T (List.ofFn o).flatten (m * T) ∧
      ∀ (i : Fin m) (t : ℕ), t < T →
        x[i.val * T + t]? = some (opCont op stamp join T ((List.ofFn o).take i) t ((o i).take t))

/-- **Each output is explained by the run that released it** (`ExplainedByRun`): two sessions deployed under one
certified stamp rule and one base operator whose observations agree before step `s` have one ingress before `s`. A
change of `y` at or after `s` so changes no run's ingress before `s`, nor what any run computes before `s`, since a
Program's rows at a step read the ingress through that step. -/
def ExplainedByRun : Prop :=
  ∀ {L K S : ℕ} {In Out : Type} (op : ℕ → List (Obs L K S) → In) (stamp : List In → List (Obs L K S) → ℕ → Out)
    (join : In → Out → In) (T n s : ℕ) (x x' : List In) (o o' : List (Obs L K S)),
    ContinuationCertified stamp → StampedIngress op stamp join T n x o → StampedIngress op stamp join T n x' o' →
    o.take s = o'.take s → x.take s = x'.take s

end NetTiming.Guarantees
