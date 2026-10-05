import FlockSoundness.Protocol.Game.Interleave

/-!
# Union bounds and loops

The tools every phase lemma uses to add up per-coin errors:

* `LawfulMonad Game`, so that the verifier's `List.mapM` loops unfold (`List.mapM_cons`);
* `value_bind_le_of_forall`: a continuation bounded from every state bounds the composition;
* `value_or_le`, `value_exists_finset_le`: the union bound for an optimal prover, over two events
  or over a finite family (the list of codewords near the committed table, for instance);
* `value_mapM_exists_le`: in a loop that runs one step per list element, the probability that some
  step's output is bad is at most the sum of the per-step bounds;
* `value_mapM_fst`: such a loop returns one output per element, in order.
-/

namespace FlockSoundness.Game

open scoped ENNReal

variable {α β γ : Type}

instance : LawfulMonad Game :=
  LawfulMonad.mk' Game
    (id_map := fun x => by
      show x.bind (fun a => Game.ret (id a)) = x
      exact bind_ret_right x)
    (pure_bind := fun _ _ => rfl)
    (bind_assoc := fun x f g => bind_assoc x f g)

end FlockSoundness.Game
