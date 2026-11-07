module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.Claims.Sampling.KS

/-!
# SPOT-Flock's deployed code: the facts its order gives

* `stepWith_fst_take`, `run_fst_take`: a run's prover messages through round `m` read only its coins before round `m`;
  the verifier's answer at round `m` may read round `m`'s coin, the prover's message there doesn't.
* `iroots_fst`: the interior roots a run shows are read off its first two prover messages.
* `interiorFixed`: fresh second-stage coins keep them (`Code.InteriorFixed` at `SpotFlock.code h`): the interior
  registration is the prover's message at round 1, before the second stage's coin.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Finset

/-- A run appends to its transcript: one more round puts the prover's message and the verifier's answer after `t`'s. -/
theorem stepWith_succ_msgs {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext : List Msg → OC Out)
    (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) :
    ∃ ext, (stepWith H P x ω vnext (n + 1) hist as t).msgs =
      t.msgs ++ (((P.next x hist ω).eval H).1, ((vnext (as ++ [((P.next x hist ω).eval H).1])).eval H).1) :: ext := by
  simp only [stepWith]
  split
  · obtain ⟨ext, he⟩ := stepWith_msgs_prefix H P x ω vnext n _ _ _
    exact ⟨ext, by rw [he]; simp⟩
  · exact ⟨[], by simp⟩
  · exact ⟨[], by simp⟩

/-- **The prover's messages through round `m` read only the verifier's answers before it**: two verifiers that answer
alike on at most `m` prover messages give runs whose first `m + 1` prover messages agree. -/
theorem stepWith_fst_take {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (vnext vnext' : List Msg → OC Out) (m : ℕ) (hv : ∀ as : List Msg, as.length ≤ m → vnext as = vnext' as) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), as.length = t.msgs.length →
      t.msgs.length ≤ m →
      ((stepWith H P x ω vnext n hist as t).msgs.map Prod.fst).take (m + 1) =
        ((stepWith H P x ω vnext' n hist as t).msgs.map Prod.fst).take (m + 1) := by
  intro n
  induction n with
  | zero => intro hist as t _ _; rfl
  | succ n ih =>
    intro hist as t hlen hm
    rcases Nat.lt_or_ge t.msgs.length m with hlt | hge
    · have hv' := hv (as ++ [((P.next x hist ω).eval H).1]) (by simp; omega)
      simp only [stepWith]
      rw [hv']
      split
      · exact ih _ _ _ (by simp [hlen]) (by simp; omega)
      · rfl
      · rfl
    · have hm' : t.msgs.length = m := le_antisymm hm hge
      obtain ⟨e, he⟩ := stepWith_succ_msgs H P x ω vnext n hist as t
      obtain ⟨e', he'⟩ := stepWith_succ_msgs H P x ω vnext' n hist as t
      rw [he, he']
      subst hm'
      simp [List.take_append]

/-- **A run's prover messages through round `m` read only its coins before round `m`.** -/
theorem run_fst_take {Coin Ω : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (ω : Ω) (m : ℕ) (r r' : Fin (V.k x) → Coin) (h : ∀ t : Fin (V.k x), t.val < m → r t = r' t) :
    ((run H P V x ω r).msgs.map Prod.fst).take (m + 1) = ((run H P V x ω r').msgs.map Prod.fst).take (m + 1) := by
  unfold run
  refine stepWith_fst_take H P x ω _ _ m (fun as has => ?_) _ _ _ _ rfl (Nat.zero_le _)
  congr 1
  refine List.map_congr_left fun i hi => ?_
  have hi' : i < m := lt_of_lt_of_le (List.mem_range.1 hi) has
  unfold coinSeq
  split
  · exact h _ hi'
  · rfl

/-- The interior roots of replay block `ρ` the round-1 message states, from a run's prover messages. -/
def irootsAt (ms : List Msg) (ρ : ℕ) : List Msg :=
  match ms with
  | _ :: m₁ :: _ => ((interiorsOf m₁).bind fun it => (it.roots.find? (·.1 == ρ)).map (·.2)).getD []
  | _ => []

/-- **The interior roots are read off the first two prover messages.** -/
theorem iroots_fst (x : Msg) (v : Seen) (ρ : ℕ) :
    iroots x v ρ = irootsAt ((v.msgs.map Prod.fst).take 2) ρ := by
  unfold iroots irootsAt
  rcases v with ⟨ms, out⟩
  match ms with
  | [] => rfl
  | [_] => rfl
  | _ :: _ :: _ => rfl

/-- **Interior fixed** at SPOT-Flock's code (`Code.InteriorFixed`): fresh second-stage coins keep the interior roots a
run shows, since the prover states them at round 1, reading only round 0's coin. -/
theorem interiorFixed (h : Held) (H : Msg → Msg) :
    ∀ x (Ω : Type) (P : Prover Ω) ω r d v,
      iroots x (run H P (code h).V x ω (forkOn ((code h).stage2 x) r d)).seen v =
        iroots x (run H P (code h).V x ω r).seen v := by
  intro x Ω P ω r d v
  rw [iroots_fst, iroots_fst]
  congr 1
  refine run_fst_take H P (code h).V x ω 1 _ _ fun t ht => ?_
  have hn : t ∉ (code h).stage2 x := by
    intro hmem
    have hlt : (order h x).first < t := (mem_filter.1 hmem).2
    have : (order h x).first.val = 0 := rfl
    rw [Fin.lt_def, this] at hlt
    omega
  simp [forkOn, hn]

end SpotFlock
