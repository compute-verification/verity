module

public import Definitions.NetworkCertifier.Guarantees.Continuation

/-!
# Continuations' guarantees, proved

The reference's stamp rule reads the releases and the frames before its edge alone (`stampsAt_certified`).
`contIngress` over `t` steps has `t` entries, its prefixes are its shorter runs, and it reads the observations before
step `t` alone (`contIngress_congr`). A deployed session's ingress agrees with it on every prefix
(`take_eq_contIngress`), by induction on the step: at an edge the rule's stamps over the whole session's records are its
stamps over the records before the edge, since the rule is certified. That gives `ContinuationIngress`, with run `i`'s
step `t` read off the observations before it, which are the earlier runs' and run `i`'s own before `t`, and
`ExplainedByRun`, two sessions agreeing before `s` having one `contIngress` before `s`.
-/

@[expose] public section

namespace NetTiming

variable {L K S : ℕ} {In Out : Type}

private theorem take_eq_take_of_le {α : Type} {l l' : List α} {m n : ℕ} (hmn : m ≤ n) (h : l.take n = l'.take n) :
    l.take m = l'.take m := by
  have := congrArg (List.take m) h
  simpa [List.take_take, Nat.min_eq_left hmn, Nat.min_eq_right hmn] using this

/-- **The reference's stamp rule is certified**: `stampsAt` reads the releases and the frames before the edge alone. -/
theorem stampsAt_certified {Tok : Type} (rel : In → List (Release Tok)) (dec : List Bool → Option (ℕ × Tok))
    (done : Release Tok → List Tok → Bool) : ContinuationCertified (L := L) (K := K) (S := S) (stampsAt rel dec done) := by
  intro x x' o o' b hx ho
  have hids : ∀ q, idsBefore dec o b q = idsBefore dec o' b q := fun q => by simp only [idsBefore, ho]
  have hrel : releasesBefore rel x b = releasesBefore rel x' b := by simp only [releasesBefore, hx]
  have hcont : continuationAt dec o b = continuationAt dec o' b := by
    funext r
    simp only [continuationAt, hids]
  simp only [stampsAt, hids, hrel, hcont]

theorem length_contIngress (op : ℕ → List (Obs L K S) → In) (stamp : List In → List (Obs L K S) → ℕ → Out)
    (join : In → Out → In) (T : ℕ) (o : List (Obs L K S)) (t : ℕ) : (contIngress op stamp join T o t).length = t := by
  induction t with
  | zero => rfl
  | succ t ih => simp [contIngress, ih]

theorem contStep_congr (op : ℕ → List (Obs L K S) → In) (stamp : List In → List (Obs L K S) → ℕ → Out)
    (join : In → Out → In) (T : ℕ) {o o' : List (Obs L K S)} (xs : List In) {t : ℕ} (h : o.take t = o'.take t) :
    contStep op stamp join T o xs t = contStep op stamp join T o' xs t := by
  simp only [contStep, h]

/-- **`contIngress` reads the observations before step `t` alone.** -/
theorem contIngress_congr (op : ℕ → List (Obs L K S) → In) (stamp : List In → List (Obs L K S) → ℕ → Out)
    (join : In → Out → In) (T : ℕ) {o o' : List (Obs L K S)} {t : ℕ} (h : o.take t = o'.take t) :
    contIngress op stamp join T o t = contIngress op stamp join T o' t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    have ht : o.take t = o'.take t := take_eq_take_of_le (Nat.le_succ t) h
    simp only [contIngress, ih ht, contStep_congr op stamp join T _ ht]

/-- **A deployed session's ingress is `contIngress` on every prefix**: at an edge, the certified rule's stamps over the
whole session's records are its stamps over the records before the edge. -/
theorem take_eq_contIngress {op : ℕ → List (Obs L K S) → In} {stamp : List In → List (Obs L K S) → ℕ → Out}
    {join : In → Out → In} {T n : ℕ} {x : List In} {o : List (Obs L K S)} (hc : ContinuationCertified stamp)
    (hx : StampedIngress op stamp join T n x o) {t : ℕ} (ht : t ≤ n) : x.take t = contIngress op stamp join T o t := by
  obtain ⟨hlen, hstep⟩ := hx
  induction t with
  | zero => simp [contIngress]
  | succ t ih =>
    have htl : t < x.length := by omega
    have hs : stamp x o t = stamp (x.take t) (o.take t) t :=
      hc _ _ _ _ _ (by simp [List.take_take]) (by simp [List.take_take])
    rw [List.take_add_one, List.getElem?_eq_getElem htl, Option.toList_some]
    simp only [contIngress]
    rw [← ih (by omega), hstep t htl, contStep, hs]

theorem eq_contIngress {op : ℕ → List (Obs L K S) → In} {stamp : List In → List (Obs L K S) → ℕ → Out}
    {join : In → Out → In} {T n : ℕ} {x : List In} {o : List (Obs L K S)} (hc : ContinuationCertified stamp)
    (hx : StampedIngress op stamp join T n x o) : x = contIngress op stamp join T o n := by
  rw [← take_eq_contIngress hc hx (Nat.le_refl n), List.take_of_length_le (Nat.le_of_eq hx.1)]

private theorem length_flatten_of_forall {α : Type} {T : ℕ} :
    ∀ l : List (List α), (∀ a ∈ l, a.length = T) → l.flatten.length = l.length * T
  | [], _ => by simp
  | a :: l, h => by
    rw [List.flatten_cons, List.length_append, h a (by simp),
      length_flatten_of_forall l fun b hb => h b (by simp [hb]), List.length_cons, Nat.add_mul, Nat.one_mul,
      Nat.add_comm]

/-- The earlier runs' observations, `T` steps each, fill the first `i·T` steps. -/
private theorem length_flatten_take {m T : ℕ} (o : Fin m → List (Obs L K S)) (hT : ∀ i, (o i).length = T)
    (i : Fin m) : ((List.ofFn o).take i).flatten.length = i.val * T := by
  have hall : ∀ a ∈ (List.ofFn o).take i, a.length = T := by
    intro a ha
    have : ∃ j, o j = a := by simpa using List.mem_of_mem_take ha
    obtain ⟨j, rfl⟩ := this
    exact hT j
  rw [length_flatten_of_forall _ hall, List.length_take, List.length_ofFn, Nat.min_eq_left i.isLt.le]

/-- The session's observations before run `i`'s step `t`: the earlier runs', then run `i`'s own before `t`. -/
private theorem take_flatten_ofFn {m T : ℕ} (o : Fin m → List (Obs L K S)) (hT : ∀ i, (o i).length = T)
    (i : Fin m) {t : ℕ} (ht : t < T) :
    (List.ofFn o).flatten.take (i.val * T + t) = ((List.ofFn o).take i).flatten ++ (o i).take t := by
  have hi : i.val < (List.ofFn o).length := by simp
  have hsplit : (List.ofFn o).flatten =
      ((List.ofFn o).take i).flatten ++ (o i ++ ((List.ofFn o).drop (i.val + 1)).flatten) := by
    conv_lhs => rw [← List.take_append_drop i.val (List.ofFn o)]
    rw [List.flatten_append, List.drop_eq_getElem_cons hi, List.flatten_cons]
    simp only [List.getElem_ofFn, Fin.eta]
  have hA : ((List.ofFn o).take i).flatten.length ≤ i.val * T + t := by
    rw [length_flatten_take o hT i]
    exact Nat.le_add_right _ _
  have hB : t ≤ (o i).length := by
    rw [hT i]
    exact ht.le
  rw [hsplit, List.take_append, List.take_of_length_le hA, length_flatten_take o hT i,
    Nat.add_sub_cancel_left, List.take_append_of_le_length hB]

theorem continuationIngress : Guarantees.ContinuationIngress := by
  intro L K S In Out op stamp join T m o x hc hT hx
  have hx' := eq_contIngress hc hx
  refine ⟨hx', fun i t ht => ?_⟩
  have h1 : i.val * T + T ≤ m * T := by simpa [Nat.add_mul] using Nat.mul_le_mul_right T i.isLt
  have hs : i.val * T + t < m * T := Nat.lt_of_lt_of_le (Nat.add_lt_add_left ht _) h1
  have hs1 : i.val * T + t + 1 ≤ m * T := hs
  have hxs : x[i.val * T + t]? = some (contStep op stamp join T (List.ofFn o).flatten
      (contIngress op stamp join T (List.ofFn o).flatten (i.val * T + t)) (i.val * T + t)) := by
    rw [← List.getElem?_take_of_lt (Nat.lt_succ_self (i.val * T + t)), take_eq_contIngress hc hx hs1]
    simp [contIngress, length_contIngress]
  have hlen : (((List.ofFn o).take i).flatten ++ (o i).take t).length ≤ i.val * T + t := by
    apply Nat.le_of_eq
    rw [List.length_append, length_flatten_take o hT i, List.length_take, hT i, Nat.min_eq_left ht.le]
  have hQ : (((List.ofFn o).take i).flatten ++ (o i).take t).take (i.val * T + t) =
      (List.ofFn o).flatten.take (i.val * T + t) := by
    rw [take_flatten_ofFn o hT i ht, List.take_of_length_le hlen]
  rw [hxs, opCont, length_flatten_take o hT i, contIngress_congr op stamp join T hQ,
    contStep_congr op stamp join T _ hQ]

theorem explainedByRun : Guarantees.ExplainedByRun := by
  intro L K S In Out op stamp join T n s x x' o o' hc hx hx' ho
  rcases Nat.le_total s n with hs | hs
  · rw [take_eq_contIngress hc hx hs, take_eq_contIngress hc hx' hs, contIngress_congr op stamp join T ho]
  · have hn : o.take n = o'.take n := take_eq_take_of_le hs ho
    rw [eq_contIngress hc hx, eq_contIngress hc hx', contIngress_congr op stamp join T hn]

end NetTiming
