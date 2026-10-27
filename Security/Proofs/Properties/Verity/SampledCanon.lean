import Proofs.Properties.Verity.SampledSim
import Std.Data.String.ToNat
import Proofs.Flock.Claims.Complete.Json.Hex
import Proofs.Flock.Claims.Complete.Json.Parse
import Proofs.Flock.Claims.Bridge.Custody
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.Exec.Retained

/-!
# The device's refusals, read in the play

`Drive.canon` reads the play's tagged releases as the record's blocks. Which releases the play's device serves depends
only on the requests (`PlayDevice.served` reads a release's answers, never its coins), so the releases it serves are
exactly those the reduction reads as served, each with the requests before it (`servesNext`, `tagOfServed`):
`served_play`. Each entry it serves names its request's family and root (`served_req`).

A group's releases, where the log releases its uses one run each and each family's roots are its uses', are the
uses' runs in turn, each release its family and root (`Challenger.runs_blocks`). What v2's acceptance runs of the
challenger's coins is `OneStageZk.roundsWhy_spec`: the draw's use, then session 0's (`OneStageZk.Session0`), the
order and groups it asks of them, every coin a record holds the challenger's, and every later use a later session's
family (`OneStageZk.sessionUses_label`), which no family of session 0's is (`Hello.slotStream_inj`). Each F128 a
record holds reads as its coin's 16 bytes, two big-endian words (`OneStageZk.f128_of_record`), so as the coin's F128
that `coinF128s` reads from its bits, where each byte holds its eight bits as the device writes them
(`f128OfHex_coin`, `bytesOfBits_get`). A session's first use
is its link points, released after the SHA-512 of its record's `root_b_sha512` digests in key order
(`OneStageZk.sessionUses_points`, `OneStageZk.pointsAfter_ok`). Its record names exactly `TABLES` there
(`OneStageZk.roundsOk_tables`), so where each digest is its root's SHA-512 (`Flock.Record.decode`'s S5), that hash is
`LiveHash.pointsHash` of the roots (`OneStageZk.sessionUses_pointsHash`).

Every use a session's record lists is its decoded record's (`OneStageZk.sessionUses_decoded`): its link points, then
each stream's rounds in the record's order, each round released after the SHA-512 of its request, at its stream's
chain over the decoded digests and with its decoded coin count (`OneStageZk.decodedUses`), the record's hex of each
use's coins reading as the decoded F128s one for one. So where `roundCoins` returns each use's release, session 0's
decoded link points and round coins are the F128s `coinF128s` reads from those releases, in the record's order
(`session0_coins`, `decodes_coins`).
-/

open Lean (Json)

namespace SampledProofs.Hello

theorem sep_split : ∀ {a b s t : List Char}, '/' ∉ a → '/' ∉ b → a ++ '/' :: s = b ++ '/' :: t → a = b ∧ s = t
  | [], [], _, _, _, _, h => by simp at h; exact ⟨rfl, h⟩
  | [], c :: b, _, _, _, hb, h => by
      simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
      exact absurd (h.1 ▸ List.mem_cons_self) hb
  | c :: a, [], _, _, ha, _, h => by
      simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
      exact absurd (h.1 ▸ List.mem_cons_self) ha
  | c :: a, d :: b, _, _, ha, hb, h => by
      simp only [List.cons_append, List.cons.injEq] at h
      have := sep_split (fun m => ha (List.mem_cons_of_mem _ m)) (fun m => hb (List.mem_cons_of_mem _ m)) h.2
      exact ⟨by rw [h.1, this.1], this.2⟩

theorem slash_not_mem_toString (i : Nat) : '/' ∉ (toString i).toList := by
  intro h
  rw [Nat.toString_eq_repr, Nat.toList_repr] at h
  have := Nat.isDigit_of_mem_toDigits (by decide) (by decide) h
  revert this; decide

/-- **A slot's families are its own**: `slot/<i>/<s>` names `i` and `s`, since `i`'s digits hold no `/`. -/
theorem slotStream_inj {i j : Nat} {s t : String} : slotStream i s = slotStream j t ↔ i = j ∧ s = t := by
  constructor
  · intro h
    have h' := congrArg String.toList h
    simp only [slotStream, String.toList_append] at h'
    have hs : (toString "/").toList = ['/'] := by decide
    simp only [hs, List.append_assoc, List.singleton_append] at h'
    have h' := List.append_cancel_left h'
    have ⟨h1, h2⟩ := sep_split (slash_not_mem_toString i) (slash_not_mem_toString j) h'
    refine ⟨Nat.repr_injective ?_, ?_⟩
    · rw [← Nat.toString_eq_repr, ← Nat.toString_eq_repr, ← String.ofList_toList (s := toString i), h1,
        String.ofList_toList]
    · rw [← String.ofList_toList (s := s), ← String.ofList_toList (s := t)]
      exact congrArg _ h2
  · rintro ⟨rfl, rfl⟩; rfl

theorem slotStream_ne {i j : Nat} {s t : String} (h : i ≠ j) : slotStream i s ≠ slotStream j t :=
  fun e => h (slotStream_inj.1 e).1

end SampledProofs.Hello

namespace SampledProofs.Challenger

/-- Two lists of labelled values with one label sequence and, label by label, one value sequence are one list. -/
theorem eq_of_labels {β : Type} : ∀ (P Q : List (String × β)), P.map (·.1) = Q.map (·.1) →
    (∀ ℓ, (P.filter (·.1 == ℓ)).map (·.2) = (Q.filter (·.1 == ℓ)).map (·.2)) → P = Q
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | (a, x) :: P, (b, y) :: Q, h, hl => by
    simp only [List.map_cons, List.cons.injEq] at h
    obtain ⟨rfl, h⟩ := h
    have hx := hl a
    simp only [List.filter_cons, beq_self_eq_true, ↓reduceIte, List.map_cons, List.cons.injEq] at hx
    obtain ⟨rfl, -⟩ := hx
    congr 1
    refine eq_of_labels P Q h fun ℓ => ?_
    have := hl ℓ
    by_cases hℓ : (a == ℓ) = true
    · simp only [List.filter_cons, hℓ, ↓reduceIte, List.map_cons, List.cons.injEq] at this
      exact this.2
    · simp only [List.filter_cons, hℓ, Bool.false_eq_true, ↓reduceIte] at this
      exact this

/-- A use's run, each its family and root, read family by family. -/
theorem flat_filter (ℓ : String) : ∀ vs : List Use,
    ((vs.flatMap fun u => List.replicate u.coins (u.label, u.after)).filter (·.1 == ℓ)).map (·.2) =
      (vs.filter (·.label == ℓ)).flatMap fun u => List.replicate u.coins u.after
  | [] => rfl
  | u :: vs => by
    rw [List.flatMap_cons, List.filter_append, List.map_append, flat_filter ℓ vs, List.filter_cons]
    by_cases h : (u.label == ℓ) = true
    · simp [h]
    · simp [h]

variable {X : Type} (lab : X → String) (rt : X → ByteArray)

/-- **A group's releases are its uses' runs**: where the log's releases `xs` of the group `g`'s families are the uses
`vs` one run each (`Session.Runs`) and each family's roots are its uses' (`Use.roots`), the group's releases, each
its family and root, are `vs`'s uses in turn, each its family and root once a coin. -/
theorem runs_blocks (g : List String) (us vs : List Use) (xs : List X)
    (hvs : ∀ ℓ, vs.filter (·.label == ℓ) = us.filter fun u => u.label == ℓ && g.contains ℓ && u.coins != 0)
    (hruns : (xs.map lab).filter g.contains = vs.flatMap fun u => List.replicate u.coins u.label)
    (hroots : ∀ ℓ ∈ g, (xs.filter (lab · == ℓ)).map rt = Use.roots us ℓ) :
    (xs.filter fun x => g.contains (lab x)).map (fun x => (lab x, rt x)) =
      vs.flatMap fun u => List.replicate u.coins (u.label, u.after) := by
  apply eq_of_labels
  · rw [List.map_map, List.map_flatMap]
    have : (xs.map lab).filter g.contains = (xs.filter fun x => g.contains (lab x)).map lab := by
      rw [List.filter_map]; rfl
    have hc : ((fun x : String × ByteArray => x.1) ∘ fun x => (lab x, rt x)) = lab := rfl
    rw [hc, ← this, hruns]
    congr 1
    funext u
    simp
  · intro ℓ
    rw [flat_filter, List.filter_map, List.map_map]
    have hl : ((fun x : String × ByteArray => x.1 == ℓ) ∘ fun x => (lab x, rt x)) = (lab · == ℓ) := rfl
    have hr : ((fun x : String × ByteArray => x.2) ∘ fun x => (lab x, rt x)) = rt := rfl
    rw [hl, hr, List.filter_filter, hvs]
    by_cases hg : g.contains ℓ = true
    · have h1 : (xs.filter fun x => (lab x == ℓ) && g.contains (lab x)) = xs.filter (lab · == ℓ) := by
        apply List.filter_congr
        intro x _
        by_cases hx : (lab x == ℓ) = true
        · rw [beq_iff_eq.1 hx, hg]; simp
        · simp [hx]
      rw [h1, hroots ℓ (by simpa using hg), Use.roots]
      simp only [hg, Bool.and_true]
      clear hvs hruns hroots h1
      induction us with
      | nil => rfl
      | cons u us ih =>
        by_cases h1 : (u.label == ℓ) = true
        · by_cases h2 : u.coins = 0
          · simp [h1, h2, ih]
          · simp [h1, h2, ih]
        · simp [h1, ih]
    · have h1 : (xs.filter fun a => (lab a == ℓ) && g.contains (lab a)) = [] := by
        rw [List.filter_eq_nil_iff]
        intro x _ hx
        simp only [Bool.and_eq_true, beq_iff_eq] at hx
        rw [hx.1] at hx
        exact hg hx.2
      have h2 : (us.filter fun u => u.label == ℓ && g.contains ℓ && u.coins != 0) = [] := by
        rw [List.filter_eq_nil_iff]
        intro u _ hu
        simp only [Bool.and_eq_true] at hu
        exact hg hu.1.2
      rw [h1, h2]
      rfl

/-- **A list whose labels are runs is its runs' blocks**: where `ys` reads, element by element, as each of `vs` in turn
repeated `c` times, `ys` is one block per member of `vs`, in order, each of its count and each element reading as it. -/
theorem unflatten {α β γ : Type} (f : α → β) (c : γ → Nat) (g : γ → β) :
    ∀ (vs : List γ) (ys : List α), ys.map f = vs.flatMap (fun u => List.replicate (c u) (g u)) →
      ∃ bs : List (γ × List α), bs.map Prod.fst = vs ∧ ys = bs.flatMap Prod.snd ∧
        ∀ b ∈ bs, b.2.length = c b.1 ∧ ∀ y ∈ b.2, f y = g b.1
  | [], ys, h => by
    simp only [List.flatMap_nil, List.map_eq_nil_iff] at h
    exact ⟨[], rfl, by simp [h], by simp⟩
  | u :: vs, ys, h => by
    rw [List.flatMap_cons] at h
    have hl : (ys.take (c u)).map f = List.replicate (c u) (g u) := by
      rw [List.map_take, h, List.take_left' (by simp)]
    have hr : (ys.drop (c u)).map f = vs.flatMap (fun u => List.replicate (c u) (g u)) := by
      rw [List.map_drop, h, List.drop_left' (by simp)]
    obtain ⟨bs, h1, h2, h3⟩ := unflatten f c g vs _ hr
    refine ⟨(u, ys.take (c u)) :: bs, by simp [h1], by simp [← h2], ?_⟩
    intro b hb
    rcases List.mem_cons.1 hb with rfl | hb
    · have hlen := congrArg List.length hl
      simp only [List.length_map, List.length_replicate] at hlen
      refine ⟨hlen, fun y hy => ?_⟩
      have : f y ∈ List.replicate (c u) (g u) := hl ▸ List.mem_map_of_mem hy
      exact (List.mem_replicate.1 this).2
    · exact h3 b hb

/-- **Releases of `p` before releases of `q` come first**: where every element `p` holds of is before every element
`q` holds of, and none is both, the elements of either are `p`'s, then `q`'s. -/
theorem filter_or_of_before {α : Type} (p q : α → Bool) (hpq : ∀ x, p x = true → q x = false) :
    ∀ (L : List α), (∀ (i j : Nat) (x y : α), L[i]? = some x → L[j]? = some y → p x = true → q y = true → i < j) →
      L.filter (fun x => p x || q x) = L.filter p ++ L.filter q
  | [], _ => rfl
  | x :: L, h => by
    have h' : ∀ (i j : Nat) (x y : α), L[i]? = some x → L[j]? = some y → p x = true → q y = true → i < j :=
      fun i j a b ha hb hpa hqb => Nat.lt_of_succ_lt_succ (h (i + 1) (j + 1) a b (by simpa using ha) (by simpa using hb) hpa hqb)
    have ih := filter_or_of_before p q hpq L h'
    by_cases hp : p x = true
    · simp [hp, hpq x hp, ih]
    · by_cases hq : q x = true
      · have hnp : L.filter p = [] := by
          rw [List.filter_eq_nil_iff]
          intro a ha hpa
          obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
          have := h (i + 1) 0 L[i] x (by simp [hi]) rfl hpa hq
          omega
        simp [hp, hq, ih, hnp]
      · simp [hp, hq, ih]


/-- Every element `P` holds of is before every element `Q` holds of. -/
def Bef {α : Type} (L : List α) (P Q : α → Prop) : Prop :=
  ∀ (i j : Nat) (x y : α), L[i]? = some x → L[j]? = some y → P x → Q y → i < j

theorem bef_tail {α : Type} {P Q : α → Prop} {x : α} {L : List α} (h : Bef (x :: L) P Q) : Bef L P Q :=
  fun i j a b ha hb hpa hqb => Nat.lt_of_succ_lt_succ (h (i + 1) (j + 1) a b (by simpa using ha) (by simpa using hb) hpa hqb)

/-- **An order survives a filter.** -/
theorem bef_filter {α : Type} {P Q : α → Prop} (r : α → Bool) : ∀ (L : List α), Bef L P Q → Bef (L.filter r) P Q
  | [], _ => fun i j x y hx => by simp at hx
  | x :: L, h => by
    have ih := bef_filter r L (bef_tail h)
    by_cases hr : r x = true
    · rw [List.filter_cons_of_pos hr]
      intro i j a b ha hb hpa hqb
      cases i with
      | zero =>
        cases j with
        | zero =>
          exact absurd (h 0 0 a b (by simpa using ha) (by simpa using hb) hpa hqb) (Nat.lt_irrefl 0)
        | succ j => exact Nat.succ_pos j
      | succ i =>
        cases j with
        | zero =>
          have hb' : x = b := by simpa using hb
          subst hb'
          obtain ⟨k, hk, hka⟩ := List.getElem_of_mem (List.mem_filter.1 (List.mem_of_getElem? (by simpa using ha))).1
          exact absurd (h (k + 1) 0 a x (by simp [hk, hka]) rfl hpa hqb) (Nat.not_lt_zero _)
        | succ j =>
          exact Nat.succ_lt_succ (ih i j a b (by simpa using ha) (by simpa using hb) hpa hqb)
    · rw [List.filter_cons_of_neg hr]
      exact ih

/-- **The runs' first use is the one whose label comes first**: where each use of `vs` has a coin, a use of `vs` is
labelled `a`, and in the runs' labels every `a` is before every other label, `vs` begins with a use labelled `a`. -/
theorem runs_head {γ : Type} (lab : γ → String) (c : γ → Nat) (a : String) :
    ∀ (vs : List γ), (∀ u ∈ vs, c u ≠ 0) → (∃ u ∈ vs, lab u = a) →
      Bef (vs.flatMap fun u => List.replicate (c u) (lab u)) (· = a) (· ≠ a) →
      ∃ u rest, vs = u :: rest ∧ lab u = a
  | [], _, ⟨_, hu, _⟩, _ => absurd hu (List.not_mem_nil)
  | u :: vs, hc, ⟨u', hu', hl'⟩, hb => by
    refine ⟨u, vs, rfl, Classical.byContradiction fun hne => ?_⟩
    have hmem : a ∈ (u :: vs).flatMap fun u => List.replicate (c u) (lab u) :=
      List.mem_flatMap.2 ⟨u', hu', List.mem_replicate.2 ⟨hc u' hu', hl'.symm⟩⟩
    obtain ⟨j, hj, hja⟩ := List.getElem_of_mem hmem
    have h0 : ((u :: vs).flatMap fun u => List.replicate (c u) (lab u))[0]? = some (lab u) := by
      obtain ⟨k, hk⟩ := Nat.exists_eq_succ_of_ne_zero (hc u List.mem_cons_self)
      simp [List.flatMap_cons, hk, List.replicate_succ]
    exact absurd (hb j 0 a (lab u) (by rw [List.getElem?_eq_getElem hj, hja]) h0 rfl hne) (Nat.not_lt_zero _)

end SampledProofs.Challenger

namespace SampledProofs.OneStageZk

/-- A loop over a list in `Except` that returns: its first step returns, and the rest runs from it. -/
theorem loop_cons_ok {α σ ε : Type} {f : α → σ → Except ε (ForInStep σ)} {x : α} {l : List α} {s s' : σ}
    (h : forIn (x :: l) s f = .ok s') :
    ∃ r, f x s = .ok r ∧ match r with
      | .done b => b = s'
      | .yield b => forIn l b f = .ok s' := by
  rw [List.forIn_cons] at h
  cases hx : f x s with
  | error e => rw [hx] at h; cases h
  | ok r =>
    rw [hx] at h
    refine ⟨r, rfl, ?_⟩
    cases r with
    | done b => cases h; rfl
    | yield b => exact h


/-- A loop that only yields returns only where each of its steps returns. -/
theorem loop_each {α ε : Type} {f : α → Unit → Except ε (ForInStep Unit)}
    (hy : ∀ x r, f x () = .ok r → r = .yield ()) :
    ∀ (l : List α) (u : Unit), forIn l () f = .ok u → ∀ x ∈ l, ∃ r, f x () = .ok r
  | [], _, _, x, hx => by cases hx
  | y :: l, u, h, x, hx => by
    obtain ⟨r, hr, hn⟩ := loop_cons_ok h
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨r, hr⟩
    · obtain rfl := hy y r hr
      exact loop_each hy l u (by simpa using hn) x hx

/-- A loop that returns relates its start to its end by any reflexive, transitive relation each step keeps. -/
theorem loop_mono {α σ ε : Type} (R : σ → σ → Prop) (hrefl : ∀ s, R s s)
    (htrans : ∀ a b c, R a b → R b c → R a c) (f : α → σ → Except ε (ForInStep σ))
    (hf : ∀ x s r, f x s = .ok r → R s r.value) :
    ∀ (l : List α) (s s' : σ), forIn l s f = .ok s' → R s s'
  | [], s, s', h => by
    rw [List.forIn_nil] at h
    cases h
    exact hrefl s
  | x :: l, s, s', h => by
    obtain ⟨r, hr, hn⟩ := loop_cons_ok h
    have h1 := hf x s r hr
    cases r with
    | done b => cases hn; exact h1
    | yield b => exact htrans _ _ _ h1 (loop_mono R hrefl htrans f hf l b s' hn)

/-- `l`'s elements, one step each, take `s` to `s'`, each step `P`. -/
inductive Steps {α σ : Type} (P : α → σ → σ → Prop) : List α → σ → σ → Prop
  | nil (s : σ) : Steps P [] s s
  | cons {x : α} {l : List α} {s s₁ s' : σ} : P x s s₁ → Steps P l s₁ s' → Steps P (x :: l) s s'

/-- A loop whose every step that returns yields a state its step relates: its run is those steps. -/
theorem steps_of_forIn {α σ ε : Type} {f : α → σ → Except ε (ForInStep σ)} (P : α → σ → σ → Prop)
    (hf : ∀ x s r, f x s = .ok r → ∃ s₁, r = .yield s₁ ∧ P x s s₁) :
    ∀ (l : List α) (s s' : σ), forIn l s f = .ok s' → Steps P l s s'
  | [], s, s', h => by
    rw [List.forIn_nil] at h
    cases h
    exact .nil s
  | x :: l, s, s', h => by
    rw [List.forIn_cons] at h
    cases hx : f x s with
    | error e => rw [hx] at h; cases h
    | ok r =>
      rw [hx] at h
      obtain ⟨s₁, rfl, hp⟩ := hf x s r hx
      exact .cons hp (steps_of_forIn P hf l s₁ s' h)

/-- A list `mapM` returns relates each element to its value by any relation each call's return implies. -/
theorem list_mapM_rel {α β ε : Type} {f : α → Except ε β} {R : α → β → Prop}
    (hf : ∀ x b, f x = .ok b → R x b) : ∀ (l : List α) (l' : List β), l.mapM f = .ok l' → List.Forall₂ R l l'
  | [], l', h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h; subst h; exact .nil
  | x :: l, l', h => by
    rw [List.mapM_cons] at h
    simp only [bind, Except.bind, pure, Except.pure] at h
    split at h
    · cases h
    · rename_i b hb
      split at h
      · cases h
      · rename_i t ht
        cases h
        exact .cons (hf x b hb) (list_mapM_rel hf l t ht)

/-- What `hexes` reads is a list of strings, as they are. -/
theorem hexes_ok {j : Json} {w : String} {cs : Array String} (h : hexes j w = .ok cs) :
    ∃ xs : Array Json, j = .arr xs ∧ List.Forall₂ (fun x c => x = .str c) xs.toList cs.toList := by
  unfold hexes at h
  split at h
  · rename_i xs
    refine ⟨xs, rfl, ?_⟩
    rw [Array.mapM_eq_mapM_toList] at h
    simp only [Functor.map, Except.map] at h
    split at h
    · cases h
    · rename_i l hl
      cases h
      refine list_mapM_rel (fun x c hx => ?_) _ _ hl
      split at hx
      · cases hx; rfl
      · cases hx
  · cases h

/-- A round's use as `usesOf` reads it, after the stream's chain so far: its digest `d` and its coins `cs`, released
after the SHA-512 of its request at the chain's next value, which the step carries on. -/
def RoundStep (slot : Nat) (name : String) (r : Json)
    (s s' : List (Challenger.Use × Array String) × ByteArray) : Prop :=
  ∃ sha d cs w, getD r "msg_sha512" = .str sha ∧ Flock.unhex sha = .ok d ∧ hexes (getD r "coins") w = .ok cs ∧
    s' = (s.1 ++ [(⟨Hello.slotStream slot name, Flock.Sha512.hash (roundRequest name (Flock.Sha512.hash (s.2 ++ d))
      cs.size), (cs.size + 3) / 4⟩, cs)], Flock.Sha512.hash (s.2 ++ d))

/-- A stream's uses as `usesOf` reads them: its name, its rounds, and their uses from an empty chain. -/
def StreamStep (slot : Nat) (s : Json) (out out' : List (Challenger.Use × Array String)) : Prop :=
  ∃ name rs st, getD s "stream" = .str name ∧ getD s "rounds" = .arr rs ∧
    Steps (RoundStep slot name) rs.toList (out, .empty) st ∧ out' = st.1

/-- **What `usesOf` reads**: the record's link points, released after `pointsAfter` of its link, then each stream's
uses, a step each (`StreamStep`). -/
theorem usesOf_steps {slot : Nat} {record : String} {us : List (Challenger.Use × Array String)}
    (h : usesOf slot record = .ok us) :
    ∃ j ps pts a ss, Flock.JsonParse.parse record = .ok j ∧ getD (getD j "link") "points" = .arr ps ∧
      ps.foldlM (m := V) (fun acc p => return acc ++ (← hexes p "a link point")) #[] = .ok pts ∧
      pointsAfter (getD j "link") = .ok a ∧ getD j "streams" = .arr ss ∧
      Steps (StreamStep slot) ss.toList [(⟨Hello.slotStream slot POINTS, a, (pts.size + 3) / 4⟩, pts)] us := by
  unfold usesOf at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  rename_i _ j hj _ ps hps _ pts hpts _ a ha _ ss hss _ out heq
  cases h
  refine ⟨j, ps, pts, a, ss, ?_, hps, hpts, ha, hss, ?_⟩
  · cases hp : Flock.JsonParse.parse record with
    | error e => rw [hp] at hj; cases hj
    | ok j' => rw [hp] at hj; cases hj; rfl
  rw [← Array.forIn_toList] at heq
  refine steps_of_forIn _ (fun s o r hr => ?_) _ _ _ heq
  repeat' split at hr
  all_goals first
    | (cases hr; done)
    | (simp [throw, throwThe, MonadExceptOf.throw] at *; done)
    | skip
  rename_i name hname _ _ _ rs hrs _ st hst
  cases hr
  refine ⟨_, rfl, name, rs, st, hname, hrs, ?_, rfl⟩
  rw [← Array.forIn_toList] at hst
  refine steps_of_forIn _ (fun r t o hr => ?_) _ _ _ hst
  repeat' split at hr
  all_goals first
    | (cases hr; done)
    | (simp [throw, throwThe, MonadExceptOf.throw] at *; done)
    | skip
  rename_i sha hsha _ _ cs hcs _ d hd
  cases hr
  exact ⟨_, rfl, sha, d, cs, _, hsha, hd, hcs, rfl⟩

/-- What each step keeps, a run keeps. -/
theorem steps_inv {α σ : Type} {P : α → σ → σ → Prop} (I : σ → Prop) (hP : ∀ x s s', P x s s' → I s → I s') :
    ∀ {l : List α} {s s' : σ}, Steps P l s s' → I s → I s'
  | _, _, _, .nil _, h => h
  | _, _, _, .cons hp t, h => steps_inv I hP t (hP _ _ _ hp h)

/-- `usesOf`'s uses grow by uses of the slot's families: each stream's rounds append one use each, after what's read. -/
theorem usesOf_grows {slot : Nat} (I : List (Challenger.Use × Array String) → Prop)
    (hI : ∀ out name u cs, I out → u.label = Hello.slotStream slot name → I (out ++ [(u, cs)]))
    {ss : List Json} {out us : List (Challenger.Use × Array String)} (h : Steps (StreamStep slot) ss out us)
    (hout : I out) : I us :=
  steps_inv I (fun _ _ _ ⟨name, _, st, _, _, hst, e⟩ hs => e ▸ steps_inv (fun st => I st.1)
    (fun _ _ _ ⟨_, _, cs, _, _, _, _, e'⟩ ht => e' ▸ hI _ name _ cs ht rfl) hst hs) h hout

/-- A session's uses are its record's (`usesOf`), and its record's streams and rounds are what they are read from
(`roundsOk`). -/
theorem usesOf_of_sessionUses {slot : Nat} {record : String} {us : List (Challenger.Use × Array String)}
    (h : sessionUses slot record = .ok us) : usesOf slot record = .ok us ∧ roundsOk record = .ok () := by
  unfold sessionUses at h
  cases hu : usesOf slot record with
  | error e => simp [hu, bind, Except.bind] at h
  | ok us' =>
    cases hr : roundsOk record with
    | error e => simp [hu, hr, bind, Except.bind] at h
    | ok u =>
      simp only [hu, hr, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      subst h
      cases u
      exact ⟨rfl, rfl⟩

/-- Every use session `slot` makes of the challenger's coins is of one of its own families, `slot/<slot>/…`. -/
theorem sessionUses_label {slot : Nat} {record : String} {us : List (Challenger.Use × Array String)}
    (h : sessionUses slot record = .ok us) : ∀ x ∈ us, ∃ n, x.1.label = Hello.slotStream slot n := by
  obtain ⟨_, _, _, _, _, _, _, _, _, _, hs⟩ := usesOf_steps (usesOf_of_sessionUses h).1
  refine usesOf_grows (fun l => ∀ x ∈ l, ∃ n, x.1.label = Hello.slotStream slot n)
    (fun out name u cs hout hu x hx => ?_) hs ?_
  · rcases List.mem_append.1 hx with hx | hx
    · exact hout x hx
    · rw [List.mem_singleton.1 hx]; exact ⟨name, hu⟩
  · intro x hx
    rw [List.mem_singleton.1 hx]
    exact ⟨_, rfl⟩


theorem loop_mono_mem {α σ ε : Type} (R : σ → σ → Prop) (hrefl : ∀ s, R s s)
    (htrans : ∀ a b c, R a b → R b c → R a c) (f : α → σ → Except ε (ForInStep σ)) :
    ∀ (l : List α), (∀ x ∈ l, ∀ s r, f x s = .ok r → R s r.value) →
      ∀ (s s' : σ), forIn l s f = .ok s' → R s s'
  | [], _, s, s', h => by
    rw [List.forIn_nil] at h
    cases h
    exact hrefl s
  | x :: l, hf, s, s', h => by
    obtain ⟨r, hr, hn⟩ := loop_cons_ok h
    have h1 := hf x List.mem_cons_self s r hr
    cases r with
    | done b => cases hn; exact h1
    | yield b =>
      exact htrans _ _ _ h1 (loop_mono_mem R hrefl htrans f l (fun y hy => hf y (List.mem_cons_of_mem _ hy)) b s' hn)

theorem mapError_ok {ε ε' α : Type} {f : ε → ε'} {x : Except ε α} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x <;> simp_all [Except.mapError]

/-- The loop's uses only grow, by uses of later sessions' families (`slot/<i>/…`, `i ≥ 1`). -/
def Later (s s' : Array Challenger.Use × Array (Option (String × Array String)) × Array Json ×
    Array (String × String) × Array (List String)) : Prop :=
  ∃ t, s'.1 = s.1 ++ t ∧ ∀ u ∈ t, ∃ i n, 1 ≤ i ∧ u.label = Hello.slotStream i n

theorem later_refl (s) : Later s s := ⟨#[], by simp, by simp⟩

theorem later_trans (a b c) (h1 : Later a b) (h2 : Later b c) : Later a c := by
  obtain ⟨t1, e1, q1⟩ := h1
  obtain ⟨t2, e2, q2⟩ := h2
  refine ⟨t1 ++ t2, by rw [e2, e1, Array.append_assoc], fun u hu => ?_⟩
  rcases Array.mem_append.1 hu with hu | hu
  · exact q1 u hu
  · exact q2 u hu

/-- The loop's state only grows: its uses and wants by appending, its order and groups by adding. -/
def Grows (s s' : Array Challenger.Use × Array (Option (String × Array String)) × Array Json ×
    Array (String × String) × Array (List String)) : Prop :=
  (∃ t, s'.1 = s.1 ++ t) ∧ (∃ t, s'.2.1 = s.2.1 ++ t) ∧ (∀ x ∈ s.2.2.2.1, x ∈ s'.2.2.2.1) ∧
    ∀ g ∈ s.2.2.2.2, g ∈ s'.2.2.2.2

theorem grows_refl (s) : Grows s s := ⟨⟨#[], by simp⟩, ⟨#[], by simp⟩, fun _ h => h, fun _ h => h⟩

theorem grows_trans (a b c) (h1 : Grows a b) (h2 : Grows b c) : Grows a c := by
  obtain ⟨⟨t1, e1⟩, ⟨t2, e2⟩, o1, g1⟩ := h1
  obtain ⟨⟨t3, e3⟩, ⟨t4, e4⟩, o2, g2⟩ := h2
  exact ⟨⟨t1 ++ t3, by rw [e3, e1, Array.append_assoc]⟩, ⟨t2 ++ t4, by rw [e4, e2, Array.append_assoc]⟩,
    fun x hx => o2 x (o1 x hx), fun g hg => g2 g (g1 g hg)⟩

/-- **Session 0's uses, as `roundsWhy` runs them**: its record in the blobs, its uses (`sessionUses 0`), one segment
each; the uses `roundCoins` reads begin with the draw's, then session 0's in order, each with the coins its record
holds; the draw is before each of session 0's families and its link points before its other families (`order`), and
its families are one group (`groups`); every use after them is a later session's (`slot/<i>/…`, `i ≥ 1`). -/
def Session0 (label : String) (draw : Challenger.Use) (bs : Core.Blobs) (s0 : Session)
    (uses : List Challenger.Use) (wants : List (Option (String × Array String))) (order : List (String × String))
    (groups : List (List String)) : Prop :=
  ∃ b record us0, bs.get s0.record s!"session {s0.name}'s record" = .ok b ∧ String.fromUTF8? b = some record ∧
    sessionUses 0 record = .ok us0 ∧ s0.coins.size = us0.length ∧
    uses[0]? = some draw ∧ wants[0]? = some none ∧
    (∀ j u, us0[j]? = some u → uses[j + 1]? = some u.1 ∧ ∃ what, wants[j + 1]? = some (some (what, u.2))) ∧
    (∀ ℓ ∈ (us0.map (·.1.label)).eraseDups, (label, ℓ) ∈ order) ∧
    (∀ ℓ ∈ (us0.map (·.1.label)).eraseDups, ℓ ≠ Hello.slotStream 0 POINTS → (Hello.slotStream 0 POINTS, ℓ) ∈ order) ∧
    (us0.map (·.1.label)).eraseDups ∈ groups ∧
    ∀ u ∈ uses.drop (us0.length + 1), ∃ i n, 1 ≤ i ∧ u.label = Hello.slotStream i n

/-- **The uses `roundsWhy` runs the round coins at**, and their wants, order, groups and coins, before it runs them:
the draw's use after the registration's digest, then each session's (`sessionUses`), with the evidence's coins and
every session's segments after the draw's. -/
def roundsIn (label : String) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Flock.V (Array Challenger.Use × Array (Option (String × Array String)) × Array (String × String) ×
      Array (List String) × Json) := do
  let draw := getD e.base.coins "segment"
  let .arr opened := getD draw "openings" | throw Challenger.SEGMENT_FORM
  let mut uses : Array Challenger.Use := #[{ label, after := c.digest, coins := opened.size }]
  let mut wants : Array (Option (String × Array String)) := #[none]
  let mut segments : Array Json := #[draw]
  let mut order : Array (String × String) := #[]
  let mut groups : Array (List String) := #[]
  for (s, i) in e.sessions.toList.zipIdx do
    let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
      | throw s!"session {s.name}'s record is not UTF-8"
    let us ← (sessionUses i record).mapError (s!"session {s.name}: {·}")
    unless s.coins.size == us.length do
      throw s!"session {s.name}: {s.coins.size} segments for its {us.length} uses of the challenger's coins"
    let points := Hello.slotStream i POINTS
    let fams := (us.map (·.1.label)).eraseDups
    order := order ++ (fams.map (label, ·)).toArray ++ ((fams.filter (· != points)).map (points, ·)).toArray
    groups := groups.push fams
    uses := uses ++ us.toArray.map (·.1)
    wants := wants ++ us.toArray.map fun (u, hs) => some (s!"{s.name}'s {u.label}", hs)
    segments := segments ++ s.coins
  let coins := Json.mkObj [("ledger", getD e.base.coins "ledger"), ("main", getD e.base.coins "main"),
    ("pins", getD e.base.coins "pins"), ("segments", Json.arr segments), ("session", getD e.base.coins "session")]
  return (uses, wants, order, groups, coins)

/-- **`roundsWhy` passes only at the uses `roundsIn` reads**: where it names no reason, v2's parameters name a
challenger key and label and a window root, `roundsIn` reads the uses, wants, order, groups and coins, and
`roundCoins` returns on them under that key on that window. -/
theorem roundsIn_of_roundsWhy {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    (h : roundsWhy p c e bs = none) :
    ∃ key label w uses wants order groups coins ξs, p.base.challenger = some (key, label) ∧
      p.base.verifier.windowRoot = some w ∧ roundsIn label c e bs = .ok (uses, wants, order, groups, coins) ∧
      Challenger.roundCoins key w uses.toList order.toList groups.toList coins = .ok ξs := by
  unfold roundsWhy at h
  split at h
  · rename_i key label w hc hw
    dsimp only at h
    split at h
    · rename_i hr
      unfold roundsIn
      simp only [bind, Except.bind, pure, Except.pure] at hr ⊢
      repeat' split at hr
      all_goals try (cases hr; done)
      rename_i hop _ st hf _ ξs hq _ _ _
      refine ⟨key, label, w, st.1, st.2.1, st.2.2.2.1, st.2.2.2.2, _, ξs, hc, hw, ?_, hq⟩
      simp only [hop]
      split
      · rename_i hx; cases hx.symm.trans hf
      · rename_i hx; cases hx.symm.trans hf; rfl
    · cases h
  · cases h

/-- An invariant of a loop's state that each step keeps holds of the state the loop returns. -/
theorem forIn_except_inv {α σ ε : Type} (I : σ → Prop) {f : α → σ → Except ε (ForInStep σ)}
    (hf : ∀ x s r, I s → f x s = .ok r → I r.value) : ∀ (l : List α) (s s' : σ), I s → forIn l s f = .ok s' → I s'
  | [], s, s', hs, h => by
    simp only [List.forIn_nil, pure, Except.pure] at h
    cases h
    exact hs
  | x :: l, s, s', hs, h => by
    rw [List.forIn_cons] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    · rename_i r hr
      have hI := hf x s r hs hr
      cases r with
      | done b => cases h; exact hI
      | yield b => exact forIn_except_inv I hf l b s' hI h

/-- **`roundsIn`'s first use is the draw's**, released after the registration's digest, and its coins carry the
evidence's, with the draw's segment first. -/
theorem roundsIn_draw {label : String} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {uses : Array Challenger.Use} {wants : Array (Option (String × Array String))} {order : Array (String × String)}
    {groups : Array (List String)} {coins : Json}
    (h : roundsIn label c e bs = .ok (uses, wants, order, groups, coins)) :
    ∃ k segs, uses.toList[0]? = some ⟨label, c.digest, k⟩ ∧ segs[0]? = some (getD e.base.coins "segment") ∧
      coins = Json.mkObj [("ledger", getD e.base.coins "ledger"), ("main", getD e.base.coins "main"),
        ("pins", getD e.base.coins "pins"), ("segments", Json.arr segs), ("session", getD e.base.coins "session")] := by
  unfold roundsIn at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · rename_i opened _
    split at h
    · cases h
    · rename_i st hf
      cases h
      have hI := forIn_except_inv (fun st : Array Challenger.Use × Array (Option (String × Array String)) × Array Json ×
          Array (String × String) × Array (List String) =>
          st.1[0]? = some ⟨label, c.digest, opened.size⟩ ∧ st.2.2.1[0]? = some (getD e.base.coins "segment"))
        (fun x s r hs hr => by
          try simp only at hr
          repeat' split at hr
          all_goals first
            | (cases hr; done)
            | (cases hr; exact hs)
            | (cases hr; simp only [ForInStep.value]; exact ⟨by rw [Array.getElem?_append_left (by
                  have := Array.getElem?_eq_some_iff.1 hs.1; exact this.1)]; exact hs.1,
                by rw [Array.getElem?_append_left (by
                  have := Array.getElem?_eq_some_iff.1 hs.2; exact this.1)]; exact hs.2⟩))
        _ _ _ ⟨rfl, rfl⟩ hf
      refine ⟨opened.size, st.2.2.1, ?_, hI.2, rfl⟩
      rw [Array.getElem?_toList]
      exact hI.1
  · cases h

/-- **What `roundsWhy` runs, where it names no reason**: v2's parameters name a challenger key and label and a window
root, the draw's segment opens `opened`, and `roundCoins` returns `ξs` for its uses, order, groups and coins, the ones
`roundsIn` reads; every coin a session's record holds is the challenger's (`coinsMismatch`, use by use); and session 0's uses are read as
`Session0` says. -/
theorem roundsWhy_spec {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    (h : roundsWhy p c e bs = none) :
    ∃ key label w opened uses wants order groups coins ξs, p.base.challenger = some (key, label) ∧
      p.base.verifier.windowRoot = some w ∧ getD (getD e.base.coins "segment") "openings" = .arr opened ∧
      Challenger.roundCoins key w uses order groups coins = .ok ξs ∧
      roundsIn label c e bs =
        .ok (List.toArray uses, List.toArray wants, List.toArray order, List.toArray groups, coins) ∧
      (∀ x ∈ ξs.zip wants, ∀ what hs, x.2 = some (what, hs) → coinsMismatch hs x.1 = none) ∧
      ∀ s0 rest, e.sessions.toList = s0 :: rest →
        Session0 label ⟨label, c.digest, opened.size⟩ bs s0 uses wants order groups := by
  unfold roundsWhy at h
  split at h
  · rename_i key label w hc hw
    dsimp only at h
    split at h
    · rename_i hr
      simp only [bind, Except.bind, pure, Except.pure] at hr
      split at hr
      · rename_i opened hop
        split at hr
        · cases hr
        · rename_i v hfor
          split at hr
          · cases hr
          · rename_i ξs hrc
            split at hr
            · cases hr
            · rename_i u hchk
              refine ⟨key, label, w, opened, v.1.toList, v.2.1.toList, v.2.2.2.1.toList, v.2.2.2.2.toList, _, ξs, hc, hw, hop, hrc,
                ?_, ?_, ?_⟩
              · unfold roundsIn
                simp only [bind, Except.bind, pure, Except.pure, hop]
                split
                · rename_i hx; cases hx.symm.trans hfor
                · rename_i hx; cases hx.symm.trans hfor; rfl
              · intro x hx what hs hxs
                obtain ⟨r, hr⟩ := loop_each (fun x r hr => by
                    repeat' split at hr
                    all_goals first | (cases hr; rfl) | cases hr | skip
                    ) _ u hchk x hx
                cases hm : coinsMismatch hs x.fst with
                | none => rfl
                | some why =>
                  exfalso
                  simp only [hxs, hm] at hr
                  cases hr
              · intro s0 rest hs0
                rw [hs0, List.zipIdx_cons] at hfor
                obtain ⟨r, hr0, hn⟩ := loop_cons_ok hfor
                dsimp only at hr0
                split at hr0
                · cases hr0
                · rename_i b hb
                  split at hr0
                  · rename_i record hrec
                    split at hr0
                    · cases hr0
                    · rename_i us0 hus
                      split at hr0
                      · rename_i hsz
                        have hus' : sessionUses 0 record = .ok us0 := by
                          cases hq : sessionUses 0 record with
                          | error e => rw [hq] at hus; cases hus
                          | ok y => rw [hq] at hus; cases hus; rfl
                        obtain rfl := Except.ok.inj hr0
                        have hg := loop_mono Grows grows_refl grows_trans _ (fun x s r hx => by
                          repeat' split at hx
                          all_goals first
                            | (cases hx; done)
                            | (rename_i heq; cases heq; done)
                            | (cases hx; exact grows_refl _)
                            | (cases hx; exact ⟨⟨_, rfl⟩, ⟨_, rfl⟩,
                                fun _ h => Array.mem_append_left _ (Array.mem_append_left _ h),
                                fun _ h => Array.mem_push_of_mem _ h⟩)) _ _ _ hn
                        obtain ⟨⟨t1, e1⟩, ⟨t2, e2⟩, ho, hgr⟩ := hg
                        have hlt := loop_mono_mem Later later_refl later_trans _ _ (fun x hx s r hr => by
                          obtain ⟨s', i⟩ := x
                          have hi := (List.mem_zipIdx hx).1
                          repeat' split at hr
                          all_goals first
                            | (cases hr; done)
                            | (rename_i heq; cases heq; done)
                            | (cases hr; exact later_refl _)
                            | skip
                          rename_i rec _ _ us heq _
                          cases hr
                          have hu := mapError_ok heq
                          refine ⟨_, rfl, fun u hu' => ?_⟩
                          obtain ⟨y, hy, rfl⟩ := Array.mem_map.1 hu'
                          obtain ⟨n, hn⟩ := sessionUses_label hu y (by simpa using hy)
                          exact ⟨i, n, by omega, hn⟩) _ _ hn
                        refine ⟨b, record, us0, hb, hrec, hus', beq_iff_eq.1 hsz, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
                        · rw [e1]; simp
                        · rw [e2]; simp
                        · intro j u hu
                          have hj : j < us0.length := (List.getElem?_eq_some_iff.1 hu).1
                          refine ⟨?_, toString s0.name ++ toString "'s " ++ toString u.1.label, ?_⟩
                          · rw [e1]
                            simp [List.getElem?_append_left (by simpa using hj : j < (us0.map (·.1)).length), hu]
                          · rw [e2]
                            simp only [Array.toList_append, Array.toList_map, List.cons_append,
                              List.getElem?_cons_succ]
                            rw [List.getElem?_append_left (by simpa using hj)]
                            simp only [List.nil_append, List.getElem?_map, hu, Option.map_some]
                        · intro ℓ hℓ
                          apply Array.mem_toList_iff.2
                          apply ho
                          simp only [Array.mem_append, List.mem_toArray, List.mem_map]
                          exact Or.inl (Or.inr ⟨ℓ, hℓ, rfl⟩)
                        · intro ℓ hℓ hne
                          apply Array.mem_toList_iff.2
                          apply ho
                          simp only [Array.mem_append, List.mem_toArray, List.mem_map, List.mem_filter]
                          exact Or.inr ⟨ℓ, ⟨hℓ, by simpa using hne⟩, rfl⟩
                        · apply Array.mem_toList_iff.2
                          apply hgr
                          simp
                        · obtain ⟨t, et, qt⟩ := hlt
                          intro u hu
                          rw [et] at hu
                          simp at hu
                          exact qt u (by simpa using hu)
                      · split at hr0
                        · cases hr0
                        · rename_i heq
                          cases heq
                  · split at hr0
                    · cases hr0
                    · rename_i heq
                      cases heq
      · cases hr
    · cases h
  · cases h

/-- Eight bytes of `b` from `o`, most significant first: the word `Flock.f128OfHex` reads there. -/
def beWord (b : ByteArray) (o : Nat) : UInt64 :=
  (List.range' 0 8).foldl (fun s i => s <<< 8 ||| (b.get! (o + i)).toUInt64) 0

/-- Bytes `o + i` of `b`, `i` in `l`, read after `acc` most significant first, as a number. -/
def beNat (b : ByteArray) (o : Nat) (l : List Nat) (acc : Nat) : Nat :=
  l.foldl (fun acc i => acc * 256 + (b.get! (o + i)).toNat) acc

theorem foldl_word (b : ByteArray) (o : Nat) : ∀ (l : List Nat) (s : UInt64) (m : Nat),
    s.toNat < 256 ^ m → m + l.length ≤ 8 →
    (l.foldl (fun (s : UInt64) i => s <<< 8 ||| (b.get! (o + i)).toUInt64) s).toNat = beNat b o l s.toNat
  | [], s, _, _, _ => rfl
  | i :: l, s, m, h1, h2 => by
    rw [List.foldl_cons]
    have hb := (b.get! (o + i)).toNat_lt
    have hm : s.toNat * 256 + (b.get! (o + i)).toNat < 256 ^ (m + 1) := by rw [Nat.pow_succ]; omega
    have h8 : 256 ^ (m + 1) ≤ 2 ^ 64 :=
      Nat.le_trans (Nat.pow_le_pow_right (n := 256) (by decide) (by simp at h2; omega)) (by decide : 256 ^ 8 ≤ 2 ^ 64)
    have hs : (s <<< 8 ||| (b.get! (o + i)).toUInt64).toNat = s.toNat * 256 + (b.get! (o + i)).toNat := by
      rw [FlockComplete.JsonRT.shift_or]
      exact Nat.mod_eq_of_lt (by omega)
    rw [foldl_word b o l _ (m + 1) (by rw [hs]; exact hm) (by simp at h2; omega), hs]
    rfl

/-- **`f128OfHex`'s word, as a number**: the eight bytes from `o`, most significant first. -/
theorem beWord_toNat (b : ByteArray) (o : Nat) : (beWord b o).toNat = beNat b o (List.range' 0 8) 0 := by
  unfold beWord
  rw [foldl_word b o _ 0 0 (by decide) (by decide)]
  rfl

/-- **Sixteen bytes' hex reads back as an F128**: `hi` the first eight bytes, `lo` the next eight, each big-endian. -/
theorem f128OfHex_hexOf (b : ByteArray) (h : b.size = 16) :
    Flock.f128OfHex (Flock.hexOf b) = .ok ⟨beWord b 8, beWord b 0⟩ := by
  unfold Flock.f128OfHex
  rw [FlockComplete.JsonRT.unhex_hexOf]
  simp only [bind, Except.bind, h, bne_self_eq_false, Bool.false_eq_true, ite_false,
    Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
  rfl

/-- Where `coinsMismatch` finds no reason, each F128 the record wants is the hex of its 16 bytes of the coins. -/
theorem coinsMismatch_none {hs : Array String} {ξ : ByteArray} (h : coinsMismatch hs ξ = none) (i : Nat)
    (hi : i < hs.size) : Flock.hexOf (ξ.extract (16 * i) (16 * i + 16)) = hs[i] := by
  unfold coinsMismatch at h
  rw [Option.map_eq_none_iff, List.find?_eq_none] at h
  have := h (hs[i], i) (List.mem_zipIdx_iff_getElem?.2 (by simp [hi]))
  simpa using this

/-- **A record's coin is the challenger's F128**: where `coinsMismatch` finds no reason, the record's `i`th F128 reads
as the coins' bytes `16 i` to `16 i + 15`. -/
theorem f128_of_record {hs : Array String} {ξ : ByteArray} (h : coinsMismatch hs ξ = none) (i : Nat)
    (hi : i < hs.size) (hξ : 16 * i + 16 ≤ ξ.size) :
    Flock.f128OfHex hs[i] =
      .ok ⟨beWord (ξ.extract (16 * i) (16 * i + 16)) 8, beWord (ξ.extract (16 * i) (16 * i + 16)) 0⟩ := by
  rw [← coinsMismatch_none h i hi]
  exact f128OfHex_hexOf _ (by simp only [ByteArray.size_extract]; omega)

theorem list_mapM_ok {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {ys : List β}, l.mapM f = .ok ys →
      ys.length = l.length ∧ ∀ i (hi : i < l.length) (hy : i < ys.length), f l[i] = .ok ys[i]
  | [], ys, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; exact ⟨rfl, fun i hi => absurd hi (Nat.not_lt_zero _)⟩
  | x :: l, ys, h => by
    rw [List.mapM_cons] at h
    cases hx : f x with
    | error e => rw [hx] at h; cases h
    | ok y =>
      rw [hx] at h
      cases hl : l.mapM f with
      | error e => rw [hl] at h; cases h
      | ok zs =>
        rw [hl] at h
        cases h
        obtain ⟨hlen, hall⟩ := list_mapM_ok hl
        refine ⟨by simp [hlen], fun i hi hy => ?_⟩
        cases i with
        | zero => exact hx
        | succ i => exact hall i (by simpa using hi) (by simpa using hy)

theorem array_mapM_ok {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : ys.size = xs.size ∧ ∀ i (hi : i < xs.size) (hy : i < ys.size), f xs[i] = .ok ys[i] := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok zs =>
    rw [hl] at h
    cases h
    obtain ⟨hlen, hall⟩ := list_mapM_ok hl
    refine ⟨by simpa using hlen, fun i hi hy => ?_⟩
    have := hall i (by simpa using hi) (by simpa using hy)
    simpa using this

theorem match_ok {ε α β : Type} {X : Except ε α} {g : α → β} {a : β}
    (h : (match X with | .error e => .error e | .ok v => .ok (g v) : Except ε β) = .ok a) :
    ∃ v, X = .ok v ∧ a = g v := by
  cases X with
  | error e => cases h
  | ok v => cases h; exact ⟨v, rfl, rfl⟩

/-- **What the link points are released after**: `pointsAfter` of a record's link is the SHA-512 of the digests its
`root_b_sha512` holds in key order, each 64 bytes of hex. -/
theorem pointsAfter_ok {link : Json} {a : ByteArray} (h : pointsAfter link = .ok a) :
    ∃ ds : Array ByteArray, ds.size = (entries (getD link "root_b_sha512")).size ∧
      (∀ i (hi : i < (entries (getD link "root_b_sha512")).size) (hd : i < ds.size), ∃ s,
        (entries (getD link "root_b_sha512"))[i].2 = .str s ∧ s.length = 128 ∧ Flock.unhex s = .ok ds[i]) ∧
      a = Flock.Sha512.hash (ds.foldl (· ++ ·) .empty) := by
  unfold pointsAfter at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · obtain ⟨ds, hds, rfl⟩ := match_ok h
    obtain ⟨hsz, hall⟩ := array_mapM_ok hds
    refine ⟨ds, hsz, fun i hi hd => ?_, rfl⟩
    have he := hall i hi hd
    generalize (entries (getD link "root_b_sha512"))[i] = e at he ⊢
    obtain ⟨t, x⟩ := e
    cases x with
    | str s =>
      dsimp only at he
      split at he
      · rename_i hlen
        exact ⟨s, rfl, beq_iff_eq.1 hlen, he⟩
      · cases he
    | _ => cases he
  · cases h

/-- **A session's first use is its link points**, released after `pointsAfter` of its record's link. -/
theorem sessionUses_points {slot : Nat} {record : String} {us : List (Challenger.Use × Array String)}
    (h : sessionUses slot record = .ok us) :
    ∃ j a pts, Flock.JsonParse.parse record = .ok j ∧ pointsAfter (getD j "link") = .ok a ∧
      us.head? = some (⟨Hello.slotStream slot POINTS, a, (pts.size + 3) / 4⟩, pts) := by
  obtain ⟨j, _, pts, a, _, hj, _, _, ha, _, hs⟩ := usesOf_steps (usesOf_of_sessionUses h).1
  refine ⟨j, a, pts, hj, ha, usesOf_grows (fun l => l.head? = some (⟨Hello.slotStream slot POINTS, a,
    (pts.size + 3) / 4⟩, pts)) (fun out _ _ _ hout _ => ?_) hs rfl⟩
  cases out with
  | nil => cases hout
  | cons x t => exact hout


/-- **A record `roundsOk` passes names exactly `TABLES`** in its link's root_B digests, in key order. -/
theorem roundsOk_tables {record : String} (h : roundsOk record = .ok ()) :
    ∃ j, Flock.JsonParse.parse record = .ok j ∧
      (entries (getD (getD j "link") "root_b_sha512")).map (·.1) = TABLES := by
  unfold roundsOk at h
  cases hj : Flock.JsonParse.parse record with
  | error e =>
    rw [hj] at h
    first
      | (cases h; done)
      | (simp [Except.mapError, bind, Except.bind] at h; done)
  | ok j =>
    refine ⟨j, rfl, ?_⟩
    rw [hj] at h
    cases hb : ((entries (getD (getD j "link") "root_b_sha512")).map (·.1) == TABLES) with
    | true => exact beq_iff_eq.1 hb
    | false =>
      first
        | (simp [Except.mapError, bind, Except.bind, hb, throw, throwThe, MonadExceptOf.throw] at h; done)
        | (simp only [Except.mapError, bind, Except.bind, hb, Bool.false_eq_true, if_false, ite_false, throw,
            throwThe, MonadExceptOf.throw] at h; cases h)

theorem foldl_append_data (L : List ByteArray) : ∀ acc : ByteArray,
    (L.foldl (· ++ ·) acc).data.toList = acc.data.toList ++ (L.map fun b => b.data.toList).flatten := by
  induction L with
  | nil => intro acc; simp
  | cons b L ih =>
      intro acc
      rw [List.foldl_cons, ih, ByteArray.data_append, Array.toList_append]
      simp

/-- The digests of the roots, concatenated, are `LiveHash.pointsHash`'s preimage. -/
theorem foldl_digests (rb : Array ByteArray) :
    (rb.map Flock.Sha512.hash).foldl (· ++ ·) .empty =
      ⟨(Flock.Guarantees.LiveHash.rootDigests rb).toArray⟩ := by
  apply ByteArray.ext
  apply Array.toList_inj.mp
  rw [← Array.foldl_toList, foldl_append_data]
  simp [Flock.Guarantees.LiveHash.rootDigests, List.map_map, Function.comp_def]

/-- A member of a parsed value is well formed. -/
theorem wf_getD {j : Json} (hw : _root_.Claims.Json.Wf j) (k : String) : _root_.Claims.Json.Wf (getD j k) := by
  cases hw with
  | obj hm hv =>
    rename_i m
    cases hg : m.get? k with
    | none =>
      have e : getD (Json.obj m) k = Json.null := by
        simp only [getD, field?, Json.getObjVal?, hg]
        try rfl
      rw [e]
      exact .null
    | some v =>
      have e : getD (Json.obj m) k = v := by
        simp only [getD, field?, Json.getObjVal?, hg]
        try rfl
      rw [e]
      exact hv k v hg
  | _ =>
    first
      | exact .null
      | (simp only [getD, field?, Json.getObjVal?]; exact .null)

theorem str_of_get {m : Std.TreeMap.Raw String Json compare} {k s x : String}
    (hg : m.get? k = some (.str s)) (h : Flock.Record.str (.obj m) k = .ok x) : s = x := by
  unfold Flock.Record.str Flock.Record.fld at h
  simp only [Json.getObjVal?] at h
  rw [hg] at h
  first
    | (simp only [bind, Except.bind, pure, Except.pure, Json.getStr?, Except.ok.injEq] at h; exact h)
    | (simp [bind, Except.bind, pure, Except.pure, Json.getStr?] at h; exact h)
    | (cases h; rfl)

/-- **The link points' hash is the roots'**: where a parsed link names the tables `ts` in its root_B digests, in key
order (`roundsOk_tables`), and each table's digest is the hex of its root's SHA-512 (`Flock.Record.decode`'s S5),
`pointsAfter` of the link is `LiveHash.pointsHash` of the roots. -/
theorem pointsAfter_eq_pointsHash {link : Json} {a : ByteArray} {ts : Array String} {rb : Array ByteArray}
    (hw : _root_.Claims.Json.Wf link) (ha : pointsAfter link = .ok a)
    (hk : (entries (getD link "root_b_sha512")).map (·.1) = ts) (hsz : rb.size = ts.size)
    (h5 : ∀ i (hi : i < ts.size) (hr : i < rb.size),
      Flock.Record.str (getD link "root_b_sha512") ts[i] = .ok (Flock.hexOf (Flock.Sha512.hash rb[i]))) :
    a = Flock.Guarantees.LiveHash.pointsHash rb := by
  subst hk
  obtain ⟨ds, hds, hall, rfl⟩ := pointsAfter_ok ha
  have hwv := wf_getD hw "root_b_sha512"
  generalize getD link "root_b_sha512" = v at hwv hds hall h5 hsz
  have hd : ds = rb.map Flock.Sha512.hash := by
    apply Array.ext
    · rw [Array.size_map, hds, hsz, Array.size_map]
    · intro i hi1 hi2
      have hie : i < (entries v).size := by omega
      have hir : i < rb.size := by rw [Array.size_map] at hi2; exact hi2
      obtain ⟨s, hs, -, hu⟩ := hall i hie hi1
      have h5i := h5 i (by rw [Array.size_map]; exact hie) hir
      simp only [Array.getElem_map] at h5i
      cases v with
      | obj m =>
        cases hwv with
        | obj hm _ =>
          have hmem : (entries (Json.obj m))[i] ∈ m.toArray := Array.getElem_mem hie
          rw [show (entries (Json.obj m))[i] = ((entries (Json.obj m))[i].1, Json.str s) from
            Prod.ext rfl hs] at hmem
          have hg : m.get? (entries (Json.obj m))[i].1 = some (.str s) := by
            rw [Std.TreeMap.Raw.get?_eq_getElem?]
            exact (Std.TreeMap.Raw.mem_toArray_iff_getElem?_eq_some hm).1 hmem
          have e := str_of_get hg h5i
          subst e
          rw [FlockComplete.JsonRT.unhex_hexOf] at hu
          rw [Array.getElem_map]
          exact (Except.ok.inj hu).symm
      | _ => simp [entries] at hie
  rw [hd, foldl_digests]
  try rfl

/-- **A session's link points are released after `pointsHash` of its roots**: where its record's table digests
(`TABLES`, the only tables `roundsOk` lets it name) are the hex of the roots' SHA-512 (`Flock.Record.decode`'s S5). -/
theorem sessionUses_pointsHash {slot : Nat} {record : String} {us : List (Challenger.Use × Array String)}
    (h : sessionUses slot record = .ok us) {rb : Array ByteArray} (hsz : rb.size = TABLES.size)
    (h5 : ∀ j, Flock.JsonParse.parse record = .ok j → ∀ i (hi : i < TABLES.size) (hr : i < rb.size),
      Flock.Record.str (getD (getD j "link") "root_b_sha512") TABLES[i] =
        .ok (Flock.hexOf (Flock.Sha512.hash rb[i]))) :
    ∃ pts, us.head? =
      some (⟨Hello.slotStream slot POINTS, Flock.Guarantees.LiveHash.pointsHash rb, (pts.size + 3) / 4⟩, pts) := by
  obtain ⟨j, a, pts, hj, ha, hhd⟩ := sessionUses_points h
  obtain ⟨j', hj', hk⟩ := roundsOk_tables (usesOf_of_sessionUses h).2
  rw [hj] at hj'
  cases hj'
  have hw := wf_getD (FlockComplete.JsonRT.parse_wf record j hj) "link"
  rw [← pointsAfter_eq_pointsHash hw ha hk hsz (h5 j hj)]
  exact ⟨pts, hhd⟩

theorem getD_of_fld {j l : Json} {k : String} (h : Flock.Record.fld j k = .ok l) : getD j k = l := by
  unfold Flock.Record.fld at h
  unfold getD field?
  cases hv : j.getObjVal? k with
  | error e => rw [hv] at h; cases h
  | ok v => rw [hv] at h; cases h; rfl

/-- **`Record.decode`'s S5**: the link of a record it decodes holds, under `root_b_<dk>`, each table's digest as the
hex of the `dg` of the table's decoded root. -/
theorem decode_rootDigests {spec : Flock.SessionSpec} {text : String} {sess : Flock.Session}
    (h : Flock.Record.decode spec text = .ok sess) :
    ∃ j l, Flock.JsonParse.parse text = .ok j ∧ Flock.Record.fld j "link" = .ok l ∧
      List.Forall₂ (fun t rb => ∃ v, Flock.Record.fld l s!"root_b_{spec.dk}" = .ok v ∧
        Flock.Record.str v t = .ok (Flock.hexOf (spec.dg rb))) spec.tables.toList sess.rootB.toList := by
  revert sess
  show FlockSoundness.Refine.Ret (Flock.Record.decode spec text) _
  rw [Flock.Record.decode.eq_def]
  cases hp : Flock.JsonParse.parse text with
  | error e => intro sess h; cases h
  | ok j =>
    simp only []
    rw [pure_bind]
    ret_walk [Flock.Record.fld, Flock.Record.arr, Array.mapM]
    all_goals
      refine ⟨j, _, rfl, by assumption, ?_⟩
      refine List.Forall₂.imp (fun t rb h => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))
      beta_reduce at h
      obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨v5, h5a, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨s5, h5b, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨_, h5c, h⟩ := FlockSoundness.Refine.ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      have e := beq_iff_eq.1 (FlockSoundness.Refine.check_ok_iff.1 h5c)
      subst e
      exact ⟨v5, h5a, h5b⟩

/-- **A session's link points are released after `pointsHash` of the roots `Record.decode` reads** from its record,
at a statement of the tables `TABLES` with SHA-512 digests. -/
theorem sessionUses_pointsHash_of_decode {slot : Nat} {record : String} {us : List (Challenger.Use × Array String)}
    (h : sessionUses slot record = .ok us) {spec : Flock.SessionSpec} {sess : Flock.Session}
    (hd : Flock.Record.decode spec record = .ok sess) (ht : spec.tables = TABLES) (hs : spec.sha512Digests = true) :
    ∃ pts, us.head? = some
      (⟨Hello.slotStream slot POINTS, Flock.Guarantees.LiveHash.pointsHash sess.rootB, (pts.size + 3) / 4⟩, pts) := by
  obtain ⟨j, l, hj, hl, hf⟩ := decode_rootDigests hd
  have hlen := hf.length_eq
  simp only [Array.length_toList] at hlen
  refine sessionUses_pointsHash h (by rw [← hlen, ht]) fun j' hj' i hi hr => ?_
  rw [hj] at hj'
  cases hj'
  rw [getD_of_fld hl]
  obtain ⟨v, hv, hstr⟩ := (List.forall₂_iff_get.1 hf).2 i (by simpa [ht] using hi) (by simpa using hr)
  have hk : s!"root_b_{spec.dk}" = "root_b_sha512" := by
    rw [show spec.dk = "sha512" by simp [Flock.SessionSpec.dk, hs]]; rfl
  have hg : ∀ b, spec.dg b = Flock.Sha512.hash b := fun b => by simp [Flock.SessionSpec.dg, hs]
  rw [hk] at hv
  rw [getD_of_fld hv]
  simp only [List.get_eq_getElem, Array.getElem_toList, hg] at hstr
  first
    | (simpa [ht] using hstr)
    | (simp only [ht] at hstr; exact hstr)

/-- A round as `Flock.Record.decode` reads it from its record: its digest under `msg_<dk>` and its coins. -/
def RoundDec (spec : Flock.SessionSpec) (r : Json) (rd : Flock.Round) : Prop :=
  ∃ h, Flock.Record.str r s!"msg_{spec.dk}" = .ok h ∧ Flock.unhex h = .ok rd.sha ∧
    ∃ cs, Flock.Record.arr r "coins" = .ok cs ∧ Flock.Record.f128s cs = .ok rd.coins

/-- A stream as `Flock.Record.decode` reads it from its record: its name and its rounds, in order. -/
def StreamDec (spec : Flock.SessionSpec) (s : Json) (sr : Flock.StreamRec) : Prop :=
  Flock.Record.str s "stream" = .ok sr.name ∧
    ∃ rs, Flock.Record.arr s "rounds" = .ok rs ∧ List.Forall₂ (RoundDec spec) rs.toList sr.rounds.toList

/-- **What `Flock.Record.decode` reads of the link points and the streams**: each point list's F128s, and each
stream's name and rounds in the record's order (`StreamDec`). -/
theorem decode_fields {spec : Flock.SessionSpec} {text : String} {sess : Flock.Session}
    (h : Flock.Record.decode spec text = .ok sess) :
    ∃ j l, Flock.JsonParse.parse text = .ok j ∧ Flock.Record.fld j "link" = .ok l ∧
      (∃ ps, Flock.Record.arr l "points" = .ok ps ∧
        List.Forall₂ (fun p pt => ∃ a, p.getArr? = .ok a ∧ Flock.Record.f128s a = .ok pt) ps.toList sess.points.toList) ∧
      ∃ ss, Flock.Record.arr j "streams" = .ok ss ∧ List.Forall₂ (StreamDec spec) ss.toList sess.streams.toList := by
  revert sess
  show FlockSoundness.Refine.Ret (Flock.Record.decode spec text) _
  rw [Flock.Record.decode.eq_def]
  cases hp : Flock.JsonParse.parse text with
  | error e => intro sess h; cases h
  | ok j =>
    simp only []
    rw [pure_bind]
    ret_walk [Flock.Record.fld, Flock.Record.arr, Array.mapM]
    all_goals
      refine ⟨j, _, rfl, by assumption, ⟨_, by assumption,
        (Claims.Bridge.Custody.mapM_forall₂ (by assumption)).imp fun p pt h => ?_⟩,
        ⟨_, by assumption, (Claims.Bridge.Custody.mapM_forall₂ (by assumption)).imp fun s sr h => ?_⟩⟩
    all_goals try dsimp only at h
    all_goals first
      | (split at h
         · exact ⟨_, by assumption, h⟩
         · cases h)
      | skip
    all_goals
      obtain ⟨name, hn, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨rs, hrs, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨rounds, hro, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
      obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      refine ⟨hn, rs, hrs, (Claims.Bridge.Custody.mapIdxM_forall₂ hro).imp fun r rd ⟨k, h⟩ => ?_⟩
      repeat' walk_step
      all_goals
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact ⟨_, by assumption, by assumption, _, by assumption, by assumption⟩


theorem getD_of_str {j : Json} {k x : String} (h : Flock.Record.str j k = .ok x) : getD j k = .str x := by
  unfold Flock.Record.str at h
  obtain ⟨v, hv, h⟩ := FlockSoundness.Refine.ebind_ok h
  rw [getD_of_fld hv]
  cases v <;> first | (cases h; done) | (simp only [Json.getStr?] at h; cases h; rfl)

theorem getD_of_arr {j : Json} {k : String} {a : Array Json} (h : Flock.Record.arr j k = .ok a) :
    getD j k = .arr a := by
  unfold Flock.Record.arr at h
  obtain ⟨v, hv, h⟩ := FlockSoundness.Refine.ebind_ok h
  rw [getD_of_fld hv]
  cases v <;> first | (cases h; done) | (simp only [Json.getArr?] at h; cases h; rfl)

/-- The record's F128 hex `hs` reads as the F128s `xs`, one for one. -/
def Decodes (hs : Array String) (xs : Array Flock.F128) : Prop :=
  List.Forall₂ (fun h x => Flock.f128OfHex h = .ok x) hs.toList xs.toList

theorem forall₂_flip {α β γ : Type} {R : α → β → Prop} {S : α → γ → Prop} {T : β → γ → Prop}
    (H : ∀ a b c, R a b → S a c → T b c) :
    ∀ {l₁ : List α} {l₂ : List β} {l₃ : List γ}, List.Forall₂ R l₁ l₂ → List.Forall₂ S l₁ l₃ → List.Forall₂ T l₂ l₃
  | _, _, _, .nil, .nil => .nil
  | _, _, _, .cons h₁ t₁, .cons h₂ t₂ => .cons (H _ _ _ h₁ h₂) (forall₂_flip H t₁ t₂)

theorem forall₂_append {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ l₂ : List α} {m₁ m₂ : List β}, List.Forall₂ R l₁ m₁ → List.Forall₂ R l₂ m₂ →
      List.Forall₂ R (l₁ ++ l₂) (m₁ ++ m₂)
  | _, _, _, _, .nil, h => h
  | _, _, _, _, .cons h t, h₂ => .cons h (forall₂_append t h₂)

theorem forall₂_length {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → l₁.length = l₂.length
  | _, _, .nil => rfl
  | _, _, .cons _ t => congrArg (· + 1) (forall₂_length t)

/-- Hex strings `hexes` reads from a list that `Flock.Record.f128s` decodes are the hex of its F128s. -/
theorem hexes_f128s {j : Json} {w : String} {cs : Array String} {a : Array Json} {ys : Array Flock.F128}
    (h : hexes j w = .ok cs) (ha : j = .arr a) (hf : Flock.Record.f128s a = .ok ys) : Decodes cs ys := by
  obtain ⟨xs, hx, hxs⟩ := hexes_ok h
  rw [hx] at ha
  cases ha
  unfold Flock.Record.f128s at hf
  refine forall₂_flip (fun x c y hxc hy => ?_) hxs (Claims.Bridge.Custody.mapM_forall₂ hf)
  subst hxc
  exact hy

/-- The uses of a stream's rounds the record decodes to, from the stream's chain `a`: each round's coins released
after the SHA-512 of its request at the chain's next value and its F128 count. -/
def roundUses (slot : Nat) (name : String) : ByteArray → List Flock.Round → List (Challenger.Use × Array Flock.F128)
  | _, [] => []
  | a, r :: rs =>
    (⟨Hello.slotStream slot name, Flock.Sha512.hash (roundRequest name (Flock.Sha512.hash (a ++ r.sha)) r.coins.size),
      (r.coins.size + 3) / 4⟩, r.coins) :: roundUses slot name (Flock.Sha512.hash (a ++ r.sha)) rs

/-- A session's uses of the challenger's coins as the record decodes: its link points, released after `after`, then
each stream's rounds in the record's order. -/
def decodedUses (slot : Nat) (after : ByteArray) (sess : Flock.Session) : List (Challenger.Use × Array Flock.F128) :=
  (⟨Hello.slotStream slot POINTS, after, (sess.points.flatten.size + 3) / 4⟩, sess.points.flatten) ::
    sess.streams.toList.flatMap fun s => roundUses slot s.name .empty s.rounds.toList

/-- A use and the record's hex of its coins, against a use and the F128s they decode to. -/
def Corr (x : Challenger.Use × Array String) (y : Challenger.Use × Array Flock.F128) : Prop :=
  x.1 = y.1 ∧ Decodes x.2 y.2

theorem dk_sha512 {spec : Flock.SessionSpec} (hs : spec.sha512Digests = true) :
    s!"msg_{spec.dk}" = "msg_sha512" := by
  rw [show spec.dk = "sha512" by simp [Flock.SessionSpec.dk, hs]]
  rfl

/-- A stream's rounds as `usesOf` reads them are the uses of its decoded rounds (`roundUses`), from the same chain. -/
theorem rounds_corr {spec : Flock.SessionSpec} (hs : spec.sha512Digests = true) (slot : Nat) (name : String) :
    ∀ {rl : List Json} {rds : List Flock.Round} {out : List (Challenger.Use × Array String)} {a : ByteArray}
      {st : List (Challenger.Use × Array String) × ByteArray} {D : List (Challenger.Use × Array Flock.F128)},
      Steps (RoundStep slot name) rl (out, a) st → List.Forall₂ (RoundDec spec) rl rds → List.Forall₂ Corr out D →
      List.Forall₂ Corr st.1 (D ++ roundUses slot name a rds)
  | [], [], _, _, _, _, .nil _, .nil, hc => by simpa [roundUses] using hc
  | r :: rl, rd :: rds, out, a, st, D, .cons hstep hrest, .cons hrd hds, hc => by
    obtain ⟨sha, d, cs, w, hsha, hd, hcs, rfl⟩ := hstep
    obtain ⟨h', hstr, hun, cs', harr, hf⟩ := hrd
    rw [dk_sha512 hs] at hstr
    rw [getD_of_str hstr] at hsha
    cases hsha
    rw [hd] at hun
    cases hun
    have hdec := hexes_f128s hcs (getD_of_arr harr) hf
    have hsz : cs.size = rd.coins.size := by
      simpa using forall₂_length hdec
    have := rounds_corr hs slot name hrest hds (D := D ++ [(⟨Hello.slotStream slot name,
      Flock.Sha512.hash (roundRequest name (Flock.Sha512.hash (a ++ rd.sha)) rd.coins.size), (rd.coins.size + 3) / 4⟩,
      rd.coins)]) (forall₂_append hc (.cons ⟨by simp only [hsz], hdec⟩ .nil))
    simpa [roundUses, hsz, List.append_assoc] using this

/-- The streams' uses as `usesOf` reads them are their decoded rounds' uses, stream by stream. -/
theorem streams_corr {spec : Flock.SessionSpec} (hs : spec.sha512Digests = true) (slot : Nat) :
    ∀ {sl : List Json} {srs : List Flock.StreamRec} {out us : List (Challenger.Use × Array String)}
      {D : List (Challenger.Use × Array Flock.F128)},
      Steps (StreamStep slot) sl out us → List.Forall₂ (StreamDec spec) sl srs → List.Forall₂ Corr out D →
      List.Forall₂ Corr us (D ++ srs.flatMap fun s => roundUses slot s.name .empty s.rounds.toList)
  | [], [], _, _, _, .nil _, .nil, hc => by simpa using hc
  | s :: sl, sr :: srs, out, us, D, .cons hstep hrest, .cons hsd hds, hc => by
    obtain ⟨name, rs, st, hname, hrs, hsteps, rfl⟩ := hstep
    obtain ⟨hstr, rs', harr, hrd⟩ := hsd
    rw [getD_of_str hstr] at hname
    cases hname
    rw [getD_of_arr harr] at hrs
    cases hrs
    have := streams_corr hs slot hrest hds (rounds_corr hs slot sr.name hsteps hrd hc)
    simpa [List.append_assoc] using this

/-- The link points' hex `usesOf` gathers decodes to the decoded point lists, concatenated. -/
theorem points_corr : ∀ {pl : List Json} {pts : List (Array Flock.F128)} {acc out : Array String}
    {A : List Flock.F128},
    pl.foldlM (m := V) (fun acc p => return acc ++ (← hexes p "a link point")) acc = .ok out →
    List.Forall₂ (fun p pt => ∃ a, p.getArr? = .ok a ∧ Flock.Record.f128s a = .ok pt) pl pts →
    List.Forall₂ (fun h x => Flock.f128OfHex h = .ok x) acc.toList A →
    List.Forall₂ (fun h x => Flock.f128OfHex h = .ok x) out.toList (A ++ (pts.map (·.toList)).flatten)
  | [], [], acc, out, A, h, .nil, hA => by
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simpa using hA
  | p :: pl, pt :: pts, acc, out, A, h, .cons ⟨a, hg, hf⟩ hrest, hA => by
    rw [List.foldlM_cons] at h
    obtain ⟨acc', h1, h⟩ := FlockSoundness.Refine.ebind_ok h
    obtain ⟨v, hv, h1⟩ := FlockSoundness.Refine.ebind_ok h1
    simp only [pure, Except.pure, Except.ok.injEq] at h1
    subst h1
    have hp : p = .arr a := by
      cases p <;> first | (cases hg; done) | (simp only [Json.getArr?] at hg; cases hg; rfl)
    have hdec := hexes_f128s hv hp hf
    have := points_corr h hrest (by simpa using forall₂_append hA hdec)
    simpa [List.append_assoc] using this

/-- **A session's uses are its decoded record's**: where `usesOf` reads the uses `us` from a record that
`Flock.Record.decode` decodes at a statement of SHA-512 digests, they are its link points, released after what
`pointsAfter` reads, and then each stream's rounds in the record's order, each round released after the SHA-512 of its
request at its stream's chain over the decoded digests and its decoded coins' count; each use's hex reads as the
decoded F128s, one for one. -/
theorem usesOf_decoded {slot : Nat} {record : String} {us : List (Challenger.Use × Array String)}
    (h : usesOf slot record = .ok us) {spec : Flock.SessionSpec} {sess : Flock.Session}
    (hd : Flock.Record.decode spec record = .ok sess) (hs : spec.sha512Digests = true) :
    ∃ j a, Flock.JsonParse.parse record = .ok j ∧ pointsAfter (getD j "link") = .ok a ∧
      List.Forall₂ Corr us (decodedUses slot a sess) := by
  obtain ⟨j, ps, pts, a, ss, hj, hps, hpts, ha, hss, hsteps⟩ := usesOf_steps h
  obtain ⟨j', l, hj', hl, ⟨ps', hps', hpd⟩, ss', hss', hsd⟩ := decode_fields hd
  rw [hj] at hj'
  cases hj'
  refine ⟨j, a, hj, ha, ?_⟩
  rw [getD_of_fld hl, getD_of_arr hps'] at hps
  cases hps
  rw [getD_of_arr hss'] at hss
  cases hss
  rw [← Array.foldlM_toList] at hpts
  have hp := points_corr hpts hpd (A := []) .nil
  have hpe : sess.points.flatten.toList = (sess.points.toList.map (·.toList)).flatten := by
    simp [Array.toList_flatten]
  have hsz : pts.size = sess.points.flatten.size := by
    have := forall₂_length hp
    simp only [Array.length_toList, List.nil_append] at this
    rw [this, ← Array.length_toList, hpe]
  have := streams_corr hs slot hsteps hsd (D := [(⟨Hello.slotStream slot POINTS, a, (sess.points.flatten.size + 3) / 4⟩,
    sess.points.flatten)]) (.cons ⟨by rw [hsz], by unfold Decodes; rw [hpe]; simpa using hp⟩ .nil)
  simpa [decodedUses] using this


theorem forall₂_getElem? {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ {j : Nat} {b : β}, l₂[j]? = some b →
      ∃ a, l₁[j]? = some a ∧ R a b
  | _, _, .nil, _, _, h => by cases h
  | _, _, .cons hab t, 0, _, h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    exact ⟨_, rfl, hab⟩
  | _, _, .cons _ t, j + 1, _, h => by
    simp only [List.getElem?_cons_succ] at h ⊢
    exact forall₂_getElem? t h

theorem mem_zip_of_getElem? {α β : Type} :
    ∀ {l₁ : List α} {l₂ : List β} {i : Nat} {a : α} {b : β}, l₁[i]? = some a → l₂[i]? = some b →
      (a, b) ∈ l₁.zip l₂
  | [], _, _, _, _, h, _ => by cases h
  | _ :: _, [], _, _, _, _, h => by cases h
  | x :: l₁, y :: l₂, 0, a, b, ha, hb => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at ha hb
    subst ha hb
    exact List.mem_cons_self
  | x :: l₁, y :: l₂, i + 1, a, b, ha, hb => by
    simp only [List.getElem?_cons_succ] at ha hb
    exact List.mem_cons_of_mem _ (mem_zip_of_getElem? ha hb)

theorem roundUses_coins (slot : Nat) (name : String) : ∀ (a : ByteArray) (rs : List Flock.Round),
    ∀ d ∈ roundUses slot name a rs, d.1.coins = (d.2.size + 3) / 4
  | _, [], d, hd => by cases hd
  | a, r :: rs, d, hd => by
    rcases List.mem_cons.1 hd with rfl | hd
    · rfl
    · exact roundUses_coins slot name _ rs d hd

/-- Each use the record decodes to reads `⌈n/4⌉` coins of 64 bytes for its `n` F128s. -/
theorem decodedUses_coins {slot : Nat} {a : ByteArray} {sess : Flock.Session} :
    ∀ d ∈ decodedUses slot a sess, d.1.coins = (d.2.size + 3) / 4 := by
  intro d hd
  rcases List.mem_cons.1 hd with rfl | hd
  · rfl
  · obtain ⟨s, -, hs⟩ := List.mem_flatMap.1 hd
    exact roundUses_coins slot s.name _ _ d hs

/-- `usesOf_decoded` with the link points' release named: where the session's first use is released after `P`. -/
theorem usesOf_decoded_at {slot : Nat} {record : String} {us : List (Challenger.Use × Array String)}
    (h : usesOf slot record = .ok us) {spec : Flock.SessionSpec} {sess : Flock.Session}
    (hd : Flock.Record.decode spec record = .ok sess) (hs : spec.sha512Digests = true) {P : ByteArray}
    (hP : ∃ pts, us.head? = some (⟨Hello.slotStream slot POINTS, P, (pts.size + 3) / 4⟩, pts)) :
    List.Forall₂ Corr us (decodedUses slot P sess) := by
  obtain ⟨j, a, -, -, hc⟩ := usesOf_decoded h hd hs
  obtain ⟨pts, hhd⟩ := hP
  have e : a = P := by
    unfold decodedUses at hc
    cases hc with
    | cons hx _ =>
      simp only [List.head?_cons, Option.some.injEq] at hhd
      subst hhd
      exact (congrArg Challenger.Use.after hx.1).symm
  rw [e] at hc
  exact hc

/-- **A session's uses are its decoded record's coins**: where `sessionUses` reads the uses `us` from a record that
`Flock.Record.decode` decodes at a statement of the tables `TABLES` with SHA-512 digests, they are its link points,
released after `LiveHash.pointsHash` of the decoded roots, then each stream's rounds in the record's order, each released
after the SHA-512 of its request at its stream's chain over the decoded digests and its decoded coin count; each use's
hex reads as the decoded F128s, one for one (`decodedUses`). -/
theorem sessionUses_decoded {slot : Nat} {record : String} {us : List (Challenger.Use × Array String)}
    (h : sessionUses slot record = .ok us) {spec : Flock.SessionSpec} {sess : Flock.Session}
    (hd : Flock.Record.decode spec record = .ok sess) (ht : spec.tables = TABLES) (hs : spec.sha512Digests = true) :
    List.Forall₂ Corr us (decodedUses slot (Flock.Guarantees.LiveHash.pointsHash sess.rootB) sess) :=
  usesOf_decoded_at (usesOf_of_sessionUses h).1 hd hs (sessionUses_pointsHash_of_decode h hd ht hs)

theorem setupOne_tags {tags : Flock.Tags} {cf pf : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Json} {program : Option ByteArray} {st : Flock.Stmt}
    (h : FlockSoundness.Discharge.HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st) :
    st.tags = tags := by
  unfold FlockSoundness.Discharge.HiddenExec.setupOne at h
  split at h
  · exact FlockSoundness.Discharge.HiddenExec.setupHidden_tags h
  · exact FlockSoundness.Discharge.Exec.setupH_tags h

/-- **A one-table `--zk` setup's spec** names `I`'s table, with SHA-512 digests where `I`'s tags commit hm96 rows. -/
theorem setupOf_zk_spec {I : FlockSoundness.Discharge.Exec.Inputs} {record : String} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} (hc : I.count ≤ 1)
    (h : FlockSoundness.Discharge.Exec.setupOf (FlockSoundness.Discharge.ZkExec.zkInputs I) record = .ok (spec, sts)) :
    spec.tables = #[I.tags.table] ∧ spec.sha512Digests = I.tags.hm96Rows := by
  unfold FlockSoundness.Discharge.Exec.setupOf at h
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny h with ⟨-, st, hst, rfl, -⟩ | ⟨hc', -⟩
  · have ht := setupOne_tags hst
    refine ⟨?_, ?_⟩
    · show #[st.tags.table] = _
      rw [ht]
      show #[(FlockSoundness.Discharge.ZkExec.zkTags I).table] = #[I.tags.table]
      rw [FlockSoundness.Discharge.ZkExec.zkTags_eq]
      try rfl
    · show st.tags.hm96Rows = _
      rw [ht]
      exact FlockSoundness.Discharge.ZkExec.zkTags_hm96Rows I
  · exact absurd hc' (by show ¬ 1 < I.count; omega)

/-- **v2's sessions set up at `TABLES` with SHA-512 digests**: at the inputs a session's statement sets up from
(`SessionInputs`), a `--zk` setup's spec names exactly `TABLES`, and its record's digests are SHA-512. -/
theorem sessionI_spec {tags0 : Flock.Tags} {circuitB pubB : ByteArray} {tables : List (String × ByteArray)} {o : Own}
    (ht : Flock.Tags.find? STATEMENT = some tags0) {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (h : FlockSoundness.Discharge.Exec.setupOf (FlockSoundness.Discharge.ZkExec.zkInputs
      (Properties.SampledProofs.OneStageZk.sessionI tags0 circuitB pubB tables o)) record = .ok (spec, sts)) :
    spec.tables = TABLES ∧ spec.sha512Digests = true := by
  have h0 : tags0 = Flock.Tags.circuit := by
    have e : Flock.Tags.find? STATEMENT = some Flock.Tags.circuit := rfl
    rw [e] at ht
    exact (Option.some.inj ht).symm
  subst h0
  obtain ⟨h1, h2⟩ := setupOf_zk_spec (Nat.le_refl 1) h
  have hw : (Properties.SampledProofs.OneStageZk.sessionI Flock.Tags.circuit circuitB pubB tables o).tags =
      Flock.Tags.circuit :=
    FlockSoundness.Discharge.ZkExec.withCoins_tree _ _
  rw [hw] at h1 h2
  exact ⟨h1.trans rfl, h2.trans rfl⟩

end SampledProofs.OneStageZk

namespace Verity.Game.LiveDrive

/-! ## The tagged releases as blocks

Where the tagged releases read, by family and root, as the draw's use's run, then the link points', then the rounds'
uses' in turn (`Challenger.unflatten`), they are `Drive.canon`'s blocks: the draw's, the link points' and one round
block a round use (`blocks_of_runs`). -/

section
variable {Msg : Type} (m : Nat) (tagOf : Msg → Option Tag)

/-- Releases of `m` bits each are the releases of a list of `m`-bit releases. -/
theorem rels_of_sizes : ∀ (T : List (Σ m' : Nat, Msg × (Fin m' → Bool))), (∀ z ∈ T, z.1 = m) →
    ∃ T' : List (Msg × (Fin m → Bool)), T = T'.map (rel m)
  | [], _ => ⟨[], rfl⟩
  | ⟨m', b, x⟩ :: T, h => by
    obtain ⟨T', hT'⟩ := rels_of_sizes T fun z hz => h z (List.mem_cons_of_mem _ hz)
    have hm : m' = m := h _ List.mem_cons_self
    subst hm
    exact ⟨(b, x) :: T', by rw [hT']; rfl⟩

/-- **A list of releases is a block**: one release per chunk of `sp`, each tagged `tg`. -/
theorem block_of_list {Yd : Type} (sp : Split Yd m) (tg : Tag) (l : List (Msg × (Fin m → Bool)))
    (hl : l.length = sp.rest.length + 1) (htg : ∀ z ∈ l, tagOf z.1 = some tg) :
    ∃ k : Block Msg m, k.rels = l ∧ k.Fits m tagOf tg sp := by
  match l, hl with
  | (b, x) :: rest, hl => exact ⟨⟨b, x, rest⟩, rfl, htg, by simpa using hl⟩

theorem flatMap_of_map {α β δ : Type} {f : α → List δ} {g : β → List δ} :
    ∀ (xs : List α) (ys : List β), xs.map f = ys.map g → xs.flatMap f = ys.flatMap g
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | x :: xs, y :: ys, h => by
    simp only [List.map_cons, List.cons.injEq] at h
    rw [List.flatMap_cons, List.flatMap_cons, h.1, flatMap_of_map xs ys h.2]

theorem rounds_of_lists {γ : Type} (RF : RoundBlock Msg m → Prop) (P : γ → List (Msg × (Fin m → Bool)) → Prop)
    (hr : ∀ u l, P u l → ∃ r : RoundBlock Msg m, r.k.rels = l ∧ RF r) :
    ∀ bs : List (γ × List (Msg × (Fin m → Bool))), (∀ b ∈ bs, P b.1 b.2) →
      ∃ rds : List (RoundBlock Msg m), rds.map (fun r => r.k.rels) = bs.map Prod.snd ∧ ∀ r ∈ rds, RF r
  | [], _ => ⟨[], rfl, by simp⟩
  | (u, l) :: bs, h => by
    obtain ⟨r, hr1, hr2⟩ := hr u l (h _ List.mem_cons_self)
    obtain ⟨rds, h1, h2⟩ := rounds_of_lists RF P hr bs fun b hb => h b (List.mem_cons_of_mem _ hb)
    refine ⟨r :: rds, by simp [hr1, h1], fun r' hr' => ?_⟩
    rcases List.mem_cons.1 hr' with rfl | hr'
    · exact hr2
    · exact h2 r' hr'

/-- **The tagged releases are the draw's block, the link points' and the rounds'**: where the releases read, by their
family and root (`key`), as the draw's use's run, the link points', then the rounds' uses' in turn, each release
tagged by its family and root (`tk`), the draw's and the points' runs one release per chunk of their coins, and each
round's run a round block (`hr`), the releases are those blocks. -/
theorem blocks_of_runs {γ Yd Yp : Type} (spΩ : Split Yd m) (spP : Split Yp m) (key : Msg → String × ByteArray)
    (run : γ → String × ByteArray) (cnt : γ → Nat) (tk : String × ByteArray → Tag) (T : List (Msg × (Fin m → Bool)))
    (dU pU : γ) (rUs : List γ)
    (hT : T.map (key ∘ Prod.fst) = (dU :: pU :: rUs).flatMap fun u => List.replicate (cnt u) (run u))
    (htag : ∀ z ∈ T, tagOf z.1 = some (tk (key z.1)))
    (hd : tk (run dU) = .draw) (hdn : cnt dU = spΩ.rest.length + 1)
    (hp : tk (run pU) = .points) (hpn : cnt pU = spP.rest.length + 1)
    (RF : RoundBlock Msg m → Prop)
    (hr : ∀ u ∈ rUs, ∀ l : List (Msg × (Fin m → Bool)), l.length = cnt u → (∀ z ∈ l, tagOf z.1 = some (tk (run u))) →
      ∃ r : RoundBlock Msg m, r.k.rels = l ∧ RF r) :
    ∃ (kd kp : Block Msg m) (rds : List (RoundBlock Msg m)),
      T = kd.rels ++ kp.rels ++ rds.flatMap (fun r => r.k.rels) ∧ kd.Fits m tagOf .draw spΩ ∧
        kp.Fits m tagOf .points spP ∧ rds.length = rUs.length ∧ ∀ r ∈ rds, RF r := by
  obtain ⟨bs, hfst, hflat, hbs⟩ := _root_.SampledProofs.Challenger.unflatten (key ∘ Prod.fst) cnt run _ T hT
  have tags : ∀ b ∈ bs, ∀ z ∈ b.2, tagOf z.1 = some (tk (run b.1)) := fun b hb z hz => by
    rw [htag z (by rw [hflat]; exact List.mem_flatMap.2 ⟨b, hb, hz⟩)]
    exact congrArg (fun k => some (tk k)) ((hbs b hb).2 z hz)
  rcases bs with _ | ⟨⟨d, l0⟩, _ | ⟨⟨p, l1⟩, bs'⟩⟩
  · cases hfst
  · cases hfst
  simp only [List.map_cons, List.cons.injEq] at hfst
  obtain ⟨rfl, rfl, hrest⟩ := hfst
  obtain ⟨kd, hkd, hkdF⟩ := block_of_list m tagOf spΩ .draw l0 (by rw [(hbs _ List.mem_cons_self).1, hdn])
    (fun z hz => by rw [tags _ List.mem_cons_self z hz, hd])
  obtain ⟨kp, hkp, hkpF⟩ := block_of_list m tagOf spP .points l1
    (by rw [(hbs _ (List.mem_cons_of_mem _ List.mem_cons_self)).1, hpn])
    (fun z hz => by rw [tags _ (List.mem_cons_of_mem _ List.mem_cons_self) z hz, hp])
  obtain ⟨rds, hrds, hRF⟩ := rounds_of_lists m RF
    (fun u l => u ∈ rUs ∧ l.length = cnt u ∧ ∀ z ∈ l, tagOf z.1 = some (tk (run u)))
    (fun u l h => hr u h.1 l h.2.1 h.2.2) bs' (fun b hb => by
      have hb' : b ∈ (d, l0) :: (p, l1) :: bs' := List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hb)
      exact ⟨hrest ▸ List.mem_map_of_mem hb, (hbs b hb').1, tags b hb'⟩)
  refine ⟨kd, kp, rds, ?_, hkdF, hkpF, ?_, hRF⟩
  · rw [hflat, hkd, hkp, List.flatMap_cons, List.flatMap_cons, List.append_assoc, flatMap_of_map rds bs' hrds]
  · rw [← hrest, List.length_map, ← List.length_map (f := fun r => r.k.rels), hrds, List.length_map]

end

end Verity.Game.LiveDrive

namespace Verity.Served

/-! ## `served`, as a fold of one step -/

variable {R A Rt M : Type} (ans : R → A) (req : A → Option (String × Rt)) (cond : String → ℕ → ℕ → Rt → Bool)

/-- One release's step: where its request names a family `l` and a root, the entry `(l, i, root, r)` is served when
`cond` holds at `l`, the family's count `i` so far, the place `k` and the root. -/
def step (acc : List (String × ℕ × Rt × R)) (r : R) : List (String × ℕ × Rt × R) :=
  match req (ans r) with
  | none => acc
  | some (l, root) =>
    let i := (acc.filter (·.1 == l)).length
    let k := acc.length
    if cond l i k root then acc ++ [(l, i, root, r)] else acc

theorem step_cases (acc : List (String × ℕ × Rt × R)) (r : R) :
    step ans req cond acc r = acc ∨ ∃ l i root, step ans req cond acc r = acc ++ [(l, i, root, r)] := by
  unfold step
  split
  · exact .inl rfl
  · dsimp only
    split
    · exact .inr ⟨_, _, _, rfl⟩
    · exact .inl rfl

/-- An entry without its release's coins. -/
def key (e : String × ℕ × Rt × R) : String × ℕ × Rt × A := (e.1, e.2.1, e.2.2.1, ans e.2.2.2)

theorem step_key (acc acc' : List (String × ℕ × Rt × R)) (h : acc.map (key ans) = acc'.map (key ans)) (r r' : R)
    (hr : ans r = ans r') : (step ans req cond acc r).map (key ans) = (step ans req cond acc' r').map (key ans) := by
  have hl : acc.length = acc'.length := by simpa using congrArg List.length h
  have hf : ∀ l : String, (acc.filter (·.1 == l)).length = (acc'.filter (·.1 == l)).length := by
    intro l
    have h1 : (acc.map (key ans)).filter (·.1 == l) = (acc'.map (key ans)).filter (·.1 == l) := by rw [h]
    rw [List.filter_map, List.filter_map] at h1
    simpa [Function.comp_def, key] using congrArg List.length h1
  unfold step
  rw [hr]
  split
  · exact h
  · dsimp only
    rw [hf, hl]
    split
    · simp [h, key, hr]
    · exact h

variable (mk : M → List Bool → R)

/-- The step serves the request `b` after the requests `h`: what it serves grows by one. -/
def sn (h : List (ℕ × M)) (b : M) : Bool :=
  ((((h.map fun z => mk z.2 []) ++ [mk b []]).foldl (step ans req cond) []).length ==
    ((h.map fun z => mk z.2 []).foldl (step ans req cond) []).length + 1)

theorem foldl_annot (hmk : ∀ b l, ans (mk b l) = ans (mk b [])) :
    ∀ (hs : List (Σ m : ℕ, M × (Fin m → Bool))) (h : List (ℕ × M)) (acc : List (String × ℕ × Rt × R)),
      acc.map (key ans) = ((h.map fun z => mk z.2 []).foldl (step ans req cond) []).map (key ans) →
      ((hs.map (Game.recOf mk)).foldl (step ans req cond) acc).map (·.2.2.2) =
        acc.map (·.2.2.2) ++ ((Game.LiveDrive.annot h hs).filter fun z => sn ans req cond mk z.2.1.1 z.2.1.2).map
          fun z => mk z.2.1.2 (List.ofFn z.2.2)
  | [], _, _, _ => by simp [Game.LiveDrive.annot]
  | ⟨m, b, x⟩ :: hs, h, acc, hacc => by
    have hk := step_key ans req cond acc _ hacc (mk b (List.ofFn x)) (mk b []) (hmk b _)
    have hk' : (step ans req cond acc (mk b (List.ofFn x))).map (key ans) =
        (((h ++ [(m, b)]).map fun z => mk z.2 []).foldl (step ans req cond) []).map (key ans) := by
      rw [hk, List.map_append, List.foldl_append]
      rfl
    have ih := foldl_annot hmk hs (h ++ [(m, b)]) _ hk'
    have hl : (step ans req cond acc (mk b (List.ofFn x))).length =
        (((h.map fun z => mk z.2 []) ++ [mk b []]).foldl (step ans req cond) []).length := by
      have := congrArg List.length hk
      simp only [List.length_map] at this
      rw [this, List.foldl_append]
      rfl
    have hl0 : acc.length = ((h.map fun z => mk z.2 []).foldl (step ans req cond) []).length := by
      simpa using congrArg List.length hacc
    show (((hs.map (Game.recOf mk)).foldl (step ans req cond) (step ans req cond acc (mk b (List.ofFn x))))).map
      (·.2.2.2) = _
    rw [ih]
    simp only [Game.LiveDrive.annot]
    by_cases hs' : sn ans req cond mk h b = true
    · simp only [List.filter_cons, hs', ↓reduceIte]
      have hsz : (step ans req cond acc (mk b (List.ofFn x))).length = acc.length + 1 := by
        rw [hl, hl0]
        exact beq_iff_eq.1 hs'
      rcases step_cases ans req cond acc (mk b (List.ofFn x)) with he | ⟨l, i, root, he⟩
      · rw [he] at hsz
        omega
      · rw [he]
        simp
    · simp only [List.filter_cons, hs', Bool.false_eq_true, ↓reduceIte]
      rcases step_cases ans req cond acc (mk b (List.ofFn x)) with he | ⟨l, i, root, he⟩
      · rw [he]
      · exfalso
        apply hs'
        have hsz := congrArg List.length he
        simp only [List.length_append, List.length_cons, List.length_nil] at hsz
        unfold sn
        rw [← hl, ← hl0, hsz]
        simp

/-- **What the step serves is read off the requests**: the releases served, in order, are those the step serves after
the requests before each. -/
theorem served_annot (hmk : ∀ b l, ans (mk b l) = ans (mk b [])) (hs : List (Σ m : ℕ, M × (Fin m → Bool))) :
    ((hs.map (Game.recOf mk)).foldl (step ans req cond) []).map (·.2.2.2) =
      ((Game.LiveDrive.annot [] hs).filter fun z => sn ans req cond mk z.2.1.1 z.2.1.2).map
        fun z => mk z.2.1.2 (List.ofFn z.2.2) := by
  have := foldl_annot ans req cond mk hmk hs [] [] rfl
  rw [List.map_nil, List.nil_append] at this
  exact this

theorem step_req (acc : List (String × ℕ × Rt × R)) (r : R)
    (hacc : ∀ e ∈ acc, req (ans e.2.2.2) = some (e.1, e.2.2.1)) :
    ∀ e ∈ step ans req cond acc r, req (ans e.2.2.2) = some (e.1, e.2.2.1) := by
  unfold step
  split
  · exact hacc
  · rename_i l root hq
    dsimp only
    split
    · intro e he
      rcases List.mem_append.1 he with he | he
      · exact hacc e he
      · rw [List.mem_singleton.1 he]
        exact hq
    · exact hacc

theorem foldl_req : ∀ (rs : List R) (acc : List (String × ℕ × Rt × R)),
    (∀ e ∈ acc, req (ans e.2.2.2) = some (e.1, e.2.2.1)) →
    ∀ e ∈ rs.foldl (step ans req cond) acc, req (ans e.2.2.2) = some (e.1, e.2.2.1)
  | [], _, h => h
  | r :: rs, acc, h => foldl_req rs _ (step_req ans req cond acc r h)

end Verity.Served

namespace Verity

/-- What the play's device checks before it serves a request of family `l`, its `i`th, at place `k`, after `root`
(`PlayDevice.served`): a coin left, and both lines within the reader's cap. -/
def PlayDevice.serveCond (D : PlayDevice) (l : String) (i k : ℕ) (root : ByteArray) : Bool :=
  (D.families.find? (·.1 == l)).any (fun f => decide (i < f.2)) &&
    decide ((PlayDevice.line (PlayDevice.rootJ l i (D.time (2 * k + 1)) root)).length ≤
      SampledProofs.Challenger.MAX_LOG_LINE) &&
    decide ((PlayDevice.line (PlayDevice.releaseJ l i (D.time (2 * k + 2)) root)).length ≤
      SampledProofs.Challenger.MAX_LOG_LINE)

theorem PlayDevice.served_eq (D : PlayDevice) (rs : List Release) :
    D.served rs = rs.foldl (Served.step Release.answers PlayDevice.requestOf D.serveCond) [] := rfl

theorem servesNext_eq (D : PlayDevice) :
    servesNext D = Served.sn Release.answers PlayDevice.requestOf D.serveCond releaseOf := rfl

/-- **The releases the play's device serves are the ones the reduction reads as served** (`servesNext`, each with the
requests before it), in order. -/
theorem served_play (D : PlayDevice) (hs : List (Σ m : ℕ, ByteArray × (Fin m → Bool))) :
    (D.served (hs.map (Game.recOf releaseOf))).map (·.2.2.2) =
      ((Game.LiveDrive.annot [] hs).filter fun z => servesNext D z.2.1.1 z.2.1.2).map
        fun z => releaseOf z.2.1.2 (List.ofFn z.2.2) := by
  rw [PlayDevice.served_eq, servesNext_eq]
  exact Served.served_annot Release.answers PlayDevice.requestOf D.serveCond releaseOf (fun _ _ => rfl) hs

/-- **Each entry the play's device serves names its request's family and root.** -/
theorem served_req (D : PlayDevice) (rs : List Release) :
    ∀ e ∈ D.served rs, PlayDevice.requestOf e.2.2.2.answers = some (e.1, e.2.2.1) := by
  rw [PlayDevice.served_eq]
  exact Served.foldl_req Release.answers PlayDevice.requestOf D.serveCond rs [] (by simp)

/-- **The releases `tagOfServed` tags are the served ones `tagOfPlay` tags**, of `m` bits, in order. -/
theorem tagged_served (D : PlayDevice) (draw : String) (streamOf : String → Option (ℕ × Fin 2)) (m : ℕ)
    (hs : List (Σ m' : ℕ, ByteArray × (Fin m' → Bool))) :
    (Game.LiveDrive.annot [] hs).filter (Game.LiveDrive.Tagged m (tagOfServed D draw streamOf)) =
      ((Game.LiveDrive.annot [] hs).filter fun z => servesNext D z.2.1.1 z.2.1.2).filter
        fun z => decide (z.1 = m) && (tagOfPlay draw streamOf z.2.1.2).isSome := by
  rw [List.filter_filter]
  refine List.filter_congr fun z _ => ?_
  simp only [Game.LiveDrive.Tagged, tagOfServed]
  cases servesNext D z.2.1.1 z.2.1.2 <;> simp

/-- **The served releases' requests are the device's entries' families and roots**, in order. -/
theorem served_requests (D : PlayDevice) (hs : List (Σ m : ℕ, ByteArray × (Fin m → Bool))) :
    ((Game.LiveDrive.annot [] hs).filter fun z => servesNext D z.2.1.1 z.2.1.2).map
        (fun z => PlayDevice.requestOf z.2.1.2) =
      (D.served (hs.map (Game.recOf releaseOf))).map fun e => some (e.1, e.2.2.1) := by
  have h := congrArg (List.map fun r : Release => PlayDevice.requestOf r.answers) (served_play D hs)
  simp only [List.map_map, Function.comp_def, releaseOf] at h
  rw [← h]
  exact List.map_congr_left fun e he => served_req D _ e he


/-! ## The record's coins as the challenger's

`coinF128s` reads a coin's bits as F128s and the record reads each F128 from the hex of 16 coin bytes
(`OneStageZk.f128_of_record`). They agree where each coin byte holds its eight bits, bit `i` of weight `2 ^ i`, as the
device's `PlayDevice.bytesOfBits` writes them (`bytesOfBits_get`): `f128OfHex_coin`. -/

open SampledProofs.OneStageZk (beWord beNat beWord_toNat)

theorem finTwoEquiv_symm_val (b : Bool) : (finTwoEquiv.symm b).val = if b then 1 else 0 := by
  cases b <;> rfl

theorem coinByte_val (b : Fin 8 → Bool) : (coinByte b).val = ∑ i : Fin 8, if b i then 2 ^ i.val else 0 := by
  simp only [coinByte, Equiv.trans_apply, Equiv.arrowCongr_apply, Equiv.coe_refl, Equiv.refl_symm,
    Function.comp_apply, id, finCongr_apply, Fin.val_cast, finFunctionFinEquiv_apply]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [finTwoEquiv_symm_val]; split <;> simp

theorem coinWord_val (w : Fin 8 → Fin 256) : (coinWord w).val = ∑ i : Fin 8, (w (Fin.rev i)).val * 256 ^ i.val := by
  simp only [coinWord, Equiv.trans_apply, Equiv.arrowCongr_apply, Equiv.coe_refl, Fin.revPerm_symm,
    Function.comp_apply, id, finCongr_apply, Fin.val_cast, finFunctionFinEquiv_apply, Fin.revPerm_apply]

theorem beNat_eight (b : ByteArray) (o : ℕ) :
    beNat b o (List.range' 0 8) 0 = ∑ i : Fin 8, (b.get! (o + (Fin.rev i).val)).toNat * 256 ^ i.val := by
  simp only [beNat, show List.range' 0 8 = [0, 1, 2, 3, 4, 5, 6, 7] from rfl, List.foldl_cons, List.foldl_nil,
    Fin.sum_univ_eight, Fin.val_rev]
  norm_num
  ring

/-- **An F128 coin is its bytes' two big-endian words**: where byte `j` of `B` holds bits `8 j` to `8 j + 7` of the
coin, bit `i` of weight `2 ^ i`, the coin reads as `B`'s F128 (`hi` its first eight bytes, `lo` the next eight), as
`Flock.f128OfHex` reads the record's hex of `B`. -/
theorem coinF128_bytes (f : ℕ → Bool) (B : ByteArray)
    (hB : ∀ j < 16, (B.get! j).toNat = ∑ i : Fin 8, if f (8 * j + i.val) then 2 ^ i.val else 0) :
    coinF128 (fun t => f t.val) = FlockLevel3.GF128.of ⟨beWord B 8, beWord B 0⟩ := by
  show _ = FlockLevel3.f128Equiv.symm ((beWord B 8).toFin, (beWord B 0).toFin)
  simp only [coinF128, Equiv.trans_apply]
  refine congrArg FlockLevel3.f128Equiv.symm ?_
  simp only [Equiv.arrowCongr_apply, Function.comp_apply, Equiv.curry_apply,
    piFinTwoEquiv_apply, Equiv.prodComm_apply, Prod.swap_prod_mk]
  refine Prod.ext (Fin.ext ?_) (Fin.ext ?_) <;>
    simp only [coinWord_val, show ∀ w : UInt64, (w.toFin : ℕ) = w.toNat from fun _ => rfl, beWord_toNat,
      beNat_eight] <;>
    refine Finset.sum_congr rfl fun i _ => ?_ <;> congr 1
  all_goals
    simp only [Function.curry_apply, coinBytes, Equiv.trans_apply, Equiv.arrowCongr_apply, Equiv.symm_trans_apply,
      Equiv.symm_symm, Equiv.refl_symm, Equiv.coe_refl, Function.comp_apply, id, Equiv.curry_apply, coinByte_val,
      finCongr_symm, finCongr_apply]
    rw [hB _ (by omega)]
    refine Finset.sum_congr rfl fun x _ => ?_
    simp only [Fin.val_cast, finProdFinEquiv_apply_val, Fin.val_one, Fin.val_zero]
    ring_nf

theorem byteArray_get! (c : ByteArray) (n : ℕ) : c.get! n = c.data[n]! := by cases c; rfl

theorem get!_extract (b : ByteArray) {s e j : ℕ} (h : s + j < e) (he : e ≤ b.size) :
    (b.extract s e).get! j = b.get! (s + j) := by
  rw [byteArray_get!, byteArray_get!, ByteArray.data_extract,
    getElem!_pos _ _ (by rw [Array.size_extract]; simp only [ByteArray.size] at he; omega),
    getElem!_pos _ _ (by simp only [ByteArray.size] at he; omega), Array.getElem_extract]

theorem foldl_bits (p : ℕ → Bool) : ∀ (l : List ℕ) (acc : ℕ),
    l.foldl (fun (acc : ℕ) i => if p i then acc + 2 ^ i else acc) acc = acc + (l.map fun i => if p i then 2 ^ i else 0).sum
  | [], acc => by simp
  | i :: l, acc => by
    rw [List.foldl_cons, foldl_bits p l, List.map_cons, List.sum_cons]
    split <;> omega

theorem bitsByte_lt (p : Fin 8 → Bool) : (∑ i : Fin 8, if p i then 2 ^ i.val else 0) < 256 := by
  have h := fun i : Fin 8 => (show (if p i then 2 ^ i.val else 0) ≤ 2 ^ i.val by split <;> simp)
  simp only [Fin.sum_univ_eight]
  have := h 0; have := h 1; have := h 2; have := h 3; have := h 4; have := h 5; have := h 6; have := h 7
  simp only [Fin.isValue, Fin.val_zero, Fin.val_one, Fin.val_two] at *
  norm_num at *
  omega

/-- **The device's coin bytes**: byte `j` of a release's bytes holds its bits `8 j` to `8 j + 7`, bit `i` of weight
`2 ^ i`. -/
theorem bytesOfBits_get (l : List Bool) {j : ℕ} (hj : j < l.length / 8) :
    ((PlayDevice.bytesOfBits l).get! j).toNat = ∑ i : Fin 8, if l.getD (8 * j + i.val) false then 2 ^ i.val else 0 := by
  rw [byteArray_get!]
  show (((List.map _ (List.range _)).toArray : Array UInt8)[j]!).toNat = _
  rw [getElem!_pos _ _ (by simp; omega), List.getElem_toArray, List.getElem_map, List.getElem_range, foldl_bits,
    Nat.zero_add]
  have e : ((List.range 8).map fun i => if l.getD (8 * j + i) false then 2 ^ i else 0).sum =
      ∑ i : Fin 8, if l.getD (8 * j + i.val) false then 2 ^ i.val else 0 := by
    simp only [show List.range 8 = [0, 1, 2, 3, 4, 5, 6, 7] from rfl, List.map_cons, List.map_nil, List.sum_cons,
      List.sum_nil, Fin.sum_univ_eight, Fin.isValue, Fin.coe_ofNat_eq_mod, Nat.reduceMod]
    ring
  rw [e, Nat.toUInt8, UInt8.toNat_ofNat', Nat.mod_eq_of_lt (bitsByte_lt _)]

theorem coinF128s_apply (k : ℕ) (f : ℕ → Bool) (i : Fin k) :
    coinF128s k (fun t => f t.val) i = coinF128 (fun t => f (128 * i.val + t.val)) := by
  simp only [coinF128s, Equiv.trans_apply, Equiv.arrowCongr_apply, Equiv.refl_symm, Equiv.coe_refl,
    Function.comp_apply, id, Equiv.curry_apply]
  refine congrArg (⇑coinF128) (funext fun t => ?_)
  simp only [Function.curry_apply, Equiv.arrowCongr_apply, Equiv.symm_symm, Equiv.coe_refl, Function.comp_apply, id,
    finProdFinEquiv_apply_val]
  ring_nf

/-- **`k` F128 coins are their bytes' words**: where byte `j` of `ξ` holds the coin's bits `8 j` to `8 j + 7`, F128 `i`
reads as bytes `16 i` to `16 i + 15` of `ξ`, as `Flock.f128OfHex` reads their hex. -/
theorem coinF128s_bytes (k : ℕ) (f : ℕ → Bool) (ξ : ByteArray) (hsz : 16 * k ≤ ξ.size)
    (hξ : ∀ j < 16 * k, (ξ.get! j).toNat = ∑ s : Fin 8, if f (8 * j + s.val) then 2 ^ s.val else 0) (i : Fin k) :
    coinF128s k (fun t => f t.val) i = FlockLevel3.GF128.of
      ⟨SampledProofs.OneStageZk.beWord (ξ.extract (16 * i.val) (16 * i.val + 16)) 8,
        SampledProofs.OneStageZk.beWord (ξ.extract (16 * i.val) (16 * i.val + 16)) 0⟩ := by
  rw [coinF128s_apply]
  refine coinF128_bytes (fun t => f (128 * i.val + t)) _ fun j hj => ?_
  have hi := i.isLt
  rw [get!_extract _ (by omega) (by omega), hξ _ (by omega)]
  refine Finset.sum_congr rfl fun s _ => ?_
  ring_nf

/-- **A record's coins are the challenger's**: where `coinsMismatch` finds no reason for the record's F128s `hs` against
the coin bytes `ξ`, whose byte `j` holds the coin's bits `8 j` to `8 j + 7`, the record's `i`th F128 reads as the
coin's `i`th. -/
theorem f128OfHex_coin {hs : Array String} {ξ : ByteArray} (h : SampledProofs.OneStageZk.coinsMismatch hs ξ = none)
    (k : ℕ) (f : ℕ → Bool) (hsz : 16 * k ≤ ξ.size)
    (hξ : ∀ j < 16 * k, (ξ.get! j).toNat = ∑ s : Fin 8, if f (8 * j + s.val) then 2 ^ s.val else 0) (i : Fin k)
    (hi : i.val < hs.size) :
    Flock.f128OfHex hs[i.val] = .ok (FlockLevel3.GF128.val (coinF128s k (fun t => f t.val) i)) := by
  rw [coinF128s_bytes k f ξ hsz hξ i, SampledProofs.OneStageZk.f128_of_record h i hi (by have := i.isLt; omega)]
  rfl

theorem bytesOfBits_size (l : List Bool) : (PlayDevice.bytesOfBits l).size = l.length / 8 := by
  simp [PlayDevice.bytesOfBits, ByteArray.size]

/-- **The decoded F128s are the release's coins**: where the record's hex `hs` decodes to the F128s `xs` and
`coinsMismatch` finds no reason for `hs` against a release's bytes, written from its bits `l`, at least `128` bits an
F128, the `i`th F128 is the `i`th that `coinF128s` reads from `l`. -/
theorem decodes_coins {hs : Array String} {xs : Array Flock.F128} (hd : SampledProofs.OneStageZk.Decodes hs xs)
    {l : List Bool} (h : SampledProofs.OneStageZk.coinsMismatch hs (PlayDevice.bytesOfBits l) = none)
    (hl : 128 * hs.size ≤ l.length) :
    xs = Array.ofFn fun i : Fin xs.size =>
      FlockLevel3.GF128.val (coinF128s xs.size (fun t => l.getD t.val false) i) := by
  unfold SampledProofs.OneStageZk.Decodes at hd
  have hsz : xs.size = hs.size := by
    have := List.Forall₂.length_eq hd
    simpa using this.symm
  refine Array.ext (by simp) fun i hi1 hi2 => ?_
  have hi : i < hs.size := by rw [← hsz]; exact hi1
  have hx : Flock.f128OfHex hs[i] = .ok xs[i] := by
    have := (List.forall₂_iff_get.1 hd).2 i (by simpa using hi) (by simpa using hi1)
    simpa using this
  have hc := f128OfHex_coin h xs.size (fun n => l.getD n false) (by rw [bytesOfBits_size]; omega)
    (fun j hj => bytesOfBits_get l (by omega)) ⟨i, hi1⟩ hi
  rw [Array.getElem_ofFn]
  exact Except.ok.inj (hx.symm.trans hc)

/-- **Session 0's link points and round coins are the challenger's releases, in the record's order**: where v2's
round-coins check names no reason and session 0's record decodes at a statement of the tables `TABLES` with SHA-512
digests, its decoded uses (`decodedUses`: the link points after `LiveHash.pointsHash` of the decoded roots, then each
stream's rounds after the SHA-512 of their requests over the stream's chain) are the uses `roundCoins` reads after the
draw's, in order; and where `roundCoins` returns, for each use, the bytes of a release of at least `512` bits a coin,
each decoded use's F128s are the F128s `coinF128s` reads from its release's bits. -/
theorem session0_coins {p : SampledProofs.OneStageZk.Params} {c : SampledProofs.OneStage.Commitment}
    {e : SampledProofs.OneStageZk.Evidence} {bs : Core.Blobs} (h : SampledProofs.OneStageZk.roundsWhy p c e bs = none)
    {s0 : SampledProofs.OneStageZk.Session} {rest : List SampledProofs.OneStageZk.Session}
    (hs0 : e.sessions.toList = s0 :: rest) :
    ∃ key label w opened uses order groups coins ξs, p.base.challenger = some (key, label) ∧
      p.base.verifier.windowRoot = some w ∧
      SampledProofs.getD (SampledProofs.getD e.base.coins "segment") "openings" = .arr opened ∧
      SampledProofs.Challenger.roundCoins key w uses order groups coins = .ok ξs ∧
      uses[0]? = some ⟨label, c.digest, opened.size⟩ ∧
      ∃ b record, bs.get s0.record s!"session {s0.name}'s record" = .ok b ∧ String.fromUTF8? b = some record ∧
        ∀ (spec : Flock.SessionSpec) (sess : Flock.Session), Flock.Record.decode spec record = .ok sess →
          spec.tables = SampledProofs.OneStageZk.TABLES → spec.sha512Digests = true →
          (∀ (i : Nat) (u : SampledProofs.Challenger.Use), uses[i]? = some u →
            ∃ l : List Bool, ξs[i]? = some (PlayDevice.bytesOfBits l) ∧ 512 * u.coins ≤ l.length) →
          ∀ (j : Nat) (d : SampledProofs.Challenger.Use × Array Flock.F128),
            (SampledProofs.OneStageZk.decodedUses 0 (Flock.Guarantees.LiveHash.pointsHash sess.rootB) sess)[j]? =
              some d →
            ∃ l : List Bool, uses[j + 1]? = some d.1 ∧ ξs[j + 1]? = some (PlayDevice.bytesOfBits l) ∧
              d.2 = Array.ofFn fun i : Fin d.2.size =>
                FlockLevel3.GF128.val (coinF128s d.2.size (fun t => l.getD t.val false) i) := by
  obtain ⟨key, label, w, opened, uses, wants, order, groups, coins, ξs, hc, hw, hop, hrc, -, hmis, hS⟩ :=
    SampledProofs.OneStageZk.roundsWhy_spec h
  have hS0 := hS s0 rest hs0
  unfold SampledProofs.OneStageZk.Session0 at hS0
  obtain ⟨b, record, us0, hb, hrec, hus, -, hu0, -, hj, -⟩ := hS0
  refine ⟨key, label, w, opened, uses, order, groups, coins, ξs, hc, hw, hop, hrc, hu0, b, record, hb, hrec,
    fun spec sess hd ht hsd hξ j d hdj => ?_⟩
  obtain ⟨u, hu, hud⟩ := SampledProofs.OneStageZk.forall₂_getElem?
    (SampledProofs.OneStageZk.sessionUses_decoded hus hd ht hsd) hdj
  obtain ⟨huse, what, hwant⟩ := hj j u hu
  obtain ⟨l, hξj, hl⟩ := hξ (j + 1) u.1 huse
  have hm := hmis _ (SampledProofs.OneStageZk.mem_zip_of_getElem? hξj hwant) what u.2 rfl
  have hcoins := SampledProofs.OneStageZk.decodedUses_coins d (List.mem_of_getElem? hdj)
  have hsz : u.2.size = d.2.size := by
    have := SampledProofs.OneStageZk.forall₂_length hud.2
    simpa using this
  rw [hud.1, hcoins] at hl
  refine ⟨l, by rw [← hud.1]; exact huse, hξj, decodes_coins hud.2 hm (by omega)⟩

end Verity
