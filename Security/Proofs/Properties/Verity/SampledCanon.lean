import Proofs.Properties.Verity.SampledSim
import Std.Data.String.ToNat
import Proofs.Flock.Claims.Complete.Json.Hex

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
(`OneStageZk.sessionUses_points`, `OneStageZk.pointsAfter_ok`).
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

/-- Every use session `slot` makes of the challenger's coins is of one of its own families, `slot/<slot>/…`. -/
theorem sessionUses_label {slot : Nat} {record : String} {us : List (Challenger.Use × Array String)}
    (h : sessionUses slot record = .ok us) : ∀ x ∈ us, ∃ n, x.1.label = Hello.slotStream slot n := by
  let P : List (Challenger.Use × Array String) → Prop := fun l => ∀ x ∈ l, ∃ n, x.1.label = Hello.slotStream slot n
  have hrefl : ∀ a, P a → P a := fun _ h => h
  have htrans : ∀ a b c, (P a → P b) → (P b → P c) → P a → P c := fun _ _ _ h1 h2 h => h2 (h1 h)
  have happ : ∀ a y n, P a → y.1.label = Hello.slotStream slot n → P (a ++ [y]) := by
    intro a y n ha hy x hx
    rcases List.mem_append.1 hx with hx | hx
    · exact ha x hx
    · rw [List.mem_singleton.1 hx]; exact ⟨n, hy⟩
  unfold sessionUses at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  rename_i heq
  cases h
  rw [← Array.forIn_toList] at heq
  refine loop_mono (fun a b => P a → P b) hrefl htrans _ ?_ _ _ _ heq ?_
  · intro x s r hr hs
    repeat' split at hr
    all_goals first | (cases hr; done) | (simp [throw, throwThe, MonadExceptOf.throw] at *; done) | skip
    rename_i heq2
    cases hr
    rw [← Array.forIn_toList] at heq2
    refine loop_mono (fun a b => P a → P b) hrefl htrans _ ?_ _ _ _ heq2 hs
    intro y t q hq ht
    repeat' split at hq
    all_goals first | (cases hq; done) | (simp [throw, throwThe, MonadExceptOf.throw] at *; done) | skip
    all_goals (cases hq; exact happ _ _ _ ht rfl)
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

/-- **What `roundsWhy` runs, where it names no reason**: v2's parameters name a challenger key and label and a window
root, the draw's segment opens `opened`, and `roundCoins` returns `ξs` for its uses, order, groups and coins; every coin
a session's record holds is the challenger's (`coinsMismatch`, use by use); and session 0's uses are read as
`Session0` says. -/
theorem roundsWhy_spec {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    (h : roundsWhy p c e bs = none) :
    ∃ key label w opened uses wants order groups coins ξs, p.base.challenger = some (key, label) ∧
      p.base.verifier.windowRoot = some w ∧ getD (getD e.base.coins "segment") "openings" = .arr opened ∧
      Challenger.roundCoins key w uses order groups coins = .ok ξs ∧
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
              refine ⟨key, label, w, opened, v.1.toList, v.2.1.toList, v.2.2.2.1.toList, v.2.2.2.2.toList, _, ξs, hc, hw, hop, hrc, ?_, ?_⟩
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
  unfold sessionUses at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  rename_i _ j hj _ _ _ _ pts _ _ a ha _ _ _ _ _ heq
  cases h
  rw [← Array.forIn_toList] at heq
  obtain ⟨t, rfl⟩ := loop_mono (fun x y => ∃ t, y = x ++ t) (fun _ => ⟨[], by simp⟩)
    (fun _ _ _ ⟨t1, e1⟩ ⟨t2, e2⟩ => ⟨t1 ++ t2, by rw [e2, e1, List.append_assoc]⟩) _ (fun x s r hr => by
      repeat' split at hr
      all_goals first
        | (cases hr; done)
        | (simp [throw, throwThe, MonadExceptOf.throw] at *; done)
        | skip
      rename_i heq2
      cases hr
      rw [← Array.forIn_toList] at heq2
      exact loop_mono (fun x y => ∃ t, y = x ++ t) (fun _ => ⟨[], by simp⟩)
        (fun _ _ _ ⟨t1, e1⟩ ⟨t2, e2⟩ => ⟨t1 ++ t2, by rw [e2, e1, List.append_assoc]⟩) _ (fun y t q hq => by
          repeat' split at hq
          all_goals first
            | (cases hq; done)
            | (simp [throw, throwThe, MonadExceptOf.throw] at *; done)
            | (cases hq; exact ⟨_, rfl⟩)) _ _ _ heq2) _ _ _ heq
  exact ⟨j, a, pts, mapError_ok hj, ha, rfl⟩

end SampledProofs.OneStageZk

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

end Verity
