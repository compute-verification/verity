import Flock.Firewall
import Flock.Firewall.Contract

/-!
# The firewall's salts reach what it sends only as commit strings

`run_resalt`: on one request stream, the firewall (`Flock.Firewall.run`) at two salt lists of one length sends the same
messages but for their commit strings, finishes at both or at neither, and its commit strings are those of the same 64-byte
row digests under each list's salts, in salt order. `run_of_one` is its use: a session whose messages at one salt list are
a template `σ` filled with the commit strings of digests `D`, finished, sends `σ` filled with `D`'s commit strings under
every salt list of that length.

What a message says but its commit strings is `blank`; its commit strings, in the order the firewall drew their salts,
`holes`; `fillAll` puts commit strings back into blanked messages. The proof follows `run`'s loop, `step`'s cases and the
salting `mapM`s at once at the two salt lists (`Rel`, a relational reading of the firewall's state monad).
-/

namespace FlockSoundness.Discharge.Recursion.FirewallRun

open Flock Flock.Firewall

/-! ## Messages without their commit strings -/

/-- A list of commit strings, each emptied. -/
def blankTop (t : List ByteArray) : List ByteArray := t.map fun _ => ByteArray.empty

/-- A table's top in `Commit`, its commit strings emptied. -/
def blankRoot (e : String × List ByteArray) : String × List ByteArray := (e.1, blankTop e.2)

/-- A message with each commit string emptied: all it says but its commit strings. -/
def blank : Sent → Sent
  | .hello h => .hello h
  | .commit roots => .commit (roots.map blankRoot)
  | .link z => .link z
  | .openStream stream parent pos => .openStream stream parent pos
  | .round stream tops level0 rows n => .round stream (tops.map blankTop) level0 (blankTop rows) n
  | .finish ok => .finish ok

/-- A message's commit strings, in the order the firewall drew their salts. -/
def holes : Sent → List ByteArray
  | .hello _ => []
  | .commit roots => roots.flatMap (·.2)
  | .link _ => []
  | .openStream _ _ _ => []
  | .round _ tops _ rows _ => tops.flatten ++ rows
  | .finish _ => []

/-- The commit strings of what the firewall sent, in salt order. -/
def holesOf (sent : Array Sent) : List ByteArray := sent.toList.flatMap holes

/-- `Commit`'s tops, their commit strings in order from `cs`. -/
def fillRoots : List (String × List ByteArray) → List ByteArray → List (String × List ByteArray)
  | [], _ => []
  | e :: es, cs => (e.1, cs.take e.2.length) :: fillRoots es (cs.drop e.2.length)

/-- A round's tops, their commit strings in order from `cs`, and what is left of `cs`. -/
def fillTops : List (List ByteArray) → List ByteArray → List (List ByteArray) × List ByteArray
  | [], cs => ([], cs)
  | t :: ts, cs => (cs.take t.length :: (fillTops ts (cs.drop t.length)).1, (fillTops ts (cs.drop t.length)).2)

/-- A message, its commit strings in order from `cs`. -/
def fill (cs : List ByteArray) : Sent → Sent
  | .hello h => .hello h
  | .commit roots => .commit (fillRoots roots cs)
  | .link z => .link z
  | .openStream stream parent pos => .openStream stream parent pos
  | .round stream tops level0 rows n =>
    .round stream (fillTops tops cs).1 level0 ((fillTops tops cs).2.take rows.length) n
  | .finish ok => .finish ok

/-- Messages, their commit strings in order from `cs`. -/
def fillAll : List Sent → List ByteArray → List Sent
  | [], _ => []
  | m :: ms, cs => fill cs m :: fillAll ms (cs.drop (holes m).length)

theorem length_blankTop (t : List ByteArray) : (blankTop t).length = t.length := List.length_map _

theorem fillRoots_blank : ∀ (roots : List (String × List ByteArray)) (cs : List ByteArray),
    fillRoots (roots.map blankRoot) cs = fillRoots roots cs
  | [], _ => rfl
  | e :: es, cs => by
    simp only [List.map_cons, fillRoots, blankRoot, length_blankTop, fillRoots_blank es]

theorem fillTops_blank : ∀ (tops : List (List ByteArray)) (cs : List ByteArray),
    fillTops (tops.map blankTop) cs = fillTops tops cs
  | [], _ => rfl
  | t :: ts, cs => by
    simp only [List.map_cons, fillTops, length_blankTop, fillTops_blank ts]

theorem fill_blank (cs : List ByteArray) (m : Sent) : fill cs (blank m) = fill cs m := by
  cases m <;> simp only [blank, fill, fillRoots_blank, fillTops_blank, length_blankTop]

theorem length_flatMap_blankRoot : ∀ roots : List (String × List ByteArray),
    ((roots.map blankRoot).flatMap (·.2)).length = (roots.flatMap (·.2)).length
  | [] => rfl
  | e :: es => by
    simp only [List.map_cons, List.flatMap_cons, List.length_append, blankRoot, length_blankTop,
      length_flatMap_blankRoot es]

theorem length_flatten_blankTop : ∀ tops : List (List ByteArray),
    (tops.map blankTop).flatten.length = tops.flatten.length
  | [] => rfl
  | t :: ts => by
    simp only [List.map_cons, List.flatten_cons, List.length_append, length_blankTop, length_flatten_blankTop ts]

theorem length_holes_blank (m : Sent) : (holes (blank m)).length = (holes m).length := by
  cases m <;> simp only [blank, holes, length_flatMap_blankRoot, List.length_append, length_flatten_blankTop,
    length_blankTop]

theorem fillRoots_holes : ∀ (roots : List (String × List ByteArray)) (rest : List ByteArray),
    fillRoots roots (roots.flatMap (·.2) ++ rest) = roots
  | [], _ => rfl
  | e :: es, rest => by
    simp only [List.flatMap_cons, List.append_assoc, fillRoots, List.take_left, List.drop_left,
      fillRoots_holes es rest, Prod.eta]

theorem fillTops_holes : ∀ (tops : List (List ByteArray)) (rest : List ByteArray),
    fillTops tops (tops.flatten ++ rest) = (tops, rest)
  | [], _ => rfl
  | t :: ts, rest => by
    simp only [List.flatten_cons, List.append_assoc, fillTops, List.take_left, List.drop_left, fillTops_holes ts rest]

theorem fill_holes (m : Sent) (rest : List ByteArray) : fill (holes m ++ rest) m = m := by
  cases m with
  | hello h => rfl
  | commit roots => simp only [holes, fill, fillRoots_holes]
  | link z => rfl
  | openStream stream parent pos => rfl
  | round stream tops level0 rows n =>
    simp only [holes, fill, List.append_assoc, fillTops_holes, List.take_left]
  | finish ok => rfl

/-- **Blanked messages filled with their own commit strings are the messages.** -/
theorem fillAll_blank_holes : ∀ (ms : List Sent) (rest : List ByteArray),
    fillAll (ms.map blank) (ms.flatMap holes ++ rest) = ms
  | [], _ => rfl
  | m :: ms, rest => by
    simp only [List.map_cons, List.flatMap_cons, fillAll, fill_blank, List.append_assoc, fill_holes,
      length_holes_blank, List.drop_left, fillAll_blank_holes ms rest]

/-! ## Commit strings -/

theorem rowDigest_size (row : Array Nat) : (rowDigest row).size = 64 := Firewall.Contract.sha512_size _

/-- **A commit string determines its 64-byte message under one salt.** -/
theorem commitString_inj {x x' y : ByteArray} (hx : x.size = 64) (hx' : x'.size = 64)
    (h : Hm96.Default512.commitString x y = Hm96.Default512.commitString x' y) : x = x' := by
  have h' : Firewall.Contract.xor64 x (Hm96.Default512.mask y) ++ Sha512.hash (Hm96.Default512.saltPrefix ++ y) =
      Firewall.Contract.xor64 x' (Hm96.Default512.mask y) ++ Sha512.hash (Hm96.Default512.saltPrefix ++ y) := h
  exact Firewall.Contract.xor64_cancel hx hx'
    (ByteArray.append_inj_left h' (by rw [Firewall.Contract.xor64_size, Firewall.Contract.xor64_size]))

theorem zipWith_append_left {α β γ : Type} (f : α → β → γ) : ∀ (a b : List α) (l : List β),
    List.zipWith f (a ++ b) l = List.zipWith f a l ++ List.zipWith f b (l.drop a.length)
  | [], b, l => by simp
  | _ :: a, b, [] => by simp
  | x :: a, b, y :: l => by simp [zipWith_append_left f a b l]

theorem drop_succ_of_drop {α : Type} {l rest : List α} {k : Nat} {y : α} (h : l.drop k = y :: rest) :
    l.drop (k + 1) = rest := by
  rw [← List.drop_drop, h] <;> rfl

theorem length_le_of_drop {α : Type} {l rest : List α} {k : Nat} {y : α} (h : l.drop k = y :: rest) :
    k + 1 ≤ l.length := by
  have := congrArg List.length h
  simp only [List.length_drop, List.length_cons] at this
  omega

/-! ## Two runs at once -/

section Rel

variable (ys ys' : List ByteArray)

/-- What two states of one session at the salt lists `ys`, `ys'` drew and sent: `z`, `z'` are their salts left, `x`, `x'`
what they sent, and `p`, `p'` the commit strings drawn and not yet sent. Both drew as many salts, the first ones of their
lists; they sent the same messages but for their commit strings; and their commit strings, the sent ones then the
pending ones, are those of the same 64-byte digests `ds` under each list's salts, in order. -/
def Drawn (p p' : List ByteArray) (z : List ByteArray) (x : Array Sent) (z' : List ByteArray) (x' : Array Sent) :
    Prop :=
  x.toList.map blank = x'.toList.map blank ∧ ys.length = ys'.length ∧
    ∃ ds : List ByteArray, (∀ d ∈ ds, d.size = 64) ∧ ds.length ≤ ys.length ∧
      z = ys.drop ds.length ∧ z' = ys'.drop ds.length ∧
      holesOf x ++ p = List.zipWith Hm96.Default512.commitString ds ys ∧
      holesOf x' ++ p' = List.zipWith Hm96.Default512.commitString ds ys'

/-- Two outcomes of the firewall's state monad, refusals or results with states. -/
def Both {α α' : Type} (P : α → α' → State → State → Prop) : V (α × State) → V (α' × State) → Prop
  | .ok (a, t), .ok (a', t') => P a a' t t'
  | .ok _, .error _ => False
  | .error _, .ok _ => False
  | .error _, .error _ => True

theorem both_ok {α α' : Type} {P : α → α' → State → State → Prop} {a : α} {a' : α'} {t t' : State} :
    Both P (.ok (a, t)) (.ok (a', t')) ↔ P a a' t t' := Iff.rfl

/-- **`m` and `m'` run alike**: from a state and one equal to it but for its salts and what it sent, `Drawn` with `p`, `p'`
pending, both refuse, or both return, with results `Q` relates to the commit strings `q`, `q'` then pending, at states
again equal but for those, and `Drawn` with `q`, `q'` pending. -/
def Rel {α α' : Type} (p p' : List ByteArray) (m : Fw α) (m' : Fw α')
    (Q : α → α' → List ByteArray → List ByteArray → Prop) : Prop :=
  ∀ (st : State) (z' : List ByteArray) (x' : Array Sent), Drawn ys ys' p p' st.salts st.sent z' x' →
    Both (fun a a' t t' => ∃ q q', Q a a' q q' ∧ ∃ z₁ x₁, t' = { t with salts := z₁, sent := x₁ } ∧
        Drawn ys ys' q q' t.salts t.sent z₁ x₁)
      (m.run st) (m'.run { st with salts := z', sent := x' })

variable {ys ys'}

theorem Rel.bind {α α' β β' : Type} {p p' : List ByteArray} {m : Fw α} {m' : Fw α'} {f : α → Fw β}
    {f' : α' → Fw β'} {Q : α → α' → List ByteArray → List ByteArray → Prop}
    {R : β → β' → List ByteArray → List ByteArray → Prop}
    (h : Rel ys ys' p p' m m' Q) (hf : ∀ a a' q q', Q a a' q q' → Rel ys ys' q q' (f a) (f' a') R) :
    Rel ys ys' p p' (m >>= f) (m' >>= f') R := by
  intro st z' x' hD
  have h1 := h st z' x' hD
  rw [StateT.run_bind, StateT.run_bind]
  rcases hm : m.run st with e | ⟨a, t⟩ <;> rcases hm' : m'.run { st with salts := z', sent := x' } with e' | ⟨a', t'⟩ <;>
    rw [hm, hm'] at h1
  · exact True.intro
  · exact False.elim h1
  · exact False.elim h1
  · obtain ⟨q, q', hQ, z₁, x₁, rfl, hD'⟩ := h1
    exact hf a a' q q' hQ t z₁ x₁ hD'

theorem Rel.conseq {α α' : Type} {p p' : List ByteArray} {m : Fw α} {m' : Fw α'}
    {Q Q' : α → α' → List ByteArray → List ByteArray → Prop}
    (h : Rel ys ys' p p' m m' Q) (hQ : ∀ a a' q q', Q a a' q q' → Q' a a' q q') : Rel ys ys' p p' m m' Q' := by
  intro st z' x' hD
  have h1 := h st z' x' hD
  revert h1
  rcases m.run st with e | ⟨a, t⟩ <;> rcases m'.run { st with salts := z', sent := x' } with e' | ⟨a', t'⟩
  · exact fun _ => True.intro
  · exact fun h1 => False.elim h1
  · exact fun h1 => False.elim h1
  · rintro ⟨q, q', hq, rest⟩
    exact ⟨q, q', hQ a a' q q' hq, rest⟩

theorem Rel.pure {α α' : Type} {p p' : List ByteArray} {Q : α → α' → List ByteArray → List ByteArray → Prop}
    {a : α} {a' : α'} (h : Q a a' p p') : Rel ys ys' p p' (Pure.pure a) (Pure.pure a') Q := by
  intro st z' x' hD
  exact both_ok.mpr ⟨p, p', h, z', x', rfl, hD⟩

theorem Rel.throw {α α' : Type} {p p' : List ByteArray} {Q : α → α' → List ByteArray → List ByteArray → Prop}
    {e e' : String} : Rel ys ys' p p' (throw e : Fw α) (throw e' : Fw α') Q :=
  fun _ _ _ _ => True.intro

theorem Rel.ite {α α' : Type} {p p' : List ByteArray} {Q : α → α' → List ByteArray → List ByteArray → Prop}
    {c c' : Prop} [Decidable c] [Decidable c'] {a b : Fw α} {a' b' : Fw α'} (hc : c ↔ c')
    (ha : c → Rel ys ys' p p' a a' Q) (hb : ¬c → Rel ys ys' p p' b b' Q) :
    Rel ys ys' p p' (if c then a else b) (if c' then a' else b') Q := by
  by_cases h : c
  · rw [ite_eq_left h, ite_eq_left (hc.mp h)]; exact ha h
  · rw [ite_eq_right h, ite_eq_right (mt hc.mpr h)]; exact hb h

theorem Rel.get {p p' : List ByteArray} :
    Rel ys ys' p p' (MonadState.get : Fw State) MonadState.get
      (fun a a' q q' => q = p ∧ q' = p' ∧ ∃ z x, a' = { a with salts := z, sent := x } ∧
        Drawn ys ys' p p' a.salts a.sent z x) := by
  intro st z' x' hD
  exact both_ok.mpr ⟨p, p', ⟨rfl, rfl, z', x', rfl, hD⟩, z', x', rfl, hD⟩

theorem Rel.set {p p' : List ByteArray} {s s' : State}
    (h : ∃ z x, s' = { s with salts := z, sent := x } ∧ Drawn ys ys' p p' s.salts s.sent z x) :
    Rel ys ys' p p' (MonadStateOf.set s : Fw PUnit) (MonadStateOf.set s') (fun _ _ q q' => q = p ∧ q' = p') := by
  intro st z' x' _
  obtain ⟨z, x, rfl, hD⟩ := h
  exact both_ok.mpr ⟨p, p', ⟨rfl, rfl⟩, z, x, rfl, hD⟩

theorem Rel.modify {p p' : List ByteArray} {f f' : State → State}
    (h : ∀ (st : State) z x, Drawn ys ys' p p' st.salts st.sent z x →
      ∃ z₁ x₁, f' { st with salts := z, sent := x } = { f st with salts := z₁, sent := x₁ } ∧
        Drawn ys ys' p p' (f st).salts (f st).sent z₁ x₁) :
    Rel ys ys' p p' (modify f : Fw PUnit) (modify f') (fun _ _ q q' => q = p ∧ q' = p') := by
  intro st z' x' hD
  obtain ⟨z₁, x₁, he, hD'⟩ := h st z' x' hD
  exact both_ok.mpr ⟨p, p', ⟨rfl, rfl⟩, z₁, x₁, he, hD'⟩

theorem holesOf_push (a : Array Sent) (m : Sent) : holesOf (a.push m) = holesOf a ++ holes m := by
  simp [holesOf]

/-- Sending messages alike but for their commit strings, the next pending ones. -/
theorem Rel.send {p p' q q' : List ByteArray} {m m' : Sent} (hb : blank m = blank m') (hp : p = holes m ++ q)
    (hp' : p' = holes m' ++ q') :
    Rel ys ys' p p' (send m) (send m') (fun _ _ r r' => r = q ∧ r' = q') := by
  intro st z' x' hD
  obtain ⟨hbl, hlen, ds, hds, hle, hz, hz', hh, hh'⟩ := hD
  refine both_ok.mpr ⟨q, q', ⟨rfl, rfl⟩, z', x'.push m', rfl, ?_⟩
  refine ⟨?_, hlen, ds, hds, hle, hz, hz', ?_, ?_⟩
  · show (st.sent.push m).toList.map blank = (x'.push m').toList.map blank
    simp only [Array.toList_push, List.map_append, List.map_cons, List.map_nil, hbl, hb]
  · show holesOf (st.sent.push m) ++ q = _
    rw [holesOf_push, List.append_assoc, ← hp, hh]
  · show holesOf (x'.push m') ++ q' = _
    rw [holesOf_push, List.append_assoc, ← hp', hh']

/-- Sending one message without commit strings. -/
theorem Rel.send_same {p p' : List ByteArray} {m : Sent} (hm : holes m = []) :
    Rel ys ys' p p' (Firewall.send m) (Firewall.send m) (fun _ _ r r' => r = p ∧ r' = p') :=
  Rel.send rfl (by rw [hm, List.nil_append]) (by rw [hm, List.nil_append])

/-- A computation the firewall's state doesn't enter (`decode`, `split`, `nodes`, `pinned`), lifted. -/
theorem Rel.lift {α : Type} {p p' : List ByteArray} {x : V α} :
    Rel ys ys' p p' (liftM x : Fw α) (liftM x) (fun a a' q q' => a = a' ∧ q = p ∧ q' = p') := by
  intro st z' x' hD
  cases x with
  | error e => exact True.intro
  | ok a => exact both_ok.mpr ⟨p, p', ⟨rfl, rfl, rfl⟩, z', x', rfl, hD⟩

theorem Rel.throw_bind {α α' β β' : Type} {p p' : List ByteArray}
    {Q : β → β' → List ByteArray → List ByteArray → Prop} {e e' : String} {f : α → Fw β} {f' : α' → Fw β'} :
    Rel ys ys' p p' ((MonadExcept.throw e : Fw α) >>= f) ((MonadExcept.throw e' : Fw α') >>= f') Q :=
  fun _ _ _ _ => True.intro

theorem Rel.pure_bind {α α' β β' : Type} {p p' : List ByteArray}
    {R : β → β' → List ByteArray → List ByteArray → Prop} {a : α} {a' : α'} {f : α → Fw β} {f' : α' → Fw β'}
    (h : Rel ys ys' p p' (f a) (f' a') R) : Rel ys ys' p p' (Pure.pure a >>= f) (Pure.pure a' >>= f') R :=
  Rel.bind (Rel.pure (Q := fun b b' q q' => b = a ∧ b' = a' ∧ q = p ∧ q' = p') ⟨rfl, rfl, rfl, rfl⟩)
    fun _ _ _ _ hb => by obtain ⟨rfl, rfl, rfl, rfl⟩ := hb; exact h

/-- A `do` block's `if` with a join point `b` after it: `b` is shown once, and the branch `a` may use it. -/
theorem Rel.ite_jp {α α' : Type} {p p' : List ByteArray} {Q : α → α' → List ByteArray → List ByteArray → Prop}
    {c c' : Prop} [Decidable c] [Decidable c'] {a b : Fw α} {a' b' : Fw α'} (hc : c ↔ c')
    (hb : Rel ys ys' p p' b b' Q) (ha : c → Rel ys ys' p p' b b' Q → Rel ys ys' p p' a a' Q) :
    Rel ys ys' p p' (if c then a else b) (if c' then a' else b') Q :=
  Rel.ite hc (fun h => ha h hb) fun _ => hb

/-- `l.mapM f` at two alike states, where `f` draws no salt. -/
theorem Rel.mapM_same {α β : Type} (f : α → Fw β)
    (hf : ∀ a q q', Rel ys ys' q q' (f a) (f a) (fun b b' r r' => b = b' ∧ r = q ∧ r' = q')) :
    ∀ (l : List α) (p p' : List ByteArray), Rel ys ys' p p' (l.mapM f) (l.mapM f)
      (fun bs bs' r r' => bs = bs' ∧ r = p ∧ r' = p')
  | [], p, p' => by
    rw [List.mapM_nil]
    exact Rel.pure ⟨rfl, rfl, rfl⟩
  | a :: l, p, p' => by
    rw [List.mapM_cons]
    refine Rel.bind (hf a p p') fun b b' q q' hb => ?_
    obtain ⟨rfl, rfl, rfl⟩ := hb
    refine Rel.bind (Rel.mapM_same f hf l _ _) fun bs bs' r r' hbs => ?_
    obtain ⟨rfl, rfl, rfl⟩ := hbs
    exact Rel.pure ⟨rfl, rfl, rfl⟩

theorem run_salted (row : Array Nat) (st : State) :
    (salted row).run st = match st.salts with
      | y :: rest => .ok (Hm96.Default512.commitString (rowDigest row) y, { st with salts := rest })
      | [] => .error "the firewall's salts ran out" := by
  cases st with
  | mk sent hello committed roots linked streams rounds stopped finished salts =>
    cases salts <;> rfl

/-- **Salting a row at two alike states**: each draws its next salt, and the commit strings are of one digest. -/
theorem Rel.salted {p p' : List ByteArray} (row : Array Nat) :
    Rel ys ys' p p' (salted row) (salted row)
      (fun c c' q q' => ByteArray.empty = ByteArray.empty ∧ q = p ++ [c] ∧ q' = p' ++ [c']) := by
  intro st z' x' hD
  obtain ⟨hb, hlen, ds, hds, hle, hz, hz', hh, hh'⟩ := hD
  rw [run_salted, run_salted]
  show Both _ (match st.salts with
      | y :: rest => .ok (Hm96.Default512.commitString (rowDigest row) y, { st with salts := rest })
      | [] => .error "the firewall's salts ran out")
    (match z' with
      | y :: rest => .ok (Hm96.Default512.commitString (rowDigest row) y,
          { st with salts := rest, sent := x' })
      | [] => .error "the firewall's salts ran out")
  rw [hz, hz']
  rcases hy : ys.drop ds.length with _ | ⟨y, rest⟩ <;> rcases hy' : ys'.drop ds.length with _ | ⟨y', rest'⟩
  · exact True.intro
  · have h1 := congrArg List.length hy; have h2 := congrArg List.length hy'
    simp only [List.length_drop, List.length_cons, List.length_nil] at h1 h2
    omega
  · have h1 := congrArg List.length hy; have h2 := congrArg List.length hy'
    simp only [List.length_drop, List.length_cons, List.length_nil] at h1 h2
    omega
  · refine both_ok.mpr ⟨_, _, ⟨rfl, rfl, rfl⟩, rest', x', rfl, hb, hlen, ds ++ [rowDigest row], ?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro d hd
      rcases List.mem_append.mp hd with hd | hd
      · exact hds d hd
      · rw [List.mem_singleton.mp hd]; exact rowDigest_size row
    · have := length_le_of_drop hy
      simp only [List.length_append, List.length_singleton]
      omega
    · show rest = ys.drop (ds ++ [rowDigest row]).length
      rw [List.length_append, List.length_singleton, drop_succ_of_drop hy]
    · show rest' = ys'.drop (ds ++ [rowDigest row]).length
      rw [List.length_append, List.length_singleton, drop_succ_of_drop hy']
    · show holesOf st.sent ++ (p ++ [_]) = _
      rw [← List.append_assoc, hh, zipWith_append_left, hy]
      rfl
    · show holesOf x' ++ (p' ++ [_]) = _
      rw [← List.append_assoc, hh', zipWith_append_left, hy']
      rfl

/-- **`l.mapM f` at two alike states**, where each `f a` returns results alike under `B` and draws the commit strings
`H` of them. -/
theorem Rel.mapM {α β γ : Type} (f : α → Fw β) (H : β → List ByteArray) (B : β → γ)
    (hf : ∀ a q q', Rel ys ys' q q' (f a) (f a) (fun b b' r r' => B b = B b' ∧ r = q ++ H b ∧ r' = q' ++ H b')) :
    ∀ (l : List α) (p p' : List ByteArray), Rel ys ys' p p' (l.mapM f) (l.mapM f)
      (fun bs bs' r r' => bs.map B = bs'.map B ∧ r = p ++ bs.flatMap H ∧ r' = p' ++ bs'.flatMap H)
  | [], p, p' => by
    rw [List.mapM_nil]
    exact Rel.pure ⟨rfl, by simp, by simp⟩
  | a :: l, p, p' => by
    rw [List.mapM_cons]
    refine Rel.bind (hf a p p') fun b b' q q' hb => ?_
    obtain ⟨hB, rfl, rfl⟩ := hb
    refine Rel.bind (Rel.mapM f H B hf l _ _) fun bs bs' r r' hbs => ?_
    obtain ⟨hBs, rfl, rfl⟩ := hbs
    exact Rel.pure ⟨by simp [hB, hBs], by simp, by simp⟩

/-- `saltedTop` at two alike states: a tree top's commit strings, one per node row. -/
theorem Rel.saltedTop {p p' : List ByteArray} (top : ByteArray) :
    Rel ys ys' p p' (saltedTop top) (saltedTop top)
      (fun l l' q q' => blankTop l = blankTop l' ∧ q = p ++ l ∧ q' = p' ++ l') := by
  refine Rel.conseq (Rel.mapM Firewall.salted (fun c => [c]) (fun _ => ByteArray.empty)
    (fun row _ _ => Rel.salted row) (topRows top) p p') ?_
  rintro l l' q q' ⟨hB, rfl, rfl⟩
  exact ⟨hB, by simp, by simp⟩

/-! ## `step` at two alike states -/

/-- What each request leaves pending: no commit string. -/
abbrev Done : Unit → Unit → List ByteArray → List ByteArray → Prop := fun _ _ q q' => q = [] ∧ q' = []

theorem step_ping (s : Schedule) : Rel ys ys' [] [] (step s .ping) (step s .ping) Done := by
  rw [step]
  exact Rel.pure ⟨rfl, rfl⟩

theorem step_prime (s : Schedule) : Rel ys ys' [] [] (step s .prime) (step s .prime) Done := by
  rw [step]
  exact Rel.throw

theorem step_register (s : Schedule) : Rel ys ys' [] [] (step s .register) (step s .register) Done := by
  rw [step]
  refine Rel.bind Rel.get fun a a' q q' h => ?_
  obtain ⟨rfl, rfl, _, _, rfl, _⟩ := h
  exact Rel.ite Iff.rfl (fun _ => Rel.throw) fun _ => Rel.pure ⟨rfl, rfl⟩

theorem step_proof (s : Schedule) : Rel ys ys' [] [] (step s .proof) (step s .proof) Done := by
  rw [step]
  refine Rel.bind Rel.get fun a a' q q' h => ?_
  obtain ⟨rfl, rfl, _, _, rfl, _⟩ := h
  exact Rel.ite Iff.rfl (fun _ => Rel.throw) fun _ => Rel.pure ⟨rfl, rfl⟩

theorem step_hello (s : Schedule) (text : String) :
    Rel ys ys' [] [] (step s (.hello text)) (step s (.hello text)) Done := by
  rw [step]
  refine Rel.bind Rel.get fun a a' q q' h => ?_
  obtain ⟨rfl, rfl, _, _, rfl, _⟩ := h
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.bind Rel.get fun a a' q q' h => ?_
  obtain ⟨rfl, rfl, _, _, rfl, _⟩ := h
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  rcases hj : Lean.Json.parse text with e | v
  · exact Rel.throw_bind
  refine Rel.pure_bind ?_
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.bind Rel.lift fun _ _ q q' h => ?_
  obtain ⟨-, rfl, rfl⟩ := h
  refine Rel.bind (Rel.modify ?_) fun _ _ q q' h => ?_
  · exact fun st z x hD => ⟨z, x, rfl, hD⟩
  obtain ⟨rfl, rfl⟩ := h
  exact Rel.send_same rfl

theorem step_commit (s : Schedule) (rootF : ByteArray) (roots publics : List (String × ByteArray)) :
    Rel ys ys' [] [] (step s (.commit rootF roots publics)) (step s (.commit rootF roots publics)) Done := by
  rw [step]
  refine Rel.bind Rel.get fun a a' q q' h => ?_
  obtain ⟨rfl, rfl, _, _, rfl, _⟩ := h
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.bind (Rel.mapM_same _ (fun e r r' => ?_) _ _ _) fun c c' q q' h => ?_
  · obtain ⟨n, v⟩ := e
    refine Rel.bind Rel.lift fun k k' r₁ r₁' h => ?_
    obtain ⟨rfl, rfl, rfl⟩ := h
    exact Rel.pure ⟨rfl, rfl, rfl⟩
  obtain ⟨rfl, rfl, rfl⟩ := h
  refine Rel.bind Rel.lift fun _ _ q q' h => ?_
  obtain ⟨-, rfl, rfl⟩ := h
  refine Rel.bind (Rel.mapM _ (fun e => e.2) blankRoot (fun e r r' => ?_) _ _ _) fun t t' q q' h => ?_
  · obtain ⟨n, v⟩ := e
    refine Rel.bind (Rel.saltedTop v) fun l l' r₁ r₁' h => ?_
    obtain ⟨hb, rfl, rfl⟩ := h
    exact Rel.pure ⟨congrArg (fun u => (n, u)) hb, rfl, rfl⟩
  obtain ⟨hT, rfl, rfl⟩ := h
  refine Rel.bind Rel.get fun b b' q q' h => ?_
  obtain ⟨rfl, rfl, z, x, rfl, hD⟩ := h
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.bind (Rel.set ?_) fun _ _ q q' h => ?_
  · exact ⟨z, x, rfl, hD⟩
  obtain ⟨rfl, rfl⟩ := h
  refine Rel.send ?_ ?_ ?_
  · exact congrArg Sent.commit hT
  · simp only [holes, List.nil_append, List.append_nil]
  · simp only [holes, List.nil_append, List.append_nil]

theorem step_link (s : Schedule) (y : ByteArray) : Rel ys ys' [] [] (step s (.link y)) (step s (.link y)) Done := by
  rw [step]
  refine Rel.bind Rel.get fun a a' q q' h => ?_
  obtain ⟨rfl, rfl, _, _, rfl, _⟩ := h
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.bind Rel.get fun b b' q q' h => ?_
  obtain ⟨rfl, rfl, z, x, rfl, hD⟩ := h
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.bind (Rel.set ?_) fun _ _ q q' h => ?_
  · exact ⟨z, x, rfl, hD⟩
  obtain ⟨rfl, rfl⟩ := h
  exact Rel.send_same rfl

theorem step_openStream (s : Schedule) (stream parent : String) (label : ByteArray) (pos : Nat) :
    Rel ys ys' [] [] (step s (.openStream stream parent label pos)) (step s (.openStream stream parent label pos))
      Done := by
  rw [step]
  refine Rel.bind Rel.get fun a a' q q' h => ?_
  obtain ⟨rfl, rfl, _, _, rfl, _⟩ := h
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  generalize List.find? _ s.streams = o
  rcases o with _ | ⟨w₁, wp, wa, w₂⟩
  · exact Rel.throw
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.bind Rel.get fun b b' q q' h => ?_
  obtain ⟨rfl, rfl, z, x, rfl, hD⟩ := h
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.set ?_
  exact ⟨z, x, rfl, hD⟩

theorem step_finish (s : Schedule) : Rel ys ys' [] [] (step s .finish) (step s .finish) Done := by
  rw [step]
  refine Rel.bind Rel.get fun a a' q q' h => ?_
  obtain ⟨rfl, rfl, z, x, rfl, hD⟩ := h
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.bind Rel.lift fun _ _ q q' h => ?_
  obtain ⟨-, rfl, rfl⟩ := h
  refine Rel.bind Rel.lift fun _ _ q q' h => ?_
  obtain ⟨-, rfl, rfl⟩ := h
  refine Rel.bind (Rel.set ?_) fun _ _ q q' h => ?_
  · exact ⟨z, x, rfl, hD⟩
  obtain ⟨rfl, rfl⟩ := h
  exact Rel.send_same rfl

theorem step_round (s : Schedule) (stream : String) (content : ByteArray) (n : Nat) :
    Rel ys ys' [] [] (step s (.round stream content n)) (step s (.round stream content n)) Done := by
  rw [step]
  refine Rel.bind Rel.get fun a a' q q' h => ?_
  obtain ⟨rfl, rfl, _, _, rfl, _⟩ := h
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.bind Rel.get fun b b' q q' h => ?_
  obtain ⟨rfl, rfl, z, x, rfl, _⟩ := h
  rw [show ({ b with salts := z, sent := x } : State).streams = b.streams from rfl]
  generalize List.find? _ b.streams = o
  rcases o with _ | ⟨w, parent, k⟩
  · exact Rel.throw
  refine Rel.bind Rel.lift fun sp sp' q q' h => ?_
  obtain ⟨rfl, rfl, rfl⟩ := h
  refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
  refine Rel.bind Rel.get fun c c' q q' h => ?_
  obtain ⟨rfl, rfl, z, x, rfl, _⟩ := h
  rw [show ({ c with salts := z, sent := x } : State).roots = c.roots from rfl]
  refine Rel.ite_jp Iff.rfl ?_ fun _ hJ => ?_
  · refine Rel.bind (Rel.mapM Firewall.saltedTop (fun l => l) blankTop (fun t _ _ => Rel.saltedTop t) _ _ _)
      fun tops tops' q q' h => ?_
    obtain ⟨hT, rfl, rfl⟩ := h
    refine Rel.bind Rel.get fun d d' q q' h => ?_
    obtain ⟨rfl, rfl, _, _, rfl, _⟩ := h
    refine Rel.bind (Rel.mapM_same _ (fun e r r' => ?_) _ _ _) fun tn tn' q q' h => ?_
    · refine Rel.bind Rel.lift fun k k' r₁ r₁' h => ?_
      obtain ⟨rfl, rfl, rfl⟩ := h
      exact Rel.pure ⟨rfl, rfl, rfl⟩
    obtain ⟨rfl, rfl, rfl⟩ := h
    refine Rel.bind Rel.lift fun _ _ q q' h => ?_
    obtain ⟨-, rfl, rfl⟩ := h
    refine Rel.bind (Rel.mapM Firewall.salted (fun c => [c]) (fun _ => ByteArray.empty) (fun row _ _ => Rel.salted row)
      _ _ _)
      fun bcs bcs' q q' h => ?_
    obtain ⟨hB, rfl, rfl⟩ := h
    refine Rel.bind Rel.get fun e e' q q' h => ?_
    obtain ⟨rfl, rfl, _, _, rfl, _⟩ := h
    refine Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => ?_
    refine Rel.ite_jp Iff.rfl ?_ fun _ hJ' => ?_
    · refine Rel.bind (Rel.modify ?_) fun _ _ q q' h => ?_
      · exact fun st z x hD => ⟨z, x, rfl, hD⟩
      obtain ⟨rfl, rfl⟩ := h
      refine Rel.send ?_ ?_ ?_
      · have hB' : blankTop bcs = blankTop bcs' := hB
        show Sent.round stream (tops.map blankTop) _ (blankTop bcs) n =
          Sent.round stream (tops'.map blankTop) _ (blankTop bcs') n
        rw [hT, hB']
      · simp only [holes, List.flatMap_id', List.flatMap_singleton', List.nil_append, List.append_nil]
      · simp only [holes, List.flatMap_id', List.flatMap_singleton', List.nil_append, List.append_nil]
    · generalize List.find? _ s.streams = o
      rcases o with _ | ⟨w₁, w₂, pos, w₃⟩
      · exact Rel.throw_bind
      refine Rel.bind (Rel.send_same rfl) fun u u' r r' h => ?_
      obtain ⟨rfl, rfl⟩ := h
      exact hJ'
  · extract_lets (config := { underBinder := false }) parts table
    generalize List.find? _ c.roots = o₁
    generalize sp.caps[0]? = o₂
    rcases o₁ with _ | ⟨w₁, rootB⟩ <;> rcases o₂ with _ | cap
    · exact Rel.throw_bind
    · exact Rel.throw_bind
    · exact Rel.throw_bind
    · exact Rel.ite Iff.rfl (fun _ => Rel.throw_bind) fun _ => hJ

/-- **`step` at two alike states**: both refuse, or both return at states alike, no commit string pending. -/
theorem step_rel (s : Schedule) (req : Req) : Rel ys ys' [] [] (step s req) (step s req) Done := by
  cases req with
  | hello text => exact step_hello s text
  | link y => exact step_link s y
  | ping => exact step_ping s
  | openStream stream parent label pos => exact step_openStream s stream parent label pos
  | round stream content n => exact step_round s stream content n
  | proof => exact step_proof s
  | finish => exact step_finish s
  | register => exact step_register s
  | commit rootF roots publics => exact step_commit s rootF roots publics
  | prime => exact step_prime s

/-- One request through the firewall, as `run` takes it: decoded, then `step`. -/
def stepOf (s : Schedule) (b : ByteArray) : Fw Unit := do step s (← decode b)

theorem stepOf_rel (s : Schedule) (b : ByteArray) : Rel ys ys' [] [] (stepOf s b) (stepOf s b) Done := by
  unfold stepOf
  refine Rel.bind Rel.lift fun req req' q q' h => ?_
  obtain ⟨rfl, rfl, rfl⟩ := h
  exact step_rel s req

end Rel

/-! ## The loop -/

/-- `run`'s loop body. -/
def body (s : Schedule) (b : ByteArray) (r : State × Bool) : Id (ForInStep (State × Bool)) :=
  if r.2 = true then pure (.yield (r.1, r.2))
  else match (stepOf s b).run r.1 with
    | .ok ((), st') => pure (.yield (st', r.2))
    | .error _ => pure (.yield (r.1, true))

/-- `run`'s loop from the salts `salts`: the state and whether the session stopped. -/
def loop (s : Schedule) (reqs salts : List ByteArray) : State × Bool :=
  Id.run (forIn reqs (({ salts := salts } : State), false) (body s))

theorem run_eq (s : Schedule) (reqs salts : List ByteArray) :
    run s reqs salts = ((loop s reqs salts).1.sent, (loop s reqs salts).1.finished && !(loop s reqs salts).2) :=
  rfl

theorem forIn_yield {α β : Type} (R : β → β → Prop) (f : α → β → Id (ForInStep β))
    (hf : ∀ a r r', R r r' → ∃ u u', f a r = ForInStep.yield u ∧ f a r' = ForInStep.yield u' ∧ R u u') :
    ∀ (l : List α) (r r' : β), R r r' → R (Id.run (forIn l r f)) (Id.run (forIn l r' f))
  | [], r, r', h => by
    rw [List.forIn_nil, List.forIn_nil]
    exact h
  | a :: l, r, r', h => by
    obtain ⟨u, u', hu, hu', hR⟩ := hf a r r' h
    rw [List.forIn_cons, List.forIn_cons, hu, hu']
    exact forIn_yield R f hf l u u' hR

/-- Two loop states of one session at the salt lists `ys`, `ys'`: both stopped or neither, alike but for their salts and
what they sent, with nothing pending. -/
def LoopRel (ys ys' : List ByteArray) (r r' : State × Bool) : Prop :=
  r.2 = r'.2 ∧ ∃ z x, r'.1 = { r.1 with salts := z, sent := x } ∧ Drawn ys ys' [] [] r.1.salts r.1.sent z x

theorem body_rel (s : Schedule) (b : ByteArray) {ys ys' : List ByteArray} :
    ∀ r r', LoopRel ys ys' r r' →
      ∃ u u', body s b r = ForInStep.yield u ∧ body s b r' = ForInStep.yield u' ∧ LoopRel ys ys' u u' := by
  rintro ⟨st, stp⟩ ⟨st', stp'⟩ ⟨hs, z, x, he, hD⟩
  dsimp only at hs he hD
  subst hs he
  cases stp
  · have H := stepOf_rel (ys := ys) (ys' := ys') s b st z x hD
    simp only [body, Bool.false_eq_true, ↓reduceIte]
    rcases hm : (stepOf s b).run st with e | ⟨⟨⟩, t⟩ <;>
      rcases hm' : (stepOf s b).run { st with salts := z, sent := x } with e' | ⟨⟨⟩, t'⟩ <;>
      rw [hm, hm'] at H
    · exact ⟨_, _, rfl, rfl, rfl, z, x, rfl, hD⟩
    · exact H.elim
    · exact H.elim
    · obtain ⟨q, q', ⟨rfl, rfl⟩, z₁, x₁, rfl, hD'⟩ := H
      exact ⟨_, _, rfl, rfl, rfl, z₁, x₁, rfl, hD'⟩
  · exact ⟨_, _, rfl, rfl, rfl, z, x, rfl, hD⟩

theorem loop_rel (s : Schedule) (reqs : List ByteArray) {ys ys' : List ByteArray} (hl : ys.length = ys'.length) :
    LoopRel ys ys' (loop s reqs ys) (loop s reqs ys') :=
  forIn_yield (LoopRel ys ys') (body s) (fun b => body_rel s b) reqs _ _
    ⟨rfl, ys', #[], rfl, rfl, hl, [], (fun _ h => nomatch h), Nat.zero_le _, rfl, rfl, rfl, rfl⟩

/-! ## What the firewall sends, at every salt list -/

/-- **The firewall at two salt lists of one length**: on one request stream it finishes at both or at neither, sends the
same messages but for their commit strings, and its commit strings are those of the same 64-byte digests `ds` under
each list's salts, in salt order. -/
theorem run_resalt (s : Schedule) (reqs : List ByteArray) {ys ys' : List ByteArray} (hl : ys.length = ys'.length) :
    (run s reqs ys).2 = (run s reqs ys').2 ∧
      (run s reqs ys).1.toList.map blank = (run s reqs ys').1.toList.map blank ∧
      ∃ ds : List ByteArray, (∀ d ∈ ds, d.size = 64) ∧
        holesOf (run s reqs ys).1 = List.zipWith Hm96.Default512.commitString ds ys ∧
        holesOf (run s reqs ys').1 = List.zipWith Hm96.Default512.commitString ds ys' := by
  obtain ⟨h2, z, x, he, hb, -, ds, hds, -, -, -, hh, hh'⟩ := loop_rel s reqs hl
  rw [run_eq, run_eq]
  refine ⟨?_, ?_, ds, hds, ?_, ?_⟩
  · simp only [he, h2]
  · show (loop s reqs ys).1.sent.toList.map blank = (loop s reqs ys').1.sent.toList.map blank
    rw [he]
    exact hb
  · exact (List.append_nil _).symm.trans hh
  · show holesOf (loop s reqs ys').1.sent = _
    rw [he]
    exact (List.append_nil _).symm.trans hh'

/-- **One run fixes them all**: if at the salts `ys₀` the firewall finishes, sends the messages `σ` but for their
commit strings, and its commit strings are those of the 64-byte digests `D` under `ys₀`, then at every salt list `ys`
of that length it finishes and sends `σ` filled with `D`'s commit strings under `ys`. -/
theorem run_of_one (s : Schedule) (reqs : List ByteArray) {nh : Nat} (σ : List Sent) (D ys₀ : Fin nh → ByteArray)
    (hD : ∀ i, (D i).size = 64) (hfin : (run s reqs (List.ofFn ys₀)).2 = true)
    (hσ : (run s reqs (List.ofFn ys₀)).1.toList.map blank = σ)
    (hh : holesOf (run s reqs (List.ofFn ys₀)).1 = List.ofFn fun i => Hm96.Default512.commitString (D i) (ys₀ i))
    (ys : Fin nh → ByteArray) :
    run s reqs (List.ofFn ys) =
      ((fillAll σ (List.ofFn fun i => Hm96.Default512.commitString (D i) (ys i))).toArray, true) := by
  obtain ⟨hf, hb, ds, hds, h1, h2⟩ :=
    run_resalt s reqs (ys := List.ofFn ys₀) (ys' := List.ofFn ys) (by simp only [List.length_ofFn])
  have E := h1.symm.trans hh
  have hlen : nh ≤ ds.length := by
    have := congrArg List.length E
    simp only [List.length_zipWith, List.length_ofFn] at this
    omega
  have hds' : ∀ (i : Nat) (hi : i < nh), ds[i]'(by omega) = D ⟨i, hi⟩ := by
    intro i hi
    have := List.getElem_of_eq E (i := i) (by simp only [List.length_zipWith, List.length_ofFn]; omega)
    simp only [List.getElem_zipWith, List.getElem_ofFn] at this
    exact commitString_inj (hds _ (List.getElem_mem _)) (hD _) this
  have hholes : holesOf (run s reqs (List.ofFn ys)).1 =
      List.ofFn fun i => Hm96.Default512.commitString (D i) (ys i) := by
    rw [h2]
    apply List.ext_getElem
    · simp only [List.length_zipWith, List.length_ofFn]
      omega
    · intro i hi hi'
      simp only [List.getElem_zipWith, List.getElem_ofFn]
      rw [hds' i (by simpa using hi')]
  have e3 := fillAll_blank_holes (run s reqs (List.ofFn ys)).1.toList []
  rw [List.append_nil, ← hb, hσ] at e3
  change fillAll σ (holesOf (run s reqs (List.ofFn ys)).1) = _ at e3
  rw [hholes] at e3
  refine Prod.ext ?_ ?_
  · show (run s reqs (List.ofFn ys)).1 = (fillAll σ _).toArray
    rw [e3, Array.toArray_toList]
  · show (run s reqs (List.ofFn ys)).2 = true
    rw [← hf, hfin]

end FlockSoundness.Discharge.Recursion.FirewallRun
