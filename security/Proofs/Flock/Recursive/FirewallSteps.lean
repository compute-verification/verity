module

public import Proofs.Flock.Recursive.FirewallObj
public import Proofs.Flock.Recursive.FirewallRun
import Proofs.Flock.Recursive.FirewallWire

/-!
# The ZK server's steps on requests it takes

`FirewallTemplate` shows what the ZK server sends wherever it doesn't refuse; this file shows where it doesn't. `Runs m st
Q`: from the state `st`, `m` returns at a state `Q` holds of. `Draws m rows P`: `m` samples one salt for each of `rows`, in
order, and changes nothing else, its result and the rows' commit strings `P`. `Goes s reqs st t`: the ZK server's loop
takes every request of `reqs` from `st`, ending at `t`. Each `run_*` is one request the ZK server takes, at the state
after it.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.FirewallSteps

open Lean (Json)
open Flock Flock.Firewall

/-! ## Total correctness of the ZK server's state monad -/

/-- From `st`, `m` returns `a` at a state `t` with `Q a t`. -/
def Runs {α : Type} (m : Fw α) (st : State) (Q : α → State → Prop) : Prop :=
  ∃ a t, m.run st = .ok (a, t) ∧ Q a t

theorem Runs.bind {α β : Type} {m : Fw α} {f : α → Fw β} {st : State} {Q : α → State → Prop}
    {R : β → State → Prop} (h : Runs m st Q) (hf : ∀ a t, Q a t → Runs (f a) t R) : Runs (m >>= f) st R := by
  obtain ⟨a, t, hm, hq⟩ := h
  obtain ⟨b, u, hb, hr⟩ := hf a t hq
  exact ⟨b, u, by rw [StateT.run_bind, hm]; exact hb, hr⟩

theorem Runs.pure {α : Type} {a : α} {st : State} {Q : α → State → Prop} (h : Q a st) :
    Runs (Pure.pure a : Fw α) st Q :=
  ⟨a, st, rfl, h⟩

theorem Runs.pure_bind {α β : Type} {a : α} {f : α → Fw β} {st : State} {R : β → State → Prop}
    (h : Runs (f a) st R) : Runs (Pure.pure a >>= f) st R := h

theorem Runs.conseq {α : Type} {m : Fw α} {st : State} {Q Q' : α → State → Prop} (h : Runs m st Q)
    (hQ : ∀ a t, Q a t → Q' a t) : Runs m st Q' := by
  obtain ⟨a, t, hm, hq⟩ := h
  exact ⟨a, t, hm, hQ a t hq⟩

theorem Runs.ite_neg {α : Type} {c : Prop} [Decidable c] {a b : Fw α} {st : State} {Q : α → State → Prop}
    (hc : ¬c) (h : Runs b st Q) : Runs (if c then a else b) st Q := by
  rw [ite_eq_right hc]
  exact h

theorem Runs.ite_pos {α : Type} {c : Prop} [Decidable c] {a b : Fw α} {st : State} {Q : α → State → Prop}
    (hc : c) (h : Runs a st Q) : Runs (if c then a else b) st Q := by
  rw [ite_eq_left hc]
  exact h

theorem Runs.get {β : Type} {f : State → Fw β} {st : State} {R : β → State → Prop} (h : Runs (f st) st R) :
    Runs (MonadState.get >>= f) st R := h

theorem Runs.set {st t : State} {Q : PUnit → State → Prop} (h : Q ⟨⟩ t) :
    Runs (MonadStateOf.set t : Fw PUnit) st Q :=
  ⟨⟨⟩, t, rfl, h⟩

theorem Runs.modify {f : State → State} {st : State} {Q : PUnit → State → Prop} (h : Q ⟨⟩ (f st)) :
    Runs (modify f : Fw PUnit) st Q :=
  ⟨⟨⟩, f st, rfl, h⟩

theorem Runs.send {m : Sent} {st : State} {Q : PUnit → State → Prop} (h : Q ⟨⟩ { st with sent := st.sent.push m }) :
    Runs (send m) st Q :=
  ⟨⟨⟩, _, rfl, h⟩

theorem Runs.lift {α : Type} {x : V α} {a : α} {st : State} {Q : α → State → Prop} (hx : x = .ok a) (h : Q a st) :
    Runs (liftM x : Fw α) st Q := by
  subst hx
  exact ⟨a, st, rfl, h⟩

/-- A `do` block's `if` with a join point `b` after it, shown once for every state `J` holds of: `st` when the `if`'s
branch is skipped, and the branch `a` may end in it. -/
theorem Runs.ite_jp {α : Type} {c : Prop} [Decidable c] {a b : Fw α} {st : State} {Q : α → State → Prop}
    (J : State → Prop) (hb : ∀ t, J t → Runs b t Q) (hv : ¬c → J st)
    (ha : c → (∀ t, J t → Runs b t Q) → Runs a st Q) : Runs (if c then a else b) st Q := by
  by_cases h : c
  · exact Runs.ite_pos h (ha h hb)
  · exact Runs.ite_neg h (hb st (hv h))

theorem Runs.lift_bind {α β : Type} {x : V α} {a : α} {f : α → Fw β} {st : State} {R : β → State → Prop}
    (hx : x = .ok a) (h : Runs (f a) st R) : Runs (liftM x >>= f) st R := by
  subst hx
  exact h

theorem Runs.modify_bind {β : Type} {g : State → State} {f : PUnit → Fw β} {st : State} {R : β → State → Prop}
    (h : Runs (f ⟨⟩) (g st) R) : Runs (_root_.modify g >>= f) st R := h

theorem Runs.set_bind {β : Type} {t : State} {f : PUnit → Fw β} {st : State} {R : β → State → Prop}
    (h : Runs (f ⟨⟩) t R) : Runs (MonadStateOf.set t >>= f) st R := h

theorem Runs.send_bind {β : Type} {m : Sent} {f : PUnit → Fw β} {st : State} {R : β → State → Prop}
    (h : Runs (f ⟨⟩) { st with sent := st.sent.push m } R) : Runs (Flock.Firewall.send m >>= f) st R := h

/-! ## Salts sampled in order -/

/-- The commit strings of `rows` under the salts `ys`, row by row. -/
def cs (rows : List (Array Nat)) (ys : List ByteArray) : List ByteArray :=
  List.zipWith (fun row y => Hm96.Default512.commitString (rowDigest row) y) rows ys

theorem cs_append (a b : List (Array Nat)) (ys : List ByteArray) :
    cs (a ++ b) ys = cs a ys ++ cs b (ys.drop a.length) :=
  FirewallRun.zipWith_append_left _ a b ys

/-- `m` samples one salt for each of `rows`, in order, and changes nothing else; `P` holds of its result and the rows'
commit strings. -/
def Draws {α : Type} (m : Fw α) (rows : List (Array Nat)) (P : α → List ByteArray → Prop) : Prop :=
  ∀ st : State, rows.length ≤ st.salts.length →
    ∃ a, m.run st = .ok (a, { st with salts := st.salts.drop rows.length }) ∧ P a (cs rows st.salts)

theorem Draws.runs {α : Type} {m : Fw α} {rows : List (Array Nat)} {P : α → List ByteArray → Prop}
    (h : Draws m rows P) {st : State} (hl : rows.length ≤ st.salts.length) :
    Runs m st (fun a t => t = { st with salts := st.salts.drop rows.length } ∧ P a (cs rows st.salts)) := by
  obtain ⟨a, hm, hp⟩ := h st hl
  exact ⟨a, _, hm, rfl, hp⟩

theorem draws_salted (row : Array Nat) : Draws (salted row) [row] (fun c l => l = [c]) := by
  intro st hl
  obtain ⟨sent, hello, committed, roots, linked, streams, rounds, stopped, finished, salts⟩ := st
  cases salts with
  | nil => exact absurd hl (by simp)
  | cons y rest => exact ⟨_, FirewallRun.run_salted row _, rfl⟩

theorem draws_lift {α : Type} {x : V α} {a : α} (hx : x = .ok a) :
    Draws (liftM x : Fw α) [] (fun b l => b = a ∧ l = []) := by
  subst hx
  intro st _
  exact ⟨a, rfl, rfl, rfl⟩

theorem draws_mapM {α β γ : Type} (f : α → Fw β) (R : α → List (Array Nat)) (Hs : β → List ByteArray) (K : β → γ)
    (k : α → γ) : ∀ (l : List α), (∀ a ∈ l, Draws (f a) (R a) (fun b c => Hs b = c ∧ K b = k a)) →
      Draws (l.mapM f) (l.flatMap R) (fun bs c => bs.flatMap Hs = c ∧ bs.map K = l.map k)
  | [], _ => fun st _ => ⟨[], rfl, rfl, rfl⟩
  | a :: l, hf => by
    intro st hl
    rw [List.flatMap_cons, List.length_append] at hl
    obtain ⟨b, hb, hb1, hb2⟩ := hf a (List.mem_cons_self ..) st (by omega)
    obtain ⟨bs, hbs, hbs1, hbs2⟩ := draws_mapM f R Hs K k l (fun a' ha' => hf a' (List.mem_cons_of_mem _ ha'))
      { st with salts := st.salts.drop (R a).length } (by simp only [List.length_drop]; omega)
    refine ⟨b :: bs, ?_, ?_, ?_⟩
    · rw [List.mapM_cons, StateT.run_bind, hb, ok_bind', StateT.run_bind, hbs]
      simp only [List.drop_drop, List.flatMap_cons, List.length_append]
      rfl
    · rw [List.flatMap_cons, List.flatMap_cons, hb1, hbs1, cs_append]
    · rw [List.map_cons, List.map_cons, hb2, hbs2]
where
  ok_bind' {δ ε : Type} (a : δ) (g : δ → V ε) : (Except.ok a : V δ) >>= g = g a := rfl

theorem draws_salteds (rows : List (Array Nat)) : Draws (rows.mapM salted) rows (fun l c => l = c) := by
  have h := draws_mapM salted (fun row => [row]) (fun c => [c]) (fun _ => ()) (fun _ => ()) rows
    (fun row _ => fun st hl => by
      obtain ⟨c, hc, hc1⟩ := draws_salted row st hl
      exact ⟨c, hc, hc1.symm, rfl⟩)
  rw [List.flatMap_singleton'] at h
  intro st hl
  obtain ⟨l, hm, hl1, _⟩ := h st hl
  exact ⟨l, hm, by rw [← hl1, List.flatMap_singleton']⟩

theorem draws_saltedTop (top : ByteArray) : Draws (saltedTop top) (topRows top) (fun l c => l = c) :=
  draws_salteds (topRows top)

/-! ## The ZK server's loop -/

/-- The ZK server's loop takes each of `reqs` in turn from `st`, ending at `t`. -/
def Goes (s : Schedule) : List ByteArray → State → State → Prop
  | [], st, t => t = st
  | b :: bs, st, t => ∃ u, (FirewallRun.stepOf s b).run st = .ok ((), u) ∧ Goes s bs u t

theorem goes_append (s : Schedule) : ∀ (a b : List ByteArray) {st t u : State}, Goes s a st t → Goes s b t u →
    Goes s (a ++ b) st u
  | [], _, _, _, _, ha, hb => by subst ha; exact hb
  | _ :: a, b, _, _, _, ⟨v, hv, ha⟩, hb => ⟨v, hv, goes_append s a b ha hb⟩

theorem forIn_goes (s : Schedule) : ∀ (reqs : List ByteArray) {st t : State}, Goes s reqs st t →
    Id.run (forIn reqs (st, false) (FirewallRun.body s)) = (t, false)
  | [], _, _, h => by subst h; rfl
  | b :: bs, st, t, ⟨u, hu, h⟩ => by
    rw [List.forIn_cons]
    have e : FirewallRun.body s b (st, false) = .yield (u, false) := by
      simp only [FirewallRun.body, Bool.false_eq_true, ↓reduceIte, hu]
      rfl
    rw [e]
    exact forIn_goes s bs h

/-- **The ZK server finishes where its loop takes every request** and ends finished. -/
theorem run_of_goes {s : Schedule} {reqs ys : List ByteArray} {t : State} (h : Goes s reqs { salts := ys } t) :
    run s reqs ys = (t.sent, t.finished) := by
  rw [FirewallRun.run_eq]
  unfold FirewallRun.loop
  rw [forIn_goes s reqs h]
  simp

theorem goes_one {s : Schedule} {b : ByteArray} {r : Req} (hb : decode b = .ok r) {st t : State}
    (h : (step s r).run st = .ok ((), t)) : Goes s [b] st t :=
  ⟨t, by unfold FirewallRun.stepOf; rw [StateT.run_bind, show (liftM (decode b) : Fw Req).run st = .ok (r, st) by
    rw [hb]; rfl]; exact h, rfl⟩

/-! ## The requests `decode` reads -/

theorem decode_putPing {b : ByteArray} (hb : b.size < 2 ^ 32) : decode (putPing b) = .ok .ping := by
  simp only [putPing, decode, FirewallWire.tag_size, FirewallWire.tag_get, Bool.false_eq_true, ite_false]
  rw [StateT.run_bind, FirewallWire.rdBytes_at (FirewallWire.at_tag 9 _) hb, FirewallWire.ok_bind]
  rfl

theorem decode_putProof {table : String} {rep : Nat} {proof : ByteArray} (h1 : table.toUTF8.size < 2 ^ 32)
    (h2 : rep < 2 ^ 32) (h3 : proof.size < 2 ^ 32) : decode (putProof table rep proof) = .ok .proof := by
  have a := FirewallWire.at_tag 5 (putStr table ++ (putLE 4 rep ++ putBytes proof))
  simp only [putProof, decode, FirewallWire.tag_size, FirewallWire.tag_get, Bool.false_eq_true, ite_false]
  rw [StateT.run_bind, FirewallWire.rdStr_at a.left h1, FirewallWire.ok_bind, StateT.run_bind,
    FirewallWire.rdU32_at a.right.left h2, FirewallWire.ok_bind, StateT.run_bind,
    FirewallWire.rdBytes_at a.right.right h3, FirewallWire.ok_bind]
  rfl

/-! ## The requests the ZK server takes -/

theorem Runs.run_eq {m : Fw Unit} {st t₀ : State} (h : Runs m st fun _ t => t = t₀) : m.run st = .ok ((), t₀) := by
  obtain ⟨⟨⟩, t, hm, rfl⟩ := h
  exact hm

theorem pinned_ok {what : String} {got want : Json} (h : canonForm got = canonForm want) :
    pinned what got want = .ok () := by
  simp only [pinned, h, bne_self_eq_false, Bool.false_eq_true, ite_false]
  rfl

theorem nodes_ok {top : ByteArray} (h : isCap top = true) : nodes top = .ok (top.size / NODE_BYTES) := by
  simp [nodes, h]
  rfl

/-- `l.mapM f` where each `f a` returns `k a` and leaves the state. -/
theorem run_mapM_pure {α β : Type} (f : α → Fw β) (k : α → β) :
    ∀ (l : List α), (∀ a ∈ l, ∀ st, (f a).run st = .ok (k a, st)) → ∀ st, (l.mapM f).run st = .ok (l.map k, st)
  | [], _, _ => rfl
  | a :: l, hf, st => by
    rw [List.mapM_cons, StateT.run_bind, hf a (List.mem_cons_self ..) st]
    show (do let p ← (List.mapM f l).run st; (Pure.pure (k a :: p.1) : Fw (List β)).run p.2) = _
    rw [run_mapM_pure f k l (fun a' ha' => hf a' (List.mem_cons_of_mem _ ha')) st]
    rfl

theorem eraseDups_of_nodup {l : List String} (h : l.Nodup) : l.eraseDups = l := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.eraseDups_cons]
    have hl := List.nodup_cons.mp h
    rw [List.filter_eq_self.mpr fun b hb => ?_, ih hl.2]
    have hne : b ≠ a := fun e => hl.1 (e ▸ hb)
    simpa using hne

theorem run_ping (s : Schedule) (st : State) : (step s .ping).run st = .ok ((), st) := rfl

theorem run_register (s : Schedule) {st : State} (hf : st.finished = false) :
    (step s .register).run st = .ok ((), st) := by
  refine Runs.run_eq ?_
  rw [step]
  exact Runs.get (Runs.ite_neg (by simp [hf]) (Runs.pure rfl))

theorem run_proof (s : Schedule) {st : State} (hf : st.finished = false) : (step s .proof).run st = .ok ((), st) := by
  refine Runs.run_eq ?_
  rw [step]
  exact Runs.get (Runs.ite_neg (by simp [hf]) (Runs.pure rfl))

/-- **Hello**: at a schedule's Hello as `FirewallObj.hello` takes it, the worker's Hello with a plain Σ. -/
theorem run_hello (s : Schedule) {H L : FirewallObj.Tree} {A B : List ((_ : String) × Json)}
    (he : s.hello = .obj ⟨⟨H⟩⟩) (hH : H.Ordered) (hpH : plainObj H = true) (hL : L.Ordered)
    (hpL : plainObj L = true) (hsplit : H.toListModel = A ++ ⟨"link", .obj ⟨⟨L⟩⟩⟩ :: B)
    (hs : ∀ e ∈ L.toListModel, e.1 ≠ "sigma") (hc : ∀ e ∈ H.toListModel, e.1 ≠ "coins") {sigma : String}
    (hsig : plainStr sigma = true) {st : State} (hf : st.finished = false) (hh : st.hello = false) :
    (step s (.hello (printJson (helloWith s.hello sigma)))).run st =
      .ok ((), { st with hello := true, sent := st.sent.push (.hello s.hello) }) := by
  obtain ⟨hp, T, hT, hsg, hco, hcan⟩ := FirewallObj.hello hH hpH hL hpL hsplit hs hc hsig
  rw [← he] at hp hT hco hcan
  refine Runs.run_eq ?_
  rw [step]
  refine Runs.get (Runs.ite_neg (by simp [hf]) ?_)
  refine Runs.get (Runs.ite_neg (by simp [hh]) ?_)
  rcases hj : Flock.JsonParse.parse (printJson (helloWith s.hello sigma)) with e | j <;>
    rw [FirewallJson.parse_printJson hp] at hj <;> cases hj
  refine Runs.pure_bind ?_
  refine Runs.ite_neg (by simp [hT, hsg]) ?_
  refine Runs.ite_neg (by simp [hco]) ?_
  refine Runs.ite_neg (by simp [FirewallJson.fractional_printJson hp]) ?_
  exact Runs.lift_bind (pinned_ok (by rw [hT]; exact hcan)) (Runs.modify_bind (Runs.send rfl))

/-- **Commit** of tables with distinct names in the schedule's order, each top a whole number of nodes, their node counts
the schedule's roots, after Hello and before any other Commit: it samples a salt for each node and sends each table's
commit strings. -/
theorem run_commit (s : Schedule) (rootF : ByteArray) (roots : List (String × ByteArray)) {st : State}
    (hf : st.finished = false) (hh : st.hello = true) (hc : st.committed = false) (hnd : (roots.map (·.1)).Nodup)
    (hcap : ∀ e ∈ roots, isCap e.2 = true)
    (hpin : canonForm (Json.mkObj (roots.map fun e => (e.1, jnat (e.2.size / NODE_BYTES)))) = canonForm s.roots)
    (hnat : ∀ e ∈ roots, natOf? (s.roots.getObjValD e.1) = some (e.2.size / NODE_BYTES))
    (hord : roots.map (·.1) = s.tables) (hl : (roots.flatMap fun e => topRows e.2).length ≤ st.salts.length) :
    Runs (step s (.commit rootF roots [])) st fun _ t => ∃ tops,
      t = { st with committed := true, roots := roots,
                    salts := st.salts.drop (roots.flatMap fun e => topRows e.2).length,
                    sent := st.sent.push (.commit tops) } ∧
      tops.map (·.1) = roots.map (·.1) ∧ tops.flatMap (·.2) = cs (roots.flatMap fun e => topRows e.2) st.salts := by
  rw [step]
  refine Runs.get (Runs.ite_neg (by simp [hf]) ?_)
  refine Runs.ite_neg (by simp) ?_
  refine Runs.ite_neg (by simp [eraseDups_of_nodup hnd]) ?_
  refine Runs.bind (Q := fun c t => c = roots.map (fun e => (e.1, jnat (e.2.size / NODE_BYTES))) ∧ t = st)
    ⟨_, _, run_mapM_pure _ (fun e => (e.1, jnat (e.2.size / NODE_BYTES))) roots ?_ st, rfl, rfl⟩ ?_
  · rintro ⟨n, v⟩ hv st'
    show (liftM (nodes v) >>= fun c => Pure.pure (n, jnat c) : Fw _).run st' = _
    rw [nodes_ok (hcap _ hv)]
    rfl
  rintro counts t ⟨rfl, rfl⟩
  refine Runs.lift_bind (pinned_ok hpin) ?_
  refine Runs.ite_neg ?_ ?_
  · simp only [List.any_eq_true, not_exists, not_and]
    rintro ⟨n, v⟩ hv
    simp [hnat _ hv]
  refine Runs.ite_neg (by simp [hord]) ?_
  refine Runs.bind ((draws_mapM (α := String × ByteArray) (β := String × List ByteArray) (γ := String) _
    (fun e => topRows e.2) (fun b => b.2) (fun b => b.1) (fun e => e.1) roots ?_).runs hl) ?_
  · rintro ⟨n, v⟩ _ st' hl'
    obtain ⟨l, hm, hl1⟩ := draws_saltedTop v st' hl'
    refine ⟨(n, l), ?_, hl1, rfl⟩
    show (saltedTop v >>= fun l => Pure.pure (n, l) : Fw _).run st' = _
    rw [StateT.run_bind, hm]
    rfl
  rintro tops t ⟨rfl, hT1, hT2⟩
  refine Runs.get (Runs.ite_neg (by simp [hh, hc]) ?_)
  exact Runs.set_bind (Runs.send ⟨tops, rfl, hT2, hT1⟩)

theorem zeros_loop (n : Nat) : ∀ (i : Nat) (r : List UInt8), (∀ x ∈ r, x = 0) →
    ∀ x ∈ ByteArray.toList.loop ⟨Array.replicate n 0⟩ i r, x = 0 := by
  intro i r
  induction i, r using ByteArray.toList.loop.induct (bs := ⟨Array.replicate n 0⟩) with
  | case1 i r hi ih =>
    intro hr
    rw [ByteArray.toList.loop, ite_eq_left hi]
    refine ih fun x hx => ?_
    rcases List.mem_cons.mp hx with rfl | hx
    · have hi' : i < n := by simpa [ByteArray.size] using hi
      simp [ByteArray.get!, hi']
    · exact hr x hx
  | case2 i r hi =>
    intro hr
    rw [ByteArray.toList.loop, ite_eq_right hi]
    simpa using hr

theorem zeros_any (n : Nat) : (ByteArray.mk (Array.replicate n 0)).toList.any (· != 0) = false := by
  rw [List.any_eq_false]
  intro x hx
  simp [zeros_loop n 0 [] (fun _ h => nomatch h) x hx]

/-- **Link** with the schedule's count of zero elements, after Commit and before any other Link. -/
theorem run_link (s : Schedule) {st : State} (hf : st.finished = false) (hc : st.committed = true)
    (hl : st.linked = false) :
    (step s (.link ⟨Array.replicate (16 * s.y) 0⟩)).run st =
      .ok ((), { st with linked := true, sent := st.sent.push (.link s.y) }) := by
  refine Runs.run_eq ?_
  rw [step]
  refine Runs.get (Runs.ite_neg (by simp [hf]) ?_)
  refine Runs.ite_neg (by simp [ByteArray.size]) ?_
  refine Runs.ite_neg (by simp [zeros_any]) ?_
  refine Runs.get (Runs.ite_neg (by simp [hc, hl]) ?_)
  exact Runs.set_bind (Runs.send rfl)

/-- **Open** of a stream of the schedule, at its parent and position, not opened before. -/
theorem run_open (s : Schedule) {stream parent : String} {label : ByteArray} {pos : Nat} {st : State} {w : String}
    {c : Nat} (hf : st.finished = false) (hfind : s.streams.find? (·.1 == stream) = some (w, parent, pos, c))
    (hnew : st.streams.any (·.1 == stream) = false) :
    (step s (.openStream stream parent label pos)).run st =
      .ok ((), { st with streams := st.streams ++ [(stream, parent, 0)] }) := by
  refine Runs.run_eq ?_
  rw [step]
  refine Runs.get (Runs.ite_neg (by simp [hf]) ?_)
  generalize ho : List.find? _ s.streams = o
  rw [hfind] at ho
  subst ho
  refine Runs.ite_neg (by simp) ?_
  exact Runs.get (Runs.ite_neg (by simp [hnew]) (Runs.set rfl))

/-- **Finish** after the link exchange and every round, each stream with the schedule's count of rounds. -/
theorem run_finish (s : Schedule) {st : State} (hf : st.finished = false) (hc : st.committed = true)
    (hl : st.linked = true) (hr : st.rounds = s.rounds.size)
    (hpin : canonForm (Json.mkObj (st.streams.map fun e => (e.1, jnat e.2.2))) =
      canonForm (Json.mkObj (s.streams.map fun e => (e.1, jnat e.2.2.2)))) :
    (step s .finish).run st = .ok ((), { st with finished := true, sent := st.sent.push (.finish true) }) := by
  refine Runs.run_eq ?_
  rw [step]
  refine Runs.get (Runs.ite_neg (by simp [hf]) ?_)
  refine Runs.ite_neg (by simp [hc, hl]) ?_
  refine Runs.lift_bind (pinned_ok (by rw [hr])) ?_
  refine Runs.ite_neg (by simp [hr]) ?_
  exact Runs.lift_bind (pinned_ok hpin) (Runs.set_bind (Runs.send rfl))

theorem bne_self (a : ByteArray) : (a != a) = false := by
  show (!(a == a)) = false
  rw [FirewallWire.beq_iff.mpr rfl]
  rfl

theorem some_bne_self (a : ByteArray) : (some a != some a) = false := bne_self a

/-- **A round** on an opened stream whose content `split` reads, as the schedule's entry has it: its binding's cap and
its first copies the committed top of its stream's table, its other caps fresh tops of the entry's node counts, its
message the entry's rows. It samples a salt for each node of its fresh tops and each message row, and sends the round
after its stream's `Open` where it is the stream's first. -/
theorem run_round (s : Schedule) (stream : String) (content : ByteArray) (n : Nat) {st : State} {w parent : String}
    {k : Nat} {sp : Split} (hf : st.finished = false)
    (hfind : st.streams.find? (·.1 == stream) = some (w, parent, k))
    (hsplit : split content (k == 0 && parent == "") = .ok sp) (hn : sp.n = n)
    (hbind : (k == 0 && parent == "") = true →
      ∃ c, sp.caps[0]? = some c ∧ (st.roots.find? (·.1 == tableOf stream)).map (·.2) = some c)
    (hcopies : copiesOf s st.rounds ≤ sp.caps.size)
    (hcopy : ∀ c ∈ sp.caps.toList.take (copiesOf s st.rounds),
      some c = (st.roots.find? (·.1 == tableOf stream)).map (·.2))
    (hcaps : ∀ c ∈ sp.caps.toList, isCap c = true)
    (hspec : roundSpec (s.rounds.getD st.rounds .null) = some (stream, k, n,
      (sp.caps.toList.drop (copiesOf s st.rounds)).map (·.size / NODE_BYTES), copiesOf s st.rounds,
      (messageRows sp.free).length))
    (hpin : canonForm (s.rounds.getD st.rounds .null) = canonForm (specJson (stream, k, n,
      (sp.caps.toList.drop (copiesOf s st.rounds)).map (·.size / NODE_BYTES), copiesOf s st.rounds,
      (messageRows sp.free).length)))
    (hl : st.linked = true) (hopen : k = 0 → (s.streams.find? (·.1 == stream)).isSome)
    (hsalts : ((sp.caps.toList.drop (copiesOf s st.rounds)).flatMap topRows).length +
      (messageRows sp.free).length ≤ st.salts.length) :
    Runs (step s (.round stream content n)) st fun _ t =>
      ∃ (x : Array Sent) (tops : List (List ByteArray)) (bcs : List ByteArray),
      t = { st with
        salts := (st.salts.drop ((sp.caps.toList.drop (copiesOf s st.rounds)).flatMap topRows).length).drop
          (messageRows sp.free).length,
        rounds := st.rounds + 1,
        streams := st.streams.map fun e => if e.1 == stream then (e.1, e.2.1, e.2.2 + 1) else e,
        sent := x.push (.round stream tops (copiesOf s st.rounds) bcs n) } ∧
      (∃ e, x.toList = st.sent.toList ++ e ∧ e.flatMap FirewallRun.holes = []) ∧
      tops.flatten = cs ((sp.caps.toList.drop (copiesOf s st.rounds)).flatMap topRows) st.salts ∧
      bcs = cs (messageRows sp.free)
        (st.salts.drop ((sp.caps.toList.drop (copiesOf s st.rounds)).flatMap topRows).length) := by
  rw [step]
  refine Runs.get (Runs.ite_neg (by simp [hf]) ?_)
  refine Runs.get ?_
  generalize ho : List.find? _ st.streams = o
  rw [hfind] at ho
  subst ho
  refine Runs.lift_bind hsplit ?_
  refine Runs.ite_neg (by simp [hn]) ?_
  refine Runs.get ?_
  refine Runs.ite_jp (fun t => t = st) (fun t ht => ?_) (fun _ => rfl) fun hfirst hJ => ?_
  · subst ht
    refine Runs.get ?_
    refine Runs.ite_neg ?_ ?_
    · simp only [Bool.or_eq_true, decide_eq_true_eq, List.any_eq_true, not_or, not_exists, not_and]
      exact ⟨by omega, fun c hc => by rw [← hcopy c hc, some_bne_self]; simp⟩
    refine Runs.bind ((draws_mapM (α := ByteArray) (β := List ByteArray) (γ := Unit) _ topRows id (fun _ => ())
      (fun _ => ()) _ fun c _ => fun st' hl' => ?_).runs (by omega)) ?_
    · obtain ⟨l, hm, hl1⟩ := draws_saltedTop c st' hl'
      exact ⟨l, hm, hl1, rfl⟩
    rintro tops t ⟨rfl, hT, -⟩
    refine Runs.bind (Q := fun c t' => c = (sp.caps.toList.drop (copiesOf s t.rounds)).map
        (fun c => jnat (c.size / NODE_BYTES)) ∧ t' = _)
      ⟨_, _, run_mapM_pure _ (fun c => jnat (c.size / NODE_BYTES)) _ ?_ _, rfl, rfl⟩ ?_
    · intro c hc st'
      show (liftM (nodes c) >>= fun x => Pure.pure (jnat x) : Fw _).run st' = _
      rw [nodes_ok (hcaps c (List.mem_of_mem_drop hc))]
      rfl
    rintro topNodes t' ⟨rfl, rfl⟩
    refine Runs.lift_bind (pinned_ok ?_) ?_
    · rw [hpin]
      simp only [specJson, List.map_map]
      rfl
    refine Runs.ite_neg (fun h => h hspec) ?_
    refine Runs.bind ((draws_salteds (messageRows sp.free)).runs (by simp only [List.length_drop]; omega)) ?_
    rintro bcs t₂ ⟨ht₂, hB⟩
    subst ht₂
    refine Runs.get (Runs.ite_neg (by simp [hl]) ?_)
    refine Runs.ite_jp (fun u => ∃ x, u = { t with
          sent := x
          salts := (t.salts.drop ((sp.caps.toList.drop (copiesOf s t.rounds)).flatMap topRows).length).drop
            (messageRows sp.free).length } ∧
        ∃ e, x.toList = t.sent.toList ++ e ∧ e.flatMap FirewallRun.holes = [])
      (fun u hu => ?_) (fun _ => ⟨t.sent, rfl, [], by simp, rfl⟩) fun hk hJ => ?_
    · obtain ⟨x, hu, e, hx, he⟩ := hu
      subst hu
      exact Runs.modify_bind (Runs.send ⟨x, tops, bcs, rfl, ⟨e, hx, he⟩, by rw [← List.flatMap_id]; exact hT, hB⟩)
    · obtain ⟨⟨w', wp, pos, c'⟩, hw⟩ := Option.isSome_iff_exists.mp (hopen (by simpa using hk))
      generalize ho : List.find? _ s.streams = o
      rw [hw] at ho
      subst ho
      exact Runs.send_bind (hJ _ ⟨_, rfl, [.openStream stream wp pos], by simp, rfl⟩)
  · obtain ⟨c, hc0, hroot⟩ := hbind hfirst
    generalize List.find? _ st.roots = o₁ at hroot hJ ⊢
    generalize sp.caps[0]? = o₂ at hc0
    subst hc0
    rcases o₁ with _ | ⟨w₁, rootB⟩
    · cases hroot
    cases hroot
    exact Runs.ite_neg (by simp [bne_self]) (hJ st rfl)

end FlockSoundness.Discharge.Recursion.FirewallSteps
