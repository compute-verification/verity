module

public import Properties.Spot.PortsHeld
public import Proofs.Properties.Spot
public import Proofs.Properties.Spot.RootHeld
public import Proofs.Flock.Soundness.Refine.Rep
import Spot.PortsOnce
import Mathlib.Data.List.Nodup
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String

/-! # Proof that each member's slice root is the committed record's root for its port (`Properties.Spot.MembersRootsRecord`) -/

@[expose] public section

open Lean (Json)
open Spot
open FlockSoundness.Refine (ebind_ok)

namespace Properties.Spot.RootsRecordProof

/-! ## From the per-member form's accept to R3 -/

theorem check_of_checkAt {p : OneStage.Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : OneStage.Evidence}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : OneStage.checkAt p c (some ξ) e = .accept pub prof) :
    OneStage.check p c e = .accept pub prof := by
  unfold OneStage.checkAt at h
  split at h
  · cases h
  · exact h

/-- **Where the per-member form's check accepts, the committed record keeps R3** (`acceptMeans_iff`, at v1's check on
the run's verdicts). -/
theorem r3_of_checkMembersAt {p : OneStageZk.Params} {ms : Array OneStageZk.Member} {c : OneStage.Commitment}
    {ξ : ByteArray} {e : OneStageZk.Evidence} {sts : Array OneStageZk.StatementFiles} {bs : Core.Blobs}
    {pub : OneStage.Public} {prof : Core.Integrity}
    (h : OneStageZk.checkMembersAt p ms c ξ e sts bs = .accept pub prof) : R3 c.record p.base.verifier := by
  obtain ⟨_, -, -, hc⟩ := OneStageZk.checkMembersAt_accept h
  obtain ⟨_, _, _, hreg, -⟩ := (acceptMeans_iff _ _ _ _ _).1 (check_of_checkAt hc)
  exact hreg.2.2.1

/-! ## R3: no two roots of the record name one port -/

/-- One step of `portsOf`'s fold: the root's port, unless a port `jeq` to it is already held. -/
def step (acc : Array Json) (r : Json) : Array Json :=
  if acc.any (jeq · (getD r "port")) then acc else acc.push (getD r "port")

theorem portsOf_eq (rec : Json) : portsOf rec = (rootsOf rec).toList.foldl step #[] := by
  rw [Array.foldl_toList]; rfl

theorem size_fold_le : ∀ (l : List Json) (acc : Array Json), (l.foldl step acc).size ≤ acc.size + l.length
  | [], acc => by simp
  | x :: l, acc => by
    simp only [List.foldl_cons, List.length_cons]
    have := size_fold_le l (step acc x)
    have hs : (step acc x).size ≤ acc.size + 1 := by unfold step; split <;> simp
    omega

theorem distinct_of_size : ∀ (l : List Json) (acc : Array Json), (l.foldl step acc).size = acc.size + l.length →
    l.Pairwise (fun a b => jeq (getD a "port") (getD b "port") = false) ∧
      ∀ r ∈ l, acc.any (jeq · (getD r "port")) = false
  | [], _, _ => by simp
  | x :: l, acc, h => by
    simp only [List.foldl_cons, List.length_cons] at h
    by_cases hx : acc.any (jeq · (getD x "port")) = true
    · have e : step acc x = acc := by simp [step, hx]
      rw [e] at h
      have := size_fold_le l acc
      omega
    · have e : step acc x = acc.push (getD x "port") := by simp [step, hx]
      rw [e] at h
      obtain ⟨hp, ha⟩ := distinct_of_size l _ (by rw [h, Array.size_push]; omega)
      refine ⟨List.pairwise_cons.2 ⟨fun y hy => ?_, hp⟩, fun r hr => ?_⟩
      · have := ha y hy
        rw [Array.any_push, Bool.or_eq_false_iff] at this
        exact this.2
      · rcases List.mem_cons.1 hr with er | hr
        · rw [er]; exact Bool.eq_false_iff.2 hx
        · have := ha r hr
          rw [Array.any_push, Bool.or_eq_false_iff] at this
          exact this.1

theorem eq_of_pairwise {R : Json → Json → Prop} {l : List Json} (hp : l.Pairwise R) (hs : ∀ a b, R a b → R b a)
    {a b : Json} (ha : a ∈ l) (hb : b ∈ l) (hab : ¬ R a b) : a = b := by
  induction l with
  | nil => cases ha
  | cons x l ih =>
    rw [List.pairwise_cons] at hp
    rcases List.mem_cons.1 ha with ea | ha <;> rcases List.mem_cons.1 hb with eb | hb
    · rw [ea, eb]
    · rw [ea] at hab; exact absurd (hp.1 b hb) hab
    · rw [eb] at hab; exact absurd (hs _ _ (hp.1 a ha)) hab
    · exact ih hp.2 ha hb

theorem jeq_symm_false {a b : Json} (h : jeq a b = false) : jeq b a = false := by
  unfold jeq at *
  cases ha : Flock.canon a <;> cases hb : Flock.canon b <;> simp only [ha, hb] at h ⊢
  rw [beq_eq_false_iff_ne] at h ⊢
  exact Ne.symm h

theorem jeq_str_self (q : String) : jeq (.str q) (.str q) = true := by
  have hs : Flock.canon (.str q) = .ok (Flock.canonStr q) := by rw [Flock.canon]; rfl
  unfold jeq
  rw [hs]
  simp

/-- **R3 names each port at most once**: two roots of the record naming one port, as a string, are one root. -/
theorem rootsOf_port_unique {rec : Json} {v : Docket.Verifier} (h3 : R3 rec v) {r0 r1 : Json} {q : String}
    (h0 : r0 ∈ rootsOf rec) (h1 : r1 ∈ rootsOf rec) (p0 : getD r0 "port" = .str q) (p1 : getD r1 "port" = .str q) :
    r1 = r0 := by
  have hsz : (portsOf rec).size = (rootsOf rec).size := h3.2.2.2.1
  rw [portsOf_eq] at hsz
  obtain ⟨hp, -⟩ := distinct_of_size (rootsOf rec).toList #[] (by rw [hsz]; simp)
  refine eq_of_pairwise hp (fun _ _ => jeq_symm_false) (Array.mem_toList_iff.2 h1) (Array.mem_toList_iff.2 h0) ?_
  rw [p0, p1, jeq_str_self]
  simp


/-- **`x` carries the record `rec`'s root of port `q`**: a root of the record naming `q`, whose root, domain and leaves
`x` has. -/
def Carries (rec : Json) (q : String) (x : Json) : Prop :=
  ∃ r0 ∈ rootsOf rec, getD r0 "port" = .str q ∧ getD x "root" = getD r0 "root" ∧
    getD x "domain" = getD r0 "domain" ∧ getD x "leaves" = getD r0 "leaves"

theorem isRecordRoot_of_carries {rec : Json} {v : Docket.Verifier} (h3 : R3 rec v) {q : String} {x : Json}
    (h : Carries rec q x) : IsRecordRoot rec q x := by
  obtain ⟨r0, hm, hp, hr, hd, hl⟩ := h
  exact ⟨r0, hm, hp, fun r1 h1 p1 => rootsOf_port_unique h3 hm h1 hp p1, hr, hd, hl⟩

/-! ## Folds -/

theorem foldl_inv {α β : Type} (P : β → Prop) (f : β → α → β) (S : α → Prop)
    (hf : ∀ b a, S a → P b → P (f b a)) :
    ∀ (l : List α) (b : β), (∀ a ∈ l, S a) → P b → P (l.foldl f b)
  | [], _, _, hb => hb
  | a :: l, b, hl, hb =>
    foldl_inv P f S hf l (f b a) (fun x hx => hl x (List.mem_cons_of_mem _ hx)) (hf b a (hl a (by simp)) hb)

theorem foldlM_inv {ε α β : Type} (P : β → Prop) (f : β → α → Except ε β) (S : α → Prop)
    (hf : ∀ b a b', S a → f b a = .ok b' → P b → P b') :
    ∀ (l : List α) (b b' : β), (∀ a ∈ l, S a) → P b → l.foldlM f b = .ok b' → P b'
  | [], b, b', _, hb, h => by cases h; exact hb
  | a :: l, b, b', hl, hb, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨c, hc, h⟩ := ebind_ok h
    exact foldlM_inv P f S hf l c b' (fun x hx => hl x (List.mem_cons_of_mem _ hx))
      (hf b a c (hl a (by simp)) hc hb) h

theorem forIn_yield_inv {ε α σ : Type} (P : σ → Prop) {f : α → σ → Except ε (ForInStep σ)}
    (hf : ∀ a s r, P s → f a s = .ok r → ∃ t, r = .yield t ∧ P t) :
    ∀ (l : List α) (s t : σ), P s → forIn l s f = .ok t → P t
  | [], s, t, hs, h => by rw [List.forIn_nil] at h; cases h; exact hs
  | a :: l, s, t, hs, h => by
    rw [List.forIn_cons] at h
    cases hfa : f a s with
    | error e => rw [hfa] at h; cases h
    | ok r =>
      rw [hfa] at h
      obtain ⟨t', rfl, ht'⟩ := hf a s r hs hfa
      exact forIn_yield_inv P hf l t' t ht' h

/-! ## Step 3: a member's own roots (`rootStep`'s fold) -/

theorem getD_e4 (d l r : Json) (b : String) :
    getD (Json.mkObj [("domain", d), ("leaves", l), ("port", .str b), ("root", r)]) "port" = .str b ∧
    getD (Json.mkObj [("domain", d), ("leaves", l), ("port", .str b), ("root", r)]) "root" = r ∧
    getD (Json.mkObj [("domain", d), ("leaves", l), ("port", .str b), ("root", r)]) "domain" = d ∧
    getD (Json.mkObj [("domain", d), ("leaves", l), ("port", .str b), ("root", r)]) "leaves" = l := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> with_unfolding_all rfl

theorem splitsFrom_mem : ∀ (pre cs : List Char) {a b : String}, (a, b) ∈ OneStageZk.splitsFrom pre cs →
    a.toList ++ '/' :: b.toList = pre.reverse ++ cs
  | _, [], _, _, h => by simp [OneStageZk.splitsFrom] at h
  | pre, c :: cs, a, b, h => by
    by_cases hc : c = '/'
    · rw [OneStageZk.splitsFrom, if_pos hc] at h
      rcases List.mem_cons.1 h with h | h
      · simp only [Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        simp [String.toList_ofList, hc]
      · have := splitsFrom_mem (c :: pre) cs h
        simpa using this
    · rw [OneStageZk.splitsFrom, if_neg hc] at h
      have := splitsFrom_mem (c :: pre) cs h
      simpa using this

/-- A way `splitsOf` reads the port `port` as `<a>/<b>` is `port` itself. -/
theorem port_of_splits {port a b : String} (h : (a, b) ∈ OneStageZk.splitsOf port) : port = a ++ "/" ++ b := by
  have e := splitsFrom_mem [] port.toList h
  have hs : ("/" : String).toList = ['/'] := by first | rfl | decide | simp
  rw [← String.toList_inj, String.toList_append, String.toList_append, hs]
  simpa using e.symm

/-- `rootStep` returns its map unchanged, or pushes on a member `name`'s roots the root `r` under `bare`, where
`splitsOf` reads `r`'s port as `<name>/<bare>`. -/
theorem rootStep_cases {ts : Std.HashSet String} {acc acc' : Std.HashMap String (Array Json)} {r : Json}
    (h : OneStageZk.rootStep ts acc r = .ok acc') :
    acc' = acc ∨ ∃ port name bare, getD r "port" = .str port ∧ (name, bare) ∈ OneStageZk.splitsOf port ∧
      acc' = acc.modify name (·.push (Json.mkObj [("domain", getD r "domain"), ("leaves", getD r "leaves"),
        ("port", .str bare), ("root", getD r "root")])) := by
  unfold OneStageZk.rootStep at h
  split at h
  · rename_i port hp
    dsimp only at h
    split at h
    · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
    · split at h
      · cases h; exact Or.inl rfl
      · split at h
        · rename_i name bare hf
          cases h
          have hm : (name, bare) ∈ (OneStageZk.splitsOf port).filter fun x => acc.contains x.1 := by
            rw [hf]; simp
          exact Or.inr ⟨port, name, bare, hp, (List.mem_filter.1 hm).1, rfl⟩
        · split at h
          · cases h; exact Or.inl rfl
          · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
        · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
  · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])

theorem mem_modify_push {acc : Std.HashMap String (Array Json)} {name n : String} {e x : Json}
    (h : x ∈ (acc.modify name (·.push e)).getD n #[]) : x ∈ acc.getD n #[] ∨ (name = n ∧ x = e) := by
  rw [Std.HashMap.getD_modify] at h
  split at h
  · rename_i hn
    have hn' : name = n := beq_iff_eq.1 hn
    subst hn'
    rw [Std.HashMap.getD_eq_getD_getElem?]
    cases ha : acc[name]? with
    | none => simp [ha] at h
    | some a =>
      simp only [ha, Option.map_some, Option.getD_some, Array.mem_push] at h ⊢
      rcases h with h | h
      · exact Or.inl h
      · exact Or.inr ⟨by simp, h⟩
  · exact Or.inl h

/-- **Each root a map holds for a member `n` is the record's root at `<n>/<b>`**, `b` its port. -/
def OwnRoots (rec : Json) (acc : Std.HashMap String (Array Json)) : Prop :=
  ∀ n, ∀ x ∈ acc.getD n #[], ∃ b, getD x "port" = .str b ∧ Carries rec (n ++ "/" ++ b) x

theorem rootStep_inv {rec : Json} {ts : Std.HashSet String} {acc acc' : Std.HashMap String (Array Json)} {r : Json}
    (hr : r ∈ rootsOf rec) (h : OneStageZk.rootStep ts acc r = .ok acc') (hacc : OwnRoots rec acc) :
    OwnRoots rec acc' := by
  rcases rootStep_cases h with rfl | ⟨port, name, bare, hp, hs, rfl⟩
  · exact hacc
  · intro n x hx
    rcases mem_modify_push hx with hx | ⟨rfl, rfl⟩
    · exact hacc n x hx
    · obtain ⟨h1, h2, h3, h4⟩ := getD_e4 (getD r "domain") (getD r "leaves") (getD r "root") bare
      refine ⟨bare, h1, r, hr, ?_, h2, h3, h4⟩
      rw [hp, ← port_of_splits hs]

/-- **Step 3: `rootsOf`'s roots are the record's**: each root it gives a member `n` is the record's root at
`<n>/<b>`, `b` its port. -/
theorem rootsOf_own {ms : Array OneStageZk.Member} {rec : Json} {acc : Std.HashMap String (Array Json)}
    (h : OneStageZk.rootsOf ms rec = .ok acc) : OwnRoots rec acc := by
  unfold OneStageZk.rootsOf at h
  split at h
  · rename_i rs hrs
    have hr : rootsOf rec = rs := by unfold rootsOf; rw [hrs]
    dsimp only at h
    split at h
    · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
    split at h
    · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
    obtain ⟨acc1, h1, h⟩ := ebind_ok h
    obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
    cases h
    rw [← Array.foldlM_toList] at h1
    refine foldlM_inv (OwnRoots rec) _ (· ∈ rootsOf rec) (fun _ _ _ ha hf hb => rootStep_inv ha hf hb) _ _ _
      (fun a ha => hr ▸ Array.mem_toList_iff.1 ha) ?_ h1
    have key := foldl_inv (fun m : Std.HashMap String (Array Json) => ∀ n, m.getD n #[] = #[])
      (fun acc (m : OneStageZk.Member) => acc.insert m.name #[]) (fun _ => True)
      (by intro b a _ hb n; rw [Std.HashMap.getD_insert]; split; rfl; exact hb n) ms.toList {} (fun _ _ => trivial)
      (fun n => Std.HashMap.getD_empty)
    rw [Array.foldl_toList] at key
    intro n x hx
    rw [key n] at hx
    simp at hx
  · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])

/-! ## Step 5: a member's ports at C's run input nodes' trees (`rootsByPort`, `runInputRootsAt`) -/

/-- **`rootsByPort`'s root at `t` is a root of the record naming `t`.** -/
theorem rootsByPort_get {rec : Json} {t : String} {r : Json} (h : (OneStageZk.rootsByPort rec).get? t = some r) :
    r ∈ rootsOf rec ∧ getD r "port" = .str t := by
  unfold OneStageZk.rootsByPort at h
  revert h r t
  split
  · rename_i rs hrs
    have hr : rootsOf rec = rs := by unfold rootsOf; rw [hrs]
    rw [hr, ← Array.foldl_toList]
    refine foldl_inv (fun m : Std.HashMap String Json => ∀ t r, m.get? t = some r → r ∈ rs ∧ getD r "port" = .str t)
      _ (· ∈ rs) ?_ rs.toList {} (fun a ha => Array.mem_toList_iff.1 ha) ?_
    · intro m a ha hm t r h
      try dsimp only at h
      split at h
      · rename_i p hp
        change (m.insert p a)[t]? = some r at h
        rw [Std.HashMap.getElem?_insert] at h
        split at h
        · rename_i hpt
          cases h
          exact ⟨ha, by rw [hp, beq_iff_eq.1 hpt]⟩
        · exact hm t r h
      · exact hm t r h
    · intro t r h
      change (∅ : Std.HashMap String Json)[t]? = some r at h
      simp at h
  · intro t r h
    change (∅ : Std.HashMap String Json)[t]? = some r at h
    simp at h

theorem getD_e5 (d l r : Json) (b t : String) :
    getD (Json.mkObj [("domain", d), ("leaves", l), ("port", .str b), ("root", r), ("tree", .str t)]) "tree" = .str t ∧
    getD (Json.mkObj [("domain", d), ("leaves", l), ("port", .str b), ("root", r), ("tree", .str t)]) "root" = r ∧
    getD (Json.mkObj [("domain", d), ("leaves", l), ("port", .str b), ("root", r), ("tree", .str t)]) "domain" = d ∧
    getD (Json.mkObj [("domain", d), ("leaves", l), ("port", .str b), ("root", r), ("tree", .str t)]) "leaves" = l := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> with_unfolding_all rfl

/-- **`x` is a root at a tree of `byPort`**: under the tree `t`, with the root, domain and leaves of `byPort`'s root
at `t`. -/
def RunTree (byPort : Std.HashMap String Json) (x : Json) : Prop :=
  ∃ t r, byPort.get? t = some r ∧ getD x "tree" = .str t ∧ getD x "root" = getD r "root" ∧
    getD x "domain" = getD r "domain" ∧ getD x "leaves" = getD r "leaves"

/-- **Each root `runInputRootsAt` returns is one of `own`, or a root at a tree of `byPort`.** -/
theorem runInputRootsAt_mem {P : Flock.Program.Prog} {rs : Array Flock.TemplateQuery.TRange} {prog : String}
    {R : Option String} {tied : Array (Nat × Nat)} {m : OneStageZk.Member} {own out : Array Json}
    {byPort : Std.HashMap String Json}
    (h : OneStageZk.runInputRootsAt P rs prog R tied m own byPort = .ok out) :
    ∀ x ∈ out, x ∈ own ∨ RunTree byPort x := by
  unfold OneStageZk.runInputRootsAt at h
  obtain ⟨ins, -, h⟩ := ebind_ok h
  rw [← Array.foldlM_toList] at h
  refine foldlM_inv (fun got => ∀ x ∈ got, x ∈ own ∨ RunTree byPort x) _ (fun _ => True) ?_ _ own out
    (fun _ _ => trivial) (fun x hx => Or.inl hx) h
  intro got a got' _ hf hgot
  obtain ⟨p, w, len⟩ := a
  dsimp only at hf
  split at hf
  · cases hf; exact hgot
  · split at hf
    · rename_i r hr
      split at hf
      · obtain ⟨⟨⟩, ht, -⟩ := ebind_ok hf
        cases ht
      obtain ⟨⟨⟩, -, hf⟩ := ebind_ok hf
      cases hf
      intro x hx
      rcases Array.mem_push.1 hx with hx | rfl
      · exact hgot x hx
      · obtain ⟨h1, h2, h3, h4⟩ := getD_e5 (getD r "domain") (getD r "leaves") (getD r "root") p
          (Flock.Registered.freshValue w)
        exact Or.inr ⟨_, r, hr, h1, h2, h3, h4⟩
    · cases hf

/-- **Step 5: `runInputRoots`'s roots**: each root it gives a member `n` is one `roots` gave `n`, or a root at a tree
of the record's roots by port (`rootsByPort`). -/
theorem runInputRoots_mem {v : Docket.Verifier} {P : Flock.Program.Prog} {rs : Array Flock.TemplateQuery.TRange}
    {ms : Array OneStageZk.Member} {rec : Json} {roots out : Std.HashMap String (Array Json)}
    (h : OneStageZk.runInputRoots v P rs ms rec roots = .ok out) :
    ∀ n, ∀ x ∈ out.getD n #[], x ∈ roots.getD n #[] ∨ RunTree (OneStageZk.rootsByPort rec) x := by
  unfold OneStageZk.runInputRoots at h
  try dsimp only at h
  obtain ⟨acc, hacc, h⟩ := ebind_ok h
  cases h
  rw [← Array.forIn_toList] at hacc
  refine forIn_yield_inv (fun s : Std.HashMap String (Array Json) × Array (Nat × Json) =>
    ∀ n, ∀ x ∈ s.1.getD n #[], x ∈ roots.getD n #[] ∨ RunTree (OneStageZk.rootsByPort rec) x) ?_ _ (roots, #[]) _
    (fun n x hx => Or.inl hx) hacc
  intro m s r hacc hf
  try dsimp only at hf
  obtain ⟨_, -, hf⟩ := ebind_ok hf
  obtain ⟨o, ho, hf⟩ := ebind_ok hf
  cases hf
  refine ⟨_, rfl, ?_⟩
  intro n x hx
  rw [Std.HashMap.getD_insert] at hx
  split at hx
  · rename_i hn
    have hn' : m.name = n := beq_iff_eq.1 hn
    subst hn'
    rcases runInputRootsAt_mem ho x hx with hx | hx
    · exact hacc _ x hx
    · exact Or.inr hx
  · exact hacc n x hx

theorem foldlM_collect {ε α β : Type} (f : Array β → α → Except ε (Array β)) (G : α → Prop) (Q : α → β → Prop)
    (hf : ∀ got a got', f got a = .ok got' → (∀ y ∈ got, y ∈ got') ∧ (G a → ∃ r ∈ got', Q a r)) :
    ∀ (l : List α) (got out : Array β), l.foldlM f got = .ok out →
      (∀ y ∈ got, y ∈ out) ∧ ∀ a ∈ l, G a → ∃ r ∈ out, Q a r
  | [], got, out, h => by cases h; exact ⟨fun y hy => hy, fun a ha => by cases ha⟩
  | a :: l, got, out, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨g1, h1, h⟩ := ebind_ok h
    obtain ⟨s1, q1⟩ := hf got a g1 h1
    obtain ⟨s2, q2⟩ := foldlM_collect f G Q hf l g1 out h
    refine ⟨fun y hy => s2 y (s1 y hy), fun a' ha' hG => ?_⟩
    rcases List.mem_cons.1 ha' with rfl | ha'
    · obtain ⟨r, hr, hq⟩ := q1 hG
      exact ⟨r, s2 r hr, hq⟩
    · exact q2 a' ha' hG

theorem forIn_collect {ε α σ : Type} (f : α → σ → Except ε (ForInStep σ)) (Le : σ → σ → Prop) (G : α → σ → Prop)
    (refl : ∀ s, Le s s) (trans : ∀ s t u, Le s t → Le t u → Le s u) (mono : ∀ a s t, G a s → Le s t → G a t)
    (hf : ∀ a s r, f a s = .ok r → ∃ t, r = .yield t ∧ Le s t ∧ G a t) :
    ∀ (l : List α) (s out : σ), forIn l s f = .ok out → Le s out ∧ ∀ a ∈ l, G a out
  | [], s, out, h => by
    rw [List.forIn_nil] at h
    cases h
    exact ⟨refl s, fun a ha => by cases ha⟩
  | a :: l, s, out, h => by
    rw [List.forIn_cons] at h
    cases hfa : f a s with
    | error e => rw [hfa] at h; cases h
    | ok r =>
      rw [hfa] at h
      obtain ⟨t, rfl, hle, hg⟩ := hf a s r hfa
      obtain ⟨h1, h2⟩ := forIn_collect f Le G refl trans mono hf l t out h
      refine ⟨trans _ _ _ hle h1, fun a' ha' => ?_⟩
      rcases List.mem_cons.1 ha' with rfl | ha'
      · exact mono _ _ _ hg h1
      · exact h2 a' ha'

/-- **`x` is a root under port `port` at the tree of run input node `w`**, with the root, domain and leaves of
`byPort`'s root of that tree. -/
def RunRootAt (byPort : Std.HashMap String Json) (port : String) (w : Nat) (x : Json) : Prop :=
  getD x "port" = .str port ∧ ∃ r0, byPort.get? (Flock.Registered.freshValue w) = some r0 ∧
    getD x "tree" = .str (Flock.Registered.freshValue w) ∧ getD x "root" = getD r0 "root" ∧
    getD x "domain" = getD r0 "domain" ∧ getD x "leaves" = getD r0 "leaves"

/-- **`runInputRootsAt` keeps `own` and adds a root at each untied run input the member reads.** -/
theorem runInputRootsAt_has {P : Flock.Program.Prog} {rs : Array Flock.TemplateQuery.TRange} {prog : String}
    {R : Option String} {tied : Array (Nat × Nat)} {m : OneStageZk.Member} {own out : Array Json}
    {byPort : Std.HashMap String Json}
    (h : OneStageZk.runInputRootsAt P rs prog R tied m own byPort = .ok out) :
    (∀ y ∈ own, y ∈ out) ∧ ∃ ins, OneStageZk.memberInputs P rs m = .ok ins ∧
      ∀ (port : String) (w len : Nat), (port, w, len) ∈ ins → Flock.Registered.runNode P w = true →
        (tied.any fun (first, count) => Partition.within P first count w) = false →
        ∃ r ∈ out, RunRootAt byPort port w r := by
  unfold OneStageZk.runInputRootsAt at h
  obtain ⟨ins, hins, h⟩ := ebind_ok h
  rw [← Array.foldlM_toList] at h
  have key := foldlM_collect _
    (fun (x : String × Nat × Nat) => Flock.Registered.runNode P x.2.1 = true ∧
      (tied.any fun (first, count) => Partition.within P first count x.2.1) = false)
    (fun x r => RunRootAt byPort x.1 x.2.1 r) ?_ ins.toList own out h
  · exact ⟨key.1, ins, hins, fun port w len hx hrun ht =>
      key.2 (port, w, len) (Array.mem_toList_iff.2 hx) ⟨hrun, ht⟩⟩
  · intro got a got' hf
    obtain ⟨p, w, len⟩ := a
    dsimp only at hf
    split at hf
    · rename_i hc
      cases hf
      refine ⟨fun y hy => hy, fun ⟨hr, ht⟩ => ?_⟩
      dsimp only at hr ht
      rw [hr, ht] at hc
      exact absurd hc (by decide)
    · split at hf
      · rename_i r hr
        split at hf
        · obtain ⟨⟨⟩, ht, -⟩ := ebind_ok hf
          cases ht
        obtain ⟨⟨⟩, -, hf⟩ := ebind_ok hf
        cases hf
        refine ⟨fun y hy => Array.mem_push.2 (Or.inl hy), fun _ => ⟨_, Array.mem_push.2 (Or.inr rfl), ?_⟩⟩
        obtain ⟨h1, h2, h3, h4⟩ := getD_e5 (getD r "domain") (getD r "leaves") (getD r "root") p
          (Flock.Registered.freshValue w)
        exact ⟨by with_unfolding_all rfl, r, hr, h1, h2, h3, h4⟩
      · cases hf

/-- **`runInputRoots` gives each member a root at each untied run input it reads**, at the record's roots by port. -/
theorem runInputRoots_has {v : Docket.Verifier} {P : Flock.Program.Prog} {rs : Array Flock.TemplateQuery.TRange}
    {ms : Array OneStageZk.Member} {rec : Json} {roots out : Std.HashMap String (Array Json)}
    (h : OneStageZk.runInputRoots v P rs ms rec roots = .ok out) :
    ∀ m ∈ ms, ∃ ins, OneStageZk.memberInputs P rs m = .ok ins ∧
      ∀ (port : String) (w len : Nat), (port, w, len) ∈ ins → Flock.Registered.runNode P w = true →
        ((Docket.tiedGates v).any fun (first, count) => Partition.within P first count w) = false →
        ∃ r ∈ out.getD m.name #[], RunRootAt (OneStageZk.rootsByPort rec) port w r := by
  unfold OneStageZk.runInputRoots at h
  try dsimp only at h
  obtain ⟨acc, hacc, h⟩ := ebind_ok h
  cases h
  rw [← Array.forIn_toList] at hacc
  have key := forIn_collect _ (fun a b : Std.HashMap String (Array Json) × Array (Nat × Json) =>
      ∀ n, ∀ y ∈ a.1.getD n #[], y ∈ b.1.getD n #[])
    (fun m (s : Std.HashMap String (Array Json) × Array (Nat × Json)) => ∃ ins, OneStageZk.memberInputs P rs m = .ok ins ∧
      ∀ (port : String) (w len : Nat), (port, w, len) ∈ ins → Flock.Registered.runNode P w = true →
        ((Docket.tiedGates v).any fun (first, count) => Partition.within P first count w) = false →
        ∃ r ∈ s.1.getD m.name #[], RunRootAt (OneStageZk.rootsByPort rec) port w r)
    (fun _ _ _ hy => hy) (fun _ _ _ h1 h2 n y hy => h2 n y (h1 n y hy))
    (fun _ _ _ ⟨ins, hins, hg⟩ hle => ⟨ins, hins, fun port w len hx hrun ht => by
      obtain ⟨r, hr, hq⟩ := hg port w len hx hrun ht
      exact ⟨r, hle _ r hr, hq⟩⟩) ?_ ms.toList (roots, #[]) _ hacc
  · exact fun m hm => key.2 m (Array.mem_toList_iff.2 hm)
  · intro m s r hf
    try dsimp only at hf
    obtain ⟨_, -, hf⟩ := ebind_ok hf
    obtain ⟨o, ho, hf⟩ := ebind_ok hf
    cases hf
    obtain ⟨hsub, hhas⟩ := runInputRootsAt_has ho
    refine ⟨_, rfl, ?_, ?_⟩
    · intro n y hy
      rw [Std.HashMap.getD_insert]
      split
      · rename_i hn
        have hn' : m.name = n := beq_iff_eq.1 hn
        subst hn'
        exact hsub y hy
      · exact hy
    · obtain ⟨ins, hins, hh⟩ := hhas
      refine ⟨ins, hins, fun port w len hx hrun ht => ?_⟩
      rw [Std.HashMap.getD_insert]
      simp only [beq_self_eq_true, ↓reduceIte]
      exact hh port w len hx hrun ht

/-- **`fieldRoots` keeps every root each member had.** -/
theorem fieldRoots_mono {W : Spot.Wires.Wiring} {bases : Array (String × Nat × Nat)} {ms : Array OneStageZk.Member}
    {ft : Std.HashMap String (Array (String × Json))} {roots out : Std.HashMap String (Array Json)}
    (h : OneStageZk.fieldRoots W bases ms ft roots = .ok out) :
    ∀ n, ∀ y ∈ roots.getD n #[], y ∈ out.getD n #[] := by
  unfold OneStageZk.fieldRoots at h
  try dsimp only at h
  rw [← Array.foldlM_toList] at h
  refine foldlM_inv (fun acc => ∀ n, ∀ y ∈ roots.getD n #[], y ∈ acc.getD n #[]) _ (fun _ => True) ?_ _ roots out
    (fun _ _ => trivial) (fun n y hy => hy) h
  intro acc m acc' _ hf hacc
  try dsimp only at hf
  split at hf
  · exact absurd hf (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
  split at hf
  · exact absurd hf (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
  cases hf
  intro n y hy
  rw [Std.HashMap.getD_insert]
  split
  · rename_i hn
    have hn' : m.name = n := beq_iff_eq.1 hn
    subst hn'
    refine foldl_inv (fun got => y ∈ got) _ (· ∈ W.wires) ?_ W.wires _ (fun _ h => h) hy
    intro got w hw hgot
    try dsimp only
    split
    · exact hgot
    · split
      · exact Array.mem_push.2 (Or.inl hgot)
      · exact hgot
  · exact hacc n y hy

/-- A root at a tree of the record's roots by port carries the record's root of that tree. -/
theorem carries_of_runTree {rec x : Json} (h : RunTree (OneStageZk.rootsByPort rec) x) :
    ∃ t, getD x "tree" = .str t ∧ Carries rec t x := by
  obtain ⟨t, r, hg, ht, hr, hd, hl⟩ := h
  obtain ⟨hm, hp⟩ := rootsByPort_get hg
  exact ⟨t, ht, r, hm, hp, hr, hd, hl⟩

/-! ## Step 4: the window's trees of its templates' fields (`fieldTreesOf`, `fieldRootIn`, `ownFieldsOf`, `fieldRoots`) -/

/-- **Each tree `fieldTreesOf` files under template `t` and field `f` is the record's root at `<t>/<f>`**, under that
tree. -/
def FieldTrees (rec : Json) (ft : Std.HashMap String (Array (String × Json))) : Prop :=
  ∀ t, ∀ fe ∈ ft.getD t #[], getD fe.2 "tree" = .str (t ++ "/" ++ fe.1) ∧ Carries rec (t ++ "/" ++ fe.1) fe.2

theorem mem_splits_of_templateOf {port t f : String} (h : OneStageZk.templateOf port = some (t, f)) :
    (t, f) ∈ OneStageZk.splitsOf port := by
  unfold OneStageZk.templateOf at h
  exact List.mem_of_getLast? h

theorem fieldTreesOf_trees (ts : Std.HashSet String) (rec : Json) : FieldTrees rec (OneStageZk.fieldTreesOf ts rec) := by
  unfold OneStageZk.fieldTreesOf
  split
  · rename_i rs hrs
    have hr : rootsOf rec = rs := by unfold rootsOf; rw [hrs]
    rw [← Array.foldl_toList]
    refine foldl_inv (FieldTrees rec) _ (· ∈ rs) ?_ rs.toList {} (fun a ha => Array.mem_toList_iff.1 ha) ?_
    · intro acc r hr' hacc t fe hfe
      try dsimp only at hfe
      split at hfe
      · rename_i port hp
        split at hfe
        · rename_i t0 f0 htf
          split at hfe
          · rw [Std.HashMap.getD_insert] at hfe
            split at hfe
            · rename_i ht
              have ht' : t0 = t := beq_iff_eq.1 ht
              subst ht'
              rcases Array.mem_push.1 hfe with hfe | rfl
              · exact hacc t0 fe hfe
              · have hport := port_of_splits (mem_splits_of_templateOf htf)
                obtain ⟨h1, h2, h3, h4⟩ := getD_e5 (getD r "domain") (getD r "leaves") (getD r "root") f0 port
                refine ⟨by rw [h1, hport], r, hr ▸ hr', ?_, h2, h3, h4⟩
                rw [hp, hport]
            · exact hacc t fe hfe
          · exact hacc t fe hfe
        · exact hacc t fe hfe
      · exact hacc t fe hfe
    · intro t fe hfe
      rw [Std.HashMap.getD_empty] at hfe
      simp at hfe
  · intro t fe hfe
    rw [Std.HashMap.getD_empty] at hfe
    simp at hfe

/-- **`fieldRootIn`'s root is the record's root at `<template>/<field>`**, under the reader's port name. -/
theorem fieldRootIn_carries {rec : Json} {ft : Std.HashMap String (Array (String × Json))} (hft : FieldTrees rec ft)
    {template field port : String} {r : Json} (h : OneStageZk.fieldRootIn ft template field port = some r) :
    getD r "port" = .str port ∧ Carries rec (template ++ "/" ++ field) r := by
  unfold OneStageZk.fieldRootIn at h
  obtain ⟨⟨f, e⟩, hfe, rfl⟩ := Option.map_eq_some_iff.1 h
  have hb := Array.find?_some hfe
  have hf : f = field := beq_iff_eq.1 hb
  subst hf
  try dsimp only
  obtain ⟨-, r0, hm, hp, hr, hd, hl⟩ := hft template (f, e) (Array.mem_of_find?_eq_some hfe)
  obtain ⟨h1, h2, h3, h4⟩ := getD_e4 (getD e "domain") (getD e "leaves") (getD e "root") port
  exact ⟨h1, r0, hm, hp, h2.trans hr, h3.trans hd, h4.trans hl⟩

/-- **`fieldRootIn`'s root names its tree `<template>/<field>`.** -/
theorem fieldRootIn_tree {ft : Std.HashMap String (Array (String × Json))} {template field port : String} {r : Json}
    (h : OneStageZk.fieldRootIn ft template field port = some r) : getD r "tree" = .str (template ++ "/" ++ field) := by
  unfold OneStageZk.fieldRootIn at h
  obtain ⟨⟨f, e⟩, hfe, rfl⟩ := Option.map_eq_some_iff.1 h
  have hb := Array.find?_some hfe
  have hf : f = field := beq_iff_eq.1 hb
  subst hf
  with_unfolding_all rfl

/-- **Each root `ownFieldsOf` returns is one of `roots`, or a tree `ft` files.** -/
theorem ownFieldsOf_mem {ft : Std.HashMap String (Array (String × Json))} {name : String} {roots : Array Json} :
    ∀ x ∈ OneStageZk.ownFieldsOf ft name roots, x ∈ roots ∨ ∃ t, ∃ fe ∈ ft.getD t #[], x = fe.2 := by
  unfold OneStageZk.ownFieldsOf
  split
  · exact fun x hx => Or.inl hx
  · rename_i t _ _
    rw [← Array.foldl_toList]
    refine foldl_inv (fun got => ∀ x ∈ got, x ∈ roots ∨ ∃ t, ∃ fe ∈ ft.getD t #[], x = fe.2) _
      (· ∈ ft.getD t #[]) ?_ _ roots (fun a ha => Array.mem_toList_iff.1 ha) (fun x hx => Or.inl hx)
    intro got a ha hgot
    obtain ⟨f, e⟩ := a
    try dsimp only
    split
    · exact hgot
    · intro x hx
      rcases Array.mem_push.1 hx with hx | rfl
      · exact hgot x hx
      · exact Or.inr ⟨t, (f, x), ha, rfl⟩

/-- **The three ways a root `x` a member named `n` reads is the record's** (`MembersRootsRecord`), each with the record's
root it carries. -/
def Good (p : OneStageZk.Params) (ms : Array OneStageZk.Member) (rec : Json) (n : String) (x : Json) : Prop :=
  (∃ b, getD x "port" = .str b ∧ Carries rec (n ++ "/" ++ b) x) ∨ (∃ q, getD x "tree" = .str q ∧ Carries rec q x) ∨
  ∃ P rs W, Program.parse p.own.program = .ok P ∧ Partition.evaluate P (getD p.own.partition "query") = .ok rs ∧
    OneStageZk.derivedWiring P.prog rs = .ok W ∧ ∃ w ∈ W.wires, (∃ m' ∈ ms, m'.name = n ∧ WireInto rs m' w) ∧
      getD x "port" = .str w.port ∧ Carries rec (w.writer ++ "/" ++ w.out) x

/-- Every root a map holds for every name is `Good`. -/
def AllGood (p : OneStageZk.Params) (ms : Array OneStageZk.Member) (rec : Json)
    (acc : Std.HashMap String (Array Json)) : Prop :=
  ∀ n, ∀ x ∈ acc.getD n #[], Good p ms rec n x

theorem ownFields_good {p : OneStageZk.Params} {ms : Array OneStageZk.Member} {rec : Json}
    {ft : Std.HashMap String (Array (String × Json))} (hft : FieldTrees rec ft) {roots : Std.HashMap String (Array Json)}
    (h : AllGood p ms rec roots) :
    AllGood p ms rec (ms.foldl (fun acc m => acc.insert m.name (OneStageZk.ownFieldsOf ft m.name (acc.getD m.name #[])))
      roots) := by
  rw [← Array.foldl_toList]
  refine foldl_inv (AllGood p ms rec) _ (fun _ => True) ?_ _ roots (fun _ _ => trivial) h
  intro acc m _ hacc n x hx
  rw [Std.HashMap.getD_insert] at hx
  split at hx
  · rename_i hn
    have hn' : m.name = n := beq_iff_eq.1 hn
    subst hn'
    rcases ownFieldsOf_mem x hx with hx | ⟨t, fe, hfe, rfl⟩
    · exact hacc _ x hx
    · obtain ⟨h1, h2⟩ := hft t fe hfe
      exact Or.inr (Or.inl ⟨_, h1, h2⟩)
  · exact hacc n x hx

theorem runInputRoots_good {p : OneStageZk.Params} {ms : Array OneStageZk.Member} {rec : Json} {v : Docket.Verifier}
    {P : Flock.Program.Prog} {rs : Array Flock.TemplateQuery.TRange} {roots out : Std.HashMap String (Array Json)}
    (hg : AllGood p ms rec roots) (h : OneStageZk.runInputRoots v P rs ms rec roots = .ok out) :
    AllGood p ms rec out := by
  intro n x hx
  rcases runInputRoots_mem h n x hx with hx | hx
  · exact hg n x hx
  · obtain ⟨q, hq, hc⟩ := carries_of_runTree hx
    exact Or.inr (Or.inl ⟨q, hq, hc⟩)

theorem of_or_not {a b : Bool} (h : ¬ (a || !b) = true) : b = true := by
  cases a <;> cases b <;> simp_all

/-- **Step 4: `fieldRoots`' roots**: each root it gives a member is one `roots` gave it, or the record's root at the
writer's output field `<w.writer>/<w.out>` of a wire `w` of `W` into a member of that name, named as its tree. -/
theorem fieldRoots_good {p : OneStageZk.Params} {ms : Array OneStageZk.Member} {rec : Json} {P : Spot.Program}
    {rs : Array Flock.TemplateQuery.TRange} {W : Spot.Wires.Wiring}
    (hP : Program.parse p.own.program = .ok P) (hrs : Partition.evaluate P (getD p.own.partition "query") = .ok rs)
    (hW : OneStageZk.derivedWiring P.prog rs = .ok W) {ft : Std.HashMap String (Array (String × Json))}
    (hft : FieldTrees rec ft) {roots out : Std.HashMap String (Array Json)} (hg : AllGood p ms rec roots)
    (h : OneStageZk.fieldRoots W (rs.map fun r => (r.template, r.first, r.count)) ms ft roots = .ok out) :
    AllGood p ms rec out := by
  unfold OneStageZk.fieldRoots at h
  try dsimp only at h
  rw [← Array.foldlM_toList] at h
  refine foldlM_inv (AllGood p ms rec) _ (· ∈ ms) ?_ _ roots out (fun a ha => Array.mem_toList_iff.1 ha) hg h
  intro acc m acc' hm hf hacc
  try dsimp only at hf
  split at hf
  · exact absurd hf (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
  split at hf
  · exact absurd hf (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
  cases hf
  intro n x hx
  rw [Std.HashMap.getD_insert] at hx
  split at hx
  · rename_i hn
    have hn' : m.name = n := beq_iff_eq.1 hn
    subst hn'
    revert x
    refine foldl_inv (fun got => ∀ x ∈ got, Good p ms rec m.name x) _ (· ∈ W.wires) ?_ W.wires _ (fun _ h => h)
      (hg m.name)
    intro got w hw hgot
    try dsimp only
    split
    · exact hgot
    · split
      · rename_i r hr
        intro x hx
        rcases Array.mem_push.1 hx with hx | rfl
        · exact hgot x hx
        · exact Or.inr (Or.inl ⟨_, fieldRootIn_tree hr, (fieldRootIn_carries hft hr).2⟩)
      · exact hgot
  · exact hacc n x hx

/-! ## Step 6: the composition (`membersRootsWired`, `membersRootsOf`) and the main theorem -/

theorem membersRootsWired_good {p : OneStageZk.Params} {ms : Array OneStageZk.Member} {rec : Json}
    {roots : Std.HashMap String (Array Json)} (h : OneStageZk.membersRootsWired p ms rec = .ok roots) :
    AllGood p ms rec roots := by
  unfold OneStageZk.membersRootsWired at h
  try dsimp only at h
  obtain ⟨r0, h0, h⟩ := ebind_ok h
  obtain ⟨P, hP, h⟩ := ebind_ok h
  obtain ⟨rs, hrs, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  obtain ⟨r2, h2, h⟩ := ebind_ok h
  have hft := fieldTreesOf_trees (OneStageZk.templatesOf (ms.toList.map (·.name))) rec
  have g0 : AllGood p ms rec r0 := fun n x hx => Or.inl (rootsOf_own h0 n x hx)
  have g2 := runInputRoots_good (ownFields_good hft g0) h2
  split at h
  · cases h; exact g2
  · obtain ⟨W, hW, h⟩ := ebind_ok h
    exact fieldRoots_good hP hrs hW hft g2 h

/-- **Where the per-member form's structure holds, no two members share a name** (`OneStageZk.namesOnce`). -/
theorem membersStructure_names {p : OneStageZk.Params} {ms : Array OneStageZk.Member} {c : OneStage.Commitment}
    {e : OneStageZk.Evidence} {sts : Array OneStageZk.StatementFiles} {bs : Core.Blobs}
    (h : OneStageZk.membersStructure p ms c e sts bs = .ok ()) : (ms.toList.map (·.name)).Nodup := by
  unfold OneStageZk.membersStructure at h
  obtain ⟨⟨⟩, h0, -⟩ := ebind_ok h
  exact OneStageZk.namesOnce_ok h0

theorem slice_roots (m : OneStageZk.Member) (rec : Json) (roots : Std.HashMap String (Array Json)) :
    getD (m.slice rec roots) "roots" = .arr (roots.getD m.name #[]) := by
  unfold OneStageZk.Member.slice
  with_unfolding_all rfl

/-! ## Tied roots (`tiedRootsAt`) -/

/-- `t` keeps every node's first entry `s` has. -/
def SeenLe (s t : Array (Nat × Json)) : Prop :=
  ∀ w x, s.find? (fun e => e.1 == w) = some x → t.find? (fun e => e.1 == w) = some x

theorem seenLe_push {s : Array (Nat × Json)} {w : Nat} {r : Json} : SeenLe s (s.push (w, r)) := by
  intro w' x hx
  rw [Array.find?_push, hx]
  rfl

/-- What a member's step leaves of node `w`: `own` holds a root at each port the member reads `w` at, and `w`'s entry
in `t` is that root or `==` to it. -/
def TiedHeld (P : Flock.Program.Prog) (rs : Array Flock.TemplateQuery.TRange) (m : OneStageZk.Member)
    (own : Array Json) (w : Nat) (t : Array (Nat × Json)) : Prop :=
  ∀ ins, OneStageZk.memberInputs P rs m = .ok ins → ∀ (port : String) (len : Nat), (port, w, len) ∈ ins →
    ∃ r : Json, (own.find? (getD · "port" == .str port)).map (getD · "root") = some r ∧
      ∃ x, t.find? (fun e => e.1 == w) = some x ∧ (x.2 = r ∨ (x.2 == r) = true)

theorem tiedHeld_mono {P : Flock.Program.Prog} {rs : Array Flock.TemplateQuery.TRange} {m : OneStageZk.Member}
    {own : Array Json} {w : Nat} {s t : Array (Nat × Json)} (h : TiedHeld P rs m own w s) (hle : SeenLe s t) :
    TiedHeld P rs m own w t := by
  intro ins hins port len hmem
  obtain ⟨r, hr, x, hx, hxr⟩ := h ins hins port len hmem
  exact ⟨r, hr, x, hle w x hx, hxr⟩

theorem foldlM_collectLe {ε α σ : Type} (f : σ → α → Except ε σ) (Le : σ → σ → Prop) (G : α → σ → Prop)
    (refl : ∀ s, Le s s) (trans : ∀ s t u, Le s t → Le t u → Le s u) (mono : ∀ a s t, G a s → Le s t → G a t)
    (hf : ∀ s a t, f s a = .ok t → Le s t ∧ G a t) :
    ∀ (l : List α) (s out : σ), l.foldlM f s = .ok out → Le s out ∧ ∀ a ∈ l, G a out
  | [], s, out, h => by
    simp only [List.foldlM_nil] at h
    cases h
    exact ⟨refl s, fun a ha => by cases ha⟩
  | a :: l, s, out, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨t, ht, h⟩ := ebind_ok h
    obtain ⟨hle, hg⟩ := hf s a t ht
    obtain ⟨hle', hall⟩ := foldlM_collectLe f Le G refl trans mono hf l t out h
    refine ⟨trans _ _ _ hle hle', fun b hb => ?_⟩
    rcases List.mem_cons.1 hb with rfl | hb
    · exact mono _ _ _ hg hle'
    · exact hall b hb

/-- `tiedRootsAt` keeps `seen`'s entries, and `own` holds a root at each port a member reads a tied node `w` at, which
`w`'s entry is or is `==` to. -/
theorem tiedRootsAt_ok {P : Flock.Program.Prog} {rs : Array Flock.TemplateQuery.TRange} {tied : Array (Nat × Nat)}
    {m : OneStageZk.Member} {own : Array Json} {seen seen' : Array (Nat × Json)}
    (h : OneStageZk.tiedRootsAt P rs tied m own seen = .ok seen') (w : Nat) (hw : Flock.Registered.runNode P w = true)
    (ht : (tied.any fun (first, count) => Partition.within P first count w) = true) :
    SeenLe seen seen' ∧ TiedHeld P rs m own w seen' := by
  unfold OneStageZk.tiedRootsAt at h
  obtain ⟨ins0, hins0, h⟩ := ebind_ok h
  rw [← Array.foldlM_toList] at h
  have H := foldlM_collectLe (ε := String) (α := String × Nat × Nat) (σ := Array (Nat × Json)) _ SeenLe
    (fun (e : String × Nat × Nat) t => e.2.1 = w → ∃ r : Json,
      (own.find? (getD · "port" == .str e.1)).map (getD · "root") = some r ∧
        ∃ x, t.find? (fun e => e.1 == w) = some x ∧ (x.2 = r ∨ (x.2 == r) = true))
    (fun _ _ _ hx => hx) (fun _ _ _ h1 h2 w x hx => h2 w x (h1 w x hx))
    (fun _ _ _ hg hle hw' => by
      obtain ⟨r, hr, x, hx, hxr⟩ := hg hw'
      exact ⟨r, hr, x, hle w x hx, hxr⟩)
    ?_ ins0.toList seen seen' h
  · refine ⟨H.1, fun ins hins port len hmem => ?_⟩
    rw [hins0] at hins
    cases hins
    exact H.2 (port, w, len) (Array.mem_toList_iff.2 hmem) rfl
  · intro s e t hst
    obtain ⟨p, w', len⟩ := e
    dsimp only at hst ⊢
    split at hst
    · rename_i hc
      cases hst
      refine ⟨fun _ _ hx => hx, fun hww => ?_⟩
      subst hww
      simp [hw, ht] at hc
    · split at hst
      · exact absurd hst (by simp [throw, throwThe, MonadExceptOf.throw])
      · rename_i r0 hr0
        split at hst
        · rename_i x r' hx
          split at hst
          · rename_i hb
            cases hst
            refine ⟨fun _ _ hx => hx, fun hww => ?_⟩
            subst hww
            exact ⟨r0, hr0, (x, r'), hx, Or.inr hb⟩
          · exact absurd hst (by simp [throw, throwThe, MonadExceptOf.throw])
        · rename_i hx
          cases hst
          refine ⟨seenLe_push, fun hww => ?_⟩
          subst hww
          refine ⟨r0, hr0, (w', r0), ?_, Or.inl rfl⟩
          rw [Array.find?_push, hx]
          simp

theorem forIn_tied {roots : Std.HashMap String (Array Json)} {P : Flock.Program.Prog}
    {rs : Array Flock.TemplateQuery.TRange} {w : Nat}
    {f : OneStageZk.Member → Std.HashMap String (Array Json) × Array (Nat × Json) →
      Except String (ForInStep (Std.HashMap String (Array Json) × Array (Nat × Json)))}
    (hf : ∀ m s r, s.1.getD m.name #[] = roots.getD m.name #[] → f m s = .ok r →
      ∃ t, r = .yield t ∧ SeenLe s.2 t.2 ∧ TiedHeld P rs m (roots.getD m.name #[]) w t.2 ∧
        ∀ n, n ≠ m.name → t.1.getD n #[] = s.1.getD n #[]) :
    ∀ (l : List OneStageZk.Member) (s out), (l.map (·.name)).Nodup →
      (∀ m ∈ l, s.1.getD m.name #[] = roots.getD m.name #[]) → forIn l s f = .ok out →
      SeenLe s.2 out.2 ∧ ∀ m ∈ l, TiedHeld P rs m (roots.getD m.name #[]) w out.2
  | [], s, out, _, _, h => by
    rw [List.forIn_nil] at h
    cases h
    exact ⟨fun _ _ hx => hx, fun m hm => by cases hm⟩
  | a :: l, s, out, hn, hq, h => by
    rw [List.forIn_cons] at h
    cases hfa : f a s with
    | error e => rw [hfa] at h; cases h
    | ok r =>
      rw [hfa] at h
      obtain ⟨t, rfl, hle, hg, hkeep⟩ := hf a s r (hq a (List.mem_cons_self ..)) hfa
      rw [List.map_cons, List.nodup_cons] at hn
      have hq' : ∀ m ∈ l, t.1.getD m.name #[] = roots.getD m.name #[] := fun m hm => by
        rw [hkeep m.name (fun he => hn.1 (he ▸ List.mem_map_of_mem hm)), hq m (List.mem_cons_of_mem _ hm)]
      obtain ⟨hle', hall⟩ := forIn_tied hf l t out hn.2 hq' h
      refine ⟨fun w x hx => hle' w x (hle w x hx), fun m hm => ?_⟩
      rcases List.mem_cons.1 hm with rfl | hm
      · exact tiedHeld_mono hg hle'
      · exact hall m hm

end Properties.Spot.RootsRecordProof

namespace Properties.Spot.SecurityProofs

open Properties.Spot.RootsRecordProof

/-- **`MembersRootsRecord`**: where the per-member form's check accepts, every root of every member's slice is the
committed record's one root of its port: the member's own port, the tree it names, or the exact writer's output field
a wire into the member names. -/
theorem membersRootsRecord : MembersRootsRecord := by
  intro p ms c ξ e sts bs pub prof h
  have h3 := r3_of_checkMembersAt h
  obtain ⟨r, hr, -, -⟩ := OneStageZk.checkMembersAt_accept h
  unfold OneStageZk.membersRun at hr
  obtain ⟨⟨⟩, hs, hr⟩ := ebind_ok hr
  have hn := membersStructure_names hs
  obtain ⟨roots, hroots, -⟩ := ebind_ok hr
  refine ⟨roots, hroots, ?_⟩
  unfold OneStageZk.membersRootsOf at hroots
  obtain ⟨roots', hw, h2⟩ := ebind_ok hroots
  obtain ⟨⟨⟩, -, h2⟩ := ebind_ok h2
  cases h2
  have hg := membersRootsWired_good hw
  intro m hm x hx
  rw [slice_roots] at hx
  try dsimp only at hx
  rcases hg m.name x hx with ⟨b, hb, hc⟩ | ⟨q, hq, hc⟩ |
    ⟨P, rs, W, hP, hrs, hW, w, hwm, ⟨m', hm', hname, hin⟩, hp, hc⟩
  · exact ⟨_, isRecordRoot_of_carries h3 hc, Or.inl ⟨b, hb, rfl⟩⟩
  · exact ⟨q, isRecordRoot_of_carries h3 hc, Or.inr (Or.inl hq)⟩
  · have hmm : m' = m := List.inj_on_of_nodup_map hn (Array.mem_toList_iff.2 hm') (Array.mem_toList_iff.2 hm) hname
    subst hmm
    exact ⟨_, isRecordRoot_of_carries h3 hc, Or.inr (Or.inr ⟨P, rs, W, hP, hrs, hW, w, hwm, hin, hp, rfl⟩)⟩

/-- **`RunInputsAtOneTree`**: on an accepted run of the per-member variant, every untied run input a member reads is
at the record's one root of its tree. -/
theorem runInputsAtOneTree : RunInputsAtOneTree := by
  intro p ms c ξ e sts bs pub prof h
  have h3 := r3_of_checkMembersAt h
  obtain ⟨r, hr, -, -⟩ := OneStageZk.checkMembersAt_accept h
  unfold OneStageZk.membersRun at hr
  obtain ⟨⟨⟩, -, hr⟩ := ebind_ok hr
  obtain ⟨roots, hroots, -⟩ := ebind_ok hr
  refine ⟨roots, hroots, ?_⟩
  unfold OneStageZk.membersRootsOf at hroots
  obtain ⟨roots', hw, h2⟩ := ebind_ok hroots
  obtain ⟨⟨⟩, -, h2⟩ := ebind_ok h2
  cases h2
  unfold OneStageZk.membersRootsWired at hw
  try dsimp only at hw
  obtain ⟨r0, -, hw⟩ := ebind_ok hw
  obtain ⟨P, hP, hw⟩ := ebind_ok hw
  obtain ⟨rs, hrs, hw⟩ := ebind_ok hw
  obtain ⟨_, -, hw⟩ := ebind_ok hw
  obtain ⟨⟨⟩, -, hw⟩ := ebind_ok hw
  obtain ⟨r2, h2, hw⟩ := ebind_ok hw
  have hmono : ∀ n, ∀ y ∈ r2.getD n #[], y ∈ roots.getD n #[] := by
    split at hw
    · cases hw
      exact fun n y hy => hy
    · obtain ⟨W, -, hw⟩ := ebind_ok hw
      exact fieldRoots_mono hw
  refine ⟨P, rs, hP, hrs, ?_⟩
  intro m hm
  obtain ⟨ins, hins, hh⟩ := runInputRoots_has h2 m hm
  refine ⟨ins, hins, fun port w len hx hrun ht => ?_⟩
  obtain ⟨x, hx2, hpt, r1, hg, htr, hro, hd, hl⟩ := hh port w len hx hrun ht
  refine ⟨x, ?_, hpt, htr, ?_⟩
  · rw [slice_roots]
    exact hmono _ x hx2
  · obtain ⟨hm', hp'⟩ := rootsByPort_get hg
    exact isRecordRoot_of_carries h3 ⟨r1, hm', hp', hro, hd, hl⟩

/-- **`TiedInputsAtOneRoot`**: where `runInputRoots` passes, every reader of a tied run input node brings a root at
each port that reads it, and those roots are one root, up to core's `==`. -/
theorem tiedInputsAtOneRoot : TiedInputsAtOneRoot := by
  intro v P rs ms rec roots out hn h w hw ht
  unfold OneStageZk.runInputRoots at h
  try dsimp only at h
  obtain ⟨acc, hacc, -⟩ := ebind_ok h
  rw [← Array.forIn_toList] at hacc
  have H := forIn_tied (roots := roots) (P := P) (rs := rs) (w := w) ?_ ms.toList (roots, #[]) acc
    (by simpa using hn) (fun _ _ => rfl) hacc
  · refine ⟨match acc.2.find? (fun e => e.1 == w) with | some x => x.2 | none => .null, ?_⟩
    intro m hm ins hins port len hmem
    obtain ⟨r, hr, x, hx, hxr⟩ := H.2 m (Array.mem_toList_iff.2 hm) ins hins port len hmem
    refine ⟨r, hr, ?_⟩
    simp only [hx]
    exact hxr
  · intro m s r hq hf
    try dsimp only at hf
    obtain ⟨seen', hseen, hf⟩ := ebind_ok hf
    obtain ⟨rr, -, hf⟩ := ebind_ok hf
    cases hf
    rw [hq] at hseen
    obtain ⟨hle, hg⟩ := tiedRootsAt_ok hseen w hw ht
    refine ⟨_, rfl, hle, hg, fun n hne => ?_⟩
    simp [Std.HashMap.getD_insert, Ne.symm hne]

end Properties.Spot.SecurityProofs
