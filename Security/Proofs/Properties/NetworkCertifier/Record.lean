import Proofs.Properties.NetworkCertifier.Wired
import SampledProofs.WiresSound

/-!
# What the circuit check's acceptance gives the count

`check_ok` and `checked_ok` read `NetworkCheck.Circuit.check`'s steps: the record `recordOf` reads, its population
(`populationOk`), its wires' summary (`SampledProofs.Wires.check`), its reader rows' leaves (`leavesOk`) and its
declarations' rows (`declRects`), none filled twice (`apart`); the count is `countsOf` at the pins' parameters.

* `recordOf_ok`: the record's templates are named once, one a template node, in order, each its node's Definition's
  rows (`templateOf`) under its descriptor id.
* `populationOk_ok`, `batch_of_argsOk`: each template node's units are the record's, and it is a batch.
* `wires_ok`, `wireCode_none`, `cutsOf_ok`: the wires' cuts divide their ports' words, and each wire's rows lie among
  its reader port's rows and units.
* `leavesOk_ok`: each wire's writer is a template node with an output row `out`.
* `declRects_ok`: the declarations' rows are their reads' patterns' (`wireRect`), their table reads' (`rootRect`) and
  their openings' ports' (`openRects`), each read's among its port's rows (`ReadOk`), each of its template's units.
* `apart_ok`, `pairwise_wires`: no two of the wires' and declarations' rows meet (`RDisj`).
-/

namespace Properties.NetworkCertifier.Counts

open Lean (Json)
open Flock.Program (Prog)
open NetworkCheck.Period (templateNodes)
open NetworkCheck.Circuit
open Properties.NetworkPeriod (ite_ok)
open Properties.NetworkScan.Fits (bind_ok)

theorem check_ok {P : Prog} {record order pins : Json} {c : Counts}
    (h : NetworkCheck.Circuit.check P record order pins = .ok c) :
    ∃ r s held, checked P record order = .ok (r, s) ∧ heldOf pins = .ok held ∧ c = countsOf r s held := by
  unfold NetworkCheck.Circuit.check at h
  obtain ⟨⟨r, s⟩, hrs, h⟩ := bind_ok h
  obtain ⟨held, hheld, h⟩ := bind_ok h
  cases h
  exact ⟨r, s, held, hrs, hheld, rfl⟩

/-- **What `checked` accepted**: the record, its population, its wires' summary, its leaves and its declarations'
rows, none filled twice. -/
theorem checked_ok {P : Prog} {record order : Json} {r : Stated} {s : SampledProofs.Wires.Summary}
    (h : checked P record order = .ok (r, s)) :
    recordOf P record = .ok r ∧ populationOk P r = true ∧ SampledProofs.Wires.check r.wiring = .ok s ∧
      leavesOk P r = true ∧ ∃ rects, declRects r.wiring r.registered = .ok rects ∧
        apart rects (r.wiring.wires.map wireRect) = true := by
  unfold checked at h
  try simp only [] at h
  obtain ⟨r', hr', h⟩ := bind_ok h
  rcases ite_ok h with ⟨hpop, h⟩ | ⟨-, h⟩
  swap
  · exact absurd h (by intro h'; injection h')
  cases hw : SampledProofs.Wires.check r'.wiring with
  | error e =>
    rw [hw] at h
    exact absurd h (by intro h'; injection h')
  | ok s' =>
    simp only [hw, pure_bind] at h
    rcases ite_ok h with ⟨hleaves, h⟩ | ⟨-, h⟩
    swap
    · exact absurd h (by intro h'; injection h')
    obtain ⟨rects, hrects, h⟩ := bind_ok h
    rcases ite_ok h with ⟨hapart, h⟩ | ⟨-, h⟩
    swap
    · exact absurd h (by intro h'; injection h')
    obtain ⟨o, -, h⟩ := bind_ok h
    cases hord : orderWhy r'.wiring o with
    | some e =>
      rw [hord] at h
      exact absurd h (by intro h'; injection h')
    | none =>
      simp only [hord, pure_bind] at h
      injection h with h
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
      exact ⟨hr', hpop, hw, hleaves, rects, hrects, hapart⟩

/-! ## The record -/

theorem mapM_forall₂ {ε α β : Type} {f : α → Except ε β} :
    ∀ {xs : List α} {ys : List β}, xs.mapM f = .ok ys → List.Forall₂ (fun a b => f a = .ok b) xs ys
  | [], ys, h => by
    rw [List.mapM_nil] at h
    injection h with h
    subst h
    exact List.Forall₂.nil
  | x :: xs, ys, h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := bind_ok h
    obtain ⟨bs, hbs, h⟩ := bind_ok h
    injection h with h
    subst h
    exact List.Forall₂.cons hb (mapM_forall₂ hbs)

/-- **The record's templates**: named once, one a template node, in order, each the rows (`templateOf`) of its node's
Definition, whose key is the template's descriptor id. -/
theorem recordOf_ok {P : Prog} {j : Json} {r : Stated} (h : recordOf P j = .ok r) :
    (r.templates.map (·.1)).Nodup ∧ (templateNodes P).length = r.templates.length ∧
      List.Forall₂ (fun (x : ℕ × String × ℕ) (y : String × Template) =>
          (templateDef P x.1).key = x.2.1 ∧ y.1 = x.2.1 ∧ templateOf (templateDef P x.1) x.2.2 = some y.2)
        ((templateNodes P).zip r.templates) r.wiring.templates := by
  unfold recordOf at h
  try simp only [] at h
  rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩
  · exact absurd h (by intro h'; injection h')
  rcases hi : j.getObjVal? "ingress" with e | a
  on_goal 2 => cases a
  all_goals simp only [hi, pure_bind] at h
  all_goals first | exact absurd h (by intro h'; injection h') | skip
  on_goal 2 =>
    rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩
    · exact absurd h (by intro h'; injection h')
  all_goals
    rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩
    swap
    · exact absurd h (by intro h'; injection h')
    rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩
    swap
    · exact absurd h (by intro h'; injection h')
    obtain ⟨cap, -, h⟩ := bind_ok h
    obtain ⟨tj, -, h⟩ := bind_ok h
    obtain ⟨tl, -, h⟩ := bind_ok h
    obtain ⟨named, -, h⟩ := bind_ok h
    rcases ite_ok h with ⟨hnd, h⟩ | ⟨-, h⟩
    swap
    · exact absurd h (by intro h'; injection h')
    rcases ite_ok h with ⟨hlen, h⟩ | ⟨-, h⟩
    swap
    · exact absurd h (by intro h'; injection h')
    obtain ⟨templates, htm, h⟩ := bind_ok h
    obtain ⟨wj, -, h⟩ := bind_ok h
    obtain ⟨wl, -, h⟩ := bind_ok h
    obtain ⟨wires, -, h⟩ := bind_ok h
    have hr : r.templates = named ∧ r.wiring.templates = templates := by
      rcases hc : j.getObjVal? "cuts" with ec | cj <;> rcases hg : j.getObjVal? "registered" with eg | rj <;>
        simp only [hc, hg, pure_bind] at h <;> repeat (obtain ⟨_, -, h⟩ := bind_ok h)
      all_goals
        injection h with h
        subst h
        exact ⟨rfl, rfl⟩
    obtain ⟨hr1, hr2⟩ := hr
    rw [hr1, hr2]
    refine ⟨of_decide_eq_true hnd, beq_iff_eq.1 hlen, (mapM_forall₂ htm).imp ?_⟩
    rintro ⟨k, tid, n⟩ b hb
    show (templateDef P k).key = tid ∧ b.1 = tid ∧ templateOf (templateDef P k) n = some b.2
    simp only [] at hb
    rcases ite_ok hb with ⟨hkey, hb⟩ | ⟨-, hb⟩
    swap
    · exact absurd hb (by intro h'; injection h')
    cases ht : templateOf (templateDef P k) n with
    | none =>
      rw [ht] at hb
      exact absurd hb (by intro h'; injection h')
    | some t =>
      simp only [ht] at hb
      injection hb with hb
      subst hb
      exact ⟨beq_iff_eq.1 hkey, rfl, rfl⟩

/-! ## The population -/

/-- **The population's template nodes**: the record's templates with their units, each node's Definition keyed as its
template, and each node's arguments whole input nodes of its own (`argsOk`). -/
theorem populationOk_ok {P : Prog} {r : Stated} (h : populationOk P r = true) :
    (templateNodes P).length = r.templates.length ∧
      (∀ x ∈ (templateNodes P).zip r.templates,
        (templateDef P x.1).key = x.2.1 ∧ ((P.defs[P.root]!).body.nodes[x.1]!).n = x.2.2) ∧
      ∀ k ∈ templateNodes P, argsOk P k = true := by
  unfold populationOk at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hlen, hall⟩, hargs⟩, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩ := h
  refine ⟨beq_iff_eq.1 hlen, ?_, fun k hk => List.all_eq_true.1 hargs k hk⟩
  rintro ⟨k, tid, n⟩ hx
  have hx' := List.all_eq_true.1 hall _ hx
  simp only [Bool.and_eq_true, beq_iff_eq] at hx'
  exact hx'

/-- A template node whose arguments `argsOk` accepts is a batch. -/
theorem batch_of_argsOk {P : Prog} {k : ℕ} (h : argsOk P k = true) :
    ((P.defs[P.root]!).body.nodes[k]!).form = .batch := by
  unfold argsOk at h
  simp only [Bool.and_eq_true] at h
  exact Properties.NetworkScan.Fits.form_batch h.1.1.1

/-! ## The wires -/

/-- **What the wires' check accepted**: the cuts, each wire's own refusal none, and no reader row wired twice. -/
theorem wires_ok {c : Wiring} {s : SampledProofs.Wires.Summary} (h : SampledProofs.Wires.check c = .ok s) :
    SampledProofs.Wires.cutsOk c = true ∧ (∀ w ∈ c.wires, SampledProofs.Wires.wireCode c w = none) ∧
      SampledProofs.Wires.pairwiseDisjoint c.wires = true := by
  unfold SampledProofs.Wires.check at h
  split at h
  · cases h
  · rename_i hcut
    split at h
    · cases h
    · rename_i hf
      split at h
      · cases h
      · rename_i hov
        exact ⟨by simpa using hcut, SampledProofs.Wires.firstWire_eq_none.1 hf, by simpa using hov⟩

/-- A wire's shape: its reader a template, and its rows among its reader port's. -/
theorem shape_ok {c : Wiring} {w : Wire} {rt wt : Template} (h : SampledProofs.Wires.shape c w = .ok (rt, wt)) :
    c.templates.lookup w.reader = some rt ∧ w.cut0 + w.cuts ≤ c.cutsOf w.reader w.port := by
  unfold SampledProofs.Wires.shape at h
  cases h1 : c.templates.lookup w.reader with
  | none => simp [h1] at h
  | some rt' =>
    cases h2 : c.templates.lookup w.writer with
    | none => simp [h1, h2] at h
    | some wt' =>
      cases h3 : rt'.rows.lookup w.port with
      | none => simp [h1, h2, h3] at h
      | some p =>
        cases h4 : wt'.outs.lookup w.out with
        | none => simp [h1, h2, h3, h4] at h
        | some ow =>
          simp only [h1, h2, h3, h4] at h
          split at h
          · cases h
          · rename_i hc
            split at h
            · cases h
            · cases h
              simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true, decide_eq_true_eq,
                beq_iff_eq] at hc
              exact ⟨rfl, hc.1.1.2⟩

/-- **A wire the check accepts** lies among its reader port's rows and its reader's units. -/
theorem wireCode_none {c : Wiring} {w : Wire} (h : SampledProofs.Wires.wireCode c w = none) :
    w.cut0 + w.cuts ≤ c.cutsOf w.reader w.port ∧
      ∀ rt, c.templates.lookup w.reader = some rt → w.start + w.readerUnits ≤ rt.units := by
  unfold SampledProofs.Wires.wireCode at h
  cases hs : SampledProofs.Wires.shape c w with
  | error e => simp [hs] at h
  | ok x =>
    obtain ⟨rt, wt⟩ := x
    simp only [hs] at h
    obtain ⟨hl, hcut⟩ := shape_ok hs
    refine ⟨hcut, fun rt' hrt' => ?_⟩
    rw [hl] at hrt'
    cases hrt'
    by_cases hb : SampledProofs.Wires.boundsOk w rt wt = true
    · simp only [SampledProofs.Wires.boundsOk, Bool.and_eq_true, decide_eq_true_eq] at hb
      exact hb.1.1
    · simp [hb] at h

theorem lookup_mem {α β : Type} [BEq α] [LawfulBEq α] {l : List (α × β)} {a : α} {b : β}
    (h : l.lookup a = some b) : (a, b) ∈ l := by
  obtain ⟨l₁, l₂, rfl, -⟩ := List.lookup_eq_some_iff.1 h
  exact List.mem_append_right _ List.mem_cons_self

/-- **A port's cut** is at least 1 and divides its words, where the cuts pass. -/
theorem cutsOf_ok {c : Wiring} (h : SampledProofs.Wires.cutsOk c = true) {tid p : String} {t : Template}
    {q : SampledProofs.Wires.Port} (ht : c.templates.lookup tid = some t) (hq : t.rows.lookup p = some q) :
    1 ≤ c.cutsOf tid p ∧ q.words % c.cutsOf tid p = 0 := by
  unfold SampledProofs.Wires.Wiring.cutsOf
  cases hkv : c.cuts.lookup tid with
  | none => simp [Nat.mod_one]
  | some kv =>
    cases hk : kv.lookup p with
    | none => simp [hk, Nat.mod_one]
    | some k =>
      simp only [Option.getD_some, hk]
      have hall := List.all_eq_true.1 h (tid, kv) (lookup_mem hkv)
      simp only [ht] at hall
      have hk' := List.all_eq_true.1 hall (p, k) (lookup_mem hk)
      simp only [hq, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hk'
      exact ⟨hk'.1.1, hk'.2⟩

/-- **A wire's writer** is a template node with an output row `out`, where the leaves pass. -/
theorem leavesOk_ok {P : Prog} {r : Stated} (h : leavesOk P r = true) {w : Wire} (hw : w ∈ r.wiring.wires) :
    ∃ kw, nodeOf P r w.writer = some kw ∧ (fieldOf (templateDef P kw).ret w.out).isSome = true := by
  have h1 := List.all_eq_true.1 h w hw
  cases hr : nodeOf P r w.reader with
  | none => simp [hr] at h1
  | some kr =>
    cases hw' : nodeOf P r w.writer with
    | none => simp [hr, hw'] at h1
    | some kw =>
      refine ⟨kw, rfl, ?_⟩
      cases hf : fieldOf (templateDef P kw).ret w.out with
      | some f => rfl
      | none =>
        exfalso
        cases hp : (templateDef P kr).params.find? (·.1 == w.port) with
        | none => simp [hr, hw', hf, hp] at h1
        | some pt =>
          obtain ⟨_, tp⟩ := pt
          simp [hr, hw', hf, hp] at h1

/-! ## The declarations -/

/-- A registered table read's rows: its units, every row of its port. -/
def rootRect (c : Wiring) (x : TableRead) : Rect :=
  ⟨x.reader, x.port, x.start, x.start + x.units, 0, c.cutsOf x.reader x.port⟩

/-- A registered opening's rows: its units, every row of each of its ports. -/
def openRects (c : Wiring) (o : TableOpening) : List Rect :=
  o.ports.map fun p => ⟨o.reader, p, o.start, o.start + o.units, 0, c.cutsOf o.reader p⟩

/-- What `declRects` checks of a registered read: at an input port its rows among the port's, its units cut whole,
and its units its template's. -/
structure ReadOk (c : Wiring) (x : Read) : Prop where
  cut : ∀ t q, c.templates.lookup x.pat.reader = some t → t.rows.lookup x.pat.port = some q →
    x.pat.cut0 + x.pat.cuts ≤ c.cutsOf x.pat.reader x.pat.port
  div : x.pat.units % x.pat.cuts = 0
  units : unitsOk c x.pat.reader x.pat.start x.pat.readerUnits = true

theorem portOf_rows {c : Wiring} {tid p : String} {t : Template} {q : SampledProofs.Wires.Port} (outs : Bool)
    (ht : c.templates.lookup tid = some t) (hq : t.rows.lookup p = some q) :
    portOf c tid p outs = some (q.words, c.cutsOf tid p) := by
  simp only [portOf, ht, hq]

theorem portOf_false {c : Wiring} {tid p : String} {words k : ℕ} (h : portOf c tid p false = some (words, k)) :
    k = c.cutsOf tid p := by
  unfold portOf at h
  cases ht : c.templates.lookup tid with
  | none => simp [ht] at h
  | some t =>
    cases hq : t.rows.lookup p with
    | none => simp [ht, hq] at h
    | some q =>
      simp only [ht, hq, Option.some.injEq, Prod.mk.injEq] at h
      exact h.2.symm

theorem forall₂_map {γ α β : Type} {f : γ → α} {g : α → β} {Q : α → Prop} :
    ∀ {l : List γ} {L : List β}, List.Forall₂ (fun p R => R = g (f p) ∧ Q (f p)) l L →
      L = (l.map f).map g ∧ ∀ x ∈ l.map f, Q x
  | [], [], .nil => ⟨rfl, fun _ h => by simp at h⟩
  | _ :: _, _ :: _, .cons ⟨h1, h2⟩ h => by
    obtain ⟨e, hq⟩ := forall₂_map h
    refine ⟨by rw [h1, e]; rfl, fun y hy => ?_⟩
    rcases List.mem_cons.1 hy with rfl | hy
    · exact h2
    · exact hq y hy

/-- **The declarations' rows**: each read's pattern's, each table read's and each opening's ports', in order; each read
among its port's rows, and each table read and opening of its template's units. -/
theorem declRects_ok {c : Wiring} {reg : Registered} {rects : List Rect} (h : declRects c reg = .ok rects) :
    rects = reg.reads.map (fun x => wireRect x.pat) ++ reg.roots.map (rootRect c) ++
        (reg.openings.map (openRects c)).flatten ∧
      (∀ x ∈ reg.reads, ReadOk c x) ∧ (∀ x ∈ reg.roots, unitsOk c x.reader x.start x.units = true) ∧
      ∀ o ∈ reg.openings, unitsOk c o.reader o.start o.units = true := by
  unfold declRects at h
  try simp only [] at h
  rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩
  swap
  · exact absurd h (by intro h'; injection h')
  obtain ⟨tables, -, h⟩ := bind_ok h
  obtain ⟨reads, hreads, h⟩ := bind_ok h
  obtain ⟨roots, hroots, h⟩ := bind_ok h
  obtain ⟨opens, hopens, h⟩ := bind_ok h
  injection h with h
  subst h
  have hr : List.Forall₂ (fun (p : Read × ℕ) R => R = wireRect p.1.pat ∧ ReadOk c p.1) reg.reads.zipIdx reads := by
    refine (mapM_forall₂ hreads).imp ?_
    rintro ⟨x, i⟩ R hR
    simp only [] at hR
    cases hp : portOf c x.pat.reader x.pat.port true with
    | none =>
      rw [hp] at hR
      exact absurd hR (by intro h'; injection h')
    | some wk =>
      obtain ⟨words, k⟩ := wk
      simp only [hp] at hR
      rcases ite_ok hR with ⟨hcut, hR⟩ | ⟨-, hR⟩
      swap
      · exact absurd hR (by intro h'; injection h')
      rcases ite_ok hR with ⟨hunits, hR⟩ | ⟨-, hR⟩
      swap
      · exact absurd hR (by intro h'; injection h')
      rcases ite_ok hR with ⟨-, hR⟩ | ⟨-, hR⟩
      swap
      · exact absurd hR (by intro h'; injection h')
      injection hR with hR
      subst hR
      simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hcut
      refine ⟨rfl, ⟨fun t q ht hq => ?_, hcut.2, hunits⟩⟩
      rw [portOf_rows true ht hq] at hp
      obtain ⟨-, hk⟩ := Prod.mk.inj (Option.some.inj hp)
      rw [hk]
      exact hcut.1.1
  have hro : List.Forall₂ (fun (p : TableRead × ℕ) R => R = rootRect c p.1 ∧
      unitsOk c p.1.reader p.1.start p.1.units = true) reg.roots.zipIdx roots := by
    refine (mapM_forall₂ hroots).imp ?_
    rintro ⟨x, i⟩ R hR
    simp only [] at hR
    cases hp : portOf c x.reader x.port false with
    | none =>
      rw [hp] at hR
      exact absurd hR (by intro h'; injection h')
    | some wk =>
      obtain ⟨words, k⟩ := wk
      simp only [hp] at hR
      rcases ite_ok hR with ⟨hc, hR⟩ | ⟨-, hR⟩
      swap
      · exact absurd hR (by intro h'; injection h')
      injection hR with hR
      subst hR
      simp only [Bool.and_eq_true] at hc
      refine ⟨?_, hc.1.1.1⟩
      rw [portOf_false hp]
      rfl
  have hop : List.Forall₂ (fun (p : TableOpening × ℕ) Rs => Rs = openRects c p.1 ∧
      unitsOk c p.1.reader p.1.start p.1.units = true) reg.openings.zipIdx opens := by
    refine (mapM_forall₂ hopens).imp ?_
    rintro ⟨o, i⟩ Rs hR
    simp only [] at hR
    rcases ite_ok hR with ⟨hc, hR⟩ | ⟨-, hR⟩
    swap
    · exact absurd hR (by intro h'; injection h')
    simp only [Bool.and_eq_true] at hc
    refine ⟨?_, hc.1.1.1.1⟩
    have hF : List.Forall₂ (fun (p : String) R =>
        R = (⟨o.reader, p, o.start, o.start + o.units, 0, c.cutsOf o.reader p⟩ : Rect) ∧ True) o.ports Rs := by
      refine (mapM_forall₂ hR).imp ?_
      intro p R hpR
      cases hp : portOf c o.reader p false with
      | none =>
        rw [hp] at hpR
        exact absurd hpR (by intro h'; injection h')
      | some wk =>
        obtain ⟨words, k⟩ := wk
        simp only [hp] at hpR
        injection hpR with hpR
        subst hpR
        refine ⟨?_, trivial⟩
        rw [portOf_false hp]
    obtain ⟨e, -⟩ := forall₂_map (f := id)
      (g := fun p => (⟨o.reader, p, o.start, o.start + o.units, 0, c.cutsOf o.reader p⟩ : Rect))
      (Q := fun _ => True) hF
    rw [List.map_id] at e
    exact e
  obtain ⟨e1, q1⟩ := forall₂_map (f := Prod.fst) (g := fun x : Read => wireRect x.pat) (Q := ReadOk c) hr
  obtain ⟨e2, q2⟩ := forall₂_map (f := Prod.fst) (g := rootRect c)
    (Q := fun x : TableRead => unitsOk c x.reader x.start x.units = true) hro
  obtain ⟨e3, q3⟩ := forall₂_map (f := Prod.fst) (g := openRects c)
    (Q := fun o : TableOpening => unitsOk c o.reader o.start o.units = true) hop
  simp only [List.zipIdx_map_fst] at e1 e2 e3 q1 q2 q3
  subst e1 e2 e3
  exact ⟨rfl, q1, q2, q3⟩

/-! ## No row filled twice -/

/-- Two rows' rectangles that don't meet: their units or their rows apart, or another template or port. -/
def RDisj (a b : Rect) : Prop :=
  a.hi ≤ b.lo ∨ b.hi ≤ a.lo ∨ a.r1 ≤ b.r0 ∨ b.r1 ≤ a.r0 ∨ a.reader ≠ b.reader ∨ a.port ≠ b.port

theorem rdisj_of {a b : Rect} (h : a.disjoint b = true) : RDisj a b := by
  simp only [Rect.disjoint, Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne] at h
  rcases h with ((((h | h) | h) | h) | h) | h
  · exact .inl h
  · exact .inr (.inl h)
  · exact .inr (.inr (.inl h))
  · exact .inr (.inr (.inr (.inl h)))
  · exact .inr (.inr (.inr (.inr (.inl h))))
  · exact .inr (.inr (.inr (.inr (.inr h))))

theorem rdisj_symm {a b : Rect} (h : RDisj a b) : RDisj b a := by
  rcases h with h | h | h | h | h | h
  · exact .inr (.inl h)
  · exact .inl h
  · exact .inr (.inr (.inr (.inl h)))
  · exact .inr (.inr (.inl h))
  · exact .inr (.inr (.inr (.inr (.inl (Ne.symm h)))))
  · exact .inr (.inr (.inr (.inr (.inr (Ne.symm h)))))

/-- **`apart`'s acceptance**: each of `rs` meets none of `earlier`, and no two of `rs` meet. -/
theorem apart_ok : ∀ (rs earlier : List Rect), apart rs earlier = true →
    (∀ a ∈ rs, ∀ b ∈ earlier, RDisj a b) ∧ rs.Pairwise RDisj
  | [], _, _ => ⟨fun _ h => by simp at h, List.Pairwise.nil⟩
  | r :: rs, earlier, h => by
    simp only [apart, Bool.and_eq_true, List.all_eq_true] at h
    obtain ⟨h1, h2⟩ := h
    obtain ⟨ih1, ih2⟩ := apart_ok rs (r :: earlier) h2
    refine ⟨fun a ha b hb => ?_,
      List.Pairwise.cons (fun a ha => rdisj_symm (ih1 a ha r List.mem_cons_self)) ih2⟩
    rcases List.mem_cons.1 ha with rfl | ha
    · exact rdisj_of (h1 b hb)
    · exact ih1 a ha b (List.mem_cons_of_mem _ hb)

theorem rdisj_wire {a b : Wire} (h : SampledProofs.Wires.disjoint a b = true) : RDisj (wireRect a) (wireRect b) := by
  simp only [SampledProofs.Wires.disjoint, Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne] at h
  unfold RDisj wireRect
  dsimp only
  rcases h with ((((h | h) | h) | h) | h) | h
  · exact .inr (.inr (.inr (.inr (.inl h))))
  · exact .inr (.inr (.inr (.inr (.inr h))))
  · exact .inl h
  · exact .inr (.inl h)
  · exact .inr (.inr (.inl h))
  · exact .inr (.inr (.inr (.inl h)))

/-- **The wires' rows meet nowhere**, where the wires' check finds no reader row wired twice. -/
theorem pairwise_wires {ws : List Wire} (h : SampledProofs.Wires.pairwiseDisjoint ws = true) :
    (ws.map wireRect).Pairwise RDisj :=
  List.pairwise_map.2 (((SampledProofs.Wires.pairwiseDisjoint_iff ws).1 h).imp rdisj_wire)

end Properties.NetworkCertifier.Counts
