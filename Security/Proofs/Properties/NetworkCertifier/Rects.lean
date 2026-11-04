import Mathlib.Algebra.Order.BigOperators.Group.List
import Proofs.Properties.NetworkCertifier.Record
import Proofs.Properties.NetworkCertifier.Slots

/-!
# The record's rows as slots

What the circuit count's proof reads off the record's templates. Template `i` of a record `recordOf` and
`populationOk` accept is the Program's template node `i` (`Aligned`), so template `tid`'s port `port` resolves to its
template's index and node and its parameter's index and type (`resolve`, as `portGates` finds them). A template's slots
are its units and its parameters' bits, each parameter on whole 16-bit words from its first bit (`at0`, `slots`). The
rows a wire or a declaration fills (`Rect`) are a box of its template's slots (`rectBox`): its units, and the bits of its
rows, each row its port's bits over its cut. Where each lies among its port's rows and its template's units (`Valid`),
two rectangles `RDisj` holds apart fill disjoint boxes (`rectBox_disjoint`), each among the slots (`rectBox_subset`).
-/

namespace Properties.NetworkCertifier.Counts

open Lean (Json)
open Flock.Program (Prog Ty)
open NetworkCheck.Period (templateNodes bitsOf)
open NetworkCheck.Circuit

/-! ## Lists -/

theorem lookup_of_mem {α β : Type} [BEq α] [LawfulBEq α] {l : List (α × β)} {a : α} {b : β}
    (hnd : (l.map (·.1)).Nodup) (h : (a, b) ∈ l) : l.lookup a = some b := by
  have hu : ∀ x ∈ l, x ≠ (a, b) → (fun p : α × β => if a == p.1 then some p.2 else none) x = none := by
    intro x hx hne
    have hxa : a ≠ x.1 := fun e => hne (List.inj_on_of_nodup_map hnd hx h e.symm)
    simp [beq_eq_false_iff_ne.2 hxa]
  rw [List.lookup_eq_findSome?, Properties.NetworkScan.Query.findSome?_eq_of_unique l h hu]
  simp

theorem findSome?_ite {α β : Type} {l : List α} {f : α → Option β} {c : β} (hl : l.findSome? f = some c)
    {p : α → Bool} {g : α → β} (hf : ∀ b, f b = if p b then some (g b) else none) :
    ∃ b ∈ l, p b = true ∧ g b = c := by
  obtain ⟨b, hb, hfb⟩ := List.exists_of_findSome?_eq_some hl
  rw [hf] at hfb
  by_cases hp : p b = true
  · rw [ite_eq_left hp] at hfb
    exact ⟨b, hb, hp, Option.some.inj hfb⟩
  · rw [ite_eq_right hp] at hfb
    cases hfb

theorem findSome?_ite_eq {α β : Type} {l : List α} {f : α → Option β} {c : β} {p : α → Bool} {g : α → β}
    (hf : ∀ b, f b = if p b then some (g b) else none) {a : α} (ha : a ∈ l) (hp : p a = true)
    (hu : ∀ b ∈ l, p b = true → b = a) (hg : g a = c) : l.findSome? f = some c := by
  have hnone : ∀ b ∈ l, b ≠ a → f b = none := by
    intro b hb hne
    rw [hf, ite_eq_right (fun hpb => hne (hu b hb hpb))]
  rw [Properties.NetworkScan.Query.findSome?_eq_of_unique l ha hnone, hf, ite_eq_left hp, hg]

theorem findIdx?_some {l : List ℕ} {k i : ℕ} (h : l.findIdx? (· == k) = some i) : ∃ hi : i < l.length, l[i] = k := by
  obtain ⟨hi, hp, -⟩ := List.findIdx?_eq_some_iff_getElem.1 h
  exact ⟨hi, by simpa using hp⟩

theorem findIdx?_getElem {l : List ℕ} (hnd : l.Nodup) {i : ℕ} (hi : i < l.length) :
    l.findIdx? (· == l[i]) = some i := by
  rw [List.findIdx?_eq_some_iff_getElem]
  refine ⟨hi, by simp, fun j hji => ?_⟩
  have hne : l[j] ≠ l[i] := fun e => by
    have := (List.Nodup.getElem_inj_iff hnd).1 e
    omega
  simpa using hne

/-- Ports `π` past `n` of the list `l`, where `F` keeps each name: the first named `a`'s image, as `l`'s `zipIdx`
finds it. -/
theorem lookup_map_zipIdx {β γ : Type} {F : String × γ → String × β} {g : γ → β} (hF : ∀ x, F x = (x.1, g x.2))
    (a : String) : ∀ (l : List (String × γ)) (n : ℕ),
      (l.map F).lookup a = ((l.zipIdx n).find? (·.1.1 == a)).map fun e => g e.1.2
  | [], _ => rfl
  | x :: l, n => by
    have ih := lookup_map_zipIdx hF a l (n + 1)
    simp only [List.map_cons, hF x, List.lookup_cons, List.zipIdx_cons, List.find?_cons]
    by_cases h : x.1 = a
    · subst h
      simp
    · simp only [beq_eq_false_iff_ne.2 (Ne.symm h), beq_eq_false_iff_ne.2 h]
      exact ih

theorem lookup_zipIdx {β γ : Type} {rows : List (String × β)} {l : List (String × γ)} {F : String × γ → String × β}
    {g : γ → β} (hl : rows = l.map F) (hF : ∀ x, F x = (x.1, g x.2)) (a : String) :
    rows.lookup a = (l.zipIdx.find? (·.1.1 == a)).map fun e => g e.1.2 := by
  rw [hl]
  exact lookup_map_zipIdx hF a l 0

/-! ## The templates -/

/-- **The record beside the Program**, as `recordOf` and `populationOk` accept it: a template a template node, named
once, template `i` node `i`'s Definition's rows (`templateOf`) under its key, with the node's units, and each node's
arguments whole input nodes of its own. -/
structure Aligned (P : Prog) (r : Stated) : Prop where
  len : (templateNodes P).length = r.templates.length
  names : (r.templates.map (·.1)).Nodup
  rel : List.Forall₂ (fun (x : ℕ × String × ℕ) (y : String × Template) =>
      (templateDef P x.1).key = x.2.1 ∧ y.1 = x.2.1 ∧ templateOf (templateDef P x.1) x.2.2 = some y.2)
    ((templateNodes P).zip r.templates) r.wiring.templates
  units : ∀ x ∈ (templateNodes P).zip r.templates, ((P.defs[P.root]!).body.nodes[x.1]!).n = x.2.2
  args : ∀ k ∈ templateNodes P, argsOk P k = true

theorem aligned {P : Prog} {j : Json} {r : Stated} (hr : recordOf P j = .ok r) (hp : populationOk P r = true) :
    Aligned P r := by
  obtain ⟨hn, hl, hrel⟩ := recordOf_ok hr
  obtain ⟨-, hu, ha⟩ := populationOk_ok hp
  exact ⟨hl, hn, hrel, fun x hx => (hu x hx).2, ha⟩

theorem nodup_templateNodes (P : Prog) : (templateNodes P).Nodup := by
  unfold templateNodes
  exact List.Nodup.filter _ List.nodup_range

/-- **Template `i`**: node `i`, the record's template `i` and the wiring's, the node's Definition's rows under its key,
with the node's units. -/
theorem Aligned.entry {P : Prog} {r : Stated} (hA : Aligned P r) {i : ℕ} (hi : i < (templateNodes P).length) :
    ∃ (h1 : i < r.templates.length) (h2 : i < r.wiring.templates.length),
      (templateDef P (templateNodes P)[i]).key = r.templates[i].1 ∧ r.wiring.templates[i].1 = r.templates[i].1 ∧
      templateOf (templateDef P (templateNodes P)[i]) r.templates[i].2 = some r.wiring.templates[i].2 ∧
      ((P.defs[P.root]!).body.nodes[(templateNodes P)[i]]!).n = r.templates[i].2 := by
  have h1 : i < r.templates.length := by have := hA.len; omega
  have hz : i < ((templateNodes P).zip r.templates).length := by rw [List.length_zip]; omega
  have h2 : i < r.wiring.templates.length := by have := hA.rel.length_eq; omega
  have hR := hA.rel.get hz h2
  simp only [List.get_eq_getElem, List.getElem_zip] at hR
  have hU := hA.units _ (List.getElem_mem hz)
  simp only [List.getElem_zip] at hU
  exact ⟨h1, h2, hR.1, hR.2.1, hR.2.2, hU⟩

theorem Aligned.length {P : Prog} {r : Stated} (hA : Aligned P r) :
    r.wiring.templates.length = (templateNodes P).length := by
  have h := hA.rel.length_eq
  rw [List.length_zip] at h
  have := hA.len
  omega

theorem Aligned.name_inj {P : Prog} {r : Stated} (hA : Aligned P r) {i j : ℕ} (hi : i < r.templates.length)
    (hj : j < r.templates.length) (h : r.templates[i].1 = r.templates[j].1) : i = j := by
  have hi' : i < (r.templates.map (·.1)).length := by rw [List.length_map]; exact hi
  have hj' : j < (r.templates.map (·.1)).length := by rw [List.length_map]; exact hj
  refine (List.Nodup.getElem_inj_iff hA.names (hi := hi') (hj := hj')).1 ?_
  rw [List.getElem_map, List.getElem_map]
  exact h

theorem Aligned.wiring_names {P : Prog} {r : Stated} (hA : Aligned P r) :
    r.wiring.templates.map (·.1) = r.templates.map (·.1) := by
  have hl := hA.length
  have hl' := hA.len
  apply List.ext_getElem (by simp only [List.length_map]; omega)
  intro i h1 h2
  rw [List.getElem_map, List.getElem_map]
  have hi : i < (templateNodes P).length := by simp only [List.length_map] at h1; omega
  obtain ⟨_, _, -, he, -, -⟩ := hA.entry hi
  exact he

theorem Aligned.wnames {P : Prog} {r : Stated} (hA : Aligned P r) : (r.wiring.templates.map (·.1)).Nodup := by
  rw [hA.wiring_names]
  exact hA.names

theorem lookup_getElem {l : List (String × Template)} (hnd : (l.map (·.1)).Nodup) {i : ℕ} (hi : i < l.length) :
    l.lookup l[i].1 = some l[i].2 :=
  lookup_of_mem hnd (List.getElem_mem hi)

/-- **The template node of template `tid`** is node `i` of the template `i` named `tid`. -/
theorem nodeOf_some {P : Prog} {r : Stated} {tid : String} {k : ℕ} (h : nodeOf P r tid = some k) :
    ∃ i, ∃ (h1 : i < (templateNodes P).length) (h2 : i < r.templates.length),
      (templateNodes P)[i] = k ∧ r.templates[i].1 = tid := by
  obtain ⟨x, hx, hp, hg⟩ := findSome?_ite h (p := fun x => x.2.1 == tid) (g := fun x => x.1) (fun _ => rfl)
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hx
  simp only [List.getElem_zip, beq_iff_eq] at hp hg
  exact ⟨i, List.lt_length_left_of_zip hi, List.lt_length_right_of_zip hi, hg, hp⟩

theorem nodeOf_at {P : Prog} {r : Stated} (hA : Aligned P r) {i : ℕ} (hi : i < (templateNodes P).length)
    (hr : i < r.templates.length) : nodeOf P r r.templates[i].1 = some (templateNodes P)[i] := by
  have hz : i < ((templateNodes P).zip r.templates).length := by rw [List.length_zip]; omega
  unfold nodeOf
  refine findSome?_ite_eq (p := fun x => x.2.1 == r.templates[i].1) (g := fun x => x.1) (fun _ => rfl)
    (List.getElem_mem hz) (by simp [List.getElem_zip]) ?_ (by simp [List.getElem_zip])
  intro b hb hp
  obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hb
  simp only [List.getElem_zip, beq_iff_eq] at hp
  obtain rfl := hA.name_inj (List.lt_length_right_of_zip hj) hr hp
  rfl

/-- **Template node `k`'s parameter `port`** is its parameter `π`. -/
theorem portParam_find {P : Prog} {k : ℕ} {port : String} {π : ℕ} {ty : Ty} (h : portParam P k port = some (π, ty)) :
    ∃ name, (templateDef P k).params.toList.zipIdx.find? (·.1.1 == port) = some ((name, ty), π) := by
  unfold portParam at h
  obtain ⟨⟨⟨name, t⟩, j⟩, hf, he⟩ := Option.map_eq_some_iff.1 h
  simp only [Prod.mk.injEq] at he
  obtain ⟨rfl, rfl⟩ := he
  exact ⟨name, hf⟩

theorem portParam_some {P : Prog} {k : ℕ} {port : String} {π : ℕ} {ty : Ty} (h : portParam P k port = some (π, ty)) :
    (templateDef P k).params.toList[π]? = some (port, ty) := by
  obtain ⟨name, hf⟩ := portParam_find h
  have hp := List.find?_some hf
  simp only [beq_iff_eq] at hp
  subst hp
  exact List.mem_zipIdx_iff_getElem?.1 (List.mem_of_find?_eq_some hf)

theorem portParam_of_find {P : Prog} {k : ℕ} {port : String} {e : (String × Ty) × ℕ}
    (h : (templateDef P k).params.toList.zipIdx.find? (·.1.1 == port) = some e) :
    portParam P k port = some (e.2, e.1.2) := by
  obtain ⟨⟨name, t⟩, i⟩ := e
  unfold portParam
  rw [h]
  rfl

/-- **A template's rows**, where `templateOf` gives them: its units, each parameter a row, and each row's words its
parameter's bits over 16. -/
theorem templateOf_some {d : Flock.Program.Defn} {n : ℕ} {t : Template} (h : templateOf d n = some t) :
    t.units = n ∧ d.params.all (rowOk ·.2) = true ∧ ∀ p, t.rows.lookup p =
      (d.params.toList.zipIdx.find? (·.1.1 == p)).map fun e =>
        (⟨bitsOf e.1.2 / 16, outerOf e.1.2⟩ : SampledProofs.Wires.Port) := by
  unfold templateOf at h
  try dsimp only at h
  split at h
  · rename_i hc
    injection h with h
    subst h
    simp only [Bool.and_eq_true] at hc
    exact ⟨rfl, hc.1.1.2, fun p => lookup_zipIdx (g := fun t => (⟨bitsOf t / 16, outerOf t⟩ : SampledProofs.Wires.Port))
      rfl (fun _ => rfl) p⟩
  · cases h

/-! ## Resolving a port -/

/-- **Where template `tid`'s port `port` is**: its template's index and node, and the port's parameter's index and
type, as `portGates` finds them. -/
def resolve (P : Prog) (r : Stated) (tid port : String) : Option (ℕ × ℕ × ℕ × Ty) :=
  (nodeOf P r tid).bind fun k => (portParam P k port).bind fun q =>
    ((templateNodes P).findIdx? (· == k)).map fun i => (i, k, q.1, q.2)

theorem resolve_some {P : Prog} {r : Stated} {tid port : String} {i k π : ℕ} {ty : Ty}
    (h : resolve P r tid port = some (i, k, π, ty)) :
    nodeOf P r tid = some k ∧ portParam P k port = some (π, ty) ∧ (templateNodes P).findIdx? (· == k) = some i := by
  unfold resolve at h
  obtain ⟨k', h1, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨⟨π', ty'⟩, h2, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨i', h3, he⟩ := Option.map_eq_some_iff.1 h
  simp only [Prod.mk.injEq] at he
  obtain ⟨rfl, rfl, rfl, rfl⟩ := he
  exact ⟨h1, h2, h3⟩

theorem resolve_of {P : Prog} {r : Stated} {tid port : String} {i k π : ℕ} {ty : Ty} (h1 : nodeOf P r tid = some k)
    (h2 : portParam P k port = some (π, ty)) (h3 : (templateNodes P).findIdx? (· == k) = some i) :
    resolve P r tid port = some (i, k, π, ty) := by
  simp only [resolve, h1, h2, h3, Option.bind_some, Option.map_some]

theorem resolve_name {P : Prog} {r : Stated} {tid port : String} {i k π : ℕ} {ty : Ty}
    (h : resolve P r tid port = some (i, k, π, ty)) : ∃ hi : i < r.templates.length, r.templates[i].1 = tid := by
  obtain ⟨h1, -, h3⟩ := resolve_some h
  obtain ⟨hi, hk⟩ := findIdx?_some h3
  obtain ⟨j, hj, hj', hjk, hjt⟩ := nodeOf_some h1
  obtain rfl : j = i := (List.Nodup.getElem_inj_iff (nodup_templateNodes P)).1 (hjk.trans hk.symm)
  exact ⟨hj', hjt⟩

/-- **A resolved port is a row of its template**: template `i` is node `k`, the port is its parameter `π`, and the
wiring's template has the port's row, of its parameter's words, and the node's units. -/
theorem resolve_entry {P : Prog} {r : Stated} {tid port : String} {i k π : ℕ} {ty : Ty} (hA : Aligned P r)
    (h : resolve P r tid port = some (i, k, π, ty)) :
    i < (templateNodes P).length ∧ (templateNodes P).getD i 0 = k ∧
      (templateDef P k).params.toList[π]? = some (port, ty) ∧
      ∃ t, r.wiring.templates.lookup tid = some t ∧
        t.rows.lookup port = some (⟨bitsOf ty / 16, outerOf ty⟩ : SampledProofs.Wires.Port) ∧
        t.units = ((P.defs[P.root]!).body.nodes[k]!).n := by
  obtain ⟨-, h2, h3⟩ := resolve_some h
  obtain ⟨hi, hk⟩ := findIdx?_some h3
  obtain ⟨hi', hname⟩ := resolve_name h
  obtain ⟨_, hw, -, hwt, htmpl, hn⟩ := hA.entry hi
  rw [hk] at htmpl hn
  obtain ⟨hunits, -, hrows⟩ := templateOf_some htmpl
  obtain ⟨name, hfind⟩ := portParam_find h2
  refine ⟨hi, by rw [Properties.NetworkScan.Query.list_getD_of_lt 0 hi, hk], portParam_some h2,
    r.wiring.templates[i].2, ?_, ?_, ?_⟩
  · rw [← hname, ← hwt]
    exact lookup_getElem hA.wnames hw
  · rw [hrows port, hfind]
    rfl
  · rw [hunits, hn]

/-- **A row of a template resolves**: where the wiring's template `tid` has the row `port`, the port resolves, its row
its parameter's words. -/
theorem resolve_of_lookup {P : Prog} {r : Stated} {tid port : String} {t : Template} {q : SampledProofs.Wires.Port}
    (hA : Aligned P r) (ht : r.wiring.templates.lookup tid = some t) (hq : t.rows.lookup port = some q) :
    ∃ i k π ty, resolve P r tid port = some (i, k, π, ty) ∧ q = ⟨bitsOf ty / 16, outerOf ty⟩ := by
  obtain ⟨j, hj, hjt⟩ := List.getElem_of_mem (lookup_mem ht)
  have hjN : j < (templateNodes P).length := by have := hA.length; omega
  obtain ⟨hr, _, -, hwt, htmpl, -⟩ := hA.entry hjN
  simp only [hjt] at hwt htmpl
  obtain ⟨-, -, hrows⟩ := templateOf_some htmpl
  rw [hrows port] at hq
  obtain ⟨e, he, rfl⟩ := Option.map_eq_some_iff.1 hq
  refine ⟨j, (templateNodes P)[j], e.2, e.1.2,
    resolve_of ?_ (portParam_of_find he) (findIdx?_getElem (nodup_templateNodes P) hjN), rfl⟩
  rw [hwt]
  exact nodeOf_at hA hjN hr

theorem portWidth_eq {P : Prog} {r : Stated} {tid port : String} {i k π : ℕ} {ty : Ty}
    (h : resolve P r tid port = some (i, k, π, ty)) : portWidth P r tid port = bitsOf ty := by
  obtain ⟨h1, h2, -⟩ := resolve_some h
  simp only [portWidth, h1, h2, Option.map_some, Option.getD_some]

/-! ## The slots -/

/-- Template node `k`'s parameters' types. -/
def psOf (P : Prog) (k : ℕ) : List Ty := (templateDef P k).params.toList.map (·.2)

/-- **Parameter `π`'s first slot** among template node `k`'s input bits, each parameter on whole 16-bit words. -/
def at0 (P : Prog) (k π : ℕ) : ℕ := 16 * (((psOf P k).take π).map fun t => (bitsOf t + 15) / 16).sum

theorem psOf_getElem? {P : Prog} {k π : ℕ} {port : String} {ty : Ty}
    (h : (templateDef P k).params.toList[π]? = some (port, ty)) : (psOf P k)[π]? = some ty := by
  rw [psOf, List.getElem?_map, h]
  rfl

theorem at0_succ (P : Prog) (k : ℕ) {π : ℕ} {ty : Ty} (h : (psOf P k)[π]? = some ty) :
    at0 P k (π + 1) = at0 P k π + 16 * ((bitsOf ty + 15) / 16) := by
  simp only [at0, List.take_add_one, h, Option.toList_some, List.map_append, List.sum_append, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil]
  omega

theorem at0_mono (P : Prog) (k : ℕ) {π π' : ℕ} (h : π ≤ π') : at0 P k π ≤ at0 P k π' := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  unfold at0
  rw [List.take_add, List.map_append, List.sum_append]
  omega

/-- **A port's rows lie in its parameter's slots**: rows below its cut `K` end at its parameter's bits, before the next
parameter's first slot. -/
theorem seg_le {P : Prog} {k π π' : ℕ} {ty : Ty} {K r1 : ℕ} (hπ : π < π') (hty : (psOf P k)[π]? = some ty)
    (hr1 : r1 ≤ K) : at0 P k π + r1 * (bitsOf ty / K) ≤ at0 P k π' := by
  have h1 := at0_succ P k hty
  have h2 := at0_mono P k (show π + 1 ≤ π' from hπ)
  have h3 : r1 * (bitsOf ty / K) ≤ bitsOf ty := (Nat.mul_le_mul_right _ hr1).trans (Nat.mul_div_le (bitsOf ty) K)
  have h4 : bitsOf ty ≤ 16 * ((bitsOf ty + 15) / 16) := by omega
  omega

/-- **The slots**: each template's units, and its parameters' bits, each parameter on whole 16-bit words. -/
def slots (P : Prog) : Finset Slot :=
  bigU ((List.range (templateNodes P).length).map fun i =>
    box i 0 ((P.defs[P.root]!).body.nodes[(templateNodes P).getD i 0]!).n 0
      (at0 P ((templateNodes P).getD i 0) (psOf P ((templateNodes P).getD i 0)).length))

/-- **The slots a rectangle fills**: its template's units `[lo, hi)`, and in its port's parameter the bits of its rows
`[r0, r1)`, each row the port's bits over its cut. None where the record names no such template or port. -/
def rectBox (P : Prog) (r : Stated) (R : Rect) : Finset Slot :=
  ((resolve P r R.reader R.port).map fun x =>
    box x.1 R.lo R.hi (at0 P x.2.1 x.2.2.1 + R.r0 * (bitsOf x.2.2.2 / r.wiring.cutsOf R.reader R.port))
      (at0 P x.2.1 x.2.2.1 + R.r1 * (bitsOf x.2.2.2 / r.wiring.cutsOf R.reader R.port))).getD ∅

theorem rectBox_some {P : Prog} {r : Stated} {R : Rect} {i k π : ℕ} {ty : Ty}
    (h : resolve P r R.reader R.port = some (i, k, π, ty)) :
    rectBox P r R = box i R.lo R.hi (at0 P k π + R.r0 * (bitsOf ty / r.wiring.cutsOf R.reader R.port))
      (at0 P k π + R.r1 * (bitsOf ty / r.wiring.cutsOf R.reader R.port)) := by
  rw [rectBox, h] <;> rfl

theorem rectBox_none {P : Prog} {r : Stated} {R : Rect} (h : resolve P r R.reader R.port = none) :
    rectBox P r R = ∅ := by
  rw [rectBox, h] <;> rfl

/-- **A rectangle among its port's rows and its template's units**: its rows end at its port's cut, and its units at
its template's. -/
structure Valid (r : Stated) (R : Rect) : Prop where
  rows : ∀ t q, r.wiring.templates.lookup R.reader = some t → t.rows.lookup R.port = some q →
    R.r1 ≤ r.wiring.cutsOf R.reader R.port
  units : ∀ t, r.wiring.templates.lookup R.reader = some t → R.hi ≤ t.units

/-- **Each rectangle's slots are slots.** -/
theorem rectBox_subset {P : Prog} {r : Stated} {R : Rect} (hA : Aligned P r) (hV : Valid r R) :
    rectBox P r R ⊆ slots P := by
  cases h : resolve P r R.reader R.port with
  | none => rw [rectBox_none h]; exact Finset.empty_subset _
  | some x =>
    obtain ⟨i, k, π, ty⟩ := x
    obtain ⟨hi, hk, hπ, t, ht, hq, hu⟩ := resolve_entry hA h
    rw [rectBox_some h]
    subst hk
    have hty := psOf_getElem? hπ
    have hlen : π < (psOf P ((templateNodes P).getD i 0)).length := (List.getElem?_eq_some_iff.1 hty).1
    refine Finset.Subset.trans (box_subset (Nat.zero_le _) ?_ (Nat.zero_le _) ?_)
      (subset_bigU (List.mem_map.2 ⟨i, List.mem_range.2 hi, rfl⟩))
    · rw [← hu]
      exact hV.units t ht
    · exact seg_le hlen hty (hV.rows t _ ht hq)

/-- **Rectangles held apart fill disjoint slots.** -/
theorem rectBox_disjoint {P : Prog} {r : Stated} {R R' : Rect} (hA : Aligned P r) (hV : Valid r R) (hV' : Valid r R')
    (h : RDisj R R') : Disjoint (rectBox P r R) (rectBox P r R') := by
  cases h1 : resolve P r R.reader R.port with
  | none => rw [rectBox_none h1]; exact Finset.disjoint_empty_left _
  | some x =>
    obtain ⟨i, k, π, ty⟩ := x
    cases h2 : resolve P r R'.reader R'.port with
    | none => rw [rectBox_none h2]; exact Finset.disjoint_empty_right _
    | some x' =>
      obtain ⟨i', k', π', ty'⟩ := x'
      rw [rectBox_some h1, rectBox_some h2]
      apply box_disjoint
      by_cases hrd : R.reader = R'.reader
      · by_cases hpt : R.port = R'.port
        · rw [hrd, hpt, h2] at h1
          simp only [Option.some.injEq, Prod.mk.injEq] at h1
          obtain ⟨rfl, rfl, rfl, rfl⟩ := h1
          rw [hrd, hpt]
          rcases h with h | h | h | h | h | h
          · exact .inr (.inl h)
          · exact .inr (.inr (.inl h))
          · exact .inr (.inr (.inr (.inl (Nat.add_le_add_left (Nat.mul_le_mul_right _ h) _))))
          · exact .inr (.inr (.inr (.inr (Nat.add_le_add_left (Nat.mul_le_mul_right _ h) _))))
          · exact absurd hrd h
          · exact absurd hpt h
        · obtain ⟨hn1, hp1, hf1⟩ := resolve_some h1
          obtain ⟨hn2, hp2, hf2⟩ := resolve_some h2
          rw [hrd, hn2] at hn1
          obtain rfl := Option.some.inj hn1
          obtain ⟨-, -, hπ1, t, ht, hq, -⟩ := resolve_entry hA h1
          obtain ⟨-, -, hπ2, t', ht', hq', -⟩ := resolve_entry hA h2
          have hππ : π ≠ π' := by
            rintro rfl
            rw [hπ1] at hπ2
            simp only [Option.some.injEq, Prod.mk.injEq] at hπ2
            exact hpt hπ2.1
          rcases Nat.lt_or_gt_of_ne hππ with hlt | hlt
          · have := seg_le hlt (psOf_getElem? hπ1) (hV.rows t _ ht hq)
            exact .inr (.inr (.inr (.inl (this.trans (Nat.le_add_right _ _)))))
          · have := seg_le hlt (psOf_getElem? hπ2) (hV'.rows t' _ ht' hq')
            exact .inr (.inr (.inr (.inr (this.trans (Nat.le_add_right _ _)))))
      · left
        rintro rfl
        obtain ⟨hi1, e1⟩ := resolve_name h1
        obtain ⟨hi2, e2⟩ := resolve_name h2
        exact hrd (e1.symm.trans e2)

theorem card_rectBox {P : Prog} {r : Stated} {R : Rect} {i k π : ℕ} {ty : Ty}
    (h : resolve P r R.reader R.port = some (i, k, π, ty)) :
    (rectBox P r R).card = (R.hi - R.lo) * (R.r1 * (bitsOf ty / r.wiring.cutsOf R.reader R.port) -
      R.r0 * (bitsOf ty / r.wiring.cutsOf R.reader R.port)) := by
  rw [rectBox_some h, card_box, Nat.add_sub_add_left]

end Properties.NetworkCertifier.Counts
