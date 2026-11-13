module

public import Properties.Ties
public import Proofs.Core.Site

/-!
# What a prefix registers, a longer prefix registers

No registration entry retires anything: a site entry adds sheets, locations and links, an information entry adds items,
and a declaration adds a day's circuit. So what the view of a prefix of the log holds, the view of every longer prefix
holds too (`view_take_prefix`): its θ, its declarations and its Site's sheets, locations and links extend. A part of the
registered hardware registered at a prefix is registered at every longer one (`parts_take_mono`), and each tie that
holds at the view of a prefix holds at the view of every longer one (`declaredIn_take_mono`, `certifiedIn_take_mono`,
`rootIn_take_mono`), since the set it reads only grows (`NamesRegistered.mono`).
-/

@[expose] public section

namespace Verity

namespace Registration

theorem NamesRegistered.mono {X Y : Type} {holds holds' : Set X} (h : holds ⊆ holds') {f : X → Y} {y : Y}
    (hn : NamesRegistered holds f y) : NamesRegistered holds' f y := by
  obtain ⟨x, hx, rfl⟩ := hn
  exact ⟨x, h hx, rfl⟩

/-- **A shorter prefix's view is a prefix of a longer one's**: its θ, its declarations and its Site's sheets,
locations and links. -/
theorem view_take_prefix (D : Digests) (c : Composite) (log : Registration) {n m : ℕ} (h : n ≤ m) :
    (view D c (log.take n)).theta <+: (view D c (log.take m)).theta ∧
      (view D c (log.take n)).declarations <+: (view D c (log.take m)).declarations ∧
      (view D c (log.take n)).site.sheets <+: (view D c (log.take m)).site.sheets ∧
      (view D c (log.take n)).site.locs <+: (view D c (log.take m)).site.locs ∧
      (view D c (log.take n)).site.links <+: (view D c (log.take m)).site.links := by
  have hp : log.take n <+: log.take m := by
    rw [show log.take n = (log.take m).take n by rw [List.take_take, Nat.min_eq_left h]]
    exact List.take_prefix n _
  obtain ⟨t, ht⟩ := hp
  simp only [view, Registration.theta, Registration.items, Site.ofLog, ← ht, List.flatMap_append,
    List.filterMap_append]
  exact ⟨⟨_, rfl⟩, ⟨_, rfl⟩, ⟨_, rfl⟩, ⟨_, rfl⟩, ⟨_, rfl⟩⟩

theorem loc?_mono {S S' : Site} (h : S.locs <+: S'.locs) {i : String} {x : Site.Loc} (hx : S.loc? i = some x) :
    S'.loc? i = some x := by
  obtain ⟨t, ht⟩ := h
  unfold Site.loc? at hx ⊢
  rw [← ht, List.find?_append, hx]
  rfl

theorem kind?_mono {S S' : Site} (h : S.locs <+: S'.locs) {i : String} {k : Site.Kind} (hk : S.kind? i = some k) :
    S'.kind? i = some k := by
  unfold Site.kind? at hk ⊢
  by_cases hi : i = Site.outside
  · simpa [hi] using hk
  · simp only [hi, ite_false] at hk ⊢
    obtain ⟨x, hx, rfl⟩ := Option.map_eq_some_iff.mp hk
    rw [loc?_mono h hx]
    rfl

theorem boundary_mono {S S' : Site} (h : S.locs <+: S'.locs) {k : Site.Link} (hb : S.Boundary k) :
    S'.Boundary k := by
  have hj : ∀ {a b : Site.Kind}, S.Joins k a b → S'.Joins k a b := by
    intro a b hj
    rcases hj with hj | hj <;> simp only [Prod.mk.injEq] at hj
    · exact Or.inl (by rw [kind?_mono h hj.1, kind?_mono h hj.2])
    · exact Or.inr (by rw [kind?_mono h hj.1, kind?_mono h hj.2])
  exact hb.imp hj hj

/-- **A part registered at a prefix is registered at every longer one**: no entry retires a location or a link, so
each device's compute, each memory leaf's memory and each boundary link's egress stays a part of the Site. -/
theorem parts_take_mono (D : Digests) (c : Composite) (log : Registration) {n m : ℕ} (h : n ≤ m) :
    (view D c (log.take n)).site.parts ⊆ (view D c (log.take m)).site.parts := by
  obtain ⟨-, -, -, hl, hk⟩ := view_take_prefix D c log h
  rintro xr (⟨x, hx, hd, rfl⟩ | ⟨x, hx, hd, hr⟩ | ⟨k, hk', hb, rfl⟩)
  · exact Or.inl ⟨x, hl.subset hx, hd, rfl⟩
  · exact Or.inr (Or.inl ⟨x, hl.subset hx, hd, hr⟩)
  · exact Or.inr (Or.inr ⟨k, hk.subset hk', boundary_mono hl hb, rfl⟩)

theorem certifiedKeys_take_mono (D : Digests) (c : Composite) (log : Registration) {n m : ℕ} (h : n ≤ m) :
    certifiedKeys (view D c (log.take n)) ⊆ certifiedKeys (view D c (log.take m)) := by
  obtain ⟨-, -, -, hl, hk⟩ := view_take_prefix D c log h
  rintro lk ⟨k, hk', hb, i, hi, x, hx, hc, rfl⟩
  exact ⟨k, hk.subset hk', boundary_mono hl hb, i, hi, x, loc?_mono hl hx, hc, rfl⟩

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

theorem declaredIn_take_mono (V : Vocab d α) (tags : Flock.Tags) (tables : List (String × ByteArray))
    (θ : ByteArray) (F : Array ByteArray) (j : ℕ) (D : Digests) (c : Composite) (log : Registration) {n m : ℕ}
    (h : n ≤ m) (hd : DeclaredIn V tags tables θ F j (view D c (log.take n))) :
    DeclaredIn V tags tables θ F j (view D c (log.take m)) := by
  obtain ⟨-, hdecl, -⟩ := view_take_prefix D c log h
  obtain ⟨p, hp, hn⟩ := hd
  refine ⟨p, hp, hn.mono fun e he => ?_⟩
  simp only [Set.mem_ofPred_eq] at he ⊢
  exact ⟨hdecl.subset he.1, he.2⟩

theorem certifiedIn_take_mono (p : NetworkCheck.Check.Params) (D : Digests) (c : Composite) (log : Registration)
    {n m : ℕ} (h : n ≤ m) (hc : CertifiedIn p (view D c (log.take n))) :
    CertifiedIn p (view D c (log.take m)) :=
  fun l hl => (hc l hl).mono (certifiedKeys_take_mono D c log h)

theorem rootIn_take_mono (t : Theta) (D : Digests) (c : Composite) (log : Registration) {n m : ℕ}
    (h : n ≤ m) (hr : RootIn t (view D c (log.take n))) : RootIn t (view D c (log.take m)) := by
  obtain ⟨hθ, -⟩ := view_take_prefix D c log h
  refine hr.mono fun x hx => ?_
  simp only [Set.mem_ofPred_eq] at hx ⊢
  exact ⟨hθ.subset hx.1, hx.2⟩

end Registration

end Verity
