module

public import Properties.Ties
public import Proofs.Core.Site

/-! No registration entry retires anything, so what a prefix of the log registers, every longer prefix registers. -/

@[expose] public section

namespace Verity

namespace Registration

theorem NamesRegistered.mono {X Y : Type} {holds holds' : Set X} (h : holds ⊆ holds') {f : X → Y} {y : Y}
    (hn : NamesRegistered holds f y) : NamesRegistered holds' f y := by
  obtain ⟨x, hx, rfl⟩ := hn
  exact ⟨x, h hx, rfl⟩

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

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem declaredIn_take_mono (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (mp : MonitoringPeriod) {k : ℕ}
    (cut : mp.span.Cut k) (w : TickSpan) (D : Digests) (c : Composite) (log : Registration) {n m : ℕ}
    (h : n ≤ m) (hd : DeclaredIn V θ F mp cut w (view D c (log.take n))) :
    DeclaredIn V θ F mp cut w (view D c (log.take m)) := by
  obtain ⟨-, hdecl, -⟩ := view_take_prefix D c log h
  obtain ⟨j, hw, hd⟩ := hd
  refine ⟨j, hw, fun r hr => ?_⟩
  obtain ⟨p, hp, hn⟩ := hd r hr
  refine ⟨p, hp, hn.mono fun e he => ?_⟩
  simp only [Set.mem_ofPred_eq] at he ⊢
  exact ⟨hdecl.subset he.1, he.2⟩

theorem rootIn_take_mono (t : Theta) (D : Digests) (c : Composite) (log : Registration) {n m : ℕ}
    (h : n ≤ m) (hr : RootIn t (view D c (log.take n))) : RootIn t (view D c (log.take m)) := by
  obtain ⟨hθ, -⟩ := view_take_prefix D c log h
  refine hr.mono fun x hx => ?_
  simp only [Set.mem_ofPred_eq] at hx ⊢
  exact ⟨hθ.subset hx.1, hx.2⟩

theorem keyIn_take_mono (k : String) (D : Digests) (c : Composite) (log : Registration) {n m : ℕ}
    (h : n ≤ m) (hk : KeyIn k (view D c (log.take n))) : KeyIn k (view D c (log.take m)) := by
  obtain ⟨-, -, -, hl, -⟩ := view_take_prefix D c log h
  refine hk.mono fun x hx => ?_
  simp only [Set.mem_ofPred_eq] at hx ⊢
  exact ⟨hl.subset hx.1, hx.2⟩

end Registration

end Verity
