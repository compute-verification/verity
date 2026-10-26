import Proofs.Properties.Nci.BodiesOk.Basic

/-!
# What `Prog.orderViolation` checks

`order_closed`: when `Prog.orderViolation` finds nothing, the Definitions its walk saw hold the root and every callee of
a composite among them, and each such composite's nodes name only parameters and earlier nodes (`Checked`). Its loops
stop at the first violation, so each level is a `for` loop whose state carries an optional verdict (`forIn_all`).
-/

namespace NciCheck.Properties.SecurityProofs.BodiesOk
open Flock.Program

/-- The references `Prog.orderViolation` checks of a part, by collection and index. -/
def partNames : Part → Array (Bool × ℕ)
  | .affine p idx _ c => if c > 0 then #[(p, idx)] else #[]
  | .strided p idx _ _ _ _ c => if c > 0 then #[(p, idx)] else #[]
  | .explicit it => it.map fun r => (r.p, r.i)

theorem get_mem_names (q : Part) (t : ℕ) (ht : t < q.count) : ((q.get t).p, (q.get t).i) ∈ partNames q := by
  cases q with
  | affine p i b c =>
    simp only [Part.count] at ht
    simp only [partNames, show c > 0 by omega, ite_true]
    exact Array.mem_singleton.2 rfl
  | strided p i b inner os is c =>
    simp only [Part.count] at ht
    simp only [partNames, show c > 0 by omega, ite_true]
    exact Array.mem_singleton.2 rfl
  | explicit it =>
    simp only [Part.count] at ht
    show (it[t]!.p, it[t]!.i) ∈ it.map fun r => (r.p, r.i)
    rw [getElem!_pos it t ht]
    exact Array.mem_map.2 ⟨it[t], Array.getElem_mem ht, rfl⟩

/-- A name node `k` of `body` may read: a parameter, or an earlier node. -/
def NameOk (body : Body) (k : ℕ) (nm : Bool × ℕ) : Prop :=
  (nm.1 = true → nm.2 < body.params.size) ∧ (nm.1 = false → nm.2 < k)

/-- Node `k` of `body` reads only parameters and earlier nodes. -/
def NodeChecked (body : Body) (k : ℕ) : Prop :=
  ∀ a < (body.nodes[k]!).args.size, ∀ q ∈ ((body.nodes[k]!).args[a]!).seq.parts, ∀ nm ∈ partNames q,
    NameOk body k nm

/-- Definition `d`'s nodes read only parameters and earlier nodes, as `Prog.orderViolation` checks. -/
def Checked (P : Prog) (d : ℕ) : Prop := ∀ k < (P.defs[d]!).body.nodes.size, NodeChecked (P.defs[d]!).body k

/-- A `for` loop in `Id` whose state carries an optional verdict, `none` while it runs: it ends with `none` and every
element passed `Q`, or with the verdict of the element that stopped it. -/
theorem forIn_all {α γ δ : Type} (f : α → Option γ × δ → Id (ForInStep (Option γ × δ))) (Q : α → Prop)
    (E : γ → Prop)
    (hf : ∀ x b, b.1 = none → match (f x b).run with
      | .yield b' => b'.1 = none ∧ Q x | .done b' => ∃ g, b'.1 = some g ∧ E g)
    (l : List α) (b : Option γ × δ) (hb : b.1 = none) :
    ((forIn l b f).run.1 = none ∧ ∀ x ∈ l, Q x) ∨ ∃ g, (forIn l b f).run.1 = some g ∧ E g := by
  have := forIn_id_inv f (fun b => ∃ g, b.1 = some g ∧ E g) l
    (fun i b => b.1 = none ∧ ∀ j (hj : j < i) (hj' : j < l.length), Q l[j]) b
    ⟨hb, fun j hj => absurd hj (Nat.not_lt_zero _)⟩ (fun i hi b hI => by
      have h1 := hf l[i] b hI.1
      rcases hfx : (f l[i] b).run with b' | b'
      all_goals rw [hfx] at h1
      · exact h1
      · refine ⟨h1.1, fun j hj hj' => ?_⟩
        rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
        · exact hI.2 j hj hj'
        · exact h1.2)
  rcases this with ⟨h1, h2⟩ | h
  · refine Or.inl ⟨h1, fun x hx => ?_⟩
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
    exact h2 j hj hj
  · exact Or.inr h

/-- Definition `d` is done once `todo` holds its callees: its nodes are checked and their callees queued. -/
def Done (P : Prog) (todo : Array ℕ) (d : ℕ) : Prop :=
  (P.defs[d]!).prim = false → Checked P d ∧
    ∀ k < (P.defs[d]!).body.nodes.size, ∃ j < todo.size, todo[j]! = ((P.defs[d]!).body.nodes[k]!).fn

/-- `b` extends `a`. -/
def Ext (a b : Array ℕ) : Prop := a.size ≤ b.size ∧ ∀ j < a.size, b[j]! = a[j]!

theorem ext_refl (a : Array ℕ) : Ext a a := ⟨le_refl _, fun _ _ => rfl⟩

theorem ext_push (a : Array ℕ) (x : ℕ) : Ext a (a.push x) :=
  ⟨by simp, fun _ hj => getElem!_push_lt _ _ hj⟩

theorem ext_trans {a b c : Array ℕ} (h1 : Ext a b) (h2 : Ext b c) : Ext a c :=
  ⟨le_trans h1.1 h2.1, fun j hj => by rw [h2.2 j (by have := h1.1; omega), h1.2 j hj]⟩

theorem done_mono {P : Prog} {a b : Array ℕ} {d : ℕ} (he : Ext a b) (h : Done P a d) : Done P b d := fun hp => by
  obtain ⟨h1, h2⟩ := h hp
  refine ⟨h1, fun k hk => ?_⟩
  obtain ⟨j, hj, e⟩ := h2 k hk
  exact ⟨j, by have := he.1; omega, by rw [he.2 j hj, e]⟩

/-- The walk's invariant: no verdict yet, the root first in `todo`, and every Definition seen done. -/
def OInv (P : Prog) (b : Option (Option Lean.Json) × Std.HashSet ℕ × Array ℕ × ℕ) : Prop :=
  b.1 = none ∧ 0 < b.2.2.1.size ∧ b.2.2.1[0]! = P.root ∧ b.2.2.2 ≤ b.2.2.1.size ∧
  (∀ j < b.2.2.2, b.2.1.contains b.2.2.1[j]! = true) ∧ (∀ d, b.2.1.contains d = true → Done P b.2.2.1 d)

/-- What `Prog.orderViolation` passing establishes: a set of Definitions holding the root and closed under calls, each
of whose nodes, if the Definition is not a primitive, reads only parameters and earlier nodes. -/
theorem order_closed (P : Prog) (h : P.orderViolation = none) :
    ∃ S : ℕ → Prop, S P.root ∧ ∀ d, S d → (P.defs[d]!).prim = false →
      Checked P d ∧ ∀ k < (P.defs[d]!).body.nodes.size, S ((P.defs[d]!).body.nodes[k]!).fn := by
  unfold Prog.orderViolation at h
  dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  generalize hR : forIn (m := Id) (List.range' _ _ _) _ _ = R at h
  have key : OInv P R.run ∨ ((OInv P R.run ∧ R.run.2.2.1.size ≤ R.run.2.2.2) ∨ ∃ j, R.run.1 = some (some j)) := by
    rw [← hR]
    refine forIn_id_inv _ (fun b => (OInv P b ∧ b.2.2.1.size ≤ b.2.2.2) ∨ ∃ j, b.1 = some (some j)) _
      (fun _ b => OInv P b) _ ?_ ?_
    · refine ⟨rfl, by simp, by simp, by simp, fun j hj => absurd hj (Nat.not_lt_zero _), fun d hd => by simp at hd⟩
    · rintro i hi ⟨r, seen, todo, at_⟩ hI
      obtain ⟨hr, hpos, h0, hle, hseen, hdone⟩ := hI
      dsimp only at hr hpos h0 hle hseen hdone ⊢
      subst hr
      by_cases hA : at_ ≥ todo.size
      · rw [ite_eq_left hA]; exact Or.inl ⟨⟨rfl, hpos, h0, hle, hseen, hdone⟩, hA⟩
      rw [ite_eq_right hA]
      have hlt : at_ < todo.size := by omega
      by_cases hB : seen.contains todo[at_]! = true
      · rw [ite_eq_left hB]
        refine ⟨rfl, hpos, h0, by dsimp only; omega, fun j hj => ?_, hdone⟩
        rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
        · exact hseen j hj
        · exact hB
      rw [ite_eq_right hB]
      have hins : ∀ j < at_ + 1, (seen.insert todo[at_]!).contains todo[j]! = true := by
        intro j hj
        rw [Std.HashSet.contains_insert]
        rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
        · simp [hseen j hj]
        · simp
      have hcont : ∀ d, (seen.insert todo[at_]!).contains d = true → d = todo[at_]! ∨ seen.contains d = true := by
        intro d hd
        rw [Std.HashSet.contains_insert] at hd
        rcases Bool.or_eq_true_iff.1 hd with hd | hd
        · have e : todo[at_]! = d := by simpa using hd
          exact Or.inl e.symm
        · exact Or.inr hd
      by_cases hC : P.defs[todo[at_]!]!.prim = true
      · rw [ite_eq_left hC]
        refine ⟨rfl, hpos, h0, by dsimp only; omega, hins, fun d hd => ?_⟩
        rcases hcont d hd with rfl | hd
        · intro hp; rw [hC] at hp; cases hp
        · exact hdone d hd
      rw [ite_eq_right hC]
      generalize hN : forIn (m := Id) (_ : Std.Legacy.Range) (none, todo) _ = N
      have hC' : (P.defs[todo[at_]!]!).prim = false := by simpa using hC
      generalize hd : todo[at_]! = d at hN hC' hins hcont ⊢
      generalize hbody : (P.defs[d]!).body = body at hN
      have hnode : (N.run.1 = none ∧ Ext todo N.run.2 ∧ (∀ k < body.nodes.size, NodeChecked body k) ∧
          ∀ k < body.nodes.size, ∃ j < N.run.2.size, N.run.2[j]! = (body.nodes[k]!).fn) ∨
          ∃ j, N.run.1 = some (some j) := by
        rw [← hN, Std.Legacy.Range.forIn_eq_forIn_range']
        refine (forIn_id_inv _ (fun b => ∃ j, b.1 = some (some j)) _
          (fun k (b : Option (Option Lean.Json) × Array ℕ) => b.1 = none ∧ Ext todo b.2 ∧
            ∀ k' < k, NodeChecked body k' ∧ ∃ j < b.2.size, b.2[j]! = (body.nodes[k']!).fn) (none, todo)
          ⟨rfl, ext_refl _, fun _ h => absurd h (Nat.not_lt_zero _)⟩ ?_).imp ?_ id
        · rintro k hk ⟨r, t⟩ ⟨hr, ht, hprev⟩
          dsimp only at hr ht hprev
          subst hr
          simp only [List.length_range', Std.Legacy.Range.size] at hk
          simp (config := { zeta := false }) only [List.getElem_range', Nat.one_mul, Nat.zero_add]
          generalize hA : forIn (m := Id) (_ : Std.Legacy.Range) (none, ()) _ = A
          have hk' : k < body.nodes.size := by simpa [Std.Legacy.Range.size] using hk
          have hAf : (A.run.1 = none ∧ NodeChecked body k) ∨ ∃ g, A.run.1 = some g ∧ ∃ j, g = some j := by
            rw [← hA, Std.Legacy.Range.forIn_eq_forIn_range']
            refine (forIn_all _ (fun a => ∀ q ∈ ((body.nodes[k]!).args[a]!).seq.parts, ∀ nm ∈ partNames q,
                NameOk body k nm) (fun g => ∃ j, g = some j) ?_ _ _ rfl).imp
              (fun ⟨h1, h2⟩ => ⟨h1, fun a ha => h2 a (by simpa [Std.Legacy.Range.size] using ha)⟩) id
            rintro a ⟨r, u⟩ hr
            dsimp only at hr; subst hr
            generalize hQ : forIn (m := Id) (_ : Array Part) (none, ()) _ = Q
            have hQf : (Q.run.1 = none ∧ ∀ q ∈ ((body.nodes[k]!).args[a]!).seq.parts, ∀ nm ∈ partNames q,
                NameOk body k nm) ∨ ∃ g, Q.run.1 = some g ∧ ∃ j, g = some j := by
              rw [← hQ, ← Array.forIn_toList]
              refine (forIn_all _ (fun q => ∀ nm ∈ partNames q, NameOk body k nm) (fun g => ∃ j, g = some j)
                ?_ _ _ rfl).imp (fun ⟨h1, h2⟩ => ⟨h1, fun q hq => h2 q (by simpa using hq)⟩) id
              rintro q ⟨r, u⟩ hr
              dsimp only at hr; subst hr
              generalize hM : forIn (m := Id) (_ : Array (Bool × ℕ)) (none, ()) _ = M
              have hMf : (M.run.1 = none ∧ ∀ nm ∈ partNames q, NameOk body k nm) ∨
                  ∃ g, M.run.1 = some g ∧ ∃ j, g = some j := by
                rw [← hM, ← Array.forIn_toList]
                refine (forIn_all _ (fun nm => NameOk body k nm) (fun g => ∃ j, g = some j)
                  ?_ _ _ rfl).imp (fun ⟨h1, h2⟩ => ⟨h1, fun nm hnm => h2 nm (Array.mem_def.1 hnm)⟩) id
                rintro ⟨_ | _, n⟩ ⟨r, u⟩ hr
                all_goals dsimp only at hr; subst hr
                · by_cases hn : n < k <;> simp [hn, NameOk]
                · by_cases hn : n < body.params.size <;> simp [hn, NameOk]
              rcases hMf with ⟨h1, h2⟩ | ⟨g, h1, h2⟩
              all_goals simp only [Id.run_bind, h1, Id.run_pure]
              · exact ⟨trivial, h2⟩
              · exact ⟨g, rfl, h2⟩
            rcases hQf with ⟨h1, h2⟩ | ⟨g, h1, h2⟩
            all_goals simp only [Id.run_bind, h1, Id.run_pure]
            · exact ⟨trivial, h2⟩
            · exact ⟨g, rfl, h2⟩
          rcases hAf with ⟨h1, h2⟩ | ⟨g, h1, h2⟩
          all_goals simp only [Id.run_bind, h1, Id.run_pure]
          · refine ⟨trivial, ext_trans ht (ext_push _ _), fun k' hk'' => ?_⟩
            rcases Nat.lt_succ_iff_lt_or_eq.1 hk'' with hk'' | rfl
            · obtain ⟨c, j, hj, e⟩ := hprev k' hk''
              exact ⟨c, j, by simp; omega, by rw [getElem!_push_lt _ _ hj, e]⟩
            · exact ⟨h2, t.size, by simp, getElem!_push_eq _ _⟩
          · obtain ⟨j, rfl⟩ := h2; exact ⟨j, rfl⟩
        · rintro ⟨h1, h2, h3⟩
          simp only [List.length_range', Std.Legacy.Range.size] at h3
          exact ⟨h1, h2, fun k hk => (h3 k (by simpa using hk)).1, fun k hk => (h3 k (by simpa using hk)).2⟩
      rcases hnode with ⟨h1, hext, hchk, hfns⟩ | ⟨j, h1⟩
      all_goals simp only [Id.run_bind, h1, Id.run_pure]
      · refine ⟨rfl, lt_of_lt_of_le hpos hext.1, by rw [hext.2 0 hpos, h0], by dsimp only; have := hext.1; omega,
          fun j hj => ?_, fun d' hd' => ?_⟩
        · dsimp only at hj ⊢; rw [hext.2 j (by omega)]; exact hins j hj
        · rcases hcont d' hd' with rfl | hd'
          · intro _
            refine ⟨?_, ?_⟩
            · unfold Checked; rw [hbody]; exact hchk
            · rw [hbody]; exact hfns
          · exact done_mono hext (hdone d' hd')
      · exact Or.inr ⟨j, rfl⟩
  have hI : OInv P R.run := by
    rcases key with hk | ⟨hk, -⟩ | ⟨j, hj⟩
    · exact hk
    · exact hk
    · simp only [Id.run_bind, hj, Id.run_pure] at h
      cases h
  have hge : R.run.2.2.1.size ≤ R.run.2.2.2 := by
    simp only [Id.run_bind, hI.1] at h
    split at h
    · cases h
    · omega
  obtain ⟨-, hpos, h0, hle, hseen, hdone⟩ := hI
  refine ⟨fun d => R.run.2.1.contains d = true, ?_, fun d hd hp => ?_⟩
  · have := hseen 0 (by omega); rwa [h0] at this
  · obtain ⟨hc, hf⟩ := hdone d hd hp
    refine ⟨hc, fun k hk => ?_⟩
    obtain ⟨j, hj, e⟩ := hf k hk
    have := hseen j (by omega); rwa [e] at this

end NciCheck.Properties.SecurityProofs.BodiesOk
