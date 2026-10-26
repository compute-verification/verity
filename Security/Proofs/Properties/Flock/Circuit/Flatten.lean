import Properties.Flock.Circuit.Flatten
import Proofs.Flock.Claims.Sampling.Deployed

/-!
# A statement read by `parseOf` is its descriptor flattened

* `flatOf_flattens`: `FlatOfFlattens`. `flatOf` locates every gate of the root and reads it with `gateOf?`, whose gate
  is an input exactly at an input id and reads exactly its operands (`gateOf?_spec`), and returns every resolved return.
* `parseOf_flattens`: `ParseOfFlattens`, from `flatOf_flattens` at the descriptor `parseOf` decoded.
* `SamplingDeploy.stmtFlattens_deploy`: `StmtFlattens` at the sampled verifier's deployment whose code reads its
  statements with `parseOf rd` (`SamplingDeploy.withParse`).
-/

namespace Properties.Flock.Circuit

open Properties.NetworkScan

theorem forall₂_of_mapM {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {r : List β}, l.mapM f = some r → List.Forall₂ (fun a b => f a = some b) l r
  | [], r, h => by
    simp at h
    subst h
    exact .nil
  | a :: l, r, h => by
    rw [List.mapM_cons] at h
    cases hb : f a with
    | none => simp [hb] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [hb, hl] at h
      | some bs =>
        simp [hb, hl] at h
        subst h
        exact .cons hb (forall₂_of_mapM hl)

theorem forall₂_getElem? {α β : Type} {R : α → β → Prop} :
    ∀ {l : List α} {r : List β}, List.Forall₂ R l r → ∀ i a, l[i]? = some a → ∃ b, r[i]? = some b ∧ R a b
  | _, _, .nil, i, a, h => by simp at h
  | _, _, .cons hab _, 0, a, h => by
    simp at h
    subst h
    exact ⟨_, List.getElem?_cons_zero, hab⟩
  | _, _, .cons _ hlr, i + 1, a, h => by simpa using forall₂_getElem? hlr i a (by simpa using h)

theorem forall₂_getElem?' {α β : Type} {R : α → β → Prop} :
    ∀ {l : List α} {r : List β}, List.Forall₂ R l r → ∀ i b, r[i]? = some b → ∃ a, l[i]? = some a ∧ R a b
  | _, _, .nil, i, b, h => by simp at h
  | _, _, .cons hab _, 0, b, h => by
    simp at h
    subst h
    exact ⟨_, List.getElem?_cons_zero, hab⟩
  | _, _, .cons _ hlr, i + 1, b, h => by simpa using forall₂_getElem?' hlr i b (by simpa using h)

/-- **`gateOf?`'s gate** is an input exactly at an input id, and reads exactly its operands. -/
theorem gateOf?_spec {key : String} {ops : Array ℕ} {b : Claims.Sampling.BGate} (h : gateOf? key ops = some b) :
    (b = .input ↔ _root_.Flock.Program.isInputId key = true) ∧ ∀ a, a ∈ b.reads ↔ a ∈ ops := by
  unfold gateOf? at h
  by_cases hi : _root_.Flock.Program.isInputId key = true
  · rw [if_pos hi] at h
    split_ifs at h with he <;> cases h
    subst he
    simp [hi, Claims.Sampling.BGate.reads]
  · rw [if_neg hi] at h
    have hr : b ≠ .input ∧ ∀ a, a ∈ b.reads ↔ a ∈ ops := by
      split_ifs at h <;> (try split at h) <;> (try split_ifs at h) <;> cases h <;>
        simp_all [Claims.Sampling.BGate.reads]
    exact ⟨⟨fun e => absurd e hr.1, fun e => absurd e hi⟩, hr.2⟩

theorem flatOf_flattens : FlatOfFlattens := by
  intro P C h
  unfold flatOf at h
  split at h
  · cases h
  rename_i gates hg
  split at h
  · cases h
  rename_i outs ho
  cases h
  have Fg := forall₂_of_mapM hg
  have Fo := forall₂_of_mapM ho
  have hlen : gates.length = (P.defs[P.root]!).gates := by simpa using Fg.length_eq.symm
  unfold Flattens
  refine ⟨hlen, ?_, ?_⟩
  · intro g hgl
    have hgn : g < (P.defs[P.root]!).gates := by rw [← hlen]; exact hgl
    obtain ⟨b, hb, hfb⟩ := forall₂_getElem? Fg g g (by simp [hgn])
    unfold flatGate at hfb
    split at hfb
    · rename_i p ops sc hgate
      obtain ⟨h1, h2⟩ := gateOf?_spec hfb
      refine ⟨p, ops, sc, hgate, ?_, ?_⟩
      · show gates[g]? = some .input ↔ _
        rw [hb]
        exact ⟨fun e => h1.1 (Option.some.inj e), fun e => congrArg some (h1.2 e)⟩
      · intro a
        show a ∈ (gates[g]?.map Claims.Sampling.BGate.reads).getD [] ↔ a ∈ ops
        rw [hb]
        exact h2 a
    · cases hfb
  · intro g
    show g ∈ outs ↔ _
    constructor
    · intro hmem
      obtain ⟨i, hi⟩ := List.mem_iff_getElem?.1 hmem
      obtain ⟨j, hj, hr⟩ := forall₂_getElem?' Fo i g hi
      simp only [List.getElem?_range] at hj
      split_ifs at hj with him <;> cases hj
      refine ⟨i, him, ?_⟩
      unfold flatOut at hr
      cases hres : _root_.Flock.Program.resolve P P.rootScope ((P.defs[P.root]!).body.ret.get i) with
      | error e => rw [hres] at hr; cases hr
      | ok g' => rw [hres] at hr; cases hr; rfl
    · rintro ⟨j, hj, hres⟩
      obtain ⟨b, hb, hfb⟩ := forall₂_getElem? Fo j j (by simp [hj])
      unfold flatOut at hfb
      rw [hres] at hfb
      cases hfb
      exact List.mem_of_getElem? hb

theorem parseOf_flattens : ParseOfFlattens := by
  intro Coin Col T D rd hD x s hx
  rw [hD] at hx
  unfold parseOf at hx
  split at hx
  · cases hx
  split at hx
  · rename_i P hP
    cases hC : flatOf P with
    | none => rw [hC] at hx; cases hx
    | some C =>
      rw [hC] at hx
      cases hx
      exact ⟨P, hP, flatOf_flattens P C hC⟩
  · cases hx

end Properties.Flock.Circuit

namespace SamplingDeploy

/-- **The sampled verifier's code reading its statements with `parseOf rd`**: `c` with its parser the framing `rd`'s
statement, its descriptor flattened as its gates. -/
def withParse (c : Code)
    (rd : Claims.Msg → Option (Claims.Msg × Claims.Sampling.Query × Claims.Sampling.Design × List Claims.Msg)) :
    Code :=
  { c with parse := Properties.Flock.Circuit.parseOf rd }

/-- **`StmtFlattens` at the sampled verifier's deployment** whose code reads its statements with `parseOf rd`. -/
theorem stmtFlattens_deploy (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)
    (rd : Claims.Msg → Option (Claims.Msg × Claims.Sampling.Query × Claims.Sampling.Design × List Claims.Msg)) :
    Properties.NetworkScan.StmtFlattens (deploy rel (withParse c rd)) :=
  Properties.Flock.Circuit.parseOf_flattens _ rd rfl

end SamplingDeploy
