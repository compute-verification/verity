import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Verifier.Registered
import Proofs.Properties.NetworkCertifier.Bytes

/-!
# The registered path check binds (frame-v3-sha512)

The claims read the deployment's path check `D.path root pos leaf π`, the registered scheme's (frame-v3-sha512). Here it
is as an oracle program that queries the code's own frames in the code's order (`Flock.Registered`):

* `climbOC`: from a digest at index `idx` of level `d`, each sibling's node frame (`nodeIn`) queried, the left child
  first at an even index, ending true when the last digest is the root.
* `regPathAt reg`: at the registered port `reg`, a position among its leaves, one 64-byte sibling per level of
  `log2ceil leaves`, the leaf's frame (`leafInAs`, the port's leaf schema at the position) queried, then the climb.
* `regPath own`: the deployed check, at the verifier's own port whose root is `root`, false at any other root.
* `regPath_binds`: **`PathBinds H (regPath own)` for every `H`.** Two different leaves accepted under one root at one
  position query two different frames; where their climbs first reach one digest, the two queries there are different
  frames with one answer. A frame-v3 frame gives each part's length, and every node has a 64-byte sibling child, so
  equal node frames have equal children of whatever length `H` answers.
* `regPath_sha512`: at `H` = SHA-512 (`sha512Msg`), the program accepts exactly when the verifier's own
  `Flock.Registered.opensLeaf` does at that port.
* `pathCommit`, `pathCommit_binds`: the trace's commitment by leaves (a leaf that refuses fails the opening) and a path
  check binds (`Commitment.Binds`) when its leaf binds the value (`LeafBinds`) and its path check binds the leaf
  (`PathBinds`). `Binding_of_regPath`: the claims' `Binding` for a commitment whose path check is the registered one,
  from its leaf's binding; at the hm96 leaf, `Sampling.Leaf`.
-/

namespace Claims.Sampling

open _root_.Flock _root_.Flock.Registered
open Properties.NetworkCertifier.Bytes (bs bs_inj be64_size)

/-- A message as a byte string. -/
def bytes (m : Msg) : ByteArray := ⟨m.toArray⟩

theorem bs_bytes (m : Msg) : bs (bytes m) = m := by simp [bs, bytes]

theorem bytes_bs (b : ByteArray) : bytes (bs b) = b := bs_inj (bs_bytes _)

theorem bytes_inj {m m' : Msg} (h : bytes m = bytes m') : m = m' := by
  rw [← bs_bytes m, h, bs_bytes]

theorem size_bytes (m : Msg) : (bytes m).size = m.length := by simp [bytes, ByteArray.size]

/-! ## Frames of one shape with equal bytes -/

theorem nodeIn_size (dom : ByteArray) (d idx : ℕ) (l r l' r' : ByteArray) :
    (nodeIn dom d idx l r).size + l'.size + r'.size = (nodeIn dom d idx l' r').size + l.size + r.size := by
  simp only [nodeIn, frame, List.foldl, step_size, be64_size]
  omega

theorem leafIn_size (schema : String) (dom : ByteArray) (pos : ℕ) (x y : ByteArray) :
    (leafInAs schema dom pos x).size + y.size = (leafInAs schema dom pos y).size + x.size := by
  simp only [leafInAs, frame, List.foldl, step_size, be64_size]
  omega

/-- Equal node frames with right children of one length have the same children. -/
theorem nodeIn_eq_of_right {dom : ByteArray} {d idx : ℕ} {l r l' r' : ByteArray} (hr : r.size = r'.size)
    (h : nodeIn dom d idx l r = nodeIn dom d idx l' r') : l = l' ∧ r = r' := by
  have hs := congrArg ByteArray.size h
  have := nodeIn_size dom d idx l r l' r'
  exact nodeIn_inj (by omega) hr h

/-- Equal node frames with left children of one length have the same children. -/
theorem nodeIn_eq_of_left {dom : ByteArray} {d idx : ℕ} {l r l' r' : ByteArray} (hl : l.size = l'.size)
    (h : nodeIn dom d idx l r = nodeIn dom d idx l' r') : l = l' ∧ r = r' := by
  have hs := congrArg ByteArray.size h
  have := nodeIn_size dom d idx l r l' r'
  exact nodeIn_inj hl (by omega) h

/-- Equal leaf frames at one position and schema have the same payload. -/
theorem leafIn_eq {schema : String} {dom : ByteArray} {pos : ℕ} {x y : ByteArray}
    (h : leafInAs schema dom pos x = leafInAs schema dom pos y) : x = y := by
  have hs := congrArg ByteArray.size h
  have := leafIn_size schema dom pos x y
  exact leafInAs_inj (by omega) h

/-! ## The check -/

/-- The climb from the digest `cur` at index `idx` of level `d` through the siblings `sibs`, lowest first: each node's
frame queried, the left child first at an even index, ending true when the last digest is `root`. -/
def climbOC (dom : ByteArray) (root : Msg) : Msg → ℕ → ℕ → List ByteArray → OC Bool
  | cur, _, _, [] => .ret (decide (cur = root))
  | cur, idx, d, sib :: rest =>
    .query (bs (if idx % 2 = 0 then nodeIn dom d (idx / 2) (bytes cur) sib else nodeIn dom d (idx / 2) sib (bytes cur)))
      fun h => climbOC dom root h (idx / 2) (d + 1) rest

/-- **The registered path check at port `reg`**: `leaf` (the row's hm96 leaf of the port's schema) at `pos` under
`root` through `π`: a position among the port's leaves, one 64-byte sibling per level, the leaf's frame-v3 frame
queried, then the climb to `root` (`Flock.Registered.opensLeaf`, its hash the oracle). -/
def regPathAt (reg : Registered.Port) (root : Msg) (pos : ℕ) (leaf π : Msg) : OC Bool :=
  if pos < reg.leaves ∧ π.length = 64 * log2ceil reg.leaves then
    .query (bs (leafInAs reg.schema reg.domain pos (bytes leaf))) fun l =>
      climbOC reg.domain root l pos 0 (sibs (bytes π) (log2ceil reg.leaves))
  else .ret false

/-- **The deployed path check**: at the verifier's own registered port whose root is `root`; no other root opens. -/
def regPath (own : Array Registered.Port) (root : Msg) (pos : ℕ) (leaf π : Msg) : OC Bool :=
  match own.find? fun p => decide (bs p.root = root) with
  | some reg => regPathAt reg root pos leaf π
  | none => .ret false

/-! ## Binding, for every `H` -/

/-- **A path check binds its leaf**: two accepted checks at one root and position, of different leaves, make `H` calls
that collide. What a Merkle path check gives: where the two walks to the root first meet, one hash has two different
inputs. -/
def PathBinds (H : Msg → Msg) (path : Msg → ℕ → Msg → Msg → OC Bool) : Prop :=
  ∀ root pos leaf leaf' π π', leaf ≠ leaf' → ((path root pos leaf π).eval H).1 = true →
    ((path root pos leaf' π').eval H).1 = true →
      LogColl (((path root pos leaf π).eval H).2 ++ ((path root pos leaf' π').eval H).2)

/-- Two climbs from different digests to one root, through 64-byte siblings of one count, make two different queries
with one answer. -/
theorem climbOC_binds (H : Msg → Msg) (dom : ByteArray) (root : Msg) :
    ∀ (s₁ s₂ : List ByteArray) (a b : Msg) (idx d : ℕ), s₁.length = s₂.length → (∀ x ∈ s₁, x.size = 64) →
      (∀ x ∈ s₂, x.size = 64) → a ≠ b → ((climbOC dom root a idx d s₁).eval H).1 = true →
        ((climbOC dom root b idx d s₂).eval H).1 = true →
          LogColl (((climbOC dom root a idx d s₁).eval H).2 ++ ((climbOC dom root b idx d s₂).eval H).2) := by
  intro s₁
  induction s₁ with
  | nil =>
    intro s₂ a b idx d hl _ _ hab h₁ h₂
    cases s₂ with
    | nil =>
      simp only [climbOC, OC.eval, decide_eq_true_eq] at h₁ h₂
      exact absurd (h₁.trans h₂.symm) hab
    | cons _ _ => simp at hl
  | cons s rest ih =>
    intro s₂ a b idx d hl hw₁ hw₂ hab h₁ h₂
    cases s₂ with
    | nil => simp at hl
    | cons t rest₂ =>
      have hst : s.size = t.size := by rw [hw₁ s (by simp), hw₂ t (by simp)]
      set x := bs (if idx % 2 = 0 then nodeIn dom d (idx / 2) (bytes a) s else nodeIn dom d (idx / 2) s (bytes a))
        with hx
      set y := bs (if idx % 2 = 0 then nodeIn dom d (idx / 2) (bytes b) t else nodeIn dom d (idx / 2) t (bytes b))
        with hy
      have hxy : x ≠ y := by
        intro e
        have e' := bs_inj e
        split_ifs at e' with hp
        · exact hab (bytes_inj (nodeIn_eq_of_right hst e').1)
        · exact hab (bytes_inj (nodeIn_eq_of_left hst e').2)
      have e₁ : climbOC dom root a idx d (s :: rest) = .query x fun h => climbOC dom root h (idx / 2) (d + 1) rest :=
        rfl
      have e₂ : climbOC dom root b idx d (t :: rest₂) = .query y fun h => climbOC dom root h (idx / 2) (d + 1) rest₂ :=
        rfl
      rw [e₁] at h₁
      rw [e₂] at h₂
      rw [e₁, e₂]
      simp only [OC.eval] at h₁ h₂ ⊢
      by_cases he : H x = H y
      · exact ⟨(x, H x), List.mem_append_left _ (List.mem_cons_self ..), (y, H y),
          List.mem_append_right _ (List.mem_cons_self ..), hxy, he⟩
      · refine (ih rest₂ (H x) (H y) (idx / 2) (d + 1) (by simpa using hl) (fun z hz => hw₁ z (by simp [hz]))
          (fun z hz => hw₂ z (by simp [hz])) he h₁ h₂).mono fun q hq => ?_
        rcases List.mem_append.1 hq with hq | hq
        · exact List.mem_append_left _ (List.mem_cons_of_mem _ hq)
        · exact List.mem_append_right _ (List.mem_cons_of_mem _ hq)

/-- At one port, two different leaves accepted at one position under one root make two different queries with one
answer. -/
theorem regPathAt_binds (H : Msg → Msg) (reg : Registered.Port) (root : Msg) (pos : ℕ) (leaf leaf' π π' : Msg)
    (hne : leaf ≠ leaf') (h₁ : ((regPathAt reg root pos leaf π).eval H).1 = true)
    (h₂ : ((regPathAt reg root pos leaf' π').eval H).1 = true) :
    LogColl (((regPathAt reg root pos leaf π).eval H).2 ++ ((regPathAt reg root pos leaf' π').eval H).2) := by
  unfold regPathAt at h₁ h₂ ⊢
  split_ifs at h₁ h₂ ⊢ with c₁ c₂
  · simp only [OC.eval] at h₁ h₂ ⊢
    set q := bs (leafInAs reg.schema reg.domain pos (bytes leaf)) with hq
    set q' := bs (leafInAs reg.schema reg.domain pos (bytes leaf')) with hq'
    have hqq : q ≠ q' := fun e => hne (bytes_inj (leafIn_eq (bs_inj e)))
    by_cases he : H q = H q'
    · exact ⟨(q, H q), List.mem_append_left _ (List.mem_cons_self ..), (q', H q'),
        List.mem_append_right _ (List.mem_cons_self ..), hqq, he⟩
    · have hsz : ∀ {p : Msg}, p.length = 64 * log2ceil reg.leaves → ∀ z ∈ sibs (bytes p) (log2ceil reg.leaves),
          z.size = 64 := fun hp => sibs_size (by rw [size_bytes, hp])
      refine (climbOC_binds H reg.domain root _ _ (H q) (H q') pos 0 (by rw [sibs_length, sibs_length])
        (hsz c₁.2) (hsz c₂.2) he h₁ h₂).mono fun z hz => ?_
      rcases List.mem_append.1 hz with hz | hz
      · exact List.mem_append_left _ (List.mem_cons_of_mem _ hz)
      · exact List.mem_append_right _ (List.mem_cons_of_mem _ hz)
  all_goals simp [OC.eval] at h₁ h₂

/-- **The registered path check binds, for every `H`**: two different leaves accepted under one root at one position
make two different logged queries with one answer. -/
theorem regPath_binds (H : Msg → Msg) (own : Array Registered.Port) : PathBinds H (regPath own) := by
  intro root pos leaf leaf' π π' hne h₁ h₂
  unfold regPath at h₁ h₂ ⊢
  cases hf : own.find? fun p => decide (bs p.root = root) with
  | none => simp [hf, OC.eval] at h₁
  | some reg =>
    simp only [hf] at h₁ h₂ ⊢
    exact regPathAt_binds H reg root pos leaf leaf' π π' hne h₁ h₂

/-! ## At SHA-512, the verifier's own check -/

/-- SHA-512 on messages. -/
def sha512Msg (m : Msg) : Msg := bs (Sha512.hash (bytes m))

theorem climbOC_sha512 (dom : ByteArray) (root : Msg) :
    ∀ (sibs : List ByteArray) (cur : Msg) (idx d : ℕ),
      ((climbOC dom root cur idx d sibs).eval sha512Msg).1 =
        decide (bs (climbL dom (bytes cur) idx d sibs) = root) := by
  intro sibs
  induction sibs with
  | nil => intro cur idx d; simp [climbOC, OC.eval, climbL, bs_bytes]
  | cons sib rest ih =>
    intro cur idx d
    have e : bytes (sha512Msg (bs (if idx % 2 = 0 then nodeIn dom d (idx / 2) (bytes cur) sib
        else nodeIn dom d (idx / 2) sib (bytes cur)))) =
        if idx % 2 = 0 then node dom d (idx / 2) (bytes cur) sib else node dom d (idx / 2) sib (bytes cur) := by
      simp only [sha512Msg, bytes_bs]
      split_ifs <;> exact (node_eq _ _ _ _ _).symm
    simp only [climbOC, OC.eval, climbL]
    rw [ih, e]
    rfl

/-- **At SHA-512 the program is the verifier's check**: it accepts exactly when the verifier's own port whose root is
`root` opens `leaf` at `pos` through `π` (`Flock.Registered.opensLeaf`). -/
theorem regPath_sha512 (own : Array Registered.Port) (root : Msg) (pos : ℕ) (leaf π : Msg) :
    ((regPath own root pos leaf π).eval sha512Msg).1 = true ↔
      ∃ reg, own.find? (fun p => decide (bs p.root = root)) = some reg ∧
        opensLeaf reg (bytes leaf) pos (bytes π) = true := by
  unfold regPath
  cases hf : own.find? fun p => decide (bs p.root = root) with
  | none => simp [OC.eval]
  | some reg =>
    have hroot : bs reg.root = root := by simpa using Array.find?_some hf
    have hiff : ((regPathAt reg root pos leaf π).eval sha512Msg).1 = true ↔
        opensLeaf reg (bytes leaf) pos (bytes π) = true := by
      unfold regPathAt opensLeaf
      simp only [Bool.and_eq_true, decide_eq_true_eq, size_bytes]
      split_ifs with c
      · simp only [OC.eval]
        rw [climbOC_sha512]
        have e : bytes (sha512Msg (bs (leafInAs reg.schema reg.domain pos (bytes leaf)))) =
            leafAs reg.schema reg.domain pos (bytes leaf) := by
          simp only [sha512Msg, bytes_bs]
          exact (leafAs_eq _ _ _ _).symm
        rw [e, decide_eq_true_eq]
        constructor
        · intro h
          rw [← hroot] at h
          exact ⟨c, bs_inj h⟩
        · intro h
          rw [← hroot]
          exact congrArg bs h.2
      · simp only [OC.eval, Bool.false_eq_true, false_iff]
        exact fun h => c h.1
    exact hiff.trans ⟨fun h => ⟨reg, rfl, h⟩, fun ⟨r, hr, h⟩ => by cases hr; exact h⟩

/-! ## The trace's commitment by leaves and a path check -/

/-- An oracle program, then another on its result. -/
def ocBind {α β : Type} : OC α → (α → OC β) → OC β
  | .ret a, f => f a
  | .query q k, f => .query q fun h => ocBind (k h) f

theorem ocBind_eval {α β : Type} (H : Msg → Msg) (p : OC α) (f : α → OC β) :
    (ocBind p f).eval H = (((f (p.eval H).1).eval H).1, (p.eval H).2 ++ ((f (p.eval H).1).eval H).2) := by
  induction p with
  | ret a => simp [ocBind, OC.eval]
  | query q k ih => simp [ocBind, OC.eval, ih]

/-- **A commitment by leaves and a path check**: an opening at position `pos` is checked under the root and at the leaf
index that the roots and `pos` name (`at_`); its leaf is computed from its value and data (`leaf`, its own hashing,
`none` when it refuses), then checked there through the path its data carries (`path`, `π`). -/
def pathCommit (at_ : List Msg → ℕ → Option (Msg × ℕ)) (leaf : Opening Bool → OC (Option Msg))
    (π : Opening Bool → Msg) (path : Msg → ℕ → Msg → Msg → OC Bool)
    (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg) : Commitment Bool where
  opens roots o :=
    match at_ roots o.pos with
    | some ri => ocBind (leaf o) fun l => match l with
      | some l => path ri.1 ri.2 l (π o)
      | none => .ret false
    | none => .ret false
  layer := layer

/-- **The leaf binds the value**: two openings at one position, of different values, with one leaf make `H` calls that
collide (the hm96 leaf: `Sampling.Leaf`). -/
def LeafBinds (H : Msg → Msg) (leaf : Opening Bool → OC (Option Msg)) : Prop :=
  ∀ (o o' : Opening Bool) (l : Msg), o.pos = o'.pos → o.val ≠ o'.val → ((leaf o).eval H).1 = some l →
    ((leaf o').eval H).1 = some l → LogColl (((leaf o).eval H).2 ++ ((leaf o').eval H).2)

/-- **The commitment by leaves and a path check binds** when its leaf binds the value and its path check the leaf:
where the leaves agree the leaves' calls collide, and otherwise the path checks'. -/
theorem pathCommit_binds {H : Msg → Msg} {at_ : List Msg → ℕ → Option (Msg × ℕ)}
    {leaf : Opening Bool → OC (Option Msg)} {π : Opening Bool → Msg} {path : Msg → ℕ → Msg → Msg → OC Bool}
    {layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg} (hleaf : LeafBinds H leaf)
    (hpath : PathBinds H path) : (pathCommit at_ leaf π path layer).Binds H := by
  intro roots o o' hpos hne h h'
  unfold Commitment.Opens at h h'
  simp only [pathCommit] at h h' ⊢
  rw [← hpos] at h' ⊢
  cases ha : at_ roots o.pos with
  | none => simp [ha, OC.eval] at h
  | some ri =>
    simp only [ha, ocBind_eval] at h h' ⊢
    cases hl : ((leaf o).eval H).1 with
    | none => simp [hl, OC.eval] at h
    | some l =>
      cases hl' : ((leaf o').eval H).1 with
      | none => simp [hl', OC.eval] at h'
      | some l' =>
        simp only [hl, hl'] at h h' ⊢
        by_cases he : l = l'
        · subst he
          exact (hleaf o o' l hpos hne hl hl').mono fun q hq => by simp only [List.mem_append] at hq ⊢; tauto
        · exact (hpath ri.1 ri.2 _ _ (π o) (π o') he h h').mono fun q hq => by
            simp only [List.mem_append] at hq ⊢; tauto

/-- **Sampling: binding, at the registered path check**: a deployment whose trace commitment checks its leaves with the
verifier's own registered path check binds, at every `H` where its leaf binds the value. -/
theorem Binding_of_regPath {Coin Col T : Type} (D : Deploy Coin Col T) (own : Array Registered.Port)
    (at_ : List Msg → ℕ → Option (Msg × ℕ)) (leaf : Opening Bool → OC (Option Msg)) (π : Opening Bool → Msg)
    (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hD : D.commit = pathCommit at_ leaf π (regPath own) layer) (hleaf : ∀ H, LeafBinds H leaf) : Binding D :=
  fun H => hD ▸ pathCommit_binds (hleaf H) (regPath_binds H own)

end Claims.Sampling
