import Proofs.Properties.Nci.BodiesOk.Decode

/-!
# `composite`'s step

`composite_succ`: a composite Definition that `composite` decodes is `DefOk` and keeps the decoder's invariant,
given that `getDef` and `staticFns` one fuel lower do (`GetSpec`, `StatSpec`). The node loop's invariant is `NodesOk`:
each node built so far calls an already decoded Definition as `NodeOk` says (`buildNode_spec`), declares the output it
computes, and holds arguments whose parts name leaves (`Walk.parsePartsIn_valid`). `specs` closes the induction on the
fuel.

The decoder is a `do` block whose join points Lean inlines once a step applies them to the decoder state, so each step
names them first (`st_get_bind`, `lift_lets`, `extract_lets`) and splits each `if` and `match` once.
-/

namespace NciCheck.Properties.SecurityProofs.BodiesOk
open Flock.Program
open private getDef composite staticFns DecState DecState.defs DecState.index DecState.busy Desc Desc.types Desc.callees
  liftV at_ paramsOf buildNode natJ? from Flock.Program

theorem inv_congr {s s' : DecState} (e1 : DecState.defs s' = DecState.defs s)
    (e2 : DecState.index s' = DecState.index s) (h : Inv s) : Inv s' := by
  unfold Inv; rw [e1, e2]; exact h

theorem st_modify {f : DecState → DecState} {s : DecState} {u : PUnit} {s' : DecState}
    (h : (modify f : DM PUnit) s = .ok (u, s')) : s' = f s := by
  injection h with h; injection h with h1 h2; exact h2.symm

/-- A loop whose every step leaves the state alone leaves it alone. -/
theorem forIn_same {α β : Type} (l : List α) (f : α → β → DM (ForInStep β))
    (hf : ∀ x b s t s', f x b s = .ok (t, s') → s' = s) :
    ∀ (b : β) (s : DecState) (r : β) (s' : DecState), forIn l b f s = .ok (r, s') → s' = s := by
  intro b s r s' h
  have := forIn_stateT_inv f (fun _ s1 => s1 = s) l (fun _ _ s1 => s1 = s) b r s s' rfl
    (fun i hi b s1 t s2 hg hf' => by
      have := (hf _ _ _ _ _ hf').trans hg
      rcases t <;> exact this) h
  rcases this with h | h <;> exact h

theorem list_mapM_same {α β : Type} (f : α → DM β) (hf : ∀ x s y s', f x s = .ok (y, s') → s' = s) :
    ∀ (l : List α) (s : DecState) (ys : List β) (s' : DecState), l.mapM f s = .ok (ys, s') → s' = s
  | [], s, ys, s', h => by
    simp only [List.mapM_nil] at h
    exact (st_pure h).2.symm
  | x :: l, s, ys, s', h => by
    rw [List.mapM_cons] at h
    obtain ⟨y, s1, h1, h⟩ := st_bind h
    obtain ⟨y2, s2, h2, h⟩ := st_bind h
    obtain ⟨-, rfl⟩ := st_pure h
    rw [list_mapM_same f hf l s1 y2 s2 h2, hf _ _ _ _ h1]

theorem array_mapM_same {α β : Type} (f : α → DM β) (hf : ∀ x s y s', f x s = .ok (y, s') → s' = s)
    (xs : Array α) (s : DecState) (ys : Array β) (s' : DecState) (h : xs.mapM f s = .ok (ys, s')) : s' = s := by
  rw [Array.mapM_eq_mapM_toList] at h
  simp only [Functor.map, StateT.map] at h
  obtain ⟨⟨y, s1⟩, h1, h⟩ := Walk.except_bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  rw [← h.2]
  exact list_mapM_same f hf _ _ _ _ h1

/-- The leaves `composite` gives `parseParts`: a parameter's, or a node's declared output's. -/
def leavesOf (params : Array (String × Ty)) (outs : Array Ty) (p : Bool) (i : ℕ) : Option ℕ :=
  if p = true then params[i]?.map (·.2.leaves) else outs[i]?.map (·.leaves)

theorem refIn_of (params : Array (String × Ty)) (outs : Array Ty) (retType : Ty) (nodes : Array Node) (ret : Seq)
    (offsets : Array ℕ) (hsz : nodes.size = outs.size) (hout : ∀ k < nodes.size, (nodes[k]!).out = outs[k]!)
    (ps : Array Part) (hv : ∀ q ∈ ps, Walk.PartValid (leavesOf params outs) q) (t : ℕ)
    (ht : t < (Seq.ofParts ps).count) :
    RefIn { params, retType, nodes, ret, offsets } ((Seq.ofParts ps).get t) := by
  obtain ⟨i, hi, h1, h2, hget⟩ := Walk.seq_get ps ht
  rw [hget]
  obtain ⟨m, hm, hj⟩ := hv ps[i]! (by rw [getElem!_pos ps i hi]; exact Array.getElem_mem hi) (t - (Seq.ofParts ps).starts[i]!) (by omega)
  generalize ps[i]!.get (t - (Seq.ofParts ps).starts[i]!) = r at hm hj
  rcases r with ⟨p, ri, rj⟩
  cases p
  · simp only [leavesOf, Bool.false_eq_true, ite_false, Option.map_eq_some_iff] at hm
    obtain ⟨ty, hty, rfl⟩ := hm
    obtain ⟨hlt, hty⟩ := Array.getElem?_eq_some_iff.1 hty
    refine ⟨fun h => absurd h (by simp), fun _ => ⟨by show ri < nodes.size; omega, ?_⟩⟩
    show rj < (nodes[ri]!).out.leaves
    rw [hout ri (by omega), getElem!_pos outs ri hlt, hty]; exact hj
  · simp only [leavesOf, ite_true, Option.map_eq_some_iff] at hm
    obtain ⟨pr, hpr, rfl⟩ := hm
    obtain ⟨hlt, hpr⟩ := Array.getElem?_eq_some_iff.1 hpr
    refine ⟨fun _ => ⟨hlt, ?_⟩, fun h => absurd h (by simp)⟩
    show rj < (params[ri]!).2.leaves
    rw [getElem!_pos params ri hlt, hpr]; exact hj

/-- What the node loop has established of the nodes it built. -/
def NodesOk (defs : Array Defn) (outs : Array Ty) (leaves : Bool → ℕ → Option ℕ) (nodes : Array Node) : Prop :=
  ∀ k < nodes.size, (nodes[k]!).fn < defs.size ∧ NodeOk defs[(nodes[k]!).fn]! (nodes[k]!) ∧
    (nodes[k]!).out = outs[k]! ∧ ∀ a < (nodes[k]!).args.size, ∃ ps : Array Part,
      ((nodes[k]!).args[a]!).seq = Seq.ofParts ps ∧ (∀ q ∈ ps, Walk.PartValid leaves q) ∧ ∀ q ∈ ps, PartNonneg q

theorem nodesOk_mono {a b : Array Defn} {outs : Array Ty} {leaves : Bool → ℕ → Option ℕ} {nodes : Array Node}
    (hp : Pre a b) (h : NodesOk a outs leaves nodes) : NodesOk b outs leaves nodes := fun k hk => by
  obtain ⟨h1, h2, h3, h4⟩ := h k hk
  exact ⟨lt_of_lt_of_le h1 hp.1, by rw [hp.2 _ h1]; exact h2, h3, h4⟩

theorem earlier_ok (defs : Array Defn) (outs : Array Ty) (leaves : Bool → ℕ → Option ℕ) (nodes : Array Node)
    (hn : NodesOk defs outs leaves nodes) :
    ∀ (k x : ℕ) (s : Seq), (nodes[k]?.bind fun (nd : Node) => (nd.args[x]?).map (fun (a : Arg) => a.seq)) = some s →
      ∀ q ∈ s.parts, Walk.PartValid leaves q := by
  intro k x s h q hq
  simp only [Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
  obtain ⟨nd, hnd, a, ha, rfl⟩ := h
  obtain ⟨hk, rfl⟩ := Array.getElem?_eq_some_iff.1 hnd
  obtain ⟨hx, rfl⟩ := Array.getElem?_eq_some_iff.1 ha
  obtain ⟨-, -, -, h4⟩ := hn k hk
  obtain ⟨ps, hps, hv, -⟩ := h4 x (by rw [getElem!_pos nodes k hk]; exact hx)
  rw [getElem!_pos nodes k hk, getElem!_pos _ x hx] at hps
  rw [hps, (Walk.ofParts_spec ps).1] at hq
  exact hv q hq

theorem earlier_ok' (defs : Array Defn) (outs : Array Ty) (leaves : Bool → ℕ → Option ℕ) (nodes : Array Node)
    (hn : NodesOk defs outs leaves nodes) :
    ∀ (k x : ℕ) (s : Seq), (if k < nodes.size then (nodes[k]!.args[x]?).map (fun (a : Arg) => a.seq) else none) = some s →
      ∀ q ∈ s.parts, Walk.PartValid leaves q := by
  intro k x s h
  split at h
  · rename_i hk
    exact earlier_ok defs outs leaves nodes hn k x s (by rw [Array.getElem?_eq_getElem hk, ← getElem!_pos nodes k hk]; exact h)
  · exact absurd h (by simp)

theorem earlier_nonneg (defs : Array Defn) (outs : Array Ty) (leaves : Bool → ℕ → Option ℕ) (nodes : Array Node)
    (hn : NodesOk defs outs leaves nodes) :
    ∀ (k x : ℕ) (s : Seq), (if k < nodes.size then (nodes[k]!.args[x]?).map (fun (a : Arg) => a.seq) else none) = some s →
      ∀ q ∈ s.parts, PartNonneg q := by
  intro k x s h q hq
  split at h
  swap; · exact absurd h (by simp)
  rename_i hk
  simp only [Option.map_eq_some_iff] at h
  obtain ⟨a, ha, rfl⟩ := h
  obtain ⟨hx, rfl⟩ := Array.getElem?_eq_some_iff.1 ha
  obtain ⟨-, -, -, h4⟩ := hn k hk
  obtain ⟨ps, hps, -, hv⟩ := h4 x hx
  rw [← getElem!_pos _ x hx, hps, (Walk.ofParts_spec ps).1] at hq
  exact hv q hq

theorem earlier_nonneg_bind (defs : Array Defn) (outs : Array Ty) (leaves : Bool → ℕ → Option ℕ)
    (nodes : Array Node) (hn : NodesOk defs outs leaves nodes) :
    ∀ (k x : ℕ) (s : Seq), (nodes[k]?.bind fun (nd : Node) => (nd.args[x]?).map (fun (a : Arg) => a.seq)) = some s →
      ∀ q ∈ s.parts, PartNonneg q := by
  intro k x s h q hq
  simp only [Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
  obtain ⟨nd, hnd, a, ha, rfl⟩ := h
  obtain ⟨hk, rfl⟩ := Array.getElem?_eq_some_iff.1 hnd
  obtain ⟨hx, rfl⟩ := Array.getElem?_eq_some_iff.1 ha
  obtain ⟨-, -, -, h4⟩ := hn k hk
  obtain ⟨ps, hps, -, hv⟩ := h4 x (by rw [getElem!_pos nodes k hk]; exact hx)
  rw [getElem!_pos nodes k hk, getElem!_pos _ x hx] at hps
  rw [hps, (Walk.ofParts_spec ps).1] at hq
  exact hv q hq

theorem st_ite {α : Type} {c : Prop} [Decidable c] {A B : DM α} {s : DecState} {r : Except String (α × DecState)}
    (h : (if c then A else B) s = r) : (c ∧ A s = r) ∨ (¬ c ∧ B s = r) := by
  split at h <;> rename_i hc
  · exact Or.inl ⟨hc, h⟩
  · exact Or.inr ⟨hc, h⟩

theorem st_throw_bind {α β : Type} {e : String} {f : α → DM β} {s : DecState} {b : β} {s' : DecState}
    (h : (throw e >>= f : DM β) s = .ok (b, s')) : False := by
  replace h := st_bind h; obtain ⟨a, s1, h1, -⟩ := h
  exact absurd h1 (by intro h; injection h)

theorem st_get_bind {β : Type} {f : DecState → DM β} {s : DecState} {r : Except String (β × DecState)}
    (h : (get >>= f : DM β) s = r) : ∃ g, f s = g ∧ g s = r := ⟨_, rfl, h⟩

theorem composite_succ (n : ℕ) (hg : GetSpec n) (hst : StatSpec n) : CompSpec (n + 1) := by
  intro desc fid d form id s defn s' h hs
  unfold composite at h
  extract_lets parts name ver bindings rec0 b nodes0 offsets0 order main at h
  split at h
  · exact absurd h (by intro h; injection h)
  simp (config := { zeta := false }) only [main] at h
  rcases hbk : bindings with _ | _ | _ | _ | _ | bkv
  all_goals rw [hbk] at h
  all_goals dsimp (config := { zeta := false }) only at h
  all_goals try (exact absurd h (by intro h; injection h))
  extract_lets names jp1 jpS jpO at h
  have hS : ∃ X, jpS X s = .ok (defn, s') := by
    split at h
    · replace h := st_bind h; obtain ⟨a, s1, h1, h2⟩ := h
      obtain ⟨rfl, rfl⟩ := st_pure h1
      exact ⟨_, h2⟩
    · split at h
      · replace h := st_bind h; obtain ⟨a, s1, h1, -⟩ := h
        exact absurd h1 (by intro h; injection h)
      · simp (config := { zeta := false }) only [jpO] at h
        replace h := st_bind h; obtain ⟨a, s1, h1, h2⟩ := h
        obtain ⟨rfl, rfl⟩ := st_pure h1
        exact ⟨_, h2⟩
  clear h
  obtain ⟨X, h⟩ := hS
  simp (config := { zeta := false }) only [jpS] at h
  replace h := st_bind h; obtain ⟨rr, s1, hf, h⟩ := h
  rw [← Array.forIn_toList] at hf
  have e1 := forIn_same _ _ (fun x b s t s' hf => by
    obtain ⟨a, s2, h1, h2⟩ := st_bind hf
    obtain ⟨-, rfl⟩ := st_liftV h1
    exact (st_pure h2).2.symm) _ _ _ _ hf
  subst s1
  clear hf
  replace h := st_bind h; obtain ⟨u, s2, hm, h⟩ := h
  have e2 := st_modify hm
  have hs2 : Inv s2 := inv_congr (by rw [e2]) (by rw [e2]) hs
  have hp2 : DecState.defs s2 = DecState.defs s := by rw [e2]
  clear hm e2
  generalize (id ++ if X.isEmpty = true then "" else "{" ++ ",".intercalate rr.toList ++ "}") = key at h
  split at h
  · replace h := st_bind h; obtain ⟨a, s3, h1, -⟩ := h
    exact absurd h1 (by intro h; injection h)
  simp (config := { zeta := false }) only [jp1] at h
  replace h := st_bind h; obtain ⟨u, s3, hf, h⟩ := h
  rw [← Array.forIn_toList] at hf
  have g3 := forIn_grows _ _ (fun x b s t s' hf => by
    obtain ⟨u2, s2, h1, h2⟩ := st_bind hf
    obtain ⟨-, rfl⟩ := st_pure h2
    exact hst _ _ _ _ _ h1) _ _ _ _ hf
  obtain ⟨hs3, hp3⟩ := g3 hs2
  rw [hp2] at hp3
  clear hf g3 hs2 hp2 hs
  replace h := st_bind h; obtain ⟨params, s4, hpa, h⟩ := h
  replace hpa := st_liftV hpa; obtain ⟨hpa, rfl⟩ := hpa
  replace h := st_bind h; obtain ⟨retType, s4, hrt, h⟩ := h
  replace hrt := st_liftV hrt; obtain ⟨hrt, rfl⟩ := hrt
  rcases hnd : b.getObjVal? "nodes" with e | (_ | _ | _ | _ | nds | _)
  all_goals rw [hnd] at h
  all_goals dsimp (config := { zeta := false }) only at h
  all_goals try (exact absurd h (by intro h; injection h))
  replace h := st_bind h; obtain ⟨outs, s4, hou, h⟩ := h
  replace hou := st_liftV hou; obtain ⟨hou, rfl⟩ := hou
  replace h := st_bind h; obtain ⟨⟨nodes, offsets⟩, s5, hloop, h⟩ := h
  have hosz : outs.size = nds.size := array_mapM_size _ _ _ hou
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  have hL := forIn_stateT_inv _ (fun _ _ => False) _
    (fun i (st : Array Node × Array ℕ) s1 => Inv s1 ∧ Pre (DecState.defs s3) (DecState.defs s1) ∧ st.1.size = i ∧
       NodesOk (DecState.defs s1) outs (leavesOf params outs) st.1) (nodes0, offsets0) _ _ _
    ⟨hs3, pre_refl _, rfl, fun k hk => absurd hk (Nat.not_lt_zero k)⟩ ?step hloop
  case step =>
    rintro i hi ⟨nodes1, offsets1⟩ s1 t s2 ⟨hs1, hp1, hsz1, hn1⟩ h
    simp only [List.length_range', Std.Legacy.Range.size] at hi
    simp (config := { zeta := false }) only [List.getElem_range', Nat.one_mul, Nat.zero_add] at h
    dsimp only at hsz1 hn1
    replace h := st_bind h; obtain ⟨fname, s3', h1, h⟩ := h
    replace h1 := st_liftV h1; obtain ⟨hfn, rfl⟩ := h1
    rcases fname with _ | _ | _ | fname | _ | _
    all_goals dsimp (config := { zeta := false }) only at h
    all_goals try (replace h := st_bind h; obtain ⟨a, s7, h1, -⟩ := h; exact absurd h1 (by intro h; injection h))
    replace h := st_bind h; obtain ⟨ci, s4, hgd, h⟩ := h
    obtain ⟨hs4, hp4, hci⟩ := hg _ _ _ _ _ hgd hs1
    replace h := st_get_bind h; obtain ⟨g, hgd', h⟩ := h
    extract_lets callee fstr keys earlier JP at hgd'
    subst hgd'
    clear_value fstr
    have hJ : ∃ F, JP F s4 = .ok (t, s2) := by
      split at h
      all_goals replace h := st_bind h; obtain ⟨F, s5, h1, h⟩ := h
      rotate_left 4
      · exact absurd h1 (by intro h; injection h)
      all_goals obtain ⟨rfl, rfl⟩ := st_pure h1; exact ⟨_, h⟩
    clear h
    obtain ⟨F, h⟩ := hJ
    simp (config := { zeta := false }) only [JP] at h
    lift_lets at h
    extract_lets jpC jpP jpK at h
    replace h := st_ite h; rcases h with ⟨-, h⟩ | ⟨hkeys, h⟩
    · exact (st_throw_bind h).elim
    simp (config := { zeta := false }) only [jpK] at h
    replace h := st_ite h; rcases h with ⟨-, h⟩ | ⟨hprim, h⟩
    · exact (st_throw_bind h).elim
    simp (config := { zeta := false }) only [jpP] at h
    replace h := st_ite h; rcases h with ⟨-, h⟩ | ⟨hcall, h⟩
    · exact (st_throw_bind h).elim
    simp (config := { zeta := false }) only [jpC] at h
    rcases hargs : nds[i]!.getObjVal? "args" with e | ⟨_ | _ | _ | _ | ajs | _⟩
    all_goals rw [hargs] at h
    all_goals dsimp (config := { zeta := false }) only at h
    all_goals try (exact (st_throw_bind h).elim)
    lift_lets at h
    extract_lets jpAx at h
    have hAx : ∃ axes, jpAx axes s4 = .ok (t, s2) := by
      replace h := st_ite h; rcases h with ⟨-, h⟩ | ⟨-, h⟩
      · rcases haxes : nds[i]!.getObjVal? "axes" with e | ⟨_ | _ | _ | _ | xs | _⟩
        all_goals rw [haxes] at h
        all_goals dsimp (config := { zeta := false }) only at h
        all_goals try (exact (st_throw_bind h).elim)
        replace h := st_bind h; obtain ⟨axes, s5, h1, h⟩ := h
        have := array_mapM_same _ (fun x s y s' hf => by
          split at hf
          · exact (st_pure hf).2.symm
          · rcases ho : natJ? x with _ | a
            all_goals rw [ho] at hf; dsimp only at hf
            · exact absurd hf (by intro h; injection h)
            · exact (st_pure hf).2.symm) _ _ _ _ h1
        subst this
        exact ⟨_, h⟩
      · replace h := st_bind h; obtain ⟨axes, s5, h1, h⟩ := h
        obtain ⟨rfl, rfl⟩ := st_pure h1
        exact ⟨_, h⟩
    clear h
    obtain ⟨axes, h⟩ := hAx
    simp (config := { zeta := false }) only [jpAx] at h
    lift_lets at h
    extract_lets jpN at h
    have hN : ∃ nn, jpN nn s4 = .ok (t, s2) := by
      replace h := st_ite h; rcases h with ⟨-, h⟩ | ⟨-, h⟩
      · rcases hn : (nds[i]!.getObjVal? "n").toOption.bind (fun a => natJ? a) with _ | nn
        all_goals rw [hn] at h; dsimp (config := { zeta := false }) only at h
        · exact (st_throw_bind h).elim
        · replace h := st_bind h; obtain ⟨a, s5, h1, h⟩ := h
          obtain ⟨rfl, rfl⟩ := st_pure h1
          exact ⟨_, h⟩
      · replace h := st_bind h; obtain ⟨a, s5, h1, h⟩ := h
        obtain ⟨rfl, rfl⟩ := st_pure h1
        exact ⟨_, h⟩
    clear h
    obtain ⟨nn, h⟩ := hN
    simp (config := { zeta := false }) only [jpN] at h
    lift_lets at h
    extract_lets jpX at h
    have hX : ∃ nxs, (F = .scan → nxs < ajs.size) ∧ jpX nxs s4 = .ok (t, s2) := by
      replace h := st_ite h; rcases h with ⟨hsc, h⟩ | ⟨hsc, h⟩
      · rcases hx : (nds[i]!.getObjVal? "n_xs").toOption.bind (fun a => natJ? a) with _ | x
        all_goals rw [hx] at h; dsimp (config := { zeta := false }) only at h
        · exact (st_throw_bind h).elim
        · replace h := st_ite h; rcases h with ⟨hlt, h⟩ | ⟨-, h⟩
          · replace h := st_bind h; obtain ⟨a, s5, h1, h⟩ := h
            obtain ⟨rfl, rfl⟩ := st_pure h1
            exact ⟨_, fun _ => hlt, h⟩
          · exact (st_throw_bind h).elim
      · replace h := st_bind h; obtain ⟨a, s5, h1, h⟩ := h
        obtain ⟨rfl, rfl⟩ := st_pure h1
        exact ⟨_, fun e => absurd (by rw [e]; rfl) hsc, h⟩
    clear h
    obtain ⟨nxs, hnxs, h⟩ := hX
    simp (config := { zeta := false }) only [jpX] at h
    replace h := st_bind h; obtain ⟨args, s5, h1, h⟩ := h
    replace h1 := st_liftV h1; obtain ⟨hmap, rfl⟩ := h1
    replace h := st_bind h; obtain ⟨⟨out, args'⟩, s5, h1, h⟩ := h
    replace h1 := st_liftV h1; obtain ⟨hbn, rfl⟩ := h1
    replace h := st_ite h; rcases h with ⟨-, h⟩ | ⟨hout, h⟩
    · exact (st_throw_bind h).elim
    obtain ⟨rfl, rfl⟩ := st_pure h
    clear jpK jpP jpC jpAx jpN jpX
    have hasz : args.size = ajs.size := array_mapM_size _ _ _ hmap
    have hel : ∀ a < args.size, ∃ ty ps, args[a]! = ({ ty, seq := Seq.ofParts ps } : Arg) ∧
        (∀ q ∈ ps, Walk.PartValid (leavesOf params outs) q) ∧ (∀ q ∈ ps, PartNonneg q) ∧
        (Seq.ofParts ps).count = ty.leaves := by
      intro a ha
      have h1 := array_mapM_get _ _ _ hmap a (by omega) ha
      beta_reduce at h1
      obtain ⟨ty, -, h2⟩ := Walk.except_bind_ok h1
      dsimp only at h2
      split at h2
      · rename_i ps hps
        obtain ⟨ps', h3, h2⟩ := Walk.except_bind_ok h2
        injection h3 with h3; subst h3
        split at h2
        · obtain ⟨u, h3, -⟩ := Walk.except_bind_ok h2
          exact absurd h3 (by intro h; injection h)
        rename_i hc
        injection h2 with h2
        refine ⟨ty, ps, by rw [getElem!_pos args a ha, ← h2], ?_, ?_, by simpa using hc⟩
        · exact Walk.parsePartsIn_valid _ _ (earlier_ok' _ _ _ _ hn1) _ _ _ hps
        · exact parsePartsIn_nonneg _ _ (earlier_nonneg _ _ _ _ hn1) _ _ _ hps
      · obtain ⟨u, h3, -⟩ := Walk.except_bind_ok h2
        exact absurd h3 (by intro h; injection h)
    obtain ⟨b1, b2, b3, b5, b4⟩ := buildNode_spec _ _ _ _ _ _ _ _ _ _ hbn
      (fun e => by have := hnxs e; omega)
      (fun a ha => by obtain ⟨ty, ps, e, -⟩ := hel a ha; rw [e])
    dsimp only
    refine ⟨hs4, pre_trans hp1 hp4, by simp [hsz1], fun k hk => ?_⟩
    simp only [Array.size_push] at hk
    rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
    · rw [getElem!_push_lt _ _ (by omega)]
      exact nodesOk_mono hp4 hn1 k (by omega)
    · rw [getElem!_push_eq]
      refine ⟨hci, ⟨?_, fun a ha => ?_, fun e => ?_, fun e => ?_, b4, b5⟩, ?_, fun a ha => ?_⟩
      · exact b1.trans b3
      · dsimp only at ha ⊢
        obtain ⟨c1, c2⟩ := b2 a (by omega)
        obtain ⟨ty, ps, e, -, -, hc⟩ := hel a (by omega)
        rw [c1, c2, e]; exact hc
      · dsimp only at e; subst e
        revert hprim; cases (DecState.defs s4)[ci]!.prim <;> simp
        decide
      · dsimp only at e
        revert hcall; rcases e with rfl | rfl <;> cases (DecState.defs s4)[ci]!.prim <;> simp <;> decide
      · rw [hsz1]; exact ty_not_beq (by simpa using hout)
      · dsimp only at ha ⊢
        obtain ⟨c1, c2⟩ := b2 a (by omega)
        obtain ⟨ty, ps, e, hv, hnn, -⟩ := hel a (by omega)
        refine ⟨ps, ?_, hv, hnn⟩
        rw [c2, e]
  simp only [List.length_range', Std.Legacy.Range.size] at hL
  rcases hL with ⟨hs5, hp5, hsz5, hn5⟩ | hL
  swap; · exact hL.elim
  dsimp only at hsz5 hn5 h
  rcases hret : parseParts ((b.getObjVal? "ret").toOption.getD Lean.Json.null) (leavesOf params outs)
    (fun k2 a => nodes[k2]?.bind fun x => Option.map (fun x => x.seq) x.args[a]?) with e | ps
  all_goals rw [show (fun p i => if p = true then Option.map (fun (x : String × Ty) => x.2.leaves) params[i]?
      else Option.map (fun (x : Ty) => x.leaves) outs[i]?) = leavesOf params outs from rfl] at h
  all_goals rw [hret] at h
  all_goals dsimp only at h
  · replace h := st_bind h; obtain ⟨a, s6, h1, -⟩ := h
    exact absurd h1 (by intro h; injection h)
  replace h := st_bind h; obtain ⟨ret, s6, h1, h⟩ := h
  obtain ⟨rfl, rfl⟩ := st_pure h1
  split at h
  · replace h := st_bind h; obtain ⟨a, s7, h1, -⟩ := h
    exact absurd h1 (by intro h; injection h)
  rename_i hcnt
  replace h := st_bind h; obtain ⟨st, s7, h1, h⟩ := h
  obtain ⟨rfl, rfl⟩ := st_get h1
  generalize hwalk : EStateM.run _ _ = r at h
  rcases r with ⟨_ | m, w⟩ | ⟨e, w⟩
  all_goals dsimp only at h
  rotate_left
  · replace h := st_bind h; obtain ⟨a, s8, h1, -⟩ := h
    exact absurd h1 (by intro h; injection h)
  · replace h := st_bind h; obtain ⟨a, s8, h1, -⟩ := h
    exact absurd h1 (by intro h; injection h)
  replace h := st_bind h; obtain ⟨u, s8, h1, h⟩ := h
  injection h1 with h1; injection h1 with hu h1; subst h1
  obtain ⟨rfl, rfl⟩ := st_pure h
  have hv := Walk.parsePartsIn_valid _ _ (earlier_ok _ _ _ _ hn5) _ _ _ hret
  have hsz : nodes.size = outs.size := by omega
  have hout : ∀ k < nodes.size, (nodes[k]!).out = outs[k]! := fun k hk => (hn5 k hk).2.2.1
  refine ⟨inv_congr ?_ ?_ hs5, pre_trans hp3 hp5, ⟨⟨fun h => absurd h (by simp), fun _ => ⟨rfl, ?_, rfl⟩⟩, ?_, ?_, ?_⟩⟩
  · rfl
  · rfl
  · simpa using hcnt
  · intro k hk
    obtain ⟨h1, h2, -, h4⟩ := hn5 k hk
    refine ⟨h1, h2, fun a ha => ?_⟩
    obtain ⟨ps', hps', hv', hnn'⟩ := h4 a ha
    refine ⟨⟨ps', hps', hnn'⟩, fun t ht => ?_⟩
    rw [hps'] at ht ⊢
    exact refIn_of params outs retType nodes _ offsets hsz hout ps' hv' t ht
  · intro t ht
    exact refIn_of params outs retType nodes _ offsets hsz hout ps hv t ht
  · exact ⟨ps, rfl, parsePartsIn_nonneg _ _ (earlier_nonneg_bind _ _ _ _ hn5) _ _ _ hret⟩

theorem getSpec_zero : GetSpec 0 := by
  intro desc fid s i s' h
  unfold getDef at h
  exact absurd h (by intro h; injection h)

theorem compSpec_zero : CompSpec 0 := by
  intro desc fid d form id s defn s' h
  unfold composite at h
  exact absurd h (by intro h; injection h)

theorem statSpec_zero : StatSpec 0 := by
  intro desc v s u s' h
  unfold staticFns at h
  exact absurd h (by intro h; injection h)

/-- **The decoder's step specs, at every fuel.** -/
theorem specs : ∀ fuel, GetSpec fuel ∧ CompSpec fuel ∧ StatSpec fuel
  | 0 => ⟨getSpec_zero, compSpec_zero, statSpec_zero⟩
  | n + 1 =>
    let ⟨g, c, s⟩ := specs n
    ⟨getDef_succ n c, composite_succ n g s, statFns_succ n g s⟩

end NciCheck.Properties.SecurityProofs.BodiesOk
