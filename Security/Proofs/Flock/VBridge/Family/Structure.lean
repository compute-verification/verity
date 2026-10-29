import Proofs.Flock.VBridge.Family.Rep
import Proofs.Flock.VBridge.Family.Openings
import Proofs.Flock.VBridge.Family.Files
import Proofs.Flock.Soundness.Refine.Regions

/-!
# The structure facts at V*'s shape

The family's shape facts, by evaluating V*'s and the verifier's Lean at `hsh` and `hstmts`:

* the shape: `RecResiduals.shape?` (`RecResiduals.lean:60`) holds only at a `fast100` schedule (`Schedule.lean:47`,
  `m ∈ [22, 35]`, `fast100_range`), whose level 0 draws queries (`fast100_q0`), so `0 < F.nOpen` (`nOpen_pos`);
* the statements: `VStar.statements` (`Stage.lean:49-60`) is the levels' `RecOpen` statements, then one `InnerRepCheck`
  per part of `parts (spec vsh)` (`statements_eq`), so level `ℓ`'s openings are statement `ℓ` (`statements_recOpen`)
  and part `p` is a statement when V* has it (`statements_inner`): `HOpenUnits` (`hOpenUnits_of`), `HCover`
  (`hCover_of`), from named facts about the circuit files and V*'s parts, and `HSomePart` (`hSomePart`): V*'s S has
  residuals (`spec_res`), so it has a part (`parts_size`);
* the setup: the verifier's setup of a statement (`Setup.ofCircuit`, `Verify.lean:29-32`) realizes its model statement
  (`setupRealizes_ofCircuit`): its extra claims are defined at every pair of link points where each region's bytes are
  its free bits (`RegionBytes`, which the verifier's `regionValue` checks, `Statement.lean:361`), and realize the
  model's there (`ofCircuit_extra`); and `RunsFrom`'s `foldKLog` and `kLog_le` from it (`kLogs_of_setup`).
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockVBridge.AccChain (levelLanes)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The shape -/

/-- `fast100` has a schedule only at `m ∈ [22, 35]` (`Schedule.lean:48`). -/
theorem fast100_range {m : ℕ} {s : Flock.Schedule} (h : Flock.fast100 m = some s) : 22 ≤ m ∧ m ≤ 35 := by
  have hn : ¬ (m < 22 ∨ 35 < m) := by
    intro hc
    have hb : (m < 22 || m > 35) = true := by
      simp only [Bool.or_eq_true, decide_eq_true_eq]; omega
    have : Flock.fast100 m = none := by
      unfold Flock.fast100
      simp [hb]
    rw [this] at h
    cases h
  omega

/-- **Every `fast100` schedule's level 0 draws queries** (`queriesAt[0] = 218`, `Schedule.lean:44,55`), by evaluating
`fast100` at each `m ∈ [22, 35]`. -/
theorem fast100_q0 {m : ℕ} {s : Flock.Schedule} (h : Flock.fast100 m = some s) :
    0 < s.levels.size ∧ 0 < (s.levels[0]!).q := by
  have key : ∀ m', 22 ≤ m' → m' ≤ 35 → (Flock.fast100 m').all
      (fun s' => decide (0 < s'.levels.size) && decide (0 < (s'.levels[0]!).q)) = true := by
    intro m' h1 h2
    interval_cases m' <;> decide +kernel
  have := key m (fast100_range h).1 (fast100_range h).2
  rw [h] at this
  simpa using this

/-- **The verifier's shape at V*'s** (`RecResiduals.lean:60-63`): `fast100`'s schedule at `m`, with `k_log ≥ 6` and at
least two claims. -/
theorem shape?_eq {m k c : ℕ} {sh : Shape} (h : RecResiduals.shape? m k c = some sh) :
    Flock.fast100 m = some sh.sched ∧ sh.m = m ∧ sh.kLog = k ∧ sh.claims = c ∧ 6 ≤ k ∧ 2 ≤ c := by
  unfold RecResiduals.shape? at h
  cases hf : Flock.fast100 m with
  | none => rw [hf] at h; cases h
  | some s =>
    rw [hf] at h
    simp at h
    obtain ⟨⟨h6, h2⟩, rfl⟩ := h
    exact ⟨rfl, rfl, rfl, rfl, h6, by omega⟩

/-- V*'s schedule is `fast100`'s (`Algebra.lean:40-41`). -/
theorem sched_fast100 {vsh : VStar.Algebra.Shape} {s : Flock.Schedule} (h : vsh.sched = .ok s) :
    Flock.fast100 vsh.m = some s := by
  unfold VStar.Algebra.Shape.sched at h
  cases hf : Flock.fast100 vsh.m with
  | none => simp [hf] at h
  | some s' =>
    simp only [hf] at h
    split at h
    · cases h
    · split at h
      · cases h
      · cases h; rfl

/-! ## V*'s statements -/

theorem foldl_push_map {α β : Type} (f : β → α) : ∀ (l : List β) (init : Array α),
    l.foldl (fun out b => out.push (f b)) init = init ++ (l.map f).toArray
  | [], init => by simp
  | b :: l, init => by
    rw [List.foldl_cons, foldl_push_map f l]
    rw [← Array.toList_inj]
    simp

/-- **V*'s statements** (`Stage.lean:49-60`): level `l`'s `RecOpen{2^lanes, d - cap}` over `2 q` instances, for each
level of the schedule, then part `j`'s `InnerRepCheck` named `rec-residuals/m…-p<j>`, for each part of `parts (spec vsh)`. -/
theorem statements_eq {vsh : VStar.Algebra.Shape} {stmts : Array VStar.Stmt} (h : VStar.statements vsh = .ok stmts) :
    ∃ s sp ps, vsh.sched = .ok s ∧ VStar.Algebra.spec vsh = .ok sp ∧ VStar.parts sp = .ok ps ∧
      stmts = ((List.range s.levels.size).map fun l => VStar.Stmt.recOpen l (2 ^ (s.levels[l]!).lanes)
          ((s.levels[l]!).d - (s.levels[l]!).cap) (2 * (s.levels[l]!).q)).toArray ++
        ((List.range ps.size).map fun j =>
          VStar.Stmt.inner j s!"rec-residuals/m{vsh.m}-k{vsh.kLog}-c{vsh.claims}-p{j}" ps[j]!).toArray := by
  unfold VStar.statements at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind, bind_pure] at h
  cases hs : vsh.sched with
  | error e => simp [hs, Bind.bind, Except.bind] at h
  | ok s =>
    cases hsp : VStar.Algebra.spec vsh with
    | error e => simp [hs, hsp, Bind.bind, Except.bind] at h
    | ok sp =>
      cases hps : VStar.parts sp with
      | error e => simp [hs, hsp, hps, Bind.bind, Except.bind] at h
      | ok ps =>
        simp only [hs, hsp, hps, Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq] at h
        subst h
        refine ⟨s, sp, ps, rfl, rfl, hps, ?_⟩
        rw [foldl_push_map, foldl_push_map]
        simp [Std.Legacy.Range.size, List.range_eq_range']

/-- **V*'s statements over a holo inner session** (`VStar/Holo.lean`, `statementsHolo`): `statements_eq`'s form, the
parts `parts (specHolo vsh)`'s. -/
theorem statementsHolo_eq {vsh : VStar.Algebra.Shape} {stmts : Array VStar.Stmt}
    (h : VStar.statementsHolo vsh = .ok stmts) :
    ∃ s sp ps, vsh.sched = .ok s ∧ VStar.Algebra.specHolo vsh = .ok sp ∧ VStar.parts sp = .ok ps ∧
      stmts = ((List.range s.levels.size).map fun l => VStar.Stmt.recOpen l (2 ^ (s.levels[l]!).lanes)
          ((s.levels[l]!).d - (s.levels[l]!).cap) (2 * (s.levels[l]!).q)).toArray ++
        ((List.range ps.size).map fun j =>
          VStar.Stmt.inner j s!"rec-residuals/m{vsh.m}-k{vsh.kLog}-c{vsh.claims}-p{j}" ps[j]!).toArray := by
  unfold VStar.statementsHolo at h
  cases hs : vsh.sched with
  | error e => simp [hs, Bind.bind, Except.bind] at h
  | ok s =>
    cases hsp : VStar.Algebra.specHolo vsh with
    | error e => simp [hs, hsp, Bind.bind, Except.bind] at h
    | ok sp =>
      cases hps : VStar.parts sp with
      | error e => simp [hs, hsp, hps, Bind.bind, Except.bind] at h
      | ok ps =>
        simp only [hs, hsp, hps, Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq] at h
        subst h
        refine ⟨s, sp, ps, rfl, rfl, hps, ?_⟩
        rw [← Array.toList_inj]
        simp only [Array.toList_append, Array.toList_map, Array.toList_range, Array.toList_mapIdx]
        congr 1
        apply List.ext_getElem
        · simp
        · intro n _ h2
          have hn : n < ps.size := by simpa using h2
          simp [List.getElem_mapIdx, getElem!_pos ps n hn]

/-- **V*'s statements over an inner session holo or plain** (`statementsOf`): `statements_eq`'s form, the parts
`parts (specOf vsh holo)`'s. -/
theorem statementsOf_eq {vsh : VStar.Algebra.Shape} {holo : Bool} {stmts : Array VStar.Stmt}
    (h : VStar.statementsOf vsh holo = .ok stmts) :
    ∃ s sp ps, vsh.sched = .ok s ∧ VStar.Algebra.specOf vsh holo = .ok sp ∧ VStar.parts sp = .ok ps ∧
      stmts = ((List.range s.levels.size).map fun l => VStar.Stmt.recOpen l (2 ^ (s.levels[l]!).lanes)
          ((s.levels[l]!).d - (s.levels[l]!).cap) (2 * (s.levels[l]!).q)).toArray ++
        ((List.range ps.size).map fun j =>
          VStar.Stmt.inner j s!"rec-residuals/m{vsh.m}-k{vsh.kLog}-c{vsh.claims}-p{j}" ps[j]!).toArray := by
  cases holo
  · exact statements_eq h
  · exact statementsHolo_eq h

/-- Statement `l` is level `l`'s openings. -/
theorem statements_recOpen {vsh : VStar.Algebra.Shape} {stmts : Array VStar.Stmt}
    (h : VStar.statements vsh = .ok stmts) {s : Flock.Schedule} (hs : vsh.sched = .ok s) {l : ℕ}
    (hl : l < s.levels.size) :
    ∃ hj : l < stmts.size, stmts[l] = .recOpen l (2 ^ (s.levels[l]!).lanes) ((s.levels[l]!).d - (s.levels[l]!).cap)
      (2 * (s.levels[l]!).q) := by
  obtain ⟨s', sp, ps, hs', -, -, rfl⟩ := statements_eq h
  rw [hs] at hs'
  cases hs'
  refine ⟨by simp; omega, ?_⟩
  rw [Array.getElem_append_left (by simpa using hl)]
  simp

/-- Each of V*'s parts is a statement. -/
theorem statements_inner {vsh : VStar.Algebra.Shape} {stmts : Array VStar.Stmt}
    (h : VStar.statements vsh = .ok stmts) {sp : VStar.RepS} (hsp : VStar.Algebra.spec vsh = .ok sp)
    {ps : Array VStar.Part} (hps : VStar.parts sp = .ok ps) {p : ℕ} (hp : p < ps.size) :
    ∃ (j : Fin stmts.size) (nm : String) (pt : VStar.Part), stmts[j] = .inner p nm pt := by
  obtain ⟨s, sp', ps', -, hsp', hps', rfl⟩ := statements_eq h
  rw [hsp] at hsp'
  cases hsp'
  rw [hps] at hps'
  cases hps'
  refine ⟨⟨s.levels.size + p, by simp; omega⟩, s!"rec-residuals/m{vsh.m}-k{vsh.kLog}-c{vsh.claims}-p{p}", ps[p]!, ?_⟩
  simp

/-! ## V*'s parts -/

/-- `supports` (`Algebra.lean:287-315`) ends with Ligerito's three residuals. -/
theorem supports_size {cx : VStar.Algebra.Ctx} {syms : Array VStar.Algebra.Sym}
    (h : VStar.Algebra.supports cx = .ok syms) : 3 ≤ syms.size := by
  unfold VStar.Algebra.supports at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp

theorem stateT_forIn_push {σ ε α β : Type} (g : α → StateT σ (Except ε) β) :
    ∀ (L : List α) (out : Array β) (s : σ) (res : Array β) (s' : σ),
      (forIn L out (fun a r => do let v ← g a; pure (ForInStep.yield (r.push v)))).run s = .ok (res, s') →
      res.size = out.size + L.length
  | [], out, s, res, s', h => by
    simp only [List.forIn_nil] at h
    cases h
    simp
  | a :: L, out, s, res, s', h => by
    rw [List.forIn_cons] at h
    simp only [StateT.run, Bind.bind, StateT.bind, Except.bind] at h
    cases hg : g a s with
    | error e => rw [hg] at h; cases h
    | ok vs =>
      rw [hg] at h
      obtain ⟨v, s₁⟩ := vs
      have := stateT_forIn_push g L (out.push v) s₁ res s' h
      simp at this ⊢
      omega

/-- `build` (`Algebra.lean:366-374`) makes one residual per support. -/
theorem build_res {cx : VStar.Algebra.Ctx} {lins : Array VStar.Algebra.Sym} {gf : VStar.GfS}
    (h : VStar.Algebra.build cx lins = .ok gf) : gf.res.length = lins.size := by
  unfold VStar.Algebra.build at h
  dsimp only at h
  obtain ⟨⟨res, st⟩, hgo, h⟩ := ebind_ok h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Array.length_toList]
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hgo
    simp only [bind_pure] at hgo
    have := stateT_forIn_push _ _ _ _ _ _ hgo
    simpa [Std.Legacy.Range.size] using this

/-- **V*'s S has residuals** (`spec`, `Algebra.lean:382-389`). -/
theorem spec_res {sh : VStar.Algebra.Shape} {sp : VStar.RepS} (h : VStar.Algebra.spec sh = .ok sp) :
    0 < sp.gf.res.length := by
  unfold VStar.Algebra.spec at h
  dsimp only at h
  obtain ⟨cx, -, h⟩ := ebind_ok h
  obtain ⟨syms, hsy, h⟩ := ebind_ok h
  obtain ⟨gf, hgf, h⟩ := ebind_ok h
  have h3 := supports_size hsy
  have hr := build_res hgf
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    dsimp only
    omega

theorem forIn_size_inv {ε α σ : Type} (Q : σ → Prop) (f : α → σ → Except ε (ForInStep σ))
    (hf : ∀ a r y, f a r = .ok y → ∃ r', y = .yield r' ∧ Q r') :
    ∀ (L : List α) (init v : σ), (Q init ∨ L ≠ []) → forIn L init f = .ok v → Q v
  | [], init, v, hq, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simpa using hq
  | a :: L, init, v, _, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := ebind_ok h
    obtain ⟨r', rfl, hq⟩ := hf a init y hy
    exact forIn_size_inv Q f hf L r' v (.inl hq) h

theorem forIn_pushes {ε α β : Type} (f : α → Array β → Except ε (ForInStep (Array β)))
    (hf : ∀ a r y, f a r = .ok y → ∃ x, y = .yield (r.push x)) :
    ∀ (L : List α) (init v : Array β), forIn L init f = .ok v → v.size = init.size + L.length
  | [], init, v, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp
  | a :: L, init, v, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := ebind_ok h
    obtain ⟨x, rfl⟩ := hf a init y hy
    have := forIn_pushes f hf L _ v h
    simp at this ⊢
    omega

/-- **V*'s parts are nonempty where its S has residuals** (`parts`, `Parts.lean:54-105`): residual 0 opens or joins a
group, and each group is one part. -/
theorem parts_size {S : VStar.RepS} {ps : Array VStar.Part} (h : VStar.parts S = .ok ps)
    (hr : 0 < S.gf.res.length) : 0 < ps.size := by
  unfold VStar.parts at h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨G, hG, h⟩ := ebind_ok h
  obtain ⟨out, hout, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hG
  have hGs : 0 < G.size := by
    refine forIn_size_inv (fun r => 0 < r.size) _ ?_ _ _ _ (.inr ?_) hG
    · intro a r y hy
      split at hy
      · rename_i gi hgi
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        subst hy
        exact ⟨_, rfl, by have := (Array.findIdx?_eq_some_iff_findIdx_eq.1 hgi).1; simp; omega⟩
      · split at hy
        · simp only [pure, Except.pure, Except.ok.injEq] at hy
          subst hy
          exact ⟨_, rfl, by simp⟩
        · obtain ⟨_, h1, -⟩ := ebind_ok hy
          cases h1
    · simp [Std.Legacy.Range.size]
      intro he
      simp [he] at hr
  rw [← Array.forIn_toList] at hout
  have := forIn_pushes _ (fun a r y hy => by
    repeat' (first | (obtain ⟨_, _, hy⟩ := ebind_ok hy) | (split at hy))
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      exact ⟨_, hy.symm⟩) _ _ _ hout
  simp at this
  omega

/-! ## The setup -/

/-- **Each region's bytes are its free bits**: block `b`'s bytes hold `2^|free|` bits, for every block. The verifier
refuses a region otherwise when it computes the region's value (`regionValue`, `Statement.lean:361`), at every link
point, so any accepted session establishes it (`regionBytes_of_extraClaims`: `verifyRep` computes the extra claims at the
record's two points, `Verify.lean:41`). -/
def RegionBytes (st : Stmt) : Prop :=
  ∀ reg ∈ st.regions.toList, ∀ b < 2 ^ st.nbl, (reg.bytes b).size * 8 = 2 ^ reg.free.size

theorem list_forIn_ok {ε α σ : Type} (f : α → σ → Except ε (ForInStep σ)) :
    ∀ (L : List α) (s : σ), (∀ a ∈ L, ∀ r, ∃ r', f a r = .ok (.yield r')) → ∃ v, forIn L s f = .ok v
  | [], s, _ => ⟨s, rfl⟩
  | a :: L, s, h => by
    obtain ⟨r', hr⟩ := h a List.mem_cons_self s
    obtain ⟨v, hv⟩ := list_forIn_ok f L r' fun b hb => h b (List.mem_cons_of_mem _ hb)
    exact ⟨v, by rw [List.forIn_cons, hr]; exact hv⟩

theorem range_forIn_ok {ε σ : Type} (k : ℕ) (s : σ) (f : ℕ → σ → Except ε (ForInStep σ))
    (h : ∀ a < k, ∀ r, ∃ r', f a r = .ok (.yield r')) : ∃ v, forIn [:k] s f = .ok v := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact list_forIn_ok f _ s fun a ha r => h a (by simp [List.mem_range'] at ha; omega) r

theorem array_forIn_ok {ε α σ : Type} (A : Array α) (s : σ) (f : α → σ → Except ε (ForInStep σ))
    (h : ∀ a ∈ A.toList, ∀ r, ∃ r', f a r = .ok (.yield r')) : ∃ v, forIn A s f = .ok v := by
  rw [← Array.forIn_toList]
  exact list_forIn_ok f _ s h

theorem bind_ok_of {ε α β : Type} {x : Except ε α} {f : α → Except ε β} (hx : ∃ a, x = .ok a)
    (hf : ∀ a, ∃ b, f a = .ok b) : ∃ b, (x >>= f) = .ok b := by
  obtain ⟨a, rfl⟩ := hx
  exact hf a

/-- `regionValue` (`Statement.lean:354-368`) returns where the region's bytes are its free bits. -/
theorem regionValue_exists (st : Stmt) (reg : Region) (r : Array F128)
    (hb : ∀ b < 2 ^ st.nbl, (reg.bytes b).size * 8 = 2 ^ reg.free.size) : ∃ v, st.regionValue reg r = .ok v := by
  unfold Stmt.regionValue
  dsimp only
  refine bind_ok_of (range_forIn_ok _ _ _ fun b hb' acc => ?_) fun a => ⟨a, rfl⟩
  simp only [hb b hb', bne_self_eq_false, Bool.false_eq_true, ite_false]
  obtain ⟨s, hs⟩ : ∃ s, (forIn [:(reg.bytes b).size] (0 : F128) fun i s => do
      let s ← forIn [:8] s fun t s =>
        if (((reg.bytes b).get! i >>> t.toUInt8) % 2 == 1) = true then
          pure (ForInStep.yield (s + (eqTable (r.extract 0 reg.free.size))[8 * i + t]!))
        else pure (ForInStep.yield s)
      pure (ForInStep.yield s) : Except String F128) = .ok s := by
    refine range_forIn_ok _ _ _ fun i _ s => ?_
    obtain ⟨s', hs'⟩ := range_forIn_ok 8 s (fun t s =>
        if (((reg.bytes b).get! i >>> t.toUInt8) % 2 == 1) = true then
          (pure (ForInStep.yield (s + (eqTable (r.extract 0 reg.free.size))[8 * i + t]!)) : Except String _)
        else pure (ForInStep.yield s)) fun t _ s => by split <;> exact ⟨_, rfl⟩
    exact ⟨s', by rw [hs']; rfl⟩
  exact ⟨_, by rw [hs]; rfl⟩

/-- `extraClaims` (`Statement.lean:371-377`) returns at every link point array where each region's bytes are its free
bits. -/
theorem extraClaims_exists (st : Stmt) (pts : Array (Array F128)) (hb : RegionBytes st) :
    ∃ e, st.extraClaims pts = .ok e := by
  unfold Stmt.extraClaims
  dsimp only
  refine bind_ok_of (array_forIn_ok _ _ _ fun reg hreg out => ?_) fun a => ⟨a, rfl⟩
  obtain ⟨o, ho⟩ := array_forIn_ok pts out (fun r out => do
      let v ← st.regionValue reg r
      let p := st.regionPoint reg r
      pure (ForInStep.yield (out.push { value := v, skip := eqTable (p.extract 0 6), x := p.extract 6 p.size })))
      fun r _ out => by
    obtain ⟨v, hv⟩ := regionValue_exists st reg r (hb reg hreg)
    exact ⟨_, by rw [hv]; rfl⟩
  exact ⟨o, by rw [ho]; rfl⟩

theorem forall₂_left {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ a ∈ l₁, ∃ b, R a b
  | _, _, .nil, _, h => by cases h
  | _, _, .cons hab h, a, hm => by
    rcases List.mem_cons.1 hm with rfl | hm
    · exact ⟨_, hab⟩
    · exact forall₂_left h a hm

/-- **An accepted session's extra claims give `RegionBytes`**: where `extraClaims` returns at a nonempty point array,
each region's value was computed, so its bytes passed `regionValue`'s check (`Statement.lean:361`). -/
theorem regionBytes_of_extraClaims {st : Stmt} {P : Array (Array F128)} {e : Array Flock.Claim}
    (h : st.extraClaims P = .ok e) (hP : 0 < P.size) : RegionBytes st := by
  obtain ⟨ess, hess, -⟩ := extraClaims_ok st P e h
  intro reg hreg
  obtain ⟨es, hes⟩ := forall₂_left hess reg hreg
  obtain ⟨_, ⟨v, hv, -⟩⟩ := forall₂_left hes P[0] (Array.getElem_mem_toList hP)
  exact (regionValue_ok st reg P[0] v hv).1

/-- An element of `𝔽` as the executable's. -/
def toF128 (a : 𝔽) : F128 := a

/-- **The verifier's setup of a statement realizes its model statement** (`Setup.ofCircuit`, `Verify.lean:29-32`): its
`m` and `k_log` are the statement's, its fold realizes `stmtOf`'s (`ofCircuit_fold`), and at every pair of link points
the arrays holding them give extra claims (`extraClaims_exists`) that realize the model's (`ofCircuit_extra`), §5 and
§16.2. The schedule is `fast100`'s at `m` (whose range gives `14 ≤ m`) and the scheme the statement's. -/
theorem setupRealizes_ofCircuit (st : Stmt) (hwf : StmtWF st) (hr : RegionsWF st) {sch : Flock.Schedule}
    (hms : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") (hf : Flock.fast100 st.m = some sch)
    (hb : RegionBytes st) :
    SetupRealizes (Setup.ofCircuit st) (stmtOf st hwf (regionsOf st)) sch PT_LOCAL st.mPts := by
  refine ⟨rfl, rfl, ofCircuit_fold st hwf _, hms, hf, ?_, hwf.hk, rfl, fun pts => ?_⟩
  · show 14 ≤ st.m
    have := (fast100_range hf).1
    omega
  let P : Array (Array F128) := Array.ofFn fun k : Fin 2 => Array.ofFn fun t : Fin st.mPts => toF128 (pts k t)
  have hPs : P.size = 2 := Array.size_ofFn
  have hPz : ∀ p ∈ P.toList, p.size = st.mPts := by
    intro p hp
    simp only [P, Array.mem_toList_iff, Array.mem_ofFn] at hp
    obtain ⟨k, rfl⟩ := hp
    exact Array.size_ofFn
  have hpts : FlockSoundness.Refine.ptsOf st.mPts P = pts := by
    funext k t
    simp only [FlockSoundness.Refine.ptsOf, P]
    rw [getElem!_pos _ k.val (by simp only [Array.size_ofFn]; exact k.isLt), Array.getElem_ofFn,
      getElem!_pos _ t.val (by simp only [Array.size_ofFn]; exact t.isLt), Array.getElem_ofFn]
    rfl
  obtain ⟨e, he⟩ := extraClaims_exists st P hb
  have hx := ofCircuit_extra st hwf hr P hPs hPz
  rw [hpts] at hx
  refine ⟨P, e, he, hx, fun j hj => ?_⟩
  obtain ⟨hlen, hrel⟩ := hx e he
  have hj' : j < (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts pts).length := hlen ▸ hj
  exact ⟨(hrel j hj').skip_size, (hrel j hj').x_size⟩

/-- **`RunsFrom`'s `foldKLog` and `kLog_le`** (`Runs.lean:312,314`), from a setup realizing the statement
(`SetupRealizes`) at V*'s shape (`ShapeIs`). -/
theorem kLogs_of_setup {st : Setup} {S : Model.Statement} {sch : Flock.Schedule} {ptLocal mPts : ℕ} {sh : Shape}
    (hs : SetupRealizes st S sch ptLocal mPts) (hsh : ShapeIs sh st sch) : st.fold.kLog = sh.kLog ∧ st.kLog ≤ st.m := by
  obtain ⟨-, -, -, -, -, -, hle, hfk, -⟩ := hs
  exact ⟨hfk.trans hsh.2.1.symm, hle⟩

/-- `RunsFrom`'s `foldKLog` and `kLog_le` at the verifier's setup of a statement: `circuitFold`'s `k_log` is the
circuit's (`Statement.lean:342`), and `StmtWF` bounds it by `m`. -/
theorem kLogs_ofCircuit (st : Stmt) (hwf : StmtWF st) :
    (Setup.ofCircuit st).fold.kLog = st.c.kLog ∧ (Setup.ofCircuit st).kLog ≤ (Setup.ofCircuit st).m :=
  ⟨rfl, hwf.hk⟩

/-! ## At the family -/

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- V*'s schedule is the verifier's shape's. -/
theorem sched_eq {s : Flock.Schedule} (hs : F.vsh.sched = .ok s) : s = F.sh.sched := by
  have h := (shape?_eq F.hsh).1
  rw [sched_fast100 hs] at h
  exact Option.some.inj h

/-- **Each rep opens at least one row**: level 0 of `fast100`'s schedule draws 218 queries. -/
theorem nOpen_pos : 0 < F.nOpen := by
  obtain ⟨hsz, hq⟩ := fast100_q0 (shape?_eq F.hsh).1
  unfold nOpen
  obtain ⟨k, hk⟩ : ∃ k, F.sh.sched.levels.size = k + 1 := ⟨_, (Nat.succ_pred_eq_of_pos hsz).symm⟩
  rw [hk, List.range_succ_eq_map, List.map_cons, List.sum_cons]
  exact Nat.lt_of_lt_of_le hq (Nat.le_add_right _ _)

/-- Obligation: discharged by `circuitFile`'s emit and parse round trip.
**The circuit of a level's statement names its unit as the statement**, `RecOpen.unitName LANES H`: `circuitFile`
writes the unit of `recOpenUnit LANES H` under `Stmt.name` (`Stage.lean:44`, `Compose.lean:217,233`) and `HmRow.parse`
reads it as the circuit's unit (`hfile`); established by the emit/parse round trip of `circuitFile`, as `HUnitName`
for the parts. -/
def HOpenName : Prop :=
  ∀ (j : Fin F.stmts.size) ℓ LANES H m, F.stmts[j] = .recOpen ℓ LANES H m →
    (F.Vs j).c.unit.name = Flock.RecOpen.unitName LANES H

/-- **`HOpenUnits`**: level `ℓ`'s statement is statement `ℓ` (`statements_recOpen`), `RecOpen{2^lanes, d - cap}` at the
verifier's schedule (`sched_eq`); its circuit parses (`ZkOuter.hc`) with a unit per VU (`upv_pos`) of `unit_log ≤ 32`
(`unitLog_le`), and the verifier's `RecOpen.check` (`ParseFacts.recNet`) at its name (`HOpenName`) gives its
parameters (`params_of_check`). -/
theorem hOpenUnits_of (hne : Nonempty (Reg × L.Ω × Reg₂)) (hname : F.HOpenName) : F.HOpenUnits := by
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  intro ℓ hℓ
  obtain ⟨s, -, -, hs, -, -, -⟩ := statements_eq F.hstmts
  have hss := F.sched_eq hs
  subst hss
  obtain ⟨hj, hst⟩ := statements_recOpen F.hstmts hs hℓ
  rw [F.hsessO ℓ ⟨ℓ, hj⟩ ⟨_, _, _, hst⟩]
  have hc := (F.zs R ω [] R₂ ⟨ℓ, hj⟩).hc
  exact ⟨upv_pos hc (F.Vs _).hscope, unitLog_le (F.Vs _).hscope, _,
    params_of_check (parse_facts hc).recNet (hname ⟨ℓ, hj⟩ _ _ _ _ hst)⟩

/-- Obligation: discharged by the verifier's check 6, whose pass is `ps.size =
gs.size`.
**The verifier's part-count check passes** (the verifier's gap): at the inner shape `vsh` (the `m`,
`k_log` and claims of the parts' unit names, `RecResiduals.params`), the verifier evaluates V*'s parts,
`VStar.parts (VStar.Algebra.spec vsh)` (`Parts.lean:54-105`), and its own, `groups (structureOf sh) (portSizes sh)`
(`partsOf`, `RecResiduals.lean:119-139`), and refuses when they number differently. Both are the same first-fit packing,
V*'s over the supports it computes (`Algebra.supports`), the verifier's over its symbolic algebra (`structureOf`). -/
def HPartCount : Prop :=
  ∀ sp ps, VStar.Algebra.spec F.vsh = .ok sp → VStar.parts sp = .ok ps → ps.size = (partsOf F.sh).size

/-- **`HCover`**: each of the verifier's parts is one of V*'s (`HPartCount`), which is a statement (`statements_inner`). -/
theorem hCover_of (hc : F.HPartCount) : F.HCover := by
  intro p hp
  obtain ⟨-, sp, ps, -, hsp, hps, -⟩ := statements_eq F.hstmts
  exact statements_inner F.hstmts hsp hps (Nat.lt_of_lt_of_le hp (hc sp ps hsp hps).ge)

/-- **`HSomePart`**: V*'s S has residuals (`spec_res`), so V* has a part (`parts_size`), which is a statement
(`statements_inner`). -/
theorem hSomePart : F.HSomePart := by
  obtain ⟨-, sp, ps, -, hsp, hps, -⟩ := statements_eq F.hstmts
  obtain ⟨j, nm, pt, hj⟩ := statements_inner F.hstmts hsp hps (parts_size hps (spec_res hsp))
  exact ⟨j, 0, nm, pt, hj⟩

/-- **The setup's schedule is `fast100`'s at its `m`**, where V*'s shape is the setup's (`ShapeIs`): `hsh` is at a
`fast100` schedule. `setupRealizes_ofCircuit`'s `hf`. -/
theorem fast100_of_shapeIs {st : Setup} {sch' : Flock.Schedule} (hS : ShapeIs F.sh st sch') :
    Flock.fast100 st.m = some sch' := by
  obtain ⟨hf, hm, -⟩ := shape?_eq F.hsh
  rw [← hS.1, hm, hf, hS.2.2]

/-- **The family's structure facts**: `HSomePart`, `HCover`, `HOpenUnits` and `0 < nOpen`, from the openings' unit
names (`HOpenName`) and V*'s part count (`HPartCount`). -/
theorem structure_of (hne : Nonempty (Reg × L.Ω × Reg₂)) (hname : F.HOpenName) (hcount : F.HPartCount) :
    F.HSomePart ∧ F.HCover ∧ F.HOpenUnits ∧ 0 < F.nOpen :=
  ⟨F.hSomePart, F.hCover_of hcount, F.hOpenUnits_of hne hname, F.nOpen_pos⟩

end VStarFamily

end family

end FlockVBridge.Family
