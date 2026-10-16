import Proofs.Flock.Claims.Complete.Json

/-!
# The honest session runs

`sessionRuns_goal`: the goal `SessionRuns`, given `StmtOk` (no keyed coin tree, so the record's `hello` is the
statement's) and one named hypothesis on the prover's pieces, `ProverRuns`: at every good case's table, the level-0
commitment returns, the table's extra claims at the link's points return, and each rep returns, on that commitment and
those claims, from any sends.

`Runs y l P`: `y` returns from the sends `l`, with `P` of its value and the sends after; `Runs.bind`, `Runs.forIn_array`
and `Runs.mapM_singleton` walk the session.
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy

/-- `y` returns from the sends `l`, with `P` of its value and the sends after. -/
def Runs {α : Type} (y : PS α) (l : Array Claims.Msg) (P : α → Array Claims.Msg → Prop) : Prop :=
  ∃ a l', y.go l = (.ok a, l') ∧ P a l'

theorem Runs.pure {α : Type} {a : α} {l : Array Claims.Msg} {P : α → Array Claims.Msg → Prop} (h : P a l) :
    Runs (Pure.pure a : PS α) l P := ⟨a, l, rfl, h⟩

theorem Runs.bind {α β : Type} {x : PS α} {f : α → PS β} {l : Array Claims.Msg} {Q : α → Array Claims.Msg → Prop}
    {P : β → Array Claims.Msg → Prop} (hx : Runs x l Q) (hf : ∀ a l', Q a l' → Runs (f a) l' P) :
    Runs (x >>= f) l P := by
  obtain ⟨a, l', e, hq⟩ := hx
  obtain ⟨b, l'', e', hp⟩ := hf a l' hq
  refine ⟨b, l'', ?_, hp⟩
  rw [go_bind, e]
  exact e'

theorem Runs.of_go {α : Type} {y : PS α} {l l' : Array Claims.Msg} {a : α} {P : α → Array Claims.Msg → Prop}
    (e : y.go l = (.ok a, l')) (h : P a l') : Runs y l P := ⟨a, l', e, h⟩

theorem Runs.forIn_list {α β : Type} (f : α → β → PS (ForInStep β)) (s : List α)
    (inv : β → Array Claims.Msg → Prop)
    (hf : ∀ a ∈ s, ∀ b l, inv b l → Runs (f a b) l fun r l' => ∃ b', r = .yield b' ∧ inv b' l') :
    ∀ xs : List α, (∀ a ∈ xs, a ∈ s) → ∀ b l, inv b l → Runs (forIn xs b f) l inv
  | [], _, b, l, h => by
    rw [List.forIn_nil]
    exact Runs.pure h
  | a :: xs, hx, b, l, h => by
    rw [List.forIn_cons]
    refine Runs.bind (hf a (hx a List.mem_cons_self) b l h) fun r l' hr => ?_
    obtain ⟨b', rfl, hb'⟩ := hr
    exact Runs.forIn_list f s inv hf xs (fun x hx' => hx x (List.mem_cons_of_mem a hx')) b' l' hb'

theorem Runs.forIn_array {α β : Type} (xs : Array α) (f : α → β → PS (ForInStep β))
    (inv : β → Array Claims.Msg → Prop)
    (hf : ∀ a ∈ xs, ∀ b l, inv b l → Runs (f a b) l fun r l' => ∃ b', r = .yield b' ∧ inv b' l')
    {b : β} {l : Array Claims.Msg} (h : inv b l) : Runs (forIn xs b f) l inv := by
  rw [← Array.forIn_toList]
  exact Runs.forIn_list f xs.toList inv (fun a ha => hf a (Array.mem_toList_iff.1 ha)) xs.toList (fun _ h => h) b l h

theorem Runs.mapM_singleton {α β : Type} (f : α → PS β) (a : α) {l : Array Claims.Msg}
    {P : Array β → Array Claims.Msg → Prop} (h : Runs (f a) l fun b l' => P #[b] l') : Runs (#[a].mapM f) l P := by
  rw [Array.mapM_eq_mapM_toList]
  simp only [List.mapM_cons, List.mapM_nil, bind_pure_comp, map_pure, Functor.map_map]
  refine Runs.bind (Q := fun b l' => P #[b] l') h fun b l' hb => ?_
  exact Runs.pure hb

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The prover's pieces return**, at every good case's table with its schedule: the level-0 commitment of the
trace's packed table, under any salts; the table's extra claims at the link's points; and each rep on such a commitment
and those claims, under any salts, nonces and domain, from any sends. -/
def ProverRuns (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ c : Case H rel, ∀ st sch, c.sts = #[st] → fast100 (Setup.ofCircuit st).m = some sch →
    (∀ salt₀, ∃ cm, commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle
      (sch.levels.getD 0 default) (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) salt₀ = .ok cm) ∧
    (∃ e, (Setup.ofCircuit st).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e) ∧
    ∀ salt₀ cm e, commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle
      (sch.levels.getD 0 default) (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) salt₀ = .ok cm →
    (Setup.ofCircuit st).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e →
    ∀ salt nonce d l, ∃ pt l', ((repP (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) sch
      (tracesOf c.sts c.w)[0]! (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) cm e salt nonce).run
      (Transcript.init #[] d)).go l = (.ok pt, l')

/-- **The session runs**: the goal `SessionRuns`. -/
theorem sessionRuns_goal (hS : StmtOk H rel) (hR : ProverRuns H rel) : SessionRuns H rel := by
  intro c
  obtain ⟨st, hst, hsp, hm1, hm2⟩ := c.table
  obtain ⟨hct, -, -, -⟩ := hS c
  have ht : c.spec.tables = #[st.tags.table] := by rw [hsp]; rfl
  obtain ⟨sch, hsch⟩ := Option.isSome_iff_exists.1 (fast100_isSome (m := (Setup.ofCircuit st).m) hm1 hm2)
  obtain ⟨hC, ⟨e, he⟩, hRep⟩ := hR c st sch hst hsch
  obtain ⟨cm, hcm⟩ := hC (c.ω.salt 0 0 0)
  rw [hst] at hcm hRep
  suffices h : Runs (sessionP (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) c.spec #[Setup.ofCircuit st]
      (tracesOf #[st] c.w) c.ω) #[] fun _ _ => True by
    obtain ⟨⟨record, proofs⟩, bodies, e, -⟩ := h
    refine ⟨record, proofs, bodies, ?_⟩
    unfold sessionS
    rw [hst, show tabsOf #[st] = #[Setup.ofCircuit st] by simp [tabsOf]]
    exact e
  unfold sessionP
  refine Runs.bind (Q := fun schs l => schs = #[sch] ∧ l = #[]) (Runs.mapM_singleton _ _ ?_) fun schs l hq => ?_
  · simp only [hsch]
    exact Runs.pure ⟨rfl, rfl⟩
  obtain ⟨rfl, rfl⟩ := hq
  dsimp only
  refine Runs.bind (Q := fun cms l => cms = #[cm] ∧ l = #[]) ?_ fun cms l hq => ?_
  · rw [show Array.range (#[Setup.ofCircuit st] : Array Setup).size = #[0] from rfl]
    refine Runs.mapM_singleton _ _ ⟨cm, #[], ?_, rfl, rfl⟩
    rw [commitP_pureS]
    simpa using hcm
  obtain ⟨rfl, rfl⟩ := hq
  refine Runs.bind (Q := fun c1 _ => c1 = coinOf (c.resp 0)) (Runs.of_go (link_go _ _ _ _) rfl) fun c1 l hq => ?_
  subst hq
  have he' : (Setup.ofCircuit st).extra (ptsOf c.spec (coinOf (c.resp 0))) = .ok e := by
    rw [show c.resp 0 = (Claims.coinSeq c.r 0).bytes from rfl, ptsOf_coin]
    exact he
  refine Runs.bind (Q := fun _ _ => True) (Runs.forIn_array _ _ (fun _ _ => True) ?_ trivial) fun b l _ => ?_
  · rintro ⟨i, r, name⟩ hmem ⟨ss, ps⟩ l -
    have hi : i = 0 := by
      rw [repStreams_one ht] at hmem
      simp at hmem
      omega
    subst hi
    dsimp only
    refine Runs.bind (Q := fun ex _ => ex = e) ⟨e, l, ?_, rfl⟩ fun ex l hq => ?_
    · rw [monadLift_go]
      show ((Setup.ofCircuit st).extra (ptsOf c.spec (coinOf (c.resp 0))), l) = _
      rw [he']
    subst ex
    refine Runs.bind (Q := fun _ _ => True) ?_ fun pt l _ => ?_
    · obtain ⟨pt, l', e'⟩ := hRep (c.ω.salt 0 0 0) cm e hcm he (c.ω.salt 0 r) (c.ω.nonce 0 r)
        (c.spec.domain c.spec.tables[0]! r).toUTF8 l
      refine ⟨pt, l', ?_, trivial⟩
      simpa using e'
    obtain ⟨p, t⟩ := pt
    exact Runs.pure ⟨_, rfl, trivial⟩
  obtain ⟨ss, ps⟩ := b
  dsimp only
  rw [recordJson_eq _ hct]
  simp only [pure_bind]
  exact Runs.pure trivial

end FlockComplete.JsonRT
