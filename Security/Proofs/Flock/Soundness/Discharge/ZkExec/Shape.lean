import Proofs.Flock.Soundness.Discharge.ZkLink
import Proofs.Flock.Soundness.Discharge.Exec.Shape

/-!
# The static shape of a compiled `--zk` rep

The coin half of R11b (`Exec.Spine`, and `Discharge/PlaysCoin`'s `decodesOfRunsAt`) needs to know where a rep's coins
fall
among its messages. Without `--zk` every run of a rep follows one spine (`Exec.repC_spine`), because a rep returns at a
reject. The compiled `--zk` rep (`ZkLink.repZCAt`) does not: at an opening reject (an extra claim's ring-switch sum
fails, which is the prover's choice) `restZC` returns at once and `repZC` goes on to the inner proof, so that run's
steps after the opening are the inner proof's, not level 0's. That run ends in a result whose Ligerito part is
`PendL.reject`, whose verdict is `false`, so no accepted outcome has it.

* `ExactOn g σ P`: every run of `g` that ends in an outcome satisfying `P` takes exactly the steps of the kinds of `σ`.
  `playsT_bind_inv` splits a run of a `bind`, and `ExactOn.bind` composes.
* `repShapeZ`: a compiled `--zk` rep's shape, which reads only the table's dimensions and schedule and the number of
  extra claims: the pads root, the masked zerocheck's, lincheck's and opening's shapes (the non-ZK session's `zcShape`,
  `lcShape`, `opShape`), level 0 masked with Ligerito's levels compiled (`ligPadShape`), `final_c'`, and the inner proof
  (`innerShape`).
* `repZCAt_kinds`: every run of rep `r` whose result's Ligerito part is not `PendL.reject` takes exactly the steps of
  `repShapeZ`. `repZCAt_spineAt` is its prefix form at a result (plays-coin's `SpineAt`, unfolded), and
  `ne_reject_of_verdict` gives the hypothesis from a Ligerito verdict that holds.
-/

namespace FlockSoundness.Discharge.ZkExec

open Game Model Accounting ZkSession Exec

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
  {mPts : ℕ} {α β : Type}

/-! ## Runs that end well -/

/-- **A run of a `bind` splits** into a run of the first game and a run of the continuation at its result. -/
theorem playsT_bind_inv {f : α → Game β} (g : Game α) :
    ∀ {ts : List Step} {b : β}, PlaysT (g.bind f) ts b →
      ∃ ts₁ ts₂ a, ts = ts₁ ++ ts₂ ∧ PlaysT g ts₁ a ∧ PlaysT (f a) ts₂ b := by
  induction g with
  | ret a => intro ts b h; exact ⟨[], ts, a, rfl, .ret a, h⟩
  | send M k ih =>
      intro ts b h
      rw [bind_send] at h
      cases h with
      | send m h' =>
          obtain ⟨ts₁, ts₂, a, rfl, h₁, h₂⟩ := ih m h'
          exact ⟨_ :: ts₁, ts₂, a, rfl, .send m h₁, h₂⟩
  | coin C k ih =>
      intro ts b h
      rw [bind_coin] at h
      cases h with
      | coin x h' =>
          obtain ⟨ts₁, ts₂, a, rfl, h₁, h₂⟩ := ih x h'
          exact ⟨_ :: ts₁, ts₂, a, rfl, .coin x h₁, h₂⟩

/-- **Exact on the runs that end well.** Every run of `g` that ends in an outcome satisfying `P` takes exactly the steps
of the kinds of `σ`. -/
def ExactOn (g : Game α) (σ : List Kind) (P : α → Prop) : Prop :=
  ∀ ts a, PlaysT g ts a → P a → ts.map kindOf = σ

theorem ExactOn.of_exact {g : Game α} {σ : List Kind} {P : α → Prop} (h : Exact g σ) : ExactOn g σ P :=
  fun _ _ hp _ => Exact.kinds hp h

theorem ExactOn.bind {g : Game α} {f : α → Game β} {σ₁ σ₂ : List Kind} {Q : α → Prop} {P : β → Prop}
    (hg : ExactOn g σ₁ Q) (hf : ∀ a, ExactOn (f a) σ₂ P) (hQ : ∀ a ts b, PlaysT (f a) ts b → P b → Q a) :
    ExactOn (g.bind f) (σ₁ ++ σ₂) P := by
  intro ts b h hb
  obtain ⟨ts₁, ts₂, a, rfl, h₁, h₂⟩ := playsT_bind_inv g h
  rw [List.map_append, hg _ _ h₁ (hQ a _ _ h₂ hb), hf a _ _ h₂ hb]

theorem ExactOn.bind_exact {g : Game α} {f : α → Game β} {σ₁ σ₂ : List Kind} {P : β → Prop} (hg : Exact g σ₁)
    (hf : ∀ a, ExactOn (f a) σ₂ P) : ExactOn (g.bind f) (σ₁ ++ σ₂) P :=
  ExactOn.bind (Q := fun _ => True) (.of_exact hg) hf fun _ _ _ _ _ => trivial

theorem ExactOn.recv_bind {M : Type} {f : M → Game β} {σ : List Kind} {P : β → Prop}
    (h : ∀ m, ExactOn (f m) σ P) : ExactOn ((recv M).bind f) (.msg M :: σ) P :=
  ExactOn.bind_exact (σ₁ := [.msg M]) (.send fun m => .ret m) h

theorem ExactOn.draw_bind {C : Type} [Fintype C] [Nonempty C] {f : C → Game β} {σ : List Kind} {P : β → Prop}
    (h : ∀ x, ExactOn (f x) σ P) : ExactOn ((draw C).bind f) (.coin C :: σ) P :=
  ExactOn.bind_exact (σ₁ := [.coin C]) (.coin fun x => .ret x) h

/-- A game followed by a return, its outcome mapped by `h`. -/
theorem ExactOn.bind_ret {g : Game α} {h : α → β} {σ : List Kind} {Q : α → Prop} {P : β → Prop}
    (hg : ExactOn g σ Q) (hQ : ∀ a, P (h a) → Q a) : ExactOn (g.bind fun a => .ret (h a)) σ P := by
  have := ExactOn.bind (σ₂ := []) (P := P) hg (fun a => .of_exact (.ret (h a))) fun a _ _ hp hb => by
    obtain ⟨-, rfl⟩ := hp.ret_inv
    exact hQ a hb
  rwa [List.append_nil] at this

theorem ExactOn.map {g : Game α} {h : α → β} {σ : List Kind} {Q : α → Prop} {P : β → Prop} (hg : ExactOn g σ Q)
    (hQ : ∀ a, P (h a) → Q a) : ExactOn (g.map h) σ P :=
  ExactOn.bind_ret hg hQ

/-- A return whose outcome fails `P` has no run that ends well. -/
theorem ExactOn.ret_not {a : α} {σ : List Kind} {P : α → Prop} (h : ¬ P a) : ExactOn (.ret a) σ P := by
  intro ts b hp hb
  obtain ⟨-, rfl⟩ := hp.ret_inv
  exact absurd hb h

/-! ## The masked phases (as the non-ZK session's shapes) -/

omit [DecidableEq F] in
theorem residualRoundsZK_exact {np : ℕ} (P : PadsA np) :
    ∀ (n j : ℕ) (req : Fin n → F) (claim : LinF F np), Exact (residualRoundsZK P n j req claim) (pairRounds F n)
  | 0, _, _, _ => .ret _
  | n + 1, j, req, claim => by
      unfold residualRoundsZK pairRounds
      refine Exact.recv_bind fun g => Exact.draw_bind fun ρ => ?_
      rw [← List.append_nil (pairRounds F n)]
      exact Exact.bind (residualRoundsZK_exact P n _ _ _) fun _ => .ret _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem zerocheckZK_exact (A : Arith F K) (m : ℕ) {np : ℕ} (P : PadsA np) (tr : Triple np) :
    Exact (zerocheckZK A m P tr) (zcShape F m) := by
  unfold zerocheckZK zcShape
  refine Exact.draw_bind fun _ => Exact.draw_bind fun rs => Exact.recv_bind fun msg => Exact.draw_bind fun z => ?_
  exact Exact.bind (residualRoundsZK_exact P _ _ _ _) fun _ => Exact.recv_bind fun v => .ret _

omit [DecidableEq F] in
theorem lincheckRoundsZK_exact {np : ℕ} (P : PadsA np) :
    ∀ (n j : ℕ) (claim : LinF F np), Exact (lincheckRoundsZK P n j claim) (pairRounds F n)
  | 0, _, _ => .ret _
  | n + 1, j, claim => by
      unfold lincheckRoundsZK pairRounds
      refine Exact.recv_bind fun q => Exact.draw_bind fun t => ?_
      rw [← List.append_nil (pairRounds F n)]
      exact Exact.bind (lincheckRoundsZK_exact P n _ _) fun _ => .ret _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem lincheckZK_exact (A : Arith F K) (S : Statement) {np : ℕ} (P : PadsA np) (zc : ZcZK F np S.m) :
    Exact (lincheckZK A S P zc) (lcShape F S.kLog) := by
  unfold lincheckZK lcShape
  refine Exact.draw_bind fun _ => Exact.draw_bind fun _ => ?_
  exact Exact.bind (lincheckRoundsZK_exact P _ _ _) fun _ => Exact.recv_bind fun _ => Exact.draw_bind fun _ => .ret _

omit [DecidableEq F] in
theorem opStep_exact {m : ℕ} (c : Claim F m) : Exact (opStep c) [.msg (Fin 128 → F), .coin (Fin 7 → F)] :=
  Exact.recv_bind fun _ => Exact.draw_bind fun _ => .ret _

omit [DecidableEq F] in
theorem opSteps_exact {m : ℕ} : ∀ cs : List (Claim F m), Exact (cs.mapM opStep) (switches F cs.length)
  | [] => .ret _
  | c :: cs => by
      rw [List.mapM_cons, List.length_cons, switches, ← List.append_nil (switches F cs.length)]
      exact Exact.bind (opStep_exact c) fun _ => Exact.bind (opSteps_exact cs) fun _ => .ret _

omit [Fintype K] [DecidableEq K] in
theorem openingZK_exact (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (a c : Claim F m) (extra : List (Claim F m)) :
    Exact (openingZK A hm a c extra) (opShape F (extra.length + 2)) := by
  have hσ : opShape F (extra.length + 2) = [.msg (Fin 128 → F), .coin (Fin 7 → F)] ++
      ([.msg (Fin 128 → F), .coin (Fin 7 → F)] ++ (switches F extra.length ++
        ([.coin (Fin (a :: c :: extra).length → F)] ++ []))) := by
    simp [opShape, switches]
  rw [hσ]
  unfold openingZK
  exact Exact.bind (opStep_exact a) fun _ => Exact.bind (opStep_exact c) fun _ =>
    Exact.bind (opSteps_exact extra) fun _ => Exact.draw_bind fun _ => .ret _

/-! ## Level 0 masked, Ligerito's levels compiled -/

/-- `k` masked folding rounds (`foldRoundsZK`): a coin of two words, then for each further round a message pair of `K`
and a coin. -/
def foldShapeZ (F K : Type) : ℕ → List Kind
  | 0 => []
  | 1 => [.coin (Fin 2 → F)]
  | k + 2 => .coin (Fin 2 → F) :: .msg (K × K) :: foldShapeZ F K (k + 1)

/-- Levels 1+ of the masked level 0 (`ligTailC`): the code-switch message, then, with a further level, its cap, its
out-of-domain samples, level 0's queries and the recursive levels' rounds. -/
def tailShape (F K D : Type) (sch : Schedule) (lv₀ : Level) : List Level → List Kind
  | [] => [.msg (K × K)]
  | lv₁ :: rest => .msg (K × K) :: .msg (Cap D) ::
      (oodShape F K (sch.m - 7 - sch.k0 + 1) ++ (qbShape F K lv₀ true ++ recShape F K D sch.ext (lv₁ :: rest)))

/-- Level 0 masked with Ligerito's levels compiled (`ligeritoPadZC`), read off the schedule's levels, `e` and `t`. -/
def ligPadShape (F K D : Type) (sch : Schedule) (e t : ℕ) : List Kind :=
  match sch.levels with
  | [] => []
  | lv₀ :: rest => .coin (Fin (sch.m - 7) → F) :: .msg F :: .coin F :: .msg (K × K) ::
      (foldShapeZ F K sch.k0 ++ .msg (Fin e → K) :: (List.replicate e (.coin (Fin 2 → F)) ++
        .msg K :: .msg (Fin t → K) :: tailShape F K D sch lv₀ rest))

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem drawK_exact (A : Arith F K) : Exact (drawK A) [.coin (Fin 2 → F)] :=
  Exact.draw_bind fun _ => .ret _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem foldRoundsZK_exact (A : Arith F K) {np e : ℕ} (P : Pads0 np e) :
    ∀ (k j : ℕ) (s : LinK K np) (q : QuadL K np), Exact (foldRoundsZK A P k j s q) (foldShapeZ F K k)
  | 0, _, _, _ => .ret _
  | 1, _, _, q => by
      unfold foldRoundsZK foldShapeZ
      exact Exact.bind_one (drawK_exact A) fun _ => .ret _
  | k + 2, j, _, q => by
      unfold foldRoundsZK foldShapeZ
      refine Exact.bind_one (drawK_exact A) fun r => Exact.recv_bind fun msg => ?_
      rw [← List.append_nil (foldShapeZ F K (k + 1))]
      exact Exact.bind (foldRoundsZK_exact A P (k + 1) _ _ _) fun _ => .ret _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem drawKs_exact (A : Arith F K) : ∀ e, Exact (drawKs A e) (List.replicate e (.coin (Fin 2 → F)))
  | 0 => .ret _
  | e + 1 => by
      unfold drawKs
      rw [List.replicate_succ, ← List.append_nil (List.replicate e _)]
      exact Exact.bind_one (drawK_exact A) fun _ => Exact.bind (drawKs_exact A e) fun _ => .ret _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem queryAndBatchPadC_exact (A : Arith F K) (r : ℕ) (lv : Level) (col : List F → ℕ → K) (start : ℕ)
    (st : LigStateC F K D) : Exact (ZkLink.queryAndBatchPadC A r lv col start st) (qbShape F K lv true) := by
  unfold ZkLink.queryAndBatchPadC qbShape
  exact Exact.draw_bind fun _ => Exact.draw_bind fun _ => Exact.recv_bind fun _ =>
    Exact.draw_bind (C := F) fun _ => .ret _

omit [DecidableEq F] [Fintype K] in
/-- Compiled Ligerito's recursive levels take exactly their shape: `recursiveRoundsC` never stops early. -/
theorem recursiveRoundsC_exact (A : Arith F K) (r n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ) :
    ∀ (levels : List Level) (st : LigStateC F K D),
      Exact (recursiveRoundsC A r n b initial ext levels st) (recShape F K D ext levels)
  | [], _ => .ret _
  | [lv], st => by
      unfold recursiveRoundsC recShape
      refine Exact.bind (foldRoundsC_exact A _ _ _) fun _ => Exact.recv_bind fun yr => ?_
      rw [← List.append_nil (qbShape F K lv false)]
      exact Exact.bind (queryAndBatchC_exact A r lv _ _ _ false) fun _ => .ret _
  | lv :: lv' :: rest, st => by
      unfold recursiveRoundsC
      rw [recShape]
      refine Exact.bind (foldRoundsC_exact A _ _ _) fun _ => Exact.recv_bind fun C' => ?_
      refine Exact.bind (twoOodC_exact A _ _ _ _) fun _ => ?_
      exact Exact.bind (queryAndBatchC_exact A r lv _ _ _ true) fun _ =>
        recursiveRoundsC_exact A r n b initial ext (lv' :: rest) _

omit [DecidableEq F] [Fintype K] in
theorem ligTailC_on (A : Arith F K) (r : ℕ) (sch : Schedule) (t : ℕ) (c₀ : F) (e : ℕ) (b : ℕ → F) (lv₀ : Level)
    (rest : List Level) (z₀ : Fin (sch.m - 7) → F) (β₀ : F) (initial : List K) (ρ : Fin e → K) (ybar : Fin t → K)
    (s : K) :
    ExactOn (ZkLink.ligTailC (D := D) A r sch t c₀ e b lv₀ rest z₀ β₀ initial ρ ybar s) (tailShape F K D sch lv₀ rest)
      (· ≠ PendL.reject) := by
  unfold ZkLink.ligTailC
  rcases rest with _ | ⟨lv₁, rest'⟩
  · exact ExactOn.recv_bind fun _ => ExactOn.ret_not fun h => h rfl
  · refine ExactOn.recv_bind fun _ => ExactOn.recv_bind fun _ => ?_
    refine ExactOn.bind_exact (twoOodC_exact A _ _ _ _) fun _ => ?_
    exact ExactOn.bind_exact (queryAndBatchPadC_exact A r lv₀ _ _ _) fun _ =>
      .of_exact (recursiveRoundsC_exact A r _ b initial sch.ext (lv₁ :: rest') _)

omit [DecidableEq F] [Fintype K] in
theorem ligeritoPadZC_on (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (sch : Schedule) (t : ℕ) (c₀ : F) (r : ℕ)
    (T : F) (b : ℕ → F) :
    ExactOn (ZkLink.ligeritoPadZC (D := D) A P sch t c₀ r T b) (ligPadShape F K D sch e t)
      (fun x => x.2 ≠ PendL.reject) := by
  unfold ZkLink.ligeritoPadZC ligPadShape
  rcases sch.levels with _ | ⟨lv₀, rest⟩
  · exact ExactOn.ret_not fun h => h rfl
  · refine ExactOn.draw_bind fun _ => ExactOn.recv_bind fun _ => ExactOn.draw_bind fun _ =>
      ExactOn.recv_bind fun _ => ?_
    refine ExactOn.bind_exact (foldRoundsZK_exact A P _ _ _ _) fun _ => ExactOn.recv_bind fun _ => ?_
    refine ExactOn.bind_exact (drawKs_exact A e) fun _ => ExactOn.recv_bind fun _ => ExactOn.recv_bind fun _ => ?_
    exact ExactOn.bind_ret (ligTailC_on A r sch t c₀ e b lv₀ rest _ _ _ _ _ _) fun _ h => h

omit [Fintype K] in
theorem restZC_on (A : Arith F K) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np) (S : Statement) (sch : Schedule)
    (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (extra : List (Claim F S.m)) (zc : ZcZK F np S.m)
    (ab : ClaimZK F np S.m) :
    ExactOn (ZkLink.restZC (D := D) A P₀ PA S sch hm t c₀ r extra zc ab)
      (opShape F (extra.length + 2) ++ (ligPadShape F K D sch e t ++ [.msg F])) (fun x => x.2 ≠ PendL.reject) := by
  unfold ZkLink.restZC
  refine ExactOn.bind_exact (openingZK_exact A _ _ _ extra) fun o => ?_
  rcases o with ⟨_ | op, ca, cc⟩
  · exact ExactOn.ret_not fun h => h rfl
  · refine ExactOn.bind (Q := fun x => x.2 ≠ PendL.reject) (ligeritoPadZC_on A P₀ sch t c₀ r _ _)
      (fun _ => ExactOn.recv_bind fun _ => .of_exact (.ret _)) fun lv ts x hp hx => ?_
    simp only [recv] at hp
    cases hp with
    | send _ h' =>
        obtain ⟨-, rfl⟩ := h'.ret_inv
        exact hx

omit [Fintype K] in
theorem repMaskedZC_on (A : Arith F K) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np) (tr : Triple np)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (extra : List (Claim F S.m)) :
    ExactOn (ZkLink.repMaskedZC (D := D) A P₀ PA tr S sch hm t c₀ r extra)
      (zcShape F S.m ++ (lcShape F S.kLog ++ (opShape F (extra.length + 2) ++ (ligPadShape F K D sch e t ++ [.msg F]))))
      (fun x => x.2 ≠ PendL.reject) := by
  unfold ZkLink.repMaskedZC
  refine ExactOn.map (Q := fun x => x.2 ≠ PendL.reject) ?_ fun _ h => h
  refine ExactOn.bind_exact (zerocheckZK_exact A S.m PA tr) fun o => ExactOn.map (Q := fun x => x.2 ≠ PendL.reject)
    ?_ fun _ h => h
  exact ExactOn.bind_exact (lincheckZK_exact A S PA o.1) fun o' =>
    ExactOn.map (restZC_on A P₀ PA S sch hm t c₀ r extra o.1 o'.1) fun _ h => h

/-! ## The inner proof, and a whole rep -/

/-- The compiled inner proof's shape (`innerGameC`) on `nA` rows: `ε`, `[ρ, σ]`, `γ`, `τ`'s commitment, `β`, `Y` and the
queries. -/
def innerShape (F D : Type) (nA s nP q : ℕ) : List Kind :=
  [.coin F, .msg (F × F), .coin (Fin (nA + 3) → F), .msg D, .coin F, .msg (Fin (s + nP) → F), .coin (Fin q → F)]

omit [DecidableEq F] in
theorem innerGameC_exact (Hs : ZkLink.Hashes F D Sl) {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N)
    {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s) (root : D) :
    Exact (ZkLink.innerGameC Hs.H Hs.Ep Hs.Lp Hs.Lτ dom s nP q pos rows tr root) (innerShape F D nA s nP q) := by
  unfold ZkLink.innerGameC innerShape
  exact Exact.draw_bind fun _ => Exact.recv_bind fun _ => Exact.draw_bind fun _ => Exact.recv_bind fun _ =>
    Exact.draw_bind fun _ => Exact.recv_bind fun _ => Exact.draw_bind fun _ => .ret _

/-- **A compiled `--zk` rep's shape**, on `nExtra` extra claims: the pads root, the masked zerocheck's, lincheck's and
opening's shapes, level 0 masked with Ligerito's levels compiled, `final_c'`, and the inner proof on seven rows. It
reads only `T`'s dimensions (`S.m`, `S.kLog`, `e`, `t`, `np`, `nP`, `q`) and schedule. -/
def repShapeZ (F K D : Type) [Field F] [Fintype F] [DecidableEq F] (T : TabZK F mPts) (nExtra : ℕ) : List Kind :=
  .msg D :: ((zcShape F T.S.m ++ (lcShape F T.S.kLog ++ (opShape F (nExtra + 2) ++
    (ligPadShape F K D T.sch T.e T.t ++ [.msg F])))) ++ innerShape F D 7 T.np T.nP T.q)

omit [Fintype K] in
theorem repZC_on (A : Arith F K) (Hs : ZkLink.Hashes F D Sl) (T : TabZK F mPts) (r : ℕ)
    (extra : List (Claim F T.S.m)) :
    ExactOn (ZkLink.repZC A Hs T r extra) (repShapeZ F K D T extra.length) (fun p => p.lig ≠ PendL.reject) := by
  unfold ZkLink.repZC repShapeZ
  refine ExactOn.recv_bind fun root => ExactOn.bind (Q := fun o => o.2 ≠ PendL.reject)
    (repMaskedZC_on A T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r extra)
    (fun o => ExactOn.bind_ret (Q := fun _ => True) (.of_exact (innerGameC_exact Hs _ _ _ _ _ _ _ _))
      fun _ _ => trivial) fun o ts p hp hpz => ?_
  simp only [bind_eq, pure_eq] at hp
  obtain ⟨_, _, _, _, _, h₂⟩ := playsT_bind_inv _ hp
  obtain ⟨-, rfl⟩ := h₂.ret_inv
  exact hpz

omit [Fintype K] in
/-- **A compiled `--zk` rep's runs that can be accepted have one shape.** Every run of rep `r` at the link points `pts`
whose result's Ligerito part is not `PendL.reject` takes exactly the steps of `repShapeZ`, which reads only the table's
dimensions and schedule and the number of extra claims. -/
theorem repZCAt_kinds (A : Arith F K) (Hs : ZkLink.Hashes F D Sl) (T : TabZK F mPts) (pts : Fin 2 → Fin mPts → F)
    (r : Fin 2) {ts : List Step} {p : ZkLink.PendZ F D Sl} (h : PlaysT (ZkLink.repZCAt A Hs T pts r) ts p)
    (hp : p.lig ≠ PendL.reject) :
    ts.map kindOf = repShapeZ F K D T (extraClaims T.S T.ptLocal mPts pts).length :=
  repZC_on A Hs T r.val _ ts p h hp

omit [Fintype K] in
/-- **A spine of a compiled `--zk` rep at a result that can be accepted**: every run of rep `r` that ends in `p` takes
steps whose kinds, in order, are a prefix of `repShapeZ` (`Discharge/PlaysCoin`'s `SpineAt … p`, unfolded). -/
theorem repZCAt_spineAt (A : Arith F K) (Hs : ZkLink.Hashes F D Sl) (T : TabZK F mPts) (pts : Fin 2 → Fin mPts → F)
    (r : Fin 2) {p : ZkLink.PendZ F D Sl} (hp : p.lig ≠ PendL.reject) :
    ∀ ts, PlaysT (ZkLink.repZCAt A Hs T pts r) ts p →
      ts.map kindOf <+: repShapeZ F K D T (extraClaims T.S T.ptLocal mPts pts).length := by
  intro ts h
  rw [repZCAt_kinds A Hs T pts r h hp]

omit [Field F] [Fintype F] [DecidableEq F] in
/-- A Ligerito result whose verdict holds on some openings is not a reject. -/
theorem ne_reject_of_verdict {p : PendL F D} {o : Opens F D} (h : p.verdict o = true) : p ≠ PendL.reject := by
  rintro rfl
  exact Bool.false_ne_true h

end FlockSoundness.Discharge.ZkExec
