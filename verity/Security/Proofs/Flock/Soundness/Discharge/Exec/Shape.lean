import Proofs.Flock.Soundness.Refine.LiveSim
import Proofs.Flock.Soundness.Game.Union

/-!
# The static shape of a rep

`DecodesAll` (`Live.lean`) asks for each rep's run on a tape cut into rounds: the messages a round's bytes decode to,
then the round's coin. Where a run's coins fall among its messages is the game's, not the run's: every game of a rep
takes its steps in an order fixed by the statement and the schedule, and no step's kind depends on the values before
it. A rep may only stop early, on a reject.

* `Kind`, `kindOf`: a tape step's kind, a message or a coin, with its type.
* `Spine g σ`: every run of `g` takes steps of the kinds of `σ`, in order, until it returns. `Exact g σ`: and it
  returns exactly at the end of `σ`.
* `Spine.prefix`, `Exact.kinds`: a run's kinds are a prefix of `σ` (all of `σ`, for `Exact`).
* `zerocheck_exact`, `lincheck_exact`, `opening_exact`: the PIOP's phases have exact shapes; `ligeritoC_spine` and
  `repC_spine`: compiled Ligerito and a whole rep have spines, `repShape`, which read only the dimensions, the number
  of claims and the schedule.
-/

namespace FlockSoundness.Discharge.Exec

open FlockSoundness.Game FlockSoundness.Accounting FlockSoundness.Model

/-- A tape step's kind: a message of type `M`, or a coin of type `C`. -/
inductive Kind : Type 1
  | msg (M : Type)
  | coin (C : Type)

/-- The kind of a step. -/
def kindOf : Step → Kind
  | .inl x => .msg x.1
  | .inr x => .coin x.1

/-- **A spine.** Every run of `g` takes steps of the kinds of `σ`, in order, until it returns; it may return before
the end of `σ`. -/
inductive Spine : {α : Type} → Game α → List Kind → Prop
  | ret {α : Type} (a : α) (σ : List Kind) : Spine (.ret a) σ
  | send {α M : Type} {k : M → Game α} {σ : List Kind} : (∀ m, Spine (k m) σ) → Spine (.send M k) (.msg M :: σ)
  | coin {α C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} {σ : List Kind} :
      (∀ x, Spine (k x) σ) → Spine (@Game.coin α C iF iN k) (.coin C :: σ)

/-- **An exact shape.** Every run of `g` takes exactly the steps of the kinds of `σ`, in order. -/
inductive Exact : {α : Type} → Game α → List Kind → Prop
  | ret {α : Type} (a : α) : Exact (.ret a) []
  | send {α M : Type} {k : M → Game α} {σ : List Kind} : (∀ m, Exact (k m) σ) → Exact (.send M k) (.msg M :: σ)
  | coin {α C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} {σ : List Kind} :
      (∀ x, Exact (k x) σ) → Exact (@Game.coin α C iF iN k) (.coin C :: σ)

variable {α β : Type}

theorem Exact.spine {g : Game α} {σ : List Kind} (h : Exact g σ) : Spine g σ := by
  induction h with
  | ret a => exact .ret a []
  | send _ ih => exact .send ih
  | coin _ ih => exact .coin ih

theorem Exact.bind {g : Game α} {f : α → Game β} {σ₁ σ₂ : List Kind} (h₁ : Exact g σ₁) (h₂ : ∀ a, Exact (f a) σ₂) :
    Exact (g >>= f) (σ₁ ++ σ₂) := by
  induction h₁ with
  | ret a => exact h₂ a
  | send _ ih => exact .send ih
  | coin _ ih => exact .coin ih

theorem Exact.bindS {g : Game α} {f : α → Game β} {σ₁ σ₂ : List Kind} (h₁ : Exact g σ₁) (h₂ : ∀ a, Spine (f a) σ₂) :
    Spine (g >>= f) (σ₁ ++ σ₂) := by
  induction h₁ with
  | ret a => exact h₂ a
  | send _ ih => exact .send ih
  | coin _ ih => exact .coin ih

theorem Exact.bind_one {g : Game α} {f : α → Game β} {c : Kind} {σ : List Kind} (h₁ : Exact g [c])
    (h₂ : ∀ a, Exact (f a) σ) : Exact (g >>= f) (c :: σ) := Exact.bind h₁ h₂

theorem Exact.recv_bind {M : Type} {f : M → Game β} {σ : List Kind} (h : ∀ m, Exact (f m) σ) :
    Exact (recv M >>= f) (.msg M :: σ) := .send h

theorem Exact.draw_bind {C : Type} [Fintype C] [Nonempty C] {f : C → Game β} {σ : List Kind}
    (h : ∀ x, Exact (f x) σ) : Exact (draw C >>= f) (.coin C :: σ) := .coin h

theorem Spine.recv_bind {M : Type} {f : M → Game β} {σ : List Kind} (h : ∀ m, Spine (f m) σ) :
    Spine (recv M >>= f) (.msg M :: σ) := .send h

theorem Spine.draw_bind {C : Type} [Fintype C] [Nonempty C] {f : C → Game β} {σ : List Kind}
    (h : ∀ x, Spine (f x) σ) : Spine (draw C >>= f) (.coin C :: σ) := .coin h

/-- **A run follows its game's spine**: its kinds, in order, are a prefix of the spine. -/
theorem Spine.prefix {g : Game α} {ts : List Step} {a : α} (hp : PlaysT g ts a) :
    ∀ {σ : List Kind}, Spine g σ → ts.map kindOf <+: σ := by
  induction hp with
  | ret a => exact fun _ => List.nil_prefix
  | send m _ ih =>
      intro σ hs
      cases hs with
      | send hk => exact List.cons_prefix_cons.mpr ⟨rfl, ih (hk m)⟩
  | coin x _ ih =>
      intro σ hs
      cases hs with
      | coin hk => exact List.cons_prefix_cons.mpr ⟨rfl, ih (hk x)⟩

/-- Messages and coin steps interleaved along kinds: a message where the kind is a message's, a coin where it is a
coin's, until a list runs out. -/
def weave : List Kind → List Msg → List Msg → List Step
  | .msg _ :: σ, m :: ms, cs => .inl m :: weave σ ms cs
  | .coin _ :: σ, ms, c :: cs => .inr c :: weave σ ms cs
  | _, _, _ => []

theorem weave_self : ∀ ts : List Step,
    weave (ts.map kindOf) (ts.filterMap Sum.getLeft?) (ts.filterMap Sum.getRight?) = ts
  | [] => rfl
  | .inl x :: ts => by
      rw [List.map_cons, List.filterMap_cons, List.filterMap_cons, Sum.getLeft?_inl, Sum.getRight?_inl]
      exact congrArg (_ :: ·) (weave_self ts)
  | .inr x :: ts => by
      rw [List.map_cons, List.filterMap_cons, List.filterMap_cons, Sum.getLeft?_inr, Sum.getRight?_inr]
      exact congrArg (_ :: ·) (weave_self ts)

/-- **A run's tape is its messages and coins woven along the spine.** Where the coins fall among the messages is the
spine's, not the run's. -/
theorem Spine.tape {g : Game α} {ts : List Step} {a : α} {σ : List Kind} (hp : PlaysT g ts a) (hs : Spine g σ) :
    ts = weave (σ.take ts.length) (ts.filterMap Sum.getLeft?) (ts.filterMap Sum.getRight?) := by
  have h := List.prefix_iff_eq_take.mp (Spine.prefix hp hs)
  rw [List.length_map] at h
  rw [← h]
  exact (weave_self ts).symm

/-- **A run takes its game's exact shape.** -/
theorem Exact.kinds {g : Game α} {ts : List Step} {a : α} (hp : PlaysT g ts a) :
    ∀ {σ : List Kind}, Exact g σ → ts.map kindOf = σ := by
  induction hp with
  | ret a =>
      intro σ hs
      cases hs
      rfl
  | send m _ ih =>
      intro σ hs
      cases hs with
      | send hk => exact congrArg (_ :: ·) (ih (hk m))
  | coin x _ ih =>
      intro σ hs
      cases hs with
      | coin hk => exact congrArg (_ :: ·) (ih (hk x))

/-! ## The PIOP's phases -/

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}

/-- `n` rounds of a message pair and a coin: the zerocheck's residual rounds (Z6) and the lincheck's rounds (L3). -/
def pairRounds (F : Type) : ℕ → List Kind
  | 0 => []
  | n + 1 => .msg (F × F) :: .coin F :: pairRounds F n

omit [DecidableEq F] in
theorem residualRounds_exact : ∀ (n : ℕ) (req : Fin n → F) (claim : F),
    Exact (residualRounds n req claim) (pairRounds F n)
  | 0, _, _ => .ret _
  | n + 1, req, claim => by
      unfold residualRounds pairRounds
      refine Exact.recv_bind fun g => Exact.draw_bind fun ρ => ?_
      rw [← List.append_nil (pairRounds F n)]
      exact Exact.bind (residualRounds_exact n _ _) fun r => .ret _

omit [DecidableEq F] in
theorem lincheckRounds_exact : ∀ (n : ℕ) (claim : F), Exact (lincheckRounds n claim) (pairRounds F n)
  | 0, _ => .ret _
  | n + 1, claim => by
      unfold lincheckRounds pairRounds
      refine Exact.recv_bind fun q => Exact.draw_bind fun t => ?_
      rw [← List.append_nil (pairRounds F n)]
      exact Exact.bind (lincheckRounds_exact n _) fun r => .ret _

/-- The zerocheck's shape (Z2–Z7). -/
def zcShape (F : Type) (m : ℕ) : List Kind :=
  .coin (Fin 6 → F) :: .coin (Fin (m - 13) → F) :: .msg ((Fin 64 → F) × (Fin 64 → F)) :: .coin F ::
    (pairRounds F (m - 6) ++ [.msg (F × F)])

omit [Fintype K] [DecidableEq K] in
theorem zerocheck_exact (A : Arith F K) (m : ℕ) : Exact (zerocheck A m) (zcShape F m) := by
  unfold zerocheck zcShape
  refine Exact.draw_bind fun _ => Exact.draw_bind fun rs => Exact.recv_bind fun msg => Exact.draw_bind fun z => ?_
  exact Exact.bind (residualRounds_exact _ _ _) fun r => Exact.recv_bind fun v => .ret _

/-- The lincheck's shape (L1–L5). -/
def lcShape (F : Type) (kLog : ℕ) : List Kind :=
  .coin F :: .coin F :: (pairRounds F (kLog - 6) ++ [.msg (Fin 64 → F), .coin F])

omit [Fintype K] [DecidableEq K] in
theorem lincheck_exact (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) :
    Exact (lincheck A S zc) (lcShape F S.kLog) := by
  unfold lincheck lcShape
  refine Exact.draw_bind fun α => Exact.draw_bind fun β => ?_
  exact Exact.bind (lincheckRounds_exact _ _) fun r => Exact.recv_bind fun zP => Exact.draw_bind fun r' => .ret _

/-- `n` ring switches: each a message `ŝ` and seven coins `r''` (O3, O5). -/
def switches (F : Type) : ℕ → List Kind
  | 0 => []
  | n + 1 => .msg (Fin 128 → F) :: .coin (Fin 7 → F) :: switches F n

omit [DecidableEq F] in
theorem switches_exact {m : ℕ} : ∀ (cs : List (Claim F m)),
    Exact (cs.mapM fun c => do
        let s ← recv (Fin 128 → F)
        let r'' ← draw (Fin 7 → F)
        pure (c, s, r'')) (switches F cs.length)
  | [] => .ret _
  | c :: cs => by
      rw [List.mapM_cons, List.length_cons, switches]
      refine Exact.recv_bind fun s => Exact.draw_bind fun r'' => ?_
      rw [← List.append_nil (switches F cs.length)]
      exact Exact.bind (switches_exact cs) fun bs => .ret _

/-- The opening's shape (O2–O6), for `n` claims. -/
def opShape (F : Type) (n : ℕ) : List Kind := switches F n ++ [.coin (Fin n → F)]

omit [Fintype K] [DecidableEq K] in
theorem opening_exact (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (claims : List (Claim F m)) :
    Exact (opening A hm claims) (opShape F claims.length) := by
  unfold opening opShape
  exact Exact.bind (switches_exact claims) fun sw => Exact.draw_bind fun γ => .ret _

/-! ## Compiled Ligerito -/

/-- `k` folding rounds: a coin of two words (`drawK`) and a message pair of `K`. -/
def foldShape (F K : Type) : ℕ → List Kind
  | 0 => []
  | k + 1 => .coin (Fin 2 → F) :: .msg (K × K) :: foldShape F K k

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem foldRoundsC_exact (A : Arith F K) : ∀ (k : ℕ) (s : Opens F D → K) (q : Opens F D → Quad K),
    Exact (foldRoundsC A k s q) (foldShape F K k)
  | 0, _, _ => .ret _
  | k + 1, s, q => by
      unfold foldRoundsC foldShape
      refine Exact.bind_one (Exact.draw_bind fun w => .ret _) fun r => Exact.recv_bind fun msg => ?_
      rw [← List.append_nil (foldShape F K k)]
      exact Exact.bind (foldRoundsC_exact A k _ _) fun r => .ret _

/-- The two out-of-domain samples of a level of dimension `dim`. -/
def oodShape (F K : Type) (dim : ℕ) : List Kind :=
  [.coin (Fin dim → F), .msg F, .msg (K × K), .coin F, .coin (Fin dim → F), .msg F, .msg (K × K), .coin F]

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem twoOodC_exact (A : Arith F K) (dim split : ℕ) (s : Opens F D → K) (q : Opens F D → Quad K) :
    Exact (twoOodC A dim split s q) (oodShape F K dim) := by
  unfold twoOodC oodShape
  refine Exact.draw_bind fun z₁ => ?_
  refine Exact.recv_bind fun y₁ => Exact.recv_bind fun m₁ => ?_
  refine Exact.draw_bind fun β₁ => ?_
  refine Exact.draw_bind fun z₂ => ?_
  refine Exact.recv_bind fun y₂ => Exact.recv_bind fun m₂ => ?_
  exact Exact.draw_bind fun β₂ => .ret _

/-- A level's queries: the positions' and the consistency coins, the round message when there is one, and the
batching coin. -/
def qbShape (F K : Type) (lv : Level) (withMessage : Bool) : List Kind :=
  .coin (Fin lv.queries → F) :: .coin (Fin (ceilLog2 lv.queries) → F) ::
    ((if withMessage then [.msg (K × K)] else []) ++ [.coin F])

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem queryAndBatchC_exact (A : Arith F K) (r : ℕ) (lv : Level) (ch : List K) (start : ℕ) (st : LigStateC F K D)
    (withMessage : Bool) : Exact (queryAndBatchC A r lv ch start st withMessage) (qbShape F K lv withMessage) := by
  unfold queryAndBatchC qbShape
  refine Exact.draw_bind fun w => Exact.draw_bind fun α => ?_
  cases withMessage with
  | false => exact Exact.draw_bind (C := F) fun β => .ret _
  | true => exact Exact.recv_bind fun m => Exact.draw_bind (C := F) fun β => .ret _

/-- The recursive levels' shape: per level its folding rounds, then the next level's cap and out-of-domain samples
and this level's queries, or at the last level the residual `yr` and its queries. -/
def recShape (F K D : Type) (ext : ℕ) : List Level → List Kind
  | [] => []
  | [lv] => foldShape F K lv.lanes ++ .msg (Fin (2 ^ (ext + 1)) → F) :: qbShape F K lv false
  | lv :: lv' :: rest => foldShape F K lv.lanes ++ .msg (Cap D) ::
      (oodShape F K (lv'.logCols + lv'.lanes) ++ (qbShape F K lv true ++ recShape F K D ext (lv' :: rest)))

omit [DecidableEq F] [Fintype K] in
theorem recursiveRoundsC_spine (A : Arith F K) (r n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ) :
    ∀ (levels : List Level) (st : LigStateC F K D),
      Spine (recursiveRoundsC A r n b initial ext levels st) (recShape F K D ext levels)
  | [], _ => .ret _ _
  | [lv], st => by
      unfold recursiveRoundsC recShape
      refine (Exact.bind (foldRoundsC_exact A _ _ _) fun x => ?_).spine
      exact Exact.recv_bind fun yr => by
        rw [← List.append_nil (qbShape F K lv false)]
        exact Exact.bind (queryAndBatchC_exact A r lv _ _ _ false) fun st' => .ret _
  | lv :: lv' :: rest, st => by
      unfold recursiveRoundsC
      rw [recShape]
      refine Exact.bindS (foldRoundsC_exact A _ _ _) fun x => ?_
      refine Spine.recv_bind fun C' => ?_
      refine Exact.bindS (twoOodC_exact A _ _ _ _) fun y => ?_
      exact Exact.bindS (queryAndBatchC_exact A r lv _ _ _ true) fun st' =>
        recursiveRoundsC_spine A r n b initial ext (lv' :: rest) _

/-- Compiled Ligerito's shape, read off the schedule's levels. -/
def ligShape (F K D : Type) (sch : Schedule) : List Kind :=
  match sch.levels with
  | [] => []
  | lv₀ :: rest => .coin (Fin (sch.m - 7) → F) :: .msg F :: .coin F :: .msg (K × K) :: (foldShape F K sch.k0 ++
      match rest with
      | [] => []
      | _ :: _ => .msg (Cap D) :: (oodShape F K (sch.m - 7 - sch.k0 + 1) ++
          (qbShape F K lv₀ true ++ recShape F K D sch.ext rest)))

omit [DecidableEq F] [Fintype K] in
theorem ligeritoC_spine (A : Arith F K) (r : ℕ) (sch : Schedule) (T : F) (b : ℕ → F) :
    Spine (ligeritoC (D := D) A r sch T b) (ligShape F K D sch) := by
  unfold ligeritoC ligShape
  rcases hl : sch.levels with _ | ⟨lv₀, rest⟩
  · exact .ret _ _
  · refine Spine.draw_bind fun z₀ => Spine.recv_bind fun y₀ => Spine.draw_bind fun β₀ => Spine.recv_bind fun msg => ?_
    refine Exact.bindS (foldRoundsC_exact A _ _ _) fun x => ?_
    rcases rest with _ | ⟨lv₁, rest'⟩
    · exact .ret _ _
    · refine Spine.recv_bind fun C₁ => ?_
      refine Exact.bindS (twoOodC_exact A _ _ _ _) fun y => ?_
      exact Exact.bindS (queryAndBatchC_exact A r lv₀ _ _ _ true) fun st =>
        recursiveRoundsC_spine A r _ b _ sch.ext (lv₁ :: rest') _

/-! ## A rep -/

/-- **A rep's shape**: the zerocheck's, the lincheck's, the opening's of the `2 + nExtra` claims, then compiled
Ligerito's. -/
def repShape (F K D : Type) (m kLog nExtra : ℕ) (sch : Schedule) : List Kind :=
  zcShape F m ++ (lcShape F kLog ++ (opShape F (nExtra + 2) ++ ligShape F K D sch))

omit [Fintype K] in
/-- **A rep has a spine.** Every run of rep `r` takes its steps in the order of `repShape`, which reads only `m`,
`k_log`, the number of extra claims and the schedule; it may stop early only on a reject. -/
theorem repC_spine (A : Arith F K) (r : ℕ) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) :
    Spine (repC (D := D) A r S sch hm extra) (repShape F K D S.m S.kLog extra.length sch) := by
  unfold repC repShape
  refine Exact.bindS (zerocheck_exact A S.m) fun z => ?_
  rcases z with _ | zc
  · exact .ret _ _
  · refine Exact.bindS (lincheck_exact A S zc) fun l => ?_
    rcases l with _ | ab
    · exact .ret _ _
    · refine Exact.bindS (opening_exact A _ (ab :: cClaim A zc :: extra)) fun o => ?_
      rcases o with _ | op
      · exact .ret _ _
      · exact ligeritoC_spine A r sch op.T op.b

omit [Fintype K] in
/-- **A rep's run follows `repShape`**: on any tape, a run of rep `r` takes steps whose kinds, in order, are a prefix
of `repShape`. -/
theorem repC_kinds (A : Arith F K) (r : ℕ) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) {ts : List Step} {p : PendL F D} (h : PlaysT (repC A r S sch hm extra) ts p) :
    ts.map kindOf <+: repShape F K D S.m S.kLog extra.length sch :=
  Spine.prefix h (repC_spine A r S sch hm extra)

omit [Fintype K] in
/-- **A rep's tape is determined by its messages and coins.** Two runs of rep `r`, with the same messages and coin
steps in order, are runs on the same tape: `repShape` places the coins among the messages. -/
theorem repC_tape (A : Arith F K) (r : ℕ) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) {ts ts' : List Step} {p p' : PendL F D} (h : PlaysT (repC A r S sch hm extra) ts p)
    (h' : PlaysT (repC A r S sch hm extra) ts' p') (hms : ts.filterMap Sum.getLeft? = ts'.filterMap Sum.getLeft?)
    (hcs : ts.filterMap Sum.getRight? = ts'.filterMap Sum.getRight?) : ts = ts' := by
  have hlen : ts.length = ts'.length := by
    have e : ∀ l : List Step, l.length = (l.filterMap Sum.getLeft?).length + (l.filterMap Sum.getRight?).length := by
      intro l
      induction l with
      | nil => rfl
      | cons s l ih =>
          cases s with
          | inl x =>
              simp only [List.filterMap_cons, Sum.getLeft?_inl, Sum.getRight?_inl, List.length_cons, ih]
              omega
          | inr x =>
              simp only [List.filterMap_cons, Sum.getLeft?_inr, Sum.getRight?_inr, List.length_cons, ih]
              omega
    rw [e ts, e ts', hms, hcs]
  rw [Spine.tape h (repC_spine A r S sch hm extra), Spine.tape h' (repC_spine A r S sch hm extra), hlen, hms, hcs]

end FlockSoundness.Discharge.Exec
