module

public import Proofs.Flock.RecGeneric.ZK
public import Proofs.Flock.Claims.Recursion.SeqPrefix
public import Proofs.Flock.Claims.Recursion.Compose

/-!
# Z1's proof

The compiled run at a nonce `t` and the outer tape `ρ` is its first part, then the outer phase (`real_msgs`). The first
part is a run whose round `i` sends its frame at the leaves `off(i), …, off(i+1) - 1` of one commitment vector
(`first`), the commitments to the workers' values under the salts, each message reading the commitments before it
(`preOf`, `msgsOf`), so that `Leaves.Hides` applies to it whole. After a first part that goes on, the server releases
the outer prover's run, tagged, against `V*` after the first part (`vAfter`, `outerRun`), or the refusal every round
(`refused`).

The outcome is read off the transcript (`outcome_pre`, `outcome_refused`, `outcome_outer_cons`): `none` exactly where
the run is the first part alone or refused (`msgs_of_none`), and otherwise the outer outcome (`someEv_of`).

* At `none` the transcript is a function of the first part's commitments (`noneOf`), and the simulator's is the same
  function of ideal ones (`sim_out_none`): `Hides` costs `n·δ` (`zk_none`).
* At `some o`, at each nonce where the server releases, the outer phase's `ZKOut` at the statement and witness the
  first part fixes and at `V*` after it bounds the outer part by the outer simulator's plus `ε_O` (`zk_some`); averaged
  over the nonce, the remaining event reads the first part's commitments and the outer simulator's coins only, and
  `Hides` with those coins in its `Ω` swaps the commitments for ideal ones, which is the simulator's run given `some o`
  (`sim_out_some`).

`Flock.SecurityProofs.GenericRecursiveZK` is Z1.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-! ## Runs -/

theorem oc_bind_eval_fst (H : Msg → Msg) {α β : Type} (o : OC α) (f : α → OC β) :
    ((o.bind f).eval H).1 = ((f (o.eval H).1).eval H).1 := by
  induction o with
  | ret a => rfl
  | query q k ih => exact ih (H q)

theorem spOf_eval_fst (H : Msg → Msg) (ask : List Msg → Out) {α : Type} (o : OC α) :
    (SP.eval H ask (spOf o)).1 = (o.eval H).1 := by
  rw [ZKMal.recSpOf_eq, ZKMal.ocSP_eval]

theorem spBind_eval_fst (H : Msg → Msg) (ask : List Msg → Out) {α β : Type} (p : SP α) (f : α → SP β) :
    (SP.eval H ask (spBind p f)).1 = (SP.eval H ask (f (SP.eval H ask p).1)).1 := by
  rw [ZKMal.recSpBind_eq, ZKMal.spBind_eval]

theorem spTag_eval (H : Msg → Msg) (ask : List Msg → Out) (pre : List Msg) {α : Type} :
    ∀ p : SP α, SP.eval H ask (spTag pre p) = SP.eval H (fun as => ask (pre ++ as.map tagOuter)) p
  | .ret _ => rfl
  | .hash q k => by simp only [spTag, SP.eval, spTag_eval H ask pre (k (H q))]
  | .ask as k => by simp only [spTag, SP.eval, spTag_eval H ask pre (k (ask (pre ++ as.map tagOuter)))]

theorem msgs_of_goesOn {k : ℕ} {τ : List (Msg × Out)} (h : ZKMal.goesOn k τ = true) :
    ∀ p ∈ τ, ZKMal.isMsg p.2 = true := by
  simp only [ZKMal.goesOn, Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at h
  exact h.2

theorem recGoesOn_cons_msg (a m : Msg) (τ : List (Msg × Out)) :
    Recursion.goesOn ((a, .msg m) :: τ) = Recursion.goesOn τ := by
  induction τ using List.reverseRecOn with
  | nil => rfl
  | append_singleton l p _ =>
    obtain ⟨b, o⟩ := p
    rw [← List.cons_append, ZKMal.recGoesOn_snoc, ZKMal.recGoesOn_snoc]

/-- `Recursion.goesOn` of a run of `n` rounds, `n = 0` included, is `goesOn n`. -/
theorem recGoesOn_run (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)),
      Recursion.goesOn (Claims.interact ask pm n hist) = ZKMal.goesOn n (Claims.interact ask pm n hist)
  | 0, _ => rfl
  | n + 1, hist => by
    simp only [Claims.interact]
    cases ask (hist.map Prod.fst ++ [pm hist]) with
    | msg m => rw [recGoesOn_cons_msg, ZKMal.goesOn_cons]; exact recGoesOn_run ask pm n _
    | acc => simp [Recursion.goesOn, ZKMal.goesOn, ZKMal.isMsg]
    | rej => simp [Recursion.goesOn, ZKMal.goesOn, ZKMal.isMsg]

/-- Two provers that agree on every history shorter than `hist`'s length plus `n` run alike for `n` rounds. -/
theorem interact_len_congr (ask : List Msg → Out) (pm pm' : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)), (∀ h : List (Msg × Msg), h.length < hist.length + n → pm' h = pm h) →
      Claims.interact ask pm' n hist = Claims.interact ask pm n hist
  | 0, _, _ => rfl
  | n + 1, hist, h => by
    have h0 : pm' hist = pm hist := h hist (by omega)
    simp only [Claims.interact, h0]
    cases ask (hist.map Prod.fst ++ [pm hist]) with
    | msg m =>
      simp only
      rw [interact_len_congr ask pm pm' n _ fun h' hl =>
        h h' (by rw [List.length_append, List.length_singleton] at hl; omega)]
    | acc => rfl
    | rej => rfl

/-- **Tagging commutes with a run**: a prover that tags its messages and reads its history untagged, against a
verifier, is the untagged prover's run against the verifier that reads its questions tagged, tagged. -/
theorem interact_tag (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)),
      Claims.interact ask (fun h => tagOuter (pm (h.map fun p => (untagOuter p.1, p.2)))) n
          (hist.map fun p => (tagOuter p.1, p.2)) =
        (Claims.interact (fun as => ask (as.map tagOuter)) pm n hist).map fun p => (tagOuter p.1, p.2)
  | 0, _ => rfl
  | n + 1, hist => by
    have hu : (hist.map fun p => (tagOuter p.1, p.2)).map (fun p => (untagOuter p.1, p.2)) = hist := by
      simp [List.map_map, Function.comp_def, tagOuter, untagOuter]
    have hf : (hist.map fun p => (tagOuter p.1, p.2)).map Prod.fst ++ [tagOuter (pm hist)] =
        (hist.map Prod.fst ++ [pm hist]).map tagOuter := by
      simp [List.map_map, Function.comp_def]
    simp only [Claims.interact]
    rw [hu, hf]
    cases ask ((hist.map Prod.fst ++ [pm hist]).map tagOuter) with
    | msg m =>
      simp only [List.map_cons]
      rw [← interact_tag ask pm n (hist ++ [(pm hist, m)])]
      simp only [List.map_append, List.map_cons, List.map_nil]
    | acc => rfl
    | rej => rfl

theorem interact_tag_nil (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (n : ℕ) :
    Claims.interact ask (fun h => tagOuter (pm (h.map fun p => (untagOuter p.1, p.2)))) n [] =
      (Claims.interact (fun as => ask (as.map tagOuter)) pm n []).map fun p => (tagOuter p.1, p.2) :=
  interact_tag ask pm n []

theorem tag_ne_refusal (m : Msg) : tagOuter m ≠ refusal := by
  intro h
  simp only [tagOuter, refusal, List.cons.injEq] at h
  exact absurd h.1 (by decide)

theorem Leaves.commits_at (Lv : Leaves) {n : ℕ} (msgs : (i : Fin n) → (Fin i → Msg) → Msg) (y : Fin n → Lv.Salt)
    (g : Fin n) :
    Lv.commits msgs y g = Lv.leaf (msgs g fun j => Lv.commits msgs y (Fin.castLE g.isLt.le j)) (y g) := by
  rw [Leaves.commits]

/-! ## The first part as a run on one commitment vector -/

namespace ZkRelease

variable {W : Type} {Lv : Leaves} (Zs : ZkRelease W Lv)

theorem off_succ (x : Msg) (i : ℕ) : Zs.off x (i + 1) = Zs.off x i + Zs.nl x i := by
  unfold off
  exact Finset.sum_range_succ _ _

theorem off_mono (x : Msg) {a b : ℕ} (h : a ≤ b) : Zs.off x a ≤ Zs.off x b :=
  Finset.sum_le_sum_of_subset (Finset.range_subset_range.2 h)

/-- The first-part round the `g`-th leaf is in. -/
def roundOf (x : Msg) (g : ℕ) : ℕ := ((Finset.range (Zs.nh x)).filter fun i => Zs.off x (i + 1) ≤ g).card

theorem roundOf_eq (x : Msg) {i j : ℕ} (hi : i < Zs.nh x) (hj : j < Zs.nl x i) :
    Zs.roundOf x (Zs.off x i + j) = i := by
  unfold roundOf
  have : (Finset.range (Zs.nh x)).filter (fun i' => Zs.off x (i' + 1) ≤ Zs.off x i + j) = Finset.range i := by
    ext i'
    simp only [Finset.mem_filter, Finset.mem_range]
    constructor
    · rintro ⟨_, h⟩
      by_contra hc
      have h1 := Zs.off_mono x (show i + 1 ≤ i' + 1 by omega)
      have h2 := Zs.off_succ x i
      omega
    · intro h
      have h1 := Zs.off_mono x (show i' + 1 ≤ i by omega)
      exact ⟨by omega, by omega⟩
  rw [this, Finset.card_range]

/-- Entry `g` of a commitment vector, `[]` past its end. -/
def cAt (x : Msg) (c : Fin (Zs.nLeaves x) → Msg) (g : ℕ) : Msg := if h : g < Zs.nLeaves x then c ⟨g, h⟩ else []

/-- First-part round `i`'s leaves in the commitment vector `c`. -/
def roundLeaves (x : Msg) (c : Fin (Zs.nLeaves x) → Msg) (i : ℕ) : List Msg :=
  List.ofFn fun j : Fin (Zs.nl x i) => Zs.cAt x c (Zs.off x i + j)

/-- The first part's prover at the commitment vector `c`: each round its frame at the round's leaves of `c`. -/
def pmC (H : Msg → Msg) (x : Msg) (c : Fin (Zs.nLeaves x) → Msg) (h : List (Msg × Msg)) : Msg :=
  ((Zs.frame x (h.map fun p => (p.1, Out.msg p.2)) (Zs.roundLeaves x c h.length)).eval H).1

/-- The first part against the verifier `ask`, at the commitment vector `c`. -/
def first (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (c : Fin (Zs.nLeaves x) → Msg) : List (Msg × Out) :=
  Claims.interact ask (Zs.pmC H x c) (Zs.nh x) []

/-- The commitments before leaf `g`, as a commitment vector (`[]` from `g` on). -/
def prevAt (x : Msg) {g : ℕ} (prev : Fin g → Msg) : Fin (Zs.nLeaves x) → Msg :=
  fun g' => if h : g'.val < g then prev ⟨g', h⟩ else []

/-- **Leaf `g`'s message**: its entry of the workers' values for its round, at the history the commitments before it
give. -/
def msgsOf (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (w : W) (ωp : Zs.Ωp x) (g : Fin (Zs.nLeaves x))
    (prev : Fin g → Msg) : Msg :=
  ((Zs.msg w x ωp (Claims.histOf (Claims.interact ask (Zs.pmC H x (Zs.prevAt x prev)) (Zs.roundOf x g) []))).eval
    H).1.getD ((g : ℕ) - Zs.off x (Zs.roundOf x g)) []

/-- **The first part at the nonce `t`**: at the commitments to the workers' values under `t`'s salts. -/
def preOf (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (w : W) (t : Zs.N x) : List (Msg × Out) :=
  Zs.first H x ask (Lv.commits (Zs.msgsOf H x ask w t.1) t.2)

theorem preOf_length_le (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (w : W) (t : Zs.N x) :
    (Zs.preOf H x ask w t).length ≤ Zs.nh x :=
  ZKMal.interact_length_le _ _ _ _

theorem roundLeaves_prevAt (x : Msg) (c : Fin (Zs.nLeaves x) → Msg) (g : Fin (Zs.nLeaves x)) (l : ℕ)
    (hl : Zs.off x (l + 1) ≤ g) :
    Zs.roundLeaves x (Zs.prevAt x fun j : Fin g => c (Fin.castLE g.isLt.le j)) l = Zs.roundLeaves x c l := by
  unfold roundLeaves
  refine congrArg List.ofFn (funext fun j => ?_)
  have h2 := Zs.off_succ x l
  have hj : Zs.off x l + j < g := by have := j.isLt; omega
  have hg : Zs.off x l + j < Zs.nLeaves x := by have := g.isLt; omega
  simp only [cAt, prevAt, hg, hj, ↓reduceDIte]
  rfl

theorem interact_prevAt (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (c : Fin (Zs.nLeaves x) → Msg)
    (g : Fin (Zs.nLeaves x)) (i : ℕ) (hi : Zs.off x i ≤ g) :
    Claims.interact ask (Zs.pmC H x (Zs.prevAt x fun j : Fin g => c (Fin.castLE g.isLt.le j))) i [] =
      Claims.interact ask (Zs.pmC H x c) i [] := by
  refine interact_len_congr ask _ _ i [] fun h hl => ?_
  simp only [List.length_nil, Nat.zero_add] at hl
  unfold pmC
  rw [Zs.roundLeaves_prevAt x c g h.length (le_trans (Zs.off_mono x (by omega)) hi)]

theorem leaves_eq (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (w : W) (t : Zs.N x) (h : List (Msg × Msg))
    (hlt : h.length < Zs.nh x)
    (hh : Claims.histOf (Claims.interact ask (Zs.pmC H x (Lv.commits (Zs.msgsOf H x ask w t.1) t.2)) h.length []) = h) :
    (List.ofFn fun j => Lv.leaf (((Zs.msg w x t.1 h).eval H).1.getD j []) (t.2 (Zs.leafAt x hlt j))) =
      Zs.roundLeaves x (Lv.commits (Zs.msgsOf H x ask w t.1) t.2) h.length := by
  unfold roundLeaves
  refine congrArg List.ofFn (funext fun j => ?_)
  have hg : Zs.off x h.length + j < Zs.nLeaves x := (Zs.leafAt x hlt j).isLt
  rw [cAt, dite_eq_left hg, Leaves.commits_at]
  refine congrArg₂ Lv.leaf ?_ rfl
  dsimp only [msgsOf]
  rw [Zs.roundOf_eq x hlt j.isLt, Nat.add_sub_cancel_left,
    Zs.interact_prevAt H x ask _ ⟨_, hg⟩ h.length (Nat.le_add_right _ _), hh]

theorem pmR_eq (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (w : W) (ω : Zs.N x × Zs.Ω x)
    (h : List (Msg × Msg)) (hlt : h.length < Zs.nh x)
    (hh : Claims.histOf (Claims.interact ask (Zs.pmC H x (Lv.commits (Zs.msgsOf H x ask w ω.1.1) ω.1.2)) h.length []) =
      h) :
    (((Zs.prover w x).next x h ω).eval H).1 = Zs.pmC H x (Lv.commits (Zs.msgsOf H x ask w ω.1.1) ω.1.2) h := by
  simp only [prover, dite_eq_left hlt]
  rw [oc_bind_eval_fst]
  exact congrArg (fun L => ((Zs.frame x (h.map fun p => (p.1, Out.msg p.2)) L).eval H).1)
    (Zs.leaves_eq H x ask w ω.1 h hlt hh)

/-- **The compiled prover's first part is `preOf`.** -/
theorem first_eq (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (w : W) (ω : Zs.N x × Zs.Ω x) :
    Claims.interact ask (fun h => (((Zs.prover w x).next x h ω).eval H).1) (Zs.nh x) [] = Zs.preOf H x ask w ω.1 := by
  unfold preOf first
  refine ZKMal.interact_congr ask _ _ (Zs.nh x) [] fun i hi => ?_
  have hle := ZKMal.interact_length_le ask (Zs.pmC H x (Lv.commits (Zs.msgsOf H x ask w ω.1.1) ω.1.2)) (Zs.nh x) []
  have hlen := ZKMal.histOf_take_length _ hi
  rw [List.nil_append]
  refine Zs.pmR_eq H x ask w ω _ (by omega) ?_
  rw [hlen, ← ZKMal.interact_take ask _ [] (by omega : i ≤ Zs.nh x)]

theorem recGoesOn_first (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (c : Fin (Zs.nLeaves x) → Msg) :
    Recursion.goesOn (Zs.first H x ask c) = ZKMal.goesOn (Zs.nh x) (Zs.first H x ask c) :=
  recGoesOn_run ask _ _ []

theorem recGoesOn_preOf (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (w : W) (t : Zs.N x) :
    Recursion.goesOn (Zs.preOf H x ask w t) = ZKMal.goesOn (Zs.nh x) (Zs.preOf H x ask w t) :=
  recGoesOn_run ask _ _ []

theorem preRun_eq (H : Msg → Msg) (w : W) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (ask : List Msg → Out)
    (hask : ask = fun as => ((Vs.next x as θ).eval H).1) (t : Zs.N x) :
    Zs.preRun H w x Vs θ t = Zs.preOf H x ask w t := by
  unfold preRun
  rw [ZKMal.stepWith_msgs H _ x _ _ (Zs.nh x) [] [] _ rfl, ← hask, Zs.first_eq H x ask w]
  rfl

/-! ## The outer phase -/

/-- The outer prover's run against `V*` after the first part `τ`, at the nonce `t` and the outer tape `ρ`. -/
def outerRun (H : Msg → Msg) (w : W) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (t : Zs.N x) (ρ : Zs.Ω x)
    (τ : List (Msg × Out)) : List (Msg × Out) :=
  (runStar H (Zs.PO (Zs.wo w x t.1 t.2 τ) x) (vAfter Vs x (Zs.ko x) (τ.map Prod.fst)) (Zs.xo x τ) ρ θ).msgs

/-- The first part `τ`, then the refusal every round against the verifier `ask`. -/
def refused (x : Msg) (ask : List Msg → Out) (τ : List (Msg × Out)) : List (Msg × Out) :=
  τ ++ Claims.interact (fun as => ask (τ.map Prod.fst ++ as)) (fun _ => refusal) (Zs.ko x) []

/-- The transcript where no outer view is released: the refusal after a first part `τ` that goes on, else `τ`. -/
def noneOf (x : Msg) (ask : List Msg → Out) (τ : List (Msg × Out)) : List (Msg × Out) :=
  if ZKMal.goesOn (Zs.nh x) τ then Zs.refused x ask τ else τ

/-- The outer simulator's transcript given `o`, against `V*` after the first part `τ`. -/
def outerSim (H : Msg → Msg) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (o : Zs.St) (τ : List (Msg × Out))
    (cO : Zs.CO x) : List (Msg × Out) :=
  ((Zs.simO x).given o).out H (vAfter Vs x (Zs.ko x) (τ.map Prod.fst)) θ (Zs.xo x τ) cO

theorem take_histOf (x : Msg) {τ : List (Msg × Out)} (hτ : ZKMal.goesOn (Zs.nh x) τ = true)
    (h' : List (Msg × Msg)) : ((Claims.histOf τ ++ h').take (Zs.nh x)).map (fun p => (p.1, Out.msg p.2)) = τ := by
  have hl : (Claims.histOf τ).length = Zs.nh x := by
    rw [ZKMal.histOf_length]; exact ZKMal.goesOn_length hτ
  rw [List.take_left' hl]
  exact RecDeploy.asRounds_histOf (msgs_of_goesOn hτ)

open Classical in
/-- **The compiled prover after a first part `τ` that goes on**: the outer prover's message, tagged, at the untagged
outer history where the server releases, and the refusal otherwise. -/
theorem next_after (H : Msg → Msg) (w : W) (x : Msg) (ω : Zs.N x × Zs.Ω x) {τ : List (Msg × Out)}
    (hτ : ZKMal.goesOn (Zs.nh x) τ = true) (h' : List (Msg × Msg)) :
    (((Zs.prover w x).next x (Claims.histOf τ ++ h') ω).eval H).1 =
      if Zs.ok w x ω.1 τ then
        tagOuter (((Zs.PO (Zs.wo w x ω.1.1 ω.1.2 τ) x).next (Zs.xo x τ)
          (h'.map fun p => (untagOuter p.1, p.2)) ω.2).eval H).1
      else refusal := by
  have hl : (Claims.histOf τ).length = Zs.nh x := by
    rw [ZKMal.histOf_length]; exact ZKMal.goesOn_length hτ
  have hn : ¬ (Claims.histOf τ ++ h').length < Zs.nh x := by rw [List.length_append]; omega
  have hd : (Claims.histOf τ ++ h').drop (Zs.nh x) = h' := List.drop_left' hl
  simp only [prover, dite_eq_right hn, Zs.take_histOf x hτ h', hd]
  split_ifs
  · exact oc_bind_eval_fst H _ _
  · rfl

open Classical in
/-- **The compiled run**: the first part `preOf`; where it goes on, the outer run tagged where the server releases, and
the refusal otherwise. -/
theorem real_msgs (H : Msg → Msg) (w : W) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = Zs.k x)
    (ask : List Msg → Out) (hask : ask = fun as => ((Vs.next x as θ).eval H).1) (ω : Zs.N x × Zs.Ω x) :
    (runStar H (Zs.prover w x) Vs x ω θ).msgs =
      if ZKMal.goesOn (Zs.nh x) (Zs.preOf H x ask w ω.1) then
        (if Zs.ok w x ω.1 (Zs.preOf H x ask w ω.1) then
          Zs.preOf H x ask w ω.1 ++
            (Zs.outerRun H w x Vs θ ω.1 ω.2 (Zs.preOf H x ask w ω.1)).map fun p => (tagOuter p.1, p.2)
        else Zs.refused x ask (Zs.preOf H x ask w ω.1))
      else Zs.preOf H x ask w ω.1 := by
  rw [ZKMal.runStar_msgs, ← hask, hk, show Zs.k x = Zs.nh x + Zs.ko x from rfl, ZKMal.interact_add,
    List.nil_append, Zs.first_eq H x ask w ω]
  unfold ZKMal.cont
  by_cases hg : ZKMal.goesOn (Zs.nh x) (Zs.preOf H x ask w ω.1) = true
  · rw [ite_eq_left hg, ite_eq_left hg, ← List.append_nil (Claims.histOf (Zs.preOf H x ask w ω.1)), ZKMal.interact_shift,
      ZKMal.histOf_map_fst]
    simp only [Zs.next_after H w x ω hg]
    by_cases hok : Zs.ok w x ω.1 (Zs.preOf H x ask w ω.1)
    · simp only [hok, ↓reduceIte]
      rw [interact_tag_nil (fun as => ask ((Zs.preOf H x ask w ω.1).map Prod.fst ++ as))
        (fun h => (((Zs.PO (Zs.wo w x ω.1.1 ω.1.2 (Zs.preOf H x ask w ω.1)) x).next
          (Zs.xo x (Zs.preOf H x ask w ω.1)) h ω.2).eval H).1)]
      unfold outerRun
      rw [ZKMal.runStar_msgs]
      subst hask
      rfl
    · simp only [hok, ↓reduceIte]
      rfl
  · rw [ite_eq_right hg, ite_eq_right hg]

theorem outerRun_nil (H : Msg → Msg) (w : W) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (t : Zs.N x) (ρ : Zs.Ω x)
    (τ : List (Msg × Out)) (hko : Zs.ko x = 0) : Zs.outerRun H w x Vs θ t ρ τ = [] := by
  unfold outerRun
  rw [ZKMal.runStar_msgs]
  show Claims.interact _ _ (Zs.ko x) [] = []
  simp only [hko, Claims.interact]

theorem outerRun_ne_nil (H : Msg → Msg) (w : W) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (t : Zs.N x)
    (ρ : Zs.Ω x) (τ : List (Msg × Out)) (hko : 0 < Zs.ko x) : Zs.outerRun H w x Vs θ t ρ τ ≠ [] := by
  unfold outerRun
  rw [ZKMal.runStar_msgs]
  exact ZKMal.interact_ne_nil _ _ _ _ hko

/-! ## The outcome -/

theorem outcome_pre (x : Msg) {τ : List (Msg × Out)} (h : τ.length ≤ Zs.nh x) : Zs.outcome x τ = none := by
  unfold outcome
  rw [ite_eq_left (Or.inl h)]

theorem outcome_refused (x : Msg) (ask : List Msg → Out) {τ : List (Msg × Out)} (hτ : τ.length = Zs.nh x) :
    Zs.outcome x (Zs.refused x ask τ) = none := by
  unfold outcome refused
  rcases Nat.eq_zero_or_pos (Zs.ko x) with h0 | hpos
  · rw [h0]
    simp [Claims.interact, hτ]
  · obtain ⟨n, hn⟩ : ∃ n, Zs.ko x = n + 1 := ⟨_, (Nat.succ_pred_eq_of_pos hpos).symm⟩
    rw [hn]
    apply ite_eq_left
    right
    rw [List.drop_left' hτ]
    simp only [Claims.interact]
    split <;> rfl

theorem outcome_outer_cons (x : Msg) {τ : List (Msg × Out)} (hτ : τ.length = Zs.nh x) {τO : List (Msg × Out)}
    (hne : τO ≠ []) :
    Zs.outcome x (τ ++ τO.map fun p => (tagOuter p.1, p.2)) = some (Zs.outO x (Zs.xo x τ) τO) := by
  obtain ⟨q, τO', rfl⟩ := List.exists_cons_of_ne_nil hne
  have hc : ¬ ((τ ++ (q :: τO').map fun p => (tagOuter p.1, p.2)).length ≤ Zs.nh x ∨
      ((τ ++ (q :: τO').map fun p => (tagOuter p.1, p.2)).drop (Zs.nh x)).head?.map Prod.fst = some refusal) := by
    rw [List.drop_left' hτ]
    rintro (h | h)
    · simp only [List.length_append, List.length_map, List.length_cons] at h
      omega
    · exact tag_ne_refusal q.1 (Option.some.inj h)
  unfold outcome
  rw [ite_eq_right hc, List.take_left' hτ, List.drop_left' hτ]
  simp [List.map_map, Function.comp_def, tagOuter, untagOuter]

/-- **No outer view released**: the compiled run is `noneOf` of its first part. -/
theorem msgs_of_none (H : Msg → Msg) (w : W) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = Zs.k x)
    (ask : List Msg → Out) (hask : ask = fun as => ((Vs.next x as θ).eval H).1) (ω : Zs.N x × Zs.Ω x)
    (h : Zs.outcome x (runStar H (Zs.prover w x) Vs x ω θ).msgs = none) :
    (runStar H (Zs.prover w x) Vs x ω θ).msgs = Zs.noneOf x ask (Zs.preOf H x ask w ω.1) := by
  rw [Zs.real_msgs H w x Vs θ hk ask hask ω] at h ⊢
  unfold noneOf
  by_cases hg : ZKMal.goesOn (Zs.nh x) (Zs.preOf H x ask w ω.1) = true
  · rw [ite_eq_left hg] at h
    rw [ite_eq_left hg, ite_eq_left hg]
    by_cases hok : Zs.ok w x ω.1 (Zs.preOf H x ask w ω.1)
    · rw [ite_eq_left hok] at h ⊢
      by_cases hnil : Zs.outerRun H w x Vs θ ω.1 ω.2 (Zs.preOf H x ask w ω.1) = []
      · have h0 : Zs.ko x = 0 := by
          by_contra hne
          exact Zs.outerRun_ne_nil H w x Vs θ ω.1 ω.2 _ (Nat.pos_of_ne_zero hne) hnil
        rw [hnil]
        unfold refused
        rw [h0]
        simp [Claims.interact]
      · rw [Zs.outcome_outer_cons x (ZKMal.goesOn_length hg) hnil] at h
        exact absurd h (by simp)
    · rw [ite_eq_right hok]
  · rw [ite_eq_right hg, ite_eq_right hg]

/-- The event at `some o` that the outer phase's zero knowledge bounds, at the nonce `t` and the outer tape `ρ`: the
server releases after a first part that goes on, the outer outcome is `o`, and the event holds. -/
def someEv (H : Msg → Msg) (w : W) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (ask : List Msg → Out) (o : Zs.St)
    (Ev : List (Msg × Out) → Prop) (t : Zs.N x) (ρ : Zs.Ω x) : Prop :=
  (ZKMal.goesOn (Zs.nh x) (Zs.preOf H x ask w t) = true ∧ Zs.ok w x t (Zs.preOf H x ask w t)) ∧
    Zs.outO x (Zs.xo x (Zs.preOf H x ask w t)) (Zs.outerRun H w x Vs θ t ρ (Zs.preOf H x ask w t)) = o ∧
    Ev (Zs.preOf H x ask w t ++
      (Zs.outerRun H w x Vs θ t ρ (Zs.preOf H x ask w t)).map fun p => (tagOuter p.1, p.2))

theorem someEv_of (H : Msg → Msg) (w : W) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = Zs.k x)
    (ask : List Msg → Out) (hask : ask = fun as => ((Vs.next x as θ).eval H).1) (o : Zs.St)
    (Ev : List (Msg × Out) → Prop) (ω : Zs.N x × Zs.Ω x)
    (h : Zs.outcome x (runStar H (Zs.prover w x) Vs x ω θ).msgs = some o ∧ Ev (runStar H (Zs.prover w x) Vs x ω θ).msgs) :
    Zs.someEv H w x Vs θ ask o Ev ω.1 ω.2 := by
  obtain ⟨hout, hev⟩ := h
  rw [Zs.real_msgs H w x Vs θ hk ask hask ω] at hout hev
  by_cases hg : ZKMal.goesOn (Zs.nh x) (Zs.preOf H x ask w ω.1) = true
  · rw [ite_eq_left hg] at hout hev
    by_cases hok : Zs.ok w x ω.1 (Zs.preOf H x ask w ω.1)
    · rw [ite_eq_left hok] at hout hev
      by_cases hnil : Zs.outerRun H w x Vs θ ω.1 ω.2 (Zs.preOf H x ask w ω.1) = []
      · rw [hnil, List.map_nil, List.append_nil, Zs.outcome_pre x (Zs.preOf_length_le H x ask w ω.1)] at hout
        exact absurd hout (by simp)
      · rw [Zs.outcome_outer_cons x (ZKMal.goesOn_length hg) hnil] at hout
        exact ⟨⟨hg, hok⟩, Option.some.inj hout, hev⟩
    · rw [ite_eq_right hok, Zs.outcome_refused x ask (ZKMal.goesOn_length hg)] at hout
      exact absurd hout (by simp)
  · rw [ite_eq_right hg, Zs.outcome_pre x (Zs.preOf_length_le H x ask w ω.1)] at hout
    exact absurd hout (by simp)

/-! ## The simulator's transcripts -/

theorem ideal_leaves (x : Msg) (r : Fin (Zs.nLeaves x) → Lv.Rand) {L : ℕ} (hL : L < Zs.nh x) :
    (List.ofFn fun j : Fin (Zs.nl x L) => Lv.ideal (r (Zs.leafAt x hL j))) =
      Zs.roundLeaves x (fun g => Lv.ideal (r g)) L := by
  unfold roundLeaves
  refine congrArg List.ofFn (funext fun j => ?_)
  have hg : Zs.off x L + j < Zs.nLeaves x := (Zs.leafAt x hL j).isLt
  rw [cAt, dite_eq_left hg]
  rfl

theorem sim_first (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (r : Fin (Zs.nLeaves x) → Lv.Rand) :
    (SP.eval H ask (simRounds (fun ms => if h : ms.length < Zs.nh x then
        spOf (Zs.frame x ms (List.ofFn fun j => Lv.ideal (r (Zs.leafAt x h j))))
      else .ret []) (Zs.nh x) [])).1 = Zs.first H x ask fun g => Lv.ideal (r g) := by
  rw [RecDeploy.simRounds_interact H ask _ (Zs.nh x) [] (by simp), List.nil_append]
  unfold first
  refine interact_len_congr ask _ _ (Zs.nh x) [] fun h hl => ?_
  have hlt : (RecDeploy.asRounds h).length < Zs.nh x := by simpa [RecDeploy.asRounds] using hl
  rw [dite_eq_left hlt, spOf_eval_fst, Zs.ideal_leaves x r hlt]
  simp only [pmC, RecDeploy.asRounds, List.length_map]

theorem sim_out_none (H : Msg → Msg) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (ask : List Msg → Out)
    (hask : ask = fun as => ((Vs.next x as θ).eval H).1) (c : (Fin (Zs.nLeaves x) → Lv.Rand) × Zs.CO x) :
    (Zs.sim.given none).out H Vs θ x c = Zs.noneOf x ask (Zs.first H x ask fun g => Lv.ideal (c.1 g)) := by
  dsimp only [Simulator.out, SimOut.given, sim]
  rw [← hask, spBind_eval_fst, Zs.sim_first H x ask c.1, Zs.recGoesOn_first]
  unfold noneOf
  by_cases hg : ZKMal.goesOn (Zs.nh x) (Zs.first H x ask fun g => Lv.ideal (c.1 g)) = true
  · rw [ite_eq_left hg, ite_eq_left hg, RecDeploy.simRounds_interact H ask _ (Zs.ko x) _ (msgs_of_goesOn hg),
      ← List.append_nil (Claims.histOf (Zs.first H x ask fun g => Lv.ideal (c.1 g))), ZKMal.interact_shift,
      ZKMal.histOf_map_fst]
    rfl
  · rw [ite_eq_right hg, ite_eq_right hg]
    rfl

theorem sim_out_some (H : Msg → Msg) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (ask : List Msg → Out)
    (hask : ask = fun as => ((Vs.next x as θ).eval H).1) (o : Zs.St)
    (c : (Fin (Zs.nLeaves x) → Lv.Rand) × Zs.CO x) :
    (Zs.sim.given (some o)).out H Vs θ x c =
      if ZKMal.goesOn (Zs.nh x) (Zs.first H x ask fun g => Lv.ideal (c.1 g)) then
        Zs.first H x ask (fun g => Lv.ideal (c.1 g)) ++
          (Zs.outerSim H x Vs θ o (Zs.first H x ask fun g => Lv.ideal (c.1 g)) c.2).map fun p => (tagOuter p.1, p.2)
      else Zs.first H x ask fun g => Lv.ideal (c.1 g) := by
  dsimp only [Simulator.out, SimOut.given, sim]
  rw [← hask, spBind_eval_fst, Zs.sim_first H x ask c.1, Zs.recGoesOn_first]
  by_cases hg : ZKMal.goesOn (Zs.nh x) (Zs.first H x ask fun g => Lv.ideal (c.1 g)) = true
  · rw [ite_eq_left hg, ite_eq_left hg, spBind_eval_fst, spTag_eval]
    subst hask
    rfl
  · rw [ite_eq_right hg, ite_eq_right hg]
    rfl

/-! ## Zero knowledge at each outcome -/

theorem zk_none (H : Msg → Msg) (εO : (x : Msg) → Msg → Zs.WO → (Θ : Type) → VStar Θ → Θ → Zs.St → ℝ) (δ : ℝ)
    (hHide : Lv.Hides δ) (x : Msg) (w : W) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = Zs.k x)
    (ask : List Msg → Out) (hask : ask = fun as => ((Vs.next x as θ).eval H).1) (Ev : List (Msg × Out) → Prop) :
    Pr (fun ω : Zs.N x × Zs.Ω x => Zs.outcome x (runStar H (Zs.prover w x) Vs x ω θ).msgs = none ∧
        Ev (runStar H (Zs.prover w x) Vs x ω θ).msgs) ≤
      Pr (fun c : Zs.sim.C x => Ev ((Zs.sim.given none).out H Vs θ x c)) + Zs.err H εO δ x w Θ Vs θ none := by
  have h1 : Pr (fun ω : Zs.N x × Zs.Ω x => Zs.outcome x (runStar H (Zs.prover w x) Vs x ω θ).msgs = none ∧
        Ev (runStar H (Zs.prover w x) Vs x ω θ).msgs) ≤
      Pr (fun t : Zs.N x => Ev (Zs.noneOf x ask (Zs.preOf H x ask w t))) := by
    refine le_trans (Pr_mono fun ω h => ?_)
      (le_of_eq (Pr_fst (fun t : Zs.N x => Ev (Zs.noneOf x ask (Zs.preOf H x ask w t)))))
    show Ev (Zs.noneOf x ask (Zs.preOf H x ask w ω.1))
    rw [← Zs.msgs_of_none H w x Vs θ hk ask hask ω h.1]
    exact h.2
  have h2 := hHide (Zs.nLeaves x) (Zs.Ωp x) (Zs.msgsOf H x ask w) fun _ cv => Ev (Zs.noneOf x ask (Zs.first H x ask cv))
  have h3 : Pr (fun p : Zs.Ωp x × (Fin (Zs.nLeaves x) → Lv.Rand) =>
        Ev (Zs.noneOf x ask (Zs.first H x ask fun i => Lv.ideal (p.2 i)))) =
      Pr (fun c : Zs.sim.C x => Ev ((Zs.sim.given none).out H Vs θ x c)) := by
    calc Pr (fun p : Zs.Ωp x × (Fin (Zs.nLeaves x) → Lv.Rand) =>
          Ev (Zs.noneOf x ask (Zs.first H x ask fun i => Lv.ideal (p.2 i))))
        = Pr (fun r : Fin (Zs.nLeaves x) → Lv.Rand => Ev (Zs.noneOf x ask (Zs.first H x ask fun i => Lv.ideal (r i)))) :=
          Pr_snd (fun r : Fin (Zs.nLeaves x) → Lv.Rand =>
            Ev (Zs.noneOf x ask (Zs.first H x ask fun i => Lv.ideal (r i))))
      _ = Pr (fun c : (Fin (Zs.nLeaves x) → Lv.Rand) × Zs.CO x =>
          Ev (Zs.noneOf x ask (Zs.first H x ask fun i => Lv.ideal (c.1 i)))) :=
          (Pr_fst (fun r : Fin (Zs.nLeaves x) → Lv.Rand =>
            Ev (Zs.noneOf x ask (Zs.first H x ask fun i => Lv.ideal (r i))))).symm
      _ = Pr (fun c : Zs.sim.C x => Ev ((Zs.sim.given none).out H Vs θ x c)) :=
          Pr_congr fun c => by rw [Zs.sim_out_none H x Vs θ ask hask c]
  have he : Zs.err H εO δ x w Θ Vs θ none = Zs.nLeaves x * δ := by simp [err]
  rw [he]
  exact h1.trans (h2.trans (add_le_add (le_of_eq h3) le_rfl))

/-- The event the hybrid at `some o` reads, at the nonce and the outer simulator's coins: the first part goes on, and
the event holds of it followed by the outer simulator's run, tagged. -/
def simEv (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (w : W) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (o : Zs.St)
    (Ev : List (Msg × Out) → Prop) (q : Zs.N x × Zs.CO x) : Prop :=
  ZKMal.goesOn (Zs.nh x) (Zs.preOf H x ask w q.1) = true ∧
    Ev (Zs.preOf H x ask w q.1 ++
      (Zs.outerSim H x Vs θ o (Zs.preOf H x ask w q.1) q.2).map fun p => (tagOuter p.1, p.2))

/-- `simEv`'s probability at the nonce `t`. -/
noncomputable def simAt (H : Msg → Msg) (x : Msg) (ask : List Msg → Out) (w : W) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (o : Zs.St) (Ev : List (Msg × Out) → Prop) (t : Zs.N x) : ℝ :=
  Pr (fun cO : Zs.CO x => Zs.simEv H x ask w Vs θ o Ev (t, cO))

open Classical in
/-- `ZkRelease.err`'s outer term at the nonce `t`. -/
noncomputable def errAt (H : Msg → Msg) (εO : (x : Msg) → Msg → Zs.WO → (Θ : Type) → VStar Θ → Θ → Zs.St → ℝ)
    (x : Msg) (w : W) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (o : Zs.St) (t : Zs.N x) : ℝ :=
  if Recursion.goesOn (Zs.preRun H w x Vs θ t) ∧ Zs.ok w x t (Zs.preRun H w x Vs θ t) then
    εO x (Zs.xo x (Zs.preRun H w x Vs θ t)) (Zs.wo w x t.1 t.2 (Zs.preRun H w x Vs θ t)) Θ
      (vAfter Vs x (Zs.ko x) ((Zs.preRun H w x Vs θ t).map Prod.fst)) θ o
  else 0

theorem zk_some (H : Msg → Msg) (εO : (x : Msg) → Msg → Zs.WO → (Θ : Type) → VStar Θ → Θ → Zs.St → ℝ) (δ : ℝ)
    (hHide : Lv.Hides δ) (hO : Zs.OuterZK H εO) (x : Msg) (w : W) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (hk : Vs.k x = Zs.k x) (ask : List Msg → Out) (hask : ask = fun as => ((Vs.next x as θ).eval H).1) (o : Zs.St)
    (Ev : List (Msg × Out) → Prop) :
    Pr (fun ω : Zs.N x × Zs.Ω x => Zs.outcome x (runStar H (Zs.prover w x) Vs x ω θ).msgs = some o ∧
        Ev (runStar H (Zs.prover w x) Vs x ω θ).msgs) ≤
      Pr (fun c : Zs.sim.C x => Ev ((Zs.sim.given (some o)).out H Vs θ x c)) +
        Zs.err H εO δ x w Θ Vs θ (some o) := by
  have h1 : Pr (fun ω : Zs.N x × Zs.Ω x => Zs.outcome x (runStar H (Zs.prover w x) Vs x ω θ).msgs = some o ∧
        Ev (runStar H (Zs.prover w x) Vs x ω θ).msgs) ≤
      Ex (fun t : Zs.N x => Pr (fun ρ : Zs.Ω x => Zs.someEv H w x Vs θ ask o Ev t ρ)) :=
    (Pr_mono fun ω h => Zs.someEv_of H w x Vs θ hk ask hask o Ev ω h).trans
      (le_of_eq (ZKMal.Pr_prod_left fun ω : Zs.N x × Zs.Ω x => Zs.someEv H w x Vs θ ask o Ev ω.1 ω.2))
  have h2 : ∀ t : Zs.N x, Pr (fun ρ : Zs.Ω x => Zs.someEv H w x Vs θ ask o Ev t ρ) ≤
      Zs.simAt H x ask w Vs θ o Ev t + Zs.errAt H εO x w Vs θ o t := by
    intro t
    have hpre := Zs.preRun_eq H w x Vs θ ask hask t
    have hgo := Zs.recGoesOn_preOf H x ask w t
    by_cases hc : ZKMal.goesOn (Zs.nh x) (Zs.preOf H x ask w t) = true ∧ Zs.ok w x t (Zs.preOf H x ask w t)
    · have hc' : Recursion.goesOn (Zs.preRun H w x Vs θ t) = true ∧ Zs.ok w x t (Zs.preRun H w x Vs θ t) := by
        rw [hpre, hgo]; exact hc
      have hA : Zs.simAt H x ask w Vs θ o Ev t = Pr (fun cO : Zs.CO x => Ev (Zs.preOf H x ask w t ++
          (Zs.outerSim H x Vs θ o (Zs.preOf H x ask w t) cO).map fun p => (tagOuter p.1, p.2))) :=
        Pr_congr fun cO => ⟨fun h => h.2, fun h => ⟨hc.1, h⟩⟩
      unfold errAt
      rw [ite_eq_left hc', hpre, hA]
      refine le_trans (le_of_eq (Pr_congr fun ρ => ?_))
        (hO x (Zs.xo x (Zs.preOf H x ask w t)) (Zs.wo w x t.1 t.2 (Zs.preOf H x ask w t)) hc.2 Θ
          (vAfter Vs x (Zs.ko x) ((Zs.preOf H x ask w t).map Prod.fst)) θ rfl o
          fun τ => Ev (Zs.preOf H x ask w t ++ τ.map fun p => (tagOuter p.1, p.2)))
      exact ⟨fun h => h.2, fun h => ⟨hc, h⟩⟩
    · have hc' : ¬ (Recursion.goesOn (Zs.preRun H w x Vs θ t) = true ∧ Zs.ok w x t (Zs.preRun H w x Vs θ t)) := by
        rw [hpre, hgo]; exact hc
      unfold errAt
      rw [ite_eq_right hc', add_zero]
      exact le_trans (le_of_eq ((Pr_congr fun ρ => iff_false_intro fun h => hc h.1).trans Pr_false)) (Pr_nonneg _)
  have h3 : Ex (fun t : Zs.N x => Pr (fun ρ : Zs.Ω x => Zs.someEv H w x Vs θ ask o Ev t ρ)) ≤
      Ex (fun t => Zs.simAt H x ask w Vs θ o Ev t) + Ex (fun t => Zs.errAt H εO x w Vs θ o t) :=
    (Ex_mono h2).trans (le_of_eq (Ex_add _ _))
  have h4 : Ex (fun t => Zs.simAt H x ask w Vs θ o Ev t) = Pr (fun q : Zs.N x × Zs.CO x => Zs.simEv H x ask w Vs θ o Ev q) :=
    (ZKMal.Pr_prod_left (Zs.simEv H x ask w Vs θ o Ev)).symm
  have h5 : Pr (fun q : Zs.N x × Zs.CO x => Zs.simEv H x ask w Vs θ o Ev q) ≤
      Pr (fun c : Zs.sim.C x => Ev ((Zs.sim.given (some o)).out H Vs θ x c)) + Zs.nLeaves x * δ := by
    let G : Zs.Ωp x × Zs.CO x → (Fin (Zs.nLeaves x) → Msg) → Prop := fun q cv =>
      ZKMal.goesOn (Zs.nh x) (Zs.first H x ask cv) = true ∧
        Ev (Zs.first H x ask cv ++ (Zs.outerSim H x Vs θ o (Zs.first H x ask cv) q.2).map fun p => (tagOuter p.1, p.2))
    have hH := hHide (Zs.nLeaves x) (Zs.Ωp x × Zs.CO x) (fun q => Zs.msgsOf H x ask w q.1) G
    let e1 : Zs.N x × Zs.CO x ≃ (Zs.Ωp x × Zs.CO x) × (Fin (Zs.nLeaves x) → Lv.Salt) :=
      ⟨fun q => ((q.1.1, q.2), q.1.2), fun p => ((p.1.1, p.2), p.1.2), fun _ => rfl, fun _ => rfl⟩
    let e2 : (Zs.Ωp x × Zs.CO x) × (Fin (Zs.nLeaves x) → Lv.Rand) ≃
        ((Fin (Zs.nLeaves x) → Lv.Rand) × Zs.CO x) × Zs.Ωp x :=
      ⟨fun p => ((p.2, p.1.2), p.1.1), fun z => ((z.2, z.1.2), z.1.1), fun _ => rfl, fun _ => rfl⟩
    let Q : (Fin (Zs.nLeaves x) → Lv.Rand) × Zs.CO x → Prop := fun c =>
      ZKMal.goesOn (Zs.nh x) (Zs.first H x ask fun i => Lv.ideal (c.1 i)) = true ∧
        Ev (Zs.first H x ask (fun i => Lv.ideal (c.1 i)) ++
          (Zs.outerSim H x Vs θ o (Zs.first H x ask fun i => Lv.ideal (c.1 i)) c.2).map fun p => (tagOuter p.1, p.2))
    have hl : Pr (fun q : Zs.N x × Zs.CO x => Zs.simEv H x ask w Vs θ o Ev q) =
        Pr (fun p : (Zs.Ωp x × Zs.CO x) × (Fin (Zs.nLeaves x) → Lv.Salt) =>
          G p.1 (Lv.commits ((fun q : Zs.Ωp x × Zs.CO x => Zs.msgsOf H x ask w q.1) p.1) p.2)) :=
      (ZKMal.Pr_equiv e1.symm (Zs.simEv H x ask w Vs θ o Ev)).symm
    have hr : Pr (fun p : (Zs.Ωp x × Zs.CO x) × (Fin (Zs.nLeaves x) → Lv.Rand) => G p.1 fun i => Lv.ideal (p.2 i)) =
        Pr Q :=
      (ZKMal.Pr_equiv e2 (fun z => Q z.1)).trans (Pr_fst Q)
    have hm : Pr Q ≤ Pr (fun c : Zs.sim.C x => Ev ((Zs.sim.given (some o)).out H Vs θ x c)) := by
      refine Pr_mono fun c hc => ?_
      show Ev ((Zs.sim.given (some o)).out H Vs θ x c)
      rw [Zs.sim_out_some H x Vs θ ask hask o c, ite_eq_left hc.1]
      exact hc.2
    exact (le_of_eq hl).trans (hH.trans (add_le_add ((le_of_eq hr).trans hm) le_rfl))
  have herr : Zs.err H εO δ x w Θ Vs θ (some o) =
      Ex (fun t => Zs.errAt H εO x w Vs θ o t) + Zs.nLeaves x * δ := rfl
  have h6 : Pr (fun ω : Zs.N x × Zs.Ω x => Zs.outcome x (runStar H (Zs.prover w x) Vs x ω θ).msgs = some o ∧
        Ev (runStar H (Zs.prover w x) Vs x ω θ).msgs) ≤
      Pr (fun q : Zs.N x × Zs.CO x => Zs.simEv H x ask w Vs θ o Ev q) +
        Ex (fun t => Zs.errAt H εO x w Vs θ o t) := by
    have := h1.trans h3
    rwa [h4] at this
  rw [herr]
  exact (h6.trans (add_le_add h5 le_rfl)).trans (le_of_eq (by ring))

end ZkRelease

end Claims.RecGeneric

open Claims Claims.RecGeneric in
/-- **Z1** (`Flock.Guarantees.GenericRecursiveZK`): at each outcome, `zk_none` or `zk_some`. -/
theorem Flock.SecurityProofs.GenericRecursiveZK : Flock.Guarantees.GenericRecursiveZK := by
  intro H W Lv Zs εO δ hHide hO x w _ Θ Vs θ hk s Ev
  obtain ⟨ask, hask⟩ : ∃ ask : List Msg → Out, ask = fun as => ((Vs.next x as θ).eval H).1 := ⟨_, rfl⟩
  cases s with
  | none => exact Zs.zk_none H εO δ hHide x w Vs θ hk ask hask Ev
  | some o => exact Zs.zk_some H εO δ hHide hO x w Vs θ hk ask hask o Ev
