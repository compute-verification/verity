import Proofs.Flock.Claims.ZK.Shvzk
import Proofs.Flock.Soundness.ZK.AdaptivePrefinal

/-!
# Lemma C at tables of their own types

`Session.adaptive_prefinal_hm96` (Lemma C) takes `J` tables of one type. The tables `verify --zk` sets up each have their
own witness, hidden-value and Ligerito types (`--session-tables J`). Lemma C reads the tables only through world `W₀`'s
witness independence at each fixed answer vector (`Session.card_cut_eq`), so:

* `prefinal_of_ideal`: Lemma C for a prover whose messages at fixed answers, at ideal leaves `H(U, c(y))`, take every
  value equally often on the two witnesses; the leaves' hybrid (T1) and the adaptive verifier (T5) are
  `Session.adaptive_prefinal_hm96`'s.
* `famSent`, `famSent_admits`, `famCard_cut_eq`: the session's schedule (`Session.sent`) and Lemma A at every coin vector
  (`Session.session_prefinal_indep_cut`) for tables of their own types, from each table's own Lemma A
  (`Table.table_prefinal_indep_cut`).
* `Proto.runs_close_of_ideal`, `Proto.refused_close_of_ideal`: `Proto.runs_close_hm96`'s and `Proto.shvzk_refused_hm96`'s
  bounds from `prefinal_of_ideal`.
-/

namespace Claims.ZKMal

open Claims Finset

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-! ## Lemma C from world `W₀` -/

/-- **Lemma C from world `W₀`'s witness independence** (`Session.adaptive_prefinal_hm96` with `hW₀` for its `hI`, `he`,
`hreg`): against any deterministic verifier, every event on the prover's transcript has probability on `w` at most its
probability on `w'` plus `2·|L|·δ₁`, when at every fixed answer vector its messages at ideal leaves `H(U, c(y))` take
every value equally often on `w` and on `w'`. -/
theorem prefinal_of_ideal {n : ℕ} {Cn M Cc Ω L Salt B D Wt : Type} [Inhabited M] [DecidableEq M] [Fintype Ω]
    [Nonempty Ω] [Fintype L] [DecidableEq L] [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt]
    [Fintype B] [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁)
    (Hl : B × Cc → D) (msgs : Wt → Ω → (Fin n → Cn) → (L → D) → Fin n → M)
    (dig : Wt → Ω → (Fin n → Cn) → L → B) (P : Wt → FlockSoundness.ZK.Strategy (Ω × (L → B × Cc)) M Cn n)
    (hP : ∀ w ω p c, (P w).msg (ω, p) c = msgs w ω c fun l => Hl (dig w ω c l + (p l).1, (p l).2)) (w w' : Wt)
    (hW₀ : ∀ c σ, #{u : Ω × (L → B × Salt) | msgs w u.1 c (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : Ω × (L → B × Salt) | msgs w' u.1 c (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (V : FlockSoundness.ZK.Responder M Cn n) (E : (Fin n → M) → Prop) :
    FlockSoundness.Game.prCoin (fun p : Ω × (L → Salt) =>
        E (FlockSoundness.ZK.run (P w) V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
      ≤ FlockSoundness.Game.prCoin (fun p : Ω × (L → Salt) =>
          E (FlockSoundness.ZK.run (P w') V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
        + 2 * Fintype.card L * δ₁ := by
  classical
  have : Nonempty Salt := ⟨0⟩
  let f : Ω × (L → B × Salt) → Ω × (L → B × Cc) := fun x => (x.1, fun l => ((x.2 l).1, cs (x.2 l).2))
  have hcount : ∀ w0 : Wt, ∀ c σ, #{x | ((P w0).comap f).msg x c = σ}
      = #{u : Ω × (L → B × Salt) | msgs w0 u.1 c (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ} := by
    intro w0 c σ
    calc #{x | ((P w0).comap f).msg x c = σ}
        = #{x : Ω × (L → B × Salt) | msgs w0 x.1 c (fun l => Hl (dig w0 x.1 c l + (x.2 l).1, cs (x.2 l).2)) = σ} := by
          congr 1
          ext x
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, FlockSoundness.ZK.Strategy.comap, f, hP]
      _ = _ := FlockSoundness.ZK.card_msg_pad cs Hl (fun ω leaves => msgs w0 ω c leaves) (fun ω => dig w0 ω c) σ
  have hruns : ∀ τ, #{x | FlockSoundness.ZK.run ((P w).comap f) V x = τ}
      = #{x | FlockSoundness.ZK.run ((P w').comap f) V x = τ} :=
    FlockSoundness.ZK.card_run_eq _ _ V fun c σ => by rw [hcount w c σ, hcount w' c σ]; exact hW₀ c σ
  have hpad : FlockSoundness.Game.prCoin (fun x : Ω × (L → B × Salt) => E (FlockSoundness.ZK.run (P w) V (f x)))
      = FlockSoundness.Game.prCoin (fun x : Ω × (L → B × Salt) => E (FlockSoundness.ZK.run (P w') V (f x))) :=
    (FlockSoundness.ZK.SameDist.of_card_fiber_eq hruns).prCoin_eq E
  have h1 := FlockSoundness.ZK.ideal_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc)
    (fun (_ : Ω) (_ : L) => (0 : B)) (fun ω leaves => E (FlockSoundness.ZK.run (P w) V (ω, leaves)))
  have h2 := FlockSoundness.ZK.real_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc)
    (fun (_ : Ω) (_ : L) => (0 : B)) (fun ω leaves => E (FlockSoundness.ZK.run (P w') V (ω, leaves)))
  simp only [id, zero_add] at h1 h2
  calc FlockSoundness.Game.prCoin (fun p : Ω × (L → Salt) =>
        E (FlockSoundness.ZK.run (P w) V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
      ≤ FlockSoundness.Game.prCoin (fun x : Ω × (L → B × Salt) => E (FlockSoundness.ZK.run (P w) V (f x)))
        + Fintype.card L * δ₁ := h1
    _ = FlockSoundness.Game.prCoin (fun x : Ω × (L → B × Salt) => E (FlockSoundness.ZK.run (P w') V (f x)))
        + Fintype.card L * δ₁ := by rw [hpad]
    _ ≤ (FlockSoundness.Game.prCoin (fun p : Ω × (L → Salt) =>
          E (FlockSoundness.ZK.run (P w') V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
          + Fintype.card L * δ₁) + Fintype.card L * δ₁ := add_le_add h2 le_rfl
    _ = _ := by ring

/-! ## The schedule and Lemma A, for tables of their own types -/

/-- **The check at table `j`, rep `r`, phase `p`** (`Session.guard`), for tables of their own types. -/
def famGuard {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] {J : ℕ} {Wf Hi VL : Fin J → Type}
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (j : Fin J) (r : Fin 2) : FlockSoundness.ZK.Session.Phase → Prop
  | .opening => (T j).RankOn fun k => rep j k ≤ r
  | .restart => Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2
  | .inner => (T j).β r ≠ 0

/-- **The session reaches past table `j`, rep `r`, phase `p`** (`Session.passes`): every check up to it in the session's
order passes. -/
def famPasses {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] {J : ℕ} {Wf Hi VL : Fin J → Type}
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (j : Fin J) (r : Fin 2) (p : FlockSoundness.ZK.Session.Phase) : Prop :=
  ∀ j' r' p', FlockSoundness.ZK.Session.pos j' r' p' ≤ FlockSoundness.ZK.Session.pos j r p → famGuard T rep j' r' p'

/-- **What table `j` sent** when the session stops at its first failing check (`Session.sent`), for tables of their own
types. -/
def famSent {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] {J : ℕ} {Wf Hi VL : Fin J → Type}
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (j : Fin J) : (T j).Sent where
  cl k := famPasses T rep j (rep j k) .opening
  y1 r := famPasses T rep j r .restart
  fin r := famPasses T rep j r .inner

/-- **The schedule** (`Session.sent_admits`): at every coin vector, what each table sent is admitted. -/
theorem famSent_admits {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] {J : ℕ} {Wf Hi VL : Fin J → Type}
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (j : Fin J) : (T j).Admits (famSent T rep j) := by
  refine ⟨?_, fun r hr => hr j r .restart le_rfl, fun r hr => hr j r .inner le_rfl⟩
  by_cases h1 : famPasses T rep j 1 .opening
  · refine FlockSoundness.ZK.Table.RankOn.mono (T j) (fun k _ => ?_) (h1 j 1 .opening le_rfl)
    have := (rep j k).isLt
    rw [Fin.le_iff_val_le_val]
    simp only [Fin.isValue, Fin.val_one]
    omega
  · by_cases h0 : famPasses T rep j 0 .opening
    · refine FlockSoundness.ZK.Table.RankOn.mono (T j) (fun k hk => ?_) (h0 j 0 .opening le_rfl)
      have := (rep j k).isLt
      rw [Fin.le_iff_val_le_val]
      by_contra hne
      have e : rep j k = 1 := Fin.ext (by simp only [Fin.isValue, Fin.val_zero, Fin.val_one] at hne ⊢; omega)
      exact h1 (e ▸ hk)
    · refine FlockSoundness.ZK.Table.rankOn_of_forall_not (T j) fun k hk =>
        h0 fun a b c hc => hk a b c (hc.trans ?_)
      simp only [FlockSoundness.ZK.Session.pos, FlockSoundness.ZK.Session.Phase.idx, Fin.isValue, Fin.val_zero]
      omega

/-- **Lemma A at every coin vector, for tables of their own types** (`Session.session_prefinal_indep_cut`): the session's
pre-final views in world `W₀` on two tuples of witnesses carrying the public region data, each table's cut to what it
sent, are equally distributed. -/
theorem famSession_prefinal_indep_cut {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype Seed]
    [DecidableEq Seed] {J : ℕ} {Wf Hi VL : Fin J → Type} [∀ j, AddCommGroup (Hi j)] [∀ j, Fintype (Hi j)]
    [∀ j, DecidableEq (Hi j)] (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j))
    (rep : ∀ j, Fin (T j).nC → Fin 2) (w w' : ∀ j, Wf j) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hreg' : ∀ j, (T j).sReg (w' j) = (T j).pubReg) :
    FlockSoundness.ZK.SameDist
      (fun ω : ∀ j, (T j).Rand => fun j => (T j).cut (famSent T rep j) ((T j).preView (w j) (ω j)))
      (fun ω : ∀ j, (T j).Rand => fun j => (T j).cut (famSent T rep j) ((T j).preView (w' j) (ω j))) :=
  FlockSoundness.ZK.SameDist.pi (fun j => fun ω => (T j).cut (famSent T rep j) ((T j).preView (w j) ω))
    (fun j => fun ω => (T j).cut (famSent T rep j) ((T j).preView (w' j) ω))
    fun j => (T j).table_prefinal_indep_cut (famSent T rep j) (famSent_admits T rep j) (w j) (w' j) (hreg j) (hreg' j)

/-- **At fixed answers, for tables of their own types** (`Session.card_cut_eq`): messages that are a function of the
session's cut pre-view, on the tables' randomness read through a map with fibers of one size, take every value as often
on two witnesses carrying the public region data. -/
theorem famCard_cut_eq {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype Seed]
    [DecidableEq Seed] {J : ℕ} {Wf Hi VL : Fin J → Type} [∀ j, AddCommGroup (Hi j)] [∀ j, Fintype (Hi j)]
    [∀ j, DecidableEq (Hi j)] {Ω V : Type} [Fintype Ω] [DecidableEq V]
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (e : Ω → ∀ j, (T j).Rand) {k : ℕ} (he : ∀ r, Nat.card {x // e x = r} = k) (out : (∀ j, (T j).Cut) → V)
    (w w' : ∀ j, Wf j) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hreg' : ∀ j, (T j).sReg (w' j) = (T j).pubReg) (v : V) :
    #{x | out (fun j => (T j).cut (famSent T rep j) ((T j).preView (w j) (e x j))) = v}
      = #{x | out (fun j => (T j).cut (famSent T rep j) ((T j).preView (w' j) (e x j))) = v} := by
  classical
  have : Nonempty (∀ j, (T j).Rand) := ⟨fun _ => 0⟩
  rw [FlockSoundness.ZK.card_comp_of_fiber e he
      (fun r => out fun j => (T j).cut (famSent T rep j) ((T j).preView (w j) (r j))),
    FlockSoundness.ZK.card_comp_of_fiber e he
      (fun r => out fun j => (T j).cut (famSent T rep j) ((T j).preView (w' j) (r j)))]
  congr 1
  exact ((famSession_prefinal_indep_cut T rep w w' hreg hreg').comp out).card_eq v

/-! ## The bounds `Flock.zkR_gkE_hm96` reads -/

namespace Proto

/-- **Lemma C's bound on the runs before the last coin, from world `W₀`** (`Proto.runs_close_hm96` with `hW₀` for its
tables): the prover's messages before the last coin on `w` and on the dummy are `P`'s on real hm96 leaves, and `P`'s
messages at fixed answers and ideal leaves take every value as often on both table witnesses. -/
theorem runs_close_of_ideal {W : Type} (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (w : W)
    {Salt B Cc D Wt Ω L : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] [Fintype Ω] [Nonempty Ω] [Fintype L] [DecidableEq L] (My : Salt → B) (cs : Salt → Cc)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (Hl : B × Cc → D)
    (msgs : Wt → Ω → (Fin (G.K x + 1) → Out) → (L → D) → Fin (G.K x + 1) → Msg)
    (dig : Wt → Ω → (Fin (G.K x + 1) → Out) → L → B)
    (P : Wt → FlockSoundness.ZK.Strategy (Ω × (L → B × Cc)) Msg Out (G.K x + 1))
    (hP : ∀ wt ω p a, (P wt).msg (ω, p) a = msgs wt ω a fun l => Hl (dig wt ω a l + (p l).1, (p l).2))
    (wt wt₀ : Wt)
    (hW₀ : ∀ a σ, #{u : Ω × (L → B × Salt) | msgs wt u.1 a (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : Ω × (L → B × Salt) | msgs wt₀ u.1 a (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (eR : G.Rand → Ω × (L → Salt)) (kR : ℕ) (heR : ∀ p, Nat.card {r // eR r = p} = kR)
    (hlink : ∀ r, (G.preStrat H x ν (G.msg x w ν)).msg r =
      (P wt).msg ((eR r).1, fun l => (My ((eR r).2 l), cs ((eR r).2 l))))
    (hlink₀ : ∀ r, (G.preStrat H x ν (G.dmsg x ν)).msg r =
      (P wt₀).msg ((eR r).1, fun l => (My ((eR r).2 l), cs ((eR r).2 l))))
    (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1)) (E : (Fin (G.K x + 1) → Msg) → Prop) :
    |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
      Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))|
      ≤ 2 * Fintype.card L * δ₁.toReal := by
  have : Nonempty Salt := ⟨0⟩
  have eW : Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (FlockSoundness.ZK.run (P wt) V (p.1, fun l => (My (p.2 l), cs (p.2 l))))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    congr 1
    funext r
    rw [run_congr _ _ V r _ (hlink r)]
  have eD : Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (FlockSoundness.ZK.run (P wt₀) V (p.1, fun l => (My (p.2 l), cs (p.2 l))))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    congr 1
    funext r
    rw [run_congr _ _ V r _ (hlink₀ r)]
  have h1 := prefinal_of_ideal My cs hT1 Hl msgs dig P hP wt wt₀ hW₀ V E
  have h2 := prefinal_of_ideal My cs hT1 Hl msgs dig P hP wt₀ wt (fun a σ => (hW₀ a σ).symm) V E
  have hε : 2 * (Fintype.card L : ENNReal) * δ₁ ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hεr : (2 * (Fintype.card L : ENNReal) * δ₁).toReal = 2 * Fintype.card L * δ₁.toReal := by
    simp [ENNReal.toReal_mul]
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h2
  rw [hεr] at a b
  rw [eW, eD, abs_le]
  constructor <;> linarith

/-- **Lemma C at a coin vector where the prover refuses, from world `W₀`** (`Proto.shvzk_refused_hm96`'s bound with `hW₀`
for its tables): the honest prover's messages at `c` and the dummy-witness prover's are `msgs` of real hm96 leaves, and
`msgs` at ideal leaves takes every value as often on both table witnesses. -/
theorem refused_close_of_ideal {W : Type} (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) {Salt B Cc D Wt Ω L : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt]
    [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Ω] [Nonempty Ω] [Fintype L] [DecidableEq L]
    (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Hl : B × Cc → D) (msgs : Wt → Ω → (L → D) → List Msg) (dig : Wt → Ω → L → B) (wt wt₀ : Wt)
    (hW₀ : ∀ σ, #{u : Ω × (L → B × Salt) | msgs wt u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : Ω × (L → B × Salt) | msgs wt₀ u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (eR : G.Rand → Ω × (L → Salt)) (kR : ℕ) (heR : ∀ p, Nat.card {r // eR r = p} = kR)
    (hR : ∀ r, G.realVec H x w ν r c =
      msgs wt (eR r).1 fun l => Hl (dig wt (eR r).1 l + My ((eR r).2 l), cs ((eR r).2 l)))
    (hD : ∀ r, G.dVec H x ν r c =
      msgs wt₀ (eR r).1 fun l => Hl (dig wt₀ (eR r).1 l + My ((eR r).2 l), cs ((eR r).2 l)))
    (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun r : G.Rand => E (G.dVec H x ν r c))|
      ≤ 2 * Fintype.card L * δ₁.toReal := by
  classical
  have : Nonempty Salt := ⟨0⟩
  have one : ∀ (m : Ω × (L → B × Salt) → List Msg) (σ : Fin 1 → List Msg),
      #{u | (fun _ : Fin 1 => m u) = σ} = #{u | m u = σ 0} := by
    intro m σ
    congr 1
    ext u
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨fun h => congrFun h 0, fun h => funext fun i => by rw [Subsingleton.elim i 0]; exact h⟩
  have hW₁ : ∀ wa wb : Wt, (∀ σ, #{u : Ω × (L → B × Salt) | msgs wa u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : Ω × (L → B × Salt) | msgs wb u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}) →
      ∀ (_ : Fin 1 → Unit) (σ : Fin 1 → List Msg),
      #{u : Ω × (L → B × Salt) | (fun _ : Fin 1 => msgs wa u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2))) = σ}
        = #{u : Ω × (L → B × Salt) | (fun _ : Fin 1 => msgs wb u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2))) = σ} :=
    fun wa wb h _ σ => by rw [one, one]; exact h (σ 0)
  have h1 := prefinal_of_ideal (n := 1) (Cn := Unit) My cs hT1 Hl (fun wt' ω _ ls _ => msgs wt' ω ls)
    (fun wt' ω _ l => dig wt' ω l)
    (fun wt' => oneRound fun p => msgs wt' p.1 fun l => Hl (dig wt' p.1 l + (p.2 l).1, (p.2 l).2))
    (fun _ _ _ _ => rfl) wt wt₀ (hW₁ wt wt₀ hW₀) (unitResp _) (fun τ => E (τ 0))
  have h2 := prefinal_of_ideal (n := 1) (Cn := Unit) My cs hT1 Hl (fun wt' ω _ ls _ => msgs wt' ω ls)
    (fun wt' ω _ l => dig wt' ω l)
    (fun wt' => oneRound fun p => msgs wt' p.1 fun l => Hl (dig wt' p.1 l + (p.2 l).1, (p.2 l).2))
    (fun _ _ _ _ => rfl) wt₀ wt (hW₁ wt₀ wt fun σ => (hW₀ σ).symm) (unitResp _) (fun τ => E (τ 0))
  simp only [run_oneRound] at h1 h2
  have eW : Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (msgs wt p.1 fun l => Hl (dig wt p.1 l + My (p.2 l), cs (p.2 l)))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    simp only [hR]
  have eD : Pr (fun r : G.Rand => E (G.dVec H x ν r c)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (msgs wt₀ p.1 fun l => Hl (dig wt₀ p.1 l + My (p.2 l), cs (p.2 l)))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    simp only [hD]
  have hε : 2 * (Fintype.card L : ENNReal) * δ₁ ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hεr : (2 * (Fintype.card L : ENNReal) * δ₁).toReal = 2 * Fintype.card L * δ₁.toReal := by
    simp [ENNReal.toReal_mul]
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h2
  rw [hεr] at a b
  rw [eW, eD, abs_le]
  constructor <;> linarith

end Proto

end Claims.ZKMal
