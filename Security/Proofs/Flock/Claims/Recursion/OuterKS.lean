import Proofs.Flock.Claims.Recursion.KS
import Proofs.Flock.Claims.Recursion.CostExp

namespace Claims.Recursion

open Claims

/-! ## Maps that keep coins uniform -/

/-- **`φ` keeps uniform coins uniform**: every average over `B` is the average over `A` read through `φ`. -/
def UnifMap {A B : Type} [Fintype A] [Fintype B] (φ : A → B) : Prop :=
  ∀ G : B → ℝ, Ex (fun a => G (φ a)) = Ex G

theorem Ex_equiv {A B : Type} [Fintype A] [Fintype B] (e : A ≃ B) (G : B → ℝ) :
    Ex (fun a => G (e a)) = Ex G := by
  unfold Ex
  rw [Fintype.card_congr e, e.sum_comp G]

theorem UnifMap.equiv {A B : Type} [Fintype A] [Fintype B] (e : A ≃ B) : UnifMap e := Ex_equiv e

theorem UnifMap.comp {A B C : Type} [Fintype A] [Fintype B] [Fintype C] {φ : A → B} {ψ : B → C}
    (hφ : UnifMap φ) (hψ : UnifMap ψ) : UnifMap (fun a => ψ (φ a)) := fun G => by
  rw [hφ (fun b => G (ψ b)), hψ G]

theorem UnifMap.Pr {A B : Type} [Fintype A] [Fintype B] {φ : A → B} (hφ : UnifMap φ) (p : B → Prop) :
    Claims.Pr (fun a => p (φ a)) = Claims.Pr p := by
  classical
  rw [← Ex_ind, ← Ex_ind]
  exact hφ fun b => if p b then (1 : ℝ) else 0

theorem UnifMap.prod {A B C D : Type} [Fintype A] [Fintype B] [Fintype C] [Fintype D] {φ : A → B} {ψ : C → D}
    (hφ : UnifMap φ) (hψ : UnifMap ψ) : UnifMap (fun p : A × C => (φ p.1, ψ p.2)) := fun G => by
  rw [Ex_prod, Ex_prod (g := G)]
  simp only [fun a => hψ (fun d => G (φ a, d))]
  exact hφ fun b => Ex fun d => G (b, d)

/-- **Coins read at an embedding are uniform**: the coordinates `e` picks out of uniform coins. -/
theorem UnifMap.comp_emb {α β γ : Type} [Fintype α] [Fintype β] [Fintype γ] [Nonempty γ] [DecidableEq α]
    [DecidableEq β] (e : β ↪ α) : UnifMap (fun f : α → γ => f ∘ e) := fun F => by
  classical
  let p : α → Prop := fun a => a ∈ Set.range e
  let E := Equiv.piEquivPiSubtypeProd p (fun _ => γ)
  let φ : β ≃ Set.range e := Equiv.ofInjective e e.injective
  let g : (Subtype p → γ) → ℝ := fun h => F (fun b => h (φ b))
  have h1 : (fun f : α → γ => F (f ∘ e)) = fun f => g (E f).1 := by
    funext f
    rfl
  have h2 : Ex g = Ex F := by
    have hg : g = fun h => F (Equiv.arrowCongr φ.symm (Equiv.refl γ) h) := funext fun h => rfl
    rw [hg]
    exact Ex_equiv (Equiv.arrowCongr φ.symm (Equiv.refl γ)) F
  rw [h1, ← Ex_equiv E.symm (fun f => g (E f).1)]
  simp only [Equiv.apply_symm_apply]
  rw [Ex_prod]
  simp only [Ex_const]
  exact h2

/-- **Uniform maps, coordinatewise**: `φ` at each of `m` coordinates. -/
theorem UnifMap.pi {A B : Type} [Fintype A] [Fintype B] {φ : A → B} (hφ : UnifMap φ) :
    ∀ m : ℕ, UnifMap (fun c : Fin m → A => fun i => φ (c i))
  | 0 => fun G => by
    unfold Ex
    rw [Fintype.sum_unique, Fintype.sum_unique, Fintype.card_unique, Fintype.card_unique]
    congr 2
    exact Subsingleton.elim _ _
  | m + 1 => fun G => by
    let e := Fin.consEquiv (fun _ : Fin (m + 1) => A)
    let e' := Fin.consEquiv (fun _ : Fin (m + 1) => B)
    have hc : ∀ q : A × (Fin m → A), (fun i => φ (e q i)) = e' (φ q.1, fun i => φ (q.2 i)) := fun q =>
      Fin.comp_cons φ q.1 q.2
    rw [← Ex_equiv e (fun c => G fun i => φ (c i)), ← Ex_equiv e' G]
    simp only [hc]
    exact (hφ.prod (UnifMap.pi hφ m)) fun q => G (e' q)

/-! ## One session's prover -/

/-- What a verifier's answer tells the prover: the message, or nothing once it decided. -/
def outMsg : Out → Msg
  | .msg m => m
  | _ => []

/-- **The prover of one session**: `P` at tape `ω`, after the messages `pre` of the rounds before the session. -/
def sessProver {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (pre : List (Msg × Out)) : Prover Unit :=
  ⟨fun _ hist _ => P.next x (pre.map (fun p => (p.1, outMsg p.2)) ++ hist) ω⟩

/-- The `n` coins from round `o` on. -/
def sessCoins {Coin : Type} [Inhabited Coin] {k : ℕ} (o : ℕ) (d : Fin k → Coin) (n : ℕ) : Fin n → Coin :=
  fun i => coinSeq d (o + i.val)

/-- Session `s`'s V* statement in the run at coins `r`, read off its inner rounds' messages. -/
def sessStmt {Coin W : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ) (r : Fin (D.V.k x) → Coin) : Msg :=
  (D.vstar x ((run H P D.V x ω r).msgs.take (I.V.k (xI D x)))).getD s []

/-- **Session `s`'s prover at the coins `r`**: `P` after the run's messages before the session's first round. -/
def sessP {Coin W Col T : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ)
    (r : Fin (D.V.k x) → Coin) : Prover Unit :=
  sessProver P x ω ((run H P D.V x ω r).msgs.take (offO H D I DO P x ω s r))

/-- `E_O`'s coins at session `s` as the outer `E^rep`'s at the session's statement: each rerun's coins from round `o`
on, the waits and truncations recounted at the statement's own. -/
def castCoins {Coin : Type} [Inhabited Coin] {W W' K K' k : ℕ} (hW : W' = W) (hK : K' = K) (o n : ℕ)
    (c : Fin W → Fin K → Fin k → Coin) : Fin W' → Fin K' → Fin n → Coin :=
  fun i j => sessCoins o (c (Fin.cast hW i) (Fin.cast hK j)) n

/-! ## What the compiled verifier does at each session (code facts) -/

/-- **Each session's numbers are the outer Flock's at its V* statement** (a code fact): whatever the inner transcript,
session `s`'s V* statement has `D.nums x s`'s rounds, waits, truncation, `N₀`, table error and weight. -/
def SessionNums {Coin Col T : Type} (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (s : Fin (D.nS x)) (ms : List (Msg × Out)),
    DO.V.k ((D.vstar x ms).getD s []) = (D.nums x s).rounds ∧
      DO.waits ((D.vstar x ms).getD s []) = (D.nums x s).waits ∧
      DO.cap ((D.vstar x ms).getD s []) = (D.nums x s).cap ∧
      DO.N₀ ((D.vstar x ms).getD s []) = (D.nums x s).N₀ ∧
      DO.εtable ((D.vstar x ms).getD s []) = (D.nums x s).εtable ∧
      DO.weight ((D.vstar x ms).getD s []) = (D.nums x s).weight

/-- **Each session's rounds are the compiled verifier's** (a code fact): session `s`'s rounds, after the inner rounds
and the earlier sessions', end by `k(x)`. -/
def SessionRounds {Coin W Col T : Type} (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (s : Fin (D.nS x)) (ms : List (Msg × Out)),
    I.V.k (xI D x) + (((D.vstar x ms).take s).map DO.V.k).sum + DO.V.k ((D.vstar x ms).getD s []) ≤ D.V.k x

/-- **Each session's table error is nonnegative.** -/
def TableErrNonneg {Coin : Type} (D : Deploy Coin) : Prop :=
  ∀ (x : Msg) (s : Fin (D.nS x)), 0 ≤ (D.nums x s).εtable

/-- **The compiled verifier runs session `s` as the outer Flock's verifier at its V* statement** (a code fact): on a run
that accepts, each fork at the session's first round shows through `D.session` the run of `P`'s continuation after the
run's earlier messages (`sessP`) against the outer verifier at the fork's coins from that round on, and the outer
verifier's SHA-512 calls in that run are among the compiled verifier's. -/
def SessionRuns {Coin W Col T : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (s : Fin (D.nS x)) (Ω : Type) (P : Prover Ω) (ω : Ω) (r d : Fin (D.V.k x) → Coin),
    (run H P D.V x ω r).out = .acc →
    (run H (sessP H D I DO P x ω s r) DO.V (sessStmt H D I P x ω s r) ()
        (sessCoins (offO H D I DO P x ω s r) d _)).seen =
      D.session x s (run H P D.V x ω (fork (offO H D I DO P x ω s r) r d)).seen ∧
    ∀ e ∈ (run H (sessP H D I DO P x ω s r) DO.V (sessStmt H D I P x ω s r) ()
        (sessCoins (offO H D I DO P x ω s r) d _)).vLog,
      e ∈ (run H P D.V x ω (fork (offO H D I DO P x ω s r) r d)).vLog

/-! ## Forks and prefixes -/

theorem fork_fork {Coin : Type} {k : ℕ} (o : ℕ) (r d e : Fin k → Coin) : fork o (fork o r d) e = fork o r e := by
  funext i
  by_cases hi : i.val < o <;> simp [fork, hi]

theorem fork_fork_le {Coin : Type} {k : ℕ} {o o' : ℕ} (h : o ≤ o') (R d : Fin k → Coin) :
    fork o' R (fork o R d) = fork o' R d := by
  funext i
  simp only [fork]
  split_ifs <;> first | rfl | omega

theorem fork_add_fork {Coin : Type} {k : ℕ} (o j : ℕ) (r d d' : Fin k → Coin) :
    fork o r (fork (o + j) d d') = fork (o + j) (fork o r d) d' := by
  funext i
  simp only [fork]
  split_ifs <;> first | rfl | omega

theorem sessCoins_fork {Coin : Type} [Inhabited Coin] {k : ℕ} (o n : ℕ) (r d : Fin k → Coin) :
    sessCoins o (fork o r d) n = sessCoins o d n := by
  funext i
  simp only [sessCoins, coinSeq, fork]
  split_ifs with h h' <;> first | rfl | omega

theorem fork_sessCoins {Coin : Type} [Inhabited Coin] {k : ℕ} (o j n : ℕ) (d d' : Fin k → Coin) :
    fork j (sessCoins o d n) (sessCoins o d' n) = sessCoins o (fork (o + j) d d') n := by
  funext i
  simp only [sessCoins, coinSeq, fork]
  split_ifs <;> first | rfl | omega

theorem le_offO {Coin W Col T : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ) (r : Fin (D.V.k x) → Coin) :
    I.V.k (xI D x) ≤ offO H D I DO P x ω s r := Nat.le_add_right _ _

theorem offO_fork {Coin W Col T : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ) (r d : Fin (D.V.k x) → Coin) :
    offO H D I DO P x ω s (fork (offO H D I DO P x ω s r) r d) = offO H D I DO P x ω s r :=
  offO_prefixDet H D I DO P x ω s r _ (fork_agree _ r d)

theorem msgs_take_fork {Coin W Col T : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ) (r d : Fin (D.V.k x) → Coin)
    {n : ℕ} (hn : n ≤ offO H D I DO P x ω s r) :
    (run H P D.V x ω (fork (offO H D I DO P x ω s r) r d)).msgs.take n = (run H P D.V x ω r).msgs.take n :=
  (run_msgs_take H P D.V x ω n fun i hi => fork_agree _ r d i (by omega)).symm

theorem sessStmt_fork {Coin W Col T : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ) (r d : Fin (D.V.k x) → Coin) :
    sessStmt H D I P x ω s (fork (offO H D I DO P x ω s r) r d) = sessStmt H D I P x ω s r := by
  unfold sessStmt
  rw [msgs_take_fork H D I DO P x ω s r d (le_offO H D I DO P x ω s r)]

theorem sessP_fork {Coin W Col T : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ) (r d : Fin (D.V.k x) → Coin) :
    sessP H D I DO P x ω s (fork (offO H D I DO P x ω s r) r d) = sessP H D I DO P x ω s r := by
  unfold sessP
  rw [offO_fork H D I DO P x ω s r d, msgs_take_fork H D I DO P x ω s r d le_rfl]

/-! ## Session `s`'s extraction is the outer Flock's `E^rep` at the session's prover -/

theorem waitAcc_eval_map {R₁ R₂ : Type} (H : Msg → Msg) (run₁ : R₁ → Trace) (run₂ : R₂ → Trace) (μ : R₁ → R₂)
    (k : ℕ) : ∀ (rs : List R₁) (acc : List Seen), (∀ r ∈ rs, (run₁ r).seen = (run₂ (μ r)).seen) →
      ((waitAcc k rs acc).eval H run₁).1 = ((waitAcc k (rs.map μ) acc).eval H run₂).1
  | [], _, _ => rfl
  | r :: rs, acc, h => by
    simp only [List.map_cons, waitAcc]
    split_ifs with hk
    · rfl
    · simp only [RP.eval]
      rw [h r List.mem_cons_self]
      exact waitAcc_eval_map H run₁ run₂ μ k rs _ fun r' hr' => h r' (List.mem_cons_of_mem _ hr')

theorem firstWait_eval_map {R₁ R₂ W' Op : Type} (H : Msg → Msg) (run₁ : R₁ → Trace) (run₂ : R₂ → Trace)
    (μ : R₁ → R₂) (k : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W') (check : W' → Bool) :
    ∀ rss : List (List R₁), (∀ l ∈ rss, ∀ r ∈ l, (run₁ r).seen = (run₂ (μ r)).seen) →
      ((firstWait k opens assemble check rss).eval H run₁).1 =
        ((firstWait k opens assemble check (rss.map (List.map μ))).eval H run₂).1
  | [], _ => rfl
  | rs :: rss, h => by
    simp only [List.map_cons, firstWait]
    rw [(RP.eval_bind H run₁ _ _).1, (RP.eval_bind H run₂ _ _).1,
      ← waitAcc_eval_map H run₁ run₂ μ k rs [] (h rs List.mem_cons_self)]
    have ih := firstWait_eval_map H run₁ run₂ μ k opens assemble check rss
      fun l hl => h l (List.mem_cons_of_mem _ hl)
    rcases assemble (((waitAcc k rs []).eval H run₁).1.flatMap opens) with _ | w
    · exact ih
    · simp only
      split_ifs
      · rfl
      · exact ih

theorem castCoins_lists {Coin : Type} [Inhabited Coin] {W W' K K' k : ℕ} (hW : W' = W) (hK : K' = K) (o n : ℕ)
    (c : Fin W → Fin K → Fin k → Coin) :
    ((List.finRange W').map fun i => (List.finRange K').map (castCoins hW hK o n c i)) =
      (List.finRange W).map fun i => (List.finRange K).map fun j => sessCoins o (c i j) n := by
  subst hW
  subst hK
  rfl

theorem EOs_eval_acc {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (x : Msg) (s : ℕ) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r : Fin (D.V.k x) → Coin) (c : (EOs H D I DO decO s).C x) (hacc : (run H P D.V x ω r).out = .acc) :
    ((EOs H D I DO decO s).eval H P x ω r c).1 =
      ((firstWait (D.nums x s).k DO.cols (Flock.assemble H DO.parse decO DO.read (sessStmt H D I P x ω s r))
        (Flock.check H DO.parse (sessStmt H D I P x ω s r))
        ((List.finRange (D.nums x s).waits).map fun i => (List.finRange (D.nums x s).cap).map fun j =>
          fork (offO H D I DO P x ω s r) r (c i j))).eval H
        (servedRun H (viewing (D.session x s)) (run H P D.V x ω))).1.map fun w => (sessStmt H D I P x ω s r, w) := by
  simp only [Extractor.eval, EOs, RP.eval, Trace.seen, hacc, ↓reduceIte]
  rw [RP.eval_bind_ret, serve_eq, (RP.serve_eval H _ _ _).1]
  rfl

theorem flockE_eval_acc {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (Q : Prover Unit) (y : Msg)
    (σ₀ : Fin (DO.V.k y) → Coin) (ρ : (Flock.E H DO decO).C y) (hacc : (run H Q DO.V y () σ₀).out = .acc) :
    ((Flock.E H DO decO).eval H Q y () σ₀ ρ).1 =
      ((firstWait (DO.k y) DO.cols (Flock.assemble H DO.parse decO DO.read y) (Flock.check H DO.parse y)
        ((List.finRange (DO.waits y)).map fun i => (List.finRange (DO.cap y)).map (ρ i))).eval H
        (run H Q DO.V y ())).1 := by
  simp only [Extractor.eval, Flock.E, ERep, RP.eval, Trace.seen, hacc, ↓reduceIte]

/-- `SessionRuns` at a prefix some continuation of which accepts: every fork at the session's first round shows the
session prover's run, and the outer verifier's calls in it are the compiled verifier's. -/
theorem sessionRuns_fork {Coin W Col T : Type} [Inhabited Coin] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} (hR : SessionRuns H D I DO) (x : Msg) (s : Fin (D.nS x)) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (r d₁ d : Fin (D.V.k x) → Coin)
    (hacc : (run H P D.V x ω (fork (offO H D I DO P x ω s r) r d₁)).out = .acc) :
    (run H (sessP H D I DO P x ω s r) DO.V (sessStmt H D I P x ω s r) ()
        (sessCoins (offO H D I DO P x ω s r) d _)).seen =
      D.session x s (run H P D.V x ω (fork (offO H D I DO P x ω s r) r d)).seen ∧
    ∀ e ∈ (run H (sessP H D I DO P x ω s r) DO.V (sessStmt H D I P x ω s r) ()
        (sessCoins (offO H D I DO P x ω s r) d _)).vLog,
      e ∈ (run H P D.V x ω (fork (offO H D I DO P x ω s r) r d)).vLog := by
  have h := hR x s Ω P ω _ d hacc
  rw [offO_fork, sessP_fork, sessStmt_fork, fork_fork] at h
  exact h

theorem sessR_ext_map {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (DO : Flock.Deploy Coin Col T) (x y : Msg) (o : Option Flock.Wit) :
    (sessR H DO).ext x (o.map fun w => (y, w)) ↔ (Flock.R H DO.parse).ext y o := by
  cases o <;> rfl

/-- **Session `s`'s failure at a fork is the outer Flock's at the session's prover**: on an accepted fork at the
session's first round where `E_O`'s session extraction fails, the session prover's run at the fork's coins from that
round on accepts, and the outer `E^rep` at the session's statement fails on `E_O`'s coins read the same way. -/
theorem bad_fork {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hS : SessionAccepts D) (hR : SessionRuns H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg)
    (s : Fin (D.nS x)) {Ω : Type} (P : Prover Ω) (ω : Ω) (r d : Fin (D.V.k x) → Coin)
    (c : (EOs H D I DO decO s).C x)
    (hW : DO.waits (sessStmt H D I P x ω s r) = (D.nums x s).waits)
    (hK : DO.cap (sessStmt H D I P x ω s r) = (D.nums x s).cap)
    (hk : DO.k (sessStmt H D I P x ω s r) = (D.nums x s).k)
    (h : (run H P D.V x ω (fork (offO H D I DO P x ω s r) r d)).out = .acc ∧
      ¬ (sessR H DO).ext x ((EOs H D I DO decO s).eval H P x ω (fork (offO H D I DO P x ω s r) r d) c).1) :
    (run H (sessP H D I DO P x ω s r) DO.V (sessStmt H D I P x ω s r) ()
        (sessCoins (offO H D I DO P x ω s r) d _)).out = .acc ∧
      ¬ (Flock.R H DO.parse).ext (sessStmt H D I P x ω s r)
        ((Flock.E H DO decO).eval H (sessP H D I DO P x ω s r) (sessStmt H D I P x ω s r) ()
          (sessCoins (offO H D I DO P x ω s r) d _) (castCoins hW hK (offO H D I DO P x ω s r) _ c)).1 := by
  obtain ⟨hacc, hbad⟩ := h
  have hrun := sessionRuns_fork hR x s P ω r d
  have hacc' : (run H (sessP H D I DO P x ω s r) DO.V (sessStmt H D I P x ω s r) ()
      (sessCoins (offO H D I DO P x ω s r) d _)).out = .acc :=
    (congrArg Seen.out (hrun d hacc).1).trans (hS x s _ s.isLt hacc)
  refine ⟨hacc', fun hF => hbad ?_⟩
  rw [flockE_eval_acc H DO decO _ _ _ (castCoins hW hK (offO H D I DO P x ω s r) _ c) hacc',
    castCoins_lists hW hK (offO H D I DO P x ω s r) _ c, hk] at hF
  rw [EOs_eval_acc H D I DO decO x s P ω _ c hacc]
  simp only [sessStmt_fork, offO_fork, fork_fork]
  rw [sessR_ext_map, firstWait_eval_map H _ (run H (sessP H D I DO P x ω s r) DO.V (sessStmt H D I P x ω s r) ())
    (fun e => sessCoins (offO H D I DO P x ω s r) e (DO.V.k (sessStmt H D I P x ω s r)))]
  · simpa only [List.map_map, Function.comp_def, sessCoins_fork] using hF
  · intro l hl e he
    obtain ⟨i, -, rfl⟩ := List.mem_map.1 hl
    obtain ⟨j, -, rfl⟩ := List.mem_map.1 he
    rw [sessCoins_fork, (hrun (c i j) hacc).1]
    rfl

/-! ## The session's coins are uniform -/

theorem UnifMap.unit {A : Type} [Fintype A] : UnifMap (fun a : A => ((), a)) :=
  UnifMap.equiv (Equiv.punitProd A).symm

theorem unifMap_sessCoins {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {k : ℕ} (o n : ℕ)
    (h : o + n ≤ k) : UnifMap (fun d : Fin k → Coin => sessCoins o d n) := by
  let e : Fin n ↪ Fin k := ⟨fun i => ⟨o + i.val, by omega⟩, fun i j hij => by
    simp only [Fin.mk.injEq, Nat.add_left_cancel_iff] at hij
    exact Fin.ext hij⟩
  have he : (fun d : Fin k → Coin => sessCoins o d n) = fun d => d ∘ e := by
    funext d i
    simp only [sessCoins, coinSeq, Function.comp_apply, show o + i.val < k by omega, ↓reduceDIte]
    rfl
  rw [he]
  exact UnifMap.comp_emb e

theorem unifMap_castCoins {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {W W' K K' k : ℕ}
    (hW : W' = W) (hK : K' = K) (o n : ℕ) (h : o + n ≤ k) :
    UnifMap (castCoins (Coin := Coin) (k := k) hW hK o n) := by
  subst hW
  subst hK
  exact UnifMap.pi (UnifMap.pi (unifMap_sessCoins o n h) _) _

end Claims.Recursion
