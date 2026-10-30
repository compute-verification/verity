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

/-- Session `s`'s V* statement in the run at coins `r`, read off its inner and staging rounds' messages. -/
def sessStmt {Coin W : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ) (r : Fin (D.V.k x) → Coin) : Msg :=
  (D.vstar x ((run H P D.V x ω r).msgs.take (I.V.k (xI D x) + D.kS x))).getD s []

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
    I.V.k (xI D x) + D.kS x + (((D.vstar x ms).take s).map DO.V.k).sum + DO.V.k ((D.vstar x ms).getD s []) ≤ D.V.k x

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
    I.V.k (xI D x) + D.kS x ≤ offO H D I DO P x ω s r := Nat.le_add_right _ _

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
    (r : Fin (D.V.k x) → Coin) (c : (EOs D I DO decO s).C x) (hacc : (run H P D.V x ω r).out = .acc) :
    ((EOs D I DO decO s).eval H P x ω r c).1 =
      ((firstWait (D.nums x s).k (fun v => [DO.cols v])
        (Flock.assemble DO.parse DO.table decO DO.read (sessStmt H D I P x ω s r))
        (Flock.check DO.parse (sessStmt H D I P x ω s r))
        ((List.finRange (D.nums x s).waits).map fun i => (List.finRange (D.nums x s).cap).map fun j =>
          fork (offO H D I DO P x ω s r) r (c i j))).eval H
        (servedRun H (viewing (D.session x s)) (run H P D.V x ω))).1.map fun w => (sessStmt H D I P x ω s r, w) := by
  simp only [Extractor.eval, EOs, RP.eval, Trace.seen, hacc, ↓reduceIte]
  rw [RP.eval_bind_ret, serve_eq, (RP.serve_eval H _ _ _).1]
  rfl

theorem flockE_eval_acc {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (Q : Prover Unit) (y : Msg)
    (σ₀ : Fin (DO.V.k y) → Coin) (ρ : (Flock.E DO decO).C y) (hacc : (run H Q DO.V y () σ₀).out = .acc) :
    ((Flock.E DO decO).eval H Q y () σ₀ ρ).1 =
      ((firstWait (DO.k y) (fun v => [DO.cols v]) (Flock.assemble DO.parse DO.table decO DO.read y)
        (Flock.check DO.parse y)
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
    (DO : Flock.Deploy Coin Col T) (x y : Msg) (o : Option Flock.Wit) :
    (sessR DO).ext x (o.map fun w => (y, w)) ↔ (Flock.R DO.parse).ext y o := by
  cases o <;> rfl

/-- **Session `s`'s failure at a fork is the outer Flock's at the session's prover**: on an accepted fork at the
session's first round where `E_O`'s session extraction fails, the session prover's run at the fork's coins from that
round on accepts, and the outer `E^rep` at the session's statement fails on `E_O`'s coins read the same way. -/
theorem bad_fork {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hS : SessionAccepts D) (hR : SessionRuns H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg)
    (s : Fin (D.nS x)) {Ω : Type} (P : Prover Ω) (ω : Ω) (r d : Fin (D.V.k x) → Coin)
    (c : (EOs D I DO decO s).C x)
    (hW : DO.waits (sessStmt H D I P x ω s r) = (D.nums x s).waits)
    (hK : DO.cap (sessStmt H D I P x ω s r) = (D.nums x s).cap)
    (hk : DO.k (sessStmt H D I P x ω s r) = (D.nums x s).k)
    (h : (run H P D.V x ω (fork (offO H D I DO P x ω s r) r d)).out = .acc ∧
      ¬ (sessR DO).ext x ((EOs D I DO decO s).eval H P x ω (fork (offO H D I DO P x ω s r) r d) c).1) :
    (run H (sessP H D I DO P x ω s r) DO.V (sessStmt H D I P x ω s r) ()
        (sessCoins (offO H D I DO P x ω s r) d _)).out = .acc ∧
      ¬ (Flock.R DO.parse).ext (sessStmt H D I P x ω s r)
        ((Flock.E DO decO).eval H (sessP H D I DO P x ω s r) (sessStmt H D I P x ω s r) ()
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

theorem Pr_prod {A B : Type} [Fintype A] [Fintype B] (q : A × B → Prop) :
    Pr q = Ex fun a => Pr fun b => q (a, b) := by
  classical
  rw [← Ex_ind, Ex_prod]
  congr 1
  funext a
  exact Ex_ind _

/-! ## Each prefix: the outer Flock's KS at the session prover -/

theorem Nums.κ_nonneg (n : Nums) (h : 0 ≤ n.εtable) : 0 ≤ n.κ := by
  unfold Nums.κ
  have h1 : (0 : ℝ) ≤ 2 ^ (-(n.waits : ℝ)) := Real.rpow_nonneg (by norm_num) _
  have h2 : (0 : ℝ) ≤ 2 * n.k / n.cap := by positivity
  have h3 := Real.exp_pos (-(n.k : ℝ) / 8)
  linarith

open Classical in
/-- **Session `s`'s collision probability at the prefix of `r`**: the outer Flock's `Pr[B]` at the session prover and
statement, where some continuation of the prefix accepts; `0` where none does. -/
noncomputable def sessPrB {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (s : ℕ) (r : Fin (D.V.k x) → Coin) : ℝ :=
  if ∃ d, (run H P D.V x ω (fork (offO H D I DO P x ω s r) r d)).out = .acc then
    PrB H DO.V (sessP H D I DO P x ω s r) (sessStmt H D I P x ω s r)
  else 0

theorem sessPrB_nonneg {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (s : ℕ) (r : Fin (D.V.k x) → Coin) : 0 ≤ sessPrB H D I DO P x ω s r := by
  unfold sessPrB
  split_ifs
  · exact PrB_nonneg _ _ _ _
  · exact le_rfl

/-- **One prefix**: over the forks of `r` at session `s`'s first round and `E_O`'s coins, session `s`'s extraction
fails at most as often as the outer Flock's `E^rep` at the session prover (`bad_fork`, the coins read uniformly), which
Flock's KS bounds at the session's numbers; it never fails where no fork accepts. -/
theorem prefix_bad_le {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T} (hF : Flock.KS H DO)
    (hN : SessionNums D DO) (hε : TableErrNonneg D) (hRd : SessionRounds D I DO) (hS : SessionAccepts D)
    (hR : SessionRuns H D I DO) (decO : List (ℕ × Col) → List T) (hdec : decOK DO decO) (x : Msg)
    (s : Fin (D.nS x)) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) :
    Pr (fun p : (Fin (D.V.k x) → Coin) × (EOs D I DO decO s).C x =>
      (run H P D.V x ω (fork (offO H D I DO P x ω s r) r p.1)).out = .acc ∧
        ¬ (sessR DO).ext x ((EOs D I DO decO s).eval H P x ω (fork (offO H D I DO P x ω s r) r p.1) p.2).1) ≤
      (D.nums x s).κ + Real.sqrt ((D.nums x s).γ * sessPrB H D I DO P x ω s r) := by
  classical
  obtain ⟨hk', hW, hK, hN₀, hεt, hwt⟩ := hN x s ((run H P D.V x ω r).msgs.take (I.V.k (xI D x) + D.kS x))
  change DO.V.k (sessStmt H D I P x ω s r) = _ at hk'
  change DO.waits (sessStmt H D I P x ω s r) = _ at hW
  change DO.cap (sessStmt H D I P x ω s r) = _ at hK
  change DO.N₀ (sessStmt H D I P x ω s r) = _ at hN₀
  change DO.εtable (sessStmt H D I P x ω s r) = _ at hεt
  change DO.weight (sessStmt H D I P x ω s r) = _ at hwt
  have hkk : DO.k (sessStmt H D I P x ω s r) = (D.nums x s).k := by
    unfold Flock.Deploy.k Nums.k
    rw [hN₀]
  have hle : offO H D I DO P x ω s r + DO.V.k (sessStmt H D I P x ω s r) ≤ D.V.k x := hRd x s _
  unfold sessPrB
  split_ifs with hA
  · have hφ : UnifMap (fun p : (Fin (D.V.k x) → Coin) × (EOs D I DO decO s).C x =>
        ((), sessCoins (offO H D I DO P x ω s r) p.1 (DO.V.k (sessStmt H D I P x ω s r)),
          castCoins hW hK (offO H D I DO P x ω s r) (DO.V.k (sessStmt H D I P x ω s r)) p.2)) :=
      UnifMap.comp ((unifMap_sessCoins _ _ hle).prod (unifMap_castCoins hW hK _ _ hle)) UnifMap.unit
    have hκ : Flock.κ DO (sessStmt H D I P x ω s r) = (D.nums x s).κ := by
      unfold Flock.κ Nums.κ
      rw [hεt, hW, hK, hkk]
    have hγ : Flock.γ DO (sessStmt H D I P x ω s r) = (D.nums x s).γ := by
      unfold Flock.γ Nums.γ
      rw [hk', hwt]
    calc _ ≤ _ := Pr_mono fun p hp => bad_fork hS hR decO x s P ω r p.1 p.2 hW hK hkk hp
      _ = PrBad H DO.V (Flock.R DO.parse) (Flock.E DO decO) (sessStmt H D I P x ω s r)
            (sessP H D I DO P x ω s r) :=
          hφ.Pr fun t : Unit × (Fin (DO.V.k (sessStmt H D I P x ω s r)) → Coin) ×
              (Flock.E DO decO).C (sessStmt H D I P x ω s r) =>
            (run H (sessP H D I DO P x ω s r) DO.V (sessStmt H D I P x ω s r) t.1 t.2.1).out = .acc ∧
              ¬ (Flock.R DO.parse).ext (sessStmt H D I P x ω s r)
                ((Flock.E DO decO).eval H (sessP H D I DO P x ω s r) (sessStmt H D I P x ω s r) t.1 t.2.1
                  t.2.2).1
      _ ≤ _ := hF decO hdec _ Unit _
      _ = _ := by rw [hκ, hγ]
  · rw [mul_zero, Real.sqrt_zero, add_zero]
    calc _ ≤ Pr (fun _ : (Fin (D.V.k x) → Coin) × (EOs D I DO decO s).C x => False) :=
          Pr_mono fun p hp => hA ⟨p.1, hp.1⟩
      _ = 0 := Pr_false
      _ ≤ _ := (D.nums x s).κ_nonneg (hε x s)

/-- **Every prefix, averaged**: a draw is a fork of a draw at session `s`'s first round (`Ex_fork`), so session `s`'s
failure is the average over prefixes of `prefix_bad_le`'s. -/
theorem PrBad_EOs_le {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T} (hF : Flock.KS H DO)
    (hN : SessionNums D DO) (hε : TableErrNonneg D) (hRd : SessionRounds D I DO) (hS : SessionAccepts D)
    (hR : SessionRuns H D I DO) (decO : List (ℕ × Col) → List T) (hdec : decOK DO decO) (x : Msg)
    (s : Fin (D.nS x)) {Ω : Type} [Fintype Ω] (P : Prover Ω) :
    PrBad H D.V (sessR DO) (EOs D I DO decO s) x P ≤
      Ex (fun p : Ω × (Fin (D.V.k x) → Coin) =>
        (D.nums x s).κ + Real.sqrt ((D.nums x s).γ * sessPrB H D I DO P x p.1 s p.2)) := by
  unfold PrBad
  rw [Pr_prod]
  conv_rhs => rw [Ex_prod]
  refine Ex_mono fun ω => ?_
  rw [Pr_prod, ← Ex_fork (offO_prefixDet H D I DO P x ω s), Ex_prod]
  refine Ex_mono fun r => le_trans (le_of_eq ?_) (prefix_bad_le hF hN hε hRd hS hR decO hdec x s P ω r)
  rw [Pr_prod]

/-! ## The collision term: the session's forks are the compiled verifier's -/

/-- Two runs of `P` at the tape `ω`, the second forked from the first at round `j`, collide among the verifier's
SHA-512 calls. -/
def vForkColl {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) (ω : Ω)
    (j : ℕ) (R R' : Fin (V.k x) → Coin) : Prop :=
  (findColl ((run H P V x ω R).vLog ++ (run H P V x ω (fork j R R')).vLog)).isSome

/-- The collision probability of `B`'s fork at round `j`, for any `j`. -/
noncomputable def forkPrN {Ω Coin : Type} [Fintype Ω] [Fintype Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) (P : Prover Ω) (x : Msg) (j : ℕ) : ℝ :=
  Pr fun t : Ω × (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => vForkColl H V P x t.1 j t.2.1 t.2.2

/-- **`k(x)·Pr[B]` is the sum of its forks' collision probabilities.** -/
theorem PrB_mul_eq {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) :
    PrB H V P x * V.k x = ∑ j ∈ Finset.range (V.k x), forkPrN H V P x j := by
  by_cases hk : 0 < V.k x
  · rw [PrB_eq_avg H V P x hk, div_mul_cancel₀ _ (by exact_mod_cast hk.ne'), ← Fin.sum_univ_eq_sum_range]
    rfl
  · simp [PrB, show V.k x = 0 by omega]

/-- A collision among calls that are all in a list of evaluations of `H` is one `findColl` finds there. -/
theorem findColl_isSome_of_sub (H : Msg → Msg) {l l' : List (Msg × Msg)} (hl' : ∀ p ∈ l', H p.1 = p.2)
    (hsub : ∀ p ∈ l, p ∈ l') (h : (findColl l).isSome) : (findColl l').isSome := by
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 h
  have hc := findColl_spec H l (fun p hp => hl' p (hsub p hp)) u v huv
  obtain ⟨hu, hv⟩ := findColl_mem l u v huv
  have hm : ∀ w ∈ l.map Prod.fst, w ∈ l'.map Prod.fst := fun w hw => by
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hw
    exact List.mem_map.2 ⟨p, hsub p hp, rfl⟩
  exact findColl_isSome_mem H l' hl' u v (hm u hu) (hm v hv) hc

theorem sum_range_shift_le (f : ℕ → ℝ) (hf : ∀ j, 0 ≤ f j) {o m k : ℕ} (h : o + m ≤ k) :
    ∑ j ∈ Finset.range m, f (o + j) ≤ ∑ j ∈ Finset.range k, f j := by
  have he := Finset.sum_Ico_eq_sum_range f o (o + m)
  rw [Nat.add_sub_cancel_left] at he
  rw [← he]
  refine Finset.sum_le_sum_of_subset_of_nonneg (fun j hj => ?_) fun j _ _ => hf j
  simp only [Finset.mem_Ico, Finset.mem_range] at hj ⊢
  omega

/-- **The session's forks are the compiled verifier's**: at a prefix some continuation of which accepts, the outer
Flock's fork at round `j` of the session prover collides only where the compiled verifier's fork at round `off + j`
does, on the same coins from the session's first round on. -/
theorem sessPrB_mul_le {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hN : SessionNums D DO) (hRd : SessionRounds D I DO) (hR : SessionRuns H D I DO) (x : Msg)
    (s : Fin (D.nS x)) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) :
    sessPrB H D I DO P x ω s r * (D.nums x s).rounds ≤
      ∑ j ∈ Finset.range (D.nums x s).rounds, Pr fun p : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) =>
        vForkColl H D.V P x ω (offO H D I DO P x ω s r + j) (fork (offO H D I DO P x ω s r) r p.1) p.2 := by
  unfold sessPrB
  split_ifs with hA
  · obtain ⟨d₁, hacc⟩ := hA
    have hk' := (hN x s ((run H P D.V x ω r).msgs.take (I.V.k (xI D x) + D.kS x))).1
    change DO.V.k (sessStmt H D I P x ω s r) = _ at hk'
    have hle : offO H D I DO P x ω s r + DO.V.k (sessStmt H D I P x ω s r) ≤ D.V.k x := hRd x s _
    rw [← hk', PrB_mul_eq]
    refine Finset.sum_le_sum fun j _ => ?_
    have hφ : UnifMap (fun p : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) =>
        ((), sessCoins (offO H D I DO P x ω s r) p.1 (DO.V.k (sessStmt H D I P x ω s r)),
          sessCoins (offO H D I DO P x ω s r) p.2 (DO.V.k (sessStmt H D I P x ω s r)))) :=
      UnifMap.comp ((unifMap_sessCoins _ _ hle).prod (unifMap_sessCoins _ _ hle)) UnifMap.unit
    unfold forkPrN
    rw [← hφ.Pr fun t => vForkColl H DO.V (sessP H D I DO P x ω s r) (sessStmt H D I P x ω s r) t.1 j t.2.1 t.2.2]
    refine Pr_mono fun p hp => ?_
    dsimp only [vForkColl] at hp ⊢
    rw [fork_sessCoins] at hp
    rw [← fork_add_fork]
    refine findColl_isSome_of_sub H (fun e he => ?_) (fun e he => ?_) hp
    · rcases List.mem_append.1 he with he | he
      · exact (run_logs H P D.V x ω _).2 e he
      · exact (run_logs H P D.V x ω _).2 e he
    · rcases List.mem_append.1 he with he | he
      · exact List.mem_append_left _ ((sessionRuns_fork hR x s P ω r d₁ p.1 hacc).2 e he)
      · exact List.mem_append_right _ ((sessionRuns_fork hR x s P ω r d₁ _ hacc).2 e he)
  · rw [zero_mul]
    exact Finset.sum_nonneg fun j _ => Pr_nonneg _

/-- **The collision term, averaged**: `𝔼[Pr[B] at the session prover]·k_s ≤ k(x)·Pr[B]`. A draw's prefix is a fork of
a draw (`Ex_fork`), so the session's forks at rounds `off + j`, `j < k_s`, are among the compiled verifier's `k(x)`
round guesses (`SessionRounds`). -/
theorem Ex_sessPrB_le {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hN : SessionNums D DO) (hRd : SessionRounds D I DO) (hR : SessionRuns H D I DO) (x : Msg)
    (s : Fin (D.nS x)) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Ex (fun p : Ω × (Fin (D.V.k x) → Coin) => sessPrB H D I DO P x p.1 s p.2) * (D.nums x s).rounds ≤
      PrB H D.V P x * D.V.k x := by
  rw [← Ex_mul_right, PrB_mul_eq, Ex_prod]
  have hω : ∀ ω : Ω, Ex (fun r : Fin (D.V.k x) → Coin => sessPrB H D I DO P x ω s r * (D.nums x s).rounds) ≤
      Ex fun R : Fin (D.V.k x) → Coin => ∑ j ∈ Finset.range (D.V.k x),
        Pr fun d' : Fin (D.V.k x) → Coin => vForkColl H D.V P x ω j R d' := by
    intro ω
    let G : (Fin (D.V.k x) → Coin) → ℝ := fun R => ∑ j ∈ Finset.range (D.nums x s).rounds,
      Pr fun d' : Fin (D.V.k x) → Coin => vForkColl H D.V P x ω (offO H D I DO P x ω s R + j) R d'
    calc Ex (fun r : Fin (D.V.k x) → Coin => sessPrB H D I DO P x ω s r * (D.nums x s).rounds)
        ≤ Ex fun r : Fin (D.V.k x) → Coin => ∑ j ∈ Finset.range (D.nums x s).rounds,
            Pr fun p : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) =>
              vForkColl H D.V P x ω (offO H D I DO P x ω s r + j) (fork (offO H D I DO P x ω s r) r p.1) p.2 :=
          Ex_mono fun r => sessPrB_mul_le hN hRd hR x s P ω r
      _ = Ex fun p : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) =>
            G (fork (offO H D I DO P x ω s p.1) p.1 p.2) := by
          rw [Ex_prod]
          congr 1
          funext r
          simp only [G, offO_fork]
          rw [Ex_sum]
          exact Finset.sum_congr rfl fun j _ => Pr_prod _
      _ = Ex G := Ex_fork (offO_prefixDet H D I DO P x ω s) G
      _ ≤ _ := Ex_mono fun R => ?_
    have hk' := (hN x s ((run H P D.V x ω R).msgs.take (I.V.k (xI D x) + D.kS x))).1
    have hle : offO H D I DO P x ω s R + DO.V.k (sessStmt H D I P x ω s R) ≤ D.V.k x := hRd x s _
    change DO.V.k (sessStmt H D I P x ω s R) = _ at hk'
    rw [hk'] at hle
    exact sum_range_shift_le (fun j => Pr fun d' : Fin (D.V.k x) → Coin => vForkColl H D.V P x ω j R d')
      (fun j => Pr_nonneg _) hle
  calc _ ≤ Ex fun ω : Ω => Ex fun R : Fin (D.V.k x) → Coin => ∑ j ∈ Finset.range (D.V.k x),
          Pr fun d' : Fin (D.V.k x) → Coin => vForkColl H D.V P x ω j R d' := Ex_mono fun ω => hω ω
    _ = ∑ j ∈ Finset.range (D.V.k x), forkPrN H D.V P x j := by
      simp only [Ex_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      unfold forkPrN
      rw [Pr_prod]
      congr 1
      funext ω
      rw [Pr_prod]

/-! ## The outer Flock's KS at each session, from Flock's own -/

/-- **`OuterKS`'s inequality with the exact collision term**: session `s`'s extraction fails with probability at most
`κ_s + √(γ_s·(k(x)/k_s)·Pr[B])`, `k_s` the session's rounds. Each prefix's failure is the outer Flock's at the session
prover (`prefix_bad_le`); Jensen moves the average under the root, and the session's `k_s` round guesses are `k_s` of
the compiled verifier's `k(x)` (`Ex_sessPrB_le`). -/
theorem outerKS_sharp {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T} (hF : Flock.KS H DO)
    (hN : SessionNums D DO) (hε : TableErrNonneg D) (hRd : SessionRounds D I DO) (hS : SessionAccepts D)
    (hR : SessionRuns H D I DO) (decO : List (ℕ × Col) → List T) (hdec : decOK DO decO) (x : Msg) (Ω : Type)
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (s : Fin (D.nS x)) :
    PrBad H D.V (sessR DO) (EOs D I DO decO s) x P ≤
      (D.nums x s).κ + Real.sqrt ((D.nums x s).γ * (D.V.k x / (D.nums x s).rounds) * PrB H D.V P x) := by
  have hZ0 : ∀ p : Ω × (Fin (D.V.k x) → Coin), 0 ≤ sessPrB H D I DO P x p.1 s p.2 := fun p =>
    sessPrB_nonneg H D I DO P x p.1 s p.2
  have hγ := (D.nums x s).γ_nonneg
  have hZ : Ex (fun p : Ω × (Fin (D.V.k x) → Coin) => sessPrB H D I DO P x p.1 s p.2) ≤
      D.V.k x / (D.nums x s).rounds * PrB H D.V P x := by
    rcases Nat.eq_zero_or_pos (D.nums x s).rounds with h0 | hpos
    · have hz : ∀ p : Ω × (Fin (D.V.k x) → Coin), sessPrB H D I DO P x p.1 s p.2 = 0 := by
        intro p
        unfold sessPrB
        split_ifs
        · have hk' := (hN x s ((run H P D.V x p.1 p.2).msgs.take (I.V.k (xI D x) + D.kS x))).1
          change DO.V.k (sessStmt H D I P x p.1 s p.2) = _ at hk'
          simp [PrB, hk', h0]
        · rfl
      simp only [hz, h0, Nat.cast_zero, div_zero, zero_mul]
      exact (Ex_const 0).le
    · rw [div_mul_eq_mul_div, le_div_iff₀ (by exact_mod_cast hpos), mul_comm (D.V.k x : ℝ)]
      exact Ex_sessPrB_le hN hRd hR x s P
  calc PrBad H D.V (sessR DO) (EOs D I DO decO s) x P
      ≤ Ex fun p : Ω × (Fin (D.V.k x) → Coin) =>
          (D.nums x s).κ + Real.sqrt ((D.nums x s).γ * sessPrB H D I DO P x p.1 s p.2) :=
        PrBad_EOs_le hF hN hε hRd hS hR decO hdec x s P
    _ = (D.nums x s).κ + Ex fun p : Ω × (Fin (D.V.k x) → Coin) =>
          Real.sqrt ((D.nums x s).γ * sessPrB H D I DO P x p.1 s p.2) := by
        rw [Ex_add, Ex_const]
    _ ≤ (D.nums x s).κ + Real.sqrt (Ex fun p : Ω × (Fin (D.V.k x) → Coin) =>
          (D.nums x s).γ * sessPrB H D I DO P x p.1 s p.2) :=
        add_le_add le_rfl (Ex_sqrt_le _ fun p => mul_nonneg hγ (hZ0 p))
    _ ≤ _ := by
        rw [Ex_mul_left, mul_assoc]
        exact add_le_add le_rfl (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hZ hγ))

/-- **The outer `--zk` Flock's KS at each session, from Flock's own KS** (`Claims.Flock.KS` at `DO`) and the code
facts that say how the compiled verifier runs each session: the session's numbers are the outer Flock's at its V*
statement (`SessionNums`), its rounds end by `k(x)` (`SessionRounds`), its table error is nonnegative
(`TableErrNonneg`), an accepted run shows each session accepting (`SessionAccepts`), and each fork at the session's
first round shows the outer verifier's run of the session prover, whose calls are the compiled verifier's
(`SessionRuns`). `outerKS_sharp`'s `k(x)/k_s` is at most `OuterKS`'s `k(x)`. -/
theorem outerKS_of_flockKS {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hF : Flock.KS H DO) (hN : SessionNums D DO) (hε : TableErrNonneg D) (hRd : SessionRounds D I DO)
    (hS : SessionAccepts D) (hR : SessionRuns H D I DO) : OuterKS H D I DO := by
  intro decO hdec x Ω _ _ P s
  refine (outerKS_sharp hF hN hε hRd hS hR decO hdec x Ω P s).trans (add_le_add le_rfl (Real.sqrt_le_sqrt ?_))
  have hm : (D.V.k x : ℝ) / (D.nums x s).rounds ≤ D.V.k x := by
    rcases Nat.eq_zero_or_pos (D.nums x s).rounds with h0 | hpos
    · simp [h0]
    · exact div_le_self (Nat.cast_nonneg _) (by exact_mod_cast hpos)
  exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hm (D.nums x s).γ_nonneg) (PrB_nonneg H D.V P x)

end Claims.Recursion
