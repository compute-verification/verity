import Proofs.Flock.Claims.ZK.Stop
import Proofs.Flock.Claims.Recursion.SeqPrefix

/-!
# The recursion's zero knowledge with the prover's public stop, by sequential composition

`zkAt` (`Recursion.Seq`) with a public stop `s` read at the first part's rounds (`Claims.ZKMal.zkmalAt_layout_stop`):
the masked reference prover with the stop (`stopP (refPM …) s (kPre IV)`) is `ZKmalAt` at `simMStop`, the first part's
simulator cut where the stop fires (`censorSim`), then each V\* session's outer simulator where the stop hasn't fired
and the silent one where it has (`stopProgB`). A session needs zero knowledge, and its witness the sessions' relation,
only at a history where the stop hasn't fired, and there only at a live first part (`zkAtStop`).

* `iterSimG_stopped_eval`: `simMStop`'s nested simulator's transcript as a fold of one step per session (`stepS`).
* `simPre_pm`: the claim's first-part simulator `simPre` at `deployRec` is a run of a message function (`pmZ`: each
  inner round's and the staging round's commit strings to the zero digest), so cut where the stop fires it is that
  function with the stop (`Claims.ZKMal.interact_censor`).
* `Claims.ZKMal.foldl_silent`: silent sessions in turn are as many silent rounds.
-/

namespace Claims.ZKMal

open Claims Claims.ZKServer

theorem finRange_one : List.finRange 1 = [(⟨0, Nat.zero_lt_one⟩ : Fin 1)] := by
  first | rfl | decide | simp [List.finRange_succ_last]

theorem recGoesOn_append (τ r : List (Msg × Out)) (hr : r ≠ []) :
    Recursion.goesOn (τ ++ r) = Recursion.goesOn r := by
  cases hl : r.getLast? with
  | none => exact absurd (List.getLast?_eq_none_iff.1 hl) hr
  | some a => simp only [Recursion.goesOn, List.getLast?_append, hl, Option.some_or]

/-- Past its first `k` rounds, a history's stop is its first `k` rounds'. -/
theorem fired_append (s : List (Msg × Msg) → Bool) (k : ℕ) (h₀ h : List (Msg × Msg)) (hl : h₀.length = k) :
    fired s k (h₀ ++ h) = fired s k h₀ := by
  rw [← fired_take (s := s) (h₀ ++ h) (le_refl k), List.take_left' hl]

/-- **The silent session under a transcript** is the empty message every round after it. -/
theorem silent_eval (H : Msg → Msg) (ask : List Msg → Out) (τ : List (Msg × Out)) (n : ℕ) :
    (SP.eval H ask (spUnder (τ.map Prod.fst) (silentSP n))).1 = interact ask (fun _ => []) n (histOf τ) := by
  have e := interact_shift ask (fun _ => []) (histOf τ) n []
  rw [List.append_nil, histOf_map_fst] at e
  rw [e, spUnder_eval, silentSP, spRun_eval] <;> rfl

/-- **Silent sessions in turn**, after a transcript of messages, are as many rounds of the empty message. -/
theorem foldl_silent {ι : Type} (ask : List Msg → Out) (ns : ι → ℕ) :
    ∀ (l : List ι) (τ : List (Msg × Out)), (∀ p ∈ τ, isMsg p.2 = true) →
      l.foldl (fun τ i => if Recursion.goesOn τ then τ ++ interact ask (fun _ => []) (ns i) (histOf τ) else τ) τ =
        τ ++ interact ask (fun _ => []) (l.map ns).sum (histOf τ)
  | [], τ, _ => by simp [interact]
  | i :: l, τ, hm => by
      rw [List.foldl_cons, if_pos (_root_.RecDeploy.recGoesOn_of_msgs hm), List.map_cons, List.sum_cons,
        interact_add ask (fun _ => []) (ns i) (l.map ns).sum (histOf τ)]
      unfold cont
      split
      · rename_i hg
        have hm' : ∀ p ∈ τ ++ interact ask (fun _ => []) (ns i) (histOf τ), isMsg p.2 = true := by
          intro p hp
          rcases List.mem_append.1 hp with hp | hp
          · exact hm p hp
          · simp only [goesOn, Bool.and_eq_true, List.all_eq_true] at hg
            exact hg.2 p hp
        rw [foldl_silent ask ns l _ hm', histOf_append, List.append_assoc]
      · rename_i hg
        have hpos : 0 < ns i := by
          rcases Nat.eq_zero_or_pos (ns i) with h0 | h0
          · exact absurd (by simp [h0, interact, goesOn]) hg
          · exact h0
        refine foldl_stuck _ (fun τ' _ h => by simp [h]) l _ ?_
        rw [recGoesOn_append _ _ (interact_ne_nil ask (fun _ => []) (ns i) (histOf τ) hpos),
          recGoesOn_interact ask (fun _ => []) (ns i) (histOf τ) hpos]
        exact Bool.eq_false_iff.2 hg

theorem sum_finRange_map (g : ℕ → ℕ) :
    ∀ n, ((List.finRange n).map fun s : Fin n => g s).sum = ((List.range n).map g).sum
  | 0 => by simp
  | n + 1 => by
      rw [List.finRange_succ_last, List.map_append, List.map_map, List.sum_append, List.range_succ, List.map_append,
        List.sum_append, ← sum_finRange_map g n] <;> first | rfl | simp

/-- `censorSim`'s transcript against `ask`: `S`'s, cut where the stop fires. -/
theorem censorSim_eval (H : Msg → Msg) (ask : List Msg → Out) (S : Simulator) (s : Msg → List (Msg × Msg) → Bool)
    (k : Msg → ℕ) (x : Msg) (c : S.C x) :
    (SP.eval H ask ((censorSim S s k).prog x c)).1 =
      cutT ask [] (stopIdx (s x) (k x) (k x) (SP.eval H ask (S.prog x c)).1) (k x) (SP.eval H ask (S.prog x c)).1 := by
  simp only [censorSim, spBind_eval, spRun_eval]
  rfl

end Claims.ZKMal

namespace RecDeploy

open _root_.Flock
open FlockDeploy (Coin)
open Claims.ZKMal (RunParts ZKmalAt SeqTape iterSimG)

variable (IV : Claims.Verifier Coin) (b : Brings)

/-! ## The first part as a message function -/

/-- Two `simRounds` from the empty transcript, `n` rounds then one, are a run of `pm`: where `f` and `g` read as
`pm` before and at round `n`. -/
theorem simRounds_pre_eval (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out)
    (f g : List (Claims.Msg × Claims.Out) → Claims.SP Claims.Msg) (n k : ℕ)
    (pm : List (Claims.Msg × Claims.Msg) → Claims.Msg) (hk : k = 1)
    (hf : ∀ h : List (Claims.Msg × Claims.Msg), h.length < n → (Claims.SP.eval H ask (f (asRounds h))).1 = pm h)
    (hg : ∀ h : List (Claims.Msg × Claims.Msg), h.length = n → (Claims.SP.eval H ask (g (asRounds h))).1 = pm h) :
    (Claims.SP.eval H ask (Claims.Recursion.simRounds g k
      (Claims.SP.eval H ask (Claims.Recursion.simRounds f n [])).1)).1 = Claims.ZKMal.interact ask pm (n + k) [] := by
  subst hk
  rw [simRounds_interact H ask f n [] (by simp), List.nil_append,
    show Claims.ZKMal.histOf ([] : List (Claims.Msg × Claims.Out)) = [] from rfl,
    Claims.ZKMal.interact_congr_lt ask _ pm n hf n [] (by simp), Claims.ZKMal.interact_add ask pm n 1 []]
  unfold Claims.ZKMal.cont
  split
  · rename_i hgo
    have hl := Claims.ZKMal.goesOn_length hgo
    have hm : ∀ p ∈ Claims.ZKMal.interact ask pm n [], Claims.ZKMal.isMsg p.2 = true := by
      have h' := hgo
      simp only [Claims.ZKMal.goesOn, Bool.and_eq_true, List.all_eq_true] at h'
      exact h'.2
    rw [simRounds_interact H ask g 1 _ hm, List.nil_append]
    congr 1
    have e := hg (Claims.ZKMal.histOf (Claims.ZKMal.interact ask pm n [])) (by rw [Claims.ZKMal.histOf_length, hl])
    rw [Claims.ZKMal.interact_one ask (fun h => (Claims.SP.eval H ask (g (asRounds h))).1)
      (Claims.ZKMal.histOf (Claims.ZKMal.interact ask pm n [])), e]
    exact (Claims.ZKMal.interact_one ask pm _).symm
  · rename_i hgo
    have hs : Claims.Recursion.goesOn (Claims.ZKMal.interact ask pm n []) = false := by
      rcases Nat.eq_zero_or_pos n with h0 | h0
      · subst h0
        exact absurd (by simp [Claims.ZKMal.interact, Claims.ZKMal.goesOn]) hgo
      · exact (Claims.ZKMal.recGoesOn_interact ask pm n [] h0).trans (Bool.eq_false_iff.2 hgo)
    rw [simRounds_stop g 1 _ hs] <;> rfl

theorem simRounds_bind_eval (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out)
    (f g : List (Claims.Msg × Claims.Out) → Claims.SP Claims.Msg) (n k : ℕ)
    (pm : List (Claims.Msg × Claims.Msg) → Claims.Msg) (hk : k = 1)
    (hf : ∀ h : List (Claims.Msg × Claims.Msg), h.length < n → (Claims.SP.eval H ask (f (asRounds h))).1 = pm h)
    (hg : ∀ h : List (Claims.Msg × Claims.Msg), h.length = n → (Claims.SP.eval H ask (g (asRounds h))).1 = pm h) :
    (Claims.SP.eval H ask (Claims.Recursion.spBind (Claims.Recursion.simRounds f n []) fun ms =>
      Claims.Recursion.simRounds g k ms)).1 = Claims.ZKMal.interact ask pm (n + k) [] := by
  rw [Claims.ZKMal.recSpBind_eq, Claims.ZKMal.spBind_eval]
  exact simRounds_pre_eval H ask f g n k pm hk hf hg

theorem simRounds_tail_eval (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out)
    (f g : List (Claims.Msg × Claims.Out) → Claims.SP Claims.Msg)
    (tail : List (Claims.Msg × Claims.Out) → Claims.SP (List (Claims.Msg × Claims.Out))) (n k : ℕ)
    (pm : List (Claims.Msg × Claims.Msg) → Claims.Msg) (hk : k = 1)
    (hf : ∀ h : List (Claims.Msg × Claims.Msg), h.length < n → (Claims.SP.eval H ask (f (asRounds h))).1 = pm h)
    (hg : ∀ h : List (Claims.Msg × Claims.Msg), h.length = n → (Claims.SP.eval H ask (g (asRounds h))).1 = pm h) :
    (Claims.SP.eval H ask (Claims.Recursion.spBind (Claims.Recursion.simRounds f n []) fun ms =>
      Claims.Recursion.spBind (Claims.Recursion.simRounds g k ms) tail)).1 =
        (Claims.SP.eval H ask (tail (Claims.ZKMal.interact ask pm (n + k) []))).1 := by
  simp only [Claims.ZKMal.recSpBind_eq, Claims.ZKMal.spBind_eval]
  rw [simRounds_pre_eval H ask f g n k pm hk hf hg]

/-- **The first part's simulated messages**: the first `n` rounds each the commit string to the zero digest under its
inner salt (`yI`), then the staging message at the commit strings to the zero digest under the staging salts (`yS`). -/
noncomputable def pmZ (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (n : ℕ) (yI : Fin n → Salt)
    (yS : Fin (b.nC x) → Salt) (h : List (Claims.Msg × Claims.Msg)) : Claims.Msg :=
  if hh : (asRounds h).length < n then List.ofFn (maskOf (yI ⟨(asRounds h).length, hh⟩)) ++
      saltH H (yI ⟨(asRounds h).length, hh⟩)
  else stageMsg b.hs x (asRounds h) ((List.finRange (b.nC x)).map fun j => List.ofFn (maskOf (yS j)) ++ saltH H (yS j))

/-- **The claim's first-part simulator is a run of `pmZ`**, where the inner protocol's rounds at the inner statement
are the inner verifier's. -/
theorem simPre_pm {W : Type} (ivScope : Claims.Msg → Bool) (I : Claims.Recursion.Inner Coin W)
    (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out) (x : Claims.Msg)
    (hI : I.V.k (xIOf x) = IV.k (xIOf x)) (c : (simPre (deployRec IV ivScope b) I).C x) :
    (Claims.SP.eval H ask ((simPre (deployRec IV ivScope b) I).prog x c)).1 =
      Claims.ZKMal.interact ask (pmZ b H x (I.V.k (xIOf x)) c.1 (c.2 ⟨0, Nat.zero_lt_one⟩)) (kPre IV x) [] := by
  have hK : I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x) + (deployRec IV ivScope b).kS x = kPre IV x := by
    show I.V.k (xIOf x) + kS x = IV.k (xIOf x) + kS x
    rw [hI]
  rw [← hK]
  refine simRounds_bind_eval H ask _ _ (I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x))
    ((deployRec IV ivScope b).kS x) _ rfl ?hf ?hg
  case hf =>
    intro h hh
    try beta_reduce
    have hl : (asRounds h).length = h.length := List.length_map _
    have h₁ : (asRounds h).length < I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x) := by
      rw [hl]; try exact hh
    rw [dif_pos h₁]
    refine (zeroCommit_eval IV b ivScope H ask _).trans ?_
    unfold pmZ
    rw [dif_pos (show (asRounds h).length < I.V.k (xIOf x) from h₁)]
    try rfl
  case hg =>
    intro h hh
    try beta_reduce
    have hl : (asRounds h).length = h.length := List.length_map _
    have hτ : (asRounds h).length - I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x) = 0 := by
      rw [hl, hh]; try exact Nat.sub_self _
    have h0 : (asRounds h).length - I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x) <
        (deployRec IV ivScope b).kS x := lt_of_eq_of_lt hτ Nat.zero_lt_one
    have hs : (⟨_, h0⟩ : Fin ((deployRec IV ivScope b).kS x)) = ⟨0, Nat.zero_lt_one⟩ := Fin.ext hτ
    have hn : ¬ (asRounds h).length < I.V.k (xIOf x) := by
      rw [hl, hh]; try exact lt_irrefl _
    rw [dif_pos h0, hs]
    simp only [Claims.ZKMal.recSpBind_eq, Claims.ZKMal.spBind_eval, zeroCommits_eval, stage_eval]
    unfold pmZ
    rw [dif_neg hn]
    first
      | exact congrArg (stageMsg b.hs x (asRounds h)) (List.map_map ..)
      | (simp only [List.map_map]; try rfl)
      | rfl

/-! ## The composed simulator with the stop -/

/-- **The composed simulator with the stop `s`** of the masked run: the first part's simulator `S₀` cut where the stop
fires within the first part (`censorSim`), then, while the claim's test (`Recursion.goesOn`) holds, session `j`'s outer
simulator `SO x j` at the V\* statement its simulated history stages where the stop hasn't fired, and the silent one
where it has (`stopProgB`), on coins of its own. -/
noncomputable def simMStop (T₀ : SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (S₀ : Claims.Simulator) (SO : Claims.Msg → ℕ → Claims.Recursion.OSim)
    (s : Claims.Msg → List (Claims.Msg × Claims.Msg) → Bool) : Claims.Simulator where
  C x := (iterSimG T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB ((parts IV b (T₀.N × T₀.Ω)).stopped s))
    (Claims.ZKMal.censorSim S₀ s (kPre IV)) (fun j x => (SO x j).C)
    (Claims.ZKMal.stopProgB s (kPre IV) (fun x j => (nums b.st x j).rounds + 1)
      fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn) (nS x)).C x
  prog x := (iterSimG T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB ((parts IV b (T₀.N × T₀.Ω)).stopped s))
    (Claims.ZKMal.censorSim S₀ s (kPre IV)) (fun j x => (SO x j).C)
    (Claims.ZKMal.stopProgB s (kPre IV) (fun x j => (nums b.st x j).rounds + 1)
      fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn) (nS x)).prog x

/-- **The recursion's zero knowledge at one statement with a public stop, by sequential composition**: `zkAt` with the
stop `s` read at the first part's rounds, where each session is `ZKmalAt` at its outer simulator at every history that
starts it and where the stop hasn't fired (`hS`), and the witness `wit` gives is in the sessions' relation there at
every live first part (`hW`: its messages the first part's prover's on the tape `t₀`). The masked reference prover
with the stop is `ZKmalAt` at `simMStop` with `ε₀ + Σ_{j < nS x} eB j` and `nS x · g`, at the masked run's rounds. -/
theorem zkAtStop (H : Claims.Msg → Claims.Msg) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (T₀ : SeqTape) (pre : T₀.N × T₀.Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (RA RF : Claims.Rel Claims.Flock.Wit) (S₀ : Claims.Simulator)
    (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) (s : Claims.Msg → List (Claims.Msg × Claims.Msg) → Bool)
    (x : Claims.Msg) (ε₀ g : ℝ) (eB : ℕ → ℝ)
    (h₀ : ZKmalAt H (kPre IV x) (preP IV b pre) (RA x) x (S₀.C x) (S₀.prog x) ε₀ 0)
    (hS : ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg), u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
      Claims.ZKMal.fired (s x) (kPre IV x) u = false →
      ZKmalAt H ((nums b.st x j).rounds + 1) PS (RF (xsOf IV b x j u)) (xsOf IV b x j u) (SO x j).C
        ((SO x j).prog (xsOf IV b x j u)) (eB j) g)
    (hW : ∀ j < nS x, ∀ w, RA x w → ∀ (t₀ : T₀.N × T₀.Ω) (u : List (Claims.Msg × Claims.Msg)),
      u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
      (∀ i (hi : i < u.length), i < kPre IV x → u[i].1 = (((preP IV b pre w).next x (u.take i) t₀).eval H).1) →
      Claims.ZKMal.fired (s x) (kPre IV x) u = false →
        RF (xsOf IV b x j u) (wit w (pre t₀) (u.take (IV.k (xIOf x))) j))
    (he0 : ∀ j < nS x, 0 ≤ eB j) (hg : 0 ≤ g) :
    ZKmalAt H (kM IV b x)
      (fun w => (Claims.ZKMal.stopP (refPM IV b pre wit PS w) s (kPre IV)).comap (RunParts.flatten T₀ 0 NB ΩB (nS x)))
      (RA x) x ((simMStop IV b T₀ NB ΩB S₀ SO s).C x) ((simMStop IV b T₀ NB ΩB S₀ SO s).prog x)
      (ε₀ + ∑ j ∈ Finset.range (nS x), eB j) (nS x * g) := by
  have hst := lay_start IV b pre wit
  have h := Claims.ZKMal.zkmalAt_layout_stop T₀ (lay IV b pre wit) s H x (nS x) (preP IV b pre) PS RA RF S₀
    (fun j x => (SO x j).C) (fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn)
    ε₀ g eB h₀ (fun j hj u hu hf => hS j hj u (by rwa [hst] at hu) hf)
    (fun j hj w hw t₀ u hu hpre hf => hW j hj w hw t₀ u (by rwa [hst] at hu) hpre hf)
    (fun j _ ask pm => Claims.ZKMal.recGoesOn_interact ask pm _ [] (by
      show 0 < (lay IV b pre wit).start x j
      rw [hst]
      exact parts_start_pos IV b _ x j))
    he0 hg
  rw [RunParts.iterSimG_seq_congr T₀ 0 NB ΩB ((lay IV b pre wit).stopped s) ((parts IV b _).stopped s) rfl rfl rfl, hst,
    parts_start] at h
  exact h.of_equiv _ (Equiv.refl _) (Equiv.refl _) fun _ _ _ => rfl

/-- One session's step of a nested simulator: while the run goes on, under the transcript so far, the session's
program `pB` at the history so far. -/
noncomputable def stepS (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out)
    (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) (x : Claims.Msg)
    (pB : (j : ℕ) → (x : Claims.Msg) → List (Claims.Msg × Claims.Msg) → (SO x j).C →
      Claims.SP (List (Claims.Msg × Claims.Out)))
    (τ : List (Claims.Msg × Claims.Out)) (j : ℕ) (c : (SO x j).C) : List (Claims.Msg × Claims.Out) :=
  if Claims.Recursion.goesOn τ then
    τ ++ (Claims.SP.eval H ask (Claims.ZKMal.spUnder (τ.map Prod.fst) (pB j x (Claims.ZKMal.histOf τ) c))).1
  else τ

/-- **The stopped parts' nested simulator's transcript**: the first part's, then one `stepS` per session, on the
coins `iterCoins` reads off the nested ones, for any sessions' programs `pB` (`simMStop`'s are `stopProgB`'s). -/
theorem iterSimG_stopped_eval (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out) (T₀ : SeqTape)
    (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (S₀ : Claims.Simulator)
    (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) (s : Claims.Msg → List (Claims.Msg × Claims.Msg) → Bool)
    (x : Claims.Msg)
    (pB : (n : ℕ) → (x : Claims.Msg) → (RunParts.seq T₀ 0 NB ΩB ((parts IV b (T₀.N × T₀.Ω)).stopped s) n).U x →
      (SO x n).C → Claims.SP (List (Claims.Msg × Claims.Out))) :
    ∀ (m : ℕ) (c : (iterSimG T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB ((parts IV b (T₀.N × T₀.Ω)).stopped s)) S₀
        (fun j x => (SO x j).C) pB (fun _ _ => Claims.Recursion.goesOn) m).C x),
      (Claims.SP.eval H ask ((iterSimG T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB ((parts IV b (T₀.N × T₀.Ω)).stopped s)) S₀
        (fun j x => (SO x j).C) pB (fun _ _ => Claims.Recursion.goesOn) m).prog x c)).1 =
      (List.finRange m).foldl (fun τ (j : Fin m) => stepS H ask SO x pB τ j
          ((Claims.ZKMal.iterCoins T₀ 0 NB ΩB _ S₀ _ _ _ x m c).2 j))
        (Claims.SP.eval H ask (S₀.prog x (Claims.ZKMal.iterCoins T₀ 0 NB ΩB _ S₀ _ _ _ x m c).1)).1
  | 0, c => by simp only [List.finRange_zero, List.foldl_nil]; rfl
  | m + 1, (a, b') => by
      rw [Claims.ZKMal.iterCoins_succ, List.finRange_succ_last, List.foldl_append, List.foldl_map]
      simp only [Fin.snoc_castSucc, Fin.snoc_last, List.foldl_cons, List.foldl_nil]
      dsimp only [Fin.val_castSucc, Fin.val_last]
      rw [← iterSimG_stopped_eval H ask T₀ NB ΩB S₀ SO s x pB m a]
      simp only [iterSimG, Claims.ZKMal.Seq.simG, Claims.ZKMal.spBind_eval]
      unfold stepS
      split
      · simp only [Claims.ZKMal.spBind_eval, Claims.SP.eval]
        rfl
      · rfl

end RecDeploy
