import Proofs.Flock.Claims.ZK.Layout
import Proofs.Flock.Claims.Recursion.DeployZK

/-!
# The recursion's zero knowledge by sequential composition

The masked recursive run is a first part (the inner session and the staging round, `kPre x` rounds), then the `nS x`
V\* sessions back to back, each a complete run of the masked outer session protocol (its own `Hello`, coin tree and
coins: `(nums x s).rounds + 1` rounds) at the V\* statement the staged rounds give, on a tape of its own. Its zero
knowledge is the first part's and the sessions' by sequential composition (`zkmalAt_layout` at `c = 0`): the first
part costs its own `ε₀` and rewinds nothing, and the sessions cost `Σ_s eB s` and `nS x · g`.

* `preP`: the reference prover's first part (`refP` before the sessions, `refP_pre`), on a finite tape that `pre`
  reads as the inner prover's tape and the staging salts.
* `lay`, `parts`, `refPM`: the run's parts, and the masked reference prover: `preP`, then each session the given
  session prover on slot `s`'s tape at the V\* statement `xsOf` reads off the rounds before the sessions, with the
  witness `wit` reads off the inner witness, the first part's tape and the inner rounds. `refWit` is the record's
  (`RefParts.wit`), which reads no staging salt.
* `kM_eq`: the masked run's rounds are the compiled verifier's and one `Hello` per V\* session, `V.k x + nS x`.
* `Claims.ZKMal.recGoesOn_interact`: the claim's test of going on (`Recursion.goesOn`, the last answer a message) is
  `goesOn` at a run's transcript, so the composed simulator may test it as `Sim` does.
* **`zkAt`**: `ZKmalAt` at one statement, on the nested tape read flat, at the composed simulator `simM` (the first
  part's simulator, then each session's `SO x s` at its statement, going on while `Recursion.goesOn` holds), from
  the first part's `ZKmalAt` (`h₀`), the sessions' at every history that starts them (`hS`), and the witnesses in the
  sessions' relation there (`hW`). `zkmalRx`: at every statement, on statement-indexed tapes (`ZKmalRx`), with `nS x`
  sessions at `x`.
-/

namespace Claims.ZKMal

open Claims

theorem interact_ne_nil (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)), 0 < n → interact ask pm n hist ≠ []
  | 0, _, h => absurd h (lt_irrefl 0)
  | n + 1, hist, _ => by
      simp only [interact]
      split <;> simp

/-- **The claim's test of going on is `goesOn` at a run's transcript**: after `n ≥ 1` rounds, the last answer is a
message exactly when every answer is, and then the run has all `n` rounds. -/
theorem recGoesOn_interact (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)), 0 < n →
      Claims.Recursion.goesOn (interact ask pm n hist) = goesOn n (interact ask pm n hist)
  | 0, _, h => absurd h (lt_irrefl 0)
  | n + 1, hist, _ => by
      simp only [interact]
      cases ask (hist.map Prod.fst ++ [pm hist]) with
      | msg m =>
          simp only
          rw [goesOn_cons]
          rcases Nat.eq_zero_or_pos n with rfl | hn
          · simp [interact, Claims.Recursion.goesOn, goesOn]
          · rw [← recGoesOn_interact ask pm n _ hn]
            obtain ⟨p, τ, hτ⟩ := List.exists_cons_of_ne_nil (interact_ne_nil ask pm n (hist ++ [(pm hist, m)]) hn)
            rw [hτ]
            simp [Claims.Recursion.goesOn, List.getLast?_cons_cons]
      | acc => simp [Claims.Recursion.goesOn, goesOn, isMsg]
      | rej => simp [Claims.Recursion.goesOn, goesOn, isMsg]

end Claims.ZKMal

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open Claims.ZKMal (RunParts ZKmalAt SeqTape iterSimG iterProver)

variable (IV : Claims.Verifier Coin) (b : Brings)

/-- The rounds before the V\* sessions: the inner session's and the staging round. -/
def kPre (x : Claims.Msg) : ℕ := IV.k (xIOf x) + kS x

/-- **The first part's prover**: the reference prover's before the V\* sessions, on a tape `pre` reads as the inner
prover's tape and the staging salts. -/
noncomputable def preP {T : Type} (pre : T → b.ref.TI × (ℕ → Salt)) (w : Claims.Flock.Wit) : Claims.Prover T :=
  ⟨fun x hist t => if hist.length < IV.k (xIOf x) then (b.ref.PI w).next (xIOf x) hist (pre t).1 else
    commitThen (fun cs => .ret (stageMsg b.hs x (asRounds (hist.take (IV.k (xIOf x)))) cs))
      ((List.range (b.nC x)).map fun j =>
        ((b.ref.digests w (pre t).1 (hist.take (IV.k (xIOf x)))).getD j 0, (pre t).2 j)) []⟩

/-- **Before the V\* sessions, the reference prover is `preP`** on a tape whose inner prover's tape and staging salts
are `refP`'s. -/
theorem refP_pre {T : Type} (pre : T → b.ref.TI × (ℕ → Salt)) (w : Claims.Flock.Wit) (x : Claims.Msg)
    (hist : List (Claims.Msg × Claims.Msg)) (ω : Tape b) (t : T) (hω : pre t = (ω.1, ω.2.1))
    (h : hist.length < kPre IV x) : (refP IV b w).next x hist ω = (preP IV b pre w).next x hist t := by
  unfold kPre at h
  by_cases h1 : hist.length < IV.k (xIOf x)
  · simp only [refP, preP, h1, ↓reduceIte, hω]
  · simp only [refP, preP, h1, h, ↓reduceIte, hω]

/-- The V\* sessions' witnesses as the record's reference prover reads them (`RefParts.wit`): off the inner witness,
the inner prover's tape and the inner rounds; no staging salt. -/
def refWit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit :=
  fun w p h s => b.ref.wit w p.1 h s

/-- **Session `s`'s V\* statement** at a history that starts it: the one the rounds before the sessions stage. -/
noncomputable def xsOf (x : Claims.Msg) (s : ℕ) (h : List (Claims.Msg × Claims.Msg)) : Claims.Msg :=
  (vstar b.st x (asRounds (h.take (kPre IV x)))).getD s []

/-- **The masked run's parts**: the first part, then session `s` of `(nums x s).rounds + 1` rounds (its `Hello`
first) at the V\* statement `xsOf` reads, with the witness `wit` reads off the inner witness, the first part's tape
(through `pre`) and the inner rounds. -/
noncomputable def lay {T : Type} (pre : T → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit) :
    RunParts Claims.Flock.Wit Claims.Flock.Wit T where
  k₀ := kPre IV
  n x s := (nums b.st x s).rounds + 1
  xs := xsOf IV b
  ws s w t x h := wit w (pre t) (h.take (IV.k (xIOf x))) s

/-- The parts with no witness: what the simulator reads. -/
noncomputable def parts (T : Type) : RunParts Claims.Flock.Wit Claims.Flock.Wit T where
  k₀ := kPre IV
  n x s := (nums b.st x s).rounds + 1
  xs := xsOf IV b
  ws _ _ _ _ _ := ⟨[]⟩

theorem lay_start {T : Type} (pre : T → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit) :
    (lay IV b pre wit).start = (parts IV b T).start :=
  RunParts.start_congr rfl rfl

/-- **The masked reference prover**: `preP`, then each of the `nS x` V\* sessions the session prover `PS` on slot
`s`'s tape, at its V\* statement and witness. -/
noncomputable def refPM {T TS : Type} (pre : T → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover TS) (w : Claims.Flock.Wit) : Claims.Prover (T × (ℕ → TS)) :=
  ⟨fun x hist ω => ((lay IV b pre wit).prover (preP IV b pre) PS (nS x) w).next x hist ω⟩

/-- **The masked run's rounds**: the first part's, then each V\* session's and its `Hello`. -/
noncomputable def kM (x : Claims.Msg) : ℕ :=
  kPre IV x + ((List.range (nS x)).map fun s => (nums b.st x s).rounds + 1).sum

theorem parts_start (T : Type) (x : Claims.Msg) :
    ∀ m, (parts IV b T).start x m = kPre IV x + ((List.range m).map fun s => (nums b.st x s).rounds + 1).sum
  | 0 => by simp [RunParts.start, parts]
  | m + 1 => by
      rw [RunParts.start, parts_start T x m, List.range_succ, List.map_append, List.sum_append]
      simp [parts, add_assoc]

/-- **The masked run's rounds are the compiled verifier's and one `Hello` per V\* session.** -/
theorem kM_eq (x : Claims.Msg) : kM IV b x = (V IV b.st).k x + nS x := by
  have e : ∀ m, ((List.range m).map fun s => (nums b.st x s).rounds + 1).sum =
      ((List.range m).map fun s => (nums b.st x s).rounds).sum + m := by
    intro m
    induction m with
    | zero => simp
    | succ m ih => simp only [List.range_succ, List.map_append, List.sum_append, ih]; simp; omega
  simp only [kM, kPre, V, e]
  omega

/-- The sessions start after the first part, at least one round in. -/
theorem parts_start_pos (T : Type) (x : Claims.Msg) (j : ℕ) : 0 < (parts IV b T).start x j := by
  rw [parts_start]
  simp [kPre, kS]

/-- **The composed simulator** of the masked run: the first part's simulator `S₀`, then, while the claim's test
(`Recursion.goesOn`) holds, session `s`'s outer simulator `SO x s` at the V\* statement its simulated history stages,
on coins of its own. -/
noncomputable def simM (T₀ : SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (S₀ : Claims.Simulator) (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) : Claims.Simulator where
  C x := (iterSimG T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB (parts IV b (T₀.N × T₀.Ω))) S₀ (fun j x => (SO x j).C)
    (fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn) (nS x)).C x
  prog x := (iterSimG T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB (parts IV b (T₀.N × T₀.Ω))) S₀ (fun j x => (SO x j).C)
    (fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn) (nS x)).prog x

/-- **The recursion's zero knowledge at one statement, by sequential composition**: at `x`, where the first part is
`ZKmalAt` at `S₀` with `ε₀` (`h₀`), each session is `ZKmalAt` at its outer simulator with `eB s` and `g` at every
history that starts it (`hS`), and there the witness `wit` gives is in the sessions' relation `RF` (`hW`), the masked
reference prover on the flat tape is `ZKmalAt` at `simM` with `ε₀ + Σ_{s < nS x} eB s` and `nS x · g`, at the masked
run's rounds `kM x`. -/
theorem zkAt (H : Claims.Msg → Claims.Msg) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (T₀ : SeqTape) (pre : T₀.N × T₀.Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (RA RF : Claims.Rel Claims.Flock.Wit) (S₀ : Claims.Simulator)
    (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) (x : Claims.Msg) (ε₀ g : ℝ) (eB : ℕ → ℝ)
    (h₀ : ZKmalAt H (kPre IV x) (preP IV b pre) (RA x) x (S₀.C x) (S₀.prog x) ε₀ 0)
    (hS : ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg), u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
      ZKmalAt H ((nums b.st x j).rounds + 1) PS (RF (xsOf IV b x j u)) (xsOf IV b x j u) (SO x j).C
        ((SO x j).prog (xsOf IV b x j u)) (eB j) g)
    (hW : ∀ w, RA x w → ∀ j < nS x, ∀ (t₀ : T₀.N × T₀.Ω) (u : List (Claims.Msg × Claims.Msg)),
      u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
        RF (xsOf IV b x j u) (wit w (pre t₀) (u.take (IV.k (xIOf x))) j))
    (he0 : ∀ j < nS x, 0 ≤ eB j) (hg : 0 ≤ g) :
    ZKmalAt H (kM IV b x) (fun w => (refPM IV b pre wit PS w).comap (RunParts.flatten T₀ 0 NB ΩB (nS x))) (RA x) x
      ((simM IV b T₀ NB ΩB S₀ SO).C x) ((simM IV b T₀ NB ΩB S₀ SO).prog x)
      (ε₀ + ∑ j ∈ Finset.range (nS x), eB j) (nS x * g) := by
  have hst := lay_start IV b pre wit
  have h := Claims.ZKMal.zkmalAt_layout T₀ 0 (lay IV b pre wit) H x (nS x) (preP IV b pre) PS RA RF S₀
    (fun j x => (SO x j).C) (fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn)
    ε₀ 0 g eB h₀ (fun j hj u hu => hS j hj u (by
      have hl := RunParts.reach_length T₀ 0 NB ΩB (lay IV b pre wit) H _ RA x hu
      rwa [hst] at hl))
    (fun j hj => RunParts.relOnAt_of T₀ 0 NB ΩB (lay IV b pre wit) H _ RA RF x fun w hw t₀ u hu =>
      hW w hw j hj t₀ u (by rwa [hst] at hu))
    (fun j _ w _ Θ Vs θ ωA => by
      rw [Claims.ZKMal.runStar_msgs]
      exact Claims.ZKMal.recGoesOn_interact _ _ _ _ (by
        show 0 < (lay IV b pre wit).start x j
        rw [hst]
        exact parts_start_pos IV b _ x j))
    (by simp) he0 hg
  rw [RunParts.iterSimG_seq_congr T₀ 0 NB ΩB (lay IV b pre wit) (parts IV b _) rfl rfl rfl, hst, parts_start] at h
  simp only [Nat.cast_zero, zero_add] at h
  exact h.of_equiv _ (Equiv.refl _) (Equiv.refl _) fun _ _ _ => rfl

/-- **The recursion's zero knowledge by sequential composition, on statement-indexed tapes**: `zkAt` at every
statement, with `nS x` sessions on `x`'s tape (`ZKmalRx`). -/
theorem zkmalRx (H : Claims.Msg → Claims.Msg) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (T₀ : SeqTape) (pre : T₀.N × T₀.Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (RA RF : Claims.Rel Claims.Flock.Wit) (S₀ : Claims.Simulator)
    (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) (ε₀ g : Claims.Msg → ℝ) (eB : ℕ → Claims.Msg → ℝ)
    (h₀ : ∀ x, ZKmalAt H (kPre IV x) (preP IV b pre) (RA x) x (S₀.C x) (S₀.prog x) (ε₀ x) 0)
    (hS : ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg), u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
      ZKmalAt H ((nums b.st x j).rounds + 1) PS (RF (xsOf IV b x j u)) (xsOf IV b x j u) (SO x j).C
        ((SO x j).prog (xsOf IV b x j u)) (eB j x) (g x))
    (hW : ∀ x w, RA x w → ∀ j < nS x, ∀ (t₀ : T₀.N × T₀.Ω) (u : List (Claims.Msg × Claims.Msg)),
      u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
        RF (xsOf IV b x j u) (wit w (pre t₀) (u.take (IV.k (xIOf x))) j))
    (he0 : ∀ j x, 0 ≤ eB j x) (hg : ∀ x, 0 ≤ g x) :
    Claims.ZKmalRx (fun x => (T₀.iter 0 NB ΩB (nS x)).N) (fun x => (T₀.iter 0 NB ΩB (nS x)).Ω) H (kM IV b)
      (fun w x => (refPM IV b pre wit PS w).comap (RunParts.flatten T₀ 0 NB ΩB (nS x))) RA (simM IV b T₀ NB ΩB S₀ SO)
      (fun x => ε₀ x + ∑ j ∈ Finset.range (nS x), eB j x) (fun x => nS x * g x) := fun x =>
  zkAt IV b H T₀ pre wit PS RA RF S₀ SO x (ε₀ x) (g x) (fun j => eB j x) (h₀ x) (hS x) (hW x) (fun j _ => he0 j x)
    (hg x)

end RecDeploy
