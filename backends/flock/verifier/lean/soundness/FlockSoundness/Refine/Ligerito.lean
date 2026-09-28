import FlockSoundness.Refine.ArithK
import FlockSoundness.Refine.ExecLM
import FlockSoundness.Refine.Schedule

/-!
# Ligerito's rounds refine the model's (PROTOCOL.md §13.3)

The executable's Ligerito verifier carries the running claim `s` and the round polynomial `q` as values; the compiled
model carries them as functions of the openings (`LigStateC`), since the consistency values read the opened rows, which
arrive after the last coin. At the openings the proof carries, the two agree.

Each phase is a pair: the executable's `LM` program, and the model's game, run on the proof's messages and the
consumed rounds' coins.

* `fold_refines`: the folding rounds (`foldRoundsC`, the executable's `challenge` loop).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {D : Type}

/-- A sumcheck message of the proof as the model's. -/
def kMsg (m : F256 × F256) : Msg := ⟨𝕂 × 𝕂, (gOf m.1, gOf m.2)⟩

/-- The proof's sumcheck messages `start, …, start + k − 1`. -/
def kMsgs (p : LigeritoProof) (start k : ℕ) : List Msg := (List.range' start k).map fun i => kMsg p.messages[i]!

theorem getElem!_of_getElem? {α : Type} [Inhabited α] {a : Array α} {i : ℕ} {x : α} (h : a[i]? = some x) :
    a[i]! = x := by
  obtain ⟨hi, hx⟩ := Array.getElem?_eq_some_iff.1 h
  rw [getElem!_pos a i hi, hx]

/-- A `K` coin: the round's two coins as `c₀ + c₁·u` (§2.4, §13.2). -/
theorem ofLimbs_coin (c : Array F128) :
    execArith.ofLimbs (fOf c[0]!) (fOf c[1]!) = gOf ⟨c.getD 0 0, c.getD 1 0⟩ := by
  rw [gOf_limbs, getElem!_eq_getD, getElem!_eq_getD]
  rfl

/-- **The folding rounds** (§13.3 step 4 and each recursive round). The executable's `k` challenges, from `stE` and
`t`, are the model's `foldRoundsC`, played with the proof's next `k` sumcheck messages and the `k` rounds consumed; at
the proof's openings `o`, the model's claim and polynomial are the executable's. -/
theorem fold_refines (p : LigeritoProof) (o : Opens 𝔽 D) :
    ∀ (k a : ℕ) (chs chs' : Array F256) (stE stE' : LState) (t t' : Transcript)
      (sM : Opens 𝔽 D → 𝕂) (qM : Opens 𝔽 D → Model.Quad 𝕂), sM o = gOf stE.s → qM o = quadOf stE.q →
      RoundsWF t.rounds →
      forIn (List.range' a k) chs (fun _ chs => challenge p >>= fun x => pure (ForInStep.yield (chs.push x)))
        stE t = .ok ((chs', stE'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + k ∧ stE'.msg = stE.msg + k ∧
        stE'.ood = stE.ood ∧ stE'.qn = stE.qn ∧ stE'.cn = stE.cn ∧ stE'.cbn = stE.cbn ∧ stE'.oods = stE.oods ∧
        stE'.conss = stE.conss ∧ chs'.size = chs.size + k ∧ (∀ i < chs.size, chs'[i]! = chs[i]!) ∧
        ∃ sM' qM', Plays fOf (Model.foldRoundsC execArith k sM qM) (kMsgs p stE.msg k) (consumed t.rounds t.k t'.k)
          (((chs'.extract chs.size).toList.map gOf), sM', qM') ∧ sM' o = gOf stE'.s ∧ qM' o = quadOf stE'.q := by
  intro k
  induction k with
  | zero =>
      intro a chs chs' stE stE' t t' sM qM hs hq _ h
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
      refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, fun _ _ => rfl, sM, qM, ?_, hs, hq⟩
      rw [consumed_self, Model.foldRoundsC]
      simpa [kMsgs] using Plays.pure (ι := fOf) ([], sM, qM)
  | succ k ih =>
      intro a chs chs' stE stE' t t' sM qM hs hq hwf h
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨hlt, hn, m, t₂, hm, hr₂, hk₂, h₁⟩ := challenge_bind h₁
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
      simp only at h₂
      set c := (t.rounds[t.k]!).coins with hc
      set rE : F256 := ⟨c.getD 0 0, c.getD 1 0⟩ with hrE
      have hwf₂ : RoundsWF t₁.rounds := hr₂ ▸ hwf
      have hcsz : c.size = 2 := by rw [hc, RoundsWF.get hwf, hn]
      have hs' : (fun o => (qM o).eval (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!))) o = gOf (stE.q.eval rE) := by
        simp only; rw [hq, ofLimbs_coin, gOf_eval]
      obtain ⟨hr', hk', hmsg', hood', hqn', hcn', hcbn', hoods', hconss', hsz', hkeep', sM', qM', hpl, hsM', hqM'⟩ :=
        ih (a + 1) (chs.push rE) chs'
          { stE with s := stE.q.eval rE, q := Poly.ofMsg m (stE.q.eval rE), msg := stE.msg + 1 } stE' t₁ t'
          (fun o => (qM o).eval (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)))
          (fun o => Model.fromMsg (gOf m.1, gOf m.2)
            ((fun o => (qM o).eval (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!))) o)) hs'
          (by rw [hs', ← quadOf_ofMsg]) hwf₂ h₂
      have hmE : p.messages[stE.msg]! = m := getElem!_of_getElem? hm
      refine ⟨hr'.trans hr₂, by rw [hk', hk₂]; omega, by rw [hmsg']; simp; omega, hood', hqn', hcn', hcbn', hoods',
        hconss', by rw [hsz']; simp; omega, fun i hi => by
          rw [hkeep' i (by simp; omega), getElem!_pos _ i (by simp; omega), Array.getElem_push_lt, getElem!_pos chs i hi],
        sM', qM', ?_, hsM', hqM'⟩
      have hdraw : Plays fOf (Model.drawK execArith) [] [c] (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)) := by
        unfold Model.drawK
        exact Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c hcsz) (Plays.pure _) (by simp) (by simp)
      unfold Model.foldRoundsC
      have hcons : consumed t.rounds t.k t'.k = c :: consumed t₁.rounds t₁.k t'.k := by
        rw [hr₂, hk₂, consumed_append t.rounds (show t.k ≤ t.k + 1 by omega) (by rw [hk', hk₂]; omega),
          consumed_succ]
        rfl
      have hex : (chs'.extract chs.size).toList.map gOf = gOf rE :: ((chs'.extract (chs.push rE).size).toList.map gOf) := by
        have h1 : chs'[chs.size]! = rE := by rw [hkeep' chs.size (by simp)]; simp
        apply List.ext_getElem
        · simp only [List.length_map, Array.length_toList, List.length_cons, Array.size_extract, Array.size_push, hsz']
          omega
        · intro i h₁ h₂
          simp only [List.getElem_map, Array.getElem_toList, Array.getElem_extract]
          rcases i with _ | i
          · simp only [List.getElem_cons_zero, Nat.add_zero]
            rw [← h1, getElem!_pos chs' chs.size (by rw [hsz', Array.size_push]; omega)]
          · simp only [List.getElem_cons_succ, List.getElem_map, Array.getElem_toList, Array.getElem_extract,
              Array.size_push]
            congr 2
            omega
      rw [hcons, hex]
      refine Plays.bind' (ms₂ := kMsgs p stE.msg (k + 1)) (cs₂ := consumed t₁.rounds t₁.k t'.k) hdraw ?_
        (by simp) (by simp)
      refine Plays.bind' (ms₂ := kMsgs p (stE.msg + 1) k) (cs₂ := consumed t₁.rounds t₁.k t'.k)
        (Plays.recv _ (gOf m.1, gOf m.2)) ?_ (by simp [kMsgs, List.range'_succ, kMsg, hmE]) (by simp)
      refine Plays.bind' (ms₂ := []) (cs₂ := []) hpl ?_ (by simp) (by simp)
      rw [ofLimbs_coin]
      exact Plays.pure _

end FlockSoundness.Refine
