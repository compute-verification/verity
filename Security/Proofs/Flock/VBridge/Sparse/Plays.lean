import Proofs.Flock.VBridge.Sparse.Residuals
import Proofs.Flock.Soundness.Refine.Plays
import Proofs.Flock.Soundness.Instance

/-!
# A run of the sparse phase on `RecSparse_v1`'s ports

`sparsePhase_plays`: the model's `sparsePhase` at `execArith`, with L4′'s `h` w's first 64 elements, fed w's messages
(`phaseMsgs`) and the phase's coins one word each (`phaseCoins`, which is `coinTape cs 0 nc`), ends at `verdictOf`:
`sparseVerdict` on the messages and coins read through `fOf`, with H2's and H3's claims the folds of `degNext` over
the rounds (`degFold`).
-/

namespace FlockVBridge.Sparse

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Game

local notation "𝔽" => FlockLevel3.GF128

noncomputable section

/-! ## Rounds and coins -/

/-- Coins `a … a + n − 1` of `cs`, one word each. -/
def coinTape (cs : Array F128) (a n : ℕ) : List (Array F128) := (List.range' a n).map fun j => #[cs[j]!]

/-- Rounds `a … a + n − 1`'s messages: round `j` sends `msg j`. -/
def msgTape {k : ℕ} (msg : ℕ → 𝔽 × (Fin k → 𝔽)) (a n : ℕ) : List Msg :=
  (List.range' a n).map fun j => ⟨𝔽 × (Fin k → 𝔽), msg j⟩

theorem coinTape_succ (cs : Array F128) (a n : ℕ) : coinTape cs a (n + 1) = #[cs[a]!] :: coinTape cs (a + 1) n := by
  simp only [coinTape, List.range'_succ, List.map_cons]

theorem coinTape_add (cs : Array F128) (a m n : ℕ) :
    coinTape cs a (m + n) = coinTape cs a m ++ coinTape cs (a + m) n := by
  simp only [coinTape, ← List.map_append, List.range'_append_1]

theorem msgTape_succ {k : ℕ} (msg : ℕ → 𝔽 × (Fin k → 𝔽)) (a n : ℕ) :
    msgTape msg a (n + 1) = ⟨𝔽 × (Fin k → 𝔽), msg a⟩ :: msgTape msg (a + 1) n := by
  simp only [msgTape, List.range'_succ, List.map_cons]

theorem getElem!_single (x : F128) : (#[x] : Array F128)[0]! = x := by
  simp

/-- **`drawFin`'s run**: `n` coins, one word each, read through `fOf`. -/
theorem drawFin_plays (cs : Array F128) : ∀ (n a : ℕ),
    Plays fOf (drawFin (F := 𝔽) n) [] (coinTape cs a n) (fun i => fOf cs[a + i.val]!)
  | 0, a => by
      have e : (Fin.elim0 : Fin 0 → 𝔽) = fun i => fOf cs[a + i.val]! := funext fun i => i.elim0
      rw [drawFin, ← e]
      exact Plays.pure _
  | n + 1, a => by
      have e : (Fin.cons (fOf (#[cs[a]!] : Array F128)[0]!) (fun i : Fin n => fOf cs[a + 1 + i.val]!) :
          Fin (n + 1) → 𝔽) = fun i => fOf cs[a + i.val]! := by
        funext i
        refine Fin.cases ?_ (fun j => ?_) i
        · simp
        · simp only [Fin.cons_succ, Fin.val_succ]
          rw [show a + 1 + j.val = a + (j.val + 1) by omega]
      rw [drawFin, coinTape_succ, ← e]
      refine Plays.bind' (ms₂ := []) (cs₂ := coinTape cs (a + 1) n) (Plays.drawOne #[cs[a]!] rfl) ?_ rfl rfl
      refine Plays.bind' (ms₂ := []) (cs₂ := []) (drawFin_plays cs n (a + 1)) ?_ rfl (List.append_nil _).symm
      exact Plays.pure _

/-- **`degRounds`' run**: round `j`'s message `msg j` and coin `cs[o + j]`, from round `a`; the claim is `degFold`. -/
theorem degRounds_plays (k : ℕ) (msg : ℕ → 𝔽 × (Fin k → 𝔽)) (cs : Array F128) (o : ℕ) :
    ∀ (n a : ℕ) (claim : 𝔽), Plays fOf (degRounds k n claim) (msgTape msg a n) (coinTape cs (o + a) n)
      (fun i => fOf cs[o + a + i.val]!, degFold msg (fun j => fOf cs[o + j]!) a n claim)
  | 0, a, claim => by
      have e : ((Fin.elim0 : Fin 0 → 𝔽), claim) =
          (fun i : Fin 0 => fOf cs[o + a + i.val]!, degFold msg (fun j => fOf cs[o + j]!) a 0 claim) := by
        refine Prod.ext (funext fun i => i.elim0) ?_
        simp [degFold]
      rw [degRounds, ← e]
      exact Plays.pure _
  | n + 1, a, claim => by
      have ih := degRounds_plays k msg cs o n (a + 1) (degNext claim (msg a) (fOf (#[cs[o + a]!] : Array F128)[0]!))
      rw [show o + (a + 1) = o + a + 1 by omega] at ih
      have e1 : (Fin.cons (fOf (#[cs[o + a]!] : Array F128)[0]!) (fun i : Fin n => fOf cs[o + a + 1 + i.val]!) :
          Fin (n + 1) → 𝔽) = fun i => fOf cs[o + a + i.val]! := by
        funext i
        refine Fin.cases ?_ (fun j => ?_) i
        · simp
        · simp only [Fin.cons_succ, Fin.val_succ]
          rw [show o + a + 1 + j.val = o + a + (j.val + 1) by omega]
      have e2 : degFold msg (fun j => fOf cs[o + j]!) (a + 1) n
          (degNext claim (msg a) (fOf (#[cs[o + a]!] : Array F128)[0]!)) =
          degFold msg (fun j => fOf cs[o + j]!) a (n + 1) claim := by
        simp only [degFold, List.range'_succ, List.foldl_cons, getElem!_single]
      rw [degRounds, msgTape_succ, coinTape_succ, ← e1, ← e2]
      refine Plays.bind' (ms₂ := msgTape msg (a + 1) n) (cs₂ := #[cs[o + a]!] :: coinTape cs (o + a + 1) n)
        (Plays.recv _ (msg a)) ?_ rfl rfl
      refine Plays.bind' (ms₂ := msgTape msg (a + 1) n) (cs₂ := coinTape cs (o + a + 1) n)
        (Plays.drawOne #[cs[o + a]!] rfl) ?_ rfl rfl
      refine Plays.bind' (ms₂ := []) (cs₂ := []) ih ?_ (List.append_nil _).symm (List.append_nil _).symm
      exact Plays.pure _

/-! ## The phase -/

variable (sh : SparseShape)

/-- The public values at the coins. -/
def pubOf (c : SCoins) : SparsePub 𝔽 sh :=
  ⟨fOf c.z, fOf c.alpha, fun i => fOf c.xRest[i.val]!, fun i => fOf c.yRest[i.val]!, fun σ => fOf c.U[σ.val]!⟩

/-- H2's round `j`'s message at `pre<j>`: `(c₀, c₂)`. -/
def msg2 (w : Array F128) (j : ℕ) : 𝔽 × (Fin 1 → 𝔽) := (fOf w[wPre j]!, fun _ => fOf w[wPre j + 1]!)

/-- H3's round `j`'s message at `sc<j>`: `c₀`, then `c₂ … c_{k+1}`. -/
def msg3 (w : Array F128) (k j : ℕ) : 𝔽 × (Fin k → 𝔽) :=
  (fOf w[wSc sh j]!, fun i => fOf w[wSc sh j + (i.val + 1)]!)

/-- The planes `u`. -/
def uOf (w : Array F128) : Fin (2 ^ sh.pLog) → 𝔽 := fun q => fOf w[wU sh + q.val]!

/-- The types' values `vt` (none sent with `T = 1`). -/
def vtOf (w : Array F128) : Fin (2 ^ sh.tLog) → 𝔽 :=
  if sh.tLog = 0 then fun _ => 0 else fun t => fOf w[wVt sh + t.val]!

def vtTape (w : Array F128) : List Msg :=
  if sh.tLog = 0 then [] else [⟨Fin (2 ^ sh.tLog) → 𝔽, fun t => fOf w[wVt sh + t.val]!⟩]

/-- The phase's messages, in the order the model receives them. -/
def phaseMsgs (w : Array F128) : List Msg :=
  msgTape (msg2 w) 0 6 ++ ([⟨𝔽, fOf w[wV2]!⟩] ++ (msgTape (msg3 sh w (2 * sh.ell)) 0 sh.nu ++
    (msgTape (msg3 sh w (2 * sh.ell + 1)) sh.nu sh.sLog ++ ([⟨Fin (2 ^ sh.pLog) → 𝔽, uOf sh w⟩] ++ vtTape sh w))))

/-- The phase's coins, in the order the model draws them. -/
def phaseCoins (cs : Array F128) : List (Array F128) :=
  coinTape cs 0 6 ++ (coinTape cs 6 6 ++ (coinTape cs 12 sh.nu ++ (coinTape cs (12 + sh.nu) sh.sLog ++
    (coinTape cs (12 + nR sh) sh.pLog ++ coinTape cs (12 + nR sh + sh.pLog) sh.tLog))))

theorem phaseCoins_eq (cs : Array F128) : phaseCoins sh cs = coinTape cs 0 (nc sh) := by
  rw [phaseCoins, show nc sh = 6 + (6 + (sh.nu + (sh.sLog + (sh.pLog + sh.tLog)))) by unfold nc nR; omega]
  rw [coinTape_add, coinTape_add, coinTape_add, coinTape_add, coinTape_add]
  simp only [Nat.zero_add, show 6 + 6 = 12 by rfl, nR, Nat.add_assoc]

/-- H1's claim at `ζ_c`. -/
def h1Claim (w cs : Array F128) : 𝔽 :=
  ∑ s : Fin 64, eqAt (fun i : Fin 6 => fOf cs[0 + i.val]!) s.val * fOf w[s.val]!

/-- H3's claim after the entries' rounds. -/
def c3Of (w cs : Array F128) : 𝔽 := degFold (msg3 sh w (2 * sh.ell)) (fun j => fOf cs[12 + j]!) 0 sh.nu (fOf w[wV2]!)

/-- **Where the run ends**: `sparseVerdict` on w's messages and the coins. -/
def verdictOf (w : Array F128) (τ : Array ℕ) (c : SCoins) : Option (RegClaim 𝔽 sh) :=
  sparseVerdict execArith sh (pubOf sh c) (fun σ => τ[σ.val]!) (fun i => fOf c.cs[0 + i.val]!)
    (fun i => fOf c.cs[6 + i.val]!) (degFold (msg2 w) (fun j => fOf c.cs[6 + j]!) 0 6 (h1Claim w c.cs))
    (fOf w[wV2]!) (fun i => fOf c.cs[12 + i.val]!) (fun i => fOf c.cs[12 + sh.nu + i.val]!)
    (degFold (msg3 sh w (2 * sh.ell + 1)) (fun j => fOf c.cs[12 + j]!) sh.nu sh.sLog (c3Of sh w c.cs))
    (uOf sh w) (fun i => fOf c.cs[12 + nR sh + i.val]!) (vtOf sh w)
    (fun i => fOf c.cs[12 + nR sh + sh.pLog + i.val]!)

/-- **A run of the sparse phase** on `RecSparse_v1`'s messages and coins, to `verdictOf`. -/
theorem sparsePhase_plays (w : Array F128) (τ : Array ℕ) (c : SCoins) :
    Plays fOf (sparsePhase execArith sh (pubOf sh c) (fun σ => τ[σ.val]!) (fun s => fOf w[s.val]!))
      (phaseMsgs sh w) (phaseCoins sh c.cs) (verdictOf sh w τ c) := by
  have h1 := drawFin_plays c.cs 6 0
  have h2 := degRounds_plays 1 (msg2 w) c.cs 6 6 0 (h1Claim w c.cs)
  have h3 := degRounds_plays (2 * sh.ell) (msg3 sh w (2 * sh.ell)) c.cs 12 sh.nu 0 (fOf w[wV2]!)
  have h4 := degRounds_plays (2 * sh.ell + 1) (msg3 sh w (2 * sh.ell + 1)) c.cs 12 sh.sLog sh.nu (c3Of sh w c.cs)
  have h5 := drawFin_plays c.cs sh.pLog (12 + nR sh)
  have h7 := drawFin_plays c.cs sh.tLog (12 + nR sh + sh.pLog)
  simp only [Nat.add_zero] at h2 h3
  unfold sparsePhase
  refine Plays.bind' (ms₂ := phaseMsgs sh w) h1 ?_ rfl rfl
  refine Plays.bind' h2 ?_ rfl rfl
  refine Plays.bind' (Plays.recv 𝔽 (fOf w[wV2]!)) ?_ rfl rfl
  refine Plays.bind' h3 ?_ rfl rfl
  refine Plays.bind' h4 ?_ rfl rfl
  refine Plays.bind' (Plays.recv _ (uOf sh w)) ?_ rfl rfl
  refine Plays.bind' h5 (ms₂ := vtTape sh w) (cs₂ := coinTape c.cs (12 + nR sh + sh.pLog) sh.tLog) ?_
    (List.nil_append _).symm rfl
  -- the model's `vt` is a do-notation join point over `if sh.tLog = 0`
  dsimp only
  by_cases ht : sh.tLog = 0
  · have hm : vtTape sh w = [] := by unfold vtTape; exact ite_eq_left ht
    have hv : vtOf sh w = fun _ => 0 := by unfold vtOf; exact ite_eq_left ht
    rw [ite_eq_left ht, hm]
    unfold verdictOf
    rw [hv]
    refine Plays.bind' (ms₂ := []) (cs₂ := coinTape c.cs (12 + nR sh + sh.pLog) sh.tLog) (Plays.pure _) ?_ rfl rfl
    refine Plays.bind' (ms₂ := []) (cs₂ := []) h7 ?_ rfl (List.append_nil _).symm
    exact Plays.pure _
  · have hm : vtTape sh w = [⟨Fin (2 ^ sh.tLog) → 𝔽, fun t => fOf w[wVt sh + t.val]!⟩] := by
      unfold vtTape; exact ite_eq_right ht
    have hv : vtOf sh w = fun t => fOf w[wVt sh + t.val]! := by unfold vtOf; exact ite_eq_right ht
    rw [ite_eq_right ht, hm]
    unfold verdictOf
    rw [hv]
    refine Plays.bind' (ms₂ := []) (cs₂ := coinTape c.cs (12 + nR sh + sh.pLog) sh.tLog)
      (Plays.recv _ (fun t : Fin (2 ^ sh.tLog) => fOf w[wVt sh + t.val]!)) ?_ rfl rfl
    refine Plays.bind' (ms₂ := []) (cs₂ := []) h7 ?_ rfl (List.append_nil _).symm
    exact Plays.pure _

end

end FlockVBridge.Sparse
