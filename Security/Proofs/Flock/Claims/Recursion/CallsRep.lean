import Proofs.Flock.Claims.Recursion.CallsLe

/-!
# A rep's SHA-512 calls on every run

§14's Merkle check costs `3 + (d − c)` calls per opened position (`cl_merkle`), and a level opens `Σ 2^c` positions over
its strata whatever the coins (`positions_size`), so a level's queries cost `qc lv`. Everything else a rep's
verification does hashes only through the transcript's squeezes, which cost nothing beyond the transcript's potential
(`CallsLe`): the zerocheck, the lincheck and the ring switch cost `0` (`cs_zerocheck`, `cs_lincheck`, `cs_ringSwitch`).
-/

namespace FlockDeploy.KS

open Flock

/-! ## §14's Merkle check -/

theorem cl_node (h : ByteArray → ByteArray) (ms : MerkleScheme) (a b : ByteArray) :
    CL ((ms.opsWith (Hashes.logged h)).node a b) 1 := by
  unfold MerkleScheme.opsWith
  split
  · split
    · exact calcCL.mono (cl_sha256 h (a ++ b)) (Nat.zero_le _)
    · split
      · exact cl_sha512 h (a ++ b)
      · split
        · exact cl_sha512 h (a ++ b)
        · exact calcCL.pure' _ _
  · exact calcCL.pure' _ _

theorem cl_leafWith (h : ByteArray → ByteArray) (x salt : ByteArray) :
    CL (Hm96.Default512.leafWith (Hashes.logged h) x salt) 2 := by
  unfold Hm96.Default512.leafWith
  dsimp only
  exact calcCL.bind (cl_sha512 h _) fun _ => cl_sha512 h _

theorem cl_leaf (h : ByteArray → ByteArray) (ms : MerkleScheme) (row : Array F128) (salt : ByteArray) :
    CL ((ms.opsWith (Hashes.logged h)).leaf row salt) 3 := by
  unfold MerkleScheme.opsWith
  split
  · split
    · exact calcCL.mono (cl_sha256 h _) (Nat.zero_le _)
    · split
      · exact calcCL.mono (cl_sha512 h _) (by omega)
      · split
        · exact calcCL.mono (calcCL.bind (cl_sha512 h _) fun _ => cl_leafWith h _ salt) (by omega)
        · exact calcCL.pure' _ _
  · exact calcCL.pure' _ _

theorem cl_climb (ops : MerkleOps LogV) (hn : ∀ a b, CL (ops.node a b) 1) :
    ∀ (acc : ByteArray) (idx : ℕ) (sibs : List ByteArray), CL (climbWith ops acc idx sibs) sibs.length
  | acc, _, [] => by simp only [climbWith, List.length_nil]; exact calcCL.pure _
  | acc, idx, s :: rest => by
    simp only [climbWith, List.length_cons]
    exact calcCL.mono (calcCL.ite' (calcCL.bind (hn _ _) fun a => cl_climb ops hn a _ rest)
      (calcCL.bind (hn _ _) fun a => cl_climb ops hn a _ rest)) (by omega)

theorem cl_opens (h : ByteArray → ByteArray) (ms : MerkleScheme) (cap : Array ByteArray) (c d lanes pos : ℕ)
    (row : Array F128) (salt : ByteArray) (sibs : List ByteArray) :
    CL (opensWith (ms.opsWith (Hashes.logged h)) ms cap c d lanes pos row salt sibs) (3 + (d - c)) := by
  unfold opensWith
  split
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    exact calcCL.mono (calcCL.bind (cl_leaf h ms row salt) fun a =>
      calcCL.bind (cl_climb _ (cl_node h ms) a pos sibs) fun _ => calcCL.pure _) (by omega)
  · exact calcCL.pure' _ _

/-- **§14's Merkle check makes at most `3 + (d − c)` calls per position**, whether it accepts or throws. -/
theorem cl_merkle (h : ByteArray → ByteArray) (ms : MerkleScheme) (cap : Array ByteArray) (c d : ℕ)
    (positions : Array ℕ) (rows : Array (Array F128)) (siblings salts : Array ByteArray) (lanes : ℕ) :
    CL (merkleCheckWith (ms.opsWith (Hashes.logged h)) ms cap c d positions rows siblings salts lanes)
      (positions.size * (3 + (d - c))) := by
  unfold merkleCheckWith
  split
  · exact calcCL.mono (cl_throw _) (Nat.zero_le _)
  · split
    · exact calcCL.mono (cl_throw _) (Nat.zero_le _)
    · refine calcCL.mono (calcCL.bind (k := 0) (calcCL.findM? (c := 3 + (d - c)) (fun j => ?_) _) fun bad => ?_)
        (by rw [List.length_range]; omega)
      · exact calcCL.mono (calcCL.bind (cl_opens h ms cap c d lanes _ _ _ _) fun _ => calcCL.pure _) (by omega)
      · split
        · exact cl_throw _
        · exact calcCL.pure _

/-! ## A level's positions -/

theorem id_forIn_add {α β : Type} (P : β → ℕ) (g : α → ℕ) {body : α → β → Id (ForInStep β)}
    (hb : ∀ a b, ∃ b', body a b = pure (.yield b') ∧ P b' = P b + g a) :
    ∀ (xs : List α) (b : β), P (forIn xs b body).run = P b + (xs.map g).sum
  | [], b => by simp
  | a :: xs, b => by
    obtain ⟨b', e, hp⟩ := hb a b
    rw [List.forIn_cons, e, pure_bind]
    dsimp only
    rw [id_forIn_add P g hb xs b', hp, List.map_cons, List.sum_cons]; omega

/-- **A level opens `Σ 2^c` positions over its strata**, whatever the coins. -/
theorem positions_size (lv : Level) (coins : Array F128) :
    (positions lv coins).size = (lv.strata.toList.map (2 ^ ·)).sum := by
  unfold positions
  dsimp only
  rw [← Array.forIn_toList]
  simp only [Id.run_bind, Id.run_pure]
  refine (id_forIn_add (fun p : Array ℕ × ℕ => p.1.size) (fun c => 2 ^ c) ?_ _ _).trans (by simp)
  intro c s
  refine ⟨_, rfl, ?_⟩
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  refine (id_forIn_add (fun p : Array ℕ × ℕ => p.1.size) (fun _ => 1) (fun a b => ⟨_, rfl, by simp⟩) _ _).trans ?_
  rw [Calc.sum_map_const, List.length_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, Nat.mul_one]

/-- A level's queries' calls: `3 + (d − cap)` for each of its positions. -/
def qc (lv : Level) : ℕ := (lv.strata.toList.map (2 ^ ·)).sum * (3 + (lv.d - lv.cap))

/-! ## What hashes nothing, continued -/

theorem keepT_throw {α : Type} (e : String) : KeepT (throw e : Transcript.TM α) := fun t a t' h => by cases h

theorem keepT_check (b : Bool) (why : String) : KeepT (check b why) := by
  unfold check
  split
  · exact keepT_pure _
  · exact keepT_throw _

theorem cs2_throw {σ τ α : Type} (pot : τ → ℕ) (e : String) : CS2 pot (throw e : StateT σ (StateT τ LogV) α) 0 :=
  cs2_lift pot (cs_throw pot e)

theorem cs2_mlift {α : Type} {x : LogV α} {n : ℕ} (hx : CL x n) : CS2 potT (monadLift x : LMW LogV α) n :=
  cs2_lift potT (cs_lift potT hx)

/-- Walks a transcript program over the logged oracle whose calls are all squeezes, at any budget. -/
macro "tw_walk" h:term : tactic => `(tactic| repeat' (first
  | with_reducible exact (calcCS potT).pure' _ _
  | with_reducible exact (calcCS potT).mono (cs_sample $h) (Nat.zero_le _)
  | with_reducible exact (calcCS potT).mono (cs_sampleVec $h _) (Nat.zero_le _)
  | with_reducible exact (calcCS potT).mono (cs_sampleK $h) (Nat.zero_le _)
  | with_reducible exact (calcCS potT).mono (cs_throw potT _) (Nat.zero_le _)
  | with_reducible exact (calcCS potT).mono (cs_liftTM (keepT_check _ _)) (Nat.zero_le _)
  | with_reducible exact (calcCS potT).mono (cs_liftTM (by keep_walk)) (Nat.zero_le _)
  | with_reducible refine (calcCS potT).mono ((calcCS potT).range_forIn (c := 0) (fun _ _ => ?_) _ _) (by simp)
  | with_reducible refine (calcCS potT).bind0 ?_ fun _ => ?_
  | split
  | dsimp only))

theorem cs_zerocheck (h : ByteArray → ByteArray) (m : ℕ) (digest cap : ByteArray) (p : ZerocheckProof) :
    CS potT (bindAndZerocheckWith (Hashes.logged h) m digest cap p) 0 := by
  unfold bindAndZerocheckWith
  tw_walk h

theorem cs_lincheck (h : ByteArray → ByteArray) (c : CircuitFold) (zc : ZcOut) (p : LincheckProof) :
    CS potT (lincheckWith (Hashes.logged h) c zc p) 0 := by
  unfold lincheckWith
  tw_walk h

theorem cs_ringSwitch (h : ByteArray → ByteArray) (m : ℕ) (claims : Array Claim) (rs : Array (Array F128)) :
    CS potT (ringSwitchWith (Hashes.logged h) m claims rs) 0 := by
  unfold ringSwitchWith
  tw_walk h

/-! ## The Ligerito layer -/

theorem Calc.bind_le {m : Type → Type} [Monad m] {P : ∀ {α : Type}, m α → ℕ → Prop} (C : Calc m P) {α β : Type}
    {x : m α} {f : α → m β} {n b : ℕ} (hx : P x n) (hf : ∀ a, P (f a) (b - n)) (h : n ≤ b) : P (x >>= f) b :=
  C.mono (C.bind hx hf) (by omega)

theorem Calc.ite_k {m : Type → Type} [Monad m] {P : ∀ {α : Type}, m α → ℕ → Prop} (_C : Calc m P) {α : Type}
    {c : Prop} [Decidable c] {t e : m α} {k : ℕ} (ht : P t k) (he : P e k) : P (if c then t else e) k := by
  split
  · exact ht
  · exact he

/-- A throw costs nothing, and what would follow it never runs. -/
theorem cs2_throw_bind {α β : Type} (e : String) (f : α → LMW LogV β) (k : ℕ) : CS2 potT (throw e >>= f) k :=
  fun s t l => by
    have : (((throw e >>= f : LMW LogV β).run s).run t).go l = (.error e, l) := rfl
    rw [this, bnd_error]
    omega

theorem cs_throw_bind {α β : Type} (e : String) (f : α → TMW LogV β) (k : ℕ) : CS potT (throw e >>= f) k :=
  fun t l => by
    have : ((throw e >>= f : TMW LogV β).run t).go l = (.error e, l) := rfl
    rw [this, bnd_error]
    omega

/-- One step of a walk over a Ligerito program whose calls are all squeezes, at any budget. -/
macro "lw_step" h:term : tactic => `(tactic| first
  | with_reducible exact cs2_throw_bind _ _ _
  | with_reducible refine (calcCS2 potT).bind0 ?_ fun _ => ?_
  | with_reducible exact (calcCS2 potT).pure' _ _
  | with_reducible exact (calcCS2 potT).mono (cs2_liftW (cs_sample $h)) (Nat.zero_le _)
  | with_reducible exact (calcCS2 potT).mono (cs2_liftW (cs_sampleVec $h _)) (Nat.zero_le _)
  | with_reducible exact (calcCS2 potT).mono (cs2_liftW (cs_sampleK $h)) (Nat.zero_le _)
  | with_reducible exact (calcCS2 potT).mono (cs2_liftLM (keepL_nextMsg _)) (Nat.zero_le _)
  | with_reducible exact (calcCS2 potT).mono (cs2_liftLM (keepL_nextOod _)) (Nat.zero_le _)
  | with_reducible exact (calcCS2 potT).mono (cs2_liftLM (keepL_claimPow _)) (Nat.zero_le _)
  | with_reducible exact (calcCS2 potT).mono (cs2_liftLM (keepL_nonceFrom _ _ _)) (Nat.zero_le _)
  | with_reducible exact (calcCS2 potT).mono (cs2_liftLM (by keep_walk)) (Nat.zero_le _)
  | with_reducible exact (calcCS2 potT).mono (cs2_get (σ := LState) (τ := Transcript) potT) (Nat.zero_le _)
  | with_reducible exact (calcCS2 potT).mono (cs2_set (σ := LState) (τ := Transcript) potT _) (Nat.zero_le _)
  | with_reducible exact (calcCS2 potT).mono (cs2_modify (σ := LState) (τ := Transcript) potT _) (Nat.zero_le _)
  | with_reducible exact (calcCS2 potT).mono (cs2_throw (σ := LState) (τ := Transcript) potT _) (Nat.zero_le _)
  | with_reducible refine (calcCS2 potT).mono ((calcCS2 potT).range_forIn (c := 0) (fun _ _ => ?_) _ _) (by simp)
  | with_reducible refine (calcCS2 potT).mono ((calcCS2 potT).forIn (c := 0) (fun _ _ => ?_) _ _) (by simp)
  | with_reducible refine (calcCS2 potT).ite_k ?_ ?_
  | split
  | dsimp only)

theorem cs2_challenge (h : ByteArray → ByteArray) (p : LigeritoProof) :
    CS2 potT (challengeWith (Hashes.logged h) p) 0 := by
  unfold challengeWith
  repeat' lw_step h

theorem cs2_oodIntro (h : ByteArray → ByteArray) (p : LigeritoProof) (dim split : ℕ) :
    CS2 potT (oodIntroWith (Hashes.logged h) p dim split) 0 := by
  unfold oodIntroWith
  repeat' lw_step h

/-- **A level's queries cost `qc lv`**: its Merkle check; every coin is a squeeze. -/
theorem cs2_queryLevel (h : ByteArray → ByteArray) (ms : MerkleScheme) (p : LigeritoProof) (lv : Level)
    (cap : Array ByteArray) (rows : Array (Array F128)) (sibs salts : Array ByteArray) (fold : Array F256) :
    CS2 potT (queryLevelWith (Hashes.logged h) ms p lv cap rows sibs salts fold) (qc lv) := by
  unfold queryLevelWith
  repeat' (first
    | with_reducible refine (calcCS2 potT).bind_le (cs2_mlift (cl_merkle h _ _ _ _ _ _ _ _ _)) (fun _ => ?_) ?_
    | lw_step h)
  rw [positions_size]
  exact Nat.le_refl _

/-- A schedule's levels' query calls together. -/
def qAll (sch : Schedule) : ℕ := (sch.levels.toList.map qc).sum

theorem le_sum_map {α : Type} (f : α → ℕ) : ∀ {l : List α} {x : α}, x ∈ l → f x ≤ (l.map f).sum
  | _ :: _, _, .head _ => by rw [List.map_cons, List.sum_cons]; omega
  | _ :: _, _, .tail _ h => by have := le_sum_map f h; rw [List.map_cons, List.sum_cons]; omega

theorem qc_getD_le (sch : Schedule) (i : ℕ) : qc (sch.levels.getD i default) ≤ qAll sch := by
  unfold Array.getD
  split
  · exact le_sum_map qc (Array.getElem_mem_toList _)
  · exact Nat.zero_le _

theorem Calc.bind0r {m : Type → Type} [Monad m] {P : ∀ {α : Type}, m α → ℕ → Prop} (C : Calc m P) {α β : Type}
    {x : m α} {f : α → m β} {k : ℕ} (hx : P x k) (hf : ∀ a, P (f a) 0) : P (x >>= f) k :=
  C.mono (C.bind hx hf) (by omega)

theorem cs_run2 {σ τ α : Type} {pot : τ → ℕ} {x : StateT σ (StateT τ LogV) α} {n : ℕ} (hx : CS2 pot x n) (s : σ) :
    CS pot (x.run s) n :=
  hx s

/-- One step of a walk over a transcript program whose calls are all squeezes, at any budget. -/
macro "tw_step" h:term : tactic => `(tactic| first
  | with_reducible exact cs_throw_bind _ _ _
  | with_reducible refine (calcCS potT).bind0 ?_ fun _ => ?_
  | with_reducible exact (calcCS potT).pure' _ _
  | with_reducible exact (calcCS potT).mono (cs_sample $h) (Nat.zero_le _)
  | with_reducible exact (calcCS potT).mono (cs_sampleVec $h _) (Nat.zero_le _)
  | with_reducible exact (calcCS potT).mono (cs_sampleK $h) (Nat.zero_le _)
  | with_reducible exact (calcCS potT).mono (cs_throw potT _) (Nat.zero_le _)
  | with_reducible exact (calcCS potT).mono (cs_liftTM (keepT_check _ _)) (Nat.zero_le _)
  | with_reducible exact (calcCS potT).mono (cs_liftTM (by keep_walk)) (Nat.zero_le _)
  | with_reducible refine (calcCS potT).mono ((calcCS potT).range_forIn (c := 0) (fun _ _ => ?_) _ _) (by simp)
  | with_reducible refine (calcCS potT).ite_k ?_ ?_
  | split
  | dsimp only)

set_option maxHeartbeats 20000000 in
/-- **Ligerito's calls are its levels' queries**: one level-0 query and one per later round, each at most `qAll`. -/
theorem cs_ligerito (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) :
    CS potT (ligeritoWith (Hashes.logged h) ms sch T cap p basisAt) ((1 + (sch.levels.size - 1)) * qAll sch) := by
  unfold ligeritoWith
  repeat' (first
    | with_reducible refine (calcCS potT).bind0r (cs_run2 ?_ _) fun _ => ?_
    | with_reducible refine (calcCS2 potT).bind_le (cs2_queryLevel h _ _ _ _ _ _ _ _) (fun _ => ?_) ?_
    | with_reducible refine (calcCS2 potT).bind_le ((calcCS2 potT).range_forIn (β := _ × _ × _ × _)
                              (c := qAll sch) (fun _ _ => ?_) _ _) (fun _ => ?_) ?_
    | with_reducible exact (calcCS2 potT).mono (cs2_challenge h _) (Nat.zero_le _)
    | with_reducible exact (calcCS2 potT).mono (cs2_oodIntro h _ _ _) (Nat.zero_le _)
    | lw_step h
    | tw_step h)
  all_goals first
    | exact qc_getD_le _ _
    | (have := qc_getD_le sch 0
       simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, Nat.add_mul, Nat.one_mul]
       omega)

end FlockDeploy.KS
