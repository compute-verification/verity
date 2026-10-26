import Proofs.Flock.Claims.KS.JointR
import Proofs.Flock.Claims.KS.ZkDecodeW

/-!
# The masked live sessions' bound at the deployed decoders, off the table on a witness lane

`ZkXferJ` at the verdict the deployed decoders are right at with no premise on the extra lanes (`ZkXferWJ`): the
mixture bounds the deployed runs accepting with no column of table `j` off `C` by its live outcomes `verify --zk`
accepts at the draw with no column of table `j` off `C` read and table `j`'s `root_B` the mixture's (`AccTxJ`), and
level 0's fork event at each table bounds the simulated strategies' `Adv₀` after `Commit`. At a masked statement in
scope table `j`'s word has level 0's `2^logLen` columns (`wordLenJ_zkTabs`, from `zkLevel0_d_eq`), so the deployed
decoders right at `OkOffWJ` (`decodesTableW_deploy`) give `ZkLiveAccWJ`'s verdict (`zkLiveAccWJ_of_xferWJ`). So
`Flock.KS` at the deployed masked record follows from `ZkXferWJ` where some run accepts and `ExtendJ`
(`zk_KS_of_xferWJ`), and table 0's bound with escapes on accepted runs counted per round from it alone
(`zk_tableBoundθAccAtR_of_xferWJ`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)

local notation "𝔽" => FlockLevel3.GF128

/-- **C-Flock ZK's level 0 is M1's**: at `m ∈ [25, 35]`, level 0's `d` is M1's level-0 `logLen`. -/
theorem zkLevel0_d_eq : ∀ m : Fin 11, (Claims.Flock.zkLevel0 (m.val + 25)).d =
    ((Accounting.m1Schedule (m.val + 25)).levels.headD default).logLen := by
  decide +kernel

namespace Gen

variable [S : ZkSetup]

/-- **Table `j`'s word has level 0's `2^logLen` columns** at C-Flock ZK's padded tables of a masked statement in
scope. -/
theorem wordLenJ_zkTabs {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {j : ℕ} (hj : j < (tables true x).length) :
    wordLenJ true x j = 2 ^ (lvOf (zkTabs h0 L hz hk hlay j).sch 0).logLen := by
  have hlen : (tables true x).length = sts.size := by
    unfold tables
    rw [hset]
    simp
  rw [hlen] at hj
  have ht : (tables true x)[j]? = some (spec, sts[j]) := tables_getElem? hset (Array.getElem?_eq_getElem hj)
  obtain ⟨h25, h35, -⟩ := zk_scopeJ hs hin hset hj
  rw [zkTabs_of_lt h0 L hz hk hlay hj, (zkTab_m1 (L j hj) (hz j hj) (hk j hj) spec (hlay j hj)).1,
    Claims.Flock.lvOf_zero]
  have he := zkLevel0_d_eq ⟨sts[j].m - 25, by omega⟩
  simp only [Nat.sub_add_cancel h25] at he
  unfold wordLenJ
  rw [ht]
  simp only [↓reduceIte, he]

end Gen

/-- **Table `j`'s word has level 0's `2^logLen` columns** at C-Flock ZK's padded tables of a masked statement in
scope. -/
theorem wordLenJ_zkTabs {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {j : ℕ} (hj : j < (tables true x).length) :
    wordLenJ true x j = 2 ^ (lvOf (zkTabs h0 L hz hk hlay j).sch 0).logLen :=
  Gen.wordLenJ_zkTabs (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (j := j) (hj := hj)

open Classical in
/-- **The masked live sessions' bound at the deployed decoders, off the table on a witness lane**: `ZkXferJ` with the
live verdict `AccTxJ` (accepted at the draw at `wrS`, no column of table `j` off `C` read, table `j`'s `root_B` the
mixture's), the simulated strategies after `Commit` at the deployed decoders right at `OkOffWJ` there
(`decodesTableW_deploy`), and level 0's fork event at each table (`Claims.Flock.IZW`) bounding their `Adv₀`, each
fork bound with the slack of the fork at its round (`Claims.Flock.forkSlack`), as `ZkSessWJ` reads. -/
def ZkXferWJ (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (rd : (j : ℕ) → Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length → Fin ((V true).k x))
      (_ : ∀ j, Function.Injective (rd j)) (_ : ∀ j i, (rd j i).val = 0 → i = .inr ()) (δ : ℕ → ℝ)
      (_ : ∀ t, 0 ≤ δ t)
      (_ : Claims.Flock.SlackFitW (tables true x).length (fun j => (zkTabs h0 L hz hk hlay j).sch)
        (fun j => (zkTabs h0 L hz hk hlay j).N) rd δ)
      (ev : (j : ℕ) → Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length → Ω →
        (Fin ((V true).k x) → Coin) → (Fin ((V true).k x) → Coin) → Prop)
      (ev₁ : ℕ → Ω → (Fin ((V true).k x) → Coin) → Prop)
      (_tied : ∀ j i ω s s', ev j i ω s s' → Claims.TiedColl sha512
        ((Claims.run sha512 P (V true) x ω s).vLog ++
          (Claims.run sha512 P (V true) x ω (Claims.forkCoins (rd j i) s s')).vLog))
      (_tied₁ : ∀ j ω s, ev₁ j ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog)
      (_acc : ∀ j i ω s s', ev j i ω s s' → (Claims.run sha512 P (V true) x ω s).out = .acc ∧
        (Claims.run sha512 P (V true) x ω (Claims.forkCoins (rd j i) s s')).out = .acc)
      (_acc₁ : ∀ j ω s, ev₁ j ω s → (Claims.run sha512 P (V true) x ω s).out = .acc),
      ∀ ω (j : ℕ) (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽),
        (∀ p, wideJ true x j (C p)) →
        ∃ (dg : ByteArray → ByteArray) (N : ℕ) (rootB₀ : Array ByteArray) (τ : Type) (_ : Fintype τ) (_ : Nonempty τ)
          (σL : τ → Strategy
            (liveSession dg (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts N))
          (hσ : ∀ u, (σL u).1 = rootB₀),
          (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
              ¬ ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
                Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
                  u ≠ C p) ≤
            (∑ u, (prob (AccTxJ ins.read.I d j rootB₀ C) _ (σL u)).toReal) / Fintype.card τ ∧
          (∀ r ℓ, (∑ u, advRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot rootB₀[j]!)
              r.val (ℓ.val + 1)
              (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j rootB₀ C)
                (AccTxJ ins.read.I d j rootB₀ C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                (fun rs j => Exec.capOfRoot rs[j.val]!) N
                (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j rootB₀ C) rootB₀
                (σL u) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj) (tabsJ_get _ _ _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inl (r, ℓ))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inl (r, ℓ)))).val ω) ∧
          (∀ r : Fin 2, (∑ u, advPadsSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
              (Exec.capOfRoot rootB₀[j]!) r.val
              (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j rootB₀ C)
                (AccTxJ ins.read.I d j rootB₀ C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                (fun rs j => Exec.capOfRoot rs[j.val]!) N
                (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j rootB₀ C) rootB₀
                (σL u) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj) (tabsJ_get _ _ _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inl r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inl r)))).val ω) ∧
          (∀ r : Fin 2, (∑ u, advTauSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
              (Exec.capOfRoot rootB₀[j]!) r.val
              (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j rootB₀ C)
                (AccTxJ ins.read.I d j rootB₀ C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                (fun rs j => Exec.capOfRoot rs[j.val]!) N
                (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j rootB₀ C) rootB₀
                (σL u) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj) (tabsJ_get _ _ _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inr r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inr r)))).val ω) ∧
          ((∑ u, adv₀SZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot rootB₀[j]!)
              (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j rootB₀ C)
                (AccTxJ ins.read.I d j rootB₀ C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                (fun rs j => Exec.capOfRoot rs[j.val]!) N
                (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j rootB₀ C) rootB₀
                (σL u) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj) (tabsJ_get _ _ _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inr ()) ω s.1 s.2) + Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inr ())).val ω) ∧
          (∑ u, (prob (LiveSelfClashZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j)
              (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
              (AccTxJ ins.read.I d j rootB₀ C)
              (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
              (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) j)
            (liveSession dg (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts N)
              (σL u)).toReal) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s

namespace Gen

variable [S : ZkSetup]

open Classical in
/-- **The masked live sessions' bound at the deployed decoders, off the table on a witness lane**: `ZkXferJ` with the
live verdict `AccTxJ` (accepted at the draw at `wrS`, no column of table `j` off `C` read, table `j`'s `root_B` the
mixture's), the simulated strategies after `Commit` at the deployed decoders right at `OkOffWJ` there
(`decodesTableW_deploy`), and level 0's fork event at each table (`Claims.Flock.IZW`) bounding their `Adv₀`, each
fork bound with the slack of the fork at its round (`Claims.Flock.forkSlack`), as `ZkSessWJ` reads. -/
def ZkXferWJ (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (rd : (j : ℕ) → Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length → Fin ((V true).k x))
      (_ : ∀ j, Function.Injective (rd j)) (_ : ∀ j i, (rd j i).val = 0 → i = .inr ()) (δ : ℕ → ℝ)
      (_ : ∀ t, 0 ≤ δ t)
      (_ : Claims.Flock.SlackFitW (tables true x).length (fun j => (zkTabs h0 L hz hk hlay j).sch)
        (fun j => (zkTabs h0 L hz hk hlay j).N) rd δ)
      (ev : (j : ℕ) → Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length → Ω →
        (Fin ((V true).k x) → Coin) → (Fin ((V true).k x) → Coin) → Prop)
      (ev₁ : ℕ → Ω → (Fin ((V true).k x) → Coin) → Prop)
      (_tied : ∀ j i ω s s', ev j i ω s s' → Claims.TiedColl sha512
        ((Claims.run sha512 P (V true) x ω s).vLog ++
          (Claims.run sha512 P (V true) x ω (Claims.forkCoins (rd j i) s s')).vLog))
      (_tied₁ : ∀ j ω s, ev₁ j ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog)
      (_acc : ∀ j i ω s s', ev j i ω s s' → (Claims.run sha512 P (V true) x ω s).out = .acc ∧
        (Claims.run sha512 P (V true) x ω (Claims.forkCoins (rd j i) s s')).out = .acc)
      (_acc₁ : ∀ j ω s, ev₁ j ω s → (Claims.run sha512 P (V true) x ω s).out = .acc),
      ∀ ω (j : ℕ) (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽),
        (∀ p, wideJ true x j (C p)) →
        ∃ (dg : ByteArray → ByteArray) (N : ℕ) (rootB₀ : Array ByteArray) (τ : Type) (_ : Fintype τ) (_ : Nonempty τ)
          (σL : τ → Strategy
            (liveSession dg (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts N))
          (hσ : ∀ u, (σL u).1 = rootB₀),
          (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
              ¬ ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
                Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
                  u ≠ C p) ≤
            (∑ u, (prob (AccTxJ ins.read.I d j rootB₀ C) _ (σL u)).toReal) / Fintype.card τ ∧
          (∀ r ℓ, (∑ u, advRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot rootB₀[j]!)
              r.val (ℓ.val + 1)
              (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j rootB₀ C)
                (AccTxJ ins.read.I d j rootB₀ C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                (fun rs j => Exec.capOfRoot rs[j.val]!) N
                (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j rootB₀ C) rootB₀
                (σL u) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj) (tabsJ_get _ _ _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inl (r, ℓ))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inl (r, ℓ)))).val ω) ∧
          (∀ r : Fin 2, (∑ u, advPadsSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
              (Exec.capOfRoot rootB₀[j]!) r.val
              (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j rootB₀ C)
                (AccTxJ ins.read.I d j rootB₀ C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                (fun rs j => Exec.capOfRoot rs[j.val]!) N
                (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j rootB₀ C) rootB₀
                (σL u) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj) (tabsJ_get _ _ _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inl r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inl r)))).val ω) ∧
          (∀ r : Fin 2, (∑ u, advTauSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
              (Exec.capOfRoot rootB₀[j]!) r.val
              (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j rootB₀ C)
                (AccTxJ ins.read.I d j rootB₀ C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                (fun rs j => Exec.capOfRoot rs[j.val]!) N
                (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j rootB₀ C) rootB₀
                (σL u) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj) (tabsJ_get _ _ _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inr r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inr r)))).val ω) ∧
          ((∑ u, adv₀SZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot rootB₀[j]!)
              (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j rootB₀ C)
                (AccTxJ ins.read.I d j rootB₀ C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                (fun rs j => Exec.capOfRoot rs[j.val]!) N
                (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j rootB₀ C) rootB₀
                (σL u) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj) (tabsJ_get _ _ _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inr ()) ω s.1 s.2) + Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inr ())).val ω) ∧
          (∑ u, (prob (LiveSelfClashZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j)
              (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
              (AccTxJ ins.read.I d j rootB₀ C)
              (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
              (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) j)
            (liveSession dg (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts N)
              (σL u)).toReal) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s

/-- **`ZkXferWJ` gives `ZkLiveAccWJ`** where some run accepts: the statement parses in scope (`acc_inScope`) and
carries a draw (`acc_draw_of_setup`), the deployed decoders are right at `OkOffWJ` on `AccTxJ`
(`decodesTableW_deploy`), and table `j`'s word has level 0's columns (`wordLenJ_zkTabs`), so that verdict at table `j`
accepts with no opening off `C` on a witness lane unless it self-clashes. -/
theorem zkLiveAccWJ_of_xferWJ {x : Claims.Msg} (hno : ¬ Claims.Flock.NoAcc sha512 (V true) x) (hx : ZkXferWJ x) :
    ZkLiveAccWJ x := by
  intro ins spec sts hset h0 L hz hk hlay Ω _ _ P
  obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
  obtain ⟨d, hd⟩ := acc_draw_of_setup hno hset
  obtain ⟨rd, hrd, hr0, δ, hδ, hfit, ev, ev₁, tied, tied₁, hacc, hacc₁, hω⟩ :=
    hx s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P
  refine ⟨rd, hrd, hr0, δ, hδ, hfit, ev, ev₁, tied, tied₁, hacc, hacc₁, fun ω j hj C hC => ?_⟩
  obtain ⟨dg, N, rootB₀, τ, _, _, σL, hσ, htr, hadv, hP, hTau, h₀, hsc⟩ := hω ω j hj C hC
  have hN := wordLenJ_zkTabs hs hin hset h0 L hz hk hlay hj
  have hg := tabsJ_get (tables true x).length (zkTabs h0 L hz hk hlay) (finJ (zkTabs h0 L hz hk hlay) hj)
  refine ⟨OkOffWJ (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)) j rootB₀ C,
    AccTxJ ins.read.I d j rootB₀ C, opnD sts _, dg, decD _, fun _ _ => true, fun _ _ _ => [],
    fun rs j => Exec.capOfRoot rs[j.val]!, N, rootB₀, fun o h => ⟨h.1, ?_⟩, fun _ _ h => h.1,
    decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j rootB₀ C, τ, inferInstance, inferInstance, σL, hσ, htr,
    hadv, hP, hTau, h₀, hsc⟩
  have h2 := h.2 rfl
  rw [hg] at h2
  exact h2 hN

end Gen

/-- **`ZkXferWJ` gives `ZkLiveAccWJ`** where some run accepts: the statement parses in scope (`acc_inScope`) and
carries a draw (`acc_draw_of_setup`), the deployed decoders are right at `OkOffWJ` on `AccTxJ`
(`decodesTableW_deploy`), and table `j`'s word has level 0's columns (`wordLenJ_zkTabs`), so that verdict at table `j`
accepts with no opening off `C` on a witness lane unless it self-clashes. -/
theorem zkLiveAccWJ_of_xferWJ {x : Claims.Msg} (hno : ¬ Claims.Flock.NoAcc sha512 (V true) x) (hx : ZkXferWJ x) :
    ZkLiveAccWJ x :=
  Gen.zkLiveAccWJ_of_xferWJ (S := S₀) (x := x) (hno := hno) (hx := hx)

/-- **The deployed masked verifier: KS**, at SHA-512 at the deployed relation, at every number of tables, from the
live sessions' bound at the deployed decoders off the table on a witness lane at the statements some run accepts
(`ZkXferWJ`) and `ExtendJ` at full-width columns (`zk_KS_of_liveAccWJ`, `zkLiveAccWJ_of_xferWJ`). -/
theorem zk_KS_of_xferWJ (hx : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkXferWJ x)
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  zk_KS_of_liveAccWJ (fun x hno => zkLiveAccWJ_of_xferWJ hno (hx x hno)) hext

namespace Gen

variable [S : ZkSetup]

/-- **The deployed masked verifier: KS**, at SHA-512 at the deployed relation, at every number of tables, from the
live sessions' bound at the deployed decoders off the table on a witness lane at the statements some run accepts
(`ZkXferWJ`) and `ExtendJ` at full-width columns (`zk_KS_of_liveAccWJ`, `zkLiveAccWJ_of_xferWJ`). -/
theorem zk_KS_of_xferWJ (hx : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkXferWJ x)
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  zk_KS_of_liveAccWJ (fun x hno => zkLiveAccWJ_of_xferWJ hno (hx x hno)) hext

end Gen

/-- **Table 0's masked bound at every statement, off the table on a witness lane, with escapes on accepted runs
counted per round**, at full-width columns, from `ZkXferWJ` at the statements some run accepts
(`zk_tableBoundθAccAtR`, `zkLiveAccWJ_of_xferWJ`). -/
theorem zk_tableBoundθAccAtR_of_xferWJ (hx : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkXferWJ x)
    (x : Claims.Msg) :
    Claims.Flock.TableBoundθAccAtR sha512 (deployZk Readable.relation) (wordLen true) (wide true) x :=
  zk_tableBoundθAccAtR (fun x hno => zkLiveAccWJ_of_xferWJ hno (hx x hno)) x

end FlockDeploy.KS
