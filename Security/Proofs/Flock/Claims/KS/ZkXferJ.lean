import Proofs.Flock.Claims.KS.ZkDecJ

/-!
# The masked live sessions' bound at the deployed decoders

`ZkLiveAccJ` asks, at each prover, tape and table `C`, for decoders right at C-Flock ZK's padded tables (`hdec`) at a
verdict that accepts table `j` with no level-0 opening off `C` unless it self-clashes (`_hok`). The deployed decoders
(`opnD`, `decD`) are right at the verdict `accepted` (`ZkDecJ`), and so at that verdict on the live outcomes whose
decoded table `j` has no such opening. So `ZkLiveAccJ` follows from the live sessions' bound at the deployed decoders,
which chooses no decoder (`ZkXferJ`):

* `decodesTable_at`: a decoder right at `ok` on `pL` is right at `ok` and `Q` at table `j₀` on the live outcomes of
  `pL` whose table `j₀` reps play, on the decoded tapes, only to outputs of `Q`.
* `OffNoClashZ`, `AccOffJ`, `OkOffJ`, `decodesOff_deploy`: an opening off `C` without a self-clash, the live outcomes
  `verify --zk` accepts at the setup at the statement's draw at `wrS` whose decoded table `j` has none, the verdict
  `accepted` with none at table `j`, and the deployed decoders right there.
* `ZkXferJ`, `zkLiveAccJ_of_xferJ`, `zk_KS_of_xferJ`: the live sessions' bound at the deployed decoders gives
  `ZkLiveAccJ` where some run accepts, so `Flock.KS` at the deployed masked record follows from it and `ExtendJ`.

At `oracleOf C` the verdict's level-0 term reads both reps' extra lanes, which a table `C` of witness-lane columns reads
as `0`, so a run whose level-0 opening has a nonzero extra lane is off `C` there (`ZkPads`): the deployed prover's own
runs, whose extra lanes are uniform, are not the mixture `ZkXferJ` asks for.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable repTapeJ)
open FlockSoundness.Discharge.Composed (outJ repJ)

local notation "𝔽" => FlockLevel3.GF128

section At

variable {P OutT Cp Op : Type} {J mPts : ℕ} {out : Fin J → Cp → P × P → Op → OutT}
  {ok : Fin J → OutT → Prop} {pL : LiveOutJ J mPts → Prop} {opn : Array ByteArray → Fin J → Op}
  {dec : Fin J → ℕ → ByteArray → List Msg} {one : Fin J → ℕ → Bool}
  {fin : Fin J → Fin 2 → Array ByteArray → List Msg} {capOf : Array ByteArray → Fin J → Cp}
  {G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P}

/-- **A decoder right at `ok` is right at `ok` and `Q` at table `j₀`**, on the live outcomes of `pL` whose table `j₀`
reps play, on the decoded tapes, only to outputs of `Q`. -/
theorem decodesTable_at (j₀ : Fin J) (Q : OutT → Prop) {i : Fin J}
    (h : DecodesTable out ok pL opn dec one fin capOf G i) :
    DecodesTable out (fun j o => ok j o ∧ (j = j₀ → Q o))
      (fun o => pL o ∧ ∀ pend : P × P, PlaysT (G j₀ o.pts 0) (repTapeJ dec one fin o j₀ 0) pend.1 →
        PlaysT (G j₀ o.pts 1) (repTapeJ dec one fin o j₀ 1) pend.2 →
        Q (out j₀ (capOf o.rootB j₀) pend (opn o.proofs j₀)))
      opn dec one fin capOf G i := by
  rintro o ⟨hp, hq⟩
  obtain ⟨pend, h₀, h₁, hok⟩ := h o hp
  exact ⟨pend, h₀, h₁, hok, fun e => by subst e; exact hq pend h₀ h₁⟩

end At

/-- **A masked output opens a level-0 position off `C` without a self-clash**, at the cap `cap₀`. -/
def OffNoClashZ {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap₀ : Model.Cap D)
    (C : Model.Oracle 𝔽) (o : OutZC 𝔽 D Sl) : Prop :=
  Off₀Z Hs T cap₀ C o ∧ ¬ SelfClashZ Hs T o

/-- **The deployed live verdict, with no opening off `C` at table `j` unless it self-clashes**: `verify --zk` accepts
at the setup at the draw `d` at `wrS` (`AcceptsAt`, which an accepting run of `Deploy.V` gives with no premise on its
record's draw: `acceptsAt_wrS`), and every output table `j`'s reps play to on the tapes `decD` decodes, at the
openings `opnD` and the cap of the outcome's `root_B[j]`, has no level-0 opening off `C` at the cap of `rootB₀[j]`
without a self-clash (`OffNoClashZ`). -/
def AccOffJ (I : Exec.Inputs) (sts : Array Stmt) {mPts : ℕ} (n : ℕ) (T : ℕ → TabZK 𝔽 mPts) (d : Lean.Json)
    {j : ℕ} (hj : j < n) (rootB₀ : Array ByteArray) (C : Model.Oracle 𝔽) (o : LiveOutJ (tabsJ n T).length mPts) :
    Prop :=
  Composed.AcceptsAt I d (Claims.Bridge.Custody.wrS I d o) ∧
    ∀ pend : PendZ 𝔽 Refine.Dig (List UInt8) × PendZ 𝔽 Refine.Dig (List UInt8),
      PlaysT (repJ execArith ZkExec.hs512 (tabsJ n T) (finJ T hj) o.pts 0)
        (repTapeJ (decD (tabsJ n T)) (fun _ _ => true) (fun _ _ _ => []) o (finJ T hj) 0) pend.1 →
      PlaysT (repJ execArith ZkExec.hs512 (tabsJ n T) (finJ T hj) o.pts 1)
        (repTapeJ (decD (tabsJ n T)) (fun _ _ => true) (fun _ _ _ => []) o (finJ T hj) 1) pend.2 →
      ¬ OffNoClashZ ZkExec.hs512 (T j) (Exec.capOfRoot rootB₀[j]!) C
        (outJ ZkExec.hs512 (tabsJ n T) (finJ T hj) (Exec.capOfRoot o.rootB[j]!) pend
          (opnD sts (tabsJ n T) o.proofs (finJ T hj)))

/-- **The verdict `accepted`, with no level-0 opening off `C` at table `j` unless it self-clashes**, at the cap of
`rootB₀[j]`. -/
def OkOffJ {mPts n : ℕ} (T : ℕ → TabZK 𝔽 mPts) {j : ℕ} (hj : j < n) (rootB₀ : Array ByteArray)
    (C : Model.Oracle 𝔽) (j' : Fin (tabsJ n T).length) (o : OutZC 𝔽 Refine.Dig (List UInt8)) : Prop :=
  o.accepted = true ∧ (j' = finJ T hj → ¬ OffNoClashZ ZkExec.hs512 (T j) (Exec.capOfRoot rootB₀[j]!) C o)

/-- **The deployed decoders are right at `OkOffJ` on `AccOffJ`** at a masked statement in scope that `Deploy.V` sets
up at the draw `d` (`decodesTable_deploy`, `decodesTable_at`). -/
theorem decodesOff_deploy {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {j : ℕ} (hj : j < (tables true x).length)
    (rootB₀ : Array ByteArray) (C : Model.Oracle 𝔽)
    (i : Fin (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (OkOffJ (zkTabs h0 L hz hk hlay) hj rootB₀ C)
      (AccOffJ ins.read.I sts (tables true x).length (zkTabs h0 L hz hk hlay) d hj rootB₀ C)
      (opnD sts (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (decD (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) i := by
  have h := decodesTable_at (finJ (zkTabs h0 L hz hk hlay) hj)
    (fun o => ¬ OffNoClashZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j) (Exec.capOfRoot rootB₀[j]!) C o)
    (decodesTable_deploy hs hin hset h0 L hz hk hlay hd i)
  exact h

open Classical in
/-- **The masked live sessions' bound at the deployed decoders**: `ZkLiveAccJ` with its decoders and verdicts fixed,
at every masked statement in scope that `Deploy.V` sets up at a draw `d` and at least one table. At C-Flock ZK's padded
tables `zkTabs`, every prover has escapes at injective fork rounds on accepted deployed runs, tied to collisions, such
that at every tape and table `C` of full-width columns a mixture of live strategies with common roots `rootB₀` bounds
the deployed runs accepting with no column off `C` by its acceptance with table `j` decoded with no opening off `C`
unless it self-clashes (`AccOffJ`), and the escapes' events bound the `Adv`s of its simulated strategies after
`Commit` (`simAtJZ` at the deployed decoders `opnD`, `decD`, no tail and table `j`'s cap from `root_B[j]`, right at
`OkOffJ` by `decodesOff_deploy`) and its decoded self-clashes of table `j` on accepted live runs. -/
def ZkXferJ (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (rd : (j : ℕ) → Claims.Flock.IZ (zkTabs h0 L hz hk hlay j).sch.levels.length → Fin ((V true).k x))
      (_ : ∀ j, Function.Injective (rd j))
      (ev : (j : ℕ) → Claims.Flock.IZ (zkTabs h0 L hz hk hlay j).sch.levels.length → Ω →
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
            (∑ u, (prob (AccOffJ ins.read.I sts (tables true x).length (zkTabs h0 L hz hk hlay) d hj rootB₀
              (Claims.Flock.oracleOf C)) _ (σL u)).toReal) / Fintype.card τ ∧
          (∀ r ℓ, (∑ u, advRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot rootB₀[j]!)
              r.val (ℓ.val + 1)
              (simAtJZ execArith ZkExec.hs512 _ (OkOffJ (zkTabs h0 L hz hk hlay) hj rootB₀ (Claims.Flock.oracleOf C))
                (AccOffJ ins.read.I sts (tables true x).length (zkTabs h0 L hz hk hlay) d hj rootB₀
                  (Claims.Flock.oracleOf C))
                (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!) N
                (decodesOff_deploy hs hin hset h0 L hz hk hlay hd hj rootB₀ (Claims.Flock.oracleOf C)) rootB₀
                (σL u) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj) (tabsJ_get _ _ _))) / Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (r, ℓ)) ω s.1 s.2) ∧
          (∀ r : Fin 2, (∑ u, advPadsSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
              (Exec.capOfRoot rootB₀[j]!) r.val
              (simAtJZ execArith ZkExec.hs512 _ (OkOffJ (zkTabs h0 L hz hk hlay) hj rootB₀ (Claims.Flock.oracleOf C))
                (AccOffJ ins.read.I sts (tables true x).length (zkTabs h0 L hz hk hlay) d hj rootB₀
                  (Claims.Flock.oracleOf C))
                (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!) N
                (decodesOff_deploy hs hin hset h0 L hz hk hlay hd hj rootB₀ (Claims.Flock.oracleOf C)) rootB₀
                (σL u) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj) (tabsJ_get _ _ _))) / Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inr (.inl r)) ω s.1 s.2) ∧
          (∀ r : Fin 2, (∑ u, advTauSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
              (Exec.capOfRoot rootB₀[j]!) r.val
              (simAtJZ execArith ZkExec.hs512 _ (OkOffJ (zkTabs h0 L hz hk hlay) hj rootB₀ (Claims.Flock.oracleOf C))
                (AccOffJ ins.read.I sts (tables true x).length (zkTabs h0 L hz hk hlay) d hj rootB₀
                  (Claims.Flock.oracleOf C))
                (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!) N
                (decodesOff_deploy hs hin hset h0 L hz hk hlay hd hj rootB₀ (Claims.Flock.oracleOf C)) rootB₀
                (σL u) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj) (tabsJ_get _ _ _))) / Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inr (.inr r)) ω s.1 s.2) ∧
          (∑ u, (prob (LiveSelfClashZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j)
              (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
              (AccOffJ ins.read.I sts (tables true x).length (zkTabs h0 L hz hk hlay) d hj rootB₀
                (Claims.Flock.oracleOf C))
              (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
              (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) j)
            (liveSession dg (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts N)
              (σL u)).toReal) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s

/-- **`ZkXferJ` gives `ZkLiveAccJ`** where some run accepts: the statement parses in scope (`acc_inScope`) and
carries a draw (`acc_draw_of_setup`), and the deployed decoders are right at `OkOffJ` on `AccOffJ`
(`decodesOff_deploy`), whose verdict at table `j` accepts with no opening off `C` unless it self-clashes. -/
theorem zkLiveAccJ_of_xferJ {x : Claims.Msg} (hno : ¬ Claims.Flock.NoAcc sha512 (V true) x) (hx : ZkXferJ x) :
    ZkLiveAccJ x := by
  intro ins spec sts hset h0 L hz hk hlay Ω _ _ P
  obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
  obtain ⟨d, hd⟩ := acc_draw_of_setup hno hset
  obtain ⟨rd, hrd, ev, ev₁, tied, tied₁, hacc, hacc₁, hω⟩ := hx s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P
  refine ⟨rd, hrd, ev, ev₁, tied, tied₁, hacc, hacc₁, fun ω j hj C hC => ?_⟩
  obtain ⟨dg, N, rootB₀, τ, _, _, σL, hσ, htr, hadv, hP, hTau, hsc⟩ := hω ω j hj C hC
  exact ⟨_, _, _, dg, _, _, _, _, N, rootB₀, fun _ h => ⟨h.1, h.2 rfl⟩,
    decodesOff_deploy hs hin hset h0 L hz hk hlay hd hj rootB₀ (Claims.Flock.oracleOf C), τ, inferInstance,
    inferInstance, σL, hσ, htr, hadv, hP, hTau, hsc⟩

/-- **The deployed masked verifier: KS**, at SHA-512 at the deployed relation, at every number of tables, from the
live sessions' bound at the deployed decoders at the statements some run accepts (`ZkXferJ`) and `ExtendJ` at
full-width columns (`zk_KS_of_liveAccJ`, `zkLiveAccJ_of_xferJ`). -/
theorem zk_KS_of_xferJ (hx : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkXferJ x)
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  zk_KS_of_liveAccJ (fun x hno => zkLiveAccJ_of_xferJ hno (hx x hno)) hext

end FlockDeploy.KS
