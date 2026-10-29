import Proofs.Flock.Recursive.Property
import Proofs.Flock.Recursive.Private

/-!
# The recursive audit at C-Flock's compiled inner protocol

`RecursiveSound` and `RecursiveRanRegistered` take the inner protocol's soundness as a hypothesis (`InnerSound I εin`)
over an abstract `I`. At C-Flock's inner protocol that hypothesis can't hold as stated, since SHA-512 is inside the
inner protocol; what holds is its fork form, proved (`Recursion.flock_inner_sound_compiled`). These are the two
guarantees at the inner protocol the verifier of record runs, with that hypothesis gone:

* `Flock.Guarantees.RecursiveSoundExec` (retired from the lock: see `RecursiveKS`): `RecursiveSound` at `innerAt T CT st` (`flockInnerCL` at the executable's
  arithmetic and SHA-512 with plain leaves, on the class, regions and schedule of `execClassAt st`), with no
  `InnerSound` and no A3. Its event adds that Lean's verifier accepts `st` as the inner statement (`AcceptsStmt`), its
  forks add an inner fork of the prover's own runs whose decoded transcripts play the compiled table to outcomes that
  clash (`InnerForkRec` at `clashPickCL`), and its inner term is the interactive table's error (`tableError`).
* `Flock.Guarantees.RecursiveRanRegisteredExec`: `RecursiveRanRegistered` at the same inner protocol, the same way,
  with the registration the executable's, rows of bytes under hm96 leaves at a registered port (`rowCommit`), whose
  binding `Flock.SecurityProofs.regRowCommit_binding` proves, so it takes no `_hbind`.

Each still takes V*'s bridge in full (`Flock.Assumptions.VBridge`), which a protocol whose last round isn't committed
(candidate 3 today) doesn't give; `RecursiveAuditC3` is the audit at candidate 3, at the committed rounds. Their
proofs, `Flock.SecurityProofs.RecursiveSoundExec` and `Flock.SecurityProofs.RecursiveRanRegisteredExec`, are gap 2
(`Recursion.zk_sessions_recursive_fork_inner`) at the compile premise of an accepted statement's class
(`Recursion.compiledForkSound_execClassAt`, `Recursion.classOk_of_accepts`).
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

/-- **The recursive audit is sound at C-Flock's compiled inner protocol.** `RecursiveSound` at `I := innerAt T CT st`,
the inner protocol the verifier of record runs on the inner statement `st` (C-Flock's compiled table at the executable's
arithmetic, SHA-512 with plain leaves, the class, regions and schedule of `execClassAt st`, under the schedule `(T, CT)`
with live coins), with the inner protocol's soundness proved rather than taken, and no A3. For every prover `σ`, given
V*'s bridge (`Flock.Assumptions.VBridge`), one of five holds:

* `RecursiveSound`'s registrant fork (`regFinder`), round fork (`RoundFork`) or session fork (`SessionFork`);
* an inner fork (`InnerForkRec` at `clashPickCL`): at some draw and round, two continuations of `σ` after its round
  commitment whose inner transcripts, decoded from V*'s layers, play the compiled table to outcomes that clash, a
  collision of SHA-512 at an outcome the fork reaches;
* $$\Pr[\text{Lean's verifier accepts } st \wedge \text{every session of V* accepts} \wedge \neg\,\text{the hidden
  statement holds}] \le \mathbb E\Big[\sum_j \text{outer bound}_j\Big] + r\cdot\mathbb E\Big[\sum_j \text{stage
  slack}_j\Big] + \varepsilon_{table},$$
  `RecursiveSound`'s bound with the interactive table's error (`tableError`, at most `2^-205` for `22 ≤ m ≤ 33` and a
  shape in range: `Recursion.recursive_sound_compiled_fast100`) as its inner term.

It needs `VBridge` in full, which a protocol whose last round isn't committed (candidate 3 today) doesn't give. Its
event asks, as `RecursiveSound`'s does, that the inner statement have no witness, which the vacuity sweep found nearly
empty (see `RecursiveSound`); the knowledge form is `Claims.Recursion.KS`.

**Retired from the lock** with `RecursiveSound`: the lock lists no theorem of it; the recursion's guarantee
is `Flock.Guarantees.RecursiveKS`. -/
def RecursiveSoundExec : Prop :=
  ∀ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
    {n : ℕ} {L : Law n} {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    -- the inner statement
    (st : Flock.Stmt)
    -- V*'s sessions at each history and registration `R₂`
    (zs : Reg → L.Ω → List (Cm × (innerAt T CT st).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg (innerAt T CT st) Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, the rounds' commit strings, the firewall's
    -- commitments and the messages they open to, and V*'s bridge
    (x : Reg → L.Ω → (innerAt T CT st).Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List (innerAt T CT st).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (innerAt T CT st).Msg)
    (_hbr : Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg),
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ x wmsg Refine.H512
        (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (execClassAt st).cls
          (execClassAt st).regs (execClassAt st).sch) ∨
      prob (fun o => AcceptsStmt st ∧ AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ¬ (innerAt T CT st).holds (x o.1 o.2.1))
          (recGame L Reg (innerAt T CT st) Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((innerAt T CT st).r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (Accounting.tableError (execClassAt st).sch
            ⟨(execClassAt st).cls.kLog, (execClassAt st).regs.length, (execClassAt st).cls.mPts⟩)

/-- **The registration's commitment at row position `i`**: a row's bytes registered as the verifier checks a registered
read. An opening `(reg, v, salt, path)` of the port `reg` to the row `v` is `Flock.Registered.opens` of `v`'s hm96
commit string under `salt` at `i` through `path` (`Flock.SecurityProofs.regRowCommit_opens`), and two openings of one
port at `i` to different rows give a SHA-512 collision (`Flock.SecurityProofs.regRowCommit_binding`). -/
noncomputable def rowCommit (i : ℕ) :
    PrivateCircuit.Commit Flock.Registered.Port ByteArray (Flock.Registered.Port × ByteArray × ByteArray × ByteArray) :=
  Flock.SecurityProofs.regRowCommit (fun b : ByteArray => b.data.toList)
    (fun _ _ h => ByteArray.ext (Array.toList_inj.1 h)) i

/-- **The recursive audit ran the registered private circuit, at C-Flock's compiled inner protocol.**
`RecursiveRanRegistered` at `I := innerAt T CT st`, as `RecursiveSoundExec` is `RecursiveSound` there, with the
registration the executable's: on its hypotheses, with no `InnerSound` and no A3, the private circuit's rows `p`
registered at the port `rt` as the verifier checks a registered read (`rowCommit`, whose binding is proved, so no
`_hbind`), the developer's openings `op₀`, and what an accepted run carries of the rows it read (`_hop`, `_hx`, as in
`RecursiveRanRegistered`), one of six holds: `RecursiveSoundExec`'s four forks, a registration break (some outcome `σ`
reaches has every session accepting, having read at some position a row other than the registered one, and its opening
and the developer's give a SHA-512 collision, `rowCommit`'s extractor's pair), or Lean's verifier accepts `st` and
every session of V* while the registered private circuit's hidden statement fails with probability at most
`RecursiveSoundExec`'s bound. It needs `VBridge` in full, which candidate 3 today doesn't give; its event is nearly
empty as `RecursiveSound`'s is. -/
def RecursiveRanRegisteredExec : Prop :=
  ∀ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
    {n : ℕ} {L : Law n} {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (st : Flock.Stmt)
    (zs : Reg → L.Ω → List (Cm × (innerAt T CT st).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg (innerAt T CT st) Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → (innerAt T CT st).Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List (innerAt T CT st).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (innerAt T CT st).Msg)
    (_hbr : Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg)
    -- the registration's port, the registered private circuit's rows, and the developer's openings of them
    (rt : Flock.Registered.Port) (p : ℕ → ByteArray)
    (op₀ : ℕ → Flock.Registered.Port × ByteArray × ByteArray × ByteArray)
    (_hreg : ∀ i, (rowCommit i).Opens rt (p i) (op₀ i))
    -- a private circuit's statement at a draw; the rows an accepted run read, and the rows and openings of `rt` it
    -- carries
    (stmt : (ℕ → ByteArray) → L.Ω → (innerAt T CT st).Stmt)
    (reads : Reg × L.Ω × List (Cm × (innerAt T CT st).Coin) × (Reg₂ × VRec Vs Reg' dg N) → ℕ → Prop)
    (row : Reg × L.Ω × List (Cm × (innerAt T CT st).Coin) × (Reg₂ × VRec Vs Reg' dg N) → ℕ → ByteArray)
    (op : Reg × L.Ω × List (Cm × (innerAt T CT st).Coin) × (Reg₂ × VRec Vs Reg' dg N) → ℕ →
      Flock.Registered.Port × ByteArray × ByteArray × ByteArray)
    (_hop : ∀ o, AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 → ∀ i, reads o i → (rowCommit i).Opens rt (row o i) (op o i))
    (_hx : ∀ o, AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 → ∀ q : ℕ → ByteArray, (∀ i, reads o i → q i = row o i) →
      x o.1 o.2.1 = stmt q o.2.1),
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ x wmsg Refine.H512
        (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (execClassAt st).cls
          (execClassAt st).regs (execClassAt st).sch) ∨
      (∃ o, FlockSoundness.Game.Outcome (recGame L Reg (innerAt T CT st) Cm fun R ω t => outerV (zs R ω t) dg N) σ o ∧
        AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ∃ i, reads o i ∧ row o i ≠ p i ∧ FlockSoundness.Binding.Collides ((rowCommit i).ext (op o i) (op₀ i))) ∨
      prob (fun o => AcceptsStmt st ∧ AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ¬ (innerAt T CT st).holds (stmt p o.2.1))
          (recGame L Reg (innerAt T CT st) Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((innerAt T CT st).r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (Accounting.tableError (execClassAt st).sch
            ⟨(execClassAt st).cls.kLog, (execClassAt st).regs.length, (execClassAt st).cls.mPts⟩)

end Flock.Guarantees

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

/-- **The recursive audit is sound at C-Flock's compiled inner protocol.** -/
theorem RecursiveSoundExec : Flock.Guarantees.RecursiveSoundExec := by
  intro nT r T CT _ _ n L Reg Cm S Vs Reg' Reg₂ st zs dg N k Rw M hk hRw hM ρ hρ hr σ x wmsg Sk cmt msg hbr
  by_cases hacc : AcceptsStmt st
  swap
  · refine Or.inr (Or.inr (Or.inr (Or.inr (le_trans ?_ (zero_le ..)))))
    exact (prob_mono_outcome _ σ fun o _ h => absurd h.1 hacc).trans (prob_false _ σ).le
  rcases zk_sessions_recursive_fork_inner zs dg N hk hRw hM hρ hr σ x wmsg Sk cmt msg hbr Refine.H512 _
      (compiledForkSound_execClassAt T CT (classOk_of_accepts hacc)) with h | h | h | h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr (Or.inl h))
  · exact Or.inr (Or.inr (Or.inr (Or.inl h)))
  · refine Or.inr (Or.inr (Or.inr (Or.inr (le_trans ?_ h))))
    exact prob_mono_outcome _ σ fun o _ ho => ho.2

/-- **The executable's row registration binds**: `regRowCommit_binding` at the rows' bytes. -/
theorem rowCommit_binding (i : ℕ) : (Flock.Guarantees.rowCommit i).Binding :=
  regRowCommit_binding _ _ i

/-- **The recursive audit ran the registered private circuit, at C-Flock's compiled inner protocol.** -/
theorem RecursiveRanRegisteredExec : Flock.Guarantees.RecursiveRanRegisteredExec := by
  intro nT r T CT _ _ n L Reg Cm S Vs Reg' Reg₂ st zs dg N k Rw M hk hRw hM ρ hρ hr σ x wmsg Sk cmt msg hbr
    rt p op₀ hreg stmt reads row op hop hx
  rcases RecursiveSoundExec T CT st zs dg N hk hRw hM hρ hr σ x wmsg Sk cmt msg hbr with hf | hf | hf | hf | hb
  · exact Or.inl hf
  · exact Or.inr (Or.inl hf)
  · exact Or.inr (Or.inr (Or.inl hf))
  · exact Or.inr (Or.inr (Or.inr (Or.inl hf)))
  · by_cases hc : ∃ o, Game.Outcome
        (recGame L Reg (Flock.Guarantees.innerAt T CT st) Cm fun R ω t => outerV (zs R ω t) dg N) σ o ∧
        AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ∃ i, reads o i ∧ row o i ≠ p i ∧
          FlockSoundness.Binding.Collides ((Flock.Guarantees.rowCommit i).ext (op o i) (op₀ i))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl hc))))
    · have hmono : prob (fun o => AcceptsStmt st ∧ AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
              ¬ (Flock.Guarantees.innerAt T CT st).holds (stmt p o.2.1))
            (recGame L Reg (Flock.Guarantees.innerAt T CT st) Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
          prob (fun o => AcceptsStmt st ∧ AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
              ¬ (Flock.Guarantees.innerAt T CT st).holds (x o.1 o.2.1))
            (recGame L Reg (Flock.Guarantees.innerAt T CT st) Cm fun R ω t => outerV (zs R ω t) dg N) σ := by
        refine prob_mono_outcome _ σ fun o hro h => ⟨h.1, h.2.1, fun hh => ?_⟩
        have hagree : ∀ i, reads o i → p i = row o i := fun i hi => by
          by_contra hne
          exact hc ⟨o, hro, h.2.1, i, hi, fun he => hne he.symm,
            rowCommit_binding i rt (row o i) (p i) (op o i) (op₀ i) (hop o h.2.1 i hi) (hreg i) fun he => hne he.symm⟩
        rw [hx o h.2.1 p hagree] at hh
        exact h.2.2 hh
      exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (hmono.trans hb)))))

end Flock.SecurityProofs
