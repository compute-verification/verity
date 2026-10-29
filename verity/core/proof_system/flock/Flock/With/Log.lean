/-
The verifier with a logging oracle. `LogV` is the verdict monad with a write-only log of SHA-512 inputs, kept under the
exception so that a rejecting run keeps its log too; `Hashes.logged h` appends each SHA-512 input to the log and answers
with `h` of it. SHA-256 and BLAKE3 stay the deployed functions, unlogged, as in `Hashes.ofSha512`: the oracle is SHA-512.

Forgetting the log gives back `verifyWith h` (`verifyLogged_fst`, `Zk.verifyLogged_fst`), from any starting log. The log's
length is `shaCalls h`, the number of SHA-512 calls a run with the counting oracle `Hashes.counted h` makes
(`verifyLogged_size`, `Zk.verifyLogged_size`). Both are instances of the simulations of `Flock.With.Sim`.
-/
import Flock.With.SimVerify
import Flock.With.SimZk

namespace Flock

/-- The verdict monad over a state that a rejection keeps. -/
abbrev StV (σ : Type) := ExceptT String (StateM σ)

/-- The verdict monad with a write-only log: the SHA-512 inputs, in call order. -/
abbrev LogV := StV (Array ByteArray)

/-- The verdict monad with a count of SHA-512 calls. -/
abbrev CountV := StV Nat

/-- A run of `StV σ` from `s`: its verdict and its final state. -/
def StV.go {σ α : Type} (x : StV σ α) (s : σ) : Except String α × σ := Id.run (x.run.run s)

/-- `h` as the SHA-512 oracle, logging each input. -/
def Hashes.logged (h : ByteArray → ByteArray) : Hashes LogV :=
  { Hashes.ofSha512 h with sha512 := fun b => do modify (·.push b); pure (h b) }

/-- `h` as the SHA-512 oracle, counting each call. -/
def Hashes.counted (h : ByteArray → ByteArray) : Hashes CountV :=
  { Hashes.ofSha512 h with sha512 := fun b => do modify (· + 1); pure (h b) }

namespace With.Sim

/-- A run of `StV σ` and a verdict of `V`: the run's verdict from every state is the verdict. -/
def forget (σ : Type) : Sim (StV σ) V where
  R x y := ∀ s, (x.go s).1 = y
  pure _ _ := rfl
  bind {α β x y f g} hxy hfg s := by
    have hx := hxy s
    simp only [StV.go, Id.run, ExceptT.run_bind, StateT.run_bind] at hx ⊢
    rcases hr : x.run.run s with ⟨r, s'⟩
    rw [hr] at hx
    subst hx
    cases r with
    | error e => rfl
    | ok a => exact hfg a s'
  throw _ _ := rfl
  lift _ _ := rfl

/-- A logging run and a counting run: from a log of `n` entries and a count of `n`, the same verdict, and the final log
has the final count's entries. -/
def logCount : Sim LogV CountV where
  R x y := ∀ l, (x.go l).1 = (y.go l.size).1 ∧ (x.go l).2.size = (y.go l.size).2
  pure _ _ := ⟨rfl, rfl⟩
  bind {α β x y f g} hxy hfg l := by
    have hx := hxy l
    simp only [StV.go, Id.run, ExceptT.run_bind, StateT.run_bind] at hx ⊢
    rcases hr : x.run.run l with ⟨r, l'⟩
    rcases hr' : y.run.run l.size with ⟨r', n'⟩
    rw [hr, hr'] at hx
    obtain ⟨h1, h2⟩ := hx
    dsimp only at h1 h2
    subst h1 h2
    cases r with
    | error e => exact ⟨rfl, rfl⟩
    | ok a => exact hfg a l'
  throw _ _ := ⟨rfl, rfl⟩
  lift _ _ := ⟨rfl, rfl⟩

theorem forget_logged (h : ByteArray → ByteArray) : (forget (Array ByteArray)).Hashes (Hashes.logged h) (Hashes.ofSha512 h) where
  sha512 _ _ := rfl
  sha256 _ _ := rfl
  blake3 _ _ := rfl
  blake3Keyed _ _ _ := rfl

theorem forget_counted (h : ByteArray → ByteArray) : (forget Nat).Hashes (Hashes.counted h) (Hashes.ofSha512 h) where
  sha512 _ _ := rfl
  sha256 _ _ := rfl
  blake3 _ _ := rfl
  blake3Keyed _ _ _ := rfl

theorem logCount_logged_counted (h : ByteArray → ByteArray) : logCount.Hashes (Hashes.logged h) (Hashes.counted h) where
  sha512 _ _ := ⟨rfl, Array.size_push ..⟩
  sha256 _ _ := ⟨rfl, rfl⟩
  blake3 _ _ := ⟨rfl, rfl⟩
  blake3Keyed _ _ _ := ⟨rfl, rfl⟩

end With.Sim

/-- `verifyWithM` with the logging oracle from the log `log`: its verdict and the log with every SHA-512 input appended. -/
def verifyLogged (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (log : Array ByteArray := #[]) : Except String Unit × Array ByteArray :=
  (verifyWithM (Hashes.logged h) spec tabs record proofs).go log

/-- The number of SHA-512 calls of `verifyWithM` with `h` as the oracle. -/
def shaCalls (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : Nat :=
  ((verifyWithM (Hashes.counted h) spec tabs record proofs).go 0).2

theorem verifyLogged_fst (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (log : Array ByteArray) :
    (verifyLogged h spec tabs record proofs log).1 = verifyWith h spec tabs record proofs :=
  With.verifyWithM_sim (With.Sim.forget _) (With.Sim.forget_logged h) spec tabs record proofs log

theorem verifyCounted_fst (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (n : Nat) :
    ((verifyWithM (Hashes.counted h) spec tabs record proofs).go n).1 = verifyWith h spec tabs record proofs :=
  With.verifyWithM_sim (With.Sim.forget _) (With.Sim.forget_counted h) spec tabs record proofs n

theorem verifyLogged_size (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : (verifyLogged h spec tabs record proofs).2.size = shaCalls h spec tabs record proofs :=
  (With.verifyWithM_sim With.Sim.logCount (With.Sim.logCount_logged_counted h) spec tabs record proofs #[]).2

namespace Zk

/-- `Zk.verifyWithM` with the logging oracle from the log `log`: its verdict and the log with every SHA-512 input
appended. -/
def verifyLogged (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (log : Array ByteArray := #[]) : Except String Unit × Array ByteArray :=
  (verifyWithM (Hashes.logged h) spec tabs record proofs).go log

/-- The number of SHA-512 calls of `Zk.verifyWithM` with `h` as the oracle. -/
def shaCalls (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : Nat :=
  ((verifyWithM (Hashes.counted h) spec tabs record proofs).go 0).2

theorem verifyLogged_fst (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (log : Array ByteArray) :
    (verifyLogged h spec tabs record proofs log).1 = verifyWith h spec tabs record proofs :=
  With.zk_verifyWithM_sim (With.Sim.forget _) (With.Sim.forget_logged h) spec tabs record proofs log

theorem verifyCounted_fst (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (n : Nat) :
    ((verifyWithM (Hashes.counted h) spec tabs record proofs).go n).1 = verifyWith h spec tabs record proofs :=
  With.zk_verifyWithM_sim (With.Sim.forget _) (With.Sim.forget_counted h) spec tabs record proofs n

theorem verifyLogged_size (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : (verifyLogged h spec tabs record proofs).2.size = shaCalls h spec tabs record proofs :=
  (With.zk_verifyWithM_sim With.Sim.logCount (With.Sim.logCount_logged_counted h) spec tabs record proofs #[]).2

end Zk

end Flock
