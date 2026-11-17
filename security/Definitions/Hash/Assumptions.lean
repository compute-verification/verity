module

public import Mathlib
public import Definitions.Hash.Basic

/-! The hash area's named assumption `LmsUnforgeable`: EUF-CMA of LMS over SHA-512, asked of one forger at one key. -/

@[expose] public section

namespace Hash.Assumptions

/-- **An LMS forger** against one key of an auditor-trusted device: on uniform coins `S` it outputs a message and the
bytes of a signature, or nothing, having had the device sign the messages `signed` lists on that coin; `cost` is its
SHA-512 calls on each coin. -/
structure LmsForger where
  S : Type
  [fin : Fintype S]
  [ne : Nonempty S]
  key : Msg
  /-- the messages the device signed under `key` on each coin, whose signatures the forger saw: its signing queries,
  each of which may depend on the signatures it saw before it -/
  signed : S → List Msg
  out : S → Option (Msg × Msg)
  cost : S → ℕ

attribute [instance] LmsForger.fin LmsForger.ne

/-- **The forger's output on the coin `s` is a forgery** under `verify` (public key, message, signature): a message the
device didn't sign on that coin, with a signature that verifies under its key. -/
def LmsForger.Forges (verify : Msg → Msg → Msg → Bool) (F : LmsForger) (s : F.S) : Prop :=
  ∃ m sig, F.out s = some (m, sig) ∧ m ∉ F.signed s ∧ verify F.key m sig = true

open Classical in
/-- `Pr[F forges under verify]`, over its uniform coins. -/
noncomputable def LmsForger.Pr (verify : Msg → Msg → Msg → Bool) (F : LmsForger) : ℝ :=
  ((Finset.univ.filter (F.Forges verify)).card : ℝ) / Fintype.card F.S

/-- **`LmsUnforgeable` (`euf-cma/lms-sha512-n64`): LMS over SHA-512 is EUF-CMA**, at one forger: if `F` makes at most `t`
SHA-512 calls on every coin and has at most `Q` messages signed on every coin, it forges under `verify` with probability
at most `ε_forge(t, Q)`. -/
def LmsUnforgeable (verify : Msg → Msg → Msg → Bool) (εForge : ℕ → ℕ → ℝ) (t Q : ℕ) (F : LmsForger) : Prop :=
  (∀ s, F.cost s ≤ t) → (∀ s, (F.signed s).length ≤ Q) → F.Pr verify ≤ εForge t Q

end Hash.Assumptions
